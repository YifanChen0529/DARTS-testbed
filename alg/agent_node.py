"""
DARTS single-agent process entrypoint.
Runs ONE DartsMARLAgent on ONE Jetson Orin Nano. Used in deployment_mode='distributed_jetson'.
Launch on each Jetson:
    PYTHONPATH=. python alg/agent_node.py --local_agent_id agent_1 --workload B1_CDR --exp_name rtss_eval_run1

The Jetson identity (``local_agent_id``), bind port, and calibration JSON on disk are
unique per node. The MARL step/auction path matches ``train_darts.run_episode`` by
constructing four ``DartsMARLAgent`` instances (frozen-eval style), as required by
``TurtleBot3Env.step(agents=...)`` today; only ``local_agent_id`` labels this process
for coordinator RPCs and local logs.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import io
import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import List, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import grpc

from alg.config_darts import DARTS_CONFIG
from alg.darts_agent import DartsMARLAgent
from alg.train_darts import (
    EVAL_CSV_HEADER,
    config_to_dict,
    run_episode,
    single_episode_eval_row_list,
)
from envs.dag_task import DAGGraph
from envs.jetson_grpc_transport import JetsonGrpcTransport
from envs.proto import darts_coordinator_pb2 as cpb2
from envs.proto import darts_coordinator_pb2_grpc as cpb2_grpc
from envs.simulation_env import LocalDAGTracker
from envs.turtlebot3_spatial_env import TurtleBot3Env
from envs.workloads import get_workload

EXPECTED_AGENTS = ("agent_1", "agent_2", "agent_3", "agent_4")


def _parse_bind_port(endpoint: str) -> int:
    _host, port_s = endpoint.rsplit(":", 1)
    return int(port_s)


def _shutdown_hook(transport: Optional[JetsonGrpcTransport], signum: int, frame: object) -> None:
    if transport is not None:
        transport.stop_server()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("agent_node")

    p = argparse.ArgumentParser(description="DARTS single-Jetson agent_node (distributed_jetson).")
    p.add_argument("--local_agent_id", type=str, default=os.environ.get("DARTS_AGENT_ID", ""))
    p.add_argument("--workload", type=str, default=None, choices=["B1_CDR", "B2_LML", "B3_IQI", "B4_DEM"])
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--deployment_mode", type=str, default="distributed_jetson", choices=["simulator", "distributed_jetson"])
    args = p.parse_args()

    local_id = (args.local_agent_id or "").strip()
    if not local_id:
        raise SystemExit("local_agent_id is required (--local_agent_id or env DARTS_AGENT_ID).")

    cfg = DARTS_CONFIG()
    if args.workload is not None:
        cfg = dataclasses.replace(cfg, workload_id=args.workload)
    cfg = dataclasses.replace(cfg, deployment_mode=args.deployment_mode, local_agent_id=local_id)

    if tuple(sorted(cfg.jetson_endpoints.keys())) != EXPECTED_AGENTS:
        raise SystemExit(f"jetson_endpoints must have keys {EXPECTED_AGENTS}, got {sorted(cfg.jetson_endpoints)}")
    if local_id not in cfg.jetson_endpoints:
        raise SystemExit(f"local_agent_id {local_id!r} must be a key of cfg.jetson_endpoints (1:1 Jetson).")

    log_dir = Path(_PROJECT_ROOT) / cfg.results_dir / args.exp_name.replace(" ", "_")
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "config.json", "w") as f:
        json.dump(config_to_dict(cfg), f, indent=2, sort_keys=True)

    bind_port = _parse_bind_port(cfg.jetson_endpoints[local_id])
    transport = JetsonGrpcTransport(
        local_agent_id=local_id,
        peer_endpoints=dict(cfg.jetson_endpoints),
        bind_port=bind_port,
        calibration_output_dir=log_dir,
    )
    signal.signal(signal.SIGTERM, lambda s, f: _shutdown_hook(transport, s, f))
    signal.signal(signal.SIGINT, lambda s, f: _shutdown_hook(transport, s, f))

    transport.start_server()
    measured = transport.run_calibration(cfg_delta_max_ms=float(cfg.delta_max_ms))
    logger.info("Local calibration max one-way ms (this node): %.4f", measured)

    ch_coord = grpc.insecure_channel(cfg.coordinator_endpoint)
    coord_stub = cpb2_grpc.TeamCoordinatorStub(ch_coord)
    detail_path = log_dir / f"measured_delta_max_ms_{local_id}.json"
    detail_json = detail_path.read_text() if detail_path.is_file() else "{}"
    coord_stub.ReportCalibration(
        cpb2.CalibrationReport(
            agent_id=local_id,
            max_observed_one_way_ms=float(measured),
            detail_json=detail_json,
        ),
        timeout=30.0,
    )

    episode_stream = coord_stub.SubscribeEpisodes(cpb2.SubscribeAgent(agent_id=local_id))

    local_csv = log_dir / f"darts_log_{local_id}.csv"
    file_exists = local_csv.is_file()
    local_f = open(local_csv, "a", newline="")
    local_writer = csv.writer(local_f)
    if not file_exists:
        local_writer.writerow(EVAL_CSV_HEADER)

    def run_one_episode(spec: cpb2.EpisodeSpec) -> str:
        ep_cfg = dataclasses.replace(cfg, delta_max_ms=float(spec.effective_delta_max_ms))
        env = TurtleBot3Env(
            num_agents=ep_cfg.num_agents,
            global_deadline_ms=ep_cfg.global_deadline_ms,
            cfg=ep_cfg,
            seed=int(spec.dag_seed),
            workload=get_workload(ep_cfg.workload_id),
            transport=transport,
        )
        dag = DAGGraph.create_from_workload_id(ep_cfg.workload_id, ep_cfg)
        spatial_state_dim = env.single_observation_size
        agents: List[DartsMARLAgent] = []
        for i in range(ep_cfg.num_agents):
            agent = DartsMARLAgent(
                agent_id=f"agent_{i + 1}",
                cfg=ep_cfg,
                logger=logging.getLogger(f"darts_agent.agent_{i + 1}"),
                seed=int(spec.dag_seed) + i,
                spatial_state_dim=spatial_state_dim,
            )
            agent.dag_tracker = LocalDAGTracker(dag)
            agents.append(agent)

        ep_rewards, _e, last_info, _died, energy_viol, _raw = run_episode(
            env,
            agents,
            ep_cfg,
            explore=False,
            store_transitions=False,
            episode_verbose=False,
            phase="A",
            distributed_local_agent_id=local_id,
            episode_rng_seed=int(spec.dag_seed),
        )
        row = single_episode_eval_row_list(
            int(spec.episode_idx),
            env,
            ep_cfg,
            ep_rewards,
            last_info,
            energy_viol,
        )
        buf = io.StringIO()
        csv.writer(buf).writerow(row)
        return buf.getvalue().strip()

    try:
        for spec in episode_stream:
            row_s = run_one_episode(spec)
            local_writer.writerow(next(csv.reader([row_s])))
            local_f.flush()
            coord_stub.ReportEpisodeEnd(
                cpb2.EpisodeEndReport(agent_id=local_id, episode_idx=spec.episode_idx, csv_row=row_s),
                timeout=120.0,
            )
    finally:
        local_f.close()
        transport.stop_server()
        ch_coord.close()


if __name__ == "__main__":
    main()
