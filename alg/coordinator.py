"""
DARTS team coordinator. Runs on ONE process (any Jetson or laptop) and coordinates
episode lifecycle across the four agent_node processes. Does NOT participate in
auctions or scheduling — pure orchestration.

Launch:
    PYTHONPATH=. python alg/coordinator.py --workload B1_CDR --exp_name rtss_eval_run1 --num_episodes 100
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import os
import queue
import sys
import threading
import time
from concurrent import futures
from typing import Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import grpc

from alg.config_darts import DARTS_CONFIG
from alg.train_darts import EVAL_CSV_HEADER
from envs.proto import darts_coordinator_pb2 as cpb2
from envs.proto import darts_coordinator_pb2_grpc as cpb2_grpc
from envs.proto import darts_messaging_pb2 as mpb2
from envs.proto import darts_messaging_pb2_grpc as mpb2_grpc


EXPECTED_AGENTS = ("agent_1", "agent_2", "agent_3", "agent_4")


class CoordinatorState:
    """Shared coordinator state (gRPC servicer + main loop)."""

    def __init__(self, cfg: DARTS_CONFIG, results_subdir: str) -> None:
        self.cfg = cfg
        self.results_subdir = results_subdir
        self.episode_queues: Dict[str, queue.Queue] = {
            aid: queue.Queue() for aid in EXPECTED_AGENTS
        }
        self.calibration_max: Dict[str, float] = {}
        self.calibration_detail: Dict[str, str] = {}
        self.cal_lock = threading.Lock()
        self.cal_ready = threading.Event()
        self.end_lock = threading.Condition()
        self.end_reports: Dict[int, Dict[str, str]] = {}
        self.team_effective_delta_ms: float = float(cfg.delta_max_ms)

    def log_path(self) -> str:
        return os.path.join(_PROJECT_ROOT, self.cfg.results_dir, self.results_subdir)


class TeamCoordinatorServicer(cpb2_grpc.TeamCoordinatorServicer):
    def __init__(self, state: CoordinatorState, logger: logging.Logger) -> None:
        self._state = state
        self._log = logger

    def ReportCalibration(self, request: cpb2.CalibrationReport, context: grpc.ServicerContext) -> cpb2.CoordinatorAck:
        aid = request.agent_id
        if aid not in EXPECTED_AGENTS:
            return cpb2.CoordinatorAck(ok=False)
        with self._state.cal_lock:
            self._state.calibration_max[aid] = float(request.max_observed_one_way_ms)
            self._state.calibration_detail[aid] = request.detail_json
            n = len(self._state.calibration_max)
        self._log.info("Calibration report from %s: max_one_way_ms=%.4f (%d/4)", aid, request.max_observed_one_way_ms, n)
        if n >= 4:
            self._state.cal_ready.set()
        return cpb2.CoordinatorAck(ok=True)

    def ReportEpisodeEnd(self, request: cpb2.EpisodeEndReport, context: grpc.ServicerContext) -> cpb2.CoordinatorAck:
        ep = int(request.episode_idx)
        aid = request.agent_id
        with self._state.end_lock:
            d = self._state.end_reports.setdefault(ep, {})
            d[aid] = request.csv_row
            self._state.end_lock.notify_all()
        return cpb2.CoordinatorAck(ok=True)

    def SubscribeEpisodes(self, request: cpb2.SubscribeAgent, context: grpc.ServicerContext):
        aid = request.agent_id
        if aid not in self._state.episode_queues:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"Unknown agent_id {aid!r}")
            return
        q = self._state.episode_queues[aid]
        self._log.info("SubscribeEpisodes open for %s", aid)
        while context.is_active():
            try:
                spec = q.get(timeout=1.0)
                yield spec
            except queue.Empty:
                continue


def _enforce_team_topology(cfg: DARTS_CONFIG) -> None:
    keys = tuple(sorted(cfg.jetson_endpoints.keys()))
    if keys != EXPECTED_AGENTS:
        raise ValueError(
            "coordinator requires jetson_endpoints keys exactly "
            f"{EXPECTED_AGENTS!r}, got {keys!r} (1:1 agent ↔ Jetson)."
        )
    hosts = [cfg.jetson_endpoints[k].split(":")[0] for k in EXPECTED_AGENTS]
    if len(set(hosts)) != len(hosts):
        raise ValueError("jetson_endpoints must use distinct host IPs for each agent (1:1 Jetson).")


def _ping_agent_messenger(hostport: str, logger: logging.Logger, timeout_s: float = 3.0) -> None:
    ch = grpc.insecure_channel(hostport)
    try:
        stub = mpb2_grpc.AgentMessengerStub(ch)
        t0 = time.time() * 1000.0
        stub.Ping(mpb2.PingRequest(sender="coordinator", t_send_ms=t0), timeout=timeout_s)
    finally:
        ch.close()


def _finalize_calibration(state: CoordinatorState, logger: logging.Logger) -> None:
    if len(state.calibration_max) < 4:
        raise RuntimeError(
            f"Calibration incomplete: expected 4 agents, got {sorted(state.calibration_max.keys())}. "
            "Ensure all agent_node processes are running and can reach the coordinator."
        )
    team_max = max(state.calibration_max.values())
    state.team_effective_delta_ms = float(team_max)
    out_dir = state.log_path()
    os.makedirs(out_dir, exist_ok=True)
    team_path = os.path.join(out_dir, "measured_delta_max_ms.json")
    payload = {
        "max_observed_one_way_ms": team_max,
        "per_agent_max_one_way_ms": dict(state.calibration_max),
        "cfg_delta_max_ms": float(state.cfg.delta_max_ms),
    }
    with open(team_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote team calibration %s (max_observed_one_way_ms=%.4f)", team_path, team_max)
    if team_max > float(state.cfg.delta_max_ms) + 1e-6:
        raise RuntimeError(
            f"Empirical network latency violates the feasibility proof's bounded-latency assumption: "
            f"measured team max one-way latency is {team_max:.4f} ms but cfg.delta_max_ms is "
            f"{state.cfg.delta_max_ms:.4f} ms. Raise cfg.delta_max_ms (and ensure τ_timeout_ms > Δ_max) "
            f"before running hardware experiments. See {team_path}."
        )


def _wait_episode_csv_rows(state: CoordinatorState, episode_idx: int, timeout_s: float) -> Dict[str, str]:
    deadline = time.time() + timeout_s
    with state.end_lock:
        while time.time() < deadline:
            got = state.end_reports.get(episode_idx, {})
            if len(got) >= 4:
                return dict(got)
            state.end_lock.wait(timeout=0.2)
    raise TimeoutError(f"Episode {episode_idx}: timed out waiting for 4× ReportEpisodeEnd.")


def run_coordinator(
    cfg: DARTS_CONFIG,
    exp_name: str,
    num_episodes: int,
    bind_host: str,
    bind_port: int,
    episode_timeout_s: float = 7200.0,
    random_seed: Optional[int] = None,
) -> None:
    _enforce_team_topology(cfg)
    if random_seed is not None:
        cfg = dataclasses.replace(cfg, random_seed=int(random_seed))
    logger = logging.getLogger("coordinator")
    state = CoordinatorState(cfg, exp_name.replace(" ", "_"))
    os.makedirs(state.log_path(), exist_ok=True)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    cpb2_grpc.add_TeamCoordinatorServicer_to_server(TeamCoordinatorServicer(state, logger), server)
    server.add_insecure_port(f"{bind_host}:{bind_port}")
    server.start()
    logger.info("TeamCoordinator listening on %s:%s", bind_host, bind_port)

    for aid in EXPECTED_AGENTS:
        hp = cfg.jetson_endpoints[aid]
        logger.info("Ping AgentMessenger at %s (%s)", aid, hp)
        _ping_agent_messenger(hp, logger)

    if not state.cal_ready.wait(timeout=600.0):
        server.stop(0)
        raise RuntimeError("Timed out waiting for calibration reports from all 4 agents (600s).")

    _finalize_calibration(state, logger)

    team_csv = os.path.join(state.log_path(), "team_log.csv")
    file_exists = os.path.isfile(team_csv)
    with open(team_csv, "a", newline="") as f_team:
        w_team = csv.writer(f_team)
        if not file_exists:
            w_team.writerow(EVAL_CSV_HEADER)

        base_seed = int(cfg.random_seed)
        for ep in range(num_episodes):
            spec = cpb2.EpisodeSpec(
                episode_idx=ep,
                dag_seed=base_seed + ep,
                release_schedule_json="{}",
                effective_delta_max_ms=float(state.team_effective_delta_ms),
            )
            for aid in EXPECTED_AGENTS:
                state.episode_queues[aid].put(spec)
            logger.info("Broadcast EpisodeSpec episode_idx=%d dag_seed=%d", ep, spec.dag_seed)
            rows = _wait_episode_csv_rows(state, ep, timeout_s=episode_timeout_s)
            primary = rows.get("agent_1") or next(iter(rows.values()))
            parsed = next(csv.reader([primary]))
            w_team.writerow(parsed)
            with state.end_lock:
                state.end_reports.pop(ep, None)
            logger.info("Episode %d merged to team_log.csv", ep)

    server.stop(2.0)
    logger.info("Coordinator finished %d episodes.", num_episodes)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="DARTS team coordinator (gRPC orchestration).")
    p.add_argument("--workload", type=str, default="B1_CDR", choices=["B1_CDR", "B2_LML", "B3_IQI", "B4_DEM"])
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--num_episodes", type=int, default=1)
    p.add_argument("--bind_host", type=str, default="0.0.0.0")
    p.add_argument("--bind_port", type=int, default=None, help="Defaults to port from cfg.coordinator_endpoint.")
    p.add_argument("--random_seed", type=int, default=None, help="Override cfg.random_seed for episode DAG/env seeding.")
    args = p.parse_args()

    cfg = dataclasses.replace(DARTS_CONFIG(), workload_id=args.workload)
    port = args.bind_port
    if port is None:
        port = int(str(cfg.coordinator_endpoint).rsplit(":", 1)[-1])
    run_coordinator(
        cfg,
        args.exp_name,
        args.num_episodes,
        args.bind_host,
        port,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
