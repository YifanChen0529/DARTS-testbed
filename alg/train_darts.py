"""
DARTS Algorithm 2: Outer-Loop Training — production script.
Mirrors REFiNE-style setup: config.json, TensorBoard, episodic MARL loop on TurtleBot3Env,
checkpointing, and terminal metrics.

Run from project root: PYTHONPATH=. python alg/train_darts.py --exp_name debug
"""

from __future__ import division, print_function

import argparse
import csv
import dataclasses
import glob
import json
import logging
import contextlib
import copy
import math
import os
import sys
import time as _time_for_auction
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

# Add project root so "envs" and "alg" resolve when run from any directory (e.g. alg/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch

from envs.turtlebot3_spatial_env import TurtleBot3Env
from envs.simulation_env import LocalDAGTracker, NODE_STATE_ASSIGNED, NODE_STATE_READY
from envs.dag_task import DAGGraph
from envs.eval_node_mapping import csv_subtask_slots, csv_task_time_ms, update_ready_times_on_completion
from envs.workloads import get_workload, runtime_workload
from alg.coverage import CoverageTracker
from alg.periodic_release import get_exp2_runtime, PeriodicReleaseManager, set_exp2_runtime
from alg.defer_fallback import (
    FallbackOutcome,
    classify_no_bidder_outcome,
    compute_fallback_cost_ms,
    determine_sponsor_idx,
    execute_fallback_assign,
)
from alg.darts_agent import DartsMARLAgent, energy_cost_wh
from alg.darts_agent import scaled_cert_envelopes
from alg.darts_agent import partitioned_alloc
from alg.darts_agent import _CERT_ENVELOPE_SCALES
from alg.config_darts import DARTS_CONFIG
from alg.tb3_instance import apply_instance_seed_runtime
from scripts.csv_schema_tb import SUBROUND_LOG_HEADER
from scripts.csv_schema_tb import RELEASE_LOG_HEADER


@contextlib.contextmanager
def _delta_auc_subround_timer(samples_ms: List[float]) -> Iterator[None]:
    """Append wall-clock ms for one ready-job auction subround (Eq. 44 δ_auc samples)."""
    t0 = _time_for_auction.perf_counter()
    try:
        yield
    finally:
        samples_ms.append((_time_for_auction.perf_counter() - t0) * 1000.0)


def log_proof_regime(cfg: DARTS_CONFIG, logger: logging.Logger) -> None:
    """Log Theorem 1 proof regime parameters for runtime visibility."""
    bound_eta = cfg.lipschitz_rho_estimate * cfg.lr_eta * cfg.lambda_rho
    bound_phi = cfg.lipschitz_rho_estimate * cfg.lr_phi * cfg.lambda_rho
    logger.info(
        "Theorem 1 proof regime: L_rho=%.2f, lr_eta=%.2e, lr_phi=%.2e, lambda_rho=%.2f. "
        "Choice (1) margins: eta=%.4f/0.25 (%.0f%%), phi=%.4f/0.25 (%.0f%%). "
        "Choice (2) requires lambda_rho^2 >= 4n(G_R^2+G_P^2)/(Theta_min * min(Lambda_eta,Lambda_phi)); "
        "G_R, G_P bounded by gradient_clip_norm=%.2f.",
        cfg.lipschitz_rho_estimate,
        cfg.lr_eta,
        cfg.lr_phi,
        cfg.lambda_rho,
        bound_eta,
        100 * bound_eta / 0.25,
        bound_phi,
        100 * bound_phi / 0.25,
        cfg.gradient_clip_norm,
    )


# ---------------------------------------------------------------------------
# Config serialization for config.json
# ---------------------------------------------------------------------------

def config_to_dict(cfg: DARTS_CONFIG) -> Dict[str, Any]:
    """Serialize config to a JSON-serializable dict (for results/<exp_name>/config.json)."""
    return {
        "num_agents": cfg.num_agents,
        "workload_id": cfg.workload_id,
        "delta_max_ms": cfg.delta_max_ms,
        "global_deadline_ms": cfg.global_deadline_ms,
        "task_wcets": dict(cfg.task_wcets),
        "max_simulation_steps": cfg.max_simulation_steps,
        "max_episode_steps": cfg.max_episode_steps,
        "num_episodes": cfg.num_episodes,
        "train_episodes": cfg.train_episodes,
        "eval_every": cfg.eval_every,
        "eval_episodes": cfg.eval_episodes,
        "policy_train_steps": cfg.policy_train_steps,
        "lr_theta": cfg.lr_theta,
        "lr_eta": cfg.lr_eta,
        "use_continuous_bid": getattr(cfg, "use_continuous_bid", True),
        "bid_sigma": getattr(cfg, "bid_sigma", 0.1),
        "incentive_train_steps": cfg.incentive_train_steps,
        "lr_phi": cfg.lr_phi,
        "gamma": cfg.gamma,
        "reward_task_completion": getattr(cfg, "reward_task_completion", 10.0),
        "lambda_D": getattr(cfg, "lambda_D", 0.001),
        "lambda_E": getattr(cfg, "lambda_E", 10.0),
        "lambda_P": getattr(cfg, "lambda_P", 0.01),
        "lambda_rho": getattr(cfg, "lambda_rho", 1.0),
        "lipschitz_rho_estimate": getattr(cfg, "lipschitz_rho_estimate", 10.0),
        "gradient_clip_norm": getattr(cfg, "gradient_clip_norm", 1.0),
        "incentive_sigma": getattr(cfg, "incentive_sigma", 0.1),
        "reward_step_penalty": getattr(cfg, "reward_step_penalty", -0.01),
        "p_idle_w": getattr(cfg, "p_idle_w", 2.5),
        "p_moving_w": getattr(cfg, "p_moving_w", 7.5),
        "battery_capacities_wh": list(getattr(cfg, "battery_capacities_wh", [1.0, 1.5, 2.0, 2.5])),
        "e_safe_wh": getattr(cfg, "e_safe_wh", 0.05),
        "tau_timeout_ms": getattr(cfg, "tau_timeout_ms", 100.0),
        "p_comm_w": getattr(cfg, "p_comm_w", 0.5),
        "max_queue_size": getattr(cfg, "max_queue_size", 10),
        "max_work_ms": getattr(cfg, "max_work_ms", 2000.0),
        "results_dir": cfg.results_dir,
        "save_interval": cfg.save_interval,
        "random_seed": cfg.random_seed,
        "log_level": cfg.log_level,
        "deployment_mode": getattr(cfg, "deployment_mode", "simulator"),
        "jetson_endpoints": dict(getattr(cfg, "jetson_endpoints", {})),
        "local_agent_id": getattr(cfg, "local_agent_id", None),
        "coordinator_endpoint": getattr(cfg, "coordinator_endpoint", None),
        "epoch_length_ms": getattr(cfg, "epoch_length_ms", 1000.0),
        "disable_feasibility_mask": getattr(cfg, "disable_feasibility_mask", False),
    }


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

# Logger for in-episode Cyber-Physical events (auction, RTA fallback, completion).
# Enable INFO during eval/first episode via set_episode_logging_verbose(True).
EPISODE_EVENTS_LOGGER = "episode_events"


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # By default silence granular episode events during training (avoid spam)
    logging.getLogger(EPISODE_EVENTS_LOGGER).setLevel(logging.WARNING)
    return logging.getLogger("train_darts")


# Loggers that emit in-episode event spam (assignments, RTA fallback, completions)
IN_EPISODE_LOGGER_NAMES = (
    EPISODE_EVENTS_LOGGER,       # "episode_events"
    "darts_agent",              # darts_agent.agent_1, agent_2, ...
    "envs.turtlebot3_spatial_env",
)


def set_episode_logging_verbose(verbose: bool, debug_auction: bool = False) -> None:
    """Enable (True) or silence (False) INFO/WARNING in-episode logs for summary episodes only.
    If debug_auction is True and verbose is True, episode_events is set to DEBUG (Sponsor Bias logs)."""
    level = logging.INFO if verbose else logging.ERROR
    for name in IN_EPISODE_LOGGER_NAMES:
        if name == EPISODE_EVENTS_LOGGER and verbose and debug_auction:
            logging.getLogger(name).setLevel(logging.DEBUG)
        else:
            logging.getLogger(name).setLevel(level)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_episode(
    env: TurtleBot3Env,
    agents: List[DartsMARLAgent],
    cfg: DARTS_CONFIG,
    explore: bool,
    store_transitions: bool = True,
    episode_verbose: bool = True,
    phase: str = "A",
    debug_auction: bool = False,
    coverage_tracker: Optional[CoverageTracker] = None,
    distributed_local_agent_id: Optional[str] = None,
    episode_rng_seed: Optional[int] = None,
    episode_idx: Optional[int] = None,
    log_per_commit: bool = False,
    subround_writer: Optional["SubroundLogWriter"] = None,
    subround_phase: Optional[str] = None,
) -> Tuple[List[float], List[float], Dict[str, Any], List[bool], List[int], float]:
    """
    Run one episode: reset env, then while not done step with agent actions,
    compute shaped reward (Eq. 22), optionally store transitions.
    phase: "A" = store payments for penalty (Trajectory A); "B" = store incentive outputs for J^inc (Trajectory B).
    When episode_verbose is True, INFO-level in-episode logs (auction, RTA, completion) are printed.
    When debug_auction is True, episode_events logger is set to DEBUG (Sponsor Bias bid details).
    ``coverage_tracker``: if None, a local tracker is created for sponsor / full-fidelity logging.
    ``distributed_local_agent_id``: when set with ``cfg.deployment_mode == "distributed_jetson"``,
    runs a peer-to-peer epoch barrier (``EPOCH_READY``) at each new epoch slot.
    ``episode_rng_seed``: passed to ``env.reset(rng_seed=...)`` for cross-Jetson reproducibility.
    Returns: (episode_rewards_per_agent, total_energy_J_per_agent, last_step_info, agent_died,
              energy_cert_violations, raw_env_reward_sum).
    """
    set_episode_logging_verbose(episode_verbose, debug_auction=debug_auction)
    max_steps = cfg.max_episode_steps
    step_dt_s = env.step_dt_s

    # Reset all agents and env so no state bleeds from previous episodes
    for agent in agents:
        agent.reset()
    observations, _ = env.reset(rng_seed=episode_rng_seed)

    # Fresh DAG tracker each episode (entry nodes READY, rest WAITING)
    workload_ep = runtime_workload(env, cfg)
    dag = DAGGraph.from_workload(workload_ep, cfg)
    global_tracker = LocalDAGTracker(dag)
    if coverage_tracker is None:
        coverage_tracker = CoverageTracker(len(dag.nodes), len(agents))

    completed_set: Set[str] = set()
    ready_time_ms = {
        nid: 0.0 for nid in workload_ep.node_ids if not workload_ep.predecessors[nid]
    }

    # EXP-2 REBIND-AFTER-INSERT: the single-shot dag / LocalDAGTracker /
    # ready_time_ms above keep executing untouched. When release_mode is
    # periodic we rebind those names to the manager. The discarded objects
    # were never registered on env (reset already ran against class ids) so
    # they have no env side effects.
    _exp2_prm = None
    _exp2_rt = get_exp2_runtime()
    if _exp2_rt.release_mode == "periodic":
        from alg.defer_fallback import reset_fallback_blocked as _reset_fb
        _reset_fb()
        _exp2_prm = PeriodicReleaseManager(
            P=float(_exp2_rt.release_period_ms),
            L=int(_exp2_rt.num_releases),
            D_v=(
                float(_exp2_rt.D_v_ms)
                if _exp2_rt.D_v_ms
                else float(workload_ep.relative_deadline_ms)
            ),
            per_period_energy=_exp2_rt.per_period_energy,
        )
        ready_time_ms, global_tracker = _exp2_prm.bootstrap(
            env, cfg, workload_ep, _exp2_prm.P, _exp2_prm.L, _exp2_prm.D_v,
        )
        dag = _exp2_prm.merged_dag
        for _a in agents:
            _a.dag_tracker = global_tracker
        if coverage_tracker is not None:
            coverage_tracker.V_total = _exp2_prm.n_jobs_expected

    episode_rewards: List[float] = [0.0] * len(agents)
    total_energy_J_per_agent: List[float] = [0.0] * len(agents)
    last_info: Dict[str, Any] = {}
    agent_died: List[bool] = [False] * len(agents)
    episode_logger = logging.getLogger(EPISODE_EVENTS_LOGGER)
    auction_interval = 1  # run auction every step so newly_completed_nodes unlock ready jobs immediately
    deadline_missed = False  # set True when we break due to end_to_end_deadline_miss
    initial_battery_wh = env.initial_battery_wh  # Eq. 22: relative depletion per agent
    crash_times_ms: List[Optional[float]] = [None] * len(agents)
    timeout_processed: List[bool] = [False] * len(agents)
    tau_timeout_ms = getattr(cfg, "tau_timeout_ms", 100.0)
    payments_received: List[float] = [0.0] * len(agents)
    payments_sent: List[float] = [0.0] * len(agents)
    task_assigned_to: Dict[str, int] = {}
    # §V.G T2 instrumentation (admission-conditional response-time tightness)
    per_commit_log: List[Dict[str, Any]] = []
    log_per_commit_enabled: bool = log_per_commit
    energy_cert_violations: List[int] = [0] * len(agents)
    commit_time_ms: Dict[str, float] = {}
    absolute_deadline_ms: Dict[str, float] = {}
    t_auc_per_epoch_ms: List[float] = []
    A_k_per_epoch: List[int] = []
    delta_auc_samples_ms: List[float] = []
    raw_env_reward_sum = 0.0
    deferred_jobs: Set[str] = set()
    V_total_episode = len(dag.nodes)
    if _exp2_prm is not None:
        V_total_episode = _exp2_prm.n_jobs_expected
    n_full_fidelity = 0
    task_outcome_dict: Dict[str, str] = {}
    n_degradable_fallback = 0
    n_nondegradable_n1 = 0
    n_nondegradable_n2 = 0
    n_defer_triggers = 0
    last_epoch_barrier_slot = -1
    # ----- TB-1 instrumentation accumulators (main-schema columns 64..76) -----
    n_bids_evaluated = 0          # Σ over subrounds of |alive agents polled for a bid|
    n_bids_cert_pass = 0          # Σ over subrounds of |C_v| (bids that cleared the certificate)
    pool_sizes: List[int] = []    # per-subround |C_v|, for mean_pool_size
    exec_ms_per_agent: List[float] = [0.0] * len(agents)
    travel_ms_per_agent: List[float] = [0.0] * len(agents)
    episode_span_ms = 0.0
    # ----- TB-1 subround sidecar (only materialised when a writer is supplied) -----
    subround_rows: List[Dict[str, Any]] = []
    _sr_phase = subround_phase if subround_phase is not None else phase

    def _sr_emit(
        clock_ms_v: float,
        job_id: str,
        pos_j: int,
        a_k: int,
        n_alive: int,
        c_v: int,
        sample_idx: int,
        t_cert: float,
        t_infer: float,
        t_fallback: float,
        t_commit: float,
        winner: int,
        outcome: str,
    ) -> None:
        """Buffer one subround record. ``t_subround_ms`` is resolved at flush time from
        ``delta_auc_samples_ms[sample_idx]``, which the timer appends on block exit."""
        if subround_writer is None:
            return
        subround_rows.append({
            "episode": int(episode_idx) if episode_idx is not None else -1,
            "phase": _sr_phase,
            "clock_ms": float(clock_ms_v),
            "job_id": job_id,
            "subround_pos_j": int(pos_j),
            "A_k_size": int(a_k),
            "n_alive": int(n_alive),
            "C_v_size": int(c_v),
            "_sample_idx": int(sample_idx),
            "t_cert_ms": float(t_cert),
            "t_infer_ms": float(t_infer),
            "t_exchange_ms": 0.0,  # simulator: no wire exchange (see scripts/csv_schema_tb.py)
            "t_fallback_ms": float(t_fallback),
            "t_commit_ms": float(t_commit),
            "queue_len_per_agent": json.dumps(
                [len(a.queue) for a in agents], separators=(",", ":")
            ),
            "winner_idx": int(winner),
            "outcome": outcome,
        })

    for step in range(max_steps):
        next_observations, env_rewards, dones, info = env.step(agents=agents)
        last_info = info
        raw_env_reward_sum += float(info.get("raw_env_reward", 0.0))

        if distributed_local_agent_id is not None and getattr(cfg, "deployment_mode", "simulator") == "distributed_jetson":
            clock_ms = float(info.get("clock_ms", env.clock_ms))
            epoch_len_ms = float(getattr(cfg, "epoch_length_ms", 1000.0))
            epoch_slot = int(clock_ms // epoch_len_ms) if epoch_len_ms > 1e-9 else 0
            if epoch_slot > last_epoch_barrier_slot:
                from alg.distributed_sync import sync_epoch_barrier

                sync_epoch_barrier(
                    env.network,
                    cfg,
                    distributed_local_agent_id,
                    epoch_slot,
                    clock_ms,
                    agents,
                )
                last_epoch_barrier_slot = epoch_slot

        # Force terminate episode when global deadline is exceeded (stop simulation at 120s)
        if info.get("end_to_end_deadline_miss", False):
            deadline_missed = True
            break

        # Record crash time for each agent (Eq. 49)
        for i in range(len(agents)):
            if dones[i] and crash_times_ms[i] is None:
                crash_times_ms[i] = env.clock_ms

        # Update central tracker with newly completed nodes (physical completion this step)
        if global_tracker is not None:
            newly_completed = info.get("newly_completed_nodes", [])
            if _exp2_prm is not None:
                _exp2_prm.note_completion_clock(float(env.clock_ms))
            for node_id in newly_completed:
                global_tracker.mark_completed(node_id)
            for node_id in newly_completed:
                completed_set.add(node_id)
            update_ready_times_on_completion(
                workload=workload_ep,
                newly_completed=newly_completed,
                completed_set=completed_set,
                clock_ms=float(env.clock_ms),
                ready_time_ms=ready_time_ms,
            )

        # Eq. 49: Process heartbeat timeouts — revert orphaned jobs to READY and clear crashed agent queue
        if global_tracker is not None:
            for i in range(len(agents)):
                if crash_times_ms[i] is not None and not timeout_processed[i]:
                    if env.clock_ms >= crash_times_ms[i] + tau_timeout_ms:
                        timeout_processed[i] = True
                        orphaned_jobs = list(agents[i].queue)
                        for job in orphaned_jobs:
                            global_tracker.node_states[job.job_id] = NODE_STATE_READY
                        agents[i].reset_queue()
                        episode_logger.critical(
                            "[Time: %.0fms] 💀 Heartbeat timeout for Agent %d (crashed at %.0fms). Reverting %d pending jobs to READY.",
                            env.clock_ms, i + 1, crash_times_ms[i], len(orphaned_jobs),
                        )

        # Algorithm 1: DARTS Auction — epoch-level bidding; only winning agent stores bid event
        battery_wh = info.get("battery_wh", [0.0] * len(agents))
        if _exp2_prm is not None:
            _restored = _exp2_prm.on_post_step_clock(float(env.clock_ms), agents, env)
            if _restored:
                battery_wh = list(env._battery_wh)
        if global_tracker is not None and (step + 1) % auction_interval == 0:
            deferred_this_epoch: Set[str] = set()
            _t_epoch_auction_start = _time_for_auction.perf_counter()
            clock_ms = env.clock_ms
            g_deadline = float(getattr(cfg, "global_deadline_ms", 180000.0))
            epoch_len_ms = float(getattr(cfg, "epoch_length_ms", 1000.0))
            max_v = float(getattr(env, "max_velocity_mps", 0.22))
            workload = getattr(env, "workload", None)
            ready_jobs = sorted(
                global_tracker.get_ready_nodes(),
                key=lambda j: (
                    ready_time_ms.get(j.node_id, clock_ms)
                    + float(j.relative_deadline or g_deadline)
                ),
            )
            _sr_pos_j = 0
            for job in ready_jobs:
                _sr_pos_j += 1
                release_time_ms = ready_time_ms.get(job.node_id, clock_ms)
                with _delta_auc_subround_timer(delta_auc_samples_ms):
                    _sr_sample_idx = len(delta_auc_samples_ms)
                    _sr_t_cert_ms = 0.0
                    _sr_t_infer_ms = 0.0
                    feasible_indices = [i for i in range(len(agents)) if not dones[i]]
                    if not feasible_indices:
                        continue
                    bids_this: List[float] = []
                    log_probs_this: List[Any] = []  # (agent_idx, log_prob, dynamic_budget_ms)
                    for i in feasible_indices:
                        is_feasible, bid_val, log_prob, dynamic_budget_ms = agents[i].compute_bid(
                            job,
                            clock_ms,
                            current_battery_wh=battery_wh[i] if i < len(battery_wh) else 0.0,
                            explore=explore,
                            env=env,
                            release_time_ms=release_time_ms,
                        )
                        if is_feasible:
                            bids_this.append(bid_val)
                            log_probs_this.append((i, log_prob, dynamic_budget_ms))
                        else:
                            bids_this.append(-float("inf"))
                            log_probs_this.append((i, None, None))
                    # TB-1: |C_v| = bids that cleared the certificate in this subround.
                    pool = sum(1 for _b in bids_this if _b > -float("inf"))
                    n_bids_evaluated += len(feasible_indices)
                    n_bids_cert_pass += pool
                    pool_sizes.append(pool)
                    for _i_sr in feasible_indices:
                        _bt = getattr(agents[_i_sr], "last_bid_timing", None) or {}
                        _sr_t_cert_ms += float(_bt.get("t_cert_ms", 0.0))
                        _sr_t_infer_ms += float(_bt.get("t_infer_ms", 0.0))
                    if not bids_this or max(bids_this) <= -float("inf"):
                        _sr_t0_fallback = _time_for_auction.perf_counter()
                        sponsor_idx = determine_sponsor_idx(
                            job,
                            env,
                            feasible_indices,
                            task_assigned_to,
                            env.completed_tasks_ms,
                        )
                        sponsor_pos = env.get_agent_position(sponsor_idx)
                        task_xy = env.task_locations.get(job.node_id) if hasattr(env, "task_locations") else None
                        c_fb: Optional[float] = None
                        if workload is not None and sponsor_pos is not None and task_xy is not None:
                            c_fb = compute_fallback_cost_ms(
                                sponsor_idx,
                                job.node_id,
                                sponsor_pos,
                                task_xy,
                                workload=workload,
                                max_velocity_mps=max_v,
                            )
                        d_v = ready_time_ms.get(job.node_id, clock_ms) + float(
                            job.relative_deadline or g_deadline
                        )
                        absolute_deadline_ms[job.node_id] = d_v
                        outcome = classify_no_bidder_outcome(
                            job.node_id,
                            clock_ms,
                            d_v,
                            epoch_length_ms=epoch_len_ms,
                            c_fb_v_ms=c_fb,
                            already_deferred_once=(job.node_id in deferred_jobs),
                        )
                        if outcome == FallbackOutcome.DEGRADABLE_DEFER:
                            deferred_this_epoch.add(job.node_id)
                            n_defer_triggers += 1
                            episode_logger.info(
                                "[t=%.0fms] Job %s: DEGRADABLE_DEFER (c_FB=%s, d_v-t_k=%.0fms)",
                                clock_ms,
                                job.node_id,
                                f"{c_fb:.0f}" if c_fb is not None else "None",
                                d_v - clock_ms,
                            )
                            _sr_emit(
                                clock_ms, job.node_id, _sr_pos_j, len(ready_jobs),
                                len(feasible_indices), pool, _sr_sample_idx,
                                _sr_t_cert_ms, _sr_t_infer_ms,
                                (_time_for_auction.perf_counter() - _sr_t0_fallback) * 1000.0,
                                0.0, -1, "defer",
                            )
                            continue
                        if outcome == FallbackOutcome.DEGRADABLE_FALLBACK_NOW:
                            c_tilde = (
                                workload.fallback_wcet_ms.get(job.node_id)
                                if workload is not None
                                else None
                            )
                            if c_tilde is None and workload is not None:
                                c_tilde = workload.fallback_wcet_ms.get(
                                    job.node_id.split("#")[0]
                                )
                            if c_tilde is not None:
                                execute_fallback_assign(
                                    agents,
                                    global_tracker,
                                    env,
                                    cfg,
                                    job,
                                    sponsor_idx,
                                    float(c_tilde),
                                    clock_ms,
                                    battery_wh,
                                    commit_time_ms,
                                    task_assigned_to,
                                    energy_cert_violations,
                                    release_time_ms=release_time_ms,
                                    episode_logger=episode_logger,
                                )
                                if _exp2_prm is not None and job.node_id in commit_time_ms:
                                    _exp2_prm.record_commit(
                                        job.node_id, agents[sponsor_idx].agent_id, clock_ms
                                    )
                                n_degradable_fallback += 1
                                task_outcome_dict[job.node_id] = "fallback_now"
                                deferred_jobs.discard(job.node_id)
                                episode_logger.warning(
                                    "[t=%.0fms] Job %s: DEGRADABLE_FALLBACK_NOW via agent %d (c_FB=%.0fms)",
                                    clock_ms,
                                    job.node_id,
                                    sponsor_idx + 1,
                                    float(c_fb) if c_fb is not None else float("nan"),
                                )
                            _sr_emit(
                                clock_ms, job.node_id, _sr_pos_j, len(ready_jobs),
                                len(feasible_indices), pool, _sr_sample_idx,
                                _sr_t_cert_ms, _sr_t_infer_ms,
                                (_time_for_auction.perf_counter() - _sr_t0_fallback) * 1000.0,
                                0.0, sponsor_idx, "fallback_now",
                            )
                            continue
                        if outcome == FallbackOutcome.NONDEGRADABLE_N2_CASUALTY:
                            c_tilde = (
                                workload.fallback_wcet_ms.get(job.node_id)
                                if workload is not None
                                else None
                            )
                            if c_tilde is None and workload is not None:
                                c_tilde = workload.fallback_wcet_ms.get(
                                    job.node_id.split("#")[0]
                                )
                            if c_tilde is not None:
                                execute_fallback_assign(
                                    agents,
                                    global_tracker,
                                    env,
                                    cfg,
                                    job,
                                    sponsor_idx,
                                    float(c_tilde),
                                    clock_ms,
                                    battery_wh,
                                    commit_time_ms,
                                    task_assigned_to,
                                    energy_cert_violations,
                                    release_time_ms=release_time_ms,
                                    episode_logger=episode_logger,
                                )
                                if _exp2_prm is not None and job.node_id in commit_time_ms:
                                    _exp2_prm.record_commit(
                                        job.node_id, agents[sponsor_idx].agent_id, clock_ms
                                    )
                                n_nondegradable_n2 += 1
                                task_outcome_dict[job.node_id] = "n2_casualty"
                                deferred_jobs.discard(job.node_id)
                                episode_logger.error(
                                    "[t=%.0fms] Job %s: NONDEGRADABLE_N2_CASUALTY (overload casualty, deadline miss)",
                                    clock_ms,
                                    job.node_id,
                                )
                            _sr_emit(
                                clock_ms, job.node_id, _sr_pos_j, len(ready_jobs),
                                len(feasible_indices), pool, _sr_sample_idx,
                                _sr_t_cert_ms, _sr_t_infer_ms,
                                (_time_for_auction.perf_counter() - _sr_t0_fallback) * 1000.0,
                                0.0, sponsor_idx, "n2_casualty",
                            )
                            continue
                        if outcome == FallbackOutcome.NONDEGRADABLE_N1_DROP:
                            n_nondegradable_n1 += 1
                            task_outcome_dict[job.node_id] = "n1_drop"
                            global_tracker.mark_dropped(job.node_id)
                            deferred_jobs.discard(job.node_id)
                            episode_logger.critical(
                                "[t=%.0fms] Job %s: NONDEGRADABLE_N1_DROP (structural-infeasibility alert raised)",
                                clock_ms,
                                job.node_id,
                            )
                            _sr_emit(
                                clock_ms, job.node_id, _sr_pos_j, len(ready_jobs),
                                len(feasible_indices), pool, _sr_sample_idx,
                                _sr_t_cert_ms, _sr_t_infer_ms,
                                (_time_for_auction.perf_counter() - _sr_t0_fallback) * 1000.0,
                                0.0, -1, "n1_drop",
                            )
                            continue
    
                    # Eq. 18: j*(v) = argmax_j { bid_{j,v} + p_{s→j}(v) } s.t. cert_{j,v} = 1
                    # Eq. 11: sponsor = agent that completed the last predecessor (triggering v's release)
                    sponsor_idx = determine_sponsor_idx(
                        job,
                        env,
                        feasible_indices,
                        task_assigned_to,
                        env.completed_tasks_ms,
                    )
                    sponsor_agent = agents[sponsor_idx]
                    coverage_tracker.record_sponsor(sponsor_idx, job.node_id)
                    # Build χ^p_s per Eq. 17: vector of bids from valid candidates + P^max_s
                    bid_vector = [0.0] * len(agents)
                    for k_idx in range(len(feasible_indices)):
                        agent_idx_k = feasible_indices[k_idx]
                        if bids_this[k_idx] > -float("inf"):
                            bid_vector[agent_idx_k] = bids_this[k_idx]
                    P_max_s = 1.0  # P^max_s = 1 per draft
                    chi_p = torch.tensor(
                        bid_vector + [P_max_s],
                        dtype=torch.float32,
                        device=sponsor_agent.device,
                    ).unsqueeze(0)
                    payment_t = sponsor_agent.incentive_net(chi_p)
                    if phase == "A":
                        payment = payment_t.squeeze().item()
                        sponsor_agent.store_payment_phase_a(payment, job.node_id, sponsor_idx)
                    else:
                        # Phase B: stochastic incentive for Eq. 25 (log_prob * R); sample and store log_prob
                        sigma_phi = getattr(cfg, "incentive_sigma", 0.1)
                        noise = torch.randn_like(payment_t, device=payment_t.device)
                        payment_raw = (payment_t + sigma_phi * noise).clamp(0.0, 1.0)
                        payment = payment_raw.squeeze().item()
                        log_prob = torch.distributions.Normal(payment_t, sigma_phi).log_prob(payment_raw).sum()
                        sponsor_agent.store_incentive_event(log_prob)
    
                    # Combined bids: Self-Incentive Matching (SIM) — payment added uniformly
                    # to all agents so allocation is purely bid-driven. Reward transfer
                    # (lines 418-420) still only flows when winner ≠ sponsor.
                    combined_bids: List[float] = []
                    # Parallel-branch penalty: discourage monopolizing co-deadline tasks
                    # at the same location (e.g., B+C both at TARGET with same deadline)
                    job_deadline = release_time_ms + float(
                        job.relative_deadline or getattr(cfg, "global_deadline_ms", 180000.0)
                    )
                    job_target = (
                        env.task_locations.get(job.node_id)
                        if env is not None and hasattr(env, "task_locations")
                        else None
                    )
                    for k in range(len(feasible_indices)):
                        agent_idx = feasible_indices[k]
                        raw_bid = bids_this[k]
                        if raw_bid <= -float("inf"):
                            combined_bids.append(-float("inf"))
                            continue
                        queue_penalty = 0.0
                        for qj in agents[agent_idx].queue:
                            if abs(qj.absolute_deadline_ms - job_deadline) < 1.0:
                                if job_target is not None:
                                    qj_target = (
                                        env.task_locations.get(qj.job_id)
                                        if env is not None
                                        else None
                                    )
                                    if qj_target == job_target:
                                        queue_penalty = 0.3
                                        break
                        effective_bid = raw_bid - queue_penalty
                        combined_bids.append(effective_bid + payment)
    
                    _sr_t0_commit = _time_for_auction.perf_counter()
                    winner_local_idx = int(max(range(len(combined_bids)), key=lambda k: combined_bids[k]))
                    winner_idx = feasible_indices[winner_local_idx]
                    winning_dynamic_budget = log_probs_this[winner_local_idx][2]
                    # Eq. 24: ALL agents store bid log_probs for REINFORCE, not just winner
                    for k_store in range(len(feasible_indices)):
                        store_idx = feasible_indices[k_store]
                        store_lp = log_probs_this[k_store][1]
                        if store_lp is not None:
                            agents[store_idx].store_bid_event(store_lp)
                    # Debug: Sponsor Bias — log job, sponsor/winner, distances, raw bid, payment, combined bid per agent
                    episode_logger.debug(
                        "[Sponsor Bias] Job %s target=%s | Sponsor=Agent %d | Winner=Agent %d",
                        job.node_id, job_target, sponsor_idx + 1, winner_idx + 1,
                    )
                    for agent_idx in range(4):
                        pos = env.get_agent_position(agent_idx) if env is not None and hasattr(env, "get_agent_position") else None
                        dist = (
                            ((pos[0] - job_target[0]) ** 2 + (pos[1] - job_target[1]) ** 2) ** 0.5
                            if job_target is not None and pos is not None else float("nan")
                        )
                        if agent_idx in feasible_indices:
                            k = feasible_indices.index(agent_idx)
                            raw_bid_k = bids_this[k]
                            inc_payment = payment if agent_idx != sponsor_idx else 0.0
                            combined_k = combined_bids[k]
                            episode_logger.debug(
                                "  Agent %d: dist=%.2f m | raw_bid=%.4f incentive=%.4f combined_bid=%.4f",
                                agent_idx + 1, dist, raw_bid_k, inc_payment, combined_k,
                            )
                        else:
                            episode_logger.debug(
                                "  Agent %d: dist=%.2f m | (not feasible, no bid)",
                                agent_idx + 1, dist,
                            )
                    best_bid = bids_this[winner_local_idx]
                    # Incentive forward pass used for auction; Eq. 25 outer-loop uses agent trajectory in update_parameters
                    agents[winner_idx].assign_job(
                        job,
                        release_time_ms=release_time_ms,
                        current_time_ms=clock_ms,
                        bid=best_bid,
                        dynamic_budget_ms=winning_dynamic_budget,
                    )
                    global_tracker.node_states[job.node_id] = NODE_STATE_ASSIGNED
                    task_assigned_to[job.node_id] = winner_idx
                    if job.node_id not in absolute_deadline_ms:
                        absolute_deadline_ms[job.node_id] = (
                            ready_time_ms.get(job.node_id, clock_ms)
                            + float(job.relative_deadline or g_deadline)
                        )
                    commit_time_ms[job.node_id] = clock_ms
                    if _exp2_prm is not None:
                        _exp2_prm.record_commit(
                            job.node_id, agents[winner_idx].agent_id, clock_ms
                        )
                    coverage_tracker.record_full_fidelity_commit(sponsor_idx, job.node_id)
                    n_full_fidelity += 1
                    task_outcome_dict[job.node_id] = "full_fidelity"
                    deferred_jobs.discard(job.node_id)
                    # Energy cert violation check (Eq. energy_feasible in draft)
                    e_safe = getattr(cfg, "e_safe_wh", 0.05)
                    battery_now = env._battery_wh[winner_idx] if winner_idx < len(env._battery_wh) else 0.0
                    delta_e_wc = (winning_dynamic_budget / 3600000.0) * getattr(cfg, "p_moving_w", 7.5)
                    if battery_now < e_safe + delta_e_wc:
                        energy_cert_violations[winner_idx] += 1
                        episode_logger.warning(
                            "[Time: %.0fms] ⚡ Agent %d ENERGY CERT VIOLATION: B=%.4f Wh < e_safe(%.4f) + ΔE_WC(%.4f)",
                            clock_ms, winner_idx + 1, battery_now, e_safe, delta_e_wc,
                        )
                    lambda_B = getattr(cfg, "lambda_B", 2.0)
                    episode_rewards[winner_idx] -= lambda_B * best_bid
                    if winner_idx != sponsor_idx and payment > 0:
                        payments_received[winner_idx] += payment
                        payments_sent[sponsor_idx] += payment
                    episode_logger.info(
                        "[Time: %.0fms] 🎯 Job %s ASSIGNED to Agent %d (Bid: %.2f)",
                        clock_ms, job.node_id, winner_idx + 1, best_bid,
                    )
                    _sr_emit(
                        clock_ms, job.node_id, _sr_pos_j, len(ready_jobs),
                        len(feasible_indices), pool, _sr_sample_idx,
                        _sr_t_cert_ms, _sr_t_infer_ms, 0.0,
                        (_time_for_auction.perf_counter() - _sr_t0_commit) * 1000.0,
                        winner_idx, "commit",
                    )

            deferred_jobs.update(deferred_this_epoch)
            _t_epoch_auction_end = _time_for_auction.perf_counter()
            if len(ready_jobs) > 0:
                t_auc_per_epoch_ms.append(
                    (_t_epoch_auction_end - _t_epoch_auction_start) * 1000.0
                )
                A_k_per_epoch.append(len(ready_jobs))

        for i, d in enumerate(dones):
            if d:
                agent_died[i] = True
        is_moving_per_agent = info.get("is_moving", [False] * len(agents))
        # TB-1 busy time. Mutually exclusive by construction: execute_task only runs at
        # target, and at target the mover reports is_moving False. Dead agents accrue
        # nothing, so exec+travel is bounded by n_agents * episode_span_ms.
        is_executing_per_agent = info.get("is_executing", [False] * len(agents))
        for i in range(len(agents)):
            if dones[i]:
                continue
            if i < len(is_moving_per_agent) and is_moving_per_agent[i]:
                travel_ms_per_agent[i] += step_dt_s * 1000.0
            if i < len(is_executing_per_agent) and is_executing_per_agent[i]:
                exec_ms_per_agent[i] += step_dt_s * 1000.0
        # Eq. 19: DC(t) = Σ_{v ∈ Done(t)} max(0, t - d_v)
        penalty_deadline = 0.0
        newly_done = info.get("newly_completed_nodes", [])
        global_deadline = getattr(cfg, "global_deadline_ms", 180000.0)
        for node_id in newly_done:
            completion_time = env.completed_tasks_ms.get(node_id, 0.0)
            d_v = global_deadline  # all tasks share D_G as absolute deadline
            penalty_deadline += max(0.0, completion_time - d_v)
        for i, agent in enumerate(agents):
            is_moving = is_moving_per_agent[i] if i < len(is_moving_per_agent) else False
            cost_energy_J = (env.p_moving_w if is_moving else env.p_idle_w) * step_dt_s
            total_energy_J_per_agent[i] += cost_energy_J
            init_wh = initial_battery_wh[i] if i < len(initial_battery_wh) else getattr(env, "battery_capacity_wh", 50.0)
            cost_energy_norm = energy_cost_wh(agent.cfg, is_moving, step_dt_s, initial_capacity_wh=init_wh)
            r_task = env_rewards[i]
            lambda_D = getattr(agent.cfg, "lambda_D", 0.001)
            lambda_E = getattr(agent.cfg, "lambda_E", 10.0)
            episode_rewards[i] += r_task - lambda_D * penalty_deadline - lambda_E * cost_energy_norm

        observations = next_observations
        if all(dones):
            break

    episode_span_ms = float(getattr(env, "clock_ms", 0.0))

    # Eq. 22: add net incentive (Σ p_{j→i} - Σ p_{i→j}) to episode return (equivalent for REINFORCE)
    for i in range(len(agents)):
        episode_rewards[i] += payments_received[i] - payments_sent[i]

    # Massive failure penalty when episode ended due to system-wide deadline miss
    if deadline_missed:
        for i in range(len(agents)):
            episode_rewards[i] += -50.0

    # ----------------------------------------------------------
    # Paper v6 Theorem 1 Part (2): per-job timing decomposition.
    #   τ(v)         = ready_time_ms[v]                 (Eq. 29)
    #   t_v^com      = commit_time_ms[v]                (Eq. 30)
    #   C_v          = env.completed_tasks_ms[v]        (physical completion)
    #   A_v := t_v^com - τ(v)                           (Eq. 31, allocation delay)
    #   E_v := C_v - t_v^com                            (Eq. 31, execution delay)
    #   R_v := A_v + E_v = C_v - τ(v)                   (Eq. 31, response time)
    #   v6 RT bound (Eq. 47): R_v ≤ 2H + 2T̄_auc + d_v - t_k
    # In simulator the auction is wall-clock-instantaneous, so T̄_auc = 0.
    # ----------------------------------------------------------
    H_ms = getattr(cfg, "epoch_length_ms", 1000.0)
    T_auc_bar_ms = float(getattr(cfg, "t_auc_bar_ms", 0.0))  # 0 in simulator
    response_times: Dict[str, Optional[float]] = {}
    allocation_delays: Dict[str, Optional[float]] = {}
    rt_violations = 0
    completed_ms = getattr(env, "completed_tasks_ms", {}) or {}
    for v in workload_ep.node_ids:
        tau = ready_time_ms.get(v)
        tc = commit_time_ms.get(v)
        cv = completed_ms.get(v)
        allocation_delays[v] = (tc - tau) if (tau is not None and tc is not None) else None
        if tau is not None and cv is not None:
            rv = cv - tau
            response_times[v] = rv
            d_v = absolute_deadline_ms.get(v)
            if d_v is not None and tc is not None:
                rt_bound = 2.0 * H_ms + T_auc_bar_ms + (d_v - tc)
                if rv > rt_bound + 1.0:
                    rt_violations += 1
                if log_per_commit_enabled:
                    agent_idx_local = task_assigned_to.get(v)
                    per_commit_log.append({
                        "episode_idx": int(episode_idx) if episode_idx is not None else -1,
                        "node_id": v,
                        "agent_idx": int(agent_idx_local) if agent_idx_local is not None else -1,
                        "tau_ms": float(tau),
                        "tc_ms": float(tc),
                        "cv_ms": float(cv),
                        "d_v_ms": float(d_v),
                        "rv_ms": float(rv),
                        "rt_bound_ms": float(rt_bound),
                    })
        else:
            response_times[v] = None
    last_info["response_times_ms"] = response_times
    last_info["allocation_delays_ms"] = allocation_delays
    last_info["max_response_time_ms"] = max(
        (x for x in response_times.values() if x is not None), default=None
    )
    last_info["max_allocation_delay_ms"] = max(
        (x for x in allocation_delays.values() if x is not None), default=None
    )
    last_info["rt_bound_violations"] = rt_violations
    # Eq. 44 instrumentation: per-epoch auction wall-clock + ready frontier
    last_info["t_auc_per_epoch_ms"] = list(t_auc_per_epoch_ms)
    last_info["A_k_per_epoch"] = list(A_k_per_epoch)
    last_info["max_t_auc_ms"] = max(t_auc_per_epoch_ms) if t_auc_per_epoch_ms else 0.0
    last_info["mean_t_auc_ms"] = (
        float(sum(t_auc_per_epoch_ms) / len(t_auc_per_epoch_ms))
        if t_auc_per_epoch_ms else 0.0
    )
    last_info["max_A_k"] = max(A_k_per_epoch) if A_k_per_epoch else 0
    last_info["mean_A_k"] = (
        float(sum(A_k_per_epoch) / len(A_k_per_epoch))
        if A_k_per_epoch else 0.0
    )
    last_info["delta_auc_samples_ms"] = list(delta_auc_samples_ms)
    last_info["max_delta_auc_ms"] = max(delta_auc_samples_ms) if delta_auc_samples_ms else 0.0
    last_info["mean_delta_auc_ms"] = (
        float(sum(delta_auc_samples_ms) / len(delta_auc_samples_ms))
        if delta_auc_samples_ms else 0.0
    )
    # ----- Eq. 44 linearity instrumentation: bucket by |A_k| -----
    # Buckets: A_k=1, A_k=2, A_k=3, A_k>=4. For each bucket, record
    # mean T_auc(k) over epochs in that bucket, plus the epoch count.
    _bucket: Dict[int, List[float]] = {1: [], 2: [], 3: [], 4: []}
    for _t_auc, _a_k in zip(t_auc_per_epoch_ms, A_k_per_epoch):
        _key = 4 if _a_k >= 4 else max(1, int(_a_k))
        _bucket[_key].append(float(_t_auc))
    last_info["t_auc_Ak1_mean_ms"] = (
        float(sum(_bucket[1]) / len(_bucket[1])) if _bucket[1] else float("nan"))
    last_info["t_auc_Ak2_mean_ms"] = (
        float(sum(_bucket[2]) / len(_bucket[2])) if _bucket[2] else float("nan"))
    last_info["t_auc_Ak3_mean_ms"] = (
        float(sum(_bucket[3]) / len(_bucket[3])) if _bucket[3] else float("nan"))
    last_info["t_auc_Ak4plus_mean_ms"] = (
        float(sum(_bucket[4]) / len(_bucket[4])) if _bucket[4] else float("nan"))
    last_info["n_epochs_Ak1"] = len(_bucket[1])
    last_info["n_epochs_Ak2"] = len(_bucket[2])
    last_info["n_epochs_Ak3"] = len(_bucket[3])
    last_info["n_epochs_Ak4plus"] = len(_bucket[4])
    n_pending = (
        V_total_episode
        - n_full_fidelity
        - n_degradable_fallback
        - n_nondegradable_n1
        - n_nondegradable_n2
    )
    last_info["n_full_fidelity"] = n_full_fidelity
    last_info["n_degradable_fallback"] = n_degradable_fallback
    last_info["n_nondegradable_n1"] = n_nondegradable_n1
    last_info["n_nondegradable_n2"] = n_nondegradable_n2
    last_info["n_pending"] = n_pending
    completed_ms = getattr(env, "completed_tasks_ms", {}) or {}
    n_miss_n1_drop = n_nondegradable_n1
    n_miss_n2_fallback_insufficient = n_nondegradable_n2
    n_miss_n3_unreached = sum(
        1
        for j, o in task_outcome_dict.items()
        if o in ("full_fidelity", "fallback_now") and j not in completed_ms
    )
    n_miss_full_fidelity_late = sum(
        1
        for j, o in task_outcome_dict.items()
        if o == "full_fidelity"
        and j in completed_ms
        and j in absolute_deadline_ms
        and completed_ms[j] > absolute_deadline_ms[j]
    )
    last_info["n_miss_n1_drop"] = n_miss_n1_drop
    last_info["n_miss_n2_fallback_insufficient"] = n_miss_n2_fallback_insufficient
    last_info["n_miss_n3_unreached"] = n_miss_n3_unreached
    last_info["n_miss_full_fidelity_late"] = n_miss_full_fidelity_late
    last_info["n_defer_triggers"] = n_defer_triggers
    last_info["n_bids_evaluated"] = n_bids_evaluated
    last_info["n_bids_cert_pass"] = n_bids_cert_pass
    last_info["mean_pool_size"] = (
        float(sum(pool_sizes) / len(pool_sizes)) if pool_sizes else float("nan")
    )
    last_info["episode_span_ms"] = episode_span_ms
    last_info["exec_ms_per_agent"] = list(exec_ms_per_agent)
    last_info["travel_ms_per_agent"] = list(travel_ms_per_agent)

    cov_stats = coverage_tracker.freeze()
    last_info["rho_hat"] = float(cov_stats.rho_hat)
    last_info["n_total_jobs"] = int(cov_stats.V_total)
    last_info["rho_sp_per_agent"] = json.dumps(
        [float(cov_stats.rho_sp_i(i)) for i in range(len(agents))],
        separators=(",", ":"),
    )
    last_info["omega_per_agent"] = json.dumps(
        [float(cov_stats.omega_i(i)) for i in range(len(agents))],
        separators=(",", ":"),
    )
    # TB-1 certified-miss numerator: jobs the auction admitted at full fidelity that then
    # finished late or never finished. V_ff holds auction FF commits only, so Eq. 23
    # fallback commits are excluded by construction.
    n_ff_missed = 0
    for _v_ff in cov_stats.V_ff:
        _c_v = completed_ms.get(_v_ff)
        _d_v = absolute_deadline_ms.get(_v_ff)
        if _c_v is None or (_d_v is not None and _c_v > _d_v):
            n_ff_missed += 1
    last_info["n_ff_missed"] = n_ff_missed
    if len(cov_stats.V_ff) != n_full_fidelity:
        episode_logger.warning(
            "CoverageTracker |V_ff|=%d vs auction counter n_full_fidelity=%d (expected match).",
            len(cov_stats.V_ff),
            n_full_fidelity,
        )
    _sum_alloc = n_full_fidelity + n_degradable_fallback + n_nondegradable_n1 + n_nondegradable_n2 + n_pending
    if _sum_alloc != V_total_episode:
        episode_logger.critical(
            "BOOKKEEPING VIOLATION: n_full_fidelity(%d)+n_degradable_fallback(%d)+n_nondegradable_n1(%d)"
            "+n_nondegradable_n2(%d)+n_pending(%d)=%d != n_total_jobs(%d).",
            n_full_fidelity,
            n_degradable_fallback,
            n_nondegradable_n1,
            n_nondegradable_n2,
            n_pending,
            _sum_alloc,
            V_total_episode,
        )
        episode_logger.critical(
            "Auction diagnostic dump: task_assigned_to=%s ready_commit_snapshot n_ff=%d n_df=%d",
            json.dumps(task_assigned_to, sort_keys=True),
            n_full_fidelity,
            n_degradable_fallback,
        )

    if log_per_commit_enabled and per_commit_log:
        last_info["per_commit_records"] = per_commit_log

    if subround_writer is not None and subround_rows:
        for _sr_row in subround_rows:
            _idx = _sr_row.pop("_sample_idx", None)
            _sr_row["t_subround_ms"] = (
                float(delta_auc_samples_ms[_idx])
                if _idx is not None and 0 <= _idx < len(delta_auc_samples_ms)
                else float("nan")
            )
        subround_writer.write_rows(subround_rows)

    if _exp2_prm is not None:
        from alg.defer_fallback import fallback_blocked_count
        _log_dir = get_exp2_runtime().log_path
        if _log_dir:
            _exp2_prm.write_sidecar(_log_dir, env, RELEASE_LOG_HEADER)
        episode_logger.info(
            "[EXP2][periodic] fallback_blocked_count=%d jobs=%s",
            fallback_blocked_count(),
            sorted(_exp2_prm._commits.keys()),
        )

    return episode_rewards, total_energy_J_per_agent, last_info, agent_died, energy_cert_violations, raw_env_reward_sum


EVAL_CSV_HEADER = [
    "episode",
    "A1_total_reward", "A1_remaining_battery_pct",
    "A2_total_reward", "A2_remaining_battery_pct",
    "A3_total_reward", "A3_remaining_battery_pct",
    "A4_total_reward", "A4_remaining_battery_pct",
    "subA_miss_rate", "subA_time",
    "subB_miss_rate", "subB_time",
    "subC_miss_rate", "subC_time",
    "subD_miss_rate", "subD_time",
    "task_miss_rate",
    "task_time",
    "rho_hat",
    "n_full_fidelity",
    "n_total_jobs",
    "rho_sp_A1", "rho_sp_A2", "rho_sp_A3", "rho_sp_A4",
    "omega_A1", "omega_A2", "omega_A3", "omega_A4",
    "A1_crash_rate", "A2_crash_rate", "A3_crash_rate", "A4_crash_rate",
    "subA_rt_ms", "subB_rt_ms", "subC_rt_ms", "subD_rt_ms",
    "max_rt_ms", "max_alloc_delay_ms", "rt_bound_violations",
    "max_t_auc_ms", "mean_t_auc_ms",
    "max_A_k", "mean_A_k",
    "max_delta_auc_ms", "mean_delta_auc_ms",
    "t_auc_Ak1_mean_ms", "t_auc_Ak2_mean_ms",
    "t_auc_Ak3_mean_ms", "t_auc_Ak4plus_mean_ms",
    "n_epochs_Ak1", "n_epochs_Ak2",
    "n_epochs_Ak3", "n_epochs_Ak4plus",
    "n_degradable_fallback", "n_nondegradable_n1", "n_nondegradable_n2", "n_pending",
    "n_miss_n1_drop", "n_miss_n2_fallback_insufficient", "n_miss_n3_unreached",
    "n_miss_full_fidelity_late",
    # --- TB-1 instrumentation, appended at END only (63 -> 76 columns) ---
    "n_bids_evaluated", "n_bids_cert_pass", "mean_pool_size",
    "n_ff_missed", "episode_span_ms",
    "exec_ms_A1", "exec_ms_A2", "exec_ms_A3", "exec_ms_A4",
    "travel_ms_A1", "travel_ms_A2", "travel_ms_A3", "travel_ms_A4",
]


def single_episode_eval_row_list(
    episode_idx: int,
    env: TurtleBot3Env,
    cfg: DARTS_CONFIG,
    episode_rewards: List[float],
    last_info: Dict[str, Any],
    energy_cert_violations: List[int],
) -> List[Any]:
    """One row aligned with EVAL_CSV_HEADER (distributed Jetson team_log / local CSV)."""
    n_agents = min(int(getattr(cfg, "num_agents", 4)), 4)
    global_deadline_ms = float(getattr(cfg, "global_deadline_ms", 180000.0))
    workload = runtime_workload(env, cfg)
    slots = csv_subtask_slots(workload)
    CSV_KEYS = ("A", "B", "C", "D")
    st = last_info.get("subtask_completion_times_ms") or {}
    battery_wh = last_info.get("battery_wh", [0.0] * n_agents)
    row: List[Any] = [episode_idx]
    for i in range(4):
        if i < len(episode_rewards) and i < n_agents:
            row.append(episode_rewards[i])
            init_wh = env.initial_battery_wh[i] if i < len(env.initial_battery_wh) else getattr(env, "battery_capacity_wh", 50.0)
            curr_wh = battery_wh[i] if i < len(battery_wh) else 0.0
            pct = (curr_wh / init_wh * 100.0) if init_wh > 0 else 0.0
            row.append(pct)
        else:
            row.append(0.0)
            row.append(float("nan"))
    for k in range(4):
        nid = slots[k]
        t = st.get(nid) if nid is not None else None
        row.append(1.0 if t is None else 0.0)
        row.append(float(t) if t is not None else float("nan"))
    task_time = csv_task_time_ms(last_info)
    if task_time is None:
        row.append(1.0)
    else:
        task_time_f = float(task_time)
        row.append(1.0 if task_time_f > global_deadline_ms else 0.0)
    if task_time is None:
        row.append(float("nan"))
    else:
        row.append(float(task_time))
    row.append(float(last_info.get("rho_hat", float("nan"))))
    row.append(float(last_info.get("n_full_fidelity", 0)))
    row.append(int(last_info.get("n_total_jobs", 0)))
    try:
        rho_sp_list = json.loads(last_info.get("rho_sp_per_agent") or "[]")
    except (json.JSONDecodeError, TypeError, ValueError):
        rho_sp_list = []
    try:
        omega_list = json.loads(last_info.get("omega_per_agent") or "[]")
    except (json.JSONDecodeError, TypeError, ValueError):
        omega_list = []
    for i in range(4):
        row.append(float(rho_sp_list[i]) if i < len(rho_sp_list) else float("nan"))
    for i in range(4):
        row.append(float(omega_list[i]) if i < len(omega_list) else float("nan"))
    for i in range(4):
        row.append(100.0 if (i < len(energy_cert_violations) and energy_cert_violations[i] > 0) else 0.0)
    rts = last_info.get("response_times_ms", {})
    for k in range(4):
        nid = slots[k]
        v = rts.get(nid) if nid is not None else None
        row.append(float(v) if v is not None else float("nan"))
    max_rt = last_info.get("max_response_time_ms")
    row.append(float(max_rt) if max_rt is not None else float("nan"))
    max_ad = last_info.get("max_allocation_delay_ms")
    row.append(float(max_ad) if max_ad is not None else float("nan"))
    row.append(last_info.get("rt_bound_violations", 0))
    row.append(float(last_info.get("max_t_auc_ms", float("nan"))))
    row.append(float(last_info.get("mean_t_auc_ms", float("nan"))))
    row.append(int(last_info.get("max_A_k", 0)))
    row.append(float(last_info.get("mean_A_k", float("nan"))))
    row.append(float(last_info.get("max_delta_auc_ms", float("nan"))))
    row.append(float(last_info.get("mean_delta_auc_ms", float("nan"))))
    row.append(float(last_info.get("t_auc_Ak1_mean_ms", float("nan"))))
    row.append(float(last_info.get("t_auc_Ak2_mean_ms", float("nan"))))
    row.append(float(last_info.get("t_auc_Ak3_mean_ms", float("nan"))))
    row.append(float(last_info.get("t_auc_Ak4plus_mean_ms", float("nan"))))
    row.append(int(last_info.get("n_epochs_Ak1", 0)))
    row.append(int(last_info.get("n_epochs_Ak2", 0)))
    row.append(int(last_info.get("n_epochs_Ak3", 0)))
    row.append(int(last_info.get("n_epochs_Ak4plus", 0)))
    row.append(float(last_info.get("n_degradable_fallback", 0)))
    row.append(float(last_info.get("n_nondegradable_n1", 0)))
    row.append(float(last_info.get("n_nondegradable_n2", 0)))
    row.append(float(last_info.get("n_pending", 0)))
    row.append(float(last_info.get("n_miss_n1_drop", 0)))
    row.append(float(last_info.get("n_miss_n2_fallback_insufficient", 0)))
    row.append(float(last_info.get("n_miss_n3_unreached", 0)))
    row.append(float(last_info.get("n_miss_full_fidelity_late", 0)))
    # --- TB-1 columns 64..76 (nan where the episode did not supply the field) ---
    row.append(float(last_info.get("n_bids_evaluated", float("nan"))))
    row.append(float(last_info.get("n_bids_cert_pass", float("nan"))))
    row.append(float(last_info.get("mean_pool_size", float("nan"))))
    row.append(float(last_info.get("n_ff_missed", float("nan"))))
    row.append(float(last_info.get("episode_span_ms", float("nan"))))
    _exec_ms = last_info.get("exec_ms_per_agent") or []
    _travel_ms = last_info.get("travel_ms_per_agent") or []
    for i in range(4):
        row.append(float(_exec_ms[i]) if i < len(_exec_ms) else float("nan"))
    for i in range(4):
        row.append(float(_travel_ms[i]) if i < len(_travel_ms) else float("nan"))
    return row


def evaluate_agents(
    env: TurtleBot3Env,
    agents: List[DartsMARLAgent],
    cfg: DARTS_CONFIG,
    num_episodes: int = 10,
    training_episode: Optional[int] = None,
    episode_rng_seed_base: Optional[int] = None,
    subround_writer: Optional["SubroundLogWriter"] = None,
) -> Dict[str, Any]:
    """
    Run num_episodes test episodes with deterministic (no exploration) actions.
    Tracks per-agent remaining battery %, per-agent crash rate, subtask/task times and miss rates.
    Returns a dict of metrics for darts_log.csv (numeric or nan; CSV writer maps nan to 'N/A').
    ``episode_rng_seed_base``: when set, test episode t resets the env with seed base + t.
    Leaving it None keeps the legacy unseeded behaviour byte-for-byte.
    """
    n_agents = len(agents)
    workload = runtime_workload(env, cfg)
    slots = csv_subtask_slots(workload)
    CSV_KEYS = ("A", "B", "C", "D")
    rewards_per_agent: List[List[float]] = [[] for _ in range(n_agents)]
    battery_pct_per_agent: List[List[float]] = [[] for _ in range(n_agents)]
    sub_times: Dict[str, List[float]] = {"A": [], "B": [], "C": [], "D": []}
    sub_misses: Dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0}
    task_times: List[float] = []
    task_miss_count = 0
    crash_counts: List[int] = [0] * n_agents
    rt_per_sub: Dict[str, List[float]] = {"A": [], "B": [], "C": [], "D": []}
    max_rt_list: List[float] = []
    max_alloc_list: List[float] = []
    max_t_auc_list: List[float] = []
    mean_t_auc_list: List[float] = []
    max_A_k_list: List[float] = []
    mean_A_k_list: List[float] = []
    max_delta_auc_list: List[float] = []
    mean_delta_auc_list: List[float] = []
    t_auc_Ak1_mean_list: List[float] = []
    t_auc_Ak2_mean_list: List[float] = []
    t_auc_Ak3_mean_list: List[float] = []
    t_auc_Ak4plus_mean_list: List[float] = []
    n_epochs_Ak1_list: List[int] = []
    n_epochs_Ak2_list: List[int] = []
    n_epochs_Ak3_list: List[int] = []
    n_epochs_Ak4plus_list: List[int] = []
    rt_violation_count: int = 0
    global_deadline_ms = getattr(cfg, "global_deadline_ms", 120000.0)
    episode_logger = logging.getLogger(EPISODE_EVENTS_LOGGER)
    n_ff_list: List[float] = []
    n_df_list: List[float] = []
    n_n1_list: List[float] = []
    n_n2_list: List[float] = []
    n_pend_list: List[float] = []
    n_miss_n1_list: List[float] = []
    n_miss_n2_list: List[float] = []
    n_miss_n3_list: List[float] = []
    n_miss_ffl_list: List[float] = []
    rho_hat_list: List[float] = []
    n_tot_jobs_list: List[int] = []
    rho_sp_rows: List[List[float]] = []
    omega_rows: List[List[float]] = []
    # --- TB-1 instrumentation, averaged over eval episodes like n_ff / n_df ---
    n_bids_eval_list: List[float] = []
    n_bids_pass_list: List[float] = []
    mean_pool_list: List[float] = []
    n_ff_missed_list: List[float] = []
    episode_span_list: List[float] = []
    exec_ms_rows: List[List[float]] = []
    travel_ms_rows: List[List[float]] = []

    set_episode_logging_verbose(True)  # Show in-episode events during evaluation
    for test_ep in range(num_episodes):
        episode_logger.info(
            "========== Test Episode %d/%d (Model from Training Ep %s) ==========",
            test_ep + 1, num_episodes, training_episode if training_episode is not None else "?",
        )
        episode_rewards, total_energy_J, last_info, agent_died, energy_cert_violations, _raw_env = run_episode(
            env, agents, cfg, explore=False, store_transitions=False, episode_verbose=True,
            episode_rng_seed=(None if episode_rng_seed_base is None else episode_rng_seed_base + test_ep),
            episode_idx=test_ep,
            subround_writer=subround_writer,
            subround_phase="EVAL",
        )
        for i in range(n_agents):
            if energy_cert_violations[i] > 0:
                crash_counts[i] += 1
            rewards_per_agent[i].append(episode_rewards[i])
            init_wh = env.initial_battery_wh[i] if i < len(env.initial_battery_wh) else getattr(env, "battery_capacity_wh", 50.0)
            curr_wh = last_info.get("battery_wh", [0.0] * n_agents)[i] if i < len(last_info.get("battery_wh", [])) else 0.0
            pct = (curr_wh / init_wh * 100.0) if init_wh > 0 else 0.0
            battery_pct_per_agent[i].append(pct)
        st = last_info.get("subtask_completion_times_ms") or {}
        for k in range(4):
            ck = CSV_KEYS[k]
            nid = slots[k]
            t = st.get(nid) if nid is not None else None
            if t is None:
                sub_misses[ck] += 1
            else:
                sub_times[ck].append(float(t))
        task_time = csv_task_time_ms(last_info)
        if task_time is None:
            task_miss_count += 1
        else:
            task_time_f = float(task_time)
            task_times.append(task_time_f)
            if task_time_f > global_deadline_ms:
                task_miss_count += 1
        rts = last_info.get("response_times_ms", {})
        for k in range(4):
            ck = CSV_KEYS[k]
            nid = slots[k]
            v = rts.get(nid) if nid is not None else None
            if v is not None:
                rt_per_sub[ck].append(v)
        max_rt = last_info.get("max_response_time_ms")
        if max_rt is not None:
            max_rt_list.append(max_rt)
        max_ad = last_info.get("max_allocation_delay_ms")
        if max_ad is not None:
            max_alloc_list.append(max_ad)
        max_t_auc_list.append(float(last_info.get("max_t_auc_ms", float("nan"))))
        mean_t_auc_list.append(float(last_info.get("mean_t_auc_ms", float("nan"))))
        max_A_k_list.append(float(last_info.get("max_A_k", float("nan"))))
        mean_A_k_list.append(float(last_info.get("mean_A_k", float("nan"))))
        max_delta_auc_list.append(float(last_info.get("max_delta_auc_ms", float("nan"))))
        mean_delta_auc_list.append(float(last_info.get("mean_delta_auc_ms", float("nan"))))
        t_auc_Ak1_mean_list.append(float(last_info.get("t_auc_Ak1_mean_ms", float("nan"))))
        t_auc_Ak2_mean_list.append(float(last_info.get("t_auc_Ak2_mean_ms", float("nan"))))
        t_auc_Ak3_mean_list.append(float(last_info.get("t_auc_Ak3_mean_ms", float("nan"))))
        t_auc_Ak4plus_mean_list.append(float(last_info.get("t_auc_Ak4plus_mean_ms", float("nan"))))
        n_epochs_Ak1_list.append(int(last_info.get("n_epochs_Ak1", 0)))
        n_epochs_Ak2_list.append(int(last_info.get("n_epochs_Ak2", 0)))
        n_epochs_Ak3_list.append(int(last_info.get("n_epochs_Ak3", 0)))
        n_epochs_Ak4plus_list.append(int(last_info.get("n_epochs_Ak4plus", 0)))
        rt_violation_count += last_info.get("rt_bound_violations", 0)
        n_ff_list.append(float(last_info.get("n_full_fidelity", 0)))
        n_df_list.append(float(last_info.get("n_degradable_fallback", 0)))
        n_n1_list.append(float(last_info.get("n_nondegradable_n1", 0)))
        n_n2_list.append(float(last_info.get("n_nondegradable_n2", 0)))
        n_pend_list.append(float(last_info.get("n_pending", 0)))
        n_miss_n1_list.append(float(last_info.get("n_miss_n1_drop", 0)))
        n_miss_n2_list.append(float(last_info.get("n_miss_n2_fallback_insufficient", 0)))
        n_miss_n3_list.append(float(last_info.get("n_miss_n3_unreached", 0)))
        n_miss_ffl_list.append(float(last_info.get("n_miss_full_fidelity_late", 0)))
        rho_hat_list.append(float(last_info.get("rho_hat", float("nan"))))
        n_tot_jobs_list.append(int(last_info.get("n_total_jobs", 0)))
        n_bids_eval_list.append(float(last_info.get("n_bids_evaluated", 0)))
        n_bids_pass_list.append(float(last_info.get("n_bids_cert_pass", 0)))
        mean_pool_list.append(float(last_info.get("mean_pool_size", float("nan"))))
        n_ff_missed_list.append(float(last_info.get("n_ff_missed", 0)))
        episode_span_list.append(float(last_info.get("episode_span_ms", float("nan"))))
        exec_ms_rows.append([float(x) for x in (last_info.get("exec_ms_per_agent") or [])])
        travel_ms_rows.append([float(x) for x in (last_info.get("travel_ms_per_agent") or [])])
        try:
            rho_sp_rows.append(json.loads(last_info.get("rho_sp_per_agent") or "[]"))
        except (json.JSONDecodeError, TypeError, ValueError):
            rho_sp_rows.append([])
        try:
            omega_rows.append(json.loads(last_info.get("omega_per_agent") or "[]"))
        except (json.JSONDecodeError, TypeError, ValueError):
            omega_rows.append([])

    num_ep = float(num_episodes)
    row: Dict[str, Any] = {}
    # crash_rate: fraction of episodes where agent received energy-infeasible assignment (Eq. energy_feasible); DARTS cert-mask should yield 0%, SADCHER/CEA-EDF may yield >0%.
    for i in range(4):
        key_r = "A{}_total_reward".format(i + 1)
        key_b = "A{}_remaining_battery_pct".format(i + 1)
        key_c = "A{}_crash_rate".format(i + 1)
        if i < n_agents:
            row[key_r] = np.mean(rewards_per_agent[i])
            row[key_b] = np.mean(battery_pct_per_agent[i]) if battery_pct_per_agent[i] else float("nan")
            row[key_c] = 100.0 * crash_counts[i] / num_episodes
        else:
            row[key_r] = 0.0
            row[key_b] = float("nan")
            row[key_c] = 0.0
    for ck in CSV_KEYS:
        row["sub{}_miss_rate".format(ck)] = sub_misses[ck] / num_ep
        times = sub_times[ck]
        row["sub{}_time".format(ck)] = np.mean(times) if times else float("nan")
    row["task_miss_rate"] = task_miss_count / num_ep
    row["rho_hat"] = float(np.nanmean(rho_hat_list)) if rho_hat_list else float("nan")
    row["n_full_fidelity"] = float(np.mean(n_ff_list)) if n_ff_list else float("nan")
    row["n_total_jobs"] = int(n_tot_jobs_list[0]) if n_tot_jobs_list else 0

    def _mean_vec(rows: List[List[float]], dim: int) -> List[float]:
        out: List[float] = []
        for i in range(dim):
            vals = [float(r[i]) for r in rows if i < len(r)]
            out.append(float(np.mean(vals)) if vals else float("nan"))
        return out

    rho_sp_list = _mean_vec(rho_sp_rows, 4) if rho_sp_rows else []
    omega_list = _mean_vec(omega_rows, 4) if omega_rows else []
    for i in range(4):
        row[f"rho_sp_A{i+1}"] = float(rho_sp_list[i]) if i < len(rho_sp_list) else float("nan")
        row[f"omega_A{i+1}"] = float(omega_list[i]) if i < len(omega_list) else float("nan")
    row["task_time"] = np.mean(task_times) if task_times else float("nan")
    for ck in CSV_KEYS:
        vals = rt_per_sub[ck]
        row["sub{}_rt_ms".format(ck)] = float(np.mean(vals)) if vals else float("nan")
    row["max_rt_ms"] = float(np.mean(max_rt_list)) if max_rt_list else float("nan")
    row["max_alloc_delay_ms"] = float(np.nanmean(max_alloc_list)) if max_alloc_list else float("nan")
    row["rt_bound_violations"] = rt_violation_count
    row["max_t_auc_ms"] = float(np.nanmean(max_t_auc_list)) if max_t_auc_list else float("nan")
    row["mean_t_auc_ms"] = float(np.nanmean(mean_t_auc_list)) if mean_t_auc_list else float("nan")
    row["max_A_k"] = float(np.nanmean(max_A_k_list)) if max_A_k_list else float("nan")
    row["mean_A_k"] = float(np.nanmean(mean_A_k_list)) if mean_A_k_list else float("nan")
    row["max_delta_auc_ms"] = float(np.nanmean(max_delta_auc_list)) if max_delta_auc_list else float("nan")
    row["mean_delta_auc_ms"] = float(np.nanmean(mean_delta_auc_list)) if mean_delta_auc_list else float("nan")
    def _nanmean_fin(vals: List[float]) -> float:
        if not vals:
            return float("nan")
        arr = np.asarray(vals, dtype=np.float64)
        return float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else float("nan")

    row["t_auc_Ak1_mean_ms"] = _nanmean_fin(t_auc_Ak1_mean_list)
    row["t_auc_Ak2_mean_ms"] = _nanmean_fin(t_auc_Ak2_mean_list)
    row["t_auc_Ak3_mean_ms"] = _nanmean_fin(t_auc_Ak3_mean_list)
    row["t_auc_Ak4plus_mean_ms"] = _nanmean_fin(t_auc_Ak4plus_mean_list)
    row["n_epochs_Ak1"] = int(np.sum(n_epochs_Ak1_list)) if n_epochs_Ak1_list else 0
    row["n_epochs_Ak2"] = int(np.sum(n_epochs_Ak2_list)) if n_epochs_Ak2_list else 0
    row["n_epochs_Ak3"] = int(np.sum(n_epochs_Ak3_list)) if n_epochs_Ak3_list else 0
    row["n_epochs_Ak4plus"] = int(np.sum(n_epochs_Ak4plus_list)) if n_epochs_Ak4plus_list else 0
    row["n_degradable_fallback"] = float(np.mean(n_df_list)) if n_df_list else float("nan")
    row["n_nondegradable_n1"] = float(np.mean(n_n1_list)) if n_n1_list else float("nan")
    row["n_nondegradable_n2"] = float(np.mean(n_n2_list)) if n_n2_list else float("nan")
    row["n_pending"] = float(np.mean(n_pend_list)) if n_pend_list else float("nan")
    row["n_miss_n1_drop"] = float(np.mean(n_miss_n1_list)) if n_miss_n1_list else float("nan")
    row["n_miss_n2_fallback_insufficient"] = (
        float(np.mean(n_miss_n2_list)) if n_miss_n2_list else float("nan")
    )
    row["n_miss_n3_unreached"] = float(np.mean(n_miss_n3_list)) if n_miss_n3_list else float("nan")
    row["n_miss_full_fidelity_late"] = (
        float(np.mean(n_miss_ffl_list)) if n_miss_ffl_list else float("nan")
    )
    # --- TB-1 columns 64..76 ---
    row["n_bids_evaluated"] = float(np.mean(n_bids_eval_list)) if n_bids_eval_list else float("nan")
    row["n_bids_cert_pass"] = float(np.mean(n_bids_pass_list)) if n_bids_pass_list else float("nan")
    row["mean_pool_size"] = _nanmean_fin(mean_pool_list)
    row["n_ff_missed"] = float(np.mean(n_ff_missed_list)) if n_ff_missed_list else float("nan")
    row["episode_span_ms"] = _nanmean_fin(episode_span_list)
    _exec_mean = _mean_vec(exec_ms_rows, 4) if exec_ms_rows else []
    _travel_mean = _mean_vec(travel_ms_rows, 4) if travel_ms_rows else []
    for i in range(4):
        row[f"exec_ms_A{i+1}"] = float(_exec_mean[i]) if i < len(_exec_mean) else float("nan")
        row[f"travel_ms_A{i+1}"] = float(_travel_mean[i]) if i < len(_travel_mean) else float("nan")
    return row


def append_eval_to_csv(log_path: str, episode: int, metrics: Dict[str, Any]) -> None:
    """Create darts_log.csv with EVAL_CSV_HEADER if missing; append one row. None/nan written as 'N/A'."""
    log_file = os.path.join(log_path, "darts_log.csv")
    file_exists = os.path.isfile(log_file)
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(EVAL_CSV_HEADER)
        row = [episode]
        for k in EVAL_CSV_HEADER[1:]:
            v = metrics.get(k, float("nan"))
            if v is None or (isinstance(v, float) and np.isnan(v)):
                row.append("N/A")
            else:
                row.append(v)
        writer.writerow(row)


class SubroundLogWriter:
    """Append-only sidecar writer for ``<exp_dir>/subround_log.csv``.

    The header comes from ``scripts.csv_schema_tb.SUBROUND_LOG_HEADER``, so the file on
    disk cannot drift from the registered schema. Rows are buffered by ``run_episode`` and
    handed over one episode at a time.
    """

    def __init__(self, exp_dir: str) -> None:
        self.path = os.path.join(exp_dir, "subround_log.csv")
        self.n_rows = 0
        if not os.path.isfile(self.path):
            with open(self.path, "a", newline="") as f:
                csv.writer(f).writerow(SUBROUND_LOG_HEADER)

    def write_rows(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            for r in rows:
                writer.writerow([r.get(k, "") for k in SUBROUND_LOG_HEADER])
        self.n_rows += len(rows)


def _available_ckpt_epochs(ckpt_dir: str, num_agents: int) -> List[int]:
    """Epochs for which agent_1..N_ep*.pth all exist in ckpt_dir, ascending."""
    per_agent: List[set] = []
    for i in range(1, num_agents + 1):
        found = set()
        for path in glob.glob(os.path.join(ckpt_dir, f"agent_{i}_ep*.pth")):
            stem = os.path.basename(path)[len(f"agent_{i}_ep"):-len(".pth")]
            if stem.isdigit():
                found.add(int(stem))
        per_agent.append(found)
    return sorted(set.intersection(*per_agent)) if per_agent else []


def _resolve_eval_checkpoints(
    args: argparse.Namespace, num_agents: int, logger: logging.Logger
) -> Tuple[List[str], int]:
    """Resolve the frozen-policy checkpoint set. Hard-fails with available epochs on any miss."""
    if not args.load_ckpt:
        raise SystemExit("[eval_only] --load_ckpt is required with --eval_only.")
    ckpt_dir = os.path.abspath(args.load_ckpt)
    if not os.path.isdir(ckpt_dir):
        raise SystemExit(f"[eval_only] --load_ckpt is not a directory: {ckpt_dir}")

    available = _available_ckpt_epochs(ckpt_dir, num_agents)
    if not available:
        raise SystemExit(
            f"[eval_only] no complete {num_agents}-agent checkpoint set in {ckpt_dir}. "
            f"Expected agent_1..{num_agents}_ep<EP>.pth."
        )

    requested = int(args.load_ckpt_ep)
    if requested == -1:
        pinned = available[-1]
        logger.warning(
            "[eval_only] UNPINNED checkpoint load: --load_ckpt_ep=-1 selected the highest "
            "available epoch %d out of %s. This is UNSAFE (stale-epoch hazard): a directory can "
            "hold checkpoints from different training eras, so the highest epoch is not "
            "necessarily the paper-era one. Pin the epoch explicitly for TB sweeps.",
            pinned, available,
        )
    else:
        pinned = requested
        if pinned not in available:
            raise SystemExit(
                f"[eval_only] requested epoch {pinned} has no complete checkpoint set in "
                f"{ckpt_dir}. Available epochs: {available}"
            )

    paths = [os.path.join(ckpt_dir, f"agent_{i + 1}_ep{pinned}.pth") for i in range(num_agents)]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise SystemExit(
            f"[eval_only] missing checkpoint file(s): {missing}. Available epochs: {available}"
        )
    return paths, pinned


def _assert_ckpt_incentive_dim(path: str, num_agents: int, device: Any) -> int:
    """Hard-fail if the saved incentive_input_dim != num_agents + 1.

    DartsMARLAgent.load_checkpoint only *warns* on a dim mismatch and silently keeps the random
    init, which would mean evaluating an untrained φ. That is acceptable when resuming training
    but never in frozen-policy evaluation, so the mismatch is fatal here.
    """
    ckpt = torch.load(path, map_location=device)
    expected = num_agents + 1
    saved = ckpt.get("incentive_input_dim")
    if saved is None:
        raise SystemExit(
            f"[eval_only] {path} has no incentive_input_dim (pre-χ^p_s checkpoint format). "
            f"load_checkpoint would keep a RANDOM incentive_net; refusing to evaluate."
        )
    if int(saved) != expected:
        raise SystemExit(
            f"[eval_only] {path} incentive_input_dim={int(saved)} but this run needs "
            f"{expected} (= num_agents + 1 = {num_agents} + 1). load_checkpoint would silently "
            f"keep a RANDOM incentive_net; refusing to evaluate."
        )
    return int(saved)


def main() -> None:
    parser = argparse.ArgumentParser(description="DARTS Algorithm 2: Outer-Loop Training (REFiNE-style).")
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name (results/<exp_name>/).")
    parser.add_argument("exp_name_pos", nargs="*", help="Experiment name as positional (e.g. exp 1).")
    parser.add_argument("--debug_auction", action="store_true", help="Set episode_events logger to DEBUG (Sponsor Bias bid details on summary episodes).")
    parser.add_argument(
        "--workload",
        type=str,
        default=None,
        choices=["B1_CDR", "B2_LML", "B3_IQI", "B4_DEM"],
        help="Benchmark workload ID (overrides cfg.workload_id).",
    )
    parser.add_argument(
        "--num_agents",
        type=int,
        default=None,
        help="Override cfg.num_agents (default 4). Used for agent-count scaling sweep.",
    )
    parser.add_argument("--global_deadline_ms", type=float, default=None, help="Override cfg.global_deadline_ms.")
    parser.add_argument(
        "--relative_deadline_ms",
        type=float,
        default=None,
        help="Override workload.relative_deadline_ms (per-chain/per-job deadline).",
    )
    parser.add_argument(
        "--battery_capacities_wh",
        type=str,
        default=None,
        help="Comma-separated Wh per agent (overrides cfg.battery_capacities_wh).",
    )
    parser.add_argument("--max_episode_steps", type=int, default=None, help="Override cfg.max_episode_steps.")
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Override training loop episode count (debug-friendly alias for cfg.train_episodes).",
    )
    parser.add_argument(
        "--log_per_commit",
        action="store_true",
        help="Log per-commit (v, agent, tau, tc, cv, d_v, rv, rt_bound) to per_commit.jsonl for §V.G T2",
    )
    parser.add_argument("--max_work_ms", type=float, default=None, help="Override cfg.max_work_ms.")
    parser.add_argument(
        "--poisson_mean_ms",
        type=float,
        default=None,
        help="Override workload release_params.poisson_mean_ms (B2_LML only).",
    )
    parser.add_argument(
        "--poisson_max_jobs",
        type=int,
        default=None,
        help="Override workload release_params.max_jobs (B2_LML only).",
    )
    parser.add_argument("--eval_every", type=int, default=None, help="Override cfg.eval_every.")
    parser.add_argument("--save_interval", type=int, default=None, help="Override cfg.save_interval.")
    parser.add_argument("--eval_episodes", type=int, default=None, help="Override evaluation episode count.")
    parser.add_argument("--random_seed", type=int, default=None, help="Override cfg.random_seed.")
    parser.add_argument("--lr_eta", type=float, default=None, help="Override cfg.lr_eta (Theorem 1 choice (1)).")
    parser.add_argument("--lr_phi", type=float, default=None, help="Override cfg.lr_phi (Theorem 1 choice (1)).")
    parser.add_argument("--lambda_rho", type=float, default=None, help="Override cfg.lambda_rho.")
    parser.add_argument(
        "--lipschitz_rho_estimate",
        type=float,
        default=None,
        help="Override cfg.lipschitz_rho_estimate (L_rho).",
    )
    parser.add_argument(
        "--gradient_clip_norm",
        type=float,
        default=None,
        help="Override cfg.gradient_clip_norm (G_R, G_P cap).",
    )
    parser.add_argument(
        "--deployment_mode",
        type=str,
        default=None,
        choices=["simulator", "distributed_jetson"],
        help="Override cfg.deployment_mode (training only supports simulator).",
    )
    parser.add_argument(
        "--local_agent_id",
        type=str,
        default=None,
        help="Override cfg.local_agent_id (for distributed_jetson agent_node; not used in train_darts).",
    )
    parser.add_argument(
        "--disable_feasibility_mask",
        action="store_true",
        help="ABLATION ONLY: disable schedulability certificate (DARTS-NoCert). "
        "Forces cert_seq=1 for all (agent, job) pairs. Use only for paper ablations.",
    )
    parser.add_argument(
        "--bilevel_mode",
        choices=["legacy", "fixed_eta"],
        default="legacy",
        help=(
            "legacy: Traj B rolled under post-update η̂^m (current behavior). "
            "fixed_eta: Traj B rolled under pre-update η^m (Path 1, paper-aligned with Thm 1(iv))."
        ),
    )
    parser.add_argument(
        "--no_incentive",
        action="store_true",
        default=False,
        help="Freeze ϕ at 0 throughout training (MC-7 NoIncentive ablation column).",
    )
    parser.add_argument(
        "--cert_conservatism",
        type=float,
        default=1.0,
        help=(
            "TB-1 κ: global scale on the envelopes read inside check_feasibility. "
            ">1 admits fewer jobs (more conservative certificate), <1 more. "
            "1.0 (default) is bit-identical to the unscaled certificate."
        ),
    )
    parser.add_argument(
        "--cert_conservatism_exec",
        type=float,
        default=1.0,
        help="TB-1: extra scale on the execution envelope inside check_feasibility (multiplies κ).",
    )
    parser.add_argument(
        "--cert_conservatism_travel",
        type=float,
        default=1.0,
        help="TB-1: extra scale on the travel envelope inside check_feasibility (multiplies κ).",
    )
    parser.add_argument(
        "--cert_conservatism_energy",
        type=float,
        default=1.0,
        help="TB-1: extra scale on the energy envelope inside check_feasibility (multiplies κ).",
    )
    parser.add_argument(
        "--deadline_scale",
        type=float,
        default=1.0,
        help=(
            "β: multiplies BOTH cfg.global_deadline_ms and workload.relative_deadline_ms "
            "AFTER any absolute overrides. TB-3 sweep axis."
        ),
    )
    parser.add_argument(
        "--instance_seed",
        type=int,
        default=None,
        help=(
            "TB-3 mission-instance generator: jitter per-job WCETs (x0.8-1.2), "
            "resample task locations in the workspace, draw per-agent initial "
            "battery within profile bounds. Deterministic private RNG; does not "
            "touch np/torch/env RNG streams. None = nominal benchmark instance."
        ),
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        default=False,
        help="Frozen-policy evaluation: skip ALL training, load checkpoints, run evaluate_agents once, exit.",
    )
    parser.add_argument(
        "--load_ckpt",
        type=str,
        default=None,
        help="Experiment dir containing agent_{1..N}_ep*.pth. Required with --eval_only.",
    )
    parser.add_argument(
        "--load_ckpt_ep",
        type=int,
        default=-1,
        help=(
            "Pin the checkpoint epoch. -1 uses the highest available and logs a warning "
            "(unpinned loading is unsafe: stale-epoch hazard). TB sweeps always pin."
        ),
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Override cfg.results_dir (output root; results/<exp_name> by default).",
    )
    parser.add_argument(
        "--eval_seed_base",
        type=int,
        default=None,
        help=(
            "Seed np.random and torch at startup and reset test episode t with base + t. "
            "Unset (default) keeps legacy unseeded behaviour byte-for-byte."
        ),
    )
    parser.add_argument(
        "--subround_log",
        action="store_true",
        default=False,
        help=(
            "Write the per-subround sidecar <exp_dir>/subround_log.csv "
            "(schema: scripts/csv_schema_tb.py). Off by default; no file is created."
        ),
    )
    parser.add_argument(
        "--release_mode",
        type=str,
        default="single",
        choices=["single", "periodic"],
        help="EXP-2: single-shot (default, inert) or periodic multi-release.",
    )
    parser.add_argument(
        "--release_period_ms",
        type=float,
        default=None,
        help="EXP-2: period P in ms. Required when --release_mode periodic; multiple of 1000.",
    )
    parser.add_argument(
        "--num_releases",
        type=int,
        default=1,
        help="EXP-2: number of releases L (default 1).",
    )
    parser.add_argument(
        "--alloc_map",
        type=str,
        default=None,
        help="EXP-2: JSON path mapping job class id -> agent id (e.g. {\"J1\": \"agent_1\"}).",
    )
    parser.add_argument(
        "--per_period_energy",
        type=str,
        default="none",
        choices=["none", "reset"],
        help="EXP-2: none (default) or reset battery to initial at each release boundary.",
    )
    parser.add_argument(
        "--stress",
        type=str,
        default=None,
        help="Optional stress profile name (light/medium/heavy/...). Applied before other CLI overrides.",
    )
    args = parser.parse_args()

    _exp2_alloc_map = None
    if getattr(args, "alloc_map", None):
        with open(args.alloc_map, "r") as _af:
            _exp2_alloc_map = json.load(_af)

    # TB-1: the whole training/eval loop runs under the certificate envelope
    # scales; they are restored on exit and touch nothing outside check_feasibility.
    with scaled_cert_envelopes(
        kappa=args.cert_conservatism,
        exec_scale=args.cert_conservatism_exec,
        travel_scale=args.cert_conservatism_travel,
        energy_scale=args.cert_conservatism_energy,
    ):
        with partitioned_alloc(_exp2_alloc_map):
            _run_training(args)


def _run_training(args: argparse.Namespace) -> None:
    """Training/eval driver for the already-parsed CLI arguments."""
    exp_name = " ".join(args.exp_name_pos) if args.exp_name_pos else args.exp_name

    cfg = DARTS_CONFIG()
    # --- Apply workload overrides from CLI ---
    if args.workload is not None:
        cfg = dataclasses.replace(cfg, workload_id=args.workload)
    if getattr(args, "stress", None):
        from alg.stress_profiles import get_stress_profile as _get_stress_profile
        _prof = _get_stress_profile(cfg.workload_id, str(args.stress))
        if _prof is None:
            raise SystemExit(
                f"--stress {args.stress!r} is not defined for workload {cfg.workload_id}"
            )
        _cfg_kw = {}
        for _k in ("global_deadline_ms", "max_episode_steps", "max_work_ms"):
            if _k in _prof:
                _cfg_kw[_k] = _prof[_k]
        if "battery_capacities_wh" in _prof:
            _cfg_kw["battery_capacities_wh"] = list(_prof["battery_capacities_wh"])
        if _cfg_kw:
            cfg = dataclasses.replace(cfg, **_cfg_kw)
        args._stress_relative_deadline_ms = _prof.get("relative_deadline_ms")
        args._stress_poisson_mean_ms = _prof.get("poisson_mean_ms")
    if args.num_agents is not None:
        cfg = dataclasses.replace(cfg, num_agents=int(args.num_agents))
    if args.global_deadline_ms is not None:
        cfg = dataclasses.replace(cfg, global_deadline_ms=args.global_deadline_ms)
    if args.battery_capacities_wh is not None:
        caps = [float(x.strip()) for x in args.battery_capacities_wh.split(",") if x.strip()]
        cfg = dataclasses.replace(cfg, battery_capacities_wh=caps)
    if args.max_episode_steps is not None:
        cfg = dataclasses.replace(cfg, max_episode_steps=args.max_episode_steps)
    if args.max_work_ms is not None:
        cfg = dataclasses.replace(cfg, max_work_ms=args.max_work_ms)
    if args.eval_every is not None:
        cfg = dataclasses.replace(cfg, eval_every=args.eval_every)
    if args.save_interval is not None:
        cfg = dataclasses.replace(cfg, save_interval=args.save_interval)
    if args.eval_episodes is not None:
        cfg = dataclasses.replace(cfg, eval_episodes=args.eval_episodes)
    if args.random_seed is not None:
        cfg = dataclasses.replace(cfg, random_seed=args.random_seed)
    if args.results_dir is not None:
        cfg = dataclasses.replace(cfg, results_dir=args.results_dir)
    if args.lr_eta is not None:
        cfg = dataclasses.replace(cfg, lr_eta=args.lr_eta)
    if args.lr_phi is not None:
        cfg = dataclasses.replace(cfg, lr_phi=args.lr_phi)
    if args.lambda_rho is not None:
        cfg = dataclasses.replace(cfg, lambda_rho=args.lambda_rho)
    if args.lipschitz_rho_estimate is not None:
        cfg = dataclasses.replace(cfg, lipschitz_rho_estimate=args.lipschitz_rho_estimate)
    if args.gradient_clip_norm is not None:
        cfg = dataclasses.replace(cfg, gradient_clip_norm=args.gradient_clip_norm)
    if args.deployment_mode is not None:
        cfg = dataclasses.replace(cfg, deployment_mode=args.deployment_mode)
    if args.local_agent_id is not None:
        cfg = dataclasses.replace(cfg, local_agent_id=args.local_agent_id)
    if args.disable_feasibility_mask:
        cfg = dataclasses.replace(cfg, disable_feasibility_mask=True)
    if cfg.deployment_mode == "distributed_jetson":
        raise NotImplementedError(
            "Training under distributed_jetson is not supported (each Jetson would have only its own "
            "trajectory data, not the team's). Use deployment_mode=simulator for training, then load "
            "the trained checkpoints on each Jetson via agent_node.py for distributed_jetson evaluation."
        )
    logger = setup_logging(cfg.log_level)
    log_proof_regime(cfg, logger)
    logger.info("Experiment: %s", exp_name)
    from alg.darts_agent import get_alloc_map as _get_alloc_map
    _am_now = _get_alloc_map()
    if _am_now:
        logger.info("[EXP2][alloc] active partitioned alloc map: %s", _am_now)
    logger.info("[bilevel] mode=%s", args.bilevel_mode)
    if args.no_incentive:
        logger.info("[ablation] no_incentive=True; ϕ frozen at 0")
    else:
        logger.info("[ablation] no_incentive=False")
    logger.info(
        "[TB1] cert envelope scales: kappa=%s exec=%s travel=%s energy=%s "
        "(certificate-only; Δ_adm/δ_auc/H/env/e_safe unscaled)",
        args.cert_conservatism,
        args.cert_conservatism_exec,
        args.cert_conservatism_travel,
        args.cert_conservatism_energy,
    )
    workload = get_workload(cfg.workload_id)
    _stress_rel = getattr(args, "_stress_relative_deadline_ms", None)
    if _stress_rel is not None and args.relative_deadline_ms is None:
        workload = dataclasses.replace(workload, relative_deadline_ms=float(_stress_rel))
    if args.poisson_mean_ms is not None or args.poisson_max_jobs is not None:
        if workload.release_pattern == "sporadic_poisson":
            new_params = dict(workload.release_params)
            if args.poisson_mean_ms is not None:
                new_params["poisson_mean_ms"] = float(args.poisson_mean_ms)
            if args.poisson_max_jobs is not None:
                new_params["max_jobs"] = int(args.poisson_max_jobs)
            workload = dataclasses.replace(workload, release_params=new_params)
        else:
            # Non-Poisson workloads accept these flags but ignore them.
            logger.warning("Ignoring --poisson_* overrides for workload %s.", cfg.workload_id)

    if args.relative_deadline_ms is not None:
        workload = dataclasses.replace(
            workload, relative_deadline_ms=float(args.relative_deadline_ms)
        )

    # TB-3 deadline tightness. Applied after the absolute --global_deadline_ms /
    # --relative_deadline_ms overrides, and before the env is built, so both
    # deadline sources reach TurtleBot3Env and DAGGraph already scaled.
    if args.deadline_scale != 1.0:
        cfg = dataclasses.replace(
            cfg, global_deadline_ms=cfg.global_deadline_ms * args.deadline_scale
        )
        workload = dataclasses.replace(
            workload, relative_deadline_ms=workload.relative_deadline_ms * args.deadline_scale
        )
    logger.info(
        "[TB3] deadline_scale=%.3f: global=%.0fms relative=%.0fms",
        args.deadline_scale,
        cfg.global_deadline_ms,
        workload.relative_deadline_ms,
    )

    from alg.periodic_release import (
        validate_periodic_horizon,
        set_exp2_runtime,
    )
    from alg.defer_fallback import reset_fallback_blocked
    _alloc_parsed = None
    if getattr(args, "alloc_map", None):
        with open(args.alloc_map, "r") as _af2:
            _alloc_parsed = json.load(_af2)
    if getattr(args, "release_mode", "single") == "periodic":
        validate_periodic_horizon(
            release_period_ms=args.release_period_ms,
            num_releases=int(args.num_releases),
            relative_deadline_ms=float(workload.relative_deadline_ms),
            max_episode_steps=int(cfg.max_episode_steps),
            global_deadline_ms=float(cfg.global_deadline_ms),
            alloc_map_active=bool(_alloc_parsed),
            disable_feasibility_mask=bool(args.disable_feasibility_mask),
        )
    set_exp2_runtime(
        release_mode=str(getattr(args, "release_mode", "single")),
        release_period_ms=(
            float(args.release_period_ms)
            if args.release_period_ms is not None
            else None
        ),
        num_releases=int(getattr(args, "num_releases", 1)),
        alloc_map=_alloc_parsed,
        alloc_map_path=(
            os.path.abspath(args.alloc_map) if getattr(args, "alloc_map", None) else None
        ),
        per_period_energy=str(getattr(args, "per_period_energy", "none")),
        D_v_ms=float(workload.relative_deadline_ms),
        workload_id=str(cfg.workload_id),
        stress=str(getattr(args, "stress", "") or ""),
        seed=int(args.eval_seed_base) if args.eval_seed_base is not None else int(cfg.random_seed),
    )
    reset_fallback_blocked()
    logger.info(
        "sim constants: overhead_ms/delta_max_ms=%.1f H/epoch_length_ms=%.1f "
        "sim T_bar_auc=0 e_safe_wh=%.4f",
        cfg.delta_max_ms,
        cfg.epoch_length_ms,
        cfg.e_safe_wh,
    )
    logger.info(
        "periodic parameters: P=%s L=%s D_v=%.1f alloc_map=%s parsed=%s "
        "per_period_energy=%s",
        args.release_period_ms,
        getattr(args, "num_releases", 1),
        float(workload.relative_deadline_ms),
        args.alloc_map,
        _alloc_parsed,
        getattr(args, "per_period_energy", "none"),
    )
    logger.info(
        "hardware delta_auc=30 ms is an offline-analysis constant and is not used on this simulation path"
    )

    if args.instance_seed is not None:
        workload, cfg = apply_instance_seed_runtime(
            workload, cfg, args.instance_seed
        )
        wcets_s = {k: round(v, 1) for k, v in workload.task_wcets_ms.items()}
        locs_s = {
            k: (round(p[0], 3), round(p[1], 3))
            for k, p in workload.task_locations_m.items()
        }
        bats_s = [round(b, 4) for b in cfg.battery_capacities_wh]
        logger.info(
            "[TB3][instance] seed=%d wcets=%s locs=%s batteries=%s",
            args.instance_seed, wcets_s, locs_s, bats_s,
        )
    else:
        logger.info(
            "[TB3][instance] instance_seed=None: nominal benchmark instance."
        )

    if args.eval_seed_base is not None:
        np.random.seed(int(args.eval_seed_base))
        torch.manual_seed(int(args.eval_seed_base))
        logger.info(
            "[eval-seed] base=%d: seeded global np.random and torch.manual_seed; per-episode "
            "env reset uses base+test_ep. NOT seeded: NetworkSimulator._rng (env._network._rng "
            "is never reseeded), so message-level jitter remains nondeterministic.",
            int(args.eval_seed_base),
        )
    else:
        logger.info("[eval-seed] disabled (--eval_seed_base unset): legacy unseeded behaviour.")

    # Resolved here rather than in the eval branch so the pinned epoch reaches config.json.
    eval_ckpt_paths: Optional[List[str]] = None
    eval_pinned_ep: Optional[int] = None
    if args.eval_only:
        eval_ckpt_paths, eval_pinned_ep = _resolve_eval_checkpoints(args, cfg.num_agents, logger)

    log_path = os.path.join(_PROJECT_ROOT, cfg.results_dir, exp_name.replace(" ", "_"))
    os.makedirs(log_path, exist_ok=True)
    set_exp2_runtime(
        log_path=log_path,
        D_v_ms=float(workload.relative_deadline_ms),
        ckpt_dir=os.path.abspath(args.load_ckpt) if args.load_ckpt else "",
        ckpt_ep=int(eval_pinned_ep) if eval_pinned_ep is not None else -1,
        seed=(
            int(args.eval_seed_base)
            if args.eval_seed_base is not None
            else int(cfg.random_seed)
        ),
        workload_id=str(cfg.workload_id),
        stress=str(getattr(args, "stress", "") or ""),
    )

    with open(os.path.join(log_path, "config.json"), "w") as f:
        cfg_dict = dict(config_to_dict(cfg))
        cfg_dict["bilevel_mode"] = args.bilevel_mode
        cfg_dict["no_incentive"] = args.no_incentive
        cfg_dict["cert_conservatism"] = float(args.cert_conservatism)
        cfg_dict["cert_conservatism_exec"] = float(args.cert_conservatism_exec)
        cfg_dict["cert_conservatism_travel"] = float(args.cert_conservatism_travel)
        cfg_dict["cert_conservatism_energy"] = float(args.cert_conservatism_energy)
        cfg_dict["deadline_scale"] = float(args.deadline_scale)
        # Provenance: the workload carries the per-node deadline and the release
        # schedule, neither of which lives on cfg. Without them a config.json
        # cannot distinguish two stress levels of the same workload_id.
        cfg_dict["workload_relative_deadline_ms"] = float(workload.relative_deadline_ms)
        cfg_dict["workload_release_pattern"] = workload.release_pattern
        cfg_dict["workload_release_params"] = dict(workload.release_params)
        cfg_dict["eval_only"] = bool(args.eval_only)
        cfg_dict["load_ckpt"] = os.path.abspath(args.load_ckpt) if args.load_ckpt else None
        cfg_dict["load_ckpt_ep"] = eval_pinned_ep
        cfg_dict["eval_seed_base"] = args.eval_seed_base
        cfg_dict["subround_log"] = bool(args.subround_log)
        # TB-3 instance provenance. cfg.task_wcets and cfg.battery_capacities_wh
        # are already dumped by config_to_dict; do not duplicate them here.
        cfg_dict["instance_seed"] = args.instance_seed
        cfg_dict["workload_task_wcets_ms"] = dict(workload.task_wcets_ms)
        cfg_dict["workload_task_locations_m"] = {
            k: list(v) for k, v in workload.task_locations_m.items()
        }
        cfg_dict["release_mode"] = getattr(args, "release_mode", "single")
        cfg_dict["release_period_ms"] = getattr(args, "release_period_ms", None)
        cfg_dict["num_releases"] = int(getattr(args, "num_releases", 1))
        cfg_dict["alloc_map"] = os.path.abspath(args.alloc_map) if getattr(args, "alloc_map", None) else None
        cfg_dict["per_period_energy"] = getattr(args, "per_period_energy", "none")
        cfg_dict["stress"] = getattr(args, "stress", None)
        json.dump(cfg_dict, f, indent=2, sort_keys=True)
    logger.info("Config saved to %s/config.json", log_path)
    if getattr(cfg, "disable_feasibility_mask", False):
        logger.warning(
            "[ABLATION] DARTS-NoCert: feasibility certificate is DISABLED. "
            "All (agent, job) bids are forced feasible. This is a paper "
            "ablation; do not use in production."
        )

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        SummaryWriter = None
        logger.warning("TensorBoard not available; install tensorboard to log metrics.")
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=os.path.join(log_path, "tb"))
    else:
        writer = None

    # TB-1 sidecar: created only when asked for, so runs without the flag leave no file.
    subround_writer: Optional[SubroundLogWriter] = None
    if args.subround_log:
        subround_writer = SubroundLogWriter(log_path)
        logger.info("[TB1][subround] sidecar enabled -> %s", subround_writer.path)

    env = TurtleBot3Env(
        num_agents=cfg.num_agents,
        global_deadline_ms=cfg.global_deadline_ms,
        cfg=cfg,
        seed=cfg.random_seed,
        workload=workload,
        enable_rta_fallback=True,
        rta_fallback_mode="rebudget",
    )
    dag = DAGGraph.from_workload(workload, cfg)
    spatial_state_dim = env.single_observation_size
    agents: List[DartsMARLAgent] = []
    for i in range(cfg.num_agents):
        agent = DartsMARLAgent(
            agent_id=f"agent_{i + 1}",
            cfg=cfg,
            logger=logging.getLogger(f"darts_agent.agent_{i + 1}"),
            seed=cfg.random_seed + i,
            spatial_state_dim=spatial_state_dim,
        )
        agent.dag_tracker = LocalDAGTracker(dag)
        agents.append(agent)

    # MC-7 NoIncentive ablation: freeze ϕ at zero for all agents.
    if args.no_incentive:
        for agent in agents:
            agent.freeze_incentive_at_zero()
        logger.info("[ablation] ϕ frozen at zero for all %d agents", len(agents))

    if args.eval_only:
        # Frozen-policy evaluation. Reached only after cfg/workload/env/agents are built exactly
        # as the training path builds them, and while the caller's scaled_cert_envelopes context
        # is still active, so the TB-1 κ envelopes apply to this evaluation.
        assert eval_ckpt_paths is not None and eval_pinned_ep is not None
        logger.info(
            "[TB1][eval_only] active cert envelope scales: %s "
            "(must match the --cert_conservatism* flags; if these read 1.0 under a non-unit "
            "flag the context placement is wrong)",
            dict(_CERT_ENVELOPE_SCALES),
        )
        if writer is not None:
            # No TensorBoard on the eval path.
            writer.close()
            writer = None
        for i, agent in enumerate(agents):
            saved_dim = _assert_ckpt_incentive_dim(eval_ckpt_paths[i], cfg.num_agents, agent.device)
            agent.load_checkpoint(eval_ckpt_paths[i])
            logger.info(
                "[eval_only] agent_%d loaded ep%d from %s (incentive_input_dim=%d, device=%s)",
                i + 1, eval_pinned_ep, eval_ckpt_paths[i], saved_dim, agent.device,
            )
        eval_episodes_n = int(args.eval_episodes) if args.eval_episodes is not None else int(
            getattr(cfg, "eval_episodes", 10)
        )
        logger.info(
            "[eval_only] evaluating %d episode(s) at pinned ep%d; eval_seed_base=%s",
            eval_episodes_n, eval_pinned_ep, args.eval_seed_base,
        )
        _eval_t0 = _time_for_auction.perf_counter()
        eval_metrics = evaluate_agents(
            env,
            agents,
            cfg,
            num_episodes=eval_episodes_n,
            training_episode=eval_pinned_ep,
            episode_rng_seed_base=args.eval_seed_base,
            subround_writer=subround_writer,
        )
        _eval_wall_s = _time_for_auction.perf_counter() - _eval_t0
        append_eval_to_csv(log_path, eval_pinned_ep, eval_metrics)
        logger.info(
            "[eval_only] wall time %.2fs total, %.2fs/episode; row for ep%d appended to %s/darts_log.csv",
            _eval_wall_s, _eval_wall_s / max(eval_episodes_n, 1), eval_pinned_ep, log_path,
        )
        return

    train_episodes = int(args.num_episodes) if args.num_episodes is not None else int(cfg.train_episodes)
    save_interval = cfg.save_interval
    V_total_cov = len(workload.node_ids)
    warned_coverage_stationary = False
    per_commit_log: List[Dict[str, Any]] = []
    log_per_commit_enabled: bool = bool(getattr(args, "log_per_commit", False))

    for episode in range(1, train_episodes + 1):
        # Logging frequency for long runs: auction logs every 500, episode summary every 100
        verbose_this_episode = (episode % 500 == 0)
        summary_this_episode = (episode % 100 == 0) or (episode <= 1)

        # Clear Phase A payment record from previous iteration (before Phase A)
        for agent in agents:
            agent.payments_phase_a.clear()

        # Phase A (Trajectory τ_k): collect episode with current (η, φ), update η by ascending J^pol (Eq. 24)
        coverage_tracker_a = CoverageTracker(V_total_cov, cfg.num_agents)
        episode_rewards, total_energy_J_per_agent, last_info_a, _, _, _ = run_episode(
            env, agents, cfg, explore=True, episode_verbose=verbose_this_episode, phase="A",
            debug_auction=getattr(args, "debug_auction", False),
            coverage_tracker=coverage_tracker_a,
            episode_idx=episode,
            log_per_commit=log_per_commit_enabled,
        )
        if log_per_commit_enabled:
            per_commit_log.extend(last_info_a.get("per_commit_records", []))
        trajectory_a_coverage = coverage_tracker_a.freeze()

        # Algorithm 2: graph inference update L_graph(ψ; τ_k) with physically verified DAG
        dag = agents[0].dag_tracker.dag if agents and agents[0].dag_tracker else None
        if dag is not None:
            device = agents[0].device
            verified_adj = dag.get_verified_adjacency_matrix().to(device)
            loss_graph_list = [agents[i].update_graph(verified_adj) for i in range(len(agents))]
            loss_graph_avg = float(np.mean(loss_graph_list))
            if writer is not None:
                writer.add_scalar("Loss/Graph_BCE", loss_graph_avg, episode)

        # Path 1 (fixed_eta): snapshot pre-update η^m before the η-update step;
        # Phase B will roll under this snapshot via temporary_bidding_weights.
        if args.bilevel_mode == "fixed_eta":
            pre_eta_snapshots = {
                a.agent_id: copy.deepcopy(a.bidding_net.state_dict()) for a in agents
            }

        loss_bid_sum = 0.0
        max_grad_eta_rho = 0.0
        for i, agent in enumerate(agents):
            ep_ret = episode_rewards[i] if i < len(episode_rewards) else 0.0
            loss_dict = agent.update_parameters(
                episode_return=ep_ret,
                coverage_stats=trajectory_a_coverage,
            )
            if loss_dict:
                loss_bid_sum += loss_dict.get("bid_loss", 0.0)
                max_grad_eta_rho = max(max_grad_eta_rho, loss_dict.get("grad_eta_rho_norm", 0.0))

        # Phase B (Trajectory τ̂_k): second episode with updated η, current φ; run_episode resets env/agents and DAG
        coverage_tracker_b = CoverageTracker(V_total_cov, cfg.num_agents)
        if args.bilevel_mode == "fixed_eta":
            with contextlib.ExitStack() as stack:
                for a in agents:
                    stack.enter_context(a.temporary_bidding_weights(pre_eta_snapshots[a.agent_id]))
                logger.info("[bilevel] m=%d: η snapshot active for Phase B", episode)
                episode_rewards_b, total_energy_B, last_info_b, _, _, trajectory_b_env_return = run_episode(
                    env, agents, cfg, explore=True, episode_verbose=False, phase="B",
                    coverage_tracker=coverage_tracker_b,
                    episode_idx=episode,
                    log_per_commit=log_per_commit_enabled,
                )
        else:
            episode_rewards_b, total_energy_B, last_info_b, _, _, trajectory_b_env_return = run_episode(
                env, agents, cfg, explore=True, episode_verbose=False, phase="B",
                coverage_tracker=coverage_tracker_b,
                episode_idx=episode,
                log_per_commit=log_per_commit_enabled,
            )
        if log_per_commit_enabled:
            per_commit_log.extend(last_info_b.get("per_commit_records", []))
        # Algorithm 2 Step 9: graph inference update L_graph(ψ; τ̂_k) with Phase B trajectory
        dag_b = agents[0].dag_tracker.dag if agents and agents[0].dag_tracker else None
        if dag_b is not None:
            verified_adj_b = dag_b.get_verified_adjacency_matrix().to(agents[0].device)
            for agent in agents:
                agent.update_graph(verified_adj_b)
        lambda_P = getattr(cfg, "lambda_P", 0.01)
        lambda_rho = getattr(cfg, "lambda_rho", 1.0)
        cov_b = coverage_tracker_b.freeze()
        r_env_B = float(trajectory_b_env_return)
        loss_phi_sum = 0.0
        clip_norm_phi = getattr(cfg, "gradient_clip_norm", 1.0)
        max_grad_phi_rho = 0.0
        if not args.no_incentive:
            for i, agent in enumerate(agents):
                if not agent.incentive_trajectory:
                    agent.incentive_trajectory.clear()
                    agent.payments_phase_a.clear()
                    continue
                V_sp_i = trajectory_a_coverage.V_sp.get(i, set())
                payments_for_my_sponsored_jobs = sum(
                    p for (p, jid, sp_idx) in agent.payments_phase_a
                    if sp_idx == i and jid in V_sp_i
                )
                inc_sum = torch.stack(agent.incentive_trajectory).sum()
                coverage_credit = lambda_rho * cov_b.omega_i(i) * cov_b.rho_sp_i(i)
                cov_t = torch.as_tensor(
                    float(coverage_credit),
                    device=inc_sum.device,
                    dtype=inc_sum.dtype,
                )
                agent.optimizer_phi.zero_grad()
                L_cov_phi = -(inc_sum * cov_t)
                grads_phi_cov = torch.autograd.grad(
                    L_cov_phi,
                    agent.incentive_net.parameters(),
                    retain_graph=True,
                    allow_unused=True,
                )
                gn_phi = math.sqrt(
                    sum(float(g.detach().float().pow(2).sum().item()) for g in grads_phi_cov if g is not None)
                )
                max_grad_phi_rho = max(max_grad_phi_rho, gn_phi)
                loss_phi = -(inc_sum * (r_env_B + coverage_credit)) + lambda_P * payments_for_my_sponsored_jobs
                loss_phi.backward()
                torch.nn.utils.clip_grad_norm_(agent.incentive_net.parameters(), max_norm=clip_norm_phi)
                agent.optimizer_phi.step()
                loss_phi_sum += float(loss_phi.detach().item())
                agent.incentive_trajectory.clear()
                agent.payments_phase_a.clear()

        theta_m_lb = max_grad_eta_rho ** 2 + max_grad_phi_rho ** 2
        if writer is not None:
            writer.add_scalar("Theorem1/grad_eta_rho_norm", float(max_grad_eta_rho), episode)
            writer.add_scalar("Theorem1/grad_phi_rho_norm", float(max_grad_phi_rho), episode)
            writer.add_scalar("Theorem1/Theta_m_lower_bound", float(theta_m_lb), episode)
            writer.add_scalar("Coverage/rho_hat", float(trajectory_a_coverage.rho_hat), episode)
            writer.add_scalar("Coverage/n_full_fidelity", float(len(trajectory_a_coverage.V_ff)), episode)
            writer.add_scalar("Coverage/total_jobs", float(trajectory_a_coverage.V_total), episode)
            for i in range(cfg.num_agents):
                aid = f"agent_{i + 1}"
                writer.add_scalar(f"Coverage/omega_{aid}", float(trajectory_a_coverage.omega_i(i)), episode)
                writer.add_scalar(f"Coverage/rho_sp_{aid}", float(trajectory_a_coverage.rho_sp_i(i)), episode)
        if theta_m_lb < 1e-6 and not warned_coverage_stationary:
            warned_coverage_stationary = True
            logger.warning(
                "Theorem 1.iii: Theta_m lower-bound proxy (||grad_eta rho||^2 + ||grad_phi rho_sp||^2)=%.3e < 1e-6 — "
                "coverage-stationary regime (training may be near convergence for coverage non-stationarity A3b).",
                theta_m_lb,
            )

        n_updates = cfg.num_agents
        if n_updates > 0:
            loss_bid_avg = loss_bid_sum / n_updates
            loss_phi_avg = loss_phi_sum / n_updates
        else:
            loss_bid_avg = 0.0
            loss_phi_avg = 0.0

        if writer is not None:
            writer.add_scalar("Loss/Bidding", loss_bid_avg, episode)
            writer.add_scalar("Loss/Incentive", loss_phi_avg, episode)
            for i in range(len(agents)):
                writer.add_scalar(f"Reward/Agent_{i}", episode_rewards[i], episode)
                writer.add_scalar(f"Energy/Agent_{i}", total_energy_J_per_agent[i], episode)
            writer.add_scalar("Reward/Total", np.sum(episode_rewards), episode)
            writer.add_scalar("Energy/Total", np.sum(total_energy_J_per_agent), episode)

        if episode % save_interval == 0:
            # Save .pth only at first and final checkpoints
            if episode == save_interval or episode == train_episodes:
                for i, agent in enumerate(agents):
                    pth_path = os.path.join(log_path, f"agent_{i + 1}_ep{episode}.pth")
                    agent.save_checkpoint(pth_path)
                logger.info("Checkpoint saved at episode %d to %s", episode, log_path)

            # Evaluation always runs every save_interval
            eval_metrics = evaluate_agents(
                env,
                agents,
                cfg,
                num_episodes=int(getattr(cfg, "eval_episodes", 10)),
                training_episode=episode,
                subround_writer=subround_writer,
            )
            append_eval_to_csv(log_path, episode, eval_metrics)
            crash_rates = [eval_metrics.get("A{}_crash_rate".format(i + 1), 0.0) for i in range(4)]
            logger.info(
                "Evaluation (10 episodes) appended to %s/darts_log.csv | task_miss_rate=%.2f, A1–A4 crash_rates=%.1f%%, %.1f%%, %.1f%%, %.1f%%",
                log_path, eval_metrics.get("task_miss_rate", float("nan")), *crash_rates,
            )

        if summary_this_episode:
            print()
            cov = trajectory_a_coverage
            st_sum = last_info_b.get("subtask_completion_times_ms") or {}
            tt_d = st_sum.get("D", None)
            dg = float(getattr(cfg, "global_deadline_ms", 180000.0))
            miss_f = 1.0 if (tt_d is None or float(tt_d) > dg) else 0.0
            task_t_disp = float(tt_d) if tt_d is not None else 0.0
            n1_tot = int(last_info_a.get("n_nondegradable_n1", 0)) + int(last_info_b.get("n_nondegradable_n1", 0))
            n2_tot = int(last_info_a.get("n_nondegradable_n2", 0)) + int(last_info_b.get("n_nondegradable_n2", 0))
            defer_tot = int(last_info_a.get("n_defer_triggers", 0)) + int(last_info_b.get("n_defer_triggers", 0))
            print(
                "[Ep {:4d}] reward={:+.2f}  miss={:.2f}  rho_hat={:.3f} ({}/{})  task_t={:.0f}ms  N1={} N2={}  defer={}".format(
                    episode,
                    float(np.sum(episode_rewards)),
                    miss_f,
                    cov.rho_hat,
                    len(cov.V_ff),
                    cov.V_total,
                    task_t_disp,
                    n1_tot,
                    n2_tot,
                    defer_tot,
                )
            )
            if verbose_this_episode:
                print("[Ep {:4d} coverage breakdown]".format(episode))
                for i in range(cfg.num_agents):
                    nsp = len(cov.V_sp.get(i, set()))
                    nffs = len(cov.V_ff_per_sponsor.get(i, set()))
                    print(
                        "  agent_{}: omega={:.3f} rho_sp={:.3f}  (sponsored {} jobs, {} reached full-fidelity)".format(
                            i + 1,
                            cov.omega_i(i),
                            cov.rho_sp_i(i),
                            nsp,
                            nffs,
                        )
                    )
            print("Ep {:5d} |".format(episode), end="")
            battery_wh = last_info_b.get("battery_wh", [0.0] * len(agents))
            init_wh = env.initial_battery_wh
            for i in range(len(agents)):
                init_i = init_wh[i] if i < len(init_wh) and init_wh[i] > 0 else 1.0
                curr_i = battery_wh[i] if i < len(battery_wh) else 0.0
                battery_pct = (curr_i / init_i) * 100.0
                print(
                    " Agent {} Reward: {:7.1f} Remaining Battery: {:5.1f}% |".format(
                        i + 1, episode_rewards_b[i], battery_pct,
                    ),
                    end="",
                )
            print()
            st = last_info_b.get("subtask_completion_times_ms") or {}
            for node_id in ("A", "B", "C", "D"):
                t = st.get(node_id, None)
                print("  Subtask {} completion: {} ms".format(node_id, f"{t:.0f}" if t is not None else "N/A"))
            task_time = st.get("D", None)
            if task_time is None:
                task_time = last_info_b.get("task_time_ms")
            if task_time is not None:
                print("  Overall task completion: {:.0f} ms".format(float(task_time)))
                miss = float(task_time) > getattr(cfg, "global_deadline_ms", 120000.0)
                print("  End-to-end deadline miss: {}".format(miss))
            else:
                print("  Overall task completion: N/A (timeout)")
                print("  End-to-end deadline miss: True")
            print()
            if verbose_this_episode:
                rts = last_info_b.get("response_times_ms", {})
                H_ms = getattr(cfg, "epoch_length_ms", 1000.0)
                print("  --- Response Times (τ→commit) ---")
                for nid in ("A", "B", "C", "D"):
                    rt = rts.get(nid)
                    bound_ok = f"≤ H={H_ms:.0f}ms ✓" if rt is not None and rt <= H_ms else ("VIOLATION ✗" if rt is not None else "N/A")
                    print(f"  Subtask {nid} RT: {f'{rt:.0f} ms' if rt is not None else 'N/A':>10}  [{bound_ok}]")

    # §V.G T2 instrumentation dump
    if log_per_commit_enabled and per_commit_log:
        import json as _json

        jsonl_path = os.path.join(log_path, "per_commit.jsonl")
        with open(jsonl_path, "w") as f:
            for entry in per_commit_log:
                f.write(_json.dumps(entry) + "\n")
        print(f"[T2-INSTR] Wrote {len(per_commit_log)} records to {jsonl_path}", flush=True)

    if writer is not None:
        writer.close()
    logger.info("Training complete. Results in %s", log_path)


if __name__ == "__main__":
    main()
