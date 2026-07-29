"""
CEA-EDF benchmark: Centralized Energy-Aware EDF for Aperiodic DAGs.
Classical CPS baseline for the DARTS paper — centralized, deterministic, no incentive mechanism.
Run from project root: PYTHONPATH=. python alg/baseline_cea_edf.py --exp_name cea_edf_baseline
"""

from __future__ import division, print_function

import argparse
import contextlib
import csv
import dataclasses
import json
import logging
import os
import sys
import time as _time_for_auction
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

# Add project root so "envs" and "alg" resolve when run from any directory (e.g. alg/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from envs.turtlebot3_spatial_env import TurtleBot3Env
from envs.simulation_env import LocalDAGTracker, NODE_STATE_ASSIGNED, NODE_STATE_READY
from envs.dag_task import DAGGraph, TaskNode
from envs.eval_node_mapping import csv_subtask_slots, csv_task_time_ms, update_ready_times_on_completion
from envs.workloads import get_workload, runtime_workload
from alg.darts_agent import (
    CommittedJob,
    MAX_TRAVEL_TIME_MS,
    _feasibility_test,
    _predict_finish_time,
    energy_cost_wh,
)
from alg.config_darts import DARTS_CONFIG
from alg.coverage import CoverageTracker
from alg.defer_fallback import (
    FallbackOutcome,
    classify_no_bidder_outcome,
    compute_fallback_cost_ms,
    determine_sponsor_idx,
    execute_fallback_assign,
)


# ---------------------------------------------------------------------------
# CEAEDFAgent: per-agent state, no neural networks
# ---------------------------------------------------------------------------

class CEAEDFAgent:
    """
    Per-agent state container: local EDF queue only. Interface-compatible with
    TurtleBot3Env.step(agents=...): get_active_job(), execute_task(), complete_active_job(),
    agent_id, dag_tracker, queue; CommittedJob has job_id and total_budget_ms for RTA.
    No bidding or incentive nets.
    """
    def __init__(
        self,
        agent_id: str,
        cfg: DARTS_CONFIG,
        logger: Optional[logging.Logger] = None,
    ):
        self.agent_id = agent_id
        self.cfg = cfg
        self._log = logger or logging.getLogger(f"cea_edf.{agent_id}")
        self._queue: List[CommittedJob] = []
        self.dag_tracker: Optional[LocalDAGTracker] = None

    @property
    def queue(self) -> List[CommittedJob]:
        return self._queue.copy()

    def reset(self) -> None:
        self._queue.clear()
        if self.dag_tracker is not None and hasattr(self.dag_tracker, "reset"):
            self.dag_tracker.reset()

    def reset_queue(self) -> None:
        self._queue.clear()

    def _sorted_queue_for_math(
        self,
        current_time_ms: float,
        env: Optional[Any] = None,
    ) -> List[Tuple[str, float, float]]:
        """Queue as (job_id, absolute_deadline_ms, c_v) for feasibility (Eq. 5–7)."""
        delta_max_ms = getattr(self.cfg, "delta_max_ms", 50.0)
        agent_idx = int(self.agent_id.split("_")[1]) - 1 if "_" in self.agent_id else 0
        result: List[Tuple[str, float, float]] = []
        for j in sorted(self._queue, key=lambda x: (x.absolute_deadline_ms, x.job_id)):
            if env is not None and getattr(env, "task_locations", None) is not None and j.job_id in env.task_locations:
                target_xy = env.task_locations[j.job_id]
                remaining_c = env.estimate_dynamic_wcet_ms(
                    agent_idx, target_xy, j.remaining_work_ms
                ) + delta_max_ms
            else:
                remaining_c = j.remaining_work_ms + MAX_TRAVEL_TIME_MS + delta_max_ms
            result.append((j.job_id, j.absolute_deadline_ms, remaining_c))
        return result

    def get_wc_energy_wh(self, job_wcet_ms: float) -> float:
        """Worst-case energy (Wh) for one job (Eq. 47)."""
        wcet_total_ms = (
            job_wcet_ms
            + MAX_TRAVEL_TIME_MS
            + getattr(self.cfg, "delta_max_ms", 50.0)
        )
        hours = wcet_total_ms / 3600000.0
        return hours * getattr(self.cfg, "p_moving_w", 7.5)

    def check_feasibility(
        self,
        new_job: TaskNode,
        current_time_ms: float,
        current_battery_wh: float = 1.0,
        dynamic_wcet_ms: Optional[float] = None,
        release_time_ms: float = 0.0,
        env: Optional[Any] = None,
    ) -> bool:
        """Temporal (Eq. 7) and energy (Eq. 47) feasibility."""
        overhead_ms = getattr(self.cfg, "delta_max_ms", 50.0)
        if dynamic_wcet_ms is not None:
            new_job_budget = dynamic_wcet_ms + overhead_ms
        else:
            new_job_budget = new_job.wcet + MAX_TRAVEL_TIME_MS + overhead_ms
        test_queue = self._sorted_queue_for_math(current_time_ms, env=env)
        d_v = release_time_ms + (new_job.relative_deadline or getattr(self.cfg, "global_deadline_ms", 180000.0))
        test_queue.append((new_job.node_id, d_v, new_job_budget))
        test_queue.sort(key=lambda x: (x[1], x[0]))
        time_feasible = _feasibility_test(test_queue, current_time_ms)
        if dynamic_wcet_ms is not None:
            new_job_energy_wh = (dynamic_wcet_ms + overhead_ms) / 3600000.0 * getattr(self.cfg, "p_moving_w", 7.5)
        else:
            new_job_energy_wh = self.get_wc_energy_wh(new_job.wcet)
        total_required_energy_wh = sum(self.get_wc_energy_wh(j.wcet) for j in self._queue) + new_job_energy_wh
        e_safe = getattr(self.cfg, "e_safe_wh", 0.05)
        energy_feasible = current_battery_wh >= (e_safe + total_required_energy_wh)
        return time_feasible and energy_feasible

    def get_active_job(self) -> Optional[CommittedJob]:
        if not self._queue:
            return None
        return self._queue[0]

    def assign_job(
        self,
        job: TaskNode,
        release_time_ms: float = 0.0,
        current_time_ms: float = 0.0,
        bid: Optional[float] = None,
        dynamic_budget_ms: Optional[float] = None,
        override_remaining_work_ms: Optional[float] = None,
    ) -> None:
        self.commit_task(job, release_time_ms, current_time_ms, dynamic_budget_ms=dynamic_budget_ms, override_remaining_work_ms=override_remaining_work_ms)

    def commit_task(
        self,
        task: TaskNode,
        release_time_ms: float,
        current_time_ms: float,
        dynamic_budget_ms: Optional[float] = None,
        override_remaining_work_ms: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        d_v = release_time_ms + (task.relative_deadline or getattr(self.cfg, "global_deadline_ms", 180000.0))
        delta_max = getattr(self.cfg, "delta_max_ms", 50.0)
        if dynamic_budget_ms is not None:
            total_c = dynamic_budget_ms
        else:
            total_c = task.wcet + MAX_TRAVEL_TIME_MS + delta_max
        job = CommittedJob(
            job_id=task.node_id,
            task_name=task.name,
            absolute_deadline_ms=d_v,
            remaining_work_ms=override_remaining_work_ms if override_remaining_work_ms is not None else task.wcet * np.random.uniform(0.7, 1.0),
            wcet=task.wcet,
            total_budget_ms=total_c,
        )
        self._queue.append(job)
        self._queue.sort(key=lambda x: (x.absolute_deadline_ms, x.job_id))
        self._log.debug("Agent %s committed Task %s (d_v=%.1f).", self.agent_id, task.node_id, d_v)

    def execute_task(self, current_time_ms: float, delta_ms: float = 1.0) -> Optional[CommittedJob]:
        active = self.get_active_job()
        if active is None:
            return None
        active.remaining_work_ms -= delta_ms
        return active

    def complete_active_job(self) -> Optional[CommittedJob]:
        if not self._queue:
            return None
        return self._queue.pop(0)


# ---------------------------------------------------------------------------
# CentralizedDispatcher: S(i,v,t_k) = ε × σ/(σ+h), allocate = argmax over feasible
# ---------------------------------------------------------------------------

class CentralizedDispatcher:
    """
    Global scheduler: composite score S(i,v,t_k) = ε × σ/(σ+h).
    ε = energy factor (e.g. battery fraction), σ = slack, h = headroom constant.
    """
    def __init__(self, cfg: DARTS_CONFIG, headroom_ms: float = 1000.0):
        self.cfg = cfg
        self.headroom_ms = headroom_ms

    def compute_score(
        self,
        agent: CEAEDFAgent,
        job: TaskNode,
        current_time_ms: float,
        battery_wh: float,
        dynamic_wcet_ms: float,
        env: Optional[Any],
        release_time_ms: float = 0.0,
    ) -> float:
        """S(i,v,t_k) = ε × σ/(σ+h). Higher = better (more slack, more energy)."""
        # ε: energy factor (normalized battery)
        cap = getattr(agent.cfg, "battery_capacities_wh", [1.0, 1.5, 2.0, 2.5])
        idx = int(agent.agent_id.split("_")[1]) - 1 if "_" in agent.agent_id else 0
        cap_wh = cap[idx] if idx < len(cap) else 1.0
        epsilon = min(1.0, battery_wh / cap_wh) if cap_wh > 0 else 0.0
        # σ: slack = d_v - predicted_finish
        test_queue = agent._sorted_queue_for_math(current_time_ms, env=env)
        d_v = release_time_ms + float(
            job.relative_deadline or getattr(self.cfg, "global_deadline_ms", 180000.0)
        )
        overhead_ms = getattr(self.cfg, "delta_max_ms", 50.0)
        budget = dynamic_wcet_ms + overhead_ms
        test_queue.append((job.node_id, d_v, budget))
        test_queue.sort(key=lambda x: (x[1], x[0]))
        predicted_finish = _predict_finish_time(test_queue, current_time_ms, d_v)
        sigma = max(0.0, d_v - predicted_finish)
        h = self.headroom_ms
        return epsilon * (sigma / (sigma + h))

    def allocate(
        self,
        job: TaskNode,
        agents: List[CEAEDFAgent],
        current_time_ms: float,
        battery_wh: List[float],
        env: Optional[Any],
        dones: List[bool],
        release_time_ms: float = 0.0,
    ) -> Optional[int]:
        """
        Argmax over feasible agents by S(i,v,t_k). Returns winner index or None.
        Uses env.estimate_dynamic_wcet_ms when available for dynamic_wcet.
        """
        feasible: List[Tuple[int, float]] = []
        for i in range(len(agents)):
            if dones[i]:
                continue
            agent = agents[i]
            if env is not None and getattr(env, "task_locations", None) is not None and job.node_id in env.task_locations:
                target_xy = env.task_locations[job.node_id]
                agent_idx = int(agent.agent_id.split("_")[1]) - 1 if "_" in agent.agent_id else i
                dynamic_wcet_ms = env.estimate_dynamic_wcet_ms(agent_idx, target_xy, job.wcet)
            else:
                dynamic_wcet_ms = job.wcet + MAX_TRAVEL_TIME_MS
            if not agent.check_feasibility(
                job, current_time_ms,
                current_battery_wh=battery_wh[i] if i < len(battery_wh) else 0.0,
                dynamic_wcet_ms=dynamic_wcet_ms,
                release_time_ms=release_time_ms,
                env=env,
            ):
                continue
            score = self.compute_score(
                agent, job, current_time_ms,
                battery_wh[i] if i < len(battery_wh) else 0.0,
                dynamic_wcet_ms, env,
                release_time_ms=release_time_ms,
            )
            feasible.append((i, score))
        if not feasible:
            return None
        return max(feasible, key=lambda x: x[1])[0]


# ---------------------------------------------------------------------------
# run_cea_edf_episode: physical loop with centralized allocation (no auction)
# ---------------------------------------------------------------------------

EPISODE_EVENTS_LOGGER = "episode_events"


@contextlib.contextmanager
def _delta_auc_subround_timer(samples_ms: List[float]) -> Iterator[None]:
    t0 = _time_for_auction.perf_counter()
    try:
        yield
    finally:
        samples_ms.append((_time_for_auction.perf_counter() - t0) * 1000.0)


def set_episode_logging_verbose(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    for name in (EPISODE_EVENTS_LOGGER, "cea_edf_agent"):
        logging.getLogger(name).setLevel(level)


def run_cea_edf_episode(
    env: TurtleBot3Env,
    agents: List[CEAEDFAgent],
    cfg: DARTS_CONFIG,
    episode_verbose: bool = False,
    dispatcher: Optional[CentralizedDispatcher] = None,
) -> Tuple[List[float], List[float], Dict[str, Any], List[bool], List[int]]:
    """
    One episode: reset env/agents, same physical loop as train_darts.run_episode,
    but replace MARL auction (lines 200–424) with CentralizedDispatcher.allocate.
    Returns: (episode_rewards, total_energy_J_per_agent, last_info, agent_died, energy_cert_violations).
    """
    set_episode_logging_verbose(episode_verbose)
    max_steps = cfg.max_episode_steps
    step_dt_s = env.step_dt_s
    if dispatcher is None:
        dispatcher = CentralizedDispatcher(cfg)

    for agent in agents:
        agent.reset()
    observations, _ = env.reset()

    workload_ep = runtime_workload(env, cfg)
    dag = DAGGraph.from_workload(workload_ep, cfg)
    global_tracker = LocalDAGTracker(dag)
    for agent in agents:
        agent.dag_tracker = global_tracker

    completed_set: Set[str] = set()
    ready_time_ms = {
        nid: 0.0 for nid in workload_ep.node_ids if not workload_ep.predecessors[nid]
    }

    episode_rewards: List[float] = [0.0] * len(agents)
    total_energy_J_per_agent: List[float] = [0.0] * len(agents)
    last_info: Dict[str, Any] = {}
    agent_died: List[bool] = [False] * len(agents)
    energy_cert_violations: List[int] = [0] * len(agents)
    episode_logger = logging.getLogger(EPISODE_EVENTS_LOGGER)
    auction_interval = 1
    deadline_missed = False
    initial_battery_wh = env.initial_battery_wh
    crash_times_ms: List[Optional[float]] = [None] * len(agents)
    timeout_processed: List[bool] = [False] * len(agents)
    tau_timeout_ms = getattr(cfg, "tau_timeout_ms", 100.0)
    commit_time_ms: Dict[str, float] = {}
    absolute_deadline_ms: Dict[str, float] = {}
    t_auc_per_epoch_ms: List[float] = []
    A_k_per_epoch: List[int] = []
    delta_auc_samples_ms: List[float] = []
    task_assigned_to: Dict[str, int] = {}
    deferred_jobs: Set[str] = set()
    V_total_episode = len(dag.nodes)
    coverage_tracker = CoverageTracker(V_total=V_total_episode, num_agents=cfg.num_agents)
    n_full_fidelity = 0
    task_outcome_dict: Dict[str, str] = {}
    n_degradable_fallback = 0
    n_nondegradable_n1 = 0
    n_nondegradable_n2 = 0
    n_defer_triggers = 0
    assignments: List[Dict[str, Any]] = []
    certified_assignments: Dict[str, Tuple[int, float]] = {}
    fallback_jobs: Set[str] = set()

    for step in range(max_steps):
        next_observations, env_rewards, dones, info = env.step(agents=agents)
        last_info = info

        if info.get("end_to_end_deadline_miss", False):
            deadline_missed = True
            break

        for i in range(len(agents)):
            if dones[i] and crash_times_ms[i] is None:
                crash_times_ms[i] = env.clock_ms

        # Centralized architecture: update global tracker DIRECTLY at completion timestep
        # (no NetworkSimulator delay). In DARTS, agent.dag_tracker is updated via
        # process_incoming_messages() when NODE_COMPLETED messages are delivered [10, 50] ms later.
        if global_tracker is not None:
            newly = info.get("newly_completed_nodes", [])
            for node_id in newly:
                global_tracker.mark_completed(node_id)
                if node_id in certified_assignments and node_id not in fallback_jobs:
                    sponsor_idx, abs_deadline_ms = certified_assignments[node_id]
                    completion_time = env.completed_tasks_ms.get(node_id, float("inf"))
                    if completion_time <= abs_deadline_ms:
                        coverage_tracker.record_full_fidelity_commit(
                            sponsor_idx=sponsor_idx,
                            job_id=node_id,
                        )
            for node_id in newly:
                completed_set.add(node_id)
            update_ready_times_on_completion(
                workload=workload_ep,
                newly_completed=newly,
                completed_set=completed_set,
                clock_ms=float(env.clock_ms),
                ready_time_ms=ready_time_ms,
            )

        # Heartbeat timeout: revert orphaned jobs to READY
        if global_tracker is not None:
            for i in range(len(agents)):
                if crash_times_ms[i] is not None and not timeout_processed[i]:
                    if env.clock_ms >= crash_times_ms[i] + tau_timeout_ms:
                        timeout_processed[i] = True
                        for job in list(agents[i].queue):
                            global_tracker.node_states[job.job_id] = NODE_STATE_READY
                        agents[i].reset_queue()
                        episode_logger.critical(
                            "[Time: %.0fms] 💀 Heartbeat timeout for Agent %d. Reverting jobs to READY.",
                            env.clock_ms, i + 1,
                        )

        # CEA-EDF: centralized allocation + Algorithm 1 no-bidder state machine (same as DARTS train_darts).
        battery_wh = info.get("battery_wh", [0.0] * len(agents))
        feasible_indices = [i for i in range(len(agents)) if not dones[i]]
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
            for job in ready_jobs:
                release_time_ms = ready_time_ms.get(job.node_id, clock_ms)
                with _delta_auc_subround_timer(delta_auc_samples_ms):
                    if not feasible_indices:
                        continue
                    winner_idx = dispatcher.allocate(
                        job,
                        agents,
                        clock_ms,
                        battery_wh,
                        env,
                        dones,
                        release_time_ms=release_time_ms,
                    )
                    if winner_idx is None:
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
                            continue
                        if outcome == FallbackOutcome.DEGRADABLE_FALLBACK_NOW:
                            c_tilde = (
                                workload.fallback_wcet_ms.get(job.node_id)
                                if workload is not None
                                else None
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
                                n_degradable_fallback += 1
                                task_outcome_dict[job.node_id] = "fallback_now"
                                fallback_jobs.add(job.node_id)
                                deferred_jobs.discard(job.node_id)
                                _fb_tx, _fb_ty = task_xy if task_xy is not None else (float("nan"), float("nan"))
                                _fb_agent_pos = env.get_agent_position(sponsor_idx)
                                _fb_ax, _fb_ay = _fb_agent_pos if _fb_agent_pos is not None else (float("nan"), float("nan"))
                                _fb_dist = (
                                    ((_fb_ax - _fb_tx) ** 2 + (_fb_ay - _fb_ty) ** 2) ** 0.5
                                    if _fb_agent_pos is not None and task_xy is not None
                                    else float("nan")
                                )
                                assignments.append({
                                    "job_id": job.node_id,
                                    "winner_agent_id": f"agent_{sponsor_idx + 1}",
                                    "target_xy": (_fb_tx, _fb_ty),
                                    "agent_xy": (_fb_ax, _fb_ay),
                                    "distance_m": _fb_dist,
                                    "composite_score": float("nan"),
                                    "rta_fallback": True,
                                    "eq23_forced": False,
                                    "algorithm1": "DEGRADABLE_FALLBACK_NOW",
                                })
                            continue
                        if outcome == FallbackOutcome.NONDEGRADABLE_N2_CASUALTY:
                            c_tilde = (
                                workload.fallback_wcet_ms.get(job.node_id)
                                if workload is not None
                                else None
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
                                n_nondegradable_n2 += 1
                                task_outcome_dict[job.node_id] = "n2_casualty"
                                fallback_jobs.add(job.node_id)
                                deferred_jobs.discard(job.node_id)
                                _fb_tx, _fb_ty = task_xy if task_xy is not None else (float("nan"), float("nan"))
                                _fb_agent_pos = env.get_agent_position(sponsor_idx)
                                _fb_ax, _fb_ay = _fb_agent_pos if _fb_agent_pos is not None else (float("nan"), float("nan"))
                                _fb_dist = (
                                    ((_fb_ax - _fb_tx) ** 2 + (_fb_ay - _fb_ty) ** 2) ** 0.5
                                    if _fb_agent_pos is not None and task_xy is not None
                                    else float("nan")
                                )
                                assignments.append({
                                    "job_id": job.node_id,
                                    "winner_agent_id": f"agent_{sponsor_idx + 1}",
                                    "target_xy": (_fb_tx, _fb_ty),
                                    "agent_xy": (_fb_ax, _fb_ay),
                                    "distance_m": _fb_dist,
                                    "composite_score": float("nan"),
                                    "rta_fallback": True,
                                    "eq23_forced": False,
                                    "algorithm1": "NONDEGRADABLE_N2_CASUALTY",
                                })
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
                            continue
                        continue
    
                    agent = agents[winner_idx]
                    if env is not None and getattr(env, "task_locations", None) is not None and job.node_id in env.task_locations:
                        target_xy = env.task_locations[job.node_id]
                        dynamic_wcet = env.estimate_dynamic_wcet_ms(
                            winner_idx, target_xy, job.wcet
                        )
                    else:
                        target_xy = None
                        dynamic_wcet = job.wcet + MAX_TRAVEL_TIME_MS
                    delta_max_ms = getattr(cfg, "delta_max_ms", 50.0)
                    cert_seq = agent.check_feasibility(
                        job,
                        current_time_ms=clock_ms,
                        current_battery_wh=(
                            battery_wh[winner_idx] if winner_idx < len(battery_wh) else 0.0
                        ),
                        dynamic_wcet_ms=dynamic_wcet,
                        release_time_ms=release_time_ms,
                        env=env,
                    )
                    if cert_seq:
                        # CEA-EDF has no native sponsorship signal; use the dispatch winner as
                        # an observational sponsor proxy for coverage accounting.
                        coverage_tracker.record_sponsor(sponsor_idx=winner_idx, job_id=job.node_id)
                        abs_deadline_ms = ready_time_ms.get(job.node_id, clock_ms) + float(
                            job.relative_deadline or g_deadline
                        )
                        absolute_deadline_ms[job.node_id] = abs_deadline_ms
                        certified_assignments[job.node_id] = (winner_idx, abs_deadline_ms)
                    agent.assign_job(
                        job,
                        release_time_ms=release_time_ms,
                        current_time_ms=clock_ms,
                        dynamic_budget_ms=dynamic_wcet + delta_max_ms,
                    )
                    global_tracker.node_states[job.node_id] = NODE_STATE_ASSIGNED
                    if job.node_id not in absolute_deadline_ms:
                        absolute_deadline_ms[job.node_id] = (
                            ready_time_ms.get(job.node_id, clock_ms)
                            + float(job.relative_deadline or g_deadline)
                        )
                    commit_time_ms[job.node_id] = clock_ms
                    task_assigned_to[job.node_id] = winner_idx
                    n_full_fidelity += 1
                    task_outcome_dict[job.node_id] = "full_fidelity"
                    deferred_jobs.discard(job.node_id)
                    # Energy cert violation check (Eq. energy_feasible in draft)
                    budget_ms = dynamic_wcet + delta_max_ms
                    e_safe = getattr(cfg, "e_safe_wh", 0.05)
                    battery_now = env._battery_wh[winner_idx] if winner_idx < len(env._battery_wh) else 0.0
                    delta_e_wc = (budget_ms / 3600000.0) * getattr(cfg, "p_moving_w", 7.5)
                    if battery_now < e_safe + delta_e_wc:
                        energy_cert_violations[winner_idx] += 1
                        episode_logger.warning(
                            "[Time: %.0fms] ⚡ Agent %d ENERGY CERT VIOLATION: B=%.4f Wh < e_safe(%.4f) + ΔE_WC(%.4f)",
                            clock_ms, winner_idx + 1, battery_now, e_safe, delta_e_wc,
                        )
                    winner_score = dispatcher.compute_score(
                        agents[winner_idx], job, clock_ms,
                        battery_wh[winner_idx] if winner_idx < len(battery_wh) else 0.0,
                        dynamic_wcet, env,
                        release_time_ms=release_time_ms,
                    )
                    episode_logger.info(
                        "[Time: %.0fms] 🎯 Job %s ASSIGNED to Agent %d (Score: %.2f)",
                        clock_ms, job.node_id, winner_idx + 1, winner_score,
                    )
                    winner_agent_pos = env.get_agent_position(winner_idx) if env is not None else None
                    winner_ax, winner_ay = winner_agent_pos if winner_agent_pos is not None else (float("nan"), float("nan"))
                    winner_tx, winner_ty = target_xy if target_xy is not None else (float("nan"), float("nan"))
                    winner_dist = ((winner_ax - winner_tx) ** 2 + (winner_ay - winner_ty) ** 2) ** 0.5 if winner_agent_pos is not None and target_xy is not None else float("nan")
                    assignments.append({
                        "job_id": job.node_id,
                        "winner_agent_id": f"agent_{winner_idx + 1}",
                        "target_xy": (winner_tx, winner_ty),
                        "agent_xy": (winner_ax, winner_ay),
                        "distance_m": winner_dist,
                        "composite_score": winner_score,
                        "rta_fallback": False,
                        "eq23_forced": False,
                        "algorithm1": "SUCCESSFUL_AUCTION",
                    })

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
        penalty_deadline = 0.0
        newly_done = info.get("newly_completed_nodes", [])
        global_deadline = getattr(cfg, "global_deadline_ms", 180000.0)
        for node_id in newly_done:
            completion_time = env.completed_tasks_ms.get(node_id, 0.0)
            penalty_deadline += max(0.0, completion_time - global_deadline)
        for i, agent in enumerate(agents):
            is_moving = is_moving_per_agent[i] if i < len(is_moving_per_agent) else False
            total_energy_J_per_agent[i] += (env.p_moving_w if is_moving else env.p_idle_w) * step_dt_s
            init_wh = initial_battery_wh[i] if i < len(initial_battery_wh) else getattr(env, "battery_capacity_wh", 50.0)
            cost_energy_norm = energy_cost_wh(agent.cfg, is_moving, step_dt_s, initial_capacity_wh=init_wh)
            r_task = env_rewards[i]
            lambda_D = getattr(agent.cfg, "lambda_D", 0.001)
            lambda_E = getattr(agent.cfg, "lambda_E", 10.0)
            episode_rewards[i] += r_task - lambda_D * penalty_deadline - lambda_E * cost_energy_norm

        observations = next_observations
        if all(dones):
            break

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
        # A_v (always defined when committed)
        allocation_delays[v] = (tc - tau) if (tau is not None and tc is not None) else None
        # R_v requires physical completion
        if tau is not None and cv is not None:
            rv = cv - tau
            response_times[v] = rv
            # v6 Eq. 47 bound; uses t_k = commit_time_ms[v]
            d_v = absolute_deadline_ms.get(v)
            if d_v is not None and tc is not None:
                rt_bound = 2.0 * H_ms + 2.0 * T_auc_bar_ms + (d_v - tc)
                if rv > rt_bound + 1.0:
                    rt_violations += 1
        else:
            response_times[v] = None
    last_info["response_times_ms"] = response_times       # R_v dict
    last_info["allocation_delays_ms"] = allocation_delays    # A_v dict
    last_info["max_response_time_ms"] = max(
        (x for x in response_times.values() if x is not None), default=None
    )
    last_info["max_allocation_delay_ms"] = max(
        (x for x in allocation_delays.values() if x is not None), default=None
    )
    last_info["rt_bound_violations"] = rt_violations
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
    last_info["assignments"] = assignments
    cov_stats = coverage_tracker.freeze()
    n_full_fidelity_cov = len(cov_stats.V_ff)
    n_pending = (
        V_total_episode
        - n_full_fidelity_cov
        - n_degradable_fallback
        - n_nondegradable_n1
        - n_nondegradable_n2
    )
    last_info["n_full_fidelity"] = n_full_fidelity_cov
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
    _sum_alloc = n_full_fidelity_cov + n_degradable_fallback + n_nondegradable_n1 + n_nondegradable_n2 + n_pending
    if _sum_alloc != V_total_episode:
        episode_logger.critical(
            "BOOKKEEPING VIOLATION (CEA-EDF): n_full_fidelity(%d)+n_degradable_fallback(%d)+n_nondegradable_n1(%d)"
            "+n_nondegradable_n2(%d)+n_pending(%d)=%d != n_total_jobs(%d).",
            n_full_fidelity_cov,
            n_degradable_fallback,
            n_nondegradable_n1,
            n_nondegradable_n2,
            n_pending,
            _sum_alloc,
            V_total_episode,
        )

    return episode_rewards, total_energy_J_per_agent, last_info, agent_died, energy_cert_violations


# ---------------------------------------------------------------------------
# Evaluation: same CSV format as train_darts
# ---------------------------------------------------------------------------

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
]


def evaluate_cea_edf(
    env: TurtleBot3Env,
    agents: List[CEAEDFAgent],
    cfg: DARTS_CONFIG,
    num_episodes: int = 10,
) -> Dict[str, Any]:
    """Run num_episodes CEA-EDF episodes; return metrics dict (same format as train_darts.evaluate_agents)."""
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
    global_deadline_ms = getattr(cfg, "global_deadline_ms", 180000.0)
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

    set_episode_logging_verbose(True)
    for test_ep in range(num_episodes):
        episode_rewards, total_energy_J, last_info, agent_died, energy_cert_violations = run_cea_edf_episode(
            env, agents, cfg, episode_verbose=True,
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
    for i in range(4):
        row[f"rho_sp_A{i+1}"] = float("nan")
        row[f"omega_A{i+1}"] = float("nan")
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
    return row


def append_eval_to_csv(log_path: str, episode: int, metrics: Dict[str, Any]) -> None:
    """Append one row to cea_edf_log.csv (same EVAL_CSV_HEADER as DARTS log.csv)."""
    log_file = os.path.join(log_path, "cea_edf_log.csv")
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CEA-EDF: Centralized Energy-Aware EDF Baseline."
    )
    parser.add_argument("--exp_name", type=str, default="debug",
                        help="Experiment name (results/<exp_name>/).")
    parser.add_argument("exp_name_pos", nargs="*",
                        help="Experiment name as positional (e.g. exp 1).")
    parser.add_argument("--num_episodes", type=int, default=20000)
    parser.add_argument("--eval_interval", type=int, default=500,
                        help="Run eval test episodes every N training episodes.")
    parser.add_argument("--eval_batch", type=int, default=10,
                        help="Number of test episodes per evaluation checkpoint.")
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
        help="Override cfg.gradient_clip_norm.",
    )
    args = parser.parse_args()
    exp_name = " ".join(args.exp_name_pos) if args.exp_name_pos else args.exp_name

    cfg = DARTS_CONFIG()
    # --- Apply workload overrides from CLI ---
    if args.workload is not None:
        cfg = dataclasses.replace(cfg, workload_id=args.workload)
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
    if args.random_seed is not None:
        cfg = dataclasses.replace(cfg, random_seed=args.random_seed)
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
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logger = logging.getLogger("cea_edf")
    logger.info("CEA-EDF baseline experiment: %s", exp_name)
    workload = get_workload(cfg.workload_id)
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

    log_path = os.path.join(_PROJECT_ROOT, cfg.results_dir, exp_name.replace(" ", "_"))
    os.makedirs(log_path, exist_ok=True)

    try:
        from torch.utils.tensorboard import SummaryWriter as _TBWriter
    except ImportError:
        _TBWriter = None  # type: ignore[misc, assignment]
    tb_writer = _TBWriter(log_dir=os.path.join(log_path, "tb")) if _TBWriter is not None else None

    with open(os.path.join(log_path, "cea_edf_config.json"), "w") as f:
        json.dump({
            "method": "CEA-EDF",
            "architecture": "centralized",
            "allocation": "energy-aware composite score S(i,v,t_k) = eps * sigma/(sigma+h)",
            "trainable_parameters": 0,
            "training_episodes": args.num_episodes,
            "eval_interval": args.eval_interval,
            "eval_batch_size": args.eval_batch,
            "workload_id": cfg.workload_id,
        }, f, indent=2)

    env = TurtleBot3Env(
        num_agents=cfg.num_agents,
        global_deadline_ms=cfg.global_deadline_ms,
        cfg=cfg,
        seed=cfg.random_seed,
        workload=workload,
    )
    dag = DAGGraph.from_workload(workload, cfg)
    dispatcher = CentralizedDispatcher(cfg)
    agents: List[CEAEDFAgent] = []
    for i in range(cfg.num_agents):
        agent = CEAEDFAgent(
            agent_id=f"agent_{i + 1}",
            cfg=cfg,
            logger=logging.getLogger(f"cea_edf.agent_{i + 1}"),
        )
        agent.dag_tracker = LocalDAGTracker(dag)
        agents.append(agent)

    total_episodes = args.num_episodes
    eval_interval = args.eval_interval
    eval_batch_size = args.eval_batch
    D_G = getattr(cfg, "global_deadline_ms", 180000.0)
    H_ms = getattr(cfg, "epoch_length_ms", 1000.0)
    checkpoint_idx = 0

    for episode_number in range(1, total_episodes + 1):
        # ---- Training episode (no logging to CSV) ----
        verbose_this_ep = (episode_number % eval_interval == 0)
        summary_this_ep = (episode_number % 100 == 0) or (episode_number == 1)

        ep_rewards, _, last_info, agent_died, energy_cert_violations = run_cea_edf_episode(
            env, agents, cfg,
            episode_verbose=verbose_this_ep,
            dispatcher=dispatcher,
        )
        if tb_writer is not None:
            tb_writer.add_scalar("Coverage/rho_hat", float(last_info.get("rho_hat", 0.0)), episode_number)
            tb_writer.add_scalar(
                "Coverage/n_full_fidelity", float(last_info.get("n_full_fidelity", 0)), episode_number
            )
            tb_writer.add_scalar(
                "Coverage/total_jobs", float(last_info.get("n_total_jobs", 0)), episode_number
            )
            for i in range(cfg.num_agents):
                aid = f"agent_{i + 1}"
                tb_writer.add_scalar(f"Coverage/omega_{aid}", 0.0, episode_number)
                tb_writer.add_scalar(f"Coverage/rho_sp_{aid}", 0.0, episode_number)

        # ---- Brief summary every 100 episodes ----
        if summary_this_ep:
            st = last_info.get("subtask_completion_times_ms") or {}
            print()
            tt_d = st.get("D", None)
            miss_f = 1.0 if (tt_d is None or float(tt_d) > D_G) else 0.0
            task_t_disp = float(tt_d) if tt_d is not None else 0.0
            n_tot = int(last_info.get("n_total_jobs", len(dag.nodes)))
            rho_h = float(last_info.get("rho_hat", 0.0))
            n_ff_i = int(last_info.get("n_full_fidelity", 0))
            print(
                "[Ep {:4d}] reward={:+.2f}  miss={:.2f}  rho_hat={:.3f} ({}/{})  task_t={:.0f}ms  N1={} N2={}  defer={}".format(
                    episode_number,
                    float(np.sum(ep_rewards)),
                    miss_f,
                    rho_h,
                    n_ff_i,
                    n_tot,
                    task_t_disp,
                    int(last_info.get("n_nondegradable_n1", 0)),
                    int(last_info.get("n_nondegradable_n2", 0)),
                    int(last_info.get("n_defer_triggers", 0)),
                )
            )
            print("Ep {:5d} |".format(episode_number), end="")
            battery_wh = last_info.get("battery_wh", [0.0] * len(agents))
            init_wh = env.initial_battery_wh
            for i in range(len(agents)):
                init_i = init_wh[i] if i < len(init_wh) and init_wh[i] > 0 else 1.0
                curr_i = battery_wh[i] if i < len(battery_wh) else 0.0
                print(" Agent {} Reward: {:7.1f} Remaining Battery: {:5.1f}% |"
                      .format(i + 1, ep_rewards[i], curr_i / init_i * 100.0), end="")
            print()
            for nid in ("A", "B", "C", "D"):
                t = st.get(nid)
                print("  Subtask {} completion: {} ms"
                      .format(nid, f"{t:.0f}" if t is not None else "N/A"))
            tt = st.get("D") or last_info.get("task_time_ms")
            if tt is not None:
                print("  Overall task completion: {:.0f} ms".format(float(tt)))
                print("  End-to-end deadline miss: {}".format(float(tt) > D_G))
            else:
                print("  Overall task completion: N/A (timeout)")
                print("  End-to-end deadline miss: True")

        # ---- Detailed report every eval_interval episodes ----
        if episode_number % eval_interval == 0:
            checkpoint_idx += 1

            # Print RT block for the training episode that landed on this checkpoint
            rts = last_info.get("response_times_ms", {})
            print()
            print("  --- Response Times (τ→commit) ---")
            for nid in ("A", "B", "C", "D"):
                rt = rts.get(nid)
                ok = f"≤ H={H_ms:.0f}ms ✓" if rt is not None and rt <= H_ms else ("VIOLATION ✗" if rt is not None else "N/A")
                print(f"  Subtask {nid} RT: {f'{rt:.0f} ms' if rt is not None else 'N/A':>10}  [{ok}]")

            # ---- CEA-EDF specific detail block (assignments from this ep) ----
            assignments = last_info.get("assignments", [])
            if assignments:
                print()
                print("  --- CEA-EDF Assignment Details ---")
                for asgn in assignments:
                    tx, ty = asgn.get("target_xy", (float("nan"), float("nan")))
                    ax, ay = asgn.get("agent_xy", (float("nan"), float("nan")))
                    dist = asgn.get("distance_m", float("nan"))
                    score = asgn.get("composite_score", float("nan"))
                    fb = "⚠️ RTA-FB" if asgn.get("rta_fallback") else ""
                    eq23 = "⚠️ Eq.23-FORCE" if asgn.get("eq23_forced") else ""
                    flags = " ".join(x for x in [fb, eq23] if x) or "✓"
                    print("  Job {:s} → Agent {:s} | tgt=({:.2f},{:.2f}) "
                          "pos=({:.2f},{:.2f}) dist={:.2f}m S={:.4f} [{}]"
                          .format(asgn.get("job_id", "?"),
                                  asgn.get("winner_agent_id", "?"),
                                  tx, ty, ax, ay, dist, score, flags))

            # ---- Run 10 test episodes and log averaged metrics to CSV ----
            rewards_per_agent = [[] for _ in range(cfg.num_agents)]
            battery_pct_per_agent = [[] for _ in range(cfg.num_agents)]
            sub_times = {"A": [], "B": [], "C": [], "D": []}
            sub_misses = {"A": 0, "B": 0, "C": 0, "D": 0}
            task_times_eval: List[float] = []
            task_miss_count = 0
            crash_counts = [0] * cfg.num_agents
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
            rt_violation_count = 0
            n_ff_eval: List[float] = []
            rho_hat_eval: List[float] = []
            n_tot_eval: List[int] = []
            n_df_eval: List[float] = []
            n_n1_eval: List[float] = []
            n_n2_eval: List[float] = []
            n_pend_eval: List[float] = []
            n_miss_n1_eval: List[float] = []
            n_miss_n2_eval: List[float] = []
            n_miss_n3_eval: List[float] = []
            n_miss_ffl_eval: List[float] = []

            logger.info("========== Eval (%d test eps) at Training Ep %d ==========",
                        eval_batch_size, episode_number)
            eval_csv_slots = csv_subtask_slots(runtime_workload(env, cfg))
            CSV_KEYS_INLINE = ("A", "B", "C", "D")
            for test_i in range(1, eval_batch_size + 1):
                logger.info("episode_events: ========== Test Episode %d/%d "
                            "(CEA-EDF checkpoint ep %d) ==========",
                            test_i, eval_batch_size, episode_number)
                t_rewards, _, t_info, t_died, t_ecv = run_cea_edf_episode(
                    env, agents, cfg,
                    episode_verbose=True,
                    dispatcher=dispatcher,
                )
                t_st = t_info.get("subtask_completion_times_ms") or {}
                for i in range(cfg.num_agents):
                    if t_ecv[i] > 0:
                        crash_counts[i] += 1
                    rewards_per_agent[i].append(t_rewards[i])
                    init_wh_i = (env.initial_battery_wh[i]
                                 if i < len(env.initial_battery_wh)
                                 else getattr(env, "battery_capacity_wh", 50.0))
                    curr_wh_i = (t_info.get("battery_wh", [0.0] * cfg.num_agents)[i]
                                 if i < len(t_info.get("battery_wh", [])) else 0.0)
                    battery_pct_per_agent[i].append(
                        curr_wh_i / init_wh_i * 100.0 if init_wh_i > 0 else 0.0)
                for k in range(4):
                    ck = CSV_KEYS_INLINE[k]
                    nid = eval_csv_slots[k]
                    tv = t_st.get(nid) if nid is not None else None
                    if tv is None:
                        sub_misses[ck] += 1
                    else:
                        sub_times[ck].append(float(tv))
                t_rts = t_info.get("response_times_ms", {})
                for k in range(4):
                    ck = CSV_KEYS_INLINE[k]
                    nid = eval_csv_slots[k]
                    v = t_rts.get(nid) if nid is not None else None
                    if v is not None:
                        rt_per_sub[ck].append(v)
                t_max_rt = t_info.get("max_response_time_ms")
                if t_max_rt is not None:
                    max_rt_list.append(t_max_rt)
                t_max_ad = t_info.get("max_allocation_delay_ms")
                if t_max_ad is not None:
                    max_alloc_list.append(t_max_ad)
                max_t_auc_list.append(float(t_info.get("max_t_auc_ms", float("nan"))))
                mean_t_auc_list.append(float(t_info.get("mean_t_auc_ms", float("nan"))))
                max_A_k_list.append(float(t_info.get("max_A_k", float("nan"))))
                mean_A_k_list.append(float(t_info.get("mean_A_k", float("nan"))))
                max_delta_auc_list.append(float(t_info.get("max_delta_auc_ms", float("nan"))))
                mean_delta_auc_list.append(float(t_info.get("mean_delta_auc_ms", float("nan"))))
                t_auc_Ak1_mean_list.append(float(t_info.get("t_auc_Ak1_mean_ms", float("nan"))))
                t_auc_Ak2_mean_list.append(float(t_info.get("t_auc_Ak2_mean_ms", float("nan"))))
                t_auc_Ak3_mean_list.append(float(t_info.get("t_auc_Ak3_mean_ms", float("nan"))))
                t_auc_Ak4plus_mean_list.append(float(t_info.get("t_auc_Ak4plus_mean_ms", float("nan"))))
                n_epochs_Ak1_list.append(int(t_info.get("n_epochs_Ak1", 0)))
                n_epochs_Ak2_list.append(int(t_info.get("n_epochs_Ak2", 0)))
                n_epochs_Ak3_list.append(int(t_info.get("n_epochs_Ak3", 0)))
                n_epochs_Ak4plus_list.append(int(t_info.get("n_epochs_Ak4plus", 0)))
                rt_violation_count += t_info.get("rt_bound_violations", 0)
                ttt = csv_task_time_ms(t_info)
                if ttt is None:
                    task_miss_count += 1
                else:
                    ttf = float(ttt)
                    task_times_eval.append(ttf)
                    if ttf > D_G:
                        task_miss_count += 1
                n_ff_eval.append(float(t_info.get("n_full_fidelity", 0)))
                rho_hat_eval.append(float(t_info.get("rho_hat", float("nan"))))
                n_tot_eval.append(int(t_info.get("n_total_jobs", 0)))
                n_df_eval.append(float(t_info.get("n_degradable_fallback", 0)))
                n_n1_eval.append(float(t_info.get("n_nondegradable_n1", 0)))
                n_n2_eval.append(float(t_info.get("n_nondegradable_n2", 0)))
                n_pend_eval.append(float(t_info.get("n_pending", 0)))
                n_miss_n1_eval.append(float(t_info.get("n_miss_n1_drop", 0)))
                n_miss_n2_eval.append(float(t_info.get("n_miss_n2_fallback_insufficient", 0)))
                n_miss_n3_eval.append(float(t_info.get("n_miss_n3_unreached", 0)))
                n_miss_ffl_eval.append(float(t_info.get("n_miss_full_fidelity_late", 0)))

            # Build metrics dict from test episodes (mirrors existing metrics block)
            num_ep = float(eval_batch_size)
            metrics: Dict[str, Any] = {}
            for i in range(4):
                if i < cfg.num_agents:
                    metrics[f"A{i+1}_total_reward"] = float(np.mean(rewards_per_agent[i]))
                    metrics[f"A{i+1}_remaining_battery_pct"] = (
                        float(np.mean(battery_pct_per_agent[i]))
                        if battery_pct_per_agent[i] else float("nan"))
                    metrics[f"A{i+1}_crash_rate"] = 100.0 * crash_counts[i] / eval_batch_size
                else:
                    metrics[f"A{i+1}_total_reward"] = 0.0
                    metrics[f"A{i+1}_remaining_battery_pct"] = float("nan")
                    metrics[f"A{i+1}_crash_rate"] = 0.0
            for ck in CSV_KEYS_INLINE:
                metrics[f"sub{ck}_miss_rate"] = sub_misses[ck] / num_ep
                metrics[f"sub{ck}_time"] = (
                    float(np.mean(sub_times[ck])) if sub_times[ck] else float("nan"))
            metrics["task_miss_rate"] = task_miss_count / num_ep
            metrics["rho_hat"] = float(np.nanmean(rho_hat_eval)) if rho_hat_eval else float("nan")
            metrics["n_full_fidelity"] = float(np.mean(n_ff_eval)) if n_ff_eval else float("nan")
            metrics["n_total_jobs"] = int(n_tot_eval[0]) if n_tot_eval else 0
            for i in range(4):
                metrics[f"rho_sp_A{i+1}"] = float("nan")
                metrics[f"omega_A{i+1}"] = float("nan")
            metrics["task_time"] = (
                float(np.mean(task_times_eval)) if task_times_eval else float("nan"))
            for ck in CSV_KEYS_INLINE:
                vals = rt_per_sub[ck]
                metrics[f"sub{ck}_rt_ms"] = float(np.mean(vals)) if vals else float("nan")
            metrics["max_rt_ms"] = (
                float(np.mean(max_rt_list)) if max_rt_list else float("nan"))
            metrics["max_alloc_delay_ms"] = (
                float(np.nanmean(max_alloc_list)) if max_alloc_list else float("nan"))
            metrics["rt_bound_violations"] = rt_violation_count
            metrics["max_t_auc_ms"] = (
                float(np.nanmean(max_t_auc_list)) if max_t_auc_list else float("nan"))
            metrics["mean_t_auc_ms"] = (
                float(np.nanmean(mean_t_auc_list)) if mean_t_auc_list else float("nan"))
            metrics["max_A_k"] = (
                float(np.nanmean(max_A_k_list)) if max_A_k_list else float("nan"))
            metrics["mean_A_k"] = (
                float(np.nanmean(mean_A_k_list)) if mean_A_k_list else float("nan"))
            metrics["max_delta_auc_ms"] = (
                float(np.nanmean(max_delta_auc_list)) if max_delta_auc_list else float("nan"))
            metrics["mean_delta_auc_ms"] = (
                float(np.nanmean(mean_delta_auc_list)) if mean_delta_auc_list else float("nan"))
            def _nanmean_fin_m(vals: List[float]) -> float:
                if not vals:
                    return float("nan")
                arr = np.asarray(vals, dtype=np.float64)
                return float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else float("nan")

            metrics["t_auc_Ak1_mean_ms"] = _nanmean_fin_m(t_auc_Ak1_mean_list)
            metrics["t_auc_Ak2_mean_ms"] = _nanmean_fin_m(t_auc_Ak2_mean_list)
            metrics["t_auc_Ak3_mean_ms"] = _nanmean_fin_m(t_auc_Ak3_mean_list)
            metrics["t_auc_Ak4plus_mean_ms"] = _nanmean_fin_m(t_auc_Ak4plus_mean_list)
            metrics["n_epochs_Ak1"] = int(np.sum(n_epochs_Ak1_list)) if n_epochs_Ak1_list else 0
            metrics["n_epochs_Ak2"] = int(np.sum(n_epochs_Ak2_list)) if n_epochs_Ak2_list else 0
            metrics["n_epochs_Ak3"] = int(np.sum(n_epochs_Ak3_list)) if n_epochs_Ak3_list else 0
            metrics["n_epochs_Ak4plus"] = int(np.sum(n_epochs_Ak4plus_list)) if n_epochs_Ak4plus_list else 0
            metrics["n_degradable_fallback"] = float(np.mean(n_df_eval)) if n_df_eval else float("nan")
            metrics["n_nondegradable_n1"] = float(np.mean(n_n1_eval)) if n_n1_eval else float("nan")
            metrics["n_nondegradable_n2"] = float(np.mean(n_n2_eval)) if n_n2_eval else float("nan")
            metrics["n_pending"] = float(np.mean(n_pend_eval)) if n_pend_eval else float("nan")
            metrics["n_miss_n1_drop"] = float(np.mean(n_miss_n1_eval)) if n_miss_n1_eval else float("nan")
            metrics["n_miss_n2_fallback_insufficient"] = (
                float(np.mean(n_miss_n2_eval)) if n_miss_n2_eval else float("nan")
            )
            metrics["n_miss_n3_unreached"] = (
                float(np.mean(n_miss_n3_eval)) if n_miss_n3_eval else float("nan")
            )
            metrics["n_miss_full_fidelity_late"] = (
                float(np.mean(n_miss_ffl_eval)) if n_miss_ffl_eval else float("nan")
            )

            append_eval_to_csv(log_path, episode_number, metrics)
            logger.info(
                "CEA-EDF Eval %d/%d appended to %s/cea_edf_log.csv | "
                "task_miss_rate=%.2f, crash=[%.1f%%, %.1f%%, %.1f%%, %.1f%%]",
                checkpoint_idx, total_episodes // eval_interval, log_path,
                metrics.get("task_miss_rate", float("nan")),
                *[metrics.get(f"A{j+1}_crash_rate", 0.0) for j in range(4)])

    if tb_writer is not None:
        tb_writer.close()
    logger.info("CEA-EDF training complete. Results in %s", log_path)


if __name__ == "__main__":
    main()
