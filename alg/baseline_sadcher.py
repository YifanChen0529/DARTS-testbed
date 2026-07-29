"""
SADCHER baseline: training and evaluation script (Bichler et al., IEEE MRS 2025).
Same structure as baseline_cea_edf.py and train_darts.py. MARL-only, no real-time guarantees.
Run: PYTHONPATH=. python alg/baseline_sadcher.py --exp_name sadcher_baseline
"""

from __future__ import division, print_function

import argparse
import copy
import csv
import dataclasses
import json
import logging
import math
import os
import sys
import time as _time_for_auction
from typing import Any, Dict, List, Optional, Set, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F

from envs.turtlebot3_spatial_env import TurtleBot3Env
from envs.simulation_env import LocalDAGTracker, NODE_STATE_ASSIGNED, NODE_STATE_READY
from envs.dag_task import DAGGraph, TaskNode
from envs.eval_node_mapping import csv_subtask_slots, csv_task_time_ms, update_ready_times_on_completion
from envs.workloads import get_workload, runtime_workload
from alg.darts_agent import CommittedJob, energy_cost_wh
from alg.sadcher_agent import (
    SadcherAgent,
    SadcherRewardNet,
    relaxed_bipartite_matching,
)
from alg.config_sadcher import SADCHER_CONFIG
from alg.coverage import CoverageTracker

# ---------------------------------------------------------------------------
# SADCHER verification: NO feasibility cert, NO EDF queue sort, NO energy mask, NO Eq.23 fallback
# (Checked at import: baseline_sadcher has no _feasibility_test/check_feasibility/cert/Feasible;
#  sadcher_agent has no queue sort by deadline; assignments ignore e_safe; no force-assign.)
# ---------------------------------------------------------------------------

# Same CSV header as DARTS / CEA-EDF (identical columns for comparison)
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

_SCHEMA_MISS_LOGGED = False

MAP_SIZE = 10.0
MAP_DIAG = MAP_SIZE * (2.0 ** 0.5)


def _sadcher_commit_is_full_fidelity(
    env: TurtleBot3Env,
    cfg: SADCHER_CONFIG,
    dag: DAGGraph,
    agent_idx: int,
    node_id: str,
    budget_ms: float,
    clock_ms: float,
    battery_wh: float,
    queue_snapshot: List[CommittedJob],
    release_time_ms: float = 0.0,
) -> bool:
    """Same temporal+energy certificate as CEA-EDF (cert_seq=1) for cross-method |V_ff|."""
    from alg.baseline_cea_edf import CEAEDFAgent

    aid = f"agent_{agent_idx + 1}"
    probe = CEAEDFAgent(agent_id=aid, cfg=cfg)
    probe._queue = [copy.copy(j) for j in queue_snapshot]
    task = dag.nodes[node_id]
    delta = float(getattr(cfg, "delta_max_ms", 50.0))
    dw = max(float(budget_ms) - delta, 1.0)
    return probe.check_feasibility(
        task,
        clock_ms,
        battery_wh,
        dynamic_wcet_ms=dw,
        release_time_ms=release_time_ms,
        env=env,
    )


def _build_robot_features(
    env: TurtleBot3Env,
    agents: List[SadcherAgent],
    cfg: SADCHER_CONFIG,
    clock_ms: float,
) -> np.ndarray:
    """Robot features [N, robot_feature_dim]: x/10, y/10, battery_norm, workload_norm, remaining_time_norm, idle_flag."""
    N = len(agents)
    caps = getattr(cfg, "battery_capacities_wh", [1.0, 1.5, 2.0, 2.5])
    max_work = getattr(cfg, "max_work_ms", 142500.0)
    global_deadline = getattr(cfg, "global_deadline_ms", 180000.0)
    out = np.zeros((N, cfg.robot_feature_dim), dtype=np.float32)
    for i in range(N):
        pos = env.get_agent_position(i) if hasattr(env, "get_agent_position") else (0.0, 0.0)
        if pos is None:
            pos = (0.0, 0.0)
        out[i, 0] = pos[0] / MAP_SIZE
        out[i, 1] = pos[1] / MAP_SIZE
        cap_i = caps[i] if i < len(caps) else 1.0
        bat = env._battery_wh[i] if hasattr(env, "_battery_wh") and i < len(env._battery_wh) else cap_i
        out[i, 2] = bat / cap_i if cap_i > 0 else 0.0
        active = agents[i].get_active_job()
        if active is not None:
            out[i, 3] = min(1.0, active.remaining_work_ms / max_work) if max_work > 0 else 0.0
            slack = active.absolute_deadline_ms - clock_ms
            out[i, 4] = max(0.0, min(1.0, slack / global_deadline)) if global_deadline > 0 else 0.0
            out[i, 5] = 0.0
        else:
            out[i, 3] = 0.0
            out[i, 4] = 1.0
            out[i, 5] = 1.0
    return out


def _build_task_features(
    env: TurtleBot3Env,
    global_tracker: LocalDAGTracker,
    cfg: SADCHER_CONFIG,
    node_ids: Tuple[str, ...],
) -> np.ndarray:
    """Task features [M, task_feature_dim]: target_x/10, target_y/10, wcet_norm, precedence_onehot(3)."""
    wcets = getattr(cfg, "task_wcets", {"A": 15000.0, "B": 25000.0, "C": 25000.0, "D": 30000.0})
    max_wcet = max(wcets.values()) if wcets else 30000.0
    states = global_tracker.node_states
    locs = getattr(env, "task_locations", {}) or {}
    M = len(node_ids)
    out = np.zeros((M, cfg.task_feature_dim), dtype=np.float32)
    for j, node_id in enumerate(node_ids):
        xy = locs.get(node_id, (0.0, 0.0))
        out[j, 0] = xy[0] / MAP_SIZE
        out[j, 1] = xy[1] / MAP_SIZE
        wcet = wcets.get(node_id, 20000.0)
        out[j, 2] = wcet / max_wcet if max_wcet > 0 else 0.0
        s = states.get(node_id, "WAITING_FOR_PREDECESSORS")
        if s == "WAITING_FOR_PREDECESSORS" or s == "WAITING":
            out[j, 3:6] = [1, 0, 0]
        elif s == "READY":
            out[j, 3:6] = [0, 1, 0]
        else:
            out[j, 3:6] = [0, 0, 1]
    return out


def _build_distances(
    env: TurtleBot3Env, node_ids: Tuple[str, ...]
) -> np.ndarray:
    """Pairwise distances [N, M] normalized by map diagonal."""
    N = env.num_agents
    locs = getattr(env, "task_locations", {}) or {}
    M = len(node_ids)
    out = np.zeros((N, M), dtype=np.float32)
    for i in range(N):
        pos = env.get_agent_position(i) if hasattr(env, "get_agent_position") else None
        if pos is None:
            pos = (0.0, 0.0)
        for j, node_id in enumerate(node_ids):
            xy = locs.get(node_id, (0.0, 0.0))
            d = ((pos[0] - xy[0]) ** 2 + (pos[1] - xy[1]) ** 2) ** 0.5
            out[i, j] = min(1.0, d / MAP_DIAG) if MAP_DIAG > 0 else 0.0
    return out


def run_sadcher_episode(
    env: TurtleBot3Env,
    agents: List[SadcherAgent],
    cfg: SADCHER_CONFIG,
    reward_net: SadcherRewardNet,
    device: torch.device,
    node_ids: Tuple[str, ...],
    episode_verbose: bool = False,
    explore: bool = True,
    episode_number: Optional[int] = None,
) -> Tuple[List[float], List[float], Dict[str, Any], List[bool], Optional[List[Dict[str, Any]]]]:
    """
    One episode: reset env/agents, DAG tracker; at every step run physics then SADCHER decision.
    Returns (episode_rewards, total_energy_J_per_agent, last_info, agent_died, energy_cert_violations, trajectory).
    trajectory is list of {R, V, assignment, step_reward} for REINFORCE (None if not explore).
    """
    max_steps = cfg.max_episode_steps
    M = len(node_ids)
    step_dt_s = env.step_dt_s
    delta_max_ms = getattr(cfg, "delta_max_ms", 50.0)
    initial_battery_wh = env.initial_battery_wh
    tau_timeout_ms = getattr(cfg, "tau_timeout_ms", 100.0)
    use_pre_moving = getattr(cfg, "use_pre_moving", True)

    # Match DARTS/CEA-EDF: only print event logs (assignment, completion) and agent-level logs when episode_verbose
    level_events = logging.INFO if episode_verbose else logging.CRITICAL
    logging.getLogger("episode_events").setLevel(level_events)
    for i in range(len(agents)):
        logging.getLogger(f"sadcher.agent_{i + 1}").setLevel(level_events)

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
    V_total_ep = len(dag.nodes)
    coverage_tracker = CoverageTracker(V_total=V_total_ep, num_agents=cfg.num_agents)

    episode_rewards: List[float] = [0.0] * len(agents)
    total_energy_J_per_agent: List[float] = [0.0] * len(agents)
    last_info: Dict[str, Any] = {}
    agent_died: List[bool] = [False] * len(agents)
    crash_times_ms: List[Optional[float]] = [None] * len(agents)
    timeout_processed: List[bool] = [False] * len(agents)
    trajectory: List[Dict[str, Any]] = [] if explore else []
    episode_logger = logging.getLogger("episode_events")
    total_assignments = 0
    energy_cert_violations: List[int] = [0] * len(agents)
    commit_time_ms: Dict[str, float] = {}
    absolute_deadline_ms: Dict[str, float] = {}
    t_auc_per_epoch_ms: List[float] = []
    A_k_per_epoch: List[int] = []
    delta_auc_samples_ms: List[float] = []
    certified_assignments: Dict[str, Tuple[int, float]] = {}

    for step in range(max_steps):
        next_observations, env_rewards, dones, info = env.step(agents=agents)
        last_info = info

        if info.get("end_to_end_deadline_miss", False):
            break

        for i in range(len(agents)):
            if dones[i] and crash_times_ms[i] is None:
                crash_times_ms[i] = env.clock_ms

        newly = info.get("newly_completed_nodes", [])
        for node_id in newly:
            global_tracker.mark_completed(node_id)
            if node_id in certified_assignments:
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

        for i in range(len(agents)):
            if crash_times_ms[i] is not None and not timeout_processed[i]:
                if env.clock_ms >= crash_times_ms[i] + tau_timeout_ms:
                    timeout_processed[i] = True
                    for job in list(agents[i].queue):
                        global_tracker.node_states[job.job_id] = NODE_STATE_READY
                    agents[i].reset_queue()

        for i, d in enumerate(dones):
            if d:
                agent_died[i] = True

        is_moving_per_agent = info.get("is_moving", [False] * len(agents))
        penalty_deadline = 0.0
        for node_id in info.get("newly_completed_nodes", []):
            completion_time = env.completed_tasks_ms.get(node_id, 0.0)
            global_deadline = getattr(cfg, "global_deadline_ms", 180000.0)
            penalty_deadline += max(0.0, completion_time - global_deadline)
        team_step_reward = 0.0
        for i, agent in enumerate(agents):
            is_moving = is_moving_per_agent[i] if i < len(is_moving_per_agent) else False
            total_energy_J_per_agent[i] += (env.p_moving_w if is_moving else env.p_idle_w) * step_dt_s
            init_wh = initial_battery_wh[i] if i < len(initial_battery_wh) else getattr(env, "battery_capacity_wh", 50.0)
            cost_energy_norm = energy_cost_wh(agent.cfg, is_moving, step_dt_s, initial_capacity_wh=init_wh)
            r_task = env_rewards[i]
            lambda_D = getattr(agent.cfg, "lambda_D", 0.001)
            lambda_E = getattr(agent.cfg, "lambda_E", 10.0)
            step_contrib = r_task - lambda_D * penalty_deadline - lambda_E * cost_energy_norm
            episode_rewards[i] += step_contrib
            team_step_reward += step_contrib

        if explore and trajectory:
            trajectory[-1]["step_reward"] = team_step_reward

        if all(dones):
            observations = next_observations
            break

        # SADCHER decision step: only run when at least one agent idle and one task READY (epoch-driven, mirrors DARTS)
        has_idle = any(agents[i].get_active_job() is None for i in range(len(agents)))
        has_ready = any(
            global_tracker.node_states.get(node_ids[j]) == "READY" for j in range(M)
        )
        if not (has_idle and has_ready):
            observations = next_observations
            continue  # skip NN inference entirely; no assignment is possible anyway

        _t_epoch_auction_start = _time_for_auction.perf_counter()
        n_assigned_this_epoch = 0
        robot_f = _build_robot_features(env, agents, cfg, env.clock_ms)
        task_f = _build_task_features(env, global_tracker, cfg, node_ids)
        dist = _build_distances(env, node_ids)
        ready_mask = torch.tensor(
            [
                1.0 if global_tracker.node_states.get(node_ids[j], "") == "READY" else 0.0
                for j in range(M)
            ],
            dtype=torch.float32, device=device,
        )
        idle_mask = torch.tensor(
            [1.0 if agents[i].get_active_job() is None else 0.0 for i in range(len(agents))],
            dtype=torch.float32, device=device,
        )

        rf_t = torch.tensor(robot_f, dtype=torch.float32, device=device).unsqueeze(0)
        tf_t = torch.tensor(task_f, dtype=torch.float32, device=device).unsqueeze(0)
        dist_t = torch.tensor(dist, dtype=torch.float32, device=device).unsqueeze(0)

        R, V = reward_net(rf_t, tf_t, None, dist_t)
        R_single = R.squeeze(0)
        assignment = relaxed_bipartite_matching(R_single, ready_mask, idle_mask, node_ids)

        assigned_tasks = [node_id for node_id in assignment.values() if node_id != "idle"]
        if len(assigned_tasks) != len(set(assigned_tasks)):
            _logger = logging.getLogger(__name__)
            _logger.error(
                "Duplicate task assignment in same epoch: episode=%s epoch_clock_ms=%.0f assignment=%s",
                episode_number if episode_number is not None else "?",
                env.clock_ms,
                assignment,
            )
            raise AssertionError(
                "no two agents may be assigned the same non-idle task in the same epoch"
            )

        if explore:
            trajectory.append({
                "robot_f": rf_t.detach(),
                "task_f": tf_t.detach(),
                "dist": dist_t.detach(),
                "assignment": assignment,
            })

        # Apply assignments (no feasibility check)
        task_wcets = getattr(cfg, "task_wcets", {"A": 15000.0, "B": 25000.0, "C": 25000.0, "D": 30000.0})
        locs = getattr(env, "task_locations", {}) or {}
        for i, node_id in assignment.items():
            if node_id == "idle":
                continue
            if global_tracker.node_states.get(node_id) != "READY":
                continue
            j = node_ids.index(node_id)
            reward_val = R_single[i, j].detach().cpu().item()
            episode_logger.info(
                "[Time: %.0fms] 🎯 Job %s ASSIGNED to Agent %d (Reward: %.2f)",
                env.clock_ms, node_id, i + 1, reward_val,
            )
            wcet = task_wcets.get(node_id, 20000.0)
            target_xy = locs.get(node_id)
            if target_xy is not None and hasattr(env, "estimate_dynamic_wcet_ms"):
                budget_ms = env.estimate_dynamic_wcet_ms(i, target_xy, wcet) + delta_max_ms
            else:
                budget_ms = wcet + 25000.0 + delta_max_ms

            if episode_verbose:
                tx, ty = target_xy if target_xy is not None else (0.0, 0.0)
                agent_pos = env.get_agent_position(i) if hasattr(env, "get_agent_position") else None
                ax, ay = agent_pos if agent_pos is not None else (0.0, 0.0)
                dist = math.hypot(tx - ax, ty - ay) if (target_xy is not None and agent_pos is not None) else float("nan")
                r_raw = R_single[i, j].detach().cpu().item()
                episode_logger.info(
                    "[Assignment] Job %s → Agent %d\n"
                    "  Target location : (%.2f, %.2f)\n"
                    "  Agent position  : (%.2f, %.2f)\n"
                    "  Distance to tgt : %.2f m\n"
                    "  R-score (raw)   : %.4f\n"
                    "  Budget assigned : %.0f ms  (WCET=%.0f ms)\n"
                    "  [No cert check — SADCHER baseline]",
                    node_id, i + 1, tx, ty, ax, ay, dist, r_raw, budget_ms, wcet,
                )

            job = dag.nodes[node_id]
            relative_deadline = float(
                getattr(job, "relative_deadline", None)
                or getattr(cfg, "global_deadline_ms", 180000.0)
            )
            abs_deadline_ms = ready_time_ms.get(node_id, env.clock_ms) + relative_deadline
            bat_wh = env._battery_wh[i] if hasattr(env, "_battery_wh") and i < len(env._battery_wh) else 0.0
            cert_seq = _sadcher_commit_is_full_fidelity(
                env,
                cfg,
                dag,
                i,
                node_id,
                budget_ms,
                env.clock_ms,
                bat_wh,
                agents[i].queue,
                release_time_ms=ready_time_ms.get(node_id, env.clock_ms),
            )
            if cert_seq:
                # SADCHER has no sponsorship mechanism; use matched agent as observational sponsor.
                coverage_tracker.record_sponsor(sponsor_idx=i, job_id=node_id)
                certified_assignments[node_id] = (i, abs_deadline_ms)
            agents[i].assign_task(node_id, budget_ms, wcet_ms=wcet)
            global_tracker.node_states[node_id] = NODE_STATE_ASSIGNED
            absolute_deadline_ms[node_id] = abs_deadline_ms
            commit_time_ms[node_id] = env.clock_ms
            total_assignments += 1

            # Energy cert violation check (Eq. energy_feasible in draft)
            e_safe = getattr(cfg, "e_safe_wh", 0.05)
            battery_now = env._battery_wh[i] if i < len(env._battery_wh) else 0.0
            delta_e_wc = (budget_ms / 3600000.0) * getattr(cfg, "p_moving_w", 7.5)
            if battery_now < e_safe + delta_e_wc:
                energy_cert_violations[i] += 1
                episode_logger.warning(
                    "[Time: %.0fms] ⚡ Agent %d ENERGY CERT VIOLATION: B=%.4f Wh < e_safe(%.4f) + ΔE_WC(%.4f)",
                    env.clock_ms, i + 1, battery_now, e_safe, delta_e_wc,
                )
            n_assigned_this_epoch += 1

        _t_epoch_ms = (_time_for_auction.perf_counter() - _t_epoch_auction_start) * 1000.0
        t_auc_per_epoch_ms.append(_t_epoch_ms)
        A_k_per_epoch.append(n_assigned_this_epoch)
        delta_auc_samples_ms.append(_t_epoch_ms / max(1, n_assigned_this_epoch))

        observations = next_observations

    if info.get("end_to_end_deadline_miss", False):
        for i in range(len(agents)):
            episode_rewards[i] += -50.0

    if episode_verbose:
        episode_logger.info(
            "[Episode Summary] Assignments made: %d (execution-level RTA removed; see DARTS Algorithm 1 in train_darts/CEA-EDF)",
            total_assignments,
        )

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
    cov_stats = coverage_tracker.freeze()
    n_ff_epi = len(cov_stats.V_ff)
    last_info["n_full_fidelity"] = int(n_ff_epi)
    last_info["n_degradable_fallback"] = 0
    last_info["n_nondegradable_n1"] = 0
    last_info["n_nondegradable_n2"] = 0
    last_info["n_pending"] = max(0, V_total_ep - n_ff_epi)
    last_info["n_defer_triggers"] = 0
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

    return episode_rewards, total_energy_J_per_agent, last_info, agent_died, energy_cert_violations, (trajectory if explore else None)


def _compute_log_prob_assignment(
    R: torch.Tensor, assignment: Dict[int, str], task_order: Tuple[str, ...]
) -> torch.Tensor:
    """Log probability of greedy assignment under softmax policy over R[i,:]. R [N, M+1]."""
    log_probs = []
    M = len(task_order)
    for i, node_id in assignment.items():
        logits = R[i, :]
        j = task_order.index(node_id) if node_id in task_order else M
        if j < logits.shape[0]:
            log_probs.append(F.log_softmax(logits, dim=0)[j])
    if not log_probs:
        return torch.tensor(0.0, device=R.device, dtype=R.dtype)
    return torch.stack(log_probs).sum()


def update_reward_net_reinforce(
    reward_net: SadcherRewardNet,
    trajectory: List[Dict[str, Any]],
    episode_return: float,
    optimizer: torch.optim.Adam,
    cfg: SADCHER_CONFIG,
    device: torch.device,
) -> None:
    """REINFORCE with baseline: re-forward from stored state, then ∇ E[G] ≈ Σ_t (G_t - V(s_t)) ∇ log π(a_t|s_t)."""
    if not trajectory:
        return
    gamma = getattr(cfg, "gamma", 0.99)
    rewards = [t.get("step_reward", 0.0) for t in trajectory]
    T = len(rewards)
    G = np.zeros(T, dtype=np.float32)
    G[T - 1] = rewards[T - 1]
    for t in range(T - 2, -1, -1):
        G[t] = rewards[t] + gamma * G[t + 1]
    G_tensor = torch.tensor(G, dtype=torch.float32, device=device)

    log_probs = []
    V_list = []
    for t in trajectory:
        rf = t["robot_f"].requires_grad_(False)
        tf = t["task_f"].requires_grad_(False)
        dist = t["dist"].requires_grad_(False)
        R, V = reward_net(rf, tf, None, dist)
        R_s = R.squeeze(0)
        V_list.append(V.squeeze(0))
        log_probs.append(
            _compute_log_prob_assignment(
                R_s, t["assignment"], reward_net.task_encoder.node_ids
            )
        )

    V_stack = torch.stack(V_list)
    advantages = G_tensor - V_stack.detach()
    log_prob_stack = torch.stack(log_probs)
    policy_loss = -(advantages * log_prob_stack).sum()
    value_loss = F.mse_loss(V_stack, G_tensor)
    optimizer.zero_grad()
    (policy_loss + 0.5 * value_loss).backward()
    optimizer.step()


def append_eval_to_csv(log_path: str, episode: int, metrics: Dict[str, Any]) -> None:
    """Append one row to sadcher_log.csv. Same EVAL_CSV_HEADER and row logic as baseline_cea_edf (N/A for None/NaN)."""
    global _SCHEMA_MISS_LOGGED
    if not _SCHEMA_MISS_LOGGED:
        logging.getLogger("sadcher").info(
            "[schema] SADCHER does not classify N1/N2/N3 misses; columns written as NaN by design."
        )
        _SCHEMA_MISS_LOGGED = True
    log_file = os.path.join(log_path, "sadcher_log.csv")
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


def evaluate_sadcher(
    env: TurtleBot3Env,
    agents: List[SadcherAgent],
    cfg: SADCHER_CONFIG,
    reward_net: SadcherRewardNet,
    device: torch.device,
    num_episodes: int = 10,
) -> Dict[str, Any]:
    """Run num_episodes without exploration; return metrics dict (same format as train_darts)."""
    n_agents = len(agents)
    workload = runtime_workload(env, cfg)
    if len(workload.node_ids) != 4:
        raise ValueError(
            f"SADCHER baseline supports only 4-node workloads (architectural "
            f"limitation of its IL-trained TaskEncoder). Got workload "
            f"{cfg.workload_id!r} with {len(workload.node_ids)} nodes: "
            f"{workload.node_ids}. Skip this (workload, sadcher) cell in "
            f"the sweep matrix or use a different baseline."
        )
    sadcher_node_ids = tuple(workload.node_ids)
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
    rho_hat_list: List[float] = []
    n_tot_jobs_list: List[int] = []

    for _ in range(num_episodes):
        ep_rewards, _, last_info, agent_died, energy_cert_violations = run_sadcher_episode(
            env,
            agents,
            cfg,
            reward_net,
            device,
            sadcher_node_ids,
            episode_verbose=False,
            explore=False,
        )[:5]
        for i in range(n_agents):
            if energy_cert_violations[i] > 0:
                crash_counts[i] += 1
            rewards_per_agent[i].append(ep_rewards[i])
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
            row[key_r] = float(np.mean(rewards_per_agent[i]))
            row[key_b] = float(np.mean(battery_pct_per_agent[i])) if battery_pct_per_agent[i] else float("nan")
            row[key_c] = 100.0 * crash_counts[i] / num_episodes
        else:
            row[key_r] = 0.0
            row[key_b] = float("nan")
            row[key_c] = 0.0
    for ck in CSV_KEYS:
        row["sub{}_miss_rate".format(ck)] = sub_misses[ck] / num_ep
        times = sub_times[ck]
        row["sub{}_time".format(ck)] = float(np.mean(times)) if times else float("nan")
    row["task_miss_rate"] = task_miss_count / num_ep
    row["rho_hat"] = float(np.nanmean(rho_hat_list)) if rho_hat_list else float("nan")
    row["n_full_fidelity"] = float(np.mean(n_ff_list)) if n_ff_list else float("nan")
    row["n_total_jobs"] = int(n_tot_jobs_list[0]) if n_tot_jobs_list else 0
    for i in range(4):
        row[f"rho_sp_A{i+1}"] = float("nan")
        row[f"omega_A{i+1}"] = float("nan")
    row["task_time"] = float(np.mean(task_times)) if task_times else float("nan")
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
    row["n_miss_n1_drop"] = float("nan")
    row["n_miss_n2_fallback_insufficient"] = float("nan")
    row["n_miss_n3_unreached"] = float("nan")
    row["n_miss_full_fidelity_late"] = float("nan")
    return row


def verify_sadcher_interface(cfg: SADCHER_CONFIG) -> None:
    """Run one episode and verify env.step(agents) compatibility: get_active_job, execute_task, complete_active_job, queue."""
    workload_v = get_workload(cfg.workload_id)
    if len(workload_v.node_ids) != 4:
        raise ValueError(
            f"SADCHER baseline supports only 4-node workloads (architectural "
            f"limitation of its IL-trained TaskEncoder). Got workload "
            f"{cfg.workload_id!r} with {len(workload_v.node_ids)} nodes: "
            f"{workload_v.node_ids}. Skip this (workload, sadcher) cell in "
            f"the sweep matrix or use a different baseline."
        )
    sadcher_node_ids = tuple(workload_v.node_ids)
    env = TurtleBot3Env(
        num_agents=cfg.num_agents,
        global_deadline_ms=cfg.global_deadline_ms,
        cfg=cfg,
        seed=cfg.random_seed,
        workload=workload_v,
        enable_rta_fallback=False,
    )
    dag = DAGGraph.from_workload(workload_v, cfg)
    agents = [
        SadcherAgent(agent_id=f"agent_{i+1}", cfg=cfg, logger=logging.getLogger(f"sadcher.agent_{i+1}"))
        for i in range(cfg.num_agents)
    ]
    for a in agents:
        a.dag_tracker = LocalDAGTracker(dag)
    reward_net = SadcherRewardNet(
        cfg, sadcher_node_ids, workload_v.predecessors
    ).to(torch.device("cpu"))
    device = torch.device("cpu")
    ep_rewards, _, last_info, agent_died, _, _ = run_sadcher_episode(
        env,
        agents,
        cfg,
        reward_net,
        device,
        sadcher_node_ids,
        episode_verbose=False,
        explore=False,
    )
    for i, agent in enumerate(agents):
        q = agent.queue
        assert isinstance(q, list), "queue must be a list"
        assert len(q) <= 1, "SADCHER: at most one active task per agent"
        if q:
            job = agent.get_active_job()
            assert job is not None
            assert hasattr(job, "job_id") and hasattr(job, "total_budget_ms"), "CommittedJob must have job_id, total_budget_ms"
            assert isinstance(job.job_id, str) and isinstance(job.total_budget_ms, (int, float))
    logging.getLogger("sadcher").info("SADCHER interface verification passed (get_active_job, queue, env.step).")


def main() -> None:
    parser = argparse.ArgumentParser(description="SADCHER baseline: MARL-only task allocation (no real-time guarantees).")
    parser.add_argument("--exp_name", type=str, default="sadcher_baseline", help="Experiment name (results/<exp_name>/).")
    parser.add_argument("exp_name_pos", nargs="*", help="Experiment name as positional (e.g. exp 1).")
    parser.add_argument("--num_episodes", type=int, default=20000)
    parser.add_argument("--eval_batch", type=int, default=10)
    parser.add_argument(
        "--eval_every",
        type=int,
        default=None,
        help="Override cfg.eval_every (CSV / eval cadence). Default: cfg.eval_every or 500.",
    )
    parser.add_argument("--eval_episodes", type=int, default=None, help="Override cfg.eval_episodes.")
    parser.add_argument("--skip_verify", action="store_true", help="Skip one-episode interface verification.")
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

    cfg = SADCHER_CONFIG()
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
    if args.eval_episodes is not None:
        cfg = dataclasses.replace(cfg, eval_episodes=args.eval_episodes)

    if not getattr(args, "skip_verify", False):
        verify_sadcher_interface(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logger = logging.getLogger("sadcher")
    logger.info("SADCHER baseline experiment: %s", exp_name)
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

    if len(workload.node_ids) != 4:
        raise ValueError(
            f"SADCHER baseline supports only 4-node workloads (architectural "
            f"limitation of its IL-trained TaskEncoder). Got workload "
            f"{cfg.workload_id!r} with {len(workload.node_ids)} nodes: "
            f"{workload.node_ids}. Skip this (workload, sadcher) cell in "
            f"the sweep matrix or use a different baseline."
        )
    SADCHER_NODE_IDS = tuple(workload.node_ids)

    log_path = os.path.join(_PROJECT_ROOT, cfg.results_dir, exp_name.replace(" ", "_"))
    os.makedirs(log_path, exist_ok=True)

    try:
        from torch.utils.tensorboard import SummaryWriter as _TBWriter
    except ImportError:
        _TBWriter = None  # type: ignore[misc, assignment]
    tb_writer = _TBWriter(log_dir=os.path.join(log_path, "tb")) if _TBWriter is not None else None

    with open(os.path.join(log_path, "sadcher_config.json"), "w") as f:
        json.dump({
            "method": "SADCHER",
            "architecture": "MARL-only (GAT + Transformer + Bipartite Matching)",
            "allocation": "learned reward prediction + relaxed bipartite matching",
            "real_time_guarantees": False,
            "feasibility_certificate": False,
            "edf_scheduling": False,
            "train_episodes": getattr(cfg, "train_episodes", 20000),
            "eval_every": getattr(cfg, "eval_every", 500),
            "eval_episodes": getattr(cfg, "eval_episodes", 10),
            "num_episodes": args.num_episodes,
            "eval_batch_size": args.eval_batch,
            "workload_id": cfg.workload_id,
        }, f, indent=2)

    env = TurtleBot3Env(
        num_agents=cfg.num_agents,
        global_deadline_ms=cfg.global_deadline_ms,
        cfg=cfg,
        seed=cfg.random_seed,
        workload=workload,
        enable_rta_fallback=False,
    )
    dag = DAGGraph.from_workload(workload, cfg)
    reward_net = SadcherRewardNet(cfg, SADCHER_NODE_IDS, workload.predecessors).to(device)
    optimizer = torch.optim.Adam(
        list(reward_net.parameters()),
        lr=getattr(cfg, "lr_reward_net", 3e-4),
    )
    agents: List[SadcherAgent] = []
    for i in range(cfg.num_agents):
        agent = SadcherAgent(
            agent_id=f"agent_{i + 1}",
            cfg=cfg,
            logger=logging.getLogger(f"sadcher.agent_{i + 1}"),
        )
        agent.dag_tracker = LocalDAGTracker(dag)
        agents.append(agent)

    train_episodes = getattr(cfg, "train_episodes", 20000)
    num_run = min(args.num_episodes, train_episodes) if args.num_episodes else train_episodes
    eval_every = args.eval_every if args.eval_every is not None else getattr(cfg, "eval_every", 500)
    save_interval = getattr(cfg, "save_interval", 500)
    eval_episodes = getattr(cfg, "eval_episodes", 10)
    D_G = getattr(cfg, "global_deadline_ms", 180000.0)
    num_batches = (num_run // eval_every) + (1 if num_run % eval_every != 0 else 0)
    batch_counter = 0

    for episode in range(1, num_run + 1):
        episode_number = episode
        verbose_this_episode = (episode_number % 500 == 0) or (episode_number <= 1)
        summary_this_episode = (episode_number % 100 == 0) or (episode_number <= 1)
        ep_rewards, _, last_info, agent_died, energy_cert_violations, trajectory = run_sadcher_episode(
            env,
            agents,
            cfg,
            reward_net,
            device,
            SADCHER_NODE_IDS,
            episode_verbose=verbose_this_episode,
            explore=True,
            episode_number=episode_number,
        )
        st = last_info.get("subtask_completion_times_ms") or {}
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

        if summary_this_episode:
            # Per-episode console output (match DARTS / CEA-EDF format)
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
                battery_pct = (curr_i / init_i) * 100.0
                print(
                    " Agent {} Reward: {:7.1f} Remaining Battery: {:5.1f}% |".format(
                        i + 1, ep_rewards[i], battery_pct,
                    ),
                    end="",
                )
            print()
            for node_id in ("A", "B", "C", "D"):
                t = st.get(node_id, None)
                print("  Subtask {} completion: {} ms".format(
                    node_id, f"{t:.0f}" if t is not None else "N/A"))
            task_time = st.get("D", None)
            if task_time is None:
                task_time = last_info.get("task_time_ms")
            if task_time is not None:
                print("  Overall task completion: {:.0f} ms".format(float(task_time)))
                miss = float(task_time) > D_G
                print("  End-to-end deadline miss: {}".format(miss))
            else:
                print("  Overall task completion: N/A (timeout)")
                print("  End-to-end deadline miss: True")
            print()
            if verbose_this_episode:
                rts = last_info.get("response_times_ms", {})
                H_ms = getattr(cfg, "epoch_length_ms", 1000.0)
                print("  --- Response Times (τ→commit) ---")
                for nid in ("A", "B", "C", "D"):
                    rt = rts.get(nid)
                    bound_ok = f"≤ H={H_ms:.0f}ms ✓" if rt is not None and rt <= H_ms else ("VIOLATION ✗" if rt is not None else "N/A")
                    print(f"  Subtask {nid} RT: {f'{rt:.0f} ms' if rt is not None else 'N/A':>10}  [{bound_ok}]")

        episode_return = float(np.sum(ep_rewards))
        update_reward_net_reinforce(reward_net, trajectory or [], episode_return, optimizer, cfg, device)

        if episode % eval_every == 0 or episode == num_run:
            batch_counter += 1
            batch = batch_counter
            metrics = evaluate_sadcher(env, agents, cfg, reward_net, device, num_episodes=eval_episodes)
            append_eval_to_csv(log_path, episode, metrics)
            logger.info(
                "SADCHER Batch %d/%d appended to %s/sadcher_log.csv | "
                "task_miss_rate=%.2f, crash=[%.1f%%, %.1f%%, %.1f%%, %.1f%%]",
                batch, num_batches, log_path,
                metrics.get("task_miss_rate", float("nan")),
                *[metrics.get(f"A{j+1}_crash_rate", 0.0) for j in range(4)],
            )

        if episode % save_interval == 0:
            # Save .pth only at first and final checkpoints
            if episode == save_interval or episode == num_run:
                ckpt_path = os.path.join(log_path, f"sadcher_reward_net_ep{episode}.pth")
                torch.save(reward_net.state_dict(), ckpt_path)
                logger.info("Checkpoint saved to %s", ckpt_path)

    if tb_writer is not None:
        tb_writer.close()
    logger.info("SADCHER training complete. Results in %s", log_path)


if __name__ == "__main__":
    main()
