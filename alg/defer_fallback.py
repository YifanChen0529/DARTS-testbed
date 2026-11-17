"""
Algorithm 1 (lines 22–34): auction-time defer / fallback / drop state machine.
Section I-B-j and Eq. (27) in the DARTS draft.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from typing import Set

from envs.simulation_env import NODE_STATE_ASSIGNED

# Match alg.darts_agent.MAX_TRAVEL_TIME_MS when env has no estimate_dynamic_wcet_ms.
_MAX_TRAVEL_FALLBACK_MS = 65000.0

# EXP-2: fallback assignments skipped because sponsor != alloc(class_of(job)).
_FALLBACK_BLOCKED_COUNT = 0
_FALLBACK_BLOCKED_JOBS: Set[str] = set()


def record_fallback_blocked(job_id: str) -> None:
    global _FALLBACK_BLOCKED_COUNT
    _FALLBACK_BLOCKED_COUNT += 1
    _FALLBACK_BLOCKED_JOBS.add(str(job_id))


def fallback_blocked_count() -> int:
    return int(_FALLBACK_BLOCKED_COUNT)


def fallback_blocked_jobs() -> Set[str]:
    return set(_FALLBACK_BLOCKED_JOBS)


def reset_fallback_blocked() -> None:
    global _FALLBACK_BLOCKED_COUNT
    _FALLBACK_BLOCKED_COUNT = 0
    _FALLBACK_BLOCKED_JOBS.clear()


class FallbackOutcome(Enum):
    DEGRADABLE_DEFER = "degradable_defer"
    DEGRADABLE_FALLBACK_NOW = "degradable_fallback_now"
    NONDEGRADABLE_N1_DROP = "nondegradable_n1_drop"
    NONDEGRADABLE_N2_CASUALTY = "nondegradable_n2_casualty"
    SUCCESSFUL_AUCTION = "successful_auction"


def compute_fallback_cost_ms(
    sponsor_idx: int,
    job_id: str,
    sponsor_pos: Tuple[float, float],
    task_xy: Tuple[float, float],
    workload: Any,
    max_velocity_mps: float,
) -> Optional[float]:
    """
    Eq. (26): c_FB_v = tau_{sp(v),v} + C̃^FB_{sp(v),v} + h_{sp(v),v}.
    Returns None iff workload.fallback_wcet_ms[job_id] is None (case N1 trigger).
    """
    _ = sponsor_idx
    c_tilde_fb = workload.fallback_wcet_ms.get(job_id)
    if c_tilde_fb is None:
        return None
    dx = task_xy[0] - sponsor_pos[0]
    dy = task_xy[1] - sponsor_pos[1]
    dist_m = (dx * dx + dy * dy) ** 0.5
    tau_fb_ms = (dist_m / max_velocity_mps) * 1000.0
    overhead = float(getattr(workload, "fallback_overhead_ms", 50.0))
    return tau_fb_ms + float(c_tilde_fb) + overhead


def classify_no_bidder_outcome(
    job_id: str,
    t_k_ms: float,
    d_v_ms: float,
    epoch_length_ms: float,
    c_fb_v_ms: Optional[float],
    already_deferred_once: bool,
) -> FallbackOutcome:
    """
    Algorithm 1 lines 22-34. Action when no agent passed cert_seq at t_k.
    """
    _ = job_id
    if c_fb_v_ms is None:
        return FallbackOutcome.NONDEGRADABLE_N1_DROP
    if t_k_ms + c_fb_v_ms > d_v_ms:
        return FallbackOutcome.NONDEGRADABLE_N2_CASUALTY
    if not already_deferred_once and t_k_ms + epoch_length_ms + c_fb_v_ms <= d_v_ms:
        return FallbackOutcome.DEGRADABLE_DEFER
    return FallbackOutcome.DEGRADABLE_FALLBACK_NOW


def determine_sponsor_idx(
    job: Any,
    env: Any,
    feasible_indices: List[int],
    task_assigned_to: Dict[str, int],
    completed_tasks_ms: Dict[str, float],
) -> int:
    """Sponsor = agent that completed the last predecessor (same convention as winning-auction path)."""
    sponsor_idx = feasible_indices[0]
    preds = getattr(job, "predecessors", None) or []
    if preds:
        latest_pred_time = -1.0
        for pred_id in preds:
            pred_completion_time = completed_tasks_ms.get(pred_id, -1.0)
            pred_completer = task_assigned_to.get(pred_id)
            if (
                pred_completion_time > latest_pred_time
                and pred_completer is not None
                and pred_completer in feasible_indices
            ):
                latest_pred_time = pred_completion_time
                sponsor_idx = pred_completer
    return sponsor_idx


def execute_fallback_assign(
    agents: List[Any],
    global_tracker: Any,
    env: Any,
    cfg: Any,
    job: Any,
    sponsor_idx: int,
    c_tilde_fb: float,
    clock_ms: float,
    battery_wh: List[float],
    commit_time_ms: Dict[str, float],
    task_assigned_to: Dict[str, int],
    energy_cert_violations: List[int],
    release_time_ms: float = 0.0,
    episode_logger: Optional[logging.Logger] = None,
) -> None:
    """
    Assign job to sponsor with fallback WCET C̃^FB (degradable path). Excluded from V^ff (Eq. 28).
    """
    # EXP-2 partition gate. Return contract: this function returns None and
    # callers (train_darts.py DEGRADABLE_FALLBACK_NOW / NONDEGRADABLE_N2) do
    # not inspect the return value; "no assignment happened" is simply not
    # calling assign_job. When the sponsor is not alloc(class_of(job)) we
    # early-return None without mutating queues or trackers.
    from alg.darts_agent import alloc_map_is_active, alloc_target_for
    _job_id = getattr(job, "node_id", None)
    if alloc_map_is_active() and _job_id is not None:
        _alloc_tgt = alloc_target_for(_job_id)
        _sponsor_id = None
        if 0 <= sponsor_idx < len(agents):
            _sponsor_id = getattr(agents[sponsor_idx], "agent_id", None)
        if _alloc_tgt is not None and _sponsor_id is not None and _sponsor_id != _alloc_tgt:
            record_fallback_blocked(_job_id)
            log = episode_logger or logging.getLogger("episode_events")
            log.warning(
                "[EXP2][alloc] fallback blocked: job=%s sponsor=%s alloc_target=%s",
                _job_id, _sponsor_id, _alloc_tgt,
            )
            return None
    delta_max_ms = getattr(cfg, "delta_max_ms", 50.0)
    log = episode_logger or logging.getLogger("episode_events")
    if (
        env is not None
        and hasattr(env, "estimate_dynamic_wcet_ms")
        and job.node_id in getattr(env, "task_locations", {})
    ):
        _fb_target = env.task_locations[job.node_id]
        _fb_travel_exec = env.estimate_dynamic_wcet_ms(sponsor_idx, _fb_target, c_tilde_fb)
    else:
        _fb_travel_exec = c_tilde_fb + _MAX_TRAVEL_FALLBACK_MS
    agents[sponsor_idx].assign_job(
        job,
        release_time_ms=release_time_ms,
        current_time_ms=clock_ms,
        dynamic_budget_ms=_fb_travel_exec + delta_max_ms,
        override_remaining_work_ms=c_tilde_fb,
    )
    global_tracker.node_states[job.node_id] = NODE_STATE_ASSIGNED
    commit_time_ms[job.node_id] = clock_ms
    task_assigned_to[job.node_id] = sponsor_idx
    _fb_budget_ms = _fb_travel_exec + delta_max_ms
    e_safe = getattr(cfg, "e_safe_wh", 0.05)
    battery_now = env._battery_wh[sponsor_idx] if sponsor_idx < len(env._battery_wh) else 0.0
    delta_e_wc = (_fb_budget_ms / 3600000.0) * getattr(cfg, "p_moving_w", 7.5)
    if battery_now < e_safe + delta_e_wc:
        energy_cert_violations[sponsor_idx] += 1
        log.warning(
            "[Time: %.0fms] ⚡ Agent %d ENERGY CERT VIOLATION: B=%.4f Wh < e_safe(%.4f) + ΔE_WC(%.4f)",
            clock_ms,
            sponsor_idx + 1,
            battery_now,
            e_safe,
            delta_e_wc,
        )
