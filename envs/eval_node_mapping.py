"""Map workload DAG node IDs onto fixed eval CSV columns (subA–subD)."""

from __future__ import annotations

from typing import Dict, Iterable, Set

from envs.workloads import Workload


def csv_subtask_slots(workload: Workload) -> tuple[str | None, str | None, str | None, str | None]:
    """Map workload node_ids onto the fixed (subA, subB, subC, subD) CSV columns.

    Strategy: take topological-prefix node_ids; if fewer than 4, pad with None.
    For B1_CDR / B3_IQI this returns ("A","B","C","D"). For B4_DEM ("J1","J2","J3","J4").
    For B2_LML ("Pk1","Hf1","Dr1","Pk2") — semantically the first DAG's three stages plus
    the second DAG's pickup. This is acceptable for the fixed-schema CSV; B2 visualization
    (darts_vis_B2.py) reads task_miss_rate / rho_hat / task_time / max_rt_ms only and is
    unaffected by which 4 nodes get logged in the sub* columns.
    """
    ids = workload.node_ids
    slots = list(ids[:4])
    while len(slots) < 4:
        slots.append(None)
    return tuple(slots)  # type: ignore[return-value]


def update_ready_times_on_completion(
    workload: Workload,
    newly_completed: Iterable[str],
    completed_set: Set[str],
    clock_ms: float,
    ready_time_ms: Dict[str, float],
) -> None:
    """DAG-topology-driven τ(v) bookkeeping when nodes finish.

    For each successor of a newly-completed node, if all of its predecessors are in
    ``completed_set``, set ``ready_time_ms[succ] = clock_ms`` (skip if ``succ`` is
    already in ``ready_time_ms`` — idempotent).

    Source nodes (no predecessors) must be seeded by the caller at episode reset
    with ``ready_time_ms[src] = 0.0``.
    """
    succ_map: Dict[str, list[str]] = {nid: [] for nid in workload.node_ids}
    for nid in workload.node_ids:
        for p in workload.predecessors[nid]:
            succ_map[p].append(nid)

    for done in newly_completed:
        for succ in succ_map.get(done, []):
            if succ in ready_time_ms:
                continue
            if all(p in completed_set for p in workload.predecessors[succ]):
                ready_time_ms[succ] = float(clock_ms)


def csv_task_time_ms(last_info: dict) -> float | None:
    """Use the env's already-correct sink-aggregated task_time_ms (max over sink_node_ids).

    Fall back to None if the env did not publish it.
    """
    v = last_info.get("task_time_ms", None)
    return None if v is None else float(v)
