"""Benchmark-specific stress profile lookup."""

from __future__ import annotations

from typing import Dict, Optional

_VALID_WORKLOADS = ("B1_CDR", "B2_LML", "B3_IQI", "B4_DEM")

_STRESS_TABLE: Dict[str, Dict[str, Optional[dict]]] = {
    "B1_CDR": {
        "light": {
            "global_deadline_ms": 200000.0,
            "battery_capacities_wh": [0.65, 0.95, 1.25, 1.55],
            "max_episode_steps": 250,
            "max_work_ms": 190000.0,
        },
        "medium": {
            "global_deadline_ms": 150000.0,
            "battery_capacities_wh": [0.50, 0.75, 1.00, 1.25],
            "max_episode_steps": 200,
            "max_work_ms": 142500.0,
        },
        "heavy": {
            "global_deadline_ms": 120000.0,
            "battery_capacities_wh": [0.40, 0.60, 0.80, 1.00],
            "max_episode_steps": 160,
            "max_work_ms": 114000.0,
        },
        "adversarial": {
            "global_deadline_ms": 80000.0,
            "battery_capacities_wh": [0.27, 0.40, 0.53, 0.67],
            "max_episode_steps": 110,
            "max_work_ms": 76000.0,
        },
        "energy_b04": {
            "global_deadline_ms": 150000.0,
            "battery_capacities_wh": [0.40, 0.60, 0.80, 1.00],
            "max_episode_steps": 200,
            "max_work_ms": 142500.0,
        },
        "energy_b03": {
            "global_deadline_ms": 150000.0,
            "battery_capacities_wh": [0.30, 0.45, 0.60, 0.75],
            "max_episode_steps": 200,
            "max_work_ms": 142500.0,
        },
        "energy_b02": {
            "global_deadline_ms": 150000.0,
            "battery_capacities_wh": [0.20, 0.30, 0.40, 0.50],
            "max_episode_steps": 200,
            "max_work_ms": 142500.0,
        },
    },
    "B2_LML": {
        "light": {
            "global_deadline_ms": 240000.0,
            "relative_deadline_ms": 200000.0,
            "battery_capacities_wh": [0.55, 0.80, 1.05, 1.30],
            "max_episode_steps": 280,
            "max_work_ms": 220000.0,
            "poisson_mean_ms": 45000.0,
        },
        "medium": {
            "global_deadline_ms": 240000.0,
            "relative_deadline_ms": 170000.0,
            "battery_capacities_wh": [0.55, 0.80, 1.05, 1.30],
            "max_episode_steps": 280,
            "max_work_ms": 220000.0,
            "poisson_mean_ms": 30000.0,
        },
        "heavy": {
            "global_deadline_ms": 240000.0,
            "relative_deadline_ms": 150000.0,
            "battery_capacities_wh": [0.55, 0.80, 1.05, 1.30],
            "max_episode_steps": 280,
            "max_work_ms": 220000.0,
            "poisson_mean_ms": 18000.0,
        },
    },
    "B3_IQI": {
        "light": None,
        "medium": {
            "global_deadline_ms": 180000.0,
            "relative_deadline_ms": 150000.0,
            "battery_capacities_wh": [0.65, 0.95, 1.25, 1.55],
            "max_episode_steps": 220,
            "max_work_ms": 170000.0,
        },
        "heavy": None,
    },
    "B4_DEM": {
        "light": None,
        "medium": {
            "global_deadline_ms": 100000.0,
            "battery_capacities_wh": [0.50, 0.75, 1.00, 1.25],
            "max_episode_steps": 160,
            "max_work_ms": 95000.0,
        },
        "heavy": None,
    },
}


def get_stress_profile(workload_id: str, stress_level: str) -> Optional[dict]:
    """
    Return stress-parameter overrides for a (workload, stress) pair.

    Returns None for unsupported stress levels in single-medium-only benchmarks.
    Raises ValueError for unknown workload IDs.
    """
    if workload_id not in _STRESS_TABLE:
        raise ValueError(
            f"Unknown workload_id {workload_id!r}. Valid IDs: {list(_VALID_WORKLOADS)}"
        )
    return _STRESS_TABLE[workload_id].get(stress_level)
