"""
CPS benchmark workloads for the DARTS framework.

This module defines four application-scenario workloads used as reproducible CPS
benchmarks. Each maps a DAG topology archetype to a real-world CPS application
domain, while all four are evaluated on the **same** physical testbed: four
TurtleBot3 platforms and four Jetson Orin Nano edge nodes, operating on a
**10 m × 10 m** map.

| workload_id | display_name                         | application_archetype              | DAG topology archetype   |
|-------------|--------------------------------------|------------------------------------|---------------------------|
| B1_CDR      | Coordinated Disaster Response        | Search-and-Rescue Robotics         | Diamond fork-join         |
| B2_LML      | Last-Mile Autonomous Logistics       | Autonomous-Vehicle                 | Multi-DAG sporadic        |
| B3_IQI      | Industrial Quality-Inspection Pipeline | Industry-4.0                   | Linear chain              |
| B4_DEM      | Distributed Environmental Monitoring | Precision-Agriculture / Wildfire-Sensing | Independent parallel |

Topology archetypes are fixed per benchmark; scheduling and allocation policies
are evaluated against identical hardware and spatial footprints across scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

# DBench-4 fallback WCETs: C̃^FB_v = 0.4 * C̃_v (full WCET). By design every registered node has a
# measurable fallback so Algorithm 1 case N1 (undefined fallback) is structurally empty for these
# benchmarks — this minimizes nondegradable jobs. Case N2 (fallback insufficient: t_k + c_FB > d_v)
# still arises when an auction fails very late in an episode, near the deadline horizon.
DB4_FALLBACK_WCET_SCALE = 0.4


@dataclass(frozen=True)
class Workload:
    workload_id: str
    display_name: str
    application_archetype: str
    node_ids: Tuple[str, ...]
    node_names: Dict[str, str]
    predecessors: Dict[str, List[str]]
    task_wcets_ms: Dict[str, float]
    task_locations_m: Dict[str, Tuple[float, float]]
    release_pattern: str
    release_params: Dict[str, Any]
    sponsorship_model: str
    relative_deadline_ms: float
    global_deadline_ms: float
    fallback_wcet_ms: Dict[str, Optional[float]] = field(default_factory=dict)
    """Per-job measurable C̃^FB. None means fallback is undefined (Algorithm 1 case N1)."""

    fallback_overhead_ms: float = 50.0
    """Communication overhead h for fallback execution (aligned with Δ_max for safety)."""

    def __post_init__(self) -> None:
        valid_patterns = {"aperiodic_single_shot", "sporadic_poisson"}
        if self.release_pattern not in valid_patterns:
            raise ValueError(
                f"Workload {self.workload_id!r}: unsupported release_pattern "
                f"{self.release_pattern!r}. Expected one of {sorted(valid_patterns)}."
            )
        if self.release_pattern == "sporadic_poisson":
            if "poisson_mean_ms" not in self.release_params:
                raise ValueError(
                    f"Workload {self.workload_id!r}: release_params must include "
                    "'poisson_mean_ms' when release_pattern='sporadic_poisson'."
                )


def _validate_workload(w: Workload) -> None:
    """Ensure graph keys, predecessor refs, and acyclicity."""
    node_set = set(w.node_ids)

    for nid in w.node_ids:
        if nid not in w.predecessors:
            raise ValueError(
                f"Workload {w.workload_id!r}: node_id {nid!r} missing from "
                f"`predecessors` keys."
            )
        if nid not in w.task_wcets_ms:
            raise ValueError(
                f"Workload {w.workload_id!r}: node_id {nid!r} missing from "
                f"`task_wcets_ms` keys."
            )
        if nid not in w.task_locations_m:
            raise ValueError(
                f"Workload {w.workload_id!r}: node_id {nid!r} missing from "
                f"`task_locations_m` keys."
            )

    for mapping_name, mapping in (
        ("predecessors", w.predecessors),
        ("task_wcets_ms", w.task_wcets_ms),
        ("task_locations_m", w.task_locations_m),
    ):
        for key in mapping:
            if key not in node_set:
                raise ValueError(
                    f"Workload {w.workload_id!r}: key {key!r} in `{mapping_name}` "
                    f"is not listed in `node_ids`."
                )

    for nid, preds in w.predecessors.items():
        for p in preds:
            if p not in node_set:
                raise ValueError(
                    f"Workload {w.workload_id!r}: predecessor {p!r} of node "
                    f"{nid!r} is not a valid `node_id`."
                )
            if p == nid:
                raise ValueError(
                    f"Workload {w.workload_id!r}: self-loop on node {nid!r} is not "
                    f"allowed."
                )

    # Kahn topological sort — cycle detection on predecessor edges (p -> nid).
    in_degree: Dict[str, int] = {nid: 0 for nid in w.node_ids}
    for nid in w.node_ids:
        for p in w.predecessors[nid]:
            in_degree[nid] += 1

    queue: List[str] = [nid for nid in w.node_ids if in_degree[nid] == 0]
    visited = 0
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        visited += 1
        for v in w.node_ids:
            if u in w.predecessors[v]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

    if visited != len(w.node_ids):
        raise ValueError(
            f"Workload {w.workload_id!r}: task graph is not acyclic (cycle "
            f"detected)."
        )


WORKLOAD_B1_CDR = Workload(
    workload_id="B1_CDR",
    display_name="Coordinated Disaster Response",
    application_archetype="Search-and-Rescue Robotics",
    node_ids=("A", "B", "C", "D"),
    node_names={
        "A": "Reconnaissance",
        "B": "Hazard_Clearance",
        "C": "Casualty_Stabilization",
        "D": "Coordinated_Evacuation",
    },
    predecessors={
        "A": [],
        "B": ["A"],
        "C": ["A"],
        "D": ["B", "C"],
    },
    task_wcets_ms={
        "A": 15000.0,
        "B": 25000.0,
        "C": 25000.0,
        "D": 30000.0,
    },
    task_locations_m={
        "A": (5.0, 5.0),
        "B": (8.5, 8.5),
        "C": (1.5, 8.5),
        "D": (5.0, 1.5),
    },
    release_pattern="aperiodic_single_shot",
    release_params={},
    sponsorship_model="first_detector",
    relative_deadline_ms=150000.0,
    global_deadline_ms=150000.0,
)

WORKLOAD_B2_LML = Workload(
    workload_id="B2_LML",
    display_name="Last-Mile Autonomous Logistics",
    application_archetype="Autonomous-Vehicle",
    node_ids=("Pk", "Hf", "Dr", "Cf"),
    node_names={
        "Pk": "Pickup",
        "Hf": "Handoff",
        "Dr": "Dropoff",
        "Cf": "Confirm",
    },
    predecessors={
        "Pk": [],
        "Hf": ["Pk"],
        "Dr": ["Hf"],
        "Cf": ["Dr"],
    },
    task_wcets_ms={
        "Pk": 12000.0,
        "Hf": 18000.0,
        "Dr": 18000.0,
        "Cf": 12000.0,
    },
    task_locations_m={
        "Pk": (1.0, 5.0),
        "Hf": (4.0, 5.0),
        "Dr": (7.0, 5.0),
        "Cf": (9.0, 5.0),
    },
    release_pattern="sporadic_poisson",
    release_params={
        "poisson_mean_ms": 30000.0,
        "max_jobs": 3,
        "first_release_ms": 0.0,
    },
    sponsorship_model="per_job_distinct",
    relative_deadline_ms=200000.0,
    global_deadline_ms=240000.0,
)

WORKLOAD_B3_IQI = Workload(
    workload_id="B3_IQI",
    display_name="Sequential Material Inspection",
    application_archetype="Warehouse Pre-Shipment Quality Assurance",
    node_ids=("A", "B", "C", "D"),
    node_names={
        "A": "Receive_and_Stage",
        "B": "Dimensional_Scan",
        "C": "Defect_Imaging",
        "D": "Final_Sign_Off",
    },
    predecessors={
        "A": [],
        "B": ["A"],
        "C": ["B"],
        "D": ["C"],
    },
    task_wcets_ms={
        "A": 10000.0,
        "B": 15000.0,
        "C": 20000.0,
        "D": 15000.0,
    },
    task_locations_m={
        "A": (1.5, 5.0),
        "B": (4.0, 5.0),
        "C": (7.0, 5.0),
        "D": (9.0, 5.0),
    },
    release_pattern="aperiodic_single_shot",
    release_params={},
    sponsorship_model="deterministic_intake",
    relative_deadline_ms=180000.0,
    global_deadline_ms=180000.0,
)

WORKLOAD_B4_DEM = Workload(
    workload_id="B4_DEM",
    display_name="Distributed Field Survey",
    application_archetype="Multi-Site Environmental Sampling",
    node_ids=("J1", "J2", "J3", "J4"),
    node_names={
        "J1": "Air_Quality_Sample",
        "J2": "Soil_Moisture_Sample",
        "J3": "Water_Turbidity_Sample",
        "J4": "Acoustic_Sensing",
    },
    predecessors={"J1": [], "J2": [], "J3": [], "J4": []},
    task_wcets_ms={"J1": 18000.0, "J2": 18000.0, "J3": 18000.0, "J4": 18000.0},
    task_locations_m={
        "J1": (1.5, 8.5),
        "J2": (8.5, 8.5),
        "J3": (1.5, 1.5),
        "J4": (8.5, 1.5),
    },
    release_pattern="aperiodic_single_shot",
    release_params={},
    sponsorship_model="round_robin",
    relative_deadline_ms=100000.0,
    global_deadline_ms=100000.0,
)

def _with_db4_fallbacks(w: Workload) -> Workload:
    """Attach measurable fallback WCETs (0.4 × full WCET) for every node in ``w``."""
    fb: Dict[str, Optional[float]] = {
        nid: DB4_FALLBACK_WCET_SCALE * w.task_wcets_ms[nid] for nid in w.node_ids
    }
    return replace(w, fallback_wcet_ms=fb, fallback_overhead_ms=50.0)


WORKLOAD_REGISTRY: Dict[str, Workload] = {
    "B1_CDR": _with_db4_fallbacks(WORKLOAD_B1_CDR),
    "B2_LML": _with_db4_fallbacks(WORKLOAD_B2_LML),
    "B3_IQI": _with_db4_fallbacks(WORKLOAD_B3_IQI),
    "B4_DEM": _with_db4_fallbacks(WORKLOAD_B4_DEM),
}


def get_workload(workload_id: str) -> Workload:
    """Return the benchmark workload for ``workload_id``."""
    if workload_id not in WORKLOAD_REGISTRY:
        valid_ids = sorted(WORKLOAD_REGISTRY)
        raise KeyError(
            f"Unknown workload_id {workload_id!r}. "
            f"Valid workload_ids: {valid_ids}"
        )
    return WORKLOAD_REGISTRY[workload_id]


def runtime_workload(env: Any, cfg: Any) -> Workload:
    """Return the workload actually attached to an environment.

    CLI overrides such as ``--relative_deadline_ms`` are applied to the
    Workload object before the environment is created, so episode-level DAGs
    must prefer ``env.workload`` over the registry default.
    """
    workload = getattr(env, "workload", None)
    if isinstance(workload, Workload):
        return workload
    return get_workload(getattr(cfg, "workload_id", "B1_CDR"))


for _wl in WORKLOAD_REGISTRY.values():
    _validate_workload(_wl)
