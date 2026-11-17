"""TB-3 mission instance: the (workload, cfg) pair the runtime would actually see.

Torch-free. Imports only stdlib, numpy, ``envs.workloads``, ``alg.stress_profiles``,
``alg.config_darts``. Never import ``envs.turtlebot3_spatial_env``.

The ``runtime_task_wcets_ms`` field exists because ``DAGGraph.from_workload``
merges ``cfg.task_wcets`` over ``workload.task_wcets_ms`` (T0 H3). That merge is
reproduced here and must not be "fixed" on the nominal (unseeded) path.

A seeded instance (``--instance_seed``) jitters DECLARED workload WCETs, writes
the same dict to both ``workload.task_wcets_ms`` and ``cfg.task_wcets``, and so
makes the from_workload merge the identity — including on B3, which therefore
runs at the declared-centered regime rather than the T0 H3 B1-override. B3 has
no checkpoints and is offline-only in TB-3.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from alg.config_darts import DARTS_CONFIG
from alg.stress_profiles import get_stress_profile
from envs.workloads import Workload, get_workload

# +/-20% around the declared profile WCET (TB-3 instance axis spec).
WCET_JITTER_LO = 0.8
WCET_JITTER_HI = 1.2

# Mix into SeedSequence so the same integer seed is not shared across workloads.
WORKLOAD_CODES = {"B1_CDR": 1, "B2_LML": 2, "B3_IQI": 3, "B4_DEM": 4}

# Independent field streams: one SeedSequence slot so resampling locations
# cannot shift WCET or battery draws.
FIELD_WCET, FIELD_LOC, FIELD_BATTERY = 1, 2, 3

# Duplicate of TurtleBot3Env map / default obstacles so this module never
# imports the torch-pulling env. Locked by tests/test_tb3_instance_gen.py g4
# against envs/turtlebot3_spatial_env.py:214-220 and MAP_SIZE_M.
TB3_MAP_SIZE_M = 10.0
TB3_OBSTACLES_M = (
    (3.0, 3.0, 1.5, 1.5),
    (5.0, 6.0, 1.0, 2.0),
    (7.0, 2.0, 1.2, 1.2),
)
LOC_SAMPLE_MAX_TRIES = 200

NOMINAL_CELLS: Tuple[str, ...] = (
    "B1_CDR:light",
    "B1_CDR:medium",
    "B1_CDR:heavy",
    "B2_LML:medium",
    "B3_IQI:medium",
    "B4_DEM:medium",
)

_PROFILE_STRESS_CANDIDATES: Tuple[str, ...] = (
    "light",
    "medium",
    "heavy",
    "adversarial",
    "energy_b04",
    "energy_b03",
    "energy_b02",
)


def effective_task_wcets(
    workload: Workload, cfg_task_wcets: Optional[Dict[str, float]]
) -> Dict[str, float]:
    """Reproduce the WCET merge ``DAGGraph.from_workload`` performs.

    Exact operations of ``envs/dag_task.py:106-110``:
    ``wcets = dict(workload.task_wcets_ms)`` then, when ``cfg_task_wcets`` is
    truthy, ``wcets = {**wcets, **cfg_task_wcets}``.

    T0 H3: with ``DARTS_CONFIG().task_wcets`` (B1 keys A–D) this changes B3_IQI
    to 15000/25000/25000/30000 (sum 95000) and leaves B1/B2/B4 unchanged.
    This is pre-existing runtime behaviour; do not "fix" it here.
    """
    wcets: Dict[str, float] = dict(workload.task_wcets_ms)
    if cfg_task_wcets:
        wcets = {**wcets, **cfg_task_wcets}
    return wcets


@dataclass(frozen=True)
class MissionInstance:
    """One (benchmark, stress[, seed, beta]) mission the offline tests evaluate.

    ``workload`` is the object the runtime would receive after the stress-profile
    relative-deadline override (B2/B3 declare that key; B1/B4 do not — T0 §2.3,
    ``scripts/sweep_tb1.py:58-64``).

    ``global_deadline_ms`` is D_e2e after profile and beta: this is what
    ``task_miss_rate`` compares against (T0 §2.3).

    ``relative_deadline_ms`` is D_rel: per-job ``d_v - r_v`` after profile and beta.
    """

    benchmark: str
    stress: str
    instance_seed: Optional[int]
    beta: float
    workload: Workload
    declared_task_wcets_ms: Dict[str, float]
    runtime_task_wcets_ms: Dict[str, float]
    global_deadline_ms: float
    relative_deadline_ms: float
    battery_capacities_wh: Tuple[float, ...]
    max_work_ms: float
    max_episode_steps: int

    def task_wcets(self, xi_source: str) -> Dict[str, float]:
        if xi_source == "declared":
            return dict(self.declared_task_wcets_ms)
        if xi_source == "runtime":
            return dict(self.runtime_task_wcets_ms)
        raise ValueError(
            f"xi_source must be 'declared' or 'runtime', got {xi_source!r}"
        )

    def scaled(self, beta: float) -> "MissionInstance":
        """Multiply BOTH deadlines, mirroring ``alg/train_darts.py:1806-1812``.

        Deadlines on the returned instance are already beta-scaled relative to
        ``self``. ``evaluate`` must compare ``R`` against those stored deadlines
        and must not multiply by ``beta`` again.
        """
        factor = float(beta)
        wl = replace(
            self.workload,
            relative_deadline_ms=self.workload.relative_deadline_ms * factor,
        )
        return MissionInstance(
            benchmark=self.benchmark,
            stress=self.stress,
            instance_seed=self.instance_seed,
            beta=self.beta * factor,
            workload=wl,
            declared_task_wcets_ms=dict(self.declared_task_wcets_ms),
            runtime_task_wcets_ms=dict(self.runtime_task_wcets_ms),
            global_deadline_ms=self.global_deadline_ms * factor,
            relative_deadline_ms=self.relative_deadline_ms * factor,
            battery_capacities_wh=self.battery_capacities_wh,
            max_work_ms=self.max_work_ms,
            max_episode_steps=self.max_episode_steps,
        )

    def cell(self) -> str:
        return f"{self.benchmark}:{self.stress}"


def parse_cell(cell: str) -> Tuple[str, str]:
    if ":" not in cell:
        raise ValueError(f"cell must be 'BENCH:stress', got {cell!r}")
    bench, stress = cell.split(":", 1)
    return bench.strip(), stress.strip()


def iter_profiled_cells() -> List[Tuple[str, str]]:
    """Every (benchmark, stress) for which ``get_stress_profile`` returns a dict."""
    out: List[Tuple[str, str]] = []
    for bench in ("B1_CDR", "B2_LML", "B3_IQI", "B4_DEM"):
        for stress in _PROFILE_STRESS_CANDIDATES:
            profile = get_stress_profile(bench, stress)
            if profile is not None:
                out.append((bench, stress))
    return out


def nominal_instance(benchmark: str, stress: str) -> MissionInstance:
    """Mirror ``scripts/sweep_tb1.py`` splat + ``_run_training`` at beta=1, no seed.

    ``profile = get_stress_profile(benchmark, stress)`` (raise when None);
    ``global_deadline_ms = profile["global_deadline_ms"]``;
    relative deadline replaced only when the profile declares that key
    (B2/B3 do; B1/B4 do not).
    """
    profile = get_stress_profile(benchmark, stress)
    if profile is None:
        raise ValueError(
            f"no stress profile for {benchmark}:{stress} "
            "(single-medium-only benchmark, or unknown stress)"
        )
    cfg = DARTS_CONFIG()
    global_deadline_ms = float(profile["global_deadline_ms"])
    workload = get_workload(benchmark)
    if "relative_deadline_ms" in profile:
        workload = replace(
            workload, relative_deadline_ms=float(profile["relative_deadline_ms"])
        )
    batteries = tuple(float(x) for x in profile["battery_capacities_wh"])
    return MissionInstance(
        benchmark=benchmark,
        stress=stress,
        instance_seed=None,
        beta=1.0,
        workload=workload,
        declared_task_wcets_ms=dict(workload.task_wcets_ms),
        runtime_task_wcets_ms=effective_task_wcets(workload, cfg.task_wcets),
        global_deadline_ms=global_deadline_ms,
        relative_deadline_ms=float(workload.relative_deadline_ms),
        battery_capacities_wh=batteries,
        max_work_ms=float(profile["max_work_ms"]),
        max_episode_steps=int(profile["max_episode_steps"]),
    )


# ---------------------------------------------------------------------------
# Deterministic instance generator (private RNG; no np/torch/random globals)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InstanceFields:
    task_wcets_ms: Dict[str, float]
    task_locations_m: Dict[str, Tuple[float, float]]
    battery_capacities_wh: Tuple[float, ...]


def _field_rng(
    workload_id: str, field_code: int, instance_seed: int
) -> np.random.Generator:
    """Private RNG for one (workload, field, instance_seed) stream.

    Field streams are independent by construction: the SeedSequence is
    ``[WORKLOAD_CODES[wid], field_code, instance_seed]``, so changing how one
    field samples cannot shift the others. The same ``instance_seed`` shares
    underlying uniforms across stress levels (battery bounds differ by
    profile; the raw draws do not).
    """
    if workload_id not in WORKLOAD_CODES:
        raise KeyError(
            f"unknown workload_id {workload_id!r}; known: {sorted(WORKLOAD_CODES)}"
        )
    return np.random.default_rng(
        [WORKLOAD_CODES[workload_id], int(field_code), int(instance_seed)]
    )


def _point_in_obstacle(x: float, y: float) -> bool:
    """Same predicate as ``TurtleBot3Env._collides_obstacles`` / ``_point_in_rect``.

    Strict (open) boundaries: ``rx < x < rx+w and ry < y < ry+h`` so points on
    obstacle edges are not inside (``envs/turtlebot3_spatial_env.py:78-80,246-249``).
    """
    for ox, oy, w, h in TB3_OBSTACLES_M:
        if ox < x < ox + w and oy < y < oy + h:
            return True
    return False


def generate_instance_fields(
    workload: Workload,
    battery_bounds_wh: Sequence[float],
    num_agents: int,
    instance_seed: int,
) -> InstanceFields:
    """Draw jittered WCETs, workspace locations, and initial batteries.

    Pure: uses only ``np.random.default_rng`` streams from ``_field_rng``; never
    touches the ``np.random`` / ``random`` globals.

    WCET jitter centers on the DECLARED ``workload.task_wcets_ms`` (not the
    T0 H3 runtime merge). On B3 a seeded instance therefore runs at the
    declared-centered regime, unlike the nominal B3 runtime. B3 has no
    checkpoints and is offline-only in TB-3.
    """
    rng_wcet = _field_rng(workload.workload_id, FIELD_WCET, instance_seed)
    rng_loc = _field_rng(workload.workload_id, FIELD_LOC, instance_seed)
    rng_bat = _field_rng(workload.workload_id, FIELD_BATTERY, instance_seed)

    task_wcets_ms: Dict[str, float] = {}
    for nid in workload.node_ids:
        task_wcets_ms[nid] = float(workload.task_wcets_ms[nid]) * float(
            rng_wcet.uniform(WCET_JITTER_LO, WCET_JITTER_HI)
        )

    task_locations_m: Dict[str, Tuple[float, float]] = {}
    for nid in workload.node_ids:
        x = y = 0.0
        found = False
        for _ in range(LOC_SAMPLE_MAX_TRIES):
            x = float(rng_loc.uniform(0.0, TB3_MAP_SIZE_M))
            y = float(rng_loc.uniform(0.0, TB3_MAP_SIZE_M))
            if not _point_in_obstacle(x, y):
                found = True
                break
        if not found:
            raise RuntimeError(
                f"location sample failed for node {nid!r} after "
                f"{LOC_SAMPLE_MAX_TRIES} tries (seed={instance_seed})"
            )
        task_locations_m[nid] = (x, y)

    bounds = [float(b) for b in battery_bounds_wh]
    lo, hi = min(bounds), max(bounds)
    batteries = tuple(
        float(rng_bat.uniform(lo, hi)) for _ in range(int(num_agents))
    )
    return InstanceFields(
        task_wcets_ms=task_wcets_ms,
        task_locations_m=task_locations_m,
        battery_capacities_wh=batteries,
    )


def apply_instance_seed(inst: MissionInstance, instance_seed: int) -> MissionInstance:
    """Harness path: jitter WCETs/locations/batteries; deadlines and max_work_ms stay.

    Both ``declared_task_wcets_ms`` and ``runtime_task_wcets_ms`` become the
    jittered dict (the runtime writes ``cfg.task_wcets`` to the same dict, so
    the from_workload merge is the identity). ``max_work_ms`` stays the
    training-time normalization constant so a frozen policy's feature scaling
    is untouched (T0 §3.5). Deadline tightness is the beta axis.
    """
    fields = generate_instance_fields(
        inst.workload,
        inst.battery_capacities_wh,
        len(inst.battery_capacities_wh),
        instance_seed,
    )
    jittered = dict(fields.task_wcets_ms)
    wl = replace(
        inst.workload,
        task_wcets_ms=jittered,
        task_locations_m=dict(fields.task_locations_m),
    )
    return MissionInstance(
        benchmark=inst.benchmark,
        stress=inst.stress,
        instance_seed=int(instance_seed),
        beta=inst.beta,
        workload=wl,
        declared_task_wcets_ms=dict(jittered),
        runtime_task_wcets_ms=dict(jittered),
        global_deadline_ms=inst.global_deadline_ms,
        relative_deadline_ms=inst.relative_deadline_ms,
        battery_capacities_wh=tuple(fields.battery_capacities_wh),
        max_work_ms=inst.max_work_ms,
        max_episode_steps=inst.max_episode_steps,
    )


def apply_instance_seed_runtime(
    workload: Workload, cfg: DARTS_CONFIG, instance_seed: int
) -> Tuple[Workload, DARTS_CONFIG]:
    """Runtime path: same ``generate_instance_fields`` as the harness.

    Writing BOTH ``workload.task_wcets_ms`` and ``cfg.task_wcets`` is mandatory
    (T0 §9: the from_workload merge and the SADCHER getattr fallback otherwise
    ignore the jitter).
    """
    fields = generate_instance_fields(
        workload,
        tuple(cfg.battery_capacities_wh),
        cfg.num_agents,
        instance_seed,
    )
    jittered = dict(fields.task_wcets_ms)
    new_wl = replace(
        workload,
        task_wcets_ms=jittered,
        task_locations_m=dict(fields.task_locations_m),
    )
    new_cfg = replace(
        cfg,
        task_wcets=dict(jittered),
        battery_capacities_wh=list(fields.battery_capacities_wh),
    )
    return new_wl, new_cfg
