"""TB-3 instance generator: purity, stream isolation, validity, dual-path agreement.

g4 documents the obstacle-duplicate against ``TurtleBot3Env._default_obstacles``.
If that env method changes, this test must be updated together with
``alg.tb3_instance.TB3_OBSTACLES_M``.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from dataclasses import replace as dc_replace

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pytest

from alg.config_darts import DARTS_CONFIG
from alg.stress_profiles import get_stress_profile
from alg.tb3_instance import (
    NOMINAL_CELLS,
    TB3_MAP_SIZE_M,
    TB3_OBSTACLES_M,
    WCET_JITTER_HI,
    WCET_JITTER_LO,
    _point_in_obstacle,
    apply_instance_seed,
    apply_instance_seed_runtime,
    effective_task_wcets,
    generate_instance_fields,
    nominal_instance,
    parse_cell,
)
from envs.workloads import get_workload
from scripts.offline_test_harness import XI_SOURCE_DEFAULT, build_parser


def _fields_as_dicts(fields):
    return {
        "wcets": dict(fields.task_wcets_ms),
        "locs": {k: list(v) for k, v in fields.task_locations_m.items()},
        "bat": list(fields.battery_capacities_wh),
    }


def _np_state_eq(a, b) -> bool:
    return (
        a[0] == b[0]
        and np.array_equal(a[1], b[1])
        and a[2:] == b[2:]
    )


def test_g1_purity_and_determinism():
    wl = get_workload("B1_CDR")
    bounds = (0.4, 0.6, 0.8, 1.0)
    a = generate_instance_fields(wl, bounds, 4, 7)
    b = generate_instance_fields(wl, bounds, 4, 7)
    assert _fields_as_dicts(a) == _fields_as_dicts(b)

    np.random.seed(12345)
    random.seed(12345)
    np_before = np.random.get_state()
    py_before = random.getstate()
    generate_instance_fields(wl, bounds, 4, 99)
    assert _np_state_eq(np.random.get_state(), np_before)
    assert random.getstate() == py_before

    code = (
        "import json, sys, os; sys.path.insert(0, os.environ['PYTHONPATH']); "
        "from envs.workloads import get_workload; "
        "from alg.tb3_instance import generate_instance_fields; "
        "f = generate_instance_fields(get_workload('B1_CDR'), (0.4, 0.6, 0.8, 1.0), 4, 7); "
        "print(json.dumps({"
        "'wcets': dict(f.task_wcets_ms), "
        "'locs': {k: list(v) for k, v in f.task_locations_m.items()}, "
        "'bat': list(f.battery_capacities_wh)"
        "}))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = _PROJECT_ROOT
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    remote = json.loads(proc.stdout)
    local = _fields_as_dicts(a)
    assert remote["wcets"] == local["wcets"]
    assert remote["locs"] == local["locs"]
    assert remote["bat"] == local["bat"]


def test_g2_stream_isolation():
    wl = get_workload("B1_CDR")
    med = tuple(get_stress_profile("B1_CDR", "medium")["battery_capacities_wh"])
    hvy = tuple(get_stress_profile("B1_CDR", "heavy")["battery_capacities_wh"])
    a = generate_instance_fields(wl, med, 4, 7)
    b = generate_instance_fields(wl, hvy, 4, 7)
    assert a.task_wcets_ms == b.task_wcets_ms
    assert a.task_locations_m == b.task_locations_m
    assert a.battery_capacities_wh != b.battery_capacities_wh

    lo_a, hi_a = min(med), max(med)
    lo_b, hi_b = min(hvy), max(hvy)
    u_a = tuple((x - lo_a) / (hi_a - lo_a) for x in a.battery_capacities_wh)
    u_b = tuple((x - lo_b) / (hi_b - lo_b) for x in b.battery_capacities_wh)
    assert u_a == pytest.approx(u_b)

    c = generate_instance_fields(wl, med, 4, 8)
    assert c.task_wcets_ms != a.task_wcets_ms
    assert c.task_locations_m != a.task_locations_m
    assert c.battery_capacities_wh != a.battery_capacities_wh


def test_g3_validity_seeds_0_to_199():
    for bench in ("B1_CDR", "B2_LML", "B3_IQI", "B4_DEM"):
        wl = get_workload(bench)
        profile = get_stress_profile(bench, "medium")
        bounds = tuple(profile["battery_capacities_wh"])
        lo, hi = min(bounds), max(bounds)
        n = len(bounds)
        for seed in range(200):
            f = generate_instance_fields(wl, bounds, n, seed)
            for nid in wl.node_ids:
                x, y = f.task_locations_m[nid]
                assert 0.0 <= x <= TB3_MAP_SIZE_M
                assert 0.0 <= y <= TB3_MAP_SIZE_M
                assert not _point_in_obstacle(x, y)
                ratio = f.task_wcets_ms[nid] / wl.task_wcets_ms[nid]
                assert WCET_JITTER_LO <= ratio <= WCET_JITTER_HI
            assert len(f.battery_capacities_wh) == n
            for bat in f.battery_capacities_wh:
                assert lo <= bat <= hi


def test_g4_obstacle_map_tripwire():
    pytest.importorskip("torch")
    from envs.turtlebot3_spatial_env import MAP_SIZE_M, TurtleBot3Env

    cfg = DARTS_CONFIG()
    env = TurtleBot3Env(
        num_agents=4,
        global_deadline_ms=cfg.global_deadline_ms,
        cfg=cfg,
        seed=0,
    )
    assert TB3_MAP_SIZE_M == MAP_SIZE_M
    assert TB3_OBSTACLES_M == tuple(env.obstacles)
    assert TB3_OBSTACLES_M == tuple(env._default_obstacles())

    probes = []
    for ox, oy, w, h in TB3_OBSTACLES_M:
        probes.extend([
            (ox, oy), (ox + w, oy), (ox, oy + h), (ox + w, oy + h),
            (ox + w / 2.0, oy), (ox + w / 2.0, oy + h),
            (ox, oy + h / 2.0), (ox + w, oy + h / 2.0),
            (ox + w / 2.0, oy + h / 2.0),
            (ox + 1e-9, oy + 1e-9),
            (ox - 1e-6, oy + h / 2.0),
            (ox + w + 1e-6, oy + h / 2.0),
        ])
    rng = np.random.default_rng(0)
    for _ in range(100):
        probes.append((
            float(rng.uniform(-1.0, 11.0)),
            float(rng.uniform(-1.0, 11.0)),
        ))
    mismatches = []
    for x, y in probes:
        ours = _point_in_obstacle(x, y)
        theirs = env._collides_obstacles(x, y)
        if ours != theirs:
            mismatches.append((x, y, ours, theirs))
    assert not mismatches, mismatches[:5]


def test_g5_both_application_paths_agree():
    cfg0 = DARTS_CONFIG()
    for cell in NOMINAL_CELLS:
        bench, stress = parse_cell(cell)
        inst = nominal_instance(bench, stress)
        cfg = dc_replace(
            cfg0,
            workload_id=bench,
            battery_capacities_wh=list(inst.battery_capacities_wh),
            num_agents=len(inst.battery_capacities_wh),
            global_deadline_ms=inst.global_deadline_ms,
            max_work_ms=inst.max_work_ms,
            max_episode_steps=inst.max_episode_steps,
        )
        wl_rt, cfg_rt = apply_instance_seed_runtime(inst.workload, cfg, 7)
        seeded = apply_instance_seed(inst, 7)
        assert wl_rt.task_wcets_ms == seeded.declared_task_wcets_ms
        assert wl_rt.task_locations_m == seeded.workload.task_locations_m
        assert tuple(cfg_rt.battery_capacities_wh) == seeded.battery_capacities_wh
        assert seeded.declared_task_wcets_ms == seeded.runtime_task_wcets_ms
        assert seeded.declared_task_wcets_ms == dict(wl_rt.task_wcets_ms)
        merged = effective_task_wcets(wl_rt, cfg_rt.task_wcets)
        assert merged == dict(wl_rt.task_wcets_ms)


def test_g6_scaled_and_seed_commute():
    for cell in NOMINAL_CELLS:
        inst = nominal_instance(*parse_cell(cell))
        a = apply_instance_seed(inst, 7).scaled(0.8)
        b = apply_instance_seed(inst.scaled(0.8), 7)
        assert a.benchmark == b.benchmark and a.stress == b.stress
        assert a.instance_seed == b.instance_seed == 7
        assert a.beta == b.beta
        assert a.global_deadline_ms == b.global_deadline_ms
        assert a.relative_deadline_ms == b.relative_deadline_ms
        assert a.max_work_ms == b.max_work_ms
        assert a.max_episode_steps == b.max_episode_steps
        assert a.declared_task_wcets_ms == b.declared_task_wcets_ms
        assert a.runtime_task_wcets_ms == b.runtime_task_wcets_ms
        assert a.battery_capacities_wh == b.battery_capacities_wh
        assert a.workload.task_wcets_ms == b.workload.task_wcets_ms
        assert a.workload.task_locations_m == b.workload.task_locations_m
        assert a.workload.relative_deadline_ms == b.workload.relative_deadline_ms


def test_g7_harness_xi_source_default_is_declared():
    assert XI_SOURCE_DEFAULT == "declared"
    ap = build_parser()
    front = ap.parse_args(["frontier"])
    assert front.xi_source == "declared"
    ev = ap.parse_args(["eval", "--cell", "B1_CDR:heavy", "--beta", "1.0"])
    assert ev.xi_source == "declared"
    assert ev.instance_seed is None
    assert ev.instance_seeds is None
