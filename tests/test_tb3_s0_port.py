"""S0 port tripwire for the TB-3 offline harness.

Mirrors ``tests/test_cert_duplicate_fidelity.py``: the duplicate (here,
``s0_path_sum``) is compared against the AE-era original
(``scripts.r_off_g.compute_r_off_g``) at exact float equality. If this
fails, the port has drifted — reconcile it rather than relaxing the
assertion.

t3 documents T0 H3 (B3 runtime WCET override). That is pre-existing
runtime behaviour; if someone later re-derives ``cfg.task_wcets`` per
workload, this test must be updated consciously together with the
paper's B3 numbers.
"""
from __future__ import annotations

import os
import subprocess
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest

from alg.config_darts import DARTS_CONFIG
from alg.tb3_instance import (
    effective_task_wcets,
    iter_profiled_cells,
    nominal_instance,
)
from envs.workloads import get_workload
from scripts.offline_test_harness import (
    PUBLISHED_ENVELOPES,
    TESTS,
    evaluate,
    s0_path_sum,
)
from scripts.r_off_g import compute_r_off_g

PUBLISHED_R_INT = {
    ("B1_CDR", "light"): 240250,
    ("B1_CDR", "medium"): 240250,
    ("B1_CDR", "heavy"): 240250,
    ("B2_LML", "medium"): 255513,
    ("B3_IQI", "medium"): 255513,
    ("B4_DEM", "medium"): 74810,
}

DEADLINE_TABLE = {
    ("B1_CDR", "medium"): (150000.0, 150000.0),
    ("B1_CDR", "heavy"): (120000.0, 150000.0),
    ("B1_CDR", "light"): (200000.0, 150000.0),
    ("B2_LML", "medium"): (240000.0, 170000.0),
    ("B3_IQI", "medium"): (180000.0, 150000.0),
    ("B4_DEM", "medium"): (100000.0, 100000.0),
}


def _ref(bench: str, stress: str):
    env = PUBLISHED_ENVELOPES
    return compute_r_off_g(
        bench, stress,
        env.delta_auc_ms, env.H_ms, env.j_bar_ms,
        env.u_mps, env.h_comm_ms, env.start_pos,
    )


def test_t1_s0_tripwire_matches_r_off_g():
    env = PUBLISHED_ENVELOPES
    for bench, stress in iter_profiled_cells():
        inst = nominal_instance(bench, stress)
        got = s0_path_sum(inst, env, "declared")
        ref = _ref(bench, stress)
        assert got.R_G_off_ms == ref["R_off_G_ms"], (bench, stress, got.R_G_off_ms, ref["R_off_G_ms"])
        assert got.longest_path == list(ref["longest_path"]), (bench, stress)
        assert got.A_bar == ref["A_bar"], (bench, stress)
        assert got.tau_bar_ms == ref["tau_bar_ms"], (bench, stress)


def test_t2_published_integers():
    env = PUBLISHED_ENVELOPES
    for (bench, stress), want in PUBLISHED_R_INT.items():
        inst = nominal_instance(bench, stress)
        got = s0_path_sum(inst, env, "declared")
        assert round(got.R_G_off_ms) == want, (bench, stress, got.R_G_off_ms, want)


def test_t3_locks_t0_h3_b3_runtime_wcet_override():
    """Documents a pre-existing runtime behaviour (T0 H3).

    If someone later re-derives ``cfg.task_wcets`` per workload, this test must
    be updated consciously together with the paper's B3 numbers.
    """
    cfg_wcets = DARTS_CONFIG().task_wcets

    def node_sum(wid: str) -> float:
        wl = get_workload(wid)
        merged = effective_task_wcets(wl, cfg_wcets)
        # T0 H3 table is the DAG-node sum. The merge dict also carries unused
        # A–D keys on B2/B4 (disjoint from Pk/Hf/Dr/Cf and J1–J4).
        return sum(merged[n] for n in wl.node_ids)

    b3_merged = effective_task_wcets(get_workload("B3_IQI"), cfg_wcets)
    assert sum(b3_merged.values()) == 95000.0
    assert node_sum("B3_IQI") == 95000.0
    assert node_sum("B1_CDR") == 95000.0
    assert node_sum("B2_LML") == 60000.0
    assert node_sum("B4_DEM") == 72000.0
    declared_b3 = sum(get_workload("B3_IQI").task_wcets_ms.values())
    assert declared_b3 == 60000.0


def test_t4_deadlines_and_scaled_no_double_multiply():
    for (bench, stress), (d_e2e, d_rel) in DEADLINE_TABLE.items():
        inst = nominal_instance(bench, stress)
        assert inst.global_deadline_ms == d_e2e, (bench, stress)
        assert inst.relative_deadline_ms == d_rel, (bench, stress)
        assert inst.beta == 1.0

    nom = nominal_instance("B1_CDR", "heavy")
    s = nom.scaled(0.8)
    assert s.global_deadline_ms == pytest.approx(96000.0)
    assert s.relative_deadline_ms == pytest.approx(120000.0)
    assert s.declared_task_wcets_ms == nom.declared_task_wcets_ms
    assert s.runtime_task_wcets_ms == nom.runtime_task_wcets_ms
    assert s.workload.task_locations_m == nom.workload.task_locations_m

    # Double-scaling distinguisher: B4 R=74810 sits between 0.8*D=80000 and
    # 0.8*(0.8*D)=64000. evaluate must accept (uses stored deadlines).
    b4 = nominal_instance("B4_DEM", "medium").scaled(0.8)
    row = evaluate(b4, ["S0"], PUBLISHED_ENVELOPES, "declared")[0]
    r = row.result.R_G_off_ms
    assert r <= b4.global_deadline_ms
    assert not (r <= b4.beta * b4.global_deadline_ms)
    assert row.accept_e2e is True
    assert row.accept_e2e == (r <= b4.global_deadline_ms)


def test_t5_envelope_constants_match_env_module():
    pytest.importorskip("torch")
    from envs.turtlebot3_spatial_env import MAP_SIZE_M, MAX_VELOCITY_MPS

    env = PUBLISHED_ENVELOPES
    assert env.u_mps == MAX_VELOCITY_MPS
    assert env.map_size_m == MAP_SIZE_M


def test_t6_torch_free_import():
    env = dict(os.environ)
    env["PYTHONPATH"] = _PROJECT_ROOT
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import scripts.offline_test_harness, alg.tb3_instance; "
            "assert 'torch' not in sys.modules",
        ],
        env=env,
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_t7_s4_stub_and_s1_sound():
    inst = nominal_instance("B1_CDR", "medium")
    env = PUBLISHED_ENVELOPES
    with pytest.raises(NotImplementedError, match="TB3-6/TB-4"):
        TESTS["S4"].fn(inst, env, "declared")
    assert TESTS["S4"].sound is False
    assert TESTS["S1"].sound is True
    assert TESTS["S0"].sound is True
    # S1 is live as of TB3-3; the old NotImplementedError stub is gone.
    got = TESTS["S1"].fn(inst, env, "declared")
    assert got.test == "S1"
    assert got.sound is True
