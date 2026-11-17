"""S1 per-node travel eccentricity: numeric lock, dominance on registered
geometries, incomparability off them, seeded flow, S0-anchor regression.

s1c's witness is NOT S1 looseness. S0's published instantiation fixes the
start set to {(0,0)} while deployment start poses are uniform over W, so S0
under-covers the first-leg worst case there (T0 §6). Never assert S1 <= S0
globally.
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest

from alg.tb3_instance import (
    NOMINAL_CELLS,
    MissionInstance,
    apply_instance_seed,
    nominal_instance,
    parse_cell,
)
from envs.workloads import get_workload
from scripts.offline_test_harness import (
    PUBLISHED_DARTS_MEAN,
    PUBLISHED_ENVELOPES,
    TESTS,
    TESTS_DEFAULT,
    XI_SOURCE_DEFAULT,
    build_parser,
    evaluate,
    s0_path_sum,
    s1_eccentricity,
)
from scripts.r_off_g import euclid

# 4-decimal lock: |R - tabulated| < 1e-4 (one unit in the 4th decimal).
# The spec's 5e-5 is tighter than the unrounded path-sum vs the rounded table
# (B1 207936.475244 vs 207936.4753).
R_TOL = 1e-4
TAU_DEC4 = {
    32141.2173,  # sqrt(50)
    35501.1349,  # sqrt(61)
    39101.4785,  # sqrt(74)
    44825.1885,  # sqrt(97.25)
    46798.3188,  # sqrt(106)
    54640.0695,  # 12.0208152802
}

S1_R = {
    "B1_CDR": 207936.4753,
    "B2_LML": 236519.2510,
    "B3_IQI": 234546.1207,
    "B4_DEM": 74810.0695,
}
S1_PATH = {
    "B1_CDR": ["A", "B", "D"],
    "B2_LML": ["Pk", "Hf", "Dr", "Cf"],
    "B3_IQI": ["A", "B", "C", "D"],
    "B4_DEM": ["J1"],
}
S1_A_BAR = {"B1_CDR": 2, "B2_LML": 1, "B3_IQI": 1, "B4_DEM": 4}
S1_RATIO = {
    ("B1_CDR", "light"): 1.90,
    ("B1_CDR", "medium"): 2.01,
    ("B1_CDR", "heavy"): 2.07,
    ("B2_LML", "medium"): 2.51,
    ("B3_IQI", "medium"): 1.93,
    ("B4_DEM", "medium"): 1.31,
}
S1_BETA_E2E = {
    ("B1_CDR", "light"): 1.0397,
    ("B1_CDR", "medium"): 1.3862,
    ("B1_CDR", "heavy"): 1.7328,
    ("B2_LML", "medium"): 0.9855,
    ("B3_IQI", "medium"): 1.3030,
    ("B4_DEM", "medium"): 0.7481,
}
S1_BETA_REL = {
    "B1_CDR": 1.3862,
    "B2_LML": 1.3913,
    "B3_IQI": 1.5636,
    "B4_DEM": 0.7481,
}
B3_RUNTIME_R = 269546.1207
B1_NODE_R = {"A": 49251.2173, "B": 81750.0695, "D": 76935.1885}


def test_s1a_numeric_expectations():
    env = PUBLISHED_ENVELOPES
    seen_tau = set()
    for cell in NOMINAL_CELLS:
        bench, stress = parse_cell(cell)
        inst = nominal_instance(bench, stress)
        got = s1_eccentricity(inst, env, "declared")
        assert got.A_bar == S1_A_BAR[bench]
        assert got.longest_path == S1_PATH[bench], (cell, got.longest_path)
        assert got.R_G_off_ms == pytest.approx(S1_R[bench], abs=R_TOL)
        assert got.sound is True
        tau_hat = got.meta["tau_hat_ms"]
        for v, t in tau_hat.items():
            seen_tau.add(round(t, 4))
        if bench == "B1_CDR":
            for nid, want in B1_NODE_R.items():
                assert got.per_node_ms[nid] == pytest.approx(want, abs=R_TOL)
        d_e2e = inst.global_deadline_ms
        d_rel = inst.relative_deadline_ms
        assert got.R_G_off_ms / d_e2e == pytest.approx(
            S1_BETA_E2E[(bench, stress)], abs=5e-5
        )
        assert got.R_G_off_ms / d_rel == pytest.approx(S1_BETA_REL[bench], abs=5e-5)
        meas = PUBLISHED_DARTS_MEAN[(bench, stress)]
        assert round(got.R_G_off_ms / meas, 2) == S1_RATIO[(bench, stress)]

    for want in TAU_DEC4:
        assert any(abs(t - want) < 5e-5 for t in seen_tau), (want, sorted(seen_tau))

    b3 = s1_eccentricity(nominal_instance("B3_IQI", "medium"), env, "runtime")
    assert b3.R_G_off_ms == pytest.approx(B3_RUNTIME_R, abs=R_TOL)

    b2 = nominal_instance("B2_LML", "medium")
    s0 = s0_path_sum(b2, env, "declared")
    s1 = s1_eccentricity(b2, env, "declared")
    assert s0.R_G_off_ms > b2.global_deadline_ms
    assert s1.R_G_off_ms <= b2.global_deadline_ms


def test_s1b_registered_geometry_dominance():
    env = PUBLISHED_ENVELOPES
    for cell in NOMINAL_CELLS:
        inst = nominal_instance(*parse_cell(cell))
        s0 = s0_path_sum(inst, env, "declared")
        s1 = s1_eccentricity(inst, env, "declared")
        for v, t in s1.meta["tau_hat_ms"].items():
            assert t <= s0.tau_bar_ms + 1e-9, (cell, v, t, s0.tau_bar_ms)
        assert s1.R_G_off_ms <= s0.R_G_off_ms + 1e-9


def test_s1c_incomparability_witness():
    """S0 under-covers first-leg travel when starts are uniform over W.

    This is NOT S1 looseness. Never assert S1 <= S0 globally.
    """
    nom = nominal_instance("B4_DEM", "medium")
    wl = replace(
        get_workload("B4_DEM"),
        task_locations_m={nid: (9.5, 0.5) for nid in get_workload("B4_DEM").node_ids},
    )
    inst = MissionInstance(
        benchmark=nom.benchmark,
        stress=nom.stress,
        instance_seed=None,
        beta=1.0,
        workload=wl,
        declared_task_wcets_ms=dict(wl.task_wcets_ms),
        runtime_task_wcets_ms=dict(wl.task_wcets_ms),
        global_deadline_ms=nom.global_deadline_ms,
        relative_deadline_ms=nom.relative_deadline_ms,
        battery_capacities_wh=nom.battery_capacities_wh,
        max_work_ms=nom.max_work_ms,
        max_episode_steps=nom.max_episode_steps,
    )
    env = PUBLISHED_ENVELOPES
    s0 = s0_path_sum(inst, env, "declared")
    s1 = s1_eccentricity(inst, env, "declared")
    diam_s0 = euclid((0.0, 0.0), (9.5, 0.5))
    assert s0.tau_bar_ms == pytest.approx((diam_s0 / env.u_mps) * 1000.0)
    assert s0.tau_bar_ms == pytest.approx(43241.4, abs=0.2)
    diam_s1 = euclid((0.0, 10.0), (9.5, 0.5))
    tau_s1 = (diam_s1 / env.u_mps) * 1000.0
    for t in s1.meta["tau_hat_ms"].values():
        assert t == pytest.approx(tau_s1)
    assert s1.tau_bar_ms == pytest.approx(61068.3, abs=0.1)
    assert s1.R_G_off_ms > s0.R_G_off_ms


def test_s1d_seeded_instance_flow():
    env = PUBLISHED_ENVELOPES
    nom = nominal_instance("B1_CDR", "heavy")
    for seed in range(10):
        inst = apply_instance_seed(nom, seed)
        rows = evaluate(inst, ["S1"], env, "declared")
        assert len(rows) == 1
        assert rows[0].result.test == "S1"
        assert rows[0].inst.instance_seed == seed
        assert rows[0].result.R_G_off_ms > 0.0


def test_s1e_s4_stub():
    inst = nominal_instance("B1_CDR", "medium")
    with pytest.raises(NotImplementedError, match="TB3-6/TB-4"):
        TESTS["S4"].fn(inst, PUBLISHED_ENVELOPES, "declared")
    assert TESTS["S4"].sound is False


def test_s1f_defaults():
    assert TESTS_DEFAULT == "S0,S1"
    assert XI_SOURCE_DEFAULT == "declared"
    ap = build_parser()
    front = ap.parse_args(["frontier"])
    assert front.xi_source == "declared"
    assert front.tests == "S0,S1"
    ev = ap.parse_args(["eval", "--cell", "B1_CDR:heavy", "--beta", "1.0"])
    assert ev.xi_source == "declared"
    assert ev.tests == "S0,S1"
