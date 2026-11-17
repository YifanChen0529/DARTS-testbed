"""Theorem 2(a) offline solver tests."""
from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.thm2a_offline_test import evaluate_thm2a, solve_ladder, _identity_alloc
from envs.workloads import get_workload


def test_b4_identity_delta_sum_monotone_in_dv():
    wl = get_workload("B4_DEM")
    alloc = _identity_alloc(list(wl.node_ids))
    dvs = [80000.0, 120000.0, 200000.0, 400000.0]
    sums = []
    for d in dvs:
        r = evaluate_thm2a("B4_DEM", alloc, D_v_ms=d, P_ms=d, impl_factors=True)
        sums.append(r["bottleneck"])
    for a, b in zip(sums, sums[1:]):
        assert b < a, f"delta_sum should drop with D_v: {sums}"


def test_solver_target_0p85_admits_and_meets_bound():
    wl = get_workload("B4_DEM")
    alloc = _identity_alloc(list(wl.node_ids))
    r = solve_ladder("B4_DEM", alloc, target_delta_sum=0.85, impl_factors=True)
    assert r["verdict"] == "ADMIT"
    assert r["bottleneck"] <= 0.85 + 1e-12
    assert r["assumption_ok"] is True
    assert r["P_ms"] == r["D_v_ms"]


def test_tiny_dv_rejects():
    wl = get_workload("B4_DEM")
    alloc = _identity_alloc(list(wl.node_ids))
    r = evaluate_thm2a("B4_DEM", alloc, D_v_ms=3000.0, P_ms=3000.0, impl_factors=True)
    assert r["verdict"] == "REJECT"


def test_b3_p_equals_dv_fails_assumption():
    wl = get_workload("B3_IQI")
    alloc = _identity_alloc(list(wl.node_ids))
    r = evaluate_thm2a("B3_IQI", alloc, D_v_ms=180000.0, P_ms=180000.0, impl_factors=True)
    assert r["assumption_ok"] is False
    assert r["verdict"] == "REJECT"
    assert r["reject_reason"] == "not a valid Theorem 2(a) instance"
    assert r["all_sources"] is False
