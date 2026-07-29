"""Known-solution unit tests for Eq. (11) prefix feasibility (_feasibility_test).

Tuple layout consumed by alg.darts_agent._feasibility_test (see its signature
and unpack ``for _jid, d_v, c_v in sorted_jobs``):
    (job_id: str, absolute_deadline_ms: float, cost_ms: float)

Jobs must arrive EDF-sorted by absolute deadline (as callers do). Feasible iff
for every prefix ℓ:  Σ_{j≤ℓ} c_j  ≤  d_ℓ − t_k.
Strict ``>`` rejects; equality on the bound is accepted (boundary-tight).
"""

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest

pytest.importorskip("torch", reason="alg.darts_agent requires torch")

from alg.darts_agent import _feasibility_test


def test_feasibility_boundary_tight_accept():
    """Case (a): t_k=0; (c=10,d=10), (c=5,d=15) → both prefixes tight-equal → FEASIBLE.

    Prefix ℓ=1: Σc = 10  ≤  d1 − t_k = 10 − 0 = 10  ✓
    Prefix ℓ=2: Σc = 15  ≤  d2 − t_k = 15 − 0 = 15  ✓
    """
    # EDF order: d=10 then d=15. Layout: (job_id, d_v, c_v).
    jobs = [
        ("j1", 10.0, 10.0),
        ("j2", 15.0, 5.0),
    ]
    assert _feasibility_test(jobs, current_time_ms=0.0) is True


def test_feasibility_second_prefix_miss():
    """Case (b): same as (a) but d2=14 → second prefix fails → INFEASIBLE.

    Prefix ℓ=1: 10 ≤ 10 − 0 = 10  ✓
    Prefix ℓ=2: 15 ≤ 14 − 0 = 14  ✗  (15 > 14)
    """
    jobs = [
        ("j1", 10.0, 10.0),
        ("j2", 14.0, 5.0),
    ]
    assert _feasibility_test(jobs, current_time_ms=0.0) is False


def test_feasibility_nonzero_tk_first_prefix_miss():
    """Case (c): case (a) jobs with t_k=5 → first prefix fails → INFEASIBLE.

    Prefix ℓ=1: 10 ≤ 10 − 5 = 5  ✗  (10 > 5)
    """
    jobs = [
        ("j1", 10.0, 10.0),
        ("j2", 15.0, 5.0),
    ]
    assert _feasibility_test(jobs, current_time_ms=5.0) is False
