"""Copy-drift tripwire for the TB-1 certificate duplicate.

``DartsMARLAgent.check_feasibility_scaled`` is a verbatim duplicate of the AE-era
``check_feasibility`` with κ factors applied to the chain, no-location and energy
envelopes. At κ=1.0 the two bodies must agree exactly.

This matters because ``scaled_cert_envelopes`` only rebinds the class attribute when some
scale is non-unit. Every AE-parity and κ=1.0 sweep run therefore executes the *original*
body and never touches the duplicate, so nothing else in the suite would notice if the two
copies drifted apart. Here the duplicate is invoked DIRECTLY -- no context manager, no
rebinding -- against the same probe grid the κ liveness probe uses (26 admission times ×
3 battery levels = 78 points), spanning both the temporal and the energy certificate.

If this test fails, the two bodies have diverged: reconcile them rather than relaxing the
assertion.
"""

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest

pytest.importorskip("torch", reason="alg.darts_agent requires torch")

from alg.config_darts import DARTS_CONFIG
from alg.darts_agent import _CERT_ENVELOPE_SCALES, DartsMARLAgent
from envs.dag_task import DAGGraph
from envs.turtlebot3_spatial_env import TurtleBot3Env
from envs.workloads import get_workload

PROBE_TIMES_MS = [float(t) for t in range(0, 130000, 5000)]
PROBE_BATTERY_WH = (0.08, 0.2, 1.0)
N_PROBE_POINTS = len(PROBE_TIMES_MS) * len(PROBE_BATTERY_WH)


@pytest.fixture(scope="module")
def probe_fixture():
    cfg = DARTS_CONFIG()
    workload = get_workload("B1_CDR")
    env = TurtleBot3Env(
        num_agents=4,
        global_deadline_ms=cfg.global_deadline_ms,
        cfg=cfg,
        seed=7,
    )
    env.reset()
    dag = DAGGraph.from_workload(workload, cfg)
    agent = DartsMARLAgent(
        agent_id="agent_1",
        cfg=cfg,
        seed=1,
        spatial_state_dim=env.single_observation_size,
    )
    return agent, dag.nodes["A"], env


def test_probe_grid_is_78_points():
    assert N_PROBE_POINTS == 78


def test_scales_are_unit_outside_any_context():
    """The direct-invocation test below is only meaningful at unit scales."""
    assert _CERT_ENVELOPE_SCALES == {
        "kappa": 1.0,
        "exec": 1.0,
        "travel": 1.0,
        "energy": 1.0,
    }


def test_scaled_duplicate_matches_original_at_unit_scales(probe_fixture):
    """check_feasibility_scaled == check_feasibility on all 78 probe points, κ=1.0.

    Invoked directly on the instance, so the rebinding path in scaled_cert_envelopes is
    bypassed entirely and only the duplicated arithmetic is under test.
    """
    agent, job, env = probe_fixture
    assert agent.check_feasibility is not agent.check_feasibility_scaled, (
        "no rebinding must be active: the two bodies have to be compared as distinct code"
    )
    mismatches = []
    n_checked = 0
    for t_now in PROBE_TIMES_MS:
        for battery_wh in PROBE_BATTERY_WH:
            kwargs = dict(
                current_time_ms=t_now,
                current_battery_wh=battery_wh,
                release_time_ms=0.0,
                env=env,
            )
            original = agent.check_feasibility(job, **kwargs)
            duplicate = agent.check_feasibility_scaled(job, **kwargs)
            n_checked += 1
            if original != duplicate:
                mismatches.append((t_now, battery_wh, original, duplicate))
    assert n_checked == N_PROBE_POINTS
    assert not mismatches, (
        "check_feasibility_scaled drifted from check_feasibility at κ=1.0 on "
        f"{len(mismatches)}/{n_checked} probe points: {mismatches[:5]}"
    )


def test_duplicate_grid_exercises_both_verdicts(probe_fixture):
    """Guard against a vacuous pass where every probe point returns the same verdict."""
    agent, job, env = probe_fixture
    verdicts = {
        agent.check_feasibility_scaled(
            job,
            current_time_ms=t_now,
            current_battery_wh=battery_wh,
            release_time_ms=0.0,
            env=env,
        )
        for t_now in PROBE_TIMES_MS
        for battery_wh in PROBE_BATTERY_WH
    }
    assert verdicts == {True, False}, (
        "probe grid no longer straddles the certificate boundary; it would pass even if "
        f"the duplicate always returned a constant (saw {verdicts})"
    )
