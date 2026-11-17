"""EXP-2 partitioned alloc + fallback circumvention tests."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest

pytest.importorskip("torch", reason="alg.darts_agent requires torch")

from alg.config_darts import DARTS_CONFIG
from alg.darts_agent import DartsMARLAgent, partitioned_alloc
from alg.defer_fallback import (
    execute_fallback_assign,
    fallback_blocked_count,
    reset_fallback_blocked,
)
from envs.dag_task import DAGGraph, TaskNode
from envs.turtlebot3_spatial_env import TurtleBot3Env
from envs.workloads import get_workload


def _minted_j1(cfg: DARTS_CONFIG) -> TaskNode:
    wl = get_workload("B4_DEM")
    dag = DAGGraph.from_workload(wl, cfg)
    src = dag.nodes["J1"]
    return TaskNode(
        node_id="J1#0",
        name=src.name,
        wcet=src.wcet,
        predecessors=[],
        successors=[],
        relative_deadline=src.relative_deadline,
    )


def test_partitioned_alloc_blocks_non_owner_and_restores():
    cfg = DARTS_CONFIG()
    wl = get_workload("B4_DEM")
    env = TurtleBot3Env(num_agents=4, cfg=cfg, workload=wl, seed=7)
    env.reset()
    env.task_locations["J1#0"] = env.task_locations["J1"]
    job = _minted_j1(cfg)
    a1 = DartsMARLAgent(agent_id="agent_1", cfg=cfg, seed=1, spatial_state_dim=env.single_observation_size)
    a2 = DartsMARLAgent(agent_id="agent_2", cfg=cfg, seed=2, spatial_state_dim=env.single_observation_size)
    bat = list(env._battery_wh)

    feas_a1_open, _, _, _ = a1.compute_bid(job, 0.0, bat[0], explore=False, env=env, release_time_ms=0.0)
    feas_a2_open, _, _, _ = a2.compute_bid(job, 0.0, bat[1], explore=False, env=env, release_time_ms=0.0)

    with partitioned_alloc({"J1": "agent_1"}):
        feas_a1_map, _, _, _ = a1.compute_bid(job, 0.0, bat[0], explore=False, env=env, release_time_ms=0.0)
        feas_a2_map, _, _, _ = a2.compute_bid(job, 0.0, bat[1], explore=False, env=env, release_time_ms=0.0)

    feas_a1_after, _, _, _ = a1.compute_bid(job, 0.0, bat[0], explore=False, env=env, release_time_ms=0.0)
    feas_a2_after, _, _, _ = a2.compute_bid(job, 0.0, bat[1], explore=False, env=env, release_time_ms=0.0)

    assert feas_a1_map == feas_a1_open
    assert feas_a2_map is False
    assert feas_a1_after == feas_a1_open
    assert feas_a2_after == feas_a2_open


def test_execute_fallback_assign_blocked_when_sponsor_not_alloc():
    reset_fallback_blocked()
    assigned = []

    class _Agent:
        def __init__(self, agent_id):
            self.agent_id = agent_id

        def assign_job(self, *args, **kwargs):
            assigned.append(self.agent_id)

    agents = [_Agent("agent_1"), _Agent("agent_2")]
    job = SimpleNamespace(node_id="J1#0")
    tracker = SimpleNamespace(node_states={})
    env = SimpleNamespace(task_locations={}, _battery_wh=[1.0, 1.0])
    cfg = SimpleNamespace(delta_max_ms=50.0, e_safe_wh=0.05, p_moving_w=7.5)

    with partitioned_alloc({"J1": "agent_1"}):
        ret = execute_fallback_assign(
            agents, tracker, env, cfg, job,
            sponsor_idx=1, c_tilde_fb=1000.0, clock_ms=0.0,
            battery_wh=[1.0, 1.0],
            commit_time_ms={},
            task_assigned_to={},
            energy_cert_violations=[0, 0],
        )
    assert ret is None
    assert assigned == []
    assert fallback_blocked_count() == 1
    reset_fallback_blocked()

    ret2 = execute_fallback_assign(
        agents, tracker, env, cfg, job,
        sponsor_idx=1, c_tilde_fb=1000.0, clock_ms=0.0,
        battery_wh=[1.0, 1.0],
        commit_time_ms={},
        task_assigned_to={},
        energy_cert_violations=[0, 0],
    )
    assert ret2 is None
    assert assigned == ["agent_2"]
    assert fallback_blocked_count() == 0
    reset_fallback_blocked()
