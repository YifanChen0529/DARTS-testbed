import dataclasses
from types import SimpleNamespace

import pytest

from alg.config_darts import DARTS_CONFIG
from envs.dag_task import DAGGraph
from envs.workloads import get_workload, runtime_workload


def test_runtime_workload_prefers_env_override_for_relative_deadline():
    cfg = dataclasses.replace(DARTS_CONFIG(), workload_id="B2_LML")
    overridden = dataclasses.replace(
        get_workload("B2_LML"),
        relative_deadline_ms=150000.0,
    )
    env = SimpleNamespace(workload=overridden)

    dag = DAGGraph.from_workload(runtime_workload(env, cfg), cfg)

    assert get_workload("B2_LML").relative_deadline_ms != 150000.0
    assert {node.relative_deadline for node in dag.nodes.values()} == {150000.0}


def test_train_episode_builds_dag_from_runtime_workload(monkeypatch):
    pytest.importorskip("torch", reason="train_darts imports torch-backed agents")
    from alg import train_darts

    cfg = dataclasses.replace(
        DARTS_CONFIG(),
        workload_id="B2_LML",
        max_episode_steps=0,
    )
    overridden = dataclasses.replace(
        get_workload("B2_LML"),
        relative_deadline_ms=150000.0,
    )
    env = SimpleNamespace(
        workload=overridden,
        step_dt_s=1.0,
        completed_tasks_ms={},
        initial_battery_wh=[1.0],
        reset=lambda rng_seed=None: ({}, {}),
    )
    agents = [SimpleNamespace(reset=lambda: None)]
    seen_deadlines = []
    original_from_workload = DAGGraph.from_workload

    def spy_from_workload(workload, cfg_arg):
        seen_deadlines.append(workload.relative_deadline_ms)
        return original_from_workload(workload, cfg_arg)

    monkeypatch.setattr(DAGGraph, "from_workload", staticmethod(spy_from_workload))

    train_darts.run_episode(
        env,
        agents,
        cfg,
        explore=False,
        store_transitions=False,
        episode_verbose=False,
    )

    assert seen_deadlines[0] == 150000.0
