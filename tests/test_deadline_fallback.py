import os
import sys

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

pytest.importorskip("torch", reason="DARTS agents require torch")

from alg.baseline_cea_edf import CEAEDFAgent
from alg.config_darts import DARTS_CONFIG
from alg.darts_agent import DartsMARLAgent
from envs.dag_task import TaskNode


def _task_without_relative_deadline(node_id: str) -> TaskNode:
    return TaskNode(
        node_id=node_id,
        name=node_id,
        wcet=1000.0,
        predecessors=[],
        successors=[],
        relative_deadline=None,
    )


def test_darts_commit_uses_global_deadline_fallback_for_optional_relative_deadline():
    cfg = DARTS_CONFIG(global_deadline_ms=12345.0)
    agent = DartsMARLAgent("agent_1", cfg=cfg)

    agent.commit_task(
        _task_without_relative_deadline("A"),
        release_time_ms=500.0,
        current_time_ms=500.0,
        override_remaining_work_ms=1000.0,
    )

    assert agent.queue[0].absolute_deadline_ms == 12845.0


def test_cea_edf_commit_uses_global_deadline_fallback_for_optional_relative_deadline():
    cfg = DARTS_CONFIG(global_deadline_ms=12345.0)
    agent = CEAEDFAgent("agent_1", cfg=cfg)

    agent.commit_task(
        _task_without_relative_deadline("A"),
        release_time_ms=500.0,
        current_time_ms=500.0,
        override_remaining_work_ms=1000.0,
    )

    assert agent.queue[0].absolute_deadline_ms == 12845.0
