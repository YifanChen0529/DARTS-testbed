import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alg.config_darts import DARTS_CONFIG
from envs.turtlebot3_spatial_env import TurtleBot3Env
from envs.workloads import get_workload


def test_completed_mission_stays_successful_after_clock_passes_deadline():
    cfg = DARTS_CONFIG(global_deadline_ms=1000.0)
    env = TurtleBot3Env(
        num_agents=1,
        cfg=cfg,
        global_deadline_ms=1000.0,
        workload=get_workload("B4_DEM"),
    )
    env.reset(init_positions=[(0.0, 0.0)])

    for sink_id in env._sink_node_ids:
        env.completed_tasks_ms[sink_id] = 500.0

    _, _, _, info = env.step(agents=None)
    assert info["clock_ms"] == 1000.0
    assert info["task_time_ms"] == 500.0
    assert info["end_to_end_deadline_miss"] is False

    _, _, _, info = env.step(agents=None)
    assert info["clock_ms"] == 2000.0
    assert info["task_time_ms"] == 500.0
    assert info["end_to_end_deadline_miss"] is False


def test_incomplete_mission_misses_once_clock_passes_deadline():
    cfg = DARTS_CONFIG(global_deadline_ms=1000.0)
    env = TurtleBot3Env(
        num_agents=1,
        cfg=cfg,
        global_deadline_ms=1000.0,
        workload=get_workload("B4_DEM"),
    )
    env.reset(init_positions=[(0.0, 0.0)])

    _, _, _, info = env.step(agents=None)
    assert info["clock_ms"] == 1000.0
    assert info["task_time_ms"] is None
    assert info["end_to_end_deadline_miss"] is False

    _, _, _, info = env.step(agents=None)
    assert info["clock_ms"] == 2000.0
    assert info["task_time_ms"] is None
    assert info["end_to_end_deadline_miss"] is True
