"""EXP-2 periodic-release unit tests (id minting, battery restore, horizon, sidecar)."""
from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest

from alg.periodic_release import (
    class_of,
    mint_instance_id,
    parse_instance_id,
    validate_periodic_horizon,
    PeriodicReleaseManager,
    reset_exp2_runtime,
)
from scripts.csv_schema_tb import MAIN_CSV_HEADER_TB76, RELEASE_LOG_HEADER


def test_id_minting_round_trip():
    assert mint_instance_id("J1", 0) == "J1#0"
    assert mint_instance_id("A", 2) == "A#2"
    assert parse_instance_id("J1#0") == ("J1", 0)
    assert parse_instance_id("A#2") == ("A", 2)
    assert parse_instance_id("J1") == ("J1", None)
    assert parse_instance_id("A") == ("A", None)
    assert class_of("J1#3") == "J1"
    assert class_of("J1") == "J1"
    assert class_of("A#0") == "A"
    assert class_of("A") == "A"


def test_restore_battery_to_initial_does_not_mutate_snapshot():
    from envs.turtlebot3_spatial_env import TurtleBot3Env

    env = TurtleBot3Env(num_agents=4, seed=1)
    env.reset()
    initial = list(env._initial_battery_wh)
    assert initial, "reset must snapshot initial batteries"
    for i in range(env.num_agents):
        env._consume_energy(i, is_moving=True, num_messages=2)
    drained = list(env._battery_wh)
    assert any(d < i for d, i in zip(drained, initial))
    restored = env.restore_battery_to_initial()
    assert restored == pytest.approx(initial)
    assert list(env._battery_wh) == pytest.approx(initial)
    assert list(env._initial_battery_wh) == pytest.approx(initial)


def test_horizon_validation_too_small_max_episode_steps():
    with pytest.raises(SystemExit) as ei:
        validate_periodic_horizon(
            release_period_ms=120000.0,
            num_releases=3,
            relative_deadline_ms=120000.0,
            max_episode_steps=100,
            global_deadline_ms=480000.0,
            disable_feasibility_mask=False,
            alloc_map_active=True,
        )
    msg = str(ei.value)
    assert "max_episode_steps*1000" in msg
    assert "num_releases*P + P" in msg
    assert "--max_episode_steps" in msg


def test_horizon_validation_accepts_g4_sizing():
    validate_periodic_horizon(
        release_period_ms=120000.0,
        num_releases=3,
        relative_deadline_ms=120000.0,
        max_episode_steps=500,
        global_deadline_ms=480000.0,
        disable_feasibility_mask=False,
        alloc_map_active=True,
    )


def test_sidecar_header_registered_and_main_csv_still_76():
    assert RELEASE_LOG_HEADER == [
        "workload", "stress", "seed", "ckpt_dir", "ckpt_ep",
        "P_ms", "L", "D_v_ms", "per_period_energy", "alloc_map_sha_or_path",
        "release_idx", "job_class", "instance_id", "agent_id",
        "r_ms", "commit_ms", "finish_ms", "deadline_ms", "met",
        "queue_empty_at_next_boundary", "battery_at_boundary_wh", "fallback_blocked",
    ]
    assert len(MAIN_CSV_HEADER_TB76) == 76


def test_sidecar_written_and_main_eval_header_untouched(tmp_path):
    pytest.importorskip("torch", reason="EVAL_CSV_HEADER lives in alg.train_darts")
    from alg.train_darts import EVAL_CSV_HEADER
    from envs.workloads import get_workload
    from alg.config_darts import DARTS_CONFIG
    from envs.turtlebot3_spatial_env import TurtleBot3Env

    assert list(EVAL_CSV_HEADER) == list(MAIN_CSV_HEADER_TB76)
    assert len(EVAL_CSV_HEADER) == 76

    reset_exp2_runtime()
    cfg = DARTS_CONFIG()
    wl = get_workload("B4_DEM")
    env = TurtleBot3Env(num_agents=4, cfg=cfg, workload=wl, seed=1)
    env.reset()
    mgr = PeriodicReleaseManager(P=120000.0, L=1, D_v=120000.0)
    mgr.bootstrap(env, cfg, wl, 120000.0, 1, 120000.0)
    path = mgr.write_sidecar(str(tmp_path), env, RELEASE_LOG_HEADER)
    with open(path, "r") as f:
        header = f.readline().strip().split(",")
    assert header == RELEASE_LOG_HEADER
    reset_exp2_runtime()
