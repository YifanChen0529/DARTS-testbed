"""Unit tests for TB-1 sweep helpers (no torch, no live eval)."""

from __future__ import annotations

import os
import sys

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.sweep_tb1 import (
    SweepConfigError,
    assert_deadline_invariant,
    beta_tag,
    build_command,
    exp_name_for,
)


_B1_PROFILE = {"global_deadline_ms": 120000.0, "max_episode_steps": 160}
_ENTRY = {"dir": "/tmp", "ep": 2500}


def _build(deadline_scale: float = 1.0):
    return build_command(
        "B1_CDR", "heavy", 100, 0.75, _ENTRY, _B1_PROFILE,
        "results_tb1_p0", 100, 100, False, deadline_scale,
    )


def test_exp_name_suffixes_non_unity_beta():
    assert exp_name_for("B1_CDR", "heavy", 0.75, 100) == "tb1_B1_CDR_heavy_k0p75_seed100"
    assert exp_name_for("B1_CDR", "heavy", 0.75, 100, 1.0) == "tb1_B1_CDR_heavy_k0p75_seed100"
    assert exp_name_for("B1_CDR", "heavy", 0.75, 100, 0.85) == (
        "tb1_B1_CDR_heavy_k0p75_seed100_b0p85"
    )
    assert beta_tag(0.85) == "b0p85"


def test_deadline_scale_appends_flag_and_suffix():
    cmd, name = _build(0.85)
    assert name == "tb1_B1_CDR_heavy_k0p75_seed100_b0p85"
    assert "--deadline_scale" in cmd
    assert cmd[cmd.index("--deadline_scale") + 1] == "0.85"
    assert_deadline_invariant("B1_CDR", "heavy", _B1_PROFILE, cmd, deadline_scale=0.85)


def test_unity_beta_omits_flag_and_suffix():
    cmd, name = _build(1.0)
    assert name == "tb1_B1_CDR_heavy_k0p75_seed100"
    assert "--deadline_scale" not in cmd
    assert_deadline_invariant("B1_CDR", "heavy", _B1_PROFILE, cmd, deadline_scale=1.0)


def test_deadline_invariant_requires_scale_flag_when_active():
    cmd, _name = _build(1.0)
    with pytest.raises(SweepConfigError, match="--deadline_scale is missing"):
        assert_deadline_invariant("B1_CDR", "heavy", _B1_PROFILE, cmd, deadline_scale=0.85)


def test_deadline_invariant_rejects_scale_flag_at_unity():
    cmd, _name = _build(0.85)
    with pytest.raises(SweepConfigError, match="unexpected --deadline_scale"):
        assert_deadline_invariant("B1_CDR", "heavy", _B1_PROFILE, cmd, deadline_scale=1.0)


def test_ld_library_path_is_inside_conda_run():
    """LD_LIBRARY_PATH must be applied after conda run so it survives an already-active env."""
    from scripts.sweep_tb1 import CONDA_ENV, LD_LIBRARY_PATH, _PROJECT_ROOT

    cmd, _name = _build(1.0)
    expected_prefix = [
        "conda", "run", "--no-capture-output", "-n", CONDA_ENV,
        "env",
        f"LD_LIBRARY_PATH={LD_LIBRARY_PATH}",
        f"PYTHONPATH={_PROJECT_ROOT}",
        "python3", "-u", "alg/train_darts.py",
    ]
    assert cmd[: len(expected_prefix)] == expected_prefix
    assert cmd[0] != "env"
