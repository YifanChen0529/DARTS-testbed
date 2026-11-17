"""EXP-2 config table + sweep_exp2 runner tests."""
from __future__ import annotations

import os
import re
import subprocess
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.sweep_exp2 import build_eval_command, load_table

TABLE = os.path.join(_PROJECT_ROOT, "docs", "exp2", "exp2_config_table.json")


def test_config_table_horizon_and_verdicts():
    rows = load_table(TABLE)
    assert len(rows) == 12
    ids = [r["config_id"] for r in rows]
    assert len(set(ids)) == 12
    assert "exp2_B4_DEM_d085" in ids
    assert "exp2_B4_DEM_d085_rep" in ids
    assert "exp2_B4_DEM_probe" in ids
    assert "exp2_B3_IQI_probe" in ids
    for r in rows:
        dv = float(r["D_v_ms"])
        p = float(r["P_ms"])
        l = int(r["L"])
        steps = int(r["max_episode_steps"])
        g = float(r["global_deadline_ms"])
        assert dv <= p + 1e-9, r["config_id"]
        assert abs(p / 1000.0 - round(p / 1000.0)) < 1e-9, r["config_id"]
        assert steps * 1000.0 + 1e-9 >= l * p + p, r["config_id"]
        assert g + 1e-9 >= l * p, r["config_id"]
        if r.get("outside_certified_region"):
            assert r["verdict"] == "REJECT", r["config_id"]
        else:
            assert r["verdict"] == "ADMIT", r["config_id"]
        assert r["per_period_energy"] == "reset"
        assert int(r["ckpt_ep"]) == 5000


def test_command_builder_golden_b4_d085():
    rows = load_table(TABLE)
    row = next(r for r in rows if r["config_id"] == "exp2_B4_DEM_d085")
    cmd = build_eval_command(row, "results_exp2_dryrun")
    s = " ".join(cmd)
    assert cmd[:6] == [
        "conda", "run", "--no-capture-output", "-n", "DARTS_nano", "env",
    ]
    assert "--eval_only" in cmd
    assert _flag(cmd, "--load_ckpt") == row["ckpt_dir"]
    assert _flag(cmd, "--load_ckpt_ep") == "5000"
    assert _flag(cmd, "--workload") == "B4_DEM"
    assert _flag(cmd, "--stress") == "medium"
    assert _flag(cmd, "--release_mode") == "periodic"
    assert _flag(cmd, "--release_period_ms") == str(int(row["P_ms"]))
    assert _flag(cmd, "--num_releases") == "30"
    assert _flag(cmd, "--relative_deadline_ms") == str(int(row["D_v_ms"]))
    assert _flag(cmd, "--per_period_energy") == "reset"
    assert _flag(cmd, "--max_episode_steps") == str(int(row["max_episode_steps"]))
    assert _flag(cmd, "--global_deadline_ms") == str(int(row["global_deadline_ms"]))
    assert _flag(cmd, "--eval_seed_base") == "100"
    assert _flag(cmd, "--eval_episodes") == "1"
    assert _flag(cmd, "--results_dir") == "results_exp2_dryrun"
    assert _flag(cmd, "--exp_name") == "exp2_B4_DEM_d085"
    assert "alloc_b4_identity.json" in _flag(cmd, "--alloc_map")
    assert "--release_period_ms" in s


def test_dry_run_prints_12_and_creates_no_dirs(tmp_path):
    results = tmp_path / "results_exp2_dry_test"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/sweep_exp2.py",
            "--dry-run",
            "--results_dir",
            str(results),
        ],
        cwd=_PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": _PROJECT_ROOT},
    )
    assert proc.returncode == 0, proc.stderr
    cmds = [ln for ln in proc.stdout.splitlines() if ln.startswith("conda ")]
    assert len(cmds) == 12, proc.stdout
    assert not results.exists()


def test_launcher_console_log_is_outside_results_root():
    """Nohup redirect must not pre-create a file inside RESULTS_DIR."""
    path = os.path.join(_PROJECT_ROOT, "sweep_exp2_launch.sh")
    text = open(path, encoding="utf-8").read()
    m = re.search(r"^CONSOLE_LOG=(.+)$", text, re.M)
    assert m, "CONSOLE_LOG assignment missing in sweep_exp2_launch.sh"
    rhs = m.group(1).strip()
    if len(rhs) >= 2 and rhs[0] == rhs[-1] and rhs[0] in "\"'":
        rhs = rhs[1:-1]
    assert not rhs.startswith("$RESULTS_DIR/"), rhs
    assert not rhs.startswith("${RESULTS_DIR}/"), rhs
    assert re.search(
        r'LAUNCH_CMD="\$LAUNCH_CMD > \$CONSOLE_LOG 2>&1 &"',
        text,
    ), "LAUNCH_CMD redirect target must be $CONSOLE_LOG"


def _flag(cmd, name):
    i = cmd.index(name)
    return cmd[i + 1]
