"""Fixture-driven test for scripts/summarize_sweep.py last-row aggregation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_summarize_sweep_last_row_values(tmp_path: Path):
    """Whitespace-padded headers + N/A cell; per-cell row matches last CSV row."""
    cell_dir = tmp_path / "exp_1" / "B1" / "medium"
    cell_dir.mkdir(parents=True)
    log_path = cell_dir / "darts_log.csv"

    # Deliberately padded headers; second row is the last-row source of truth.
    # max_rt_ms on last row is N/A → summarizer coerces to None/empty.
    log_path.write_text(
        " episode , task_miss_rate , rt_bound_violations , max_rt_ms , "
        "A1_crash_rate , A2_crash_rate , A3_crash_rate , A4_crash_rate \n"
        "100, 0.50, 2.0, 1000.0, 0.1, 0.0, 0.0, 0.0\n"
        "200, 0.25, 0.0, N/A, 0.0, 0.0, 0.0, 0.0\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(_PROJECT_ROOT / "scripts" / "summarize_sweep.py"),
        "--results_dir",
        str(tmp_path),
        "--out_dir",
        str(out_dir),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    rc = subprocess.call(cmd, cwd=str(_PROJECT_ROOT), env=env)
    assert rc == 0, "summarize_sweep.py failed"

    per_cell = out_dir / "sweep_summary_per_cell.csv"
    assert per_cell.is_file()
    df = pd.read_csv(per_cell)
    assert len(df) == 1

    row = df.iloc[0]
    assert row["method"] == "darts"
    assert row["benchmark"] == "B1"
    assert row["stress"] == "medium"
    assert int(row["seed"]) == 100  # exp_1 → seed = 1 * 100
    assert int(row["last_ep"]) == 200
    assert float(row["task_miss_rate"]) == pytest.approx(0.25)
    assert float(row["rt_bound_violations"]) == pytest.approx(0.0)
    assert pd.isna(row["max_rt_ms"])
    assert float(row["A1_crash"]) == pytest.approx(0.0)
    assert float(row["A2_crash"]) == pytest.approx(0.0)
    assert float(row["A3_crash"]) == pytest.approx(0.0)
    assert float(row["A4_crash"]) == pytest.approx(0.0)
