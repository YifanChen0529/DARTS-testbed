"""Slow end-to-end schema check against run_artifact_demo.py (simulator mode)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_artifact_demo_csv_schema_and_rt_ok(tmp_path: Path):
    """Reduced demo run: exit 0, EVAL_CSV_HEADER match, rt_bound_violations all 0."""
    pytest.importorskip("torch", reason="demo entry scripts require torch")

    # train_darts writes to join(PROJECT_ROOT, "results", exp_name). When exp_name
    # is absolute, os.path.join discards the prefix — so the workspace lands under
    # tmp_path without touching production code. Demo lookup uses the same absolute
    # segment via Path arithmetic when --results_dir is also absolute under tmp_path.
    exp_dir = tmp_path / "demo_schema_b1"
    cmd = [
        sys.executable,
        str(_PROJECT_ROOT / "run_artifact_demo.py"),
        "--benchmark",
        "B1",
        "--stress",
        "medium",
        "--episodes",
        "4",
        "--eval_interval",
        "2",
        "--eval_batch",
        "1",
        "--seed",
        "0",
        "--method",
        "darts",
        "--results_dir",
        str(tmp_path),
        "--exp_name",
        str(exp_dir),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    # Jetson/CUDA workstation envs sometimes need cusparselt on LD_LIBRARY_PATH.
    cusparse = Path(sys.prefix) / "lib" / "cusparselt"
    if cusparse.is_dir():
        env["LD_LIBRARY_PATH"] = str(cusparse) + os.pathsep + env.get("LD_LIBRARY_PATH", "")

    proc = subprocess.run(
        cmd,
        cwd=str(_PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"demo failed rc={proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    csv_path = exp_dir / "darts_log.csv"
    assert csv_path.is_file(), f"missing CSV: {csv_path}"

    from alg.train_darts import EVAL_CSV_HEADER

    df = pd.read_csv(csv_path)
    header = [c.strip() for c in df.columns.tolist()]
    assert header == list(EVAL_CSV_HEADER)

    assert len(df) >= 1
    viol = pd.to_numeric(df["rt_bound_violations"].replace("N/A", float("nan")), errors="coerce")
    assert (viol.fillna(0) == 0).all(), f"nonzero rt_bound_violations: {viol.tolist()}"
