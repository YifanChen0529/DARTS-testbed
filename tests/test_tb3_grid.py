"""TB-3 offline acceptance grid (M=200 x 15 beta x 5 cells) and F3 draft.

t1/t4 run ``grid`` into a tmp dir (seeds 0..19). t2/t5/t6 read committed
aggregates under ``docs/tb3/`` after the smoke-gate full-scale run. t3
reproduces the TB3-2 five-seed R lock from a tiny grid, not from untracked
raw output.
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.csv_schema_tb import TB3_CURVES_HEADER, TB3_FRONTIER_HEADER, TB3_OFFLINE_HEADER
from scripts.offline_test_harness import (
    BETA_GRID_TB3,
    GRID_CELLS_TB3,
    TESTS_DEFAULT,
    XI_SOURCE_DEFAULT,
    main as harness_main,
)


def _run_grid(out_dir: Path, seeds: str, **extra) -> None:
    curves = extra.get("curves", out_dir / "curves.csv")
    frontier = extra.get("frontier", out_dir / "frontier.csv")
    cells = extra.get("cells")
    beta_grid = extra.get("beta_grid")
    tests = extra.get("tests", TESTS_DEFAULT)
    argv = [
        "grid",
        "--xi-source",
        XI_SOURCE_DEFAULT,
        "--tests",
        tests,
        "--seeds",
        seeds,
        "--out",
        str(out_dir),
        "--curves",
        str(curves),
        "--frontier",
        str(frontier),
    ]
    if cells is not None:
        argv.extend(["--cells", cells])
    if beta_grid is not None:
        argv.extend(["--beta-grid", beta_grid])
    rc = harness_main(argv)
    assert rc == 0


def _read_grid(path: Path) -> list:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == TB3_OFFLINE_HEADER if rows else True
    return rows


def _tb3_2_r_values() -> dict:
    """Parse the five B1_CDR:heavy S0 declared R values from TB3-2_report.md."""
    text = Path(_PROJECT_ROOT, "docs", "tb3", "TB3-2_report.md").read_text(
        encoding="utf-8"
    )
    # | 0 | 222455.777299 | 1.8538 |
    found = {}
    for m in re.finditer(
        r"^\|\s*(\d+)\s*\|\s*([0-9]+\.[0-9]+)\s*\|\s*[0-9]+\.[0-9]+\s*\|",
        text,
        flags=re.M,
    ):
        seed = int(m.group(1))
        if seed in found or seed > 4:
            continue
        found[seed] = m.group(2)
        if len(found) == 5:
            break
    assert set(found) == {0, 1, 2, 3, 4}, found
    return found


def test_t1_invariants(tmp_path):
    """Fixed (cell, seed, test): one R; accept monotone; accept == R <= D."""
    _run_grid(tmp_path, "0..19")
    rows = _read_grid(tmp_path / "tb3_offline_grid.csv")
    n_cells = len(GRID_CELLS_TB3)
    n_seeds = 20
    n_beta = len(BETA_GRID_TB3)
    n_tests = 2
    assert len(rows) == n_cells * n_seeds * n_beta * n_tests

    groups = {}
    for row in rows:
        cell = f"{row['benchmark']}:{row['stress']}"
        key = (cell, int(row["instance_seed"]), row["test"])
        groups.setdefault(key, []).append(row)

    for key, grp in groups.items():
        grp_sorted = sorted(grp, key=lambda r: float(r["beta"]))
        rs = {r["R_G_off_ms"] for r in grp_sorted}
        assert len(rs) == 1, key
        prev_acc = -1
        prev_beta = None
        for r in grp_sorted:
            beta = float(r["beta"])
            d_e2e = float(r["D_e2e_ms"])
            r_ms = float(r["R_G_off_ms"])
            acc = int(r["accept_e2e"])
            # Deadlines in the row are already beta-scaled (MissionInstance.scaled).
            # Paper form: accept <=> R <= beta * D_nominal, and D_e2e = beta * D_nominal.
            assert acc == int(r_ms <= d_e2e)
            d_nom = d_e2e / beta
            assert acc == int(beta * d_nom >= r_ms)
            assert acc in (0, 1)
            assert acc >= prev_acc
            if prev_beta is not None:
                assert beta > prev_beta
            prev_acc = acc
            prev_beta = beta


def test_t2_saturation_gates():
    path = Path(_PROJECT_ROOT, "docs", "tb3", "tb3_acceptance_curves.csv")
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    by = {(r["cell"], r["test"], float(r["beta"])): r for r in rows}
    for cell in GRID_CELLS_TB3:
        for test in ("S0", "S1"):
            frac = float(by[(cell, test, 2.4)]["accept_frac_e2e"])
            assert frac == 1.0, (cell, test, frac)
    for test in ("S0", "S1"):
        for beta in BETA_GRID_TB3:
            if beta < 0.9:
                continue
            frac = float(by[("B4_DEM:medium", test, beta)]["accept_frac_e2e"])
            assert frac == 1.0, (test, beta, frac)
        frac_lo = float(by[("B1_CDR:heavy", test, 0.5)]["accept_frac_e2e"])
        assert frac_lo == 0.0, (test, frac_lo)


def test_t3_continuity_tb3_2(tmp_path):
    expected = _tb3_2_r_values()
    _run_grid(
        tmp_path,
        "0..4",
        cells="B1_CDR:heavy",
        beta_grid="1.0",
        tests="S0",
    )
    rows = _read_grid(tmp_path / "tb3_offline_grid.csv")
    got = {}
    for r in rows:
        assert r["benchmark"] == "B1_CDR" and r["stress"] == "heavy"
        assert r["test"] == "S0"
        assert r["xi_source"] == "declared"
        assert abs(float(r["beta"]) - 1.0) < 1e-12
        got[int(r["instance_seed"])] = r["R_G_off_ms"]
    assert set(got) == set(expected)
    for seed, rec in expected.items():
        # Match at the precision recorded in the TB3-2 report.
        assert format(float(got[seed]), f".{len(rec.split('.', 1)[1])}f") == rec, (
            seed,
            got[seed],
            rec,
        )


def test_t4_determinism(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _run_grid(a, "0..19")
    _run_grid(b, "0..19")
    bytes_a = (a / "tb3_offline_grid.csv").read_bytes()
    bytes_b = (b / "tb3_offline_grid.csv").read_bytes()
    assert bytes_a == bytes_b
    assert bytes_a.startswith(b"benchmark,")


def test_t5_curves_shape():
    curves_path = Path(_PROJECT_ROOT, "docs", "tb3", "tb3_acceptance_curves.csv")
    frontier_path = Path(_PROJECT_ROOT, "docs", "tb3", "tb3_frontier_summary.csv")
    with curves_path.open(newline="") as fh:
        curves = list(csv.DictReader(fh))
        assert list(curves[0].keys()) == TB3_CURVES_HEADER
    with frontier_path.open(newline="") as fh:
        frontier = list(csv.DictReader(fh))
        assert list(frontier[0].keys()) == TB3_FRONTIER_HEADER
    assert len(curves) == 5 * 2 * 15 == 150
    assert len(frontier) == 10
    cells = {r["cell"] for r in curves}
    tests = {r["test"] for r in curves}
    betas = {float(r["beta"]) for r in curves}
    assert cells == set(GRID_CELLS_TB3)
    assert tests == {"S0", "S1"}
    assert betas == set(BETA_GRID_TB3)
    for r in curves:
        assert int(r["n_instances"]) == 200
    for r in frontier:
        assert int(r["n_instances"]) == 200
        assert 0 <= int(r["n_S1_gt_S0"]) <= 200


def test_t6_figure_smoke(tmp_path):
    out_pdf = tmp_path / "FIG_tb3_f3_draft.pdf"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/fig_tb3_f3.py",
            "--curves",
            "docs/tb3/tb3_acceptance_curves.csv",
            "--out",
            str(out_pdf),
        ],
        cwd=_PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": _PROJECT_ROOT},
    )
    stdout = proc.stdout
    assert out_pdf.is_file() and out_pdf.stat().st_size > 0
    assert "[style] no shared helper" in stdout
    assert "fig1_b1_grid.py" in stdout
    assert "path-sum test" in stdout
    assert "per-node travel refinement" in stdout
    caption = out_pdf.with_name("FIG_tb3_f3_caption.txt").read_text(encoding="utf-8")
    assert "path-sum test" in stdout or "path-sum test" in caption
    assert "S0" not in caption
    assert "Lemma" in caption


def test_grid_row_order_is_canonical(tmp_path):
    _run_grid(tmp_path, "0..1", cells="B1_CDR:medium,B4_DEM:medium", beta_grid="0.5,1.0")
    rows = _read_grid(tmp_path / "tb3_offline_grid.csv")
    keys = [
        (
            f"{r['benchmark']}:{r['stress']}",
            int(r["instance_seed"]),
            float(r["beta"]),
            r["test"],
        )
        for r in rows
    ]
    assert keys == sorted(
        keys,
        key=lambda t: (
            ["B1_CDR:medium", "B4_DEM:medium"].index(t[0]),
            t[1],
            t[2],
            ["S0", "S1"].index(t[3]),
        ),
    )
