"""Unit tests for TB-1 analysis helpers (no torch, no sweep runs)."""

from __future__ import annotations

import ast
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.tb1_analysis import certified_miss_split, derived_metrics, _fmt_metric, load_manifest

# Paper-era pin shape: B1 last-row ep 2500, B2 last-row ep 5000. Dirs live under tmp_path.
_FIXTURE_CELLS = (
    ("B1_CDR", "medium", (100, 300, 500), 2500),
    ("B1_CDR", "heavy", (100, 300, 500), 2500),
    ("B2_LML", "medium", (100, 200, 300), 5000),
)


@pytest.fixture
def tb1_manifest(tmp_path: Path) -> dict:
    """In-memory / tmp_path TB-1 manifest with the production per-cell schema.

    Does not read the repo-root generated artifact (Jetson-absolute checkpoint
    paths, not required to be present on a fresh clone).
    """
    data = {}
    for bench, stress, seeds, ep in _FIXTURE_CELLS:
        for seed in seeds:
            ckpt_dir = tmp_path / "ckpts" / f"darts_{bench}_{stress}_seed{seed}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            # Empty log → ReferenceError (caught by check_kappa1), not FileNotFoundError.
            (ckpt_dir / "darts_log.csv").write_text("")
            data[f"{bench}|{stress}|{seed}"] = {
                "dir": str(ckpt_dir),
                "ep": ep,
                "bilevel_mode": "fixed_eta",
                "no_incentive": False,
            }
    path = tmp_path / "tb1_fixture_manifest.json"
    path.write_text(json.dumps(data))
    return load_manifest(str(path))


def test_certified_miss_split_sums():
    row = {
        "n_full_fidelity": "4",
        "n_ff_missed": "3",
        "n_miss_full_fidelity_late": "2",
    }
    certified, late, unreached = certified_miss_split(row)
    assert late == 2 / 4
    assert unreached == 1 / 4
    assert certified == late + unreached == 3 / 4


def test_certified_miss_split_zero_ff_uses_denom_one():
    row = {
        "n_full_fidelity": "0",
        "n_ff_missed": "0",
        "n_miss_full_fidelity_late": "0",
    }
    certified, late, unreached = certified_miss_split(row)
    assert late == 0.0
    assert unreached == 0.0
    assert certified == 0.0


def test_certified_miss_split_59col_is_nan():
    """B2 reference rows have neither n_miss_* nor n_ff_missed."""
    row = {"n_full_fidelity": "4", "_width": 59}
    certified, late, unreached = certified_miss_split(row)
    assert math.isnan(late) and math.isnan(unreached)
    assert _fmt_metric(late) == "n/a"
    assert _fmt_metric(unreached) == "n/a"


def test_certified_miss_split_63col_late_only():
    """AE 63-col has n_miss_full_fidelity_late but not n_ff_missed."""
    row = {
        "n_full_fidelity": "4",
        "n_miss_full_fidelity_late": "1",
        "_width": 63,
    }
    certified, late, unreached = certified_miss_split(row)
    assert late == 0.25
    assert math.isnan(unreached)
    assert _fmt_metric(unreached) == "n/a"


def test_derived_metrics_carries_split():
    row = {
        "n_bids_evaluated": "16",
        "n_bids_cert_pass": "16",
        "episode_span_ms": "1000",
        "exec_ms_A1": "10", "exec_ms_A2": "0", "exec_ms_A3": "0", "exec_ms_A4": "0",
        "travel_ms_A1": "0", "travel_ms_A2": "0", "travel_ms_A3": "0", "travel_ms_A4": "0",
        "rho_hat": "1.0",
        "task_miss_rate": "0.0",
        "n_full_fidelity": "4",
        "n_ff_missed": "1",
        "n_miss_full_fidelity_late": "1",
    }
    m = derived_metrics(row)
    assert m["certified_miss"] == m["certified_miss_late"] + m["certified_miss_unreached"]
    assert m["certified_miss_late"] == 0.25
    assert m["certified_miss_unreached"] == 0.0


def test_endpoint_report_missing_runs_is_insufficient(tmp_path, capsys, tb1_manifest):
    """5.1 guard: vacuous extension is forbidden when NO group can be judged."""
    from scripts.tb1_analysis import endpoint_report, parse_cells

    rc = endpoint_report(str(tmp_path), tb1_manifest, parse_cells("B1_CDR:medium"), 0.5, 2.0)
    captured = capsys.readouterr().out
    assert rc == 1
    assert "verdict skipped (missing:" in captured
    assert "EXTENSION VERDICT" in captured
    verdict = captured.split("EXTENSION VERDICT", 1)[1]
    assert "INSUFFICIENT DATA" in verdict
    assert "no group could be judged" in verdict
    assert "flat on all metrics" not in verdict
    assert "no left extension needed" not in verdict
    assert "run κ=0.25" not in verdict
    assert "run κ=3.0" not in verdict
    assert "cell verdict:" not in captured


def test_endpoint_report_incomplete_group_does_not_fail_complete_sibling(tmp_path, capsys, tb1_manifest):
    """5.1 guard: a sibling missing --high does not fail a complete group."""
    import csv
    from scripts.csv_schema_tb import MAIN_CSV_HEADER_TB76
    from scripts.tb1_analysis import endpoint_report, exp_name_for, parse_cells

    def _write_log(exp: str) -> None:
        d = tmp_path / exp
        d.mkdir(parents=True, exist_ok=True)
        row = {c: "0" for c in MAIN_CSV_HEADER_TB76}
        row.update({
            "episode": "1",
            "n_bids_evaluated": "16",
            "n_bids_cert_pass": "16",
            "episode_span_ms": "1000",
            "exec_ms_A1": "10",
            "rho_hat": "1.0",
            "task_miss_rate": "0.0",
            "n_full_fidelity": "4",
            "n_ff_missed": "0",
            "n_miss_full_fidelity_late": "0",
        })
        with open(d / "darts_log.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=MAIN_CSV_HEADER_TB76)
            w.writeheader()
            w.writerow(row)

    seeds = (100, 300, 500)
    for seed in seeds:
        for kappa in (1.0, 0.5, 2.0):
            _write_log(exp_name_for("B1_CDR", "medium", kappa, seed, 1.0))
        for kappa in (1.0, 0.5):  # beta=0.85 has no κ=2.0
            _write_log(exp_name_for("B1_CDR", "medium", kappa, seed, 0.85))

    rc = endpoint_report(str(tmp_path), tb1_manifest, parse_cells("B1_CDR:medium"), 0.5, 2.0)
    captured = capsys.readouterr().out
    assert rc == 0
    assert "verdict skipped (missing:" in captured
    assert "cell verdict:" in captured
    assert "INSUFFICIENT DATA" not in captured.split("EXTENSION VERDICT", 1)[1]
    assert "1 group(s) judged, 1 skipped" in captured


def test_metric_tolerance_floors_and_std():
    from scripts.tb1_analysis import COUNT_FLOOR, RATE_FLOOR, metric_tolerance

    # Constant rates: floor 0.05. Constant counts: floor 0.15.
    rate_rows = [{"task_miss_rate": "0.10"}] * 3
    assert metric_tolerance("task_miss_rate", rate_rows) == RATE_FLOOR
    count_rows = [{"n_full_fidelity": "4.0"}] * 3
    assert metric_tolerance("n_full_fidelity", count_rows) == COUNT_FLOOR
    # std-driven: 2*pop-std of {0, 1, 2} = 2*sqrt(2/3) ≈ 1.633 > 0.15
    spread = [
        {"n_full_fidelity": "0"},
        {"n_full_fidelity": "1"},
        {"n_full_fidelity": "2"},
    ]
    tol = metric_tolerance("n_full_fidelity", spread)
    assert tol > COUNT_FLOOR
    assert abs(tol - 2.0 * math.sqrt(2.0 / 3.0)) < 1e-9


def test_n_pending_is_not_gated():
    from scripts.tb1_analysis import GATED_METRICS, REGRESSION_METRICS

    assert "n_pending" in REGRESSION_METRICS
    assert "n_pending" not in GATED_METRICS
    assert "n_full_fidelity" in GATED_METRICS


def test_exp_name_beta_suffix_absent_at_unity():
    from scripts.tb1_analysis import exp_name_for, parse_run_dirname, plot_pdf_path

    unsuffixed = exp_name_for("B1_CDR", "heavy", 0.75, 100)
    assert unsuffixed == "tb1_B1_CDR_heavy_k0p75_seed100"
    assert parse_run_dirname(unsuffixed, "B1_CDR", "heavy") == (0.75, 100, 1.0)

    scaled = exp_name_for("B1_CDR", "heavy", 0.75, 100, 0.85)
    assert scaled == "tb1_B1_CDR_heavy_k0p75_seed100_b0p85"
    assert parse_run_dirname(scaled, "B1_CDR", "heavy") == (0.75, 100, 0.85)

    assert plot_pdf_path("/tmp", "B2_LML", "medium").endswith("tb1_F1_draft_B2_LML_medium.pdf")
    assert plot_pdf_path("/tmp", "B2_LML", "medium", 0.85).endswith(
        "tb1_F1_draft_B2_LML_medium_b0p85.pdf"
    )


def test_discover_kappas_filters_by_beta(tmp_path):
    from scripts.tb1_analysis import count_nonzero_beta_runs, discover_betas, discover_kappas

    for name in (
        "tb1_B2_LML_medium_k0p5_seed100",
        "tb1_B2_LML_medium_k0p75_seed100",
        "tb1_B2_LML_medium_k1p0_seed100",
        "tb1_B2_LML_medium_k1p5_seed100",
        "tb1_B2_LML_medium_k2p0_seed100",
        "tb1_B2_LML_medium_k3p0_seed100",
        "tb1_B2_LML_medium_k4p0_seed100",
        "tb1_B2_LML_medium_k1p0_seed100_b0p85",
    ):
        (tmp_path / name).mkdir()

    assert discover_kappas(str(tmp_path), "B2_LML", "medium") == [
        0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0,
    ]
    assert discover_kappas(str(tmp_path), "B2_LML", "medium", beta=0.85) == [1.0]
    assert discover_betas(str(tmp_path), "B2_LML", "medium") == [0.85, 1.0]
    assert count_nonzero_beta_runs(str(tmp_path), [("B2_LML", "medium")]) == 1


def test_check_kappa1_notes_skipped_nonzero_beta(tmp_path, capsys, tb1_manifest):
    from scripts.tb1_analysis import check_kappa1, parse_cells

    (tmp_path / "tb1_B1_CDR_medium_k1p0_seed100_b0p85").mkdir()
    check_kappa1(str(tmp_path), tb1_manifest, parse_cells("B1_CDR:medium"))
    captured = capsys.readouterr().out
    assert "skipped 1 beta!=1.0 run(s)" in captured
    assert "paper-era reference comparison is undefined" in captured


def test_cells_filter_does_not_overwrite_full_csv(tmp_path, tb1_manifest):
    import csv
    from scripts.csv_schema_tb import MAIN_CSV_HEADER_TB76
    from scripts.tb1_analysis import (
        DEFAULT_CELLS,
        exp_name_for,
        is_cells_filter,
        parse_cells,
        write_per_cell_csv,
    )

    assert not is_cells_filter(parse_cells(DEFAULT_CELLS))
    assert is_cells_filter(parse_cells("B1_CDR:heavy"))

    d = tmp_path / exp_name_for("B1_CDR", "medium", 1.0, 100)
    d.mkdir()
    row = {c: "0" for c in MAIN_CSV_HEADER_TB76}
    row.update({
        "episode": "1",
        "n_bids_evaluated": "16",
        "n_bids_cert_pass": "16",
        "episode_span_ms": "1000",
        "exec_ms_A1": "10",
        "rho_hat": "1.0",
        "task_miss_rate": "0.0",
        "n_full_fidelity": "4",
        "n_ff_missed": "0",
        "n_miss_full_fidelity_late": "0",
    })
    with open(d / "darts_log.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MAIN_CSV_HEADER_TB76)
        w.writeheader()
        w.writerow(row)

    full = tmp_path / "tb1_per_cell.csv"
    full.write_text("SENTINEL\n")
    write_per_cell_csv(
        str(tmp_path), tb1_manifest, parse_cells("B1_CDR:medium"),
        filename="tb1_per_cell_filtered.csv",
    )
    assert full.read_text() == "SENTINEL\n"
    filtered = tmp_path / "tb1_per_cell_filtered.csv"
    assert filtered.is_file()
    assert "SENTINEL" not in filtered.read_text()
    assert "bench" in filtered.read_text().splitlines()[0]


def test_fig_tb1_is_always_four_columns():
    from scripts.fig_tb1_tradeoff import COLUMNS, UNSOUND_LABEL, ZETA_TICKS

    assert len(COLUMNS) == 4
    assert COLUMNS[-1] == ("B1_CDR", "heavy", "B1 heavy (deadlines ×0.7)", 0.7)
    assert 0.25 in ZETA_TICKS
    assert "ζ" in UNSOUND_LABEL
    assert "κ" not in UNSOUND_LABEL and "β" not in UNSOUND_LABEL


def test_fig_tb1_zeta_ordinal_and_caption():
    from scripts.fig_tb1_tradeoff import caption_text, zeta_to_ordinal

    assert zeta_to_ordinal(0.25) == 0.0
    assert zeta_to_ordinal(0.5) == 1.0
    assert zeta_to_ordinal(1.0) == 3.0
    assert zeta_to_ordinal(1.25) == 3.5
    assert zeta_to_ordinal(4.0) == 7.0
    text = caption_text(0.0087)
    assert "0.0087" in text
    assert "ζ sampled at the labeled values (ordinal spacing)" in text
    assert "dotted line: deployed calibration ζ=1" in text.lower()
    assert "tightened-deadline probe" in text
    assert "ζ≥3" in text
    assert "U stays within" in text
    assert (
        "Shaded bands: seed min–max across the three frozen policies; "
        "hatched area: certificate evaluated below calibration (unsound)."
    ) in text


def test_fig_tb1_unsound_hatch_legend():
    from scripts.fig_tb1_tradeoff import (
        UNSOUND_ALPHA,
        UNSOUND_EDGE,
        UNSOUND_FACE,
        UNSOUND_HATCH,
        UNSOUND_LABEL,
        UNSOUND_PATCH_KW,
        UNSOUND_ZORDER,
        _legend_handles,
    )

    assert UNSOUND_FACE == "0.92"
    assert UNSOUND_EDGE == "0.55"
    assert UNSOUND_HATCH == "///"
    assert UNSOUND_ALPHA == 1.0
    assert UNSOUND_ZORDER == 0
    assert UNSOUND_PATCH_KW["hatch"] == "///"
    handles, labels = _legend_handles()
    assert labels[-1] == UNSOUND_LABEL == "unsound region (ζ<1)"
    swatch = handles[-1]
    assert swatch.get_hatch() == "///"
    fc = swatch.get_facecolor()
    assert fc[0] == fc[1] == fc[2]
    assert abs(fc[0] - 0.92) < 1e-6
    assert abs(fc[3] - 1.0) < 1e-6
    ec = swatch.get_edgecolor()
    assert abs(ec[0] - 0.55) < 1e-6


def _is_project_root_name(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "_PROJECT_ROOT"


def _unwrap_path_call(node: ast.AST) -> ast.AST:
    """Strip str(...) / Path(...) wrappers."""
    while isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in (
        "str",
        "Path",
    ) and node.args:
        node = node.args[0]
    return node


def _relpath_from_project_root_expr(node: ast.AST) -> str | None:
    """If `node` builds a path under `_PROJECT_ROOT`, return the posix relative path.

    Recognizes `os.path.join(_PROJECT_ROOT, ...)`, `_PROJECT_ROOT / ...`, and
    `Path(_PROJECT_ROOT) / ...`. Dynamic (non-string) segments return None.
    """
    node = _unwrap_path_call(node)
    if _is_project_root_name(node):
        return "."

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join":
        if not node.args:
            return None
        root = _unwrap_path_call(node.args[0])
        if not _is_project_root_name(root):
            return None
        parts: list[str] = []
        for arg in node.args[1:]:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                parts.append(arg.value.replace("\\", "/"))
            else:
                return None
        return "/".join(parts) if parts else "."

    parts = []
    cur = node
    while isinstance(cur, ast.BinOp) and isinstance(cur.op, ast.Div):
        right = cur.right
        if isinstance(right, ast.Constant) and isinstance(right.value, str):
            parts.append(right.value.replace("\\", "/"))
            cur = _unwrap_path_call(cur.left)
            continue
        return None
    if _is_project_root_name(cur):
        parts.reverse()
        return "/".join(parts) if parts else "."
    return None


def _require_git_checkout():
    """Skip the calling test when the project root is not the top level of a git work tree.

    The test compares files with what git tracks. A downloaded archive of the
    repository carries no git metadata, so there is nothing to compare with.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-prefix"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
    except OSError:
        pytest.skip("git is not installed")
    if proc.returncode != 0 or proc.stdout.strip():
        pytest.skip("the project root is not the top level of a git work tree")


def test_suite_does_not_read_untracked_project_paths():
    """No test may open a `_PROJECT_ROOT`-relative path that git does not track.

    Generated artifacts (checkpoint manifests, sweep trees) are machine-local; a
    fresh clone must run the suite without them.
    """
    _require_git_checkout()
    repo = Path(_PROJECT_ROOT)
    tracked = set(
        subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=repo, text=True
        ).split("\0")
    )
    tracked.discard("")
    tracked_dirs = {"."}
    for rel in tracked:
        parent = Path(rel).parent
        while True:
            tracked_dirs.add(parent.as_posix())
            if parent == Path("."):
                break
            parent = parent.parent

    offenders: list[str] = []
    for test_file in sorted((repo / "tests").glob("test_*.py")):
        tree = ast.parse(test_file.read_text(encoding="utf-8"), filename=str(test_file))
        for node in ast.walk(tree):
            rel = _relpath_from_project_root_expr(node)
            if rel is None:
                continue
            rel = rel.strip("/") or "."
            if rel == ".":
                continue
            if rel in tracked or rel in tracked_dirs:
                continue
            offenders.append(f"{test_file.relative_to(repo)}: {rel}")

    assert not offenders, (
        "tests read `_PROJECT_ROOT` paths that are absent from git "
        "(suite would fail on a fresh clone):\n  " + "\n  ".join(offenders)
    )

