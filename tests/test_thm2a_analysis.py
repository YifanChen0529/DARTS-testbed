"""EXP-2 Theorem 2(a) analysis and gate-evaluator tests."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.thm2a_analysis import (  # noqa: E402
    d1_prefix,
    d1_sha256,
    evaluate_gates,
    format_results,
    load_table,
    main as analysis_main,
)
from scripts.fig_exp2_response_cdf import rendered_strings  # noqa: E402

GATES_PATH = os.path.join(_PROJECT_ROOT, "docs", "exp2", "exp2_gates.md")
EXPECTED_D1_SHA256 = (
    "7ed827276a2603197bc2ca0594583cefcd8748ee70722739b40b49cf7f839f23"
)
FORBIDDEN = (
    (r"d030", "d030"),
    (r"d050", "d050"),
    (r"d070", "d070"),
    (r"d085", "d085"),
    (r"\bprobe\b", "probe"),
    (r"exp2_", "exp2_"),
    (r"config_id", "config_id"),
    (r"\bS0\b", "S0"),
    (r"\bS1\b", "S1"),
    (r"\brep\b", "rep"),
    (r"repeat", "repeat"),
)

CLASSES = ("J1", "J2", "J3", "J4")
ALLOC = {"J1": "agent_1", "J2": "agent_2", "J3": "agent_3", "J4": "agent_4"}


def _rec(**kwargs):
    base = {
        "config_id": "syn_admit",
        "workload": "B4_DEM",
        "verdict": "ADMIT",
        "outside_certified_region": False,
        "eval_seed_base": 100,
        "target_delta_sum": 0.85,
        "n_rows": 120,
        "n_miss": 0,
        "n_drain_fail": 0,
        "n_fallback_blocked": 0,
        "n_alloc_mismatch": 0,
        "has_release_log": True,
        "rc": "0",
        "provenance_ok": True,
        "miss_rows": [],
        "drain_fail_rows": [],
        "ratio_max": 0.2,
        "ratio_p99": 0.15,
        "ratio_max_by_class": {"J1": 0.2},
    }
    base.update(kwargs)
    return base


def test_ga_fail_on_admit_met_false():
    admit = _rec(n_miss=1, miss_rows=["J1#0"])
    probe = _rec(
        config_id="syn_reject",
        verdict="REJECT",
        outside_certified_region=True,
        target_delta_sum=None,
    )
    ev = evaluate_gates([admit, probe])
    assert ev["ga_pass"] is False
    assert ev["gb_pass"] is True
    assert ev["n_admit_miss"] == 1


def test_gb_fail_on_admit_drain_false():
    admit = _rec(n_drain_fail=1, drain_fail_rows=["J2#3"])
    probe = _rec(
        config_id="syn_reject",
        verdict="REJECT",
        outside_certified_region=True,
        target_delta_sum=None,
    )
    ev = evaluate_gates([admit, probe])
    assert ev["gb_pass"] is False
    assert ev["ga_pass"] is True


def test_probe_fabrications_do_not_flip_ga_gb():
    admit = _rec()
    probe = _rec(
        config_id="syn_reject",
        verdict="REJECT",
        outside_certified_region=True,
        target_delta_sum=None,
        n_miss=7,
        n_drain_fail=4,
        miss_rows=["J1#0"],
        drain_fail_rows=["J1#1"],
    )
    ev = evaluate_gates([admit, probe])
    assert ev["ga_pass"] is True
    assert ev["gb_pass"] is True
    assert ev["n_admit_miss"] == 0
    assert ev["n_admit_drain"] == 0


def _write_release_log(path: Path, *, met: bool = True, drain: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "workload", "stress", "seed", "ckpt_dir", "ckpt_ep",
        "P_ms", "L", "D_v_ms", "per_period_energy", "alloc_map_sha_or_path",
        "release_idx", "job_class", "instance_id", "agent_id",
        "r_ms", "commit_ms", "finish_ms", "deadline_ms", "met",
        "queue_empty_at_next_boundary", "battery_at_boundary_wh", "fallback_blocked",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for ell in range(30):
            for cls in CLASSES:
                r_ms = float(ell * 1000)
                w.writerow({
                    "workload": "B4_DEM",
                    "stress": "medium",
                    "seed": "100",
                    "ckpt_dir": "results/darts_B4_DEM_medium_seed100",
                    "ckpt_ep": "5000",
                    "P_ms": "100000",
                    "L": "30",
                    "D_v_ms": "100000",
                    "per_period_energy": "reset",
                    "alloc_map_sha_or_path": "alloc.json",
                    "release_idx": str(ell),
                    "job_class": cls,
                    "instance_id": f"{cls}#{ell}",
                    "agent_id": ALLOC[cls],
                    "r_ms": str(r_ms),
                    "commit_ms": str(r_ms + 1000),
                    "finish_ms": str(r_ms + 20000),
                    "deadline_ms": str(r_ms + 100000),
                    "met": "true" if met else "false",
                    "queue_empty_at_next_boundary": "true" if drain else "false",
                    "battery_at_boundary_wh": "0.4",
                    "fallback_blocked": "false",
                })


def _write_cfg(path: Path, *, seed: int, p: float, dv: float, alloc: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "release_period_ms": p,
        "workload_relative_deadline_ms": dv,
        "num_releases": 30,
        "eval_seed_base": seed,
        "load_ckpt_ep": 5000,
        "load_ckpt": "results/darts_B4_DEM_medium_seed100",
        "alloc_map": str(alloc),
    }), encoding="utf-8")


def _two_config_fixture(tmp: Path):
    alloc = tmp / "alloc.json"
    alloc.write_text(json.dumps(ALLOC), encoding="utf-8")
    table = {
        "configs": [
            {
                "config_id": "syn_admit",
                "workload": "B4_DEM",
                "alloc_map_file": str(alloc),
                "target_delta_sum": 0.85,
                "outside_certified_region": False,
                "D_v_ms": 100000.0,
                "P_ms": 100000.0,
                "L": 30,
                "delta_sum_factored": {
                    "agent_1": 0.85, "agent_2": 0.85,
                    "agent_3": 0.85, "agent_4": 0.85,
                },
                "verdict": "ADMIT",
                "ckpt_dir": "results/darts_B4_DEM_medium_seed100",
                "ckpt_ep": 5000,
                "eval_seed_base": 100,
            },
            {
                "config_id": "syn_reject",
                "workload": "B4_DEM",
                "alloc_map_file": str(alloc),
                "target_delta_sum": None,
                "outside_certified_region": True,
                "D_v_ms": 80000.0,
                "P_ms": 80000.0,
                "L": 30,
                "delta_sum_factored": {
                    "agent_1": 1.07, "agent_2": 1.07,
                    "agent_3": 1.07, "agent_4": 1.07,
                },
                "verdict": "REJECT",
                "ckpt_dir": "results/darts_B4_DEM_medium_seed100",
                "ckpt_ep": 5000,
                "eval_seed_base": 100,
            },
        ]
    }
    table_path = tmp / "table.json"
    table_path.write_text(json.dumps(table), encoding="utf-8")
    root = tmp / "results_exp2_20990101"
    for cid, p, dv in (("syn_admit", 100000.0, 100000.0), ("syn_reject", 80000.0, 80000.0)):
        rundir = root / cid
        _write_release_log(rundir / "release_log.csv")
        _write_cfg(rundir / "config.json", seed=100, p=p, dv=dv, alloc=alloc)
        (rundir / "darts_log.csv").write_text(
            "episode,rt_bound_violations\n5000,0\n", encoding="utf-8"
        )
    (root / "sweep_exp2_progress.log").write_text(
        "t syn_admit rc=0 wall_s=1 (1/2)\n"
        "t syn_reject rc=0 wall_s=1 (2/2)\n",
        encoding="utf-8",
    )
    gates_src = Path(GATES_PATH).read_text(encoding="utf-8")
    gates = tmp / "gates.md"
    gates.write_text(d1_prefix(gates_src), encoding="utf-8")
    return table_path, root, gates, tmp / "sup.csv", tmp / "sup.md"


def test_analysis_e2e_two_config_fixture(tmp_path):
    table_path, root, gates, sup_csv, sup_md = _two_config_fixture(tmp_path)
    before = d1_sha256(gates.read_text(encoding="utf-8"))
    rc = analysis_main([
        "--results_root", str(root),
        "--config_table", str(table_path),
        "--gates_path", str(gates),
        "--sup_csv", str(sup_csv),
        "--sup_md", str(sup_md),
        "--repo", _PROJECT_ROOT,
    ])
    assert rc == 0
    text = gates.read_text(encoding="utf-8")
    assert d1_sha256(text) == before
    assert "## RESULTS" in text
    assert "GI PASS: 2/2 rc=0" in text
    assert "GA PASS:" in text
    assert "GB PASS:" in text
    assert "GP (report-only, REJECT-verdict configs):" in text
    assert "syn_reject: misses=0" in text
    rows = list(csv.DictReader(sup_csv.open(encoding="utf-8")))
    assert [r["config_id"] for r in rows] == ["syn_admit", "syn_reject"]
    assert rows[0]["n_rows"] == "120"
    assert rows[0]["n_miss"] == "0"
    assert rows[1]["verdict"] == "REJECT"
    md = sup_md.read_text(encoding="utf-8")
    assert "syn_admit" in md and "syn_reject" in md


def test_d1_section_hash_pinned_and_stable_after_analysis(tmp_path):
    src = Path(GATES_PATH).read_text(encoding="utf-8")
    prefix = d1_prefix(src)
    assert hashlib.sha256(prefix.encode("utf-8")).hexdigest() == EXPECTED_D1_SHA256
    table_path, root, gates, sup_csv, sup_md = _two_config_fixture(tmp_path)
    assert d1_sha256(gates.read_text(encoding="utf-8")) == EXPECTED_D1_SHA256
    analysis_main([
        "--results_root", str(root),
        "--config_table", str(table_path),
        "--gates_path", str(gates),
        "--sup_csv", str(sup_csv),
        "--sup_md", str(sup_md),
        "--repo", _PROJECT_ROOT,
    ])
    assert d1_sha256(gates.read_text(encoding="utf-8")) == EXPECTED_D1_SHA256


def test_figure_label_strings_have_no_internal_identifiers():
    from scripts.fig_exp2_response_cdf import CAPTION, LEGEND_LABELS, PANEL_TITLES

    caption = Path(
        _PROJECT_ROOT, "figures", "FIG_exp2_response_cdf_caption.txt"
    )
    blobs = [rendered_strings()]
    if caption.is_file():
        blobs.append(caption.read_text(encoding="utf-8"))
    hay = "\n".join(blobs)
    for pat, name in FORBIDDEN:
        assert re.search(pat, hay) is None, f"forbidden {name!r} in paper-facing strings"
    assert "no plotted job release falls in the omitted interval" in hay
    assert list(LEGEND_LABELS) == [
        "admitted (density increases left to right)",
        "outside the certified region",
        "second evaluation seed",
    ]
    for lab in LEGEND_LABELS:
        assert re.search(r"\d", lab) is None, lab
    assert list(PANEL_TITLES) == ["B3", "B4"]
    caption_text = caption.read_text(encoding="utf-8") if caption.is_file() else CAPTION
    name_hay = "\n".join([*PANEL_TITLES, CAPTION, caption_text])
    for word in ("Inspection", "Monitoring", "Survey", "Pipeline", "Material"):
        assert word not in name_hay, f"benchmark expansion {word!r} in titles or caption"


def test_broken_axis_self_check_uses_table_maxima_not_sweep_dir():
    from scripts.fig_exp2_response_cdf import assert_ratios_fit_break

    try:
        assert_ratios_fit_break([{"values": [0.10, 0.80]}])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a ratio inside (0.72, 0.96)")
    try:
        assert_ratios_fit_break([{"values": [0.10, 0.99]}])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a ratio >= 0.96")

    path = os.path.join(
        _PROJECT_ROOT, "docs", "exp2", "exp2_supplementary_table.csv"
    )
    maxima = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            maxima.append(float(row["ratio_max"]))
    assert maxima, "supplementary table has no ratio_max column"
    table_max = max(maxima)
    assert abs(table_max - 0.6846153846153846) < 1e-12
    assert_ratios_fit_break([{"values": maxima}])


def test_config_table_still_loads():
    rows = load_table(
        os.path.join(_PROJECT_ROOT, "docs", "exp2", "exp2_config_table.json")
    )
    assert len(rows) == 12
    assert format_results(
        evaluate_gates([_rec()]),
        [_rec()],
    ).startswith("GI ")
