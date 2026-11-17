"""TB-3 P6 analysis: partition identity, over-rejection, ceiling, F3 symbols.

Does not open untracked result trees. Committed CSVs under docs/tb3/ and
figures/ are the inputs. Grid SHA lives in scripts/tb3_analysis.py:main.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.csv_schema_tb import (  # noqa: E402
    TB3_ADJUDICATION_HEADER,
    TB3_ADJUDICATION_V2_HEADER,
    TB3_CEILING_HEADER,
    TB3_OVERREJ_HEADER,
)
from scripts.fig_tb3_f3 import (  # noqa: E402
    CAPTION,
    CAPTION_ADDENDUM,
    rendered_strings,
)
from scripts.tb3_analysis import (  # noqa: E402
    BOOTSTRAP_SEED,
    EXPECTED_ADJ_V1_SHA256,
    EXPECTED_OVER_REJECTION,
    N_BOOTSTRAP,
    PARTITION_TOL,
    R1_TOL,
    _aligned_bits,
    adjudicate_instance_v2,
    bootstrap_over_rejection,
    build_adjudication,
    build_ceiling,
    build_overrejection,
    clopper_pearson,
    fmt_float,
    global_timing_conditions,
    partition_residual,
    point_over_rejection_from_feas,
    r1_census,
    r1_deviation,
    r1_holds_run,
    read_csv,
    residual_census,
    residual_of_run,
    sha256_file,
    write_csv,
)

DOCS = Path(_PROJECT_ROOT) / "docs" / "tb3"
RUNS_CSV = DOCS / "tb3_empirical_runs.csv"
FEAS_CSV = DOCS / "tb3_empirical_feasibility.csv"
OVER_CSV = DOCS / "tb3_overrejection.csv"
ADJ_CSV = DOCS / "tb3_soundness_adjudication.csv"
ADJ_V2_CSV = DOCS / "tb3_soundness_adjudication_v2.csv"
CEIL_CSV = DOCS / "tb3_empirical_ceiling.csv"
CAPTION_PATH = Path(_PROJECT_ROOT) / "figures" / "FIG_tb3_f3_caption.txt"

FORBIDDEN_TOKENS = (
    "beta",
    "\u03b2",
    "S0",
    "S1",
    "10 instances",
    "offline acceptance ratio",
)


def _synthetic_run(**overrides):
    row = {
        "cell": "B1_CDR:medium",
        "beta": "1.0",
        "instance_seed": "0",
        "ckpt_seed": "100",
        "task_miss_rate": "0.0",
        "n_miss_full_fidelity_late": "0.0",
        "n_miss_n1_drop": "0.0",
        "n_miss_n2_fallback_insufficient": "0.0",
        "n_miss_n3_unreached": "0.0",
        "rt_bound_violations": "0.0",
    }
    row.update(overrides)
    return row


def test_a1_partition_identity_synthetic_and_real():
    clean = _synthetic_run()
    assert abs(residual_of_run(clean)) <= PARTITION_TOL

    injected = _synthetic_run(task_miss_rate="0.5")  # lhs=2, rhs=0
    assert partition_residual(0.5, 0.0, 0.0, 0.0, 0.0) == pytest.approx(2.0)
    census_syn, max_r, n_nz = residual_census([clean, injected])
    assert n_nz == 1
    assert max_r == pytest.approx(2.0)
    caught = [r for r in census_syn if r["partition_residual"] > PARTITION_TOL]
    assert len(caught) == 1
    assert caught[0]["ckpt_seed"] == 100

    runs = read_csv(RUNS_CSV)
    assert len(runs) == 180
    census, max_real, n_real = residual_census(runs)
    assert len(census) == 180
    # Count is computed from the committed runs CSV, not assumed.
    assert n_real == sum(
        1 for r in census if r["partition_residual"] > PARTITION_TOL
    )
    assert n_real == 96
    assert max_real == pytest.approx(2.4)
    assert all(r["partition_residual"] >= -PARTITION_TOL for r in census)


def test_a2_overrejection_points_match_feas_and_prompt():
    feas = read_csv(FEAS_CSV)
    recomputed = point_over_rejection_from_feas(feas)
    assert set(recomputed) == set(EXPECTED_OVER_REJECTION)
    for key, (n_rej, n_feas) in EXPECTED_OVER_REJECTION.items():
        got_rej, got_feas, hat = recomputed[key]
        assert (got_rej, got_feas) == (n_rej, n_feas)
        assert hat == pytest.approx(n_rej / n_feas)

    over_rows = read_csv(OVER_CSV)
    assert list(over_rows[0].keys()) == TB3_OVERREJ_HEADER
    by = {(r["cell"], r["test"], float(r["beta"])): r for r in over_rows}
    for (cell, test, beta), (n_rej, n_feas) in EXPECTED_OVER_REJECTION.items():
        r = by[(cell, test, beta)]
        assert int(r["n_rej_feas"]) == n_rej
        assert int(r["n_feasible_e2e_all"]) == n_feas
        assert float(r["over_rejection_hat"]) == pytest.approx(n_rej / n_feas)


def test_a3_bootstrap_determinism(tmp_path):
    feas = read_csv(FEAS_CSV)
    a = build_overrejection(feas)
    b = build_overrejection(feas)
    p1 = tmp_path / "a.csv"
    p2 = tmp_path / "b.csv"
    write_csv(p1, TB3_OVERREJ_HEADER, a)
    write_csv(p2, TB3_OVERREJ_HEADER, b)
    assert p1.read_bytes() == p2.read_bytes()

    bits_f = _aligned_bits(feas, "B1_CDR:medium", 1.0, "feasible_e2e_all")
    bits_a = _aligned_bits(feas, "B1_CDR:medium", 1.0, "s0_accept_e2e")
    c1 = bootstrap_over_rejection(
        bits_f, bits_a, np.random.default_rng(BOOTSTRAP_SEED)
    )
    c2 = bootstrap_over_rejection(
        bits_f, bits_a, np.random.default_rng(BOOTSTRAP_SEED)
    )
    assert fmt_float(c1[0]) == fmt_float(c2[0])
    assert fmt_float(c1[1]) == fmt_float(c2[1])
    assert c1[2] == c2[2]
    assert c1[2] < N_BOOTSTRAP  # some defined draws exist


def test_a4_ceiling_header_and_feas_counts():
    feas = read_csv(FEAS_CSV)
    rows = read_csv(CEIL_CSV)
    assert list(rows[0].keys()) == TB3_CEILING_HEADER
    rebuilt = build_ceiling(feas)
    assert len(rows) == len(rebuilt) == 6
    for got, exp in zip(rows, rebuilt):
        assert got["cell"] == exp["cell"]
        assert float(got["beta"]) == pytest.approx(float(exp["beta"]))
        assert int(got["n_instances"]) == 10
        assert int(got["n_feasible"]) == int(exp["n_feasible"])
        n_from_feas = int(
            _aligned_bits(
                feas, got["cell"], float(got["beta"]), "feasible_e2e_all"
            ).sum()
        )
        assert int(got["n_feasible"]) == n_from_feas
        assert float(got["feasible_frac"]) == pytest.approx(n_from_feas / 10.0)
        assert float(got["ci_lo"]) <= float(got["feasible_frac"]) <= float(
            got["ci_hi"]
        )


def test_a5_timing_violation_iff_global_conditions():
    runs = read_csv(RUNS_CSV)
    adj = read_csv(ADJ_CSV)
    assert list(adj[0].keys()) == TB3_ADJUDICATION_HEADER
    cond = global_timing_conditions(runs, adj)
    has_tv = any(r["verdict"] == "TIMING_VIOLATION" for r in adj)
    assert has_tv == cond["has_timing_violation"]
    assert has_tv == (not cond["all_three_hold"])
    ffl0 = sum(float(r["n_miss_full_fidelity_late"]) for r in runs) == 0.0
    rt0 = sum(float(r["rt_bound_violations"]) for r in runs) == 0.0
    acc_resid = any(
        int(r["accept_e2e"]) and float(r["partition_residual_max"]) > PARTITION_TOL
        for r in adj
    )
    assert ffl0 and rt0
    assert acc_resid == cond["accepted_residual_gt_tol"]
    rebuilt = build_adjudication(runs, read_csv(FEAS_CSV))
    assert [r["verdict"] for r in rebuilt] == [r["verdict"] for r in adj]


def test_f3_symbol_gates_on_caption_and_rendered_text():
    caption = CAPTION_PATH.read_text(encoding="utf-8")
    assert CAPTION in caption
    assert CAPTION_ADDENDUM.strip() in caption
    blob = "\n".join(
        [
            caption,
            rendered_strings(empirical=False),
            rendered_strings(empirical=True),
        ]
    )
    for tok in FORBIDDEN_TOKENS:
        assert tok not in blob, f"forbidden token {tok!r} in paper-facing text"
        assert tok not in caption
        assert tok not in rendered_strings(empirical=True)
        assert tok not in rendered_strings(empirical=False)
    assert "deadline scale" in rendered_strings(empirical=True)
    assert "empirical feasibility (all-clean)" in rendered_strings(empirical=True)
    assert "empirical feasibility (all-clean, 10 instances)" not in rendered_strings(
        empirical=True
    )
    assert "empirical feasibility (all-clean)" not in rendered_strings(empirical=False)
    assert caption.count("50 instances") == 1
    assert "fraction of instances" in rendered_strings(empirical=True)


def test_b1_r1_holds_on_all_180_committed_runs():
    runs = read_csv(RUNS_CSV)
    n_pass, n_fail, max_dev, fails = r1_census(runs)
    print(f"R1 n_pass={n_pass}/180 n_fail={n_fail} max_|dev|={max_dev}")
    assert n_pass + n_fail == 180
    assert n_pass == 180
    assert n_fail == 0
    assert fails == []
    assert max_dev < R1_TOL


def test_b2_v1_timing_violations_flip_to_energy_scope():
    rows = read_csv(ADJ_V2_CSV)
    assert list(rows[0].keys()) == TB3_ADJUDICATION_V2_HEADER
    v1_tv = [r for r in rows if r["verdict_v1"] == "TIMING_VIOLATION"]
    assert len(v1_tv) == 23
    bad = []
    for r in v1_tv:
        if r["verdict_v2"] != "energy_scope_miss" or int(r["r1_holds"]) != 1:
            bad.append(
                (
                    r["cell"],
                    r["beta"],
                    r["test"],
                    r["instance_seed"],
                    r["verdict_v1"],
                    r["verdict_v2"],
                    r["r1_holds"],
                )
            )
    if bad:
        print("b2 disagreements:")
        for rec in bad:
            print(f"  {rec}")
    assert bad == []


def test_b3_v1_csv_sha_frozen():
    digest = sha256_file(ADJ_CSV)
    assert digest == EXPECTED_ADJ_V1_SHA256


def test_b4_jobcount_zero_episode_fails_r1_and_stays_tv():
    # One of five episodes misses e2e with zero per-job channel mass:
    # residual = 0.2 * 4 = 0.8 (the step-0.8 channel).
    injected = _synthetic_run(task_miss_rate="0.2")
    assert residual_of_run(injected) == pytest.approx(0.8)
    assert abs(r1_deviation(injected)) == pytest.approx(0.2)
    assert not r1_holds_run(injected)
    extra = adjudicate_instance_v2(1, 0, [injected])
    assert extra["r1_holds"] is False
    assert extra["verdict_v2"] == "TIMING_VIOLATION"


def test_b5_caption_grep_gate_and_new_sentences():
    caption = CAPTION_PATH.read_text(encoding="utf-8")
    assert "Accepted instances that are not all-clean fail only through" in caption
    assert "fail only through energy-unreached jobs" in caption
    assert "fail only through single energy-unreached" not in caption
    assert "The response-time test contains no energy term" in caption
    assert "energy admission is owned by the runtime energy predicate" in caption
    assert "partition residual" not in caption
    blob = "\n".join(
        [caption, rendered_strings(empirical=True), rendered_strings(empirical=False)]
    )
    for tok in FORBIDDEN_TOKENS:
        assert tok not in blob, f"forbidden token {tok!r} in paper-facing text"
    assert CAPTION_ADDENDUM.strip() in caption


def test_q4_clopper_pearson_fixtures_and_monotonicity():
    cases = {
        (3, 10): (0.0667, 0.6525),
        (1, 10): (0.0025, 0.4450),
    }
    for (k, n), (elo, ehi) in cases.items():
        for numpy_only in (False, True):
            lo, hi = clopper_pearson(k, n, numpy_only=numpy_only)
            assert round(lo, 4) == elo
            assert round(hi, 4) == ehi
    lo0, hi0 = clopper_pearson(0, 200)
    assert lo0 == 0.0
    assert round(hi0, 4) == 0.0183
    _lo_n, hi_n = clopper_pearson(0, 200, numpy_only=True)
    assert round(hi_n, 4) == 0.0183
    a = clopper_pearson(3, 10)
    b = clopper_pearson(30, 100)
    assert (b[1] - b[0]) < (a[1] - a[0])


def test_q5_v2_rule_on_accepted_infeasible_and_r1_fixture():
    clean_miss = _synthetic_run(
        task_miss_rate="0.2",
        n_miss_n3_unreached="0.2",
    )
    assert r1_holds_run(clean_miss)
    extra = adjudicate_instance_v2(1, 0, [clean_miss, clean_miss, clean_miss])
    assert extra["verdict_v2"] == "energy_scope_miss"
    extra_ok = adjudicate_instance_v2(1, 1, [clean_miss, clean_miss, clean_miss])
    assert extra_ok["verdict_v2"] == "consistent"
    injected = _synthetic_run(task_miss_rate="0.2")
    extra_tv = adjudicate_instance_v2(1, 0, [injected])
    assert extra_tv["verdict_v2"] == "TIMING_VIOLATION"
    assert extra_tv["r1_holds"] is False


def test_q6_f3_caption_cp_and_sample_size_sentences():
    caption = CAPTION_PATH.read_text(encoding="utf-8")
    assert "bars are exact 95% Clopper-Pearson intervals" in caption
    assert "the deployed-deadline point aggregates 50 instances" in caption.lower()
    assert "the two relaxed points 10 each" in caption
    assert "fail only through energy-unreached jobs" in caption
    assert "single energy-unreached" not in caption
    blob = "\n".join(
        [caption, rendered_strings(empirical=True), rendered_strings(empirical=False)]
    )
    for tok in FORBIDDEN_TOKENS:
        assert tok not in blob


EXPECTED_TB3_7_STOP_SHA256 = (
    "3a39bc6a60f33b2f5916cfba00a37f0329b7ceabd99eb89c9426b4f18486a45a"
)
STOP_REPORT = DOCS / "TB3-7_report.md"
RUNS_P7_CSV = DOCS / "tb3_empirical_runs_p7.csv"
FEAS_P7_CSV = DOCS / "tb3_empirical_feasibility_p7.csv"
ADJ_P7_CSV = DOCS / "tb3_soundness_adjudication_p7.csv"


def test_r3_known_strict_r1_failures_pass_g5_on_rejected():
    from scripts.tb3_analysis import g5_holds_run, r1_deviation, R1_TOL

    runs = read_csv(RUNS_P7_CSV)
    feas = read_csv(FEAS_P7_CSV)
    adj = read_csv(ADJ_P7_CSV)
    fails = [r for r in runs if abs(r1_deviation(r)) >= R1_TOL]
    assert fails, "expected strict-R1 failures in the merged p7 runs CSV"
    by_feas = {
        (f["cell"], f["beta"], int(f["instance_seed"])): f for f in feas
    }
    for row in fails:
        assert g5_holds_run(row), (
            f"{row['cell']} i{row['instance_seed']}c{row['ckpt_seed']} "
            f"dev={r1_deviation(row)}"
        )
        f = by_feas[(row["cell"], row["beta"], int(row["instance_seed"]))]
        assert int(f["s0_accept_e2e"]) == 0 and int(f["s1_accept_e2e"]) == 0
        inst_adj = [
            a
            for a in adj
            if a["cell"] == row["cell"]
            and a["beta"] == row["beta"]
            and int(a["instance_seed"]) == int(row["instance_seed"])
        ]
        assert inst_adj
        assert all(a["verdict_v2"] == "consistent" for a in inst_adj)


def test_r5_caption_grep_gates_and_amended_sentences():
    caption = CAPTION_PATH.read_text(encoding="utf-8")
    assert "bars are exact 95% Clopper-Pearson intervals" in caption
    assert "the deployed-deadline point aggregates 50 instances" in caption.lower()
    assert "the two relaxed points 10 each" in caption
    assert "fail only through energy-unreached jobs" in caption
    blob = "\n".join(
        [caption, rendered_strings(empirical=True), rendered_strings(empirical=False)]
    )
    for tok in FORBIDDEN_TOKENS:
        assert tok not in blob
    assert CAPTION_ADDENDUM.strip() in caption


def test_r6_stop_record_byte_identical():
    assert STOP_REPORT.is_file()
    assert sha256_file(STOP_REPORT) == EXPECTED_TB3_7_STOP_SHA256

