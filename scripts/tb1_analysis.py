#!/usr/bin/env python3
"""TB-1 sweep analysis: κ=1.0 regression gate and κ endpoint saturation report.

Two modes:

``--check-kappa1``
    Regress the sweep's κ=1.0 cells against the paper-era reference rows in the manifest
    checkpoint dirs. κ=1.0 must reproduce published behaviour; if it does not, the knob is
    not inert at unity and every other κ column is meaningless.

``--endpoint-report``
    For each (bench, stress) cell, compare κ=0.5 and κ=2.0 against κ=1.0 on the saturation
    metrics and emit a per-cell D1 extension verdict. A right endpoint that is flat on
    every metric *for that cell* calls for a {3.0} run on that cell. Admission saturation
    (bid_accept >= 0.999 at κ=1.0) means the left side is degenerate on-policy, so a
    {0.25} run is not applicable; otherwise a flat left endpoint (no certified miss
    below κ=1) still calls for {0.25}. No sweep-code change is needed for either;
    ``sweep_tb1.py --kappas`` already takes arbitrary lists. ``--plot`` writes a vector
    PDF per (bench, stress, β) under the results dir.

Reading the reference logs safely
---------------------------------
The manifest dirs' darts_log.csv files are append-mode survivors of several run
generations and interleave three row widths:

  59 cols  the 63-column schema minus the four n_miss_* columns
  62 cols  the 59-column schema with cert_attempts / cert_rejects / cert_reject_rate
           inserted before max_t_auc_ms, shifting every later field by 3
  63 cols  the AE-validated schema

Each file carries exactly ONE header row, at the top, describing only one of those widths.
Rows of the other widths have no header of their own anywhere in the file, so "align each
row to the most recent header in the file" is not sufficient on its own -- canonical
headers per width have to be supplied, which is what WIDTH_HEADERS does. They are derived
from the single source of truth in scripts/csv_schema_tb.py rather than re-typed, and are
cross-checked against whichever header the file does carry.

Reading positionally instead corrupts exactly the fields the regression consumes
(n_nondegradable_*, rt_bound_violations, crash rates, A_k buckets) -- for instance it
reports max_A_k of about 21 on B2 files, because a 59-column row read against the 62-column
header picks up values three fields to the left.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.build_tb1_manifest import _raw_last_episode  # noqa: E402  (raw-tail reader)
from scripts.csv_schema_tb import MAIN_CSV_HEADER_TB76  # noqa: E402

DEFAULT_MANIFEST = os.path.join(_PROJECT_ROOT, "ckpt_manifest_tb1.json")
DEFAULT_CELLS = "B1_CDR:medium,B1_CDR:heavy,B2_LML:medium"

# ---------------------------------------------------------------------------
# Canonical headers per schema era, derived from the 76-column source of truth.
# ---------------------------------------------------------------------------
HEADER_63 = list(MAIN_CSV_HEADER_TB76[:63])
_N_MISS_COLS = (
    "n_miss_n1_drop",
    "n_miss_n2_fallback_insufficient",
    "n_miss_n3_unreached",
    "n_miss_full_fidelity_late",
)
HEADER_59 = [c for c in HEADER_63 if c not in _N_MISS_COLS]
_CERT_COLS = ("cert_attempts", "cert_rejects", "cert_reject_rate")
_CERT_INSERT_BEFORE = "max_t_auc_ms"
HEADER_62 = list(HEADER_59)
HEADER_62[HEADER_62.index(_CERT_INSERT_BEFORE):HEADER_62.index(_CERT_INSERT_BEFORE)] = list(_CERT_COLS)

WIDTH_HEADERS: Dict[int, List[str]] = {
    59: HEADER_59,
    62: HEADER_62,
    63: HEADER_63,
    76: list(MAIN_CSV_HEADER_TB76),
}

# Wall-clock perf_counter measurements. Nondeterministic across processes by nature, so
# they are excluded from every row-level comparison and reported as diagnostics only.
TIMING_COLS = [
    "max_t_auc_ms", "mean_t_auc_ms",
    "max_delta_auc_ms", "mean_delta_auc_ms",
    "t_auc_Ak1_mean_ms", "t_auc_Ak2_mean_ms",
    "t_auc_Ak3_mean_ms", "t_auc_Ak4plus_mean_ms",
]

# Metrics the κ=1.0 regression gate compares. All exist in every reference era.
# n_pending is listed for display but is not gated: it is complementary to
# n_full_fidelity (sum ~= n_total_jobs), so a single deviation must not yield two FAILs.
REGRESSION_METRICS = [
    "task_miss_rate",
    "rho_hat",
    "n_full_fidelity",
    "n_degradable_fallback",
    "n_nondegradable_n1",
    "n_nondegradable_n2",
    "n_pending",
    "rt_bound_violations",
    "max_A_k",
    "mean_A_k",
    "A1_crash_rate", "A2_crash_rate", "A3_crash_rate", "A4_crash_rate",
]
GATED_METRICS = [m for m in REGRESSION_METRICS if m != "n_pending"]

RATE_FLOOR = 0.05    # rate-like metrics bounded in [0, 1]
COUNT_FLOOR = 0.15   # per-episode counts
COUNT_METRICS = frozenset({
    "n_full_fidelity",
    "n_pending",
    "n_degradable_fallback",
    "n_nondegradable_n1",
    "n_nondegradable_n2",
    "n_total_jobs",
    "rt_bound_violations",
    "max_A_k",
    "mean_A_k",
})

ADMISSION_SATURATED = 0.999


class ReferenceError(RuntimeError):
    """Raised when a reference log cannot be read the way the gate requires."""


# ---------------------------------------------------------------------------
# Schema-exact reading
# ---------------------------------------------------------------------------

def _to_float(text: str) -> float:
    t = (text or "").strip()
    if t in ("", "N/A", "nan", "None"):
        return float("nan")
    try:
        return float(t)
    except ValueError:
        return float("nan")


def read_rows_schema_exact(path: str) -> List[Dict[str, Any]]:
    """Every data row of a darts_log.csv, keyed by name against its own era's header.

    Each row is matched to a canonical header by its width. The file's own header row (of
    whichever single width it happens to be) is verified against the canonical header for
    that width, so a drifted canonical list is caught rather than silently mis-keying.
    """
    with open(path, newline="") as fh:
        raw = [r for r in csv.reader(fh) if r and r[0].strip()]
    if not raw:
        raise ReferenceError(f"{path}: empty")

    out: List[Dict[str, Any]] = []
    for lineno, row in enumerate(raw):
        width = len(row)
        header = WIDTH_HEADERS.get(width)
        if header is None:
            raise ReferenceError(
                f"{path}:{lineno + 1}: unknown schema width {width} "
                f"(known: {sorted(WIDTH_HEADERS)})"
            )
        if row[0].strip() == "episode":
            if [c.strip() for c in row] != header:
                raise ReferenceError(
                    f"{path}:{lineno + 1}: in-file {width}-column header disagrees with the "
                    "canonical header derived from scripts/csv_schema_tb.py; the schema "
                    "reconstruction is stale"
                )
            continue
        rec: Dict[str, Any] = {"_width": width, "_lineno": lineno + 1}
        for name, cell in zip(header, row):
            rec[name] = cell
        out.append(rec)
    return out


def trailing_generation(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The last run generation appended to a reference log.

    A generation is a maximal trailing block that shares one schema width AND whose
    episode column is strictly increasing. Both conditions are needed: width alone would
    merge B2_LML seed 200's two consecutive 59-column runs (episodes 500..5000 twice) into
    one 20-row block, and episode monotonicity alone would merge eras that happen to
    continue the numbering.
    """
    if not rows:
        raise ReferenceError("no data rows")
    width = rows[-1]["_width"]
    i = len(rows) - 1
    while i - 1 >= 0:
        prev = rows[i - 1]
        if prev["_width"] != width:
            break
        if _to_float(prev["episode"]) >= _to_float(rows[i]["episode"]):
            break
        i -= 1
    return list(rows[i:])


def reference_window(log_path: str, pinned_ep: int) -> Dict[str, Any]:
    """Reference rows and tolerance for one manifest cell.

    Steps, in order:
      1. isolate the trailing run generation (see trailing_generation);
      2. hard-fail unless its last-row episode equals the manifest's pinned epoch;
      3. reference = the last TWO rows, i.e. episodes {ep-500, ep} at the eval cadence;
      4. per-metric tolerance = max(floor_m, 2*std of metric m over the last THREE rows),
         with floor_m = 0.05 for rates and 0.15 for counts (see metric_tolerance).

    Rows outside the trailing generation are never read for regression purposes: in the B1
    files the rows at episodes {4500, 5000} belong to an earlier pre-Path1 run under a
    different policy and code generation.
    """
    rows = read_rows_schema_exact(log_path)
    block = trailing_generation(rows)
    last_ep = int(_to_float(block[-1]["episode"]))

    raw_ep = _raw_last_episode(log_path)
    if raw_ep is not None and raw_ep != last_ep:
        raise ReferenceError(
            f"{log_path}: raw tail read says ep{raw_ep} but the trailing generation ends "
            f"at ep{last_ep}; segment logic is inconsistent with the file"
        )
    if last_ep != int(pinned_ep):
        raise ReferenceError(
            f"{log_path}: trailing {block[0]['_width']}-column generation ends at "
            f"ep{last_ep}, but the manifest pins ep{pinned_ep}. Either the segment logic "
            "or the manifest is stale; refusing to regress against it."
        )
    if len(block) < 2:
        raise ReferenceError(
            f"{log_path}: trailing generation has {len(block)} row(s); need at least 2 "
            "for the {ep-500, ep} reference window"
        )

    ref_rows = block[-2:]
    tail3_rows = block[-3:]
    return {
        "width": block[0]["_width"],
        "n_generation_rows": len(block),
        "generation_episodes": [int(_to_float(r["episode"])) for r in block],
        "ref_episodes": [int(_to_float(r["episode"])) for r in ref_rows],
        "ref_rows": ref_rows,
        "tail3_rows": tail3_rows,
    }


def _std(vals: Sequence[float]) -> float:
    """Population standard deviation; 0.0 for fewer than two samples."""
    n = len(vals)
    if n < 2:
        return 0.0
    mean = sum(vals) / n
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / n)


def _mean(vals: Sequence[float]) -> float:
    finite = [v for v in vals if not math.isnan(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def metric_floor(metric: str) -> float:
    if metric in COUNT_METRICS or metric.startswith("n_"):
        return COUNT_FLOOR
    return RATE_FLOOR


def metric_tolerance(metric: str, tail3_rows: Sequence[Dict[str, Any]]) -> float:
    """tol_m = max(floor_m, 2*std of metric m over the last 3 reference-generation rows)."""
    vals = [_to_float(r.get(metric, "")) for r in tail3_rows]
    vals = [v for v in vals if not math.isnan(v)]
    return max(metric_floor(metric), 2.0 * _std(vals))


# ---------------------------------------------------------------------------
# Sweep-side reading
# ---------------------------------------------------------------------------

def sweep_row(results_dir: str, exp_name: str) -> Optional[Dict[str, Any]]:
    """The single eval row a TB sweep run writes. TB outputs are uniformly 76 columns."""
    path = os.path.join(results_dir, exp_name, "darts_log.csv")
    if not os.path.isfile(path):
        return None
    rows = read_rows_schema_exact(path)
    if not rows:
        return None
    if rows[-1]["_width"] != 76:
        raise ReferenceError(
            f"{path}: TB sweep output should be 76 columns, saw {rows[-1]['_width']}"
        )
    return rows[-1]


def _ff_denom(row: Dict[str, Any]) -> float:
    ff = _to_float(row.get("n_full_fidelity", ""))
    if math.isnan(ff):
        ff = 0.0
    return max(1.0, ff)


def certified_miss_split(row: Dict[str, Any]) -> Tuple[float, float, float]:
    """(certified_miss, certified_miss_late, certified_miss_unreached).

    late       = n_miss_full_fidelity_late / max(1, n_full_fidelity)
                 temporal-certificate violations; the cliff metric
    unreached  = (n_ff_missed - n_miss_full_fidelity_late) / max(1, n_full_fidelity)
                 never-completed; energy-layer residual
    certified_miss = their sum (equals n_ff_missed / max(1, n_full_fidelity) when both
                 counts are present).

    A missing count on the *reference* side is returned as nan (rendered n/a). TB-side
    76-column rows always carry both counts.
    """
    denom = _ff_denom(row)
    has_late = "n_miss_full_fidelity_late" in row
    has_missed = "n_ff_missed" in row
    late_count = _to_float(row["n_miss_full_fidelity_late"]) if has_late else float("nan")
    missed_count = _to_float(row["n_ff_missed"]) if has_missed else float("nan")
    late = late_count / denom if has_late and not math.isnan(late_count) else float("nan")
    if has_missed and has_late and not math.isnan(missed_count) and not math.isnan(late_count):
        unreached = (missed_count - late_count) / denom
        certified = missed_count / denom
    elif has_missed and not math.isnan(missed_count):
        unreached = float("nan")
        certified = missed_count / denom
    else:
        unreached = float("nan")
        certified = late if not math.isnan(late) else float("nan")
    return certified, late, unreached


def derived_metrics(row: Dict[str, Any]) -> Dict[str, float]:
    """Saturation metrics for the endpoint report / plot / per-cell CSV.

    bid_accept     n_bids_cert_pass / n_bids_evaluated  (certificate admission rate)
    U_team         (Σ exec + Σ travel) / (n_agents * episode_span_ms)  (team utilisation)
    certified_miss n_ff_missed / max(1, n_full_fidelity), split into late + unreached
    """
    evaluated = _to_float(row.get("n_bids_evaluated", ""))
    passed = _to_float(row.get("n_bids_cert_pass", ""))
    bid_accept = passed / evaluated if evaluated > 0 else float("nan")

    span = _to_float(row.get("episode_span_ms", ""))
    busy = 0.0
    n_busy = 0
    for i in range(1, 5):
        for prefix in ("exec_ms_A", "travel_ms_A"):
            v = _to_float(row.get(f"{prefix}{i}", ""))
            if not math.isnan(v):
                busy += v
                n_busy += 1
    u_team = busy / (4.0 * span) if span > 0 and n_busy else float("nan")

    certified, late, unreached = certified_miss_split(row)
    return {
        "bid_accept": bid_accept,
        "rho_hat": _to_float(row.get("rho_hat", "")),
        "U_team": u_team,
        "task_miss_rate": _to_float(row.get("task_miss_rate", "")),
        "certified_miss": certified,
        "certified_miss_late": late,
        "certified_miss_unreached": unreached,
    }


def _fmt_metric(v: float) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"


def _mean_ref_split(ref_rows: Sequence[Dict[str, Any]]) -> Tuple[float, float, float]:
    """Mean certified-miss split over the reference window. nan -> n/a for 59-col."""
    triples = [certified_miss_split(r) for r in ref_rows]
    return (
        _mean([t[0] for t in triples]),
        _mean([t[1] for t in triples]),
        _mean([t[2] for t in triples]),
    )


ENDPOINT_METRICS = (
    "bid_accept", "rho_hat", "U_team", "task_miss_rate",
    "certified_miss", "certified_miss_late", "certified_miss_unreached",
)
ENDPOINT_COL_W = 22


# ---------------------------------------------------------------------------
# Manifest helpers (shared by both modes)
# ---------------------------------------------------------------------------

def load_manifest(path: str) -> Dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def manifest_cells(manifest: Dict[str, Any], cells: Sequence[Tuple[str, str]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for bench, stress in cells:
        prefix = f"{bench}|{stress}|"
        seeds = sorted(
            int(k[len(prefix):]) for k in manifest
            if k != "_meta" and k.startswith(prefix) and k[len(prefix):].isdigit()
        )
        if not seeds:
            raise ReferenceError(f"manifest has no entries for {bench}:{stress}")
        for seed in seeds:
            entry = manifest[f"{bench}|{stress}|{seed}"]
            out.append({
                "bench": bench, "stress": stress, "seed": seed,
                "dir": entry["dir"], "ep": int(entry["ep"]),
            })
    return out


def parse_cells(raw: str) -> List[Tuple[str, str]]:
    cells = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if chunk:
            bench, stress = chunk.split(":", 1)
            cells.append((bench.strip(), stress.strip()))
    return cells


def is_cells_filter(cells: Sequence[Tuple[str, str]]) -> bool:
    """True when --cells is a proper subset/superset of the default three-cell grid."""
    return set(cells) != set(parse_cells(DEFAULT_CELLS))


def kappa_tag(kappa: float) -> str:
    return "k" + str(float(kappa)).replace(".", "p")


def beta_tag(beta: float) -> str:
    return "b" + str(float(beta)).replace(".", "p")


def is_unity_beta(beta: float) -> bool:
    return float(beta) == 1.0


def exp_name_for(
    bench: str, stress: str, kappa: float, seed: int, beta: float = 1.0
) -> str:
    name = f"tb1_{bench}_{stress}_{kappa_tag(kappa)}_seed{seed}"
    if not is_unity_beta(beta):
        name += f"_{beta_tag(beta)}"
    return name


_RUN_REST = re.compile(
    r"^k(?P<k>\d+p\d+)_seed(?P<seed>\d+)(?:_b(?P<b>\d+p\d+))?$"
)


def parse_run_dirname(
    name: str, bench: str, stress: str
) -> Optional[Tuple[float, int, float]]:
    """Parse (kappa, seed, beta) from a sweep dir name. Absent _b token -> beta=1.0."""
    prefix = f"tb1_{bench}_{stress}_"
    if not name.startswith(prefix):
        return None
    m = _RUN_REST.fullmatch(name[len(prefix):])
    if not m:
        return None
    kappa = float(m.group("k").replace("p", "."))
    seed = int(m.group("seed"))
    beta = float(m.group("b").replace("p", ".")) if m.group("b") else 1.0
    return kappa, seed, beta


def iter_cell_runs(
    results_dir: str, bench: str, stress: str
) -> List[Tuple[str, float, int, float]]:
    """(exp_name, kappa, seed, beta) for dirs matching this cell."""
    out: List[Tuple[str, float, int, float]] = []
    if not os.path.isdir(results_dir):
        return out
    for name in os.listdir(results_dir):
        parsed = parse_run_dirname(name, bench, stress)
        if parsed is None:
            continue
        kappa, seed, beta = parsed
        out.append((name, kappa, seed, beta))
    return out


def count_nonzero_beta_runs(
    results_dir: str, cells: Sequence[Tuple[str, str]]
) -> int:
    n = 0
    for bench, stress in cells:
        n += sum(1 for _n, _k, _s, b in iter_cell_runs(results_dir, bench, stress)
                 if not is_unity_beta(b))
    return n


def discover_betas(results_dir: str, bench: str, stress: str) -> List[float]:
    found = {b for _n, _k, _s, b in iter_cell_runs(results_dir, bench, stress)}
    return sorted(found)


def plot_pdf_path(results_dir: str, bench: str, stress: str, beta: float = 1.0) -> str:
    stem = f"tb1_F1_draft_{bench}_{stress}"
    if not is_unity_beta(beta):
        stem += f"_{beta_tag(beta)}"
    return os.path.join(results_dir, stem + ".pdf")


# ---------------------------------------------------------------------------
# Mode: --check-kappa1
# ---------------------------------------------------------------------------

def check_kappa1(results_dir: str, manifest: Dict[str, Any], cells: Sequence[Tuple[str, str]]) -> int:
    print("=" * 100)
    print("TB-1 κ=1.0 REGRESSION GATE  (sweep κ=1.0 vs paper-era reference window)")
    print("=" * 100)
    print(f"Reference window : trailing run generation, last two rows {{ep-500, ep}}")
    print("Tolerance        : per-metric max(floor_m, 2*std_m over last 3 ref-generation rows);")
    print(f"                   floor_m={RATE_FLOOR} for rates, {COUNT_FLOOR} for counts.")
    print("                   n_pending is complementary to n_full_fidelity; display-only.")
    print(f"Excluded         : {', '.join(TIMING_COLS)}")
    print("                   (wall-clock perf_counter; nondeterministic across processes)")
    n_beta_skip = count_nonzero_beta_runs(results_dir, cells)
    print(f"note: skipped {n_beta_skip} beta!=1.0 run(s) "
          "(changed environment; paper-era reference comparison is undefined)")
    print()

    n_fail = 0
    n_cells = 0
    for cell in manifest_cells(manifest, cells):
        label = f"{cell['bench']}|{cell['stress']}|{cell['seed']}"
        log = os.path.join(cell["dir"], "darts_log.csv")
        try:
            win = reference_window(log, cell["ep"])
        except ReferenceError as exc:
            print(f"[{label}] REFERENCE FAIL: {exc}")
            n_fail += 1
            continue

        exp = exp_name_for(cell["bench"], cell["stress"], 1.0, cell["seed"])
        try:
            got = sweep_row(results_dir, exp)
        except ReferenceError as exc:
            print(f"[{label}] SWEEP FAIL: {exc}")
            n_fail += 1
            continue
        if got is None:
            print(f"[{label}] MISSING sweep run {exp} under {results_dir}")
            n_fail += 1
            continue

        n_cells += 1
        ref_width = win["width"]
        comparable = [
            m for m in GATED_METRICS
            if m in WIDTH_HEADERS[ref_width] and m not in TIMING_COLS
        ]
        skipped = [m for m in GATED_METRICS if m not in comparable]

        print(f"[{label}] ref: {ref_width}-col generation, {win['n_generation_rows']} rows, "
              f"window eps {win['ref_episodes']}")
        if skipped:
            print(f"           metrics absent from the {ref_width}-column reference schema, "
                  f"not compared: {', '.join(skipped)}")

        cell_fail = 0
        n_compared = 0
        for metric in comparable:
            ref_vals = [_to_float(r[metric]) for r in win["ref_rows"]]
            ref = _mean(ref_vals)
            new = _to_float(got.get(metric, ""))
            if math.isnan(ref) and math.isnan(new):
                continue
            n_compared += 1
            delta = abs(new - ref)
            tol_m = metric_tolerance(metric, win["tail3_rows"])
            ok = not math.isnan(delta) and delta <= tol_m
            if not ok:
                cell_fail += 1
                print(f"           FAIL {metric:<28} ref={ref:<12.4f} new={new:<12.4f} "
                      f"|Δ|={delta:.4f} > tol_m={tol_m:.4f}")
        if cell_fail:
            n_fail += 1
        else:
            print(f"           ok  ({n_compared} gated metrics within per-metric tolerance)")

        if "n_pending" in WIDTH_HEADERS[ref_width]:
            ref_pend = _mean([_to_float(r["n_pending"]) for r in win["ref_rows"]])
            new_pend = _to_float(got.get("n_pending", ""))
            delta_pend = abs(new_pend - ref_pend) if not (math.isnan(ref_pend) or math.isnan(new_pend)) else float("nan")
            print(
                f"           n_pending (diagnostic, not gated) "
                f"ref={_fmt_metric(ref_pend):<12} new={_fmt_metric(new_pend)} "
                f"|Δ|={_fmt_metric(delta_pend)}"
            )

        tb_split = derived_metrics(got)
        ref_certified, ref_late, ref_unreached = _mean_ref_split(win["ref_rows"])
        print(
            f"           certified_miss_late      ref={_fmt_metric(ref_late):<12} "
            f"new={_fmt_metric(tb_split['certified_miss_late'])}"
        )
        print(
            f"           certified_miss_unreached ref={_fmt_metric(ref_unreached):<12} "
            f"new={_fmt_metric(tb_split['certified_miss_unreached'])}"
        )

        diag = {m: _to_float(got.get(m, "")) for m in TIMING_COLS if m in got}
        if diag:
            print("           timing diagnostics (not gated): "
                  + ", ".join(f"{k}={v:.3f}" for k, v in diag.items()))
        print()

    print("-" * 100)
    print(f"κ=1.0 gate: {n_cells - n_fail}/{n_cells} cells within tolerance"
          f"{f', {n_fail} FAILING' if n_fail else ''}")
    return 1 if n_fail else 0


# ---------------------------------------------------------------------------
# Mode: --endpoint-report
# ---------------------------------------------------------------------------

def endpoint_report(
    results_dir: str,
    manifest: Dict[str, Any],
    cells: Sequence[Tuple[str, str]],
    low: float,
    high: float,
) -> int:
    print("=" * 100)
    print("TB-1 ENDPOINT SATURATION REPORT")
    print("=" * 100)
    print(f"Each cell: κ={low} and κ={high} against κ=1.0, per metric.")
    print("noise = 2*std of that metric over the cell's κ=1.0 seeds; 'flat' means |Δ| <= noise.")
    print("Extension rule (per cell): right endpoint flat on every metric -> run κ=3.0 for that cell;")
    print("                bid_accept>=0.999 at κ=1.0 -> left side degenerate, no κ=0.25;")
    print("                else certified_miss_late zero below κ=1 -> run κ=0.25 for that cell.")
    print("Grouped by (bench, stress, beta); each group is judged independently.")
    print("A group missing --low/--high is skipped; the report fails only if none can be judged.")
    print()

    n_judged = 0
    n_skipped = 0

    for bench, stress in cells:
        prefix = f"{bench}|{stress}|"
        seeds = sorted(
            int(k[len(prefix):]) for k in manifest
            if k != "_meta" and k.startswith(prefix) and k[len(prefix):].isdigit()
        )
        betas = discover_betas(results_dir, bench, stress) or [1.0]

        for beta in betas:
            group_missing: List[str] = []

            def gather(kappa: float, _beta: float = beta) -> Dict[str, List[float]]:
                acc: Dict[str, List[float]] = {m: [] for m in ENDPOINT_METRICS}
                for seed in seeds:
                    exp = exp_name_for(bench, stress, kappa, seed, _beta)
                    row = sweep_row(results_dir, exp)
                    if row is None:
                        group_missing.append(exp)
                        continue
                    for m, v in derived_metrics(row).items():
                        acc[m].append(v)
                return acc

            base = gather(1.0)
            acc_by_kappa = {low: gather(low), high: gather(high)}
            beta_label = f"β={beta}" if not is_unity_beta(beta) else "β=1.0"
            print(f"--- {bench}:{stress}  {beta_label}  (seeds {seeds}) ---")
            if not any(base[m] for m in ENDPOINT_METRICS):
                print(f"    verdict skipped (missing: {', '.join(group_missing)})\n")
                n_skipped += 1
                continue

            noise = {m: 2.0 * _std([v for v in base[m] if not math.isnan(v)]) for m in ENDPOINT_METRICS}
            base_mean = {m: _mean(base[m]) for m in ENDPOINT_METRICS}

            print(f"    {'':<10} " + " ".join(f"{m:>{ENDPOINT_COL_W}}" for m in ENDPOINT_METRICS))
            print(f"    {'k=1.0':<10} " + " ".join(f"{_fmt_metric(base_mean[m]):>{ENDPOINT_COL_W}}" for m in ENDPOINT_METRICS))
            print(f"    {'noise(2s)':<10} " + " ".join(f"{_fmt_metric(noise[m]):>{ENDPOINT_COL_W}}" for m in ENDPOINT_METRICS))

            cell_complete = True
            cell_right_flat = None
            cell_left_late = None
            n_expected = len(seeds)

            for kappa in (low, high):
                acc = acc_by_kappa[kappa]
                n_have = max((len(acc[m]) for m in ENDPOINT_METRICS), default=0)
                if n_have == 0:
                    print(f"    {('k=' + str(kappa)):<10} " + " ".join(f"{'--':>{ENDPOINT_COL_W}}" for _ in ENDPOINT_METRICS))
                    cell_complete = False
                    continue
                means = {m: _mean(acc[m]) for m in ENDPOINT_METRICS}
                deltas = {m: means[m] - base_mean[m] for m in ENDPOINT_METRICS}
                flags = {
                    m: ("flat" if (math.isnan(deltas[m]) or abs(deltas[m]) <= noise[m]) else "MOVED")
                    for m in ENDPOINT_METRICS
                }
                print(f"    {('k=' + str(kappa)):<10} " + " ".join(f"{_fmt_metric(means[m]):>{ENDPOINT_COL_W}}" for m in ENDPOINT_METRICS))
                print(f"    {'  delta':<10} " + " ".join(
                    f"{('n/a' if math.isnan(deltas[m]) else f'{deltas[m]:+.4f}'):>{ENDPOINT_COL_W}}"
                    for m in ENDPOINT_METRICS
                ))
                print(f"    {'  vs noise':<10} " + " ".join(f"{flags[m]:>{ENDPOINT_COL_W}}" for m in ENDPOINT_METRICS))
                if n_have < n_expected:
                    cell_complete = False
                    continue
                if kappa == high:
                    cell_right_flat = all(f != "MOVED" for f in flags.values())
                if kappa == low:
                    cell_left_late = any(
                        v > 0 for v in acc["certified_miss_late"] if not math.isnan(v)
                    )

            if not cell_complete or cell_right_flat is None or cell_left_late is None:
                print(f"    verdict skipped (missing: {', '.join(group_missing)})")
                n_skipped += 1
            else:
                n_judged += 1
                print("    cell verdict:")
                if cell_right_flat:
                    print(f"      right (κ={high}): flat on all metrics for this cell -> run κ=3.0")
                else:
                    print(f"      right (κ={high}): moved on at least one metric -> no right extension needed")
                ba = base_mean.get("bid_accept", float("nan"))
                if not math.isnan(ba) and ba >= ADMISSION_SATURATED:
                    print("      left: left side degenerate on-policy "
                          "(identical trajectories for kappa<1); left extension not applicable")
                elif cell_left_late:
                    print(f"      left (κ={low}): certified_miss_late observed -> no left extension needed")
                else:
                    print(f"      left (κ={low}): certified_miss_late zero below κ=1 -> run κ=0.25")
            print()

    print("-" * 100)
    print("EXTENSION VERDICT")
    if n_judged == 0:
        print("INSUFFICIENT DATA: no group could be judged")
        return 1
    print(f"  {n_judged} group(s) judged, {n_skipped} skipped (missing endpoints).")
    print("  per-cell verdicts above; do not OR one cell's movement into another.")
    print("  (extend with: scripts/sweep_tb1.py --kappas 3.0   /   --kappas 0.25)")
    return 0


# ---------------------------------------------------------------------------
# Sweep inventory, per-cell CSV, --plot
# ---------------------------------------------------------------------------

def discover_kappas(
    results_dir: str, bench: str, stress: str, beta: float = 1.0
) -> List[float]:
    """κ values present for this (bench, stress, beta)."""
    found = {
        k for _n, k, _s, b in iter_cell_runs(results_dir, bench, stress)
        if b == float(beta)
    }
    return sorted(found)


def collect_cell_series(
    results_dir: str,
    bench: str,
    stress: str,
    seeds: Sequence[int],
    kappas: Sequence[float],
    beta: float = 1.0,
) -> Dict[str, Dict[float, List[float]]]:
    series: Dict[str, Dict[float, List[float]]] = {
        m: {k: [] for k in kappas} for m in ENDPOINT_METRICS
    }
    for kappa in kappas:
        for seed in seeds:
            row = sweep_row(results_dir, exp_name_for(bench, stress, kappa, seed, beta))
            if row is None:
                continue
            for m, v in derived_metrics(row).items():
                series[m][kappa].append(v)
    return series


def write_per_cell_csv(
    results_dir: str,
    manifest: Dict[str, Any],
    cells: Sequence[Tuple[str, str]],
    filename: str = "tb1_per_cell.csv",
) -> str:
    """One row per (bench, stress, kappa, seed, beta) with the saturation metrics + split."""
    path = os.path.join(results_dir, filename)
    fieldnames = ["bench", "stress", "kappa", "seed", "beta"] + list(ENDPOINT_METRICS)
    n_rows = 0
    os.makedirs(results_dir, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for bench, stress in cells:
            prefix = f"{bench}|{stress}|"
            seeds = sorted(
                int(k[len(prefix):]) for k in manifest
                if k != "_meta" and k.startswith(prefix) and k[len(prefix):].isdigit()
            )
            betas = discover_betas(results_dir, bench, stress) or [1.0]
            for beta in betas:
                for kappa in discover_kappas(results_dir, bench, stress, beta):
                    for seed in seeds:
                        row = sweep_row(
                            results_dir, exp_name_for(bench, stress, kappa, seed, beta)
                        )
                        if row is None:
                            continue
                        metrics = derived_metrics(row)
                        out = {
                            "bench": bench, "stress": stress, "kappa": kappa,
                            "seed": seed, "beta": beta,
                        }
                        for m in ENDPOINT_METRICS:
                            v = metrics[m]
                            out[m] = "" if math.isnan(v) else f"{v:.6g}"
                        writer.writerow(out)
                        n_rows += 1
    print(f"[tb1] wrote {n_rows} rows -> {path}")
    return path


def plot_cells(
    results_dir: str,
    manifest: Dict[str, Any],
    cells: Sequence[Tuple[str, str]],
) -> List[str]:
    """Draft Fig.1: log-x κ, left ρ̂/U_team, right miss rates, vline at κ=1, late-miss shade.

    One vector PDF per (bench, stress, beta). β=1.0 keeps the stem
    tb1_F1_draft_<bench>_<stress>.pdf; other β values append _b<beta>.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    written: List[str] = []
    for bench, stress in cells:
        prefix = f"{bench}|{stress}|"
        seeds = sorted(
            int(k[len(prefix):]) for k in manifest
            if k != "_meta" and k.startswith(prefix) and k[len(prefix):].isdigit()
        )
        betas = discover_betas(results_dir, bench, stress)
        if not betas:
            print(f"[plot] no sweep dirs for {bench}:{stress} under {results_dir}")
            continue
        for beta in betas:
            kappas = discover_kappas(results_dir, bench, stress, beta)
            if not kappas:
                continue
            series = collect_cell_series(
                results_dir, bench, stress, seeds, kappas, beta
            )

            fig, ax_l = plt.subplots()
            ax_r = ax_l.twinx()
            ax_l.set_xscale("log")

            left_specs = (("rho_hat", "C0", "-"), ("U_team", "C1", "--"))
            right_specs = (("task_miss_rate", "C2", "-"), ("certified_miss", "C3", "--"))

            def _band(ax, metric, color, ls, label, _kappas=kappas, _series=series):
                xs = [k for k in _kappas if _series[metric][k]]
                if not xs:
                    return
                means = [_mean(_series[metric][k]) for k in xs]
                lo = [min(_series[metric][k]) for k in xs]
                hi = [max(_series[metric][k]) for k in xs]
                ax.plot(xs, means, color=color, linestyle=ls, marker="o", label=label)
                ax.fill_between(xs, lo, hi, color=color, alpha=0.18, linewidth=0)

            for metric, color, ls in left_specs:
                _band(ax_l, metric, color, ls, metric)
            for metric, color, ls in right_specs:
                _band(ax_r, metric, color, ls, metric)

            ax_l.axvline(1.0, color="k", linestyle=":", linewidth=1.2, label="κ=1.0")

            late_positive = [
                k for k in kappas
                if k < 1.0 and series["certified_miss_late"][k]
                and _mean(series["certified_miss_late"][k]) > 0
            ]
            if late_positive:
                ax_l.axvspan(min(late_positive), 1.0, color="C3", alpha=0.12, zorder=0)

            ax_l.set_xticks(kappas)
            ax_l.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
            ax_l.xaxis.set_minor_locator(mticker.NullLocator())
            ax_l.set_xlabel("κ (log)")
            ax_l.set_ylabel("rho_hat, U_team")
            ax_r.set_ylabel("task_miss_rate, certified_miss")
            title = f"TB-1 draft  {bench} {stress}"
            if not is_unity_beta(beta):
                title += f"  β={beta}"
            ax_l.set_title(title)
            h1, l1 = ax_l.get_legend_handles_labels()
            h2, l2 = ax_r.get_legend_handles_labels()
            ax_l.legend(h1 + h2, l1 + l2, loc="best", fontsize="small")
            fig.tight_layout()
            out = plot_pdf_path(results_dir, bench, stress, beta)
            fig.savefig(out)
            plt.close(fig)
            print(f"[plot] {out}  kappas={kappas}")
            written.append(out)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results-dir", required=True, help="Sweep output root used by sweep_tb1.py.")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--cells", default=DEFAULT_CELLS)
    ap.add_argument("--check-kappa1", action="store_true",
                    help="Regress sweep κ=1.0 against the paper-era reference window.")
    ap.add_argument("--endpoint-report", action="store_true",
                    help="Per-cell κ endpoint saturation table and D1 extension verdict.")
    ap.add_argument("--plot", action="store_true",
                    help="Write tb1_F1_draft_<bench>_<stress>.pdf per (cell, beta) under --results-dir.")
    ap.add_argument("--low", type=float, default=0.5, help="Left endpoint κ (default 0.5).")
    ap.add_argument("--high", type=float, default=2.0, help="Right endpoint κ (default 2.0).")
    ap.add_argument("--show-reference", action="store_true",
                    help="Print the resolved reference window per cell and exit.")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    cells = parse_cells(args.cells)

    if args.show_reference:
        for cell in manifest_cells(manifest, cells):
            log = os.path.join(cell["dir"], "darts_log.csv")
            label = f"{cell['bench']}|{cell['stress']}|{cell['seed']}"
            try:
                win = reference_window(log, cell["ep"])
            except ReferenceError as exc:
                print(f"[{label}] FAIL: {exc}")
                continue
            tols = {
                m: metric_tolerance(m, win["tail3_rows"])
                for m in GATED_METRICS
                if m in WIDTH_HEADERS[win["width"]]
            }
            print(f"[{label}] manifest ep={cell['ep']} width={win['width']} "
                  f"generation eps={win['generation_episodes']} "
                  f"ref={win['ref_episodes']}")
            print("           tol_m: " + ", ".join(f"{m}={v:.4f}" for m, v in tols.items()))
        return 0

    rc = 0
    if args.check_kappa1:
        rc |= check_kappa1(args.results_dir, manifest, cells)
    if args.plot or args.endpoint_report or args.check_kappa1:
        filtered = is_cells_filter(cells)
        csv_name = "tb1_per_cell_filtered.csv" if filtered else "tb1_per_cell.csv"
        write_per_cell_csv(args.results_dir, manifest, cells, filename=csv_name)
    if args.plot:
        plot_cells(args.results_dir, manifest, cells)
    if args.endpoint_report:
        rc |= endpoint_report(args.results_dir, manifest, cells, args.low, args.high)
    if not (args.check_kappa1 or args.endpoint_report or args.show_reference or args.plot):
        ap.error("pick at least one of --check-kappa1 / --endpoint-report / --plot / --show-reference")
    return rc


if __name__ == "__main__":
    sys.exit(main())
