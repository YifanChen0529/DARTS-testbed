#!/usr/bin/env python3
"""EXP-3 full-session gate adjudication and per-config tables.

Reads ``cbba_log.csv`` (CBBA_LOG_HEADER). Null rows feed only the overhead
floor; warmup rows are excluded from every statistic; recorded rows are the
200 per configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.csv_schema_tb import CBBA_LOG_HEADER  # noqa: E402

DEFAULT_ROOT = os.path.join(_PROJECT_ROOT, "results_exp3_20260829")
DEFAULT_ANALYSIS_MD = os.path.join(
    _PROJECT_ROOT, "docs", "exp3", "exp3_analysis_output.md"
)
DEFAULT_GATES = os.path.join(_PROJECT_ROOT, "docs", "exp3", "exp3_gates.md")
DEFAULT_SUMMARY_MD = os.path.join(
    _PROJECT_ROOT, "docs", "exp3", "exp3_summary_table.md"
)
DEFAULT_SUMMARY_CSV = os.path.join(
    _PROJECT_ROOT, "docs", "exp3", "exp3_summary_table.csv"
)

EXECUTION_NOTE = (
    "All wall-time columns are measured on the loopback tier: all four agent "
    "processes executed on a single Jetson-class node over local loopback; "
    "per-round compute, serialization, and synchronization are measured; the "
    "physical wireless channel was not traversed, so any cross-machine network "
    "term is calibrated rather than measured."
)

REF_MEAN_MS = 9.6
REF_P99_MS = 23.38
REF_ENVELOPE_MS = 30.0

RESULTS_MARKER = "\n## RESULTS"


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile; ``p`` is in [0, 100]."""
    ys = sorted(float(v) for v in values)
    n = len(ys)
    if n == 0:
        raise ValueError("percentile of empty sample")
    if n == 1:
        return ys[0]
    if p <= 0.0:
        return ys[0]
    if p >= 100.0:
        return ys[-1]
    k = (p / 100.0) * (n - 1)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return ys[lo]
    w = k - lo
    return ys[lo] * (1.0 - w) + ys[hi] * w


def median(values: Sequence[float]) -> float:
    return percentile(values, 50.0)


def dispersion_ratio(values: Sequence[float]) -> float:
    """P99 / median. Empty or zero-median samples raise or return inf."""
    med = median(values)
    p99 = percentile(values, 99.0)
    if med == 0.0:
        return math.inf if p99 != 0.0 else float("nan")
    return p99 / med


def half_split_medians(
    values_in_order: Sequence[float],
) -> Tuple[float, float, float]:
    """Return (first-half median, second-half median, signed percent delta).

    Percent delta is ``100 * (second - first) / first``. First-half is the
    leading ``n//2`` values in trial order; the remainder is the second half.
    """
    ys = [float(v) for v in values_in_order]
    n = len(ys)
    if n < 2:
        raise ValueError("half_split_medians needs at least 2 values")
    mid = n // 2
    first = median(ys[:mid])
    second = median(ys[mid:])
    if first == 0.0:
        pct = math.inf if second != 0.0 else float("nan")
    else:
        pct = 100.0 * (second - first) / first
    return first, second, pct


def mean(values: Sequence[float]) -> float:
    ys = [float(v) for v in values]
    if not ys:
        raise ValueError("mean of empty sample")
    return sum(ys) / len(ys)


def partition_rows(
    rows: Sequence[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    nulls: List[Dict[str, str]] = []
    warmups: List[Dict[str, str]] = []
    recorded: List[Dict[str, str]] = []
    for row in rows:
        if int(row["null"]) == 1:
            nulls.append(row)
        elif int(row["warmup"]) == 1:
            warmups.append(row)
        else:
            recorded.append(row)
    return nulls, warmups, recorded


def load_log(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != list(CBBA_LOG_HEADER):
            raise SystemExit(
                f"header mismatch in {path}: {reader.fieldnames!r} != {list(CBBA_LOG_HEADER)!r}"
            )
        return list(reader)


def _sum_json_floats(blob: str) -> float:
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        return float("nan")
    if isinstance(parsed, list):
        return float(sum(float(x) for x in parsed))
    return float("nan")


def _fmt(x: float, nd: int = 4) -> str:
    if x != x:  # NaN
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:.{nd}f}"


def _config_sort_key(row: Dict[str, str]) -> Tuple[int, str, int]:
    cid = row["config_id"]
    tok = "ceil" if cid.endswith("ltceil") else "1"
    return (int(row["m"]), tok, int(row["l_t"]))


def _bundle_words(cid: str, l_t: int) -> str:
    if cid.endswith("ltceil"):
        return "ceiling of m over four"
    return "one"


def _group_recorded(
    recorded: Sequence[Dict[str, str]],
) -> List[Tuple[str, List[Dict[str, str]]]]:
    buckets: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in recorded:
        buckets[row["config_id"]].append(row)
    items = list(buckets.items())
    items.sort(key=lambda kv: _config_sort_key(kv[1][0]))
    return items


def adjudicate(
    nulls: Sequence[Dict[str, str]],
    recorded: Sequence[Dict[str, str]],
) -> Dict[str, Any]:
    floor_from_nulls = median([float(r["t_total_ms"]) for r in nulls]) if nulls else float("nan")
    col_vals = [float(r["t_overhead_floor_ms"]) for r in list(nulls) + list(recorded)]
    col_unique = sorted(set(f"{v:.9f}" for v in col_vals))
    col_floor = float(col_vals[0]) if col_vals else float("nan")
    floor_discrepancy = None
    if col_vals:
        if abs(col_floor - floor_from_nulls) > 1e-4:
            floor_discrepancy = (floor_from_nulls, col_floor, col_unique)

    gv_fail = [r for r in recorded if int(r["valid"]) != 1]
    gc_fail = [
        r
        for r in recorded
        if int(r["rounds"]) > int(r["l_t"]) * int(r["m"])
    ]
    gr_fail = [r for r in recorded if int(r["ref_agree"]) != 1]

    gs_rows = []
    for cid, group in _group_recorded(recorded):
        ordered = sorted(group, key=lambda r: int(r["trial"]))
        times = [float(r["t_total_ms"]) for r in ordered]
        h1, h2, pct = half_split_medians(times)
        gs_rows.append(
            {
                "config_id": cid,
                "m": int(group[0]["m"]),
                "l_t": int(group[0]["l_t"]),
                "n": len(times),
                "first": h1,
                "second": h2,
                "pct": pct,
                "flag": abs(pct) > 10.0,
            }
        )

    compute_floors = []
    for cid, group in _group_recorded(recorded):
        comps = [_sum_json_floats(r["per_node_compute_ms_json"]) for r in group]
        tmin = min(float(r["t_total_ms"]) for r in group)
        compute_floors.append(
            {
                "config_id": cid,
                "m": int(group[0]["m"]),
                "l_t": int(group[0]["l_t"]),
                "median_node_compute_sum_ms": median(comps),
                "min_t_total_ms": tmin,
            }
        )

    return {
        "floor_from_nulls": floor_from_nulls,
        "n_null": len(nulls),
        "col_floor": col_floor,
        "floor_discrepancy": floor_discrepancy,
        "n_recorded": len(recorded),
        "gv_fail": gv_fail,
        "gc_fail": gc_fail,
        "gr_fail": gr_fail,
        "gv_pass": len(gv_fail) == 0,
        "gc_pass": len(gc_fail) == 0,
        "gr_pass": len(gr_fail) == 0,
        "gs_rows": gs_rows,
        "compute_floors": compute_floors,
    }


def _row_verbatim(row: Dict[str, str]) -> str:
    return ", ".join(f"{k}={row[k]}" for k in CBBA_LOG_HEADER)


def config_stats(group: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    m = int(group[0]["m"])
    l_t = int(group[0]["l_t"])
    cid = group[0]["config_id"]
    rounds = [float(r["rounds"]) for r in group]
    times = [float(r["t_total_ms"]) for r in group]
    msgs = [float(r["msgs_total"]) for r in group]
    byts = [float(r["bytes_total"]) for r in group]
    cap = l_t * m
    r_max = max(rounds)
    return {
        "config_id": cid,
        "m": m,
        "l_t": l_t,
        "n": len(group),
        "cap": cap,
        "rounds_mean": mean(rounds),
        "rounds_p50": percentile(rounds, 50),
        "rounds_p95": percentile(rounds, 95),
        "rounds_p99": percentile(rounds, 99),
        "rounds_max": r_max,
        "rounds_max_to_cap": r_max / cap if cap else float("nan"),
        "rounds_disp": dispersion_ratio(rounds),
        "t_mean": mean(times),
        "t_p50": percentile(times, 50),
        "t_p95": percentile(times, 95),
        "t_p99": percentile(times, 99),
        "t_max": max(times),
        "t_disp": dispersion_ratio(times),
        "msgs_mean": mean(msgs),
        "bytes_mean": mean(byts),
        "ref_96": m * REF_MEAN_MS,
        "ref_2338": m * REF_P99_MS,
        "ref_30": m * REF_ENVELOPE_MS,
        "bundle_words": _bundle_words(cid, l_t),
    }


def render_gate_section(adj: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("## GATE ADJUDICATION")
    lines.append("")
    lines.append(
        "Registrations quoted from `docs/exp3/exp3_gates.md`: G-V strict "
        "(`valid == 1` on 100% of recorded rows); G-C strict "
        "(`rounds <= l_t * m` on every recorded row); G-R strict "
        "(`ref_agree == 1` on every recorded row); G-S informational "
        "(first-half vs second-half median `t_total_ms` in trial order, "
        "flag beyond about 10%); G-F report-only (overhead floor; per-config "
        "loopback compute floor context)."
    )
    lines.append("")
    n = adj["n_recorded"]
    n_fail = len(adj["gv_fail"])
    gv = "PASS" if adj["gv_pass"] else "FAIL"
    lines.append(
        f"### G-V (strict) {gv}: valid==1 on {n - n_fail}/{n} recorded rows."
    )
    if adj["gv_fail"]:
        lines.append("Failing rows (verbatim):")
        for row in adj["gv_fail"]:
            lines.append(f"  {_row_verbatim(row)}")
    lines.append("")
    n_fail = len(adj["gc_fail"])
    gc = "PASS" if adj["gc_pass"] else "FAIL"
    lines.append(
        f"### G-C (strict) {gc}: rounds <= l_t*m on {n - n_fail}/{n} recorded rows."
    )
    if adj["gc_fail"]:
        lines.append("Failing rows (verbatim):")
        for row in adj["gc_fail"]:
            lines.append(f"  {_row_verbatim(row)}")
    lines.append("")
    n_fail = len(adj["gr_fail"])
    gr = "PASS" if adj["gr_pass"] else "FAIL"
    lines.append(
        f"### G-R (strict) {gr}: ref_agree==1 on {n - n_fail}/{n} recorded rows."
    )
    if adj["gr_fail"]:
        lines.append("Failing rows (verbatim):")
        for row in adj["gr_fail"]:
            lines.append(f"  {_row_verbatim(row)}")
    lines.append("")
    lines.append("### G-S (informational)")
    lines.append(
        "Per configuration, first-half vs second-half median t_total_ms "
        "(trial order). Flag if |percent delta| exceeds about 10%."
    )
    for g in adj["gs_rows"]:
        flag = " FLAG" if g["flag"] else ""
        direction = (
            "second half higher"
            if g["pct"] > 0
            else ("second half lower" if g["pct"] < 0 else "unchanged")
        )
        lines.append(
            f"- {g['config_id']} m={g['m']} l_t={g['l_t']} n={g['n']}: "
            f"first-half median {_fmt(g['first'], 4)} ms, second-half median "
            f"{_fmt(g['second'], 4)} ms, delta {_fmt(g['pct'], 2)}% "
            f"({direction}).{flag}"
        )
    lines.append("")
    lines.append("### G-F (report-only)")
    lines.append(
        f"Overhead floor from {adj['n_null']} null-trial median t_total_ms: "
        f"{_fmt(adj['floor_from_nulls'], 4)} ms."
    )
    if adj["floor_discrepancy"] is None:
        lines.append(
            f"Cross-check against t_overhead_floor_ms column: "
            f"{_fmt(adj['col_floor'], 4)} ms (agreement within 0.1 ms)."
        )
    else:
        a, b, uniq = adj["floor_discrepancy"]
        lines.append(
            f"DISCREPANCY: null-trial median {_fmt(a, 6)} ms vs column "
            f"{_fmt(b, 6)} ms; unique column values {uniq}."
        )
    lines.append("Per-config loopback compute-floor context (no threshold):")
    for c in adj["compute_floors"]:
        lines.append(
            f"- {c['config_id']}: median summed per-node compute "
            f"{_fmt(c['median_node_compute_sum_ms'], 4)} ms; "
            f"min t_total_ms {_fmt(c['min_t_total_ms'], 4)} ms."
        )
    lines.append("")
    return "\n".join(lines)


def render_table_section(stats: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("## PER-CONFIG TABLE")
    lines.append("")
    lines.append(EXECUTION_NOTE)
    lines.append("")
    lines.append(
        "| tasks | bundle length | n | rounds mean | rounds P50 | rounds P95 | "
        "rounds P99 | rounds max | cap (l_t*m) | max/cap | t_total mean | "
        "t_total P50 | t_total P95 | t_total P99 | t_total max | "
        "rounds P99/median | t_total P99/median | msgs mean | bytes mean | "
        "m x 9.6 ms | m x 23.38 ms | m x 30.0 ms |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in stats:
        lines.append(
            "| {m} | {bw} | {n} | {rm} | {r50} | {r95} | {r99} | {rmax} | "
            "{cap} | {mtc} | {tm} | {t50} | {t95} | {t99} | {tmax} | "
            "{rd} | {td} | {msg} | {byt} | {a} | {b} | {c} |".format(
                m=s["m"],
                bw=s["bundle_words"],
                n=s["n"],
                rm=_fmt(s["rounds_mean"], 3),
                r50=_fmt(s["rounds_p50"], 3),
                r95=_fmt(s["rounds_p95"], 3),
                r99=_fmt(s["rounds_p99"], 3),
                rmax=_fmt(s["rounds_max"], 3),
                cap=s["cap"],
                mtc=_fmt(s["rounds_max_to_cap"], 3),
                tm=_fmt(s["t_mean"], 3),
                t50=_fmt(s["t_p50"], 3),
                t95=_fmt(s["t_p95"], 3),
                t99=_fmt(s["t_p99"], 3),
                tmax=_fmt(s["t_max"], 3),
                rd=_fmt(s["rounds_disp"], 3),
                td=_fmt(s["t_disp"], 3),
                msg=_fmt(s["msgs_mean"], 1),
                byt=_fmt(s["bytes_mean"], 1),
                a=_fmt(s["ref_96"], 1),
                b=_fmt(s["ref_2338"], 2),
                c=_fmt(s["ref_30"], 1),
            )
        )
    lines.append("")
    lines.append("Footnotes (anchor provenance, T0 R5):")
    lines.append(
        "- 9.6 ms mean per-subround at frontier size 1 exists in the paper "
        "supplementary only, no repo artifact (T0 R5)."
    )
    lines.append(
        "- 23.38 ms P99 from vg_analysis_output.md line 8."
    )
    lines.append(
        "- 30.0 ms calibrated envelope from vg_t3_combined.md line 3."
    )
    lines.append("")
    return "\n".join(lines)


SUMMARY_FIELDS = [
    ("tasks", "tasks"),
    ("bundle_length", "bundle length"),
    ("n_recorded", "recorded trials"),
    ("rounds_mean", "rounds to convergence, mean"),
    ("rounds_p50", "rounds to convergence, 50th percentile"),
    ("rounds_p95", "rounds to convergence, 95th percentile"),
    ("rounds_p99", "rounds to convergence, 99th percentile"),
    ("rounds_max", "rounds to convergence, maximum"),
    ("cap", "round cap (bundle length times number of tasks)"),
    ("max_to_cap", "maximum rounds divided by the round cap"),
    ("t_mean", "convergence wall time, mean (milliseconds)"),
    ("t_p50", "convergence wall time, 50th percentile (milliseconds)"),
    ("t_p95", "convergence wall time, 95th percentile (milliseconds)"),
    ("t_p99", "convergence wall time, 99th percentile (milliseconds)"),
    ("t_max", "convergence wall time, maximum (milliseconds)"),
    ("rounds_disp", "rounds to convergence, 99th percentile over median"),
    ("t_disp", "convergence wall time, 99th percentile over median"),
    ("msgs_mean", "messages sent, mean"),
    ("bytes_mean", "bytes sent, mean"),
    ("ref_96", "tasks times 9.6 milliseconds"),
    ("ref_2338", "tasks times 23.38 milliseconds"),
    ("ref_30", "tasks times 30.0 milliseconds"),
]


def write_summary_tables(stats: Sequence[Dict[str, Any]], md_path: str, csv_path: str) -> None:
    os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)
    worded = [lab for _, lab in SUMMARY_FIELDS]
    md_lines = [
        "# Convergence-latency summary",
        "",
        EXECUTION_NOTE,
        "",
        "| " + " | ".join(worded) + " |",
        "|" + "|".join(["---"] * len(worded)) + "|",
    ]
    csv_rows: List[Dict[str, str]] = []
    for s in stats:
        rec = {
            "tasks": str(s["m"]),
            "bundle_length": s["bundle_words"],
            "n_recorded": str(s["n"]),
            "rounds_mean": _fmt(s["rounds_mean"], 3),
            "rounds_p50": _fmt(s["rounds_p50"], 3),
            "rounds_p95": _fmt(s["rounds_p95"], 3),
            "rounds_p99": _fmt(s["rounds_p99"], 3),
            "rounds_max": _fmt(s["rounds_max"], 3),
            "cap": str(s["cap"]),
            "max_to_cap": _fmt(s["rounds_max_to_cap"], 3),
            "t_mean": _fmt(s["t_mean"], 3),
            "t_p50": _fmt(s["t_p50"], 3),
            "t_p95": _fmt(s["t_p95"], 3),
            "t_p99": _fmt(s["t_p99"], 3),
            "t_max": _fmt(s["t_max"], 3),
            "rounds_disp": _fmt(s["rounds_disp"], 3),
            "t_disp": _fmt(s["t_disp"], 3),
            "msgs_mean": _fmt(s["msgs_mean"], 1),
            "bytes_mean": _fmt(s["bytes_mean"], 1),
            "ref_96": _fmt(s["ref_96"], 1),
            "ref_2338": _fmt(s["ref_2338"], 2),
            "ref_30": _fmt(s["ref_30"], 1),
        }
        csv_rows.append(rec)
        md_lines.append("| " + " | ".join(rec[k] for k, _ in SUMMARY_FIELDS) + " |")
    md_lines.append("")
    md_lines.append(
        "Footnotes: 9.6 ms mean per-subround at frontier size 1 exists in the "
        "paper supplementary only, no repo artifact (T0 R5). 23.38 ms P99 from "
        "vg_analysis_output.md line 8. 30.0 ms calibrated envelope from "
        "vg_t3_combined.md line 3."
    )
    md_lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=worded)
        w.writeheader()
        for rec in csv_rows:
            w.writerow({lab: rec[k] for k, lab in SUMMARY_FIELDS})


def append_results_section(gates_path: str, body: str) -> None:
    text = open(gates_path, encoding="utf-8").read()
    if RESULTS_MARKER not in text:
        raise SystemExit(f"RESULTS marker missing in {gates_path}")
    prefix = text.split(RESULTS_MARKER, 1)[0]
    new = prefix.rstrip("\n") + "\n\n## RESULTS\n\n" + body.rstrip("\n") + "\n"
    with open(gates_path, "w", encoding="utf-8") as f:
        f.write(new)


def render_results_body(adj: Dict[str, Any], stats: Sequence[Dict[str, Any]]) -> str:
    gv = "PASS" if adj["gv_pass"] else "FAIL"
    gc = "PASS" if adj["gc_pass"] else "FAIL"
    gr = "PASS" if adj["gr_pass"] else "FAIL"
    lines = [
        f"G-V {gv}: valid==1 on {adj['n_recorded'] - len(adj['gv_fail'])}/{adj['n_recorded']} recorded rows.",
        f"G-C {gc}: rounds <= l_t*m on {adj['n_recorded'] - len(adj['gc_fail'])}/{adj['n_recorded']} recorded rows.",
        f"G-R {gr}: ref_agree==1 on {adj['n_recorded'] - len(adj['gr_fail'])}/{adj['n_recorded']} recorded rows.",
        f"G-F report-only: null-trial overhead floor { _fmt(adj['floor_from_nulls'], 4) } ms "
        f"(n_null={adj['n_null']}).",
        "",
        "G-S informational (first-half vs second-half median t_total_ms):",
    ]
    for g in adj["gs_rows"]:
        flag = " beyond ~10%" if g["flag"] else ""
        lines.append(
            f"- {g['config_id']}: {_fmt(g['first'], 3)} vs {_fmt(g['second'], 3)} ms "
            f"({_fmt(g['pct'], 2)}%){flag}."
        )
    lines.append("")
    lines.append("Recorded-row medians (t_total_ms / rounds P50):")
    for s in stats:
        lines.append(
            f"- {s['config_id']}: t_p50={_fmt(s['t_p50'], 3)} ms, "
            f"rounds_p50={_fmt(s['rounds_p50'], 3)}, n={s['n']}."
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out_md", default=DEFAULT_ANALYSIS_MD)
    ap.add_argument("--gates", default=DEFAULT_GATES)
    ap.add_argument("--summary_md", default=DEFAULT_SUMMARY_MD)
    ap.add_argument("--summary_csv", default=DEFAULT_SUMMARY_CSV)
    args = ap.parse_args(list(argv) if argv is not None else None)

    log_path = os.path.join(args.root, "cbba_log.csv")
    rows = load_log(log_path)
    nulls, _warmups, recorded = partition_rows(rows)
    adj = adjudicate(nulls, recorded)
    stats = [config_stats(g) for _, g in _group_recorded(recorded)]
    body = render_gate_section(adj) + render_table_section(stats)
    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(body)
        if not body.endswith("\n"):
            f.write("\n")
    sys.stdout.write(body)
    if not body.endswith("\n"):
        sys.stdout.write("\n")
    write_summary_tables(stats, args.summary_md, args.summary_csv)
    append_results_section(args.gates, render_results_body(adj, stats))
    print(f"wrote {args.out_md}", file=sys.stderr)
    print(f"wrote {args.summary_md}", file=sys.stderr)
    print(f"wrote {args.summary_csv}", file=sys.stderr)
    print(f"appended RESULTS to {args.gates}", file=sys.stderr)
    strict_fail = not (adj["gv_pass"] and adj["gc_pass"] and adj["gr_pass"])
    return 1 if strict_fail else 0


if __name__ == "__main__":
    sys.exit(main())
