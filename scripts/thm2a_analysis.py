#!/usr/bin/env python3
"""EXP-2 Theorem 2(a) sweep analysis.

Joins each run's release_log.csv with docs/exp2/exp2_config_table.json,
evaluates pre-registered gates GI/GA/GB, and reports GC/GP/GS. Writes the
supplementary table and appends a RESULTS section to docs/exp2/exp2_gates.md
without mutating the D1 prefix.

rt_bound_violations in periodic mode is VACUOUS (class-id iteration; see
docs/exp2/exp2_gates.md D0). The column is recorded but not gated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

D1_MARKER = "\n## RESULTS"
DEFAULT_TABLE = os.path.join(_PROJECT_ROOT, "docs", "exp2", "exp2_config_table.json")
DEFAULT_GATES = os.path.join(_PROJECT_ROOT, "docs", "exp2", "exp2_gates.md")
DEFAULT_SUP_CSV = os.path.join(
    _PROJECT_ROOT, "docs", "exp2", "exp2_supplementary_table.csv"
)
DEFAULT_SUP_MD = os.path.join(
    _PROJECT_ROOT, "docs", "exp2", "exp2_supplementary_table.md"
)
PROGRESS_LOG_NAME = "sweep_exp2_progress.log"
RELEASE_LOG_NAME = "release_log.csv"
MAIN_CSV_NAME = "darts_log.csv"
CONFIG_JSON_NAME = "config.json"

WORKLOAD_DISPLAY = {
    "B4_DEM": "B4 Distributed Environmental Monitoring",
    "B3_IQI": "B3 Industrial Quality-Inspection Pipeline",
}

SUP_CSV_FIELDS = [
    "config_id", "workload", "verdict", "eval_seed_base",
    "D_v_ms", "P_ms", "L",
    "delta_sum_bottleneck",
    "delta_sum_agent_1", "delta_sum_agent_2",
    "delta_sum_agent_3", "delta_sum_agent_4",
    "n_rows", "n_miss", "n_drain_fail", "n_fallback_blocked", "n_alloc_mismatch",
    "ratio_max", "ratio_p99",
    "ratio_max_by_class",
    "battery_min_by_agent",
    "rt_bound_violations",
    "rc", "provenance_ok",
]


def d1_prefix(text: str) -> str:
    if D1_MARKER in text:
        return text.split(D1_MARKER, 1)[0]
    return text


def d1_sha256(text: str) -> str:
    return hashlib.sha256(d1_prefix(text).encode("utf-8")).hexdigest()


def load_table(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    rows = blob.get("configs") if isinstance(blob, dict) else blob
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"config table {path} has no configs")
    return rows


def discover_results_root(repo: str) -> Path:
    """Unique launcher-style results_exp2_YYYYMMDD directory.

    results_exp2_dryrun and P1 engineering trees (p1_g*) are excluded.
    """
    base = Path(repo)
    cands = [
        p for p in sorted(base.iterdir())
        if p.is_dir() and re.fullmatch(r"results_exp2_\d{8}", p.name)
    ]
    if len(cands) == 0:
        raise SystemExit("REFUSAL: no results_exp2_YYYYMMDD root found")
    if len(cands) > 1:
        names = ", ".join(p.name for p in cands)
        raise SystemExit(f"REFUSAL: multiple results_exp2_YYYYMMDD roots: {names}")
    return cands[0]


def parse_progress(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            continue
        m = re.search(r"(\S+) rc=(\S+)", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _as_bool(v: Any) -> bool:
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no", ""):
        return False
    raise ValueError(f"not a boolean: {v!r}")


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if s in ("", "N/A", "None", "nan", "NaN"):
        return None
    return float(s)


def _load_alloc(repo: str, rel: str) -> Dict[str, str]:
    path = rel if os.path.isabs(rel) else os.path.join(repo, rel)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _provenance_ok(row: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[bool, List[str]]:
    bad: List[str] = []

    def close(a: Any, b: Any) -> bool:
        try:
            return abs(float(a) - float(b)) < 1e-6
        except (TypeError, ValueError):
            return False

    if not close(cfg.get("release_period_ms"), row["P_ms"]):
        bad.append("P")
    rel = cfg.get("workload_relative_deadline_ms", cfg.get("relative_deadline_ms"))
    if not close(rel, row["D_v_ms"]):
        bad.append("D_v")
    if int(cfg.get("num_releases", -1)) != int(row["L"]):
        bad.append("L")
    if int(cfg.get("eval_seed_base", -1)) != int(row["eval_seed_base"]):
        bad.append("seed")
    if int(cfg.get("load_ckpt_ep", -1)) != int(row["ckpt_ep"]):
        bad.append("ep")
    ckpt = str(cfg.get("load_ckpt", ""))
    want = str(row["ckpt_dir"])
    if want not in ckpt and not ckpt.endswith(Path(want).name):
        bad.append("ckpt")
    alloc = str(cfg.get("alloc_map", ""))
    if Path(alloc).name != Path(row["alloc_map_file"]).name:
        bad.append("alloc")
    return (not bad), bad


def _read_rt_bound_sum(main_csv: Path) -> Optional[float]:
    if not main_csv.is_file():
        return None
    with main_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows or "rt_bound_violations" not in (reader.fieldnames or []):
        return None
    total = 0.0
    for r in rows:
        v = _as_float(r.get("rt_bound_violations"))
        if v is not None:
            total += v
    return total


def analyze_config(
    row: Dict[str, Any],
    run_dir: Path,
    repo: str,
    rc: Optional[str],
) -> Dict[str, Any]:
    cid = row["config_id"]
    alloc = _load_alloc(repo, row["alloc_map_file"])
    d_v = float(row["D_v_ms"])
    factored = dict(row.get("delta_sum_factored") or {})
    bottleneck = max(float(v) for v in factored.values()) if factored else float("nan")

    rec: Dict[str, Any] = {
        "config_id": cid,
        "workload": row["workload"],
        "verdict": row["verdict"],
        "outside_certified_region": bool(row.get("outside_certified_region")),
        "eval_seed_base": int(row["eval_seed_base"]),
        "target_delta_sum": row.get("target_delta_sum"),
        "D_v_ms": d_v,
        "P_ms": float(row["P_ms"]),
        "L": int(row["L"]),
        "delta_sum_factored": factored,
        "delta_sum_bottleneck": bottleneck,
        "rc": rc if rc is not None else "MISSING",
        "n_rows": 0,
        "n_miss": 0,
        "n_drain_fail": 0,
        "n_fallback_blocked": 0,
        "n_alloc_mismatch": 0,
        "ratios": [],
        "ratio_by_class": {},
        "battery_min_by_agent": {},
        "rt_bound_violations": None,
        "provenance_ok": False,
        "provenance_bad": ["no_config_json"],
        "miss_rows": [],
        "drain_fail_rows": [],
        "alloc_mismatch_rows": [],
        "run_dir": str(run_dir),
        "has_release_log": False,
    }

    cfg_path = run_dir / CONFIG_JSON_NAME
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        ok, bad = _provenance_ok(row, cfg)
        rec["provenance_ok"] = ok
        rec["provenance_bad"] = bad

    rec["rt_bound_violations"] = _read_rt_bound_sum(run_dir / MAIN_CSV_NAME)

    rl = run_dir / RELEASE_LOG_NAME
    if not rl.is_file():
        return rec
    rec["has_release_log"] = True
    with rl.open(encoding="utf-8", newline="") as f:
        jobs = list(csv.DictReader(f))
    rec["n_rows"] = len(jobs)

    batt: Dict[str, float] = {}
    ratio_by_class: Dict[str, List[float]] = {}
    ratios: List[float] = []
    for j in jobs:
        inst = j.get("instance_id", "")
        cls = j.get("job_class", "")
        agent = j.get("agent_id", "")
        want = alloc.get(cls)
        if want is not None and agent != want:
            rec["n_alloc_mismatch"] += 1
            rec["alloc_mismatch_rows"].append(inst)
        if _as_bool(j.get("fallback_blocked")):
            rec["n_fallback_blocked"] += 1
        if not _as_bool(j.get("met")):
            rec["n_miss"] += 1
            rec["miss_rows"].append(inst)
        if not _as_bool(j.get("queue_empty_at_next_boundary")):
            rec["n_drain_fail"] += 1
            rec["drain_fail_rows"].append(inst)
        b = _as_float(j.get("battery_at_boundary_wh"))
        if b is not None and agent:
            if agent not in batt or b < batt[agent]:
                batt[agent] = b
        finish = _as_float(j.get("finish_ms"))
        r_ms = _as_float(j.get("r_ms"))
        if finish is not None and r_ms is not None and d_v > 0:
            ratio = (finish - r_ms) / d_v
            ratios.append(ratio)
            ratio_by_class.setdefault(cls, []).append(ratio)

    rec["ratios"] = ratios
    rec["ratio_by_class"] = ratio_by_class
    rec["battery_min_by_agent"] = batt
    rec["ratio_max"] = float(max(ratios)) if ratios else float("nan")
    rec["ratio_p99"] = (
        float(np.percentile(np.asarray(ratios, dtype=float), 99))
        if ratios else float("nan")
    )
    rec["ratio_max_by_class"] = {
        k: float(max(vs)) for k, vs in sorted(ratio_by_class.items())
    }
    return rec


def analyze_sweep(
    table: Sequence[Dict[str, Any]],
    results_root: Path,
    repo: str,
) -> List[Dict[str, Any]]:
    progress = parse_progress(results_root / PROGRESS_LOG_NAME)
    out = []
    for row in table:
        cid = row["config_id"]
        out.append(analyze_config(row, results_root / cid, repo, progress.get(cid)))
    return out


def evaluate_gates(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    n_rc0 = sum(1 for r in records if str(r.get("rc")) == "0")
    n_120 = sum(1 for r in records if int(r.get("n_rows", 0)) == 120 and r.get("has_release_log"))
    n_mismatch = sum(int(r.get("n_alloc_mismatch", 0)) for r in records)
    admit = [r for r in records if r.get("verdict") == "ADMIT"]
    n_admit_rows = sum(int(r.get("n_rows", 0)) for r in admit)
    n_admit_miss = sum(int(r.get("n_miss", 0)) for r in admit)
    n_admit_drain = sum(int(r.get("n_drain_fail", 0)) for r in admit)
    n_admit_fb = sum(int(r.get("n_fallback_blocked", 0)) for r in admit)
    miss_rows = [(r["config_id"], r["miss_rows"]) for r in admit if r.get("miss_rows")]
    drain_rows = [(r["config_id"], r["drain_fail_rows"]) for r in admit if r.get("drain_fail_rows")]

    gi_pass = (
        n_rc0 == n
        and n_120 == n
        and n_mismatch == 0
        and n_admit_fb == 0
        and all(r.get("has_release_log") for r in records)
        and all(r.get("provenance_ok") for r in records)
    )
    ga_pass = n_admit_miss == 0 and n_admit_rows == len(admit) * 120
    gb_pass = n_admit_drain == 0 and n_admit_rows == len(admit) * 120

    return {
        "n_configs": n,
        "n_rc0": n_rc0,
        "n_120": n_120,
        "n_alloc_mismatch": n_mismatch,
        "n_admit": len(admit),
        "n_admit_rows": n_admit_rows,
        "n_admit_miss": n_admit_miss,
        "n_admit_drain": n_admit_drain,
        "n_admit_fallback_blocked": n_admit_fb,
        "gi_pass": gi_pass,
        "ga_pass": ga_pass,
        "gb_pass": gb_pass,
        "miss_rows": miss_rows,
        "drain_fail_rows": drain_rows,
    }


def _fmt_float(x: Any, digits: int = 12) -> str:
    if x is None:
        return "N/A"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if np.isnan(v):
        return "N/A"
    return f"{v:.{digits}g}"


def format_results(ev: Dict[str, Any], records: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    gi = "PASS" if ev["gi_pass"] else "FAIL"
    ga = "PASS" if ev["ga_pass"] else "FAIL"
    gb = "PASS" if ev["gb_pass"] else "FAIL"
    lines.append(
        f"GI {gi}: {ev['n_rc0']}/{ev['n_configs']} rc=0; "
        f"{ev['n_120']}/{ev['n_configs']} release_log.csv with 120 rows; "
        f"alloc-map mismatches {ev['n_alloc_mismatch']}; "
        f"ADMIT fallback_blocked {ev['n_admit_fallback_blocked']}."
    )
    lines.append(
        f"GA {ga}: met=true on {ev['n_admit_rows'] - ev['n_admit_miss']} of "
        f"{ev['n_admit_rows']} ADMIT job rows "
        f"(misses {ev['n_admit_miss']}). "
        f"rt_bound_violations clause not applicable, see D0 rationale."
    )
    lines.append(
        f"GB {gb}: queue_empty_at_next_boundary=true on "
        f"{ev['n_admit_rows'] - ev['n_admit_drain']} of {ev['n_admit_rows']} "
        f"ADMIT job rows (drain failures {ev['n_admit_drain']})."
    )
    lines.append("")
    lines.append("GC (informational):")
    for r in records:
        per = ", ".join(
            f"{k}={_fmt_float(v, 6)}"
            for k, v in (r.get("ratio_max_by_class") or {}).items()
        )
        lines.append(
            f"  {r['config_id']}: max={_fmt_float(r.get('ratio_max'), 6)} "
            f"P99={_fmt_float(r.get('ratio_p99'), 6)} per-class max [{per}]"
        )
    lines.append("")
    lines.append("GP (report-only, REJECT-verdict configs):")
    probes = [r for r in records if r.get("verdict") == "REJECT"]
    if not probes:
        lines.append("  (none)")
    for r in probes:
        lines.append(
            f"  {r['config_id']}: misses={r['n_miss']} "
            f"drain_failures={r['n_drain_fail']} "
            f"fallback_blocked={r['n_fallback_blocked']} "
            f"ratio_max={_fmt_float(r.get('ratio_max'), 6)} "
            f"ratio_P99={_fmt_float(r.get('ratio_p99'), 6)}"
        )
    lines.append("")
    lines.append("GS (report-only, density-0.85 vs seed-200 repetition):")
    for wl in ("B4_DEM", "B3_IQI"):
        primary = [
            r for r in records
            if r["workload"] == wl
            and r.get("target_delta_sum") == 0.85
            and int(r.get("eval_seed_base", 0)) == 100
        ]
        second = [
            r for r in records
            if r["workload"] == wl
            and r.get("target_delta_sum") == 0.85
            and int(r.get("eval_seed_base", 0)) == 200
        ]
        if not primary or not second:
            lines.append(f"  {wl}: missing pair")
            continue
        a, b = primary[0], second[0]
        lines.append(
            f"  {wl}: {a['config_id']} (seed 100) vs {b['config_id']} (seed 200); "
            f"GA misses {a['n_miss']}/{b['n_miss']}; "
            f"GB drain {a['n_drain_fail']}/{b['n_drain_fail']}; "
            f"GC max {_fmt_float(a.get('ratio_max'), 6)}/"
            f"{_fmt_float(b.get('ratio_max'), 6)} "
            f"P99 {_fmt_float(a.get('ratio_p99'), 6)}/"
            f"{_fmt_float(b.get('ratio_p99'), 6)}"
        )
    return "\n".join(lines) + "\n"


def write_supplementary(
    records: Sequence[Dict[str, Any]],
    csv_path: str,
    md_path: str,
) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUP_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            factored = r.get("delta_sum_factored") or {}
            w.writerow({
                "config_id": r["config_id"],
                "workload": r["workload"],
                "verdict": r["verdict"],
                "eval_seed_base": r["eval_seed_base"],
                "D_v_ms": r["D_v_ms"],
                "P_ms": r["P_ms"],
                "L": r["L"],
                "delta_sum_bottleneck": r.get("delta_sum_bottleneck"),
                "delta_sum_agent_1": factored.get("agent_1", ""),
                "delta_sum_agent_2": factored.get("agent_2", ""),
                "delta_sum_agent_3": factored.get("agent_3", ""),
                "delta_sum_agent_4": factored.get("agent_4", ""),
                "n_rows": r["n_rows"],
                "n_miss": r["n_miss"],
                "n_drain_fail": r["n_drain_fail"],
                "n_fallback_blocked": r["n_fallback_blocked"],
                "n_alloc_mismatch": r["n_alloc_mismatch"],
                "ratio_max": r.get("ratio_max"),
                "ratio_p99": r.get("ratio_p99"),
                "ratio_max_by_class": json.dumps(r.get("ratio_max_by_class") or {}),
                "battery_min_by_agent": json.dumps(r.get("battery_min_by_agent") or {}),
                "rt_bound_violations": r.get("rt_bound_violations"),
                "rc": r.get("rc"),
                "provenance_ok": r.get("provenance_ok"),
            })

    lines = [
        "# EXP-2 supplementary table (internal)",
        "",
        "config ids are allowed in this internal artifact. "
        "`rt_bound_violations` is recorded from the main eval CSV but is "
        "**VACUOUS** in periodic mode (class-id iteration; see "
        "`exp2_gates.md` D0). Claims A/B/C are adjudicated from "
        "`release_log.csv` only.",
        "",
        "| config_id | workload | verdict | seed | D_v | P | L | "
        "δ_sum bottleneck | n_rows | misses | drain fail | fallback | "
        "alloc mismatch | ratio max | ratio P99 | rt_bound_violations |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        lines.append(
            "| {cid} | {wl} | {ver} | {seed} | {dv:g} | {p:g} | {L} | "
            "{bot:.16g} | {n} | {miss} | {drain} | {fb} | {am} | "
            "{rmax} | {rp99} | {rt} |".format(
                cid=r["config_id"],
                wl=r["workload"],
                ver=r["verdict"],
                seed=r["eval_seed_base"],
                dv=r["D_v_ms"],
                p=r["P_ms"],
                L=r["L"],
                bot=float(r.get("delta_sum_bottleneck") or float("nan")),
                n=r["n_rows"],
                miss=r["n_miss"],
                drain=r["n_drain_fail"],
                fb=r["n_fallback_blocked"],
                am=r["n_alloc_mismatch"],
                rmax=_fmt_float(r.get("ratio_max"), 16),
                rp99=_fmt_float(r.get("ratio_p99"), 16),
                rt=_fmt_float(r.get("rt_bound_violations"), 6),
            )
        )
    Path(md_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_results_section(gates_path: str, body: str) -> str:
    """Rewrite file as D1 prefix + RESULTS. Returns the D1 sha256 after write."""
    original = Path(gates_path).read_text(encoding="utf-8")
    prefix = d1_prefix(original)
    before = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
    new = prefix.rstrip("\n") + "\n\n## RESULTS\n\n" + body.rstrip("\n") + "\n"
    Path(gates_path).write_text(new, encoding="utf-8")
    after = hashlib.sha256(d1_prefix(new).encode("utf-8")).hexdigest()
    if before != after:
        raise SystemExit("REFUSAL: D1 prefix hash changed while appending RESULTS")
    return after


def collect_plot_curves(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Payload for the response-ratio CDF. Omits the seed-200 pair (GS table)."""
    curves = []
    for r in records:
        if int(r.get("eval_seed_base", 100)) == 200:
            continue
        curves.append({
            "workload": r["workload"],
            "display": WORKLOAD_DISPLAY.get(r["workload"], r["workload"]),
            "density": float(r.get("delta_sum_bottleneck") or float("nan")),
            "outside": bool(r.get("outside_certified_region")),
            "values": list(r.get("ratios") or []),
        })
    return curves


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_root", default=None)
    ap.add_argument("--config_table", default=DEFAULT_TABLE)
    ap.add_argument("--gates_path", default=DEFAULT_GATES)
    ap.add_argument("--sup_csv", default=DEFAULT_SUP_CSV)
    ap.add_argument("--sup_md", default=DEFAULT_SUP_MD)
    ap.add_argument("--repo", default=_PROJECT_ROOT)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    repo = os.path.abspath(args.repo)
    table = load_table(args.config_table)
    root = (
        Path(args.results_root)
        if args.results_root
        else discover_results_root(repo)
    )
    records = analyze_sweep(table, root, repo)
    ev = evaluate_gates(records)
    body = format_results(ev, records)
    write_supplementary(records, args.sup_csv, args.sup_md)
    append_results_section(args.gates_path, body)
    sys.stdout.write(body)
    if not (ev["gi_pass"] and ev["ga_pass"] and ev["gb_pass"]):
        if ev["miss_rows"]:
            sys.stderr.write("[thm2a_analysis] GA failing instance ids:\n")
            for cid, ids in ev["miss_rows"]:
                sys.stderr.write(f"  {cid}: {ids}\n")
        if ev["drain_fail_rows"]:
            sys.stderr.write("[thm2a_analysis] GB failing instance ids:\n")
            for cid, ids in ev["drain_fail_rows"]:
                sys.stderr.write(f"  {cid}: {ids}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
