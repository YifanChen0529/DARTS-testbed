#!/usr/bin/env python3
"""Collect TB-3 empirical-anchor runs into committed CSVs + gate block.

Walks ``results_tb3_emp/*/``, header-keyed via ``safe_read_csv`` (never
positional). Provenance is a hard cross-check against
``generate_instance_fields`` and the signed-off deadline / ckpt / eval flags.

Phase 1 smoke: ``--expect-n 1`` without ``--grid``. Part B: expect 180 and
join the offline grid.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alg.stress_profiles import get_stress_profile  # noqa: E402
from alg.tb3_instance import generate_instance_fields  # noqa: E402
from envs.workloads import get_workload  # noqa: E402
from scripts.csv_schema_tb import (  # noqa: E402
    TB3_EMP_FEAS_HEADER,
    TB3_EMP_FEAS_P7_HEADER,
    TB3_EMP_RUNS_HEADER,
    TB3_EMP_RUNS_P7_HEADER,
)
from scripts.sweep_tb3 import (  # noqa: E402
    PLAN_SPECS,
    TB3_EMP_CKPT_SEEDS,
    TB3_EMP_EVAL_EPISODES,
    TB3_EMP_INSTANCE_SEEDS,
    TB3_EMP_N_RUNS,
    TB3_EMP_PINNED_EP,
    TB3_EMP_PLAN,
    parse_cell,
    parse_exp_name,
)
from scripts.sweep_tb1 import DEFAULT_MANIFEST, SweepConfigError, load_manifest  # noqa: E402
from scripts.util_csv import safe_read_csv  # noqa: E402

NOMINAL_RELATIVE_MS = 150000.0
NOMINAL_GLOBAL_MS = {"medium": 150000.0, "heavy": 120000.0}
MISS_COUNTERS = (
    "n_miss_full_fidelity_late",
    "n_miss_n1_drop",
    "n_miss_n2_fallback_insufficient",
    "n_miss_n3_unreached",
)


class CollectError(RuntimeError):
    """Hard failure: missing join, provenance mismatch, or incomplete plan."""


def e2e_clean(task_miss_rate: Any) -> bool:
    v = _as_float(task_miss_rate)
    return (not math.isnan(v)) and v == 0.0


def perjob_clean(counters: Sequence[Any]) -> bool:
    if len(counters) != 4:
        return False
    for c in counters:
        v = _as_float(c)
        if math.isnan(v) or v != 0.0:
            return False
    return True


def majority(bits: Sequence[bool], *, n_runs: int = 3, need: int = 2) -> bool:
    if len(bits) != n_runs:
        return False
    return sum(1 for b in bits if b) >= need


def _as_float(v: Any) -> float:
    if v is None:
        return float("nan")
    if isinstance(v, float) and math.isnan(v):
        return float("nan")
    if isinstance(v, str):
        s = v.strip()
        if not s or s.upper() in ("N/A", "NAN", "NONE"):
            return float("nan")
        return float(s)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _float_map(d: Any) -> Dict[str, float]:
    if not isinstance(d, dict):
        raise CollectError(f"expected dict, got {type(d).__name__}")
    return {str(k): float(v) for k, v in d.items()}


def _loc_map(d: Any) -> Dict[str, List[float]]:
    if not isinstance(d, dict):
        raise CollectError(f"expected loc dict, got {type(d).__name__}")
    out: Dict[str, List[float]] = {}
    for k, v in d.items():
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise CollectError(f"location {k!r} is not a length-2 pair: {v!r}")
        out[str(k)] = [float(v[0]), float(v[1])]
    return out


def _maps_equal(a: Dict[str, float], b: Dict[str, float]) -> bool:
    if set(a) != set(b):
        return False
    return all(a[k] == b[k] for k in a)


def _locs_equal(a: Dict[str, List[float]], b: Dict[str, List[float]]) -> bool:
    if set(a) != set(b):
        return False
    return all(a[k][0] == b[k][0] and a[k][1] == b[k][1] for k in a)


def expected_keys(
    plans: Sequence[str] = ("default",),
) -> List[Tuple[str, float, int, int]]:
    keys: List[Tuple[str, float, int, int]] = []
    seen = set()
    for name in plans:
        if name not in PLAN_SPECS:
            raise CollectError(f"unknown plan {name!r}")
        spec = PLAN_SPECS[name]
        for cell, betas in spec["plan"].items():
            for beta in betas:
                for inst in spec["seeds"]:
                    for ckpt in TB3_EMP_CKPT_SEEDS:
                        key = (cell, float(beta), int(inst), int(ckpt))
                        if key in seen:
                            raise CollectError(f"plan union collision on {key}")
                        seen.add(key)
                        keys.append(key)
    return keys


def plan_of_key(
    cell: str, beta: float, inst: int, plans: Sequence[str]
) -> str:
    matches = []
    for name in plans:
        spec = PLAN_SPECS[name]
        if cell not in spec["plan"]:
            continue
        if int(inst) not in spec["seeds"]:
            continue
        if float(beta) not in {float(b) for b in spec["plan"][cell]}:
            continue
        matches.append(name)
    if len(matches) != 1:
        raise CollectError(
            f"plan assignment for {cell} s={beta} i{inst} in {list(plans)}: {matches}"
        )
    return matches[0]


def expected_fields(cell: str, instance_seed: int):
    bench, stress = parse_cell(cell)
    wl = get_workload(bench)
    profile = get_stress_profile(bench, stress)
    if profile is None:
        raise CollectError(f"no stress profile for {cell}")
    bats = profile["battery_capacities_wh"]
    n_agents = len(bats)
    return generate_instance_fields(wl, bats, n_agents, instance_seed)


def check_provenance(
    cfg: Dict[str, Any],
    meta: Dict[str, Any],
) -> None:
    """Hard-fail on any signed-off provenance mismatch."""
    cell = meta["cell"]
    beta = float(meta["beta"])
    inst = int(meta["instance_seed"])
    ckpt = int(meta["ckpt_seed"])
    bench, stress = parse_cell(cell)

    got_inst = cfg.get("instance_seed")
    if int(got_inst) != inst:
        raise CollectError(
            f"{meta['exp_name']}: config instance_seed={got_inst!r} != {inst}"
        )

    fields = expected_fields(cell, inst)
    want_w = {str(k): float(v) for k, v in fields.task_wcets_ms.items()}
    want_l = {str(k): [float(p[0]), float(p[1])] for k, p in fields.task_locations_m.items()}
    got_w = _float_map(cfg.get("workload_task_wcets_ms"))
    got_l = _loc_map(cfg.get("workload_task_locations_m"))
    if not _maps_equal(got_w, want_w):
        raise CollectError(
            f"{meta['exp_name']}: workload_task_wcets_ms != generate_instance_fields"
        )
    if not _locs_equal(got_l, want_l):
        raise CollectError(
            f"{meta['exp_name']}: workload_task_locations_m != generate_instance_fields"
        )
    cfg_w = _float_map(cfg.get("task_wcets"))
    if not _maps_equal(cfg_w, want_w):
        raise CollectError(
            f"{meta['exp_name']}: cfg.task_wcets != jittered dict"
        )

    rel = _as_float(cfg.get("workload_relative_deadline_ms"))
    want_rel = NOMINAL_RELATIVE_MS * beta
    if rel != want_rel:
        raise CollectError(
            f"{meta['exp_name']}: workload_relative_deadline_ms={rel} != {want_rel}"
        )
    glob = _as_float(cfg.get("global_deadline_ms"))
    want_glob = NOMINAL_GLOBAL_MS[stress] * beta
    if glob != want_glob:
        raise CollectError(
            f"{meta['exp_name']}: global_deadline_ms={glob} != {want_glob} "
            f"({stress} x {beta})"
        )
    if float(cfg.get("cert_conservatism")) != 1.0:
        raise CollectError(f"{meta['exp_name']}: cert_conservatism != 1.0")
    if int(cfg.get("load_ckpt_ep")) != TB3_EMP_PINNED_EP:
        raise CollectError(f"{meta['exp_name']}: load_ckpt_ep != {TB3_EMP_PINNED_EP}")
    if int(cfg.get("eval_seed_base")) != ckpt:
        raise CollectError(
            f"{meta['exp_name']}: eval_seed_base={cfg.get('eval_seed_base')} != ckpt {ckpt}"
        )
    if cfg.get("eval_only") not in (True, "true", "True", 1):
        raise CollectError(f"{meta['exp_name']}: eval_only is not true")
    ds = _as_float(cfg.get("deadline_scale"))
    if ds != float(beta):
        raise CollectError(
            f"{meta['exp_name']}: deadline_scale={ds} != beta {beta}"
        )


def _last_eval_row(exp_dir: Path) -> Dict[str, Any]:
    path = exp_dir / "darts_log.csv"
    df = safe_read_csv(path)
    if df is None or df.empty:
        raise CollectError(f"no readable eval row in {path}")
    row = df.iloc[-1].to_dict()
    return row


def _load_config(exp_dir: Path) -> Dict[str, Any]:
    path = exp_dir / "config.json"
    if not path.is_file():
        raise CollectError(f"missing {path}")
    with path.open() as fh:
        cfg = json.load(fh)
    if not isinstance(cfg, dict):
        raise CollectError(f"{path} is not a JSON object")
    return cfg


def walk_runs(
    results_dir: Any,
    *,
    plans: Sequence[str] = ("default",),
) -> List[Dict[str, Any]]:
    dirs: List[Path]
    if isinstance(results_dir, (list, tuple)):
        dirs = [Path(d) for d in results_dir]
    else:
        dirs = [Path(results_dir)]
    found: List[Dict[str, Any]] = []
    seen_names = set()
    for results_dir in dirs:
        if not results_dir.is_dir():
            raise CollectError(f"results dir not found: {results_dir}")
        for child in sorted(results_dir.iterdir()):
            if not child.is_dir():
                continue
            if not (child / "darts_log.csv").is_file():
                continue
            try:
                meta = parse_exp_name(child.name)
            except SweepConfigError:
                continue
            if child.name in seen_names:
                raise CollectError(f"duplicate exp_name across dirs: {child.name}")
            seen_names.add(child.name)
            cfg = _load_config(child)
            row = _last_eval_row(child)
            check_provenance(cfg, meta)
            eval_eps = cfg.get("eval_episodes", TB3_EMP_EVAL_EPISODES)
            miss = [row.get(k) for k in MISS_COUNTERS]
            rec = {
                "cell": meta["cell"],
                "beta": meta["beta"],
                "instance_seed": meta["instance_seed"],
                "ckpt_seed": meta["ckpt_seed"],
                "eval_seed_base": int(cfg.get("eval_seed_base", meta["ckpt_seed"])),
                "eval_episodes": int(eval_eps),
                "task_miss_rate": _as_float(row.get("task_miss_rate")),
                "n_miss_full_fidelity_late": _as_float(row.get("n_miss_full_fidelity_late")),
                "n_miss_n1_drop": _as_float(row.get("n_miss_n1_drop")),
                "n_miss_n2_fallback_insufficient": _as_float(
                    row.get("n_miss_n2_fallback_insufficient")
                ),
                "n_miss_n3_unreached": _as_float(row.get("n_miss_n3_unreached")),
                "n_ff_missed": _as_float(row.get("n_ff_missed")),
                "rho_hat": _as_float(row.get("rho_hat")),
                "rt_bound_violations": _as_float(row.get("rt_bound_violations")),
                "max_rt_ms": _as_float(row.get("max_rt_ms")),
                "crash_A1": _as_float(row.get("A1_crash_rate")),
                "crash_A2": _as_float(row.get("A2_crash_rate")),
                "crash_A3": _as_float(row.get("A3_crash_rate")),
                "crash_A4": _as_float(row.get("A4_crash_rate")),
                "episode_span_ms": _as_float(row.get("episode_span_ms")),
                "exp_name": child.name,
                "plan": plan_of_key(
                    meta["cell"], float(meta["beta"]), int(meta["instance_seed"]), plans
                ),
            }
            rec["e2e_clean"] = int(e2e_clean(rec["task_miss_rate"]))
            rec["perjob_clean"] = int(perjob_clean(miss))
            found.append(rec)
    return found


def _load_grid_accept(grid_path: Path) -> Dict[Tuple[str, int, float, str], int]:
    """(cell, instance_seed, beta, test) -> accept_e2e. Hard-fail on dups."""
    index: Dict[Tuple[str, int, float, str], int] = {}
    counts: Dict[Tuple[str, int, float, str], int] = defaultdict(int)
    with grid_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cell = f"{row['benchmark']}:{row['stress']}"
            test = row["test"]
            if test not in ("S0", "S1"):
                continue
            xi = row.get("xi_source", "declared")
            if xi != "declared":
                continue
            key = (
                cell,
                int(row["instance_seed"]),
                float(row["beta"]),
                test,
            )
            counts[key] += 1
            index[key] = int(row["accept_e2e"])
    dups = [k for k, n in counts.items() if n != 1]
    if dups:
        raise CollectError(
            f"grid join keys not unique ({len(dups)} duplicates), e.g. {dups[0]}"
        )
    return index


def build_feasibility(
    runs: Sequence[Dict[str, Any]],
    grid_index: Optional[Dict[Tuple[str, int, float, str], int]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, float, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        grouped[(r["cell"], float(r["beta"]), int(r["instance_seed"]))].append(r)
    rows: List[Dict[str, Any]] = []
    for (cell, beta, inst) in sorted(grouped, key=lambda t: (t[0], t[1], t[2])):
        grp = grouped[(cell, beta, inst)]
        n = len(grp)
        e2e_bits = [bool(r["e2e_clean"]) for r in grp]
        pj_bits = [bool(r["perjob_clean"]) for r in grp]
        miss_rates = [_as_float(r["task_miss_rate"]) for r in grp]
        mean_miss = sum(miss_rates) / n if n else float("nan")
        rec: Dict[str, Any] = {
            "cell": cell,
            "beta": beta,
            "instance_seed": inst,
            "n_runs": n,
            "feasible_e2e_all": int(n == 3 and all(e2e_bits)),
            "feasible_e2e_majority": int(majority(e2e_bits, n_runs=n) if n == 3 else False),
            "epfrac_e2e_clean": (1.0 - mean_miss) if not math.isnan(mean_miss) else float("nan"),
            "feasible_perjob_all": int(n == 3 and all(pj_bits)),
            "feasible_perjob_majority": int(
                majority(pj_bits, n_runs=n) if n == 3 else False
            ),
            "plan": grp[0].get("plan", ""),
        }
        if grid_index is not None:
            k0 = (cell, inst, float(beta), "S0")
            k1 = (cell, inst, float(beta), "S1")
            if k0 not in grid_index:
                raise CollectError(f"grid join miss S0 for {k0}")
            if k1 not in grid_index:
                raise CollectError(f"grid join miss S1 for {k1}")
            rec["s0_accept_e2e"] = grid_index[k0]
            rec["s1_accept_e2e"] = grid_index[k1]
        else:
            rec["s0_accept_e2e"] = ""
            rec["s1_accept_e2e"] = ""
        rows.append(rec)
    return rows


def gate_block(
    runs: Sequence[Dict[str, Any]],
    *,
    expect_n: int,
    expected: Sequence[Tuple[str, float, int, int]],
    feas: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[List[str], bool]:
    """Amended P7b gates G1-G6. Strict R1 over all runs is informational.

    Detection limit (G5): a zero-channel miss episode cancelled by a
    multi-job episode in the same run nets to non-positive aggregate
    ``dev`` and is invisible here.
    """
    from scripts.tb3_analysis import (  # noqa: WPS433 — gate-time import
        DETECTION_LIMIT,
        R1_TOL,
        g5_census,
        g6_census,
        r1_census,
    )

    lines: List[str] = []
    keys = [
        (r["cell"], float(r["beta"]), int(r["instance_seed"]), int(r["ckpt_seed"]))
        for r in runs
    ]
    n = len(runs)
    n_dup = n - len(set(keys))
    missing = [k for k in expected if k not in set(keys)]
    extra = [k for k in keys if k not in set(expected)]
    g1 = n == expect_n and n_dup == 0 and not missing and not extra
    lines.append(
        f"G1 completeness {n}/{expect_n}  duplicates={n_dup}  "
        f"missing={len(missing)}  extra={len(extra)}  "
        f"{'PASS' if g1 else 'FAIL'}"
    )
    if missing:
        lines.append(f"  missing e.g. {missing[0]}")
    rt_sum = sum(_as_float(r["rt_bound_violations"]) for r in runs)
    g2 = rt_sum == 0.0
    lines.append(
        f"G2 sum(rt_bound_violations)={rt_sum}  {'PASS' if g2 else 'FAIL'}"
    )
    late_sum = sum(_as_float(r["n_miss_full_fidelity_late"]) for r in runs)
    g3 = late_sum == 0.0
    lines.append(
        f"G3 sum(n_miss_full_fidelity_late)={late_sum}  "
        f"{'PASS' if g3 else 'FAIL FINDING'}"
    )
    crash = [sum(_as_float(r[f"crash_A{i}"]) for r in runs) for i in range(1, 5)]
    lines.append(
        f"crash totals A1..A4={crash}  (reported, not gated)"
    )
    g4 = True  # walk_runs already hard-failed on provenance
    lines.append(f"G4 provenance cross-check {n}/{n}  PASS")

    n_pos, max_signed, _pos = g5_census(runs)
    g5 = n_pos == 0
    lines.append(
        f"G5 directional identity (dev<=+{R1_TOL}) n_pos={n_pos}/{n}  "
        f"max_signed_dev={max_signed}  {'PASS' if g5 else 'FAIL'}"
    )
    feas_rows = list(feas) if feas is not None else []
    n_checked, n_g6, _g6f = g6_census(runs, feas_rows)
    g6 = n_g6 == 0
    lines.append(
        f"G6 strict identity on accepted-instance runs "
        f"n_checked={n_checked} n_fail={n_g6}  {'PASS' if g6 else 'FAIL'}"
    )
    n_pass, n_fail, max_abs, fails = r1_census(runs)
    lines.append(
        f"INFO strict-R1 census {n_pass}/{n}  n_fail={n_fail}  "
        f"max_|dev|={max_abs}  (not gated)"
    )
    lines.append(f"INFO detection-limit: {DETECTION_LIMIT}")
    if fails:
        lines.append(
            "INFO strict-R1 failure table "
            "(cell, scale, instance, ckpt, tmr, N3, dev):"
        )
        for rec in fails:
            lines.append(
                f"  {rec['cell']} s={rec['beta']} "
                f"i{rec['instance_seed']} c{rec['ckpt_seed']} "
                f"tmr={rec['task_miss_rate']} N3={rec['n3']} "
                f"dev={rec['r1_deviation']}"
            )
    all_ok = g1 and g2 and g3 and g4 and g5 and g6 and n_dup == 0
    lines.append(f"GATES {'PASS' if all_ok else 'FAIL'}")
    return lines, all_ok


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(header), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def collect(
    results_dir: Any,
    *,
    expect_n: Optional[int] = None,
    grid_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    expected: Optional[Sequence[Tuple[str, float, int, int]]] = None,
    plans: Sequence[str] = ("default",),
) -> Dict[str, Any]:
    plans = tuple(plans)
    runs = walk_runs(results_dir, plans=plans)
    if expected is not None:
        exp = list(expected)
    else:
        exp = expected_keys(plans)
        if expect_n is not None and expect_n != len(exp):
            exp = exp[:expect_n]
    if expect_n is None:
        expect_n = len(exp)
    grid_index = _load_grid_accept(grid_path) if grid_path is not None else None
    feas = build_feasibility(runs, grid_index)
    lines, ok = gate_block(runs, expect_n=expect_n, expected=exp, feas=feas)
    write_p7 = "p7" in plans
    if out_dir is not None:
        if write_p7:
            _write_csv(out_dir / "tb3_empirical_runs_p7.csv", TB3_EMP_RUNS_P7_HEADER, runs)
            _write_csv(
                out_dir / "tb3_empirical_feasibility_p7.csv",
                TB3_EMP_FEAS_P7_HEADER,
                feas,
            )
        else:
            _write_csv(out_dir / "tb3_empirical_runs.csv", TB3_EMP_RUNS_HEADER, runs)
            _write_csv(
                out_dir / "tb3_empirical_feasibility.csv", TB3_EMP_FEAS_HEADER, feas
            )
    return {
        "runs": runs,
        "feasibility": feas,
        "gate_lines": lines,
        "ok": ok,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results_dir",
        "--results-dir",
        dest="results_dir",
        required=True,
        help="Comma-separated result roots.",
    )
    ap.add_argument(
        "--plans",
        default="default",
        help="Comma-separated plan names (default, p7). Union is the expected set.",
    )
    ap.add_argument("--grid", default=None, help="Raw offline grid CSV for S0/S1 join.")
    ap.add_argument("--out", default=None, help="Directory for committed CSVs.")
    ap.add_argument("--expect-n", dest="expect_n", type=int, default=None)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = ap.parse_args(argv)
    dirs = [Path(p.strip()) for p in args.results_dir.split(",") if p.strip()]
    plans = tuple(p.strip() for p in args.plans.split(",") if p.strip())
    grid_path = Path(args.grid) if args.grid else None
    out_dir = Path(args.out) if args.out else None
    try:
        load_manifest(args.manifest)
        result = collect(
            dirs if len(dirs) > 1 else dirs[0],
            expect_n=args.expect_n,
            grid_path=grid_path,
            out_dir=out_dir,
            plans=plans,
        )
    except (CollectError, SweepConfigError) as exc:
        print(f"GATES FAIL")
        print(f"FATAL: {exc}")
        return 2
    print("=== TB3-7b empirical gates ===")
    for line in result["gate_lines"]:
        print(line)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
