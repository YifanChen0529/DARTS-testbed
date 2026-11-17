#!/usr/bin/env python3
"""Build and verify the TB-1 checkpoint manifest.

Checkpoints live under ``results/<method>_<workload>_<stress>_seed<seed>/`` as
``agent_{1..4}_ep{N}.pth``. train_darts.py writes .pth only at the FIRST and
FINAL save_interval of a run, so a directory re-used across runs accumulates
checkpoints from several eras. The paper-era epoch for a cell is the last row
of the ``darts_log.csv`` that was copied into the publication aggregate tree
``results_a1a2b1_20260521/exp_<seed//100>/<bench-dirname>/<stress>/``.

Three provenance checks per cell:
  EP-PIN           last-row episode E of D/darts_log.csv, with agent_1..4_ep{E}.pth present
  ERA              D/config.json carries "bilevel_mode" (post-Path1 code only)
  AGGREGATE-MATCH  D/darts_log.csv agrees with its aggregate-tree copy

Exits nonzero if any cell fails EP-PIN or AGGREGATE-MATCH(last-row).
"""
from __future__ import annotations

import argparse
import csv
import filecmp
import json
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.util_csv import safe_read_csv  # noqa: E402

METHOD_PREFIX = "darts_"
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
AGGREGATE_ROOT = os.path.join(_PROJECT_ROOT, "results_a1a2b1_20260521")
DEFAULT_OUT = os.path.join(_PROJECT_ROOT, "ckpt_manifest_tb1.json")

# (workload_id, stress, candidate seeds)
CELLS: List[Tuple[str, str, List[int]]] = [
    ("B1_CDR", "medium", [100, 300, 500]),
    ("B1_CDR", "heavy", [100, 300, 500]),
    ("B2_LML", "medium", [100, 200, 300]),
]

N_AGENTS = 4

# The aggregate tree holds darts_no_incentive logs for B2 but no plain-darts
# counterpart, so AGGREGATE-MATCH cannot be evaluated for those cells.
AGGREGATE_NA_BENCHES = {"B2_LML"}

CONFIG_CAVEAT = (
    "config.json in every era omits relative_deadline_ms and all poisson_* keys "
    "(they live on the Workload dataclass, not cfg). A checkpoint's stress "
    "provenance therefore rests solely on its directory name."
)


def cell_dir(bench: str, stress: str, seed: int) -> str:
    """Absolute path of the training-output directory for one cell."""
    return os.path.join(RESULTS_DIR, f"{METHOD_PREFIX}{bench}_{stress}_seed{seed}")


def aggregate_path(bench: str, stress: str, seed: int) -> str:
    """Absolute path of the publication-aggregate copy of this cell's darts_log.csv."""
    return os.path.join(
        AGGREGATE_ROOT, f"exp_{seed // 100}", bench.split("_", 1)[0], stress, "darts_log.csv"
    )


def _raw_last_episode(path: str) -> Optional[int]:
    """Last-row episode read straight off the file, ignoring schema width."""
    try:
        with open(path, newline="") as fh:
            rows = [r for r in csv.reader(fh) if r and r[0].strip()]
    except OSError:
        return None
    if len(rows) < 2:
        return None
    try:
        return int(float(rows[-1][0].strip()))
    except ValueError:
        return None


def read_last_episode(path: str) -> Dict[str, Any]:
    """Resolve the pinned episode E from column 0 of a darts_log.csv.

    safe_read_csv is the project's standard reader, but it drops rows *wider*
    than the header. These logs interleave 59/62/63-column eras, so the newest
    (widest) rows -- the ones that pin the paper-era epoch -- can vanish. The
    raw tail read is authoritative whenever the two disagree.
    """
    raw_ep = _raw_last_episode(path)
    df = safe_read_csv(path, usecols=[0])
    safe_ep: Optional[int] = None
    if df is not None and len(df) > 0:
        try:
            safe_ep = int(float(str(df.iloc[-1, 0]).strip()))
        except (ValueError, TypeError):
            safe_ep = None
    return {
        "ep": raw_ep if raw_ep is not None else safe_ep,
        "safe_read_csv_ep": safe_ep,
        "raw_tail_ep": raw_ep,
        "tail_guard_triggered": raw_ep is not None and safe_ep is not None and raw_ep != safe_ep,
    }


def checkpoints_present(directory: str, ep: int) -> Tuple[bool, List[int]]:
    """Whether agent_1..N_ep{ep}.pth all exist; also the missing agent indices."""
    missing = [
        i
        for i in range(1, N_AGENTS + 1)
        if not os.path.isfile(os.path.join(directory, f"agent_{i}_ep{ep}.pth"))
    ]
    return (not missing), missing


def _last_data_row(path: str) -> Optional[List[str]]:
    try:
        with open(path, newline="") as fh:
            rows = [r for r in csv.reader(fh) if r and r[0].strip()]
    except OSError:
        return None
    return rows[-1] if len(rows) >= 2 else None


def compare_to_aggregate(log_path: str, agg_path: str) -> Dict[str, Any]:
    """Byte and last-row comparison of a cell log against its aggregate copy.

    Last-row comparison strips per-field whitespace: at least one aggregate log
    (exp_1/B1/heavy) was written with padded fields, which is cosmetic.
    """
    if not os.path.isfile(agg_path):
        return {"byte_equal": False, "last_row_equal": False, "note": "aggregate file missing"}
    byte_equal = filecmp.cmp(log_path, agg_path, shallow=False)
    lhs, rhs = _last_data_row(log_path), _last_data_row(agg_path)
    if lhs is None or rhs is None:
        return {"byte_equal": byte_equal, "last_row_equal": False, "note": "unreadable last row"}
    equal = [c.strip() for c in lhs] == [c.strip() for c in rhs]
    note = None if byte_equal else "byte mismatch; last row equal after whitespace strip"
    return {"byte_equal": byte_equal, "last_row_equal": equal, "note": note if equal else "last row differs"}


def discover_seeds(bench: str, stress: str) -> List[int]:
    """Every seed with a training directory for this (bench, stress), ascending."""
    pattern = re.compile(rf"^{re.escape(METHOD_PREFIX)}{re.escape(bench)}_{re.escape(stress)}_seed(\d+)$")
    seeds = []
    if os.path.isdir(RESULTS_DIR):
        for name in os.listdir(RESULTS_DIR):
            m = pattern.match(name)
            if m:
                seeds.append(int(m.group(1)))
    return sorted(seeds)


def suggest_substitute(bench: str, stress: str, exclude: List[int], target_ep: Optional[int]) -> Optional[Dict[str, Any]]:
    """Next seed in this cell whose log ends at target_ep with a complete checkpoint set."""
    if target_ep is None:
        return None
    for seed in discover_seeds(bench, stress):
        if seed in exclude:
            continue
        d = cell_dir(bench, stress, seed)
        log = os.path.join(d, "darts_log.csv")
        if not os.path.isfile(log):
            continue
        info = read_last_episode(log)
        if info["ep"] != target_ep:
            continue
        complete, _ = checkpoints_present(d, target_ep)
        if complete:
            return {"seed": seed, "dir": d, "ep": target_ep}
    return None


def evaluate_cell(bench: str, stress: str, seed: int) -> Dict[str, Any]:
    """Run all three provenance checks for one (bench, stress, seed)."""
    d = cell_dir(bench, stress, seed)
    log = os.path.join(d, "darts_log.csv")
    cfg_path = os.path.join(d, "config.json")
    agg = aggregate_path(bench, stress, seed)

    row: Dict[str, Any] = {
        "key": f"{bench}|{stress}|{seed}",
        "bench": bench,
        "stress": stress,
        "seed": seed,
        "dir": d,
        "ep": None,
        "bilevel_mode": None,
        "no_incentive": None,
        "ep_pin": False,
        "era": False,
        "aggregate": None,  # True / False / None == NOT_APPLICABLE
        "notes": [],
    }

    if not os.path.isdir(d):
        row["notes"].append("cell directory missing")
        return row
    if not os.path.isfile(log):
        row["notes"].append("darts_log.csv missing")
        return row

    # (a) EP-PIN
    ep_info = read_last_episode(log)
    row["ep"] = ep_info["ep"]
    row["ep_detail"] = ep_info
    if ep_info["tail_guard_triggered"]:
        row["notes"].append(
            f"tail guard: safe_read_csv saw ep{ep_info['safe_read_csv_ep']}, "
            f"raw last row is ep{ep_info['raw_tail_ep']} (wide rows dropped); using raw"
        )
    if row["ep"] is None:
        row["notes"].append("could not resolve last-row episode")
    else:
        complete, missing = checkpoints_present(d, row["ep"])
        row["ep_pin"] = complete
        if not complete:
            row["notes"].append(f"missing agent checkpoints at ep{row['ep']}: {missing}")

    # (b) ERA
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path) as fh:
                cfg = json.load(fh)
        except (OSError, json.JSONDecodeError):
            cfg = None
            row["notes"].append("config.json unreadable")
        if cfg is not None:
            row["era"] = "bilevel_mode" in cfg
            row["bilevel_mode"] = cfg.get("bilevel_mode")
            row["no_incentive"] = cfg.get("no_incentive")
            if not row["era"]:
                row["notes"].append("config.json lacks bilevel_mode (pre-Path1 era)")
    else:
        row["notes"].append("config.json missing")

    # (c) AGGREGATE-MATCH
    if bench in AGGREGATE_NA_BENCHES:
        row["aggregate"] = None
        row["provenance"] = {
            "byte_equal": None,
            "last_row_equal": None,
            "aggregate_path": agg,
            "status": "NOT_APPLICABLE",
            "note": "aggregate tree carries only darts_no_incentive logs for B2; "
                    "no plain-darts counterpart exists. Cell passes on EP-PIN + ERA alone.",
        }
    else:
        cmp_res = compare_to_aggregate(log, agg)
        row["aggregate"] = bool(cmp_res["last_row_equal"])
        row["provenance"] = {
            "byte_equal": bool(cmp_res["byte_equal"]),
            "last_row_equal": bool(cmp_res["last_row_equal"]),
            "aggregate_path": agg,
            "status": "OK" if cmp_res["last_row_equal"] else "MISMATCH",
            "note": cmp_res.get("note"),
        }
        if cmp_res.get("note"):
            row["notes"].append(cmp_res["note"])
    return row


def cell_failed(row: Dict[str, Any]) -> bool:
    """A cell fails on EP-PIN or on AGGREGATE-MATCH(last-row); ERA is recorded only."""
    return (not row["ep_pin"]) or (row["aggregate"] is False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT, help="Manifest output path (default: repo-root ckpt_manifest_tb1.json).")
    args = ap.parse_args()

    rows = [evaluate_cell(b, s, sd) for b, s, seeds in CELLS for sd in seeds]

    def mark(v: Optional[bool]) -> str:
        if v is None:
            return "N/A"
        return "PASS" if v else "FAIL"

    print("=" * 108)
    print("TB-1 CHECKPOINT MANIFEST — PROVENANCE SUMMARY")
    print("=" * 108)
    print(f"{'cell key':<26} {'ep':>6}  {'EP-PIN':<7} {'ERA':<7} {'AGG':<7} {'byte':<6} {'bilevel_mode':<12} {'no_inc':<7} result")
    print("-" * 108)
    for r in rows:
        prov = r.get("provenance", {})
        be = prov.get("byte_equal")
        print(
            f"{r['key']:<26} {str(r['ep']):>6}  {mark(r['ep_pin']):<7} {mark(r['era']):<7} "
            f"{mark(r['aggregate']):<7} {('N/A' if be is None else str(be)):<6} "
            f"{str(r['bilevel_mode']):<12} {str(r['no_incentive']):<7} "
            f"{'FAIL' if cell_failed(r) else 'ok'}"
        )
    print("-" * 108)

    notes = [(r["key"], n) for r in rows for n in r["notes"]]
    if notes:
        print("Notes:")
        for key, n in notes:
            print(f"  [{key}] {n}")
        print("-" * 108)

    eps = Counter(r["ep"] for r in rows if r["ep"] is not None)
    print(f"Pinned epochs: {dict(eps)}")
    print(f"Caveat: {CONFIG_CAVEAT}")

    failures = [r for r in rows if cell_failed(r)]
    if failures:
        print("-" * 108)
        print("SUBSTITUTE SUGGESTIONS (not auto-applied):")
        for r in failures:
            cell_rows = [x for x in rows if x["bench"] == r["bench"] and x["stress"] == r["stress"]]
            healthy = Counter(x["ep"] for x in cell_rows if not cell_failed(x) and x["ep"] is not None)
            target = healthy.most_common(1)[0][0] if healthy else r["ep"]
            sub = suggest_substitute(r["bench"], r["stress"], [x["seed"] for x in cell_rows], target)
            if sub:
                print(f"  [{r['key']}] -> seed {sub['seed']} (log ends at ep{sub['ep']}, complete set): {sub['dir']}")
            else:
                print(f"  [{r['key']}] -> no substitute seed found ending at ep{target}")

    manifest: Dict[str, Any] = {
        "_meta": {
            "description": "TB-1 checkpoint manifest: paper-era DARTS checkpoints pinned by last-row episode.",
            "generated_by": "scripts/build_tb1_manifest.py",
            "method_prefix": METHOD_PREFIX,
            "n_agents": N_AGENTS,
            "results_dir": RESULTS_DIR,
            "aggregate_root": AGGREGATE_ROOT,
            "caveats": [
                CONFIG_CAVEAT,
                "AGGREGATE-MATCH is NOT_APPLICABLE for B2_LML: the aggregate tree carries only "
                "darts_no_incentive logs for B2, so those cells pass on EP-PIN + ERA alone.",
                "safe_read_csv drops rows wider than the header; these logs interleave 59/62/63-column "
                "eras, so the pinned epoch is taken from a raw tail read whenever the two disagree. "
                "Without that guard the B1 cells pin stale ep5000 checkpoints instead of ep2500.",
            ],
        }
    }
    for r in rows:
        manifest[r["key"]] = {
            "dir": r["dir"],
            "ep": r["ep"],
            "bilevel_mode": r["bilevel_mode"],
            "no_incentive": r["no_incentive"],
            "provenance": r.get("provenance", {}),
        }

    with open(args.out, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print("-" * 108)
    print(f"Wrote {args.out} ({len(rows)} cells, {len(failures)} failing)")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
