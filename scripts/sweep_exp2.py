#!/usr/bin/env python3
"""EXP-2 Theorem 2(a) empirical sweep runner.

Bare python3 (do not wrap this file in conda run). Inner evals are conda-wrapped.
Reads ``docs/exp2/exp2_config_table.json`` and launches one frozen-policy
``alg/train_darts.py --eval_only`` per row. Shared helpers come from
``scripts.sweep_tb1`` / ``scripts.sweep_tb3`` rather than being retyped.

  python3 scripts/sweep_exp2.py --dry-run
  python3 scripts/sweep_exp2.py --results_dir results_exp2_dryrun --only exp2_B4_DEM_d085
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import scripts.sweep_tb1 as _tb1  # noqa: E402
from scripts.sweep_tb1 import (  # noqa: E402
    CONDA_ENV,
    LD_LIBRARY_PATH,
    SweepConfigError,
    append_progress as _tb1_append_progress,
    notify as _tb1_notify,
)

DEFAULT_TABLE = os.path.join(_PROJECT_ROOT, "docs", "exp2", "exp2_config_table.json")
PROGRESS_LOG_NAME = "sweep_exp2_progress.log"
PANE = "EXP2"
RUN_TIMEOUT_S = 1800
EVAL_EPISODES = 1  # --eval_episodes already exists on alg/train_darts.py (default None → cfg 10)


def utc_results_root() -> str:
    return "results_exp2_" + datetime.now(timezone.utc).strftime("%Y%m%d")


def load_table(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        blob = json.load(f)
    rows = blob.get("configs") if isinstance(blob, dict) else blob
    if not isinstance(rows, list) or not rows:
        raise SweepConfigError(f"config table {path} has no configs")
    ids = [r["config_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SweepConfigError("duplicate config_id in table")
    return rows


def append_progress(results_dir: str, line: str) -> None:
    old = _tb1.PROGRESS_LOG_NAME
    _tb1.PROGRESS_LOG_NAME = PROGRESS_LOG_NAME
    try:
        _tb1_append_progress(results_dir, line)
    finally:
        _tb1.PROGRESS_LOG_NAME = old


def notify(message: str, priority: str = "default", pane: str = PANE) -> None:
    _tb1_notify(pane, message, priority=priority)


def build_eval_command(row: Dict[str, Any], results_dir: str) -> List[str]:
    """One conda-wrapped train_darts.py --eval_only invocation for a table row."""
    alloc = row["alloc_map_file"]
    if not os.path.isabs(alloc):
        alloc = os.path.join(_PROJECT_ROOT, alloc)
    inner: List[str] = [
        "python3", "-u", "alg/train_darts.py",
        "--eval_only",
        "--load_ckpt", str(row["ckpt_dir"]),
        "--load_ckpt_ep", str(int(row["ckpt_ep"])),
        "--workload", str(row["workload"]),
        "--stress", "medium",
        "--release_mode", "periodic",
        "--release_period_ms", str(int(row["P_ms"])),
        "--num_releases", str(int(row["L"])),
        "--relative_deadline_ms", str(int(row["D_v_ms"])),
        "--alloc_map", alloc,
        "--per_period_energy", str(row["per_period_energy"]),
        "--max_episode_steps", str(int(row["max_episode_steps"])),
        "--global_deadline_ms", str(int(row["global_deadline_ms"])),
        "--eval_seed_base", str(int(row["eval_seed_base"])),
        "--eval_episodes", str(int(row.get("eval_episodes", EVAL_EPISODES))),
        "--results_dir", results_dir,
        "--exp_name", str(row["config_id"]),
    ]
    return [
        "conda", "run", "--no-capture-output", "-n", CONDA_ENV,
        "env",
        f"LD_LIBRARY_PATH={LD_LIBRARY_PATH}",
        f"PYTHONPATH={_PROJECT_ROOT}",
        *inner,
    ]


def plan_runs(
    rows: Sequence[Dict[str, Any]],
    results_dir: str,
    only: Optional[str] = None,
) -> List[Dict[str, Any]]:
    ordered = list(rows)  # table order is the deterministic order
    if only:
        ordered = [r for r in ordered if r["config_id"] == only]
        if not ordered:
            known = [r["config_id"] for r in rows]
            raise SweepConfigError(
                f"--only {only!r} is not a config_id. Known: {known}"
            )
    runs = []
    for row in ordered:
        cmd = build_eval_command(row, results_dir)
        runs.append({
            "config_id": row["config_id"],
            "exp_name": row["config_id"],
            "cmd": cmd,
            "row": row,
        })
    return runs


def assert_fresh_results_root(results_dir: str) -> None:
    """Refuse a non-empty results root (no --resume)."""
    if not os.path.isdir(results_dir):
        return
    leftover = sorted(os.listdir(results_dir))
    if leftover:
        raise SweepConfigError(
            "REFUSAL: results root is not empty (no --resume; move or delete "
            f"it first): {results_dir}\n  " + "\n  ".join(leftover[:20])
        )


def _run_one(cmd: Sequence[str]) -> Tuple[str, float]:
    t0 = time.time()
    try:
        proc = subprocess.run(list(cmd), cwd=_PROJECT_ROOT, timeout=RUN_TIMEOUT_S)
        tag = str(proc.returncode)
    except subprocess.TimeoutExpired:
        tag = "TIMEOUT"
    elapsed = time.time() - t0
    return tag, elapsed


def _fmt_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}:{m:02d}"


def execute_runs(runs: Sequence[Dict[str, Any]], results_dir: str) -> int:
    os.makedirs(results_dir, exist_ok=True)
    total = len(runs)
    start_msg = f"{total} runs, timeout={RUN_TIMEOUT_S}s"
    append_progress(
        results_dir,
        f"# pane={PANE} total={total} start={time.strftime('%F %T')} {start_msg}",
    )
    notify(start_msg, pane=PANE)

    n_fail = 0
    sweep_t0 = time.time()
    for i, r in enumerate(runs, 1):
        print(f"\n===== [{i}/{total}] {r['config_id']} =====", flush=True)
        print(" ".join(r["cmd"]), flush=True)
        tag, elapsed = _run_one(r["cmd"])
        ok = tag == "0"
        if not ok:
            n_fail += 1
            notify(
                f"HIGH: {r['config_id']} rc={tag} ({i}/{total})",
                priority="high",
                pane=PANE,
            )
        append_progress(
            results_dir,
            f"{time.strftime('%F %T')} {r['config_id']} rc={tag} "
            f"wall_s={elapsed:.3f} ({i}/{total})",
        )
        notify(
            f"{r['config_id']} rc={tag} wall_s={elapsed:.1f} ({i}/{total})",
            pane=PANE,
        )
        if not ok:
            sys.stderr.write(f"[sweep_exp2] {r['config_id']} rc={tag}; continuing.\n")

    n_ok = total - n_fail
    elapsed = time.time() - sweep_t0
    summary = f"EXP2 sweep done: {n_ok}/{total} ok, {n_fail} failed, elapsed {_fmt_hms(elapsed)}"
    append_progress(results_dir, f"# {summary}")
    notify(summary, priority=("high" if n_fail else "default"), pane=PANE)
    print(f"\n[sweep_exp2] {summary}")
    return 1 if n_fail else 0


def _fatal(exc: BaseException, results_dir: Optional[str]) -> int:
    first = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    msg = f"REFUSAL: {first}"
    sys.stderr.write(f"[sweep_exp2] FATAL: {exc}\n")
    if results_dir:
        try:
            append_progress(results_dir, f"# {msg}")
        except OSError:
            pass
    notify(msg, priority="high")
    return 2


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--table",
        default=DEFAULT_TABLE,
        help="Path to exp2_config_table.json.",
    )
    ap.add_argument(
        "--results_dir",
        "--results-dir",
        dest="results_dir",
        default=None,
        help="Output root (default results_exp2_<UTC YYYYMMDD>).",
    )
    ap.add_argument(
        "--dry-run",
        "--dry_run",
        dest="dry_run",
        action="store_true",
        default=False,
        help="Print every command and exit 0 without creating result dirs.",
    )
    ap.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run a single config_id (e.g. exp2_B4_DEM_d085).",
    )
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    results_dir = args.results_dir or utc_results_root()
    try:
        rows = load_table(args.table)
        runs = plan_runs(rows, results_dir, only=args.only)
        if args.dry_run:
            print(
                f"# EXP2 plan: runs={len(runs)} results_dir={results_dir} "
                f"eval_episodes={EVAL_EPISODES} table={args.table}"
            )
            print("# order: docs/exp2/exp2_config_table.json configs[] order")
            for i, r in enumerate(runs, 1):
                print(f"\n# [{i}/{len(runs)}] {r['config_id']}")
                print(" ".join(r["cmd"]))
            return 0
        assert_fresh_results_root(results_dir)
    except SweepConfigError as exc:
        return _fatal(exc, results_dir if not args.dry_run else None)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(traceback.format_exc())
        return _fatal(exc, results_dir if not args.dry_run else None)

    try:
        return execute_runs(runs, results_dir)
    except SweepConfigError as exc:
        return _fatal(exc, results_dir)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(traceback.format_exc())
        return _fatal(exc, results_dir)


if __name__ == "__main__":
    sys.exit(main())
