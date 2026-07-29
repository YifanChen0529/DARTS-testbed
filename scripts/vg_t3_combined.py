#!/usr/bin/env python3
"""Combined §V.G T3 table: R̄^off_G, R_measured_G, and their ratio.

Outputs Markdown to stdout. Default δ_auc = 30 ms (tight WCET); override via CLI.
Reuses compute_r_off_g() from scripts/r_off_g.py.

Run:
    python3 scripts/vg_t3_combined.py --delta_auc_ms 30
"""
from __future__ import annotations
import argparse, glob, sys, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from util_csv import safe_read_csv

# Dynamically import r_off_g.py (it has CLI main(); we only want compute_r_off_g)
spec = importlib.util.spec_from_file_location("r_off_g_mod", SCRIPTS_DIR / "r_off_g.py")
r_off_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r_off_mod)

NEW_SWEEP   = REPO_ROOT / "results_a1a2b1_20260521"
OLD_RESULTS = REPO_ROOT / "results"
CONV = 1000  # convergence window

CELLS = [
    ('B1_CDR', 'light',  ['darts', 'darts_nocert', 'darts_no_incentive', 'cea_edf', 'sadcher']),
    ('B1_CDR', 'medium', ['darts', 'darts_nocert', 'darts_no_incentive', 'cea_edf', 'sadcher']),
    ('B1_CDR', 'heavy',  ['darts', 'darts_nocert', 'darts_no_incentive', 'cea_edf', 'sadcher']),
    ('B2_LML', 'medium', ['darts', 'cea_edf', 'sadcher']),
    ('B3_IQI', 'medium', ['darts', 'cea_edf', 'sadcher']),
    ('B4_DEM', 'medium', ['darts', 'cea_edf', 'sadcher']),
]

def measured_r_g(workload_id: str, stress: str, method: str):
    """Mean ± std of task_time over convergence window, across seeds."""
    if workload_id == 'B1_CDR':
        pattern = f"{NEW_SWEEP}/exp_*/B1/{stress}/{method}_log.csv"
    else:
        # Old workspace dirs at results/___seed*/
        pattern = f"{OLD_RESULTS}/{method}_{workload_id}_{stress}_seed*/{method}_log.csv"
    cells = []
    for p in sorted(glob.glob(pattern)):
        df = safe_read_csv(p)
        if df is None or 'task_time' not in df.columns:
            continue
        tail = df.iloc[-CONV:] if len(df) >= CONV else df
        s = pd.to_numeric(tail['task_time'], errors='coerce').dropna()
        if len(s) > 0:
            cells.append(s.mean())
    if not cells:
        return None, None, 0
    return float(np.mean(cells)), float(np.std(cells)), len(cells)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--delta_auc_ms', type=float, default=30.0,
                    help='Calibrated subround WCET. Default 30 ms = tight (just above max single-subround obs).')
    ap.add_argument('--H_ms',         type=float, default=1000.0)
    ap.add_argument('--j_bar_ms',     type=float, default=0.0)
    ap.add_argument('--ui_max_mps',   type=float, default=0.22)
    ap.add_argument('--h_comm_ms',    type=float, default=50.0)
    ap.add_argument('--start_x',      type=float, default=0.0)
    ap.add_argument('--start_y',      type=float, default=0.0)
    ap.add_argument(
        '--new_sweep_dir',
        type=str,
        default=str(REPO_ROOT / "results_a1a2b1_20260521"),
        help="Root for NEW_SWEEP globs (default: <repo_root>/results_a1a2b1_20260521)",
    )
    ap.add_argument(
        '--old_results_dir',
        type=str,
        default=str(REPO_ROOT / "results"),
        help="Root for OLD_RESULTS globs (default: <repo_root>/results)",
    )
    args = ap.parse_args()
    global NEW_SWEEP, OLD_RESULTS
    NEW_SWEEP = Path(args.new_sweep_dir).expanduser().resolve()
    OLD_RESULTS = Path(args.old_results_dir).expanduser().resolve()
    print(f"[vg_t3_combined] NEW_SWEEP={NEW_SWEEP}", file=sys.stderr)
    print(f"[vg_t3_combined] OLD_RESULTS={OLD_RESULTS}", file=sys.stderr)

    print("# §V.G T3 — Combined R̄^off_G vs R_measured_G")
    print()
    print(f"- δ_auc = {args.delta_auc_ms} ms")
    print(f"- H = {args.H_ms} ms, J̄ = {args.j_bar_ms} ms")
    print(f"- u_i,max = {args.ui_max_mps} m/s, h = {args.h_comm_ms} ms")
    print(f"- start_pos = ({args.start_x}, {args.start_y}) m (worst-case agent start)")
    print()
    print("| Benchmark | Stress | Method | R̄^off_G (ms) | R_measured_G (ms) ± std | n_seeds | Ratio (off / meas) |")
    print("|---|---|---|---|---|---|---|")

    start_pos = (args.start_x, args.start_y)
    for bench, stress, methods in CELLS:
        r = r_off_mod.compute_r_off_g(
            bench, stress,
            args.delta_auc_ms, args.H_ms, args.j_bar_ms,
            args.ui_max_mps, args.h_comm_ms, start_pos,
        )
        r_off = r['R_off_G_ms']
        for method in methods:
            mean_r, std_r, n = measured_r_g(bench, stress, method)
            if mean_r is None:
                print(f"| {bench} | {stress} | {method} | {r_off:.0f} | — | 0 | — |")
                continue
            ratio = r_off / mean_r
            print(f"| {bench} | {stress} | {method} | {r_off:.0f} | {mean_r:.0f} ± {std_r:.0f} | {n} | **{ratio:.2f}×** |")

if __name__ == '__main__':
    main()
