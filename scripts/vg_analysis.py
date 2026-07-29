#!/usr/bin/env python3
"""Compute V.G T1, V.G T3 R_measured, and Table II / D1 breakdown.

All numbers come from existing CSVs:
- NEW sweep (B1 × {light, medium, heavy, energy_b04, energy_b03, energy_b02, medium_n2, medium_n6, medium_n8})
  at results_a1a2b1_20260521/
- OLD workspace dirs (B2/B3/B4 medium, pre-bugfix codebase) at results/
"""
from __future__ import annotations
import argparse
import sys, glob, math
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from util_csv import safe_read_csv

REPO_ROOT = Path(__file__).resolve().parent.parent
_parser = argparse.ArgumentParser(description="V.G T1 / T3 R_measured / Table II–D1 breakdown.")
_parser.add_argument(
    "--new_sweep_dir",
    type=str,
    default=str(REPO_ROOT / "results_a1a2b1_20260521"),
    help="Root for NEW_SWEEP globs (default: <repo_root>/results_a1a2b1_20260521)",
)
_parser.add_argument(
    "--old_results_dir",
    type=str,
    default=str(REPO_ROOT / "results"),
    help="Root for OLD_RESULTS globs (default: <repo_root>/results)",
)
_args = _parser.parse_args()
NEW_SWEEP = Path(_args.new_sweep_dir).expanduser().resolve()
OLD_RESULTS = Path(_args.old_results_dir).expanduser().resolve()
print(f"[vg_analysis] NEW_SWEEP={NEW_SWEEP}", file=sys.stderr)
print(f"[vg_analysis] OLD_RESULTS={OLD_RESULTS}", file=sys.stderr)

CONV = 1000  # convergence window (last N episodes)

A_BAR = {'B1_CDR': 2, 'B2_LML': 3, 'B3_IQI': 1, 'B4_DEM': 4}


def col_tail(p, col, window=CONV):
    df = safe_read_csv(p)
    if df is None or col not in df.columns:
        return None
    tail = df.iloc[-window:] if len(df) >= window else df
    return pd.to_numeric(tail[col], errors='coerce').dropna()


def cell_mean(p, col, window=CONV):
    s = col_tail(p, col, window)
    return s.mean() if s is not None and len(s) > 0 else None


def cell_max(p, col):
    df = safe_read_csv(p)
    if df is None or col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors='coerce').dropna()
    return s.max() if len(s) > 0 else None


# ============ §V.G T1 ============
print("# §V.G T1 — Auction-Latency Bound Tightness")
print()

# (1) Global δ_obs,max across ALL cells with max_t_auc_ms column
all_max = []
patterns_for_t_auc = [
    f"{NEW_SWEEP}/exp_*/B1/*/darts_log.csv",
    f"{NEW_SWEEP}/exp_*/B1/*/darts_nocert_log.csv",
    f"{NEW_SWEEP}/exp_*/B1/*/cea_edf_log.csv",
    f"{NEW_SWEEP}/exp_*/B1/*/sadcher_log.csv",
    f"{OLD_RESULTS}/darts_B2_LML_*/darts_log.csv",
    f"{OLD_RESULTS}/darts_B3_IQI_*/darts_log.csv",
    f"{OLD_RESULTS}/darts_B4_DEM_*/darts_log.csv",
]
n_cells_with_t_auc = 0
for pat in patterns_for_t_auc:
    for p in glob.glob(pat):
        m = cell_max(p, 'max_t_auc_ms')
        if m is not None:
            all_max.append(m)
            n_cells_with_t_auc += 1

if not all_max:
    print("⚠ ERROR: no max_t_auc_ms data found anywhere. Check that column exists in CSVs.")
    sys.exit(1)

delta_obs_max = max(all_max)
delta_auc_suggested = math.ceil(delta_obs_max / 5) * 5
print(f"- **δ_obs,max** (global max of `max_t_auc_ms` across {n_cells_with_t_auc} cells): **{delta_obs_max:.2f} ms**")
print(f"- **Suggested δ_auc** (next multiple of 5 above δ_obs,max): **{delta_auc_suggested} ms**")
print()
print("  *Adjust upward (e.g., to 1.2× δ_obs,max) for a more honest WCET margin; paper §V.G T1 expects δ_auc/δ_obs ≈ 1.2-2.5.*")
print()

# (2) Per-frontier P99 (|A_k|=1 from B2; |A_k|=4 from B4)
def p99_from(pattern):
    vals = []
    for p in glob.glob(pattern):
        s = col_tail(p, 'max_t_auc_ms', window=500)
        if s is not None:
            vals.extend(s.tolist())
    return (np.percentile(vals, 99), len(vals)) if vals else (None, 0)

p99_ak1, n_ak1 = p99_from(f"{OLD_RESULTS}/darts_B2_LML_medium_seed*/darts_log.csv")
p99_ak4, n_ak4 = p99_from(f"{OLD_RESULTS}/darts_B4_DEM_medium_seed*/darts_log.csv")

print(f"- **|A_k|=1 P99** (B2/medium DARTS, n={n_ak1} samples): **{p99_ak1:.2f} ms**" if p99_ak1 else "- |A_k|=1: NO DATA")
print(f"- **|A_k|=4 P99** (B4/medium DARTS, n={n_ak4} samples): **{p99_ak4:.2f} ms**" if p99_ak4 else "- |A_k|=4: NO DATA")
print()

# (3) Per-benchmark T̄_auc and bound-to-P99 ratio
print("### Bound-to-observation ratio per benchmark (using suggested δ_auc)")
print()
print("| Benchmark | Ā | T̄_auc = Ā·δ_auc | P99 observed | T̄_auc / P99 |")
print("|---|---|---|---|---|")
for bench, A in A_BAR.items():
    T_bar = A * delta_auc_suggested
    if bench == 'B2_LML':
        p99 = p99_ak1
    elif bench == 'B4_DEM':
        p99 = p99_ak4
    elif bench == 'B1_CDR':
        # |A_k|=2 from B1 new sweep
        vals = []
        for p in glob.glob(f"{NEW_SWEEP}/exp_*/B1/medium/darts_log.csv"):
            s = col_tail(p, 'max_t_auc_ms', window=500)
            if s is not None:
                vals.extend(s.tolist())
        p99 = np.percentile(vals, 99) if vals else None
    else:
        # |A_k|=1 from B3 (linear chain)
        vals = []
        for p in glob.glob(f"{OLD_RESULTS}/darts_B3_IQI_*/darts_log.csv"):
            s = col_tail(p, 'max_t_auc_ms', window=500)
            if s is not None:
                vals.extend(s.tolist())
        p99 = np.percentile(vals, 99) if vals else None
    if p99 is not None:
        ratio = T_bar / p99
        print(f"| {bench} | {A} | {T_bar:.0f} ms | {p99:.2f} ms | **{ratio:.2f}×** |")
    else:
        print(f"| {bench} | {A} | {T_bar:.0f} ms | — | — |")

# (4) Agent-scaling latency table (B1 priority — paper §V-? for advisor B1)
print()
print("### B1 priority — auction latency vs. n (linear-growth check)")
print()
print("| n | Stress dir | n_seeds | P99 (ms) | mean (ms) |")
print("|---|---|---|---|---|")
for n_agents, stress_dir in [(2, 'medium_n2'), (4, 'medium'), (6, 'medium_n6'), (8, 'medium_n8')]:
    vals = []
    n_seeds = 0
    for p in glob.glob(f"{NEW_SWEEP}/exp_*/B1/{stress_dir}/darts_log.csv"):
        s = col_tail(p, 'max_t_auc_ms', window=500)
        if s is not None and len(s) > 0:
            vals.extend(s.tolist())
            n_seeds += 1
    if vals:
        print(f"| {n_agents} | {stress_dir} | {n_seeds} | {np.percentile(vals, 99):.2f} | {np.mean(vals):.2f} |")
    else:
        print(f"| {n_agents} | {stress_dir} | 0 | — | — |")


# ============ §V.G T3 R_measured ============
print()
print("# §V.G T3 — R_measured_G per (benchmark, stress, method)")
print()
print("Convergence window: last {} episodes. Uses `task_time` column.".format(CONV))
print()
print("| Benchmark | Stress | Method | n_seeds | R_measured (ms) | ± std |")
print("|---|---|---|---|---|---|")

# B1 new sweep
for stress in ['light', 'medium', 'heavy']:
    for method in ['darts', 'darts_nocert', 'darts_no_incentive', 'cea_edf', 'sadcher']:
        cells = []
        for p in sorted(glob.glob(f"{NEW_SWEEP}/exp_*/B1/{stress}/{method}_log.csv")):
            v = cell_mean(p, 'task_time')
            if v is not None and not np.isnan(v):
                cells.append(v)
        if cells:
            print(f"| B1 | {stress} | {method} | {len(cells)} | {np.mean(cells):.1f} | {np.std(cells):.1f} |")

# B2-B4 old sweep
for bench in ['B2_LML', 'B3_IQI', 'B4_DEM']:
    for method in ['darts', 'darts_nocert', 'cea_edf', 'sadcher']:
        cells = []
        for p in sorted(glob.glob(f"{OLD_RESULTS}/{method}_{bench}_medium_seed*/{method}_log.csv")):
            v = cell_mean(p, 'task_time')
            if v is not None and not np.isnan(v):
                cells.append(v)
        if cells:
            print(f"| {bench} | medium | {method} | {len(cells)} | {np.mean(cells):.1f} | {np.std(cells):.1f} |")


# Column fallback for cells with legacy (62-col) schema.
# When new column is absent, fall back to legacy; if no fallback, skip.
BREAKDOWN_COL_FALLBACKS = {
    'n1':  ['n_miss_n1_drop',                  'n_nondegradable_n1'],
    'n2':  ['n_miss_n2_fallback_insufficient', 'n_nondegradable_n2'],
    'n3':  ['n_miss_n3_unreached'],
    'ffl': ['n_miss_full_fidelity_late'],
}


def extract_breakdown_value(df, key):
    """Return Series for the breakdown column, trying each candidate in order.
    Returns (Series, schema_tag) where schema_tag is 'new', 'legacy', or 'missing'."""
    for i, col in enumerate(BREAKDOWN_COL_FALLBACKS[key]):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(s) > 0:
                tag = 'new' if i == 0 else 'legacy'
                return s, tag
    return None, 'missing'


# ============ Table II / D1 N1/N2/N3 breakdown ============
print()
print("# Table II / D1 — Structural Miss Breakdown")
print()
print("Convergence-window mean (last {} episodes) of per-episode counts.".format(CONV))
print("Higher is worse for all four columns. SADCHER omitted: it does not classify misses.")
print()
print("| Benchmark | Stress | Method | task_miss% | N1 drop | N2 fallback-insuf. | N3 unreached | FF late | Schema |")
print("|---|---|---|---|---|---|---|---|---|")

methods_breakdown = ['darts', 'darts_nocert', 'darts_no_incentive', 'cea_edf']

for stress in ['light', 'medium', 'heavy', 'energy_b04', 'energy_b03', 'energy_b02']:
    for method in methods_breakdown:
        cells = {'miss': [], 'n1': [], 'n2': [], 'n3': [], 'ffl': []}
        for p in sorted(glob.glob(f"{NEW_SWEEP}/exp_*/B1/{stress}/{method}_log.csv")):
            df = safe_read_csv(p)
            if df is None:
                continue
            tail = df.iloc[-CONV:] if len(df) >= CONV else df
            cells['miss'].append(pd.to_numeric(tail.get('task_miss_rate', pd.Series()), errors='coerce').dropna().mean())
            for k in ('n1', 'n2', 'n3', 'ffl'):
                tag = 'missing'
                for col in BREAKDOWN_COL_FALLBACKS[k]:
                    if col in tail.columns:
                        s = pd.to_numeric(tail[col], errors='coerce').dropna()
                        if len(s) > 0:
                            cells[k].append(s.mean())
                            tag = 'new' if col == BREAKDOWN_COL_FALLBACKS[k][0] else 'legacy'
                            break
                cells.setdefault(f'{k}_schema', []).append(tag)
        if cells['miss'] and not all(np.isnan(cells['miss'])):
            miss_pct = np.nanmean(cells['miss']) * 100
            row_cells = [
                f"{miss_pct:.1f}",
                f"{np.nanmean(cells['n1']):.2f}" if cells['n1'] else "—",
                f"{np.nanmean(cells['n2']):.2f}" if cells['n2'] else "—",
                f"{np.nanmean(cells['n3']):.2f}" if cells['n3'] else "—",
                f"{np.nanmean(cells['ffl']):.2f}" if cells['ffl'] else "—",
            ]
            n1_tag = cells.get('n1_schema', ['missing'])
            majority_schema = max(set(n1_tag), key=n1_tag.count) if n1_tag else 'missing'
            print(f"| B1 | {stress} | {method} | {' | '.join(row_cells)} | {majority_schema} |")

# B2-B4 medium
for bench in ['B2_LML', 'B3_IQI', 'B4_DEM']:
    for method in methods_breakdown:
        if method == 'darts_no_incentive' and bench != 'B2_LML':
            continue   # only B2 had no_incentive in old sweep
        if method == 'darts_nocert' and bench in ['B3_IQI', 'B4_DEM']:
            continue
        cells = {'miss': [], 'n1': [], 'n2': [], 'n3': [], 'ffl': []}
        for p in sorted(glob.glob(f"{OLD_RESULTS}/{method}_{bench}_medium_seed*/{method}_log.csv")):
            df = safe_read_csv(p)
            if df is None:
                continue
            tail = df.iloc[-CONV:] if len(df) >= CONV else df
            cells['miss'].append(pd.to_numeric(tail.get('task_miss_rate', pd.Series()), errors='coerce').dropna().mean())
            for k in ('n1', 'n2', 'n3', 'ffl'):
                tag = 'missing'
                for col in BREAKDOWN_COL_FALLBACKS[k]:
                    if col in tail.columns:
                        s = pd.to_numeric(tail[col], errors='coerce').dropna()
                        if len(s) > 0:
                            cells[k].append(s.mean())
                            tag = 'new' if col == BREAKDOWN_COL_FALLBACKS[k][0] else 'legacy'
                            break
                cells.setdefault(f'{k}_schema', []).append(tag)
        if cells['miss'] and not all(np.isnan(cells['miss'])):
            miss_pct = np.nanmean(cells['miss']) * 100
            row_cells = [
                f"{miss_pct:.1f}",
                f"{np.nanmean(cells['n1']):.2f}" if cells['n1'] else "—",
                f"{np.nanmean(cells['n2']):.2f}" if cells['n2'] else "—",
                f"{np.nanmean(cells['n3']):.2f}" if cells['n3'] else "—",
                f"{np.nanmean(cells['ffl']):.2f}" if cells['ffl'] else "—",
            ]
            n1_tag = cells.get('n1_schema', ['missing'])
            majority_schema = max(set(n1_tag), key=n1_tag.count) if n1_tag else 'missing'
            print(f"| {bench} | medium | {method} | {' | '.join(row_cells)} | {majority_schema} |")

print()
print("=== END vg_analysis.py ===")
