#!/usr/bin/env python3
"""Aggregate DARTS sweep results into method × bench × stress summary."""
import argparse
import os, re, glob
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from util_csv import safe_read_csv
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(_script_dir) if os.path.basename(_script_dir) == "scripts" else _script_dir

parser = argparse.ArgumentParser(description="Aggregate DARTS sweep results into method × bench × stress summary.")
parser.add_argument("--results_dir", type=str, default=os.path.join(repo_root, "results"),
                    help="Root directory containing exp_*/B*/*/*_log.csv (default: <repo_root>/results)")
parser.add_argument("--out_dir", type=str, default=repo_root,
                    help="Directory for sweep_summary_*.csv outputs (default: <repo_root>)")
args = parser.parse_args()
results_dir = args.results_dir
out_dir = args.out_dir

# Mtime gate (LAUNCH_REF/LAUNCH_T) removed: script now targets curated directories (AE plan T5); stale-file exclusion is the curation step's responsibility.

records = []
for csv in glob.glob(f"{results_dir}/exp_*/B*/*/*_log.csv"):
    m = re.match(r".*/exp_(\d+)/(B\d+)/(\w+)/(\w+)_log\.csv$", csv)
    if not m:
        continue
    exp_idx, bench, stress, method = m.groups()
    seed = int(exp_idx) * 100
    try:
        df = safe_read_csv(csv)
        if df is None:
            continue
        df.columns = df.columns.str.strip()
        if len(df) == 0:
            continue
        last = df.iloc[-1]
    except Exception as e:
        print(f"[skip] {csv}: {e}")
        continue

    def f(col):
        try:
            v = last[col]
            return float(v) if pd.notna(v) and str(v).replace(".", "").replace("-", "").isdigit() else None
        except (KeyError, ValueError, TypeError):
            try:
                return float(last[col])
            except Exception:
                return None

    records.append({
        "method": method, "benchmark": bench, "stress": stress, "seed": seed,
        "last_ep": int(last.get("episode", -1)),
        "task_miss_rate": f("task_miss_rate"),
        "rt_bound_violations": f("rt_bound_violations"),
        "max_rt_ms": f("max_rt_ms"),
        "A1_crash": f("A1_crash_rate"), "A2_crash": f("A2_crash_rate"),
        "A3_crash": f("A3_crash_rate"), "A4_crash": f("A4_crash_rate"),
    })

df = pd.DataFrame(records)
print(f"\n{'='*65}\nTOTAL CELLS WITH DATA: {len(df)}\n{'='*65}")
print(f"  reached ep=5000: {(df['last_ep']==5000).sum()}")
print(f"  partial:         {(df['last_ep']<5000).sum()}")

# Aggregate
print(f"\n{'='*90}\nTASK_MISS_RATE   (method × bench × stress)   — Table II driver\n{'='*90}")
agg = (df.groupby(['benchmark','stress','method'])
         .agg(n=('seed','count'),
              miss_mean=('task_miss_rate','mean'),
              miss_std=('task_miss_rate','std'),
              rt_viol=('rt_bound_violations','mean'),
              crash_A1=('A1_crash','mean'))
         .round(3))
print(agg.to_string())

print(f"\n{'='*90}\nRT_BOUND_VIOLATIONS (DARTS should have 0; baselines may have >0)\n{'='*90}")
rt_agg = (df.groupby(['benchmark','stress','method'])['rt_bound_violations']
            .agg(['mean','max','sum']).round(2))
print(rt_agg.to_string())

# Save per-cell + aggregated
os.makedirs(out_dir, exist_ok=True)
out1 = f"{out_dir}/sweep_summary_per_cell.csv"
out2 = f"{out_dir}/sweep_summary_aggregated.csv"
df.to_csv(out1, index=False)
agg.to_csv(out2)
print(f"\n[saved] per-cell:    {out1}")
print(f"[saved] aggregated:  {out2}")
