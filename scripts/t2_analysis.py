"""Compute §V.G T2 numbers from per_commit.jsonl logs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/yifan/DARTS-Simulation")
OUT = REPO / "t2_analysis_output.md"
CONV_EPISODE = 4000  # convergence window: episodes >= 4000

rows = []
for fp in sorted(REPO.glob("results/t2_instr_*/per_commit.jsonl")):
    exp = fp.parent.name.replace("t2_instr_", "")
    # e.g., B2_LML_medium_seed100
    parts = exp.split("_")
    benchmark = f"{parts[0]}_{parts[1]}"
    seed = int(parts[3].replace("seed", ""))
    with open(fp) as f:
        for line in f:
            d = json.loads(line)
            d["benchmark"] = benchmark
            d["seed"] = seed
            rows.append(d)

df = pd.DataFrame(rows)
print(f"Loaded {len(df)} commits from {df['benchmark'].nunique()} benchmarks × {df['seed'].nunique()} seeds")

# Convergence window
df = df[df["episode_idx"] >= CONV_EPISODE].copy()
print(f"After convergence (ep>={CONV_EPISODE}): {len(df)}")

# Full-fidelity filter: rv_ms within relative deadline (not missed)
df["d_v_rel_ms"] = df["d_v_ms"] - df["tau_ms"]
df = df[df["rv_ms"] <= df["d_v_rel_ms"]].copy()
print(f"After full-fidelity filter (rv <= d_v relative): {len(df)}")

# Slack
df["slack_ms"] = df["rt_bound_ms"] - df["rv_ms"]

YY = len(df)
BB = float(np.median(df["slack_ms"]))
CC = float(np.percentile(df["slack_ms"], 99))

# ZZ approximation: I_i^pi = 0 iff no other commit lands on same agent during [tc, cv]
# NOTE: lower-bound estimate (loose definition: counts any later commit, not just preempting)
n_iv_zero = 0
total = 0
for (_b, _s, _agent), g in df.groupby(["benchmark", "seed", "agent_idx"]):
    g = g.sort_values("tc_ms").reset_index(drop=True)
    tcs = g["tc_ms"].values
    cvs = g["cv_ms"].values
    for i in range(len(g)):
        tc_i = tcs[i]
        cv_i = cvs[i]
        mask = (tcs > tc_i) & (tcs < cv_i)
        n_later = int(mask.sum())
        if n_later == 0:
            n_iv_zero += 1
        total += 1
ZZ = 100.0 * n_iv_zero / max(total, 1)

per_bench = []
for b, g in df.groupby("benchmark"):
    slack = g["slack_ms"].values
    per_bench.append({
        "benchmark": b,
        "n_commits": len(g),
        "median_slack_ms": float(np.median(slack)),
        "p99_slack_ms": float(np.percentile(slack, 99)),
        "min_slack_ms": float(np.min(slack)),
    })
bd = pd.DataFrame(per_bench)

with open(OUT, "w") as f:
    f.write("# §V.G T2 — Per-job response-time tightness (instrumented sweep)\n\n")
    f.write(f"Convergence window: episodes >= {CONV_EPISODE}. Full-fidelity filter applied.\n\n")
    f.write("## Headline numbers\n\n")
    f.write("| | Value |\n|---|---|\n")
    f.write(f"| **YY** (total full-fidelity commits) | **{YY:,}** |\n")
    f.write(f"| **ZZ** (% with I_i^π = 0, lower bound) | **{ZZ:.1f}%** |\n")
    f.write(f"| **BB** (median slack, rt_bound − R_v) | **{BB:,.0f} ms** |\n")
    f.write(f"| **CC** (P99 slack) | **{CC:,.0f} ms** |\n\n")
    f.write("## Per-benchmark breakdown\n\n")
    f.write(bd.to_markdown(index=False, floatfmt=",.0f") + "\n")

print("\n=== T2 RESULTS ===")
print(f"YY = {YY:,}")
print(f"ZZ = {ZZ:.1f}%  (lower-bound; true ZZ may be higher)")
print(f"BB = {BB:,.0f} ms")
print(f"CC = {CC:,.0f} ms")
print(f"\nReport: {OUT}")
