#!/usr/bin/env bash
set -euo pipefail
SMOKE_DIR="results_smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SMOKE_DIR"
echo "[smoke] writing to $SMOKE_DIR"

NUM_EP=200

# --- A2 energy plumbing: one DARTS cell at b02 (tightest battery) ---
python run_all_experiments_energy.py \
  --benchmarks B1_CDR --methods darts \
  --num_episodes $NUM_EP \
  --results_dir "$SMOKE_DIR" 2>&1 | tee "$SMOKE_DIR/smoke_darts_energy.log"

# --- A2 energy SADCHER (should show crashes at b02) ---
python run_all_experiments_energy.py \
  --benchmarks B1_CDR --methods sadcher \
  --num_episodes $NUM_EP \
  --results_dir "$SMOKE_DIR" 2>&1 | tee "$SMOKE_DIR/smoke_sadcher_energy.log"

# --- B1 agent-scaling at n=2 (under-resourced edge case) ---
python run_all_experiments_agent_scaling.py \
  --benchmarks B1_CDR --methods darts \
  --num_episodes $NUM_EP \
  --results_dir "$SMOKE_DIR" 2>&1 | tee "$SMOKE_DIR/smoke_darts_n.log"

# --- B1 agent-scaling at n=8 via CEA-EDF (over-resourced + non-DARTS code path) ---
# We trim the runner's AGENT_COUNTS to [8] only via a one-shot env override is not
# trivial; instead invoke the underlying script directly with --num_agents 8 to
# test that the baseline CLI plumbing works end-to-end.
python alg/baseline_cea_edf.py \
  --workload B1_CDR --num_agents 8 \
  --battery_capacities_wh 0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25 \
  --global_deadline_ms 150000 \
  --max_episode_steps 200 --max_work_ms 142500 \
  --num_episodes $NUM_EP \
  --eval_interval 200 \
  --random_seed 100 \
  --exp_name cea_smoke_n8 2>&1 | tee "$SMOKE_DIR/smoke_cea_n8.log"

# --- Post-smoke validation ---
echo ""
echo "=== Post-smoke validation ==="
python - <<PYEOF
import pandas as pd, glob, sys, os
sd = "$SMOKE_DIR"
expected_cols = {"n_miss_n1_drop","n_miss_n2_fallback_insufficient","n_miss_n3_unreached","n_miss_full_fidelity_late"}
fails = []
# Walk both top-level smoke dirs AND the per-experiment workspace dirs that train_darts writes
candidate_csvs = []
for root, _, files in os.walk(sd):
    for f in files:
        if f.endswith("_log.csv"):
            candidate_csvs.append(os.path.join(root, f))
# Also check results/cea_smoke_n8 (workspace dir)
for f in glob.glob("results/cea_smoke_n8/*_log.csv"):
    candidate_csvs.append(f)
if not candidate_csvs:
    fails.append("NO CSV files found under smoke dir or results/cea_smoke_n8")
for csv_path in candidate_csvs:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        fails.append(f"PARSE FAIL {csv_path}: {e}")
        continue
    missing = expected_cols - set(df.columns)
    if missing:
        fails.append(f"MISSING COLS {csv_path}: {sorted(missing)}")
        continue
    last = df.iloc[-1]
    n1 = last["n_miss_n1_drop"]; n2 = last["n_miss_n2_fallback_insufficient"]
    n3 = last["n_miss_n3_unreached"]; ffl = last["n_miss_full_fidelity_late"]
    crash_cols = [c for c in df.columns if c.endswith("_crash_rate")]
    crash_sum = sum(float(last[c]) for c in crash_cols) if crash_cols else 0.0
    print(f"OK {csv_path}: n={len(df)} cols={len(df.columns)} last_breakdown=(N1={n1},N2={n2},N3={n3},FFL={ffl}) crash_sum={crash_sum}")
if fails:
    print("\n=== SMOKE FAILED ===")
    for f in fails: print(f)
    sys.exit(1)
print("\n=== SMOKE PASSED ===")
PYEOF
