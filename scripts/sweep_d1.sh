#!/bin/bash
# Path1 + NoIncentive sweep launcher (D1 evening, 5-parallel)
# Total: 108 cells (60 medium + 24 light + 24 heavy), ~14-15h max wall-clock.
set -e
REPO=$(cd "$(dirname "$0")/.."; pwd)
cd "$REPO"

ENV_PREFIX="conda run -n DARTS_nano env PYTHONPATH=$REPO python3 -u"
LOG_DIR="sweep_logs_d1_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOG_DIR"
echo "[info] sweep logs → $REPO/$LOG_DIR"

launch_pane() {
  local name="$1"; shift
  tmux new-session -d -s "$name" "$* 2>&1 | tee $REPO/$LOG_DIR/${name}.log; echo '[done]' && sleep 30"
  echo "[launch] $name"
}

# Pane 1: medium B1 (15c) + heavy B1 (12c) — paper-critical
launch_pane sweep_p1 \
  "$ENV_PREFIX run_all_experiments_medium.py --benchmarks B1_CDR --methods darts,darts_nocert,darts_no_incentive,cea_edf,sadcher; \
   $ENV_PREFIX run_all_experiments_heavy.py  --benchmarks B1_CDR --methods darts,darts_nocert,cea_edf,sadcher"

# Pane 2: medium B2 (15c) + heavy B2 (12c)
launch_pane sweep_p2 \
  "$ENV_PREFIX run_all_experiments_medium.py --benchmarks B2_LML --methods darts,darts_nocert,darts_no_incentive,cea_edf,sadcher; \
   $ENV_PREFIX run_all_experiments_heavy.py  --benchmarks B2_LML --methods darts,darts_nocert,cea_edf,sadcher"

# Pane 3: medium B3 (15c) — topology probe
launch_pane sweep_p3 \
  "$ENV_PREFIX run_all_experiments_medium.py --benchmarks B3_IQI --methods darts,cea_edf,sadcher"

# Pane 4: light B1+B2 (24c)
launch_pane sweep_p4 \
  "$ENV_PREFIX run_all_experiments_light.py --benchmarks B1_CDR,B2_LML --methods darts,darts_nocert,cea_edf,sadcher"

# Pane 5: medium B4 (15c)
launch_pane sweep_p5 \
  "$ENV_PREFIX run_all_experiments_medium.py --benchmarks B4_DEM --methods darts,cea_edf,sadcher"

echo ""
echo "=== All 5 panes launched ==="
tmux ls | grep sweep_
echo ""
echo "Reattach: tmux a -t sweep_p<N>     (N=1..5)"
echo "Detach:   Ctrl-b then d"
echo "Logs:     $REPO/$LOG_DIR/"
