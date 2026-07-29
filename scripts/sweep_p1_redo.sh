#!/bin/bash
set -e
REPO=$(cd "$(dirname "$0")/.."; pwd)
cd "$REPO"
ENV_PREFIX="conda run -n DARTS_nano env PYTHONPATH=$REPO python3 -u"
LOG_DIR="sweep_logs_d1_20260518_2129"

# 单个 runner，没有 `;` 链，tee 正常 bind
tmux new-session -d -s sweep_p1_redo \
  "$ENV_PREFIX run_all_experiments_medium.py --benchmarks B1_CDR --methods darts,darts_nocert,darts_no_incentive,cea_edf,sadcher --resume 2>&1 | tee $REPO/$LOG_DIR/sweep_p1_redo.log; echo '[done]'; sleep 30"

echo "[launch] sweep_p1_redo"
tmux ls | grep sweep_
