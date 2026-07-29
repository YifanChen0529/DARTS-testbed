#!/bin/bash
set -e
REPO=$(cd "$(dirname "$0")/.."; pwd)
cd "$REPO"
ENV_PREFIX="conda run -n DARTS_nano env PYTHONPATH=$REPO python3 -u"
LOG_DIR="sweep_logs_d1_20260518_2129"

tmux new-session -d -s sweep_p4_v2 \
  "$ENV_PREFIX run_all_experiments_light.py --benchmarks B1_CDR,B2_LML --methods darts,darts_nocert,cea_edf,sadcher --resume 2>&1 | tee $REPO/$LOG_DIR/sweep_p4_v2.log; echo '[done]'; sleep 30"
echo "[launch] sweep_p4_v2"
tmux ls | grep sweep_
