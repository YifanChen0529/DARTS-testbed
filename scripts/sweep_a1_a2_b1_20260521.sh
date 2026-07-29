#!/usr/bin/env bash
set -e
SWEEP_DIR="results_a1a2b1_20260521"
LOG_DIR="sweep_logs_a1a2b1_$(date +%H%M%S)"
NUM_EP=2500
A1_METHODS="darts,darts_nocert,darts_no_incentive,cea_edf,sadcher"
A2_METHODS="darts,darts_nocert,cea_edf,sadcher"
B1_METHODS="darts,darts_nocert,cea_edf,sadcher"
REPO="$(pwd)"

SENTINEL_DIR="$LOG_DIR/sentinels"
mkdir -p "$SWEEP_DIR" "$LOG_DIR" "$SENTINEL_DIR"
echo "[launch] sweep_dir=$SWEEP_DIR  log_dir=$LOG_DIR  sentinel_dir=$SENTINEL_DIR"

# Notification backend: set ONE of these in your shell before launching.
# Recommended: ntfy.sh (zero setup; visit URL on phone/laptop to subscribe).
#   export NTFY_TOPIC_URL="https://ntfy.sh/darts-yifan-<random>"
# Or Gmail SMTP (requires app password; recommended for true email):
#   export SMTP_FROM="yifan.darts.bot@gmail.com"
#   export SMTP_PASS="xxxx xxxx xxxx xxxx"
#   export SMTP_TO="ychen1320@ucr.edu"
if [ -z "${NTFY_TOPIC_URL:-}" ] && [ -z "${SMTP_FROM:-}" ]; then
  echo "[launch] WARNING: no notification backend set; sweep will run silently."
  echo "[launch] Set NTFY_TOPIC_URL or SMTP_FROM/SMTP_PASS/SMTP_TO and re-launch to enable alerts."
  read -p "Continue without notifications? [y/N] " ans
  [ "$ans" = "y" ] || [ "$ans" = "Y" ] || exit 1
fi

SESSION="sweep_a1a2b1"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n light

launch_pane() {
  local WINDOW="$1"
  local RUNNER="$2"
  local METHODS="$3"
  local LOG="$LOG_DIR/sweep_${WINDOW}.log"
  # Build the inner command (the runner itself) as a quoted string
  local INNER="conda run --no-capture-output -n DARTS_nano env PYTHONPATH=$REPO python3 -u $RUNNER --benchmarks B1_CDR --methods $METHODS --num_episodes $NUM_EP --results_dir $SWEEP_DIR --resume"
  # Send environment variables explicitly so the tmux pane inherits them
  tmux send-keys -t "$SESSION:$WINDOW" "cd $REPO && export NTFY_TOPIC_URL='${NTFY_TOPIC_URL:-}' SMTP_FROM='${SMTP_FROM:-}' SMTP_PASS='${SMTP_PASS:-}' SMTP_TO='${SMTP_TO:-}' && bash scripts/_pane_wrapper.sh $WINDOW $LOG $SENTINEL_DIR $INNER" C-m
}

# Pane 1: A1 light  (already attached as window 0)
launch_pane "light"  "run_all_experiments_light.py"        "$A1_METHODS"

# Pane 2: A1 medium
tmux new-window -t "$SESSION" -n medium
launch_pane "medium" "run_all_experiments_medium.py"       "$A1_METHODS"

# Pane 3: A1 heavy
tmux new-window -t "$SESSION" -n heavy
launch_pane "heavy"  "run_all_experiments_heavy.py"        "$A1_METHODS"

# Pane 4: A2 energy (3 stress levels iterated by the runner itself)
tmux new-window -t "$SESSION" -n energy
launch_pane "energy" "run_all_experiments_energy.py"       "$A2_METHODS"

# Pane 5: B1 agent-scaling (3 n-values iterated by the runner itself)
tmux new-window -t "$SESSION" -n agent
launch_pane "agent"  "run_all_experiments_agent_scaling.py" "$B1_METHODS"

# Pane 6: monitor (stall detection + all-done summary)
tmux new-window -t "$SESSION" -n monitor
tmux send-keys -t "$SESSION:monitor" "cd $REPO && export NTFY_TOPIC_URL='${NTFY_TOPIC_URL:-}' SMTP_FROM='${SMTP_FROM:-}' SMTP_PASS='${SMTP_PASS:-}' SMTP_TO='${SMTP_TO:-}' && python3 scripts/sweep_monitor.py $SENTINEL_DIR $SWEEP_DIR 2>&1 | tee $LOG_DIR/sweep_monitor.log" C-m

echo ""
echo "Launched. Windows: light medium heavy energy agent monitor"
echo "  tmux attach -t $SESSION"
echo ""
echo "Test that notifications are working RIGHT NOW (recommended before walking away):"
echo "  python3 scripts/notify.py --title 'DARTS sweep: launched' --message 'Sweep started at $(date)' --priority default"
echo ""
echo "Monitor without attaching:"
echo "  tail -f $LOG_DIR/sweep_*.log"
echo "  tail -f $LOG_DIR/sweep_monitor.log"
echo "  find $SWEEP_DIR -name '*_log.csv' | wc -l   # cell count over time"
echo ""
echo "After all panes finish:"
echo "  python scripts/summarize_sweep.py --results_dir $SWEEP_DIR --out sweep_summary_a1a2b1"
