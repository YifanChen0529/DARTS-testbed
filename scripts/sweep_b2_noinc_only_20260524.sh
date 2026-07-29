#!/usr/bin/env bash
# B2/medium NoIncen completion ONLY — seeds 4-10 via --resume.
# DO NOT launch until T2 sweep ALL DONE.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

if [ -z "${NTFY_TOPIC_URL:-}" ]; then
    echo "[FATAL] NTFY_TOPIC_URL not set"; exit 1
fi

# Sanity: B1 NoIncen must already be 30/30 complete (avoid accidental rerun)
B1_DONE=$(ls results_a1a2b1_20260521/exp_*/B1/*/darts_no_incentive_log.csv 2>/dev/null | wc -l)
if [ "$B1_DONE" -lt 30 ]; then
    echo "[FATAL] B1 NoIncen incomplete ($B1_DONE/30) — not safe to run"; exit 1
fi
echo "[preflight] B1 NoIncen: $B1_DONE/30 ✓"

# Sanity: T2 sweep must be fully done (no more train_darts.py running)
T2_RUNNING=$(pgrep -af "train_darts.py" | grep -v grep | wc -l)
if [ "$T2_RUNNING" -gt 0 ]; then
    echo "[FATAL] Other train_darts.py processes running ($T2_RUNNING) — wait for T2 sweep ALL DONE first"
    pgrep -af "train_darts.py" | head -3
    exit 1
fi
echo "[preflight] No concurrent train_darts.py ✓"

TS=$(date +%Y%m%d_%H%M)
LOG_DIR="$REPO/sweep_logs_b2_noinc_${TS}"
SENTINEL_DIR="$LOG_DIR/sentinels"
mkdir -p "$SENTINEL_DIR"

SESSION="b2_noinc"
tmux kill-session -t "$SESSION" 2>/dev/null || true

ENV_PREFIX="conda run --no-capture-output -n DARTS_nano env PYTHONPATH=$REPO python3 -u"
RESULTS_DIR="results_a1a2b1_20260521"
NTFY_EXPORT="export NTFY_TOPIC_URL='${NTFY_TOPIC_URL}'"

INNER="${ENV_PREFIX} run_all_experiments_medium.py --methods darts_no_incentive --benchmarks B2_LML --num_episodes 5000 --resume --results_dir ${RESULTS_DIR}"

LOGFILE="$LOG_DIR/b2_noinc.log"

# Worker pane
tmux new-session -d -s "$SESSION" -n b2_noinc \
    "cd $REPO && ${NTFY_EXPORT} && bash scripts/_pane_wrapper.sh b2_noinc $LOGFILE $SENTINEL_DIR ${INNER}"

# Progress monitor pane (poll every 10 min)
tmux new-window -t "$SESSION" -n monitor
tmux send-keys -t "$SESSION:monitor" "
    START=\$(date +%s)
    while true; do
        n_done=\$(ls results_a1a2b1_20260521/exp_*/B2/medium/darts_no_incentive_log.csv 2>/dev/null | wc -l)
        ELAPSED=\$(( (\$(date +%s) - \$START) / 60 ))
        echo \"\$(date +%H:%M:%S) B2 NoIncen: \$n_done/10  elapsed=\${ELAPSED}min\"
        if [ \$n_done -ge 10 ] || [ -f $SENTINEL_DIR/PANE_DONE_b2_noinc ]; then
            curl -s -d \"B2 NoIncen ALL DONE: \$n_done/10 in \${ELAPSED}min\" \"$NTFY_TOPIC_URL\" >/dev/null
            echo \"DONE\"
            break
        fi
        sleep 600
    done
" C-m

curl -s -d "B2 NoIncen LAUNCHED (--resume, 7 cells, est ~25h)" "$NTFY_TOPIC_URL" >/dev/null || true
echo "[LAUNCHED] tmux attach -t $SESSION  |  logs: $LOG_DIR"
