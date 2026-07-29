#!/usr/bin/env bash
# Args: $1=pane_name, $2=log_path, $3=sentinel_dir, $4...=command
set -u
PANE="$1"
LOG="$2"
SENTINEL_DIR="$3"
shift 3
mkdir -p "$(dirname "$LOG")" "$SENTINEL_DIR"

START_TS=$(date +%s)
echo "[wrapper] pane=$PANE START $(date)" | tee -a "$LOG"

# Run the actual command; capture exit code
"$@" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
ELAPSED_H=$((ELAPSED / 3600))
ELAPSED_M=$(((ELAPSED % 3600) / 60))

# Count cells completed by this pane (workspace dirs created)
CELLS=$(find results -maxdepth 1 -type d -newer "$LOG" 2>/dev/null | wc -l)

# Write sentinel
echo "$RC" > "$SENTINEL_DIR/PANE_DONE_$PANE"
echo "[wrapper] pane=$PANE END rc=$RC elapsed=${ELAPSED_H}h${ELAPSED_M}m $(date)" | tee -a "$LOG"

# Notify
if [ "$RC" -eq 0 ]; then
  python3 scripts/notify.py \
    --priority default \
    --title "DARTS sweep: pane $PANE DONE" \
    --message "Pane=$PANE exit=0 elapsed=${ELAPSED_H}h${ELAPSED_M}m. Log tail: $(tail -5 "$LOG" | tr '\n' ' ')" || true
else
  python3 scripts/notify.py \
    --priority urgent \
    --title "DARTS sweep: pane $PANE FAILED (rc=$RC)" \
    --message "Pane=$PANE exit=$RC elapsed=${ELAPSED_H}h${ELAPSED_M}m. Last log lines: $(tail -15 "$LOG" | tr '\n' ' ' | head -c 800)" || true
fi
exit "$RC"
