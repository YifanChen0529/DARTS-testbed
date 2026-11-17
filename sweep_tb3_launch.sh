#!/usr/bin/env bash
#
# Launch the TB-3 empirical-anchor sweep as a single nohup child.
#
# NTFY_TOPIC_URL must be exported in the launching shell BEFORE invoking this
# script. The nohup child inherits it and must survive both tmux and SSH death.
#
# Pipe-binding hazard: this file is exactly one launch command (the nohup
# line). Do not wrap it in conda run; the runner is bare python3 and builds
# conda-wrapped inner evals itself.
#
#   export NTFY_TOPIC_URL=https://ntfy.sh/<your-topic>   # required
#   bash sweep_tb3_launch.sh              # launch, return immediately
#   bash sweep_tb3_launch.sh --wait       # launch, then poll progress until 180
#   bash sweep_tb3_launch.sh --wait --plan p7   # 240-run deployed-deadline expansion
#   bash sweep_tb3_launch.sh --dry-run    # print the nohup line, launch nothing
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

DRY_RUN=0
WAIT=0
PLAN="default"
FORWARD=()
args=("$@")
i=0
n=${#args[@]}
while [[ $i -lt $n ]]; do
  arg="${args[$i]}"
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --wait)    WAIT=1 ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    --plan)
      i=$((i + 1))
      if [[ $i -ge $n ]]; then
        echo "missing value for --plan" >&2
        exit 2
      fi
      PLAN="${args[$i]}"
      FORWARD+=(--plan "$PLAN")
      ;;
    --plan=*)
      PLAN="${arg#--plan=}"
      FORWARD+=(--plan "$PLAN")
      ;;
    *) FORWARD+=("$arg") ;;
  esac
  i=$((i + 1))
done

if [[ "$PLAN" == "p7" ]]; then
  RESULTS_DIR="${TB3_RESULTS_DIR:-results_tb3_emp_p7}"
  EXPECTED="${TB3_EXPECTED_RUNS:-240}"
else
  RESULTS_DIR="${TB3_RESULTS_DIR:-results_tb3_emp}"
  EXPECTED="${TB3_EXPECTED_RUNS:-180}"
fi

CONSOLE_LOG="$RESULTS_DIR/sweep_tb3_console.log"
PROGRESS_LOG="$RESULTS_DIR/sweep_tb3_progress.log"

if [[ -z "${NTFY_TOPIC_URL:-}" ]]; then
  echo "REFUSAL: export NTFY_TOPIC_URL in this shell before launching." >&2
  exit 2
fi

mkdir -p "$RESULTS_DIR"

# Single command: redirect is bound to nohup, not to a caller pipeline.
LAUNCH_CMD="nohup python3 scripts/sweep_tb3.py --results_dir $RESULTS_DIR"
if [[ ${#FORWARD[@]} -gt 0 ]]; then
  LAUNCH_CMD="$LAUNCH_CMD ${FORWARD[*]}"
fi
LAUNCH_CMD="$LAUNCH_CMD > $CONSOLE_LOG 2>&1 &"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] $LAUNCH_CMD"
  echo "[dry-run] NTFY_TOPIC_URL is set (${#NTFY_TOPIC_URL} chars)"
  exit 0
fi

echo "repo        : $REPO"
echo "results dir : $RESULTS_DIR"
echo "console log : $CONSOLE_LOG"
echo "progress log: $PROGRESS_LOG"
echo "NTFY_TOPIC_URL set (${#NTFY_TOPIC_URL} chars)"
echo

eval "$LAUNCH_CMD"
echo "[launch] sweep_tb3.py pid $!  (survives tmux/SSH death via nohup)"
echo "  tail -f $PROGRESS_LOG"
echo "  tail -f $CONSOLE_LOG"

wait_for_progress() {
  local n_done n_fail last_len=0
  echo
  echo "[wait] expecting $EXPECTED completion lines (rc=) in $PROGRESS_LOG"
  echo "[wait] aborting on a REFUSAL marker"
  while true; do
    n_done=0
    n_fail=0
    if [[ -f "$PROGRESS_LOG" ]]; then
      if grep -q "REFUSAL" "$PROGRESS_LOG" 2>/dev/null; then
        echo "[wait] REFUSAL marker in $PROGRESS_LOG"
        grep "REFUSAL" "$PROGRESS_LOG" || true
        return 1
      fi
      n_done=$(grep -cE " rc=" "$PROGRESS_LOG" 2>/dev/null || true)
      n_fail=$(grep -E " rc=" "$PROGRESS_LOG" 2>/dev/null | grep -vcE " rc=0 " || true)
      local len
      len=$(wc -l < "$PROGRESS_LOG" | tr -d ' ')
      if [[ "$len" -gt "$last_len" ]]; then
        tail -n $((len - last_len)) "$PROGRESS_LOG"
        last_len=$len
      fi
    fi
    echo "[wait] $(date '+%H:%M:%S')  ${n_done}/${EXPECTED} done  ${n_fail} failed"
    if [[ "${n_done:-0}" -ge "$EXPECTED" ]]; then
      echo "[wait] $EXPECTED completion lines present (failed=$n_fail)"
      return 0
    fi
    sleep 15
  done
}

if [[ "$WAIT" -eq 1 ]]; then
  wait_for_progress
fi
