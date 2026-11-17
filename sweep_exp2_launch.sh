#!/usr/bin/env bash
#
# Launch the EXP-2 Theorem 2(a) sweep as a single nohup child.
#
# NTFY_TOPIC_URL must be exported in the launching shell BEFORE invoking this
# script. The nohup child inherits it and must survive both tmux and SSH death.
#
# Pipe-binding hazard: this file is exactly one launch command (the nohup
# line). Do not wrap it in conda run; the runner is bare python3 and builds
# conda-wrapped inner evals itself.
#
#   export NTFY_TOPIC_URL=https://ntfy.sh/<your-topic>   # required
#   bash sweep_exp2_launch.sh              # launch, return immediately
#   bash sweep_exp2_launch.sh --dry-run    # print the nohup line, launch nothing
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

DRY_RUN=0
FORWARD=()
args=("$@")
i=0
n=${#args[@]}
while [[ $i -lt $n ]]; do
  arg="${args[$i]}"
  case "$arg" in
    --dry-run|--dry_run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,18p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) FORWARD+=("$arg") ;;
  esac
  i=$((i + 1))
done

RESULTS_DIR="${EXP2_RESULTS_DIR:-results_exp2_$(date -u +%Y%m%d)}"
# Sibling of RESULTS_DIR: nohup creates this file before the runner starts, so
# it must not live inside the root (that was a stale-dir false positive).
CONSOLE_LOG="${RESULTS_DIR%/}_console.log"
PROGRESS_LOG="$RESULTS_DIR/sweep_exp2_progress.log"

if [[ -z "${NTFY_TOPIC_URL:-}" ]]; then
  echo "REFUSAL: export NTFY_TOPIC_URL in this shell before launching." >&2
  exit 2
fi

mkdir -p "$RESULTS_DIR"

# Single command: redirect is bound to nohup, not to a caller pipeline.
LAUNCH_CMD="nohup python3 scripts/sweep_exp2.py --results_dir $RESULTS_DIR"
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
echo "[launch] sweep_exp2.py pid $!  (survives tmux/SSH death via nohup)"
echo "  tail -f $PROGRESS_LOG"
echo "  tail -f $CONSOLE_LOG"
