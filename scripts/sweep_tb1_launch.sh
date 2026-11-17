#!/usr/bin/env bash
#
# Launch TB-1 sweep panes in detached tmux sessions.
#
# Default (no args) starts ONLY the κ=1.0 pane. That pane is the regression gate: until
# tb1_analysis.py --check-kappa1 says κ=1.0 reproduces published behaviour, the other
# κ columns are not worth spending wall time on.
#
#   ./scripts/sweep_tb1_launch.sh            # κ=1.0 pane only  (session tb1_k10)
#   ./scripts/sweep_tb1_launch.sh --all      # the four off-unit panes
#   ./scripts/sweep_tb1_launch.sh --wait     # launch, then poll progress.log until done
#   ./scripts/sweep_tb1_launch.sh --all --wait
#   ./scripts/sweep_tb1_launch.sh --dry-run  # print the pane commands, launch nothing
#
# --wait polls $RESULTS_DIR/sweep_tb1_progress.log every 15s until the expected number
# of `rc=` lines for the launched kappas is present (gate: 9; --all: 36 with default
# cells), then exits 0, or nonzero if any of those lines has rc!=0. Default without
# --wait is unchanged.
# Pipe-binding hazard: each pane's `| tee` must run INSIDE the session's shell, not bind
# to the tmux client. Every session is therefore started by exactly one command whose
# whole pipeline is a single quoted string handed to `bash -c`.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export LD_LIBRARY_PATH="/home/yifan/miniconda3/envs/DARTS_nano/lib/cusparselt:${LD_LIBRARY_PATH:-}"
export NTFY_TOPIC_URL="${NTFY_TOPIC_URL:-}"  # opt-in: set it in the launching shell

RESULTS_DIR="${TB1_RESULTS_DIR:-results_tb1_p0}"
LOGDIR="${TB1_LOGDIR:-sweep_logs_tb1_$(date +%Y%m%d_%H%M%S)}"
CELLS="${TB1_CELLS:-B1_CDR:medium,B1_CDR:heavy,B2_LML:medium}"
MANIFEST="${TB1_MANIFEST:-ckpt_manifest_tb1.json}"
EVAL_EPISODES="${TB1_EVAL_EPISODES:-100}"

MODE="gate"
DRY_RUN=0
WAIT=0
for arg in "$@"; do
  case "$arg" in
    --all)     MODE="all" ;;
    --dry-run) DRY_RUN=1 ;;
    --wait)    WAIT=1 ;;
    -h|--help) sed -n '2,28p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)         echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$LOGDIR"

notify() {
  python3 scripts/notify.py --title "TB1 launch" --message "$1" --priority "${2:-default}" || true
}

pane_cmd() {
  # $1 session name, $2 kappa list
  printf 'python3 scripts/sweep_tb1.py --kappas %s --cells %s --manifest %s --results-dir %s --eval-episodes %s --subround-log --pane %s 2>&1 | tee %s/%s.log' \
    "$2" "$CELLS" "$MANIFEST" "$RESULTS_DIR" "$EVAL_EPISODES" "$1" "$LOGDIR" "$1"
}

launch_pane() {
  # $1 session name, $2 kappa list. Exactly one tmux command; the pipeline stays inside
  # the quoted bash -c string so tee runs in the session shell.
  local session="$1" kappas="$2" cmd
  cmd="$(pane_cmd "$session" "$kappas")"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[skip] session $session already exists (kill it first: tmux kill-session -t $session)"
    return 0
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] tmux new-session -d -s $session -c $REPO bash -c '$cmd'"
    return 0
  fi
  tmux new-session -d -s "$session" -c "$REPO" "bash -c '$cmd'"
  echo "[launch] $session  (κ=$kappas)  log: $LOGDIR/$session.log"
  notify "pane $session started (κ=$kappas) -> $RESULTS_DIR"
}

echo "repo         : $REPO"
echo "results dir  : $RESULTS_DIR"
echo "log dir      : $LOGDIR"
echo "cells        : $CELLS"
echo "eval episodes: $EVAL_EPISODES"
echo "mode         : $MODE"
echo

if [[ "$MODE" == "gate" ]]; then
  launch_pane tb1_k10 1.0
else
  launch_pane tb1_k05 0.5
  launch_pane tb1_k075 0.75
  launch_pane tb1_k15 1.5
  launch_pane tb1_k20 2.0
fi

echo
echo "reattach:"
if [[ "$MODE" == "gate" ]]; then
  echo "  tmux attach -t tb1_k10"
else
  for s in tb1_k05 tb1_k075 tb1_k15 tb1_k20; do echo "  tmux attach -t $s"; done
fi
echo "  tmux ls                      # sessions still running"
echo "  tail -f $LOGDIR/*.log        # follow every pane"
echo "  cat $RESULTS_DIR/sweep_tb1_progress.log   # per-run rc and timing"
echo
echo "when the κ=1.0 pane finishes, gate before launching the rest:"
echo "  python3 scripts/tb1_analysis.py --results-dir $RESULTS_DIR --check-kappa1"
echo "after --all completes:"
echo "  python3 scripts/tb1_analysis.py --results-dir $RESULTS_DIR --endpoint-report"

if [[ "$DRY_RUN" -eq 0 ]]; then
  notify "launcher done (mode=$MODE, results=$RESULTS_DIR, logs=$LOGDIR)"
fi

wait_for_progress() {
  # Poll sweep_tb1_progress.log until every launched κ has its rc= lines.
  # Default cells × 3 seeds: gate 9, --all 36. Count is computed from CELLS so a
  # custom TB1_CELLS still composes correctly.
  local -a kappas
  if [[ "$MODE" == "gate" ]]; then
    kappas=(1.0)
  else
    kappas=(0.5 0.75 1.5 2.0)
  fi
  local n_cells n_kappas expected log kpat n_done n_fail
  n_cells=$(echo "$CELLS" | awk -F',' '{print NF}')
  n_kappas=${#kappas[@]}
  expected=$((n_cells * 3 * n_kappas))
  log="$RESULTS_DIR/sweep_tb1_progress.log"
  kpat=$(printf '%s|' "${kappas[@]}")
  kpat="${kpat%|}"
  kpat="${kpat//./\\.}"

  echo
  echo "[wait] expecting $expected rc= lines for κ ∈ ${kappas[*]} in $log"
  while true; do
    n_done=0
    n_fail=0
    if [[ -f "$log" ]]; then
      n_done=$(grep -E " rc=[0-9]+ .* kappa=(${kpat}) " "$log" 2>/dev/null | wc -l | tr -d ' ')
      n_fail=$(grep -E " rc=[0-9]+ .* kappa=(${kpat}) " "$log" 2>/dev/null | grep -vcE " rc=0 " || true)
    fi
    echo "[wait] $(date '+%H:%M:%S')  ${n_done}/${expected} done  ${n_fail} failed"
    if [[ "${n_done:-0}" -ge "$expected" ]]; then
      if [[ "${n_fail:-0}" -gt 0 ]]; then
        echo "[wait] finished with ${n_fail} nonzero rc"
        return 1
      fi
      echo "[wait] all ${expected} runs rc=0"
      return 0
    fi
    sleep 15
  done
}

if [[ "$WAIT" -eq 1 && "$DRY_RUN" -eq 0 ]]; then
  wait_for_progress
fi
