#!/usr/bin/env bash
# NoIncentive completion sweep for FIG 1 + §V.F (2026-05-23)
# Track 1: B1 light/heavy (10+10) + B1 medium seeds 4-10 (7) = 27 cells
# Track 2: B2 medium (10) = 10 cells
# Total: 37 cells across 4 parallel tmux panes + monitor
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

# ── STEP 0: PRE-FLIGHT ──────────────────────────────────────────────────────
if [ -z "${NTFY_TOPIC_URL:-}" ]; then
  echo "[FATAL] NTFY_TOPIC_URL not set"
  exit 1
fi

for f in scripts/notify.py scripts/_pane_wrapper.sh scripts/sweep_monitor.py; do
  [ -f "$f" ] || { echo "[FATAL] missing $f"; exit 1; }
done

if ! conda env list | grep -q "^DARTS_nano "; then
  echo "[FATAL] conda env DARTS_nano missing"
  exit 1
fi

_dry_run_check() {
  local runner="$1" bench="$2"
  if ! conda run -n DARTS_nano python "$runner" \
      --methods darts_no_incentive --benchmarks "$bench" --dry_run 2>&1 | grep -q "CLI:"; then
    echo "[FATAL] runner does not support darts_no_incentive on $bench"
    echo "Inspect METHODS / _WORKSPACE_LOG_NAMES dicts in $runner"
    exit 1
  fi
}

_dry_run_check run_all_experiments_medium.py B2_LML
_dry_run_check run_all_experiments_light.py B1_CDR
_dry_run_check run_all_experiments_heavy.py B1_CDR
_dry_run_check run_all_experiments_medium.py B1_CDR

echo "[preflight] OK (NTFY, scripts, conda env, dry-run)"

# ── STEP 1: BACKUP PARTIAL CSVs ─────────────────────────────────────────────
TS=$(date +%Y%m%d_%H%M)
BACKUP="results_a1a2b1_20260521/_partial_noinc_backup_${TS}"
LOG_DIR="sweep_logs_noinc_completion_${TS}"
SENTINEL_DIR="$LOG_DIR/sentinels"
mkdir -p "$BACKUP" "$LOG_DIR" "$SENTINEL_DIR"

moved=0
for stress in light heavy; do
  for s in $(seq 1 10); do
    src="results_a1a2b1_20260521/exp_${s}/B1/${stress}/darts_no_incentive_log.csv"
    if [ -f "$src" ]; then
      mv "$src" "$BACKUP/exp_${s}_B1_${stress}.csv"
      moved=$((moved + 1))
    fi
  done
done
for s in $(seq 4 10); do
  src="results_a1a2b1_20260521/exp_${s}/B1/medium/darts_no_incentive_log.csv"
  if [ -f "$src" ]; then
    mv "$src" "$BACKUP/exp_${s}_B1_medium.csv"
    moved=$((moved + 1))
  fi
done
echo "[backup] moved $moved partial CSV(s) to $BACKUP/"

for s in 1 2 3; do
  f="results_a1a2b1_20260521/exp_${s}/B1/medium/darts_no_incentive_log.csv"
  [ -f "$f" ] || echo "[WARN] expected seed $s medium CSV missing: $f"
done

# ── STEP 2: LAUNCH 4 SWEEP PANES + MONITOR ──────────────────────────────────
ENV_PREFIX="conda run --no-capture-output -n DARTS_nano env PYTHONPATH=$REPO python3 -u"
RESULTS_DIR="results_a1a2b1_20260521"
NTFY_EXPORT="export NTFY_TOPIC_URL='${NTFY_TOPIC_URL}'"

launch_pane() {
  local session="$1" pane="$2" runner="$3" benchmarks="$4" logfile="$5"
  local inner="${ENV_PREFIX} ${runner} --methods darts_no_incentive --benchmarks ${benchmarks} --num_episodes 5000 --resume --results_dir ${RESULTS_DIR}"
  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" \
    "cd $REPO && ${NTFY_EXPORT} && bash scripts/_pane_wrapper.sh $pane $logfile $SENTINEL_DIR $inner"
  echo "[launch] tmux session=$session pane=$pane"
}

launch_pane noinc_b1_light  noinc_b1_light  run_all_experiments_light.py  B1_CDR "$LOG_DIR/noinc_b1_light.log"
launch_pane noinc_b1_heavy  noinc_b1_heavy  run_all_experiments_heavy.py B1_CDR "$LOG_DIR/noinc_b1_heavy.log"
launch_pane noinc_b1_medium noinc_b1_medium run_all_experiments_medium.py B1_CDR "$LOG_DIR/noinc_b1_medium.log"
launch_pane noinc_b2_medium noinc_b2_medium run_all_experiments_medium.py B2_LML "$LOG_DIR/noinc_b2_medium.log"

tmux kill-session -t noinc_monitor 2>/dev/null || true
tmux new-session -d -s noinc_monitor \
  "cd $REPO && ${NTFY_EXPORT} && python3 scripts/sweep_monitor.py \
    --sentinel_dir $SENTINEL_DIR \
    --log_dir $LOG_DIR \
    --results_dir $RESULTS_DIR \
    --sessions noinc_b1_light,noinc_b1_heavy,noinc_b1_medium,noinc_b2_medium \
    --stall_minutes 60 \
    2>&1 | tee $LOG_DIR/MONITOR.log"

echo "[launch] tmux session=noinc_monitor"

# ── STEP 3: NTFY + ATTACH INSTRUCTIONS ──────────────────────────────────────
python3 scripts/notify.py \
  --title "NoIncen sweep started ($TS)" \
  --message "Tracks: 27 B1 cells + 10 B2 cells = 37 total
4 panes parallel, ETA ~8-18h
Attach: tmux a -t noinc_b1_light (or _heavy/_medium/_b2_medium)
Logs: $REPO/$LOG_DIR/
Backup: $REPO/$BACKUP/ ($moved files)"

echo "===== LAUNCHED ====="
tmux ls | grep -E "noinc_" || true
echo
echo "Backup:    $BACKUP/  ($moved partial CSVs moved aside)"
echo "Logs:      $LOG_DIR/"
echo "Sentinels: $SENTINEL_DIR/"
echo "Attach:    tmux a -t noinc_b1_light"
echo "           tmux a -t noinc_b1_heavy"
echo "           tmux a -t noinc_b1_medium"
echo "           tmux a -t noinc_b2_medium"
echo "           tmux a -t noinc_monitor"
echo "Detach:    Ctrl-b d  (inside any tmux session)"
echo "Kill all:  tmux kill-session -t noinc_b1_light  (repeat for each)"
echo
echo "Expected ntfy events:"
echo "  - On each pane exit: pane DONE/FAILED with rc and duration"
echo "  - On stall (>60 min no log activity): warning"
echo "  - On all 4 sweep panes done: summary"
echo
echo "After completion, refresh FIG 1:"
echo "  conda run -n DARTS_nano python scripts/fig1_b1_grid.py"
echo
echo "===== RISK NOTES ====="
echo "1. Jetson Orin Nano = 6 ARM cores. 4 parallel train_darts.py may saturate."
echo "   If after 10 min, $LOG_DIR/noinc_b1_light.log shows no episode progress,"
echo "   consider killing one pane (e.g., noinc_b2_medium) and rerunning later."
echo "2. --resume + backup means re-launching this script is safe; existing"
echo "   v9 CSVs (seeds 1-3 medium) are preserved."
echo "3. If runner errors with 'KeyError: darts_no_incentive', the on-Jetson copy"
echo "   of run_all_experiments_*.py is missing dict entries. Patch:"
echo "     METHODS['darts_no_incentive'] = ROOT/'alg'/'train_darts.py'"
echo "     _WORKSPACE_LOG_NAMES['darts_no_incentive'] = 'darts_no_incentive_log.csv'"
echo "     _AGGREGATE_LOG_NAMES['darts_no_incentive'] = 'darts_no_incentive_log.csv'"
echo "     _WORKSPACE_CONFIG_NAMES['darts_no_incentive'] = 'darts_no_incentive_config.json'"
echo "   + in _build_cmd: 'if method == \"darts_no_incentive\": cmd.append(\"--no_incentive\")'"
