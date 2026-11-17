#!/usr/bin/env bash
set -euo pipefail
REPO=/home/yifan/DARTS-Simulation
cd "$REPO"
NTFY="${NTFY_TOPIC_URL:?NTFY_TOPIC_URL not set; export it before launch}"
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$REPO/sweep_logs_t2_${TS}"
mkdir -p "$LOG_DIR"
SESSION="t2_instr"

stress_args_for() {
  case "$1" in
    B2_LML)
      echo "--global_deadline_ms 240000.0 --relative_deadline_ms 170000.0"
      echo "--battery_capacities_wh 0.55,0.80,1.05,1.30 --max_episode_steps 280"
      echo "--max_work_ms 220000.0 --poisson_mean_ms 30000.0"
      ;;
    B3_IQI)
      echo "--global_deadline_ms 180000.0 --relative_deadline_ms 150000.0"
      echo "--battery_capacities_wh 0.65,0.95,1.25,1.55 --max_episode_steps 220"
      echo "--max_work_ms 170000.0"
      ;;
    B4_DEM)
      echo "--global_deadline_ms 100000.0"
      echo "--battery_capacities_wh 0.50,0.75,1.00,1.25 --max_episode_steps 160"
      echo "--max_work_ms 95000.0"
      ;;
    *)
      echo "Unknown benchmark: $1" >&2
      return 1
      ;;
  esac
}

tmux kill-session -t "$SESSION" 2>/dev/null || true

declare -a CELLS=(
  "B2_LML 100" "B2_LML 200" "B2_LML 300"
  "B3_IQI 100" "B3_IQI 200" "B3_IQI 300"
  "B4_DEM 100" "B4_DEM 200" "B4_DEM 300"
)
N_CELLS=${#CELLS[@]}

# Window 0: monitor
tmux new-session -d -s "$SESSION" -n "monitor"
tmux send-keys -t "$SESSION:monitor" "
  echo '[MONITOR] T2 sweep ${TS}, ${N_CELLS} cells, logs in ${LOG_DIR}'
  START=\$(date +%s)
  while true; do
    DONE=\$(ls ${LOG_DIR}/done_*.flag 2>/dev/null | wc -l)
    ELAPSED=\$(( (\$(date +%s) - \$START) / 60 ))
    echo \"\$(date +%H:%M:%S) [MONITOR] done=\$DONE/${N_CELLS} elapsed=\${ELAPSED}min\"
    if [ \$DONE -ge ${N_CELLS} ]; then
      MSG=\"T2 sweep ALL DONE: ${N_CELLS}/${N_CELLS} cells in \${ELAPSED}min\"
      curl -s -d \"\$MSG\" \"${NTFY}\" >/dev/null
      echo \"\$MSG\"
      break
    fi
    sleep 300
  done
" C-m

# One window per cell
for cell in "${CELLS[@]}"; do
  read -r BENCH SEED <<< "$cell"
  WIN="${BENCH:0:2}_s${SEED}"
  EXP="t2_instr_${BENCH}_medium_seed${SEED}"
  FLAG="$LOG_DIR/done_${EXP}.flag"
  LOG="$LOG_DIR/${EXP}.log"
  STRESS_ARGS=$(stress_args_for "$BENCH" | tr '\n' ' ')
  tmux new-window -t "$SESSION" -n "$WIN"
  tmux send-keys -t "$SESSION:$WIN" "
    source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \\
      source ~/anaconda3/etc/profile.d/conda.sh
    conda activate DARTS_nano
    cd ${REPO}
    echo \"[START ${EXP}] \$(date)\" | tee ${LOG}
    PYTHONPATH=. python3 -u alg/train_darts.py \\
      --workload ${BENCH} --bilevel_mode fixed_eta \\
      ${STRESS_ARGS} \\
      --num_episodes 5000 --random_seed ${SEED} \\
      --log_per_commit \\
      --exp_name ${EXP} 2>&1 | tee -a ${LOG}
    RC=\${PIPESTATUS[0]}
    echo \"[END ${EXP} rc=\$RC] \$(date)\" | tee -a ${LOG}
    touch ${FLAG}
    curl -s -d \"T2 cell ${EXP} done rc=\$RC\" \"${NTFY}\" >/dev/null 2>&1 || true
  " C-m
done

curl -s -d "T2 sweep LAUNCHED ${N_CELLS} cells" "${NTFY}" >/dev/null 2>&1 || true
echo "[LAUNCHED] tmux attach -t ${SESSION}   |   logs: ${LOG_DIR}"
