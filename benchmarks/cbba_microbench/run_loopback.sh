#!/usr/bin/env bash
# Layout: four background node processes on 127.0.0.1:55201..55204 with
# mutual peer lists, then the runner, then cleanup killing the four nodes.
# Console log is a SIBLING of out_dir, never inside it (EXP-2 P2b: the
# launcher must not write into the results root before the runner).
#
# Usage (from repo root, DARTS_nano available):
#   bash benchmarks/cbba_microbench/run_loopback.sh --out_dir results_exp3_smoke --trials 10 --warmups 2 --null_trials 5
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

OUT_DIR=""
FORWARD=()
args=("$@")
i=0
n=${#args[@]}
while [[ $i -lt $n ]]; do
  arg="${args[$i]}"
  case "$arg" in
    --out_dir)
      i=$((i + 1))
      OUT_DIR="${args[$i]}"
      FORWARD+=("--out_dir" "$OUT_DIR")
      ;;
    *)
      FORWARD+=("$arg")
      ;;
  esac
  i=$((i + 1))
done

if [[ -z "$OUT_DIR" ]]; then
  echo "REFUSAL: --out_dir is required" >&2
  exit 2
fi

# Sibling of OUT_DIR: never $OUT_DIR/... (stale-dir false positive).
CONSOLE_LOG="${OUT_DIR%/}_console.log"

PY="$(conda run -n DARTS_nano python3 -c 'import sys; print(sys.executable)')"
export PYTHONPATH="$REPO"

cleanup() {
  if [[ ${#NODE_PIDS[@]} -gt 0 ]]; then
    kill "${NODE_PIDS[@]}" 2>/dev/null || true
    wait "${NODE_PIDS[@]}" 2>/dev/null || true
  fi
}
NODE_PIDS=()
trap cleanup EXIT

HOST=127.0.0.1
P0=55201
P1=55202
P2=55203
P3=55204

"$PY" benchmarks/cbba_microbench/cbba_node.py --agent_id 0 --port "$P0" --n_agents 4 --peers "${HOST}:${P1},${HOST}:${P2},${HOST}:${P3}" &
NODE_PIDS+=($!)
"$PY" benchmarks/cbba_microbench/cbba_node.py --agent_id 1 --port "$P1" --n_agents 4 --peers "${HOST}:${P0},${HOST}:${P2},${HOST}:${P3}" &
NODE_PIDS+=($!)
"$PY" benchmarks/cbba_microbench/cbba_node.py --agent_id 2 --port "$P2" --n_agents 4 --peers "${HOST}:${P0},${HOST}:${P1},${HOST}:${P3}" &
NODE_PIDS+=($!)
"$PY" benchmarks/cbba_microbench/cbba_node.py --agent_id 3 --port "$P3" --n_agents 4 --peers "${HOST}:${P0},${HOST}:${P1},${HOST}:${P2}" &
NODE_PIDS+=($!)

"$PY" benchmarks/cbba_microbench/cbba_trial_runner.py --bind_host "$HOST" --ports "${P0},${P1},${P2},${P3}" "${FORWARD[@]}" > "$CONSOLE_LOG" 2>&1
