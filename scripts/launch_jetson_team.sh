#!/bin/bash
# Launches one DARTS agent process per Jetson Orin Nano via SSH.
# Prerequisites: passwordless SSH from coordinator host to all 4 Jetsons; DARTS-Simulation
# repo synced to /home/yifan/DARTS-Simulation on each Jetson; DARTS_nano venv active.
set -euo pipefail
JETSONS=("yifan@jetson1" "yifan@jetson2" "yifan@jetson3" "yifan@jetson4")
AGENT_IDS=("agent_1" "agent_2" "agent_3" "agent_4")
WORKLOAD="${1:-B1_CDR}"
EXP_NAME="${2:-rtss_eval_$(date +%Y%m%d_%H%M%S)}"
for k in 0 1 2 3; do
    ssh "${JETSONS[$k]}" "mkdir -p ~/darts_logs && cd ~/DARTS-Simulation && source DARTS_nano/bin/activate && \
        PYTHONPATH=. python alg/agent_node.py \
            --local_agent_id ${AGENT_IDS[$k]} \
            --workload $WORKLOAD \
            --exp_name $EXP_NAME \
            --deployment_mode distributed_jetson \
            > ~/darts_logs/${AGENT_IDS[$k]}_${EXP_NAME}.log 2>&1 &"
    echo "Launched ${AGENT_IDS[$k]} on ${JETSONS[$k]}"
done
echo "All 4 agent_nodes launched. Now run alg/coordinator.py from this host."
