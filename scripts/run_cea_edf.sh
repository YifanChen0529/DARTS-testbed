#!/bin/bash
set -e
echo "=== Running CEA-EDF Baseline ==="
echo "Centralized Energy-Aware EDF for Aperiodic DAGs"
echo "Architecture: Centralized dispatcher, no learning, no incentives"
echo ""

PYTHONPATH=. python alg/baseline_cea_edf.py \
    --exp_name cea_edf_baseline \
    --num_episodes 40

echo ""
echo "=== CEA-EDF Complete ==="
echo "Results: results/cea_edf_baseline/log.csv"
echo "Compare with: results/<darts_exp>/log.csv"
