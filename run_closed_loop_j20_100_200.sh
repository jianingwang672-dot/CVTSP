#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

exec python3 -u run_self_improvement.py \
  --teacher-checkpoint outputs/train__fresh/best.pt \
  --output-dir outputs/self_improvement_closed_loop_j20_100_epoch200_seed20260812 \
  --rounds 3 \
  --train-instances 8 \
  --validation-instances 2 \
  --promotion-instances 2 \
  --min-problem-size 20 \
  --max-problem-size 100 \
  --teacher-candidates 4 \
  --teacher-decode greedy \
  --local-search-refreshes 10 \
  --imitation-epochs 200 \
  --imitation-batch-size 4 \
  --learning-rate 1e-4 \
  --edge-top-k 30 \
  --edge-max-anchors-per-instance 4 \
  --edge-epochs 50 \
  --edge-batch-size 8 \
  --edge-learning-rate 1e-3 \
  --threads 1 \
  --device cpu \
  --seed 20260812 \
  "$@"
