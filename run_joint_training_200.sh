#!/usr/bin/env bash
set -euo pipefail

project_dir="/home/Mingfan/wjn/CVTSP"
output_dir="$project_dir/outputs/train__joint_si_sparse_gurobi_j20_100_e200_v2"
mkdir -p "$output_dir"
cd "$project_dir"

python3 -u train_joint_self_improvement.py \
  --epochs 200 \
  --train-instances 100000 \
  --batch-size 512 \
  --min-problem-size 20 \
  --max-problem-size 100 \
  --pomo-size 100 \
  --cuda-device 0 \
  --self-improvement-workers 32 \
  --top-k 4 \
  --max-local-moves 30 \
  --predictor-warmup-batches 0 \
  --reference-validation-interval 5 \
  --reference-validation-instances 8 \
  --reference-validation-candidates 4 \
  --checkpoint-interval 1 \
  --checkpoint-path outputs/train__fresh/best.pt \
  --output-dir "$output_dir" \
  2>&1 | tee -a "$output_dir/live.log"
