#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="cvtsp_host_training_stress"
OUTPUT="$ROOT/outputs/host_training_stress_20260928"
PYTHON="$ROOT/.venv/bin/python"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
if [[ -d "$OUTPUT" ]] && find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "refusing to overwrite non-empty stress output: $OUTPUT" >&2
  exit 1
fi

mkdir -p "$OUTPUT"

cmd=(
  timeout 20m "$PYTHON" -B -X faulthandler -u "$ROOT/train_joint_self_improvement.py"
  --epochs 1
  --scheduler-total-epochs 200
  --learning-rate 1e-4
  --min-learning-rate 1e-7
  --warmup-epochs 1
  --train-instances 20000
  --batch-size 64
  --min-problem-size 20
  --max-problem-size 100
  --pomo-size 100
  --cuda-device 0
  --self-improvement-workers 32
  --top-k 30
  --top-k-min 30
  --top-k-ratio 0.5
  --top-k-diversity-count 10
  --top-k-diversity-pool-size 60
  --top-k-diversity-max-relative-gap 0.05
  --near-miss-rate 0.05
  --near-miss-min-rate 0.01
  --adaptive-near-miss
  --max-local-moves 30
  --reference-candidates 200
  --changed-expert-il-weight 5
  --imitation-inner-epochs 5
  --imitation-inner-epochs-schedule 1:5,51:2,101:1
  --rl-reward-mode hybrid_pomo_exact_expert
  --exact-expert-aux-max-weight 1.0
  --exact-expert-advantage-clip 1.0
  --competence-il-start 0.50
  --competence-il-full 0.85
  --competence-target-better-full 0.20
  --expert-replay-capacity 20000
  --expert-replay-fraction 0.10
  --imitation-micro-batch-size 128
  --reference-validation-per-size 1
  --reference-validation-candidates 1
  --validation-dir "$OUTPUT/validation"
  --checkpoint-interval 1
  --output-dir "$OUTPUT"
  --seed 20260813
)

printf -v quoted_cmd '%q ' "${cmd[@]}"
printf -v inner '%q ' env CUDA_VISIBLE_DEVICES=0 CVTSP_VECTORIZED_LOCAL_SEARCH=1 \
  CVTSP_TORCH_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  GRB_LICENSE_FILE=/home/marmot/wangjianing/gurobi.lic \
  bash -lc "cd '$ROOT' && ulimit -c 0 && taskset -c 0-31 $quoted_cmd 2>&1 | tee '$OUTPUT/live.log'; rc=\${PIPESTATUS[0]}; printf '\\nSTRESS_EXIT=%s time=%s\\n' \"\$rc\" \"\$(date --iso-8601=seconds)\" | tee -a '$OUTPUT/live.log'"

tmux new-session -d -s "$SESSION" -c "$ROOT" "$inner"
echo "Started: $SESSION"
echo "Log: $OUTPUT/live.log"
