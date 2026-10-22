#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
WORKERS="${2:-128}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TMUX_SESSION="${TMUX_SESSION:-cvtsp_scratch_e500_qdhybrid_w128}"
CPU_AFFINITY="${CPU_AFFINITY:-16-31}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$PROJECT_ROOT/scripts/$(basename "${BASH_SOURCE[0]}")"
VALIDATION_DIR="$PROJECT_ROOT/validation/macsim_fixed"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/train__scratch_e500_lr1e4_min1e7_w128_qdhybrid_replay30}"

train_cmd=(
  "$PYTHON_BIN" -B -X faulthandler -u "$PROJECT_ROOT/train_joint_self_improvement.py"
  --epochs 500
  --learning-rate 1e-4
  --min-learning-rate 1e-7
  --warmup-epochs 1
  --train-instances 10000
  --batch-size 64
  --min-problem-size 20
  --max-problem-size 100
  --pomo-size 100
  --cuda-device 0
  --self-improvement-workers "$WORKERS"
  --top-k 30
  --top-k-diversity-count 10
  --top-k-diversity-pool-size 60
  --top-k-diversity-max-relative-gap 0.05
  --near-miss-rate 0.05
  --near-miss-min-rate 0.01
  --adaptive-near-miss
  --max-local-moves 30
  --reference-candidates 200
  --changed-expert-il-weight 5
  --rl-reward-mode hybrid_pomo_exact_expert
  --exact-expert-aux-max-weight 1.0
  --exact-expert-advantage-clip 1.0
  --competence-il-start 0.50
  --competence-il-full 0.85
  --competence-target-better-full 0.20
  --expert-replay-capacity 50000
  --expert-replay-fraction 0.30
  --imitation-micro-batch-size 128
  --reference-validation-per-size 200
  --reference-validation-candidates 1
  --validation-dir "$VALIDATION_DIR"
  --checkpoint-interval 1
  --output-dir "$OUTPUT_DIR"
  --seed 20260813
)

if [[ "${RUN_TRAINING_INNER:-0}" == "1" ]]; then
  cd "$PROJECT_ROOT"
  mkdir -p "$OUTPUT_DIR"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  set +e
  set -o pipefail
  taskset -c "$CPU_AFFINITY" "${train_cmd[@]}" 2>&1 | tee -a "$OUTPUT_DIR/live.log"
  train_exit="${PIPESTATUS[0]}"
  set -e
  printf '\nTRAIN_EXIT=%s\n' "$train_exit"
  exec bash
fi

command -v tmux >/dev/null 2>&1 || {
  echo "tmux is not installed" >&2
  exit 1
}
command -v taskset >/dev/null 2>&1 || {
  echo "taskset is not installed" >&2
  exit 1
}
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
}
[[ -f "$VALIDATION_DIR/J100.pt" ]] || {
  echo "Missing fixed validation data: $VALIDATION_DIR" >&2
  exit 1
}
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "tmux session already exists: $TMUX_SESSION" >&2
  exit 1
fi
if [[ -d "$OUTPUT_DIR" ]] && find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "Refusing to overwrite non-empty from-scratch output: $OUTPUT_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
printf -v inner_command '%q ' \
  env \
  RUN_TRAINING_INNER=1 \
  GPU_ID="$GPU_ID" \
  WORKERS="$WORKERS" \
  PYTHON_BIN="$PYTHON_BIN" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  CPU_AFFINITY="$CPU_AFFINITY" \
  bash "$SCRIPT_PATH"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY_RUN: %s\n' "$inner_command"
  printf 'TRAIN: '
  printf '%q ' "${train_cmd[@]}"
  printf '\n'
  exit 0
fi

tmux new-session -d -s "$TMUX_SESSION" -c "$PROJECT_ROOT" "$inner_command"

echo "Started tmux session: $TMUX_SESSION"
echo "Physical GPU: $GPU_ID (Python logical cuda:0); workers: $WORKERS"
echo "CPU affinity: $CPU_AFFINITY"
echo "From scratch: epoch 1; final epoch: 500"
echo "Output: $OUTPUT_DIR"
echo "Attach: tmux attach -t $TMUX_SESSION"
echo "Log: tail -f '$OUTPUT_DIR/live.log'"
