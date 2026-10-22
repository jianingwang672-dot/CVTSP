#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-${GPU_ID:-0}}"
# Defaults sized for this 32-thread host: one single-threaded Gurobi/search
# worker per logical CPU. A 2026-10-19 benchmark (first 32 batches, two
# repeats each) measured 32 workers ~2.5% faster than 30 with 13.5 GB peak RSS
# and no swap. The archived run used 128 workers pinned to 16-31 on a larger
# host; the worker count changes only throughput, not the per-batch results.
WORKERS="${2:-${WORKERS:-32}}"
TMUX_SESSION="${TMUX_SESSION:-cvtsp_scratch_e500_restored}"
CPU_AFFINITY="${CPU_AFFINITY:-0-$(( $(nproc) - 1 ))}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
SCRIPT_PATH="$PROJECT_ROOT/scripts/$(basename "${BASH_SOURCE[0]}")"
VALIDATION_DIR="$PROJECT_ROOT/validation/macsim_fixed"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/train__scratch_e500_restored}"

train_cmd=(
  "$PYTHON_BIN" -B -X faulthandler -u "$PROJECT_ROOT/train_joint_self_improvement.py"
  --training-profile e500
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
  --local-search-backend swap_two_opt
  --move-types swap two_opt
  --reference-candidates 200
  --changed-expert-il-weight 5
  --rl-reward-mode hybrid_pomo_exact_expert
  --dense-reward-source hybrid_upper_bound
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
  --reference-validation-seed 20352150
  --validation-dir "$VALIDATION_DIR"
  --checkpoint-interval 1
  --output-dir "$OUTPUT_DIR"
  --seed 20260813
)

cd "$PROJECT_ROOT"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  "${train_cmd[@]}" --print-config
  exit 0
fi

if [[ "${RUN_TRAINING_INNER:-0}" == "1" ]]; then
  cd "$PROJECT_ROOT"
  mkdir -p "$OUTPUT_DIR"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  export CVTSP_VECTORIZED_LOCAL_SEARCH=1
  export CVTSP_TORCH_THREADS="${CVTSP_TORCH_THREADS:-1}"
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
if (( WORKERS > $(nproc) )); then
  echo "Warning: $WORKERS workers exceed $(nproc) logical CPUs; expect oversubscription" >&2
fi
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
"${train_cmd[@]}" --print-config > "$OUTPUT_DIR/resolved_config.json"
printf -v inner_command '%q ' \
  env \
  RUN_TRAINING_INNER=1 \
  GPU_ID="$GPU_ID" \
  WORKERS="$WORKERS" \
  PYTHON_BIN="$PYTHON_BIN" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  CPU_AFFINITY="$CPU_AFFINITY" \
  bash "$SCRIPT_PATH"

tmux new-session -d -s "$TMUX_SESSION" -c "$PROJECT_ROOT" "$inner_command"

echo "Started tmux session: $TMUX_SESSION"
echo "Physical GPU: $GPU_ID (Python logical cuda:0); workers: $WORKERS"
echo "CPU affinity: $CPU_AFFINITY"
echo "Vectorized local search: enabled"
echo "From scratch: epoch 1; final epoch: 500"
echo "Output: $OUTPUT_DIR"
echo "Attach: tmux attach -t $TMUX_SESSION"
echo "Log: tail -f '$OUTPUT_DIR/live.log'"
