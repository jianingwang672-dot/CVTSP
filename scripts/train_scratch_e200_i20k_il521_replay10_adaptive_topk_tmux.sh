#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-${GPU_ID:-0}}"
WORKERS="${2:-${WORKERS:-128}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TMUX_SESSION="${TMUX_SESSION:-cvtsp_scratch_e200_i20k_il521}"
CPU_AFFINITY="${CPU_AFFINITY:-0-31}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$PROJECT_ROOT/scripts/$(basename "${BASH_SOURCE[0]}")"
VALIDATION_DIR="$PROJECT_ROOT/validation/macsim_fixed"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/train__scratch_e200_i20k_lr1e4_il521_replay10_topk30to50}"

train_cmd=(
  "$PYTHON_BIN" -B -X faulthandler -u "$PROJECT_ROOT/train_joint_self_improvement.py"
  --epochs 200
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
  --self-improvement-workers "$WORKERS"
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
  export CVTSP_VECTORIZED_LOCAL_SEARCH=1
  export CVTSP_TORCH_THREADS="${CVTSP_TORCH_THREADS:-1}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-/home/marmot/wangjianing/gurobi.lic}"
  ulimit -c 0
  set +e
  set -o pipefail
  taskset -c "$CPU_AFFINITY" "${train_cmd[@]}" 2>&1 | tee -a "$OUTPUT_DIR/live.log"
  train_exit="${PIPESTATUS[0]}"
  set -e
  printf '\nTRAIN_EXIT=%s time=%s\n' "$train_exit" "$(date --iso-8601=seconds)" \
    | tee -a "$OUTPUT_DIR/live.log"
  exec bash
fi

command -v tmux >/dev/null 2>&1 || { echo "tmux is not installed" >&2; exit 1; }
command -v taskset >/dev/null 2>&1 || { echo "taskset is not installed" >&2; exit 1; }
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
cat > "$OUTPUT_DIR/PROTOCOL.md" <<'EOF'
# From-scratch E200 / 20k protocol

- Random initialization; no checkpoint or historical model is loaded.
- 200 epochs, 20,000 generated instances per epoch, J uniformly sampled from 20–100, batch 64, POMO=J.
- Adam LR 1e-4 with one warmup epoch and epoch-based cosine decay to 1e-7 at E200.
- IL passes are predeclared as E1–50: 5, E51–100: 2, E101–200: 1.
- Strictly improved Expert routes have weight 5. Replay uses 10% of IL examples and a FIFO capacity of 20,000 improved Experts.
- Local-search Top-K is max(30, ceil(0.5J)): 30 at J<=60, 40 at J80, and 50 at J100. Ten starts remain quality-gated diverse starts.
- All other hybrid dense-POMO/exact-Expert, near-miss, Gurobi, validation, vectorized local-search, and checkpoint settings follow the completed E500 experiment.
EOF
printf -v inner_command '%q ' env RUN_TRAINING_INNER=1 GPU_ID="$GPU_ID" \
  WORKERS="$WORKERS" PYTHON_BIN="$PYTHON_BIN" OUTPUT_DIR="$OUTPUT_DIR" \
  CPU_AFFINITY="$CPU_AFFINITY" bash "$SCRIPT_PATH"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY_RUN: %s\n' "$inner_command"
  printf 'TRAIN: '; printf '%q ' "${train_cmd[@]}"; printf '\n'
  exit 0
fi

tmux new-session -d -s "$TMUX_SESSION" -c "$PROJECT_ROOT" "$inner_command"
echo "Started tmux session: $TMUX_SESSION"
echo "Log: $OUTPUT_DIR/live.log"
