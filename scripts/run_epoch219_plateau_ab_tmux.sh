#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-A}"
GPU_ID="${2:-4}"
WORKERS="${3:-96}"
CHECKPOINT="${4:-/home/marmot/wangjianing/self-improving cvtsp/outputs/train__resume_epoch196_to290_exactrl_replay30_diverse10/checkpoint-219.pt}"
EXTRA_EPOCHS="${EXTRA_EPOCHS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VALIDATION_DIR="${VALIDATION_DIR:-$PROJECT_ROOT/validation/macsim_fixed}"

case "${VARIANT^^}" in
  A)
    VARIANT_NAME="oropt"
    PERTURBATION_RESTARTS=0
    ;;
  B)
    VARIANT_NAME="oropt_perturb"
    PERTURBATION_RESTARTS=1
    ;;
  *)
    echo "Usage: $0 A|B [physical_gpu_id] [workers] [checkpoint-219.pt]" >&2
    exit 2
    ;;
esac

if ! [[ "$EXTRA_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXTRA_EPOCHS must be a positive integer" >&2
  exit 2
fi

END_EPOCH=$((219 + EXTRA_EPOCHS))
TMUX_SESSION="${TMUX_SESSION:-cvtsp_e219_${EXTRA_EPOCHS}ep_${VARIANT_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/train__e219_${EXTRA_EPOCHS}ep_${VARIANT_NAME}}"

command -v tmux >/dev/null 2>&1 || {
  echo "tmux is not installed" >&2
  exit 1
}
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
}
[[ -f "$CHECKPOINT" ]] || {
  echo "Missing Epoch 219 checkpoint: $CHECKPOINT" >&2
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

train_cmd=(
  "$PYTHON_BIN" -u "$PROJECT_ROOT/train_joint_self_improvement.py"
  --epochs "$EXTRA_EPOCHS"
  --learning-rate 1e-5
  --min-learning-rate 1e-6
  --warmup-epochs 1
  --scheduler-total-epochs 95
  --train-instances 10000
  --batch-size 64
  --min-problem-size 20
  --max-problem-size 100
  --pomo-size 100
  --cuda-device 0
  --self-improvement-workers "$WORKERS"
  --top-k 30
  --top-k-diversity-count 10
  --near-miss-rate 0.05
  --max-local-moves 30
  --move-types swap two_opt relocate or_opt_2 or_opt_3
  --perturbation-restarts "$PERTURBATION_RESTARTS"
  --reference-candidates 200
  --changed-expert-il-weight 5
  --rl-reward-mode exact_target_expert
  --expert-replay-capacity 50000
  --expert-replay-fraction 0.30
  --reference-validation-per-size 200
  --reference-validation-candidates 1
  --validation-dir "$VALIDATION_DIR"
  --checkpoint-interval 1
  --checkpoint-path "$CHECKPOINT"
  --resume-extra-epochs
  --output-dir "$OUTPUT_DIR"
  --seed 20260813
)

printf -v train_command '%q ' "${train_cmd[@]}"
printf -v tmux_command \
  'cd %q && export CUDA_VISIBLE_DEVICES=%q && %s 2>&1 | tee %q; printf "\nTRAIN_EXIT=%%s\n" "${PIPESTATUS[0]}"; exec bash' \
  "$PROJECT_ROOT" "$GPU_ID" "$train_command" "$OUTPUT_DIR/live.log"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN: $tmux_command"
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
tmux new-session -d -s "$TMUX_SESSION" "$tmux_command"

echo "Started Branch ${VARIANT^^} in tmux session: $TMUX_SESSION"
echo "Physical GPU: $GPU_ID; workers: $WORKERS"
echo "Checkpoint: $CHECKPOINT (Epoch 219 -> $END_EPOCH)"
echo "Output: $OUTPUT_DIR"
echo "Attach: tmux attach -t $TMUX_SESSION"
echo "Log: tail -f '$OUTPUT_DIR/live.log'"
