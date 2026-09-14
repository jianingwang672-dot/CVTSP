#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
WORKERS="${2:-96}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TMUX_SESSION="${TMUX_SESSION:-cvtsp_e196_290}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="$PROJECT_ROOT/remote_resume/epoch196/checkpoints/checkpoint-195.pt"
VALIDATION_DIR="$PROJECT_ROOT/validation/macsim_fixed"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/train__resume_epoch196_to290_exactrl_replay30_diverse10}"

command -v tmux >/dev/null 2>&1 || {
  echo "tmux is not installed" >&2
  exit 1
}
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
}
[[ -f "$CHECKPOINT" ]] || {
  echo "Missing checkpoint: $CHECKPOINT (run git lfs pull)" >&2
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

mkdir -p "$OUTPUT_DIR"

train_cmd=(
  "$PYTHON_BIN" -u "$PROJECT_ROOT/train_joint_self_improvement.py"
  --epochs 95
  --learning-rate 1e-5
  --min-learning-rate 1e-6
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
  --near-miss-rate 0.05
  --max-local-moves 30
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
  --reset-main-scheduler-on-resume
  --output-dir "$OUTPUT_DIR"
  --seed 20260813
)

printf -v train_command '%q ' "${train_cmd[@]}"
printf -v tmux_command \
  'cd %q && export CUDA_VISIBLE_DEVICES=%q && %s 2>&1 | tee %q; printf "\\nTRAIN_EXIT=%%s\\n" "${PIPESTATUS[0]}"; exec bash' \
  "$PROJECT_ROOT" "$GPU_ID" "$train_command" "$OUTPUT_DIR/live.log"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN: $tmux_command"
  exit 0
fi

tmux new-session -d -s "$TMUX_SESSION" "$tmux_command"

echo "Started tmux session: $TMUX_SESSION"
echo "Physical GPU: $GPU_ID; workers: $WORKERS"
echo "Checkpoint: $CHECKPOINT (next epoch: 196; final epoch: 290)"
echo "Output: $OUTPUT_DIR"
echo "Attach: tmux attach -t $TMUX_SESSION"
echo "Log: tail -f '$OUTPUT_DIR/live.log'"
