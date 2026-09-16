#!/usr/bin/env bash
set -euo pipefail

GPU_A="${1:-6}"
GPU_B="${2:-7}"
WORKERS="${3:-48}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VALIDATION_DIR="${VALIDATION_DIR:-$PROJECT_ROOT/validation/macsim_fixed}"

command -v tmux >/dev/null 2>&1 || {
  echo "tmux is not installed" >&2
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

launch_branch() {
  local variant="$1"
  local gpu_id="$2"
  local perturbation_restarts="$3"
  local session="cvtsp_scratch_e1_${variant}"
  local output_dir="$PROJECT_ROOT/outputs/train__scratch_e1_${variant}"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 1
  fi
  if [[ -e "$output_dir" ]]; then
    echo "Output already exists; refusing to mix runs: $output_dir" >&2
    exit 1
  fi

  local -a train_cmd=(
    "$PYTHON_BIN" -u "$PROJECT_ROOT/train_joint_self_improvement.py"
    --epochs 1
    --learning-rate 1e-4
    --min-learning-rate 1e-7
    --warmup-epochs 3
    --scheduler-total-epochs 200
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
    --perturbation-restarts "$perturbation_restarts"
    --reference-candidates 200
    --changed-expert-il-weight 5
    --rl-reward-mode exact_target_expert
    --expert-replay-capacity 50000
    --expert-replay-fraction 0.30
    --reference-validation-per-size 200
    --reference-validation-candidates 1
    --validation-dir "$VALIDATION_DIR"
    --checkpoint-interval 1
    --output-dir "$output_dir"
    --seed 20260813
  )

  local train_command tmux_command
  printf -v train_command '%q ' "${train_cmd[@]}"
  printf -v tmux_command \
    'cd %q && export CUDA_VISIBLE_DEVICES=%q && %s 2>&1 | tee %q; printf "\nTRAIN_EXIT=%%s\n" "${PIPESTATUS[0]}"; exec bash' \
    "$PROJECT_ROOT" "$gpu_id" "$train_command" "$output_dir/live.log"

  mkdir -p "$output_dir"
  tmux new-session -d -s "$session" "$tmux_command"
  echo "Started $variant: session=$session physical_gpu=$gpu_id workers=$WORKERS"
  echo "Log: $output_dir/live.log"
}

launch_branch "oropt" "$GPU_A" 0
launch_branch "oropt_perturb" "$GPU_B" 1

echo "Both from-scratch Epoch-1 branches are running in parallel."
