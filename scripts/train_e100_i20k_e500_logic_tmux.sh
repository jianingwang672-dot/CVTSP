#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-${GPU_ID:-0}}"
WORKERS="${2:-${WORKERS:-128}}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TMUX_SESSION="${TMUX_SESSION:-cvtsp_e100_i20k_e500logic}"
DATA_ROOT="$ROOT/validation/e500_logic_e100_20261010"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/train__e100_i20k_e500logic_val2000_test500}"

export CVTSP_VECTORIZED_LOCAL_SEARCH=1
export CVTSP_TORCH_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-/home/marmot/wangjianing/gurobi.lic}"

train_cmd=(
  "$PYTHON_BIN" -B -X faulthandler -u "$ROOT/train_joint_self_improvement.py"
  --training-profile e500
  --reference-validation-candidates 1
  --epochs 100 --scheduler-total-epochs 100
  --train-instances 20000
  --self-improvement-workers "$WORKERS" --cuda-device 0
  --reference-validation-per-size 400
  --reference-validation-seed 20261010
  --validation-dir "$DATA_ROOT/validation2000"
  --seed 20260813 --output-dir "$OUTPUT_DIR"
)
cd "$ROOT"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  "${train_cmd[@]}" --print-config
  exit 0
fi

if [[ "${RUN_TRAINING_INNER:-0}" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  ulimit -c 0
  set +e
  "${train_cmd[@]}" 2>&1 | tee -a "$OUTPUT_DIR/live.log"
  train_exit="${PIPESTATUS[0]}"
  printf '\nTRAIN_EXIT=%s time=%s\n' "$train_exit" "$(date --iso-8601=seconds)" | tee -a "$OUTPUT_DIR/live.log"
  exit "$train_exit"
fi

command -v tmux >/dev/null
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }
"$PYTHON_BIN" -B -m scripts.prepare_e500_logic_datasets --verify-only
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "Session already exists: $TMUX_SESSION" >&2; exit 1
fi
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite nonempty output: $OUTPUT_DIR" >&2; exit 1
fi
mkdir -p "$OUTPUT_DIR"
"${train_cmd[@]}" --print-config > "$OUTPUT_DIR/resolved_config.json"
printf -v inner '%q ' env RUN_TRAINING_INNER=1 GPU_ID="$GPU_ID" WORKERS="$WORKERS" \
  PYTHON_BIN="$PYTHON_BIN" OUTPUT_DIR="$OUTPUT_DIR" bash "$ROOT/scripts/train_e100_i20k_e500_logic_tmux.sh"
tmux new-session -d -s "$TMUX_SESSION" -c "$ROOT" "$inner"
echo "Started $TMUX_SESSION: physical GPU $GPU_ID -> logical cuda:0; workers $WORKERS"
echo "Log: $OUTPUT_DIR/live.log"
