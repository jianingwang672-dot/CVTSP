#!/usr/bin/env bash
set -euo pipefail
GROUP="${1:?group must be nm05, nm10, or nm20}"
GPU_ID="${2:-0}"
WORKERS="${3:-8}"
case "$GROUP" in nm05) RATE=0.05 ;; nm10) RATE=0.10 ;; nm20) RATE=0.20 ;; *) echo "Invalid group: $GROUP" >&2; exit 1 ;; esac
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/near_miss_quota_20261013/$GROUP}"
TMUX_SESSION="cvtsp_nearmiss_${GROUP}_e20"
export CVTSP_VECTORIZED_LOCAL_SEARCH=1
export CVTSP_TORCH_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-/home/marmot/wangjianing/gurobi.lic}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
train_cmd=(
  "$PYTHON_BIN" -B -X faulthandler -u "$ROOT/train_joint_self_improvement.py"
  --training-profile e500 --ablation-group both --near-miss-ablation-rate "$RATE"
  --epochs 20 --scheduler-total-epochs 100 --move-schedule-horizon 100 --move-schedule-final 30
  --train-instances 20000 --self-improvement-workers "$WORKERS" --cuda-device 0
  --reference-validation-per-size 400 --reference-validation-seed 20261010
  --validation-dir "$ROOT/validation/e500_logic_e100_20261010/validation2000"
  --diagnostic-dir "$ROOT/validation/search_ablation_20261012/diagnostic100"
  --seed 20260813 --output-dir "$OUTPUT_DIR"
)
cd "$ROOT"
if [[ "${DRY_RUN:-0}" == 1 ]]; then "${train_cmd[@]}" --print-config; exit 0; fi
if [[ "${RUN_TRAINING_INNER:-0}" == 1 ]]; then
  ulimit -c 0
  set +e
  "${train_cmd[@]}" 2>&1 | tee -a "$OUTPUT_DIR/live.log"
  train_exit="${PIPESTATUS[0]}"
  printf '\nTRAIN_EXIT=%s time=%s\n' "$train_exit" "$(date --iso-8601=seconds)" | tee -a "$OUTPUT_DIR/live.log"
  "$PYTHON_BIN" -B -m scripts.summarize_near_miss_quota --root "$(dirname "$OUTPUT_DIR")"
  exit "$train_exit"
fi
"$PYTHON_BIN" -B -m scripts.prepare_e500_logic_datasets --verify-only
"$PYTHON_BIN" -B -m scripts.prepare_search_ablation_diagnostics --verify-only
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then echo "Session exists: $TMUX_SESSION" >&2; exit 1; fi
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite nonempty output: $OUTPUT_DIR" >&2; exit 1
fi
mkdir -p "$OUTPUT_DIR"
"${train_cmd[@]}" --print-config > "$OUTPUT_DIR/resolved_config.json"
"$PYTHON_BIN" -B -m scripts.capture_search_ablation_provenance "$OUTPUT_DIR" \
  --launcher "$ROOT/scripts/train_near_miss_quota_tmux.sh" --protocol "$ROOT/docs/near_miss_quota_experiment_20261013.md"
printf -v inner '%q ' env RUN_TRAINING_INNER=1 PYTHON_BIN="$PYTHON_BIN" OUTPUT_DIR="$OUTPUT_DIR" \
  bash "$ROOT/scripts/train_near_miss_quota_tmux.sh" "$GROUP" "$GPU_ID" "$WORKERS"
tmux new-session -d -s "$TMUX_SESSION" -c "$ROOT" "$inner"
echo "Started $TMUX_SESSION: GPU $GPU_ID -> cuda:0, workers=$WORKERS, quota=$RATE, fixed moves=30"
