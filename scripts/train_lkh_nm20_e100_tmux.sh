#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:-0}"
WORKERS="${2:-32}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/lkh_nm20_e100_20261014/train}"
SESSION=cvtsp_lkh_nm20_e100
LKH_BINARY="$ROOT/third_party/lkh3_20261014/LKH"
VALIDATION_CANDIDATES="${VALIDATION_CANDIDATES:-1}"

export CVTSP_VECTORIZED_LOCAL_SEARCH=1 CVTSP_TORCH_THREADS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-/home/marmot/wangjianing/gurobi.lic}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

train_cmd=( "$PYTHON_BIN" -B -X faulthandler -u "$ROOT/train_joint_self_improvement.py"
 --training-profile e500 --ablation-group both --near-miss-ablation-rate .20
 --local-search-backend lkh --lkh-binary "$LKH_BINARY" --lkh-budget-coefficient 0.00015
 --lkh-desired-scale 1000000 --lkh-trials 1 --lkh-max-candidates 5 --lkh-move-type 5
 --epochs 100 --scheduler-total-epochs 100 --learning-rate 0.0001
 --min-learning-rate 0.00001 --warmup-epochs 1
 --move-schedule-horizon 100 --move-schedule-final 30
 --train-instances 20000 --batch-size 64 --self-improvement-workers "$WORKERS" --cuda-device 0
 --reference-validation-per-size 400 --reference-validation-seed 20261010
 --reference-validation-candidates "$VALIDATION_CANDIDATES"
 --validation-dir "$ROOT/validation/e500_logic_e100_20261010/validation2000"
 --diagnostic-dir "$ROOT/validation/search_ablation_20261012/diagnostic100"
 --diagnostic-epochs 1 5 10 20 50 100
 --seed 20260813 --checkpoint-interval 1 --output-dir "$OUTPUT_DIR" )

cd "$ROOT"
if [[ "${DRY_RUN:-0}" == 1 ]]; then "${train_cmd[@]}" --print-config; exit 0; fi
if [[ "${RUN_TRAINING_INNER:-0}" == 1 ]]; then
  ulimit -c 0
  set +e
  "${train_cmd[@]}" 2>&1 | tee -a "$OUTPUT_DIR/live.log"
  train_exit="${PIPESTATUS[0]}"
  printf '\nTRAIN_EXIT=%s time=%s\n' "$train_exit" "$(date --iso-8601=seconds)" | tee -a "$OUTPUT_DIR/live.log"
  exit "$train_exit"
fi

"$PYTHON_BIN" -B -m scripts.prepare_e500_logic_datasets --verify-only
"$PYTHON_BIN" -B -m scripts.prepare_search_ablation_diagnostics --verify-only
test -x "$LKH_BINARY"
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "Session already exists: $SESSION" >&2; exit 1; fi
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing nonempty output: $OUTPUT_DIR" >&2; exit 1
fi
mkdir -p "$OUTPUT_DIR"
"${train_cmd[@]}" --print-config > "$OUTPUT_DIR/resolved_config.json"
"$PYTHON_BIN" -B -m scripts.capture_search_ablation_provenance "$OUTPUT_DIR" \
  --launcher "$ROOT/scripts/train_lkh_nm20_e100_tmux.sh" \
  --protocol "$ROOT/docs/lkh_nm20_e100_training_20261014.md"
sha256sum "$LKH_BINARY" > "$OUTPUT_DIR/lkh_binary.sha256"
printf -v inner '%q ' env RUN_TRAINING_INNER=1 PYTHON_BIN="$PYTHON_BIN" OUTPUT_DIR="$OUTPUT_DIR" VALIDATION_CANDIDATES="$VALIDATION_CANDIDATES" \
  bash "$ROOT/scripts/train_lkh_nm20_e100_tmux.sh" "$GPU_ID" "$WORKERS"
tmux new-session -d -s "$SESSION" -c "$ROOT" "$inner"
echo "Started $SESSION: fresh100, GPU $GPU_ID -> cuda:0, workers=$WORKERS, LKH, near-miss20%, 20000/epoch"
