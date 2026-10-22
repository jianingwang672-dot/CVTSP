#!/usr/bin/env bash
# B groups of the 2x2 reward x search study. Each B run copies its A run's
# launcher arguments and changes only the dense POMO reward source:
#   b1 <-> outputs/near_miss_quota_20261013/nm20      (swap+2-opt, near-miss 20%)
#   b2 <-> outputs/lkh_nm20_e100_20261014/train E1-20 (LKH, near-miss 20%)
set -euo pipefail
GROUP="${1:?group must be b1 or b2}"
GPU_ID="${2:-0}"
WORKERS="${3:-16}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
STUDY_DIR="$ROOT/outputs/random_subset_reward_20261015"
OUTPUT_DIR="${OUTPUT_DIR:-$STUDY_DIR/$GROUP}"
SESSION="cvtsp_rsr_${GROUP}"
LKH_BINARY="$ROOT/third_party/lkh3_20261014/LKH"

export CVTSP_VECTORIZED_LOCAL_SEARCH=1 CVTSP_TORCH_THREADS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-/home/marmot/wangjianing/gurobi.lic}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

common=( --training-profile e500 --ablation-group both --near-miss-ablation-rate .20
 --scheduler-total-epochs 100 --move-schedule-horizon 100 --move-schedule-final 30
 --train-instances 20000 --self-improvement-workers "$WORKERS" --cuda-device 0
 --reference-validation-per-size 400 --reference-validation-seed 20261010
 --reference-validation-candidates 1
 --validation-dir "$ROOT/validation/e500_logic_e100_20261010/validation2000"
 --diagnostic-dir "$ROOT/validation/search_ablation_20261012/diagnostic100"
 --seed 20260813 --output-dir "$OUTPUT_DIR"
 --dense-reward-source random_subset_exact
 --random-subset-min-routes 4 --random-subset-max-routes 8
 --gpu-gate-dir "$STUDY_DIR/.gpu_slots" )
case "$GROUP" in
  b1) A_DIR="$ROOT/outputs/near_miss_quota_20261013/nm20"
      extra=( --epochs 20 ) ;;
  b2) A_DIR="$ROOT/outputs/lkh_nm20_e100_20261014/train"
      extra=( --local-search-backend lkh --lkh-binary "$LKH_BINARY" --lkh-budget-coefficient 0.00015
              --lkh-desired-scale 1000000 --lkh-trials 1 --lkh-max-candidates 5 --lkh-move-type 5
              --epochs 20 --learning-rate 0.0001 --min-learning-rate 0.00001 --warmup-epochs 1
              --batch-size 64 --diagnostic-epochs 1 5 10 20 50 100 --checkpoint-interval 1 ) ;;
  *) echo "Invalid group: $GROUP" >&2; exit 1 ;;
esac
train_cmd=( "$PYTHON_BIN" -B -X faulthandler -u "$ROOT/train_joint_self_improvement.py" "${common[@]}" "${extra[@]}" )

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
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "Session exists: $SESSION" >&2; exit 1; fi
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing nonempty output: $OUTPUT_DIR" >&2; exit 1
fi
mkdir -p "$OUTPUT_DIR"
"${train_cmd[@]}" --print-config > "$OUTPUT_DIR/resolved_config.json"
# Only the reward source and run plumbing may differ from the paired A run.
"$PYTHON_BIN" -B -m scripts.compare_paired_config "$A_DIR/resolved_config.json" "$OUTPUT_DIR/resolved_config.json" \
  | tee "$OUTPUT_DIR/config_diff_vs_A.txt"
"$PYTHON_BIN" -B -m scripts.capture_search_ablation_provenance "$OUTPUT_DIR" \
  --launcher "$ROOT/scripts/train_random_subset_reward_tmux.sh" \
  --protocol "$ROOT/docs/random_subset_reward_experiment_20261015.md"
printf -v inner '%q ' env RUN_TRAINING_INNER=1 PYTHON_BIN="$PYTHON_BIN" OUTPUT_DIR="$OUTPUT_DIR" \
  bash "$ROOT/scripts/train_random_subset_reward_tmux.sh" "$GROUP" "$GPU_ID" "$WORKERS"
tmux new-session -d -s "$SESSION" -c "$ROOT" "$inner"
echo "Started $SESSION: GPU $GPU_ID -> cuda:0, workers=$WORKERS, paired A=$A_DIR"
