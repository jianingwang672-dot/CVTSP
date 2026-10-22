#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="cvtsp_epoch100_controlled_branches"
OUTPUT="$ROOT/outputs/epoch100_controlled_branches"
SOURCE="$ROOT/outputs/train__scratch_e500_lr1e4_min1e7_w128_qdhybrid_replay30_vectorized"
PYTHON_BIN="$ROOT/.venv/bin/python"

if [[ "${RUN_INNER:-0}" == "1" ]]; then
  cd "$ROOT"
  mkdir -p "$OUTPUT"
  export PYTHONPATH="$ROOT"
  export CUDA_VISIBLE_DEVICES=0
  export CVTSP_VECTORIZED_LOCAL_SEARCH=1
  export CVTSP_TORCH_THREADS=1
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export GRB_LICENSE_FILE=/home/marmot/wangjianing/gurobi.lic
  ulimit -c 0
  resume_main() {
    local resume_epoch
    resume_epoch="$(cat "$OUTPUT/main_resume_epoch.txt")"
    printf 'RESUME_MAIN from_epoch=%s time=%s\n' "$resume_epoch" "$(date --iso-8601=seconds)" | tee -a "$OUTPUT/live.log"
    RESUME_EPOCH="$resume_epoch" \
    TMUX_SESSION="cvtsp_scratch_e500_resume${resume_epoch}_after_branches" \
    PYTHON_BIN="$PYTHON_BIN" \
    bash "$ROOT/scripts/resume_scratch_e500_from293_tmux.sh" 0 128 \
      2>&1 | tee -a "$OUTPUT/live.log"
  }
  for mode in il2 lr150; do
    printf 'START_BRANCH mode=%s time=%s\n' "$mode" "$(date --iso-8601=seconds)" | tee -a "$OUTPUT/live.log"
    set +e
    set -o pipefail
    taskset -c 16-31 "$PYTHON_BIN" -B -X faulthandler -u \
      scripts/run_epoch100_branch.py --mode "$mode" \
      2>&1 | tee -a "$OUTPUT/$mode/live.log" "$OUTPUT/live.log"
    code="${PIPESTATUS[0]}"
    set -e
    printf 'BRANCH_EXIT mode=%s code=%s\n' "$mode" "$code" | tee -a "$OUTPUT/live.log"
    if [[ "$code" != 0 ]]; then
      resume_main
      exec bash
    fi
    if [[ ! -s "$OUTPUT/$mode/train/checkpoint-120.pt" ]]; then
      echo "Missing E120 branch checkpoint for $mode" | tee -a "$OUTPUT/live.log"
      resume_main
      exec bash
    fi
  done
  resume_main
  printf 'START_EVALUATION time=%s\n' "$(date --iso-8601=seconds)" | tee -a "$OUTPUT/live.log"
  set +e
  set -o pipefail
  taskset -c 0-15 "$PYTHON_BIN" -B -X faulthandler -u \
    -m scripts.benchmark_pomo_selection_holdout \
    --baseline "$SOURCE/checkpoint-100.pt" \
    --baseline-label epoch100 \
    --model "il2_e120=$OUTPUT/il2/train/checkpoint-120.pt" \
    --model "lr150_e120=$OUTPUT/lr150/train/checkpoint-120.pt" \
    --cached-baseline-results "$ROOT/outputs/pomo_selection_20260921/runs/calibration_e100_e178_e296/results.json" \
    --fixed-validation-dir "$ROOT/validation/pomo_selection_20260921" \
    --output-dir "$OUTPUT/evaluation" \
    --sizes 20 40 60 80 100 --instances-per-size 200 \
    --batch-size 16 --workers 4 --device cpu --seed 20260921 \
    --torch-threads 1 --resume \
    2>&1 | tee -a "$OUTPUT/evaluation/live.log" "$OUTPUT/live.log"
  code="${PIPESTATUS[0]}"
  set -e
  printf 'EVALUATION_EXIT=%s\n' "$code" | tee -a "$OUTPUT/live.log"
  if [[ "$code" == 0 ]]; then
    "$PYTHON_BIN" -m scripts.summarize_epoch100_branches \
      2>&1 | tee -a "$OUTPUT/live.log"
    printf 'EXPERIMENTS_COMPLETE time=%s\n' "$(date --iso-8601=seconds)" | tee -a "$OUTPUT/live.log"
  fi
  exec bash
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Existing session: $SESSION" >&2
  exit 1
fi
mkdir -p "$OUTPUT/il2" "$OUTPUT/lr150" "$OUTPUT/evaluation"
test -s "$OUTPUT/main_resume_epoch.txt" || {
  echo "Missing completed main-training resume epoch marker" >&2
  exit 1
}
printf -v command '%q ' env RUN_INNER=1 bash "$ROOT/scripts/$(basename "$0")"
tmux new-session -d -s "$SESSION" -c "$ROOT" "$command"
echo "Started $SESSION; log $OUTPUT/live.log"
