#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="$ROOT/.venv/bin/python"
CHECKPOINT_DIR="$ROOT/outputs/train__scratch_e500_lr1e4_min1e7_w128_qdhybrid_replay30_vectorized"
VALIDATION_DIR="$ROOT/validation/pomo_selection_20260921"
OUTPUT_DIR="$ROOT/outputs/pomo_selection_20260921"
CALIBRATION="$OUTPUT_DIR/runs/calibration_e100_e178_e296/results.json"

export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GRB_LICENSE_FILE=/home/marmot/wangjianing/gurobi.lic
ulimit -c 0
mkdir -p "$OUTPUT_DIR/runs"

evaluate() {
  local name="$1"
  shift
  local run_dir="$OUTPUT_DIR/runs/$name"
  mkdir -p "$run_dir"
  printf '%s\n' "$name" > "$OUTPUT_DIR/current_run.txt"
  local workers
  for workers in 4 2 1 1; do
    printf 'START %s workers=%s time=%s\n' "$name" "$workers" "$(date --iso-8601=seconds)" | tee -a "$run_dir/live.log"
    if taskset -c 0-15 "$PYTHON_BIN" -B -X faulthandler -u \
      -m scripts.benchmark_pomo_selection_holdout \
      --baseline "$CHECKPOINT_DIR/checkpoint-100.pt" \
      --baseline-label epoch100 \
      --fixed-validation-dir "$VALIDATION_DIR" \
      --output-dir "$run_dir" \
      --sizes 20 40 60 80 100 \
      --instances-per-size 200 \
      --batch-size 16 \
      --workers "$workers" \
      --device cpu \
      --seed 20260921 \
      --torch-threads 1 \
      --resume "$@" >> "$run_dir/live.log" 2>&1; then
      printf 'COMPLETE %s time=%s\n' "$name" "$(date --iso-8601=seconds)" | tee -a "$run_dir/live.log"
      "$PYTHON_BIN" -m scripts.summarize_pomo_selection_holdout \
        > "$OUTPUT_DIR/summary.log" 2>&1
      return 0
    fi
    printf 'RETRY %s after evaluation failure with workers=%s\n' "$name" "$workers" | tee -a "$run_dir/live.log"
  done
  printf 'FAILED %s after all worker counts\n' "$name" | tee -a "$run_dir/live.log"
  return 1
}

wait_for_checkpoint() {
  local epoch="$1"
  local path="$CHECKPOINT_DIR/checkpoint-$epoch.pt"
  local appeared=0
  while [[ ! -s "$path" ]]; do
    appeared=1
    if grep -Eq 'TRAIN_EXIT=[1-9]' "$CHECKPOINT_DIR/live.log"; then
      printf 'Training stopped before checkpoint %s appeared\n' "$epoch" >&2
      return 1
    fi
    sleep 60
  done
  # The trainer writes checkpoint files directly, so allow an in-progress save
  # to finish before hashing/loading a newly detected file.
  if [[ "$appeared" -eq 1 ]]; then
    sleep 15
  fi
}

on_exit() {
  local code="$?"
  if [[ "$code" -ne 0 ]]; then
    printf '{"status":"failed","finished_at":"%s","exit_code":%s}\n' \
      "$(date --iso-8601=seconds)" "$code" > "$OUTPUT_DIR/status.json"
  fi
}
trap on_exit EXIT

printf '{"status":"running","started_at":"%s"}\n' "$(date --iso-8601=seconds)" > "$OUTPUT_DIR/status.json"

evaluate calibration_e100_e178_e296 \
  --model "epoch178=$CHECKPOINT_DIR/checkpoint-178.pt" \
  --model "epoch296=$CHECKPOINT_DIR/checkpoint-296.pt"

# Dense checks around Epoch 100, then roughly every 20 epochs.  The frozen
# Epoch 100 scores from calibration are reused exactly for every paired run.
for epoch in 90 95 105 110 120 80 140 160 180 200 220 240 260 280 300 60; do
  wait_for_checkpoint "$epoch"
  evaluate "epoch$epoch" \
    --model "epoch$epoch=$CHECKPOINT_DIR/checkpoint-$epoch.pt" \
    --cached-baseline-results "$CALIBRATION"
done

for epoch in 320 340 360 380 400 420 440 460 480 500; do
  wait_for_checkpoint "$epoch"
  evaluate "epoch$epoch" \
    --model "epoch$epoch=$CHECKPOINT_DIR/checkpoint-$epoch.pt" \
    --cached-baseline-results "$CALIBRATION"
done

printf '{"status":"complete","finished_at":"%s"}\n' "$(date --iso-8601=seconds)" > "$OUTPUT_DIR/status.json"
rm -f "$OUTPUT_DIR/current_run.txt"
