#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="$ROOT/outputs/eval_vectorized_epoch365_external"
REAL="$OUT/real529"
SYNTH="$OUT/synthetic5000"
CHECKPOINT="outputs/train__scratch_e500_lr1e4_min1e7_w128_qdhybrid_replay30_vectorized/checkpoint-365.pt"
BASELINE="outputs/train__cvtsp_random_n100_nearest_quarter/checkpoint-200.pt"
mkdir -p "$REAL" "$SYNTH"

export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GRB_LICENSE_FILE=/home/marmot/wangjianing/gurobi.lic
ulimit -c 0

current=real529
on_exit() {
  code=$?
  if [[ "$code" -ne 0 ]]; then
    printf '{"status":"failed","current":"%s","exit_code":%s,"time":"%s"}\n' \
      "$current" "$code" "$(date --iso-8601=seconds)" > "$OUT/status.json"
  fi
}
trap on_exit EXIT

printf '{"status":"running","current":"real529","started_at":"%s"}\n' \
  "$(date --iso-8601=seconds)" > "$OUT/status.json"

taskset -c 0-15 .venv/bin/python -B -X faulthandler -u \
  outputs/eval_vectorized_epoch100_vs_gurobi_epoch200_real529/benchmark_real_cached.py \
  --baseline "$BASELINE" \
  --baseline-label epoch200 \
  --model "epoch365=$CHECKPOINT" \
  --output-dir "$REAL" \
  --workers 4 \
  --seed 1234 \
  --pomo-size 100 \
  --start-node-strategy nearest_depot \
  --cached-results outputs/eval_epoch200_vs275_20260916/real529/results.json \
  --resume > "$REAL/live.log" 2>&1

current=synthetic5000
printf '{"status":"running","current":"synthetic5000","started_at":"%s"}\n' \
  "$(date --iso-8601=seconds)" > "$OUT/status.json"

taskset -c 0-15 .venv/bin/python -B -X faulthandler -u -c \
  'import runpy; runpy.run_path("outputs/eval_epoch200_vs275_20260916/source_snapshot/benchmark_synthetic_checkpoints.py", run_name="__main__")' \
  --baseline "$BASELINE" \
  --baseline-label epoch200 \
  --model "epoch365=$CHECKPOINT" \
  --output-dir "$SYNTH" \
  --sizes 20 40 60 80 100 \
  --instances-per-size 1000 \
  --batch-size 64 \
  --workers 4 \
  --device cpu \
  --seed 20260827 \
  --torch-threads 1 \
  --cached-baseline-results outputs/eval_epoch200_vs275_20260916/synthetic5000/results.json \
  --resume > "$SYNTH/live.log" 2>&1

current=summary
.venv/bin/python -B scripts/summarize_epoch365_external.py > "$OUT/summary.log" 2>&1
printf '{"status":"complete","finished_at":"%s"}\n' \
  "$(date --iso-8601=seconds)" > "$OUT/status.json"
