#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/marmot/wangjianing/self-improving cvtsp"
OUT="$ROOT/outputs/current_e200_pomo_selection_20260930"
RUN="$OUT/broad"
CURRENT="$ROOT/outputs/train__scratch_e200_i20k_lr1e4_il521_replay10_topk30to50"
OLD="$ROOT/outputs/train__scratch_e500_lr1e4_min1e7_w128_qdhybrid_replay30_vectorized"
HOLDOUT="$ROOT/validation/pomo_selection_20260921"
CACHE="$ROOT/outputs/pomo_selection_20260921/runs/calibration_e100_e178_e296/results.json"

cd "$ROOT"
mkdir -p "$RUN"
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GRB_LICENSE_FILE=/home/marmot/wangjianing/gurobi.lic
ulimit -c 0

printf '{"status":"running","phase":"broad","workers":16,"started_at":"%s"}\n' \
  "$(date --iso-8601=seconds)" > "$OUT/status.json"

status=0
nice -n 10 taskset -c 0-31 .venv/bin/python -B -X faulthandler -u \
  -m scripts.benchmark_pomo_selection_holdout \
  --baseline "$OLD/checkpoint-100.pt" \
  --baseline-label epoch100 \
  --model "current_e20=$CURRENT/checkpoint-20.pt" \
  --model "current_e40=$CURRENT/checkpoint-40.pt" \
  --model "current_e60=$CURRENT/checkpoint-60.pt" \
  --model "current_e80=$CURRENT/checkpoint-80.pt" \
  --model "current_e100=$CURRENT/checkpoint-100.pt" \
  --model "current_e120=$CURRENT/checkpoint-120.pt" \
  --model "current_e140=$CURRENT/checkpoint-140.pt" \
  --model "current_e160=$CURRENT/checkpoint-160.pt" \
  --model "current_e176=$CURRENT/checkpoint-176.pt" \
  --model "current_e180=$CURRENT/checkpoint-180.pt" \
  --fixed-validation-dir "$HOLDOUT" \
  --output-dir "$RUN" \
  --sizes 20 40 60 80 100 \
  --instances-per-size 200 \
  --batch-size 16 \
  --workers 16 \
  --device cpu \
  --seed 20260921 \
  --torch-threads 1 \
  --cached-baseline-results "$CACHE" \
  --resume > "$RUN/live.log" 2>&1 || status=$?

if [[ "$status" -ne 0 ]]; then
  printf '{"status":"failed","phase":"broad","exit_code":%s,"finished_at":"%s"}\n' \
    "$status" "$(date --iso-8601=seconds)" > "$OUT/status.json"
  exit "$status"
fi

.venv/bin/python -B - "$RUN/results.json" "$OUT/broad_ranking.json" <<'PY'
import json
import sys
from pathlib import Path

results = json.loads(Path(sys.argv[1]).read_text())
overall = results["overall"]
rows = []
for label, value in overall.items():
    if label.startswith("current_e"):
        rows.append({"label": label, "epoch": int(label.removeprefix("current_e")),
                     "mean_makespan": value["mean_makespan"]})
rows.sort(key=lambda row: row["mean_makespan"])
Path(sys.argv[2]).write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(json.dumps(rows, indent=2))
PY

printf '{"status":"broad_complete","phase":"broad","workers":16,"finished_at":"%s"}\n' \
  "$(date --iso-8601=seconds)" > "$OUT/status.json"
