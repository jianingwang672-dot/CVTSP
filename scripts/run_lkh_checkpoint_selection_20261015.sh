#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/marmot/wangjianing/self-improving cvtsp"
TRAIN="$ROOT/outputs/lkh_nm20_e100_20261014/train"
OUT="$ROOT/outputs/lkh_nm20_checkpoint_selection_20261015/coarse"

cd "$ROOT"
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GRB_LICENSE_FILE=/home/marmot/wangjianing/gurobi.lic
ulimit -c 0

on_exit() {
  code=$?
  if [[ "$code" -ne 0 ]]; then
    printf '{"status":"failed","exit_code":%s,"time":"%s"}\n' \
      "$code" "$(date --iso-8601=seconds)" > "$OUT/status.json"
  fi
}
trap on_exit EXIT

printf '{"status":"running","stage":"coarse","started_at":"%s"}\n' \
  "$(date --iso-8601=seconds)" > "$OUT/status.json"

nice -n 15 taskset -c 0-31 .venv/bin/python -B -X faulthandler -u \
  scripts/benchmark_pomo_selection_holdout.py \
  --baseline "$TRAIN/checkpoint-5.pt" \
  --baseline-label epoch5 \
  --model "epoch10=$TRAIN/checkpoint-10.pt" \
  --model "epoch15=$TRAIN/checkpoint-15.pt" \
  --model "epoch24=$TRAIN/checkpoint-24.pt" \
  --model "epoch39=$TRAIN/checkpoint-39.pt" \
  --model "epoch46=$TRAIN/checkpoint-46.pt" \
  --model "epoch52=$TRAIN/checkpoint-52.pt" \
  --model "epoch59=$TRAIN/checkpoint-59.pt" \
  --model "epoch67=$TRAIN/checkpoint-67.pt" \
  --output-dir "$OUT" \
  --fixed-validation-dir validation/pomo_selection_20260921 \
  --sizes 20 40 60 80 100 \
  --instances-per-size 200 \
  --batch-size 16 \
  --workers 16 \
  --device cpu \
  --seed 20260921 \
  --torch-threads 1 \
  --resume > "$OUT/live.log" 2>&1

.venv/bin/python -B - "$OUT/results.json" "$OUT/selection.json" <<'PY'
import json
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:])
payload = json.loads(source.read_text())
labels = list(payload["params"]["checkpoints"])
rows = payload["records"]
ranking = []
for label in labels:
    values = [float(row["objectives"][label]) for row in rows]
    per_size = {}
    for size in payload["params"]["sizes"]:
        group = [float(row["objectives"][label]) for row in rows
                 if int(row["problem_size"]) == int(size)]
        per_size[str(size)] = sum(group) / len(group)
    ranking.append({
        "label": label,
        "epoch": int(payload["params"]["checkpoint_epochs"][label]),
        "mean_makespan": sum(values) / len(values),
        "per_size_mean": per_size,
    })
ranking.sort(key=lambda row: row["mean_makespan"])
result = {
    "purpose": "checkpoint_selection_not_final_test",
    "protocol": {
        "instances": len(rows),
        "sizes": payload["params"]["sizes"],
        "instances_per_size": payload["params"]["instances_per_size"],
        "decode_type": payload["params"]["decode_type"],
        "augmentation": payload["params"]["augmentation"],
        "pomo_size": payload["params"]["pomo_size"],
        "route_score": "minimum exact Gurobi makespan over all POMO routes",
    },
    "best_epoch": ranking[0]["epoch"],
    "ranking": ranking,
}
destination.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY

printf '{"status":"complete","stage":"coarse","finished_at":"%s"}\n' \
  "$(date --iso-8601=seconds)" > "$OUT/status.json"
