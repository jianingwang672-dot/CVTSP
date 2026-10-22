#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/marmot/wangjianing/self-improving cvtsp"
TRAIN="$ROOT/outputs/lkh_nm20_e100_20261014/train"
BASE="$ROOT/outputs/lkh_nm20_checkpoint_selection_20261015"
COARSE="$BASE/coarse"
OUT="$BASE/refine"

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

printf '{"status":"running","stage":"refine","started_at":"%s"}\n' \
  "$(date --iso-8601=seconds)" > "$OUT/status.json"

nice -n 15 taskset -c 0-31 .venv/bin/python -B -X faulthandler -u \
  scripts/benchmark_pomo_selection_holdout.py \
  --baseline "$TRAIN/checkpoint-46.pt" \
  --baseline-label epoch46 \
  --model "epoch38=$TRAIN/checkpoint-38.pt" \
  --model "epoch40=$TRAIN/checkpoint-40.pt" \
  --model "epoch42=$TRAIN/checkpoint-42.pt" \
  --model "epoch44=$TRAIN/checkpoint-44.pt" \
  --model "epoch45=$TRAIN/checkpoint-45.pt" \
  --model "epoch47=$TRAIN/checkpoint-47.pt" \
  --model "epoch48=$TRAIN/checkpoint-48.pt" \
  --model "epoch50=$TRAIN/checkpoint-50.pt" \
  --output-dir "$OUT" \
  --fixed-validation-dir validation/pomo_selection_20260921 \
  --sizes 20 40 60 80 100 \
  --instances-per-size 200 \
  --batch-size 16 \
  --workers 16 \
  --device cpu \
  --seed 20260921 \
  --torch-threads 1 \
  --cached-baseline-results "$COARSE/results.json" \
  --resume > "$OUT/live.log" 2>&1

.venv/bin/python -B - "$COARSE/results.json" "$OUT/results.json" "$BASE/refined_selection.json" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

coarse_path, refine_path, destination = map(Path, sys.argv[1:])
payloads = [json.loads(coarse_path.read_text()), json.loads(refine_path.read_text())]
fingerprints = [[row["instance_fingerprint"] for row in p["records"]] for p in payloads]
if fingerprints[0] != fingerprints[1]:
    raise RuntimeError("coarse and refinement instance order differs")
rows_by_label = {}
epochs = {}
sizes = np.asarray([int(row["problem_size"]) for row in payloads[0]["records"]])
for payload in payloads:
    for label, epoch in payload["params"]["checkpoint_epochs"].items():
        values = np.asarray([float(row["objectives"][label]) for row in payload["records"]])
        if label in rows_by_label and not np.array_equal(values, rows_by_label[label]):
            raise RuntimeError(f"cached values differ for {label}")
        rows_by_label[label] = values
        epochs[label] = int(epoch)
ranking = sorted(rows_by_label, key=lambda label: float(rows_by_label[label].mean()))
best = ranking[0]
rng = np.random.default_rng(20261015)
strata = [np.flatnonzero(sizes == size) for size in sorted(set(sizes))]
details = []
for label in ranking:
    values = rows_by_label[label]
    difference = values - rows_by_label[best]
    samples = []
    for _ in range(3000):
        selected = np.concatenate([group[rng.integers(len(group), size=len(group))]
                                   for group in strata])
        samples.append(float(difference[selected].mean()))
    details.append({
        "label": label,
        "epoch": epochs[label],
        "mean_makespan": float(values.mean()),
        "difference_from_best": float(difference.mean()),
        "difference_from_best_ci95": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
        "better_tie_worse_than_best": [
            int(np.count_nonzero(difference < -1e-6)),
            int(np.count_nonzero(np.abs(difference) <= 1e-6)),
            int(np.count_nonzero(difference > 1e-6)),
        ],
        "per_size_mean": {str(size): float(values[sizes == size].mean())
                          for size in sorted(set(sizes))},
    })
result = {
    "purpose": "checkpoint_selection_not_final_test",
    "instances": len(sizes),
    "best_epoch": epochs[best],
    "ranking": details,
}
destination.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY

printf '{"status":"complete","stage":"refine","finished_at":"%s"}\n' \
  "$(date --iso-8601=seconds)" > "$OUT/status.json"
