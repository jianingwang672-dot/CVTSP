#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/marmot/wangjianing/self-improving cvtsp"
OUT="$ROOT/outputs/eval_lkh_nm20_epoch59_external_20261015"
REAL="$OUT/real529"
SYNTH="$OUT/synthetic5000"
CHECKPOINT="$ROOT/outputs/lkh_nm20_e100_20261014/train/checkpoint-59.pt"
BASELINE="$ROOT/outputs/train__cvtsp_random_n100_nearest_quarter/checkpoint-200.pt"

cd "$ROOT"
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

nice -n 15 taskset -c 0-15 .venv/bin/python -B -X faulthandler -u \
  outputs/eval_vectorized_epoch100_vs_gurobi_epoch200_real529/benchmark_real_cached.py \
  --baseline "$BASELINE" \
  --baseline-label epoch200 \
  --model "epoch59=$CHECKPOINT" \
  --output-dir "$REAL" \
  --workers 16 \
  --seed 1234 \
  --pomo-size 100 \
  --start-node-strategy nearest_depot \
  --cached-results outputs/eval_epoch200_vs275_20260916/real529/results.json \
  --resume > "$REAL/live.log" 2>&1

current=synthetic5000
printf '{"status":"running","current":"synthetic5000","started_at":"%s"}\n' \
  "$(date --iso-8601=seconds)" > "$OUT/status.json"

nice -n 15 taskset -c 0-15 .venv/bin/python -B -X faulthandler -u -c \
  'import runpy; runpy.run_path("outputs/eval_epoch200_vs275_20260916/source_snapshot/benchmark_synthetic_checkpoints.py", run_name="__main__")' \
  --baseline "$BASELINE" \
  --baseline-label epoch200 \
  --model "epoch59=$CHECKPOINT" \
  --output-dir "$SYNTH" \
  --sizes 20 40 60 80 100 \
  --instances-per-size 1000 \
  --batch-size 64 \
  --workers 16 \
  --device cpu \
  --seed 20260827 \
  --torch-threads 1 \
  --cached-baseline-results outputs/eval_epoch200_vs275_20260916/synthetic5000/results.json \
  --resume > "$SYNTH/live.log" 2>&1

current=summary
.venv/bin/python -B - "$OUT" <<'PY' > "$OUT/summary.log" 2>&1
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
real = json.loads((root / "real529" / "results.json").read_text())
synth = json.loads((root / "synthetic5000" / "results.json").read_text())
trained_sizes = {20, 25, 30, 35, 40, 45, 50, 100}
real_subset = [r for r in real["records"] if int(r["problem_size"]) in trained_sizes]

def means(records):
    n = len(records)
    return {
        "num_instances": n,
        "epoch200_mean_makespan": sum(r["objectives"]["epoch200"] for r in records) / n,
        "epoch59_mean_makespan": sum(r["objectives"]["epoch59"] for r in records) / n,
    }

payload = {
    "checkpoint": "epoch59",
    "fixed_validation_raw_mean": 18.672909,
    "real529": real["overall"],
    "real529_J20_100_training_sizes": means(real_subset),
    "real529_by_size": real["summary_by_size"],
    "synthetic5000": synth["overall"],
    "synthetic5000_by_size": synth["summary_by_size"],
}
(root / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

printf '{"status":"complete","finished_at":"%s"}\n' \
  "$(date --iso-8601=seconds)" > "$OUT/status.json"
