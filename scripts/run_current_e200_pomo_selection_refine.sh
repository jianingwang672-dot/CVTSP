#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/marmot/wangjianing/self-improving cvtsp"
OUT="$ROOT/outputs/current_e200_pomo_selection_20260930"
BROAD="$OUT/broad"
RUN="$OUT/refine"
CURRENT="$ROOT/outputs/train__scratch_e200_i20k_lr1e4_il521_replay10_topk30to50"
OLD="$ROOT/outputs/train__scratch_e500_lr1e4_min1e7_w128_qdhybrid_replay30_vectorized"
HOLDOUT="$ROOT/validation/pomo_selection_20260921"
CACHE="$ROOT/outputs/pomo_selection_20260921/runs/calibration_e100_e178_e296/results.json"

cd "$ROOT"
mkdir -p "$RUN"

while [[ ! -f "$OUT/status.json" ]] || ! grep -q '"status":"broad_complete"' "$OUT/status.json"; do
  if [[ -f "$OUT/status.json" ]] && grep -q '"status":"failed"' "$OUT/status.json"; then
    exit 1
  fi
  sleep 30
done

best_epoch="$(.venv/bin/python -B - "$OUT/broad_ranking.json" <<'PY'
import json
import sys
from pathlib import Path

rows = json.loads(Path(sys.argv[1]).read_text())
print(rows[0]["epoch"])
PY
)"

mapfile -t refine_epochs < <(.venv/bin/python -B - "$best_epoch" "$CURRENT" <<'PY'
import sys
from pathlib import Path

best = int(sys.argv[1])
root = Path(sys.argv[2])
for epoch in sorted({best - 15, best - 10, best - 5, best + 5, best + 10, best + 15}):
    if epoch >= 1 and (root / f"checkpoint-{epoch}.pt").is_file():
        print(epoch)
PY
)

model_args=()
for epoch in "${refine_epochs[@]}"; do
  model_args+=(--model "current_e${epoch}=$CURRENT/checkpoint-${epoch}.pt")
done

export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GRB_LICENSE_FILE=/home/marmot/wangjianing/gurobi.lic
ulimit -c 0

printf '{"status":"running","phase":"refine","workers":16,"broad_best_epoch":%s,"started_at":"%s"}\n' \
  "$best_epoch" "$(date --iso-8601=seconds)" > "$OUT/status.json"

status=0
nice -n 10 taskset -c 0-31 .venv/bin/python -B -X faulthandler -u \
  -m scripts.benchmark_pomo_selection_holdout \
  --baseline "$OLD/checkpoint-100.pt" \
  --baseline-label epoch100 \
  "${model_args[@]}" \
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
  printf '{"status":"failed","phase":"refine","exit_code":%s,"finished_at":"%s"}\n' \
    "$status" "$(date --iso-8601=seconds)" > "$OUT/status.json"
  exit "$status"
fi

.venv/bin/python -B - "$BROAD/results.json" "$RUN/results.json" "$OUT/final_selection.json" <<'PY'
import json
import sys
from pathlib import Path

broad = json.loads(Path(sys.argv[1]).read_text())
refine = json.loads(Path(sys.argv[2]).read_text())
rows = {}
per_size = {}
for result in (broad, refine):
    for label, value in result["overall"].items():
        if label.startswith("current_e"):
            epoch = int(label.removeprefix("current_e"))
            rows[epoch] = float(value["mean_makespan"])
    for size, summary in result["summary_by_size"].items():
        per_size.setdefault(size, {})
        for label, value in summary.items():
            if label.startswith("current_e"):
                epoch = int(label.removeprefix("current_e"))
                per_size[size][epoch] = float(value["mean_makespan"])

ranking = [
    {"epoch": epoch, "mean_makespan": mean}
    for epoch, mean in sorted(rows.items(), key=lambda item: (item[1], item[0]))
]
best = ranking[0]
best_epoch = best["epoch"]
payload = {
    "status": "complete",
    "selected_epoch": best_epoch,
    "selected_mean_makespan": best["mean_makespan"],
    "selected_per_size": {
        size: values[best_epoch] for size, values in per_size.items()
    },
    "ranking": ranking,
    "old_run_epoch100_mean_makespan": float(
        broad["overall"]["epoch100_mean_makespan"]
    ),
}
Path(sys.argv[3]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

printf '{"status":"complete","phase":"refine","workers":16,"finished_at":"%s"}\n' \
  "$(date --iso-8601=seconds)" > "$OUT/status.json"
