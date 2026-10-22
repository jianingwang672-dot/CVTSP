#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="$ROOT/outputs/eval_sample_e200_vs_e365"
SESSION="cvtsp_sample_e200_e365_external"
PYTHON_BIN="$ROOT/.venv/bin/python"

if [[ "${RUN_INNER:-0}" == "1" ]]; then
  cd "$ROOT"
  mkdir -p "$OUTPUT/real529" "$OUTPUT/synthetic5000"
  export PYTHONPATH="$ROOT"
  export CUDA_VISIBLE_DEVICES=
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export GRB_LICENSE_FILE=/home/marmot/wangjianing/gurobi.lic
  ulimit -c 0
  printf '{"status":"running","started_at":"%s"}\n' "$(date --iso-8601=seconds)" > "$OUTPUT/status.json"
  for dataset in real529 synthetic5000; do
    complete=0
    for workers in 4 2 1; do
      printf 'START dataset=%s workers=%s time=%s\n' "$dataset" "$workers" "$(date --iso-8601=seconds)" \
        | tee -a "$OUTPUT/live.log"
      set +e
      set -o pipefail
      taskset -c 0-15 "$PYTHON_BIN" -B -X faulthandler -u \
        -m scripts.benchmark_sample_external \
        --dataset "$dataset" --output-dir "$OUTPUT/$dataset" \
        --workers "$workers" --save-interval 10 --resume \
        2>&1 | tee -a "$OUTPUT/$dataset/live.log" "$OUTPUT/live.log"
      code="${PIPESTATUS[0]}"
      set -e
      if [[ "$code" == 0 ]]; then
        complete=1
        break
      fi
      printf 'RETRY dataset=%s after exit=%s\n' "$dataset" "$code" | tee -a "$OUTPUT/live.log"
    done
    if [[ "$complete" != 1 ]]; then
      printf '{"status":"failed","dataset":"%s","finished_at":"%s"}\n' \
        "$dataset" "$(date --iso-8601=seconds)" > "$OUTPUT/status.json"
      printf 'FAILED dataset=%s\n' "$dataset" | tee -a "$OUTPUT/live.log"
      exec bash
    fi
  done
  "$PYTHON_BIN" -m scripts.summarize_sample_external 2>&1 | tee -a "$OUTPUT/live.log"
  printf '{"status":"complete","finished_at":"%s"}\n' "$(date --iso-8601=seconds)" > "$OUTPUT/status.json"
  printf 'EVALUATION_COMPLETE time=%s\n' "$(date --iso-8601=seconds)" | tee -a "$OUTPUT/live.log"
  exec bash
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Existing session: $SESSION" >&2
  exit 1
fi
mkdir -p "$OUTPUT"
printf -v command '%q ' env RUN_INNER=1 bash "$ROOT/scripts/$(basename "$0")"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY_RUN: %s\n' "$command"
  exit 0
fi
tmux new-session -d -s "$SESSION" -c "$ROOT" "$command"
echo "Started $SESSION; log $OUTPUT/live.log"
