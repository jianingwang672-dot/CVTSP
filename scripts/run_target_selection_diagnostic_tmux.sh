#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="cvtsp_target_selection_diagnostic"
OUTPUT="$ROOT/outputs/target_selection_diagnostic_20260922"
PYTHON_BIN="$ROOT/.venv/bin/python"

if [[ "${RUN_INNER:-0}" == "1" ]]; then
  cd "$ROOT"
  mkdir -p "$OUTPUT"
  export PYTHONPATH="$ROOT"
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export CVTSP_VECTORIZED_LOCAL_SEARCH=1
  set +e
  set -o pipefail
  taskset -c 0-15 "$PYTHON_BIN" -B -X faulthandler -u \
    scripts/diagnose_target_route_selection.py \
    --checkpoint "e100=$ROOT/outputs/train__scratch_e500_lr1e4_min1e7_w128_qdhybrid_replay30_vectorized/checkpoint-100.pt" \
    --checkpoint "e365=$ROOT/outputs/train__scratch_e500_lr1e4_min1e7_w128_qdhybrid_replay30_vectorized/checkpoint-365.pt" \
    --output-dir "$OUTPUT" --per-size 40 --seed 20260922 \
    --batch-size 8 --workers 4 2>&1 | tee -a "$OUTPUT/live.log"
  code="${PIPESTATUS[0]}"
  printf '\nDIAGNOSTIC_EXIT=%s\n' "$code"
  exec bash
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Existing session: $SESSION" >&2
  exit 1
fi
printf -v command '%q ' env RUN_INNER=1 bash "$ROOT/scripts/$(basename "$0")"
tmux new-session -d -s "$SESSION" -c "$ROOT" "$command"
echo "Started $SESSION; log $OUTPUT/live.log"
