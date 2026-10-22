#!/usr/bin/env bash
# Launch the progressive-recipe from-scratch run (S500 LR schedule, 200 epochs).
set -euo pipefail
GPU_ID="${1:-0}"
WORKERS="${2:-32}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/progressive_recipe_20261020/train}"
SESSION=cvtsp_progressive_recipe
CPU_AFFINITY="${CPU_AFFINITY:-0-$(( $(nproc) - 1 ))}"
if [[ "${RUN_INNER:-0}" == 1 ]]; then
  cd "$ROOT"
  export PYTHONPATH="$ROOT" CUDA_VISIBLE_DEVICES="$GPU_ID"
  export CVTSP_VECTORIZED_LOCAL_SEARCH=1 CVTSP_TORCH_THREADS=1
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-/home/marmot/wangjianing/gurobi.lic}"
  ulimit -c 0
  mkdir -p "$OUTPUT_DIR"
  set +e
  taskset -c "$CPU_AFFINITY" "$PYTHON_BIN" -B -X faulthandler -u scripts/train_progressive_recipe.py \
    --output-dir "$OUTPUT_DIR" --workers "$WORKERS" 2>&1 | tee -a "$OUTPUT_DIR/live.log"
  code="${PIPESTATUS[0]}"
  printf '\nTRAIN_EXIT=%s time=%s\n' "$code" "$(date --iso-8601=seconds)" | tee -a "$OUTPUT_DIR/live.log"
  exec bash
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "Session exists: $SESSION" >&2; exit 1; fi
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing nonempty output: $OUTPUT_DIR" >&2; exit 1
fi
printf -v inner '%q ' env RUN_INNER=1 PYTHON_BIN="$PYTHON_BIN" OUTPUT_DIR="$OUTPUT_DIR" CPU_AFFINITY="$CPU_AFFINITY" \
  bash "$ROOT/scripts/train_progressive_recipe_tmux.sh" "$GPU_ID" "$WORKERS"
tmux new-session -d -s "$SESSION" -c "$ROOT" "$inner"
echo "Started $SESSION: GPU $GPU_ID, $WORKERS workers, CPU $CPU_AFFINITY, output $OUTPUT_DIR"
