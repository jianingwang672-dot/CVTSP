#!/usr/bin/env bash
# Launch E133 continuation branches concurrently (one tmux session).
# usage: run_recipe_continuation_tmux.sh [GPU_ID] [WORKERS_PER_BRANCH] [MODES...]
set -euo pipefail
GPU_ID="${1:-0}"; WORKERS="${2:-16}"; shift $(( $# > 2 ? 2 : $# ))
MODES=("$@"); [[ ${#MODES[@]} -gt 0 ]] || MODES=(ctrl multi_c0)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
OUT="$ROOT/outputs/recipe_diversity_20261022"
SESSION=cvtsp_recipe_diversity
CPU_AFFINITY="${CPU_AFFINITY:-0-$(( $(nproc) - 1 ))}"
if [[ "${RUN_INNER:-0}" == 1 ]]; then
  cd "$ROOT"
  export PYTHONPATH="$ROOT" CUDA_VISIBLE_DEVICES="$GPU_ID" CVTSP_VECTORIZED_LOCAL_SEARCH=1 CVTSP_TORCH_THREADS=1
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-/home/marmot/wangjianing/gurobi.lic}"
  ulimit -c 0
  for mode in "${MODES[@]}"; do
    mkdir -p "$OUT/$mode"
    ( set +e; taskset -c "$CPU_AFFINITY" "$PYTHON_BIN" -B -X faulthandler -u scripts/run_recipe_continuation.py \
        --mode "$mode" --workers "$WORKERS" >> "$OUT/$mode/live.log" 2>&1
      printf '\nTRAIN_EXIT=%s time=%s\n' "$?" "$(date --iso-8601=seconds)" >> "$OUT/$mode/live.log" ) &
  done
  wait
  echo "ALL_BRANCHES_DONE $(date --iso-8601=seconds)" >> "$OUT/branches.log"
  exec bash
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "Session exists: $SESSION" >&2; exit 1; fi
for mode in "${MODES[@]}"; do
  if [[ -d "$OUT/$mode/train" ]] && [[ -n "$(find "$OUT/$mode/train" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing nonempty output: $OUT/$mode/train" >&2; exit 1
  fi
done
printf -v inner '%q ' env RUN_INNER=1 PYTHON_BIN="$PYTHON_BIN" CPU_AFFINITY="$CPU_AFFINITY" \
  bash "$ROOT/scripts/run_recipe_continuation_tmux.sh" "$GPU_ID" "$WORKERS" "${MODES[@]}"
tmux new-session -d -s "$SESSION" -c "$ROOT" "$inner"
echo "Started $SESSION: ${MODES[*]} on GPU $GPU_ID, $WORKERS workers each, CPU $CPU_AFFINITY"
