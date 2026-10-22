#!/usr/bin/env bash
# Launch the S500 E100->E120 plateau branches in one tmux session, PER_WAVE at a time.
# usage: [PER_WAVE=2] run_e100_plateau_branches_tmux.sh [GPU_ID] [WORKERS_PER_BRANCH] [MODES...]
set -euo pipefail
GPU_ID="${1:-0}"
WORKERS="${2:-8}"
shift $(( $# > 2 ? 2 : $# ))
MODES=("$@")
[[ ${#MODES[@]} -gt 0 ]] || MODES=(ctrl il_changed strong_expert both)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
OUTPUT="$ROOT/outputs/e100_plateau_branches_20261019"
CPU_AFFINITY="${CPU_AFFINITY:-0-$(( $(nproc) - 1 ))}"

if [[ "${RUN_INNER:-0}" == 1 ]]; then
  MODE="$MODES"
  cd "$ROOT"
  export PYTHONPATH="$ROOT" CUDA_VISIBLE_DEVICES="$GPU_ID"
  export CVTSP_VECTORIZED_LOCAL_SEARCH=1 CVTSP_TORCH_THREADS=1
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-/home/marmot/wangjianing/gurobi.lic}"
  ulimit -c 0
  set +e
  taskset -c "$CPU_AFFINITY" "$PYTHON_BIN" -B -X faulthandler -u \
    scripts/run_e100_plateau_branch.py --mode "$MODE" --workers "$WORKERS" \
    2>&1 | tee -a "$OUTPUT/$MODE/live.log"
  code="${PIPESTATUS[0]}"
  printf '\nTRAIN_EXIT=%s time=%s\n' "$code" "$(date --iso-8601=seconds)" | tee -a "$OUTPUT/$MODE/live.log"
  exit "$code"
fi

PER_WAVE="${PER_WAVE:-2}"
if [[ "${WAVES_INNER:-0}" == 1 ]]; then
  # Run the modes in waves of PER_WAVE concurrent branches: four S500
  # processes need ~5-7 GB GPU each and do not fit on one 24 GB card together.
  cd "$ROOT"
  for (( i = 0; i < ${#MODES[@]}; i += PER_WAVE )); do
    wave=("${MODES[@]:i:PER_WAVE}")
    printf 'START_WAVE %s time=%s\n' "${wave[*]}" "$(date --iso-8601=seconds)" | tee -a "$OUTPUT/waves.log"
    pids=()
    for mode in "${wave[@]}"; do
      mkdir -p "$OUTPUT/$mode"
      env RUN_INNER=1 PYTHON_BIN="$PYTHON_BIN" CPU_AFFINITY="$CPU_AFFINITY" \
        bash "$ROOT/scripts/run_e100_plateau_branches_tmux.sh" "$GPU_ID" "$WORKERS" "$mode" > /dev/null 2>&1 &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" || true; done
    printf 'END_WAVE %s time=%s\n' "${wave[*]}" "$(date --iso-8601=seconds)" | tee -a "$OUTPUT/waves.log"
  done
  exec bash
fi

for mode in "${MODES[@]}"; do
  if tmux has-session -t cvtsp_e100_plateau 2>/dev/null; then echo "Session exists: cvtsp_e100_plateau" >&2; exit 1; fi
  if [[ -d "$OUTPUT/$mode/train" ]] && [[ -n "$(find "$OUTPUT/$mode/train" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing nonempty output: $OUTPUT/$mode/train" >&2; exit 1
  fi
done
printf -v inner '%q ' env WAVES_INNER=1 PER_WAVE="$PER_WAVE" PYTHON_BIN="$PYTHON_BIN" \
  CPU_AFFINITY="$CPU_AFFINITY" bash "$ROOT/scripts/run_e100_plateau_branches_tmux.sh" \
  "$GPU_ID" "$WORKERS" "${MODES[@]}"
tmux new-session -d -s cvtsp_e100_plateau -c "$ROOT" "$inner"
echo "Started cvtsp_e100_plateau: waves of $PER_WAVE over ${MODES[*]}; GPU $GPU_ID, $WORKERS workers each, CPU $CPU_AFFINITY"
