#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
OUTPUT="$ROOT/outputs/lkh_search_pilot_20261014"
SESSION=cvtsp_lkh_search_pilot
export CUDA_VISIBLE_DEVICES=''
export CVTSP_VECTORIZED_LOCAL_SEARCH=1 CVTSP_TORCH_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-/home/marmot/wangjianing/gurobi.lic}"
cd "$ROOT"
if [[ "${PILOT_INNER:-0}" == 1 ]]; then
 ulimit -c 0
 set +e
 "$PYTHON_BIN" -B -X faulthandler -u -m scripts.benchmark_lkh_fixed_bound --per-size 20 --workers 2 --output-dir "$OUTPUT" 2>&1 | tee -a "$OUTPUT/live.log"
 status="${PIPESTATUS[0]}"
 printf '\nPILOT_EXIT=%s time=%s\n' "$status" "$(date --iso-8601=seconds)" | tee -a "$OUTPUT/live.log"
 exit "$status"
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "Pilot already exists" >&2;exit 1;fi
if [[ -d "$OUTPUT" ]] && [[ -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then echo "Refusing nonempty pilot output" >&2;exit 1;fi
mkdir -p "$OUTPUT"
cp "$ROOT/docs/lkh_search_pilot_20261014.md" "$OUTPUT/protocol.md"
printf -v command '%q ' env PILOT_INNER=1 PYTHON_BIN="$PYTHON_BIN" bash "$ROOT/scripts/run_lkh_search_pilot_tmux.sh"
tmux new-session -d -s "$SESSION" -c "$ROOT" "$command"
echo "Started $SESSION: CPU only, 2 workers, 100 fresh instances; B training stays live"
