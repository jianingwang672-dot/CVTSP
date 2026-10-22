#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${1:?Usage: bash scripts/evaluate_e500_logic_test500.sh CHECKPOINT [WORKERS]}"
WORKERS="${2:-8}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
BASELINE="${BASELINE:-$ROOT/outputs/train__cvtsp_random_n100_nearest_quarter/checkpoint-200.pt}"
OUT="${OUTPUT_DIR:-$ROOT/outputs/eval_e500logic_test500_$(basename "$CHECKPOINT" .pt)}"
cd "$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$PYTHON_BIN" -B -m scripts.prepare_e500_logic_datasets --verify-only
"$PYTHON_BIN" -B -u -m scripts.benchmark_pomo_selection_holdout \
  --baseline "$BASELINE" --baseline-label gurobi200 \
  --model "target=$CHECKPOINT" --output-dir "$OUT" \
  --fixed-validation-dir "$ROOT/validation/e500_logic_e100_20261010/test500" \
  --sizes 20 40 60 80 100 --instances-per-size 100 --seed 20262010 \
  --workers "$WORKERS" --batch-size 16 --device cpu --include-free-start
