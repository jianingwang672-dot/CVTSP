#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
SESSION=cvtsp_nearmiss_observer_e20
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "Observer already live: $SESSION"; exit 0; fi
printf -v command '%q ' env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PYTHON_BIN" -B -u -m scripts.watch_near_miss_quota
tmux new-session -d -s "$SESSION" -c "$ROOT" \
  "$command >> outputs/near_miss_quota_20261013/observer.log 2>&1"
echo "Started $SESSION"
