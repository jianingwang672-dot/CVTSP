#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="$ROOT/outputs/epoch100_controlled_branches"
SESSION="cvtsp_epoch100_branches_report"

if [[ "${RUN_INNER:-0}" == "1" ]]; then
  cd "$ROOT"
  while true; do
    if grep -q '^EXPERIMENTS_COMPLETE ' "$OUTPUT/live.log"; then
      printf 'Already summarized.\n'
      break
    fi
    if grep -q '^EVALUATION_EXIT=0$' "$OUTPUT/live.log"; then
      "$ROOT/.venv/bin/python" -m scripts.summarize_epoch100_branches
      printf 'COMPLETE report=%s time=%s\n' "$OUTPUT/report.md" "$(date --iso-8601=seconds)" \
        | tee -a "$OUTPUT/report_watch.log"
      break
    fi
    if grep -Eq '^EVALUATION_EXIT=[1-9]|^BRANCH_EXIT mode=[^ ]+ code=[1-9]' "$OUTPUT/live.log"; then
      printf 'FAILED time=%s\n' "$(date --iso-8601=seconds)" \
        | tee -a "$OUTPUT/report_watch.log"
      break
    fi
    sleep 60
  done
  exec bash
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Existing session: $SESSION" >&2
  exit 1
fi
printf -v command '%q ' env RUN_INNER=1 bash "$ROOT/scripts/$(basename "$0")"
tmux new-session -d -s "$SESSION" -c "$ROOT" "$command"
echo "Started $SESSION"
