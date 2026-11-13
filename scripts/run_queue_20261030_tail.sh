#!/usr/bin/env bash
# Tail of scripts/run_queue_20261030.sh (environment inherited from it):
# 4A + 5a (POMO side): POMO + exact Gurobi trained from scratch with the G200
# recipe for 24 h of wall time on 32 workers. Epoch checkpoints and budget
# checkpoints (E133's solves / instances) serve ablation A; hourly time
# checkpoints serve the 5a curve. The timeout stops it just after time-024.00h.
cd "/home/marmot/wangjianing/self-improving cvtsp"
Q=outputs/queue_20261030
A=outputs/pomo_gurobi_ablationA_20261030; mkdir -p $A
echo "START 4A+5a-pomo $(date --iso-8601=seconds)" >> $Q/queue.log
timeout --signal=INT --kill-after=10m 87000s taskset -c 0-31 .venv/bin/python -B -X faulthandler -u \
  scripts/train_pomo_gurobi_budget.py --output-dir $A --epochs 100 --workers 32 --checkpoint-hours 1 \
  2>&1 | grep --line-buffered -v "Set parameter\|WLS\|Academic" >> $A/live.log
echo "EXIT 4A+5a-pomo ${PIPESTATUS[0]} $(date --iso-8601=seconds)" >> $Q/queue.log
