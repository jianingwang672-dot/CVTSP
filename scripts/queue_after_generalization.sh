#!/usr/bin/env bash
# Runs the pure-Gurobi pilot (3 test500 instances per J, 7200 s each) once the
# generalization evaluation has released the CPUs, so neither run skews the
# other's timing.
cd "/home/marmot/wangjianing/self-improving cvtsp"
export PYTHONPATH=. GRB_LICENSE_FILE=/home/marmot/wangjianing/gurobi.lic OMP_NUM_THREADS=1
while kill -0 1050444 2>/dev/null; do sleep 60; done
echo "PILOT_START $(date --iso-8601=seconds)" >> outputs/full_gurobi_test500_20261029/pilot_7200s/queue.log
.venv/bin/python -B scripts/run_full_gurobi_test500.py --per-size 3 --time-limit 7200 --workers 15 --output-dir outputs/full_gurobi_test500_20261029/pilot_7200s 2>&1 | grep --line-buffered -v "Set parameter\|WLS\|Academic" >> outputs/full_gurobi_test500_20261029/pilot_7200s/live.log
echo "PILOT_EXIT ${PIPESTATUS[0]} $(date --iso-8601=seconds)" >> outputs/full_gurobi_test500_20261029/pilot_7200s/queue.log
