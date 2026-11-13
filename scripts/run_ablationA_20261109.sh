#!/usr/bin/env bash
# Relaunch of 4A+5a after the queue tail exited at once (timeout rejected "24h10m").
cd "/home/marmot/wangjianing/self-improving cvtsp"
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 CVTSP_VECTORIZED_LOCAL_SEARCH=1 CVTSP_TORCH_THREADS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GRB_LICENSE_FILE=/home/marmot/wangjianing/gurobi.lic
ulimit -c 0
mv outputs/pomo_gurobi_ablationA_20261030/live.log outputs/pomo_gurobi_ablationA_20261030/failed_start_20261107.log 2>/dev/null
echo "NOTE 4A+5a relaunched (timeout interval fix) $(date --iso-8601=seconds)" >> outputs/queue_20261030/queue.log
scripts/run_queue_20261030_tail.sh
echo "QUEUE_DONE $(date --iso-8601=seconds)" >> outputs/queue_20261030/queue.log
