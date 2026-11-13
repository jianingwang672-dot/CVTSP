#!/usr/bin/env bash
# Ablation B (no shadow price, real Gurobi): train to E100 with the budgeted
# exact search (anchor start, first-improvement 2-opt, <=100 solves/instance).
cd "/home/marmot/wangjianing/self-improving cvtsp"
O=outputs/ablation_20261028/exact_search/train
mkdir -p $O
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 CVTSP_VECTORIZED_LOCAL_SEARCH=1 CVTSP_TORCH_THREADS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GRB_LICENSE_FILE=/home/marmot/wangjianing/gurobi.lic
export CVTSP_EXACT_BUDGET=100
ulimit -c 0
echo "START $(date --iso-8601=seconds)" >> $O/run.log
taskset -c 0-31 .venv/bin/python -B -X faulthandler -u scripts/train_ablation.py exact_search --output-dir $O \
  2>&1 | grep --line-buffered -v "Set parameter\|WLS\|Academic" >> $O/live.log
echo "EXIT ${PIPESTATUS[0]} $(date --iso-8601=seconds)" >> $O/run.log
