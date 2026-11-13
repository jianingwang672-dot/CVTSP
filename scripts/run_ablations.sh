#!/usr/bin/env bash
# Runs the four from-scratch ablations (scripts/train_ablation.py) one after
# another to E100, once the SA reruns have finished so their timing is clean.
# A variant whose train/ already holds checkpoint-100.pt is skipped.
cd "/home/marmot/wangjianing/self-improving cvtsp"
O=outputs/ablation_20261028
mkdir -p $O
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 CVTSP_VECTORIZED_LOCAL_SEARCH=1 CVTSP_TORCH_THREADS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GRB_LICENSE_FILE=/home/marmot/wangjianing/gurobi.lic
ulimit -c 0
until grep -qs "^SA_3X_EXIT=" outputs/pomo_gurobi_equal_budget_20261027/after.log; do sleep 120; done
for v in ${VARIANTS:-no_il no_exact_expert no_search no_recipe}; do
  out=$O/$v/train
  [ -s $out/checkpoint-100.pt ] && continue
  mkdir -p $out
  echo "START $v $(date --iso-8601=seconds)" >> $O/queue.log
  taskset -c 0-31 .venv/bin/python -B -X faulthandler -u scripts/train_ablation.py $v --output-dir $out \
    2>&1 | grep --line-buffered -v "Set parameter\|WLS\|Academic" >> $out/live.log
  echo "EXIT $v ${PIPESTATUS[0]} $(date --iso-8601=seconds)" >> $O/queue.log
done
echo "QUEUE_DONE $(date --iso-8601=seconds)" >> $O/queue.log
