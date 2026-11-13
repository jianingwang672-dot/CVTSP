#!/usr/bin/env bash
# Experiment queue in the user's order (2 done): 1 full Gurobi on Real529 and
# test500 (Model+, 3600 s single-threaded, 32 in parallel), then ablations
# 4B (exact_search, budgeted search, to E100) and 4D (no_diversity to E100). 4A and 5a are
# appended once their code is ready. Each stage logs START/EXIT to queue.log.
cd "/home/marmot/wangjianing/self-improving cvtsp"
Q=outputs/queue_20261030; mkdir -p $Q
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 CVTSP_VECTORIZED_LOCAL_SEARCH=1 CVTSP_TORCH_THREADS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GRB_LICENSE_FILE=/home/marmot/wangjianing/gurobi.lic
ulimit -c 0
FILTER="Set parameter\|WLS\|Academic"
G=outputs/full_gurobi_20261030
for set in real529 test500; do
  flag=""; [ $set = real529 ] && flag="--real529"
  mkdir -p $G/${set}_3600s
  echo "START 1-$set $(date --iso-8601=seconds)" >> $Q/queue.log
  .venv/bin/python -B -u scripts/run_full_gurobi_test500.py $flag --time-limit 3600 --workers 32 \
    --output-dir $G/${set}_3600s 2>&1 | grep --line-buffered -v "$FILTER" >> $G/${set}_3600s/live.log
  echo "EXIT 1-$set ${PIPESTATUS[0]} $(date --iso-8601=seconds)" >> $Q/queue.log
done
echo "START 4B $(date --iso-8601=seconds)" >> $Q/queue.log
scripts/run_exact_search_24h.sh
echo "EXIT 4B $(date --iso-8601=seconds)" >> $Q/queue.log
out=outputs/ablation_20261028/no_diversity/train; mkdir -p $out
echo "START 4D $(date --iso-8601=seconds)" >> $Q/queue.log
taskset -c 0-31 .venv/bin/python -B -X faulthandler -u scripts/train_ablation.py no_diversity --output-dir $out \
  2>&1 | grep --line-buffered -v "$FILTER" >> $out/live.log
echo "EXIT 4D ${PIPESTATUS[0]} $(date --iso-8601=seconds)" >> $Q/queue.log
[ -x scripts/run_queue_20261030_tail.sh ] && scripts/run_queue_20261030_tail.sh
echo "QUEUE_DONE $(date --iso-8601=seconds)" >> $Q/queue.log
