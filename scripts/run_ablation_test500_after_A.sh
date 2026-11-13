#!/usr/bin/env bash
# After 4A+5a finishes: test500 eight-view evaluation of ablations B and D and
# the main run at checkpoint-100 (ablation table at epoch 100). Waits for A so
# the evaluation does not share cores with it and distort the 5a wall-time curve.
cd "/home/marmot/wangjianing/self-improving cvtsp"
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export GRB_LICENSE_FILE=/home/marmot/wangjianing/gurobi.lic
ulimit -c 0
Q=outputs/queue_20261030/queue.log
while pgrep -f train_pomo_gurobi_budget.py > /dev/null; do sleep 300; done
O=outputs/ablation_20261028/test500_aug8_e100; mkdir -p $O
echo "START ablation-test500 $(date --iso-8601=seconds)" >> $Q
.venv/bin/python -B -u scripts/benchmark_test500_aug8.py --workers 32 --output-dir $O \
  --model main_e100=outputs/progressive_recipe_20261020/train/checkpoint-100.pt \
  --model exact_search_e100=outputs/ablation_20261028/exact_search/train/checkpoint-100.pt \
  --model no_diversity_e100=outputs/ablation_20261028/no_diversity/train/checkpoint-100.pt \
  2>&1 | grep --line-buffered -v "Set parameter\|WLS\|Academic" >> $O/live.log
echo "EXIT ablation-test500 ${PIPESTATUS[0]} $(date --iso-8601=seconds)" >> $Q
