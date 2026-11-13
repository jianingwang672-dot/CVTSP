#!/usr/bin/env bash
# Runs once the equal-budget POMO+Gurobi training has finished (CPU free):
#  1. SA reruns with their own 1x and 3x budgets (cooling schedule tied to the
#     actual budget), pinned to the eight P-cores like the original run.
#  2. Full-POMO validation2000 evaluation of the POMO+Gurobi epoch / budget
#     checkpoints and the recipe run's early epochs (E1-E5) for the
#     data-efficiency curve, pinned to the E-cores so SA timing is unaffected.
cd "/home/marmot/wangjianing/self-improving cvtsp"
P=outputs/pomo_gurobi_equal_budget_20261027
R=outputs/progressive_recipe_20261020/train
G200=outputs/train__cvtsp_random_n100_nearest_quarter/checkpoint-200.pt
export GRB_LICENSE_FILE=/home/marmot/wangjianing/gurobi.lic OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
until grep -qs "^EXIT=" $P/live.log; do sleep 120; done
echo "AFTER_START $(date --iso-8601=seconds)" >> $P/after.log

( for f in 1 3; do
    .venv/bin/python -B -u scripts/run_metaheuristics_real529.py --methods sa --factor $f \
      --output-dir outputs/metaheuristics_real529_sa_${f}x_20261027 2>&1 \
      | grep --line-buffered -v "Set parameter\|WLS\|Academic" > /dev/null
    echo "SA_${f}X_EXIT=${PIPESTATUS[0]} $(date --iso-8601=seconds)" >> $P/after.log
  done ) &

( out=$P/curve_validation2000; mkdir -p $out; args=()
  for e in $(seq 1 14); do [ -s $P/train/checkpoint-$e.pt ] && args+=(--model "pomo_e$e=$P/train/checkpoint-$e.pt"); done
  for b in equal_solves equal_instances; do [ -s $P/train/budget-$b.pt ] && args+=(--model "pomo_$b=$P/train/budget-$b.pt"); done
  for e in 1 2 3 5; do args+=(--model "recipe_e$e=$R/checkpoint-$e.pt"); done
  CUDA_VISIBLE_DEVICES=0 taskset -c 16-31 .venv/bin/python -B -u scripts/benchmark_pomo_selection_holdout.py \
    --baseline $G200 --baseline-label g200_e200 "${args[@]}" \
    --output-dir $out --fixed-validation-dir validation/e500_logic_e100_20261010/validation2000 \
    --instances-per-size 400 --seed 20261010 --workers 16 --device cuda \
    2>&1 | grep --line-buffered -v "Set parameter\|WLS\|Academic\|^$" > $out/live.log
  echo "CURVE_EXIT=${PIPESTATUS[0]} $(date --iso-8601=seconds)" >> $P/after.log ) &
wait
echo "AFTER_DONE $(date --iso-8601=seconds)" >> $P/after.log
