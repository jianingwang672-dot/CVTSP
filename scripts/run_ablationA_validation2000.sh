#!/usr/bin/env bash
# Full-POMO validation2000 evaluation of ablation A (POMO + exact Gurobi from
# scratch, 24 h): hourly time checkpoints (5a curve), the two budget
# checkpoints (ablation A row) and every fourth epoch. Same protocol and
# instances as outputs/pomo_gurobi_equal_budget_20261027/curve_validation2000,
# whose G200 scores are reused.
cd "/home/marmot/wangjianing/self-improving cvtsp"
export GRB_LICENSE_FILE=/home/marmot/wangjianing/gurobi.lic OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
A=outputs/pomo_gurobi_ablationA_20261030/train
out=outputs/pomo_gurobi_ablationA_20261030/validation2000; mkdir -p $out; args=()
for h in $(seq -f "%03g" 1 24); do args+=(--model "pomoA_time${h}h=$A/time-${h}.00h.pt"); done
for b in equal_solves equal_instances; do args+=(--model "pomoA_$b=$A/budget-$b.pt"); done
for e in $(seq 4 4 36); do args+=(--model "pomoA_e$e=$A/checkpoint-$e.pt"); done
echo "START $(date --iso-8601=seconds)" >> $out/run.log
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -B -u scripts/benchmark_pomo_selection_holdout.py \
  --baseline outputs/train__cvtsp_random_n100_nearest_quarter/checkpoint-200.pt --baseline-label g200_e200 "${args[@]}" \
  --cached-baseline-results outputs/pomo_gurobi_equal_budget_20261027/curve_validation2000/results.json \
  --output-dir $out --fixed-validation-dir validation/e500_logic_e100_20261010/validation2000 \
  --instances-per-size 400 --seed 20261010 --workers 32 --device cuda \
  2>&1 | grep --line-buffered -v "Set parameter\|WLS\|Academic\|^$" > $out/live.log
echo "EXIT ${PIPESTATUS[0]} $(date --iso-8601=seconds)" >> $out/run.log
