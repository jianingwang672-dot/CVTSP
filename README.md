# CVTSP

The formal joint training entry is `train_joint_self_improvement.py`. It
combines POMO reinforcement learning, fixed-anchor primal-upper-bound rewards,
at most two exact Gurobi solves per instance, solver-free swap/asymmetric
2-opt self-improvement, in-batch expert imitation, a detached directed-cost
predictor, and a frozen reference/target promotion loop. See
`docs/joint_self_improvement_training.md`.

The default joint-training configuration has been restored to the completed
500-epoch vectorized run: 10,000 instances/epoch, adaptive 1–5% near-miss,
swap + 2-opt, and the original 1,000-instance single-route fixed validation.
See [the restored configuration](docs/e500_configuration_restored_20261019.md).
Inspect it without starting training with
`DRY_RUN=1 bash scripts/train_scratch_e500_vectorized_w128_tmux.sh`.

CVTSP with:
- upper level: DRL sequence policy
- lower level: Gurobi-based exact CVP solver
- dataset: 529 real instances in `instance/Data`

## Main Structure
- `CVTSP_SOCP.py`: exact lower-level CVP solver
- `verify_reference_cases.py`: reference-case verification script
- `instance/Data`: real benchmark instances
- `src/CVPSolver.py`: lower-level Gurobi CVP wrapper
- `src/CVXPYLayerSolver.py`: preserved legacy cvxpylayer experiment; inactive in the current workflow
- `src/TSPEnv.py`: rollout / environment logic
- `src/TSPModel.py`: DRL sequence model
- `src/TSPTrainer.py`: training loop + training entry
- `src/TSPTester.py`: evaluation / inference entry
- `src/TSProblemDef.py`: real-instance loading and split definition
- `src/TSPUtils.py`: checkpoint / logging / seed / validation / lightweight config helpers
- `src/plot_solution.py`: solution plotting
- `src/CertifiedUpperBound.py`: fixed-Gurobi-solution directed upper bound and tour moves
- `src/CertifiedLocalSearch.py`: formal fixed-anchor solver-free search plus preserved offline refresh search
- `verify_certified_upper_bound.py`: randomized upper-bound correctness gate
- `run_certified_local_search.py`: small-scale certified local-search entry
- `src/PseudoExpert.py`: certified pseudo-expert generation and exact policy gate
- `src/ImitationLearning.py`: teacher-forced cross-entropy imitation training
- `run_self_improvement.py`: complete multi-round teacher/student + edge-predictor loop
- `compare_policy_checkpoints.py`: exact independent teacher/student comparison
- `src/EdgeCostPredictor.py`: directed edge-cost labels, predictor, ranking metrics, and O(1) move scores
- `src/JointSelfImprovement.py`: sparse one-anchor/optional-final exact data flow
- `src/JointSelfImprovementTrainer.py`: formal upper/lower joint training loop
- `src/ReferencePolicy.py`: fixed exact validation and reference promotion
- `run_edge_predictor_phase.py`: safe predictor-guided Top-K certified-search experiment
- `gurobi/POMO_Gurobi_Jianing`: preserved external reference folder

## Common Commands
Train:
```bash
python -m src.TSPTrainer --output-dir outputs/train_run
```

Formal sparse-Gurobi joint training:

```bash
python3 train_joint_self_improvement.py --smoke-test \
  --output-dir outputs/joint_si_sparse_smoke
```

`train_n100.py` and `test_n100.py` use `solver_backend="gurobi"`. Training reward
is the negative Gurobi fixed-tour objective. The active loss is standard POMO
policy gradient; REBAR and cvxpylayer auxiliary losses are disabled.

Evaluate test split:
```bash
python -m src.TSPTester --checkpoint-path outputs/train_run/best.pt --split test --output-dir outputs/eval_test
```

Infer one instance:
```bash
python -m src.TSPTester --checkpoint-path outputs/train_run/best.pt --instance-path instance/Data/Example_1.txt --output-dir outputs/infer_one
```

Certified local-search stage and verification commands are documented in
[`docs/certified_local_search.md`](docs/certified_local_search.md).

Pseudo-expert generation, cross-entropy imitation, and guarded teacher/student
self-improvement are documented in
[`docs/self_improvement.md`](docs/self_improvement.md).

Directed edge-cost prediction and safe Top-K candidate pruning are documented
in [`docs/edge_cost_predictor.md`](docs/edge_cost_predictor.md).
