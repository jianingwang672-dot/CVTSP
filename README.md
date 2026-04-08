# CVTSP

CVTSP with:
- upper level: DRL sequence policy
- lower level: Gurobi-based exact CVP solver
- dataset: 529 real instances in `instance/Data`

## Main Structure
- `CVTSP_SOCP.py`: exact lower-level CVP solver
- `verify_reference_cases.py`: reference-case verification script
- `instance/Data`: real benchmark instances
- `src/CVPSolver.py`: lower-level Gurobi CVP wrapper
- `src/TSPEnv.py`: rollout / environment logic
- `src/TSPModel.py`: DRL sequence model
- `src/TSPTrainer.py`: training loop + training entry
- `src/TSPTester.py`: evaluation / inference entry
- `src/TSProblemDef.py`: real-instance loading and split definition
- `src/TSPUtils.py`: checkpoint / logging / seed / validation / lightweight config helpers
- `src/plot_solution.py`: solution plotting
- `gurobi/POMO_Gurobi_Jianing`: preserved external reference folder

## Common Commands
Train:
```bash
python -m src.TSPTrainer --output-dir outputs/train_run
```

Evaluate test split:
```bash
python -m src.TSPTester --checkpoint-path outputs/train_run/best.pt --split test --output-dir outputs/eval_test
```

Infer one instance:
```bash
python -m src.TSPTester --checkpoint-path outputs/train_run/best.pt --instance-path instance/Data/Example_1.txt --output-dir outputs/infer_one
```
