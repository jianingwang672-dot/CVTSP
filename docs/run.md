# Run Guide

## Dependencies
```bash
python -m pip install -r requirements.txt
```

需要保证以下依赖可用：
- `torch`
- `gurobipy`
- `numpy`
- `pandas`
- `PyYAML`
- `loguru`

## Train
```bash
python -m src.TSPTrainer \
  --output-dir outputs/train_run
```

训练输出：
- `run.log`
- `resolved_config.yaml`
- `metrics.csv`
- `best.pt`
- `last.pt`
- `train_summary.json`

## Evaluate One Real Instance
```bash
python -m src.TSPTester \
  --checkpoint-path outputs/train_run/best.pt \
  --instance-path instance/Data/Example_1.txt \
  --output-dir outputs/eval_one
```

## Evaluate Real Test Split
```bash
python -m src.TSPTester \
  --checkpoint-path outputs/train_run/best.pt \
  --split test \
  --output-dir outputs/eval_test
```

## Infer One Real Instance
```bash
python -m src.TSPTester \
  --checkpoint-path outputs/train_run/best.pt \
  --instance-path instance/Data/Example_1.txt \
  --output-dir outputs/infer_one
```

## Notes
- 当前仓库只保留真实 `529` 个实例。
- 上层为 DRL sequence policy，下层为 exact Gurobi solver。
- `sample` 模式会生成多个 candidate，再由 exact solver 选最优。
