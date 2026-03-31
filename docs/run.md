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

## Train DRL Master
```bash
python -m src.cli.train --config src/configs/default.yaml --output-dir outputs/train_run
```

可选 smoke test：
```bash
python -m src.cli.train --config src/configs/default.yaml --epochs 1 --batch-size 2 --max-batches-per-epoch 6 --output-dir outputs/smoke_train
```

训练输出：
- `outputs/.../run.log`
- `outputs/.../resolved_config.yaml`
- `outputs/.../metrics.csv`
- `outputs/.../best.pt`
- `outputs/.../last.pt`
- `outputs/.../train_summary.json`

## DRL + SP Inference On One Instance
```bash
python -m src.cli.infer --config src/configs/default.yaml --checkpoint-path outputs/train_run/best.pt --instance-path instance/Gam72/Example_1.txt --output-dir outputs/infer_run
```

可选采样候选：
```bash
python -m src.cli.infer --config src/configs/default.yaml --checkpoint-path outputs/train_run/best.pt --instance-path instance/Gam72/Example_1.txt --decode-type sample --num-candidates 16 --output-dir outputs/infer_sample
```

## Evaluate A Single Instance
DRL:
```bash
python -m src.cli.evaluate --config src/configs/default.yaml --checkpoint-path outputs/train_run/best.pt --instance-path instance/Gam72/Example_1.txt --output-dir outputs/eval_one
```

Baseline:
```bash
python -m src.cli.evaluate --config src/configs/default.yaml --instance-path instance/Gam72/Example_1.txt --baseline nearest_neighbor --output-dir outputs/eval_baseline_one
```

## Evaluate A Split
DRL:
```bash
python -m src.cli.evaluate --config src/configs/default.yaml --checkpoint-path outputs/train_run/best.pt --split test --output-dir outputs/eval_test
```

Baseline:
```bash
python -m src.cli.evaluate --config src/configs/default.yaml --split test --baseline input_order --output-dir outputs/eval_input_order
python -m src.cli.evaluate --config src/configs/default.yaml --split test --baseline nearest_neighbor --output-dir outputs/eval_nn
python -m src.cli.evaluate --config src/configs/default.yaml --split test --baseline random_k --num-candidates 16 --output-dir outputs/eval_random
```

## Basic Comparison Workflow
```bash
python -m src.cli.train --config src/configs/default.yaml --output-dir outputs/train_run
python -m src.cli.evaluate --config src/configs/default.yaml --checkpoint-path outputs/train_run/best.pt --split test --output-dir outputs/eval_drl
python -m src.cli.evaluate --config src/configs/default.yaml --split test --baseline nearest_neighbor --output-dir outputs/eval_nn
python -m src.cli.evaluate --config src/configs/default.yaml --split test --baseline input_order --output-dir outputs/eval_input
```

用 `evaluate_*.json` 里的 `summary.avg_objective` 做最基础比较。

## Notes
- 首版 exact SP 只在完整 sequence 生成后调用一次。
- `sample` 模式会生成多候选 sequence，并用 exact SP 逐个重排评估。
- `greedy` 模式默认仍会利用 POMO 风格多起点内部候选，但最终只输出 exact SP 评估后的最佳 candidate。
