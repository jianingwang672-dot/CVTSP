#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def _last_metrics(path: Path) -> dict[str, str]:
    metrics_path = path / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"missing metrics file: {metrics_path}")
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty metrics file: {metrics_path}")
    return rows[-1]


def _number(row: dict[str, str], key: str) -> float:
    if key not in row or row[key] == "":
        raise KeyError(f"metrics row is missing {key!r}")
    return float(row[key])


def _format(value: float) -> str:
    if not math.isfinite(value):
        return str(value)
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the paired one-epoch Or-opt/perturbation diagnostic."
    )
    parser.add_argument("branch_a", type=Path, help="Or-opt output directory")
    parser.add_argument(
        "branch_b", type=Path, help="Or-opt + perturbation output directory"
    )
    args = parser.parse_args()

    branch_a = _last_metrics(args.branch_a)
    branch_b = _last_metrics(args.branch_b)
    epoch_a = int(_number(branch_a, "epoch"))
    epoch_b = int(_number(branch_b, "epoch"))
    if epoch_a != epoch_b:
        raise ValueError(f"epoch mismatch: A={epoch_a}, B={epoch_b}")
    if _number(branch_a, "perturbation_restarts") != 0:
        raise ValueError("Branch A must have perturbation_restarts=0")
    if _number(branch_b, "perturbation_restarts") <= 0:
        raise ValueError("Branch B must enable perturbation_restarts")

    metrics = (
        ("固定验证 makespan", "target_mean_objective", "lower"),
        ("训练 Expert makespan", "expert_score", "lower"),
        ("Expert 改善比例", "expert_improvement_percent", "higher"),
        ("新 Expert 比例", "expert_changed_ratio", "higher"),
        ("扰动尝试数", "perturbation_attempts", "higher"),
        ("扰动命中数", "perturbation_improvements", "higher"),
        ("Epoch 时间（秒）", "epoch_runtime_seconds", "lower"),
    )

    print(f"Epoch {epoch_a} paired diagnostic")
    print("| 指标 | A: Or-opt | B: Or-opt+扰动 | B-A | 趋势 |")
    print("|---|---:|---:|---:|---|")
    for label, key, direction in metrics:
        value_a = _number(branch_a, key)
        value_b = _number(branch_b, key)
        delta = value_b - value_a
        print(
            f"| {label} | {_format(value_a)} | {_format(value_b)} | "
            f"{_format(delta)} | {direction} |"
        )

    attempts = _number(branch_b, "perturbation_attempts")
    hits = _number(branch_b, "perturbation_improvements")
    hit_rate = hits / max(attempts, 1.0)
    runtime_a = _number(branch_a, "epoch_runtime_seconds")
    runtime_b = _number(branch_b, "epoch_runtime_seconds")
    overhead = runtime_b / max(runtime_a, 1e-12) - 1.0
    validation_delta = (
        _number(branch_b, "target_mean_objective")
        - _number(branch_a, "target_mean_objective")
    )
    expert_delta = _number(branch_b, "expert_score") - _number(branch_a, "expert_score")

    print()
    print(f"扰动命中率: {100.0 * hit_rate:.3f}%")
    print(f"时间开销: {100.0 * overhead:+.2f}%")
    print(f"验证 makespan 差值(B-A): {validation_delta:+.6f}")
    print(f"Expert makespan 差值(B-A): {expert_delta:+.6f}")
    if hits <= 0:
        conclusion = "扰动无直接命中；不建议进入正式分支。"
    elif expert_delta >= 0.0 and validation_delta >= 0.0:
        conclusion = "扰动虽有命中，但本轮未转化为Expert或验证收益；暂不晋级。"
    elif overhead > 0.20:
        conclusion = "扰动产生收益，但时间增加超过20%；需权衡或降低扰动频率。"
    else:
        conclusion = (
            "扰动有直接命中，且Expert或验证指标改善、时间开销不超过20%；"
            "建议晋级做更长验证。"
        )
    print(f"结论: {conclusion}")
    print("说明: 单个epoch用于筛选，不足以作为最终性能结论。")


if __name__ == "__main__":
    main()
