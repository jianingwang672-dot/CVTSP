"""Summarize retained E500 metrics without running models or solvers."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/train__scratch_e500_lr1e4_min1e7_w128_qdhybrid_replay30_vectorized"
OUT = ROOT / "outputs/e500_log_review_20261010"
FIELDS = (
    "epoch", "learning_rate", "target_mean_objective", "target_exact_reward_makespan",
    "anchor_exact_makespan", "expert_exact_makespan", "rl_loss", "hybrid_policy_loss",
    "exact_expert_policy_loss", "exact_expert_aux_weight", "il_loss",
    "imitation_token_accuracy", "edge_loss", "expert_better_ratio",
    "expert_improvement_percent", "exact_rl_advantage_mean", "exact_rl_target_better_ratio",
    "actual_top_k", "actual_diversity_starts", "accepted_moves", "evaluated_moves",
    "local_search_runtime_seconds", "sparse_phase_wall_runtime_seconds",
    "gurobi_calls_per_instance", "epoch_runtime_seconds", "validation_runtime_seconds",
    "reference_promoted",
)


def mean(rows, field):
    return sum(float(row[field]) for row in rows) / len(rows)


def main():
    with (RUN / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(float(row["epoch"])) for row in rows]
    if epochs != list(range(1, 501)):
        raise ValueError("Expected exactly one metric row for each epoch 1–500")
    for row in rows:
        for field in FIELDS:
            if not math.isfinite(float(row[field])):
                raise ValueError(f"Nonfinite E{row['epoch']} {field}")
    # The fixed budget comes from both retained config and epoch audit files.
    budget = json.loads((RUN / "params.json").read_text())["trainer_params"]["train_episodes"]
    if budget != 10000:
        raise ValueError("Unexpected historical training budget")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "epoch_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS + ("accepted_moves_per_start",))
        writer.writeheader()
        for row in rows:
            writer.writerow({**{key: row[key] for key in FIELDS},
                             "accepted_moves_per_start": float(row["accepted_moves"]) / (budget * float(row["actual_top_k"]))})
    phases = []
    for start, end in ((1, 20), (21, 100), (101, 200), (201, 365), (366, 500)):
        subset = rows[start - 1:end]
        starts = sum(budget * float(row["actual_top_k"]) for row in subset)
        phases.append({
            "epochs": [start, end],
            "validation_start": float(subset[0]["target_mean_objective"]),
            "validation_end": float(subset[-1]["target_mean_objective"]),
            "expert_improvement_percent": mean(subset, "expert_improvement_percent"),
            "expert_better_percent": 100 * mean(subset, "expert_better_ratio"),
            "il_accuracy_percent": 100 * mean(subset, "imitation_token_accuracy"),
            "accepted_moves_per_start": sum(float(row["accepted_moves"]) for row in subset) / starts,
            "reference_promotions": sum(int(float(row["reference_promoted"])) for row in subset),
            "train_minutes": mean(subset, "epoch_runtime_seconds") / 60,
            "validation_minutes": mean(subset, "validation_runtime_seconds") / 60,
        })
    best = min(rows, key=lambda row: float(row["target_mean_objective"]))
    key_epochs = [{key: row[key] for key in FIELDS} for row in rows if int(float(row["epoch"])) in (1, 20, 50, 100, 200, 365, 500)]
    missing = [
        "RL/IL gradient norms before and after clipping were not persisted.",
        "Swap and two_opt accepted counts and upper-bound improvements were not persisted separately.",
        "Per-start accepted-move distributions and frequency of hitting the 30-move cap were not persisted.",
        "Per-batch local-search histories exist only in memory; old checkpoints cannot reconstruct these observations.",
    ]
    summary = {
        "source": str((RUN / "metrics.csv").relative_to(ROOT)), "epochs": len(rows),
        "train_instances_per_epoch": budget,
        "best_epoch": int(float(best["epoch"])),
        "best_fixed_validation": float(best["target_mean_objective"]),
        "e100_fixed_validation": float(rows[99]["target_mean_objective"]),
        "e500_fixed_validation": float(rows[-1]["target_mean_objective"]),
        "e100_to_best_improvement_percent": 100 * (float(rows[99]["target_mean_objective"]) - float(best["target_mean_objective"])) / float(rows[99]["target_mean_objective"]),
        "phases": phases, "key_epochs": key_epochs, "missing_observations": missing,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# E500 日志整理（2026-10-10）", "",
        "来源：完整的 500 行 metrics.csv；只分析已保存数据。", "",
        f"- Epoch 100 固定验证：{summary['e100_fixed_validation']:.6f}。",
        f"- 全程最低：Epoch {summary['best_epoch']}，{summary['best_fixed_validation']:.6f}。",
        f"- Epoch 100 到全程最低只改善 {summary['e100_to_best_improvement_percent']:.3f}%；Epoch 500 为 {summary['e500_fixed_validation']:.6f}。",
        "", "## 按阶段观察", "",
        "| Epoch | 固定验证 起→止 | Expert 改善幅度均值 | Expert 改善实例占比 | IL token accuracy | 接受 move/搜索起点 |",
        "|---|---|---|---|---|---|",
    ]
    for p in phases:
        lines.append(f"| {p['epochs'][0]}–{p['epochs'][1]} | {p['validation_start']:.4f}→{p['validation_end']:.4f} | {p['expert_improvement_percent']:.3f}% | {p['expert_better_percent']:.2f}% | {p['il_accuracy_percent']:.2f}% | {p['accepted_moves_per_start']:.2f} |")
    lines += ["", "Expert 改善是相对于 Anchor 的真实 makespan 改善；move 是全部 Top-K 搜索链上的上界改进，不能等同于真实解的改进次数。", "",
              "## 已有与缺失的观察", "",
              "已有：RL/IL/Edge loss、混合与精确辅助 loss、精确 reward/优势、Target 胜率、IL 准确率、总体局部搜索计数、耗时、LR、Reference 更新。",
              "", "缺失：RL/IL 梯度范数；swap/2-opt 各自的接受次数与贡献；各搜索链的步数分布及达到 30 步上限的比例。旧日志无法补算，需后续运行时记录。", "",
              "随机训练实例的规模与数据逐 epoch 变化，因此 train makespan 不能直接当作固定验证曲线。负的 RL loss 不等于出错。", "",
              "## 外部测试（既有结果）", "",
              "| 旧版模型 | 固定验证 | Real529 | 人工5000 |", "|---|---|---|---|",
              "| Epoch 100 | 18.218866 | 333.459454 | 17.751879 |",
              "| Epoch 365 | 18.139799 | 334.177603 | 17.855958 |", "",
              "外部数值来源：outputs/eval_vectorized_epoch365_external/summary.json。该背离不单独证明过拟合。", "",
              "## 恢复与新配置", "",
              "恢复 E500：固定 Top-K=30（质量20+多样性10）、多样性前60/5%质量门槛、swap/2-opt≤30、密集POMO+精确Expert辅助RL、IL固定5遍、改进Expert权重5、replay30%/50000、adaptive near-miss 1%–5%、向量化。", "",
              "新预算：100 epoch × 20000随机新实例；固定验证2000（每种J400）；独立人工测试500（每种J100）。LR从1e-4按100 epoch余弦衰减到1e-7，warmup1 epoch。", "",
              "验证保持 E500 的单greedy候选规则；测试用完整POMO、无增强、Gurobi精算取最优，同时输出单greedy结果供核对。旧Real529与人工5000完整数据保留。", "",
              "未发现 E500 完整代码快照：算法按保留的参数、日志及现有条件分支恢复，不声称源文件逐字节回退。E200 的自适应Top-K、IL5/2/1、replay10%在新profile中关闭。", "",
              "尚未启动训练。测试500只用于报告，不参与Reference更新或调参。",
             ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("best_epoch", "best_fixed_validation", "e100_to_best_improvement_percent")}, indent=2))
    print(f"Report: {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
