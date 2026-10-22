"""Export the meeting's requested observations from existing experiment records."""
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from scripts.summarize_search_ablation import search_by_size


ROOT = Path("outputs/search_ablation_20261012")
GROUPS = ("swap", "two_opt", "both")


def csv_write(name, rows):
    with (ROOT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    comparison = json.loads((ROOT / "comparison.json").read_text())
    assert comparison["complete"]
    network, searches, operators, winners, rewards = [], [], [], [], []
    fields = ("learning_rate", "rl_loss", "hybrid_policy_loss", "exact_expert_policy_loss",
              "exact_expert_aux_weight", "imitation_loss", "edge_loss", "imitation_token_accuracy",
              "exact_rl_advantage_mean", "exact_rl_target_better_ratio", "anchor_exact_makespan",
              "expert_exact_makespan", "target_exact_reward_makespan", "expert_improvement_percent",
              "expert_better_ratio", "expert_replay_effective_fraction", "expert_replay_pool_size")
    for group in GROUPS:
        directory = ROOT / group
        config = json.loads((directory / "resolved_config.json").read_text())
        clip = config["trainer_params"]["grad_clip"]
        with (directory / "metrics.csv").open() as handle:
            metrics = list(csv.DictReader(handle))
        for row in metrics:
            epoch = int(float(row["epoch"]))
            raw = json.loads((directory / "diagnostics" / f"epoch-{epoch:03d}_summary.json").read_text())
            record = {"group": group, "epoch": epoch, **{key: float(row[key]) for key in fields}}
            for phase in ("rl", "il"):
                gradients = [x for x in raw["gradients"] if x["phase"] == phase]
                record.update({f"{phase}_updates": len(gradients),
                               f"{phase}_gradient_before_mean": float(np.mean([x["before"] for x in gradients])),
                               f"{phase}_gradient_after_mean": float(np.mean([x["after"] for x in gradients])),
                               f"{phase}_clip_rate": float(np.mean([x["before"] > clip for x in gradients])),
                               f"{phase}_parameter_delta_l2": raw[f"{phase}_parameter_delta_l2"]})
            components = [x for x in raw["rl_components"] if "cosine" in x]
            assert len(components) <= 1
            for key in ("cosine", "hybrid_gradient_norm", "weighted_exact_gradient_norm"):
                record[f"first_diagnostic_batch_{key}"] = components[0][key] if components else None
            network.append(record)
            for size, value in search_by_size(raw["search"]).items():
                searches.append({"group": group, "epoch": epoch, "J": size, "move_cap": raw["move_cap"],
                                 **{key: value[key] for key in ("chains", "mean_steps", "p90_steps", "cap_hit_rate",
                                                              "above10_rate", "gain_after10_fraction")}})
            accepted, evaluated, gains = Counter(), Counter(), Counter()
            for value in raw["search"].values():
                accepted.update(value["accepted"])
                evaluated.update(value["evaluated"])
                gains.update(value["upper_bound_gain"])
            enabled = config["trainer_params"]["self_improvement_move_types"]
            for operator in enabled:
                operators.append({"group": group, "epoch": epoch, "operator": operator,
                                  "accepted": accepted[operator], "evaluated": evaluated[operator],
                                  "upper_bound_gain": gains[operator],
                                  "accepted_share": accepted[operator] / sum(accepted.values()),
                                  "upper_bound_gain_share": gains[operator] / sum(gains.values()),
                                  "upper_bound_gain_per_accepted_move": gains[operator] / accepted[operator]
                                  if accepted[operator] else None})
            rows = [x for batch in raw["winner_chains"].values() for x in batch]
            changed = [x for x in rows if x["source"] != "anchor"]
            known = [x for x in changed if x["steps"] is not None]
            counts = Counter(x["source"] for x in rows)
            winners.append({"group": group, "epoch": epoch, "instances": len(rows),
                            "anchor_count": counts["anchor"], "changed_count": len(changed),
                            "certified_count": counts["certified_upper_bound"],
                            "near_miss_count": counts["near_miss_exact"],
                            "changed_winners_with_known_chain": len(known),
                            "changed_winner_mean_moves": float(np.mean([x["steps"] for x in known])) if known else None,
                            "changed_winner_above10_rate": float(np.mean([x["steps"] > 10 for x in known])) if known else None,
                            "changed_winner_above20_rate": float(np.mean([x["steps"] > 20 for x in known])) if known else None})
        for epoch, by_size in comparison["groups"][group]["reward_ranking"].items():
            for size, value in by_size.items():
                rewards.append({"group": group, "epoch": int(epoch), "J": int(size), "instances": value["regret"]["count"],
                                **{key: value[key]["mean"] for key in ("regret", "regret_percent", "selected_rank",
                                                                       "mean_bound_gap", "advantage_sign_agreement")}})
    assert len(network) == len(winners) == 60
    assert all(x["instances"] == 20000 for x in winners)
    csv_write("meeting_network.csv", network)
    csv_write("meeting_local_search.csv", searches)
    csv_write("meeting_operators.csv", operators)
    csv_write("meeting_expert_winners.csv", winners)
    csv_write("meeting_reward_ranking.csv", rewards)

    lines = ["# 会议要求的数据与本次实验的对应记录", "",
             "来源：用户提供的元宝会议纪要，以及本次三组各20轮的metrics.csv和diagnostics。纪要中的自动总结、术语转写和过拟合判断不当作已验证事实。", "",
             "## 会议检查清单", "",
             "| 会议时点 | 要求观察/实验 | 本次状态 | 数据入口 |",
             "|---|---|---|---|",
             "| 16:49、29:04 | 训练曲线、验证指标、先查旧日志 | 已完成 | fixed_validation_curve.png；旧outputs/e500_log_review_20261010/report.md |",
             "| 18:35、30:56 | 实际local search步数与预算 | 已完成前20轮诊断；固定30与平滑预算未对照 | meeting_local_search.csv、meeting_expert_winners.csv |",
             "| 39:19、40:59 | swap/2-opt正向改进与各自贡献 | 已记录接受数、上界下降、三组端到端结果；逐算子真实makespan边际贡献未测 | meeting_operators.csv；comparison.json |",
             "| 34:29 | loss、梯度、reward | 已完成前20轮记录 | meeting_network.csv、meeting_reward_ranking.csv |",
             "| 27:49、40:59 | 短步数LKH对照 | 未测；后续方案把本轮限定为三个现有算子组 | 无 |",
             "| 34:29、39:19 | 100轮、新实例数增至1万/2万，观察学习率影响 | 配置为2万/轮、100轮调度，按后续约定只执行前20；没有1万vs2万或LR单因素对照 | resolved_config.json；meeting_network.csv |",
             "| 36:10、37:43 | 数据/标签多样性、增强、新训练对照 | 已从头训练且每轮生成新实例；增强和replay消融未测 | initialization.json、batch_data.jsonl；无增强消融结果 |",
             "| 20:20、21:59、37:43 | 基线预算对齐与重跑 | 未做；依用户后续决定暂不重训Gurobi基线 | 旧基线不能当作本轮单因素对照 |",
             "| 13:20、32:27 | validation/test背离 | 旧背离已整理；本轮有单greedy/完整POMO独立诊断，正式测试500和Real529未测 | comparison.json、diagnostic_stages.csv |", "",
             "## 合用组：loss、梯度与reward", "",
             "梯度为每次更新的裁剪前L2范数均值；clip上限1。loss符号与量纲不用于直接比较RL和IL谁更强。原始精确优势=Expert makespan−Target所选路线makespan，正值表示Target更好。", "",
             "| Epoch | 使用LR | RL loss | IL loss | RL梯度 | IL梯度 | Target胜Expert率 | IL准确率 | replay实际比例 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for e in (1, 5, 10, 20):
        x = next(x for x in network if x["group"] == "both" and x["epoch"] == e)
        lines.append(f"| {e} | {x['learning_rate']:.8g} | {x['rl_loss']:.6f} | {x['imitation_loss']:.6f} | "
                     f"{x['rl_gradient_before_mean']:.4f} | {x['il_gradient_before_mean']:.4f} | "
                     f"{x['exact_rl_target_better_ratio']:.1%} | {x['imitation_token_accuracy']:.1%} | {x['expert_replay_effective_fraction']:.1%} |")
    lines += ["", "RL所记录更新均被裁剪到约1；这表明需观察裁剪与分项方向，并不自动证明学习率或clip有错。", "",
              "| Epoch | RL阶段参数净变化L2 | IL阶段参数净变化L2 | 精确辅助权重 | 诊断首batch两RL分项梯度cosine |",
              "|---|---:|---:|---:|---:|"]
    for e in (1, 5, 10, 20):
        x = next(x for x in network if x["group"] == "both" and x["epoch"] == e)
        lines.append(f"| {e} | {x['rl_parameter_delta_l2']:.4f} | {x['il_parameter_delta_l2']:.4f} | "
                     f"{x['exact_expert_aux_weight']:.4f} | {x['first_diagnostic_batch_cosine']:.4f} |")
    lines += ["", "cosine只来自各诊断epoch的首batch，比较密集POMO与精确Expert辅助两个RL分项，不是RL与IL的梯度比较；Epoch1辅助权重0。不能外推到全部batch。", "",
              "## 合用组：算子贡献（Epoch20）", "",
              "| 算子 | 接受move数 | 接受数占比 | 累计上界下降 | 上界下降占比 | 每次接受平均上界下降 |",
              "|---|---:|---:|---:|---:|---:|"]
    for x in operators:
        if x["group"] == "both" and x["epoch"] == 20:
            lines.append(f"| {x['operator']} | {x['accepted']} | {x['accepted_share']:.1%} | {x['upper_bound_gain']:.3f} | "
                         f"{x['upper_bound_gain_share']:.1%} | {x['upper_bound_gain_per_accepted_move']:.4f} |")
    lines += ["", "贡献累计覆盖本轮全部候选搜索链，属于上界下降；中间move未调用Gurobi，所以不能把这些数当作逐算子的真实makespan贡献。", "",
              "## 候选搜索与最终Expert应分开看（Epoch20）", "",
              "| 组别 | 保留Anchor比例 | 改进Expert数/20000 | 改进Expert关联链平均move | 关联链超过10步 | Expert相对Anchor平均真实改善 |",
              "|---|---:|---:|---:|---:|---:|"]
    for g in GROUPS:
        w = next(x for x in winners if x["group"] == g and x["epoch"] == 20)
        n = next(x for x in network if x["group"] == g and x["epoch"] == 20)
        lines.append(f"| {g} | {w['anchor_count']/w['instances']:.1%} | {w['changed_count']} | {w['changed_winner_mean_moves']:.2f} | "
                     f"{w['changed_winner_above10_rate']:.1%} | {n['expert_improvement_percent']:.4f}% |")
    lines += ["", "这里的关联链是已记录结果中最终路线匹配的、最终上界最低的一条链。平均move与超过10步的比例只统计非Anchor且能找到关联链的Expert，不等同于缩短预算后的反事实结果。", "",
              "## 合用组：按J的候选链预算（Epoch20，上限26）", "",
              "| J | 平均move | P90 | 上限命中 | 第10步后上界下降占比 |",
              "|---|---:|---:|---:|---:|"]
    for x in sorted(searches, key=lambda x: x["J"]):
        if x["group"] == "both" and x["epoch"] == 20 and x["J"] in (20, 40, 60, 80, 100):
            lines.append(f"| {x['J']} | {x['mean_steps']:.2f} | {x['p90_steps']} | {x['cap_hit_rate']:.1%} | {x['gain_after10_fraction']:.1%} |")
    lines += ["", "## 评价与奖励诊断", "",
              "RL前后/IL前后的逐J质量和95%区间见report.md和diagnostic_stages.csv。上界选路regret与优势方向一致率见meeting_reward_ranking.csv；参照最优是当前采样候选中的精算最优，不是全局最优。", "",
              "## 尚不能下的结论", "",
              "- 本轮没有LKH、固定30步、1万/2万实例、不同LR、数据增强或replay比例的控制组，不能评价这些改动是否有效。",
              "- 三组都采用平滑move预算，前20轮只下降到26；不能证明后期统一10步足够。",
              "- min-LR已按后续要求改为1e-5，当前只观察前20轮，不能据此排除旧500轮后期LR过小的问题。",
              "- 只训练20轮，不能判定旧Epoch100之后背离的原因，也不能证明发生了过拟合。",
              "- 初期辅助权重0时，未记录逐batch归一化精确优势分布；原始精确优势均值/Target胜率和独立50例全精算诊断可查。", "",
              "## 数据文件", "",
              "- meeting_network.csv：三组全部20轮loss、reward/优势、梯度、更新量、LR和replay。",
              "- meeting_local_search.csv：全部已采样J的候选链步数和预算命中。",
              "- meeting_operators.csv：每轮每算子的接受/评估次数和上界贡献。",
              "- meeting_expert_winners.csv：每轮最终Expert来源及关联链步数。",
              "- meeting_reward_ranking.csv：Epoch1/5/10/20独立诊断的排序误差。",
              "- report.md：三组评价结果、RL/IL阶段质量、耗时及曲线。"]
    (ROOT / "meeting_diagnostics.md").write_text("\n".join(lines) + "\n")
    print("meeting diagnostics exported", {"network_rows": len(network), "search_rows": len(searches),
                                           "operator_rows": len(operators), "winner_rows": len(winners),
                                           "reward_rows": len(rewards)})


if __name__ == "__main__":
    main()
