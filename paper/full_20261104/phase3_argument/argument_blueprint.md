# Argument Blueprint

Phase 3，2026-11-04。`criteria_binding_unavailable`：不声称符合期刊要求。

## Central Thesis

当路线评价需要解 SOCP 时，DRL 训练不必给每条路线都调用求解器：一次精确解给出的可证上界可以提供大部分训练信号，只用少量精确验证，就能以约 1/78 的求解次数得到质量接近全精算 POMO 的策略。代价是合成集上仍有可测的差距，且差距随规模变化。

## Sub-Arguments

### SA1：一次精确解足以给所有访问顺序提供上界

- **Evidence：** Proposition 1；`src/CertifiedUpperBound.py` 的实现和训练中逐次断言。
- **Reasoning：** 飞行约束只依赖目标本身，顺序只影响载具段，所以固定起降点后任何顺序都是可行解。
- **Counter-argument：** 这个结论太简单，不算贡献。
- **Rebuttal：** 承认结论简单，贡献放在训练中的用法。证明之所以短，是因为利用了 CVTSP 的结构。

### SA2：这个上界足以驱动训练，尽管它并不紧

- **Evidence：** 求解次数 2.89 次/实例；训练实例少 15 倍；上界收紧 0.32% 没有带来效果（多 C0 实验）。
- **Reasoning：** 优势估计和局部搜索需要的是排序信息，而且最优路线附近的上界是紧的，所以廉价和有保证比紧更重要。
- **Counter-argument：** 松的上界会误导学习。
- **Rebuttal：** 接受部分：专家在进入 IL 前都做精确验证，所以错误的上界不会变成错误的标签；RL 项里的误差确实存在，消融 B 用来检验。

### SA3：质量接近 G200，但不是全面优于它

- **Evidence：** Real529 MoR +0.21%；test500 不增强 +2.60%；泛化 8 视角 +0.72%；J200 −1.25%。
- **Reasoning：** 按规模和协议分别报告，差距的大小和方向才清楚。
- **Counter-argument：** 差 2.6% 不能算接近。
- **Rebuttal：** 承认这是局限。不增强时差距明显，8 视角下缩小，但 E133 的 test500 8 视角尚未跑（占位）。不把 Real529 的数字当作主要证据。

### SA4：节省的是训练代价，不是推理时间

- **Evidence：** 测试时间相同（0.78 s）；Gurobi 调用 78×；实例 15×。
- **Reasoning：** 两个策略用相同的网络和测试协议，差别只在训练。
- **Counter-argument：** G200 的预训练只需一次，推理时才重要。
- **Rebuttal：** 概括到新分布、新约束时，每次重训都要付这笔代价。

### SA5：方法有明确的适用边界

- **Evidence：** ILS、VNS 在 10 倍时间下更好，尤其是聚类实例；POMO 路线重合度高于 G200。
- **Reasoning：** 要求快速出解时才有优势，时间充足时元启发式仍更好。
- **Counter-argument：** 那 DRL 就没有必要。
- **Rebuttal：** 看时间预算：3 倍时间时，ILS 和 VNS 都差于 E133。

## Logical Flow

问题代价结构 → 文献未处理训练代价 → 可证上界 → 用上界训练 → 质量（差距如实）→ 代价（主要收益）→ 适用边界。

## Argument Strength Assessment

| Sub-Argument | Evidence Strength | Logic Validity | Counter-Arg Risk |
|---|---|---|---|
| SA1 | Strong | Valid | Medium（简单性） |
| SA2 | Moderate（消融未完成） | Qualified | Medium |
| SA3 | Moderate（缺 E133 8 视角，单种子） | Qualified | High |
| SA4 | Strong（G200 CPU 为估计） | Valid | Low |
| SA5 | Moderate | Valid | Low |

## Notes for Draft Writer

- 不写 beats、outperforms G200；写 close to、remains behind。
- J200 优势只作为 25 个算例上的观察。
- 表格以逐例平均为主。G200 的 CPU 标 estimated，墙钟注明另一台机器。
- 新颖性都带检索范围。Liu、Yan、Lee 只写题目能确认的内容。
- 未完成的结果用红色占位，不写估计值。
