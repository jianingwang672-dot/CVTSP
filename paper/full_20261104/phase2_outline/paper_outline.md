# Paper Outline

Phase 2，2026-11-04。结果数字见 [numbers_ledger.md](numbers_ledger.md)。

## Structure Pattern: IMRaD（按运筹期刊习惯，方法部分拆成问题和方法两节）

## Overview

论文先说明 CVTSP 的路线评价本身是 SOCP，这让训练 DRL 的代价很高。然后提出核心机制：一次精确解给出对所有访问顺序都成立的上界，训练和搜索的大部分信号都从这张表里取，只有少量路线做精确验证。实验先在合成集上看质量和泛化，再看文献基准，最后量化训练代价和各部分的作用。

## Review Criteria Coverage Plan

`criteria_binding_unavailable`：系统内没有 TR-E 的评审标准，本大纲不声称符合期刊要求。

## Detailed Outline

### 1. Introduction（约 1,100 词）

**Purpose：** 提出问题、空白和贡献。
**Serves：** RQ 框定。

- 1.1 实际背景：载具带无人机访问目标，起降点连续，影响总完工时间。
- 1.2 难点：顺序是离散的，起降点是连续的；固定顺序后是 SOCP。精确方法已解到约 25 个目标（Vieira 2026），更大规模靠启发式。
- 1.3 DRL 的障碍：测试时可以用，但训练要评价数亿条路线，每条都要解 SOCP。
- 1.4 研究空白（G3，带检索范围限定）。
- 1.5 贡献：
  1. 证明一次精确解给出覆盖所有访问顺序的可证上界。
  2. 用它驱动 RL 奖励和无求解器搜索，只对少量路线做精确验证。
  3. 与 POMO + 全精算相比，用约 1/15 的实例和 1/78 的求解次数，得到接近的质量，并按规模说明差距。

**Sources：** Garone 2011、Li 2025、Vieira 2026、Ren 2026、Kwon 2020
**Transition：** 从空白转到既有工作的三条线索。

### 2. Related Work（约 1,500 词）

**Purpose：** 建立三条文献线索，并收束到空白。

- 2.1 CVTSP 及其扩展：起源、精确方法、启发式、多无人机扩展。说明 Li 2025 的矩阵和本文上界的区别。
- 2.2 路径问题的 DRL：构造式方法、POMO、增强和 EAS；CETSP、母舰和连续空间 RL。Liu、Yan、Lee 的描述要等全文。
- 2.3 昂贵目标和自改进：Kim 2024、Sun 2026、SurCo；Pirnay、Luo、Luttmann。
- 2.4 空白小结：用文献矩阵收束，所有新颖性表述带检索范围。

**Sources：** 见 Phase 1 分章节来源表
**Transition：** 空白需要先把问题的代价结构写清楚。

### 3. Problem Formulation（约 700 词）

**Purpose：** 给出模型，点明上界依赖的结构。

- 3.1 CVTSP 定义、符号、目标函数和约束。
- 3.2 固定顺序的 CVP 和 SOCP 形式。
- 3.3 代价结构：飞行约束只和目标自身有关，顺序只影响载具段。这是 Proposition 1 的基础。

**Sources：** Garone 2011、Lobo 1998；模型以 `src/CVPSolver.py` 为准
**Transition：** 这个结构给出下一节的上界。

### 4. Methodology（约 2,300 词）

**Purpose：** 完整描述算法，可供复现。

- 4.1 总体框架：DRL 决定顺序，SOCP 决定起降点。
- 4.2 可证上界：
  - 按目标重排原始解，构造有向时间表。
  - Proposition 1 及证明：可行性、对锚点路线的紧性。
  - 使用条件：锚点须为最优解并通过可行性审计，上界在求解器容差内成立。
  - 多张表只能按整条路线取最小，不能逐项取最小。
- 4.3 策略网络和 POMO 解码。
- 4.4 无求解器搜索：起点选择、2-opt、LKH，O(1) 增量评估。
- 4.5 精确验证、模仿学习标签和混合 RL 奖励。
- 4.6 渐进方案：按 J 和 epoch 的搜索强度、near-miss、C0 刷新、IL 量归一化。
- 4.7 求解次数分析：每实例约 2.9 次。

**Sources：** Kwon 2020、Williams 1992、Lin & Kernighan、Helsgaun 2000；方法以 `src/CertifiedUpperBound.py`、`src/ProgressiveRecipeTrainer.py` 为准
**Transition：** 方法的效果需要从质量、规模和代价三方面检验。

### 5. Computational Experiments（约 2,500 词）

**Purpose：** 回答 RQ，并如实给出差距。

- 5.1 实验设置：数据集、选模（validation2000）、测试协议、指标定义（MoR 为主）、硬件。披露 test500 的查看情况。
- 5.2 合成集结果（主要证据）：test500 按 J 分档，不增强和 8 视角。**差距随 J 增大，要直接写出来。**
- 5.3 大规模泛化：J150–500。
- 5.4 文献基准 Real529：按子集和 J 分档，对比 BKS（含 Vieira）、G200、元启发式。J200 的优势要说明是小样本，不写成总体更优。
- 5.5 精确求解参照：完整 Gurobi 的 UB、LB 和时间。**[占位]**
- 5.6 训练效率：实例、求解次数、CPU、墙钟；24 h 曲线。**[占位]** G200 的 CPU 标为估计，墙钟注明不同机器。
- 5.7 消融：A、B、C、D。**[占位]** 明确哪些结论能从消融得出。

**Sources：** 数字台账
**Transition：** 从结果转到含义和适用边界。

### 6. Managerial Insights and Discussion（约 1,000 词）

**Purpose：** 给出实际含义和局限。

- 6.1 何时值得用：训练预算有限、需要在线快速出解的场景。
- 6.2 参数敏感性（速度比、续航、目标分布）。**[占位，尚未开始]**
- 6.3 与元启发式的取舍：时间充足时 ILS、VNS 更好，尤其是聚类实例。
- 6.4 局限：差距来自 POMO 多样性塌缩；上界对远离锚点的路线较松；目前单一种子。

**Sources：** Ren 2026、Zhao 2022（管理启示写法）
**Transition：** 收束到结论。

### 7. Conclusion（约 400 词）

总结回答 RQ，给出未来方向。

### 声明部分

Data Availability、Ethics、CRediT、Conflict of Interest、Funding、AI 使用声明。作者和基金目前为占位。

## Evidence Map

| 章节 | 证据 | 状态 |
|---|---|---|
| §1–2 | Phase 1 文献库 | 可写；G1 用保守写法 |
| §3–4 | 代码、`docs/` | 可写 |
| §5.1 | `docs/progressive_recipe_20261020.md`、manifest | 可写 |
| §5.2 不增强 | 台账 | 可写 |
| §5.2 8 视角 | 缺 E133 | **需新跑** |
| §5.3 | 台账 | 可写 |
| §5.4 | 台账；缺 Vieira BKS 和 Gurobi | 部分可写 |
| §5.5 | 队列 | 占位 |
| §5.6 | 台账；缺 24 h 曲线 | 部分可写 |
| §5.7 | 队列 | 占位 |
| §6.2 | 无 | 占位 |

## Word Count Summary

| Section | Target Words |
|---|---:|
| 1 Introduction | 1,100 |
| 2 Related Work | 1,500 |
| 3 Problem Formulation | 700 |
| 4 Methodology | 2,300 |
| 5 Computational Experiments | 2,500 |
| 6 Managerial Insights and Discussion | 1,000 |
| 7 Conclusion | 400 |
| Total | 9,500 |
