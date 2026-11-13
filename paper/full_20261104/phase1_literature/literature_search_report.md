# Literature Search Report

Phase 1，2026-11-04。沿用同日 research gap 检索，并补做逐条书目核验。

## Search Strategy

- **来源：** OpenAlex（28 组关键词）、arXiv API、Crossref、网页检索。Semantic Scholar 因限流未用上；Scopus、Web of Science 未查。
- **施引追踪：** Garone 2011、Garone 2014、Gambella 2018、Erdoğan 2021、Poikonen 2019、Li 2025、Vieira 2026 的施引文献，用学习类词过滤后得到 46 条唯一记录。
- **最后检索：** 2026-11-04。
- **概念组：**
  - 问题：carrier-vehicle TSP、mothership and drone routing、vessel-UAV routing、continuous take-off and landing。
  - 学习：deep reinforcement learning、neural combinatorial optimization、POMO、self-improvement。
  - 代价：expensive objective、surrogate、upper bound、second-order cone program、fixed-order subproblem。
- **纳入：** 期刊、会议和 arXiv 预印本（`cs_ml`），与 RQ 的问题、方法或最接近工作直接相关。
- **排除：** 与路径问题无关的通用强化学习、只有标题匹配的记录。

这是聚焦检索，不是系统综述。所有"未发现"都只在上述范围内成立。

## Coverage Distribution Advisory

语料偏向两类：运筹期刊的 CVTSP 精确和启发式方法，机器学习会议的神经组合优化。两者几乎不互相引用，这本身支持研究空白，但也意味着单一数据库容易漏掉交叉工作。投稿前需用 Scopus 或 Web of Science 复查。

## Screening Results

- 书目核验：下列 45 条中，44 条通过 Crossref DOI 或 arXiv ID 核到了真实记录，Zhao 2022 以上传的 PDF 为准。
- `refs.bib` 原有 26 条全部保留为候选，但只在 Phase 4 实际用到时才进入新文献表。
- **阅读级别：** A = 读过全文或关键章节；B = 读过摘要或检索摘要；C = 只核实书目。

## Annotated Bibliography: 最接近的工作

### Li, Zhou & Côté. Exact Methods and a Two-Stage Iterative Heuristic for the CVTSP. INFORMS JoC. 10.1287/ijoc.2025.1140
- **Type / 阅读：** Journal article / A（§5.1–5.3）
- **Method：** MISOCP、分解和 TSIH 启发式
- **Key Findings：** 用近似矩阵 T 指导访问顺序搜索；解完 CVP 后只更新走过的弧，其余保持估计值。
- **Relevance：** 与 G3 最接近的工作。
- **Quality：** 高。
- **Method weaknesses：** T 没有上界性质，需要靠多样化重新生成。
- **Potential Use：** §2.1、§2.4、§5.4 的 BKS 来源。
- **书目注意：** Crossref 的出版年是 2026，Phase 5 按最终卷期号引用。

### Vieira, Frazzon & Coelho (2026). An Exact Linearization-Based Refinement Algorithm to the CVTSP. Transportation Science. 10.1287/trsc.2025.0292
- **Type / 阅读：** Journal article / A（CIRRELT-2025-22 预印本，正式版未读）
- **Method：** 线性化精炼的精确算法
- **Key Findings：** 解到全部 25 个客户以内的基准算例，证明一个 35 客户算例最优，覆盖 10–200 客户。
- **Relevance：** 精确方法的最新边界，Real529 的 BKS 要与它核对。
- **Quality：** 高。
- **Method weaknesses：** 大规模仍需长时间计算；预印本未与 TSIH 比较。
- **Potential Use：** §1、§2.1、§5.4。

### Ren, Li, Sun, Bai & Chen. A tractable model and heuristic for vessel routing with multiple UAVs. TR-E. 10.1016/j.tre.2026.105178
- **Type / 阅读：** Journal article / B
- **Method：** MISOCP + PALNS，含敏感性分析
- **Key Findings：** 把 CVTSP 扩展到一艘船带多架无人机。
- **Relevance：** 说明选题和期刊相符，也是敏感性分析的写法参照。
- **Quality：** 高（期刊已接收）。
- **Method weaknesses：** 全文未读。
- **Potential Use：** §2.1、§6。
- **书目注意：** Crossref 的出版年是 2027，Phase 5 按最终卷期号引用。

### Kim et al. (2024). Symmetric Replay Training. arXiv 2306.01276
- **Type / 阅读：** Conference paper / A
- **Method：** 利用解的对称性复用昂贵目标的评价
- **Key Findings：** 面向昂贵目标，但只复用已评价解，没有给未评价解的界。
- **Relevance：** G2 的反例，所以不能写"所有方法都假设评价便宜"。
- **Quality：** 高。
- **Method weaknesses：** 对称性是否适用取决于问题。
- **Potential Use：** §2.3。

### Sun, Fügenschuh & Vaze (2026). Physics-aware truck and drone delivery planning. C&OR. 10.1016/j.cor.2026.107665
- **Type / 阅读：** Journal article / A（arXiv 版前约 10 万字符）
- **Method：** 离线训练神经网络估计无人机时间，再用先排序后分割的启发式
- **Key Findings：** 用学习代理替代轨迹优化。
- **Relevance：** 路径问题中处理昂贵子问题的邻近工作。
- **Quality：** 高。
- **Method weaknesses：** 读到的部分没有上界保证，路线不由 DRL 决定。
- **Potential Use：** §2.3。

### Fan et al. (2025). A Unified DRL Approach for Close Enough TSP. arXiv 2510.03065
- **Type / 阅读：** Preprint / A
- **Method：** 离散化路点，奖励为路径长度
- **Key Findings：** 训练时不调用求解器；SOCP 只出现在基线方法里。
- **Relevance：** 连续路点问题的 DRL 邻近工作。
- **Quality：** 中（预印本）。
- **Method weaknesses：** 离散化绕开了连续代价，不处理本文的训练代价问题。
- **Potential Use：** §2.2。

### Liu, Yu, Tian & Sun (2026). Routing optimization for an eVTOL-and-drone delivery system in continuous space with no-fly zones. TR-E. 10.1016/j.tre.2026.104741
- **Type / 阅读：** Journal article / C
- **Relevance：** 同一期刊的连续空间 RL 路径工作，可能影响 G1。
- **Potential Use：** §2.2，**读到全文前不描述其方法**。

### Yan et al. (2026). Mothership-based truck-multi-drone cooperative delivery to moving targets. SSRN. 10.2139/ssrn.7371806
- **Type / 阅读：** Preprint / B
- **Relevance：** 母舰式配送的多智能体 RL，可能影响 G1。
- **Potential Use：** §2.2，读到全文前只写题目可确认的内容。

### Lee, Park, Lee & Bang (2023). Sequential Cooperative RL of Mothership Routing Problem. AIAA SciTech. 10.2514/6.2023-2658
- **Type / 阅读：** Conference paper / C（Crossref 已核实；全文和 KAIST 硕士论文未读）
- **Relevance：** 母舰路径 RL，可能影响 G1。
- **Potential Use：** §2.2，读到全文前不描述其方法。

## 其余来源

| 主题 | 来源 | 标识 | 阅读 |
|---|---|---|---|
| CVTSP 起源 | Garone et al. 2008, CDC | 10.1109/cdc.2008.4739357 | C |
| | Garone et al. 2010, CDC | 10.1109/cdc.2010.5717171 | C |
| | Garone, Naldi & Casavola 2011, JGCD | 10.2514/1.50539 | C |
| | Garone, Determe & Naldi 2012, CDC | 10.1109/cdc.2012.6426488 | C |
| | Garone, Determe & Naldi 2014, JGCD | 10.2514/1.62126 | C |
| CVTSP 求解 | Gambella, Lodi & Vigo 2018, Transp. Sci. | 10.1287/trsc.2017.0771 | B |
| | Erdoğan & Yıldırım 2021, Transp. Sci. | 10.1287/trsc.2020.0999 | B |
| | Poikonen & Golden 2020, IJOC | 10.1287/ijoc.2018.0879 | B |
| | Fahradyan et al. 2020, LNCS | 10.1007/978-3-030-43890-6_14 | C |
| | Li, Wang, Zhou & Wang 2024, C&OR | 10.1016/j.cor.2024.106723 | C |
| 无人机协同路径 | Murray & Chu 2015, TR-C | 10.1016/j.trc.2015.03.005 | C |
| | Agatz, Bouman & Schmidt 2018, Transp. Sci. | 10.1287/trsc.2017.0791 | C |
| | Otto et al. 2018, Networks | 10.1002/net.21818 | C |
| | Macrina et al. 2020, TR-C | 10.1016/j.trc.2020.102762 | C |
| | Chung, Sah & Lee 2020, C&OR | 10.1016/j.cor.2020.105004 | C |
| | Zhao et al. 2022, TR-E 168:102967 | 上传 PDF | C |
| 神经组合优化 | Vinyals et al. 2015 | arXiv 1506.03134 | C |
| | Nazari et al. 2018 | arXiv 1802.04240 | C |
| | Kool et al. 2019 | arXiv 1803.08475 | C |
| | Kwon et al. 2020（POMO） | arXiv 2010.16011 | B |
| | Kim et al. 2022（Sym-NCO） | arXiv 2205.13209 | C |
| | Hottung et al. 2022（EAS） | arXiv 2106.05126 | C |
| | Wu et al. 2022, TNNLS | 10.1109/tnnls.2021.3068828 | C |
| | Bengio et al. 2021, EJOR | 10.1016/j.ejor.2020.07.063 | C |
| | Mazyavkina et al. 2021, C&OR | 10.1016/j.cor.2021.105400 | C |
| | Wu et al. 2026, TR-E 209:104742 | 10.1016/j.tre.2026.104742 | B |
| 自改进 | Luo et al. 2023（LEHD） | 10.52202/075280-0387 | B |
| | Pirnay & Grimm 2024 | arXiv 2403.15180 | B |
| | Luo et al. 2024（SIL） | arXiv 2403.19561 | B |
| | Luttmann & Xie 2025（MACSIM） | arXiv 2510.12273 | B |
| 代理和昂贵目标 | Ferber et al. 2022/2023（SurCo） | arXiv 2210.12547 | B |
| 工具 | Lobo et al. 1998, LAA | 10.1016/s0024-3795(98)10032-0 | C |
| | Lin & Kernighan 1973, Oper. Res. | 10.1287/opre.21.2.498 | C |
| | Helsgaun 2000, EJOR | 10.1016/s0377-2217(99)00284-2 | C |
| | Williams 1992 | 10.1007/978-1-4615-3618-5_2 | C |
| | Vaswani et al. 2017 | arXiv 1706.03762 | C |

## Literature Matrix

| 工作 | 连续子问题 | 精确反馈用于训练 | 对未评价路线有界 | 上界覆盖所有顺序 | DRL 决定路线 |
|---|---|---|---|---|---|
| Li 2025 | 是 | 否 | 否 | 否 | 否 |
| Vieira 2026 | 是 | 否 | 否（精确求解） | 否 | 否 |
| Ren 2026 | 是 | 否 | 否 | 否 | 否 |
| Kim 2024 | 否 | 否 | 否 | 否 | 是 |
| Sun 2026 | 是 | 离线数据 | 读到的部分没有 | 否 | 否 |
| Fan 2025 | 离散化 | 否 | 否 | 否 | 是 |
| Pirnay 2024 / Luo 2024 | 否 | 否 | 否 | 否 | 是 |
| **本文** | **是** | **是，稀疏** | **是** | **是** | **是** |

Liu 2026、Yan 2026、Lee 2023 未读，所以没有放进矩阵。

## Identified Gaps

1. **G3（核心）：** 检索范围内，未发现把一次精确原始解复用为所有访问顺序的可证上界、并同时用于 DRL 奖励和专家搜索的工作。
2. **G1（保守写法）：** 只说检索范围内未发现明确针对 CVTSP 的学习方法，不写"never"。
3. **G2（不单独成立）：** 只写"多数神经路径求解器的路线评价是闭式计算"，并点名 Kim 2024、Sun 2026。
4. **G4（依附 G3）：** 自改进的既有工作主要在 TSP、CVRP 等评价便宜的问题上。

Proposition 1 的证明很短，创新点要放在训练中的用法，不放在定理本身。

## Recommended Sources by Paper Section

| 章节 | 来源 |
|---|---|
| §1 | Garone 2011、Li 2025、Vieira 2026、Ren 2026、Kwon 2020 |
| §2.1 | Garone 系列、Gambella、Erdoğan、Poikonen、Fahradyan、Li 2024、Li 2025、Vieira、Ren |
| §2.2 | Kool、Kwon、Sym-NCO、EAS、LEHD、Wu 2026、Fan 2025、Liu、Yan、Lee |
| §2.3 | Kim 2024、Sun 2026、SurCo、Pirnay、Luo 2024、Luttmann |
| §3 | Garone 2011、Lobo 1998 |
| §4 | Kwon 2020、Williams 1992、Lin & Kernighan、Helsgaun 2000 |
| §5.4 | Li 2025、Vieira 2026 |
