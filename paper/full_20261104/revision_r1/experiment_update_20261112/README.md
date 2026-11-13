# 实验部分更新记录 — 2026-11-12

> 历史记录：随后作者要求全部只展示八视角质量结果，并将 Real529 前移到 §5.2。最新修改和核验见 `../augmentation_revision_20261112/README.md`。本目录保留当时的证据；原 A 组质量表已从最新正文撤下，表号和章节位置已变化，本目录旧同步及范围断言脚本不适用于最新正文。

## 作者确认的范围

原始请求：“用这个skill的哪个帮我更新一下实验部分，文章风格还是沿用现在的”。

本轮向作者提出的范围：“只更新第 5 节：补全已完成的 A/B/D 消融实验和 Gurobi 对照、核正训练成本及表格；保持现有文风，摘要、方法、讨论和结论暂不改。”作者回复：“确认，按这个范围更新”。

使用 academic-research-suite 的 academic-paper/revision 写作指导及证据、文风和范围保护规则。该包的自动修订补丁链使用 Markdown HTML 区块标记，不直接套入 LaTeX；本轮明确采用 LaTeX 局部差异补丁、原文快照、数据复算及范围断言作为适配方式。本记录不是 #670 授权见证或完整 ARS 审稿/完整性流程通过的证明。

## 已更新

1. §5.5：新增完整混合整数 Gurobi 对照表（表 8）。报告 1,029 个结果文件中的状态、可行解数量、上/下界、相对求解器差距和时间，并明确各均值的分母。没有可行解的实例不被赋予无限成本。E133 相对于限时上界的优势不被解释为全局最优性保证。
2. §5.6：新增 A 组预算检查点表（表 12），区分新的 A 组重训和旧的等求解预算运行。A24h 的算术均值差 +0.293%，逐实例平均相对差距 -0.170%；正文解释两种指标的反转，不宣称任一策略全面胜出。A 组只在 validation2000 上评估，未补做测试实验。
3. §5.6：E133 累计求解器运行时间改为 59.8834 h，平均 0.05614 s/call。初始表搜索 67.6116 h，刷新混合时间 26.8652 h 的求解部分已被前者的求解器统计包含，不能相加。只报告有依据的累计工作进程时间范围，不把它当作实测进程 CPU 时间。
4. §5.7：新增 E100 的 B/C/D 按规模结果（表 13），保留单次训练、非等求解预算、搜索配置共同变化的限制。B 总体区间跨零，不写为质量改善或等价性证明；D 总体变差，但不能单独归因多样性。
5. §5.7：修正遗漏调用。B 的 100,000,000 次精确邻域求解另记于 exact_progress.jsonl，CSV 中另外包含目标奖励等调用，训练总量为 102,762,944。C 含刷新为 2,643,875，B/C=38.8683，不是旧文中的 101.8 M/2.3 M。D 含刷新为 2,111,419。各组另有 200,000 次训练期间验证求解，未混入训练总量。

## 原始证据与复算

`verify_results.py` 只读取已保存的 JSON/CSV/JSONL，不调用模型、训练、求解器或网络。它复算 95% 正态区间、配对差距、胜平负、预算、计时及 Gurobi 汇总，并输出 `evidence.json`。该文件含输入文件的 SHA-256。

关键原始文件（相对项目根）：

- `outputs/ablation_20261028/test500_aug8_e100/results.json`
- `outputs/ablation_20261028/exact_search/train/metrics.csv`
- `outputs/ablation_20261028/exact_search/train/exact_progress.jsonl`
- `outputs/ablation_20261028/no_diversity/train/metrics.csv`
- `outputs/progressive_recipe_20261020/train/metrics.csv`
- `outputs/pomo_gurobi_ablationA_20261030/validation2000/results.json`
- `outputs/pomo_gurobi_ablationA_20261030/train/budget_checkpoints.json`
- `outputs/pomo_gurobi_ablationA_20261030/train/time_checkpoints.json`
- `outputs/pomo_gurobi_equal_budget_20261027/eval_equal_solves_val2000/results.json`
- `outputs/full_gurobi_20261030/real529_3600s/*.json`
- `outputs/full_gurobi_20261030/test500_3600s/*.json`
- `outputs/progressive_recipe_20261020/final_external_e133/real529/results.json`
- `outputs/pomo_gurobi_equal_budget_20261027/eval_equal_solves_test500_aug8/results.json`

最后一个文件才是 E133 的增强 test500 对照。`final_external_e133/test500/results.json` 是无增强评估，未混入八视角比较。

## 原文保护与中英同步

修改前的四个文本文件在 `before/` 中。`verify_results.py` 逐字节断言英文和中英稿第 5 节以外均未修改；引用键及顺序也单独核对不变。

`sync_bilingual.py` 用原文单元内容匹配保留既有翻译，仅对 `zh_updates.txt` 中明确写出的 18 个更新单元更换翻译，并机械重排翻译编号。现有双语生成器据更新后的英文主稿重建双语稿，结果逐字符相同。正文和方法图不做整体重写。

后续作者澄清：“这个baseline是我之前训练过的”。据此将 §5.1 明确为作者此前自行训练的 G200 基线，本次使用其第 200 轮检查点；删除“没有为本文重新训练”这一容易引起误解的说法。两套完整训练方案的比较范围和另行重训实验的说明保持不变。

作者随后要求删除 §5.1 的“在选择范围内，验证均值非常接近：E133 为 17.703，E150 为 17.705。”及对应英文。已同步删除，未改其他实验内容。

正文计数参考：TeXcount 的主文本词数由 9,695 增至约 10,455；标题和题注另计，不含文献表。旧配置的 9,500 是默认目标而非作者指定硬上限。本轮新增已完成实验的报告，未为凑字数改写其他章节。

## 验证边界

- 英文和中英 PDF 均通过 latexmk 编译；新增表格已用 Poppler 渲染并目视检查，无新增表格裁切或越界。
- 书目仍有五项既存的 empty-pages 警告，未改动参考文献资料。
- 技能的独立 PDF 结构预检因 pypdf 未安装而为 `UNAVAILABLE`，不是 `PASS`。对应 sidecar 为 `preflight_english.json`、`preflight_bilingual.json`。这与编译及 Poppler 渲染成功是不同检查。
- 缩写检查使用最终英文 PDF 提取的纯文本；其输出见 `acronyms.md`、`acronyms.json`，覆盖为 `partial`，含既有缩写和提取造成的提示，未据此改写范围外内容，不宣称缩写检查全通过。
- 本轮没有新增训练、测试或求解，没有进行正式模拟审稿，也没有重做全文文献真实性/引用支持核验。

## 明确保留的未完成项

1. §5.4 的两个原有占位仍保留：合并 Vieira 等人的逐实例参考值，以及合并后核验潜在更优解。现有文件中没有相应逐实例数据，不能用论文中的分组均值替代。
2. 按作者确认，本轮没有重写摘要、方法、讨论和结论。摘要没有重新生成；其中既有的主要数字未被本轮结果推翻。
3. §6.1 仍有旧的“154.3 核时/约 6,000 实例”说法，未获本轮修改范围授权，故保持原字节。它必须在后续讨论部分同步修订，不能在投稿前与本轮的非重复计时说明并存。按目前日志，累计求解加搜索时间只能给出范围，GPU 和未记录开销也不在内。
4. 引言贡献占位等其他章节的历史未定项本轮没有处理。因此两个 PDF 是更新后的工作稿，不是已经通过全文一致性与投稿就绪检查的最终稿。
