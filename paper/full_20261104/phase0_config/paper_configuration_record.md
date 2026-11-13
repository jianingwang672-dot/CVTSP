# Paper Configuration Record

用户于 2026-11-04 确认。字数沿用默认值，作者和基金留占位。

| Parameter | Value |
|---|---|
| Title | Deep Reinforcement Learning with Sparse Exact Feedback for the Carrier–Vehicle Traveling Salesman Problem |
| Research Question | 路线评价需要解 SOCP 时，用一次精确解得到的可证上界，能否大幅减少训练所需的实例数和精确求解次数，同时保持接近 POMO + 全精算（G200）的解质量？ |
| Paper Type | IMRaD |
| Discipline | 运筹优化 / 物流运输 + 组合优化中的深度强化学习 |
| Target Journal | Transportation Research Part E |
| Venue Profile | absent |
| Citation Format | Author-year（`elsarticle-harv`） |
| Output Format | LaTeX（`elsarticle`） |
| Body Language | EN |
| Abstract | EN-only |
| Word Count Target | 9,500 words（默认值，用户未修改） |
| Existing Materials | 代码、结果文件、实验笔记、2026-11-04 文献检索结果、Li 2025、Wu 2026、proposal |
| Co-Authors | 占位，待用户提供 |
| Funding / COI | 占位，待用户提供 |
| Style Profile | null |
| Domain Evidence Profile | cs_ml |
| Citation Verification | advisory (mark only, default) |
| Retraction Policy | advisory (mark only, default) |
| Operational Mode | full |

## Notes

- `paper/trE_overleaf/main.tex` 不作依据。方法取自代码，数字取自结果文件。
- 不写 "beats G200"；结果按 J 分档报告；模型选择用 validation2000，测试用 test500。
- `criteria_binding_unavailable`：系统内没有 TR-E 的评审标准，不声称符合期刊要求。
