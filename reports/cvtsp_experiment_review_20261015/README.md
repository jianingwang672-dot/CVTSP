# CVTSP实验完整复盘资料包

证据截点：2026-10-15 15:57:55（UTC+8）。PDF共31页，包含23个主章节和5个附录。

建议先阅读`CVTSP_实验完整复盘_500epoch至LKH.pdf`；Word版本可编辑，HTML版本可离线阅读并点击目录，Markdown版本便于检索。

报告覆盖旧500轮训练及外部测试、独立完整POMO选模、上界选路诊断、sample测试、IL/学习率控制分支、200轮重训、三组搜索算子实验、三组near-miss配额、提前停止的质量加权IL、LKH搜索pilot和当前LKH训练。

配套CSV包含284行总体/逐规模外部结果、三个训练在共同开发集上的选模结果、RL/IL更新次数和指标快照。`evidence_snapshot.json`保留提取数据，`source_manifest.json`列出83份原始证据文件的路径和读取时SHA256。

原始证据链接指向本项目中的文件；将资料包拷到其他电脑后，报告、图表和配套数据仍可独立阅读，但这些原始文件链接需要原项目目录。资料包不含训练checkpoint或全量逐实例搜索日志。

本次完成审计见`completion_audit.json`；已核对外部结果的均值、gap、配对计数、逐规模加权、选模最小值、相同样本量下的更新次数、文档内容和PDF数值，并抽查目录、结果页与逐规模表格的实际渲染。

复算入口：`build_review.py`生成当前项目证据截点的报告与数据，`export_formats.py`输出Word/HTML，`audit_review.py`核验已有交付物。运行生成脚本会更新截点；已交付文件的校验和见`DELIVERY_SHA256SUMS`。
