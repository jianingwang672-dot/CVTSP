# CVTSP会议诊断记录：Grill / Discovery Notes
Date: 2026-10-14 · Goal: 核对新B组训练实际记录是否覆盖会议要求，明确剩余诊断需求。

## Summary / key decisions
- 用户已选择仅运行B组，从头20轮，near-miss固定20%，IL5遍；本轮不改RL损失或权重公式。
- 用户确认新增Target比较路线日志在“下次训练启用（推荐）”；当前B组连续完成20轮，不为这项日志重启或严格续训。
- 本次用户询问记录覆盖情况，未授权改变正在运行的训练或追加实验。
- 已核验B组Epoch1–3实际diagnostics产物，会议要求的local search步数、swap/2-opt贡献、loss/梯度/reward记录被保留。
- B新增每轮IL入口的标签精确VT/VE、g/q、来源、重复次数、按逻辑batch归一化的CE贡献份额，以及额外精算和耗时记录。
- 现有记录含全量搜索统计与每个观测J的最多2个实例详细链条抽样；逐算子贡献是上界下降，不是逐move真实makespan。
- RL/IL阶段评价不是每轮全量精算：单greedy在Epoch1/5/10/20，完整POMO50例在Epoch5/20。
- 目前尚缺逐标签对应的Target比较路线；仅存精确VT数值和Target模型快照hash，无法从该日志直接复算那条路线。
- 两个RL分项的梯度夹角只测诊断epoch首batch；RL与IL在同一状态/同一批数据上的梯度夹角未测。

## Q&A log

### 用户本次关注 — 会议中要求的具体过程记录
- Asked: “新的训练中关于需要记录的具体过程有修改吗？会议中说的”
- Captured: 用户希望核对会议要求与新训练的过程记录，不能只给最终成绩。
- Flags: 哪一项尚未覆盖的过程诊断优先补充 -> 阅读证据后再向用户询问。

## Evidence / verification
- 会议原文：/home/marmot/.codex/attachments/8e1f33c5-d1c1-407d-9f13-2b4b8f462a22/咔咔咔咔的快速会议-元宝纪要.txt。
- 会议要求记录训练曲线与验证指标（16:49、29:04）、local search实际步数（30:56）、swap/2-opt正向贡献（39:19、40:59）、loss/梯度/reward（34:29）。
- LKH、增强、实例预算/LR对照属于额外实验，不能因为有诊断日志就称这些实验已经完成。
- 旧核对报告：outputs/search_ablation_20261012/meeting_diagnostics.md；当前实现：src/SearchAblationTrainer.py、src/QualityILTrainer.py。
- 实际已生成epoch-001/002/003_summary.json：每步RL/IL梯度裁剪前后范数、每batch奖励/优势分布、搜索步数直方图与预算命中、每算子接受/评估数和上界下降、RL/IL参数净变化和耗时。
- epoch-003_search_samples.jsonl：initial/final路线、完整上界history、move history、终止原因等详细抽样；统计覆盖所有候选，详细链条不是全量保存。
- epoch-003_il_labels.jsonl：fingerprint、expert_sequence、J、source、effective_copies、target_exact、expert_exact、relative_gap、quality_multiplier；没有target_sequence字段。
- B原有quality_il记录在训练开始前实现并在实际产物核验；最初核验未改源码。收到Q1回复后，仅补充下一次进程加载时保存Target路线和所选上界的字段，当前训练未重启。
- 用户确认优先补存用于比较的Target路线。当前训练保持运行；新增记录只能在新启动或严格恢复的进程中生效，不能声称已有历史记录包含这些路线。

### Q1 — 可复算标签质量的记录粒度（已回答）
- Asked: 是否优先补存用于比较的Target路线，使其与已有Expert路线、VT/VE和权重配对？
- Recommendation: 补存；先让标签质量与权重可逐例复算，不全量保存所有候选move轨迹。
- Captured: 用户原话：“优先补存Target比较路线（推荐）”。确认新增Target比较路线记录的优先级；保持当前训练运行的约束不变。
- Flags: 应用时点已由Q2确认，当前训练不自动中断。

### Q2 — 新日志的应用时点（已回答）
- Asked: 新日志在下一次训练启用，还是本轮最近checkpoint保存后严格恢复启用？
- Recommendation: 下一次训练启用；当前B的原会议诊断已覆盖核心要求，优先保持本轮连续完成20轮。
- Captured: 用户原话：“下次训练启用（推荐）”。新增Target比较路线记录只在下次训练启用，不中断本轮20轮B组训练。
- Flags: 无需当前重启；下次启动时核验实际日志出现target_sequence和target_selected_upper_bound字段 -> Codex。

## Open flags (pending input)
- 本次记录范围和应用时点已确定，无待用户回答事项。
- 下次启动训练时核验新日志字段实际生效 -> Codex；不因此自动启动额外训练。
- RL多batch分项梯度与RL/IL梯度夹角仍未覆盖，属于未来诊断候选，未获追加实验授权。

## Session close / indexing
- 已核验新增测试：日志target_sequence与实际送入精算器的路线一致，未误存Expert路线或其他POMO候选；5项quality IL单元测试通过。
- 本次讨论完成；已整理到notes/INDEX.md的本地索引，可用项目文件检索查找。未声称已写入Notion或外部语义检索系统。
- 可复用内容候选：会议诊断覆盖清单；仅提出整理建议，未自动修改项目长期指令或技能。
