# CVTSP 验证与选模方案：Grill / Discovery Notes
Date: 2026-09-22 · Goal: 明确 Reference 更新、checkpoint 选择和最终测试的验证协议，并找出 E100 后训练停滞、未超过 Gurobi E200 的可检验改进路径。

## Summary / key decisions

- 讨论中的两项验证改进：每 50 个 epoch 更换验证实例；每例改为完整 POMO=J 精确评测。用户进一步提出 Reference 晋升与 checkpoint 选模使用同一批验证实例，每 50 epoch 换一批。Reference 晋升虽不直接指定最终 checkpoint，但会改变后续 Expert 与训练轨迹。跨验证批次的绝对 makespan 不可直接比较，最终全程选模方法待定。
- 新分支：用户怀疑学习率衰减过慢，提出“每个epoch设置2万instance，然后200个epoch结束，学习率衰减趋势变快”。尚未决定是否执行。
- 用户进一步希望获得能区分原因的实验次序，而不是直接把多个改动合并重训；目前偏好单套验证实例同时用于 Reference 晋升和阶段内选模、每 50 epoch 换批，需解决跨窗口选模。
- 用户当前关切：即使 500 epoch 后最佳仍是 E100，它也未超过 Gurobi E200。需要优先找到能改善真实完整 POMO 泛化的训练改动，而非只改善选模。Gurobi 基线 E200 已见约 2000 万训练实例；当前 Self-improving E100 约 100 万、E500 约 500 万，样本预算不同。
- 用户明确授权先执行两项诊断：①同批新实例上比较 E100/E365 的上界选中 Target 路线与完整 POMO 精确最优路线；②从 E100 分别运行 IL 内训 5→2 和更快学习率衰减的短程控制实验。其余方案暂缓。

## Initial proposal
- 用户原话："1.validation不用固定的用随机的，比如学习50个epoch更换一套validation。2.不用一条free-start也用J条。"
- 背景：旧固定验证每个 epoch 用同一批 J=20/40/60/80/100 各 200 例，`reference_validation_candidates=1`，即每例一条 greedy free-start 路线；人工 5000 例使用完整 POMO=J、无增强、Gurobi 精算，Real529 使用 POMO=min(J,100)、8 倍增强、Gurobi 精算。
- 观察：Epoch 100→365，旧固定集 18.218866→18.139799，但人工 5000 例 17.751879→17.855958，Real529 333.459454→334.177603。

## Q&A log

### Q1 — 轮换验证集的用途
- Asked: 每 50 个 epoch 更换一批 validation，主要用于 Reference 晋升、checkpoint 选模，还是两者？建议只用于训练中的 Reference 晋升，选模和最终测试使用独立数据。
- Captured: 用户先说：“选模不是根据固定集吗？那reference更不更新都和选模无关啊”；后续补充：“他们不能是同一个数据集吗？reference更新和选模。每50个epoch换一批”。当前偏好：同一批数据可承担两项决策，但每 50 epoch 更换该批数据。Reference 更新会间接影响之后的训练，不直接决定被报告的最佳 checkpoint。
- Flags: 跨 50-epoch 窗口如何统一选择 checkpoint -> 用户。

### 用户提出的新分支 — 学习率、每轮实例数和总轮数
- Captured: 用户原话：“会不会和学习率有关，其实我是不是应该每个epoch设置2万instance，然后200个epoch结束，学习率衰减趋势变快”。
- Code facts: 当前训练是每轮 10000 例、batch 64、500 epoch，`cosine_warmup` 按 epoch 更新学习率；每轮 157 次 RL 优化步、约 405 次 IL 优化步。提议方案每轮约 313 次 RL 步，IL 在当前分组/批量规则下仍约 405 步；总实例数从 500 万变为 400 万，且 RL/IL 更新比例也改变。
- Flags: 是否把学习率作为单独变量做消融、或把 2 万例/200 轮作为联合新方案 -> 用户；Q1 仍待答。

### 用户追问 — 怎样安排诊断和改进实验
- Asked: “所以综合考虑你认为我们应该做什么实验去得出改进思路”
- Captured: 用户希望综合验证集轮换、完整 POMO、学习率和每轮实例数，安排能辨别改进来源的实验。
- Flags: 用户提出单套验证集共用并每 50 epoch 换批；跨窗口选择方法待定 -> 用户。

### Q2 — 共用验证集及轮换窗口
- Asked: 用户原话：“他们不能是同一个数据集吗？reference更新和选模。每50个epoch换一批”。
- Captured: 可以共用；用户明确希望每 50 个 epoch 换一批。尚未明确每个 epoch 是否仍评估 Reference，暂不将“换批”理解成“只每 50 轮评估一次”。
- Flags: 由于不同窗口实例难度不同，最终如何用共同标准比较各窗口的候选 checkpoint -> 用户。

### 用户追问 — E500 可能仍不及 Gurobi E200
- Asked: “我估计500个测完最好的也是epoch100了，但是他还是没有超过gurobi epoch200，这该怎么办”
- Captured: 用户目标不只是选出当前最佳，还希望训练方法在真实外部测试上超过 Gurobi 训练基线 E200。
- Facts: 同配置人工 5000 例 E100=17.751879、Gurobi E200=17.282643；Real529 E100=333.459454、Gurobi E200=332.420655。Gurobi E200 训练到约 2000 万实例；当前 E100 为约 100 万，E500 为约 500 万。E100→E365 的外部退步说明单纯增加当前配方的数据/轮数未带来持续收益。
- Flags: 需先用控制变量实验定位学习率、IL 频率、上界奖励/精确目标、Reference 验证反馈各自的影响。

### Q3 — 先执行的实验
- Asked: 前述建议的第 1 项离线奖励/选路诊断与第 2 项 E100 两条短程控制分支。
- Captured: 用户原话：“先执行1和2”。授权立即实施离线诊断与两个 E100 分支；不把 2 万例／200 轮合并为当前实验。
- Flags: 资源安排、学习率衰减终点、开发集评测频率由实验方案预先固定并记录；当前 E500 训练不得无故中断。

## Open flags (pending input)

- 每 50 轮换批后，是否接受在训练结束时把各窗口候选 checkpoint 放到同一批实例上统一复测再选最终模型 -> 用户。
- Reference 是每个 epoch 评估并每 50 epoch 换批，还是只每 50 epoch 评估一次 -> 用户；从语境推断为前者，尚待明确。
- 是否希望优先检验学习率假设，还是将 2 万例/200 轮作为一套联合新方案 -> 用户。

## Authorized experiment execution (2026-09-22)

- Experiment 1 completed on 200 fresh instances: E100/E365 sampled training-style POMO, restored Reference/Edge/Anchor/Expert upper-bound pipeline, exact Gurobi on all Target POMO routes. E100 selected-route regret 0.604309; E365 0.482806; E365−E100 paired change −0.121503, 95% bootstrap CI [−0.237174, −0.010105]. Exact best among sampled POMO routes 17.925014 vs 17.923651, effectively unchanged. Thus worsening upper-bound route selection does not explain E365 external test regression on this diagnostic. See `outputs/target_selection_diagnostic_20260922/report.md` and raw JSON.
- Experiment 2 protocol frozen in `outputs/epoch100_controlled_branches/PROTOCOL.md`: from exact E100, E101–E120; branch IL inner epochs 5→2 only; branch LR cosine continuation anchored at actual E100 optimizer LR 9.057123776136868e-5 down to 1e-7 at E150 only. E500 main was paused only after verified E420 checkpoint, with strict resume automated after both branches train. `il2` started in tmux `cvtsp_epoch100_controlled_branches` at 14:10 local, restored E100 policy/optimizer/Reference/Edge/replay(50000)/RNG, and its first three E101 batch metrics exactly matched original E101. Branches evaluate on the same 1000-instance greedy full-POMO exact holdout; original E120 result is already available. Final results pending.
