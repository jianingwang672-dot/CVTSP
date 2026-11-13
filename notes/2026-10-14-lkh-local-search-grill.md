# CVTSP接入LKH局部搜索：Grill / Discovery Notes
Date: 2026-10-14 · Goal: 确认LKH能否接入固定严格上界表搜索，以及下一步实验范围。

## Summary / key decisions
- 用户询问“我的local search部分可以直接借用LKH的算法吗”，随后授权小规模搜索对照；未授权替换当前训练算法。
- 当前B组20轮训练保持现有swap+2-opt，不因本次讨论中断。
- 用户已选择“先小规模搜索对照（推荐）”：授权准备并运行同实例、同Anchor/C0、同Top-K起点的LKH搜索对照，不授权替换训练算法。
- 已读代码确认搜索目标为固定Anchor的有向严格上界C0之和，路线为depot→全部J个目标→depot；可建模为J+1节点ATSP环。
- 可调用现有LKH实现搜索该矩阵，不需要改Gurobi下层；输入转换与返回路线校验需要适配。
- LKH优化固定上界，不是直接优化原CVTSP连续位置/时间makespan；结果仍应使用原浮点C0校验并走原精算/near-miss门槛。

## Q&A log

### 用户本次问题 — 直接借用LKH
- Asked: “我的local search部分可以直接借用LKH的算法吗”
- Captured: 用户在考虑会议提出的LKH局部搜索方向，目前询问可行性。
- Flags: 希望先做同实例离线搜索对照，还是下一轮从头训练替换 -> 用户。

### Q1 — 接入范围（已回答）
- Asked: 先把LKH作为同一C0和Top-K起点的小规模搜索对照，还是下一轮直接从头训练LKH对照？
- Recommendation: 先做搜索对照，比较耗时、上界下降、精算后改进Expert数量和真实改善，再决定是否用于训练。
- Captured: 用户原话：“先小规模搜索对照（推荐）”。先比较搜索质量与成本，当前B组保持原算法运行。
- Flags: 对照预算、版本与运行参数在确定实验范围后再决定。

## Evidence / verification
- src/JointSelfImprovement.py: directed_route_step_costs明确从depot出发并返回depot；矩阵包括depot与目标，有向。
- src/JointSelfImprovement.py: local search在固定majorizer上执行；Top-K质量20/多样性10与最后严格上界精算门槛可保留。
- LKH作者原文/代码的源码镜像：https://github.com/cerebis/LKH3；官方大学网页本次直接访问超时，未据此猜测版本或参数默认值。
- 官方索引可读的作者报告：https://webhotel4.ruc.dk/~keld/research/CLKH/CLKH_Report.pdf，说明支持非对称问题及EXPLICIT/FULL_MATRIX输入。
- 已读作者原始ReadProblem.c源码镜像，确认ATSP、EXPLICIT/FULL_MATRIX：https://raw.githubusercontent.com/cerebis/LKH3/master/SRC/ReadProblem.c。
- 该实现可以读取浮点边权，但内部执行round(Scale*w)并存入整数；须控制量化误差和整数范围。接口保留方向、depot与节点映射；输出路线需在原始浮点C0上重新核算，严格性不依赖量化目标。
- 已读ReadParameters.c的作者参数说明：https://raw.githubusercontent.com/cerebis/LKH3/master/SRC/ReadParameters.c，支持INITIAL_TOUR_FILE、MAX_TRIALS、TIME_LIMIT；trials不同于当前接受move数。
- LKH可探索组合k-opt和扰动，可能改善固定矩阵下的路线；是否提高真实makespan、改进Expert比例和学习效果仍需实验。若瓶颈在固定上界松紧、候选精算门槛或IL标签，替换搜索不能保证解决。
- LKH的trials/k-opt与现有30个接受move含义不同，不能把MAX_TRIALS=30当作等量搜索预算；同墙钟或记录实际成本更合适。

## Open flags (pending input)
- 小规模同输入搜索对照 -> 已完成100例，独立复核输入、原始浮点路线成本、汇总均值与精算配额通过。
- 是否从头训练LKH对照 -> 尚未授权，搜索结果不能证明训练效果。

## 2026-10-14 小规模对照结果
- 固定nm20 Epoch20生成候选；J=20/40/60/80/100各20例，共100例。CPU双worker，当前B组训练持续运行。
- 同Anchor、C0、Top-K；near-miss名义20%，每种J最多4例。搜索中不调用Gurobi。
- swap+2-opt：Expert平均18.4658655643，14/100真改善，搜索平均0.244335781秒/例。
- LKH全起点：Expert平均18.3570288391，12/100真改善，0.301189957秒/例，平均30起点完成。
- LKH近似同时间上限：Expert平均18.3263765272，18/100真改善，0.157781724秒/例，平均19.54起点完成；有29次起点超时，不是硬实时等耗时。
- 配对Expert差值LKH全起点−baseline=-0.108837，95%区间[-0.1988363471,-0.0335976678]；近似同时间上限=-0.139489，区间[-0.2280539921,-0.0608610440]。单批冻结策略搜索证据，不是泛化或训练提升结论。
- 全起点LKH平均只留下1.22条不同终点路线，near-miss只精算1例；基线留下24.43条。收敛到相似终点会影响非Anchor候选覆盖，需单独验证，未改变本次实验规则。
- 完整结果：outputs/lkh_search_pilot_20261014/report.md、comparison.json、completion_audit.json。

### 用户澄清 — “local search 50次是指move数吗”
- max_local_moves=50指每条起点路线最多接受50个改善move，提前无改善可停止；每个move会检查多个候选。
- LKH MAX_TRIALS=50指50轮trial，每轮可能多个move，不能等同。
- 当前B组仍为30个move，本次LKH每条起点为1 trial；用户此问题不构成改为50的授权。
