# LKH＋near-miss20%：从头训练100轮

本实验从随机初始化开始，不加载历史模型、optimizer、Reference、Predictor或replay。

唯一计划内算法/预算变化：固定near-miss 20%；固定C0局部搜索改为LKH；学习率1e-4到1e-5余弦100轮；每轮20000实例；共100轮。其余继承完成的E500逻辑：J20–100、batch64、POMO=J、Reference采样200、Top-K30（质量20＋多样性10，前60/+5%门槛）、普通IL5遍、changed权重5、replay30%/50000、hybrid POMO＋有界精确Expert辅助奖励、固定验证2000、每轮保存checkpoint。质量加权IL不启用。

LKH使用同一Anchor精确解生成的有向浮点C0，以J+1节点ATSP运行。32-worker实际并发下，每例时间预算为max(0.005, 1.5e-4*J^2)秒；该值由两次128例满核预检标定，目标是维持小规模搜索对照中约20个完成起点/例。质量与多样性起点交错尝试，未运行的起点保留原路线。LKH整数目标scale最高1e6，所有候选最终在原始浮点C0上重新计算和验收。搜索期间不调用Gurobi；之后沿用严格上界门槛和20% near-miss精算。

## 日志

- 每轮：训练/固定验证曲线、学习率、Reference更新、RL/IL/Edge loss与准确率、梯度、参数变化、replay、Gurobi调用及分阶段耗时。
- 每batch：J和数据hash、near-miss合格/配额/精算/接受、RL两项优势与权重。
- 每实例：LKH调用/完成/超时/接受输出、搜索耗时、不同终点数、返回Anchor比例、Expert来源与精确改善、near-miss结果。
- 每个观测J每轮最多2例完整复算样本：原实例、Anchor、C0、Top-K、每个LKH输入/输出及状态。
- 每个训练实例：精算Target路线、Target值、Expert路线、Expert值及来源。
- IL五遍分别记录loss、token准确率、optimizer步数和梯度范数。

训练使用物理GPU0（进程内cuda:0）与32个self-improvement workers；Gurobi/LKH/BLAS均为每worker单线程。启动后以实际前几个batch和首轮耗时评估吞吐与稳定性。
