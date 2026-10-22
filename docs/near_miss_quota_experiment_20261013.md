# Near-miss配额：三组从头训练

用户选择直接三组从头训练20轮，不做旧模型离线筛查。本轮比较标称固定精算配额5%、10%、20%，不从旧模型恢复任何状态。

| 项目 | 公共设置 |
|---|---|
| 本阶段/调度总计划 | 20轮 / 100轮；结束后不自动继续 |
| 每轮训练实例、batch | 20000、64 |
| J/POMO、Reference候选 | J=20–100、POMO=J、200条 |
| 算子与move | swap＋2-opt，最多30次接受move，全部20轮固定30 |
| Top-K | 30：质量20＋质量受限多样性10；候选前60、5%质量门槛 |
| RL/IL/replay | 保留E500算法逻辑：密集POMO＋有界精确Expert辅助；IL5遍、改进Expert权重5；replay30%/50000 |
| 学习率 | 1e-4→1e-5，余弦总长100轮、warmup1 |
| 固定验证 | 2000，J20/40/60/80/100各400，原单greedy精算 |
| 独立诊断 | 复用固定100例；完整POMO诊断固定50例，无增强 |
| 初始化/训练数据 | 同seed20260813、相同全新Target/Reference/Predictor，空replay，实例RNG隔离并逐batch校验 |
| 资源 | RTX4090物理GPU0→cuda:0，每组8worker，合计24；大显存阶段最多两组并发 |

## 唯一组间算法变量

- nm05：near-miss minimum=maximum=0.05。
- nm10：near-miss minimum=maximum=0.10。
- nm20：near-miss minimum=maximum=0.20。

相等的上下限使现有自适应公式恒等于指定配额；精确Expert辅助权重仍按原训练信号公式调整，不因组别更换公式。

沿用现有每batch规则：按原始上界相对Anchor的gap排序，稳定处理同分，最多选择ceil(rate×batch_size)个符合条件的实例，每个只精算一个不同于Anchor的候选。候选不足则少选。标准64例batch的上限分别为4、7、13，即6.25%、10.9375%、20.3125%。**标称配额不等于实际精算比例**，结果同时报告标称值、eligible/attempted/accepted和真实调用成本。

所有额外候选都通过相同Gurobi参数精算、最优状态与可行性检查；仅严格优于Anchor才替换Expert。搜索中间不调用Gurobi，每实例Anchor/Expert阶段仍最多两次精算，Target奖励精算另记。

本轮固定move30，按用户最新意见保留预算。上一次算子实验按100轮从30降至10（Epoch20=26）；因此不能把其合用组当作本轮nm05控制组。三组全部重训，组间仅配额不同。

## 数据与诊断

每轮保存checkpoint及optimizer/Reference/Predictor/replay/RNG完整状态。逐轮记录Expert真实改善、改进率、Target胜Expert率、IL准确率、RL/IL loss与梯度、实际near-miss调用率、接受率、候选gap、训练与诊断耗时。

Epoch1/5/10/20记录固定100例RL前/后、IL后单greedy精算；Epoch5/20记录固定50例RL后/IL后完整POMO精算；Epoch1/5/10/20记录训练采样POMO全精算排序诊断。诊断独立、不生成训练标签、不回灌replay、不消耗训练RNG。

正式测试500、Real529不参与本轮调参。一个seed、前20轮是初筛，不能证明100轮终点最优；同时比较Epoch20与验证best，不能仅根据Expert比例选择获胜配额。

## 输出与完成条件

输出根目录：outputs/near_miss_quota_20261013；组目录nm05、nm10、nm20。
会话：cvtsp_nearmiss_nm05_e20、cvtsp_nearmiss_nm10_e20、cvtsp_nearmiss_nm20_e20；后台监测cvtsp_nearmiss_observer_e20。

全部20轮退出0、所有checkpoint/诊断完整、初始化及全部训练数据一致、固定配额和move30实际生效、状态恢复字段及源码校验通过后，生成completion_audit.json、comparison.json、comparison_epochs.csv、report.md及逐规模曲线，再统一报告。普通进度不推送。
