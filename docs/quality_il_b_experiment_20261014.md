# IL标签质量加权：只运行B组

用户选择仅运行B组，从头训练20轮，near-miss固定20%，IL5遍。
旧nm20训练仅作历史参考；单个seed、20轮不能确定泛化获益或因果归因。

共同配置继承 outputs/near_miss_quota_20261013/nm20/resolved_config.json：
seed20260813，随机J20–100，POMO=J，20000实例/轮，batch64，
Reference候选200，Top-K30（质量20、多样性10，前60和5%质量门槛），
swap+2-opt/固定30步；学习率1e-4→1e-5，余弦100轮/warmup1；
IL5遍，changed重复5次，IL逻辑batch1000/显存microbatch128；
replay30%/容量50000；固定验证2000，独立100例与完整POMO50例诊断不变。
RL及精确辅助项权重公式不变。执行20轮后停止，不自动续到100。

## B组标签质量

在每轮RL完成之后、IL开始前，用此时的Target快照重新测量当前与抽到的历史标签。
使用相同训练采样模式（sample、POMO=J），按Anchor/Expert严格上界中完整路线的较小值选一条路线，
再Gurobi精算。仅比较这条被精算Target路线，不声称比较了Target所有路线中的真实最优。
精确Expert值及其严格上界矩阵来自原合法求解，缓存复用；不以预测值代替精确值。
测量使用独立seed20281014+epoch、模型副本，并完整恢复训练RNG与模型环境缓存。

g=(VT-VE)/max(abs(VT),1e-12)，q=clip(0.5+g/0.02,0.1,2)。
这些常数是提前设定的待检验方案，不是已经确认的最优阈值。
沿用原changed重复5次与replay30%取样，只在CE上增加质量系数q。
每个IL逻辑batch采用 sum(q_i*token_NLL_i)/sum(q_i*J)，
microbatch以其权重质量占完整逻辑batch的比例累积梯度，不能各microbatch独立等权平均。
所有标签保留正权重，5遍共用IL入口测得的固定权重，不增加重复次数或更新次数来实现质量权重。
重复标签去重测量；每个不同(instance,expert)标签新增一次Target精算，额外调用与时间单列。
fresh replay为空；原replay格式不改，B的缓存单独存checkpoint并随FIFO清理，保证可严格恢复。

## 核验与结果

语法、旧行为回归、权重阈值、权重归一化与显存microbatch梯度等价测试；
启动前进行J100/batch64小规模真实CUDA/Gurobi运行，不使用它的checkpoint启动正式训练。
正式输出 outputs/quality_il_b_20261014/b，tmux cvtsp_quality_il_b_e20，GPU0→cuda:0，8workers。
记录每轮标签source/精确VT/VE/g/q/重复次数，以及按CE逻辑batch归一化后的标签贡献份额。
记录额外精算、测量时间、RL/IL阶段质量、奖励排序与梯度诊断。
20轮完整退出并审计checkpoint、实际配额、label质量和数据流后生成报告，对比历史nm20。
正式test500/Real529不用于本轮参数选择。仅完成或异常时汇报。
