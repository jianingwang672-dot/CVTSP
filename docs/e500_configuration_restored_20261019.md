# 恢复500 epoch版本配置（2026-10-19）

依据已完成训练 `outputs/train__scratch_e500_lr1e4_min1e7_w128_qdhybrid_replay30_vectorized/params.json` 和原启动脚本，恢复主训练入口的默认配置。恢复的是配置与算法分支，不是加载Epoch 500模型，也不是将整个工作区回退到旧代码快照。

| 项目 | 恢复值 |
|---|---|
| 训练预算 | 500 epoch，每轮10000实例 |
| 学习率 | 1e-4 → 1e-7，余弦衰减总长500轮，warmup 1轮 |
| batch / J / POMO | 64 / 20–100 / J |
| worker | 归档运行为128（更大的主机）；本机启动脚本默认32，单worker求解线程1。worker数只影响速度，不影响每个batch的结果 |
| Reference候选 | 200条 |
| Top-K | 30：质量20＋至多10条多样性；候选前60且上界差距≤5% |
| 局部搜索 | 向量化swap＋2-opt，最多30 moves，无递减调度 |
| near-miss | 自适应1%–5% |
| RL | 标准化密集POMO主项＋有界精确Target–Expert辅助项 |
| 密集奖励 | 1条Target精确值，其余严格上界；不启用随机子集精算实验 |
| IL | 固定5遍；改进Expert重复权重5；不启用质量差距加权 |
| replay | 30%，容量50000 |
| 固定验证 | 原`validation/macsim_fixed`，1000例，J20/40/60/80/100各200 |
| 验证方式 | 每例1条free-start greedy，Gurobi精算，无增强，每epoch一次 |
| Reference / best | 原有Reference规模门槛、固定集平均makespan选best |
| 模型与求解器 | 与归档参数一致；不使用cvxpylayer/REBAR |
| 随机种子 | 训练20260813；原固定验证20352150 |
| checkpoint | 每epoch保存；新启动默认随机初始化 |

主入口参数集中于 `src/E500TrainingProfile.py` 的 `E500_RUN_DEFAULTS`。显式实验参数仍可覆盖默认值；`--training-profile e500` 保持原先“固定算法项、保留实验预算”的用途。历史LKH/100轮等专用脚本仍保留以便核对实验；它们不是本次恢复配置的启动入口。它们本轮也撤销了完整POMO默认值。

配置核验（不启动训练）：

```bash
.venv/bin/python -B train_joint_self_improvement.py --print-config
DRY_RUN=1 bash scripts/train_scratch_e500_vectorized_w128_tmux.sh
```

以后启动本配置使用 `scripts/train_scratch_e500_vectorized_w128_tmux.sh <物理GPU> [worker数]`。本机为32线程，脚本默认32个worker、CPU affinity为全部逻辑核（0–31），可分别用第二个参数/`WORKERS`和`CPU_AFFINITY`覆盖；worker数超过逻辑核数时会打印警告。归档运行在更大的主机上用128个worker、绑定16–31。隔离物理GPU后Python使用逻辑cuda:0。默认输出改为`outputs/train__scratch_e500_restored`、tmux为`cvtsp_scratch_e500_restored`，避免覆盖旧实验。空输出目录检查保留。

原params.json保存于一次续训中，包含checkpoint-420及strict_resume；恢复新训练配置时这两项明确清空/关闭，不载入旧模型、优化器、Replay或RNG。输出目录也改为新目录。历史数据与结果保留，外部Real529/人工5000测试协议未修改。

完整POMO验证实现保留为显式候选数0的可选功能；默认1不会进入该分支。按原1000例、两个模型估算，每epoch固定验证共2000次Gurobi精算。本次仅恢复配置和核验，没有启动训练。

已核验主入口及启动脚本各91项归档参数，无差异（上述新运行路径及状态字段除外）。原1000例验证数据指纹一致，文件未改变。35项相关单元测试、3个启动脚本语法检查及diff空白检查通过。核验结果与两份解析后的配置保存在 `outputs/e500_configuration_restore_20261019/`。

## 本机启动默认值调整（2026-10-19）

启动脚本默认worker由128改为32，CPU affinity由16–31改为0–31，以适配本机32个逻辑线程（i9-13900KF：8个性能核含超线程＋16个能效核，内存31 GB）。2026-10-19在本机用当前代码、16个worker从头运行前4个batch，Anchor/Expert/dV/near-miss计数与S500原始日志逐项一致，RL loss仅在小数点后第4位有差异（−6.5577对−6.5578、−4.9282对−4.9279、−4.0962对−4.0973），来自不同硬件/版本的浮点差异；长时间训练不会逐位复现原轨迹。

### worker数基准测试（2026-10-19）

用恢复后的S500配置从头运行前32个batch，30与32个worker交替各测两次，均绑定0–31：

| worker | batch 3–32耗时合计 | 到第32个batch的墙钟 | 内存峰值（主进程＋worker） | swap |
|---:|---:|---:|---:|---|
| 30 | 63.2 秒（两次相同） | 76.9 秒 | 12.8 GB | 无增长 |
| 32 | 61.6 秒（两次相同） | 73.9 秒 | 13.5 GB | 无增长 |

32个worker约快2.5%（30个batch中有22个更快），因此默认改为32。不测64个worker：worker以spawn启动、互不共享内存，单个约0.37 GB起步，64个约需25 GB以上，接近整机内存上限，会落入swap；且CPU只有32个逻辑线程，超额订阅不增加算力。另外观察到两次运行约第10个batch之后数值开始出现微小分歧（GPU浮点非确定性），训练实例序列不受影响。
