# E500 算法恢复与 100 epoch 实验准备

恢复来源：`outputs/train__scratch_e500_lr1e4_min1e7_w128_qdhybrid_replay30_vectorized/params.json`、`metrics.csv` 和原 E500 启动脚本。
未找到该次训练的完整代码快照；按现有实现的参数分支恢复算法行为，未对工作区做整体回退。

| 设置 | 本次配置 |
|---|---|
| 预算 | 100 epoch × 20000 instances |
| J / batch / POMO | 20–100 / 64 / J |
| LR | 1e-4→1e-7，余弦衰减总长100，warmup1 epoch |
| Top-K | 固定30，质量20＋多样性10；候选前60且上界不超过质量末位105% |
| 局部搜索 | swap/2-opt≤30 moves，向量化 |
| RL | 标准化密集POMO＋有界精确Expert辅助 |
| IL | 固定5遍；改进Expert权重5 |
| Replay | 30%，容量50000 |
| Near-miss | 随训练能力从1%到5% |
| 固定验证 | 2000例，J20/40/60/80/100各400；单greedy候选 |
| 人工测试 | 500例，每种J100；完整POMO、无增强、Gurobi取最优，同时输出单greedy |
| 保存 | 每epoch；best按固定验证平均makespan |

测试500是新的独立人工集。Real529及旧人工5000原样保留。验证/测试采用不同种子，校验与旧固定集、开发集和旧人工5000的内容重复；完整历史训练数据未保存，无法逐条查重。

`--training-profile e500` 固定算法项，并覆盖同名算法命令行设置；预算、数据目录、种子、worker及设备设置保留。通用入口仍兼容既有命令；本次使用专用脚本。

只读核验配置：

```bash
DRY_RUN=1 bash scripts/train_e100_i20k_e500_logic_tmux.sh
```

以后启动（当前尚未启动）：

```bash
bash scripts/train_e100_i20k_e500_logic_tmux.sh <物理GPU> <workers>
```

tmux：`cvtsp_e100_i20k_e500logic`。随机初始化，不加载历史checkpoint；旧checkpoint、实验与工作区已有改动保留。

现有日志整理：`outputs/e500_log_review_20261010/report.md`。
缺失观察：梯度范数、swap/2-opt分别的接受次数与上界贡献、搜索链步数分布及30步上限命中率；需新运行采集。
