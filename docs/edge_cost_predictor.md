# 第三阶段：有向边代价预测与安全候选筛选

这一阶段在前两阶段之上增加“学习排序”，目标是少算没有希望的候选，而不是让神经网络替代 Gurobi 或数学上严格的上界。预测器已经接入 `run_self_improvement.py`：首轮结束后训练，下一轮自动用于 pseudo-expert local search。

## 先区分两张有向价值表

Gurobi 求解某条锚点路线后，会给出每个目标的起飞点 `s[j]` 和降落点 `l[j]`。此时可以精确计算固定解下的有向载车时间：

```text
c[depot,j] = ||depot - s[j]|| / Vc
c[i,j]     = ||l[i] - s[j]|| / Vc
c[i,depot] = ||l[i] - depot|| / Vc
```

所以原顺序即使是 `depot-A-B-C-D-depot`，表里仍然知道 `A -> C`：它直接用 A 的降落点和 C 的起飞点计算距离，不要求这条边在原路线中出现。该表是精确的，但只对当前 Gurobi 固定解成立；接受换边并重新优化位置后必须刷新。

预测器输出的是另一张表。它只读取 depot、目标坐标及实例参数等静态特征，不读取 Gurobi 已优化的 `s/l`。否则边值本来就能直接算出，训练预测器没有意义。由于同一实例在不同锚点解下可能产生不同的 `s/l`，模型学习的是“给定静态实例时，该有向边代价通常有多大”，因此它适合排序，不具备上界证明资格。

## 数据和模型

每个第二阶段实例最多选择四条不重复锚点路线，包括 pseudo expert、teacher best、改进前和改进后的候选。每条路线精确求解一次 Gurobi，并保存：

- 完整的 `(J+1) x (J+1)` 有向时间矩阵；
- 锚点路线、真实目标值和固定飞行时间；
- 数据划分、标签来源和最大可行性残差。

监督标签归一化为：

```text
y[i,j] = c[i,j] * Vc / instance_scale
```

模型复用当前 POMO 策略的 encoder。对有序点对 `(i,j)`，decoder 拼接 `h_i`、`h_j`、`h_j-h_i` 和 `h_i*h_j`，用 MLP 预测非负边值。`(i,j)` 与 `(j,i)` 的输入次序不同，因此保留非对称性。默认冻结 encoder，只训练 edge decoder；训练损失是非对角边上的 Huber loss，最佳 checkpoint 按 validation normalized MAE 选择。

## 安全的 Top-K 搜索流程

```text
当前路线 q
  -> Gurobi 精确求解，得到真实值 V(q) 和精确固定解上界表
  -> 预测器产生静态有向边表
  -> 枚举 swap / relocate / asymmetric 2-opt
  -> 用预测表对全部候选排序
  -> 只对前 K 个候选计算精确 fixed-solution UB
  -> 仅当 exact UB(q') < V(q)-epsilon 时接受最佳候选
  -> Gurobi 重新求解 q'，检查 V(q') <= UB(q') 且 V(q') < V(q)
  -> 刷新精确上界并继续；没有可证明候选时停止
```

预测分数绝不参与接受判定。预测器差时可能把好候选排到 Top-K 外面，导致“漏掉改进”；但 Top-K 内的候选仍经过精确上界和 Gurobi 两道检查，所以不会产生“假改进”。

候选排序本身也使用增量计算。一次 `O(n)` 准备后，swap 和 relocate 只更新常数条边；非对称 2-opt 用反向内部边差的前缀和。因此三类 move 的预测打分均为 `O(1)`，无需为每个候选重新累加整条路线。

## 正式小规模结果

固定随机种子 `20260812`、问题规模 `J=10`：

- 12 个实例、48 个 Gurobi 锚点标签；train/validation/promotion 为 32/8/8；
- validation normalized MAE 从 `0.3624` 降到 `0.0974`；
- promotion normalized MAE 为 `0.1409`，候选 Spearman 为 `0.6476`；
- promotion 中有可证明改进的 2 个锚点，Top-30 均命中其最佳可证明 move；
- 完整搜索与 Top-30 搜索都生成 580 个候选，精确 UB 计算由 580 次降至 120 次，减少 `79.31%`；
- 两个对照实例中，接受 move 数、Gurobi 调用数和最终真实目标值均完全一致。

在 `J=10` 上，预测引导总时间约 `0.062s`，完整枚举约 `0.040s`。也就是说，小规模上神经网络推理和排序开销大于节省的简单数值计算，当前结果证明了“安全减少精确候选评估”，尚不能声称 wall-clock 加速。是否加速需要在 `J=50/100` 上单独做规模实验。

## 运行

```bash
python3 run_edge_predictor_phase.py \
  --pseudo-expert-dataset outputs/self_improvement_phase2_j10_seed20260812/round_001/pseudo_experts.json \
  --policy-checkpoint outputs/self_improvement_phase2_j10_seed20260812/round_001/student/best.pt \
  --output-dir outputs/edge_predictor_phase3_j10_seed20260812 \
  --max-anchors-per-instance 4 \
  --epochs 50 \
  --top-k 30 \
  --max-refreshes 10 \
  --threads 1 \
  --device cpu \
  --seed 20260812
```

主要产物是 `edge_cost_dataset.json`、`predictor/best.pt`、训练指标和 `summary.json`。汇总同时记录边预测误差、候选排序召回率、完整/Top-K 搜索对照以及每次接受 move 的安全检查数据。

`run_edge_predictor_phase.py` 仍用于独立消融实验；正式多轮闭环请使用 `run_self_improvement.py`，其参数和目录结构见 `docs/self_improvement.md`。

## 验证

```bash
python3 -B -m unittest discover -s tests -v
```

单元测试覆盖数据往返、归一化、预测矩阵形状和非负性、方向敏感性、排名搜索只精确评估 Top-K，以及随机非对称矩阵中三类 move 的增量分数与完整路线重算的一致性。
