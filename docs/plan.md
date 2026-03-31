# CVTSP `src/` DRL + Gurobi MVP Plan

## Repository Understanding
- 仓库当前最重要的现有资产是顶层固定顺序 CVP 求解器 `CVTSP_SOCP.py`、实例解析/验证脚本 `batch_verify_gam72.py`、`instance/Gam72` 样例集，以及 `gurobi/POMO_Gurobi_Jianing/TSP/POMO` 下的 POMO/CETSP 风格 DRL 骨架。
- `instance/Gam72` 有 72 个文本实例，`J` 覆盖 10 到 15；每个实例由 `J`、`MT`、`Vd`、`Vv` 和 `p: position coordinates =` 组成。当前解析逻辑把 `Vd` 映射为 UAV speed，把 `Vv` 映射为 carrier speed。
- 现有 DRL 代码的本质是 “masked permutation policy + episode-end exact reward”：它的 decoder 生成离散访问顺序，环境末端再调用 Gurobi SOCP 返回真实 objective。可复用的是 encoder/decoder、mask、POMO、多轨 REINFORCE；不可直接复用的是 `radius` 特征、随机数据生成和环境内求解器耦合。
- 现有 `CVTSP_SOCP.py` 已经实现固定访问顺序下的 CVP：输入 `depot, targets, tour, Vc, Vv, endurance_a`，返回 objective 和连续解 `sx/lx/t1/t2/tau/Tseg`。

## Proposal Understanding
- 当前 proposal 的完整愿景是分层求解 carrier-vehicle TSP：上层决定 target visiting sequence，下层在 sequence 固定后决定 take-off / landing 等连续变量。
- 当前阶段只实现上层 sequence DRL；DRL 不输出连续几何变量；下层暂不做 learned policy，而是用 exact Gurobi SP 作为 oracle。
- 这意味着当前的 MVP 目标是跑通 `instance -> master policy -> legal permutation -> exact SP -> objective/reward`，而不是复现 LBBD、Benders cuts 或 proposal 终态里的双层学习框架。

## CETSP / UD3RL Reuse Analysis
- 直接复用的部分：Transformer-style encoder、masked decoding、POMO 多起点、REINFORCE 训练、sample-train / greedy-test 范式。
- 必须裁剪的部分：CETSP `radius` 特征、随机问题生成、把 exact solver 写进环境类的耦合方式。
- 如果参考 UD3RL 论文，其最值得保留的是 node selection policy，而不是 waypoint / loc-decoder 相关逻辑；当前项目的连续变量统一交给 exact SP。

## DRL Master Design
- **State / Observation**：`depot + targets` 节点图，节点特征固定为 `[x_rel, y_rel, dist_rel, is_depot, Vc_over_Vv, Vv_times_a_over_scale]`。坐标先相对 depot 平移，再按实例最大目标距离归一化。
- **Action**：从未访问 target 中选择下一个 target。内部 decoder 保留 `depot + targets` 索引以复用 POMO；最终对外输出时移除 depot，并把 target 索引映射为 `0..J-1` 的 permutation。
- **Termination**：所有 targets 被访问一次后结束。内部 rollout 步数等于 `J + 1`，其中第一步固定为 depot。
- **Reward**：仅在 episode 结束后调用 exact SP，`reward = -objective_from_sp`。如果 solver 失败，则使用固定惩罚 `-1e6`。
- **Batching**：一个共享模型覆盖 `J=10..15`，但 batch 只混合同 `J` 的实例，避免 padding/mask 复杂度并保证 POMO second-step 逻辑简洁。

## DRL + SP Interface
- `load_instance(path) -> CVTSPInstance`
- `load_dataset(dataset_dir, split) -> list[CVTSPInstance]`
- `solve(instance, sequence, config) -> SPSolution`
- `generate_sequences(model, instance, device, decode_type, num_candidates) -> list[list[int]]`
- `evaluate_instance(instance, config, model|sequences|baseline) -> EvaluationResult`

## MVP Scope
- 必做：
  - `src/data`：实例解析、split、同尺寸 batch。
  - `src/master`：masked permutation policy，支持 sample 和 greedy。
  - `src/subproblem`：桥接现有 exact CVP solver，并统一返回结构化结果。
  - `src/pipeline`：训练、推理、评估、baseline、reward cache。
  - `src/cli`：`train / infer / evaluate` 三个入口。
  - `docs/run.md`：依赖、命令、输出说明。
- 暂不做：
  - LBBD/MP/cuts/Benders。
  - lower-level learned policy。
  - 仓库外 `refs` 依赖。
  - 大规模并行 solver 调度或 surrogate reward。

## Follow-up Optimizations
- 对 exact SP 加入更细的 cache 持久化。
- 对推理增加 beam / rerank。
- 如果后续要扩展到 proposal 的完整形态，可在当前 `src/subproblem` 接口后面替换成 learned lower policy，而不改上层训练/推理调用链。

