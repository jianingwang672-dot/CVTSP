# 固定验证评测修正（2026-10-19）

> 后续决定：用户要求全部配置恢复为历史500 epoch版本。完整POMO不再是默认；当前默认已恢复为原1000例、每例1条free-start greedy精算。本页下文保留为先前改动记录，当前生效配置见 `e500_configuration_restored_20261019.md`。只有显式设置候选数0才启用完整POMO。

新训练默认以完整POMO评价Target与Reference：每例greedy生成J条POMO路线，不加入额外free-start路线、不做几何增强；对每条不同路线调用原Gurobi配置精算，先取每例最小makespan，再对实例求平均。Reference晋升和best.pt共用这一评测结果，原J80/J100不退步的Reference门槛保留。

现有2000例验证实例不更换，仍为J20/40/60/80/100各400例。验证频率保留每epoch，末轮也必评。完整POMO选模的1000例仍是另一个开发集，两套实例的绝对均值不能混比。Real529仍有8倍增强，不能说本验证与Real529完全同协议。

## 配置

- `--reference-validation-candidates 0`：新默认，表示每例完整POMO=J；不是0条候选。
- 正整数：保留旧候选生成方式（free-start优先并按数量截断）。历史单路线为`1`。
- E500 profile不再覆盖验证候选设置；其余训练算法参数不受本次修改影响。
- LKH100启动脚本默认0，`VALIDATION_CANDIDATES=1`可显式重现历史评测。
- 历史random-subset配对实验脚本显式固定1，以继续匹配已保存的A组设置。

精算复用已有self-improvement进程池，每个求解仍使用同一Gurobi配置。路线生成按J分组、小批量执行，避免对完整验证集同时保留GPU计算图。保存协议名称、无增强标记、J条起点、候选生成方式和精确取最小的规则，防止把旧单路线分数当成完整POMO分数。

## Checkpoint与生效时点

checkpoint和Reference快照新增验证协议元数据；旧checkpoint可用原候选数恢复。无论是否strict resume，协议不同都拒绝沿用旧best分数和Reference评测历史；从旧单路线切到完整POMO的训练应明确新开实验，不能将两种分数接成同一条曲线。

本次没有启动新训练或重启历史训练。当前未发现存活的tmux服务器，磁盘上的LKH历史日志停在E95阶段，不能声称代码修改已在该历史进程生效。修改将在下一次启动时加载。

## 验证要求

单元测试覆盖：完整J起点且不注入free-start、包括最后一条候选、按实例取最小后平均、去重和worker路径、低POMO上限配置的临时覆盖与恢复、失败后恢复模型状态、训练RNG不变、旧评测兼容和跨协议checkpoint拒绝。

已通过35项相关单元测试。另使用E46在J20/40/60/80/100各1例，4个worker执行600次真实Gurobi精算；5例的新结果与既有完整POMO选模结果逐例完全一致（最大绝对误差0），模型模式、配置及CPU RNG也保持原状态。结果保存于`outputs/full_pomo_validation_check_20261019/result.json`。这只是验证实现正确性，不是新的泛化评测或完整2000例实验。
