# LKH同输入搜索对照：小规模100例

用户确认先做小规模搜索对照，不替换或中断当前B组训练。

- 新生成100例：J20/40/60/80/100各20，seed20261014；与已知固定验证2000、test500、独立诊断100核验不重叠。
- 使用已完成nm20 Epoch20 checkpoint内的Reference和Edge Predictor生成共同候选；这是冻结策略的搜索诊断，不是从头训练效果对照。
- 相同200个候选、相同预测Anchor、共同一次Gurobi精算和原精度C0；Top30质量20＋多样性10、前60与5%门槛不变。
- 原算法：结果等价向量化swap＋2-opt，最多30个接受move，原epsilon。每个worker/J预热索引缓存，不把一次性缓存构造算入常规搜索耗时。
- LKH来源：隔离下载的作者原始LKH-3.0.6源码镜像，保留许可证及源码archive SHA256；不是声称最新版本。项目third_party/lkh3_20261014编译，未安装到系统。
- LKH输入TYPE=ATSP、EXPLICIT/FULL_MATRIX，包括depot；原始非对称矩阵不对称化。整数scale最高1e6并根据32位边权安全范围下调，PRECISION=1；验收只使用原始浮点C0，量化目标不作为严格上界。
- LKH原始参数：RUNS1/MAX_TRIALS1/MOVE_TYPE5/MAX_CANDIDATES5，固定seed、原Top-K起点INITIAL_TOUR_FILE，native TIME_LIMIT2秒与外部5秒兜底。

## 两种LKH预算（同一算法的两个诊断口径）

1. lkh_matched_budget：主要口径，以本例原30起点搜索实测墙钟作为预算，含LKH矩阵导出、子进程启动、解析成本。固定交错质量/多样性起点；同一Top30全部可作fallback，但时间不够时只运行部分起点。短deadline可能被启动/kill开销超过，报告实测耗时不截断。
2. lkh_all_starts：每个共同起点一次LK trial，测量完整30起点下的质量与实际成本；不把30trials说成30moves或等计算量。

每个profile沿用严格上界门槛；否则选择不同于Anchor的最低gap候选，20%near-miss配额按同J的20例分组，最多4次。每例每profile最多一个额外精算；Anchor精算三组共享。记录重复最终路线数，特别观察是否收敛回Anchor使near-miss候选消失。

## 核验与报告

已测试ATSP方向/节点映射、量化误差范围、实际LKH定向环实例；五个规模各一例的端到端运行通过，属于工程检查，不作为正式100例结论。
正式记录每例原始输入、C0/Top-K hash、各起点输入/输出路线、浮点上界、原接受move数、LKH完成/超时起点、耗时与精算后的Expert。
报告逐J/总体Expert makespan、改进率/幅度、certified/near-miss接受情况、重复路线与搜索成本，以及配对胜平负和95%区间。
仅使用CPU2workers、torch1线程、CUDA_VISIBLE_DEVICES为空，保持B组GPU训练；输出outputs/lkh_search_pilot_20261014，tmux cvtsp_lkh_search_pilot。
这一小规模实验不能证明LKH训练效果、cold-start收敛或其他实现/预算的优劣。
