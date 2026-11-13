"""Build an evidence-backed Chinese experiment review without changing training."""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import os
import re
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STEM = "CVTSP_实验完整复盘_500epoch至LKH"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cvtsp_review_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCES: dict[str, dict] = {}
CACHE: dict[str, bytes] = {}
SNAPSHOT = datetime.now().astimezone().isoformat(timespec="seconds")


def read(path):
    path = Path(path)
    p = path if path.is_absolute() else ROOT / path
    key = str(p)
    if key not in CACHE:
        raw = p.read_bytes()
        CACHE[key] = raw
        SOURCES[key] = {
            "path": key, "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "read_snapshot": SNAPSHOT,
        }
    return CACHE[key].decode("utf-8")


def js(path):
    return json.loads(read(path))


def rows(path):
    return list(csv.DictReader(io.StringIO(read(path))))


def n(value, digits=6):
    if value is None:
        return "待补充"
    return f"{float(value):.{digits}f}"


def pct(value, digits=3):
    return n(100 * float(value), digits) + "%"


def ci(v):
    return f"[{float(v[0]):+.6f}, {float(v[1]):+.6f}]"


def table(headers, values):
    def esc(x):
        return str(x).replace("|", "／").replace("\n", " ")
    return "\n".join([
        "| " + " | ".join(map(esc, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(esc, row)) + " |" for row in values),
    ]) + "\n"


def write_csv(name, data):
    if not data:
        return
    fields = list(dict.fromkeys(k for row in data for k in row))
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(data)


OLD = "outputs/train__scratch_e500_lr1e4_min1e7_w128_qdhybrid_replay30_vectorized"
R200 = "outputs/train__scratch_e200_i20k_lr1e4_il521_replay10_topk30to50"
LKH = "outputs/lkh_nm20_e100_20261014/train"
NM = "outputs/near_miss_quota_20261013"
ABL = "outputs/search_ablation_20261012"
QIL = "outputs/quality_il_b_20261014/b"

runs = {}
for label, base in [("S500", OLD), ("R200", R200), ("LKH100", LKH), ("QIL-B", QIL)]:
    rr = rows(f"{base}/metrics.csv")
    runs[label] = {
        "base": base, "params": js(f"{base}/params.json"), "rows": rr,
        "by_epoch": {int(float(r["epoch"])): r for r in rr},
        "latest": rr[-1],
        "best": min(rr, key=lambda r: float(r["target_mean_objective"])),
    }

old_review = js("outputs/e500_log_review_20261010/summary.json")
old_selection = js("outputs/pomo_selection_20260921/summary.json")
lkh_selection = js("outputs/lkh_nm20_checkpoint_selection_20261015/refined_selection.json")
r200_selection = js("outputs/current_e200_pomo_selection_20260930/final_selection.json")
ablation = js(f"{ABL}/comparison.json")
near = js(f"{NM}/comparison.json")
pilot = js("outputs/lkh_search_pilot_20261014/comparison.json")
branches = js("outputs/epoch100_controlled_branches/summary.json")
sample_results = js("outputs/eval_sample_e200_vs_e365/summary.json")
regret = js("outputs/target_selection_diagnostic_20260922/paired_effects.json")
nm_audit = js(f"{NM}/completion_audit.json")
ab_audit = js(f"{ABL}/completion_audit.json")
pilot_audit = js("outputs/lkh_search_pilot_20261014/completion_audit.json")

external = []


def add_external(run, epoch, dataset, part, size=None):
    if "model_mean" in part:
        baseline = part["baseline_mean"]
        model = part["model_mean"]
        count = part["n"]
        win, tie, loss = (part[k] for k in ["wins", "ties", "losses"])
    else:
        label = f"epoch{epoch}"
        baseline = part["epoch200_mean_makespan"]
        model = part[label]["mean_makespan"]
        count = part["num_instances"]
        win, tie, loss = (part[label][k] for k in ["wins", "ties", "losses"])
    external.append({
        "run": run, "epoch": epoch, "dataset": dataset, "J": size or "all",
        "n": count, "baseline_mean": baseline, "model_mean": model,
        "model_minus_baseline": model - baseline,
        "gap_percent": 100 * (model / baseline - 1),
        "wins": win, "ties": tie, "losses": loss,
    })


for ep, dataset in [
    (4, "synthetic5000"), (12, "real529"), (16, "synthetic5000"),
    (100, "real529"), (100, "synthetic5000"),
    (111, "real529"), (111, "synthetic5000"),
    (127, "real529"), (127, "synthetic5000"),
    (178, "real529"), (178, "synthetic5000"),
    (296, "real529"), (296, "synthetic5000"),
]:
    d = js(f"outputs/eval_vectorized_epoch{ep}_vs_gurobi_epoch200_{dataset}/summary.json")
    add_external("S500", ep, dataset, d.get("all_529", d.get("overall")))
    if "training_range_J20_100" in d:
        add_external("S500", ep, "real529_J20_100", d["training_range_J20_100"])
    for size, part in d["by_size"].items():
        add_external("S500", ep, dataset, part, int(size))

d365 = js("outputs/eval_vectorized_epoch365_external/summary.json")
for dataset in ["real529", "real529_J20_100", "synthetic5000"]:
    d = d365[dataset]
    external.append({
        "run": "S500", "epoch": 365, "dataset": dataset, "J": "all",
        "n": d["instances"], "baseline_mean": d["gurobi_epoch200_mean"],
        "model_mean": d["epoch365_mean"],
        "model_minus_baseline": d["epoch365_mean"] - d["gurobi_epoch200_mean"],
        "gap_percent": 100 * (d["epoch365_mean"] / d["gurobi_epoch200_mean"] - 1),
        "wins": "见逐例结果", "ties": "见逐例结果", "losses": "见逐例结果",
    })
    for size, part in d["by_size"].items():
        lookup_dataset='real529' if dataset=='real529_J20_100' else dataset
        known = next(r for r in external if r['run']=='S500' and r['epoch']==100
                     and r['dataset']==lookup_dataset and r['J']==int(size))
        external.append({
            "run": "S500", "epoch": 365, "dataset": dataset, "J": int(size),
            "n": part.get("instances", known['n']),
            "baseline_mean": known['baseline_mean'],
            "model_mean": part["epoch365_mean"],
            "model_minus_baseline":part["epoch365_mean"]-known['baseline_mean'],
            "gap_percent":100*(part["epoch365_mean"]/known['baseline_mean']-1),
        })

formal176 = js("outputs/eval_formal_scratch_epoch176_external/summary.json")
for dataset in ["real529", "synthetic5000"]:
    add_external("R200", 176, dataset, formal176[dataset])
    for size, part in formal176[dataset + "_by_size"].items():
        add_external("R200", 176, dataset, part, int(size))
formal11 = js("outputs/eval_formal_scratch_epoch11_external/real529/results.json")
add_external("R200", 11, "real529", formal11["overall"])
formal11_status = js("outputs/eval_formal_scratch_epoch11_external/status.json")

lkh_external = {}
for ep in [46, 59]:
    d = js(f"outputs/eval_lkh_nm20_epoch{ep}_external_20261015/summary.json")
    lkh_external[ep] = d
    for dataset in ["real529", "synthetic5000"]:
        add_external("LKH100", ep, dataset, d[dataset])
        for size, part in d[dataset + "_by_size"].items():
            add_external("LKH100", ep, dataset, part, int(size))
    sub = d["real529_J20_100_training_sizes"]
    base = sub["epoch200_mean_makespan"]
    value = sub[f"epoch{ep}_mean_makespan"]
    external.append({"run": "LKH100", "epoch": ep, "dataset": "real529_J20_100", "J": "all",
                     "n": sub["num_instances"], "baseline_mean": base, "model_mean": value,
                     "model_minus_baseline": value-base, "gap_percent": 100*(value/base-1)})


def ext(run, ep, dataset, size="all"):
    return next((r for r in external if r["run"] == run and r["epoch"] == ep
                 and r["dataset"] == dataset and r["J"] == size), None)


def score(run, ep, dataset, size="all"):
    r = ext(run, ep, dataset, size)
    return None if r is None else r["model_mean"]


def paired_stage(base, ep, mode):
    a = js(f"{base}/diagnostics/epoch-{ep:03d}_after_rl_{mode}.json")
    b = js(f"{base}/diagnostics/epoch-{ep:03d}_after_il_{mode}.json")
    assert [x["fingerprint"] for x in a["records"]] == [x["fingerprint"] for x in b["records"]]
    diff = np.asarray([y["makespan"]-x["makespan"] for x,y in zip(a["records"],b["records"])])
    rng = np.random.default_rng(20261015)
    samples = rng.choice(diff, size=(10000,len(diff)), replace=True).mean(axis=1)
    return {"epoch": ep, "mode": mode, "n": len(diff), "after_rl": a["mean"], "after_il": b["mean"],
            "il_minus_rl": float(diff.mean()), "ci95": np.quantile(samples,[.025,.975]).tolist()}


lkh_stages = [paired_stage(LKH, ep, mode) for ep in [5,20] for mode in ["free_greedy","full_pomo_greedy"]]
quality_stages = [paired_stage(QIL, 5, mode) for mode in ["free_greedy","full_pomo_greedy"]]
step_rows = []
for run, epoch in [("S500",50),("S500",100),("S500",200),("LKH100",25),("LKH100",46),("LKH100",50),("LKH100",59)]:
    state = runs[run]
    if epoch not in state["by_epoch"]:
        continue
    rr = state["by_epoch"][epoch]
    rl = sum(int(float(r["batches"])) for r in state["rows"] if int(float(r["epoch"])) <= epoch)
    total = int(float(rr["global_step"]))
    step_rows.append({"run":run,"epoch":epoch,"instances":epoch*int(state["params"]["trainer_params"]["train_episodes"]),
                      "rl_steps":rl,"il_steps":total-rl,"total_steps":total,"learning_rate":float(rr["learning_rate"])})

init = js(f"{LKH}/initialization.json")
lkh50 = js(f"{LKH}/diagnostics/epoch-050_lkh_summary.json")
learning_stats=[]
for label,base,ep in [('LS合用',f'{ABL}/both',20),('NM20',f'{NM}/nm20',20),
                      ('QIL-B',QIL,6),('LKH',LKH,20),('LKH',LKH,50)]:
    d=js(f'{base}/diagnostics/epoch-{ep:03d}_summary.json')
    record={'group':label,'epoch':ep,'rl_parameter_delta':d['rl_parameter_delta_l2'],
            'il_parameter_delta':d['il_parameter_delta_l2']}
    for phase in ['rl','il']:
        gg=[g for g in d['gradients'] if g['phase']==phase]
        record[phase+'_steps']=len(gg)
        record[phase+'_preclip_mean']=statistics.mean(g['before'] for g in gg)
        record[phase+'_postclip_mean']=statistics.mean(g['after'] for g in gg)
        record[phase+'_clip_fraction']=sum(g['before']>1.000001 for g in gg)/len(gg)
    full_grad=next((r for r in d['rl_components'] if 'hybrid_gradient_norm' in r),None)
    record['rl_component_gradient_sample']=full_grad
    learning_stats.append(record)
lkh50_full=js(f'{LKH}/diagnostics/epoch-050_summary.json')
qrows = runs["QIL-B"]["rows"]
nmrows = rows(f"{NM}/nm20/metrics.csv")

def report_section(path, begin, end=None):
    text = read(path)
    part = text.split(begin,1)[1]
    if end and end in part:
        part = part.split(end,1)[0]
    return part.strip()


def select_epoch_mean(data, ep):
    d = data["by_epoch"][str(ep)]
    if "mean_makespan" in d:
        return float(d["mean_makespan"])
    if "overall_mean" in d:
        return float(d["overall_mean"])
    raise KeyError(list(d))


# Export tabular evidence alongside the readable report.
write_csv("external_results_all_sizes.csv", external)
write_csv("optimizer_steps_equal_samples.csv", step_rows)
write_csv("S500_metrics_snapshot.csv", runs["S500"]["rows"])
write_csv("R200_metrics_snapshot.csv", runs["R200"]["rows"])
write_csv("LKH100_metrics_snapshot.csv", runs["LKH100"]["rows"])
selection_rows=[]
for ep in old_selection["evaluated_epochs"]:
    selection_rows.append({"run":"S500","epoch":ep,"mean_makespan":select_epoch_mean(old_selection,ep)})
for r in lkh_selection["ranking"]:
    selection_rows.append({"run":"LKH100","epoch":r["epoch"],"mean_makespan":r["mean_makespan"],
                           "difference_from_best":r["difference_from_best"],"ci95_low":r["difference_from_best_ci95"][0],
                           "ci95_high":r["difference_from_best_ci95"][1]})
for r in r200_selection["ranking"]:
    selection_rows.append({"run":"R200","epoch":r["epoch"],"mean_makespan":r["mean_makespan"]})
write_csv("checkpoint_selection_all_runs.csv",selection_rows)


# Scientific plots use separate axes for different data/protocols.
fig, ax=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
x=[int(float(r["epoch"])) for r in runs["S500"]["rows"]]
y=[float(r["target_mean_objective"]) for r in runs["S500"]["rows"]]
ax[0,0].plot(x,y,lw=1.2,color="#2563eb")
ax[0,0].scatter([100,365,500],[y[99],y[364],y[499]],color="#c2410c",s=20)
ax[0,0].set(title="S500: original single-route validation",xlabel="Epoch",ylabel="Mean makespan")
for dataset,a,title in [("real529",ax[0,1],"S500: Real529 (8x augmentation)"),("synthetic5000",ax[1,0],"S500: synthetic 5000 (no augmentation)")]:
    rr=sorted([r for r in external if r["run"]=="S500" and r["dataset"]==dataset and r["J"]=="all"],key=lambda r:r["epoch"])
    a.plot([r["epoch"] for r in rr],[r["model_mean"] for r in rr],"o-",color="#15803d",lw=1.2,ms=4)
    a.axhline(rr[0]["baseline_mean"],color="#dc2626",ls="--",label="Gurobi-trained E200")
    a.set(title=title,xlabel="Epoch",ylabel="Mean makespan");a.legend(fontsize=8)
for label,mult,color in [("S500",10000,"#2563eb"),("R200",20000,"#b45309"),("LKH100",20000,"#15803d")]:
    rr=sorted([r for r in selection_rows if r["run"]==label],key=lambda r:r["epoch"])
    ax[1,1].plot([r["epoch"]*mult/1e6 for r in rr],[r["mean_makespan"] for r in rr],"o-",ms=3,lw=1.1,label=label,color=color)
ax[1,1].set(title="Same 1000-case full-POMO selection set",xlabel="Generated training instances (millions)",ylabel="Mean makespan")
ax[1,1].legend(fontsize=8)
for a in ax.flat:a.grid(alpha=.2)
fig.savefig(OUT/"validation_and_generalization.png",dpi=180);plt.close(fig)

fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for a,data,labels,title in [(ax[0],ablation,["swap","two_opt","both"],"Local-search ablation at E20"),(ax[1],near,["nm05","nm10","nm20"],"Near-miss quota ablation at E20")]:
    vals=[]
    for label in labels:
        g=data["groups"][label]
        key=next(k for k in g["stages"] if "020_after_il_full_pomo" in k)
        vals.append(g["stages"][key]["mean"])
    a.bar(labels,vals,color=["#3b82f6","#f59e0b","#22c55e"])
    a.set(title=title,ylabel="Full-POMO diagnostic makespan (50 cases)",ylim=(17.6,18.3))
    a.grid(axis="y",alpha=.2)
    for i,v in enumerate(vals):a.text(i,v+.012,f"{v:.4f}",ha="center",fontsize=9)
fig.savefig(OUT/"ablation_full_pomo.png",dpi=180);plt.close(fig)

sections=[]
def add(text):sections.append(text.strip()+"\n")

add(fr"""
# CVTSP 实验完整复盘：从500个epoch训练到LKH＋near-miss20%

**证据截点：{SNAPSHOT}（UTC+8）。**

本报告覆盖旧版500轮训练、历次外部测试与独立回测、奖励选路诊断、短程IL/学习率控制实验、200轮重训、会议后的局部搜索/near-miss/IL标签实验，以及目前的LKH100轮训练。内容依据当前项目中保存的参数、指标CSV、逐例结果JSON、完成审计、源码与会议纪要整理；对话中的近似数值不覆盖已核实的原始结果。

报告只汇总已有实验，不改变训练、Reference更新规则或选模程序。当前训练数据是动态产生的，本报告保留读取时的指标快照；后续epoch不在此截点内。文中的“最好”均限定于相应训练和已评测的checkpoint集合。

**数值约定：**makespan越低越好；除特别注明，`gap=(本方法均值/基线均值−1)×100%`，正值表示本方法更差。不同数据集、不同解码协议的绝对均值不能直接比较。95%区间若来自实例配对bootstrap，只说明这些实例上的差异，不包含不同训练seed的不确定性。

## 1. 当前能够确认的结论

1. 旧版S500的固定验证最优在Epoch365（18.139799），但目前已完成的外部测试中Epoch100更好：Real529为333.459454，人工5000为17.751879。固定验证和最终测试确实存在排序背离。
2. 独立完整POMO选模集在旧版已回测29个checkpoint中选出Epoch100，说明改评测方式有用；这并不解释或消除训练本身的退步。
3. 旧版E100→E365的训练式采样诊断中，上界选路regret反而下降，完整POMO精确最优基本不变。不能把后期退步归因于“上界排序越来越差”。
4. 从E100继续20轮，IL5→2或加快LR衰减均未超过E100；相对原E120的改善区间均跨零，尚无明确获益证据。
5. 三组从头20轮局部搜索实验中，swap-only明显落后；2-opt-only在单路线固定验证更好，swap＋2-opt在独立完整POMO50例中均值最好。不能只报一个有利指标。
6. near-miss20%增加了精算覆盖和改进Expert比例，并改善前20轮单路线固定验证；独立完整POMO50例的均值最好是10%组。没有证据证明20%是最终最优比例。
7. LKH在冻结策略、相同输入的100例搜索实验中改善了Expert平均质量；此证据只覆盖搜索阶段，不能等同于从头训练和泛化提升。
8. 当前LKH训练重新选模后，E46的外部表现优于旧验证选出的E59，但仍不及旧版S500-E100，更未超过Gurobi训练基线E200。
9. 每epoch实例数从1万改到2万，改变了优化节奏：同样100万实例，RL更新数接近，IL更新数从40,510降至21,165，学习率也更低。“IL都是5遍”不足以说明训练强度对齐。
10. 质量加权IL的B组只完成6轮，第7轮中途停止；不具备20轮终点结论。后续LKH训练没有启用质量加权IL。

## 2. 训练与实验代号：避免混淆同名Epoch

{table(['代号','对应实验','计划与实际','主要用途'],[
['G200','POMO＋Gurobi训练基线的Epoch200','200×100,000≈2000万训练实例','共同外部基线；不是纯Gurobi全局TSP最优解'],
['S500','旧版向量化、质量受限多样性＋密集POMO训练','500×10,000＝500万，已完成','历史主实验，E100为已测外部最好'],
['E100-IL2 / LR150','从S500-E100继续的两个控制分支','E101–120，各20轮','只解释旧模型后期训练，不是从头终点成绩'],
['R200','2万实例、IL5/2/1、replay10%、TopK随J增加','200×20,000＝400万，已完成','另一次从头重训；不要把其E176当成S500-E178'],
['LS-A/B/C','swap-only / 2-opt-only / swap＋2-opt','各20×20,000＝40万，均已完成','只改变搜索算子，move按100轮30→10'],
['NM05/10/20','固定near-miss配额5%/10%/20%','各20×20,000＝40万，均已完成','固定30move，组间只改变配额'],
['QIL-B','按Expert相对Target真实质量加权CE','计划20轮，实际完成6轮，第7轮中止','未完成实验；不能认定加权普遍有效或无效'],
['LKH-pilot','冻结NM20-E20策略的LKH搜索对照','100例，无网络训练','验证搜索适配与Expert质量'],
['LKH100','LKH＋near-miss20%的当前从头训练',f'计划100×20,000；截点完成{int(float(runs["LKH100"]["latest"]["epoch"]))}轮','本轮旧验证best E59，已测选模best E46'],
])}

### 2.1 时间线

{table(['记录日期','实验/工作','结果状态'],[
['2026-09-17至09-23','S500向量化主训练及多次外部测试','500轮完成；E100后外部测试未同步改善'],
['2026-09-21起','建立独立1000例完整POMO选模集并回测','S500共29个checkpoint完成回测'],
['2026-09-22','E100/E365奖励选路诊断；IL2/LR150控制分支；sample测试','结果均已落盘'],
['2026-09-23至09-30','R200从头重训、硬件/worker异常排查及压力测试','9月28日压力测试退出0；正式200轮最终退出0'],
['2026-09-30','R200独立选模','在16个已测checkpoint中选E80，仍不及S500-E100'],
['2026-10-10','整理旧日志，准备2000验证＋500测试，恢复E500参数分支','准备动作，不冒充新增完整训练结果'],
['2026-10-12至10-13','三组局部搜索从头20轮及过程诊断','全部完成并通过完成审计'],
['2026-10-13至10-14','三组near-miss配额从头20轮','全部完成并通过完成审计'],
['2026-10-14','质量加权IL-B、LKH100例pilot、启动LKH100','B提前停止；pilot完成；LKH从随机初始化开始'],
['2026-10-15','LKH-E59/E46外部测试与独立选模','两套外部集均完成；LKH100训练尚未结束'],
])}

## 3. 核心算法：哪些模块一直保留，哪些后来改变

### 3.1 每个训练epoch的整体流程

1. 随机生成新的CVTSP训练实例，J在20–100范围随机抽取；同batch采用同一J。Reference为每例采样200条候选。
2. Edge Predictor给候选的有向边代价打分，选Anchor；固定Anchor访问顺序，用Gurobi精算连续位置和时间。
3. 由Anchor的可行连续解构造有向严格上界表C0。它给定路线一个可实现的上界，不能当成该路线优化后的精确makespan。
4. 按C0选Top-K：通常固定30，质量部分20，多样性部分最多10；多样性候选同时受前60和相对质量门槛5%约束。合格多样性不足时质量补齐，因此实际并非每例恰好20＋10。
5. 以这些路线为起点，在固定C0上搜索。S500及大部分消融用swap/2-opt；LKH100改用LKH。中间不调用Gurobi。
6. 候选上界严格优于Anchor时精算验证；未通过严格改善门槛但仍有不同路线的实例，可按near-miss配额额外精算一个候选。只有精确makespan严格改善才替换Anchor成为改进Expert，否则保留Anchor标签。
7. Target生成POMO路线。Anchor/Expert分别给出完整路线的严格上界，先求各自整条路线总和，再在完整上界之间取小；不能逐边取小拼接。
8. 选一条上界最低的Target路线做Gurobi精算，其余POMO路线用上界，进行RL。整个epoch的RL结束后，再汇总标签做IL和Edge Predictor更新；改进Expert进入历史池。它不是每个batch完成一次完整RL→IL交替。
9. 每轮按固定验证比较Target和Reference，严格更好时更新Reference；保存checkpoint、优化器、scheduler、Reference、Predictor、replay和RNG。

### 3.2 RL、IL和Replay的准确含义

密集POMO主奖励使用一条精确Target值和其余路线的严格上界，每个实例内部对优势中心化并标准化，让全部POMO路线参与梯度。精确辅助项比较那条被精算Target与Expert：

$$A_E=\operatorname{{clip}}\left(\frac{{V_E-V_T}}{{\max(|V_E|,10^{{-12}})}},-1,1\right),\qquad
L_{{RL}}=L_{{POMO}}+\lambda_E L_{{exact}}.$$

`λE`按预先定义的训练能力指标增加到1，不由外部测试决定。密集奖励仍混合了上界与精确值，因此不是“J条Target全部精算”的RL。

常规IL采用Expert访问序列的teacher-forcing交叉熵，固定5遍。改进Expert的权重5在实现上为重复5份；保留Anchor的标签为1份。Replay通常占扩充后IL样本的30%，只存历史真实改进Expert，容量50,000；达到容量后按现有淘汰规则更新，不是停止产生新Expert。固定容量本身不能证明平台的原因。

**Expert比例、Expert真实改善、IL准确率、POMO最好路线质量是不同指标。**新Expert多不等于标签一定适合当前Target；token准确率高不等于整条路线或完整POMO的最优makespan持续下降。

### 3.3 旧版Epoch100之前已存在的主要算法变化

- 从不受质量约束的“差异最大10条”改为有质量门槛的多样性：仍探索不同访问结构，但避免非常差的路线仅凭差异进入Top-K。
- 恢复密集、标准化POMO主奖励，加入有界精确Expert辅助，缓解只比较一条Target与Expert时的冷启动信号问题。
- near-miss按训练能力从1%到5%变化，改进Expert权重5、IL5遍、replay30%。
- 局部搜索改成结果等价的向量化swap/2-opt，保留候选、接受条件和tie-breaking。

**不能算入旧E100的后续变化：**固定near-miss20%、LKH、质量加权IL、2万实例/轮、100轮到1e-5的LR计划，以及R200的IL5/2/1、replay10%、自适应Top-K。旧S500没有完整源文件快照，后来的“恢复E500逻辑”依据保留参数和现有实现分支，不等于逐字节回退到旧代码。

### 3.4 向量化属于哪一类变化

原swap/2-opt逐候选在Python中计算，向量化实现把同一路线的候选代价变化集中到数组计算，并缓存每种J的候选索引，减少函数调用、tuple创建和解释器循环。其设计目标是保持候选集合、接受条件、tie-breaking和最终路线一致，属于工程加速；不能把速度变化当作一次新的算法质量提升。原S500启动脚本明确启用`CVTSP_VECTORIZED_LOCAL_SEARCH=1`。本报告未重新运行训练或声称复验全部历史等价性测试；当前LKH是另一种搜索算法，不能用“向量化等价”说明LKH与原算法等价。

## 4. 数据集与评测协议：哪些数字可以放在同一张表

{table(['集合','数量与J','解码与精算','用途'],[
['旧固定验证 macsim_fixed','1000；5种J各200','单条free-start greedy，Gurobi精算','S500/R200的Reference更新与best'],
['新固定验证 validation2000','2000；5种J各400','仍为单条free-start greedy，Gurobi精算','局部搜索、配额、B组、LKH训练在线验证'],
['独立完整POMO选模集','1000；5种J各200；seed20260921','greedy，POMO=J，无增强，精算所有不同候选取最优','跨checkpoint回测及选模；是开发集'],
['新独立诊断集','100；5种J各20；seed20261012','单路线；其中预先固定50例做完整POMO','阶段/梯度/搜索诊断，不是最终测试'],
['新人工test500','500；5种J各100；seed20262010','完整POMO，无增强，Gurobi精算','已准备；本报告未找到正式完整测试结果'],
['旧人工synthetic5000','5000；5种J各1000；seed20260827','greedy，POMO=J，无增强，Gurobi精算','一直保留的外部评测集'],
['Real529全体','529，J10至200共19种规模','greedy，POMO=min(J,100)，8倍几何增强，Gurobi精算','外部真实实例评测'],
['Real529训练规模子集','284；J=20/25/30/35/40/45/50/100','与全529采用同一评测结果筛子集','范围内分析；不能替代全529'],
])}

新验证/测试已对已保存的旧固定集、完整POMO开发集和人工5000等做精确内容查重。历史全部随机训练实例未保存，因此只能说明独立RNG和对已保存集合无重复，不能声称已逐条证明与所有历史训练数据不重叠。

完整POMO选模的分数为“每例先在候选间取最小，再对实例取平均”，不是把全部路线makespan直接平均。它与人工5000的核心评测方式一致；Real529额外有8倍增强、J>100时POMO上限100，不能说三者完全同协议。

G200与Self-improving测试时最终都用Gurobi精算固定访问顺序。比较的是训练后的上层策略，不是“Gurobi求解器对LKH求解器”。LKH100的这些外部测试**没有在测试时额外运行LKH**。

### 4.1 Gurobi训练基线的实际配置

G200来源为`outputs/train__cvtsp_random_n100_nearest_quarter/checkpoint-200.pt`。保存参数：每轮100,000实例、batch512、学习率5e-5、seed1234，网络embedding128、encoder6层、8头；Self-improving参数为embedding256、encoder4层、16头、dropout0.1。G200在E200名义累计约2000万实例，S500-E100约100万。它是既有实验基线，不是训练数据预算和网络架构完全相同的单变量对照；不能把相近测试成绩单独归因于某一个模块。
""")

add(fr"""
## 5. S500：500轮旧主实验的训练轨迹

### 5.1 实际配置与完成状态

{table(['项目','S500配置'],[
['总轮数 / 每轮实例 / 总实例','500 / 10,000 / 5,000,000'],['J / batch / POMO','20–100 / 64 / J'],
['学习率','1e-4→1e-7；按epoch的余弦，warmup1轮'],['Reference候选 / Top-K','200 / 固定30（质量20＋质量受限多样性最多10）'],
['局部搜索','向量化swap＋2-opt，每起点最多30接受move'],['near-miss','按训练能力在1%–5%范围变化；ceil配额使实际比例可能更高'],
['IL / changed权重','5遍 / 5'],['Replay','容量50,000，实际IL占比约30%'],['验证','旧1000例，单greedy，每轮'],
['实际完成','metrics.csv共500行，最后一轮E500，保存主优化器累计281,015步'],
])}

### 5.2 分阶段汇总

{table(['轮次','固定验证起→止','Expert平均真实改善','改进Expert比例','IL token准确率','平均move/起点'],[
[f'{p["epochs"][0]}–{p["epochs"][1]}',f'{n(p["validation_start"],4)}→{n(p["validation_end"],4)}',n(p['expert_improvement_percent'],3)+'%',n(p['expert_better_percent'],2)+'%',n(p['il_accuracy_percent'],2)+'%',n(p['accepted_moves_per_start'],2)]
for p in old_review['phases']])}

这些Expert改善百分比是逐实例相对Anchor改善的汇总，不能直接用两个训练makespan均值相除替代。训练实例的J和难度每轮变化，训练makespan不是固定评测曲线。

### 5.3 关键epoch的固定验证与外部结果

{table(['Epoch','累计实例','旧固定验证','Real529','Real284','人工5000'],[
[ep,f'{ep/100:.2f}百万',n(float(runs['S500']['by_epoch'][ep]['target_mean_objective'])),n(score('S500',ep,'real529')),n(score('S500',ep,'real529_J20_100')),n(score('S500',ep,'synthetic5000'))]
for ep in [4,12,16,100,111,127,178,296,365,500]])}

“待补充”表示没有找到该checkpoint对应的已完成测试，不能由相邻epoch插值。S500-E500完成训练，不代表E500完成外部测试。

E100固定验证18.218866，E365降到18.139799，改善约0.434%；E500为18.156394。与此同时E100→E365，Real529上升0.718149，人工5000上升0.104079。报告中的原始逐例配对区间分别为[+0.387246,+1.039872]和[+0.092016,+0.116680]，支持在这些固定外部实例上E365退步，并不是只差一个随机小数。

![旧训练固定验证、外部表现与共同选模集](validation_and_generalization.png)

图中不同集合使用不同子图；右下才是同一完整POMO选模集上的跨训练比较，横轴换算为累计新训练实例。连线只连接已经评测的checkpoint，不代表其间每轮都测试。

### 5.4 E100与G200的差距

{table(['集合','G200','S500-E100','gap','E100胜/平/负'],[
[ds,n(ext('S500',100,ds)['baseline_mean']),n(score('S500',100,ds)),n(ext('S500',100,ds)['gap_percent'],3)+'%',f"{ext('S500',100,ds)['wins']}/{ext('S500',100,ds)['ties']}/{ext('S500',100,ds)['losses']}"]
for ds in ['real529','real529_J20_100','synthetic5000']])}

### 5.5 解释边界

- 这500轮证实：同一配方继续训练，旧单路线验证可以改善，而完整POMO外部表现没有持续改善。
- 它不证明E100以后所有训练计算都毫无作用：上界选路、单路线质量、IL准确率等确实变化；问题是这些变化没有转化成目标测试表现。
- 固定验证集未参加梯度更新，但用于Reference晋升，因而会间接改变后续训练。存在开发集选择反馈，但仅凭验证/测试背离不能把原因确定为过拟合。
- “验证集固定”本身不是错误；单路线验证与最终多路线取最优的目标不一致，是已确认的评测差异。

### 5.6 旧训练保存的时间数据

{table(['轮次','平均训练字段/分钟','平均固定验证/分钟'],[
[f'{p["epochs"][0]}–{p["epochs"][1]}',n(p['train_minutes'],2),n(p['validation_minutes'],2)] for p in old_review['phases']])}

时间来源为旧metrics及其分阶段汇总，训练字段与验证字段分列；这不是含暂停、checkpoint保存、所有资源等待的整段自然日时间。三组并行实验中的“每轮完整墙钟”另有定义，不能把不同口径的分钟数直接解释成某算法慢了几倍。会议提到的“近七倍提速”在本报告中不作为已完成同机严控benchmark结论。
""")

add(fr"""
## 6. 独立完整POMO回测：验证选模方式

使用seed20260921新建的1000例固定开发集，J20/40/60/80/100各200。每例完整POMO，greedy、无增强、Gurobi精算所有不同候选。已回测的29个S500 checkpoint中，E100均值最低。

{table(['Epoch','完整POMO选模makespan','相对E100的均值差'],[
[ep,n(select_epoch_mean(old_selection,ep)),n(select_epoch_mean(old_selection,ep)-select_epoch_mean(old_selection,100))]
for ep in sorted(old_selection['evaluated_epochs'])])}

可核实的代表结果：E100=17.990319；E365并未列入这29个回测点，邻近E360=18.080753；E500=18.083433。不要把E360的选模分数冒充E365的分数。

**结论：**这套评测能在历史候选中选到外部测试也较好的E100。但它已经反复用于开发判断，属于开发验证，不再是未接触的独立测试集；亦不能声称“在全部500个checkpoint中已经穷尽证明E100最好”。

## 7. 诊断一：是否因为Target上界选错路线而退步

### 7.1 实验设置

200个新固定实例，五种J各40，seed20260922；同实例分别恢复S500-E100和E365的Reference/Predictor/上界流程。按训练sample模式生成Target POMO，上界最低者为训练规则选中路线；再把本次全部Target候选都Gurobi精算，得到“本次候选中的精确最优”。这不是全局CVTSP最优，也不是greedy外部评测。

{table(['指标','E100','E365','E365−E100的95%区间'],[
['上界选中路线的精确值','18.529323','18.406458',ci(regret['selected_exact']['ci95'])],
['全部POMO中的精确最好','17.925014','17.923651',ci(regret['best_exact']['ci95'])],
['选择regret＝选中−最好','0.604309','0.482806',ci(regret['regret']['ci95'])],
['恰好选中精确最好的比例','16.5%','17.5%','原报告未给比例区间'],
['所有POMO路线的平均上界松弛','13.503542','7.949639',ci(regret['mean_bound_gap']['ci95'])],
])}

**结论：**上界选择仍有误差，但E365的选择regret更低，并非越来越差；这份诊断不支持“后期退步由上界误排序恶化导致”。采样候选中的真实最好值基本没进步，更值得检查路线分布、RL/IL学习过程与测试目标。

## 8. 诊断二：sample解码、不做8倍增强

两模型每例各做一次随机POMO rollout，逐例重置为相同seed；精算全部不同候选取最优。相同seed不能保证不同模型采到同一路线，作用是对齐随机输入。此结果是单seed的sample协议，不与greedy/8倍增强成绩混成一列。

{table(['集合','G200 sample','S500-E365 sample','E365−G200','95%区间','胜/平/负'],[
[label,n(p['gurobi_e200_mean']),n(p['self_e365_mean']),n(p['self_minus_gurobi']),ci(p['ci95']),f"{p['self_wins']}/{p['ties']}/{p['self_losses']}"]
for label,p in [('Real529',sample_results['real529']['results']['all']),('Real284',sample_results['real529']['results']['J20_100']),('人工5000',sample_results['synthetic5000']['results']['all'])]])}

**结论：**切换sample并未让E365超过G200；不能据此认为只要改测试解码就能消除差距，也不能凭单seed确定sample永远更差。

## 9. 从E100继续的IL与学习率控制实验

### 9.1 设计

均从同一个S500-E100恢复参数、optimizer、Reference、Edge、replay50,000和RNG，只观察E101–120。IL2分支把5遍改为2遍；LR150分支将余弦继续轨迹锚定在E100实际优化器LR约9.0571e-5，计划到E150降至1e-7。S500原E120作为已有对照；测试用相同1000例完整POMO开发集。

{table(['模型','选模makespan','相对E100改善','95%区间','相对原E120改善'],[
['E100',n(branches['all']['mean_makespan']['epoch100']),'—','—','—'],
*[ [name,n(branches['all']['mean_makespan'][key]),n(branches['all']['vs_epoch100'][key]['mean_improvement']),ci(branches['all']['vs_epoch100'][key]['ci95']),n(branches['all']['vs_original_epoch120'].get(key,{}).get('mean_improvement')) if key!='epoch120' else '—']
   for name,key in [('原E120','epoch120'),('IL2-E120','il2_e120'),('LR150-E120','lr150_e120')] ],
])}

此表“改善”为E100或原E120减分支，正值更好，与全报告默认gap符号不同。IL2与LR150相对原E120的区间分别为[-0.010420,+0.021529]、[-0.005049,+0.028237]，均跨零。

**结论：**两个改动的点估计都略好于原E120，但未超过E100，也未提供相对原E120的清晰改善证据。这只能限制“旧E100后的20轮续训”解释，不能推导从头训练必然失败；也不能作为最终重训成绩。

## 10. R200：2万实例＋IL5/2/1＋replay10%＋随J增加Top-K

### 10.1 与S500同时改变的因素

- 总轮数200，每轮20,000，总400万新实例；学习率1e-4→1e-7，200轮余弦。
- IL预先固定为E1–50五遍、E51–100两遍、E101–200一遍。
- Replay30%/50,000改为10%/20,000。
- Top-K从固定30改为`max(30,ceil(0.5J))`，J80时40，J100时50，保留最多10条质量受限多样性起点。
- 保留密集POMO＋有界精确Expert辅助、swap/2-opt30步、adaptive near-miss1%–5%、旧单路线验证。

这是多项共同变化的从头训练，不能用它单独断言IL、Replay或Top-K各自的效果。

### 10.2 实际训练与选模

R200已完整训练200轮，最终TRAIN_EXIT=0（2026-09-30 14:49:33 UTC+8）。旧固定验证best为E176=18.534560；E200为18.683525，末轮IL为1遍，累计主优化器99,497步。

同一1000例完整POMO开发集回测16个checkpoint，选出E80=18.105771；旧S500-E100在同集合为17.990319，R200选模最好仍落后0.115452。

{table(['R200 Epoch','完整POMO选模makespan'],[[r['epoch'],n(r['mean_makespan'])] for r in sorted(r200_selection['ranking'],key=lambda r:r['epoch'])])}

### 10.3 外部测试

{table(['模型/集合','makespan','G200','gap','完成情况'],[
['R200-E11 / Real529',n(score('R200',11,'real529')),n(ext('R200',11,'real529')['baseline_mean']),n(ext('R200',11,'real529')['gap_percent'],3)+'%','529例完整'],
['R200-E11 / 人工5000','待补充','17.282643','待补充',f'用户停止；仅{formal11_status["synthetic_completed_instances"]}例，不计完整成绩'],
['R200-E176 / Real529',n(score('R200',176,'real529')),n(ext('R200',176,'real529')['baseline_mean']),n(ext('R200',176,'real529')['gap_percent'],3)+'%','完整'],
['R200-E176 / 人工5000',n(score('R200',176,'synthetic5000')),n(ext('R200',176,'synthetic5000')['baseline_mean']),n(ext('R200',176,'synthetic5000')['gap_percent'],3)+'%','完整'],
['R200-E80 / 两套外部集','待补充','见上','待补充','本次未找到完整正式结果'],
])}

**结论：**这套联合改动没有产生优于S500-E100的独立选模成绩。E176的完整外部测试也更差；不能因此认定“2万实例本身无效”，因为IL、Replay、Top-K、LR和更新比例同时变化。

### 10.4 崩溃与压力测试的记录边界

9月23日压力测试日志有两处`Fatal Python error: Segmentation fault`；工作区保留带`blocked_host_mce`等命名的失败目录。仅凭目录名和Python段错误，不能在本报告确定物理内存、CPU或BIOS的根因。9月28日新压力测试完成，日志`STRESS_EXIT=0`（17:59），正式R200最终也完整退出0。压力测试产物不计入正式训练成绩，单次通过不证明所有硬件隐患永久消失。未找到可复核的完整MemTest86/BIOS操作报告，标为待补充。
""")

add(fr"""
## 11. 会议要求、日志补齐与实验执行顺序

会议纪要里的“Group/groovy/promo”等为转写误差，技术名词以代码中的Gurobi/POMO为准；纪要中“已过拟合”“最优327–330”等表述不能替代数值实验或全局最优证明。

{table(['会议关注项','对应动作','已有证据/剩余缺口'],[
['训练曲线和验证指标（16:49、29:04）','整理500行metrics和外部结果；生成曲线','本报告第5节、配套CSV和图；不同协议分开画'],
['局部搜索实际步数（30:56）','统计每J/起点的均值、分位数、上限命中、步数分布','三组LS实验实际记录；旧500轮缺这些细项，无法补造'],
['swap与2-opt分别正向贡献（39:19、40:59）','从头三组算子对照；记录接受数与上界下降','已有真实日志；上界贡献不等于逐move精确makespan贡献'],
['loss、梯度和reward（34:29）','记录RL/IL每次更新梯度、RL两分项和优势分布','诊断epoch首batch有RL分项梯度方向；无全量RL/IL同状态夹角'],
['LKH对照（27:49、40:59）','冻结策略100例搜索比较，再启动从头LKH训练','pilot完成，训练仍进行；两者证据范围分开'],
['实例预算和学习率','2万实例/轮；100轮、minLR1e-5','新实验实际生效，但未做实例数×LR的单变量因果对照'],
['标签多样性/数据增强','保留质量受限多样性和Replay，检查终点多样性','没有完成新的训练几何增强消融'],
])}

10月10日先整理日志、恢复E500算法参数分支并准备新数据；之后用户将最初“旧三checkpoint离线分析”调整为三组全新初始化训练，故正式LS实验没有加载旧E20/E100/E365来当训练起点。

## 12. 三组从头20轮：swap-only、2-opt-only、swap＋2-opt

### 12.1 控制条件与完成核验

三组相同初始化seed20260813、相同Target/Reference/Predictor初始哈希、空Replay；隔离训练实例RNG并逐epoch校验数据指纹。每组40万新实例。LR按100轮从1e-4到1e-5，move按100轮30→10，第20轮上限26；其余保留E500训练逻辑。GPU4090 24GB，32个CPU线程，每组8worker，总24；大显存阶段最多两组同时占用GPU。

完成审计显示三组20轮checkpoint、诊断与源码快照齐全，训练数据一致，各组从头开始，后续候选自然随模型分化。三组都用平滑move计划，所以不能从本实验推导“平滑比固定30更好”。

### 12.2 固定验证与完整POMO诊断

{table(['组别','固定验证best epoch','固定验证best','E20固定验证','E20完整POMO50例'],[
[name,g['best']['epoch'],n(g['best']['fixed_validation']),n(g['epoch20']['fixed_validation']),n(g['stages'][next(k for k in g['stages'] if '020_after_il_full_pomo' in k)]['mean'])]
for name,g in ablation['groups'].items()])}

独立完整POMO的50例中合用均值最好，但单路线固定验证由2-opt-only领先；两个指标测量的能力不同。一个seed、50例诊断不足以证明最终100轮合用显著优于2-opt-only。

![局部搜索和near-miss实验的完整POMO诊断](ablation_full_pomo.png)

### 12.3 RL后与IL后的分阶段结果

{report_section(f'{ABL}/report.md','## Epoch20网络学习诊断','## Epoch20局部搜索诊断')}

合用组单路线IL后显著上升，但完整POMO变化接近零且区间跨零。因此不能把“单路线IL后变差”直接解释为“完整POMO的IL一定有害”。

### 12.4 平均move、上限命中与算子贡献

{report_section(f'{ABL}/report.md','## Epoch20局部搜索诊断','## 实际训练成本')}

大规模J100中合用组平均25.04步，上限26的命中率93.3%，第10步后的上界下降占38.3%。这与旧500轮总体平均move后期较低不矛盾：新训练早期、不同J和不同起点的分布不同。**不能因为旧总体均值低，就断言新训练的J100只需10步。**上界下降占比不能解释为同百分比的真实makespan改善。

合用组swap占大部分接受次数和上界下降，并不意味着2-opt没有价值：较少但不同结构的move仍可能影响最终路线；需结合完整POMO结果，不能只按接受次数删算子。

### 12.5 时间成本

{report_section(f'{ABL}/report.md','## 实际训练成本','已核对三组训练数据一致')}

三组并发时的每轮时间包含GPU等待、固定验证、诊断和保存；不能拿它与单作业训练主循环的分钟数直接比较。该实验没有重新变回串行搜索。

**结论与当时决策：**swap-only不优先；2-opt-only和合用都保留竞争力，之后继续采用swap＋2-opt。未找到足够理由仅据这组实验增加Top-K或将move大幅减到10，后续配额实验固定30。

## 13. near-miss固定配额5% / 10% / 20%：三组从头20轮

### 13.1 控制设置

三组相同新初始化与全部20轮训练实例指纹；各40万新实例。固定swap＋2-opt30步，Top-K30、前60/5%多样性质量门槛，IL5遍、changed权重5、Replay30%/50,000，LR100轮到1e-5，验证2000不变。与前一LS合用组不同，此处move固定30，不能把前一组直接作为NM05的等价控制组。

配额为每batch最多`ceil(rate×batch_size)`个合格实例，每个精算一个不同于Anchor的候选；不足就少选。标准batch64时5%/10%/20%分别上限4/7/13，即6.25%/10.9375%/20.3125%。near-miss不是随机放过同等比例的坏标签：精算后严格改善才接纳。

### 13.2 质量、Expert与成本

{report_section(f'{NM}/report.md','状态：','逐J结果、全部RL/IL阶段质量')}

### 13.3 20轮累计真实配额

{table(['组别','额外精算次数','接纳的near-miss Expert','40万实例实际精算占比'],[
[name,g['attempts'],g['near_miss_accepted'],pct(g['actual_attempt_fraction'])]
for name,g in nm_audit['groups'].items()])}

注意上面的累计比例与“E20当轮实际精算比例”不是同一统计窗口；不能拿19.7765%与20.32%当作日志矛盾。原metrics字段`near_miss_attempt_rate`有按eligible计的口径，真实全实例配额应优先核算attempt_count/实例数或读取审计的`actual_attempt_fraction`，不能把eligible内60%误报为全实例60%。

### 13.4 结论与局限

20%比5%显著增加得到改进Expert的机会，但随着配额增加，边际候选的精算成功率下降。单路线固定验证中20%最好；完整POMO50例中10%=17.777700，20%=17.823019，5%=17.888187。因此“near-miss值得调”有证据，“20%是最佳并会提高最终泛化”没有证据。后续选择20%是待检验的配置决策，不是已证明的最终最优。

## 14. IL标签质量加权B组：完成6轮，提前停止

### 14.1 设计与公式

用户选择只运行B组，没有同期新训练A组；NM20作为历史普通IL参照。保持2万实例、near-miss20%、swap＋2-opt30步、IL5遍与Replay30%。每轮RL完成后，用该时刻Target快照重新测量当前与抽到的历史标签，按sample POMO上界选一条Target路线并精算：

$$g=\frac{{V_T-V_E}}{{\max(|V_T|,10^{{-12}})}},\qquad q=\operatorname{{clip}}(0.5+g/0.02,\ 0.1,\ 2).$$

标签CE按`sum(q×token_NLL)/sum(q×J)`归一化，保留正权重；changed重复5次和5遍IL规则仍保留。测量仅比较一条上界选中Target，不是与全部Target候选中的精确最好比较。每个不同(instance,expert)标签新增一次Target精算，额外成本单列。

### 14.2 实际早期结果

{table(['Epoch','普通NM20固定验证','质量加权B固定验证','B−普通','普通IL准确率','B IL准确率'],[
[int(float(b['epoch'])),n(float(a['target_mean_objective'])),n(float(b['target_mean_objective'])),n(float(b['target_mean_objective'])-float(a['target_mean_objective'])),pct(a['imitation_token_accuracy'],2),pct(b['imitation_token_accuracy'],2)]
for a,b in zip(nmrows,qrows)])}

E5完整POMO50例：普通NM20=18.233370，B=18.190758，B在此小诊断上略好，**并非所有指标都更差**。E6单路线固定验证B=20.058656、普通NM20=19.769582，B更差0.289074。B各轮额外质量测量约130–224秒，不含它就不能公平报告该方案的总成本。

用户随后要求结束B组；文件显示完整metrics仅E1–E6，第7轮RL日志到17,536/20,000，未产生完整E7结果。旧`running_status.json`停留在10月14日15:00的观察值，不能据其`tmux_alive=true`认定现在仍运行。

### 14.3 当前可下的结论

这一具体加权公式没有展示稳定优势，不能认定“质量加权已成功”，也不能把未完成6轮实验推成“IL标签质量完全无用”。更可能需要检验Target比较基准、权重尺度、历史标签与当前策略的差距，而非直接增加IL遍数。此为待验证解释，不是已定位根因。

日志已保存Expert路线、VT/VE、g/q及贡献份额；B进程最初没有保存对应Target比较路线。后来补好的字段按用户选择在下一次训练启用，不能倒称B历史日志已经具备逐例Target路线。当前LKH新训练实际已记录Target与Expert两条路线，但未启用B的质量加权CE。
""")

add(fr"""
## 15. LKH冻结策略搜索对照：100例

### 15.1 接入方式与预算

冻结NM20-E20的Reference和Edge Predictor，生成100例共同候选（五种J各20，seed20261014）。同一Anchor精确解、同一C0、同一Top-K；原搜索与LKH分别处理这些起点，不更新网络。

有向C0连同depot映射为J+1节点ATSP，使用保存的LKH-3.0.6源码和二进制。RUNS1、MAX_TRIALS1、MOVE_TYPE5、MAX_CANDIDATES5。整数scale最高1e6并控制32位安全范围；最终路线必须在原始浮点C0重新验算。搜索中不调用Gurobi，之后按同一严格上界和near-miss20%规则精算。

- `all_starts`：30起点各一次trial；trial不是一个接受move。
- `matched_budget`：按本例原swap/2-opt实测时间作为近似预算，起点交错质量/多样性；未运行起点保留原路线。有启动/kill超额开销，不能称严格等墙钟。

### 15.2 全体与逐规模结果

{report_section('outputs/lkh_search_pilot_20261014/report.md','| J |','局限：').join(['| J |',''])}

### 15.3 终点覆盖面与“Expert更多”的区别

全起点LKH平均只保留约1.22条不同终点，原swap＋2-opt约24.43条；全起点LKH在100例仅触发1次near-miss精算。搜索更容易聚到同一条或Anchor附近的路线，可能减少可供额外精算的不同候选；这是候选覆盖面的观察，不证明网络最终多样性必然下降。

**已证明：**该冻结策略和该100例下，LKH得到更低的Expert均值，配对区间不跨零；matched profile也更快。**未证明：**cold-start训练更好、最终网络更好、任意预算/实例/seed都更好。全起点LKH的改进Expert仅12例，少于原14例，却平均质量更好，已经说明“Expert数量”不能替代“改善幅度”。

## 16. 当前LKH100训练：实际配置与恢复核查

### 16.1 配置

{table(['设置','本轮'],[
['初始化','随机新Target/Reference/Predictor；不加载历史模型；初始replay0'],
['轮数 / 每轮实例','100 / 20,000（总计划200万）'],['LR','1e-4→1e-5，100轮余弦，warmup1'],
['J / batch / workers','20–100 / 64 / 32，每worker求解单线程'],['候选 / Top-K','200 / 固定30，质量20＋质量受限多样性最多10'],
['局部搜索','LKH；每例时间预算max(0.005,0.00015×J²)秒'],['LKH参数','最高scale1e6，trial1，move_type5，max_candidates5'],
['near-miss','固定20%，实际每batch ceil并受eligible限制'],['RL','原密集标准化POMO＋有界精确Expert辅助'],
['IL / Replay','普通IL5遍，changed权重5；30% / 50,000'],['在线固定验证','新2000例，仍是单greedy，每轮'],
['设备 / 会话','物理GPU0→逻辑cuda:0 / cvtsp_lkh_nm20_e100'],
])}

新训练的initialization.json记录`fresh_initialization=true`、start_epoch1、replay_size0；Target初始SHA256为`{init['target_sha256']}`，与Reference相同，且和NM20实验的全新初始化一致。此证据支持从头初始化；不能说本轮借用了旧E100/E238模型。

### 16.2 截点训练状态

{table(['指标','截点最后完成一轮','在线固定验证best'],[
['Epoch',int(float(runs['LKH100']['latest']['epoch'])),int(float(runs['LKH100']['best']['epoch']))],
['新固定验证',n(runs['LKH100']['latest']['target_mean_objective']),n(runs['LKH100']['best']['target_mean_objective'])],
['学习率',f"{float(runs['LKH100']['latest']['learning_rate']):.8g}",f"{float(runs['LKH100']['best']['learning_rate']):.8g}"],
['改进Expert比例',pct(runs['LKH100']['latest']['expert_changed_ratio']),pct(runs['LKH100']['best']['expert_changed_ratio'])],
['Expert平均相对Anchor改善',n(runs['LKH100']['latest']['expert_improvement_percent'],3)+'%',n(runs['LKH100']['best']['expert_improvement_percent'],3)+'%'],
['IL token准确率',pct(runs['LKH100']['latest']['imitation_token_accuracy']),pct(runs['LKH100']['best']['imitation_token_accuracy'])],
['replay池大小',int(float(runs['LKH100']['latest']['expert_replay_pool_size'])),int(float(runs['LKH100']['best']['expert_replay_pool_size']))],
])}

这里的18.x在线验证来自新2000例，不能与S500旧1000例的18.x直接横比。训练尚未完成100轮，因此本报告没有本轮最终E100成绩，也不外推训练结束时间。

### 16.3 实际LKH搜索覆盖：以E50为例

{table(['指标','E50日志'],[
['训练实例',lkh50['instances']],['LKH native calls',lkh50['native_calls']],['完成起点 / 超时起点',f"{lkh50['completed_starts']} / {lkh50['timed_out_starts']}"],
['平均完成起点/实例',n(lkh50['completed_starts_per_instance']['mean'],3)],['平均不同终点/实例',n(lkh50['distinct_final_routes']['mean'],3)],
['全部Top-K中返回Anchor的比例',pct(lkh50['returned_anchor_fraction'])],['改进Expert',lkh50['changed_experts']],
['strict/certified来源 / near-miss来源',f"{lkh50['certified_experts']} / {lkh50['near_miss_experts']}"],['平均搜索worker时间/实例',n(lkh50['search_seconds']['mean'],4)+'秒'],
])}

当前运行E50平均完成27.38起点，不是pilot的19.54；实际并发负载、实例和策略都在变化，不能把pilot耗时或完成起点数当作整轮不变常数。返回Anchor比例83.60%是“搜索起点输出”比例，不是83.60%的训练实例都没有改进Expert。

## 17. 当前模型选模：E59为什么被E46替代

原在线单路线验证选E59=18.672909。对LKH100的17个checkpoint使用同一1000例完整POMO开发集，先粗筛再在候选区间细筛，选出E46。

{table(['Epoch','选模makespan','比E46高多少','配对95%区间'],[
[r['epoch'],n(r['mean_makespan']),n(r['difference_from_best']),ci(r['difference_from_best_ci95'])]
for r in lkh_selection['ranking']])}

E46与E39的均值差仅0.002252，区间跨零，不能声称E46显著优于E39。E59−E46=0.048682，区间[+0.022282,+0.074550]，在此开发集上差异更清楚。E46是17个已测点中的最好，并未测完当前所有epoch，更未测未来E100。

共同开发集比较：S500-E100=17.990319，R200-E80=18.105771，LKH100-E46=18.126339。LKH目前选出的模型仍落后旧S500-E100约0.136020；选模更准确没有把这次训练变成更好的模型。

### 17.1 将选模集用于正式固定验证：已讨论，尚未实施

从评价目标看，完整POMO开发集比单路线更适合作为多路线求解的checkpoint选择依据。对话中已讨论每5轮评测、与Reference晋升统一等方案；但当前启动参数仍指向新2000例、`reference_validation_candidates=1`，本报告没有发现运行中已替换验证集的证据，不能把建议写成已完成改动。

单路线旧/新固定验证并非“完全无意义”：它们能评价单路线和支持原Reference逻辑；但已证明无法可靠排序完整POMO最终表现。若换正式验证，应先固定协议和频率，再据实记录新的Reference轨迹。置信区间晋升门槛也属于新规则，本轮未启用。

## 18. LKH-E46 / E59的外部测试与历史最好对比

{table(['模型','Real529','Real284','人工5000'],[
['G200','332.420655','357.318427','17.282643'],
*[ [name,n(score(run,ep,'real529')),n(score(run,ep,'real529_J20_100')),n(score(run,ep,'synthetic5000'))]
for name,run,ep in [('S500-E100','S500',100),('S500-E365','S500',365),('LKH-E46','LKH100',46),('LKH-E59','LKH100',59)] ],
])}

{table(['LKH模型/集合','相对G200的绝对差','gap','胜/平/负'],[
[f'E{ep} / {ds}',n(ext('LKH100',ep,ds)['model_minus_baseline']),n(ext('LKH100',ep,ds)['gap_percent'],3)+'%',f"{ext('LKH100',ep,ds)['wins']}/{ext('LKH100',ep,ds)['ties']}/{ext('LKH100',ep,ds)['losses']}"]
for ep in [46,59] for ds in ['real529','synthetic5000']])}

E46相对E59：Real529改善0.598532，人工5000改善0.042657；相对旧S500-E100却分别差0.979108、0.152394。两套已完成LKH测试全部实例可行率100%。这里“可行率100%”不等于全局最优率100%。

**结论：**新的选模方式找到了比E59更好的外部模型；新训练本身当前仍退步。不能只归咎于在线验证选错模型，也不能预言剩余训练必然或必然不能反超。

## 19. 每轮1万→2万的隐藏变化：更新次数与学习率

### 19.1 同样100万新实例时的实际optimizer步数

{table(['模型位置','新实例数','RL更新','IL更新','总更新','学习率'],[
[f"{r['run']}-E{r['epoch']}",r['instances'],r['rl_steps'],r['il_steps'],r['total_steps'],f"{r['learning_rate']:.8g}"]
for r in step_rows if (r['run'],r['epoch']) in [('S500',100),('LKH100',50)]])}

在相同100万实例处，本轮IL更新次数比旧版少约47.75%，而RL更新数基本相同。原因是IL先按J分组，每个逻辑batch最多1000例；每轮数据更多时，很多原来不足1000的分组变大，但batch数量不会按实例数线性翻倍。每轮仍是5遍，并不代表每个实例对应相同次数的optimizer更新；较大batch会改变梯度估计，不能简单断言更新少就一定更差。

本轮按100轮余弦到1e-5，旧版按500轮到1e-7。终点1e-5虽然更高，但同样100万实例时，本轮已经走过50/100、旧版仅100/500，实际本轮LR反而更低。**终点LR高不等于训练前半段LR高。**

### 19.2 长RL阶段与IL阶段对完整POMO的影响

{table(['组/轮次/模式','RL后','IL后','IL−RL','95%区间'],[
*[ [f'NM20-E20 / full POMO',n(near['groups']['nm20']['rl_to_il_transitions']['epoch20:full_pomo_greedy']['after_rl_mean']),n(near['groups']['nm20']['rl_to_il_transitions']['epoch20:full_pomo_greedy']['after_il_mean']),n(near['groups']['nm20']['rl_to_il_transitions']['epoch20:full_pomo_greedy']['il_minus_rl']['mean_right_minus_left']),ci(near['groups']['nm20']['rl_to_il_transitions']['epoch20:full_pomo_greedy']['il_minus_rl']['ci95'])] ],
*[ [f"LKH-E{r['epoch']} / {r['mode']}",n(r['after_rl']),n(r['after_il']),n(r['il_minus_rl']),ci(r['ci95'])] for r in lkh_stages ],
])}

LKH-E20完整POMO在这一阶段从17.818598升到17.887040，点估计变差0.068442；这里新增区间按相同实例10,000次配对bootstrap计算，区间是否跨零应以上表为准。只有50例和少量时点，不能据此认定IL整体有害，也不能将RL/IL参数净变化量直接当作两个损失在同一状态的梯度冲突。

训练每轮2万时，连续RL约313次后才进入IL；旧每轮1万时约157次。Reference检查和刷新训练标签的节奏也随epoch边界改变。这些都是实际训练逻辑的耦合，解释了为何“更多instance＋两个搜索改动”不是一个保证提升的简单叠加。

### 19.3 当前诊断最支持的判断

目前有直接证据说明：新训练的优化节奏、学习率轨迹、标签分布和搜索终点覆盖同时变化；有局部证据提示RL得到的完整POMO收益未必被IL保留。仍不能确定首要根因是IL强弱、标签结构、Reference反馈、LR，还是它们之间的组合。

### 19.4 梯度和阶段参数变化：日志中的量化结果

{table(['组/轮次','RL步数','RL裁剪前均值','RL触发裁剪','IL步数','IL裁剪前均值','IL触发裁剪'],[
[f"{r['group']}-E{r['epoch']}",r['rl_steps'],n(r['rl_preclip_mean'],3),pct(r['rl_clip_fraction'],1),r['il_steps'],n(r['il_preclip_mean'],3),pct(r['il_clip_fraction'],1)]
for r in learning_stats])}

此处“裁剪前均值”为每次主网络梯度L2范数的算术平均；阈值为1。RL和IL在不同参数状态、不同目标和不同batch上计算，范数大小不能直接解释为有效学习贡献的大小；频繁裁剪也不自动等于训练异常。

{table(['组/轮次','RL阶段参数净L2变化','IL阶段参数净L2变化'],[
[f"{r['group']}-E{r['epoch']}",n(r['rl_parameter_delta'],4),n(r['il_parameter_delta'],4)]for r in learning_stats])}

下表为E20诊断首batch的两个RL分项梯度，**不是RL与IL之间的夹角**：

{table(['组/轮次','密集RL梯度范数','精确辅助权重','加权辅助梯度范数','两个RL分项cosine'],[
[f"{r['group']}-E{r['epoch']}",n(r['rl_component_gradient_sample']['hybrid_gradient_norm'],4),n(r['rl_component_gradient_sample']['weight'],4),n(r['rl_component_gradient_sample']['weighted_exact_gradient_norm'],4),n(r['rl_component_gradient_sample']['cosine'],4)]
for r in learning_stats if r['group'] in ['NM20','LKH'] and r['epoch']==20])}

两个样本中cosine均为负，说明该batch两项梯度存在方向分歧；这不证明整轮持续冲突，更不证明把辅助权重降为零就会改善。诊断只覆盖少数预定batch，不能据此直接确定最佳RL权重。

### 19.5 当前LKH-E50的五遍IL

{table(['IL遍次','CE loss','token准确率','optimizer步数','裁剪前梯度均值'],[
[r['pass'],n(r['loss']),pct(r['accuracy'],3),r['optimizer_steps'],n(r['gradient_before']['mean'],4)] for r in lkh50_full['il_passes']])}

这些数据反映对同批标签的拟合程度，不包含每一遍之后的完整POMO解质量。因此即使CE继续下降、token准确率继续上升，也不能由此证明第5遍让最终路线更好；要判断需在同一固定开发集记录每遍后的精确路线质量。
""")

add("""
## 20. 各次实验到底证明了什么

"""+table(['问题','现有证据','可接受结论','不能据此宣称'],[
['固定验证持续下降是否代表泛化提升','S500 E100/E365外部配对结果反向','本项目旧单路线验证不能可靠代表完整POMO测试','固定集一定导致过拟合；所有验证都无意义'],
['换完整POMO选模是否有用','旧训练选E100；LKH选E46且两外部集优于E59','已观察到更符合目标的选模排序','已穷尽选出所有checkpoint全局最好'],
['上界排序是不是后期越来越错','E365 regret从0.6043降到0.4828','该诊断不支持误排序恶化假设','上界奖励完全没有问题'],
['降低IL遍数或更快衰减能否解决','E100继续20轮，均未超过E100','这两条短程分支无明确收益','从头IL2或其他LR一定失败'],
['swap和2-opt是否都有效','合用有两算子接受与上界贡献；完整POMO最好均值','swap-only不优先，保留2-opt或合用有依据','按接受数多就保留swap、删2-opt'],
['30move是否过多','大J新训练早期上限命中很高','不能只凭旧总体均值减到10','所有J和所有时期30都最优'],
['near-miss20%是不是最佳','改进Expert比例更高；完整POMO10%均值更好','精算配额是有效可调模块','20%已证明最终最佳'],
['质量加权IL是否反作用','B6轮固定验证多处更差，E5完整POMO略好','当前公式未展示稳定优势，实验未完成','标签质量不值得研究或所有加权都无效'],
['LKH是否有效','冻结100例Expert配对均值改善','有搜索阶段收益','网络训练和最终泛化已提升'],
['Replay满5万是否引起平台','正常容量达到上限；无容量消融','满池不是异常；需观察样本年龄与收益','满池一定坏，或一定无影响'],
['2万实例是否比1万无效','LR、IL更新与阶段频率同时变化','现有联合配置未复现旧最好','增加新数据本身一定有害'],
['是否已证明过拟合','验证/测试背离和阶段指标分化','存在泛化/评价目标问题，原因待定位','已锁定某一种统计过拟合机制'],
]) )

add("""
## 21. 建议的后续执行顺序（尚未执行）

这些是根据现有证据推导的待验证方案，不计作已经产生的实验结论。本报告没有启动新实验。

1. **先冻结评测目标。**完整POMO固定开发验证作为checkpoint筛选主指标；Reference是否同步改为同协议、评测频率和晋升门槛需预先定好。旧单路线值可保留为诊断。
2. **把横轴从epoch扩展到累计实例、RL/IL步数、墙钟与精算调用。**固定当前标准后，比较能看出算法效果，也能看出效率；不再用“都是5遍IL”代表相同优化强度。
3. **先做匹配优化节奏的搜索对照。**相同从头初始化、实例流、LR按实例/更新进度、RL→IL交替频率与IL逻辑batch；只改变swap＋2-opt与LKH。否则无法归因。
4. **观察阶段质量而非只看CE准确率。**在预先固定的较大开发样本上评价RL前/后、IL每遍后完整POMO，并给配对区间；同时记录路线不同终点数、Target候选覆盖、标签对当前Target的质量差。
5. **near-miss暂作候选参数，保留10%与20%的竞争。**若只选一个继续运行，应明确是成本/早期结果下的选择；冻结更完整证据前不声称20%最优。
6. **IL标签机制单独试验。**先检查当前Expert在IL入口是否仍优于Target、Replay标签年龄与收益，避免同时改标签权重、IL遍数、LR和搜索。B组本公式的提前停止只限制这一个方案。
7. **冻结配置后再报告外部测试。**Real529与人工5000已经被多次查看，后续研究中应说明其开发接触历史；准备好的新test500可作为一项补充，但目前尚无正式结果。一个seed的初筛不替代多seed验证。

## 22. 诊断日志现在记录了什么，还缺什么

### 22.1 已记录

- 每轮：LR、RL/IL/Edge loss、IL token accuracy、主优化器global_step、Reference晋升、Replay大小/新增、Gurobi次数、各阶段耗时和固定验证。
- 训练batch：J、数据指纹、near-miss名义配额/eligible/attempted/accepted、RL密集与精确辅助优势、实际辅助权重。
- LS消融每J/起点：搜索链总数、move均值/直方图/上限命中、10步后的上界收益、swap/2-opt接受数及上界贡献、选中Expert来源与精确改善。
- 诊断epoch：RL/IL每步裁剪前后梯度范数、阶段净参数位移；首个诊断batch的RL分项梯度强度和方向关系；固定样本RL前/后和IL后质量。
- B组特有：IL入口逐标签VT/VE、g/q、来源、重复份数、实际CE归一化贡献、额外精算与测量成本；初始进程缺Target比较路线。
- 当前LKH特有：每实例native调用/完成/超时、浮点验收、不同终点数、返回Anchor比例；每训练实例Target精算路线和Expert路线/值；每观测J每轮最多2例完整C0及输入输出复算样本；IL5遍分别的loss、准确率、更新步数和梯度摘要。

### 22.2 仍未覆盖或不能倒推

- 旧S500未保存的逐算子/逐链细项和梯度不能从旧汇总里重建。
- 搜索日志中的每一步上界改善不是每一步的真实连续优化改善；要获得后者需新增精算，会改变诊断成本。
- 完整POMO阶段诊断并非每epoch都运行；当前继承代码只在E5/E20保存这类50例阶段结果，不能把E50/E100配置中的诊断标记理解为所有模式都运行。
- 没有全量、同一参数状态和同一批数据下RL与IL梯度夹角；阶段净位移不能代替它。
- 少量详细复算样本不是所有训练实例的完整搜索轨迹；每实例汇总与全轨迹须区别。
- 没有多个训练seed的正式复现实验，不能把实例bootstrap当作训练方差置信区间。

## 23. 缺失数据与尚未落实事项

"""+table(['项目','实际状态'],[
['S500-E500的Real529/人工5000完整外部测试','待补充；训练结束不等于测试完成'],
['R200-E80独立选模best的完整外部测试','本次未找到，待补充'],
['R200-E11人工5000完整分数','未完成：2256/5000后按用户要求停止；不报告总体'],
['新test500正式评测','数据已准备，未找到完整结果，待补充'],
['质量IL-B20轮终点','未产生：只完成6轮，第7轮中止'],
['LKH100最终E100模型与外部测试','训练未结束，待产生'],
['运行中用完整POMO选模集替换单路线固定验证','尚未实施，当前配置仍为2000单路线'],
['验证每50轮换一批','仅讨论，未找到本轮实际启用证据'],
['数据增强训练消融','未找到已完成结果'],
['1万/2万实例、LR和RL/IL频率严格单变量对照','尚未完成；目前是发现耦合的日志证据'],
['完整硬件根因、MemTest86与BIOS操作结果','待补充；不得由段错误日志推定'],
['旧500轮全源码逐字节快照','未找到；参数、日志、checkpoint与部分测试快照保留'],
]) )

add(fr"""
## 附录A. 人工5000逐规模：历史最好、后期模型和当前选模最好

{table(['J','例数','G200','S500-E100','S500-E365','LKH-E46','LKH-E59'],[
[j,1000,n(ext('S500',100,'synthetic5000',j)['baseline_mean']),n(score('S500',100,'synthetic5000',j)),n(score('S500',365,'synthetic5000',j)),n(score('LKH100',46,'synthetic5000',j)),n(score('LKH100',59,'synthetic5000',j))]
for j in [20,40,60,80,100]])}

## 附录B. Real529全部19种规模

{table(['J','例数','G200','S500-E100','S500-E365','LKH-E46','LKH-E46 gap'],[
[j,ext('S500',100,'real529',j)['n'],n(ext('S500',100,'real529',j)['baseline_mean']),n(score('S500',100,'real529',j)),n(score('S500',365,'real529',j)),n(score('LKH100',46,'real529',j)),n(ext('LKH100',46,'real529',j)['gap_percent'],3)+'%']
for j in sorted({int(r['J']) for r in external if r['run']=='S500' and r['epoch']==100 and r['dataset']=='real529' and r['J']!='all'})])}

不能因为个别规模（例如J200）当前LKH优于G200，就省略总体和训练范围内的退步。总体是实例等权平均，不是各J均值再等权平均，因此Real529中实例数量多的J影响更大。

## 附录C. 更新次数的扩展核对表

{table(['训练','Epoch','新实例数','RL更新','IL更新','总更新','LR'],[
[r['run'],r['epoch'],r['instances'],r['rl_steps'],r['il_steps'],r['total_steps'],f"{r['learning_rate']:.8g}"]for r in step_rows])}

此处IL更新由`global_step−累计batches`计算，结合代码确认主优化器只在RL和IL更新；Edge有独立optimizer，不混入主步数。实际步数是观测值，其质量影响仍需控制实验。

## 附录D. 配套数据文件与复算方式

- `external_results_all_sizes.csv`：所有本报告纳入的外部测试总体、Real284和逐J结果；含训练代号，防止跨训练混淆Epoch。
- `checkpoint_selection_all_runs.csv`：S500、R200和LKH100在共同1000例上的已测结果。
- `optimizer_steps_equal_samples.csv`：相同累计实例及代表时点的RL/IL更新与LR。
- `S500_metrics_snapshot.csv`、`R200_metrics_snapshot.csv`、`LKH100_metrics_snapshot.csv`：本次读取的指标快照。
- `evidence_snapshot.json`：关键提取结果及本次新增的阶段配对区间。
- `source_manifest.json`：使用过的原始文件绝对路径、大小、读取时间及SHA256；运行中源文件以后可能改变，以报告快照为准。
- `build_review.py`：生成本报告、表格和图的脚本，只读训练来源、写本报告目录。重新执行将更新证据截点，不会启动或改变训练。

## 附录E. 原始证据索引
""")

source_index=[
("S01","S500参数与500行训练指标",f"{OLD}/params.json",f"{OLD}/metrics.csv"),
("S02","旧训练分阶段日志整理","outputs/e500_log_review_20261010/report.md","outputs/e500_log_review_20261010/summary.json"),
("S03","S500-E100 Real529与人工5000","outputs/eval_vectorized_epoch100_vs_gurobi_epoch200_real529/summary.json","outputs/eval_vectorized_epoch100_vs_gurobi_epoch200_synthetic5000/summary.json"),
("S04","S500-E365对E100外部配对比较","outputs/eval_vectorized_epoch365_external/report.md","outputs/eval_vectorized_epoch365_external/summary.json"),
("S05","旧训练完整POMO独立回测","outputs/pomo_selection_20260921/report.md","outputs/pomo_selection_20260921/summary.json"),
("S06","Target上界选路诊断","outputs/target_selection_diagnostic_20260922/report.md","outputs/target_selection_diagnostic_20260922/paired_effects.json"),
("S07","sample无增强外部测试","outputs/eval_sample_e200_vs_e365/report.md","outputs/eval_sample_e200_vs_e365/summary.json"),
("S08","E100短程IL/LR分支","outputs/epoch100_controlled_branches/report.md","outputs/epoch100_controlled_branches/summary.json"),
("S09","R200配置与最终指标",f"{R200}/params.json",f"{R200}/metrics.csv"),
("S10","R200-E176正式外部测试与E80选模","outputs/eval_formal_scratch_epoch176_external/summary.json","outputs/current_e200_pomo_selection_20260930/final_selection.json"),
("S11","会议后执行协议与E500逻辑恢复","docs/cvtsp_experiment_execution_20261012_v2.md","docs/e500_logic_e100_protocol.md"),
("S12","三组算子对照",f"{ABL}/report.md",f"{ABL}/completion_audit.json"),
("S13","三组配额对照",f"{NM}/report.md",f"{NM}/completion_audit.json"),
("S14","质量加权IL公式和实际指标","docs/quality_il_b_experiment_20261014.md",f"{QIL}/metrics.csv"),
("S15","LKH冻结搜索100例","outputs/lkh_search_pilot_20261014/report.md","outputs/lkh_search_pilot_20261014/completion_audit.json"),
("S16","LKH100协议与新初始化","docs/lkh_nm20_e100_training_20261014.md",f"{LKH}/initialization.json"),
("S17","LKH100完整POMO选模","outputs/lkh_nm20_checkpoint_selection_20261015/refined_selection.json",),
("S18","LKH-E46和E59完整外部测试","outputs/eval_lkh_nm20_epoch46_external_20261015/summary.json","outputs/eval_lkh_nm20_epoch59_external_20261015/summary.json"),
("S19","当前实际训练/IL实现","src/JointSelfImprovementTrainer.py","src/LKHTrainingTrainer.py"),
("S20","验证集合协议与独立性说明","validation/pomo_selection_20260921/manifest.json","validation/e500_logic_e100_20261010/validation2000/manifest.json"),
("S21","G200原训练参数","outputs/train__cvtsp_random_n100_nearest_quarter/params.json",),
("S22","B组中止与诊断日志范围","notes/2026-10-14-cvtsp-diagnostics-grill.md",f"{QIL}/live.log"),
("S23","硬件压力测试结果","outputs/host_training_stress_20260928/live.log","outputs/host_training_stress_20260923__failed_mce/live.log"),
("S24","会议记录原文","/home/marmot/.codex/attachments/8e1f33c5-d1c1-407d-9f13-2b4b8f462a22/咔咔咔咔的快速会议-元宝纪要.txt",),
]
for entry in source_index:
    ident,title,*paths=entry
    links=[]
    for path in paths:
        p=Path(path) if Path(path).is_absolute() else ROOT/path
        read(p)
        links.append(f"[{p.name}](<{p}>)")
    add(f"- **{ident} {title}**："+"；".join(links))

document="\n\n".join(sections)+"\n"
# Keep evidence citations close to the relevant section, in addition to the index.
section_sources = {
    3: "S01、S11、S19", 4: "S03、S20、S21", 5: "S01–S04",
    6: "S05", 7: "S06", 8: "S07", 9: "S08", 10: "S09、S10、S23",
    11: "S11、S24", 12: "S12", 13: "S13", 14: "S14、S22",
    15: "S15", 16: "S16、S19", 17: "S05、S10、S17", 18: "S03、S04、S18",
    19: "S01、S13、S16、S19", 22: "S12、S14、S19、S22",
}
for section, sources in section_sources.items():
    pattern = rf"(^## {section}\. [^\n]+\n)"
    document = re.sub(pattern, lambda m: m[1]+f"\n证据索引：{sources}（原文件见附录E）。\n", document, flags=re.M)
(OUT/f"{STEM}.md").write_text(document,encoding="utf-8")

evidence={
    "snapshot":SNAPSHOT,"purpose":"detailed_history_not_new_training_or_test",
    "run_latest":{name:{"last_completed_epoch":int(float(v['latest']['epoch'])),
                         "best_single_route_epoch":int(float(v['best']['epoch'])),
                         "best_single_route_mean":float(v['best']['target_mean_objective'])}
                  for name,v in runs.items()},
    "external_results":external,"selection":selection_rows,"optimizer_steps":step_rows,
    "lkh_stage_transitions":lkh_stages,"quality_il_stage_transitions":quality_stages,
    "learning_diagnostics":learning_stats,"lkh_epoch50_il_passes":lkh50_full['il_passes'],
    "audits":{"local_search":ab_audit,"near_miss":nm_audit,"lkh_pilot":pilot_audit},
    "lkh_initialization":init,"lkh_epoch50_search":lkh50,
}
(OUT/"evidence_snapshot.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding="utf-8")
(OUT/"source_manifest.json").write_text(json.dumps({"snapshot":SNAPSHOT,"sources":list(SOURCES.values())},ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"markdown":str(OUT/f'{STEM}.md'),"characters":len(document),"sections":len(sections),
                  "source_files":len(SOURCES),"external_rows":len(external),"snapshot":SNAPSHOT},ensure_ascii=False))
