"""Compare quota, effective Experts, learning quality, and cost without test selection."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
from itertools import combinations
from collections import Counter

from src.SearchDiagnostics import atomic_json
from scripts.summarize_search_ablation import (
    diagnostics_by_epoch, epoch_cycle_times, paired_values, stage_transitions,
)
from scripts.watch_search_ablation import completed_metric_rows

GROUPS = {"nm05": .05, "nm10": .10, "nm20": .20}
SIZES = (20, 40, 60, 80, 100)


def summarize(root):
    records, summaries, evidence = {}, {}, {}
    for group, rate in GROUPS.items():
        directory = root / group
        rows = completed_metric_rows(directory / "metrics.csv")
        cycles = epoch_cycle_times(directory / "live.log")
        records[group] = []
        for row in rows:
            epoch = int(float(row["epoch"]))
            accepted = float(row["near_miss_accept_count"])
            changed = round(20000 * float(row["expert_better_ratio"]))
            source_path = directory / "diagnostics" / f"epoch-{epoch:03d}_summary.json"
            if source_path.exists():
                source = json.loads(source_path.read_text())
                counts = Counter(x["source"] for batch in source["winner_chains"].values() for x in batch)
                changed = sum(n for name, n in counts.items() if name != "anchor")
                certified = counts["certified_upper_bound"]
            else:
                certified = changed - round(accepted)
            records[group].append({
                "group": group, "epoch": epoch, "nominal_rate": rate,
                "fixed_validation": float(row["target_mean_objective"]),
                **{f"J{j}": float(row[f"target_J{j}"]) for j in SIZES},
                "near_miss_eligible": float(row["near_miss_eligible_count"]),
                "near_miss_attempted": float(row["near_miss_attempt_count"]),
                "near_miss_accepted": accepted,
                "actual_attempt_fraction": float(row["near_miss_attempt_count"]) / 20000,
                "near_miss_accept_rate": float(row["near_miss_accept_rate"]),
                "changed_expert_count": changed, "certified_expert_count": certified,
                "changed_expert_fraction": float(row["expert_better_ratio"]),
                **{key: float(row[key]) for key in (
                    "expert_improvement_percent", "anchor_exact_makespan", "expert_exact_makespan",
                    "target_exact_reward_makespan", "exact_rl_target_better_ratio", "imitation_token_accuracy",
                    "rl_loss", "imitation_loss", "learning_rate", "max_local_moves", "expert_replay_effective_fraction",
                    "gurobi_calls_per_instance", "validation_runtime_seconds", "diagnostic_wall_seconds")},
                "full_epoch_cycle_wall_seconds": cycles.get(epoch),
                "train_data_sha256": row["train_data_sha256"],
            })
        stages = {}
        for path in sorted((directory / "diagnostics").glob("epoch-*_greedy.json")):
            value = json.loads(path.read_text())
            stages[path.name] = {k: value[k] for k in ("epoch", "stage", "mode", "mean", "by_J", "gurobi_calls")}
        best = min(records[group], key=lambda r: r["fixed_validation"]) if records[group] else None
        summaries[group] = {"completed_epochs": len(rows), "best": best,
                            "epoch20": next((r for r in records[group] if r["epoch"] == 20), None),
                            "stages": stages, "rl_to_il_transitions": stage_transitions(directory),
                            "diagnostics_by_epoch": diagnostics_by_epoch(directory)}
        log = (directory / "live.log").read_text() if (directory / "live.log").exists() else ""
        evidence[group] = {"epochs_1_to_20": [r["epoch"] for r in records[group]] == list(range(1, 21)),
                           "checkpoint20": (directory / "checkpoint-20.pt").is_file(),
                           "last_checkpoint": (directory / "last.pt").is_file(),
                           "training_summary": (directory / "train_summary.json").is_file(),
                           "exit_zero": "TRAIN_EXIT=0 " in log}
    common = sorted(set.intersection(*(set(r["epoch"] for r in rows) for rows in records.values())))
    for epoch in common:
        hashes = [next(r["train_data_sha256"] for r in rows if r["epoch"] == epoch) for rows in records.values()]
        if not all(hashes) or len(set(hashes)) != 1:
            raise RuntimeError(f"training instance streams differ at epoch {epoch}")
    paired = {}
    for left, right in combinations(GROUPS, 2):
        common_files = set(summaries[left]["stages"]) & set(summaries[right]["stages"])
        for name in sorted(common_files):
            a = json.loads((root / left / "diagnostics" / name).read_text())
            b = json.loads((root / right / "diagnostics" / name).read_text())
            if [x["fingerprint"] for x in a["records"]] != [x["fingerprint"] for x in b["records"]]:
                raise RuntimeError("paired diagnostic data mismatch")
            paired[f"{left}_vs_{right}:{name}"] = {
                "epoch": a["epoch"], "stage": a["stage"], "mode": a["mode"],
                "all": paired_values([x["makespan"] for x in a["records"]], [x["makespan"] for x in b["records"]]),
                "by_J": {str(j): paired_values([x["makespan"] for x in a["records"] if x["J"] == j],
                                              [x["makespan"] for x in b["records"] if x["J"] == j]) for j in SIZES}}
    complete = all(all(e.values()) for e in evidence.values())
    atomic_json(root / "comparison.json", {"complete": complete, "groups": summaries, "epoch_rows": records,
                                           "common_data_epochs_verified": common, "completion_evidence": evidence,
                                           "paired_development_evaluations": paired,
                                           "ci_scope": "paired instance bootstrap; not uncertainty across training seeds"})
    all_rows = [r for rows in records.values() for r in rows]
    if all_rows:
        with (root / "comparison_epochs.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader(); writer.writerows(all_rows)
    lines = ["# Near-miss固定配额从头训练对照", "",
             f"状态：{'三组20轮完成' if complete else '运行中/未全部完成'}；共同核验数据的轮次：{common}。", "",
             "标称配额5%/10%/20%；保持swap＋2-opt、Top-K 30、质量门槛、固定30步、2万实例/轮和100轮LR调度。标准64例batch按ceil规则上限4/7/13个，实际比例另报。", "",
             "| 组 | 完成轮数 | 验证best epoch | 验证best | E20固定验证 | E20完整POMO独立50例 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for group, s in summaries.items():
        best, last = s["best"], s["epoch20"]
        full = s["stages"].get("epoch-020_after_il_full_pomo_greedy.json")
        lines.append(f"| {group} | {s['completed_epochs']} | {best['epoch'] if best else '待完成'} | "
                     f"{best['fixed_validation'] if best else '待完成'} | {last['fixed_validation'] if last else '待完成'} | "
                     f"{full['mean'] if full else '待完成'} |")
    lines += ["", "| 组 | E20实际精算比例 | near-miss精算成功率 | 总改进Expert比例 | Expert真实改善/Anchor | calls/实例 | 每轮平均墙钟/分钟 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for group, s in summaries.items():
        last = s["epoch20"]
        if not last:
            lines.append(f"| {group} | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |")
            continue
        times = [r["full_epoch_cycle_wall_seconds"] for r in records[group] if r["full_epoch_cycle_wall_seconds"] is not None]
        lines.append(f"| {group} | {last['actual_attempt_fraction']:.2%} | {last['near_miss_accept_rate']:.2%} | "
                     f"{last['changed_expert_fraction']:.2%} | {last['expert_improvement_percent']:.4f}% | "
                     f"{last['gurobi_calls_per_instance']:.4f} | {sum(times)/len(times)/60:.2f} |")
    lines += ["", "逐J结果、全部RL/IL阶段质量、梯度与搜索统计、配对95%区间见comparison.json。",
              "Expert比例并非学习效果：同时比较验证/独立POMO质量与成本；一个seed和20轮仅作初筛。正式测试500/Real529未用于选参数。",
              "置信区间只反映独立诊断实例的配对差异，不包含训练seed差异。", "",
              "曲线：fixed_validation_by_size.png、diagnostic_free_greedy_by_size.png、diagnostic_full_pomo_greedy_by_size.png、expert_quota_cost.png。"]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    if all_rows:
        plot(root, records, summaries)
    print("NEAR_MISS_COMPLETE" if complete else "NEAR_MISS_IN_PROGRESS", flush=True)


def plot(root, records, summaries):
    os.environ.setdefault("MPLCONFIGDIR", str((root / ".matplotlib_cache").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(zip(GROUPS, ("C0", "C1", "C2")))
    for mode in ("fixed_validation", "free_greedy", "full_pomo_greedy"):
        if mode != "fixed_validation" and not any(r["mode"] == mode for s in summaries.values() for r in s["stages"].values()):
            continue
        figure, axes = plt.subplots(2, 3, figsize=(12, 7))
        for axis, size in zip(axes.flat, (None, *SIZES)):
            for group, rate in GROUPS.items():
                if mode == "fixed_validation":
                    rows = records[group]
                    if rows:
                        axis.plot([r["epoch"] for r in rows], [r["fixed_validation" if size is None else f"J{size}"] for r in rows],
                                  color=colors[group], marker="o", markersize=3, label=f"Nominal {rate:.0%}")
                else:
                    for stage, style in (("after_rl", "--"), ("after_il", "-")):
                        rows = sorted((r for r in summaries[group]["stages"].values() if r["mode"] == mode and r["stage"] == stage), key=lambda r: r["epoch"])
                        if rows:
                            axis.plot([r["epoch"] for r in rows], [r["mean"] if size is None else r["by_J"][str(size)]["mean"] for r in rows],
                                      color=colors[group], linestyle=style, marker="o", markersize=3, label=f"{rate:.0%} {stage}")
            axis.set(title="All sizes" if size is None else f"J = {size}", xlabel="Epoch", ylabel="Makespan")
            axis.grid(alpha=.25)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if handles: axes.flat[0].legend(fontsize=7)
        figure.suptitle(f"Near-miss quota: {mode}"); figure.tight_layout()
        name = "fixed_validation_by_size" if mode == "fixed_validation" else f"diagnostic_{mode}_by_size"
        for extension in ("png", "pdf"): figure.savefig(root / f"{name}.{extension}", dpi=180)
        plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    for axis, (key, label) in zip(axes.flat, (("actual_attempt_fraction", "Actual near-miss fraction"),
                                          ("changed_expert_fraction", "Improved Expert fraction"),
                                          ("gurobi_calls_per_instance", "Training Gurobi calls / instance"),
                                          ("full_epoch_cycle_wall_seconds", "Full epoch wall time (minutes)"))):
        for group, rows in records.items():
            valid = [r for r in rows if r[key] is not None]
            if valid: axis.plot([r["epoch"] for r in valid], [r[key]/60 if key.endswith("seconds") else r[key] for r in valid],
                                color=colors[group], marker="o", markersize=3, label=f"Nominal {GROUPS[group]:.0%}")
        axis.set(xlabel="Epoch", ylabel=label); axis.grid(alpha=.25)
    axes.flat[0].legend(); figure.tight_layout()
    for extension in ("png", "pdf"): figure.savefig(root / f"expert_quota_cost.{extension}", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/near_miss_quota_20261013"))
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / ".summary.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        summarize(args.root)


if __name__ == "__main__": main()
