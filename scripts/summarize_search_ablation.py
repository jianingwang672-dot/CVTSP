"""Live/completed ablation summaries; no test-based checkpoint selection."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
import re
from pathlib import Path
from collections import Counter
from itertools import combinations

import numpy as np

from src.SearchDiagnostics import atomic_json, finite_summary


def load(path):
    return json.loads(path.read_text())


def epoch_cycle_times(path):
    """Wall time through validation and checkpoint saving, from trainer logs."""
    if not path.exists():
        return {}
    previous, times = None, {}
    with path.open() as handle:
        for line in handle:
            if "Generator stats:" not in line and not re.search(r"Epoch\s+\d+/\s*20:", line):
                continue
            timestamp = datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S.%f")
            if "Generator stats:" in line:
                previous = timestamp
                continue
            epoch = int(re.search(r"Epoch\s+(\d+)/\s*20:", line).group(1))
            if previous is not None:
                times[epoch] = (timestamp - previous).total_seconds()
            previous = timestamp
    return times


def search_by_size(search):
    """Combine quality/diversity chains without averaging their quantiles."""
    groups = {}
    for key, record in search.items():
        size = int(key.split("_")[0][1:])
        combined = groups.setdefault(size, {"chains": 0, "histogram": Counter(),
                                           "cap_hits": 0., "gain_after10": 0.,
                                           "accepted": Counter(), "upper_bound_gain": Counter()})
        combined["chains"] += record["chains"]
        combined["histogram"].update({int(k): count for k, count in record["histogram"].items()})
        combined["cap_hits"] += record["cap_hit_rate"] * record["chains"]
        combined["gain_after10"] += record["gain_after_move10"]
        combined["accepted"].update(record["accepted"])
        combined["upper_bound_gain"].update(record["upper_bound_gain"])
    for combined in groups.values():
        n, histogram = combined["chains"], combined["histogram"]
        combined["mean_steps"] = sum(steps * count for steps, count in histogram.items()) / n
        cumulative = 0
        for steps, count in sorted(histogram.items()):
            cumulative += count
            if cumulative >= .9 * n:
                combined["p90_steps"] = steps
                break
        combined["cap_hit_rate"] = combined["cap_hits"] / n
        combined["above10_rate"] = sum(count for steps, count in histogram.items() if steps > 10) / n
        total_gain = sum(combined["upper_bound_gain"].values())
        combined["gain_after10_fraction"] = combined["gain_after10"] / total_gain if total_gain else 0.
    return groups


def paired_values(left, right):
    """Right minus left; uncertainty over these fixed development instances."""
    differences = np.asarray(right, dtype=float) - np.asarray(left, dtype=float)
    rng = np.random.default_rng(20261012)
    means = rng.choice(differences, size=(2000, len(differences)), replace=True).mean(axis=1)
    return {"count": len(differences), "mean_right_minus_left": float(differences.mean()),
            "ci95": np.quantile(means, [.025, .975]).tolist(),
            "right_wins": int(np.sum(differences < -1e-6)),
            "ties": int(np.sum(np.abs(differences) <= 1e-6)),
            "right_losses": int(np.sum(differences > 1e-6))}


def diagnostics_by_epoch(directory):
    out = {}
    for path in sorted((directory / "diagnostics").glob("epoch-*_summary.json")):
        record = load(path)
        phases = {phase: {field: finite_summary([r[field] for r in record["gradients"] if r["phase"] == phase])
                          for field in ("before", "after")}
                  for phase in ("rl", "il")}
        winners = {}
        for size, rows in record["winner_chains"].items():
            winners[size] = {"source_counts": dict(Counter(r["source"] for r in rows)),
                             "winner_steps": finite_summary([r["steps"] for r in rows if r["steps"] is not None]),
                             "exact_gain": finite_summary([r["exact_gain"] for r in rows])}
        out[str(record["epoch"])] = {"search": record["search"], "gradients": phases,
                                     "winner_chains": winners,
                                     "rl_parameter_delta_l2": record["rl_parameter_delta_l2"],
                                     "il_parameter_delta_l2": record["il_parameter_delta_l2"],
                                     "timings": record["timings"]}
    return out


def compare_stage_records(root):
    out = {}
    for left, right in combinations(("swap", "two_opt", "both"), 2):
        left_dir, right_dir = root / left / "diagnostics", root / right / "diagnostics"
        common = {p.name for p in left_dir.glob("epoch-*_greedy.json")} & {p.name for p in right_dir.glob("epoch-*_greedy.json")}
        for name in sorted(common):
            a, b = load(left_dir / name), load(right_dir / name)
            if [r["fingerprint"] for r in a["records"]] != [r["fingerprint"] for r in b["records"]]:
                raise RuntimeError(f"stage datasets differ: {left}/{right}/{name}")
            key = f"{left}_vs_{right}:{name}"
            out[key] = {"left": left, "right": right, "epoch": a["epoch"], "stage": a["stage"], "mode": a["mode"],
                        "all": paired_values([r["makespan"] for r in a["records"]], [r["makespan"] for r in b["records"]]),
                        "by_J": {str(size): paired_values([r["makespan"] for r in a["records"] if r["J"] == size],
                                                          [r["makespan"] for r in b["records"] if r["J"] == size])
                                 for size in sorted({r["J"] for r in a["records"]})}}
    return out


def stage_transitions(directory):
    out = {}
    paths = list((directory / "diagnostics").glob("epoch-*_after_il_*_greedy.json"))
    for path in sorted(paths):
        after = load(path)
        before_path = path.with_name(path.name.replace("_after_il_", "_after_rl_"))
        if not before_path.exists():
            continue
        before = load(before_path)
        if [r["fingerprint"] for r in before["records"]] != [r["fingerprint"] for r in after["records"]]:
            raise RuntimeError("RL/IL diagnostic instance mismatch")
        out[f"epoch{after['epoch']}:{after['mode']}"] = {
            "epoch": after["epoch"], "mode": after["mode"],
            "after_rl_mean": before["mean"], "after_il_mean": after["mean"],
            "il_minus_rl": paired_values([r["makespan"] for r in before["records"]],
                                         [r["makespan"] for r in after["records"]]),
            "by_J": {str(j): paired_values([r["makespan"] for r in before["records"] if r["J"] == j],
                                           [r["makespan"] for r in after["records"] if r["J"] == j])
                     for j in sorted({r["J"] for r in after["records"]})}}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/search_ablation_20261012"))
    args = parser.parse_args()
    import fcntl
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / ".summary.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        summarize(args.root)


def summarize(root):
    args = argparse.Namespace(root=root)
    groups, summaries = {}, {}
    for group in ("swap", "two_opt", "both"):
        directory = args.root / group
        path = directory / "metrics.csv"
        rows = list(csv.DictReader(path.open())) if path.exists() else []
        cycle_times = epoch_cycle_times(directory / "live.log")
        epoch_rows = []
        for row in rows:
            epoch = int(float(row["epoch"]))
            summary_path = directory / "diagnostics" / f"epoch-{epoch:03d}_summary.json"
            diagnostics = load(summary_path) if summary_path.exists() else {}
            epoch_rows.append({"group": group, "epoch": epoch,
                               "fixed_validation": float(row["target_mean_objective"]),
                               **{key: float(row[key]) for key in ("expert_improvement_percent", "expert_better_ratio",
                                                                  "imitation_token_accuracy", "max_local_moves", "learning_rate",
                                                                  "anchor_exact_makespan", "expert_exact_makespan",
                                                                  "target_exact_reward_makespan", "gurobi_calls_per_instance",
                                                                  "imitation_runtime_seconds", "edge_update_runtime_seconds",
                                                                  "local_search_runtime_seconds", "sparse_phase_wall_runtime_seconds")},
                               "train_data_sha256": row.get("train_data_sha256"),
                               "wall_seconds_including_diagnostics": diagnostics.get("wall_seconds_including_diagnostics"),
                               "full_epoch_cycle_wall_seconds": cycle_times.get(epoch),
                               "validation_seconds": float(row["validation_runtime_seconds"]),
                               "gpu_wait_seconds": diagnostics.get("timings", {}).get("gpu_wait"),
                               **{f"J{j}": float(row[f"target_J{j}"]) for j in (20, 40, 60, 80, 100)}})
        groups[group] = epoch_rows
        best = min(epoch_rows, key=lambda x: x["fixed_validation"]) if epoch_rows else None
        stage_rows = {}
        for path in sorted((directory / "diagnostics").glob("epoch-*_greedy.json")):
            record = load(path)
            stage_rows[path.stem] = {k: record[k] for k in ("epoch", "stage", "mode", "mean", "by_J", "gurobi_calls")}
        ranking = {}
        for path in sorted((directory / "diagnostics").glob("epoch-*_reward_ranking.json")):
            record = load(path)
            ranking[str(record["epoch"])] = record["by_J"]
        summary_files = sorted((directory / "diagnostics").glob("epoch-*_summary.json"))
        summaries[group] = {"completed_epochs": len(epoch_rows), "best": best,
                            "epoch20": next((r for r in epoch_rows if r["epoch"] == 20), None),
                            "stages": stage_rows, "reward_ranking": ranking,
                            "rl_to_il_transitions": stage_transitions(directory),
                            "diagnostics_by_epoch": diagnostics_by_epoch(directory),
                            "latest_search": load(summary_files[-1])["search"] if summary_files else {}}
    common = sorted(set.intersection(*(set(row["epoch"] for row in rows) for rows in groups.values())))
    for epoch in common:
        hashes = [next(row for row in rows if row["epoch"] == epoch)["train_data_sha256"] for rows in groups.values()]
        if not all(hashes) or len(set(hashes)) != 1:
            raise RuntimeError(f"training instance sequence differs across groups at epoch {epoch}")
    completion_evidence = {}
    for group, rows in groups.items():
        directory = args.root / group
        log = (directory / "live.log").read_text() if (directory / "live.log").exists() else ""
        completion_evidence[group] = {
            "epochs_1_to_20": [r["epoch"] for r in rows] == list(range(1, 21)),
            "checkpoint20": (directory / "checkpoint-20.pt").is_file(),
            "last_checkpoint": (directory / "last.pt").is_file(),
            "training_summary": (directory / "train_summary.json").is_file(),
            "exit_zero": "TRAIN_EXIT=0 " in log,
        }
    complete = all(all(evidence.values()) for evidence in completion_evidence.values())
    atomic_json(args.root / "comparison.json", {"complete": complete, "common_data_epochs_verified": common,
                                              "groups": summaries, "epoch_rows": groups,
                                              "completion_evidence": completion_evidence,
                                              "paired_development_evaluations": compare_stage_records(args.root),
                                              "ci_scope": "paired instance bootstrap; descriptive, not uncertainty across training seeds"})
    path = args.root / "comparison_epochs.csv"
    all_rows = [row for rows in groups.values() for row in rows]
    if all_rows:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    lines = ["# 三组局部搜索实验", "", f"完成状态：{'三组20轮均完成' if complete else '运行中/未全部完成'}。",
             "", "| 组别 | 完成轮数 | 验证best epoch | 验证best | Epoch20 |", "|---|---:|---:|---:|---:|"]
    for group, summary in summaries.items():
        best, last = summary["best"], summary["epoch20"]
        lines.append(f"| {group} | {summary['completed_epochs']} | {best['epoch'] if best else '待补充'} | "
                     f"{best['fixed_validation'] if best else '待补充'} | {last['fixed_validation'] if last else '待补充'} |")
    lines += ["", "## 固定验证：各组best与Epoch20", "",
              "| 组别 | 检查点 | Epoch | 总体 | J20 | J40 | J60 | J80 | J100 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group, summary in summaries.items():
        for label, row in (("验证best", summary["best"]), ("Epoch20", summary["epoch20"])):
            if row is None:
                lines.append(f"| {group} | {label} | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |")
            else:
                values = " | ".join(f"{row[key]:.6f}" for key in ("fixed_validation", "J20", "J40", "J60", "J80", "J100"))
                lines.append(f"| {group} | {label} | {row['epoch']} | {values} |")
    lines += ["", "## Epoch20网络学习诊断", "",
              "IL后减RL后：正值表示IL后makespan升高。完整POMO与单greedy使用各自预先固定的样本，不能直接横向比较两种均值。", "",
              "| 组别 | 解码/样本数 | RL后 | IL后 | IL后减RL后 | 配对95%区间 |",
              "|---|---|---:|---:|---:|---|"]
    for group, summary in summaries.items():
        for mode, count in (("free_greedy", 100), ("full_pomo_greedy", 50)):
            transition = summary["rl_to_il_transitions"].get(f"epoch20:{mode}")
            if transition is None:
                lines.append(f"| {group} | {mode}/{count} | 待完成 | 待完成 | 待完成 | 待完成 |")
            else:
                delta = transition["il_minus_rl"]
                lo, hi = delta["ci95"]
                lines.append(f"| {group} | {mode}/{count} | {transition['after_rl_mean']:.6f} | "
                             f"{transition['after_il_mean']:.6f} | {delta['mean_right_minus_left']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines += ["", "## Epoch20局部搜索诊断", "",
              "下表合并质量/多样性起点，列出训练中的五个代表规模；实际出现的20–100规模和起点类别的完整统计见comparison.json。P90采用合并直方图的90%累计分位，第10步后贡献是上界下降的占比，非精确makespan贡献。", "",
              "| 组别 | J | 链数 | 平均move | P90 move | 上限命中 | 超过10步 | 第10步后上界下降占比 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group, summary in summaries.items():
        record = summary["diagnostics_by_epoch"].get("20")
        if record is None:
            lines.append(f"| {group} | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |")
            continue
        combined = search_by_size(record["search"])
        for size in (20, 40, 60, 80, 100):
            statistics = combined.get(size)
            if statistics is None:
                lines.append(f"| {group} | {size} | 本轮无记录 | — | — | — | — | — |")
                continue
            lines.append(f"| {group} | {size} | {statistics['chains']} | {statistics['mean_steps']:.2f} | "
                         f"{statistics['p90_steps']} | {statistics['cap_hit_rate']:.1%} | "
                         f"{statistics['above10_rate']:.1%} | {statistics['gain_after10_fraction']:.1%} |")
    lines += ["", "| 组别 | swap接受数 | 2-opt接受数 | swap上界下降占比 | 2-opt上界下降占比 |",
              "|---|---:|---:|---:|---:|"]
    for group, summary in summaries.items():
        record = summary["diagnostics_by_epoch"].get("20")
        if record is None:
            lines.append(f"| {group} | 待完成 | 待完成 | 待完成 | 待完成 |")
            continue
        accepted, gains = Counter(), Counter()
        for statistics in record["search"].values():
            accepted.update(statistics["accepted"])
            gains.update(statistics["upper_bound_gain"])
        total = sum(gains.values())
        enabled = {"swap": ("swap",), "two_opt": ("two_opt",), "both": ("swap", "two_opt")}[group]
        values = [str(accepted[operator]) if operator in enabled else "未启用" for operator in ("swap", "two_opt")]
        shares = [f"{gains[operator]/total:.1%}" if total and operator in enabled else "—"
                  for operator in ("swap", "two_opt")]
        lines.append(f"| {group} | {' | '.join(values + shares)} |")
    lines += ["", "## 实际训练成本", "",
              "本机RTX4090 24GB、32 CPU线程，每组8 worker、合计24；大显存阶段最多两组并发。下表统计已结束轮次，训练calls不包含额外诊断和固定验证。", "",
              "| 组别 | 已记录完整轮次 | 每轮墙钟均值/分钟 | 最短/分钟 | 最长/分钟 | 训练Gurobi calls/实例均值 |",
              "|---|---:|---:|---:|---:|---:|"]
    for group, rows in groups.items():
        seconds = [row["full_epoch_cycle_wall_seconds"] for row in rows if row["full_epoch_cycle_wall_seconds"] is not None]
        if seconds:
            calls = sum(row["gurobi_calls_per_instance"] for row in rows) / len(rows)
            lines.append(f"| {group} | {len(seconds)} | {np.mean(seconds)/60:.2f} | "
                         f"{min(seconds)/60:.2f} | {max(seconds)/60:.2f} | {calls:.4f} |")
    lines += ["", f"已核对三组训练数据一致的epoch：{common}。",
              "", "本实验比较同一平滑move预算下的算子，不能单独证明平滑预算优于固定30；20轮和一个seed仅供初筛。",
              "", "算子定义沿用原实现：2-opt-only包含相邻反转；合用组跳过与swap重复的相邻反转。候选数量因此不是两组简单相加。",
              "", "开发诊断的95%区间来自实例配对bootstrap，不包含不同训练seed的不确定性；正式测试未参与选择。",
              "", "阶段质量、奖励排序与逐J搜索统计见comparison.json；正式测试未用于选参数。",
              "", "耗时口径：full_epoch_cycle_wall_seconds来自训练开始/上一轮结束至本轮结束的日志时间差，包含诊断、固定验证、资源等待和checkpoint保存；worker累计时间不等于墙钟时间。",
              "", "图表与明细：",
              "", "- [固定验证总体曲线](fixed_validation_curve.png)",
              "- [固定验证逐规模曲线](fixed_validation_by_size.png)",
              "- [独立诊断：单greedy，RL后/IL后](diagnostic_free_greedy_by_size.png)",
              "- [独立诊断：完整POMO，RL后/IL后](diagnostic_full_pomo_greedy_by_size.png)（产生对应诊断后生成）",
              "- [每轮指标](comparison_epochs.csv)；[诊断阶段逐规模指标](diagnostic_stages.csv)"]
    (args.root / "report.md").write_text("\n".join(lines) + "\n")
    stage_rows = []
    for group, summary in summaries.items():
        for record in summary["stages"].values():
            for size, statistics in record["by_J"].items():
                stage_rows.append({"group": group, "epoch": record["epoch"],
                                   "stage": record["stage"], "mode": record["mode"],
                                   "J": int(size), "count": statistics["count"],
                                   "mean_makespan": statistics["mean"]})
    if stage_rows:
        with (args.root / "diagnostic_stages.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(stage_rows[0]))
            writer.writeheader()
            writer.writerows(stage_rows)
    if all_rows:
        os.environ.setdefault("MPLCONFIGDIR", str((args.root / ".matplotlib_cache").resolve()))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(8, 4.5))
        labels = {"swap": "Swap", "two_opt": "2-opt", "both": "Swap + 2-opt"}
        for group, rows in groups.items():
            if rows:
                axis.plot([r["epoch"] for r in rows], [r["fixed_validation"] for r in rows],
                          marker="o", markersize=3, label=labels[group])
        axis.set(xlabel="Epoch", ylabel="Fixed validation makespan", title="Search operator ablation")
        axis.grid(alpha=.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(args.root / "fixed_validation_curve.png", dpi=180)
        figure.savefig(args.root / "fixed_validation_curve.pdf")
        plt.close(figure)
        figure, axes = plt.subplots(2, 3, figsize=(12, 7))
        for axis, size in zip(axes.flat, (None, 20, 40, 60, 80, 100)):
            for group, rows in groups.items():
                if rows:
                    key = "fixed_validation" if size is None else f"J{size}"
                    axis.plot([r["epoch"] for r in rows], [r[key] for r in rows],
                              marker="o", markersize=3, label=labels[group])
            axis.set(title="All sizes" if size is None else f"J = {size}",
                     xlabel="Epoch", ylabel="Makespan")
            axis.grid(alpha=.25)
        axes.flat[0].legend(fontsize=8)
        figure.suptitle("Fixed validation: overall and by problem size")
        figure.tight_layout()
        figure.savefig(args.root / "fixed_validation_by_size.png", dpi=180)
        figure.savefig(args.root / "fixed_validation_by_size.pdf")
        plt.close(figure)
        colors = {"swap": "C0", "two_opt": "C1", "both": "C2"}
        for mode in ("free_greedy", "full_pomo_greedy"):
            if not any(r["mode"] == mode and r["stage"] == "after_il" for r in stage_rows):
                continue
            figure, axes = plt.subplots(2, 3, figsize=(12, 7))
            for axis, size in zip(axes.flat, (None, 20, 40, 60, 80, 100)):
                for group, summary in summaries.items():
                    for stage, style in (("after_rl", "--"), ("after_il", "-")):
                        records = sorted((r for r in summary["stages"].values()
                                          if r["mode"] == mode and r["stage"] == stage),
                                         key=lambda r: r["epoch"])
                        if records:
                            values = [r["mean"] if size is None else r["by_J"][str(size)]["mean"]
                                      for r in records]
                            axis.plot([r["epoch"] for r in records], values, linestyle=style,
                                      color=colors[group], marker="o", markersize=3,
                                      label=f"{labels[group]} {stage.replace('_', ' ')}")
                axis.set(title="All sizes" if size is None else f"J = {size}",
                         xlabel="Epoch", ylabel="Makespan")
                axis.grid(alpha=.25)
            axes.flat[0].legend(fontsize=7)
            figure.suptitle(f"Independent diagnostics: {mode.replace('_', ' ')}")
            figure.tight_layout()
            figure.savefig(args.root / f"diagnostic_{mode}_by_size.png", dpi=180)
            figure.savefig(args.root / f"diagnostic_{mode}_by_size.pdf")
            plt.close(figure)
    print("ABLATION_COMPLETE" if complete else "ABLATION_IN_PROGRESS", flush=True)


if __name__ == "__main__":
    main()
