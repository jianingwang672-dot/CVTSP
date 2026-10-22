"""Summarize paired one-sample POMO E200/E365 external benchmarks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/eval_sample_e200_vs_e365"


def paired_interval(rows: list[dict], rng: np.random.Generator, samples: int = 5000):
    # Preserve the number of instances at each J in every bootstrap replicate.
    groups = []
    for size in sorted({row["J"] for row in rows}):
        groups.append(np.asarray([
            row["objectives"]["self_e365"] - row["objectives"]["gurobi_e200"]
            for row in rows if row["J"] == size
        ], dtype=float))
    boot = np.empty(samples, dtype=float)
    for start in range(0, samples, 250):
        stop = min(start + 250, samples)
        subtotal = np.zeros(stop - start, dtype=float)
        for delta in groups:
            indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
            subtotal += delta[indices].sum(axis=1)
        boot[start:stop] = subtotal / len(rows)
    low, high = np.quantile(boot, [0.025, 0.975])
    return [float(low), float(high)]


def main():
    rng = np.random.default_rng(20260922)
    reports = {}
    lines = [
        "# Sampled POMO, no augmentation: Gurobi-trained E200 vs Self-improving E365",
        "",
        "Each model uses one stochastic sample rollout per instance with POMO=min(J,100). The sample seed is fixed per instance and reset to the same value for both models. Every distinct sampled route is evaluated by Gurobi; the lowest exact makespan is reported. There is no eightfold augmentation. Lower is better. Both test pipelines use Gurobi for final evaluation; the comparison is between the trained upper policies.",
        "",
        "| Dataset | Cases | Gurobi E200 | Self E365 | Self − Gurobi | Paired 95% CI | Self win/tie/loss |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for dataset, expected in (("real529", 529), ("synthetic5000", 5000)):
        path = BASE / dataset / "results.json"
        data = json.loads(path.read_text())
        if not data["complete"] or len(data["records"]) != expected:
            raise RuntimeError(f"{dataset} is incomplete")
        params = data["params"]
        if params["decode_type"] != "sample" or params["augmentation"] or params["dataset_count"] != expected:
            raise RuntimeError(f"{dataset} protocol differs from requested sample/no-augmentation")
        rows = data["records"]
        groups = {"all": rows}
        if dataset == "real529":
            subset = [row for row in rows if 20 <= row["J"] <= 100]
            if len(subset) != 284:
                raise RuntimeError(f"expected Real529 J20-100 subset of 284, got {len(subset)}")
            groups["J20_100"] = subset
        for size in sorted({row["J"] for row in rows}):
            groups[f"J{size}"] = [row for row in rows if row["J"] == size]
        details = {}
        for name, group in groups.items():
            base = np.asarray([row["objectives"]["gurobi_e200"] for row in group], dtype=float)
            self_ = np.asarray([row["objectives"]["self_e365"] for row in group], dtype=float)
            delta = self_ - base
            details[name] = {
                "instances": len(group),
                "gurobi_e200_mean": float(base.mean()),
                "self_e365_mean": float(self_.mean()),
                "self_minus_gurobi": float(delta.mean()),
                "ci95": paired_interval(group, rng),
                "self_wins": int(np.sum(delta < -1e-6)),
                "ties": int(np.sum(np.abs(delta) <= 1e-6)),
                "self_losses": int(np.sum(delta > 1e-6)),
            }
        reports[dataset] = {"params": params, "results": details}
        for name in ("all", "J20_100") if dataset == "real529" else ("all",):
            item = details[name]
            label = dataset if name == "all" else f"{dataset} {name}"
            lines.append(
                f"| {label} | {item['instances']} | {item['gurobi_e200_mean']:.6f} | "
                f"{item['self_e365_mean']:.6f} | {item['self_minus_gurobi']:+.6f} | "
                f"[{item['ci95'][0]:+.6f}, {item['ci95'][1]:+.6f}] | "
                f"{item['self_wins']}/{item['ties']}/{item['self_losses']} |"
            )
    lines.append("")
    for dataset in ("real529", "synthetic5000"):
        lines += [f"## {dataset} by J", "", "| J | Cases | Gurobi E200 | Self E365 | Self − Gurobi |", "| ---: | ---: | ---: | ---: | ---: |"]
        details = reports[dataset]["results"]
        for key in sorted((key for key in details if key.startswith("J") and key[1:].isdigit()), key=lambda value: int(value[1:])):
            item = details[key]
            lines.append(
                f"| {key[1:]} | {item['instances']} | {item['gurobi_e200_mean']:.6f} | "
                f"{item['self_e365_mean']:.6f} | {item['self_minus_gurobi']:+.6f} |"
            )
        lines.append("")
    lines.append("These are single-seed stochastic sample results. They should be read alongside the existing greedy tests; a different sample seed can change the exact sampled candidate set.")
    lines.append("")
    (BASE / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    (BASE / "report.md").write_text("\n".join(lines))
    print(f"Wrote {BASE / 'report.md'}")


if __name__ == "__main__":
    main()
