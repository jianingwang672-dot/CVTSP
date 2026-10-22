"""Pair the two E100 branch E120 checkpoints with the original E120 control."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/epoch100_controlled_branches"
CONTROL = ROOT / "outputs/pomo_selection_20260921/runs/epoch120/results.json"
BRANCH = BASE / "evaluation/results.json"


def mean(value):
    return float(np.mean(value))


def paired_improvement(baseline, candidate, rng):
    diff = baseline - candidate
    indices = rng.integers(0, len(diff), size=(10000, len(diff)))
    low, high = np.quantile(diff[indices].mean(axis=1), [0.025, 0.975])
    return {
        "mean_improvement": mean(diff),
        "ci95": [float(low), float(high)],
        "wins": int(np.sum(diff > 1e-6)),
        "ties": int(np.sum(np.abs(diff) <= 1e-6)),
        "losses": int(np.sum(diff < -1e-6)),
    }


def main():
    control = json.loads(CONTROL.read_text())
    branch = json.loads(BRANCH.read_text())
    if not control["complete"] or not branch["complete"]:
        raise RuntimeError("both full-POMO evaluations must be complete")
    if control["params"]["validation_manifest_sha256"] != branch["params"]["validation_manifest_sha256"]:
        raise RuntimeError("control and branch evaluation datasets differ")
    left = control["records"]
    right = branch["records"]
    if len(left) != 1000 or len(right) != 1000:
        raise RuntimeError("expected exactly 1000 paired instances")
    if [x["instance_fingerprint"] for x in left] != [x["instance_fingerprint"] for x in right]:
        raise RuntimeError("control and branch records are not aligned")
    labels = ("epoch100", "epoch120", "il2_e120", "lr150_e120")
    values = {
        "epoch100": np.asarray([row["objectives"]["epoch100"] for row in left]),
        "epoch120": np.asarray([row["objectives"]["epoch120"] for row in left]),
        "il2_e120": np.asarray([row["objectives"]["il2_e120"] for row in right]),
        "lr150_e120": np.asarray([row["objectives"]["lr150_e120"] for row in right]),
    }
    if not np.allclose(values["epoch100"], np.asarray([row["objectives"]["epoch100"] for row in right]), atol=1e-8, rtol=0):
        raise RuntimeError("cached E100 scores differ between runs")
    groups = {"all": np.arange(len(left))}
    groups.update({str(j): np.asarray([i for i, row in enumerate(left) if row["problem_size"] == j])
                   for j in (20, 40, 60, 80, 100)})
    rng = np.random.default_rng(20260922)
    report = {}
    for name, indices in groups.items():
        sub = {label: values[label][indices] for label in labels}
        report[name] = {
            "count": len(indices),
            "mean_makespan": {label: mean(sub[label]) for label in labels},
            "vs_epoch100": {label: paired_improvement(sub["epoch100"], sub[label], rng)
                            for label in labels if label != "epoch100"},
            "vs_original_epoch120": {
                label: paired_improvement(sub["epoch120"], sub[label], rng)
                for label in ("il2_e120", "lr150_e120")
            },
        }
    (BASE / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# E100 controlled continuation: E120 full-POMO results", "",
             "Same 1,000 frozen instances, greedy POMO=J, no augmentation, exact Gurobi for all distinct routes. Lower makespan is better. Positive paired improvement means the branch beat the comparator.", "",
             "| Checkpoint | Mean makespan | Paired improvement vs E100 (95% bootstrap CI) | Paired improvement vs original E120 (95% bootstrap CI) |",
             "| --- | ---: | ---: | ---: |"]
    overall = report["all"]
    for label in labels:
        score = overall["mean_makespan"][label]
        vs100 = overall["vs_epoch100"].get(label)
        vs120 = overall["vs_original_epoch120"].get(label)
        format_effect = lambda x: "—" if x is None else f"{x['mean_improvement']:.6f} [{x['ci95'][0]:.6f}, {x['ci95'][1]:.6f}]"
        lines.append(f"| {label} | {score:.6f} | {format_effect(vs100)} | {format_effect(vs120)} |")
    lines.extend(["", "By J:", "", "| J | E100 | Original E120 | IL2 E120 | Faster LR E120 |",
                  "| ---: | ---: | ---: | ---: | ---: |"])
    for j in (20, 40, 60, 80, 100):
        scores = report[str(j)]["mean_makespan"]
        lines.append(f"| {j} | {scores['epoch100']:.6f} | {scores['epoch120']:.6f} | {scores['il2_e120']:.6f} | {scores['lr150_e120']:.6f} |")
    lines.extend(["", "These results use a checkpoint-selection set already explored in historical backtests; an independent external test is required before claiming generalized improvement.", ""])
    (BASE / "report.md").write_text("\n".join(lines))
    print("Wrote", BASE / "report.md")


if __name__ == "__main__":
    main()
