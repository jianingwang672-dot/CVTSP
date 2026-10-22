"""Compare Epoch 365 with Epoch 100 on the identical external test instances."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/eval_vectorized_epoch365_external"
OLD = ROOT / "outputs"


def load_pair(kind: str) -> tuple[dict, dict]:
    old_path = OLD / f"eval_vectorized_epoch100_vs_gurobi_epoch200_{kind}/full/results.json"
    new_path = OUT / kind / "results.json"
    old, new = json.loads(old_path.read_text()), json.loads(new_path.read_text())
    if not old["complete"] or not new["complete"]:
        raise ValueError(f"{kind}: incomplete comparison")
    p, q = old["params"], new["params"]
    keys = ("baseline_label", "sizes", "decode_type", "augmentation", "pomo_size",
            "device", "torch_threads", "torch_version", "workers", "seed")
    if any(p[key] != q[key] for key in keys):
        raise ValueError(f"{kind}: evaluation protocols differ")
    if p["checkpoint_sha256"]["epoch200"] != q["checkpoint_sha256"]["epoch200"]:
        raise ValueError(f"{kind}: Gurobi-trained baseline checkpoint differs")
    if kind == "real529":
        keys = ("dataset_manifest", "aug_factor", "solver_config")
        if any(p[key] != q[key] for key in keys):
            raise ValueError("Real529 instance hashes, augmentation, or Gurobi config differ")
    else:
        keys = ("instances_per_size", "batch_size", "start_node_strategy",
                "solver_threads_per_worker")
        if any(p[key] != q[key] for key in keys):
            raise ValueError("synthetic5000 protocol differs")
    if len(old["records"]) != len(new["records"]):
        raise ValueError(f"{kind}: different numbers of instances")
    for left, right in zip(old["records"], new["records"]):
        if any(left[key] != right[key] for key in
               ("instance_id", "problem_size", "instance_fingerprint")):
            raise ValueError(f"{kind}: paired instance data differ")
        if left["objectives"]["epoch200"] != right["objectives"]["epoch200"]:
            raise ValueError(f"{kind}: cached baseline score changed")
        if not all((left["feasible"]["epoch100"],
                    right["feasible"]["epoch365"], right["feasible"]["epoch200"])):
            raise ValueError(f"{kind}: infeasible result")
    return old, new


def summarize(rows: list[tuple[dict, dict]], *, seed: int) -> dict:
    size = np.asarray([left["problem_size"] for left, _ in rows])
    baseline = np.asarray([left["objectives"]["epoch200"] for left, _ in rows])
    earlier = np.asarray([left["objectives"]["epoch100"] for left, _ in rows])
    later = np.asarray([right["objectives"]["epoch365"] for _, right in rows])
    difference = later - earlier
    strata = [np.flatnonzero(size == value) for value in sorted(set(size))]
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(3000)
    for index in range(len(bootstrap)):
        sampled = np.concatenate([
            group[rng.integers(len(group), size=len(group))] for group in strata
        ])
        bootstrap[index] = difference[sampled].mean()
    return {
        "instances": len(rows),
        "gurobi_epoch200_mean": float(baseline.mean()),
        "epoch100_mean": float(earlier.mean()),
        "epoch365_mean": float(later.mean()),
        "epoch365_minus_epoch100": float(difference.mean()),
        "epoch365_minus_epoch100_ci95": [
            float(value) for value in np.quantile(bootstrap, (0.025, 0.975))
        ],
        "epoch365_better_tie_worse_than_epoch100": [
            int(np.count_nonzero(difference < -1e-6)),
            int(np.count_nonzero(np.abs(difference) <= 1e-6)),
            int(np.count_nonzero(difference > 1e-6)),
        ],
        "by_size": {
            str(value): {
                "instances": int(np.count_nonzero(size == value)),
                "epoch100_mean": float(earlier[size == value].mean()),
                "epoch365_mean": float(later[size == value].mean()),
            }
            for value in sorted(set(size))
        },
    }


def main() -> None:
    real_old, real_new = load_pair("real529")
    synthetic_old, synthetic_new = load_pair("synthetic5000")
    all_real = list(zip(real_old["records"], real_new["records"]))
    trained_sizes = {20, 25, 30, 35, 40, 45, 50, 100}
    trained_real = [pair for pair in all_real if pair[0]["problem_size"] in trained_sizes]
    all_synthetic = list(zip(synthetic_old["records"], synthetic_new["records"]))
    if len(all_real) != 529 or len(trained_real) != 284 or len(all_synthetic) != 5000:
        raise ValueError("unexpected external test sizes")
    result = {
        "purpose": "external_test_epoch365_vs_epoch100_and_gurobi_epoch200",
        "real529": summarize(all_real, seed=365529),
        "real529_J20_100": summarize(trained_real, seed=365284),
        "synthetic5000": summarize(all_synthetic, seed=3655000),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    report = [
        "# Epoch 365 external tests",
        "",
        "The Real529 protocol uses greedy POMO=min(J,100), eightfold augmentation and exact Gurobi.",
        "The synthetic5000 protocol uses greedy POMO=J, no augmentation and exact Gurobi.",
        "Every comparison uses the same frozen instances and previously verified Epoch 200 baseline scores.",
        "Lower makespan is better. The confidence interval is a stratified paired bootstrap interval for E365−E100.",
        "",
        "| Dataset | Cases | Gurobi E200 | E100 | E365 | E365−E100 | 95% CI | E365 win/tie/loss |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for name in ("real529", "real529_J20_100", "synthetic5000"):
        row = result[name]
        lo, hi = row["epoch365_minus_epoch100_ci95"]
        w, t, l = row["epoch365_better_tie_worse_than_epoch100"]
        report.append(
            f"| {name} | {row['instances']} | {row['gurobi_epoch200_mean']:.6f} | "
            f"{row['epoch100_mean']:.6f} | {row['epoch365_mean']:.6f} | "
            f"{row['epoch365_minus_epoch100']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | "
            f"{w}/{t}/{l} |"
        )
    report.extend(("", "## Per-size means", ""))
    for name in ("real529", "synthetic5000"):
        report.extend((f"### {name}", "", "| J | Cases | E100 | E365 |", "|---:|---:|---:|---:|"))
        for size, row in result[name]["by_size"].items():
            report.append(
                f"| {size} | {row['instances']} | {row['epoch100_mean']:.6f} | "
                f"{row['epoch365_mean']:.6f} |"
            )
        report.append("")
    (OUT / "report.md").write_text("\n".join(report))
    for name in ("real529", "real529_J20_100", "synthetic5000"):
        row = result[name]
        print(f"{name}: E365={row['epoch365_mean']:.6f}, "
              f"E100={row['epoch100_mean']:.6f}, "
              f"difference={row['epoch365_minus_epoch100']:+.6f}")


if __name__ == "__main__":
    main()
