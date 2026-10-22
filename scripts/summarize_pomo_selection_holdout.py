"""Rank checkpoints on the independent exact full-POMO selection holdout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/pomo_selection_20260921"
RUNS = OUTPUT / "runs"
MANIFEST = ROOT / "validation/pomo_selection_20260921/manifest.json"


def paired_ci(difference: np.ndarray, sizes: np.ndarray, *, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(sizes == size) for size in sorted(set(sizes))]
    samples = np.empty(3000, dtype=np.float64)
    for index in range(len(samples)):
        selected = np.concatenate(
            [group[rng.integers(len(group), size=len(group))] for group in strata]
        )
        samples[index] = difference[selected].mean()
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    expected_manifest_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    rows_by_epoch: dict[int, list[dict]] = {}
    checkpoint_hashes: dict[int, str] = {}
    for path in sorted(RUNS.glob("*/results.json")):
        payload = json.loads(path.read_text())
        if not payload["complete"]:
            continue
        params = payload["params"]
        if (
            params["validation_manifest_sha256"] != expected_manifest_sha
            or params["validation_content_sha256"] != manifest["content_fingerprints_sha256"]
            or params["augmentation"]
            or params["pomo_size"] != "J"
            or params["start_node_strategy"] != "nearest_depot"
        ):
            raise ValueError(f"incompatible exact POMO validation run: {path}")
        rows = payload["records"]
        if len(rows) != manifest["count"] or [
            row["instance_fingerprint"] for row in rows
        ] != manifest["content_fingerprints"]:
            raise ValueError(f"incomplete or reordered instances: {path}")
        for label, epoch in params["checkpoint_epochs"].items():
            epoch = int(epoch)
            digest = params["checkpoint_sha256"][label]
            if epoch in checkpoint_hashes and checkpoint_hashes[epoch] != digest:
                raise ValueError(f"checkpoint hash changed for epoch {epoch}")
            checkpoint_hashes[epoch] = digest
            if epoch in rows_by_epoch:
                previous = rows_by_epoch[epoch]
                if any(
                    a["objectives"][f"epoch{epoch}"] != b["objectives"][label]
                    for a, b in zip(previous, rows)
                ):
                    raise ValueError(f"cached score changed for epoch {epoch}")
                continue
            if any(not row["feasible"][label] for row in rows):
                raise ValueError(f"infeasible score for epoch {epoch}")
            rows_by_epoch[epoch] = [
                {
                    "problem_size": row["problem_size"],
                    "objectives": {f"epoch{epoch}": row["objectives"][label]},
                }
                for row in rows
            ]
    if not rows_by_epoch:
        raise RuntimeError("no completed full-POMO validation runs")
    sizes = np.asarray([row["problem_size"] for row in next(iter(rows_by_epoch.values()))])
    baseline = np.asarray(
        [row["objectives"]["epoch100"] for row in rows_by_epoch[100]], dtype=np.float64
    )
    details = {}
    for epoch, rows in rows_by_epoch.items():
        values = np.asarray(
            [row["objectives"][f"epoch{epoch}"] for row in rows], dtype=np.float64
        )
        diff = values - baseline
        details[epoch] = {
            "mean_makespan": float(values.mean()),
            "per_size_mean": {
                str(size): float(values[sizes == size].mean()) for size in sorted(set(sizes))
            },
            "difference_from_epoch100": float(diff.mean()),
            "difference_from_epoch100_ci95": paired_ci(diff, sizes, seed=20260921 + epoch),
            "epoch_better_tie_worse_than_100": [
                int(np.count_nonzero(diff < -1e-6)),
                int(np.count_nonzero(np.abs(diff) <= 1e-6)),
                int(np.count_nonzero(diff > 1e-6)),
            ],
            "checkpoint_sha256": checkpoint_hashes[epoch],
        }
    ranking = sorted(details, key=lambda epoch: details[epoch]["mean_makespan"])
    summary = {
        "purpose": "checkpoint_selection_not_final_test",
        "validation_manifest_sha256": expected_manifest_sha,
        "validation_content_sha256": manifest["content_fingerprints_sha256"],
        "evaluated_epochs": sorted(details),
        "ranking": ranking,
        "best_epoch": ranking[0],
        "by_epoch": {str(epoch): details[epoch] for epoch in sorted(details)},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = [
        "# 独立固定集：完整 POMO 精确选模",
        "",
        "J=20/40/60/80/100 各 200 例；greedy、POMO=J、无增强；每条候选均用 Gurobi 精算后取最优。",
        "这批实例仅用于 checkpoint 选择，不作为最终泛化测试。数值越低越好。",
        "",
        "| 排名 | Epoch | 总体 makespan | J20 | J40 | J60 | J80 | J100 | 相对 E100 的配对差值 95% CI |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for position, epoch in enumerate(ranking, start=1):
        row = details[epoch]
        groups = row["per_size_mean"]
        lo, hi = row["difference_from_epoch100_ci95"]
        report.append(
            f"| {position} | {epoch} | {row['mean_makespan']:.6f} | "
            + " | ".join(f"{groups[str(size)]:.6f}" for size in (20, 40, 60, 80, 100))
            + f" | [{lo:+.5f}, {hi:+.5f}] |"
        )
    (OUTPUT / "report.md").write_text("\n".join(report) + "\n")
    print(f"Evaluated {len(details)} epochs; best={ranking[0]} makespan={details[ranking[0]]['mean_makespan']:.6f}")


if __name__ == "__main__":
    main()
