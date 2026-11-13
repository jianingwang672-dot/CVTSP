"""Summarise SA / VNS / ILS on Real529 against G200 and E133 at matched budgets.

A run with budget factor F traces every improvement, so the result at a
smaller factor f is the best value found within f x (E133 single-core time),
and at an evaluation budget m x (E133 exact solves) the best value within that
many SOCP evaluations. Gaps use the paper definition 100 (UB - BKS) / UB.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BKS = ROOT / "outputs/bks_comparison_20261023/best_known_all_methods.json"  # min UB over all methods in Li et al. (2025)
TIMING = ROOT / "outputs/bks_comparison_20261023/timing"


def best_within(trace, *, seconds=math.inf, evaluations=math.inf):
    value = math.inf
    for evals, secs, makespan in trace:
        if evals <= evaluations and secs <= seconds:
            value = min(value, makespan)
    return value


def family(instance_id: str) -> str:
    line = (ROOT / "instance/Data" / f"{instance_id}.txt").open().readline()
    name = re.sub(r"^\d+\.", "", line.strip().replace("//Name=", "").replace(".dat", ""))
    return re.sub(r"\d+$", "", name)


def group(instance_id: str, size: int) -> str:
    fam = family(instance_id)
    if fam in ("Regular", "Clustered"):
        return f"Poi325 {'Regular' if fam == 'Regular' else 'Cluster'}"
    return "Gam72" if size <= 15 else ("Erd132 16-20" if size <= 20 else "Erd132 25-50")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/metaheuristics_real529_20261026")
    parser.add_argument("--time-factors", type=float, nargs="+", default=[1, 3, 10])
    parser.add_argument("--eval-factors", type=float, nargs="+", default=[1, 3, 10])
    args = parser.parse_args()
    params = json.loads((args.run_dir / "params.json").read_text())
    run_factor = float(params["factor"])
    bks = {k: v["best"] for k, v in json.loads(BKS.read_text()).items()}
    models = {label: {r["instance_id"]: r for r in json.loads((TIMING / f"{label}.json").read_text())["records"]}
              for label in ("epoch200", "new_e133")}
    e133 = models["new_e133"]
    rows = defaultdict(dict)  # (method, budget) -> instance -> makespan
    for label, name in (("epoch200", "G200"), ("new_e133", "E133")):
        for i, r in models[label].items():
            rows[(name, "-")][i] = r["objective"]
    for path in sorted(args.run_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            i, method = r["instance_id"], r["method"].upper()
            single_core = r["time_budget"] / run_factor
            for f in args.time_factors:
                if f <= run_factor:
                    rows[(method, f"{f:g}x time")][i] = best_within(r["trace"], seconds=f * single_core)
            for m in args.eval_factors:
                rows[(method, f"{m:g}x evals")][i] = best_within(r["trace"], evaluations=m * e133[i]["routes"])
    complete = [k for k, v in rows.items() if len(v) == 529]
    instances = sorted(e133)
    groups = defaultdict(list)
    for i in instances:
        groups[group(i, e133[i]["problem_size"])].append(i)
    order = ["Gam72", "Erd132 16-20", "Erd132 25-50", "Poi325 Regular", "Poi325 Cluster"]
    header = "| Method | Budget | " + " | ".join(order) + " | Real529 Δ | mean UB | vs E133 W/T/L |"
    print(header)
    print("|" + "---|" * (len(order) + 5))
    for key in sorted(complete, key=lambda k: (k[0] not in ("G200", "E133"), k[0], k[1])):
        values = rows[key]
        gap = lambda ids: np.mean([100 * (values[i] - bks[i]) / values[i] for i in ids])
        diff = [values[i] - e133[i]["objective"] for i in instances]
        wtl = f"{sum(d < -1e-4 for d in diff)}/{sum(abs(d) <= 1e-4 for d in diff)}/{sum(d > 1e-4 for d in diff)}"
        cells = " | ".join(f"{gap(groups[g]):+.2f}%" for g in order)
        print(f"| {key[0]} | {key[1]} | {cells} | {gap(instances):+.2f}% | "
              f"{np.mean([values[i] for i in instances]):.2f} | {wtl} |")
    partial = {k: len(v) for k, v in rows.items() if len(v) != 529}
    if partial:
        print(f"\n(incomplete, instances done: {partial})")


if __name__ == "__main__":
    main()
