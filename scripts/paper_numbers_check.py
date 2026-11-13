"""Recompute the training-budget and G200-gap numbers quoted in the paper.

None of these numbers is stored as-is in a file: each is a sum or a mean over
raw per-epoch or per-instance records. This script recomputes them from the raw
outputs and prints every number with the file it came from.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
G200_METRICS = ROOT / "outputs/train__cvtsp_random_n100_nearest_quarter/metrics.csv"
E133_METRICS = ROOT / "outputs/progressive_recipe_20261020/train/metrics.csv"
DATA_EFFICIENCY = ROOT / "outputs/data_efficiency_20261027/data_efficiency.csv"
REAL529 = ROOT / "outputs/bks_comparison_20261023/timing"
TEST500 = ROOT / "outputs/pomo_gurobi_equal_budget_20261027/eval_equal_solves_test500_aug8/results.json"
SELECTION_SOLVES = 2_235_364 + 1_498_832 + 1_441_537  # unique_solves in the three selection live.logs


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def gap_summary(ours: list[float], base: list[float]) -> str:
    gaps = [100 * (a - b) / b for a, b in zip(ours, base)]
    mean = statistics.mean(gaps)
    half = 1.96 * statistics.stdev(gaps) / math.sqrt(len(gaps))
    ratio = 100 * (sum(ours) - sum(base)) / sum(base)
    wins = sum(a < b * (1 - 1e-6) for a, b in zip(ours, base))
    losses = sum(a > b * (1 + 1e-6) for a, b in zip(ours, base))
    return (f"n={len(gaps)}  per-instance mean gap {mean:+.2f}% [{mean - half:+.2f}, {mean + half:+.2f}]  "
            f"ratio of means {ratio:+.2f}% ({statistics.mean(ours):.3f} vs {statistics.mean(base):.3f})  "
            f"E133 wins/ties/losses {wins}/{len(gaps) - wins - losses}/{losses}")


def main() -> None:
    print("== 1. G200 training solves")
    print(f"source: {rel(G200_METRICS)} (column solver_calls; epochs that were restarted appear twice)")
    last, all_rows = {}, 0.0
    for row in csv.DictReader(G200_METRICS.open()):
        epoch = float(row["epoch"])
        if epoch <= 200:
            last[epoch] = float(row["solver_calls"])
            all_rows += float(row["solver_calls"])
    g200_train, g200_as_run = sum(last.values()), all_rows
    print(f"E1-E200, last row per epoch (training only): {g200_train:,.0f}")
    print(f"E1-E200, all rows including restarted epochs: {g200_as_run:,.0f}")

    print("\n== 2. E133 solves")
    print(f"source: {rel(E133_METRICS)} (columns total_gurobi_calls_all, validation_gurobi_calls)")
    rows = list(csv.DictReader(E133_METRICS.open()))
    train = lambda last_epoch: sum(float(r["total_gurobi_calls_all"] or 0) for r in rows if float(r["epoch"]) <= last_epoch)
    validation = sum(float(r["validation_gurobi_calls"] or 0) for r in rows)
    to_e133, to_e200 = train(133), train(200)
    as_run = to_e200 + validation + SELECTION_SOLVES
    print(f"training to E133: {to_e133:,.0f}")
    print(f"training to E200: {to_e200:,.0f}")
    print(f"in-training validation: {validation:,.0f}")
    print(f"checkpoint selection on validation2000: {SELECTION_SOLVES:,.0f} (unique_solves in the live.log of "
          "select_recipe_validation2000, refine_select_a, refine_select_b)")
    print(f"as run in total: {as_run:,.0f}")

    print("\n== 3. Budget ratios (G200 / E133)")
    print(f"training only, to the checkpoint used: {g200_train / to_e133:.1f}x")
    print(f"training only, whole run:              {g200_train / to_e200:.1f}x")
    print(f"as run, both sides (G200 with restarts): {g200_as_run / as_run:.1f}x")
    print(f"as run, G200 without restarts (old 25x figure, mixed scope): {g200_train / as_run:.1f}x")

    print("\n== 4. Equal-solve POMO vs E133 on validation2000")
    print(f"source: {rel(DATA_EFFICIENCY)}")
    eff = list(csv.DictReader(DATA_EFFICIENCY.open()))
    ours = next(r for r in eff if r["series"] == "proposed" and r["epoch"] == "133")
    pomo = min((r for r in eff if r["series"] == "pomo_gurobi"),
               key=lambda r: abs(float(r["cumulative_solves"]) - float(ours["cumulative_solves"])))
    a, b = float(pomo["validation2000"]), float(ours["validation2000"])
    print(f"E133: {float(ours['cumulative_solves']):,.0f} solves, validation2000 {b:.3f}")
    print(f"POMO: {float(pomo['cumulative_solves']):,.0f} solves, {int(pomo['cumulative_instances']):,} instances, "
          f"validation2000 {a:.3f}")
    print(f"POMO worse by {100 * (a - b) / b:+.2f}% (ratio of means)")

    print("\n== 5. E133 vs G200 on Real529 (eight views)")
    print(f"source: {rel(REAL529 / 'new_e133.json')} and {rel(REAL529 / 'epoch200.json')}")
    load = lambda name: {r["instance_id"]: r["objective"] for r in json.loads((REAL529 / name).read_text())["records"]}
    e133, g200 = load("new_e133.json"), load("epoch200.json")
    ids = sorted(e133)
    print(gap_summary([e133[i] for i in ids], [g200[i] for i in ids]))

    print("\n== 6. E133 vs G200 on test500 (eight views)")
    print(f"source: {rel(TEST500)} (keys e133, g200)")
    records = json.loads(TEST500.read_text())["records"]
    print(gap_summary([r["objectives"]["e133"] for r in records], [r["objectives"]["g200"] for r in records]))


if __name__ == "__main__":
    main()
