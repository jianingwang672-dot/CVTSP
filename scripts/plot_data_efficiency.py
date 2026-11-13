"""Data-efficiency curves: validation2000 full-POMO score vs. training budget.

Two panels sharing the y-axis: cumulative exact SOCP solves and cumulative
training instances (both log scale). Series: the proposed method (progressive
recipe run, every evaluated epoch), POMO + exact Gurobi reward trained from
scratch with the G200 recipe (epochs 1-14 and the equal-budget checkpoints),
and G200 itself (E200) as a single point.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "outputs/progressive_recipe_20261020"
POMO = ROOT / "outputs/pomo_gurobi_equal_budget_20261027"
G200_METRICS = ROOT / "outputs/train__cvtsp_random_n100_nearest_quarter/metrics.csv"
OURS, BASE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"


def scores(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    out = {k: v["mean_makespan"] for k, v in d["overall"].items() if isinstance(v, dict) and "mean_makespan" in v}
    for k, v in d["overall"].items():
        if k.endswith("_mean_makespan"):
            out[k[: -len("_mean_makespan")]] = v
    return out


def cumulative(path: Path, column: str, per_epoch_instances: int) -> dict[int, tuple[float, float]]:
    rows = {}
    with path.open() as fp:
        for row in csv.DictReader(fp):
            rows[int(float(row["epoch"]))] = float(row[column] or 0)
    out, total = {}, 0.0
    for epoch in sorted(rows):
        total += rows[epoch]
        out[epoch] = (total, epoch * per_epoch_instances)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/data_efficiency_20261027/data_efficiency")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    curve = scores(POMO / "curve_validation2000/results.json")
    select = scores(RECIPE / "select_recipe_validation2000/results.json")
    # the selected model itself (refined selection, same protocol and instances)
    select.update({k: v for k, v in scores(RECIPE / "refine_select_a/results.json").items() if k == "recipe_e133"})
    recipe_budget = cumulative(RECIPE / "train/metrics.csv", "total_gurobi_calls_all", 10_000)

    ours = []
    for label, value in {**select, **curve}.items():
        m = re.fullmatch(r"recipe_e(\d+)", label)
        if m and int(m.group(1)) in recipe_budget:
            ours.append((*recipe_budget[int(m.group(1))], value, int(m.group(1))))
    ours.sort()

    pomo = []
    if (POMO / "train/metrics.csv").exists():
        pomo_budget = cumulative(POMO / "train/metrics.csv", "solver_calls", 100_000)
        for label, value in curve.items():
            m = re.fullmatch(r"pomo_e(\d+)", label)
            if m and int(m.group(1)) in pomo_budget:
                pomo.append((*pomo_budget[int(m.group(1))], value, int(m.group(1))))
        budget_file = POMO / "train/budget_checkpoints.json"
        if budget_file.exists():
            for entry in json.loads(budget_file.read_text()):
                if f"pomo_{entry['name']}" in curve:
                    pomo.append((entry["solves"], entry["instances"], curve[f"pomo_{entry['name']}"], None))
    pomo.sort()

    g200 = None
    if "g200_e200" in curve:
        with G200_METRICS.open() as fp:  # resumed epochs appear twice; the last row wins
            calls = {int(float(r["epoch"])): float(r["solver_calls"]) for r in csv.DictReader(fp)}
        g200 = (sum(v for e, v in calls.items() if e <= 200), 200 * 100_000, curve["g200_e200"])

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, idx, xlabel in ((axes[0], 0, "Cumulative exact SOCP solves in training"),
                            (axes[1], 1, "Cumulative training instances")):
        if ours:
            ax.plot([p[idx] for p in ours], [p[2] for p in ours], color=OURS, lw=2, marker="o", ms=4,
                    label="Proposed (self-improving)")
        if pomo:
            ax.plot([p[idx] for p in pomo], [p[2] for p in pomo], color=BASE, lw=2, marker="s", ms=4,
                    label="POMO + exact Gurobi reward")
        if g200:
            ax.axhline(g200[2], color=BASE, lw=0.8, ls=(0, (4, 3)), zorder=0)
            ax.plot([g200[idx]], [g200[2]], color=BASE, marker="D", ms=7, ls="none", mfc="white", mew=2,
                    label="G200 (POMO + Gurobi, 200 epochs)")
            ax.annotate(f"G200\n{g200[idx]:.2e}".replace("e+0", "e"), (g200[idx], g200[2]), xytext=(0, 10),
                        textcoords="offset points", ha="center", va="bottom", color=INK, fontsize=8)
        e133 = next((p for p in ours if p[3] == 133), None)
        if e133:
            ax.plot([e133[idx]], [e133[2]], color=OURS, marker="o", ms=8, ls="none", mec="white", mew=1.5, zorder=5)
            ax.annotate(f"E{e133[3]} ({e133[idx]:.2e})".replace("e+0", "e"), (e133[idx], e133[2]),
                        xytext=(-2, -9), textcoords="offset points", ha="right", va="top", color=INK, fontsize=8)
        if g200 and e133:
            ratio = g200[idx] / e133[idx]
            ax.annotate("", xy=(e133[idx], g200[2] - 0.12), xytext=(g200[idx], g200[2] - 0.12),
                        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
            ax.text((e133[idx] * g200[idx]) ** 0.5, g200[2] - 0.15, f"{ratio:.0f}x fewer", ha="center",
                    va="top", color=MUTED, fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.grid(True, which="major", color="#e6e5e1", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("validation2000 mean makespan")
    axes[0].set_ylim(16.9, None)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.output.with_suffix(f".{ext}"), dpi=200)
    rows = [("proposed", *p) for p in ours] + [("pomo_gurobi", *p) for p in pomo]
    if g200:
        rows.append(("g200", *g200, 200))
    with args.output.with_suffix(".csv").open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["series", "cumulative_solves", "cumulative_instances", "validation2000", "epoch"])
        w.writerows(rows)
    print(f"wrote {args.output}.png/.pdf/.csv  ours={len(ours)} pomo={len(pomo)} g200={'yes' if g200 else 'no'}")


if __name__ == "__main__":
    main()
