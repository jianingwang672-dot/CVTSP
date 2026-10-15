"""Write report.md for the reward/diversity zero-training diagnostic."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SIZES = ("20", "40", "60", "80", "100")


def main() -> None:
    out = Path(sys.argv[1]).resolve()
    reward = json.loads((out / "reward_diagnostic.json").read_text())
    s = reward["summary"]
    labels = list(s)
    lines = ["# Reward and POMO diversity diagnostic (zero training)", ""]
    lines += ["## 1. Hybrid reward vs all-exact oracle (sampled Target POMO, 200 instances)", ""]
    lines += ["| metric | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    rows = [
        ("selected_z_current", "selected-route z (current hybrid)"),
        ("selected_z_if_upper_bound", "selected-route z if UB were used"),
        ("selected_z_oracle", "selected-route z (all exact)"),
        ("selected_share_of_positive_advantage", "selected share of positive advantage"),
        ("mean_slack_percent", "mean UB slack % over POMO"),
        ("selected_slack_percent", "selected-route UB slack %"),
        ("spearman_upper_vs_exact", "Spearman(UB, exact) in group"),
        ("best_exact_route_negative_advantage", "best exact route gets negative adv."),
        ("exact_mean_minus_best_percent", "exact mean - best over sampled POMO %"),
    ]
    for key, name in rows:
        lines.append(f"| {name} | " + " | ".join(f"{s[l]['all'][key]:.3f}" for l in labels) + " |")
    lines += ["", "Cosine between each reward construction's advantage vector and the "
              "all-exact advantage (1 = unbiased direction):", ""]
    lines += ["| variant | " + " | ".join(f"{l} all | {l} J100" for l in labels) + " |",
              "|---|" + "---:|" * (2 * len(labels))]
    for key in [k for k in s[labels[0]]["all"] if k.startswith("cos_")]:
        lines.append(f"| {key[4:]} | " + " | ".join(
            f"{s[l]['all'][key]:.3f} | {s[l]['100'][key]:.3f}" for l in labels) + " |")
    lines += ["", "Per J: Spearman(UB, exact) / cos(current) / mean slack %", "",
              "| J | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for j in SIZES:
        lines.append(f"| {j} | " + " | ".join(
            f"{s[l][j]['spearman_upper_vs_exact']:.3f} / {s[l][j]['cos_current_hybrid']:.3f} / "
            f"{s[l][j]['mean_slack_percent']:.1f}" for l in labels) + " |")
    if reward.get("paired"):
        a, b = reward["paired"]["labels"]
        lines += ["", f"Paired bootstrap {b} - {a} (95% CI):", ""]
        for key, value in reward["paired"]["effects"].items():
            if key in dict(rows) or key == "cos_current_hybrid":
                lo, hi = value["ci95"]
                lines.append(f"- {key}: {value['mean_delta_second_minus_first']:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    files = sorted((out / "diversity").glob("*.json"),
                   key=lambda p: (not p.stem.startswith("s500"), json.loads(p.read_text())["epoch"]))
    if files:
        lines += ["", "## 2. Greedy POMO diversity and exact quality (test protocol, 200 instances)", ""]
        cols = [("best_of_pomo", "best", 4), ("mean_of_pomo", "mean", 4),
                ("mean_minus_best_percent", "mean-best %", 2),
                ("routes_within_0p5pct_of_best", "near-best frac", 3),
                ("greedy_unique_fraction", "uniq frac", 3),
                ("greedy_pairwise_edge_jaccard", "edge Jaccard", 3),
                ("greedy_step_entropy", "step entropy", 4),
                ("greedy_step_max_prob", "step max p", 3),
                ("sample_pairwise_edge_jaccard", "sample Jaccard", 3)]
        lines += ["| model | epoch | " + " | ".join(c[1] for c in cols) + " |",
                  "|---|---:|" + "---:|" * len(cols)]
        for path in files:
            d = json.loads(path.read_text())
            a = d["summary"]["all"]
            lines.append(f"| {d['label']} | {d['epoch']} | " + " | ".join(
                f"{a[k]:.{p}f}" for k, _, p in cols) + " |")
        lines += ["", "Per J best-of-POMO / mean-of-POMO:", "",
                  "| model | " + " | ".join(f"J{j}" for j in SIZES) + " |", "|---|" + "---|" * 5]
        for path in files:
            d = json.loads(path.read_text())
            lines.append(f"| {d['label']} | " + " | ".join(
                f"{d['summary'][j]['best_of_pomo']:.3f} / {d['summary'][j]['mean_of_pomo']:.3f}"
                for j in SIZES) + " |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
