"""Recover per-batch values from the cumulative per-batch training log lines.

Each "Epoch N: [...] k/10000 J=.. Anchor=.. Expert=.. dV=.. RL=.. Calls=..
NM=e/a/c ... Sec=.." line reports running epoch averages; consecutive lines are
differenced into per-batch values, then aggregated per epoch and J bucket.
When an epoch appears several times (resumed runs), its last complete pass is
used.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

LINE = re.compile(
    r"Epoch\s+(\d+): \[[#-]+\]\s+(\d+)/\s*(\d+)\(.*?\) J=\s*(\d+) Anchor=([\d.]+) Expert=([\d.]+) "
    r"dV=([-\d.]+) RL=([-\d.]+) IL/Edge=pending Calls=([\d.]+)/inst .*?NM=(\d+)/(\d+)/(\d+) "
    r"NMgap=([-\d.]+) Sec=([\d.]+)"
)
BUCKETS = [(20, 35), (36, 51), (52, 67), (68, 83), (84, 100)]


def bucket(j: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= j <= hi:
            return f"J{lo}-{hi}"
    return "other"


def parse(path: Path) -> dict:
    passes: dict[int, list[list]] = defaultdict(list)
    for line in path.read_text(errors="replace").splitlines():
        m = LINE.search(line)
        if not m:
            continue
        epoch, done, total = int(m[1]), int(m[2]), int(m[3])
        row = [done, int(m[4]), float(m[5]), float(m[6]), float(m[8]), float(m[9]),
               int(m[10]), int(m[11]), int(m[12]), float(m[14])]
        runs = passes[epoch]
        if not runs or done <= runs[-1][-1][0]:
            runs.append([])  # a new pass of this epoch (restart)
        runs[-1].append(row)
    out = {}
    for epoch, runs in sorted(passes.items()):
        rows = runs[-1]
        if rows[-1][0] != 10000 and len(runs) > 1:
            rows = next((r for r in reversed(runs) if r[-1][0] == 10000), rows)
        agg = defaultdict(lambda: defaultdict(float))
        prev = [0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0.0]
        for row in rows:
            n = row[0] - prev[0]
            if n <= 0:
                prev = row
                continue
            anchor = (row[2] * row[0] - prev[2] * prev[0]) / n
            expert = (row[3] * row[0] - prev[3] * prev[0]) / n
            rl = (row[4] * row[0] - prev[4] * prev[0]) / n
            calls = (row[5] * row[0] - prev[5] * prev[0]) / n
            for key in ("all", bucket(row[1])):
                a = agg[key]
                a["instances"] += n
                a["anchor_sum"] += anchor * n
                a["expert_sum"] += expert * n
                a["improvement_pct_sum"] += 100 * (anchor - expert) / max(anchor, 1e-9) * n
                a["rl_sum"] += rl * n
                a["calls_sum"] += calls * n
                a["nm_eligible"] += row[6] - prev[6]
                a["nm_attempt"] += row[7] - prev[7]
                a["nm_accept"] += row[8] - prev[8]
                a["seconds"] += row[9]
                a["batches"] += 1
            prev = row
        result = {}
        for key, a in agg.items():
            n = max(a["instances"], 1)
            result[key] = {
                "instances": a["instances"],
                "anchor": a["anchor_sum"] / n,
                "expert": a["expert_sum"] / n,
                "improvement_pct": a["improvement_pct_sum"] / n,
                "rl_loss": a["rl_sum"] / n,
                "calls_per_instance": a["calls_sum"] / n,
                "nm_attempt_per_instance": a["nm_attempt"] / n,
                "nm_accept_per_instance": a["nm_accept"] / n,
                "nm_accept_rate": a["nm_accept"] / max(a["nm_attempt"], 1),
                "seconds_per_batch": a["seconds"] / max(a["batches"], 1),
                "batches": a["batches"],
            }
        out[epoch] = result
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-epoch", type=int, default=0)
    args = parser.parse_args()
    data = parse(args.log)
    if args.max_epoch:
        data = {e: v for e, v in data.items() if e <= args.max_epoch}
    args.output.write_text(json.dumps(data))
    complete = sum(1 for v in data.values() if v["all"]["instances"] == 10000)
    print(f"{args.log}: {len(data)} epochs parsed, {complete} complete")


if __name__ == "__main__":
    main()
