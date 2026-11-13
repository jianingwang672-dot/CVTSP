"""Solve test500 (or Real529, --real529) instances with the full MISOCP (Model+) of Li, Zhou & Cote (2025).

Pure-Gurobi baseline for the synthetic test set: routing and all take-off /
landing decisions are optimised jointly, one single-threaded Gurobi process
per instance. Each instance writes <id>.json (result) and <id>.log (Gurobi log,
from which the incumbent / bound trajectory can be read); finished instances
are skipped on restart. --per-size N takes the first N instances of every J
group (the pilot uses 3 per group to choose the final time limit). --real529
solves the literature benchmark instead (instance/Data, all 529 instances).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

from src.FullCVTSPSolver import FullCVTSPConfig, solve_full_cvtsp_gurobi
from src.PseudoExpert import instance_from_dict
from src.TSProblemDef import load_dataset

ROOT = Path(__file__).resolve().parents[1]


def solve_one(job):
    instance, out_dir, time_limit = job
    path = out_dir / f"{instance.instance_id}.json"
    config = FullCVTSPConfig(formulation="enhanced", time_limit=time_limit, threads=1,
                             output_flag=1, log_file=str(path.with_suffix(".log")))
    started = time.perf_counter()
    result = solve_full_cvtsp_gurobi(instance, config).to_dict()
    result["problem_size"] = int(instance.J)
    result["wall_seconds"] = time.perf_counter() - started
    path.write_text(json.dumps(result))
    return instance.instance_id, instance.J, result["status"], result["objective"], result["mip_gap"], result["wall_seconds"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "validation/e500_logic_e100_20261010/test500")
    parser.add_argument("--real529", action="store_true", help="solve instance/Data instead of test500")
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 40, 60, 80, 100])
    parser.add_argument("--per-size", type=int, default=None)
    parser.add_argument("--time-limit", type=float, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.real529:
        instances = load_dataset(str(ROOT / "instance/Data"), "all")
    else:
        instances = []
        for size in args.sizes:
            rows = torch.load(args.data_dir / f"J{size}.pt", weights_only=False)["instances"]
            instances += [instance_from_dict(row) for row in rows[: args.per_size]]
    jobs = [(instance, args.output_dir, args.time_limit) for instance in instances
            if not (args.output_dir / f"{instance.instance_id}.json").exists()]
    # largest instances first so the tail of the run is not one long J100 solve
    jobs.sort(key=lambda job: -job[0].J)
    print(f"{len(jobs)} instances to solve, time limit {args.time_limit:g}s, {args.workers} workers", flush=True)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for instance_id, size, status, objective, gap, seconds in pool.imap_unordered(solve_one, jobs):
            print(f"{instance_id} J={size} status={status} obj={objective} gap={gap} wall={seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
