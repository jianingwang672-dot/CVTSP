"""Run SA / VNS / ILS on Real529 with budgets matched to E133.

Per instance, the time budget is ``--factor`` x E133's single-core cost on that
instance (GPU decode + summed single-thread Gurobi time, from
outputs/bks_comparison_20261023/timing/new_e133.json). Every improvement is
traced as (evaluations, seconds, makespan), so the results can be read at any
smaller time budget (1x, 3x, ...) or at evaluation budgets equal to E133's
number of exact SOCP solves. Workers are pinned one per physical P-core and run
Gurobi with one thread; E133's single-core cost was measured with 16
concurrent unpinned workers, so the matched budget is, if anything, generous
to the baselines.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
TIMING = ROOT / "outputs/bks_comparison_20261023/timing/new_e133.json"

_STATE = {}


def _init(cpu_queue, dataset_dir: str) -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    cpu = cpu_queue.get()
    os.sched_setaffinity(0, {cpu})
    import gurobipy as gp
    from src.CVPSolver import SolverConfig
    from src.TSProblemDef import load_dataset
    _STATE["env"] = gp.Env(params={"OutputFlag": 0})
    _STATE["solver"] = SolverConfig(solver_backend="gurobi", gurobi_threads=1, output_flag=0)
    _STATE["instances"] = {x.instance_id: x for x in load_dataset(dataset_dir, "all")}
    _STATE["cpu"] = cpu


def _run(task):
    from src.Metaheuristics import run_metaheuristic
    method, instance_id, budget, seed = task
    instance = _STATE["instances"][instance_id]
    result = run_metaheuristic(method, instance, _STATE["solver"], env=_STATE["env"],
                               time_budget=budget, seed=seed)
    return {
        "method": method, "instance_id": instance_id, "problem_size": instance.J, "seed": seed,
        "time_budget": budget, "best_makespan": result.best_makespan, "initial_makespan": result.initial_makespan,
        "best_sequence": result.best_sequence, "evaluations": result.evaluations,
        "cache_hits": result.cache_hits, "seconds": result.seconds, "trace": result.trace, "cpu": _STATE["cpu"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=["sa", "vns", "ils"])
    parser.add_argument("--factor", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpus", default="0,2,4,6,8,10,12,14", help="one worker per listed CPU")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/metaheuristics_real529_20261026")
    parser.add_argument("--dataset-dir", default=str(ROOT / "instance/Data"))
    parser.add_argument("--instance-ids", nargs="+", help="subset for smoke tests")
    args = parser.parse_args()
    timing = {r["instance_id"]: r for r in json.loads(TIMING.read_text())["records"]}
    budgets = {i: args.factor * (r["decode_seconds"] + r["gurobi_cpu_seconds"]) for i, r in timing.items()}
    ids = args.instance_ids or sorted(budgets, key=lambda i: -budgets[i])  # longest first
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "params.json").write_text(json.dumps({
        "factor": args.factor, "seed": args.seed, "cpus": args.cpus, "timing_reference": str(TIMING),
        "budget": "factor x (E133 decode + summed single-thread Gurobi seconds) per instance",
        "solver": "Gurobi fixed-sequence SOCP, 1 thread", "initial": "nearest neighbour + Euclidean 2-opt",
        "granular_k": 8}, indent=2))
    cpus = [int(c) for c in args.cpus.split(",")]
    ctx = mp.get_context("spawn")
    cpu_queue = ctx.Queue()
    for cpu in cpus:
        cpu_queue.put(cpu)
    with ctx.Pool(len(cpus), initializer=_init, initargs=(cpu_queue, args.dataset_dir)) as pool:
        for method in args.methods:
            path = args.output_dir / f"{method}.jsonl"
            done = {json.loads(l)["instance_id"] for l in path.read_text().splitlines()} if path.exists() else set()
            tasks = [(method, i, budgets[i], args.seed) for i in ids if i not in done]
            started = time.perf_counter()
            with path.open("a") as out:
                for k, record in enumerate(pool.imap_unordered(_run, tasks), 1):
                    out.write(json.dumps(record) + "\n")
                    out.flush()
                    if k % 25 == 0 or k == len(tasks):
                        print(f"{method} {k}/{len(tasks)} elapsed={time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
