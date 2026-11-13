"""Per-instance wall time of the Real529 protocol, one model at a time.

Protocol (same as the Real529 benchmark): greedy POMO=min(J,100) with
nearest-depot starts on all eight geometric views, every distinct route solved
exactly by Gurobi (1 thread per solve), score = minimum. For each instance we
record the decode time, the wall time of the parallel exact solves, and the
summed single-thread Gurobi time, so both wall-clock and single-core cost can
be reported. Models run sequentially so their timings do not interfere.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch

from src.CVPSolver import SolverConfig
from src.ImitationLearning import load_policy_checkpoint
from src.RewardWorkerPool import RewardWorkerPool
from src.TSPEnv import generate_sequences
from src.TSProblemDef import augment_instance_by_8_fold, load_dataset
from src.TSPUtils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="LABEL=CHECKPOINT")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", default="instance/Data")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    torch.set_num_threads(1)
    set_seed(args.seed)
    device = torch.device(args.device)
    instances = load_dataset(args.dataset_dir, "all")
    if len(instances) != 529:
        raise ValueError(f"expected 529 real instances, found {len(instances)}")
    solver = SolverConfig(solver_backend="gurobi", gurobi_threads=1, output_flag=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pool = RewardWorkerPool(args.workers, env_output_flag=0)
    try:
        # Warm up CUDA and the worker pool so the first instance is not charged for start-up.
        warm = instances[0]
        for item in args.model:
            model, _ = load_policy_checkpoint(Path(item.split("=", 1)[1]), device=device)
            with torch.no_grad():
                generate_sequences(model.eval(), warm, device=device, decode_type="greedy",
                                   num_candidates=min(warm.J, 100))
        pool.map([(warm, list(range(warm.J)), solver)] * args.workers)
        for item in args.model:
            label, path = item.split("=", 1)
            model, _ = load_policy_checkpoint(Path(path), device=device)
            model.config = replace(model.config, max_pomo_size=100, pomo_divisor=None,
                                   start_node_strategy="nearest_depot")
            model.eval()
            records = []
            for index, instance in enumerate(instances):
                t0 = time.perf_counter()
                with torch.no_grad():
                    combined = []
                    for view in augment_instance_by_8_fold(instance):
                        combined += [tuple(r) for r in generate_sequences(
                            model, view, device=device, decode_type="greedy",
                            num_candidates=model.config.max_pomo_size)]
                if device.type == "cuda":
                    torch.cuda.synchronize()
                unique = list(dict.fromkeys(combined))
                t1 = time.perf_counter()
                solved = pool.map([(instance, list(r), solver) for r in unique])
                t2 = time.perf_counter()
                values = []
                for _, _, sol in solved:
                    if not sol.success or str(sol.status).strip() != "2" or not math.isfinite(sol.objective):
                        raise RuntimeError(f"non-optimal solve on {instance.instance_id}: {sol.status}")
                    values.append(float(sol.objective))
                records.append({
                    "instance_id": instance.instance_id, "problem_size": instance.J,
                    "objective": min(values), "routes": len(unique),
                    "decode_seconds": t1 - t0, "solve_wall_seconds": t2 - t1,
                    "wall_seconds": t2 - t0,
                    "gurobi_cpu_seconds": float(sum(sol.solve_time for _, _, sol in solved)),
                })
                if (index + 1) % 50 == 0 or index + 1 == len(instances):
                    print(f"{label} {index + 1}/{len(instances)} "
                          f"mean_wall={sum(r['wall_seconds'] for r in records) / len(records):.2f}s", flush=True)
            (args.output_dir / f"{label}.json").write_text(json.dumps({
                "checkpoint": path, "device": args.device, "workers": args.workers,
                "protocol": "8-view greedy POMO=min(J,100), nearest-depot, exact min, Gurobi 1 thread",
                "records": records}, indent=1))
    finally:
        pool.close()


if __name__ == "__main__":
    main()
