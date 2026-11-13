"""test500 with the Real529 eight-view protocol (greedy POMO per view, exact min).

Mirrors outputs/eval_vectorized_epoch100_vs_gurobi_epoch200_real529/
benchmark_real_cached.py: for every instance, each model decodes greedy POMO
on all eight geometric views, every distinct route is solved exactly with
Gurobi, and the instance score is the minimum. The view-0 (unaugmented)
minimum is recorded too.
"""

from __future__ import annotations

import argparse
import hashlib
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
from src.PseudoExpert import instance_from_dict, instance_to_dict
from src.RewardWorkerPool import RewardWorkerPool
from src.TSPEnv import generate_sequences
from src.TSProblemDef import augment_instance_by_8_fold
from src.TSPUtils import set_seed

SIZES = (20, 40, 60, 80, 100)


def fingerprint(instance) -> str:
    payload = instance_to_dict(instance)
    payload.pop("instance_id", None)
    payload.pop("path", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="LABEL=CHECKPOINT")
    parser.add_argument("--data-dir", type=Path, default=Path("validation/e500_logic_e100_20261010/test500"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N instances (timing probes)")
    args = parser.parse_args()
    torch.set_num_threads(1)
    set_seed(args.seed)
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    instances = []
    for size in args.sizes:
        path = args.data_dir / f"J{size}.pt"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files_sha256"][path.name]:
            raise RuntimeError(f"data file changed: {path}")
        instances += [instance_from_dict(r) for r in torch.load(path, weights_only=False)["instances"]]
    per, sizes = manifest["per_size"], manifest["sizes"]
    expected = [f for i, f in enumerate(manifest["content_fingerprints"]) if sizes[i // per] in args.sizes]
    if [fingerprint(x) for x in instances] != expected:
        raise RuntimeError("test500 fingerprints differ from the manifest")
    if args.limit:
        instances = instances[:args.limit]
    models = {}
    for item in args.model:
        label, path = item.split("=", 1)
        model, _ = load_policy_checkpoint(Path(path), device=torch.device("cpu"))
        model.config = replace(model.config, max_pomo_size=100, pomo_divisor=None,
                               start_node_strategy="nearest_depot")
        models[label] = model.eval()
    solver = SolverConfig(solver_backend="gurobi", gurobi_threads=1, output_flag=0)  # Real529 protocol
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "results.json"
    records = json.loads(result_path.read_text())["records"] if result_path.exists() else []
    started = time.perf_counter()
    pool = RewardWorkerPool(args.workers, env_output_flag=0)
    try:
        for index in range(len(records), len(instances)):
            instance = instances[index]
            routes, view0 = {}, {}
            with torch.no_grad():
                for label, model in models.items():
                    combined = []
                    for view_index, view in enumerate(augment_instance_by_8_fold(instance)):
                        cands = [tuple(r) for r in generate_sequences(
                            model, view, device=torch.device("cpu"), decode_type="greedy",
                            num_candidates=model.config.max_pomo_size)]
                        if view_index == 0:
                            view0[label] = cands
                        combined += cands
                    routes[label] = list(dict.fromkeys(combined))
            unique = list(dict.fromkeys(r for group in routes.values() for r in group))
            solved = pool.map([(instance, list(r), solver) for r in unique])
            value = {}
            for _, route, sol in solved:
                if not sol.success or str(sol.status).strip() != "2" or not math.isfinite(sol.objective):
                    raise RuntimeError(f"non-optimal solve on {instance.instance_id}: {sol.status}")
                value[tuple(route)] = float(sol.objective)
            records.append({
                "instance_id": instance.instance_id, "problem_size": instance.J,
                "instance_fingerprint": fingerprint(instance),
                "objectives": {l: min(value[r] for r in routes[l]) for l in models},
                "unaugmented_objectives": {l: min(value[r] for r in view0[l]) for l in models},
                "gurobi_calls": len(unique),
            })
            done = len(records) == len(instances)
            tmp = result_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"complete": done, "protocol": "8-view greedy POMO=J, exact min; Real529 solver config",
                                       "models": {i.split('=', 1)[0]: i.split('=', 1)[1] for i in args.model},
                                       "records": records}))
            tmp.replace(result_path)
            print(f"{len(records)}/{len(instances)} J={instance.J} {records[-1]['objectives']} "
                  f"solves={len(unique)} elapsed={time.perf_counter()-started:.0f}s", flush=True)
    finally:
        pool.close()


if __name__ == "__main__":
    main()
