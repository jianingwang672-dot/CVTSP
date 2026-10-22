"""Paired sampled-POMO, no-augmentation, exact-Gurobi external evaluation.

One stochastic POMO rollout is made per model and instance.  A fixed,
label-independent seed is reset before each rollout, so results are
reproducible and a resumed run evaluates the same sampled candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch

from src.CVPSolver import SolverConfig
from src.CertifiedUpperBound import is_optimal_gurobi_solution
from src.ImitationLearning import load_policy_checkpoint
from src.PseudoExpert import instance_to_dict
from src.RewardWorkerPool import RewardWorkerPool
from src.TSPEnv import rollout_batch
from src.TSProblemDef import RandomCVTSPGenerator, load_dataset
from src.TSPUtils import set_seed


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SELF = ROOT / "outputs/train__scratch_e500_lr1e4_min1e7_w128_qdhybrid_replay30_vectorized/checkpoint-365.pt"
TRAIN_BASE = ROOT / "outputs/train__cvtsp_random_n100_nearest_quarter/checkpoint-200.pt"
PRIOR_REAL = ROOT / "outputs/eval_vectorized_epoch365_external/real529/results.json"
PRIOR_SYNTH = ROOT / "outputs/eval_vectorized_epoch365_external/synthetic5000/results.json"
SIZES = (20, 40, 60, 80, 100)


def fingerprint(instance) -> str:
    payload = instance_to_dict(instance)
    payload.pop("instance_id", None)
    payload.pop("path", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_instances(dataset: str, seed: int):
    if dataset == "real529":
        instances = load_dataset(ROOT / "instance/Data", "all")
        if len(instances) != 529:
            raise RuntimeError(f"expected 529 real instances, got {len(instances)}")
        previous = json.loads(PRIOR_REAL.read_text())
        expected = [row["instance_fingerprint"] for row in previous["records"]]
    else:
        instances = []
        for size in SIZES:
            generator = RandomCVTSPGenerator(
                dataset_dir=ROOT / "instance/Data",
                min_problem_size=size,
                max_problem_size=size,
                problem_sizes=[size],
                seed=seed + size,
            )
            instances.extend(generator.sample_batch(1000, problem_size=size))
        previous = json.loads(PRIOR_SYNTH.read_text())
        expected = [row["instance_fingerprint"] for row in previous["records"]]
    actual = [fingerprint(instance) for instance in instances]
    if len(actual) != len(expected) or actual != expected:
        raise RuntimeError("new evaluation instances differ from the prior external test")
    return instances, actual


def solver_config_for(dataset: str) -> SolverConfig:
    if dataset == "real529":
        # Preserve the prior Real529 final-evaluation formulation.
        return SolverConfig(solver_backend="gurobi", gurobi_threads=1, output_flag=0)
    # Preserve the prior synthetic5000 final-evaluation formulation.
    return SolverConfig(
        solver_backend="gurobi", gurobi_threads=1, output_flag=0,
        gurobi_params={"FeasibilityTol": 1e-8, "OptimalityTol": 1e-8, "NumericFocus": 2},
        gurobi_explicit_qcp=True,
    )


def summarize(records, dataset: str):
    if not records:
        return {}
    groups = {"all": records}
    if dataset == "real529":
        groups["J20_100"] = [x for x in records if 20 <= x["J"] <= 100]
    for size in sorted({x["J"] for x in records}):
        groups[f"J{size}"] = [x for x in records if x["J"] == size]
    output = {}
    for name, rows in groups.items():
        if not rows:
            continue
        base = np.asarray([x["objectives"]["gurobi_e200"] for x in rows], dtype=float)
        self_ = np.asarray([x["objectives"]["self_e365"] for x in rows], dtype=float)
        diff = self_ - base
        output[name] = {
            "instances": len(rows),
            "gurobi_e200_mean": float(base.mean()),
            "self_e365_mean": float(self_.mean()),
            "self_minus_gurobi": float(diff.mean()),
            "self_wins": int(np.sum(diff < -1e-6)),
            "ties": int(np.sum(np.abs(diff) <= 1e-6)),
            "self_losses": int(np.sum(diff > 1e-6)),
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["real529", "synthetic5000"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0, help="Bounded preflight only")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0 or args.save_interval <= 0 or args.limit < 0:
        raise ValueError("workers, save-interval and limit must be valid")
    seed = args.seed if args.seed is not None else (1234 if args.dataset == "real529" else 20260827)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    set_seed(seed)
    instances, fingerprints = load_instances(args.dataset, seed)
    if args.limit:
        instances = instances[:args.limit]
        fingerprints = fingerprints[:args.limit]
    paths = {"gurobi_e200": TRAIN_BASE, "self_e365": TRAIN_SELF}
    models = {}
    epochs = {}
    for label, path in paths.items():
        model, checkpoint = load_policy_checkpoint(path, device=torch.device("cpu"))
        model.config = replace(
            model.config, max_pomo_size=100, pomo_divisor=None,
            start_node_strategy="nearest_depot",
        )
        models[label] = model.eval()
        epochs[label] = int(checkpoint["epoch"])
    if epochs != {"gurobi_e200": 200, "self_e365": 365}:
        raise RuntimeError(f"unexpected checkpoint epochs: {epochs}")
    solver_config = solver_config_for(args.dataset)
    manifest_hash = hashlib.sha256("\n".join(fingerprints).encode()).hexdigest()
    params = {
        "dataset": args.dataset,
        "dataset_count": len(instances),
        "dataset_manifest_sha256": manifest_hash,
        "dataset_fingerprints": fingerprints,
        "checkpoints": {label: str(path) for label, path in paths.items()},
        "checkpoint_sha256": {label: sha256(path) for label, path in paths.items()},
        "checkpoint_epochs": epochs,
        "model_configs": {label: asdict(model.config) for label, model in models.items()},
        "decode_type": "sample",
        "sample_rollouts_per_model_instance": 1,
        "sample_seed_rule": "seed + 1000003 * (global_instance_index + 1); reset for each model",
        "augmentation": False,
        "pomo_size": "min(J,100)",
        "start_node_strategy": "nearest_depot",
        "final_score": "minimum exact Gurobi makespan over distinct sampled POMO routes",
        "solver_config": asdict(solver_config),
        "device": "cpu", "torch_threads": 1, "torch_version": torch.__version__,
        "numpy_version": np.__version__, "seed": seed,
        "workers": args.workers, "limit": args.limit,
    }
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "results.json"
    records = []
    old_runtime = 0.0
    if result_path.exists():
        if not args.resume:
            raise FileExistsError(result_path)
        previous = json.loads(result_path.read_text())
        old_params = {k: v for k, v in previous["params"].items() if k != "workers"}
        new_params = {k: v for k, v in params.items() if k != "workers"}
        if old_params != new_params:
            raise RuntimeError("resume protocol differs from saved results")
        if previous["complete"]:
            print(f"Already complete: {result_path}", flush=True)
            return
        records = previous["records"]
        old_runtime = float(previous["runtime_seconds"])
        if [x["fingerprint"] for x in records] != fingerprints[:len(records)]:
            raise RuntimeError("saved records do not match source dataset")
    save_json(out / "params.json", params)
    started = time.perf_counter()
    pool = RewardWorkerPool(args.workers, env_output_flag=0)
    try:
        for index in range(len(records), len(instances)):
            instance = instances[index]
            sample_seed = seed + 1000003 * (index + 1)
            routes = {}
            with torch.inference_mode():
                for label, model in models.items():
                    torch.manual_seed(sample_seed)
                    rollout = rollout_batch(
                        model, [instance], device=torch.device("cpu"),
                        decode_type="sample", use_pomo_start=True,
                    )
                    routes[label] = [tuple(route) for route in rollout.to_sequence_lists()[0]]
            unique = list(dict.fromkeys(route for label in models for route in routes[label]))
            solved = pool.map([(instance, list(route), solver_config) for route in unique])
            by_route = {}
            for instance_id, route, solution in solved:
                if instance_id != instance.instance_id or not is_optimal_gurobi_solution(solution):
                    raise RuntimeError(f"nonoptimal exact route: {instance_id} {solution.status}")
                by_route[route] = float(solution.objective)
            objectives = {label: min(by_route[route] for route in routes[label]) for label in models}
            best_routes = {label: list(min(routes[label], key=lambda route: by_route[route]))
                           for label in models}
            row = {
                "instance_id": instance.instance_id,
                "J": instance.J,
                "fingerprint": fingerprints[index],
                "sample_seed": sample_seed,
                "objectives": objectives,
                "best_routes": best_routes,
                "candidate_counts": {label: len(routes[label]) for label in models},
                "unique_candidate_counts": {label: len(set(routes[label])) for label in models},
                "gurobi_calls": len(unique),
            }
            records.append(row)
            if (len(records) % args.save_interval == 0) or len(records) == len(instances):
                elapsed = old_runtime + time.perf_counter() - started
                save_json(result_path, {
                    "params": params, "complete": len(records) == len(instances),
                    "runtime_seconds": elapsed,
                    "summary": summarize(records, args.dataset),
                    "records": records,
                })
                summary = summarize(records, args.dataset)["all"]
                print(
                    f"{args.dataset} {len(records)}/{len(instances)} "
                    f"E200={summary['gurobi_e200_mean']:.6f} "
                    f"E365={summary['self_e365_mean']:.6f} "
                    f"delta={summary['self_minus_gurobi']:+.6f} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
    finally:
        pool.close()
    print(f"COMPLETE {args.dataset} {result_path}", flush=True)


if __name__ == "__main__":
    main()
