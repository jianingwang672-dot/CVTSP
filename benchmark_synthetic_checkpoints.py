from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from src.CVPSolver import SolverConfig
from src.ImitationLearning import load_policy_checkpoint
from src.RewardWorkerPool import RewardWorkerPool
from src.TSPEnv import rollout_batch
from src.TSProblemDef import RandomCVTSPGenerator
from src.TSPUtils import set_seed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired greedy evaluation of checkpoints on fixed synthetic CVTSP instances."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--baseline-label", default="epoch275")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Comparison model as LABEL=CHECKPOINT; may be repeated.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-dir", default="instance/Data")
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 40, 60, 80, 100])
    parser.add_argument("--instances-per-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260827)
    return parser


def _parse_models(values: list[str], baseline_label: str) -> dict[str, Path]:
    models: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--model must be LABEL=CHECKPOINT, got {value!r}")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        path = Path(raw_path).expanduser().resolve()
        if not label or label == baseline_label:
            raise ValueError(f"comparison label must be nonempty and differ from {baseline_label}")
        if label in models:
            raise ValueError(f"duplicate model label: {label}")
        if not path.is_file():
            raise FileNotFoundError(path)
        models[label] = path
    return models


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _summarize(records: list[dict], labels: list[str], baseline_label: str) -> dict:
    baseline = np.asarray([row["objectives"][baseline_label] for row in records], dtype=float)
    result: dict[str, object] = {
        "num_instances": len(records),
        f"{baseline_label}_mean_makespan": float(baseline.mean()),
        f"{baseline_label}_feasible_ratio": float(
            np.mean([row["feasible"][baseline_label] for row in records])
        ),
    }
    for label in labels:
        values = np.asarray([row["objectives"][label] for row in records], dtype=float)
        improvement = baseline - values
        paired_percent = 100.0 * improvement / np.maximum(np.abs(baseline), 1e-12)
        result[label] = {
            "mean_makespan": float(values.mean()),
            f"mean_improvement_vs_{baseline_label}": float(improvement.mean()),
            "relative_improvement_of_means_percent": float(
                100.0 * improvement.mean() / max(abs(float(baseline.mean())), 1e-12)
            ),
            "mean_paired_improvement_percent": float(paired_percent.mean()),
            "wins": int(np.sum(improvement > 1e-6)),
            "ties": int(np.sum(np.abs(improvement) <= 1e-6)),
            "losses": int(np.sum(improvement < -1e-6)),
            "feasible_ratio": float(np.mean([row["feasible"][label] for row in records])),
        }
    return result


def main() -> None:
    args = _parser().parse_args()
    if args.instances_per_size <= 0 or args.batch_size <= 0 or args.workers <= 0:
        raise ValueError("instances-per-size, batch-size, and workers must be positive")
    sizes = sorted({int(size) for size in args.sizes})
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("sizes must contain positive integers")

    baseline_path = Path(args.baseline).expanduser().resolve()
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)
    baseline_label = args.baseline_label.strip()
    if not baseline_label:
        raise ValueError("--baseline-label must be nonempty")
    comparison_paths = _parse_models(args.model, baseline_label)
    checkpoint_paths = {baseline_label: baseline_path, **comparison_paths}
    labels = list(comparison_paths)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    models = {}
    checkpoint_epochs = {}
    for label, checkpoint_path in checkpoint_paths.items():
        model, checkpoint = load_policy_checkpoint(checkpoint_path, device=device)
        model.config = replace(
            model.config,
            max_pomo_size=max(sizes),
            pomo_divisor=None,
            start_node_strategy="nearest_depot",
        )
        model.eval()
        models[label] = model
        checkpoint_epochs[label] = int(checkpoint.get("epoch", -1))

    params = {
        "checkpoints": {label: str(path) for label, path in checkpoint_paths.items()},
        "baseline_label": baseline_label,
        "checkpoint_epochs": checkpoint_epochs,
        "sizes": sizes,
        "instances_per_size": args.instances_per_size,
        "batch_size": args.batch_size,
        "decode_type": "greedy",
        "augmentation": False,
        "pomo_size": "J",
        "start_node_strategy": "nearest_depot",
        "workers": args.workers,
        "solver_threads_per_worker": 1,
        "seed": args.seed,
        "device": str(device),
    }
    _write_json(output_dir / "params.json", params)

    solver_config = SolverConfig(
        solver_backend="gurobi",
        gurobi_threads=1,
        output_flag=0,
        gurobi_params={
            "FeasibilityTol": 1e-8,
            "OptimalityTol": 1e-8,
            "NumericFocus": 2,
        },
        gurobi_explicit_qcp=True,
    )

    started = time.perf_counter()
    all_records: list[dict] = []
    by_size: dict[str, dict] = {}
    pool = RewardWorkerPool(args.workers, env_output_flag=0)
    try:
        for problem_size in sizes:
            generator = RandomCVTSPGenerator(
                dataset_dir=args.dataset_dir,
                min_problem_size=problem_size,
                max_problem_size=problem_size,
                problem_sizes=[problem_size],
                seed=args.seed + problem_size,
            )
            size_records: list[dict] = []
            completed = 0
            while completed < args.instances_per_size:
                current_batch = min(args.batch_size, args.instances_per_size - completed)
                instances = generator.sample_batch(current_batch, problem_size=problem_size)

                routes_by_label: dict[str, list[list[list[int]]]] = {}
                with torch.no_grad():
                    for label, model in models.items():
                        rollout = rollout_batch(
                            model,
                            instances,
                            device=device,
                            decode_type="greedy",
                            use_pomo_start=True,
                        )
                        routes_by_label[label] = rollout.to_sequence_lists()

                unique_tasks: list[tuple[object, list[int], SolverConfig]] = []
                task_keys: list[tuple[str, tuple[int, ...]]] = []
                for instance_index, instance in enumerate(instances):
                    seen: set[tuple[int, ...]] = set()
                    for label in models:
                        for sequence in routes_by_label[label][instance_index]:
                            key = tuple(int(node) for node in sequence)
                            if key in seen:
                                continue
                            seen.add(key)
                            task_keys.append((instance.instance_id, key))
                            unique_tasks.append((instance, list(key), solver_config))

                solved = pool.map(unique_tasks)
                objective_by_key = {
                    (instance_id, sequence): (
                        float(solution.objective) if solution.success else math.inf,
                        bool(solution.success),
                    )
                    for instance_id, sequence, solution in solved
                }

                for instance_index, instance in enumerate(instances):
                    objectives: dict[str, float] = {}
                    feasible: dict[str, bool] = {}
                    for label in models:
                        candidates = [
                            objective_by_key[(instance.instance_id, tuple(sequence))]
                            for sequence in routes_by_label[label][instance_index]
                        ]
                        best_objective, best_feasible = min(candidates, key=lambda item: item[0])
                        objectives[label] = float(best_objective)
                        feasible[label] = bool(best_feasible and math.isfinite(best_objective))
                    row = {
                        "instance_id": instance.instance_id,
                        "problem_size": problem_size,
                        "objectives": objectives,
                        "feasible": feasible,
                    }
                    for label in labels:
                        row[f"{label}_improvement_vs_{baseline_label}"] = float(
                            objectives[baseline_label] - objectives[label]
                        )
                    size_records.append(row)

                completed += current_batch
                elapsed = time.perf_counter() - started
                print(
                    f"J={problem_size:3d} {completed:4d}/{args.instances_per_size} "
                    f"unique_solves={len(unique_tasks):5d} elapsed={elapsed:.1f}s",
                    flush=True,
                )

            all_records.extend(size_records)
            by_size[str(problem_size)] = _summarize(size_records, labels, baseline_label)
            partial = {
                "params": params,
                "summary_by_size": by_size,
                "overall": _summarize(all_records, labels, baseline_label),
                "runtime_seconds": time.perf_counter() - started,
                "complete": False,
                "records": all_records,
            }
            _write_json(output_dir / "results.json", partial)
    finally:
        pool.close()

    payload = {
        "params": params,
        "summary_by_size": by_size,
        "overall": _summarize(all_records, labels, baseline_label),
        "runtime_seconds": time.perf_counter() - started,
        "complete": True,
        "records": all_records,
    }
    _write_json(output_dir / "results.json", payload)
    print(json.dumps({"summary_by_size": by_size, "overall": payload["overall"]}, indent=2))
    print(f"results_path={output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
