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

import numpy as np
import torch

from src.CVPSolver import SolverConfig
from src.ImitationLearning import load_policy_checkpoint
from src.PseudoExpert import instance_from_dict, instance_to_dict
from src.RewardWorkerPool import RewardWorkerPool
from src.TSPEnv import rollout_batch
from src.TSPUtils import set_seed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact full-POMO evaluation on immutable checkpoint-selection instances."
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
    parser.add_argument("--fixed-validation-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 40, 60, 80, 100])
    parser.add_argument("--instances-per-size", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--include-free-start", action="store_true",
                        help="Also score the single greedy route used by fixed validation.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from completed batches in output-dir/results.json.")
    parser.add_argument("--cached-baseline-results", type=Path,
                        help="Reuse verified paired baseline scores from a complete identical benchmark.")
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


def _summarize(records: list[dict], labels: list[str], baseline_label: str,
               objective_key: str = "objectives") -> dict:
    baseline = np.asarray([row[objective_key][baseline_label] for row in records], dtype=float)
    result: dict[str, object] = {
        "num_instances": len(records),
        f"{baseline_label}_mean_makespan": float(baseline.mean()),
        f"{baseline_label}_feasible_ratio": float(
            np.mean([row["feasible"][baseline_label] for row in records])
        ),
    }
    for label in labels:
        values = np.asarray([row[objective_key][label] for row in records], dtype=float)
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


def _instance_fingerprint(instance) -> str:
    payload = instance_to_dict(instance)
    # Identity names are local to each generator; compare the actual problem data.
    payload.pop("instance_id", None)
    payload.pop("path", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _load_frozen_groups(directory: Path, sizes: list[int], per_size: int, seed: int):
    directory = directory.resolve()
    manifest_path = directory / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (
        manifest["base_seed"] != seed
        or manifest["sizes"] != sizes
        or manifest["per_size"] != per_size
        or manifest["count"] != len(sizes) * per_size
    ):
        raise ValueError("fixed validation manifest does not match requested dataset")
    groups = {}
    fingerprints = []
    for size in sizes:
        path = directory / f"J{size}.pt"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files_sha256"][path.name]:
            raise ValueError(f"fixed validation file changed: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            int(payload["base_seed"]) != seed
            or int(payload["group_seed"]) != seed + 1009 * size
            or int(payload["problem_size"]) != size
            or int(payload["count"]) != per_size
        ):
            raise ValueError(f"fixed validation metadata changed: {path}")
        instances = [instance_from_dict(row) for row in payload["instances"]]
        if len(instances) != per_size or any(instance.J != size for instance in instances):
            raise ValueError(f"fixed validation instances changed: {path}")
        groups[size] = instances
        fingerprints.extend(_instance_fingerprint(instance) for instance in instances)
    if (
        fingerprints != manifest["content_fingerprints"]
        or hashlib.sha256("\n".join(fingerprints).encode()).hexdigest()
        != manifest["content_fingerprints_sha256"]
    ):
        raise ValueError("fixed validation instance fingerprint mismatch")
    return groups, manifest, hashlib.sha256(manifest_bytes).hexdigest()


def _result_payload(params, records, labels, baseline_label, runtime, *, complete,
                    execution_history=None):
    groups = {}
    for row in records:
        groups.setdefault(str(row["problem_size"]), []).append(row)
    payload = {
        "params": params,
        "summary_by_size": {size: _summarize(group, labels, baseline_label)
                            for size, group in groups.items()},
        "overall": _summarize(records, labels, baseline_label),
        "runtime_seconds": runtime,
        "complete": complete,
        "records": records,
    }
    if execution_history is not None:
        payload["execution_history"] = execution_history
    if params["include_free_start"]:
        payload["free_start"] = {
            "summary_by_size": {
                size: _summarize(group, labels, baseline_label, "free_start_objectives")
                for size, group in groups.items()
            },
            "overall": _summarize(records, labels, baseline_label, "free_start_objectives"),
        }
    return payload


def main() -> None:
    args = _parser().parse_args()
    if args.instances_per_size <= 0 or args.batch_size <= 0 or args.workers <= 0:
        raise ValueError("instances-per-size, batch-size, and workers must be positive")
    if args.torch_threads <= 0:
        raise ValueError("torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
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
    fixed_groups, fixed_manifest, manifest_sha256 = _load_frozen_groups(
        args.fixed_validation_dir, sizes, args.instances_per_size, args.seed
    )

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
        "fixed_validation_dir": str(args.fixed_validation_dir.resolve()),
        "validation_manifest_sha256": manifest_sha256,
        "validation_content_sha256": fixed_manifest["content_fingerprints_sha256"],
        "device": str(device),
        "include_free_start": args.include_free_start,
        "torch_threads": args.torch_threads,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "checkpoint_sha256": {label: hashlib.sha256(path.read_bytes()).hexdigest()
                              for label, path in checkpoint_paths.items()},
    }
    cached_baseline = None
    if args.cached_baseline_results:
        cache_path = args.cached_baseline_results.resolve()
        cache_bytes = cache_path.read_bytes()
        cached = json.loads(cache_bytes)
        old = cached["params"]
        protocol_keys = ("sizes", "instances_per_size", "batch_size", "decode_type",
                         "augmentation", "pomo_size", "start_node_strategy", "seed",
                         "device", "include_free_start", "torch_threads", "torch_version",
                         "numpy_version", "solver_threads_per_worker",
                         "fixed_validation_dir", "validation_manifest_sha256",
                         "validation_content_sha256")
        if not cached["complete"] or any(old[k] != params[k] for k in protocol_keys):
            raise ValueError("cached baseline protocol is not complete or does not match")
        if old["checkpoint_sha256"].get(baseline_label) != params["checkpoint_sha256"][baseline_label]:
            raise ValueError("cached baseline checkpoint hash does not match")
        cached_baseline = {row["instance_fingerprint"]: row for row in cached["records"]}
        if len(cached_baseline) != len(sizes) * args.instances_per_size:
            raise ValueError("cached baseline must contain all unique test instances")
        for row in cached_baseline.values():
            if not row["feasible"][baseline_label] or not math.isfinite(row["objectives"][baseline_label]):
                raise ValueError("cached baseline contains invalid objectives")
            if args.include_free_start and not math.isfinite(row["free_start_objectives"][baseline_label]):
                raise ValueError("cached baseline contains invalid single-route objectives")
        params["cached_baseline_results"] = str(cache_path)
        params["cached_baseline_results_sha256"] = hashlib.sha256(cache_bytes).hexdigest()
        del models[baseline_label]
        print(f"Reusing {len(cached_baseline)} verified {baseline_label} baseline scores", flush=True)
    result_path = output_dir / "results.json"
    all_records: list[dict] = []
    previous_runtime = 0.0
    execution_history = [{"workers": args.workers, "start_instance": 0}]
    if result_path.exists():
        if not args.resume:
            raise FileExistsError(f"{result_path} exists; use --resume or a new output directory")
        previous = json.loads(result_path.read_text())
        old_protocol = {k: v for k, v in previous["params"].items() if k != "workers"}
        new_protocol = {k: v for k, v in params.items() if k != "workers"}
        if old_protocol != new_protocol:
            raise ValueError("resume parameters or checkpoint hashes do not match saved results")
        if previous["complete"]:
            print(f"Already complete: {result_path}")
            return
        all_records = previous["records"]
        previous_runtime = float(previous["runtime_seconds"])
        execution_history = previous.get("execution_history", [
            {"workers": previous["params"]["workers"], "start_instance": 0}
        ])
        if execution_history[-1]["workers"] != args.workers:
            execution_history.append({"workers": args.workers,
                                      "start_instance": len(all_records)})
        print(f"Resuming {len(all_records)} completed instances", flush=True)
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
    pool = RewardWorkerPool(args.workers, env_output_flag=0)
    try:
        for problem_size in sizes:
            fixed_instances = fixed_groups[problem_size]
            saved = [row for row in all_records if row["problem_size"] == problem_size]
            for row, instance in zip(saved, fixed_instances):
                if row["instance_fingerprint"] != _instance_fingerprint(instance):
                    raise ValueError("resumed fixed instance differs from saved data")
            completed = len(saved)
            while completed < args.instances_per_size:
                batch_started = time.perf_counter()
                current_batch = min(args.batch_size, args.instances_per_size - completed)
                instances = fixed_instances[completed : completed + current_batch]

                routes_by_label: dict[str, list[list[list[int]]]] = {}
                free_routes_by_label = {}
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
                        if args.include_free_start:
                            free_rollout = rollout_batch(
                                model, instances, device=device, decode_type="greedy",
                                use_pomo_start=False,
                            )
                            free_routes_by_label[label] = [
                                routes[0] for routes in free_rollout.to_sequence_lists()
                            ]

                unique_tasks: list[tuple[object, list[int], SolverConfig]] = []
                task_keys: list[tuple[str, tuple[int, ...]]] = []
                for instance_index, instance in enumerate(instances):
                    seen: set[tuple[int, ...]] = set()
                    for label in models:
                        sequences = list(routes_by_label[label][instance_index])
                        if args.include_free_start:
                            sequences.append(free_routes_by_label[label][instance_index])
                        for sequence in sequences:
                            key = tuple(int(node) for node in sequence)
                            if key in seen:
                                continue
                            seen.add(key)
                            task_keys.append((instance.instance_id, key))
                            unique_tasks.append((instance, list(key), solver_config))

                solved = pool.map(unique_tasks)
                for instance_id, sequence, solution in solved:
                    if (not solution.success or str(solution.status).strip() != "2"
                            or not math.isfinite(solution.objective)):
                        raise RuntimeError(
                            f"Non-optimal exact evaluation: {instance_id}, "
                            f"status={solution.status}, sequence={sequence}"
                        )
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
                    fingerprint = _instance_fingerprint(instance)
                    cached_row = None
                    if cached_baseline is not None:
                        cached_row = cached_baseline[fingerprint]
                        if cached_row["problem_size"] != problem_size:
                            raise ValueError("cached instance problem size does not match")
                        objectives[baseline_label] = cached_row["objectives"][baseline_label]
                        feasible[baseline_label] = cached_row["feasible"][baseline_label]
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
                        "instance_fingerprint": fingerprint,
                    }
                    if args.include_free_start:
                        row["free_start_objectives"] = {
                            label: objective_by_key[(instance.instance_id,
                                tuple(free_routes_by_label[label][instance_index]))][0]
                            for label in models
                        }
                        if cached_row is not None:
                            row["free_start_objectives"][baseline_label] = cached_row["free_start_objectives"][baseline_label]
                    for label in labels:
                        row[f"{label}_improvement_vs_{baseline_label}"] = float(
                            objectives[baseline_label] - objectives[label]
                        )
                    all_records.append(row)

                completed += current_batch
                elapsed = previous_runtime + time.perf_counter() - started
                partial = _result_payload(
                    params, all_records, labels, baseline_label, elapsed, complete=False,
                    execution_history=execution_history,
                )
                _write_json(result_path, partial)
                print(
                    f"J={problem_size:3d} {completed:4d}/{args.instances_per_size} "
                    f"unique_solves={len(unique_tasks):5d} elapsed={elapsed:.1f}s "
                    f"batch_seconds={time.perf_counter() - batch_started:.2f}",
                    flush=True,
                )
    finally:
        pool.close()

    payload = _result_payload(
        params, all_records, labels, baseline_label,
        previous_runtime + time.perf_counter() - started, complete=True,
        execution_history=execution_history,
    )
    _write_json(output_dir / "results.json", payload)
    print(json.dumps({"summary_by_size": payload["summary_by_size"],
                      "overall": payload["overall"]}, indent=2))
    print(f"results_path={output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
