from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from src.PseudoExpert import instance_from_dict, instance_to_dict
from src.TSProblemDef import CVTSPInstance, RandomCVTSPGenerator


FIXED_VALIDATION_SCHEMA_VERSION = 1


def validation_fingerprint(instances: Iterable[CVTSPInstance]) -> str:
    payload = [instance_to_dict(instance) for instance in instances]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_group(
    instances: list[CVTSPInstance],
    *,
    problem_size: int,
    per_size: int,
    path: Path,
) -> None:
    if len(instances) != per_size:
        raise ValueError(
            f"{path} must contain {per_size} instances, found {len(instances)}"
        )
    if any(instance.J != problem_size for instance in instances):
        raise ValueError(f"{path} contains an instance whose J is not {problem_size}")


def load_or_create_fixed_validation_sets(
    *,
    dataset_dir: str | Path,
    validation_dir: str | Path,
    problem_sizes: Iterable[int] = (20, 40, 60, 80, 100),
    per_size: int = 200,
    seed: int = 24610,
) -> dict[int, list[CVTSPInstance]]:
    """Load immutable fixed validation files, creating only missing files once."""

    sizes = tuple(int(size) for size in problem_sizes)
    if not sizes or len(set(sizes)) != len(sizes) or any(size <= 0 for size in sizes):
        raise ValueError("problem_sizes must be distinct positive integers")
    if per_size <= 0:
        raise ValueError("per_size must be positive")
    root = Path(validation_dir)
    root.mkdir(parents=True, exist_ok=True)
    groups: dict[int, list[CVTSPInstance]] = {}

    for problem_size in sizes:
        path = root / f"J{problem_size}.pt"
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if int(payload.get("schema_version", -1)) != FIXED_VALIDATION_SCHEMA_VERSION:
                raise ValueError(f"unsupported fixed validation schema in {path}")
            if int(payload.get("problem_size", -1)) != problem_size:
                raise ValueError(f"fixed validation size metadata mismatch in {path}")
            instances = [instance_from_dict(item) for item in payload.get("instances", [])]
            _validate_group(
                instances,
                problem_size=problem_size,
                per_size=per_size,
                path=path,
            )
            groups[problem_size] = instances
            continue

        group_seed = int(seed) + 1009 * problem_size
        generator = RandomCVTSPGenerator(
            dataset_dir=dataset_dir,
            min_problem_size=problem_size,
            max_problem_size=problem_size,
            problem_sizes=(problem_size,),
            seed=group_seed,
        )
        instances = [
            generator.sample_instance(problem_size=problem_size) for _ in range(per_size)
        ]
        _validate_group(
            instances,
            problem_size=problem_size,
            per_size=per_size,
            path=path,
        )
        payload = {
            "schema_version": FIXED_VALIDATION_SCHEMA_VERSION,
            "base_seed": int(seed),
            "group_seed": group_seed,
            "problem_size": problem_size,
            "count": per_size,
            "instances": [instance_to_dict(instance) for instance in instances],
        }
        payload["fingerprint"] = validation_fingerprint(instances)
        torch.save(payload, path)
        groups[problem_size] = instances

    return groups


def flatten_fixed_validation_sets(
    groups: dict[int, list[CVTSPInstance]],
) -> list[CVTSPInstance]:
    return [instance for size in sorted(groups) for instance in groups[size]]


def summarize_fixed_validation_sets(
    groups: dict[int, list[CVTSPInstance]],
) -> dict:
    all_instances = flatten_fixed_validation_sets(groups)
    return {
        "problem_sizes": sorted(groups),
        "counts": {str(size): len(groups[size]) for size in sorted(groups)},
        "total": len(all_instances),
        "fingerprint": validation_fingerprint(all_instances),
        "shapes": {
            str(size): {
                "instances": len(groups[size]),
                "targets": [len(groups[size]), size, 2],
            }
            for size in sorted(groups)
        },
    }


__all__ = [
    "FIXED_VALIDATION_SCHEMA_VERSION",
    "flatten_fixed_validation_sets",
    "load_or_create_fixed_validation_sets",
    "summarize_fixed_validation_sets",
    "validation_fingerprint",
]
