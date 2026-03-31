from __future__ import annotations

import random
import re
from collections import defaultdict
from math import floor
from pathlib import Path

from src.data.instance import CVTSPInstance, load_instance


def _example_index(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"unable to parse instance index from {path}")
    return int(match.group(1))


def discover_instance_paths(dataset_dir: str | Path) -> list[Path]:
    root = Path(dataset_dir)
    candidates = list(root.glob("Example_*.txt")) + list(root.glob("Synthetic_*.txt"))
    return sorted(candidates, key=_example_index)


def _compute_bucket_split_counts(
    bucket_size: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    if bucket_size <= 0:
        return 0, 0, 0

    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("split ratios must sum to a positive value")

    train_ratio = train_ratio / total_ratio
    val_ratio = val_ratio / total_ratio
    test_ratio = test_ratio / total_ratio

    train_count = floor(bucket_size * train_ratio)
    val_count = floor(bucket_size * val_ratio)
    test_count = bucket_size - train_count - val_count

    if test_ratio > 0 and test_count == 0 and bucket_size > 1:
        if train_count > 1:
            train_count -= 1
        elif val_count > 0:
            val_count -= 1
        test_count = 1

    if train_ratio > 0 and train_count == 0 and bucket_size > 1:
        if test_count > 1:
            test_count -= 1
        elif val_count > 0:
            val_count -= 1
        train_count = 1

    while train_count + val_count + test_count < bucket_size:
        train_count += 1
    while train_count + val_count + test_count > bucket_size:
        if test_count > 0:
            test_count -= 1
        elif val_count > 0:
            val_count -= 1
        else:
            train_count -= 1

    return train_count, val_count, test_count


def build_splits(
    dataset_dir: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.0,
    test_ratio: float = 0.2,
    split_seed: int = 1234,
) -> dict[str, list[CVTSPInstance]]:
    grouped: dict[int, list[CVTSPInstance]] = defaultdict(list)
    for path in discover_instance_paths(dataset_dir):
        instance = load_instance(path)
        grouped[instance.J].append(instance)

    splits = {"train": [], "val": [], "test": [], "all": []}
    rng = random.Random(split_seed)
    for _, instances in sorted(grouped.items()):
        ordered = sorted(instances, key=lambda item: _example_index(item.path))
        rng.shuffle(ordered)
        train_count, val_count, test_count = _compute_bucket_split_counts(
            len(ordered),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        splits["train"].extend(ordered[:train_count])
        splits["val"].extend(ordered[train_count : train_count + val_count])
        splits["test"].extend(ordered[train_count + val_count : train_count + val_count + test_count])
        splits["all"].extend(ordered)
    return splits


def load_dataset(
    dataset_dir: str | Path,
    split: str = "all",
    train_ratio: float = 0.8,
    val_ratio: float = 0.0,
    test_ratio: float = 0.2,
    split_seed: int = 1234,
) -> list[CVTSPInstance]:
    splits = build_splits(
        dataset_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_seed=split_seed,
    )
    if split not in splits:
        raise ValueError(f"unknown split '{split}', available: {sorted(splits)}")
    return list(splits[split])


def group_instances_by_size(instances: list[CVTSPInstance]) -> dict[int, list[CVTSPInstance]]:
    grouped: dict[int, list[CVTSPInstance]] = defaultdict(list)
    for instance in instances:
        grouped[instance.J].append(instance)
    return dict(sorted(grouped.items()))


def iter_grouped_batches(
    instances: list[CVTSPInstance],
    batch_size: int,
    shuffle: bool = False,
    seed: int = 0,
):
    rng = random.Random(seed)
    grouped = group_instances_by_size(instances)
    problem_sizes = list(grouped.keys())
    if shuffle:
        rng.shuffle(problem_sizes)

    for problem_size in problem_sizes:
        bucket = list(grouped[problem_size])
        if shuffle:
            rng.shuffle(bucket)
        for start in range(0, len(bucket), batch_size):
            yield bucket[start : start + batch_size]
