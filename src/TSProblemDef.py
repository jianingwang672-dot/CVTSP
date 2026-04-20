from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import Any

import numpy as np


_NUM_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


@dataclass(frozen=True)
class CVTSPInstance:
    instance_id: str
    path: Path
    J: int
    depot: np.ndarray
    targets: np.ndarray
    carrier_speed: float
    uav_speed: float
    endurance: float
    scale: float

    @property
    def problem_size(self) -> int:
        return self.J

    @property
    def node_features(self) -> np.ndarray:
        coords = np.vstack([self.depot[None, :], self.targets]).astype(np.float32)
        rel = coords - self.depot[None, :]
        scale = max(float(self.scale), 1.0)
        rel = rel / scale
        dist_rel = np.linalg.norm(rel, axis=1, keepdims=True)
        is_depot = np.zeros((self.J + 1, 1), dtype=np.float32)
        is_depot[0, 0] = 1.0
        vc_over_vv = np.full(
            (self.J + 1, 1),
            self.carrier_speed / max(self.uav_speed, 1e-8),
            dtype=np.float32,
        )
        reach_ratio = np.full(
            (self.J + 1, 1),
            (self.uav_speed * self.endurance) / scale,
            dtype=np.float32,
        )
        return np.concatenate([rel, dist_rel, is_depot, vc_over_vv, reach_ratio], axis=1)


@dataclass(frozen=True)
class RandomInstanceStats:
    dataset_dir: str
    instance_count: int
    uav_speed_min: float
    uav_speed_max: float
    carrier_speed_min: float
    carrier_speed_max: float
    endurance_min: float
    endurance_max: float
    depot_x_min: float
    depot_x_max: float
    depot_y_min: float
    depot_y_max: float
    speed_scale: float
    endurance_scale: float

    @property
    def normalized_uav_speed_range(self) -> tuple[float, float]:
        return (self.uav_speed_min / self.speed_scale, self.uav_speed_max / self.speed_scale)

    @property
    def normalized_carrier_speed_range(self) -> tuple[float, float]:
        return (self.carrier_speed_min / self.speed_scale, self.carrier_speed_max / self.speed_scale)

    @property
    def normalized_endurance_range(self) -> tuple[float, float]:
        return (self.endurance_min / self.endurance_scale, self.endurance_max / self.endurance_scale)

    @property
    def normalized_depot_x_range(self) -> tuple[float, float]:
        return _safe_normalized_range(self.depot_x_min, self.depot_x_max)

    @property
    def normalized_depot_y_range(self) -> tuple[float, float]:
        return _safe_normalized_range(self.depot_y_min, self.depot_y_max)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_dir": self.dataset_dir,
            "instance_count": self.instance_count,
            "uav_speed_min": self.uav_speed_min,
            "uav_speed_max": self.uav_speed_max,
            "carrier_speed_min": self.carrier_speed_min,
            "carrier_speed_max": self.carrier_speed_max,
            "endurance_min": self.endurance_min,
            "endurance_max": self.endurance_max,
            "depot_x_min": self.depot_x_min,
            "depot_x_max": self.depot_x_max,
            "depot_y_min": self.depot_y_min,
            "depot_y_max": self.depot_y_max,
            "speed_scale": self.speed_scale,
            "endurance_scale": self.endurance_scale,
            "normalized_uav_speed_range": self.normalized_uav_speed_range,
            "normalized_carrier_speed_range": self.normalized_carrier_speed_range,
            "normalized_endurance_range": self.normalized_endurance_range,
            "normalized_depot_x_range": self.normalized_depot_x_range,
            "normalized_depot_y_range": self.normalized_depot_y_range,
        }


def _safe_normalized_range(min_value: float, max_value: float) -> tuple[float, float]:
    scale = max(float(max_value), 1e-8)
    return (float(min_value) / scale, float(max_value) / scale)


def parse_instance_text(text: str, path: str | Path = "") -> CVTSPInstance:
    instance_path = Path(path) if path else Path("<memory>")

    J = int(re.search(r"\bJ\s*=\s*(\d+)", text).group(1))
    endurance = float(re.search(r"\bMT\s*=\s*(" + _NUM_RE + r")", text).group(1))
    uav_speed = float(re.search(r"\bVd\s*=\s*(" + _NUM_RE + r")", text).group(1))
    carrier_speed = float(re.search(r"\bVv\s*=\s*(" + _NUM_RE + r")", text).group(1))

    match = re.search(r"p:\s*position coordinates\s*=\s*([\s\S]+)", text, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"{instance_path}: missing coordinate block")

    coords = []
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^[A-Za-z]", stripped):
            break
        parts = stripped.split()
        if len(parts) >= 2 and re.match(r"^[\+\-]?\d", parts[0]):
            coords.append([float(parts[0]), float(parts[1])])

    points = np.asarray(coords, dtype=np.float32)
    if points.shape[0] != J + 1:
        raise ValueError(f"{instance_path}: expected {J + 1} points, got {points.shape[0]}")

    depot = points[0]
    targets = points[1:]
    scale = float(np.max(np.linalg.norm(targets - depot[None, :], axis=1)))
    instance_id = instance_path.stem if instance_path.name else "memory_instance"

    return CVTSPInstance(
        instance_id=instance_id,
        path=instance_path,
        J=J,
        depot=depot,
        targets=targets,
        carrier_speed=carrier_speed,
        uav_speed=uav_speed,
        endurance=endurance,
        scale=max(scale, 1.0),
    )


def load_instance(path: str | Path) -> CVTSPInstance:
    instance_path = Path(path)
    text = instance_path.read_text(encoding="utf-8", errors="ignore")
    return parse_instance_text(text, instance_path)


def augment_instance_by_8_fold(instance: CVTSPInstance) -> list[CVTSPInstance]:
    coords = np.vstack([instance.depot[None, :], instance.targets]).astype(np.float32)
    rel = coords - instance.depot[None, :]

    x = rel[:, [0]]
    y = rel[:, [1]]

    variants = [
        np.concatenate((x, y), axis=1),
        np.concatenate((-x, y), axis=1),
        np.concatenate((x, -y), axis=1),
        np.concatenate((-x, -y), axis=1),
        np.concatenate((y, x), axis=1),
        np.concatenate((-y, x), axis=1),
        np.concatenate((y, -x), axis=1),
        np.concatenate((-y, -x), axis=1),
    ]

    augmented_instances: list[CVTSPInstance] = []
    for aug_index, rel_coords in enumerate(variants):
        aug_coords = (rel_coords + instance.depot[None, :]).astype(np.float32)
        augmented_instances.append(
            CVTSPInstance(
                instance_id=f"{instance.instance_id}__aug{aug_index}",
                path=instance.path,
                J=instance.J,
                depot=aug_coords[0],
                targets=aug_coords[1:],
                carrier_speed=instance.carrier_speed,
                uav_speed=instance.uav_speed,
                endurance=instance.endurance,
                scale=instance.scale,
            )
        )

    return augmented_instances


def _example_index(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"unable to parse instance index from {path}")
    return int(match.group(1))


def discover_instance_paths(dataset_dir: str | Path) -> list[Path]:
    root = Path(dataset_dir)
    candidates = list(root.glob("Example_*.txt"))
    return sorted(candidates, key=_example_index)


def compute_random_instance_stats(dataset_dir: str | Path = "instance/Data") -> RandomInstanceStats:
    instances = [load_instance(path) for path in discover_instance_paths(dataset_dir)]
    if not instances:
        raise ValueError(f"no real instances found in {dataset_dir}")

    uav_speeds = [instance.uav_speed for instance in instances]
    carrier_speeds = [instance.carrier_speed for instance in instances]
    endurances = [instance.endurance for instance in instances]
    depot_x = [float(instance.depot[0]) for instance in instances]
    depot_y = [float(instance.depot[1]) for instance in instances]

    return RandomInstanceStats(
        dataset_dir=str(dataset_dir),
        instance_count=len(instances),
        uav_speed_min=min(uav_speeds),
        uav_speed_max=max(uav_speeds),
        carrier_speed_min=min(carrier_speeds),
        carrier_speed_max=max(carrier_speeds),
        endurance_min=min(endurances),
        endurance_max=max(endurances),
        depot_x_min=min(depot_x),
        depot_x_max=max(depot_x),
        depot_y_min=min(depot_y),
        depot_y_max=max(depot_y),
        speed_scale=max(max(uav_speeds), max(carrier_speeds), 1.0),
        endurance_scale=max(max(endurances), 1.0),
    )


class RandomCVTSPGenerator:
    def __init__(
        self,
        dataset_dir: str | Path = "instance/Data",
        min_problem_size: int = 20,
        max_problem_size: int = 100,
        problem_sizes: list[int] | tuple[int, ...] | None = None,
        seed: int = 1234,
    ):
        if problem_sizes:
            normalized = tuple(sorted({int(size) for size in problem_sizes}))
            if any(size <= 0 for size in normalized):
                raise ValueError("problem_sizes must be positive")
            self.problem_sizes = normalized
            min_problem_size = min(normalized)
            max_problem_size = max(normalized)
        else:
            self.problem_sizes = None
        if min_problem_size <= 0 or max_problem_size < min_problem_size:
            raise ValueError("invalid problem size range")
        self.dataset_dir = str(dataset_dir)
        self.min_problem_size = int(min_problem_size)
        self.max_problem_size = int(max_problem_size)
        self.stats = compute_random_instance_stats(dataset_dir)
        self.rng = np.random.default_rng(seed)
        self.instance_counter = 0

    def sample_problem_size(self, problem_size: int | None = None) -> int:
        if problem_size is not None:
            problem_size = int(problem_size)
            if self.problem_sizes is not None:
                if problem_size not in self.problem_sizes:
                    raise ValueError(f"problem_size must be one of {list(self.problem_sizes)}, got {problem_size}")
            elif not (self.min_problem_size <= problem_size <= self.max_problem_size):
                raise ValueError(
                    f"problem_size must be in [{self.min_problem_size}, {self.max_problem_size}], got {problem_size}"
                )
            return problem_size
        if self.problem_sizes is not None:
            index = int(self.rng.integers(0, len(self.problem_sizes)))
            return int(self.problem_sizes[index])
        return int(self.rng.integers(self.min_problem_size, self.max_problem_size + 1))

    def _sample_uniform(self, low: float, high: float) -> float:
        if abs(high - low) < 1e-12:
            return float(low)
        return float(self.rng.uniform(low, high))

    def _next_instance_id(self, J: int) -> str:
        self.instance_counter += 1
        return f"Random_{self.instance_counter:06d}_J{J}"

    def sample_instance(self, problem_size: int | None = None, instance_id: str | None = None) -> CVTSPInstance:
        J = self.sample_problem_size(problem_size)
        instance_name = instance_id or self._next_instance_id(J)

        depot = np.asarray(
            [
                self._sample_uniform(*self.stats.normalized_depot_x_range),
                self._sample_uniform(*self.stats.normalized_depot_y_range),
            ],
            dtype=np.float32,
        )
        targets = self.rng.random((J, 2)).astype(np.float32)
        carrier_speed = self._sample_uniform(*self.stats.normalized_carrier_speed_range)
        uav_speed = self._sample_uniform(*self.stats.normalized_uav_speed_range)
        endurance = self._sample_uniform(*self.stats.normalized_endurance_range)
        scale = float(np.max(np.linalg.norm(targets - depot[None, :], axis=1)))

        return CVTSPInstance(
            instance_id=instance_name,
            path=Path(f"<generated:{instance_name}>"),
            J=J,
            depot=depot,
            targets=targets,
            carrier_speed=carrier_speed,
            uav_speed=uav_speed,
            endurance=endurance,
            scale=max(scale, 1.0),
        )

    def sample_batch(self, batch_size: int, problem_size: int | None = None) -> list[CVTSPInstance]:
        J = self.sample_problem_size(problem_size)
        return [self.sample_instance(problem_size=J) for _ in range(batch_size)]


def get_random_instances(
    batch_size: int,
    problem_size: int | None = None,
    dataset_dir: str | Path = "instance/Data",
    min_problem_size: int = 20,
    max_problem_size: int = 100,
    problem_sizes: list[int] | tuple[int, ...] | None = None,
    seed: int = 1234,
) -> list[CVTSPInstance]:
    generator = RandomCVTSPGenerator(
        dataset_dir=dataset_dir,
        min_problem_size=min_problem_size,
        max_problem_size=max_problem_size,
        problem_sizes=problem_sizes,
        seed=seed,
    )
    return generator.sample_batch(batch_size=batch_size, problem_size=problem_size)


def get_random_problems(
    batch_size: int,
    problem_size: int | None = None,
    dataset_dir: str | Path = "instance/Data",
    min_problem_size: int = 20,
    max_problem_size: int = 100,
    problem_sizes: list[int] | tuple[int, ...] | None = None,
    seed: int = 1234,
) -> list[CVTSPInstance]:
    return get_random_instances(
        batch_size=batch_size,
        problem_size=problem_size,
        dataset_dir=dataset_dir,
        min_problem_size=min_problem_size,
        max_problem_size=max_problem_size,
        problem_sizes=problem_sizes,
        seed=seed,
    )


def format_instance_text(instance: CVTSPInstance) -> str:
    lines = [
        f"J = {instance.J}",
        f"MT = {instance.endurance:.6f}",
        f"Vd = {instance.uav_speed:.6f}",
        f"Vv = {instance.carrier_speed:.6f}",
        "p: position coordinates =",
        f"{float(instance.depot[0]):.6f} {float(instance.depot[1]):.6f}",
    ]
    for point in instance.targets:
        lines.append(f"{float(point[0]):.6f} {float(point[1]):.6f}")
    return "\n".join(lines) + "\n"


def save_random_dataset(
    output_dir: str | Path,
    num_instances: int,
    dataset_dir: str | Path = "instance/Data",
    min_problem_size: int = 20,
    max_problem_size: int = 100,
    problem_sizes: list[int] | tuple[int, ...] | None = None,
    seed: int = 1234,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    generator = RandomCVTSPGenerator(
        dataset_dir=dataset_dir,
        min_problem_size=min_problem_size,
        max_problem_size=max_problem_size,
        problem_sizes=problem_sizes,
        seed=seed,
    )

    counts_by_j: dict[int, int] = defaultdict(int)
    for _ in range(num_instances):
        instance = generator.sample_instance()
        counts_by_j[instance.J] += 1
        (output_root / f"{instance.instance_id}.txt").write_text(
            format_instance_text(instance),
            encoding="utf-8",
        )

    manifest = {
        "num_instances": num_instances,
        "dataset_dir": str(dataset_dir),
        "min_problem_size": min_problem_size,
        "max_problem_size": max_problem_size,
        "problem_sizes": list(problem_sizes) if problem_sizes is not None else None,
        "seed": seed,
        "stats": generator.stats.to_dict(),
        "counts_by_j": dict(sorted(counts_by_j.items())),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


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
