from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from src.data.instance import CVTSPInstance, load_instance


@dataclass(frozen=True)
class SyntheticGeneratorConfig:
    real_dataset_dir: str = "instance/Data"
    output_dir: str = "instance/SyntheticTrain_20_100"
    num_instances: int = 2100
    seed: int = 1234
    target_counts: tuple[int, ...] = (20, 30, 40, 50, 60, 80, 100)
    min_target_count: int = 20
    max_target_count: int = 100
    prototype_prob: float = 0.7
    clustered_prob: float = 0.2
    ring_prob: float = 0.1
    jitter_sigma: float = 0.035
    scale_jitter_sigma: float = 0.08


@dataclass
class SyntheticTemplateBank:
    templates: list[CVTSPInstance]
    templates_by_j: dict[int, list[CVTSPInstance]]
    scale_by_j: dict[int, np.ndarray]
    available_js: list[int]
    normalized_targets_by_template: dict[str, np.ndarray]
    radial_samples: np.ndarray
    parameter_modes_by_j: dict[int, list[tuple[float, float, float]]]


def load_template_bank(config: SyntheticGeneratorConfig) -> SyntheticTemplateBank:
    real_root = Path(config.real_dataset_dir)
    templates = []
    templates_by_j: dict[int, list[CVTSPInstance]] = defaultdict(list)
    normalized_targets_by_template: dict[str, np.ndarray] = {}
    radial_samples = []

    for path in sorted(real_root.glob("Example_*.txt")):
        instance = load_instance(path)
        if not (config.min_target_count <= instance.J <= config.max_target_count):
            continue
        templates.append(instance)
        templates_by_j[instance.J].append(instance)
        normalized = (instance.targets - instance.depot[None, :]) / max(instance.scale, 1e-8)
        normalized_targets_by_template[instance.instance_id] = normalized.astype(np.float64)
        radial_samples.extend(np.linalg.norm(normalized, axis=1).tolist())

    if not templates:
        raise ValueError(
            f"no real templates found in {real_root} for J in "
            f"[{config.min_target_count}, {config.max_target_count}]"
        )

    scale_by_j = {
        j: np.asarray([instance.scale for instance in bucket], dtype=np.float64)
        for j, bucket in templates_by_j.items()
    }
    parameter_modes_by_j = {
        j: [(instance.carrier_speed, instance.uav_speed, instance.endurance) for instance in bucket]
        for j, bucket in templates_by_j.items()
    }

    return SyntheticTemplateBank(
        templates=templates,
        templates_by_j=dict(templates_by_j),
        scale_by_j=scale_by_j,
        available_js=sorted(templates_by_j.keys()),
        normalized_targets_by_template=normalized_targets_by_template,
        radial_samples=np.asarray(radial_samples, dtype=np.float64),
        parameter_modes_by_j=parameter_modes_by_j,
    )


def _nearest_reference_j(target_j: int, available_js: list[int]) -> int:
    return min(available_js, key=lambda value: (abs(value - target_j), value))


def _sample_target_count(config: SyntheticGeneratorConfig, rng: np.random.Generator) -> int:
    if config.target_counts:
        return int(rng.choice(np.asarray(config.target_counts, dtype=np.int64)))
    return int(rng.integers(config.min_target_count, config.max_target_count + 1))


def _sample_scale(target_j: int, template: CVTSPInstance, config: SyntheticGeneratorConfig, rng: np.random.Generator) -> float:
    scale = float(template.scale) * ((target_j + 1.0) / (template.J + 1.0)) ** 0.22
    scale *= float(np.exp(rng.normal(0.0, config.scale_jitter_sigma)))
    return max(scale, 10.0)


def _apply_random_transform(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    rot = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    transformed = points @ rot.T

    if rng.random() < 0.5:
        transformed[:, 0] *= -1.0
    if rng.random() < 0.5:
        transformed[:, 1] *= -1.0

    stretch = np.diag(rng.uniform(0.9, 1.12, size=2))
    transformed = transformed @ stretch
    return transformed


def _ensure_reasonable_scale(points: np.ndarray, target_scale: float) -> np.ndarray:
    max_radius = float(np.linalg.norm(points, axis=1).max(initial=1.0))
    if max_radius < 1e-8:
        return points
    return points * (target_scale / max_radius)


def _generate_from_template(
    target_j: int,
    template: CVTSPInstance,
    config: SyntheticGeneratorConfig,
    rng: np.random.Generator,
    bank: SyntheticTemplateBank,
) -> np.ndarray:
    normalized = bank.normalized_targets_by_template[template.instance_id]
    replace = normalized.shape[0] < target_j
    indices = rng.choice(normalized.shape[0], size=target_j, replace=replace)
    points = normalized[indices].copy()
    points += rng.normal(0.0, config.jitter_sigma, size=points.shape)
    points = _apply_random_transform(points, rng)
    scale = _sample_scale(target_j, template, config, rng)
    return _ensure_reasonable_scale(points, scale)


def _sample_cluster_centers(rng: np.random.Generator, num_clusters: int) -> np.ndarray:
    angles = rng.uniform(0.0, 2.0 * np.pi, size=num_clusters)
    radii = rng.uniform(0.25, 0.85, size=num_clusters)
    centers = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    return centers


def _generate_clustered_geometry(
    target_j: int,
    template: CVTSPInstance,
    config: SyntheticGeneratorConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    num_clusters = int(rng.integers(2, min(6, max(3, target_j // 10 + 2))))
    centers = _sample_cluster_centers(rng, num_clusters)
    cluster_ids = rng.integers(0, num_clusters, size=target_j)
    points = centers[cluster_ids] + rng.normal(0.0, 0.08, size=(target_j, 2))
    points = _apply_random_transform(points, rng)
    scale = _sample_scale(target_j, template, config, rng)
    return _ensure_reasonable_scale(points, scale)


def _generate_ring_geometry(
    target_j: int,
    template: CVTSPInstance,
    config: SyntheticGeneratorConfig,
    rng: np.random.Generator,
    bank: SyntheticTemplateBank,
) -> np.ndarray:
    angles = rng.uniform(0.0, 2.0 * np.pi, size=target_j)
    radial_indices = rng.choice(bank.radial_samples.shape[0], size=target_j, replace=True)
    base_r = np.clip(bank.radial_samples[radial_indices], 0.45, 1.0)
    radii = np.clip(base_r + rng.normal(0.0, 0.06, size=target_j), 0.1, 1.0)
    points = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    points = _apply_random_transform(points, rng)
    scale = _sample_scale(target_j, template, config, rng)
    return _ensure_reasonable_scale(points, scale)


def _generate_uniform_geometry(
    target_j: int,
    template: CVTSPInstance,
    config: SyntheticGeneratorConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    points = rng.uniform(-1.0, 1.0, size=(target_j, 2))
    points = _apply_random_transform(points, rng)
    scale = _sample_scale(target_j, template, config, rng)
    return _ensure_reasonable_scale(points, scale)


def _generate_target_coordinates(
    target_j: int,
    template: CVTSPInstance,
    config: SyntheticGeneratorConfig,
    rng: np.random.Generator,
    bank: SyntheticTemplateBank,
) -> np.ndarray:
    draw = rng.random()
    if draw < config.prototype_prob:
        return _generate_from_template(target_j, template, config, rng, bank)
    if draw < config.prototype_prob + config.clustered_prob:
        return _generate_clustered_geometry(target_j, template, config, rng)
    if draw < config.prototype_prob + config.clustered_prob + config.ring_prob:
        return _generate_ring_geometry(target_j, template, config, rng, bank)
    return _generate_uniform_geometry(target_j, template, config, rng)


def _sample_template_for_j(target_j: int, bank: SyntheticTemplateBank, rng: np.random.Generator) -> CVTSPInstance:
    reference_j = _nearest_reference_j(target_j, bank.available_js)
    candidates = bank.templates_by_j[reference_j]
    return candidates[int(rng.integers(0, len(candidates)))]


def _format_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") if "." in f"{value:.6f}" else f"{value:.6f}"


def serialize_instance_text(
    instance_id: str,
    depot: np.ndarray,
    targets: np.ndarray,
    carrier_speed: float,
    uav_speed: float,
    endurance: float,
) -> str:
    lines = [
        f"//Synthetic={instance_id}",
        "",
        f"J={targets.shape[0]}",
        f"MT={_format_float(endurance)}",
        f"Vd={_format_float(uav_speed)}",
        f"Vv={_format_float(carrier_speed)}",
        "",
        "p: position coordinates = ",
        f"{_format_float(float(depot[0]))} {_format_float(float(depot[1]))}",
    ]
    for point in targets:
        lines.append(f"{_format_float(float(point[0]))} {_format_float(float(point[1]))}")
    lines.append("")
    return "\n".join(lines)


def write_synthetic_instance(
    output_dir: str | Path,
    instance_id: str,
    depot: np.ndarray,
    targets: np.ndarray,
    carrier_speed: float,
    uav_speed: float,
    endurance: float,
) -> Path:
    output_path = Path(output_dir) / f"{instance_id}.txt"
    output_path.write_text(
        serialize_instance_text(
            instance_id=instance_id,
            depot=depot,
            targets=targets,
            carrier_speed=carrier_speed,
            uav_speed=uav_speed,
            endurance=endurance,
        ),
        encoding="utf-8",
    )
    return output_path


def generate_synthetic_dataset(config: SyntheticGeneratorConfig) -> dict[str, object]:
    rng = np.random.default_rng(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bank = load_template_bank(config)
    j_counter: Counter[int] = Counter()
    metadata_rows = []

    for index in range(1, config.num_instances + 1):
        target_j = _sample_target_count(config, rng)
        template = _sample_template_for_j(target_j, bank, rng)
        targets = _generate_target_coordinates(target_j, template, config, rng, bank)
        carrier_speed, uav_speed, endurance = template.carrier_speed, template.uav_speed, template.endurance
        depot = np.zeros(2, dtype=np.float64)
        instance_id = f"Synthetic_{index:05d}_J{target_j}"
        path = write_synthetic_instance(
            output_dir=output_dir,
            instance_id=instance_id,
            depot=depot,
            targets=targets,
            carrier_speed=carrier_speed,
            uav_speed=uav_speed,
            endurance=endurance,
        )
        metadata_rows.append(
            {
                "instance_id": instance_id,
                "path": str(path),
                "J": target_j,
                "template_id": template.instance_id,
                "carrier_speed": carrier_speed,
                "uav_speed": uav_speed,
                "endurance": endurance,
                "scale": float(np.max(np.linalg.norm(targets - depot[None, :], axis=1))),
            }
        )
        j_counter[target_j] += 1

    manifest = {
        "config": {
            **asdict(config),
            "target_counts": list(config.target_counts),
        },
        "num_instances": config.num_instances,
        "j_distribution": {str(key): int(value) for key, value in sorted(j_counter.items())},
        "source_real_dataset": config.real_dataset_dir,
        "output_dir": str(output_dir),
        "files": metadata_rows,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
