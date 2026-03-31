from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

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

