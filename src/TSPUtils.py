from __future__ import annotations

import os
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from loguru import logger
from src.CVPSolver import SolverConfig
from src.TSPModel import ModelConfig


@dataclass
class DataConfig:
    dataset_dir: str = "instance/Data"
    split: str = "train"
    eval_split: str = "test"
    train_ratio: float = 0.8
    val_ratio: float = 0.0
    test_ratio: float = 0.2
    split_seed: int = 1234


@dataclass
class OptimizerConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-6


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 4
    grad_clip: float = 1.0
    penalty_reward: float = -1e6
    validate_every: int = 1
    checkpoint_every: int = 1
    max_batches_per_epoch: int | None = None


@dataclass
class DecodeConfig:
    decode_type: str = "greedy"
    num_candidates: int = 8
    sample_max_rollouts: int = 32


@dataclass
class RuntimeConfig:
    seed: int = 1234
    device: str = "cpu"
    output_dir: str = "outputs/real529_run"
    checkpoint_path: str = ""
    log_level: str = "INFO"


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> AppConfig:
    config = AppConfig()
    if path is None:
        return config

    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    merged = _deep_update(asdict(config), raw)
    return AppConfig(
        data=DataConfig(**merged["data"]),
        model=ModelConfig(**merged["model"]),
        optimizer=OptimizerConfig(**merged["optimizer"]),
        train=TrainConfig(**merged["train"]),
        decode=DecodeConfig(**merged["decode"]),
        solver=SolverConfig(**merged["solver"]),
        runtime=RuntimeConfig(**merged["runtime"]),
    )


def save_config_snapshot(config: AppConfig, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    best_metric: float | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_metric": best_metric,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, checkpoint_path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def setup_logger(output_dir: str | Path, level: str = "INFO"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level=level, enqueue=False, backtrace=False, diagnose=False)
    logger.add(
        output_path / "run.log",
        level=level,
        enqueue=False,
        backtrace=False,
        diagnose=False,
        rotation="10 MB",
    )
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
