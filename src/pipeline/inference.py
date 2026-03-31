from __future__ import annotations

from pathlib import Path

import torch

from src.common.checkpoint import load_checkpoint
from src.common.config import AppConfig
from src.data.instance import CVTSPInstance, load_instance
from src.master.model import MasterPolicyModel
from src.pipeline.evaluator import EvaluationResult, RewardCache, evaluate_instance


def load_model_for_inference(config: AppConfig, device: torch.device) -> tuple[MasterPolicyModel, dict]:
    model = MasterPolicyModel(config.model).to(device)
    if not config.runtime.checkpoint_path:
        raise ValueError("checkpoint_path is required for inference")
    checkpoint = load_checkpoint(config.runtime.checkpoint_path, model, map_location=device)
    model.eval()
    return model, checkpoint


def run_inference(
    config: AppConfig,
    instance_path: str | Path,
    cache: RewardCache | None = None,
) -> tuple[CVTSPInstance, EvaluationResult]:
    device = torch.device(config.runtime.device)
    model, _ = load_model_for_inference(config, device)
    instance = load_instance(instance_path)
    with torch.no_grad():
        result = evaluate_instance(instance=instance, config=config, model=model, device=device, cache=cache)
    return instance, result

