from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from src.common.checkpoint import load_checkpoint, save_checkpoint
from src.common.config import AppConfig, save_config_snapshot
from src.data.dataset import iter_grouped_batches, load_dataset
from src.master.model import MasterPolicyModel
from src.master.rollout import rollout_batch
from src.pipeline.evaluator import RewardCache, evaluate_dataset


def _compute_batch_rewards(
    instances,
    rollout_result,
    config: AppConfig,
    cache: RewardCache,
    device: torch.device,
):
    sequence_lists = rollout_result.to_sequence_lists()
    batch_size = len(sequence_lists)
    pomo_size = len(sequence_lists[0]) if sequence_lists else 0
    rewards = torch.empty((batch_size, pomo_size), dtype=torch.float32, device=device)
    feasible_count = 0
    solver_calls = 0

    for batch_index, instance in enumerate(instances):
        for pomo_index, sequence in enumerate(sequence_lists[batch_index]):
            solution, from_cache = cache.get_or_solve(instance, sequence, config.solver)
            if not from_cache:
                solver_calls += 1
            if solution.success:
                rewards[batch_index, pomo_index] = -float(solution.objective)
                feasible_count += 1
            else:
                rewards[batch_index, pomo_index] = float(config.train.penalty_reward)
    feasible_ratio = feasible_count / max(batch_size * max(pomo_size, 1), 1)
    return rewards, feasible_ratio, solver_calls


def train_one_epoch(
    model: MasterPolicyModel,
    instances,
    optimizer: torch.optim.Optimizer,
    config: AppConfig,
    device: torch.device,
    cache: RewardCache,
    epoch: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_objective = 0.0
    total_feasible = 0.0
    total_batches = 0
    total_solver_calls = 0
    max_batches = config.train.max_batches_per_epoch

    for batch_instances in iter_grouped_batches(
        list(instances),
        batch_size=config.train.batch_size,
        shuffle=True,
        seed=config.runtime.seed + epoch,
    ):
        if max_batches is not None and total_batches >= max_batches:
            break
        rollout_result = rollout_batch(
            model=model,
            instances=batch_instances,
            device=device,
            decode_type="sample",
            use_pomo_start=True,
        )
        rewards, feasible_ratio, solver_calls = _compute_batch_rewards(
            batch_instances,
            rollout_result,
            config=config,
            cache=cache,
            device=device,
        )

        advantage = rewards - rewards.mean(dim=1, keepdim=True)
        loss = -(advantage * rollout_result.log_probs).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        optimizer.step()

        best_reward = rewards.max(dim=1).values
        total_loss += float(loss.item())
        total_objective += float((-best_reward).mean().item())
        total_feasible += feasible_ratio
        total_batches += 1
        total_solver_calls += solver_calls

    return {
        "epoch": float(epoch),
        "train_loss": total_loss / max(total_batches, 1),
        "train_objective": total_objective / max(total_batches, 1),
        "train_feasible_ratio": total_feasible / max(total_batches, 1),
        "solver_calls": float(total_solver_calls),
        "cache_hits": float(cache.hits),
        "cache_misses": float(cache.misses),
    }


def _append_metrics_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_training(config: AppConfig) -> dict[str, Any]:
    output_dir = Path(config.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config, output_dir / "resolved_config.yaml")

    device = torch.device(config.runtime.device)
    model = MasterPolicyModel(config.model).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )

    start_epoch = 1
    best_val = float("inf")
    if config.runtime.checkpoint_path and Path(config.runtime.checkpoint_path).exists():
        checkpoint = load_checkpoint(config.runtime.checkpoint_path, model, optimizer=optimizer, map_location=device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val = float(checkpoint.get("best_metric", float("inf")))
        logger.info("Resumed training from epoch {}", start_epoch)

    val_instances = load_dataset(
        config.data.dataset_dir,
        "val",
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        split_seed=config.data.split_seed,
    )
    train_instances = load_dataset(
        config.data.dataset_dir,
        "train",
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        split_seed=config.data.split_seed,
    )
    test_instances = load_dataset(
        config.data.dataset_dir,
        "test",
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        split_seed=config.data.split_seed,
    )
    cache = RewardCache()

    split_manifest = {
        "train_count": len(train_instances),
        "val_count": len(val_instances),
        "test_count": len(test_instances),
        "train_instances": [instance.instance_id for instance in train_instances],
        "val_instances": [instance.instance_id for instance in val_instances],
        "test_instances": [instance.instance_id for instance in test_instances],
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    summary: dict[str, Any] = {}
    metrics_path = output_dir / "metrics.csv"

    for epoch in range(start_epoch, config.train.epochs + 1):
        train_metrics = train_one_epoch(model, train_instances, optimizer, config, device, cache, epoch)
        log_payload = dict(train_metrics)

        if val_instances and epoch % config.train.validate_every == 0:
            model.eval()
            with torch.no_grad():
                _, val_summary = evaluate_dataset(
                    val_instances,
                    config=config,
                    model=model,
                    device=device,
                    cache=cache,
                )
            log_payload.update({f"val_{key}": value for key, value in val_summary.items()})
            current_val = float(val_summary["avg_objective"])
            if current_val < best_val:
                best_val = current_val
                save_checkpoint(
                    output_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_metric=best_val,
                    metrics=log_payload,
                )
                logger.info("Saved best checkpoint at epoch {} with avg objective {:.4f}", epoch, best_val)
        elif not val_instances:
            current_metric = float(train_metrics["train_objective"])
            if current_metric < best_val:
                best_val = current_metric
                save_checkpoint(
                    output_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_metric=best_val,
                    metrics=log_payload,
                )
                logger.info("Saved best checkpoint at epoch {} with train objective {:.4f}", epoch, best_val)

        if epoch % config.train.checkpoint_every == 0:
            save_checkpoint(
                output_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_val,
                metrics=log_payload,
            )

        _append_metrics_row(metrics_path, log_payload)
        logger.info(
            "Epoch {} | train_loss={:.4f} train_obj={:.4f} train_feas={:.3f}",
            epoch,
            log_payload["train_loss"],
            log_payload["train_objective"],
            log_payload["train_feasible_ratio"],
        )
        if "val_avg_objective" in log_payload:
            logger.info(
                "Epoch {} | val_obj={:.4f} val_feas={:.3f}",
                epoch,
                log_payload["val_avg_objective"],
                log_payload["val_feasible_ratio"],
            )
        summary = log_payload

    return {
        "output_dir": str(output_dir),
        "best_val_objective": best_val,
        "last_metrics": summary,
        "config": asdict(config),
    }
