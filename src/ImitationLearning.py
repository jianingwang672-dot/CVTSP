from __future__ import annotations

import copy
import csv
import json
import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import torch

from src.PseudoExpert import PseudoExpertRecord
from src.TSPEnv import TSPEnv
from src.TSPModel import ModelConfig, TSPModel, sample_block_assignment
from src.TSPUtils import set_seed


@dataclass(frozen=True)
class ImitationTrainingConfig:
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    seed: int = 1234
    shuffle: bool = True


@dataclass(frozen=True)
class ImitationMetrics:
    nll: float
    token_accuracy: float
    sequence_accuracy: float
    records: int
    tokens: int

    def to_dict(self) -> dict:
        return asdict(self)


def infer_model_config_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    fallback: ModelConfig | None = None,
) -> ModelConfig:
    fallback = fallback or ModelConfig()
    embedding_weight = state_dict["encoder.embedding.weight"]
    embedding_dim, node_feature_dim = embedding_weight.shape
    layer_indices = {
        int(match.group(1))
        for key in state_dict
        if (match := re.match(r"encoder\.layers\.(\d+)\.", key))
    }
    layer_count = max(layer_indices) + 1 if layer_indices else fallback.encoder_layer_num
    qkv_total = int(state_dict["encoder.layers.0.wq.weight"].shape[0])
    head_num = fallback.head_num if qkv_total % fallback.head_num == 0 else 1
    qkv_dim = qkv_total // head_num
    ff_hidden_dim = int(state_dict["encoder.layers.0.feed_forward.w1.weight"].shape[0])
    return ModelConfig(
        node_feature_dim=int(node_feature_dim),
        embedding_dim=int(embedding_dim),
        encoder_layer_num=int(layer_count),
        qkv_dim=int(qkv_dim),
        head_num=int(head_num),
        logit_clipping=float(fallback.logit_clipping),
        ff_hidden_dim=int(ff_hidden_dim),
        activation=str(fallback.activation),
        dropout=float(fallback.dropout),
        block_size=int(fallback.block_size),
        max_pomo_size=int(fallback.max_pomo_size),
        pomo_divisor=fallback.pomo_divisor,
        start_node_strategy=fallback.start_node_strategy,
    )


def load_policy_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    max_pomo_size: int | None = None,
    start_node_strategy: str | None = None,
) -> tuple[TSPModel, dict]:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    raw_config = checkpoint.get("model_config")
    if raw_config:
        allowed = set(ModelConfig.__dataclass_fields__)
        config = ModelConfig(**{key: value for key, value in raw_config.items() if key in allowed})
    else:
        config = infer_model_config_from_state_dict(state_dict)
    if max_pomo_size is not None:
        config = replace(config, max_pomo_size=int(max_pomo_size))
    if start_node_strategy is not None:
        config = replace(config, start_node_strategy=str(start_node_strategy))
    model = TSPModel(config).to(device)
    model.load_state_dict(state_dict, strict=True)
    return model, checkpoint


def _teacher_forcing_batch(
    model: TSPModel,
    records: list[PseudoExpertRecord],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, int, int, int]:
    if not records:
        raise ValueError("teacher-forcing batch must not be empty")
    problem_size = records[0].problem_size
    if any(record.problem_size != problem_size for record in records):
        raise ValueError("teacher-forcing batches must contain a single problem size")

    return teacher_forcing_loss(
        model,
        [record.to_instance() for record in records],
        [record.expert_sequence for record in records],
        device=device,
    )


def teacher_forcing_loss(
    model: TSPModel,
    instances,
    expert_sequences,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, int, int, int]:
    """Ordered token or block-slot CE on self-improved expert tours."""

    if not instances:
        raise ValueError("teacher-forcing instances must not be empty")
    problem_size = instances[0].J
    if any(instance.J != problem_size for instance in instances):
        raise ValueError("teacher-forcing instances must have a single problem size")
    if len(expert_sequences) != len(instances):
        raise ValueError("expert sequence count must match instance count")
    if int(getattr(model.config, "block_size", 1)) > 1:
        return block_teacher_forcing_loss(
            model,
            instances,
            expert_sequences,
            device=device,
        )
    expert = torch.tensor(
        expert_sequences,
        device=device,
        dtype=torch.long,
    )
    env = TSPEnv(
        problem_size=problem_size,
        pomo_size=1,
        use_pomo_start=False,
    )
    env.load_problems(instances, device)
    reset_state, _, _ = env.reset()
    model.pre_forward(reset_state.node_features)
    state, _, _ = env.pre_step()

    depot = torch.zeros((len(instances), 1), dtype=torch.long, device=device)
    state, _, done = env.step(depot)
    if done:
        raise RuntimeError("environment ended immediately after selecting depot")

    negative_log_likelihoods = []
    token_correct = 0
    sequence_correct = torch.ones(len(instances), dtype=torch.bool, device=device)
    for position in range(problem_size):
        _, _, probabilities = model(
            state,
            decode_type="greedy",
            use_pomo_start=False,
            return_all_probs=True,
        )
        expert_node = expert[:, position : position + 1] + 1
        expert_probability = probabilities.gather(2, expert_node[:, :, None]).squeeze(2)
        if bool((expert_probability <= 0).any()):
            raise RuntimeError("expert sequence is incompatible with the current visited-node mask")
        negative_log_likelihoods.append(-expert_probability.clamp_min(1e-12).log())

        predicted = probabilities.argmax(dim=2)
        is_correct = predicted.eq(expert_node)
        token_correct += int(is_correct.sum().item())
        sequence_correct &= is_correct.squeeze(1)

        if position == 0:
            if model.encoded_nodes is None:
                raise RuntimeError("model encoder state is unavailable")
            gather_index = expert_node[:, :, None].expand(
                len(instances), 1, model.encoded_nodes.size(2)
            )
            encoded_expert_first = model.encoded_nodes.gather(1, gather_index)
            # model.forward set q_first from its own greedy choice; teacher
            # forcing must condition all later decisions on the expert first node.
            model.decoder.set_q1(encoded_expert_first)
        state, _, done = env.step(expert_node)

    if not done:
        raise RuntimeError("teacher-forced rollout did not visit every target")
    losses = torch.stack(negative_log_likelihoods, dim=2)
    return (
        losses.mean(),
        token_correct,
        int(sequence_correct.sum().item()),
        int(losses.numel()),
    )


def block_teacher_forcing_loss(
    model: TSPModel,
    instances,
    expert_sequences,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, int, int, int]:
    """Cross entropy for ordered ``slot_k -> expert_node_k`` assignments."""

    if int(getattr(model.config, "block_size", 1)) <= 1:
        raise ValueError("block_teacher_forcing_loss requires block_size > 1")
    problem_size = instances[0].J
    expert = torch.as_tensor(expert_sequences, device=device, dtype=torch.long)
    if expert.shape != (len(instances), problem_size):
        raise ValueError("expert tours must have shape [B, J]")
    expected = torch.arange(problem_size, device=device).view(1, problem_size)
    if not bool((expert.sort(dim=1).values == expected).all()):
        raise ValueError("every expert tour must contain every target exactly once")

    env = TSPEnv(problem_size=problem_size, pomo_size=1, use_pomo_start=False)
    env.load_problems(instances, device)
    reset_state, _, _ = env.reset()
    model.pre_forward(reset_state.node_features)
    state, _, _ = env.pre_step()
    depot = torch.zeros((len(instances), 1), dtype=torch.long, device=device)
    state, _, done = env.step(depot)
    if done:
        raise RuntimeError("environment ended immediately after selecting depot")

    token_losses: list[torch.Tensor] = []
    token_correct = 0
    sequence_correct = torch.ones(len(instances), dtype=torch.bool, device=device)
    block_size = int(model.config.block_size)
    for position in range(0, problem_size, block_size):
        logits, available_nodes, active_slot_mask = model.block_logits(state)
        active_count = min(block_size, problem_size - position)
        if int(active_slot_mask.sum().item()) != active_count:
            raise RuntimeError("active block slots do not match remaining expert nodes")
        expert_block = expert[:, position : position + active_count, None] + 1
        expert_block = expert_block[:, None, :, :]
        matches = available_nodes[:, :, None, :].eq(expert_block)
        if not bool((matches.sum(dim=3) == 1).all()):
            raise RuntimeError("expert block conflicts with the current visited-node mask")
        target_positions = matches.to(torch.long).argmax(dim=3)
        active_logits = logits[:, :, :active_count, :]
        losses = torch.nn.functional.cross_entropy(
            active_logits.reshape(-1, active_logits.size(3)),
            target_positions.reshape(-1),
            reduction="none",
        )
        token_losses.append(losses)

        predicted_nodes, _ = sample_block_assignment(
            logits,
            available_nodes,
            active_slot_mask,
            decode_type="greedy",
        )
        expert_raw_nodes = expert_block.squeeze(3)
        is_correct = predicted_nodes.eq(expert_raw_nodes)
        token_correct += int(is_correct.sum().item())
        sequence_correct &= is_correct.squeeze(1).all(dim=1)

        if position == 0:
            if model.encoded_nodes is None:
                raise RuntimeError("model encoder state is unavailable")
            first_node = expert_raw_nodes[:, :, 0]
            gather_index = first_node[:, :, None].expand(
                len(instances), 1, model.encoded_nodes.size(2)
            )
            model.decoder.set_q1(model.encoded_nodes.gather(1, gather_index))
        for slot in range(active_count):
            state, _, done = env.step(expert_raw_nodes[:, :, slot])

    if not done:
        raise RuntimeError("block teacher-forced rollout did not visit every target")
    all_losses = torch.cat(token_losses, dim=0)
    return (
        all_losses.mean(),
        token_correct,
        int(sequence_correct.sum().item()),
        int(all_losses.numel()),
    )


def _grouped_batches(
    records: list[PseudoExpertRecord],
    batch_size: int,
    *,
    shuffle: bool,
    generator: torch.Generator,
) -> list[list[PseudoExpertRecord]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    groups: dict[int, list[PseudoExpertRecord]] = {}
    for record in records:
        groups.setdefault(record.problem_size, []).append(record)
    batches = []
    for problem_size in sorted(groups):
        items = groups[problem_size]
        if shuffle:
            order = torch.randperm(len(items), generator=generator).tolist()
            items = [items[index] for index in order]
        batches.extend(items[start : start + batch_size] for start in range(0, len(items), batch_size))
    if shuffle and batches:
        order = torch.randperm(len(batches), generator=generator).tolist()
        batches = [batches[index] for index in order]
    return batches


def evaluate_imitation(
    model: TSPModel,
    records: list[PseudoExpertRecord],
    *,
    device: torch.device,
    batch_size: int = 64,
) -> ImitationMetrics:
    if not records:
        return ImitationMetrics(nll=0.0, token_accuracy=0.0, sequence_accuracy=0.0, records=0, tokens=0)
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    token_correct = 0
    sequence_correct = 0
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for batch in _grouped_batches(
            records,
            batch_size,
            shuffle=False,
            generator=generator,
        ):
            loss, batch_token_correct, batch_sequence_correct, batch_tokens = _teacher_forcing_batch(
                model,
                batch,
                device=device,
            )
            total_nll += float(loss.item()) * batch_tokens
            total_tokens += batch_tokens
            token_correct += batch_token_correct
            sequence_correct += batch_sequence_correct
    model.train(was_training)
    return ImitationMetrics(
        nll=total_nll / max(total_tokens, 1),
        token_accuracy=token_correct / max(total_tokens, 1),
        sequence_accuracy=sequence_correct / max(len(records), 1),
        records=len(records),
        tokens=total_tokens,
    )


def _save_student_checkpoint(
    path: Path,
    *,
    model: TSPModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    source_checkpoint: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": float(metrics["selection_nll"]),
            "metrics": metrics,
            "model_config": asdict(model.config),
            "training_mode": "pseudo_expert_cross_entropy",
            "source_teacher_checkpoint": str(source_checkpoint),
        },
        path,
    )


def train_imitation_student(
    model: TSPModel,
    train_records: list[PseudoExpertRecord],
    validation_records: list[PseudoExpertRecord],
    *,
    device: torch.device,
    output_dir: str | Path,
    source_checkpoint: str | Path,
    config: ImitationTrainingConfig,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    if not train_records:
        raise ValueError("imitation training requires at least one training record")
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    set_seed(config.seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    shuffle_generator = torch.Generator().manual_seed(config.seed)
    initial_train = evaluate_imitation(model, train_records, device=device, batch_size=config.batch_size)
    initial_validation = evaluate_imitation(
        model,
        validation_records,
        device=device,
        batch_size=config.batch_size,
    )
    selection_initial = initial_validation.nll if validation_records else initial_train.nll
    initial_state = copy.deepcopy(model.state_dict())
    initial_metrics = {
        "epoch": 0,
        "train": initial_train.to_dict(),
        "validation": initial_validation.to_dict(),
        "selection_nll": selection_initial,
    }
    _save_student_checkpoint(
        output_path / "best.pt",
        model=model,
        optimizer=optimizer,
        epoch=0,
        metrics=initial_metrics,
        source_checkpoint=str(source_checkpoint),
    )

    rows = []
    best_nll = selection_initial
    best_epoch = 0
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        running_nll = 0.0
        running_tokens = 0
        for batch in _grouped_batches(
            train_records,
            config.batch_size,
            shuffle=config.shuffle,
            generator=shuffle_generator,
        ):
            loss, _, _, token_count = _teacher_forcing_batch(model, batch, device=device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            running_nll += float(loss.detach().item()) * token_count
            running_tokens += token_count

        train_metrics = evaluate_imitation(
            model,
            train_records,
            device=device,
            batch_size=config.batch_size,
        )
        validation_metrics = evaluate_imitation(
            model,
            validation_records,
            device=device,
            batch_size=config.batch_size,
        )
        selection_nll = validation_metrics.nll if validation_records else train_metrics.nll
        row = {
            "epoch": epoch,
            "optimization_nll": running_nll / max(running_tokens, 1),
            "train_nll": train_metrics.nll,
            "train_token_accuracy": train_metrics.token_accuracy,
            "train_sequence_accuracy": train_metrics.sequence_accuracy,
            "validation_nll": validation_metrics.nll,
            "validation_token_accuracy": validation_metrics.token_accuracy,
            "validation_sequence_accuracy": validation_metrics.sequence_accuracy,
            "selection_nll": selection_nll,
        }
        rows.append(row)
        if progress_callback is not None:
            progress_callback(dict(row))
        _save_student_checkpoint(
            output_path / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=row,
            source_checkpoint=str(source_checkpoint),
        )
        if selection_nll < best_nll - 1e-12:
            best_nll = selection_nll
            best_epoch = epoch
            _save_student_checkpoint(
                output_path / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=row,
                source_checkpoint=str(source_checkpoint),
            )

    best_checkpoint = torch.load(
        output_path / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    final_train = evaluate_imitation(model, train_records, device=device, batch_size=config.batch_size)
    final_validation = evaluate_imitation(
        model,
        validation_records,
        device=device,
        batch_size=config.batch_size,
    )

    metrics_path = output_path / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "training_mode": "pseudo_expert_cross_entropy",
        "source_checkpoint": str(source_checkpoint),
        "config": asdict(config),
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "initial_train": initial_train.to_dict(),
        "initial_validation": initial_validation.to_dict(),
        "best_epoch": best_epoch,
        "best_selection_nll": best_nll,
        "final_train": final_train.to_dict(),
        "final_validation": final_validation.to_dict(),
        "runtime": time.perf_counter() - started,
        "best_checkpoint": str(output_path / "best.pt"),
        "last_checkpoint": str(output_path / "last.pt"),
        "metrics_path": str(metrics_path),
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Keep the original model if validation never improved.  This assignment is
    # explicit so future changes to checkpoint handling cannot silently return
    # the final, potentially worse epoch.
    if best_epoch == 0:
        model.load_state_dict(initial_state)
    return summary
