from __future__ import annotations

from dataclasses import dataclass

import torch

from src.common.validation import batch_validate_sequences, deduplicate_sequences
from src.data.instance import CVTSPInstance
from src.master.model import MasterPolicyModel, RolloutState


@dataclass
class RolloutResult:
    sequences: torch.Tensor
    log_probs: torch.Tensor
    raw_node_indices: torch.Tensor

    def to_sequence_lists(self) -> list[list[list[int]]]:
        nested: list[list[list[int]]] = []
        for batch_sequences in self.sequences.cpu().tolist():
            nested.append([list(map(int, sequence)) for sequence in batch_sequences])
        return nested


def build_feature_batch(instances: list[CVTSPInstance], device: torch.device) -> torch.Tensor:
    features = [torch.tensor(instance.node_features, dtype=torch.float32) for instance in instances]
    return torch.stack(features, dim=0).to(device)


def _build_start_nodes(problem_size: int, pomo_size: int, device: torch.device) -> torch.Tensor:
    if pomo_size >= problem_size:
        return torch.arange(1, problem_size + 1, device=device, dtype=torch.long)
    starts = []
    for idx in range(pomo_size):
        value = 1 + (idx * problem_size) // pomo_size
        starts.append(value)
    unique = []
    seen = set()
    for value in starts:
        if value not in seen:
            unique.append(value)
            seen.add(value)
    candidate = 1
    while len(unique) < pomo_size and candidate <= problem_size:
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
        candidate += 1
    return torch.tensor(unique[:pomo_size], device=device, dtype=torch.long)


def rollout_batch(
    model: MasterPolicyModel,
    instances: list[CVTSPInstance],
    device: torch.device,
    decode_type: str = "sample",
    use_pomo_start: bool = True,
) -> RolloutResult:
    if not instances:
        raise ValueError("instances must not be empty")
    problem_size = instances[0].J
    if any(instance.J != problem_size for instance in instances):
        raise ValueError("all instances in a batch must have the same J")

    batch_size = len(instances)
    pomo_size = min(problem_size, model.config.max_pomo_size) if use_pomo_start else 1

    node_features = build_feature_batch(instances, device)
    model.pre_forward(node_features)

    batch_idx = torch.arange(batch_size, device=device)[:, None].expand(batch_size, pomo_size)
    pomo_idx = torch.arange(pomo_size, device=device)[None, :].expand(batch_size, pomo_size)
    ninf_mask = torch.zeros((batch_size, pomo_size, problem_size + 1), device=device)
    start_nodes = None
    if use_pomo_start:
        start_nodes = _build_start_nodes(problem_size, pomo_size, device)[None, :].expand(batch_size, pomo_size)
    state = RolloutState(
        batch_idx=batch_idx,
        pomo_idx=pomo_idx,
        selected_count=0,
        current_node=None,
        ninf_mask=ninf_mask,
        start_nodes=start_nodes,
    )

    selected_nodes = []
    log_prob_sum = torch.zeros((batch_size, pomo_size), device=device)
    total_steps = problem_size + 1

    while state.selected_count < total_steps:
        selected, prob = model(state, decode_type=decode_type, use_pomo_start=use_pomo_start)
        selected_nodes.append(selected)
        log_prob_sum = log_prob_sum + prob.clamp_min(1e-12).log()

        state.selected_count += 1
        state.current_node = selected
        state.ninf_mask[state.batch_idx, state.pomo_idx, selected] = float("-inf")

    raw_node_indices = torch.stack(selected_nodes, dim=2)
    sequences = raw_node_indices[:, :, 1:] - 1
    sequence_lists = [
        [list(map(int, sequence)) for sequence in batch_sequences]
        for batch_sequences in sequences.cpu().tolist()
    ]
    for instance, batch_sequences in zip(instances, sequence_lists):
        batch_validate_sequences(batch_sequences, instance.J)
    return RolloutResult(sequences=sequences, log_probs=log_prob_sum, raw_node_indices=raw_node_indices)


def generate_sequences(
    model: MasterPolicyModel,
    instance: CVTSPInstance,
    device: torch.device,
    decode_type: str,
    num_candidates: int,
    sample_max_rollouts: int = 32,
) -> list[list[int]]:
    candidates: list[list[int]] = []
    if decode_type == "greedy":
        rollout = rollout_batch(model, [instance], device=device, decode_type="greedy", use_pomo_start=True)
        return deduplicate_sequences(rollout.to_sequence_lists()[0])

    attempts = 0
    while len(candidates) < max(1, num_candidates) and attempts < max(1, sample_max_rollouts):
        rollout = rollout_batch(model, [instance], device=device, decode_type="sample", use_pomo_start=True)
        candidates.extend(rollout.to_sequence_lists()[0])
        candidates = deduplicate_sequences(candidates)
        attempts += 1
    return candidates[: max(1, num_candidates)]
