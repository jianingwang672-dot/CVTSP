from __future__ import annotations

from dataclasses import dataclass

import torch

from src.TSProblemDef import CVTSPInstance, RandomCVTSPGenerator


@dataclass
class Reset_State:
    node_features: torch.Tensor


@dataclass
class Step_State:
    batch_idx: torch.Tensor
    pomo_idx: torch.Tensor
    selected_count: int
    current_node: torch.Tensor | None = None
    ninf_mask: torch.Tensor | None = None
    start_nodes: torch.Tensor | None = None


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


def _deduplicate_sequences(sequences) -> list[list[int]]:
    unique: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for sequence in sequences:
        key = tuple(sequence)
        if key in seen:
            continue
        seen.add(key)
        unique.append(list(sequence))
    return unique


def _batch_validate_sequences(sequences, problem_size: int) -> None:
    for sequence in sequences:
        seq = list(sequence)
        if len(seq) != problem_size:
            raise ValueError(f"invalid sequence: expected length {problem_size}, got {len(seq)}")
        if any(node < 0 or node >= problem_size for node in seq):
            raise ValueError(f"invalid sequence: sequence contains indices outside 0..{problem_size - 1}")
        if len(set(seq)) != problem_size:
            raise ValueError("invalid sequence: sequence contains duplicates or omissions")


def build_feature_batch(instances: list[CVTSPInstance], device: torch.device) -> torch.Tensor:
    features = [torch.tensor(instance.node_features, dtype=torch.float32) for instance in instances]
    return torch.stack(features, dim=0).to(device)


def _normalize_start_node_strategy(strategy: str | None) -> str:
    normalized = str(strategy or "spread").strip().lower()
    aliases = {
        "kg0": "nearest_depot",
        "nearest": "nearest_depot",
        "nearest_depot": "nearest_depot",
        "spread": "spread",
    }
    return aliases.get(normalized, normalized)


def _resolve_pomo_size(problem_size: int, pomo_size_limit: int, pomo_divisor: int | None = None) -> int:
    if problem_size <= 0:
        raise ValueError("problem_size must be positive")
    effective_limit = max(1, int(pomo_size_limit))
    dynamic_pomo = problem_size
    if pomo_divisor is not None:
        divisor = int(pomo_divisor)
        if divisor <= 0:
            raise ValueError("pomo_divisor must be positive when provided")
        dynamic_pomo = max(1, problem_size // divisor)
    return min(problem_size, effective_limit, dynamic_pomo)


def _build_spread_start_nodes(problem_size: int, pomo_size: int, device: torch.device) -> torch.Tensor:
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


def _build_nearest_depot_start_nodes(
    instances: list[CVTSPInstance],
    pomo_size: int,
    device: torch.device,
) -> torch.Tensor:
    start_nodes: list[list[int]] = []
    for instance in instances:
        ranked_targets = list(range(instance.J))
        depot_x, depot_y = float(instance.depot[0]), float(instance.depot[1])
        ranked_targets.sort(
            key=lambda idx: (
                (float(instance.targets[idx, 0]) - depot_x) ** 2
                + (float(instance.targets[idx, 1]) - depot_y) ** 2,
                idx,
            )
        )
        start_nodes.append([idx + 1 for idx in ranked_targets[:pomo_size]])
    return torch.tensor(start_nodes, device=device, dtype=torch.long)


class TSPEnv:
    def __init__(
        self,
        problem_size: int | None,
        pomo_size: int,
        pomo_divisor: int | None = None,
        use_pomo_start: bool = True,
        start_node_strategy: str = "spread",
        dataset_dir: str = "instance/Data",
        min_problem_size: int | None = None,
        max_problem_size: int | None = None,
        problem_sizes: list[int] | tuple[int, ...] | None = None,
        seed: int = 1234,
        online_random: bool = False,
    ):
        self.problem_size = problem_size
        self.pomo_size_limit = pomo_size
        self.pomo_divisor = int(pomo_divisor) if pomo_divisor is not None else None
        self.pomo_size = pomo_size
        self.use_pomo_start = use_pomo_start
        self.start_node_strategy = _normalize_start_node_strategy(start_node_strategy)
        self.online_random = online_random

        self.batch_size: int | None = None
        self.device: torch.device | None = None
        self.instances: list[CVTSPInstance] = []
        self.node_features: torch.Tensor | None = None
        self.batch_idx: torch.Tensor | None = None
        self.pomo_idx: torch.Tensor | None = None
        self.start_nodes: torch.Tensor | None = None

        self.selected_count = 0
        self.current_node: torch.Tensor | None = None
        self.selected_node_list: torch.Tensor | None = None
        self.step_state: Step_State | None = None

        self.generator: RandomCVTSPGenerator | None = None
        if online_random:
            low = int(min_problem_size if min_problem_size is not None else (problem_size or 20))
            high = int(max_problem_size if max_problem_size is not None else (problem_size or low))
            self.generator = RandomCVTSPGenerator(
                dataset_dir=dataset_dir,
                min_problem_size=low,
                max_problem_size=high,
                problem_sizes=problem_sizes,
                seed=seed,
            )

    def _prepare_loaded_instances(self, instances: list[CVTSPInstance], device: torch.device) -> None:
        if not instances:
            raise ValueError("instances must not be empty")
        problem_size = instances[0].J
        if any(instance.J != problem_size for instance in instances):
            raise ValueError("all instances in a batch must have the same J")

        self.problem_size = problem_size
        self.pomo_size = _resolve_pomo_size(self.problem_size, self.pomo_size_limit, self.pomo_divisor)
        self.instances = list(instances)
        self.batch_size = len(instances)
        self.device = device
        self.node_features = build_feature_batch(self.instances, device)
        self.batch_idx = torch.arange(self.batch_size, device=device)[:, None].expand(self.batch_size, self.pomo_size)
        self.pomo_idx = torch.arange(self.pomo_size, device=device)[None, :].expand(self.batch_size, self.pomo_size)
        self.start_nodes = None
        if self.use_pomo_start:
            if self.start_node_strategy == "nearest_depot":
                self.start_nodes = _build_nearest_depot_start_nodes(self.instances, self.pomo_size, device)
            else:
                self.start_nodes = _build_spread_start_nodes(self.problem_size, self.pomo_size, device)[None, :].expand(
                    self.batch_size, self.pomo_size
                )

    def load_problems(
        self,
        instances_or_batch_size: list[CVTSPInstance] | int,
        device: torch.device,
        problem_size: int | None = None,
    ) -> None:
        if isinstance(instances_or_batch_size, int):
            if self.generator is None:
                raise ValueError("online_random generator is not configured for this environment")
            instances = self.generator.sample_batch(batch_size=instances_or_batch_size, problem_size=problem_size)
        else:
            instances = list(instances_or_batch_size)
        self._prepare_loaded_instances(instances, device)

    def reset(self):
        if self.batch_size is None or self.node_features is None or self.batch_idx is None or self.pomo_idx is None:
            raise RuntimeError("call load_problems before reset")

        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long, device=self.device)
        self.step_state = Step_State(
            batch_idx=self.batch_idx,
            pomo_idx=self.pomo_idx,
            selected_count=0,
            current_node=None,
            ninf_mask=torch.zeros((self.batch_size, self.pomo_size, self.problem_size + 1), device=self.device),
            start_nodes=self.start_nodes,
        )
        return Reset_State(self.node_features), None, False

    def pre_step(self):
        if self.step_state is None:
            raise RuntimeError("call reset before pre_step")
        return self.step_state, None, False

    def step(self, selected: torch.Tensor):
        if self.step_state is None or self.selected_node_list is None:
            raise RuntimeError("call reset before step")

        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat((self.selected_node_list, selected[:, :, None]), dim=2)

        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask[self.batch_idx, self.pomo_idx, self.current_node] = float("-inf")
        self.step_state.selected_count = self.selected_count

        done = self.selected_count == self.problem_size + 1
        return self.step_state, None, done


def rollout_batch(
    model,
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

    pomo_size = (
        _resolve_pomo_size(problem_size, model.config.max_pomo_size, model.config.pomo_divisor)
        if use_pomo_start
        else 1
    )
    env = TSPEnv(
        problem_size=problem_size,
        pomo_size=model.config.max_pomo_size if use_pomo_start else 1,
        pomo_divisor=model.config.pomo_divisor if use_pomo_start else None,
        use_pomo_start=use_pomo_start,
        start_node_strategy=model.config.start_node_strategy,
    )
    env.load_problems(instances, device)
    reset_state, _, _ = env.reset()
    model.pre_forward(reset_state.node_features)

    state, reward, done = env.pre_step()
    del reward
    batch_size = len(instances)
    log_prob_sum = torch.zeros((batch_size, pomo_size), device=device)

    while not done:
        selected, prob = model(state, decode_type=decode_type, use_pomo_start=use_pomo_start)
        state, _, done = env.step(selected)
        log_prob_sum = log_prob_sum + prob.clamp_min(1e-12).log()

    raw_node_indices = env.selected_node_list
    sequences = raw_node_indices[:, :, 1:] - 1
    sequence_lists = [
        [list(map(int, sequence)) for sequence in batch_sequences]
        for batch_sequences in sequences.cpu().tolist()
    ]
    for instance, batch_sequences in zip(instances, sequence_lists):
        _batch_validate_sequences(batch_sequences, instance.J)
    return RolloutResult(sequences=sequences, log_probs=log_prob_sum, raw_node_indices=raw_node_indices)


def generate_sequences(
    model,
    instance: CVTSPInstance,
    device: torch.device,
    decode_type: str,
    num_candidates: int,
    sample_max_rollouts: int = 32,
) -> list[list[int]]:
    candidates: list[list[int]] = []
    if decode_type == "greedy":
        rollout = rollout_batch(model, [instance], device=device, decode_type="greedy", use_pomo_start=True)
        return _deduplicate_sequences(rollout.to_sequence_lists()[0])

    attempts = 0
    while len(candidates) < max(1, num_candidates) and attempts < max(1, sample_max_rollouts):
        rollout = rollout_batch(model, [instance], device=device, decode_type="sample", use_pomo_start=True)
        candidates.extend(rollout.to_sequence_lists()[0])
        candidates = _deduplicate_sequences(candidates)
        attempts += 1
    return candidates[: max(1, num_candidates)]
