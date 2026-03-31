from __future__ import annotations

import random

import numpy as np

from src.common.validation import deduplicate_sequences
from src.data.instance import CVTSPInstance


def input_order(instance: CVTSPInstance) -> list[int]:
    return list(range(instance.J))


def nearest_neighbor(instance: CVTSPInstance) -> list[int]:
    remaining = set(range(instance.J))
    current = instance.depot.astype(np.float64)
    sequence: list[int] = []

    while remaining:
        next_node = min(
            remaining,
            key=lambda idx: float(np.linalg.norm(instance.targets[idx].astype(np.float64) - current)),
        )
        sequence.append(next_node)
        current = instance.targets[next_node].astype(np.float64)
        remaining.remove(next_node)
    return sequence


def random_k(instance: CVTSPInstance, num_candidates: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    base = list(range(instance.J))
    candidates = []
    for _ in range(max(1, num_candidates)):
        perm = list(base)
        rng.shuffle(perm)
        candidates.append(perm)
    return deduplicate_sequences(candidates)


def generate_baseline_sequences(
    instance: CVTSPInstance,
    baseline_name: str,
    num_candidates: int,
    seed: int,
) -> list[list[int]]:
    if baseline_name == "input_order":
        return [input_order(instance)]
    if baseline_name == "nearest_neighbor":
        return [nearest_neighbor(instance)]
    if baseline_name == "random_k":
        return random_k(instance, num_candidates=num_candidates, seed=seed)
    raise ValueError(f"unsupported baseline '{baseline_name}'")

