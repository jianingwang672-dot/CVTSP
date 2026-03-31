from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Iterable

import torch

from src.common.config import AppConfig, SolverConfig
from src.common.validation import deduplicate_sequences
from src.data.instance import CVTSPInstance
from src.master.model import MasterPolicyModel
from src.master.rollout import generate_sequences
from src.pipeline.baselines import generate_baseline_sequences
from src.subproblem.solver import SPSolution, solve


@dataclass
class CandidateEvaluation:
    sequence: list[int]
    solution: SPSolution

    def to_dict(self) -> dict[str, Any]:
        return {"sequence": list(self.sequence), "solution": self.solution.to_dict()}


@dataclass
class EvaluationResult:
    instance_id: str
    decode_type: str
    num_candidates: int
    success: bool
    best_sequence: list[int]
    best_objective: float
    best_status: str
    candidates: list[CandidateEvaluation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        best_solution = None
        for candidate in self.candidates:
            if list(candidate.sequence) == list(self.best_sequence):
                best_solution = candidate.solution.to_dict()
                break

        return {
            "instance_id": self.instance_id,
            "decode_type": self.decode_type,
            "num_candidates": self.num_candidates,
            "success": self.success,
            "best_sequence": list(self.best_sequence),
            "best_target_sequence_0based": list(self.best_sequence),
            "best_target_sequence_1based": [node + 1 for node in self.best_sequence],
            "best_route_with_depot": best_solution.get("route_with_depot") if best_solution else None,
            "best_objective": float(self.best_objective),
            "best_status": self.best_status,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


class RewardCache:
    def __init__(self):
        self._cache: dict[tuple[str, tuple[int, ...]], SPSolution] = {}
        self.hits = 0
        self.misses = 0

    def get_or_solve(
        self,
        instance: CVTSPInstance,
        sequence: list[int],
        solver_config: SolverConfig,
    ) -> tuple[SPSolution, bool]:
        key = (instance.instance_id, tuple(sequence))
        if key in self._cache:
            self.hits += 1
            return self._cache[key], True

        solution = solve(instance, sequence, solver_config)
        self._cache[key] = solution
        self.misses += 1
        return solution, False


def _normalize_sequence_input(sequences: Iterable[Iterable[int]]) -> list[list[int]]:
    normalized = [list(map(int, sequence)) for sequence in sequences]
    return deduplicate_sequences(normalized)


def evaluate_sequences(
    instance: CVTSPInstance,
    sequences: Iterable[Iterable[int]],
    solver_config: SolverConfig,
    cache: RewardCache | None = None,
) -> list[CandidateEvaluation]:
    reward_cache = cache or RewardCache()
    candidates = []
    for sequence in _normalize_sequence_input(sequences):
        solution, _ = reward_cache.get_or_solve(instance, sequence, solver_config)
        candidates.append(CandidateEvaluation(sequence=sequence, solution=solution))
    return candidates


def _select_best_candidate(candidates: list[CandidateEvaluation]) -> CandidateEvaluation:
    successful = [candidate for candidate in candidates if candidate.solution.success]
    if successful:
        return min(successful, key=lambda item: item.solution.objective)
    return min(candidates, key=lambda item: item.solution.objective)


def evaluate_instance(
    instance: CVTSPInstance,
    config: AppConfig,
    model: MasterPolicyModel | None = None,
    sequences: Iterable[Iterable[int]] | None = None,
    device: torch.device | None = None,
    cache: RewardCache | None = None,
    baseline_name: str | None = None,
) -> EvaluationResult:
    if baseline_name is not None:
        candidate_sequences = generate_baseline_sequences(
            instance,
            baseline_name=baseline_name,
            num_candidates=config.decode.num_candidates,
            seed=config.runtime.seed,
        )
        decode_type = baseline_name
    elif sequences is not None:
        candidate_sequences = _normalize_sequence_input(sequences)
        decode_type = config.decode.decode_type
    elif model is not None and device is not None:
        candidate_sequences = generate_sequences(
            model=model,
            instance=instance,
            device=device,
            decode_type=config.decode.decode_type,
            num_candidates=config.decode.num_candidates,
            sample_max_rollouts=config.decode.sample_max_rollouts,
        )
        decode_type = config.decode.decode_type
    else:
        raise ValueError("provide either baseline_name, sequences, or model+device")

    candidate_records = evaluate_sequences(
        instance,
        candidate_sequences,
        solver_config=config.solver,
        cache=cache,
    )
    best_candidate = _select_best_candidate(candidate_records)
    return EvaluationResult(
        instance_id=instance.instance_id,
        decode_type=decode_type,
        num_candidates=len(candidate_records),
        success=best_candidate.solution.success,
        best_sequence=list(best_candidate.sequence),
        best_objective=float(best_candidate.solution.objective),
        best_status=best_candidate.solution.status,
        candidates=candidate_records,
    )


def evaluate_dataset(
    instances: list[CVTSPInstance],
    config: AppConfig,
    model: MasterPolicyModel | None = None,
    device: torch.device | None = None,
    cache: RewardCache | None = None,
    baseline_name: str | None = None,
) -> tuple[list[EvaluationResult], dict[str, float]]:
    results = []
    for instance in instances:
        result = evaluate_instance(
            instance=instance,
            config=config,
            model=model,
            device=device,
            cache=cache,
            baseline_name=baseline_name,
        )
        results.append(result)

    objectives = [result.best_objective if isfinite(result.best_objective) else 1e6 for result in results]
    feasible = [1.0 if result.success else 0.0 for result in results]
    summary = {
        "num_instances": float(len(results)),
        "avg_objective": float(sum(objectives) / max(len(objectives), 1)),
        "feasible_ratio": float(sum(feasible) / max(len(feasible), 1)),
    }
    return results, summary
