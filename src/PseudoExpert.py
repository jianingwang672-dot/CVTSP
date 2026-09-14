from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

from src.CertifiedLocalSearch import (
    CertifiedLocalSearchConfig,
    CertifiedLocalSearchResult,
    run_certified_local_search,
)
from src.CVPSolver import SolverConfig, solve
from src.TSPEnv import generate_sequences, rollout_batch
from src.TSProblemDef import CVTSPInstance


PSEUDO_EXPERT_SCHEMA_VERSION = 2
SUPPORTED_PSEUDO_EXPERT_SCHEMA_VERSIONS = {1, 2}


def instance_to_dict(instance: CVTSPInstance) -> dict:
    return {
        "instance_id": instance.instance_id,
        "path": str(instance.path),
        "J": int(instance.J),
        "depot": np.asarray(instance.depot, dtype=float).tolist(),
        "targets": np.asarray(instance.targets, dtype=float).tolist(),
        "carrier_speed": float(instance.carrier_speed),
        "uav_speed": float(instance.uav_speed),
        "endurance": float(instance.endurance),
        "scale": float(instance.scale),
    }


def instance_from_dict(payload: dict) -> CVTSPInstance:
    targets = np.asarray(payload["targets"], dtype=np.float32)
    instance = CVTSPInstance(
        instance_id=str(payload["instance_id"]),
        path=Path(str(payload.get("path", f"<pseudo:{payload['instance_id']}>"))),
        J=int(payload["J"]),
        depot=np.asarray(payload["depot"], dtype=np.float32),
        targets=targets,
        carrier_speed=float(payload["carrier_speed"]),
        uav_speed=float(payload["uav_speed"]),
        endurance=float(payload["endurance"]),
        scale=float(payload["scale"]),
    )
    if instance.depot.shape != (2,) or targets.shape != (instance.J, 2):
        raise ValueError(f"invalid serialized instance shape for {instance.instance_id}")
    return instance


@dataclass(frozen=True)
class PseudoExpertRecord:
    split: str
    instance: dict
    teacher_candidates: list[dict]
    teacher_best_sequence: list[int]
    teacher_best_objective: float
    expert_sequence: list[int]
    expert_objective: float
    expert_source_candidate: int
    expert_improvement_over_teacher: float
    evaluated_moves: int
    accepted_moves: int
    gurobi_calls: int
    runtime: float
    generated_moves: int = 0

    def __post_init__(self) -> None:
        # Schema-v1 and small hand-built records only stored exact evaluations;
        # for an unguided search generation and evaluation counts are equal.
        if self.generated_moves == 0 and self.evaluated_moves > 0:
            object.__setattr__(self, "generated_moves", self.evaluated_moves)

    @property
    def instance_id(self) -> str:
        return str(self.instance["instance_id"])

    @property
    def problem_size(self) -> int:
        return int(self.instance["J"])

    def to_instance(self) -> CVTSPInstance:
        return instance_from_dict(self.instance)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["pruned_moves"] = self.pruned_moves
        return payload

    @property
    def pruned_moves(self) -> int:
        return self.generated_moves - self.evaluated_moves

    @classmethod
    def from_dict(cls, payload: dict) -> "PseudoExpertRecord":
        return cls(
            split=str(payload["split"]),
            instance=dict(payload["instance"]),
            teacher_candidates=[dict(item) for item in payload["teacher_candidates"]],
            teacher_best_sequence=[int(node) for node in payload["teacher_best_sequence"]],
            teacher_best_objective=float(payload["teacher_best_objective"]),
            expert_sequence=[int(node) for node in payload["expert_sequence"]],
            expert_objective=float(payload["expert_objective"]),
            expert_source_candidate=int(payload["expert_source_candidate"]),
            expert_improvement_over_teacher=float(payload["expert_improvement_over_teacher"]),
            evaluated_moves=int(payload["evaluated_moves"]),
            accepted_moves=int(payload["accepted_moves"]),
            gurobi_calls=int(payload["gurobi_calls"]),
            runtime=float(payload["runtime"]),
            generated_moves=int(payload.get("generated_moves", payload["evaluated_moves"])),
        )


@dataclass(frozen=True)
class PseudoExpertGenerationConfig:
    decode_type: str = "greedy"
    max_candidates: int = 8
    sample_max_rollouts: int = 32
    max_local_search_refreshes: int = 10
    absolute_epsilon: float = 1e-4
    relative_epsilon: float = 1e-8
    upper_bound_tolerance: float = 2e-5
    feasibility_tolerance: float = 1e-5
    move_types: tuple[str, ...] = ("swap", "relocate", "two_opt")


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_policy_candidates(
    model,
    instance: CVTSPInstance,
    *,
    device: torch.device,
    decode_type: str,
    max_candidates: int,
    sample_max_rollouts: int = 32,
) -> list[list[int]]:
    """Generate one free-start route plus diverse POMO routes.

    The free-start route is important for imitation learning because its first
    target is predicted by the decoder.  POMO routes retain the existing
    multi-start exploration behavior.
    """

    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    normalized_decode = str(decode_type).strip().lower()
    if normalized_decode not in {"greedy", "sample"}:
        raise ValueError("decode_type must be 'greedy' or 'sample'")

    candidates: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    def add(sequence: Iterable[int]) -> None:
        candidate = [int(node) for node in sequence]
        key = tuple(candidate)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)

    with torch.no_grad():
        free_rollout = rollout_batch(
            model,
            [instance],
            device=device,
            decode_type=normalized_decode,
            use_pomo_start=False,
        )
        add(free_rollout.to_sequence_lists()[0][0])
        if len(candidates) < max_candidates:
            pomo_candidates = generate_sequences(
                model,
                instance,
                device=device,
                decode_type=normalized_decode,
                num_candidates=max_candidates,
                sample_max_rollouts=sample_max_rollouts,
            )
            for candidate in pomo_candidates:
                add(candidate)
                if len(candidates) >= max_candidates:
                    break
    return candidates


def generate_pseudo_expert_record(
    model,
    instance: CVTSPInstance,
    *,
    split: str,
    device: torch.device,
    solver_config: SolverConfig,
    generation_config: PseudoExpertGenerationConfig,
    candidate_ranker_factory: Callable[[CVTSPInstance], Callable[[tuple[int, ...]], float]]
    | None = None,
    max_exact_candidates: int | None = None,
) -> PseudoExpertRecord:
    started = time.perf_counter()
    candidates = generate_policy_candidates(
        model,
        instance,
        device=device,
        decode_type=generation_config.decode_type,
        max_candidates=generation_config.max_candidates,
        sample_max_rollouts=generation_config.sample_max_rollouts,
    )
    if not candidates:
        raise RuntimeError(f"teacher produced no routes for {instance.instance_id}")
    if max_exact_candidates is not None and candidate_ranker_factory is None:
        raise ValueError("max_exact_candidates requires a candidate_ranker_factory")
    if max_exact_candidates is not None and max_exact_candidates <= 0:
        raise ValueError("max_exact_candidates must be positive")

    search_config = CertifiedLocalSearchConfig(
        max_refreshes=generation_config.max_local_search_refreshes,
        absolute_epsilon=generation_config.absolute_epsilon,
        relative_epsilon=generation_config.relative_epsilon,
        upper_bound_tolerance=generation_config.upper_bound_tolerance,
        feasibility_tolerance=generation_config.feasibility_tolerance,
        move_types=generation_config.move_types,
        require_optimal=True,
    )
    candidate_ranker = (
        candidate_ranker_factory(instance) if candidate_ranker_factory is not None else None
    )
    results: list[CertifiedLocalSearchResult] = []
    for candidate in candidates:
        results.append(
            run_certified_local_search(
                instance,
                candidate,
                solver_config=solver_config,
                search_config=search_config,
                candidate_ranker=candidate_ranker,
                max_exact_candidates=max_exact_candidates,
            )
        )

    teacher_best_index = min(range(len(results)), key=lambda idx: results[idx].initial_objective)
    expert_index = min(range(len(results)), key=lambda idx: results[idx].final_objective)
    teacher_best = results[teacher_best_index]
    expert = results[expert_index]
    improvement = float(teacher_best.initial_objective - expert.final_objective)
    tolerance = generation_config.upper_bound_tolerance
    if improvement < -tolerance:
        raise RuntimeError(
            "pseudo expert is worse than the best teacher candidate: "
            f"teacher={teacher_best.initial_objective:.12g}, expert={expert.final_objective:.12g}"
        )

    candidate_payloads = []
    for index, result in enumerate(results):
        candidate_payloads.append(
            {
                "candidate_index": index,
                "initial_sequence": list(result.initial_sequence),
                "initial_objective": float(result.initial_objective),
                "improved_sequence": list(result.final_sequence),
                "improved_objective": float(result.final_objective),
                "objective_improvement": float(result.objective_improvement),
                "generated_moves": int(result.generated_moves),
                "evaluated_moves": int(result.evaluated_moves),
                "pruned_moves": int(result.pruned_moves),
                "accepted_moves": int(result.accepted_moves),
                "gurobi_calls": int(result.gurobi_calls),
                "termination_reason": result.termination_reason,
                "history": [move.to_dict() for move in result.history],
            }
        )

    return PseudoExpertRecord(
        split=str(split),
        instance=instance_to_dict(instance),
        teacher_candidates=candidate_payloads,
        teacher_best_sequence=list(teacher_best.initial_sequence),
        teacher_best_objective=float(teacher_best.initial_objective),
        expert_sequence=list(expert.final_sequence),
        expert_objective=float(expert.final_objective),
        expert_source_candidate=int(expert_index),
        expert_improvement_over_teacher=max(0.0, improvement),
        evaluated_moves=sum(result.evaluated_moves for result in results),
        accepted_moves=sum(result.accepted_moves for result in results),
        gurobi_calls=sum(result.gurobi_calls for result in results),
        runtime=time.perf_counter() - started,
        generated_moves=sum(result.generated_moves for result in results),
    )


def summarize_pseudo_experts(records: list[PseudoExpertRecord]) -> dict:
    if not records:
        return {
            "records": 0,
            "teacher_candidates": 0,
            "improved_records": 0,
            "mean_teacher_objective": 0.0,
            "mean_expert_objective": 0.0,
            "mean_improvement": 0.0,
            "max_improvement": 0.0,
            "generated_moves": 0,
            "evaluated_moves": 0,
            "pruned_moves": 0,
            "exact_ub_evaluation_reduction_percent": 0.0,
            "accepted_moves": 0,
            "gurobi_calls": 0,
            "runtime": 0.0,
        }
    improvements = [record.expert_improvement_over_teacher for record in records]
    generated_moves = sum(record.generated_moves for record in records)
    evaluated_moves = sum(record.evaluated_moves for record in records)
    return {
        "records": len(records),
        "split_counts": {
            split: sum(record.split == split for record in records)
            for split in sorted({record.split for record in records})
        },
        "teacher_candidates": sum(len(record.teacher_candidates) for record in records),
        "improved_records": sum(value > 1e-9 for value in improvements),
        "mean_teacher_objective": float(np.mean([record.teacher_best_objective for record in records])),
        "mean_expert_objective": float(np.mean([record.expert_objective for record in records])),
        "mean_improvement": float(np.mean(improvements)),
        "max_improvement": float(max(improvements)),
        "generated_moves": generated_moves,
        "evaluated_moves": evaluated_moves,
        "pruned_moves": generated_moves - evaluated_moves,
        "exact_ub_evaluation_reduction_percent": (
            100.0 * (generated_moves - evaluated_moves) / max(generated_moves, 1)
        ),
        "accepted_moves": sum(record.accepted_moves for record in records),
        "gurobi_calls": sum(record.gurobi_calls for record in records),
        "runtime": float(sum(record.runtime for record in records)),
    }


def save_pseudo_expert_dataset(
    path: str | Path,
    records: list[PseudoExpertRecord],
    *,
    metadata: dict | None = None,
) -> dict:
    payload = {
        "schema_version": PSEUDO_EXPERT_SCHEMA_VERSION,
        "metadata": dict(metadata or {}),
        "summary": summarize_pseudo_experts(records),
        "records": [record.to_dict() for record in records],
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return payload


def load_pseudo_expert_dataset(path: str | Path) -> tuple[list[PseudoExpertRecord], dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = int(payload.get("schema_version", -1))
    if version not in SUPPORTED_PSEUDO_EXPERT_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported pseudo-expert schema {version}; "
            f"expected one of {sorted(SUPPORTED_PSEUDO_EXPERT_SCHEMA_VERSIONS)}"
        )
    records = [PseudoExpertRecord.from_dict(item) for item in payload.get("records", [])]
    for record in records:
        instance = record.to_instance()
        expected = set(range(instance.J))
        if set(record.expert_sequence) != expected or len(record.expert_sequence) != instance.J:
            raise ValueError(f"invalid expert tour in record {record.instance_id}")
    return records, payload


def evaluate_policy_exact(
    model,
    records: list[PseudoExpertRecord],
    *,
    device: torch.device,
    solver_config: SolverConfig,
    decode_type: str,
    max_candidates: int,
    sample_max_rollouts: int = 32,
) -> dict:
    """Evaluate raw policy tours exactly; local search is intentionally excluded."""

    started = time.perf_counter()
    details = []
    gurobi_calls = 0
    for record in records:
        instance = record.to_instance()
        candidates = generate_policy_candidates(
            model,
            instance,
            device=device,
            decode_type=decode_type,
            max_candidates=max_candidates,
            sample_max_rollouts=sample_max_rollouts,
        )
        solved = []
        for sequence in candidates:
            solution = solve(instance, sequence, solver_config)
            gurobi_calls += 1
            if not solution.success or str(solution.status).strip() != "2":
                raise RuntimeError(
                    f"exact policy evaluation failed for {instance.instance_id}: {solution.status}"
                )
            solved.append((float(solution.objective), sequence))
        best_objective, best_sequence = min(solved, key=lambda item: item[0])
        details.append(
            {
                "instance_id": instance.instance_id,
                "split": record.split,
                "teacher_objective": float(record.teacher_best_objective),
                "student_objective": best_objective,
                "student_sequence": list(best_sequence),
                "objective_improvement": float(record.teacher_best_objective - best_objective),
                "candidate_count": len(candidates),
            }
        )

    teacher_values = [item["teacher_objective"] for item in details]
    student_values = [item["student_objective"] for item in details]
    improvements = [item["objective_improvement"] for item in details]
    return {
        "instances": len(details),
        "teacher_mean_objective": float(np.mean(teacher_values)) if details else 0.0,
        "student_mean_objective": float(np.mean(student_values)) if details else 0.0,
        "mean_objective_improvement": float(np.mean(improvements)) if details else 0.0,
        "relative_improvement_percent": (
            100.0 * float(np.mean(improvements)) / max(abs(float(np.mean(teacher_values))), 1e-12)
            if details
            else 0.0
        ),
        "student_wins": sum(value > 1e-6 for value in improvements),
        "ties": sum(abs(value) <= 1e-6 for value in improvements),
        "student_losses": sum(value < -1e-6 for value in improvements),
        "gurobi_calls": gurobi_calls,
        "runtime": time.perf_counter() - started,
        "details": details,
    }
