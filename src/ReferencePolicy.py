from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from src.CertifiedUpperBound import is_optimal_gurobi_solution, validate_tour
from src.CVPSolver import SolverConfig, solve
from src.PseudoExpert import generate_policy_candidates
from src.TSProblemDef import CVTSPInstance
from src.TSPModel import TSPModel
from src.TSPEnv import rollout_batch


REFERENCE_STATE_VERSION = 1
_EXACT_COMPARISON_TOLERANCE = 1e-6


def reference_validation_protocol(max_candidates: int) -> dict[str, Any]:
    """Zero means complete POMO; positive limits reproduce legacy validation."""
    if max_candidates < 0:
        raise ValueError("max_candidates must be non-negative (0 means full POMO)")
    return {
        "version": 2 if max_candidates == 0 else 1,
        "mode": "full_pomo" if max_candidates == 0 else "legacy_candidates",
        "decode_type": "greedy",
        "augmentation": False,
        "pomo_size": "J" if max_candidates == 0 else "model_config",
        "max_candidates": int(max_candidates),
        "include_free_start": max_candidates > 0,
        "route_score": "minimum_exact_gurobi_makespan_over_distinct_candidates",
    }


def require_matching_validation_protocol(checkpoint: Mapping[str, Any], max_candidates: int) -> None:
    """Never carry a validation best/history across incompatible score definitions."""
    saved = checkpoint.get("reference_validation_protocol")
    if saved is None:
        trajectory = checkpoint.get("strict_resume_signature", {}).get("trajectory_params", {})
        saved_count = trajectory.get("reference_validation_candidates")
        if saved_count is None:
            metrics = checkpoint.get("reference_state", {}).get("last_metrics") or {}
            saved_count = metrics.get("max_candidates", 1)
        saved = reference_validation_protocol(int(saved_count))
    current = reference_validation_protocol(max_candidates)
    if saved != current:
        raise ValueError(
            "checkpoint validation protocol differs from the requested protocol; "
            "old best scores and Reference history cannot be compared with new scores. "
            "Use --reference-validation-candidates "
            f"{saved['max_candidates']} to resume the original protocol, or start a fresh run."
        )


def _solve_validation_route(payload):
    """Top-level worker function, usable by the existing spawn process pool."""
    instance, sequence, solver_config = payload
    solution = solve(instance, list(sequence), solver_config)
    if not is_optimal_gurobi_solution(solution) or not math.isfinite(solution.objective):
        raise RuntimeError(
            f"exact validation solve failed for {instance.instance_id}: status={solution.status}"
        )
    if tuple(validate_tour(solution.sequence, instance.J)) != tuple(sequence):
        raise RuntimeError(f"validation solver changed the route for {instance.instance_id}")
    return {
        "sequence": list(sequence),
        "objective": float(solution.objective),
        "solve_time": float(solution.solve_time),
    }


def _evaluate_full_pomo_candidates(model, instances, device, solver_config, *, executor, batch_size):
    """Generate J starts without a free route, then solve each distinct tour."""
    records = [[] for _ in instances]
    groups: dict[int, list[int]] = {}
    for index, instance in enumerate(instances):
        groups.setdefault(int(instance.J), []).append(index)
    original_config = model.config
    try:
        model.config = replace(
            model.config, max_pomo_size=max(groups), pomo_divisor=None,
            start_node_strategy="nearest_depot",
        )
        with torch.no_grad():
            for size, indices in groups.items():
                for offset in range(0, len(indices), batch_size):
                    batch_indices = indices[offset:offset + batch_size]
                    batch = [instances[i] for i in batch_indices]
                    sequences = rollout_batch(
                        model, batch, device=device, decode_type="greedy", use_pomo_start=True,
                    ).to_sequence_lists()
                    if len(sequences) != len(batch):
                        raise RuntimeError("validation rollout batch size mismatch")
                    tasks, owners = [], []
                    for index, tours in zip(batch_indices, sequences):
                        if len(tours) != size:
                            raise RuntimeError(f"full-POMO validation requires J={size} starts")
                        unique = dict.fromkeys(tuple(validate_tour(t, size)) for t in tours)
                        for tour in unique:
                            tasks.append((instances[index], tour, solver_config))
                            owners.append(index)
                    solved = (
                        map(_solve_validation_route, tasks) if executor is None else
                        executor.map(_solve_validation_route, tasks, chunksize=1)
                    )
                    count = 0
                    for owner, result in zip(owners, solved):
                        records[owner].append(result)
                        count += 1
                    if count != len(tasks):
                        raise RuntimeError("validation did not solve every distinct candidate")
    finally:
        model.config = original_config
    return records


@dataclass
class ReferenceState:
    """Checkpoint-safe state for the reference/target promotion loop."""

    version: int = REFERENCE_STATE_VERSION
    policy_version: int = 0
    validation_rounds: int = 0
    promotions: int = 0
    last_validation_epoch: int | None = None
    last_promoted_epoch: int | None = None
    validation_gurobi_calls: int = 0
    last_metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached payload suitable for inclusion in a checkpoint."""

        return copy.deepcopy(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ReferenceState":
        """Restore state while accepting checkpoints written before fields existed."""

        if payload is None:
            return cls()
        allowed = cls.__dataclass_fields__
        values = {key: copy.deepcopy(value) for key, value in payload.items() if key in allowed}
        state = cls(**values)
        # Normalize containers from hand-written or older checkpoint payloads.
        state.history = [dict(item) for item in (state.history or [])]
        state.last_metrics = (
            None if state.last_metrics is None else dict(state.last_metrics)
        )
        return state


def freeze_reference_policy(model: TSPModel) -> TSPModel:
    """Put a reference policy in inference mode and make all parameters immutable."""

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return model


def clone_reference_policy(
    target: TSPModel,
    *,
    device: torch.device | str | None = None,
) -> TSPModel:
    """Create an independent, frozen TSPModel snapshot of ``target``."""

    if not isinstance(target, TSPModel):
        raise TypeError("target must be a TSPModel")
    if device is None:
        try:
            destination = next(target.parameters()).device
        except StopIteration:  # pragma: no cover - TSPModel always has parameters
            destination = torch.device("cpu")
    else:
        destination = torch.device(device)
    reference = TSPModel(copy.deepcopy(target.config)).to(destination)
    reference.load_state_dict(target.state_dict(), strict=True)
    return freeze_reference_policy(reference)


def _solve_all_candidates_exact(
    *,
    role: str,
    model: TSPModel,
    instance: CVTSPInstance,
    device: torch.device,
    solver_config: SolverConfig,
    max_candidates: int,
) -> tuple[list[dict[str, Any]], int]:
    candidates = generate_policy_candidates(
        model,
        instance,
        device=device,
        decode_type="greedy",
        max_candidates=max_candidates,
    )
    if not candidates:
        raise RuntimeError(f"{role} produced no candidates for {instance.instance_id}")

    records: list[dict[str, Any]] = []
    calls = 0
    for sequence in candidates:
        solution = solve(instance, list(sequence), solver_config)
        calls += 1
        if not is_optimal_gurobi_solution(solution) or not math.isfinite(solution.objective):
            raise RuntimeError(
                f"{role} exact validation solve failed for {instance.instance_id}: "
                f"status={solution.status}"
            )
        records.append(
            {
                "sequence": [int(node) for node in sequence],
                "objective": float(solution.objective),
                "solve_time": float(solution.solve_time),
            }
        )
    return records, calls


def evaluate_policy_pair_exact(
    reference: TSPModel,
    target: TSPModel,
    instances: Iterable[CVTSPInstance],
    device: torch.device | str,
    solver_config: SolverConfig,
    max_candidates: int,
    *,
    executor=None,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Compare two policies on identical fixed instances using exact objectives.

    With max_candidates=0, generate exactly J greedy POMO starts, with no
    augmentation or free-start injection, and solve every distinct tour.
    Positive limits retain the legacy free-start-first evaluation for old runs.
    Validation calls are separate from the training Gurobi budget.
    """

    protocol = reference_validation_protocol(max_candidates)
    if batch_size <= 0:
        raise ValueError("validation batch_size must be positive")
    fixed_instances = list(instances)
    if not fixed_instances:
        raise ValueError("instances must not be empty")
    if str(solver_config.solver_backend).strip().lower() != "gurobi":
        raise ValueError("exact reference/target validation requires solver_backend='gurobi'")

    evaluation_device = torch.device(device)
    started = time.perf_counter()
    reference_was_training = reference.training
    target_was_training = target.training
    reference.eval()
    target.eval()
    details: list[dict[str, Any]] = []
    validation_gurobi_calls = 0

    try:
        if max_candidates == 0:
            ref_all = _evaluate_full_pomo_candidates(
                reference, fixed_instances, evaluation_device, solver_config,
                executor=executor, batch_size=batch_size,
            )
            target_all = _evaluate_full_pomo_candidates(
                target, fixed_instances, evaluation_device, solver_config,
                executor=executor, batch_size=batch_size,
            )
        for index, instance in enumerate(fixed_instances):
            if max_candidates == 0:
                reference_records = ref_all[index]
                target_records = target_all[index]
                reference_calls, target_calls = len(reference_records), len(target_records)
            else:
                reference_records, reference_calls = _solve_all_candidates_exact(
                    role="reference",
                    model=reference,
                    instance=instance,
                    device=evaluation_device,
                    solver_config=solver_config,
                    max_candidates=max_candidates,
                )
                target_records, target_calls = _solve_all_candidates_exact(
                    role="target",
                    model=target,
                    instance=instance,
                    device=evaluation_device,
                    solver_config=solver_config,
                    max_candidates=max_candidates,
                )
            validation_gurobi_calls += reference_calls + target_calls
            reference_best = min(reference_records, key=lambda item: item["objective"])
            target_best = min(target_records, key=lambda item: item["objective"])
            improvement = float(reference_best["objective"] - target_best["objective"])
            if improvement > _EXACT_COMPARISON_TOLERANCE:
                outcome = "target_win"
            elif improvement < -_EXACT_COMPARISON_TOLERANCE:
                outcome = "target_loss"
            else:
                outcome = "tie"
            details.append(
                {
                    "instance_id": str(instance.instance_id),
                    "problem_size": int(instance.J),
                    "reference_best_objective": float(reference_best["objective"]),
                    "reference_best_sequence": list(reference_best["sequence"]),
                    "target_best_objective": float(target_best["objective"]),
                    "target_best_sequence": list(target_best["sequence"]),
                    "objective_improvement": improvement,
                    "relative_improvement": float(
                        improvement / max(abs(float(reference_best["objective"])), 1e-12)
                    ),
                    "outcome": outcome,
                    "reference_candidates": reference_records,
                    "target_candidates": target_records,
                    "reference_gurobi_calls": int(reference_calls),
                    "target_gurobi_calls": int(target_calls),
                }
            )
    finally:
        reference.train(reference_was_training)
        target.train(target_was_training)

    reference_values = np.asarray(
        [row["reference_best_objective"] for row in details], dtype=float
    )
    target_values = np.asarray(
        [row["target_best_objective"] for row in details], dtype=float
    )
    improvements = reference_values - target_values
    reference_mean = float(reference_values.mean())
    target_mean = float(target_values.mean())
    mean_improvement = float(improvements.mean())
    wins = sum(row["outcome"] == "target_win" for row in details)
    ties = sum(row["outcome"] == "tie" for row in details)
    losses = sum(row["outcome"] == "target_loss" for row in details)
    runtime = float(time.perf_counter() - started)
    group_metrics: dict[str, dict[str, float | int]] = {}
    for problem_size in sorted({int(row["problem_size"]) for row in details}):
        rows = [row for row in details if int(row["problem_size"]) == problem_size]
        group_metrics[str(problem_size)] = {
            "instances": len(rows),
            "reference_mean": float(
                np.mean([row["reference_best_objective"] for row in rows])
            ),
            "target_mean": float(
                np.mean([row["target_best_objective"] for row in rows])
            ),
            "delta": float(np.mean([row["relative_improvement"] for row in rows])),
        }
    delta_all = float(np.mean([row["relative_improvement"] for row in details]))

    result = {
        "instances": len(details),
        "max_candidates": int(max_candidates),
        "validation_protocol": protocol,
        "reference_mean": reference_mean,
        "target_mean": target_mean,
        "reference_mean_objective": reference_mean,
        "target_mean_objective": target_mean,
        "mean_objective_improvement": mean_improvement,
        "relative_improvement_percent": float(
            100.0 * mean_improvement / max(abs(reference_mean), 1e-12)
        ),
        "delta_all": delta_all,
        "group_metrics": group_metrics,
        "wins": int(wins),
        "target_wins": int(wins),
        "ties": int(ties),
        "losses": int(losses),
        "target_losses": int(losses),
        "calls": int(validation_gurobi_calls),
        "validation_gurobi_calls": int(validation_gurobi_calls),
        "runtime": runtime,
        "validation_runtime": runtime,
        "details": details,
    }
    for size, values in group_metrics.items():
        result[f"delta_{size}"] = float(values["delta"])
    return result


def promote_reference_if_better(
    reference: TSPModel,
    target: TSPModel,
    metrics: Mapping[str, Any],
    epsilon: float = 0.0,
) -> bool:
    """Promote target only when its exact mean objective is strictly better."""

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if reference is target:
        raise ValueError("reference and target must be different model objects")
    try:
        reference_mean = float(
            metrics["reference_mean_objective"]
            if "reference_mean_objective" in metrics
            else metrics["reference_mean"]
        )
        target_mean = float(
            metrics["target_mean_objective"]
            if "target_mean_objective" in metrics
            else metrics["target_mean"]
        )
    except KeyError as exc:
        raise KeyError(
            "metrics must contain reference_mean_objective and target_mean_objective"
        ) from exc
    if not math.isfinite(reference_mean) or not math.isfinite(target_mean):
        raise ValueError("policy means must be finite")
    if not target_mean < reference_mean - float(epsilon):
        return False

    reference.load_state_dict(target.state_dict(), strict=True)
    reference.encoded_nodes = None
    freeze_reference_policy(reference)
    return True


def promote_reference_with_size_gate(
    reference: TSPModel,
    target: TSPModel,
    metrics: Mapping[str, Any],
    *,
    required_non_degrading_sizes: tuple[int, ...] = (80, 100),
    epsilon: float = 0.0,
) -> bool:
    """Promote on positive overall relative gain with no large-size regression."""

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if reference is target:
        raise ValueError("reference and target must be different model objects")
    delta_all = float(metrics["delta_all"])
    groups = metrics.get("group_metrics", {})
    if not math.isfinite(delta_all):
        raise ValueError("delta_all must be finite")
    if not delta_all > epsilon:
        return False
    for problem_size in required_non_degrading_sizes:
        key = str(int(problem_size))
        if key not in groups:
            raise KeyError(f"validation metrics missing required J={problem_size} group")
        delta = float(groups[key]["delta"])
        if not math.isfinite(delta):
            raise ValueError(f"delta_{problem_size} must be finite")
        if delta < 0.0:
            return False

    reference.load_state_dict(target.state_dict(), strict=True)
    reference.encoded_nodes = None
    freeze_reference_policy(reference)
    return True


__all__ = [
    "REFERENCE_STATE_VERSION",
    "ReferenceState",
    "clone_reference_policy",
    "evaluate_policy_pair_exact",
    "freeze_reference_policy",
    "promote_reference_if_better",
    "promote_reference_with_size_gate",
]
