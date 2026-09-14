from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.CertifiedLocalSearch import (
    FixedAnchorLocalSearchConfig,
    FixedAnchorLocalSearchResult,
    run_fixed_anchor_local_search,
)
from src.CertifiedUpperBound import FixedSolutionUpperBound, is_optimal_gurobi_solution, validate_tour
from src.CVPSolver import SPSolution, SolverConfig, solve
from src.DualContributions import FixedDualLowerBound, directed_total_contribution_matrix
from src.TSPModel import ModelConfig
from src.TSProblemDef import CVTSPInstance


class SharedDirectedEdgeDecoder(nn.Module):
    """Lower-level directed evaluator fed detached upper-level encodings."""

    def __init__(self, policy_config: ModelConfig, hidden_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        edge_input_dim = 4 * policy_config.embedding_dim
        self.network = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, encoded_nodes: torch.Tensor) -> torch.Tensor:
        source = encoded_nodes[:, :, None, :].expand(-1, -1, encoded_nodes.size(1), -1)
        destination = encoded_nodes[:, None, :, :].expand(-1, encoded_nodes.size(1), -1, -1)
        pair = torch.cat((source, destination, destination - source, source * destination), dim=-1)
        return F.softplus(self.network(pair).squeeze(-1))


@dataclass(frozen=True)
class SparseSelfImprovementTask:
    """One instance, one anchor solve, and at most one final validation solve."""

    instance: CVTSPInstance
    sampled_sequences: list[list[int]]
    anchor_sequence: list[int]
    anchor_solver_config: SolverConfig
    final_solver_config: SolverConfig
    local_search_config: FixedAnchorLocalSearchConfig
    route_start_top_k: int = 4
    route_start_diversity_count: int = 0
    use_dual_pruning: bool = False
    upper_bound_tolerance: float = 2e-5
    feasibility_tolerance: float = 1e-5


@dataclass(frozen=True)
class NearMissValidationTask:
    """One optional exact solve for a locally improved, uncertified route."""

    instance: CVTSPInstance
    candidate_sequence: list[int]
    solver_config: SolverConfig


@dataclass(frozen=True)
class TargetRewardSolveTask:
    """One exact Target-route solve used to sharpen the upper-level reward."""

    instance: CVTSPInstance
    target_sequence: list[int]
    solver_config: SolverConfig


@dataclass
class SparseSelfImprovementResult:
    instance_id: str
    anchor_sequence: list[int]
    anchor_solution: SPSolution
    anchor_exact_makespan: float
    anchor_upper_bound_matrix: np.ndarray
    sampled_upper_bounds: np.ndarray
    top_k_start_sequences: list[list[int]]
    local_search_results: list[FixedAnchorLocalSearchResult]
    expert_sequence: list[int]
    expert_upper_bound: float
    expert_solution: SPSolution
    expert_exact_makespan: float
    supervision_upper_bound_matrix: np.ndarray
    anchor_gurobi_calls: int
    expert_gurobi_calls: int
    gurobi_runtime: float
    local_search_runtime: float
    total_runtime: float
    generated_moves: int
    evaluated_moves: int
    dual_pruned_moves: int
    accepted_moves: int
    linear_dual_count: int
    quadratic_dual_count: int
    final_validation_failed: bool = False
    near_miss_candidate_sequence: list[int] | None = None
    near_miss_candidate_upper_bound: float = float("inf")
    near_miss_relative_gap: float = float("inf")
    near_miss_attempted: bool = False
    near_miss_accepted: bool = False
    near_miss_exact_makespan: float | None = None
    near_miss_validation_failed: bool = False
    expert_source: str = "anchor"
    # Retained for result/checkpoint compatibility and diagnostics only.
    rl_baseline_upper_bound: float = float("nan")
    quality_start_count: int = 0
    diversity_start_count: int = 0

    @property
    def total_gurobi_calls(self) -> int:
        return int(self.anchor_gurobi_calls + self.expert_gurobi_calls)

    @property
    def expert_changed(self) -> bool:
        return tuple(self.expert_sequence) != tuple(self.anchor_sequence)

    @property
    def expert_upper_bound_matrix(self) -> np.ndarray | None:
        """Return C^E only when an exact second solve accepted a changed expert.

        ``supervision_upper_bound_matrix`` is constructed from the accepted
        expert's exact Gurobi solution.  Reusing it here keeps the RL envelope
        free of additional solver calls.  An unchanged expert deliberately
        exposes no second table so callers fall back to C^0.
        """

        if not self.expert_changed:
            return None
        if self.expert_gurobi_calls != 1:
            raise RuntimeError(
                "a changed expert upper-bound table requires exactly one expert Gurobi call"
            )
        return self.supervision_upper_bound_matrix

    @property
    def exact_improvement(self) -> float:
        return float(self.anchor_exact_makespan - self.expert_exact_makespan)

    @property
    def exact_improvement_percent(self) -> float:
        denominator = max(abs(float(self.anchor_exact_makespan)), 1e-12)
        return float(100.0 * self.exact_improvement / denominator)

    @property
    def expert_better(self) -> bool:
        return self.exact_improvement > 1e-7


def _require_optimal(solution: SPSolution, role: str, instance_id: str) -> None:
    if not is_optimal_gurobi_solution(solution):
        raise RuntimeError(f"{role} Gurobi solve failed for {instance_id}: {solution.status}")


def _deduplicate_tours(sequences: Iterable[Iterable[int]], problem_size: int) -> list[list[int]]:
    result: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for sequence in sequences:
        tour = validate_tour(sequence, problem_size)
        if tour in seen:
            continue
        seen.add(tour)
        result.append(list(tour))
    return result


def _best_non_anchor_local_candidate(
    searches: Iterable[FixedAnchorLocalSearchResult],
    anchor: Iterable[int],
    anchor_exact_makespan: float,
) -> tuple[list[int] | None, float, float]:
    """Return the lowest raw anchor-table U0 among distinct final routes."""

    anchor_tour = tuple(int(node) for node in anchor)
    best_sequence: list[int] | None = None
    best_upper_bound = float("inf")
    for search in searches:
        sequence = tuple(int(node) for node in search.final_sequence)
        upper_bound = float(search.final_upper_bound)
        if sequence == anchor_tour or not np.isfinite(upper_bound):
            continue
        if upper_bound < best_upper_bound:
            best_sequence = list(sequence)
            best_upper_bound = upper_bound

    if best_sequence is None:
        return None, float("inf"), float("inf")
    denominator = max(abs(float(anchor_exact_makespan)), 1e-12)
    relative_gap = (best_upper_bound - float(anchor_exact_makespan)) / denominator
    return best_sequence, best_upper_bound, float(relative_gap)


def directed_route_step_costs(sequence: Iterable[int], matrix: np.ndarray) -> np.ndarray:
    """Per-decision costs for depot -> tour -> depot under one directed table."""

    costs = np.asarray(matrix, dtype=float)
    node_count = costs.shape[0]
    if costs.ndim != 2 or costs.shape[1] != node_count or node_count < 2:
        raise ValueError("directed edge matrix must be square and include depot plus targets")
    tour = validate_tour(sequence, node_count - 1)
    route_nodes = np.asarray(tour, dtype=int) + 1
    step_costs = np.empty(len(tour), dtype=float)
    step_costs[0] = costs[0, route_nodes[0]]
    if len(tour) > 1:
        step_costs[1:] = costs[route_nodes[:-1], route_nodes[1:]]
    step_costs[-1] += costs[route_nodes[-1], 0]
    return step_costs


def directed_route_upper_bound(sequence: Iterable[int], matrix: np.ndarray) -> float:
    return float(directed_route_step_costs(sequence, matrix).sum())


def batched_directed_step_costs(
    sequence_lists: list[list[list[int]]],
    matrices: list[np.ndarray],
) -> np.ndarray:
    if len(sequence_lists) != len(matrices):
        raise ValueError("route batches and directed matrices have different lengths")
    return np.asarray(
        [
            [directed_route_step_costs(sequence, matrix) for sequence in routes]
            for routes, matrix in zip(sequence_lists, matrices)
        ],
        dtype=np.float32,
    )


def dense_reward_to_go(step_costs: np.ndarray) -> np.ndarray:
    costs = np.asarray(step_costs, dtype=float)
    if costs.ndim < 1:
        raise ValueError("step costs must have at least one dimension")
    return -np.flip(np.cumsum(np.flip(costs, axis=-1), axis=-1), axis=-1).copy()


def select_quality_diverse_start_indices(
    sequences: list[list[int]],
    scores: np.ndarray,
    *,
    top_k: int,
    diversity_count: int,
) -> tuple[np.ndarray, int, int]:
    """Select low-score starts plus starts far from that quality subset.

    Diversity is the minimum normalized positional Hamming distance from a
    candidate tour to any quality-selected tour.  Score and original stable
    rank break equal-distance ties, so disabling diversity reproduces the
    original stable Top-K selection exactly.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0 <= diversity_count < top_k:
        raise ValueError("diversity_count must be in [0, top_k)")
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size != len(sequences):
        raise ValueError("scores must contain one scalar per sequence")
    if not sequences:
        return np.empty(0, dtype=np.int64), 0, 0

    order = np.argsort(values, kind="stable")
    limit = min(int(top_k), len(sequences))
    effective_diversity = min(int(diversity_count), max(limit - 1, 0))
    quality_count = limit - effective_diversity
    quality_indices = order[:quality_count]
    if effective_diversity == 0:
        return quality_indices.astype(np.int64), quality_count, 0

    route_lengths = {len(sequence) for sequence in sequences}
    if len(route_lengths) != 1:
        raise ValueError("all candidate sequences must have equal length")
    candidates = order[quality_count:]
    candidate_routes = np.asarray(
        [sequences[int(index)] for index in candidates], dtype=np.int64
    )
    quality_routes = np.asarray(
        [sequences[int(index)] for index in quality_indices], dtype=np.int64
    )
    min_distances = np.mean(
        candidate_routes[:, None, :] != quality_routes[None, :, :], axis=2
    ).min(axis=1)
    # lexsort uses the final key as primary: largest distance first, then the
    # better upper-bound score, then the stable rank from ``order``.
    diverse_order = np.lexsort(
        (
            np.arange(len(candidates), dtype=np.int64),
            values[candidates],
            -min_distances,
        )
    )
    diverse_indices = candidates[diverse_order[:effective_diversity]]
    selected = np.concatenate((quality_indices, diverse_indices)).astype(np.int64)
    return selected, quality_count, effective_diversity


def run_sparse_self_improvement_task(task: SparseSelfImprovementTask) -> SparseSelfImprovementResult:
    """Execute the formal <=2-Gurobi training pipeline for one instance."""

    started = time.perf_counter()
    instance = task.instance
    anchor = list(validate_tour(task.anchor_sequence, instance.J))
    sampled = _deduplicate_tours(task.sampled_sequences, instance.J)
    if not sampled:
        raise ValueError("self-improvement requires at least one sampled route")
    if tuple(anchor) not in {tuple(sequence) for sequence in sampled}:
        sampled.append(anchor)
    if task.route_start_top_k <= 0:
        raise ValueError("route_start_top_k must be positive")
    if not 0 <= task.route_start_diversity_count < task.route_start_top_k:
        raise ValueError(
            "route_start_diversity_count must be in [0, route_start_top_k)"
        )

    anchor_solution = solve(instance, anchor, task.anchor_solver_config)
    _require_optimal(anchor_solution, "anchor", instance.instance_id)
    anchor_calls = 1
    anchor_majorizer = FixedSolutionUpperBound.from_solution(
        instance,
        anchor_solution,
        feasibility_tolerance=task.feasibility_tolerance,
        require_optimal=True,
    )
    anchor_matrix = directed_total_contribution_matrix(anchor_majorizer)
    anchor_upper_bound = directed_route_upper_bound(anchor, anchor_matrix)
    if not np.isclose(
        anchor_upper_bound,
        float(anchor_solution.objective),
        rtol=0.0,
        atol=task.upper_bound_tolerance,
    ):
        raise RuntimeError(
            "anchor upper-bound identity failed: "
            f"instance={instance.instance_id}, U0={anchor_upper_bound:.12g}, "
            f"V={anchor_solution.objective:.12g}"
        )
    sampled_upper_bounds = np.asarray(
        [directed_route_upper_bound(sequence, anchor_matrix) for sequence in sampled],
        dtype=float,
    )
    top_indices, quality_start_count, diversity_start_count = (
        select_quality_diverse_start_indices(
            sampled,
            sampled_upper_bounds,
            top_k=task.route_start_top_k,
            diversity_count=task.route_start_diversity_count,
        )
    )
    top_starts = [sampled[int(index)] for index in top_indices]

    dual_lower_bound = None
    if task.use_dual_pruning:
        dual_lower_bound = FixedDualLowerBound.from_solution(instance, anchor_solution)

    searches: list[FixedAnchorLocalSearchResult] = []
    local_started = time.perf_counter()
    for sequence in top_starts:
        searches.append(
            run_fixed_anchor_local_search(
                anchor_majorizer,
                sequence,
                task.local_search_config,
                dual_lower_bound=dual_lower_bound,
            )
        )
    local_runtime = time.perf_counter() - local_started
    (
        near_miss_candidate_sequence,
        near_miss_candidate_upper_bound,
        near_miss_relative_gap,
    ) = _best_non_anchor_local_candidate(
        searches,
        anchor,
        float(anchor_solution.objective),
    )

    # Keeping the anchor in this pool makes fallback exact and ensures that a
    # changed expert is selected only after a strict certified UB decrease.
    expert_sequence = anchor
    expert_upper_bound = float(anchor_solution.objective)
    for search in searches:
        if search.final_upper_bound < expert_upper_bound:
            expert_sequence = list(search.final_sequence)
            expert_upper_bound = float(search.final_upper_bound)

    acceptance_epsilon = task.local_search_config.acceptance_epsilon(anchor_solution.objective)
    expert_calls = 0
    final_solve_time = 0.0
    validation_failed = False
    expert_solution = anchor_solution
    expert_source = "anchor"
    rl_baseline_upper_bound = float(anchor_solution.objective)
    if (
        tuple(expert_sequence) != tuple(anchor)
        and expert_upper_bound < float(anchor_solution.objective) - acceptance_epsilon
    ):
        attempted = solve(instance, expert_sequence, task.final_solver_config)
        expert_calls = 1
        final_solve_time = float(attempted.solve_time)
        if is_optimal_gurobi_solution(attempted):
            if attempted.objective > expert_upper_bound + task.upper_bound_tolerance:
                raise RuntimeError(
                    "fixed-anchor upper-bound violation: "
                    f"instance={instance.instance_id}, exact={attempted.objective:.12g}, "
                    f"upper={expert_upper_bound:.12g}"
                )
            if attempted.objective >= anchor_solution.objective - acceptance_epsilon:
                raise RuntimeError(
                    "certified changed expert did not improve exact makespan: "
                    f"instance={instance.instance_id}, anchor={anchor_solution.objective:.12g}, "
                    f"expert={attempted.objective:.12g}"
                )
            expert_solution = attempted
            expert_source = "certified_upper_bound"
            rl_baseline_upper_bound = float(expert_upper_bound)
        else:
            # Do not train on an unverified pseudo expert.  The attempted call
            # still counts against the hard two-call budget.
            validation_failed = True
            expert_sequence = anchor
            expert_upper_bound = float(anchor_solution.objective)
    else:
        expert_sequence = anchor
        expert_upper_bound = float(anchor_solution.objective)

    expert_majorizer = FixedSolutionUpperBound.from_solution(
        instance,
        expert_solution,
        feasibility_tolerance=task.feasibility_tolerance,
        require_optimal=True,
    )
    supervision_matrix = directed_total_contribution_matrix(expert_majorizer)
    total_calls = anchor_calls + expert_calls
    if total_calls > 2:
        raise RuntimeError(f"per-instance Gurobi budget exceeded: {total_calls} > 2")

    return SparseSelfImprovementResult(
        instance_id=instance.instance_id,
        anchor_sequence=anchor,
        anchor_solution=anchor_solution,
        anchor_exact_makespan=float(anchor_solution.objective),
        anchor_upper_bound_matrix=anchor_matrix,
        sampled_upper_bounds=sampled_upper_bounds,
        top_k_start_sequences=top_starts,
        local_search_results=searches,
        expert_sequence=list(expert_sequence),
        expert_upper_bound=float(expert_upper_bound),
        expert_solution=expert_solution,
        expert_exact_makespan=float(expert_solution.objective),
        supervision_upper_bound_matrix=supervision_matrix,
        anchor_gurobi_calls=anchor_calls,
        expert_gurobi_calls=expert_calls,
        gurobi_runtime=float(anchor_solution.solve_time + final_solve_time),
        local_search_runtime=local_runtime,
        total_runtime=time.perf_counter() - started,
        generated_moves=sum(search.generated for search in searches),
        evaluated_moves=sum(search.evaluated for search in searches),
        dual_pruned_moves=sum(search.dual_pruned for search in searches),
        accepted_moves=sum(search.accepted for search in searches),
        linear_dual_count=len(anchor_solution.linear_duals),
        quadratic_dual_count=len(anchor_solution.quadratic_duals),
        final_validation_failed=validation_failed,
        near_miss_candidate_sequence=near_miss_candidate_sequence,
        near_miss_candidate_upper_bound=float(near_miss_candidate_upper_bound),
        near_miss_relative_gap=float(near_miss_relative_gap),
        expert_source=expert_source,
        rl_baseline_upper_bound=float(rl_baseline_upper_bound),
        quality_start_count=int(quality_start_count),
        diversity_start_count=int(diversity_start_count),
    )


def run_near_miss_validation_task(task: NearMissValidationTask) -> SPSolution:
    """Precisely re-optimize one fixed near-miss route with Gurobi."""

    if str(task.solver_config.solver_backend).strip().lower() != "gurobi":
        raise ValueError("near-miss validation requires the Gurobi solver backend")
    candidate = list(validate_tour(task.candidate_sequence, task.instance.J))
    return solve(task.instance, candidate, task.solver_config)


def run_target_reward_solve_task(task: TargetRewardSolveTask) -> SPSolution:
    """Precisely evaluate one selected Target POMO route with Gurobi."""

    if str(task.solver_config.solver_backend).strip().lower() != "gurobi":
        raise ValueError("exact Target reward requires the Gurobi solver backend")
    target = list(validate_tour(task.target_sequence, task.instance.J))
    return solve(task.instance, target, task.solver_config)


def _restore_anchor_expert(result: SparseSelfImprovementResult) -> None:
    result.expert_sequence = list(result.anchor_sequence)
    result.expert_upper_bound = float(result.anchor_exact_makespan)
    result.expert_solution = result.anchor_solution
    result.expert_exact_makespan = float(result.anchor_exact_makespan)
    result.supervision_upper_bound_matrix = np.asarray(
        result.anchor_upper_bound_matrix,
        dtype=float,
    ).copy()
    result.expert_source = "anchor"
    result.rl_baseline_upper_bound = float(result.anchor_exact_makespan)


def apply_near_miss_validation(
    result: SparseSelfImprovementResult,
    instance: CVTSPInstance,
    solution: SPSolution,
    *,
    acceptance_epsilon: float,
    upper_bound_tolerance: float,
    feasibility_tolerance: float,
) -> SparseSelfImprovementResult:
    """Audit and optionally accept one exact near-miss validation solve.

    The solve is charged to the per-instance budget regardless of its status.
    A valid exact improvement may be accepted even when its anchor-table upper
    bound did not itself certify improvement. The legacy diagnostic baseline
    scalar is capped at the exact anchor value; policy RL no longer consumes it.
    """

    if instance.instance_id != result.instance_id:
        raise ValueError(
            "near-miss instance mismatch: "
            f"result={result.instance_id}, instance={instance.instance_id}"
        )
    if result.near_miss_candidate_sequence is None:
        raise ValueError("near-miss validation requires a candidate sequence")
    candidate = list(validate_tour(result.near_miss_candidate_sequence, instance.J))
    candidate_upper_bound = float(result.near_miss_candidate_upper_bound)
    if not np.isfinite(candidate_upper_bound):
        raise ValueError("near-miss candidate upper bound must be finite")
    if acceptance_epsilon < 0.0 or upper_bound_tolerance < 0.0 or feasibility_tolerance < 0.0:
        raise ValueError("near-miss validation tolerances must be non-negative")
    if result.near_miss_attempted:
        raise RuntimeError(f"near-miss validation already attempted for {result.instance_id}")
    if result.expert_changed or result.expert_gurobi_calls != 0:
        raise RuntimeError(
            "near-miss validation is only available before an expert validation solve"
        )
    if result.total_gurobi_calls >= 2:
        raise RuntimeError(
            f"per-instance Gurobi budget already exhausted: {result.total_gurobi_calls}"
        )

    result.near_miss_attempted = True
    result.expert_gurobi_calls += 1
    solve_time = float(solution.solve_time)
    charged_runtime = solve_time if np.isfinite(solve_time) and solve_time > 0.0 else 0.0
    result.gurobi_runtime += charged_runtime
    result.total_runtime += charged_runtime
    if result.total_gurobi_calls > 2:
        raise RuntimeError(
            f"per-instance Gurobi budget exceeded: {result.total_gurobi_calls} > 2"
        )

    if not is_optimal_gurobi_solution(solution) or not np.isfinite(float(solution.objective)):
        result.near_miss_validation_failed = True
        result.final_validation_failed = True
        _restore_anchor_expert(result)
        return result

    try:
        solved_sequence = list(validate_tour(solution.sequence, instance.J))
    except ValueError:
        result.near_miss_validation_failed = True
        result.final_validation_failed = True
        _restore_anchor_expert(result)
        raise
    if tuple(solved_sequence) != tuple(candidate):
        result.near_miss_validation_failed = True
        result.final_validation_failed = True
        _restore_anchor_expert(result)
        raise RuntimeError(
            "near-miss solver returned a different fixed route: "
            f"expected={candidate}, received={solved_sequence}"
        )

    exact_makespan = float(solution.objective)
    result.near_miss_exact_makespan = exact_makespan
    if exact_makespan > candidate_upper_bound + upper_bound_tolerance:
        result.near_miss_validation_failed = True
        result.final_validation_failed = True
        _restore_anchor_expert(result)
        raise RuntimeError(
            "near-miss fixed-anchor upper-bound violation: "
            f"instance={instance.instance_id}, exact={exact_makespan:.12g}, "
            f"upper={candidate_upper_bound:.12g}"
        )

    if exact_makespan >= float(result.anchor_exact_makespan) - acceptance_epsilon:
        _restore_anchor_expert(result)
        return result

    try:
        expert_majorizer = FixedSolutionUpperBound.from_solution(
            instance,
            solution,
            feasibility_tolerance=feasibility_tolerance,
            require_optimal=True,
        )
        supervision_matrix = directed_total_contribution_matrix(expert_majorizer)
    except (TypeError, ValueError) as exc:
        result.near_miss_validation_failed = True
        result.final_validation_failed = True
        _restore_anchor_expert(result)
        raise RuntimeError(
            f"near-miss exact solution could not produce supervision for {instance.instance_id}"
        ) from exc

    result.near_miss_accepted = True
    result.expert_sequence = candidate
    result.expert_upper_bound = candidate_upper_bound
    result.expert_solution = solution
    result.expert_exact_makespan = exact_makespan
    result.supervision_upper_bound_matrix = supervision_matrix
    result.expert_source = "near_miss_exact"
    result.rl_baseline_upper_bound = min(
        candidate_upper_bound,
        float(result.anchor_exact_makespan),
    )
    return result


def normalize_contribution_matrix(matrix: np.ndarray, instance: CVTSPInstance) -> np.ndarray:
    factor = float(instance.carrier_speed) / max(float(instance.scale), 1e-8)
    return np.asarray(matrix, dtype=np.float32) * factor


def denormalize_contribution_matrix(matrix: np.ndarray, instance: CVTSPInstance) -> np.ndarray:
    factor = max(float(instance.scale), 1e-8) / float(instance.carrier_speed)
    return np.asarray(matrix, dtype=float) * factor


def edge_decoder_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    huber_beta: float = 0.1,
) -> torch.Tensor:
    if predictions.shape != targets.shape:
        raise ValueError(f"edge prediction shape {predictions.shape} != target shape {targets.shape}")
    node_count = predictions.size(-1)
    mask = ~torch.eye(node_count, dtype=torch.bool, device=predictions.device)
    return F.smooth_l1_loss(predictions[:, mask], targets[:, mask], beta=huber_beta)


def serialize_joint_checkpoint_metadata(
    *,
    edge_hidden_dim: int,
    edge_dropout: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "architecture": "upper_self_improving_route_lower_learned_primal_dual",
        "edge_hidden_dim": int(edge_hidden_dim),
        "edge_dropout": float(edge_dropout),
        "gurobi_budget": "one_anchor_plus_one_target_plus_optional_one_expert_per_instance",
        **dict(extra or {}),
    }
