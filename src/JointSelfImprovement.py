from __future__ import annotations

import time
import tempfile
from dataclasses import dataclass, field, replace
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
from src.CertifiedUpperBound import (
    FixedSolutionUpperBound,
    apply_double_bridge,
    is_optimal_gurobi_solution,
    validate_tour,
)
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
    route_start_diversity_pool_size: int = 0
    route_start_diversity_max_relative_gap: float = float("inf")
    use_dual_pruning: bool = False
    upper_bound_tolerance: float = 2e-5
    feasibility_tolerance: float = 1e-5
    perturbation_restarts: int = 0
    local_search_backend: str = "swap_two_opt"
    lkh_binary: str = ""
    lkh_budget_coefficient: float = 1.5e-4
    lkh_desired_scale: int = 1_000_000
    lkh_trials: int = 1
    lkh_max_candidates: int = 5
    lkh_move_type: int = 5
    lkh_native_seconds: float = 2.0
    lkh_process_timeout_seconds: float = 5.0
    lkh_work_root: str = ""
    local_search_seed: int = 0
    # Hybrid search with the swap_two_opt backend: this fraction of the Top-K
    # starts is searched by LKH (interleaving quality/diversity starts), the
    # rest by the move search; the LKH time budget scales with the fraction.
    lkh_start_fraction: float = 0.0
    # Non-zero: fraction and LKH budget are counted against this nominal number
    # of starts, and every actual start gets LKH on top of the move search
    # (ablation D keeps the 30-start budget on its single route).
    lkh_budget_starts: int = 0


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
    perturbation_attempts: int = 0
    perturbation_improvements: int = 0
    local_search_backend: str = "swap_two_opt"
    local_search_budget_seconds: float = 0.0
    lkh_problem_export_seconds: float = 0.0
    lkh_quantization_scale: int = 0
    lkh_rounding_error_bound: float = 0.0
    lkh_native_calls: int = 0
    lkh_completed_starts: int = 0
    lkh_timed_out_starts: int = 0
    lkh_accepted_outputs: int = 0
    lkh_distinct_final_routes: int = 0
    lkh_returned_anchor_starts: int = 0
    # Optional C0-refresh rounds; their exact solves are tracked separately
    # from the <=2-call anchor/expert budget above.
    c0_refresh_rounds_run: int = 0
    c0_refresh_gurobi_calls: int = 0
    c0_refresh_accepted: int = 0
    c0_refresh_runtime: float = 0.0
    # C0 tables of every additional exactly solved route (refresh bases and
    # candidates); each is a valid upper bound for any route of the instance.
    extra_upper_bound_matrices: list = field(default_factory=list)

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
    diversity_pool_size: int = 0,
    diversity_max_relative_gap: float = float("inf"),
) -> tuple[np.ndarray, int, int]:
    """Select low-score starts plus quality-gated, structurally distinct starts.

    Diversity is the minimum normalized positional Hamming distance from a
    candidate tour to the routes already selected. When a quality gate is
    configured, diverse candidates must lie inside both the requested score
    rank pool and the relative score band. Any unfilled diversity slots fall
    back to the next-best routes. Disabling the gate preserves the legacy
    selector for checkpoint-compatible ablations.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0 <= diversity_count < top_k:
        raise ValueError("diversity_count must be in [0, top_k)")
    if diversity_pool_size < 0:
        raise ValueError("diversity_pool_size must be non-negative")
    if diversity_max_relative_gap < 0.0:
        raise ValueError("diversity_max_relative_gap must be non-negative")
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size != len(sequences):
        raise ValueError("scores must contain one scalar per sequence")
    if not sequences:
        return np.empty(0, dtype=np.int64), 0, 0

    order = np.argsort(values, kind="stable")
    limit = min(int(top_k), len(sequences))
    # Keep the configured quality core intact when fewer than ``top_k``
    # distinct POMO routes are available (common for J < top_k or duplicate
    # rollouts).  For the standard 20-quality + 10-diverse setup this means
    # 20 routes produce 20 quality starts, while 25 routes produce 20 quality
    # and at most 5 diverse starts.
    requested_quality_count = int(top_k) - int(diversity_count)
    quality_count = min(limit, requested_quality_count)
    effective_diversity = min(int(diversity_count), limit - quality_count)
    quality_indices = order[:quality_count]
    if effective_diversity == 0:
        return quality_indices.astype(np.int64), quality_count, 0

    route_lengths = {len(sequence) for sequence in sequences}
    if len(route_lengths) != 1:
        raise ValueError("all candidate sequences must have equal length")
    gate_disabled = diversity_pool_size == 0 and np.isinf(
        diversity_max_relative_gap
    )
    if gate_disabled:
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
        # lexsort uses the final key as primary: largest distance first, then
        # the better upper-bound score, then the stable rank from ``order``.
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

    pool_limit = len(order)
    if diversity_pool_size > 0:
        pool_limit = min(pool_limit, max(int(diversity_pool_size), quality_count))
    candidates = order[quality_count:pool_limit]
    if np.isfinite(diversity_max_relative_gap):
        quality_boundary = float(values[quality_indices[-1]])
        threshold = quality_boundary + abs(quality_boundary) * float(
            diversity_max_relative_gap
        )
        candidates = candidates[values[candidates] <= threshold + 1e-12]

    # Greedy farthest-point selection keeps accepted routes apart from the
    # quality core and from one another. Score and stable rank break ties.
    selected_indices = [int(index) for index in quality_indices]
    diverse_indices: list[int] = []
    remaining = [int(index) for index in candidates]
    while remaining and len(diverse_indices) < effective_diversity:
        selected_routes = np.asarray(
            [sequences[index] for index in selected_indices], dtype=np.int64
        )
        candidate_routes = np.asarray(
            [sequences[index] for index in remaining], dtype=np.int64
        )
        min_distances = np.mean(
            candidate_routes[:, None, :] != selected_routes[None, :, :], axis=2
        ).min(axis=1)
        best_local = min(
            range(len(remaining)),
            key=lambda local: (
                -float(min_distances[local]),
                float(values[remaining[local]]),
                local,
            ),
        )
        chosen = remaining.pop(best_local)
        diverse_indices.append(chosen)
        selected_indices.append(chosen)

    selected_set = set(selected_indices)
    for index in order:
        value = int(index)
        if value in selected_set:
            continue
        selected_indices.append(value)
        selected_set.add(value)
        if len(selected_indices) >= limit:
            break
    selected = np.asarray(selected_indices[:limit], dtype=np.int64)
    actual_diversity = len(diverse_indices)
    actual_quality = len(selected) - actual_diversity
    if len(selected) != limit:
        raise RuntimeError("quality-diverse selection returned the wrong number of starts")
    return selected, actual_quality, actual_diversity


def _run_lkh_fixed_anchor_searches(
    matrix: np.ndarray,
    starts: list[list[int]],
    *,
    quality_count: int,
    diversity_count: int,
    task: SparseSelfImprovementTask,
) -> tuple[list[FixedAnchorLocalSearchResult], dict[str, float | int]]:
    """Run the predeclared time-capped LKH arm and retain untouched fallbacks.

    The budget is a deterministic function of J calibrated from the completed
    swap+2-opt pilot.  Quality and diversity starts are interleaved, so a short
    deadline does not systematically discard all diverse starts.  Every LKH
    output is re-evaluated on the original floating-point C0 matrix.
    """
    from scripts.lkh_fixed_bound import LKHConfig, prepare_problem, run_start

    if task.use_dual_pruning or task.perturbation_restarts:
        raise ValueError("LKH training does not support dual pruning or perturbation restarts")
    if not task.lkh_binary:
        raise ValueError("LKH backend requires a binary path")
    if task.lkh_budget_coefficient <= 0:
        raise ValueError("LKH budget coefficient must be positive")
    config = LKHConfig(
        binary=task.lkh_binary,
        desired_scale=task.lkh_desired_scale,
        trials=task.lkh_trials,
        max_candidates=task.lkh_max_candidates,
        move_type=task.lkh_move_type,
        native_seconds=task.lkh_native_seconds,
        process_timeout_seconds=task.lkh_process_timeout_seconds,
    )
    problem_size = len(matrix) - 1
    budget = max(0.005, float(task.lkh_budget_coefficient) * problem_size * problem_size)
    begun = time.perf_counter()
    deadline = begun + budget
    initial_bounds = [directed_route_upper_bound(route, matrix) for route in starts]
    raw = [
        {
            "initial_sequence": list(route), "final_sequence": list(route),
            "initial_upper_bound": bound, "final_upper_bound": bound,
            "called": False, "completed": False, "timed_out": False,
            "accepted_output": False, "wall_seconds": 0.0,
            "native_output_upper_bound": None,
        }
        for route, bound in zip(starts, initial_bounds)
    ]
    root = task.lkh_work_root or None
    with tempfile.TemporaryDirectory(prefix="cvtsp-lkh-", dir=root) as directory:
        export_started = time.perf_counter()
        exported = prepare_problem(matrix, directory, config)
        export_seconds = time.perf_counter() - export_started
        order: list[int] = []
        for offset in range(max(quality_count, diversity_count)):
            if offset < quality_count:
                order.append(offset)
            if offset < diversity_count:
                order.append(quality_count + offset)
        # If selection ever changes shape, retain deterministic coverage.
        order.extend(index for index in range(len(starts)) if index not in set(order))
        for index in order:
            if time.perf_counter() >= deadline:
                break
            raw[index] = run_start(
                matrix,
                starts[index],
                directory,
                config,
                prefix=f"start-{index:02d}",
                seed=int((task.local_search_seed + index) % 2_000_000_000) + 1,
                deadline=deadline,
                absolute_epsilon=task.local_search_config.absolute_epsilon,
                relative_epsilon=task.local_search_config.relative_epsilon,
            )
    searches: list[FixedAnchorLocalSearchResult] = []
    for item in raw:
        if not item["called"]:
            termination = "lkh_budget_not_called"
        elif item["timed_out"]:
            termination = "lkh_timeout_with_output" if item["accepted_output"] else "lkh_timeout"
        elif item["accepted_output"]:
            termination = "lkh_improved"
        else:
            termination = "lkh_no_float_improvement"
        searches.append(FixedAnchorLocalSearchResult(
            initial_sequence=list(item["initial_sequence"]),
            final_sequence=list(item["final_sequence"]),
            initial_upper_bound=float(item["initial_upper_bound"]),
            final_upper_bound=float(item["final_upper_bound"]),
            generated=0, evaluated=0, dual_pruned=0, accepted=0,
            runtime=float(item["wall_seconds"]), termination_reason=termination,
            search_backend="lkh",
            backend_diagnostics={
                "called": bool(item["called"]),
                "completed": bool(item["completed"]),
                "timed_out": bool(item["timed_out"]),
                "accepted_output": bool(item["accepted_output"]),
                "native_output_upper_bound": item["native_output_upper_bound"],
            },
        ))
    finals = {tuple(search.final_sequence) for search in searches}
    return searches, {
        "budget_seconds": budget,
        "export_seconds": export_seconds,
        "quantization_scale": int(exported["scale"]),
        "rounding_error_bound": float(exported["route_rounding_error_bound"]),
        "native_calls": sum(bool(item["called"]) for item in raw),
        "completed_starts": sum(bool(item["completed"]) for item in raw),
        "timed_out_starts": sum(bool(item["timed_out"]) for item in raw),
        "accepted_outputs": sum(bool(item["accepted_output"]) for item in raw),
        "distinct_final_routes": len(finals),
    }


def hybrid_lkh_start_indices(
    start_count: int, quality_count: int, diversity_count: int, fraction: float,
) -> list[int]:
    """Top-K positions searched by LKH in the hybrid backend.

    Quality and diversity starts are interleaved (as in the LKH backend) so a
    partial fraction covers both kinds; the count is round(fraction * K).
    """

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("lkh_start_fraction must be in [0, 1]")
    count = int(round(fraction * start_count))
    if count <= 0:
        return []
    order: list[int] = []
    for offset in range(max(quality_count, diversity_count)):
        if offset < quality_count:
            order.append(offset)
        if offset < diversity_count:
            order.append(quality_count + offset)
    order.extend(index for index in range(start_count) if index not in set(order))
    return sorted(order[:count])


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
    if task.route_start_diversity_pool_size < 0:
        raise ValueError("route_start_diversity_pool_size must be non-negative")
    if task.route_start_diversity_max_relative_gap < 0.0:
        raise ValueError(
            "route_start_diversity_max_relative_gap must be non-negative"
        )
    if task.perturbation_restarts < 0:
        raise ValueError("perturbation_restarts must be non-negative")

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
            diversity_pool_size=task.route_start_diversity_pool_size,
            diversity_max_relative_gap=task.route_start_diversity_max_relative_gap,
        )
    )
    top_starts = [sampled[int(index)] for index in top_indices]

    searches: list[FixedAnchorLocalSearchResult] = []
    perturbation_attempts = 0
    perturbation_improvements = 0
    local_started = time.perf_counter()
    lkh_stats: dict[str, float | int] = {}
    backend = str(task.local_search_backend).strip().lower()
    if backend == "lkh":
        searches, lkh_stats = _run_lkh_fixed_anchor_searches(
            anchor_matrix, top_starts,
            quality_count=quality_start_count,
            diversity_count=diversity_start_count,
            task=task,
        )
    elif backend == "swap_two_opt":
        dual_lower_bound = None
        if task.use_dual_pruning:
            dual_lower_bound = FixedDualLowerBound.from_solution(instance, anchor_solution)
        nominal_starts = int(task.lkh_budget_starts)
        if nominal_starts:
            lkh_count = int(round(float(task.lkh_start_fraction) * nominal_starts))
            lkh_indices = list(range(len(top_starts))) if lkh_count else []
            lkh_share = lkh_count / nominal_starts
        else:
            lkh_indices = hybrid_lkh_start_indices(
                len(top_starts), quality_start_count, diversity_start_count,
                float(task.lkh_start_fraction),
            )
            lkh_share = len(lkh_indices) / max(len(top_starts), 1)
        lkh_by_index: dict[int, FixedAnchorLocalSearchResult] = {}
        if lkh_indices:
            subset = [top_starts[index] for index in lkh_indices]
            scaled_task = replace(
                task,
                lkh_budget_coefficient=float(task.lkh_budget_coefficient) * lkh_share,
            )
            lkh_searches, lkh_stats = _run_lkh_fixed_anchor_searches(
                anchor_matrix, subset,
                quality_count=len(subset), diversity_count=0, task=scaled_task,
            )
            lkh_by_index = dict(zip(lkh_indices, lkh_searches))
        for start_index, sequence in enumerate(top_starts):
            if start_index in lkh_by_index:
                searches.append(lkh_by_index[start_index])
                if not nominal_starts:
                    continue
            base_search = run_fixed_anchor_local_search(
                anchor_majorizer,
                sequence,
                task.local_search_config,
                dual_lower_bound=dual_lower_bound,
            )
            searches.append(base_search)

            # Optional double-bridge restarts for the move-search backend.
            restart_seed = tuple(base_search.final_sequence)
            for _ in range(task.perturbation_restarts):
                if base_search.termination_reason != "no_upper_bound_improving_move":
                    break
                perturbed = apply_double_bridge(restart_seed)
                if perturbed == restart_seed:
                    break
                perturbation_attempts += 1
                restarted_search = run_fixed_anchor_local_search(
                    anchor_majorizer,
                    perturbed,
                    task.local_search_config,
                    dual_lower_bound=dual_lower_bound,
                )
                searches.append(restarted_search)
                if (
                    restarted_search.final_upper_bound
                    < base_search.final_upper_bound
                    - task.local_search_config.acceptance_epsilon(
                        base_search.final_upper_bound
                    )
                ):
                    perturbation_improvements += 1
                restart_seed = tuple(restarted_search.final_sequence)
    else:
        raise ValueError(f"unsupported local search backend: {task.local_search_backend}")
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
        perturbation_attempts=int(perturbation_attempts),
        perturbation_improvements=int(perturbation_improvements),
        local_search_backend=backend,
        local_search_budget_seconds=float(lkh_stats.get("budget_seconds", 0.0)),
        lkh_problem_export_seconds=float(lkh_stats.get("export_seconds", 0.0)),
        lkh_quantization_scale=int(lkh_stats.get("quantization_scale", 0)),
        lkh_rounding_error_bound=float(lkh_stats.get("rounding_error_bound", 0.0)),
        lkh_native_calls=int(lkh_stats.get("native_calls", 0)),
        lkh_completed_starts=int(lkh_stats.get("completed_starts", 0)),
        lkh_timed_out_starts=int(lkh_stats.get("timed_out_starts", 0)),
        lkh_accepted_outputs=int(lkh_stats.get("accepted_outputs", 0)),
        lkh_distinct_final_routes=int(lkh_stats.get("distinct_final_routes", 0)),
        lkh_returned_anchor_starts=sum(
            tuple(search.final_sequence) == tuple(anchor) for search in searches
        ) if backend == "lkh" else 0,
    )


@dataclass
class C0RefreshTask:
    """One refresh round: C0 from the last exactly solved route, then search."""

    instance: CVTSPInstance
    base_solution: SPSolution
    pool_sequences: list[list[int]]
    solved_sequences: list[list[int]]
    incumbent_exact_makespan: float
    final_solver_config: SolverConfig
    local_search_config: FixedAnchorLocalSearchConfig
    route_start_top_k: int
    route_start_diversity_count: int
    route_start_diversity_pool_size: int
    route_start_diversity_max_relative_gap: float
    feasibility_tolerance: float
    upper_bound_tolerance: float
    return_matrices: bool = False


@dataclass
class C0RefreshOutcome:
    final_sequences: list[list[int]]
    candidate_sequence: list[int] | None
    candidate_upper_bound: float
    solution: SPSolution | None
    solve_failed: bool
    accepted: bool
    supervision_matrix: np.ndarray | None
    search_runtime: float
    solve_time: float
    base_matrix: np.ndarray | None = None
    solution_matrix: np.ndarray | None = None


def run_c0_refresh_task(task: C0RefreshTask) -> C0RefreshOutcome:
    """Search on a C0 that is tight around the most recently solved route.

    Exactly one unsolved final route (lowest C0) is solved. It is accepted only
    when its exact makespan is strictly below the current incumbent Expert.
    """

    instance = task.instance
    started = time.perf_counter()
    majorizer = FixedSolutionUpperBound.from_solution(
        instance,
        task.base_solution,
        feasibility_tolerance=task.feasibility_tolerance,
        require_optimal=True,
    )
    matrix = directed_total_contribution_matrix(majorizer)
    base = list(validate_tour(task.base_solution.sequence, instance.J))
    if not np.isclose(
        directed_route_upper_bound(base, matrix),
        float(task.base_solution.objective),
        rtol=0.0,
        atol=task.upper_bound_tolerance,
    ):
        raise RuntimeError(f"refresh C0 identity failed for {instance.instance_id}")
    pool = _deduplicate_tours(list(task.pool_sequences) + [base], instance.J)
    bounds = np.asarray([directed_route_upper_bound(route, matrix) for route in pool], dtype=float)
    indices, _, _ = select_quality_diverse_start_indices(
        pool,
        bounds,
        top_k=task.route_start_top_k,
        diversity_count=task.route_start_diversity_count,
        diversity_pool_size=task.route_start_diversity_pool_size,
        diversity_max_relative_gap=task.route_start_diversity_max_relative_gap,
    )
    searches = [
        run_fixed_anchor_local_search(majorizer, pool[int(index)], task.local_search_config)
        for index in indices
    ]
    search_runtime = time.perf_counter() - started
    finals = [list(search.final_sequence) for search in searches]
    solved = {tuple(int(node) for node in route) for route in task.solved_sequences}
    candidates = [
        (float(search.final_upper_bound), list(search.final_sequence))
        for search in searches
        if tuple(search.final_sequence) not in solved
    ]
    base_matrix = matrix if task.return_matrices else None
    if not candidates:
        return C0RefreshOutcome(finals, None, float("inf"), None, False, False, None,
                                search_runtime, 0.0, base_matrix)
    upper_bound, candidate = min(candidates, key=lambda item: item[0])
    solution = solve(instance, candidate, task.final_solver_config)
    solve_time = float(solution.solve_time) if np.isfinite(float(solution.solve_time)) else 0.0
    if not is_optimal_gurobi_solution(solution) or not np.isfinite(float(solution.objective)):
        return C0RefreshOutcome(finals, candidate, upper_bound, None, True, False, None,
                                search_runtime, solve_time, base_matrix)
    if tuple(validate_tour(solution.sequence, instance.J)) != tuple(candidate):
        raise RuntimeError("refresh solver returned a different fixed route")
    exact = float(solution.objective)
    if exact > upper_bound + task.upper_bound_tolerance:
        raise RuntimeError(
            f"refresh upper-bound violation: instance={instance.instance_id}, "
            f"exact={exact:.12g}, upper={upper_bound:.12g}"
        )
    epsilon = task.local_search_config.acceptance_epsilon(task.incumbent_exact_makespan)
    accepted = exact < float(task.incumbent_exact_makespan) - epsilon
    supervision = None
    if accepted or task.return_matrices:
        supervision = directed_total_contribution_matrix(FixedSolutionUpperBound.from_solution(
            instance, solution, feasibility_tolerance=task.feasibility_tolerance,
            require_optimal=True,
        ))
    return C0RefreshOutcome(finals, candidate, upper_bound, solution, False, accepted,
                            supervision if accepted else None, search_runtime, solve_time,
                            base_matrix, supervision if task.return_matrices else None)


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


def fixed_start_two_opt(
    matrix: np.ndarray,
    sequence: Iterable[int],
    *,
    max_moves: int,
    acceptance_epsilon: float,
) -> tuple[list[int], float]:
    """Best-improvement asymmetric 2-opt on a directed C0 table, first target fixed.

    The route is depot -> sequence -> depot; matrix index 0 is the depot and
    target j is index j + 1. Reversals only touch positions 2..J so the first
    visited target (the POMO start) never changes. Returns the final route and
    its table value.
    """

    costs = np.asarray(matrix, dtype=float)
    route = [int(node) for node in sequence]
    if len(route) < 3:
        return route, directed_route_upper_bound(route, costs)
    for _ in range(max(0, int(max_moves))):
        nodes = np.asarray([0] + [node + 1 for node in route] + [0], dtype=np.int64)
        forward = np.concatenate(([0.0], np.cumsum(costs[nodes[:-1], nodes[1:]])))
        backward = np.concatenate(([0.0], np.cumsum(costs[nodes[1:], nodes[:-1]])))
        n = len(route)
        # Reverse the segment of route positions [i, k] (nodes index i..k), 2 <= i < k <= n.
        i_idx, k_idx = np.triu_indices(n + 1, k=1)
        keep = i_idx >= 2
        i_idx, k_idx = i_idx[keep], k_idx[keep]
        delta = (
            costs[nodes[i_idx - 1], nodes[k_idx]]
            + (backward[k_idx] - backward[i_idx])
            + costs[nodes[i_idx], nodes[k_idx + 1]]
            - costs[nodes[i_idx - 1], nodes[i_idx]]
            - (forward[k_idx] - forward[i_idx])
            - costs[nodes[k_idx], nodes[k_idx + 1]]
        )
        best = int(np.argmin(delta)) if delta.size else -1
        if best < 0 or delta[best] >= -acceptance_epsilon:
            break
        i, k = int(i_idx[best]) - 1, int(k_idx[best]) - 1  # route positions
        route[i : k + 1] = reversed(route[i : k + 1])
    return route, directed_route_upper_bound(route, costs)


@dataclass
class PerStartLabelTask:
    """Certified self-improvement of the policy's own routes from chosen starts."""

    instance: CVTSPInstance
    routes: list[list[int]]
    solver_config: SolverConfig
    max_moves: int
    absolute_epsilon: float
    relative_epsilon: float
    feasibility_tolerance: float
    upper_bound_tolerance: float


@dataclass
class PerStartLabel:
    original_sequence: list[int]
    original_exact: float
    label_sequence: list[int] | None
    label_upper_bound: float
    solve_failed: bool


def run_per_start_label_task(task: PerStartLabelTask) -> list[PerStartLabel]:
    """Solve each route exactly, then 2-opt it on its own C0 with the start fixed.

    A label is returned only when its own-route C0 value is strictly below the
    original exact makespan, which certifies an exact improvement without a
    second solve (exact(label) <= C0(label) < exact(original)).
    """

    out = []
    for route in task.routes:
        route = list(validate_tour(route, task.instance.J))
        solution = solve(task.instance, route, task.solver_config)
        if not is_optimal_gurobi_solution(solution) or not np.isfinite(float(solution.objective)):
            out.append(PerStartLabel(route, float("nan"), None, float("inf"), True))
            continue
        exact = float(solution.objective)
        matrix = directed_total_contribution_matrix(FixedSolutionUpperBound.from_solution(
            task.instance, solution, feasibility_tolerance=task.feasibility_tolerance,
            require_optimal=True,
        ))
        if abs(directed_route_upper_bound(route, matrix) - exact) > task.upper_bound_tolerance:
            raise RuntimeError(f"per-start C0 identity failed for {task.instance.instance_id}")
        epsilon = max(task.absolute_epsilon, task.relative_epsilon * max(1.0, abs(exact)))
        improved, upper = fixed_start_two_opt(
            matrix, route, max_moves=task.max_moves, acceptance_epsilon=epsilon)
        if improved[0] != route[0]:
            raise RuntimeError("fixed-start 2-opt changed the start node")
        certified = improved != route and upper < exact - epsilon
        out.append(PerStartLabel(route, exact, improved if certified else None, upper, False))
    return out
