from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable

from src.CertifiedUpperBound import (
    FixedSolutionUpperBound,
    TourMove,
    find_best_fixed_anchor_move,
    find_best_certified_move,
    find_best_certified_move_ranked,
    find_best_certified_move_dual_pruned,
    find_best_certified_move_ranked_dual_pruned,
    is_optimal_gurobi_solution,
    validate_tour,
)
from src.CVPSolver import SPSolution, SolverConfig, solve
from src.DualContributions import FixedDualLowerBound
from src.TSProblemDef import CVTSPInstance


@dataclass(frozen=True)
class FixedAnchorLocalSearchConfig:
    """Configuration for a solver-free search on one anchor majorizer."""

    max_local_moves: int = 30
    absolute_epsilon: float = 1e-6
    relative_epsilon: float = 1e-8
    move_types: tuple[str, ...] = ("swap", "two_opt")
    use_dual_pruning: bool = False

    def acceptance_epsilon(self, upper_bound: float) -> float:
        return max(
            float(self.absolute_epsilon),
            float(self.relative_epsilon) * max(1.0, abs(float(upper_bound))),
        )


@dataclass(frozen=True)
class FixedAnchorAcceptedMove:
    iteration: int
    move: TourMove
    old_upper_bound: float
    new_upper_bound: float
    upper_bound_improvement: float

    def to_dict(self) -> dict:
        return {
            "iteration": int(self.iteration),
            "move": self.move.to_dict(),
            "old_upper_bound": float(self.old_upper_bound),
            "new_upper_bound": float(self.new_upper_bound),
            "upper_bound_improvement": float(self.upper_bound_improvement),
        }


@dataclass
class FixedAnchorLocalSearchResult:
    """Result of a multi-move chain that never refreshes its Gurobi anchor."""

    initial_sequence: list[int]
    final_sequence: list[int]
    initial_upper_bound: float
    final_upper_bound: float
    generated: int
    evaluated: int
    dual_pruned: int
    accepted: int
    runtime: float
    termination_reason: str
    history: list[FixedAnchorAcceptedMove] = field(default_factory=list)

    @property
    def upper_bound_improvement(self) -> float:
        return float(self.initial_upper_bound - self.final_upper_bound)

    @property
    def upper_bound_history(self) -> list[float]:
        return [
            float(self.initial_upper_bound),
            *(float(item.new_upper_bound) for item in self.history),
        ]

    def to_dict(self) -> dict:
        return {
            "initial_sequence": list(self.initial_sequence),
            "final_sequence": list(self.final_sequence),
            "initial_upper_bound": float(self.initial_upper_bound),
            "final_upper_bound": float(self.final_upper_bound),
            "upper_bound_improvement": float(self.upper_bound_improvement),
            "upper_bound_history": self.upper_bound_history,
            "generated": int(self.generated),
            "evaluated": int(self.evaluated),
            "dual_pruned": int(self.dual_pruned),
            "accepted": int(self.accepted),
            "runtime": float(self.runtime),
            "termination_reason": self.termination_reason,
            "history": [item.to_dict() for item in self.history],
        }


def run_fixed_anchor_local_search(
    majorizer: FixedSolutionUpperBound,
    initial_sequence: Iterable[int],
    config: FixedAnchorLocalSearchConfig,
    dual_lower_bound: FixedDualLowerBound | None = None,
) -> FixedAnchorLocalSearchResult:
    """Run up to ``max_local_moves`` using one fixed-anchor UB edge table.

    The function deliberately has no solver configuration and never performs
    an exact solve.  Every neighborhood scan and accepted swap/asymmetric
    2-opt/relocate is evaluated against the same ``majorizer``.
    """

    if config.max_local_moves < 0:
        raise ValueError("max_local_moves must be non-negative")
    if config.absolute_epsilon < 0 or config.relative_epsilon < 0:
        raise ValueError("local-search epsilons must be non-negative")
    if config.absolute_epsilon == 0 and config.relative_epsilon == 0:
        raise ValueError("at least one local-search epsilon must be positive")
    if config.use_dual_pruning and dual_lower_bound is None:
        raise ValueError("use_dual_pruning=True requires dual_lower_bound")
    if dual_lower_bound is not None:
        if dual_lower_bound.instance_id != majorizer.instance_id:
            raise ValueError("dual lower bound and majorizer belong to different instances")
        if dual_lower_bound.problem_size != majorizer.problem_size:
            raise ValueError("dual lower bound and majorizer have different problem sizes")

    initial = validate_tour(initial_sequence, majorizer.problem_size)
    current = initial
    initial_upper_bound = float(majorizer.upper_bound(initial))
    current_upper_bound = initial_upper_bound
    generated = 0
    evaluated = 0
    dual_pruned = 0
    history: list[FixedAnchorAcceptedMove] = []
    started = time.perf_counter()
    termination_reason = "max_local_moves_reached"

    for iteration in range(1, config.max_local_moves + 1):
        epsilon = config.acceptance_epsilon(current_upper_bound)
        best_move, generated_count, evaluated_count, dual_pruned_count = (
            find_best_fixed_anchor_move(
                majorizer,
                current,
                epsilon=epsilon,
                move_types=config.move_types,
                candidate_lower_bound=(
                    dual_lower_bound.lower_bound
                    if config.use_dual_pruning and dual_lower_bound is not None
                    else None
                ),
            )
        )
        generated += generated_count
        evaluated += evaluated_count
        dual_pruned += dual_pruned_count
        if best_move is None:
            termination_reason = "no_upper_bound_improving_move"
            break

        # Recompute once from the full directed table as an invariant check.
        # This guards the O(1) asymmetric 2-opt delta, including all reversed
        # internal edges, without involving Gurobi.
        recomputed = float(majorizer.upper_bound(best_move.sequence))
        numerical_tolerance = max(1e-10, 1e-10 * max(1.0, abs(recomputed)))
        if abs(recomputed - best_move.upper_bound) > numerical_tolerance:
            raise RuntimeError(
                "incremental fixed-anchor upper bound is inconsistent with full route sum: "
                f"incremental={best_move.upper_bound:.12g}, full={recomputed:.12g}"
            )
        if recomputed >= current_upper_bound - epsilon:
            raise RuntimeError(
                "local search attempted to accept a non-improving fixed-anchor move: "
                f"old={current_upper_bound:.12g}, new={recomputed:.12g}"
            )

        history.append(
            FixedAnchorAcceptedMove(
                iteration=iteration,
                move=best_move,
                old_upper_bound=current_upper_bound,
                new_upper_bound=recomputed,
                upper_bound_improvement=current_upper_bound - recomputed,
            )
        )
        current = best_move.sequence
        current_upper_bound = recomputed
    else:
        termination_reason = "max_local_moves_reached"

    return FixedAnchorLocalSearchResult(
        initial_sequence=list(initial),
        final_sequence=list(current),
        initial_upper_bound=initial_upper_bound,
        final_upper_bound=current_upper_bound,
        generated=generated,
        evaluated=evaluated,
        dual_pruned=dual_pruned,
        accepted=len(history),
        runtime=time.perf_counter() - started,
        termination_reason=termination_reason,
        history=history,
    )


@dataclass(frozen=True)
class CertifiedLocalSearchConfig:
    max_refreshes: int = 10
    absolute_epsilon: float = 1e-4
    relative_epsilon: float = 1e-8
    upper_bound_tolerance: float = 2e-5
    feasibility_tolerance: float = 1e-5
    move_types: tuple[str, ...] = ("swap", "relocate", "two_opt")
    require_optimal: bool = True

    def acceptance_epsilon(self, objective: float) -> float:
        return max(self.absolute_epsilon, self.relative_epsilon * max(1.0, abs(objective)))


@dataclass(frozen=True)
class AcceptedMove:
    iteration: int
    move: TourMove
    old_objective: float
    new_objective: float
    predicted_upper_bound: float
    certified_margin: float
    realized_improvement: float
    solve_time: float
    max_anchor_feasibility_violation: float

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["move"] = self.move.to_dict()
        return payload


@dataclass
class CertifiedLocalSearchResult:
    instance_id: str
    initial_sequence: list[int]
    final_sequence: list[int]
    initial_objective: float
    final_objective: float
    generated_moves: int
    evaluated_moves: int
    dual_pruned_moves: int
    accepted_moves: int
    gurobi_calls: int
    runtime: float
    termination_reason: str
    history: list[AcceptedMove] = field(default_factory=list)
    final_solution: SPSolution | None = field(default=None, repr=False)

    @property
    def objective_improvement(self) -> float:
        return self.initial_objective - self.final_objective

    @property
    def pruned_moves(self) -> int:
        return self.generated_moves - self.evaluated_moves

    @property
    def relative_improvement_percent(self) -> float:
        if self.initial_objective == 0:
            return 0.0
        return 100.0 * self.objective_improvement / abs(self.initial_objective)

    def to_dict(self, include_solution: bool = True) -> dict:
        payload = {
            "instance_id": self.instance_id,
            "initial_sequence": list(self.initial_sequence),
            "final_sequence": list(self.final_sequence),
            "initial_objective": self.initial_objective,
            "final_objective": self.final_objective,
            "objective_improvement": self.objective_improvement,
            "relative_improvement_percent": self.relative_improvement_percent,
            "generated_moves": self.generated_moves,
            "evaluated_moves": self.evaluated_moves,
            "dual_pruned_moves": self.dual_pruned_moves,
            "pruned_moves": self.pruned_moves,
            "accepted_moves": self.accepted_moves,
            "gurobi_calls": self.gurobi_calls,
            "runtime": self.runtime,
            "termination_reason": self.termination_reason,
            "history": [item.to_dict() for item in self.history],
        }
        if include_solution and self.final_solution is not None:
            payload["final_solution"] = self.final_solution.to_dict()
        return payload


def run_certified_local_search(
    instance: CVTSPInstance,
    initial_sequence: Iterable[int],
    *,
    solver_config: SolverConfig | None = None,
    search_config: CertifiedLocalSearchConfig | None = None,
    candidate_ranker: Callable[[tuple[int, ...]], float] | None = None,
    max_exact_candidates: int | None = None,
    initial_solution: SPSolution | None = None,
    use_dual_lower_bound: bool = False,
) -> CertifiedLocalSearchResult:
    """Improve a tour using cheap fixed-solution UBs and exact refreshes.

    Exactly one candidate is sent to Gurobi after each neighborhood scan.  No
    rejected candidate triggers a solver call.
    """

    solver_config = solver_config or SolverConfig(solver_backend="gurobi", gurobi_threads=1)
    search_config = search_config or CertifiedLocalSearchConfig()
    if str(solver_config.solver_backend).strip().lower() != "gurobi":
        raise ValueError("certified local search requires solver_backend='gurobi'")
    if search_config.max_refreshes < 0:
        raise ValueError("max_refreshes must be non-negative")
    if max_exact_candidates is not None and candidate_ranker is None:
        raise ValueError("max_exact_candidates requires a candidate_ranker")
    if max_exact_candidates is not None and max_exact_candidates <= 0:
        raise ValueError("max_exact_candidates must be positive")

    initial = validate_tour(initial_sequence, instance.J)
    start_time = time.perf_counter()
    if initial_solution is None:
        current_solution = _solve_exact(instance, initial, solver_config, search_config.require_optimal)
        gurobi_calls = 1
    else:
        if tuple(initial_solution.sequence) != initial:
            raise ValueError("initial_solution sequence does not match initial_sequence")
        if not initial_solution.success:
            raise ValueError("initial_solution is unsuccessful")
        if search_config.require_optimal and not is_optimal_gurobi_solution(initial_solution):
            raise ValueError("initial_solution is not proven optimal")
        current_solution = initial_solution
        gurobi_calls = 0
    initial_objective = float(current_solution.objective)
    generated_moves = 0
    evaluated_moves = 0
    dual_pruned_moves = 0
    history: list[AcceptedMove] = []
    termination_reason = "max_refreshes_reached"

    for iteration in range(1, search_config.max_refreshes + 1):
        majorizer = FixedSolutionUpperBound.from_solution(
            instance,
            current_solution,
            feasibility_tolerance=search_config.feasibility_tolerance,
            require_optimal=search_config.require_optimal,
        )
        epsilon = search_config.acceptance_epsilon(current_solution.objective)
        dual_lower_bound = (
            FixedDualLowerBound.from_solution(instance, current_solution)
            if use_dual_lower_bound
            else None
        )
        if dual_lower_bound is not None and candidate_ranker is not None and max_exact_candidates is not None:
            best_move, generated_count, dual_pruned_count, exact_count = (
                find_best_certified_move_ranked_dual_pruned(
                    majorizer,
                    dual_lower_bound,
                    current_solution.objective,
                    epsilon=epsilon,
                    ranker=candidate_ranker,
                    max_exact_candidates=max_exact_candidates,
                    move_types=search_config.move_types,
                )
            )
        elif dual_lower_bound is not None:
            best_move, generated_count, dual_pruned_count, exact_count = (
                find_best_certified_move_dual_pruned(
                    majorizer,
                    dual_lower_bound,
                    current_solution.objective,
                    epsilon=epsilon,
                    move_types=search_config.move_types,
                )
            )
        elif candidate_ranker is not None and max_exact_candidates is not None:
            best_move, generated_count, exact_count = find_best_certified_move_ranked(
                majorizer,
                current_solution.objective,
                epsilon=epsilon,
                ranker=candidate_ranker,
                max_exact_candidates=max_exact_candidates,
                move_types=search_config.move_types,
            )
            dual_pruned_count = 0
        else:
            best_move, exact_count = find_best_certified_move(
                majorizer,
                current_solution.objective,
                epsilon=epsilon,
                move_types=search_config.move_types,
            )
            generated_count = exact_count
            dual_pruned_count = 0
        generated_moves += generated_count
        evaluated_moves += exact_count
        dual_pruned_moves += dual_pruned_count
        if best_move is None:
            termination_reason = "no_certified_improving_move"
            break

        old_solution = current_solution
        current_solution = _solve_exact(
            instance,
            best_move.sequence,
            solver_config,
            search_config.require_optimal,
        )
        gurobi_calls += 1

        if current_solution.objective > best_move.upper_bound + search_config.upper_bound_tolerance:
            raise RuntimeError(
                "upper-bound violation after accepted move: "
                f"true={current_solution.objective:.12g}, "
                f"upper={best_move.upper_bound:.12g}, "
                f"tolerance={search_config.upper_bound_tolerance:.3e}"
            )
        if current_solution.objective >= old_solution.objective:
            raise RuntimeError(
                "a certified move failed to strictly improve the true Gurobi objective: "
                f"old={old_solution.objective:.12g}, new={current_solution.objective:.12g}, "
                f"upper={best_move.upper_bound:.12g}"
            )

        history.append(
            AcceptedMove(
                iteration=iteration,
                move=best_move,
                old_objective=float(old_solution.objective),
                new_objective=float(current_solution.objective),
                predicted_upper_bound=float(best_move.upper_bound),
                certified_margin=float(old_solution.objective - best_move.upper_bound),
                realized_improvement=float(old_solution.objective - current_solution.objective),
                solve_time=float(current_solution.solve_time),
                max_anchor_feasibility_violation=float(majorizer.feasibility.max_violation),
            )
        )
    else:
        termination_reason = "max_refreshes_reached"

    return CertifiedLocalSearchResult(
        instance_id=instance.instance_id,
        initial_sequence=list(initial),
        final_sequence=list(current_solution.sequence),
        initial_objective=initial_objective,
        final_objective=float(current_solution.objective),
        generated_moves=generated_moves,
        evaluated_moves=evaluated_moves,
        dual_pruned_moves=dual_pruned_moves,
        accepted_moves=len(history),
        gurobi_calls=gurobi_calls,
        runtime=time.perf_counter() - start_time,
        termination_reason=termination_reason,
        history=history,
        final_solution=current_solution,
    )


def _solve_exact(
    instance: CVTSPInstance,
    sequence: Iterable[int],
    solver_config: SolverConfig,
    require_optimal: bool,
) -> SPSolution:
    solution = solve(instance, list(sequence), solver_config)
    if not solution.success:
        raise RuntimeError(
            f"Gurobi failed for {instance.instance_id}: {solution.status} {solution.raw_debug}"
        )
    if require_optimal and not is_optimal_gurobi_solution(solution):
        raise RuntimeError(
            f"Gurobi did not prove optimality for {instance.instance_id}: status={solution.status}"
        )
    return solution
