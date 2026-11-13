"""Ablation B: self-improvement with real Gurobi instead of the shadow-price table.

``run_exact_self_improvement_task`` is a drop-in replacement for
``run_sparse_self_improvement_task``. Without the fixed-anchor upper-bound
table C0 every route the search looks at must be solved exactly, and one full
2-opt neighbourhood alone is J(J-1)/2 solves, so the search is cut to what
training to E100 can afford (1028 meeting: "fewer local-search iterations"):

  * one start, the policy's anchor route (scoring the 200-route Reference pool
    to pick starts would itself cost 200 solves);
  * first-improvement 2-opt in a seeded random move order, each candidate
    solved exactly; an improving move is accepted at once and the scan restarts
    from the new route;
  * at most ``CVTSP_EXACT_BUDGET`` (default 100) search solves per instance; the
    search also stops at a 2-opt local optimum.

The Expert is the best route found (already exact). The result keeps the
trainer's anchor/expert call fields (1 + at most 1, the Expert being re-solved
with duals for the supervision table) so the unchanged training loop accepts
it; the search's own solves are in ``backend_diagnostics["exact_calls"]`` and,
when ``CVTSP_EXACT_PROGRESS`` is set, one JSONL line per instance goes there.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import replace

import numpy as np

from src.CertifiedLocalSearch import FixedAnchorLocalSearchResult
from src.CertifiedUpperBound import (
    FixedSolutionUpperBound,
    _materialize_move,
    iter_candidate_move_indices,
    validate_tour,
)
from src.CVPSolver import solve
from src.DualContributions import directed_total_contribution_matrix
from src.JointSelfImprovement import (
    SparseSelfImprovementResult,
    SparseSelfImprovementTask,
    _deduplicate_tours,
    _require_optimal,
)


class _ExactScorer:
    """Exact makespan of a route, cached; counts the solves it really makes."""

    def __init__(self, instance, solver_config):
        self.instance = instance
        self.config = replace(solver_config, collect_duals=False)
        self.cache: dict[tuple[int, ...], float] = {}
        self.calls = 0
        self.seconds = 0.0

    def __call__(self, tour: tuple[int, ...]) -> float:
        value = self.cache.get(tour)
        if value is None:
            solution = solve(self.instance, list(tour), self.config)
            _require_optimal(solution, "exact search", self.instance.instance_id)
            self.calls += 1
            self.seconds += float(solution.solve_time)
            value = self.cache[tour] = float(solution.objective)
        return value


def _log_progress(record: dict) -> None:
    path = os.environ.get("CVTSP_EXACT_PROGRESS")
    if path:
        with open(path, "a") as fp:
            fp.write(json.dumps(record) + "\n")


def exact_first_improvement_search(scorer: _ExactScorer, start, config, *, budget: int,
                                   seed: int) -> FixedAnchorLocalSearchResult:
    """First-improvement 2-opt where every candidate is solved exactly."""

    rng = random.Random(seed)
    current = validate_tour(start, scorer.instance.J)
    initial_value = current_value = scorer(current)
    calls_before, began = scorer.calls, time.perf_counter()
    evaluated = accepted = 0
    termination = "exact_budget_exhausted"
    while scorer.calls - calls_before < budget:
        moves = list(iter_candidate_move_indices(current, move_types=config.move_types))
        rng.shuffle(moves)
        epsilon = config.acceptance_epsilon(current_value)
        improved = False
        for kind, first, second in moves:
            if scorer.calls - calls_before >= budget:
                break
            candidate = _materialize_move(current, kind, first, second)
            value = scorer(candidate)
            evaluated += 1
            if value < current_value - epsilon:
                current, current_value, improved = candidate, value, True
                accepted += 1
                break
        else:
            termination = "no_exact_improving_move"
            break
        if not improved:
            break
    return FixedAnchorLocalSearchResult(
        initial_sequence=list(start), final_sequence=list(current),
        initial_upper_bound=initial_value, final_upper_bound=current_value,
        generated=evaluated, evaluated=evaluated, dual_pruned=0, accepted=accepted,
        runtime=time.perf_counter() - began, termination_reason=termination,
        search_backend="exact_first_improvement",
        backend_diagnostics={"exact_calls": scorer.calls - calls_before},
    )


def run_exact_self_improvement_task(task: SparseSelfImprovementTask) -> SparseSelfImprovementResult:
    started = time.perf_counter()
    instance = task.instance
    budget = int(os.environ.get("CVTSP_EXACT_BUDGET", "100"))
    anchor = list(validate_tour(task.anchor_sequence, instance.J))
    sampled = _deduplicate_tours(task.sampled_sequences, instance.J)

    anchor_solution = solve(instance, anchor, task.anchor_solver_config)
    _require_optimal(anchor_solution, "anchor", instance.instance_id)
    # C0 is still built because the unchanged RL envelope reads it; the search
    # below never uses it.
    anchor_matrix = directed_total_contribution_matrix(FixedSolutionUpperBound.from_solution(
        instance, anchor_solution, feasibility_tolerance=task.feasibility_tolerance, require_optimal=True))
    scorer = _ExactScorer(instance, task.final_solver_config)
    scorer.cache[tuple(anchor)] = float(anchor_solution.objective)
    search = exact_first_improvement_search(scorer, anchor, task.local_search_config, budget=budget,
                                            seed=task.local_search_seed)

    expert_sequence, expert_value = anchor, float(anchor_solution.objective)
    expert_solution, expert_calls, expert_seconds = anchor_solution, 0, 0.0
    if search.accepted:
        expert_sequence, expert_value = list(search.final_sequence), float(search.final_upper_bound)
        # re-solved with duals for the supervision table (the cache keeps objectives only)
        expert_solution = solve(instance, expert_sequence, task.final_solver_config)
        _require_optimal(expert_solution, "expert", instance.instance_id)
        expert_calls, expert_seconds = 1, float(expert_solution.solve_time)
    supervision_matrix = directed_total_contribution_matrix(FixedSolutionUpperBound.from_solution(
        instance, expert_solution, feasibility_tolerance=task.feasibility_tolerance, require_optimal=True))
    _log_progress({"instance_id": instance.instance_id, "J": int(instance.J),
                   "search_calls": scorer.calls, "total_calls": scorer.calls + 1 + expert_calls,
                   "accepted": search.accepted, "termination": search.termination_reason,
                   "anchor": float(anchor_solution.objective), "expert": float(expert_solution.objective),
                   "seconds": time.perf_counter() - started, "pid": os.getpid(), "finished_at": time.time()})

    return SparseSelfImprovementResult(
        instance_id=instance.instance_id,
        anchor_sequence=anchor,
        anchor_solution=anchor_solution,
        anchor_exact_makespan=float(anchor_solution.objective),
        anchor_upper_bound_matrix=anchor_matrix,
        # the pool is not scored; NaN keeps the field's shape without inventing values
        sampled_upper_bounds=np.full(len(sampled), np.nan),
        top_k_start_sequences=[anchor],
        local_search_results=[search],
        expert_sequence=list(expert_sequence),
        expert_upper_bound=float(expert_value),
        expert_solution=expert_solution,
        expert_exact_makespan=float(expert_solution.objective),
        supervision_upper_bound_matrix=supervision_matrix,
        anchor_gurobi_calls=1,
        expert_gurobi_calls=expert_calls,
        gurobi_runtime=float(anchor_solution.solve_time + expert_seconds + scorer.seconds),
        local_search_runtime=float(search.runtime),
        total_runtime=time.perf_counter() - started,
        generated_moves=search.generated,
        evaluated_moves=search.evaluated,
        dual_pruned_moves=0,
        accepted_moves=search.accepted,
        linear_dual_count=len(anchor_solution.linear_duals),
        quadratic_dual_count=len(anchor_solution.quadratic_duals),
        expert_source="exact_search" if expert_calls else "anchor",
        rl_baseline_upper_bound=float(expert_value),
        quality_start_count=1,
        local_search_backend="exact_first_improvement",
    )
