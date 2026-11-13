"""Sequence-space metaheuristic baselines for the CVTSP (SA, VNS, ILS).

All three search over target visiting orders and score every order with the
same exact fixed-sequence SOCP (Gurobi) used by the learned policies, so the
only difference to our method is how candidate orders are generated. They
share one initial order (nearest neighbour + Euclidean 2-opt, no SOCP calls)
and one exact evaluator with an orientation-free cache (a route and its
reverse have the same makespan). The evaluator records every improvement as
(evaluations, seconds, best makespan), so one long run yields the whole
quality-vs-budget curve.

Local search uses granular neighbourhoods (moves that create an edge to one
of the k nearest targets), the standard way to keep exact-evaluated 2-opt /
relocate passes affordable at J = 100-200.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from src.CVPSolver import SolverConfig, solve


class BudgetExhausted(Exception):
    pass


@dataclass
class SearchResult:
    method: str
    best_sequence: list[int]
    best_makespan: float
    initial_makespan: float
    evaluations: int
    cache_hits: int
    seconds: float
    trace: list[tuple[int, float, float]] = field(default_factory=list)


class ExactEvaluator:
    """Exact SOCP makespan with a cache, a time/evaluation budget and an improvement trace."""

    def __init__(self, instance, solver: SolverConfig, *, env: Any = None, time_budget: float = math.inf,
                 max_evaluations: int | None = None, objective: Callable[[list[int]], float] | None = None):
        self.instance = instance
        self.solver = solver
        self.env = env
        self.time_budget = float(time_budget)
        self.max_evaluations = max_evaluations
        self._objective = objective
        self.cache: dict[tuple[int, ...], float] = {}
        self.evaluations = 0
        self.cache_hits = 0
        self.best_value = math.inf
        self.best_sequence: list[int] | None = None
        self.trace: list[tuple[int, float, float]] = []
        self.started = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def expired(self) -> bool:
        if self.max_evaluations is not None and (
                self.evaluations >= self.max_evaluations
                # tiny instances can exhaust every order; stop cycling through the cache
                or self.cache_hits >= 100 * self.max_evaluations):
            return True
        return self.elapsed() >= self.time_budget

    @staticmethod
    def key(sequence) -> tuple[int, ...]:
        forward = tuple(int(x) for x in sequence)
        backward = forward[::-1]
        return min(forward, backward)

    def __call__(self, sequence) -> float:
        if self.expired():
            raise BudgetExhausted
        key = self.key(sequence)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        if self._objective is not None:
            value = float(self._objective(list(sequence)))
        else:
            solution = solve(self.instance, list(sequence), self.solver, env=self.env)
            if not solution.success or str(solution.status).strip() != "2" or not math.isfinite(solution.objective):
                raise RuntimeError(f"non-optimal SOCP on {self.instance.instance_id}: {solution.status}")
            value = float(solution.objective)
        self.evaluations += 1
        self.cache[key] = value
        if value < self.best_value - 1e-9:
            self.best_value = value
            self.best_sequence = list(sequence)
            self.trace.append((self.evaluations, self.elapsed(), value))
        return value


# ----------------------------------------------------------------- geometry
def _distance_matrix(instance) -> np.ndarray:
    points = np.vstack([np.asarray(instance.depot, float)[None, :], np.asarray(instance.targets, float)])
    return np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)


def euclidean_tour(instance) -> list[int]:
    """Nearest-neighbour tour from the depot improved by Euclidean 2-opt (targets 0-based)."""
    dist = _distance_matrix(instance)
    n = instance.J
    tour, left, current = [], set(range(1, n + 1)), 0
    while left:
        nxt = min(left, key=lambda j: dist[current, j])
        tour.append(nxt)
        left.remove(nxt)
        current = nxt
    route = [0] + tour + [0]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                a, b, c, d = route[i - 1], route[i], route[j], route[j + 1]
                if dist[a, c] + dist[b, d] < dist[a, b] + dist[c, d] - 1e-10:
                    route[i:j + 1] = route[i:j + 1][::-1]
                    improved = True
    return [node - 1 for node in route[1:-1]]


def nearest_targets(instance, k: int) -> list[list[int]]:
    dist = _distance_matrix(instance)[1:, 1:]
    np.fill_diagonal(dist, np.inf)
    k = min(k, instance.J - 1)
    return [list(map(int, np.argsort(row)[:k])) for row in dist]


# -------------------------------------------------------------------- moves
def two_opt_move(seq: list[int], i: int, j: int) -> list[int]:
    return seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]


def relocate_move(seq: list[int], i: int, j: int) -> list[int]:
    """Remove seq[i] and insert it so that it ends up at position j."""
    out = seq[:i] + seq[i + 1:]
    out.insert(j, seq[i])
    return out


def swap_move(seq: list[int], i: int, j: int) -> list[int]:
    out = list(seq)
    out[i], out[j] = out[j], out[i]
    return out


def or_opt_move(seq: list[int], i: int, length: int, j: int) -> list[int]:
    segment = seq[i:i + length]
    rest = seq[:i] + seq[i + length:]
    j = min(j, len(rest))
    return rest[:j] + segment + rest[j:]


def double_bridge(seq: list[int], rng: np.random.Generator) -> list[int]:
    n = len(seq)
    if n < 8:
        return random_move(seq, rng)
    a, b, c = sorted(rng.choice(np.arange(1, n), size=3, replace=False))
    return seq[:a] + seq[b:c] + seq[a:b] + seq[c:]


def random_move(seq: list[int], rng: np.random.Generator) -> list[int]:
    n = len(seq)
    kind = int(rng.integers(4))
    i, j = sorted(rng.choice(n, size=2, replace=False))
    if kind == 0:
        return two_opt_move(seq, i, j)
    if kind == 1:
        return relocate_move(seq, int(rng.integers(n)), int(rng.integers(n)))
    if kind == 2:
        return swap_move(seq, i, j)
    length = int(rng.integers(2, 4))
    start = int(rng.integers(0, max(n - length, 0) + 1))
    return or_opt_move(seq, start, length, int(rng.integers(0, n - length + 1)))


# -------------------------------------------------------------- local search
def granular_local_search(seq: list[int], value: float, evaluate: ExactEvaluator, neighbours: list[list[int]],
                          rng: np.random.Generator) -> tuple[list[int], float]:
    """First-improvement 2-opt + relocate restricted to k-nearest edges, until a pass finds nothing."""
    improved = True
    while improved:
        improved = False
        position = {node: idx for idx, node in enumerate(seq)}
        for node in rng.permutation(len(seq)):
            node = int(node)
            for other in neighbours[node]:
                i, j = position[node], position[other]
                candidates = []
                # 2-opt making (node, other) adjacent: reverse the segment between them.
                if abs(i - j) > 1:
                    lo, hi = min(i, j), max(i, j)
                    candidates.append(two_opt_move(seq, lo + 1, hi))
                # relocate node right after / before other.
                target = j if i < j else j + 1
                if target != i:
                    candidates.append(relocate_move(seq, i, target))
                target = j - 1 if i < j else j
                if target != i and target >= 0:
                    candidates.append(relocate_move(seq, i, target))
                for candidate in candidates:
                    candidate_value = evaluate(candidate)
                    if candidate_value < value - 1e-9:
                        seq, value, improved = candidate, candidate_value, True
                        break
                if improved:
                    break
            if improved:
                break
    return seq, value


# ------------------------------------------------------------------ methods
def simulated_annealing(seq, value, evaluate: ExactEvaluator, rng, *, end_ratio: float = 1e-3,
                        calibration_moves: int = 20, initial_accept: float = 0.5):
    """Metropolis acceptance, geometric cooling in elapsed fraction of the time (or evaluation) budget."""
    worsening = []
    for _ in range(calibration_moves):
        delta = evaluate(random_move(seq, rng)) - value
        if delta > 0:
            worsening.append(delta)
    t0 = (np.mean(worsening) if worsening else 1e-3 * value) / -math.log(initial_accept)
    current, current_value = seq, value
    while True:
        if evaluate.max_evaluations is not None and math.isinf(evaluate.time_budget):
            fraction = evaluate.evaluations / evaluate.max_evaluations
        else:
            fraction = evaluate.elapsed() / evaluate.time_budget
        temperature = t0 * end_ratio ** min(fraction, 1.0)
        candidate = random_move(current, rng)
        candidate_value = evaluate(candidate)
        delta = candidate_value - current_value
        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            current, current_value = candidate, candidate_value


def variable_neighbourhood_search(seq, value, evaluate: ExactEvaluator, rng, neighbours, *, k_max: int = 5):
    """Basic VNS: shake with k random moves, granular local search, move or k <- k + 1."""
    best, best_value = granular_local_search(seq, value, evaluate, neighbours, rng)
    k = 1
    while True:
        shaken = best
        for _ in range(k):
            shaken = random_move(shaken, rng)
        candidate, candidate_value = granular_local_search(shaken, evaluate(shaken), evaluate, neighbours, rng)
        if candidate_value < best_value - 1e-9:
            best, best_value, k = candidate, candidate_value, 1
        else:
            k = k % k_max + 1


def iterated_local_search(seq, value, evaluate: ExactEvaluator, rng, neighbours):
    """ILS: granular local search, double-bridge kick of the incumbent, better-acceptance."""
    best, best_value = granular_local_search(seq, value, evaluate, neighbours, rng)
    while True:
        kicked = double_bridge(best, rng)
        candidate, candidate_value = granular_local_search(kicked, evaluate(kicked), evaluate, neighbours, rng)
        if candidate_value <= best_value:
            best, best_value = candidate, candidate_value


METHODS = ("sa", "vns", "ils")


def run_metaheuristic(method: str, instance, solver: SolverConfig, *, env: Any = None, time_budget: float = math.inf,
                      max_evaluations: int | None = None, seed: int = 0, granular_k: int = 8,
                      objective: Callable[[list[int]], float] | None = None) -> SearchResult:
    if method not in METHODS:
        raise ValueError(f"unknown method {method}")
    if math.isinf(time_budget) and max_evaluations is None:
        raise ValueError("a time or evaluation budget is required")
    rng = np.random.default_rng(seed)
    evaluate = ExactEvaluator(instance, solver, env=env, time_budget=time_budget,
                              max_evaluations=max_evaluations, objective=objective)
    start = euclidean_tour(instance)
    initial = evaluate(start)
    neighbours = nearest_targets(instance, granular_k)
    try:
        if instance.J >= 3:
            if method == "sa":
                simulated_annealing(start, initial, evaluate, rng)
            elif method == "vns":
                variable_neighbourhood_search(start, initial, evaluate, rng, neighbours)
            else:
                iterated_local_search(start, initial, evaluate, rng, neighbours)
    except BudgetExhausted:
        pass
    return SearchResult(method=method, best_sequence=list(evaluate.best_sequence), best_makespan=evaluate.best_value,
                        initial_makespan=initial, evaluations=evaluate.evaluations, cache_hits=evaluate.cache_hits,
                        seconds=evaluate.elapsed(), trace=list(evaluate.trace))
