from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Callable, Iterable, Iterator

import numpy as np

from src.CVPSolver import SPSolution
from src.TSProblemDef import CVTSPInstance


GUROBI_OPTIMAL_STATUS = "2"


def validate_tour(sequence: Iterable[int], problem_size: int) -> tuple[int, ...]:
    tour = tuple(int(node) for node in sequence)
    if len(tour) != problem_size:
        raise ValueError(f"expected a tour of length {problem_size}, got {len(tour)}")
    if set(tour) != set(range(problem_size)):
        raise ValueError(f"tour must be a permutation of 0..{problem_size - 1}")
    return tour


def is_optimal_gurobi_solution(solution: SPSolution) -> bool:
    return solution.success and str(solution.status).strip() == GUROBI_OPTIMAL_STATUS


@dataclass(frozen=True)
class SolutionFeasibility:
    max_outbound_violation: float
    max_inbound_violation: float
    max_sync_violation: float
    max_endurance_violation: float
    max_tau_equality_violation: float
    max_carrier_violation: float
    final_return_violation: float

    @property
    def max_violation(self) -> float:
        return max(0.0, *asdict(self).values())

    def to_dict(self) -> dict[str, float]:
        payload = {key: float(value) for key, value in asdict(self).items()}
        payload["max_violation"] = float(self.max_violation)
        return payload


@dataclass(frozen=True)
class FixedSolutionUpperBound:
    """A primal, fixed-solution upper bound for every tour of one CVTSP instance.

    The Gurobi solution arrays are indexed by visit position.  This class first
    relabels all per-visit variables by target id.  For a candidate tour it
    reuses each target's takeoff/landing point and flight times, and replaces
    only the carrier travel variables with the exact travel times induced by
    the candidate adjacency.

    The resulting objective is a feasible candidate objective (up to the
    configured solver feasibility tolerance), hence an upper bound on the
    candidate tour's re-optimized fixed-tour CVP value.
    """

    instance_id: str
    anchor_sequence: tuple[int, ...]
    anchor_objective: float
    takeoff_by_target: np.ndarray
    landing_by_target: np.ndarray
    t1_by_target: np.ndarray
    t2_by_target: np.ndarray
    tau_by_target: np.ndarray
    fixed_flight_time: float
    directed_carrier_time: np.ndarray
    anchor_reused_objective: float
    feasibility: SolutionFeasibility
    feasibility_tolerance: float

    @classmethod
    def from_solution(
        cls,
        instance: CVTSPInstance,
        solution: SPSolution,
        *,
        feasibility_tolerance: float = 1e-5,
        require_optimal: bool = True,
    ) -> "FixedSolutionUpperBound":
        if not solution.success:
            raise ValueError(f"cannot build an upper bound from failed solution {solution.status}")
        if require_optimal and not is_optimal_gurobi_solution(solution):
            raise ValueError(
                "certified local search requires an optimal Gurobi anchor; "
                f"received status {solution.status}"
            )
        if not isfinite(float(solution.objective)):
            raise ValueError("anchor objective must be finite")
        if instance.carrier_speed <= 0 or instance.uav_speed <= 0:
            raise ValueError("vehicle speeds must be positive")

        tour = validate_tour(solution.sequence, instance.J)
        position_arrays = {
            "takeoff_points": solution.takeoff_points,
            "landing_points": solution.landing_points,
            "t1": solution.t1,
            "t2": solution.t2,
            "tau": solution.tau,
            "Tseg": solution.Tseg,
        }
        missing = [name for name, value in position_arrays.items() if value is None]
        if missing:
            raise ValueError(f"solution is missing arrays required by the upper bound: {missing}")

        takeoff_visit = np.asarray(solution.takeoff_points, dtype=float)
        landing_visit = np.asarray(solution.landing_points, dtype=float)
        t1_visit = np.asarray(solution.t1, dtype=float)
        t2_visit = np.asarray(solution.t2, dtype=float)
        tau_visit = np.asarray(solution.tau, dtype=float)
        tseg_visit = np.asarray(solution.Tseg, dtype=float)
        expected_shapes = {
            "takeoff_points": (instance.J, 2),
            "landing_points": (instance.J, 2),
            "t1": (instance.J,),
            "t2": (instance.J,),
            "tau": (instance.J,),
            "Tseg": (instance.J + 1,),
        }
        actual_arrays = {
            "takeoff_points": takeoff_visit,
            "landing_points": landing_visit,
            "t1": t1_visit,
            "t2": t2_visit,
            "tau": tau_visit,
            "Tseg": tseg_visit,
        }
        for name, expected in expected_shapes.items():
            if actual_arrays[name].shape != expected:
                raise ValueError(f"{name} has shape {actual_arrays[name].shape}, expected {expected}")

        takeoff = np.empty_like(takeoff_visit)
        landing = np.empty_like(landing_visit)
        t1 = np.empty_like(t1_visit)
        t2 = np.empty_like(t2_visit)
        tau = np.empty_like(tau_visit)
        for visit_position, target_id in enumerate(tour):
            takeoff[target_id] = takeoff_visit[visit_position]
            landing[target_id] = landing_visit[visit_position]
            t1[target_id] = t1_visit[visit_position]
            t2[target_id] = t2_visit[visit_position]
            tau[target_id] = tau_visit[visit_position]

        feasibility = _audit_solution_feasibility(
            instance=instance,
            tour=tour,
            takeoff_visit=takeoff_visit,
            landing_visit=landing_visit,
            t1_visit=t1_visit,
            t2_visit=t2_visit,
            tau_visit=tau_visit,
            tseg_visit=tseg_visit,
        )
        if feasibility.max_violation > feasibility_tolerance:
            raise ValueError(
                "anchor solution exceeds the permitted feasibility tolerance: "
                f"{feasibility.max_violation:.3e} > {feasibility_tolerance:.3e}"
            )

        directed = _build_directed_carrier_time(instance, takeoff, landing)
        fixed_flight_time = float(tau.sum())
        anchor_reused = _route_objective(tour, fixed_flight_time, directed)

        # At an optimal anchor the original Tseg variables and their geometric
        # lower bounds agree up to solver tolerances.  A large discrepancy means
        # that visit-position data were mapped incorrectly or the model changed.
        consistency_tolerance = max(
            10.0 * feasibility_tolerance,
            1e-8 * max(1.0, abs(float(solution.objective))),
        )
        if abs(anchor_reused - float(solution.objective)) > consistency_tolerance:
            raise ValueError(
                "reconstructed anchor objective is inconsistent with Gurobi: "
                f"reused={anchor_reused:.12g}, solver={solution.objective:.12g}, "
                f"tolerance={consistency_tolerance:.3e}"
            )

        return cls(
            instance_id=instance.instance_id,
            anchor_sequence=tour,
            anchor_objective=float(solution.objective),
            takeoff_by_target=takeoff,
            landing_by_target=landing,
            t1_by_target=t1,
            t2_by_target=t2,
            tau_by_target=tau,
            fixed_flight_time=fixed_flight_time,
            directed_carrier_time=directed,
            anchor_reused_objective=anchor_reused,
            feasibility=feasibility,
            feasibility_tolerance=float(feasibility_tolerance),
        )

    @property
    def problem_size(self) -> int:
        return len(self.anchor_sequence)

    def upper_bound(self, sequence: Iterable[int]) -> float:
        tour = validate_tour(sequence, self.problem_size)
        return _route_objective(tour, self.fixed_flight_time, self.directed_carrier_time)

    def carrier_time(self, predecessor: int | None, successor: int | None) -> float:
        """Return depot/target directed time; ``None`` denotes the depot."""
        row = 0 if predecessor is None else int(predecessor) + 1
        column = 0 if successor is None else int(successor) + 1
        return float(self.directed_carrier_time[row, column])

    def to_dict(self, include_arrays: bool = False) -> dict:
        payload = {
            "instance_id": self.instance_id,
            "anchor_sequence": list(self.anchor_sequence),
            "anchor_objective": self.anchor_objective,
            "anchor_reused_objective": self.anchor_reused_objective,
            "fixed_flight_time": self.fixed_flight_time,
            "feasibility_tolerance": self.feasibility_tolerance,
            "feasibility": self.feasibility.to_dict(),
        }
        if include_arrays:
            payload.update(
                {
                    "takeoff_by_target": self.takeoff_by_target.tolist(),
                    "landing_by_target": self.landing_by_target.tolist(),
                    "t1_by_target": self.t1_by_target.tolist(),
                    "t2_by_target": self.t2_by_target.tolist(),
                    "tau_by_target": self.tau_by_target.tolist(),
                    "directed_carrier_time": self.directed_carrier_time.tolist(),
                }
            )
        return payload


@dataclass(frozen=True)
class TourMove:
    kind: str
    first: int
    second: int
    sequence: tuple[int, ...]
    upper_bound: float
    delta: float

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "first": self.first,
            "second": self.second,
            "sequence": list(self.sequence),
            "upper_bound": self.upper_bound,
            "delta": self.delta,
        }


class _FixedUpperBoundMoveScorer:
    """O(1) exact fixed-solution UB deltas for swap/relocate/asymmetric 2-opt."""

    def __init__(
        self,
        majorizer: FixedSolutionUpperBound,
        sequence: Iterable[int] | None = None,
    ):
        self.majorizer = majorizer
        self.tour = (
            majorizer.anchor_sequence
            if sequence is None
            else validate_tour(sequence, majorizer.problem_size)
        )
        self.matrix = majorizer.directed_carrier_time
        # The majorizer remains fixed, but a local-search chain changes the
        # current permutation after every accepted move.  Deltas must therefore
        # be based on the current tour, not always on the Gurobi anchor tour.
        self.base_upper_bound = majorizer.upper_bound(self.tour)
        differences = [
            self._edge(self.tour[index + 1], self.tour[index])
            - self._edge(self.tour[index], self.tour[index + 1])
            for index in range(len(self.tour) - 1)
        ]
        self.reverse_prefix = np.concatenate((np.zeros(1, dtype=float), np.cumsum(differences)))

    def _edge(self, predecessor: int | None, successor: int | None) -> float:
        row = 0 if predecessor is None else predecessor + 1
        column = 0 if successor is None else successor + 1
        return float(self.matrix[row, column])

    def upper_bound(
        self,
        kind: str,
        first: int,
        second: int,
        candidate: tuple[int, ...] | None = None,
    ) -> float:
        if kind == "swap":
            delta = self._swap_delta(first, second)
        elif kind == "relocate":
            delta = self._relocate_delta(first, second)
        elif kind == "two_opt":
            delta = self._two_opt_delta(first, second)
        else:
            if candidate is None:
                raise ValueError(f"unsupported move type {kind!r} requires a candidate tour")
            return self.majorizer.upper_bound(candidate)
        return self.base_upper_bound + delta

    def _swap_delta(self, first: int, second: int) -> float:
        if first > second:
            first, second = second, first
        left, right = self.tour[first], self.tour[second]
        before_left = None if first == 0 else self.tour[first - 1]
        after_right = None if second == len(self.tour) - 1 else self.tour[second + 1]
        if second == first + 1:
            old = self._edge(before_left, left) + self._edge(left, right) + self._edge(right, after_right)
            new = self._edge(before_left, right) + self._edge(right, left) + self._edge(left, after_right)
            return new - old
        after_left, before_right = self.tour[first + 1], self.tour[second - 1]
        old = (
            self._edge(before_left, left)
            + self._edge(left, after_left)
            + self._edge(before_right, right)
            + self._edge(right, after_right)
        )
        new = (
            self._edge(before_left, right)
            + self._edge(right, after_left)
            + self._edge(before_right, left)
            + self._edge(left, after_right)
        )
        return new - old

    def _relocate_delta(self, source: int, destination: int) -> float:
        node = self.tour[source]
        before = None if source == 0 else self.tour[source - 1]
        after = None if source == len(self.tour) - 1 else self.tour[source + 1]
        removal = self._edge(before, after) - self._edge(before, node) - self._edge(node, after)
        shortened = self.tour[:source] + self.tour[source + 1 :]
        insertion_before = None if destination == 0 else shortened[destination - 1]
        insertion_after = None if destination == len(shortened) else shortened[destination]
        insertion = (
            self._edge(insertion_before, node)
            + self._edge(node, insertion_after)
            - self._edge(insertion_before, insertion_after)
        )
        return removal + insertion

    def _two_opt_delta(self, first: int, second: int) -> float:
        left, right = self.tour[first], self.tour[second]
        before = None if first == 0 else self.tour[first - 1]
        after = None if second == len(self.tour) - 1 else self.tour[second + 1]
        boundary = (
            self._edge(before, right)
            + self._edge(left, after)
            - self._edge(before, left)
            - self._edge(right, after)
        )
        internal = float(self.reverse_prefix[second] - self.reverse_prefix[first])
        return boundary + internal


def apply_swap(sequence: Iterable[int], first: int, second: int) -> tuple[int, ...]:
    tour = list(sequence)
    if not (0 <= first < len(tour) and 0 <= second < len(tour)):
        raise IndexError("swap positions are outside the tour")
    tour[first], tour[second] = tour[second], tour[first]
    return tuple(tour)


def apply_relocate(sequence: Iterable[int], source: int, destination: int) -> tuple[int, ...]:
    """Move one node to ``destination`` in the final sequence."""
    tour = list(sequence)
    if not (0 <= source < len(tour) and 0 <= destination < len(tour)):
        raise IndexError("relocate positions are outside the tour")
    node = tour.pop(source)
    tour.insert(destination, node)
    return tuple(tour)


def apply_two_opt(sequence: Iterable[int], first: int, second: int) -> tuple[int, ...]:
    """Reverse the inclusive interval [first, second] for asymmetric costs."""
    tour = list(sequence)
    if not (0 <= first < second < len(tour)):
        raise IndexError("2-opt requires 0 <= first < second < tour length")
    tour[first : second + 1] = reversed(tour[first : second + 1])
    return tuple(tour)


def iter_candidate_sequences(
    sequence: Iterable[int],
    move_types: Iterable[str] = ("swap", "relocate", "two_opt"),
) -> Iterator[tuple[str, int, int, tuple[int, ...]]]:
    tour = tuple(int(node) for node in sequence)
    enabled = tuple(str(kind).strip().lower() for kind in move_types)
    unknown = set(enabled) - {"swap", "relocate", "two_opt", "2opt"}
    if unknown:
        raise ValueError(f"unsupported move types: {sorted(unknown)}")

    seen: set[tuple[int, ...]] = set()
    n = len(tour)
    if "swap" in enabled:
        for first in range(n - 1):
            for second in range(first + 1, n):
                candidate = apply_swap(tour, first, second)
                if candidate not in seen:
                    seen.add(candidate)
                    yield "swap", first, second, candidate

    if "relocate" in enabled:
        for source in range(n):
            for destination in range(n):
                if source == destination:
                    continue
                candidate = apply_relocate(tour, source, destination)
                if candidate not in seen:
                    seen.add(candidate)
                    yield "relocate", source, destination, candidate

    if "two_opt" in enabled or "2opt" in enabled:
        for first in range(n - 1):
            for second in range(first + 1, n):
                candidate = apply_two_opt(tour, first, second)
                if candidate not in seen:
                    seen.add(candidate)
                    yield "two_opt", first, second, candidate


def iter_candidate_move_indices(
    sequence: Iterable[int],
    move_types: Iterable[str] = ("swap", "relocate", "two_opt"),
) -> Iterator[tuple[str, int, int]]:
    """Enumerate move indices without copying a length-n candidate each time."""

    tour = tuple(int(node) for node in sequence)
    enabled = tuple(str(kind).strip().lower() for kind in move_types)
    unknown = set(enabled) - {"swap", "relocate", "two_opt", "2opt"}
    if unknown:
        raise ValueError(f"unsupported move types: {sorted(unknown)}")
    n = len(tour)
    swap_enabled = "swap" in enabled
    if swap_enabled:
        for first in range(n - 1):
            for second in range(first + 1, n):
                yield "swap", first, second
    if "relocate" in enabled:
        for source in range(n):
            for destination in range(n):
                if source != destination:
                    yield "relocate", source, destination
    if "two_opt" in enabled or "2opt" in enabled:
        for first in range(n - 1):
            for second in range(first + 1, n):
                # Reversing two adjacent nodes is identical to their swap.
                if swap_enabled and second == first + 1:
                    continue
                yield "two_opt", first, second


def _materialize_move(
    tour: tuple[int, ...], kind: str, first: int, second: int
) -> tuple[int, ...]:
    if kind == "swap":
        return apply_swap(tour, first, second)
    if kind == "relocate":
        return apply_relocate(tour, first, second)
    if kind == "two_opt":
        return apply_two_opt(tour, first, second)
    raise ValueError(f"unsupported move type {kind!r}")


def find_best_certified_move(
    majorizer: FixedSolutionUpperBound,
    current_objective: float,
    *,
    epsilon: float,
    move_types: Iterable[str] = ("swap", "relocate", "two_opt"),
) -> tuple[TourMove | None, int]:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    best: TourMove | None = None
    evaluated = 0
    base_upper_bound = majorizer.upper_bound(majorizer.anchor_sequence)
    move_scorer = _FixedUpperBoundMoveScorer(majorizer)
    for kind, first, second in iter_candidate_move_indices(
        majorizer.anchor_sequence,
        move_types=move_types,
    ):
        evaluated += 1
        candidate_upper_bound = move_scorer.upper_bound(kind, first, second)
        if candidate_upper_bound >= current_objective - epsilon:
            continue
        candidate = _materialize_move(majorizer.anchor_sequence, kind, first, second)
        move = TourMove(
            kind=kind,
            first=first,
            second=second,
            sequence=candidate,
            upper_bound=candidate_upper_bound,
            delta=candidate_upper_bound - base_upper_bound,
        )
        if best is None or move.upper_bound < best.upper_bound:
            best = move
    return best, evaluated


def find_best_fixed_anchor_move(
    majorizer: FixedSolutionUpperBound,
    current_sequence: Iterable[int],
    *,
    epsilon: float,
    move_types: Iterable[str] = ("swap", "two_opt"),
    ranker: Callable[[tuple[int, ...]], float] | None = None,
    max_ranked_candidates: int | None = None,
    candidate_lower_bound: Callable[[tuple[int, ...]], float] | None = None,
) -> tuple[TourMove | None, int, int, int]:
    """Find the best UB-improving move under one unchanged anchor solution.

    Unlike :func:`find_best_certified_move`, this function can start from any
    permutation and compares candidates with that permutation's fixed-anchor
    upper bound.  It performs no optimization-solver call.  Repeated calls with
    the returned sequence therefore implement a complete local-search chain on
    one reusable Gurobi majorizer.

    ``ranker`` is only a screening heuristic.  Acceptance always uses the exact
    fixed-anchor upper bound.  When ``max_ranked_candidates`` is supplied, only
    that many predictor-ranked moves are evaluated with the upper-bound table.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if max_ranked_candidates is not None:
        if ranker is None:
            raise ValueError("max_ranked_candidates requires a ranker")
        if max_ranked_candidates <= 0:
            raise ValueError("max_ranked_candidates must be positive")

    tour = validate_tour(current_sequence, majorizer.problem_size)
    scorer = _FixedUpperBoundMoveScorer(majorizer, tour)
    indices = list(iter_candidate_move_indices(tour, move_types=move_types))
    generated = len(indices)

    if ranker is not None:
        prepare = getattr(ranker, "prepare", None)
        if prepare is not None:
            prepare(tour)
        score_move = getattr(ranker, "score_move", None)
        ranked: list[tuple[float, str, int, int]] = []
        for kind, first, second in indices:
            if score_move is None:
                candidate = _materialize_move(tour, kind, first, second)
                ranking_score = float(ranker(candidate))
            else:
                ranking_score = float(score_move(kind, first, second, ()))
            if not isfinite(ranking_score):
                raise ValueError("candidate ranker returned a non-finite score")
            ranked.append((ranking_score, kind, first, second))
        ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        if max_ranked_candidates is not None:
            ranked = ranked[:max_ranked_candidates]
        selected = [(kind, first, second) for _, kind, first, second in ranked]
    else:
        selected = indices

    best: TourMove | None = None
    evaluated = 0
    lower_bound_pruned = 0
    for kind, first, second in selected:
        candidate: tuple[int, ...] | None = None
        if candidate_lower_bound is not None:
            candidate = _materialize_move(tour, kind, first, second)
            if float(candidate_lower_bound(candidate)) >= scorer.base_upper_bound - epsilon:
                lower_bound_pruned += 1
                continue
        evaluated += 1
        candidate_upper_bound = scorer.upper_bound(kind, first, second)
        if candidate_upper_bound >= scorer.base_upper_bound - epsilon:
            continue
        if candidate is None:
            candidate = _materialize_move(tour, kind, first, second)
        move = TourMove(
            kind=kind,
            first=first,
            second=second,
            sequence=candidate,
            upper_bound=float(candidate_upper_bound),
            delta=float(candidate_upper_bound - scorer.base_upper_bound),
        )
        if best is None or move.upper_bound < best.upper_bound:
            best = move
    return best, generated, evaluated, lower_bound_pruned


def find_best_certified_move_ranked(
    majorizer: FixedSolutionUpperBound,
    current_objective: float,
    *,
    epsilon: float,
    ranker: Callable[[tuple[int, ...]], float],
    max_exact_candidates: int,
    move_types: Iterable[str] = ("swap", "relocate", "two_opt"),
) -> tuple[TourMove | None, int, int]:
    """Rank the whole neighborhood, then certify only the best-ranked subset.

    The ranker is never used for acceptance.  Every returned move has passed
    the same exact fixed-solution upper-bound test as
    :func:`find_best_certified_move`.  A poor ranker can miss an improving move,
    but it cannot create a false accepted move.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if max_exact_candidates <= 0:
        raise ValueError("max_exact_candidates must be positive")

    prepare = getattr(ranker, "prepare", None)
    if prepare is not None:
        prepare(majorizer.anchor_sequence)
    score_move = getattr(ranker, "score_move", None)

    ranked_candidates = []
    for kind, first, second in iter_candidate_move_indices(
        majorizer.anchor_sequence,
        move_types=move_types,
    ):
        if score_move is None:
            candidate = _materialize_move(majorizer.anchor_sequence, kind, first, second)
            ranking_score = float(ranker(candidate))
        else:
            ranking_score = float(score_move(kind, first, second, ()))
        if not isfinite(ranking_score):
            raise ValueError("candidate ranker returned a non-finite score")
        ranked_candidates.append((ranking_score, kind, first, second))
    ranked_candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

    best: TourMove | None = None
    base_upper_bound = majorizer.upper_bound(majorizer.anchor_sequence)
    move_scorer = _FixedUpperBoundMoveScorer(majorizer)
    selected = ranked_candidates[:max_exact_candidates]
    for _, kind, first, second in selected:
        candidate_upper_bound = move_scorer.upper_bound(kind, first, second)
        if candidate_upper_bound >= current_objective - epsilon:
            continue
        candidate = _materialize_move(majorizer.anchor_sequence, kind, first, second)
        move = TourMove(
            kind=kind,
            first=first,
            second=second,
            sequence=candidate,
            upper_bound=candidate_upper_bound,
            delta=candidate_upper_bound - base_upper_bound,
        )
        if best is None or move.upper_bound < best.upper_bound:
            best = move
    return best, len(ranked_candidates), len(selected)


def find_best_certified_move_dual_pruned(
    majorizer: FixedSolutionUpperBound,
    dual_lower_bound,
    current_objective: float,
    *,
    epsilon: float,
    move_types: Iterable[str] = ("swap", "relocate", "two_opt"),
) -> tuple[TourMove | None, int, int, int]:
    """Use a reusable dual LB to reject impossible improvements before UB work."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    best: TourMove | None = None
    generated = 0
    dual_pruned = 0
    exact_ub_evaluated = 0
    base_upper_bound = majorizer.upper_bound(majorizer.anchor_sequence)
    move_scorer = _FixedUpperBoundMoveScorer(majorizer)
    for kind, first, second in iter_candidate_move_indices(
        majorizer.anchor_sequence,
        move_types=move_types,
    ):
        generated += 1
        if dual_lower_bound.move_lower_bound_indices(kind, first, second) >= current_objective - epsilon:
            dual_pruned += 1
            continue
        exact_ub_evaluated += 1
        candidate_upper_bound = move_scorer.upper_bound(kind, first, second)
        if candidate_upper_bound >= current_objective - epsilon:
            continue
        candidate = _materialize_move(majorizer.anchor_sequence, kind, first, second)
        move = TourMove(
            kind=kind,
            first=first,
            second=second,
            sequence=candidate,
            upper_bound=candidate_upper_bound,
            delta=candidate_upper_bound - base_upper_bound,
        )
        if best is None or move.upper_bound < best.upper_bound:
            best = move
    return best, generated, dual_pruned, exact_ub_evaluated


def find_best_certified_move_ranked_dual_pruned(
    majorizer: FixedSolutionUpperBound,
    dual_lower_bound,
    current_objective: float,
    *,
    epsilon: float,
    ranker: Callable[[tuple[int, ...]], float],
    max_exact_candidates: int,
    move_types: Iterable[str] = ("swap", "relocate", "two_opt"),
) -> tuple[TourMove | None, int, int, int]:
    """Dual-prune, predictor-rank survivors, then certify Top-K with the primal UB."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if max_exact_candidates <= 0:
        raise ValueError("max_exact_candidates must be positive")
    prepare = getattr(ranker, "prepare", None)
    if prepare is not None:
        prepare(majorizer.anchor_sequence)
    score_move = getattr(ranker, "score_move", None)

    ranked_candidates = []
    generated = 0
    dual_pruned = 0
    for kind, first, second in iter_candidate_move_indices(
        majorizer.anchor_sequence,
        move_types=move_types,
    ):
        generated += 1
        if dual_lower_bound.move_lower_bound_indices(kind, first, second) >= current_objective - epsilon:
            dual_pruned += 1
            continue
        ranking_score = (
            float(ranker(_materialize_move(majorizer.anchor_sequence, kind, first, second)))
            if score_move is None
            else float(score_move(kind, first, second, ()))
        )
        if not isfinite(ranking_score):
            raise ValueError("candidate ranker returned a non-finite score")
        ranked_candidates.append((ranking_score, kind, first, second))
    ranked_candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

    best: TourMove | None = None
    base_upper_bound = majorizer.upper_bound(majorizer.anchor_sequence)
    move_scorer = _FixedUpperBoundMoveScorer(majorizer)
    selected = ranked_candidates[:max_exact_candidates]
    for _, kind, first, second in selected:
        candidate_upper_bound = move_scorer.upper_bound(kind, first, second)
        if candidate_upper_bound >= current_objective - epsilon:
            continue
        candidate = _materialize_move(majorizer.anchor_sequence, kind, first, second)
        move = TourMove(
            kind=kind,
            first=first,
            second=second,
            sequence=candidate,
            upper_bound=candidate_upper_bound,
            delta=candidate_upper_bound - base_upper_bound,
        )
        if best is None or move.upper_bound < best.upper_bound:
            best = move
    return best, generated, dual_pruned, len(selected)


def _build_directed_carrier_time(
    instance: CVTSPInstance,
    takeoff_by_target: np.ndarray,
    landing_by_target: np.ndarray,
) -> np.ndarray:
    n = instance.J
    matrix = np.zeros((n + 1, n + 1), dtype=float)
    matrix[0, 1:] = np.linalg.norm(takeoff_by_target - instance.depot[None, :], axis=1)
    matrix[1:, 0] = np.linalg.norm(landing_by_target - instance.depot[None, :], axis=1)
    matrix[1:, 1:] = np.linalg.norm(
        landing_by_target[:, None, :] - takeoff_by_target[None, :, :],
        axis=2,
    )
    matrix /= float(instance.carrier_speed)
    return matrix


def _route_objective(
    tour: tuple[int, ...],
    fixed_flight_time: float,
    directed_carrier_time: np.ndarray,
) -> float:
    node_ids = np.asarray(tour, dtype=int) + 1
    carrier = float(directed_carrier_time[0, node_ids[0]])
    if len(node_ids) > 1:
        carrier += float(directed_carrier_time[node_ids[:-1], node_ids[1:]].sum())
    carrier += float(directed_carrier_time[node_ids[-1], 0])
    return float(fixed_flight_time + carrier)


def _positive_max(values: np.ndarray | list[float]) -> float:
    array = np.asarray(values, dtype=float)
    return max(0.0, float(array.max(initial=0.0)))


def _audit_solution_feasibility(
    *,
    instance: CVTSPInstance,
    tour: tuple[int, ...],
    takeoff_visit: np.ndarray,
    landing_visit: np.ndarray,
    t1_visit: np.ndarray,
    t2_visit: np.ndarray,
    tau_visit: np.ndarray,
    tseg_visit: np.ndarray,
) -> SolutionFeasibility:
    ordered_targets = instance.targets[np.asarray(tour, dtype=int)]
    outbound = np.linalg.norm(takeoff_visit - ordered_targets, axis=1) - instance.uav_speed * t1_visit
    inbound = np.linalg.norm(landing_visit - ordered_targets, axis=1) - instance.uav_speed * t2_visit
    sync = np.linalg.norm(landing_visit - takeoff_visit, axis=1) - instance.carrier_speed * tau_visit
    endurance = tau_visit - instance.endurance
    tau_equality = np.abs(tau_visit - t1_visit - t2_visit)

    carrier_residuals = []
    for visit_position in range(instance.J):
        previous = instance.depot if visit_position == 0 else landing_visit[visit_position - 1]
        carrier_residuals.append(
            np.linalg.norm(takeoff_visit[visit_position] - previous)
            - instance.carrier_speed * tseg_visit[visit_position]
        )
    final_return = (
        np.linalg.norm(instance.depot - landing_visit[-1])
        - instance.carrier_speed * tseg_visit[-1]
    )
    return SolutionFeasibility(
        max_outbound_violation=_positive_max(outbound),
        max_inbound_violation=_positive_max(inbound),
        max_sync_violation=_positive_max(sync),
        max_endurance_violation=_positive_max(endurance),
        max_tau_equality_violation=_positive_max(tau_equality),
        max_carrier_violation=_positive_max(carrier_residuals),
        final_return_violation=max(0.0, float(final_return)),
    )
