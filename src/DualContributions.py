from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Iterable

import numpy as np

from src.CertifiedUpperBound import FixedSolutionUpperBound, is_optimal_gurobi_solution, validate_tour
from src.CVPSolver import SPSolution
from src.TSProblemDef import CVTSPInstance


@dataclass(frozen=True)
class FixedDualLowerBound:
    """A reusable dual lower bound for every permutation of one instance.

    The fixed-tour QCP is indexed by visit position.  Changing a tour only
    changes the target coordinates on the right-hand side of ``out_*_def``
    and ``in_*_def``; the objective, variables, and constraint matrix remain
    unchanged.  Consequently an optimal dual solution for an anchor tour is
    dual-feasible for every other permutation.

    ``assignment_contribution[k, j]`` is the route-dependent part of the dual
    objective when target ``j`` is assigned to visit position ``k``.  The
    remaining route-independent part is stored in ``constant``.  This gives a
    rigorous lower bound and an O(1) delta for swapping two positions.
    """

    instance_id: str
    anchor_sequence: tuple[int, ...]
    anchor_objective: float
    constant: float
    assignment_contribution: np.ndarray
    position_coordinate_gradient: np.ndarray
    anchor_reconstructed_objective: float
    max_anchor_error: float

    @classmethod
    def from_solution(
        cls,
        instance: CVTSPInstance,
        solution: SPSolution,
        *,
        reconstruction_tolerance: float = 5e-4,
    ) -> "FixedDualLowerBound":
        if not is_optimal_gurobi_solution(solution):
            raise ValueError(f"dual lower bound requires an optimal Gurobi solution, got {solution.status}")
        if not solution.linear_duals or not solution.quadratic_duals:
            raise ValueError("solution has no Pi/QCPi values; solve with explicit_qcp and collect_duals")
        tour = validate_tour(solution.sequence, instance.J)

        gradients = np.zeros((instance.J, 2), dtype=float)
        for position in range(instance.J):
            required = (
                f"out_x_def[{position}]",
                f"out_y_def[{position}]",
                f"in_x_def[{position}]",
                f"in_y_def[{position}]",
            )
            missing = [name for name in required if name not in solution.linear_duals]
            if missing:
                raise ValueError(f"missing target-coordinate duals at position {position}: {missing}")

            # Gurobi stores these equalities as, e.g., out_x - sx = -p_x.
            # Pi is the derivative with respect to that RHS, hence the
            # route-dependent dual term is -(Pi_out + Pi_in) * p.
            gradients[position, 0] = -(
                float(solution.linear_duals[required[0]]["pi"])
                + float(solution.linear_duals[required[2]]["pi"])
            )
            gradients[position, 1] = -(
                float(solution.linear_duals[required[1]]["pi"])
                + float(solution.linear_duals[required[3]]["pi"])
            )

        assignments = gradients @ np.asarray(instance.targets, dtype=float).T
        anchor_assignment = float(sum(assignments[k, target] for k, target in enumerate(tour)))
        constant = float(solution.objective) - anchor_assignment
        reconstructed = constant + anchor_assignment
        error = abs(reconstructed - float(solution.objective))
        if not np.isfinite(assignments).all() or not isfinite(constant):
            raise ValueError("dual lower-bound contributions contain non-finite values")
        if error > reconstruction_tolerance:
            raise ValueError(
                "dual anchor reconstruction failed: "
                f"error={error:.3e}, tolerance={reconstruction_tolerance:.3e}"
            )
        return cls(
            instance_id=instance.instance_id,
            anchor_sequence=tour,
            anchor_objective=float(solution.objective),
            constant=constant,
            assignment_contribution=assignments,
            position_coordinate_gradient=gradients,
            anchor_reconstructed_objective=reconstructed,
            max_anchor_error=error,
        )

    @property
    def problem_size(self) -> int:
        return len(self.anchor_sequence)

    def lower_bound(self, sequence: Iterable[int]) -> float:
        tour = validate_tour(sequence, self.problem_size)
        return self.constant + float(
            sum(self.assignment_contribution[position, target] for position, target in enumerate(tour))
        )

    def swap_delta(self, first: int, second: int, sequence: Iterable[int] | None = None) -> float:
        tour = self.anchor_sequence if sequence is None else validate_tour(sequence, self.problem_size)
        if not (0 <= first < self.problem_size and 0 <= second < self.problem_size):
            raise IndexError("swap positions are outside the tour")
        left, right = tour[first], tour[second]
        old = self.assignment_contribution[first, left] + self.assignment_contribution[second, right]
        new = self.assignment_contribution[first, right] + self.assignment_contribution[second, left]
        return float(new - old)

    def move_lower_bound(
        self,
        kind: str,
        first: int,
        second: int,
        candidate: Iterable[int],
    ) -> float:
        """Incremental dual bound; a two-position exchange is O(1)."""

        normalized = str(kind).strip().lower()
        if normalized == "swap":
            return self.anchor_objective + self.swap_delta(first, second)
        candidate_tour = validate_tour(candidate, self.problem_size)
        anchor = self.anchor_sequence
        if normalized == "two_opt":
            changed = range(first, second + 1)
        elif normalized == "relocate":
            changed = range(min(first, second), max(first, second) + 1)
        else:
            return self.lower_bound(candidate_tour)
        delta = 0.0
        for position in changed:
            delta += (
                self.assignment_contribution[position, candidate_tour[position]]
                - self.assignment_contribution[position, anchor[position]]
            )
        return self.anchor_objective + float(delta)

    def move_lower_bound_indices(self, kind: str, first: int, second: int) -> float:
        """Index-only variant that avoids materializing every candidate tour."""

        normalized = str(kind).strip().lower()
        if normalized == "swap":
            return self.anchor_objective + self.swap_delta(first, second)
        delta = 0.0
        if normalized == "two_opt":
            for position in range(first, second + 1):
                new_target = self.anchor_sequence[first + second - position]
                old_target = self.anchor_sequence[position]
                delta += (
                    self.assignment_contribution[position, new_target]
                    - self.assignment_contribution[position, old_target]
                )
            return self.anchor_objective + float(delta)
        if normalized == "relocate":
            source_target = self.anchor_sequence[first]
            if first < second:
                for position in range(first, second):
                    new_target = self.anchor_sequence[position + 1]
                    old_target = self.anchor_sequence[position]
                    delta += (
                        self.assignment_contribution[position, new_target]
                        - self.assignment_contribution[position, old_target]
                    )
            elif first > second:
                for position in range(second + 1, first + 1):
                    new_target = self.anchor_sequence[position - 1]
                    old_target = self.anchor_sequence[position]
                    delta += (
                        self.assignment_contribution[position, new_target]
                        - self.assignment_contribution[position, old_target]
                    )
            delta += (
                self.assignment_contribution[second, source_target]
                - self.assignment_contribution[second, self.anchor_sequence[second]]
            )
            return self.anchor_objective + float(delta)
        raise ValueError(f"unsupported move type {kind!r}")

    def target_pressure_by_position(self) -> np.ndarray:
        """Magnitude of target-coordinate shadow prices at each visit position."""
        return np.linalg.norm(self.position_coordinate_gradient, axis=1)

    def to_dict(self, include_arrays: bool = False) -> dict:
        payload = {
            "instance_id": self.instance_id,
            "anchor_sequence": list(self.anchor_sequence),
            "anchor_objective": self.anchor_objective,
            "constant": self.constant,
            "anchor_reconstructed_objective": self.anchor_reconstructed_objective,
            "max_anchor_error": self.max_anchor_error,
            "target_pressure_by_position": self.target_pressure_by_position().tolist(),
        }
        if include_arrays:
            payload.update(
                {
                    "assignment_contribution": self.assignment_contribution.tolist(),
                    "position_coordinate_gradient": self.position_coordinate_gradient.tolist(),
                }
            )
        return payload


def directed_total_contribution_matrix(majorizer: FixedSolutionUpperBound) -> np.ndarray:
    """Decompose the reused feasible objective into directed edge contributions.

    For target-target edge i->j we add half of tau_i and half of tau_j to the
    carrier time.  Depot has zero flight contribution.  Every Hamiltonian
    depot cycle therefore counts every target's tau exactly once, so summing
    this matrix along any tour equals ``majorizer.upper_bound(tour)``.
    """

    node_tau = np.concatenate((np.zeros(1, dtype=float), np.asarray(majorizer.tau_by_target, dtype=float)))
    matrix = np.asarray(majorizer.directed_carrier_time, dtype=float).copy()
    matrix += 0.5 * node_tau[:, None]
    matrix += 0.5 * node_tau[None, :]
    np.fill_diagonal(matrix, 0.0)
    return matrix


def ordered_step_costs(solution: SPSolution) -> np.ndarray:
    """Return exact dense per-decision costs whose sum is the Gurobi objective."""

    if solution.tau is None or solution.Tseg is None:
        raise ValueError("solution is missing tau/Tseg arrays")
    tau = np.asarray(solution.tau, dtype=float)
    segments = np.asarray(solution.Tseg, dtype=float)
    if segments.shape != (tau.size + 1,):
        raise ValueError(f"Tseg shape {segments.shape} does not match tau length {tau.size}")
    contributions = tau + segments[:-1]
    contributions[-1] += segments[-1]
    reconstruction_error = abs(float(contributions.sum()) - float(solution.objective))
    tolerance = max(5e-4, 1e-7 * max(1.0, abs(float(solution.objective))))
    if reconstruction_error > tolerance:
        raise ValueError(
            "dense step-cost reconstruction failed: "
            f"sum={contributions.sum():.12g}, objective={solution.objective:.12g}, "
            f"error={reconstruction_error:.3e}"
        )
    return contributions
