from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from src import CVPSolver
from src.CertifiedLocalSearch import (
    FixedAnchorLocalSearchConfig,
    run_fixed_anchor_local_search,
)
from src.CertifiedUpperBound import FixedSolutionUpperBound
from src.DualContributions import (
    directed_total_contribution_matrix,
    ordered_step_costs,
)
from src.TSProblemDef import load_instance


def _directed_route_step_costs(sequence: list[int], matrix: np.ndarray) -> np.ndarray:
    """Read depot-tour-depot edge costs from a directed contribution table."""

    route = np.asarray([0, *(node + 1 for node in sequence), 0], dtype=int)
    return np.asarray(matrix[route[:-1], route[1:]], dtype=float)


class SparseJointPipelineGurobiSmokeTest(unittest.TestCase):
    """Minimal real-solver check for the one-anchor/two-solve data flow."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.instance = load_instance("instance/Data/Example_1.txt")
        cls.anchor = list(range(cls.instance.J))
        cls.solver_config = CVPSolver.SolverConfig(
            solver_backend="gurobi",
            gurobi_threads=1,
            gurobi_explicit_qcp=True,
            output_flag=0,
            gurobi_params={
                "FeasibilityTol": 1e-8,
                "OptimalityTol": 1e-8,
                "NumericFocus": 2,
            },
        )

        # The complete instance flow has exactly two explicit exact solves:
        # one anchor solve and, because the route changes, one final solve.
        original_solve = CVPSolver.solve
        with patch("src.CVPSolver.solve", wraps=original_solve) as counted_solve:
            cls.anchor_solution = CVPSolver.solve(
                cls.instance,
                cls.anchor,
                cls.solver_config,
            )
            if not cls.anchor_solution.success:
                raise unittest.SkipTest(
                    "Gurobi is unavailable: "
                    f"{cls.anchor_solution.status} {cls.anchor_solution.raw_debug}"
                )

            cls.majorizer = FixedSolutionUpperBound.from_solution(
                cls.instance,
                cls.anchor_solution,
                feasibility_tolerance=2e-5,
            )
            search_config = FixedAnchorLocalSearchConfig(
                max_local_moves=6,
                absolute_epsilon=1e-6,
                relative_epsilon=1e-8,
                move_types=("swap", "two_opt"),
            )

            # This imported binding is the old solver entry point in the local
            # search module.  Any exact refresh inside the move chain is a bug.
            with patch(
                "src.CertifiedLocalSearch.solve",
                side_effect=AssertionError("fixed-anchor local search invoked Gurobi"),
            ):
                cls.local_result = run_fixed_anchor_local_search(
                    cls.majorizer,
                    cls.anchor,
                    search_config,
                )

            cls.final_solution = CVPSolver.solve(
                cls.instance,
                cls.local_result.final_sequence,
                cls.solver_config,
            )
            cls.exact_call_count = counted_solve.call_count

        if not cls.final_solution.success:
            raise AssertionError(
                "final expert Gurobi solve failed: "
                f"{cls.final_solution.status} {cls.final_solution.raw_debug}"
            )

    def test_changed_expert_uses_two_solves_and_improves_real_makespan(self) -> None:
        result = self.local_result
        anchor_value = float(self.anchor_solution.objective)
        final_upper_bound = float(result.final_upper_bound)
        final_value = float(self.final_solution.objective)

        self.assertEqual(self.exact_call_count, 2)
        self.assertNotEqual(result.final_sequence, self.anchor)
        self.assertGreater(result.accepted, 0)
        self.assertLessEqual(result.accepted, 6)
        self.assertTrue(
            all(item.move.kind in {"swap", "two_opt"} for item in result.history)
        )

        # This is the central certificate and the requested real improvement:
        # V(expert) <= U_anchor(expert) < V(anchor).
        self.assertLessEqual(final_value, final_upper_bound)
        self.assertLess(final_upper_bound, anchor_value)
        self.assertLess(final_value, anchor_value)
        self.assertGreater(anchor_value - final_value, 1e-3)

    def test_directed_table_route_steps_reconstruct_upper_bounds(self) -> None:
        matrix = directed_total_contribution_matrix(self.majorizer)
        self.assertEqual(
            matrix.shape,
            (self.instance.J + 1, self.instance.J + 1),
        )
        self.assertFalse(np.allclose(matrix, matrix.T))

        for sequence in (self.anchor, self.local_result.final_sequence):
            step_costs = _directed_route_step_costs(sequence, matrix)
            self.assertEqual(step_costs.shape, (self.instance.J + 1,))
            self.assertTrue(np.isfinite(step_costs).all())
            self.assertAlmostEqual(
                float(step_costs.sum()),
                self.majorizer.upper_bound(sequence),
                places=8,
            )

        exact_anchor_steps = ordered_step_costs(self.anchor_solution)
        self.assertAlmostEqual(
            float(exact_anchor_steps.sum()),
            float(self.anchor_solution.objective),
            places=4,
        )


if __name__ == "__main__":
    unittest.main()
