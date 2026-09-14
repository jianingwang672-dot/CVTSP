from __future__ import annotations

import pickle
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.CertifiedUpperBound import FixedSolutionUpperBound
from src.CVPSolver import SPSolution, SolverConfig
from src.DualContributions import directed_total_contribution_matrix
from src.JointSelfImprovement import (
    NearMissValidationTask,
    SparseSelfImprovementResult,
    _best_non_anchor_local_candidate,
    apply_near_miss_validation,
    run_near_miss_validation_task,
)
from src.TSProblemDef import CVTSPInstance


class NearMissCoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.instance = CVTSPInstance(
            instance_id="near-miss-unit",
            path=Path("<near-miss-unit>"),
            J=3,
            depot=np.array([0.0, 0.0]),
            targets=np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 3.0]]),
            carrier_speed=1.0,
            uav_speed=5.0,
            endurance=10.0,
            scale=3.0,
        )
        self.worse_route = [0, 2, 1]
        self.better_route = [0, 1, 2]
        self.worse_solution = self._solution(self.worse_route, solve_time=0.4)
        self.better_solution = self._solution(self.better_route, solve_time=0.25)
        self.assertLess(self.better_solution.objective, self.worse_solution.objective)

    def _solution(self, sequence: list[int], *, solve_time: float) -> SPSolution:
        points = self.instance.targets[sequence]
        route_points = np.vstack([self.instance.depot, points, self.instance.depot])
        tseg = np.linalg.norm(np.diff(route_points, axis=0), axis=1)
        objective = float(tseg.sum())
        return SPSolution(
            objective=objective,
            makespan=objective,
            success=True,
            status="2",
            solve_time=solve_time,
            sequence=list(sequence),
            takeoff_points=points.copy(),
            landing_points=points.copy(),
            t1=np.zeros(self.instance.J),
            t2=np.zeros(self.instance.J),
            tau=np.zeros(self.instance.J),
            Tseg=tseg,
        )

    def _result(
        self,
        anchor_solution: SPSolution,
        candidate_sequence: list[int],
        candidate_upper_bound: float,
    ) -> SparseSelfImprovementResult:
        anchor_majorizer = FixedSolutionUpperBound.from_solution(
            self.instance,
            anchor_solution,
        )
        anchor_matrix = directed_total_contribution_matrix(anchor_majorizer)
        anchor_value = float(anchor_solution.objective)
        return SparseSelfImprovementResult(
            instance_id=self.instance.instance_id,
            anchor_sequence=list(anchor_solution.sequence),
            anchor_solution=anchor_solution,
            anchor_exact_makespan=anchor_value,
            anchor_upper_bound_matrix=anchor_matrix,
            sampled_upper_bounds=np.array([anchor_value]),
            top_k_start_sequences=[list(anchor_solution.sequence)],
            local_search_results=[],
            expert_sequence=list(anchor_solution.sequence),
            expert_upper_bound=anchor_value,
            expert_solution=anchor_solution,
            expert_exact_makespan=anchor_value,
            supervision_upper_bound_matrix=anchor_matrix.copy(),
            anchor_gurobi_calls=1,
            expert_gurobi_calls=0,
            gurobi_runtime=float(anchor_solution.solve_time),
            local_search_runtime=0.1,
            total_runtime=0.5,
            generated_moves=0,
            evaluated_moves=0,
            dual_pruned_moves=0,
            accepted_moves=0,
            linear_dual_count=0,
            quadratic_dual_count=0,
            near_miss_candidate_sequence=list(candidate_sequence),
            near_miss_candidate_upper_bound=float(candidate_upper_bound),
            near_miss_relative_gap=(candidate_upper_bound - anchor_value) / anchor_value,
            expert_source="anchor",
            rl_baseline_upper_bound=anchor_value,
        )

    def _apply(self, result: SparseSelfImprovementResult, solution: SPSolution):
        return apply_near_miss_validation(
            result,
            self.instance,
            solution,
            acceptance_epsilon=1e-6,
            upper_bound_tolerance=1e-8,
            feasibility_tolerance=1e-8,
        )

    def test_best_non_anchor_candidate_uses_raw_smallest_upper_bound(self) -> None:
        anchor_value = 10.0
        searches = [
            SimpleNamespace(final_sequence=self.worse_route, final_upper_bound=10.4),
            SimpleNamespace(final_sequence=self.better_route, final_upper_bound=10.1),
            SimpleNamespace(final_sequence=self.worse_route, final_upper_bound=10.2),
        ]
        sequence, upper_bound, relative_gap = _best_non_anchor_local_candidate(
            searches,
            self.worse_route,
            anchor_value,
        )
        self.assertEqual(sequence, self.better_route)
        self.assertAlmostEqual(upper_bound, 10.1)
        self.assertAlmostEqual(relative_gap, 0.01)

    def test_near_miss_task_is_pickleable_and_only_calls_fixed_route_solver(self) -> None:
        task = NearMissValidationTask(
            instance=self.instance,
            candidate_sequence=self.better_route,
            solver_config=SolverConfig(gurobi_threads=1),
        )
        restored = pickle.loads(pickle.dumps(task))
        with patch(
            "src.JointSelfImprovement.solve",
            return_value=self.better_solution,
        ) as mocked_solve:
            solution = run_near_miss_validation_task(restored)
        self.assertIs(solution, self.better_solution)
        mocked_solve.assert_called_once()
        solved_instance, solved_sequence, solved_config = mocked_solve.call_args.args
        self.assertEqual(solved_instance.instance_id, self.instance.instance_id)
        self.assertEqual(solved_sequence, self.better_route)
        self.assertEqual(solved_config, restored.solver_config)

    def test_exact_improvement_accepts_loose_near_miss_upper_bound(self) -> None:
        candidate_upper_bound = float(self.worse_solution.objective + 0.2)
        result = self._result(
            self.worse_solution,
            self.better_route,
            candidate_upper_bound,
        )
        initial_runtime = result.gurobi_runtime

        updated = self._apply(result, self.better_solution)

        self.assertIs(updated, result)
        self.assertTrue(result.near_miss_attempted)
        self.assertTrue(result.near_miss_accepted)
        self.assertFalse(result.near_miss_validation_failed)
        self.assertEqual(result.total_gurobi_calls, 2)
        self.assertAlmostEqual(
            result.gurobi_runtime,
            initial_runtime + self.better_solution.solve_time,
        )
        self.assertEqual(result.expert_sequence, self.better_route)
        self.assertEqual(result.expert_source, "near_miss_exact")
        self.assertAlmostEqual(result.expert_upper_bound, candidate_upper_bound)
        self.assertAlmostEqual(
            result.rl_baseline_upper_bound,
            self.worse_solution.objective,
        )
        self.assertAlmostEqual(
            result.near_miss_exact_makespan,
            self.better_solution.objective,
        )
        expected_supervision = directed_total_contribution_matrix(
            FixedSolutionUpperBound.from_solution(
                self.instance,
                self.better_solution,
                feasibility_tolerance=1e-8,
            )
        )
        np.testing.assert_allclose(
            result.supervision_upper_bound_matrix,
            expected_supervision,
        )
        np.testing.assert_allclose(
            result.expert_upper_bound_matrix,
            expected_supervision,
        )
        self.assertTrue(result.expert_better)

    def test_upper_bound_violation_is_charged_and_rejected(self) -> None:
        result = self._result(
            self.worse_solution,
            self.better_route,
            float(self.better_solution.objective - 0.01),
        )

        with self.assertRaisesRegex(RuntimeError, "upper-bound violation"):
            self._apply(result, self.better_solution)

        self.assertEqual(result.total_gurobi_calls, 2)
        self.assertTrue(result.near_miss_attempted)
        self.assertFalse(result.near_miss_accepted)
        self.assertTrue(result.near_miss_validation_failed)
        self.assertEqual(result.expert_source, "anchor")

    def test_exact_non_improvement_is_charged_and_falls_back_to_anchor(self) -> None:
        candidate_upper_bound = float(self.worse_solution.objective + 0.2)
        result = self._result(
            self.better_solution,
            self.worse_route,
            candidate_upper_bound,
        )

        self._apply(result, self.worse_solution)

        self.assertTrue(result.near_miss_attempted)
        self.assertFalse(result.near_miss_accepted)
        self.assertFalse(result.near_miss_validation_failed)
        self.assertEqual(result.total_gurobi_calls, 2)
        self.assertEqual(result.expert_sequence, self.better_route)
        self.assertEqual(result.expert_source, "anchor")
        self.assertAlmostEqual(
            result.rl_baseline_upper_bound,
            self.better_solution.objective,
        )
        self.assertAlmostEqual(
            result.near_miss_exact_makespan,
            self.worse_solution.objective,
        )

    def test_failed_validation_is_charged_and_marked_failed(self) -> None:
        result = self._result(
            self.worse_solution,
            self.better_route,
            float(self.worse_solution.objective + 0.2),
        )
        failed = SPSolution(
            objective=float("inf"),
            makespan=float("inf"),
            success=False,
            status="ERROR:RuntimeError",
            solve_time=0.75,
            # A failed backend is allowed to return no usable route.  The
            # failure path must still charge the solve and fall back safely.
            sequence=[],
        )

        self._apply(result, failed)

        self.assertEqual(result.total_gurobi_calls, 2)
        self.assertTrue(result.near_miss_attempted)
        self.assertFalse(result.near_miss_accepted)
        self.assertTrue(result.near_miss_validation_failed)
        self.assertTrue(result.final_validation_failed)
        self.assertEqual(result.expert_source, "anchor")


if __name__ == "__main__":
    unittest.main()
