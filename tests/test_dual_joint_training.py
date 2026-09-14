from __future__ import annotations

import unittest

import numpy as np
import torch

from src.CertifiedUpperBound import FixedSolutionUpperBound
from src.CertifiedUpperBound import (
    _materialize_move,
    iter_candidate_move_indices,
)
from src.CVPSolver import SolverConfig, solve
from src.DualContributions import (
    FixedDualLowerBound,
    directed_total_contribution_matrix,
    ordered_step_costs,
)
from src.EdgeCostPredictor import directed_route_cost
from src.JointSelfImprovement import SharedDirectedEdgeDecoder, dense_reward_to_go
from src.TSPModel import ModelConfig
from src.TSProblemDef import load_instance


class DualAndJointTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.instance = load_instance("instance/Data/Example_1.txt")
        cls.config = SolverConfig(
            solver_backend="gurobi",
            gurobi_threads=1,
            gurobi_explicit_qcp=True,
            collect_duals=True,
            gurobi_params={
                "FeasibilityTol": 1e-8,
                "OptimalityTol": 1e-8,
                "NumericFocus": 2,
            },
        )
        cls.anchor = list(range(cls.instance.J))
        cls.solution = solve(cls.instance, cls.anchor, cls.config)
        if not cls.solution.success:
            raise unittest.SkipTest("Gurobi is not available")

    def test_gurobi_returns_linear_and_quadratic_duals(self) -> None:
        self.assertIn("out_x_def[0]", self.solution.linear_duals)
        self.assertIn("out_qc[0]", self.solution.quadratic_duals)
        self.assertTrue(np.isfinite(self.solution.linear_duals["out_x_def[0]"]["pi"]))
        self.assertTrue(np.isfinite(self.solution.quadratic_duals["out_qc[0]"]["qcpi"]))

    def test_reused_dual_is_lower_bound_for_changed_tours(self) -> None:
        dual = FixedDualLowerBound.from_solution(self.instance, self.solution)
        rng = np.random.default_rng(20260813)
        self.assertAlmostEqual(dual.lower_bound(self.anchor), self.solution.objective, places=6)
        for _ in range(3):
            candidate = rng.permutation(self.instance.J).tolist()
            candidate_solution = solve(self.instance, candidate, self.config)
            self.assertTrue(candidate_solution.success)
            self.assertLessEqual(
                dual.lower_bound(candidate),
                candidate_solution.objective + 5e-4,
            )

    def test_index_only_dual_move_score_matches_materialized_tour(self) -> None:
        dual = FixedDualLowerBound.from_solution(self.instance, self.solution)
        for kind, first, second in iter_candidate_move_indices(
            dual.anchor_sequence,
            move_types=("swap", "relocate", "two_opt"),
        ):
            candidate = _materialize_move(dual.anchor_sequence, kind, first, second)
            self.assertAlmostEqual(
                dual.move_lower_bound_indices(kind, first, second),
                dual.lower_bound(candidate),
                places=9,
            )

    def test_dense_contributions_reconstruct_objective(self) -> None:
        majorizer = FixedSolutionUpperBound.from_solution(self.instance, self.solution)
        matrix = directed_total_contribution_matrix(majorizer)
        self.assertAlmostEqual(
            directed_route_cost(self.anchor, matrix),
            self.solution.objective,
            places=4,
        )
        step_costs = ordered_step_costs(self.solution)
        self.assertAlmostEqual(float(step_costs.sum()), self.solution.objective, places=4)
        returns = dense_reward_to_go(step_costs)
        self.assertAlmostEqual(float(returns[0]), -self.solution.objective, places=4)

    def test_parallel_edge_decoder_shape_and_gradient_boundary(self) -> None:
        decoder = SharedDirectedEdgeDecoder(ModelConfig(), hidden_dim=32)
        encoded = torch.randn(2, self.instance.J + 1, 128, requires_grad=True)
        prediction = decoder(encoded.detach())
        self.assertEqual(tuple(prediction.shape), (2, self.instance.J + 1, self.instance.J + 1))
        prediction.mean().backward()
        self.assertIsNone(encoded.grad)
        self.assertTrue(any(parameter.grad is not None for parameter in decoder.parameters()))


if __name__ == "__main__":
    unittest.main()
