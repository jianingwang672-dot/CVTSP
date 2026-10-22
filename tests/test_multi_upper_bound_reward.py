from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.CVPSolver import SPSolution, SolverConfig
from src.JointSelfImprovementTrainer import JointSelfImprovementTrainer


class _ScalarPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))


class MultiUpperBoundRewardTest(unittest.TestCase):
    @staticmethod
    def _matrix(route_01: tuple[float, float, float], route_10: tuple[float, float, float]):
        matrix = np.zeros((3, 3), dtype=np.float32)
        matrix[0, 1], matrix[1, 2], matrix[2, 0] = route_01
        matrix[0, 2], matrix[2, 1], matrix[1, 0] = route_10
        return matrix

    def _trainer(self) -> JointSelfImprovementTrainer:
        trainer = object.__new__(JointSelfImprovementTrainer)
        trainer.device = torch.device("cpu")
        trainer.grad_clip = 1.0
        trainer.model = _ScalarPolicy()
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.0)
        trainer.scheduler = None
        trainer.scheduler_interval = "epoch"
        trainer.global_step = 0
        trainer.si_final_solver_config = SolverConfig(solver_backend="gurobi")
        trainer.si_upper_bound_tolerance = 1e-4
        trainer._si_executor = None
        trainer.exact_expert_aux_max_weight = 1.0
        trainer.exact_expert_advantage_clip = 1.0
        trainer.hybrid_advantage_epsilon = 1e-6
        trainer.competence_il_start = 0.5
        trainer.competence_il_full = 0.85
        trainer._previous_epoch_il_accuracy = 0.5
        return trainer

    @staticmethod
    def _exact_solution(task, objective: float) -> SPSolution:
        return SPSolution(
            objective=objective,
            makespan=objective,
            success=True,
            status="2",
            solve_time=0.25,
            sequence=list(task.target_sequence),
        )

    def test_uses_route_level_minimum_then_exactly_solves_best_target(self) -> None:
        trainer = self._trainer()
        anchor_matrix = self._matrix((1.0, 100.0, 1.0), (8.0, 7.0, 5.0))
        expert_matrix = self._matrix((100.0, 1.0, 100.0), (2.0, 1.0, 2.0))
        results = [
            SimpleNamespace(
                anchor_upper_bound_matrix=anchor_matrix,
                expert_upper_bound_matrix=expert_matrix,
            )
        ]
        target_sequences = [[[0, 1], [1, 0]]]
        target_step_log_probs = trainer.model.weight * torch.tensor(
            [[[1.0], [2.0]]], dtype=torch.float32
        )

        instance = SimpleNamespace(J=2, instance_id="reward-unit")
        with patch(
            "src.JointSelfImprovementTrainer.run_target_reward_solve_task",
            side_effect=lambda task: self._exact_solution(task, 4.0),
        ) as mocked_solve:
            loss, upper_bounds, reward_costs, diagnostics = trainer._policy_rl_update(
                target_step_log_probs,
                target_sequences,
                results,
                [instance],
            )

        # Complete-route bounds are min((102, 20), (201, 5)) = (102, 5).
        # An invalid edge-wise minimum would instead report 3 for route [0, 1].
        np.testing.assert_allclose(upper_bounds, np.array([[102.0, 5.0]], dtype=np.float32))
        np.testing.assert_allclose(reward_costs, np.array([[102.0, 4.0]], dtype=np.float32))
        # The best-bound route receives its exact cost. Rewards (-102, -4)
        # give advantages (-49, 49), hence the synthetic loss is -24.5.
        self.assertAlmostEqual(loss, -24.5)
        self.assertEqual(int(diagnostics["selected_indices"][0]), 1)
        self.assertEqual(diagnostics["gurobi_calls"], 1)
        self.assertEqual(diagnostics["gurobi_failures"], 0)
        mocked_solve.assert_called_once()

    def test_missing_expert_table_falls_back_to_anchor_table(self) -> None:
        trainer = self._trainer()
        anchor_matrix = self._matrix((1.0, 100.0, 1.0), (8.0, 7.0, 5.0))
        results = [
            SimpleNamespace(
                anchor_upper_bound_matrix=anchor_matrix,
                expert_upper_bound_matrix=None,
            )
        ]
        target_sequences = [[[0, 1], [1, 0]]]
        target_step_log_probs = trainer.model.weight * torch.tensor(
            [[[1.0], [2.0]]], dtype=torch.float32
        )

        instance = SimpleNamespace(J=2, instance_id="fallback-unit")
        with patch(
            "src.JointSelfImprovementTrainer.run_target_reward_solve_task",
            side_effect=lambda task: self._exact_solution(task, 18.0),
        ):
            loss, upper_bounds, reward_costs, _ = trainer._policy_rl_update(
                target_step_log_probs,
                target_sequences,
                results,
                [instance],
            )

        np.testing.assert_allclose(upper_bounds, np.array([[102.0, 20.0]], dtype=np.float32))
        np.testing.assert_allclose(reward_costs, np.array([[102.0, 18.0]], dtype=np.float32))
        self.assertAlmostEqual(loss, -21.0)

    def test_exact_target_expert_mode_uses_exact_expert_baseline(self) -> None:
        trainer = self._trainer()
        trainer.rl_reward_mode = "exact_target_expert"
        anchor_matrix = self._matrix((1.0, 100.0, 1.0), (8.0, 7.0, 5.0))
        results = [
            SimpleNamespace(
                anchor_upper_bound_matrix=anchor_matrix,
                expert_upper_bound_matrix=None,
                expert_exact_makespan=10.0,
            )
        ]
        target_sequences = [[[0, 1], [1, 0]]]
        target_step_log_probs = trainer.model.weight * torch.tensor(
            [[[1.0], [2.0]]], dtype=torch.float32
        )
        instance = SimpleNamespace(J=2, instance_id="exact-baseline-unit")

        with patch(
            "src.JointSelfImprovementTrainer.run_target_reward_solve_task",
            side_effect=lambda task: self._exact_solution(task, 4.0),
        ):
            loss, _, _, diagnostics = trainer._policy_rl_update(
                target_step_log_probs,
                target_sequences,
                results,
                [instance],
            )

        # Best-U Target is route index 1. Its exact advantage is 10 - 4 = 6,
        # and its synthetic log probability is 2, hence loss = -(6 * 2).
        self.assertAlmostEqual(loss, -12.0)
        self.assertEqual(diagnostics["exact_rl_valid_count"], 1)
        self.assertAlmostEqual(diagnostics["exact_rl_advantage_sum"], 6.0)
        self.assertEqual(diagnostics["exact_rl_positive_advantage_count"], 1)

    def test_combined_mode_standardizes_pomo_and_bounds_exact_auxiliary(self) -> None:
        trainer = self._trainer()
        trainer.rl_reward_mode = "hybrid_pomo_exact_expert"
        trainer._previous_epoch_il_accuracy = 0.85
        anchor_matrix = self._matrix((1.0, 100.0, 1.0), (8.0, 7.0, 5.0))
        results = [
            SimpleNamespace(
                anchor_upper_bound_matrix=anchor_matrix,
                expert_upper_bound_matrix=None,
                expert_exact_makespan=10.0,
            )
        ]
        target_sequences = [[[0, 1], [1, 0]]]
        target_step_log_probs = trainer.model.weight * torch.tensor(
            [[[1.0], [2.0]]], dtype=torch.float32
        )
        instance = SimpleNamespace(J=2, instance_id="combined-unit")

        with patch(
            "src.JointSelfImprovementTrainer.run_target_reward_solve_task",
            side_effect=lambda task: self._exact_solution(task, 4.0),
        ):
            loss, _, _, diagnostics = trainer._policy_rl_update(
                target_step_log_probs,
                target_sequences,
                results,
                [instance],
            )

        # Hybrid rewards (-102, -4) standardize to advantages (-1, 1),
        # yielding -0.5. The exact relative advantage is (10-4)/10=0.6;
        # with selected log probability 2 its auxiliary loss is -1.2.
        self.assertAlmostEqual(diagnostics["hybrid_policy_loss"], -0.5)
        self.assertAlmostEqual(diagnostics["exact_expert_policy_loss"], -1.2)
        self.assertAlmostEqual(diagnostics["exact_expert_aux_weight"], 1.0)
        self.assertAlmostEqual(loss, -1.7)

    def test_combined_exact_auxiliary_is_zero_at_cold_start(self) -> None:
        trainer = self._trainer()
        trainer.rl_reward_mode = "hybrid_pomo_exact_expert"
        trainer._previous_epoch_il_accuracy = 0.5
        anchor_matrix = self._matrix((1.0, 100.0, 1.0), (8.0, 7.0, 5.0))
        results = [
            SimpleNamespace(
                anchor_upper_bound_matrix=anchor_matrix,
                expert_upper_bound_matrix=None,
                expert_exact_makespan=10.0,
            )
        ]
        target_sequences = [[[0, 1], [1, 0]]]
        target_step_log_probs = trainer.model.weight * torch.tensor(
            [[[1.0], [2.0]]], dtype=torch.float32
        )
        instance = SimpleNamespace(J=2, instance_id="cold-start-unit")

        with patch(
            "src.JointSelfImprovementTrainer.run_target_reward_solve_task",
            side_effect=lambda task: self._exact_solution(task, 4.0),
        ):
            loss, _, _, diagnostics = trainer._policy_rl_update(
                target_step_log_probs,
                target_sequences,
                results,
                [instance],
            )

        self.assertAlmostEqual(diagnostics["exact_expert_aux_weight"], 0.0)
        self.assertAlmostEqual(loss, diagnostics["hybrid_policy_loss"])


if __name__ == "__main__":
    unittest.main()
