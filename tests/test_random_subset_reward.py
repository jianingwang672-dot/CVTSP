from __future__ import annotations

import random
import unittest
from concurrent.futures import ThreadPoolExecutor
from itertools import permutations
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.CVPSolver import SPSolution, SolverConfig
from src.JointSelfImprovementTrainer import JointSelfImprovementTrainer


def _solution(task, objective: float) -> SPSolution:
    return SPSolution(objective=objective, makespan=objective, success=True, status="2",
                      solve_time=0.1, sequence=list(task.target_sequence))


def _objective(task) -> float:
    # Deterministic, route-dependent exact value.
    return 10.0 + sum((i + 1) * node for i, node in enumerate(task.target_sequence))


class RandomSubsetRewardTest(unittest.TestCase):
    def _trainer(self, min_routes=4, max_routes=8) -> JointSelfImprovementTrainer:
        trainer = object.__new__(JointSelfImprovementTrainer)
        trainer.device = torch.device("cpu")
        trainer.env_params = {"min_problem_size": 20, "max_problem_size": 100}
        trainer.trainer_params = {"seed": 123}
        trainer.random_subset_min_routes = min_routes
        trainer.random_subset_max_routes = max_routes
        trainer.dense_reward_source = "random_subset_exact"
        trainer.hybrid_advantage_epsilon = 1e-6
        trainer.si_final_solver_config = SolverConfig(solver_backend="gurobi")
        trainer._si_executor = None
        trainer._joint_batch_index = 7
        trainer._pending_after_sparse_submit = None
        trainer._random_subset_handle = None
        return trainer

    @staticmethod
    def _batch(problem_size=5, pomo=12, instances=3):
        routes = [list(p) for p in permutations(range(problem_size))][:pomo]
        return ([SimpleNamespace(J=problem_size, instance_id=f"i{k}") for k in range(instances)],
                [[list(r) for r in routes] for _ in range(instances)])

    def test_subset_size_is_linear_in_j(self) -> None:
        trainer = self._trainer()
        self.assertEqual([trainer._random_subset_size(j) for j in (20, 40, 60, 80, 100)],
                         [4, 5, 6, 7, 8])
        self.assertEqual(trainer._random_subset_size(10), 4)
        self.assertEqual(trainer._random_subset_size(150), 8)

    def test_selection_is_deterministic_and_leaves_global_rng_untouched(self) -> None:
        trainer = self._trainer()
        instances, routes = self._batch()
        np.random.seed(0); torch.manual_seed(0); random.seed(0)
        before = (np.random.get_state()[1].copy(), torch.get_rng_state().clone(), random.getstate())
        first = trainer._start_random_subset_solves(instances, routes)
        after = (np.random.get_state()[1], torch.get_rng_state(), random.getstate())
        self.assertTrue(np.array_equal(before[0], after[0]))
        self.assertTrue(torch.equal(before[1], after[1]))
        self.assertEqual(before[2], after[2])
        second = trainer._start_random_subset_solves(instances, routes)
        for a, b in zip(first["subsets"], second["subsets"]):
            self.assertTrue(np.array_equal(a, b))
            self.assertEqual(len(a), 4)  # J=5 is below min_problem_size
            self.assertEqual(len(set(a.tolist())), len(a))
        trainer._joint_batch_index = 8
        third = trainer._start_random_subset_solves(instances, routes)
        self.assertFalse(all(np.array_equal(a, b) for a, b in zip(first["subsets"], third["subsets"])))

    def test_dense_loss_matches_manual_subset_advantages(self) -> None:
        trainer = self._trainer()
        instances, routes = self._batch()
        weights = torch.nn.Parameter(torch.zeros(3, 12))
        log_probs = weights + torch.arange(36, dtype=torch.float32).view(3, 12) * 0.01
        trainer._random_subset_handle = trainer._start_random_subset_solves(instances, routes)
        with patch("src.JointSelfImprovementTrainer.run_target_reward_solve_task",
                   side_effect=lambda task: _solution(task, _objective(task))):
            loss, advantages, stats = trainer._random_subset_dense_loss(log_probs)
        expected = 0.0
        count = 0
        for k, subset in enumerate(trainer._random_subset_handle["subsets"]):
            values = np.asarray([_objective(SimpleNamespace(target_sequence=routes[k][i])) for i in subset])
            adv = -(values - values.mean()) / values.std()
            expected -= float(np.sum(adv * log_probs.detach().numpy()[k, subset]))
            count += len(subset)
        self.assertAlmostEqual(float(loss), expected / count, places=5)
        self.assertEqual(stats["random_subset_routes"], 12)
        self.assertEqual(stats["random_subset_gurobi_calls"], 12)
        self.assertEqual(advantages.numel(), 12)
        loss.backward()
        grad = weights.grad.numpy()
        for k, subset in enumerate(trainer._random_subset_handle["subsets"]):
            outside = np.setdiff1d(np.arange(12), subset)
            self.assertTrue(np.all(grad[k, outside] == 0.0))
            self.assertAlmostEqual(float(np.sum(grad[k, subset])), 0.0, places=6)

    def test_executor_hook_submits_once_and_results_are_collected(self) -> None:
        trainer = self._trainer()
        instances, routes = self._batch()
        trainer._si_executor = ThreadPoolExecutor(max_workers=2)
        try:
            with patch("src.JointSelfImprovementTrainer.run_target_reward_solve_task",
                       side_effect=lambda task: _solution(task, _objective(task))) as solve:
                handle = trainer._start_random_subset_solves(instances, routes)
                trainer._pending_after_sparse_submit = handle["submit"]
                trainer._run_after_sparse_submit()
                trainer._run_after_sparse_submit()  # consumed: no second submission
                self.assertIsNone(trainer._pending_after_sparse_submit)
                values, calls, failures, _ = trainer._collect_random_subset_solves(handle)
                self.assertEqual(solve.call_count, calls)
            self.assertEqual(failures, 0)
            self.assertTrue(all(np.all(np.isfinite(v)) and len(v) == 4 for v in values))
        finally:
            trainer._si_executor.shutdown()

    def test_failed_solves_are_dropped_from_the_group(self) -> None:
        trainer = self._trainer()
        instances, routes = self._batch(instances=1)
        trainer._random_subset_handle = trainer._start_random_subset_solves(instances, routes)
        failing = trainer._random_subset_handle["tasks"][0].target_sequence

        def solve(task):
            if task.target_sequence == failing:
                return SPSolution(objective=float("inf"), makespan=float("inf"), success=False,
                                  status="ERROR:x", solve_time=0.0, sequence=list(task.target_sequence))
            return _solution(task, _objective(task))

        with patch("src.JointSelfImprovementTrainer.run_target_reward_solve_task", side_effect=solve):
            _, _, stats = trainer._random_subset_dense_loss(torch.zeros(1, 12, requires_grad=True))
        self.assertEqual(stats["random_subset_gurobi_failures"], 1)
        self.assertEqual(stats["random_subset_routes"], 3)


if __name__ == "__main__":
    unittest.main()
