from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.CertifiedLocalSearch import FixedAnchorLocalSearchConfig
from src.CVPSolver import SPSolution, SolverConfig
from src.JointSelfImprovement import C0RefreshOutcome
from src.JointSelfImprovementTrainer import JointSelfImprovementTrainer


def _solution(sequence, objective):
    return SPSolution(objective=float(objective), makespan=float(objective), success=True,
                      status="2", solve_time=0.5, sequence=list(sequence))


def _result(anchor, anchor_value, *, near_miss=True, calls=1, failed=False):
    return SimpleNamespace(
        anchor_sequence=list(anchor), anchor_exact_makespan=float(anchor_value),
        expert_sequence=list(anchor), expert_exact_makespan=float(anchor_value),
        expert_solution=_solution(anchor, anchor_value), expert_upper_bound=float(anchor_value),
        supervision_upper_bound_matrix=np.zeros((3, 3)), expert_source="anchor",
        expert_gurobi_calls=calls, final_validation_failed=failed,
        near_miss_attempted=near_miss, near_miss_candidate_sequence=[1, 0, 2],
        local_search_results=[SimpleNamespace(final_sequence=[2, 1, 0])],
        c0_refresh_rounds_run=0, c0_refresh_gurobi_calls=0, c0_refresh_accepted=0,
        c0_refresh_runtime=0.0, gurobi_runtime=1.0,
    )


class C0RefreshTrainerTest(unittest.TestCase):
    def _trainer(self, rounds=2):
        trainer = object.__new__(JointSelfImprovementTrainer)
        trainer.si_c0_refresh_rounds = rounds
        trainer._si_executor = None
        trainer.si_final_solver_config = SolverConfig(solver_backend="gurobi")
        trainer.si_local_search_config = FixedAnchorLocalSearchConfig()
        trainer.si_route_start_diversity_count = 10
        trainer.si_route_start_diversity_pool_size = 60
        trainer.si_route_start_diversity_max_relative_gap = 0.05
        trainer.si_feasibility_tolerance = 1e-5
        trainer.si_upper_bound_tolerance = 1e-4
        trainer._route_start_top_k = lambda j: 30
        return trainer

    def test_rounds_accept_reject_and_track_solved_routes(self) -> None:
        trainer = self._trainer(rounds=3)
        instance = SimpleNamespace(J=3, instance_id="x")
        result = _result([0, 1, 2], 10.0)
        near_miss_solution = _solution([1, 0, 2], 10.5)  # rejected near-miss
        seen = []
        outcomes = iter([
            # round 1: better route found from the near-miss-centred C0
            C0RefreshOutcome([[2, 0, 1]], [2, 0, 1], 10.2, _solution([2, 0, 1], 9.0), False, True,
                             np.ones((3, 3)), 0.1, 0.2),
            # round 2: candidate solved but worse than the new incumbent
            C0RefreshOutcome([[1, 2, 0]], [1, 2, 0], 9.5, _solution([1, 2, 0], 9.4), False, False,
                             None, 0.1, 0.2),
            # round 3: no unsolved candidate -> instance leaves the loop
            C0RefreshOutcome([[2, 0, 1]], None, float("inf"), None, False, False, None, 0.1, 0.0),
        ])

        def fake(task):
            seen.append((list(task.base_solution.sequence), [list(r) for r in task.solved_sequences],
                         task.incumbent_exact_makespan))
            return next(outcomes)

        with patch("src.JointSelfImprovementTrainer.run_c0_refresh_task", side_effect=fake):
            trainer._run_c0_refresh([instance], [[[0, 1, 2], [0, 2, 1]]], [result], {0: near_miss_solution})

        # Round 1 refreshes C0 around the rejected near-miss and excludes solved routes.
        self.assertEqual(seen[0], ([1, 0, 2], [[0, 1, 2], [1, 0, 2]], 10.0))
        # Round 2 is centred on the newly solved route with the new incumbent.
        self.assertEqual(seen[1], ([2, 0, 1], [[0, 1, 2], [1, 0, 2], [2, 0, 1]], 9.0))
        # Round 3 is centred on the rejected round-2 route.
        self.assertEqual(seen[2][0], [1, 2, 0])
        self.assertEqual(result.expert_sequence, [2, 0, 1])
        self.assertEqual(result.expert_exact_makespan, 9.0)
        self.assertEqual(result.expert_source, "c0_refresh")
        self.assertTrue(np.array_equal(result.supervision_upper_bound_matrix, np.ones((3, 3))))
        self.assertEqual(result.expert_gurobi_calls, 1)  # sparse budget untouched
        self.assertEqual(result.c0_refresh_gurobi_calls, 2)
        self.assertEqual(result.c0_refresh_accepted, 1)
        self.assertEqual(result.c0_refresh_rounds_run, 3)

    def test_only_instances_with_one_expert_solve_are_refreshed(self) -> None:
        trainer = self._trainer(rounds=1)
        instances = [SimpleNamespace(J=3, instance_id=str(k)) for k in range(3)]
        results = [
            _result([0, 1, 2], 10.0, near_miss=False, calls=0),  # no expert-side solve
            _result([0, 1, 2], 10.0, near_miss=True, calls=1, failed=True),
            _result([0, 1, 2], 10.0, near_miss=False, calls=1),  # certified expert
        ]
        results[2].expert_sequence = [2, 1, 0]
        results[2].expert_solution = _solution([2, 1, 0], 9.5)
        calls = []

        def fake(task):
            calls.append(task.instance.instance_id)
            return C0RefreshOutcome([], None, float("inf"), None, False, False, None, 0.0, 0.0)

        with patch("src.JointSelfImprovementTrainer.run_c0_refresh_task", side_effect=fake):
            trainer._run_c0_refresh(instances, [[[0, 1, 2]]] * 3, results, {})
        self.assertEqual(calls, ["2"])

    def test_failed_refresh_solve_is_charged_and_stops(self) -> None:
        trainer = self._trainer(rounds=2)
        result = _result([0, 1, 2], 10.0)
        outcome = C0RefreshOutcome([[2, 0, 1]], [2, 0, 1], 9.0, None, True, False, None, 0.0, 0.3)
        with patch("src.JointSelfImprovementTrainer.run_c0_refresh_task", return_value=outcome) as run:
            trainer._run_c0_refresh([SimpleNamespace(J=3, instance_id="x")], [[[0, 1, 2]]], [result],
                                    {0: _solution([1, 0, 2], 10.5)})
        self.assertEqual(run.call_count, 1)
        self.assertEqual(result.c0_refresh_gurobi_calls, 1)
        self.assertEqual(result.expert_sequence, [0, 1, 2])


class UnchangedIlWeightTest(unittest.TestCase):
    def _trainer(self, unchanged_weight):
        trainer = object.__new__(JointSelfImprovementTrainer)
        trainer.model = torch.nn.Linear(1, 1)
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.0)
        trainer.scheduler = None
        trainer.scheduler_interval = "epoch"
        trainer.global_step = 0
        trainer.device = torch.device("cpu")
        trainer.grad_clip = 1.0
        trainer.changed_expert_il_weight = 5
        trainer.unchanged_expert_il_weight = unchanged_weight
        trainer.expert_replay_fraction = 0.0
        trainer.imitation_inner_epochs = 1
        trainer.imitation_batch_size = 100
        trainer._expert_replay_pool = []
        return trainer

    def _count(self, unchanged_weight):
        trainer = self._trainer(unchanged_weight)
        instances = [SimpleNamespace(J=2) for _ in range(4)]

        def fake(model, batch, experts, *, device):
            loss = next(model.parameters()).sum() * 0.0 + 1.0
            return loss, len(batch), None, len(batch)

        with patch("src.JointSelfImprovementTrainer.teacher_forcing_loss", side_effect=fake):
            _, _, effective, _ = trainer._imitation_update(
                instances, [[0, 1]] * 4, [True, False, False, False])
        return effective

    def test_zero_weight_trains_only_on_changed_experts(self) -> None:
        self.assertEqual(self._count(1), 5 + 3)
        self.assertEqual(self._count(0), 5)


if __name__ == "__main__":
    unittest.main()
