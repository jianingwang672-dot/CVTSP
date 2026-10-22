from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.JointSelfImprovementTrainer import JointSelfImprovementTrainer


def _result(
    gap: float,
    *,
    changed: bool = False,
    expert_calls: int = 0,
    has_candidate: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        near_miss_candidate_sequence=[1, 0] if has_candidate else None,
        anchor_sequence=[0, 1],
        near_miss_relative_gap=gap,
        expert_changed=changed,
        expert_gurobi_calls=expert_calls,
    )


class DynamicTopKTest(unittest.TestCase):
    def test_ratio_uses_ceil_per_problem_size(self) -> None:
        trainer = object.__new__(JointSelfImprovementTrainer)
        trainer.si_route_start_top_k = 4
        trainer.si_route_start_top_k_ratio = 0.15
        trainer.si_route_start_top_k_min = 0

        self.assertEqual(
            {size: trainer._route_start_top_k(size) for size in (20, 21, 60, 100)},
            {20: 3, 21: 4, 60: 9, 100: 15},
        )

    def test_zero_ratio_uses_fixed_fallback(self) -> None:
        trainer = object.__new__(JointSelfImprovementTrainer)
        trainer.si_route_start_top_k = 4
        trainer.si_route_start_top_k_ratio = 0.0
        trainer.si_route_start_top_k_min = 0
        self.assertEqual(trainer._route_start_top_k(100), 4)

    def test_ratio_can_use_a_fixed_minimum(self) -> None:
        trainer = object.__new__(JointSelfImprovementTrainer)
        trainer.si_route_start_top_k = 30
        trainer.si_route_start_top_k_ratio = 0.5
        trainer.si_route_start_top_k_min = 30
        self.assertEqual(
            {size: trainer._route_start_top_k(size) for size in (20, 60, 80, 100)},
            {20: 30, 60: 30, 80: 40, 100: 50},
        )


class ImitationScheduleTest(unittest.TestCase):
    def test_predeclared_epoch_schedule(self) -> None:
        trainer = object.__new__(JointSelfImprovementTrainer)
        trainer.imitation_inner_epochs_schedule = ((1, 5), (51, 2), (101, 1))
        self.assertEqual(
            [trainer._imitation_inner_epochs_for_epoch(epoch) for epoch in (1, 50, 51, 100, 101, 200)],
            [5, 5, 2, 2, 1, 1],
        )


class NearMissBatchSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.results = [
            _result(0.20),
            _result(0.10),
            _result(0.10),
            _result(0.30),
            _result(0.40),
            _result(0.50),
            _result(0.60),
            _result(0.70),
            _result(0.01, changed=True),
            _result(0.02, expert_calls=1),
            _result(0.03, has_candidate=False),
            _result(float("inf")),
        ]

    def test_one_percent_of_512_selects_six_with_stable_ties(self) -> None:
        eligible, selected = JointSelfImprovementTrainer._select_near_miss_indices(
            self.results,
            batch_size=512,
            rate=0.01,
        )

        self.assertEqual(len(selected), 6)
        self.assertEqual(selected[:2], [1, 2])
        self.assertNotIn(8, eligible)
        self.assertNotIn(9, eligible)
        self.assertNotIn(10, eligible)
        self.assertNotIn(11, eligible)

    def test_zero_rate_submits_no_validation(self) -> None:
        eligible, selected = JointSelfImprovementTrainer._select_near_miss_indices(
            self.results,
            batch_size=512,
            rate=0.0,
        )

        self.assertTrue(eligible)
        self.assertEqual(selected, [])


class AdaptiveNearMissRateTest(unittest.TestCase):
    def _trainer(self) -> JointSelfImprovementTrainer:
        trainer = object.__new__(JointSelfImprovementTrainer)
        trainer.si_adaptive_near_miss = True
        trainer.si_near_miss_min_rate = 0.01
        trainer.si_near_miss_rate = 0.05
        trainer.competence_il_start = 0.5
        trainer.competence_il_full = 0.85
        trainer.competence_target_better_full = 0.2
        return trainer

    def test_cold_start_uses_one_percent(self) -> None:
        trainer = self._trainer()
        trainer._previous_epoch_il_accuracy = 0.5
        trainer._previous_epoch_target_better_ratio = 0.0
        self.assertAlmostEqual(trainer._effective_near_miss_rate(), 0.01)

    def test_mature_target_uses_five_percent(self) -> None:
        trainer = self._trainer()
        trainer._previous_epoch_il_accuracy = 0.90
        trainer._previous_epoch_target_better_ratio = 0.25
        self.assertAlmostEqual(trainer._effective_near_miss_rate(), 0.05)


if __name__ == "__main__":
    unittest.main()
