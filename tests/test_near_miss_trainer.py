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

        self.assertEqual(
            {size: trainer._route_start_top_k(size) for size in (20, 21, 60, 100)},
            {20: 3, 21: 4, 60: 9, 100: 15},
        )

    def test_zero_ratio_uses_fixed_fallback(self) -> None:
        trainer = object.__new__(JointSelfImprovementTrainer)
        trainer.si_route_start_top_k = 4
        trainer.si_route_start_top_k_ratio = 0.0
        self.assertEqual(trainer._route_start_top_k(100), 4)


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


if __name__ == "__main__":
    unittest.main()
