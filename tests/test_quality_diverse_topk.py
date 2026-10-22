from __future__ import annotations

import unittest

import numpy as np

from src.JointSelfImprovement import select_quality_diverse_start_indices


class QualityDiverseTopKTest(unittest.TestCase):
    def setUp(self) -> None:
        self.routes = [
            [0, 1, 2, 3],
            [0, 1, 3, 2],
            [0, 2, 1, 3],
            [1, 0, 2, 3],
            [3, 2, 1, 0],
            [2, 3, 0, 1],
        ]
        self.scores = np.asarray([1.0, 2.0, 3.0, 4.0, 10.0, 11.0])

    def test_zero_diversity_reproduces_stable_score_topk(self) -> None:
        selected, quality_count, diversity_count = select_quality_diverse_start_indices(
            self.routes, self.scores, top_k=4, diversity_count=0
        )
        self.assertEqual(selected.tolist(), [0, 1, 2, 3])
        self.assertEqual(quality_count, 4)
        self.assertEqual(diversity_count, 0)

    def test_reserves_tail_for_routes_far_from_quality_subset(self) -> None:
        selected, quality_count, diversity_count = select_quality_diverse_start_indices(
            self.routes, self.scores, top_k=4, diversity_count=2
        )
        self.assertEqual(selected[:2].tolist(), [0, 1])
        self.assertEqual(set(selected[2:].tolist()), {4, 5})
        self.assertEqual(quality_count, 2)
        self.assertEqual(diversity_count, 2)

    def test_quality_gate_rejects_far_but_bad_routes_and_fills_by_quality(self) -> None:
        selected, quality_count, diversity_count = select_quality_diverse_start_indices(
            self.routes,
            self.scores,
            top_k=4,
            diversity_count=2,
            diversity_pool_size=6,
            diversity_max_relative_gap=0.50,
        )

        # The quality boundary is score 2, so only score 3 remains eligible
        # for diversity. The last slot falls back to the next-best score 4;
        # scores 10 and 11 can no longer enter solely because they are far.
        self.assertEqual(set(selected.tolist()), {0, 1, 2, 3})
        self.assertEqual(quality_count, 3)
        self.assertEqual(diversity_count, 1)

    def test_fewer_routes_than_topk_preserves_configured_quality_core(self) -> None:
        routes = [list(np.roll(np.arange(25), index)) for index in range(25)]
        scores = np.linspace(1.0, 1.04, 25)

        selected, quality_count, diversity_count = select_quality_diverse_start_indices(
            routes,
            scores,
            top_k=30,
            diversity_count=10,
            diversity_pool_size=60,
            diversity_max_relative_gap=0.05,
        )

        self.assertEqual(len(selected), 25)
        self.assertEqual(len(set(selected.tolist())), 25)
        self.assertEqual(quality_count, 20)
        self.assertEqual(diversity_count, 5)

    def test_quality_core_consumes_all_routes_when_at_most_twenty_exist(self) -> None:
        routes = [list(np.roll(np.arange(20), index)) for index in range(20)]
        scores = np.arange(20, dtype=float)

        selected, quality_count, diversity_count = select_quality_diverse_start_indices(
            routes,
            scores,
            top_k=30,
            diversity_count=10,
            diversity_pool_size=60,
            diversity_max_relative_gap=0.05,
        )

        self.assertEqual(selected.tolist(), list(range(20)))
        self.assertEqual(quality_count, 20)
        self.assertEqual(diversity_count, 0)


if __name__ == "__main__":
    unittest.main()
