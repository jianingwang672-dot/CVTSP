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


if __name__ == "__main__":
    unittest.main()
