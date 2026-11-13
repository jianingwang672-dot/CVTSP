from __future__ import annotations

import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

from src.JointSelfImprovement import hybrid_lkh_start_indices
from src.ProgressiveRecipeTrainer import ProgressiveRecipeTrainer


def _trainer():
    trainer = object.__new__(ProgressiveRecipeTrainer)
    trainer.recipe_j_threshold = 60
    trainer.recipe_ramp_start = 40
    trainer.recipe_ramp_end = 100
    trainer.recipe_lkh_max_fraction = 0.5
    trainer.recipe_refresh_start = 60
    trainer.recipe_refresh_rounds = 2
    trainer.recipe_il_mass_target = 1.44
    trainer.changed_expert_il_weight = 5
    trainer.unchanged_expert_il_weight = 1
    trainer.trainer_params = {"seed": 7}
    trainer._recipe_epoch = 1
    trainer._recipe_stats = defaultdict(float)
    return trainer


class RecipeKnobTest(unittest.TestCase):
    def test_small_j_keeps_s500_search(self) -> None:
        trainer = _trainer()
        for epoch in (1, 70, 200):
            self.assertEqual(trainer.recipe_knobs(40, epoch, 0.03),
                             {"lkh_fraction": 0.0, "near_miss_rate": 0.03, "refresh_rounds": 0})

    def test_large_j_ramps_with_epoch(self) -> None:
        trainer = _trainer()
        early = trainer.recipe_knobs(80, 40, 0.03)
        mid = trainer.recipe_knobs(80, 70, 0.03)
        late = trainer.recipe_knobs(80, 150, 0.03)
        self.assertEqual(early, {"lkh_fraction": 0.0, "near_miss_rate": 0.03, "refresh_rounds": 0})
        self.assertAlmostEqual(mid["lkh_fraction"], 0.25)
        self.assertAlmostEqual(mid["near_miss_rate"], 0.03 + 0.5 * 0.97)
        self.assertEqual(mid["refresh_rounds"], 2)
        self.assertEqual(trainer.recipe_knobs(80, 60, 0.03)["refresh_rounds"], 0)
        self.assertEqual(late, {"lkh_fraction": 0.5, "near_miss_rate": 1.0, "refresh_rounds": 2})


class RecipeIlMassTest(unittest.TestCase):
    def test_weight_keeps_il_mass_at_target(self) -> None:
        trainer = _trainer()
        instances = [SimpleNamespace(J=20)] * 100 + [SimpleNamespace(J=100)] * 100
        flags = [k < 11 for k in range(100)] + [k < 59 for k in range(100)]
        weights = trainer.recipe_il_weights(instances, flags)
        self.assertAlmostEqual(weights[20], 5.0)  # (1.44 - 0.89) / 0.11
        self.assertAlmostEqual(weights[100], (1.44 - 0.41) / 0.59)
        self.assertEqual(trainer.recipe_il_weights([SimpleNamespace(J=30)], [False])[30], 5.0)
        self.assertEqual(trainer.recipe_il_weights([SimpleNamespace(J=30)] * 2, [True, True])[30], 1.44)

    def test_expanded_il_set_matches_target_mass(self) -> None:
        trainer = _trainer()
        instances = [SimpleNamespace(J=100) for _ in range(1000)]
        flags = [k % 100 < 59 for k in range(1000)]
        captured = {}

        def parent(self, inst, experts, changed):
            captured["n"] = len(inst)
            captured["changed"] = list(changed)
            captured["unchanged_weight"] = self.unchanged_expert_il_weight
            return 0.0, 0.0, len(inst), 0

        with patch("src.JointSelfImprovementTrainer.JointSelfImprovementTrainer._imitation_update", parent):
            trainer._imitation_update(instances, [[0, 1]] * 1000, flags)
        self.assertAlmostEqual(captured["n"] / 1000, 1.44, delta=0.05)
        self.assertFalse(any(captured["changed"]))
        self.assertEqual(captured["unchanged_weight"], 1)
        self.assertEqual(trainer.unchanged_expert_il_weight, 1)


class HybridIndexTest(unittest.TestCase):
    def test_interleaves_quality_and_diversity(self) -> None:
        self.assertEqual(hybrid_lkh_start_indices(30, 20, 10, 0.0), [])
        self.assertEqual(hybrid_lkh_start_indices(30, 20, 10, 0.2), [0, 1, 2, 20, 21, 22])
        self.assertEqual(len(hybrid_lkh_start_indices(30, 20, 10, 0.5)), 15)
        self.assertEqual(hybrid_lkh_start_indices(30, 20, 10, 1.0), list(range(30)))


if __name__ == "__main__":
    unittest.main()


class FixedStartTwoOptTest(unittest.TestCase):
    def test_start_is_kept_and_value_matches_matrix(self) -> None:
        import numpy as np
        from src.JointSelfImprovement import directed_route_upper_bound, fixed_start_two_opt
        rng = np.random.default_rng(0)
        for n in (3, 6, 12):
            matrix = rng.uniform(1, 10, size=(n + 1, n + 1))
            np.fill_diagonal(matrix, 0)
            route = list(rng.permutation(n))
            before = directed_route_upper_bound(route, matrix)
            improved, value = fixed_start_two_opt(matrix, route, max_moves=50, acceptance_epsilon=1e-9)
            self.assertEqual(improved[0], route[0])
            self.assertEqual(sorted(improved), list(range(n)))
            self.assertAlmostEqual(value, directed_route_upper_bound(improved, matrix))
            self.assertLessEqual(value, before + 1e-12)
            # no improving fixed-start reversal remains
            from itertools import combinations
            for i, k in combinations(range(1, n), 2):
                cand = improved[:i] + improved[i:k + 1][::-1] + improved[k + 1:]
                self.assertGreaterEqual(directed_route_upper_bound(cand, matrix), value - 1e-9)


class PerStartLabelFlowTest(unittest.TestCase):
    def test_labels_are_appended_after_normalisation_and_cleared(self) -> None:
        from src.PerStartILTrainer import PerStartILTrainer
        trainer = object.__new__(PerStartILTrainer)
        for k, v in vars(_trainer()).items():
            setattr(trainer, k, v)
        extra = SimpleNamespace(J=40)
        trainer._per_start_labels = [(extra, [1, 0, 2])]
        captured = {}

        def parent(self, inst, experts, changed):
            captured["inst"], captured["experts"] = list(inst), list(experts)
            return 0.0, 0.0, len(inst), 0

        instances = [SimpleNamespace(J=40) for _ in range(10)]
        with patch("src.JointSelfImprovementTrainer.JointSelfImprovementTrainer._imitation_update", parent):
            trainer._imitation_update(instances, [[0, 1, 2]] * 10, [False] * 10)
        self.assertEqual(len(captured["inst"]), 11)
        self.assertIs(captured["inst"][-1], extra)
        self.assertEqual(captured["experts"][-1], [1, 0, 2])
        self.assertEqual(trainer._per_start_labels, [])
