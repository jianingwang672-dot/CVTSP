from __future__ import annotations

import copy
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

from src.E500TrainingProfile import apply_e500_training_profile
from src.SearchDiagnostics import SearchAccumulator, move_budget
from src.SearchAblationTrainer import SearchAblationTrainer
from src.JointSelfImprovementTrainer import JointSelfImprovementTrainer
from train_joint_self_improvement import build_parser


class SearchAblationTests(unittest.TestCase):
    def test_statistics_preserve_route_and_count_both_neighborhoods(self):
        from tests.test_certified_upper_bound import CertifiedUpperBoundUnitTest
        from src.CertifiedLocalSearch import FixedAnchorLocalSearchConfig, run_fixed_anchor_local_search
        fixture = CertifiedUpperBoundUnitTest()
        fixture.setUp()
        majorizer = fixture._asymmetric_majorizer()
        for types in (("swap",), ("two_opt",), ("swap", "two_opt")):
            config = FixedAnchorLocalSearchConfig(max_local_moves=30, move_types=types)
            chain = run_fixed_anchor_local_search(majorizer, [0, 1, 2, 3], config)
            before = chain.to_dict()
            result = SimpleNamespace(local_search_results=[chain], top_k_start_sequences=[[0, 1, 2, 3]],
                                     quality_start_count=1, diversity_start_count=0)
            accumulator = SearchAccumulator(30, types)
            accumulator.add(fixture.instance, result)
            group = accumulator.summary()["J4_quality"]
            self.assertEqual(sum(group["evaluated"].values()), chain.evaluated)
            self.assertEqual(sum(group["accepted"].values()), chain.accepted)
            self.assertAlmostEqual(sum(group["upper_bound_gain"].values()), chain.upper_bound_improvement)
            self.assertEqual(before, chain.to_dict())

    def test_diversity_classification_excludes_quality_fallback(self):
        chain = SimpleNamespace(accepted=0, termination_reason="no_upper_bound_improving_move",
                                runtime=0., history=[], evaluated=190, generated=190, dual_pruned=0)
        result = SimpleNamespace(local_search_results=[chain] * 30,
                                 top_k_start_sequences=[list(range(20))] * 30,
                                 quality_start_count=27, diversity_start_count=3)
        accumulator = SearchAccumulator(30, ("swap",))
        accumulator.add(SimpleNamespace(J=20), result)
        groups = accumulator.summary()
        self.assertEqual(groups["J20_quality"]["chains"], 27)
        self.assertEqual(groups["J20_diversity"]["chains"], 3)

    def test_move_schedule_preserves_100_epoch_horizon_and_clamp(self):
        values = [move_budget(e) for e in range(1, 121)]
        self.assertEqual(values[0], 30)
        self.assertEqual(values[19], 26)
        self.assertEqual(values[99], 10)
        self.assertEqual(values[-1], 10)
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))
        for e in range(1, 100):
            self.assertLessEqual(abs(values[e] - values[e-1]), 1)

    def test_extending_execution_cap_keeps_strict_trajectory_signature(self):
        trainer = object.__new__(SearchAblationTrainer)
        trainer.trainer_params = {"move_schedule_horizon": 100}
        trainer.epochs = 20
        with patch.object(JointSelfImprovementTrainer, "_strict_resume_signature",
                          side_effect=lambda: {"planned_epochs": trainer.epochs}):
            short = trainer._strict_resume_signature()
            trainer.epochs = 100
            self.assertEqual(short, trainer._strict_resume_signature())
            self.assertEqual(short["planned_epochs"], 100)

    def test_additional_gradient_observation_does_not_change_update(self):
        torch.manual_seed(12)
        model = torch.nn.Linear(3, 2)
        observed = copy.deepcopy(model)
        baseline_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        observed_optimizer = torch.optim.Adam(observed.parameters(), lr=1e-4)
        x = torch.randn(5, 3)
        trainer = object.__new__(SearchAblationTrainer)
        trainer.model = observed
        trainer.device = torch.device("cpu")
        trainer.grad_clip = 1.
        trainer.global_step = 0
        trainer._phase = "rl"
        trainer._gradient_rows = []
        trainer._component_rows = []
        trainer._joint_batch_index = 0
        trainer._diag_epoch = 1
        trainer._diagnostic_epochs = {1}
        output = model(x)
        base_hybrid, base_exact = output.square().mean(), output.mean()
        (base_hybrid + .7 * base_exact).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        baseline_optimizer.step()
        observed_output = observed(x)
        hybrid, exact = observed_output.square().mean(), observed_output.mean()
        trainer._observe_rl_losses(hybrid, exact, .7, torch.randn(2, 3), np.asarray([.2, -.1]))
        self.assertTrue(all(p.grad is None for p in observed.parameters()))
        (hybrid + .7 * exact).backward()
        trainer._clip_main_gradients()
        observed_optimizer.step()
        for a, b in zip(model.parameters(), observed.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(len(trainer._gradient_rows), 1)
        self.assertIn("cosine", trainer._component_rows[0])

    def test_evaluation_context_restores_rng_model_and_environment_on_error(self):
        trainer = object.__new__(SearchAblationTrainer)
        trainer.model = torch.nn.Linear(3, 2)
        trainer.env = object()
        trainer.device = torch.device("cpu")
        trainer._record_search = True
        trainer._active_near_miss_rate = .05
        state = torch.get_rng_state().clone()
        restored = []
        trainer._capture_rng_state = lambda: state.clone()
        def restore(s):
            torch.set_rng_state(s)
            restored.append(True)
        trainer._restore_rng_state = restore
        original_model, original_env = trainer.model, trainer.env
        with patch("src.SearchAblationTrainer.clone_reference_policy", side_effect=lambda m, **k: copy.deepcopy(m)):
            with self.assertRaisesRegex(RuntimeError, "probe"):
                with trainer._isolated_evaluation(9):
                    torch.rand(20)
                    trainer._active_near_miss_rate = 0.
                    raise RuntimeError("probe")
        self.assertIs(trainer.model, original_model)
        self.assertIs(trainer.env, original_env)
        self.assertTrue(trainer._record_search)
        self.assertEqual(trainer._active_near_miss_rate, .05)
        torch.testing.assert_close(torch.get_rng_state(), state)
        self.assertTrue(restored)


if __name__ == "__main__":
    unittest.main()
