from __future__ import annotations

import unittest

from src.E500TrainingProfile import apply_e500_training_profile
from src.JointSelfImprovementTrainer import JointSelfImprovementTrainer
from train_joint_self_improvement import build_parser


class E500TrainingProfileTest(unittest.TestCase):
    def test_restores_fixed_search_and_il_without_changing_budget_or_data(self):
        args = build_parser().parse_args([
            "--epochs", "100", "--train-instances", "20000",
            "--top-k-ratio", "0.5", "--top-k-min", "30",
            "--imitation-inner-epochs-schedule", "1:5,51:2,101:1",
            "--expert-replay-fraction", "0.10", "--expert-replay-capacity", "20000",
            "--reference-validation-per-size", "400", "--validation-dir", "custom-validation",
        ])
        apply_e500_training_profile(args)
        trainer = object.__new__(JointSelfImprovementTrainer)
        trainer.si_route_start_top_k = args.top_k
        trainer.si_route_start_top_k_ratio = args.top_k_ratio
        trainer.si_route_start_top_k_min = args.top_k_min
        trainer.imitation_inner_epochs_schedule = args.imitation_inner_epochs_schedule
        for size in (20, 40, 60, 80, 100):
            self.assertEqual(trainer._route_start_top_k(size), 30)
        for epoch in (1, 50, 51, 100):
            self.assertEqual(trainer._imitation_inner_epochs_for_epoch(epoch), 5)
        self.assertEqual(args.expert_replay_fraction, 0.30)
        self.assertEqual(args.expert_replay_capacity, 50000)
        self.assertEqual(args.top_k_diversity_count, 10)
        self.assertEqual(args.top_k_diversity_pool_size, 60)
        self.assertEqual(args.top_k_diversity_max_relative_gap, 0.05)
        self.assertEqual(args.rl_reward_mode, "hybrid_pomo_exact_expert")
        self.assertEqual(args.epochs, 100)
        self.assertEqual(args.train_instances, 20000)
        self.assertEqual(args.reference_validation_per_size, 400)
        self.assertEqual(args.validation_dir, "custom-validation")
        self.assertEqual(args.checkpoint_path, "")
        self.assertEqual(args.reference_validation_candidates, 1)

    def test_e500_profile_allows_explicit_full_pomo_validation(self):
        args = build_parser().parse_args(["--reference-validation-candidates", "0"])
        apply_e500_training_profile(args)
        self.assertEqual(args.reference_validation_candidates, 0)


if __name__ == "__main__":
    unittest.main()
