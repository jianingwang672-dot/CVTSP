import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import unittest

from src.JointSelfImprovementTrainer import JointSelfImprovementTrainer
from src.NearMissQuotaTrainer import NearMissQuotaTrainer
from src.SearchAblationTrainer import SearchAblationTrainer


class NearMissQuotaTests(unittest.TestCase):
    def test_fixed_rate_is_independent_of_training_competence(self):
        for rate in (.05, .10, .20):
            trainer = object.__new__(NearMissQuotaTrainer)
            trainer.si_adaptive_near_miss = True
            trainer.si_near_miss_min_rate = trainer.si_near_miss_rate = rate
            trainer.competence_il_start, trainer.competence_il_full = .5, .85
            trainer.competence_target_better_full = .2
            for accuracy, win in ((0., 0.), (.6, .1), (.9, .8)):
                trainer._previous_epoch_il_accuracy = accuracy
                trainer._previous_epoch_target_better_ratio = win
                self.assertEqual(trainer._effective_near_miss_rate(), rate)

    def test_existing_rounding_and_gap_order_are_preserved(self):
        results = [SimpleNamespace(near_miss_candidate_sequence=[1, 0], anchor_sequence=[0, 1],
                                   near_miss_relative_gap=float(63 - i), expert_changed=False,
                                   expert_gurobi_calls=0) for i in range(64)]
        for rate, limit in ((.05, 4), (.10, 7), (.20, 13)):
            eligible, selected = JointSelfImprovementTrainer._select_near_miss_indices(results, batch_size=64, rate=rate)
            self.assertEqual(len(eligible), 64)
            self.assertEqual(selected, list(range(63, 63 - limit, -1)))
        results[63].expert_changed = True
        _, selected = JointSelfImprovementTrainer._select_near_miss_indices(results, batch_size=64, rate=.05)
        self.assertEqual(selected, [62, 61, 60, 59])

    def test_batch_audit_rejects_an_unexecuted_quota(self):
        trainer = object.__new__(NearMissQuotaTrainer)
        trainer.trainer_params = {"near_miss_ablation_rate": .10}
        trainer._active_near_miss_rate = .10
        trainer._joint_batch_index, trainer._diag_epoch = 1, 1
        with tempfile.TemporaryDirectory() as directory:
            trainer.output_dir = Path(directory)
            (trainer.output_dir / "diagnostics").mkdir()
            correct = {"near_miss_attempt_count": 7, "near_miss_eligible_count": 20, "near_miss_accept_count": 5}
            with patch.object(SearchAblationTrainer, "_train_one_batch", return_value=correct):
                result = trainer._train_one_batch(64)
            self.assertIs(result, correct)
            row = json.loads((trainer.output_dir / "diagnostics/epoch-001_quota_batches.jsonl").read_text())
            self.assertEqual(row["quota_limit"], 7)
            self.assertEqual(row["accepted"], 5)
            incorrect = dict(correct, near_miss_attempt_count=4)
            with patch.object(SearchAblationTrainer, "_train_one_batch", return_value=incorrect):
                with self.assertRaisesRegex(RuntimeError, "quota"):
                    trainer._train_one_batch(64)


if __name__ == "__main__": unittest.main()
