from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.JointSelfImprovementTrainer import JointSelfImprovementTrainer


class ExpertReplayTest(unittest.TestCase):
    def test_il_micro_batches_preserve_one_logical_optimizer_step(self) -> None:
        trainer = object.__new__(JointSelfImprovementTrainer)
        trainer.model = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(trainer.model.weight)
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1.0)
        trainer.scheduler = None
        trainer.scheduler_interval = "epoch"
        trainer.global_step = 0
        trainer.device = torch.device("cpu")
        trainer.grad_clip = 10.0
        trainer.changed_expert_il_weight = 1
        trainer.expert_replay_fraction = 0.0
        trainer.imitation_inner_epochs = 1
        trainer.imitation_batch_size = 5
        trainer.imitation_micro_batch_size = 2
        trainer._expert_replay_pool = []
        current_instances = [SimpleNamespace(J=2) for _ in range(5)]

        calls: list[int] = []

        def fake_teacher_forcing(model, instances, experts, *, device):
            calls.append(len(instances))
            loss = next(model.parameters()).sum()
            token_count = len(instances) * 2
            return loss, token_count, None, token_count

        with patch(
            "src.JointSelfImprovementTrainer.teacher_forcing_loss",
            side_effect=fake_teacher_forcing,
        ):
            trainer._imitation_update(
                current_instances,
                [[0, 1] for _ in current_instances],
                [False for _ in current_instances],
            )

        self.assertEqual(calls, [2, 2, 1])
        self.assertAlmostEqual(float(trainer.model.weight.item()), -1.0)
        self.assertEqual(trainer.global_step, 1)

    def test_il_mix_is_seventy_current_thirty_replay(self) -> None:
        trainer = object.__new__(JointSelfImprovementTrainer)
        trainer.model = torch.nn.Linear(1, 1)
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.0)
        trainer.scheduler = None
        trainer.scheduler_interval = "epoch"
        trainer.global_step = 0
        trainer.device = torch.device("cpu")
        trainer.grad_clip = 1.0
        trainer.changed_expert_il_weight = 1
        trainer.expert_replay_fraction = 0.3
        trainer.imitation_inner_epochs = 1
        trainer.imitation_batch_size = 100
        replay_instance = SimpleNamespace(J=2)
        trainer._expert_replay_pool = [(replay_instance, [0, 1])]
        current_instances = [SimpleNamespace(J=2) for _ in range(7)]

        def fake_teacher_forcing(model, instances, experts, *, device):
            loss = next(model.parameters()).sum() * 0.0 + 1.0
            return loss, len(instances), None, len(instances)

        with patch(
            "src.JointSelfImprovementTrainer.teacher_forcing_loss",
            side_effect=fake_teacher_forcing,
        ):
            _, _, effective_examples, replay_examples = trainer._imitation_update(
                current_instances,
                [[0, 1] for _ in current_instances],
                [False for _ in current_instances],
            )

        self.assertEqual(effective_examples, 10)
        self.assertEqual(replay_examples, 3)

    def test_pool_keeps_only_recent_changed_experts(self) -> None:
        trainer = object.__new__(JointSelfImprovementTrainer)
        trainer.expert_replay_capacity = 2
        old = SimpleNamespace(J=2)
        trainer._expert_replay_pool = [(old, [0, 1])]
        instances = [SimpleNamespace(J=2) for _ in range(3)]

        added = trainer._add_changed_experts_to_replay(
            instances,
            [[0, 1], [1, 0], [0, 1]],
            [True, False, True],
        )

        self.assertEqual(added, 2)
        self.assertEqual(len(trainer._expert_replay_pool), 2)
        self.assertIs(trainer._expert_replay_pool[0][0], instances[0])
        self.assertIs(trainer._expert_replay_pool[1][0], instances[2])


if __name__ == "__main__":
    unittest.main()
