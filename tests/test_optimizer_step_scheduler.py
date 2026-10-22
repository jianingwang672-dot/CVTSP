from __future__ import annotations

import math
import unittest

import torch

from src.TSPTrainer import (
    _assert_scheduler_optimizer_lr_match,
    _build_scheduler,
    _require_checkpoint_keys,
    _scheduler_interval,
)


class OptimizerStepSchedulerTest(unittest.TestCase):
    def _make(self):
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.SGD([parameter], lr=2e-5)
        config = {
            "name": "cosine_warmup_steps",
            "total_steps": 20,
            "warmup_steps": 4,
            "min_lr": 1e-6,
        }
        return parameter, optimizer, _build_scheduler(optimizer, config), config

    @staticmethod
    def _step(parameter, optimizer, scheduler) -> None:
        optimizer.zero_grad()
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()

    def test_warmup_cosine_and_floor_are_update_based(self) -> None:
        parameter, optimizer, scheduler, config = self._make()
        self.assertEqual(_scheduler_interval(config), "step")
        self.assertEqual(scheduler.last_epoch, 0)

        rates = [optimizer.param_groups[0]["lr"]]
        for _ in range(25):
            self._step(parameter, optimizer, scheduler)
            rates.append(optimizer.param_groups[0]["lr"])

        self.assertAlmostEqual(rates[3], 2e-5)
        self.assertAlmostEqual(rates[4], 2e-5)
        self.assertAlmostEqual(rates[20], 1e-6)
        self.assertTrue(all(rate >= 1e-6 for rate in rates))

    def test_optimizer_and_scheduler_state_resume_at_same_global_step(self) -> None:
        parameter, optimizer, scheduler, _ = self._make()
        global_step = 0
        for _ in range(7):
            self._step(parameter, optimizer, scheduler)
            global_step += 1

        saved_lr = optimizer.param_groups[0]["lr"]
        optimizer_state = optimizer.state_dict()
        scheduler_state = scheduler.state_dict()

        new_parameter, new_optimizer, new_scheduler, _ = self._make()
        new_optimizer.load_state_dict(optimizer_state)
        new_scheduler.load_state_dict(scheduler_state)

        self.assertEqual(new_scheduler.last_epoch, global_step)
        self.assertTrue(
            math.isclose(new_optimizer.param_groups[0]["lr"], saved_lr, rel_tol=0.0, abs_tol=1e-15)
        )

        self._step(parameter, optimizer, scheduler)
        self._step(new_parameter, new_optimizer, new_scheduler)
        self.assertEqual(new_scheduler.last_epoch, global_step + 1)
        self.assertTrue(
            math.isclose(
                new_optimizer.param_groups[0]["lr"],
                optimizer.param_groups[0]["lr"],
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        )

    def test_strict_resume_accepts_matching_optimizer_scheduler_rates(self) -> None:
        parameter, optimizer, scheduler, _ = self._make()
        for _ in range(7):
            self._step(parameter, optimizer, scheduler)

        _assert_scheduler_optimizer_lr_match(optimizer, scheduler)

    def test_strict_resume_rejects_learning_rate_override(self) -> None:
        parameter, optimizer, scheduler, _ = self._make()
        for _ in range(7):
            self._step(parameter, optimizer, scheduler)
        optimizer.param_groups[0]["lr"] = 1e-4

        with self.assertRaisesRegex(ValueError, "learning-rate mismatch"):
            _assert_scheduler_optimizer_lr_match(optimizer, scheduler)

    def test_strict_resume_rejects_missing_checkpoint_state(self) -> None:
        with self.assertRaisesRegex(KeyError, "optimizer_state_dict"):
            _require_checkpoint_keys(
                {"model_state_dict": {}},
                {"model_state_dict", "optimizer_state_dict"},
                context="test state",
            )

    def test_epoch_100_cosine_branch_keeps_lr_continuous(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.SGD([parameter], lr=1e-4)
        original = _build_scheduler(
            optimizer,
            {"name": "cosine_warmup", "total_epochs": 500,
             "warmup_epochs": 1, "min_lr": 1e-7},
        )
        for _ in range(100):
            self._step(parameter, optimizer, original)
        e100_lr = optimizer.param_groups[0]["lr"]
        saved_optimizer = optimizer.state_dict()
        saved_scheduler = original.state_dict()

        branch_parameter = torch.nn.Parameter(torch.tensor(0.0))
        branch_optimizer = torch.optim.SGD([branch_parameter], lr=1e-4)
        branch = _build_scheduler(
            branch_optimizer,
            {"name": "cosine_continuation", "start_epoch": 100,
             "end_epoch": 150, "start_lr": e100_lr, "min_lr": 1e-7},
        )
        branch_optimizer.load_state_dict(saved_optimizer)
        branch.load_state_dict(saved_scheduler)
        _assert_scheduler_optimizer_lr_match(branch_optimizer, branch)
        self.assertEqual(branch.last_epoch, 100)
        self.assertAlmostEqual(branch_optimizer.param_groups[0]["lr"], e100_lr)
        self._step(branch_parameter, branch_optimizer, branch)
        self.assertLess(branch_optimizer.param_groups[0]["lr"], e100_lr)
        self.assertGreater(branch_optimizer.param_groups[0]["lr"], e100_lr * 0.99)
        for _ in range(49):
            self._step(branch_parameter, branch_optimizer, branch)
        self.assertAlmostEqual(branch_optimizer.param_groups[0]["lr"], 1e-7)


if __name__ == "__main__":
    unittest.main()
