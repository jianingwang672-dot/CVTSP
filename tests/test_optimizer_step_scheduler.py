from __future__ import annotations

import math
import unittest

import torch

from src.TSPTrainer import _build_scheduler, _scheduler_interval


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


if __name__ == "__main__":
    unittest.main()
