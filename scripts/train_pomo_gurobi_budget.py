"""Equal-budget POMO + exact Gurobi reward baseline.

Trains the G200 recipe from scratch (outputs/train__cvtsp_random_n100_nearest_quarter:
J ~ U[20, 100], nearest-depot POMO starts with J // 4 starts, 100k instances per
epoch, batch 512, Adam lr 5e-5, wd 1e-6, embedding 128 / 6 layers / 8 heads,
seed 1234) and saves checkpoints the moment cumulative training budgets reach
those of the proposed method's selected model E133:

  * exact SOCP solves  3,839,970  (E133 training solves, validation excluded)
  * training instances 1,330,000

Every batch draws fresh instances, so the trainer's RewardCache never hits
(the first run logged 0 hits in 13.2M solves) yet kept every SPSolution and
grew the main process to 26 GB, OOM-killing it in E10. The cache is emptied
after each batch; rewards and training are unchanged.

--checkpoint-hours H also saves time-XXXh.pt every H hours of wall time
since run() started (time_checkpoints.json records the budgets at each), so the
same run gives both ablation A (instance/solve budgets) and the POMO curve of
experiment 5a (validation makespan vs training wall time); the queue stops it
with a timeout just after the 24 h checkpoint.

G200's own log shows ~15.2 solves per instance at mean J ~ 61, i.e. J // 4
starts, which is why pomo_divisor = 4 here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

from src.TSPTrainer import OnlineTSPTrainer  # noqa: E402

E133_SOLVES = 3_839_970
E133_INSTANCES = 1_330_000


class BudgetTrainer(OnlineTSPTrainer):
    def __init__(self, *args, budgets: dict[str, tuple[str, int]], checkpoint_hours: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.budgets = dict(budgets)  # name -> (counter, threshold)
        self.counters = {"solves": 0, "instances": 0}
        self.budget_log = []
        self._epoch = 0
        self.checkpoint_seconds = 3600.0 * checkpoint_hours
        self.time_log = []
        self._started = None

    def run(self):
        self._started = time.perf_counter()
        return super().run()

    def _train_one_epoch(self, epoch: int):
        self._epoch = epoch
        return super()._train_one_epoch(epoch)

    def _train_one_batch(self, batch_size: int):
        out = super()._train_one_batch(batch_size)
        self.cache._cache.clear()  # never hits on fresh instances; see module docstring
        self.counters["solves"] += int(out[4])
        self.counters["instances"] += int(batch_size)
        for name, (counter, threshold) in list(self.budgets.items()):
            if self.counters[counter] >= threshold:
                path = self.output_dir / f"budget-{name}.pt"
                metrics = {"epoch_in_progress": self._epoch, **self.counters, "threshold": threshold}
                self._save_checkpoint(self._epoch, metrics, path)
                self.budget_log.append({"name": name, "path": str(path), **metrics})
                (self.output_dir / "budget_checkpoints.json").write_text(json.dumps(self.budget_log, indent=2))
                logger.info("Saved {} at {} (epoch {} in progress)", path.name, self.counters, self._epoch)
                del self.budgets[name]
        if self.checkpoint_seconds > 0:
            elapsed = time.perf_counter() - self._started
            due = (len(self.time_log) + 1) * self.checkpoint_seconds
            if elapsed >= due:
                hours = due / 3600.0
                path = self.output_dir / f"time-{hours:06.2f}h.pt"
                metrics = {"epoch_in_progress": self._epoch, **self.counters, "wall_seconds": elapsed}
                self._save_checkpoint(self._epoch, metrics, path)
                self.time_log.append({"hours": hours, "path": str(path), **metrics})
                (self.output_dir / "time_checkpoints.json").write_text(json.dumps(self.time_log, indent=2))
                logger.info("Saved {} at {:.0f}s {}", path.name, elapsed, self.counters)
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/pomo_gurobi_equal_budget_20261027")
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--train-episodes", type=int, default=100_000)
    parser.add_argument("--checkpoint-hours", type=float, default=0.0, help="also save a checkpoint every H wall-clock hours")
    parser.add_argument("--smoke", action="store_true", help="one 512-instance epoch to check solves per instance")
    args = parser.parse_args()
    env_params = {
        "dataset_dir": "instance/Data", "min_problem_size": 20, "max_problem_size": 100,
        "problem_sizes": None, "problem_size": 100, "pomo_size": 100, "pomo_divisor": 4,
        "start_node_strategy": "nearest_depot",
    }
    model_params = {"embedding_dim": 128, "encoder_layer_num": 6, "qkv_dim": 16, "head_num": 8,
                    "logit_clipping": 10, "ff_hidden_dim": 512}
    optimizer_params = {"optimizer": {"lr": 5e-5, "weight_decay": 1e-6},
                        "scheduler": {"milestones": [501], "gamma": 0.1}}
    trainer_params = {
        "use_cuda": True, "cuda_device_num": 0, "epochs": 1 if args.smoke else args.epochs,
        "resume_extra_epochs": False, "train_episodes": 512 if args.smoke else args.train_episodes,
        "train_batch_size": 512, "checkpoint_interval": 1, "solver_backend": "gurobi", "gurobi_threads": 1,
        "cvxpylayer_rebar_loss_enable": False, "cvxpylayer_aux_enable": False, "penalty_reward": -1e6,
        "grad_clip": 1.0, "seed": 1234, "result_folder": str(args.output_dir / ("smoke" if args.smoke else "train")),
        "log_level": "INFO", "progress_log_percent": 1.0, "progress_bar_width": 24,
        "reward_backend": "persistent_pool", "reward_parallel_workers": args.workers,
        "parallel_solver_threads": 1, "reward_parallel_chunksize": 1, "checkpoint_path": None,
    }
    budgets = {"equal_solves": ("solves", E133_SOLVES), "equal_instances": ("instances", E133_INSTANCES)}
    trainer = BudgetTrainer(env_params, model_params, optimizer_params, trainer_params, budgets=budgets,
                            checkpoint_hours=args.checkpoint_hours)
    trainer.run()
    print(f"COUNTERS {trainer.counters} solves/instance={trainer.counters['solves'] / max(trainer.counters['instances'], 1):.2f}")


if __name__ == "__main__":
    main()
