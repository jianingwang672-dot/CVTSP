"""Continue the progressive-recipe run from E133 for diversity experiments.

Every branch restores the full E133 state of outputs/progressive_recipe_20261020
(model, optimizer, scheduler, Reference, Edge Predictor, replay, RNG) and keeps
its resolved configuration and LR schedule, changing only:

  ctrl      nothing
  multi_c0     RL Target bound = route-level minimum over the C0 tables of every
            exactly solved route (rl_multi_c0_bound)
  il_mass_1p0  IL mass target 1.44 -> 1.0 samples per instance: improved Experts
            are no longer repeated, weakening imitation (recipe_il_mass_target)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ProgressiveRecipeTrainer import ProgressiveRecipeTrainer  # noqa: E402

RUN = ROOT / "outputs/progressive_recipe_20261020"
OUT = ROOT / "outputs/recipe_diversity_20261022"
CHANGES = {
    "ctrl": {},
    "multi_c0": {"rl_multi_c0_bound": True},
    "il_mass_1p0": {"recipe_il_mass_target": 1.0},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=sorted(CHANGES))
    parser.add_argument("--source-epoch", type=int, default=133)
    parser.add_argument("--end-epoch", type=int, default=163)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source = RUN / f"train/checkpoint-{args.source_epoch}.pt"
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if int(checkpoint["epoch"]) != args.source_epoch:
        raise RuntimeError("unexpected source checkpoint epoch")
    source_lr = float(checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"])
    del checkpoint
    config = json.loads((RUN / "resolved_config.json").read_text())
    params = config["trainer_params"]
    branch = OUT / args.mode
    branch.mkdir(parents=True, exist_ok=True)
    params.update(CHANGES[args.mode])
    params.update({
        "use_cuda": not args.preflight,
        "epochs": args.end_epoch,
        "strict_resume": False,
        "resume_extra_epochs": False,
        "reset_main_scheduler_on_resume": False,
        "checkpoint_path": str(source),
        "self_improvement_parallel_workers": 0 if args.preflight else args.workers,
        "result_folder": str(branch / ("preflight" if args.preflight else "train")),
    })
    audit = {
        "mode": args.mode, "source_checkpoint": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "start_epoch": args.source_epoch + 1, "end_epoch": args.end_epoch,
        "source_optimizer_lr": source_lr, "changes": CHANGES[args.mode],
        "base_config": str(RUN / "resolved_config.json"), "workers": args.workers,
        "git_tag_of_base_recipe": "recipe-e133-20261022",
    }
    (branch / ("preflight_audit.json" if args.preflight else "audit.json")).write_text(
        json.dumps(audit, indent=2) + "\n")
    trainer = ProgressiveRecipeTrainer(
        config["env_params"], config["model_params"], config["optimizer_params"], params)
    try:
        lr = float(trainer.optimizer.param_groups[0]["lr"])
        if trainer.start_epoch != args.source_epoch + 1 or abs(lr - source_lr) > 1e-12:
            raise RuntimeError(f"continuity failed: start={trainer.start_epoch} lr={lr} source={source_lr}")
        print(f"BRANCH_READY mode={args.mode} E{trainer.start_epoch}->E{args.end_epoch} lr={lr:.6g} "
              f"replay={len(trainer._expert_replay_pool)} multi_c0={trainer.rl_multi_c0_bound}", flush=True)
        if not args.preflight:
            trainer.run()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
