"""E100 -> E120 continuations of S500 that target the post-E100 plateau.

Every branch restores the complete S500 Epoch-100 state (model, optimizer,
scheduler, Reference, Edge Predictor, replay, adaptive state, RNG) and keeps
the archived S500 configuration, changing only:

  ctrl           nothing (current code; separates code drift from the edits)
  il_changed     unchanged Anchor labels are dropped from IL (weight 0)
  strong_expert  near-miss fixed at 100% plus 2 C0-refresh rounds
  both           il_changed + strong_expert
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch

from src.JointSelfImprovementTrainer import JointSelfImprovementTrainer

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "outputs/train__scratch_e500_lr1e4_min1e7_w128_qdhybrid_replay30_vectorized"
SOURCE = TRAIN_DIR / "checkpoint-100.pt"
BRANCH_ROOT = ROOT / "outputs/e100_plateau_branches_20261019"
MODES = ("ctrl", "il_changed", "strong_expert", "both")

IL_CHANGED = {"unchanged_expert_il_weight": 0}
STRONG_EXPERT = {
    "self_improvement_near_miss_rate": 1.0,
    "self_improvement_near_miss_min_rate": 1.0,
    "self_improvement_adaptive_near_miss": False,
    "self_improvement_c0_refresh_rounds": 2,
}
CHANGES = {
    "ctrl": {},
    "il_changed": IL_CHANGED,
    "strong_expert": STRONG_EXPERT,
    "both": {**IL_CHANGED, **STRONG_EXPERT},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(mode: str, *, end_epoch: int, workers: int, preflight: bool):
    params = json.loads((TRAIN_DIR / "params.json").read_text())
    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=False)
    if int(checkpoint["epoch"]) != 100:
        raise ValueError("expected the exact S500 Epoch 100 checkpoint")
    source_lr = float(checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"])
    branch_dir = BRANCH_ROOT / mode
    branch_dir.mkdir(parents=True, exist_ok=True)
    trainer_params = params["trainer_params"]
    old = {key: trainer_params.get(key) for key in CHANGES[mode]}
    trainer_params.update(CHANGES[mode])
    trainer_params.update({
        "use_cuda": not preflight,
        "cuda_device_num": 0,
        "epochs": end_epoch,
        "resume_extra_epochs": False,
        "reset_main_scheduler_on_resume": False,
        # Intentional single-factor branches cannot match the archived strict
        # trajectory signature; continuity is audited explicitly below.
        "strict_resume": False,
        "checkpoint_path": str(SOURCE),
        "self_improvement_parallel_workers": 0 if preflight else workers,
        "result_folder": str(branch_dir / ("preflight" if preflight else "train")),
    })
    audit = {
        "mode": mode,
        "source_checkpoint": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "start_epoch": 101,
        "end_epoch": end_epoch,
        "source_optimizer_lr": source_lr,
        "changed_parameters": {key: {"old": old[key], "new": value}
                               for key, value in CHANGES[mode].items()},
        "workers": workers,
        "all_other_parameters": "archived S500 params.json (restored E500 configuration)",
        "cpu_preflight": preflight,
    }
    (branch_dir / ("preflight_audit.json" if preflight else "audit.json")).write_text(
        json.dumps(audit, indent=2) + "\n")
    return params["env_params"], params["model_params"], params["optimizer_params"], trainer_params, source_lr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--end-epoch", type=int, default=120)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    env, model, optimizer, trainer_params, source_lr = prepare(
        args.mode, end_epoch=args.end_epoch, workers=args.workers, preflight=args.preflight)
    trainer = JointSelfImprovementTrainer(env, model, optimizer, trainer_params)
    try:
        actual = float(trainer.optimizer.param_groups[0]["lr"])
        if trainer.start_epoch != 101 or abs(actual - source_lr) > 1e-12:
            raise RuntimeError(
                f"E100 continuity failed: start={trainer.start_epoch} lr={actual} source={source_lr}")
        print(
            f"BRANCH_READY mode={args.mode} E{trainer.start_epoch}->E{args.end_epoch} lr={actual:.12g} "
            f"replay={len(trainer._expert_replay_pool)} near_miss={trainer.si_near_miss_min_rate}:"
            f"{trainer.si_near_miss_rate} adaptive={trainer.si_adaptive_near_miss} "
            f"refresh={trainer.si_c0_refresh_rounds} unchanged_il={trainer.unchanged_expert_il_weight}",
            flush=True)
        if not args.preflight:
            trainer.run()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
