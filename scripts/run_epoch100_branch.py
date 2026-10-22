"""Controlled E100->E120 continuation for one-factor IL or LR experiments."""

from __future__ import annotations

import argparse
import copy
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
BRANCH_ROOT = ROOT / "outputs/epoch100_controlled_branches"
END_EPOCH = 120
LR_END_EPOCH = 150
MIN_LR = 1e-7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(mode: str, *, preflight: bool):
    params = json.loads((TRAIN_DIR / "params.json").read_text())
    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=False)
    if checkpoint["epoch"] != 100:
        raise ValueError("expected an exact Epoch 100 checkpoint")
    source_lr = float(checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"])
    saved_lr = float(checkpoint["scheduler_state_dict"]["_last_lr"][0])
    if abs(source_lr - saved_lr) > 1e-12:
        raise ValueError("source optimizer and scheduler learning rates differ")
    branch_dir = BRANCH_ROOT / mode
    branch_dir.mkdir(parents=True, exist_ok=True)
    env_params = params["env_params"]
    model_params = params["model_params"]
    optimizer_params = params["optimizer_params"]
    trainer_params = params["trainer_params"]
    trainer_params.update({
        "use_cuda": not preflight,
        "cuda_device_num": 0,
        "epochs": END_EPOCH,
        "resume_extra_epochs": False,
        "reset_main_scheduler_on_resume": False,
        # An intentional single-parameter experiment cannot pass the original
        # full-trajectory strict signature; we audit all resumed states below.
        "strict_resume": False,
        "result_folder": str(branch_dir / ("preflight" if preflight else "train")),
        "self_improvement_parallel_workers": 0 if preflight else 128,
        "checkpoint_path": str(SOURCE),
    })
    changed_parameter = {}
    if mode == "il2":
        original = int(trainer_params["imitation_inner_epochs"])
        if original != 5:
            raise ValueError(f"expected original IL inner epochs 5, got {original}")
        trainer_params["imitation_inner_epochs"] = 2
        changed_parameter = {"name": "imitation_inner_epochs", "old": 5, "new": 2}
    elif mode == "lr150":
        if optimizer_params["scheduler"] != checkpoint["main_scheduler_config"]:
            raise ValueError("original scheduler config mismatch")
        continuation = {
            "name": "cosine_continuation",
            "start_epoch": 100,
            "end_epoch": LR_END_EPOCH,
            "start_lr": source_lr,
            "min_lr": MIN_LR,
        }
        optimizer_params["scheduler"] = continuation
        branch_checkpoint = copy.copy(checkpoint)
        branch_checkpoint["main_scheduler_config"] = copy.deepcopy(continuation)
        branch_checkpoint["scheduler_state_dict"] = copy.deepcopy(checkpoint["scheduler_state_dict"])
        branch_checkpoint["scheduler_state_dict"]["_last_lr"] = [source_lr]
        # Only scheduler metadata is rewritten. Model, optimizer moments,
        # Reference, Edge Predictor, replay, and RNG remain in the source state.
        branch_source = branch_dir / "epoch100_scheduler_branch.pt"
        if branch_source.exists():
            existing = torch.load(branch_source, map_location="cpu", weights_only=False)
            if existing["main_scheduler_config"] != continuation or existing["epoch"] != 100:
                raise RuntimeError("existing branch checkpoint has a different scheduler")
        else:
            torch.save(branch_checkpoint, branch_source)
        trainer_params["checkpoint_path"] = str(branch_source)
        changed_parameter = {
            "name": "main_scheduler",
            "old": checkpoint["main_scheduler_config"],
            "new": continuation,
        }
    else:
        raise ValueError(f"unknown branch mode: {mode}")
    audit = {
        "source_checkpoint": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "source_epoch": 100,
        "start_epoch": 101,
        "end_epoch": END_EPOCH,
        "source_optimizer_lr": source_lr,
        "changed_parameter": changed_parameter,
        "all_other_training_parameters": "loaded from source params.json",
        "resumption": "all E100 model/optimizer/Reference/Edge/replay/RNG states restored; strict signature intentionally relaxed for single-factor branch",
        "cpu_preflight": preflight,
    }
    (branch_dir / ("preflight_audit.json" if preflight else "audit.json")).write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    return env_params, model_params, optimizer_params, trainer_params, source_lr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["il2", "lr150"])
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    env, model, optimizer, trainer_params, source_lr = prepare(args.mode, preflight=args.preflight)
    trainer = JointSelfImprovementTrainer(env, model, optimizer, trainer_params)
    try:
        actual = float(trainer.optimizer.param_groups[0]["lr"])
        scheduled = float(trainer.scheduler.get_last_lr()[0])
        if trainer.start_epoch != 101 or abs(actual - source_lr) > 1e-12 or abs(scheduled - source_lr) > 1e-12:
            raise RuntimeError(
                f"E100 state continuity failed: start={trainer.start_epoch}, "
                f"optimizer_lr={actual}, scheduler_lr={scheduled}, source_lr={source_lr}"
            )
        print(
            f"BRANCH_READY mode={args.mode} E{trainer.start_epoch}->E{END_EPOCH} "
            f"lr={actual:.12g} replay={len(trainer._expert_replay_pool)} "
            f"IL_inner={trainer.imitation_inner_epochs}", flush=True
        )
        if not args.preflight:
            trainer.run()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
