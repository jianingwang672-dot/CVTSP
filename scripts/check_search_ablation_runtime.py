"""Small, disposable runtime checks; never produces a formal experiment model."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("CVTSP_VECTORIZED_LOCAL_SEARCH", "1")

import torch

from src.SearchAblationTrainer import SearchAblationTrainer
from src.SearchDiagnostics import atomic_json, tensor_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-size", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = json.loads(args.config.read_text())
    env, params = config["env_params"], config["trainer_params"]
    params.update({"epochs": args.epochs, "train_episodes": 128 if args.full_size else 4,
                   "train_batch_size": 64 if args.full_size else 2,
                   "reference_validation_per_size": 1,
                   "reference_validation_seed": 20271012,
                   "reference_validation_dir": str(args.output_dir / "validation"),
                   "result_folder": str(args.output_dir),
                   "diagnostic_dir": "", "diagnostic_epochs": (1,)})
    if args.full_size:
        env["min_problem_size"] = env["max_problem_size"] = env["problem_size"] = 100
    else:
        env["min_problem_size"] = env["max_problem_size"] = env["problem_size"] = 20
        params["self_improvement_parallel_workers"] = 2
    trainer_class = SearchAblationTrainer
    if "near_miss_ablation_rate" in params:
        from src.NearMissQuotaTrainer import NearMissQuotaTrainer
        trainer_class = NearMissQuotaTrainer
    if params.get("quality_weighted_il"):
        from src.QualityILTrainer import QualityILTrainer
        trainer_class = QualityILTrainer
    trainer = trainer_class(env, config["model_params"], config["optimizer_params"], params)
    try:
        # Demonstrate real RNG/cache isolation without a legacy checkpoint.
        from src.FixedValidation import load_or_create_fixed_validation_sets
        groups = load_or_create_fixed_validation_sets(
            dataset_dir=env["dataset_dir"], validation_dir=args.output_dir / "diagnostic_probe",
            problem_sizes=(20,), per_size=1, seed=20281012)
        trainer._diagnostic_groups = groups
        trainer._diag_epoch = 1
        before = trainer._capture_rng_state()
        target_hash = tensor_digest(trainer.model.state_dict())
        reference_hash = tensor_digest(trainer.reference_policy.state_dict())
        original_model, original_env = trainer.model, trainer.env
        trainer._stage_evaluation("probe")
        assert tensor_digest(trainer.model.state_dict()) == target_hash
        assert tensor_digest(trainer.reference_policy.state_dict()) == reference_hash
        assert trainer.model is original_model and trainer.env is original_env
        assert torch.equal(trainer._capture_rng_state()["torch_cpu"], before["torch_cpu"])
        assert trainer.generator.rng.bit_generator.state == before["online_generator_bit_state"]
        trainer._reward_diagnostic()
        # Exercise full-POMO phase path with one J20 instance, independent of epoch budget.
        trainer._diag_epoch = 5
        trainer._diagnostic_epochs = {1, 5}
        trainer._stage_evaluation("after_rl")
        trainer._diagnostic_groups = {}
        trainer._diag_epoch = 0
        trainer.run()
        atomic_json(args.output_dir / "runtime_check.json", {
            "passed": True, "full_size": args.full_size,
            "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_MiB": torch.cuda.max_memory_reserved() / 1024**2,
            "initialization": trainer._initial_manifest,
        })
        print("RUNTIME_CHECK_PASSED", flush=True)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
