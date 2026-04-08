from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from src.TSPEnv import RolloutResult
from src.TSPTrainer import OnlineTSPTrainer
from src.TSProblemDef import load_dataset


def _build_instances_and_rollout() -> tuple[list, RolloutResult, list[list[int]]]:
    instances = [instance for instance in load_dataset("instance/Data", "all") if instance.J == 20][:2]
    if len(instances) < 2:
        raise ValueError("need at least two real J=20 instances for verification")

    identity = list(range(20))
    reverse = list(reversed(identity))
    roll1 = identity[1:] + identity[:1]
    zigzag = identity[::2] + identity[1::2]
    sequences = [identity, reverse, roll1, zigzag]

    sequence_tensor = torch.tensor([sequences for _ in instances], dtype=torch.long)
    log_probs = torch.zeros((len(instances), len(sequences)), dtype=torch.float32)
    raw_node_indices = torch.zeros((len(instances), len(sequences), 21), dtype=torch.long)
    rollout = RolloutResult(
        sequences=sequence_tensor,
        log_probs=log_probs,
        raw_node_indices=raw_node_indices,
    )
    return instances, rollout, sequences


def _build_trainer(
    result_folder: Path,
    reward_backend: str,
    reward_parallel_workers: int,
    parallel_solver_threads: int,
) -> OnlineTSPTrainer:
    env_params = {
        "dataset_dir": "instance/Data",
        "min_problem_size": 20,
        "max_problem_size": 100,
        "problem_sizes": [20, 40, 60, 80, 100],
        "problem_size": 100,
        "pomo_size": 100,
    }
    model_params = {
        "embedding_dim": 128,
        "encoder_layer_num": 6,
        "qkv_dim": 16,
        "head_num": 8,
        "logit_clipping": 10,
        "ff_hidden_dim": 512,
    }
    optimizer_params = {
        "optimizer": {
            "lr": 1e-4,
            "weight_decay": 1e-6,
        },
        "scheduler": {
            "milestones": [501],
            "gamma": 0.1,
        },
    }
    trainer_params = {
        "use_cuda": False,
        "cuda_device_num": 0,
        "epochs": 1,
        "train_episodes": 1,
        "train_batch_size": 1,
        "checkpoint_interval": 1,
        "gurobi_threads": 1,
        "penalty_reward": -1e6,
        "grad_clip": 1.0,
        "seed": 1234,
        "result_folder": str(result_folder),
        "log_level": "INFO",
        "reward_backend": reward_backend,
        "reward_parallel_workers": reward_parallel_workers,
        "parallel_solver_threads": parallel_solver_threads,
        "reward_parallel_chunksize": 1,
    }
    return OnlineTSPTrainer(
        env_params=env_params,
        model_params=model_params,
        optimizer_params=optimizer_params,
        trainer_params=trainer_params,
    )


def main() -> int:
    base_dir = Path("outputs/verify_parallel_reward")
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    instances, rollout, sequences = _build_instances_and_rollout()
    serial_trainer = _build_trainer(
        base_dir / "serial",
        reward_backend="serial",
        reward_parallel_workers=0,
        parallel_solver_threads=1,
    )
    executor_trainer = _build_trainer(
        base_dir / "executor",
        reward_backend="executor",
        reward_parallel_workers=2,
        parallel_solver_threads=1,
    )
    persistent_trainer = _build_trainer(
        base_dir / "persistent_pool",
        reward_backend="persistent_pool",
        reward_parallel_workers=2,
        parallel_solver_threads=1,
    )

    try:
        serial_rewards, serial_feasible_ratio, serial_solver_calls = serial_trainer._compute_batch_rewards_serial(
            instances, rollout
        )
        executor_rewards, executor_feasible_ratio, executor_solver_calls = executor_trainer._compute_batch_rewards(
            instances, rollout
        )
        persistent_rewards, persistent_feasible_ratio, persistent_solver_calls = persistent_trainer._compute_batch_rewards(
            instances, rollout
        )

        serial_cpu = serial_rewards.detach().cpu()
        executor_cpu = executor_rewards.detach().cpu()
        persistent_cpu = persistent_rewards.detach().cpu()
        executor_exact_equal = torch.equal(serial_cpu, executor_cpu)
        persistent_exact_equal = torch.equal(serial_cpu, persistent_cpu)
        executor_allclose_equal = torch.allclose(serial_cpu, executor_cpu, atol=1e-7, rtol=0.0)
        persistent_allclose_equal = torch.allclose(serial_cpu, persistent_cpu, atol=1e-7, rtol=0.0)
        executor_max_abs_diff = (
            float(torch.max(torch.abs(serial_cpu - executor_cpu)).item()) if serial_cpu.numel() > 0 else 0.0
        )
        persistent_max_abs_diff = (
            float(torch.max(torch.abs(serial_cpu - persistent_cpu)).item()) if serial_cpu.numel() > 0 else 0.0
        )
        executor_feasible_ratio_match = abs(serial_feasible_ratio - executor_feasible_ratio) < 1e-12
        persistent_feasible_ratio_match = abs(serial_feasible_ratio - persistent_feasible_ratio) < 1e-12

        result = {
            "instances": [instance.instance_id for instance in instances],
            "problem_size": instances[0].J,
            "num_instances": len(instances),
            "num_sequences_per_instance": len(sequences),
            "fixed_sequences": sequences,
            "serial_solver_threads": 1,
            "serial_solver_calls": serial_solver_calls,
            "serial_feasible_ratio": serial_feasible_ratio,
            "executor_workers": 2,
            "executor_solver_threads": 1,
            "executor_solver_calls": executor_solver_calls,
            "executor_feasible_ratio": executor_feasible_ratio,
            "executor_feasible_ratio_match": executor_feasible_ratio_match,
            "executor_exact_reward_match": executor_exact_equal,
            "executor_allclose_reward_match": executor_allclose_equal,
            "executor_max_abs_reward_diff": executor_max_abs_diff,
            "persistent_workers": 2,
            "persistent_solver_threads": 1,
            "persistent_solver_calls": persistent_solver_calls,
            "persistent_feasible_ratio": persistent_feasible_ratio,
            "persistent_feasible_ratio_match": persistent_feasible_ratio_match,
            "persistent_exact_reward_match": persistent_exact_equal,
            "persistent_allclose_reward_match": persistent_allclose_equal,
            "persistent_max_abs_reward_diff": persistent_max_abs_diff,
            "serial_rewards": serial_cpu.tolist(),
            "executor_rewards": executor_cpu.tolist(),
            "persistent_rewards": persistent_cpu.tolist(),
        }

        result_path = base_dir / "result.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0 if (
            executor_exact_equal
            and executor_feasible_ratio_match
            and persistent_exact_equal
            and persistent_feasible_ratio_match
        ) else 1
    finally:
        serial_trainer.close()
        executor_trainer.close()
        persistent_trainer.close()


if __name__ == "__main__":
    raise SystemExit(main())
