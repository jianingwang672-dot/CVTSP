from __future__ import annotations

import argparse
import logging
import os

# Apply conservative thread defaults before NumPy/PyTorch/Gurobi dependencies
# are imported. Explicit user environment values still take precedence.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch

from src.JointSelfImprovementTrainer import JointSelfImprovementTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Formal joint POMO/Gurobi-dual/self-improvement/imitation training."
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-7)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--train-instances", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-problem-size", type=int, default=20)
    parser.add_argument("--max-problem-size", type=int, default=100)
    parser.add_argument("--pomo-size", type=int, default=100)
    parser.add_argument("--cuda-device", type=int, default=4)
    parser.add_argument("--self-improvement-workers", type=int, default=32)
    parser.add_argument(
        "--top-k",
        "--self-improvement-candidates",
        dest="top_k",
        type=int,
        default=30,
        help=(
            "Fixed number of solver-free local-search starts; used only when "
            "--top-k-ratio is 0."
        ),
    )
    parser.add_argument(
        "--top-k-ratio",
        type=float,
        default=0.0,
        help="Use ceil(ratio * J) local-search starts per instance; 0 uses --top-k.",
    )
    parser.add_argument(
        "--near-miss-rate",
        type=float,
        default=0.01,
        help=(
            "Per batch, exactly validate at most ceil(rate * batch_size) "
            "lowest-gap uncertified candidates."
        ),
    )
    parser.add_argument(
        "--top-k-diversity-count",
        type=int,
        default=0,
        help=(
            "Reserve this many Top-K starts for routes with maximum positional "
            "distance from the low-C0 quality subset; 0 keeps the original ranking."
        ),
    )
    parser.add_argument("--max-local-moves", type=int, default=30)
    parser.add_argument("--predictor-warmup-batches", type=int, default=0)
    parser.add_argument("--dual-pruning", action="store_true")
    parser.add_argument("--reference-candidates", type=int, default=200)
    parser.add_argument(
        "--changed-expert-il-weight",
        type=int,
        default=1,
        help="Repeat strictly improved Expert trajectories this many times during IL.",
    )
    parser.add_argument(
        "--rl-reward-mode",
        choices=("hybrid_pomo", "exact_target_expert"),
        default="hybrid_pomo",
        help=(
            "Use the original hybrid POMO reward or compare the already exact-solved "
            "Target route directly with the exact Expert."
        ),
    )
    parser.add_argument("--expert-replay-capacity", type=int, default=0)
    parser.add_argument("--expert-replay-fraction", type=float, default=0.0)
    parser.add_argument("--reference-validation-per-size", type=int, default=200)
    parser.add_argument("--reference-validation-candidates", type=int, default=1)
    parser.add_argument("--validation-dir", default="validation/macsim_fixed")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1,
        help="Save a numbered checkpoint every N completed epochs; use 1 to keep every epoch.",
    )
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument(
        "--resume-extra-epochs",
        action="store_true",
        help="Interpret --epochs as the number of additional epochs after a checkpoint.",
    )
    parser.add_argument(
        "--reset-main-scheduler-on-resume",
        action="store_true",
        help="Start a fresh main-model scheduler phase while retaining optimizer moments.",
    )
    parser.add_argument(
        "--output-dir", default="outputs/train__macsim_cvtsp_from_scratch_e200"
    )
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.checkpoint_interval <= 0:
        raise ValueError("--checkpoint-interval must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("require 0 <= --warmup-epochs < --epochs")
    if not 0.0 <= args.min_learning_rate <= args.learning_rate:
        raise ValueError("require 0 <= --min-learning-rate <= --learning-rate")
    if not 0.0 <= args.top_k_ratio <= 1.0:
        raise ValueError("--top-k-ratio must be in [0, 1]")
    if not 0.0 <= args.near_miss_rate <= 1.0:
        raise ValueError("--near-miss-rate must be in [0, 1]")
    if args.changed_expert_il_weight < 1:
        raise ValueError("--changed-expert-il-weight must be at least 1")
    if not 0 <= args.top_k_diversity_count < args.top_k:
        raise ValueError("--top-k-diversity-count must be in [0, --top-k)")
    if args.expert_replay_capacity < 0:
        raise ValueError("--expert-replay-capacity must be non-negative")
    if not 0.0 <= args.expert_replay_fraction < 1.0:
        raise ValueError("--expert-replay-fraction must be in [0, 1)")
    if args.expert_replay_fraction > 0.0 and args.expert_replay_capacity == 0:
        raise ValueError(
            "positive --expert-replay-fraction requires --expert-replay-capacity > 0"
        )
    # Large BLAS/OpenMP thread pools make the small encoder and many
    # instance-level Gurobi workers contend badly on high-core servers.  Users
    # can still override these before launching if they benchmark otherwise.
    torch_threads = int(os.environ.get("CVTSP_TORCH_THREADS", "1"))
    try:
        torch.set_num_threads(max(torch_threads, 1))
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch allows setting inter-op threads only before parallel work.
        pass
    if args.smoke_test:
        args.epochs = 2
        args.warmup_epochs = 1
        args.train_instances = 2
        args.batch_size = 2
        args.min_problem_size = 10
        args.max_problem_size = 10
        args.pomo_size = 10
        args.self_improvement_workers = 0
        args.top_k = 2
        args.top_k_ratio = 0.0
        args.top_k_diversity_count = 0
        args.near_miss_rate = 0.01
        args.max_local_moves = 5
        args.predictor_warmup_batches = 0
        args.reference_candidates = 8
        args.reference_validation_per_size = 1
        args.reference_validation_candidates = 1
        args.checkpoint_interval = 1
        args.checkpoint_path = ""
        args.validation_dir = os.path.join(args.output_dir, "validation_smoke")

    env_params = {
        "dataset_dir": "instance/Data",
        "min_problem_size": args.min_problem_size,
        "max_problem_size": args.max_problem_size,
        "problem_sizes": None,
        "problem_size": args.max_problem_size,
        "pomo_size": args.pomo_size,
        "pomo_divisor": None,
        "start_node_strategy": "nearest_depot",
    }
    model_params = {
        "embedding_dim": 256,
        "encoder_layer_num": 4,
        "qkv_dim": 16,
        "head_num": 16,
        "logit_clipping": 10,
        "ff_hidden_dim": 512,
        "activation": "gelu",
        "dropout": 0.1,
        "block_size": 1,
    }
    optimizer_params = {
        "optimizer": {"lr": args.learning_rate, "weight_decay": 1e-6},
        "scheduler": {
            "name": "cosine_warmup",
            "total_epochs": args.epochs,
            "warmup_epochs": args.warmup_epochs,
            "min_lr": args.min_learning_rate,
        },
    }
    trainer_params = {
        "use_cuda": not args.smoke_test,
        "cuda_device_num": args.cuda_device,
        "epochs": args.epochs,
        "resume_extra_epochs": args.resume_extra_epochs,
        "reset_main_scheduler_on_resume": args.reset_main_scheduler_on_resume,
        "train_episodes": args.train_instances,
        "train_batch_size": args.batch_size,
        "checkpoint_interval": args.checkpoint_interval,
        "solver_backend": "gurobi",
        "gurobi_threads": 1,
        # Anchor, selected Target reward, and final Expert solves use the same
        # convex QCP formulation. Other POMO rewards remain table lookups.
        "gurobi_explicit_qcp": True,
        "gurobi_params": {
            "FeasibilityTol": 1e-8,
            "OptimalityTol": 1e-8,
            "NumericFocus": 2,
        },
        "cvxpylayer_rebar_loss_enable": False,
        "cvxpylayer_aux_enable": False,
        "penalty_reward": -1e6,
        "grad_clip": 1.0,
        "seed": args.seed,
        "result_folder": args.output_dir,
        "log_level": "INFO",
        "progress_log_percent": 0.5,
        "progress_bar_width": 24,
        # Formal sparse training precisely solves one selected Target rollout,
        # never every POMO rollout.
        "reward_backend": "serial",
        "reward_parallel_workers": 0,
        "parallel_solver_threads": 1,
        "reward_parallel_chunksize": 1,
        "checkpoint_path": args.checkpoint_path,
        # In-batch self-improvement baseline.
        "self_improvement_parallel_workers": args.self_improvement_workers,
        "self_improvement_solver_threads": 1,
        "self_improvement_route_start_top_k": args.top_k,
        "self_improvement_route_start_top_k_ratio": args.top_k_ratio,
        "self_improvement_route_start_diversity_count": args.top_k_diversity_count,
        "self_improvement_near_miss_rate": args.near_miss_rate,
        "self_improvement_max_local_moves": args.max_local_moves,
        "self_improvement_move_types": ("swap", "two_opt"),
        "self_improvement_use_dual_pruning": args.dual_pruning,
        "self_improvement_predictor_warmup_batches": args.predictor_warmup_batches,
        # The fixed-solution table reconstructs a QCP objective from primal
        # variables.  At J=100 Gurobi's floating-point solution can differ by
        # a few 1e-5, so keep the certification margin above that noise floor.
        "self_improvement_absolute_epsilon": 1e-4,
        "self_improvement_relative_epsilon": 1e-8,
        "self_improvement_upper_bound_tolerance": 1e-4,
        "self_improvement_feasibility_tolerance": 1e-5,
        # The imitation and edge objectives use independent optimizer steps.
        "imitation_batch_size": 1000,
        "imitation_inner_epochs": 5,
        "changed_expert_il_weight": args.changed_expert_il_weight,
        "rl_reward_mode": args.rl_reward_mode,
        "expert_replay_capacity": args.expert_replay_capacity,
        "expert_replay_fraction": args.expert_replay_fraction,
        "reference_candidate_count": args.reference_candidates,
        "edge_hidden_dim": 128,
        "edge_dropout": 0.0,
        "edge_learning_rate": 1e-3,
        "edge_weight_decay": 1e-6,
        "edge_scheduler": {"name": "constant_steps"},
        "edge_huber_beta": 0.1,
        "edge_grad_clip": 1.0,
        "edge_batch_size": 8,
        # Frozen-reference/target promotion uses a separate fixed exact set.
        "reference_validation_interval": 1,
        "reference_validation_sizes": (20, 40, 60, 80, 100),
        "reference_validation_per_size": args.reference_validation_per_size,
        "reference_validation_candidates": args.reference_validation_candidates,
        "reference_validation_seed": args.seed + 91_337,
        "reference_validation_dir": args.validation_dir,
        "reference_promotion_epsilon": 0.0,
    }

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("root").info(
        "MACSIM-style joint training: epochs=%d instances/epoch=%d J=[%d,%d] "
        "POMO=J block_size=%d beta=%d TopK=%d(%d diverse) reward=%s "
        "replay=%.0f%%/%d RefVal=5x%d output=%s",
        args.epochs,
        args.train_instances,
        args.min_problem_size,
        args.max_problem_size,
        model_params["block_size"],
        args.reference_candidates,
        args.top_k,
        args.top_k_diversity_count,
        args.rl_reward_mode,
        100.0 * args.expert_replay_fraction,
        args.expert_replay_capacity,
        args.reference_validation_per_size,
        os.path.abspath(args.output_dir),
    )
    trainer = JointSelfImprovementTrainer(
        env_params=env_params,
        model_params=model_params,
        optimizer_params=optimizer_params,
        trainer_params=trainer_params,
    )
    trainer.run()


if __name__ == "__main__":
    main()
