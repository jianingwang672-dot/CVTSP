from __future__ import annotations

import argparse
import json
import logging
import os

# Apply conservative thread defaults before NumPy/PyTorch/Gurobi dependencies
# are imported. Explicit user environment values still take precedence.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch

from src.JointSelfImprovementTrainer import JointSelfImprovementTrainer
from src.E500TrainingProfile import E500_RUN_DEFAULTS, apply_e500_training_profile


def _parse_epoch_int_schedule(value: str) -> tuple[tuple[int, int], ...]:
    """Parse a predeclared ``start_epoch:value`` schedule."""

    entries: list[tuple[int, int]] = []
    for raw_entry in value.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        try:
            raw_epoch, raw_value = entry.split(":", maxsplit=1)
            start_epoch = int(raw_epoch)
            scheduled_value = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(
                "schedule entries must use start_epoch:value, for example "
                "1:5,51:2,101:1"
            ) from exc
        if start_epoch <= 0 or scheduled_value <= 0:
            raise argparse.ArgumentTypeError(
                "schedule epochs and values must both be positive"
            )
        entries.append((start_epoch, scheduled_value))
    if not entries or entries[0][0] != 1:
        raise argparse.ArgumentTypeError("schedule must start at epoch 1")
    if any(left[0] >= right[0] for left, right in zip(entries, entries[1:])):
        raise argparse.ArgumentTypeError("schedule epochs must be strictly increasing")
    return tuple(entries)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Formal joint POMO/Gurobi-dual/self-improvement/imitation training."
    )
    parser.add_argument(
        "--training-profile", choices=("e500",), default=None,
        help="Restore the completed E500 algorithm settings; retain budget, data, workers and seed.",
    )
    parser.add_argument(
        "--print-config", action="store_true",
        help="Print resolved configuration and exit before creating a trainer or running training.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-7)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument(
        "--scheduler-total-epochs",
        type=int,
        default=0,
        help=(
            "Scheduler horizon stored in the checkpoint; 0 uses --epochs. "
            "Set this to the original horizon for short resumed A/B runs."
        ),
    )
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
        "--top-k-min",
        type=int,
        default=0,
        help=(
            "Minimum local-search starts when --top-k-ratio is positive; "
            "0 preserves the ratio-only behavior."
        ),
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
        "--near-miss-min-rate",
        type=float,
        default=0.01,
        help="Cold-start floor used with --adaptive-near-miss.",
    )
    parser.add_argument(
        "--adaptive-near-miss",
        action=argparse.BooleanOptionalAction,
        help=(
            "Interpolate near-miss validation from the minimum to maximum rate "
            "using only previous-epoch IL accuracy and Target/Expert wins."
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
    parser.add_argument(
        "--top-k-diversity-pool-size",
        type=int,
        default=0,
        help=(
            "Restrict diverse starts to this many best upper-bound routes; "
            "0 keeps the legacy unrestricted pool."
        ),
    )
    parser.add_argument(
        "--top-k-diversity-max-relative-gap",
        type=float,
        default=float("inf"),
        help=(
            "Require diverse starts to remain within this relative gap of the "
            "worst quality-selected route."
        ),
    )
    parser.add_argument("--max-local-moves", type=int, default=30)
    parser.add_argument(
        "--local-search-backend", choices=("swap_two_opt", "lkh"),
        default="swap_two_opt",
        help="Fixed-C0 local search backend. LKH is opt-in and keeps the same Top-K pool.",
    )
    parser.add_argument("--lkh-binary", default="third_party/lkh3_20261014/LKH")
    parser.add_argument("--lkh-budget-coefficient", type=float, default=1.5e-4,
                        help="Per-instance LKH wall budget is max(0.005, coefficient*J^2) seconds.")
    parser.add_argument("--lkh-desired-scale", type=int, default=1_000_000)
    parser.add_argument("--lkh-trials", type=int, default=1)
    parser.add_argument("--lkh-max-candidates", type=int, default=5)
    parser.add_argument("--lkh-move-type", type=int, default=5)
    parser.add_argument("--ablation-group", choices=("swap", "two_opt", "both"), default=None,
                        help="Fresh-run search ablation; overrides E500 move types and min LR.")
    parser.add_argument("--move-schedule-horizon", type=int, default=100)
    parser.add_argument("--move-schedule-final", type=int, default=10)
    parser.add_argument("--near-miss-ablation-rate", type=float, choices=(0.05, 0.10, 0.20), default=None,
                        help="Fresh near-miss quota study: fixed nominal rate and 30-move budget.")
    parser.add_argument("--quality-weighted-il", action="store_true",
                        help="Experiment B: fresh pre-IL exact Target comparison and normalized quality-weighted CE.")
    parser.add_argument("--diagnostic-dir", default="")
    parser.add_argument("--diagnostic-epochs", type=int, nargs="+", default=(1, 5, 10, 20))
    parser.add_argument(
        "--move-types",
        nargs="+",
        choices=("swap", "two_opt", "relocate", "or_opt_1", "or_opt_2", "or_opt_3"),
        default=("swap", "two_opt"),
        help="Solver-free neighborhoods used under the fixed C0 table.",
    )
    parser.add_argument(
        "--perturbation-restarts",
        type=int,
        default=0,
        help="Double-bridge restarts after a local optimum; 0 disables Branch B.",
    )
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
        choices=(
            "hybrid_pomo",
            "exact_target_expert",
            "hybrid_pomo_exact_expert",
        ),
        default="hybrid_pomo",
        help=(
            "Use the original hybrid POMO reward, exact Target/Expert reward, or "
            "standardized dense POMO reward with a bounded exact auxiliary."
        ),
    )
    parser.add_argument(
        "--dense-reward-source",
        choices=("hybrid_upper_bound", "random_subset_exact"),
        default="hybrid_upper_bound",
        help=(
            "Dense POMO term of hybrid_pomo_exact_expert: the original one-exact-"
            "plus-upper-bounds group, or exact values of m(J) uniformly sampled "
            "Target routes with advantages computed only among them."
        ),
    )
    parser.add_argument("--random-subset-min-routes", type=int, default=4)
    parser.add_argument("--random-subset-max-routes", type=int, default=8)
    parser.add_argument(
        "--gpu-gate-dir", default="",
        help="Override the shared GPU-phase gate directory of ablation runs.",
    )
    parser.add_argument("--exact-expert-aux-max-weight", type=float, default=1.0)
    parser.add_argument("--exact-expert-advantage-clip", type=float, default=1.0)
    parser.add_argument("--competence-il-start", type=float, default=0.5)
    parser.add_argument("--competence-il-full", type=float, default=0.85)
    parser.add_argument(
        "--competence-target-better-full", type=float, default=0.2
    )
    parser.add_argument("--expert-replay-capacity", type=int, default=0)
    parser.add_argument("--expert-replay-fraction", type=float, default=0.0)
    parser.add_argument("--imitation-inner-epochs", type=int, default=5)
    parser.add_argument(
        "--imitation-inner-epochs-schedule",
        type=_parse_epoch_int_schedule,
        default=(),
        metavar="EPOCH:COUNT,...",
        help=(
            "Fixed epoch schedule for IL passes, for example 1:5,51:2,101:1. "
            "When omitted, --imitation-inner-epochs is constant."
        ),
    )
    parser.add_argument(
        "--imitation-micro-batch-size",
        type=int,
        default=0,
        help=(
            "Maximum teacher-forcing forward/backward chunk size. 0 uses the "
            "logical imitation batch size; smaller values accumulate gradients "
            "before the same optimizer step to reduce peak CUDA memory."
        ),
    )
    parser.add_argument("--reference-validation-per-size", type=int, default=200)
    parser.add_argument("--reference-validation-seed", type=int, default=None)
    parser.add_argument(
        "--reference-validation-candidates", type=int, default=1,
        help=(
            "1 (default): historical single free-start greedy route, exact Gurobi. "
            "0 explicitly enables complete POMO=J, no augmentation, exact minimum "
            "over all distinct routes (higher validation cost)."
        ),
    )
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
        "--strict-resume",
        action="store_true",
        help=(
            "Require an exact-state joint checkpoint and reject missing state, "
            "configuration changes, scheduler resets, and learning-rate overrides."
        ),
    )
    parser.add_argument(
        "--output-dir", default="outputs/train__macsim_cvtsp_from_scratch_e200"
    )
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--smoke-test", action="store_true")
    parser.set_defaults(**E500_RUN_DEFAULTS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.training_profile == "e500":
        apply_e500_training_profile(args)
    if args.checkpoint_interval <= 0:
        raise ValueError("--checkpoint-interval must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.warmup_epochs < 0:
        raise ValueError("require --warmup-epochs >= 0")
    if not 0.0 <= args.min_learning_rate <= args.learning_rate:
        raise ValueError("require 0 <= --min-learning-rate <= --learning-rate")
    if not 0.0 <= args.top_k_ratio <= 1.0:
        raise ValueError("--top-k-ratio must be in [0, 1]")
    if args.top_k_min < 0:
        raise ValueError("--top-k-min must be non-negative")
    if not 0.0 <= args.near_miss_rate <= 1.0:
        raise ValueError("--near-miss-rate must be in [0, 1]")
    if not 0.0 <= args.near_miss_min_rate <= args.near_miss_rate:
        raise ValueError(
            "--near-miss-min-rate must be in [0, --near-miss-rate]"
        )
    if args.changed_expert_il_weight < 1:
        raise ValueError("--changed-expert-il-weight must be at least 1")
    if not 0 <= args.top_k_diversity_count < args.top_k:
        raise ValueError("--top-k-diversity-count must be in [0, --top-k)")
    if args.top_k_diversity_pool_size < 0:
        raise ValueError("--top-k-diversity-pool-size must be non-negative")
    if args.top_k_diversity_max_relative_gap < 0.0:
        raise ValueError(
            "--top-k-diversity-max-relative-gap must be non-negative"
        )
    if args.exact_expert_aux_max_weight < 0.0:
        raise ValueError("--exact-expert-aux-max-weight must be non-negative")
    if args.exact_expert_advantage_clip <= 0.0:
        raise ValueError("--exact-expert-advantage-clip must be positive")
    if not args.competence_il_start < args.competence_il_full:
        raise ValueError("--competence-il-start must be below --competence-il-full")
    if args.competence_target_better_full <= 0.0:
        raise ValueError("--competence-target-better-full must be positive")
    if args.expert_replay_capacity < 0:
        raise ValueError("--expert-replay-capacity must be non-negative")
    if not 0.0 <= args.expert_replay_fraction < 1.0:
        raise ValueError("--expert-replay-fraction must be in [0, 1)")
    if args.expert_replay_fraction > 0.0 and args.expert_replay_capacity == 0:
        raise ValueError(
            "positive --expert-replay-fraction requires --expert-replay-capacity > 0"
        )
    if args.imitation_inner_epochs <= 0:
        raise ValueError("--imitation-inner-epochs must be positive")
    if args.imitation_micro_batch_size < 0:
        raise ValueError("--imitation-micro-batch-size must be non-negative")
    if args.perturbation_restarts < 0:
        raise ValueError("--perturbation-restarts must be non-negative")
    if args.local_search_backend == "lkh":
        if args.lkh_budget_coefficient <= 0 or args.lkh_desired_scale <= 0:
            raise ValueError("LKH budget coefficient and scale must be positive")
        if args.lkh_trials <= 0 or args.lkh_max_candidates <= 0 or args.lkh_move_type <= 0:
            raise ValueError("LKH trials, candidates and move type must be positive")
        args.lkh_binary = os.path.abspath(args.lkh_binary)
        if not os.path.isfile(args.lkh_binary):
            raise FileNotFoundError(f"LKH binary not found: {args.lkh_binary}")
    if args.strict_resume:
        if not args.checkpoint_path:
            raise ValueError("--strict-resume requires --checkpoint-path")
        if args.resume_extra_epochs:
            raise ValueError(
                "--strict-resume cannot be combined with --resume-extra-epochs; "
                "reuse the original absolute --epochs endpoint"
            )
        if args.reset_main_scheduler_on_resume:
            raise ValueError(
                "--strict-resume cannot be combined with "
                "--reset-main-scheduler-on-resume"
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
        args.top_k_min = 0
        args.top_k_diversity_count = 0
        args.near_miss_rate = 0.01
        args.max_local_moves = 5
        args.move_types = ("swap", "two_opt")
        args.perturbation_restarts = 0
        args.predictor_warmup_batches = 0
        args.reference_candidates = 8
        args.reference_validation_per_size = 1
        args.checkpoint_interval = 1
        if not args.strict_resume:
            args.checkpoint_path = ""
        args.validation_dir = os.path.join(args.output_dir, "validation_smoke")

    scheduler_total_epochs = args.scheduler_total_epochs or args.epochs
    if args.ablation_group:
        if args.training_profile != "e500":
            raise ValueError("--ablation-group requires --training-profile e500")
        args.min_learning_rate = 1e-5
        args.move_types = {"swap": ("swap",), "two_opt": ("two_opt",),
                           "both": ("swap", "two_opt")}[args.ablation_group]
        if args.move_schedule_horizon < 2 or not 0 <= args.move_schedule_final <= args.max_local_moves:
            raise ValueError("invalid move schedule")
    if args.near_miss_ablation_rate is not None:
        if args.training_profile != "e500" or args.ablation_group != "both":
            raise ValueError("near-miss quota study requires E500 profile and swap+2-opt")
        args.near_miss_min_rate = args.near_miss_ablation_rate
        args.near_miss_rate = args.near_miss_ablation_rate
        args.max_local_moves = args.move_schedule_final = 30
    if args.quality_weighted_il and args.near_miss_ablation_rate != .20:
        raise ValueError("quality IL experiment requires fixed near-miss rate .20")
    if args.local_search_backend == "lkh":
        if args.training_profile != "e500" or args.ablation_group != "both":
            raise ValueError("LKH training requires E500 profile and diagnostic group both")
        if args.near_miss_ablation_rate != .20:
            raise ValueError("LKH training requires fixed near-miss rate .20")
        if args.quality_weighted_il:
            raise ValueError("LKH training uses the established unweighted IL objective")
    if scheduler_total_epochs <= args.warmup_epochs:
        raise ValueError("--scheduler-total-epochs must exceed --warmup-epochs")

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
            "total_epochs": scheduler_total_epochs,
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
        "strict_resume": args.strict_resume,
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
        "self_improvement_route_start_top_k_min": args.top_k_min,
        "self_improvement_route_start_diversity_count": args.top_k_diversity_count,
        "self_improvement_route_start_diversity_pool_size": (
            args.top_k_diversity_pool_size
        ),
        "self_improvement_route_start_diversity_max_relative_gap": (
            args.top_k_diversity_max_relative_gap
        ),
        "self_improvement_near_miss_rate": args.near_miss_rate,
        "self_improvement_near_miss_min_rate": args.near_miss_min_rate,
        "self_improvement_adaptive_near_miss": args.adaptive_near_miss,
        "self_improvement_competence_il_start": args.competence_il_start,
        "self_improvement_competence_il_full": args.competence_il_full,
        "self_improvement_competence_target_better_full": (
            args.competence_target_better_full
        ),
        "self_improvement_max_local_moves": args.max_local_moves,
        "self_improvement_move_types": tuple(args.move_types),
        "self_improvement_perturbation_restarts": args.perturbation_restarts,
        "self_improvement_use_dual_pruning": args.dual_pruning,
        "self_improvement_predictor_warmup_batches": args.predictor_warmup_batches,
        # The fixed-solution table reconstructs a QCP objective from primal
        # variables.  At J=100 Gurobi's floating-point solution can differ by
        # a few 1e-5, so keep the certification margin above that noise floor.
        "self_improvement_absolute_epsilon": 1e-4,
        "self_improvement_relative_epsilon": 1e-8,
        "self_improvement_upper_bound_tolerance": 1e-4,
        "self_improvement_feasibility_tolerance": 1e-5,
        "self_improvement_local_search_backend": args.local_search_backend,
        "self_improvement_lkh_binary": args.lkh_binary,
        "self_improvement_lkh_budget_coefficient": args.lkh_budget_coefficient,
        "self_improvement_lkh_desired_scale": args.lkh_desired_scale,
        "self_improvement_lkh_trials": args.lkh_trials,
        "self_improvement_lkh_max_candidates": args.lkh_max_candidates,
        "self_improvement_lkh_move_type": args.lkh_move_type,
        "self_improvement_lkh_native_seconds": 2.0,
        "self_improvement_lkh_process_timeout_seconds": 5.0,
        # The imitation and edge objectives use independent optimizer steps.
        "imitation_batch_size": 1000,
        "imitation_micro_batch_size": args.imitation_micro_batch_size or 1000,
        "imitation_inner_epochs": args.imitation_inner_epochs,
        "imitation_inner_epochs_schedule": (
            args.imitation_inner_epochs_schedule
            or ((1, args.imitation_inner_epochs),)
        ),
        "changed_expert_il_weight": args.changed_expert_il_weight,
        "rl_reward_mode": args.rl_reward_mode,
        "dense_reward_source": args.dense_reward_source,
        "random_subset_min_routes": args.random_subset_min_routes,
        "random_subset_max_routes": args.random_subset_max_routes,
        "exact_expert_aux_max_weight": args.exact_expert_aux_max_weight,
        "exact_expert_advantage_clip": args.exact_expert_advantage_clip,
        "hybrid_advantage_epsilon": 1e-6,
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
        "reference_validation_seed": (
            args.seed + 91_337
            if args.reference_validation_seed is None
            else args.reference_validation_seed
        ),
        "reference_validation_dir": args.validation_dir,
        "reference_promotion_epsilon": 0.0,
    }
    if args.ablation_group:
        trainer_params.update({
            "ablation_group": args.ablation_group,
            "move_schedule_horizon": args.move_schedule_horizon,
            "move_schedule_final": args.move_schedule_final,
            "diagnostic_dir": args.diagnostic_dir,
            "diagnostic_epochs": tuple(args.diagnostic_epochs),
            "gpu_max_active": 2,
            "gpu_gate_dir": str(os.path.abspath("outputs/search_ablation_20261012/.gpu_slots")),
        })
    if args.near_miss_ablation_rate is not None:
        trainer_params.update({
            "near_miss_ablation_rate": args.near_miss_ablation_rate,
            "gpu_gate_dir": str(os.path.abspath("outputs/near_miss_quota_20261013/.gpu_slots")),
        })

    if args.quality_weighted_il:
        trainer_params.update({"quality_weighted_il": True,
                               "quality_il_scale": .02, "quality_il_offset": .5,
                               "quality_il_min": .1, "quality_il_max": 2.,
                               "quality_il_measurement": "post_rl_sample_pomo_bound_selected_exact_v1",
                               "quality_il_batch_size": 64,
                               "gpu_gate_dir": str(os.path.abspath("outputs/quality_il_b_20261014/.gpu_slots"))})
    if args.local_search_backend == "lkh":
        trainer_params.update({
            "lkh_training_protocol": "nm20_lkh_matched_budget_v1",
            "gpu_max_active": 1,
            "gpu_gate_dir": str(os.path.abspath("outputs/lkh_nm20_e100_20261014/.gpu_slots")),
        })
    if args.gpu_gate_dir:
        if not args.ablation_group:
            raise ValueError("--gpu-gate-dir only applies to ablation trainers")
        trainer_params.update({
            "gpu_max_active": 2,
            "gpu_gate_dir": os.path.abspath(args.gpu_gate_dir),
        })
    if args.print_config:
        print(json.dumps({
            "training_profile": args.training_profile,
            "env_params": env_params,
            "model_params": model_params,
            "optimizer_params": optimizer_params,
            "trainer_params": trainer_params,
        }, indent=2))
        return

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("root").info(
        "MACSIM-style joint training: epochs=%d instances/epoch=%d J=[%d,%d] "
        "POMO=J block_size=%d beta=%d TopK=%d/min%d/ratio%.3f(%d diverse) "
        "search=%s moves=%s perturb=%d reward=%s IL=%s replay=%.0f%%/%d RefVal=5x%d output=%s",
        args.epochs,
        args.train_instances,
        args.min_problem_size,
        args.max_problem_size,
        model_params["block_size"],
        args.reference_candidates,
        args.top_k,
        args.top_k_min,
        args.top_k_ratio,
        args.top_k_diversity_count,
        args.local_search_backend,
        tuple(args.move_types),
        args.perturbation_restarts,
        args.rl_reward_mode,
        str(
            args.imitation_inner_epochs_schedule
            or ((1, args.imitation_inner_epochs),)
        ),
        100.0 * args.expert_replay_fraction,
        args.expert_replay_capacity,
        args.reference_validation_per_size,
        os.path.abspath(args.output_dir),
    )
    trainer_class = JointSelfImprovementTrainer
    if args.ablation_group:
        from src.SearchAblationTrainer import SearchAblationTrainer
        trainer_class = SearchAblationTrainer
    if args.near_miss_ablation_rate is not None:
        from src.NearMissQuotaTrainer import NearMissQuotaTrainer
        trainer_class = NearMissQuotaTrainer
    if args.quality_weighted_il:
        from src.QualityILTrainer import QualityILTrainer
        trainer_class = QualityILTrainer
    if args.local_search_backend == "lkh":
        from src.LKHTrainingTrainer import LKHTrainingTrainer
        trainer_class = LKHTrainingTrainer
    trainer = trainer_class(
        env_params=env_params,
        model_params=model_params,
        optimizer_params=optimizer_params,
        trainer_params=trainer_params,
    )
    trainer.run()


if __name__ == "__main__":
    main()
