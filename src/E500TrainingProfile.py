"""Algorithm switches recorded by the completed 500-epoch vectorized run.

E500_RUN_DEFAULTS restores the full historical configuration for the main entry.
The explicit e500 algorithm profile still permits controlled experiments with
different budgets/data/workers. No checkpoint/model/optimizer state is imported.
"""

E500_ALGORITHM_ARGS = {
    "learning_rate": 1e-4,
    "min_learning_rate": 1e-7,
    "warmup_epochs": 1,
    "batch_size": 64,
    "min_problem_size": 20,
    "max_problem_size": 100,
    "pomo_size": 100,
    "top_k": 30,
    "top_k_ratio": 0.0,
    "top_k_min": 0,
    "top_k_diversity_count": 10,
    "top_k_diversity_pool_size": 60,
    "top_k_diversity_max_relative_gap": 0.05,
    "near_miss_rate": 0.05,
    "near_miss_min_rate": 0.01,
    "adaptive_near_miss": True,
    "max_local_moves": 30,
    "move_types": ("swap", "two_opt"),
    "perturbation_restarts": 0,
    "predictor_warmup_batches": 0,
    "dual_pruning": False,
    "reference_candidates": 200,
    "changed_expert_il_weight": 5,
    "imitation_inner_epochs": 5,
    "imitation_inner_epochs_schedule": ((1, 5),),
    "rl_reward_mode": "hybrid_pomo_exact_expert",
    "exact_expert_aux_max_weight": 1.0,
    "exact_expert_advantage_clip": 1.0,
    "competence_il_start": 0.50,
    "competence_il_full": 0.85,
    "competence_target_better_full": 0.20,
    "expert_replay_capacity": 50000,
    "expert_replay_fraction": 0.30,
    "imitation_micro_batch_size": 128,
    "checkpoint_interval": 1,
}


E500_RUN_DEFAULTS = {
    **E500_ALGORITHM_ARGS,
    "epochs": 500,
    "scheduler_total_epochs": 0,  # Resolves to the 500-epoch training horizon.
    "train_instances": 10000,
    "self_improvement_workers": 128,
    "cuda_device": 0,
    "seed": 20260813,
    "reference_validation_per_size": 200,
    "reference_validation_candidates": 1,
    "reference_validation_seed": None,  # seed + 91337 = 20352150.
    "validation_dir": "validation/macsim_fixed",
    "local_search_backend": "swap_two_opt",
    "dense_reward_source": "hybrid_upper_bound",
    "ablation_group": None,
    "near_miss_ablation_rate": None,
    "quality_weighted_il": False,
    # Empty means a constant schedule using imitation_inner_epochs (5).
    "imitation_inner_epochs_schedule": (),
    "output_dir": "outputs/train__scratch_e500_restored",
}


def apply_e500_training_profile(args):
    """Restore E500 algorithm choices while retaining the requested budget."""
    for name, value in E500_ALGORITHM_ARGS.items():
        setattr(args, name, value)
    return args
