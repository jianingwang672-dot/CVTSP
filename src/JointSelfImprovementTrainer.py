from __future__ import annotations

import copy
import json
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from src.CertifiedLocalSearch import FixedAnchorLocalSearchConfig
from src.CertifiedUpperBound import (
    VECTORIZED_LOCAL_SEARCH_ENABLED,
    is_optimal_gurobi_solution,
    validate_tour,
)
from src.CVPSolver import SolverConfig
from src.FixedValidation import (
    flatten_fixed_validation_sets,
    load_or_create_fixed_validation_sets,
    summarize_fixed_validation_sets,
)
from src.ImitationLearning import teacher_forcing_loss
from src.JointSelfImprovement import (
    C0RefreshTask,
    NearMissValidationTask,
    SparseSelfImprovementResult,
    SparseSelfImprovementTask,
    SharedDirectedEdgeDecoder,
    TargetRewardSolveTask,
    apply_near_miss_validation,
    batched_directed_step_costs,
    denormalize_contribution_matrix,
    directed_route_upper_bound,
    edge_decoder_loss,
    normalize_contribution_matrix,
    run_c0_refresh_task,
    run_near_miss_validation_task,
    run_sparse_self_improvement_task,
    run_target_reward_solve_task,
    serialize_joint_checkpoint_metadata,
)
from src.ReferencePolicy import (
    ReferenceState,
    clone_reference_policy,
    evaluate_policy_pair_exact,
    freeze_reference_policy,
    promote_reference_with_size_gate,
    reference_validation_protocol,
    require_matching_validation_protocol,
)
from src.TSPEnv import RolloutResult, generate_candidate_pool_batch, rollout_batch
from src.TSPTrainer import (
    OnlineTSPTrainer,
    _assert_scheduler_optimizer_lr_match,
    _build_scheduler,
    _format_progress_bar,
    _require_checkpoint_keys,
)
from src.TSPUtils import save_checkpoint


class JointSelfImprovementTrainer(OnlineTSPTrainer):
    """Sparse-Gurobi joint trainer for policy, pseudo expert, and edge evaluator.

    The training budget is structural rather than heuristic: each instance
    performs one anchor solve, one exact solve for the best upper-bound Target
    route, and at most one final Expert solve. The remaining POMO routes use
    certified upper-bound table lookups, so the total budget is 2--3 Gurobi
    calls per instance rather than one call for every POMO route.
    """

    _STRICT_TRAJECTORY_PARAMETER_KEYS = (
        "use_cuda",
        "train_episodes",
        "train_batch_size",
        "solver_backend",
        "gurobi_threads",
        "gurobi_time_limit",
        "gurobi_explicit_qcp",
        "gurobi_params",
        "penalty_reward",
        "grad_clip",
        "seed",
        "self_improvement_route_start_top_k",
        "self_improvement_route_start_top_k_ratio",
        "self_improvement_route_start_top_k_min",
        "self_improvement_route_start_diversity_count",
        "self_improvement_route_start_diversity_pool_size",
        "self_improvement_route_start_diversity_max_relative_gap",
        "self_improvement_parallel_workers",
        "self_improvement_solver_threads",
        "self_improvement_near_miss_rate",
        "self_improvement_near_miss_min_rate",
        "self_improvement_adaptive_near_miss",
        "self_improvement_competence_il_start",
        "self_improvement_competence_il_full",
        "self_improvement_competence_target_better_full",
        "self_improvement_max_local_moves",
        "self_improvement_move_types",
        "self_improvement_perturbation_restarts",
        "self_improvement_use_dual_pruning",
        "self_improvement_predictor_warmup_batches",
        "self_improvement_absolute_epsilon",
        "self_improvement_relative_epsilon",
        "self_improvement_upper_bound_tolerance",
        "self_improvement_feasibility_tolerance",
        "self_improvement_local_search_backend",
        "self_improvement_lkh_binary",
        "self_improvement_lkh_budget_coefficient",
        "self_improvement_lkh_desired_scale",
        "self_improvement_lkh_trials",
        "self_improvement_lkh_max_candidates",
        "self_improvement_lkh_move_type",
        "self_improvement_lkh_native_seconds",
        "self_improvement_lkh_process_timeout_seconds",
        "imitation_batch_size",
        "imitation_micro_batch_size",
        "imitation_inner_epochs",
        "imitation_inner_epochs_schedule",
        "changed_expert_il_weight",
        "unchanged_expert_il_weight",
        "self_improvement_c0_refresh_rounds",
        "rl_multi_c0_bound",
        "rl_reward_mode",
        "dense_reward_source",
        "random_subset_min_routes",
        "random_subset_max_routes",
        "exact_expert_aux_max_weight",
        "exact_expert_advantage_clip",
        "hybrid_advantage_epsilon",
        "expert_replay_capacity",
        "expert_replay_fraction",
        "reference_candidate_count",
        "edge_hidden_dim",
        "edge_dropout",
        "edge_learning_rate",
        "edge_weight_decay",
        "edge_scheduler",
        "edge_huber_beta",
        "edge_grad_clip",
        "edge_batch_size",
        "reference_validation_interval",
        "reference_validation_sizes",
        "reference_validation_per_size",
        "reference_validation_candidates",
        "reference_validation_seed",
        "reference_promotion_epsilon",
    )

    def __init__(self, env_params, model_params, optimizer_params, trainer_params):
        # The parent trainer normally creates a B*P reward pool.  Formal joint
        # training must never enter that path, so disable it before construction.
        sanitized_params = dict(trainer_params)
        requested_reward_workers = int(sanitized_params.get("reward_parallel_workers", 0))
        sanitized_params["reward_parallel_workers"] = 0
        sanitized_params["reward_backend"] = "serial"
        super().__init__(env_params, model_params, optimizer_params, sanitized_params)
        self.requested_reward_workers = requested_reward_workers

        if str(self.solver_config.solver_backend).strip().lower() != "gurobi":
            raise ValueError("joint self-improvement training requires solver_backend='gurobi'")
        if not self.solver_config.gurobi_explicit_qcp:
            raise ValueError(
                "joint training requires gurobi_explicit_qcp=True so anchor duals, "
                "primal upper bounds, and exact validation use one formulation"
            )
        if (
            self.cvxpylayer_rebar_loss_enable
            or self.cvxpylayer_aux_enable
            or self.cvxpylayer_objective_loss_enable
            or self.cvxpylayer_hard_objective_loss_enable
        ):
            raise ValueError("formal joint training contains no REBAR or cvxpylayer loss")

        self.edge_hidden_dim = int(self.trainer_params.get("edge_hidden_dim", 128))
        self.edge_dropout = float(self.trainer_params.get("edge_dropout", 0.0))
        self.edge_decoder = SharedDirectedEdgeDecoder(
            self.model_config,
            hidden_dim=self.edge_hidden_dim,
            dropout=self.edge_dropout,
        ).to(self.device)
        self.edge_optimizer = torch.optim.Adam(
            self.edge_decoder.parameters(),
            lr=float(self.trainer_params.get("edge_learning_rate", 1e-3)),
            weight_decay=float(self.trainer_params.get("edge_weight_decay", 1e-6)),
        )
        self.edge_scheduler_params = dict(
            self.trainer_params.get("edge_scheduler", {"name": "constant_steps"})
        )
        self.edge_scheduler = _build_scheduler(
            self.edge_optimizer, self.edge_scheduler_params
        )
        self.edge_global_step = 0
        self.edge_huber_beta = float(self.trainer_params.get("edge_huber_beta", 0.1))
        self.edge_grad_clip = float(self.trainer_params.get("edge_grad_clip", 1.0))
        self.edge_batch_size = int(self.trainer_params.get("edge_batch_size", 8))
        self.imitation_batch_size = int(self.trainer_params.get("imitation_batch_size", 64))
        self.imitation_micro_batch_size = int(
            self.trainer_params.get(
                "imitation_micro_batch_size", self.imitation_batch_size
            )
        )
        self.imitation_inner_epochs = int(
            self.trainer_params.get("imitation_inner_epochs", 1)
        )
        raw_il_schedule = self.trainer_params.get(
            "imitation_inner_epochs_schedule",
            ((1, self.imitation_inner_epochs),),
        )
        self.imitation_inner_epochs_schedule = tuple(
            (int(start_epoch), int(inner_epochs))
            for start_epoch, inner_epochs in raw_il_schedule
        )
        self._active_imitation_inner_epochs = self.imitation_inner_epochs
        self.changed_expert_il_weight = int(
            self.trainer_params.get("changed_expert_il_weight", 1)
        )
        # Repeats of an unchanged (Anchor-labelled) instance in IL; 0 trains IL
        # only on strictly improved Experts and the replay pool.
        self.unchanged_expert_il_weight = int(
            self.trainer_params.get("unchanged_expert_il_weight", 1)
        )
        # Extra solver rounds after the sparse phase: rebuild C0 from the route
        # solved last, re-search, and exactly solve one unsolved candidate.
        self.si_c0_refresh_rounds = int(
            self.trainer_params.get("self_improvement_c0_refresh_rounds", 0)
        )
        # RL Target bound: route-level minimum over the C0 tables of every
        # exactly solved route (Anchor, Expert, refresh bases/candidates).
        self.rl_multi_c0_bound = bool(self.trainer_params.get("rl_multi_c0_bound", False))
        self.rl_reward_mode = str(
            self.trainer_params.get("rl_reward_mode", "hybrid_pomo")
        ).strip().lower()
        self.exact_expert_aux_max_weight = float(
            self.trainer_params.get("exact_expert_aux_max_weight", 1.0)
        )
        self.exact_expert_advantage_clip = float(
            self.trainer_params.get("exact_expert_advantage_clip", 1.0)
        )
        self.hybrid_advantage_epsilon = float(
            self.trainer_params.get("hybrid_advantage_epsilon", 1e-6)
        )
        # Dense POMO term source. "random_subset_exact" solves m(J) uniformly
        # sampled Target routes exactly and computes advantages only among
        # them, so the dense gradient is unbiased w.r.t. the all-exact one.
        self.dense_reward_source = str(
            self.trainer_params.get("dense_reward_source", "hybrid_upper_bound")
        ).strip().lower()
        self.random_subset_min_routes = int(
            self.trainer_params.get("random_subset_min_routes", 4)
        )
        self.random_subset_max_routes = int(
            self.trainer_params.get("random_subset_max_routes", 8)
        )
        self._pending_after_sparse_submit = None
        self._random_subset_handle = None
        self.expert_replay_capacity = int(
            self.trainer_params.get("expert_replay_capacity", 0)
        )
        self.expert_replay_fraction = float(
            self.trainer_params.get("expert_replay_fraction", 0.0)
        )
        self.reference_candidate_count = int(
            self.trainer_params.get("reference_candidate_count", 200)
        )
        if self.imitation_batch_size <= 0:
            raise ValueError("imitation_batch_size must be positive")
        if self.imitation_micro_batch_size <= 0:
            raise ValueError("imitation_micro_batch_size must be positive")
        if self.imitation_inner_epochs <= 0:
            raise ValueError("imitation_inner_epochs must be positive")
        if (
            not self.imitation_inner_epochs_schedule
            or self.imitation_inner_epochs_schedule[0][0] != 1
            or any(
                start_epoch <= 0 or inner_epochs <= 0
                for start_epoch, inner_epochs in self.imitation_inner_epochs_schedule
            )
            or any(
                left[0] >= right[0]
                for left, right in zip(
                    self.imitation_inner_epochs_schedule,
                    self.imitation_inner_epochs_schedule[1:],
                )
            )
        ):
            raise ValueError(
                "imitation_inner_epochs_schedule must start at epoch 1 and use "
                "strictly increasing positive epochs and positive counts"
            )
        if self.changed_expert_il_weight < 1:
            raise ValueError("changed_expert_il_weight must be at least 1")
        if self.unchanged_expert_il_weight < 0:
            raise ValueError("unchanged_expert_il_weight must be non-negative")
        if self.si_c0_refresh_rounds < 0:
            raise ValueError("self_improvement_c0_refresh_rounds must be non-negative")
        if self.si_c0_refresh_rounds and str(
            self.trainer_params.get("self_improvement_local_search_backend", "swap_two_opt")
        ).strip().lower() != "swap_two_opt":
            raise ValueError("C0 refresh is implemented for the swap_two_opt backend only")
        if self.rl_reward_mode not in {
            "hybrid_pomo",
            "exact_target_expert",
            "hybrid_pomo_exact_expert",
        }:
            raise ValueError(
                "rl_reward_mode must be 'hybrid_pomo', 'exact_target_expert', "
                "or 'hybrid_pomo_exact_expert'"
            )
        if self.dense_reward_source not in {"hybrid_upper_bound", "random_subset_exact"}:
            raise ValueError(
                "dense_reward_source must be 'hybrid_upper_bound' or 'random_subset_exact'"
            )
        if getattr(self, "dense_reward_source", "hybrid_upper_bound") == "random_subset_exact":
            if self.rl_reward_mode != "hybrid_pomo_exact_expert":
                raise ValueError(
                    "random_subset_exact replaces the dense term of hybrid_pomo_exact_expert"
                )
            if not 2 <= self.random_subset_min_routes <= self.random_subset_max_routes:
                raise ValueError("require 2 <= random_subset_min_routes <= max_routes")
        if self.exact_expert_aux_max_weight < 0.0:
            raise ValueError("exact_expert_aux_max_weight must be non-negative")
        if self.exact_expert_advantage_clip <= 0.0:
            raise ValueError("exact_expert_advantage_clip must be positive")
        if self.hybrid_advantage_epsilon <= 0.0:
            raise ValueError("hybrid_advantage_epsilon must be positive")
        if self.expert_replay_capacity < 0:
            raise ValueError("expert_replay_capacity must be non-negative")
        if not 0.0 <= self.expert_replay_fraction < 1.0:
            raise ValueError("expert_replay_fraction must be in [0, 1)")
        if self.expert_replay_fraction > 0.0 and self.expert_replay_capacity == 0:
            raise ValueError(
                "positive expert_replay_fraction requires expert_replay_capacity > 0"
            )
        if self.reference_candidate_count <= 0:
            raise ValueError("reference_candidate_count must be positive")

        self.si_route_start_top_k = int(
            self.trainer_params.get(
                "self_improvement_route_start_top_k",
                self.trainer_params.get("self_improvement_candidates_per_instance", 4),
            )
        )
        self.si_route_start_top_k_ratio = float(
            self.trainer_params.get("self_improvement_route_start_top_k_ratio", 0.0)
        )
        self.si_route_start_top_k_min = int(
            self.trainer_params.get("self_improvement_route_start_top_k_min", 0)
        )
        self.si_route_start_diversity_count = int(
            self.trainer_params.get("self_improvement_route_start_diversity_count", 0)
        )
        self.si_route_start_diversity_pool_size = int(
            self.trainer_params.get(
                "self_improvement_route_start_diversity_pool_size", 0
            )
        )
        self.si_route_start_diversity_max_relative_gap = float(
            self.trainer_params.get(
                "self_improvement_route_start_diversity_max_relative_gap",
                float("inf"),
            )
        )
        self.si_near_miss_rate = float(
            self.trainer_params.get("self_improvement_near_miss_rate", 0.0)
        )
        self.si_near_miss_min_rate = float(
            self.trainer_params.get(
                "self_improvement_near_miss_min_rate", self.si_near_miss_rate
            )
        )
        self.si_adaptive_near_miss = bool(
            self.trainer_params.get("self_improvement_adaptive_near_miss", False)
        )
        self.competence_il_start = float(
            self.trainer_params.get("self_improvement_competence_il_start", 0.5)
        )
        self.competence_il_full = float(
            self.trainer_params.get("self_improvement_competence_il_full", 0.85)
        )
        self.competence_target_better_full = float(
            self.trainer_params.get(
                "self_improvement_competence_target_better_full", 0.2
            )
        )
        self.si_max_local_moves = int(
            self.trainer_params.get("self_improvement_max_local_moves", 30)
        )
        self.si_predictor_warmup_batches = int(
            self.trainer_params.get("self_improvement_predictor_warmup_batches", 0)
        )
        self.si_move_types = tuple(
            str(item).strip()
            for item in self.trainer_params.get(
                "self_improvement_move_types", ("swap", "two_opt")
            )
            if str(item).strip()
        )
        self.si_use_dual_pruning = bool(
            self.trainer_params.get("self_improvement_use_dual_pruning", False)
        )
        self.si_perturbation_restarts = int(
            self.trainer_params.get("self_improvement_perturbation_restarts", 0)
        )
        self.si_local_search_backend = str(
            self.trainer_params.get("self_improvement_local_search_backend", "swap_two_opt")
        ).strip().lower()
        self.si_lkh_binary = str(
            self.trainer_params.get("self_improvement_lkh_binary", "")
        )
        self.si_lkh_budget_coefficient = float(
            self.trainer_params.get("self_improvement_lkh_budget_coefficient", 1.5e-4)
        )
        self.si_lkh_desired_scale = int(
            self.trainer_params.get("self_improvement_lkh_desired_scale", 1_000_000)
        )
        self.si_lkh_trials = int(self.trainer_params.get("self_improvement_lkh_trials", 1))
        self.si_lkh_max_candidates = int(
            self.trainer_params.get("self_improvement_lkh_max_candidates", 5)
        )
        self.si_lkh_move_type = int(
            self.trainer_params.get("self_improvement_lkh_move_type", 5)
        )
        self.si_lkh_native_seconds = float(
            self.trainer_params.get("self_improvement_lkh_native_seconds", 2.0)
        )
        self.si_lkh_process_timeout_seconds = float(
            self.trainer_params.get("self_improvement_lkh_process_timeout_seconds", 5.0)
        )
        if self.si_route_start_top_k <= 0:
            raise ValueError("self_improvement_route_start_top_k must be positive")
        if not 0.0 <= self.si_route_start_top_k_ratio <= 1.0:
            raise ValueError(
                "self_improvement_route_start_top_k_ratio must be in [0, 1]"
            )
        if self.si_route_start_top_k_min < 0:
            raise ValueError("self_improvement_route_start_top_k_min must be non-negative")
        if not 0 <= self.si_route_start_diversity_count < self.si_route_start_top_k:
            raise ValueError(
                "self_improvement_route_start_diversity_count must be in "
                "[0, self_improvement_route_start_top_k)"
            )
        if self.si_route_start_diversity_pool_size < 0:
            raise ValueError(
                "self_improvement_route_start_diversity_pool_size must be non-negative"
            )
        if self.si_route_start_diversity_max_relative_gap < 0.0:
            raise ValueError(
                "self_improvement_route_start_diversity_max_relative_gap must be "
                "non-negative"
            )
        if not 0.0 <= self.si_near_miss_rate <= 1.0:
            raise ValueError("self_improvement_near_miss_rate must be in [0, 1]")
        if not 0.0 <= self.si_near_miss_min_rate <= self.si_near_miss_rate:
            raise ValueError(
                "self_improvement_near_miss_min_rate must be in "
                "[0, self_improvement_near_miss_rate]"
            )
        if not self.competence_il_start < self.competence_il_full:
            raise ValueError(
                "self_improvement_competence_il_start must be below "
                "self_improvement_competence_il_full"
            )
        if self.competence_target_better_full <= 0.0:
            raise ValueError(
                "self_improvement_competence_target_better_full must be positive"
            )
        if self.model_config.block_size > 1 and self.si_near_miss_rate != 0.0:
            raise ValueError(
                "MACSIM-style strict Expert generation disables near-miss exact acceptance"
            )
        if self.si_max_local_moves < 0:
            raise ValueError("self_improvement_max_local_moves must be non-negative")
        if not self.si_move_types:
            raise ValueError("self_improvement_move_types must not be empty")
        if self.si_perturbation_restarts < 0:
            raise ValueError("self_improvement_perturbation_restarts must be non-negative")
        if self.si_local_search_backend not in {"swap_two_opt", "lkh"}:
            raise ValueError("self_improvement_local_search_backend must be swap_two_opt or lkh")
        if self.si_local_search_backend == "lkh":
            binary = Path(self.si_lkh_binary)
            if not binary.is_file():
                raise FileNotFoundError(f"LKH binary not found: {binary}")
            if self.si_use_dual_pruning or self.si_perturbation_restarts:
                raise ValueError("LKH backend requires dual pruning and perturbation restarts disabled")
            if (
                self.si_lkh_budget_coefficient <= 0
                or self.si_lkh_desired_scale <= 0
                or self.si_lkh_trials <= 0
                or self.si_lkh_max_candidates <= 0
                or self.si_lkh_move_type <= 0
                or self.si_lkh_native_seconds <= 0
                or self.si_lkh_process_timeout_seconds <= 0
            ):
                raise ValueError("invalid LKH training configuration")

        self.si_local_search_config = FixedAnchorLocalSearchConfig(
            max_local_moves=self.si_max_local_moves,
            absolute_epsilon=float(
                self.trainer_params.get("self_improvement_absolute_epsilon", 1e-6)
            ),
            relative_epsilon=float(
                self.trainer_params.get("self_improvement_relative_epsilon", 1e-8)
            ),
            move_types=self.si_move_types,
            use_dual_pruning=self.si_use_dual_pruning,
        )
        self.si_upper_bound_tolerance = float(
            self.trainer_params.get("self_improvement_upper_bound_tolerance", 2e-5)
        )
        self.si_feasibility_tolerance = float(
            self.trainer_params.get("self_improvement_feasibility_tolerance", 1e-5)
        )
        self.si_lkh_work_root = self.output_dir / ".lkh_tmp"
        if self.si_local_search_backend == "lkh":
            self.si_lkh_work_root.mkdir(parents=True, exist_ok=True)
        solver_params = dict(self.solver_config.gurobi_params)
        solver_params.setdefault("FeasibilityTol", 1e-8)
        solver_params.setdefault("OptimalityTol", 1e-8)
        solver_params.setdefault("NumericFocus", 2)
        solver_threads = int(self.trainer_params.get("self_improvement_solver_threads", 1))
        self.si_anchor_solver_config = SolverConfig(
            solver_backend="gurobi",
            gurobi_time_limit=self.solver_config.gurobi_time_limit,
            gurobi_threads=solver_threads,
            output_flag=self.solver_config.output_flag,
            gurobi_params=solver_params,
            gurobi_explicit_qcp=True,
            collect_duals=True,
        )
        self.si_final_solver_config = SolverConfig(
            solver_backend="gurobi",
            gurobi_time_limit=self.solver_config.gurobi_time_limit,
            gurobi_threads=solver_threads,
            output_flag=self.solver_config.output_flag,
            gurobi_params=solver_params,
            gurobi_explicit_qcp=True,
            collect_duals=False,
        )

        self.si_parallel_workers = int(
            self.trainer_params.get("self_improvement_parallel_workers", 0)
        )
        self._si_executor: ProcessPoolExecutor | None = None
        if self.si_parallel_workers > 0:
            self._si_executor = self._new_si_executor()
        self._si_executor_restarts = 0

        # A frozen reference generates pseudo-expert candidates.  The target is
        # self.model and is updated by two separate policy optimizer steps.
        self.reference_policy = clone_reference_policy(self.model, device=self.device)
        self.reference_state = ReferenceState()
        self.reference_validation_interval = int(
            self.trainer_params.get("reference_validation_interval", 5)
        )
        self.reference_validation_sizes = tuple(
            int(size)
            for size in self.trainer_params.get(
                "reference_validation_sizes", (20, 40, 60, 80, 100)
            )
        )
        self.reference_validation_per_size = int(
            self.trainer_params.get("reference_validation_per_size", 200)
        )
        self.reference_validation_count = (
            len(self.reference_validation_sizes) * self.reference_validation_per_size
        )
        self.reference_validation_candidates = int(
            self.trainer_params.get("reference_validation_candidates", 1)
        )
        self.reference_validation_seed = int(
            self.trainer_params.get(
                "reference_validation_seed", int(self.trainer_params.get("seed", 1234)) + 91_337
            )
        )
        self.reference_promotion_epsilon = float(
            self.trainer_params.get("reference_promotion_epsilon", 1e-6)
        )
        self.reference_validation_dir = Path(
            self.trainer_params.get("reference_validation_dir", "validation/macsim_fixed")
        )
        if self.reference_validation_interval < 0:
            raise ValueError("reference_validation_interval must be non-negative")
        if not self.reference_validation_sizes or len(set(self.reference_validation_sizes)) != len(
            self.reference_validation_sizes
        ):
            raise ValueError("reference_validation_sizes must be distinct and non-empty")
        if self.reference_validation_per_size <= 0:
            raise ValueError("reference_validation_per_size must be positive")
        if (
            self.reference_validation_count > 0
            and self.reference_validation_candidates < 0
        ):
            raise ValueError("reference_validation_candidates must be non-negative; 0 means full POMO")
        self.reference_validation_protocol = reference_validation_protocol(
            self.reference_validation_candidates
        )
        if self.reference_promotion_epsilon < 0:
            raise ValueError("reference_promotion_epsilon must be non-negative")
        self.reference_validation_instances = []
        self._joint_batch_index = 0
        self._last_batch_exact_examples: list[dict[str, Any]] = []
        self._last_batch_training_records: list[
            tuple[Any, list[int], np.ndarray, bool]
        ] = []
        self._expert_replay_pool: list[tuple[Any, list[int]]] = []
        self._previous_epoch_il_accuracy = float(self.competence_il_start)
        self._previous_epoch_target_better_ratio = 0.0
        self._active_near_miss_rate = float(self.si_near_miss_min_rate)
        self._strict_saved_reference_validation_summary: dict[str, Any] | None = None
        self.joint_origin_epoch = self.start_epoch - 1
        self._restore_joint_checkpoint()
        self.reference_validation_groups = load_or_create_fixed_validation_sets(
            dataset_dir=self.env_params.get("dataset_dir", "instance/Data"),
            validation_dir=self.reference_validation_dir,
            problem_sizes=self.reference_validation_sizes,
            per_size=self.reference_validation_per_size,
            seed=self.reference_validation_seed,
        )
        self.reference_validation_instances = flatten_fixed_validation_sets(
            self.reference_validation_groups
        )
        self.reference_validation_summary = summarize_fixed_validation_sets(
            self.reference_validation_groups
        )
        if self.strict_resume:
            if (
                self._strict_saved_reference_validation_summary
                != self.reference_validation_summary
            ):
                raise ValueError(
                    "strict resume fixed-validation fingerprint or shape mismatch: "
                    f"checkpoint={self._strict_saved_reference_validation_summary}, "
                    f"current={self.reference_validation_summary}"
                )
            logger.info(
                "Strict resume accepted: policy, optimizer, scheduler, reference, "
                "predictor, replay, validation, and RNG state checks passed"
            )
        self._write_reference_validation_manifest()

        logger.info(
            "Formal sparse-Gurobi mode: rollout reward workers {} -> 0; "
            "training calls 2-3/instance; TopK=fixed:{} ratio:{} near_miss={}:{} "
            "diverse_starts={} gate=top{}/+{:.1%} local_search={} local_moves={} moves={} "
            "perturb_restarts={} dual_pruning={} vectorized_moves={} "
            "reward={} replay={:.0%}/{}",
            requested_reward_workers,
            self.si_route_start_top_k,
            self.si_route_start_top_k_ratio,
            self.si_near_miss_min_rate,
            self.si_near_miss_rate,
            self.si_route_start_diversity_count,
            self.si_route_start_diversity_pool_size,
            self.si_route_start_diversity_max_relative_gap,
            self.si_local_search_backend,
            self.si_max_local_moves,
            self.si_move_types,
            self.si_perturbation_restarts,
            self.si_use_dual_pruning,
            VECTORIZED_LOCAL_SEARCH_ENABLED,
            self.rl_reward_mode,
            self.expert_replay_fraction,
            self.expert_replay_capacity,
        )

    def _write_reference_validation_manifest(self) -> None:
        payload = {
            "seed": self.reference_validation_seed,
            "directory": str(self.reference_validation_dir.resolve()),
            "evaluation_protocol": self.reference_validation_protocol,
            **self.reference_validation_summary,
            "files": {
                str(size): str((self.reference_validation_dir / f"J{size}.pt").resolve())
                for size in self.reference_validation_sizes
            },
        }
        (self.output_dir / "reference_validation_instances.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def _strict_resume_signature(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "planned_epochs": int(self.epochs),
            "env_params": copy.deepcopy(self.env_params),
            "model_config": asdict(self.model_config),
            "optimizer_params": copy.deepcopy(self.optimizer_params),
            "trajectory_params": {
                key: copy.deepcopy(self.trainer_params.get(key))
                for key in self._STRICT_TRAJECTORY_PARAMETER_KEYS
            },
        }

    def _restore_joint_checkpoint(self) -> None:
        checkpoint_path = self.trainer_params.get("checkpoint_path")
        if not checkpoint_path or not Path(checkpoint_path).exists():
            self.best_score = float("inf")
            return
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        joint_metadata = checkpoint.get("joint_training")
        if not joint_metadata:
            if self.strict_resume:
                raise KeyError(
                    "strict resume requires a joint schema-v3 checkpoint; "
                    "plain POMO checkpoints cannot restore joint state"
                )
            # Plain POMO checkpoint: the target and initial reference already
            # contain the loaded weights; joint model selection starts afresh.
            self.best_score = float("inf")
            self.joint_origin_epoch = self.start_epoch - 1
            return
        require_matching_validation_protocol(checkpoint, self.reference_validation_candidates)
        schema_version = int(joint_metadata.get("schema_version", 0))
        if schema_version != 3:
            raise ValueError(
                f"unsupported joint checkpoint schema {schema_version}; "
                "resume from a MACSIM-style schema-v3 checkpoint"
            )
        saved_edge_hidden_dim = int(joint_metadata.get("edge_hidden_dim", -1))
        saved_edge_dropout = float(joint_metadata.get("edge_dropout", float("nan")))
        if saved_edge_hidden_dim != self.edge_hidden_dim or not np.isclose(
            saved_edge_dropout, self.edge_dropout, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "joint checkpoint edge-decoder architecture is incompatible with the "
                "current configuration: "
                f"checkpoint=(hidden={saved_edge_hidden_dim}, dropout={saved_edge_dropout}), "
                f"current=(hidden={self.edge_hidden_dim}, dropout={self.edge_dropout})"
            )

        if self.strict_resume:
            _require_checkpoint_keys(
                checkpoint,
                {
                    "edge_decoder_state_dict",
                    "edge_optimizer_state_dict",
                    "edge_global_step",
                    "edge_scheduler_config",
                    "edge_scheduler_state_dict",
                    "reference_policy_state_dict",
                    "reference_state",
                    "reference_validation_summary",
                    "joint_batch_index",
                    "expert_replay_pool",
                    "joint_rng_state",
                    "strict_resume_signature",
                },
                context="joint training state",
            )
            if (
                self.rl_reward_mode == "hybrid_pomo_exact_expert"
                or self.si_adaptive_near_miss
            ):
                _require_checkpoint_keys(
                    checkpoint,
                    {"adaptive_training_state"},
                    context="adaptive joint training state",
                )
            saved_signature = checkpoint["strict_resume_signature"]
            if not isinstance(saved_signature, dict):
                raise TypeError("strict resume signature must be a dictionary")
            current_signature = self._strict_resume_signature()
            if saved_signature != current_signature:
                differing_sections = sorted(
                    key
                    for key in set(saved_signature) | set(current_signature)
                    if saved_signature.get(key) != current_signature.get(key)
                )
                raise ValueError(
                    "strict resume trajectory configuration mismatch in sections: "
                    f"{differing_sections}"
                )
            self._strict_saved_reference_validation_summary = copy.deepcopy(
                checkpoint["reference_validation_summary"]
            )

        self.joint_origin_epoch = int(
            joint_metadata.get("origin_epoch", int(checkpoint.get("epoch", 0)))
        )
        if "edge_decoder_state_dict" in checkpoint:
            self.edge_decoder.load_state_dict(checkpoint["edge_decoder_state_dict"], strict=True)
        if "edge_optimizer_state_dict" in checkpoint:
            self.edge_optimizer.load_state_dict(checkpoint["edge_optimizer_state_dict"])
        saved_edge_global_step = checkpoint.get("edge_global_step")
        saved_edge_scheduler_config = checkpoint.get("edge_scheduler_config")
        if saved_edge_global_step is not None:
            if (
                saved_edge_scheduler_config is not None
                and dict(saved_edge_scheduler_config) != self.edge_scheduler_params
            ):
                raise ValueError(
                    "checkpoint edge scheduler configuration does not match "
                    "the requested edge scheduler"
                )
            if self.edge_scheduler is None:
                raise ValueError(
                    "checkpoint contains an edge scheduler step but none is configured"
                )
            if "edge_scheduler_state_dict" not in checkpoint:
                raise KeyError(
                    "edge step-scheduler checkpoint is missing edge_scheduler_state_dict"
                )
            self.edge_scheduler.load_state_dict(
                checkpoint["edge_scheduler_state_dict"]
            )
            self.edge_global_step = int(saved_edge_global_step)
            if int(self.edge_scheduler.last_epoch) != self.edge_global_step:
                raise ValueError(
                    "checkpoint edge scheduler/global_step mismatch: "
                    f"last_epoch={self.edge_scheduler.last_epoch}, "
                    f"edge_global_step={self.edge_global_step}"
                )
            if self.strict_resume:
                _assert_scheduler_optimizer_lr_match(
                    self.edge_optimizer,
                    self.edge_scheduler,
                )
        if "reference_policy_state_dict" in checkpoint:
            self.reference_policy.load_state_dict(
                checkpoint["reference_policy_state_dict"], strict=True
            )
            freeze_reference_policy(self.reference_policy)
        self.reference_state = ReferenceState.from_dict(checkpoint.get("reference_state"))
        self._joint_batch_index = int(checkpoint.get("joint_batch_index", 0))
        adaptive_state = checkpoint.get("adaptive_training_state", {})
        if adaptive_state:
            self._previous_epoch_il_accuracy = float(
                adaptive_state.get(
                    "previous_epoch_il_accuracy", self.competence_il_start
                )
            )
            self._previous_epoch_target_better_ratio = float(
                adaptive_state.get("previous_epoch_target_better_ratio", 0.0)
            )
        if self.expert_replay_capacity > 0:
            restored_replay: list[tuple[Any, list[int]]] = []
            for record in checkpoint.get("expert_replay_pool", []):
                instance = record["instance"]
                expert = list(validate_tour(record["expert_sequence"], instance.J))
                restored_replay.append((instance, expert))
            self._expert_replay_pool = restored_replay[-self.expert_replay_capacity :]
        self._restore_rng_state(checkpoint.get("joint_rng_state"))
        logger.info(
            "Restored sparse joint state: batch={} reference_promotions={} "
            "validation_rounds={} replay_pool={}",
            self._joint_batch_index,
            self.reference_state.promotions,
            self.reference_state.validation_rounds,
            len(self._expert_replay_pool),
        )

    def _capture_rng_state(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "python_random": random.getstate(),
            "numpy_global": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "problem_size_generator": self._problem_size_rng.get_state(),
            "online_generator_bit_state": copy.deepcopy(
                self.generator.rng.bit_generator.state
            ),
            "online_generator_instance_counter": int(self.generator.instance_counter),
        }
        if torch.cuda.is_available():
            payload["torch_cuda_all"] = torch.cuda.get_rng_state_all()
        return payload

    def _restore_rng_state(self, payload: dict[str, Any] | None) -> None:
        if not payload:
            if self.strict_resume:
                raise KeyError("strict resume checkpoint is missing joint_rng_state")
            logger.warning(
                "Joint checkpoint has no RNG state; model/optimizer state is restored, "
                "but data and sampling continuation cannot be bitwise exact"
            )
            return
        required = {
            "python_random",
            "numpy_global",
            "torch_cpu",
            "problem_size_generator",
            "online_generator_bit_state",
            "online_generator_instance_counter",
        }
        if torch.cuda.is_available():
            required.add("torch_cuda_all")
        if self.strict_resume:
            _require_checkpoint_keys(payload, required, context="RNG state")
            if (
                torch.cuda.is_available()
                and len(payload["torch_cuda_all"]) != torch.cuda.device_count()
            ):
                raise ValueError(
                    "strict resume CUDA RNG topology mismatch: "
                    f"checkpoint={len(payload['torch_cuda_all'])} visible devices, "
                    f"current={torch.cuda.device_count()}"
                )
        random.setstate(payload["python_random"])
        np.random.set_state(payload["numpy_global"])
        torch.set_rng_state(payload["torch_cpu"].cpu())
        self._problem_size_rng.set_state(payload["problem_size_generator"].cpu())
        self.generator.rng.bit_generator.state = copy.deepcopy(
            payload["online_generator_bit_state"]
        )
        self.generator.instance_counter = int(
            payload["online_generator_instance_counter"]
        )
        if "torch_cuda_all" in payload and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(
                    [state.cpu() for state in payload["torch_cuda_all"]]
                )
            except RuntimeError as exc:
                if self.strict_resume:
                    raise RuntimeError(
                        "strict resume could not restore CUDA RNG states"
                    ) from exc
                logger.warning("Could not restore CUDA RNG states: {}", exc)

    def _save_checkpoint(self, epoch: int, metrics: dict[str, Any], path: Path) -> None:
        save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            best_metric=self.best_score,
            metrics=metrics,
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload.update(
            {
                "model_config": asdict(self.model_config),
                "edge_decoder_state_dict": self.edge_decoder.state_dict(),
                "edge_optimizer_state_dict": self.edge_optimizer.state_dict(),
                "edge_global_step": int(self.edge_global_step),
                "edge_scheduler_config": dict(self.edge_scheduler_params),
                "global_step": int(self.global_step),
                "main_scheduler_config": dict(self.scheduler_params),
                "reference_policy_state_dict": self.reference_policy.state_dict(),
                "reference_state": self.reference_state.to_dict(),
                "reference_validation_summary": dict(self.reference_validation_summary),
                "reference_validation_protocol": dict(self.reference_validation_protocol),
                "reference_validation_seed": self.reference_validation_seed,
                "joint_batch_index": self._joint_batch_index,
                "joint_rng_state": self._capture_rng_state(),
                "adaptive_training_state": {
                    "previous_epoch_il_accuracy": float(
                        self._previous_epoch_il_accuracy
                    ),
                    "previous_epoch_target_better_ratio": float(
                        self._previous_epoch_target_better_ratio
                    ),
                },
                "strict_resume_signature": self._strict_resume_signature(),
                "expert_replay_pool": [
                    {"instance": instance, "expert_sequence": list(expert)}
                    for instance, expert in self._expert_replay_pool
                ],
                "joint_training": serialize_joint_checkpoint_metadata(
                    edge_hidden_dim=self.edge_hidden_dim,
                    edge_dropout=self.edge_dropout,
                    extra={
                        "origin_epoch": self.joint_origin_epoch,
                        "anchor_selection": "lower_level_predictor",
                        "training_gurobi_budget": (
                            "one_anchor_plus_one_target_plus_optional_one_expert"
                        ),
                        "rollout_gurobi_calls_per_instance": 1,
                        "reward": {
                            "exact_target_expert": "negative_exact_selected_target",
                            "hybrid_pomo": (
                                "negative_hybrid_one_exact_target_plus_upper_bounds"
                            ),
                            "hybrid_pomo_exact_expert": (
                                "standardized_hybrid_pomo_plus_bounded_exact_expert"
                            ),
                        }[self.rl_reward_mode],
                        "baseline": {
                            "exact_target_expert": "negative_exact_expert_makespan",
                            "hybrid_pomo": "per_instance_pomo_mean_reward",
                            "hybrid_pomo_exact_expert": (
                                "pomo_mean_plus_exact_expert_auxiliary"
                            ),
                        }[self.rl_reward_mode],
                        "rl_reward_mode": self.rl_reward_mode,
                        "policy_updates": "separate_reinforce_then_teacher_forced_cross_entropy",
                        "changed_expert_il_weight": self.changed_expert_il_weight,
                        "expert_replay_capacity": self.expert_replay_capacity,
                        "expert_replay_fraction": self.expert_replay_fraction,
                        "expert_replay_pool_size": len(self._expert_replay_pool),
                        "edge_update": "independent_detached_encoder_directed_huber",
                        "reference_update": "fixed_exact_validation_strict_promotion",
                        "top_k": self.si_route_start_top_k,
                        "top_k_mode": (
                            "ceil_problem_size_ratio"
                            if self.si_route_start_top_k_ratio > 0.0
                            else "fixed"
                        ),
                        "top_k_ratio": self.si_route_start_top_k_ratio,
                        "top_k_min": self.si_route_start_top_k_min,
                        "top_k_diversity_count": self.si_route_start_diversity_count,
                        "top_k_diversity_pool_size": (
                            self.si_route_start_diversity_pool_size
                        ),
                        "top_k_diversity_max_relative_gap": (
                            self.si_route_start_diversity_max_relative_gap
                        ),
                        "actual_top_k_mean": float(
                            metrics.get("actual_top_k", self.si_route_start_top_k)
                        ),
                        "actual_top_k_ratio_mean": float(
                            metrics.get("actual_top_k_ratio", 0.0)
                        ),
                        "near_miss_rate": float(
                            metrics.get("near_miss_rate", self.si_near_miss_rate)
                        ),
                        "near_miss_rate_range": [
                            self.si_near_miss_min_rate,
                            self.si_near_miss_rate,
                        ],
                        "near_miss_selection": "lowest_raw_relative_gap_stable",
                        "near_miss_eligible_count": float(
                            metrics.get("near_miss_eligible_count", 0.0)
                        ),
                        "near_miss_attempt_count": float(
                            metrics.get("near_miss_attempt_count", 0.0)
                        ),
                        "near_miss_accept_count": float(
                            metrics.get("near_miss_accept_count", 0.0)
                        ),
                        "near_miss_accept_rate": float(
                            metrics.get("near_miss_accept_rate", 0.0)
                        ),
                        "near_miss_mean_eligible_relative_gap": float(
                            metrics.get("near_miss_mean_eligible_relative_gap", 0.0)
                        ),
                        "near_miss_mean_attempt_relative_gap": float(
                            metrics.get("near_miss_mean_attempt_relative_gap", 0.0)
                        ),
                        "near_miss_mean_accept_relative_gap": float(
                            metrics.get("near_miss_mean_accept_relative_gap", 0.0)
                        ),
                        "reference_validation_instances": self.reference_validation_count,
                        "reference_validation_sizes": list(self.reference_validation_sizes),
                        "reference_validation_per_size": self.reference_validation_per_size,
                        "max_local_moves": self.si_max_local_moves,
                        "move_types": list(self.si_move_types),
                        "vectorized_local_search": (
                            VECTORIZED_LOCAL_SEARCH_ENABLED
                        ),
                        "perturbation_restarts": self.si_perturbation_restarts,
                        "dual_pruning": self.si_use_dual_pruning,
                        "cvxpylayer": False,
                        "rebar": False,
                    },
                ),
            }
        )
        if self.scheduler is not None:
            payload["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.edge_scheduler is not None:
            payload["edge_scheduler_state_dict"] = self.edge_scheduler.state_dict()
        torch.save(payload, path)

    def _save_reference_snapshot(self, epoch: int, metrics: dict[str, Any]) -> None:
        torch.save(
            {
                "epoch": int(epoch),
                "model_state_dict": self.reference_policy.state_dict(),
                "model_config": asdict(self.model_config),
                "reference_state": self.reference_state.to_dict(),
                "validation_metrics": dict(metrics),
                "validation_summary": dict(self.reference_validation_summary),
                "validation_protocol": dict(self.reference_validation_protocol),
            },
            self.output_dir / "reference.pt",
        )

    def close(self) -> None:
        if self._si_executor is not None:
            self._shutdown_si_executor(self._si_executor)
            self._si_executor = None
        super().close()

    def _new_si_executor(self) -> ProcessPoolExecutor:
        import multiprocessing as mp

        return ProcessPoolExecutor(
            max_workers=self.si_parallel_workers,
            mp_context=mp.get_context("spawn"),
        )

    @staticmethod
    def _shutdown_si_executor(executor: ProcessPoolExecutor) -> None:
        # A native Gurobi worker crash marks ProcessPoolExecutor as broken. On
        # Python 3.10 its normal shutdown path can leave surviving spawned
        # workers re-parented to PID 1. Terminate only this executor's children.
        if bool(getattr(executor, "_broken", False)):
            processes = getattr(executor, "_processes", None) or {}
            for process in processes.values():
                if process.is_alive():
                    process.terminate()
        executor.shutdown(wait=True, cancel_futures=True)

    def _restart_si_executor(self) -> None:
        if self._si_executor is not None:
            self._shutdown_si_executor(self._si_executor)
        self._si_executor = self._new_si_executor()
        self._si_executor_restarts += 1

    def _rollout_target(self) -> tuple[RolloutResult, torch.Tensor]:
        """Roll out target POMO once and retain per-step log probabilities."""

        state, _, done = self.env.pre_step()
        log_prob_steps: list[torch.Tensor] = []
        while not done:
            selected_count_before = state.selected_count
            use_block = (
                int(getattr(self.model.config, "block_size", 1)) > 1
                and selected_count_before >= 2
            )
            if use_block:
                selected_block, probabilities, _, _, _ = self.model.forward_block(
                    state,
                    decode_type="sample",
                    use_pomo_start=True,
                )
                for slot in range(selected_block.size(2)):
                    state, _, done = self.env.step(selected_block[:, :, slot])
                    log_prob_steps.append(
                        probabilities[:, :, slot].clamp_min(1e-12).log()
                    )
            else:
                selected, probability = self.model(
                    state, decode_type="sample", use_pomo_start=True
                )
                state, _, done = self.env.step(selected)
                if selected_count_before >= 1:
                    log_prob_steps.append(probability.clamp_min(1e-12).log())
        if not log_prob_steps:
            raise RuntimeError("target rollout produced no route decisions")
        raw_node_indices = self.env.selected_node_list
        if raw_node_indices is None:
            raise RuntimeError("target environment lost its selected route")
        rollout = RolloutResult(
            sequences=raw_node_indices[:, :, 1:] - 1,
            log_probs=torch.stack(log_prob_steps, dim=2).sum(dim=2),
            raw_node_indices=raw_node_indices,
        )
        return rollout, torch.stack(log_prob_steps, dim=2)

    def _predict_contribution_matrices(self, instances) -> list[np.ndarray]:
        if self.model.encoded_nodes is None:
            raise RuntimeError("target POMO encoder state is unavailable")
        matrices: list[np.ndarray] = []
        self.edge_decoder.eval()
        with torch.no_grad():
            encoded = self.model.encoded_nodes.detach()
            for start in range(0, len(instances), self.edge_batch_size):
                stop = min(start + self.edge_batch_size, len(instances))
                normalized = self.edge_decoder(encoded[start:stop]).cpu().numpy()
                for offset, matrix in enumerate(normalized):
                    matrices.append(
                        denormalize_contribution_matrix(matrix, instances[start + offset])
                    )
        self.edge_decoder.train()
        return matrices

    def _generate_reference_candidates(self, instances) -> list[list[list[int]]]:
        """Generate exactly beta sampled Reference routes for every instance."""

        if not instances:
            raise ValueError("reference candidate generation requires instances")
        with torch.inference_mode():
            return generate_candidate_pool_batch(
                self.reference_policy,
                instances,
                device=self.device,
                num_candidates=self.reference_candidate_count,
                decode_type="sample",
            )

    @staticmethod
    def _geometric_fallback_matrix(instance) -> np.ndarray:
        nodes = np.concatenate((instance.depot[None, :], instance.targets), axis=0)
        distances = np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=-1)
        return distances / max(float(instance.carrier_speed), 1e-12)

    def _select_anchor_sequences(
        self,
        instances,
        reference_sequences: list[list[list[int]]],
        predicted_matrices: list[np.ndarray],
    ) -> tuple[list[list[int]], list[float], str]:
        use_predictor = self._joint_batch_index >= self.si_predictor_warmup_batches
        anchors: list[list[int]] = []
        anchor_scores: list[float] = []
        for instance, routes, prediction in zip(
            instances, reference_sequences, predicted_matrices
        ):
            if not routes:
                raise RuntimeError(f"reference produced no route for {instance.instance_id}")
            ranking_matrix = (
                prediction if use_predictor else self._geometric_fallback_matrix(instance)
            )
            scores = [directed_route_upper_bound(route, ranking_matrix) for route in routes]
            best_index = int(np.argmin(np.asarray(scores, dtype=float)))
            anchors.append(list(routes[best_index]))
            anchor_scores.append(float(scores[best_index]))
        return anchors, anchor_scores, (
            "lower_level_predictor" if use_predictor else "geometry_warmup"
        )

    def _route_start_top_k(self, problem_size: int) -> int:
        """Resolve the per-instance local-search breadth.

        A positive ratio takes precedence over the fixed compatibility value,
        while the optional minimum prevents small instances from losing the
        established search breadth.
        """

        if problem_size <= 0:
            raise ValueError("problem_size must be positive")
        if self.si_route_start_top_k_ratio > 0.0:
            return max(
                1,
                self.si_route_start_top_k_min,
                int(math.ceil(self.si_route_start_top_k_ratio * problem_size)),
            )
        return self.si_route_start_top_k

    def _imitation_inner_epochs_for_epoch(self, epoch: int) -> int:
        if epoch <= 0:
            raise ValueError("epoch must be positive")
        active = self.imitation_inner_epochs_schedule[0][1]
        for start_epoch, inner_epochs in self.imitation_inner_epochs_schedule:
            if epoch < start_epoch:
                break
            active = inner_epochs
        return int(active)

    @staticmethod
    def _unit_interval(value: float) -> float:
        return min(max(float(value), 0.0), 1.0)

    def _il_competence(self) -> float:
        return self._unit_interval(
            (
                float(self._previous_epoch_il_accuracy)
                - self.competence_il_start
            )
            / (self.competence_il_full - self.competence_il_start)
        )

    def _exact_expert_aux_weight(self) -> float:
        """Increase the exact auxiliary only after IL leaves cold start."""

        return float(self.exact_expert_aux_max_weight * self._il_competence())

    def _effective_near_miss_rate(self) -> float:
        if not self.si_adaptive_near_miss:
            return float(self.si_near_miss_rate)
        target_competence = self._unit_interval(
            float(self._previous_epoch_target_better_ratio)
            / self.competence_target_better_full
        )
        competence = 0.5 * (self._il_competence() + target_competence)
        return float(
            self.si_near_miss_min_rate
            + competence * (self.si_near_miss_rate - self.si_near_miss_min_rate)
        )

    @staticmethod
    def _is_near_miss_eligible(result: SparseSelfImprovementResult) -> bool:
        candidate = result.near_miss_candidate_sequence
        return bool(
            not result.expert_changed
            and result.expert_gurobi_calls == 0
            and candidate is not None
            and tuple(candidate) != tuple(result.anchor_sequence)
            and np.isfinite(float(result.near_miss_relative_gap))
        )

    @classmethod
    def _select_near_miss_indices(
        cls,
        results: list[SparseSelfImprovementResult],
        *,
        batch_size: int,
        rate: float,
    ) -> tuple[list[int], list[int]]:
        if batch_size < 0:
            raise ValueError("batch_size must be non-negative")
        if not 0.0 <= rate <= 1.0:
            raise ValueError("near-miss rate must be in [0, 1]")
        eligible_indices = [
            index
            for index, result in enumerate(results)
            if cls._is_near_miss_eligible(result)
        ]
        # Python's sort is stable, so equal raw gaps retain instance order.
        eligible_indices.sort(
            key=lambda index: float(results[index].near_miss_relative_gap)
        )
        attempt_limit = int(math.ceil(rate * batch_size)) if rate > 0.0 else 0
        return eligible_indices, eligible_indices[: min(attempt_limit, len(eligible_indices))]

    def _run_sparse_self_improvement(
        self,
        instances,
        reference_sequences: list[list[list[int]]],
        anchor_sequences: list[list[int]],
    ) -> tuple[list[SparseSelfImprovementResult], float, int]:
        tasks = [
            SparseSelfImprovementTask(
                instance=instance,
                sampled_sequences=[list(route) for route in routes],
                anchor_sequence=list(anchor),
                anchor_solver_config=self.si_anchor_solver_config,
                final_solver_config=self.si_final_solver_config,
                local_search_config=self.si_local_search_config,
                route_start_top_k=self._route_start_top_k(instance.J),
                route_start_diversity_count=self.si_route_start_diversity_count,
                route_start_diversity_pool_size=(
                    self.si_route_start_diversity_pool_size
                ),
                route_start_diversity_max_relative_gap=(
                    self.si_route_start_diversity_max_relative_gap
                ),
                use_dual_pruning=self.si_use_dual_pruning,
                upper_bound_tolerance=self.si_upper_bound_tolerance,
                feasibility_tolerance=self.si_feasibility_tolerance,
                perturbation_restarts=self.si_perturbation_restarts,
                local_search_backend=self.si_local_search_backend,
                lkh_binary=self.si_lkh_binary,
                lkh_budget_coefficient=self.si_lkh_budget_coefficient,
                lkh_desired_scale=self.si_lkh_desired_scale,
                lkh_trials=self.si_lkh_trials,
                lkh_max_candidates=self.si_lkh_max_candidates,
                lkh_move_type=self.si_lkh_move_type,
                lkh_native_seconds=self.si_lkh_native_seconds,
                lkh_process_timeout_seconds=self.si_lkh_process_timeout_seconds,
                lkh_work_root=str(self.si_lkh_work_root),
                lkh_start_fraction=float(getattr(self, "si_lkh_start_fraction", 0.0)),
                lkh_budget_starts=int(getattr(self, "si_lkh_budget_starts", 0)),
                local_search_seed=(
                    int(self.trainer_params.get("seed", 0))
                    + self._joint_batch_index * 100_003
                    + task_index * 101
                ) % 2_000_000_000,
            )
            for task_index, (instance, routes, anchor) in enumerate(zip(
                instances, reference_sequences, anchor_sequences
            ))
        ]
        started = time.perf_counter()
        if self._si_executor is None:
            results = [run_sparse_self_improvement_task(task) for task in tasks]
        else:
            # Gurobi's native library can very rarely terminate a spawned
            # worker without a Python exception. Rebuild the pool and replay
            # the same pure task list so an isolated native crash does not
            # discard the epoch. No model/RNG state changes inside these tasks.
            max_pool_retries = 2
            for attempt in range(max_pool_retries + 1):
                try:
                    # map() queues every task eagerly; extra exact solves are
                    # queued behind them before blocking on the results.
                    iterator = self._si_executor.map(
                        run_sparse_self_improvement_task,
                        tasks,
                        chunksize=1,
                    )
                    self._run_after_sparse_submit()
                    results = list(iterator)
                    break
                except BrokenProcessPool:
                    if attempt >= max_pool_retries:
                        raise
                    logger.warning(
                        "Self-improvement worker pool crashed; rebuilding and "
                        "retrying the same batch ({}/{})",
                        attempt + 1,
                        max_pool_retries,
                    )
                    self._restart_si_executor()

        # Phase B is deliberately coordinated only after every Phase-A task has
        # returned.  This makes selection a true batch-level lowest-gap policy
        # and avoids nested futures inside ProcessPoolExecutor workers.
        if self._active_near_miss_rate == 0.0:
            # Formal MACSIM-style mode uses only strict U0-certified Experts.
            eligible_indices, selected_indices = [], []
        else:
            eligible_indices, selected_indices = self._select_near_miss_indices(
                results,
                batch_size=len(instances),
                rate=self._active_near_miss_rate,
            )
        phase_b_solutions: dict[int, Any] = {}
        if selected_indices:
            validation_tasks = [
                NearMissValidationTask(
                    instance=instances[index],
                    candidate_sequence=list(
                        results[index].near_miss_candidate_sequence or []
                    ),
                    solver_config=self.si_final_solver_config,
                )
                for index in selected_indices
            ]
            if self._si_executor is None:
                validation_solutions = [
                    run_near_miss_validation_task(task) for task in validation_tasks
                ]
            else:
                validation_solutions = list(
                    self._si_executor.map(
                        run_near_miss_validation_task,
                        validation_tasks,
                        chunksize=1,
                    )
                )
            phase_b_solutions = dict(zip(selected_indices, validation_solutions))
            for index, solution in zip(selected_indices, validation_solutions):
                result = results[index]
                results[index] = apply_near_miss_validation(
                    result,
                    instances[index],
                    solution,
                    acceptance_epsilon=self.si_local_search_config.acceptance_epsilon(
                        result.anchor_exact_makespan
                    ),
                    upper_bound_tolerance=self.si_upper_bound_tolerance,
                    feasibility_tolerance=self.si_feasibility_tolerance,
                )
        if int(getattr(self, "si_c0_refresh_rounds", 0)) > 0:
            self._run_c0_refresh(
                instances,
                reference_sequences,
                results,
                phase_b_solutions,
            )
        wall_runtime = time.perf_counter() - started
        if len(results) != len(instances):
            raise RuntimeError("sparse self-improvement returned the wrong result count")
        total_calls = sum(result.total_gurobi_calls for result in results)
        if total_calls > 2 * len(instances):
            raise RuntimeError(
                f"batch Gurobi budget exceeded: {total_calls} > {2 * len(instances)}"
            )
        if any(result.total_gurobi_calls > 2 for result in results):
            raise RuntimeError("per-instance Gurobi budget exceeded")
        return results, wall_runtime, len(eligible_indices)

    def _run_c0_refresh(self, instances, reference_sequences, results, phase_b_solutions) -> None:
        """Run the configured C0-refresh rounds after the sparse phase.

        Only instances whose sparse phase made exactly one expert-side solve
        (certified or near-miss) take part: that solve supplies the first
        refresh base, and an accepted refresh keeps the one-expert-call
        invariant required by ``expert_upper_bound_matrix``. Refresh solves are
        counted in ``c0_refresh_gurobi_calls``, outside the <=2-call budget.
        """

        states: dict[int, dict[str, Any]] = {}
        for index, result in enumerate(results):
            if result.final_validation_failed or result.expert_gurobi_calls != 1:
                continue
            if result.near_miss_attempted:
                base = phase_b_solutions.get(index)
                first = list(result.near_miss_candidate_sequence or [])
            else:
                base = result.expert_solution
                first = list(result.expert_sequence)
            if base is None or not is_optimal_gurobi_solution(base):
                continue
            states[index] = {
                "base": base,
                "pool": [list(route) for route in reference_sequences[index]]
                + [list(search.final_sequence) for search in result.local_search_results],
                "solved": [list(result.anchor_sequence), first],
            }
        for _ in range(self.si_c0_refresh_rounds):
            active = sorted(states)
            if not active:
                break
            tasks = [
                C0RefreshTask(
                    instance=instances[index],
                    base_solution=states[index]["base"],
                    pool_sequences=states[index]["pool"],
                    solved_sequences=states[index]["solved"],
                    incumbent_exact_makespan=float(results[index].expert_exact_makespan),
                    final_solver_config=self.si_final_solver_config,
                    local_search_config=self.si_local_search_config,
                    route_start_top_k=self._route_start_top_k(instances[index].J),
                    route_start_diversity_count=self.si_route_start_diversity_count,
                    route_start_diversity_pool_size=self.si_route_start_diversity_pool_size,
                    route_start_diversity_max_relative_gap=(
                        self.si_route_start_diversity_max_relative_gap
                    ),
                    feasibility_tolerance=self.si_feasibility_tolerance,
                    upper_bound_tolerance=self.si_upper_bound_tolerance,
                    return_matrices=bool(getattr(self, "rl_multi_c0_bound", False)),
                )
                for index in active
            ]
            if self._si_executor is None:
                outcomes = [run_c0_refresh_task(task) for task in tasks]
            else:
                outcomes = list(self._si_executor.map(run_c0_refresh_task, tasks, chunksize=1))
            for index, outcome in zip(active, outcomes):
                result = results[index]
                state = states[index]
                result.c0_refresh_rounds_run += 1
                result.c0_refresh_runtime += float(outcome.search_runtime + outcome.solve_time)
                for table in (outcome.base_matrix, outcome.solution_matrix):
                    if table is not None:
                        result.extra_upper_bound_matrices.append(table)
                state["pool"].extend(outcome.final_sequences)
                if outcome.candidate_sequence is None:
                    del states[index]
                    continue
                result.c0_refresh_gurobi_calls += 1
                result.gurobi_runtime += float(outcome.solve_time)
                state["solved"].append(list(outcome.candidate_sequence))
                if outcome.solve_failed or outcome.solution is None:
                    del states[index]
                    continue
                state["base"] = outcome.solution
                if outcome.accepted:
                    result.c0_refresh_accepted += 1
                    result.expert_sequence = list(outcome.candidate_sequence)
                    result.expert_upper_bound = float(outcome.candidate_upper_bound)
                    result.expert_solution = outcome.solution
                    result.expert_exact_makespan = float(outcome.solution.objective)
                    result.supervision_upper_bound_matrix = outcome.supervision_matrix
                    result.expert_source = "c0_refresh"

    def _step_main_optimizer(self) -> None:
        """Advance the shared RL/IL optimizer and count the parameter update."""

        self.optimizer.step()
        self.global_step = int(getattr(self, "global_step", 0)) + 1
        scheduler = getattr(self, "scheduler", None)
        if scheduler is not None and getattr(self, "scheduler_interval", "step") == "step":
            scheduler.step()

    def _clip_main_gradients(self):
        return torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

    def _step_edge_optimizer(self) -> None:
        """Advance the predictor optimizer without touching the policy schedule."""

        self.edge_optimizer.step()
        self.edge_global_step = int(getattr(self, "edge_global_step", 0)) + 1
        edge_scheduler = getattr(self, "edge_scheduler", None)
        if edge_scheduler is not None:
            edge_scheduler.step()

    def _solve_target_reward_routes(
        self,
        instances: list[Any],
        target_sequences: list[list[list[int]]],
        target_upper_bounds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float]:
        """Replace one best predicted Target bound per instance by its exact value."""

        bounds = np.asarray(target_upper_bounds, dtype=np.float32)
        if bounds.ndim != 2 or bounds.shape[0] != len(instances):
            raise ValueError("Target upper-bound matrix has an invalid batch shape")
        if len(target_sequences) != len(instances):
            raise ValueError("Target route batch and instance batch have different lengths")
        selected_indices = np.argmin(bounds, axis=1).astype(np.int64)
        tasks = [
            TargetRewardSolveTask(
                instance=instance,
                target_sequence=list(target_sequences[index][int(selected_indices[index])]),
                solver_config=self.si_final_solver_config,
            )
            for index, instance in enumerate(instances)
        ]
        if self._si_executor is None:
            solutions = [run_target_reward_solve_task(task) for task in tasks]
        else:
            solutions = list(
                self._si_executor.map(
                    run_target_reward_solve_task,
                    tasks,
                    chunksize=1,
                )
            )

        reward_costs = bounds.copy()
        exact_objectives = np.full(len(instances), np.nan, dtype=np.float64)
        failures = 0
        solver_runtime = 0.0
        for index, (instance, task, solution) in enumerate(
            zip(instances, tasks, solutions)
        ):
            solve_time = float(solution.solve_time)
            if np.isfinite(solve_time) and solve_time > 0.0:
                solver_runtime += solve_time
            if not is_optimal_gurobi_solution(solution) or not np.isfinite(
                float(solution.objective)
            ):
                failures += 1
                continue
            solved_sequence = list(validate_tour(solution.sequence, instance.J))
            if tuple(solved_sequence) != tuple(task.target_sequence):
                raise RuntimeError(
                    f"exact Target solve returned a different route for {instance.instance_id}"
                )
            route_index = int(selected_indices[index])
            exact_objective = float(solution.objective)
            certified_upper_bound = float(bounds[index, route_index])
            if exact_objective > certified_upper_bound + self.si_upper_bound_tolerance:
                raise RuntimeError(
                    "exact Target objective exceeds its certified upper bound: "
                    f"instance={instance.instance_id}, V={exact_objective:.12g}, "
                    f"U={certified_upper_bound:.12g}"
                )
            reward_costs[index, route_index] = exact_objective
            exact_objectives[index] = exact_objective

        return (
            reward_costs,
            selected_indices,
            exact_objectives,
            len(tasks),
            failures,
            float(solver_runtime),
        )

    def _random_subset_size(self, problem_size: int) -> int:
        """Linear m(J) from min_routes at the smallest J to max_routes at the largest."""

        low = int(self.env_params.get("min_problem_size", 20))
        high = int(self.env_params.get("max_problem_size", 100))
        fraction = min(max((int(problem_size) - low) / max(high - low, 1), 0.0), 1.0)
        span = self.random_subset_max_routes - self.random_subset_min_routes
        return int(math.floor(self.random_subset_min_routes + fraction * span + 0.5))

    def _start_random_subset_solves(self, instances, target_sequences) -> dict[str, Any]:
        """Choose m(J) uniform Target routes per instance; solving is submitted later.

        A dedicated generator keyed by (seed, batch index) keeps the global
        torch/numpy/python RNG streams, and hence the training data, unchanged.
        """

        rng = np.random.default_rng(
            [int(self.trainer_params.get("seed", 0)) & 0xFFFFFFFF,
             int(self._joint_batch_index), 0x5EED]
        )
        subsets, tasks, mapping = [], [], []
        for instance, routes in zip(instances, target_sequences):
            count = min(self._random_subset_size(instance.J), len(routes))
            chosen = np.sort(rng.choice(len(routes), size=count, replace=False))
            seen: dict[tuple[int, ...], int] = {}
            positions = []
            for route_index in chosen:
                key = tuple(int(node) for node in routes[int(route_index)])
                if key not in seen:
                    seen[key] = len(tasks)
                    tasks.append(TargetRewardSolveTask(
                        instance=instance,
                        target_sequence=list(key),
                        solver_config=self.si_final_solver_config,
                    ))
                positions.append(seen[key])
            subsets.append(chosen.astype(np.int64))
            mapping.append(positions)
        handle: dict[str, Any] = {
            "subsets": subsets, "tasks": tasks, "mapping": mapping, "futures": None,
        }

        def submit() -> None:
            # Called right after the sparse phase has queued its tasks, so the
            # short exact solves fill workers idle during the search tail.
            if self._si_executor is not None:
                handle["futures"] = [
                    self._si_executor.submit(run_target_reward_solve_task, task)
                    for task in tasks
                ]

        handle["submit"] = submit
        return handle

    def _run_after_sparse_submit(self) -> None:
        hook = getattr(self, "_pending_after_sparse_submit", None)
        self._pending_after_sparse_submit = None
        if hook is not None:
            hook()

    def _collect_random_subset_solves(self, handle: dict[str, Any]):
        tasks = handle["tasks"]
        solutions = None
        if handle["futures"] is not None:
            try:
                solutions = [future.result() for future in handle["futures"]]
            except BrokenProcessPool:
                logger.warning("Worker pool crashed during random-subset Target solves; re-solving")
                self._restart_si_executor()
        if solutions is None:
            if self._si_executor is None:
                solutions = [run_target_reward_solve_task(task) for task in tasks]
            else:
                solutions = list(self._si_executor.map(
                    run_target_reward_solve_task, tasks, chunksize=1
                ))
        objectives = np.full(len(tasks), np.nan, dtype=np.float64)
        failures = 0
        runtime = 0.0
        for index, (task, solution) in enumerate(zip(tasks, solutions)):
            solve_time = float(solution.solve_time)
            if np.isfinite(solve_time) and solve_time > 0.0:
                runtime += solve_time
            if not is_optimal_gurobi_solution(solution) or not np.isfinite(
                float(solution.objective)
            ):
                failures += 1
                continue
            if tuple(validate_tour(solution.sequence, task.instance.J)) != tuple(
                task.target_sequence
            ):
                raise RuntimeError("random-subset exact solve returned a different route")
            objectives[index] = float(solution.objective)
        values = [objectives[np.asarray(positions, dtype=np.int64)] for positions in handle["mapping"]]
        return values, len(tasks), failures, float(runtime)

    def _random_subset_dense_loss(self, route_log_probs: torch.Tensor):
        """Dense POMO term from exact values of the random subset only."""

        handle = self._random_subset_handle
        if handle is None:
            raise RuntimeError("random-subset Target solves were not started for this batch")
        values, calls, failures, runtime = self._collect_random_subset_solves(handle)
        advantages = torch.zeros_like(route_log_probs)
        mask = torch.zeros_like(route_log_probs, dtype=torch.bool)
        used_routes = 0
        used_instances = 0
        for instance_index, (subset, exact) in enumerate(zip(handle["subsets"], values)):
            finite = np.isfinite(exact)
            if int(finite.sum()) < 2:
                continue
            routes = subset[finite]
            costs = exact[finite]
            centered = -(costs - costs.mean())
            scale = max(float(centered.std()), self.hybrid_advantage_epsilon)
            index_tensor = torch.as_tensor(routes, dtype=torch.long, device=self.device)
            advantages[instance_index, index_tensor] = torch.as_tensor(
                centered / scale, dtype=advantages.dtype, device=self.device
            )
            mask[instance_index, index_tensor] = True
            used_routes += len(routes)
            used_instances += 1
        if used_routes:
            # Mean over contributing routes keeps the per-route scale of the
            # original mean over all B*P standardized hybrid advantages.
            loss = -(advantages.detach() * route_log_probs)[mask].sum() / used_routes
        else:
            loss = route_log_probs.sum() * 0.0
        stats = {
            "random_subset_gurobi_calls": int(calls),
            "random_subset_gurobi_failures": int(failures),
            "random_subset_gurobi_runtime": float(runtime),
            "random_subset_routes": int(used_routes),
            "random_subset_instances": int(used_instances),
        }
        return loss, advantages[mask], stats

    def _policy_rl_update(
        self,
        target_step_log_probs: torch.Tensor,
        target_sequences: list[list[list[int]]],
        results: list[SparseSelfImprovementResult],
        instances: list[Any],
    ) -> tuple[float, np.ndarray, np.ndarray, dict[str, Any]]:
        anchor_matrices = [result.anchor_upper_bound_matrix for result in results]
        anchor_step_costs = batched_directed_step_costs(target_sequences, anchor_matrices)
        target_upper_bounds = anchor_step_costs.sum(axis=-1).astype(np.float32)
        for instance_index, result in enumerate(results):
            expert_matrix = result.expert_upper_bound_matrix
            if expert_matrix is None:
                continue
            expert_step_costs = batched_directed_step_costs(
                [target_sequences[instance_index]],
                [expert_matrix],
            )
            expert_upper_bounds = expert_step_costs[0].sum(axis=-1).astype(np.float32)
            # Take the minimum after summing each complete route. Taking an
            # edge-wise minimum could splice two feasible solutions together
            # and would no longer be a certified upper bound.
            target_upper_bounds[instance_index] = np.minimum(
                target_upper_bounds[instance_index],
                expert_upper_bounds,
            )
        extra_tables = 0
        extra_reduction_sum = 0.0
        if getattr(self, "rl_multi_c0_bound", False):
            for instance_index, result in enumerate(results):
                tables = list(getattr(result, "extra_upper_bound_matrices", None) or [])
                if not tables:
                    continue
                routes = target_sequences[instance_index]
                costs = batched_directed_step_costs([routes] * len(tables), tables).sum(axis=-1)
                before = target_upper_bounds[instance_index].copy()
                target_upper_bounds[instance_index] = np.minimum(
                    before, costs.min(axis=0).astype(np.float32)
                )
                extra_tables += len(tables)
                extra_reduction_sum += float(np.mean(
                    (before - target_upper_bounds[instance_index]) / np.maximum(before, 1e-9)
                ))
        (
            target_reward_costs,
            selected_indices,
            exact_objectives,
            target_reward_calls,
            target_reward_failures,
            target_reward_runtime,
        ) = self._solve_target_reward_routes(
            instances,
            target_sequences,
            target_upper_bounds,
        )
        route_log_probs = target_step_log_probs.sum(dim=2)
        reward_mode = getattr(self, "rl_reward_mode", "hybrid_pomo")
        expert_exact = np.asarray(
            [
                float(getattr(result, "expert_exact_makespan", float("nan")))
                for result in results
            ],
            dtype=np.float64,
        )
        exact_advantages = expert_exact - exact_objectives
        exact_valid = np.isfinite(exact_objectives) & np.isfinite(expert_exact)
        exact_valid_count = int(exact_valid.sum())
        exact_advantage_sum = float(exact_advantages[exact_valid].sum())
        exact_positive_advantage_count = int(
            np.sum(exact_advantages[exact_valid] > 0.0)
        )
        selected_tensor = torch.as_tensor(
            selected_indices, dtype=torch.long, device=self.device
        ).view(-1, 1)
        selected_log_probs = route_log_probs.gather(1, selected_tensor).squeeze(1)
        valid_tensor = torch.as_tensor(
            exact_valid, dtype=torch.bool, device=self.device
        )

        hybrid_loss = route_log_probs.sum() * 0.0
        exact_loss = route_log_probs.sum() * 0.0
        exact_aux_weight = 0.0
        subset_stats = {
            "random_subset_gurobi_calls": 0,
            "random_subset_gurobi_failures": 0,
            "random_subset_gurobi_runtime": 0.0,
            "random_subset_routes": 0,
            "random_subset_instances": 0,
        }
        if getattr(self, "dense_reward_source", "hybrid_upper_bound") == "random_subset_exact":
            hybrid_loss, hybrid_advantages, subset_stats = self._random_subset_dense_loss(
                route_log_probs
            )
        elif reward_mode in {"hybrid_pomo", "hybrid_pomo_exact_expert"}:
            target_rewards = torch.as_tensor(
                -target_reward_costs, dtype=torch.float32, device=self.device
            )
            pomo_mean_baselines = target_rewards.mean(dim=1, keepdim=True)
            hybrid_advantages = target_rewards - pomo_mean_baselines
            if reward_mode == "hybrid_pomo_exact_expert":
                # Normalize within each instance so a difficult random-start
                # batch cannot dominate merely because its costs are larger.
                scale = hybrid_advantages.std(
                    dim=1, keepdim=True, unbiased=False
                ).clamp_min(self.hybrid_advantage_epsilon)
                hybrid_advantages = hybrid_advantages / scale
            if route_log_probs.shape != hybrid_advantages.shape:
                raise RuntimeError(
                    "route log-probability and scalar advantage shapes differ: "
                    f"{tuple(route_log_probs.shape)} != "
                    f"{tuple(hybrid_advantages.shape)}"
                )
            hybrid_loss = -(
                hybrid_advantages.detach() * route_log_probs
            ).mean()

        if reward_mode in {"exact_target_expert", "hybrid_pomo_exact_expert"}:
            if exact_valid_count:
                if reward_mode == "hybrid_pomo_exact_expert":
                    relative_advantages = exact_advantages / np.maximum(
                        np.abs(expert_exact), 1e-12
                    )
                    relative_advantages = np.clip(
                        relative_advantages,
                        -self.exact_expert_advantage_clip,
                        self.exact_expert_advantage_clip,
                    )
                    exact_aux_weight = self._exact_expert_aux_weight()
                    advantage_values = relative_advantages
                else:
                    exact_aux_weight = 1.0
                    advantage_values = exact_advantages
                advantage_tensor = torch.as_tensor(
                    advantage_values, dtype=torch.float32, device=self.device
                )
                exact_loss = -(
                    advantage_tensor[valid_tensor].detach()
                    * selected_log_probs[valid_tensor]
                ).mean()
            elif reward_mode == "exact_target_expert":
                # Every exact Target solve failed. Do not silently train on an
                # upper-bound fallback when exact-target mode was requested.
                exact_aux_weight = 1.0

        if reward_mode == "exact_target_expert":
            policy_loss = exact_loss
        elif reward_mode == "hybrid_pomo_exact_expert":
            policy_loss = hybrid_loss + exact_aux_weight * exact_loss
        else:
            policy_loss = hybrid_loss

        diagnostic_hook = getattr(self, "_observe_rl_losses", None)
        if diagnostic_hook is not None:
            diagnostic_hook(hybrid_loss, exact_loss, exact_aux_weight,
                            locals().get("hybrid_advantages"), advantage_values
                            if exact_valid_count and exact_aux_weight else None)
        self.optimizer.zero_grad()
        policy_loss.backward()
        self._clip_main_gradients()
        self._step_main_optimizer()
        diagnostics = {
            "selected_indices": selected_indices,
            "exact_objectives": exact_objectives,
            "gurobi_calls": int(target_reward_calls),
            "gurobi_failures": int(target_reward_failures),
            "gurobi_runtime": float(target_reward_runtime),
            "exact_rl_valid_count": exact_valid_count,
            "exact_rl_advantage_sum": exact_advantage_sum,
            "exact_rl_positive_advantage_count": exact_positive_advantage_count,
            "hybrid_policy_loss": float(hybrid_loss.detach().item()),
            "exact_expert_policy_loss": float(exact_loss.detach().item()),
            "exact_expert_aux_weight": float(exact_aux_weight),
            **subset_stats,
            "rl_extra_bound_tables": int(extra_tables),
            "rl_extra_bound_reduction_sum": float(extra_reduction_sum),
        }
        return (
            float(policy_loss.detach().item()),
            target_upper_bounds,
            target_reward_costs,
            diagnostics,
        )

    def _imitation_update(
        self,
        instances,
        expert_sequences,
        expert_changed_flags,
    ) -> tuple[float, float, int, int]:
        if not (
            len(instances) == len(expert_sequences) == len(expert_changed_flags)
        ):
            raise ValueError("IL instances, experts, and changed flags must align")
        weighted_loss = 0.0
        correct = 0
        tokens = 0
        training_instances: list[Any] = []
        training_experts: list[list[int]] = []
        for index, (instance, changed) in enumerate(
            zip(instances, expert_changed_flags)
        ):
            repeats = (
                self.changed_expert_il_weight
                if bool(changed)
                else int(getattr(self, "unchanged_expert_il_weight", 1))
            )
            training_instances.extend([instance] * repeats)
            training_experts.extend([expert_sequences[index]] * repeats)

        current_effective_examples = len(training_instances)
        replay_examples = 0
        if self.expert_replay_fraction > 0.0 and self._expert_replay_pool:
            replay_examples = int(
                round(
                    current_effective_examples
                    * self.expert_replay_fraction
                    / (1.0 - self.expert_replay_fraction)
                )
            )
            for _ in range(replay_examples):
                replay_instance, replay_expert = self._expert_replay_pool[
                    random.randrange(len(self._expert_replay_pool))
                ]
                training_instances.append(replay_instance)
                training_experts.append(replay_expert)

        groups: dict[int, list[int]] = {}
        for index, instance in enumerate(training_instances):
            groups.setdefault(int(instance.J), []).append(index)
        active_inner_epochs = int(
            getattr(self, "_active_imitation_inner_epochs", self.imitation_inner_epochs)
        )
        for pass_index in range(1, active_inner_epochs + 1):
            pass_loss = 0.0
            pass_correct = 0
            pass_tokens = 0
            pass_steps = 0
            for problem_size in sorted(groups):
                indices = groups[problem_size]
                for start in range(0, len(indices), self.imitation_batch_size):
                    local_indices = indices[start : start + self.imitation_batch_size]
                    self.optimizer.zero_grad()
                    logical_token_count = len(local_indices) * int(problem_size)
                    micro_batch_size = int(
                        getattr(
                            self,
                            "imitation_micro_batch_size",
                            self.imitation_batch_size,
                        )
                    )
                    # Keep the logical IL batch and optimizer-step count unchanged
                    # while bounding the teacher-forcing graph resident on CUDA.
                    # Token weighting makes the accumulated gradient equal to the
                    # gradient of the original full-batch mean CE objective.
                    for micro_start in range(0, len(local_indices), micro_batch_size):
                        micro_indices = local_indices[
                            micro_start : micro_start + micro_batch_size
                        ]
                        loss, token_correct, _, token_count = teacher_forcing_loss(
                            self.model,
                            [training_instances[index] for index in micro_indices],
                            [training_experts[index] for index in micro_indices],
                            device=self.device,
                        )
                        scaled_loss = loss * (
                            float(token_count) / max(logical_token_count, 1)
                        )
                        scaled_loss.backward()
                        weighted_loss += float(loss.detach().item()) * token_count
                        correct += token_correct
                        tokens += token_count
                        pass_loss += float(loss.detach().item()) * token_count
                        pass_correct += token_correct
                        pass_tokens += token_count
                    # Separate optimizer step: CE is not mixed into the RL objective.
                    self._clip_main_gradients()
                    self._step_main_optimizer()
                    pass_steps += 1
            hook = getattr(self, "_observe_il_pass", None)
            if hook is not None:
                hook(
                    pass_index=pass_index,
                    loss=pass_loss / max(pass_tokens, 1),
                    accuracy=pass_correct / max(pass_tokens, 1),
                    tokens=pass_tokens,
                    optimizer_steps=pass_steps,
                )
        return (
            weighted_loss / max(tokens, 1),
            correct / max(tokens, 1),
            len(training_instances),
            replay_examples,
        )

    def _add_changed_experts_to_replay(
        self,
        instances: list[Any],
        expert_sequences: list[list[int]],
        expert_changed_flags: list[bool],
    ) -> int:
        """Append this epoch's strict improvements after its IL update."""

        if self.expert_replay_capacity == 0:
            return 0
        added = 0
        for instance, expert, changed in zip(
            instances, expert_sequences, expert_changed_flags
        ):
            if not changed:
                continue
            self._expert_replay_pool.append((instance, list(expert)))
            added += 1
        overflow = len(self._expert_replay_pool) - self.expert_replay_capacity
        if overflow > 0:
            del self._expert_replay_pool[:overflow]
        return added

    def _edge_update(self, instances, target_matrices) -> float:
        weighted_loss = 0.0
        edge_count = 0
        self.edge_decoder.train()
        groups: dict[int, list[int]] = {}
        for index, instance in enumerate(instances):
            groups.setdefault(int(instance.J), []).append(index)
        for problem_size in sorted(groups):
            indices = groups[problem_size]
            for start in range(0, len(indices), self.edge_batch_size):
                local_indices = indices[start : start + self.edge_batch_size]
                local_instances = [instances[index] for index in local_indices]
                features = torch.as_tensor(
                    np.stack([instance.node_features for instance in local_instances]),
                    dtype=torch.float32,
                    device=self.device,
                )
                # The upper encoder supplies features only.  Detach/no_grad keeps
                # this lower-level regression out of the policy optimizer entirely.
                with torch.no_grad():
                    encoded = self.model.encoder(features).detach()
                targets = torch.as_tensor(
                    np.stack(
                        [
                            normalize_contribution_matrix(
                                target_matrices[index], instances[index]
                            )
                            for index in local_indices
                        ]
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
                predictions = self.edge_decoder(encoded)
                loss = edge_decoder_loss(
                    predictions, targets, huber_beta=self.edge_huber_beta
                )
                self.edge_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.edge_decoder.parameters(), self.edge_grad_clip
                )
                self._step_edge_optimizer()
                node_count = targets.size(1)
                local_edges = len(local_instances) * node_count * (node_count - 1)
                weighted_loss += float(loss.detach().item()) * local_edges
                edge_count += local_edges
        return weighted_loss / max(edge_count, 1)

    def _train_one_batch(self, batch_size: int) -> dict[str, Any]:
        batch_started = time.perf_counter()
        self.model.train()
        problem_size = self._sample_problem_size()
        self.env.load_problems(batch_size, self.device, problem_size=problem_size)
        instances = list(self.env.instances)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state.node_features)

        # Predictor ranking is solver-free and happens before the sole anchor
        # solve.  The target rollout remains in-graph for the RL step.
        predicted_matrices = self._predict_contribution_matrices(instances)
        target_rollout, target_step_log_probs = self._rollout_target()
        target_sequences = target_rollout.to_sequence_lists()
        reference_sequences = self._generate_reference_candidates(instances)
        anchors, anchor_prediction_scores, anchor_selection = self._select_anchor_sequences(
            instances, reference_sequences, predicted_matrices
        )

        self._random_subset_handle = None
        if getattr(self, "dense_reward_source", "hybrid_upper_bound") == "random_subset_exact":
            self._random_subset_handle = self._start_random_subset_solves(
                instances, target_sequences
            )
            self._pending_after_sparse_submit = self._random_subset_handle["submit"]
        try:
            results, sparse_wall_runtime, near_miss_eligible_count = (
                self._run_sparse_self_improvement(
                    instances, reference_sequences, anchors
                )
            )
        finally:
            # Serial mode or an unexpected exit: never leave the hook armed.
            self._run_after_sparse_submit()
        experts = [list(result.expert_sequence) for result in results]
        (
            policy_loss,
            target_upper_bounds,
            target_reward_costs,
            target_reward_diagnostics,
        ) = self._policy_rl_update(
            target_step_log_probs,
            target_sequences,
            results,
            instances,
        )
        self._random_subset_handle = None
        # IL and Predictor updates are delayed until every batch in the epoch
        # has completed Expert generation and RL, matching the formal phase
        # order.  Only detached instances, tours, and C0 labels are buffered.
        self._last_batch_training_records = [
            (
                instance,
                list(expert),
                np.asarray(result.anchor_upper_bound_matrix, dtype=float).copy(),
                bool(result.expert_changed),
            )
            for instance, expert, result in zip(instances, experts, results)
        ]
        imitation_loss = 0.0
        imitation_accuracy = 0.0
        edge_loss = 0.0

        anchor_exact = np.asarray(
            [result.anchor_exact_makespan for result in results], dtype=float
        )
        expert_exact = np.asarray(
            [result.expert_exact_makespan for result in results], dtype=float
        )
        exact_improvements = anchor_exact - expert_exact
        if float(exact_improvements.min()) < -self.si_upper_bound_tolerance:
            raise RuntimeError("a self-improved expert is worse than its anchor")
        sparse_calls = sum(result.total_gurobi_calls for result in results)
        target_reward_calls = int(target_reward_diagnostics["gurobi_calls"])
        calls = sparse_calls + target_reward_calls
        if target_reward_calls != len(instances):
            raise RuntimeError("each instance must receive exactly one exact Target reward solve")
        if calls < 2 * len(instances) or calls > 3 * len(instances):
            raise RuntimeError("hard sparse-Gurobi batch budget was violated")
        pomo_size = len(target_sequences[0])
        requested_top_k = np.asarray(
            [self._route_start_top_k(instance.J) for instance in instances],
            dtype=float,
        )
        actual_top_k = np.asarray(
            [len(result.top_k_start_sequences) for result in results],
            dtype=float,
        )
        actual_quality_starts = np.asarray(
            [result.quality_start_count for result in results], dtype=float
        )
        actual_diversity_starts = np.asarray(
            [result.diversity_start_count for result in results], dtype=float
        )
        actual_top_k_ratios = actual_top_k / np.asarray(
            [instance.J for instance in instances], dtype=float
        )
        eligible_near_misses = []
        if self.si_near_miss_rate > 0.0:
            eligible_near_misses = [
                result
                for result in results
                if np.isfinite(float(result.near_miss_relative_gap))
                and result.near_miss_candidate_sequence is not None
                and (
                    result.near_miss_attempted
                    or (
                        not result.expert_changed
                        and result.expert_gurobi_calls == 0
                    )
                )
            ]
        if len(eligible_near_misses) != near_miss_eligible_count:
            raise RuntimeError(
                "near-miss eligibility changed unexpectedly between batch phases: "
                f"{len(eligible_near_misses)} != {near_miss_eligible_count}"
            )
        attempted_near_misses = [
            result for result in results if result.near_miss_attempted
        ]
        accepted_near_misses = [
            result for result in results if result.near_miss_accepted
        ]
        near_miss_eligible_gap_sum = float(
            sum(result.near_miss_relative_gap for result in eligible_near_misses)
        )
        near_miss_attempt_gap_sum = float(
            sum(result.near_miss_relative_gap for result in attempted_near_misses)
        )
        near_miss_accept_gap_sum = float(
            sum(result.near_miss_relative_gap for result in accepted_near_misses)
        )
        self._last_batch_exact_examples = [
            {
                "instance_id": result.instance_id,
                "problem_size": int(instance.J),
                "anchor_exact_makespan": float(result.anchor_exact_makespan),
                "expert_upper_bound": float(result.expert_upper_bound),
                "expert_exact_makespan": float(result.expert_exact_makespan),
                "exact_improvement": float(result.exact_improvement),
                "exact_improvement_percent": float(result.exact_improvement_percent),
                "expert_changed": bool(result.expert_changed),
                "expert_source": str(result.expert_source),
                "near_miss_attempted": bool(result.near_miss_attempted),
                "near_miss_accepted": bool(result.near_miss_accepted),
                "near_miss_relative_gap": (
                    float(result.near_miss_relative_gap)
                    if np.isfinite(float(result.near_miss_relative_gap))
                    else None
                ),
                "target_reward_route_index": int(
                    target_reward_diagnostics["selected_indices"][index]
                ),
                "target_reward_sequence": list(
                    target_sequences[index][
                        int(target_reward_diagnostics["selected_indices"][index])
                    ]
                ),
                "target_reward_exact_makespan": (
                    float(target_reward_diagnostics["exact_objectives"][index])
                    if np.isfinite(
                        float(target_reward_diagnostics["exact_objectives"][index])
                    )
                    else None
                ),
                "training_gurobi_calls": int(result.total_gurobi_calls + 1),
            }
            for index, (instance, result) in enumerate(zip(instances, results))
        ]
        finite_target_exact = np.asarray(
            target_reward_diagnostics["exact_objectives"], dtype=float
        )
        finite_target_exact = finite_target_exact[np.isfinite(finite_target_exact)]
        self._joint_batch_index += 1
        return {
            "score": float(expert_exact.mean()),
            "anchor_score": float(anchor_exact.mean()),
            "expert_score": float(expert_exact.mean()),
            "target_best_upper_bound": float(np.min(target_upper_bounds, axis=1).mean()),
            "target_best_reward_cost": float(
                np.min(target_reward_costs, axis=1).mean()
            ),
            "target_exact_reward_makespan": (
                float(finite_target_exact.mean())
                if finite_target_exact.size
                else float(np.min(target_upper_bounds, axis=1).mean())
            ),
            "target_exact_reward_success_ratio": float(
                finite_target_exact.size / max(len(instances), 1)
            ),
            "exact_rl_valid_examples": float(
                target_reward_diagnostics["exact_rl_valid_count"]
            ),
            "exact_rl_advantage_sum": float(
                target_reward_diagnostics["exact_rl_advantage_sum"]
            ),
            "exact_rl_positive_advantage_count": float(
                target_reward_diagnostics["exact_rl_positive_advantage_count"]
            ),
            "expert_improvement": float(exact_improvements.mean()),
            "expert_improvement_percent": float(
                np.mean(100.0 * exact_improvements / np.maximum(np.abs(anchor_exact), 1e-12))
            ),
            "improved_ratio": float(np.mean(exact_improvements > 1e-7)),
            "expert_changed_ratio": float(np.mean([result.expert_changed for result in results])),
            "policy_loss": policy_loss,
            "hybrid_policy_loss": float(
                target_reward_diagnostics["hybrid_policy_loss"]
            ),
            "exact_expert_policy_loss": float(
                target_reward_diagnostics["exact_expert_policy_loss"]
            ),
            "exact_expert_aux_weight": float(
                target_reward_diagnostics["exact_expert_aux_weight"]
            ),
            "imitation_loss": float(imitation_loss),
            "imitation_token_accuracy": float(imitation_accuracy),
            "edge_loss": float(edge_loss),
            "problem_size": float(problem_size),
            "pomo_size": float(pomo_size),
            "reference_candidates": float(self.reference_candidate_count),
            "requested_top_k": float(requested_top_k.mean()),
            "actual_top_k": float(actual_top_k.mean()),
            "actual_top_k_ratio": float(actual_top_k_ratios.mean()),
            "actual_quality_starts": float(actual_quality_starts.mean()),
            "actual_diversity_starts": float(actual_diversity_starts.mean()),
            "anchor_prediction_score": float(np.mean(anchor_prediction_scores)),
            "anchor_selection_predictor_ratio": float(
                anchor_selection == "lower_level_predictor"
            ),
            "training_instances": float(len(instances)),
            "training_gurobi_calls": float(calls),
            "anchor_gurobi_calls": float(
                sum(result.anchor_gurobi_calls for result in results)
            ),
            "expert_gurobi_calls": float(
                sum(result.expert_gurobi_calls for result in results)
            ),
            "target_reward_gurobi_calls": float(target_reward_calls),
            "target_reward_gurobi_failures": float(
                target_reward_diagnostics["gurobi_failures"]
            ),
            "old_rollout_gurobi_calls_estimate": float(len(instances) * pomo_size),
            "gurobi_runtime": float(
                sum(result.gurobi_runtime for result in results)
                + target_reward_diagnostics["gurobi_runtime"]
            ),
            "target_reward_gurobi_runtime": float(
                target_reward_diagnostics["gurobi_runtime"]
            ),
            "random_subset_gurobi_calls": float(
                target_reward_diagnostics["random_subset_gurobi_calls"]
            ),
            "random_subset_gurobi_failures": float(
                target_reward_diagnostics["random_subset_gurobi_failures"]
            ),
            "random_subset_gurobi_runtime": float(
                target_reward_diagnostics["random_subset_gurobi_runtime"]
            ),
            "random_subset_routes": float(target_reward_diagnostics["random_subset_routes"]),
            "random_subset_instances": float(
                target_reward_diagnostics["random_subset_instances"]
            ),
            "c0_refresh_gurobi_calls": float(
                sum(result.c0_refresh_gurobi_calls for result in results)
            ),
            "c0_refresh_accepted": float(sum(result.c0_refresh_accepted for result in results)),
            "c0_refresh_rounds_run": float(sum(result.c0_refresh_rounds_run for result in results)),
            "c0_refresh_runtime": float(sum(result.c0_refresh_runtime for result in results)),
            "rl_extra_bound_tables": float(target_reward_diagnostics["rl_extra_bound_tables"]),
            "rl_extra_bound_reduction_sum": float(
                target_reward_diagnostics["rl_extra_bound_reduction_sum"]
            ),
            "local_search_runtime": float(
                sum(result.local_search_runtime for result in results)
            ),
            "sparse_phase_wall_runtime": float(sparse_wall_runtime),
            "generated_moves": float(sum(result.generated_moves for result in results)),
            "evaluated_moves": float(sum(result.evaluated_moves for result in results)),
            "dual_pruned_moves": float(sum(result.dual_pruned_moves for result in results)),
            "accepted_moves": float(sum(result.accepted_moves for result in results)),
            "perturbation_attempts": float(
                sum(result.perturbation_attempts for result in results)
            ),
            "perturbation_improvements": float(
                sum(result.perturbation_improvements for result in results)
            ),
            "linear_duals": float(sum(result.linear_dual_count for result in results)),
            "quadratic_duals": float(
                sum(result.quadratic_dual_count for result in results)
            ),
            "final_validation_failures": float(
                sum(result.final_validation_failed for result in results)
            ),
            "near_miss_eligible_count": float(near_miss_eligible_count),
            "near_miss_effective_rate": float(self._active_near_miss_rate),
            "near_miss_attempt_count": float(len(attempted_near_misses)),
            "near_miss_accept_count": float(len(accepted_near_misses)),
            "near_miss_validation_failures": float(
                sum(result.near_miss_validation_failed for result in results)
            ),
            "near_miss_eligible_gap_sum": near_miss_eligible_gap_sum,
            "near_miss_attempt_gap_sum": near_miss_attempt_gap_sum,
            "near_miss_accept_gap_sum": near_miss_accept_gap_sum,
            "near_miss_batch_mean_eligible_relative_gap": (
                near_miss_eligible_gap_sum / max(len(eligible_near_misses), 1)
            ),
            "near_miss_batch_mean_attempt_relative_gap": (
                near_miss_attempt_gap_sum / max(len(attempted_near_misses), 1)
            ),
            "near_miss_batch_mean_accept_relative_gap": (
                near_miss_accept_gap_sum / max(len(accepted_near_misses), 1)
            ),
            "batch_runtime": float(time.perf_counter() - batch_started),
        }

    def _train_one_epoch(self, epoch: int) -> dict[str, Any]:
        self._active_imitation_inner_epochs = self._imitation_inner_epochs_for_epoch(
            epoch
        )
        self._active_near_miss_rate = self._effective_near_miss_rate()
        totals: dict[str, float] = {}
        extensive_keys = {
            "training_instances",
            "training_gurobi_calls",
            "anchor_gurobi_calls",
            "expert_gurobi_calls",
            "target_reward_gurobi_calls",
            "target_reward_gurobi_failures",
            "exact_rl_valid_examples",
            "exact_rl_advantage_sum",
            "exact_rl_positive_advantage_count",
            "old_rollout_gurobi_calls_estimate",
            "gurobi_runtime",
            "target_reward_gurobi_runtime",
            "random_subset_gurobi_calls",
            "random_subset_gurobi_failures",
            "random_subset_gurobi_runtime",
            "random_subset_routes",
            "random_subset_instances",
            "c0_refresh_gurobi_calls",
            "c0_refresh_accepted",
            "c0_refresh_rounds_run",
            "c0_refresh_runtime",
            "rl_extra_bound_tables",
            "rl_extra_bound_reduction_sum",
            "local_search_runtime",
            "sparse_phase_wall_runtime",
            "generated_moves",
            "evaluated_moves",
            "dual_pruned_moves",
            "accepted_moves",
            "perturbation_attempts",
            "perturbation_improvements",
            "linear_duals",
            "quadratic_duals",
            "final_validation_failures",
            "near_miss_eligible_count",
            "near_miss_attempt_count",
            "near_miss_accept_count",
            "near_miss_validation_failures",
            "near_miss_eligible_gap_sum",
            "near_miss_attempt_gap_sum",
            "near_miss_accept_gap_sum",
            "batch_runtime",
        }
        episode = 0
        batches = 0
        progress_step = max(
            int(self.train_episodes * (self.progress_log_percent / 100.0)), 1
        )
        next_progress_log = progress_step
        epoch_started = time.perf_counter()
        exact_examples: list[dict[str, Any]] = []
        epoch_instances: list[Any] = []
        epoch_experts: list[list[int]] = []
        epoch_c0_matrices: list[np.ndarray] = []
        epoch_expert_changed: list[bool] = []
        while episode < self.train_episodes:
            current_batch = min(self.train_batch_size, self.train_episodes - episode)
            row = self._train_one_batch(current_batch)
            for instance, expert, c0_matrix, expert_changed in self._last_batch_training_records:
                epoch_instances.append(instance)
                epoch_experts.append(expert)
                epoch_c0_matrices.append(c0_matrix)
                epoch_expert_changed.append(expert_changed)
            changed_examples = [
                item for item in self._last_batch_exact_examples if item["expert_changed"]
            ]
            unchanged_examples = [
                item for item in self._last_batch_exact_examples if not item["expert_changed"]
            ]
            for item in changed_examples + unchanged_examples:
                if len(exact_examples) >= 3:
                    break
                exact_examples.append(dict(item))
            for key, value in row.items():
                weight = 1 if key in extensive_keys else current_batch
                totals[key] = totals.get(key, 0.0) + float(value) * weight
            episode += current_batch
            batches += 1
            should_log_initial = batches <= self.log_first_batch_count
            should_log_progress = episode >= next_progress_log or episode == self.train_episodes
            if should_log_initial or should_log_progress:
                while next_progress_log <= episode:
                    next_progress_log += progress_step
                logger.info(
                    "Epoch {:3d}: {} {:6d}/{:6d}({:5.1f}%) J={:3.0f} "
                    "Anchor={:.4f} Expert={:.4f} dV={:.4f} "
                    "RL={:.4f} IL/Edge=pending Calls={:.2f}/inst "
                    "TopK={:.1f}({:.3f}J) NM={:.0f}/{:.0f}/{:.0f} "
                    "NMgap={:.5f} Sec={:.2f}",
                    epoch,
                    _format_progress_bar(
                        episode, self.train_episodes, self.progress_bar_width
                    ),
                    episode,
                    self.train_episodes,
                    100.0 * episode / self.train_episodes,
                    row["problem_size"],
                    totals["anchor_score"] / episode,
                    totals["expert_score"] / episode,
                    totals["expert_improvement"] / episode,
                    totals["policy_loss"] / episode,
                    totals["training_gurobi_calls"] / episode,
                    row["actual_top_k"],
                    row["actual_top_k_ratio"],
                    totals["near_miss_eligible_count"],
                    totals["near_miss_attempt_count"],
                    totals["near_miss_accept_count"],
                    row["near_miss_batch_mean_attempt_relative_gap"],
                    row["batch_runtime"],
                )

        average = {
            key: (value if key in extensive_keys else value / max(episode, 1))
            for key, value in totals.items()
        }
        if len(epoch_instances) != episode:
            raise RuntimeError(
                f"epoch training buffer size {len(epoch_instances)} != instances {episode}"
            )
        imitation_started = time.perf_counter()
        (
            imitation_loss,
            imitation_accuracy,
            imitation_effective_examples,
            expert_replay_examples,
        ) = self._imitation_update(
            epoch_instances,
            epoch_experts,
            epoch_expert_changed,
        )
        imitation_runtime = time.perf_counter() - imitation_started
        expert_replay_added = self._add_changed_experts_to_replay(
            epoch_instances,
            epoch_experts,
            epoch_expert_changed,
        )
        edge_started = time.perf_counter()
        edge_loss = self._edge_update(epoch_instances, epoch_c0_matrices)
        edge_runtime = time.perf_counter() - edge_started
        average["imitation_loss"] = float(imitation_loss)
        average["imitation_token_accuracy"] = float(imitation_accuracy)
        average["edge_loss"] = float(edge_loss)
        calls = average["training_gurobi_calls"]
        old_calls = average["old_rollout_gurobi_calls_estimate"]
        near_miss_eligible_count = average["near_miss_eligible_count"]
        near_miss_attempt_count = average["near_miss_attempt_count"]
        near_miss_accept_count = average["near_miss_accept_count"]
        if calls < 2 * episode or calls > 3 * episode:
            raise RuntimeError(
                f"epoch Gurobi budget outside [2, 3] per instance: "
                f"{calls} not in [{2 * episode}, {3 * episode}]"
            )
        epoch_runtime = time.perf_counter() - epoch_started
        audit_payload = {
            "epoch": int(epoch),
            "training_instances": int(episode),
            "mean_anchor_exact_makespan": average["anchor_score"],
            "mean_expert_exact_makespan": average["expert_score"],
            "expert_better_ratio": average["improved_ratio"],
            "mean_exact_improvement": average["expert_improvement"],
            "mean_exact_improvement_percent": average["expert_improvement_percent"],
            "mean_target_exact_reward_makespan": average[
                "target_exact_reward_makespan"
            ],
            "target_exact_reward_success_ratio": average[
                "target_exact_reward_success_ratio"
            ],
            "exact_rl_valid_ratio": average["exact_rl_valid_examples"]
            / max(episode, 1),
            "exact_rl_advantage_mean": average["exact_rl_advantage_sum"]
            / max(average["exact_rl_valid_examples"], 1.0),
            "exact_rl_target_better_ratio": average[
                "exact_rl_positive_advantage_count"
            ]
            / max(average["exact_rl_valid_examples"], 1.0),
            "near_miss_eligible_count": near_miss_eligible_count,
            "near_miss_attempt_count": near_miss_attempt_count,
            "near_miss_accept_count": near_miss_accept_count,
            "examples": exact_examples,
        }
        (self.output_dir / f"exact_improvement_epoch_{epoch}.json").write_text(
            json.dumps(audit_payload, indent=2), encoding="utf-8"
        )
        for item in exact_examples:
            logger.info(
                "Exact example {}: anchor V={:.6f}, expert U={:.6f}, "
                "expert V={:.6f}, improvement={:.3f}%, calls={}",
                item["instance_id"],
                item["anchor_exact_makespan"],
                item["expert_upper_bound"],
                item["expert_exact_makespan"],
                item["exact_improvement_percent"],
                item["training_gurobi_calls"],
            )
        self._previous_epoch_il_accuracy = float(imitation_accuracy)
        self._previous_epoch_target_better_ratio = float(
            average["exact_rl_positive_advantage_count"]
            / max(average["exact_rl_valid_examples"], 1.0)
        )
        return {
            "epoch": float(epoch),
            "global_step": float(self.global_step),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "edge_global_step": float(self.edge_global_step),
            "edge_learning_rate": float(self.edge_optimizer.param_groups[0]["lr"]),
            "train_score": average["expert_score"],
            "anchor_score": average["anchor_score"],
            "expert_score": average["expert_score"],
            "anchor_exact_makespan": average["anchor_score"],
            "expert_exact_makespan": average["expert_score"],
            "target_best_upper_bound": average["target_best_upper_bound"],
            "target_best_reward_cost": average["target_best_reward_cost"],
            "target_exact_reward_makespan": average[
                "target_exact_reward_makespan"
            ],
            "target_exact_reward_success_ratio": average[
                "target_exact_reward_success_ratio"
            ],
            "exact_rl_valid_ratio": average["exact_rl_valid_examples"]
            / max(episode, 1),
            "exact_rl_advantage_mean": average["exact_rl_advantage_sum"]
            / max(average["exact_rl_valid_examples"], 1.0),
            "exact_rl_target_better_ratio": average[
                "exact_rl_positive_advantage_count"
            ]
            / max(average["exact_rl_valid_examples"], 1.0),
            "expert_improvement": average["expert_improvement"],
            "expert_improvement_percent": average["expert_improvement_percent"],
            "improved_ratio": average["improved_ratio"],
            "expert_better_ratio": average["improved_ratio"],
            "expert_changed_ratio": average["expert_changed_ratio"],
            "mean_makespan_improvement": average["expert_improvement"],
            "mean_makespan_improvement_percent": average[
                "expert_improvement_percent"
            ],
            "train_loss": average["policy_loss"],
            "policy_loss": average["policy_loss"],
            "rl_loss": average["policy_loss"],
            "hybrid_policy_loss": average["hybrid_policy_loss"],
            "exact_expert_policy_loss": average["exact_expert_policy_loss"],
            "exact_expert_aux_weight": average["exact_expert_aux_weight"],
            "imitation_loss": average["imitation_loss"],
            "il_loss": average["imitation_loss"],
            "imitation_token_accuracy": average["imitation_token_accuracy"],
            "edge_loss": average["edge_loss"],
            "edge_predictor_loss": average["edge_loss"],
            "loss_mode": "separate_rl_then_il_then_detached_edge",
            "reward_definition": {
                "exact_target_expert": "negative_exact_selected_target",
                "hybrid_pomo": "negative_hybrid_one_exact_target_plus_upper_bounds",
                "hybrid_pomo_exact_expert": (
                    "standardized_hybrid_pomo_plus_bounded_exact_expert"
                ),
            }[self.rl_reward_mode],
            "baseline_definition": {
                "exact_target_expert": "negative_exact_expert_makespan",
                "hybrid_pomo": "per_instance_pomo_mean_reward",
                "hybrid_pomo_exact_expert": (
                    "per_instance_pomo_mean_plus_exact_expert_auxiliary"
                ),
            }[self.rl_reward_mode],
            "anchor_selection": "lower_level_predictor",
            "anchor_selection_predictor_ratio": average[
                "anchor_selection_predictor_ratio"
            ],
            "anchor_prediction_score": average["anchor_prediction_score"],
            "cvxpylayer_route_loss": 0.0,
            "cvxpylayer_route_count": 0.0,
            "cvxpylayer_aux_loss": 0.0,
            "cvxpylayer_aux_count": 0.0,
            "cvxpylayer_objective_loss": 0.0,
            "cvxpylayer_objective_count": 0.0,
            "rebar_loss": 0.0,
            "train_feasible_ratio": 1.0,
            "avg_problem_size": average["problem_size"],
            "avg_pomo_size": average["pomo_size"],
            "batch_size": float(self.train_batch_size),
            "pomo_size": average["pomo_size"],
            "top_k": float(self.si_route_start_top_k),
            "top_k_mode": (
                "ceil_problem_size_ratio"
                if self.si_route_start_top_k_ratio > 0.0
                else "fixed"
            ),
            "top_k_ratio": float(self.si_route_start_top_k_ratio),
            "requested_top_k": average["requested_top_k"],
            "actual_top_k": average["actual_top_k"],
            "actual_top_k_ratio": average["actual_top_k_ratio"],
            "top_k_diversity_count": float(self.si_route_start_diversity_count),
            "top_k_diversity_pool_size": float(
                self.si_route_start_diversity_pool_size
            ),
            "top_k_diversity_max_relative_gap": float(
                self.si_route_start_diversity_max_relative_gap
            ),
            "actual_quality_starts": average["actual_quality_starts"],
            "actual_diversity_starts": average["actual_diversity_starts"],
            "max_local_moves": float(self.si_max_local_moves),
            "dual_pruning_enabled": float(self.si_use_dual_pruning),
            "vectorized_local_search": float(VECTORIZED_LOCAL_SEARCH_ENABLED),
            "solver_calls": calls,
            "training_gurobi_calls": calls,
            "new_actual_gurobi_calls": calls,
            "anchor_gurobi_calls": average["anchor_gurobi_calls"],
            "expert_gurobi_calls": average["expert_gurobi_calls"],
            "target_reward_gurobi_calls": average[
                "target_reward_gurobi_calls"
            ],
            "target_reward_gurobi_failures": average[
                "target_reward_gurobi_failures"
            ],
            "gurobi_calls_per_instance": calls / max(episode, 1),
            "rollout_gurobi_calls": average["target_reward_gurobi_calls"],
            "old_rollout_gurobi_calls_estimate": old_calls,
            "gurobi_call_reduction_factor": old_calls / max(calls, 1.0),
            "gurobi_call_reduction_percent": 100.0
            * max(old_calls - calls, 0.0)
            / max(old_calls, 1.0),
            "gurobi_runtime_seconds": average["gurobi_runtime"],
            "target_reward_gurobi_runtime_seconds": average[
                "target_reward_gurobi_runtime"
            ],
            "dense_reward_random_subset": float(
                getattr(self, "dense_reward_source", "") == "random_subset_exact"
            ),
            "random_subset_gurobi_calls": average["random_subset_gurobi_calls"],
            "random_subset_gurobi_failures": average["random_subset_gurobi_failures"],
            "random_subset_gurobi_runtime_seconds": average["random_subset_gurobi_runtime"],
            "random_subset_routes_per_instance": average["random_subset_routes"]
            / max(episode, 1),
            "random_subset_instance_coverage": average["random_subset_instances"]
            / max(episode, 1),
            "total_gurobi_calls_including_random_subset": calls
            + average["random_subset_gurobi_calls"],
            "total_gurobi_calls_per_instance_including_random_subset": (
                calls + average["random_subset_gurobi_calls"]
            )
            / max(episode, 1),
            "c0_refresh_rounds": float(getattr(self, "si_c0_refresh_rounds", 0)),
            "unchanged_expert_il_weight": float(getattr(self, "unchanged_expert_il_weight", 1)),
            "c0_refresh_gurobi_calls": average["c0_refresh_gurobi_calls"],
            "c0_refresh_calls_per_instance": average["c0_refresh_gurobi_calls"] / max(episode, 1),
            "c0_refresh_accepted": average["c0_refresh_accepted"],
            "c0_refresh_accepted_ratio": average["c0_refresh_accepted"] / max(episode, 1),
            "c0_refresh_rounds_run": average["c0_refresh_rounds_run"],
            "c0_refresh_runtime_seconds": average["c0_refresh_runtime"],
            "rl_multi_c0_bound": float(getattr(self, "rl_multi_c0_bound", False)),
            "rl_extra_bound_tables_per_instance": average["rl_extra_bound_tables"]
            / max(episode, 1),
            "rl_extra_bound_mean_relative_reduction": average["rl_extra_bound_reduction_sum"]
            / max(episode, 1),
            "total_gurobi_calls_all": calls
            + average["random_subset_gurobi_calls"]
            + average["c0_refresh_gurobi_calls"],
            "total_gurobi_calls_per_instance_all": (
                calls
                + average["random_subset_gurobi_calls"]
                + average["c0_refresh_gurobi_calls"]
            )
            / max(episode, 1),
            "local_search_runtime_seconds": average["local_search_runtime"],
            "sparse_phase_wall_runtime_seconds": average["sparse_phase_wall_runtime"],
            "generated_moves": average["generated_moves"],
            "evaluated_moves": average["evaluated_moves"],
            "dual_pruned_moves": average["dual_pruned_moves"],
            "accepted_moves": average["accepted_moves"],
            "perturbation_restarts": float(self.si_perturbation_restarts),
            "perturbation_attempts": average["perturbation_attempts"],
            "perturbation_improvements": average["perturbation_improvements"],
            "linear_duals": average["linear_duals"],
            "quadratic_duals": average["quadratic_duals"],
            "final_validation_failures": average["final_validation_failures"],
            "near_miss_rate": average["near_miss_effective_rate"],
            "near_miss_min_rate": float(self.si_near_miss_min_rate),
            "near_miss_max_rate": float(self.si_near_miss_rate),
            "near_miss_eligible_count": near_miss_eligible_count,
            "near_miss_attempt_count": near_miss_attempt_count,
            "near_miss_accept_count": near_miss_accept_count,
            "near_miss_eligible_ratio": near_miss_eligible_count
            / max(episode, 1),
            "near_miss_attempt_rate": near_miss_attempt_count
            / max(near_miss_eligible_count, 1.0),
            "near_miss_accept_rate": near_miss_accept_count
            / max(near_miss_attempt_count, 1.0),
            "near_miss_validation_failures": average[
                "near_miss_validation_failures"
            ],
            "near_miss_mean_eligible_relative_gap": average[
                "near_miss_eligible_gap_sum"
            ]
            / max(near_miss_eligible_count, 1.0),
            "near_miss_mean_attempt_relative_gap": average[
                "near_miss_attempt_gap_sum"
            ]
            / max(near_miss_attempt_count, 1.0),
            "near_miss_mean_accept_relative_gap": average[
                "near_miss_accept_gap_sum"
            ]
            / max(near_miss_accept_count, 1.0),
            "epoch_runtime_seconds": epoch_runtime,
            "imitation_inner_epochs": float(self._active_imitation_inner_epochs),
            "imitation_batch_size": float(self.imitation_batch_size),
            "imitation_micro_batch_size": float(self.imitation_micro_batch_size),
            "changed_expert_il_weight": float(self.changed_expert_il_weight),
            "imitation_effective_examples": float(imitation_effective_examples),
            "expert_replay_capacity": float(self.expert_replay_capacity),
            "expert_replay_fraction_requested": float(self.expert_replay_fraction),
            "expert_replay_pool_size": float(len(self._expert_replay_pool)),
            "expert_replay_added": float(expert_replay_added),
            "expert_replay_examples": float(expert_replay_examples),
            "expert_replay_effective_fraction": float(
                expert_replay_examples / max(imitation_effective_examples, 1)
            ),
            "imitation_runtime_seconds": float(imitation_runtime),
            "edge_update_runtime_seconds": float(edge_runtime),
            "mean_batch_runtime_seconds": average["batch_runtime"] / max(batches, 1),
            "batch_runtime": average["batch_runtime"] / max(batches, 1),
            "gurobi_runtime": average["gurobi_runtime"],
            "local_search_runtime": average["local_search_runtime"],
            "instances_per_second": episode / max(epoch_runtime, 1e-12),
            "batches": float(batches),
            "cache_hits": 0.0,
            "cache_misses": 0.0,
        }

    def _reference_metrics_defaults(self) -> dict[str, Any]:
        defaults = {
            "reference_validation_ran": 0.0,
            "reference_promoted": 0.0,
            "reference_mean_objective": float("nan"),
            "target_mean_objective": float("nan"),
            "reference_mean_improvement": float("nan"),
            "reference_relative_improvement_percent": float("nan"),
            "reference_target_wins": 0.0,
            "reference_target_ties": 0.0,
            "reference_target_losses": 0.0,
            "validation_gurobi_calls": 0.0,
            "validation_runtime_seconds": 0.0,
            "reference_policy_version": float(
                getattr(self.reference_state, "policy_version", self.reference_state.promotions)
            ),
            "delta_all": float("nan"),
        }
        for size in self.reference_validation_sizes:
            defaults[f"reference_J{size}"] = float("nan")
            defaults[f"target_J{size}"] = float("nan")
            defaults[f"delta_{size}"] = float("nan")
        return defaults

    def _validate_and_maybe_promote(self, epoch: int) -> dict[str, Any]:
        if not self.reference_validation_instances:
            return self._reference_metrics_defaults()
        logger.info("Fixed validation: protocol={}, instances={}, POMO={}, augmentation=False",
                    self.reference_validation_protocol["mode"],
                    len(self.reference_validation_instances),
                    self.reference_validation_protocol["pomo_size"])
        rng_state = self._capture_rng_state()
        try:
            metrics = evaluate_policy_pair_exact(
                self.reference_policy,
                self.model,
                self.reference_validation_instances,
                device=self.device,
                solver_config=self.si_final_solver_config,
                max_candidates=self.reference_validation_candidates,
                executor=self._si_executor,
            )
        finally:
            self._restore_rng_state(rng_state)
        promoted = promote_reference_with_size_gate(
            self.reference_policy,
            self.model,
            metrics,
            required_non_degrading_sizes=(80, 100),
            epsilon=self.reference_promotion_epsilon,
        )
        compact = {
            key: value for key, value in metrics.items() if key != "details"
        }
        compact["epoch"] = int(epoch)
        compact["promoted"] = bool(promoted)
        self.reference_state.validation_rounds += 1
        self.reference_state.last_validation_epoch = int(epoch)
        self.reference_state.validation_gurobi_calls += int(
            metrics["validation_gurobi_calls"]
        )
        self.reference_state.last_metrics = compact
        self.reference_state.history.append(compact)
        logger.info("Validation:")
        for size in self.reference_validation_sizes:
            group = metrics["group_metrics"][str(size)]
            logger.info(
                "J={:<3d} Ref={:.6f} Target={:.6f} Delta={:.6%}",
                size,
                group["reference_mean"],
                group["target_mean"],
                group["delta"],
            )
        logger.info("Overall Delta={:.6%}", metrics["delta_all"])
        if promoted:
            self.reference_state.promotions += 1
            if hasattr(self.reference_state, "policy_version"):
                self.reference_state.policy_version += 1
            self.reference_state.last_promoted_epoch = int(epoch)
            self._save_reference_snapshot(epoch, compact)
            logger.info(
                "Reference promoted at epoch {}: {:.6f} -> {:.6f} ({:.3f}%)",
                epoch,
                metrics["reference_mean_objective"],
                metrics["target_mean_objective"],
                metrics["relative_improvement_percent"],
            )
        else:
            logger.info(
                "Reference retained at epoch {}: ref={:.6f}, target={:.6f}",
                epoch,
                metrics["reference_mean_objective"],
                metrics["target_mean_objective"],
            )
        result = {
            "reference_validation_ran": 1.0,
            "reference_promoted": float(promoted),
            "reference_mean_objective": float(metrics["reference_mean_objective"]),
            "target_mean_objective": float(metrics["target_mean_objective"]),
            "reference_mean_improvement": float(metrics["mean_objective_improvement"]),
            "reference_relative_improvement_percent": float(
                metrics["relative_improvement_percent"]
            ),
            "reference_target_wins": float(metrics["target_wins"]),
            "reference_target_ties": float(metrics["ties"]),
            "reference_target_losses": float(metrics["target_losses"]),
            "validation_gurobi_calls": float(metrics["validation_gurobi_calls"]),
            "validation_runtime_seconds": float(metrics["validation_runtime"]),
            "reference_policy_version": float(
                getattr(self.reference_state, "policy_version", self.reference_state.promotions)
            ),
            "delta_all": float(metrics["delta_all"]),
        }
        for size in self.reference_validation_sizes:
            group = metrics["group_metrics"][str(size)]
            result[f"reference_J{size}"] = float(group["reference_mean"])
            result[f"target_J{size}"] = float(group["target_mean"])
            result[f"delta_{size}"] = float(group["delta"])
        return result

    def run(self) -> dict[str, Any]:
        last_metrics: dict[str, Any] = {}
        logger.info("Generator stats: {}", self.generator.stats.to_dict())
        try:
            for epoch in range(self.start_epoch, self.epochs + 1):
                metrics = self._train_one_epoch(epoch)
                reference_metrics = self._reference_metrics_defaults()
                should_validate = (
                    self.reference_validation_interval > 0
                    and epoch % self.reference_validation_interval == 0
                ) or epoch == self.epochs
                if should_validate:
                    reference_metrics = self._validate_and_maybe_promote(epoch)
                metrics.update(reference_metrics)
                if self.scheduler is not None and self.scheduler_interval == "epoch":
                    # Advance once after all RL/IL updates from this epoch. The
                    # post-step state is then included in every saved checkpoint.
                    self.scheduler.step()
                self._append_metrics_row(metrics)
                last_metrics = metrics

                # ``best.pt`` tracks the best Target score observed on the
                # fixed exact validation set.  This is intentionally separate
                # from the stricter Reference gate: a Target can set a new
                # validation best even when a protected large-size group keeps
                # the frozen Reference from being promoted.
                current_score = float(reference_metrics["target_mean_objective"])
                if np.isfinite(current_score) and current_score < self.best_score:
                    self.best_score = current_score
                    self._save_checkpoint(epoch, metrics, self.output_dir / "best.pt")
                if epoch % self.checkpoint_interval == 0 or epoch == self.epochs:
                    self._save_checkpoint(
                        epoch, metrics, self.output_dir / f"checkpoint-{epoch}.pt"
                    )
                self._save_checkpoint(epoch, metrics, self.output_dir / "last.pt")

                logger.info(
                    "Epoch {:4d}/{:4d}: Expert={:.4f} dV={:.4f} ({:.3f}%) "
                    "Calls={:.3f}/inst Reduce={:.1f}x TopK={:.1f}({:.3f}J) "
                    "Pert={:.0f}/{:.0f} NM={:.0f}/{:.0f}/{:.0f} NMgap={:.5f} "
                    "Runtime={:.1f}s RefPromote={}",
                    epoch,
                    self.epochs,
                    metrics["expert_score"],
                    metrics["expert_improvement"],
                    metrics["expert_improvement_percent"],
                    metrics["gurobi_calls_per_instance"],
                    metrics["gurobi_call_reduction_factor"],
                    metrics["actual_top_k"],
                    metrics["actual_top_k_ratio"],
                    metrics["perturbation_improvements"],
                    metrics["perturbation_attempts"],
                    metrics["near_miss_eligible_count"],
                    metrics["near_miss_attempt_count"],
                    metrics["near_miss_accept_count"],
                    metrics["near_miss_mean_attempt_relative_gap"],
                    metrics["epoch_runtime_seconds"],
                    bool(reference_metrics["reference_promoted"]),
                )
            summary = {
                "output_dir": str(self.output_dir),
                "best_exact_reference_score": (
                    None if not np.isfinite(self.best_score) else self.best_score
                ),
                "last_metrics": last_metrics,
                "reference_state": self.reference_state.to_dict(),
                "generator_stats": self.generator.stats.to_dict(),
                "env_params": self.env_params,
                "model_params": self.model_params,
                "optimizer_params": self.optimizer_params,
                "trainer_params": self.trainer_params,
            }
            (self.output_dir / "train_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            return summary
        finally:
            self.close()
