"""S500 trainer with a J- and epoch-dependent Expert-search recipe.

Every training batch has a single J, so the knobs below are set per batch
right after its J is sampled:

* large J (>= ``recipe_j_threshold``), ramp r = clip((epoch - ramp_start) /
  (ramp_end - ramp_start), 0, 1):
    - LKH start fraction  r * ``recipe_lkh_max_fraction`` (rest: 2-opt),
    - near-miss rate      adaptive + r * (1 - adaptive),
    - C0 refresh rounds   ``recipe_refresh_rounds`` once epoch > refresh_start;
* small J: 2-opt only, the S500 adaptive near-miss rate, no refresh.

IL mass normalisation: per J, the changed-Expert repeat weight is chosen so
that the expected IL samples per instance stay at ``recipe_il_mass_target``
(S500 level near E100: 0.11 * 5 + 0.89 ~= 1.44), clipped to
[1, changed_expert_il_weight]. Fractional weights use stochastic rounding with
a dedicated generator, so the global RNG streams are untouched.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from loguru import logger

from src.JointSelfImprovementTrainer import JointSelfImprovementTrainer


class ProgressiveRecipeTrainer(JointSelfImprovementTrainer):
    _STRICT_TRAJECTORY_PARAMETER_KEYS = JointSelfImprovementTrainer._STRICT_TRAJECTORY_PARAMETER_KEYS + (
        "recipe_j_threshold", "recipe_ramp_start", "recipe_ramp_end",
        "recipe_lkh_max_fraction", "recipe_refresh_start", "recipe_refresh_rounds",
        "recipe_il_mass_target",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        params = self.trainer_params
        self.recipe_j_threshold = int(params.get("recipe_j_threshold", 60))
        self.recipe_ramp_start = int(params.get("recipe_ramp_start", 40))
        self.recipe_ramp_end = int(params.get("recipe_ramp_end", 100))
        self.recipe_lkh_max_fraction = float(params.get("recipe_lkh_max_fraction", 0.5))
        self.recipe_refresh_start = int(params.get("recipe_refresh_start", 60))
        self.recipe_refresh_rounds = int(params.get("recipe_refresh_rounds", 2))
        self.recipe_il_mass_target = float(params.get("recipe_il_mass_target", 1.44))
        if self.recipe_ramp_end <= self.recipe_ramp_start:
            raise ValueError("recipe_ramp_end must exceed recipe_ramp_start")
        if not 0.0 <= self.recipe_lkh_max_fraction <= 1.0:
            raise ValueError("recipe_lkh_max_fraction must be in [0, 1]")
        if self.recipe_il_mass_target < 1.0:
            raise ValueError("recipe_il_mass_target must be at least 1")
        if tuple(self.si_move_types) != ("two_opt",):
            raise ValueError("the progressive recipe uses 2-opt as its move search")
        if self.si_local_search_backend != "swap_two_opt":
            raise ValueError("the hybrid recipe runs on the move-search backend")
        self.si_lkh_work_root.mkdir(parents=True, exist_ok=True)
        self.si_lkh_start_fraction = 0.0
        self._recipe_epoch = int(getattr(self, "start_epoch", 1))
        self._recipe_stats = defaultdict(float)
        logger.info(
            "Progressive recipe: J>={} ramp E{}->E{} LKH<= {:.2f} near-miss->1.0, "
            "refresh {} rounds after E{}, IL mass target {:.2f}",
            self.recipe_j_threshold, self.recipe_ramp_start, self.recipe_ramp_end,
            self.recipe_lkh_max_fraction, self.recipe_refresh_rounds,
            self.recipe_refresh_start, self.recipe_il_mass_target,
        )

    # ------------------------------------------------------------------ knobs
    def _recipe_ramp(self, epoch: int) -> float:
        span = self.recipe_ramp_end - self.recipe_ramp_start
        return min(max((epoch - self.recipe_ramp_start) / span, 0.0), 1.0)

    def recipe_knobs(self, problem_size: int, epoch: int, adaptive_rate: float) -> dict:
        if problem_size < self.recipe_j_threshold:
            return {"lkh_fraction": 0.0, "near_miss_rate": float(adaptive_rate), "refresh_rounds": 0}
        ramp = self._recipe_ramp(epoch)
        return {
            "lkh_fraction": ramp * self.recipe_lkh_max_fraction,
            "near_miss_rate": float(adaptive_rate) + ramp * (1.0 - float(adaptive_rate)),
            "refresh_rounds": self.recipe_refresh_rounds if epoch > self.recipe_refresh_start else 0,
        }

    def _sample_problem_size(self) -> int:
        problem_size = super()._sample_problem_size()
        knobs = self.recipe_knobs(problem_size, self._recipe_epoch, self._effective_near_miss_rate())
        self.si_lkh_start_fraction = knobs["lkh_fraction"]
        self._active_near_miss_rate = knobs["near_miss_rate"]
        self.si_c0_refresh_rounds = knobs["refresh_rounds"]
        stats = self._recipe_stats
        stats["batches"] += 1
        stats["large_batches"] += float(problem_size >= self.recipe_j_threshold)
        stats["lkh_fraction_sum"] += knobs["lkh_fraction"]
        stats["near_miss_rate_sum"] += knobs["near_miss_rate"]
        stats["refresh_batches"] += float(knobs["refresh_rounds"] > 0)
        return problem_size

    # --------------------------------------------------------------- IL mass
    def recipe_il_weights(self, instances, changed_flags) -> dict[int, float]:
        counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        for instance, changed in zip(instances, changed_flags):
            counts[int(instance.J)][0] += 1
            counts[int(instance.J)][1] += int(bool(changed))
        weights = {}
        for problem_size, (total, changed) in counts.items():
            if changed == 0:
                weights[problem_size] = float(self.changed_expert_il_weight)
                continue
            fraction = changed / total
            raw = (self.recipe_il_mass_target - (1.0 - fraction)) / fraction
            weights[problem_size] = float(min(max(raw, 1.0), self.changed_expert_il_weight))
        return weights

    def _imitation_update(self, instances, expert_sequences, expert_changed_flags):
        weights = self.recipe_il_weights(instances, expert_changed_flags)
        rng = np.random.default_rng(
            [int(self.trainer_params.get("seed", 0)) & 0xFFFFFFFF, int(self._recipe_epoch), 0x11]
        )
        expanded_instances, expanded_experts = [], []
        changed_total = 0
        for instance, expert, changed in zip(instances, expert_sequences, expert_changed_flags):
            if changed:
                weight = weights[int(instance.J)]
                repeats = int(math.floor(weight)) + int(rng.random() < weight - math.floor(weight))
                changed_total += 1
            else:
                repeats = int(self.unchanged_expert_il_weight)
            expanded_instances.extend([instance] * repeats)
            expanded_experts.extend([list(expert)] * repeats)
        extra_instances, extra_experts = self._extra_il_examples()
        expanded_instances.extend(extra_instances)
        expanded_experts.extend(extra_experts)
        large = [w for j, w in weights.items() if j >= self.recipe_j_threshold]
        small = [w for j, w in weights.items() if j < self.recipe_j_threshold]
        self._recipe_stats["il_mass_per_instance"] = len(expanded_instances) / max(len(instances), 1)
        self._recipe_stats["il_changed_ratio"] = changed_total / max(len(instances), 1)
        self._recipe_stats["il_weight_small_j"] = float(np.mean(small)) if small else float("nan")
        self._recipe_stats["il_weight_large_j"] = float(np.mean(large)) if large else float("nan")
        # Repeats are already applied: pass every copy as weight-1 "unchanged"
        # so the parent neither re-weights nor double-counts them.
        saved = self.unchanged_expert_il_weight
        self.unchanged_expert_il_weight = 1
        try:
            return super()._imitation_update(
                expanded_instances, expanded_experts, [False] * len(expanded_instances)
            )
        finally:
            self.unchanged_expert_il_weight = saved

    def _extra_il_examples(self):
        """Additional weight-1 IL examples appended after mass normalisation."""
        return [], []

    # ----------------------------------------------------------------- epoch
    def _train_one_epoch(self, epoch: int):
        self._recipe_epoch = int(epoch)
        self._recipe_stats = defaultdict(float)
        metrics = super()._train_one_epoch(epoch)
        stats = self._recipe_stats
        batches = max(stats["batches"], 1.0)
        metrics.update({
            "recipe_ramp": self._recipe_ramp(epoch),
            "recipe_large_j_batch_ratio": stats["large_batches"] / batches,
            "recipe_mean_lkh_fraction": stats["lkh_fraction_sum"] / batches,
            "recipe_mean_near_miss_rate": stats["near_miss_rate_sum"] / batches,
            "recipe_refresh_batch_ratio": stats["refresh_batches"] / batches,
            "recipe_il_mass_per_instance": stats["il_mass_per_instance"],
            "recipe_il_changed_ratio": stats["il_changed_ratio"],
            "recipe_il_weight_small_j": stats["il_weight_small_j"],
            "recipe_il_weight_large_j": stats["il_weight_large_j"],
        })
        return metrics
