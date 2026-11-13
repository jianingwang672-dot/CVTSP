"""Progressive recipe plus certified per-start self-improvement IL labels.

After each batch's RL step, ``per_start_label_count`` of the batch's sampled
Target POMO routes per instance (distinct start nodes, dedicated RNG) are
solved exactly; each is then 2-opt improved with its first target fixed on the
C0 table of its *own* exact solution. Only certified improvements
(C0(label) < exact(original)) become labels, so every label is a strictly
better route from the same start. They join the epoch's IL with weight 1,
after the recipe's IL mass normalisation of the Expert labels. The intent is
to train each start towards a better version of its own route instead of
pulling every start towards the single Expert route.
"""

from __future__ import annotations

import numpy as np

from src.JointSelfImprovement import PerStartLabelTask, run_per_start_label_task
from src.ProgressiveRecipeTrainer import ProgressiveRecipeTrainer


class PerStartILTrainer(ProgressiveRecipeTrainer):
    _STRICT_TRAJECTORY_PARAMETER_KEYS = ProgressiveRecipeTrainer._STRICT_TRAJECTORY_PARAMETER_KEYS + (
        "per_start_label_count", "per_start_max_moves",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.per_start_label_count = int(self.trainer_params.get("per_start_label_count", 2))
        self.per_start_max_moves = int(self.trainer_params.get("per_start_max_moves", 30))
        if self.per_start_label_count < 1:
            raise ValueError("per_start_label_count must be positive")
        self._per_start_labels: list[tuple[object, list[int]]] = []
        self._per_start_stats = {"attempts": 0, "labels": 0, "failures": 0, "improvement_pct_sum": 0.0,
                                 "calls": 0}

    def _per_start_tasks(self, target_sequences, instances):
        rng = np.random.default_rng(
            [int(self.trainer_params.get("seed", 0)) & 0xFFFFFFFF, int(self._joint_batch_index), 0xC1]
        )
        tasks = []
        for instance, routes in zip(instances, target_sequences):
            count = min(self.per_start_label_count, len(routes))
            chosen = rng.choice(len(routes), size=count, replace=False)
            tasks.append(PerStartLabelTask(
                instance=instance,
                routes=[list(map(int, routes[int(k)])) for k in chosen],
                solver_config=self.si_final_solver_config,
                max_moves=self.per_start_max_moves,
                absolute_epsilon=float(self.si_local_search_config.absolute_epsilon),
                relative_epsilon=float(self.si_local_search_config.relative_epsilon),
                feasibility_tolerance=self.si_feasibility_tolerance,
                upper_bound_tolerance=self.si_upper_bound_tolerance,
            ))
        return tasks

    def _policy_rl_update(self, target_step_log_probs, target_sequences, results, instances):
        output = super()._policy_rl_update(target_step_log_probs, target_sequences, results, instances)
        tasks = self._per_start_tasks(target_sequences, instances)
        if self._si_executor is None:
            outcomes = [run_per_start_label_task(task) for task in tasks]
        else:
            outcomes = list(self._si_executor.map(run_per_start_label_task, tasks, chunksize=1))
        stats = self._per_start_stats
        for instance, labels in zip(instances, outcomes):
            for label in labels:
                stats["attempts"] += 1
                stats["calls"] += 1
                if label.solve_failed:
                    stats["failures"] += 1
                    continue
                if label.label_sequence is not None:
                    stats["labels"] += 1
                    stats["improvement_pct_sum"] += 100.0 * (
                        label.original_exact - label.label_upper_bound) / label.original_exact
                    self._per_start_labels.append((instance, list(label.label_sequence)))
        return output

    def _extra_il_examples(self):
        labels, self._per_start_labels = self._per_start_labels, []
        return [inst for inst, _ in labels], [route for _, route in labels]

    def _train_one_epoch(self, epoch: int):
        self._per_start_labels = []
        self._per_start_stats = {key: 0 if key != "improvement_pct_sum" else 0.0
                                 for key in self._per_start_stats}
        metrics = super()._train_one_epoch(epoch)
        stats = self._per_start_stats
        attempts = max(stats["attempts"], 1)
        instances = max(float(self.train_episodes), 1.0)
        metrics.update({
            "per_start_attempts": float(stats["attempts"]),
            "per_start_labels": float(stats["labels"]),
            "per_start_label_ratio": stats["labels"] / attempts,
            "per_start_failures": float(stats["failures"]),
            "per_start_certified_improvement_pct": stats["improvement_pct_sum"] / max(stats["labels"], 1),
            "per_start_gurobi_calls_per_instance": stats["calls"] / instances,
        })
        return metrics
