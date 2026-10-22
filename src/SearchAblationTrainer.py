"""Fresh-run, three-condition search experiment with isolated diagnostics.

The legacy trainer remains the default. All extra evaluation uses disposable
policy/environment copies and restores every training RNG.
"""
from __future__ import annotations

import copy
import hashlib
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from src.CertifiedUpperBound import is_optimal_gurobi_solution
from src.FixedValidation import load_or_create_fixed_validation_sets
from src.JointSelfImprovement import (TargetRewardSolveTask, batched_directed_step_costs,
                                    run_target_reward_solve_task)
from src.JointSelfImprovementTrainer import JointSelfImprovementTrainer
from src.ReferencePolicy import clone_reference_policy
from src.SearchDiagnostics import (SearchAccumulator, atomic_json, fingerprint,
                                   finite_summary, move_budget, tensor_digest)
from src.TSPEnv import TSPEnv, rollout_batch
from src.TSPUtils import set_seed
from src.GPUPhaseGate import gpu_phase_slot


class SearchAblationTrainer(JointSelfImprovementTrainer):
    _STRICT_TRAJECTORY_PARAMETER_KEYS = JointSelfImprovementTrainer._STRICT_TRAJECTORY_PARAMETER_KEYS + (
        "ablation_group", "move_schedule_horizon", "move_schedule_final",
        "diagnostic_dir", "diagnostic_epochs",
        "gpu_max_active", "gpu_gate_dir",
    )

    def _strict_resume_signature(self):
        signature = super()._strict_resume_signature()
        # The 20-epoch execution cap is a checkpoint boundary in the 100-epoch
        # experiment. Extending that cap must not reset or reject its schedule.
        signature["planned_epochs"] = int(self.trainer_params["move_schedule_horizon"])
        return signature

    @contextmanager
    def _gpu_phase(self):
        if self.device.type != "cuda":
            yield
            return
        directory = self.trainer_params.get("gpu_gate_dir", "outputs/search_ablation_20261012/.gpu_slots")
        with gpu_phase_slot(directory, int(self.trainer_params.get("gpu_max_active", 2))) as waited:
            self._timings["gpu_wait"] += waited
            try:
                yield
            finally:
                # Release inactive allocator blocks before another job enters.
                torch.cuda.empty_cache()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._move_start = int(self.trainer_params["self_improvement_max_local_moves"])
        self._move_final = int(self.trainer_params["move_schedule_final"])
        self._move_horizon = int(self.trainer_params["move_schedule_horizon"])
        self._diagnostic_epochs = set(self.trainer_params["diagnostic_epochs"])
        self._diagnostic_groups = {}
        directory = self.trainer_params.get("diagnostic_dir")
        if directory:
            self._diagnostic_groups = load_or_create_fixed_validation_sets(
                dataset_dir=self.env_params["dataset_dir"], validation_dir=directory,
                problem_sizes=(20, 40, 60, 80, 100), per_size=20, seed=20261012,
            )
        self._diag_epoch = 0
        self._phase = "idle"
        self._record_search = False
        self._epoch_stats = None
        self._timings = defaultdict(float)
        self._gradient_rows = []
        self._component_rows = []
        self._sample_counts = defaultdict(int)
        self._winner_histograms = defaultdict(list)
        self._initial_manifest = {
            "fresh_initialization": not bool(self.trainer_params.get("checkpoint_path")),
            "seed": int(self.trainer_params["seed"]), "group": self.trainer_params["ablation_group"],
            "target_sha256": tensor_digest(self.model.state_dict()),
            "reference_sha256": tensor_digest(self.reference_policy.state_dict()),
            "predictor_sha256": tensor_digest(self.edge_decoder.state_dict()),
            "replay_size": len(self._expert_replay_pool), "start_epoch": self.start_epoch,
            "lr_horizon": self.scheduler_params["total_epochs"],
            "move_horizon": self._move_horizon, "min_lr": self.scheduler_params["min_lr"],
            "data_rng_sha256": hashlib.sha256(repr(self.generator.rng.bit_generator.state).encode()).hexdigest(),
            "problem_size_rng_sha256": tensor_digest({"state": self._problem_size_rng.get_state()}),
        }
        if not self.trainer_params.get("checkpoint_path"):
            atomic_json(self.output_dir / "initialization.json", self._initial_manifest)
            logger.info("Fresh ablation {}: init={}, replay=0, LR horizon=100, move horizon=100",
                        self.trainer_params["ablation_group"], self._initial_manifest["target_sha256"])

    @contextmanager
    def _isolated_evaluation(self, seed):
        with self._gpu_phase():
            with self._isolated_evaluation_unlocked(seed):
                yield

    @contextmanager
    def _isolated_evaluation_unlocked(self, seed):
        rng = self._capture_rng_state()
        original_model, original_env = self.model, self.env
        record = self._record_search
        active_rate = self._active_near_miss_rate
        try:
            set_seed(seed)
            self.model = clone_reference_policy(original_model, device=self.device)
            self.env = TSPEnv(problem_size=100, pomo_size=100, start_node_strategy="nearest_depot")
            self._record_search = False
            yield
        finally:
            self.model, self.env = original_model, original_env
            self._active_near_miss_rate = active_rate
            self._record_search = record
            self._restore_rng_state(rng)

    def _exact_solve(self, instances, route_lists):
        tasks, mappings = [], []
        for instance, routes in zip(instances, route_lists):
            seen = {}
            mapping = []
            for route in routes:
                key = tuple(route)
                if key not in seen:
                    seen[key] = len(tasks)
                    tasks.append(TargetRewardSolveTask(instance, list(route), self.si_final_solver_config))
                mapping.append(seen[key])
            mappings.append(mapping)
        if self._si_executor is None:
            solutions = [run_target_reward_solve_task(task) for task in tasks]
        else:
            solutions = list(self._si_executor.map(run_target_reward_solve_task, tasks, chunksize=1))
        if any(not is_optimal_gurobi_solution(s) or not np.isfinite(s.objective) for s in solutions):
            raise RuntimeError("nonoptimal Gurobi solve in isolated diagnostic")
        return [np.asarray([solutions[i].objective for i in mapping], dtype=float)
                for mapping in mappings], len(tasks)

    def _stage_evaluation(self, stage):
        if self._diag_epoch not in self._diagnostic_epochs or not self._diagnostic_groups:
            return
        modes = ["free_greedy"]
        if self._diag_epoch in (5, 20) and stage in ("after_rl", "after_il"):
            modes.append("full_pomo_greedy")
        started = time.perf_counter()
        for mode in modes:
            path = self.output_dir / "diagnostics" / f"epoch-{self._diag_epoch:03d}_{stage}_{mode}.json"
            records, calls = [], 0
            with self._isolated_evaluation(20261012 + self._diag_epoch):
                self.model.eval()
                with torch.no_grad():
                    for size, instances in sorted(self._diagnostic_groups.items()):
                        instances = instances if mode == "free_greedy" else instances[:10]
                        for offset in range(0, len(instances), 8):
                            batch = instances[offset:offset + 8]
                            rollout = rollout_batch(self.model, batch, device=self.device,
                                                    decode_type="greedy", use_pomo_start=mode != "free_greedy")
                            values, count = self._exact_solve(batch, rollout.to_sequence_lists())
                            calls += count
                            records.extend({"fingerprint": fingerprint(instance), "J": size,
                                            "makespan": float(cost.min()), "routes": len(cost)}
                                           for instance, cost in zip(batch, values))
            atomic_json(path, {"epoch": self._diag_epoch, "stage": stage, "mode": mode,
                               "gurobi_calls": calls, "mean": float(np.mean([r["makespan"] for r in records])),
                               "by_J": {str(j): finite_summary([r["makespan"] for r in records if r["J"] == j])
                                        for j in self._diagnostic_groups}, "records": records})
        self._timings["stage_diagnostics"] += time.perf_counter() - started

    def _reward_diagnostic(self):
        if self._diag_epoch not in self._diagnostic_epochs or not self._diagnostic_groups:
            return
        started = time.perf_counter()
        records, calls = [], 0
        with self._isolated_evaluation(20271012 + self._diag_epoch):
            # Match the actual training sample/dropout mode, but isolate its RNG and caches.
            self.model.train()
            with torch.no_grad():
                for size, group in sorted(self._diagnostic_groups.items()):
                    for offset in range(0, min(10, len(group)), 5):
                        instances = group[offset:offset + 5]
                        self.env.load_problems(instances, self.device)
                        state, _, _ = self.env.reset()
                        self.model.pre_forward(state.node_features)
                        predicted = self._predict_contribution_matrices(instances)
                        rollout, _ = self._rollout_target()
                        routes = rollout.to_sequence_lists()
                        reference = self._generate_reference_candidates(instances)
                        anchors, _, _ = self._select_anchor_sequences(instances, reference, predicted)
                        results, _, _ = self._run_sparse_self_improvement(instances, reference, anchors)
                        bounds = batched_directed_step_costs(routes, [r.anchor_upper_bound_matrix for r in results]).sum(-1).astype(np.float32)
                        for i, result in enumerate(results):
                            if result.expert_upper_bound_matrix is not None:
                                tighter = batched_directed_step_costs([routes[i]], [result.expert_upper_bound_matrix])[0].sum(-1).astype(np.float32)
                                bounds[i] = np.minimum(bounds[i], tighter)
                        values, count = self._exact_solve(instances, routes)
                        calls += count + sum(r.total_gurobi_calls for r in results)
                        for instance, upper, exact, result in zip(instances, bounds, values, results):
                            if np.any(exact > upper.astype(float) + self.si_upper_bound_tolerance):
                                raise RuntimeError("diagnostic strict upper-bound violation")
                            selected = int(np.argmin(upper))
                            mixed = upper.astype(float).copy()
                            mixed[selected] = exact[selected]
                            mixed_a = mixed.mean() - mixed
                            exact_a = exact.mean() - exact
                            records.append({"fingerprint": fingerprint(instance), "J": size,
                                            "selected_rank": 1 + int(np.sum(exact < exact[selected] - 1e-6)),
                                            "selected_exact": float(exact[selected]), "best_exact": float(exact.min()),
                                            "regret": float(exact[selected] - exact.min()),
                                            "regret_percent": float(100 * (exact[selected] - exact.min()) / max(abs(exact.min()), 1e-12)),
                                            "mean_bound_gap": float(np.mean(upper.astype(float) - exact)),
                                            "advantage_sign_agreement": float(np.mean(np.sign(mixed_a) == np.sign(exact_a))),
                                            "bounds": upper.tolist(), "exact": exact.tolist(),
                                            "expert_exact": result.expert_exact_makespan})
        self._timings["reward_diagnostics"] += time.perf_counter() - started
        atomic_json(self.output_dir / "diagnostics" / f"epoch-{self._diag_epoch:03d}_reward_ranking.json",
                    {"epoch": self._diag_epoch, "gurobi_calls": calls, "records": records,
                     "by_J": {str(j): {key: finite_summary([r[key] for r in records if r["J"] == j])
                                        for key in ("regret", "regret_percent", "selected_rank", "mean_bound_gap", "advantage_sign_agreement")}
                              for j in self._diagnostic_groups}})

    def _run_sparse_self_improvement(self, instances, reference_sequences, anchor_sequences):
        results, wall, eligible = super()._run_sparse_self_improvement(instances, reference_sequences, anchor_sequences)
        if self._record_search:
            self._timings["sparse_wall"] += wall
            for instance, result in zip(instances, results):
                self._epoch_stats.add(instance, result)
                matching = [chain for chain in result.local_search_results if chain.final_sequence == result.expert_sequence]
                winner = min(matching, key=lambda x: x.final_upper_bound) if matching else None
                self._winner_histograms[str(instance.J)].append({
                    "steps": winner.accepted if winner else None, "source": result.expert_source,
                    "exact_gain": result.exact_improvement,
                })
                if self._sample_counts[instance.J] < 2:
                    self._sample_counts[instance.J] += 1
                    path = self.output_dir / "diagnostics" / f"epoch-{self._diag_epoch:03d}_search_samples.jsonl"
                    row = {"fingerprint": fingerprint(instance), "J": instance.J,
                           "expert_source": result.expert_source, "anchor_exact": result.anchor_exact_makespan,
                           "expert_exact": result.expert_exact_makespan,
                           "winner": winner.to_dict() if winner else None,
                           "chains": [c.to_dict() for c in result.local_search_results]}
                    import json
                    with path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
        return results, wall, eligible

    def _clip_main_gradients(self):
        before = super()._clip_main_gradients()
        gradients = [p.grad.detach().norm(2) for p in self.model.parameters() if p.grad is not None]
        after = torch.stack(gradients).norm(2) if gradients else torch.tensor(0.)
        self._gradient_rows.append({"phase": self._phase, "step": self.global_step,
                                    "before": float(before), "after": float(after)})
        return before

    def _observe_rl_losses(self, hybrid, exact, weight, advantages, exact_advantages):
        row = {"batch": self._joint_batch_index, "weight": weight,
               "dense_advantage": finite_summary(advantages.detach().cpu().numpy()) if advantages is not None else {},
               "exact_advantage": finite_summary(exact_advantages) if exact_advantages is not None else {}}
        if self._diag_epoch in self._diagnostic_epochs and not self._component_rows:
            params = tuple(self.model.parameters())
            a = torch.autograd.grad(hybrid, params, retain_graph=True, allow_unused=True)
            b = torch.autograd.grad(exact, params, retain_graph=True, allow_unused=True)
            norm_a = sum((x.detach().square().sum() for x in a if x is not None), torch.tensor(0., device=self.device)).sqrt()
            norm_b = sum((x.detach().square().sum() for x in b if x is not None), torch.tensor(0., device=self.device)).sqrt()
            dot = sum((x.detach().mul(y.detach()).sum() for x, y in zip(a, b) if x is not None and y is not None), torch.tensor(0., device=self.device))
            row.update({"hybrid_gradient_norm": float(norm_a), "exact_gradient_norm": float(norm_b),
                        "weighted_exact_gradient_norm": float(weight * norm_b),
                        "cosine": float(dot / (norm_a * norm_b).clamp_min(1e-12))})
        self._component_rows.append(row)

    def _parameter_snapshot(self):
        return {name: p.detach().clone() for name, p in self.model.named_parameters()}

    def _parameter_delta(self, snapshot):
        return float(sum(((p.detach() - snapshot[name]).square().sum()
                          for name, p in self.model.named_parameters()),
                         torch.tensor(0., device=self.device)).sqrt())

    def _imitation_update(self, *args, **kwargs):
        self._rl_delta = self._parameter_delta(self._phase_snapshot)
        self._phase_snapshot = None
        self._stage_evaluation("after_rl")
        self._phase = "il"
        snapshot = self._parameter_snapshot()
        started = time.perf_counter()
        with self._gpu_phase():
            result = super()._imitation_update(*args, **kwargs)
        self._timings["il"] += time.perf_counter() - started
        self._il_delta = self._parameter_delta(snapshot)
        del snapshot
        self._stage_evaluation("after_il")
        self._phase = "edge"
        return result

    def _policy_rl_update(self, *args, **kwargs):
        started = time.perf_counter()
        result = super()._policy_rl_update(*args, **kwargs)
        self._timings["rl_with_target_solve"] += time.perf_counter() - started
        return result

    def _generate_reference_candidates(self, *args, **kwargs):
        started = time.perf_counter()
        result = super()._generate_reference_candidates(*args, **kwargs)
        if self._record_search:
            self._timings["reference_sampling"] += time.perf_counter() - started
        return result

    def _rollout_target(self, *args, **kwargs):
        started = time.perf_counter()
        result = super()._rollout_target(*args, **kwargs)
        if self._record_search:
            self._timings["target_sampling"] += time.perf_counter() - started
        return result

    def _train_one_batch(self, batch_size):
        with self._gpu_phase():
            result = super()._train_one_batch(batch_size)
        fingerprints = [fingerprint(row[0]) for row in self._last_batch_training_records]
        self._data_digest.update("".join(fingerprints).encode())
        import json
        with (self.output_dir / "diagnostics" / f"epoch-{self._diag_epoch:03d}_batch_data.jsonl").open("a") as handle:
            handle.write(json.dumps({"batch": self._joint_batch_index, "J": int(result["problem_size"]),
                                     "count": batch_size, "sha256": hashlib.sha256("".join(fingerprints).encode()).hexdigest()}) + "\n")
        return result

    def _train_one_epoch(self, epoch):
        self._diag_epoch = epoch
        self.si_max_local_moves = move_budget(epoch, self._move_start, self._move_final, self._move_horizon)
        self.si_local_search_config = replace(self.si_local_search_config, max_local_moves=self.si_max_local_moves)
        self._epoch_stats = SearchAccumulator(self.si_max_local_moves, self.si_move_types)
        self._timings, self._gradient_rows, self._component_rows = defaultdict(float), [], []
        self._winner_histograms, self._sample_counts = defaultdict(list), defaultdict(int)
        self._data_digest = hashlib.sha256()
        (self.output_dir / "diagnostics").mkdir(exist_ok=True)
        self._record_search = False
        started = time.perf_counter()
        self._stage_evaluation("before_rl")
        self._phase = "rl"
        self._phase_snapshot = self._parameter_snapshot()
        self._record_search = True
        logger.info(
            "Experiment {} epoch {}/{}: local search {}, move cap {}, LR {:.8g}, LR/move horizons={}/{}",
            self.trainer_params["ablation_group"], epoch, self.epochs,
            self.si_local_search_backend, self.si_max_local_moves,
            self.optimizer.param_groups[0]["lr"], self.scheduler_params["total_epochs"],
            self._move_horizon,
        )
        metrics = super()._train_one_epoch(epoch)
        self._record_search = False
        self._reward_diagnostic()
        payload = {"epoch": epoch, "move_cap": self.si_max_local_moves,
                   "learning_rate_used": self.optimizer.param_groups[0]["lr"],
                   "train_data_sha256": self._data_digest.hexdigest(),
                   "search": self._epoch_stats.summary(), "winner_chains": dict(self._winner_histograms),
                   "gradients": self._gradient_rows, "rl_components": self._component_rows,
                   "rl_parameter_delta_l2": self._rl_delta, "il_parameter_delta_l2": self._il_delta,
                   "timings": dict(self._timings), "wall_seconds_including_diagnostics": time.perf_counter() - started}
        atomic_json(self.output_dir / "diagnostics" / f"epoch-{epoch:03d}_summary.json", payload)
        metrics["max_local_moves"] = float(self.si_max_local_moves)
        metrics["diagnostic_wall_seconds"] = float(self._timings["stage_diagnostics"] + self._timings["reward_diagnostics"])
        metrics["train_data_sha256"] = self._data_digest.hexdigest()
        return metrics
