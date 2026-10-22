"""Experiment B: exact post-RL label quality, frozen across the five IL passes."""
from __future__ import annotations

import json
import random
import time
from collections import defaultdict

import numpy as np
import torch
from loguru import logger

from src.ImitationLearning import teacher_forcing_loss
from src.JointSelfImprovement import batched_directed_step_costs
from src.NearMissQuotaTrainer import NearMissQuotaTrainer
from src.SearchDiagnostics import atomic_json, finite_summary, fingerprint, tensor_digest
from src.TSPEnv import rollout_batch


def quality_multiplier(target, expert, scale=.02, offset=.5, minimum=.1, maximum=2.):
    target, expert = float(target), float(expert)
    if not np.isfinite(target) or not np.isfinite(expert) or target <= 0 or expert <= 0:
        raise ValueError("IL quality requires valid positive exact makespans")
    if scale <= 0 or not 0 < minimum <= maximum:
        raise ValueError("invalid quality multiplier parameters")
    gap = (target - expert) / max(abs(target), 1e-12)
    return gap, float(np.clip(offset + gap / scale, minimum, maximum))


class QualityILTrainer(NearMissQuotaTrainer):
    _STRICT_TRAJECTORY_PARAMETER_KEYS = NearMissQuotaTrainer._STRICT_TRAJECTORY_PARAMETER_KEYS + (
        "quality_weighted_il", "quality_il_scale", "quality_il_offset",
        "quality_il_min", "quality_il_max", "quality_il_measurement", "quality_il_batch_size",
    )

    def __init__(self, *args, **kwargs):
        self._label_cache = {}
        super().__init__(*args, **kwargs)
        if not self.trainer_params.get("quality_weighted_il") or self.trainer_params["near_miss_ablation_rate"] != .2:
            raise ValueError("Experiment B requires quality weighting and fixed near-miss .20")
        if self.model.config.block_size != 1:
            raise ValueError("quality IL experiment requires ordered token CE")
        checkpoint_path = self.trainer_params.get("checkpoint_path")
        if checkpoint_path:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if "quality_il_label_cache" not in payload:
                raise KeyError("quality IL checkpoint lacks replay exact values and bounds")
            self._label_cache = payload["quality_il_label_cache"]
        self._initial_manifest.update({"group": "quality_b", "experiment": "quality_weighted_il",
            "formula": "clip(0.5 + ((VT-VE)/abs(VT))/0.02, 0.1, 2)",
            "quality_measurement": self.trainer_params["quality_il_measurement"],
            "reference_run": "outputs/near_miss_quota_20261013/nm20"})
        if not checkpoint_path:
            atomic_json(self.output_dir / "initialization.json", self._initial_manifest)
        logger.info("QUALITY_IL_B: fresh={}, IL passes={}, near-miss=20%, post-RL quality frozen per epoch",
                    not bool(checkpoint_path), self.imitation_inner_epochs)

    @staticmethod
    def _label_key(instance, expert):
        return fingerprint(instance), tuple(map(int, expert))

    def _run_sparse_self_improvement(self, instances, reference_sequences, anchor_sequences):
        result = super()._run_sparse_self_improvement(instances, reference_sequences, anchor_sequences)
        if self._record_search:
            for instance, label in zip(instances, result[0]):
                value = float(label.expert_exact_makespan)
                if not np.isfinite(value) or value <= 0:
                    raise RuntimeError("quality IL received invalid exact Expert")
                self._label_cache[self._label_key(instance, label.expert_sequence)] = {
                    "expert_exact": value,
                    "anchor_matrix": np.asarray(label.anchor_upper_bound_matrix, dtype=np.float64),
                    "expert_matrix": (None if label.expert_upper_bound_matrix is None else
                                      np.asarray(label.expert_upper_bound_matrix, dtype=np.float64)),
                }
        return result

    def _prepare_il_rows(self, instances, experts, changed):
        if not (len(instances) == len(experts) == len(changed)):
            raise ValueError("unaligned quality IL labels")
        rows = []
        for instance, expert, improved in zip(instances, experts, changed):
            source = "current_changed" if improved else "current_anchor"
            rows.extend([(instance, expert, source, self._label_key(instance, expert))] *
                        (self.changed_expert_il_weight if improved else 1))
        current = len(rows)
        replay_count = (int(round(current * self.expert_replay_fraction / (1 - self.expert_replay_fraction)))
                        if self.expert_replay_fraction > 0 and self._expert_replay_pool else 0)
        for _ in range(replay_count):
            instance, expert = self._expert_replay_pool[random.randrange(len(self._expert_replay_pool))]
            rows.append((instance, expert, "replay", self._label_key(instance, expert)))
        return rows, replay_count

    def _measure_label_quality(self, rows):
        """One exact Target solve per distinct label; restore all training RNG/caches."""
        started = time.perf_counter()
        unique = {}
        for row in rows:
            unique.setdefault(row[3], row)
        by_size = defaultdict(list)
        for key, row in unique.items():
            if key not in self._label_cache:
                raise RuntimeError("quality IL label missing its certified bound/exact value")
            by_size[row[0].J].append((key, row[0]))
        measured, calls = {}, 0
        snapshot_hash = tensor_digest(self.model.state_dict())
        rng_before = self._capture_rng_state()
        with self._isolated_evaluation(20281014 + self._diag_epoch):
            # Same sampling/dropout mode and J POMO starts as the training reward.
            self.model.train()
            with torch.no_grad():
                for size, items in sorted(by_size.items()):
                    for offset in range(0, len(items), self.trainer_params["quality_il_batch_size"]):
                        batch = items[offset:offset + self.trainer_params["quality_il_batch_size"]]
                        keys, instances = zip(*batch)
                        routes = rollout_batch(self.model, list(instances), device=self.device,
                                               decode_type="sample", use_pomo_start=True).to_sequence_lists()
                        if any(len(tours) != size for tours in routes):
                            raise RuntimeError("quality IL comparison must use POMO=J")
                        selected, selected_bounds = [], []
                        for key, tours in zip(keys, routes):
                            cache = self._label_cache[key]
                            bound = batched_directed_step_costs([tours], [cache["anchor_matrix"]])[0].sum(-1).astype(np.float32)
                            if cache["expert_matrix"] is not None:
                                other = batched_directed_step_costs([tours], [cache["expert_matrix"]])[0].sum(-1).astype(np.float32)
                                bound = np.minimum(bound, other)
                            index = int(np.argmin(bound))
                            selected.append([tours[index]])
                            selected_bounds.append(float(bound[index]))
                        values, count = self._exact_solve(list(instances), selected)
                        calls += count
                        for key, cost, upper, target_tours in zip(keys, values, selected_bounds, selected):
                            target = float(cost[0])
                            if target > upper + self.si_upper_bound_tolerance:
                                raise RuntimeError("post-RL Target exact value violates certified bound")
                            expert = self._label_cache[key]["expert_exact"]
                            gap, q = quality_multiplier(target, expert,
                                self.trainer_params["quality_il_scale"], self.trainer_params["quality_il_offset"],
                                self.trainer_params["quality_il_min"], self.trainer_params["quality_il_max"])
                            measured[key] = {"target_exact": target, "expert_exact": expert,
                                             "target_sequence": list(target_tours[0]),
                                             "target_selected_upper_bound": upper,
                                             "relative_gap": gap, "quality_multiplier": q}
        if snapshot_hash != tensor_digest(self.model.state_dict()):
            raise RuntimeError("quality measurement changed the training model")
        rng_after = self._capture_rng_state()
        for key in ("torch_cpu", "problem_size_generator"):
            if not torch.equal(rng_before[key], rng_after[key]):
                raise RuntimeError(f"quality measurement changed training RNG: {key}")
        if rng_before["python_random"] != rng_after["python_random"] or rng_before["online_generator_bit_state"] != rng_after["online_generator_bit_state"]:
            raise RuntimeError("quality measurement changed the data/replay RNG")
        before_np, after_np = rng_before["numpy_global"], rng_after["numpy_global"]
        if before_np[0] != after_np[0] or not np.array_equal(before_np[1], after_np[1]) or before_np[2:] != after_np[2:]:
            raise RuntimeError("quality measurement changed the numpy RNG")
        if any(not torch.equal(a,b) for a,b in zip(rng_before.get("torch_cuda_all",[]),rng_after.get("torch_cuda_all",[]))):
            raise RuntimeError("quality measurement changed CUDA RNG")
        if calls != len(unique):
            raise RuntimeError("quality IL exact comparison budget mismatch")
        return measured, {"extra_target_gurobi_calls": calls, "unique_labels": len(unique),
                          "measurement_seconds": time.perf_counter() - started,
                          "post_rl_target_sha256": snapshot_hash,
                          "training_rng_restored": True,
                          "comparison": self.trainer_params["quality_il_measurement"]}

    def _weighted_imitation(self, rows, measured):
        groups = defaultdict(list)
        for index, row in enumerate(rows):
            groups[row[0].J].append(index)
        weights = [measured[row[3]]["quality_multiplier"] for row in rows]
        total_loss, total_mass, correct, tokens, steps = 0., 0., 0, 0, 0
        contribution = defaultdict(float)
        self.model.train()
        for _ in range(self._active_imitation_inner_epochs):
            for size, indices in sorted(groups.items()):
                for offset in range(0, len(indices), self.imitation_batch_size):
                    logical = indices[offset:offset + self.imitation_batch_size]
                    denominator = sum(weights[i] for i in logical)
                    self.optimizer.zero_grad()
                    for index in logical:
                        gap = measured[rows[index][3]]["relative_gap"]
                        quality = "better" if gap > 1e-6 else "worse" if gap < -1e-6 else "tie"
                        contribution[f"{rows[index][2]}:{quality}"] += weights[index] / denominator
                    for start in range(0, len(logical), self.imitation_micro_batch_size):
                        micro = logical[start:start + self.imitation_micro_batch_size]
                        local_weights = [weights[i] for i in micro]
                        loss, token_correct, _, token_count = teacher_forcing_loss(self.model,
                            [rows[i][0] for i in micro], [rows[i][1] for i in micro],
                            device=self.device, sample_weights=local_weights)
                        mass = sum(local_weights)
                        (loss * (mass / denominator)).backward()
                        total_loss += float(loss.detach()) * mass * size
                        total_mass += mass * size
                        correct += token_correct
                        tokens += token_count
                    self._clip_main_gradients()
                    self._step_main_optimizer()
                    steps += 1
        return total_loss / max(total_mass, 1e-12), correct / max(tokens, 1), {
            "optimizer_steps": steps,
            "normalized_ce_contribution_share": {key: value / max(steps, 1) for key, value in contribution.items()},
        }

    def _imitation_update(self, instances, expert_sequences, expert_changed_flags):
        self._rl_delta = self._parameter_delta(self._phase_snapshot)
        self._phase_snapshot = None
        self._stage_evaluation("after_rl")
        rows, replay_count = self._prepare_il_rows(instances, expert_sequences, expert_changed_flags)
        logger.info("QUALITY_IL_START epoch {}: current instances={}, effective labels={}, replay={}, measuring post-RL Target",
                    self._diag_epoch, len(instances), len(rows), replay_count)
        measured, audit = self._measure_label_quality(rows)
        self._phase = "il"
        snapshot = self._parameter_snapshot()
        started = time.perf_counter()
        with self._gpu_phase():
            loss, accuracy, weighted_audit = self._weighted_imitation(rows, measured)
        self._timings["il"] += time.perf_counter() - started
        self._timings["quality_measurement"] += audit["measurement_seconds"]
        self._il_delta = self._parameter_delta(snapshot)
        audit.update(weighted_audit)
        audit.update({"epoch": self._diag_epoch, "current_instances": len(instances),
                      "effective_labels": len(rows), "replay_examples": replay_count,
                      "passes": self._active_imitation_inner_epochs, "frozen_across_passes": True,
                      "quality_multiplier": finite_summary([measured[row[3]]["quality_multiplier"] for row in rows])})
        path = self.output_dir / "diagnostics" / f"epoch-{self._diag_epoch:03d}_il_labels.jsonl"
        with path.open("w") as handle:
            counts = defaultdict(int)
            row_values = {}
            for row in rows:
                key = row[3], row[2]
                counts[key] += 1
                row_values[key] = row
            for key, count in counts.items():
                row = row_values[key]
                handle.write(json.dumps({"fingerprint": row[3][0], "expert_sequence": list(row[3][1]),
                    "J": row[0].J, "source": row[2], "effective_copies": count, **measured[row[3]]}) + "\n")
        atomic_json(self.output_dir / "diagnostics" / f"epoch-{self._diag_epoch:03d}_quality_il.json", audit)
        logger.info("QUALITY_IL_DONE epoch {}: extra exact calls={}, quality time={:.1f}s, IL steps={}, weighted loss={:.6f}",
                    self._diag_epoch, audit["extra_target_gurobi_calls"], audit["measurement_seconds"], audit["optimizer_steps"], loss)
        self._stage_evaluation("after_il")
        self._phase = "edge"
        return loss, accuracy, len(rows), replay_count

    def _add_changed_experts_to_replay(self, instances, expert_sequences, expert_changed_flags):
        added = super()._add_changed_experts_to_replay(instances, expert_sequences, expert_changed_flags)
        retained = {self._label_key(instance, expert) for instance, expert in self._expert_replay_pool}
        self._label_cache = {key: self._label_cache[key] for key in retained}
        return added

    def _save_checkpoint(self, epoch, metrics, path):
        super()._save_checkpoint(epoch, metrics, path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["quality_il_label_cache"] = self._label_cache
        payload["quality_il_protocol"] = {key: value for key, value in self.trainer_params.items() if key.startswith("quality_")}
        temporary = path.with_suffix(path.suffix + ".quality.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def _train_one_epoch(self, epoch):
        metrics = super()._train_one_epoch(epoch)
        audit = json.loads((self.output_dir / "diagnostics" / f"epoch-{epoch:03d}_quality_il.json").read_text())
        metrics["quality_il_extra_target_calls"] = float(audit["extra_target_gurobi_calls"])
        metrics["quality_il_measurement_seconds"] = float(audit["measurement_seconds"])
        metrics["quality_il_optimizer_steps"] = float(audit["optimizer_steps"])
        metrics["training_gurobi_calls_including_quality"] = (float(metrics["training_gurobi_calls"])
                                                             + audit["extra_target_gurobi_calls"])
        return metrics
