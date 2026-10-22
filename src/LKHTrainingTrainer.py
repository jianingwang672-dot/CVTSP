"""Fresh 100-epoch LKH training with auditable search and learning traces."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict

import numpy as np

from src.PseudoExpert import instance_to_dict
from src.SearchAblationTrainer import SearchAblationTrainer
from src.SearchDiagnostics import atomic_json, fingerprint, finite_summary


class LKHTrainingTrainer(SearchAblationTrainer):
    _STRICT_TRAJECTORY_PARAMETER_KEYS = SearchAblationTrainer._STRICT_TRAJECTORY_PARAMETER_KEYS + (
        "lkh_training_protocol", "near_miss_ablation_rate",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.si_local_search_backend != "lkh":
            raise ValueError("LKHTrainingTrainer requires the LKH backend")
        if float(self.trainer_params["near_miss_ablation_rate"]) != .20:
            raise ValueError("LKH training requires fixed near-miss 20%")
        if self.si_near_miss_min_rate != .20 or self.si_near_miss_rate != .20:
            raise ValueError("LKH training near-miss rate must remain fixed")
        self._initial_manifest.update({
            "group": "lkh_nm20_e100",
            "experiment": "lkh_near_miss_training",
            "local_search_backend": "lkh",
            "near_miss_rate": .20,
            "lkh_budget": f"max(0.005,{self.si_lkh_budget_coefficient}*J^2) seconds/instance",
            "lkh_binary": self.si_lkh_binary,
        })
        if not self.trainer_params.get("checkpoint_path"):
            atomic_json(self.output_dir / "initialization.json", self._initial_manifest)

    def _observe_il_pass(self, *, pass_index, loss, accuracy, tokens, optimizer_steps):
        gradients = [row for row in self._gradient_rows if row["phase"] == "il"]
        previous = sum(row["optimizer_steps"] for row in self._il_pass_rows)
        current = gradients[previous:previous + optimizer_steps]
        self._il_pass_rows.append({
            "pass": int(pass_index), "loss": float(loss), "accuracy": float(accuracy),
            "tokens": int(tokens), "optimizer_steps": int(optimizer_steps),
            "gradient_before": finite_summary([row["before"] for row in current]),
            "gradient_after": finite_summary([row["after"] for row in current]),
        })

    def _run_sparse_self_improvement(self, instances, reference_sequences, anchor_sequences):
        results, wall, eligible = super()._run_sparse_self_improvement(
            instances, reference_sequences, anchor_sequences
        )
        if not self._record_search:
            return results, wall, eligible
        path = self.output_dir / "diagnostics" / f"epoch-{self._diag_epoch:03d}_lkh_instances.jsonl"
        detail_path = self.output_dir / "diagnostics" / f"epoch-{self._diag_epoch:03d}_lkh_replay_samples.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            for instance, result in zip(instances, results):
                termination = Counter(x.termination_reason for x in result.local_search_results)
                top_hash = hashlib.sha256(
                    json.dumps(result.top_k_start_sequences, separators=(",", ":")).encode()
                ).hexdigest()
                row = {
                    "fingerprint": fingerprint(instance), "J": int(instance.J),
                    "anchor_exact": float(result.anchor_exact_makespan),
                    "expert_exact": float(result.expert_exact_makespan),
                    "expert_upper_bound": float(result.expert_upper_bound),
                    "exact_improvement": float(result.exact_improvement),
                    "expert_source": result.expert_source,
                    "expert_changed": bool(result.expert_changed),
                    "near_miss_candidate_upper_bound": (
                        float(result.near_miss_candidate_upper_bound)
                        if np.isfinite(result.near_miss_candidate_upper_bound) else None
                    ),
                    "near_miss_relative_gap": (
                        float(result.near_miss_relative_gap)
                        if np.isfinite(result.near_miss_relative_gap) else None
                    ),
                    "near_miss_attempted": bool(result.near_miss_attempted),
                    "near_miss_accepted": bool(result.near_miss_accepted),
                    "top_k_count": len(result.top_k_start_sequences),
                    "quality_starts": int(result.quality_start_count),
                    "diversity_starts": int(result.diversity_start_count),
                    "top_k_sha256": top_hash,
                    "budget_seconds": float(result.local_search_budget_seconds),
                    "search_seconds": float(result.local_search_runtime),
                    "export_seconds": float(result.lkh_problem_export_seconds),
                    "native_calls": int(result.lkh_native_calls),
                    "completed_starts": int(result.lkh_completed_starts),
                    "timed_out_starts": int(result.lkh_timed_out_starts),
                    "accepted_outputs": int(result.lkh_accepted_outputs),
                    "distinct_final_routes": int(result.lkh_distinct_final_routes),
                    "returned_anchor_starts": int(result.lkh_returned_anchor_starts),
                    "quantization_scale": int(result.lkh_quantization_scale),
                    "rounding_error_bound": float(result.lkh_rounding_error_bound),
                    "termination": dict(termination),
                }
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                self._lkh_rows.append(row)
                # Two fully replayable examples per observed J and epoch.
                if self._lkh_detail_counts[int(instance.J)] < 2:
                    self._lkh_detail_counts[int(instance.J)] += 1
                    detail = {
                        **row,
                        "instance": instance_to_dict(instance),
                        "anchor_sequence": list(result.anchor_sequence),
                        "expert_sequence": list(result.expert_sequence),
                        "near_miss_candidate_sequence": result.near_miss_candidate_sequence,
                        "anchor_upper_bound_matrix": np.asarray(
                            result.anchor_upper_bound_matrix, dtype=float
                        ).tolist(),
                        "top_k_start_sequences": result.top_k_start_sequences,
                        "searches": [search.to_dict() for search in result.local_search_results],
                    }
                    with detail_path.open("a", encoding="utf-8") as detail_handle:
                        detail_handle.write(json.dumps(detail, allow_nan=False) + "\n")
        return results, wall, eligible

    def _train_one_batch(self, batch_size):
        result = super()._train_one_batch(batch_size)
        attempted = int(result["near_miss_attempt_count"])
        eligible = int(result["near_miss_eligible_count"])
        accepted = int(result["near_miss_accept_count"])
        limit = math.ceil(.20 * batch_size)
        if self._active_near_miss_rate != .20 or attempted != min(limit, eligible):
            raise RuntimeError("LKH training near-miss quota mismatch")
        with (self.output_dir / "diagnostics" / f"epoch-{self._diag_epoch:03d}_quota_batches.jsonl").open("a") as handle:
            handle.write(json.dumps({
                "batch": self._joint_batch_index, "instances": batch_size,
                "nominal_rate": .20, "quota_limit": limit, "eligible": eligible,
                "attempted": attempted, "accepted": accepted,
            }) + "\n")
        comparison_path = self.output_dir / "diagnostics" / f"epoch-{self._diag_epoch:03d}_target_expert.jsonl"
        with comparison_path.open("a", encoding="utf-8") as handle:
            for record, training in zip(self._last_batch_exact_examples, self._last_batch_training_records):
                instance, expert_sequence, _, _ = training
                handle.write(json.dumps({
                    "batch": self._joint_batch_index,
                    "fingerprint": fingerprint(instance),
                    "J": int(instance.J),
                    "target_sequence": record["target_reward_sequence"],
                    "target_route_index": record["target_reward_route_index"],
                    "target_exact": record["target_reward_exact_makespan"],
                    "expert_sequence": list(expert_sequence),
                    "expert_exact": record["expert_exact_makespan"],
                    "expert_source": record["expert_source"],
                    "expert_changed": record["expert_changed"],
                }, allow_nan=False) + "\n")
        return result

    def _train_one_epoch(self, epoch):
        self._lkh_rows = []
        self._lkh_detail_counts = defaultdict(int)
        self._il_pass_rows = []
        metrics = super()._train_one_epoch(epoch)
        summary_path = self.output_dir / "diagnostics" / f"epoch-{epoch:03d}_summary.json"
        payload = json.loads(summary_path.read_text())
        lkh = {
            "instances": len(self._lkh_rows),
            "native_calls": sum(row["native_calls"] for row in self._lkh_rows),
            "completed_starts": sum(row["completed_starts"] for row in self._lkh_rows),
            "timed_out_starts": sum(row["timed_out_starts"] for row in self._lkh_rows),
            "accepted_outputs": sum(row["accepted_outputs"] for row in self._lkh_rows),
            "changed_experts": sum(row["expert_changed"] for row in self._lkh_rows),
            "certified_experts": sum(row["expert_source"] == "certified_upper_bound" for row in self._lkh_rows),
            "near_miss_experts": sum(row["expert_source"] == "near_miss_exact" for row in self._lkh_rows),
            "completed_starts_per_instance": finite_summary([row["completed_starts"] for row in self._lkh_rows]),
            "distinct_final_routes": finite_summary([row["distinct_final_routes"] for row in self._lkh_rows]),
            "returned_anchor_fraction": float(sum(row["returned_anchor_starts"] for row in self._lkh_rows)
                                              / max(sum(row["top_k_count"] for row in self._lkh_rows), 1)),
            "search_seconds": finite_summary([row["search_seconds"] for row in self._lkh_rows]),
            "exact_improvement": finite_summary([row["exact_improvement"] for row in self._lkh_rows]),
            "by_J": {
                str(size): {
                    "count": sum(row["J"] == size for row in self._lkh_rows),
                    "changed_experts": sum(row["J"] == size and row["expert_changed"] for row in self._lkh_rows),
                    "completed_starts": finite_summary([row["completed_starts"] for row in self._lkh_rows if row["J"] == size]),
                    "distinct_final_routes": finite_summary([row["distinct_final_routes"] for row in self._lkh_rows if row["J"] == size]),
                }
                for size in sorted({row["J"] for row in self._lkh_rows})
            },
        }
        payload["lkh"] = lkh
        payload["il_passes"] = self._il_pass_rows
        atomic_json(summary_path, payload)
        atomic_json(self.output_dir / "diagnostics" / f"epoch-{epoch:03d}_lkh_summary.json", lkh)
        metrics.update({
            "lkh_completed_starts_per_instance": lkh["completed_starts_per_instance"].get("mean", 0.0),
            "lkh_distinct_final_routes": lkh["distinct_final_routes"].get("mean", 0.0),
            "lkh_returned_anchor_fraction": lkh["returned_anchor_fraction"],
            "lkh_accepted_outputs": float(lkh["accepted_outputs"]),
        })
        return metrics
