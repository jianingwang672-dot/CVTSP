"""Fixed-quota, fresh-training comparison using the existing search diagnostics."""
import json
import math
from loguru import logger

from src.SearchAblationTrainer import SearchAblationTrainer
from src.SearchDiagnostics import atomic_json


class NearMissQuotaTrainer(SearchAblationTrainer):
    _STRICT_TRAJECTORY_PARAMETER_KEYS = SearchAblationTrainer._STRICT_TRAJECTORY_PARAMETER_KEYS + (
        "near_miss_ablation_rate",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rate = float(self.trainer_params["near_miss_ablation_rate"])
        if rate not in (.05, .10, .20):
            raise ValueError("unsupported nominal near-miss quota")
        if self.si_near_miss_min_rate != rate or self.si_near_miss_rate != rate:
            raise ValueError("near-miss quota must remain fixed through training")
        if tuple(self.si_move_types) != ("swap", "two_opt") or self._move_start != 30 or self._move_final != 30:
            raise ValueError("quota study requires swap+2-opt and fixed 30 moves")
        self._initial_manifest.update({"group": f"nm{round(rate * 100):02d}",
                                       "experiment": "near_miss_quota",
                                       "nominal_near_miss_rate": rate,
                                       "fixed_move_cap": 30,
                                       "quota_rounding": "ceil(rate * batch_size), capped by eligible instances"})
        if not self.trainer_params.get("checkpoint_path"):
            atomic_json(self.output_dir / "initialization.json", self._initial_manifest)
        logger.info("Near-miss fresh study {}: fixed nominal rate {:.0%}, move cap 30, replay initially {}",
                    self._initial_manifest["group"], rate, len(self._expert_replay_pool))

    def _train_one_batch(self, batch_size):
        result = super()._train_one_batch(batch_size)
        rate = float(self.trainer_params["near_miss_ablation_rate"])
        attempted = int(result["near_miss_attempt_count"])
        eligible = int(result["near_miss_eligible_count"])
        accepted = int(result["near_miss_accept_count"])
        limit = math.ceil(rate * batch_size)
        if self._active_near_miss_rate != rate or attempted != min(limit, eligible) or not 0 <= accepted <= attempted:
            raise RuntimeError("fixed near-miss quota did not match the executed batch")
        row = {"batch": self._joint_batch_index, "instances": batch_size, "nominal_rate": rate,
               "quota_limit": limit, "eligible": eligible, "attempted": attempted, "accepted": accepted}
        with (self.output_dir / "diagnostics" / f"epoch-{self._diag_epoch:03d}_quota_batches.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        return result
