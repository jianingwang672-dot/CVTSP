"""Pure statistics for the fresh search ablation; no training state changes."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.CertifiedUpperBound import iter_candidate_move_indices
from src.PseudoExpert import instance_to_dict


def move_budget(epoch: int, start: int = 30, final: int = 10, horizon: int = 100) -> int:
    if horizon < 2 or epoch < 1 or not 0 <= final <= start:
        raise ValueError("invalid move schedule")
    fraction = min(1.0, (epoch - 1) / (horizon - 1))
    return max(final, min(start, int(round(start - (start - final) * fraction))))


def fingerprint(instance) -> str:
    payload = instance_to_dict(instance)
    payload.pop("instance_id", None)
    payload.pop("path", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def tensor_digest(state) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@lru_cache(None)
def neighborhood_counts(size, move_types):
    return Counter(kind for kind, _, _ in iter_candidate_move_indices(range(size), move_types))


class SearchAccumulator:
    def __init__(self, budget, move_types):
        self.budget = budget
        self.move_types = tuple(move_types)
        self.groups = {}

    def add(self, instance, result):
        for index, chain in enumerate(result.local_search_results):
            core = min(20, len(result.top_k_start_sequences))
            category = ("diversity" if core <= index < core + result.diversity_start_count
                        else "quality")
            key = f"J{instance.J}_{category}"
            group = self.groups.setdefault(key, {
                "chains": 0, "histogram": Counter(), "termination": Counter(),
                "accepted": Counter(), "upper_bound_gain": Counter(), "evaluated": Counter(),
                "gain_after_move10": 0.0, "worker_seconds": 0.0,
            })
            group["chains"] += 1
            group["histogram"][chain.accepted] += 1
            group["termination"][chain.termination_reason] += 1
            group["worker_seconds"] += chain.runtime
            for item in chain.history:
                group["accepted"][item.move.kind] += 1
                group["upper_bound_gain"][item.move.kind] += item.upper_bound_improvement
                if item.iteration > 10:
                    group["gain_after_move10"] += item.upper_bound_improvement
            counts = neighborhood_counts(instance.J, self.move_types)
            total = sum(counts.values())
            if chain.evaluated != chain.generated or chain.dual_pruned:
                raise ValueError("ablation candidate counts require exhaustive unpruned search")
            if total and chain.generated % total:
                raise ValueError("search count does not match whole-neighborhood scans")
            scans = chain.generated // total if total else 0
            for kind, count in counts.items():
                group["evaluated"][kind] += scans * count

    def summary(self):
        out = {}
        for key, group in sorted(self.groups.items()):
            histogram = group["histogram"]
            steps = np.repeat(list(histogram), list(histogram.values()))
            n = max(1, group["chains"])
            out[key] = {**{k: dict(v) if isinstance(v, Counter) else v
                          for k, v in group.items()},
                        "mean_steps": float(np.mean(steps)),
                        "p50_steps": float(np.quantile(steps, .5)),
                        "p90_steps": float(np.quantile(steps, .9)),
                        "p99_steps": float(np.quantile(steps, .99)),
                        "cap_hit_rate": histogram[self.budget] / n,
                        "above10_rate": sum(v for k, v in histogram.items() if k > 10) / n,
                        "above20_rate": sum(v for k, v in histogram.items() if k > 20) / n}
        return out


def finite_summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0}
    return {"count": int(values.size), "mean": float(values.mean()),
            "min": float(values.min()), "max": float(values.max()),
            "std": float(values.std()), "positive_rate": float(np.mean(values > 0))}
