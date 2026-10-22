"""Create the immutable, independent POMO checkpoint-selection instances."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from src.FixedValidation import load_or_create_fixed_validation_sets
from src.PseudoExpert import instance_from_dict, instance_to_dict


ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = ROOT / "validation/pomo_selection_20260921"
EXISTING_FIXED = ROOT / "validation/macsim_fixed"
SYNTHETIC_RESULTS = (
    ROOT / "outputs/eval_vectorized_epoch100_vs_gurobi_epoch200_synthetic5000/full/results.json"
)
TRAIN_CHECKPOINT = (
    ROOT
    / "outputs/train__scratch_e500_lr1e4_min1e7_w128_qdhybrid_replay30_vectorized/checkpoint-300.pt"
)
SIZES = (20, 40, 60, 80, 100)
PER_SIZE = 200
SEED = 20260921


def fingerprint(instance) -> str:
    payload = instance_to_dict(instance)
    payload.pop("instance_id", None)
    payload.pop("path", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if HOLDOUT.exists():
        raise FileExistsError(f"refusing to replace existing holdout: {HOLDOUT}")
    if not SYNTHETIC_RESULTS.is_file() or not TRAIN_CHECKPOINT.is_file():
        raise FileNotFoundError("comparison data and checkpoint-300.pt are required")

    forbidden = set()
    for size in SIZES:
        payload = torch.load(EXISTING_FIXED / f"J{size}.pt", map_location="cpu", weights_only=False)
        forbidden.update(fingerprint(instance_from_dict(row)) for row in payload["instances"])
    fixed_count = len(forbidden)

    synthetic = json.loads(SYNTHETIC_RESULTS.read_text())
    if not synthetic["complete"] or len(synthetic["records"]) != 5000:
        raise ValueError("the prior synthetic 5000-instance test is incomplete")
    forbidden.update(row["instance_fingerprint"] for row in synthetic["records"])
    synthetic_count = len(forbidden) - fixed_count

    checkpoint = torch.load(TRAIN_CHECKPOINT, map_location="cpu", weights_only=False)
    replay = checkpoint.get("expert_replay_pool", [])
    for row in replay:
        instance = row["instance"]
        if isinstance(instance, dict):
            instance = instance_from_dict(instance)
        forbidden.add(fingerprint(instance))

    groups = load_or_create_fixed_validation_sets(
        dataset_dir=ROOT / "instance/Data",
        validation_dir=HOLDOUT,
        problem_sizes=SIZES,
        per_size=PER_SIZE,
        seed=SEED,
    )
    ordered = [fingerprint(instance) for size in SIZES for instance in groups[size]]
    if len(ordered) != len(set(ordered)) or set(ordered) & forbidden:
        raise RuntimeError("new holdout overlaps existing validation, synthetic test, or replay")
    files = {}
    for size in SIZES:
        path = HOLDOUT / f"J{size}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload["base_seed"]) != SEED or int(payload["group_seed"]) != SEED + 1009 * size:
            raise RuntimeError(f"unexpected seed in {path}")
        files[path.name] = sha256(path)
    manifest = {
        "purpose": "checkpoint_selection_only",
        "generated_before_any_holdout_model_scores": True,
        "base_seed": SEED,
        "sizes": list(SIZES),
        "per_size": PER_SIZE,
        "count": len(ordered),
        "protocol": {
            "decode_type": "greedy",
            "pomo_size": "J",
            "augmentation": False,
            "start_node_strategy": "nearest_depot",
            "route_score": "minimum_exact_gurobi_makespan_over_all_pomo_routes",
        },
        "content_fingerprints_sha256": hashlib.sha256("\n".join(ordered).encode()).hexdigest(),
        "content_fingerprints": ordered,
        "files_sha256": files,
        "disjoint_from": {
            "existing_fixed_validation": fixed_count,
            "synthetic5000": synthetic_count,
            "checkpoint_300_expert_replay": len(replay),
        },
        "training_independence": "separate seed; replay snapshot checked for exact content duplicates",
        "source_checkpoint_sha256": sha256(TRAIN_CHECKPOINT),
    }
    (HOLDOUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Created {len(ordered)} independent fixed instances at {HOLDOUT}")
    print(f"Dataset fingerprint: {manifest['content_fingerprints_sha256']}")
    print(f"Checked against {fixed_count} old fixed, {synthetic_count} synthetic, and {len(replay)} replay instances")


if __name__ == "__main__":
    main()
