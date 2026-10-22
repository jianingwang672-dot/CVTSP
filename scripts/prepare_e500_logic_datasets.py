"""Prepare/verify immutable validation2000 and synthetic test500 splits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.FixedValidation import load_or_create_fixed_validation_sets
from src.PseudoExpert import instance_from_dict, instance_to_dict

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "validation/e500_logic_e100_20261010"
SIZES = (20, 40, 60, 80, 100)
SPLITS = (("validation2000", 400, 20261010), ("test500", 100, 20262010))


def fingerprint(instance):
    payload = instance_to_dict(instance)
    payload.pop("instance_id", None)
    payload.pop("path", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_for(directory, groups, per_size, seed, purpose):
    ordered = [fingerprint(instance) for size in SIZES for instance in groups[size]]
    return {
        "purpose": purpose,
        "base_seed": seed, "sizes": list(SIZES), "per_size": per_size,
        "count": len(ordered),
        "content_fingerprints": ordered,
        "content_fingerprints_sha256": hashlib.sha256("\n".join(ordered).encode()).hexdigest(),
        "files_sha256": {f"J{size}.pt": file_sha256(directory / f"J{size}.pt") for size in SIZES},
        "training_independence": "Separate RNG seeds from training; exact duplicates checked against listed archived datasets. Historical full training data was not retained.",
    }


def archived_fingerprints():
    forbidden = set()
    sources = []
    for directory in [ROOT / "validation/macsim_fixed", ROOT / "validation/pomo_selection_20260921"]:
        count = 0
        for size in SIZES:
            path = directory / f"J{size}.pt"
            if path.is_file():
                payload = torch.load(path, map_location="cpu", weights_only=False)
                for row in payload["instances"]:
                    forbidden.add(fingerprint(instance_from_dict(row)))
                    count += 1
        sources.append({"path": str(directory.relative_to(ROOT)), "count": count})
    path = ROOT / "outputs/eval_vectorized_epoch100_vs_gurobi_epoch200_synthetic5000/full/results.json"
    if path.is_file():
        records = json.loads(path.read_text())["records"]
        forbidden.update(row["instance_fingerprint"] for row in records)
        sources.append({"path": str(path.relative_to(ROOT)), "count": len(records)})
    return forbidden, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    forbidden, sources = archived_fingerprints()
    seen = set()
    for name, per_size, seed in SPLITS:
        directory = DATA_ROOT / name
        manifest_path = directory / "manifest.json"
        if args.verify_only or directory.exists():
            # Existing splits are immutable; missing/partial data is an error.
            original = json.loads(manifest_path.read_text())
            for size in SIZES:
                path = directory / f"J{size}.pt"
                if file_sha256(path) != original["files_sha256"][path.name]:
                    raise ValueError(f"Dataset file modified: {path}")
        else:
            original = None
        groups = load_or_create_fixed_validation_sets(
            dataset_dir=ROOT / "instance/Data", validation_dir=directory,
            problem_sizes=SIZES, per_size=per_size, seed=seed,
        )
        current = manifest_for(directory, groups, per_size, seed, name)
        for size in SIZES:
            payload = torch.load(directory / f"J{size}.pt", map_location="cpu", weights_only=False)
            if payload["base_seed"] != seed or payload["group_seed"] != seed + 1009 * size:
                raise ValueError(f"Wrong seed: {directory}/J{size}.pt")
        values = current["content_fingerprints"]
        if len(set(values)) != len(values) or set(values) & (seen | forbidden):
            raise ValueError(f"Duplicate/overlapping instances in {directory}")
        if original is not None:
            for key in ("base_seed", "sizes", "per_size", "count", "content_fingerprints", "files_sha256"):
                if original[key] != current[key]:
                    raise ValueError(f"Manifest mismatch: {name}/{key}")
        else:
            current["checked_disjoint_from"] = sources
            current["protocol"] = {
                "decode_type": "greedy", "augmentation": False,
                "pomo_size": 1 if name == "validation2000" else "J",
                "note": "Validation keeps the E500 one-candidate promotion rule; test uses exact full-POMO minimum. Test is not used for promotion or selection.",
            }
            manifest_path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        seen.update(values)
        print(f"{name}: {len(values)} instances ({per_size} per J), seed={seed}, checksum OK")
    print("Validation and test are disjoint; archived dataset duplicate checks passed.")


if __name__ == "__main__":
    main()
