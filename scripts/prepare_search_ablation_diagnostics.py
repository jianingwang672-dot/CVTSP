"""Freeze the shared diagnostic100 split independently of formal test data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.prepare_e500_logic_datasets import archived_fingerprints
from src.FixedValidation import load_or_create_fixed_validation_sets
from src.PseudoExpert import instance_from_dict
from src.SearchDiagnostics import atomic_json, fingerprint

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "validation/search_ablation_20261012/diagnostic100"
SIZES = (20, 40, 60, 80, 100)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest_path = DIRECTORY / "manifest.json"
    original = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if args.verify_only and original is None:
        raise RuntimeError("diagnostic split has not been created")
    if original:
        import hashlib
        for name, sha in original["files_sha256"].items():
            if hashlib.sha256((DIRECTORY / name).read_bytes()).hexdigest() != sha:
                raise RuntimeError(f"modified diagnostic data: {name}")
    groups = load_or_create_fixed_validation_sets(
        dataset_dir=ROOT / "instance/Data", validation_dir=DIRECTORY,
        problem_sizes=SIZES, per_size=20, seed=20261012,
    )
    forbidden, sources = archived_fingerprints()
    for directory in (ROOT / "validation/e500_logic_e100_20261010/validation2000",
                      ROOT / "validation/e500_logic_e100_20261010/test500"):
        count = 0
        for size in SIZES:
            data = torch.load(directory / f"J{size}.pt", map_location="cpu", weights_only=False)
            forbidden.update(fingerprint(instance_from_dict(row)) for row in data["instances"])
            count += len(data["instances"])
        sources.append({"path": str(directory.relative_to(ROOT)), "count": count})
    values = [fingerprint(instance) for size in SIZES for instance in groups[size]]
    if len(set(values)) != 100 or set(values) & forbidden:
        raise RuntimeError("duplicate diagnostic instances or overlap with archived splits")
    import hashlib
    manifest = {
        "seed": 20261012, "per_size": 20, "count": 100, "sizes": list(SIZES),
        "fingerprints": values,
        "full_pomo_fingerprints": [fingerprint(x) for size in SIZES for x in groups[size][:10]],
        "files_sha256": {f"J{size}.pt": hashlib.sha256((DIRECTORY / f"J{size}.pt").read_bytes()).hexdigest() for size in SIZES},
        "checked_disjoint_from": sources,
        "purpose": "training diagnosis and development; never counted as formal test results",
    }
    if original is not None and original != manifest:
        raise RuntimeError("diagnostic manifest changed")
    if original is None:
        atomic_json(manifest_path, manifest)
    print("diagnostic100: checksums/duplicates OK; fixed full-POMO subset=50", flush=True)


if __name__ == "__main__":
    main()
