"""Fail unless a B config differs from its paired A config only in allowed keys."""

from __future__ import annotations

import json
import sys

ALLOWED = {
    "trainer_params.dense_reward_source",
    "trainer_params.random_subset_min_routes",
    "trainer_params.random_subset_max_routes",
    "trainer_params.self_improvement_parallel_workers",
    "trainer_params.gpu_gate_dir",
    "trainer_params.gpu_max_active",
    "trainer_params.result_folder",
    "trainer_params.output_dir",
    "trainer_params.epochs",
}


def flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: value}


def main() -> None:
    a = flatten(json.load(open(sys.argv[1])))
    b = flatten(json.load(open(sys.argv[2])))
    differences = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    backend = "trainer_params.self_improvement_local_search_backend"
    # LKH knobs were added after the swap+2-opt A runs; with the default
    # swap+2-opt backend they are inert, so their appearance is allowed.
    inert_lkh = (a.get(backend) is None and b.get(backend) == "swap_two_opt")
    allowed = set(ALLOWED)
    if inert_lkh:
        allowed.add(backend)
        allowed.update(k for k in differences
                       if k.startswith("trainer_params.self_improvement_lkh_") and k not in a)
    unexpected = [k for k in differences if k not in allowed]
    for key in differences:
        flag = "UNEXPECTED" if key in unexpected else "allowed"
        print(f"{flag}\t{key}\tA={a.get(key)!r}\tB={b.get(key)!r}")
    if unexpected:
        sys.exit(f"{len(unexpected)} unexpected config differences")
    print(f"OK: {len(differences)} allowed differences")


if __name__ == "__main__":
    main()
