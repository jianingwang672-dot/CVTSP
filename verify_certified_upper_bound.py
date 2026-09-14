from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from src.CertifiedUpperBound import (
    FixedSolutionUpperBound,
    apply_relocate,
    apply_swap,
    apply_two_opt,
    is_optimal_gurobi_solution,
)
from src.CVPSolver import SolverConfig, solve
from src.TSProblemDef import RandomCVTSPGenerator, load_instance


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Empirically verify the fixed-Gurobi-solution tour upper bound."
    )
    parser.add_argument("--instances", type=int, default=5)
    parser.add_argument("--candidates-per-instance", type=int, default=20)
    parser.add_argument("--problem-size", type=int, default=10)
    parser.add_argument("--dataset-dir", default="instance/Data")
    parser.add_argument(
        "--real-instance-path",
        action="append",
        default=[],
        help="also test this real instance; may be supplied more than once",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=2e-5)
    parser.add_argument("--feasibility-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--output-json",
        default="outputs/certified_upper_bound/verification.json",
    )
    return parser


def _random_candidate(
    tour: tuple[int, ...],
    kind: str,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    n = len(tour)
    if kind == "random":
        candidate = tuple(int(node) for node in rng.permutation(n))
        return candidate if candidate != tour else tuple(reversed(candidate))
    first, second = sorted(int(value) for value in rng.choice(n, size=2, replace=False))
    if kind == "swap":
        return apply_swap(tour, first, second)
    if kind == "two_opt":
        return apply_two_opt(tour, first, second)
    if kind == "relocate":
        source, destination = int(first), int(second)
        if bool(rng.integers(0, 2)):
            source, destination = destination, source
        return apply_relocate(tour, source, destination)
    raise ValueError(kind)


def main() -> None:
    args = _build_parser().parse_args()
    if args.instances <= 0 or args.candidates_per_instance <= 0:
        raise ValueError("instances and candidates-per-instance must be positive")
    if args.problem_size < 3:
        raise ValueError("problem-size must be at least 3")
    if args.tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    generator = RandomCVTSPGenerator(
        dataset_dir=args.dataset_dir,
        min_problem_size=args.problem_size,
        max_problem_size=args.problem_size,
        seed=args.seed,
    )
    solver_config = SolverConfig(
        solver_backend="gurobi",
        gurobi_threads=args.threads,
        output_flag=0,
        gurobi_params={
            "FeasibilityTol": 1e-8,
            "OptimalityTol": 1e-8,
            "NumericFocus": 2,
        },
        gurobi_explicit_qcp=True,
    )

    gaps: list[float] = []
    excesses: list[float] = []
    records: list[dict] = []
    kind_counts: Counter[str] = Counter()
    max_anchor_feasibility_violation = 0.0
    max_anchor_reconstruction_error = 0.0
    gurobi_calls = 0
    kinds = ("random", "swap", "relocate", "two_opt")

    test_instances = [
        ("generated", generator.sample_instance(problem_size=args.problem_size))
        for _ in range(args.instances)
    ]
    test_instances.extend(("real", load_instance(path)) for path in args.real_instance_path)

    for instance_index, (instance_source, instance) in enumerate(test_instances):
        anchor_tour = tuple(int(node) for node in rng.permutation(instance.J))
        anchor_solution = solve(instance, list(anchor_tour), solver_config)
        gurobi_calls += 1
        if not is_optimal_gurobi_solution(anchor_solution):
            raise RuntimeError(
                f"anchor Gurobi solve was not optimal: {anchor_solution.status} "
                f"{anchor_solution.raw_debug}"
            )
        majorizer = FixedSolutionUpperBound.from_solution(
            instance,
            anchor_solution,
            feasibility_tolerance=args.feasibility_tolerance,
            require_optimal=True,
        )
        max_anchor_feasibility_violation = max(
            max_anchor_feasibility_violation,
            majorizer.feasibility.max_violation,
        )
        max_anchor_reconstruction_error = max(
            max_anchor_reconstruction_error,
            abs(majorizer.anchor_reused_objective - anchor_solution.objective),
        )

        seen: set[tuple[int, ...]] = {anchor_tour}
        candidates: list[tuple[str, tuple[int, ...]]] = []
        attempts = 0
        while len(candidates) < args.candidates_per_instance:
            kind = kinds[len(candidates) % len(kinds)]
            candidate = _random_candidate(anchor_tour, kind, rng)
            attempts += 1
            if candidate in seen:
                if attempts > args.candidates_per_instance * 100:
                    raise RuntimeError("could not generate enough unique candidate tours")
                continue
            seen.add(candidate)
            candidates.append((kind, candidate))

        for kind, candidate in candidates:
            candidate_upper_bound = majorizer.upper_bound(candidate)
            candidate_solution = solve(instance, list(candidate), solver_config)
            gurobi_calls += 1
            if not is_optimal_gurobi_solution(candidate_solution):
                raise RuntimeError(
                    f"candidate Gurobi solve was not optimal: {candidate_solution.status} "
                    f"{candidate_solution.raw_debug}"
                )
            gap = float(candidate_upper_bound - candidate_solution.objective)
            excess = float(candidate_solution.objective - candidate_upper_bound)
            gaps.append(gap)
            excesses.append(excess)
            kind_counts[kind] += 1
            records.append(
                {
                    "instance_index": instance_index,
                    "instance_id": instance.instance_id,
                    "instance_source": instance_source,
                    "kind": kind,
                    "anchor_tour": list(anchor_tour),
                    "candidate_tour": list(candidate),
                    "anchor_objective": anchor_solution.objective,
                    "candidate_true_objective": candidate_solution.objective,
                    "candidate_upper_bound": candidate_upper_bound,
                    "upper_bound_gap": gap,
                    "violation_excess": max(0.0, excess),
                }
            )

    violation_count = sum(excess > args.tolerance for excess in excesses)
    summary = {
        "passed": violation_count == 0,
        "random_instances": args.instances,
        "real_instances": len(args.real_instance_path),
        "anchor_instances": len(test_instances),
        "generated_problem_size": args.problem_size,
        "tested_candidates": len(gaps),
        "candidate_kind_counts": dict(kind_counts),
        "violations": violation_count,
        "tolerance": args.tolerance,
        "max_violation": max(0.0, max(excesses, default=0.0)),
        "mean_gap": float(np.mean(gaps)) if gaps else 0.0,
        "max_gap": max(gaps, default=0.0),
        "min_gap": min(gaps, default=0.0),
        "max_anchor_feasibility_violation": max_anchor_feasibility_violation,
        "max_anchor_reconstruction_error": max_anchor_reconstruction_error,
        "gurobi_calls": gurobi_calls,
        "runtime": time.perf_counter() - started,
        "seed": args.seed,
    }
    payload = {
        "summary": summary,
        "formula": (
            "UB(q)=sum_i tau_i* + ||depot-s_q1*||/Vc + "
            "sum_k ||l_qk*-s_q(k+1)*||/Vc + ||l_qn*-depot||/Vc"
        ),
        "records": records,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"details_path={output_path}")
    if violation_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
