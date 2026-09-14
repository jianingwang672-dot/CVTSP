from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.CertifiedLocalSearch import CertifiedLocalSearchConfig, run_certified_local_search
from src.CVPSolver import SolverConfig
from src.TSProblemDef import CVTSPInstance, load_instance


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Guaranteed-improvement local search for one fixed-tour CVTSP instance."
    )
    parser.add_argument("instance_path")
    parser.add_argument(
        "--initial-tour",
        choices=["random", "identity", "nearest", "model"],
        default="random",
    )
    parser.add_argument("--sequence", help="comma-separated 0-based target permutation")
    parser.add_argument("--checkpoint-path", help="required when --initial-tour model")
    parser.add_argument("--model-device", default="cpu")
    parser.add_argument("--model-pomo-size", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-refreshes", type=int, default=10)
    parser.add_argument("--absolute-epsilon", type=float, default=1e-4)
    parser.add_argument("--relative-epsilon", type=float, default=1e-8)
    parser.add_argument("--upper-bound-tolerance", type=float, default=2e-5)
    parser.add_argument("--feasibility-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--moves",
        default="swap,relocate,two_opt",
        help="comma-separated subset of swap,relocate,two_opt",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/certified_local_search/result.json",
    )
    return parser


def _initial_sequence(instance: CVTSPInstance, args: argparse.Namespace) -> list[int]:
    if args.sequence:
        sequence = [int(value.strip()) for value in args.sequence.split(",") if value.strip()]
        if len(sequence) != instance.J or set(sequence) != set(range(instance.J)):
            raise ValueError(f"--sequence must be a permutation of 0..{instance.J - 1}")
        return sequence
    if args.initial_tour == "identity":
        return list(range(instance.J))
    if args.initial_tour == "nearest":
        remaining = set(range(instance.J))
        sequence: list[int] = []
        current = np.asarray(instance.depot, dtype=float)
        while remaining:
            next_node = min(
                remaining,
                key=lambda node: (float(np.linalg.norm(instance.targets[node] - current)), node),
            )
            sequence.append(next_node)
            remaining.remove(next_node)
            current = np.asarray(instance.targets[next_node], dtype=float)
        return sequence
    if args.initial_tour == "model":
        return _model_sequence(instance, args)
    return [int(node) for node in np.random.default_rng(args.seed).permutation(instance.J)]


def _model_sequence(instance: CVTSPInstance, args: argparse.Namespace) -> list[int]:
    if not args.checkpoint_path:
        raise ValueError("--checkpoint-path is required when --initial-tour model")
    import torch

    from src.TSPEnv import rollout_batch
    from src.TSPModel import ModelConfig, TSPModel
    from src.TSPUtils import load_checkpoint

    device = torch.device(args.model_device)
    model_config = ModelConfig(
        max_pomo_size=args.model_pomo_size or instance.J,
        pomo_divisor=None,
        start_node_strategy="nearest_depot",
    )
    model = TSPModel(model_config).to(device)
    load_checkpoint(args.checkpoint_path, model, map_location=device)
    model.eval()
    with torch.no_grad():
        rollout = rollout_batch(model, [instance], device=device, decode_type="greedy", use_pomo_start=True)
    # The first POMO rollout starts at the closest target under nearest_depot.
    return rollout.to_sequence_lists()[0][0]


def main() -> None:
    args = _build_parser().parse_args()
    instance = load_instance(args.instance_path)
    initial_sequence = _initial_sequence(instance, args)
    move_types = tuple(value.strip() for value in args.moves.split(",") if value.strip())
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
    search_config = CertifiedLocalSearchConfig(
        max_refreshes=args.max_refreshes,
        absolute_epsilon=args.absolute_epsilon,
        relative_epsilon=args.relative_epsilon,
        upper_bound_tolerance=args.upper_bound_tolerance,
        feasibility_tolerance=args.feasibility_tolerance,
        move_types=move_types,
        require_optimal=True,
    )
    result = run_certified_local_search(
        instance,
        initial_sequence,
        solver_config=solver_config,
        search_config=search_config,
    )
    payload = result.to_dict(include_solution=True)
    payload["config"] = {
        "instance_path": args.instance_path,
        "initial_tour": args.initial_tour,
        "checkpoint_path": args.checkpoint_path,
        "seed": args.seed,
        "threads": args.threads,
        "search": {
            "max_refreshes": search_config.max_refreshes,
            "absolute_epsilon": search_config.absolute_epsilon,
            "relative_epsilon": search_config.relative_epsilon,
            "upper_bound_tolerance": search_config.upper_bound_tolerance,
            "feasibility_tolerance": search_config.feasibility_tolerance,
            "move_types": list(search_config.move_types),
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = {key: payload[key] for key in (
        "instance_id",
        "initial_objective",
        "final_objective",
        "objective_improvement",
        "relative_improvement_percent",
        "generated_moves",
        "evaluated_moves",
        "pruned_moves",
        "accepted_moves",
        "gurobi_calls",
        "runtime",
        "termination_reason",
    )}
    print(json.dumps(summary, indent=2))
    print(f"details_path={output_path}")


if __name__ == "__main__":
    main()
