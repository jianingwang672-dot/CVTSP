from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from src.CertifiedLocalSearch import CertifiedLocalSearchConfig, run_certified_local_search
from src.CVPSolver import SolverConfig
from src.EdgeCostPredictor import (
    EdgePredictorConfig,
    EdgeTrainingConfig,
    build_edge_cost_dataset,
    evaluate_candidate_ranking,
    evaluate_edge_predictor,
    initialize_edge_predictor,
    make_matrix_ranker,
    predict_directed_time_matrix,
    save_edge_cost_dataset,
    train_edge_cost_predictor,
)
from src.PseudoExpert import load_pseudo_expert_dataset
from src.TSPUtils import set_seed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a safe ranking-only directed edge-cost predictor."
    )
    parser.add_argument("--pseudo-expert-dataset", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/edge_predictor_phase3")
    parser.add_argument("--max-anchors-per-instance", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--unfreeze-encoder", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--huber-beta", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--max-refreshes", type=int, default=10)
    parser.add_argument("--moves", default="swap,relocate,two_opt")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def _solver_config(threads: int) -> SolverConfig:
    return SolverConfig(
        solver_backend="gurobi",
        gurobi_threads=threads,
        output_flag=0,
        gurobi_params={
            "FeasibilityTol": 1e-8,
            "OptimalityTol": 1e-8,
            "NumericFocus": 2,
        },
        gurobi_explicit_qcp=True,
    )


def _compare_local_search(
    predictor,
    records,
    *,
    device: torch.device,
    solver_config: SolverConfig,
    search_config: CertifiedLocalSearchConfig,
    top_k: int,
) -> dict:
    details = []
    for record in records:
        instance = record.to_instance()
        initial_sequence = list(record.teacher_best_sequence)

        full_started = time.perf_counter()
        full = run_certified_local_search(
            instance,
            initial_sequence,
            solver_config=solver_config,
            search_config=search_config,
        )
        full_total_runtime = time.perf_counter() - full_started

        predicted_started = time.perf_counter()
        predicted_matrix = predict_directed_time_matrix(predictor, instance, device=device)
        prediction_runtime = time.perf_counter() - predicted_started
        guided_started = time.perf_counter()
        guided = run_certified_local_search(
            instance,
            initial_sequence,
            solver_config=solver_config,
            search_config=search_config,
            candidate_ranker=make_matrix_ranker(predicted_matrix),
            max_exact_candidates=top_k,
        )
        guided_search_runtime = time.perf_counter() - guided_started

        if guided.final_objective > guided.initial_objective + 1e-9:
            raise RuntimeError("predictor-guided certified search increased the true objective")
        for accepted in guided.history:
            if accepted.new_objective > accepted.predicted_upper_bound + search_config.upper_bound_tolerance:
                raise RuntimeError("predictor-guided move violated its exact upper bound")

        details.append(
            {
                "instance_id": instance.instance_id,
                "initial_sequence": initial_sequence,
                "initial_objective": full.initial_objective,
                "full": full.to_dict(include_solution=False),
                "guided": guided.to_dict(include_solution=False),
                "prediction_runtime": prediction_runtime,
                "full_total_runtime": full_total_runtime,
                "guided_search_runtime": guided_search_runtime,
                "guided_total_runtime": prediction_runtime + guided_search_runtime,
                "guided_minus_full_final_objective": guided.final_objective - full.final_objective,
            }
        )

    full_evaluations = sum(item["full"]["evaluated_moves"] for item in details)
    guided_evaluations = sum(item["guided"]["evaluated_moves"] for item in details)
    full_generated = sum(item["full"]["generated_moves"] for item in details)
    guided_generated = sum(item["guided"]["generated_moves"] for item in details)
    full_final = [item["full"]["final_objective"] for item in details]
    guided_final = [item["guided"]["final_objective"] for item in details]
    return {
        "instances": len(details),
        "top_k": top_k,
        "full_generated_moves": full_generated,
        "full_exact_ub_evaluations": full_evaluations,
        "guided_generated_moves": guided_generated,
        "guided_exact_ub_evaluations": guided_evaluations,
        "exact_ub_evaluation_reduction_percent": (
            100.0 * (full_evaluations - guided_evaluations) / max(full_evaluations, 1)
        ),
        "full_accepted_moves": sum(item["full"]["accepted_moves"] for item in details),
        "guided_accepted_moves": sum(item["guided"]["accepted_moves"] for item in details),
        "full_gurobi_calls": sum(item["full"]["gurobi_calls"] for item in details),
        "guided_gurobi_calls": sum(item["guided"]["gurobi_calls"] for item in details),
        "full_mean_final_objective": float(np.mean(full_final)) if details else 0.0,
        "guided_mean_final_objective": float(np.mean(guided_final)) if details else 0.0,
        "guided_minus_full_mean_objective": (
            float(np.mean(guided_final) - np.mean(full_final)) if details else 0.0
        ),
        "same_final_objective_instances": sum(
            abs(guided - full) <= 1e-6 for guided, full in zip(guided_final, full_final)
        ),
        "full_runtime": sum(item["full_total_runtime"] for item in details),
        "guided_runtime_including_prediction": sum(item["guided_total_runtime"] for item in details),
        "details": details,
    }


def main() -> None:
    args = _parser().parse_args()
    if min(
        args.max_anchors_per_instance,
        args.hidden_dim,
        args.epochs,
        args.batch_size,
        args.top_k,
    ) <= 0:
        raise ValueError("anchor count, dimensions, epochs, batch size, and top-k must be positive")
    started = time.perf_counter()
    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, pseudo_payload = load_pseudo_expert_dataset(args.pseudo_expert_dataset)
    solver_config = _solver_config(args.threads)

    samples, dataset_summary = build_edge_cost_dataset(
        records,
        solver_config=solver_config,
        max_anchors_per_instance=args.max_anchors_per_instance,
    )
    dataset_path = output_dir / "edge_cost_dataset.json"
    save_edge_cost_dataset(
        dataset_path,
        samples,
        metadata={
            "pseudo_expert_dataset": str(Path(args.pseudo_expert_dataset).resolve()),
            "policy_checkpoint": str(Path(args.policy_checkpoint).resolve()),
            "pseudo_expert_schema_version": pseudo_payload["schema_version"],
            "seed": args.seed,
        },
        summary=dataset_summary,
    )
    train_samples = [sample for sample in samples if sample.split == "train"]
    validation_samples = [sample for sample in samples if sample.split == "validation"]
    promotion_samples = [sample for sample in samples if sample.split == "promotion"]

    predictor_config = EdgePredictorConfig(
        hidden_dim=args.hidden_dim,
        freeze_encoder=not args.unfreeze_encoder,
    )
    training_config = EdgeTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        huber_beta=args.huber_beta,
        seed=args.seed,
    )
    predictor = initialize_edge_predictor(
        args.policy_checkpoint,
        device=device,
        config=predictor_config,
    )
    training = train_edge_cost_predictor(
        predictor,
        train_samples,
        validation_samples,
        device=device,
        output_dir=output_dir / "predictor",
        policy_checkpoint=args.policy_checkpoint,
        config=training_config,
    )
    edge_evaluation = {
        "train": evaluate_edge_predictor(predictor, train_samples, device=device),
        "validation": evaluate_edge_predictor(predictor, validation_samples, device=device),
        "promotion": evaluate_edge_predictor(predictor, promotion_samples, device=device),
    }
    ranking_evaluation = {
        "validation": evaluate_candidate_ranking(
            predictor,
            validation_samples,
            device=device,
            top_k=args.top_k,
        ),
        "promotion": evaluate_candidate_ranking(
            predictor,
            promotion_samples,
            device=device,
            top_k=args.top_k,
        ),
    }
    comparison_records = [record for record in records if record.split == "promotion"]
    if not comparison_records:
        comparison_records = [record for record in records if record.split == "validation"]
    if not comparison_records:
        comparison_records = [record for record in records if record.split == "train"][:2]
    search_config = CertifiedLocalSearchConfig(
        max_refreshes=args.max_refreshes,
        move_types=tuple(item.strip() for item in args.moves.split(",") if item.strip()),
    )
    local_search_comparison = _compare_local_search(
        predictor,
        comparison_records,
        device=device,
        solver_config=solver_config,
        search_config=search_config,
        top_k=args.top_k,
    )

    summary = {
        "phase": "ranking_only_directed_edge_cost_predictor",
        "safety_contract": (
            "predictor ranks/prunes only; every accepted move uses the exact fixed-solution "
            "upper bound and a Gurobi refresh"
        ),
        "config": vars(args),
        "predictor_config": asdict(predictor_config),
        "training_config": asdict(training_config),
        "dataset": dataset_summary,
        "training": training,
        "edge_evaluation": edge_evaluation,
        "candidate_ranking": ranking_evaluation,
        "local_search_comparison": local_search_comparison,
        "runtime": time.perf_counter() - started,
        "dataset_path": str(dataset_path),
        "predictor_checkpoint": str(output_dir / "predictor" / "best.pt"),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset": dataset_summary,
                "promotion_edge_metrics": edge_evaluation["promotion"],
                "promotion_ranking": {
                    key: value
                    for key, value in ranking_evaluation["promotion"].items()
                    if key != "details"
                },
                "local_search_comparison": {
                    key: value
                    for key, value in local_search_comparison.items()
                    if key != "details"
                },
                "runtime": summary["runtime"],
                "summary_path": str(summary_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
