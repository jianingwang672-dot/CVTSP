from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.CVPSolver import SolverConfig
from src.EdgeCostPredictor import (
    EdgeCostSample,
    EdgePredictorConfig,
    EdgeTrainingConfig,
    build_edge_cost_dataset,
    evaluate_candidate_ranking,
    evaluate_edge_predictor,
    initialize_edge_predictor,
    load_edge_cost_dataset,
    load_edge_predictor_checkpoint,
    make_matrix_ranker,
    predict_directed_time_matrix,
    save_edge_cost_dataset,
    train_edge_cost_predictor,
)
from src.ImitationLearning import (
    ImitationTrainingConfig,
    load_policy_checkpoint,
    train_imitation_student,
)
from src.PseudoExpert import (
    PseudoExpertGenerationConfig,
    checkpoint_sha256,
    evaluate_policy_exact,
    generate_pseudo_expert_record,
    load_pseudo_expert_dataset,
    save_pseudo_expert_dataset,
    summarize_pseudo_experts,
)
from src.TSProblemDef import RandomCVTSPGenerator
from src.TSPUtils import set_seed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Predictor-guided certified pseudo-expert, cross-entropy, and exact-promotion loop."
        )
    )
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/self_improvement")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed rounds and completed pseudo-expert records in output-dir",
    )
    parser.add_argument("--dataset-dir", default="instance/Data")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--train-instances", type=int, default=8)
    parser.add_argument("--validation-instances", type=int, default=2)
    parser.add_argument("--promotion-instances", type=int, default=2)
    parser.add_argument(
        "--problem-size",
        type=int,
        help="optional fixed J; omit it to sample each instance independently from the size range",
    )
    parser.add_argument("--min-problem-size", type=int, default=20)
    parser.add_argument("--max-problem-size", type=int, default=100)
    parser.add_argument("--teacher-candidates", type=int, default=4)
    parser.add_argument("--teacher-decode", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--sample-max-rollouts", type=int, default=32)
    parser.add_argument("--local-search-refreshes", type=int, default=10)
    parser.add_argument("--moves", default="swap,relocate,two_opt")
    parser.add_argument("--imitation-epochs", type=int, default=20)
    parser.add_argument("--imitation-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--promotion-epsilon", type=float, default=1e-5)
    parser.add_argument(
        "--require-no-promotion-losses",
        action="store_true",
        help="also reject promotion if any promotion instance regresses",
    )
    parser.add_argument("--start-node-strategy", default="nearest_depot")
    parser.add_argument(
        "--disable-edge-predictor",
        action="store_true",
        help="run every round with the complete certified neighborhood and skip edge training",
    )
    parser.add_argument(
        "--initial-edge-predictor-checkpoint",
        help="optional predictor used for Top-K search in round one",
    )
    parser.add_argument("--edge-top-k", type=int, default=30)
    parser.add_argument("--edge-max-anchors-per-instance", type=int, default=4)
    parser.add_argument("--edge-hidden-dim", type=int, default=128)
    parser.add_argument("--edge-unfreeze-encoder", action="store_true")
    parser.add_argument("--edge-epochs", type=int, default=50)
    parser.add_argument("--edge-batch-size", type=int, default=8)
    parser.add_argument("--edge-learning-rate", type=float, default=1e-3)
    parser.add_argument("--edge-weight-decay", type=float, default=1e-6)
    parser.add_argument("--edge-grad-clip", type=float, default=1.0)
    parser.add_argument("--edge-huber-beta", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "rounds": args.rounds,
        "train_instances": args.train_instances,
        "teacher_candidates": args.teacher_candidates,
        "imitation_epochs": args.imitation_epochs,
        "imitation_batch_size": args.imitation_batch_size,
    }
    if args.problem_size is not None:
        positive["problem_size"] = args.problem_size
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"these arguments must be positive: {invalid}")
    if args.validation_instances < 0 or args.promotion_instances < 0:
        raise ValueError("validation/promotion instance counts must be non-negative")
    if args.min_problem_size <= 0 or args.max_problem_size < args.min_problem_size:
        raise ValueError("invalid problem-size range")
    if not args.disable_edge_predictor:
        edge_positive = {
            "edge_top_k": args.edge_top_k,
            "edge_max_anchors_per_instance": args.edge_max_anchors_per_instance,
            "edge_hidden_dim": args.edge_hidden_dim,
            "edge_epochs": args.edge_epochs,
            "edge_batch_size": args.edge_batch_size,
            "edge_learning_rate": args.edge_learning_rate,
            "edge_grad_clip": args.edge_grad_clip,
            "edge_huber_beta": args.edge_huber_beta,
        }
        invalid_edge = [name for name, value in edge_positive.items() if value <= 0]
        if invalid_edge:
            raise ValueError(f"these edge-predictor arguments must be positive: {invalid_edge}")
    if args.disable_edge_predictor and args.initial_edge_predictor_checkpoint:
        raise ValueError("an initial edge predictor cannot be used when edge prediction is disabled")


def _emit(stage: str, **payload) -> None:
    print(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                "stage": stage,
                **payload,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def _make_ranker_factory(edge_predictor, *, device: torch.device):
    def factory(instance):
        predicted_matrix = predict_directed_time_matrix(
            edge_predictor,
            instance,
            device=device,
        )
        return make_matrix_ranker(predicted_matrix)

    return factory


def _validate_resume_config(output_root: Path, args: argparse.Namespace) -> None:
    summary_path = output_root / "summary.json"
    if not args.resume or not summary_path.exists():
        return
    stored = json.loads(summary_path.read_text(encoding="utf-8")).get("config", {})
    current = vars(args)
    ignored = {"rounds", "resume"}
    mismatches = {
        key: {"stored": stored.get(key), "current": value}
        for key, value in current.items()
        if key not in ignored and key in stored and stored.get(key) != value
    }
    if mismatches:
        raise ValueError(f"resume configuration mismatch: {mismatches}")


def _restore_completed_round(
    round_dir: Path,
    *,
    expected_round: int,
    expected_teacher: str,
    edge_enabled: bool,
    device: torch.device,
):
    summary_path = round_dir / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("round", -1)) != expected_round:
        raise ValueError(f"resume round mismatch in {summary_path}")
    if str(summary.get("teacher_before")) != expected_teacher:
        raise ValueError(
            f"resume teacher mismatch in round {expected_round}: "
            f"stored={summary.get('teacher_before')}, current={expected_teacher}"
        )
    edge_summary = summary.get("edge_predictor", {})
    if bool(edge_summary.get("enabled", False)) != edge_enabled:
        raise ValueError(f"resume edge-predictor setting mismatch in round {expected_round}")

    predictor_checkpoint = edge_summary.get("checkpoint_after")
    predictor = None
    cumulative_samples: list[EdgeCostSample] = []
    if edge_enabled:
        if not predictor_checkpoint or not Path(predictor_checkpoint).exists():
            raise ValueError(f"completed round {expected_round} is missing its edge checkpoint")
        predictor, _ = load_edge_predictor_checkpoint(predictor_checkpoint, device=device)
        cumulative_path = round_dir / "edge_predictor" / "cumulative_edge_cost_dataset.json"
        cumulative_samples, _ = load_edge_cost_dataset(cumulative_path)
    return (
        summary,
        str(summary["teacher_after"]),
        predictor,
        str(predictor_checkpoint) if predictor_checkpoint else None,
        cumulative_samples,
    )


def _load_partial_records(
    path: Path,
    *,
    instances,
    splits: list[str],
    teacher_checkpoint: str,
    search_mode: str,
    requested_records: int,
):
    if not path.exists():
        return []
    records, payload = load_pseudo_expert_dataset(path)
    metadata = payload.get("metadata", {})
    expected = {
        "teacher_checkpoint": teacher_checkpoint,
        "certified_search_mode": search_mode,
        "requested_records": requested_records,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"cannot resume {path}: metadata {key}={metadata.get(key)!r}, expected {value!r}"
            )
    if len(records) > requested_records:
        raise ValueError(f"partial dataset contains too many records: {len(records)}")
    for index, record in enumerate(records):
        expected_instance = instances[index]
        restored = record.to_instance()
        if record.split != splits[index] or record.instance_id != expected_instance.instance_id:
            raise ValueError(f"partial record order mismatch at index {index}")
        if not (
            np.allclose(restored.depot, expected_instance.depot)
            and np.allclose(restored.targets, expected_instance.targets)
        ):
            raise ValueError(f"partial record instance data mismatch at index {index}")
    return records


def _train_round_edge_predictor(
    records,
    cumulative_samples: list[EdgeCostSample],
    *,
    policy_checkpoint: str,
    round_dir: Path,
    solver_config: SolverConfig,
    device: torch.device,
    args: argparse.Namespace,
    round_index: int,
):
    _emit("edge_label_generation_started", round=round_index, records=len(records))
    current_samples, label_summary = build_edge_cost_dataset(
        records,
        solver_config=solver_config,
        max_anchors_per_instance=args.edge_max_anchors_per_instance,
    )
    _emit(
        "edge_label_generation_completed",
        round=round_index,
        samples=label_summary["samples"],
        gurobi_calls=label_summary["gurobi_calls"],
        runtime=label_summary["runtime"],
    )
    cumulative_samples.extend(current_samples)
    edge_dir = round_dir / "edge_predictor"
    current_dataset_path = edge_dir / "edge_cost_dataset.json"
    cumulative_dataset_path = edge_dir / "cumulative_edge_cost_dataset.json"
    metadata = {
        "round": round_index,
        "policy_checkpoint": policy_checkpoint,
        "pseudo_expert_dataset": str((round_dir / "pseudo_experts.json").resolve()),
        "seed": args.seed,
    }
    save_edge_cost_dataset(
        current_dataset_path,
        current_samples,
        metadata=metadata,
        summary=label_summary,
    )
    save_edge_cost_dataset(
        cumulative_dataset_path,
        cumulative_samples,
        metadata={**metadata, "cumulative_through_round": round_index},
        summary={
            "samples": len(cumulative_samples),
            "split_counts": {
                split: sum(sample.split == split for sample in cumulative_samples)
                for split in sorted({sample.split for sample in cumulative_samples})
            },
        },
    )

    predictor_config = EdgePredictorConfig(
        hidden_dim=args.edge_hidden_dim,
        freeze_encoder=not args.edge_unfreeze_encoder,
    )
    training_config = EdgeTrainingConfig(
        epochs=args.edge_epochs,
        batch_size=args.edge_batch_size,
        learning_rate=args.edge_learning_rate,
        weight_decay=args.edge_weight_decay,
        grad_clip=args.edge_grad_clip,
        huber_beta=args.edge_huber_beta,
        seed=args.seed + 1000 * round_index,
    )
    predictor = initialize_edge_predictor(
        policy_checkpoint,
        device=device,
        config=predictor_config,
    )
    train_samples = [sample for sample in cumulative_samples if sample.split == "train"]
    validation_samples = [sample for sample in cumulative_samples if sample.split == "validation"]
    training = train_edge_cost_predictor(
        predictor,
        train_samples,
        validation_samples,
        device=device,
        output_dir=edge_dir / "checkpoints",
        policy_checkpoint=policy_checkpoint,
        config=training_config,
        progress_callback=lambda row: _emit(
            "edge_epoch",
            round=round_index,
            epoch=row["epoch"],
            train_normalized_mae=row["train_normalized_mae"],
            validation_normalized_mae=row["validation_normalized_mae"],
            validation_edge_spearman=row["validation_edge_spearman"],
        ),
    )
    current_by_split = {
        split: [sample for sample in current_samples if sample.split == split]
        for split in ("train", "validation", "promotion")
    }
    edge_evaluation = {
        split: evaluate_edge_predictor(
            predictor,
            samples,
            device=device,
            batch_size=args.edge_batch_size,
        )
        for split, samples in current_by_split.items()
    }
    ranking_evaluation = {
        split: evaluate_candidate_ranking(
            predictor,
            samples,
            device=device,
            top_k=args.edge_top_k,
            move_types=tuple(item.strip() for item in args.moves.split(",") if item.strip()),
        )
        for split, samples in current_by_split.items()
        if split != "train"
    }
    checkpoint = str((edge_dir / "checkpoints" / "best.pt").resolve())
    summary = {
        "label_generation": label_summary,
        "current_samples": len(current_samples),
        "cumulative_samples": len(cumulative_samples),
        "current_dataset": str(current_dataset_path.resolve()),
        "cumulative_dataset": str(cumulative_dataset_path.resolve()),
        "training": training,
        "edge_evaluation": edge_evaluation,
        "candidate_ranking": ranking_evaluation,
        "checkpoint": checkpoint,
    }
    (edge_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return predictor, checkpoint, summary


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    set_seed(args.seed)
    device = torch.device(args.device)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    _validate_resume_config(output_root, args)
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
    generation_config = PseudoExpertGenerationConfig(
        decode_type=args.teacher_decode,
        max_candidates=args.teacher_candidates,
        sample_max_rollouts=args.sample_max_rollouts,
        max_local_search_refreshes=args.local_search_refreshes,
        move_types=tuple(item.strip() for item in args.moves.split(",") if item.strip()),
    )
    imitation_config = ImitationTrainingConfig(
        epochs=args.imitation_epochs,
        batch_size=args.imitation_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        seed=args.seed,
    )

    active_teacher = str(Path(args.teacher_checkpoint).resolve())
    edge_enabled = not args.disable_edge_predictor
    active_edge_predictor_checkpoint = (
        str(Path(args.initial_edge_predictor_checkpoint).resolve())
        if args.initial_edge_predictor_checkpoint
        else None
    )
    active_edge_predictor = None
    if active_edge_predictor_checkpoint is not None:
        active_edge_predictor, _ = load_edge_predictor_checkpoint(
            active_edge_predictor_checkpoint,
            device=device,
        )
    cumulative_edge_samples: list[EdgeCostSample] = []
    rounds = []
    total_started = time.perf_counter()
    _emit(
        "run_started",
        rounds=args.rounds,
        problem_size=args.problem_size,
        min_problem_size=(args.problem_size or args.min_problem_size),
        max_problem_size=(args.problem_size or args.max_problem_size),
        imitation_epochs=args.imitation_epochs,
        edge_epochs=(args.edge_epochs if edge_enabled else 0),
        teacher=active_teacher,
        output_dir=str(output_root.resolve()),
    )
    for round_index in range(1, args.rounds + 1):
        round_started = time.perf_counter()
        round_dir = output_root / f"round_{round_index:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        if args.resume:
            restored_round = _restore_completed_round(
                round_dir,
                expected_round=round_index,
                expected_teacher=active_teacher,
                edge_enabled=edge_enabled,
                device=device,
            )
            if restored_round is not None:
                (
                    restored_summary,
                    active_teacher,
                    active_edge_predictor,
                    active_edge_predictor_checkpoint,
                    cumulative_edge_samples,
                ) = restored_round
                rounds.append(restored_summary)
                _emit(
                    "round_restored",
                    round=round_index,
                    teacher_after=active_teacher,
                    edge_predictor_after=active_edge_predictor_checkpoint,
                )
                continue
        teacher, _ = load_policy_checkpoint(
            active_teacher,
            device=device,
            max_pomo_size=args.teacher_candidates,
            start_node_strategy=args.start_node_strategy,
        )
        teacher.eval()
        teacher_digest = checkpoint_sha256(active_teacher)
        predictor_before = active_edge_predictor_checkpoint
        ranker_factory = (
            _make_ranker_factory(active_edge_predictor, device=device)
            if edge_enabled and active_edge_predictor is not None
            else None
        )
        search_mode = "predictor_guided_top_k" if ranker_factory is not None else "full_certified"
        _emit(
            "round_started",
            round=round_index,
            teacher=active_teacher,
            search_mode=search_mode,
            edge_predictor=predictor_before,
        )

        count = args.train_instances + args.validation_instances + args.promotion_instances
        min_problem_size = args.problem_size or args.min_problem_size
        max_problem_size = args.problem_size or args.max_problem_size
        generator = RandomCVTSPGenerator(
            dataset_dir=args.dataset_dir,
            min_problem_size=min_problem_size,
            max_problem_size=max_problem_size,
            seed=args.seed + 1000 * (round_index - 1),
        )
        # Sample every record independently. RandomCVTSPGenerator.sample_batch
        # intentionally shares one J across a tensor batch, which is not the
        # desired mixed-size self-improvement dataset.
        instances = [
            generator.sample_instance(problem_size=args.problem_size)
            for _ in range(count)
        ]
        _emit(
            "round_instances_sampled",
            round=round_index,
            problem_sizes=[instance.J for instance in instances],
            min_problem_size=min(instance.J for instance in instances),
            max_problem_size=max(instance.J for instance in instances),
        )
        splits = (
            ["train"] * args.train_instances
            + ["validation"] * args.validation_instances
            + ["promotion"] * args.promotion_instances
        )
        pseudo_expert_path = round_dir / "pseudo_experts.json"
        records = (
            _load_partial_records(
                pseudo_expert_path,
                instances=instances,
                splits=splits,
                teacher_checkpoint=active_teacher,
                search_mode=search_mode,
                requested_records=count,
            )
            if args.resume
            else []
        )
        completed_records = len(records)
        if completed_records:
            _emit(
                "partial_records_restored",
                round=round_index,
                completed=completed_records,
                requested=count,
            )
        pending = zip(instances[completed_records:], splits[completed_records:])
        for record_index, (instance, split) in enumerate(
            pending,
            start=completed_records + 1,
        ):
            record = generate_pseudo_expert_record(
                teacher,
                instance,
                split=split,
                device=device,
                solver_config=solver_config,
                generation_config=generation_config,
                candidate_ranker_factory=ranker_factory,
                max_exact_candidates=(args.edge_top_k if ranker_factory is not None else None),
            )
            records.append(record)
            _emit(
                "pseudo_expert_record_completed",
                round=round_index,
                record=record_index,
                requested=count,
                instance_id=record.instance_id,
                split=record.split,
                teacher_objective=record.teacher_best_objective,
                expert_objective=record.expert_objective,
                generated_moves=record.generated_moves,
                exact_ub_evaluations=record.evaluated_moves,
                pruned_moves=record.pruned_moves,
                gurobi_calls=record.gurobi_calls,
            )
            # Persist after every expensive record so an interrupted run keeps
            # all completed expert solves.
            save_pseudo_expert_dataset(
                pseudo_expert_path,
                records,
                metadata={
                    "round": round_index,
                    "teacher_checkpoint": active_teacher,
                    "teacher_checkpoint_sha256": teacher_digest,
                    "generation_config": asdict(generation_config),
                    "certified_search_mode": search_mode,
                    "edge_predictor_checkpoint": predictor_before,
                    "edge_top_k": args.edge_top_k if ranker_factory is not None else None,
                    "completed_records": record_index,
                    "requested_records": count,
                    "seed": args.seed + 1000 * (round_index - 1),
                    "problem_sizes": [item.J for item in instances],
                },
            )

        train_records = [record for record in records if record.split == "train"]
        validation_records = [record for record in records if record.split == "validation"]
        promotion_records = [record for record in records if record.split == "promotion"]
        student, _ = load_policy_checkpoint(
            active_teacher,
            device=device,
            max_pomo_size=args.teacher_candidates,
            start_node_strategy=args.start_node_strategy,
        )
        training = train_imitation_student(
            student,
            train_records,
            validation_records,
            device=device,
            output_dir=round_dir / "student",
            source_checkpoint=active_teacher,
            config=imitation_config,
            progress_callback=lambda row, current_round=round_index: _emit(
                "imitation_epoch",
                round=current_round,
                epoch=row["epoch"],
                train_nll=row["train_nll"],
                validation_nll=row["validation_nll"],
                train_token_accuracy=row["train_token_accuracy"],
                validation_token_accuracy=row["validation_token_accuracy"],
            ),
        )

        gate_records = promotion_records or validation_records or train_records
        promotion = evaluate_policy_exact(
            student,
            gate_records,
            device=device,
            solver_config=solver_config,
            decode_type=args.teacher_decode,
            max_candidates=args.teacher_candidates,
            sample_max_rollouts=args.sample_max_rollouts,
        )
        promotion["gate_split"] = gate_records[0].split
        promotion["required_improvement"] = args.promotion_epsilon
        mean_improved = (
            promotion["student_mean_objective"]
            < promotion["teacher_mean_objective"] - args.promotion_epsilon
        )
        no_regression_gate_passed = (
            not args.require_no_promotion_losses or promotion["student_losses"] == 0
        )
        promotion["rule"] = (
            "strict mean improvement plus zero per-instance regressions"
            if args.require_no_promotion_losses
            else "strict mean improvement"
        )
        promotion["mean_improvement_gate_passed"] = mean_improved
        promotion["no_regression_gate_required"] = args.require_no_promotion_losses
        promotion["no_regression_gate_passed"] = no_regression_gate_passed
        promotion["promoted"] = mean_improved and no_regression_gate_passed
        _emit(
            "promotion_completed",
            round=round_index,
            teacher_mean_objective=promotion["teacher_mean_objective"],
            student_mean_objective=promotion["student_mean_objective"],
            student_wins=promotion["student_wins"],
            student_losses=promotion["student_losses"],
            promoted=promotion["promoted"],
        )
        candidate_student = str((round_dir / "student" / "best.pt").resolve())
        teacher_before = active_teacher
        if promotion["promoted"]:
            active_teacher = candidate_student

        edge_training = None
        if edge_enabled:
            (
                active_edge_predictor,
                active_edge_predictor_checkpoint,
                edge_training,
            ) = _train_round_edge_predictor(
                records,
                cumulative_edge_samples,
                policy_checkpoint=active_teacher,
                round_dir=round_dir,
                solver_config=solver_config,
                device=device,
                args=args,
                round_index=round_index,
            )

        round_summary = {
            "round": round_index,
            "teacher_before": teacher_before,
            "student_checkpoint": candidate_student,
            "teacher_after": active_teacher,
            "pseudo_experts": summarize_pseudo_experts(records),
            "certified_search": {
                "mode": search_mode,
                "predictor_checkpoint_before": predictor_before,
                "top_k": args.edge_top_k if ranker_factory is not None else None,
            },
            "imitation_training": training,
            "promotion": promotion,
            "edge_predictor": {
                "enabled": edge_enabled,
                "checkpoint_before": predictor_before,
                "used_for_generation": ranker_factory is not None,
                "checkpoint_after": active_edge_predictor_checkpoint,
                "round_training": edge_training,
            },
            "runtime": time.perf_counter() - round_started,
        }
        (round_dir / "summary.json").write_text(
            json.dumps(round_summary, indent=2),
            encoding="utf-8",
        )
        rounds.append(round_summary)
        _emit(
            "round_completed",
            round=round_index,
            teacher_after=active_teacher,
            edge_predictor_after=active_edge_predictor_checkpoint,
            runtime=round_summary["runtime"],
        )

    summary = {
        "phase": "predictor_guided_certified_self_improvement",
        "initial_teacher": str(Path(args.teacher_checkpoint).resolve()),
        "final_teacher": active_teacher,
        "initial_edge_predictor": (
            str(Path(args.initial_edge_predictor_checkpoint).resolve())
            if args.initial_edge_predictor_checkpoint
            else None
        ),
        "final_edge_predictor": active_edge_predictor_checkpoint,
        "rounds_requested": args.rounds,
        "rounds_completed": len(rounds),
        "config": vars(args),
        "rounds": rounds,
        "runtime": time.perf_counter() - total_started,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _emit(
        "run_completed",
        rounds_completed=len(rounds),
        promotions=sum(item["promotion"]["promoted"] for item in rounds),
        final_teacher=active_teacher,
        final_edge_predictor=active_edge_predictor_checkpoint,
        runtime=summary["runtime"],
        summary_path=str(summary_path.resolve()),
    )
    print(json.dumps({
        "rounds_completed": len(rounds),
        "initial_teacher": summary["initial_teacher"],
        "final_teacher": summary["final_teacher"],
        "final_edge_predictor": summary["final_edge_predictor"],
        "promotions": sum(item["promotion"]["promoted"] for item in rounds),
        "runtime": summary["runtime"],
        "summary_path": str(summary_path),
    }, indent=2))


if __name__ == "__main__":
    main()
