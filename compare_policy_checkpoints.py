from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.CVPSolver import SolverConfig, solve
from src.ImitationLearning import load_policy_checkpoint
from src.PseudoExpert import generate_policy_candidates
from src.TSProblemDef import RandomCVTSPGenerator
from src.TSPUtils import set_seed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare teacher and student raw routes using exact Gurobi objectives."
    )
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dataset-dir", default="instance/Data")
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--problem-size", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--start-node-strategy", default="nearest_depot")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.instances <= 0 or args.problem_size <= 0 or args.candidates <= 0:
        raise ValueError("instances, problem-size, and candidates must be positive")
    started = time.perf_counter()
    set_seed(args.seed)
    device = torch.device(args.device)
    teacher, _ = load_policy_checkpoint(
        args.teacher_checkpoint,
        device=device,
        max_pomo_size=args.candidates,
        start_node_strategy=args.start_node_strategy,
    )
    student, _ = load_policy_checkpoint(
        args.student_checkpoint,
        device=device,
        max_pomo_size=args.candidates,
        start_node_strategy=args.start_node_strategy,
    )
    teacher.eval()
    student.eval()
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

    rows = []
    gurobi_calls = 0
    for _ in range(args.instances):
        instance = generator.sample_instance(problem_size=args.problem_size)
        best = []
        for role, model in (("teacher", teacher), ("student", student)):
            candidates = generate_policy_candidates(
                model,
                instance,
                device=device,
                decode_type="greedy",
                max_candidates=args.candidates,
            )
            solved = []
            for sequence in candidates:
                solution = solve(instance, sequence, solver_config)
                gurobi_calls += 1
                if not solution.success or str(solution.status).strip() != "2":
                    raise RuntimeError(
                        f"{role} exact solve failed for {instance.instance_id}: {solution.status}"
                    )
                solved.append((float(solution.objective), sequence))
            best.append(min(solved, key=lambda item: item[0]))
        teacher_best, student_best = best
        rows.append(
            {
                "instance_id": instance.instance_id,
                "teacher_objective": teacher_best[0],
                "teacher_sequence": list(teacher_best[1]),
                "student_objective": student_best[0],
                "student_sequence": list(student_best[1]),
                "objective_improvement": float(teacher_best[0] - student_best[0]),
            }
        )

    teacher_values = np.asarray([row["teacher_objective"] for row in rows], dtype=float)
    student_values = np.asarray([row["student_objective"] for row in rows], dtype=float)
    improvements = teacher_values - student_values
    summary = {
        "instances": len(rows),
        "problem_size": args.problem_size,
        "candidates_per_policy": args.candidates,
        "teacher_mean_objective": float(teacher_values.mean()),
        "student_mean_objective": float(student_values.mean()),
        "mean_objective_improvement": float(improvements.mean()),
        "relative_improvement_percent": float(
            100.0 * improvements.mean() / max(abs(teacher_values.mean()), 1e-12)
        ),
        "student_wins": int((improvements > 1e-6).sum()),
        "ties": int((np.abs(improvements) <= 1e-6).sum()),
        "student_losses": int((improvements < -1e-6).sum()),
        "max_student_regression": float(max(0.0, -float(improvements.min()))),
        "gurobi_calls": gurobi_calls,
        "runtime": time.perf_counter() - started,
        "seed": args.seed,
    }
    payload = {
        "teacher_checkpoint": str(Path(args.teacher_checkpoint).resolve()),
        "student_checkpoint": str(Path(args.student_checkpoint).resolve()),
        "summary": summary,
        "records": rows,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"details_path={output_path}")


if __name__ == "__main__":
    main()
