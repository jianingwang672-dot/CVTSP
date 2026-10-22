"""Frozen-policy search pilot: does refreshing C0 from new exact solutions help?

Reuses the saved inputs of ``outputs/lkh_search_pilot_20261014`` (frozen nm20
E20 Reference/Predictor, 100 instances, exact Anchor solution, 200 candidates).
For each instance and each search backend (swap+2-opt, LKH with the training
time budget) it runs R rounds:

1. build C0 from the exact solution of the route solved last (round 1: the
   Anchor), whether or not it beat the incumbent, so the tight region of the
   bound moves away from the Anchor;
2. reselect the quality-diverse Top-30 starts under that C0 from the original
   200 candidates plus every route this backend has already produced;
3. search on C0 without Gurobi;
4. solve the lowest-C0 final route that has not been solved yet (one Gurobi
   call per round, near-miss always attempted); if its exact makespan is
   strictly below the incumbent, it becomes the incumbent (the Expert).

Round 1 equals the training pipeline with a 100% near-miss quota; later rounds
are the "C0 refresh" variant. No model is trained.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("CVTSP_VECTORIZED_LOCAL_SEARCH", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "outputs/lkh_search_pilot_20261014"
BACKENDS = ("swap_two_opt", "lkh")
SIZES = (20, 40, 60, 80, 100)

# Values of the paired nm20/LKH training runs (resolved_config.json).
ABSOLUTE_EPSILON = 1e-4
RELATIVE_EPSILON = 1e-8
UPPER_BOUND_TOLERANCE = 1e-4
FEASIBILITY_TOLERANCE = 1e-5
LKH_BUDGET_COEFFICIENT = 1.5e-4


def solver_config():
    from src.CVPSolver import SolverConfig

    return SolverConfig(
        solver_backend="gurobi", gurobi_threads=1, output_flag=0,
        gurobi_params={"FeasibilityTol": 1e-8, "OptimalityTol": 1e-8, "NumericFocus": 2},
        gurobi_explicit_qcp=True, collect_duals=True,
    )


def search_config():
    from src.CertifiedLocalSearch import FixedAnchorLocalSearchConfig

    return FixedAnchorLocalSearchConfig(
        max_local_moves=30, absolute_epsilon=ABSOLUTE_EPSILON,
        relative_epsilon=RELATIVE_EPSILON, move_types=("swap", "two_opt"),
    )


def build_c0(instance, solution):
    from src.CertifiedUpperBound import FixedSolutionUpperBound
    from src.DualContributions import directed_total_contribution_matrix
    from src.JointSelfImprovement import directed_route_upper_bound

    majorizer = FixedSolutionUpperBound.from_solution(
        instance, solution, feasibility_tolerance=FEASIBILITY_TOLERANCE, require_optimal=True)
    matrix = directed_total_contribution_matrix(majorizer)
    identity = directed_route_upper_bound(list(solution.sequence), matrix)
    if abs(identity - float(solution.objective)) > UPPER_BOUND_TOLERANCE:
        raise RuntimeError("incumbent upper-bound identity failed")
    return majorizer, matrix


def run_search(backend, majorizer, matrix, starts, quality, diversity, seed, work_root):
    from src.CertifiedLocalSearch import run_fixed_anchor_local_search
    from src.JointSelfImprovement import _run_lkh_fixed_anchor_searches

    config = search_config()
    if backend == "swap_two_opt":
        searches = [run_fixed_anchor_local_search(majorizer, route, config) for route in starts]
        return [(list(s.final_sequence), float(s.final_upper_bound)) for s in searches], {}
    task = SimpleNamespace(
        use_dual_pruning=False, perturbation_restarts=0,
        lkh_binary=str(ROOT / "third_party/lkh3_20261014/LKH"),
        lkh_budget_coefficient=LKH_BUDGET_COEFFICIENT, lkh_desired_scale=1_000_000,
        lkh_trials=1, lkh_max_candidates=5, lkh_move_type=5,
        lkh_native_seconds=2.0, lkh_process_timeout_seconds=5.0,
        lkh_work_root=str(work_root), local_search_seed=seed, local_search_config=config,
    )
    searches, stats = _run_lkh_fixed_anchor_searches(
        matrix, starts, quality_count=quality, diversity_count=diversity, task=task)
    return [(list(s.final_sequence), float(s.final_upper_bound)) for s in searches], stats


def refresh_case(task):
    from src.CertifiedUpperBound import is_optimal_gurobi_solution, validate_tour
    from src.CVPSolver import solve
    from src.JointSelfImprovement import (_deduplicate_tours, directed_route_upper_bound,
                                          select_quality_diverse_start_indices)

    case_dir, rounds, work_root = task
    data = torch.load(Path(case_dir) / "input.pt", map_location="cpu", weights_only=False)
    instance = data["instance"]
    anchor_solution = data["anchor_solution"]
    anchor = list(validate_tour(anchor_solution.sequence, instance.J))
    anchor_value = float(anchor_solution.objective)
    candidates = _deduplicate_tours(data["candidates"], instance.J)
    # Round-1 inputs must reproduce the saved pilot exactly.
    _, first_matrix = build_c0(instance, anchor_solution)
    if not np.allclose(first_matrix, data["matrix"], rtol=0, atol=1e-9):
        raise RuntimeError(f"rebuilt C0 differs from the saved pilot matrix: {case_dir}")
    eps = search_config().acceptance_epsilon(anchor_value)
    record = {"case": Path(case_dir).name, "J": int(instance.J), "anchor_exact": anchor_value,
              "backends": {}}
    for backend in BACKENDS:
        incumbent_solution = anchor_solution
        incumbent = list(anchor)
        base_solution = anchor_solution
        solved = {tuple(anchor)}
        produced: list[list[int]] = []
        calls = 0
        history = []
        for round_index in range(1, rounds + 1):
            started = time.perf_counter()
            incumbent_value = float(incumbent_solution.objective)
            base = list(validate_tour(base_solution.sequence, instance.J))
            majorizer, matrix = build_c0(instance, base_solution)
            pool = _deduplicate_tours(candidates + produced + [incumbent, base], instance.J)
            bounds = np.asarray([directed_route_upper_bound(r, matrix) for r in pool])
            selected, quality, diversity = select_quality_diverse_start_indices(
                pool, bounds, top_k=30, diversity_count=10, diversity_pool_size=60,
                diversity_max_relative_gap=0.05)
            starts = [pool[int(k)] for k in selected]
            if round_index == 1 and starts != [list(map(int, s)) for s in data["starts"]]:
                raise RuntimeError(f"round-1 starts differ from the saved pilot: {case_dir}")
            finals, stats = run_search(backend, majorizer, matrix, starts, quality, diversity,
                                       seed=20261016 + round_index * 1000, work_root=work_root)
            search_seconds = time.perf_counter() - started
            for route, upper in finals:
                if abs(directed_route_upper_bound(route, matrix) - upper) > UPPER_BOUND_TOLERANCE:
                    raise RuntimeError("search returned an inconsistent C0 value")
            produced.extend(route for route, _ in finals)
            others = [(upper, route) for route, upper in finals if tuple(route) not in solved]
            row = {
                "round": round_index,
                "incumbent_before": incumbent_value,
                "c0_base_is_incumbent": base == incumbent,
                "distinct_finals": len({tuple(r) for r, _ in finals}),
                "finals_equal_incumbent": float(np.mean([r == incumbent for r, _ in finals])),
                "finals_equal_c0_base": float(np.mean([r == base for r, _ in finals])),
                "best_final_upper_bound": float(min(u for _, u in finals)),
                "search_seconds": search_seconds,
                "lkh_completed_starts": stats.get("completed_starts"),
                "candidate_kind": None, "candidate_exact": None, "candidate_upper_bound": None,
                "accepted": False,
            }
            if others:
                upper, route = min(others, key=lambda item: item[0])
                kind = "certified" if upper < incumbent_value - eps else "near_miss"
                solution = solve(instance, list(route), solver_config())
                calls += 1
                solved.add(tuple(route))
                if not is_optimal_gurobi_solution(solution) or not np.isfinite(solution.objective):
                    raise RuntimeError("candidate exact solve failed")
                value = float(solution.objective)
                if value > upper + UPPER_BOUND_TOLERANCE:
                    raise RuntimeError("candidate violated its strict upper bound")
                if kind == "certified" and value >= incumbent_value - eps:
                    raise RuntimeError("certified candidate did not improve the incumbent")
                row.update(candidate_kind=kind, candidate_exact=value, candidate_upper_bound=upper)
                if value < incumbent_value - eps:
                    row["accepted"] = True
                    incumbent_solution, incumbent = solution, list(route)
                # Refresh: the next C0 is tight around the route just solved.
                base_solution = solution
            row["incumbent_after"] = float(incumbent_solution.objective)
            row["cumulative_extra_gurobi_calls"] = calls
            history.append(row)
        record["backends"][backend] = history
    return record


def paired(left, right, resamples=10000, seed=0):
    delta = np.asarray(right) - np.asarray(left)
    rng = np.random.default_rng(seed)
    boot = delta[rng.integers(0, len(delta), (resamples, len(delta)))].mean(axis=1)
    return {"mean_right_minus_left": float(delta.mean()),
            "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "right_better": int(np.sum(delta < -1e-9)), "ties": int(np.sum(np.abs(delta) <= 1e-9)),
            "right_worse": int(np.sum(delta > 1e-9))}


def summarize(records, rounds, output):
    tables = {}
    for size in ("all",) + SIZES:
        group = records if size == "all" else [r for r in records if r["J"] == size]
        if not group:
            continue
        tables[str(size)] = {}
        for backend in BACKENDS:
            for r in range(rounds):
                rows = [x["backends"][backend][r] for x in group]
                anchor = np.asarray([x["anchor_exact"] for x in group])
                after = np.asarray([row["incumbent_after"] for row in rows])
                tables[str(size)][f"{backend}_r{r + 1}"] = {
                    "count": len(group),
                    "expert_mean": float(after.mean()),
                    "improvement_percent_vs_anchor": float(np.mean(100 * (anchor - after) / anchor)),
                    "improved_vs_anchor": int(np.sum(after < anchor - 1e-9)),
                    "accepted_this_round": int(sum(row["accepted"] for row in rows)),
                    "certified_candidates": int(sum(row["candidate_kind"] == "certified" for row in rows)),
                    "no_candidate": int(sum(row["candidate_kind"] is None for row in rows)),
                    "extra_gurobi_calls_per_instance": float(np.mean(
                        [row["cumulative_extra_gurobi_calls"] for row in rows])),
                    "distinct_finals_mean": float(np.mean([row["distinct_finals"] for row in rows])),
                    "finals_equal_incumbent_mean": float(np.mean(
                        [row["finals_equal_incumbent"] for row in rows])),
                    "search_seconds_mean": float(np.mean([row["search_seconds"] for row in rows])),
                }
    value = {f"{b}_r{r + 1}": [x["backends"][b][r]["incumbent_after"] for x in records]
             for b in BACKENDS for r in range(rounds)}
    comparisons = {}
    for key in value:
        if key != "swap_two_opt_r1":
            comparisons[f"{key} - swap_two_opt_r1"] = paired(value["swap_two_opt_r1"], value[key])
    for r in range(1, rounds + 1):
        comparisons[f"lkh_r{r} - swap_two_opt_r{r}"] = paired(
            value[f"swap_two_opt_r{r}"], value[f"lkh_r{r}"])
    result = {"complete": True, "instances": len(records), "rounds": rounds,
              "by_J": tables, "paired_expert_makespan": comparisons, "records": records}
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")

    lines = ["# C0刷新搜索对照（冻结nm20-E20策略，100例）", "",
             "每轮：用上一轮刚精算的路线的精确解重建C0（第1轮为Anchor）→ 重新选Top-30 → 搜索 → 精算C0最低的、尚未精算过的候选；只有精确值严格更好才更新Expert。",
             "第1轮等同于训练流程（near-miss总是尝试）；第2轮起是C0刷新。", "",
             "| J | 方法_轮 | Expert makespan | 相对Anchor改善 | 改进实例 | 本轮被接受 | 额外精算/例 | 不同终点 | 终点=incumbent | 搜索秒/例 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for size, values in tables.items():
        for name, v in values.items():
            lines.append(
                f"| {size} | {name} | {v['expert_mean']:.6f} | {v['improvement_percent_vs_anchor']:.3f}% | "
                f"{v['improved_vs_anchor']}/{v['count']} | {v['accepted_this_round']} | "
                f"{v['extra_gurobi_calls_per_instance']:.2f} | {v['distinct_finals_mean']:.2f} | "
                f"{100 * v['finals_equal_incumbent_mean']:.1f}% | {v['search_seconds_mean']:.3f} |")
    lines += ["", "配对差值（负值表示右侧方法的Expert更好）：", ""]
    for name, v in comparisons.items():
        lo, hi = v["ci95"]
        lines.append(f"- {name}: {v['mean_right_minus_left']:+.6f}，95%区间[{lo:+.6f}, {hi:+.6f}]；"
                     f"胜/平/负={v['right_better']}/{v['ties']}/{v['right_worse']}")
    lines += ["", "局限：冻结策略的搜索诊断，不是训练效果；LKH按训练时间预算max(0.005,1.5e-4·J²)秒/轮，墙钟受机器负载影响。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/c0_refresh_pilot_20261016")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Smoke test: first N cases per J.")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if (output / "comparison.json").exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    (output / "cases").mkdir(parents=True, exist_ok=True)
    work_root = output / ".lkh_tmp"
    work_root.mkdir(exist_ok=True)
    cases = sorted((PILOT / "cases").glob("case-*"))
    if args.limit:
        cases = [c for size in SIZES for c in [x for x in cases if x.name.endswith(f"J{size}")][:args.limit]]
    tasks = [(str(c), args.rounds, str(work_root)) for c in cases]
    records = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as executor:
        for record in executor.map(refresh_case, tasks, chunksize=1):
            (output / "cases" / f"{record['case']}.json").write_text(json.dumps(record, indent=2))
            records.append(record)
            print(f"{len(records)}/{len(tasks)} {record['case']} "
                  + " ".join(f"{b}={record['backends'][b][-1]['incumbent_after']:.4f}" for b in BACKENDS),
                  flush=True)
    summarize(records, args.rounds, output)
    print(f"C0_REFRESH_COMPLETE seconds={time.perf_counter() - started:.0f}", flush=True)


if __name__ == "__main__":
    main()
