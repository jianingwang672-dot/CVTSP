"""Measure exact regret of the Target route selected by training's UB table.

The same fresh instances are evaluated with two joint checkpoints.  For each
checkpoint, Reference sampling, predictor ranking, Anchor solve, local search,
and the minimum of whole-route Anchor/Expert bounds follow the trainer.  Every
sampled Target POMO route is then solved with the same exact Gurobi formulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch

from src.CertifiedUpperBound import is_optimal_gurobi_solution
from src.FixedValidation import load_or_create_fixed_validation_sets
from src.JointSelfImprovement import batched_directed_step_costs
from src.JointSelfImprovementTrainer import JointSelfImprovementTrainer
from src.PseudoExpert import instance_to_dict
from src.RewardWorkerPool import RewardWorkerPool
from src.TSPUtils import set_seed


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "outputs/train__scratch_e500_lr1e4_min1e7_w128_qdhybrid_replay30_vectorized"
SIZES = (20, 40, 60, 80, 100)


def fingerprint(instance) -> str:
    payload = instance_to_dict(instance)
    payload.pop("instance_id", None)
    payload.pop("path", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def create_dataset(directory: Path, *, per_size: int, seed: int):
    groups = load_or_create_fixed_validation_sets(
        dataset_dir=ROOT / "instance/Data",
        validation_dir=directory,
        problem_sizes=SIZES,
        per_size=per_size,
        seed=seed,
    )
    all_hashes = [fingerprint(x) for size in SIZES for x in groups[size]]
    if len(set(all_hashes)) != len(all_hashes):
        raise RuntimeError("diagnostic dataset contains duplicate instances")
    for other in (ROOT / "validation/macsim_fixed", ROOT / "validation/pomo_selection_20260921"):
        old = set()
        for size in SIZES:
            data = torch.load(other / f"J{size}.pt", map_location="cpu", weights_only=False)
            from src.PseudoExpert import instance_from_dict
            old.update(fingerprint(instance_from_dict(row)) for row in data["instances"])
        if set(all_hashes) & old:
            raise RuntimeError(f"diagnostic dataset overlaps {other}")
    manifest = {
        "purpose": "target_UB_route_selection_regret_diagnostic_only",
        "seed": seed,
        "sizes": list(SIZES),
        "per_size": per_size,
        "count": len(all_hashes),
        "content_sha256": hashlib.sha256("\n".join(all_hashes).encode()).hexdigest(),
        "files_sha256": {
            f"J{size}.pt": hashlib.sha256((directory / f"J{size}.pt").read_bytes()).hexdigest()
            for size in SIZES
        },
        "disjoint_from": ["validation/macsim_fixed", "validation/pomo_selection_20260921"],
    }
    existing = directory / "manifest.json"
    if existing.exists() and json.loads(existing.read_text()) != manifest:
        raise RuntimeError("diagnostic manifest changed")
    save_json(existing, manifest)
    return groups, manifest


def load_trainer(checkpoint: Path, output_dir: Path, workers: int):
    params = json.loads((TRAIN_DIR / "params.json").read_text())
    env = params["env_params"]
    model = params["model_params"]
    optimizer = params["optimizer_params"]
    trainer_params = params["trainer_params"]
    trainer_params.update({
        "use_cuda": False,
        "strict_resume": False,
        "self_improvement_parallel_workers": workers,
        "checkpoint_path": str(checkpoint),
        "result_folder": str(output_dir / "trainer_state"),
    })
    trainer = JointSelfImprovementTrainer(env, model, optimizer, trainer_params)
    trainer.model.train()
    trainer.edge_decoder.eval()
    return trainer


def examine_batch(trainer, instances, pool, seed):
    # Use one reproducible sampled POMO draw, as in the RL training path.
    set_seed(seed)
    with torch.no_grad():
        trainer.env.load_problems(instances, trainer.device)
        reset_state, _, _ = trainer.env.reset()
        trainer.model.pre_forward(reset_state.node_features)
        predicted = trainer._predict_contribution_matrices(instances)
        rollout, _ = trainer._rollout_target()
        target_routes = rollout.to_sequence_lists()
        reference_routes = trainer._generate_reference_candidates(instances)
        anchors, _, _ = trainer._select_anchor_sequences(instances, reference_routes, predicted)
    trainer._active_near_miss_rate = trainer._effective_near_miss_rate()
    results, _, _ = trainer._run_sparse_self_improvement(instances, reference_routes, anchors)
    matrices = [result.anchor_upper_bound_matrix for result in results]
    bounds = batched_directed_step_costs(target_routes, matrices).sum(axis=-1).astype(np.float32)
    for index, result in enumerate(results):
        if result.expert_upper_bound_matrix is None:
            continue
        expert_bounds = batched_directed_step_costs(
            [target_routes[index]], [result.expert_upper_bound_matrix]
        )[0].sum(axis=-1).astype(np.float32)
        bounds[index] = np.minimum(bounds[index], expert_bounds)

    tasks = []
    mapping = []
    for instance_index, (instance, routes) in enumerate(zip(instances, target_routes)):
        seen = {}
        for route in routes:
            key = tuple(route)
            if key not in seen:
                seen[key] = len(tasks)
                tasks.append((instance, route, trainer.si_final_solver_config))
        mapping.append([seen[tuple(route)] for route in routes])
    solved = pool.map(tasks)
    exact = []
    for _, _, solution in solved:
        if not is_optimal_gurobi_solution(solution) or not np.isfinite(solution.objective):
            raise RuntimeError(f"nonoptimal exact POMO route: {solution.status}")
        exact.append(float(solution.objective))

    rows = []
    for index, instance in enumerate(instances):
        values = np.asarray([exact[k] for k in mapping[index]], dtype=float)
        upper = bounds[index].astype(float)
        selected = int(np.argmin(bounds[index]))
        best = int(np.argmin(values))
        exact_selected = float(values[selected])
        exact_best = float(values[best])
        if exact_selected + 1e-6 < exact_best:
            raise RuntimeError("selected exact is below best exact")
        if np.any(values > upper + trainer.si_upper_bound_tolerance):
            raise RuntimeError("exact route exceeds certified upper bound")
        rank = 1 + int(np.sum(values < exact_selected - 1e-6))
        rows.append({
            "instance_id": instance.instance_id,
            "fingerprint": fingerprint(instance),
            "J": instance.J,
            "route_count": len(values),
            "unique_route_count": len(set(map(tuple, target_routes[index]))),
            "selected_index": selected,
            "best_exact_index": best,
            "selected_exact_rank": rank,
            "selected_exact": exact_selected,
            "best_exact": exact_best,
            "regret": exact_selected - exact_best,
            "regret_percent": 100 * (exact_selected - exact_best) / max(abs(exact_best), 1e-12),
            "selected_upper_bound": float(upper[selected]),
            "best_exact_route_upper_bound": float(upper[best]),
            "mean_bound_gap": float(np.mean(upper - values)),
            "bounds": upper.tolist(),
            "exact": values.tolist(),
        })
    return rows


def summarize(rows):
    result = {}
    for label, subset in [("all", rows)] + [(str(size), [x for x in rows if x["J"] == size]) for size in SIZES]:
        if not subset:
            continue
        regrets = np.asarray([x["regret"] for x in subset])
        result[label] = {
            "instances": len(subset),
            "selection_exact_match_rate": float(np.mean(regrets <= 1e-6)),
            "mean_regret": float(regrets.mean()),
            "median_regret": float(np.median(regrets)),
            "p90_regret": float(np.quantile(regrets, .9)),
            "mean_regret_percent": float(np.mean([x["regret_percent"] for x in subset])),
            "selected_exact_mean": float(np.mean([x["selected_exact"] for x in subset])),
            "best_exact_mean": float(np.mean([x["best_exact"] for x in subset])),
            "mean_selected_rank": float(np.mean([x["selected_exact_rank"] for x in subset])),
            "mean_bound_gap": float(np.mean([x["mean_bound_gap"] for x in subset])),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="LABEL=PATH, e.g. e100=outputs/.../checkpoint-100.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    data_dir = ROOT / f"validation/target_selection_diagnostic_{args.seed}"
    groups, manifest = create_dataset(data_dir, per_size=args.per_size, seed=args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = {}
    for item in args.checkpoint:
        label, path = item.split("=", 1)
        path = Path(path).resolve()
        if not path.is_file() or label in configs:
            raise ValueError(f"missing or duplicate checkpoint: {item}")
        configs[label] = path
    params = {
        "dataset": str(data_dir), "manifest": manifest, "batch_size": args.batch_size,
        "workers": args.workers, "target_rollout": "sampled_POMO_J_training_mode",
        "augmentation": False, "checkpoint_sha256": {
            label: hashlib.sha256(path.read_bytes()).hexdigest() for label, path in configs.items()
        },
    }
    save_json(output_dir / "params.json", params)
    all_rows = {}
    pool = RewardWorkerPool(args.workers, env_output_flag=0)
    try:
        for label, checkpoint in configs.items():
            print(f"START {label} {checkpoint}", flush=True)
            trainer = load_trainer(checkpoint, output_dir / label, args.workers)
            try:
                rows = []
                path = output_dir / label / "results.json"
                if path.exists():
                    old = json.loads(path.read_text())
                    rows = old["records"]
                completed = len(rows)
                ordered = [instance for size in SIZES for instance in groups[size]]
                for start in range(completed, len(ordered), args.batch_size):
                    batch = ordered[start:start + args.batch_size]
                    new = examine_batch(trainer, batch, pool, args.seed + start)
                    rows.extend(new)
                    save_json(path, {"label": label, "complete": len(rows) == len(ordered),
                                     "summary": summarize(rows), "records": rows})
                    print(f"{label} {len(rows)}/{len(ordered)} regret={summarize(rows)['all']['mean_regret']:.6f}", flush=True)
                all_rows[label] = rows
            finally:
                trainer.close()
    finally:
        pool.close()
    if len(all_rows) == 2:
        labels = list(all_rows)
        left, right = all_rows[labels[0]], all_rows[labels[1]]
        if [x["fingerprint"] for x in left] != [x["fingerprint"] for x in right]:
            raise RuntimeError("models were not evaluated on the same instances")
        delta = np.asarray([b["regret"] - a["regret"] for a, b in zip(left, right)])
        paired = {"labels": labels, "mean_regret_delta_second_minus_first": float(delta.mean()),
                  "second_more_regret": int(np.sum(delta > 1e-6)),
                  "ties": int(np.sum(np.abs(delta) <= 1e-6)),
                  "second_less_regret": int(np.sum(delta < -1e-6))}
        save_json(output_dir / "paired_summary.json", paired)
        print("PAIRED", json.dumps(paired), flush=True)


if __name__ == "__main__":
    main()
