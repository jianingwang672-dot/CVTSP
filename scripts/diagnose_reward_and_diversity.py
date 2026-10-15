"""Zero-training diagnostics for the hybrid POMO reward and POMO diversity.

``reward``: reuse the saved Target-selection diagnostic (every sampled Target
POMO route has its certified upper bound and exact Gurobi makespan) and rebuild
the advantages that ``hybrid_pomo_exact_expert`` training actually used: the
lowest-bound route carries its exact value, every other route its upper bound,
then per-instance centering and std normalization.  The same groups are scored
under alternative reward constructions and compared with the all-exact oracle.

``diversity``: greedy POMO (test protocol: nearest_depot starts, POMO=J, no
augmentation) and one sampled POMO draw for several checkpoints on the same
fixed diagnostic instances.  Every distinct greedy route is solved exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
SIZES = (20, 40, 60, 80, 100)
DEFAULT_SOURCE = ROOT / "outputs/target_selection_diagnostic_20260922"
DEFAULT_DATA = ROOT / "validation/target_selection_diagnostic_20260922"
EPS = 1e-6


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalized_advantage(cost: np.ndarray) -> np.ndarray:
    """Training rule: reward=-cost, subtract group mean, divide by group std."""
    advantage = -(cost - cost.mean())
    return advantage / max(float(advantage.std()), EPS)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator > 0 else float("nan")


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.ptp(a) <= 0 or np.ptp(b) <= 0:
        return float("nan")
    return float(spearmanr(a, b)[0])


def reward_variants(upper: np.ndarray, exact: np.ndarray) -> dict[str, np.ndarray]:
    """Advantage vectors over all POMO routes for each reward construction."""
    order = np.argsort(upper, kind="stable")
    selected = int(order[0])
    variants = {}
    hybrid = upper.copy()
    hybrid[selected] = exact[selected]
    variants["current_hybrid"] = normalized_advantage(hybrid)
    variants["all_upper_bound"] = normalized_advantage(upper)
    for m in (2, 4, 8):
        # Exact only for the m lowest-bound routes, advantages only among them.
        solved = order[: min(m, len(order))]
        vector = np.zeros_like(exact)
        if len(solved) > 1:
            vector[solved] = normalized_advantage(exact[solved])
        variants[f"top{m}_exact_only"] = vector
    for m in (4, 8, 16):
        # Same mixed rule as training, with m lowest-bound routes made exact.
        mixed = upper.copy()
        mixed[order[:m]] = exact[order[:m]]
        variants[f"hybrid_top{m}_exact"] = normalized_advantage(mixed)
    for m in (8, 16):
        # Lowest-bound route plus random routes made exact (fixed per-group seed).
        rng = np.random.default_rng(len(upper) * 7919 + int(round(float(upper.sum()) * 1e3)) % 100003)
        chosen = np.concatenate(([selected], rng.choice(order[1:], size=min(m - 1, len(order) - 1),
                                                         replace=False)))
        mixed = upper.copy()
        mixed[chosen] = exact[chosen]
        variants[f"hybrid_sel_plus_random{m}_exact"] = normalized_advantage(mixed)
    for m in (4, 8, 16):
        # Exact for m uniformly random routes, advantages only among them; the
        # vector is the expectation over 64 draws (bias, not single-draw noise).
        rng = np.random.default_rng(m * 104729 + len(upper))
        total = np.zeros_like(exact)
        for _ in range(64):
            chosen = rng.choice(len(exact), size=min(m, len(exact)), replace=False)
            total[chosen] += normalized_advantage(exact[chosen])
        variants[f"random{m}_exact_only_expected"] = total / 64
    variants["oracle_all_exact"] = normalized_advantage(exact)
    return variants


def reward_instance_metrics(record: dict) -> dict:
    upper = np.asarray(record["bounds"], dtype=float)
    exact = np.asarray(record["exact"], dtype=float)
    selected = int(np.argmin(upper))
    variants = reward_variants(upper, exact)
    hybrid = variants["current_hybrid"]
    oracle = variants["oracle_all_exact"]
    consistent = variants["all_upper_bound"]
    positive = hybrid[hybrid > 0]
    best_exact = int(np.argmin(exact))
    row = {
        "J": int(record["J"]),
        "route_count": int(len(exact)),
        "unique_routes": int(record["unique_route_count"]),
        "selected_z_current": float(hybrid[selected]),
        "selected_z_if_upper_bound": float(consistent[selected]),
        "selected_z_oracle": float(oracle[selected]),
        "selected_is_top_advantage": float(hybrid[selected] >= hybrid.max() - 1e-9),
        "selected_share_of_positive_advantage": float(
            hybrid[selected] / positive.sum()) if positive.size else float("nan"),
        "selected_slack_percent": float(100 * (upper[selected] - exact[selected]) / exact[selected]),
        "mean_slack_percent": float(np.mean(100 * (upper - exact) / exact)),
        "spearman_upper_vs_exact": safe_spearman(upper, exact),
        "spearman_hybrid_vs_exact": safe_spearman(
            np.where(np.arange(len(upper)) == selected, exact, upper), exact),
        "best_exact_route_z_current": float(hybrid[best_exact]),
        "best_exact_route_negative_advantage": float(hybrid[best_exact] < 0),
        "exact_spread_percent": float(100 * (exact.max() - exact.min()) / exact.min()),
        "exact_mean_minus_best_percent": float(100 * (exact.mean() - exact.min()) / exact.min()),
    }
    for name, vector in variants.items():
        if name == "oracle_all_exact":
            continue
        row[f"cos_{name}"] = cosine(vector, oracle)
        nonzero = np.abs(vector) > 1e-12
        row[f"sign_agree_{name}"] = float(np.mean(
            np.sign(vector[nonzero]) == np.sign(oracle[nonzero]))) if nonzero.any() else float("nan")
    return row


def summarize_rows(rows: list[dict]) -> dict:
    out = {}
    groups = [("all", rows)] + [(str(j), [r for r in rows if r["J"] == j]) for j in SIZES]
    for label, subset in groups:
        if not subset:
            continue
        keys = [k for k in subset[0] if k not in ("J", "instance_id")]
        out[label] = {"instances": len(subset)}
        for key in keys:
            values = np.asarray([r[key] for r in subset], dtype=float)
            out[label][key] = float(np.nanmean(values))
    return out


def paired_bootstrap(left: list[dict], right: list[dict], keys: list[str],
                     resamples: int = 10000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(left)
    indices = rng.integers(0, n, size=(resamples, n))
    result = {}
    for key in keys:
        delta = np.asarray([b[key] - a[key] for a, b in zip(left, right)], dtype=float)
        valid = np.isfinite(delta)
        filled = np.where(valid, delta, 0.0)
        boot = filled[indices].sum(axis=1) / np.maximum(valid[indices].sum(axis=1), 1)
        result[key] = {
            "mean_delta_second_minus_first": float(np.nanmean(delta)),
            "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        }
    return result


def run_reward(args) -> None:
    source = args.source.resolve()
    output = args.output_dir.resolve()
    per_label = {}
    for label in args.labels:
        payload = json.loads((source / label / "results.json").read_text())
        if not payload.get("complete"):
            raise RuntimeError(f"{label} diagnostic is incomplete")
        per_label[label] = payload["records"]
    rows = {label: [reward_instance_metrics(r) for r in records]
            for label, records in per_label.items()}
    summary = {label: summarize_rows(r) for label, r in rows.items()}
    paired = None
    if len(args.labels) == 2:
        a, b = args.labels
        if [r["fingerprint"] for r in per_label[a]] != [r["fingerprint"] for r in per_label[b]]:
            raise RuntimeError("labels were not evaluated on the same instances")
        keys = [k for k in rows[a][0] if k != "J"]
        paired = {"labels": [a, b], "effects": paired_bootstrap(rows[a], rows[b], keys)}
    save_json(output / "reward_diagnostic.json", {
        "source": str(source),
        "advantage_rule": "hybrid_pomo_exact_expert: lowest-UB route exact, others UB, "
                          "per-instance centered and std-normalized",
        "variants": {
            "current_hybrid": "training rule as implemented",
            "all_upper_bound": "same normalization, every route by its upper bound",
            "topM_exact_only": "exact for the M lowest-bound routes, advantages only among them",
            "oracle_all_exact": "every route exact (reference for cosine/sign agreement)",
        },
        "summary": summary,
        "paired": paired,
        "per_instance": rows,
    })
    print(json.dumps({k: v["all"] for k, v in summary.items()}, indent=2))


# ----------------------------------------------------------------- diversity


def load_instances(directory: Path) -> list:
    from src.PseudoExpert import instance_from_dict

    manifest = json.loads((directory / "manifest.json").read_text())
    instances = []
    for size in SIZES:
        path = directory / f"J{size}.pt"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest["files_sha256"][path.name]:
            raise RuntimeError(f"diagnostic file changed: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        instances.extend(instance_from_dict(row) for row in payload["instances"])
    return instances


def directed_edges(route: list[int]) -> set[tuple[int, int]]:
    nodes = [-1] + list(route) + [-1]  # -1 is the depot
    return set(zip(nodes[:-1], nodes[1:]))


def mean_pairwise_jaccard(routes: list[list[int]]) -> float:
    edges = [directed_edges(r) for r in routes]
    values = []
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            values.append(len(edges[i] & edges[j]) / len(edges[i] | edges[j]))
    return float(np.mean(values)) if values else 1.0


def rollout_with_entropy(model, instances, device, decode_type):
    from src.TSPEnv import TSPEnv, _resolve_pomo_size

    if int(getattr(model.config, "block_size", 1)) > 1:
        raise RuntimeError("block decoding is not supported by this diagnostic")
    problem_size = instances[0].J
    pomo_size = _resolve_pomo_size(problem_size, model.config.max_pomo_size, model.config.pomo_divisor)
    env = TSPEnv(problem_size=problem_size, pomo_size=model.config.max_pomo_size,
                 pomo_divisor=model.config.pomo_divisor, use_pomo_start=True,
                 start_node_strategy=model.config.start_node_strategy)
    env.load_problems(instances, device)
    reset_state, _, _ = env.reset()
    model.pre_forward(reset_state.node_features)
    state, _, done = env.pre_step()
    entropies, max_probs = [], []
    while not done:
        before = state.selected_count
        selected, _, probs = model(state, decode_type=decode_type,
                                   use_pomo_start=True, return_all_probs=True)
        if before >= 1:  # first decision is the fixed POMO start
            p = probs.clamp_min(1e-12)
            entropies.append(-(probs * p.log()).sum(dim=-1))
            max_probs.append(probs.max(dim=-1).values)
        state, _, done = env.step(selected)
    sequences = (env.selected_node_list[:, :, 1:] - 1).cpu().tolist()
    entropy = torch.stack(entropies, dim=2).mean(dim=(1, 2)).cpu().numpy()
    max_prob = torch.stack(max_probs, dim=2).mean(dim=(1, 2)).cpu().numpy()
    if any(len(routes) != pomo_size for routes in sequences):
        raise RuntimeError("unexpected POMO size")
    return sequences, entropy, max_prob


def run_diversity(args) -> None:
    from src.CVPSolver import SolverConfig
    from src.CertifiedUpperBound import is_optimal_gurobi_solution
    from src.ImitationLearning import load_policy_checkpoint
    from src.RewardWorkerPool import RewardWorkerPool
    from src.TSPUtils import set_seed

    torch.set_num_threads(args.torch_threads)
    output = args.output_dir.resolve()
    instances = load_instances(args.data_dir.resolve())
    by_size = {j: [x for x in instances if x.J == j] for j in SIZES}
    solver_config = SolverConfig(
        solver_backend="gurobi", gurobi_threads=1, output_flag=0,
        gurobi_params={"FeasibilityTol": 1e-8, "OptimalityTol": 1e-8, "NumericFocus": 2},
        gurobi_explicit_qcp=True,
    )
    device = torch.device(args.device)
    pool = RewardWorkerPool(args.workers, env_output_flag=0)
    try:
        for item in args.checkpoint:
            label, path = item.split("=", 1)
            result_path = output / "diversity" / f"{label}.json"
            if result_path.exists() and json.loads(result_path.read_text()).get("complete"):
                print(f"SKIP {label} (complete)", flush=True)
                continue
            started = time.perf_counter()
            model, checkpoint = load_policy_checkpoint(Path(path), device=device)
            model.config = replace(model.config, max_pomo_size=max(SIZES), pomo_divisor=None,
                                   start_node_strategy="nearest_depot")
            model.eval()
            rows = []
            for size in SIZES:
                group = by_size[size]
                for start in range(0, len(group), args.batch_size):
                    batch = group[start:start + args.batch_size]
                    with torch.no_grad():
                        greedy, g_entropy, g_maxp = rollout_with_entropy(model, batch, device, "greedy")
                        set_seed(args.seed + size * 1000 + start)
                        sampled, s_entropy, _ = rollout_with_entropy(model, batch, device, "sample")
                    tasks, mapping = [], []
                    for instance, routes in zip(batch, greedy):
                        seen = {}
                        for route in routes:
                            key = tuple(int(v) for v in route)
                            if key not in seen:
                                seen[key] = len(tasks)
                                tasks.append((instance, list(key), solver_config))
                        mapping.append([seen[tuple(int(v) for v in r)] for r in routes])
                    solved = pool.map(tasks)
                    exact = []
                    for _, _, solution in solved:
                        if not is_optimal_gurobi_solution(solution) or not np.isfinite(solution.objective):
                            raise RuntimeError(f"nonoptimal exact route: {solution.status}")
                        exact.append(float(solution.objective))
                    for k, instance in enumerate(batch):
                        values = np.asarray([exact[m] for m in mapping[k]], dtype=float)
                        best = float(values.min())
                        rows.append({
                            "instance_id": instance.instance_id,
                            "J": size,
                            "greedy_unique_fraction": len(set(map(tuple, greedy[k]))) / len(greedy[k]),
                            "greedy_pairwise_edge_jaccard": mean_pairwise_jaccard(greedy[k]),
                            "sample_unique_fraction": len(set(map(tuple, sampled[k]))) / len(sampled[k]),
                            "sample_pairwise_edge_jaccard": mean_pairwise_jaccard(sampled[k]),
                            "greedy_step_entropy": float(g_entropy[k]),
                            "greedy_step_max_prob": float(g_maxp[k]),
                            "sample_step_entropy": float(s_entropy[k]),
                            "best_of_pomo": best,
                            "mean_of_pomo": float(values.mean()),
                            "median_of_pomo": float(np.median(values)),
                            "mean_minus_best_percent": float(100 * (values.mean() - best) / best),
                            "routes_within_0p5pct_of_best": float(np.mean(values <= best * 1.005)),
                        })
                    print(f"{label} J{size} {len(rows)}/{len(instances)}", flush=True)
            summary = summarize_rows(rows)
            save_json(result_path, {
                "label": label, "checkpoint": str(Path(path).resolve()),
                "epoch": int(checkpoint.get("epoch", -1)), "complete": True,
                "protocol": "greedy POMO=J nearest_depot no augmentation; one sampled POMO draw",
                "runtime_seconds": time.perf_counter() - started,
                "summary": summary, "records": rows,
            })
            s = summary["all"]
            print(f"DONE {label} best={s['best_of_pomo']:.6f} mean={s['mean_of_pomo']:.6f} "
                  f"uniq={s['greedy_unique_fraction']:.3f} jacc={s['greedy_pairwise_edge_jaccard']:.3f} "
                  f"H={s['greedy_step_entropy']:.4f}", flush=True)
    finally:
        pool.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    reward = sub.add_parser("reward")
    reward.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    reward.add_argument("--labels", nargs="+", default=["e100", "e365"])
    reward.add_argument("--output-dir", type=Path, required=True)
    diversity = sub.add_parser("diversity")
    diversity.add_argument("--checkpoint", action="append", required=True, help="LABEL=PATH")
    diversity.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    diversity.add_argument("--output-dir", type=Path, required=True)
    diversity.add_argument("--batch-size", type=int, default=20)
    diversity.add_argument("--workers", type=int, default=6)
    diversity.add_argument("--torch-threads", type=int, default=2)
    diversity.add_argument("--device", default="cpu")
    diversity.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()
    if args.command == "reward":
        run_reward(args)
    else:
        run_diversity(args)


if __name__ == "__main__":
    main()
