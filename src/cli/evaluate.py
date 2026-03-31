from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from src.common.config import AppConfig, load_config
from src.common.logging_utils import setup_logger
from src.common.seed import set_seed
from src.data.dataset import load_dataset
from src.data.instance import load_instance
from src.pipeline.evaluator import RewardCache, evaluate_dataset, evaluate_instance
from src.pipeline.inference import load_model_for_inference


def _compact_solution_for_export(solution: dict) -> dict:
    return {
        "status": solution.get("status"),
        "solve_time": solution.get("solve_time"),
        "takeoff_points": solution.get("takeoff_points"),
        "landing_points": solution.get("landing_points"),
        "t1": solution.get("t1"),
        "t2": solution.get("t2"),
        "Tij": solution.get("Tij"),
    }


def _write_best_result_bundle(output_dir: Path, results: list[dict], summary: dict, split_name: str) -> None:
    bundle_dir = output_dir / "best_results"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for result in results:
        best_candidate = min(
            result["candidates"],
            key=lambda candidate: float(candidate["solution"]["objective"]),
        )
        payload = {
            "instance_id": result["instance_id"],
            "success": result["success"],
            "objective": result["best_objective"],
            "final_route": best_candidate["solution"].get("route_with_depot"),
            "solution": _compact_solution_for_export(best_candidate["solution"]),
        }
        (bundle_dir / f"{result['instance_id']}.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        summary_rows.append(
            {
                "instance_id": result["instance_id"],
                "success": result["success"],
                "best_objective": result["best_objective"],
                "best_status": result["best_status"],
                "best_sequence": " ".join(map(str, result["best_sequence"])),
                "best_route_with_depot": " ".join(map(str, best_candidate["solution"].get("route_with_depot", []))),
            }
        )

    summary_path = output_dir / f"{split_name}_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["instance_id"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    (output_dir / f"{split_name}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a split or a single instance.")
    parser.add_argument("--config", default="src/configs/default.yaml")
    parser.add_argument("--split")
    parser.add_argument("--instance-path")
    parser.add_argument("--baseline", choices=["input_order", "nearest_neighbor", "random_k"])
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--decode-type")
    parser.add_argument("--num-candidates", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--log-level")
    return parser


def _apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.split:
        config.data.eval_split = args.split
    if args.checkpoint_path:
        config.runtime.checkpoint_path = args.checkpoint_path
    if args.output_dir:
        config.runtime.output_dir = args.output_dir
    if args.decode_type:
        config.decode.decode_type = args.decode_type
    if args.num_candidates is not None:
        config.decode.num_candidates = args.num_candidates
    if args.seed is not None:
        config.runtime.seed = args.seed
    if args.device:
        config.runtime.device = args.device
    if args.log_level:
        config.runtime.log_level = args.log_level
    return config


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = _apply_overrides(load_config(args.config), args)
    setup_logger(config.runtime.output_dir, config.runtime.log_level)
    set_seed(config.runtime.seed)

    output_dir = Path(config.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = RewardCache()

    if args.instance_path:
        instance = load_instance(args.instance_path)
        if args.baseline:
            result = evaluate_instance(instance, config=config, baseline_name=args.baseline, cache=cache)
        else:
            device = torch.device(config.runtime.device)
            model, _ = load_model_for_inference(config, device)
            with torch.no_grad():
                result = evaluate_instance(instance, config=config, model=model, device=device, cache=cache)
        payload = result.to_dict()
        (output_dir / f"evaluate_{instance.instance_id}.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "instance_id": payload["instance_id"],
                    "success": payload["success"],
                    "best_objective": payload["best_objective"],
                    "best_sequence": payload["best_sequence"],
                    "num_candidates": payload["num_candidates"],
                    "details_path": str(output_dir / f"evaluate_{instance.instance_id}.json"),
                },
                indent=2,
            )
        )
        return

    instances = load_dataset(
        config.data.dataset_dir,
        config.data.eval_split,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        split_seed=config.data.split_seed,
    )
    if args.baseline:
        results, summary = evaluate_dataset(instances, config=config, baseline_name=args.baseline, cache=cache)
    else:
        device = torch.device(config.runtime.device)
        model, _ = load_model_for_inference(config, device)
        with torch.no_grad():
            results, summary = evaluate_dataset(instances, config=config, model=model, device=device, cache=cache)

    payload = {"summary": summary, "results": [result.to_dict() for result in results]}
    (output_dir / f"evaluate_{config.data.eval_split}.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    _write_best_result_bundle(output_dir, payload["results"], payload["summary"], config.data.eval_split)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
