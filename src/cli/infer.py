from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common.config import AppConfig, load_config
from src.common.logging_utils import setup_logger
from src.common.seed import set_seed
from src.pipeline.inference import run_inference


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run single-instance DRL + SP inference.")
    parser.add_argument("--config", default="src/configs/default.yaml")
    parser.add_argument("--instance-path", required=True)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--decode-type")
    parser.add_argument("--num-candidates", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--log-level")
    return parser


def _apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
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

    instance, result = run_inference(config, args.instance_path)
    output_path = Path(config.runtime.output_dir) / f"infer_{instance.instance_id}.json"
    payload = result.to_dict()
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "instance_id": payload["instance_id"],
                "success": payload["success"],
                "best_objective": payload["best_objective"],
                "best_sequence": payload["best_sequence"],
                "best_route_with_depot": payload.get("best_route_with_depot"),
                "num_candidates": payload["num_candidates"],
                "details_path": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
