from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common.config import AppConfig, load_config
from src.common.logging_utils import setup_logger
from src.common.seed import set_seed
from src.pipeline.trainer import run_training


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the CVTSP master policy.")
    parser.add_argument("--config", default="src/configs/default.yaml")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-batches-per-epoch", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--log-level")
    return parser


def _apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.dataset_dir:
        config.data.dataset_dir = args.dataset_dir
    if args.output_dir:
        config.runtime.output_dir = args.output_dir
    if args.checkpoint_path:
        config.runtime.checkpoint_path = args.checkpoint_path
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.max_batches_per_epoch is not None:
        config.train.max_batches_per_epoch = args.max_batches_per_epoch
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

    summary = run_training(config)
    summary_path = Path(config.runtime.output_dir) / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
