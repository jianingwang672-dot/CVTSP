from __future__ import annotations

import argparse
import json

from src.common.logging_utils import setup_logger
from src.common.seed import set_seed
from src.data.synthetic import SyntheticGeneratorConfig, generate_synthetic_dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a real-mimicking synthetic CVTSP training dataset.")
    parser.add_argument("--real-dataset-dir", default="instance/Data")
    parser.add_argument("--output-dir", default="instance/SyntheticTrain_20_100")
    parser.add_argument("--num-instances", type=int, default=2100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--target-counts",
        type=int,
        nargs="*",
        default=[20, 30, 40, 50, 60, 80, 100],
        help="Target-count buckets to sample during generation.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    setup_logger(args.output_dir)
    set_seed(args.seed)

    config = SyntheticGeneratorConfig(
        real_dataset_dir=args.real_dataset_dir,
        output_dir=args.output_dir,
        num_instances=args.num_instances,
        seed=args.seed,
        target_counts=tuple(args.target_counts),
        min_target_count=min(args.target_counts) if args.target_counts else 20,
        max_target_count=max(args.target_counts) if args.target_counts else 100,
    )
    manifest = generate_synthetic_dataset(config)
    print(
        json.dumps(
            {
                "output_dir": manifest["output_dir"],
                "num_instances": manifest["num_instances"],
                "j_distribution": manifest["j_distribution"],
                "manifest_path": f"{manifest['output_dir']}/manifest.json",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
