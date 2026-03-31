from __future__ import annotations

import argparse
import json

from src.visualization.plot_solution import generate_plots


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot route and constraint figures from a solved instance result.")
    parser.add_argument("--instance-path", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--output-dir", default="plots")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    payload = generate_plots(
        instance_path=args.instance_path,
        result_path=args.result_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
