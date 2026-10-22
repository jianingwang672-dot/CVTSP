"""From-scratch S500 run with the progressive J-gated Expert-search recipe.

The base configuration is the restored S500 configuration resolved by
``train_joint_self_improvement.py --print-config``; changes are listed in
docs/progressive_recipe_20261020.md. The LR follows the S500 500-epoch cosine
exactly while training stops at ``--epochs`` (default 200), so S500 is the
paired learning-rate control for every epoch run here.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ProgressiveRecipeTrainer import ProgressiveRecipeTrainer  # noqa: E402

RECIPE = {
    "recipe_j_threshold": 60,
    "recipe_ramp_start": 40,
    "recipe_ramp_end": 100,
    "recipe_lkh_max_fraction": 0.5,
    "recipe_refresh_start": 60,
    "recipe_refresh_rounds": 2,
    "recipe_il_mass_target": 1.44,
}


def resolve_config(args) -> dict:
    command = [
        sys.executable, "-B", str(ROOT / "train_joint_self_improvement.py"), "--print-config",
        "--move-types", "two_opt",
        "--epochs", str(args.epochs), "--scheduler-total-epochs", "500",
        "--validation-dir", args.validation_dir,
        "--reference-validation-seed", str(args.validation_seed),
        "--self-improvement-workers", str(args.workers),
        "--output-dir", args.output_dir,
    ]
    output = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout
    return json.loads(output[output.index("{"):])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/progressive_recipe_20261020/train")
    parser.add_argument("--validation-dir", default="validation/progressive_recipe_20261020/val1000")
    parser.add_argument("--validation-seed", type=int, default=20261020)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    config = resolve_config(args)
    trainer_params = config["trainer_params"]
    if config["optimizer_params"]["scheduler"].get("total_epochs") != 500:
        raise RuntimeError("the LR schedule must stay the S500 500-epoch cosine")
    trainer_params.update(RECIPE)
    trainer_params["self_improvement_lkh_binary"] = str(ROOT / "third_party/lkh3_20261014/LKH")
    if args.print_config:
        print(json.dumps(config, indent=2))
        return
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    trainer = ProgressiveRecipeTrainer(
        config["env_params"], config["model_params"], config["optimizer_params"], trainer_params)
    try:
        trainer.run()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
