"""From-scratch component ablations of the progressive-recipe run (E133).

Each variant reuses the exact configuration of scripts/train_progressive_recipe.py
(S500 500-epoch cosine LR, seed, validation set, 32 workers) and switches off
one component; all variants stop at E100 and are compared with the main run's
own checkpoint-100.pt on validation2000.

  no_il            no imitation update: the epoch's Expert routes are still scored
                   with teacher forcing (no gradient) so the IL-competence signal
                   that schedules the exact-Expert weight and near-miss rate keeps
                   reflecting how well the policy reproduces the Experts
  no_exact_expert  exact-Expert RL term off (exact_expert_aux_max_weight = 0); the
                   Target exact solves still run, so the Gurobi budget is unchanged
  no_search        no Expert search: no 2-opt moves, no LKH starts, no C0 refresh
                   (the Expert is the policy's own best route under the UB table)
  no_recipe        no progressive J-gated recipe: every batch takes the small-J
                   branch (2-opt only, adaptive near-miss, no refresh) and the
                   changed-Expert IL weight stays at its plain value of 5
  no_diversity     no candidate diversity in self-improvement (1028 meeting
                   design D): instead of 200 sampled POMO Reference routes, the
                   Expert pool is the policy's single greedy route from the
                   first (nearest-depot) start, so local search has one start;
                   that start keeps the total budget of the main run's 30 starts
                   (30 x 30 = 900 2-opt moves, plus LKH with the main run's
                   start fraction and time budget); the RL rollout is unchanged
  exact_search     no shadow price, real Gurobi (1028 meeting design B): without
                   the C0 table every searched route is solved exactly, so the
                   search is cut to one start (the anchor) and first-improvement
                   2-opt with at most 100 exact solves per instance
                   (src/ExactSearchAblation.py); LKH starts and C0 refresh are
                   off because both exist only to exploit the table
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.ImitationLearning import teacher_forcing_loss  # noqa: E402
from src.ProgressiveRecipeTrainer import ProgressiveRecipeTrainer  # noqa: E402
from src.TSPEnv import generate_candidate_pool_batch  # noqa: E402
from train_progressive_recipe import RECIPE, resolve_config  # noqa: E402

NOMINAL_STARTS = 30  # route_start_top_k of the main run

VARIANTS = {
    "no_il": {},
    "no_exact_expert": {"exact_expert_aux_max_weight": 0.0},
    "no_search": {"self_improvement_max_local_moves": 0, "recipe_lkh_max_fraction": 0.0,
                  "recipe_refresh_rounds": 0},
    # IL weight = clip((T - (1 - f)) / f, 1, 5) equals 5 for any changed fraction f once T >= 5
    "no_recipe": {"recipe_j_threshold": 10**9, "recipe_il_mass_target": 5.0},
    "no_diversity": {"reference_candidate_count": 1, "self_improvement_route_start_diversity_count": 0,
                     "self_improvement_max_local_moves": NOMINAL_STARTS * 30},
    "exact_search": {"recipe_lkh_max_fraction": 0.0, "recipe_refresh_rounds": 0},
}


class NoImitationTrainer(ProgressiveRecipeTrainer):
    """Measures Expert token accuracy without taking any IL optimizer step."""

    def _imitation_update(self, instances, expert_sequences, expert_changed_flags):
        groups: dict[int, list[int]] = {}
        for index, instance in enumerate(instances):
            groups.setdefault(int(instance.J), []).append(index)
        micro = int(getattr(self, "imitation_micro_batch_size", self.imitation_batch_size))
        weighted_loss, correct, tokens = 0.0, 0, 0
        with torch.no_grad():
            for indices in groups.values():
                for start in range(0, len(indices), micro):
                    batch = indices[start:start + micro]
                    loss, token_correct, _, token_count = teacher_forcing_loss(
                        self.model, [instances[i] for i in batch], [expert_sequences[i] for i in batch],
                        device=self.device)
                    weighted_loss += float(loss.item()) * token_count
                    correct += token_correct
                    tokens += token_count
        return weighted_loss / max(tokens, 1), correct / max(tokens, 1), 0, 0


class NoDiversityTrainer(ProgressiveRecipeTrainer):
    """One greedy Reference route per instance instead of a sampled pool."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_lkh_budget_starts = NOMINAL_STARTS

    def _generate_reference_candidates(self, instances):
        with torch.inference_mode():
            return generate_candidate_pool_batch(self.reference_policy, instances, device=self.device,
                                                 num_candidates=1, decode_type="greedy")


TRAINERS = {"no_il": NoImitationTrainer, "no_diversity": NoDiversityTrainer}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=sorted(VARIANTS))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--validation-dir", default="validation/progressive_recipe_20261020/val1000")
    parser.add_argument("--validation-seed", type=int, default=20261020)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--train-episodes", type=int, default=None, help="smoke tests only")
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir or f"outputs/ablation_20261028/{args.variant}/train"
    config = resolve_config(args)
    trainer_params = config["trainer_params"]
    if config["optimizer_params"]["scheduler"].get("total_epochs") != 500:
        raise RuntimeError("the LR schedule must stay the S500 500-epoch cosine")
    trainer_params.update(RECIPE)
    trainer_params["self_improvement_lkh_binary"] = str(ROOT / "third_party/lkh3_20261014/LKH")
    trainer_params.update(VARIANTS[args.variant])
    trainer_params["ablation_variant"] = args.variant
    if args.train_episodes:
        trainer_params["train_episodes"] = args.train_episodes
    if args.print_config:
        print(json.dumps(config, indent=2))
        return
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.variant == "exact_search":
        import src.JointSelfImprovementTrainer as joint_trainer
        from src.ExactSearchAblation import run_exact_self_improvement_task
        joint_trainer.run_sparse_self_improvement_task = run_exact_self_improvement_task
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("CVTSP_EXACT_BUDGET", "100")
        os.environ.setdefault("CVTSP_EXACT_PROGRESS", str(Path(args.output_dir) / "exact_progress.jsonl"))
    cls = TRAINERS.get(args.variant, ProgressiveRecipeTrainer)
    trainer = cls(config["env_params"], config["model_params"], config["optimizer_params"], trainer_params)
    try:
        trainer.run()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
