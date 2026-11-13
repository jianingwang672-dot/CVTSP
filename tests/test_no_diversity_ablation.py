from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import src.JointSelfImprovement as joint
from src.CertifiedLocalSearch import FixedAnchorLocalSearchConfig
from src.CVPSolver import SolverConfig
from src.PseudoExpert import instance_from_dict

ROOT = Path(__file__).resolve().parents[1]


class NoDiversityBudgetTests(unittest.TestCase):
    def _search(self, **task_fields):
        rows = torch.load(ROOT / "validation/e500_logic_e100_20261010/test500/J20.pt", weights_only=False)
        instance = instance_from_dict(rows["instances"][0])
        route = list(range(instance.J))
        config = SolverConfig(gurobi_threads=1, gurobi_explicit_qcp=True, collect_duals=True)
        with tempfile.TemporaryDirectory() as work_root:
            task = joint.SparseSelfImprovementTask(
                instance=instance, sampled_sequences=[route], anchor_sequence=route,
                anchor_solver_config=config, final_solver_config=config,
                local_search_config=FixedAnchorLocalSearchConfig(max_local_moves=900, move_types=("two_opt",)),
                route_start_top_k=30, lkh_start_fraction=0.5, lkh_work_root=work_root,
                lkh_binary=str(ROOT / "third_party/lkh3_20261014/LKH"), **task_fields)
            with patch.object(joint, "_run_lkh_fixed_anchor_searches",
                              wraps=joint._run_lkh_fixed_anchor_searches) as lkh:
                return joint.run_sparse_self_improvement_task(task), lkh

    def test_single_route_gets_lkh_with_the_main_run_budget_and_2opt(self):
        result, lkh = self._search(lkh_budget_starts=30)
        self.assertEqual(lkh.call_count, 1)
        self.assertAlmostEqual(lkh.call_args.kwargs["task"].lkh_budget_coefficient, 1.5e-4 * 15 / 30)
        self.assertEqual(len(result.local_search_results), 2)

    def test_default_rule_is_unchanged(self):
        result, lkh = self._search()
        self.assertEqual(lkh.call_count, 0)  # round(0.5 * 1) == 0
        self.assertEqual(len(result.local_search_results), 1)

    def test_config_gives_single_route_the_30_start_move_budget(self):
        out = subprocess.run(
            [sys.executable, "-B", "scripts/train_ablation.py", "no_diversity", "--print-config"],
            cwd=ROOT, capture_output=True, text=True, check=True, env={**os.environ, "PYTHONPATH": "."})
        params = json.loads(out.stdout[out.stdout.index("{"):])["trainer_params"]
        self.assertEqual(params["self_improvement_max_local_moves"], 900)
        self.assertEqual(params["reference_candidate_count"], 1)


if __name__ == "__main__":
    unittest.main()
