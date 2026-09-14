from __future__ import annotations

import unittest

import numpy as np

from src.CertifiedLocalSearch import FixedAnchorLocalSearchConfig
from src.CVPSolver import SolverConfig
from src.JointSelfImprovement import (
    SparseSelfImprovementTask,
    directed_route_upper_bound,
    run_sparse_self_improvement_task,
)
from src.TSProblemDef import load_instance


class MacsimSingleInstancePipelineTest(unittest.TestCase):
    def test_beta200_top30_strict_pipeline_and_gurobi_budget(self) -> None:
        instance = load_instance("instance/Data/Example_1.txt")
        anchor = list(range(instance.J))
        rng = np.random.default_rng(20260828)
        candidates = [anchor]
        seen = {tuple(anchor)}
        while len(candidates) < 200:
            candidate = tuple(int(node) for node in rng.permutation(instance.J))
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(list(candidate))

        common = {
            "solver_backend": "gurobi",
            "gurobi_threads": 1,
            "gurobi_explicit_qcp": True,
            "output_flag": 0,
            "gurobi_params": {
                "FeasibilityTol": 1e-8,
                "OptimalityTol": 1e-8,
                "NumericFocus": 2,
            },
        }
        anchor_config = SolverConfig(**common, collect_duals=True)
        final_config = SolverConfig(**common, collect_duals=False)
        task = SparseSelfImprovementTask(
            instance=instance,
            sampled_sequences=candidates,
            anchor_sequence=anchor,
            anchor_solver_config=anchor_config,
            final_solver_config=final_config,
            local_search_config=FixedAnchorLocalSearchConfig(
                max_local_moves=30,
                absolute_epsilon=1e-6,
                relative_epsilon=1e-8,
                move_types=("swap", "two_opt"),
            ),
            route_start_top_k=30,
            upper_bound_tolerance=2e-5,
            feasibility_tolerance=1e-5,
        )
        try:
            result = run_sparse_self_improvement_task(task)
        except Exception as exc:
            if "gurobi" in str(exc).lower() or "license" in str(exc).lower():
                raise unittest.SkipTest(f"Gurobi unavailable: {exc}") from exc
            raise

        self.assertEqual(len(result.sampled_upper_bounds), 200)
        self.assertEqual(len(result.top_k_start_sequences), 30)
        anchor_u0 = directed_route_upper_bound(
            result.anchor_sequence,
            result.anchor_upper_bound_matrix,
        )
        self.assertAlmostEqual(anchor_u0, result.anchor_exact_makespan, places=5)
        self.assertEqual(result.anchor_gurobi_calls, 1)
        self.assertLessEqual(result.total_gurobi_calls, 2)
        if result.expert_changed:
            self.assertEqual(result.expert_gurobi_calls, 1)
            self.assertLess(result.expert_upper_bound, result.anchor_exact_makespan)
            self.assertLessEqual(result.expert_exact_makespan, result.expert_upper_bound + 2e-5)
            self.assertLess(result.expert_exact_makespan, result.anchor_exact_makespan)
            self.assertAlmostEqual(
                directed_route_upper_bound(
                    result.expert_sequence,
                    result.expert_upper_bound_matrix,
                ),
                result.expert_exact_makespan,
                delta=2e-5,
            )
        else:
            self.assertEqual(result.expert_gurobi_calls, 0)
            self.assertEqual(result.total_gurobi_calls, 1)
            self.assertIsNone(result.expert_upper_bound_matrix)


if __name__ == "__main__":
    unittest.main()
