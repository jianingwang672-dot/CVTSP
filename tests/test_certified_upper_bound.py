from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.CertifiedUpperBound import (
    FixedSolutionUpperBound,
    _FixedUpperBoundMoveScorer,
    _materialize_move,
    apply_double_bridge,
    apply_or_opt,
    apply_relocate,
    apply_swap,
    apply_two_opt,
    find_best_certified_move_ranked,
    iter_candidate_move_indices,
    iter_candidate_sequences,
)
from src.CVPSolver import SPSolution
from src.CertifiedLocalSearch import (
    FixedAnchorLocalSearchConfig,
    run_fixed_anchor_local_search,
)
from src.TSProblemDef import CVTSPInstance


class CertifiedUpperBoundUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.instance = CVTSPInstance(
            instance_id="unit",
            path=Path("<unit>"),
            J=4,
            depot=np.array([0.0, 0.0]),
            targets=np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 1.0]]),
            carrier_speed=1.0,
            uav_speed=5.0,
            endurance=1.0,
            scale=2.0,
        )
        # All UAV tasks have zero flight time because the carrier visits each
        # target coordinate.  Arrays deliberately follow a non-identity visit
        # order to exercise the visit-position -> target-id relabeling.
        self.anchor = [2, 0, 3, 1]
        points = self.instance.targets[self.anchor]
        tseg = []
        previous = self.instance.depot
        for point in points:
            tseg.append(float(np.linalg.norm(point - previous)))
            previous = point
        tseg.append(float(np.linalg.norm(previous - self.instance.depot)))
        objective = float(sum(tseg))
        self.solution = SPSolution(
            objective=objective,
            makespan=objective,
            success=True,
            status="2",
            solve_time=0.0,
            sequence=list(self.anchor),
            takeoff_points=points.copy(),
            landing_points=points.copy(),
            t1=np.zeros(4),
            t2=np.zeros(4),
            tau=np.zeros(4),
            Tseg=np.asarray(tseg),
        )

    def _asymmetric_majorizer(self) -> FixedSolutionUpperBound:
        majorizer = FixedSolutionUpperBound.from_solution(self.instance, self.solution)
        matrix = np.random.default_rng(1).uniform(0.1, 10.0, size=(5, 5))
        np.fill_diagonal(matrix, 0.0)

        def route_cost(sequence) -> float:
            nodes = np.asarray(sequence, dtype=int) + 1
            return float(
                matrix[0, nodes[0]]
                + matrix[nodes[-1], 0]
                + matrix[nodes[:-1], nodes[1:]].sum()
            )

        anchor_cost = route_cost(majorizer.anchor_sequence)
        return replace(
            majorizer,
            fixed_flight_time=0.0,
            directed_carrier_time=matrix,
            anchor_objective=anchor_cost,
            anchor_reused_objective=anchor_cost,
        )

    def test_relabels_visit_arrays_and_reconstructs_anchor(self) -> None:
        upper = FixedSolutionUpperBound.from_solution(self.instance, self.solution)
        np.testing.assert_allclose(upper.takeoff_by_target, self.instance.targets)
        np.testing.assert_allclose(upper.landing_by_target, self.instance.targets)
        self.assertAlmostEqual(upper.upper_bound(self.anchor), self.solution.objective)
        self.assertEqual(upper.feasibility.max_violation, 0.0)

    def test_candidate_upper_bound_matches_manual_route_sum(self) -> None:
        upper = FixedSolutionUpperBound.from_solution(self.instance, self.solution)
        candidate = (0, 1, 3, 2)
        route_points = np.vstack(
            [self.instance.depot, self.instance.targets[list(candidate)], self.instance.depot]
        )
        manual = float(np.linalg.norm(np.diff(route_points, axis=0), axis=1).sum())
        self.assertAlmostEqual(upper.upper_bound(candidate), manual)

    def test_move_operators_return_valid_expected_tours(self) -> None:
        tour = (0, 1, 2, 3)
        self.assertEqual(apply_swap(tour, 0, 2), (2, 1, 0, 3))
        self.assertEqual(apply_relocate(tour, 0, 2), (1, 2, 0, 3))
        self.assertEqual(
            apply_or_opt((0, 1, 2, 3, 4), 1, 3, block_size=2),
            (0, 3, 4, 1, 2),
        )
        self.assertEqual(apply_two_opt(tour, 1, 3), (0, 3, 2, 1))
        candidates = list(iter_candidate_sequences(tour))
        self.assertTrue(candidates)
        for _, _, _, candidate in candidates:
            self.assertEqual(set(candidate), set(tour))
            self.assertEqual(len(candidate), len(tour))

    def test_double_bridge_is_a_deterministic_nontrivial_permutation(self) -> None:
        tour = tuple(range(12))
        perturbed = apply_double_bridge(tour)
        self.assertNotEqual(perturbed, tour)
        self.assertEqual(set(perturbed), set(tour))
        self.assertEqual(apply_double_bridge(tour), perturbed)

    def test_or_opt_incremental_scores_match_full_directed_route_sum(self) -> None:
        majorizer = self._asymmetric_majorizer()
        current = (0, 1, 2, 3)
        scorer = _FixedUpperBoundMoveScorer(majorizer, current)
        for kind, first, second in iter_candidate_move_indices(
            current,
            move_types=("relocate", "or_opt_2", "or_opt_3"),
        ):
            candidate = _materialize_move(current, kind, first, second)
            self.assertAlmostEqual(
                scorer.upper_bound(kind, first, second, candidate),
                majorizer.upper_bound(candidate),
                places=10,
            )

    def test_ranked_search_only_certifies_top_k(self) -> None:
        upper = FixedSolutionUpperBound.from_solution(self.instance, self.solution)

        def exact_ranker(sequence: tuple[int, ...]) -> float:
            return upper.upper_bound(sequence)

        best, generated, evaluated = find_best_certified_move_ranked(
            upper,
            self.solution.objective,
            epsilon=1e-6,
            ranker=exact_ranker,
            max_exact_candidates=1,
        )
        self.assertGreater(generated, evaluated)
        self.assertEqual(evaluated, 1)
        self.assertIsNotNone(best)
        self.assertLess(best.upper_bound, self.solution.objective)

    def test_incremental_fixed_upper_bound_matches_full_recomputation(self) -> None:
        majorizer = self._asymmetric_majorizer()
        current = (0, 1, 2, 3)
        scorer = _FixedUpperBoundMoveScorer(majorizer, current)
        for kind, first, second, candidate in iter_candidate_sequences(
            current,
            move_types=("swap", "relocate", "two_opt"),
        ):
            self.assertAlmostEqual(
                scorer.upper_bound(kind, first, second, candidate),
                majorizer.upper_bound(candidate),
                places=10,
            )

    def test_fixed_anchor_chain_is_solver_free_and_tracks_directed_two_opt_ubs(self) -> None:
        majorizer = self._asymmetric_majorizer()
        config = FixedAnchorLocalSearchConfig(
            max_local_moves=3,
            absolute_epsilon=1e-9,
            relative_epsilon=0.0,
            move_types=("two_opt",),
        )
        with patch("src.CertifiedLocalSearch.solve", side_effect=AssertionError("solver called")):
            result = run_fixed_anchor_local_search(
                majorizer,
                (0, 1, 2, 3),
                config,
            )

        self.assertEqual(result.accepted, 3)
        self.assertEqual(len(result.history), result.accepted)
        self.assertEqual(len(result.upper_bound_history), result.accepted + 1)
        self.assertTrue(
            all(
                right < left
                for left, right in zip(
                    result.upper_bound_history,
                    result.upper_bound_history[1:],
                )
            )
        )
        self.assertAlmostEqual(
            result.final_upper_bound,
            majorizer.upper_bound(result.final_sequence),
            places=10,
        )
        self.assertEqual(result.termination_reason, "max_local_moves_reached")


if __name__ == "__main__":
    unittest.main()
