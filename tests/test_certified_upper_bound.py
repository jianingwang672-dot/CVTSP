from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.CertifiedUpperBound import (
    FixedSolutionUpperBound,
    SolutionFeasibility,
    _FixedUpperBoundMoveScorer,
    _cached_vectorized_move_indices,
    _materialize_move,
    _vectorized_fixed_anchor_upper_bounds,
    apply_double_bridge,
    apply_or_opt,
    apply_relocate,
    apply_swap,
    apply_two_opt,
    find_best_fixed_anchor_move,
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

    @staticmethod
    def _random_majorizer(problem_size: int, seed: int) -> FixedSolutionUpperBound:
        rng = np.random.default_rng(seed)
        matrix = rng.uniform(0.01, 10.0, size=(problem_size + 1, problem_size + 1))
        np.fill_diagonal(matrix, 0.0)
        anchor = tuple(int(node) for node in rng.permutation(problem_size))
        fixed_flight_time = float(rng.uniform(0.0, 5.0))
        nodes = np.asarray(anchor, dtype=np.int64) + 1
        anchor_cost = float(
            fixed_flight_time
            + matrix[0, nodes[0]]
            + matrix[nodes[:-1], nodes[1:]].sum()
            + matrix[nodes[-1], 0]
        )
        feasibility = SolutionFeasibility(
            max_outbound_violation=0.0,
            max_inbound_violation=0.0,
            max_sync_violation=0.0,
            max_endurance_violation=0.0,
            max_tau_equality_violation=0.0,
            max_carrier_violation=0.0,
            final_return_violation=0.0,
        )
        return FixedSolutionUpperBound(
            instance_id=f"random-{problem_size}-{seed}",
            anchor_sequence=anchor,
            anchor_objective=anchor_cost,
            takeoff_by_target=np.zeros((problem_size, 2), dtype=float),
            landing_by_target=np.zeros((problem_size, 2), dtype=float),
            t1_by_target=np.zeros(problem_size, dtype=float),
            t2_by_target=np.zeros(problem_size, dtype=float),
            tau_by_target=np.zeros(problem_size, dtype=float),
            fixed_flight_time=fixed_flight_time,
            directed_carrier_time=matrix,
            anchor_reused_objective=anchor_cost,
            feasibility=feasibility,
            feasibility_tolerance=1e-8,
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

    def test_vectorized_swap_two_opt_scores_match_every_scalar_candidate(self) -> None:
        for problem_size in (4, 20, 60, 100):
            with self.subTest(problem_size=problem_size):
                majorizer = self._random_majorizer(problem_size, seed=problem_size)
                rng = np.random.default_rng(1000 + problem_size)
                current = tuple(int(node) for node in rng.permutation(problem_size))
                scorer = _FixedUpperBoundMoveScorer(majorizer, current)
                neighborhood = _cached_vectorized_move_indices(
                    problem_size,
                    ("swap", "two_opt"),
                )
                vectorized = _vectorized_fixed_anchor_upper_bounds(
                    scorer,
                    neighborhood,
                )
                scalar = np.asarray(
                    [
                        scorer.upper_bound(kind, int(first), int(second))
                        for kind, first, second in zip(
                            neighborhood.kinds,
                            neighborhood.first,
                            neighborhood.second,
                        )
                    ],
                    dtype=float,
                )
                np.testing.assert_allclose(
                    vectorized,
                    scalar,
                    rtol=1e-13,
                    atol=1e-11,
                )

    def test_vectorized_best_move_matches_scalar_scan_and_counts(self) -> None:
        move_configurations = (("swap",), ("two_opt",), ("swap", "two_opt"))
        for problem_size in (4, 20, 60, 100):
            majorizer = self._random_majorizer(problem_size, seed=2000 + problem_size)
            rng = np.random.default_rng(3000 + problem_size)
            for move_types in move_configurations:
                with self.subTest(problem_size=problem_size, move_types=move_types):
                    current = tuple(int(node) for node in rng.permutation(problem_size))
                    scalar = find_best_fixed_anchor_move(
                        majorizer,
                        current,
                        epsilon=1e-9,
                        move_types=move_types,
                        use_vectorized=False,
                    )
                    vectorized = find_best_fixed_anchor_move(
                        majorizer,
                        current,
                        epsilon=1e-9,
                        move_types=move_types,
                        use_vectorized=True,
                    )
                    self.assertEqual(vectorized[1:], scalar[1:])
                    scalar_move, vectorized_move = scalar[0], vectorized[0]
                    self.assertEqual(vectorized_move is None, scalar_move is None)
                    if scalar_move is not None and vectorized_move is not None:
                        self.assertEqual(vectorized_move.kind, scalar_move.kind)
                        self.assertEqual(vectorized_move.first, scalar_move.first)
                        self.assertEqual(vectorized_move.second, scalar_move.second)
                        self.assertEqual(vectorized_move.sequence, scalar_move.sequence)
                        self.assertAlmostEqual(
                            vectorized_move.upper_bound,
                            scalar_move.upper_bound,
                            places=12,
                        )

    def test_vectorized_fixed_anchor_chain_matches_scalar_history(self) -> None:
        majorizer = self._random_majorizer(40, seed=4040)
        current = tuple(reversed(range(40)))
        config = FixedAnchorLocalSearchConfig(
            max_local_moves=12,
            absolute_epsilon=1e-9,
            relative_epsilon=0.0,
            move_types=("swap", "two_opt"),
        )
        with patch(
            "src.CertifiedUpperBound.VECTORIZED_LOCAL_SEARCH_ENABLED",
            False,
        ):
            scalar = run_fixed_anchor_local_search(majorizer, current, config)
        with patch(
            "src.CertifiedUpperBound.VECTORIZED_LOCAL_SEARCH_ENABLED",
            True,
        ):
            vectorized = run_fixed_anchor_local_search(majorizer, current, config)

        self.assertEqual(vectorized.final_sequence, scalar.final_sequence)
        self.assertEqual(vectorized.accepted, scalar.accepted)
        self.assertEqual(vectorized.generated, scalar.generated)
        self.assertEqual(vectorized.evaluated, scalar.evaluated)
        self.assertEqual(
            [item.move.to_dict() for item in vectorized.history],
            [item.move.to_dict() for item in scalar.history],
        )
        self.assertAlmostEqual(
            vectorized.final_upper_bound,
            scalar.final_upper_bound,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
