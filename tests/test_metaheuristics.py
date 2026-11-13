from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

import numpy as np

from src.CVPSolver import SolverConfig
from src.Metaheuristics import (
    ExactEvaluator, double_bridge, euclidean_tour, granular_local_search, nearest_targets,
    or_opt_move, random_move, relocate_move, run_metaheuristic, swap_move, two_opt_move,
)


def _instance(n=12, seed=0):
    rng = np.random.default_rng(seed)
    return SimpleNamespace(instance_id=f"toy{n}", J=n, depot=np.array([50.0, 50.0]),
                           targets=rng.uniform(0, 100, size=(n, 2)))


def _tour_length(instance):
    points = np.vstack([instance.depot[None, :], instance.targets])

    def length(seq):
        route = [0] + [s + 1 for s in seq] + [0]
        return float(sum(np.linalg.norm(points[a] - points[b]) for a, b in zip(route, route[1:])))
    return length


class MoveTest(unittest.TestCase):
    def test_moves_are_permutations(self) -> None:
        rng = np.random.default_rng(1)
        seq = list(range(10))
        self.assertEqual(two_opt_move(seq, 2, 5), [0, 1, 5, 4, 3, 2, 6, 7, 8, 9])
        self.assertEqual(relocate_move(seq, 1, 4), [0, 2, 3, 4, 1, 5, 6, 7, 8, 9])
        self.assertEqual(swap_move(seq, 0, 9)[::9], [9, 0])
        self.assertEqual(or_opt_move(seq, 0, 2, 3), [2, 3, 4, 0, 1, 5, 6, 7, 8, 9])
        for _ in range(200):
            self.assertEqual(sorted(random_move(seq, rng)), seq)
            self.assertEqual(sorted(double_bridge(seq, rng)), seq)


class EvaluatorTest(unittest.TestCase):
    def test_cache_is_orientation_free_and_trace_records_improvements(self) -> None:
        inst = _instance()
        calls = []
        evaluate = ExactEvaluator(inst, SolverConfig(), max_evaluations=10,
                                  objective=lambda s: (calls.append(s), _tour_length(inst)(s))[1])
        seq = list(range(12))
        evaluate(seq)
        evaluate(seq[::-1])
        self.assertEqual(len(calls), 1)
        self.assertEqual(evaluate.cache_hits, 1)
        self.assertEqual(len(evaluate.trace), 1)

    def test_evaluation_budget_is_respected(self) -> None:
        inst = _instance(20)
        result = run_metaheuristic("sa", inst, SolverConfig(), max_evaluations=50, objective=_tour_length(inst))
        self.assertEqual(result.evaluations, 50)
        self.assertEqual(sorted(result.best_sequence), list(range(20)))

    def test_tiny_instance_terminates(self) -> None:
        inst = _instance(3)
        for method in ("sa", "vns", "ils"):
            result = run_metaheuristic(method, inst, SolverConfig(), max_evaluations=100,
                                       objective=_tour_length(inst))
            self.assertLessEqual(result.evaluations, 3)


class SearchTest(unittest.TestCase):
    def test_methods_improve_a_scrambled_start(self) -> None:
        inst = _instance(25, seed=3)
        length = _tour_length(inst)
        scrambled = list(np.random.default_rng(0).permutation(25))
        evaluate = ExactEvaluator(inst, SolverConfig(), max_evaluations=5000, objective=length)
        seq, value = granular_local_search(scrambled, evaluate(scrambled), evaluate, nearest_targets(inst, 8),
                                           np.random.default_rng(0))
        self.assertLess(value, length(scrambled))
        start = length(euclidean_tour(inst))
        for method in ("sa", "vns", "ils"):
            result = run_metaheuristic(method, inst, SolverConfig(), max_evaluations=2000, seed=1,
                                       objective=length)
            self.assertLessEqual(result.best_makespan, start + 1e-9)
            self.assertAlmostEqual(result.best_makespan, length(result.best_sequence))
            self.assertEqual(result.trace[-1][2], result.best_makespan)


@unittest.skipUnless(os.environ.get("GRB_LICENSE_FILE"), "needs a Gurobi licence")
class ExactSocpTest(unittest.TestCase):
    def test_reverse_route_has_the_same_makespan(self) -> None:
        from src.TSProblemDef import load_dataset
        instance = next(x for x in load_dataset("instance/Data", "all") if x.instance_id == "Example_260")
        solver = SolverConfig(solver_backend="gurobi", gurobi_threads=1, output_flag=0)
        seq = [12, 5, 6, 14, 16, 19, 13, 3, 1, 0, 9, 8, 11, 15, 2, 4, 10, 18, 7, 17]
        a = ExactEvaluator(instance, solver, max_evaluations=2)
        b = ExactEvaluator(instance, solver, max_evaluations=2)
        self.assertAlmostEqual(a(seq), 266.5118, places=3)
        b.cache.clear()
        self.assertAlmostEqual(b(seq[::-1]), a(seq), places=4)
        result = run_metaheuristic("ils", instance, solver, max_evaluations=40, seed=0)
        self.assertLessEqual(result.best_makespan, result.initial_makespan)


if __name__ == "__main__":
    unittest.main()
