from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.CVPSolver import SPSolution, SolverConfig
from src.ReferencePolicy import (
    ReferenceState,
    clone_reference_policy,
    evaluate_policy_pair_exact,
    promote_reference_if_better,
    promote_reference_with_size_gate,
)
from src.TSProblemDef import CVTSPInstance
from src.TSPModel import ModelConfig, TSPModel


def _model() -> TSPModel:
    return TSPModel(
        ModelConfig(
            embedding_dim=16,
            encoder_layer_num=1,
            qkv_dim=4,
            head_num=2,
            ff_hidden_dim=32,
            max_pomo_size=2,
        )
    )


def _instance(instance_id: str, shift: float) -> CVTSPInstance:
    return CVTSPInstance(
        instance_id=instance_id,
        path=Path(f"<{instance_id}>"),
        J=3,
        depot=np.asarray([0.0, 0.0], dtype=np.float32),
        targets=np.asarray(
            [[1.0 + shift, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32
        ),
        carrier_speed=1.0,
        uav_speed=2.0,
        endurance=2.0,
        scale=1.0,
    )


class ReferencePolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        torch.set_num_threads(1)

    def test_reference_state_round_trip_and_backward_compatible_defaults(self) -> None:
        state = ReferenceState(
            validation_rounds=3,
            promotions=1,
            last_validation_epoch=9,
            last_promoted_epoch=6,
            validation_gurobi_calls=24,
            last_metrics={"target_mean_objective": 4.0},
            history=[{"epoch": 9, "promoted": False}],
        )
        restored = ReferenceState.from_dict(state.to_dict())
        self.assertEqual(restored, state)
        restored.history[0]["promoted"] = True
        self.assertFalse(state.history[0]["promoted"])

        older = ReferenceState.from_dict({"validation_rounds": 2})
        self.assertEqual(older.validation_rounds, 2)
        self.assertEqual(older.promotions, 0)
        self.assertEqual(older.history, [])

    def test_clone_and_promote_are_independent_and_frozen(self) -> None:
        target = _model()
        reference = clone_reference_policy(target, device="cpu")
        self.assertIsNot(reference, target)
        self.assertFalse(reference.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in reference.parameters()))

        with torch.no_grad():
            next(target.parameters()).add_(1.0)
        self.assertFalse(
            torch.equal(next(reference.parameters()), next(target.parameters()))
        )
        old_reference = {key: value.clone() for key, value in reference.state_dict().items()}
        not_promoted = promote_reference_if_better(
            reference,
            target,
            {"reference_mean_objective": 10.0, "target_mean_objective": 9.95},
            epsilon=0.1,
        )
        self.assertFalse(not_promoted)
        self.assertTrue(
            all(torch.equal(old_reference[key], value) for key, value in reference.state_dict().items())
        )

        promoted = promote_reference_if_better(
            reference,
            target,
            {"reference_mean_objective": 10.0, "target_mean_objective": 9.8},
            epsilon=0.1,
        )
        self.assertTrue(promoted)
        self.assertTrue(
            all(
                torch.equal(reference.state_dict()[key], value)
                for key, value in target.state_dict().items()
            )
        )
        self.assertFalse(reference.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in reference.parameters()))

    def test_size_gate_requires_overall_gain_and_no_large_size_regression(self) -> None:
        target = _model()
        reference = clone_reference_policy(target)
        with torch.no_grad():
            next(target.parameters()).add_(0.25)
        blocked = promote_reference_with_size_gate(
            reference,
            target,
            {
                "delta_all": 0.01,
                "group_metrics": {
                    "80": {"delta": -1e-4},
                    "100": {"delta": 0.02},
                },
            },
        )
        self.assertFalse(blocked)
        promoted = promote_reference_with_size_gate(
            reference,
            target,
            {
                "delta_all": 0.01,
                "group_metrics": {
                    "80": {"delta": 0.0},
                    "100": {"delta": 0.02},
                },
            },
        )
        self.assertTrue(promoted)
        self.assertFalse(reference.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in reference.parameters()))

    def test_exact_pair_evaluation_solves_every_greedy_candidate(self) -> None:
        reference = clone_reference_policy(_model())
        target = _model()
        target.train()
        instances = [_instance("a", 0.0), _instance("b", 0.1)]

        def fake_candidates(model, instance, **kwargs):
            self.assertEqual(kwargs["decode_type"], "greedy")
            self.assertEqual(kwargs["max_candidates"], 2)
            if model is reference:
                return [[0, 1, 2], [1, 0, 2]]
            return [[2, 1, 0], [2, 0, 1]]

        objectives = {
            "a": {
                (0, 1, 2): 10.0,
                (1, 0, 2): 11.0,
                (2, 1, 0): 9.0,
                (2, 0, 1): 12.0,
            },
            "b": {
                (0, 1, 2): 8.0,
                (1, 0, 2): 9.0,
                (2, 1, 0): 9.0,
                (2, 0, 1): 10.0,
            },
        }

        def fake_solve(instance, sequence, config):
            objective = objectives[instance.instance_id][tuple(sequence)]
            return SPSolution(
                objective=objective,
                makespan=objective,
                success=True,
                status="2",
                solve_time=0.01,
                sequence=list(sequence),
            )

        with (
            patch("src.ReferencePolicy.generate_policy_candidates", side_effect=fake_candidates),
            patch("src.ReferencePolicy.solve", side_effect=fake_solve) as mocked_solve,
        ):
            metrics = evaluate_policy_pair_exact(
                reference,
                target,
                instances,
                torch.device("cpu"),
                SolverConfig(solver_backend="gurobi"),
                max_candidates=2,
            )

        self.assertEqual(mocked_solve.call_count, 8)
        self.assertEqual(metrics["validation_gurobi_calls"], 8)
        self.assertAlmostEqual(metrics["reference_mean_objective"], 9.0)
        self.assertAlmostEqual(metrics["target_mean_objective"], 9.0)
        self.assertEqual(metrics["target_wins"], 1)
        self.assertEqual(metrics["ties"], 0)
        self.assertEqual(metrics["target_losses"], 1)
        self.assertEqual(len(metrics["details"]), 2)
        self.assertTrue(target.training)
        self.assertFalse(reference.training)


if __name__ == "__main__":
    unittest.main()
