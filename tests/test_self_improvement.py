from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.ImitationLearning import (
    ImitationTrainingConfig,
    evaluate_imitation,
    train_imitation_student,
)
from src.PseudoExpert import (
    PseudoExpertGenerationConfig,
    PseudoExpertRecord,
    generate_pseudo_expert_record,
    instance_from_dict,
    instance_to_dict,
    load_pseudo_expert_dataset,
    save_pseudo_expert_dataset,
)
from src.CVPSolver import SPSolution, SolverConfig
from src.EdgeCostPredictor import make_matrix_ranker
from src.TSPModel import ModelConfig, TSPModel
from src.TSProblemDef import CVTSPInstance


def _instance(instance_id: str, shift: float = 0.0) -> CVTSPInstance:
    targets = np.asarray(
        [[0.1 + shift, 0.2], [0.8, 0.1 + shift], [0.7, 0.9], [0.2, 0.8]],
        dtype=np.float32,
    )
    depot = np.asarray([0.5, 0.5], dtype=np.float32)
    return CVTSPInstance(
        instance_id=instance_id,
        path=Path(f"<{instance_id}>"),
        J=4,
        depot=depot,
        targets=targets,
        carrier_speed=0.2,
        uav_speed=0.8,
        endurance=0.5,
        scale=1.0,
    )


def _record(instance: CVTSPInstance, sequence: list[int], split: str) -> PseudoExpertRecord:
    return PseudoExpertRecord(
        split=split,
        instance=instance_to_dict(instance),
        teacher_candidates=[
            {
                "candidate_index": 0,
                "initial_sequence": [0, 1, 2, 3],
                "initial_objective": 10.0,
                "improved_sequence": sequence,
                "improved_objective": 8.0,
                "objective_improvement": 2.0,
                "evaluated_moves": 1,
                "accepted_moves": 1,
                "gurobi_calls": 2,
                "termination_reason": "test",
                "history": [],
            }
        ],
        teacher_best_sequence=[0, 1, 2, 3],
        teacher_best_objective=10.0,
        expert_sequence=sequence,
        expert_objective=8.0,
        expert_source_candidate=0,
        expert_improvement_over_teacher=2.0,
        evaluated_moves=1,
        accepted_moves=1,
        gurobi_calls=2,
        runtime=0.01,
    )


class SelfImprovementTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.model = TSPModel(
            ModelConfig(
                embedding_dim=16,
                encoder_layer_num=1,
                qkv_dim=4,
                head_num=2,
                ff_hidden_dim=32,
                max_pomo_size=4,
            )
        )

    def test_instance_and_dataset_round_trip(self) -> None:
        instance = _instance("roundtrip")
        restored = instance_from_dict(instance_to_dict(instance))
        np.testing.assert_allclose(restored.depot, instance.depot)
        np.testing.assert_allclose(restored.targets, instance.targets)
        self.assertEqual(restored.J, instance.J)

        record = _record(instance, [2, 0, 3, 1], "train")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experts.json"
            payload = save_pseudo_expert_dataset(path, [record], metadata={"test": True})
            loaded, raw = load_pseudo_expert_dataset(path)
        self.assertEqual(payload["summary"]["records"], 1)
        self.assertTrue(raw["metadata"]["test"])
        self.assertEqual(loaded[0].expert_sequence, record.expert_sequence)

    def test_teacher_forcing_cross_entropy_learns_expert_sequence(self) -> None:
        train_records = [
            _record(_instance("train_1", 0.00), [2, 3, 0, 1], "train"),
            _record(_instance("train_2", 0.02), [2, 3, 0, 1], "train"),
            _record(_instance("train_3", 0.04), [2, 3, 0, 1], "train"),
            _record(_instance("train_4", 0.06), [2, 3, 0, 1], "train"),
        ]
        validation = [_record(_instance("validation", 0.03), [2, 3, 0, 1], "validation")]
        initial = evaluate_imitation(
            self.model,
            train_records,
            device=torch.device("cpu"),
            batch_size=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            teacher_path = Path(directory) / "teacher.pt"
            torch.save({"model_state_dict": self.model.state_dict()}, teacher_path)
            summary = train_imitation_student(
                self.model,
                train_records,
                validation,
                device=torch.device("cpu"),
                output_dir=Path(directory) / "student",
                source_checkpoint=teacher_path,
                config=ImitationTrainingConfig(
                    epochs=20,
                    batch_size=4,
                    learning_rate=2e-3,
                    weight_decay=0.0,
                    grad_clip=1.0,
                    seed=7,
                ),
            )
            self.assertTrue((Path(directory) / "student" / "best.pt").exists())
        final = evaluate_imitation(
            self.model,
            train_records,
            device=torch.device("cpu"),
            batch_size=4,
        )
        self.assertLess(final.nll, initial.nll)
        self.assertGreater(final.token_accuracy, initial.token_accuracy)
        self.assertGreater(summary["best_epoch"], 0)

    def test_pseudo_expert_uses_ranker_only_for_top_k_screening(self) -> None:
        instance = _instance("guided_record")

        def fake_solve(local_instance, sequence, _config):
            points = local_instance.targets[np.asarray(sequence, dtype=int)]
            previous = local_instance.depot
            segments = []
            for point in points:
                segments.append(float(np.linalg.norm(point - previous) / local_instance.carrier_speed))
                previous = point
            segments.append(
                float(np.linalg.norm(previous - local_instance.depot) / local_instance.carrier_speed)
            )
            objective = float(sum(segments))
            return SPSolution(
                objective=objective,
                makespan=objective,
                success=True,
                status="2",
                solve_time=0.0,
                sequence=list(sequence),
                takeoff_points=points.copy(),
                landing_points=points.copy(),
                t1=np.zeros(local_instance.J),
                t2=np.zeros(local_instance.J),
                tau=np.zeros(local_instance.J),
                Tseg=np.asarray(segments),
            )

        coords = np.vstack([instance.depot, instance.targets])
        matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
        matrix /= instance.carrier_speed
        factory_calls = []

        def ranker_factory(local_instance):
            factory_calls.append(local_instance.instance_id)
            return make_matrix_ranker(matrix)

        with (
            patch(
                "src.PseudoExpert.generate_policy_candidates",
                return_value=[[0, 2, 1, 3]],
            ),
            patch("src.CertifiedLocalSearch.solve", side_effect=fake_solve),
        ):
            record = generate_pseudo_expert_record(
                self.model,
                instance,
                split="train",
                device=torch.device("cpu"),
                solver_config=SolverConfig(solver_backend="gurobi"),
                generation_config=PseudoExpertGenerationConfig(
                    max_candidates=1,
                    max_local_search_refreshes=2,
                ),
                candidate_ranker_factory=ranker_factory,
                max_exact_candidates=1,
            )

        self.assertEqual(factory_calls, [instance.instance_id])
        self.assertGreater(record.generated_moves, record.evaluated_moves)
        self.assertEqual(record.pruned_moves, record.generated_moves - record.evaluated_moves)
        self.assertLessEqual(record.expert_objective, record.teacher_best_objective)
        candidate = record.teacher_candidates[0]
        self.assertEqual(candidate["evaluated_moves"], 2)
        self.assertGreater(candidate["pruned_moves"], 0)


if __name__ == "__main__":
    unittest.main()
