from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.EdgeCostPredictor import (
    DirectedEdgeCostPredictor,
    EdgeCostSample,
    EdgePredictorConfig,
    directed_route_cost,
    evaluate_edge_predictor,
    load_edge_cost_dataset,
    make_matrix_ranker,
    save_edge_cost_dataset,
)
from src.CertifiedUpperBound import iter_candidate_sequences
from src.PseudoExpert import instance_to_dict
from src.TSPModel import ModelConfig
from src.TSProblemDef import CVTSPInstance


class EdgeCostPredictorTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        torch.set_num_threads(1)
        self.instance = CVTSPInstance(
            instance_id="edge_unit",
            path=Path("<edge_unit>"),
            J=3,
            depot=np.asarray([0.0, 0.0], dtype=np.float32),
            targets=np.asarray([[1.0, 0.0], [0.0, 2.0], [2.0, 2.0]], dtype=np.float32),
            carrier_speed=2.0,
            uav_speed=5.0,
            endurance=1.0,
            scale=4.0,
        )
        self.matrix = np.asarray(
            [
                [0.0, 0.5, 1.0, 1.5],
                [0.4, 0.1, 0.8, 1.2],
                [0.9, 0.7, 0.2, 0.6],
                [1.4, 1.1, 0.5, 0.3],
            ],
            dtype=np.float32,
        )
        self.sample = EdgeCostSample(
            split="train",
            instance=instance_to_dict(self.instance),
            anchor_sequence=[0, 1, 2],
            anchor_objective=5.0,
            fixed_flight_time=2.0,
            directed_carrier_time=self.matrix.tolist(),
            source="unit",
            max_feasibility_violation=0.0,
        )
        self.model = DirectedEdgeCostPredictor(
            ModelConfig(
                embedding_dim=16,
                encoder_layer_num=1,
                qkv_dim=4,
                head_num=2,
                ff_hidden_dim=32,
            ),
            EdgePredictorConfig(hidden_dim=16, freeze_encoder=False),
        )

    def test_dataset_round_trip_and_normalization(self) -> None:
        expected = self.matrix * self.instance.carrier_speed / self.instance.scale
        np.testing.assert_allclose(self.sample.normalized_matrix(), expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "edges.json"
            save_edge_cost_dataset(path, [self.sample], metadata={"unit": True})
            samples, payload = load_edge_cost_dataset(path)
        self.assertTrue(payload["metadata"]["unit"])
        np.testing.assert_allclose(samples[0].time_matrix(), self.matrix)

    def test_predictor_is_positive_and_reports_metrics(self) -> None:
        features = torch.tensor(self.instance.node_features[None, :, :])
        prediction = self.model(features)
        self.assertEqual(tuple(prediction.shape), (1, 4, 4))
        self.assertTrue(bool((prediction > 0).all()))
        metrics = evaluate_edge_predictor(
            self.model,
            [self.sample],
            device=torch.device("cpu"),
            batch_size=1,
        )
        self.assertEqual(metrics["edges"], 12)
        self.assertTrue(np.isfinite(metrics["normalized_mae"]))
        self.assertTrue(-1.0 <= metrics["mean_edge_spearman"] <= 1.0)

    def test_directed_matrix_ranker_uses_route_direction(self) -> None:
        forward = directed_route_cost((0, 1, 2), self.matrix)
        reverse = directed_route_cost((2, 1, 0), self.matrix)
        self.assertNotEqual(forward, reverse)
        ranker = make_matrix_ranker(self.matrix)
        self.assertAlmostEqual(ranker((0, 1, 2)), forward, places=6)

    def test_incremental_ranker_matches_full_asymmetric_route_cost(self) -> None:
        rng = np.random.default_rng(20260812)
        matrix = rng.uniform(0.01, 3.0, size=(8, 8))
        np.fill_diagonal(matrix, 0.0)
        anchor = (4, 1, 6, 0, 3, 5, 2)
        ranker = make_matrix_ranker(matrix)
        ranker.prepare(anchor)
        for kind, first, second, candidate in iter_candidate_sequences(anchor):
            with self.subTest(kind=kind, first=first, second=second):
                incremental = ranker.score_move(kind, first, second, candidate)
                exact = directed_route_cost(candidate, matrix)
                self.assertAlmostEqual(incremental, exact, places=10)


if __name__ == "__main__":
    unittest.main()
