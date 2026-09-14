from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from src.TSPEnv import TSPEnv, generate_candidate_pool_batch, rollout_batch
from src.ImitationLearning import teacher_forcing_loss
from src.TSPModel import ModelConfig, TSPModel
from src.TSProblemDef import CVTSPInstance


def _instance(problem_size: int = 8) -> CVTSPInstance:
    angles = np.linspace(0.0, 2.0 * np.pi, problem_size, endpoint=False)
    targets = np.stack((np.cos(angles), np.sin(angles)), axis=1).astype(np.float32)
    return CVTSPInstance(
        instance_id=f"block_j{problem_size}",
        path=Path(f"<block_j{problem_size}>"),
        J=problem_size,
        depot=np.asarray([0.0, 0.0], dtype=np.float32),
        targets=targets,
        carrier_speed=0.25,
        uav_speed=1.0,
        endurance=2.0,
        scale=1.0,
    )


class BlockMultiActionDecoderTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)
        torch.set_num_threads(1)
        self.model = TSPModel(
            ModelConfig(
                embedding_dim=32,
                encoder_layer_num=2,
                qkv_dim=8,
                head_num=4,
                ff_hidden_dim=64,
                activation="gelu",
                dropout=0.0,
                block_size=4,
                max_pomo_size=8,
            )
        )

    def test_j8_k4_rollout_visits_every_node_once(self) -> None:
        rollout = rollout_batch(
            self.model,
            [_instance(8)],
            device=torch.device("cpu"),
            decode_type="greedy",
            use_pomo_start=True,
        )
        self.assertEqual(tuple(rollout.sequences.shape), (1, 8, 8))
        for sequence in rollout.sequences[0].tolist():
            self.assertEqual(len(sequence), 8)
            self.assertEqual(sorted(sequence), list(range(8)))

    def test_logits_shape_mask_and_short_final_block(self) -> None:
        instance = _instance(8)
        env = TSPEnv(problem_size=8, pomo_size=8, use_pomo_start=True)
        env.load_problems([instance], torch.device("cpu"))
        reset_state, _, _ = env.reset()
        self.model.pre_forward(reset_state.node_features)
        state, _, _ = env.pre_step()

        depot, _ = self.model(state, decode_type="greedy", use_pomo_start=True)
        state, _, _ = env.step(depot)
        starts, _ = self.model(state, decode_type="greedy", use_pomo_start=True)
        state, _, _ = env.step(starts)

        selected, _, logits, available, active = self.model.forward_block(
            state,
            decode_type="greedy",
            use_pomo_start=True,
        )
        self.assertEqual(tuple(logits.shape), (1, 8, 4, 7))
        self.assertEqual(tuple(available.shape), (1, 8, 7))
        self.assertEqual(active.tolist(), [True, True, True, True])
        self.assertTrue(all(len(set(row)) == 4 for row in selected[0].tolist()))
        for slot in range(selected.size(2)):
            state, _, _ = env.step(selected[:, :, slot])

        selected, _, logits, available, active = self.model.forward_block(
            state,
            decode_type="greedy",
            use_pomo_start=True,
        )
        self.assertEqual(tuple(logits.shape), (1, 8, 4, 3))
        self.assertEqual(tuple(available.shape), (1, 8, 3))
        self.assertEqual(active.tolist(), [True, True, True, False])
        self.assertEqual(tuple(selected.shape), (1, 8, 3))
        self.assertTrue(all(len(set(row)) == 3 for row in selected[0].tolist()))

        for slot in range(selected.size(2)):
            state, _, done = env.step(selected[:, :, slot])
        self.assertTrue(done)
        raw = env.selected_node_list
        self.assertIsNotNone(raw)
        for route in raw[0, :, 1:].tolist():
            self.assertEqual(sorted(node - 1 for node in route), list(range(8)))

    def test_ordered_block_imitation_loss_decreases(self) -> None:
        instances = [_instance(8) for _ in range(4)]
        experts = [list(range(8)) for _ in instances]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        initial, _, _, tokens = teacher_forcing_loss(
            self.model,
            instances,
            experts,
            device=torch.device("cpu"),
        )
        initial_value = float(initial.detach().item())
        self.assertEqual(tokens, 32)
        for _ in range(35):
            loss, _, _, _ = teacher_forcing_loss(
                self.model,
                instances,
                experts,
                device=torch.device("cpu"),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        final, _, _, _ = teacher_forcing_loss(
            self.model,
            instances,
            experts,
            device=torch.device("cpu"),
        )
        self.assertLess(float(final.detach().item()), initial_value * 0.5)

    def test_reference_pool_contains_exactly_beta_200_routes(self) -> None:
        pool = generate_candidate_pool_batch(
            self.model,
            [_instance(8)],
            device=torch.device("cpu"),
            num_candidates=200,
            decode_type="sample",
        )
        self.assertEqual(len(pool), 1)
        self.assertEqual(len(pool[0]), 200)
        for route in pool[0]:
            self.assertEqual(sorted(route), list(range(8)))


if __name__ == "__main__":
    unittest.main()
