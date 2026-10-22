import copy
from contextlib import nullcontext
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.ImitationLearning import teacher_forcing_loss
from src.QualityILTrainer import QualityILTrainer, quality_multiplier
from src.TSPModel import ModelConfig, TSPModel
from tests.test_block_multi_action_decoder import _instance


class QualityILTests(unittest.TestCase):
    def test_logged_target_route_is_the_route_passed_to_exact_solver(self):
        trainer=object.__new__(QualityILTrainer)
        trainer.model=torch.nn.Linear(1,1)
        trainer.device=torch.device('cpu')
        trainer._diag_epoch=1
        trainer.si_upper_bound_tolerance=1e-4
        trainer.trainer_params={"quality_il_batch_size":64,"quality_il_scale":.02,
            "quality_il_offset":.5,"quality_il_min":.1,"quality_il_max":2.,
            "quality_il_measurement":"post_rl_sample_pomo_bound_selected_exact_v1"}
        instance=_instance(3)
        expert=[2,1,0]
        key=trainer._label_key(instance,expert)
        trainer._label_cache={key:{"expert_exact":5.,"anchor_matrix":np.ones((4,4))*100,
                                  "expert_matrix":None}}
        trainer._isolated_evaluation=lambda seed:nullcontext()
        trainer._capture_rng_state=lambda:{"torch_cpu":torch.get_rng_state(),
            "problem_size_generator":torch.get_rng_state(),"python_random":random.getstate(),
            "online_generator_bit_state":{},"numpy_global":np.random.get_state()}
        routes=[[[0,1,2],[1,2,0],[2,0,1]]]
        solved=[]
        def exact(instances,selected):
            solved.extend(selected)
            return [np.asarray([6.])],1
        trainer._exact_solve=exact
        with patch('src.QualityILTrainer.rollout_batch',return_value=SimpleNamespace(to_sequence_lists=lambda:routes)):
            measured,audit=trainer._measure_label_quality([(instance,expert,'current_changed',key)])
        self.assertEqual(measured[key]['target_sequence'],solved[0][0])
        self.assertEqual(measured[key]['target_sequence'],[0,1,2])
        self.assertNotEqual(measured[key]['target_sequence'],expert)
        self.assertEqual(measured[key]['target_exact'],6.)
        self.assertEqual(audit['extra_target_gurobi_calls'],1)

    def test_gap_sign_bounds_and_invalid_exact_values(self):
        for expert, expected in ((97, 2), (99, 1), (100, .5), (101, .1)):
            gap, q = quality_multiplier(100, expert)
            self.assertAlmostEqual(q, expected)
            self.assertAlmostEqual(gap, (100-expert)/100)
        for value in (float('nan'), float('inf'), 0, -1):
            with self.assertRaises(ValueError): quality_multiplier(value, 10)
            with self.assertRaises(ValueError): quality_multiplier(10, value)

    def test_real_teacher_forcing_weighted_micro_gradients_match_full_batch(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        model = TSPModel(ModelConfig(embedding_dim=32, encoder_layer_num=1, qkv_dim=8,
                                    head_num=4, ff_hidden_dim=64, dropout=0., block_size=1))
        micro = copy.deepcopy(model)
        instances = [_instance(5) for _ in range(3)]
        experts = [[0,1,2,3,4], [4,3,2,1,0], [2,0,4,1,3]]
        weights = [.1, 2., .5]
        full, *_ = teacher_forcing_loss(model, instances, experts, device=torch.device('cpu'), sample_weights=weights)
        full.backward()
        for start, end in ((0,2), (2,3)):
            loss, *_ = teacher_forcing_loss(micro, instances[start:end], experts[start:end],
                        device=torch.device('cpu'), sample_weights=weights[start:end])
            (loss * (sum(weights[start:end])/sum(weights))).backward()
        for a,b in zip(model.parameters(), micro.parameters()):
            if a.grad is not None: torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)
        plain, *_ = teacher_forcing_loss(model, instances, experts, device=torch.device('cpu'))
        uniform, *_ = teacher_forcing_loss(model, instances, experts, device=torch.device('cpu'), sample_weights=[1]*3)
        torch.testing.assert_close(plain, uniform)

    def test_weighted_optimizer_step_count_and_logical_batch_normalization(self):
        trainer = object.__new__(QualityILTrainer)
        trainer.model = torch.nn.Linear(1,1,bias=False)
        trainer.model.weight.data.zero_()
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1)
        trainer.scheduler, trainer.scheduler_interval, trainer.global_step = None, 'epoch', 0
        trainer.device, trainer.grad_clip = torch.device('cpu'), 100
        trainer._active_imitation_inner_epochs = 1
        trainer.imitation_batch_size, trainer.imitation_micro_batch_size = 3, 2
        trainer._phase, trainer._gradient_rows = 'il', []
        coeffs, weights = [1.,10.,-4.], [.1,2.,.5]
        rows = [(SimpleNamespace(J=2, coefficient=c), [0,1], 'current_anchor', i) for i,c in enumerate(coeffs)]
        measured = {i:{'quality_multiplier':w, 'relative_gap':0.} for i,w in enumerate(weights)}
        def fake(model, instances, experts, *, device, sample_weights):
            average = sum(x.coefficient*w for x,w in zip(instances,sample_weights))/sum(sample_weights)
            return model.weight.sum()*average, 0, 0, len(instances)*2
        with patch('src.QualityILTrainer.teacher_forcing_loss', side_effect=fake):
            _,_,audit = trainer._weighted_imitation(rows, measured)
        self.assertEqual(trainer.global_step, 1)
        self.assertEqual(audit['optimizer_steps'], 1)
        self.assertAlmostEqual(float(trainer.model.weight), -sum(c*w for c,w in zip(coeffs,weights))/sum(weights), places=5)
        self.assertAlmostEqual(sum(audit['normalized_ce_contribution_share'].values()), 1)

    def test_metadata_retains_replay_labels_after_fifo_eviction(self):
        trainer=object.__new__(QualityILTrainer)
        trainer.expert_replay_capacity=1
        trainer._expert_replay_pool=[]
        a,b=_instance(3),_instance(4)
        ka,kb=trainer._label_key(a,[0,1,2]),trainer._label_key(b,[0,1,2,3])
        trainer._label_cache={ka:{'expert_exact':10}, kb:{'expert_exact':20}}
        trainer._add_changed_experts_to_replay([a,b], [[0,1,2],[0,1,2,3]], [True,True])
        self.assertEqual(set(trainer._label_cache), {kb})


if __name__ == '__main__': unittest.main()
