from pathlib import Path
import tempfile
import unittest
import numpy as np

from scripts.lkh_fixed_bound import LKHConfig,prepare_problem,quantize_matrix,read_tour,run_start
from src.JointSelfImprovement import directed_route_upper_bound


class LKHAdapterTests(unittest.TestCase):
    def test_rotated_asymmetric_tour_is_not_reversed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'out.tour'
            p.write_text('TYPE: TOUR\nTOUR_SECTION\n3\n1\n2\n4\n-1\nEOF\n')
            self.assertEqual(read_tour(p,3),[0,2,1])
            p.write_text('TOUR_SECTION\n1\n2\n2\n4\n-1\n')
            with self.assertRaises(ValueError): read_tour(p,3)

    def test_quantization_error_bound_and_overflow_guard(self):
        rng=np.random.default_rng(9)
        m=rng.random((6,6))*1e5
        q,s,error=quantize_matrix(m)
        self.assertLessEqual(int(q.max()),np.iinfo(np.int32).max//8)
        for route in ([0,1,2,3,4],[4,3,2,1,0]):
            delta=abs(directed_route_upper_bound(route,m)-directed_route_upper_bound(route,q)/s)
            self.assertLessEqual(delta,error+1e-8)
        with self.assertRaises(ValueError): quantize_matrix(np.full((3,3),np.inf))

    def test_native_lkh_finds_planted_directed_cycle(self):
        binary=Path('third_party/lkh3_20261014/LKH').resolve()
        if not binary.exists(): self.skipTest('isolated LKH executable unavailable')
        m=np.full((4,4),100.)
        np.fill_diagonal(m,0)
        for a,b in ((0,1),(1,3),(3,2),(2,0)):m[a,b]=1.
        with tempfile.TemporaryDirectory() as d:
            config=LKHConfig(str(binary))
            prepare_problem(m,d,config)
            result=run_start(m,[0,1,2],d,config,prefix='probe',seed=7)
            self.assertTrue(result['completed'])
            self.assertTrue(result['accepted_output'])
            self.assertEqual(result['final_sequence'],[0,2,1])
            self.assertEqual(result['final_upper_bound'],4.)


if __name__=='__main__':unittest.main()
