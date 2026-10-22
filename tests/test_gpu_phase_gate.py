import multiprocessing as mp
from pathlib import Path
import tempfile
import time
import unittest

from src.GPUPhaseGate import gpu_phase_slot


def enter_slot(directory, active, maximum, counter_lock):
    with gpu_phase_slot(directory, 2):
        with counter_lock:
            active.value += 1
            maximum.value = max(maximum.value, active.value)
        time.sleep(.15)
        with counter_lock:
            active.value -= 1


class GPUPhaseGateTests(unittest.TestCase):
    def test_three_jobs_never_exceed_two_slots_and_locks_release(self):
        context = mp.get_context("fork")
        with tempfile.TemporaryDirectory() as directory:
            active, maximum = context.Value("i", 0), context.Value("i", 0)
            lock = context.Lock()
            jobs = [context.Process(target=enter_slot, args=(directory, active, maximum, lock)) for _ in range(3)]
            for job in jobs:
                job.start()
            for job in jobs:
                job.join(5)
                self.assertEqual(job.exitcode, 0)
            self.assertEqual(maximum.value, 2)
            self.assertEqual(active.value, 0)
            self.assertEqual(list(Path(directory).glob("waiting-*")), [])
            with self.assertRaises(RuntimeError):
                with gpu_phase_slot(directory, 1):
                    raise RuntimeError("probe")
            with gpu_phase_slot(directory, 1):
                pass


if __name__ == "__main__":
    unittest.main()
