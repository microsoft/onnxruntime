import copy
import unittest

from run_gpu_fused_slice_model import CPUS, UUID, verify_placement


class ParallelPlacementTests(unittest.TestCase):
    def setUp(self):
        self.raw = dict(cpu_affinity=CPUS.copy(), worker_affinity_override=dict(target_cpus=CPUS.copy()),
                        thread_cpu_affinity={"main": CPUS.copy(), "worker": CPUS.copy()})
        self.guard = dict(gpu_index=1, gpu_uuid=UUID)

    def test_requires_actual_gpu1_identity(self):
        verify_placement(self.raw, self.guard)
        for changed in (dict(gpu_index=7, gpu_uuid=UUID), dict(gpu_index=1, gpu_uuid="wrong")):
            with self.assertRaises(RuntimeError):
                verify_placement(self.raw, changed)

    def test_rejects_empty_or_escaped_workers(self):
        for masks in ({}, {"worker": [48]}, {"worker": []}, {"worker": CPUS[:-1]}):
            raw = copy.deepcopy(self.raw)
            raw["thread_cpu_affinity"] = masks
            with self.assertRaises(RuntimeError):
                verify_placement(raw, self.guard)

    def test_rejects_mismatched_override(self):
        raw = copy.deepcopy(self.raw)
        raw["worker_affinity_override"]["target_cpus"] = list(range(48, 56))
        with self.assertRaises(RuntimeError):
            verify_placement(raw, self.guard)


if __name__ == "__main__":
    unittest.main()
