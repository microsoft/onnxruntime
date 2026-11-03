import copy
import unittest

from build_dense_tile4 import FLAG
from qualify_hc_up_parallel2_standard import verify_standard
import test_qualify_qsa_score16_standard as fixtures


class StandardDenseTile4Tests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.StandardScore16Tests()
        self.fixture.setUp()
        self.fixture.raw["load_experiment_flags"].update(
            {FLAG: "1", "ORT_WEBGPU_HC_UP_PARALLEL2": "1", "ORT_WEBGPU_HC_UP_PREMIX": "1"})

    def verify(self, raw, full=True):
        f = self.fixture
        return verify_standard(raw, f.guard, f.native, f.parent, f.references, 2048, full,
                               experiment_flag=FLAG)

    def test_exact_metrics_in_both_validation_modes(self):
        self.assertEqual(self.verify(self.fixture.raw), self.fixture.expected)
        raw = copy.deepcopy(self.fixture.raw)
        raw["webgpu_validation_mode_override"] = None
        self.assertEqual(self.verify(raw, False), self.fixture.expected)

    def test_requires_candidate_and_all_parent_selectors(self):
        for flag in (FLAG, "ORT_WEBGPU_HC_UP_PARALLEL2", "ORT_WEBGPU_HC_UP_PREMIX",
                     "ORT_WEBGPU_DENSE_BITCAST_UNPACK", fixtures.FLAG):
            raw = copy.deepcopy(self.fixture.raw)
            raw["load_experiment_flags"][flag] = "0"
            with self.assertRaises(AssertionError):
                self.verify(raw)

    def test_rejects_windows_profiling_and_major_faults(self):
        for key, value in (("comparison", "dense_tile4_decode_windows"),
                           ("profiling", True), ("profile_decode_after_tokens", 35)):
            raw = copy.deepcopy(self.fixture.raw)
            raw[key] = value
            with self.assertRaises(AssertionError):
                self.verify(raw)
        raw = copy.deepcopy(self.fixture.raw)
        raw["samples"][0]["process_resource_usage"]["total"]["major_page_faults"] = 1
        with self.assertRaises(AssertionError):
            self.verify(raw)


if __name__ == "__main__":
    unittest.main()
