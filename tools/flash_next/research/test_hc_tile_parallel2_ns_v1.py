from pathlib import Path
import unittest

from prepare_hc_tile_parallel2_ns_v1 import builder_source, controller_source


ROOT = Path(__file__).parent


class PreciseHcProjectionTests(unittest.TestCase):
    def test_byte_identical_compute_and_precise_control_relinks(self):
        source = builder_source((ROOT / "build_dense_two_long_k_ns_v2.py").read_text())
        self.assertIn('candidate["original_hc_object_byte_identical"]', source)
        self.assertIn('sha(original_control) != control["provider_sha256"]', source)
        self.assertIn('sha(original_candidate) != candidate["provider_sha256"]', source)
        self.assertIn('CONTROL / "webgpu_context.cc.o"', source)
        self.assertIn("dawn-compute-scope-reuse/native-v1/native_compare", source)
        compile(source, "precise-hc-builder", "exec")

    def test_unchanged_native_operator_and_fallback_scope(self):
        source = controller_source((ROOT / "run_hc_tile_parallel2_v1.py").read_text())
        self.assertIn("len(fixtures) != 36", source)
        self.assertIn("total_exact_elements=607143936", source)
        self.assertIn("verify_static_stats(", source)
        self.assertIn("len(kernels) != 144", source)
        self.assertIn('":512,1,1:" if selected else ":256,1,1:"', source)
        self.assertIn("precise_duration_us(event)", source)
        self.assertNotIn('return [event["dur"]', source)
        compile(source, "precise-hc-controller", "exec")


if __name__ == "__main__":
    unittest.main()
