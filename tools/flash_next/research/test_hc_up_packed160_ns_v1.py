from pathlib import Path
import unittest

from prepare_hc_up_packed160_ns_v1 import builder_source, controller_source


ROOT = Path(__file__).parent


class PreciseHcUpTests(unittest.TestCase):
    def test_unchanged_compute_ownership_and_original_relink_proof(self):
        actual = builder_source((ROOT / "build_dense_two_long_k_ns_v2.py").read_text())
        self.assertIn('not candidate["original_hc_up_object_byte_identical"]', actual)
        self.assertIn('OUTPUT / "original-control" / "libonnxruntime_providers_webgpu.so"', actual)
        self.assertIn('OUTPUT / "unmodified-candidate" / "libonnxruntime_providers_webgpu.so"', actual)
        self.assertIn('CONTROL / "webgpu_context.cc.o"', actual)
        self.assertIn('sha(original_candidate) != candidate["provider_sha256"]', actual)
        self.assertIn('"hc_up_compare"', actual)
        self.assertNotIn("original_dense_object", actual)
        compile(actual, "precise-hc-up-builder", "exec")

    def test_scope_geometry_and_thresholds_do_not_change(self):
        actual = controller_source((ROOT / "run_hc_up_packed160_v1.py").read_text())
        self.assertIn("len(kernels) != 288", actual)
        self.assertIn("return precise[33::2]", actual)
        self.assertIn("threads = 160 if selected else 128", actual)
        self.assertIn("ratio > 1.01 and positive >= 120", actual)
        self.assertIn("str(2560 * 144 * 2)", actual)
        self.assertIn('control_manifest["provider_sha256"]', actual)
        self.assertIn('precise_duration_us(event) for event in kernels', actual)
        self.assertNotIn('return [event["dur"]', actual)
        compile(actual, "precise-hc-up-controller", "exec")


if __name__ == "__main__":
    unittest.main()
