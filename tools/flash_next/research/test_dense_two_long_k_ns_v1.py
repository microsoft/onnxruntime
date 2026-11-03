from pathlib import Path
import unittest

from prepare_dense_two_long_k_ns_v1 import controller_source
from prepare_dense_two_long_k_ns_v2 import builder_source


ROOT = Path(__file__).parent


class PreciseDenseOperatorTests(unittest.TestCase):
    def test_relink_proof_keeps_original_elf_base_version_library_basename(self):
        actual = builder_source((ROOT / "build_dense_two_long_k_ns_v1.py").read_text())
        self.assertIn('OUTPUT / "original-control" / "libonnxruntime_providers_webgpu.so"', actual)
        self.assertIn('OUTPUT / "unmodified-candidate" / "libonnxruntime_providers_webgpu.so"', actual)
        self.assertIn('library.parent.mkdir(parents=True, exist_ok=True)', actual)
        self.assertIn('sha(original_control) != control["provider_sha256"]', actual)
        self.assertIn('sha(original_candidate) != candidate["provider_sha256"]', actual)
        self.assertNotIn('OUTPUT / "original-control-relink.so"', actual)
        compile(actual, "precise-dense-builder-v2", "exec")

    def test_precision_changes_not_shader_scope_or_admission_thresholds(self):
        actual = controller_source((ROOT / "run_dense_tile_parallel2_long_k_v1.py").read_text())
        self.assertIn("precise = [precise_duration_us(event) for event in kernels]", actual)
        self.assertIn("return precise[16:]", actual)
        self.assertIn("len(kernels) != 144", actual)
        self.assertIn("positive >= 120", actual)
        self.assertIn("ratio > 1.01", actual)
        self.assertIn('"selected": (k, n) == (6144, 2560)', actual)
        self.assertIn('control_manifest["provider_sha256"]', actual)
        self.assertIn('manifest["all_candidate_compute_code_unchanged"]', actual)
        self.assertIn("precise_gpu_dense_tile_parallel2_long_k_operator_screen_completed", actual)
        self.assertNotIn('return [event["dur"]', actual)
        compile(actual, "precise-dense-controller", "exec")


if __name__ == "__main__":
    unittest.main()
