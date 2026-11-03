import collections
import json
import unittest

from run_dense_packed4_model import DIRECTORY, FLAG, ROOT, TEMPLATE, model_command, prefix_for, verify_profile


class DensePacked4ModelTests(unittest.TestCase):
    def test_standard_flags_are_explicit_without_cpu_or_runtime_changes(self):
        template = json.loads((ROOT / f"{TEMPLATE}.guard.json").read_text())["command"]
        for mode in ("control", "packed4"):
            command = model_command(template, DIRECTORY / "libonnxruntime_providers_webgpu.so",
                                    prefix_for(mode, 2048, True, 0), 2048, True, mode, "standard")
            self.assertIn(f"export {FLAG}={int(mode == 'packed4')}; ", command[-1])
            self.assertIn("--tokens 256 --repetitions 3", command[-1])
            self.assertNotIn("--inherit-worker-cpu-affinity", command[-1])
            self.assertNotIn("ORT_GENAI_WEBGPU_ALLOCATOR_SINGLE_THREAD", command[-1])

    def test_profile_only_last_decode_token(self):
        template = json.loads((ROOT / f"{TEMPLATE}.guard.json").read_text())["command"]
        command = model_command(template, DIRECTORY / "libonnxruntime_providers_webgpu.so",
                                prefix_for("packed4", 2048, True, 0, "profile"),
                                2048, True, "packed4", "profile")
        self.assertIn("--tokens 36 --repetitions 1 --profile-decode-after 35", command[-1])

    def test_profile_requires_exact_actual_path_and_no_repacking(self):
        counts = collections.Counter(MatMulNBitsPacked4F16=159, DynamicSparseAttentionQueryCache=12)
        verify_profile(counts, "packed4")
        for name in ("MatMulNBitsBlock32F16", "MatMulNBitsTile4F16", "DensePack4Columns", "DynamicSparseAttention"):
            counts[name] = 1
            with self.assertRaises(AssertionError):
                verify_profile(counts, "packed4")
            counts[name] = 0


if __name__ == "__main__":
    unittest.main()
