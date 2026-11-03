import math
import unittest

from prepare_dawn_scope_qualification import fingerprint_comparator
from run_dawn_scope_qualification_v1 import BUILD, NATIVE, ROOT, command_for, speed_ratios


class ScopeQualificationTests(unittest.TestCase):
    def test_operator_speed_uses_matched_geometric_controls(self):
        ratios = speed_ratios([[4.0] * 128, [3.0] * 128, [9.0] * 128])
        self.assertEqual(ratios, [2.0] * 128)

    def test_rejects_bad_operator_values_and_count(self):
        for rows in (
                [[1.0], [1.0], [1.0]],
                [[1.0] * 128, [math.nan] * 128, [1.0] * 128]):
            with self.assertRaises(ValueError):
                speed_ratios(rows)

    def test_native_command_targets_owned_gpu_and_explicit_private_fc2_off(self):
        command = command_for(NATIVE / "native_compare", BUILD / "libonnxruntime_providers_webgpu.so",
                              [], ROOT / "scope-test-stats.jsonl")
        self.assertEqual(command[:5], ["taskset", "-c", "0-7", "bash", "-lc"])
        self.assertIn("GPU-c1593c4e-0f5f-7bf7-3567-6c3c19703afd", command[-1])
        self.assertIn("ORT_WEBGPU_QMOE_FC2_PRIVATE_A=0", command[-1])
        self.assertIn("ORT_WEBGPU_STATIC_NODE_REPLAY_STATS_FILE=/evidence/", command[-1])

    def test_comparator_keeps_original_proof_and_fingerprints_every_byte(self):
        source = (BUILD.parent.parent /
                  "qwen38-int4-hc-post-norm-projection/build-v2/native_compare.cpp").read_text()
        changed = fingerprint_comparator(source)
        self.assertIn("fingerprint ^= observed[i * bytes + byte]", changed)
        self.assertIn("if (!finite || std::memcmp", changed)
        self.assertIn("++comparisons;", changed)
        self.assertEqual(changed.count("fingerprint *= 1099511628211ull"), 1)


if __name__ == "__main__":
    unittest.main()
