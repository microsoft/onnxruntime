from pathlib import Path
import unittest
from unittest.mock import patch

from build_fc1_u32_partials import FLAG, PROGRAM, transform, transform_program, transform_comparator


REPOS = Path("/home/sayanshaw/repos")
OVERLAY = REPOS / "qwen38-int4-qmoe-prefill256/build-v1/overlay/contrib_ops/webgpu/moe"


class U32PartialsTests(unittest.TestCase):
    def test_half_update_fold_and_all_final_reads_have_exact_original_fallbacks(self):
        original = (OVERLAY / "qmoe_fc1_activation.inc").read_text()
        result = transform_program(original)
        self.assertIn(PROGRAM, result)
        self.assertEqual(result.count("<< (u32_partials_ ?"), 6)
        for arithmetic in ("sum += dot(tile_a[lane * 8u + i * 2u], w0)",
                           "value += bias[expert * uniforms.cols * 2u + base_col + local_idx]",
                           "var value = f16(0)", "kidx += 1024u"):
            self.assertEqual(result.count(arithmetic), original.count(arithmetic))
        self.assertIn("array<array<u32, 32>, 8>", result)
        self.assertIn("array<array<f16, 32>, 8>", result)
        self.assertIn(".x + sum, f16(0)", result)
        self.assertIn("value += bitcast<vec2<f16>>(partials[local_idx][i]).x", result)
        self.assertNotIn("array<array<f32", result)

    def test_selector_is_validated_twice_keyed_and_storage_guarded(self):
        original = (REPOS / "qwen38-int4-qmoe-grouped-prefill/build-v1/qmoe.cc").read_text()
        result = transform(original, (OVERLAY / "qmoe_fc1_activation.inc").read_text())
        self.assertEqual(result.count(f'"{FLAG}", 0'), 2)
        self.assertEqual(result.count(f'"{FLAG} must be 0 or 1"'), 2)
        self.assertEqual(result.count("key.push_back(fc1_u32_partials)"), 1)
        self.assertIn("maxComputeWorkgroupStorageSize >= 3072", result)

    def test_comparator_selects_only_new_candidate(self):
        comparator = REPOS / "qwen38-int4-expert-bitcast-unpack/build-v2/native_compare.cpp"
        with patch("build_fc2_subgroup_fold.COMPARATOR", comparator):
            result = transform_comparator("")
        self.assertIn(f'"{FLAG}"', result)
        self.assertEqual(result.count(f'"{FLAG}"'), 4)
        self.assertNotIn('"ORT_WEBGPU_QMOE_FC2_SUBGROUP_FOLD"', result)

    def test_benchmark_metadata_selection_and_all_context_repeat_gates(self):
        from run_hc_up_parallel2_model import ROW4_EXPERT_VARIANTS, ROW4_REPEAT_VARIANTS
        from qsa_rank_decode_windows import FC1_U32_PARTIALS_FLAG
        self.assertEqual(FC1_U32_PARTIALS_FLAG, FLAG)
        for variants in (ROW4_EXPERT_VARIANTS, ROW4_REPEAT_VARIANTS):
            self.assertIn("fc1-u32-partials-v1", variants)
        source = Path(__file__).with_name("benchmark_webgpu_int4.py").read_text()
        self.assertIn("--compare-fc1-u32-partials-windows", source)
        self.assertIn("FC1_U32_PARTIALS_FLAG if args.compare_fc1_u32_partials_windows", source)
        self.assertIn('"ORT_WEBGPU_QMOE_FC1_U32_PARTIALS"', source)


if __name__ == "__main__":
    unittest.main()
