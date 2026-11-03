from pathlib import Path
import unittest

from build_fc1_expert_pairs import FLAG, transform


class ExpertPairsTests(unittest.TestCase):
    def test_exact_output_and_partial_slot_coverage(self):
        outputs = [(group // 160 * 2 + writer // 128, group % 160 * 4 + writer % 128)
                   for group in range(800) for writer in (*range(4), *range(128, 132))]
        self.assertEqual(sorted(outputs), [(expert, col) for expert in range(10) for col in range(640)])
        writes = [((thread // 128) * 8 + offset + (thread % 128) // 32, thread % 32)
                  for thread in range(256) for offset in (0, 4)]
        self.assertEqual(sorted(writes), [(col, lane) for col in range(16) for lane in range(32)])
        folds = [(thread // 128) * 8 + thread % 128 for thread in (*range(8), *range(128, 136))]
        self.assertEqual(folds, list(range(16)))

    def test_frozen_dot_partition_half_updates_fold_and_admission(self):
        root = Path("/home/sayanshaw/repos")
        source = (root / "qwen38-int4-qmoe-grouped-prefill/build-v1/qmoe.cc").read_text()
        program = (root / "qwen38-int4-qmoe-prefill256/build-v1/overlay/contrib_ops/webgpu/moe/"
                   "qmoe_fc1_activation.inc").read_text()
        actual = transform(source, program)
        for expression in ("kidx += 1024u", "var sum = f16(0)",
                           "sum += dot(tile_a[lane * 8u + i * 2u], w0)",
                           "var value = f16(0)", "i < 32u", "f16(glu * (linear_value + uniforms.beta))"):
            self.assertIn(expression, actual)
        self.assertEqual(actual.count(f'"{FLAG}", 0'), 2)
        self.assertEqual(actual.count(f"{FLAG} must be 0 or 1"), 2)
        self.assertIn("key.push_back(fc1_expert_pairs)", actual)
        self.assertIn("routed_rows == 10", actual)
        self.assertIn("maxComputeWorkgroupStorageSize >= 3072", actual)
        self.assertIn(".SetWorkgroupSize(use_fc1_expert_pairs ? 256 : 128)", actual)
        self.assertNotIn("ORT_WEBGPU_QMOE_FC1_EXPERTS10", actual)

    def test_model_window_metadata_and_conservative_repeat_gates_are_wired(self):
        from qsa_rank_decode_windows import FC1_EXPERT_PAIRS_FLAG
        from run_hc_up_parallel2_model import ROW4_EXPERT_VARIANTS, ROW4_REPEAT_VARIANTS
        self.assertEqual(FLAG, FC1_EXPERT_PAIRS_FLAG)
        self.assertIn("fc1-expert-pairs-v1", ROW4_EXPERT_VARIANTS)
        self.assertIn("fc1-expert-pairs-v1", ROW4_REPEAT_VARIANTS)
        source = Path(__file__).with_name("benchmark_webgpu_int4.py").read_text()
        self.assertIn("--compare-fc1-expert-pairs-windows", source)
        self.assertIn("FC1_EXPERT_PAIRS_FLAG if args.compare_fc1_expert_pairs_windows", source)
        self.assertIn('"ORT_WEBGPU_QMOE_FC1_EXPERT_PAIRS"', source)


if __name__ == "__main__":
    unittest.main()
