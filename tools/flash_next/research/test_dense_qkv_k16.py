from pathlib import Path
import unittest

from build_dense_qkv_k16 import FLAG, transform, transform_template
from run_hc_up_parallel2_model import ROW4_REPEAT_VARIANTS, verify_fc2_subgroup_window_support


class DenseQkvK16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc").read_text()
        cls.template = Path("/home/sayanshaw/repos/qwen38-int4-dense-block32/block32-template-v1/"
                            "dense_block32_template.inc").read_text()

    def test_partial_slots_preserve_each_original_half_update_sequence(self):
        for lane in range(32):
            original = [tile // 32 + lane for tile in range(0, 2560, 1024)
                        if tile // 32 + lane < 80]
            candidate = [tile // 32 + local for tile in range(0, 2560, 512)
                         for local in range(16) if (tile // 32) % 32 + local == lane]
            self.assertEqual(candidate, original)
        self.assertEqual([(thread // 16, thread % 16) for thread in range(128)],
                         [(col, lane) for col in range(8) for lane in range(16)])

    def test_dot_expressions_and_original_32_slot_fold_remain(self):
        actual = transform_template(self.template)
        for line in self.template.splitlines():
            if line.startswith("ss <<") and ("sum +=" in line or "dot(" in line):
                self.assertIn(line, actual)
        self.assertIn("ss << 32u;", actual)
        self.assertIn("inter_results[local_row_offset + idy][(kidx / 32u) % 32u + idx] += sum", actual)
        self.assertIn("output_value += inter_results[local_idx][b]", actual)
        self.assertEqual(actual.count('ss << "    workgroupBarrier();'), 2)

    def test_shape_limits_flags_and_row4_fallback_preserved(self):
        actual = transform(self.source)
        self.assertIn(f"{FLAG} must be 0 or 1", actual)
        self.assertIn("K_ == 2560 && N_ == 16384", actual)
        self.assertIn("maxComputeWorkgroupStorageSize >= 1536", actual)
        self.assertIn("qkv_k16_route ? 20", actual)
        self.assertIn("constexpr uint32_t tile_size = 8", actual)
        self.assertIn("GroupedPrefillRows4Program program", actual)
        self.assertIn("ApplyDenseBlock32MatMulNBits", actual)

    def test_model_comparison_profile_and_repeat_admission(self):
        root = Path(__file__).parent
        benchmark = (root / "benchmark_webgpu_int4.py").read_text()
        self.assertIn("--compare-dense-qkv-k16-windows", benchmark)
        self.assertIn("DENSE_QKV_K16_FLAG if args.compare_dense_qkv_k16_windows", benchmark)
        self.assertIn("dense-qkv-k16-v1", ROW4_REPEAT_VARIANTS)
        wrapper = (root / "run_dense_qkv_k16_model.py").read_text()
        self.assertIn("dispatches=36, original_dispatches=123", wrapper)

    def test_positive_request_support_and_matched_ratio_required(self):
        result = dict(geometric_mean_matched_window_ratio=1.01, matched_window_pairs=16,
                      all_request_output_tokens_match=True,
                      per_request=[dict(repetition=i, dense_qkv_k160_median_window_tps=80.0,
                                        dense_qkv_k161_median_window_tps=81.0 if i < 3 else 79.0)
                                   for i in range(4)])
        verify_fc2_subgroup_window_support(result, prefix="dense_qkv_k16")
        result["geometric_mean_matched_window_ratio"] = 1.0
        with self.assertRaises(AssertionError):
            verify_fc2_subgroup_window_support(result, prefix="dense_qkv_k16")


if __name__ == "__main__":
    unittest.main()
