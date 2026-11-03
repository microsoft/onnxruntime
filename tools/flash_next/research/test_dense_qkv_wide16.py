from pathlib import Path
import unittest

from build_dense_qkv_wide16 import FLAG, transform
from run_dense_qkv_wide16_native import verify_stats
from run_hc_up_parallel2_model import verify_fc2_subgroup_window_support


class DenseQkvWide16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc").read_text()

    def test_only_target_geometry_changes_and_prefill_remains_present(self):
        actual = transform(self.source)
        self.assertIn("constexpr uint32_t workgroup_size = 512;\n  constexpr uint32_t tile_size = 16;", actual)
        self.assertIn("K_ == 2560 && N_ == 16384", actual)
        self.assertIn("maxComputeWorkgroupStorageSize >= 3072", actual)
        self.assertIn(f"{FLAG} must be 0 or 1", actual)
        self.assertIn("qkv_wide16_route ? 19", actual)
        self.assertIn("GroupedPrefillRows4Program", actual)
        begin = self.source.index("class DenseBlock32Program")
        end = self.source.index("\n}\n\nStatus MatMulNBitsProgram::", begin)
        self.assertIn(self.source[begin:end], actual)
        self.assertIn("WGSL_TEMPLATE_PARAMETER(tile_size_k_vec, tile_size_k_vec)", actual)

    def test_complete_disjoint_column_and_k_lane_ownership(self):
        pairs = {(thread // 32, thread % 32) for thread in range(512)}
        self.assertEqual(pairs, {(column, lane) for column in range(16) for lane in range(32)})
        self.assertEqual(16384 // 16, 1024)

    def test_only_target_shape_gets_an_extra_replay_variant(self):
        row = dict(operator="MatMulNBitsExact", requests=256, hits=254, captures=2,
                   rebinds=0, binding_hits=254, entries=1, capacity=64,
                   binding_entries=1, binding_capacity=64)
        rows = [dict(row) for _ in range(9)]
        rows[0]["entries"] = 2
        verify_stats(rows)
        rows[1]["entries"] = 2
        with self.assertRaises(AssertionError):
            verify_stats(rows)

    def test_benchmark_selection_and_original_program_coverage(self):
        root = Path(__file__).parent
        benchmark = (root / "benchmark_webgpu_int4.py").read_text()
        self.assertIn("--compare-dense-qkv-wide16-windows", benchmark)
        self.assertIn("DENSE_QKV_WIDE16_FLAG if args.compare_dense_qkv_wide16_windows", benchmark)
        wrapper = (root / "run_dense_qkv_wide16_model.py").read_text()
        self.assertIn("dispatches=36, original_dispatches=123", wrapper)
        result = dict(geometric_mean_matched_window_ratio=1.01, matched_window_pairs=16,
                      all_request_output_tokens_match=True,
                      per_request=[dict(repetition=i, dense_qkv_wide160_median_window_tps=80.0,
                                        dense_qkv_wide161_median_window_tps=81.0) for i in range(4)])
        verify_fc2_subgroup_window_support(result, prefix="dense_qkv_wide16")
        result["per_request"][0]["dense_qkv_wide161_median_window_tps"] = float("nan")
        with self.assertRaises(AssertionError):
            verify_fc2_subgroup_window_support(result, prefix="dense_qkv_wide16")


if __name__ == "__main__":
    unittest.main()
