from pathlib import Path
import unittest

from build_fc1_full_a import FLAG, transform, transform_program
from run_hc_up_parallel2_model import verify_fc2_subgroup_window_support


class Fc1FullATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-grouped-prefill/build-v1/qmoe.cc").read_text()
        cls.program = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill256/build-v1/overlay/"
                           "contrib_ops/webgpu/moe/qmoe_fc1_activation.inc").read_text()

    def test_all_activation_addresses_and_unchanged_lane_products(self):
        loads = [address for thread in range(128) for address in range(thread, 640, 128)]
        self.assertEqual(sorted(loads), list(range(640)))
        for kidx in (0, 1024, 2048):
            for lane in range(32):
                if kidx // 32 + lane < 80:
                    for i in range(4):
                        for pair in (0, 1):
                            address = kidx // 4 + lane * 8 + i * 2 + pair
                            self.assertLess(address, 640)
                            self.assertEqual(address * 4, kidx + lane * 32 + i * 8 + pair * 4)

    def test_original_partial_updates_fold_and_activation_remain(self):
        actual = transform_program(self.program)
        self.assertIn("if (!full_a_)", actual)
        self.assertIn("tile_a: array<vec4<f16>, 640>", actual)
        self.assertIn("partials[offset + group][lane] += sum", actual)
        start = self.program.index('         << "if (local_idx < 8u')
        stop = self.program.index("    return Status::OK();", start)
        self.assertIn(self.program[start:stop], actual)
        self.assertIn('full_a_ ? "" : "  workgroupBarrier();', actual)
        self.assertIn("for (var id = local_idx; id < 640u; id += 128u)", actual)

    def test_validation_replay_and_original_geometry(self):
        actual = transform(self.source, self.program)
        self.assertEqual(actual.count(f"{FLAG} must be 0 or 1"), 2)
        self.assertIn("key.push_back(fc1_full_a)", actual)
        self.assertIn("key.push_back(grouped_prefill)", actual)
        self.assertIn("maxComputeWorkgroupStorageSize >= 5632", actual)
        self.assertIn(".SetWorkgroupSize(128).SetDispatchGroupSize(tiles, routed_rows)", actual)

    def test_model_selector_and_request_support(self):
        root = Path(__file__).parent
        benchmark = (root / "benchmark_webgpu_int4.py").read_text()
        self.assertIn("--compare-fc1-full-a-windows", benchmark)
        self.assertIn("FC1_FULL_A_FLAG if args.compare_fc1_full_a_windows", benchmark)
        result = dict(geometric_mean_matched_window_ratio=1.01, matched_window_pairs=16,
                      all_request_output_tokens_match=True,
                      per_request=[dict(repetition=i, fc1_full_a0_median_window_tps=80.0,
                                        fc1_full_a1_median_window_tps=81.0 if i < 3 else 79.0)
                                   for i in range(4)])
        verify_fc2_subgroup_window_support(result, prefix="fc1_full_a")
        result["geometric_mean_matched_window_ratio"] = 1.0
        with self.assertRaises(AssertionError):
            verify_fc2_subgroup_window_support(result, prefix="fc1_full_a")


if __name__ == "__main__":
    unittest.main()
