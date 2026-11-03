from pathlib import Path
import unittest

from build_fc1_columns8 import FLAG, transform, transform_program
from run_hc_up_parallel2_model import verify_row4_window_prerequisite


class Fc1Columns8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-grouped-prefill/build-v1/qmoe.cc").read_text()
        cls.program = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill256/build-v1/overlay/"
                           "contrib_ops/webgpu/moe/qmoe_fc1_activation.inc").read_text()

    def test_unique_columns_experts_and_activation_staging(self):
        columns = [offset + thread // 32 for offset in (0, 8) for thread in range(0, 256, 32)]
        self.assertEqual(sorted(columns), list(range(16)))
        outputs = [(group // 80, (group % 80) * 8 + writer)
                   for group in range(800) for writer in range(8)]
        self.assertEqual(sorted(outputs), [(expert, col) for expert in range(10) for col in range(640)])
        loads = [id for thread in range(256) for id in range(thread, 256, 256)]
        self.assertEqual(sorted(loads), list(range(256)))
        self.assertEqual(len(outputs), 6400)

    def test_original_products_partial_updates_and_fold_preserved(self):
        actual = transform_program(self.program)
        for expression in ("partials[offset + group][lane] += sum",
                           "for (var i = 0u; i < 32u; i++) { value += partials[local_idx][i]; }",
                           "sum += dot(tile_a[lane * 8u + i * 2u], w0)",
                           "f16(glu * (linear_value + uniforms.beta))"):
            self.assertIn(expression, actual)
        self.assertIn("columns8_ ?", actual)
        self.assertIn("partials: array<array<f16, 32>, 16>", actual)

    def test_limits_validation_replay_and_dispatch(self):
        actual = transform(self.source, self.program)
        self.assertEqual(actual.count(f"{FLAG} must be 0 or 1"), 2)
        self.assertIn("key.push_back(fc1_columns8)", actual)
        self.assertIn("maxComputeWorkgroupStorageSize >= 3072", actual)
        self.assertIn("CeilDiv(cols * 2u, use_fc1_columns8 ? 16u : 8u)", actual)
        self.assertIn(".SetWorkgroupSize(use_fc1_columns8 ? 256 : 128)", actual)

    def test_model_wiring_and_strict_long_prerequisite(self):
        root = Path(__file__).parent
        benchmark = (root / "benchmark_webgpu_int4.py").read_text()
        self.assertIn("--compare-fc1-columns8-windows", benchmark)
        self.assertIn("FC1_COLUMNS8_FLAG if args.compare_fc1_columns8_windows", benchmark)
        proof = dict(status="original_sequences_qualified", phase="windows", context=2048,
                     full_validation=True, provider_sha256="provider", original_output_prefixes_match=True,
                     zero_major_faults=True, fixed_grouped_prefill_selector=1,
                     result=dict(geometric_mean_matched_window_ratio=1.01, matched_window_pairs=16,
                                 all_request_output_tokens_match=True,
                                 per_request=[dict(repetition=i, fc1_columns80_median_window_tps=80.0,
                                                   fc1_columns81_median_window_tps=81.0)
                                              for i in range(4)]))
        verify_row4_window_prerequisite(proof, "provider", 2048, True, prefix="fc1_columns8")
        for field, value in (("context", 8192), ("full_validation", False),
                             ("provider_sha256", "wrong"), ("fixed_grouped_prefill_selector", 0),
                             ("zero_major_faults", False)):
            with self.subTest(field=field), self.assertRaises(AssertionError):
                verify_row4_window_prerequisite({**proof, field: value}, "provider", 2048, True,
                                               prefix="fc1_columns8")


if __name__ == "__main__":
    unittest.main()
