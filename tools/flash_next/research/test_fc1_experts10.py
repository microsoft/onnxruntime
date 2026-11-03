from pathlib import Path
import unittest

from build_fc1_experts10 import FLAG, transform, transform_program
from run_hc_up_parallel2_model import verify_fc2_subgroup_window_support


class Fc1Experts10Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-grouped-prefill/build-v1/qmoe.cc").read_text()
        cls.program = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill256/build-v1/overlay/"
                           "contrib_ops/webgpu/moe/qmoe_fc1_activation.inc").read_text()

    def test_complete_original_expert_column_lane_ownership(self):
        tasks = {(thread // 32, column, thread % 32)
                 for thread in range(320) for column in range(2)}
        self.assertEqual(tasks, {(rank, column, lane) for rank in range(10)
                                 for column in range(2) for lane in range(32)})
        loads = [address for thread in range(320) for address in range(thread, 256, 320)]
        self.assertEqual(sorted(loads), list(range(256)))
        original_outputs = {(workgroup // 160, (workgroup % 160) * 4 + column)
                            for workgroup in range(1600) for column in range(4)}
        candidate_outputs = {(rank, column) for rank in range(10) for column in range(640)}
        self.assertEqual(original_outputs, candidate_outputs)

    def test_original_dots_fold_and_activation_expressions(self):
        actual = transform_program(self.program)
        start = self.program.index('         << "    let col')
        stop = self.program.index('         << "      partials', start)
        self.assertIn(self.program[start:stop], actual)
        self.assertIn('for (var i = 0u; i < 32u; i++) { value += partials[local_idx][i]; }', actual)
        start = self.program.index('         << "    var gate_value')
        stop = self.program.index('         << output.SetByOffset', start)
        self.assertIn(self.program[start:stop], actual)
        self.assertIn("partials[row * 2u + offset][lane] += sum", actual)
        self.assertIn("experts[local_idx / 2u]", actual)
        self.assertIn('"local_idx * uniforms.cols + col"', actual)

    def test_explicit_validation_replay_and_unchanged_admission(self):
        actual = transform(self.source, self.program)
        self.assertEqual(actual.count(f"{FLAG} must be 0 or 1"), 2)
        self.assertIn("key.push_back(fc1_experts10)", actual)
        self.assertIn("key.push_back(grouped_prefill)", actual)
        self.assertIn("maxComputeWorkgroupStorageSize >= 3328", actual)
        self.assertIn("SetWorkgroupSize(use_fc1_experts10 ? 320 : 128)", actual)
        start = self.program.index("bool QMoE::CanFuseFc1Activation")
        self.assertIn(self.program[start:], actual)

    def test_model_selector_and_positive_request_gate(self):
        root = Path(__file__).parent
        benchmark = (root / "benchmark_webgpu_int4.py").read_text()
        self.assertIn("--compare-fc1-experts10-windows", benchmark)
        self.assertIn("FC1_EXPERTS10_FLAG if args.compare_fc1_experts10_windows", benchmark)
        result = dict(geometric_mean_matched_window_ratio=1.01, matched_window_pairs=16,
                      all_request_output_tokens_match=True,
                      per_request=[dict(repetition=i, fc1_experts100_median_window_tps=80.0,
                                        fc1_experts101_median_window_tps=81.0 if i < 3 else 79.0)
                                   for i in range(4)])
        verify_fc2_subgroup_window_support(result, prefix="fc1_experts10")
        result["per_request"][2]["fc1_experts101_median_window_tps"] = 79.0
        with self.assertRaises(AssertionError):
            verify_fc2_subgroup_window_support(result, prefix="fc1_experts10")


if __name__ == "__main__":
    unittest.main()
