from pathlib import Path
import unittest

from build_fc2_compact20 import FLAG, transform, transform_program
from run_hc_up_parallel2_model import verify_row4_window_prerequisite, verify_fc2_subgroup_window_support


class Fc2Compact20Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-grouped-prefill/build-v1/qmoe.cc").read_text()
        cls.program = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill256/build-v1/overlay/"
                           "contrib_ops/webgpu/moe/qmoe_fc2_final_mix.inc").read_text()

    def test_exact_active_lane_ownership_and_complete_activation_loads(self):
        original = {(thread // 32, thread % 32)
                    for thread in range(320) if thread % 32 < 20}
        compact = {(thread // 20, thread % 20) for thread in range(256) if thread < 200}
        self.assertEqual(original, compact)
        self.assertEqual(len(compact), 200)
        addresses = [address for thread in range(256) for address in range(thread, 1600, 256)]
        self.assertEqual(sorted(addresses), list(range(1600)))

    def test_dot_fold_bias_and_rank_mix_remain_original(self):
        actual = transform_program(self.program)
        start = self.program.index('         << "  for (var c = 0u;')
        stop = self.program.index("    return Status::OK();", start)
        self.assertIn(self.program[start:stop], actual)
        self.assertIn('compact_ ? "" : "let expert = experts[rank];', actual)
        self.assertIn('if (local_idx < 200u) {\\n  let expert = experts[rank];', actual)
        self.assertEqual(actual.count('<< "workgroupBarrier();'), 3)
        self.assertIn("array<array<f16, 32>, 20>", actual)
        self.assertNotIn("subgroupBroadcast", actual)

    def test_selector_replay_geometry_and_unchanged_fallback_admission(self):
        actual = transform(self.source, self.program)
        self.assertEqual(actual.count(f"{FLAG} must be 0 or 1"), 2)
        self.assertIn("key.push_back(fc2_compact20)", actual)
        self.assertIn("key.push_back(grouped_prefill)", actual)
        self.assertIn("SetWorkgroupSize(use_fc2_compact20 ? 256 : 320)", actual)
        self.assertIn('context.AdapterInfo().vendor == std::string_view{"nvidia"}', actual)
        start = self.program.index("bool QMoE::CanFuseFc2FinalMix")
        self.assertIn(self.program[start:], actual)

    def test_model_selector_and_strict_request_support(self):
        root = Path(__file__).parent
        benchmark = (root / "benchmark_webgpu_int4.py").read_text()
        self.assertIn("--compare-fc2-compact20-windows", benchmark)
        self.assertIn("FC2_COMPACT20_FLAG if args.compare_fc2_compact20_windows", benchmark)
        self.assertIn(FLAG, benchmark)
        result = dict(geometric_mean_matched_window_ratio=1.01, matched_window_pairs=16,
                      all_request_output_tokens_match=True,
                      per_request=[dict(repetition=i, fc2_compact200_median_window_tps=80.0,
                                        fc2_compact201_median_window_tps=81.0 if i < 3 else 79.0)
                                   for i in range(4)])
        verify_fc2_subgroup_window_support(result, prefix="fc2_compact20")
        proof = dict(status="original_sequences_qualified", phase="windows", context=8192,
                     full_validation=True, provider_sha256="provider",
                     original_output_prefixes_match=True, zero_major_faults=True,
                     fixed_grouped_prefill_selector=1, result=result)
        verify_row4_window_prerequisite(proof, "provider", 8192, True)
        for field, value in (("status", "incomplete"), ("phase", "profile"), ("context", 2048),
                             ("full_validation", False), ("provider_sha256", "different"),
                             ("original_output_prefixes_match", False), ("zero_major_faults", False),
                             ("fixed_grouped_prefill_selector", 0)):
            with self.assertRaises(AssertionError):
                verify_row4_window_prerequisite({**proof, field: value}, "provider", 8192, True)
        result["per_request"][2]["fc2_compact201_median_window_tps"] = 79.0
        with self.assertRaises(AssertionError):
            verify_fc2_subgroup_window_support(result, prefix="fc2_compact20")


if __name__ == "__main__":
    unittest.main()
