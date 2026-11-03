from pathlib import Path
import unittest

import numpy as np

from build_fc2_subgroup_fold import FLAG, transform, transform_program
from run_hc_up_parallel2_model import verify_fc2_subgroup_window_support


class Fc2SubgroupFoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-grouped-prefill/build-v1/qmoe.cc").read_text()
        cls.program = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill256/build-v1/overlay/"
                           "contrib_ops/webgpu/moe/qmoe_fc2_final_mix.inc").read_text()

    def test_typed_shared_materialization_and_original_prefix_mix_are_preserved(self):
        actual = transform_program(self.program)
        prefix_start = self.program.index('    code << "let base_col')
        prefix_stop = self.program.index('         << "if (local_idx < 20u)')
        self.assertIn(self.program[prefix_start:prefix_stop], actual)
        mix_start = self.program.index('         << "if (local_idx < 2u)')
        mix_stop = self.program.index("    return Status::OK();", mix_start)
        self.assertIn(self.program[mix_start:mix_stop], actual)
        self.assertIn("bitcast<u32>(vec2<f16>(partials[rank * 2u][lane]", actual)
        self.assertIn("source_lane < 32; source_lane++", actual)
        self.assertIn("subgroupBroadcast(packed_partial", actual)
        self.assertIn("partials[rank * 2u + 1u][0] = values.y", actual)
        self.assertNotIn("subgroupBroadcast(f16", actual)

    def test_admission_replay_key_and_explicit_validation(self):
        actual = transform(self.source, self.program)
        self.assertEqual(actual.count(f"{FLAG} must be 0 or 1"), 2)
        self.assertIn("key.push_back(fc2_subgroup_fold)", actual)
        self.assertIn("context.HasFeature(wgpu::FeatureName::Subgroups)", actual)
        self.assertIn('context.AdapterInfo().vendor == std::string_view{"nvidia"}', actual)
        self.assertIn("program.SetSubgroupSize(32)", actual)
        self.assertIn("key.push_back(grouped_prefill)", actual)
        self.assertEqual(actual.count("QmoeFC2FinalMixProgram program"), 1)

    def test_packed_half_bits_and_independent_ordered_folds(self):
        rng = np.random.default_rng(3110)
        bits = rng.integers(0, 65536, (256, 2), dtype=np.uint16)
        words = bits[:, 0].astype(np.uint32) | (bits[:, 1].astype(np.uint32) << 16)
        self.assertTrue(np.array_equal((words & 65535).astype(np.uint16), bits[:, 0]))
        self.assertTrue(np.array_equal((words >> 16).astype(np.uint16), bits[:, 1]))
        for _ in range(128):
            partials = rng.uniform(-0.01, 0.01, (32, 2)).astype(np.float16)
            partials[20:] = np.float16(0)
            pair = np.zeros(2, dtype=np.float16)
            scalar = [np.float16(0), np.float16(0)]
            for lane in range(32):
                pair += partials[lane]
                for column in range(2):
                    scalar[column] = np.float16(scalar[column] + partials[lane, column])
            self.assertTrue(np.array_equal(pair.view(np.uint16),
                                           np.asarray(scalar, dtype=np.float16).view(np.uint16)))

    def test_model_wiring_preserves_parent_and_window_support_gate(self):
        root = Path(__file__).parent
        benchmark = (root / "benchmark_webgpu_int4.py").read_text()
        self.assertIn("--compare-fc2-subgroup-fold-windows", benchmark)
        self.assertIn("FC2_SUBGROUP_FOLD_FLAG if args.compare_fc2_subgroup_fold_windows", benchmark)
        self.assertIn(FLAG, benchmark)
        runner = (root / "run_hc_up_parallel2_model.py").read_text()
        self.assertIn('parent_variant == "qmoe-prefill-rows4-v2"', runner)
        self.assertIn('command[-1] = "export ORT_WEBGPU_QMOE_GROUPED_PREFILL=1;', runner)
        result = dict(geometric_mean_matched_window_ratio=1.01, matched_window_pairs=16,
                      all_request_output_tokens_match=True,
                      per_request=[dict(repetition=i, fc2_subgroup_fold0_median_window_tps=80.0,
                                        fc2_subgroup_fold1_median_window_tps=81.0 if i < 3 else 79.0)
                                   for i in range(4)])
        verify_fc2_subgroup_window_support(result)
        result["per_request"][2]["fc2_subgroup_fold1_median_window_tps"] = 79.0
        with self.assertRaises(AssertionError):
            verify_fc2_subgroup_window_support(result)


if __name__ == "__main__":
    unittest.main()
