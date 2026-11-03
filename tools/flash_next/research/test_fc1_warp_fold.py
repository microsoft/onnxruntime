from pathlib import Path
import unittest
from unittest.mock import patch

import build_fc1_warp_fold as builder


REPOS = Path("/home/sayanshaw/repos")
OVERLAY = REPOS / "qwen38-int4-qmoe-prefill256/build-v1/overlay/contrib_ops/webgpu/moe"


class Fc1WarpFoldTests(unittest.TestCase):
    def test_each_original_column_has_exactly_one_warp_leader(self):
        owners = {(offset + thread // 32): thread
                  for thread in range(128) if thread % 32 == 0 for offset in (0, 4)}
        self.assertEqual(sorted(owners), list(range(8)))
        self.assertEqual([owners[column] for column in range(8)], [0, 32, 64, 96] * 2)

    def test_original_dot_activation_shared_stores_and_fold_order_remain(self):
        original = (OVERLAY / "qmoe_fc1_activation.inc").read_text()
        candidate = builder.transform_program(original)
        start = original.index('         << "if (local_idx < 8u && base_col + local_idx')
        stop = original.index('         << "workgroupBarrier();\\n"', start)
        self.assertIn(original[start:stop], candidate)
        self.assertIn("for (var i = 0u; i < 32u; i++) { value += partials[fold_col][i]; }", candidate)
        self.assertIn("partials[fold_col][0] = value;", candidate)
        for line in original.splitlines():
            if "sum +=" in line or "dot(" in line or "output.SetByOffset" in line or "gate_value =" in line:
                self.assertIn(line, candidate)
        self.assertEqual(candidate.count("workgroupBarrier();"), original.count("workgroupBarrier();"))
        self.assertIn("array<array<f16, 32>, 8>", candidate)

    def test_selector_key_and_original_128_thread_dispatch_are_preserved(self):
        original = (REPOS / "qwen38-int4-qmoe-prefill1024/build-v1/qmoe.cc").read_text()
        with patch.object(builder, "OVERLAY", OVERLAY):
            candidate = builder.transform(original)
        self.assertEqual(candidate.count(f'ParseEnvironmentVariableWithDefault<int>("{builder.FLAG}", 0)'), 2)
        self.assertEqual(candidate.count(f'"{builder.FLAG} must be 0 or 1"'), 2)
        self.assertIn("key.push_back(fc1_warp_fold)", candidate)
        self.assertIn("QmoeFC1WarpFoldBitcastF16", candidate)
        self.assertEqual(candidate.count(".SetWorkgroupSize(128)"), original.count(".SetWorkgroupSize(128)"))
        self.assertIn("input->Shape().Size() == 2560", candidate)

    def test_native_comparator_toggles_only_new_selector(self):
        with patch.object(builder, "COMPARATOR", REPOS / "qwen38-int4-expert-bitcast-unpack/build-v2/native_compare.cpp"):
            source = builder.transform_comparator("")
        self.assertEqual(source.count(f'"{builder.FLAG}"'), 4)
        self.assertNotIn('"ORT_WEBGPU_EXPERT_BITCAST_UNPACK"', source)

    def test_actual_model_window_selector_and_gates_are_wired(self):
        benchmark = Path(__file__).with_name("benchmark_webgpu_int4.py").read_text()
        self.assertIn("--compare-fc1-warp-fold-windows", benchmark)
        self.assertIn("fc1_warp_fold_decode_windows", benchmark)
        self.assertIn("FC1_WARP_FOLD_FLAG if args.compare_fc1_warp_fold_windows", benchmark)
        self.assertIn(builder.FLAG, benchmark)
        runner = Path(__file__).with_name("run_hc_up_parallel2_model.py").read_text()
        self.assertIn('variant in ("fc1-warp-fold-v1", "fc1-warp-fold-v3", "fc2-paired-dot-v1")', runner)
        self.assertIn('else "warp_fold_tested"', runner)
        self.assertIn('native[mode_field] == [0, 1]', runner)

    def test_vec2_maps_independent_original_ordered_folds_without_cross_column_reduction(self):
        source = builder.transform_program((OVERLAY / "qmoe_fc1_activation.inc").read_text(), vector_fold=True)
        self.assertIn("let fold_col = group * 2u;", source)
        self.assertIn("for (var i = 0u; i < 32u; i++)", source)
        self.assertIn("values += vec2<f16>(partials[fold_col][i], partials[fold_col + 1u][i])", source)
        self.assertIn("partials[fold_col][0] = values.x;", source)
        self.assertIn("partials[fold_col + 1u][0] = values.y;", source)
        self.assertIn("base_col + fold_col + 1u < uniforms.cols * 2u", source)
        self.assertIn("QmoeFC1WarpFoldVec2BitcastF16", source)
        self.assertNotIn("code <", source.replace("code <<", "code_stream"))
        self.assertNotIn("<;", source)
        self.assertIn('    code          << "workgroupBarrier();\\n"', source)
        for columns in range(2, 66, 2):
            owners = [base + group * 2 + component for base in range(0, columns, 8)
                      for group in range(4) if base + group * 2 + 1 < columns
                      for component in range(2)]
            self.assertEqual(owners, list(range(columns)))

    def test_default_transform_remains_byte_identical_to_built_v1(self):
        original = (REPOS / "qwen38-int4-qmoe-prefill1024/build-v1/qmoe.cc").read_text()
        with patch.object(builder, "OVERLAY", OVERLAY):
            source = builder.transform(original)
        self.assertEqual(source, (REPOS / "qwen38-int4-fc1-warp-fold/build-v1/qmoe.cc").read_text())


if __name__ == "__main__":
    unittest.main()
