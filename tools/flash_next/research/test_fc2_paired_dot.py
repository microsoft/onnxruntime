from pathlib import Path
import unittest
from unittest.mock import patch

import build_fc2_paired_dot as builder


REPOS = Path("/home/sayanshaw/repos")
OVERLAY = REPOS / "qwen38-int4-qmoe-prefill256/build-v1/overlay/contrib_ops/webgpu/moe"


class Fc2PairedDotTests(unittest.TestCase):
    def test_original_fallback_folds_mix_barriers_and_geometry_are_preserved(self):
        original = (OVERLAY / "qmoe_fc2_final_mix.inc").read_text()
        source = builder.transform_program(original)
        start = original.index('         << "  for (var c = 0u; c < 2u; c++)')
        stop = original.index('         << "}\\n"\n         << "workgroupBarrier();\\n"', start)
        self.assertIn(original[start:stop], source)
        fold = original.index('         << "if (local_idx < 20u)')
        end = original.index("    return Status::OK();", fold)
        self.assertIn(original[fold:end], source)
        self.assertEqual(source.count("workgroupBarrier();"), original.count("workgroupBarrier();"))
        self.assertIn("array<vec4<f16>, 1600>", source)
        self.assertIn("array<array<f16, 32>, 20>", source)
        self.assertNotIn("<;", source)
        self.assertNotIn("code <", source.replace("code <<", "code_stream"))

    def test_paired_dot_has_independent_sums_and_shared_half_stores(self):
        source = builder.transform_program((OVERLAY / "qmoe_fc2_final_mix.inc").read_text())
        self.assertIn("dot(a0, w00) + dot(a1, w01), dot(a0, w10) + dot(a1, w11)", source)
        self.assertIn("partials[rank * 2u][lane] += sums.x", source)
        self.assertIn("partials[rank * 2u + 1u][lane] += sums.y", source)
        self.assertIn("let index1 = index0 + 20u", source)
        self.assertIn("var i = 0u; i < 4u; i++", source)
        self.assertIn('if (lane < 20u)', source)

    def test_selector_validation_key_and_dispatch_shape_are_wired(self):
        original = (REPOS / "qwen38-int4-qmoe-prefill1024/build-v1/qmoe.cc").read_text()
        with patch.object(builder, "OVERLAY", OVERLAY):
            source = builder.transform(original)
        self.assertEqual(source.count(f'ParseEnvironmentVariableWithDefault<int>("{builder.FLAG}", 0)'), 2)
        self.assertIn("key.push_back(fc2_paired_dot)", source)
        self.assertIn(".SetWorkgroupSize(320).SetDispatchGroupSize(1280)", source)
        self.assertIn("QmoeFC2PairedDotBitcastF16", source)

    def test_model_window_selector_and_native_variant_gate_are_wired(self):
        root = Path(__file__).parent
        benchmark = (root / "benchmark_webgpu_int4.py").read_text()
        self.assertIn("--compare-fc2-paired-dot-windows", benchmark)
        self.assertIn("FC2_PAIRED_DOT_FLAG if args.compare_fc2_paired_dot_windows", benchmark)
        self.assertIn("fc2_paired_dot_decode_windows", benchmark)
        self.assertIn(builder.FLAG, benchmark)
        runner = (root / "run_hc_up_parallel2_model.py").read_text()
        self.assertIn('"paired_dot_tested" if variant == "fc2-paired-dot-v1"', runner)


if __name__ == "__main__":
    unittest.main()
