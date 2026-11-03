import unittest
from pathlib import Path

from build_dense_output_shape_static import FLAG, KERNEL, comparator_transform, header_transform, transform
from prepare_dense_output_shape_static_model import ROOT, model_source, screen_source
from prepare_dense_output_shape_static_ranks import transform as rank_transform


class StaticShapeTests(unittest.TestCase):
    def test_setting_is_constructor_only(self):
        root = Path("/home/sayanshaw/repos")
        header = (root / "qwen38-int4-static-node-replay/build-v8/overlay/"
                  "contrib_ops/webgpu/quantization/matmul_nbits.h").read_text()
        header = header.replace("MatMulNBits(", KERNEL + "(")
        generated = header_transform(header)
        self.assertEqual(generated.count(f'"{FLAG}", 0'), 1)
        self.assertIn("output_shape_static_ = output_shape_static == 1", generated)
        source = transform((root / "qwen38-int4-qmoe-prefill-rows4/build-v2/matmul_nbits.cc").read_text())
        self.assertNotIn(f'"{FLAG}"', source)
        self.assertIn("if (output_shape_static_ && K_ > 0", source)
        self.assertNotIn("  TensorShape output_shape;\n", source)
        self.assertIn("auto output_shape = helper.OutputShape();\n    y = context.Output(0, output_shape);", source)
        self.assertIn("auto output_shape = a_shape;", source)

    def test_two_sessions_and_constructor_rejection(self):
        from build_dense_double_buffer import transform_compare
        from build_qmoe_node_replay import replace_once
        original = Path("/home/sayanshaw/repos/qwen38-int4-dense-block32/build-v1/native_compare.cpp").read_text()
        source = transform_compare(replace_once(original,
            'setenv("ORT_WEBGPU_DENSE_BLOCK32", value, 1)', f'setenv("{FLAG}", value, 1)'), flag=FLAG)
        generated = comparator_transform(source)
        self.assertIn('flag("0");\n      Ort::Session session', generated)
        self.assertIn('flag("1");\n      Ort::Session candidate_session', generated)
        self.assertIn("auto actual = candidate_session.Run", generated)
        self.assertIn("Ort::Session invalid_session{env, argv[file], options};", generated)

    def test_actual_rank_proof_counts_both_sessions(self):
        source = rank_transform((ROOT / "run_dense_output_shape_ranks.py").read_text())
        self.assertIn("assert len(rows) == 72", source)
        self.assertIn('r["requests"] == 128', source)
        self.assertIn("assert len(shapes) == 72", source)
        self.assertIn('sum(r["fast_shape_hits"] == 0 for r in shapes) == 45', source)

    def test_original_provider_controls_and_actual_constructor_counters(self):
        model = model_source((ROOT / "run_dense_output_shape_model.py").read_text())
        screen = screen_source((ROOT / "run_dense_output_shape_screen.py").read_text())
        self.assertIn("native = original_native", model)
        self.assertIn("constructor_scoped_experiment=True", model)
        self.assertIn("ORT_WEBGPU_DENSE_OUTPUT_SHAPE_STATIC_STATS_FILE", model)
        self.assertIn('candidate_native["provider_sha256"] if "-screen-" in path.name', model)
        self.assertIn("from run_dense_output_shape_static_model import", screen)
        self.assertIn("bracketed_separate_load_original_provider_net_screen", screen)
        self.assertIn('providers["control"] != providers["shape"]', screen)


if __name__ == "__main__":
    unittest.main()
