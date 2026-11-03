import unittest
from pathlib import Path

from build_dense_output_shape import FLAG, ORIGINAL, SHAPE, transform
from prepare_dense_output_shape_model import ROOT, model_source, transform as prepare_transform
from prepare_dense_output_shape_net_screen import transform as net_transform


class OutputShapeTests(unittest.TestCase):
    def test_narrow_guard_and_unchanged_fallback(self):
        source = transform(Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/"
                                "build-v2/matmul_nbits.cc").read_text())
        self.assertIn(f'"{FLAG}", 0', source)
        self.assertIn("output_shape_fast == 0 || output_shape_fast == 1", source)
        self.assertIn("K_ > 0 && a_shape.NumDimensions() >= 2", source)
        self.assertIn("a_shape.Size() == K_", source)
        self.assertIn("a_shape[a_shape.NumDimensions() - 1] == K_", source)
        self.assertIn("helper.Compute(a_shape, b_shape, false, true)", source)
        self.assertEqual(source.count("Tensor* y = context.Output(0, output_shape);"), 1)
        self.assertEqual(source.count(ORIGINAL), 0)
        self.assertEqual(source.count(SHAPE), 1)

    def test_flattened_reference_shape_and_offsets(self):
        helper = Path("/home/sayanshaw/repos/onnxruntime-webgpu-h200-perf/onnxruntime/"
                      "core/providers/cpu/math/matmul_helper.h").read_text()
        reference = helper[helper.index("      M_ = static_cast<ptrdiff_t>"):
                           helper.index("      return Status::OK();")]
        self.assertIn("output_shape_ = orig_left_shape;", reference)
        self.assertIn("output_shape_[left_num_dims - 1] = N_;", reference)
        for offsets in ("output_offsets_", "left_offsets_", "right_offsets_"):
            self.assertIn(f"{offsets} = {{0}};", reference)
        self.assertNotIn("offsets_", SHAPE)
        for rank in range(2, 9):
            for k in (1, 32, 640, 2560, 6144, 10240):
                for n in (1, 320, 2560, 16384):
                    shape = [1] * (rank - 1) + [k]
                    expected = shape[:-1] + [n]
                    actual = shape.copy()
                    actual[-1] = n
                    self.assertEqual(actual, expected)

    def test_model_family_requires_actual_shape_and_rank_proofs(self):
        source = model_source((ROOT / "run_dense_replay_scratch_model.py").read_text())
        screen = prepare_transform((ROOT / "run_dense_replay_scratch_screen.py").read_text())
        self.assertNotIn("scratch", source.lower())
        self.assertNotIn("scratch", screen.lower())
        self.assertIn("from build_dense_output_shape import FLAG", source)
        self.assertIn("dense_output_shape_changing_inputs_finite_bitwise_qualified", source)
        self.assertIn("dense_output_shape_rank1_fallback_rank234_finite_bitwise_qualified", source)
        self.assertIn('"MatMulNBitsOutputShapeExact"', source)
        self.assertIn("qwen38-int4-dense-output-shape/build-v2", source)
        self.assertEqual(source.count("export ORT_WEBGPU_DENSE_OUTPUT_SHAPE_STATS_FILE="), 1)
        self.assertIn('prefix.name + "-static-stats.jsonl"', source)
        self.assertIn('prefix.name + "-shape-stats.jsonl"', source)
        self.assertIn("from run_dense_output_shape_model import", screen)

    def test_original_provider_net_screen(self):
        source = net_transform(prepare_transform((ROOT / "run_dense_replay_scratch_screen.py").read_text()))
        self.assertIn("from run_dense_output_shape_net_model import", source)
        self.assertIn('providers["control"] != providers["shape"]', source)
        self.assertIn('original_baseline_provider_sha256=providers["control"]', source)
        self.assertIn("0955e1d6cc30cac13044450f692100ffa5f33361aaef229e25ff0efa20a8c8d4", source)
        self.assertIn('proof["original_provider_control"] == (mode == "control")', source)


if __name__ == "__main__":
    unittest.main()
