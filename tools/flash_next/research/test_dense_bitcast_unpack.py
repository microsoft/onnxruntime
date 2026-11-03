from pathlib import Path
import unittest

from build_dense_bitcast_unpack import transform, transform_replay, transform_template


REPOS = Path("/home/sayanshaw/repos")
FROZEN = REPOS / "qwen38-int4-static-node-replay/build-v8/overlay"
GENERATED = REPOS / "onnxruntime-webgpu-fp8-checkpoint/build-webgpu-fp8/Release/wgsl_generated/wgsl_template_gen"


class DenseBitcastTests(unittest.TestCase):
    def test_isolated_program_and_original_fallback(self):
        source = (REPOS / "qwen38-int4-generic-matmul-replay/build-v1/matmul_nbits.cc").read_text()
        header = (FROZEN / "contrib_ops/webgpu/quantization/matmul_nbits.h").read_text()
        result = transform(source, header)
        self.assertIn('Program{"MatMulNBitsBitcastF16"}', result)
        self.assertIn("return static_replay_.Run(context, {y}, compute, 0, dense_bitcast_route);", result)
        original_function = source[source.index("Status ApplyMatMulNBits("):source.index("\n}  // namespace webgpu")]
        self.assertIn(original_function, result)
        self.assertIn("a->Shape().Size() == K_ && K_ % 32 == 0", result)
        self.assertEqual(result.count("  DenseBitcastProgram program{"), 1)
        self.assertEqual(result.count("  MatMulNBitsProgram program{"), 1)
        self.assertIn("ORT_WEBGPU_DENSE_BITCAST_UNPACK must be 0 or 1", result)

    def test_frozen_template_only_changes_four_unpack_expressions(self):
        source = (GENERATED / "generated/quantization/matmul_nbits.h").read_text()
        strings = (GENERATED / "string_table.h").read_text()
        result = transform_template(source, strings)
        self.assertEqual(result.count("= decode_low_nibbles("), 4)
        self.assertEqual(result.count("fn decode_low_nibbles("), 1)
        self.assertNotIn("__str_", result)
        self.assertNotIn("MainFunctionStart", result)
        self.assertIn("Status ApplyDenseBitcastTemplate(", result)
        self.assertIn("sum += dot", result)

    def test_replay_key_discriminates_variants_without_layout_changes(self):
        source = (FROZEN / "static_node_replay.h").read_text()
        result = transform_replay(source)
        self.assertIn("uint64_t variant = 0", result)
        self.assertIn("key.push_back(variant);", result)
        self.assertEqual(source[source.index("\n private:"):], result[result.index("\n private:"):])


if __name__ == "__main__":
    unittest.main()
