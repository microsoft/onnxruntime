from pathlib import Path
import unittest

from build_qmoe_fc1_activation import ACTIVATED, FLAG, FUSED, transform


ROOT = Path(__file__).parent
SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-node-replay/build-v5/overlay/contrib_ops/webgpu/moe/qmoe.cc")


class QmoeFC1ActivationTests(unittest.TestCase):
    def test_scoped_rewrite_and_variant_key(self):
        source = SOURCE.read_text()
        result = transform(source)
        self.assertIn(f'"{FLAG} must be 0 or 1"', result)
        self.assertIn("key.push_back(fc1_activation)", result)
        self.assertIn("expected_commands = CanFuseFc1Activation(context, fc1_activation == 1) ? 4 : 5", result)
        self.assertEqual(result.count(ACTIVATED), 1)
        self.assertEqual(result.count(FUSED), 1)
        before = source[source.index("    status = is_nvfp4_ || use_int4_cooperative",
                                     source.index("if (use_indirect_experts)")):
                        source.index("    TensorShape fc2_output_shape")]
        self.assertIn(before.replace(ACTIVATED, ""), result)
        self.assertEqual(result[result.index("    TensorShape fc2_output_shape"):],
                         source[source.index("    TensorShape fc2_output_shape"):])

    def test_projection_rounding_and_interleaved_activation(self):
        shader = (ROOT / "qmoe_fc1_activation.inc").read_text()
        self.assertIn("partials[local_idx][0] = value", shader)
        self.assertIn("f32(partials[local_idx * 2u][0])", shader)
        self.assertIn("f32(partials[local_idx * 2u + 1u][0])", shader)
        for expression in ("gate_value = min(gate_value, uniforms.swiglu_limit)",
                           "linear_value = clamp(linear_value, -uniforms.swiglu_limit, uniforms.swiglu_limit)",
                           "gate_value / (1.0 + exp(-uniforms.alpha * gate_value))",
                           "f16(glu * (linear_value + uniforms.beta))"):
            self.assertIn(expression, shader)

    def test_intermediate_ownership_and_pairing(self):
        owners = {(offset + lane // 32, lane % 32) for lane in range(128) for offset in (0, 4)}
        self.assertEqual(len(owners), 256)
        self.assertEqual([(2 * lane, 2 * lane + 1) for lane in range(4)],
                         [(0, 1), (2, 3), (4, 5), (6, 7)])

    def test_only_qualified_shape_and_math(self):
        source = (ROOT / "qmoe_fc1_activation.inc").read_text()
        for guard in ("const auto* scales = context.Input<Tensor>(3);",
                      "const auto* bias = context.Input<Tensor>(4);",
                      "MoEActivationType::SwiGLU", "fc1_expert_weight_bits_ == 4",
                      "block_size_ == 32", "input->Shape().Size() == 2560",
                      "k_ == 10", "!context.EnableMatmulFp32Accumulation()",
                      "weights->Shape().NumDimensions() == 3", "weights->Shape().GetDims()[2] == 1280",
                      "scales->Shape().NumDimensions() == 3", "scales->Shape().GetDims()[2] == 80",
                      "context.Input<Tensor>(8) == nullptr", "context.Input<Tensor>(11) == nullptr",
                      "context.Input<Tensor>(14) == nullptr"):
            self.assertIn(guard, source)


if __name__ == "__main__":
    unittest.main()
