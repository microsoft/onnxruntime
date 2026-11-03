from pathlib import Path
import unittest

from build_fc1_private_a import transform


class PrivateFC1Tests(unittest.TestCase):
    def test_keeps_tile_accumulation_fold_and_activation_order(self):
        original = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill256/build-v1/overlay/contrib_ops/webgpu/moe/qmoe_fc1_activation.inc").read_text()
        candidate = transform(original)
        for fragment in ("for (var kidx = 0u; kidx < uniforms.K; kidx += 1024u)",
                         "sum += dot(cached_a[i * 2u], w0) + dot(cached_a[i * 2u + 1u], w1)",
                         "partials[offset + group][lane] += sum",
                         "for (var i = 0u; i < 32u; i++) { value += partials[local_idx][i]; }",
                         "let glu = gate_value / (1.0 + exp(-uniforms.alpha * gate_value))"):
            self.assertIn(fragment, candidate)
        self.assertIn("cached_a[i] = vec4<f16>(0)", candidate)
        self.assertIn("if (offset < uniforms.K / 4u)", candidate)
        self.assertIn('private_a_ ? "workgroupBarrier();\\n" : ""', candidate)
        self.assertIn('if (!private_a_)', candidate)


if __name__ == "__main__":
    unittest.main()
