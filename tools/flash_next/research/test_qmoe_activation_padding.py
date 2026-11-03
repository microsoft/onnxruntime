from collections import Counter
from pathlib import Path
import unittest

from build_qmoe_activation_padding import FLAG, transform, transform_shader


ROOT = Path(__file__).parent
SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-fc2-final-mix/build-v2/overlay/contrib_ops/webgpu/moe/qmoe.cc")


class ActivationPaddingTests(unittest.TestCase):
    def test_variant_key_and_capability_guard(self):
        source = transform(SOURCE.read_text())
        self.assertIn("key.push_back(activation_padding)", source)
        self.assertIn(f'"{FLAG} must be 0 or 1"', source)
        self.assertIn("maxComputeWorkgroupStorageSize >= 15680", source)
        for number in (1, 2):
            self.assertIn(f".CacheHint(fc{number}_experts_bias_optional != nullptr, padded_activations)", source)
        self.assertEqual(1800 * 8 + 20 * 32 * 2, 15680)

    def test_both_shaders_preserve_math_and_loop_bounds(self):
        for number, name in ((1, "qmoe_fc1_activation.inc"), (2, "qmoe_fc2_final_mix.inc")):
            source = (ROOT / name).read_text()
            padded = transform_shader(source, number)
            self.assertIn("ActivationPaddedF16", padded)
            self.assertIn("var sum = f16(0)", padded)
            self.assertIn("for (var i = 0u; i < 32u; i++)", padded)
            self.assertEqual(source.count("workgroupBarrier()"), padded.count("workgroupBarrier()"))
            self.assertEqual(source.count("dot("), padded.count("dot("))
            if number == 2:
                self.assertIn("for (var r = 0u; r < uniforms.k; r++)", padded)
                self.assertIn("acc += router_value * partials[r * 2u + local_idx][0]", padded)
            else:
                self.assertIn("let glu = gate_value / (1.0 + exp(-uniforms.alpha * gate_value))", padded)
                self.assertIn("} else { tile_a[", padded)

    def test_staged_cells_and_bank_distribution(self):
        for blocks in (32, 200):
            staged = {(index // 8) * 9 + index % 8 for index in range(blocks * 8)}
            reads = {block * 9 + offset for block in range(blocks) for offset in range(8)}
            self.assertEqual(staged, reads)
            self.assertLess(max(staged), blocks * 9)
        banks = Counter((lane * 18 + word) % 32 for lane in range(32) for word in range(2))
        self.assertEqual(len(banks), 32)
        self.assertEqual(max(banks.values()), 2)


if __name__ == "__main__":
    unittest.main()
