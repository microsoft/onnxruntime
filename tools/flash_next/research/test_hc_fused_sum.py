from pathlib import Path
import unittest

from build_hc_fused_sum import FLAG, transform


class HCFusedSumTests(unittest.TestCase):
    def test_scoped_fusion_preserves_rounding_and_reductions(self):
        source = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/build-v2/"
                      "hc_post_norm_projection.cc").read_text()
        actual = transform(source)
        start = source.index("class HCPostNormProjectionProgram final")
        self.assertEqual(source[:start], actual[:start])
        self.assertIn("HCPostNormProjectionFusedSumF16", actual)
        self.assertIn('code << "let rms_value" << stream << " = unpack2x16float(pack2x16float', actual)
        self.assertIn("if (!fused_sum_)", actual)
        self.assertIn("}, 0, fused_sum);", actual)
        self.assertIn(f'"{FLAG} must be 0 or 1"', actual)
        for expression in (
                "let stored", " = f16(value", "for (var width = 128u; width > 0u; width >>= 1u)",
                "sum += dot(x0, w0) + dot(x1, w1)", "total += sum;",
                "for (var i = 0u; i < 32u; i++) { value += partials[local_idx + i]; }"):
            self.assertEqual(source.count(expression), actual.count(expression), expression)
        self.assertEqual(actual.count('code << "var sum"'), 2)
        self.assertLess(actual.index('if (fused_sum_)'), actual.index('code << "var sum"'))
        self.assertEqual(source[source.index(" protected:"):], actual[actual.index(" protected:"):])


if __name__ == "__main__":
    unittest.main()
