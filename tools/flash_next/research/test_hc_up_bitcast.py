from pathlib import Path
import unittest

from build_hc_up_bitcast import FLAG, transform


class HCUpBitcastTests(unittest.TestCase):
    def test_scoped_conversion_and_variant_key(self):
        source = Path("/home/sayanshaw/repos/qwen38-int4-hc-up-premix/build-v2/hc_up_premix.cc").read_text()
        actual = transform(source)
        self.assertIn("HCUpPreMixBitcastF16", actual)
        self.assertIn("decode_low_nibbles(packed[i] >> 4u)", actual)
        self.assertIn(f'"{FLAG} must be 0 or 1"', actual)
        self.assertIn("}, 0, bitcast);", actual)
        for expression in ("if (lane < 10u)", "sum += dot(x0, w0) + dot(x1, w1)",
                           "total += sum;", "value += partials[local_idx + i]",
                           "sum * uniforms.reduction_scale", "workgroupBarrier()"):
            self.assertEqual(source.count(expression), actual.count(expression), expression)
        start = source.index("class HCUpPreMixF16")
        tail = source.index("\n private:", start)
        self.assertEqual(source[tail:], actual[actual.index("\n private:", actual.index("class HCUpPreMixF16")):])


if __name__ == "__main__":
    unittest.main()
