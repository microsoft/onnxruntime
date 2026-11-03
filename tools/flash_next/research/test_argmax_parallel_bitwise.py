import math
import struct
import unittest

from build_argmax_parallel_bitwise import PROGRAM


def argmax_bits(values, fp16):
    sign, infinity, mask = (0x8000, 0x7c00, 0xffff) if fp16 else (0x80000000, 0x7f800000, 0xffffffff)
    bits = [int.from_bytes(struct.pack("<e" if fp16 else "<f", value), "little") for value in values]
    if bits[0] & (sign - 1) > infinity:
        return 0
    def key(value):
        normalized = 0 if value & (sign - 1) == 0 else value
        return (~normalized) & mask if normalized & sign else normalized ^ sign
    index = 0
    for i in range(1, len(bits)):
        if bits[i] & (sign - 1) <= infinity and key(bits[i]) > key(bits[index]):
            index = i
    return index


class BitwiseArgMaxTests(unittest.TestCase):
    def test_ieee_order_and_sticky_nan(self):
        for fp16 in (False, True):
            tiny = 2 ** (-24 if fp16 else -149)
            values = [-math.inf, -1.0, -tiny, -0.0, 0.0, tiny, 1.0, math.inf, math.nan]
            for a in values:
                for b in values:
                    for c in (0.0, tiny, math.nan):
                        row = [a, b, c]
                        expected = 0
                        for i in (1, 2):
                            if row[i] > row[expected]:
                                expected = i
                        self.assertEqual(argmax_bits(row, fp16), expected)

    def test_no_floating_comparison_or_shared_float_keys(self):
        self.assertNotIn("var best = f32(", PROGRAM)
        self.assertNotIn("array<f32", PROGRAM)
        self.assertIn("bitcast<u32>(vec2<f16>", PROGRAM)
        self.assertIn("!first_nan && !is_nan(raw)", PROGRAM)
        self.assertIn("normalized", PROGRAM)


if __name__ == "__main__":
    unittest.main()
