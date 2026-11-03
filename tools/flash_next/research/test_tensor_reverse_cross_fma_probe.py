from pathlib import Path
import unittest

from build_tensor_cross_fma_probe import transform as cross_fma
from build_tensor_reverse_cross_fma_probe import transform


class ReverseCrossFmaTests(unittest.TestCase):
    def test_both_operands_reverse_first_product_positions_without_changing_pairs(self):
        source = Path("/home/sayanshaw/repos/qwen38-int4-dense-block32/build-v1/matmul_nbits.cc").read_text()
        original = cross_fma(source, materialize=True)
        actual = transform(source)
        for old, new in (
            ('"TensorCrossFmaProbeF16"', '"TensorReverseCrossFmaProbeF16"'),
            ('a.GetByOffset("row")', 'a.GetByOffset("(row ^ 2u)")'),
            ('a.GetByOffset("col")', 'a.GetByOffset("(col ^ 2u)")'),
            ("((row % 8u) * 4u)", "(((row ^ 2u) % 8u) * 4u)"),
        ):
            self.assertIn(new, actual)
            actual = actual.replace(new, old, 1)
        self.assertEqual(actual, original)
        self.assertEqual([row ^ 2 for row in range(16)],
                         [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13])


if __name__ == "__main__":
    unittest.main()
