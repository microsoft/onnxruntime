# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Host tests for the independently decoded references and fixture layouts."""

import unittest

import numpy as np
from reference import assert_fp32_accuracy, errors, pack_nibbles, pack_zero_points, projection


class ReferenceTest(unittest.TestCase):
    def test_all_nibbles_and_signed_offset(self):
        q = np.tile(np.arange(16, dtype=np.uint8), (3, 2)).reshape(3, 1, 32)
        a = np.arange(32, dtype=np.float16).reshape(1, 32) / 32
        scales = np.asarray([[0.125], [-0.25], [0]], np.float16)
        full, rounded, _ = projection(a, pack_nibbles(q), scales, 32, 32)
        expected = np.asarray(
            [[sum(float(a[0, k]) * (int(q[n, 0, k]) - 8) * float(scales[n, 0]) for k in range(32)) for n in range(3)]]
        )
        np.testing.assert_array_equal(full, expected)
        np.testing.assert_array_equal(rounded, expected)

    def test_per_column_zero_point_padding(self):
        zeros = np.arange(9, dtype=np.uint8).reshape(3, 3)
        packed = pack_zero_points(zeros)
        np.testing.assert_array_equal(packed, [[0x10, 0x02], [0x43, 0x05], [0x76, 0x08]])
        q = np.full((3, 3, 32), 15, np.uint8)
        full, _, _ = projection(
            np.ones((1, 96), np.float16), pack_nibbles(q), np.ones((3, 3), np.float16), 96, 32, packed
        )
        expected = [[sum(32 * (15 - int(z)) for z in row) for row in zeros]]
        np.testing.assert_array_equal(full, expected)

    def test_nonmultiple_k_and_block64(self):
        q = np.full((2, 2, 64), 15, np.uint8)
        a = np.zeros((1, 100), np.float16)
        a[0, -1] = 1
        scales = np.asarray([[0.125, 0.25], [0.25, 0.5]], np.float16)
        zeros = np.asarray([[7, 6], [5, 4]], np.uint8)
        full, _, _ = projection(a, pack_nibbles(q), scales, 100, 64, pack_zero_points(zeros))
        np.testing.assert_array_equal(full, [[(15 - 6) * 0.25, (15 - 4) * 0.5]])

    def test_half_dequantization_is_separate_from_accumulation(self):
        q = np.full((3, 1, 32), 15, np.uint8)
        a = np.ones((1, 32), np.float16)
        scales = np.full((3, 1), 1 / 3, np.float16)
        full, rounded, _ = projection(a, pack_nibbles(q), scales, 32, 32)
        self.assertNotEqual(full[0, 0], rounded[0, 0])
        self.assertEqual(full[0, 0], 32 * 7 * float(scales[0, 0]))
        self.assertEqual(rounded[0, 0], 32 * float(np.float16(7 * float(scales[0, 0]))))

    def test_single_scale_matches_existing_zero_broadcast(self):
        q = np.full((2, 1, 32), 15, np.uint8)
        zeros = pack_zero_points(np.asarray([[3], [7]], np.uint8))
        full, _, _ = projection(
            np.ones((1, 32), np.float16), pack_nibbles(q), np.asarray([0.125], np.float16), 32, 64, zeros
        )
        np.testing.assert_array_equal(full, [[48, 48]])

    def test_rank_and_flat_scale_layouts(self):
        q = np.full((3, 2, 32), 9, np.uint8)
        a = np.ones((2, 1, 2, 63), np.float16)
        full, _, _ = projection(a, pack_nibbles(q), np.ones(6, np.float16), 63, 32)
        self.assertEqual(full.shape, (2, 1, 2, 3))
        np.testing.assert_array_equal(full, np.full(full.shape, 63))

    def test_bias_is_added_after_k_sum(self):
        q = np.full((2, 1, 32), 8, np.uint8)
        bias = np.asarray([0.25, -0.5], np.float16)
        full, rounded, _ = projection(
            np.ones((32,), np.float16), pack_nibbles(q), np.ones((2, 1), np.float16), 32, 32, bias=bias
        )
        np.testing.assert_array_equal(full, bias)
        np.testing.assert_array_equal(rounded, bias)

    def test_forward_bound_includes_final_half_rounding(self):
        reference = np.asarray([[1.001, -0.123]], np.float64)
        assert_fp32_accuracy(reference.astype(np.float16), reference, np.abs(reference), 5120)
        with self.assertRaises(AssertionError):
            assert_fp32_accuracy(reference + 1, reference, np.abs(reference), 5120)

    def test_nonfinite_is_reported_and_rejected(self):
        record = errors(np.asarray([np.inf]), np.asarray([1]))
        self.assertFalse(record["finite"])
        self.assertEqual(record["nonfinite"], 1)
        record = errors(np.asarray([1]), np.asarray([np.inf]))
        self.assertFalse(record["finite"])
        self.assertEqual(record["nonfinite"], 0)
        self.assertEqual(record["expected_nonfinite"], 1)
        with self.assertRaises(AssertionError):
            assert_fp32_accuracy(np.asarray([np.inf]), np.asarray([1]), np.asarray([1]), 32)

    def test_malformed_quantization_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            pack_nibbles(np.asarray([16, 0], np.uint8))
        with self.assertRaises(ValueError):
            pack_zero_points(np.asarray([[1, 2]], np.float32))
        with self.assertRaises(ValueError):
            projection(np.ones((1, 32), np.float16), np.zeros((2, 16), np.uint8), np.ones(3, np.float16), 32, 32)


if __name__ == "__main__":
    unittest.main()
