import unittest

import numpy as np

from reference_int4_gdn import recurrent_reference


class RecurrentReferenceTests(unittest.TestCase):
    def test_zero_inputs(self):
        output, state = recurrent_reference(
            np.zeros((1, 2, 6)), np.zeros((1, 2, 1)), np.zeros((1, 2, 1)),
            np.zeros(1), np.zeros(1), np.zeros((1, 1, 2, 2)))
        np.testing.assert_array_equal(output, np.zeros((2, 1, 2)))
        np.testing.assert_array_equal(state, np.zeros((1, 1, 2, 2)))

    def test_two_analytically_computed_updates(self):
        output, state = recurrent_reference(
            np.array([[[1, 0, 1, 0, 2, 0], [1, 0, 1, 0, 0, 0]]]),
            np.zeros((1, 2, 1)), np.zeros((1, 2, 1)),
            np.zeros(1), np.zeros(1), np.zeros((1, 1, 2, 2)))
        np.testing.assert_array_equal(output[:, 0, 0],
                                      np.array([1 / np.sqrt(2), .25 / np.sqrt(2)], dtype=np.float16))
        np.testing.assert_allclose(state, [[[[.25, 0], [0, 0]]]], atol=1e-7)

    def test_invalid_batch_rejected(self):
        with self.assertRaisesRegex(ValueError, "batch1"):
            recurrent_reference(np.zeros((1, 1, 6)), np.zeros((1, 1, 1)),
                                np.zeros((1, 1, 1)), np.zeros(1), np.zeros(1),
                                np.zeros((2, 1, 2, 2)))
