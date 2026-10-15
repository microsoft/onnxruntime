import tempfile
import unittest
from pathlib import Path

import numpy as np
from qmoe_expert_activation_correlations import (
    ActivationTrace,
    binary_activation_correlation,
    compute_activation_correlations,
    write_correlation_archive,
)


class TestQMoEExpertActivationCorrelations(unittest.TestCase):
    def test_binary_activation_correlation(self):
        activations = np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1],
            ],
            dtype=np.uint8,
        )
        correlations, current_counts, context_counts = binary_activation_correlation(activations, activations)

        np.testing.assert_allclose(correlations, np.eye(2))
        np.testing.assert_array_equal(current_counts, [2, 2])
        np.testing.assert_array_equal(context_counts, [2, 2])

    def test_correlates_previous_token_without_crossing_prompt_boundaries(self):
        identities = (
            ("main", 1, "QMoE", "/model/layers.0/moe/QMoE"),
            ("main", 2, "QMoE", "/model/layers.1/moe/QMoE"),
        )
        prompt = np.array(
            [
                [[1, 0], [1, 0]],
                [[0, 1], [0, 1]],
                [[1, 0], [1, 0]],
                [[0, 1], [0, 1]],
            ],
            dtype=np.uint8,
        )
        trace = ActivationTrace(identities, (prompt, prompt), expert_count=2)

        same_qmoe, previous_qmoe = compute_activation_correlations(trace)

        self.assertEqual(same_qmoe[0].samples, 6)
        np.testing.assert_allclose(same_qmoe[0].values, [[-1, 1], [1, -1]])
        self.assertEqual(len(previous_qmoe), 1)
        self.assertEqual(previous_qmoe[0].samples, 8)
        np.testing.assert_allclose(previous_qmoe[0].values, [[1, -1], [-1, 1]])

    def test_constant_activation_has_undefined_correlation(self):
        current = np.array([[1], [1], [1]], dtype=np.uint8)
        context = np.array([[0, 1], [1, 0], [0, 1]], dtype=np.uint8)

        correlations, _, _ = binary_activation_correlation(current, context)

        self.assertTrue(np.isnan(correlations).all())

    def test_archive_supports_trace_without_previous_layer(self):
        identity = ("main", 1, "QMoE", "/model/layers.0/moe/QMoE")
        prompt = np.array([[[1, 0]], [[0, 1]], [[1, 0]]], dtype=np.uint8)
        same_qmoe, previous_qmoe = compute_activation_correlations(
            ActivationTrace((identity,), (prompt,), expert_count=2)
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "correlations.npz"
            write_correlation_archive(path, same_qmoe, previous_qmoe)
            with np.load(path) as archive:
                self.assertEqual(archive["same_qmoe_previous_token"].shape, (1, 2, 2))
                self.assertEqual(archive["previous_qmoe_same_token"].shape, (0, 2, 2))
                self.assertEqual(archive["previous_qmoe_current_activation_counts"].shape, (0, 2))


if __name__ == "__main__":
    unittest.main()
