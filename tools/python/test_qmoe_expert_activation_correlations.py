import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from qmoe_expert_activation_correlations import (
    ActivationTrace,
    CorrelationMatrix,
    _correlation_tails,
    _positive_and_negative_extremes,
    binary_activation_correlation,
    compute_activation_correlations,
    compute_last_qmoe_previous_token_correlations,
    write_correlation_archive,
    write_correlation_plots,
    write_selected_experts,
)


class TestQMoEExpertActivationCorrelations(unittest.TestCase):
    def test_writes_selected_expert_table(self):
        rows = ((1, 0, 12, 34), (1, 0, 56, 78), (2, 1, 90, 123))
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "selected.csv"
            write_selected_experts(path, rows)
            with path.open(newline="", encoding="utf-8") as stream:
                csv_rows = list(csv.reader(stream))

        self.assertEqual(
            csv_rows,
            [
                ["iteration", "qmoe_number", "expert_number", "position"],
                ["1", "0", "12", "34"],
                ["1", "0", "56", "78"],
                ["2", "1", "90", "123"],
            ],
        )

    def test_selects_positive_and_negative_extremes(self):
        positive, negative = _positive_and_negative_extremes(
            np.array([[0.1, -0.8, np.nan], [0.7, -0.2, 0.4]]),
            count=2,
        )

        np.testing.assert_allclose(positive, [0.7, 0.4])
        np.testing.assert_allclose(negative, [-0.8, -0.2])

    def test_selects_highest_and_lowest_correlations(self):
        identity = ("main", 1, "QMoE", "/model/layers.0/moe/QMoE")
        correlation = CorrelationMatrix(
            "same_qmoe_previous_token",
            identity,
            identity,
            np.array([[0.1, -0.8], [0.7, -0.2]]),
            np.array([20, 20]),
            np.array([20, 20]),
            100,
        )

        tails = _correlation_tails(correlation, pairs_per_tail=2, min_activations=10)

        self.assertEqual(tails[0], ("highest", 1, 1, 0, 0.7))
        self.assertEqual(tails[1], ("highest", 2, 0, 0, 0.1))
        self.assertEqual(tails[2], ("lowest", 1, 0, 1, -0.8))
        self.assertEqual(tails[3], ("lowest", 2, 1, 1, -0.2))

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

    def test_correlates_first_layer_within_each_graph_scope(self):
        identities = (
            ("main", 1, "QMoE", "/model/layers.0/moe/QMoE"),
            ("main", 2, "QMoE", "/model/layers.2/moe/QMoE"),
            ("main", 3, "QMoE", "/model/layers.4/moe/QMoE"),
            ("nested", 4, "QMoE", "/model/layers.7/moe/QMoE"),
            ("nested", 5, "QMoE", "/model/layers.9/moe/QMoE"),
        )
        prompt = np.array(
            [
                [[1, 0], [1, 0], [0, 1], [0, 1], [1, 0]],
                [[0, 1], [1, 0], [1, 0], [1, 0], [0, 1]],
                [[1, 0], [0, 1], [0, 1], [0, 1], [1, 0]],
                [[0, 1], [0, 1], [1, 0], [1, 0], [0, 1]],
            ],
            dtype=np.uint8,
        )
        trace = ActivationTrace(identities, (prompt, prompt), expert_count=2)

        same_qmoe, first_qmoe = compute_activation_correlations(trace, reference_layer="first")
        last_qmoe = compute_last_qmoe_previous_token_correlations(trace)

        self.assertEqual(len(first_qmoe), 3)
        self.assertEqual([c.context_identity for c in first_qmoe], [identities[0], identities[0], identities[3]])
        self.assertEqual([c.current_identity for c in first_qmoe], [identities[1], identities[2], identities[4]])
        self.assertTrue(all(c.relation == "first_qmoe_same_token" and c.samples == 8 for c in first_qmoe))
        np.testing.assert_allclose(first_qmoe[0].values, np.zeros((2, 2)))
        np.testing.assert_allclose(first_qmoe[1].values, [[-1, 1], [1, -1]])
        np.testing.assert_allclose(first_qmoe[2].values, [[-1, 1], [1, -1]])
        self.assertEqual(same_qmoe[0].samples, 6)
        np.testing.assert_allclose(same_qmoe[0].values, [[-1, 1], [1, -1]])
        self.assertEqual(
            [c.context_identity for c in last_qmoe],
            [identities[2], identities[2], identities[2], identities[4], identities[4]],
        )
        self.assertTrue(all(c.relation == "last_qmoe_previous_token" and c.samples == 6 for c in last_qmoe))
        np.testing.assert_allclose(last_qmoe[0].values, [[1, -1], [-1, 1]])
        np.testing.assert_allclose(last_qmoe[2].values, same_qmoe[2].values)
        np.testing.assert_allclose(last_qmoe[4].values, same_qmoe[4].values)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "first.npz"
            write_correlation_archive(
                path, same_qmoe, first_qmoe, reference_layer="first", last_qmoe_previous_token=last_qmoe
            )
            with np.load(path) as archive:
                self.assertEqual(archive["first_qmoe_same_token"].shape, (3, 2, 2))
                np.testing.assert_allclose(archive["first_qmoe_same_token"][1], [[-1, 1], [1, -1]])
                np.testing.assert_array_equal(archive["first_qmoe_current_activation_counts"], [[4, 4]] * 3)
                np.testing.assert_array_equal(archive["first_qmoe_context_activation_counts"], [[4, 4]] * 3)
                np.testing.assert_array_equal(
                    archive["first_qmoe_context_node_names"], [identities[0][3], identities[0][3], identities[3][3]]
                )
                self.assertNotIn("previous_qmoe_same_token", archive)
                self.assertEqual(archive["last_qmoe_previous_token"].shape, (5, 2, 2))
                np.testing.assert_allclose(archive["last_qmoe_previous_token"][0], [[1, -1], [-1, 1]])
                np.testing.assert_array_equal(
                    archive["last_qmoe_context_node_names"], [c.context_identity[3] for c in last_qmoe]
                )
            plot_path = Path(temporary_directory) / "first.pdf"
            write_correlation_plots(plot_path, same_qmoe[:2], first_qmoe[:1], last_qmoe[:2])
            self.assertGreater(plot_path.stat().st_size, 0)

    def test_first_layer_matrix_uses_current_experts_as_rows(self):
        identities = (
            ("main", 1, "QMoE", "/model/layers.0/moe/QMoE"),
            ("main", 2, "QMoE", "/model/layers.3/moe/QMoE"),
        )
        experts = np.eye(3, dtype=np.uint8)
        prompt = np.stack((experts[[0, 0, 1, 1, 2, 2]], experts[[1, 1, 0, 2, 0, 2]]), axis=1)

        _, first_qmoe = compute_activation_correlations(
            ActivationTrace(identities, (prompt,), expert_count=3), reference_layer="first"
        )

        np.testing.assert_allclose(first_qmoe[0].values, [[-0.5, 0.25, 0.25], [1, -0.5, -0.5], [-0.5, 0.25, 0.25]])

    def test_last_layer_previous_token_alignment_and_orientation(self):
        identities = (
            ("main", 1, "QMoE", "/model/layers.0/moe/QMoE"),
            ("main", 2, "QMoE", "/model/layers.3/moe/QMoE"),
        )
        experts = np.eye(3, dtype=np.uint8)
        prompts = (
            np.stack((experts[[0, 0, 1, 2]], experts[[0, 1, 2, 0]]), axis=1),
            np.stack((experts[[2, 1, 0, 0]], experts[[2, 0, 1, 2]]), axis=1),
        )

        last_qmoe = compute_last_qmoe_previous_token_correlations(ActivationTrace(identities, prompts, expert_count=3))

        self.assertEqual(last_qmoe[0].samples, 6)
        np.testing.assert_array_equal(last_qmoe[0].current_activation_counts, [3, 2, 1])
        np.testing.assert_array_equal(last_qmoe[0].context_activation_counts, [2, 2, 2])
        np.testing.assert_allclose(
            last_qmoe[0].values,
            [
                [1 / np.sqrt(2), 0, -1 / np.sqrt(2)],
                [-0.5, 0.25, 0.25],
                [-1 / np.sqrt(10), -1 / np.sqrt(10), 2 / np.sqrt(10)],
            ],
        )

    def test_archive_supports_trace_without_previous_layer(self):
        identity = ("main", 1, "QMoE", "/model/layers.0/moe/QMoE")
        prompt = np.array([[[1, 0]], [[0, 1]], [[1, 0]]], dtype=np.uint8)
        for reference in ("previous", "first"):
            with self.subTest(reference=reference), tempfile.TemporaryDirectory() as temporary_directory:
                same_qmoe, context_qmoe = compute_activation_correlations(
                    ActivationTrace((identity,), (prompt,), expert_count=2), reference_layer=reference
                )
                path = Path(temporary_directory) / "correlations.npz"
                write_correlation_archive(path, same_qmoe, context_qmoe, reference_layer=reference)
                with np.load(path) as archive:
                    self.assertEqual(archive["same_qmoe_previous_token"].shape, (1, 2, 2))
                    self.assertEqual(archive[f"{reference}_qmoe_same_token"].shape, (0, 2, 2))
                    self.assertEqual(archive[f"{reference}_qmoe_current_activation_counts"].shape, (0, 2))
                    self.assertNotIn("last_qmoe_previous_token", archive)
                plot_path = Path(temporary_directory) / "correlations.pdf"
                write_correlation_plots(plot_path, same_qmoe, context_qmoe)
                self.assertGreater(plot_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
