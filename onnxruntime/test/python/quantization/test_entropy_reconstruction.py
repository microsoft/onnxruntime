#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from onnx import TensorProto, checker, helper, load, numpy_helper, save

import onnxruntime as ort
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.calibrate import (
    CalibrationDataReader,
    CalibrationMethod,
    HistogramCollector,
    create_calibrator,
    save_tensors_data,
)


def reference_threshold(hist, edges, quantized_bins, observed_min=None, observed_max=None):
    """Enumerate the current candidate ranges using real-valued bucket reconstruction."""

    def normalized(values):
        zeros = sum(value == 0 for value in values)
        nonzeros = len(values) - zeros
        if not nonzeros:
            return None
        correction = 0.0001 * zeros / nonzeros
        smoothed = [0.0001 if value == 0 else value - correction for value in values]
        total = sum(smoothed)
        return [value / total for value in smoothed]

    candidates = []
    for half_width in range(quantized_bins // 2, len(hist) // 2 + 1):
        left = len(hist) // 2 - half_width
        right = min(len(hist) // 2 + half_width + 1, len(hist))
        retained = [int(value) for value in hist[left:right]]
        reference = retained.copy()
        reference[0] += sum(int(value) for value in hist[:left])
        reference[-1] += sum(int(value) for value in hist[right:])
        bucket_width = len(retained) // quantized_bins
        reconstructed = [0.0] * len(retained)
        for bucket in range(quantized_bins):
            end = len(retained) if bucket == quantized_bins - 1 else (bucket + 1) * bucket_width
            indices = range(bucket * bucket_width, end)
            support = [index for index in indices if reference[index] != 0]
            mass = sum(retained[index] for index in indices)
            for index in support:
                reconstructed[index] = mass / len(support)
        # The reconstructed distribution retains every in-range observation and
        # adds no mass to a bin absent from the reference distribution.
        assert math.isclose(sum(reconstructed), sum(retained), rel_tol=1e-14, abs_tol=0)
        assert all(value == 0 for value, count in zip(reconstructed, reference, strict=True) if count == 0)
        p, q = normalized(reference), normalized(reconstructed)
        divergence = math.inf if p is None or q is None else sum(x * math.log(x / y) for x, y in zip(p, q, strict=True))
        candidates.append((divergence, left, right))

    _, left, right = min(candidates, key=lambda candidate: candidate[0])
    low = float(edges[left])
    high = float(edges[right])
    return (
        max(low, float(edges[0] if observed_min is None else observed_min)),
        min(high, float(edges[-1] if observed_max is None else observed_max)),
    )


class ArrayReader(CalibrationDataReader):
    def __init__(self, values):
        self.values = iter(values)

    def get_next(self):
        values = next(self.values, None)
        return None if values is None else {"X": values}


class TestEntropyReconstruction(unittest.TestCase):
    def check_histogram(self, counts, quantized_bins=4):
        hist = np.array(counts, dtype=np.int64)
        for dtype in (np.float16, np.float32):
            with self.subTest(dtype=dtype):
                edges = np.arange(-len(hist) // 2, len(hist) // 2 + 1, dtype=dtype)
                original_hist, original_edges = hist.copy(), edges.copy()
                expected = reference_threshold(hist, edges, quantized_bins)
                collector = HistogramCollector("entropy", False, len(hist), quantized_bins, 99.99, "same")
                actual = collector.get_entropy_threshold((hist, edges, edges[0], edges[-1]), quantized_bins)
                self.assertEqual(tuple(actual), expected)
                self.assertEqual(actual[0].dtype, dtype)
                self.assertEqual(actual[1].dtype, dtype)
                np.testing.assert_array_equal(hist, original_hist)
                np.testing.assert_array_equal(edges, original_edges)

    def test_fractional_bucket_averages(self):
        self.check_histogram([2, 2, 2, 4, 2, 4, 4, 4, 4, 5, 3, 1, 2, 4, 4, 3])

    def test_sparse_histogram_support(self):
        self.check_histogram([0, 0, 5, 1, 0, 4, 5, 0, 3, 1, 0, 4, 0, 1, 3, 0])

    def test_nondivisible_candidate_widths(self):
        self.check_histogram([4, 1, 5, 0, 5, 1, 1, 2, 4, 5, 0, 3, 4, 4, 4, 4])

    def test_six_quantized_bins(self):
        self.check_histogram([0, 5, 3, 1, 5, 0, 1, 1, 3, 2, 0, 3, 3, 0, 2, 2], 6)

    def test_eight_quantized_bins(self):
        self.check_histogram([1, 3, 4, 4, 3, 3, 5, 4, 0, 5, 0, 3, 0, 0, 2, 2], 8)

    def test_unmerged_bin_control(self):
        self.check_histogram([1, 3, 0, 4, 2, 0, 3, 1], 8)

    @staticmethod
    def make_model(path, dtype):
        elem_type = TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT
        graph = helper.make_graph(
            [helper.make_node("Add", ["X", "zero"], ["Y"])],
            "entropy_calibration",
            [helper.make_tensor_value_info("X", elem_type, [None])],
            [helper.make_tensor_value_info("Y", elem_type, [None])],
            [numpy_helper.from_array(np.array(0, dtype=dtype), "zero")],
        )
        save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)]), path)

    @staticmethod
    def calibration_values(dtype):
        counts = np.array([4, 1, 5, 0, 5, 1, 1, 2, 4, 5, 0, 3, 4, 4, 4, 4])
        values = np.repeat(np.arange(-7.5, 8, dtype=dtype), counts)
        # Exact extrema fix the histogram edges without changing bin counts.
        values[0], values[-1] = -8, 8
        return values, counts

    def test_native_cpu_calibration_ranges(self):
        for dtype in (np.float16, np.float32):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                model_path = Path(directory) / "model.onnx"
                self.make_model(model_path, dtype)
                values, counts = self.calibration_values(dtype)
                calibrator = create_calibrator(
                    model_path,
                    ["Add"],
                    str(Path(directory) / "augmented.onnx"),
                    calibrate_method=CalibrationMethod.Entropy,
                    providers=["CPUExecutionProvider"],
                    extra_options={"num_bins": 16, "num_quantized_bins": 4},
                )
                calibrator.collect_data(ArrayReader([values]))
                ranges = calibrator.compute_data()
                expected = reference_threshold(counts, np.arange(-8, 9, dtype=dtype), 4)
                for name in ("X", "Y"):
                    np.testing.assert_array_equal(calibrator.collector.histogram_dict[name][0], counts)
                    self.assertEqual(tuple(ranges[name].range_value), expected)

    def test_cached_ranges_drive_static_quantization(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            model_path = directory / "model.onnx"
            self.make_model(model_path, np.float32)
            values, counts = self.calibration_values(np.float32)
            calibrator = create_calibrator(
                model_path,
                ["Add"],
                str(directory / "augmented.onnx"),
                calibrate_method=CalibrationMethod.Entropy,
                providers=["CPUExecutionProvider"],
                extra_options={"num_bins": 16, "num_quantized_bins": 4},
            )
            calibrator.collect_data(ArrayReader([values]))
            cache_path = directory / "ranges.json"
            save_tensors_data(calibrator.compute_data(), cache_path)
            quantized_path = directory / "quantized.onnx"
            quantize_static(
                model_path,
                quantized_path,
                calibration_data_reader=None,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                calibrate_method=CalibrationMethod.Entropy,
                calibration_cache_path=cache_path,
                op_types_to_quantize=["Add"],
            )
            model = load(quantized_path)
            checker.check_model(model)
            initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
            expected_low, expected_high = reference_threshold(counts, np.arange(-8, 9, dtype=np.float32), 4)
            expected_scale = np.float32((expected_high - expected_low) / 255)
            self.assertEqual(initializers["X_scale"], expected_scale)
            session = ort.InferenceSession(str(quantized_path), providers=["CPUExecutionProvider"])
            inputs = np.array([-2, -0.5, 0, 0.5, 1.25], dtype=np.float32)
            outputs = session.run(None, {"X": inputs})[0]
            np.testing.assert_allclose(outputs, inputs, rtol=0, atol=2 * float(expected_scale))


if __name__ == "__main__":
    unittest.main()
