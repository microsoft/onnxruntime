#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import math
import statistics
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.calibrate import (
    CalibrationDataReader,
    CalibrationMethod,
    HistogramCollector,
    create_calibrator,
)


def reference_moments(hist, edges, power=1):
    observations = []
    for count, left, right in zip(hist, edges[:-1], edges[1:], strict=True):
        midpoint = (float(left) + float(right)) / 2
        value = math.copysign(abs(midpoint) ** power, midpoint)
        observations.extend([value] * int(count))
    return statistics.mean(observations), statistics.pstdev(observations)


class ArrayReader(CalibrationDataReader):
    def __init__(self, values):
        self.values = iter(values)

    def get_next(self):
        values = next(self.values, None)
        return None if values is None else {"X": values}


class TestDistributionMoments(unittest.TestCase):
    def check_moments(self, edges, hist=None, power=1):
        if hist is None:
            hist = np.ones(len(edges) - 1, dtype=np.int64)
        original_edges = edges.copy()
        original_hist = hist.copy()
        expected_mean, expected_std = reference_moments(hist, edges, power)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            mean, std = HistogramCollector._avg_std(hist, edges, power)
        self.assertEqual(mean.dtype, edges.dtype)
        self.assertEqual(std.dtype, edges.dtype)
        np.testing.assert_allclose(mean, np.array(expected_mean, dtype=edges.dtype), rtol=1e-6, atol=0)
        np.testing.assert_allclose(std, np.array(expected_std, dtype=edges.dtype), rtol=1e-6, atol=0)
        np.testing.assert_array_equal(edges, original_edges)
        np.testing.assert_array_equal(hist, original_hist)

    def test_float16_squared_midpoints(self):
        self.check_moments(np.array([1000, 1001, 1002], dtype=np.float16))

    def test_float16_midpoint_addition(self):
        self.check_moments(np.array([49024, 50016, 51008], dtype=np.float16))

    def test_float32_cancellation(self):
        self.check_moments(np.array([1, 1.0001, 1.0002], dtype=np.float32))

    def test_float32_small_spread(self):
        self.check_moments(np.array([-2e-30, 0, 2e-30], dtype=np.float32))

    def test_float32_large_spread(self):
        self.check_moments(np.array([-2e25, 0, 2e25], dtype=np.float32))

    def test_signed_power_scenarios(self):
        for dtype in (np.float16, np.float32):
            for power in (1 / 3, 2, 3):
                with self.subTest(dtype=dtype, power=power):
                    self.check_moments(
                        np.array([-27, -9, -1, 1, 9, 27], dtype=dtype),
                        np.array([1, 3, 2, 1, 2]),
                        power,
                    )

    def test_single_occupied_bin(self):
        self.check_moments(np.array([1, 1.0001, 1.0002], dtype=np.float32), np.array([0, 7]))

    def test_noncontiguous_readonly_inputs(self):
        edges = np.array([-4, 100, -2, 100, 0, 100, 2, 100, 4], dtype=np.float32)[::2]
        hist = np.array([1, 100, 3, 100, 2, 100, 4], dtype=np.int64)[::2]
        edges.setflags(write=False)
        hist.setflags(write=False)
        self.check_moments(edges, hist)

    def test_zero_counts_remain_undefined(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            mean, std = HistogramCollector._avg_std(np.array([0, 0]), np.array([-1, 0, 1], dtype=np.float32))
        self.assertTrue(np.isnan(mean))
        self.assertTrue(np.isnan(std))

    def test_invalid_power(self):
        for power in (0, -1):
            with self.subTest(power=power), self.assertRaisesRegex(ValueError, "invalid"):
                HistogramCollector._avg_std(np.array([1]), np.array([0, 1], dtype=np.float32), power)

    @staticmethod
    def make_model(path, dtype):
        tensor_type = TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["X", "W"], ["Y"])],
            "distribution_moments",
            [helper.make_tensor_value_info("X", tensor_type, [None, 2])],
            [helper.make_tensor_value_info("Y", tensor_type, [None, 2])],
            [numpy_helper.from_array(np.eye(2, dtype=dtype), "W")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
        onnx.checker.check_model(model)
        onnx.save(model, path)

    def test_native_cpu_distribution_calibration(self):
        for dtype, magnitude in ((np.float16, 1000), (np.float32, 1e20)):
            for scenario in ("same", "p3"):
                with self.subTest(dtype=dtype, scenario=scenario), tempfile.TemporaryDirectory() as temporary_directory:
                    directory = Path(temporary_directory)
                    model_path = directory / "model.onnx"
                    self.make_model(model_path, dtype)
                    values = np.array([[-magnitude, -magnitude / 2], [magnitude / 2, magnitude]], dtype=dtype)
                    calibrator = create_calibrator(
                        model_path,
                        calibrate_method=CalibrationMethod.Distribution,
                        augmented_model_path=directory / "augmented.onnx",
                        providers=["CPUExecutionProvider"],
                        extra_options={"num_bins": 512, "scenario": scenario},
                    )
                    calibrator.collect_data(ArrayReader([values]))
                    tensors = calibrator.compute_data()
                    self.assertEqual(set(tensors), {"X", "Y"})
                    power = 1 if scenario == "same" else 1 / 3
                    for name in tensors:
                        tensor = tensors[name]
                        expected = reference_moments(tensor.hist, tensor.hist_edges, power)
                        np.testing.assert_allclose(
                            tensor.avg_std,
                            np.asarray(expected, dtype=dtype),
                            rtol=1e-6,
                            atol=np.finfo(dtype).eps * max(abs(expected[0]), 1),
                        )
                        self.assertTrue(np.isfinite(tensor.std))
                        self.assertGreater(tensor.std, 0)

    def test_fp8_static_quantization_scales(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            model_path = directory / "model.onnx"
            quantized_path = directory / "quantized.onnx"
            self.make_model(model_path, np.float32)
            values = np.array([[-1e20, -5e19], [5e19, 1e20]], dtype=np.float32)
            quantize_static(
                model_path,
                quantized_path,
                ArrayReader([values]),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QFLOAT8E4M3FN,
                weight_type=QuantType.QFLOAT8E4M3FN,
                calibrate_method=CalibrationMethod.Distribution,
                calibration_providers=["CPUExecutionProvider"],
            )
            model = onnx.load(quantized_path)
            onnx.checker.check_model(model)
            scales = [
                numpy_helper.to_array(value) for value in model.graph.initializer if value.name.endswith("_scale")
            ]
            self.assertGreaterEqual(len(scales), 3)
            for scale in scales:
                self.assertTrue(np.all(np.isfinite(scale)))
                self.assertTrue(np.all(scale > 0))
            output = ReferenceEvaluator(model).run(None, {"X": values})[0]
            self.assertTrue(np.all(np.isfinite(output)))
            np.testing.assert_allclose(output, values, rtol=0.15, atol=0)


if __name__ == "__main__":
    unittest.main()
