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
import onnx
from ml_dtypes import float8_e4m3fn
from onnx import TensorProto, helper, numpy_helper

import onnxruntime as ort
from onnxruntime.quantization.quant_utils import compute_data_quant_params, quantize_data


def weight_values(dtype, magnitude):
    return np.array([-1, -0.5, 0, 0.5, 1], dtype=dtype) * dtype(magnitude)


def reference_scale(data):
    values = [float(value) for value in data.flat]
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    # Preserve the existing FP8 reference distribution; only weight statistics
    # need a wider accumulator.
    codes = np.arange(256, dtype=np.uint8).view(float8_e4m3fn).astype(np.float32)
    reference_std = np.std(codes[np.isfinite(codes)])
    return np.array(math.sqrt(variance) / float(reference_std), dtype=data.dtype)


class TestFloat8WeightStatistics(unittest.TestCase):
    def test_large_finite_weights_have_finite_scale(self):
        for dtype, magnitude in ((np.float16, 60000), (np.float16, 1000), (np.float32, 1e30), (np.float32, 1e20)):
            with self.subTest(dtype=dtype, magnitude=magnitude):
                data = weight_values(dtype, magnitude)
                zero, scale = compute_data_quant_params(data, TensorProto.FLOAT8E4M3FN, True)
                self.assertTrue(np.isfinite(scale))
                self.assertGreater(float(scale), 0)
                self.assertEqual(scale.dtype, data.dtype)
                self.assertEqual(zero.dtype, np.dtype(float8_e4m3fn))
                np.testing.assert_array_equal(scale, reference_scale(data))

    def test_small_nonconstant_weights_have_nonzero_scale(self):
        for dtype, magnitude in ((np.float16, 1e-4), (np.float16, 1e-3), (np.float32, 1e-30), (np.float32, 1e-24)):
            with self.subTest(dtype=dtype, magnitude=magnitude):
                data = weight_values(dtype, magnitude)
                zero, scale = compute_data_quant_params(data, TensorProto.FLOAT8E4M3FN, True)
                self.assertTrue(np.isfinite(scale))
                self.assertGreater(float(scale), 0)
                self.assertEqual(scale.dtype, data.dtype)
                np.testing.assert_array_equal(scale, reference_scale(data))
                self.assertEqual(float(zero), 0)

    def test_quantized_codes_match_independent_scale_reference(self):
        for dtype, magnitude in (
            (np.float16, 60000),
            (np.float16, 1e-4),
            (np.float32, 1e30),
            (np.float32, 1e-30),
            (np.float16, 1),
            (np.float32, 1),
        ):
            with self.subTest(dtype=dtype, magnitude=magnitude):
                data = weight_values(dtype, magnitude)
                _, scale, quantized = quantize_data(data, TensorProto.FLOAT8E4M3FN, True)
                expected_scale = reference_scale(data)
                expected = np.array(
                    [float(value) / float(expected_scale) for value in data],
                    dtype=float8_e4m3fn,
                )
                self.assertEqual(quantized.dtype, np.dtype(float8_e4m3fn))
                np.testing.assert_array_equal(scale, expected_scale)
                np.testing.assert_array_equal(quantized.view(np.uint8), expected.view(np.uint8))
                self.assertTrue(np.isfinite(quantized.astype(np.float32)).all())

    def test_native_cpu_dequantization_roundtrip(self):
        for magnitude in (1e30, 1e-30):
            with self.subTest(magnitude=magnitude), tempfile.TemporaryDirectory() as directory:
                data = weight_values(np.float32, magnitude)
                zero, scale, quantized = quantize_data(data, TensorProto.FLOAT8E4M3FN, True)
                expected_scale = reference_scale(data)
                model = helper.make_model(
                    helper.make_graph(
                        [
                            helper.make_node("DequantizeLinear", ["weight", "scale", "zero"], ["bias"]),
                            helper.make_node("Add", ["X", "bias"], ["Y"]),
                        ],
                        "float8_weight_statistics",
                        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [5])],
                        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [5])],
                        [
                            numpy_helper.from_array(quantized, "weight"),
                            numpy_helper.from_array(scale, "scale"),
                            numpy_helper.from_array(zero, "zero"),
                        ],
                    ),
                    opset_imports=[helper.make_opsetid("", 21)],
                    ir_version=10,
                )
                onnx.checker.check_model(model)
                path = Path(directory) / "float8.onnx"
                onnx.save_model(model, path)
                stored = onnx.load_model(path)
                stored_scale = next(item for item in stored.graph.initializer if item.name == "scale")
                np.testing.assert_array_equal(numpy_helper.to_array(stored_scale), expected_scale)
                options = ort.SessionOptions()
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
                actual = session.run(None, {"X": np.zeros(5, dtype=np.float32)})[0]
                expected_codes = np.array(
                    [float(value) / float(expected_scale) for value in data], dtype=float8_e4m3fn
                ).astype(np.float32)
                expected = expected_codes * expected_scale
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_allclose(actual, data, rtol=0.04, atol=0)

    def test_strided_readonly_weights_preserve_input(self):
        for dtype, magnitude in ((np.float16, 60000), (np.float32, 1e-30)):
            with self.subTest(dtype=dtype):
                data = np.tile(weight_values(dtype, magnitude), (3, 1)).T[::-1, ::2]
                data.flags.writeable = False
                before = data.copy()
                _, scale = compute_data_quant_params(data, TensorProto.FLOAT8E4M3FN, False)
                np.testing.assert_array_equal(scale, reference_scale(data))
                np.testing.assert_array_equal(data, before)
                self.assertFalse(data.flags.writeable)

    def test_ordinary_scale_dtype_and_options(self):
        for dtype in (np.float16, np.float32):
            for symmetric in (False, True):
                with self.subTest(dtype=dtype, symmetric=symmetric):
                    data = np.array([-3, -1, 0, 2, 5], dtype=dtype)
                    zero, scale = compute_data_quant_params(
                        data, TensorProto.FLOAT8E4M3FN, symmetric, rmin_override=-100, rmax_override=100
                    )
                    np.testing.assert_array_equal(scale, reference_scale(data))
                    self.assertEqual(scale.shape, ())
                    self.assertEqual(scale.dtype, data.dtype)
                    self.assertEqual(float(zero), 0)

    def test_other_array_dtypes_keep_direct_helper_result_dtype(self):
        for dtype in (np.int32, np.int64, np.float64, np.complex64, np.complex128, np.bool_):
            with self.subTest(dtype=dtype):
                data = np.array([0, 1, 0, 1], dtype=dtype)
                # Direct callers have historically received NumPy's default
                # statistics dtype for types outside the FP16/FP32 weight path.
                original_std = np.std(data)
                codes = np.arange(256, dtype=np.uint8).view(float8_e4m3fn).astype(np.float32)
                expected = np.array(original_std / np.std(codes[np.isfinite(codes)]), dtype=original_std.dtype)
                zero, scale = compute_data_quant_params(data, TensorProto.FLOAT8E4M3FN, True)
                self.assertEqual(scale.dtype, expected.dtype)
                np.testing.assert_array_equal(scale, expected)
                self.assertGreater(float(scale), 0)
                self.assertEqual(float(zero), 0)

    def test_zero_variance_and_reduce_range_contract_unchanged(self):
        for dtype in (np.float16, np.float32):
            with self.subTest(dtype=dtype):
                # This change only fixes intermediate arithmetic range; it does
                # not choose a new policy for the existing zero-variance case.
                zero, scale = compute_data_quant_params(np.zeros(5, dtype=dtype), TensorProto.FLOAT8E4M3FN, True)
                self.assertEqual(float(scale), 0)
                self.assertEqual(float(zero), 0)
                with self.assertRaisesRegex(RuntimeError, "reduce_range=True"):
                    compute_data_quant_params(
                        np.ones(5, dtype=dtype), TensorProto.FLOAT8E4M3FN, True, reduce_range=True
                    )


if __name__ == "__main__":
    unittest.main()
