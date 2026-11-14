# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import tempfile
import unittest
from pathlib import Path

import numpy as np
from onnx import TensorProto, checker, helper, load_model, numpy_helper, save_model

import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.quant_utils import compute_scale_zp_blocked, quantize_onnx_initializer


def _reference(weight, quant_type, axis, block_size, symmetric):
    # Python floats keep the range arithmetic independent of the input dtype.
    qmin, qmax, zp_dtype = (-127 if symmetric else -128, 127, np.int8)
    if quant_type == TensorProto.UINT8:
        qmin, qmax, zp_dtype = 0, 255, np.uint8
    shape = list(weight.shape)
    shape[axis] = (shape[axis] + block_size - 1) // block_size
    scales = np.empty(shape, dtype=weight.dtype)
    zeros = np.empty(shape, dtype=zp_dtype)
    for index in np.ndindex(*shape):
        selection = list(index)
        selection[axis] = slice(index[axis] * block_size, (index[axis] + 1) * block_size)
        values = [float(value) for value in weight[tuple(selection)]]
        low, high = min(0.0, *values), max(0.0, *values)
        if symmetric:
            high = max(-low, high)
            low = -high
        scale = weight.dtype.type((high - low) / (qmax - qmin))
        info = np.finfo(weight.dtype)
        floor = info.smallest_subnormal if weight.dtype == np.float16 else info.tiny
        degenerate = scale < floor
        scale = weight.dtype.type(1) if degenerate else scale
        zero = round((qmin + qmax) / 2) if symmetric else round(qmin - low / float(scale))
        scales[index] = scale
        zeros[index] = 0 if degenerate and not symmetric else min(qmax, max(qmin, zero))
    return zeros, scales


def _weights(dtype):
    magnitude = 60000 if dtype == np.float16 else 2e38
    return np.array([[1, -1, 0.4], [-1, 0.4, -0.4], [0.75, -0.25, 0]], dtype=dtype) * dtype(magnitude)


class _DataReader(CalibrationDataReader):
    def __init__(self, data):
        self._data = iter([{"A": data}])

    def get_next(self):
        return next(self._data, None)


class TestBlockedQuantizationRange(unittest.TestCase):
    def _check_parameters(self, weight, quant_type, axis, symmetric):
        original = weight.copy()
        expected_zero, expected_scale = _reference(weight, quant_type, axis, 2, symmetric)
        zero, scale = compute_scale_zp_blocked(weight, quant_type, axis, 2, symmetric)
        self.assertEqual(scale.dtype, weight.dtype)
        self.assertEqual(zero.dtype, expected_zero.dtype)
        self.assertTrue(np.isfinite(scale).all())
        self.assertTrue((scale > 0).all())
        np.testing.assert_array_equal(scale, expected_scale)
        np.testing.assert_array_equal(zero, expected_zero)
        np.testing.assert_array_equal(weight, original)

    def test_large_float16_ranges(self):
        for axis in (0, 1):
            for quant_type in (TensorProto.INT8, TensorProto.UINT8):
                for symmetric in (False, True):
                    with self.subTest(axis=axis, quant_type=quant_type, symmetric=symmetric):
                        self._check_parameters(_weights(np.float16), quant_type, axis, symmetric)

    def test_large_float32_ranges(self):
        for axis in (0, 1):
            for quant_type in (TensorProto.INT8, TensorProto.UINT8):
                for symmetric in (False, True):
                    with self.subTest(axis=axis, quant_type=quant_type, symmetric=symmetric):
                        self._check_parameters(_weights(np.float32), quant_type, axis, symmetric)

    def test_symmetric_single_sign_ranges(self):
        for dtype in (np.float16, np.float32):
            for sign in (-1, 1):
                with self.subTest(dtype=dtype, sign=sign):
                    self._check_parameters(np.abs(_weights(dtype)) * sign, TensorProto.INT8, 0, True)

    def test_noncontiguous_readonly_weights(self):
        for dtype in (np.float16, np.float32):
            with self.subTest(dtype=dtype):
                weight = _weights(dtype).T[:, ::-1]
                weight.flags.writeable = False
                self.assertFalse(weight.flags.c_contiguous)
                self._check_parameters(weight, TensorProto.UINT8, 1, False)

    def test_regular_small_and_zero_ranges(self):
        for dtype in (np.float16, np.float32):
            for magnitude in (0, 2**-10, 1):
                for symmetric in (False, True):
                    with self.subTest(dtype=dtype, magnitude=magnitude, symmetric=symmetric):
                        weight = np.array([[-4, 2, 0], [2, 8, 0], [-2, 1, 0]], dtype=dtype) * dtype(magnitude)
                        self._check_parameters(weight, TensorProto.INT8, 0, symmetric)

    def test_rank_limit_is_unchanged(self):
        with self.assertRaises(NotImplementedError):
            compute_scale_zp_blocked(np.zeros((2, 2, 2), dtype=np.float32), TensorProto.INT8, 0, 2, True)

    def test_native_cpu_blocked_dequantization(self):
        for dtype in (np.float16, np.float32):
            for axis in (0, 1):
                with self.subTest(dtype=dtype, axis=axis):
                    weight = _weights(dtype)
                    zero, scale = compute_scale_zp_blocked(weight, TensorProto.INT8, axis, 2, True)
                    quantized = quantize_onnx_initializer(
                        numpy_helper.from_array(weight, "W"), TensorProto.INT8, zero, scale, axis=axis, block_size=2
                    )
                    node = helper.make_node(
                        "DequantizeLinear", [quantized.name, "scale", "zero"], ["Y"], axis=axis, block_size=2
                    )
                    graph = helper.make_graph(
                        [node],
                        "blocked_dequantization",
                        [],
                        [
                            helper.make_tensor_value_info(
                                "Y", helper.np_dtype_to_tensor_dtype(weight.dtype), weight.shape
                            )
                        ],
                        [quantized, numpy_helper.from_array(scale, "scale"), numpy_helper.from_array(zero, "zero")],
                    )
                    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
                    checker.check_model(model)
                    options = ort.SessionOptions()
                    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                    session = ort.InferenceSession(
                        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
                    )
                    actual = session.run(None, {})[0]
                    _, expected_scale = _reference(weight, TensorProto.INT8, axis, 2, True)
                    expanded = np.take(np.repeat(expected_scale, 2, axis=axis), range(weight.shape[axis]), axis=axis)
                    expected_codes = np.clip(np.rint(weight.astype(np.float64) / expanded), -127, 127)
                    expected = (expected_codes * expanded.astype(np.float64)).astype(dtype)
                    self.assertEqual(actual.dtype, weight.dtype)
                    self.assertTrue(np.isfinite(actual).all())
                    np.testing.assert_allclose(actual, expected, rtol=np.finfo(dtype).eps, atol=0)
                    self.assertTrue((np.abs(actual.astype(np.float64) - weight) <= expanded.astype(np.float64)).all())

    def test_static_quantization_uses_finite_block_scales(self):
        for dtype in (np.float16, np.float32):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                magnitude = dtype(60000 if dtype == np.float16 else 2e38)
                weight = np.array([[1, -1], [-1, 1], [0.5, -0.5], [-0.5, 0.5]], dtype=dtype) * magnitude
                data = np.array([[0.001, 0, 0, 0]], dtype=dtype)
                element_type = helper.np_dtype_to_tensor_dtype(weight.dtype)
                model = helper.make_model(
                    helper.make_graph(
                        [helper.make_node("MatMul", ["A", "W"], ["Y"])],
                        "blocked_matmul",
                        [helper.make_tensor_value_info("A", element_type, [1, 4])],
                        [helper.make_tensor_value_info("Y", element_type, [1, 2])],
                        [numpy_helper.from_array(weight, "W")],
                    ),
                    opset_imports=[helper.make_opsetid("", 21)],
                    ir_version=10,
                )
                source, destination = Path(directory) / "float.onnx", Path(directory) / "qdq.onnx"
                save_model(model, source)
                quantize_static(
                    source,
                    destination,
                    _DataReader(data),
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8,
                    op_types_to_quantize=["MatMul"],
                    extra_options={"WeightSymmetric": True, "ActivationSymmetric": True, "BlockSize": 2},
                )
                quantized_model = load_model(destination)
                checker.check_model(quantized_model)
                stored_scale = numpy_helper.to_array(
                    next(i for i in quantized_model.graph.initializer if i.name == "W_scale")
                )
                _, expected_scale = _reference(weight, TensorProto.INT8, 0, 2, True)
                np.testing.assert_array_equal(stored_scale, expected_scale)
                options = ort.SessionOptions()
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                session = ort.InferenceSession(str(destination), options, providers=["CPUExecutionProvider"])
                actual = session.run(None, {"A": data})[0]
                self.assertTrue(np.isfinite(actual).all())
                expected = data.astype(np.float64) @ weight.astype(np.float64)
                np.testing.assert_allclose(actual, expected, rtol=0.03, atol=0)


if __name__ == "__main__":
    unittest.main()
