# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import itertools
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import onnxruntime as ort
from onnxruntime.quantization.quant_utils import compute_scale_zp_blocked, quantize_onnx_initializer


class TestBlockedNegativeAxes(unittest.TestCase):
    @staticmethod
    def _weight(dtype=np.float32):
        return np.array(
            [[-3.2, 0.4, 1.7, 2.1, -0.8], [0.6, -1.3, 4.2, -2.6, 1.1], [2.7, 3.4, -0.9, 1.8, -4.1]],
            dtype=dtype,
        )

    def test_negative_axes_match_positive_parameters(self):
        for dtype, qtype, symmetric, axis, block_size in itertools.product(
            (np.float16, np.float32), (TensorProto.INT8, TensorProto.UINT8), (False, True), (0, 1), (2, 3)
        ):
            with self.subTest(dtype=dtype, qtype=qtype, symmetric=symmetric, axis=axis, block_size=block_size):
                weight = self._weight(dtype)
                expected = compute_scale_zp_blocked(weight, qtype, axis, block_size, symmetric)
                actual = compute_scale_zp_blocked(weight, qtype, axis - weight.ndim, block_size, symmetric)
                shape = list(weight.shape)
                shape[axis] = (shape[axis] + block_size - 1) // block_size
                for result, reference in zip(actual, expected, strict=True):
                    np.testing.assert_array_equal(result, reference)
                    self.assertEqual(result.shape, tuple(shape))
                    self.assertEqual(result.dtype, reference.dtype)

    def test_negative_axis_initializers_keep_quantized_values(self):
        for dtype, qtype, axis in itertools.product(
            (np.float16, np.float32), (TensorProto.INT8, TensorProto.UINT8), (0, 1)
        ):
            with self.subTest(dtype=dtype, qtype=qtype, axis=axis):
                weight = self._weight(dtype)
                zero_point, scale = compute_scale_zp_blocked(weight, qtype, axis, 2, False)
                initializer = numpy_helper.from_array(weight, "weight")
                original = initializer.SerializeToString()
                expected = quantize_onnx_initializer(initializer, qtype, zero_point, scale, axis=axis, block_size=2)
                actual = quantize_onnx_initializer(initializer, qtype, zero_point, scale, axis=axis - 2, block_size=2)
                self.assertEqual(actual.SerializeToString(), expected.SerializeToString())
                self.assertEqual(initializer.SerializeToString(), original)
                self.assertEqual(tuple(actual.dims), weight.shape)

    def test_noncontiguous_readonly_inputs(self):
        weight = self._weight()[:, ::-1].T
        weight.setflags(write=False)
        before = weight.copy()
        for axis in (0, 1):
            with self.subTest(axis=axis):
                expected = compute_scale_zp_blocked(weight, TensorProto.INT8, axis, 2, True)
                actual = compute_scale_zp_blocked(weight, TensorProto.INT8, axis - 2, 2, True)
                for result, reference in zip(actual, expected, strict=True):
                    np.testing.assert_array_equal(result, reference)
        np.testing.assert_array_equal(weight, before)
        self.assertFalse(weight.flags.writeable)

    def test_native_cpu_dequantization_and_matmul(self):
        for qtype, axis in itertools.product((TensorProto.INT8, TensorProto.UINT8), (0, 1)):
            with self.subTest(qtype=qtype, axis=axis):
                weight = self._weight()
                block_size = 2
                negative_axis = axis - weight.ndim
                zero_point, scale = compute_scale_zp_blocked(weight, qtype, negative_axis, block_size, False)
                quantized = quantize_onnx_initializer(
                    numpy_helper.from_array(weight, "weight"),
                    qtype,
                    zero_point,
                    scale,
                    axis=negative_axis,
                    block_size=block_size,
                )
                quantized_values = numpy_helper.to_array(quantized)
                expected_weight = np.empty_like(weight)
                for index in np.ndindex(weight.shape):
                    parameter_index = list(index)
                    parameter_index[axis] //= block_size
                    parameter_index = tuple(parameter_index)
                    expected_weight[index] = (int(quantized_values[index]) - int(zero_point[parameter_index])) * float(
                        scale[parameter_index]
                    )
                inputs = np.array([[0.4, -0.7, 1.2], [-0.6, 0.3, 0.8]], dtype=np.float32)
                graph = helper.make_graph(
                    [
                        helper.make_node(
                            "DequantizeLinear",
                            [quantized.name, "scale", "zero_point"],
                            ["dequantized_weight"],
                            axis=negative_axis,
                            block_size=block_size,
                        ),
                        helper.make_node("MatMul", ["input", "dequantized_weight"], ["output"]),
                    ],
                    "blocked-negative-axis",
                    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3])],
                    [
                        helper.make_tensor_value_info("dequantized_weight", TensorProto.FLOAT, list(weight.shape)),
                        helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 5]),
                    ],
                    [
                        quantized,
                        numpy_helper.from_array(scale, "scale"),
                        numpy_helper.from_array(zero_point, "zero_point"),
                    ],
                )
                model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
                onnx.checker.check_model(model)
                options = ort.SessionOptions()
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                session = ort.InferenceSession(
                    model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
                )
                actual_weight, actual_output = session.run(None, {"input": inputs})
                np.testing.assert_allclose(actual_weight, expected_weight, rtol=1e-6, atol=1e-7)
                np.testing.assert_allclose(actual_output, inputs @ expected_weight, rtol=1e-6, atol=1e-6)

    def test_per_channel_negative_axis_control(self):
        weight = self._weight()
        initializer = numpy_helper.from_array(weight, "weight")
        for axis in (0, 1):
            with self.subTest(axis=axis):
                scale = np.full(weight.shape[axis], 0.1, dtype=weight.dtype)
                zero_point = np.zeros(weight.shape[axis], dtype=np.int8)
                expected = quantize_onnx_initializer(initializer, TensorProto.INT8, zero_point, scale, axis=axis)
                actual = quantize_onnx_initializer(initializer, TensorProto.INT8, zero_point, scale, axis=axis - 2)
                self.assertEqual(actual.SerializeToString(), expected.SerializeToString())

    def test_out_of_range_axes_remain_invalid(self):
        weight = self._weight()
        initializer = numpy_helper.from_array(weight, "weight")
        zero_point, scale = compute_scale_zp_blocked(weight, TensorProto.INT8, 0, 2, False)
        for axis in (-3, 2):
            with self.subTest(axis=axis):
                with self.assertRaises(IndexError):
                    compute_scale_zp_blocked(weight, TensorProto.INT8, axis, 2, False)
                with self.assertRaises(IndexError):
                    quantize_onnx_initializer(initializer, TensorProto.INT8, zero_point, scale, axis=axis, block_size=2)

    def test_rank_limit_is_unchanged(self):
        with self.assertRaises(NotImplementedError):
            compute_scale_zp_blocked(np.ones((2, 3, 4), dtype=np.float32), TensorProto.INT8, -1, 2, False)


if __name__ == "__main__":
    unittest.main()
