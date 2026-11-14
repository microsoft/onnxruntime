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
from onnxruntime.quantization.quant_utils import compute_data_quant_params, quantize_data

_INTEGER_TYPES = (
    TensorProto.INT8,
    TensorProto.UINT8,
    TensorProto.INT16,
    TensorProto.UINT16,
    TensorProto.INT4,
    TensorProto.UINT4,
)


class TestQuantizeDataShapes(unittest.TestCase):
    def test_scalar_quantization_matches_single_element_tensor(self):
        for dtype in (np.float16, np.float32):
            for quant_type in _INTEGER_TYPES:
                for value in (-2, 0, 2):
                    for symmetric in (False, True):
                        with self.subTest(dtype=dtype, quant_type=quant_type, value=value, symmetric=symmetric):
                            data = np.array(value, dtype=dtype)
                            original = data.copy()
                            expected = quantize_data(data.reshape(1), quant_type, symmetric)
                            zero, scale, codes = quantize_data(data, quant_type, symmetric)
                            self.assertEqual(scale.shape, ())
                            self.assertEqual(zero.shape, ())
                            self.assertEqual(codes.shape, ())
                            self.assertEqual(scale.dtype, data.dtype)
                            np.testing.assert_array_equal(zero, expected[0])
                            np.testing.assert_array_equal(scale, expected[1])
                            np.testing.assert_array_equal(codes, expected[2].reshape(()))
                            np.testing.assert_array_equal(data, original)

    def test_scalar_known_scale_zero_point_and_code(self):
        cases = (
            (2.0, TensorProto.INT8, True, 0, 2 / 127, 127),
            (-2.0, TensorProto.INT8, True, 0, 2 / 127, -127),
            (2.0, TensorProto.UINT8, False, 0, 2 / 255, 255),
            (-2.0, TensorProto.INT8, False, 127, 2 / 255, -128),
            (0.0, TensorProto.UINT8, False, 0, 1, 0),
        )
        for value, quant_type, symmetric, expected_zero, expected_scale, expected_code in cases:
            with self.subTest(value=value, quant_type=quant_type, symmetric=symmetric):
                zero, scale, code = quantize_data(np.array(value, dtype=np.float32), quant_type, symmetric)
                self.assertEqual(zero.item(), expected_zero)
                self.assertEqual(scale.item(), np.float32(expected_scale))
                self.assertEqual(code.item(), expected_code)

    def test_empty_tensors_keep_shape_and_default_parameters(self):
        for dtype in (np.float16, np.float32):
            for quant_type in _INTEGER_TYPES:
                for shape in ((0,), (2, 0), (2, 0, 3), (0, 2), (1, 2, 0)):
                    with self.subTest(dtype=dtype, quant_type=quant_type, shape=shape):
                        data = np.empty(shape, dtype=dtype)
                        zero, scale, codes = quantize_data(data, quant_type, symmetric=True, reduce_range=True)
                        self.assertEqual(codes.shape, shape)
                        self.assertEqual(codes.size, 0)
                        self.assertEqual(scale.shape, ())
                        self.assertEqual(scale.dtype, data.dtype)
                        self.assertEqual(scale.item(), 1.0)
                        self.assertEqual(zero.shape, ())
                        self.assertEqual(zero.item(), 0)

    def test_range_overrides_and_minimum_range_are_shape_independent(self):
        options = (
            {"rmin_override": -4.0},
            {"rmax_override": 4.0},
            {"rmin_override": -4.0, "rmax_override": 4.0},
            {"min_real_range": 8.0},
            {"reduce_range": True},
        )
        for kwargs in options:
            with self.subTest(options=kwargs):
                data = np.array(2.0, dtype=np.float32)
                actual = quantize_data(data, TensorProto.INT8, False, **kwargs)
                expected = quantize_data(data.reshape(1), TensorProto.INT8, False, **kwargs)
                for result, reference in zip(actual[:2], expected[:2], strict=True):
                    np.testing.assert_array_equal(result, reference)
                self.assertEqual(actual[2].item(), expected[2].item())

        # Empty tensors have always used the identity parameters, even with overrides.
        zero, scale, codes = quantize_data(
            np.empty((2, 0), dtype=np.float16), TensorProto.UINT8, False, rmin_override=-4, rmax_override=4
        )
        self.assertEqual((zero.item(), scale.item(), codes.shape), (0, 1.0, (2, 0)))

    def test_readonly_inputs_and_nonempty_tensor_controls(self):
        for shape in ((), (2, 0), (2, 3)):
            with self.subTest(shape=shape):
                data = np.arange(np.prod(shape, dtype=int), dtype=np.float32).reshape(shape)
                data.flags.writeable = False
                before = data.copy()
                expected = quantize_data(data.reshape(-1), TensorProto.INT8, True)
                zero, scale, codes = quantize_data(data, TensorProto.INT8, True)
                np.testing.assert_array_equal(zero, expected[0])
                np.testing.assert_array_equal(scale, expected[1])
                np.testing.assert_array_equal(codes, expected[2].reshape(shape))
                np.testing.assert_array_equal(data, before)
                self.assertFalse(data.flags.writeable)

    def test_native_cpu_scalar_quantization_and_serialization(self):
        for value in (-2.0, 0.0, 2.0):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                data = np.array(value, dtype=np.float32)
                zero, scale, code = quantize_data(data, TensorProto.INT8, True)
                model = helper.make_model(
                    helper.make_graph(
                        [
                            helper.make_node("DequantizeLinear", ["weight", "scale", "zero"], ["bias"]),
                            helper.make_node("Add", ["X", "bias"], ["Y"]),
                        ],
                        "scalar_quantized_bias",
                        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [3])],
                        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])],
                        [
                            numpy_helper.from_array(code, "weight"),
                            numpy_helper.from_array(scale, "scale"),
                            numpy_helper.from_array(zero, "zero"),
                        ],
                    ),
                    opset_imports=[helper.make_opsetid("", 21)],
                    ir_version=10,
                )
                checker.check_model(model)
                path = Path(directory) / "scalar.onnx"
                save_model(model, path)
                stored = load_model(path)
                self.assertTrue(all(tuple(t.dims) == () for t in stored.graph.initializer))
                options = ort.SessionOptions()
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
                inputs = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
                actual = session.run(None, {"X": inputs})[0]
                np.testing.assert_allclose(actual, inputs + value, rtol=0, atol=float(scale) / 2)

    def test_invalid_argument_controls(self):
        with self.assertRaisesRegex(TypeError, "Weight must be given as an array"):
            compute_data_quant_params([1.0], TensorProto.INT8, True)
        with self.assertRaisesRegex(ValueError, "Unexpected value for quant_type"):
            compute_data_quant_params(np.ones(1, dtype=np.float32), TensorProto.DOUBLE, True)


if __name__ == "__main__":
    unittest.main()
