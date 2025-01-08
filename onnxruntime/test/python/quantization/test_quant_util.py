#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import tempfile
import unittest
from pathlib import Path

import numpy
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnxruntime.quantization.quant_utils import (
    compute_scale_zp,
    load_model_with_shape_infer,
    model_has_infer_metadata,
    pack_bytes_to_4bit,
    quantize_data,
)


class TestQuantUtil(unittest.TestCase):
    def test_compute_scale_zp(self):
        def _compute_scale_zp(rmin, rmax, qmin, qmax, qtype, symmetric=False, min_real_range=None):
            zp, scale = compute_scale_zp(
                numpy.array(rmin, dtype=numpy.float32),
                numpy.array(rmax, dtype=numpy.float32),
                numpy.array(qmin, dtype=qtype),
                numpy.array(qmax, dtype=qtype),
                symmetric=symmetric,
                min_real_range=min_real_range,
            )
            assert isinstance(zp, numpy.ndarray)
            assert isinstance(scale, numpy.ndarray)
            return [float(zp), float(scale)]

        numpy.testing.assert_allclose(_compute_scale_zp(0.0, 0.0, -127, 127, numpy.int8, symmetric=True), [0, 1.0])
        numpy.testing.assert_allclose(_compute_scale_zp(1.0, -1.0, -127, 127, numpy.int8, symmetric=True), [0, 1.0])
        numpy.testing.assert_allclose(_compute_scale_zp(0.0, 0.0, 0, 255, numpy.uint8, symmetric=True), [0, 1.0])
        numpy.testing.assert_allclose(_compute_scale_zp(1.0, -1.0, 0, 255, numpy.uint8, symmetric=True), [0, 1.0])

        numpy.testing.assert_allclose(
            _compute_scale_zp(-1.0, 2.0, -127, 127, numpy.int8, symmetric=True), [0, numpy.float32(2.0 / 127)]
        )
        numpy.testing.assert_allclose(
            _compute_scale_zp(-1.0, 2.0, -127, 127, numpy.int8, symmetric=False), [-42, numpy.float32(3.0 / 254)]
        )

        numpy.testing.assert_allclose(
            _compute_scale_zp(-1.0, 2.0, 0, 255, numpy.uint8, symmetric=True), [128, numpy.float32(4.0 / 255)]
        )
        numpy.testing.assert_allclose(
            _compute_scale_zp(-1.0, 2.0, 0, 255, numpy.uint8, symmetric=False), [85, numpy.float32(3.0 / 255)]
        )

        tiny_float = numpy.float32(numpy.finfo(numpy.float32).tiny * 0.1)
        numpy.testing.assert_allclose(
            _compute_scale_zp(-tiny_float, tiny_float, 0, 255, numpy.uint8, symmetric=True), [0, 1.0]
        )
        numpy.testing.assert_allclose(
            _compute_scale_zp(-tiny_float, 0.0, 0, 255, numpy.uint8, symmetric=False), [0, 1.0]
        )

        # Test enforcing a minimum floatint-point range.
        numpy.testing.assert_allclose(
            _compute_scale_zp(0.0, 0.0, 0, 255, numpy.uint8, symmetric=False, min_real_range=0.0001), [0, 0.0001 / 255]
        )
        numpy.testing.assert_allclose(
            _compute_scale_zp(0.0, 0.0, -128, 127, numpy.int8, symmetric=True, min_real_range=0.0001), [0, 0.0002 / 255]
        )
        numpy.testing.assert_allclose(
            _compute_scale_zp(0.0, 0.0, 0, 65535, numpy.uint16, symmetric=False, min_real_range=0.0001),
            [0, 0.0001 / 65535],
        )
        numpy.testing.assert_allclose(
            _compute_scale_zp(0.0, 0.0, -32768, 32767, numpy.int16, symmetric=True, min_real_range=0.0001),
            [0, 0.0002 / 65535],
        )

    def test_load_external_model(self):
        input_name = "input"
        output_name = "output"
        add_shape = [1024, 1024]

        initializers = []
        weight_name = "weight"
        weight_data = numpy.random.normal(0, 0.1, add_shape).astype(numpy.float32)
        initializers.append(numpy_helper.from_array(weight_data, name=weight_name))
        add_node = helper.make_node("Add", [input_name, weight_name], [output_name], name="add_node")

        # make graph
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, add_shape)
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, add_shape)
        graph_name = "test_load_external_model"
        graph = helper.make_graph(
            [add_node],
            graph_name,
            [input_tensor],
            [output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertFalse(model_has_infer_metadata(model))
            model_file_path = temp_dir + "/test_load_external_model.onnx"
            onnx.save(model, model_file_path, save_as_external_data=True)
            model_reloaded = load_model_with_shape_infer(Path(model_file_path))
            self.assertTrue(model_has_infer_metadata(model_reloaded))

    def test_pack_bytes_to_4bit(self):
        """
        Tests the pack_bytes_to_4bit() utility.
        """
        subtest_configs = [
            (-8, 6, True),  # Odd num elems, signed
            (-8, 7, True),  # Even num elems, signed
            (0, 14, False),  # Odd num elems, unsigned
            (0, 15, False),  # Even num elems, unsigned
        ]
        for min_val, max_val, signed in subtest_configs:
            with self.subTest(min_val=min_val, max_val=max_val, signed=signed):
                src_float = numpy.arange(min_val, max_val + 1).astype(numpy.float32)
                src_int = src_float.astype(numpy.int8 if signed else numpy.uint8)

                actual_packed_vals = bytes(pack_bytes_to_4bit(src_int.tobytes()))
                expected_packed_vals = onnx.helper.pack_float32_to_4bit(src_float, signed).tobytes()
                self.assertEqual(actual_packed_vals, expected_packed_vals)

    def test_quantize_data_4bit(self):
        """
        Test that calling quantize_data for int4 quantization returns data of the correct type and range.
        """
        data_float = numpy.arange(-20, 17).astype(numpy.float32)

        subtest_configs = [
            (onnx.TensorProto.INT4, True),  # int4, symmetric quant
            (onnx.TensorProto.INT4, False),  # int4, symmetric quant
            (onnx.TensorProto.UINT4, True),  # uint4, symmetric quant
            (onnx.TensorProto.UINT4, False),  # uint4, symmetric quant
        ]

        for onnx_type, symmetric in subtest_configs:
            with self.subTest(onnx_type=onnx_type, symmetric=symmetric):
                zero_point, scale, data_quant = quantize_data(data_float, onnx_type, symmetric)
                is_signed = onnx_type == onnx.TensorProto.INT4
                np_int_type = numpy.int8 if is_signed else numpy.uint8
                qmin = numpy.array(-8 if is_signed else 0, dtype=np_int_type)
                qmax = numpy.array(7 if is_signed else 15, dtype=np_int_type)

                self.assertEqual(zero_point.dtype, np_int_type)
                self.assertEqual(scale.dtype, data_float.dtype)

                expected_zp, expected_scale = compute_scale_zp(
                    data_float.min(), data_float.max(), qmin, qmax, symmetric=symmetric
                )
                self.assertEqual(zero_point, expected_zp)
                self.assertEqual(scale, expected_scale)

                # Even int4 quantization generates 8-bit numpy values.
                self.assertEqual(data_quant.dtype, np_int_type)
                for index, actual_quant_val in enumerate(data_quant.flatten()):
                    self.assertTrue(actual_quant_val >= qmin and actual_quant_val <= qmax)

                    expected_quant_val = numpy.asarray((data_float[index] / scale).round() + zero_point).astype(
                        np_int_type
                    )
                    numpy.clip(expected_quant_val, qmin, qmax, out=expected_quant_val)

                    self.assertEqual(numpy.array(actual_quant_val), expected_quant_val)


if __name__ == "__main__":
    unittest.main()
