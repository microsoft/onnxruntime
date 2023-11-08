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

from onnxruntime.quantization.quant_utils import compute_scale_zp, load_model_with_shape_infer, model_has_infer_metadata


class TestQuantUtil(unittest.TestCase):
    def test_compute_scale_zp(self):
        self.assertEqual(compute_scale_zp(0.0, 0.0, -127, 127, symmetric=True), [0, 1.0])
        self.assertEqual(compute_scale_zp(1.0, -1.0, -127, 127, symmetric=True), [0, 1.0])
        self.assertEqual(compute_scale_zp(0.0, 0.0, 0, 255, symmetric=True), [0, 1.0])
        self.assertEqual(compute_scale_zp(1.0, -1.0, 0, 255, symmetric=True), [0, 1.0])

        self.assertEqual(compute_scale_zp(-1.0, 2.0, -127, 127, symmetric=True), [0, 2.0 / 127])
        self.assertEqual(compute_scale_zp(-1.0, 2.0, -127, 127, symmetric=False), [-42, 3.0 / 254])

        self.assertEqual(compute_scale_zp(-1.0, 2.0, 0, 255, symmetric=True), [128, 4.0 / 255])
        self.assertEqual(compute_scale_zp(-1.0, 2.0, 0, 255, symmetric=False), [85, 3.0 / 255])

        tiny_float = numpy.float32(numpy.finfo(numpy.float32).tiny * 0.1)
        self.assertEqual(compute_scale_zp(-tiny_float, tiny_float, 0, 255, symmetric=True), [0, 1.0])
        self.assertEqual(compute_scale_zp(-tiny_float, 0.0, 0, 255, symmetric=False), [0, 1.0])

        # Test enforcing a minimum floatint-point range.
        self.assertEqual(compute_scale_zp(0.0, 0.0, 0, 255, symmetric=False, min_real_range=0.0001), [0, 0.0001 / 255])
        self.assertEqual(compute_scale_zp(0.0, 0.0, -128, 127, symmetric=True, min_real_range=0.0001), [0, 0.0002 / 255])
        self.assertEqual(compute_scale_zp(0.0, 0.0, 0, 65535, symmetric=False, min_real_range=0.0001), [0, 0.0001 / 65535])
        self.assertEqual(
            compute_scale_zp(0.0, 0.0, -32768, 32767, symmetric=True, min_real_range=0.0001), [0, 0.0002 / 65535]
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


if __name__ == "__main__":
    unittest.main()
