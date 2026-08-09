# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import unittest

import numpy
from onnx import ModelProto, TensorProto, helper
from onnx.external_data_helper import set_external_data

from onnxruntime.transformers.onnx_utils import extract_raw_data_from_model, has_external_data


class TestOnnxUtils(unittest.TestCase):
    def test_extract_raw_data_from_model(self):
        model = self._get_model_proto_with_raw_data(False)
        external_names, external_values = extract_raw_data_from_model(model)
        self.assertEqual(list(external_names), ["inputs"])
        self.assertEqual(len(external_values), 1)
        self.assertEqual(external_values[0].numpy(), [0.0])

    def test_has_external_data(self):
        model = self._get_model_proto_with_raw_data()
        self.assertTrue(has_external_data(model))

    def test_has_external_data_with_no_external_data(self):
        model = self._get_model_proto_with_raw_data(False)
        self.assertFalse(has_external_data(model))

    def _get_model_proto_with_raw_data(self, has_external_data: bool = True) -> ModelProto:
        input = helper.make_tensor_value_info("inputs", TensorProto.FLOAT, [None])
        output = helper.make_tensor_value_info("outputs", TensorProto.FLOAT, [None])
        raw_data = numpy.array([0.0], dtype=numpy.float32).tobytes()
        tensor = helper.make_tensor("inputs", TensorProto.FLOAT, [1], raw_data, True)
        if has_external_data:
            set_external_data(tensor, location="foo.bin")
        node = helper.make_node("Identity", inputs=["inputs"], outputs=["outputs"])
        return helper.make_model(helper.make_graph([node], "graph", [input], [output], initializer=[tensor]))
