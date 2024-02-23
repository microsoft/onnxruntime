#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Tests quantization of InstanceNormalization operator.
"""

import unittest

import numpy as np
import onnx
from onnx import TensorProto, parser
from op_test_utils import TestDataFeeds, check_model_correctness, check_op_type_count, check_qtype_by_node_type

from onnxruntime.quantization import QuantFormat, QuantType, quantize_static


class TestOpInstanceNormalization(unittest.TestCase):
    """
    Class with test_* methods that test quantization of the InstanceNormalization operator.
    """

    def input_feeds(self, num_test_inputs, name2shape):
        """
        Returns a data reader of input test data.

        :param num_test_inputs: The number of testing inputs to generate.
        :param name2shape: A dictionary mapping the name of an input to its shape.

        :return: A data reader.
        """
        input_data_list = []
        for _ in range(num_test_inputs):
            inputs = {}
            for name, shape in name2shape.items():
                inputs.update({name: np.random.randint(-1, 2, shape).astype(np.float32)})
            input_data_list.extend([inputs])
        data_reader = TestDataFeeds(input_data_list)
        return data_reader

    def construct_model(self, output_model_path):
        """
        Constructs an ONNX model containing a single InstanceNormalization node, and saves
        the model to the specified output path.

        :param output_model_path: The output filepath in which to save the model.
        """

        model_description = """
        <
          ir_version: 8,
          opset_import: ["" : 13]
        >
        agraph (float[1, 3, 224, 224] input) => (float[1, 3, 224, 224] output)
        <float[3] scale = {1.0, 1.0, 1.0}, float[3] B = {0.0, 0.5, 1.0}>
        {
          output = InstanceNormalization<epsilon : float = 0.009999999776482582>(input, scale, B)
        }
        """

        model = parser.parse_model(model_description)
        onnx.save(model, output_model_path)

    def static_quant_test_qdq(
        self,
        model_fp32_path,
        data_reader,
        activation_type,
        weight_type,
        extra_options=None,
    ):
        """
        Quantizes an FP32 model and checks various properties about the generated quantized model.

        :param model_fp32_path: The path to the FP32 ONNX model to quantize and test.
        :param data_reader: Data reader that generates input data.
        :param activation_type: The quantized type for activations. One of QuantType.QUInt8 or QuantType.QInt8.
        :param weight_type: The quantized type for weights. One of QuantType.QUInt8 or QuantType.QInt8.
        :param extra_options: Dictionary of extra options for the quantize_static() function.
        """

        activation_proto_qtype = TensorProto.UINT8 if activation_type == QuantType.QUInt8 else TensorProto.INT8
        activation_type_str = "u8" if (activation_type == QuantType.QUInt8) else "s8"
        weight_type_str = "u8" if (weight_type == QuantType.QUInt8) else "s8"
        model_int8_path = f"instance_normalization_fp32.quant_dqd_{activation_type_str}{weight_type_str}.onnx"

        data_reader.rewind()
        quantize_static(
            model_fp32_path,
            model_int8_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
        )

        # Check node counts in quantized model.
        quant_node_counts = {"InstanceNormalization": 1, "QuantizeLinear": 2, "DequantizeLinear": 4}
        check_op_type_count(self, model_int8_path, **quant_node_counts)

        # Check input/output types for QuantizeLinear nodes.
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        check_qtype_by_node_type(self, model_int8_path, qnode_io_qtypes)

        # Check model correctness.
        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_int8_path, data_reader.get_next())

    def test_quantize_instance_normalization(self):
        """
        Unit test that quantizes (uint8) an ONNX model containing an InstanceNormalization operator.
        """

        np.random.seed(1)
        model_fp32_path = "instance_normalization_fp32.onnx"
        self.construct_model(model_fp32_path)
        data_reader = self.input_feeds(1, {"input": [1, 3, 224, 224]})

        self.static_quant_test_qdq(
            model_fp32_path,
            data_reader,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QUInt8,
        )

    def test_quantize_instance_normalization_s8s8(self):
        """
        Unit test that quantizes (int8) an ONNX model containing an InstanceNormalization operator.
        """

        np.random.seed(1)
        model_fp32_path = "instance_normalization_fp32.onnx"
        self.construct_model(model_fp32_path)
        data_reader = self.input_feeds(1, {"input": [1, 3, 224, 224]})

        self.static_quant_test_qdq(
            model_fp32_path,
            data_reader,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            extra_options={"ActivationSymmetric": True},
        )


if __name__ == "__main__":
    unittest.main()
