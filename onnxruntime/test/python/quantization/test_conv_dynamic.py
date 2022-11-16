#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from op_test_utils import TestCaseTempDir, check_model_correctness, check_op_type_count, check_qtype_by_node_type

from onnxruntime.quantization import DynamicQuantConfig, QuantFormat, QuantType, quantize, quantize_dynamic


def generate_input_initializer(tensor_shape, tensor_dtype, input_name):
    """
    Helper function to generate initializers for test inputs
    """
    tensor = np.random.normal(0, 0.3, tensor_shape).astype(tensor_dtype)
    init = numpy_helper.from_array(tensor, input_name)
    return init


class TestONNXModel(TestCaseTempDir):
    def construct_model(self, model_path):
        #       input
        #      /    |
        #     /     |
        #  Conv(1)  |
        #     |     |
        #    Relu  Conv(2)
        #     |     |
        #     \     /
        #       Add
        #        |
        #       (output)
        initializers = []
        input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 2, 8, 8])
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [4, 2, 8, 8])

        initializers.append(generate_input_initializer([2, 2, 1, 1], np.float32, "W1"))
        initializers.append(generate_input_initializer([2, 2, 1, 1], np.float32, "W2"))
        initializers.append(generate_input_initializer([2], np.float32, "B"))
        conv_node_1 = onnx.helper.make_node("Conv", ["input", "W1", "B"], ["Conv1_O"], name="Conv1")
        conv_node_2 = onnx.helper.make_node("Conv", ["input", "W2", "B"], ["Conv2_O"], name="Conv2")
        relu_node = onnx.helper.make_node("Relu", ["Conv1_O"], ["Relu_O"], name="Relu")
        add_node = onnx.helper.make_node("Add", ["Relu_O", "Conv2_O"], ["output"], name="Add")
        graph = helper.make_graph(
            [conv_node_1, relu_node, conv_node_2, add_node],
            "onnx_model_test",
            [input],
            [output],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        onnx.save(model, model_path)

    def dynamic_qop_conv_test(self, model_fp32_path, weight_type, extra_options={}, use_quant_config=False):
        activation_proto_qtype = TensorProto.UINT8
        activation_type_str = "u8"
        weight_type_str = "u8" if (weight_type == QuantType.QUInt8) else "s8"
        conf_type_str = ".conf" if (use_quant_config) else ""
        model_qop_path = (
            Path(self._tmp_model_dir.name)
            .joinpath("conv_bias.qop.{}{}{}.onnx".format(activation_type_str, weight_type_str, conf_type_str))
            .as_posix()
        )

        # Test Dynamic QOperator
        if use_quant_config:
            quant_config = DynamicQuantConfig(weight_type=weight_type, extra_options=extra_options)
            quantize(model_fp32_path, model_qop_path, quant_config)
        else:
            quantize_dynamic(
                model_fp32_path,
                model_qop_path,
                weight_type=weight_type,
                extra_options=extra_options,
            )
        quant_nodes = {"ConvInteger": 2}
        check_op_type_count(self, model_qop_path, **quant_nodes)
        qnode_io_qtypes = {"ConvInteger": [["i", 2, activation_proto_qtype]]}
        check_qtype_by_node_type(self, model_qop_path, qnode_io_qtypes)
        check_model_correctness(
            self,
            model_fp32_path,
            model_qop_path,
            {"input": np.random.rand(4, 2, 8, 8).astype(np.float32)},
        )

    def dynamic_qdq_conv_test(self, model_fp32_path, activation_type, weight_type, extra_options={}):
        activation_proto_qtype = TensorProto.UINT8 if activation_type == QuantType.QUInt8 else TensorProto.INT8
        activation_type_str = "u8" if (activation_type == QuantType.QUInt8) else "s8"
        weight_type_str = "u8" if (weight_type == QuantType.QUInt8) else "s8"
        model_qdq_path = (
            Path(self._tmp_model_dir.name)
            .joinpath("conv_bias.qdq.{}{}.onnx".format(activation_type_str, weight_type_str))
            .as_posix()
        )

        # Test Dynamic QDQ
        quantize_dynamic(
            model_fp32_path,
            model_qdq_path,
            quant_format=QuantFormat.QDQ,
            weight_type=weight_type,
            activation_type=activation_type,
            extra_options=extra_options,
        )
        quant_nodes = {"ComputeQuantizationParameters": 1, "QuantizeLinear": 1, "DequantizeLinear": 3, "Conv": 2}
        check_op_type_count(self, model_qdq_path, **quant_nodes)
        # TODO: check weight type
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ],
        }
        check_qtype_by_node_type(self, model_qdq_path, qnode_io_qtypes)
        check_model_correctness(
            self,
            model_fp32_path,
            model_qdq_path,
            {"input": np.random.rand(4, 2, 8, 8).astype(np.float32)},
        )

    def test_quant_conv(self):
        np.random.seed(1)
        model_fp32_path = Path(self._tmp_model_dir.name).joinpath("conv_bias.fp32.onnx").as_posix()
        self.construct_model(model_fp32_path)

        for use_quant_config in [True, False]:
            self.dynamic_qop_conv_test(model_fp32_path, QuantType.QUInt8, use_quant_config=use_quant_config)
        # TODO: uncomment following after ConvInteger s8 supportted
        #    self.dynamic_qop_conv_test(model_fp32_path, QuantType.QInt8, extra_options={'ActivationSymmetric': True})

        for activation_type in [QuantType.QInt8, QuantType.QUInt8]:
            for weight_type in [QuantType.QInt8, QuantType.QUInt8]:
                self.dynamic_qdq_conv_test(model_fp32_path, activation_type, weight_type)


if __name__ == "__main__":
    unittest.main()
