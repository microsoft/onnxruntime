#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from op_test_utils import TestDataFeeds  # noqa: F401
from op_test_utils import check_op_type_order  # noqa: F401
from op_test_utils import check_model_correctness, check_op_type_count, check_qtype_by_node_type

from onnxruntime.quantization import DynamicQuantConfig, QuantType, quantize, quantize_dynamic


def generate_input_initializer(tensor_shape, tensor_dtype, input_name):
    """
    Helper function to generate initializers for test inputs
    """
    tensor = np.random.normal(0, 0.3, tensor_shape).astype(tensor_dtype)
    init = numpy_helper.from_array(tensor, input_name)
    return init


class TestONNXModel(unittest.TestCase):
    def construct_model(self, model_path, onnx_type=TensorProto.FLOAT, opset=13, ir_version=7):
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
        input = helper.make_tensor_value_info("input", onnx_type, [4, 2, 8, 8])
        output = helper.make_tensor_value_info("output", onnx_type, [4, 2, 8, 8])

        dtype = onnx.helper.tensor_dtype_to_np_dtype(onnx_type)
        initializers.append(generate_input_initializer([2, 2, 1, 1], dtype, "W1"))
        initializers.append(generate_input_initializer([2, 2, 1, 1], dtype, "W2"))
        initializers.append(generate_input_initializer([2], dtype, "B"))
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
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=ir_version)
        onnx.save(model, model_path)

    def dynamic_quant_conv_test(
        self, onnx_type, opset, ir_version, weight_type, extra_options=None, use_quant_config=False
    ):
        if extra_options is None:
            extra_options = {}
        np.random.seed(1)
        model_fp32_path = "conv_bias.fp32.onnx"
        self.construct_model(model_fp32_path, onnx_type, opset, ir_version)

        activation_proto_qtype = TensorProto.UINT8
        activation_type_str = "u8"
        weight_type_str = "u8" if (weight_type == QuantType.QUInt8) else "s8"
        model_int8_path = f"conv_bias.quant.{activation_type_str}{weight_type_str}.onnx"

        if use_quant_config:
            quant_config = DynamicQuantConfig(weight_type=weight_type, extra_options=extra_options)
            quantize(model_fp32_path, model_int8_path, quant_config)
        else:
            quantize_dynamic(
                model_fp32_path,
                model_int8_path,
                weight_type=weight_type,
                extra_options=extra_options,
            )
        quant_nodes = {"ConvInteger": 2}
        check_op_type_count(self, model_int8_path, **quant_nodes)
        qnode_io_qtypes = {"ConvInteger": [["i", 2, activation_proto_qtype]]}
        check_qtype_by_node_type(self, model_int8_path, qnode_io_qtypes)
        dtype = onnx.helper.tensor_dtype_to_np_dtype(onnx_type)
        check_model_correctness(
            self,
            model_fp32_path,
            model_int8_path,
            {"input": np.random.rand(4, 2, 8, 8).astype(dtype)},
        )

    def test_quant_conv(self):
        for use_quant_config in [True, False]:
            self.dynamic_quant_conv_test(
                TensorProto.FLOAT, 13, 7, QuantType.QUInt8, extra_options={}, use_quant_config=use_quant_config
            )

    @unittest.skipIf(onnx.defs.onnx_opset_version() < 20, reason="Shape inference bug, see onnx PR #5709")
    def test_quant_conv_fp16(self):
        for use_quant_config in [True, False]:
            self.dynamic_quant_conv_test(
                TensorProto.FLOAT16, 19, 9, QuantType.QUInt8, extra_options={}, use_quant_config=use_quant_config
            )

    # TODO: uncomment following after ConvInteger s8 supported
    # def test_quant_conv_s8s8(self):
    #    self.dynamic_quant_conv_test(QuantType.QInt8, extra_options={'ActivationSymmetric': True})


if __name__ == "__main__":
    unittest.main()
