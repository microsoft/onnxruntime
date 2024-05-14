#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper
from op_test_utils import TestDataFeeds, check_model_correctness, check_op_type_count

from onnxruntime.quantization import quantize_dynamic


class TestOpAttention(unittest.TestCase):
    def input_feeds(self, n, name2shape):
        input_data_list = []
        for _i in range(n):
            inputs = {}
            for name, shape in name2shape.items():
                inputs.update({name: np.random.randint(-1, 2, shape).astype(np.float32)})
            input_data_list.extend([inputs])
        dr = TestDataFeeds(input_data_list)
        return dr

    def construct_model_attention_and_matmul(self, output_model_path):
        #      (input)
        #         |
        #     Attention
        #         |
        #       MatMul
        #         |
        #      (output)
        input_name = "input"
        output_name = "output"
        initializers = []

        def make_attention_node(input_name, weight_shape, weight_name, bias_shape, bias_name, output_name):
            weight_data = np.random.normal(0, 0.1, weight_shape).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(weight_data, name=weight_name))

            bias_data = np.random.normal(0, 0.1, bias_shape).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(bias_data, name=bias_name))

            return onnx.helper.make_node("Attention", [input_name, weight_name, bias_name], [output_name])

        def make_matmul_node(input_name, weight_shape, weight_name, output_name):
            weight_data = np.random.normal(0, 0.1, weight_shape).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(weight_data, name=weight_name))

            return onnx.helper.make_node("MatMul", [input_name, weight_name], [output_name])

        # make attention node
        attention_output_name = "attention_output"
        attention_node = make_attention_node(
            input_name, [10, 30], "qkv.weight", [30], "qkv.bias", attention_output_name
        )
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend([helper.make_attribute("num_heads", 5)])

        # make matmul node
        matmul_node = make_matmul_node(attention_output_name, [10, 10], "matmul.weight", output_name)

        # make graph
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, -1, 10])
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, -1, 10])
        graph_name = "attention_test"
        graph = helper.make_graph(
            [attention_node, matmul_node],
            graph_name,
            [input_tensor],
            [output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid("com.microsoft", 1)]
        )
        model.ir_version = onnx.IR_VERSION

        onnx.save(model, output_model_path)

    def dynamic_attention_quant_test(self, model_fp32_path, model_int8_path, per_channel, reduce_range):
        quantize_dynamic(
            model_fp32_path,
            model_int8_path,
            per_channel=per_channel,
            reduce_range=reduce_range,
        )
        quant_nodes = {"QAttention": 1, "MatMulInteger": 1}
        check_op_type_count(self, model_int8_path, **quant_nodes)
        check_model_correctness(
            self,
            model_fp32_path,
            model_int8_path,
            {"input": np.random.rand(1, 5, 10).astype(np.float32)},
        )

    def test_quantize_attention(self):
        np.random.seed(1)
        model_fp32_path = "attention_fp32.onnx"
        model_int8_path = "attention_fp32.quant.onnx"
        self.construct_model_attention_and_matmul(model_fp32_path)

        self.dynamic_attention_quant_test(model_fp32_path, model_int8_path, True, True)
        self.dynamic_attention_quant_test(model_fp32_path, model_int8_path, True, False)
        self.dynamic_attention_quant_test(model_fp32_path, model_int8_path, False, True)
        self.dynamic_attention_quant_test(model_fp32_path, model_int8_path, False, False)


if __name__ == "__main__":
    unittest.main()
