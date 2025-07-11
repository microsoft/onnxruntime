# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest

import numpy as np
from onnx import TensorProto, helper, save
from op_test_utils import TestDataFeeds, check_model_correctness, check_op_type_count, check_qtype_by_node_type

from onnxruntime.quantization import QuantFormat, QuantType, quantize_static


class TestTopKModel(unittest.TestCase):
    @staticmethod
    def construct_model(model_path, input_shape, axis_attr, k):
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
        k_tensor = helper.make_tensor("k", TensorProto.INT64, [1], [k])
        output_shape = input_shape[:]
        output_shape[axis_attr] = k
        output_values = helper.make_tensor_value_info("values", TensorProto.FLOAT, [1, k])
        output_indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [1, k])

        node = helper.make_node(
            "TopK", inputs=["input", "k"], outputs=["values", "indices"], name="topk_node", axis=axis_attr
        )

        graph = helper.make_graph(
            [node],
            "quant_topk_op_test",
            [input_tensor],
            [output_values, output_indices],
            initializer=[k_tensor],
        )

        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 16), helper.make_opsetid("com.microsoft", 1)]
        )
        save(model, model_path)

    def quantize_topk_test(self, activation_type, weight_type, extra_options={}):  # noqa: B006
        model_fp32_path = "topk_fp32.onnx"
        input_shape = [1, 10]
        axis = 1
        k = 3
        self.construct_model(model_fp32_path, input_shape, axis, k)

        input_data_list = [
            {"input": np.array([[1.8, 2.5, -5.9, 5.2, 4.1, 7.3, 0.2, -0.5, 0.845, 3.9]], dtype=np.float32)}
        ]
        data_reader = TestDataFeeds(input_data_list)

        activation_proto_qtype = TensorProto.UINT8 if activation_type == QuantType.QUInt8 else TensorProto.INT8
        activation_type_str = "u8" if (activation_type == QuantType.QUInt8) else "s8"
        weight_type_str = "u8" if (weight_type == QuantType.QUInt8) else "s8"
        model_qdq_path = f"topk_{activation_type_str}{weight_type_str}_{'QNoInCk' if extra_options['ForceQuantizeNoInputCheck'] else 'NoQNoInCk'}_qdq.onnx"

        # Verify QDQ mode
        data_reader.rewind()
        quantize_static(
            model_fp32_path,
            model_qdq_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
        )
        qdqnode_counts = (
            {
                "TopK": 1,
                "QuantizeLinear": 2,
                "DequantizeLinear": 2,
            }
            if extra_options["ForceQuantizeNoInputCheck"]
            else {
                "TopK": 1,
                "QuantizeLinear": 0,
                "DequantizeLinear": 0,
            }
        )
        check_op_type_count(self, model_qdq_path, **qdqnode_counts)
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        check_qtype_by_node_type(self, model_qdq_path, qnode_io_qtypes)
        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_qdq_path, data_reader.get_next())

    def test_quantize_topk_u8u8(self):
        self.quantize_topk_test(QuantType.QUInt8, QuantType.QUInt8, extra_options={"ForceQuantizeNoInputCheck": True})

    def test_quantize_topk_u8u8_no_force_quantize_no_input_check(self):
        self.quantize_topk_test(QuantType.QUInt8, QuantType.QUInt8, extra_options={"ForceQuantizeNoInputCheck": False})


if __name__ == "__main__":
    unittest.main()
