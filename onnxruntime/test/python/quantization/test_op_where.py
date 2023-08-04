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


class TestWhereModel(unittest.TestCase):
    @staticmethod
    def input_feeds_for_where(n, name2shape):
        input_data_list = []
        for _i in range(n):
            inputs = {}
            for name, shape in name2shape.items():
                if name == "condition":
                    inputs.update({name: np.ones(shape).astype(np.bool_)})
                else:
                    inputs.update({name: np.ones(shape).astype(np.float32)})
            input_data_list.extend([inputs])

        dr = TestDataFeeds(input_data_list)
        return dr

    @staticmethod
    def construct_model(model_path, input_shape):
        initializers = []
        input_condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, input_shape)
        input_x = helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)
        input_y = helper.make_tensor_value_info("y", TensorProto.FLOAT, input_shape)
        out_put = helper.make_tensor_value_info("z", TensorProto.FLOAT, input_shape)
        node = helper.make_node(
            "Where",
            inputs=["condition", "x", "y"],
            outputs=["z"],
            name="where_node",
        )

        graph = helper.make_graph(
            [node],
            "quant_where_op_test",
            [input_condition, input_x, input_y],
            [out_put],
            initializer=initializers,
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 16), helper.make_opsetid("com.microsoft", 1)]
        )
        save(model, model_path)

    def quantize_where_test(self, activation_type, weight_type, extra_options={}):  # noqa: B006
        model_fp32_path = "where_fp32.onnx"
        input_shape = [2, 2]
        self.construct_model(model_fp32_path, input_shape)
        data_reader = self.input_feeds_for_where(
            1,
            {
                "condition": input_shape,
                "x": input_shape,
                "y": input_shape,
            },
        )
        activation_proto_qtype = TensorProto.UINT8 if activation_type == QuantType.QUInt8 else TensorProto.INT8
        activation_type_str = "u8" if (activation_type == QuantType.QUInt8) else "s8"
        weight_type_str = "u8" if (weight_type == QuantType.QUInt8) else "s8"
        model_uint8_path = f"where_{activation_type_str}{weight_type_str}_{'QNoInCk' if extra_options['ForceQuantizeNoInputCheck'] else 'NoQNoInCk'}.onnx"
        model_uint8_qdq_path = f"where_{activation_type_str}{weight_type_str}_{'QNoInCk' if extra_options['ForceQuantizeNoInputCheck'] else 'NoQNoInCk'}_qdq.onnx"

        # Verify QOperator mode
        data_reader.rewind()
        quantize_static(
            model_fp32_path,
            model_uint8_path,
            data_reader,
            quant_format=QuantFormat.QOperator,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
        )
        qnode_counts = (
            {
                "QLinearWhere": 1,
                "QuantizeLinear": 2,
                "DequantizeLinear": 1,
            }
            if extra_options["ForceQuantizeNoInputCheck"]
            else {
                "Where": 1,
                "QuantizeLinear": 0,
                "DequantizeLinear": 0,
            }
        )
        check_op_type_count(self, model_uint8_path, **qnode_counts)
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        check_qtype_by_node_type(self, model_uint8_path, qnode_io_qtypes)
        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_uint8_path, data_reader.get_next())

        # Verify QDQ mode
        data_reader.rewind()
        quantize_static(
            model_fp32_path,
            model_uint8_qdq_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
        )
        qdqnode_counts = (
            {
                "Where": 1,
                "QuantizeLinear": 3,
                "DequantizeLinear": 3,
            }
            if extra_options["ForceQuantizeNoInputCheck"]
            else {
                "Where": 1,
                "QuantizeLinear": 0,
                "DequantizeLinear": 0,
            }
        )
        check_op_type_count(self, model_uint8_qdq_path, **qdqnode_counts)
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        check_qtype_by_node_type(self, model_uint8_qdq_path, qnode_io_qtypes)
        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_uint8_qdq_path, data_reader.get_next())

    def test_quantize_where_u8u8(self):
        self.quantize_where_test(QuantType.QUInt8, QuantType.QUInt8, extra_options={"ForceQuantizeNoInputCheck": True})
        print(__name__)

    def test_quantize_where_u8u8_no_force_quantize_no_input_check(self):
        self.quantize_where_test(QuantType.QUInt8, QuantType.QUInt8, extra_options={"ForceQuantizeNoInputCheck": False})
        print(__name__)


if __name__ == "__main__":
    unittest.main()
