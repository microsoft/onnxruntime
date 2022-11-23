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
from onnx import TensorProto, helper
from op_test_utils import (
    TestCaseTempDir,
    check_model_correctness,
    check_op_nodes,
    check_op_type_count,
    check_qtype_by_node_type,
    input_feeds_negone_zero_one,
)

from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic, quantize_static


class TestOpReshape(TestCaseTempDir):
    def construct_model_matmul_reshape(self, output_model_path, input_shape, weight_shape, output_shape):
        #    (input)
        #      |
        #     MatMul
        #      |
        #    Reshape
        #      |
        #    (output)
        input_name = "input"
        output_name = "output"
        initializers = []

        # make MatMul node
        weight_name = "matmul_weight"
        matmul_output_name = "matmul_output"
        matmul_inputs = [input_name, weight_name]
        matmul_outputs = [matmul_output_name]
        matmul_name = "matmul_node"
        matmul_weight_data = np.random.normal(0, 0.1, weight_shape).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(matmul_weight_data, name=weight_name))

        matmul_node = onnx.helper.make_node("MatMul", matmul_inputs, matmul_outputs, name=matmul_name)

        # make Reshape node
        reshape_shape = "reshape_shape"
        reshape_inputs = [matmul_output_name, reshape_shape]
        reshape_output = [output_name]
        reshape_name = "reshape_node"
        initializers.append(onnx.numpy_helper.from_array(np.array(output_shape, dtype=np.int64), name=reshape_shape))
        reshape_node = onnx.helper.make_node("Reshape", reshape_inputs, reshape_output, name=reshape_name)

        # make graph
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)
        graph_name = "Reshape_Quant_Test"
        graph = helper.make_graph(
            [matmul_node, reshape_node],
            graph_name,
            [input_tensor],
            [output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
        model.ir_version = 7  # use stable onnx ir version

        onnx.save(model, output_model_path)

    def quantize_reshape_test(self, activation_type, weight_type, extra_options={}):
        np.random.seed(1)
        model_fp32_path = "reshape_fp32.onnx"
        model_fp32_path = Path(self._tmp_model_dir.name).joinpath(model_fp32_path).as_posix()

        self.construct_model_matmul_reshape(model_fp32_path, [3, 7], [7, 3], [1, 9])

        activation_proto_qtype = TensorProto.UINT8 if activation_type == QuantType.QUInt8 else TensorProto.INT8
        activation_type_str = "u8" if (activation_type == QuantType.QUInt8) else "s8"
        weight_type_str = "u8" if (weight_type == QuantType.QUInt8) else "s8"
        model_uint8_path = "reshape_{}{}.onnx".format(activation_type_str, weight_type_str)
        model_uint8_path = Path(self._tmp_model_dir.name).joinpath(model_uint8_path).as_posix()
        model_uint8_qdq_path = "reshape_{}{}_qdq.onnx".format(activation_type_str, weight_type_str)
        model_uint8_qdq_path = Path(self._tmp_model_dir.name).joinpath(model_uint8_qdq_path).as_posix()
        model_uint8_qdq_dyn_path = "reshape_{}{}_qdq_dyn.onnx".format(activation_type_str, weight_type_str)
        model_uint8_qdq_dyn_path = Path(self._tmp_model_dir.name).joinpath(model_uint8_qdq_dyn_path).as_posix()

        # Verify QOperator mode
        data_reader = input_feeds_negone_zero_one(1, {"input": [3, 7]})
        quantize_static(
            model_fp32_path,
            model_uint8_path,
            data_reader,
            quant_format=QuantFormat.QOperator,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
        )
        # make sure reshape become xint8 operator, its input name could tell that
        check_op_nodes(
            self,
            model_uint8_path,
            lambda node: (node.name != "reshape_node" or node.input[0] != "matmul_output"),
        )
        qnode_counts = {
            "QLinearMatMul": 1,
            "QuantizeLinear": 1,
            "DequantizeLinear": 1,
            "Reshape": 1,
        }
        check_op_type_count(self, model_uint8_path, **qnode_counts)
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        qnode_io_qtypes.update({"DequantizeLinear": [["i", 2, activation_proto_qtype]]})
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
        qdqnode_counts = {
            "MatMul": 1,
            "QuantizeLinear": 3,
            "DequantizeLinear": 4,
            "Reshape": 1,
        }
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

        # Verify QDQ Dynamic mode
        data_reader.rewind()
        quantize_dynamic(
            model_fp32_path,
            model_uint8_qdq_dyn_path,
            quant_format=QuantFormat.QDQ,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
            op_types_to_quantize=["Reshape", "MatMul"],
        )
        qdqnode_counts = {
            "MatMul": 1,
            "QuantizeLinear": 1,
            "DequantizeLinear": 2,
            "Reshape": 1,
        }
        check_op_type_count(self, model_uint8_qdq_dyn_path, **qdqnode_counts)
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        check_qtype_by_node_type(self, model_uint8_qdq_dyn_path, qnode_io_qtypes)
        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_uint8_qdq_dyn_path, data_reader.get_next())

    def test_quantize_reshape(self):
        self.quantize_reshape_test(QuantType.QUInt8, QuantType.QUInt8)

    def test_quantize_reshape_s8s8(self):
        self.quantize_reshape_test(
            QuantType.QInt8,
            QuantType.QInt8,
            extra_options={"ActivationSymmetric": True},
        )


if __name__ == "__main__":
    unittest.main()
