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
    check_op_type_count,
    check_qtype_by_node_type,
    input_feeds_negone_zero_one,
)

from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic, quantize_static


class TestOpAveragePool(TestCaseTempDir):
    def construct_model_conv_avgpool(
        self,
        output_model_path,
        conv_input_shape,
        conv_weight_shape,
        avgpool_input_shape,
        avgpool_attributes,
        output_shape,
    ):
        #      (input)
        #          \
        #         Conv
        #        /    \
        #   Identity  AveragePool
        #    /            \
        # (identity_out)  (output)
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, conv_input_shape)

        conv_weight_arr = np.random.randint(-1, 2, conv_weight_shape).astype(np.float32)
        conv_weight_initializer = onnx.numpy_helper.from_array(conv_weight_arr, name="conv1_weight")
        conv_node = onnx.helper.make_node("Conv", ["input", "conv1_weight"], ["conv_output"], name="conv_node")

        identity_out = helper.make_tensor_value_info("identity_out", TensorProto.FLOAT, avgpool_input_shape)
        identity_node = helper.make_node("Identity", ["conv_output"], ["identity_out"], name="IdentityNode")

        initializers = [conv_weight_initializer]

        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
        avgpool_node = helper.make_node(
            "AveragePool", ["conv_output"], ["output"], name="avgpool_node", **avgpool_attributes
        )

        graph = helper.make_graph(
            [conv_node, identity_node, avgpool_node],
            "TestOpQuantizerAveragePool_test_model",
            [input_tensor],
            [identity_out, output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
        model.ir_version = 7  # use stable onnx ir version
        onnx.save(model, output_model_path)

    def quantize_avgpool_test(self, activation_type, weight_type, extra_options={}):
        np.random.seed(1)
        model_fp32_path = "avgpool_fp32.onnx"
        model_fp32_path = Path(self._tmp_model_dir.name).joinpath(model_fp32_path).as_posix()
        self.construct_model_conv_avgpool(
            model_fp32_path,
            [1, 2, 26, 42],
            [3, 2, 3, 3],
            [1, 3, 24, 40],
            {"kernel_shape": [3, 3]},
            [1, 3, 22, 38],
        )
        data_reader = input_feeds_negone_zero_one(1, {"input": [1, 2, 26, 42]})

        activation_proto_qtype = TensorProto.UINT8 if activation_type == QuantType.QUInt8 else TensorProto.INT8
        activation_type_str = "u8" if (activation_type == QuantType.QUInt8) else "s8"
        weight_type_str = "u8" if (weight_type == QuantType.QUInt8) else "s8"
        model_q8_path = "avgpool_{}{}.onnx".format(activation_type_str, weight_type_str)
        model_q8_path = Path(self._tmp_model_dir.name).joinpath(model_q8_path).as_posix()
        model_q8_qdq_path = "avgpool_qdq_{}{}.onnx".format(activation_type_str, weight_type_str)
        model_q8_qdq_path = Path(self._tmp_model_dir.name).joinpath(model_q8_qdq_path).as_posix()
        model_q8_qdq_dyn_path = "avgpool_qdq_dyn_{}{}.onnx".format(activation_type_str, weight_type_str)
        model_q8_qdq_dyn_path = Path(self._tmp_model_dir.name).joinpath(model_q8_qdq_dyn_path).as_posix()

        # Verify QOperator mode
        data_reader.rewind()
        quantize_static(
            model_fp32_path,
            model_q8_path,
            data_reader,
            quant_format=QuantFormat.QOperator,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
        )
        qnode_counts = {
            "QLinearConv": 1,
            "QuantizeLinear": 1,
            "DequantizeLinear": 2,
            "QLinearAveragePool": 1,
        }
        check_op_type_count(self, model_q8_path, **qnode_counts)
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        qnode_io_qtypes.update(
            {
                "QLinearConv": [
                    ["i", 2, activation_proto_qtype],
                    ["i", 7, activation_proto_qtype],
                    ["o", 0, activation_proto_qtype],
                ]
            }
        )
        qnode_io_qtypes.update(
            {"QLinearAveragePool": [["i", 4, activation_proto_qtype]]}
        )  # shape info note workig on custom ops
        check_qtype_by_node_type(self, model_q8_path, qnode_io_qtypes)
        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_q8_path, data_reader.get_next())

        # Verify QDQ mode
        data_reader.rewind()
        quantize_static(
            model_fp32_path,
            model_q8_qdq_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
        )
        qdqnode_counts = {
            "Conv": 1,
            "QuantizeLinear": 3,
            "DequantizeLinear": 4,
            "AveragePool": 1,
        }
        check_op_type_count(self, model_q8_qdq_path, **qdqnode_counts)
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        check_qtype_by_node_type(self, model_q8_qdq_path, qnode_io_qtypes)
        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_q8_qdq_path, data_reader.get_next())

        # Verify QDQ Dynamic mode
        data_reader.rewind()
        quantize_dynamic(
            model_fp32_path,
            model_q8_qdq_dyn_path,
            quant_format=QuantFormat.QDQ,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
            op_types_to_quantize=["Conv", "AveragePool"],
        )
        qdqnode_counts = {
            "Conv": 1,
            "QuantizeLinear": 1,
            "DequantizeLinear": 2,
            "AveragePool": 1,
        }
        check_op_type_count(self, model_q8_qdq_dyn_path, **qdqnode_counts)
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        check_qtype_by_node_type(self, model_q8_qdq_dyn_path, qnode_io_qtypes)
        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_q8_qdq_dyn_path, data_reader.get_next())

    def test_quantize_avgpool(self):
        self.quantize_avgpool_test(QuantType.QUInt8, QuantType.QUInt8)

    def test_quantize_avgpool_s8s8(self):
        self.quantize_avgpool_test(
            QuantType.QInt8,
            QuantType.QInt8,
            extra_options={"ActivationSymmetric": True},
        )


if __name__ == "__main__":
    unittest.main()
