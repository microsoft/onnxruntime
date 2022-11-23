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

import onnxruntime
from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic, quantize_static


def check_fraction_correct(testcase, model_path_origin, model_path_to_check, inputs, tolerance=0.05):
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL  # TODO: ENABLE_ALL?

    origin_sess = onnxruntime.InferenceSession(
        model_path_origin, sess_options=sess_options, providers=["CPUExecutionProvider"]
    )
    origin_results = origin_sess.run([], inputs)
    # enable QDQ transformers
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    target_sess = onnxruntime.InferenceSession(
        model_path_to_check,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    target_results = target_sess.run([], inputs)
    testcase.assertEqual(len(origin_results), len(target_results), "result count are different")
    # np.set_printoptions(threshold=sys.maxsize)
    for idx, ref_output in enumerate(origin_results):
        output = target_results[idx]
        a = np.array(output)
        b = np.array(ref_output)
        fraction_wrong = np.sum(a != b) / (a.shape[0] * a.shape[1] * a.shape[2])
        assert fraction_wrong < tolerance, (
            "fraction incorrect (" + str(fraction_wrong) + ") exceeds tolerance (" + str(tolerance) + ")"
        )


class TestOpArgMax(TestCaseTempDir):
    def construct_model_argmax(self, output_model_path, input_shape, output_shape):
        #     (input)
        #        |
        #       Conv
        #        |
        #      ArgMax
        #        |
        #     (output)
        input_name = "input"
        output_name = "output"
        initializers = []

        # make Conv node
        conv_weight_name = "conv_weight"
        # conv_weight_arr = np.random.randint(-1, 2, [32, 256, 1, 1]).astype(np.float32)
        conv_weight_arr = np.random.normal(0.0, 0.1, (32, 256, 1, 1)).astype(np.float32)
        conv_weight_initializer = onnx.numpy_helper.from_array(conv_weight_arr, name=conv_weight_name)
        conv_output_name = "conv_output"
        conv_inputs = [input_name, conv_weight_name]
        conv_outputs = [conv_output_name]
        conv_name = "conv_node"
        conv_node = onnx.helper.make_node(
            "Conv",
            conv_inputs,
            conv_outputs,
            dilations=[1, 1],
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
            name=conv_name,
        )

        # make ArgMax node
        argmax_inputs = [conv_output_name]
        argmax_outputs = [output_name]
        argmax_name = "argmax_node"
        argmax_node = onnx.helper.make_node(
            "ArgMax",
            argmax_inputs,
            argmax_outputs,
            axis=3,
            keepdims=0,
            name=argmax_name,
        )

        initializers = [conv_weight_initializer]

        # make graph
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.INT64, output_shape)
        graph_name = "ArgMax_Quant_Test"
        graph = helper.make_graph(
            [conv_node, argmax_node],
            graph_name,
            [input_tensor],
            [output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7  # use stable onnx ir version

        onnx.save(model, output_model_path)

    def quantize_argmax_test(self, activation_type, weight_type, extra_options=None):
        np.random.seed(1)
        model_fp32_path = "argmax_fp32.onnx"
        model_fp32_path = Path(self._tmp_model_dir.name).joinpath(model_fp32_path).as_posix()

        self.construct_model_argmax(model_fp32_path, [1, 256, 128, 128], [1, 32, 128])

        activation_proto_qtype = TensorProto.UINT8 if activation_type == QuantType.QUInt8 else TensorProto.INT8
        activation_type_str = "u8" if (activation_type == QuantType.QUInt8) else "s8"
        weight_type_str = "u8" if (weight_type == QuantType.QUInt8) else "s8"
        model_uint8_path = (
            Path(self._tmp_model_dir.name).joinpath(f"argmax_{activation_type_str}{weight_type_str}.onnx").as_posix()
        )
        model_uint8_qdq_path = (
            Path(self._tmp_model_dir.name)
            .joinpath(f"argmax_{activation_type_str}{weight_type_str}_qdq.onnx")
            .as_posix()
        )
        model_uint8_qdq_trt_path = (
            Path(self._tmp_model_dir.name)
            .joinpath(f"argmax_{activation_type_str}{weight_type_str}_qdq_trt.onnx")
            .as_posix()
        )
        model_uint8_qdq_dyn_path = (
            Path(self._tmp_model_dir.name)
            .joinpath(f"argmax_{activation_type_str}{weight_type_str}_qdq_dyn.onnx")
            .as_posix()
        )
        model_t_uint8_qdq_dyn_path = (
            Path(self._tmp_model_dir.name)
            .joinpath(f"t_u_argmax_{activation_type_str}{weight_type_str}_qdq_dyn.onnx")
            .as_posix()
        )
        model_t_int8_qdq_dyn_path = (
            Path(self._tmp_model_dir.name)
            .joinpath(f"t_i_argmax_{activation_type_str}{weight_type_str}_qdq_dyn.onnx")
            .as_posix()
        )

        # Verify QOperator mode
        data_reader = input_feeds_negone_zero_one(1, {"input": [1, 256, 128, 128]})
        quantize_static(
            model_fp32_path,
            model_uint8_path,
            data_reader,
            quant_format=QuantFormat.QOperator,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
        )
        # make sure argmax become xint8 operator, its input name could tell that
        check_op_nodes(
            self,
            model_uint8_path,
            lambda node: not (node.name == "argmax_node" and node.input[0] == "conv_output"),
        )
        qnode_counts = {"QuantizeLinear": 1, "QLinearConv": 1, "ArgMax": 1}
        check_op_type_count(self, model_uint8_path, **qnode_counts)
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        check_qtype_by_node_type(self, model_uint8_path, qnode_io_qtypes)
        data_reader.rewind()
        check_fraction_correct(self, model_fp32_path, model_uint8_path, data_reader.get_next())

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
        qdqnode_counts = {"QuantizeLinear": 2, "DequantizeLinear": 3, "ArgMax": 1}
        check_op_type_count(self, model_uint8_qdq_path, **qdqnode_counts)
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        check_qtype_by_node_type(self, model_uint8_qdq_path, qnode_io_qtypes)
        data_reader.rewind()
        check_fraction_correct(self, model_fp32_path, model_uint8_qdq_path, data_reader.get_next())

        # Verify QDQ mode for TensorRT
        data_reader.rewind()
        quantize_static(
            model_fp32_path,
            model_uint8_qdq_trt_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
            op_types_to_quantize=["ArgMax", "Conv"],
        )
        qdqnode_counts = {"QuantizeLinear": 2, "DequantizeLinear": 3, "ArgMax": 1}
        check_op_type_count(self, model_uint8_qdq_trt_path, **qdqnode_counts)
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        check_qtype_by_node_type(self, model_uint8_qdq_trt_path, qnode_io_qtypes)
        data_reader.rewind()
        check_fraction_correct(self, model_fp32_path, model_uint8_qdq_trt_path, data_reader.get_next())

        # Verify QDQ Dynamic
        data_reader.rewind()
        quantize_dynamic(
            model_fp32_path,
            model_uint8_qdq_dyn_path,
            quant_format=QuantFormat.QDQ,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
            op_types_to_quantize=["ArgMax", "Conv"],
        )
        qdqnode_counts = {"QuantizeLinear": 1, "DequantizeLinear": 2, "ArgMax": 1}
        check_op_type_count(self, model_uint8_qdq_dyn_path, **qdqnode_counts)
        data_reader.rewind()
        check_fraction_correct(self, model_fp32_path, model_uint8_qdq_dyn_path, data_reader.get_next())

        data_reader.rewind()
        quantize_dynamic(
            model_fp32_path,
            model_t_uint8_qdq_dyn_path,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=weight_type,
            extra_options=extra_options,
            op_types_to_quantize=["ArgMax", "Conv"],
        )

        data_reader.rewind()
        quantize_dynamic(
            model_fp32_path,
            model_t_int8_qdq_dyn_path,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=weight_type,
            extra_options=extra_options,
            op_types_to_quantize=["ArgMax", "Conv"],
        )

        data_reader.rewind()
        check_fraction_correct(self, model_t_int8_qdq_dyn_path, model_t_uint8_qdq_dyn_path, data_reader.get_next())

    def test_quantize_argmax(self):
        self.quantize_argmax_test(QuantType.QUInt8, QuantType.QUInt8)

    def test_quantize_argmax_s8s8(self):
        self.quantize_argmax_test(
            QuantType.QInt8,
            QuantType.QInt8,
            extra_options={"ActivationSymmetric": True},
        )


if __name__ == "__main__":
    unittest.main()
