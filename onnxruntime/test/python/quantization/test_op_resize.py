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


class TestOpResize(TestCaseTempDir):
    def construct_model_conv_resize(
        self,
        output_model_path,
        conv_input_shape,
        conv_weight_shape,
        resize_input_shape,
        resize_output_shape,
        resize_attrs,
        resize_roi,
        resize_scales,
        resize_sizes,
    ):
        #      (input)
        #          \
        #         Conv
        #        /    \
        #   Identity   Resize
        #    /            \
        # (identity_out)  (output)
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, conv_input_shape)

        conv_weight_arr = np.random.normal(0, 0.1, conv_weight_shape).astype(np.float32)
        conv_weight_initializer = onnx.numpy_helper.from_array(conv_weight_arr, name="conv1_weight")
        conv_node = onnx.helper.make_node("Conv", ["input", "conv1_weight"], ["conv_output"], name="conv_node")

        identity_out = helper.make_tensor_value_info("identity_out", TensorProto.FLOAT, resize_input_shape)
        identity_node = helper.make_node("Identity", ["conv_output"], ["identity_out"], name="IdentityNode")

        initializers = [conv_weight_initializer]

        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, resize_output_shape)
        resize_inputs = ["conv_output"]
        resize_node = helper.make_node("Resize", resize_inputs, ["output"], name="resize_node", **resize_attrs)

        if resize_roi is not None:
            resize_roi_name = "resize_roi"
            resize_roi_initializer = helper.make_tensor(
                resize_roi_name, TensorProto.FLOAT, [len(resize_roi)], resize_roi
            )
            initializers.extend([resize_roi_initializer])
            resize_node.input.extend([resize_roi_name])
        else:
            resize_node.input.extend([""])

        if resize_scales is not None:
            resize_scales_name = "resize_scales"
            resize_scales_initializer = helper.make_tensor(
                resize_scales_name,
                TensorProto.FLOAT,
                [len(resize_scales)],
                resize_scales,
            )
            initializers.extend([resize_scales_initializer])
            resize_node.input.extend([resize_scales_name])
        else:
            resize_node.input.extend([""])

        if resize_sizes is not None:
            resize_sizes_name = "resize_sizes"
            resize_sizes_initializer = helper.make_tensor(
                resize_sizes_name, TensorProto.INT64, [len(resize_sizes)], resize_sizes
            )
            initializers.extend([resize_sizes_initializer])
            resize_node.input.extend([resize_sizes_name])

        graph = helper.make_graph(
            [conv_node, identity_node, resize_node],
            "TestOpQuantizerResize_test_model",
            [input_tensor],
            [identity_out, output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7  # use stable onnx ir version
        onnx.save(model, output_model_path)

    def quantize_resize_test(self, activation_type, weight_type, extra_options={}):
        np.random.seed(1)
        model_fp32_path = "resize_fp32.onnx"
        model_fp32_path = Path(self._tmp_model_dir.name).joinpath(model_fp32_path).as_posix()

        kwargs = {
            "coordinate_transformation_mode": "asymmetric",
            "mode": "nearest",
            "nearest_mode": "floor",
        }
        self.construct_model_conv_resize(
            model_fp32_path,
            [1, 2, 26, 42],
            [3, 2, 3, 3],
            [1, 3, 24, 40],
            [1, 3, 48, 80],
            kwargs,
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 2.0, 2.0],
            None,
        )

        activation_proto_qtype = TensorProto.UINT8 if activation_type == QuantType.QUInt8 else TensorProto.INT8
        activation_type_str = "u8" if (activation_type == QuantType.QUInt8) else "s8"
        weight_type_str = "u8" if (weight_type == QuantType.QUInt8) else "s8"
        model_uint8_path = "resize_{}{}.onnx".format(activation_type_str, weight_type_str)
        model_uint8_path = Path(self._tmp_model_dir.name).joinpath(model_uint8_path).as_posix()
        model_uint8_qdq_path = "resize_{}{}_qdq.onnx".format(activation_type_str, weight_type_str)
        model_uint8_qdq_path = Path(self._tmp_model_dir.name).joinpath(model_uint8_qdq_path).as_posix()
        model_uint8_qdq_dyn_path = "resize_{}{}_qdq_dyn.onnx".format(activation_type_str, weight_type_str)
        model_uint8_qdq_dyn_path = Path(self._tmp_model_dir.name).joinpath(model_uint8_qdq_dyn_path).as_posix()

        # Verify QOperator mode
        data_reader = input_feeds_negone_zero_one(1, {"input": [1, 2, 26, 42]})
        quantize_static(
            model_fp32_path,
            model_uint8_path,
            data_reader,
            quant_format=QuantFormat.QOperator,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
        )
        # make sure resize become xint8 operator, its input name could tell that
        check_op_nodes(
            self,
            model_uint8_path,
            lambda node: (node.name != "resize_node" or node.input[0] != "conv_output"),
        )
        qnode_counts = {
            "QLinearConv": 1,
            "QuantizeLinear": 1,
            "DequantizeLinear": 2,
            "Resize": 1,
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
            "Conv": 1,
            "QuantizeLinear": 3,
            "DequantizeLinear": 4,
            "Resize": 1,
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
            op_types_to_quantize=["Resize", "Conv"],
        )
        qdqnode_counts = {
            "Conv": 1,
            "QuantizeLinear": 1,
            "DequantizeLinear": 2,
            "Resize": 1,
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

    def test_quantize_resize(self):
        self.quantize_resize_test(QuantType.QUInt8, QuantType.QUInt8)

    # TODO: Uncomment following after resize s8 support is enabled
    # def test_quantize_resize_s8s8(self):
    #     self.quantize_resize_test(QuantType.QInt8, QuantType.QInt8, extra_options = {'ActivationSymmetric' : True})


if __name__ == "__main__":
    unittest.main()
