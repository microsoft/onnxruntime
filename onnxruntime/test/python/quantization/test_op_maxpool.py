#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper
from op_test_utils import (
    TestDataFeeds,
    check_model_correctness,
    check_op_nodes,
    check_op_type_count,
    check_qtype_by_node_type,
)

from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.quant_utils import FLOAT8_TYPES


class TestOpMaxPool(unittest.TestCase):
    def input_feeds(self, n, name2shape):
        input_data_list = []
        for _i in range(n):
            inputs = {}
            for name, shape in name2shape.items():
                inputs.update({name: np.random.randint(-1, 2, shape).astype(np.float32)})
            input_data_list.extend([inputs])
        dr = TestDataFeeds(input_data_list)
        return dr

    def construct_model_conv_maxpool(
        self,
        output_model_path,
        conv_input_shape,
        conv_weight_shape,
        maxpool_input_shape,
        maxpool_attributes,
        output_shape,
    ):
        #      (input)
        #          \
        #         Conv
        #        /    \
        #   Identity   MaxPool
        #    /            \
        # (identity_out)  (output)
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, conv_input_shape)

        conv_weight_arr = np.random.randint(-1, 2, conv_weight_shape).astype(np.float32)
        conv_weight_initializer = onnx.numpy_helper.from_array(conv_weight_arr, name="conv1_weight")
        conv_node = onnx.helper.make_node("Conv", ["input", "conv1_weight"], ["conv_output"], name="conv_node")

        identity_out = helper.make_tensor_value_info("identity_out", TensorProto.FLOAT, maxpool_input_shape)
        identity_node = helper.make_node("Identity", ["conv_output"], ["identity_out"], name="IdentityNode")

        initializers = [conv_weight_initializer]

        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
        maxpool_node = helper.make_node(
            "MaxPool", ["conv_output"], ["output"], name="maxpool_node", **maxpool_attributes
        )

        graph = helper.make_graph(
            [conv_node, identity_node, maxpool_node],
            "TestOpQuantizerMaxPool_test_model",
            [input_tensor],
            [identity_out, output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
        model.ir_version = 7  # use stable onnx ir version
        onnx.save(model, output_model_path)

    def quantize_maxpool_test(self, activation_type, weight_type, extra_options={}):  # noqa: B006
        np.random.seed(1)
        model_fp32_path = "maxpool_fp32.onnx"
        self.construct_model_conv_maxpool(
            model_fp32_path,
            [1, 2, 26, 42],
            [3, 2, 3, 3],
            [1, 3, 24, 40],
            {"kernel_shape": [3, 3]},
            [1, 3, 22, 38],
        )
        data_reader = self.input_feeds(1, {"input": [1, 2, 26, 42]})

        activation_proto_qtype = TensorProto.UINT8 if activation_type == QuantType.QUInt8 else TensorProto.INT8
        activation_type_str = "u8" if (activation_type == QuantType.QUInt8) else "s8"
        weight_type_str = "u8" if (weight_type == QuantType.QUInt8) else "s8"
        model_q8_path = f"maxpool_{activation_type_str}{weight_type_str}.onnx"
        model_q8_qdq_path = f"maxpool_dqd_{activation_type_str}{weight_type_str}.onnx"

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
        # make sure maxpool become xint8 operator, its input name could tell that
        check_op_nodes(
            self,
            model_q8_path,
            lambda node: (node.name != "maxpool_node" or node.input[0] != "conv_output"),
        )
        qnode_counts = {
            "QLinearConv": 1,
            "QuantizeLinear": 1,
            "DequantizeLinear": 2,
            "MaxPool": 1,
        }
        check_op_type_count(self, model_q8_path, **qnode_counts)
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        qnode_io_qtypes.update({"DequantizeLinear": [["i", 2, activation_proto_qtype]]})
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
            "MaxPool": 1,
        }
        check_op_type_count(self, model_q8_qdq_path, **qdqnode_counts)
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        qnode_io_qtypes.update({"DequantizeLinear": [["i", 2, activation_proto_qtype]]})
        check_qtype_by_node_type(self, model_q8_qdq_path, qnode_io_qtypes)
        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_q8_qdq_path, data_reader.get_next())

    def test_quantize_maxpool(self):
        self.quantize_maxpool_test(QuantType.QUInt8, QuantType.QUInt8, extra_options={})

    def test_quantize_maxpool_s8s8(self):
        self.quantize_maxpool_test(
            QuantType.QInt8,
            QuantType.QInt8,
            extra_options={"ActivationSymmetric": True},
        )


if __name__ == "__main__":
    unittest.main()


class TestOpMaxPoolFP8(unittest.TestCase):
    """Tests for MaxPool with FP8 quantization (issue #21090)."""

    def setUp(self):
        np.random.seed(1)

    def construct_model_conv_maxpool(self, output_model_path):
        conv_input_shape = [1, 2, 26, 42]
        conv_weight_shape = [3, 2, 3, 3]
        maxpool_input_shape = [1, 3, 24, 40]
        output_shape = [1, 3, 22, 38]

        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, conv_input_shape)
        conv_weight_arr = np.random.randint(-1, 2, conv_weight_shape).astype(np.float32)
        conv_weight_initializer = onnx.numpy_helper.from_array(conv_weight_arr, name="conv1_weight")
        conv_node = onnx.helper.make_node("Conv", ["input", "conv1_weight"], ["conv_output"], name="conv_node")
        identity_out = helper.make_tensor_value_info("identity_out", TensorProto.FLOAT, maxpool_input_shape)
        identity_node = helper.make_node("Identity", ["conv_output"], ["identity_out"], name="IdentityNode")
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
        maxpool_node = helper.make_node(
            "MaxPool", ["conv_output"], ["output"], name="maxpool_node", kernel_shape=[3, 3]
        )
        graph = helper.make_graph(
            [conv_node, identity_node, maxpool_node],
            "TestMaxPoolFP8",
            [input_tensor],
            [identity_out, output_tensor],
            initializer=[conv_weight_initializer],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
        model.ir_version = 7
        onnx.save(model, output_model_path)

    def test_quantize_maxpool_fp8(self):
        """MaxPool must be left unquantized (pass-through) when FP8 types are used.

        Verifies graph structure only — no InferenceSession, since QLinearConv-FP8
        is not runnable on the CPU EP (separate concern from issue #21090).
        """

        with tempfile.TemporaryDirectory() as tmp:
            model_fp32_path = os.path.join(tmp, "maxpool_fp32.onnx")
            model_fp8_qdq_path = os.path.join(tmp, "maxpool_fp8_qdq.onnx")

            self.construct_model_conv_maxpool(model_fp32_path)

            input_data_list = [{"input": np.random.randint(-1, 2, [1, 2, 26, 42]).astype(np.float32)}]
            data_reader = TestDataFeeds(input_data_list)

            # quantize_static must complete without raising ValueError
            quantize_static(
                model_fp32_path,
                model_fp8_qdq_path,
                data_reader,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QFLOAT8E4M3FN,
                weight_type=QuantType.QFLOAT8E4M3FN,
                calibrate_method=CalibrationMethod.Distribution,
            )

            # Inspect graph structure: MaxPool must not have FP8 tensors on its
            # inputs or outputs, confirming it was skipped during quantization.
            qdq_model = onnx.load(model_fp8_qdq_path)
            tensor_types = {}
            for vi in qdq_model.graph.value_info:
                tensor_types[vi.name] = vi.type.tensor_type.elem_type
            for inp in qdq_model.graph.input:
                tensor_types[inp.name] = inp.type.tensor_type.elem_type
            for out in qdq_model.graph.output:
                tensor_types[out.name] = out.type.tensor_type.elem_type

            for node in qdq_model.graph.node:
                if node.op_type == "MaxPool":
                    for tensor_name in list(node.input) + list(node.output):
                        if tensor_name in tensor_types:
                            self.assertNotIn(
                                tensor_types[tensor_name],
                                FLOAT8_TYPES,
                                f"MaxPool tensor {tensor_name!r} must not be FP8 type",
                            )

            # Assert no QuantizeLinear node whose output feeds directly into MaxPool
            # (i.e., no FP8 QDQ pair was inserted around MaxPool).
            maxpool_inputs = set()
            for node in qdq_model.graph.node:
                if node.op_type == "MaxPool":
                    maxpool_inputs.update(node.input)

            for node in qdq_model.graph.node:
                if node.op_type == "QuantizeLinear":
                    for out in node.output:
                        self.assertNotIn(
                            out,
                            maxpool_inputs,
                            f"QuantizeLinear output {out!r} must not feed directly into MaxPool",
                        )

    def test_quantize_maxpool_fp8_qoperator(self):
        """QMaxPool (QOperator format) guard: MaxPool must remain unquantized under FP8.

        The guard lives in QMaxPool.quantize() in operators/maxpool.py.  When
        activation_qType is in FLOAT8_TYPES the method falls back to
        QuantOperatorBase.quantize(), which copies the node unchanged.  This
        test exercises that path and confirms the output graph still contains
        exactly one plain MaxPool node with no FP8-typed inputs/outputs.
        """
        with tempfile.TemporaryDirectory() as tmp:
            model_fp32_path = os.path.join(tmp, "maxpool_fp32.onnx")
            model_fp8_qop_path = os.path.join(tmp, "maxpool_fp8_qoperator.onnx")

            self.construct_model_conv_maxpool(model_fp32_path)

            input_data_list = [{"input": np.random.randint(-1, 2, [1, 2, 26, 42]).astype(np.float32)}]
            data_reader = TestDataFeeds(input_data_list)

            # quantize_static with QOperator format must complete without raising.
            quantize_static(
                model_fp32_path,
                model_fp8_qop_path,
                data_reader,
                quant_format=QuantFormat.QOperator,
                activation_type=QuantType.QFLOAT8E4M3FN,
                weight_type=QuantType.QFLOAT8E4M3FN,
                calibrate_method=CalibrationMethod.Distribution,
            )

            qop_model = onnx.load(model_fp8_qop_path)

            # Exactly one MaxPool node must survive (unquantized).
            maxpool_nodes = [n for n in qop_model.graph.node if n.op_type == "MaxPool"]
            self.assertEqual(len(maxpool_nodes), 1, "Expected exactly one MaxPool node in QOperator FP8 model")

            # No FP8-typed tensor may appear on MaxPool inputs or outputs.
            tensor_types = {}
            for vi in qop_model.graph.value_info:
                tensor_types[vi.name] = vi.type.tensor_type.elem_type
            for inp in qop_model.graph.input:
                tensor_types[inp.name] = inp.type.tensor_type.elem_type
            for out in qop_model.graph.output:
                tensor_types[out.name] = out.type.tensor_type.elem_type

            for node in maxpool_nodes:
                for tensor_name in list(node.input) + list(node.output):
                    if tensor_name in tensor_types:
                        self.assertNotIn(
                            tensor_types[tensor_name],
                            FLOAT8_TYPES,
                            f"MaxPool tensor {tensor_name!r} must not be FP8 type in QOperator model",
                        )
