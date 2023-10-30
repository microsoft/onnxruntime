#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Tests quantization of ConvTranspose operator.
"""

import unittest
import packaging.version as pv

import numpy as np
import onnx
from onnx import TensorProto, helper
from op_test_utils import TestDataFeeds, check_model_correctness, check_op_type_count, check_qtype_by_node_type

from onnxruntime.quantization import QuantFormat, QuantType, quantize_static


class TestOpConvTranspose(unittest.TestCase):
    """
    Class with test_* methods that test quantization of the ConvTranspose operator.
    """

    def input_feeds(self, num_test_inputs, name2shape, dtype):
        """
        Returns a data reader of input test data.

        :param num_test_inputs: The number of testing inputs to generate.
        :param name2shape: A dictionary mapping the name of an input to its shape.

        :return: A data reader.
        """
        input_data_list = []
        for _ in range(num_test_inputs):
            inputs = {}
            for name, shape in name2shape.items():
                inputs.update({name: np.random.randint(-1, 2, shape).astype(dtype)})
            input_data_list.extend([inputs])
        data_reader = TestDataFeeds(input_data_list)
        return data_reader

    def construct_model(self, output_model_path, onnx_type=TensorProto.FLOAT, opset=13, ir_version=7):
        """
        Constructs an ONNX model containing a single ConvTranspose node, and saves
        the model to the specified output path.

        :param output_model_path: The output filepath in which to save the model.
        """

        input_tensor = helper.make_tensor_value_info("input", onnx_type, [1, 1, 7, 7])
        output_tensor = helper.make_tensor_value_info("output", onnx_type, [1, 1, 8, 8])
        ini_w = helper.make_tensor("weight", onnx_type, [1, 1, 2, 2], [1.0, 1.0, 1.0, 1.0])
        ini_b = helper.make_tensor("bias", onnx_type, [1], [0.17])
        conv_tranpose_node = onnx.helper.make_node(
            "ConvTranspose",
            ["input", "weight", "bias"],
            ["output"],
            kernel_shape=[2, 2],
            output_padding=[0, 0],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
            dilations=[1, 1],
            group=1,
        )
        graph = helper.make_graph(
            [conv_tranpose_node],
            "conv_transpose_test",
            [input_tensor],
            [output_tensor],
            initializer=[ini_w, ini_b],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
        model.ir_version = ir_version  # use stable onnx ir version

        onnx.save(model, output_model_path)

    def static_quant_test_qdq(
        self,
        model_fp32_path,
        data_reader,
        activation_type,
        weight_type,
        extra_options=None,
    ):
        """
        Quantizes an FP32 model and checks various properties about the generated quantized model.

        :param model_fp32_path: The path to the FP32 ONNX model to quantize and test.
        :param data_reader: Data reader that generates input data.
        :param activation_type: The quantized type for activations. One of QuantType.QUInt8 or QuantType.QInt8.
        :param weight_type: The quantized type for weights. One of QuantType.QUInt8 or QuantType.QInt8.
        :param extra_options: Dictionary of extra options for the quantize_static() function.
        """

        activation_proto_qtype = TensorProto.UINT8 if activation_type == QuantType.QUInt8 else TensorProto.INT8
        activation_type_str = "u8" if (activation_type == QuantType.QUInt8) else "s8"
        weight_type_str = "u8" if (weight_type == QuantType.QUInt8) else "s8"
        model_int8_path = f"conv_transpose_fp32.quant_dqd_{activation_type_str}{weight_type_str}.onnx"

        data_reader.rewind()
        quantize_static(
            model_fp32_path,
            model_int8_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
        )

        # Check node counts in quantized model.
        quant_node_counts = {"ConvTranspose": 1, "QuantizeLinear": 2, "DequantizeLinear": 4}
        check_op_type_count(self, model_int8_path, **quant_node_counts)

        # Check input/output types for QuantizeLinear nodes.
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        check_qtype_by_node_type(self, model_int8_path, qnode_io_qtypes)

        # Check model correctness.
        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_int8_path, data_reader.get_next())

    def quantize_conv_transpose_u8u8(self, onnx_type, opset, ir_version):
        """
        Unit test that quantizes (uint8) an ONNX model containing an ConvTranspose operator.
        """

        np.random.seed(1)
        model_fp32_path = "conv_transpose_fp32.onnx"
        self.construct_model(model_fp32_path, onnx_type, opset, ir_version)
        dtype = onnx.helper.tensor_dtype_to_np_dtype(onnx_type)
        data_reader = self.input_feeds(1, {"input": [1, 1, 7, 7]}, dtype)

        self.static_quant_test_qdq(
            model_fp32_path,
            data_reader,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QUInt8,
        )

    def test_quantize_conv_transpose_u8u8(self):
        self.quantize_conv_transpose_u8u8(TensorProto.FLOAT, 13, 7)

    @unittest.skipIf(
        pv.Version(onnx.__version__) < pv.Version("1.15.1"), reason="Shape inference bug, see onnx PR #5709"
    )
    def test_quantize_conv_transpose_u8u8_fp16(self):
        self.quantize_conv_transpose_u8u8(TensorProto.FLOAT16, 19, 9)

    def quantize_conv_transpose_s8s8(self, onnx_type, opset, ir_version):
        """
        Unit test that quantizes (int8) an ONNX model containing an ConvTranspose operator.
        """

        np.random.seed(1)
        model_fp32_path = "conv_transpose_fp32.onnx"
        self.construct_model(model_fp32_path)
        dtype = onnx.helper.tensor_dtype_to_np_dtype(onnx_type)
        data_reader = self.input_feeds(1, {"input": [1, 1, 7, 7]}, dtype)

        self.static_quant_test_qdq(
            model_fp32_path,
            data_reader,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            extra_options={"ActivationSymmetric": True},
        )

    def test_quantize_conv_transpose_s8s8(self):
        self.quantize_conv_transpose_s8s8(TensorProto.FLOAT, 13, 7)

    @unittest.skipIf(
        pv.Version(onnx.__version__) < pv.Version("1.15.1"), reason="Shape inference bug, see onnx PR #5709"
    )
    def test_quantize_conv_transpose_s8s8_fp16(self):
        self.quantize_conv_transpose_s8s8(TensorProto.FLOAT16, 19, 9)


if __name__ == "__main__":
    unittest.main()
