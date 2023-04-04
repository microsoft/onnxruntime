#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

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
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from onnxruntime.quantization import (
    CalibrationDataReader,
    create_calibrator,
    CalibrationMethod,
    write_calibration_table,
    QuantType,
    QuantizationMode,
    QDQQuantizer,
)
from onnxruntime.quantization.qdq_quantizer import insert_smooth_factors
from onnxruntime.quantization.quant_utils import (
    QuantFormat,
    QuantizationMode,
    QuantType,
    load_model,
    model_has_pre_process_metadata,
)

from pathlib import Path


class TestOpReshape(unittest.TestCase):
    def input_feeds(self, n, name2shape):
        input_data_list = []
        for i in range(n):
            inputs = {}
            for name, shape in name2shape.items():
                # inputs.update({name: np.random.randint(-1, 2, shape).astype(np.float32)})
                inputs.update({name: np.random.randint(-16, 9, shape).astype(np.int64)})
            input_data_list.extend([inputs])
        dr = TestDataFeeds(input_data_list)
        return dr

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
        matmul_weight_data = np.array([[2, 1, -2], [1, -1, -1], [2, -1, -2], [-1, -1, 1]]).astype(np.float32)
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
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7  # use stable onnx ir version

        onnx.save(model, output_model_path)

    def quantize_reshape_test(self, activation_type, weight_type, extra_options={}):
        np.random.seed(1)
        model_fp32_path = "test_smoothquant_fp32.onnx"

        self.construct_model_matmul_reshape(model_fp32_path, [2, 4], [4, 3], [1, 6])

        activation_proto_qtype = TensorProto.UINT8 if activation_type == QuantType.QUInt8 else TensorProto.INT8
        activation_type_str = "u8" if (activation_type == QuantType.QUInt8) else "s8"
        weight_type_str = "u8" if (weight_type == QuantType.QUInt8) else "s8"
        model_uint8_path = "test_smoothquant_{}{}.onnx".format(activation_type_str, weight_type_str)
        model_uint8_qdq_path = "test_smoothquant_{}{}_qdq.onnx".format(activation_type_str, weight_type_str)
        model_inserted_fp32_path = "test_smoothquant_inserted_fp32.onnx"

        # Verify QDQ mode
        data_reader = TestDataFeeds(
            [
                {"input": np.array([[1, -16, 2, 6], [-2, 8, -1, -9]]).astype(np.float32)},
            ]
        )  # self.input_feeds(1, {"input": [3, 7]})
        calibrator = create_calibrator(
            model_fp32_path,
            op_types_to_calibrate=["MatMul", "Gemm"],
            calibrate_method=CalibrationMethod.Smooth,
            use_external_data_format=True,
        )
        calibrator.collect_data(data_reader)
        smooth_factors = calibrator.compute_range()
        print("smooth_factors")
        print(smooth_factors)

        insert_smooth_factors(model_fp32_path, smooth_factors, model_inserted_fp32_path)

        data_reader.rewind()
        op_types_to_quantize = ["MatMul", "Gemm"]

        """
        calibrators_2 = create_calibrator(
            model_inserted_fp32_path,
            op_types_to_calibrate=["MatMul", "Gemm"],
            calibrate_method=CalibrationMethod.MinMax,
            use_external_data_format=False,
        )

        calibrators_2.collect_data(data_reader)
        compute_range = calibrators_2.compute_range()

        model = load_model(Path(model_inserted_fp32_path), True)
        quantizer = QDQQuantizer(
            model,
            False,  # per_channel
            False,  # reduce_range
            QuantizationMode.QLinearOps,
            True,  # static
            QuantType.QInt8,  # weight_type
            QuantType.QInt8,  # activation_type
            compute_range,
            [],  # nodes_to_quantize
            None,
            op_types_to_quantize,
            {
                "ActivationSymmetric": True,
                "MatMulConstBOnly": True,
                "OpTypesToExcludeOutputQuantization": op_types_to_quantize,
                "QDQOpTypePerChannelSupportToAxis": {"MatMul": 1},
            },
        )  # extra_options
        quantizer.quantize_model()
        quantizer.model.save_model_to_file(model_uint8_qdq_path, False)
        """
        # """
        quantize_static(
            model_inserted_fp32_path,
            model_uint8_qdq_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            op_types_to_quantize=op_types_to_quantize,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options={
                "ActivationSymmetric": True,
                "MatMulConstBOnly": True,
                "OpTypesToExcludeOutputQuantization": op_types_to_quantize,
                "QDQOpTypePerChannelSupportToAxis": {"MatMul": 1},
            },
        )
        # """
        """
        calibrators_2 = create_calibrator(
            model_fp32_path,
            op_types_to_calibrate=["MatMul", "Gemm"],
            calibrate_method=CalibrationMethod.MinMax,
            use_external_data_format=False,
        )

        calibrators_2.collect_data(data_reader)
        compute_range = calibrators_2.compute_range()

        model = load_model(Path(model_fp32_path), True)  # onnx.load_model(model_inserted_fp32_path)
        quantizer = QDQQuantizer(
            model,
            False,  # per_channel
            False,  # reduce_range
            QuantizationMode.QLinearOps,
            True,  # static
            QuantType.QInt8,  # weight_type
            QuantType.QInt8,  # activation_type
            compute_range,
            [],  # nodes_to_quantize
            None,
            op_types_to_quantize,
            {
                "ActivationSymmetric": True,
                "MatMulConstBOnly": True,
                "OpTypesToExcludeOutputQuantization": op_types_to_quantize,
                "QDQOpTypePerChannelSupportToAxis": {"MatMul": 1},
            },
        )  # extra_options
        quantizer.quantize_model()
        quantizer.model.save_model_to_file(model_uint8_qdq_path, False)
        """
        """
        quantize_static(
            model_fp32_path,
            model_uint8_qdq_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options={
                "ActivationSymmetric": True,
                "MatMulConstBOnly": True,
                "OpTypesToExcludeOutputQuantization": op_types_to_quantize,
                "QDQOpTypePerChannelSupportToAxis": {"MatMul": 1},
            },
        )
        """

        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_uint8_qdq_path, data_reader.get_next())

    def test_quantize_reshape_s8s8(self):
        op_types_to_quantize = ["MatMul", "Gemm"]
        self.quantize_reshape_test(
            QuantType.QInt8,
            QuantType.QInt8,
            extra_options={
                "ActivationSymmetric": True,
                "MatMulConstBOnly": True,
                "OpTypesToExcludeOutputQuantization": op_types_to_quantize,
                "QDQOpTypePerChannelSupportToAxis": {"MatMul": 1},
            },
        )


if __name__ == "__main__":
    unittest.main()
