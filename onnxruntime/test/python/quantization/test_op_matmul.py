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
from op_test_utils import TestDataFeeds, check_model_correctness, check_op_type_count, check_qtype_by_node_type

from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_dynamic, quantize_static


class TestOpMatMul(unittest.TestCase):
    def input_feeds(self, n, name2shape, dtype):
        input_data_list = []
        for _i in range(n):
            inputs = {}
            for name, shape in name2shape.items():
                inputs.update({name: np.random.randint(-1, 2, shape).astype(dtype)})
            input_data_list.extend([inputs])
        dr = TestDataFeeds(input_data_list)
        return dr

    def construct_model_matmul(
        self, output_model_path, add_clip=True, tensor_type=onnx.TensorProto.FLOAT, opset=18, ir_version=8
    ):
        #      (input)
        #         |
        #        MatMul
        #         |
        #        Clip
        #         |
        #        MatMul
        #         |
        #      (output)
        dtype = np.float32 if tensor_type == onnx.TensorProto.FLOAT else np.float16
        input_name = "input"
        output_name = "output"
        initializers = []

        def make_matmul(input_name, weight_shape, weight_name, output_name):
            weight_data = np.random.normal(0, 0.1, weight_shape).astype(dtype)
            initializers.append(onnx.numpy_helper.from_array(weight_data, name=weight_name))
            return onnx.helper.make_node("MatMul", [input_name, weight_name], [output_name])

        # make mm1 node
        mm1_output_name = "mm1_output"
        mm1_node = make_matmul(
            input_name,
            [10, 100],
            "linear1.weight",
            mm1_output_name,
        )

        if add_clip:
            # make Clip
            clip_min_name = "clip_min"
            clip_max_name = "clip_max"
            clip_output_name = "clip_output"
            clip_inputs = [mm1_output_name, clip_min_name, clip_max_name]
            clip_outputs = [clip_output_name]
            initializers.append(onnx.numpy_helper.from_array(np.array(-1.0, dtype=dtype), name=clip_min_name))
            initializers.append(onnx.numpy_helper.from_array(np.array(1.0, dtype=dtype), name=clip_max_name))
            clip_node = onnx.helper.make_node("Clip", clip_inputs, clip_outputs)

        else:
            clip_output_name = "clip_output"
            clip_node = onnx.helper.make_node("Identity", [mm1_output_name], [clip_output_name])

        # make mm2 node
        mm2_node = make_matmul(
            clip_output_name,
            [100, 10],
            "linear2.weight",
            output_name,
        )

        # make graph
        input_tensor = helper.make_tensor_value_info(input_name, tensor_type, [-1, 10])
        output_tensor = helper.make_tensor_value_info(output_name, tensor_type, [-1, 10])
        graph_name = "matmul_test"
        graph = helper.make_graph(
            [mm1_node, clip_node, mm2_node],
            graph_name,
            [input_tensor],
            [output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
        model.ir_version = ir_version

        onnx.save(model, output_model_path)

    @staticmethod
    def str_type(qtype):
        if qtype == QuantType.QUInt8:
            return "u8"
        if qtype == QuantType.QInt8:
            return "s8"
        if qtype == QuantType.QFLOAT8E4M3FN:
            return "f8e4m3fn"
        raise ValueError(f"Unexpected value for qtype={qtype}")

    def static_quant_test(
        self,
        model_fp_path,
        data_reader,
        activation_type,
        weight_type,
        extra_options={},  # noqa: B006
        calibrate_method=CalibrationMethod.MinMax,
    ):
        activation_proto_qtype = activation_type.tensor_type
        activation_type_str = self.str_type(activation_type)
        weight_type_str = self.str_type(weight_type)
        model_qtype_path = f"matmul_fp.quant_{activation_type_str}{weight_type_str}.onnx"

        data_reader.rewind()
        quantize_static(
            model_fp_path,
            model_qtype_path,
            data_reader,
            quant_format=QuantFormat.QOperator,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
            calibrate_method=calibrate_method,
        )

        if activation_type == QuantType.QFLOAT8E4M3FN or weight_type == QuantType.QFLOAT8E4M3FN:
            quant_nodes = {"QLinearMatMul": 2, "QuantizeLinear": 2, "DequantizeLinear": 2, "Identity": 1}
            qnode_io_qtypes = {
                "QuantizeLinear": [
                    ["i", 2, activation_proto_qtype],
                    ["o", 0, activation_proto_qtype],
                ]
            }
        else:
            qdq_count = 1 if activation_type != QuantType.QInt8 else 2
            clip_count = 0 if activation_type != QuantType.QInt8 else 1
            quant_nodes = {
                "QLinearMatMul": 2,
                "QuantizeLinear": qdq_count,
                "DequantizeLinear": qdq_count,
                "Clip": clip_count,
            }
            qnode_io_qtypes = {
                "QuantizeLinear": [
                    ["i", 2, activation_proto_qtype],
                    ["o", 0, activation_proto_qtype],
                ]
            }

        if activation_type_str == "f8e4m3fn" and weight_type_str == "f8e4m3fn":
            with open(model_qtype_path, "rb") as f:
                onx = onnx.load(f)

            nf8 = 0
            for init in onx.graph.initializer:
                if init.data_type not in (TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.FLOAT8E4M3FN):
                    raise AssertionError(f"Unexpected data_type={init.data_type} for initializer {init.name!r}.")
                if init.data_type == TensorProto.FLOAT8E4M3FN:
                    nf8 += 1
            if nf8 < 4:
                raise AssertionError(f"Unexpected low number of float 8 initializer ({nf8}).")

        check_op_type_count(self, model_qtype_path, **quant_nodes)
        qnode_io_qtypes.update({"DequantizeLinear": [["i", 2, activation_proto_qtype]]})
        if activation_type_str != "f8e4m3fn":
            # QLinearMatMul belongs to domain com.microsoft for this type and shape inference does not work
            check_qtype_by_node_type(self, model_qtype_path, qnode_io_qtypes)
        data_reader.rewind()
        if activation_type_str == "f8e4m3fn" and weight_type_str == "f8e4m3fn":
            check_model_correctness(
                self,
                model_fp_path,
                model_qtype_path,
                data_reader.get_next(),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                is_gemm=True,
                op_matmul=True,
            )
        else:
            check_model_correctness(
                self, model_fp_path, model_qtype_path, data_reader.get_next(), is_gemm=True, op_matmul=True
            )

    def static_quant_test_qdq(
        self,
        model_fp_path,
        data_reader,
        activation_type,
        weight_type,
        extra_options={},  # noqa: B006
        calibrate_method=CalibrationMethod.MinMax,
    ):
        activation_proto_qtype = activation_type.tensor_type
        activation_type_str = self.str_type(activation_type)
        weight_type_str = self.str_type(weight_type)
        model_qtype_path = f"matmul_fp.quant_dqd_{activation_type_str}{weight_type_str}.onnx"

        data_reader.rewind()
        quantize_static(
            model_fp_path,
            model_qtype_path,
            data_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
            calibrate_method=calibrate_method,
        )

        if activation_type == QuantType.QUInt8:
            clip_count = 0
            q_count = 3
            dq_count = 5
            cast_count = 0
        elif activation_type == QuantType.QInt8:
            clip_count = 1
            q_count = 4
            dq_count = 6
            cast_count = 0
        elif activation_type == QuantType.QFLOAT8E4M3FN:
            clip_count = 0
            q_count = 4
            dq_count = 6
            cast_count = 0
        else:
            raise AssertionError(f"Test not implemented for activation_type={activation_type}.")

        quant_nodes = {
            "MatMul": 2,
            "QuantizeLinear": q_count,
            "DequantizeLinear": dq_count,
            "Clip": clip_count,
            "Cast": cast_count,
        }
        check_op_type_count(self, model_qtype_path, **quant_nodes)
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        check_qtype_by_node_type(self, model_qtype_path, qnode_io_qtypes)
        data_reader.rewind()
        check_model_correctness(
            self, model_fp_path, model_qtype_path, data_reader.get_next(), is_gemm=True, op_matmul=True
        )

    def dynamic_quant_test(
        self,
        model_fp_path,
        data_reader,
        activation_type,
        weight_type,
        extra_options={},  # noqa: B006
    ):
        activation_proto_qtype = TensorProto.UINT8 if activation_type == QuantType.QUInt8 else TensorProto.INT8
        activation_type_str = "u8" if (activation_type == QuantType.QUInt8) else "s8"
        weight_type_str = "u8" if (weight_type == QuantType.QUInt8) else "s8"
        model_qtype_path = f"matmul_fp.quant_dynamic_{activation_type_str}{weight_type_str}.onnx"

        quantize_dynamic(
            model_fp_path,
            model_qtype_path,
            weight_type=weight_type,
            extra_options=extra_options,
        )
        quant_nodes = {"MatMulInteger": 2}
        check_op_type_count(self, model_qtype_path, **quant_nodes)
        qnode_io_qtypes = {"MatMulInteger": [["i", 2, activation_proto_qtype]]}
        check_qtype_by_node_type(self, model_qtype_path, qnode_io_qtypes)
        data_reader.rewind()
        onx = onnx.load(model_fp_path)
        tt = onx.graph.input[0].type.tensor_type.elem_type
        check_model_correctness(
            self,
            model_fp_path,
            model_qtype_path,
            {"input": np.random.rand(5, 10).astype(np.float32 if tt == onnx.TensorProto.FLOAT else np.float16)},
            dynamic=True,
            is_gemm=True,
            op_matmul=True,
        )

    def test_quantize_matmul_u8u8(self):
        for tt in [onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16]:
            np.random.seed(1)
            model_fp_path = "matmul_fp.onnx"
            self.construct_model_matmul(model_fp_path, tensor_type=tt)
            data_reader = self.input_feeds(
                1, {"input": [5, 10]}, np.float32 if tt == onnx.TensorProto.FLOAT else np.float16
            )

            self.static_quant_test(
                model_fp_path,
                data_reader,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QUInt8,
            )
            self.static_quant_test_qdq(
                model_fp_path,
                data_reader,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QUInt8,
            )
            self.dynamic_quant_test(
                model_fp_path,
                data_reader,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QUInt8,
            )

    def test_quantize_matmul_s8s8(self):
        for tt in [onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16]:
            np.random.seed(1)
            model_fp_path = "matmul_fp.onnx"
            self.construct_model_matmul(model_fp_path, tensor_type=tt)
            data_reader = self.input_feeds(
                1, {"input": [5, 10]}, np.float32 if tt == onnx.TensorProto.FLOAT else np.float16
            )

            self.static_quant_test(
                model_fp_path,
                data_reader,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                extra_options={"ActivationSymmetric": True},
            )
            self.static_quant_test_qdq(
                model_fp_path,
                data_reader,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                extra_options={"ActivationSymmetric": True},
            )

            # dynamic quantization doesn't support activation:int8
            # self.dynamic_quant_test(model_fp_path, data_reader, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
            #                        extra_options={'ActivationSymmetric': True})

    def test_quantize_matmul_e4m3fn_same(self):
        for tt in [onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16]:
            np.random.seed(1)
            model_fp_path = "matmul_fp.onnx"
            self.construct_model_matmul(model_fp_path, add_clip=False, tensor_type=tt)
            data_reader = self.input_feeds(
                1, {"input": [5, 10]}, np.float32 if tt == onnx.TensorProto.FLOAT else np.float16
            )

            self.static_quant_test_qdq(
                model_fp_path,
                data_reader,
                activation_type=QuantType.QFLOAT8E4M3FN,
                weight_type=QuantType.QFLOAT8E4M3FN,
                extra_options={"scenario": "same"},
                calibrate_method=CalibrationMethod.Distribution,
            )
            self.static_quant_test(
                model_fp_path,
                data_reader,
                activation_type=QuantType.QFLOAT8E4M3FN,
                weight_type=QuantType.QFLOAT8E4M3FN,
                extra_options={"scenario": "same"},
                calibrate_method=CalibrationMethod.Distribution,
            )

    def test_quantize_matmul_e4m3fn_p3(self):
        for tt in [onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16]:
            np.random.seed(1)
            model_fp_path = "matmul_fp.onnx"
            self.construct_model_matmul(model_fp_path, add_clip=False, tensor_type=tt)
            data_reader = self.input_feeds(
                1, {"input": [5, 10]}, np.float32 if tt == onnx.TensorProto.FLOAT else np.float16
            )

            self.static_quant_test_qdq(
                model_fp_path,
                data_reader,
                activation_type=QuantType.QFLOAT8E4M3FN,
                weight_type=QuantType.QFLOAT8E4M3FN,
                extra_options={"scenario": "p3"},
                calibrate_method=CalibrationMethod.Distribution,
            )
            self.static_quant_test(
                model_fp_path,
                data_reader,
                activation_type=QuantType.QFLOAT8E4M3FN,
                weight_type=QuantType.QFLOAT8E4M3FN,
                extra_options={"scenario": "p3"},
                calibrate_method=CalibrationMethod.Distribution,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
