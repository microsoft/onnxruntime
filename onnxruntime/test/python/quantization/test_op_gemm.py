#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
import itertools
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import onnx
from numpy.testing import assert_allclose
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator
from op_test_utils import (
    QGemm,
    TestDataFeeds,
    check_model_correctness,
    check_op_type_count,
    check_qtype_by_node_type,
    onnx_recent_enough,
)

from onnxruntime import InferenceSession
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_dynamic, quantize_static
from onnxruntime.quantization.onnx_model import ONNXModel


class TestOpGemm(unittest.TestCase):
    def input_feeds(self, n, name2shape):
        input_data_list = []
        for _i in range(n):
            inputs = {}
            for name, shape in name2shape.items():
                inputs.update({name: np.random.randint(-1, 2, shape).astype(np.float32)})
            input_data_list.extend([inputs])
        dr = TestDataFeeds(input_data_list)
        return dr

    def construct_model_gemm(self, output_model_path, add_clip=True):
        #      (input)
        #         |
        #        Gemm
        #         |
        #        Clip
        #         |
        #        Gemm
        #         |
        #      (output)
        input_name = "input"
        output_name = "output"
        initializers = []

        def make_gemm(input_name, weight_shape, weight_name, bias_shape, bias_name, output_name):
            weight_data = np.random.normal(0, 0.1, weight_shape).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(weight_data, name=weight_name))

            bias_data = np.random.normal(0, 0.1, bias_shape).astype(np.float32)
            initializers.append(onnx.numpy_helper.from_array(bias_data, name=bias_name))

            return onnx.helper.make_node(
                "Gemm",
                [input_name, weight_name, bias_name],
                [output_name],
                alpha=1.0,
                beta=1.0,
                transB=1,
            )

        # make gemm1 node
        gemm1_output_name = "gemm1_output"
        gemm1_node = make_gemm(
            input_name,
            [100, 10],
            "linear1.weight",
            [100],
            "linear1.bias",
            gemm1_output_name,
        )

        if add_clip:
            # make Clip
            clip_min_name = "clip_min"
            clip_max_name = "clip_max"
            clip_output_name = "clip_output"
            clip_inputs = [gemm1_output_name, clip_min_name, clip_max_name]
            clip_outputs = [clip_output_name]
            initializers.append(onnx.numpy_helper.from_array(np.array(-1.0, dtype=np.float32), name=clip_min_name))
            initializers.append(onnx.numpy_helper.from_array(np.array(1.0, dtype=np.float32), name=clip_max_name))
            clip_node = onnx.helper.make_node("Clip", clip_inputs, clip_outputs)

        else:
            clip_output_name = "clip_output"
            clip_node = onnx.helper.make_node("Identity", [gemm1_output_name], [clip_output_name])

        # make gemm2 node
        gemm2_node = make_gemm(
            clip_output_name,
            [10, 100],
            "linear2.weight",
            [10],
            "linear2.bias",
            output_name,
        )

        # make graph
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [-1, 10])
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [-1, 10])
        graph_name = "gemm_test"
        graph = helper.make_graph(
            [gemm1_node, clip_node, gemm2_node],
            graph_name,
            [input_tensor],
            [output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7  # use stable onnx ir version

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
        model_fp32_path,
        data_reader,
        activation_type,
        weight_type,
        extra_options={},  # noqa: B006
        calibrate_method=CalibrationMethod.MinMax,
    ):
        activation_proto_qtype = activation_type.tensor_type
        activation_type_str = self.str_type(activation_type)
        weight_type_str = self.str_type(weight_type)
        model_int8_path = f"gemm_fp32.quant_{activation_type_str}{weight_type_str}.onnx"

        data_reader.rewind()
        quantize_static(
            model_fp32_path,
            model_int8_path,
            data_reader,
            quant_format=QuantFormat.QOperator,
            activation_type=activation_type,
            weight_type=weight_type,
            extra_options=extra_options,
            calibrate_method=calibrate_method,
        )

        if activation_type == QuantType.QFLOAT8E4M3FN or weight_type == QuantType.QFLOAT8E4M3FN:
            quant_nodes = {"QGemm": 2, "QuantizeLinear": 2, "DequantizeLinear": 2, "Identity": 1}
            qnode_io_qtypes = {
                "QuantizeLinear": [
                    ["i", 2, activation_proto_qtype],
                    ["o", 0, activation_proto_qtype],
                ]
            }
        else:
            qdq_count = 1 if activation_type != QuantType.QInt8 else 2
            clip_count = 0 if activation_type != QuantType.QInt8 else 1
            quant_nodes = {"QGemm": 2, "QuantizeLinear": qdq_count, "DequantizeLinear": qdq_count, "Clip": clip_count}
            qnode_io_qtypes = {
                "QuantizeLinear": [
                    ["i", 2, activation_proto_qtype],
                    ["o", 0, activation_proto_qtype],
                ]
            }

        if activation_type_str == "f8e4m3fn" and weight_type_str == "f8e4m3fn":
            with open(model_int8_path, "rb") as f:
                onx = onnx.load(f)

            nf8 = 0
            for init in onx.graph.initializer:
                if init.data_type not in (TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.FLOAT8E4M3FN):
                    raise AssertionError(f"Unexpected data_type={init.data_type} for initializer {init.name!r}.")
                if init.data_type == TensorProto.FLOAT8E4M3FN:
                    nf8 += 1
            if nf8 < 4:
                raise AssertionError(f"Unexpected low number of float 8 initializer ({nf8}).")

        check_op_type_count(self, model_int8_path, **quant_nodes)
        qnode_io_qtypes.update({"DequantizeLinear": [["i", 2, activation_proto_qtype]]})
        check_qtype_by_node_type(self, model_int8_path, qnode_io_qtypes)
        data_reader.rewind()
        if activation_type_str == "f8e4m3fn" and weight_type_str == "f8e4m3fn":
            # QGemm for float 8 is not implemented. The test should be updated when it is.
            warnings.warn("Fix this test when QGemm is implemented for float 8 types.")
            return
        else:
            check_model_correctness(self, model_fp32_path, model_int8_path, data_reader.get_next(), is_gemm=True)

    def static_quant_test_qdq(
        self,
        model_fp32_path,
        data_reader,
        activation_type,
        weight_type,
        extra_options={},  # noqa: B006
        calibrate_method=CalibrationMethod.MinMax,
    ):
        activation_proto_qtype = activation_type.tensor_type
        activation_type_str = self.str_type(activation_type)
        weight_type_str = self.str_type(weight_type)
        model_int8_path = f"gemm_fp32.quant_dqd_{activation_type_str}{weight_type_str}.onnx"

        data_reader.rewind()
        quantize_static(
            model_fp32_path,
            model_int8_path,
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
            dq_count = 7
            cast_count = 0
        elif activation_type == QuantType.QInt8:
            clip_count = 1
            q_count = 4
            dq_count = 8
            cast_count = 0
        elif activation_type == QuantType.QFLOAT8E4M3FN:
            clip_count = 0
            q_count = 4
            dq_count = 6
            cast_count = 2
        else:
            raise AssertionError(f"Test not implemented for activation_type={activation_type}.")

        quant_nodes = {
            "Gemm": 2,
            "QuantizeLinear": q_count,
            "DequantizeLinear": dq_count,
            "Clip": clip_count,
            "Cast": cast_count,
        }
        check_op_type_count(self, model_int8_path, **quant_nodes)
        qnode_io_qtypes = {
            "QuantizeLinear": [
                ["i", 2, activation_proto_qtype],
                ["o", 0, activation_proto_qtype],
            ]
        }
        check_qtype_by_node_type(self, model_int8_path, qnode_io_qtypes)
        data_reader.rewind()
        check_model_correctness(self, model_fp32_path, model_int8_path, data_reader.get_next(), is_gemm=True)

    def dynamic_quant_test(
        self,
        model_fp32_path,
        data_reader,
        activation_type,
        weight_type,
        extra_options={},  # noqa: B006
    ):
        activation_proto_qtype = TensorProto.UINT8 if activation_type == QuantType.QUInt8 else TensorProto.INT8
        activation_type_str = "u8" if (activation_type == QuantType.QUInt8) else "s8"
        weight_type_str = "u8" if (weight_type == QuantType.QUInt8) else "s8"
        model_int8_path = f"gemm_fp32.quant_dynamic_{activation_type_str}{weight_type_str}.onnx"

        quantize_dynamic(
            model_fp32_path,
            model_int8_path,
            weight_type=weight_type,
            extra_options=extra_options,
        )
        quant_nodes = {"MatMulInteger": 2}
        check_op_type_count(self, model_int8_path, **quant_nodes)
        qnode_io_qtypes = {"MatMulInteger": [["i", 2, activation_proto_qtype]]}
        check_qtype_by_node_type(self, model_int8_path, qnode_io_qtypes)
        data_reader.rewind()
        check_model_correctness(
            self,
            model_fp32_path,
            model_int8_path,
            {"input": np.random.rand(5, 10).astype(np.float32)},
            dynamic=True,
            is_gemm=True,
        )

    def test_quantize_gemm(self):
        np.random.seed(1)
        model_fp32_path = "gemm_fp32.onnx"
        self.construct_model_gemm(model_fp32_path)
        data_reader = self.input_feeds(1, {"input": [5, 10]})

        self.static_quant_test(
            model_fp32_path,
            data_reader,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QUInt8,
        )
        self.static_quant_test_qdq(
            model_fp32_path,
            data_reader,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QUInt8,
        )
        self.dynamic_quant_test(
            model_fp32_path,
            data_reader,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QUInt8,
        )

    @unittest.skip(reason="Shape inference bug, see onnx PR #6080")
    def test_quantize_qop_gemm_s8s8(self):
        np.random.seed(1)
        model_fp32_path = "gemm_fp32.onnx"
        self.construct_model_gemm(model_fp32_path)
        data_reader = self.input_feeds(1, {"input": [5, 10]})

        self.static_quant_test(
            model_fp32_path,
            data_reader,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            extra_options={"ActivationSymmetric": True},
        )

        # dynamic quantization doesn't support activation:int8
        # self.dynamic_quant_test(model_fp32_path, data_reader, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
        #                        extra_options={'ActivationSymmetric': True})

    def test_quantize_qdq_gemm_s8s8(self):
        np.random.seed(1)
        model_fp32_path = "gemm_fp32.onnx"
        self.construct_model_gemm(model_fp32_path)
        data_reader = self.input_feeds(1, {"input": [5, 10]})

        self.static_quant_test_qdq(
            model_fp32_path,
            data_reader,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            extra_options={"ActivationSymmetric": True},
        )

    def test_quantize_qdq_gemm_e4m3fn_same(self):
        np.random.seed(1)
        model_fp32_path = "gemm_fp32.onnx"
        self.construct_model_gemm(model_fp32_path, add_clip=False)
        data_reader = self.input_feeds(1, {"input": [5, 10]})

        self.static_quant_test_qdq(
            model_fp32_path,
            data_reader,
            activation_type=QuantType.QFLOAT8E4M3FN,
            weight_type=QuantType.QFLOAT8E4M3FN,
            extra_options={"scenario": "same"},
            calibrate_method=CalibrationMethod.Distribution,
        )

    @unittest.skip(reason="Shape inference bug, see onnx PR #6080")
    def test_quantize_qop_gemm_e4m3fn_same(self):
        np.random.seed(1)
        model_fp32_path = "gemm_fp32.onnx"
        self.construct_model_gemm(model_fp32_path, add_clip=False)
        data_reader = self.input_feeds(1, {"input": [5, 10]})

        self.static_quant_test(
            model_fp32_path,
            data_reader,
            activation_type=QuantType.QFLOAT8E4M3FN,
            weight_type=QuantType.QFLOAT8E4M3FN,
            extra_options={"scenario": "same"},
            calibrate_method=CalibrationMethod.Distribution,
        )

    def test_quantize_qdq_gemm_e4m3fn_p3(self):
        np.random.seed(1)
        model_fp32_path = "gemm_fp32.onnx"
        self.construct_model_gemm(model_fp32_path, add_clip=False)
        data_reader = self.input_feeds(1, {"input": [5, 10]})

        self.static_quant_test_qdq(
            model_fp32_path,
            data_reader,
            activation_type=QuantType.QFLOAT8E4M3FN,
            weight_type=QuantType.QFLOAT8E4M3FN,
            extra_options={"scenario": "p3"},
            calibrate_method=CalibrationMethod.Distribution,
        )

    @unittest.skip(reason="Shape inference bug, see onnx PR #6080")
    def test_quantize_qop_gemm_e4m3fn_p3(self):
        np.random.seed(1)
        model_fp32_path = "gemm_fp32.onnx"
        self.construct_model_gemm(model_fp32_path, add_clip=False)
        data_reader = self.input_feeds(1, {"input": [5, 10]})

        self.static_quant_test(
            model_fp32_path,
            data_reader,
            activation_type=QuantType.QFLOAT8E4M3FN,
            weight_type=QuantType.QFLOAT8E4M3FN,
            extra_options={"scenario": "p3"},
            calibrate_method=CalibrationMethod.Distribution,
        )

    def test_qgemm_ref_uint8(self):
        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                [
                    onnx.helper.make_node(
                        "QGemm",
                        ["A", "scaleA", "zpA", "B", "scaleB", "zpB", "C", "scale", "zp"],
                        ["Y"],
                        alpha=1.0,
                        transB=1,
                        domain="com.microsoft",
                    )
                ],
                "qgemm_graph",
                [
                    onnx.helper.make_tensor_value_info("A", TensorProto.UINT8, [None, None]),
                    onnx.helper.make_tensor_value_info("scaleA", TensorProto.FLOAT, [1]),
                    onnx.helper.make_tensor_value_info("zpA", TensorProto.UINT8, [1]),
                    onnx.helper.make_tensor_value_info("B", TensorProto.UINT8, [None, None]),
                    onnx.helper.make_tensor_value_info("scaleB", TensorProto.FLOAT, [1]),
                    onnx.helper.make_tensor_value_info("zpB", TensorProto.UINT8, [1]),
                    onnx.helper.make_tensor_value_info("C", TensorProto.INT32, [None, None]),
                    onnx.helper.make_tensor_value_info("scale", TensorProto.FLOAT, [1]),
                    onnx.helper.make_tensor_value_info("zp", TensorProto.UINT8, [1]),
                ],
                [
                    onnx.helper.make_tensor_value_info("Y", TensorProto.UINT8, [None, None]),
                ],
            ),
            opset_imports=[onnx.helper.make_opsetid("", 18), onnx.helper.make_opsetid("com.microsoft", 1)],
        )

        sess = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        ref = ReferenceEvaluator(model, new_ops=[QGemm])

        # simple case
        A = np.array([[2, 1], [1, 0]], dtype=np.uint8)
        scaleA = np.array([1], dtype=np.float32)
        zpA = np.array([0], dtype=np.uint8)
        B = np.array([[0, 1], [1, 3]], dtype=np.uint8)
        scaleB = np.array([1], dtype=np.float32)
        zpB = np.array([0], dtype=np.uint8)
        C = np.array([[0, 0], [0, 0]], dtype=np.int32)
        scale = np.array([1], dtype=np.float32)
        zp = np.array([0], dtype=np.uint8)
        feeds = dict(A=A, scaleA=scaleA, zpA=zpA, B=B, scaleB=scaleB, zpB=zpB, C=C, scale=scale, zp=zp)
        expected = sess.run(None, feeds)[0]
        got = ref.run(None, feeds)[0]
        assert_allclose(expected, got)

        # different scale for A
        scaleA *= 2
        expected = sess.run(None, feeds)[0]
        got = ref.run(None, feeds)[0]
        assert_allclose(expected, got)

        # different scale for B
        scaleB *= 20
        expected = sess.run(None, feeds)[0]
        got = ref.run(None, feeds)[0]
        assert_allclose(expected, got)

        # different scale for output
        scale *= 0.5
        expected = sess.run(None, feeds)[0]
        got = ref.run(None, feeds)[0]
        assert_allclose(expected, got)

        # negative scaleA
        scaleA *= -1
        expected = sess.run(None, feeds)[0]
        got = ref.run(None, feeds)[0]
        assert_allclose(expected, got)

        # zpA != 0
        zpA += 5
        expected = sess.run(None, feeds)[0]
        got = ref.run(None, feeds)[0]
        assert_allclose(expected, got)

        # zpB != 0
        zpB += 105
        expected = sess.run(None, feeds)[0]
        got = ref.run(None, feeds)[0]
        assert_allclose(expected, got)

        # zp != 0
        zp += 77
        expected = sess.run(None, feeds)[0]
        got = ref.run(None, feeds)[0]
        assert_allclose(expected, got)

    def test_qgemm_ref_int8(self):
        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                [
                    onnx.helper.make_node(
                        "QGemm",
                        ["A", "scaleA", "zpA", "B", "scaleB", "zpB", "C", "scale", "zp"],
                        ["Y"],
                        alpha=1.0,
                        transB=1,
                        domain="com.microsoft",
                    )
                ],
                "qgemm_graph",
                [
                    onnx.helper.make_tensor_value_info("A", TensorProto.INT8, [None, None]),
                    onnx.helper.make_tensor_value_info("scaleA", TensorProto.FLOAT, [1]),
                    onnx.helper.make_tensor_value_info("zpA", TensorProto.INT8, [1]),
                    onnx.helper.make_tensor_value_info("B", TensorProto.INT8, [None, None]),
                    onnx.helper.make_tensor_value_info("scaleB", TensorProto.FLOAT, [1]),
                    onnx.helper.make_tensor_value_info("zpB", TensorProto.INT8, [1]),
                    onnx.helper.make_tensor_value_info("C", TensorProto.INT32, [None, None]),
                    onnx.helper.make_tensor_value_info("scale", TensorProto.FLOAT, [1]),
                    onnx.helper.make_tensor_value_info("zp", TensorProto.INT8, [1]),
                ],
                [
                    onnx.helper.make_tensor_value_info("Y", TensorProto.INT8, [None, None]),
                ],
            ),
            opset_imports=[onnx.helper.make_opsetid("", 18), onnx.helper.make_opsetid("com.microsoft", 1)],
        )

        sess = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        ref = ReferenceEvaluator(model, new_ops=[QGemm])

        # simple case
        A = np.array([[2, 1], [1, 0]], dtype=np.int8)
        scaleA = np.array([1], dtype=np.float32)
        zpA = np.array([0], dtype=np.int8)
        B = np.array([[0, 1], [1, 3]], dtype=np.int8)
        scaleB = np.array([1], dtype=np.float32)
        zpB = np.array([0], dtype=np.int8)
        C = np.array([[0, 0], [0, 0]], dtype=np.int32)
        scale = np.array([1], dtype=np.float32)
        zp = np.array([0], dtype=np.int8)
        feeds = {
            "A": A,
            "scaleA": scaleA,
            "zpA": zpA,
            "B": B,
            "scaleB": scaleB,
            "zpB": zpB,
            "C": C,
            "scale": scale,
            "zp": zp,
        }
        expected = sess.run(None, feeds)[0]
        got = ref.run(None, feeds)[0]
        assert_allclose(expected, got)

        # different scale for A
        scaleA *= 2
        expected = sess.run(None, feeds)[0]
        got = ref.run(None, feeds)[0]
        assert_allclose(expected, got)

        # different scale for B
        scaleB *= 20
        expected = sess.run(None, feeds)[0]
        got = ref.run(None, feeds)[0]
        assert_allclose(expected, got)

        # different scale for output
        scale *= 0.5
        expected = sess.run(None, feeds)[0]
        got = ref.run(None, feeds)[0]
        assert_allclose(expected, got)

        # negative scaleA
        scaleA *= -1
        expected = sess.run(None, feeds)[0]
        got = ref.run(None, feeds)[0]
        assert_allclose(expected, got)

        # zpA != 0
        zpA += 5
        expected = sess.run(None, feeds)[0]
        got = ref.run(None, feeds)[0]
        assert_allclose(expected, got)

        # zpB != 0
        zpB += 105
        expected = sess.run(None, feeds)[0]
        got = ref.run(None, feeds)[0]
        assert_allclose(expected, got)

        # zp != 0
        zp -= 77
        expected = sess.run(None, feeds)[0]
        got = ref.run(None, feeds)[0]
        assert_allclose(expected, got)

    def test_q_ref_uint8(self):
        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                [
                    onnx.helper.make_node(
                        "QuantizeLinear",
                        ["A", "scaleA", "zpA"],
                        ["Y"],
                        axis=0,
                    )
                ],
                "qgemm_graph",
                [
                    onnx.helper.make_tensor_value_info("A", TensorProto.FLOAT, [None, None]),
                    onnx.helper.make_tensor_value_info("scaleA", TensorProto.FLOAT, [1]),
                    onnx.helper.make_tensor_value_info("zpA", TensorProto.UINT8, [1]),
                ],
                [
                    onnx.helper.make_tensor_value_info("Y", TensorProto.UINT8, [None, None]),
                ],
            ),
            opset_imports=[onnx.helper.make_opsetid("", 18), onnx.helper.make_opsetid("com.microsoft", 1)],
        )

        sess = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        ref = ReferenceEvaluator(model, new_ops=[QGemm])

        # simple case
        A = np.array([[2, 1], [1, 0]], dtype=np.float32)
        scaleA = np.array([1], dtype=np.float32)
        zpA = np.array([0], dtype=np.uint8)
        feeds = {"A": A, "scaleA": scaleA, "zpA": zpA}
        expected = sess.run(None, feeds)[0]
        got = ref.run(None, feeds)[0]
        assert_allclose(expected, got)

        # different scale for A
        scaleA *= 2
        expected = sess.run(None, feeds)[0]
        got = ref.run(None, feeds)[0]
        if onnx_recent_enough:
            # Test with ReferenceEvaluator requires PR https://github.com/onnx/onnx/pull/5408/.
            assert_allclose(expected, got)
        else:
            self.assertEqual(expected.shape, got.shape)

        # negative scaleA
        scaleA *= -1
        expected = sess.run(None, feeds)[0]
        got = ref.run(None, feeds)[0]
        if onnx_recent_enough:
            # Test with ReferenceEvaluator requires PR https://github.com/onnx/onnx/pull/5408/.
            assert_allclose(expected, got)
        else:
            self.assertEqual(expected.shape, got.shape)

        # zpA != 0
        zpA += 5
        expected = sess.run(None, feeds)[0]
        got = ref.run(None, feeds)[0]
        if onnx_recent_enough:
            # Test with ReferenceEvaluator requires PR https://github.com/onnx/onnx/pull/5408/.
            assert_allclose(expected, got)
        else:
            self.assertEqual(expected.shape, got.shape)

    def test_q_ref_int8(self):
        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                [
                    onnx.helper.make_node(
                        "QuantizeLinear",
                        ["A", "scaleA", "zpA"],
                        ["Y"],
                        axis=0,
                    )
                ],
                "qgemm_graph",
                [
                    onnx.helper.make_tensor_value_info("A", TensorProto.FLOAT, [None, None]),
                    onnx.helper.make_tensor_value_info("scaleA", TensorProto.FLOAT, [1]),
                    onnx.helper.make_tensor_value_info("zpA", TensorProto.INT8, [1]),
                ],
                [
                    onnx.helper.make_tensor_value_info("Y", TensorProto.INT8, [None, None]),
                ],
            ),
            opset_imports=[onnx.helper.make_opsetid("", 18), onnx.helper.make_opsetid("com.microsoft", 1)],
        )

        sess = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        ref = ReferenceEvaluator(model, new_ops=[QGemm])

        # simple case
        A = np.array([[2, 1], [1, 0]], dtype=np.float32)
        scaleA = np.array([1], dtype=np.float32)
        zpA = np.array([0], dtype=np.int8)
        feeds = {"A": A, "scaleA": scaleA, "zpA": zpA}
        expected = sess.run(None, feeds)[0]
        got = ref.run(None, feeds)[0]
        assert_allclose(expected, got)

        # different scale for A
        scaleA *= 2
        expected = sess.run(None, feeds)[0]
        got = ref.run(None, feeds)[0]
        if onnx_recent_enough:
            # Test with ReferenceEvaluator requires PR https://github.com/onnx/onnx/pull/5408/.
            assert_allclose(expected, got)
        else:
            self.assertEqual(expected.shape, got.shape)

        # negative scaleA
        scaleA *= -1
        expected = sess.run(None, feeds)[0]
        got = ref.run(None, feeds)[0]
        if onnx_recent_enough:
            # Test with ReferenceEvaluator requires PR https://github.com/onnx/onnx/pull/5408/.
            assert_allclose(expected, got)
        else:
            self.assertEqual(expected.shape, got.shape)

        # zpA != 0
        zpA += 5
        expected = sess.run(None, feeds)[0]
        got = ref.run(None, feeds)[0]
        if onnx_recent_enough:
            # Test with ReferenceEvaluator requires PR https://github.com/onnx/onnx/pull/5408/.
            assert_allclose(expected, got)
        else:
            self.assertEqual(expected.shape, got.shape)

    def test_qgemm_ref_uint8_specific_example(self):
        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                [
                    onnx.helper.make_node(
                        "QGemm",
                        ["A", "scaleA", "zpA", "B", "scaleB", "zpB", "C", "scale", "zp"],
                        ["Y"],
                        alpha=1.0,
                        transB=1,
                        domain="com.microsoft",
                    )
                ],
                "qgemm_graph",
                [
                    onnx.helper.make_tensor_value_info("A", TensorProto.UINT8, [None, None]),
                    onnx.helper.make_tensor_value_info("scaleA", TensorProto.FLOAT, [1]),
                    onnx.helper.make_tensor_value_info("zpA", TensorProto.UINT8, [1]),
                    onnx.helper.make_tensor_value_info("B", TensorProto.UINT8, [None, None]),
                    onnx.helper.make_tensor_value_info("scaleB", TensorProto.FLOAT, [1]),
                    onnx.helper.make_tensor_value_info("zpB", TensorProto.UINT8, [1]),
                    onnx.helper.make_tensor_value_info("C", TensorProto.INT32, [None, None]),
                    onnx.helper.make_tensor_value_info("scale", TensorProto.FLOAT, [1]),
                    onnx.helper.make_tensor_value_info("zp", TensorProto.UINT8, [1]),
                ],
                [
                    onnx.helper.make_tensor_value_info("Y", TensorProto.UINT8, [None, None]),
                ],
            ),
            opset_imports=[onnx.helper.make_opsetid("", 18), onnx.helper.make_opsetid("com.microsoft", 1)],
        )

        sess = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        ref = ReferenceEvaluator(model, new_ops=[QGemm])

        # simple case
        feeds = {
            "A": np.array([[1, 1, 128, 255, 128, 1, 255], [1, 128, 1, 1, 255, 255, 128]], dtype=np.uint8),
            "B": np.array(
                [[170, 89, 92, 72, 142, 27, 174], [164, 36, 99, 97, 152, 71, 105], [71, 153, 144, 129, 144, 86, 107]],
                dtype=np.uint8,
            ),
            "C": np.array([[-710, -11278, 2355]], dtype=np.int32),
            "scale": np.array([0.00784314], dtype=np.float32),
            "scaleA": np.array([0.0062805], dtype=np.float32),
            "scaleB": np.array([0.00274995], dtype=np.float32),
            "zp": np.array([128], dtype=np.uint8),
            "zpA": np.array([137], dtype=np.uint8),
            "zpB": np.array([111], dtype=np.uint8),
        }

        expected = sess.run(None, feeds)[0]
        got = ref.run(None, feeds)[0]
        assert_allclose(expected, got)

    def test_dynamic_quantization_preserves_transposed_weight(self):
        for shape, value_info, shared, per_channel, bias_name in itertools.product(
            [(3, 8), (3, 3)],
            [False, True],
            [False, True],
            [False, True],
            ["W_Transposed", "W_Transposed_quantized", "W_Transposed_scale", "W_Transposed_zero_point"],
        ):
            with self.subTest(
                shape=shape, value_info=value_info, shared=shared, per_channel=per_channel, bias_name=bias_name
            ):
                rows, cols = shape
                weight = np.arange(rows * cols, dtype=np.float32).reshape(shape) - 4
                bias = np.ones(rows, dtype=np.float32)
                inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, cols])]
                outputs = [helper.make_tensor_value_info("W_Transposed_1", TensorProto.FLOAT, [1, rows])]
                nodes = [helper.make_node("Gemm", ["X", "W", bias_name], ["W_Transposed_1"], transB=1)]
                feeds = {"X": np.arange(cols, dtype=np.float32).reshape(1, cols) / cols}
                if shared:
                    inputs.append(helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, rows]))
                    outputs.extend(
                        [
                            helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [1, rows]),
                            helper.make_tensor_value_info("Y3", TensorProto.FLOAT, [1, cols]),
                            helper.make_tensor_value_info("W_original", TensorProto.FLOAT, shape),
                        ]
                    )
                    nodes.extend(
                        [
                            helper.make_node("Gemm", ["X", "W"], ["Y2"], transB=1),
                            helper.make_node("Gemm", ["Z", "W"], ["Y3"]),
                            helper.make_node("Identity", ["W"], ["W_original"]),
                        ]
                    )
                    feeds["Z"] = np.arange(rows, dtype=np.float32).reshape(1, rows) / rows
                graph = helper.make_graph(
                    nodes,
                    "shared_gemm_weight",
                    inputs,
                    outputs,
                    initializer=[
                        onnx.numpy_helper.from_array(weight, "W"),
                        onnx.numpy_helper.from_array(bias, bias_name),
                    ],
                    value_info=[helper.make_tensor_value_info("W", TensorProto.FLOAT, shape)] if value_info else [],
                )
                model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)], ir_version=9)
                onnx.checker.check_model(model, full_check=True)
                expected = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(
                    None, feeds
                )
                with tempfile.TemporaryDirectory() as directory:
                    output_path = Path(directory) / "quantized.onnx"
                    quantize_dynamic(model, output_path, per_channel=per_channel, weight_type=QuantType.QUInt8)
                    quantized = onnx.load(output_path)
                    onnx.checker.check_model(quantized, full_check=True)
                    self.assertEqual(
                        sum(n.op_type == "MatMulInteger" for n in quantized.graph.node), 3 if shared else 1
                    )
                    if not shared:
                        self.assertNotIn("W", {init.name for init in quantized.graph.initializer})
                    actual = InferenceSession(str(output_path), providers=["CPUExecutionProvider"]).run(None, feeds)
                for expected_output, actual_output in zip(expected, actual, strict=True):
                    assert_allclose(actual_output, expected_output, rtol=0.03, atol=0.15)

    def test_dynamic_gemm_preserves_quantization_overrides(self):
        weight = np.arange(24, dtype=np.float32).reshape(3, 8)
        override_cases = [
            [{"reduce_range": True}],
            [{"scale": np.float32(0.5), "zero_point": np.uint8(3)}],
        ] + [
            [
                {"axis": axis, "scale": np.float32(scale), "zero_point": np.uint8(zp)}
                for scale, zp in [(0.25, 1), (0.5, 2), (1.0, 3)]
            ]
            for axis in (0, -2)
        ]
        for overrides in override_cases:
            with self.subTest(overrides=overrides):
                per_channel = "axis" in overrides[0]
                nodes = [helper.make_node("Gemm", ["X", "W"], ["Y"], transB=1)]
                outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])]
                if not per_channel:
                    nodes.append(helper.make_node("MatMul", ["Z", "W"], ["Z_out"]))
                    outputs.append(helper.make_tensor_value_info("Z_out", TensorProto.FLOAT, [1, 8]))
                model = helper.make_model(
                    helper.make_graph(
                        nodes,
                        "gemm_overrides",
                        [
                            helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8]),
                            helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 3]),
                        ],
                        outputs,
                        initializer=[onnx.numpy_helper.from_array(weight, "W")],
                    ),
                    opset_imports=[helper.make_opsetid("", 20)],
                    ir_version=9,
                )
                options = {"TensorQuantOverrides": {"W": overrides}}
                original_options = copy.deepcopy(options)
                with tempfile.TemporaryDirectory() as directory:
                    output_path = Path(directory) / "quantized.onnx"
                    quantize_dynamic(
                        model, output_path, per_channel=per_channel, weight_type=QuantType.QUInt8, extra_options=options
                    )
                    quantized = onnx.load(output_path)
                onnx.checker.check_model(quantized, full_check=True)
                initializers = {init.name: onnx.numpy_helper.to_array(init) for init in quantized.graph.initializer}
                for name in ["W_Transposed"] if per_channel else ["W", "W_Transposed"]:
                    if "reduce_range" in overrides[0]:
                        self.assertEqual(initializers[name + "_quantized"].max(), 127)
                        assert_allclose(initializers[name + "_scale"], np.float32(23 / 127))
                    else:
                        scales = [override["scale"] for override in overrides]
                        zero_points = [override["zero_point"] for override in overrides]
                        np.testing.assert_array_equal(
                            initializers[name + "_scale"], scales if per_channel else scales[0]
                        )
                        np.testing.assert_array_equal(
                            initializers[name + "_zero_point"], zero_points if per_channel else zero_points[0]
                        )
                self.assertEqual(options["TensorQuantOverrides"], original_options["TensorQuantOverrides"])

    def test_replace_gemm_preserves_outer_scope_weight(self):
        weight = np.arange(24, dtype=np.float32).reshape(3, 8)
        branches = {}
        for branch in ("then_branch", "else_branch"):
            branches[branch] = helper.make_graph(
                [helper.make_node("Gemm", ["X", "W"], ["Y"], transB=1)],
                branch,
                [],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])],
                value_info=[helper.make_tensor_value_info("W", TensorProto.FLOAT, [3, 8])],
            )
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node("If", ["cond"], ["Y"], **branches),
                    helper.make_node("Identity", ["W"], ["original_weight"]),
                ],
                "outer_scope_weight",
                [
                    helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8]),
                    helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
                ],
                [
                    helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3]),
                    helper.make_tensor_value_info("original_weight", TensorProto.FLOAT, [3, 8]),
                ],
                initializer=[onnx.numpy_helper.from_array(weight, "W")],
            ),
            opset_imports=[helper.make_opsetid("", 20)],
            ir_version=9,
        )
        onnx.checker.check_model(model, full_check=True)
        original = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        ONNXModel(model).replace_gemm_with_matmul()
        onnx.checker.check_model(model, full_check=True)
        converted = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        for condition in (True, False):
            feeds = {"X": np.ones((1, 8), dtype=np.float32), "cond": np.array(condition)}
            for expected, actual in zip(original.run(None, feeds), converted.run(None, feeds), strict=True):
                np.testing.assert_array_equal(actual, expected)

    def test_dynamic_quantization(self):
        # dummy_model.onnx from Olive
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "Gemm", ["input", "fc1.weight", "fc1.bias"], ["gemm0"], alpha=1.0, beta=1.0, transB=1
                    ),
                    helper.make_node("Relu", ["gemm0"], ["output"]),
                ],
                "g",
                [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 1])],
                [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 10])],
                [
                    onnx.numpy_helper.from_array(np.random.randn(10, 1).astype(np.float32), name="fc1.weight"),
                    onnx.numpy_helper.from_array(np.random.randn(10).astype(np.float32), name="fc1.bias"),
                ],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
            ir_version=9,
        )
        onnx.checker.check_model(model)
        run_config = {
            "weight_type": QuantType.QInt8,
            "op_types_to_quantize": None,
            "nodes_to_quantize": None,
            "nodes_to_exclude": None,
            "per_channel": False,
            "reduce_range": False,
            "extra_options": {
                "extra.Sigmoid.nnapi": False,
                "ActivationSymmetric": False,
                "WeightSymmetric": True,
                "EnableSubgraph": False,
                "ForceQuantizeNoInputCheck": False,
                "MatMulConstBOnly": True,
            },
        }
        model_path = "test_dynamic_quantization.onnx"
        with open(model_path, "wb") as f:
            f.write(model.SerializeToString())
        qpath = "test_dynamic_quantization.quantized.onnx"
        quantize_dynamic(model_input=model_path, model_output=qpath, use_external_data_format=True, **run_config)
        onx = onnx.load(qpath)
        self.assertIn("DynamicQuantizeLinear", {n.op_type for n in onx.graph.node})


if __name__ == "__main__":
    unittest.main(verbosity=2)
