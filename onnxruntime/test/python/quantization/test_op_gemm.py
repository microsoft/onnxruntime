#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
import warnings

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
            # QGemm is not implemented for CPU.
            try:
                check_model_correctness(
                    self,
                    model_fp32_path,
                    model_int8_path,
                    data_reader.get_next(),
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    is_gemm=True,
                )
            except Exception as e:
                if (
                    "Type 'tensor(float8e4m3fn)' of input parameter (input_quantized) of operator (QGemm) in node () is invalid."
                    in str(e)
                ):
                    warnings.warn("Fix this test when QGemm is implemented.")
                    return
                raise e
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

    def test_quantize_gemm_s8s8(self):
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
        self.static_quant_test_qdq(
            model_fp32_path,
            data_reader,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            extra_options={"ActivationSymmetric": True},
        )

        # dynamic quantization doesn't support activation:int8
        # self.dynamic_quant_test(model_fp32_path, data_reader, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
        #                        extra_options={'ActivationSymmetric': True})

    def test_quantize_gemm_e4m3fn_same(self):
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
        self.static_quant_test(
            model_fp32_path,
            data_reader,
            activation_type=QuantType.QFLOAT8E4M3FN,
            weight_type=QuantType.QFLOAT8E4M3FN,
            extra_options={"scenario": "same"},
            calibrate_method=CalibrationMethod.Distribution,
        )

    def test_quantize_gemm_e4m3fn_p3(self):
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
        feeds = dict(A=A, scaleA=scaleA, zpA=zpA)
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
        feeds = dict(A=A, scaleA=scaleA, zpA=zpA)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
