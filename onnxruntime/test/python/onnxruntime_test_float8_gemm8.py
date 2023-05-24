# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# pylint: disable=C0116,W0212,R1720,C0103,C0114

import unittest

import numpy as np
from numpy.testing import assert_allclose
from onnx import TensorProto
from onnx.checker import check_model
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor, make_tensor_value_info
from onnx.numpy_helper import from_array


class TestFloat8Gemm8(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from onnxruntime import InferenceSession

        cls.InferenceSession = InferenceSession
        # cls.available_providers = [provider for provider in onnxruntime.get_available_providers()]

    def get_model_gemm(self, float_name, alpha=1.0, beta=0.0, transA=1, transB=0, add_bias=False):
        proto_type = getattr(TensorProto, float_name)

        a = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
        b = make_tensor_value_info("B", TensorProto.FLOAT, [None, None])
        c = None if not add_bias else make_tensor_value_info("C", TensorProto.FLOAT, [None, None])
        d = make_tensor_value_info("D", TensorProto.FLOAT, [None, None])

        nodes = [
            make_node("Cast", ["A"], ["Af"], to=proto_type),
            make_node("Cast", ["B"], ["Bf"], to=proto_type),
            None if c is None else make_node("Cast", ["C"], ["Cf"], to=proto_type),
            make_node(
                "Gemm",
                ["Af", "Bf"] if c is None else ["Af", "Bf", "Cf"],
                ["Df"],
                transA=transA,
                transB=transB,
                alpha=alpha,
                beta=beta,
            ),
            make_node("Cast", ["Df"], ["D"], to=TensorProto.FLOAT),
        ]
        nodes = [n for n in nodes if n is not None]
        graph = make_graph(nodes, "gemm", [a, b] if c is None else [a, b, c], [d])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)], ir_version=9)
        check_model(onnx_model)
        return onnx_model

    def get_model_gemm_float8(
        self,
        float_types,
        alpha=1.0,
        beta=0.0,
        transA=1,
        transB=0,
        sm_count=0,
        fastAccumulationMode=1,
        compute_type="CUBLAS_COMPUTE_16F",
        add_bias=False,
    ):
        proto_type = [getattr(TensorProto, float_name) for float_name in float_types]

        a = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
        b = make_tensor_value_info("B", TensorProto.FLOAT, [None, None])
        c = None if not add_bias else make_tensor_value_info("C", TensorProto.FLOAT, [None, None])
        d = make_tensor_value_info("D", TensorProto.FLOAT, [None, None])
        zero = from_array(np.array([0], dtype=np.float32), name="zero")

        nodes = [
            make_node("Cast", ["zero"], ["zerof"], to=proto_type[4]),
            make_node("Cast", ["A"], ["Af"], to=proto_type[0]),
            make_node("Cast", ["B"], ["Bf"], to=proto_type[1]),
            make_node("Cast", ["zero" if c is None else "C"], ["Cf"], to=proto_type[2]),
            make_node(
                "GemmFloat8",
                ["Af", "Bf", "Cf", "zerof", "zerof"],
                ["Df"],
                domain="com.microsoft",
                transA=transA,
                transB=transB,
                smCount=sm_count,
                fastAccumulationMode=fastAccumulationMode,
                alpha=alpha,
                beta=beta,
                name="gemmf8",
                computeType=compute_type,
            ),
            make_node("Cast", ["Df"], ["D"], to=TensorProto.FLOAT),
        ]
        nodes = [n for n in nodes if n is not None]
        graph = make_graph(nodes, "gemmf8model", [a, b] if c is None else [a, b, c], [d], [zero])
        onnx_model = make_model(
            graph, opset_imports=[make_opsetid("", 19), make_opsetid("com.microsoft", 1)], ir_version=9
        )
        check_model(onnx_model)
        return onnx_model

    def common_test_model_gemm(self, float_type, compute_type="CUBLAS_COMPUTE_16F"):
        a = np.arange(9).reshape((3, 3)).astype(np.float32)
        a[:, :] *= 0
        a[0, 1] = 1
        b = (2 ** np.arange(9).reshape((3, 3))).astype(np.float32)
        expected = a.T @ b
        feeds = {"A": a, "B": b}

        onnx_model = self.get_model_gemm("FLOAT")
        if float_type == "FLOAT8E4M3FN":
            float_types = ["FLOAT8E4M3FN", "FLOAT8E4M3FN", "FLOAT8E4M3FN", "FLOAT16", "FLOAT8E4M3FN"]
        elif float_type == "FLOAT8E5M2":
            float_types = ["FLOAT8E5M2", "FLOAT8E4M3FN", "FLOAT8E4M3FN", "FLOAT16", "FLOAT8E4M3FN"]
        elif float_type == "FLOAT16":
            float_types = ["FLOAT16", "FLOAT16", "FLOAT16", "FLOAT16", "FLOAT16"]
        elif float_type == "FLOAT":
            float_types = ["FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT"]
        else:
            raise AssertionError(f"Unexpected float_type={float_type!r}.")

        onnx_model_f8 = self.get_model_gemm_float8(float_types, compute_type=compute_type)
        ref = self.InferenceSession(
            onnx_model.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        y = ref.run(None, feeds)[0]
        with self.subTest(name="Gemm"):
            assert_allclose(expected, y)
            self.assertEqual(expected.shape, y.shape)
            self.assertEqual(expected.dtype, y.dtype)

        ref8 = self.InferenceSession(
            onnx_model_f8.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        y = ref8.run(None, feeds)[0]
        with self.subTest(name="GemmFloat8"):
            assert_allclose(expected, y)
            self.assertEqual(expected.shape, y.shape)
            self.assertEqual(expected.dtype, y.dtype)

    def test_model_gemm_float(self):
        self.common_test_model_gemm("FLOAT", "CUBLAS_COMPUTE_32F")

    def test_model_gemm_float16(self):
        self.common_test_model_gemm("FLOAT16", "CUBLAS_COMPUTE_16F")

    def test_model_gemm_float16_ct32(self):
        self.common_test_model_gemm("FLOAT16", "CUBLAS_COMPUTE_32F")

    def test_model_gemm_float_ct16(self):
        self.common_test_model_gemm("FLOAT", "CUBLAS_COMPUTE_32F_FAST_16F")

    def test_model_gemm_e4m3(self):
        self.common_test_model_gemm("FLOAT8E4M3FN", "CUBLAS_COMPUTE_32F")

    def test_model_gemm_e5m2(self):
        self.common_test_model_gemm("FLOAT8E5M2", "CUBLAS_COMPUTE_32F")


if __name__ == "__main__":
    unittest.main(verbosity=2)
