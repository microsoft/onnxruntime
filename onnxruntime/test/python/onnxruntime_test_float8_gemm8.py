# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# pylint: disable=C0116,W0212,R1720,C0103,C0114

import os
import platform
import sys
import unittest

import numpy as np
import parameterized
from numpy.testing import assert_allclose
from onnx import TensorProto
from onnx.checker import check_model
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor, make_tensor_value_info
from onnx.numpy_helper import from_array

import onnxruntime

# handle change from python 3.8 and on where loading a dll from the current directory needs to be explicitly allowed.
if platform.system() == "Windows" and sys.version_info[:2] >= (3, 8):
    os.add_dll_directory(os.getcwd())

available_providers = [provider for provider in onnxruntime.get_available_providers()]


class TestInferenceSessionFloat8Gemm8(unittest.TestCase):
    def get_model_gemm(self, float_name, alpha=1.0, beta=0.0, transA=0, transB=0, add_bias=False):
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
        transA=0,
        transB=0,
        sm_count=0,
        fastAccumulationMode=1,
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
                "GemmFloatByte",
                ["Af", "Bf", "Cf", "zerof", "zerof"],
                ["Df"],
                domain="com.microsoft",
                transA=transA,
                transB=transB,
                sm_count=sm_count,
                fastAccumulationMode=fastAccumulationMode,
                alpha=alpha,
                beta=beta,
                name="gemmf8",
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

    def test_model_gemm(self):
        a = np.arange(9).reshape((3, 3)).astype(np.float32)
        b = (2 ** np.arange(9).reshape((3, 3))).astype(np.float32)
        expected = a @ b
        feeds = {"A": a, "B": b}

        onnx_model = self.get_model_gemm("FLOAT")
        onnx_model_f8 = self.get_model_gemm_float8(["FLOAT8E4M3FN", "FLOAT8E4M3FN", "FLOAT16", "FLOAT16", "FLOAT16"])
        ref = onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        y = ref.run(None, feeds)[0]
        assert_allclose(expected, y)
        self.assertEqual(expected.shape, y.shape)
        self.assertEqual(expected.dtype, y.dtype)

        print("C")
        ref8 = onnxruntime.InferenceSession(
            onnx_model_f8.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        print("D")
        y = ref8.run(None, feeds)[0]
        print("E")
        assert_allclose(expected, y)
        self.assertEqual(expected.shape, y.shape)
        self.assertEqual(expected.dtype, y.dtype)


if __name__ == "__main__":
    unittest.main(verbosity=2)
