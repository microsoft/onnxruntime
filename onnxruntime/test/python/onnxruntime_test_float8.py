# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# pylint: disable=C0116,W0212,R1720,C0103,C0114

import platform
import unittest

import numpy as np
import onnx
from numpy.testing import assert_allclose
from onnx import TensorProto
from onnx.checker import check_model
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.reference import ReferenceEvaluator

import onnxruntime

# handle change from python 3.8 and on where loading a dll from the current directory needs to be explicitly allowed.
if platform.system() == "Windows" and sys.version_info.major >= 3 and sys.version_info.minor >= 8:
    os.add_dll_directory(os.getcwd())

available_providers = [provider for provider in onnxruntime.get_available_providers()]


class TestInferenceSession(unittest.TestCase):
    x = np.array([0, 1e-7, 1e-3, 1e-2, 1e-1, 1, 2, 10, 100, 1000, 1e4, 1e5, np.inf, -np.inf, np.nan], dtype=np.float32)
    expected = (
        {}
        if not hasattr(TensorProto, "FLOAT8E4M3FN")
        else {
            TensorProto.FLOAT8E4M3FN: np.array(
                [
                    0.000000e00,
                    0.000000e00,
                    1.953125e-03,
                    9.765625e-03,
                    1.015625e-01,
                    1.000000e00,
                    2.000000e00,
                    1.000000e01,
                    1.040000e02,
                    4.480000e02,
                    4.480000e02,
                    4.480000e02,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                dtype=np.float32,
            ),
            TensorProto.FLOAT8E4M3FNUZ: np.array(
                [
                    0.000000e00,
                    0.000000e00,
                    9.765625e-04,
                    9.765625e-03,
                    1.015625e-01,
                    1.000000e00,
                    2.000000e00,
                    1.000000e01,
                    1.040000e02,
                    2.400000e02,
                    2.400000e02,
                    2.400000e02,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                dtype=np.float32,
            ),
            TensorProto.FLOAT8E5M2: np.array(
                [
                    0.000000e00,
                    0.000000e00,
                    9.765625e-04,
                    9.765625e-03,
                    9.375000e-02,
                    1.000000e00,
                    2.000000e00,
                    1.000000e01,
                    9.600000e01,
                    1.024000e03,
                    1.024000e04,
                    5.734400e04,
                    np.inf,
                    -np.inf,
                    np.nan,
                ],
                dtype=np.float32,
            ),
            TensorProto.FLOAT8E5M2FNUZ: np.array(
                [
                    0.000000e00,
                    0.000000e00,
                    9.765625e-04,
                    9.765625e-03,
                    9.375000e-02,
                    1.000000e00,
                    2.000000e00,
                    1.000000e01,
                    9.600000e01,
                    1.024000e03,
                    1.024000e04,
                    5.734400e04,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                dtype=np.float32,
            ),
        }
    )

    def model_cast_cast(self, to):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("Cast", ["X"], ["T"], to=to)
        node2 = make_node("Cast", ["T"], ["Y"], to=TensorProto.FLOAT)
        graph = make_graph([node1, node2], "lr", [X], [Y])
        onnx_model = make_model(graph)
        check_model(onnx_model)
        return onnx_model

    @unittest.skipIf(not hasattr(TensorProto, "FLOAT8E4M3FN"), reason="needs onnx>=1.14.0")
    def test_model_cast_cast_reference(self):
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

        expected = TestInferenceSession.expected
        x = TestInferenceSession.x

        for to, expect in expected.items():
            with self.subTest(to=to):
                onnx_model = self.model_cast_cast(to)
                ref = ReferenceEvaluator(onnx_model)
                y = ref.run(None, {"X": x})[0]
                assert_allclose(expect, y)
                self.assertEqual(expect.shape, y.shape)
                self.assertEqual(expect.dtype, y.dtype)

    @unittest.skipIf(not hasattr(TensorProto, "FLOAT8E4M3FN"), reason="needs onnx>=1.14.0")
    def test_model_cast_cast_cpu(self):
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

        expected = TestInferenceSession.expected
        x = TestInferenceSession.x

        for to, expect in expected.items():
            with self.subTest(to=to):
                onnx_model = self.model_cast_cast(to)
                sess = onnxruntime.InferenceSession(
                    onnx_model.SerializeToString(), so, providers=["CPUExecutionProvider"]
                )
                y = sess.run(None, {"X": x})[0]
                assert_allclose(expect, y)
                self.assertEqual(expect.shape, y.shape)
                self.assertEqual(expect.dtype, y.dtype)

    @unittest.skipIf(not hasattr(TensorProto, "FLOAT8E4M3FN"), reason="needs onnx>=1.14.0")
    @unittest.skipIf("CUDAExecutionProvider" not in available_providers, reason="Not running on CUDA.")
    def test_model_cast_cast_cuda(self):
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

        expected = TestInferenceSession.expected
        x = TestInferenceSession.x

        for to, expect in expected.items():
            with self.subTest(to=to):
                if to not in {TensorProto.FLOAT8E4M3FN, TensorProto.FLOAT8E5M2}:
                    # only those types are available on CUDA.
                    continue
                onnx_model = self.model_cast_cast(to)
                sess = onnxruntime.InferenceSession(
                    onnx_model.SerializeToString(), so, providers=["CUDAExecutionProvider"]
                )
                y = sess.run(None, {"X": x})[0]
                assert_allclose(expect, y)
                self.assertEqual(expect.shape, y.shape)
                self.assertEqual(expect.dtype, y.dtype)

    @unittest.skipIf(not hasattr(TensorProto, "FLOAT8E4M3FN"), reason="needs onnx>=1.14.0")
    @unittest.skipIf("CUDAExecutionProvider" not in available_providers, reason="Not running on CUDA.")
    def test_model_cast_cast_cuda_ortvalue(self):
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

        expected = TestInferenceSession.expected
        x = TestInferenceSession.x

        for to, expect in expected.items():
            with self.subTest(to=to):
                if to not in {TensorProto.FLOAT8E4M3FN, TensorProto.FLOAT8E5M2}:
                    # only those types are available on CUDA.
                    continue
                onnx_model = self.model_cast_cast(to)
                sess = onnxruntime.InferenceSession(
                    onnx_model.SerializeToString(), so, providers=["CUDAExecutionProvider"]
                )
                ortv = onnxruntime.OrtValue.ortvalue_from_numpy(x, device="cuda")
                y = sess.run_with_ort_values(["Y"], {"X": ortv})[0].numpy()
                assert_allclose(expect, y)
                self.assertEqual(expect.shape, y.shape)
                self.assertEqual(expect.dtype, y.dtype)


if __name__ == "__main__":
    # TestInferenceSession().test_model_cast_cast_cuda()
    unittest.main()
