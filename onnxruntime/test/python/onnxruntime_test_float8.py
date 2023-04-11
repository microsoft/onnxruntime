# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# pylint: disable=C0116,W0212,R1720,C0103,C0114

import os
import platform
import sys
import unittest

import numpy as np
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
    x = np.array(
        [0.4068359375, 352, 416, 336, 304, 272, -248, -100, 1e-4, 1e-2, 416, 432, 1e5, np.inf, -np.inf, np.nan],
        dtype=np.float32,
    )
    expected_saturate = (
        {}
        if not hasattr(TensorProto, "FLOAT8E4M3FN")
        else {
            TensorProto.FLOAT8E4M3FN: np.array(
                [
                    0.40625,
                    352.0,
                    416.0,
                    320.0,
                    320.0,
                    256.0,
                    -256.0,
                    -96.0,
                    0.0,
                    0.009765625,
                    416.0,
                    448.0,
                    448.0,
                    448.0,
                    -448.0,
                    np.nan,
                ],
                dtype=np.float32,
            ),
            TensorProto.FLOAT8E4M3FNUZ: np.array(
                [
                    0.40625,
                    240.0,
                    240.0,
                    240.0,
                    240.0,
                    240.0,
                    -240.0,
                    -104.0,
                    0.0,
                    0.009765625,
                    240.0,
                    240.0,
                    240.0,
                    240.0,
                    -240.0,
                    np.nan,
                ],
                dtype=np.float32,
            ),
            TensorProto.FLOAT8E5M2: np.array(
                [
                    0.4375,
                    384.0,
                    384.0,
                    320.0,
                    320.0,
                    256.0,
                    -256.0,
                    -96.0,
                    0.0001068115234375,
                    0.009765625,
                    384.0,
                    448.0,
                    57344.0,
                    57344.0,
                    -57344.0,
                    np.nan,
                ],
                dtype=np.float32,
            ),
            TensorProto.FLOAT8E5M2FNUZ: np.array(
                [
                    0.4375,
                    384.0,
                    448.0,
                    320.0,
                    320.0,
                    256.0,
                    -256.0,
                    -96.0,
                    0.0001068115234375,
                    0.009765625,
                    448.0,
                    448.0,
                    57344.0,
                    57344.0,
                    -57344.0,
                    np.nan,
                ],
                dtype=np.float32,
            ),
        }
    )

    expected_no_saturate = (
        {}
        if not hasattr(TensorProto, "FLOAT8E4M3FN")
        else {
            TensorProto.FLOAT8E4M3FN: np.array(
                [
                    0.40625,
                    352.0,
                    416.0,
                    320.0,
                    320.0,
                    256.0,
                    -256.0,
                    -96.0,
                    0.0,
                    0.009765625,
                    416.0,
                    448.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                dtype=np.float32,
            ),
            TensorProto.FLOAT8E4M3FNUZ: np.array(
                [
                    0.40625,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    -104.0,
                    0.0,
                    0.009765625,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                dtype=np.float32,
            ),
            TensorProto.FLOAT8E5M2: np.array(
                [
                    0.4375,
                    384.0,
                    384.0,
                    320.0,
                    320.0,
                    256.0,
                    -256.0,
                    -96.0,
                    0.0001068115234375,
                    0.009765625,
                    384.0,
                    448.0,
                    np.inf,
                    np.inf,
                    -np.inf,
                    np.nan,
                ],
                dtype=np.float32,
            ),
            TensorProto.FLOAT8E5M2FNUZ: np.array(
                [
                    0.4375,
                    384.0,
                    448.0,
                    320.0,
                    320.0,
                    256.0,
                    -256.0,
                    -96.0,
                    0.0001068115234375,
                    0.009765625,
                    448.0,
                    448.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                dtype=np.float32,
            ),
        }
    )

    def model_cast_cast(self, to, saturate):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = make_node("Cast", ["X"], ["T"], to=to, saturate=saturate)
        node2 = make_node("Cast", ["T"], ["Y"], to=TensorProto.FLOAT)
        graph = make_graph([node1, node2], "lr", [X], [Y])
        onnx_model = make_model(graph)
        check_model(onnx_model)
        return onnx_model

    @unittest.skipIf(not hasattr(TensorProto, "FLOAT8E4M3FN"), reason="needs onnx>=1.14.0")
    def test_model_cast_cast_reference_saturate(self):
        expected = TestInferenceSession.expected_saturate
        x = TestInferenceSession.x

        for to, expect in expected.items():
            with self.subTest(to=to):
                onnx_model = self.model_cast_cast(to, 1)
                ref = ReferenceEvaluator(onnx_model)
                y = ref.run(None, {"X": x})[0]
                assert_allclose(expect, y)
                self.assertEqual(expect.shape, y.shape)
                self.assertEqual(expect.dtype, y.dtype)

    @unittest.skipIf(not hasattr(TensorProto, "FLOAT8E4M3FN"), reason="needs onnx>=1.14.0")
    def test_model_cast_cast_reference_no_saturate(self):
        expected = TestInferenceSession.expected_no_saturate
        x = TestInferenceSession.x

        for to, expect in expected.items():
            with self.subTest(to=to):
                onnx_model = self.model_cast_cast(to, 0)
                ref = ReferenceEvaluator(onnx_model)
                y = ref.run(None, {"X": x})[0]
                assert_allclose(expect, y)
                self.assertEqual(expect.shape, y.shape)
                self.assertEqual(expect.dtype, y.dtype)

    @unittest.skipIf(not hasattr(TensorProto, "FLOAT8E4M3FN"), reason="needs onnx>=1.14.0")
    def test_model_cast_cast_cpu_saturate(self):
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.add_session_config_entry("session.allow_released_opsets_only", "0")

        expected = TestInferenceSession.expected_saturate
        x = TestInferenceSession.x

        for to, expect in expected.items():
            with self.subTest(to=to):
                onnx_model = self.model_cast_cast(to, 1)
                sess = onnxruntime.InferenceSession(
                    onnx_model.SerializeToString(), so, providers=["CPUExecutionProvider"], read_config_from_model=1
                )
                y = sess.run(None, {"X": x})[0]
                assert_allclose(expect, y)
                self.assertEqual(expect.shape, y.shape)
                self.assertEqual(expect.dtype, y.dtype)

    @unittest.skipIf(not hasattr(TensorProto, "FLOAT8E4M3FN"), reason="needs onnx>=1.14.0")
    def test_model_cast_cast_cpu_no_saturate(self):
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.add_session_config_entry("session.allow_released_opsets_only", "0")

        expected = TestInferenceSession.expected_no_saturate
        x = TestInferenceSession.x

        for to, expect in expected.items():
            with self.subTest(to=to):
                onnx_model = self.model_cast_cast(to, 0)
                sess = onnxruntime.InferenceSession(
                    onnx_model.SerializeToString(), so, providers=["CPUExecutionProvider"], read_config_from_model=1
                )
                y = sess.run(None, {"X": x})[0]
                try:
                    assert_allclose(expect, y)
                except AssertionError as e:
                    d = np.abs(y - expect)
                    assert_allclose(np.isnan(expect), np.isnan(y))
                    d = [~np.isnan(d)].max()
                    self.assertEqual(d, 0)
                self.assertEqual(expect.shape, y.shape)
                self.assertEqual(expect.dtype, y.dtype)

    @unittest.skipIf(not hasattr(TensorProto, "FLOAT8E4M3FN"), reason="needs onnx>=1.14.0")
    @unittest.skipIf("CUDAExecutionProvider" not in available_providers, reason="Not running on CUDA.")
    def test_model_cast_cast_cuda_saturate(self):
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.add_session_config_entry("session.allow_released_opsets_only", "0")

        expected = TestInferenceSession.expected_saturate
        x = TestInferenceSession.x

        for to, expect in expected.items():
            with self.subTest(to=to):
                if to not in {TensorProto.FLOAT8E4M3FN, TensorProto.FLOAT8E5M2}:
                    # only those types are available on CUDA.
                    continue
                onnx_model = self.model_cast_cast(to, 1)
                sess = onnxruntime.InferenceSession(
                    onnx_model.SerializeToString(), so, providers=["CUDAExecutionProvider"], read_config_from_model=1
                )
                y = sess.run(None, {"X": x})[0]
                assert_allclose(expect, y)
                self.assertEqual(expect.shape, y.shape)
                self.assertEqual(expect.dtype, y.dtype)

    @unittest.skipIf(not hasattr(TensorProto, "FLOAT8E4M3FN"), reason="needs onnx>=1.14.0")
    @unittest.skipIf("CUDAExecutionProvider" not in available_providers, reason="Not running on CUDA.")
    def test_model_cast_cast_cuda_no_saturate(self):
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.add_session_config_entry("session.allow_released_opsets_only", "0")

        expected = TestInferenceSession.expected_no_saturate
        x = TestInferenceSession.x

        for to, expect in expected.items():
            with self.subTest(to=to):
                if to not in {TensorProto.FLOAT8E4M3FN, TensorProto.FLOAT8E5M2}:
                    # only those types are available on CUDA.
                    continue
                onnx_model = self.model_cast_cast(to, 0)
                sess = onnxruntime.InferenceSession(
                    onnx_model.SerializeToString(), so, providers=["CUDAExecutionProvider"], read_config_from_model=1
                )
                y = sess.run(None, {"X": x})[0]
                assert_allclose(expect, y)
                self.assertEqual(expect.shape, y.shape)
                self.assertEqual(expect.dtype, y.dtype)

    @unittest.skipIf(not hasattr(TensorProto, "FLOAT8E4M3FN"), reason="needs onnx>=1.14.0")
    @unittest.skipIf("CUDAExecutionProvider" not in available_providers, reason="Not running on CUDA.")
    def test_model_cast_cast_cuda_ortvalue_saturate(self):
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.add_session_config_entry("session.allow_released_opsets_only", "0")

        expected = TestInferenceSession.expected_saturate
        x = TestInferenceSession.x

        for to, expect in expected.items():
            with self.subTest(to=to):
                if to not in {TensorProto.FLOAT8E4M3FN, TensorProto.FLOAT8E5M2}:
                    # only those types are available on CUDA.
                    continue
                onnx_model = self.model_cast_cast(to, 1)
                ortv = onnxruntime.OrtValue.ortvalue_from_numpy(x)  # , device_type="cuda")
                sess = onnxruntime.InferenceSession(
                    onnx_model.SerializeToString(), so, providers=["CUDAExecutionProvider"], read_config_from_model=1
                )
                y = sess.run_with_ort_values(["Y"], {"X": ortv})[0].numpy()
                assert_allclose(expect, y)
                self.assertEqual(expect.shape, y.shape)
                self.assertEqual(expect.dtype, y.dtype)


if __name__ == "__main__":
    unittest.main(verbosity=2)
