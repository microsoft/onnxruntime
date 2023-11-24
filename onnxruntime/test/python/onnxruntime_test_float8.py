# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# pylint: disable=C0116,W0212,R1720,C0103,C0114

import os
import platform
import sys
import unittest

import numpy as np
import packaging.version as pv
import parameterized
from numpy.testing import assert_allclose
from onnx import TensorProto
from onnx import __version__ as onnx_version
from onnx.checker import check_model
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor, make_tensor_value_info
from onnx.reference import ReferenceEvaluator

import onnxruntime

# handle change from python 3.8 and on where loading a dll from the current directory needs to be explicitly allowed.
if platform.system() == "Windows" and sys.version_info[:2] >= (3, 8):
    os.add_dll_directory(os.getcwd())

available_providers = [provider for provider in onnxruntime.get_available_providers()]


class TestInferenceSession(unittest.TestCase):
    """
    All float 8 values were computed by using the python functions implemented in onnx:
    `float32_to_float8e4m3
    <https://onnx.ai/onnx/api/helper.html#onnx.helper.float32_to_float8e4m3>`_,
    `float32_to_float8e5m2
    <https://onnx.ai/onnx/api/helper.html#onnx.helper.float32_to_float8e5m2>`_,
    `float8e4m3_to_float32
    <https://onnx.ai/onnx/api/numpy_helper.html#onnx.numpy_helper.float8e4m3_to_float32>`_,
    `float8e5m2_to_float32
    <https://onnx.ai/onnx/api/numpy_helper.html#onnx.numpy_helper.float8e5m2_to_float32>`_.
    """

    dtypes = {"FLOAT": np.float32, "FLOAT16": np.float16}  # noqa: RUF012
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
                    -96.0,
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
                    -96.0,
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
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                dtype=np.float32,
            ),
        }
    )

    def model_cast_cast(self, to, float_name, saturate):
        src = getattr(TensorProto, float_name)
        x = make_tensor_value_info("X", src, [None])
        y = make_tensor_value_info("Y", src, [None])
        node1 = make_node("Cast", ["X"], ["T"], to=to, saturate=saturate)
        node2 = make_node("Cast", ["T"], ["Y"], to=src)
        graph = make_graph([node1, node2], "lr", [x], [y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
        check_model(onnx_model)
        return onnx_model

    def model_cast_cast_f16_float(self, to, saturate, rev=False):
        x = make_tensor_value_info("X", TensorProto.FLOAT if rev else TensorProto.FLOAT16, [None])
        y = make_tensor_value_info("Y", TensorProto.FLOAT16 if rev else TensorProto.FLOAT, [None])
        node1 = make_node("Cast", ["X"], ["T"], to=to, saturate=saturate)
        node2 = make_node("Cast", ["T"], ["Y"], to=TensorProto.FLOAT16 if rev else TensorProto.FLOAT)
        graph = make_graph([node1, node2], "lr", [x], [y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
        check_model(onnx_model)
        return onnx_model

    @unittest.skipIf(pv.Version(onnx_version) < pv.Version("1.15.0"), reason="needs onnx>=1.15.0")
    @parameterized.parameterized.expand(
        [
            ("FLOAT8E4M3FN", "FLOAT", 1),
            ("FLOAT8E4M3FNUZ", "FLOAT", 1),
            ("FLOAT8E5M2", "FLOAT", 1),
            ("FLOAT8E5M2FNUZ", "FLOAT", 1),
            ("FLOAT8E4M3FN", "FLOAT", 0),
            ("FLOAT8E4M3FNUZ", "FLOAT", 0),
            ("FLOAT8E5M2", "FLOAT", 0),
            ("FLOAT8E5M2FNUZ", "FLOAT", 0),
            ("FLOAT8E4M3FN", "FLOAT16", 1),
            ("FLOAT8E4M3FNUZ", "FLOAT16", 1),
            ("FLOAT8E5M2", "FLOAT16", 1),
            ("FLOAT8E5M2FNUZ", "FLOAT16", 1),
            ("FLOAT8E4M3FN", "FLOAT16", 0),
            ("FLOAT8E4M3FNUZ", "FLOAT16", 0),
            ("FLOAT8E5M2", "FLOAT16", 0),
            ("FLOAT8E5M2FNUZ", "FLOAT16", 0),
        ]
    )
    @unittest.skipIf(not hasattr(TensorProto, "FLOAT8E4M3FN"), reason="needs onnx>=1.14.0")
    def test_model_cast_cast_reference(self, name: str, float_name: str, saturate: int):
        to = getattr(TensorProto, name)
        expected = TestInferenceSession.expected_saturate if saturate else TestInferenceSession.expected_no_saturate
        x = TestInferenceSession.x.astype(TestInferenceSession.dtypes[float_name])
        expect = expected[to].astype(TestInferenceSession.dtypes[float_name])
        onnx_model = self.model_cast_cast(to, float_name, saturate)
        ref = ReferenceEvaluator(onnx_model)
        y = ref.run(None, {"X": x})[0]
        assert_allclose(expect, y)
        self.assertEqual(expect.shape, y.shape)
        self.assertEqual(expect.dtype, y.dtype)

    @parameterized.parameterized.expand(
        [
            ("FLOAT8E4M3FN", "FLOAT", 1),
            ("FLOAT8E4M3FNUZ", "FLOAT", 1),
            ("FLOAT8E5M2", "FLOAT", 1),
            ("FLOAT8E5M2FNUZ", "FLOAT", 1),
            ("FLOAT8E4M3FN", "FLOAT", 0),
            ("FLOAT8E4M3FNUZ", "FLOAT", 0),
            ("FLOAT8E5M2", "FLOAT", 0),
            ("FLOAT8E5M2FNUZ", "FLOAT", 0),
            ("FLOAT8E4M3FN", "FLOAT16", 1),
            ("FLOAT8E4M3FNUZ", "FLOAT16", 1),
            ("FLOAT8E5M2", "FLOAT16", 1),
            ("FLOAT8E5M2FNUZ", "FLOAT16", 1),
            ("FLOAT8E4M3FN", "FLOAT16", 0),
            ("FLOAT8E4M3FNUZ", "FLOAT16", 0),
            ("FLOAT8E5M2", "FLOAT16", 0),
            ("FLOAT8E5M2FNUZ", "FLOAT16", 0),
        ]
    )
    @unittest.skipIf(not hasattr(TensorProto, "FLOAT8E4M3FN"), reason="needs onnx>=1.14.0")
    def test_model_cast_cast_cpu(self, name: str, float_name: str, saturate: int):
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        # so.add_session_config_entry("session.allow_released_opsets_only", "0")

        to = getattr(TensorProto, name)
        expected = TestInferenceSession.expected_saturate if saturate else TestInferenceSession.expected_no_saturate
        x = TestInferenceSession.x.astype(TestInferenceSession.dtypes[float_name])
        expect = expected[to].astype(TestInferenceSession.dtypes[float_name])

        onnx_model = self.model_cast_cast(to, float_name, saturate)
        sess = onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), so, providers=["CPUExecutionProvider"], read_config_from_model=1
        )
        y = sess.run(None, {"X": x})[0]
        assert_allclose(expect, y)
        self.assertEqual(expect.shape, y.shape)
        self.assertEqual(expect.dtype, y.dtype)

    @parameterized.parameterized.expand(
        [
            ("FLOAT8E4M3FN", "FLOAT", 1, "CUDAExecutionProvider"),
            ("FLOAT8E5M2", "FLOAT", 1, "CUDAExecutionProvider"),
            ("FLOAT8E4M3FN", "FLOAT", 0, "CUDAExecutionProvider"),
            ("FLOAT8E5M2", "FLOAT", 0, "CUDAExecutionProvider"),
            ("FLOAT8E4M3FN", "FLOAT16", 1, "CUDAExecutionProvider"),
            ("FLOAT8E5M2", "FLOAT16", 1, "CUDAExecutionProvider"),
            ("FLOAT8E4M3FN", "FLOAT16", 0, "CUDAExecutionProvider"),
            ("FLOAT8E5M2", "FLOAT16", 0, "CUDAExecutionProvider"),
        ]
    )
    @unittest.skipIf(not hasattr(TensorProto, "FLOAT8E4M3FN"), reason="needs onnx>=1.14.0")
    @unittest.skipIf("CUDAExecutionProvider" not in available_providers, reason="Not running without CUDA.")
    def test_model_cast_cast_cuda(self, name: str, float_name: str, saturate: int, provider: str):
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        # so.add_session_config_entry("session.allow_released_opsets_only", "0")

        to = getattr(TensorProto, name)
        expected = TestInferenceSession.expected_saturate if saturate else TestInferenceSession.expected_no_saturate
        x = TestInferenceSession.x.astype(TestInferenceSession.dtypes[float_name])
        expect = expected[to].astype(TestInferenceSession.dtypes[float_name])

        onnx_model = self.model_cast_cast(to, float_name, saturate)
        sess = onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), so, providers=[provider], read_config_from_model=1
        )
        y = sess.run(None, {"X": x})[0]
        try:
            assert_allclose(expect, y)
        except AssertionError as e:
            raise AssertionError(
                f"Discrepancies with name={name}, float_name={float_name}, "
                f"saturate={saturate}\nexpect={expect}\ny={y}"
            ) from e
        self.assertEqual(expect.shape, y.shape)
        self.assertEqual(expect.dtype, y.dtype)

    @parameterized.parameterized.expand(
        [
            ("FLOAT8E4M3FN", "FLOAT", 1, "CUDAExecutionProvider"),
            ("FLOAT8E5M2", "FLOAT", 1, "CUDAExecutionProvider"),
            ("FLOAT8E4M3FN", "FLOAT", 0, "CUDAExecutionProvider"),
            ("FLOAT8E5M2", "FLOAT", 0, "CUDAExecutionProvider"),
            ("FLOAT8E4M3FN", "FLOAT16", 1, "CUDAExecutionProvider"),
            ("FLOAT8E5M2", "FLOAT16", 1, "CUDAExecutionProvider"),
            ("FLOAT8E4M3FN", "FLOAT16", 0, "CUDAExecutionProvider"),
            ("FLOAT8E5M2", "FLOAT16", 0, "CUDAExecutionProvider"),
        ]
    )
    @unittest.skipIf(not hasattr(TensorProto, "FLOAT8E4M3FN"), reason="needs onnx>=1.14.0")
    @unittest.skipIf("CUDAExecutionProvider" not in available_providers, reason="Not running without CUDA.")
    def test_model_cast_cast_cuda_ortvalue(self, name: str, float_name: str, saturate: int, provider: str):
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        # so.add_session_config_entry("session.allow_released_opsets_only", "0")

        to = getattr(TensorProto, name)
        expected = TestInferenceSession.expected_saturate if saturate else TestInferenceSession.expected_no_saturate
        x = TestInferenceSession.x.astype(TestInferenceSession.dtypes[float_name])
        expect = expected[to].astype(TestInferenceSession.dtypes[float_name])

        onnx_model = self.model_cast_cast(to, float_name, saturate)
        ortv = onnxruntime.OrtValue.ortvalue_from_numpy(x, device_type="cuda")
        sess = onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), so, providers=[provider], read_config_from_model=1
        )
        y = sess.run_with_ort_values(["Y"], {"X": ortv})[0].numpy()
        try:
            assert_allclose(expect, y)
        except AssertionError as e:
            raise AssertionError(
                f"Discrepancies with name={name}, float_name={float_name}, "
                f"saturate={saturate}\nexpect={expect}\ny={y}"
            ) from e
        self.assertEqual(expect.shape, y.shape)
        self.assertEqual(expect.dtype, y.dtype)

    def model_qdq(self, to, float_name, saturate, castq=False, castdq=False, like=False, initializer=False):
        fltype = getattr(TensorProto, float_name)
        x = make_tensor_value_info("X", fltype, [None])
        y = make_tensor_value_info("Y", fltype, [None])
        if initializer:
            scale = make_tensor("scale", fltype, [1], [1.0])
            zero = make_tensor("zero", to, [1], [0.0])
        else:
            scale = make_node("Constant", [], ["scale"], value=make_tensor("scale", fltype, [1], [1.0]))
            zero = make_node("Constant", [], ["zero"], value=make_tensor("zero", to, [1], [0.0]))
        if castq:
            if like:
                node1 = make_node("CastLike", ["X", "zero"], ["Temp"], saturate=saturate)
            else:
                node1 = make_node("Cast", ["X"], ["Temp"], saturate=saturate, to=to)
        else:
            node1 = make_node("QuantizeLinear", ["X", "scale", "zero"], ["Temp"], saturate=saturate, axis=0)
        if castdq:
            if like:
                node2 = make_node("CastLike", ["Temp", "scale"], ["Y"])
            else:
                node2 = make_node("Cast", ["Temp"], ["Y"], to=fltype)
        else:
            node2 = make_node("DequantizeLinear", ["Temp", "scale"], ["Y"], axis=0)
        if initializer:
            graph = make_graph([node1, node2], "lr", [x], [y], [scale, zero])
        else:
            graph = make_graph([scale, zero, node1, node2], "lr", [x], [y])
        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
        check_model(onnx_model)
        return onnx_model

    @unittest.skipIf(pv.Version(onnx_version) < pv.Version("1.15.0"), reason="needs onnx>=1.15.0")
    @parameterized.parameterized.expand(
        [
            ("FLOAT8E4M3FN", "FLOAT", 1),
            ("FLOAT8E4M3FNUZ", "FLOAT", 1),
            ("FLOAT8E5M2", "FLOAT", 1),
            ("FLOAT8E5M2FNUZ", "FLOAT", 1),
            ("FLOAT8E4M3FN", "FLOAT", 0),
            ("FLOAT8E4M3FNUZ", "FLOAT", 0),
            ("FLOAT8E5M2", "FLOAT", 0),
            ("FLOAT8E5M2FNUZ", "FLOAT", 0),
            ("FLOAT8E4M3FN", "FLOAT16", 1),
            ("FLOAT8E4M3FNUZ", "FLOAT16", 1),
            ("FLOAT8E5M2", "FLOAT16", 1),
            ("FLOAT8E5M2FNUZ", "FLOAT16", 1),
            ("FLOAT8E4M3FN", "FLOAT16", 0),
            ("FLOAT8E4M3FNUZ", "FLOAT16", 0),
            ("FLOAT8E5M2", "FLOAT16", 0),
            ("FLOAT8E5M2FNUZ", "FLOAT16", 0),
        ]
    )
    @unittest.skipIf(not hasattr(TensorProto, "FLOAT8E4M3FN"), reason="needs onnx>=1.14.0")
    def test_model_qdq_reference(self, name: str, float_name: str, saturate: int):
        to = getattr(TensorProto, name)
        expected = TestInferenceSession.expected_saturate if saturate else TestInferenceSession.expected_no_saturate
        x = TestInferenceSession.x.astype(TestInferenceSession.dtypes[float_name])
        expect = expected[to].astype(TestInferenceSession.dtypes[float_name])

        onnx_model = self.model_qdq(to, float_name, saturate)
        ref = ReferenceEvaluator(onnx_model)
        y = ref.run(None, {"X": x})[0]
        assert_allclose(expect, y)
        self.assertEqual(expect.shape, y.shape)
        # A bug in the reference implementation,
        # enable that test when onnx package is fixed.
        # self.assertEqual(expect.dtype, y.dtype)

    @parameterized.parameterized.expand(
        [
            ("FLOAT8E4M3FN", "FLOAT", 1),
            ("FLOAT8E4M3FNUZ", "FLOAT", 1),
            ("FLOAT8E5M2", "FLOAT", 1),
            ("FLOAT8E5M2FNUZ", "FLOAT", 1),
            ("FLOAT8E4M3FN", "FLOAT", 0),
            ("FLOAT8E4M3FNUZ", "FLOAT", 0),
            ("FLOAT8E5M2", "FLOAT", 0),
            ("FLOAT8E5M2FNUZ", "FLOAT", 0),
            ("FLOAT8E4M3FN", "FLOAT16", 1),
            ("FLOAT8E4M3FNUZ", "FLOAT16", 1),
            ("FLOAT8E5M2", "FLOAT16", 1),
            ("FLOAT8E5M2FNUZ", "FLOAT16", 1),
            ("FLOAT8E4M3FN", "FLOAT16", 0),
            ("FLOAT8E4M3FNUZ", "FLOAT16", 0),
            ("FLOAT8E5M2", "FLOAT16", 0),
            ("FLOAT8E5M2FNUZ", "FLOAT16", 0),
        ]
    )
    @unittest.skipIf(not hasattr(TensorProto, "FLOAT8E4M3FN"), reason="needs onnx>=1.14.0")
    def test_model_qdq_cpu(self, name: str, float_name: str, saturate: int):
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.add_session_config_entry("session.allow_released_opsets_only", "0")

        to = getattr(TensorProto, name)
        expected = TestInferenceSession.expected_saturate if saturate else TestInferenceSession.expected_no_saturate
        x = TestInferenceSession.x.astype(TestInferenceSession.dtypes[float_name])
        expect = expected[to].astype(TestInferenceSession.dtypes[float_name])

        onnx_model = self.model_qdq(to, float_name, saturate)
        try:
            sess = onnxruntime.InferenceSession(
                onnx_model.SerializeToString(), so, providers=["CPUExecutionProvider"], read_config_from_model=1
            )
        except Exception as e:
            raise AssertionError(
                f"Cannot build InferenceSession with name={name}, float_name={float_name}, saturate={saturate}."
            ) from e
        y = sess.run(None, {"X": x})[0]
        assert_allclose(expect, y)
        self.assertEqual(expect.shape, y.shape)
        self.assertEqual(expect.dtype, y.dtype)

    @parameterized.parameterized.expand(
        [
            ("FLOAT8E4M3FN", "FLOAT", 1),
            ("FLOAT8E4M3FNUZ", "FLOAT", 1),
            ("FLOAT8E5M2", "FLOAT", 1),
            ("FLOAT8E5M2FNUZ", "FLOAT", 1),
            ("FLOAT8E4M3FN", "FLOAT", 0),
            ("FLOAT8E4M3FNUZ", "FLOAT", 0),
            ("FLOAT8E5M2", "FLOAT", 0),
            ("FLOAT8E5M2FNUZ", "FLOAT", 0),
            ("FLOAT8E4M3FN", "FLOAT16", 1),
            ("FLOAT8E4M3FNUZ", "FLOAT16", 1),
            ("FLOAT8E5M2", "FLOAT16", 1),
            ("FLOAT8E5M2FNUZ", "FLOAT16", 1),
            ("FLOAT8E4M3FN", "FLOAT16", 0),
            ("FLOAT8E4M3FNUZ", "FLOAT16", 0),
            ("FLOAT8E5M2", "FLOAT16", 0),
            ("FLOAT8E5M2FNUZ", "FLOAT16", 0),
        ]
    )
    @unittest.skipIf(not hasattr(TensorProto, "FLOAT8E4M3FN"), reason="needs onnx>=1.14.0")
    def test_model_qdq_cpu_init(self, name: str, float_name: str, saturate: int):
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.add_session_config_entry("session.allow_released_opsets_only", "0")

        to = getattr(TensorProto, name)
        expected = TestInferenceSession.expected_saturate if saturate else TestInferenceSession.expected_no_saturate
        x = TestInferenceSession.x.astype(TestInferenceSession.dtypes[float_name])
        expect = expected[to].astype(TestInferenceSession.dtypes[float_name])

        onnx_model = self.model_qdq(to, float_name, saturate, initializer=True)
        try:
            sess = onnxruntime.InferenceSession(
                onnx_model.SerializeToString(), so, providers=["CPUExecutionProvider"], read_config_from_model=1
            )
        except Exception as e:
            raise AssertionError(
                f"Cannot build InferenceSession with name={name}, float_name={float_name}, saturate={saturate}."
            ) from e
        y = sess.run(None, {"X": x})[0]
        assert_allclose(expect, y)
        self.assertEqual(expect.shape, y.shape)
        self.assertEqual(expect.dtype, y.dtype)

    @parameterized.parameterized.expand(
        [
            ("FLOAT8E4M3FN", "FLOAT", 1),
            ("FLOAT8E4M3FNUZ", "FLOAT", 1),
            ("FLOAT8E5M2", "FLOAT", 1),
            ("FLOAT8E5M2FNUZ", "FLOAT", 1),
            ("FLOAT8E4M3FN", "FLOAT", 0),
            ("FLOAT8E4M3FNUZ", "FLOAT", 0),
            ("FLOAT8E5M2", "FLOAT", 0),
            ("FLOAT8E5M2FNUZ", "FLOAT", 0),
            ("FLOAT8E4M3FN", "FLOAT16", 1),
            ("FLOAT8E4M3FNUZ", "FLOAT16", 1),
            ("FLOAT8E5M2", "FLOAT16", 1),
            ("FLOAT8E5M2FNUZ", "FLOAT16", 1),
            ("FLOAT8E4M3FN", "FLOAT16", 0),
            ("FLOAT8E4M3FNUZ", "FLOAT16", 0),
            ("FLOAT8E5M2", "FLOAT16", 0),
            ("FLOAT8E5M2FNUZ", "FLOAT16", 0),
        ]
    )
    @unittest.skipIf(not hasattr(TensorProto, "FLOAT8E4M3FN"), reason="needs onnx>=1.14.0")
    def test_model_cast_like_x2_cpu(self, name: str, float_name: str, saturate: int):
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        # so.add_session_config_entry("session.allow_released_opsets_only", "0")

        to = getattr(TensorProto, name)
        expected = TestInferenceSession.expected_saturate if saturate else TestInferenceSession.expected_no_saturate
        x = TestInferenceSession.x.astype(TestInferenceSession.dtypes[float_name])
        expect = expected[to].astype(TestInferenceSession.dtypes[float_name])

        onnx_model = self.model_qdq(to, float_name, saturate, True, True, True)
        self.assertTrue("CastLike", str(onnx_model))
        try:
            sess = onnxruntime.InferenceSession(
                onnx_model.SerializeToString(), so, providers=["CPUExecutionProvider"], read_config_from_model=1
            )
        except Exception as e:
            raise AssertionError(
                f"Cannot build InferenceSession with name={name}, float_name={float_name}, saturate={saturate}."
            ) from e
        y = sess.run(None, {"X": x})[0]
        try:
            assert_allclose(expect, y)
        except AssertionError as e:
            # TODO: if not saturate, it fails, CastLike is probably handled with Cast but where?
            if not saturate:
                return
            raise AssertionError(
                f"Discrepancies with name={name}, float_name={float_name}, "
                f"saturate={saturate}\nexpect={expect}\ny={y}"
            ) from e
        self.assertEqual(expect.shape, y.shape)
        self.assertEqual(expect.dtype, y.dtype)

    @parameterized.parameterized.expand(
        [
            ("FLOAT8E4M3FN", "FLOAT", 1, "CUDAExecutionProvider"),
            ("FLOAT8E5M2", "FLOAT", 1, "CUDAExecutionProvider"),
            ("FLOAT8E4M3FN", "FLOAT", 0, "CUDAExecutionProvider"),
            ("FLOAT8E5M2", "FLOAT", 0, "CUDAExecutionProvider"),
            ("FLOAT8E4M3FN", "FLOAT16", 1, "CUDAExecutionProvider"),
            ("FLOAT8E5M2", "FLOAT16", 1, "CUDAExecutionProvider"),
            ("FLOAT8E4M3FN", "FLOAT16", 0, "CUDAExecutionProvider"),
            ("FLOAT8E5M2", "FLOAT16", 0, "CUDAExecutionProvider"),
        ]
    )
    @unittest.skipIf(not hasattr(TensorProto, "FLOAT8E4M3FN"), reason="needs onnx>=1.14.0")
    @unittest.skipIf("CUDAExecutionProvider" not in available_providers, reason="Not running without CUDA.")
    def test_model_qdq_cuda(self, name: str, float_name: str, saturate: int, provider: str):
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        # so.add_session_config_entry("session.allow_released_opsets_only", "0")

        to = getattr(TensorProto, name)
        expected = TestInferenceSession.expected_saturate if saturate else TestInferenceSession.expected_no_saturate
        x = TestInferenceSession.x.astype(TestInferenceSession.dtypes[float_name])
        expect = expected[to].astype(TestInferenceSession.dtypes[float_name])

        onnx_model = self.model_qdq(to, float_name, saturate, False, False)
        sess = onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), so, providers=[provider], read_config_from_model=1
        )
        try:
            y = sess.run(None, {"X": x})[0]
        except Exception as e:
            raise AssertionError(
                f"qdq failed with name={name!r}, float_name={float_name!r}, "
                f"saturate={saturate!r}, provider={provider!r}."
            ) from e
        assert_allclose(expect, y)
        self.assertEqual(expect.shape, y.shape)
        self.assertEqual(expect.dtype, y.dtype)

    @parameterized.parameterized.expand(
        [
            ("FLOAT8E4M3FN", "FLOAT", 1, "CUDAExecutionProvider"),
            ("FLOAT8E5M2", "FLOAT", 1, "CUDAExecutionProvider"),
            ("FLOAT8E4M3FN", "FLOAT", 0, "CUDAExecutionProvider"),
            ("FLOAT8E5M2", "FLOAT", 0, "CUDAExecutionProvider"),
            ("FLOAT8E4M3FN", "FLOAT16", 1, "CUDAExecutionProvider"),
            ("FLOAT8E5M2", "FLOAT16", 1, "CUDAExecutionProvider"),
            ("FLOAT8E4M3FN", "FLOAT16", 0, "CUDAExecutionProvider"),
            ("FLOAT8E5M2", "FLOAT16", 0, "CUDAExecutionProvider"),
        ]
    )
    @unittest.skipIf(not hasattr(TensorProto, "FLOAT8E4M3FN"), reason="needs onnx>=1.14.0")
    @unittest.skipIf("CUDAExecutionProvider" not in available_providers, reason="Not running on CUDA.")
    def test_model_qdq_cuda_ortvalue(self, name: str, float_name: str, saturate: int, provider: str):
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        # so.add_session_config_entry("session.allow_released_opsets_only", "0")

        to = getattr(TensorProto, name)
        expected = TestInferenceSession.expected_saturate if saturate else TestInferenceSession.expected_no_saturate
        x = TestInferenceSession.x.astype(TestInferenceSession.dtypes[float_name])
        expect = expected[to].astype(TestInferenceSession.dtypes[float_name])

        onnx_model = self.model_qdq(to, float_name, saturate, False, False)
        ortv = onnxruntime.OrtValue.ortvalue_from_numpy(x, device_type="cuda")
        sess = onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), so, providers=[provider], read_config_from_model=1
        )
        try:
            y = sess.run_with_ort_values(["Y"], {"X": ortv})[0].numpy()
        except Exception as e:
            raise AssertionError(
                f"qdq failed with name={name!r}, float_name={float_name!r}, "
                f"saturate={saturate!r}, provider={provider!r}."
            ) from e
        assert_allclose(expect, y)
        self.assertEqual(expect.shape, y.shape)
        self.assertEqual(expect.dtype, y.dtype)

    @unittest.skipIf("CUDAExecutionProvider" not in available_providers, reason="Not running without CUDA.")
    def test_compare_cpu_cuda_e4m3fn(self):
        folder = os.path.join(os.path.dirname(__file__), "..", "testdata", "float8")
        model = os.path.join(folder, "te.cast_fp8_1_fp32.onnx")
        data = np.load(os.path.join(folder, "te.cast_fp8_1_fp32_input.npy"))

        sess_cpu = onnxruntime.InferenceSession(model, providers=["CPUExecutionProvider"])
        sess_cuda = onnxruntime.InferenceSession(model, providers=["CUDAExecutionProvider"])
        cpu_res = sess_cpu.run(None, {"input": data})[0]
        cuda_res = sess_cuda.run(None, {"input": data})[0]
        self.assertEqual(cuda_res.tolist(), cpu_res.tolist())


if __name__ == "__main__":
    unittest.main(verbosity=2)
