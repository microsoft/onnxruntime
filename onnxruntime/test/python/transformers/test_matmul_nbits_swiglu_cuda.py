# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Exact packed INT2 projection/SwiGLU fusion parity, including GEMM prefill."""

import tempfile
import unittest

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

import onnxruntime as ort


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
    "INT2 CUDA tests require SM80 or newer",
)
class TestMatMulNBitsSwiGLUCuda(unittest.TestCase):
    def _model(self, k=3072, n=2048, fused=True):
        rng = np.random.default_rng(309)
        weight = rng.integers(0, 256, (n, k // 128, 32), dtype=np.uint8)
        scales = rng.uniform(0.001, 0.025, (n, k // 128)).astype(np.float16)
        initializers = [
            numpy_helper.from_array(weight, "weight"),
            numpy_helper.from_array(scales, "scales"),
        ]
        if fused:
            nodes = [
                helper.make_node(
                    "MatMulNBitsSwiGLU", ["x", "weight", "scales"], ["y"], domain="com.microsoft", K=k, N=n
                )
            ]
        else:
            initializers.append(numpy_helper.from_array(np.array([n // 2, n // 2], dtype=np.int64), "sizes"))
            nodes = [
                helper.make_node(
                    "MatMulNBits",
                    ["x", "weight", "scales"],
                    ["projected"],
                    domain="com.microsoft",
                    K=k,
                    N=n,
                    bits=2,
                    block_size=128,
                ),
                helper.make_node("Split", ["projected", "sizes"], ["gate", "up"], axis=-1),
                helper.make_node("QuickGelu", ["gate"], ["swish"], domain="com.microsoft", alpha=1.0),
                helper.make_node("Mul", ["swish", "up"], ["y"]),
            ]
        return helper.make_model(
            helper.make_graph(
                nodes,
                "int2_swiglu",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT16, ["rows", k])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT16, ["rows", n // 2])],
                initializers,
            ),
            opset_imports=[helper.make_opsetid("", 24), helper.make_opsetid("com.microsoft", 1)],
        )

    def _session(self, model, cuda_graph=False, optimize=True):
        options = ort.SessionOptions()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        if not optimize:
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        return ort.InferenceSession(
            model.SerializeToString(),
            options,
            providers=[("CUDAExecutionProvider", {"enable_cuda_graph": str(int(cuda_graph))})],
        )

    def test_exact_projection_activation_parity(self):
        rng = np.random.default_rng(143)
        for k, n in ((128, 16), (640, 48), (3072, 2048)):
            fused = self._session(self._model(k, n))
            reference = self._session(self._model(k, n, fused=False))
            for rows in (0, 1, 2, 3, 4, 5, 8, 9, 16, 128, 1024):
                with self.subTest(k=k, n=n, rows=rows):
                    x = rng.normal(0, 0.5, (rows, k)).astype(np.float16)
                    expected = reference.run(None, {"x": x})[0]
                    actual = fused.run(None, {"x": x})[0]
                    self.assertTrue(np.isfinite(actual).all())
                    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
                    self.assertEqual(actual.shape, (rows, n // 2))

    def test_decode_row_count_invariance_and_graph_replay(self):
        rng = np.random.default_rng(245)
        x = rng.normal(0, 0.5, (8, 3072)).astype(np.float16)
        model = self._model()
        session = self._session(model)
        reference = np.concatenate([session.run(None, {"x": row[None]})[0] for row in x])
        for rows in (1, 2, 3, 4, 5, 8):
            with self.subTest(rows=rows):
                captured = self._session(model, cuda_graph=True)
                binding = captured.io_binding()
                binding.bind_ortvalue_input("x", ort.OrtValue.ortvalue_from_numpy(x[:rows], "cuda", 0))
                output = ort.OrtValue.ortvalue_from_shape_and_type([rows, 1024], np.float16, "cuda", 0)
                binding.bind_ortvalue_output("y", output)
                for _ in range(2):
                    captured.run_with_iobinding(binding)
                    np.testing.assert_array_equal(output.numpy().view(np.uint16), reference[:rows].view(np.uint16))

    def test_activation_tails_and_zero_input(self):
        fused = self._session(self._model())
        reference = self._session(self._model(fused=False))
        rng = np.random.default_rng(193)
        for magnitude in (0.0, 0.0625, 1.0, 8.0):
            with self.subTest(magnitude=magnitude):
                x = (rng.normal(size=(8, 3072)) * magnitude).astype(np.float16)
                actual = fused.run(None, {"x": x})[0]
                expected = reference.run(None, {"x": x})[0]
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))

    def test_connected_silu_uses_optimized_quickgelu_rounding(self):
        model = self._model(k=128, n=32, fused=False)
        nodes = list(model.graph.node)
        model.graph.ClearField("node")
        model.graph.node.extend(
            [
                *nodes[:2],
                helper.make_node("Sigmoid", ["gate"], ["sigmoid"]),
                helper.make_node("Mul", ["gate", "sigmoid"], ["swish"]),
                nodes[-1],
            ]
        )
        options = ort.SessionOptions()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        with tempfile.TemporaryDirectory() as directory:
            path = directory + "/optimized.onnx"
            options.optimized_model_filepath = path
            reference = ort.InferenceSession(model.SerializeToString(), options, providers=["CUDAExecutionProvider"])
            optimized = onnx.load(path)
            self.assertTrue(
                any(node.domain == "com.microsoft" and node.op_type == "QuickGelu" for node in optimized.graph.node)
            )
        fused = self._session(self._model(k=128, n=32))
        rng = np.random.default_rng(315)
        for rows in (1, 2, 8, 1024):
            with self.subTest(rows=rows):
                x = rng.normal(0, 1, (rows, 128)).astype(np.float16)
                actual = fused.run(None, {"x": x})[0]
                expected = reference.run(None, {"x": x})[0]
                np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))

    def test_invalid_attributes_and_input_shapes(self):
        for attr, value in (("K", 127), ("K", 0), ("N", 17), ("N", 0), ("N", 2**31)):
            with self.subTest(attr=attr, value=value):
                model = self._model(k=128, n=16)
                for attribute in model.graph.node[0].attribute:
                    if attribute.name == attr:
                        attribute.i = value
                with self.assertRaisesRegex(Exception, "positive multiple|int32"):
                    self._session(model, optimize=False)
        for index in (0, 1, 2):
            with self.subTest(input=index):
                model = self._model(k=128, n=16)
                if index == 0:
                    model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 256
                else:
                    tensor = model.graph.initializer[index - 1]
                    array = numpy_helper.to_array(tensor).copy()[:-1]
                    tensor.CopyFrom(numpy_helper.from_array(array, tensor.name))
                width = 256 if index == 0 else 128
                with self.assertRaisesRegex(Exception, "must have|must be"):
                    session = self._session(model, optimize=False)
                    session.run(None, {"x": np.zeros((1, width), dtype=np.float16)})


if __name__ == "__main__":
    unittest.main()
