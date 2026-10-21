# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

import numpy as np
from onnx import TensorProto, helper
from parameterized import parameterized

import onnxruntime as ort


def make_model(k, n, *, fused=True):
    inputs = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT16, ["M", k]),
        helper.make_tensor_value_info("w", TensorProto.FLOAT16, [1, k]),
    ] + [
        helper.make_tensor_value_info(name, TensorProto.FLOAT16, ["M", n]) for name in ("shared", "routed", "residual")
    ]
    if fused:
        nodes = [
            helper.make_node(
                "MatMulSigmoidGatedResidual",
                ["x", "w", "shared", "routed", "residual"],
                ["output"],
                domain="com.microsoft",
            )
        ]
    else:
        nodes = [
            helper.make_node("Transpose", ["w"], ["wt"], perm=[1, 0]),
            helper.make_node("MatMul", ["x", "wt"], ["dot"]),
            helper.make_node("Sigmoid", ["dot"], ["gate"]),
            helper.make_node("Mul", ["shared", "gate"], ["gated"]),
            helper.make_node("Add", ["routed", "gated"], ["combined"]),
            helper.make_node("Add", ["residual", "combined"], ["output"]),
        ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "gated",
            inputs,
            [helper.make_tensor_value_info("output", TensorProto.FLOAT16, ["M", n])],
        ),
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    return model.SerializeToString()


def session(model):
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return ort.InferenceSession(model, options, providers=["CUDAExecutionProvider"])


@unittest.skipUnless("CUDAExecutionProvider" in ort.get_available_providers(), "CUDA required")
class TestMatMulSigmoidGatedResidual(unittest.TestCase):
    @parameterized.expand([(1, 64, 65), (2, 3072, 3072), (8, 3072, 3072), (129, 513, 257), (0, 64, 32)])
    def test_matches_unfused(self, rows, k, n):
        rng = np.random.default_rng(83)
        feed = {
            "x": rng.normal(0, 0.1, (rows, k)).astype(np.float16),
            "w": rng.normal(0, 0.1, (1, k)).astype(np.float16),
            **{name: rng.normal(size=(rows, n)).astype(np.float16) for name in ("shared", "routed", "residual")},
        }
        actual = session(make_model(k, n)).run(None, feed)[0]
        expected = session(make_model(k, n, fused=False)).run(None, feed)[0]
        np.testing.assert_allclose(actual, expected, atol=0.004, rtol=0.003)

    def test_query_shape_invariance(self):
        rng = np.random.default_rng(741)
        feed = {
            "x": rng.normal(0, 0.1, (8, 3072)).astype(np.float16),
            "w": rng.normal(0, 0.1, (1, 3072)).astype(np.float16),
            **{name: rng.normal(size=(8, 3072)).astype(np.float16) for name in ("shared", "routed", "residual")},
        }
        model = session(make_model(3072, 3072))
        actual = model.run(None, feed)[0]
        expected = np.concatenate(
            [
                model.run(None, {name: value if name == "w" else value[i : i + 1] for name, value in feed.items()})[0]
                for i in range(8)
            ]
        )
        np.testing.assert_array_equal(actual, expected)

    def test_preserves_elementwise_rounding(self):
        rng = np.random.default_rng(19)
        feed = {
            "x": np.zeros((2, 64), np.float16),
            "w": np.zeros((1, 64), np.float16),
            **{name: rng.normal(size=(2, 513)).astype(np.float16) for name in ("shared", "routed", "residual")},
        }
        actual = session(make_model(64, 513)).run(None, feed)[0]
        expected = feed["residual"] + (feed["routed"] + feed["shared"] * np.float16(0.5))
        np.testing.assert_array_equal(actual, expected)

    def test_rejects_invalid_shapes(self):
        model = session(make_model(None, None))
        base = {
            "x": np.zeros((1, 64), np.float16),
            "w": np.zeros((1, 32), np.float16),
            **{name: np.zeros((1, 64), np.float16) for name in ("shared", "routed", "residual")},
        }
        with self.assertRaisesRegex(Exception, "weight must have shape"):
            model.run(None, base)
        base["w"] = np.zeros((1, 64), np.float16)
        base["routed"] = np.zeros((1, 32), np.float16)
        with self.assertRaisesRegex(Exception, "must have the same"):
            model.run(None, base)

    def test_saturated_and_nonfinite_gate(self):
        feed = {
            "x": np.zeros((5, 64), np.float16),
            "w": np.ones((1, 64), np.float16),
            "shared": np.full((5, 33), 2, np.float16),
            "routed": np.full((5, 33), 3, np.float16),
            "residual": np.full((5, 33), 4, np.float16),
        }
        feed["x"][:, 0] = [-1000, 1000, -np.inf, np.inf, np.nan]
        actual = session(make_model(64, 33)).run(None, feed)[0]
        expected = np.broadcast_to(np.array([7, 9, 7, 9, np.nan], np.float16)[:, None], (5, 33))
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
