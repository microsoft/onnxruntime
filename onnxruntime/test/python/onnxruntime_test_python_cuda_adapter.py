# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""CUDA-plugin adapter regressions requiring actual operator-name semantics."""

import os
import unittest

import numpy as np
from onnx import TensorProto, helper, numpy_helper

import onnxruntime as ort


def group_norm_model(name, dtype, activation, downstream=False):
    element = TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT
    shape = ["N", 2, 2, 320]
    nodes = [
        helper.make_node(
            "SkipGroupNorm",
            ["x", "gamma", "beta", "skip"],
            ["y", "residual"],
            domain="com.microsoft",
            name=name,
            groups=32,
            activation=activation,
            epsilon=1e-5,
        )
    ]
    outputs = [helper.make_tensor_value_info("y", element, shape)]
    if downstream:
        nodes.extend(
            [
                helper.make_node(
                    "Transpose", ["residual"], ["transposed"], perm=[0, 3, 1, 2], name="residual_transpose"
                ),
                helper.make_node("Shape", ["transposed"], ["shape"], name="residual_shape"),
            ]
        )
        outputs.extend(
            [
                helper.make_tensor_value_info("transposed", element, ["N", 320, 2, 2]),
                helper.make_tensor_value_info("shape", TensorProto.INT64, [4]),
            ]
        )
    else:
        outputs.append(helper.make_tensor_value_info("residual", element, shape))
    graph = helper.make_graph(
        nodes,
        "adapter_operator_identity",
        [helper.make_tensor_value_info("x", element, shape), helper.make_tensor_value_info("skip", element, shape)],
        outputs,
        [
            numpy_helper.from_array(np.ones(320, dtype=np.float32), "gamma"),
            numpy_helper.from_array(np.zeros(320, dtype=np.float32), "beta"),
        ],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 14), helper.make_opsetid("com.microsoft", 1)], ir_version=7
    )


def reference(x, skip, activation):
    values = x.astype(np.float32) + skip.astype(np.float32)
    grouped = values.reshape(1, 2, 2, 32, 10)
    mean = grouped.mean(axis=(1, 2, 4), keepdims=True)
    variance = ((grouped - mean) ** 2).mean(axis=(1, 2, 4), keepdims=True)
    normalized = ((grouped - mean) / np.sqrt(variance + np.float32(1e-5))).reshape(values.shape)
    if activation:
        normalized *= 1 / (1 + np.exp(-normalized))
    return normalized.astype(x.dtype), values.astype(x.dtype)


class TestCudaAdapterOperatorIdentity(unittest.TestCase):
    def setUp(self):
        if "cuda-plugin-ep=1" not in ort.get_build_info():
            if os.environ.get("ORT_STATEFUL_REQUIRE_CUDA_PLUGIN") == "1":
                self.fail("Qualification requires the actual CUDA-plugin Python extension")
            self.skipTest("CUDA-plugin adapter qualification requires the plugin build")
        self.options = ort.SessionOptions()
        self.options.intra_op_num_threads = 1
        self.options.inter_op_num_threads = 1
        self.options.add_session_config_entry("session.record_ep_graph_assignment_info", "1")
        devices = [
            value
            for value in ort.get_ep_devices()
            if value.ep_name == "CUDAExecutionProvider"
            and value.memory_info(ort.OrtDeviceMemoryType.DEFAULT).device_id == 0
        ]
        self.assertEqual(len(devices), 1)
        self.options.add_provider_for_devices(devices, {})

    def run_case(self, name, dtype, activation, downstream):
        model = group_norm_model(name, dtype, activation, downstream)
        session = ort.InferenceSession(model.SerializeToString(), self.options, enable_fallback=False)
        assignments = session.get_provider_graph_assignment_info()
        self.assertTrue(
            any(
                group.ep_name == "CUDAExecutionProvider"
                and any(node.op_type == "SkipGroupNorm" for node in group.get_nodes())
                for group in assignments
            )
        )
        x = np.random.default_rng(42).standard_normal((1, 2, 2, 320)).astype(dtype)
        skip = np.full(x.shape, 0.5, dtype=dtype)
        expected, residual = reference(x, skip, activation)
        actual = session.run(None, {"x": x, "skip": skip})
        np.testing.assert_allclose(actual[0], expected, rtol=2e-2, atol=2e-2)
        if downstream:
            np.testing.assert_array_equal(actual[1], residual.transpose(0, 3, 1, 2))
            np.testing.assert_array_equal(actual[2], [1, 320, 2, 2])
        else:
            np.testing.assert_array_equal(actual[1], residual)

    def test_operator_identity_is_independent_of_node_name(self):
        for name in ("SkipGroupNorm", "SkipGroupNorm_1", "arbitrary_residual_block", ""):
            for dtype in (np.float32, np.float16):
                for activation in (0, 1):
                    with self.subTest(name=name, dtype=dtype, activation=activation):
                        self.run_case(name, dtype, activation, False)

    def test_residual_reaches_downstream_transpose_and_shape(self):
        for dtype in (np.float32, np.float16):
            with self.subTest(dtype=dtype):
                self.run_case("residual_block_with_a_distinct_node_name", dtype, 1, True)


if __name__ == "__main__":
    unittest.main()
