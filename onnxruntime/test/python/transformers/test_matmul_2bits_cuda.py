# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Exact row-count invariance for the FP16 INT2 CUDA decode kernels."""

import unittest

import numpy as np
import torch
from onnx import TensorProto, helper, numpy_helper

import onnxruntime as ort


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
    "INT2 CUDA decode tests require SM80 or newer",
)
class TestMatMul2BitsCuda(unittest.TestCase):
    def _check_rows(self, k, n, block_size=128, explicit_zero_points=False):
        rng = np.random.default_rng(184)
        blocks = k // block_size
        weight = rng.integers(0, 256, (n, blocks, block_size // 4), dtype=np.uint8)
        scales = rng.uniform(0.001, 0.025, (n, blocks)).astype(np.float16)
        initializers = [
            numpy_helper.from_array(weight, "weight"),
            numpy_helper.from_array(scales, "scales"),
        ]
        inputs = ["x", "weight", "scales"]
        if explicit_zero_points:
            zero_points = rng.integers(0, 256, (n, (blocks + 3) // 4), dtype=np.uint8)
            initializers.append(numpy_helper.from_array(zero_points, "zero_points"))
            inputs.append("zero_points")
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "MatMulNBits",
                        inputs,
                        ["y"],
                        domain="com.microsoft",
                        K=k,
                        N=n,
                        block_size=block_size,
                        bits=2,
                    )
                ],
                "int2_decode",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT16, ["rows", k])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT16, ["rows", n])],
                initializers,
            ),
            opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
        )
        options = ort.SessionOptions()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        session = ort.InferenceSession(
            model.SerializeToString(),
            options,
            providers=["CUDAExecutionProvider"],
        )
        x = rng.normal(0, 0.5, (8, k)).astype(np.float16)
        # M=1 uses the separate non-batched kernel with the same half2 accumulation order.
        reference = np.concatenate([session.run(None, {"x": row[None]})[0] for row in x])
        self.assertTrue(np.isfinite(reference).all())
        for rows in (2, 3, 4, 5, 8):
            with self.subTest(k=k, n=n, rows=rows, block_size=block_size, zero_points=explicit_zero_points):
                actual = session.run(None, {"x": x[:rows]})[0]
                np.testing.assert_array_equal(actual, reference[:rows])

    def test_ornith_projection_shapes(self):
        for k, n in ((8192, 3072), (3072, 2048), (1024, 3072), (3072, 20608), (3072, 17408)):
            self._check_rows(k, n)

    def test_unroll_boundaries_and_partial_iterations(self):
        for k, n in ((128, 16), (640, 16), (3072, 24), (7680, 80), (8192, 4096), (8192, 4112), (8320, 32), (16384, 32)):
            self._check_rows(k, n)

    def test_other_group_sizes_and_explicit_zero_points(self):
        for block_size in (16, 32, 64, 128, 256):
            for explicit in (False, True):
                self._check_rows(1024, 32, block_size, explicit)


if __name__ == "__main__":
    unittest.main()
