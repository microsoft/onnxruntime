# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Standalone CUDA extension tests; no PyTorch or publisher runtime is required."""

from __future__ import annotations

import ctypes
import json
import os
import tempfile
import unittest
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx_ir as ir
from onnxscript import GraphBuilder

import onnxruntime as ort

DOMAIN = "com.microsoft.source_compat"
BF16 = ml_dtypes.bfloat16


class SourceCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        library = os.environ.get("ORT_SOURCE_COMPAT_CUDA_LIBRARY")
        if not library:
            raise unittest.SkipTest("Set ORT_SOURCE_COMPAT_CUDA_LIBRARY to the compiled extension")
        cls.library = str(Path(library).resolve(strict=True))
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise unittest.SkipTest("CUDAExecutionProvider is unavailable")

    def run_operator(self, name, feeds, attributes, outputs=1):
        with tempfile.TemporaryDirectory() as directory:
            graph = ir.Graph([], [], nodes=[], opset_imports={"": 24, DOMAIN: 1})
            builder = GraphBuilder(graph)
            inputs = [
                builder.input(
                    key,
                    dtype=ir.DataType.INT64 if value.dtype == np.int64 else ir.DataType.BFLOAT16,
                    shape=value.shape,
                )
                for key, value in feeds.items()
            ]
            result = builder.op._call_op(
                name, inputs, {**attributes, "_domain": DOMAIN, "_version": 1, "_outputs": outputs}
            )
            for index, value in enumerate((result,) if outputs == 1 else result):
                value.type = ir.TensorType(ir.DataType.BFLOAT16)
                value.shape = ir.Shape(next(iter(feeds.values())).shape)
                builder.add_output(value, f"output_{index}")
            path = Path(directory) / "test.onnx"
            ir.save(ir.Model(graph, ir_version=10), path)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.enable_profiling = True
            options.profile_file_prefix = str(Path(directory) / "profile")
            options.register_custom_ops_library(self.library)
            session = ort.InferenceSession(str(path), options, providers=["CUDAExecutionProvider"])
            binding = session.io_binding()
            values = {}
            for key, source_array in feeds.items():
                array = np.ascontiguousarray(source_array)
                values[key] = (
                    ort.OrtValue.ortvalue_from_numpy(array)
                    if array.dtype == np.int64
                    else ort.OrtValue.ortvalue_from_numpy_with_onnx_type(array.view(np.uint16), 16)
                )
                binding.bind_ortvalue_input(key, values[key])
            for output in session.get_outputs():
                binding.bind_output(output.name, "cpu")
            session.run_with_iobinding(binding)
            binding.synchronize_outputs()
            actual = []
            for value in binding.get_outputs():
                count = int(np.prod(value.shape()))
                raw = (ctypes.c_uint16 * count).from_address(value.data_ptr())
                actual.append(np.ctypeslib.as_array(raw).copy().view(BF16).reshape(value.shape()).astype(np.float32))
            profile = json.loads(Path(session.end_profiling()).read_text())
            assert any(
                event.get("args", {}).get("op_name") == name
                and event["args"].get("provider") == "CUDAExecutionProvider"
                for event in profile
            ), "The custom operator did not execute on CUDA"
            return actual

    def test_norm_and_unrounded_residual(self):
        rng = np.random.default_rng(17)
        hidden = rng.normal(size=(1, 3, 2048)).astype(BF16)
        residual = rng.normal(size=hidden.shape).astype(BF16)
        weight = rng.normal(size=(2048,)).astype(BF16)
        for skip in (False, True):
            with self.subTest(skip=skip):
                combined = hidden.astype(np.float32)
                feeds = {"hidden": hidden}
                if skip:
                    combined += residual.astype(np.float32)
                    feeds["residual"] = residual
                feeds["weight"] = weight
                expected = (
                    (
                        combined
                        / np.sqrt(np.mean(combined**2, axis=-1, keepdims=True) + 1e-6)
                        * weight.astype(np.float32)
                    )
                    .astype(BF16)
                    .astype(np.float32)
                )
                actual = self.run_operator(
                    "SkipRMSNorm" if skip else "RMSNorm",
                    feeds,
                    {"epsilon": 1e-6},
                    outputs=2 if skip else 1,
                )
                np.testing.assert_allclose(actual[0], expected, atol=1e-2, rtol=1e-2)
                if skip:
                    np.testing.assert_array_equal(actual[1], combined.astype(BF16).astype(np.float32))

    def test_attention_prefill_decode_and_window_edges(self):
        rng = np.random.default_rng(31)
        for window, query_length, total in [
            (-1, 1, 1),
            (-1, 17, 17),
            (512, 512, 512),
            (512, 1, 513),
            (512, 1, 514),
            (512, 17, 530),
        ]:
            with self.subTest(window=window, query_length=query_length, total=total):
                q = (rng.normal(size=(1, query_length, 4 * 128)) * 0.2).astype(BF16)
                k = (rng.normal(size=(1, total, 2 * 128)) * 0.2).astype(BF16)
                v = rng.normal(size=k.shape).astype(BF16)
                sinks = np.array([-0.7, 0.2, 0.5, 0.9], dtype=BF16)
                query = q.astype(np.float32).reshape(query_length, 4, 128).transpose(1, 0, 2)
                key = np.repeat(k.astype(np.float32).reshape(total, 2, 128), 2, axis=1).transpose(1, 0, 2)
                value = np.repeat(v.astype(np.float32).reshape(total, 2, 128), 2, axis=1).transpose(1, 0, 2)
                scores = (query @ key.transpose(0, 2, 1)) * 128**-0.5
                distance = np.arange(total - query_length, total)[:, None] - np.arange(total)[None, :]
                allowed = distance >= 0
                if window != -1:
                    allowed &= distance <= window
                scores = np.where(allowed[None], scores, -np.inf)
                sink = np.broadcast_to(sinks.astype(np.float32)[:, None, None], (4, query_length, 1))
                scores = np.concatenate((scores, sink), axis=-1)
                probability = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
                probability /= probability.sum(axis=-1, keepdims=True)
                expected = (probability[..., :-1] @ value).transpose(1, 0, 2).reshape(q.shape)
                actual = self.run_operator(
                    "SinkAttention",
                    {"q": q, "k": k, "v": v, "sinks": sinks, "mask": np.ones((1, total), np.int64)},
                    {"q_num_heads": 4, "kv_num_heads": 2, "window_left": window, "scale": 128**-0.5},
                )[0]
                np.testing.assert_allclose(actual, expected, atol=1e-2, rtol=1e-2)

    def test_rejects_padding_and_invalid_norm_contracts(self):
        with self.assertRaisesRegex(Exception, "all-one attention mask"):
            self.run_operator(
                "SinkAttention",
                {
                    "q": np.zeros((1, 1, 512), BF16),
                    "k": np.zeros((1, 2, 256), BF16),
                    "v": np.zeros((1, 2, 256), BF16),
                    "sinks": np.zeros(4, BF16),
                    "mask": np.array([[0, 1]], np.int64),
                },
                {"q_num_heads": 4, "kv_num_heads": 2, "window_left": -1, "scale": 128**-0.5},
            )
        with self.assertRaisesRegex(Exception, "finite and positive"):
            self.run_operator(
                "RMSNorm",
                {"hidden": np.ones((1, 8), BF16), "weight": np.ones(8, BF16)},
                {"epsilon": 0.0},
            )
        with self.assertRaisesRegex(Exception, "matching one-dimensional scale"):
            self.run_operator(
                "RMSNorm",
                {"hidden": np.ones((1, 8), BF16), "weight": np.ones(16, BF16)},
                {"epsilon": 1e-6},
            )


if __name__ == "__main__":
    unittest.main()
