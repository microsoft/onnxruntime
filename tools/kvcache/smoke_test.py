# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Executable Python cache feed/fetch example and dense-reference integration test."""

import unittest

import numpy as np
import onnx

import onnxruntime as ort


def model_bytes(window=0):
    cache_type = onnx.TypeProto()
    cache_type.kv_cache_type.elem_type = onnx.TensorProto.FLOAT
    for name, size in (("batch_size", 1), ("num_kv_heads", 1), ("key_head_size", 2), ("value_head_size", 3)):
        getattr(cache_type.kv_cache_type, name).dim_value = size
    cache_type.kv_cache_type.sliding_window = window
    cache_in = onnx.ValueInfoProto(name="past_cache")
    cache_in.type.CopyFrom(cache_type)
    cache_out = onnx.ValueInfoProto(name="present_cache")
    cache_out.type.CopyFrom(cache_type)
    node = onnx.helper.make_node(
        "AttentionWithCache",
        ["q", "k", "v", "past_cache", "mask"],
        ["y", "present_cache"],
        is_causal=1,
        sliding_window=window,
    )
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Cast", ["mask_int64"], ["mask"], to=onnx.TensorProto.BOOL), node],
        "kv_cache_smoke",
        [
            onnx.helper.make_tensor_value_info("q", onnx.TensorProto.FLOAT, [1, 2, "S", 2]),
            onnx.helper.make_tensor_value_info("k", onnx.TensorProto.FLOAT, [1, 1, "S", 2]),
            onnx.helper.make_tensor_value_info("v", onnx.TensorProto.FLOAT, [1, 1, "S", 3]),
            onnx.helper.make_tensor_value_info("mask_int64", onnx.TensorProto.INT64, [1, 1, "S", "total"]),
            cache_in,
        ],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 2, "S", 3]), cache_out],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 29)], ir_version=15)
    onnx.checker.check_model(model)
    return model.SerializeToString()


def dense_attention(q, keys, values, start, window):
    end = keys.shape[2]
    scores = np.matmul(q.astype(np.float64), np.swapaxes(keys.astype(np.float64), -2, -1)) / np.sqrt(2)
    positions = np.arange(end)
    queries = start + np.arange(q.shape[2])
    visible = positions[None, :] <= queries[:, None]
    if window:
        visible &= positions[None, :] >= queries[:, None] - window + 1
    scores = np.where(visible, scores, -np.inf)
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights /= np.sum(weights, axis=-1, keepdims=True)
    return np.matmul(weights, values).astype(np.float32)


class KVCacheIntegrationTest(unittest.TestCase):
    def test_swish_float32_in_lifted_graph(self):
        for alpha in (1.0, 0.0, -0.75):
            graph = onnx.helper.make_graph(
                [onnx.helper.make_node("Swish", ["x"], ["y"], alpha=alpha)],
                "swish",
                [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [7])],
                [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [7])],
            )
            model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 29)], ir_version=15)
            session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
            x = np.array([-100, -10, -1, 0, 1, 10, 100], dtype=np.float32)
            expected = x.astype(np.float64) / (1 + np.exp(-alpha * x.astype(np.float64)))
            np.testing.assert_allclose(session.run(None, {"x": x})[0], expected, atol=2e-6, rtol=2e-6)

    def test_identical_model_with_runtime_selected_storage(self):
        rng = np.random.default_rng(42)
        chunks = [
            (
                rng.normal(size=(1, 2, count, 2)).astype(np.float32),
                rng.normal(size=(1, 1, count, 2)).astype(np.float32),
                rng.normal(size=(1, 1, count, 3)).astype(np.float32),
            )
            for count in (3, 7, 1, 2)
        ]
        for window in (0, 1, 4):
            options = ort.SessionOptions()
            options.add_session_config_entry("session.allow_released_opsets_only", "0")
            options.intra_op_num_threads = 1
            session = ort.InferenceSession(model_bytes(window), options, providers=["CPUExecutionProvider"])
            for backend in ("contiguous", "paged", "ring"):
                if backend == "ring" and not window:
                    continue
                with self.subTest(window=window, backend=backend):
                    cache = ort.OrtValue.kv_cache(1, 1, 2, 3, backend=backend, page_size=2, sliding_window=window)
                    keys = np.empty((1, 1, 0, 2), dtype=np.float32)
                    values = np.empty((1, 1, 0, 3), dtype=np.float32)
                    for q, k, v in chunks:
                        start = keys.shape[2]
                        keys = np.concatenate((keys, k), axis=2)
                        values = np.concatenate((values, v), axis=2)
                        old = cache
                        feeds = {
                            name: ort.OrtValue.ortvalue_from_numpy(array)
                            for name, array in (("q", q), ("k", k), ("v", v))
                        }
                        feeds["past_cache"] = cache
                        feeds["mask_int64"] = ort.OrtValue.ortvalue_from_numpy(
                            np.ones((1, 1, q.shape[2], keys.shape[2]), dtype=np.int64)
                        )
                        y, cache = session.run_with_ort_values(["y", "present_cache"], feeds)
                        np.testing.assert_allclose(
                            y.numpy(), dense_attention(q, keys, values, start, window), atol=2e-6
                        )
                        self.assertEqual(old.kv_cache_describe()["computed_tokens"], start)
                        end = keys.shape[2]
                        self.assertEqual(
                            cache.kv_cache_describe(),
                            {
                                "computed_tokens": end,
                                "retained_begin": max(0, end - window) if window else 0,
                                "backend": backend,
                            },
                        )


if __name__ == "__main__":
    unittest.main()
