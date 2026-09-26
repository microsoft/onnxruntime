# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


def create_model(context_length: int, head_size: int, num_heads: int, use_mask: bool) -> bytes:
    token_budget = 2048
    compress_ratio = 4
    capacity = token_budget + compress_ratio - 1
    inputs = [
        helper.make_tensor_value_info("query", TensorProto.FLOAT16, [1, 1, num_heads * head_size]),
        helper.make_tensor_value_info("key", TensorProto.FLOAT16, [1, 1, head_size]),
        helper.make_tensor_value_info("weight", TensorProto.FLOAT16, [head_size]),
        helper.make_tensor_value_info("cosine", TensorProto.FLOAT16, [1, context_length, head_size]),
        helper.make_tensor_value_info("sine", TensorProto.FLOAT16, [1, context_length, head_size]),
    ]
    if use_mask:
        inputs.append(helper.make_tensor_value_info("mask", TensorProto.INT64, [1, context_length]))
    inputs.extend(
        [
            helper.make_tensor_value_info("key_cache", TensorProto.FLOAT16, [1, context_length, head_size]),
            helper.make_tensor_value_info("past_sequence_length", TensorProto.INT32, [1]),
        ]
    )
    outputs = [
        helper.make_tensor_value_info("selected", TensorProto.INT32, [1, 1, capacity]),
        helper.make_tensor_value_info("present_key", TensorProto.FLOAT16, [1, context_length, head_size]),
    ]
    node = helper.make_node(
        "SparseAttentionIndexer",
        [
            "query",
            "key",
            "weight",
            "cosine",
            "sine",
            "mask" if use_mask else "",
            "key_cache",
            "",
            "",
            "",
            "",
            "past_sequence_length",
        ],
        ["selected", "present_key"],
        domain="com.microsoft",
        policy_mode="qsa",
        compress_ratio=compress_ratio,
        token_budget=token_budget,
    )
    graph = helper.make_graph([node], "sparse_attention_indexer_benchmark", inputs, outputs)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )
    return model.SerializeToString()


def benchmark(context_length: int, warmup: int, iterations: int, use_mask: bool) -> float:
    head_size = 128
    num_heads = 4
    session = ort.InferenceSession(
        create_model(context_length, head_size, num_heads, use_mask),
        providers=["CUDAExecutionProvider"],
    )
    inputs = {
        "query": np.zeros((1, 1, num_heads * head_size), dtype=np.float16),
        "key": np.zeros((1, 1, head_size), dtype=np.float16),
        "weight": np.ones((head_size,), dtype=np.float16),
        "cosine": np.ones((1, context_length, head_size), dtype=np.float16),
        "sine": np.zeros((1, context_length, head_size), dtype=np.float16),
        "key_cache": np.zeros((1, context_length, head_size), dtype=np.float16),
        "past_sequence_length": np.array([context_length - 1], dtype=np.int32),
    }
    if use_mask:
        inputs["mask"] = np.ones((1, context_length), dtype=np.int64)
    io_binding = session.io_binding()
    device_inputs = {}
    for name, value in inputs.items():
        device_inputs[name] = ort.OrtValue.ortvalue_from_numpy(value, "cuda", 0)
        io_binding.bind_ortvalue_input(name, device_inputs[name])
    selected = ort.OrtValue.ortvalue_from_shape_and_type([1, 1, 2051], np.int32, "cuda", 0)
    io_binding.bind_ortvalue_output("selected", selected)
    io_binding.bind_ortvalue_output("present_key", device_inputs["key_cache"])

    for _ in range(warmup):
        session.run_with_iobinding(io_binding)
    io_binding.synchronize_outputs()

    start = time.perf_counter()
    for _ in range(iterations):
        session.run_with_iobinding(io_binding)
    io_binding.synchronize_outputs()
    return (time.perf_counter() - start) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CUDA QSA SparseAttentionIndexer decode")
    parser.add_argument("--contexts", nargs="+", type=int, default=[8192, 32768, 65536, 131072, 262144])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--use-mask", action="store_true")
    args = parser.parse_args()

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider is unavailable")
    for context_length in args.contexts:
        latency = benchmark(context_length, args.warmup, args.iterations, args.use_mask)
        print(f"context={context_length:>6} latency_ms={latency:.3f}")


if __name__ == "__main__":
    main()
