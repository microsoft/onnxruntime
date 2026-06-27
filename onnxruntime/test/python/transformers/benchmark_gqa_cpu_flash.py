#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Benchmark CPU GroupQueryAttention: Flash Attention vs Naive (full materialization).

Runs the actual GQA operator via InferenceSession, toggling between flash and
naive paths using the ORT_GQA_DISABLE_FLASH_ATTENTION environment variable.

Usage:
    python benchmark_gqa_cpu_flash.py
    python benchmark_gqa_cpu_flash.py --decode_only
    python benchmark_gqa_cpu_flash.py --prompt_only
"""

import argparse
import os
import time

import numpy as np
from onnx import TensorProto, helper

from onnxruntime import InferenceSession, SessionOptions


def create_quantized_gqa_graph(
    batch_size,
    seq_len,
    num_heads,
    kv_num_heads,
    head_size,
    quant_type,
    bit_width,
    buffer_seq_len=None,
):
    """Create an ONNX graph for GroupQueryAttention with quantized KV cache."""
    if buffer_seq_len is None:
        buffer_seq_len = seq_len

    hidden_size = num_heads * head_size
    kv_hidden_size = kv_num_heads * head_size
    packed_head_size = head_size // 2 if bit_width == 4 else head_size
    cache_ort_type = TensorProto.UINT8 if bit_width == 4 else TensorProto.INT8

    inputs = [
        "query",
        "key",
        "value",
        "past_key",
        "past_value",
        "seqlens_k",
        "total_sequence_length",
        "",
        "",
        "",
        "",
        "",  # cos, sin, position_ids, attention_bias, head_sink
        "k_scale",
        "v_scale",
    ]
    while inputs and inputs[-1] == "":
        inputs.pop()

    node = helper.make_node(
        op_type="GroupQueryAttention",
        inputs=inputs,
        outputs=["output", "present_key", "present_value"],
        name="GroupQueryAttention_0",
        num_heads=num_heads,
        kv_num_heads=kv_num_heads,
        k_quant_type=quant_type,
        v_quant_type=quant_type,
        kv_cache_bit_width=bit_width,
        domain="com.microsoft",
    )

    graph_input = [
        helper.make_tensor_value_info("query", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]),
        helper.make_tensor_value_info("key", TensorProto.FLOAT, [batch_size, seq_len, kv_hidden_size]),
        helper.make_tensor_value_info("value", TensorProto.FLOAT, [batch_size, seq_len, kv_hidden_size]),
        helper.make_tensor_value_info(
            "past_key", cache_ort_type, [batch_size, kv_num_heads, buffer_seq_len, packed_head_size]
        ),
        helper.make_tensor_value_info(
            "past_value", cache_ort_type, [batch_size, kv_num_heads, buffer_seq_len, packed_head_size]
        ),
        helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [batch_size]),
        helper.make_tensor_value_info("total_sequence_length", TensorProto.INT32, [1]),
        helper.make_tensor_value_info("k_scale", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("v_scale", TensorProto.FLOAT, None),
    ]

    graph_output = [
        helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]),
        helper.make_tensor_value_info(
            "present_key", cache_ort_type, [batch_size, kv_num_heads, buffer_seq_len, packed_head_size]
        ),
        helper.make_tensor_value_info(
            "present_value", cache_ort_type, [batch_size, kv_num_heads, buffer_seq_len, packed_head_size]
        ),
    ]

    graph = helper.make_graph([node], "BenchGQA", graph_input, graph_output)
    model = helper.make_model(graph)
    return model.SerializeToString()


def create_fp32_gqa_graph(
    batch_size,
    seq_len,
    num_heads,
    kv_num_heads,
    head_size,
    buffer_seq_len=None,
):
    """Create an ONNX graph for GroupQueryAttention with a non-quantized FP32 KV cache."""
    if buffer_seq_len is None:
        buffer_seq_len = seq_len

    hidden_size = num_heads * head_size
    kv_hidden_size = kv_num_heads * head_size

    inputs = [
        "query",
        "key",
        "value",
        "past_key",
        "past_value",
        "seqlens_k",
        "total_sequence_length",
    ]

    node = helper.make_node(
        op_type="GroupQueryAttention",
        inputs=inputs,
        outputs=["output", "present_key", "present_value"],
        name="GroupQueryAttention_0",
        num_heads=num_heads,
        kv_num_heads=kv_num_heads,
        domain="com.microsoft",
    )

    graph_input = [
        helper.make_tensor_value_info("query", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]),
        helper.make_tensor_value_info("key", TensorProto.FLOAT, [batch_size, seq_len, kv_hidden_size]),
        helper.make_tensor_value_info("value", TensorProto.FLOAT, [batch_size, seq_len, kv_hidden_size]),
        helper.make_tensor_value_info(
            "past_key", TensorProto.FLOAT, [batch_size, kv_num_heads, buffer_seq_len, head_size]
        ),
        helper.make_tensor_value_info(
            "past_value", TensorProto.FLOAT, [batch_size, kv_num_heads, buffer_seq_len, head_size]
        ),
        helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [batch_size]),
        helper.make_tensor_value_info("total_sequence_length", TensorProto.INT32, [1]),
    ]

    graph_output = [
        helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]),
        helper.make_tensor_value_info(
            "present_key", TensorProto.FLOAT, [batch_size, kv_num_heads, buffer_seq_len, head_size]
        ),
        helper.make_tensor_value_info(
            "present_value", TensorProto.FLOAT, [batch_size, kv_num_heads, buffer_seq_len, head_size]
        ),
    ]

    graph = helper.make_graph([node], "BenchGQA", graph_input, graph_output)
    model = helper.make_model(graph)
    return model.SerializeToString()


def benchmark_gqa(
    batch_size,
    seq_len,
    num_heads,
    kv_num_heads,
    head_size,
    quant_type,
    bit_width,
    past_seq_len=0,
    warmup=5,
    repeats=20,
    non_quantized=False,
):
    """Benchmark a single GQA configuration. Returns elapsed time in ms."""
    hidden_size = num_heads * head_size
    kv_hidden_size = kv_num_heads * head_size
    packed_head_size = head_size // 2 if bit_width == 4 else head_size

    total_seqlen = past_seq_len + seq_len
    buffer_seq_len = total_seqlen

    sess_options = SessionOptions()
    sess_options.intra_op_num_threads = 8

    np.random.seed(42)
    query = np.random.uniform(-0.5, 0.5, (batch_size, seq_len, hidden_size)).astype(np.float32)
    key = np.random.uniform(-0.5, 0.5, (batch_size, seq_len, kv_hidden_size)).astype(np.float32)
    value = np.random.uniform(-0.5, 0.5, (batch_size, seq_len, kv_hidden_size)).astype(np.float32)
    seqlens_k = np.array([total_seqlen - 1] * batch_size, dtype=np.int32)
    total_seq = np.array([total_seqlen], dtype=np.int32)

    if non_quantized:
        onnx_model_str = create_fp32_gqa_graph(
            batch_size,
            seq_len,
            num_heads,
            kv_num_heads,
            head_size,
            buffer_seq_len=buffer_seq_len,
        )
        sess = InferenceSession(onnx_model_str, sess_options, providers=["CPUExecutionProvider"])

        past_k = np.random.uniform(-0.5, 0.5, (batch_size, kv_num_heads, buffer_seq_len, head_size)).astype(np.float32)
        past_v = np.random.uniform(-0.5, 0.5, (batch_size, kv_num_heads, buffer_seq_len, head_size)).astype(np.float32)

        feeds = {
            "query": query,
            "key": key,
            "value": value,
            "past_key": past_k,
            "past_value": past_v,
            "seqlens_k": seqlens_k,
            "total_sequence_length": total_seq,
        }
    else:
        onnx_model_str = create_quantized_gqa_graph(
            batch_size,
            seq_len,
            num_heads,
            kv_num_heads,
            head_size,
            quant_type,
            bit_width,
            buffer_seq_len=buffer_seq_len,
        )
        sess = InferenceSession(onnx_model_str, sess_options, providers=["CPUExecutionProvider"])

        cache_dtype = np.uint8 if bit_width == 4 else np.int8
        past_k = np.random.randint(
            0, 255, (batch_size, kv_num_heads, buffer_seq_len, packed_head_size), dtype=np.uint8
        ).view(cache_dtype)
        past_v = np.random.randint(
            0, 255, (batch_size, kv_num_heads, buffer_seq_len, packed_head_size), dtype=np.uint8
        ).view(cache_dtype)

        per_channel = quant_type == "PER_CHANNEL"
        scale_size = kv_num_heads * head_size if per_channel else 1
        k_scale = np.full(scale_size, 0.01, dtype=np.float32)
        v_scale = np.full(scale_size, 0.01, dtype=np.float32)

        feeds = {
            "query": query,
            "key": key,
            "value": value,
            "past_key": past_k,
            "past_value": past_v,
            "seqlens_k": seqlens_k,
            "total_sequence_length": total_seq,
            "k_scale": k_scale,
            "v_scale": v_scale,
        }

    # Warmup
    for _ in range(warmup):
        sess.run(None, feeds)

    # Benchmark
    start = time.perf_counter()
    for _ in range(repeats):
        sess.run(None, feeds)
    elapsed_ms = (time.perf_counter() - start) / repeats * 1000.0

    return elapsed_ms


def run_benchmarks(args):
    """Run flash vs naive benchmarks for various configurations."""

    configs = []

    if not args.decode_only:
        # Prefill configurations: seq_len = total_seqlen (prompt phase)
        for total_seqlen in [512, 1024, 2048, 4096]:
            configs.append(
                {
                    "label": f"Prefill S={total_seqlen}",
                    "batch_size": 1,
                    "seq_len": total_seqlen,
                    "num_heads": 16,
                    "kv_num_heads": 8,
                    "head_size": 128,
                    "quant_type": "PER_TENSOR",
                    "bit_width": 8,
                    "past_seq_len": 0,
                }
            )

    if not args.prompt_only:
        # Decode configurations: seq_len=1, varying past
        for past_seqlen in [512, 1024, 2048, 4096]:
            configs.append(
                {
                    "label": f"Decode T={past_seqlen + 1}",
                    "batch_size": 1,
                    "seq_len": 1,
                    "num_heads": 16,
                    "kv_num_heads": 8,
                    "head_size": 128,
                    "quant_type": "PER_TENSOR",
                    "bit_width": 8,
                    "past_seq_len": past_seqlen,
                }
            )

    if not args.decode_only and not args.prompt_only:
        # Batch decode
        configs.append(
            {
                "label": "Decode B=4 T=2049",
                "batch_size": 4,
                "seq_len": 1,
                "num_heads": 16,
                "kv_num_heads": 8,
                "head_size": 128,
                "quant_type": "PER_TENSOR",
                "bit_width": 8,
                "past_seq_len": 2048,
            }
        )
        # INT4 prefill (quantized mode only)
        if not args.fp32:
            configs.append(
                {
                    "label": "Prefill S=2048 INT4",
                    "batch_size": 1,
                    "seq_len": 2048,
                    "num_heads": 16,
                    "kv_num_heads": 8,
                    "head_size": 128,
                    "quant_type": "PER_TENSOR",
                    "bit_width": 4,
                    "past_seq_len": 0,
                }
            )

    warmup = args.warmup
    repeats = args.repeats

    # Save and restore env var to avoid side effects on callers
    saved_env = os.environ.get("ORT_GQA_DISABLE_FLASH_ATTENTION")

    kv_mode = "FP32 (non-quantized)" if args.fp32 else "INT8/INT4 quantized"
    print("\nBenchmark: CPU GroupQueryAttention — Flash vs Naive")
    print(f"KV cache: {kv_mode}, Threads: {8}, Warmup: {warmup}, Repeats: {repeats}")
    print(f"{'Config':<25} {'Naive (ms)':>12} {'Flash (ms)':>12} {'Speedup':>10}")
    print("-" * 62)

    for cfg in configs:
        label = cfg.pop("label")
        cfg["non_quantized"] = args.fp32

        # Flash path (default)
        os.environ.pop("ORT_GQA_DISABLE_FLASH_ATTENTION", None)
        flash_ms = benchmark_gqa(**cfg, warmup=warmup, repeats=repeats)

        # Naive path (disabled flash)
        os.environ["ORT_GQA_DISABLE_FLASH_ATTENTION"] = "1"
        naive_ms = benchmark_gqa(**cfg, warmup=warmup, repeats=repeats)

        speedup = naive_ms / flash_ms if flash_ms > 0 else float("inf")
        print(f"{label:<25} {naive_ms:>10.3f}ms {flash_ms:>10.3f}ms {speedup:>8.2f}x")

    # Restore original env state
    if saved_env is not None:
        os.environ["ORT_GQA_DISABLE_FLASH_ATTENTION"] = saved_env
    else:
        os.environ.pop("ORT_GQA_DISABLE_FLASH_ATTENTION", None)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark GQA flash vs naive on CPU")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=20, help="Measurement iterations")
    parser.add_argument("--decode_only", action="store_true", help="Only run decode benchmarks")
    parser.add_argument("--prompt_only", action="store_true", help="Only run prompt benchmarks")
    parser.add_argument("--fp32", action="store_true", help="Use non-quantized FP32 KV cache instead of quantized")
    args = parser.parse_args()
    run_benchmarks(args)
