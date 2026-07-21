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
    python benchmark_gqa_cpu_flash.py --fp32 --block_table --block_table_only --decode_only --trials 3
    python benchmark_gqa_cpu_flash.py --fp32 --block_table --decode_only --csv_output .\\gqa_report.csv

Flag notes:
    --block_table: Include FP32 block_table cache scenarios.
    --block_table_only: Run only block_table scenarios (requires --fp32 --block_table).
    --trials: Run independent trial repeats per config and report mean/min/max speedup.
    --csv_output: Optional CSV file path for exporting benchmark results.
"""

import argparse
import csv
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
        helper.make_tensor_value_info("present_key", cache_ort_type, None),
        helper.make_tensor_value_info("present_value", cache_ort_type, None),
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
    use_block_table=False,
    block_size=64,
):
    """Create an ONNX graph for GroupQueryAttention with a non-quantized FP32 KV cache."""
    if buffer_seq_len is None:
        buffer_seq_len = seq_len

    hidden_size = num_heads * head_size
    kv_hidden_size = kv_num_heads * head_size

    if use_block_table:
        # input 16 is block_table; inputs 7..15 remain optional placeholders.
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
            "",
            "",
            "",
            "",
            "",
            "block_table",
        ]
    else:
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

    if use_block_table:
        max_num_blocks_per_seq = (buffer_seq_len + block_size - 1) // block_size
        num_blocks = batch_size * max_num_blocks_per_seq
        graph_input = [
            helper.make_tensor_value_info("query", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]),
            helper.make_tensor_value_info("key", TensorProto.FLOAT, [batch_size, seq_len, kv_hidden_size]),
            helper.make_tensor_value_info("value", TensorProto.FLOAT, [batch_size, seq_len, kv_hidden_size]),
            helper.make_tensor_value_info(
                "past_key", TensorProto.FLOAT, [num_blocks, block_size, kv_num_heads, head_size]
            ),
            helper.make_tensor_value_info(
                "past_value", TensorProto.FLOAT, [num_blocks, block_size, kv_num_heads, head_size]
            ),
            helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [batch_size]),
            helper.make_tensor_value_info("total_sequence_length", TensorProto.INT32, [1]),
            helper.make_tensor_value_info("block_table", TensorProto.INT32, [batch_size, max_num_blocks_per_seq]),
        ]

        graph_output = [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]),
            helper.make_tensor_value_info("present_key", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("present_value", TensorProto.FLOAT, None),
        ]
    else:
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
            helper.make_tensor_value_info("present_key", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("present_value", TensorProto.FLOAT, None),
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
    block_table_mode=False,
    block_size=64,
    threads=8,
):
    """Benchmark a single GQA configuration. Returns elapsed time in ms."""
    hidden_size = num_heads * head_size
    kv_hidden_size = kv_num_heads * head_size
    packed_head_size = head_size // 2 if bit_width == 4 else head_size

    total_seqlen = past_seq_len + seq_len
    buffer_seq_len = total_seqlen

    sess_options = SessionOptions()
    sess_options.intra_op_num_threads = threads
    # Suppress warning log noise during micro-benchmark loops.
    sess_options.log_severity_level = 3

    np.random.seed(42)
    query = np.random.uniform(-0.5, 0.5, (batch_size, seq_len, hidden_size)).astype(np.float32)
    key = np.random.uniform(-0.5, 0.5, (batch_size, seq_len, kv_hidden_size)).astype(np.float32)
    value = np.random.uniform(-0.5, 0.5, (batch_size, seq_len, kv_hidden_size)).astype(np.float32)
    # seqlens_k represents the number of existing KV tokens before appending current key/value.
    seqlens_k = np.array([past_seq_len] * batch_size, dtype=np.int32)
    total_seq = np.array([total_seqlen], dtype=np.int32)

    if non_quantized:
        onnx_model_str = create_fp32_gqa_graph(
            batch_size,
            seq_len,
            num_heads,
            kv_num_heads,
            head_size,
            buffer_seq_len=buffer_seq_len,
            use_block_table=block_table_mode,
            block_size=block_size,
        )
        sess = InferenceSession(onnx_model_str, sess_options, providers=["CPUExecutionProvider"])

        if block_table_mode:
            max_num_blocks_per_seq = (buffer_seq_len + block_size - 1) // block_size
            num_blocks = batch_size * max_num_blocks_per_seq

            past_k = np.random.uniform(-0.5, 0.5, (num_blocks, block_size, kv_num_heads, head_size)).astype(np.float32)
            past_v = np.random.uniform(-0.5, 0.5, (num_blocks, block_size, kv_num_heads, head_size)).astype(np.float32)

            block_table = np.zeros((batch_size, max_num_blocks_per_seq), dtype=np.int32)
            for b in range(batch_size):
                base = b * max_num_blocks_per_seq
                block_table[b, :] = np.arange(base, base + max_num_blocks_per_seq, dtype=np.int32)

            feeds = {
                "query": query,
                "key": key,
                "value": value,
                "past_key": past_k,
                "past_value": past_v,
                "seqlens_k": seqlens_k,
                "total_sequence_length": total_seq,
                "block_table": block_table,
            }
        else:
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

    if args.block_table and not args.fp32:
        raise ValueError("--block_table is only supported with --fp32 in this benchmark.")

    if args.block_table_only and not args.fp32:
        raise ValueError("--block_table_only requires --fp32 in this benchmark.")

    if args.block_table_only and not args.block_table:
        raise ValueError("--block_table_only requires --block_table.")

    if args.block_table and args.fp32:
        block_cfgs = []
        if not args.decode_only:
            for total_seqlen in [512, 1024, 2048, 4096]:
                block_cfgs.append(
                    {
                        "label": f"Prefill S={total_seqlen} block_table",
                        "batch_size": 1,
                        "seq_len": total_seqlen,
                        "num_heads": 16,
                        "kv_num_heads": 8,
                        "head_size": 128,
                        "quant_type": "PER_TENSOR",
                        "bit_width": 0,
                        "past_seq_len": 0,
                        "block_table_mode": True,
                    }
                )

        if not args.prompt_only:
            for past_seqlen in [512, 1024, 2048, 4096]:
                block_cfgs.append(
                    {
                        "label": f"Decode T={past_seqlen + 1} block_table",
                        "batch_size": 1,
                        "seq_len": 1,
                        "num_heads": 16,
                        "kv_num_heads": 8,
                        "head_size": 128,
                        "quant_type": "PER_TENSOR",
                        "bit_width": 0,
                        "past_seq_len": past_seqlen,
                        "block_table_mode": True,
                    }
                )

        configs.extend(block_cfgs)

    warmup = args.warmup
    repeats = args.repeats
    trials = args.trials
    csv_output = args.csv_output
    threads = args.threads
    csv_rows = []

    # Save and restore env var to avoid side effects on callers
    saved_env = os.environ.get("ORT_GQA_DISABLE_FLASH_ATTENTION")

    kv_mode = "FP32 (non-quantized)" if args.fp32 else "INT8/INT4 quantized"
    print("\nBenchmark: CPU GroupQueryAttention — Flash vs Naive")
    print(f"KV cache: {kv_mode}, Threads: {threads}, Warmup: {warmup}, Repeats: {repeats}, Trials: {trials}")
    def run_config_group(group_title, group_configs):
        if not group_configs:
            return

        print(f"\n{group_title}")
        if trials <= 1:
            print(f"{'Config':<25} {'Naive (ms)':>12} {'Flash (ms)':>12} {'Speedup':>10}")
            print("-" * 62)
        else:
            print(f"{'Config':<25} {'Naive':>9} {'Flash':>9} {'Mean':>8} {'Min':>7} {'Max':>7}")
            print("-" * 74)

        for cfg in group_configs:
            cfg_copy = dict(cfg)
            label = cfg_copy.pop("label")
            cfg_copy["non_quantized"] = args.fp32
            cfg_copy.setdefault("block_table_mode", False)

            naive_runs_ms = []
            flash_runs_ms = []
            speedups = []
            for _ in range(trials):
                # Flash path (default)
                os.environ.pop("ORT_GQA_DISABLE_FLASH_ATTENTION", None)
                flash_ms = benchmark_gqa(**cfg_copy, warmup=warmup, repeats=repeats, threads=threads)

                # Naive path (disabled flash)
                os.environ["ORT_GQA_DISABLE_FLASH_ATTENTION"] = "1"
                naive_ms = benchmark_gqa(**cfg_copy, warmup=warmup, repeats=repeats, threads=threads)

                naive_runs_ms.append(naive_ms)
                flash_runs_ms.append(flash_ms)
                speedups.append(naive_ms / flash_ms if flash_ms > 0 else float("inf"))

            naive_mean_ms = float(np.mean(naive_runs_ms))
            flash_mean_ms = float(np.mean(flash_runs_ms))
            csv_row = {
                "group": group_title,
                "config": label,
                "naive_ms": naive_mean_ms,
                "flash_ms": flash_mean_ms,
            }

            if trials <= 1:
                speedup = speedups[0]
                print(f"{label:<25} {naive_mean_ms:>10.3f}ms {flash_mean_ms:>10.3f}ms {speedup:>8.2f}x")
                csv_row["speedup_mean"] = speedup
                csv_row["speedup_min"] = speedup
                csv_row["speedup_max"] = speedup
            else:
                speedup_mean = float(np.mean(speedups))
                speedup_min = float(np.min(speedups))
                speedup_max = float(np.max(speedups))
                print(
                    f"{label:<25} {naive_mean_ms:>10.3f}ms {flash_mean_ms:>10.3f}ms "
                    f"{speedup_mean:>12.2f}x {speedup_min:>7.2f}x {speedup_max:>7.2f}x"
                )
                csv_row["speedup_mean"] = speedup_mean
                csv_row["speedup_min"] = speedup_min
                csv_row["speedup_max"] = speedup_max

            csv_rows.append(csv_row)

    contiguous_configs = [cfg for cfg in configs if not cfg.get("block_table_mode", False)]
    block_table_configs = [cfg for cfg in configs if cfg.get("block_table_mode", False)]

    if args.block_table_only:
        contiguous_configs = []

    run_config_group("Contiguous cache mode", contiguous_configs)
    run_config_group("Block-table cache mode", block_table_configs)

    if csv_output:
        csv_dir = os.path.dirname(csv_output)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

        with open(csv_output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["group", "config", "naive_ms", "flash_ms", "speedup_mean", "speedup_min", "speedup_max"],
            )
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"\nWrote CSV results to: {csv_output}")

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
    parser.add_argument("--threads", type=int, default=8, help="intra_op_num_threads for CPUExecutionProvider")
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of independent benchmark trials per config (prints mean/min/max speedup when >1).",
    )
    parser.add_argument(
        "--csv_output",
        type=str,
        default="",
        help="Optional path to write benchmark results as CSV.",
    )
    parser.add_argument("--decode_only", action="store_true", help="Only run decode benchmarks")
    parser.add_argument("--prompt_only", action="store_true", help="Only run prompt benchmarks")
    parser.add_argument("--fp32", action="store_true", help="Use non-quantized FP32 KV cache instead of quantized")
    parser.add_argument(
        "--block_table",
        action="store_true",
        help="Include FP32 block_table cache benchmarks (flash vs naive). Requires --fp32.",
    )
    parser.add_argument(
        "--block_table_only",
        action="store_true",
        help="Run only block_table scenarios. Requires --fp32 --block_table.",
    )
    args = parser.parse_args()
    run_benchmarks(args)
