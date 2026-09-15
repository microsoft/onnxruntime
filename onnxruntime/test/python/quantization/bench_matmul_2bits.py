#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Benchmark for MatMulNBits 2-bit dequantization performance on CPU.

This benchmark measures the performance of the multi-threaded
2-bit dequantization path (PR #28589 / issue #28552).
It exercises the MatMulNBits operator with 2-bit quantization
and float zero points on the CPU execution provider.
To compare against a baseline, run this script on two different builds
and compare the reported latencies.

Usage:
    python bench_matmul_2bits.py [--warmup N] [--repeats N] [--threads N]
"""

import argparse
import time

import numpy as np
from onnx import TensorProto, helper, numpy_helper

import onnxruntime as ort


def create_matmul_nbits_model(
    M: int,
    K: int,
    N: int,
    block_size: int,
    bits: int = 2,
    has_zero_point: bool = True,
) -> bytes:
    """
    Creates an ONNX model with a single MatMulNBits node.

    The model structure:
        input A [M, K] (float32) -> MatMulNBits -> output [M, N] (float32)

    With quantized weight B [N, K] packed as 2-bit or 4-bit values.

    Args:
        M: Batch/sequence dimension.
        K: Input features (rows of weight matrix).
        N: Output features (columns of weight matrix).
        block_size: Quantization block size along K.
        bits: Number of quantization bits (2 or 4).
        has_zero_point: Whether to include float zero points.

    Returns:
        Serialized ONNX model bytes.
    """
    k_blocks = (K + block_size - 1) // block_size

    # Input: A [M, K]
    input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [M, K])

    # Output
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [M, N])

    # Weight B: packed values as uint8, shape [N, k_blocks, blob_size]
    elements_per_byte = 8 // bits  # 4 for 2-bit, 2 for 4-bit
    blob_size = block_size // elements_per_byte
    b_data = np.random.randint(0, 256, size=(N, k_blocks, blob_size), dtype=np.uint8)
    b_initializer = numpy_helper.from_array(b_data, name="B")

    # Scales: [N, k_blocks] as float32
    scales_data = np.random.uniform(0.001, 0.1, size=(N, k_blocks)).astype(np.float32)
    scales_initializer = numpy_helper.from_array(scales_data, name="scales")

    initializers = [b_initializer, scales_initializer]
    input_names = ["A", "B", "scales"]

    if has_zero_point:
        # Float zero points: [N, k_blocks] as float32
        zp_data = np.random.uniform(0.0, 3.0, size=(N, k_blocks)).astype(np.float32)
        zp_initializer = numpy_helper.from_array(zp_data, name="zero_points")
        initializers.append(zp_initializer)
        input_names.append("zero_points")

    # MatMulNBits node
    node = helper.make_node(
        "MatMulNBits",
        inputs=input_names,
        outputs=["output"],
        name="MatMulNBits_0",
        domain="com.microsoft",
        bits=bits,
        block_size=block_size,
        K=K,
        N=N,
    )

    graph = helper.make_graph(
        [node],
        "matmul_nbits_2bit_bench",
        [input_a],
        [output],
        initializer=initializers,
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 9

    return model.SerializeToString()


def benchmark_matmul_nbits(
    M: int,
    K: int,
    N: int,
    block_size: int,
    bits: int,
    num_threads: int,
    warmup: int = 5,
    repeats: int = 50,
    has_zero_point: bool = True,
) -> dict:
    """
    Benchmark MatMulNBits with n-bit quantization on CPU.

    Returns:
        Dictionary with timing results.
    """
    model_bytes = create_matmul_nbits_model(M, K, N, block_size, bits, has_zero_point)

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = num_threads
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    session = ort.InferenceSession(
        model_bytes,
        sess_options,
        providers=["CPUExecutionProvider"],
    )

    # Create input
    input_a = np.random.randn(M, K).astype(np.float32)
    feeds = {"A": input_a}

    # Warmup
    for _ in range(warmup):
        session.run(None, feeds)

    # Benchmark
    latencies = []
    for _ in range(repeats):
        start = time.perf_counter()
        session.run(None, feeds)
        end = time.perf_counter()
        latencies.append(end - start)

    latencies_ms = [t * 1000 for t in latencies]
    return {
        "M": M,
        "K": K,
        "N": N,
        "block_size": block_size,
        "bits": bits,
        "threads": num_threads,
        "has_zp": has_zero_point,
        "mean_ms": np.mean(latencies_ms),
        "median_ms": np.median(latencies_ms),
        "min_ms": np.min(latencies_ms),
        "max_ms": np.max(latencies_ms),
        "std_ms": np.std(latencies_ms),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark MatMulNBits 2-bit dequantization on CPU")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--repeats", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Thread counts to benchmark",
    )
    parser.add_argument("--m", type=int, nargs="+", default=[1, 32], help="M dimensions (batch)")
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 4], help="Quantization bits to compare")
    args = parser.parse_args()

    # Typical LLM weight shapes
    configs = [
        # (K, N, block_size) — typical LLM layers
        (4096, 4096, 128),  # hidden projection
        (4096, 11008, 128),  # FFN up/gate
        (11008, 4096, 128),  # FFN down
        # Smaller shapes for quick validation
        (1024, 1024, 128),
        (4096, 4096, 32),
    ]

    print("=" * 110)
    print("MatMulNBits 2-bit vs 4-bit Dequantization Benchmark (float zero points, CPU)")
    print(f"ORT version: {ort.__version__}")
    print(f"Warmup: {args.warmup}, Repeats: {args.repeats}")
    print("=" * 110)
    print()

    header = f"{'Bits':>4} {'M':>5} {'K':>6} {'N':>6} {'BS':>4} {'Thr':>4} {'Mean(ms)':>10} {'Med(ms)':>10} {'Min(ms)':>10} {'Std(ms)':>10}"
    print(header)
    print("-" * len(header))

    results = []
    for k, n, block_size in configs:
        for m in args.m:
            for bits in args.bits:
                for num_threads in args.threads:
                    try:
                        result = benchmark_matmul_nbits(
                            M=m,
                            K=k,
                            N=n,
                            block_size=block_size,
                            bits=bits,
                            num_threads=num_threads,
                            warmup=args.warmup,
                            repeats=args.repeats,
                            has_zero_point=True,
                        )
                        results.append(result)
                        print(
                            f"{result['bits']:>4} {result['M']:>5} {result['K']:>6} {result['N']:>6} "
                            f"{result['block_size']:>4} {result['threads']:>4} "
                            f"{result['mean_ms']:>10.3f} {result['median_ms']:>10.3f} "
                            f"{result['min_ms']:>10.3f} {result['std_ms']:>10.3f}"
                        )
                    except Exception as e:
                        print(f"  FAILED: bits={bits} M={m} K={k} N={n} bs={block_size} threads={num_threads}: {e}")

        print()  # Blank line between config groups

    # Summary: compare 2-bit vs 4-bit and show multi-thread speedup
    print("\n" + "=" * 110)
    print("Speedup Summary")
    print("=" * 110)

    # Multi-thread speedup for 2-bit
    print("\n--- 2-bit: Multi-thread speedup (vs 1 thread) ---")
    header2 = f"{'M':>5} {'K':>6} {'N':>6} {'BS':>4} {'1-thr(ms)':>10} {'best-thr':>9} {'best(ms)':>10} {'Speedup':>8}"
    print(header2)
    print("-" * len(header2))

    for k, n, block_size in configs:
        for m in args.m:
            group = [
                r
                for r in results
                if r["K"] == k and r["N"] == n and r["block_size"] == block_size and r["M"] == m and r["bits"] == 2
            ]
            if not group:
                continue
            single = next((r for r in group if r["threads"] == 1), None)
            if single is None:
                continue
            best = min(group, key=lambda r: r["median_ms"])
            speedup = single["median_ms"] / best["median_ms"] if best["median_ms"] > 0 else 0
            print(
                f"{m:>5} {k:>6} {n:>6} {block_size:>4} "
                f"{single['median_ms']:>10.3f} {best['threads']:>9} "
                f"{best['median_ms']:>10.3f} {speedup:>7.2f}x"
            )

    # 2-bit vs 4-bit comparison (same thread count)
    if 4 in args.bits and 2 in args.bits:
        print("\n--- 2-bit vs 4-bit comparison (same thread count) ---")
        header3 = f"{'M':>5} {'K':>6} {'N':>6} {'BS':>4} {'Thr':>4} {'4-bit(ms)':>10} {'2-bit(ms)':>10} {'Ratio':>8}"
        print(header3)
        print("-" * len(header3))

        for k, n, block_size in configs:
            for m in args.m:
                for num_threads in args.threads:
                    r2 = next(
                        (
                            r
                            for r in results
                            if r["K"] == k
                            and r["N"] == n
                            and r["block_size"] == block_size
                            and r["M"] == m
                            and r["bits"] == 2
                            and r["threads"] == num_threads
                        ),
                        None,
                    )
                    r4 = next(
                        (
                            r
                            for r in results
                            if r["K"] == k
                            and r["N"] == n
                            and r["block_size"] == block_size
                            and r["M"] == m
                            and r["bits"] == 4
                            and r["threads"] == num_threads
                        ),
                        None,
                    )
                    if r2 and r4:
                        ratio = r2["median_ms"] / r4["median_ms"] if r4["median_ms"] > 0 else 0
                        print(
                            f"{m:>5} {k:>6} {n:>6} {block_size:>4} {num_threads:>4} "
                            f"{r4['median_ms']:>10.3f} {r2['median_ms']:>10.3f} {ratio:>7.2f}x"
                        )


if __name__ == "__main__":
    main()
