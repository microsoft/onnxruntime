# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Simple profiling script for GroupQueryAttention with quantized KV cache.

Usage:
  cd /onnxruntime/test/python/transformers
  python profile_gqa.py

  # Profile with Nsight Compute (kernel-level analysis)
  ncu --set full -o gqa_fp16 python profile_gqa.py --mode fp16 --warmup 5 --repeat 1
  ncu --set full -o gqa_int8 python profile_gqa.py --mode int8 --warmup 5 --repeat 1

  # Profile with Nsight Systems (timeline analysis) and extract kernel timings
  nsys profile -o gqa_int8 --export=sqlite python profile_gqa.py --mode int8 --warmup 5 --repeat 10
  python parse_nsys.py gqa_int8.sqlite
"""

import argparse
import os
import time

import torch

try:
    from gqa_test_helper import GroupQueryAttentionConfig, OrtGroupQueryAttention
except ImportError:
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from gqa_test_helper import GroupQueryAttentionConfig, OrtGroupQueryAttention

# Optional NVTX support for nsys range markers
try:
    import nvtx

    HAS_NVTX = True
except ImportError:
    HAS_NVTX = False

    # Dummy context manager when NVTX is not available
    class DummyNvtxRange:
        def __init__(self, name):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class nvtx:  # noqa: N801
        @staticmethod
        def annotate(name, color=None):
            return DummyNvtxRange(name)


def create_gqa_config(
    mode: str = "fp16",
    batch_size: int = 1,
    sequence_length: int = 1,
    past_sequence_length: int = 2048,
    max_sequence_length: int = 4096,
    num_heads: int = 32,
    kv_num_heads: int = 8,
    head_size: int = 128,
    local_window_size: int = -1,
    is_packed_qkv: bool = False,
    do_rotary: bool = True,
    has_head_sink: bool = False,
    device: str = "cuda",
    share_kv_scale: bool = False,
) -> GroupQueryAttentionConfig:
    """Create a GQA config based on the mode."""
    if mode == "fp16":
        k_quant_type = "NONE"
        v_quant_type = "NONE"
        kv_cache_type = "float16"
        dtype = torch.float16
    elif mode == "bf16":
        k_quant_type = "NONE"
        v_quant_type = "NONE"
        kv_cache_type = "bfloat16"
        dtype = torch.bfloat16
    elif mode == "int8":
        k_quant_type = "PER_TENSOR"
        v_quant_type = "PER_TENSOR"
        kv_cache_type = "int8"
        dtype = torch.float16
    elif mode == "int4":
        k_quant_type = "PER_CHANNEL"
        v_quant_type = "PER_CHANNEL"
        kv_cache_type = "int4"
        dtype = torch.float16
    else:
        raise ValueError(f"Unknown mode: {mode}")

    config = GroupQueryAttentionConfig(
        batch_size=batch_size,
        sequence_length=sequence_length,
        max_sequence_length=max_sequence_length,
        past_sequence_length=past_sequence_length,
        num_heads=num_heads,
        kv_num_heads=kv_num_heads,
        head_size=head_size,
        local_window_size=local_window_size,
        do_rotary=do_rotary,
        rotary_interleaved=False,
        dtype=dtype,
        is_packed_qkv=is_packed_qkv,
        use_smooth_softmax=False,
        has_head_sink=has_head_sink,
        device=device,
        k_quant_type=k_quant_type,
        v_quant_type=v_quant_type,
        kv_cache_type=kv_cache_type,
        share_kv_scale=share_kv_scale,
    )
    return config


def benchmark_gqa(config: GroupQueryAttentionConfig, warmup: int = 50, repeat: int = 100, mode: str = ""):
    """Run benchmark and return average time in ms."""
    obj = OrtGroupQueryAttention(config)

    # Warmup phase with NVTX annotation
    with nvtx.annotate(f"warmup_{mode}", color="yellow"):
        for _ in range(warmup):
            obj.infer()
        torch.cuda.synchronize()

    # Benchmark phase with NVTX annotation
    with nvtx.annotate(f"benchmark_{mode}", color="green"):
        start = time.perf_counter()
        for _ in range(repeat):
            obj.infer()
        torch.cuda.synchronize()
        end = time.perf_counter()

    avg_ms = (end - start) * 1000 / repeat
    return avg_ms


def run_comparison(args):
    """Compare FP16/BF16 vs quantized performance."""
    # Auto-adjust max_sequence_length to be at least total_sequence_length
    total_sequence_length = args.past_sequence_length + args.sequence_length
    if args.max_sequence_length < total_sequence_length:
        args.max_sequence_length = total_sequence_length
        print(f"Note: max_sequence_length auto-adjusted to {args.max_sequence_length}")

    print(f"\n{'=' * 70}")
    print("GQA Performance Comparison")
    print(f"{'=' * 70}")
    print(f"Config: batch={args.batch_size}, seq_len={args.sequence_length}, past_seq={args.past_sequence_length}")
    print(f"        num_heads={args.num_heads}, kv_heads={args.kv_num_heads}, head_size={args.head_size}")
    print(f"        packed_qkv={args.is_packed_qkv}, rotary={not args.no_rotary}, head_sink={args.head_sink}")
    print(f"        warmup={args.warmup}, repeat={args.repeat}")
    print(f"{'=' * 70}\n")

    modes = ["fp16", "bf16", "int8", "int4"] if args.mode == "all" else [args.mode]
    results = {}

    for mode in modes:
        config = create_gqa_config(
            mode=mode,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            past_sequence_length=args.past_sequence_length,
            max_sequence_length=args.max_sequence_length,
            num_heads=args.num_heads,
            kv_num_heads=args.kv_num_heads,
            head_size=args.head_size,
            local_window_size=args.local_window_size,
            is_packed_qkv=args.is_packed_qkv,
            do_rotary=not args.no_rotary,
            has_head_sink=args.head_sink,
            share_kv_scale=args.share_kv_scale,
        )
        avg_ms = benchmark_gqa(config, warmup=args.warmup, repeat=args.repeat, mode=mode)
        results[mode] = avg_ms
        print(f"  {mode.upper():6s} (dtype={config.dtype}): {avg_ms:.4f} ms")

    # Print comparison if we have baseline
    baseline = "fp16" if "fp16" in results else ("bf16" if "bf16" in results else None)
    if baseline and len(results) > 1:
        print(f"\n  Relative to {baseline.upper()}:")
        for mode, ms in results.items():
            if mode != baseline:
                ratio = ms / results[baseline]
                print(f"    {mode.upper()}: {ratio:.2f}x slower")


def main():
    parser = argparse.ArgumentParser(description="Profile GQA with quantized KV cache")
    parser.add_argument(
        "--mode", choices=["fp16", "bf16", "int8", "int4", "all"], default="all", help="Quantization mode to test"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--sequence-length", type=int, default=1, help="Query sequence length (1 for token generation)")
    parser.add_argument("--past-sequence-length", type=int, default=2048, help="Past KV cache sequence length")
    parser.add_argument("--max-sequence-length", type=int, default=4096, help="Max sequence length for KV cache buffer")
    parser.add_argument("--num-heads", type=int, default=32, help="Number of query heads")
    parser.add_argument("--kv-num-heads", type=int, default=8, help="Number of KV heads")
    parser.add_argument("--head-size", type=int, default=128, help="Head dimension")
    parser.add_argument(
        "--local-window-size",
        type=int,
        default=-1,
        help="Local attention window size (-1 disables sliding window, e.g. gpt-oss uses 128)",
    )
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--is-packed-qkv", action="store_true", help="Use packed QKV")
    parser.add_argument("--head-sink", action="store_true", help="Add a head_sink input")

    parser.add_argument("--no-rotary", action="store_true", help="Disable rotary embeddings")
    parser.add_argument("--share-kv-scale", action="store_true", help="Share KV scale tensor for XQA")

    args = parser.parse_args()

    # Check CUDA
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    major, minor = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name()} (SM{major}{minor})")

    with torch.cuda.stream(torch.cuda.Stream()), torch.no_grad():
        run_comparison(args)


if __name__ == "__main__":
    main()
