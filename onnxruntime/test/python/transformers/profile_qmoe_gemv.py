# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Profiling script for the CUDA QMoE GEMV decode path.

Usage:
  python profile_qmoe_gemv.py --case m1_top2_fp16_128x256 --warmup 5 --repeat 100

  nsys profile -t cuda,nvtx -o qmoe_gemv --export=sqlite \
      python profile_qmoe_gemv.py --case m1_top2_fp16_128x256 --warmup 5 --repeat 100
  python parse_nsys.py qmoe_gemv.sqlite --nvtx-range benchmark
"""

import argparse
import json
import os

import torch
from test_qmoe_cuda import (
    _QMOE_GEMV_BENCHMARK_RESULT_PREFIX,
    _qmoe_gemv_benchmark_case,
    _qmoe_gemv_benchmark_cases,
    run_qmoe_gemv_benchmark,
)


def _custom_case_from_args(args):
    case = dict(_qmoe_gemv_benchmark_case(args.case))
    custom_fields = {
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "onnx_dtype": args.dtype,
        "quant_bits": args.quant_bits,
        "block_size": args.block_size,
    }
    case.update({key: value for key, value in custom_fields.items() if value is not None})

    if any(value is not None for value in custom_fields.values()):
        case["name"] = (
            f"custom_m{case['batch_size'] * case['sequence_length']}_top{case['top_k']}_"
            f"{case['onnx_dtype'].lower()}_{case['hidden_size']}x{case['intermediate_size']}_"
            f"e{case['num_experts']}_int{case.get('quant_bits', 4)}_b{case.get('block_size', 0)}"
        )

    return case


def main():
    parser = argparse.ArgumentParser(description="Profile CUDA QMoE GEMV decode")
    parser.add_argument("--case", default="m1_top2_fp16_128x256", help="Benchmark case name")
    parser.add_argument("--list-cases", action="store_true", help="List available benchmark case names and exit")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--sequence-length", type=int, help="Override sequence length")
    parser.add_argument("--hidden-size", type=int, help="Override hidden size")
    parser.add_argument("--intermediate-size", type=int, help="Override intermediate size")
    parser.add_argument("--num-experts", type=int, help="Override number of experts")
    parser.add_argument("--top-k", type=int, help="Override top-k experts per token")
    parser.add_argument("--dtype", choices=["FLOAT16", "BFLOAT16"], help="Override ONNX dtype")
    parser.add_argument("--quant-bits", type=int, choices=[4, 8], help="Override QMoE integer weight bits")
    parser.add_argument("--block-size", type=int, choices=[0, 32, 64, 128], help="Override QMoE INT block size")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations before the benchmark NVTX range")
    parser.add_argument("--repeat", type=int, default=100, help="Benchmark iterations")
    parser.add_argument(
        "--disable-gemv",
        action="store_true",
        help="Run the grouped GEMM fallback by setting ORT_DISABLE_MOE_GEMV=1 before session creation",
    )
    parser.add_argument(
        "--splitk2-swiglu",
        action="store_true",
        help="Deprecated compatibility flag; split-K2 two-pass FC1 SwiGLU GEMV is enabled by default when supported",
    )
    parser.add_argument(
        "--disable-splitk2-swiglu",
        action="store_true",
        help="Disable split-K2 two-pass FC1 SwiGLU GEMV by setting ORT_DISABLE_MOE_GEMV_SPLITK2_SWIGLU=1",
    )
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="Wrap the measured loop in an NVTX range named 'benchmark'",
    )
    args = parser.parse_args()

    if args.list_cases:
        for case in _qmoe_gemv_benchmark_cases():
            print(case["name"])
        return

    case = _custom_case_from_args(args)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for QMoE GEMV profiling")

    os.environ["ORT_QMOE_GEMV_BENCHMARK_REPEATS"] = str(max(1, args.repeat))
    os.environ["ORT_QMOE_GEMV_BENCHMARK_WARMUP"] = str(max(0, args.warmup))
    if args.nvtx:
        os.environ["ORT_QMOE_GEMV_BENCHMARK_NVTX"] = "1"
    else:
        os.environ.pop("ORT_QMOE_GEMV_BENCHMARK_NVTX", None)

    if args.disable_gemv:
        os.environ["ORT_DISABLE_MOE_GEMV"] = "1"
    else:
        os.environ.pop("ORT_DISABLE_MOE_GEMV", None)

    if args.disable_splitk2_swiglu:
        os.environ["ORT_DISABLE_MOE_GEMV_SPLITK2_SWIGLU"] = "1"
    else:
        os.environ.pop("ORT_DISABLE_MOE_GEMV_SPLITK2_SWIGLU", None)
    os.environ.pop("ORT_MOE_GEMV_SPLITK2_SWIGLU", None)

    result = run_qmoe_gemv_benchmark(case)
    if result["has_invalid_output"]:
        raise RuntimeError("QMoE GEMV profiling produced NaN or Inf output")

    print(_QMOE_GEMV_BENCHMARK_RESULT_PREFIX + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
