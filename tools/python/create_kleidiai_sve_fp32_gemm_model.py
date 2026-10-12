#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

"""Create a constant-weight FP32 MatMul model for KleidiAI SVE benchmarking."""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("kleidiai_sve_fp32_gemm/model.onnx"),
        help="Model path; test_data_set_0 is created beside it.",
    )
    parser.add_argument("--m", type=int, default=96, help="Rows in the input matrix (default: 96).")
    parser.add_argument("--n", type=int, default=1024, help="Columns in the weight matrix (default: 1024).")
    parser.add_argument("--k", type=int, default=1024, help="Reduction dimension (default: 1024).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.m, args.n, args.k) <= 0:
        raise ValueError("M, N, and K must all be positive")

    # A short repeating pattern keeps the model deterministic and compressible
    # without making the constant weight an all-zero optimization candidate.
    weight_indices = np.arange(args.k * args.n, dtype=np.int64)
    weights = ((weight_indices % 17) - 8).astype(np.float32).reshape(args.k, args.n) / 16.0

    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["output"], name="KleidiAISveFp32MatMul")],
        "kleidiai_sve_fp32_gemm",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [args.m, args.k])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [args.m, args.n])],
        [numpy_helper.from_array(weights, name="weight")],
    )
    model = helper.make_model(
        graph,
        producer_name="onnxruntime-kleidiai-sve-benchmark",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    # Keep the synthetic asset loadable by older ORT benchmark binaries even
    # when it is generated with a newer onnx package.
    model.ir_version = 8
    helper.set_model_props(
        model,
        {
            "benchmark": "KleidiAI Ops FP32 packed GEMM",
            "shape": f"M={args.m},N={args.n},K={args.k}",
            "sve_256_alignment": (
                f"M multiple of 6: {args.m % 6 == 0}; N multiple of 32 FP32 columns: {args.n % 32 == 0}"
            ),
        },
    )
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, args.output)

    input_indices = np.arange(args.m * args.k, dtype=np.int64)
    input_values = ((input_indices % 23) - 11).astype(np.float32).reshape(args.m, args.k) / 16.0
    input_tensor = numpy_helper.from_array(input_values, name="input")
    test_data_path = args.output.parent / "test_data_set_0" / "input_0.pb"
    test_data_path.parent.mkdir(parents=True, exist_ok=True)
    test_data_path.write_bytes(input_tensor.SerializeToString())

    print(f"Wrote {args.output} ({args.m}x{args.k} @ {args.k}x{args.n})")
    print(f"Wrote {test_data_path}")


if __name__ == "__main__":
    main()
