# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Profiling script for SkipLayerNormalization CUDA kernel.

Usage:
  cd onnxruntime/test/python/transformers
  python profile_skip_layer_norm.py

  # Profile with Nsight Systems (timeline analysis) and extract kernel timings:
  nsys profile -o sln_fp16 --export=sqlite python profile_skip_layer_norm.py --mode fp16 --warmup 5 --repeat 100
  python parse_nsys.py sln_fp16.sqlite --nvtx-range benchmark

"""

import argparse
import os
import tempfile
import time

import numpy as np
from onnx import TensorProto, helper, save_model

import onnxruntime as ort

# Optional NVTX support for nsys range markers
try:
    import nvtx

    HAS_NVTX = True
except ImportError:
    HAS_NVTX = False

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


def create_skip_layer_norm_model(batch_size, seq_len, hidden_size, data_type, simplified=False):
    """Create an ONNX model with a single SkipLayerNormalization op."""
    onnx_type = TensorProto.FLOAT16 if data_type == np.float16 else TensorProto.FLOAT

    input_tensor = helper.make_tensor_value_info("INPUT", onnx_type, [batch_size, seq_len, hidden_size])
    skip_tensor = helper.make_tensor_value_info("SKIP", onnx_type, [batch_size, seq_len, hidden_size])
    gamma_tensor = helper.make_tensor_value_info("GAMMA", onnx_type, [hidden_size])
    beta_tensor = helper.make_tensor_value_info("BETA", onnx_type, [hidden_size])
    bias_tensor = helper.make_tensor_value_info("BIAS", onnx_type, [hidden_size])

    output_tensor = helper.make_tensor_value_info("OUTPUT", onnx_type, [batch_size, seq_len, hidden_size])

    op_type = "SkipSimplifiedLayerNormalization" if simplified else "SkipLayerNormalization"
    if simplified:
        inputs = ["INPUT", "SKIP", "GAMMA", "BIAS"]
        input_list = [input_tensor, skip_tensor, gamma_tensor, bias_tensor]
    else:
        inputs = ["INPUT", "SKIP", "GAMMA", "BETA", "BIAS"]
        input_list = [input_tensor, skip_tensor, gamma_tensor, beta_tensor, bias_tensor]

    node = helper.make_node(
        op_type,
        inputs=inputs,
        outputs=["OUTPUT", "", "", ""],
        domain="com.microsoft",
        epsilon=1e-5,
    )

    graph = helper.make_graph([node], "skip_layer_norm_profile", input_list, [output_tensor])

    opset_imports = [
        helper.make_opsetid("", 17),
        helper.make_opsetid("com.microsoft", 1),
    ]

    model = helper.make_model(graph, opset_imports=opset_imports)
    model.ir_version = 7
    return model


def run_profiling(args):
    """Run profiling for SkipLayerNormalization."""
    data_type = np.float16 if args.mode == "fp16" else np.float32

    print(f"\n{'=' * 70}")
    print("SkipLayerNormalization Profiling")
    print(f"{'=' * 70}")
    print(f"Config: batch={args.batch_size}, seq_len={args.seq_len}, hidden_size={args.hidden_size}")
    print(f"        mode={args.mode}, simplified={args.simplified}")
    print(f"        warmup={args.warmup}, repeat={args.repeat}")
    print(f"{'=' * 70}\n")

    model = create_skip_layer_norm_model(
        args.batch_size, args.seq_len, args.hidden_size, data_type, simplified=args.simplified
    )

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        model_path = f.name
        save_model(model, model_path)

    try:
        sess_opt = ort.SessionOptions()
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(model_path, sess_options=sess_opt, providers=["CUDAExecutionProvider"])

        # Create inputs
        np.random.seed(42)
        if args.simplified:
            feeds = {
                "INPUT": np.random.rand(args.batch_size, args.seq_len, args.hidden_size).astype(data_type),
                "SKIP": np.random.rand(args.batch_size, args.seq_len, args.hidden_size).astype(data_type),
                "GAMMA": np.random.rand(args.hidden_size).astype(data_type),
                "BIAS": np.random.rand(args.hidden_size).astype(data_type),
            }
        else:
            feeds = {
                "INPUT": np.random.rand(args.batch_size, args.seq_len, args.hidden_size).astype(data_type),
                "SKIP": np.random.rand(args.batch_size, args.seq_len, args.hidden_size).astype(data_type),
                "GAMMA": np.random.rand(args.hidden_size).astype(data_type),
                "BETA": np.random.rand(args.hidden_size).astype(data_type),
                "BIAS": np.random.rand(args.hidden_size).astype(data_type),
            }

        # Warmup
        with nvtx.annotate("warmup", color="yellow"):
            for _ in range(args.warmup):
                sess.run(None, feeds)

        # Benchmark with NVTX annotation
        with nvtx.annotate("benchmark", color="green"):
            start = time.perf_counter()
            for _ in range(args.repeat):
                sess.run(None, feeds)
            end = time.perf_counter()

        avg_ms = (end - start) * 1000 / args.repeat
        elem_size = 2 if data_type == np.float16 else 4
        total_elements = args.batch_size * args.seq_len * args.hidden_size
        bytes_transferred = 4 * total_elements * elem_size
        throughput_gbps = bytes_transferred / (avg_ms * 1e-3) / 1e9

        print(f"  Average time: {avg_ms:.4f} ms")
        print(f"  Throughput:   {throughput_gbps:.2f} GB/s")

    finally:
        os.unlink(model_path)


def main():
    parser = argparse.ArgumentParser(description="Profile SkipLayerNormalization CUDA kernel")
    parser.add_argument("--mode", choices=["fp16", "fp32"], default="fp16", help="Data type")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--hidden-size", type=int, default=4096, help="Hidden size")
    parser.add_argument("--simplified", action="store_true", help="Use SkipSimplifiedLayerNormalization")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=100, help="Benchmark iterations")

    args = parser.parse_args()
    run_profiling(args)


if __name__ == "__main__":
    main()
