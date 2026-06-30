# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Microbenchmark for the fused MXFP4 GEMV (W4A16) decode path.

Reuses the FP4 QMoE test helpers to build a single-MoE-layer ONNX model at the
gpt-oss-20b decode shape (hidden=inter=2880, E=32, top_k=4, 1 token) and times
steady-state Run() latency. Run with:

    ORT_ENABLE_FP4_GEMV=1 ORT_FP4_GEMV_AUTOTUNE=1 ORT_FP4_GEMV_AUTOTUNE_LOG=1 \
      python bench_fp4_gemv_autotune.py

The autotune candidate log lines (one per cfg per FC) are emitted by ORT at
WARNING severity; set --log to surface them.
"""

import argparse
import time

import numpy
import onnx
import torch
from cuda_plugin_ep_helper import resolve_cuda_plugin_ep
from onnx import TensorProto

# Reuse the model/quant helpers from the unit-test module.
from test_qmoe_fp4_cuda import (
    create_fp4_moe_onnx_graph,
    quantize_weight_to_mxfp4,
)

import onnxruntime

device = torch.device("cuda:0")


def build_session(hidden, inter, num_experts, top_k, num_tokens, onnx_dtype, use_swiglu, log):
    torch.manual_seed(42)
    numpy.random.seed(42)
    torch_dtype = torch.float16 if onnx_dtype == TensorProto.FLOAT16 else torch.bfloat16
    onnx_elem = onnx_dtype

    fc1_n = 2 * inter if use_swiglu else inter
    fc1_k = hidden
    fc2_n = hidden
    fc2_k = inter

    fc1_packed, fc1_bs, fc1_gs = [], [], []
    fc2_packed, fc2_bs, fc2_gs = [], [], []
    for _ in range(num_experts):
        w1 = torch.randn(fc1_n, fc1_k, device=device) * 0.1
        p1, b1, g1, _ = quantize_weight_to_mxfp4(w1, 32)
        fc1_packed.append(p1)
        fc1_bs.append(b1)
        fc1_gs.append(torch.tensor(g1, dtype=torch.float32))
        w2 = torch.randn(fc2_n, fc2_k, device=device) * 0.1
        p2, b2, g2, _ = quantize_weight_to_mxfp4(w2, 32)
        fc2_packed.append(p2)
        fc2_bs.append(b2)
        fc2_gs.append(torch.tensor(g2, dtype=torch.float32))

    fc1_weights = torch.stack(fc1_packed, dim=0)
    fc2_weights = torch.stack(fc2_packed, dim=0)
    fc1_block_scales = torch.stack(fc1_bs, dim=0)
    fc2_block_scales = torch.stack(fc2_bs, dim=0)
    fc1_global_scale = torch.stack(fc1_gs)
    fc2_global_scale = torch.stack(fc2_gs)

    onnx_model = create_fp4_moe_onnx_graph(
        num_tokens=num_tokens,
        hidden_size=hidden,
        inter_size=inter,
        num_experts=num_experts,
        top_k=top_k,
        onnx_dtype=onnx_elem,
        fc1_weights=fc1_weights,
        fc2_weights=fc2_weights,
        fc1_block_scales=fc1_block_scales,
        fc1_global_scale=fc1_global_scale,
        fc2_block_scales=fc2_block_scales,
        fc2_global_scale=fc2_global_scale,
        use_swiglu=use_swiglu,
    )

    # create_fp4_moe_onnx_graph stamps the default onnx opset (27 for onnx>=1.22),
    # which ORT rejects (official ai.onnx support is up to opset 26). The MoE op is
    # a com.microsoft contrib op, so clamping the ai.onnx opset to 26 is safe.
    model_proto = onnx.load_model_from_string(onnx_model)
    for op in model_proto.opset_import:
        if op.domain in ("", "ai.onnx") and op.version > 26:
            op.version = 26
    onnx_model = model_proto.SerializeToString()

    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    if log:
        # The autotune candidate lines are emitted via LOGS_DEFAULT(WARNING), which uses the
        # environment's default logger -- set both the global and session severity to WARNING.
        onnxruntime.set_default_logger_severity(2)
        opts.log_severity_level = 2
    session = onnxruntime.InferenceSession(
        onnx_model, opts, providers=[resolve_cuda_plugin_ep("CUDAExecutionProvider")]
    )

    input_tensor = torch.randn(num_tokens, hidden, device=device, dtype=torch_dtype)
    router_logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch_dtype)
    output_tensor = torch.zeros(num_tokens, hidden, device=device, dtype=torch_dtype)

    iob = session.io_binding()
    iob.bind_input("input", "cuda", 0, onnx_elem, input_tensor.shape, input_tensor.data_ptr())
    iob.bind_input("router_probs", "cuda", 0, onnx_elem, router_logits.shape, router_logits.data_ptr())
    iob.bind_output("output", "cuda", 0, onnx_elem, output_tensor.shape, output_tensor.data_ptr())
    iob.synchronize_inputs()
    return session, iob


def bench(session, iob, warmup, iters):
    for _ in range(warmup):
        session.run_with_iobinding(iob)
    iob.synchronize_outputs()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        session.run_with_iobinding(iob)
    iob.synchronize_outputs()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1e3  # ms/run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=2880)
    ap.add_argument("--inter", type=int, default=2880)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--log", action="store_true")
    args = ap.parse_args()

    onnx_dtype = TensorProto.FLOAT16 if args.dtype == "fp16" else TensorProto.BFLOAT16
    cap = torch.cuda.get_device_capability()
    print(f"device cap sm{cap[0]}{cap[1]}  build_info={onnxruntime.get_build_info()}")
    print(
        f"shape hidden={args.hidden} inter={args.inter} E={args.experts} top_k={args.top_k} "
        f"tokens={args.tokens} dtype={args.dtype}  (fc1 n={2 * args.inter} k={args.hidden}; "
        f"fc2 n={args.hidden} k={args.inter}; expanded={args.tokens * args.top_k})"
    )
    session, iob = build_session(
        args.hidden, args.inter, args.experts, args.top_k, args.tokens, onnx_dtype, True, args.log
    )
    best = float("inf")
    for r in range(args.reps):
        ms = bench(session, iob, args.warmup if r == 0 else 2, args.iters)
        best = min(best, ms)
        print(f"  rep{r}: {ms * 1e3:.2f} us/run  ({1e3 / ms:.1f} runs/ms)")
    print(f"BEST: {best * 1e3:.2f} us/run  ({1e3 / best:.1f} runs/ms)")


if __name__ == "__main__":
    main()
