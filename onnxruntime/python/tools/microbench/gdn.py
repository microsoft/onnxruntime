# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Microbenchmark for com.microsoft::GatedDeltaNet.

Builds a single-node model per shape and times it with IOBinding, so every tensor is
device-resident and the timed region contains only Run(). ``--op LinearAttention`` runs the
padded (B, T, H*D) operator over the same logical problem for comparison.

The default shapes are the Qwen3.8-27B linear-attention geometry (16 query/key heads,
48 value heads, head size 128). One call is all 48 value heads of one layer; the model has
48 such layers, so the reported ``x48`` column is the cost across a whole forward pass.

Example:

    python gdn.py --op GatedDeltaNet
    python gdn.py --op LinearAttention
    python gdn.py --op GatedDeltaNet --batch-size 4 --seq-lens 1,4,1024

At decode lengths the measured time is dominated by ORT session overhead rather than the
kernel. Use Nsight Compute for kernel-only numbers:

    ncu -k regex:"GatedDeltaNet" --metrics gpu__time_duration.sum python gdn.py
"""

import argparse
import statistics
import time
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import TensorProto, helper

import onnxruntime as ort

# Deliberately not importing benchmark.py: it pulls in torch for its tensor-based IOBinding
# helper, and this script binds OrtValues directly, so it stays runnable in a plain
# onnxruntime environment.
_PROVIDER_MAP = {"cuda": "CUDAExecutionProvider", "cpu": "CPUExecutionProvider"}


def provider_name(name):
    return _PROVIDER_MAP[name]


def get_default_provider():
    if "CUDAExecutionProvider" in ort.get_available_providers():
        return "CUDAExecutionProvider"
    return "CPUExecutionProvider"


# Qwen3.8-27B linear attention geometry.
DEFAULT_NUM_HEADS_Q = 16
DEFAULT_NUM_HEADS_V = 48
DEFAULT_HEAD_SIZE = 128
DEFAULT_NUM_LAYERS = 48


@dataclass
class OpParam:
    batch_size: int
    seq_len: int
    num_heads_q: int  # also the key head count
    num_heads_v: int
    head_size_qk: int
    head_size_v: int
    data_type: type


def _elem_type(data_type):
    return TensorProto.FLOAT16 if data_type == np.float16 else TensorProto.FLOAT


def _make_model(nodes, inputs, outputs):
    graph = helper.make_graph(nodes, "microbench", inputs, outputs)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=False)
    return model.SerializeToString()


def build_gated_delta_net(p: OpParam, ragged: bool):
    """Token-major inputs, V-major float32 state."""
    total = p.batch_size * p.seq_len
    elem = _elem_type(p.data_type)
    out_heads = max(p.num_heads_q, p.num_heads_v)

    inputs = [
        helper.make_tensor_value_info("query", elem, [total, p.num_heads_q, p.head_size_qk]),
        helper.make_tensor_value_info("key", elem, [total, p.num_heads_q, p.head_size_qk]),
        helper.make_tensor_value_info("value", elem, [total, p.num_heads_v, p.head_size_v]),
    ]
    names = ["query", "key", "value"]
    if ragged:
        inputs.append(helper.make_tensor_value_info("cu_seqlens", TensorProto.INT32, [p.batch_size + 1]))
        names.append("cu_seqlens")
    else:
        names.append("")
    inputs += [
        helper.make_tensor_value_info("decay", TensorProto.FLOAT, [total, p.num_heads_v]),
        helper.make_tensor_value_info("beta", TensorProto.FLOAT, [total, p.num_heads_v]),
        helper.make_tensor_value_info(
            "initial_state",
            TensorProto.FLOAT,
            [p.batch_size, p.num_heads_v, p.head_size_v, p.head_size_qk],
        ),
    ]
    names += ["decay", "beta", "initial_state"]

    outputs = [
        helper.make_tensor_value_info("output", elem, [total, out_heads, p.head_size_v]),
        helper.make_tensor_value_info(
            "final_state",
            TensorProto.FLOAT,
            [p.batch_size, p.num_heads_v, p.head_size_v, p.head_size_qk],
        ),
    ]
    node = helper.make_node(
        "GatedDeltaNet",
        names,
        ["output", "final_state"],
        domain="com.microsoft",
        update_rule="gated_delta",
        scale=1.0,
    )
    return _make_model([node], inputs, outputs)


def build_linear_attention(p: OpParam, state_window: int):
    """Padded (B, T, H*D) inputs with the state in the input dtype."""
    elem = _elem_type(p.data_type)
    b, t = p.batch_size, p.seq_len
    state_dims = ([state_window] if state_window > 0 else []) + [
        b,
        p.num_heads_v,
        p.head_size_qk,
        p.head_size_v,
    ]
    out_hidden = max(p.num_heads_q, p.num_heads_v) * p.head_size_v

    inputs = [
        helper.make_tensor_value_info("query", elem, [b, t, p.num_heads_q * p.head_size_qk]),
        helper.make_tensor_value_info("key", elem, [b, t, p.num_heads_q * p.head_size_qk]),
        helper.make_tensor_value_info("value", elem, [b, t, p.num_heads_v * p.head_size_v]),
        helper.make_tensor_value_info("past_state", elem, state_dims),
        helper.make_tensor_value_info("decay", elem, [b, t, p.num_heads_v]),
        helper.make_tensor_value_info("beta", elem, [b, t, p.num_heads_v]),
    ]
    outputs = [
        helper.make_tensor_value_info("output", elem, [b, t, out_hidden]),
        helper.make_tensor_value_info("present_state", elem, state_dims),
    ]
    node = helper.make_node(
        "LinearAttention",
        ["query", "key", "value", "past_state", "decay", "beta"],
        ["output", "present_state"],
        domain="com.microsoft",
        update_rule="gated_delta",
        q_num_heads=p.num_heads_q,
        kv_num_heads=p.num_heads_v,
        scale=1.0,
        state_window=state_window,
    )
    return _make_model([node], inputs, outputs)


def create_inputs(sess, p: OpParam, seed: int = 0):
    rng = np.random.default_rng(seed)
    feeds = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        if inp.name == "cu_seqlens":
            feeds[inp.name] = (np.arange(p.batch_size + 1, dtype=np.int32) * p.seq_len).astype(np.int32)
            continue
        np_dtype = np.float16 if "float16" in inp.type else np.float32
        if inp.name == "decay":
            # Log-space decay must be non-positive for the recurrence to contract.
            arr = (-0.05 * (rng.random(shape) + 1.2)).astype(np_dtype)
        elif inp.name == "beta":
            arr = (0.5 + 0.025 * rng.standard_normal(shape)).astype(np_dtype)
        elif inp.name in ("past_state", "initial_state"):
            arr = (0.05 * rng.standard_normal(shape)).astype(np_dtype)
        elif inp.name == "key":
            # The delta family requires L2-normalized keys; without them (I + M) is
            # arbitrarily ill-conditioned and the recurrence diverges.
            arr = rng.standard_normal(shape).astype(np.float32)
            flat = arr.reshape(-1, p.head_size_qk)
            flat /= np.linalg.norm(flat, axis=-1, keepdims=True) + 1e-12
            arr = flat.reshape(shape).astype(np_dtype)
        else:
            arr = (0.5 * rng.standard_normal(shape)).astype(np_dtype)
        feeds[inp.name] = arr
    return feeds


def run_case(model_bytes, p: OpParam, provider, device_id, iters, warmup):
    sess_opt = ort.SessionOptions()
    sess_opt.log_severity_level = 3
    # Keep the single node intact so the measurement is the operator, not a fused graph.
    sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(model_bytes, sess_opt, providers=[(provider, {"device_id": device_id})])

    feeds = create_inputs(sess, p)
    binding = sess.io_binding()
    device = "cuda" if provider == "CUDAExecutionProvider" else "cpu"
    keep = []
    for inp in sess.get_inputs():
        ov = ort.OrtValue.ortvalue_from_numpy(feeds[inp.name], device, device_id)
        keep.append(ov)
        binding.bind_ortvalue_input(inp.name, ov)
    for out in sess.get_outputs():
        shape = [d if isinstance(d, int) else 1 for d in out.shape]
        np_dtype = np.float16 if "float16" in out.type else np.float32
        ov = ort.OrtValue.ortvalue_from_numpy(np.zeros(shape, np_dtype), device, device_id)
        keep.append(ov)
        binding.bind_ortvalue_output(out.name, ov)

    run_opt = ort.RunOptions()
    for _ in range(warmup):
        sess.run_with_iobinding(binding, run_opt)
    binding.synchronize_outputs()

    times_us = []
    for _ in range(iters):
        start = time.perf_counter()
        sess.run_with_iobinding(binding, run_opt)
        binding.synchronize_outputs()
        times_us.append((time.perf_counter() - start) * 1e6)
    # Median, not mean: the distribution has a tail that a mean would misreport.
    return statistics.median(times_us)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--provider",
        required=False,
        type=str,
        choices=["cuda", "cpu", None],
        default=None,
        help="Execution provider. Defaults to cuda when available, else cpu.",
    )
    parser.add_argument(
        "--precision",
        required=False,
        type=str,
        choices=["fp16", "fp32"],
        default="fp16",
        help="Number format for query/key/value. State and gates are always float32.",
    )
    parser.add_argument(
        "--op",
        choices=["GatedDeltaNet", "LinearAttention"],
        default="GatedDeltaNet",
        help="Operator to benchmark.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--seq-lens",
        type=str,
        default="1,2,4,256,1024,2048,8192",
        help="Comma-separated sequence lengths.",
    )
    parser.add_argument("--num-heads-q", type=int, default=DEFAULT_NUM_HEADS_Q)
    parser.add_argument("--num-heads-v", type=int, default=DEFAULT_NUM_HEADS_V)
    parser.add_argument("--head-size", type=int, default=DEFAULT_HEAD_SIZE)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--state-window", type=int, default=0, help="LinearAttention only.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    provider = get_default_provider() if args.provider is None else provider_name(args.provider)
    if args.op == "GatedDeltaNet" and provider != "CUDAExecutionProvider":
        parser.error("GatedDeltaNet is CUDA-only; select the CUDA execution provider.")
    data_type = np.float16 if args.precision == "fp16" else np.float32

    print(f"provider: {provider}, precision: {args.precision}, op: {args.op}")
    print(f"{'shape':<16}{'op':<18}{'us':>10}{'us/token':>11}{f'x{args.num_layers} layers ms':>18}")

    for seq_len in [int(s) for s in args.seq_lens.split(",")]:
        p = OpParam(
            batch_size=args.batch_size,
            seq_len=seq_len,
            num_heads_q=args.num_heads_q,
            num_heads_v=args.num_heads_v,
            head_size_qk=args.head_size,
            head_size_v=args.head_size,
            data_type=data_type,
        )
        if args.op == "GatedDeltaNet":
            model_bytes = build_gated_delta_net(p, ragged=True)
        else:
            model_bytes = build_linear_attention(p, args.state_window)

        us = run_case(model_bytes, p, provider, args.device_id, args.iters, args.warmup)
        tokens = p.batch_size * p.seq_len
        tag = f"B{p.batch_size}_T{seq_len}"
        print(f"{tag:<16}{args.op:<18}{us:>10.1f}{us / tokens:>11.3f}{us * args.num_layers / 1000:>18.2f}")


if __name__ == "__main__":
    main()
