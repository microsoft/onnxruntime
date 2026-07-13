# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Benchmark ONNX Attention (opset 23/24, CUDA EP) against contrib GroupQueryAttention
for single-stream decode (issue #28352 baseline recipe).

Arms, all decode-shaped (S_q = 1, causal, no RoPE, no mask, no softcap):

  gqa_xqa       GroupQueryAttention, shared KV buffer, XQA decode kernel.
  gqa_flash     GroupQueryAttention pinned to the Flash Attention kernels
                (XQA and cuDNN SDPA disabled).
  gqa_cudnn     GroupQueryAttention with XQA disabled and cuDNN SDPA enabled
                (GQA's default non-XQA dispatch on SM90+).
  attn_past     ONNX Attention with past_key/past_value inputs and present_*
                outputs (LaunchConcatNewToPastKV copies the past each step).
  attn_scatter  ONNX Attention opset-24 external cache: in-place TensorScatter
                appends the new token, Attention reads the full cache with
                nonpad_kv_seqlen. present_* outputs are omitted (requesting
                them adds a full-cache copy).

Measurement notes:
  - attn_scatter includes the TensorScatter nodes so both ops pay their cache
    append; --attention-only times the bare Attention node instead.
  - ONNX arms disable memory-efficient attention so a Flash ineligibility
    surfaces as the (obviously slow) unfused kernel, never a silent MEA flip.
  - ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO=1 prints the resolved GQA backend.
    ONNX Attention has no such print; verify it from kernel names in an
    Nsight Systems capture (--profile).
  - --max-seq-len (KV buffer length) directly affects attn_scatter latency:
    the valid length is a device tensor, so the host sizes the Flash split-KV
    launch from the buffer length (attention.cc, nonpad_kv_seqlen path). GQA
    sizes splits from its CPU-side total_sequence_length input.

Usage:
  python benchmark_onnx_attention_vs_gqa.py --dtype float16 --csv results.csv
  python benchmark_onnx_attention_vs_gqa.py --sanity
  nsys profile python benchmark_onnx_attention_vs_gqa.py \
      --profile --arms attn_scatter --past-seq-len 2048
"""

import argparse
import contextlib
import csv
import os
import subprocess
import sys

import torch
from gqa_test_helper import GroupQueryAttentionConfig, create_gqa_ort_session
from onnx import TensorProto, helper
from onnxruntime.transformers.io_binding_helper import CudaSession

import onnxruntime
from onnxruntime import InferenceSession

try:
    from triton.testing import do_bench

    TIMER = "triton.testing.do_bench(warmup=15ms, rep=100ms)"
except ImportError:
    do_bench = None
    TIMER = "cuda-event fallback (20 warmup iters, 100 timed iters, mean)"

ALL_ARMS = ["gqa_xqa", "gqa_flash", "gqa_cudnn", "attn_past", "attn_scatter"]

TORCH_DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16}

# Kernel-dispatch pins per arm. ORT reads these at session creation, so they
# are scoped around InferenceSession construction (see build_arm). cuDNN SDPA
# must be pinned off for gqa_flash: SM90+ auto-enables it when XQA is off.
ARM_ENV = {
    "gqa_xqa": {"ORT_ENABLE_XQA": "1"},
    "gqa_flash": {"ORT_ENABLE_XQA": "0", "ORT_ENABLE_CUDNN_FLASH_ATTENTION": "0"},
    "gqa_cudnn": {"ORT_ENABLE_XQA": "0", "ORT_ENABLE_CUDNN_FLASH_ATTENTION": "1"},
    "attn_past": {"ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION": "1"},
    "attn_scatter": {"ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION": "1"},
}


@contextlib.contextmanager
def scoped_env(env: dict):
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# #################################################################################################
#  ONNX Attention graph builders
# #################################################################################################


def create_attention_past_model(batch, q_heads, kv_heads, head_size, past_seq_len, onnx_dtype):
    """ONNX Attention (opset 23) decode graph with past_key/past_value inputs."""
    q_hidden = q_heads * head_size
    kv_hidden = kv_heads * head_size
    total = past_seq_len + 1

    node = helper.make_node(
        "Attention",
        inputs=["query", "key", "value", "", "past_key", "past_value"],
        outputs=["output", "present_key", "present_value"],
        name="Attention_0",
        is_causal=1,
        q_num_heads=q_heads,
        kv_num_heads=kv_heads,
        softcap=0.0,
        qk_matmul_output_mode=0,
        domain="",
    )
    graph_inputs = [
        helper.make_tensor_value_info("query", onnx_dtype, [batch, 1, q_hidden]),
        helper.make_tensor_value_info("key", onnx_dtype, [batch, 1, kv_hidden]),
        helper.make_tensor_value_info("value", onnx_dtype, [batch, 1, kv_hidden]),
        helper.make_tensor_value_info("past_key", onnx_dtype, [batch, kv_heads, past_seq_len, head_size]),
        helper.make_tensor_value_info("past_value", onnx_dtype, [batch, kv_heads, past_seq_len, head_size]),
    ]
    graph_outputs = [
        helper.make_tensor_value_info("output", onnx_dtype, [batch, 1, q_hidden]),
        helper.make_tensor_value_info("present_key", onnx_dtype, [batch, kv_heads, total, head_size]),
        helper.make_tensor_value_info("present_value", onnx_dtype, [batch, kv_heads, total, head_size]),
    ]
    graph = helper.make_graph([node], "AttentionPast_Graph", graph_inputs, graph_outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])
    return model.SerializeToString()


def create_attention_scatter_model(
    batch, q_heads, kv_heads, head_size, buffer_len, onnx_dtype, attention_only, cache_4d=False
):
    """ONNX Attention (opset 24) external-cache decode graph.

    TensorScatter writes the new K/V token into the cache in place (updated_*
    outputs are bound to the same buffers as the *_cache inputs), then
    Attention consumes the full cache with nonpad_kv_seqlen. No present_*
    outputs. attention_only=True drops the TensorScatter nodes.

    cache_4d=False: 3D BSNH cache [B, S_buf, H_kv*D], scatter axis=1.
    cache_4d=True: 4D BNSH cache [B, H_kv, S_buf, D], scatter axis=2,
    Q/output [B, H_q, 1, D].
    """
    q_hidden = q_heads * head_size
    kv_hidden = kv_heads * head_size

    scatter_axis = 2 if cache_4d else 1
    if cache_4d:
        cache_shape = [batch, kv_heads, buffer_len, head_size]
        new_kv_shape = [batch, kv_heads, 1, head_size]
        q_shape = [batch, q_heads, 1, head_size]
        out_shape = [batch, q_heads, 1, head_size]
    else:
        cache_shape = [batch, buffer_len, kv_hidden]
        new_kv_shape = [batch, 1, kv_hidden]
        q_shape = [batch, 1, q_hidden]
        out_shape = [batch, 1, q_hidden]

    nodes = []
    outputs = [helper.make_tensor_value_info("output", onnx_dtype, out_shape)]
    if attention_only:
        k_name, v_name = "key_cache", "value_cache"
    else:
        k_name, v_name = "updated_key_cache", "updated_value_cache"
        nodes.append(
            helper.make_node(
                "TensorScatter",
                inputs=["key_cache", "new_k", "write_indices"],
                outputs=["updated_key_cache"],
                name="TensorScatterKey",
                axis=scatter_axis,
            )
        )
        nodes.append(
            helper.make_node(
                "TensorScatter",
                inputs=["value_cache", "new_v", "write_indices"],
                outputs=["updated_value_cache"],
                name="TensorScatterValue",
                axis=scatter_axis,
            )
        )
        outputs.extend(
            [
                helper.make_tensor_value_info("updated_key_cache", onnx_dtype, cache_shape),
                helper.make_tensor_value_info("updated_value_cache", onnx_dtype, cache_shape),
            ]
        )

    nodes.append(
        helper.make_node(
            "Attention",
            inputs=["query", k_name, v_name, "", "", "", "nonpad_kv_seqlen"],
            outputs=["output"],
            name="Attention_0",
            is_causal=1,
            q_num_heads=q_heads,
            kv_num_heads=kv_heads,
            softcap=0.0,
            qk_matmul_output_mode=0,
            domain="",
        )
    )

    graph_inputs = [
        helper.make_tensor_value_info("key_cache", onnx_dtype, cache_shape),
        helper.make_tensor_value_info("value_cache", onnx_dtype, cache_shape),
        helper.make_tensor_value_info("query", onnx_dtype, q_shape),
        helper.make_tensor_value_info("nonpad_kv_seqlen", TensorProto.INT64, [batch]),
    ]
    if not attention_only:
        graph_inputs.extend(
            [
                helper.make_tensor_value_info("new_k", onnx_dtype, new_kv_shape),
                helper.make_tensor_value_info("new_v", onnx_dtype, new_kv_shape),
                helper.make_tensor_value_info("write_indices", TensorProto.INT64, [batch]),
            ]
        )

    graph = helper.make_graph(nodes, "AttentionScatter_Graph", graph_inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 24)])
    return model.SerializeToString()


# #################################################################################################
#  Arm runners: build (CudaSession, feed_dict) for one sweep point
# #################################################################################################


def create_cuda_session(model_bytes, device, buffer_sharing=None):
    device_id = torch.cuda.current_device()
    provider_options = CudaSession.get_cuda_provider_options(
        device_id, enable_cuda_graph=False, stream=torch.cuda.current_stream().cuda_stream
    )
    ort_session = InferenceSession(
        model_bytes, providers=[("CUDAExecutionProvider", provider_options), "CPUExecutionProvider"]
    )
    session = CudaSession(ort_session, device)
    if buffer_sharing:
        for input_name, output_name in buffer_sharing.items():
            session.set_buffer_sharing(input_name, output_name)
    return session


def make_gqa_arm(args, past_seq_len, torch_dtype, canonical=None):
    config = GroupQueryAttentionConfig(
        batch_size=args.batch,
        sequence_length=1,
        max_sequence_length=args.max_seq_len,
        past_sequence_length=past_seq_len,
        num_heads=args.q_heads,
        kv_num_heads=args.kv_heads,
        head_size=args.head_size,
        do_rotary=False,
        device=args.device,
        dtype=torch_dtype,
        kv_cache_type="float16" if torch_dtype == torch.float16 else "bfloat16",
    )
    session = create_gqa_ort_session(config)
    feeds = config.random_inputs()
    if canonical is not None:
        q, past_k, past_v, new_k, new_v = canonical
        feeds["query"] = q.reshape(args.batch, 1, -1).contiguous()
        feeds["key"] = new_k.reshape(args.batch, 1, -1).contiguous()
        feeds["value"] = new_v.reshape(args.batch, 1, -1).contiguous()
        feeds["past_key"].zero_()
        feeds["past_value"].zero_()
        feeds["past_key"][:, :, :past_seq_len, :] = past_k
        feeds["past_value"][:, :, :past_seq_len, :] = past_v
    return session, feeds


def make_attn_past_arm(args, past_seq_len, torch_dtype, canonical=None):
    onnx_dtype = TensorProto.FLOAT16 if torch_dtype == torch.float16 else TensorProto.BFLOAT16
    model = create_attention_past_model(
        args.batch, args.q_heads, args.kv_heads, args.head_size, past_seq_len, onnx_dtype
    )
    session = create_cuda_session(model, args.device)
    shape_dict = {
        "output": (args.batch, 1, args.q_heads * args.head_size),
        "present_key": (args.batch, args.kv_heads, past_seq_len + 1, args.head_size),
        "present_value": (args.batch, args.kv_heads, past_seq_len + 1, args.head_size),
    }
    session.allocate_buffers(shape_dict)

    if canonical is not None:
        q, past_k, past_v, new_k, new_v = canonical
    else:
        q, past_k, past_v, new_k, new_v = random_canonical_inputs(args, past_seq_len, torch_dtype)
    feeds = {
        "query": q.reshape(args.batch, 1, -1).contiguous(),
        "key": new_k.reshape(args.batch, 1, -1).contiguous(),
        "value": new_v.reshape(args.batch, 1, -1).contiguous(),
        "past_key": past_k.contiguous(),
        "past_value": past_v.contiguous(),
    }
    return session, feeds


def make_attn_scatter_arm(args, past_seq_len, torch_dtype, canonical=None):
    onnx_dtype = TensorProto.FLOAT16 if torch_dtype == torch.float16 else TensorProto.BFLOAT16
    model = create_attention_scatter_model(
        args.batch,
        args.q_heads,
        args.kv_heads,
        args.head_size,
        args.max_seq_len,
        onnx_dtype,
        args.attention_only,
        args.cache_4d,
    )
    buffer_sharing = (
        None if args.attention_only else {"key_cache": "updated_key_cache", "value_cache": "updated_value_cache"}
    )
    session = create_cuda_session(model, args.device, buffer_sharing)
    if args.cache_4d:
        session.allocate_buffers({"output": (args.batch, args.q_heads, 1, args.head_size)})
    else:
        session.allocate_buffers({"output": (args.batch, 1, args.q_heads * args.head_size)})

    if canonical is not None:
        q, past_k, past_v, new_k, new_v = canonical
    else:
        q, past_k, past_v, new_k, new_v = random_canonical_inputs(args, past_seq_len, torch_dtype)

    # Valid tokens occupy [0, past_seq_len) before the scatter appends position
    # past_seq_len. 3D cache is BSNH flattened to [batch, buffer_len, kv_hidden];
    # 4D cache is BNSH [batch, kv_heads, buffer_len, head_size].
    kv_hidden = args.kv_heads * args.head_size
    if args.cache_4d:
        key_cache = torch.zeros(
            args.batch, args.kv_heads, args.max_seq_len, args.head_size, dtype=torch_dtype, device=args.device
        )
        value_cache = torch.zeros_like(key_cache)
        key_cache[:, :, :past_seq_len, :] = past_k
        value_cache[:, :, :past_seq_len, :] = past_v
        query = q.transpose(1, 2).contiguous()  # [B, 1, H, D] -> [B, H, 1, D]
        new_k_feed = new_k.transpose(1, 2).contiguous()  # [B, 1, Hkv, D] -> [B, Hkv, 1, D]
        new_v_feed = new_v.transpose(1, 2).contiguous()
    else:
        key_cache = torch.zeros(args.batch, args.max_seq_len, kv_hidden, dtype=torch_dtype, device=args.device)
        value_cache = torch.zeros_like(key_cache)
        key_cache[:, :past_seq_len, :] = past_k.transpose(1, 2).reshape(args.batch, past_seq_len, kv_hidden)
        value_cache[:, :past_seq_len, :] = past_v.transpose(1, 2).reshape(args.batch, past_seq_len, kv_hidden)
        query = q.reshape(args.batch, 1, -1).contiguous()
        new_k_feed = new_k.reshape(args.batch, 1, -1).contiguous()
        new_v_feed = new_v.reshape(args.batch, 1, -1).contiguous()

    feeds = {
        "key_cache": key_cache,
        "value_cache": value_cache,
        "query": query,
        "nonpad_kv_seqlen": torch.full((args.batch,), past_seq_len + 1, dtype=torch.int64, device=args.device),
    }
    if args.attention_only:
        # No scatter node: pre-place the new token in the cache directly.
        if args.cache_4d:
            feeds["key_cache"][:, :, past_seq_len, :] = new_k_feed.reshape(args.batch, args.kv_heads, args.head_size)
            feeds["value_cache"][:, :, past_seq_len, :] = new_v_feed.reshape(args.batch, args.kv_heads, args.head_size)
        else:
            feeds["key_cache"][:, past_seq_len, :] = new_k_feed.reshape(args.batch, kv_hidden)
            feeds["value_cache"][:, past_seq_len, :] = new_v_feed.reshape(args.batch, kv_hidden)
    else:
        feeds["new_k"] = new_k_feed
        feeds["new_v"] = new_v_feed
        feeds["write_indices"] = torch.full((args.batch,), past_seq_len, dtype=torch.int64, device=args.device)
    return session, feeds


def random_canonical_inputs(args, past_seq_len, torch_dtype):
    """One decode step of inputs in canonical layouts shared by every arm.

    q:      [batch, 1, q_heads, head_size]
    past_k: [batch, kv_heads, past_seq_len, head_size]  (BNSH)
    past_v: [batch, kv_heads, past_seq_len, head_size]
    new_k:  [batch, 1, kv_heads, head_size]
    new_v:  [batch, 1, kv_heads, head_size]
    """
    torch.manual_seed(123)
    device = args.device

    def randn(*shape):
        return torch.empty(*shape, device=device, dtype=torch_dtype).normal_(mean=0, std=0.1)

    q = randn(args.batch, 1, args.q_heads, args.head_size)
    past_k = randn(args.batch, args.kv_heads, past_seq_len, args.head_size)
    past_v = randn(args.batch, args.kv_heads, past_seq_len, args.head_size)
    new_k = randn(args.batch, 1, args.kv_heads, args.head_size)
    new_v = randn(args.batch, 1, args.kv_heads, args.head_size)
    return q, past_k, past_v, new_k, new_v


ARM_BUILDERS = {
    "gqa_xqa": make_gqa_arm,
    "gqa_flash": make_gqa_arm,
    "gqa_cudnn": make_gqa_arm,
    "attn_past": make_attn_past_arm,
    "attn_scatter": make_attn_scatter_arm,
}


def build_arm(arm, args, past_seq_len, torch_dtype, canonical=None):
    with scoped_env(ARM_ENV[arm]):
        return ARM_BUILDERS[arm](args, past_seq_len, torch_dtype, canonical)


# #################################################################################################
#  Timing
# #################################################################################################


def event_bench(fn, warmup=20, rep=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


def bench_arm(session, feeds):
    def run():
        # The session runs on torch's current stream, so CUDA-event timers see
        # its work without per-call host syncs.
        session.infer(feeds, synchronize=False)

    if do_bench is not None:
        return do_bench(run, warmup=15, rep=100)
    return event_bench(run)


# #################################################################################################
#  Modes
# #################################################################################################


def print_provenance(args):
    device_id = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_id)
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        driver = "unknown"
    print("## Provenance")
    print(f"- GPU: {props.name} (SM{props.major}{props.minor}, {props.total_memory / (1 << 30):.0f} GiB)")
    print(f"- Driver: {driver}, torch {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"- onnxruntime: {onnxruntime.__version__}")
    print(f"- ORT build info: {onnxruntime.get_build_info().strip()}")
    print(f"- Timer: {TIMER}")
    print(f"- Args: {vars(args)}")
    pinned = sorted({k for env in ARM_ENV.values() for k in env} | {"ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO"})
    ambient = {k: os.environ.get(k) for k in pinned}
    print(f"- Ambient env (per-arm overrides in ARM_ENV apply at session creation): {ambient}")
    print(f"- Per-arm env: {ARM_ENV}")
    print()


def run_sweep(args):
    torch_dtype = TORCH_DTYPE[args.dtype]
    rows = []
    for past_seq_len in args.sweep:
        row = {"past_seq_len": past_seq_len}
        for arm in args.arms:
            session, feeds = build_arm(arm, args, past_seq_len, torch_dtype)
            latency_ms = bench_arm(session, feeds)
            row[arm] = latency_ms * 1000.0  # microseconds
            del session
        rows.append(row)
        torch.cuda.empty_cache()

    header = ["past_seq_len"] + [f"{arm} (us)" for arm in args.arms]
    print(
        f"## Decode latency, dtype={args.dtype}, B={args.batch}, "
        f"H={args.q_heads}/{args.kv_heads}, D={args.head_size}, buffer={args.max_seq_len}"
    )
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        cells = [str(row["past_seq_len"])] + [f"{row[arm]:.1f}" for arm in args.arms]
        print("| " + " | ".join(cells) + " |")
    print()

    if args.csv:
        write_header = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    [
                        "dtype",
                        "arm",
                        "batch",
                        "q_heads",
                        "kv_heads",
                        "head_size",
                        "past_seq_len",
                        "buffer_len",
                        "latency_us",
                        "timer",
                    ]
                )
            for row in rows:
                for arm in args.arms:
                    writer.writerow(
                        [
                            args.dtype,
                            arm,
                            args.batch,
                            args.q_heads,
                            args.kv_heads,
                            args.head_size,
                            row["past_seq_len"],
                            args.max_seq_len,
                            f"{row[arm]:.3f}",
                            TIMER,
                        ]
                    )
        print(f"CSV appended to {args.csv}")


def run_sanity(args):
    """Run one decode step per arm on identical inputs and compare outputs.

    A mismatch means the arms are not computing the same attention (config
    drift), which would invalidate the latency comparison.
    """
    torch_dtype = TORCH_DTYPE[args.dtype]
    past_seq_len = args.past_seq_len
    canonical = random_canonical_inputs(args, past_seq_len, torch_dtype)

    outputs = {}
    for arm in args.arms:
        session, feeds = build_arm(arm, args, past_seq_len, torch_dtype, canonical)
        result = session.infer(feeds)
        # 3D arms emit [B, 1, H*D]; the 4D scatter variant emits BNSH [B, H, 1, D].
        # Both flatten to the same head-major element order.
        outputs[arm] = result["output"].reshape(args.batch, 1, -1).float().cpu().clone()
        del session
        torch.cuda.empty_cache()

    reference_arm = args.arms[0]
    # Cross-kernel comparison of half-precision outputs; config drift shows up
    # as O(0.1+) diffs, far above these thresholds.
    threshold = 1e-2 if args.dtype == "float16" else 5e-2
    print(f"## Sanity: cross-arm output parity at past_seq_len={past_seq_len}, dtype={args.dtype}")
    print(f"(reference arm: {reference_arm}; max |diff| threshold {threshold})")
    ok = True
    for arm, out in outputs.items():
        if arm == reference_arm:
            continue
        max_diff = (out - outputs[reference_arm]).abs().max().item()
        status = "PASS" if max_diff <= threshold else "FAIL"
        ok &= status == "PASS"
        print(f"- {arm} vs {reference_arm}: max |diff| = {max_diff:.6f} [{status}]")
    print()
    return ok


def run_profile(args):
    """Run one fixed config for --profile-iters iterations with NVTX ranges,
    for capture under `nsys profile`. Use a single arm per invocation so
    kernel names attribute cleanly."""
    if len(args.arms) != 1:
        sys.exit("--profile requires exactly one arm (e.g. --arms attn_scatter)")
    arm = args.arms[0]
    torch_dtype = TORCH_DTYPE[args.dtype]
    session, feeds = build_arm(arm, args, args.past_seq_len, torch_dtype)

    for _ in range(20):  # warmup: allocation + flash autotune outside the ranges
        session.infer(feeds, synchronize=False)
    torch.cuda.synchronize()

    for _ in range(args.profile_iters):
        torch.cuda.nvtx.range_push("benchmark")
        session.infer(feeds, synchronize=False)
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    print(
        f"Profiled {args.profile_iters} iterations of {arm} at past_seq_len={args.past_seq_len} "
        f"(dtype={args.dtype}), after 20 warmup iterations; NVTX range name 'benchmark'."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", default=ALL_ARMS, choices=ALL_ARMS)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=32, help="Llama-3-8B default")
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=8192, help="KV cache buffer length for buffered arms")
    parser.add_argument(
        "--sweep", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096], help="past KV lengths to sweep"
    )
    parser.add_argument("--past-seq-len", type=int, default=2048, help="past KV length for --sanity/--profile")
    parser.add_argument(
        "--attention-only",
        action="store_true",
        help="drop the TensorScatter nodes from the attn_scatter arm (attribution runs)",
    )
    parser.add_argument(
        "--cache-4d",
        action="store_true",
        help="attn_scatter arm uses a 4D BNSH external cache (scatter axis=2) instead of 3D BSNH",
    )
    parser.add_argument("--csv", default=None, help="append results to this CSV file")
    parser.add_argument("--sanity", action="store_true", help="cross-arm output parity check, no timing")
    parser.add_argument("--profile", action="store_true", help="fixed-config NVTX-annotated loop for nsys")
    parser.add_argument("--profile-iters", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not torch.cuda.is_available() or "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
        sys.exit("This benchmark requires a CUDA device and an onnxruntime build with the CUDA EP.")
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        sys.exit(f"SM{major}{minor} < SM80: Flash/XQA decode paths are unavailable; results would be meaningless.")
    if max([*args.sweep, args.past_seq_len]) >= args.max_seq_len:
        sys.exit("--max-seq-len must exceed every swept past length (buffer holds past + 1 new token).")

    print_provenance(args)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.no_grad():
        if args.profile:
            run_profile(args)
        elif args.sanity:
            ok = run_sanity(args)
            sys.exit(0 if ok else 1)
        else:
            run_sweep(args)


if __name__ == "__main__":
    main()
