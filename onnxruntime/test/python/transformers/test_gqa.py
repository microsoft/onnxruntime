# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# Copyright 2020 The HuggingFace Inc. team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# -------------------------------------------------------------------------
import gc
import math
import os
import platform
import random
import re
import sys
import threading
import typing
import unittest
from copy import deepcopy
from dataclasses import dataclass

import numpy
import torch
from cuda_plugin_ep_helper import get_cuda_provider_name, resolve_cuda_plugin_ep
from einops import rearrange, repeat
from env_var_helper import scoped_env_var

# --- ONNX and Torch/Numpy Dtype Mappings ---
from gqa_test_helper import (
    ONNX_TENSOR_TYPE_MAP,
    TORCH_DTYPE_MAP,
    compute_scale,
    dequantize_tensor,
    quantize_tensor_with_scale,
)
from onnx import TensorProto, helper
from packaging import version
from parameterized import parameterized

from onnxruntime import InferenceSession, SessionOptions, get_build_info
from onnxruntime import __version__ as ort_version

# Set seed for reproducibility
torch.manual_seed(0)
random.seed(69)

try:
    from rotary_flash import apply_rotary_emb
except ImportError:
    apply_rotary_emb = None

# Reduces number of tests to run for faster pipeline checks
pipeline_mode = os.getenv("PIPELINE_MODE", "1") == "1"

# Number of values per parameter (compared to pipeline mode)
param_count = int(os.getenv("PARAM_COUNT", "3")) if not pipeline_mode else 2

# When quick build is used, flash attention only supports head_size=128
quick_build = ", quick-build=" in get_build_info()

has_int4_kv_cache = ", int4-kv-cache=" in get_build_info()

has_fp8_kv_cache = ", fp8-kv-cache=" in get_build_info()

# Enable debug print if tensor or node dumping is enabled in build.
enable_debug_print = ("dump-tensor" in get_build_info()) or ("dump-node" in get_build_info())

enable_deterministic_check = True
# #################################################################################################
#  Configuration and Helper Classes
# #################################################################################################


class CaptureStdout:
    """Capture output written to OS file descriptor 1 (C++ stdout).

    Uses fd-level dup2 redirection rather than contextlib.redirect_stdout because the kernel
    debug info is emitted by the native ONNX Runtime library directly to fd 1, which Python's
    redirect_stdout (which only swaps sys.stdout) cannot intercept.
    """

    def __init__(self):
        self.fd = 1
        self.chunk_size = 1024
        self.output = b""

    def _capture(self):
        chunks = []
        while chunk := os.read(self._pipe_reader, self.chunk_size):
            chunks.append(chunk)
        self.output = b"".join(chunks)

    def __enter__(self):
        sys.stdout.flush()
        self._duped_fd = os.dup(self.fd)
        self._pipe_reader, pipe_writer = os.pipe()
        os.dup2(pipe_writer, self.fd)
        os.close(pipe_writer)
        self._capture_thread = threading.Thread(target=self._capture)
        self._capture_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.flush()
        os.dup2(self._duped_fd, self.fd)
        self._capture_thread.join()
        os.close(self._pipe_reader)
        os.close(self._duped_fd)


def get_sdpa_kernel_from_debug_info(run_func):
    captured_text = None
    with scoped_env_var("ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO", "1"):
        with CaptureStdout() as captured:
            run_func()
        captured_text = captured.output.decode(errors="replace")

    if captured_text is not None:
        match = re.search(r"SdpaKernel=(?P<kernel>[A-Z_]+)", captured_text)
        if match is not None:
            return match.group("kernel")

        print("Failed to get sdpa kernel from debug info:", captured_text)

    return None


@dataclass
class GQAConfig:
    batch_size: int
    q_sequence_length: int
    kv_sequence_length: int
    num_heads: int
    kv_num_heads: int
    head_size: int
    past_kv_sequence_length: int = 0
    buffer_sequence_length: int = 0
    # Test-specific parameters
    local_window_size: int = -1
    # Opt into the cache-relative (windowed) KV cache. Requires local_window_size > 0 and a
    # past/present shared buffer whose capacity C is at least local_window_size. There is no
    # requirement involving q_sequence_length: a multi-token step that would need more than C
    # entries runs against a longer staging buffer, so prefill of any length works at C == W.
    # Slack above local_window_size is worth allocating anyway: it lets the append point drift, so
    # the CPU kernel compacts once every C - W + 1 steps instead of on every step.
    sliding_window_cache: int = 0
    # Length of the cos/sin caches. 0 means "same as buffer_sequence_length". A windowed KV cache is
    # shorter than the sequence, so RoPE positions must be driven by a separately sized rotary cache.
    rope_cache_length: int = 0
    rotary: bool = False
    rotary_interleaved: bool = False
    packed: bool = False
    softcap: float = 0.0
    use_smooth_softmax: bool = False
    has_head_sink: bool = False
    # When True, head_sink is baked into the model as a constant initializer (instead of a runtime
    # input). This exercises the GroupQueryAttention::PrePack path that converts the constant
    # head_sink to a cached FP32 XQA buffer.
    head_sink_as_initializer: bool = False
    kv_cache_type: str = ""
    share_buffer: bool = True
    share_kv_scale: bool = False
    causal: bool = True

    has_position_ids: bool = False
    has_attention_bias: bool = False
    # attention_bias leading dims: dim 0 is batch_size (or 1 when broadcast), dim 1 is
    # 1 (broadcast over heads) or num_heads. Defaults keep the original [batch, 1, ...] shape.
    attention_bias_broadcast_dim_0: bool = False
    attention_bias_per_head: bool = False

    # Quantization parameters
    k_quant_type: str = "NONE"
    v_quant_type: str = "NONE"
    kv_cache_bit_width: int = 0

    # Fused per-head Q/K RMSNorm (QK-Norm) applied before RoPE. Weight shape (head_size,) shared across heads.
    has_qk_norm: bool = False
    qk_norm_epsilon: float = 1e-6


# #################################################################################################
#  Rotary Embedding Implementations (CPU and CUDA)
# #################################################################################################


# PyTorch implementation for CPU and fallback
class LlamaMSRotaryEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def rotate_tensor(self, x, cos, sin, pos, interleaved):
        rot_dim = 2 * cos.shape[3]
        x_rot = x[:, :, :, :rot_dim]

        if interleaved:
            x1 = x_rot[:, :, :, 0::2]
            x2 = x_rot[:, :, :, 1::2]
        else:
            half = x_rot.shape[-1] // 2
            x1 = x_rot[:, :, :, 0:half]
            x2 = x_rot[:, :, :, half : 2 * half]

        seq_len = x.shape[1]
        batch_size = x.shape[0]

        cos = cos.squeeze(0).squeeze(1)
        sin = sin.squeeze(0).squeeze(1)

        if seq_len == 1:
            pos_i = pos.long()
            cos_x = cos[pos_i].unsqueeze(1)
            sin_x = sin[pos_i].unsqueeze(1)
        else:
            cos_x_list = []
            sin_x_list = []
            for b in range(batch_size):
                pos_b = pos[b]
                cos_x_list.append(cos[pos_b : pos_b + seq_len])
                sin_x_list.append(sin[pos_b : pos_b + seq_len])
            cos_x = torch.stack(cos_x_list, dim=0)
            sin_x = torch.stack(sin_x_list, dim=0)

        cos_x = cos_x.unsqueeze(2)
        sin_x = sin_x.unsqueeze(2)

        real = cos_x * x1 - sin_x * x2
        imag = sin_x * x1 + cos_x * x2

        if interleaved:
            x_rot[:, :, :, 0::2] = real
            x_rot[:, :, :, 1::2] = imag
        else:
            x_rot = torch.cat((real, imag), dim=-1)

        return torch.cat((x_rot, x[:, :, :, rot_dim:]), dim=-1)

    def forward(self, x, cos, sin, pos, interleaved):
        return self.rotate_tensor(x, cos, sin, pos, interleaved)


# Triton-based implementation for CUDA
def rotary_embedding_cuda(*args, **kwargs):
    if apply_rotary_emb is None:
        raise ImportError("rotary_flash not found")
    return apply_rotary_emb(*args, **kwargs)


# Unified wrapper for rotary embeddings
def apply_rotary_embedding(x, cos, sin, pos, interleaved, device="cpu"):
    """Applies rotary embedding, using Triton for CUDA if available, otherwise fallback to PyTorch."""
    use_cuda_triton = device == "cuda" and platform.system() == "Linux"
    if use_cuda_triton:
        try:
            return rotary_embedding_cuda(x, cos, sin, seqlen_offsets=pos, interleaved=interleaved)
        except ImportError:
            print("WARNING: Triton-based rotary embedding not found. Falling back to PyTorch version.")

    # PyTorch implementation for CPU or as a fallback for CUDA
    rot = LlamaMSRotaryEmbedding().to(device)
    # Unsqueeze to match the expected shape in the PyTorch version
    cos_unsqueezed = cos.unsqueeze(0).unsqueeze(2)
    sin_unsqueezed = sin.unsqueeze(0).unsqueeze(2)
    return rot(x, cos_unsqueezed, sin_unsqueezed, pos, interleaved)


# #################################################################################################
#  ONNX Graph Creation
# #################################################################################################


def make_head_sink_initializer(head_sink, ort_type, num_heads):
    """Build a constant head_sink initializer (fp16/bf16) so GroupQueryAttention::PrePack runs.

    The 16-bit float bits are reinterpreted as uint16 and stored as raw bytes, which works for
    both float16 and bfloat16 without relying on numpy bfloat16 support.
    """
    raw = head_sink.detach().reshape(num_heads).cpu().contiguous().view(torch.uint16).numpy().tobytes()
    return helper.make_tensor(name="head_sink", data_type=ort_type, dims=[num_heads], vals=raw, raw=True)


def make_qk_norm_weights(head_size, device, torch_type, seed=7):
    """Generate deterministic per-head Q/K RMSNorm weights of shape (head_size,)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    q_w = (1.0 + 0.1 * torch.randn(head_size, generator=gen, device=device, dtype=torch.float32)).to(torch_type)
    k_w = (1.0 + 0.1 * torch.randn(head_size, generator=gen, device=device, dtype=torch.float32)).to(torch_type)
    return q_w, k_w


def apply_qk_rmsnorm(x, weight, eps):
    """Reference per-head RMSNorm over the last (head_size) dim, computed in float32 then cast back.

    x_norm[c] = x[c] * rsqrt(mean(x^2) + eps) * weight[c]
    """
    dtype = x.dtype
    xf = x.to(torch.float32)
    inv_rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * inv_rms * weight.to(torch.float32)).to(dtype)


def create_gqa_node_and_io(
    config: GQAConfig,
    ort_type,
    share_buffer=True,
    is_past=False,
    output_qk: int = 0,  # CUDA does not support output_qk for GQA
    head_sink_values=None,
):
    if is_past:
        if share_buffer:
            past_kv_seqlen = config.buffer_sequence_length
            present_kv_seqlen = config.buffer_sequence_length
        else:
            past_kv_seqlen = config.past_kv_sequence_length
            present_kv_seqlen = config.past_kv_sequence_length + config.kv_sequence_length
    else:  # Prompt
        past_kv_seqlen = config.buffer_sequence_length if share_buffer else 0
        present_kv_seqlen = config.buffer_sequence_length if share_buffer else config.kv_sequence_length

    if not config.kv_cache_type:
        config.kv_cache_type = "float16" if ort_type == TensorProto.FLOAT16 else "bfloat16"

    initializers = []

    # --- Node Definition ---
    outputs = [
        "output",
        "present_key",
        "present_value",
    ]

    if output_qk > 0:
        outputs.append("output_qk")

    # Ensure kv_cache_bit_width is set correctly based on cache type if not provided
    bit_width = config.kv_cache_bit_width
    if bit_width == 0:
        if config.kv_cache_type == "int4":
            bit_width = 4
        elif config.kv_cache_type == "int8":
            bit_width = 8

    inputs = [
        "query",
        "key" if not config.packed else "",
        "value" if not config.packed else "",
        "past_key" if is_past or share_buffer or config.k_quant_type != "NONE" else "",
        "past_value" if is_past or share_buffer or config.k_quant_type != "NONE" else "",
        "seqlens_k",
        "total_sequence_length",
        "cos_cache" if config.rotary else "",
        "sin_cache" if config.rotary else "",
        "position_ids" if config.has_position_ids else "",
        "attention_bias" if config.has_attention_bias else "",
        "head_sink" if config.has_head_sink else "",
        "k_scale" if config.k_quant_type != "NONE" else "",
        "k_scale"
        if config.share_kv_scale and config.k_quant_type != "NONE"
        else ("v_scale" if config.v_quant_type != "NONE" else ""),
        "q_norm_weight" if config.has_qk_norm else "",
        "k_norm_weight" if config.has_qk_norm else "",
    ]

    # Remove trailing empty strings
    while inputs and inputs[-1] == "":
        inputs.pop()

    quantization_attributes = (
        {
            "k_quant_type": config.k_quant_type,
            "v_quant_type": config.v_quant_type,
            "kv_cache_bit_width": bit_width,
        }
        if config.k_quant_type != "NONE"
        else {}
    )

    qk_norm_attributes = {"qk_norm_epsilon": config.qk_norm_epsilon} if config.has_qk_norm else {}

    windowed_cache_attributes = {"sliding_window_cache": 1} if config.sliding_window_cache else {}

    node = helper.make_node(
        op_type="GroupQueryAttention",
        inputs=inputs,
        outputs=outputs,
        name="GroupQueryAttention_0",
        num_heads=config.num_heads,
        kv_num_heads=config.kv_num_heads,
        local_window_size=config.local_window_size,
        do_rotary=config.rotary,
        rotary_interleaved=config.rotary_interleaved,
        softcap=config.softcap,
        causal=1 if config.causal else 0,
        smooth_softmax=1 if config.use_smooth_softmax else 0,
        qk_output=output_qk,
        **quantization_attributes,
        **qk_norm_attributes,
        **windowed_cache_attributes,
        domain="com.microsoft",
    )

    # --- Graph Inputs ---
    q_hidden_size = (
        (config.num_heads * config.head_size)
        if not config.packed
        else (config.num_heads * config.head_size + 2 * config.kv_num_heads * config.head_size)
    )
    graph_input = [
        helper.make_tensor_value_info("query", ort_type, [config.batch_size, config.q_sequence_length, q_hidden_size]),
        helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [config.batch_size]),
        helper.make_tensor_value_info("total_sequence_length", TensorProto.INT32, [1]),
    ]
    cache_ort_type = ONNX_TENSOR_TYPE_MAP[config.kv_cache_type]

    if not config.packed:
        graph_input.extend(
            [
                helper.make_tensor_value_info(
                    "key",
                    ort_type,
                    [config.batch_size, config.kv_sequence_length, config.kv_num_heads * config.head_size],
                ),
                helper.make_tensor_value_info(
                    "value",
                    ort_type,
                    [config.batch_size, config.kv_sequence_length, config.kv_num_heads * config.head_size],
                ),
            ]
        )

    if is_past or share_buffer or config.k_quant_type != "NONE":
        k_shape = [config.batch_size, config.kv_num_heads, past_kv_seqlen, config.head_size]
        if config.kv_cache_type == "int4":
            k_shape[-1] //= 2
        graph_input.extend(
            [
                helper.make_tensor_value_info("past_key", cache_ort_type, k_shape),
                helper.make_tensor_value_info("past_value", cache_ort_type, k_shape),
            ]
        )
        if config.k_quant_type != "NONE":
            # Scales are always float32
            graph_input.append(helper.make_tensor_value_info("k_scale", TensorProto.FLOAT, None))
        if config.v_quant_type != "NONE" and not config.share_kv_scale:
            graph_input.append(helper.make_tensor_value_info("v_scale", TensorProto.FLOAT, None))

    if config.rotary:
        rotary_dim = (math.floor(config.head_size / 16) * 16) // 2
        cache_seq_len = config.rope_cache_length or config.buffer_sequence_length
        graph_input.extend(
            [
                helper.make_tensor_value_info("cos_cache", ort_type, [cache_seq_len, rotary_dim]),
                helper.make_tensor_value_info("sin_cache", ort_type, [cache_seq_len, rotary_dim]),
            ]
        )

    if config.has_position_ids:
        graph_input.append(
            helper.make_tensor_value_info(
                "position_ids", TensorProto.INT64, [config.batch_size, config.q_sequence_length]
            )
        )
    if config.has_attention_bias:
        # Per the op spec the last dim is total_sequence_length (past + new), which the kernel
        # validates at runtime; declare it symbolic so one graph serves all cache states.
        bias_dim_0 = 1 if config.attention_bias_broadcast_dim_0 else config.batch_size
        bias_dim_1 = config.num_heads if config.attention_bias_per_head else 1
        graph_input.append(
            helper.make_tensor_value_info(
                "attention_bias", ort_type, [bias_dim_0, bias_dim_1, config.q_sequence_length, "total_sequence_length"]
            )
        )
    if config.has_head_sink:
        if config.head_sink_as_initializer and head_sink_values is not None:
            # Constant initializer (not a graph input) so ORT treats it as a constant and PrePack runs.
            initializers.append(make_head_sink_initializer(head_sink_values, ort_type, config.num_heads))
        else:
            graph_input.append(helper.make_tensor_value_info("head_sink", ort_type, [config.num_heads]))

    if config.has_qk_norm:
        graph_input.append(helper.make_tensor_value_info("q_norm_weight", ort_type, [config.head_size]))
        graph_input.append(helper.make_tensor_value_info("k_norm_weight", ort_type, [config.head_size]))

    # --- Graph Outputs ---
    output_k_shape = [config.batch_size, config.kv_num_heads, present_kv_seqlen, config.head_size]
    if config.kv_cache_type == "int4":
        output_k_shape[-1] //= 2

    graph_output = [
        helper.make_tensor_value_info(
            "output", ort_type, [config.batch_size, config.q_sequence_length, config.num_heads * config.head_size]
        ),
        helper.make_tensor_value_info("present_key", cache_ort_type, output_k_shape),
        helper.make_tensor_value_info("present_value", cache_ort_type, output_k_shape),
    ]

    if output_qk > 0:
        graph_output.append(
            helper.make_tensor_value_info(
                "output_qk",
                ort_type,
                [config.batch_size, config.num_heads, config.q_sequence_length, present_kv_seqlen],
            )
        )

    return node, graph_input, graph_output, initializers


def create_group_query_attention_graph_prompt(config: GQAConfig, ort_type, share_buffer=True):
    node, graph_input, graph_output, initializers = create_gqa_node_and_io(
        config, ort_type, share_buffer, is_past=False
    )
    graph = helper.make_graph([node], "GroupQueryAttention_Graph", graph_input, graph_output, initializer=initializers)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 26), helper.make_opsetid("com.microsoft", 1)]
    )
    return model.SerializeToString()


def create_group_query_attention_graph_past(config: GQAConfig, ort_type, share_buffer=True, head_sink_values=None):
    node, graph_input, graph_output, initializers = create_gqa_node_and_io(
        config, ort_type, share_buffer, is_past=True, head_sink_values=head_sink_values
    )
    graph = helper.make_graph([node], "GroupQueryAttention_Graph", graph_input, graph_output, initializer=initializers)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 26), helper.make_opsetid("com.microsoft", 1)]
    )
    return model.SerializeToString()


# #################################################################################################
#  ONNX Runtime Execution Functions
# #################################################################################################


def bind_tensor(io_binding, name, tensor, device, ort_type):
    # Helper to bind a tensor to ONNX Runtime based on its device and type
    if tensor is None:
        return
    # Assuming tensor is a torch tensor. This works for both CPU and GPU tensors.
    io_binding.bind_input(
        name,
        tensor.device.type,
        0,
        ort_type,
        tuple(tensor.shape),
        tensor.data_ptr(),
    )


def bind_output_tensor(io_binding, name, tensor, device, ort_type):
    if tensor is None:
        return
    io_binding.bind_output(
        name,
        tensor.device.type,
        0,
        ort_type,
        tuple(tensor.shape),
        tensor.data_ptr(),
    )


def gqa_prompt_func(
    q,
    k,
    v,
    config: GQAConfig,
    new_k,
    new_v,
    cos,
    sin,
    seqlens_k,
    position_ids,
    attention_bias,
    head_sink,
    k_scale,
    v_scale,
    ep,
    device,
    share_buffer=True,
    ort_type=TensorProto.FLOAT16,
    q_norm_weight=None,
    k_norm_weight=None,
):
    if not config.kv_cache_type:
        config.kv_cache_type = "float16" if ort_type == TensorProto.FLOAT16 else "bfloat16"

    onnx_model_str = create_group_query_attention_graph_prompt(
        config=config,
        ort_type=ort_type,
        share_buffer=share_buffer,
    )

    q = torch.reshape(q, (config.batch_size, config.q_sequence_length, -1))
    if new_k is not None:
        kv_hidden_size = config.kv_num_heads * config.head_size
        new_k = torch.reshape(new_k, (config.batch_size, config.kv_sequence_length, kv_hidden_size))
        new_v = torch.reshape(new_v, (config.batch_size, config.kv_sequence_length, kv_hidden_size))

    sess_options = SessionOptions()
    ort_session = InferenceSession(onnx_model_str, sess_options, providers=[resolve_cuda_plugin_ep(ep)])
    io_binding = ort_session.io_binding()

    # Determine input device for binding
    # We assume primary inputs are on the target device

    # 1. Bind 'query'
    bind_tensor(io_binding, "query", q, device, ort_type)

    # 2. Bind 'key', 'value' (from new_k, new_v)
    if new_k is not None:
        bind_tensor(io_binding, "key", new_k, device, ort_type)
        bind_tensor(io_binding, "value", new_v, device, ort_type)

    # 3. Bind 'past_key', 'past_value' (if share_buffer or quantized)
    if share_buffer or config.k_quant_type != "NONE":
        # cache_ort_type corresponds to config.kv_cache_type
        cache_ort_type = ort_type
        if config.kv_cache_type:
            cache_ort_type = ONNX_TENSOR_TYPE_MAP[config.kv_cache_type]

        # Use full buffer if sharing, otherwise empty tensor for prompt phase
        k_to_bind = k if share_buffer else k[:, :, :0, :].contiguous()
        v_to_bind = v if share_buffer else v[:, :, :0, :].contiguous()

        bind_tensor(io_binding, "past_key", k_to_bind, device, cache_ort_type)
        bind_tensor(io_binding, "past_value", v_to_bind, device, cache_ort_type)

    # Scales are bound below in section 6

    # 4. Bind scalars/1D tensors
    # seqlens_k is INT32
    bind_tensor(io_binding, "seqlens_k", seqlens_k.to(torch.int32), device, TensorProto.INT32)

    # total_sequence_length is INT32 [1]
    # Schema requires this to be on CPU (OrtMemTypeCPUInput)
    cpu_device = torch.device("cpu")
    tsl = torch.tensor([config.q_sequence_length], dtype=torch.int32, device=cpu_device)
    bind_tensor(io_binding, "total_sequence_length", tsl, cpu_device, TensorProto.INT32)

    # 5. Optional inputs
    if cos is not None:
        bind_tensor(io_binding, "cos_cache", cos, device, ort_type)
        bind_tensor(io_binding, "sin_cache", sin, device, ort_type)

    if config.has_position_ids and position_ids is not None:
        bind_tensor(io_binding, "position_ids", position_ids, device, TensorProto.INT64)

    if config.has_attention_bias and attention_bias is not None:
        bind_tensor(io_binding, "attention_bias", attention_bias, device, ort_type)

    if config.has_head_sink and head_sink is not None:
        bind_tensor(io_binding, "head_sink", head_sink, device, ort_type)

    if config.has_qk_norm and q_norm_weight is not None and k_norm_weight is not None:
        bind_tensor(io_binding, "q_norm_weight", q_norm_weight, device, ort_type)
        bind_tensor(io_binding, "k_norm_weight", k_norm_weight, device, ort_type)

    # 6. Quantization scales
    if k_scale is not None:
        k_scale_ort_type = TensorProto.FLOAT
        if k_scale.dtype != torch.float32:
            k_scale = k_scale.to(torch.float32)
        k_scale = k_scale.contiguous()
        bind_tensor(io_binding, "k_scale", k_scale, device, k_scale_ort_type)
    if v_scale is not None:
        v_scale_ort_type = TensorProto.FLOAT
        if v_scale.dtype != torch.float32:
            v_scale = v_scale.to(torch.float32)
        v_scale = v_scale.contiguous()
        if not config.share_kv_scale:
            bind_tensor(io_binding, "v_scale", v_scale, device, v_scale_ort_type)

    # 7. Bind Outputs
    # output shape calculation
    hidden_size = config.num_heads * config.head_size

    out_dtype = TORCH_DTYPE_MAP.get(config.kv_cache_type, torch.float16)
    if ort_type == TensorProto.BFLOAT16:
        out_dtype = torch.bfloat16
    elif ort_type == TensorProto.FLOAT16:
        out_dtype = torch.float16
    else:
        out_dtype = torch.float32

    out_torch = torch.zeros((config.batch_size, config.q_sequence_length, hidden_size), dtype=out_dtype, device=device)
    bind_output_tensor(io_binding, "output", out_torch, device, ort_type)

    # present_dims logic
    if share_buffer:
        present_seqlen = config.buffer_sequence_length
    else:
        present_seqlen = config.kv_sequence_length

    present_dims = [config.batch_size, config.kv_num_heads, present_seqlen, config.head_size]

    # Update present shape when kv cache has quantization (int4 packs 2 values)
    if config.kv_cache_bit_width == 4:
        present_dims[-1] //= 2

    # Determine dtype for cache tensors
    cache_dtype = out_dtype
    cache_ort_type = ort_type
    if config.kv_cache_type in ONNX_TENSOR_TYPE_MAP:
        cache_ort_type = ONNX_TENSOR_TYPE_MAP[config.kv_cache_type]

    if config.kv_cache_type in TORCH_DTYPE_MAP:
        cache_dtype = TORCH_DTYPE_MAP[config.kv_cache_type]

    if share_buffer:
        # We bind output to the input buffer 'k' / 'v' (in-place update)
        # Assuming k and v are large enough buffers provided as input
        io_binding.bind_output("present_key", device, 0, cache_ort_type, tuple(k.shape), k.data_ptr())
        io_binding.bind_output("present_value", device, 0, cache_ort_type, tuple(v.shape), v.data_ptr())
        present_k = k
        present_v = v
    else:
        present_k = torch.zeros(tuple(present_dims), dtype=cache_dtype, device=device)
        present_v = torch.zeros(tuple(present_dims), dtype=cache_dtype, device=device)
        bind_output_tensor(io_binding, "present_key", present_k, device, cache_ort_type)
        bind_output_tensor(io_binding, "present_value", present_v, device, cache_ort_type)

    io_binding.synchronize_inputs()
    ort_session.run_with_iobinding(io_binding)
    io_binding.synchronize_outputs()

    return out_torch, present_k, present_v


def gqa_past_func(
    q,
    k,
    v,
    config: GQAConfig,
    new_k,
    new_v,
    cos,
    sin,
    seqlens_k,
    position_ids,
    attention_bias,
    head_sink,
    k_scale,
    v_scale,
    ep,
    device,
    share_buffer=True,
    ort_type=TensorProto.FLOAT16,
    q_norm_weight=None,
    k_norm_weight=None,
):
    if not config.kv_cache_type:
        config.kv_cache_type = "float16" if ort_type == TensorProto.FLOAT16 else "bfloat16"

    head_sink_as_initializer = config.has_head_sink and config.head_sink_as_initializer and head_sink is not None
    onnx_model_str = create_group_query_attention_graph_past(
        config=config,
        ort_type=ort_type,
        share_buffer=share_buffer,
        head_sink_values=head_sink if head_sink_as_initializer else None,
    )

    q = torch.reshape(q, (config.batch_size, config.q_sequence_length, -1))
    if new_k is not None:
        kv_hidden_size = config.kv_num_heads * config.head_size
        new_k = torch.reshape(new_k, (config.batch_size, config.kv_sequence_length, kv_hidden_size))
        new_v = torch.reshape(new_v, (config.batch_size, config.kv_sequence_length, kv_hidden_size))

    sess_options = SessionOptions()
    # sess_options.log_severity_level = 0
    ort_session = InferenceSession(onnx_model_str, sess_options, providers=[resolve_cuda_plugin_ep(ep)])
    io_binding = ort_session.io_binding()

    # Common inputs
    total_seq_len = config.past_kv_sequence_length + config.kv_sequence_length

    # 1. Bind 'query'
    bind_tensor(io_binding, "query", q, device, ort_type)

    # 2. Bind 'key', 'value' (from new_k, new_v) --> wait, past func takes separate new_k/new_v inputs?
    # In past_func, new_k/new_v are the *new* tokens to accept.
    if new_k is not None:
        bind_tensor(io_binding, "key", new_k, device, ort_type)
        bind_tensor(io_binding, "value", new_v, device, ort_type)

    # 3. Bind 'past_key', 'past_value'
    # These are required inputs for past_func
    # cache_ort_type corresponds to config.kv_cache_type
    cache_ort_type = ONNX_TENSOR_TYPE_MAP[config.kv_cache_type]

    if share_buffer:
        # If sharing buffer, we bind 'past_key' to the large buffer 'k'
        bind_tensor(io_binding, "past_key", k, device, cache_ort_type)
        bind_tensor(io_binding, "past_value", v, device, cache_ort_type)
    else:
        # If not sharing buffer, 'k' and 'v' are the *past* states passed in.
        # We must slice the buffer to the valid past length expected by the graph.
        past_len = config.past_kv_sequence_length
        k_sliced = k[:, :, :past_len, :].contiguous()
        v_sliced = v[:, :, :past_len, :].contiguous()
        bind_tensor(io_binding, "past_key", k_sliced, device, cache_ort_type)
        bind_tensor(io_binding, "past_value", v_sliced, device, cache_ort_type)

    # 4. Scalars
    seqlens_k_int32 = seqlens_k.to(dtype=torch.int32, device=device)
    bind_tensor(io_binding, "seqlens_k", seqlens_k_int32, device, TensorProto.INT32)

    # GroupQueryAttention expects total_sequence_length as CPU input.
    cpu_device = torch.device("cpu")
    tsl = torch.tensor([total_seq_len], dtype=torch.int32, device=cpu_device)
    bind_tensor(io_binding, "total_sequence_length", tsl, cpu_device, TensorProto.INT32)

    # 5. Optional inputs
    if cos is not None:
        bind_tensor(io_binding, "cos_cache", cos, device, ort_type)
        bind_tensor(io_binding, "sin_cache", sin, device, ort_type)

    if config.has_position_ids and position_ids is not None:
        bind_tensor(io_binding, "position_ids", position_ids, device, TensorProto.INT64)

    if config.has_attention_bias and attention_bias is not None:
        bind_tensor(io_binding, "attention_bias", attention_bias, device, ort_type)

    if config.has_head_sink and head_sink is not None and not head_sink_as_initializer:
        bind_tensor(io_binding, "head_sink", head_sink, device, ort_type)

    if config.has_qk_norm and q_norm_weight is not None and k_norm_weight is not None:
        bind_tensor(io_binding, "q_norm_weight", q_norm_weight, device, ort_type)
        bind_tensor(io_binding, "k_norm_weight", k_norm_weight, device, ort_type)

    # 6. Quantization
    if k_scale is not None:
        k_scale_ort_type = TensorProto.FLOAT
        if k_scale.dtype != torch.float32:
            k_scale = k_scale.to(torch.float32)
        k_scale = k_scale.contiguous()
        bind_tensor(io_binding, "k_scale", k_scale, device, k_scale_ort_type)

    if v_scale is not None and not config.share_kv_scale:
        v_scale_ort_type = TensorProto.FLOAT
        if v_scale.dtype != torch.float32:
            v_scale = v_scale.to(torch.float32)
        v_scale = v_scale.contiguous()
        # Even if share_kv_scale is True, the node might have two scale inputs named "k_scale" and "v_scale"
        # depending on the graph creation logic. We should bind "v_scale" if it's expected by the graph.
        # In create_gqa_node_and_io, if share_kv_scale is True, Input 13 is named "k_scale".
        # But if it's False, it's named "v_scale".
        if not config.share_kv_scale:
            bind_tensor(io_binding, "v_scale", v_scale, device, v_scale_ort_type)

    # 7. Outputs
    # output shape calculation
    hidden_size = config.num_heads * config.head_size

    out_dtype = TORCH_DTYPE_MAP.get(config.kv_cache_type, torch.float16)
    if ort_type == TensorProto.BFLOAT16:
        out_dtype = torch.bfloat16
    elif ort_type == TensorProto.FLOAT16:
        out_dtype = torch.float16
    else:
        out_dtype = torch.float32

    # Initialize to zeros
    out_torch = torch.zeros((config.batch_size, config.q_sequence_length, hidden_size), dtype=out_dtype, device=device)
    bind_output_tensor(io_binding, "output", out_torch, device, ort_type)

    # present_dims logic
    if share_buffer:
        present_seqlen = config.buffer_sequence_length
    else:
        present_seqlen = total_seq_len  # For past_func, total seq len is accumulated

    present_dims = [config.batch_size, config.kv_num_heads, present_seqlen, config.head_size]
    if config.kv_cache_bit_width == 4:
        present_dims[-1] //= 2

    cache_dtype = out_dtype
    cache_ort_type = ort_type
    if config.kv_cache_type in ONNX_TENSOR_TYPE_MAP:
        cache_ort_type = ONNX_TENSOR_TYPE_MAP[config.kv_cache_type]
        if config.kv_cache_type in TORCH_DTYPE_MAP:
            cache_dtype = TORCH_DTYPE_MAP[config.kv_cache_type]

    if share_buffer:
        # In-place update to k/v buffers
        io_binding.bind_output("present_key", device, 0, cache_ort_type, tuple(k.shape), k.data_ptr())
        io_binding.bind_output("present_value", device, 0, cache_ort_type, tuple(v.shape), v.data_ptr())
        present_k = k
        present_v = v
    else:
        present_k = torch.zeros(tuple(present_dims), dtype=cache_dtype, device=device)
        present_v = torch.zeros(tuple(present_dims), dtype=cache_dtype, device=device)
        bind_output_tensor(io_binding, "present_key", present_k, device, cache_ort_type)
        bind_output_tensor(io_binding, "present_value", present_v, device, cache_ort_type)

    io_binding.synchronize_inputs()
    ort_session.run_with_iobinding(io_binding)
    io_binding.synchronize_outputs()

    return out_torch, present_k, present_v


# #################################################################################################
#  Reference Attention Implementation
# #################################################################################################


def construct_local_mask(seqlen_q, seqlen_k, window_size, query_padding_mask, key_padding_mask, device):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    sk = seqlen_k if key_padding_mask is None else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    sq = seqlen_q if query_padding_mask is None else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx <= row_idx + sk - sq - window_size[0],
        )


def smooth_softmax_ref(x, head_sink):
    b, n, s, _ = x.shape
    if head_sink is not None:
        sink = head_sink.reshape(1, n, 1, 1).expand(b, -1, s, -1)
    else:
        sink = torch.zeros(b, n, s, 1, dtype=x.dtype, device=x.device)

    y = torch.cat([x, sink], dim=-1)
    y = torch.softmax(y, dim=-1)
    return y[..., :-1]


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attention_bias=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    use_smooth_softmax=False,
    head_sink=None,
):
    if causal:
        window_size = (window_size[0], 0)

    dtype_og = q.dtype
    q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]

    # Repeat K/V heads for Grouped-Query Attention
    if k.shape[2] != q.shape[2]:
        k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    if v.shape[2] != q.shape[2]:
        v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])

    scores = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])

    if softcap > 0:
        scores = (scores / softcap).tanh() * softcap

    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))

    local_mask = None
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q, seqlen_k, window_size, query_padding_mask, key_padding_mask, q.device
        )
        scores.masked_fill_(local_mask, float("-inf"))

    # Add custom attention bias if provided (for CPU tests)
    if attention_bias is not None:
        # The bias should only be applied to the relevant part of the scores matrix,
        # matching the sequence length of the bias tensor.
        scores[..., : attention_bias.shape[-1]] += attention_bias

    if use_smooth_softmax or (head_sink is not None):
        # Note that the sink directly joins softmax. No scaling and softcap is needed!
        attention = smooth_softmax_ref(scores, head_sink)
    else:
        attention = torch.softmax(scores, dim=-1)

    # Fill NaNs with 0
    if local_mask is not None:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)

    output = torch.einsum("bhts,bshd->bthd", attention, v)

    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)

    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


# #################################################################################################
# Parity Check (Core Test Logic)
# #################################################################################################
def get_static_scale(config: GQAConfig, device, torch_type, std):
    """Generates calibration data and computes the static quantization scale."""
    calibration_batch_size = 1
    calibration_sequence_length = 1024
    calibration_data_k = (
        torch.randn(
            calibration_batch_size,
            config.kv_num_heads,
            calibration_sequence_length,
            config.head_size,
            device=device,
            dtype=torch_type,
        )
        * std
    )
    calibration_data_v = torch.randn_like(calibration_data_k) * std

    # TODO: handle config.share_kv_scale here.
    k_scale = compute_scale(calibration_data_k, config.k_quant_type, config.kv_cache_type)
    if config.share_kv_scale:
        v_scale = k_scale
    else:
        v_scale = compute_scale(calibration_data_v, config.v_quant_type, config.kv_cache_type)
    return k_scale, v_scale


def parity_check_gqa_prompt(
    config: GQAConfig,
    ep,
    device,
    torch_type,
    ort_type,
    causal,
    rtol,
    atol,
    std=0.2,
):
    config.causal = causal
    torch.manual_seed(0)
    q = (
        torch.randn(
            config.batch_size,
            config.q_sequence_length,
            config.num_heads,
            config.head_size,
            device=device,
            dtype=torch_type,
        )
        * std
    )

    # Initialize the KV cache to zeros since no past context in prompt testing.
    cache_dtype = torch_type
    if config.kv_cache_type:
        cache_dtype = TORCH_DTYPE_MAP[config.kv_cache_type]

    k = torch.zeros(
        config.batch_size,
        config.kv_num_heads,
        config.buffer_sequence_length,
        config.head_size if config.kv_cache_bit_width != 4 else config.head_size // 2,
        device=device,
        dtype=cache_dtype,
    )
    v = torch.zeros_like(k)

    new_k = (
        torch.randn(
            config.batch_size,
            config.kv_sequence_length,
            config.kv_num_heads,
            config.head_size,
            device=device,
            dtype=torch_type,
        )
        * std
    )
    new_v = torch.randn_like(new_k) * std

    k_scale, v_scale = get_static_scale(config, device, torch_type, std)
    if k_scale is not None:
        k_scale = k_scale.to(torch_type)
    if v_scale is not None:
        v_scale = v_scale.to(torch_type)

    head_sink = torch.rand(config.num_heads, dtype=torch_type, device=device) if config.has_head_sink else None
    window_size = (-1, -1)
    if config.local_window_size > 0:
        window_size = (config.local_window_size, 0)
    elif causal:
        window_size = (-1, 0)

    if config.kv_cache_bit_width == 4 or config.kv_cache_type == "int8" or config.kv_cache_type == "fp8":
        # k/v are already quantized (int8/fp8) in inputs
        k_ref_dequant = dequantize_tensor(k, k_scale, config.k_quant_type, config.kv_cache_type)
        v_ref_dequant = dequantize_tensor(v, v_scale, config.v_quant_type, config.kv_cache_type)
    else:
        k_ref_dequant = dequantize_tensor(
            quantize_tensor_with_scale(
                k,
                k_scale.to(torch.float32) if k_scale is not None else None,
                config.k_quant_type,
                config.kv_cache_type,
            ),
            k_scale.to(torch.float32) if k_scale is not None else None,
            config.k_quant_type,
            config.kv_cache_type,
        )
        v_ref_dequant = dequantize_tensor(
            quantize_tensor_with_scale(
                v,
                v_scale.to(torch.float32) if v_scale is not None else None,
                config.v_quant_type,
                config.kv_cache_type,
            ),
            v_scale.to(torch.float32) if v_scale is not None else None,
            config.v_quant_type,
            config.kv_cache_type,
        )
    k_cache_ref = k_ref_dequant.clone().transpose(1, 2)
    v_cache_ref = v_ref_dequant.clone().transpose(1, 2)
    cache_seqlens = torch.full((config.batch_size,), config.kv_sequence_length, device=device, dtype=torch.int32)
    rotary_seqlens = torch.zeros(config.batch_size, device=device, dtype=torch.long)

    cos, sin, q_ro, k_ro = None, None, q, new_k
    q_norm_weight, k_norm_weight = None, None
    if config.has_qk_norm:
        q_norm_weight, k_norm_weight = make_qk_norm_weights(config.head_size, device, torch_type)
        q_ro = apply_qk_rmsnorm(q, q_norm_weight, config.qk_norm_epsilon)
        k_ro = apply_qk_rmsnorm(new_k, k_norm_weight, config.qk_norm_epsilon)
    if config.rotary:
        rotary_dim = math.floor(config.head_size / 16) * 16
        angle = torch.rand(config.buffer_sequence_length, rotary_dim // 2, device=device) * 2 * math.pi
        cos = torch.cos(angle).to(dtype=torch_type)
        sin = torch.sin(angle).to(dtype=torch_type)
        q_ro = apply_rotary_embedding(q_ro.clone(), cos, sin, rotary_seqlens, config.rotary_interleaved, device)
        k_ro = apply_rotary_embedding(k_ro.clone(), cos, sin, rotary_seqlens, config.rotary_interleaved, device)

    position_ids, attention_bias = None, None
    if config.has_position_ids:
        position_ids = (
            torch.arange(config.q_sequence_length, device=device)
            .unsqueeze(0)
            .expand(config.batch_size, -1)
            .contiguous()
        )
    if config.has_attention_bias:
        # Random (non-zero) bias so that a kernel silently ignoring the input fails parity.
        attention_bias = (
            torch.randn(
                1 if config.attention_bias_broadcast_dim_0 else config.batch_size,
                config.num_heads if config.attention_bias_per_head else 1,
                config.q_sequence_length,
                config.kv_sequence_length,
                device=device,
                dtype=torch_type,
            )
            * 0.5
        )

    arange = rearrange(torch.arange(config.buffer_sequence_length, device=device), "s -> 1 s")
    kv_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
    update_mask = arange < kv_seqlens_expanded

    k_to_cache = k_ro
    v_to_cache = new_v
    if config.kv_cache_type != "none":
        k_scale_bsnh = k_scale
        v_scale_bsnh = v_scale
        if config.k_quant_type == "PER_CHANNEL" and k_scale is not None:
            k_scale_bsnh = k_scale.transpose(1, 2)  # (1, H, 1, D) -> (1, 1, H, D)
        if config.v_quant_type == "PER_CHANNEL" and v_scale is not None:
            v_scale_bsnh = v_scale.transpose(1, 2)  # (1, H, 1, D) -> (1, 1, H, D)

        k_to_cache = dequantize_tensor(
            quantize_tensor_with_scale(k_ro, k_scale_bsnh, config.k_quant_type, config.kv_cache_type),
            k_scale_bsnh,
            config.k_quant_type,
            config.kv_cache_type,
        ).to(torch_type)
        v_to_cache = dequantize_tensor(
            quantize_tensor_with_scale(new_v, v_scale_bsnh, config.v_quant_type, config.kv_cache_type),
            v_scale_bsnh,
            config.v_quant_type,
            config.kv_cache_type,
        ).to(torch_type)

    k_cache_ref[update_mask] = rearrange(k_to_cache, "b s ... -> (b s) ...").to(k_cache_ref.dtype)
    v_cache_ref[update_mask] = rearrange(v_to_cache, "b s ... -> (b s) ...").to(v_cache_ref.dtype)

    out_ref, _ = attention_ref(
        q=q_ro,
        k=k_ro,
        v=new_v,
        key_padding_mask=None,
        attention_bias=attention_bias,
        causal=causal,
        window_size=window_size,
        softcap=config.softcap,
        use_smooth_softmax=config.use_smooth_softmax,
        head_sink=head_sink,
    )
    out_ref_np = out_ref.to(torch.float32).detach().cpu().numpy()

    # --- ONNX Runtime Path ---
    q_ort, new_k_ort, new_v_ort = q, new_k, new_v
    if config.packed:
        q_ort = torch.cat([q, new_k, new_v], dim=2)
        new_k_ort, new_v_ort = None, None

    ort_seqlens = cache_seqlens - 1
    out, present_k, present_v = gqa_prompt_func(
        q=q_ort,
        k=k,
        v=v,
        config=config,
        new_k=new_k_ort,
        new_v=new_v_ort,
        cos=cos,
        sin=sin,
        seqlens_k=ort_seqlens,
        position_ids=position_ids,
        attention_bias=attention_bias,
        head_sink=head_sink,
        k_scale=k_scale,
        v_scale=v_scale,
        ep=ep,
        device=device,
        share_buffer=config.share_buffer,
        ort_type=ort_type,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
    )
    out = torch.reshape(out, (config.batch_size, config.q_sequence_length, config.num_heads, config.head_size))
    out_np = out.to(torch.float32).detach().cpu().numpy()

    # --- Comparison ---
    # Check for NaN in output
    nan_count = numpy.sum(numpy.isnan(out_np))
    if nan_count > 0:
        nan_indices = numpy.argwhere(numpy.isnan(out_np))
        print(f"DEBUG_NAN: Found {nan_count} NaN values in output!")
        print(f"DEBUG_NAN: First 5 NaN indices: {nan_indices[:5]}")
        # Also check where non-nan exists in reference
        ref_nan_count = numpy.sum(numpy.isnan(out_ref_np))
        print(f"DEBUG_NAN: Reference has {ref_nan_count} NaN values")

    # Compare KV cache
    # Use float32 for comparison to support bfloat16 and avoid numpy issues
    # Transpose reference back to BNSH to match ORT output
    k_cache_ref_np = k_cache_ref.transpose(1, 2).to(torch.float32).detach().cpu().numpy()
    v_cache_ref_np = v_cache_ref.transpose(1, 2).to(torch.float32).detach().cpu().numpy()
    present_k_np = present_k.to(torch.float32).detach().cpu().numpy()
    present_v_np = present_v.to(torch.float32).detach().cpu().numpy()

    if not config.share_buffer:
        k_cache_ref_np = k_cache_ref_np[:, :, : config.kv_sequence_length, :]
        v_cache_ref_np = v_cache_ref_np[:, :, : config.kv_sequence_length, :]

    if config.k_quant_type == "NONE":
        numpy.testing.assert_allclose(present_k_np, k_cache_ref_np, rtol=rtol, atol=atol)
        numpy.testing.assert_allclose(present_v_np, v_cache_ref_np, rtol=rtol, atol=atol)

    print_diff_statistics(torch.tensor(out_np - out_ref_np), "out")
    numpy.testing.assert_allclose(out_np, out_ref_np, rtol=rtol, atol=atol)

    # Compare quantized cache with proper masking per batch
    if config.k_quant_type != "NONE":
        # Convert numpy array to torch tensor with correct dtype
        if isinstance(present_k, torch.Tensor):
            present_k_torch = present_k.to(device)
            # If tensor is int8/uint8, it should be preserved.
        else:
            if config.kv_cache_type == "int4":
                # For int4, present_k is uint8 packed data
                present_k_torch = torch.from_numpy(present_k).to(device)
            elif config.kv_cache_type == "int8":
                # For int8, present_k is int8 data
                present_k_torch = torch.from_numpy(present_k.astype(numpy.int8)).to(device)
            elif config.kv_cache_type == "fp8":
                # For fp8, present_k is float8_e4m3fn data, returned as uint8/int8 by ORT python
                present_k_torch = torch.from_numpy(present_k).view(torch.float8_e4m3fn).to(device)
            else:
                present_k_torch = torch.from_numpy(present_k).to(device)

        present_k_dequant = (
            dequantize_tensor(present_k_torch, k_scale, config.k_quant_type, config.kv_cache_type)
            .detach()
            .cpu()
            .numpy()
        )

        # Mask the reference cache to only valid regions
        k_cache_ref_masked = k_cache_ref.transpose(1, 2).clone()
        arange = torch.arange(config.buffer_sequence_length, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        cache_seqlens_expanded = cache_seqlens.unsqueeze(1).unsqueeze(1).unsqueeze(-1)
        mask = arange >= cache_seqlens_expanded
        k_cache_ref_masked[mask.expand_as(k_cache_ref_masked)] = 0
        k_cache_ref_dequant = k_cache_ref_masked.cpu().numpy()

        for b in range(config.batch_size):
            valid_len = cache_seqlens[b].item()
            print_diff_statistics(
                torch.tensor(present_k_dequant[b, :, :valid_len, :] - k_cache_ref_dequant[b, :, :valid_len, :]),
                f"present_k[{b}]",
            )
            numpy.testing.assert_allclose(
                present_k_dequant[b, :, :valid_len, :], k_cache_ref_dequant[b, :, :valid_len, :], rtol=rtol, atol=atol
            )

    if config.v_quant_type != "NONE":
        # Convert numpy array to torch tensor with correct dtype
        if isinstance(present_v, torch.Tensor):
            present_v_torch = present_v.to(device)
        else:
            if config.kv_cache_type == "int4":
                present_v_torch = torch.from_numpy(present_v).to(device)
            elif config.kv_cache_type == "int8":
                present_v_torch = torch.from_numpy(present_v.astype(numpy.int8)).to(device)
            elif config.kv_cache_type == "fp8":
                present_v_torch = torch.from_numpy(present_v).view(torch.float8_e4m3fn).to(device)
            else:
                present_v_torch = torch.from_numpy(present_v).to(device)

        present_v_dequant = (
            dequantize_tensor(present_v_torch, v_scale, config.v_quant_type, config.kv_cache_type)
            .detach()
            .cpu()
            .numpy()
        )

        # Mask the reference cache to only valid regions
        v_cache_ref_masked = v_cache_ref.transpose(1, 2).clone()
        arange = torch.arange(config.buffer_sequence_length, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        cache_seqlens_expanded = cache_seqlens.unsqueeze(1).unsqueeze(1).unsqueeze(-1)
        mask = arange >= cache_seqlens_expanded
        v_cache_ref_masked[mask.expand_as(v_cache_ref_masked)] = 0
        v_cache_ref_dequant = v_cache_ref_masked.cpu().numpy()

        for b in range(config.batch_size):
            valid_len = cache_seqlens[b].item()
            print_diff_statistics(
                torch.tensor(present_v_dequant[b, :, :valid_len, :] - v_cache_ref_dequant[b, :, :valid_len, :]),
                f"present_v[{b}]",
            )
            numpy.testing.assert_allclose(
                present_v_dequant[b, :, :valid_len, :], v_cache_ref_dequant[b, :, :valid_len, :], rtol=rtol, atol=atol
            )


def parity_check_gqa_past(
    config: GQAConfig,
    ep,
    device,
    torch_type,
    ort_type,
    causal,
    rtol,
    atol,
    std=0.2,
):
    config.causal = causal
    if ort_type == TensorProto.FLOAT16:
        torch_type = torch.float16
    elif ort_type == TensorProto.BFLOAT16:
        torch_type = torch.bfloat16
    else:
        torch_type = torch.float32
    torch.manual_seed(0)
    # --- Test Data Generation ---
    q = (
        torch.randn(
            config.batch_size,
            config.q_sequence_length,
            config.num_heads,
            config.head_size,
            device=device,
            dtype=torch_type,
        )
        * std
    )
    k = (
        torch.randn(
            config.batch_size,
            config.kv_num_heads,
            config.buffer_sequence_length,
            config.head_size,
            device=device,
            dtype=torch_type,
        )
        * std
    )
    v = torch.randn_like(k) * std

    # Random past sequence lengths. This tests paddings in decoding.
    # Use a separate generator to ensure deterministic behavior independent of prior RNG state.
    cache_seqlens_gen = torch.Generator(device=device).manual_seed(42)
    cache_seqlens = torch.randint(
        1,
        config.past_kv_sequence_length + 1,
        (config.batch_size,),
        device=device,
        dtype=torch.long,
        generator=cache_seqlens_gen,
    )

    for i in range(config.batch_size):
        past_len = cache_seqlens[i].item()
        k[i, :, past_len:, :] = 0
        v[i, :, past_len:, :] = 0

    new_k = (
        torch.randn(
            config.batch_size,
            config.kv_sequence_length,
            config.kv_num_heads,
            config.head_size,
            device=device,
            dtype=torch_type,
        )
        * std
    )
    new_v = torch.randn_like(new_k) * std
    head_sink = torch.rand(config.num_heads, dtype=torch_type, device=device) if config.has_head_sink else None
    window_size = (-1, -1)
    if config.local_window_size > 0:
        window_size = (config.local_window_size, 0)
    elif causal:
        window_size = (-1, 0)

    k_scale, v_scale = get_static_scale(config, device, torch_type, std)
    if k_scale is not None:
        k_scale = k_scale.to(torch_type)
    if v_scale is not None:
        v_scale = v_scale.to(torch_type)

    # --- PyTorch Reference Path ---
    # Transpose BNSH cache to BSNH format for reference implementation

    k_ref_dequant = dequantize_tensor(
        quantize_tensor_with_scale(k, k_scale, config.k_quant_type, config.kv_cache_type),
        k_scale,
        config.k_quant_type,
        config.kv_cache_type,
    )
    v_ref_dequant = dequantize_tensor(
        quantize_tensor_with_scale(v, v_scale, config.v_quant_type, config.kv_cache_type),
        v_scale,
        config.v_quant_type,
        config.kv_cache_type,
    )
    k_cache_ref = k_ref_dequant.clone().transpose(1, 2)
    v_cache_ref = v_ref_dequant.clone().transpose(1, 2)

    cos, sin, q_ro, k_ro = None, None, q, new_k
    q_norm_weight, k_norm_weight = None, None
    if config.has_qk_norm:
        q_norm_weight, k_norm_weight = make_qk_norm_weights(config.head_size, device, torch_type)
        q_ro = apply_qk_rmsnorm(q, q_norm_weight, config.qk_norm_epsilon)
        k_ro = apply_qk_rmsnorm(new_k, k_norm_weight, config.qk_norm_epsilon)
    if config.rotary:
        rotary_dim = math.floor(config.head_size / 16) * 16
        angle = torch.rand(config.buffer_sequence_length, rotary_dim // 2, device=device) * 2 * math.pi
        cos = torch.cos(angle).to(dtype=torch_type)
        sin = torch.sin(angle).to(dtype=torch_type)
        q_ro = apply_rotary_embedding(q_ro.clone(), cos, sin, cache_seqlens, config.rotary_interleaved, device)
        k_ro = apply_rotary_embedding(k_ro.clone(), cos, sin, cache_seqlens, config.rotary_interleaved, device)

    position_ids, attention_bias = None, None
    total_seq_len = config.past_kv_sequence_length + config.kv_sequence_length
    if config.has_position_ids:
        position_ids = (cache_seqlens.unsqueeze(1) + torch.arange(config.q_sequence_length, device=device)).long()
    if config.has_attention_bias:
        # Random (non-zero) bias so that a kernel silently ignoring the input fails parity.
        # Positions beyond each batch's valid length get a large negative finite value
        # (matching the CPU test convention; avoids inf-inf NaN when composed with masking).
        # A batch-broadcast bias (dim 0 == 1) cannot carry per-batch tails; it relies on
        # the kernel's seqlens_k cutoff and the reference mask never reading those
        # positions — which the dim0-broadcast past cases below verify holds.
        attention_bias = (
            torch.randn(
                1 if config.attention_bias_broadcast_dim_0 else config.batch_size,
                config.num_heads if config.attention_bias_per_head else 1,
                config.q_sequence_length,
                total_seq_len,
                device=device,
                dtype=torch_type,
            )
            * 0.5
        )
        if not config.attention_bias_broadcast_dim_0:
            for b in range(config.batch_size):
                end_pos = cache_seqlens[b] + config.q_sequence_length
                attention_bias[b, :, :, end_pos:] = -10000.0

    arange = rearrange(torch.arange(config.buffer_sequence_length, device=device), "s -> 1 s")
    cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
    update_mask = torch.logical_and(
        cache_seqlens_expanded <= arange, arange < cache_seqlens_expanded + config.kv_sequence_length
    )

    k_to_cache = k_ro
    v_to_cache = new_v
    if config.kv_cache_type != "none":
        k_scale_bsnh = k_scale
        v_scale_bsnh = v_scale
        if config.k_quant_type == "PER_CHANNEL" and k_scale is not None:
            k_scale_bsnh = k_scale.transpose(1, 2)  # (1, H, 1, D) -> (1, 1, H, D)
        if config.v_quant_type == "PER_CHANNEL" and v_scale is not None:
            v_scale_bsnh = v_scale.transpose(1, 2)  # (1, H, 1, D) -> (1, 1, H, D)

        k_to_cache = dequantize_tensor(
            quantize_tensor_with_scale(k_ro, k_scale_bsnh, config.k_quant_type, config.kv_cache_type),
            k_scale_bsnh,
            config.k_quant_type,
            config.kv_cache_type,
        ).to(torch_type)
        v_to_cache = dequantize_tensor(
            quantize_tensor_with_scale(new_v, v_scale_bsnh, config.v_quant_type, config.kv_cache_type),
            v_scale_bsnh,
            config.v_quant_type,
            config.kv_cache_type,
        ).to(torch_type)

    k_cache_ref[update_mask] = rearrange(k_to_cache, "b s ... -> (b s) ...").to(k_cache_ref.dtype)
    v_cache_ref[update_mask] = rearrange(v_to_cache, "b s ... -> (b s) ...").to(v_cache_ref.dtype)
    key_padding_mask = arange < cache_seqlens_expanded + config.kv_sequence_length

    out_ref, _ = attention_ref(
        q=q_ro,
        k=k_cache_ref,
        v=v_cache_ref,
        key_padding_mask=key_padding_mask,
        attention_bias=attention_bias,
        causal=causal,
        window_size=window_size,
        softcap=config.softcap,
        use_smooth_softmax=config.use_smooth_softmax,
        head_sink=head_sink,
    )
    out_ref_np = out_ref.to(torch.float32).detach().cpu().numpy()

    # --- ONNX Runtime Path ---

    q_ort, new_k_ort, new_v_ort = q, new_k, new_v
    if config.packed:
        q_ort = torch.cat([q, new_k, new_v], dim=2)
        new_k_ort, new_v_ort = None, None

    # Quantize k and v for ORT when using quantized KV cache
    # Quantize k and v for ORT when using quantized KV cache
    k_ort = k
    v_ort = v
    if config.kv_cache_type in ["int8", "int4", "fp8"]:
        # NOTE: Quantize returns tensor with kv_cache_type (int8, int4, or fp8)
        k_ort = quantize_tensor_with_scale(k, k_scale, config.k_quant_type, config.kv_cache_type)
        v_ort = quantize_tensor_with_scale(v, v_scale, config.v_quant_type, config.kv_cache_type)

        # Ensure they are contiguous for binding
        k_ort = k_ort.contiguous()
        v_ort = v_ort.contiguous()

    ort_seqlens = cache_seqlens + config.kv_sequence_length - 1

    out, present_k, present_v = gqa_past_func(
        q=q_ort,
        k=k_ort,
        v=v_ort,
        config=config,
        new_k=new_k_ort,
        new_v=new_v_ort,
        cos=cos,
        sin=sin,
        seqlens_k=ort_seqlens.int(),
        position_ids=position_ids,
        attention_bias=attention_bias,
        head_sink=head_sink,
        k_scale=k_scale,
        v_scale=v_scale,
        ep=ep,
        device=device,
        share_buffer=config.share_buffer,
        ort_type=ort_type,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
    )
    out = torch.reshape(out, (config.batch_size, config.q_sequence_length, config.num_heads, config.head_size))
    out_np = out.to(torch.float32).detach().cpu().numpy()

    if enable_debug_print:
        print(f"[DEBUG] out_np non-zeros: {numpy.count_nonzero(out_np)} / {out_np.size}")
        print(f"[DEBUG] out_ref_np non-zeros: {numpy.count_nonzero(out_ref_np)} / {out_ref_np.size}")

    if numpy.count_nonzero(out_ref_np) > 0 and numpy.count_nonzero(out_np) == 0:
        raise RuntimeError("Output is all zeros")

    print_diff_statistics(torch.tensor(out_np - out_ref_np), "out")
    numpy.testing.assert_allclose(out_np, out_ref_np, rtol=rtol, atol=atol)

    # --- Comparison ---
    compare_kv = (config.k_quant_type == "NONE" and config.v_quant_type == "NONE") or (config.kv_cache_type == "fp8")
    if compare_kv:
        # Compare KV cache
        # Transpose reference back to BNSH to match ORT output
        k_cache_ref_np = k_cache_ref.transpose(1, 2).to(torch.float32).detach().cpu().numpy()
        v_cache_ref_np = v_cache_ref.transpose(1, 2).to(torch.float32).detach().cpu().numpy()

        if isinstance(present_k, torch.Tensor):
            present_k_torch = present_k.to(device)
            present_v_torch = present_v.to(device)
        else:
            present_k_torch = torch.from_numpy(present_k).to(device)
            present_v_torch = torch.from_numpy(present_v).to(device)

        if config.kv_cache_type == "fp8":
            # FP8 cache needs dequantization for comparison with float reference
            present_k_dequant = dequantize_tensor(present_k_torch, k_scale, config.k_quant_type, config.kv_cache_type)
            present_v_dequant = dequantize_tensor(present_v_torch, v_scale, config.v_quant_type, config.kv_cache_type)
            present_k_np = present_k_dequant.to(torch.float32).detach().cpu().numpy()
            present_v_np = present_v_dequant.to(torch.float32).detach().cpu().numpy()
        else:
            present_k_np = present_k_torch.to(torch.float32).detach().cpu().numpy()
            present_v_np = present_v_torch.to(torch.float32).detach().cpu().numpy()

        numpy.testing.assert_allclose(present_k_np, k_cache_ref_np, rtol=rtol, atol=atol)
        numpy.testing.assert_allclose(present_v_np, v_cache_ref_np, rtol=rtol, atol=atol)

    # Compare quantized cache with proper masking per batch
    if config.k_quant_type != "NONE":
        if isinstance(present_k, torch.Tensor):
            present_k_torch = present_k.to(device)
        else:
            if config.kv_cache_type == "int4":
                present_k_torch = torch.from_numpy(present_k).to(device)
            elif config.kv_cache_type == "int8":
                present_k_torch = torch.from_numpy(present_k.astype(numpy.int8)).to(device)
            elif config.kv_cache_type == "fp8":
                present_k_torch = torch.from_numpy(present_k).view(torch.float8_e4m3fn).to(device)
            else:
                present_k_torch = torch.from_numpy(present_k).to(device)

        present_k_dequant = (
            dequantize_tensor(present_k_torch, k_scale, config.k_quant_type, config.kv_cache_type)
            .detach()
            .cpu()
            .numpy()
        )

        # Mask the reference cache to only valid regions
        k_cache_ref_masked = k_cache_ref.transpose(1, 2).clone()
        total_seqlens = cache_seqlens + config.q_sequence_length
        arange = torch.arange(config.buffer_sequence_length, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        total_seqlens_expanded = total_seqlens.unsqueeze(1).unsqueeze(1).unsqueeze(-1)
        mask = arange >= total_seqlens_expanded
        k_cache_ref_masked[mask.expand_as(k_cache_ref_masked)] = 0
        k_cache_ref_dequant = k_cache_ref_masked.cpu().numpy()

        for b in range(config.batch_size):
            valid_len = (cache_seqlens[b] + config.q_sequence_length).item()
            print_diff_statistics(
                torch.tensor(present_k_dequant[b, :, :valid_len, :] - k_cache_ref_dequant[b, :, :valid_len, :]),
                f"present_k[{b}]",
            )
            numpy.testing.assert_allclose(
                present_k_dequant[b, :, :valid_len, :],
                k_cache_ref_dequant[b, :, :valid_len, :],
                rtol=rtol,
                atol=atol,
            )

    if config.v_quant_type != "NONE":
        if isinstance(present_v, torch.Tensor):
            present_v_torch = present_v.to(device)
        else:
            if config.kv_cache_type == "int4":
                present_v_torch = torch.from_numpy(present_v).to(device)
            elif config.kv_cache_type == "int8":
                present_v_torch = torch.from_numpy(present_v.astype(numpy.int8)).to(device)
            elif config.kv_cache_type == "fp8":
                present_v_torch = torch.from_numpy(present_v).view(torch.float8_e4m3fn).to(device)
            else:
                present_v_torch = torch.from_numpy(present_v).to(device)

        present_v_dequant = (
            dequantize_tensor(present_v_torch, v_scale, config.v_quant_type, config.kv_cache_type)
            .detach()
            .cpu()
            .numpy()
        )

        # Mask the reference cache to only valid regions
        v_cache_ref_masked = v_cache_ref.transpose(1, 2).clone()
        total_seqlens = cache_seqlens + config.q_sequence_length
        arange = torch.arange(config.buffer_sequence_length, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        total_seqlens_expanded = total_seqlens.unsqueeze(1).unsqueeze(1).unsqueeze(-1)
        mask = arange >= total_seqlens_expanded
        v_cache_ref_masked[mask.expand_as(v_cache_ref_masked)] = 0
        v_cache_ref_dequant = v_cache_ref_masked.cpu().numpy()

        for b in range(config.batch_size):
            valid_len = (cache_seqlens[b] + config.q_sequence_length).item()
            print_diff_statistics(
                torch.tensor(present_v_dequant[b, :, :valid_len, :] - v_cache_ref_dequant[b, :, :valid_len, :]),
                f"present_v[{b}]",
            )
            numpy.testing.assert_allclose(
                present_v_dequant[b, :, :valid_len, :],
                v_cache_ref_dequant[b, :, :valid_len, :],
                rtol=rtol,
                atol=atol,
            )


def parity_test_gqa_padding_prompt():
    device = "cuda"
    torch_type = torch.float16
    ort_type = TensorProto.FLOAT16

    # config
    config = GQAConfig(
        batch_size=2,
        q_sequence_length=16,
        kv_sequence_length=16,
        num_heads=8,
        kv_num_heads=2,
        head_size=128,
        buffer_sequence_length=16,
        share_buffer=True,
        packed=False,
        rotary=True,
    )

    # Inputs
    torch.manual_seed(0)
    std = 0.02
    q = (
        torch.randn(
            config.batch_size,
            config.q_sequence_length,
            config.num_heads,
            config.head_size,
            device=device,
            dtype=torch_type,
        )
        * std
    )
    k = (
        torch.randn(
            config.batch_size,
            config.kv_num_heads,
            config.kv_sequence_length,
            config.head_size,
            device=device,
            dtype=torch_type,
        )
        * std
    )
    v = torch.randn_like(k) * std

    new_k = k.transpose(1, 2).contiguous()
    new_v = v.transpose(1, 2).contiguous()

    seqlens_k = torch.tensor([9, 15], dtype=torch.int32, device=device)

    # Generate Rotary Embeddings
    rotary_dim = config.head_size
    max_seq_len = config.buffer_sequence_length
    cos = torch.randn(1, max_seq_len, 1, rotary_dim // 2, device=device, dtype=torch_type)
    sin = torch.randn(1, max_seq_len, 1, rotary_dim // 2, device=device, dtype=torch_type)

    # Apply Rotary to inputs for Reference
    rotary_op = LlamaMSRotaryEmbedding()
    pos = torch.zeros(config.batch_size, device=device, dtype=torch.long)

    # In ORT, we pass raw Q/K and ORT applies rotary.
    # For REF, we must apply rotary manually.
    # But wait, ORT only rotates 'q' and 'k' inside the attention kernel.
    # Wait, if `share_buffer=True`, `past_key` is used.
    # In prompt mode, `new_k` is appended to `past_key`.
    # ORT will apply rotary to Q.
    # Does ORT apply rotary to K? Yes, if `do_rotary` is true.
    # So we rotate Q and K for REF.

    q_ref = rotary_op.rotate_tensor(q, cos, sin, pos, False)
    k_ref = rotary_op.rotate_tensor(new_k, cos, sin, pos, False)
    v_ref = new_v

    # Run ONNX Runtime
    out_ort, present_key_ort, present_value_ort = gqa_prompt_func(
        q=q,
        k=k,
        v=v,
        config=config,
        new_k=new_k,
        new_v=new_v,
        cos=cos.squeeze(2).squeeze(0),
        sin=sin.squeeze(2).squeeze(0),
        seqlens_k=seqlens_k,
        position_ids=None,
        attention_bias=None,
        head_sink=None,
        k_scale=None,
        v_scale=None,
        ep="CUDAExecutionProvider",
        device=device,
        share_buffer=config.share_buffer,
        ort_type=ort_type,
    )

    # Compare present_key and present_value with reference
    # ORT present_key is BNSH format: [batch, kv_num_heads, seq, head_size]
    # k_ref is BSNH format: [batch, seq, kv_num_heads, head_size]
    # Transpose k_ref to BNSH for comparison
    k_ref_bnsh = k_ref.transpose(1, 2)  # BSNH -> BNSH
    v_ref_bnsh = v_ref.transpose(1, 2)  # BSNH -> BNSH

    # Compare only valid positions (positions 0..9 for Batch 0, 0..15 for Batch 1)
    torch.testing.assert_close(present_key_ort[0, :, :10, :], k_ref_bnsh[0, :, :10, :], rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(present_key_ort[1, :, :16, :], k_ref_bnsh[1, :, :16, :], rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(present_value_ort[0, :, :10, :], v_ref_bnsh[0, :, :10, :], rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(present_value_ort[1, :, :16, :], v_ref_bnsh[1, :, :16, :], rtol=1e-3, atol=1e-3)

    # Run Reference
    # key_padding_mask is a "Validity Mask" where True=Valid, False=Invalid
    key_padding_mask = torch.zeros((config.batch_size, config.q_sequence_length), dtype=torch.bool, device=device)

    # Batch 0: Valid 0..9 (length 10)
    key_padding_mask[0, :10] = True

    # Batch 1: Valid 0..15 (length 16)
    key_padding_mask[1, :16] = True

    out_ref, _ = attention_ref(
        q_ref, k_ref, v_ref, key_padding_mask=key_padding_mask, query_padding_mask=key_padding_mask, causal=True
    )

    # Compare
    # Batch 0: 10..15 are padding
    out_ort[0, 10:] = 0
    out_ref[0, 10:] = 0

    # Reshape ref to match ORT
    out_ref = out_ref.reshape(config.batch_size, config.q_sequence_length, -1)

    # Debugging
    diff = (out_ort - out_ref).abs()
    max_diff = diff.max()
    # Check Batch 0
    b0_diff = diff[0].max()
    # Check Batch 1
    b1_diff = diff[1].max()

    if not torch.allclose(out_ort, out_ref, rtol=1e-2, atol=1e-2):
        msg = f"Mismatch! Max Diff: {max_diff}, Batch 0 Max: {b0_diff}, Batch 1 Max: {b1_diff}\n"
        raise AssertionError(msg)

    torch.testing.assert_close(out_ort, out_ref, rtol=1e-2, atol=1e-2)


# #################################################################################################
#  Test Utilities
# #################################################################################################


def print_diff_statistics(diff_tensor: torch.Tensor, prefix: str = ""):
    """
    Print percentile statistics (75%, 95%, 99%) for a difference tensor.
    This helps assess parity quality beyond just max difference.

    Args:
        diff_tensor: Tensor containing absolute differences between expected and actual outputs.
        prefix: Optional prefix string for the output message.
    """
    if not enable_debug_print:
        return

    diff_flat = diff_tensor.flatten().float()
    if diff_flat.numel() == 0:
        print(f"{prefix}Diff statistics: empty tensor")
        return

    # Compute percentiles
    sorted_diff, _ = torch.sort(diff_flat)
    n = sorted_diff.numel()

    p75_idx = min(int(n * 0.75), n - 1)
    p90_idx = min(int(n * 0.90), n - 1)
    p95_idx = min(int(n * 0.95), n - 1)
    p99_idx = min(int(n * 0.99), n - 1)
    p999_idx = min(int(n * 0.999), n - 1)

    p75 = sorted_diff[p75_idx].item()
    p90 = sorted_diff[p90_idx].item()
    p95 = sorted_diff[p95_idx].item()
    p99 = sorted_diff[p99_idx].item()
    p999 = sorted_diff[p999_idx].item()
    max_val = sorted_diff[-1].item()
    mean_val = diff_flat.mean().item()

    print(
        f"{prefix} Diff stats - mean: {mean_val:.6f}, p75: {p75:.6f}, p90: {p90:.6f}, p95: {p95:.6f}, p99: {p99:.6f}, p999: {p999:.6f}, max: {max_val:.6f}"
    )


# #################################################################################################
#  Test Case Generators
# #################################################################################################


def get_cuda_rotary_options():
    return [(False, False), (True, False), (True, True)]


def get_cpu_rotary_options():
    return [(False, False), (True, False), (True, True)]


def get_softmax_options(allow_head_sink: bool = True):
    options = [(False, False), (False, True), (True, False)]
    if not allow_head_sink:
        options = [opt for opt in options if not opt[1]]
    return options


def gqa_cuda_prompt_test_cases(allow_head_sink: bool = True, allow_local: bool = True):
    batches = [3, 1, 5]
    seqs = [(35, 35), (1, 1), (64, 64), (128, 128), (240, 240), (2000, 2000)]
    heads = [(6, 3), (3, 1), (32, 8)]
    h_sizes = [128] if quick_build else [128, 32, 64, 80, 160, 256]
    smmoth_softmax__head_sink = get_softmax_options(allow_head_sink)

    rotary_opts = list(get_cuda_rotary_options())
    packed_opts = [False, True]
    share_buffer_opts = [True, False]
    softcap_opts = [0.0, 50.0]

    # Use new strategy for both modes: iterate over key code path parameters
    # The difference between modes is the number of head_sizes tested
    # Pipeline mode: h_sizes[:1] = [128] -> 12 combinations (fast)
    # Comprehensive mode: all h_sizes -> 40+ combinations (thorough)
    h_sizes_to_test = h_sizes[:1] if pipeline_mode else h_sizes

    combo_index = 0
    for h in h_sizes_to_test:
        for packed in packed_opts:
            for rotary, rotary_interleaved in rotary_opts:
                # Skip invalid: rotary requires head_size divisible by 16
                if rotary and h % 16 > 0:
                    continue

                for share_buffer in share_buffer_opts:
                    # Rotate secondary parameters
                    b = batches[combo_index % len(batches)]
                    sq, skv = seqs[combo_index % len(seqs)]
                    n, n2 = heads[combo_index % len(heads)]
                    lws_opts = [-1, max(1, skv // 2)] if allow_local else [-1]
                    lws = lws_opts[combo_index % len(lws_opts)]
                    softcap = softcap_opts[combo_index % len(softcap_opts)]
                    use_smooth_softmax, has_head_sink = smmoth_softmax__head_sink[
                        combo_index % len(smmoth_softmax__head_sink)
                    ]
                    has_position_ids = False if pipeline_mode else combo_index % 2 == 0

                    combo_index += 1

                    if softcap > 0 and (use_smooth_softmax or has_head_sink):
                        continue

                    config = GQAConfig(
                        batch_size=b,
                        q_sequence_length=sq,
                        kv_sequence_length=skv,
                        past_kv_sequence_length=0,
                        buffer_sequence_length=sq + skv + 8,
                        num_heads=n,
                        kv_num_heads=n2,
                        head_size=h,
                        local_window_size=lws,
                        rotary=rotary,
                        rotary_interleaved=rotary_interleaved,
                        packed=packed,
                        share_buffer=share_buffer,
                        softcap=softcap,
                        use_smooth_softmax=use_smooth_softmax,
                        has_head_sink=has_head_sink,
                        has_position_ids=has_position_ids,
                    )
                    name = f"b{b}_sq{sq}_skv{skv}_nh{n}_{n2}_h{h}_w{lws}_rot{rotary}{rotary_interleaved}_pkd{packed}_sb{share_buffer}_sc{softcap}_sm{use_smooth_softmax}_{has_head_sink}_pid{has_position_ids}"
                    yield name, config


def gqa_cuda_past_test_cases(
    allow_head_sink: bool = True, allow_local: bool = True, enforce_share_buffer: bool = False
):
    batches = [2, 1, 3]
    # s: new sequence length, s2: past sequence length``
    seqs = [(1, 1), (1, 128), (1, 2048), (1, 5000)]
    subsequent_prompt_seqs = [(3, 256)]
    heads = [(32, 8), (6, 3), (9, 9)]
    h_sizes = [128] if quick_build else [128, 40, 64, 80, 256]
    smmoth_softmax__head_sink = get_softmax_options(allow_head_sink)

    rotary_opts = list(get_cuda_rotary_options())
    packed_opts = [False, True]
    # For past test: pipeline tests share_buffer=True only, comprehensive tests both
    share_buffer_opts = [True] if pipeline_mode or enforce_share_buffer else [True, False]
    softcap_opts = [0.0, 50.0]

    # Use new strategy for both modes: iterate over key code path parameters
    # The difference between modes is the number of head_sizes tested
    # Pipeline mode: h_sizes[:1] = [128] -> 6 combinations (share_buffer=[True] only)
    # Comprehensive mode: all h_sizes -> 36+ combinations
    h_sizes_to_test = h_sizes[:1] if pipeline_mode else h_sizes
    all_seqs = seqs + subsequent_prompt_seqs

    combo_index = 0
    for h in h_sizes_to_test:
        for packed in packed_opts:
            for rotary, rotary_interleaved in rotary_opts:
                # Skip invalid: rotary requires head_size divisible by 16
                if rotary and h % 16 > 0:
                    continue

                for share_buffer in share_buffer_opts:
                    # Rotate secondary parameters
                    b = batches[combo_index % len(batches)]
                    s, s2 = all_seqs[combo_index % len(all_seqs)]

                    # Skip subsequent prompt for batch > 1
                    if s > 1 and b > 1:
                        b = 1  # Force batch=1 for subsequent prompt

                    n, n2 = heads[combo_index % len(heads)]
                    lws_opts = [-1, max(1, s2 // 2)] if allow_local else [-1]
                    lws = lws_opts[combo_index % len(lws_opts)]
                    softcap = softcap_opts[combo_index % len(softcap_opts)]
                    use_smooth_softmax, has_head_sink = smmoth_softmax__head_sink[
                        combo_index % len(smmoth_softmax__head_sink)
                    ]
                    has_position_ids = False if pipeline_mode else s > 1

                    combo_index += 1

                    if softcap > 0 and (use_smooth_softmax or has_head_sink):
                        continue

                    config = GQAConfig(
                        batch_size=b,
                        q_sequence_length=s,
                        kv_sequence_length=s,
                        past_kv_sequence_length=s2,
                        buffer_sequence_length=s + s2 + 8,
                        num_heads=n,
                        kv_num_heads=n2,
                        head_size=h,
                        local_window_size=lws,
                        rotary=rotary,
                        rotary_interleaved=rotary_interleaved,
                        packed=packed,
                        share_buffer=share_buffer,
                        softcap=softcap,
                        use_smooth_softmax=use_smooth_softmax,
                        has_head_sink=has_head_sink,
                        has_position_ids=has_position_ids,
                    )
                    name = f"b{b}_s{s}_{s2}_nh{n}_{n2}_h{h}_w{lws}_rot{rotary}{rotary_interleaved}_pkd{packed}_sb{share_buffer}_sc{softcap}_sm{use_smooth_softmax}_{has_head_sink}_pid{has_position_ids}"
                    yield name, config


def gqa_cuda_attention_bias_test_cases(is_past: bool):
    """Focused cases for the attention_bias input. Dispatch must route these away from
    the bias-incapable fused paths (flash/XQA/cuDNN), so no kernel-pinning env vars are
    needed. Covers packed/unpacked QKV, shared/separate KV buffer, rotary, odd head
    sizes (unfused supports any head_size) and a subsequent multi-token prompt."""
    if is_past:
        # (batch, new_seq, past_seq, num_heads, kv_num_heads, head_size, packed, share_buffer,
        #  rotary, per_head_bias)
        cases = [
            (2, 1, 128, 32, 8, 128, False, True, False, False),
            (1, 1, 2048, 6, 3, 128, True, True, True, False),
            (3, 1, 500, 9, 9, 80, False, False, False, False),
            (1, 3, 256, 6, 3, 128, True, True, False, False),  # subsequent prompt
            (2, 1, 128, 6, 3, 40, False, True, False, False),  # odd head size
            (2, 1, 128, 32, 8, 128, False, True, False, True),  # per-head bias (dim 1 == num_heads)
        ]
        cases = [(*c, False) for c in cases] + [
            (3, 1, 500, 9, 9, 80, False, False, False, False, True),  # batch-broadcast bias (dim 0 == 1)
            (2, 1, 128, 6, 3, 40, False, True, False, False, True),  # batch-broadcast, shared buffer
        ]
        for b, s, s2, n, n2, h, packed, share_buffer, rotary, per_head, bcast0 in cases:
            # The past-parity harness compares the full present buffer; without buffer
            # sharing the op emits exactly past+new, so the buffer must match that size.
            config = GQAConfig(
                batch_size=b,
                q_sequence_length=s,
                kv_sequence_length=s,
                past_kv_sequence_length=s2,
                buffer_sequence_length=s + s2 + (8 if share_buffer else 0),
                num_heads=n,
                kv_num_heads=n2,
                head_size=h,
                rotary=rotary,
                packed=packed,
                share_buffer=share_buffer,
                has_attention_bias=True,
                attention_bias_per_head=per_head,
                attention_bias_broadcast_dim_0=bcast0,
            )
            name = (
                f"bias_b{b}_s{s}_{s2}_nh{n}_{n2}_h{h}_pkd{packed}_sb{share_buffer}_rot{rotary}_ph{per_head}_bc0{bcast0}"
            )
            yield name, config
    else:
        # (batch, seq, num_heads, kv_num_heads, head_size, packed, share_buffer, rotary,
        #  bias_broadcast_dim_0, per_head_bias)
        cases = [
            (2, 127, 6, 3, 128, False, True, False, False, False),
            (1, 500, 32, 8, 128, True, True, True, False, False),
            (3, 64, 9, 9, 80, False, False, False, False, False),
            (2, 127, 6, 3, 40, True, True, False, False, False),  # odd head size
            (3, 64, 9, 9, 80, False, False, False, True, False),  # batch-broadcast bias (dim 0 == 1)
            (2, 127, 6, 3, 128, False, True, False, False, True),  # per-head bias (dim 1 == num_heads)
        ]
        for b, sq, n, n2, h, packed, share_buffer, rotary, bcast0, per_head in cases:
            config = GQAConfig(
                batch_size=b,
                q_sequence_length=sq,
                kv_sequence_length=sq,
                past_kv_sequence_length=0,
                buffer_sequence_length=sq + 8,
                num_heads=n,
                kv_num_heads=n2,
                head_size=h,
                rotary=rotary,
                packed=packed,
                share_buffer=share_buffer,
                has_attention_bias=True,
                attention_bias_broadcast_dim_0=bcast0,
                attention_bias_per_head=per_head,
            )
            name = f"bias_b{b}_sq{sq}_nh{n}_{n2}_h{h}_pkd{packed}_sb{share_buffer}_rot{rotary}_bc0{bcast0}_ph{per_head}"
            yield name, config


def gqa_cuda_quantized_test_cases(is_past: bool):
    base_cases = (
        gqa_cuda_past_test_cases(allow_local=True, enforce_share_buffer=True)
        if is_past
        else gqa_cuda_prompt_test_cases(allow_local=True)
    )

    kv_types = ["int8"]
    if has_int4_kv_cache:
        kv_types.append("int4")
    if has_fp8_kv_cache:
        kv_types.append("fp8")

    for name, config in base_cases:
        for kv_type in kv_types:
            for quant_mode in ["PER_TENSOR", "PER_CHANNEL"]:
                share_scales_options = [False]
                if quant_mode == "PER_TENSOR" and kv_type == "int8":
                    share_scales_options = [True]

                for share_scales in share_scales_options:
                    q_config = deepcopy(config)
                    q_config.k_quant_type = quant_mode
                    q_config.v_quant_type = quant_mode
                    q_config.kv_cache_type = kv_type
                    q_config.share_kv_scale = share_scales

                    if kv_type == "int4":
                        if q_config.head_size % 2 != 0:
                            continue
                        q_config.kv_cache_bit_width = 4
                    elif kv_type == "int8":
                        q_config.kv_cache_bit_width = 8
                    elif kv_type == "fp8":
                        q_config.kv_cache_bit_width = 8

                    q_name = f"{name}_quant_{kv_type}_{quant_mode}"
                    if share_scales:
                        q_name += "_shared"
                    yield q_name, q_config


# #################################################################################################
#  Unit Test Classes
# #################################################################################################


def has_cuda_provider():
    return get_cuda_provider_name() is not None


def has_cuda_device(min_capability: int = 80):
    if not has_cuda_provider() or not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor >= min_capability


def has_flash_attention(bf16=False):
    if not has_cuda_device(80):
        return False
    if bf16:
        return torch.cuda.is_bf16_supported()
    return True


def has_xqa():
    # The XQA decode kernels require Ampere (SM 8.0) or newer.
    return has_cuda_device(80)


def has_fp8_xqa():
    # FP8 XQA kernels require Hopper (SM 8.9) or newer.
    return has_cuda_device(89)


rtol = {
    "fp16": 5e-3,
    "bf16": 5e-2,
    "int8_fp16": 5e-2,
    "int4_fp16": 5e-2,
    "int8_bf16": 5e-2,
    "int4_bf16": 5e-2,
    "fp8_fp16": 5e-2,
    "fp8_bf16": 5e-2,
}
atol = {
    "fp16": 5e-3,
    "bf16": 1e-2,
    "int8_fp16": 1e-1,
    "int4_fp16": 1e-1,
    "int8_bf16": 2e-1,
    "int4_bf16": 2e-1,
    "fp8_fp16": 1e-1,
    "fp8_bf16": 2e-1,
}


def has_quantized_kv_cache():
    return version.parse(ort_version) >= version.parse("1.25.0")


@unittest.skipIf(not has_flash_attention(), "Flash Attention is not available, skipping tests.")
class TestFlashGQA(unittest.TestCase):
    def tearDown(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

    @parameterized.expand(gqa_cuda_prompt_test_cases())
    def test_gqa_prompt_flash_attention(self, name, config):
        if enable_debug_print:
            print("-" * 20)
            print(f"test_case: {name}\n{config}")

        with scoped_env_var("ORT_DISABLE_FLASH_ATTENTION", "0"):
            parity_check_gqa_prompt(
                config=config,
                ep="CUDAExecutionProvider",
                device="cuda",
                torch_type=torch.float16,
                ort_type=TensorProto.FLOAT16,
                causal=True,
                rtol=rtol["fp16"],
                atol=atol["fp16"],
            )

    def test_gqa_prompt_feature_interactions(self):
        config = GQAConfig(
            batch_size=2,
            q_sequence_length=4,
            kv_sequence_length=4,
            buffer_sequence_length=12,
            num_heads=4,
            kv_num_heads=2,
            head_size=64,
            rotary=True,
            rotary_interleaved=True,
            softcap=2.0,
            has_attention_bias=True,
            attention_bias_per_head=True,
            has_qk_norm=True,
            share_buffer=True,
        )

        with scoped_env_var("ORT_DISABLE_FLASH_ATTENTION", "0"):
            parity_check_gqa_prompt(
                config=config,
                ep="CUDAExecutionProvider",
                device="cuda",
                torch_type=torch.float16,
                ort_type=TensorProto.FLOAT16,
                causal=True,
                rtol=rtol["fp16"],
                atol=atol["fp16"],
            )

    def test_gqa_prompt_bidirectional_attention_bias_broadcast(self):
        config = GQAConfig(
            batch_size=2,
            q_sequence_length=4,
            kv_sequence_length=4,
            buffer_sequence_length=12,
            num_heads=4,
            kv_num_heads=2,
            head_size=64,
            rotary=True,
            has_attention_bias=True,
            attention_bias_broadcast_dim_0=True,
            attention_bias_per_head=False,
            share_buffer=True,
        )

        with scoped_env_var("ORT_DISABLE_FLASH_ATTENTION", "0"):
            parity_check_gqa_prompt(
                config=config,
                ep="CUDAExecutionProvider",
                device="cuda",
                torch_type=torch.float16,
                ort_type=TensorProto.FLOAT16,
                causal=False,
                rtol=rtol["fp16"],
                atol=atol["fp16"],
            )

    def test_gqa_decode_rotary_head_sink_softcap_qk_norm_batch(self):
        config = GQAConfig(
            batch_size=2,
            q_sequence_length=1,
            kv_sequence_length=1,
            past_kv_sequence_length=63,
            buffer_sequence_length=72,
            num_heads=8,
            kv_num_heads=2,
            head_size=64,
            rotary=True,
            softcap=2.0,
            has_head_sink=True,
            has_qk_norm=True,
            share_buffer=True,
        )

        with scoped_env_var("ORT_DISABLE_FLASH_ATTENTION", "0"):
            parity_check_gqa_past(
                config=config,
                ep="CUDAExecutionProvider",
                device="cuda",
                torch_type=torch.float16,
                ort_type=TensorProto.FLOAT16,
                causal=True,
                rtol=rtol["fp16"],
                atol=atol["fp16"],
            )

    def test_gqa_decode_bidirectional(self):
        config = GQAConfig(
            batch_size=2,
            q_sequence_length=2,
            kv_sequence_length=2,
            past_kv_sequence_length=4,
            buffer_sequence_length=8,
            num_heads=4,
            kv_num_heads=2,
            head_size=64,
            share_buffer=True,
        )

        with scoped_env_var("ORT_DISABLE_FLASH_ATTENTION", "0"):
            parity_check_gqa_past(
                config=config,
                ep="CUDAExecutionProvider",
                device="cuda",
                torch_type=torch.float16,
                ort_type=TensorProto.FLOAT16,
                causal=False,
                rtol=rtol["fp16"],
                atol=atol["fp16"],
            )

    @unittest.skipUnless(has_quantized_kv_cache(), "Quantized KV cache is not available")
    def test_gqa_decode_rotary_quantized_head_sink_batch(self):
        config = GQAConfig(
            batch_size=2,
            q_sequence_length=1,
            kv_sequence_length=1,
            past_kv_sequence_length=63,
            buffer_sequence_length=72,
            num_heads=8,
            kv_num_heads=2,
            head_size=64,
            rotary=True,
            has_head_sink=True,
            k_quant_type="PER_TENSOR",
            v_quant_type="PER_CHANNEL",
            kv_cache_type="int8",
            kv_cache_bit_width=8,
            share_buffer=True,
        )

        parity_check_gqa_past(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            ort_type=TensorProto.FLOAT16,
            causal=True,
            rtol=rtol["int8_fp16"],
            atol=atol["int8_fp16"],
        )

    @parameterized.expand(gqa_cuda_past_test_cases())
    def test_gqa_past_flash_attention(self, name, config):
        if enable_debug_print:
            print("-" * 20)
            print(f"test_case: {name}\n{config}")

        with scoped_env_var("ORT_DISABLE_FLASH_ATTENTION", "0"):
            parity_check_gqa_past(
                config=config,
                ep="CUDAExecutionProvider",
                device="cuda",
                torch_type=torch.float16,
                ort_type=TensorProto.FLOAT16,
                causal=True,
                rtol=rtol["fp16"],
                atol=atol["fp16"],
            )


@unittest.skipIf(not has_cuda_device(53), "Quantized bidirectional GQA requires a CUDA GPU, skipping tests.")
@unittest.skipUnless(has_quantized_kv_cache(), "Quantized KV cache is not available")
class TestQuantizedBidirectionalGQA(unittest.TestCase):
    @staticmethod
    def _config():
        return GQAConfig(
            batch_size=1,
            q_sequence_length=2,
            kv_sequence_length=2,
            past_kv_sequence_length=4,
            buffer_sequence_length=8,
            num_heads=4,
            kv_num_heads=2,
            head_size=64,
            k_quant_type="PER_TENSOR",
            v_quant_type="PER_TENSOR",
            kv_cache_type="int8",
            kv_cache_bit_width=8,
            share_buffer=True,
        )

    @unittest.skipIf(not has_flash_attention(), "Flash Attention is not available")
    def test_gqa_past_flash_attention(self):
        with scoped_env_var("ORT_DISABLE_FLASH_ATTENTION", "0"):
            parity_check_gqa_past(
                config=self._config(),
                ep="CUDAExecutionProvider",
                device="cuda",
                torch_type=torch.float16,
                ort_type=TensorProto.FLOAT16,
                causal=False,
                rtol=rtol["int8_fp16"],
                atol=atol["int8_fp16"],
            )

    def test_gqa_past_without_supported_backend(self):
        with (
            scoped_env_var("ORT_DISABLE_FLASH_ATTENTION", "1"),
            self.assertRaisesRegex(Exception, "No available GroupQueryAttention kernel supports"),
        ):
            parity_check_gqa_past(
                config=self._config(),
                ep="CUDAExecutionProvider",
                device="cuda",
                torch_type=torch.float16,
                ort_type=TensorProto.FLOAT16,
                causal=False,
                rtol=rtol["int8_fp16"],
                atol=atol["int8_fp16"],
            )


@unittest.skipIf(not has_flash_attention(bf16=True), "Flash Attention is not available, skipping tests.")
class TestFlashGQABF16(unittest.TestCase):
    def tearDown(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

    @parameterized.expand(gqa_cuda_prompt_test_cases())
    def test_gqa_prompt_flash_attention_bf16(self, name, config):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BFloat16 not supported on this device")

        if enable_debug_print:
            print("-" * 20)
            print(f"test_case: {name}\n{config}")

        config.kv_cache_type = "bfloat16"
        with scoped_env_var("ORT_DISABLE_FLASH_ATTENTION", "0"):
            parity_check_gqa_prompt(
                config=config,
                ep="CUDAExecutionProvider",
                device="cuda",
                torch_type=torch.bfloat16,
                ort_type=TensorProto.BFLOAT16,
                causal=True,
                rtol=rtol["bf16"],
                atol=atol["bf16"],
            )

    @parameterized.expand(gqa_cuda_past_test_cases())
    def test_gqa_past_flash_attention_bf16(self, name, config):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BFloat16 not supported on this device")

        if enable_debug_print:
            print("-" * 20)
            print(f"test_case: {name}\n{config}")

        config.kv_cache_type = "bfloat16"
        with scoped_env_var("ORT_DISABLE_FLASH_ATTENTION", "0"):
            parity_check_gqa_past(
                config=config,
                ep="CUDAExecutionProvider",
                device="cuda",
                torch_type=torch.bfloat16,
                ort_type=TensorProto.BFLOAT16,
                causal=True,
                rtol=rtol["bf16"],
                atol=atol["bf16"],
            )


def gqa_qk_norm_test_cases(is_past: bool):
    """Configs exercising the fused per-head Q/K RMSNorm (QK-Norm) prologue before RoPE."""
    head_sizes = [64, 128]
    head_groups = [(8, 2), (4, 4)]
    rotary_opts = [(False, False), (True, False), (True, True)]
    packed_opts = [False, True]
    idx = 0
    for h in head_sizes:
        for n, n2 in head_groups:
            for rotary, interleaved in rotary_opts:
                if rotary and h % 16 != 0:
                    continue
                packed = packed_opts[idx % len(packed_opts)]
                idx += 1
                if is_past:
                    b, s, s2 = 2, 1, 127
                    config = GQAConfig(
                        batch_size=b,
                        q_sequence_length=s,
                        kv_sequence_length=s,
                        past_kv_sequence_length=s2,
                        buffer_sequence_length=s + s2 + 8,
                        num_heads=n,
                        kv_num_heads=n2,
                        head_size=h,
                        rotary=rotary,
                        rotary_interleaved=interleaved,
                        packed=packed,
                        share_buffer=True,
                        has_qk_norm=True,
                    )
                else:
                    b, s = 2, 64
                    config = GQAConfig(
                        batch_size=b,
                        q_sequence_length=s,
                        kv_sequence_length=s,
                        buffer_sequence_length=s + 8,
                        num_heads=n,
                        kv_num_heads=n2,
                        head_size=h,
                        rotary=rotary,
                        rotary_interleaved=interleaved,
                        packed=packed,
                        share_buffer=True,
                        has_qk_norm=True,
                    )
                name = f"{'past' if is_past else 'prompt'}_b{b}_nh{n}_{n2}_h{h}_rot{rotary}{interleaved}_pkd{packed}"
                yield name, config


@unittest.skipIf(not has_cuda_device(80), "CUDA GQA QK-Norm requires Ampere or higher GPU, skipping tests.")
class TestGQAQKNorm(unittest.TestCase):
    def tearDown(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

    @parameterized.expand(gqa_qk_norm_test_cases(is_past=False))
    def test_gqa_qk_norm_prompt(self, name, config):
        if enable_debug_print:
            print("-" * 20)
            print(f"test_case: {name}\n{config}")

        with scoped_env_var("ORT_DISABLE_FLASH_ATTENTION", "0"):
            parity_check_gqa_prompt(
                config=config,
                ep="CUDAExecutionProvider",
                device="cuda",
                torch_type=torch.float16,
                ort_type=TensorProto.FLOAT16,
                causal=True,
                rtol=rtol["fp16"],
                atol=atol["fp16"],
            )

    @parameterized.expand(gqa_qk_norm_test_cases(is_past=True))
    def test_gqa_qk_norm_past(self, name, config):
        if enable_debug_print:
            print("-" * 20)
            print(f"test_case: {name}\n{config}")

        with scoped_env_var("ORT_DISABLE_FLASH_ATTENTION", "0"):
            parity_check_gqa_past(
                config=config,
                ep="CUDAExecutionProvider",
                device="cuda",
                torch_type=torch.float16,
                ort_type=TensorProto.FLOAT16,
                causal=True,
                rtol=rtol["fp16"],
                atol=atol["fp16"],
            )

    def test_gqa_qk_norm_past_xqa(self):
        config = GQAConfig(
            batch_size=2,
            q_sequence_length=1,
            kv_sequence_length=1,
            past_kv_sequence_length=127,
            buffer_sequence_length=136,
            num_heads=32,
            kv_num_heads=8,
            head_size=128,
            rotary=True,
            rotary_interleaved=False,
            packed=False,
            share_buffer=True,
            has_qk_norm=True,
        )

        parity_check_gqa_past(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            ort_type=TensorProto.FLOAT16,
            causal=True,
            rtol=rtol["fp16"],
            atol=atol["fp16"],
        )

    def test_gqa_qk_norm_past_shared_kv(self):
        config = GQAConfig(
            batch_size=2,
            q_sequence_length=1,
            kv_sequence_length=0,
            past_kv_sequence_length=127,
            buffer_sequence_length=135,
            num_heads=8,
            kv_num_heads=2,
            head_size=64,
            rotary=False,
            packed=False,
            share_buffer=True,
            has_qk_norm=True,
        )

        parity_check_gqa_past(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            ort_type=TensorProto.FLOAT16,
            causal=True,
            rtol=rtol["fp16"],
            atol=atol["fp16"],
        )

    def test_gqa_qk_norm_past_xqa_bf16(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BFloat16 not supported on this device")

        config = GQAConfig(
            batch_size=2,
            q_sequence_length=1,
            kv_sequence_length=1,
            past_kv_sequence_length=127,
            buffer_sequence_length=136,
            num_heads=32,
            kv_num_heads=8,
            head_size=128,
            rotary=True,
            rotary_interleaved=False,
            packed=False,
            share_buffer=True,
            has_qk_norm=True,
        )
        config.kv_cache_type = "bfloat16"

        parity_check_gqa_past(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.bfloat16,
            ort_type=TensorProto.BFLOAT16,
            causal=True,
            rtol=rtol["bf16"],
            atol=atol["bf16"],
        )

    @parameterized.expand(gqa_qk_norm_test_cases(is_past=True))
    def test_gqa_qk_norm_past_bf16(self, name, config):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BFloat16 not supported on this device")

        if enable_debug_print:
            print("-" * 20)
            print(f"test_case: {name}\n{config}")

        config.kv_cache_type = "bfloat16"
        with scoped_env_var("ORT_DISABLE_FLASH_ATTENTION", "0"):
            parity_check_gqa_past(
                config=config,
                ep="CUDAExecutionProvider",
                device="cuda",
                torch_type=torch.bfloat16,
                ort_type=TensorProto.BFLOAT16,
                causal=True,
                rtol=rtol["bf16"],
                atol=atol["bf16"],
            )


@unittest.skipIf(not has_flash_attention(), "Flash Attention is not available, skipping tests.")
@unittest.skipIf(not has_quantized_kv_cache(), "Quantized KV Cache is not available, skipping tests.")
class TestFlashGQABF16QuantizedKV(unittest.TestCase):
    def manual_seed(self):
        # Reset random seeds before each test to ensure test isolation
        torch.manual_seed(0)
        random.seed(69)
        numpy.random.seed(42)

    def setUp(self):
        self.manual_seed()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def tearDown(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

    @parameterized.expand(gqa_cuda_quantized_test_cases(is_past=False))
    def test_gqa_quantized_prompt_bf16(self, name, config):
        if enable_debug_print:
            print("-" * 20)
            print(f"test_case: {name}\n{config}")

        self.manual_seed()

        with scoped_env_var("ORT_DISABLE_FLASH_ATTENTION", "0"):
            parity_check_gqa_prompt(
                config=config,
                ep="CUDAExecutionProvider",
                device="cuda",
                torch_type=torch.bfloat16,
                ort_type=TensorProto.BFLOAT16,
                causal=True,
                rtol=rtol[f"{config.kv_cache_type}_bf16"],
                atol=atol[f"{config.kv_cache_type}_bf16"],
            )

    @parameterized.expand(gqa_cuda_quantized_test_cases(is_past=True))
    def test_gqa_quantized_past_bf16(self, name, config):
        if enable_debug_print:
            print("-" * 20)
            print(f"test_case: {name}\n{config}")

        self.manual_seed()

        with scoped_env_var("ORT_DISABLE_FLASH_ATTENTION", "0"):
            parity_check_gqa_past(
                config=config,
                ep="CUDAExecutionProvider",
                device="cuda",
                torch_type=torch.bfloat16,
                ort_type=TensorProto.BFLOAT16,
                causal=True,
                rtol=rtol[f"{config.kv_cache_type}_bf16"],
                atol=atol[f"{config.kv_cache_type}_bf16"],
            )


@unittest.skipIf(not has_cuda_device(53), "Memory Efficient Attention is not available, skipping tests.")
class TestMemoryEfficientGQA(unittest.TestCase):
    @parameterized.expand(gqa_cuda_prompt_test_cases(allow_head_sink=False))
    def test_gqa_prompt_memory_efficient(self, name, config):
        if enable_debug_print:
            print("-" * 20)
            print(f"test_case: {name}\n{config}")

        with scoped_env_var("ORT_DISABLE_FLASH_ATTENTION", "1"):
            parity_check_gqa_prompt(
                config=config,
                ep="CUDAExecutionProvider",
                device="cuda",
                torch_type=torch.float16,
                ort_type=TensorProto.FLOAT16,
                causal=True,
                rtol=rtol["fp16"],
                atol=atol["fp16"],
            )

    @parameterized.expand(gqa_cuda_past_test_cases(allow_head_sink=False))
    def test_gqa_past_memory_efficient(self, name, config):
        if enable_debug_print:
            print("-" * 20)
            print(f"test_case: {name}\n{config}")

        with scoped_env_var("ORT_DISABLE_FLASH_ATTENTION", "1"):
            parity_check_gqa_past(
                config=config,
                ep="CUDAExecutionProvider",
                device="cuda",
                torch_type=torch.float16,
                ort_type=TensorProto.FLOAT16,
                causal=True,
                rtol=rtol["fp16"],
                atol=atol["fp16"],
            )


@unittest.skipIf(not has_cuda_device(80), "BF16 requires Ampere or higher GPU, skipping tests.")
class TestBF16MemoryEfficientGQA(unittest.TestCase):
    @parameterized.expand(gqa_cuda_past_test_cases(allow_head_sink=False))
    def test_gqa_past_memory_efficient_bf16(self, name, config):
        if enable_debug_print:
            print("-" * 20)
            print(f"test_case: {name}\n{config}")

        with scoped_env_var("ORT_DISABLE_FLASH_ATTENTION", "1"):
            parity_check_gqa_past(
                config=config,
                ep="CUDAExecutionProvider",
                device="cuda",
                torch_type=torch.bfloat16,
                ort_type=TensorProto.BFLOAT16,
                causal=True,
                rtol=rtol["bf16"],
                atol=atol["bf16"],
            )


@unittest.skipIf(not has_cuda_device(53), "attention_bias CUDA parity tests require a CUDA GPU, skipping tests.")
class TestGQAAttentionBias(unittest.TestCase):
    """Parity for the optional attention_bias input (CUDA). No kernel-pinning env vars:
    bias-aware dispatch itself must route to a bias-capable path."""

    @parameterized.expand(gqa_cuda_attention_bias_test_cases(is_past=False))
    def test_gqa_prompt_attention_bias(self, name, config):
        if enable_debug_print:
            print("-" * 20)
            print(f"test_case: {name}\n{config}")

        parity_check_gqa_prompt(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            ort_type=TensorProto.FLOAT16,
            causal=True,
            rtol=rtol["fp16"],
            atol=atol["fp16"],
        )

    @parameterized.expand(gqa_cuda_attention_bias_test_cases(is_past=True))
    def test_gqa_past_attention_bias(self, name, config):
        if enable_debug_print:
            print("-" * 20)
            print(f"test_case: {name}\n{config}")

        parity_check_gqa_past(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            ort_type=TensorProto.FLOAT16,
            causal=True,
            rtol=rtol["fp16"],
            atol=atol["fp16"],
        )

    def test_windowed_cache_attention_bias_is_rejected(self):
        config = GQAConfig(
            batch_size=1,
            q_sequence_length=1,
            kv_sequence_length=1,
            past_kv_sequence_length=128,
            buffer_sequence_length=129,
            num_heads=8,
            kv_num_heads=2,
            head_size=64,
            local_window_size=128,
            share_buffer=True,
            sliding_window_cache=1,
            has_attention_bias=True,
            attention_bias_per_head=True,
        )

        with self.assertRaisesRegex(Exception, "attention_bias with sliding_window_cache"):
            parity_check_gqa_past(
                config=config,
                ep="CUDAExecutionProvider",
                device="cuda",
                torch_type=torch.float16,
                ort_type=TensorProto.FLOAT16,
                causal=True,
                rtol=rtol["fp16"],
                atol=atol["fp16"],
            )


@unittest.skipIf(not has_flash_attention(), "Flash Attention is not available, skipping tests.")
class TestFlashGQAPaddingPrompt(unittest.TestCase):
    def test_gqa_padding_prompt_flash_attention(self):
        if enable_debug_print:
            print("-" * 20)
            print("test_case: test_gqa_padding_prompt_flash_attention")

        with scoped_env_var("ORT_DISABLE_FLASH_ATTENTION", "0"):
            parity_test_gqa_padding_prompt()


@unittest.skipIf(not has_cuda_device(53), "Memory Efficient Attention is not available, skipping tests.")
class TestMemoryEfficientGQAPaddingPrompt(unittest.TestCase):
    def test_gqa_padding_prompt_memory_efficient_attention(self):
        if enable_debug_print:
            print("-" * 20)
            print("test_case: test_gqa_padding_prompt_memory_efficient_attention")

        with scoped_env_var("ORT_DISABLE_FLASH_ATTENTION", "1"):
            parity_test_gqa_padding_prompt()


# #################################################################################################
# Fused Kernel Parity Tests (ORT_DISABLE_FUSED_KV and ORT_DISABLE_FLASH_DECODE)
# #################################################################################################


def fused_kernel_test_cases():
    """Test cases specifically for fused vs unfused kernel parity."""
    configs = [
        # Decoding with RoPE and shared buffer
        GQAConfig(
            batch_size=2,
            q_sequence_length=1,
            kv_sequence_length=1,
            num_heads=16,
            kv_num_heads=4,
            head_size=128,
            past_kv_sequence_length=128,
            buffer_sequence_length=256,
            rotary=True,
            packed=False,
            share_buffer=True,
        ),
        # Packed QKV decoding with RoPE
        GQAConfig(
            batch_size=2,
            q_sequence_length=1,
            kv_sequence_length=1,
            num_heads=8,
            kv_num_heads=2,
            head_size=128,
            past_kv_sequence_length=64,
            buffer_sequence_length=128,
            rotary=True,
            packed=True,
            share_buffer=True,
        ),
        # Subsequent prompt with RoPE
        GQAConfig(
            batch_size=1,
            q_sequence_length=4,
            kv_sequence_length=4,
            num_heads=8,
            kv_num_heads=4,
            head_size=128,
            past_kv_sequence_length=32,
            buffer_sequence_length=64,
            rotary=True,
            packed=False,
            share_buffer=True,
        ),
    ]
    for i, config in enumerate(configs):
        yield f"fused_config_{i}", config


@unittest.skipIf(not has_flash_attention(), "Flash Attention is not available, skipping tests.")
class TestFlashDecodeMultiTokenParity(unittest.TestCase):
    def test_shared_buffer_multitoken_decode_matches_flash_attention(self):
        device = "cuda"
        torch_type = torch.float16
        ort_type = TensorProto.FLOAT16

        config = GQAConfig(
            batch_size=1,
            q_sequence_length=4,
            kv_sequence_length=4,
            num_heads=16,
            kv_num_heads=2,
            head_size=128,
            past_kv_sequence_length=64,
            buffer_sequence_length=128,
            rotary=False,
            packed=False,
            share_buffer=True,
            softcap=0.0,
        )

        torch.manual_seed(123)
        std = 0.1
        q = (
            torch.randn(
                config.batch_size,
                config.q_sequence_length,
                config.num_heads,
                config.head_size,
                device=device,
                dtype=torch_type,
            )
            * std
        )
        k = (
            torch.randn(
                config.batch_size,
                config.kv_num_heads,
                config.buffer_sequence_length,
                config.head_size,
                device=device,
                dtype=torch_type,
            )
            * std
        )
        v = torch.randn_like(k) * std

        past_len = config.past_kv_sequence_length
        k[:, :, past_len:, :] = 0
        v[:, :, past_len:, :] = 0

        new_k = (
            torch.randn(
                config.batch_size,
                config.kv_sequence_length,
                config.kv_num_heads,
                config.head_size,
                device=device,
                dtype=torch_type,
            )
            * std
        )
        new_v = torch.randn_like(new_k) * std

        seqlens_k = torch.tensor([past_len + config.kv_sequence_length - 1], dtype=torch.int32, device=device)

        def run_once(disable_flash_decode: bool):
            with (
                scoped_env_var("ORT_ENABLE_XQA", "0"),
                scoped_env_var("ORT_ENABLE_CUDNN_FLASH_ATTENTION", "0"),
                scoped_env_var("ORT_DISABLE_FLASH_ATTENTION", "0"),
                scoped_env_var("ORT_DISABLE_FLASH_DECODE", "1" if disable_flash_decode else "0"),
            ):
                out, present_k, present_v = gqa_past_func(
                    q=q,
                    k=k.clone(),
                    v=v.clone(),
                    config=config,
                    new_k=new_k,
                    new_v=new_v,
                    cos=None,
                    sin=None,
                    seqlens_k=seqlens_k,
                    position_ids=None,
                    attention_bias=None,
                    head_sink=None,
                    k_scale=None,
                    v_scale=None,
                    ep="CUDAExecutionProvider",
                    device=device,
                    share_buffer=True,
                    ort_type=ort_type,
                )
            return out, present_k, present_v

        out_fast, pk_fast, pv_fast = run_once(disable_flash_decode=False)
        out_ref, pk_ref, pv_ref = run_once(disable_flash_decode=True)

        numpy.testing.assert_allclose(
            out_fast.to(torch.float32).detach().cpu().numpy(),
            out_ref.to(torch.float32).detach().cpu().numpy(),
            rtol=2e-3,
            atol=2e-3,
        )

        valid_len = past_len + config.kv_sequence_length
        numpy.testing.assert_allclose(
            pk_fast[:, :, :valid_len, :].to(torch.float32).detach().cpu().numpy(),
            pk_ref[:, :, :valid_len, :].to(torch.float32).detach().cpu().numpy(),
            rtol=2e-3,
            atol=2e-3,
        )
        numpy.testing.assert_allclose(
            pv_fast[:, :, :valid_len, :].to(torch.float32).detach().cpu().numpy(),
            pv_ref[:, :, :valid_len, :].to(torch.float32).detach().cpu().numpy(),
            rtol=2e-3,
            atol=2e-3,
        )


def gqa_xqa_test_cases():
    # Decoding config (seq_len=1, share_buffer=True)
    # Testing different group sizes and query types
    for torch_type, ort_type in [(torch.float16, TensorProto.FLOAT16), (torch.bfloat16, TensorProto.BFLOAT16)]:
        for group_size in [4, 8, 16, 32]:
            for past_kv_sequence_length in [1, 4]:
                for rotary in [False, True]:
                    for packed in [False, True]:
                        for head_size in [256, 128, 64]:
                            kv_num_heads = 4
                            num_heads = kv_num_heads * group_size
                            config = GQAConfig(
                                batch_size=2,
                                q_sequence_length=1,
                                kv_sequence_length=1,
                                num_heads=num_heads,
                                kv_num_heads=kv_num_heads,
                                head_size=head_size,
                                past_kv_sequence_length=past_kv_sequence_length,
                                buffer_sequence_length=past_kv_sequence_length + 128,
                                rotary=rotary,
                                packed=packed,
                                share_buffer=True,
                                k_quant_type="PER_TENSOR",
                                v_quant_type="PER_TENSOR",
                                kv_cache_type="int8",
                                share_kv_scale=True,
                            )
                            type_str = "bf16" if torch_type == torch.bfloat16 else "fp16"
                            rot_str = "rot" if rotary else "norot"
                            pkd_str = "pkd" if packed else "sep"
                            name = f"{type_str}_g_{group_size}_h{head_size}_past{past_kv_sequence_length}_{rot_str}_{pkd_str}"
                            yield name, config, torch_type, ort_type


@unittest.skipIf(not has_flash_attention(), "Flash Attention is not available, skipping tests.")
@unittest.skipIf(not has_quantized_kv_cache(), "Quantized KV Cache is not available, skipping tests.")
class TestXQAQuantizedParity(unittest.TestCase):
    """Tests that verify fused kernels produce the same results as unfused kernels."""

    def setUp(self):
        # Force XQA on so the test is hermetic even if ORT_ENABLE_XQA=0 is set in the environment.
        self._prev_enable_xqa = os.environ.get("ORT_ENABLE_XQA")
        os.environ["ORT_ENABLE_XQA"] = "1"

    def tearDown(self):
        """Clear CUDA cache after each test to prevent memory corruption in batch runs."""
        if self._prev_enable_xqa is None:
            os.environ.pop("ORT_ENABLE_XQA", None)
        else:
            os.environ["ORT_ENABLE_XQA"] = self._prev_enable_xqa
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

    @parameterized.expand(gqa_xqa_test_cases())
    def test_xqa_quantized_parity(self, name, config, torch_type, ort_type):
        """Test XQA per-tensor INT8 quantized parity."""
        parity_check_gqa_past(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch_type,
            ort_type=ort_type,
            causal=True,
            rtol=rtol["int8_bf16"] if torch_type == torch.bfloat16 else rtol["int8_fp16"],
            atol=atol["int8_bf16"] if torch_type == torch.bfloat16 else atol["int8_fp16"],
            std=0.1,
        )


def gqa_xqa_head_sink_test_cases():
    # Non-quantized global decode with a head_sink (attention sink) input.
    # These configs exercise the XQA attention-sink path added for GPT-OSS style models:
    # seq_len=1, shared KV buffer, no softcap, no local window, head_size in {64, 128},
    # and group_size in {1, 2, 4, 5, 8, 16, 32}.
    for torch_type, ort_type in [(torch.float16, TensorProto.FLOAT16), (torch.bfloat16, TensorProto.BFLOAT16)]:
        for group_size in [1, 4, 5, 8]:
            for head_size in [64, 128]:
                for rotary in [False, True]:
                    kv_num_heads = 4
                    num_heads = kv_num_heads * group_size
                    config = GQAConfig(
                        batch_size=2,
                        q_sequence_length=1,
                        kv_sequence_length=1,
                        num_heads=num_heads,
                        kv_num_heads=kv_num_heads,
                        head_size=head_size,
                        past_kv_sequence_length=4,
                        buffer_sequence_length=4 + 128,
                        rotary=rotary,
                        packed=False,
                        share_buffer=True,
                        has_head_sink=True,
                    )
                    type_str = "bf16" if torch_type == torch.bfloat16 else "fp16"
                    rot_str = "rot" if rotary else "norot"
                    name = f"{type_str}_g{group_size}_h{head_size}_{rot_str}"
                    yield name, config, torch_type, ort_type


def gqa_xqa_head_sink_prepack_test_cases():
    # Same XQA attention-sink decode path as gqa_xqa_head_sink_test_cases(), but head_sink is baked
    # into the model as a constant initializer. This exercises GroupQueryAttention::PrePack, which
    # converts the constant head_sink once into the cached FP32 XQA buffer (use_prepacked_xqa_head_sink),
    # instead of the per-launch conversion scratch path used for runtime head_sink inputs.
    for torch_type, ort_type in [(torch.float16, TensorProto.FLOAT16), (torch.bfloat16, TensorProto.BFLOAT16)]:
        for group_size in [1, 4]:
            for rotary in [False, True]:
                kv_num_heads = 4
                num_heads = kv_num_heads * group_size
                config = GQAConfig(
                    batch_size=2,
                    q_sequence_length=1,
                    kv_sequence_length=1,
                    num_heads=num_heads,
                    kv_num_heads=kv_num_heads,
                    head_size=128,
                    past_kv_sequence_length=4,
                    buffer_sequence_length=4 + 128,
                    rotary=rotary,
                    packed=False,
                    share_buffer=True,
                    has_head_sink=True,
                    head_sink_as_initializer=True,
                )
                type_str = "bf16" if torch_type == torch.bfloat16 else "fp16"
                rot_str = "rot" if rotary else "norot"
                name = f"{type_str}_g{group_size}_h128_{rot_str}_prepack"
                yield name, config, torch_type, ort_type


@unittest.skipIf(not has_xqa(), "XQA is not available, skipping tests.")
@unittest.skipIf(not has_flash_attention(), "Flash Attention is not available, skipping tests.")
class TestXQAHeadSinkParity(unittest.TestCase):
    """Verify the non-quantized XQA attention-sink (head_sink) decode path matches the reference."""

    def setUp(self):
        # XQA is enabled by default for fp16/bf16 (ORT_ENABLE_XQA defaults to 1).
        # Pop any override so we exercise the real default behavior.
        self._prev_enable_xqa = os.environ.pop("ORT_ENABLE_XQA", None)

    def tearDown(self):
        # Restore the environment so other tests run with the default XQA setting.
        if self._prev_enable_xqa is None:
            os.environ.pop("ORT_ENABLE_XQA", None)
        else:
            os.environ["ORT_ENABLE_XQA"] = self._prev_enable_xqa
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

    @parameterized.expand(gqa_xqa_head_sink_test_cases())
    def test_xqa_head_sink_parity(self, name, config, torch_type, ort_type):
        """Test XQA non-quantized parity with a head_sink (attention sink) input."""
        parity_check_gqa_past(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch_type,
            ort_type=ort_type,
            causal=True,
            rtol=rtol["bf16"] if torch_type == torch.bfloat16 else rtol["fp16"],
            atol=atol["bf16"] if torch_type == torch.bfloat16 else atol["fp16"],
            std=0.1,
        )

    @parameterized.expand(gqa_xqa_head_sink_prepack_test_cases())
    def test_xqa_head_sink_prepack_parity(self, name, config, torch_type, ort_type):
        """Test XQA parity when head_sink is a constant initializer (exercises PrePack)."""
        parity_check_gqa_past(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch_type,
            ort_type=ort_type,
            causal=True,
            rtol=rtol["bf16"] if torch_type == torch.bfloat16 else rtol["fp16"],
            atol=atol["bf16"] if torch_type == torch.bfloat16 else atol["fp16"],
            std=0.1,
        )


def gqa_xqa_sliding_window_test_cases():
    # Non-quantized sliding-window (local attention) decode through the XQA kernel.
    #
    # The XQA decode path now supports local_window_size > 0 on the non-quantized fp16/bf16
    # path (GPT-OSS style sliding-window layers). XQA selection requires: decode (q_seq=1),
    # a shared KV buffer, softcap==0, head_size in {64, 128, 256} and 64 % group_size == 0.
    #
    # Two window/past relationships are covered:
    #   past > window  -> the sliding mask drops the oldest keys (the new code path).
    #   past <= window -> the window spans the whole cache (parity with global attention).
    #   past + 1 == window -> the exact guard boundary (cacheSeqLen == slidingWinSize) that locks
    #                         down the kernel's `>` vs `>=` window comparison.
    # has_head_sink toggles the GPT-OSS attention-sink input, which composes with the window
    # in-kernel; both with and without a sink are exercised.
    for torch_type, ort_type in [(torch.float16, TensorProto.FLOAT16), (torch.bfloat16, TensorProto.BFLOAT16)]:
        for head_size in [64, 128]:
            for group_size in [4, 8]:
                for past_kv_sequence_length, local_window_size in [(512, 128), (64, 128), (127, 128)]:
                    for has_head_sink in [False, True]:
                        kv_num_heads = 4
                        num_heads = kv_num_heads * group_size
                        config = GQAConfig(
                            batch_size=2,
                            q_sequence_length=1,
                            kv_sequence_length=1,
                            num_heads=num_heads,
                            kv_num_heads=kv_num_heads,
                            head_size=head_size,
                            past_kv_sequence_length=past_kv_sequence_length,
                            buffer_sequence_length=past_kv_sequence_length + 128,
                            local_window_size=local_window_size,
                            rotary=True,
                            rotary_interleaved=False,
                            packed=False,
                            share_buffer=True,
                            has_head_sink=has_head_sink,
                        )
                        type_str = "bf16" if torch_type == torch.bfloat16 else "fp16"
                        sink_str = "sink" if has_head_sink else "nosink"
                        win_str = f"past{past_kv_sequence_length}_win{local_window_size}"
                        name = f"{type_str}_g{group_size}_h{head_size}_{win_str}_{sink_str}"
                        yield name, config, torch_type, ort_type


@unittest.skipIf(not has_xqa(), "XQA is not available, skipping tests.")
class TestXQASlidingWindowParity(unittest.TestCase):
    """Verify the non-quantized XQA sliding-window (local attention) decode path matches the reference."""

    def setUp(self):
        # Force XQA on so the test is hermetic even if ORT_ENABLE_XQA=0 is set in the environment.
        self._prev_enable_xqa = os.environ.get("ORT_ENABLE_XQA")
        os.environ["ORT_ENABLE_XQA"] = "1"

    def tearDown(self):
        """Clear CUDA cache after each test to prevent memory corruption in batch runs."""
        if self._prev_enable_xqa is None:
            os.environ.pop("ORT_ENABLE_XQA", None)
        else:
            os.environ["ORT_ENABLE_XQA"] = self._prev_enable_xqa
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

    @parameterized.expand(gqa_xqa_sliding_window_test_cases())
    def test_xqa_sliding_window_parity(self, name, config, torch_type, ort_type):
        """Test XQA non-quantized parity with a sliding (local) attention window."""

        def run_parity_check():
            parity_check_gqa_past(
                config=config,
                ep="CUDAExecutionProvider",
                device="cuda",
                torch_type=torch_type,
                ort_type=ort_type,
                causal=True,
                rtol=rtol["bf16"] if torch_type == torch.bfloat16 else rtol["fp16"],
                atol=atol["bf16"] if torch_type == torch.bfloat16 else atol["fp16"],
                std=0.1,
            )

        # XQA is enabled by default for fp16/bf16, so no head_sink input is required to select it.
        self.assertEqual("XQA", get_sdpa_kernel_from_debug_info(run_parity_check))


def gqa_xqa_quantized_sliding_window_test_cases():
    # Quantized (INT8 / FP8) sliding-window (local attention) decode through the XQA kernel.
    #
    # The XQA decode path now supports local_window_size > 0 on the quantized KV-cache paths as
    # well. Quantized XQA selection requires: decode (q_seq=1), a shared KV buffer, supported
    # head_size and group_size combinations for the active kernel configuration. Attention sinks
    # are covered separately by
    # gqa_xqa_quantized_head_sink_test_cases(), so no head_sink here.
    #
    # Two window/past relationships are covered:
    #   past > window  -> the sliding mask drops the oldest keys (the new code path).
    #   past <= window -> the window spans the whole cache (parity with global attention).
    #   past + 1 == window -> the exact guard boundary (cacheSeqLen == slidingWinSize).
    kv_cache_types = ["int8"]
    if has_fp8_kv_cache and has_fp8_xqa():
        kv_cache_types.append("fp8")
    for kv_cache_type in kv_cache_types:
        for torch_type, ort_type in [(torch.float16, TensorProto.FLOAT16), (torch.bfloat16, TensorProto.BFLOAT16)]:
            for head_size in [64, 128]:
                for group_size in [4, 8]:
                    for past_kv_sequence_length, local_window_size in [(512, 128), (64, 128), (127, 128)]:
                        kv_num_heads = 4
                        num_heads = kv_num_heads * group_size
                        config = GQAConfig(
                            batch_size=2,
                            q_sequence_length=1,
                            kv_sequence_length=1,
                            num_heads=num_heads,
                            kv_num_heads=kv_num_heads,
                            head_size=head_size,
                            past_kv_sequence_length=past_kv_sequence_length,
                            buffer_sequence_length=past_kv_sequence_length + 128,
                            local_window_size=local_window_size,
                            rotary=True,
                            rotary_interleaved=False,
                            packed=False,
                            share_buffer=True,
                            k_quant_type="PER_TENSOR",
                            v_quant_type="PER_TENSOR",
                            kv_cache_type=kv_cache_type,
                            share_kv_scale=True,
                        )
                        type_str = "bf16" if torch_type == torch.bfloat16 else "fp16"
                        win_str = f"past{past_kv_sequence_length}_win{local_window_size}"
                        name = f"{kv_cache_type}_{type_str}_g{group_size}_h{head_size}_{win_str}"
                        yield name, config, torch_type, ort_type


@unittest.skipIf(not has_xqa(), "XQA is not available, skipping tests.")
@unittest.skipIf(not has_quantized_kv_cache(), "Quantized KV Cache is not available, skipping tests.")
class TestXQAQuantizedSlidingWindowParity(unittest.TestCase):
    """Verify the quantized (INT8/FP8) XQA sliding-window (local attention) decode path matches the reference."""

    def setUp(self):
        # Force XQA on so the test is hermetic even if ORT_ENABLE_XQA=0 is set in the environment.
        self._prev_enable_xqa = os.environ.get("ORT_ENABLE_XQA")
        os.environ["ORT_ENABLE_XQA"] = "1"

    def tearDown(self):
        """Clear CUDA cache after each test to prevent memory corruption in batch runs."""
        if self._prev_enable_xqa is None:
            os.environ.pop("ORT_ENABLE_XQA", None)
        else:
            os.environ["ORT_ENABLE_XQA"] = self._prev_enable_xqa
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

    @parameterized.expand(gqa_xqa_quantized_sliding_window_test_cases())
    def test_xqa_quantized_sliding_window_parity(self, name, config, torch_type, ort_type):
        """Test XQA quantized parity with a sliding (local) attention window."""
        type_str = "bf16" if torch_type == torch.bfloat16 else "fp16"
        rtol_key = f"{config.kv_cache_type}_{type_str}"

        def run_parity_check():
            parity_check_gqa_past(
                config=config,
                ep="CUDAExecutionProvider",
                device="cuda",
                torch_type=torch_type,
                ort_type=ort_type,
                causal=True,
                rtol=rtol[rtol_key],
                atol=atol[rtol_key],
                std=0.1,
            )

        # XQA is enabled by default for fp16/bf16, so the quantized sliding-window path is selected.
        self.assertEqual("XQA", get_sdpa_kernel_from_debug_info(run_parity_check))


def gqa_xqa_quantized_head_sink_test_cases():
    # Quantized (INT8 / FP8) decode with an attention sink (head_sink / smooth softmax) through
    # the XQA kernel -- the GPT-OSS style configuration.
    #
    # The sink term is folded into the softmax row sum by the same kernel code used on the
    # non-quantized path, and the KV dequant scale is applied to the QK scores before the row
    # max/sum are computed, so the sink and the scores live in the same dequantized domain.
    #
    # Coverage:
    #   local_window_size == -1 -> global attention with a sink.
    #   local_window_size > 0 with past > window -> sliding window composed with a sink.
    #   past_kv_sequence_length 512 also forces the multi-block (Flash Decoding) reduction, where
    #   the sink is added to the merged row sum instead of the per-CTA row sum.
    #   head_sink_as_initializer toggles the PrePack'd FP32 sink buffer vs. the per-launch
    #   conversion scratch path.
    kv_cache_types = ["int8"]
    if has_fp8_kv_cache:
        kv_cache_types.append("fp8")
    for kv_cache_type in kv_cache_types:
        for torch_type, ort_type in [(torch.float16, TensorProto.FLOAT16), (torch.bfloat16, TensorProto.BFLOAT16)]:
            for head_size in [64, 128]:
                for group_size in [4, 8]:
                    for past_kv_sequence_length, local_window_size in [(4, -1), (512, -1), (512, 128)]:
                        for head_sink_as_initializer in [False, True]:
                            kv_num_heads = 4
                            num_heads = kv_num_heads * group_size
                            config = GQAConfig(
                                batch_size=2,
                                q_sequence_length=1,
                                kv_sequence_length=1,
                                num_heads=num_heads,
                                kv_num_heads=kv_num_heads,
                                head_size=head_size,
                                past_kv_sequence_length=past_kv_sequence_length,
                                buffer_sequence_length=past_kv_sequence_length + 128,
                                local_window_size=local_window_size,
                                rotary=True,
                                rotary_interleaved=False,
                                packed=False,
                                share_buffer=True,
                                has_head_sink=True,
                                head_sink_as_initializer=head_sink_as_initializer,
                                k_quant_type="PER_TENSOR",
                                v_quant_type="PER_TENSOR",
                                kv_cache_type=kv_cache_type,
                                share_kv_scale=True,
                            )
                            type_str = "bf16" if torch_type == torch.bfloat16 else "fp16"
                            win_str = f"past{past_kv_sequence_length}_win{local_window_size}"
                            pack_str = "prepack" if head_sink_as_initializer else "runtime"
                            name = f"{kv_cache_type}_{type_str}_g{group_size}_h{head_size}_{win_str}_{pack_str}"
                            yield name, config, torch_type, ort_type


@unittest.skipIf(not has_xqa(), "XQA is not available, skipping tests.")
@unittest.skipIf(not has_quantized_kv_cache(), "Quantized KV Cache is not available, skipping tests.")
class TestXQAQuantizedHeadSinkParity(unittest.TestCase):
    """Verify the quantized (INT8/FP8) XQA attention-sink decode path matches the reference."""

    def setUp(self):
        # Force XQA on so the test is hermetic even if ORT_ENABLE_XQA=0 is set in the environment.
        self._prev_enable_xqa = os.environ.get("ORT_ENABLE_XQA")
        os.environ["ORT_ENABLE_XQA"] = "1"

    def tearDown(self):
        """Clear CUDA cache after each test to prevent memory corruption in batch runs."""
        if self._prev_enable_xqa is None:
            os.environ.pop("ORT_ENABLE_XQA", None)
        else:
            os.environ["ORT_ENABLE_XQA"] = self._prev_enable_xqa
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

    @parameterized.expand(gqa_xqa_quantized_head_sink_test_cases())
    def test_xqa_quantized_head_sink_parity(self, name, config, torch_type, ort_type):
        """Test XQA quantized parity with an attention sink (head_sink) input."""
        type_str = "bf16" if torch_type == torch.bfloat16 else "fp16"
        rtol_key = f"{config.kv_cache_type}_{type_str}"

        def run_parity_check():
            parity_check_gqa_past(
                config=config,
                ep="CUDAExecutionProvider",
                device="cuda",
                torch_type=torch_type,
                ort_type=ort_type,
                causal=True,
                rtol=rtol[rtol_key],
                atol=atol[rtol_key],
                std=0.1,
            )

        # Guard against silently falling back: the quantized + sink combination must reach XQA.
        self.assertEqual("XQA", get_sdpa_kernel_from_debug_info(run_parity_check))


def gqa_xqa_separate_kv_scale_test_cases():
    # Quantized (INT8 / FP8) decode where K and V use *different* per-tensor scales.
    #
    # The XQA kernel keeps the two scales apart: k_scale is folded into qkScale (applied to Q*K.T
    # before the softmax row max/sum) and v_scale into voScale (applied to the P*V accumulator).
    # These tests guard both the numerics and the dispatch decision -- XQA used to require the two
    # scale inputs to be the same tensor, which silently disqualified any model that calibrates K
    # and V independently.
    #
    # Coverage includes global attention, sliding window, the multi-block (Flash Decoding)
    # reduction (past 512), and attention sinks, since each of those paths touches the rescales.
    kv_cache_types = ["int8"]
    if has_fp8_kv_cache:
        kv_cache_types.append("fp8")
    for kv_cache_type in kv_cache_types:
        for torch_type, ort_type in [(torch.float16, TensorProto.FLOAT16), (torch.bfloat16, TensorProto.BFLOAT16)]:
            for head_size in [64, 128, 256]:
                for group_size in [4, 8]:
                    for past_kv_sequence_length, local_window_size in [(4, -1), (512, -1), (512, 128)]:
                        for has_head_sink in [False, True]:
                            kv_num_heads = 4
                            num_heads = kv_num_heads * group_size
                            config = GQAConfig(
                                batch_size=2,
                                q_sequence_length=1,
                                kv_sequence_length=1,
                                num_heads=num_heads,
                                kv_num_heads=kv_num_heads,
                                head_size=head_size,
                                past_kv_sequence_length=past_kv_sequence_length,
                                buffer_sequence_length=past_kv_sequence_length + 128,
                                local_window_size=local_window_size,
                                rotary=True,
                                rotary_interleaved=False,
                                packed=False,
                                share_buffer=True,
                                has_head_sink=has_head_sink,
                                head_sink_as_initializer=True,
                                k_quant_type="PER_TENSOR",
                                v_quant_type="PER_TENSOR",
                                kv_cache_type=kv_cache_type,
                                share_kv_scale=False,
                            )
                            type_str = "bf16" if torch_type == torch.bfloat16 else "fp16"
                            win_str = f"past{past_kv_sequence_length}_win{local_window_size}"
                            sink_str = "sink" if has_head_sink else "nosink"
                            name = f"{kv_cache_type}_{type_str}_g{group_size}_h{head_size}_{win_str}_{sink_str}"
                            yield name, config, torch_type, ort_type


@unittest.skipIf(not has_xqa(), "XQA is not available, skipping tests.")
@unittest.skipIf(not has_quantized_kv_cache(), "Quantized KV Cache is not available, skipping tests.")
class TestXQASeparateKVScaleParity(unittest.TestCase):
    """Verify XQA handles independent K and V dequantization scales."""

    def setUp(self):
        # Force XQA on so the test is hermetic even if ORT_ENABLE_XQA=0 is set in the environment.
        self._prev_enable_xqa = os.environ.get("ORT_ENABLE_XQA")
        os.environ["ORT_ENABLE_XQA"] = "1"

    def tearDown(self):
        """Clear CUDA cache after each test to prevent memory corruption in batch runs."""
        if self._prev_enable_xqa is None:
            os.environ.pop("ORT_ENABLE_XQA", None)
        else:
            os.environ["ORT_ENABLE_XQA"] = self._prev_enable_xqa
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

    @parameterized.expand(gqa_xqa_separate_kv_scale_test_cases())
    def test_xqa_separate_kv_scale_parity(self, name, config, torch_type, ort_type):
        """Test XQA quantized parity when k_scale and v_scale are distinct tensors."""
        type_str = "bf16" if torch_type == torch.bfloat16 else "fp16"
        rtol_key = f"{config.kv_cache_type}_{type_str}"

        def run_parity_check():
            parity_check_gqa_past(
                config=config,
                ep="CUDAExecutionProvider",
                device="cuda",
                torch_type=torch_type,
                ort_type=ort_type,
                causal=True,
                rtol=rtol[rtol_key],
                atol=atol[rtol_key],
                std=0.1,
            )

        # Distinct k_scale/v_scale tensors must not disqualify the op from XQA.
        self.assertEqual("XQA", get_sdpa_kernel_from_debug_info(run_parity_check))


def gqa_xqa_per_channel_scale_test_cases():
    # Quantized (INT8 / FP8) decode with PER_CHANNEL k/v scales on the XQA path.
    #
    # XQA only understands a scalar dequant scale, so per-channel scales are folded out of the
    # kernel: k_scale into Q (the channel dim is contracted by Q*K.T) and v_scale into the
    # attention output (the channel dim is free in P*V). Both foldings are exact because
    # dequantization is linear, so these tests are ordinary parity tests -- but they also guard
    # the dispatch decision, since per-channel used to fall back to
    # "dequantize the whole cache, then run flash attention" on every decode step.
    #
    # Coverage: global attention, sliding window, the multi-block (Flash Decoding) reduction
    # (past 512), attention sinks, and rotary on/off (rotary decides whether the folded Q lands in
    # the RoPE scratch buffer or in a scratch buffer allocated only for the folding).
    kv_cache_types = ["int8"]
    if has_fp8_kv_cache:
        kv_cache_types.append("fp8")
    for kv_cache_type in kv_cache_types:
        for torch_type, ort_type in [(torch.float16, TensorProto.FLOAT16), (torch.bfloat16, TensorProto.BFLOAT16)]:
            for head_size in [64, 128]:
                for group_size in [4, 8]:
                    for past_kv_sequence_length, local_window_size in [(4, -1), (512, -1), (512, 128)]:
                        for has_head_sink, rotary in [(False, True), (True, True), (True, False)]:
                            kv_num_heads = 4
                            num_heads = kv_num_heads * group_size
                            config = GQAConfig(
                                batch_size=2,
                                q_sequence_length=1,
                                kv_sequence_length=1,
                                num_heads=num_heads,
                                kv_num_heads=kv_num_heads,
                                head_size=head_size,
                                past_kv_sequence_length=past_kv_sequence_length,
                                buffer_sequence_length=past_kv_sequence_length + 128,
                                local_window_size=local_window_size,
                                rotary=rotary,
                                rotary_interleaved=False,
                                packed=False,
                                share_buffer=True,
                                has_head_sink=has_head_sink,
                                head_sink_as_initializer=True,
                                k_quant_type="PER_CHANNEL",
                                v_quant_type="PER_CHANNEL",
                                kv_cache_type=kv_cache_type,
                                share_kv_scale=False,
                            )
                            type_str = "bf16" if torch_type == torch.bfloat16 else "fp16"
                            win_str = f"past{past_kv_sequence_length}_win{local_window_size}"
                            sink_str = "sink" if has_head_sink else "nosink"
                            rot_str = "rot" if rotary else "norot"
                            name = (
                                f"{kv_cache_type}_{type_str}_g{group_size}_h{head_size}_{win_str}_{sink_str}_{rot_str}"
                            )
                            yield name, config, torch_type, ort_type


@unittest.skipIf(not has_xqa(), "XQA is not available, skipping tests.")
@unittest.skipIf(not has_quantized_kv_cache(), "Quantized KV Cache is not available, skipping tests.")
class TestXQAPerChannelScaleParity(unittest.TestCase):
    """Verify XQA serves PER_CHANNEL quantized KV caches by folding the scales into Q / the output."""

    def setUp(self):
        # Force XQA on so the test is hermetic even if ORT_ENABLE_XQA=0 is set in the environment.
        self._prev_enable_xqa = os.environ.get("ORT_ENABLE_XQA")
        os.environ["ORT_ENABLE_XQA"] = "1"

    def tearDown(self):
        """Clear CUDA cache after each test to prevent memory corruption in batch runs."""
        if self._prev_enable_xqa is None:
            os.environ.pop("ORT_ENABLE_XQA", None)
        else:
            os.environ["ORT_ENABLE_XQA"] = self._prev_enable_xqa
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

    @parameterized.expand(gqa_xqa_per_channel_scale_test_cases())
    def test_xqa_per_channel_scale_parity(self, name, config, torch_type, ort_type):
        """Test XQA quantized parity with per-channel k/v dequantization scales."""
        type_str = "bf16" if torch_type == torch.bfloat16 else "fp16"
        rtol_key = f"{config.kv_cache_type}_{type_str}"

        def run_parity_check():
            parity_check_gqa_past(
                config=config,
                ep="CUDAExecutionProvider",
                device="cuda",
                torch_type=torch_type,
                ort_type=ort_type,
                causal=True,
                rtol=rtol[rtol_key],
                atol=atol[rtol_key],
                std=0.1,
            )

        # Per-channel scales must reach XQA instead of the dequantize-the-whole-cache fallback.
        self.assertEqual("XQA", get_sdpa_kernel_from_debug_info(run_parity_check))


@unittest.skipIf(not has_flash_attention(), "Flash Attention is not available, skipping tests.")
@unittest.skipIf(not has_quantized_kv_cache(), "Quantized KV Cache is not available, skipping tests.")
class TestGQARegressions(unittest.TestCase):
    """Specific regression tests for historical bugs."""

    def test_gqa_cuda_rejects_zero_local_window_size(self):
        if not has_cuda_provider():
            self.skipTest("CUDA required")

        config = GQAConfig(
            batch_size=1,
            num_heads=4,
            kv_num_heads=4,
            head_size=64,
            q_sequence_length=1,
            kv_sequence_length=1,
            past_kv_sequence_length=0,
            buffer_sequence_length=1,
            local_window_size=0,
            share_buffer=True,
        )
        onnx_model_str = create_group_query_attention_graph_prompt(config, TensorProto.FLOAT16)

        with self.assertRaisesRegex(Exception, "local_window_size must be -1 or greater than 0"):
            InferenceSession(
                onnx_model_str,
                SessionOptions(),
                providers=[resolve_cuda_plugin_ep("CUDAExecutionProvider")],
            )

    def test_gqa_rope_separate_qkv_bug(self):
        """
        Regression test for separate QKV + RoPE + FlashAttention bug.
        The bug caused q_out to be nullptr when unpacking separate QKV with only Q rotation (standard GQA),
        leading to unrotated Q being used in Attention.
        """
        if not has_cuda_provider():
            self.skipTest("CUDA required")

        # Config that triggers the path: Prompt phase, Separate QKV inputs, RoPE enabled
        config = GQAConfig(
            batch_size=1,
            num_heads=4,
            kv_num_heads=4,
            head_size=128,
            q_sequence_length=16,
            kv_sequence_length=16,
            past_kv_sequence_length=0,
            buffer_sequence_length=16,
            rotary=True,
            rotary_interleaved=False,
            share_buffer=True,
        )

        torch_type = torch.float16
        ort_type = TensorProto.FLOAT16
        device = "cuda"

        parity_check_gqa_prompt(
            config=config,
            ep="CUDAExecutionProvider",
            device=device,
            torch_type=torch_type,
            ort_type=ort_type,
            causal=True,
            rtol=1e-3,
            atol=1e-3,
            std=1.0,
        )

    def test_gqa_int8_large_seq_batch4(self):
        """
        Regression test for batch_size=4 + max_seq_len=8192 + int8 KV cache crash.
        This reproduces a CUDA illegal memory access due to scratch size under-allocation.
        """
        if not has_cuda_provider():
            self.skipTest("CUDA required")

        # Config that triggers the crash: batch=4, large max_seq_len, int8 kv
        config = GQAConfig(
            batch_size=4,
            num_heads=32,
            kv_num_heads=8,
            head_size=128,
            q_sequence_length=1,
            kv_sequence_length=1,
            past_kv_sequence_length=8191,
            buffer_sequence_length=8192,
            rotary=True,
            rotary_interleaved=False,
            k_quant_type="PER_TENSOR",
            v_quant_type="PER_TENSOR",
            kv_cache_type="int8",
            share_buffer=True,
            share_kv_scale=True,
        )

        torch_type = torch.float16
        ort_type = TensorProto.FLOAT16
        device = "cuda"

        parity_check_gqa_past(
            config=config,
            ep="CUDAExecutionProvider",
            device=device,
            torch_type=torch_type,
            ort_type=ort_type,
            causal=True,
            rtol=5e-2,
            atol=5e-2,
        )

    def test_gqa_local_window_large_context_decode(self):
        """
        Regression test for FlashDecode split planning with a local attention window.

        Mirrors a gpt-oss-style decode step: a large past KV context combined with a small
        sliding (local) window. The split-K planning is clamped to the local window length,
        so only the windowed portion of the KV cache participates in the decode. This verifies
        that the narrowed split planning still produces correct results.
        """
        if not has_flash_attention():
            self.skipTest("Flash Attention is not available")

        # Decode (q_sequence_length=1) with a large past context but a small local window.
        config = GQAConfig(
            batch_size=2,
            num_heads=64,
            kv_num_heads=8,
            head_size=64,
            q_sequence_length=1,
            kv_sequence_length=1,
            past_kv_sequence_length=4096,
            buffer_sequence_length=4096 + 8,
            local_window_size=128,
            rotary=True,
            rotary_interleaved=False,
            share_buffer=True,
        )

        torch_type = torch.float16
        ort_type = TensorProto.FLOAT16
        device = "cuda"

        with scoped_env_var("ORT_DISABLE_FLASH_ATTENTION", "0"):
            parity_check_gqa_past(
                config=config,
                ep="CUDAExecutionProvider",
                device=device,
                torch_type=torch_type,
                ort_type=ort_type,
                causal=True,
                rtol=rtol["fp16"],
                atol=atol["fp16"],
            )

    @unittest.skipIf(not has_cuda_device(89) or not has_fp8_kv_cache, "FP8 KV cache is not available, skipping tests.")
    def test_gqa_fp8_kv_cache(self):
        """
        Test GQA with FP8 E4M3 quantized KV cache.
        Requires SM89+ (Ada Lovelace or newer) and USE_FP8_KV_CACHE build flag.
        """
        config = GQAConfig(
            batch_size=2,
            num_heads=32,
            kv_num_heads=8,
            head_size=128,
            q_sequence_length=1,
            kv_sequence_length=1,
            past_kv_sequence_length=127,
            buffer_sequence_length=128,
            rotary=True,
            rotary_interleaved=False,
            k_quant_type="PER_TENSOR",
            v_quant_type="PER_TENSOR",
            kv_cache_type="fp8",
            share_buffer=True,
            share_kv_scale=True,
        )

        torch_type = torch.float16
        ort_type = TensorProto.FLOAT16
        device = "cuda"

        try:
            parity_check_gqa_past(
                config=config,
                ep="CUDAExecutionProvider",
                device=device,
                torch_type=torch_type,
                ort_type=ort_type,
                causal=True,
                rtol=5e-2,
                atol=5e-2,
            )
        except Exception as e:
            # FP8 may not be built, skip if kernel not registered
            if "Float8E4M3FN" in str(e) or "fp8" in str(e).lower():
                self.skipTest(f"FP8 KV cache not available: {e}")
            raise

    @unittest.skipIf(not has_cuda_device(89) or not has_fp8_kv_cache, "FP8 KV cache is not available, skipping tests.")
    def test_gqa_fp8_prompt(self):
        """
        Test GQA Prompt phase with FP8 E4M3 quantized KV cache.
        """
        config = GQAConfig(
            batch_size=2,
            num_heads=32,
            kv_num_heads=8,
            head_size=128,
            q_sequence_length=128,
            kv_sequence_length=128,
            past_kv_sequence_length=0,
            buffer_sequence_length=128,
            rotary=True,
            rotary_interleaved=False,
            k_quant_type="PER_TENSOR",
            v_quant_type="PER_TENSOR",
            kv_cache_type="fp8",
            share_buffer=True,
            share_kv_scale=True,
            kv_cache_bit_width=8,
        )

        torch_type = torch.float16
        ort_type = TensorProto.FLOAT16
        device = "cuda"

        try:
            parity_check_gqa_prompt(
                config=config,
                ep="CUDAExecutionProvider",
                device=device,
                torch_type=torch_type,
                ort_type=ort_type,
                causal=True,
                rtol=5e-2,
                atol=5e-2,
            )
        except Exception as e:
            if "Float8E4M3FN" in str(e) or "fp8" in str(e).lower():
                self.skipTest(f"FP8 KV cache not available: {e}")
            raise

    @unittest.skipIf(not has_cuda_device(89) or not has_fp8_kv_cache, "FP8 KV cache is not available, skipping tests.")
    @unittest.skipIf(quick_build, "Quick build only has hdim128 flash attention kernels; head_size=48 needs hdim64.")
    def test_gqa_fp8_fallback_unsupported_head_size(self):
        """
        Test GQA with FP8 KV cache on a head size not supported by XQA.
        This forces fallback to the generic generic kernel (if available) or ensures graceful failure/correctness.
        """
        config = GQAConfig(
            batch_size=2,
            num_heads=32,
            kv_num_heads=8,
            head_size=48,  # Valid head size (multiple of 16) but not supported by XQA (supports 64, 128, 256)
            q_sequence_length=1,
            kv_sequence_length=1,
            past_kv_sequence_length=64,
            buffer_sequence_length=128,
            rotary=True,
            rotary_interleaved=False,
            k_quant_type="PER_TENSOR",
            v_quant_type="PER_TENSOR",
            kv_cache_type="fp8",
            share_buffer=True,
            share_kv_scale=True,
        )

        torch_type = torch.float16
        ort_type = TensorProto.FLOAT16
        device = "cuda"

        parity_check_gqa_past(
            config=config,
            ep="CUDAExecutionProvider",
            device=device,
            torch_type=torch_type,
            ort_type=ort_type,
            causal=True,
            rtol=5e-2,
            atol=5e-2,
        )

    # ------------------------------------------------------------------------
    # Gemma 4 global attention layers (issue #28195): num_attention_heads=8,
    # num_key_value_heads=4, head_dim=512. The unfused CUDA runner produced
    # NaN at head_dim=512, scale=1.0 because raw Q*K^T overflowed fp16 even
    # though cuBLAS accumulated in FP32 (output C was fp16). The new GQA
    # unfused kernel writes QK to an FP32 scratch and fixes this.
    # ------------------------------------------------------------------------
    def _run_gemma4_gqa(
        self,
        torch_type,
        ort_type,
        q_sequence_length,
        past_kv_sequence_length,
        is_prompt,
        local_window_size=-1,
        softcap=0.0,
    ):
        if not has_cuda_provider():
            self.skipTest("CUDA required")
        if torch_type == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            self.skipTest("BFloat16 not supported on this device")

        # Force the unfused path: disable Flash (doesn't support head_size>256)
        # and Memory-Efficient Attention (cutlass FMHA caps at head_size 256 too).
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "1"
        os.environ["ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION"] = "1"
        self.addCleanup(os.environ.pop, "ORT_DISABLE_FLASH_ATTENTION", None)
        self.addCleanup(os.environ.pop, "ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION", None)

        config = GQAConfig(
            batch_size=1,
            num_heads=8,
            kv_num_heads=4,
            head_size=512,
            q_sequence_length=q_sequence_length,
            kv_sequence_length=q_sequence_length,
            past_kv_sequence_length=past_kv_sequence_length,
            buffer_sequence_length=q_sequence_length + past_kv_sequence_length + 8,
            local_window_size=local_window_size,
            rotary=False,
            rotary_interleaved=False,
            packed=False,
            share_buffer=True,
            softcap=softcap,
            use_smooth_softmax=False,
            has_head_sink=False,
            has_position_ids=False,
        )

        dtype_key = "fp16" if ort_type == TensorProto.FLOAT16 else "bf16"
        check = parity_check_gqa_prompt if is_prompt else parity_check_gqa_past
        check(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch_type,
            ort_type=ort_type,
            causal=True,
            rtol=rtol[dtype_key],
            atol=atol[dtype_key],
        )

    def test_gqa_gemma4_global_prompt_fp16(self):
        """#28195 exact repro: fp16 prompt with head_dim=512, Gemma 4 head config."""
        self._run_gemma4_gqa(
            torch.float16, TensorProto.FLOAT16, q_sequence_length=16, past_kv_sequence_length=0, is_prompt=True
        )

    def test_gqa_gemma4_global_decode_fp16(self):
        """#28195: fp16 decode with past KV at head_dim=512."""
        self._run_gemma4_gqa(
            torch.float16, TensorProto.FLOAT16, q_sequence_length=1, past_kv_sequence_length=64, is_prompt=False
        )

    def test_gqa_gemma4_global_decode_fp16_long(self):
        """Gemma 4 global attention with longer past at head_dim=512."""
        self._run_gemma4_gqa(
            torch.float16, TensorProto.FLOAT16, q_sequence_length=1, past_kv_sequence_length=2048, is_prompt=False
        )

    def test_gqa_gemma4_global_prompt_bf16(self):
        """Gemma 4 global attention in bf16 prompt phase at head_dim=512."""
        self._run_gemma4_gqa(
            torch.bfloat16, TensorProto.BFLOAT16, q_sequence_length=16, past_kv_sequence_length=0, is_prompt=True
        )

    def test_gqa_gemma4_global_decode_bf16(self):
        """Gemma 4 global attention in bf16 decode phase at head_dim=512."""
        self._run_gemma4_gqa(
            torch.bfloat16, TensorProto.BFLOAT16, q_sequence_length=1, past_kv_sequence_length=64, is_prompt=False
        )

    def test_gqa_gemma4_global_prompt_fp16_softcap(self):
        """Gemma 4 global attention with softcap (Gemma family uses logit softcap)."""
        self._run_gemma4_gqa(
            torch.float16,
            TensorProto.FLOAT16,
            q_sequence_length=16,
            past_kv_sequence_length=0,
            is_prompt=True,
            softcap=50.0,
        )

    def test_gqa_gemma4_local_window_decode_fp16(self):
        """
        Gemma 4 has mixed global + sliding-window (local) attention layers. This
        exercises the unfused kernel's sliding-window mask at head_dim=512.
        """
        self._run_gemma4_gqa(
            torch.float16,
            TensorProto.FLOAT16,
            q_sequence_length=1,
            past_kv_sequence_length=256,
            is_prompt=False,
            local_window_size=128,
        )


# #################################################################################################
#  Windowed (cache-relative) KV cache: sliding_window_cache=1
# #################################################################################################


def _windowed_make_session(config: GQAConfig, ort_type, providers=None):
    onnx_model_str = create_group_query_attention_graph_past(config, ort_type=ort_type, share_buffer=True)
    if providers is None:
        providers = [resolve_cuda_plugin_ep("CUDAExecutionProvider")]
    return InferenceSession(onnx_model_str, SessionOptions(), providers=providers)


def _windowed_resident_count(total_length: int, capacity: int, window: int) -> int:
    """`L(T)` from the operator spec: the number of KV positions resident after a step."""
    if total_length <= capacity:
        return total_length
    gap = capacity - window + 1
    overflow = total_length - capacity
    return total_length - gap * ((overflow + gap - 1) // gap)


def _windowed_run_steps(
    base_config: GQAConfig,
    buffer_sequence_length: int,
    sliding_window_cache: int,
    step_lengths,
    q_all,
    k_all,
    v_all,
    cos,
    sin,
    device,
    ort_type,
    torch_type,
    head_sink=None,
    attention_bias_all=None,
    position_ids_all=None,
    k_scale=None,
    v_scale=None,
    providers=None,
):
    """Drives a GroupQueryAttention node token-chunk by token-chunk over a shared past/present buffer.

    `q_all`/`k_all`/`v_all` hold the whole sequence in BSNH layout, so the same inputs can be replayed
    against a full-length cache and against a windowed one. Returns the per-step `output` tensors
    together with the final `present_key` / `present_value` buffers.
    """
    batch_size = base_config.batch_size
    kv_hidden_size = base_config.kv_num_heads * base_config.head_size
    q_hidden_size = base_config.num_heads * base_config.head_size

    cache_torch_type = TORCH_DTYPE_MAP[base_config.kv_cache_type] if base_config.kv_cache_type else torch_type
    cache_ort_type = ONNX_TENSOR_TYPE_MAP[base_config.kv_cache_type] if base_config.kv_cache_type else ort_type
    dense_head_size = base_config.head_size // 2 if base_config.kv_cache_type == "int4" else base_config.head_size
    cache_shape = (batch_size, base_config.kv_num_heads, buffer_sequence_length, dense_head_size)
    cache_k = torch.zeros(cache_shape, dtype=cache_torch_type, device=device)
    cache_v = torch.zeros(cache_shape, dtype=cache_torch_type, device=device)

    sessions = {}
    outputs = []
    past_length = 0

    for step_length in step_lengths:
        config = deepcopy(base_config)
        config.q_sequence_length = step_length
        config.kv_sequence_length = step_length
        config.past_kv_sequence_length = past_length
        config.buffer_sequence_length = buffer_sequence_length
        config.sliding_window_cache = sliding_window_cache

        if step_length not in sessions:
            sessions[step_length] = _windowed_make_session(config, ort_type, providers=providers)
        session = sessions[step_length]

        io_binding = session.io_binding()

        q = q_all[:, past_length : past_length + step_length].reshape(batch_size, step_length, q_hidden_size)
        k = k_all[:, past_length : past_length + step_length].reshape(batch_size, step_length, kv_hidden_size)
        v = v_all[:, past_length : past_length + step_length].reshape(batch_size, step_length, kv_hidden_size)
        # Keep references: io_binding only records data pointers, so temporaries would be freed.
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        bind_tensor(io_binding, "query", q, device, ort_type)
        bind_tensor(io_binding, "key", k, device, ort_type)
        bind_tensor(io_binding, "value", v, device, ort_type)
        bind_tensor(io_binding, "past_key", cache_k, device, cache_ort_type)
        bind_tensor(io_binding, "past_value", cache_v, device, cache_ort_type)

        total_length = past_length + step_length
        seqlens_k = torch.full((batch_size,), total_length - 1, dtype=torch.int32, device=device)
        bind_tensor(io_binding, "seqlens_k", seqlens_k, device, TensorProto.INT32)
        cpu_device = torch.device("cpu")
        # Keep a reference: io_binding only records the data pointer, so a temporary would be freed.
        total_sequence_length = torch.tensor([total_length], dtype=torch.int32, device=cpu_device)
        bind_tensor(
            io_binding,
            "total_sequence_length",
            total_sequence_length,
            cpu_device,
            TensorProto.INT32,
        )

        if cos is not None:
            bind_tensor(io_binding, "cos_cache", cos, device, ort_type)
            bind_tensor(io_binding, "sin_cache", sin, device, ort_type)

        if position_ids_all is not None:
            position_ids = position_ids_all[:, past_length:total_length].contiguous()
            bind_tensor(io_binding, "position_ids", position_ids, device, TensorProto.INT64)

        if attention_bias_all is not None:
            attention_bias = attention_bias_all[:, :, past_length:total_length, :total_length].contiguous()
            bind_tensor(io_binding, "attention_bias", attention_bias, device, ort_type)

        if head_sink is not None:
            bind_tensor(io_binding, "head_sink", head_sink, device, ort_type)

        if k_scale is not None:
            bind_tensor(io_binding, "k_scale", k_scale, device, TensorProto.FLOAT)
            bind_tensor(io_binding, "v_scale", v_scale, device, TensorProto.FLOAT)

        out = torch.zeros((batch_size, step_length, q_hidden_size), dtype=torch_type, device=device)
        bind_output_tensor(io_binding, "output", out, device, ort_type)
        io_binding.bind_output("present_key", device, 0, cache_ort_type, tuple(cache_k.shape), cache_k.data_ptr())
        io_binding.bind_output("present_value", device, 0, cache_ort_type, tuple(cache_v.shape), cache_v.data_ptr())

        io_binding.synchronize_inputs()
        session.run_with_iobinding(io_binding)
        io_binding.synchronize_outputs()

        outputs.append(out.clone())
        past_length = total_length

    return outputs, cache_k, cache_v


class TestGQAWindowedKvCache(unittest.TestCase):
    """`sliding_window_cache=1` stores only the most recent `capacity` positions of a sliding-window
    layer, in cache-relative coordinates. Because RoPE stays absolute and the retained positions are
    exactly those the window can attend to, results must match a full-length cache."""

    max_length = 1024
    window_size = 128
    # CUDA requires the cache capacity to equal local_window_size; the CPU subclass overrides this.
    slack = 0

    device = "cuda"
    torch_type = torch.float16
    ort_type = TensorProto.FLOAT16
    providers = None  # None -> resolve the CUDA (plugin) EP.

    def _require_ep(self):
        if not has_flash_attention():
            self.skipTest("Flash Attention is not available, skipping test.")

    def _base_config(self, **overrides):
        config = GQAConfig(
            batch_size=2,
            q_sequence_length=1,
            kv_sequence_length=1,
            num_heads=8,
            kv_num_heads=2,
            head_size=64,
            local_window_size=self.window_size,
            share_buffer=True,
            rope_cache_length=self.max_length,
        )
        for name, value in overrides.items():
            setattr(config, name, value)
        return config

    def _check_parity(self, base_config, step_lengths, rtol=1e-3, atol=1e-3, buffer_sequence_length=None):
        self._require_ep()

        device = self.device
        torch_type = self.torch_type
        ort_type = self.ort_type
        capacity = buffer_sequence_length or min(self.max_length, self.window_size + self.slack)
        total_length = sum(step_lengths)
        self.assertLessEqual(total_length, self.max_length)

        torch.manual_seed(0)
        shape = (base_config.batch_size, total_length, base_config.num_heads, base_config.head_size)
        kv_shape = (base_config.batch_size, total_length, base_config.kv_num_heads, base_config.head_size)
        q_all = torch.randn(shape, device=device, dtype=torch_type) * 0.2
        k_all = torch.randn(kv_shape, device=device, dtype=torch_type) * 0.2
        v_all = torch.randn(kv_shape, device=device, dtype=torch_type) * 0.2

        cos, sin = None, None
        if base_config.rotary:
            rotary_dim = math.floor(base_config.head_size / 16) * 16
            angle = torch.rand(self.max_length, rotary_dim // 2, device=device) * 2 * math.pi
            cos = torch.cos(angle).to(dtype=torch_type)
            sin = torch.sin(angle).to(dtype=torch_type)

        head_sink = None
        if base_config.has_head_sink:
            head_sink = torch.rand(base_config.num_heads, dtype=torch_type, device=device)

        attention_bias_all = None
        if base_config.has_attention_bias:
            bias_shape = (
                1 if base_config.attention_bias_broadcast_dim_0 else base_config.batch_size,
                base_config.num_heads if base_config.attention_bias_per_head else 1,
                total_length,
                total_length,
            )
            attention_bias_all = torch.randn(bias_shape, dtype=torch_type, device=device) * 0.5

        position_ids_all = None
        if base_config.has_position_ids:
            position_ids_all = (torch.arange(total_length, dtype=torch.int64, device=device) * 7) % self.max_length
            position_ids_all = position_ids_all.unsqueeze(0)
            position_ids_all = position_ids_all.expand(base_config.batch_size, -1).contiguous()

        k_scale, v_scale = None, None
        if base_config.k_quant_type != "NONE":
            k_scale, v_scale = get_static_scale(base_config, device, torch_type, 0.2)
            k_scale = k_scale.to(torch.float32).contiguous()
            v_scale = v_scale.to(torch.float32).contiguous()

        common = {
            "step_lengths": step_lengths,
            "q_all": q_all,
            "k_all": k_all,
            "v_all": v_all,
            "cos": cos,
            "sin": sin,
            "device": device,
            "ort_type": ort_type,
            "torch_type": torch_type,
            "head_sink": head_sink,
            "attention_bias_all": attention_bias_all,
            "position_ids_all": position_ids_all,
            "k_scale": k_scale,
            "v_scale": v_scale,
            "providers": self.providers,
        }
        reference, reference_cache_k, reference_cache_v = _windowed_run_steps(base_config, self.max_length, 0, **common)
        if base_config.k_quant_type != "NONE" and any(torch.isnan(step).any() for step in reference):
            # A quantized KV cache is only read correctly by the flash-attention prefill kernels.
            # Builds without them route the first prompt through memory-efficient attention, which
            # reinterprets the quantized cache as unquantized and produces NaN for both the
            # windowed and the full-length run, so there is nothing to compare.
            self.skipTest("quantized prefill needs a build with flash attention enabled")
        windowed, cache_k, cache_v = _windowed_run_steps(base_config, capacity, 1, **common)

        for step_index, (expected, actual) in enumerate(zip(reference, windowed, strict=True)):
            numpy.testing.assert_allclose(
                actual.float().cpu().numpy(),
                expected.float().cpu().numpy(),
                rtol=rtol,
                atol=atol,
                err_msg=f"mismatch at step {step_index} (step_lengths={step_lengths})",
            )

        # Layout contract: rows [0, L) of present_key/present_value hold the L most recent positions
        # in increasing position order. The full-length reference run stores every position at its
        # absolute row, so its tail is the exact byte-for-byte expectation for the windowed rows --
        # this holds for RoPE'd and quantized caches too, because both runs write the same values.
        resident = _windowed_resident_count(total_length, capacity, base_config.local_window_size)
        for name, actual_cache, reference_cache in (
            ("present_key", cache_k, reference_cache_k),
            ("present_value", cache_v, reference_cache_v),
        ):
            numpy.testing.assert_array_equal(
                actual_cache[:, :, :resident].cpu().numpy(),
                reference_cache[:, :, total_length - resident : total_length].cpu().numpy(),
                err_msg=f"{name} resident range mismatch (L={resident}, T={total_length}, C={capacity})",
            )

    def test_prompt_shorter_than_capacity_then_decode(self):
        # Prompt fits in the cache; eviction only starts once the window slides past the capacity.
        steps = [64, *([1] * 400)]
        self._check_parity(self._base_config(), steps)

    def test_prompt_longer_than_capacity_then_decode(self):
        # Prompt exceeds the capacity, so the first call must attend over the full prompt while
        # seeding the cache with only the tail. Exercises the windowed first-prompt path.
        steps = [512, *([1] * 200)]
        self._check_parity(self._base_config(), steps)

    def test_chunked_prefill(self):
        # Multi-token chunks arriving on a partially filled cache.
        # GroupQueryAttention only allows a subsequent prompt (1 < S < T) at batch_size 1.
        steps = [128] * 6 + [1] * 32
        self._check_parity(self._base_config(batch_size=1), steps)

    def test_eviction_boundaries(self):
        # Land exactly on capacity - 1, capacity and capacity + 1 with single-token steps.
        capacity = self.window_size + self.slack
        steps = [capacity - 2, *([1] * 8)]
        self._check_parity(self._base_config(), steps)

    def test_with_rotary(self):
        # RoPE positions stay absolute even though the cache is indexed relative to its start.
        steps = [64, *([1] * 400)]
        self._check_parity(self._base_config(rotary=True), steps)

    def test_gpt_oss_shape_with_head_sink(self):
        # gpt-oss: attention sinks, RoPE, and alternating 128-token sliding-window layers.
        steps = [512, *([1] * 200)]
        self._check_parity(
            self._base_config(num_heads=8, kv_num_heads=2, head_size=64, rotary=True, has_head_sink=True), steps
        )

    @unittest.skipIf(not has_quantized_kv_cache(), "Quantized KV cache is not available, skipping test.")
    def test_int8_quantized_cache(self):
        # gpt-oss stores the cache as PER_CHANNEL int8; eviction and staging must move the
        # quantized rows around without dequantizing them.
        steps = [512, *([1] * 200)]
        self._check_parity(
            self._base_config(
                num_heads=8,
                kv_num_heads=2,
                head_size=64,
                rotary=True,
                has_head_sink=True,
                kv_cache_type="int8",
                k_quant_type="PER_CHANNEL",
                v_quant_type="PER_CHANNEL",
                kv_cache_bit_width=8,
            ),
            steps,
            rtol=5e-2,
            atol=1e-1,
        )

    def test_batch_one_packed_head_layout(self):
        steps = [200, *([1] * 300)]
        self._check_parity(self._base_config(batch_size=1, num_heads=4, kv_num_heads=4, head_size=128), steps)

    def test_capacity_equal_to_window_size(self):
        # C == W leaves no slack, so a multi-token step needs more entries than the cache can hold.
        # The op stages such a step, so this must still match a full cache exactly.
        capacity = self.window_size
        self._check_parity(
            self._base_config(batch_size=1), step_lengths=[32, 16, 1, 1], buffer_sequence_length=capacity
        )

    def test_capacity_larger_than_window(self):
        # CUDA evicts the minimum number of rows per step, which reproduces the documented layout
        # only when there is no slack above the window, so C > W is rejected.
        self._require_ep()

        with self.assertRaisesRegex(Exception, "sliding_window_cache"):
            self._check_parity(
                self._base_config(batch_size=1),
                step_lengths=[32, 1, 1],
                buffer_sequence_length=self.window_size + 8,
            )

    def test_capacity_too_small_is_rejected(self):
        self._require_ep()

        device = self.device
        torch_type = self.torch_type
        ort_type = self.ort_type
        # A cache shorter than the attention window can never hold everything the window covers.
        capacity = self.window_size - 1
        config = self._base_config(rope_cache_length=capacity)
        with self.assertRaisesRegex(Exception, "sliding_window_cache"):
            _windowed_run_steps(
                config,
                capacity,
                1,
                step_lengths=[32, 16],
                q_all=torch.zeros(
                    (config.batch_size, 48, config.num_heads, config.head_size), device=device, dtype=torch_type
                ),
                k_all=torch.zeros(
                    (config.batch_size, 48, config.kv_num_heads, config.head_size), device=device, dtype=torch_type
                ),
                v_all=torch.zeros(
                    (config.batch_size, 48, config.kv_num_heads, config.head_size), device=device, dtype=torch_type
                ),
                cos=None,
                sin=None,
                device=device,
                ort_type=ort_type,
                torch_type=torch_type,
                providers=self.providers,
            )

    def test_requires_local_window_size(self):
        self._require_ep()

        config = self._base_config(
            local_window_size=-1,
            sliding_window_cache=1,
            buffer_sequence_length=self.window_size + self.slack,
            rope_cache_length=self.window_size + self.slack,
        )
        with self.assertRaisesRegex(Exception, "sliding_window_cache"):
            _windowed_make_session(config, self.ort_type, providers=self.providers)


class TestGQAWindowedKvCacheCpu(TestGQAWindowedKvCache):
    """Same parity contract as the CUDA suite, run against the CPU GroupQueryAttention kernel."""

    device = "cpu"
    torch_type = torch.float32
    ort_type = TensorProto.FLOAT
    providers: typing.ClassVar[list[str]] = ["CPUExecutionProvider"]
    # The CPU kernel accepts any capacity at or above the window and uses the slack to amortize
    # compaction, so it is exercised with a capacity well above local_window_size.
    slack = 256

    def _require_ep(self):
        pass

    def _base_config(self, **overrides):
        # The float32 CPU kernel keeps an unquantized float32 cache by default.
        overrides.setdefault("kv_cache_type", "float32")
        return super()._base_config(**overrides)

    def test_capacity_larger_than_window(self):
        # CPU accepts slack above the window; the resident count then sawtooths between W and C
        # instead of staying at min(T, C). Exercised in depth by the two small-slack tests below.
        self._check_parity(
            self._base_config(batch_size=1),
            step_lengths=[32, 1, 1],
            buffer_sequence_length=self.window_size + 8,
        )

    def test_small_slack_many_compactions(self):
        # C == W + 8 reclaims only 9 rows per compaction, so a long decode run crosses the
        # compaction boundary dozens of times instead of once or twice.
        self._check_parity(
            self._base_config(), step_lengths=[64, *([1] * 400)], buffer_sequence_length=self.window_size + 8
        )

    def test_chunked_prefill_small_slack(self):
        # 32-token chunks onto a cache with 9 free rows: once the cache has filled, every chunk
        # drops rows its own first query still reads, so it must stage from a drifted append point.
        # GroupQueryAttention only allows a subsequent prompt (1 < S < T) at batch_size 1.
        self._check_parity(
            self._base_config(batch_size=1),
            step_lengths=[32] * 10 + [1] * 20,
            buffer_sequence_length=self.window_size + 8,
        )

    def test_attention_bias_with_position_ids(self):
        self._check_parity(
            self._base_config(
                rotary=True,
                has_position_ids=True,
                has_attention_bias=True,
                attention_bias_per_head=True,
            ),
            step_lengths=[64, *([1] * 400)],
        )

    def test_attention_bias_with_position_ids_non_flash(self):
        with scoped_env_var("ORT_GQA_DISABLE_FLASH_ATTENTION", "1"):
            self._check_parity(
                self._base_config(
                    rotary=True,
                    has_position_ids=True,
                    has_attention_bias=True,
                    attention_bias_per_head=True,
                ),
                step_lengths=[self.window_size + self.slack, 1, 1],
            )


if __name__ == "__main__":
    unittest.main()
