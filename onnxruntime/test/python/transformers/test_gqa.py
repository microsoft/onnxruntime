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
import math
import os
import platform
import random
import unittest
from dataclasses import dataclass

import numpy
import torch
from einops import rearrange, repeat
from onnx import TensorProto, helper
from parameterized import parameterized

from onnxruntime import InferenceSession, OrtValue, SessionOptions, get_available_providers

# Set seed for reproducibility
torch.manual_seed(0)
random.seed(69)

# Reduces number of tests to run for faster pipeline checks
pipeline_mode = os.getenv("PIPELINE_MODE", "1") == "1"

# #################################################################################################
#  Configuration and Helper Classes
# #################################################################################################


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
    rotary: bool = False
    rotary_interleaved: bool = False
    packed: bool = False
    softcap: float = 0.0
    use_smooth_softmax: bool = False
    # CPU-only parameters
    has_position_ids: bool = False
    has_attention_bias: bool = False
    has_head_sink: bool = False


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
    from rotary_flash import apply_rotary_emb  # noqa: PLC0415

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


def create_group_query_attention_graph_prompt(
    config: GQAConfig,
    ort_type,
    share_buffer=True,
):
    assert not (config.has_head_sink and config.use_smooth_softmax)
    past_kv_seqlen = config.buffer_sequence_length if share_buffer else 0
    present_kv_seqlen = config.buffer_sequence_length if share_buffer else config.kv_sequence_length

    nodes = [
        helper.make_node(
            op_type="GroupQueryAttention",
            inputs=[
                "query",
                "key" if not config.packed else "",
                "value" if not config.packed else "",
                "past_key" if share_buffer else "",
                "past_value" if share_buffer else "",
                "seqlens_k",
                "total_sequence_length",
                "cos_cache" if config.rotary else "",
                "sin_cache" if config.rotary else "",
                "position_ids" if config.has_position_ids else "",
                "attention_bias" if config.has_attention_bias else "",
                "head_sink" if config.has_head_sink else "",
            ],
            outputs=["output", "present_key", "present_value"],
            name="GroupQueryAttention_0",
            num_heads=config.num_heads,
            kv_num_heads=config.kv_num_heads,
            local_window_size=config.local_window_size,
            do_rotary=config.rotary,
            rotary_interleaved=config.rotary_interleaved,
            softcap=config.softcap,
            smooth_softmax=1 if config.use_smooth_softmax else 0,
            domain="com.microsoft",
        ),
    ]

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
    if share_buffer:
        # Shape is (batch_size, kv_num_heads, sequence_length, head_size)
        k_shape = [config.batch_size, config.kv_num_heads, past_kv_seqlen, config.head_size]
        v_shape = k_shape
        graph_input.extend(
            [
                helper.make_tensor_value_info("past_key", ort_type, k_shape),
                helper.make_tensor_value_info("past_value", ort_type, v_shape),
            ]
        )
    if config.rotary:
        rotary_dim = (math.floor(config.head_size / 16) * 16) // 2
        cache_seq_len = config.buffer_sequence_length if share_buffer else config.kv_sequence_length
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
        graph_input.append(
            helper.make_tensor_value_info(
                "attention_bias", ort_type, [config.batch_size, 1, config.q_sequence_length, config.kv_sequence_length]
            )
        )
    if config.has_head_sink:
        graph_input.append(helper.make_tensor_value_info("head_sink", ort_type, [config.num_heads]))

    # Shape is (batch_size, kv_num_heads, sequence_length, head_size)
    output_k_shape = [config.batch_size, config.kv_num_heads, present_kv_seqlen, config.head_size]
    output_v_shape = output_k_shape

    graph_output = [
        helper.make_tensor_value_info(
            "output", ort_type, [config.batch_size, config.q_sequence_length, config.num_heads * config.head_size]
        ),
        helper.make_tensor_value_info("present_key", ort_type, output_k_shape),
        helper.make_tensor_value_info("present_value", ort_type, output_v_shape),
    ]

    graph = helper.make_graph(nodes, "GroupQueryAttention_Graph", graph_input, graph_output)
    model = helper.make_model(graph)
    return model.SerializeToString()


def create_group_query_attention_graph_past(
    config: GQAConfig,
    ort_type,
    share_buffer=True,
):
    assert not (config.has_head_sink and config.use_smooth_softmax)

    if share_buffer:
        past_kv_seqlen = config.buffer_sequence_length
        present_kv_seqlen = config.buffer_sequence_length
    else:
        past_kv_seqlen = config.past_kv_sequence_length
        present_kv_seqlen = config.past_kv_sequence_length + config.kv_sequence_length

    nodes = [
        helper.make_node(
            "GroupQueryAttention",
            [
                "query",
                "key" if not config.packed else "",
                "value" if not config.packed else "",
                "past_key",
                "past_value",
                "seqlens_k",
                "total_sequence_length",
                "cos_cache" if config.rotary else "",
                "sin_cache" if config.rotary else "",
                "position_ids" if config.has_position_ids else "",
                "attention_bias" if config.has_attention_bias else "",
                "head_sink" if config.has_head_sink else "",
            ],
            ["output", "present_key", "present_value"],
            "GroupQueryAttention_0",
            num_heads=config.num_heads,
            kv_num_heads=config.kv_num_heads,
            local_window_size=config.local_window_size,
            do_rotary=config.rotary,
            rotary_interleaved=config.rotary_interleaved,
            softcap=config.softcap,
            smooth_softmax=1 if config.use_smooth_softmax else 0,
            domain="com.microsoft",
        ),
    ]

    q_hidden_size = (
        (config.num_heads * config.head_size)
        if not config.packed
        else (config.num_heads * config.head_size + 2 * config.kv_num_heads * config.head_size)
    )
    # Shape is (batch_size, kv_num_heads, sequence_length, head_size)
    past_k_shape = [config.batch_size, config.kv_num_heads, past_kv_seqlen, config.head_size]
    graph_input = [
        helper.make_tensor_value_info("query", ort_type, [config.batch_size, config.q_sequence_length, q_hidden_size]),
        helper.make_tensor_value_info("past_key", ort_type, past_k_shape),
        helper.make_tensor_value_info("past_value", ort_type, past_k_shape),
        helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [config.batch_size]),
        helper.make_tensor_value_info("total_sequence_length", TensorProto.INT32, [1]),
    ]

    if not config.packed:
        graph_input.extend(
            [
                helper.make_tensor_value_info(
                    "key",
                    ort_type,
                    [config.batch_size, config.q_sequence_length, config.kv_num_heads * config.head_size],
                ),
                helper.make_tensor_value_info(
                    "value",
                    ort_type,
                    [config.batch_size, config.q_sequence_length, config.kv_num_heads * config.head_size],
                ),
            ]
        )

    if config.rotary:
        rotary_dim = (math.floor(config.head_size / 16) * 16) // 2
        cache_len = config.buffer_sequence_length
        graph_input.extend(
            [
                helper.make_tensor_value_info("cos_cache", ort_type, [cache_len, rotary_dim]),
                helper.make_tensor_value_info("sin_cache", ort_type, [cache_len, rotary_dim]),
            ]
        )

    if config.has_position_ids:
        graph_input.append(
            helper.make_tensor_value_info(
                "position_ids", TensorProto.INT64, [config.batch_size, config.q_sequence_length]
            )
        )
    if config.has_attention_bias:
        graph_input.append(
            helper.make_tensor_value_info(
                "attention_bias", ort_type, [config.batch_size, 1, config.q_sequence_length, present_kv_seqlen]
            )
        )
    if config.has_head_sink:
        graph_input.append(helper.make_tensor_value_info("head_sink", ort_type, [config.num_heads]))

    output_k_shape = [
        config.batch_size,
        config.kv_num_heads,
        present_kv_seqlen,
        config.head_size,
    ]

    graph_output = [
        helper.make_tensor_value_info(
            "output", ort_type, [config.batch_size, config.q_sequence_length, config.num_heads * config.head_size]
        ),
        helper.make_tensor_value_info("present_key", ort_type, output_k_shape),
        helper.make_tensor_value_info("present_value", ort_type, output_k_shape),
    ]

    graph = helper.make_graph(nodes, "GroupQueryAttention_Graph", graph_input, graph_output)
    model = helper.make_model(graph)
    return model.SerializeToString()


# #################################################################################################
#  ONNX Runtime Execution Functions
# #################################################################################################


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
    ep,
    device,
    share_buffer=True,
    ort_type=TensorProto.FLOAT16,
    numpy_type=numpy.float16,
):
    onnx_model_str = create_group_query_attention_graph_prompt(
        config=config,
        ort_type=ort_type,
        share_buffer=share_buffer,
    )

    q = torch.reshape(q, (config.batch_size, config.q_sequence_length, -1))
    if new_k is not None:
        new_k = torch.reshape(new_k, (config.batch_size, config.kv_sequence_length, -1))
        new_v = torch.reshape(new_v, (config.batch_size, config.kv_sequence_length, -1))

    sess_options = SessionOptions()
    ort_session = InferenceSession(onnx_model_str, sess_options, providers=[ep])
    io_binding = ort_session.io_binding()

    # Common inputs
    ort_inputs = {
        "query": q.detach().cpu().numpy(),
        "seqlens_k": seqlens_k.detach().cpu().numpy().astype(numpy.int32),
        "total_sequence_length": torch.tensor([config.q_sequence_length], dtype=torch.int32).detach().cpu().numpy(),
    }
    io_binding.bind_cpu_input("query", ort_inputs["query"])
    io_binding.bind_cpu_input("seqlens_k", ort_inputs["seqlens_k"])
    io_binding.bind_cpu_input("total_sequence_length", ort_inputs["total_sequence_length"])

    if new_k is not None:
        ort_inputs["key"] = new_k.detach().cpu().numpy()
        ort_inputs["value"] = new_v.detach().cpu().numpy()
        io_binding.bind_cpu_input("key", ort_inputs["key"])
        io_binding.bind_cpu_input("value", ort_inputs["value"])
    if cos is not None:
        ort_inputs["cos_cache"] = cos.detach().cpu().numpy()
        ort_inputs["sin_cache"] = sin.detach().cpu().numpy()
        io_binding.bind_cpu_input("cos_cache", ort_inputs["cos_cache"])
        io_binding.bind_cpu_input("sin_cache", ort_inputs["sin_cache"])

    # CPU-specific inputs
    if config.has_position_ids:
        ort_inputs["position_ids"] = position_ids.detach().cpu().numpy()
        io_binding.bind_cpu_input("position_ids", ort_inputs["position_ids"])
    if config.has_attention_bias:
        ort_inputs["attention_bias"] = attention_bias.detach().cpu().numpy()
        io_binding.bind_cpu_input("attention_bias", ort_inputs["attention_bias"])
    if config.has_head_sink:
        ort_inputs["head_sink"] = head_sink.detach().cpu().numpy()
        io_binding.bind_cpu_input("head_sink", ort_inputs["head_sink"])

    if share_buffer:
        past_k_ort = OrtValue.ortvalue_from_numpy(k.detach().cpu().numpy(), device, 0)
        past_v_ort = OrtValue.ortvalue_from_numpy(v.detach().cpu().numpy(), device, 0)
        io_binding.bind_input("past_key", device, 0, numpy_type, past_k_ort.shape(), past_k_ort.data_ptr())
        io_binding.bind_input("past_value", device, 0, numpy_type, past_v_ort.shape(), past_v_ort.data_ptr())
        io_binding.bind_output("output")
        io_binding.bind_ortvalue_output("present_key", past_k_ort)
        io_binding.bind_ortvalue_output("present_value", past_v_ort)
    else:
        io_binding.bind_output("output")
        io_binding.bind_output("present_key")
        io_binding.bind_output("present_value")

    ort_session.run_with_iobinding(io_binding)
    ort_output, present_k, present_v = io_binding.copy_outputs_to_cpu()
    return torch.tensor(ort_output), present_k, present_v


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
    ep,
    device,
    share_buffer=True,
    ort_type=TensorProto.FLOAT16,
    numpy_type=numpy.float16,
):
    onnx_model_str = create_group_query_attention_graph_past(
        config=config,
        ort_type=ort_type,
        share_buffer=share_buffer,
    )

    q = torch.reshape(q, (config.batch_size, config.q_sequence_length, -1))
    if new_k is not None:
        new_k = torch.reshape(new_k, (config.batch_size, config.q_sequence_length, -1))
        new_v = torch.reshape(new_v, (config.batch_size, config.q_sequence_length, -1))

    sess_options = SessionOptions()
    ort_session = InferenceSession(onnx_model_str, sess_options, providers=[ep])
    io_binding = ort_session.io_binding()

    # Common inputs
    total_seq_len = (
        config.past_kv_sequence_length if share_buffer else config.past_kv_sequence_length + config.q_sequence_length
    )
    ort_inputs = {
        "query": q.detach().cpu().numpy(),
        "seqlens_k": seqlens_k.detach().cpu().numpy().astype(numpy.int32),
        "total_sequence_length": torch.tensor([total_seq_len], dtype=torch.int32).detach().cpu().numpy(),
    }
    io_binding.bind_cpu_input("query", ort_inputs["query"])
    io_binding.bind_cpu_input("seqlens_k", ort_inputs["seqlens_k"])
    io_binding.bind_cpu_input("total_sequence_length", ort_inputs["total_sequence_length"])

    if new_k is not None:
        ort_inputs["key"] = new_k.detach().cpu().numpy()
        ort_inputs["value"] = new_v.detach().cpu().numpy()
        io_binding.bind_cpu_input("key", ort_inputs["key"])
        io_binding.bind_cpu_input("value", ort_inputs["value"])
    if cos is not None:
        ort_inputs["cos_cache"] = cos.detach().cpu().numpy()
        ort_inputs["sin_cache"] = sin.detach().cpu().numpy()
        io_binding.bind_cpu_input("cos_cache", ort_inputs["cos_cache"])
        io_binding.bind_cpu_input("sin_cache", ort_inputs["sin_cache"])

    # CPU-specific inputs
    if config.has_position_ids:
        ort_inputs["position_ids"] = position_ids.detach().cpu().numpy()
        io_binding.bind_cpu_input("position_ids", ort_inputs["position_ids"])
    if config.has_attention_bias:
        ort_inputs["attention_bias"] = attention_bias.detach().cpu().numpy()
        io_binding.bind_cpu_input("attention_bias", ort_inputs["attention_bias"])
    if config.has_head_sink:
        ort_inputs["head_sink"] = head_sink.detach().cpu().numpy()
        io_binding.bind_cpu_input("head_sink", ort_inputs["head_sink"])

    # Binding past and present KV
    if share_buffer:
        past_k_ort = OrtValue.ortvalue_from_numpy(k.detach().cpu().numpy(), device, 0)
        past_v_ort = OrtValue.ortvalue_from_numpy(v.detach().cpu().numpy(), device, 0)
        io_binding.bind_input("past_key", device, 0, numpy_type, past_k_ort.shape(), past_k_ort.data_ptr())
        io_binding.bind_input("past_value", device, 0, numpy_type, past_v_ort.shape(), past_v_ort.data_ptr())
        io_binding.bind_output("output")
        io_binding.bind_ortvalue_output("present_key", past_k_ort)
        io_binding.bind_ortvalue_output("present_value", past_v_ort)
    else:
        ort_inputs["past_key"] = k.detach().cpu().numpy()
        ort_inputs["past_value"] = v.detach().cpu().numpy()
        io_binding.bind_cpu_input("past_key", ort_inputs["past_key"])
        io_binding.bind_cpu_input("past_value", ort_inputs["past_value"])
        io_binding.bind_output("output")
        io_binding.bind_output("present_key")
        io_binding.bind_output("present_value")

    ort_session.run_with_iobinding(io_binding)
    ort_output, present_k, present_v = io_binding.copy_outputs_to_cpu()
    return torch.tensor(ort_output), present_k, present_v


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
    b, n, s, t = x.shape
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


def parity_check_gqa_prompt(
    config: GQAConfig,
    ep,
    device,
    torch_type,
    numpy_type,
    ort_type,
    causal,
    rtol,
    atol,
):
    # Q/K/V have normal distribution with mean = 0 and standard deviation = 0.02.
    # If we use standard deviation = 1, numerical stability issues may occur.
    std = 0.02

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

    # k and v are the cache buffers, created in BNSH format
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
    v = torch.randn_like(k)

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

    # --- PyTorch Reference Path ---
    # Transpose BNSH cache to BSNH format for reference implementation
    k_cache_ref = k.clone().transpose(1, 2)
    v_cache_ref = v.clone().transpose(1, 2)

    cache_seqlens = torch.full((config.batch_size,), config.kv_sequence_length, device=device, dtype=torch.int32)
    rotary_seqlens = torch.zeros(config.batch_size, device=device, dtype=torch.long)

    cos, sin, q_ro, k_ro = None, None, q, new_k
    if config.rotary:
        rotary_dim = math.floor(config.head_size / 16) * 16
        angle = torch.rand(config.buffer_sequence_length, rotary_dim // 2, device=device) * 2 * math.pi
        cos = torch.cos(angle).to(dtype=torch_type)
        sin = torch.sin(angle).to(dtype=torch_type)
        q_ro = apply_rotary_embedding(q.clone(), cos, sin, rotary_seqlens, config.rotary_interleaved, device)
        k_ro = apply_rotary_embedding(new_k.clone(), cos, sin, rotary_seqlens, config.rotary_interleaved, device)

    position_ids = None
    attention_bias = None
    if ep == "CPUExecutionProvider":
        if config.has_position_ids:
            position_ids = (
                torch.arange(config.q_sequence_length, device=device).unsqueeze(0).expand(config.batch_size, -1)
            )
        if config.has_attention_bias:
            attention_bias = torch.zeros(
                config.batch_size,
                1,
                config.q_sequence_length,
                config.kv_sequence_length,
                device=device,
                dtype=torch_type,
            )

    arange = rearrange(torch.arange(config.buffer_sequence_length, device=device), "s -> 1 s")
    kv_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
    update_mask = arange < kv_seqlens_expanded

    k_cache_ref[update_mask] = rearrange(k_ro, "b s ... -> (b s) ...").to(dtype=torch_type)
    v_cache_ref[update_mask] = rearrange(new_v, "b s ... -> (b s) ...").to(dtype=torch_type)
    key_padding_mask = arange < kv_seqlens_expanded

    out_ref, _ = attention_ref(
        q=q_ro,
        k=k_cache_ref,
        v=v_cache_ref,
        query_padding_mask=None,
        key_padding_mask=key_padding_mask,
        attention_bias=attention_bias,
        causal=True,
        window_size=window_size,
        softcap=config.softcap,
        use_smooth_softmax=config.use_smooth_softmax,
        head_sink=head_sink,
    )
    out_ref_np = out_ref.detach().cpu().numpy()

    # Transpose reference cache back to BNSH for comparison
    k_cache_ref_np = k_cache_ref.transpose(1, 2).detach().cpu().numpy()
    v_cache_ref_np = v_cache_ref.transpose(1, 2).detach().cpu().numpy()

    # --- ONNX Runtime Path ---
    q_ort, k_ort, v_ort, new_k_ort, new_v_ort = q, k, v, new_k, new_v
    if config.packed:
        q_ort = torch.cat([q, new_k, new_v], dim=2)
        new_k_ort, new_v_ort = None, None

    # seqlens_k for GQA op is past_seq_len + seq_len - 1
    ort_seqlens = cache_seqlens - 1
    out, present_k, present_v = gqa_prompt_func(
        q=q_ort,
        k=k_ort,
        v=v_ort,
        config=config,
        new_k=new_k_ort,
        new_v=new_v_ort,
        cos=cos,
        sin=sin,
        seqlens_k=ort_seqlens,
        position_ids=position_ids,
        attention_bias=attention_bias,
        head_sink=head_sink,
        ep=ep,
        device=device,
        share_buffer=True,
        ort_type=ort_type,
        numpy_type=numpy_type,
    )
    out = torch.reshape(out, (config.batch_size, config.q_sequence_length, config.num_heads, config.head_size))
    out_np = out.detach().cpu().numpy()

    # --- Comparison ---
    numpy.testing.assert_allclose(present_k, k_cache_ref_np, rtol=rtol, atol=atol)
    numpy.testing.assert_allclose(present_v, v_cache_ref_np, rtol=rtol, atol=atol)
    numpy.testing.assert_allclose(out_np, out_ref_np, rtol=rtol, atol=atol)


def parity_check_gqa_past(
    config: GQAConfig,
    ep,
    device,
    torch_type,
    numpy_type,
    ort_type,
    causal,
    rtol,
    atol,
):
    # --- Test Data Generation ---
    q = torch.randn(
        config.batch_size,
        config.q_sequence_length,
        config.num_heads,
        config.head_size,
        device=device,
        dtype=torch_type,
    )
    # k and v are the cache buffers, created in BNSH format
    k = torch.randn(
        config.batch_size,
        config.kv_num_heads,
        config.buffer_sequence_length,
        config.head_size,
        device=device,
        dtype=torch_type,
    )
    v = torch.randn_like(k)
    new_k = torch.randn(
        config.batch_size,
        config.q_sequence_length,
        config.kv_num_heads,
        config.head_size,
        device=device,
        dtype=torch_type,
    )
    new_v = torch.randn_like(new_k)

    head_sink = torch.rand(config.num_heads, dtype=torch_type, device=device) if config.has_head_sink else None
    window_size = (-1, -1)
    if config.local_window_size > 0:
        window_size = (config.local_window_size, 0)
    elif causal:
        window_size = (-1, 0)

    # --- PyTorch Reference Path ---
    # Transpose BNSH cache to BSNH format for reference implementation
    k_cache_ref = k.clone().transpose(1, 2)
    v_cache_ref = v.clone().transpose(1, 2)

    cache_seqlens = torch.randint(
        0,
        config.past_kv_sequence_length - config.q_sequence_length + 1,
        (config.batch_size,),
        device=device,
        dtype=torch.long,
    )

    cos, sin, q_ro, k_ro = None, None, q, new_k
    if config.rotary:
        rotary_dim = math.floor(config.head_size / 16) * 16
        angle = torch.rand(config.buffer_sequence_length, rotary_dim // 2, device=device) * 2 * math.pi
        cos = torch.cos(angle).to(dtype=torch_type)
        sin = torch.sin(angle).to(dtype=torch_type)
        q_ro = apply_rotary_embedding(q.clone(), cos, sin, cache_seqlens, config.rotary_interleaved, device)
        k_ro = apply_rotary_embedding(new_k.clone(), cos, sin, cache_seqlens, config.rotary_interleaved, device)

    position_ids = None
    attention_bias = None
    total_seq_len = config.past_kv_sequence_length
    if ep == "CPUExecutionProvider":
        if config.has_position_ids:
            position_ids = (cache_seqlens.unsqueeze(1) + torch.arange(config.q_sequence_length, device=device)).long()
        if config.has_attention_bias:
            attention_bias = torch.zeros(
                config.batch_size, 1, config.q_sequence_length, total_seq_len, device=device, dtype=torch_type
            )
            for b in range(config.batch_size):
                end_pos = cache_seqlens[b] + config.q_sequence_length
                attention_bias[b, :, :, end_pos:] = float("-inf")

    arange = rearrange(torch.arange(config.buffer_sequence_length, device=device), "s -> 1 s")
    cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
    update_mask = torch.logical_and(
        cache_seqlens_expanded <= arange, arange < cache_seqlens_expanded + config.q_sequence_length
    )
    k_cache_ref[update_mask] = rearrange(k_ro, "b s ... -> (b s) ...").to(dtype=torch_type)
    v_cache_ref[update_mask] = rearrange(new_v, "b s ... -> (b s) ...").to(dtype=torch_type)
    key_padding_mask = arange < cache_seqlens_expanded + config.q_sequence_length

    out_ref, _ = attention_ref(
        q=q_ro,
        k=k_cache_ref,
        v=v_cache_ref,
        query_padding_mask=None,
        key_padding_mask=key_padding_mask,
        attention_bias=attention_bias,
        causal=True,
        window_size=window_size,
        softcap=config.softcap,
        use_smooth_softmax=config.use_smooth_softmax,
        head_sink=head_sink,
    )
    out_ref_np = out_ref.detach().cpu().numpy()

    # Transpose reference cache back to BNSH for comparison
    k_cache_ref_np = k_cache_ref.transpose(1, 2).detach().cpu().numpy()
    v_cache_ref_np = v_cache_ref.transpose(1, 2).detach().cpu().numpy()

    # --- ONNX Runtime Path ---
    q_ort, k_ort, v_ort, new_k_ort, new_v_ort = q, k, v, new_k, new_v
    if config.packed:
        q_ort = torch.cat([q, new_k, new_v], dim=2)
        new_k_ort, new_v_ort = None, None

    ort_seqlens = cache_seqlens + config.q_sequence_length - 1
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
        ep=ep,
        device=device,
        share_buffer=True,
        ort_type=ort_type,
        numpy_type=numpy_type,
    )
    out = torch.reshape(out, (config.batch_size, config.q_sequence_length, config.num_heads, config.head_size))
    out_np = out.detach().cpu().numpy()

    numpy.testing.assert_allclose(present_k, k_cache_ref_np, rtol=rtol, atol=atol)
    numpy.testing.assert_allclose(present_v, v_cache_ref_np, rtol=rtol, atol=atol)
    numpy.testing.assert_allclose(out_np, out_ref_np, rtol=rtol, atol=atol)


# #################################################################################################
#  Test Case Generators
# #################################################################################################


def get_cuda_rotary_options():
    return [(False, False)] if pipeline_mode else [(True, False), (True, True), (False, False)]


def get_cpu_rotary_options():
    return [(False, False), (True, False), (True, True)]


def get_softmax_options(allow_head_sink: bool = True):
    head_sink_option = (False, True) if allow_head_sink else (False, False)
    return [(False, False), head_sink_option] if pipeline_mode else [(False, False), (False, True), (True, False)]


def gqa_cuda_prompt_test_cases(allow_head_sink: bool = True):
    batches = [3] if pipeline_mode else [1, 3, 5]
    seqs = [(35, 35)] if pipeline_mode else [(35, 35), (127, 127), (240, 240), (2000, 2000)]
    num_h = [(6, 3)] if pipeline_mode else [(6, 3), (9, 9), (32, 8)]
    h_sizes = [32] if pipeline_mode else [32, 64, 128, 256]
    smmoth_softmax__head_sink = get_softmax_options(allow_head_sink)

    for b in batches:
        for sq, skv in seqs:
            for n, n2 in num_h:
                for h in h_sizes:
                    for lws in [-1, random.randint(1, skv)]:
                        for rotary, rotary_interleaved in get_cuda_rotary_options():
                            for packed in [False, True]:
                                for softcap in [0.0, 50.0]:
                                    if rotary and h % 16 > 0:
                                        continue
                                    for use_smooth_softmax, has_head_sink in smmoth_softmax__head_sink:
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
                                            softcap=softcap,
                                            use_smooth_softmax=use_smooth_softmax,
                                            has_head_sink=has_head_sink,
                                        )
                                        name = f"b{b}_sq{sq}_skv{skv}_nh{n}_{n2}_h{h}_w{lws}_rot{rotary}{rotary_interleaved}_pkd{packed}_sc{softcap}_sm{use_smooth_softmax}_{has_head_sink}"
                                        yield name, config


def gqa_cuda_past_test_cases(allow_head_sink: bool = True):
    batches = [5] if pipeline_mode else [1, 3, 5]
    # s: new sequence length, s2: past sequence length
    seqs = [(1, 1024)] if pipeline_mode else [(1, 128), (1, 1024), (1, 2048), (1, 5000)]
    num_h = [(32, 8)] if pipeline_mode else [(6, 3), (9, 9), (32, 8)]
    h_sizes = [256] if pipeline_mode else [64, 128, 256]
    smmoth_softmax__head_sink = get_softmax_options(allow_head_sink)

    for b in batches:
        for s, s2 in seqs:
            for n, n2 in num_h:
                for h in h_sizes:
                    for lws in [-1, random.randint(1, s2)]:
                        for rotary, rotary_interleaved in get_cuda_rotary_options():
                            for packed in [False, True]:
                                for softcap in [0.0, 50.0]:
                                    if rotary and h % 16 > 0:
                                        continue
                                    for use_smooth_softmax, has_head_sink in smmoth_softmax__head_sink:
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
                                            softcap=softcap,
                                            use_smooth_softmax=use_smooth_softmax,
                                            has_head_sink=has_head_sink,
                                        )
                                        name = f"b{b}_s{s}_{s2}_nh{n}_{n2}_h{h}_w{lws}_rot{rotary}{rotary_interleaved}_pkd{packed}_sc{softcap}_sm{use_smooth_softmax}_{has_head_sink}"
                                        yield name, config


# #################################################################################################
#  Unit Test Classes
# #################################################################################################


def has_cuda_provider():
    return "CUDAExecutionProvider" in get_available_providers()


def has_flash_attention():
    if not has_cuda_provider() or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def has_memory_efficient():
    if not has_cuda_provider() or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 5


@unittest.skipIf(not has_flash_attention(), "Flash Attention is not available, skipping tests.")
class TestFlashGQA(unittest.TestCase):
    @parameterized.expand(gqa_cuda_prompt_test_cases())
    def test_gqa_prompt_flash_attention(self, name, config):
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "0"
        parity_check_gqa_prompt(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            numpy_type=numpy.float16,
            ort_type=TensorProto.FLOAT16,
            causal=True,
            rtol=5e-3,
            atol=5e-3,
        )

    @parameterized.expand(gqa_cuda_past_test_cases())
    def test_gqa_past_flash_attention(self, name, config):
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "0"
        parity_check_gqa_past(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            numpy_type=numpy.float16,
            ort_type=TensorProto.FLOAT16,
            causal=True,
            rtol=5e-3,
            atol=5e-3,
        )


@unittest.skipIf(not has_memory_efficient(), "Memory Efficient Attention is not available, skipping tests.")
class TestMemoryEfficientGQA(unittest.TestCase):
    @parameterized.expand(gqa_cuda_prompt_test_cases(allow_head_sink=False))
    def test_gqa_prompt_memory_efficient(self, name, config):
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "1"
        parity_check_gqa_prompt(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            numpy_type=numpy.float16,
            ort_type=TensorProto.FLOAT16,
            causal=True,
            rtol=5e-3,
            atol=5e-3,
        )

    @parameterized.expand(gqa_cuda_past_test_cases(allow_head_sink=False))
    def test_gqa_past_memory_efficient(self, name, config):
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "1"
        parity_check_gqa_past(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            numpy_type=numpy.float16,
            ort_type=TensorProto.FLOAT16,
            causal=True,
            rtol=5e-3,
            atol=5e-3,
        )


if __name__ == "__main__":
    unittest.main()
