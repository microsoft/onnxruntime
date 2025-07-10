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
import torch.nn.functional as F
from einops import rearrange, repeat
from onnx import TensorProto, helper
from parameterized import parameterized

from onnxruntime import InferenceSession, OrtValue, SessionOptions, get_available_providers

# Set seed for reproducibility
torch.manual_seed(0)
random.seed(69)

# Reduces number of tests to run for faster pipeline checks
pipeline_mode = os.getenv("PIPELINE_MODE", "1") == "1"

# Terminal colors for printing results
RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"


# #################################################################################################
#  BERT Padding Utilities (from https://github.com/Dao-AILab/flash-attention/blob/2286d7cea7ca8264165c16b2442b6436c43140de/flash_attn/bert_padding.py)
# #################################################################################################


def index_first_axis(input, indices):
    assert input.ndim >= 2
    _first_axis_dim, other_shape = input.shape[0], input.shape[1:]
    second_dim = other_shape.numel()
    return torch.gather(rearrange(input, "b ... -> b (...)"), 0, repeat(indices, "z -> z d", d=second_dim)).reshape(
        -1, *other_shape
    )


def index_put_first_axis(values, indices, first_axis_dim):
    assert indices.ndim == 1
    assert values.ndim >= 2
    output = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype)
    output[indices] = values
    return output


def unpad_input(hidden_states, attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def pad_input(hidden_states, indices, batch, seqlen):
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


# #################################################################################################
#  Configuration and Helper Classes
# #################################################################################################


class Formats:
    BSNH = 0
    BNSH = 1


@dataclass
class Config:
    batch_size: int = 0
    sequence_length: int = 0
    kv_sequence_length: int = 0
    num_heads: int = 0
    kv_num_heads: int = 0
    head_size: int = 0
    has_position_ids: bool = False
    has_attention_bias: bool = False
    has_head_sink: bool = False


@dataclass
class PromptConfig:
    batch_size: int = 0
    q_sequence_length: int = 0
    kv_sequence_length: int = 0
    buffer_sequence_length: int = 0
    num_heads: int = 0
    kv_num_heads: int = 0
    head_size: int = 0
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
    from rotary_flash import apply_rotary_emb

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
            use_cuda_triton = False

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
    config,
    ort_type,
    past_kv_format=Formats.BSNH,
    share_buffer=True,
    local_window_size=-1,
    rotary=False,
    rotary_interleaved=False,
    packed=False,
    softcap=0.0,
    use_smooth_softmax=False,
):
    past_kv_seqlen = config.buffer_sequence_length if share_buffer else 0
    present_kv_seqlen = config.buffer_sequence_length if share_buffer else config.kv_sequence_length
    nodes = [
        helper.make_node(
            "GroupQueryAttention",
            [
                "query",
                "key" if not packed else "",
                "value" if not packed else "",
                "past_key" if share_buffer else "",
                "past_value" if share_buffer else "",
                "seqlens_k",
                "total_sequence_length",
                "cos_cache" if rotary else "",
                "sin_cache" if rotary else "",
                "position_ids" if config.has_position_ids else "",
                "attention_bias" if config.has_attention_bias else "",
                "head_sink" if config.has_head_sink else "",
            ],
            ["output", "present_key", "present_value"],
            "GroupQueryAttention_0",
            num_heads=config.num_heads,
            kv_num_heads=config.kv_num_heads,
            local_window_size=local_window_size,
            do_rotary=rotary,
            rotary_interleaved=rotary_interleaved,
            softcap=softcap,
            smooth_softmax=1 if use_smooth_softmax else 0,
            domain="com.microsoft",
        ),
    ]

    q_hidden_size = (
        (config.num_heads * config.head_size)
        if not packed
        else (config.num_heads * config.head_size + 2 * config.kv_num_heads * config.head_size)
    )
    graph_input = [
        helper.make_tensor_value_info("query", ort_type, [config.batch_size, config.q_sequence_length, q_hidden_size]),
        helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [config.batch_size]),
        helper.make_tensor_value_info("total_sequence_length", TensorProto.INT32, [1]),
    ]

    if not packed:
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
        k_shape = [
            config.batch_size,
            past_kv_seqlen if past_kv_format == Formats.BSNH else config.kv_num_heads,
            config.kv_num_heads if past_kv_format == Formats.BSNH else past_kv_seqlen,
            config.head_size,
        ]
        v_shape = k_shape
        graph_input.extend(
            [
                helper.make_tensor_value_info("past_key", ort_type, k_shape),
                helper.make_tensor_value_info("past_value", ort_type, v_shape),
            ]
        )
    if rotary:
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

    output_k_shape = [
        config.batch_size,
        present_kv_seqlen if past_kv_format == Formats.BSNH else config.kv_num_heads,
        config.kv_num_heads if past_kv_format == Formats.BSNH else present_kv_seqlen,
        config.head_size,
    ]
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
    config,
    ort_type,
    past_kv_format=Formats.BSNH,
    share_buffer=True,
    local_window_size=-1,
    rotary=False,
    rotary_interleaved=False,
    packed=False,
    softcap=0.0,
    use_smooth_softmax=False,
):
    past_kv_seqlen = config.kv_sequence_length
    present_kv_seqlen = (
        config.kv_sequence_length if share_buffer else config.kv_sequence_length + config.sequence_length
    )
    nodes = [
        helper.make_node(
            "GroupQueryAttention",
            [
                "query",
                "key" if not packed else "",
                "value" if not packed else "",
                "past_key",
                "past_value",
                "seqlens_k",
                "total_sequence_length",
                "cos_cache" if rotary else "",
                "sin_cache" if rotary else "",
                "position_ids" if config.has_position_ids else "",
                "attention_bias" if config.has_attention_bias else "",
                "head_sink" if config.has_head_sink else "",
            ],
            ["output", "present_key", "present_value"],
            "GroupQueryAttention_0",
            num_heads=config.num_heads,
            kv_num_heads=config.kv_num_heads,
            local_window_size=local_window_size,
            do_rotary=rotary,
            rotary_interleaved=rotary_interleaved,
            softcap=softcap,
            smooth_softmax=1 if use_smooth_softmax else 0,
            domain="com.microsoft",
        ),
    ]

    q_hidden_size = (
        (config.num_heads * config.head_size)
        if not packed
        else (config.num_heads * config.head_size + 2 * config.kv_num_heads * config.head_size)
    )
    past_k_shape = [
        config.batch_size,
        past_kv_seqlen if past_kv_format == Formats.BSNH else config.kv_num_heads,
        config.kv_num_heads if past_kv_format == Formats.BSNH else past_kv_seqlen,
        config.head_size,
    ]
    graph_input = [
        helper.make_tensor_value_info("query", ort_type, [config.batch_size, config.sequence_length, q_hidden_size]),
        helper.make_tensor_value_info("past_key", ort_type, past_k_shape),
        helper.make_tensor_value_info("past_value", ort_type, past_k_shape),
        helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [config.batch_size]),
        helper.make_tensor_value_info("total_sequence_length", TensorProto.INT32, [1]),
    ]

    if not packed:
        graph_input.extend(
            [
                helper.make_tensor_value_info(
                    "key", ort_type, [config.batch_size, config.sequence_length, config.kv_num_heads * config.head_size]
                ),
                helper.make_tensor_value_info(
                    "value",
                    ort_type,
                    [config.batch_size, config.sequence_length, config.kv_num_heads * config.head_size],
                ),
            ]
        )

    if rotary:
        rotary_dim = (math.floor(config.head_size / 16) * 16) // 2
        cache_len = config.kv_sequence_length + (0 if share_buffer else config.sequence_length)
        graph_input.extend(
            [
                helper.make_tensor_value_info("cos_cache", ort_type, [cache_len, rotary_dim]),
                helper.make_tensor_value_info("sin_cache", ort_type, [cache_len, rotary_dim]),
            ]
        )

    if config.has_position_ids:
        graph_input.append(
            helper.make_tensor_value_info(
                "position_ids", TensorProto.INT64, [config.batch_size, config.sequence_length]
            )
        )
    if config.has_attention_bias:
        graph_input.append(
            helper.make_tensor_value_info(
                "attention_bias", ort_type, [config.batch_size, 1, config.sequence_length, present_kv_seqlen]
            )
        )
    if config.has_head_sink:
        graph_input.append(helper.make_tensor_value_info("head_sink", ort_type, [config.num_heads]))

    output_k_shape = [
        config.batch_size,
        present_kv_seqlen if past_kv_format == Formats.BSNH else config.kv_num_heads,
        config.kv_num_heads if past_kv_format == Formats.BSNH else present_kv_seqlen,
        config.head_size,
    ]

    graph_output = [
        helper.make_tensor_value_info(
            "output", ort_type, [config.batch_size, config.sequence_length, config.num_heads * config.head_size]
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
    config,
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
    window_size=-1,
    past_kv_format=Formats.BSNH,
    share_buffer=True,
    rotary_interleaved=False,
    softcap=0.0,
    use_smooth_softmax=False,
    ort_type=TensorProto.FLOAT16,
    numpy_type=numpy.float16,
):
    onnx_model_str = create_group_query_attention_graph_prompt(
        config,
        ort_type,
        past_kv_format,
        share_buffer,
        local_window_size=window_size,
        rotary=cos is not None,
        rotary_interleaved=rotary_interleaved,
        packed=new_k is None,
        softcap=softcap,
        use_smooth_softmax=use_smooth_softmax,
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
    config,
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
    past_kv_format=Formats.BSNH,
    share_buffer=True,
    window_size=-1,
    rotary_interleaved=False,
    softcap=0.0,
    use_smooth_softmax=False,
    ort_type=TensorProto.FLOAT16,
    numpy_type=numpy.float16,
):
    onnx_model_str = create_group_query_attention_graph_past(
        config,
        ort_type,
        past_kv_format,
        share_buffer,
        local_window_size=window_size,
        rotary=cos is not None,
        rotary_interleaved=rotary_interleaved,
        packed=new_k is None,
        softcap=softcap,
        use_smooth_softmax=use_smooth_softmax,
    )

    q = torch.reshape(q, (config.batch_size, config.sequence_length, -1))
    if new_k is not None:
        new_k = torch.reshape(new_k, (config.batch_size, config.sequence_length, -1))
        new_v = torch.reshape(new_v, (config.batch_size, config.sequence_length, -1))

    sess_options = SessionOptions()
    ort_session = InferenceSession(onnx_model_str, sess_options, providers=[ep])
    io_binding = ort_session.io_binding()

    # Common inputs
    total_seq_len = config.kv_sequence_length if share_buffer else config.kv_sequence_length + config.sequence_length
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
            col_idx < row_idx + sk - sq - window_size[0],
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

    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q, seqlen_k, window_size, query_padding_mask, key_padding_mask, q.device
        )
        scores.masked_fill_(local_mask, float("-inf"))

    # Add custom attention bias if provided (for CPU tests)
    if attention_bias is not None:
        scores += attention_bias

    if use_smooth_softmax or (head_sink is not None):
        attention = smooth_softmax_ref(scores, head_sink)
    else:
        attention = torch.softmax(scores, dim=-1)

    # Fill NaNs with 0
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)

    output = torch.einsum("bhts,bshd->bthd", attention, v)

    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)

    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


# #################################################################################################
#  Parity Check (Core Test Logic)
# #################################################################################################


def parity_check_gqa_prompt(
    config,
    ep,
    device,
    torch_type,
    numpy_type,
    ort_type,
    causal,
    local,
    past_format,
    rotary,
    rotary_interleaved,
    packed,
    softcap,
    use_smooth_softmax,
    rtol,
    atol,
):
    # --- Test Data Generation ---
    q = torch.randn(
        config.batch_size, config.q_sequence_length, config.num_heads, config.head_size, device=device, dtype=torch_type
    )
    k = torch.randn(
        config.batch_size,
        config.buffer_sequence_length if past_format == Formats.BSNH else config.kv_num_heads,
        config.kv_num_heads if past_format == Formats.BSNH else config.buffer_sequence_length,
        config.head_size,
        device=device,
        dtype=torch_type,
    )
    v = torch.randn_like(k)
    new_k = torch.randn(
        config.batch_size,
        config.kv_sequence_length,
        config.kv_num_heads,
        config.head_size,
        device=device,
        dtype=torch_type,
    )
    new_v = torch.randn_like(new_k)

    head_sink = torch.rand(config.num_heads, dtype=torch_type, device=device) if config.has_head_sink else None

    window_size = (-1, -1)
    left_window_size = -1
    if local:
        left_window_size = random.randint(1, config.kv_sequence_length)
        window_size = (left_window_size, 0)
    elif causal:
        window_size = (-1, 0)

    # --- PyTorch Reference Path ---
    k_cache_ref = k.clone()
    v_cache_ref = v.clone()
    if past_format == Formats.BNSH:
        k_cache_ref = k_cache_ref.transpose(1, 2)
        v_cache_ref = v_cache_ref.transpose(1, 2)

    cache_seqlens = torch.full((config.batch_size,), config.kv_sequence_length, device=device, dtype=torch.int32)
    rotary_seqlens = torch.zeros(config.batch_size, device=device, dtype=torch.long)

    cos, sin, q_ro, k_ro = None, None, q, new_k
    if rotary:
        rotary_dim = math.floor(config.head_size / 16) * 16
        angle = torch.rand(config.buffer_sequence_length, rotary_dim // 2, device=device) * 2 * math.pi
        cos = torch.cos(angle).to(dtype=torch_type)
        sin = torch.sin(angle).to(dtype=torch_type)
        q_ro = apply_rotary_embedding(q.clone(), cos, sin, rotary_seqlens, rotary_interleaved, device)
        k_ro = apply_rotary_embedding(new_k.clone(), cos, sin, rotary_seqlens, rotary_interleaved, device)

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
        q_ro,
        k_cache_ref,
        v_cache_ref,
        None,
        key_padding_mask,
        attention_bias,
        causal=True,
        window_size=window_size,
        softcap=softcap,
        use_smooth_softmax=use_smooth_softmax,
        head_sink=head_sink,
    )
    out_ref_np = out_ref.detach().cpu().numpy()

    if past_format == Formats.BNSH:
        k_cache_ref = k_cache_ref.transpose(1, 2)
        v_cache_ref = v_cache_ref.transpose(1, 2)
    k_cache_ref_np = k_cache_ref.detach().cpu().numpy()
    v_cache_ref_np = v_cache_ref.detach().cpu().numpy()

    # --- ONNX Runtime Path ---
    q_ort, k_ort, v_ort, new_k_ort, new_v_ort = q, k, v, new_k, new_v
    if packed:
        q_ort = torch.cat([q, new_k, new_v], dim=2)
        new_k_ort, new_v_ort = None, None

    # seqlens_k for GQA op is past_seq_len + seq_len - 1
    ort_seqlens = cache_seqlens - 1
    out, present_k, present_v = gqa_prompt_func(
        q_ort,
        k_ort,
        v_ort,
        config,
        new_k_ort,
        new_v_ort,
        cos,
        sin,
        ort_seqlens,
        position_ids,
        attention_bias,
        head_sink,
        ep,
        device,
        left_window_size,
        past_format,
        True,
        rotary_interleaved,
        softcap,
        use_smooth_softmax,
        ort_type,
        numpy_type,
    )
    out = torch.reshape(out, (config.batch_size, config.q_sequence_length, config.num_heads, config.head_size))
    out_np = out.detach().cpu().numpy()

    # --- Comparison ---
    numpy.testing.assert_allclose(present_k, k_cache_ref_np, rtol=rtol, atol=atol)
    numpy.testing.assert_allclose(present_v, v_cache_ref_np, rtol=rtol, atol=atol)
    numpy.testing.assert_allclose(out_np, out_ref_np, rtol=rtol, atol=atol)


def parity_check_gqa_past(
    config,
    ep,
    device,
    torch_type,
    numpy_type,
    ort_type,
    causal,
    local,
    past_format,
    rotary,
    rotary_interleaved,
    packed,
    softcap,
    use_smooth_softmax,
    rtol,
    atol,
):
    # --- Test Data Generation ---
    q = torch.randn(
        config.batch_size, config.sequence_length, config.num_heads, config.head_size, device=device, dtype=torch_type
    )
    k = torch.randn(
        config.batch_size,
        config.kv_sequence_length if past_format == Formats.BSNH else config.kv_num_heads,
        config.kv_num_heads if past_format == Formats.BSNH else config.kv_sequence_length,
        config.head_size,
        device=device,
        dtype=torch_type,
    )
    v = torch.randn_like(k)
    new_k = torch.randn(
        config.batch_size,
        config.sequence_length,
        config.kv_num_heads,
        config.head_size,
        device=device,
        dtype=torch_type,
    )
    new_v = torch.randn_like(new_k)

    head_sink = torch.rand(config.num_heads, dtype=torch_type, device=device) if config.has_head_sink else None
    window_size = (-1, -1)
    left_window_size = -1
    if local:
        left_window_size = random.randint(1, config.kv_sequence_length)
        window_size = (left_window_size, 0)
    elif causal:
        window_size = (-1, 0)

    # --- PyTorch Reference Path ---
    k_cache_ref = k.clone()
    v_cache_ref = v.clone()
    if past_format == Formats.BNSH:
        k_cache_ref = k_cache_ref.transpose(1, 2)
        v_cache_ref = v_cache_ref.transpose(1, 2)

    cache_seqlens = torch.randint(
        0, config.kv_sequence_length - config.sequence_length + 1, (config.batch_size,), device=device, dtype=torch.long
    )

    cos, sin, q_ro, k_ro = None, None, q, new_k
    if rotary:
        rotary_dim = math.floor(config.head_size / 16) * 16
        angle = torch.rand(config.kv_sequence_length, rotary_dim // 2, device=device) * 2 * math.pi
        cos = torch.cos(angle).to(dtype=torch_type)
        sin = torch.sin(angle).to(dtype=torch_type)
        q_ro = apply_rotary_embedding(q.clone(), cos, sin, cache_seqlens, rotary_interleaved, device)
        k_ro = apply_rotary_embedding(new_k.clone(), cos, sin, cache_seqlens, rotary_interleaved, device)

    position_ids = None
    attention_bias = None
    total_seq_len = config.kv_sequence_length
    if ep == "CPUExecutionProvider":
        if config.has_position_ids:
            position_ids = (cache_seqlens.unsqueeze(1) + torch.arange(config.sequence_length, device=device)).long()
        if config.has_attention_bias:
            attention_bias = torch.zeros(
                config.batch_size, 1, config.sequence_length, total_seq_len, device=device, dtype=torch_type
            )
            for b in range(config.batch_size):
                end_pos = cache_seqlens[b] + config.sequence_length
                attention_bias[b, :, :, end_pos:] = float("-inf")

    arange = rearrange(torch.arange(config.kv_sequence_length, device=device), "s -> 1 s")
    cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
    update_mask = torch.logical_and(
        cache_seqlens_expanded <= arange, arange < cache_seqlens_expanded + config.sequence_length
    )
    k_cache_ref[update_mask] = rearrange(k_ro, "b s ... -> (b s) ...").to(dtype=torch_type)
    v_cache_ref[update_mask] = rearrange(new_v, "b s ... -> (b s) ...").to(dtype=torch_type)
    key_padding_mask = arange < cache_seqlens_expanded + config.sequence_length

    out_ref, _ = attention_ref(
        q_ro,
        k_cache_ref,
        v_cache_ref,
        None,
        key_padding_mask,
        attention_bias,
        causal=True,
        window_size=window_size,
        softcap=softcap,
        use_smooth_softmax=use_smooth_softmax,
        head_sink=head_sink,
    )
    out_ref_np = out_ref.detach().cpu().numpy()
    if past_format == Formats.BNSH:
        k_cache_ref = k_cache_ref.transpose(1, 2)
        v_cache_ref = v_cache_ref.transpose(1, 2)
    k_cache_ref_np = k_cache_ref.detach().cpu().numpy()
    v_cache_ref_np = v_cache_ref.detach().cpu().numpy()

    # --- ONNX Runtime Path ---
    q_ort, k_ort, v_ort, new_k_ort, new_v_ort = q, k, v, new_k, new_v
    if packed:
        q_ort = torch.cat([q, new_k, new_v], dim=2)
        new_k_ort, new_v_ort = None, None

    ort_seqlens = cache_seqlens + config.sequence_length - 1
    out, present_k, present_v = gqa_past_func(
        q_ort,
        k_ort,
        v_ort,
        config,
        new_k_ort,
        new_v_ort,
        cos,
        sin,
        ort_seqlens.int(),
        position_ids,
        attention_bias,
        head_sink,
        ep,
        device,
        past_format,
        True,
        left_window_size,
        rotary_interleaved,
        softcap,
        use_smooth_softmax,
        ort_type,
        numpy_type,
    )
    out = torch.reshape(out, (config.batch_size, config.sequence_length, config.num_heads, config.head_size))
    out_np = out.detach().cpu().numpy()

    # --- Comparison ---
    numpy.testing.assert_allclose(present_k, k_cache_ref_np, rtol=rtol, atol=atol)
    numpy.testing.assert_allclose(present_v, v_cache_ref_np, rtol=rtol, atol=atol)
    numpy.testing.assert_allclose(out_np, out_ref_np, rtol=rtol, atol=atol)


# #################################################################################################
#  Test Case Generators
# #################################################################################################


def get_cuda_rotary_options():
    return [(False, False)] if platform.system() != "Linux" else [(True, False), (True, True), (False, False)]


def get_cpu_rotary_options():
    return [(False, False), (True, False), (True, True)]


def gqa_cuda_prompt_test_cases():
    batches = [3] if pipeline_mode else [1, 3, 5]
    seqs = [(127, 127), (240, 240)] if pipeline_mode else [(35, 35), (127, 127), (240, 240), (2000, 2000)]
    num_h = [(32, 8)] if pipeline_mode else [(6, 3), (9, 9), (32, 8)]
    h_sizes = [128] if pipeline_mode else [64, 128, 256]
    smmoth_softmax__head_sink = (
        [(False, False), (False, True)] if pipeline_mode else [(False, False), (False, True), (True, False)]
    )

    for b in batches:
        for sq, skv in seqs:
            for n, n2 in num_h:
                for h in h_sizes:
                    for local in [False, True]:
                        for rotary, rotary_interleaved in get_cuda_rotary_options():
                            for packed in [False, True]:
                                for softcap in [0.0, 50.0]:
                                    if rotary and h % 16 > 0:
                                        continue
                                    for use_smooth_softmax, has_sink in smmoth_softmax__head_sink:
                                        config = PromptConfig(b, sq, skv, sq + skv + 8, n, n2, h)
                                        options = (
                                            f"{softcap=}_{rotary=}_{rotary_interleaved=}_"
                                            f"{local=}_{packed=}_{use_smooth_softmax=}_{has_sink=}"
                                        )
                                        name = f"b{b}_sq{sq}_skv{skv}_nh{n}k{n2}_h{h}_{options}"
                                        yield (
                                            name,
                                            config,
                                            local,
                                            rotary,
                                            rotary_interleaved,
                                            packed,
                                            softcap,
                                        )


def gqa_cuda_past_test_cases():
    batches = [5] if pipeline_mode else [1, 3, 5]
    seqs = [(1, 1024)] if pipeline_mode else [(1, 128), (1, 1024), (1, 2048), (1, 5000)]
    num_h = [(32, 8)] if pipeline_mode else [(6, 3), (9, 9), (32, 8)]
    h_sizes = [256] if pipeline_mode else [64, 128, 256]
    smmoth_softmax__head_sink = (
        [(False, False), (False, True)] if pipeline_mode else [(False, False), (False, True), (True, False)]
    )

    for b in batches:
        for s, s2 in seqs:
            for n, n2 in num_h:
                for h in h_sizes:
                    for local in [False, True]:
                        for rotary, rotary_interleaved in get_cuda_rotary_options():
                            for packed in [False, True]:
                                for softcap in [0.0, 50.0]:
                                    if rotary and h % 16 > 0:
                                        continue
                                    for use_smooth_softmax, has_sink in smmoth_softmax__head_sink:
                                        config = Config(b, s, s2, n, n2, h)
                                        options = (
                                            f"{softcap=}_{rotary=}_{rotary_interleaved=}_"
                                            f"{local=}_{packed=}_{use_smooth_softmax=}_{has_sink=}"
                                        )
                                        name = f"b{b}_s{s}_s2{s2}_nh{n}k{n2}_h{h}_{options}"
                                        yield name, config, local, rotary, rotary_interleaved, packed, softcap


def gqa_cpu_prompt_test_cases():
    precisions = [
        {"ort": TensorProto.FLOAT, "torch": torch.float32, "numpy": numpy.float32, "rtol": 1e-5, "atol": 1e-5},
        {"ort": TensorProto.FLOAT16, "torch": torch.float16, "numpy": numpy.float16, "rtol": 1e-2, "atol": 1e-2},
    ]
    batches = [3] if pipeline_mode else [1, 3]
    seqs = [(64, 64)] if pipeline_mode else [(35, 35), (127, 127)]
    num_h = [(6, 3)] if pipeline_mode else [(6, 6), (6, 3)]
    h_sizes = [64] if pipeline_mode else [32, 64]
    has_pos_ids__attn_bias = (
        [(False, False), (True, True)]
        if pipeline_mode
        else [(False, False), (True, True), (False, True), (True, False)]
    )
    smmoth_softmax__head_sink = (
        [(False, False), (False, True)] if pipeline_mode else [(False, False), (False, True), (True, False)]
    )

    for p in precisions:
        for b in batches:
            for sq, skv in seqs:
                for n, n2 in num_h:
                    for h in h_sizes:
                        for local in [False, True]:
                            for rotary, rotary_interleaved in get_cpu_rotary_options():
                                for packed in [False, True]:
                                    for softcap in [0.0, 50.0]:
                                        for use_smooth_softmax, has_sink in smmoth_softmax__head_sink:
                                            for has_pos, has_attn_bias in has_pos_ids__attn_bias:
                                                config = PromptConfig(
                                                    b, sq, skv, sq + skv + 8, n, n2, h, has_pos, has_attn_bias, has_sink
                                                )
                                                p_str = "fp32" if p["ort"] == TensorProto.FLOAT else "fp16"
                                                options = (
                                                    f"{softcap=}_{rotary=}_{rotary_interleaved=}_"
                                                    f"{local=}_{packed=}_{use_smooth_softmax=}_"
                                                    f"{has_sink=}_{has_pos=}_{has_attn_bias=}"
                                                )
                                                name = f"{p_str}_b{b}_sq{sq}_{skv}_n{n}_{n2}_h{h}_{options}"
                                                yield (
                                                    name,
                                                    config,
                                                    p,
                                                    local,
                                                    rotary,
                                                    rotary_interleaved,
                                                    packed,
                                                    softcap,
                                                    use_smooth_softmax,
                                                )


# #################################################################################################
#  unittest Test Classes
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
    def run_test(self, config, local, rotary, rotary_interleaved, packed, softcap, use_smooth_softmax=False):
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "0"
        parity_check_gqa_prompt(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            numpy_type=numpy.float16,
            ort_type=TensorProto.FLOAT16,
            causal=True,
            local=local,
            past_format=Formats.BNSH,
            rotary=rotary,
            rotary_interleaved=rotary_interleaved,
            packed=packed,
            softcap=softcap,
            use_smooth_softmax=use_smooth_softmax,
            rtol=5e-3,
            atol=5e-3,
        )

    @parameterized.expand(gqa_cuda_prompt_test_cases())
    def test_gqa_prompt_flash_attention(self, name, config, local, rotary, rotary_interleaved, packed, softcap):
        self.run_test(config, local, rotary, rotary_interleaved, packed, softcap, use_smooth_softmax=False)
        self.run_test(config, local, rotary, rotary_interleaved, packed, softcap, use_smooth_softmax=True)

    @parameterized.expand(gqa_cuda_past_test_cases())
    def test_gqa_past_flash_attention(self, name, config, local, rotary, rotary_interleaved, packed, softcap):
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "0"
        parity_check_gqa_past(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            numpy_type=numpy.float16,
            ort_type=TensorProto.FLOAT16,
            causal=True,
            local=local,
            past_format=Formats.BNSH,
            rotary=rotary,
            rotary_interleaved=rotary_interleaved,
            packed=packed,
            softcap=softcap,
            use_smooth_softmax=False,
            rtol=5e-3,
            atol=5e-3,
        )


@unittest.skipIf(not has_memory_efficient(), "Memory Efficient Attention is not available, skipping tests.")
class TestMemoryEfficientGQA(unittest.TestCase):
    def run_test(self, config, local, rotary, rotary_interleaved, packed, softcap, use_smooth_softmax=False):
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "1"
        parity_check_gqa_prompt(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            numpy_type=numpy.float16,
            ort_type=TensorProto.FLOAT16,
            causal=True,
            local=local,
            past_format=Formats.BNSH,
            rotary=rotary,
            rotary_interleaved=rotary_interleaved,
            packed=packed,
            softcap=softcap,
            use_smooth_softmax=use_smooth_softmax,
            rtol=5e-3,
            atol=5e-3,
        )

    @parameterized.expand(gqa_cuda_prompt_test_cases())
    def test_gqa_prompt_memory_efficient(self, name, config, local, rotary, rotary_interleaved, packed, softcap):
        self.run_test(config, local, rotary, rotary_interleaved, packed, softcap, use_smooth_softmax=False)
        self.run_test(config, local, rotary, rotary_interleaved, packed, softcap, use_smooth_softmax=True)

    @parameterized.expand(gqa_cuda_past_test_cases())
    def test_gqa_past_memory_efficient(self, name, config, local, rotary, rotary_interleaved, packed, softcap):
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "1"
        for use_smooth_softmax in [True, False]:
            parity_check_gqa_past(
                config=config,
                ep="CUDAExecutionProvider",
                device="cuda",
                torch_type=torch.float16,
                numpy_type=numpy.float16,
                ort_type=TensorProto.FLOAT16,
                causal=True,
                local=local,
                past_format=Formats.BNSH,
                rotary=rotary,
                rotary_interleaved=rotary_interleaved,
                packed=packed,
                softcap=softcap,
                use_smooth_softmax=use_smooth_softmax,
                rtol=5e-3,
                atol=5e-3,
            )


class TestGQACPU(unittest.TestCase):
    @parameterized.expand(gqa_cpu_prompt_test_cases())
    def test_gqa_prompt_cpu(
        self, name, config, precision, local, rotary, rotary_interleaved, packed, softcap, use_smooth_softmax
    ):
        parity_check_gqa_prompt(
            config=config,
            ep="CPUExecutionProvider",
            device="cpu",
            torch_type=precision["torch"],
            numpy_type=precision["numpy"],
            ort_type=precision["ort"],
            causal=True,
            local=local,
            past_format=Formats.BNSH,
            rotary=rotary,
            rotary_interleaved=rotary_interleaved,
            packed=packed,
            softcap=softcap,
            use_smooth_softmax=use_smooth_softmax,
            rtol=precision["rtol"],
            atol=precision["atol"],
        )


if __name__ == "__main__":
    unittest.main()
