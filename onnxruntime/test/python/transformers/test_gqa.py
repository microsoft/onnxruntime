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
from copy import deepcopy
from dataclasses import dataclass

import numpy
import torch
from einops import rearrange, repeat
from onnx import TensorProto, helper
from packaging import version
from parameterized import parameterized

import onnxruntime
from onnxruntime import InferenceSession, OrtValue, SessionOptions, get_available_providers

# Set seed for reproducibility
torch.manual_seed(0)
random.seed(69)

# Reduces number of tests to run for faster pipeline checks
pipeline_mode = os.getenv("PIPELINE_MODE", "1") == "1"

# Number of values per parameter (compared to pipeline mode)
param_count = int(os.getenv("PARAM_COUNT", "2")) if not pipeline_mode else 1

# #################################################################################################
#  Configuration and Helper Classes
# #################################################################################################


# --- ONNX and Torch/Numpy Dtype Mappings ---
ONNX_TENSOR_TYPE_MAP = {
    "float32": TensorProto.FLOAT,
    "float16": TensorProto.FLOAT16,
    "bfloat16": TensorProto.BFLOAT16,
    "int32": TensorProto.INT32,
    "int8": TensorProto.INT8,
    "int4": TensorProto.UINT8,
}

TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int4": torch.uint8,
}

NUMPY_DTYPE_MAP = {
    "float32": numpy.float32,
    "float16": numpy.float16,
    "bfloat16": numpy.uint16,
    "int8": numpy.int8,
    "int4": numpy.uint8,
}


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
    # Quantization parameters
    k_quant_type: str = "NONE"
    v_quant_type: str = "NONE"
    kv_cache_type: str = "float16"
    kv_cache_bit_width: int = 0


# #################################################################################################
#  Quantization Helpers
# #################################################################################################


def get_q_range(q_type_str):
    if q_type_str == "int8":
        return -128, 127
    if q_type_str == "int4":
        return -8, 7
    raise ValueError(f"Unsupported quantization type for range: {q_type_str}")


def pack_int4(tensor_int8):
    assert tensor_int8.shape[-1] % 2 == 0
    t_low = tensor_int8[..., 0::2] + 8
    t_high = tensor_int8[..., 1::2] + 8
    packed = (t_low & 0x0F) | (t_high << 4)
    return packed.to(torch.uint8)


def unpack_int4(packed_tensor_uint8):
    t_low = (packed_tensor_uint8 & 0x0F) - 8
    t_high = (packed_tensor_uint8 >> 4) - 8
    unpacked = torch.empty(
        (*packed_tensor_uint8.shape[:-1], packed_tensor_uint8.shape[-1] * 2),
        dtype=torch.int8,
        device=packed_tensor_uint8.device,
    )
    unpacked[..., 0::2] = t_low
    unpacked[..., 1::2] = t_high
    return unpacked


def compute_scale(tensor_float, quant_type, q_type_str):
    if quant_type == "NONE":
        return None

    qmin, qmax = get_q_range(q_type_str)

    if quant_type == "PER_TENSOR":
        t_max = torch.max(torch.abs(tensor_float))
        scale = t_max / qmax if t_max > 1e-6 else torch.tensor(1.0, device=tensor_float.device, dtype=torch.float32)
        return scale.unsqueeze(0).to(torch.float32)

    if quant_type == "PER_CHANNEL":
        # Per-channel scale is computed independently for each channel across the batch and sequence length dimensions.
        t_max = torch.max(torch.abs(tensor_float), dim=2, keepdim=True)[0]
        t_max = torch.max(t_max, dim=0, keepdim=True)[0]
        scale = t_max / qmax
        scale[scale < 1e-6] = 1.0
        return scale.to(torch.float32)

    raise ValueError(f"Unsupported quant_type: {quant_type}")


def dequantize_tensor(quantized_tensor, scale, quant_type, q_type_str):
    if quant_type == "NONE":
        return quantized_tensor

    # Ensure scale is on the same device as quantized_tensor
    if isinstance(scale, torch.Tensor):
        scale = scale.to(quantized_tensor.device)

    unpacked_tensor = quantized_tensor
    if q_type_str == "int4":
        unpacked_tensor = unpack_int4(quantized_tensor)

    return unpacked_tensor.to(torch.float32) * scale


def quantize_tensor_with_scale(tensor_float, scale, quant_type, q_type_str):
    """Quantizes a tensor using a provided scale."""
    if quant_type == "NONE":
        return tensor_float

    qmin, qmax = get_q_range(q_type_str)
    quantized = torch.clamp(torch.round(tensor_float / scale), qmin, qmax)

    if q_type_str == "int4":
        quantized = pack_int4(quantized.to(torch.int8))
    else:
        quantized = quantized.to(TORCH_DTYPE_MAP[q_type_str])
    return quantized


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


def create_gqa_node_and_io(
    config: GQAConfig,
    ort_type,
    share_buffer=True,
    is_past=False,
    output_qk: int = 0,  # CUDA does not support output_qk for GQA
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
        "past_key" if is_past or share_buffer else "",
        "past_value" if is_past or share_buffer else "",
        "seqlens_k",
        "total_sequence_length",
        "cos_cache" if config.rotary else "",
        "sin_cache" if config.rotary else "",
        "position_ids" if config.has_position_ids else "",
        "attention_bias" if config.has_attention_bias else "",
        "head_sink" if config.has_head_sink else "",
        "k_scale" if config.k_quant_type != "NONE" and (is_past or share_buffer) else "",
        "v_scale" if config.v_quant_type != "NONE" and (is_past or share_buffer) else "",
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
        smooth_softmax=1 if config.use_smooth_softmax else 0,
        qk_output=output_qk,
        **quantization_attributes,
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

    if is_past or share_buffer:
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
            graph_input.append(helper.make_tensor_value_info("k_scale", ort_type, None))
        if config.v_quant_type != "NONE":
            graph_input.append(helper.make_tensor_value_info("v_scale", ort_type, None))

    if config.rotary:
        rotary_dim = (math.floor(config.head_size / 16) * 16) // 2
        cache_seq_len = config.buffer_sequence_length
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
        bias_len = present_kv_seqlen if is_past else config.kv_sequence_length
        graph_input.append(
            helper.make_tensor_value_info(
                "attention_bias", ort_type, [config.batch_size, 1, config.q_sequence_length, bias_len]
            )
        )
    if config.has_head_sink:
        graph_input.append(helper.make_tensor_value_info("head_sink", ort_type, [config.num_heads]))

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

    return node, graph_input, graph_output


def create_group_query_attention_graph_prompt(config: GQAConfig, ort_type, share_buffer=True):
    node, graph_input, graph_output = create_gqa_node_and_io(config, ort_type, share_buffer, is_past=False)
    graph = helper.make_graph([node], "GroupQueryAttention_Graph", graph_input, graph_output)
    model = helper.make_model(graph)
    return model.SerializeToString()


def create_group_query_attention_graph_past(config: GQAConfig, ort_type, share_buffer=True):
    node, graph_input, graph_output = create_gqa_node_and_io(config, ort_type, share_buffer, is_past=True)
    graph = helper.make_graph([node], "GroupQueryAttention_Graph", graph_input, graph_output)
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
    k_scale,
    v_scale,
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
    seqlens_k_ort = OrtValue.ortvalue_from_numpy(seqlens_k.detach().cpu().numpy().astype(numpy.int32), device, 0)
    io_binding.bind_ortvalue_input("seqlens_k", seqlens_k_ort)
    ort_inputs = {
        "query": q.detach().cpu().numpy(),
        "total_sequence_length": torch.tensor([config.q_sequence_length], dtype=torch.int32).detach().cpu().numpy(),
    }
    io_binding.bind_cpu_input("query", ort_inputs["query"])
    io_binding.bind_cpu_input("total_sequence_length", ort_inputs["total_sequence_length"])

    if new_k is not None:
        io_binding.bind_cpu_input("key", new_k.detach().cpu().numpy())
        io_binding.bind_cpu_input("value", new_v.detach().cpu().numpy())
    if cos is not None:
        io_binding.bind_cpu_input("cos_cache", cos.detach().cpu().numpy())
        io_binding.bind_cpu_input("sin_cache", sin.detach().cpu().numpy())

    # CPU-specific inputs
    if config.has_position_ids:
        io_binding.bind_cpu_input("position_ids", position_ids.detach().cpu().numpy())
    if config.has_attention_bias:
        io_binding.bind_cpu_input("attention_bias", attention_bias.detach().cpu().numpy())
    if config.has_head_sink:
        io_binding.bind_cpu_input("head_sink", head_sink.detach().cpu().numpy())

    # Quantization inputs
    if k_scale is not None:
        io_binding.bind_cpu_input("k_scale", k_scale.detach().cpu().numpy().astype(numpy_type))
    if v_scale is not None:
        io_binding.bind_cpu_input("v_scale", v_scale.detach().cpu().numpy().astype(numpy_type))

    # Outputs
    io_binding.bind_output("output")

    if share_buffer:
        cache_numpy_type = NUMPY_DTYPE_MAP[config.kv_cache_type]
        past_k_ort = OrtValue.ortvalue_from_numpy(k.detach().cpu().numpy(), device, 0)
        past_v_ort = OrtValue.ortvalue_from_numpy(v.detach().cpu().numpy(), device, 0)
        io_binding.bind_input("past_key", device, 0, cache_numpy_type, past_k_ort.shape(), past_k_ort.data_ptr())
        io_binding.bind_input("past_value", device, 0, cache_numpy_type, past_v_ort.shape(), past_v_ort.data_ptr())
        io_binding.bind_ortvalue_output("present_key", past_k_ort)
        io_binding.bind_ortvalue_output("present_value", past_v_ort)

    else:
        io_binding.bind_output("present_key")
        io_binding.bind_output("present_value")

    ort_session.run_with_iobinding(io_binding)
    ort_output, present_k, present_v = io_binding.copy_outputs_to_cpu()
    return (
        torch.tensor(ort_output),
        present_k,
        present_v,
    )


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
    total_seq_len = config.past_kv_sequence_length + config.q_sequence_length
    seqlens_k_ort = OrtValue.ortvalue_from_numpy(seqlens_k.detach().cpu().numpy().astype(numpy.int32), device, 0)
    io_binding.bind_ortvalue_input("seqlens_k", seqlens_k_ort)
    ort_inputs = {
        "query": q.detach().cpu().numpy(),
        "total_sequence_length": torch.tensor([total_seq_len], dtype=torch.int32).detach().cpu().numpy(),
    }
    io_binding.bind_cpu_input("query", ort_inputs["query"])
    io_binding.bind_cpu_input("total_sequence_length", ort_inputs["total_sequence_length"])

    if new_k is not None:
        io_binding.bind_cpu_input("key", new_k.detach().cpu().numpy())
        io_binding.bind_cpu_input("value", new_v.detach().cpu().numpy())
    if cos is not None:
        io_binding.bind_cpu_input("cos_cache", cos.detach().cpu().numpy())
        io_binding.bind_cpu_input("sin_cache", sin.detach().cpu().numpy())

    # CPU-specific inputs
    if config.has_position_ids:
        io_binding.bind_cpu_input("position_ids", position_ids.detach().cpu().numpy())
    if config.has_attention_bias:
        io_binding.bind_cpu_input("attention_bias", attention_bias.detach().cpu().numpy())
    if config.has_head_sink:
        io_binding.bind_cpu_input("head_sink", head_sink.detach().cpu().numpy())

    # Quantization inputs
    if k_scale is not None:
        io_binding.bind_cpu_input("k_scale", k_scale.detach().cpu().numpy().astype(numpy_type))
    if v_scale is not None:
        io_binding.bind_cpu_input("v_scale", v_scale.detach().cpu().numpy().astype(numpy_type))

    # Outputs
    io_binding.bind_output("output")

    cache_numpy_type = NUMPY_DTYPE_MAP[config.kv_cache_type]

    if share_buffer:
        past_k_ort = OrtValue.ortvalue_from_numpy(k.detach().cpu().numpy(), device, 0)
        past_v_ort = OrtValue.ortvalue_from_numpy(v.detach().cpu().numpy(), device, 0)
        io_binding.bind_input("past_key", device, 0, cache_numpy_type, past_k_ort.shape(), past_k_ort.data_ptr())
        io_binding.bind_input("past_value", device, 0, cache_numpy_type, past_v_ort.shape(), past_v_ort.data_ptr())
        io_binding.bind_ortvalue_output("present_key", past_k_ort)
        io_binding.bind_ortvalue_output("present_value", past_v_ort)

    else:
        io_binding.bind_cpu_input("past_key", k.detach().cpu().numpy())
        io_binding.bind_cpu_input("past_value", v.detach().cpu().numpy())
        io_binding.bind_output("present_key")
        io_binding.bind_output("present_value")

    ort_session.run_with_iobinding(io_binding)
    ort_output, present_k, present_v = io_binding.copy_outputs_to_cpu()
    return (
        torch.tensor(ort_output),
        present_k,
        present_v,
    )


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

    k_scale = compute_scale(calibration_data_k, config.k_quant_type, config.kv_cache_type)
    v_scale = compute_scale(calibration_data_v, config.v_quant_type, config.kv_cache_type)
    return k_scale, v_scale


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

    # Initialize the KV cache to zeros since no past context in prompt testing.
    k = (
        torch.zeros(
            config.batch_size,
            config.kv_num_heads,
            config.buffer_sequence_length,
            config.head_size,
            device=device,
            dtype=torch_type,
        )
        * std
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

    # --- PyTorch Reference Path ---
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

    position_ids, attention_bias = None, None
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

    # Explicitly cast the source tensor to the destination's dtype before assignment.
    source_k = rearrange(k_ro, "b s ... -> (b s) ...")
    k_cache_ref[update_mask] = source_k.to(k_cache_ref.dtype)

    source_v = rearrange(new_v, "b s ... -> (b s) ...")
    v_cache_ref[update_mask] = source_v.to(v_cache_ref.dtype)

    key_padding_mask = arange < kv_seqlens_expanded

    out_ref, _ = attention_ref(
        q=q_ro,
        k=k_cache_ref,
        v=v_cache_ref,
        key_padding_mask=key_padding_mask,
        attention_bias=attention_bias,
        causal=True,
        window_size=window_size,
        softcap=config.softcap,
        use_smooth_softmax=config.use_smooth_softmax,
        head_sink=head_sink,
    )
    out_ref_np = out_ref.detach().cpu().numpy()

    # --- ONNX Runtime Path ---
    q_ort, new_k_ort, new_v_ort = q, new_k, new_v
    if config.packed:
        q_ort = torch.cat([q, new_k, new_v], dim=2)
        new_k_ort, new_v_ort = None, None

    k_quant = quantize_tensor_with_scale(k, k_scale, config.k_quant_type, config.kv_cache_type)
    v_quant = quantize_tensor_with_scale(v, v_scale, config.v_quant_type, config.kv_cache_type)

    ort_seqlens = cache_seqlens - 1
    out, present_k, present_v = gqa_prompt_func(
        q=q_ort,
        k=k_quant,
        v=v_quant,
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
        share_buffer=True,
        ort_type=ort_type,
        numpy_type=numpy_type,
    )
    out = torch.reshape(out, (config.batch_size, config.q_sequence_length, config.num_heads, config.head_size))
    out_np = out.detach().cpu().numpy()

    # --- Comparison ---
    numpy.testing.assert_allclose(out_np, out_ref_np, rtol=rtol, atol=atol)

    # Compare quantized cache with proper masking per batch
    if config.k_quant_type != "NONE":
        # Convert numpy array to torch tensor with correct dtype
        if config.kv_cache_type == "int4":
            # For int4, present_k is uint8 packed data
            present_k_torch = torch.from_numpy(present_k).to(device)
        elif config.kv_cache_type == "int8":
            # For int8, present_k is int8 data
            present_k_torch = torch.from_numpy(present_k.astype(numpy.int8)).to(device)
        else:
            present_k_torch = torch.from_numpy(present_k).to(device)

        present_k_dequant = (
            dequantize_tensor(present_k_torch, k_scale, config.k_quant_type, config.kv_cache_type).cpu().numpy()
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
            numpy.testing.assert_allclose(
                present_k_dequant[b, :, :valid_len, :], k_cache_ref_dequant[b, :, :valid_len, :], rtol=rtol, atol=atol
            )

    if config.v_quant_type != "NONE":
        # Convert numpy array to torch tensor with correct dtype
        if config.kv_cache_type == "int4":
            present_v_torch = torch.from_numpy(present_v).to(device)
        elif config.kv_cache_type == "int8":
            present_v_torch = torch.from_numpy(present_v.astype(numpy.int8)).to(device)
        else:
            present_v_torch = torch.from_numpy(present_v).to(device)

        present_v_dequant = (
            dequantize_tensor(present_v_torch, v_scale, config.v_quant_type, config.kv_cache_type).cpu().numpy()
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
            numpy.testing.assert_allclose(
                present_v_dequant[b, :, :valid_len, :], v_cache_ref_dequant[b, :, :valid_len, :], rtol=rtol, atol=atol
            )


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
    torch.manual_seed(0)
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

    k_scale, v_scale = get_static_scale(config, device, torch_type, std)
    if k_scale is not None:
        k_scale = k_scale.to(torch_type)
    if v_scale is not None:
        v_scale = v_scale.to(torch_type)

    cache_seqlens = torch.randint(
        0,
        config.past_kv_sequence_length - config.q_sequence_length + 1,
        (config.batch_size,),
        device=device,
        dtype=torch.long,
    )

    for i in range(config.batch_size):
        past_len = cache_seqlens[i].item()
        k[i, :, past_len:, :] = 0
        v[i, :, past_len:, :] = 0

    new_k = (
        torch.randn(
            config.batch_size,
            config.q_sequence_length,
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
    if config.rotary:
        rotary_dim = math.floor(config.head_size / 16) * 16
        angle = torch.rand(config.buffer_sequence_length, rotary_dim // 2, device=device) * 2 * math.pi
        cos = torch.cos(angle).to(dtype=torch_type)
        sin = torch.sin(angle).to(dtype=torch_type)
        q_ro = apply_rotary_embedding(q.clone(), cos, sin, cache_seqlens, config.rotary_interleaved, device)
        k_ro = apply_rotary_embedding(new_k.clone(), cos, sin, cache_seqlens, config.rotary_interleaved, device)

    position_ids, attention_bias = None, None
    total_seq_len = config.past_kv_sequence_length + config.q_sequence_length
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
    k_cache_ref[update_mask] = rearrange(k_ro, "b s ... -> (b s) ...").to(k_cache_ref.dtype)
    v_cache_ref[update_mask] = rearrange(new_v, "b s ... -> (b s) ...").to(v_cache_ref.dtype)
    key_padding_mask = arange < cache_seqlens_expanded + config.q_sequence_length

    out_ref, _ = attention_ref(
        q=q_ro,
        k=k_cache_ref,
        v=v_cache_ref,
        key_padding_mask=key_padding_mask,
        attention_bias=attention_bias,
        causal=True,
        window_size=window_size,
        softcap=config.softcap,
        use_smooth_softmax=config.use_smooth_softmax,
        head_sink=head_sink,
    )
    out_ref_np = out_ref.detach().cpu().numpy()

    # --- ONNX Runtime Path ---
    q_ort, new_k_ort, new_v_ort = q, new_k, new_v
    if config.packed:
        q_ort = torch.cat([q, new_k, new_v], dim=2)
        new_k_ort, new_v_ort = None, None

    k_quant = quantize_tensor_with_scale(k, k_scale, config.k_quant_type, config.kv_cache_type)
    v_quant = quantize_tensor_with_scale(v, v_scale, config.v_quant_type, config.kv_cache_type)

    ort_seqlens = cache_seqlens + config.q_sequence_length - 1
    out, present_k, present_v = gqa_past_func(
        q=q_ort,
        k=k_quant,
        v=v_quant,
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
        share_buffer=True,
        ort_type=ort_type,
        numpy_type=numpy_type,
    )
    out = torch.reshape(out, (config.batch_size, config.q_sequence_length, config.num_heads, config.head_size))
    out_np = out.detach().cpu().numpy()

    # --- Comparison ---
    numpy.testing.assert_allclose(out_np, out_ref_np, rtol=rtol, atol=atol)

    # Compare quantized cache with proper masking per batch
    if config.k_quant_type != "NONE":
        if config.kv_cache_type == "int4":
            present_k_torch = torch.from_numpy(present_k).to(device)
        elif config.kv_cache_type == "int8":
            present_k_torch = torch.from_numpy(present_k.astype(numpy.int8)).to(device)
        else:
            present_k_torch = torch.from_numpy(present_k).to(device)

        present_k_dequant = (
            dequantize_tensor(present_k_torch, k_scale, config.k_quant_type, config.kv_cache_type).cpu().numpy()
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
            numpy.testing.assert_allclose(
                present_k_dequant[b, :, :valid_len, :],
                k_cache_ref_dequant[b, :, :valid_len, :],
                rtol=rtol,
                atol=atol,
            )

    if config.v_quant_type != "NONE":
        if config.kv_cache_type == "int4":
            present_v_torch = torch.from_numpy(present_v).to(device)
        elif config.kv_cache_type == "int8":
            present_v_torch = torch.from_numpy(present_v.astype(numpy.int8)).to(device)
        else:
            present_v_torch = torch.from_numpy(present_v).to(device)

        present_v_dequant = (
            dequantize_tensor(present_v_torch, v_scale, config.v_quant_type, config.kv_cache_type).cpu().numpy()
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
            numpy.testing.assert_allclose(
                present_v_dequant[b, :, :valid_len, :],
                v_cache_ref_dequant[b, :, :valid_len, :],
                rtol=rtol,
                atol=atol,
            )


# #################################################################################################
#  Test Case Generators
# #################################################################################################


def get_cuda_rotary_options():
    options = [(False, False), (True, False), (True, True)]
    return options[:param_count]


def get_cpu_rotary_options():
    return [(False, False), (True, False), (True, True)]


def get_softmax_options(allow_head_sink: bool = True):
    options = [(False, False), (False, True), (True, False)]
    return options[:2] if pipeline_mode else options[:param_count]


def gqa_cuda_prompt_test_cases(allow_head_sink: bool = True):
    batches = [3, 1, 5]
    seqs = [(35, 35), (127, 127), (240, 240), (2000, 2000)]
    heads = [(6, 3), (9, 9), (32, 8)]
    h_sizes = [128, 32, 64, 256]
    smmoth_softmax__head_sink = get_softmax_options(allow_head_sink)

    for b in batches[:param_count]:
        for sq, skv in seqs[:param_count]:
            for n, n2 in heads[:param_count]:
                for h in h_sizes[:param_count]:
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
    batches = [2, 1, 3]
    # s: new sequence length, s2: past sequence length
    seqs = [(1, 128), (1, 1024), (1, 2048), (1, 5000)]
    heads = [(32, 8), (6, 3), (9, 9)]
    # We test 128 in pipeline since quantized kv cache is only enabled for head_size=128 in flash attention.
    h_sizes = [128, 64, 256]
    smmoth_softmax__head_sink = get_softmax_options(allow_head_sink)

    for b in batches[:param_count]:
        for s, s2 in seqs[:param_count]:
            for n, n2 in heads[:param_count]:
                for h in h_sizes[:param_count]:
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


def gqa_cuda_quantized_test_cases(is_past):
    base_cases = gqa_cuda_past_test_cases() if is_past else gqa_cuda_prompt_test_cases()
    for name, config in base_cases:
        if config.packed:  # Quantization is not supported with packed QKV yet
            continue
        for kv_type in ["int8", "int4"]:
            for quant_mode in ["PER_TENSOR", "PER_CHANNEL"]:
                q_config = deepcopy(config)
                q_config.k_quant_type = quant_mode
                q_config.v_quant_type = quant_mode
                q_config.kv_cache_type = kv_type
                if kv_type == "int4":
                    if q_config.head_size % 2 != 0:
                        continue
                    q_config.kv_cache_bit_width = 4
                elif kv_type == "int8":
                    q_config.kv_cache_bit_width = 8
                q_name = f"{name}_quant_{kv_type}_{quant_mode}"
                yield q_name, q_config


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


def has_quantized_kv_cache():
    return version.parse(onnxruntime.__version__) >= version.parse("1.24.0")


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
            rtol=2e-2,
            atol=2e-2,
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
            atol=2e-2,
        )


@unittest.skipIf(not has_flash_attention(), "Flash Attention is not available, skipping tests.")
@unittest.skipIf(not has_quantized_kv_cache(), "Quantized KV Cache is not available, skipping tests.")
class TestQuantizedGQA(unittest.TestCase):
    @parameterized.expand(gqa_cuda_quantized_test_cases(is_past=False))
    def test_gqa_quantized_prompt(self, name, config):
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "0"
        parity_check_gqa_prompt(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            numpy_type=numpy.float16,
            ort_type=TensorProto.FLOAT16,
            causal=True,
            rtol=5e-2,
            atol=0.15,
        )

    @parameterized.expand(gqa_cuda_quantized_test_cases(is_past=True))
    def test_gqa_quantized_past(self, name, config):
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "0"
        parity_check_gqa_past(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            numpy_type=numpy.float16,
            ort_type=TensorProto.FLOAT16,
            causal=True,
            rtol=5e-2,
            atol=0.15,
        )


if __name__ == "__main__":
    unittest.main()
