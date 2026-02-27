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

"""
Shared utilities for ONNX Attention op (opset 23) tests.

Contains configuration, ONNX graph builders, reference implementation,
and parity check helpers used by both GQA and MHA test modules.
"""

import math
import os
from dataclasses import dataclass

import torch
from einops import rearrange, repeat
from onnx import TensorProto, helper

from onnxruntime import (
    InferenceSession,
    SessionOptions,
    get_available_providers,
    get_build_info,
)

# Set seed for reproducibility
torch.manual_seed(0)

# Reduces number of tests to run for faster pipeline checks
pipeline_mode = os.getenv("PIPELINE_MODE", "1") == "1"

# Number of values per parameter (compared to pipeline mode)
param_count = int(os.getenv("PARAM_COUNT", "3")) if not pipeline_mode else 2

# When quick build is used, flash attention only supports head_size=128
quick_build = ", quick-build=" in get_build_info()

enable_debug_print = quick_build

enable_deterministic_check = True

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

TORCH_DTYPE_TO_ONNX_MAP = {
    torch.float32: TensorProto.FLOAT,
    torch.float16: TensorProto.FLOAT16,
    torch.bfloat16: TensorProto.BFLOAT16,
    torch.int32: TensorProto.INT32,
    torch.int8: TensorProto.INT8,
}

TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int4": torch.uint8,
}


@dataclass
class AttentionConfig:
    batch_size: int
    q_sequence_length: int
    kv_sequence_length: int
    q_num_heads: int
    kv_num_heads: int
    head_size: int
    is_causal: int = 0
    past_kv_sequence_length: int = 0
    softcap: float = 0.0
    kv_cache_type: str = ""
    has_attn_mask: bool = False
    attn_mask_dims: int = 2  # 2D, 3D, or 4D boolean mask
    attn_mask_type: str = "bool"  # "bool" for GQA path, "additive" for MHA path


# #################################################################################################
#  ONNX Graph Creation
# #################################################################################################


def create_attention_node_and_io(
    config: AttentionConfig,
    ort_type,
    is_past=False,
    output_qk: int = 0,  # CUDA does not support output_qk for GQA path
):
    """
    Create ONNX Attention op node and I/O definitions for testing.

    ONNX Attention op (opset 23) inputs:
    - 0: Q (query) - required
    - 1: K (key) - required
    - 2: V (value) - required
    - 3: attn_mask - optional
    - 4: past_key - optional
    - 5: past_value - optional

    ONNX Attention op outputs:
    - 0: Y (output)
    - 1: present_key (optional)
    - 2: present_value (optional)
    - 3: output_qk (optional)
    """
    # For ONNX Attention op, present KV cache grows (not fixed buffer like GQA)
    if is_past:
        present_kv_seqlen = config.past_kv_sequence_length + config.kv_sequence_length
    else:  # Prompt (no past KV cache)
        present_kv_seqlen = config.kv_sequence_length

    if not config.kv_cache_type:
        config.kv_cache_type = {
            TensorProto.FLOAT16: "float16",
            TensorProto.BFLOAT16: "bfloat16",
            TensorProto.FLOAT: "float32",
        }.get(ort_type, "float16")

    # --- Node Definition ---
    outputs = [
        "output",
        "present_key",
        "present_value",
    ]

    if output_qk > 0:
        outputs.append("output_qk")

    # ONNX Attention op inputs: Q, K, V, attn_mask, past_key, past_value
    # attn_mask is used as padding mask (additive bias: 0.0 for valid, -inf for masked)
    inputs = [
        "query",
        "key",
        "value",
        "attn_mask" if config.has_attn_mask else "",
        "past_key" if is_past else "",
        "past_value" if is_past else "",
    ]

    # Remove trailing empty strings
    while inputs and inputs[-1] == "":
        inputs.pop()

    # ONNX Attention op attributes (opset 23)
    node = helper.make_node(
        op_type="Attention",
        inputs=inputs,
        outputs=outputs,
        name="Attention_0",
        is_causal=config.is_causal,
        kv_num_heads=config.kv_num_heads,
        q_num_heads=config.q_num_heads,
        softcap=config.softcap,
        qk_matmul_output_mode=output_qk,
        domain="",  # ai.onnx domain
    )

    # --- Graph Inputs ---
    # ONNX Attention op uses 3D inputs: [batch, seq_len, hidden_size]
    q_hidden_size = config.q_num_heads * config.head_size
    kv_hidden_size = config.kv_num_heads * config.head_size

    graph_input = [
        helper.make_tensor_value_info("query", ort_type, [config.batch_size, config.q_sequence_length, q_hidden_size]),
        helper.make_tensor_value_info("key", ort_type, [config.batch_size, config.kv_sequence_length, kv_hidden_size]),
        helper.make_tensor_value_info(
            "value", ort_type, [config.batch_size, config.kv_sequence_length, kv_hidden_size]
        ),
    ]

    if isinstance(config.kv_cache_type, torch.dtype):
        cache_ort_type = TORCH_DTYPE_TO_ONNX_MAP[config.kv_cache_type]
    else:
        cache_ort_type = ONNX_TENSOR_TYPE_MAP[config.kv_cache_type]

    # attn_mask for ONNX Attention op
    if config.has_attn_mask:
        mask_seq_len = present_kv_seqlen

        # Determine mask ONNX type
        if config.attn_mask_type == "bool":
            mask_ort_type = TensorProto.BOOL
        else:
            mask_ort_type = ort_type  # additive mask uses same type as Q/K/V

        # Mask shapes differ between GQA (bool) and MHA (additive) paths:
        # GQA bool: 2D=[batch, total_seq], 3D=[heads, q_seq, total_seq], 4D=[batch, heads, q_seq, total_seq]
        # MHA additive: 2D=[q_seq, total_seq], 3D=[heads, q_seq, total_seq], 4D=[batch, heads, q_seq, total_seq]
        # ONNX broadcasting aligns from the right: 3D [A, B, C] → [_, A, B, C] where A=heads
        if config.attn_mask_type == "bool":
            if config.attn_mask_dims == 2:
                mask_shape = [config.batch_size, mask_seq_len]
            elif config.attn_mask_dims == 3:
                mask_shape = [config.q_num_heads, config.q_sequence_length, mask_seq_len]
            else:  # 4D
                mask_shape = [config.batch_size, config.q_num_heads, config.q_sequence_length, mask_seq_len]
        else:  # additive
            if config.attn_mask_dims == 2:
                mask_shape = [config.q_sequence_length, mask_seq_len]
            elif config.attn_mask_dims == 3:
                # 3D aligns to [_, heads, q_seq, total_seq] — dim 0 must be 1 or num_heads
                mask_shape = [config.q_num_heads, config.q_sequence_length, mask_seq_len]
            else:  # 4D
                mask_shape = [config.batch_size, config.q_num_heads, config.q_sequence_length, mask_seq_len]
        graph_input.append(helper.make_tensor_value_info("attn_mask", mask_ort_type, mask_shape))

    # past_key and past_value for ONNX Attention op
    # Shape: [batch, num_heads, past_seq_len, head_size] (4D BNSH format)
    if is_past:
        past_k_shape = [config.batch_size, config.kv_num_heads, config.past_kv_sequence_length, config.head_size]
        graph_input.extend(
            [
                helper.make_tensor_value_info("past_key", cache_ort_type, past_k_shape),
                helper.make_tensor_value_info("past_value", cache_ort_type, past_k_shape),
            ]
        )

    # --- Graph Outputs ---
    output_k_shape = [config.batch_size, config.kv_num_heads, present_kv_seqlen, config.head_size]

    graph_output = [
        helper.make_tensor_value_info(
            "output", ort_type, [config.batch_size, config.q_sequence_length, config.q_num_heads * config.head_size]
        ),
        helper.make_tensor_value_info("present_key", cache_ort_type, output_k_shape),
        helper.make_tensor_value_info("present_value", cache_ort_type, output_k_shape),
    ]

    if output_qk > 0:
        graph_output.append(
            helper.make_tensor_value_info(
                "output_qk",
                ort_type,
                [config.batch_size, config.q_num_heads, config.q_sequence_length, present_kv_seqlen],
            )
        )

    return node, graph_input, graph_output


def create_attention_graph_prompt(config: AttentionConfig, ort_type):
    """Create ONNX graph for prompt phase (no past KV cache)."""
    node, graph_input, graph_output = create_attention_node_and_io(config, ort_type, is_past=False)
    graph = helper.make_graph([node], "Attention_Graph", graph_input, graph_output)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])
    return model.SerializeToString()


def create_attention_graph_past(config: AttentionConfig, ort_type):
    """Create ONNX graph for decoding phase (with past KV cache)."""
    node, graph_input, graph_output = create_attention_node_and_io(config, ort_type, is_past=True)
    graph = helper.make_graph([node], "Attention_Graph", graph_input, graph_output)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])
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


def _get_out_dtype(ort_type):
    """Get the torch dtype for output tensors given an ORT type."""
    if ort_type == TensorProto.BFLOAT16:
        return torch.bfloat16
    elif ort_type == TensorProto.FLOAT16:
        return torch.float16
    else:
        return torch.float32


def _get_mask_ort_type(config: AttentionConfig, ort_type):
    """Get the ORT type for the attention mask."""
    if config.attn_mask_type == "bool":
        return TensorProto.BOOL
    else:
        return ort_type


def attention_prompt_func(
    q,
    k,
    v,
    config: AttentionConfig,
    attn_mask,
    ep,
    device,
    ort_type=TensorProto.FLOAT16,
):
    """
    Run ONNX Attention op for prompt phase (no past KV cache).

    Args:
        q: Query tensor [batch, q_seq_len, q_num_heads, head_size]
        k: Key tensor [batch, kv_seq_len, kv_num_heads, head_size]
        v: Value tensor [batch, kv_seq_len, kv_num_heads, head_size]
        config: AttentionConfig with model parameters
        attn_mask: Optional attention mask tensor
        ep: Execution provider (e.g., "CUDAExecutionProvider")
        device: Device string (e.g., "cuda")
        ort_type: ONNX tensor type
    """
    if not config.kv_cache_type:
        config.kv_cache_type = {
            TensorProto.FLOAT16: "float16",
            TensorProto.BFLOAT16: "bfloat16",
            TensorProto.FLOAT: "float32",
        }.get(ort_type, "float16")

    onnx_model_str = create_attention_graph_prompt(
        config=config,
        ort_type=ort_type,
    )

    # Reshape to 3D [batch, seq_len, hidden_size]
    q_3d = torch.reshape(q, (config.batch_size, config.q_sequence_length, -1))
    k_3d = torch.reshape(k, (config.batch_size, config.kv_sequence_length, -1))
    v_3d = torch.reshape(v, (config.batch_size, config.kv_sequence_length, -1))

    sess_options = SessionOptions()
    ort_session = InferenceSession(onnx_model_str, sess_options, providers=[ep])
    io_binding = ort_session.io_binding()

    # Bind inputs
    bind_tensor(io_binding, "query", q_3d, device, ort_type)
    bind_tensor(io_binding, "key", k_3d, device, ort_type)
    bind_tensor(io_binding, "value", v_3d, device, ort_type)

    # Bind optional attention mask
    if config.has_attn_mask and attn_mask is not None:
        mask_ort_type = _get_mask_ort_type(config, ort_type)
        bind_tensor(io_binding, "attn_mask", attn_mask, device, mask_ort_type)

    # Bind Outputs
    hidden_size = config.q_num_heads * config.head_size
    out_dtype = _get_out_dtype(ort_type)

    out_torch = torch.zeros((config.batch_size, config.q_sequence_length, hidden_size), dtype=out_dtype, device=device)
    bind_output_tensor(io_binding, "output", out_torch, device, ort_type)

    # present KV shape for prompt (no past)
    present_seqlen = config.kv_sequence_length
    present_dims = [config.batch_size, config.kv_num_heads, present_seqlen, config.head_size]

    # Determine dtype for cache tensors
    cache_dtype = out_dtype
    if isinstance(config.kv_cache_type, torch.dtype):
        cache_ort_type = TORCH_DTYPE_TO_ONNX_MAP[config.kv_cache_type]
    else:
        cache_ort_type = ONNX_TENSOR_TYPE_MAP[config.kv_cache_type]

    present_k = torch.zeros(tuple(present_dims), dtype=cache_dtype, device=device)
    present_v = torch.zeros(tuple(present_dims), dtype=cache_dtype, device=device)
    bind_output_tensor(io_binding, "present_key", present_k, device, cache_ort_type)
    bind_output_tensor(io_binding, "present_value", present_v, device, cache_ort_type)

    ort_session.run_with_iobinding(io_binding)

    return out_torch, present_k, present_v


def attention_past_func(
    q,
    past_k,
    past_v,
    new_k,
    new_v,
    config: AttentionConfig,
    attn_mask,
    ep,
    device,
    ort_type=TensorProto.FLOAT16,
):
    """
    Run ONNX Attention op for decoding phase (with past KV cache).

    Args:
        q: Query tensor [batch, q_seq_len, q_num_heads, head_size]
        past_k: Past key tensor [batch, kv_num_heads, past_seq_len, head_size]
        past_v: Past value tensor [batch, kv_num_heads, past_seq_len, head_size]
        new_k: New key tensor [batch, kv_seq_len, kv_num_heads, head_size]
        new_v: New value tensor [batch, kv_seq_len, kv_num_heads, head_size]
        config: AttentionConfig with model parameters
        attn_mask: Optional attention mask tensor
        ep: Execution provider (e.g., "CUDAExecutionProvider")
        device: Device string (e.g., "cuda")
        ort_type: ONNX tensor type
    """
    if not config.kv_cache_type:
        config.kv_cache_type = {
            TensorProto.FLOAT16: "float16",
            TensorProto.BFLOAT16: "bfloat16",
            TensorProto.FLOAT: "float32",
        }.get(ort_type, "float16")

    onnx_model_str = create_attention_graph_past(
        config=config,
        ort_type=ort_type,
    )

    # Reshape to 3D [batch, seq_len, hidden_size]
    q_3d = torch.reshape(q, (config.batch_size, config.q_sequence_length, -1))
    new_k_3d = torch.reshape(new_k, (config.batch_size, config.kv_sequence_length, -1))
    new_v_3d = torch.reshape(new_v, (config.batch_size, config.kv_sequence_length, -1))

    sess_options = SessionOptions()
    ort_session = InferenceSession(onnx_model_str, sess_options, providers=[ep])
    io_binding = ort_session.io_binding()

    # Total sequence length for present KV
    total_seq_len = config.past_kv_sequence_length + config.kv_sequence_length

    # Bind inputs
    bind_tensor(io_binding, "query", q_3d, device, ort_type)
    bind_tensor(io_binding, "key", new_k_3d, device, ort_type)
    bind_tensor(io_binding, "value", new_v_3d, device, ort_type)

    # Bind optional attention mask
    if config.has_attn_mask and attn_mask is not None:
        mask_ort_type = _get_mask_ort_type(config, ort_type)
        bind_tensor(io_binding, "attn_mask", attn_mask, device, mask_ort_type)

    # Bind past_key and past_value
    if isinstance(config.kv_cache_type, torch.dtype):
        cache_ort_type = TORCH_DTYPE_TO_ONNX_MAP[config.kv_cache_type]
    else:
        cache_ort_type = ONNX_TENSOR_TYPE_MAP[config.kv_cache_type]

    # past_k and past_v should be sliced to actual past length
    past_len = config.past_kv_sequence_length
    past_k_sliced = past_k[:, :, :past_len, :].contiguous()
    past_v_sliced = past_v[:, :, :past_len, :].contiguous()
    bind_tensor(io_binding, "past_key", past_k_sliced, device, cache_ort_type)
    bind_tensor(io_binding, "past_value", past_v_sliced, device, cache_ort_type)

    # Bind Outputs
    hidden_size = config.q_num_heads * config.head_size
    out_dtype = _get_out_dtype(ort_type)

    out_torch = torch.zeros((config.batch_size, config.q_sequence_length, hidden_size), dtype=out_dtype, device=device)
    bind_output_tensor(io_binding, "output", out_torch, device, ort_type)

    # present KV shape (past + new)
    present_seqlen = total_seq_len
    present_dims = [config.batch_size, config.kv_num_heads, present_seqlen, config.head_size]

    cache_dtype = out_dtype
    present_k = torch.zeros(tuple(present_dims), dtype=cache_dtype, device=device)
    present_v = torch.zeros(tuple(present_dims), dtype=cache_dtype, device=device)
    bind_output_tensor(io_binding, "present_key", present_k, device, cache_ort_type)
    bind_output_tensor(io_binding, "present_value", present_v, device, cache_ort_type)

    ort_session.run_with_iobinding(io_binding)

    return out_torch, present_k, present_v


# #################################################################################################
#  Reference Attention Implementation
# #################################################################################################


def construct_causal_mask(seqlen_q, seqlen_k, device):
    """Construct a causal mask for attention."""
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    # Causal: positions can only attend to earlier positions
    return col_idx > row_idx + seqlen_k - seqlen_q


def attention_ref(
    q,
    k,
    v,
    key_padding_mask=None,
    attn_bias=None,
    causal=False,
    softcap=0.0,
):
    """
    Reference implementation of scaled dot-product attention with GQA support.

    Args:
        q: Query tensor [batch, seq_q, num_heads, head_size]
        k: Key tensor [batch, seq_k, kv_num_heads, head_size]
        v: Value tensor [batch, seq_k, kv_num_heads, head_size]
        key_padding_mask: Boolean mask [batch, seq_k] - True for valid, False for masked
        attn_bias: Additive attention bias [broadcastable to batch, num_heads, seq_q, seq_k]
        causal: Whether to apply causal masking
        softcap: Softcap value for attention scores (0.0 = disabled)

    Returns:
        output: Attention output [batch, seq_q, num_heads, head_size]
        attention: Attention weights [batch, num_heads, seq_q, seq_k]
    """
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

    if attn_bias is not None:
        scores = scores + attn_bias.float()

    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))

    if causal:
        causal_mask = construct_causal_mask(seqlen_q, seqlen_k, q.device)
        scores.masked_fill_(causal_mask, float("-inf"))

    attention = torch.softmax(scores, dim=-1)

    output = torch.einsum("bhts,bshd->bthd", attention, v)

    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


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
#  Attention Mask Helper Functions
# #################################################################################################


def create_boolean_mask_from_seqlens(
    seqlens: torch.Tensor,
    total_seq_len: int,
    mask_dims: int,
    q_seq_len: int = 1,
    num_heads: int = 1,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Create a boolean attention mask from sequence lengths.

    ONNX broadcasting aligns dimensions from the right (trailing dimensions).
    Target broadcast shape: (batch_size, q_num_heads, q_seq_len, total_seq_len)

    Args:
        seqlens: Tensor of shape [batch_size] containing the valid sequence length for each batch.
        total_seq_len: The total sequence length (last dimension of the mask).
        mask_dims: Number of dimensions for the mask (2, 3, or 4).
        q_seq_len: Query sequence length (for 3D/4D masks).
        num_heads: Number of q_heads (for 3D/4D masks).
        device: Device for the tensor.

    Returns:
        Boolean mask where True = valid, False = padding.
        - 2D: [batch_size, total_seq_len] - broadcasts to [batch, 1, 1, total_seq]
        - 3D: [num_heads, q_seq_len, total_seq_len] - broadcasts to [1, num_heads, q_seq, total_seq]
        - 4D: [batch_size, num_heads, q_seq_len, total_seq_len] - no broadcasting
    """
    batch_size = seqlens.shape[0]

    # Create base 2D mask [batch_size, total_seq_len]
    # mask[b, i] = True if i < seqlens[b]
    arange = torch.arange(total_seq_len, device=device).unsqueeze(0)  # [1, total_seq_len]
    seqlens_expanded = seqlens.unsqueeze(1)  # [batch_size, 1]
    mask_2d = arange < seqlens_expanded  # [batch_size, total_seq_len]

    if mask_dims == 2:
        return mask_2d
    elif mask_dims == 3:
        # 3D mask: [num_heads, q_seq_len, total_seq_len]
        # For right-padding tests, all batches should have the same mask pattern per position.
        # Since seqlens can vary per batch, we use the first batch's pattern and expand across heads.
        # This is valid for testing because the 3D mask broadcasts across batches (dim 0 becomes 1).
        # For a more general case, 3D masks would need uniform seqlens across batches.
        mask_1d = mask_2d[0:1, :]  # Take first batch pattern [1, total_seq_len]
        mask_3d = mask_1d.unsqueeze(0).expand(num_heads, q_seq_len, total_seq_len).contiguous()
        return mask_3d
    else:  # 4D
        # Expand to [batch_size, num_heads, q_seq_len, total_seq_len]
        # The mask is the same for all heads and query positions
        return mask_2d.unsqueeze(1).unsqueeze(1).expand(batch_size, num_heads, q_seq_len, total_seq_len).contiguous()


def create_additive_mask_from_seqlens(
    seqlens: torch.Tensor,
    total_seq_len: int,
    mask_dims: int,
    q_seq_len: int = 1,
    num_heads: int = 1,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Create an additive attention mask from sequence lengths.

    Valid positions get 0.0, masked positions get -inf.
    This is used for the MHA path which expects additive bias.

    Args:
        seqlens: Tensor of shape [batch_size] containing the valid sequence length for each batch.
        total_seq_len: The total sequence length (last dimension of the mask).
        mask_dims: Number of dimensions for the mask (2, 3, or 4).
        q_seq_len: Query sequence length (for 3D/4D masks).
        num_heads: Number of heads (for 3D/4D masks).
        device: Device for the tensor.
        dtype: Torch dtype for the mask.

    Returns:
        Additive mask where 0.0 = valid, -inf = masked.
        - 2D: [q_seq_len, total_seq_len]
        - 3D: [num_heads, q_seq_len, total_seq_len]
        - 4D: [batch_size, num_heads, q_seq_len, total_seq_len]
    """
    batch_size = seqlens.shape[0]

    # Create base boolean mask [batch_size, total_seq_len]
    arange = torch.arange(total_seq_len, device=device).unsqueeze(0)
    seqlens_expanded = seqlens.unsqueeze(1)
    bool_mask = arange < seqlens_expanded  # True for valid

    # Convert to additive: 0.0 for valid, -inf for masked
    additive_4d = torch.zeros(batch_size, num_heads, q_seq_len, total_seq_len, device=device, dtype=dtype)
    # Expand bool mask to 4D [batch, 1, 1, total_seq] and apply
    additive_4d.masked_fill_(~bool_mask.unsqueeze(1).unsqueeze(1), float("-inf"))

    if mask_dims == 2:
        # 2D: [q_seq_len, total_seq_len] — only works when all batches have same seqlens
        return additive_4d[0, 0, :, :]
    elif mask_dims == 3:
        # 3D: [heads, q_seq_len, total_seq_len] — batch always broadcasts, use first batch
        return additive_4d[0, :, :, :]
    else:  # 4D
        return additive_4d


# #################################################################################################
#  Hardware / Provider Helpers
# #################################################################################################


def has_cuda_provider():
    return "CUDAExecutionProvider" in get_available_providers()


def has_cuda_device(min_capability: int = 80):
    if not has_cuda_provider() or not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor >= min_capability


def has_flash_attention():
    return has_cuda_device(80)


# Default tolerances
# Note: fp32 tolerances are relaxed because TF32 is enabled by default on Ampere+ GPUs
# (see attention.cc: use_tf32 = UseTF32()), giving roughly fp16-level matmul precision.
rtol = {"fp16": 5e-3, "fp32": 5e-3, "bf16": 5e-2}
atol = {"fp16": 5e-3, "fp32": 5e-3, "bf16": 1e-2}
