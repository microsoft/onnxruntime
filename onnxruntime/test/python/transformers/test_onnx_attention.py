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
import unittest
from dataclasses import dataclass

import numpy
import torch
from einops import rearrange, repeat
from onnx import TensorProto, helper
from parameterized import parameterized

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
        config.kv_cache_type = "float16" if ort_type == TensorProto.FLOAT16 else "bfloat16"

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

    # attn_mask for ONNX Attention op - boolean padding mask
    # GQA path expects boolean mask: True for valid, False for masked
    # Supports 2D, 3D, or 4D masks that broadcast to (batch, q_num_heads, q_seq, total_seq):
    #   2D: [batch_size, total_seq_len] - broadcasts across heads and query positions
    #   3D: [q_num_heads, q_seq_len, total_seq_len] - broadcasts across batches (ONNX aligns from right)
    #   4D: [batch_size, q_num_heads, q_seq_len, total_seq_len] - no broadcasting needed
    # The kernel converts this to seqlens_k internally
    if config.has_attn_mask:
        mask_seq_len = present_kv_seqlen
        if config.attn_mask_dims == 2:
            mask_shape = [config.batch_size, mask_seq_len]
        elif config.attn_mask_dims == 3:
            # 3D mask: [q_num_heads, q_seq_len, total_seq_len] broadcasts to [1, q_num_heads, q_seq_len, total_seq_len]
            mask_shape = [config.q_num_heads, config.q_sequence_length, mask_seq_len]
        else:  # 4D
            mask_shape = [config.batch_size, config.q_num_heads, config.q_sequence_length, mask_seq_len]
        graph_input.append(helper.make_tensor_value_info("attn_mask", TensorProto.BOOL, mask_shape))

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
        attn_mask: Optional attention mask tensor (additive bias, 0.0 for valid, -inf for masked)
        ep: Execution provider (e.g., "CUDAExecutionProvider")
        device: Device string (e.g., "cuda")
        ort_type: ONNX tensor type
    """
    if not config.kv_cache_type:
        config.kv_cache_type = "float16" if ort_type == TensorProto.FLOAT16 else "bfloat16"

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

    # Bind optional attention mask (boolean padding mask: True=valid, False=masked)
    if config.has_attn_mask and attn_mask is not None:
        bind_tensor(io_binding, "attn_mask", attn_mask, device, TensorProto.BOOL)

    # Bind Outputs
    hidden_size = config.q_num_heads * config.head_size

    if ort_type == TensorProto.BFLOAT16:
        out_dtype = torch.bfloat16
    elif ort_type == TensorProto.FLOAT16:
        out_dtype = torch.float16
    else:
        out_dtype = torch.float32

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
        attn_mask: Optional attention mask tensor (additive bias, 0.0 for valid, -inf for masked)
        ep: Execution provider (e.g., "CUDAExecutionProvider")
        device: Device string (e.g., "cuda")
        ort_type: ONNX tensor type
    """
    if not config.kv_cache_type:
        config.kv_cache_type = "float16" if ort_type == TensorProto.FLOAT16 else "bfloat16"

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

    # Bind optional attention mask (boolean padding mask: True=valid, False=masked)
    if config.has_attn_mask and attn_mask is not None:
        bind_tensor(io_binding, "attn_mask", attn_mask, device, TensorProto.BOOL)

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

    if ort_type == TensorProto.BFLOAT16:
        out_dtype = torch.bfloat16
    elif ort_type == TensorProto.FLOAT16:
        out_dtype = torch.float16
    else:
        out_dtype = torch.float32

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

    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))

    if causal:
        causal_mask = construct_causal_mask(seqlen_q, seqlen_k, q.device)
        scores.masked_fill_(causal_mask, float("-inf"))

    attention = torch.softmax(scores, dim=-1)

    output = torch.einsum("bhts,bshd->bthd", attention, v)

    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


# #################################################################################################
# Parity Check (Core Test Logic)
# #################################################################################################


def parity_check_attention_prompt(
    config: AttentionConfig,
    ep,
    device,
    torch_type,
    ort_type,
    causal,
    rtol,
    atol,
    std=0.2,
):
    """
    Parity check for ONNX Attention op in prompt phase (no past KV cache).

    This tests that the ONNX Attention op produces the same output as a PyTorch
    reference implementation for the initial prompt processing.
    """
    torch.manual_seed(0)

    # Generate Q, K, V tensors in BSNH format (batch, seq, num_heads, head_size)
    q = (
        torch.randn(
            config.batch_size,
            config.q_sequence_length,
            config.q_num_heads,
            config.head_size,
            device=device,
            dtype=torch_type,
        )
        * std
    )
    k = (
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
    v = torch.randn_like(k) * std

    # --- Create attn_mask as boolean padding mask (simulating seqlens_k) ---
    # For testing, we use full sequence length (no actual padding)
    # attn_mask: [batch, kv_seq_len] - True for valid, False for masked
    # GQA kernel converts this to seqlens_k internally
    attn_mask = None
    key_padding_mask = None
    if config.has_attn_mask:
        # All positions are valid (no padding) for this test
        # Create a 2D boolean mask of True (all valid positions)
        attn_mask = torch.ones(
            config.batch_size,
            config.kv_sequence_length,
            device=device,
            dtype=torch.bool,
        )
        # key_padding_mask for reference: all True (all valid)
        key_padding_mask = torch.ones(
            config.batch_size,
            config.kv_sequence_length,
            device=device,
            dtype=torch.bool,
        )

    # --- PyTorch Reference Path ---
    out_ref, _ = attention_ref(
        q=q,
        k=k,
        v=v,
        key_padding_mask=key_padding_mask,
        causal=causal,
        softcap=config.softcap,
    )
    out_ref_np = out_ref.to(torch.float32).detach().cpu().numpy()

    # --- ONNX Runtime Path ---
    num_runs = 2 if enable_deterministic_check else 1
    for i in range(num_runs):
        out, present_k, present_v = attention_prompt_func(
            q=q,
            k=k,
            v=v,
            config=config,
            attn_mask=attn_mask,
            ep=ep,
            device=device,
            ort_type=ort_type,
        )
        if i == 0:
            first_out = out.clone()
            first_present_k = present_k.clone() if present_k is not None else None
            first_present_v = present_v.clone() if present_v is not None else None
        else:
            if present_k is not None:
                torch.testing.assert_close(
                    present_k, first_present_k, rtol=0, atol=0, msg="present_k mismatch between two runs"
                )
            if present_v is not None:
                torch.testing.assert_close(
                    present_v, first_present_v, rtol=0, atol=0, msg="present_v mismatch between two runs"
                )
            torch.testing.assert_close(out, first_out, rtol=0, atol=0, msg="Output mismatch between two runs")

    out = torch.reshape(out, (config.batch_size, config.q_sequence_length, config.q_num_heads, config.head_size))
    out_np = out.to(torch.float32).detach().cpu().numpy()

    # --- Comparison ---
    # Check for NaN in output
    nan_count = numpy.sum(numpy.isnan(out_np))
    if nan_count > 0:
        nan_indices = numpy.argwhere(numpy.isnan(out_np))
        print(f"DEBUG_NAN: Found {nan_count} NaN values in output!")
        print(f"DEBUG_NAN: First 5 NaN indices: {nan_indices[:5]}")

    # Compare KV cache (present_k should match k, present_v should match v)
    # K/V are in BSNH, present_k/v are in BNSH - need to transpose for comparison
    k_ref_bnsh = k.transpose(1, 2)  # BSNH -> BNSH
    v_ref_bnsh = v.transpose(1, 2)  # BSNH -> BNSH

    k_ref_np = k_ref_bnsh.to(torch.float32).detach().cpu().numpy()
    v_ref_np = v_ref_bnsh.to(torch.float32).detach().cpu().numpy()
    present_k_np = present_k.to(torch.float32).detach().cpu().numpy()
    present_v_np = present_v.to(torch.float32).detach().cpu().numpy()

    print_diff_statistics(torch.tensor(present_k_np - k_ref_np), "present_k")
    numpy.testing.assert_allclose(present_k_np, k_ref_np, rtol=rtol, atol=atol)
    print_diff_statistics(torch.tensor(present_v_np - v_ref_np), "present_v")
    numpy.testing.assert_allclose(present_v_np, v_ref_np, rtol=rtol, atol=atol)

    print_diff_statistics(torch.tensor(out_np - out_ref_np), "out")
    numpy.testing.assert_allclose(out_np, out_ref_np, rtol=rtol, atol=atol)


def parity_check_attention_past(
    config: AttentionConfig,
    ep,
    device,
    torch_type,
    ort_type,
    causal,
    rtol,
    atol,
    std=0.2,
):
    """
    Parity check for ONNX Attention op in decoding phase (with past KV cache).

    This tests that the ONNX Attention op produces the same output as a PyTorch
    reference implementation for token-by-token decoding with KV cache.
    """
    if ort_type == TensorProto.FLOAT16:
        torch_type = torch.float16
    elif ort_type == TensorProto.BFLOAT16:
        torch_type = torch.bfloat16
    else:
        torch_type = torch.float32
    torch.manual_seed(0)

    # --- Test Data Generation ---
    # Query for new tokens
    q = (
        torch.randn(
            config.batch_size,
            config.q_sequence_length,
            config.q_num_heads,
            config.head_size,
            device=device,
            dtype=torch_type,
        )
        * std
    )

    # Past KV cache in BNSH format
    past_k = (
        torch.randn(
            config.batch_size,
            config.kv_num_heads,
            config.past_kv_sequence_length,
            config.head_size,
            device=device,
            dtype=torch_type,
        )
        * std
    )
    past_v = torch.randn_like(past_k) * std

    # New K/V for current tokens in BSNH format
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

    # --- PyTorch Reference Path ---
    # Concatenate past and new KV for reference
    # past_k is BNSH, new_k is BSNH - need to transpose new_k
    new_k_bnsh = new_k.transpose(1, 2)  # BSNH -> BNSH
    new_v_bnsh = new_v.transpose(1, 2)  # BSNH -> BNSH

    full_k_bnsh = torch.cat([past_k, new_k_bnsh], dim=2)  # [B, N, past+new, H]
    full_v_bnsh = torch.cat([past_v, new_v_bnsh], dim=2)  # [B, N, past+new, H]

    # Convert to BSNH for reference attention
    full_k_bsnh = full_k_bnsh.transpose(1, 2)
    full_v_bsnh = full_v_bnsh.transpose(1, 2)

    total_seq_len = config.past_kv_sequence_length + config.kv_sequence_length

    # --- Create attn_mask as boolean padding mask (simulating seqlens_k) ---
    # For testing, we use full sequence length (no actual padding)
    # attn_mask: [batch, total_seq_len] - True for valid, False for masked
    # GQA kernel converts this to seqlens_k internally
    attn_mask = None
    key_padding_mask = None
    if config.has_attn_mask:
        # All positions are valid (no padding) for this test
        attn_mask = torch.ones(
            config.batch_size,
            total_seq_len,
            device=device,
            dtype=torch.bool,
        )
        # key_padding_mask for reference: all True (all valid)
        key_padding_mask = torch.ones(
            config.batch_size,
            total_seq_len,
            device=device,
            dtype=torch.bool,
        )

    out_ref, _ = attention_ref(
        q=q,
        k=full_k_bsnh,
        v=full_v_bsnh,
        key_padding_mask=key_padding_mask,
        causal=causal,
        softcap=config.softcap,
    )
    out_ref_np = out_ref.to(torch.float32).detach().cpu().numpy()

    # --- ONNX Runtime Path ---
    num_runs = 2 if enable_deterministic_check else 1
    for i in range(num_runs):
        out, present_k, present_v = attention_past_func(
            q=q,
            past_k=past_k,
            past_v=past_v,
            new_k=new_k,
            new_v=new_v,
            config=config,
            attn_mask=attn_mask,
            ep=ep,
            device=device,
            ort_type=ort_type,
        )
        if i == 0:
            first_out = out.clone()
            first_present_k = present_k.clone() if present_k is not None else None
            first_present_v = present_v.clone() if present_v is not None else None
        else:
            torch.testing.assert_close(out, first_out, rtol=0, atol=0, msg="Output mismatch between two runs")
            if present_k is not None:
                torch.testing.assert_close(
                    present_k, first_present_k, rtol=0, atol=0, msg="present_k mismatch between two runs"
                )
            if present_v is not None:
                torch.testing.assert_close(
                    present_v, first_present_v, rtol=0, atol=0, msg="present_v mismatch between two runs"
                )

    out = torch.reshape(out, (config.batch_size, config.q_sequence_length, config.q_num_heads, config.head_size))
    out_np = out.to(torch.float32).detach().cpu().numpy()

    if enable_debug_print:
        print(f"[DEBUG] out_np non-zeros: {numpy.count_nonzero(out_np)} / {out_np.size}")
        print(f"[DEBUG] out_ref_np non-zeros: {numpy.count_nonzero(out_ref_np)} / {out_ref_np.size}")

    if numpy.count_nonzero(out_ref_np) > 0 and numpy.count_nonzero(out_np) == 0:
        raise RuntimeError("Output is all zeros")

    # --- Comparison ---
    # Compare KV cache (present should be concat of past + new)
    full_k_ref_np = full_k_bnsh.to(torch.float32).detach().cpu().numpy()
    full_v_ref_np = full_v_bnsh.to(torch.float32).detach().cpu().numpy()
    present_k_np = present_k.to(torch.float32).detach().cpu().numpy()
    present_v_np = present_v.to(torch.float32).detach().cpu().numpy()

    print_diff_statistics(torch.tensor(present_k_np - full_k_ref_np), "present_k")
    numpy.testing.assert_allclose(present_k_np, full_k_ref_np, rtol=rtol, atol=atol)
    print_diff_statistics(torch.tensor(present_v_np - full_v_ref_np), "present_v")
    numpy.testing.assert_allclose(present_v_np, full_v_ref_np, rtol=rtol, atol=atol)

    print_diff_statistics(torch.tensor(out_np - out_ref_np), "out")
    numpy.testing.assert_allclose(out_np, out_ref_np, rtol=rtol, atol=atol)


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


def parity_check_attention_prompt_with_padding(
    config: AttentionConfig,
    seqlens: torch.Tensor,
    ep,
    device,
    torch_type,
    ort_type,
    rtol,
    atol,
    std=0.2,
):
    """
    Parity check for ONNX Attention op in prompt phase with padding.

    This tests that the ONNX Attention op correctly handles boolean padding masks
    where some batches have shorter valid sequences than others (right-padding).

    Args:
        config: AttentionConfig with model parameters (has_attn_mask should be True).
        seqlens: Tensor of shape [batch_size] containing valid sequence lengths for each batch.
        ep: Execution provider.
        device: Device string.
        torch_type: PyTorch dtype.
        ort_type: ONNX tensor type.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.
        std: Standard deviation for random input generation.
    """
    torch.manual_seed(0)

    # Generate Q, K, V tensors in BSNH format (batch, seq, num_heads, head_size)
    q = (
        torch.randn(
            config.batch_size,
            config.q_sequence_length,
            config.q_num_heads,
            config.head_size,
            device=device,
            dtype=torch_type,
        )
        * std
    )
    k = (
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
    v = torch.randn_like(k) * std

    # Zero out padded positions in K, V for proper comparison
    for b in range(config.batch_size):
        valid_len = seqlens[b].item()
        if valid_len < config.kv_sequence_length:
            k[b, valid_len:, :, :] = 0
            v[b, valid_len:, :, :] = 0

    # Create boolean attention mask based on seqlens
    attn_mask = create_boolean_mask_from_seqlens(
        seqlens=seqlens,
        total_seq_len=config.kv_sequence_length,
        mask_dims=config.attn_mask_dims,
        q_seq_len=config.q_sequence_length,
        num_heads=config.q_num_heads,
        device=device,
    )

    # Create key_padding_mask for reference (always 2D for attention_ref)
    key_padding_mask = create_boolean_mask_from_seqlens(
        seqlens=seqlens,
        total_seq_len=config.kv_sequence_length,
        mask_dims=2,
        device=device,
    )

    # --- PyTorch Reference Path ---
    out_ref, _ = attention_ref(
        q=q,
        k=k,
        v=v,
        key_padding_mask=key_padding_mask,
        causal=config.is_causal == 1,
        softcap=config.softcap,
    )

    # --- ONNX Runtime Path ---
    out, present_k, present_v = attention_prompt_func(
        q=q,
        k=k,
        v=v,
        config=config,
        attn_mask=attn_mask,
        ep=ep,
        device=device,
        ort_type=ort_type,
    )

    out = torch.reshape(out, (config.batch_size, config.q_sequence_length, config.q_num_heads, config.head_size))

    # --- Comparison ---
    # Zero out padded positions in both outputs for fair comparison
    for b in range(config.batch_size):
        valid_len = seqlens[b].item()
        if valid_len < config.q_sequence_length:
            out[b, valid_len:, :, :] = 0
            out_ref[b, valid_len:, :, :] = 0

    out_np = out.to(torch.float32).detach().cpu().numpy()
    out_ref_np = out_ref.to(torch.float32).detach().cpu().numpy()

    print_diff_statistics(torch.tensor(out_np - out_ref_np), "out")
    numpy.testing.assert_allclose(out_np, out_ref_np, rtol=rtol, atol=atol)


def parity_check_attention_past_with_padding(
    config: AttentionConfig,
    past_seqlens: torch.Tensor,
    ep,
    device,
    torch_type,
    ort_type,
    rtol,
    atol,
    std=0.2,
):
    """
    Parity check for ONNX Attention op in decoding phase with padding.

    This tests that the ONNX Attention op correctly handles boolean padding masks
    during token generation with KV cache.

    Args:
        config: AttentionConfig with model parameters (has_attn_mask should be True).
        past_seqlens: Tensor of shape [batch_size] containing valid past sequence lengths.
        ep: Execution provider.
        device: Device string.
        torch_type: PyTorch dtype.
        ort_type: ONNX tensor type.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.
        std: Standard deviation for random input generation.
    """
    torch.manual_seed(0)

    # Generate query for new tokens
    q = (
        torch.randn(
            config.batch_size,
            config.q_sequence_length,
            config.q_num_heads,
            config.head_size,
            device=device,
            dtype=torch_type,
        )
        * std
    )

    # Past KV cache in BNSH format
    past_k = (
        torch.randn(
            config.batch_size,
            config.kv_num_heads,
            config.past_kv_sequence_length,
            config.head_size,
            device=device,
            dtype=torch_type,
        )
        * std
    )
    past_v = torch.randn_like(past_k) * std

    # Zero out padded positions in past KV cache
    for b in range(config.batch_size):
        valid_past_len = past_seqlens[b].item()
        if valid_past_len < config.past_kv_sequence_length:
            past_k[b, :, valid_past_len:, :] = 0
            past_v[b, :, valid_past_len:, :] = 0

    # New K/V for current tokens in BSNH format
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

    # Total sequence lengths = past_seqlens + new_seq_len
    total_seqlens = past_seqlens + config.kv_sequence_length
    total_seq_len = config.past_kv_sequence_length + config.kv_sequence_length

    # Create boolean attention mask based on total seqlens
    attn_mask = create_boolean_mask_from_seqlens(
        seqlens=total_seqlens,
        total_seq_len=total_seq_len,
        mask_dims=config.attn_mask_dims,
        q_seq_len=config.q_sequence_length,
        num_heads=config.q_num_heads,
        device=device,
    )

    # Create key_padding_mask for reference (always 2D)
    key_padding_mask = create_boolean_mask_from_seqlens(
        seqlens=total_seqlens,
        total_seq_len=total_seq_len,
        mask_dims=2,
        device=device,
    )

    # --- PyTorch Reference Path ---
    # Concatenate past and new KV
    new_k_bnsh = new_k.transpose(1, 2)
    new_v_bnsh = new_v.transpose(1, 2)
    full_k_bnsh = torch.cat([past_k, new_k_bnsh], dim=2)
    full_v_bnsh = torch.cat([past_v, new_v_bnsh], dim=2)
    full_k_bsnh = full_k_bnsh.transpose(1, 2)
    full_v_bsnh = full_v_bnsh.transpose(1, 2)

    out_ref, _ = attention_ref(
        q=q,
        k=full_k_bsnh,
        v=full_v_bsnh,
        key_padding_mask=key_padding_mask,
        causal=config.is_causal == 1,
        softcap=config.softcap,
    )

    # --- ONNX Runtime Path ---
    out, present_k, present_v = attention_past_func(
        q=q,
        past_k=past_k,
        past_v=past_v,
        new_k=new_k,
        new_v=new_v,
        config=config,
        attn_mask=attn_mask,
        ep=ep,
        device=device,
        ort_type=ort_type,
    )

    out = torch.reshape(out, (config.batch_size, config.q_sequence_length, config.q_num_heads, config.head_size))

    # --- Comparison ---
    out_np = out.to(torch.float32).detach().cpu().numpy()
    out_ref_np = out_ref.to(torch.float32).detach().cpu().numpy()

    print_diff_statistics(torch.tensor(out_np - out_ref_np), "out")
    numpy.testing.assert_allclose(out_np, out_ref_np, rtol=rtol, atol=atol)


# #################################################################################################
#  Test Case Generators
# #################################################################################################


def attention_prompt_test_cases():
    """
    Generate test cases for ONNX Attention op in prompt phase.

    The ONNX Attention op (opset 23) supports:
    - GQA (kv_num_heads != q_num_heads)
    - MHA (kv_num_heads == q_num_heads)
    - Causal attention via is_causal attribute
    - softcap

    It does NOT support (handled by external ops):
    - Rotary embeddings
    - Smooth softmax / head_sink
    - Local window attention
    - Packed QKV
    """
    batches = [1, 2, 3]
    seqs = [(16, 16), (64, 64), (128, 128)]
    # GQA head configurations only (kv_heads != q_heads)
    heads = [(8, 2), (8, 4)]  # (q_heads, kv_heads)
    h_sizes = [128] if quick_build else [64, 128]
    softcap_opts = [0.0]  # softcap not yet supported in CUDA implementation

    h_sizes_to_test = h_sizes[:1] if pipeline_mode else h_sizes

    combo_index = 0
    for h in h_sizes_to_test:
        for b in batches[:2] if pipeline_mode else batches:
            for sq, skv in seqs[:2] if pipeline_mode else seqs:
                for n, n2 in heads:
                    softcap = softcap_opts[combo_index % len(softcap_opts)]
                    combo_index += 1

                    config = AttentionConfig(
                        batch_size=b,
                        q_sequence_length=sq,
                        kv_sequence_length=skv,
                        past_kv_sequence_length=0,
                        q_num_heads=n,
                        kv_num_heads=n2,
                        head_size=h,
                        is_causal=1,  # Causal attention
                        softcap=softcap,
                    )
                    name = f"b{b}_sq{sq}_skv{skv}_nh{n}_{n2}_h{h}_sc{softcap}"
                    yield name, config


def attention_past_test_cases():
    """
    Generate test cases for ONNX Attention op in decoding phase (with past KV cache).
    """
    batches = [1, 2]
    # (new_seq_len, past_seq_len)
    seqs = [(1, 32), (1, 128), (1, 512)]
    # GQA head configurations only (kv_heads != q_heads)
    heads = [(8, 2), (8, 4)]  # (q_heads, kv_heads)
    h_sizes = [128] if quick_build else [64, 128]
    softcap_opts = [0.0]

    h_sizes_to_test = h_sizes[:1] if pipeline_mode else h_sizes

    combo_index = 0
    for h in h_sizes_to_test:
        for b in batches[:1] if pipeline_mode else batches:
            for s, s2 in seqs[:2] if pipeline_mode else seqs:
                for n, n2 in heads:
                    softcap = softcap_opts[combo_index % len(softcap_opts)]
                    combo_index += 1

                    config = AttentionConfig(
                        batch_size=b,
                        q_sequence_length=s,
                        kv_sequence_length=s,  # new K/V has same length as Q
                        past_kv_sequence_length=s2,
                        q_num_heads=n,
                        kv_num_heads=n2,
                        head_size=h,
                        is_causal=1,  # Causal attention
                        softcap=softcap,
                    )
                    name = f"b{b}_s{s}_past{s2}_nh{n}_{n2}_h{h}_sc{softcap}"
                    yield name, config


def attention_prompt_padding_test_cases():
    """
    Generate test cases for ONNX Attention op with boolean padding masks.

    Tests 2D, 3D, and 4D boolean masks for right-padding scenarios.
    ONNX broadcasting aligns from the right:
    - 2D [batch, total_seq] broadcasts to [batch, 1, 1, total_seq]
    - 3D [num_heads, q_seq, total_seq] broadcasts to [1, num_heads, q_seq, total_seq]
    - 4D [batch, num_heads, q_seq, total_seq] - no broadcasting

    Note: 3D mask tests use uniform seqlens since 3D broadcasts across batches.
    """
    # Test configurations
    batches = [2]  # Need multiple batches to test different padding per batch
    seqs = [(16, 16)]  # (q_seq_len, kv_seq_len)
    heads = [(8, 2)]  # (q_heads, kv_heads)
    h_sizes = [128]
    # Test 2D, 3D, and 4D masks
    mask_dims_options = [2, 3, 4]

    for h in h_sizes:
        for b in batches:
            for sq, skv in seqs:
                for n, n2 in heads:
                    for mask_dims in mask_dims_options:
                        config = AttentionConfig(
                            batch_size=b,
                            q_sequence_length=sq,
                            kv_sequence_length=skv,
                            past_kv_sequence_length=0,
                            q_num_heads=n,
                            kv_num_heads=n2,
                            head_size=h,
                            is_causal=1,
                            has_attn_mask=True,
                            attn_mask_dims=mask_dims,
                        )
                        name = f"b{b}_sq{sq}_skv{skv}_nh{n}_{n2}_h{h}_mask{mask_dims}d"
                        yield name, config


def attention_past_padding_test_cases():
    """
    Generate test cases for ONNX Attention op with boolean padding masks in decoding phase.

    Note: Past/decoding phase with per-batch variable padding is complex because
    the ONNX Attention op expects uniform past_sequence_length across all batches.
    These tests use the full past sequence length for all batches (no per-batch variation).

    ONNX broadcasting for 3D masks: [num_heads, q_seq, total_seq] -> [1, num_heads, q_seq, total_seq]
    """
    batches = [2]
    seqs = [(1, 32)]  # (new_seq_len, past_seq_len)
    heads = [(8, 2)]
    h_sizes = [128]
    # Test 2D, 3D, and 4D masks
    mask_dims_options = [2, 3, 4]

    for h in h_sizes:
        for b in batches:
            for s, s2 in seqs:
                for n, n2 in heads:
                    for mask_dims in mask_dims_options:
                        config = AttentionConfig(
                            batch_size=b,
                            q_sequence_length=s,
                            kv_sequence_length=s,
                            past_kv_sequence_length=s2,
                            q_num_heads=n,
                            kv_num_heads=n2,
                            head_size=h,
                            is_causal=1,
                            has_attn_mask=True,
                            attn_mask_dims=mask_dims,
                        )
                        name = f"b{b}_s{s}_past{s2}_nh{n}_{n2}_h{h}_mask{mask_dims}d"
                        yield name, config


# #################################################################################################
#  Unit Test Classes
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


rtol = {"fp16": 5e-3, "bf16": 5e-2}
atol = {"fp16": 5e-3, "bf16": 1e-2}


@unittest.skipIf(not has_flash_attention(), "Flash Attention is not available, skipping tests.")
class TestONNXAttentionFlashGQA(unittest.TestCase):
    """Test ONNX Attention op (opset 23) GQA path with Flash Attention."""

    @parameterized.expand(attention_prompt_test_cases())
    def test_attention_prompt_flash(self, name, config):
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "0"
        parity_check_attention_prompt(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            ort_type=TensorProto.FLOAT16,
            causal=True,
            rtol=rtol["fp16"],
            atol=atol["fp16"],
        )

    @parameterized.expand(attention_past_test_cases())
    def test_attention_past_flash(self, name, config):
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "0"
        parity_check_attention_past(
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
class TestONNXAttentionFlashGQABF16(unittest.TestCase):
    """Test ONNX Attention op (opset 23) GQA path with Flash Attention using BFloat16."""

    @parameterized.expand(attention_prompt_test_cases())
    def test_attention_prompt_flash_bf16(self, name, config):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BFloat16 not supported on this device")

        config.kv_cache_type = "bfloat16"
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "0"
        parity_check_attention_prompt(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.bfloat16,
            ort_type=TensorProto.BFLOAT16,
            causal=True,
            rtol=rtol["bf16"],
            atol=atol["bf16"],
        )

    @parameterized.expand(attention_past_test_cases())
    def test_attention_past_flash_bf16(self, name, config):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BFloat16 not supported on this device")

        config.kv_cache_type = "bfloat16"
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "0"
        parity_check_attention_past(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.bfloat16,
            ort_type=TensorProto.BFLOAT16,
            causal=True,
            rtol=rtol["bf16"],
            atol=atol["bf16"],
        )


@unittest.skipIf(not has_cuda_device(53), "Memory Efficient Attention is not available, skipping tests.")
class TestONNXAttentionMemoryEfficientGQA(unittest.TestCase):
    """Test ONNX Attention op (opset 23) GQA path with Memory Efficient Attention."""

    @parameterized.expand(attention_prompt_test_cases())
    def test_attention_prompt_memory_efficient(self, name, config):
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "1"
        parity_check_attention_prompt(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            ort_type=TensorProto.FLOAT16,
            causal=True,
            rtol=rtol["fp16"],
            atol=atol["fp16"],
        )

    @parameterized.expand(attention_past_test_cases())
    def test_attention_past_memory_efficient(self, name, config):
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "1"
        parity_check_attention_past(
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
class TestONNXAttentionMemoryEfficientGQABF16(unittest.TestCase):
    """Test ONNX Attention op (opset 23) GQA path with Memory Efficient Attention using BFloat16."""

    @parameterized.expand(attention_past_test_cases())
    def test_attention_past_memory_efficient_bf16(self, name, config):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BFloat16 not supported on this device")

        config.kv_cache_type = "bfloat16"
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "1"
        parity_check_attention_past(
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
class TestONNXAttentionPaddingMask(unittest.TestCase):
    """
    Test ONNX Attention op (opset 23) with boolean padding masks.

    These tests verify that the boolean attn_mask is correctly converted to
    sequence lengths on GPU and that the attention computation respects the
    padding. Tests cover 2D, 3D, and 4D mask shapes.
    """

    @parameterized.expand(attention_prompt_padding_test_cases())
    def test_attention_prompt_padding_flash(self, name, config):
        """Test prompt phase with padding mask using Flash Attention."""
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "0"

        # Create sequence lengths: first batch has shorter sequence
        # e.g., for batch_size=2, kv_seq_len=16: seqlens = [10, 16]
        seqlens = torch.tensor(
            [config.kv_sequence_length - 6, config.kv_sequence_length],
            dtype=torch.int32,
            device="cuda",
        )

        parity_check_attention_prompt_with_padding(
            config=config,
            seqlens=seqlens,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            ort_type=TensorProto.FLOAT16,
            rtol=rtol["fp16"],
            atol=atol["fp16"],
        )

    @parameterized.expand(attention_past_padding_test_cases())
    def test_attention_past_padding_flash(self, name, config):
        """Test decoding phase with padding mask using Flash Attention."""
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "0"

        # For past/decoding tests, use uniform past sequence length across all batches
        # (per-batch variable past length is complex with ONNX Attention's fixed-shape past tensors)
        past_seqlens = torch.full(
            (config.batch_size,),
            config.past_kv_sequence_length,
            dtype=torch.int32,
            device="cuda",
        )

        parity_check_attention_past_with_padding(
            config=config,
            past_seqlens=past_seqlens,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            ort_type=TensorProto.FLOAT16,
            rtol=rtol["fp16"],
            atol=atol["fp16"],
        )


@unittest.skipIf(not has_cuda_device(53), "Memory Efficient Attention is not available, skipping tests.")
class TestONNXAttentionPaddingMaskMemoryEfficient(unittest.TestCase):
    """
    Test ONNX Attention op (opset 23) with boolean padding masks using Memory Efficient Attention.

    These tests verify that the boolean attn_mask is correctly converted to
    sequence lengths on GPU and that the attention computation respects the
    padding. Tests cover 2D, 3D, and 4D mask shapes.
    """

    @parameterized.expand(attention_prompt_padding_test_cases())
    def test_attention_prompt_padding_mea(self, name, config):
        """Test prompt phase with padding mask using Memory Efficient Attention."""
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "1"

        # Create sequence lengths: first batch has shorter sequence
        seqlens = torch.tensor(
            [config.kv_sequence_length - 6, config.kv_sequence_length],
            dtype=torch.int32,
            device="cuda",
        )

        parity_check_attention_prompt_with_padding(
            config=config,
            seqlens=seqlens,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            ort_type=TensorProto.FLOAT16,
            rtol=rtol["fp16"],
            atol=atol["fp16"],
        )

    @parameterized.expand(attention_past_padding_test_cases())
    def test_attention_past_padding_mea(self, name, config):
        """Test decoding phase with padding mask using Memory Efficient Attention."""
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "1"

        # For past/decoding tests, use uniform past sequence length across all batches
        past_seqlens = torch.full(
            (config.batch_size,),
            config.past_kv_sequence_length,
            dtype=torch.int32,
            device="cuda",
        )

        parity_check_attention_past_with_padding(
            config=config,
            past_seqlens=past_seqlens,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            ort_type=TensorProto.FLOAT16,
            rtol=rtol["fp16"],
            atol=atol["fp16"],
        )


if __name__ == "__main__":
    unittest.main()
