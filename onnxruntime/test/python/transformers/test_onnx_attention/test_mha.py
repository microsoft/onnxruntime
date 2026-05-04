# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Tests for ONNX Attention op (opset 23+) — MHA path (kv_num_heads == q_num_heads).

The MHA path in attention.cc is exercised when kv_num_heads == q_num_heads.
On Ampere+ GPUs with fp16/bf16, the dispatch cascade routes to flash attention
or memory efficient attention first. The unfused kernel handles fp32 and
cases where flash/MEA are ineligible. Tests below marked "unfused" explicitly
disable flash and MEA to verify the unfused kernel.

Supports:
  - float32, float16, bfloat16
  - 3D inputs (BSNH format) and 4D inputs (BNSH format)
  - Causal and non-causal attention
  - Self-attention and cross-attention (kv_seq != q_seq)
  - Additive attention bias (and boolean masks converted to additive bias)
  - Past KV cache
  - 2D, 3D, 4D additive/bool masks with broadcasting
"""

import os
import unittest
from unittest.mock import patch

import numpy
import torch
from onnx import TensorProto
from parameterized import parameterized

from test_onnx_attention.common import (
    AttentionConfig,
    atol,
    attention_past_func,
    attention_prompt_func,
    attention_ref,
    create_additive_mask_from_seqlens,
    create_boolean_mask_from_seqlens,
    enable_deterministic_check,
    has_cuda_device,
    pipeline_mode,
    print_diff_statistics,
    quick_build,
    rtol,
)

# #################################################################################################
#  MHA Parity Check Functions
# #################################################################################################


def parity_check_mha_prompt(
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
    Parity check for ONNX Attention op MHA path in prompt phase (no past KV cache).

    Tests self-attention and cross-attention (when q_seq != kv_seq).
    """
    torch.manual_seed(0)

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

    # MHA path uses additive attention bias, not boolean masks
    attn_mask = None
    attn_bias_ref = None
    if config.has_attn_mask:
        # Create additive mask (0 for valid, -inf for masked)
        # For prompt without padding, create a causal-style or zero mask
        seqlens = torch.full((config.batch_size,), config.kv_sequence_length, dtype=torch.int32, device=device)
        attn_mask = create_additive_mask_from_seqlens(
            seqlens=seqlens,
            total_seq_len=config.kv_sequence_length,
            mask_dims=config.attn_mask_dims,
            q_seq_len=config.q_sequence_length,
            num_heads=config.q_num_heads,
            device=device,
            dtype=torch_type,
        )
        # For reference: expand to 4D [batch, heads, q_seq, kv_seq]
        if config.attn_mask_dims == 2:
            attn_bias_ref = attn_mask.unsqueeze(0).unsqueeze(0).expand(config.batch_size, config.q_num_heads, -1, -1)
        elif config.attn_mask_dims == 3:
            # 3D [heads, q_seq, total_seq]: batch broadcasts
            attn_bias_ref = attn_mask.unsqueeze(0).expand(config.batch_size, -1, -1, -1)
        else:
            attn_bias_ref = attn_mask

    # --- PyTorch Reference Path ---
    out_ref, _ = attention_ref(
        q=q,
        k=k,
        v=v,
        attn_bias=attn_bias_ref,
        causal=causal,
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
        else:
            torch.testing.assert_close(out, first_out, rtol=0, atol=0, msg="Output mismatch between two runs")

    out = torch.reshape(out, (config.batch_size, config.q_sequence_length, config.q_num_heads, config.head_size))
    out_np = out.to(torch.float32).detach().cpu().numpy()

    # --- Comparison ---
    # Check KV cache correctness
    k_ref_bnsh = k.transpose(1, 2)
    v_ref_bnsh = v.transpose(1, 2)

    present_k_np = present_k.to(torch.float32).detach().cpu().numpy()
    present_v_np = present_v.to(torch.float32).detach().cpu().numpy()
    k_ref_np = k_ref_bnsh.to(torch.float32).detach().cpu().numpy()
    v_ref_np = v_ref_bnsh.to(torch.float32).detach().cpu().numpy()

    print_diff_statistics(torch.tensor(present_k_np - k_ref_np), "present_k")
    numpy.testing.assert_allclose(present_k_np, k_ref_np, rtol=rtol, atol=atol)
    print_diff_statistics(torch.tensor(present_v_np - v_ref_np), "present_v")
    numpy.testing.assert_allclose(present_v_np, v_ref_np, rtol=rtol, atol=atol)

    print_diff_statistics(torch.tensor(out_np - out_ref_np), "out")
    numpy.testing.assert_allclose(out_np, out_ref_np, rtol=rtol, atol=atol)


def parity_check_mha_past(
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
    Parity check for ONNX Attention op MHA path in decoding phase (with past KV cache).
    """
    torch.manual_seed(0)

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
        causal=causal,
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
            attn_mask=None,
            ep=ep,
            device=device,
            ort_type=ort_type,
        )
        if i == 0:
            first_out = out.clone()
        else:
            torch.testing.assert_close(out, first_out, rtol=0, atol=0, msg="Output mismatch between two runs")

    out = torch.reshape(out, (config.batch_size, config.q_sequence_length, config.q_num_heads, config.head_size))
    out_np = out.to(torch.float32).detach().cpu().numpy()

    # --- Comparison ---
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


def parity_check_mha_prompt_with_attn_bias(
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
    Parity check for ONNX Attention op MHA path with additive attention bias.

    Tests that additive masks (0 for valid, -inf for masked) are correctly
    applied with broadcasting for 2D, 3D, and 4D shapes.
    """
    torch.manual_seed(0)

    # Compute effective per-batch seqlens based on mask broadcasting.
    # For 2D mask, all batches see the same mask (first batch's pattern).
    effective_seqlens = seqlens.clone()
    if config.attn_mask_dims == 2:
        effective_seqlens[:] = seqlens[0]

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

    # Zero out padded positions in K, V based on effective seqlens
    for b in range(config.batch_size):
        valid_len = effective_seqlens[b].item()
        if valid_len < config.kv_sequence_length:
            k[b, valid_len:, :, :] = 0
            v[b, valid_len:, :, :] = 0

    # Create additive attention mask
    attn_mask = create_additive_mask_from_seqlens(
        seqlens=seqlens,
        total_seq_len=config.kv_sequence_length,
        mask_dims=config.attn_mask_dims,
        q_seq_len=config.q_sequence_length,
        num_heads=config.q_num_heads,
        device=device,
        dtype=torch_type,
    )

    # Create 4D reference bias by broadcasting the reduced mask the same way ORT does.
    # This ensures the reference matches the actual broadcasting behavior.
    # 2D [q_seq, total_seq] → [1, 1, q_seq, total_seq] → [batch, heads, q_seq, total_seq]
    # 3D [heads, q_seq, total_seq] → [1, heads, q_seq, total_seq] → [batch, heads, q_seq, total_seq]
    # 4D [batch, heads, q_seq, total_seq] → as-is
    if config.attn_mask_dims == 2:
        attn_bias_ref = (
            attn_mask.unsqueeze(0).unsqueeze(0).expand(config.batch_size, config.q_num_heads, -1, -1).contiguous()
        )
    elif config.attn_mask_dims == 3:
        # 3D [heads, q_seq, total_seq]: batch broadcasts, heads per-head
        attn_bias_ref = attn_mask.unsqueeze(0).expand(config.batch_size, -1, -1, -1).contiguous()
    else:
        attn_bias_ref = attn_mask

    # --- PyTorch Reference Path ---
    out_ref, _ = attention_ref(
        q=q,
        k=k,
        v=v,
        attn_bias=attn_bias_ref,
        causal=config.is_causal == 1,
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
    # Zero out padded positions in both outputs based on effective seqlens
    for b in range(config.batch_size):
        valid_len = effective_seqlens[b].item()
        if valid_len < config.q_sequence_length:
            out[b, valid_len:, :, :] = 0
            out_ref[b, valid_len:, :, :] = 0

    out_np = out.to(torch.float32).detach().cpu().numpy()
    out_ref_np = out_ref.to(torch.float32).detach().cpu().numpy()

    print_diff_statistics(torch.tensor(out_np - out_ref_np), "out")
    numpy.testing.assert_allclose(out_np, out_ref_np, rtol=rtol, atol=atol)


# #################################################################################################
#  Test Case Generators
# #################################################################################################


def mha_prompt_test_cases():
    """
    Generate test cases for MHA path — prompt (prefill) phase.

    Practical LLM scenarios: causal self-attention for decoder models.
    """
    batches = [1, 2, 3]
    seqs = [(16, 16), (64, 64), (128, 128)]
    heads = [(8, 8), (4, 4)]  # MHA: q_heads == kv_heads
    h_sizes = [128] if quick_build else [64, 128]

    h_sizes_to_test = h_sizes[:1] if pipeline_mode else h_sizes

    for h in h_sizes_to_test:
        for b in batches[:2] if pipeline_mode else batches:
            for sq, skv in seqs[:2] if pipeline_mode else seqs:
                for n, n2 in heads[:1] if pipeline_mode else heads:
                    config = AttentionConfig(
                        batch_size=b,
                        q_sequence_length=sq,
                        kv_sequence_length=skv,
                        past_kv_sequence_length=0,
                        q_num_heads=n,
                        kv_num_heads=n2,
                        head_size=h,
                        is_causal=1,
                        attn_mask_type="additive",
                    )
                    name = f"b{b}_sq{sq}_skv{skv}_nh{n}_h{h}_causal"
                    yield name, config


def mha_prompt_noncausal_test_cases():
    """
    Generate test cases for MHA path — non-causal prompt phase.

    Practical LLM scenarios: encoder models (BERT-style), non-causal attention.
    """
    batches = [1, 2]
    seqs = [(16, 16), (64, 64)]
    heads = [(8, 8)]
    h_sizes = [128] if quick_build else [64, 128]

    h_sizes_to_test = h_sizes[:1] if pipeline_mode else h_sizes

    for h in h_sizes_to_test:
        for b in batches[:1] if pipeline_mode else batches:
            for sq, skv in seqs[:1] if pipeline_mode else seqs:
                for n, n2 in heads:
                    config = AttentionConfig(
                        batch_size=b,
                        q_sequence_length=sq,
                        kv_sequence_length=skv,
                        past_kv_sequence_length=0,
                        q_num_heads=n,
                        kv_num_heads=n2,
                        head_size=h,
                        is_causal=0,
                        attn_mask_type="additive",
                    )
                    name = f"b{b}_sq{sq}_skv{skv}_nh{n}_h{h}_noncausal"
                    yield name, config


def mha_cross_attention_test_cases():
    """
    Generate test cases for MHA path — cross-attention.

    Practical LLM scenarios: encoder-decoder models where q_seq != kv_seq.
    """
    batches = [1, 2]
    # (q_seq_len, kv_seq_len) — different lengths for cross-attention
    seqs = [(1, 64), (16, 128), (32, 64)]
    heads = [(8, 8)]
    h_sizes = [128] if quick_build else [64, 128]

    h_sizes_to_test = h_sizes[:1] if pipeline_mode else h_sizes

    for h in h_sizes_to_test:
        for b in batches[:1] if pipeline_mode else batches:
            for sq, skv in seqs[:2] if pipeline_mode else seqs:
                for n, n2 in heads:
                    config = AttentionConfig(
                        batch_size=b,
                        q_sequence_length=sq,
                        kv_sequence_length=skv,
                        past_kv_sequence_length=0,
                        q_num_heads=n,
                        kv_num_heads=n2,
                        head_size=h,
                        is_causal=0,
                        attn_mask_type="additive",
                    )
                    name = f"b{b}_sq{sq}_skv{skv}_nh{n}_h{h}_cross"
                    yield name, config


def mha_past_test_cases():
    """
    Generate test cases for MHA path — decoding with KV cache.

    Practical LLM scenarios: autoregressive token generation.
    """
    batches = [1, 2]
    # (new_seq_len, past_seq_len)
    seqs = [(1, 32), (1, 128), (1, 512)]
    heads = [(8, 8), (4, 4)]
    h_sizes = [128] if quick_build else [64, 128]

    h_sizes_to_test = h_sizes[:1] if pipeline_mode else h_sizes

    for h in h_sizes_to_test:
        for b in batches[:1] if pipeline_mode else batches:
            for s, s2 in seqs[:2] if pipeline_mode else seqs:
                for n, n2 in heads[:1] if pipeline_mode else heads:
                    config = AttentionConfig(
                        batch_size=b,
                        q_sequence_length=s,
                        kv_sequence_length=s,
                        past_kv_sequence_length=s2,
                        q_num_heads=n,
                        kv_num_heads=n2,
                        head_size=h,
                        is_causal=1,
                        attn_mask_type="additive",
                    )
                    name = f"b{b}_s{s}_past{s2}_nh{n}_h{h}"
                    yield name, config


def mha_attn_bias_test_cases():
    """
    Generate test cases for MHA path with additive attention bias.

    Tests 2D, 3D, and 4D additive masks with padding simulation.
    """
    batches = [2]
    seqs = [(16, 16)]
    heads = [(8, 8)]
    h_sizes = [128]
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
                            is_causal=0,
                            has_attn_mask=True,
                            attn_mask_dims=mask_dims,
                            attn_mask_type="additive",
                        )
                        name = f"b{b}_sq{sq}_skv{skv}_nh{n}_h{h}_bias{mask_dims}d"
                        yield name, config


def mha_bool_mask_test_cases():
    """
    Generate test cases for MHA path with boolean attention mask.

    Tests 2D, 3D, and 4D boolean masks for right-padding scenarios.
    The MHA path in attention.cc converts bool masks to additive bias
    (True -> 0.0, False -> mask_filter_value).

    For the MHA path, ONNX right-aligned broadcasting maps:
      2D [q_seq, total_seq] → [1, 1, q_seq, total_seq] (all batches share one mask)
      3D [heads, q_seq, total_seq] → [1, heads, q_seq, total_seq]
      4D [batch, heads, q_seq, total_seq] → per-batch, per-head masks
    """
    batches = [2]
    seqs = [(16, 16)]
    heads = [(8, 8)]
    h_sizes = [128]
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
                            is_causal=0,
                            has_attn_mask=True,
                            attn_mask_dims=mask_dims,
                            attn_mask_type="bool",
                        )
                        name = f"b{b}_sq{sq}_skv{skv}_nh{n}_h{h}_bool{mask_dims}d"
                        yield name, config


def parity_check_mha_prompt_with_bool_mask(
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
    Parity check for ONNX Attention op MHA path with boolean attention mask.

    The MHA path converts bool masks to additive bias (True -> 0.0, False -> -inf).
    Tests 2D, 3D, and 4D boolean masks with padding simulation.
    """
    torch.manual_seed(0)

    # Compute effective per-batch seqlens based on mask broadcasting.
    # For 2D bool mask [q_seq, total_seq]: all batches share the same mask (first batch's pattern).
    # For 3D bool mask [heads, q_seq, total_seq]: batch broadcasts, use first batch's pattern.
    # For 4D bool mask [batch, heads, q_seq, total_seq]: per-batch seqlens apply directly.
    effective_seqlens = seqlens.clone()
    if config.attn_mask_dims in (2, 3):
        effective_seqlens[:] = seqlens[0]

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

    # Zero out padded positions in K, V based on effective seqlens
    for b in range(config.batch_size):
        valid_len = effective_seqlens[b].item()
        if valid_len < config.kv_sequence_length:
            k[b, valid_len:, :, :] = 0
            v[b, valid_len:, :, :] = 0

    # Create boolean mask for ORT.
    # For the MHA path, 2D bool mask shape is [q_seq, total_seq] per ONNX broadcasting rules,
    # so we build it from the first batch's seqlen (all batches share the same mask).
    if config.attn_mask_dims == 2:
        # 2D: [q_seq, total_seq] — single mask pattern for all batches
        arange = torch.arange(config.kv_sequence_length, device=device)
        mask_1d = arange < seqlens[0]  # [total_seq]
        attn_mask = mask_1d.unsqueeze(0).expand(config.q_sequence_length, -1).contiguous()  # [q_seq, total_seq]
    else:
        attn_mask = create_boolean_mask_from_seqlens(
            seqlens=seqlens,
            total_seq_len=config.kv_sequence_length,
            mask_dims=config.attn_mask_dims,
            q_seq_len=config.q_sequence_length,
            num_heads=config.q_num_heads,
            device=device,
        )

    # Per-batch key_padding_mask [batch, kv_seq] for reference.
    # Must NOT use create_boolean_mask_from_seqlens(..., mask_dims=2) here because that
    # returns [q_seq, total_seq] using only the first batch's seqlen, which is wrong
    # when effective_seqlens vary per batch (4D mask case).
    arange_kv = torch.arange(config.kv_sequence_length, device=device).unsqueeze(0)
    key_padding_mask = arange_kv < effective_seqlens.unsqueeze(1)  # [batch, kv_seq]

    # --- PyTorch Reference Path ---
    out_ref, _ = attention_ref(
        q=q,
        k=k,
        v=v,
        key_padding_mask=key_padding_mask,
        causal=config.is_causal == 1,
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
    # Zero out padded positions in both outputs based on effective seqlens
    for b in range(config.batch_size):
        valid_len = effective_seqlens[b].item()
        if valid_len < config.q_sequence_length:
            out[b, valid_len:, :, :] = 0
            out_ref[b, valid_len:, :, :] = 0

    out_np = out.to(torch.float32).detach().cpu().numpy()
    out_ref_np = out_ref.to(torch.float32).detach().cpu().numpy()

    print_diff_statistics(torch.tensor(out_np - out_ref_np), "out")
    numpy.testing.assert_allclose(out_np, out_ref_np, rtol=rtol, atol=atol)


# #################################################################################################
#  Unit Test Classes
# #################################################################################################


@unittest.skipIf(not has_cuda_device(53), "CUDA device not available, skipping MHA tests.")
class TestONNXAttentionMHAPromptFP16(unittest.TestCase):
    """Test ONNX Attention op MHA path — causal self-attention prompt, float16."""

    @parameterized.expand(mha_prompt_test_cases())
    def test_mha_prompt_fp16(self, name, config):
        parity_check_mha_prompt(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            ort_type=TensorProto.FLOAT16,
            causal=True,
            rtol=rtol["fp16"],
            atol=atol["fp16"],
        )


@unittest.skipIf(not has_cuda_device(53), "CUDA device not available, skipping MHA tests.")
class TestONNXAttentionMHAPromptFP32(unittest.TestCase):
    """Test ONNX Attention op MHA path — causal self-attention prompt, float32."""

    @parameterized.expand(mha_prompt_test_cases())
    def test_mha_prompt_fp32(self, name, config):
        config.kv_cache_type = "float32"
        parity_check_mha_prompt(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float32,
            ort_type=TensorProto.FLOAT,
            causal=True,
            rtol=rtol["fp32"],
            atol=atol["fp32"],
        )


@unittest.skipIf(not has_cuda_device(80), "BF16 requires Ampere or higher GPU, skipping tests.")
class TestONNXAttentionMHAPromptBF16(unittest.TestCase):
    """Test ONNX Attention op MHA path — causal self-attention prompt, bfloat16."""

    @parameterized.expand(mha_prompt_test_cases())
    def test_mha_prompt_bf16(self, name, config):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BFloat16 not supported on this device")

        config.kv_cache_type = "bfloat16"
        parity_check_mha_prompt(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.bfloat16,
            ort_type=TensorProto.BFLOAT16,
            causal=True,
            rtol=rtol["bf16"],
            atol=atol["bf16"],
        )


@unittest.skipIf(not has_cuda_device(53), "CUDA device not available, skipping MHA tests.")
class TestONNXAttentionMHANonCausal(unittest.TestCase):
    """Test ONNX Attention op MHA path — non-causal self-attention (encoder)."""

    @parameterized.expand(mha_prompt_noncausal_test_cases())
    def test_mha_prompt_noncausal_fp16(self, name, config):
        parity_check_mha_prompt(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            ort_type=TensorProto.FLOAT16,
            causal=False,
            rtol=rtol["fp16"],
            atol=atol["fp16"],
        )


@unittest.skipIf(not has_cuda_device(53), "CUDA device not available, skipping MHA tests.")
class TestONNXAttentionMHACrossAttention(unittest.TestCase):
    """Test ONNX Attention op MHA path — cross-attention (encoder-decoder)."""

    @parameterized.expand(mha_cross_attention_test_cases())
    def test_mha_cross_attention_fp16(self, name, config):
        parity_check_mha_prompt(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            ort_type=TensorProto.FLOAT16,
            causal=False,
            rtol=rtol["fp16"],
            atol=atol["fp16"],
        )


@unittest.skipIf(not has_cuda_device(53), "CUDA device not available, skipping MHA tests.")
class TestONNXAttentionMHAPastFP16(unittest.TestCase):
    """Test ONNX Attention op MHA path — decoding with KV cache, float16."""

    @parameterized.expand(mha_past_test_cases())
    def test_mha_past_fp16(self, name, config):
        parity_check_mha_past(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            ort_type=TensorProto.FLOAT16,
            causal=True,
            rtol=rtol["fp16"],
            atol=atol["fp16"],
        )


@unittest.skipIf(not has_cuda_device(53), "CUDA device not available, skipping MHA tests.")
class TestONNXAttentionMHAPastFP32(unittest.TestCase):
    """Test ONNX Attention op MHA path — decoding with KV cache, float32."""

    @parameterized.expand(mha_past_test_cases())
    def test_mha_past_fp32(self, name, config):
        config.kv_cache_type = "float32"
        parity_check_mha_past(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float32,
            ort_type=TensorProto.FLOAT,
            causal=True,
            rtol=rtol["fp32"],
            atol=atol["fp32"],
        )


@unittest.skipIf(not has_cuda_device(53), "CUDA device not available, skipping MHA tests.")
class TestONNXAttentionMHAAttnBias(unittest.TestCase):
    """
    Test ONNX Attention op MHA path with additive attention bias.

    Tests 2D, 3D, and 4D additive masks that are used to simulate padding
    or custom attention patterns. This exercises the broadcast_attn_bias
    logic in attention.cc.
    """

    @parameterized.expand(mha_attn_bias_test_cases())
    def test_mha_attn_bias_fp16(self, name, config):
        seqlens = torch.tensor(
            [config.kv_sequence_length - 6, config.kv_sequence_length],
            dtype=torch.int32,
            device="cuda",
        )

        parity_check_mha_prompt_with_attn_bias(
            config=config,
            seqlens=seqlens,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            ort_type=TensorProto.FLOAT16,
            rtol=rtol["fp16"],
            atol=atol["fp16"],
        )


@unittest.skipIf(not has_cuda_device(53), "CUDA device not available, skipping MHA tests.")
class TestONNXAttentionMHABoolMask(unittest.TestCase):
    """
    Test ONNX Attention op MHA path with boolean attention mask.

    Tests 2D, 3D, and 4D boolean masks that are converted to additive bias
    (True -> 0.0, False -> mask_filter_value) in attention.cc. This exercises
    the LaunchConvertBoolMaskToAttentionBias kernel for the MHA path.
    """

    @parameterized.expand(mha_bool_mask_test_cases())
    def test_mha_bool_mask_fp16(self, name, config):
        seqlens = torch.tensor(
            [config.kv_sequence_length - 6, config.kv_sequence_length],
            dtype=torch.int32,
            device="cuda",
        )

        parity_check_mha_prompt_with_bool_mask(
            config=config,
            seqlens=seqlens,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            ort_type=TensorProto.FLOAT16,
            rtol=rtol["fp16"],
            atol=atol["fp16"],
        )


# #################################################################################################
#  Parity Check with nonpad_kv_seqlen (Opset 24)
# #################################################################################################


def parity_check_mha_prompt_with_nonpad_kv_seqlen(
    config: AttentionConfig,
    nonpad_seqlens: torch.Tensor,
    ep,
    device,
    torch_type,
    ort_type,
    rtol,
    atol,
    std=0.2,
):
    """
    Parity check for ONNX Attention op (opset 24) MHA path with nonpad_kv_seqlen.

    nonpad_kv_seqlen tells the op how many KV positions per batch are valid.
    Positions beyond the valid length are treated as padding and masked out.
    Cannot be used together with past_key/past_value.
    """
    torch.manual_seed(0)

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
        valid_len = nonpad_seqlens[b].item()
        if valid_len < config.kv_sequence_length:
            k[b, valid_len:, :, :] = 0
            v[b, valid_len:, :, :] = 0

    # Per-batch key_padding_mask [batch, kv_seq] for reference
    arange_kv = torch.arange(config.kv_sequence_length, device=device).unsqueeze(0)
    key_padding_mask = arange_kv < nonpad_seqlens.unsqueeze(1).to(device)  # [batch, kv_seq]

    out_ref, _ = attention_ref(
        q=q,
        k=k,
        v=v,
        key_padding_mask=key_padding_mask,
        causal=config.is_causal == 1,
        softcap=config.softcap,
    )

    # ORT path: use nonpad_kv_seqlen (int64 tensor)
    nonpad_kv_seqlen_tensor = nonpad_seqlens.to(torch.int64).to(device)

    out, present_k, present_v = attention_prompt_func(
        q=q,
        k=k,
        v=v,
        config=config,
        attn_mask=None,
        ep=ep,
        device=device,
        ort_type=ort_type,
        nonpad_kv_seqlen=nonpad_kv_seqlen_tensor,
    )

    out = torch.reshape(out, (config.batch_size, config.q_sequence_length, config.q_num_heads, config.head_size))

    # When nonpad_kv_seqlen=0 for a batch, all KV positions are masked → softmax yields NaN.
    # Zero out those batches in both ORT and reference for comparison.
    for b in range(config.batch_size):
        if nonpad_seqlens[b].item() == 0:
            out[b, :, :, :] = 0
            out_ref[b, :, :, :] = 0

    out_np = out.to(torch.float32).detach().cpu().numpy()
    out_ref_np = out_ref.to(torch.float32).detach().cpu().numpy()

    print_diff_statistics(torch.tensor(out_np - out_ref_np), "out")
    numpy.testing.assert_allclose(out_np, out_ref_np, rtol=rtol, atol=atol)


def mha_nonpad_kv_seqlen_test_cases():
    """
    Generate test cases for ONNX Attention op (opset 24) MHA path with nonpad_kv_seqlen.

    MHA supports partial masking in prompt mode via FlashAttentionForExternalKVCache.
    Full-length tests verify the no-masking case; partial-length tests verify actual masking.
    """
    h = 128
    sq = 16
    skv = 16
    n = 8

    seqlen_scenarios = [
        (1, [16], "single_batch"),
        (2, [16, 16], "full_len"),
        (2, [3, 5], "partial_mask"),
        (4, [16, 16, 16, 16], "multi_batch"),
    ]

    for batch_size, seqlens, label in seqlen_scenarios:
        config = AttentionConfig(
            batch_size=batch_size,
            q_sequence_length=sq,
            kv_sequence_length=skv,
            past_kv_sequence_length=0,
            q_num_heads=n,
            kv_num_heads=n,
            head_size=h,
            is_causal=0,
            has_nonpad_kv_seqlen=True,
            attn_mask_type="additive",
        )
        name = f"b{batch_size}_sq{sq}_skv{skv}_nh{n}_h{h}_{label}"
        yield name, config, seqlens

    # Causal variation with full length
    config_c = AttentionConfig(
        batch_size=2,
        q_sequence_length=sq,
        kv_sequence_length=skv,
        past_kv_sequence_length=0,
        q_num_heads=n,
        kv_num_heads=n,
        head_size=h,
        is_causal=1,
        has_nonpad_kv_seqlen=True,
        attn_mask_type="additive",
    )
    yield f"b2_sq{sq}_skv{skv}_nh{n}_h{h}_causal", config_c, [16, 16]


def mha_nonpad_kv_seqlen_cpu_test_cases():
    """CPU-only test cases including zero_seqlen (triggers CUDA_KERNEL_ASSERT in debug builds)."""
    yield from mha_nonpad_kv_seqlen_test_cases()

    h = 128
    sq = 16
    skv = 16
    n = 8
    config = AttentionConfig(
        batch_size=2,
        q_sequence_length=sq,
        kv_sequence_length=skv,
        past_kv_sequence_length=0,
        q_num_heads=n,
        kv_num_heads=n,
        head_size=h,
        is_causal=0,
        has_nonpad_kv_seqlen=True,
        attn_mask_type="additive",
    )
    yield f"b2_sq{sq}_skv{skv}_nh{n}_h{h}_zero_seqlen", config, [0, 5]


@unittest.skipIf(not has_cuda_device(53), "CUDA device not available, skipping MHA tests.")
class TestONNXAttentionMHANonpadKVSeqlen(unittest.TestCase):
    """Test ONNX Attention op (opset 24) MHA path with nonpad_kv_seqlen on CUDA."""

    @parameterized.expand(mha_nonpad_kv_seqlen_test_cases())
    def test_mha_nonpad_kv_seqlen_fp16(self, name, config, seqlens):
        nonpad_seqlens = torch.tensor(seqlens, dtype=torch.int64, device="cuda")

        parity_check_mha_prompt_with_nonpad_kv_seqlen(
            config=config,
            nonpad_seqlens=nonpad_seqlens,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            ort_type=TensorProto.FLOAT16,
            rtol=rtol["fp16"],
            atol=atol["fp16"],
        )

    @parameterized.expand(mha_nonpad_kv_seqlen_test_cases())
    def test_mha_nonpad_kv_seqlen_fp32(self, name, config, seqlens):
        config.kv_cache_type = "float32"
        nonpad_seqlens = torch.tensor(seqlens, dtype=torch.int64, device="cuda")

        parity_check_mha_prompt_with_nonpad_kv_seqlen(
            config=config,
            nonpad_seqlens=nonpad_seqlens,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float32,
            ort_type=TensorProto.FLOAT,
            rtol=rtol["fp32"],
            atol=atol["fp32"],
        )


class TestONNXAttentionMHANonpadKVSeqlenCPU(unittest.TestCase):
    """Test ONNX Attention op (opset 24) MHA path with nonpad_kv_seqlen on CPU (includes zero_seqlen)."""

    @parameterized.expand(mha_nonpad_kv_seqlen_cpu_test_cases())
    def test_mha_nonpad_kv_seqlen_cpu(self, name, config, seqlens):
        config.kv_cache_type = "float32"
        nonpad_seqlens = torch.tensor(seqlens, dtype=torch.int64, device="cpu")

        parity_check_mha_prompt_with_nonpad_kv_seqlen(
            config=config,
            nonpad_seqlens=nonpad_seqlens,
            ep="CPUExecutionProvider",
            device="cpu",
            torch_type=torch.float32,
            ort_type=TensorProto.FLOAT,
            rtol=rtol["fp32"],
            atol=atol["fp32"],
        )


# #################################################################################################
#  Unfused Kernel Tests
# #################################################################################################


def mha_unfused_test_cases():
    """
    Generate test cases that force the unfused attention kernel.

    On Ampere+ GPUs, fp16/bf16 normally route to flash attention. By disabling
    both flash and MEA, we force the unfused path even for fp16. These tests
    verify the unfused kernel handles MHA correctly.
    """
    cases = [
        (
            "prompt_causal",
            AttentionConfig(
                batch_size=2,
                q_sequence_length=16,
                kv_sequence_length=16,
                q_num_heads=8,
                kv_num_heads=8,
                head_size=64,
                is_causal=1,
                attn_mask_type="additive",
            ),
        ),
        (
            "prompt_noncausal",
            AttentionConfig(
                batch_size=2,
                q_sequence_length=16,
                kv_sequence_length=16,
                q_num_heads=8,
                kv_num_heads=8,
                head_size=64,
                is_causal=0,
                attn_mask_type="additive",
            ),
        ),
        (
            "decode_causal",
            AttentionConfig(
                batch_size=2,
                q_sequence_length=1,
                kv_sequence_length=1,
                past_kv_sequence_length=32,
                q_num_heads=8,
                kv_num_heads=8,
                head_size=64,
                is_causal=1,
                attn_mask_type="additive",
            ),
        ),
    ]
    return cases


@unittest.skipIf(not has_cuda_device(53), "CUDA device not available, skipping unfused tests.")
@patch.dict(os.environ, {"ORT_DISABLE_FLASH_ATTENTION": "1", "ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION": "1"})
class TestONNXAttentionMHAUnfused(unittest.TestCase):
    """
    Test the unfused attention kernel by disabling flash and MEA.

    On Ampere+ GPUs, fp16 normally routes to flash attention. These tests
    disable both flash and MEA to exercise the unfused path.
    """

    @parameterized.expand(mha_unfused_test_cases())
    def test_mha_unfused_fp16(self, name, config):
        if "decode" in name:
            parity_check_mha_past(
                config=config,
                ep="CUDAExecutionProvider",
                device="cuda",
                torch_type=torch.float16,
                ort_type=TensorProto.FLOAT16,
                causal=config.is_causal == 1,
                rtol=rtol["fp16"],
                atol=atol["fp16"],
            )
        else:
            parity_check_mha_prompt(
                config=config,
                ep="CUDAExecutionProvider",
                device="cuda",
                torch_type=torch.float16,
                ort_type=TensorProto.FLOAT16,
                causal=config.is_causal == 1,
                rtol=rtol["fp16"],
                atol=atol["fp16"],
            )


# #################################################################################################
#  Broadcast Mask (1,1,q,kv) Tests
# #################################################################################################


@unittest.skipIf(not has_cuda_device(53), "CUDA device not available, skipping broadcast mask tests.")
class TestONNXAttentionMHABroadcastMask(unittest.TestCase):
    """
    Test attention with a (1,1,q_seq,kv_seq) mask that broadcasts across batch and heads.

    This is a 4D mask with dim_0=1 (batch) and dim_1=1 (heads), verifying that
    the broadcast_attn_bias_dim_0 and broadcast_attn_bias_dim_1 flags work correctly.
    """

    def test_mha_broadcast_mask_additive(self):
        """Test broadcast additive mask (1,1,q,kv) with MHA on CUDA."""
        config = AttentionConfig(
            batch_size=2,
            q_sequence_length=16,
            kv_sequence_length=16,
            q_num_heads=8,
            kv_num_heads=8,
            head_size=128,
            is_causal=0,
            has_attn_mask=True,
            attn_mask_dims=4,
            attn_mask_type="additive",
            broadcast_mask_batch=True,
            broadcast_mask_heads=True,
        )

        torch.manual_seed(0)
        device = "cuda"
        torch_type = torch.float16

        q = torch.randn(2, 16, 8, 128, device=device, dtype=torch_type) * 0.2
        k = torch.randn(2, 16, 8, 128, device=device, dtype=torch_type) * 0.2
        v = torch.randn_like(k) * 0.2

        # Create (1,1,q,kv) additive mask: lower-triangular causal pattern
        mask_filter = float(torch.finfo(torch_type).min)
        mask_2d = torch.zeros(16, 16, device=device, dtype=torch_type)
        for i in range(16):
            mask_2d[i, i + 1 :] = mask_filter
        attn_mask = mask_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, 16, 16)

        # Reference: expand to full (B, H, Q, K)
        attn_bias_ref = attn_mask.expand(2, 8, -1, -1).contiguous()
        out_ref, _ = attention_ref(q=q, k=k, v=v, attn_bias=attn_bias_ref, causal=False)

        # ORT path
        out_ort, _, _ = attention_prompt_func(
            q=q,
            k=k,
            v=v,
            config=config,
            attn_mask=attn_mask,
            ep="CUDAExecutionProvider",
            device=device,
            ort_type=TensorProto.FLOAT16,
        )
        out_ort = out_ort.reshape(2, 16, 8, 128)

        out_np = out_ort.float().detach().cpu().numpy()
        out_ref_np = out_ref.float().detach().cpu().numpy()
        numpy.testing.assert_allclose(out_np, out_ref_np, rtol=rtol["fp16"], atol=atol["fp16"])


# #################################################################################################
#  2D Mask Broadcast Regression Test
# #################################################################################################


@unittest.skipIf(not has_cuda_device(53), "CUDA device not available, skipping 2D mask broadcast tests.")
class TestONNXAttentionMHA2DMaskBroadcast(unittest.TestCase):
    """
    Regression test for 2D mask [q_seq, total_seq] broadcast correctness.

    Per ONNX spec, a 2D attention mask has shape [q_seq, total_seq] and broadcasts
    over batch and heads. This test uses batch_size > q_seq with a non-uniform
    mask (different values per row) to verify correct broadcast behavior.

    The old bug indexed the 2D mask by batch index instead of query position,
    causing OOB reads when batch_size > q_seq.
    """

    def test_2d_additive_mask_batch_gt_qseq(self):
        """2D additive mask [q_seq=2, total_seq=8] with batch=4 — would OOB on old code."""
        config = AttentionConfig(
            batch_size=4,
            q_sequence_length=2,
            kv_sequence_length=8,
            q_num_heads=4,
            kv_num_heads=4,
            head_size=64,
            is_causal=0,
            has_attn_mask=True,
            attn_mask_dims=2,
            attn_mask_type="additive",
        )

        torch.manual_seed(42)
        device = "cuda"
        torch_type = torch.float16
        mask_filter_value = torch.finfo(torch_type).min

        q = (
            torch.randn(
                config.batch_size,
                config.q_sequence_length,
                config.q_num_heads,
                config.head_size,
                device=device,
                dtype=torch_type,
            )
            * 0.2
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
            * 0.2
        )
        v = torch.randn_like(k) * 0.2

        # Create a non-uniform 2D causal-style mask [q_seq=2, kv_seq=8]:
        # Row 0: attend to positions 0-3, mask out 4-7
        # Row 1: attend to positions 0-5, mask out 6-7
        attn_mask = torch.zeros(config.q_sequence_length, config.kv_sequence_length, device=device, dtype=torch_type)
        attn_mask[0, 4:] = mask_filter_value
        attn_mask[1, 6:] = mask_filter_value

        # Reference: broadcast [2, 8] → [4, 4, 2, 8] (same mask for all batches/heads)
        attn_bias_ref = attn_mask.unsqueeze(0).unsqueeze(0).expand(config.batch_size, config.q_num_heads, -1, -1)

        # PyTorch reference
        out_ref, _ = attention_ref(q=q, k=k, v=v, attn_bias=attn_bias_ref, causal=False)

        # ORT path
        out_ort, _, _ = attention_prompt_func(
            q=q,
            k=k,
            v=v,
            config=config,
            attn_mask=attn_mask,
            ep="CUDAExecutionProvider",
            device=device,
            ort_type=TensorProto.FLOAT16,
        )
        out_ort = out_ort.reshape(config.batch_size, config.q_sequence_length, config.q_num_heads, config.head_size)

        out_np = out_ort.float().detach().cpu().numpy()
        out_ref_np = out_ref.float().detach().cpu().numpy()
        numpy.testing.assert_allclose(out_np, out_ref_np, rtol=rtol["fp16"], atol=atol["fp16"])

    def test_2d_bool_mask_batch_gt_qseq(self):
        """2D bool mask [q_seq=2, total_seq=8] with batch=4 — would OOB on old code."""
        config = AttentionConfig(
            batch_size=4,
            q_sequence_length=2,
            kv_sequence_length=8,
            q_num_heads=4,
            kv_num_heads=4,
            head_size=64,
            is_causal=0,
            has_attn_mask=True,
            attn_mask_dims=2,
            attn_mask_type="bool",
        )

        torch.manual_seed(42)
        device = "cuda"
        torch_type = torch.float16

        q = (
            torch.randn(
                config.batch_size,
                config.q_sequence_length,
                config.q_num_heads,
                config.head_size,
                device=device,
                dtype=torch_type,
            )
            * 0.2
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
            * 0.2
        )
        v = torch.randn_like(k) * 0.2

        # Create non-uniform 2D bool mask [q_seq=2, kv_seq=8]:
        # Row 0: True for positions 0-3, False for 4-7
        # Row 1: True for positions 0-5, False for 6-7
        attn_mask = torch.ones(config.q_sequence_length, config.kv_sequence_length, device=device, dtype=torch.bool)
        attn_mask[0, 4:] = False
        attn_mask[1, 6:] = False

        # Build additive bias from the bool mask for the PyTorch reference.
        # 2D mask broadcasts identically across batches and heads.
        mask_filter_value = torch.finfo(torch_type).min
        attn_bias_ref = torch.where(
            attn_mask.unsqueeze(0).unsqueeze(0).expand(config.batch_size, config.q_num_heads, -1, -1),
            torch.tensor(0.0, dtype=torch_type, device=device),
            torch.tensor(mask_filter_value, dtype=torch_type, device=device),
        )

        # PyTorch reference with explicit bias
        out_ref, _ = attention_ref(q=q, k=k, v=v, attn_bias=attn_bias_ref, causal=False)

        # ORT path
        out_ort, _, _ = attention_prompt_func(
            q=q,
            k=k,
            v=v,
            config=config,
            attn_mask=attn_mask,
            ep="CUDAExecutionProvider",
            device=device,
            ort_type=TensorProto.FLOAT16,
        )
        out_ort = out_ort.reshape(config.batch_size, config.q_sequence_length, config.q_num_heads, config.head_size)

        out_np = out_ort.float().detach().cpu().numpy()
        out_ref_np = out_ref.float().detach().cpu().numpy()
        numpy.testing.assert_allclose(out_np, out_ref_np, rtol=rtol["fp16"], atol=atol["fp16"])


# NOTE: GQA fully-masked batch fix (ZeroOutputForFullyMaskedBatches) is validated by
# C++ test Attention_NonPadKVSeqLen_AllMasked_FP16_GQA. Python graph-level test omitted
# because the fix is a CUDA kernel in the MEA path — a CPU-only test cannot validate it,
# and CUDA debug builds trigger kernel assertions for zero seqlens (pre-existing issue
# with 4D mask alignment in attention_transpose.cu).


if __name__ == "__main__":
    unittest.main()
