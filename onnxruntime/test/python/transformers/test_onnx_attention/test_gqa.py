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
Tests for ONNX Attention op (opset 23) — GQA path (kv_num_heads != q_num_heads).

The GQA path in attention.cc is exercised when kv_num_heads != q_num_heads.
It requires:
  - float16 or bfloat16 (no float32)
  - 3D inputs (BSNH format)
  - Causal attention (is_causal=1)
  - Self-attention only (kv_seq == q_seq)
  - Boolean padding mask (converted to seqlens_k internally)
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
    enable_debug_print,
    enable_deterministic_check,
    has_cuda_device,
    has_flash_attention,
    pipeline_mode,
    print_diff_statistics,
    quick_build,
    rtol,
)

# #################################################################################################
#  Parity Check (Core Test Logic)
# #################################################################################################


def parity_check_gqa_prompt(
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
    attn_mask = None
    key_padding_mask = None
    if config.has_attn_mask:
        attn_mask = torch.ones(
            config.batch_size,
            config.kv_sequence_length,
            device=device,
            dtype=torch.bool,
        )
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

    if config.use_4d_bnsh:
        # Torch SDPA outputs [B, num_heads, q_seq, head_size] (BNSH format).
        # For 4D BNSH test configs, transpose to [B, q_seq, num_heads, head_size] (BSNH)
        # to match ORT's 3D output convention for comparison.
        out = out.transpose(1, 2).contiguous()
    else:
        out = torch.reshape(out, (config.batch_size, config.q_sequence_length, config.q_num_heads, config.head_size))
    out_np = out.to(torch.float32).detach().cpu().numpy()

    # --- Comparison ---
    nan_count = numpy.sum(numpy.isnan(out_np))
    if nan_count > 0:
        nan_indices = numpy.argwhere(numpy.isnan(out_np))
        print(f"DEBUG_NAN: Found {nan_count} NaN values in output!")
        print(f"DEBUG_NAN: First 5 NaN indices: {nan_indices[:5]}")

    # Compare KV cache (present_k should match k, present_v should match v)
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


def parity_check_gqa_past(
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
    new_k_bnsh = new_k.transpose(1, 2)
    new_v_bnsh = new_v.transpose(1, 2)

    full_k_bnsh = torch.cat([past_k, new_k_bnsh], dim=2)
    full_v_bnsh = torch.cat([past_v, new_v_bnsh], dim=2)

    full_k_bsnh = full_k_bnsh.transpose(1, 2)
    full_v_bsnh = full_v_bnsh.transpose(1, 2)

    total_seq_len = config.past_kv_sequence_length + config.kv_sequence_length

    attn_mask = None
    key_padding_mask = None
    if config.has_attn_mask:
        attn_mask = torch.ones(
            config.batch_size,
            total_seq_len,
            device=device,
            dtype=torch.bool,
        )
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

    if config.use_4d_bnsh:
        out = out.transpose(1, 2).contiguous()
    else:
        out = torch.reshape(out, (config.batch_size, config.q_sequence_length, config.q_num_heads, config.head_size))
    out_np = out.to(torch.float32).detach().cpu().numpy()

    if enable_debug_print:
        print(f"[DEBUG] out_np non-zeros: {numpy.count_nonzero(out_np)} / {out_np.size}")
        print(f"[DEBUG] out_ref_np non-zeros: {numpy.count_nonzero(out_ref_np)} / {out_ref_np.size}")

    if numpy.count_nonzero(out_ref_np) > 0 and numpy.count_nonzero(out_np) == 0:
        raise RuntimeError("Output is all zeros")

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


# #################################################################################################
#  Parity Checks with Padding Masks
# #################################################################################################


def parity_check_gqa_prompt_with_padding(
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

    # 2D and 3D masks broadcast across batches (no per-batch dimension), so they can only
    # represent one padding pattern. The mask uses batch 0's seqlen for all batches.
    # Adjust effective_seqlens so the reference comparison matches the actual mask.
    if config.attn_mask_dims in (2, 3):
        effective_seqlens = torch.full_like(seqlens, seqlens[0].item())
    else:
        effective_seqlens = seqlens

    # Zero out padded positions in K, V for proper comparison
    for b in range(config.batch_size):
        valid_len = effective_seqlens[b].item()
        if valid_len < config.kv_sequence_length:
            k[b, valid_len:, :, :] = 0
            v[b, valid_len:, :, :] = 0

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
    for b in range(config.batch_size):
        valid_len = effective_seqlens[b].item()
        if valid_len < config.q_sequence_length:
            out[b, valid_len:, :, :] = 0
            out_ref[b, valid_len:, :, :] = 0

    out_np = out.to(torch.float32).detach().cpu().numpy()
    out_ref_np = out_ref.to(torch.float32).detach().cpu().numpy()

    print_diff_statistics(torch.tensor(out_np - out_ref_np), "out")
    numpy.testing.assert_allclose(out_np, out_ref_np, rtol=rtol, atol=atol)


def parity_check_gqa_past_with_padding(
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

    for b in range(config.batch_size):
        valid_past_len = past_seqlens[b].item()
        if valid_past_len < config.past_kv_sequence_length:
            past_k[b, :, valid_past_len:, :] = 0
            past_v[b, :, valid_past_len:, :] = 0

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

    total_seqlens = past_seqlens + config.kv_sequence_length
    total_seq_len = config.past_kv_sequence_length + config.kv_sequence_length

    attn_mask = create_boolean_mask_from_seqlens(
        seqlens=total_seqlens,
        total_seq_len=total_seq_len,
        mask_dims=config.attn_mask_dims,
        q_seq_len=config.q_sequence_length,
        num_heads=config.q_num_heads,
        device=device,
    )

    key_padding_mask = create_boolean_mask_from_seqlens(
        seqlens=total_seqlens,
        total_seq_len=total_seq_len,
        mask_dims=2,
        device=device,
    )

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


def gqa_prompt_test_cases():
    """
    Generate test cases for ONNX Attention op GQA path in prompt phase.

    The GQA path requires:
    - kv_num_heads != q_num_heads
    - Causal attention (is_causal=1)
    - Self-attention (kv_seq == q_seq)
    - float16 or bfloat16 only
    """
    batches = [1, 2, 3]
    seqs = [(16, 16), (64, 64), (128, 128)]
    heads = [(8, 2), (8, 4)]
    h_sizes = [128] if quick_build else [64, 128]
    softcap_opts = [0.0, 50.0]

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
                        is_causal=1,
                        softcap=softcap,
                    )
                    name = f"b{b}_sq{sq}_skv{skv}_nh{n}_{n2}_h{h}_sc{softcap}"
                    yield name, config


def gqa_past_test_cases():
    """
    Generate test cases for ONNX Attention op GQA path in decoding phase (with past KV cache).
    """
    batches = [1, 2]
    seqs = [(1, 32), (1, 128), (1, 512)]
    heads = [(8, 2), (8, 4)]
    h_sizes = [128] if quick_build else [64, 128]
    softcap_opts = [0.0, 50.0]

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
                        kv_sequence_length=s,
                        past_kv_sequence_length=s2,
                        q_num_heads=n,
                        kv_num_heads=n2,
                        head_size=h,
                        is_causal=1,
                        softcap=softcap,
                    )
                    name = f"b{b}_s{s}_past{s2}_nh{n}_{n2}_h{h}_sc{softcap}"
                    yield name, config


def gqa_prompt_padding_test_cases():
    """
    Generate test cases for ONNX Attention op GQA path with boolean padding masks.

    Tests 2D, 3D, and 4D boolean masks for right-padding scenarios.
    Includes a batch_size=4, q_seq=1 case where batch_size != q_seq_len to
    guard against 2D mask shape bugs (must be [q_seq, total_seq] not [batch, total_seq]).
    """
    batches = [2]
    seqs = [(16, 16)]
    heads = [(8, 2)]
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
                            is_causal=1,
                            has_attn_mask=True,
                            attn_mask_dims=mask_dims,
                        )
                        name = f"b{b}_sq{sq}_skv{skv}_nh{n}_{n2}_h{h}_mask{mask_dims}d"
                        yield name, config

    # Guard case: batch_size=4 != q_seq_len=1 (decode). This catches the original bug
    # where 2D mask was [batch, total_seq] instead of [q_seq, total_seq].
    for mask_dims in mask_dims_options:
        config = AttentionConfig(
            batch_size=4,
            q_sequence_length=1,
            kv_sequence_length=32,
            past_kv_sequence_length=0,
            q_num_heads=8,
            kv_num_heads=2,
            head_size=128,
            is_causal=1,
            has_attn_mask=True,
            attn_mask_dims=mask_dims,
        )
        name = f"b4_sq1_skv32_nh8_2_h128_mask{mask_dims}d_shape_guard"
        yield name, config


def gqa_past_padding_test_cases():
    """
    Generate test cases for ONNX Attention op GQA path with boolean padding masks in decoding phase.
    """
    batches = [2]
    seqs = [(1, 32)]
    heads = [(8, 2)]
    h_sizes = [128]
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


@unittest.skipIf(not has_flash_attention(), "Flash Attention is not available, skipping tests.")
@patch.dict(os.environ, {"ORT_DISABLE_FLASH_ATTENTION": "0"})
class TestONNXAttentionFlashGQA(unittest.TestCase):
    """Test ONNX Attention op (opset 23) GQA path with Flash Attention.

    Requires SM80+: tests explicitly force Flash via ORT_DISABLE_FLASH_ATTENTION=0.
    """

    @parameterized.expand(gqa_prompt_test_cases())
    def test_gqa_prompt_flash(self, name, config):
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

    @parameterized.expand(gqa_past_test_cases())
    def test_gqa_past_flash(self, name, config):
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
@patch.dict(os.environ, {"ORT_DISABLE_FLASH_ATTENTION": "0"})
class TestONNXAttentionFlashGQABF16(unittest.TestCase):
    """Test ONNX Attention op (opset 23) GQA path with Flash Attention using BFloat16.

    Requires SM80+: tests explicitly force Flash via ORT_DISABLE_FLASH_ATTENTION=0,
    and BFloat16 requires Ampere or higher.
    """

    @parameterized.expand(gqa_prompt_test_cases())
    def test_gqa_prompt_flash_bf16(self, name, config):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BFloat16 not supported on this device")

        config.kv_cache_type = "bfloat16"
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

    @parameterized.expand(gqa_past_test_cases())
    def test_gqa_past_flash_bf16(self, name, config):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BFloat16 not supported on this device")

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


@unittest.skipIf(not has_cuda_device(53), "Memory Efficient Attention is not available, skipping tests.")
@patch.dict(os.environ, {"ORT_DISABLE_FLASH_ATTENTION": "1"})
class TestONNXAttentionMemoryEfficientGQA(unittest.TestCase):
    """Test ONNX Attention op (opset 23) GQA path with Memory Efficient Attention."""

    @parameterized.expand(gqa_prompt_test_cases())
    def test_gqa_prompt_memory_efficient(self, name, config):
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

    # Note: GQA past tests removed — MEA is ineligible when past_key is present
    # (ComputeInternal requires past_key == nullptr for MEA). GQA past requires
    # flash attention.


# TODO(titaiwang): Re-enable once PR #27851 merges (MEA supports past_key for GQA).
# Flash now rejects attn_mask (requires attn_mask==nullptr). GQA + bool mask + past_key
# has no runner until MEA supports past_key. See issue #27885.
@unittest.skip(
    "Flash now rejects attn_mask. GQA + bool mask + past_key has no runner "
    "until PR #27851 (MEA with past_key). See issue #27885."
)
@unittest.skipIf(not has_flash_attention(), "Flash Attention is not available, skipping tests.")
@patch.dict(os.environ, {"ORT_DISABLE_FLASH_ATTENTION": "0"})
class TestONNXAttentionPaddingMaskGQA(unittest.TestCase):
    """
    Test ONNX Attention op (opset 23) GQA path with boolean padding masks.

    SKIPPED: Flash now requires attn_mask == nullptr. GQA + bool attn_mask +
    past_key currently has no runner (Flash rejected, unfused doesn't support GQA,
    MEA blocked by past_key != nullptr). Will be re-enabled when PR #27851 lands.

    These tests verify that the boolean attn_mask is correctly converted to
    sequence lengths on GPU and that the attention computation respects the
    padding. Tests cover 2D, 3D, and 4D mask shapes.
    """

    @parameterized.expand(gqa_past_padding_test_cases())
    def test_gqa_past_padding_flash(self, name, config):
        """Test decoding phase with padding mask using Flash Attention."""
        past_seqlens = torch.full(
            (config.batch_size,),
            config.past_kv_sequence_length,
            dtype=torch.int32,
            device="cuda",
        )

        parity_check_gqa_past_with_padding(
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
@patch.dict(os.environ, {"ORT_DISABLE_FLASH_ATTENTION": "1"})
class TestONNXAttentionPaddingMaskMemoryEfficientGQA(unittest.TestCase):
    """
    Test ONNX Attention op (opset 23) GQA path with boolean padding masks
    using Memory Efficient Attention.
    """

    @parameterized.expand(gqa_prompt_padding_test_cases())
    def test_gqa_prompt_padding_mea(self, name, config):
        """Test prompt phase with padding mask using Memory Efficient Attention."""
        # Create seqlens with config.batch_size elements.
        # First batch has shorter valid length, rest at full length.
        seqlens_list = [config.kv_sequence_length - 6] + [config.kv_sequence_length] * (config.batch_size - 1)
        seqlens = torch.tensor(
            seqlens_list,
            dtype=torch.int32,
            device="cuda",
        )

        parity_check_gqa_prompt_with_padding(
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


def parity_check_gqa_prompt_with_nonpad_kv_seqlen(
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
    Parity check for ONNX Attention op (opset 24) GQA path with nonpad_kv_seqlen.

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


def gqa_nonpad_kv_seqlen_test_cases():
    """
    Generate test cases for ONNX Attention op (opset 24) GQA path with nonpad_kv_seqlen.

    In prompt mode (q_seq == kv_seq), the GQA kernel ignores seqlens_k and uses
    padded_seq_lens = sequence_length unconditionally. nonpad_kv_seqlen masking is only
    meaningful for decode (q_seq != kv_seq), which routes to FlashAttentionForExternalKVCache.
    In the real TensorScatter workflow, prompt mode always has all tokens valid, so
    nonpad_kv_seqlen = total_kv_sequence_length (mask nothing).
    """
    h = 128
    sq = 16
    skv = 16
    n = 8
    n2 = 2

    # In prompt mode, nonpad_kv_seqlen should equal total_kv_sequence_length (all tokens valid).
    # Partial masking (nonpad < kv_sequence_length) is not supported by the GQA kernel in prompt mode.
    seqlen_scenarios = [
        (1, [16], "single_batch"),
        (2, [16, 16], "full_len"),
        (4, [16, 16, 16, 16], "multi_batch"),
    ]

    for batch_size, seqlens, label in seqlen_scenarios:
        config = AttentionConfig(
            batch_size=batch_size,
            q_sequence_length=sq,
            kv_sequence_length=skv,
            past_kv_sequence_length=0,
            q_num_heads=n,
            kv_num_heads=n2,
            head_size=h,
            is_causal=1,
            has_nonpad_kv_seqlen=True,
        )
        name = f"b{batch_size}_sq{sq}_skv{skv}_nh{n}_{n2}_h{h}_{label}"
        yield name, config, seqlens


def gqa_nonpad_kv_seqlen_cpu_test_cases():
    """CPU-only test cases including zero_seqlen (triggers CUDA_KERNEL_ASSERT in debug builds)."""
    yield from gqa_nonpad_kv_seqlen_test_cases()

    h = 128
    sq = 16
    skv = 16
    n = 8
    n2 = 2
    config = AttentionConfig(
        batch_size=2,
        q_sequence_length=sq,
        kv_sequence_length=skv,
        past_kv_sequence_length=0,
        q_num_heads=n,
        kv_num_heads=n2,
        head_size=h,
        is_causal=1,
        has_nonpad_kv_seqlen=True,
    )
    yield f"b2_sq{sq}_skv{skv}_nh{n}_{n2}_h{h}_zero_seqlen", config, [0, 16]


@unittest.skipIf(not has_flash_attention(), "Flash Attention is not available, skipping tests.")
@patch.dict(os.environ, {"ORT_DISABLE_FLASH_ATTENTION": "0"})
class TestONNXAttentionGQANonpadKVSeqlen(unittest.TestCase):
    """Test ONNX Attention op (opset 24) GQA path with nonpad_kv_seqlen (Flash Attention).

    Requires SM80+: tests explicitly force Flash via ORT_DISABLE_FLASH_ATTENTION=0.
    """

    @parameterized.expand(gqa_nonpad_kv_seqlen_test_cases())
    def test_gqa_nonpad_kv_seqlen_flash(self, name, config, seqlens):
        nonpad_seqlens = torch.tensor(seqlens, dtype=torch.int64, device="cuda")

        parity_check_gqa_prompt_with_nonpad_kv_seqlen(
            config=config,
            nonpad_seqlens=nonpad_seqlens,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            ort_type=TensorProto.FLOAT16,
            rtol=rtol["fp16"],
            atol=atol["fp16"],
        )


@unittest.skipIf(not has_cuda_device(53), "CUDA device not available, skipping tests.")
@patch.dict(os.environ, {"ORT_DISABLE_FLASH_ATTENTION": "1"})
class TestONNXAttentionGQANonpadKVSeqlenMEA(unittest.TestCase):
    """Test ONNX Attention op (opset 24) GQA path with nonpad_kv_seqlen (Memory Efficient Attention)."""

    @parameterized.expand(gqa_nonpad_kv_seqlen_test_cases())
    def test_gqa_nonpad_kv_seqlen_mea(self, name, config, seqlens):
        nonpad_seqlens = torch.tensor(seqlens, dtype=torch.int64, device="cuda")

        parity_check_gqa_prompt_with_nonpad_kv_seqlen(
            config=config,
            nonpad_seqlens=nonpad_seqlens,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            ort_type=TensorProto.FLOAT16,
            rtol=rtol["fp16"],
            atol=atol["fp16"],
        )


class TestONNXAttentionGQANonpadKVSeqlenCPU(unittest.TestCase):
    """Test ONNX Attention op (opset 24) GQA path with nonpad_kv_seqlen on CPU (includes zero_seqlen)."""

    @parameterized.expand(gqa_nonpad_kv_seqlen_cpu_test_cases())
    def test_gqa_nonpad_kv_seqlen_cpu(self, name, config, seqlens):
        nonpad_seqlens = torch.tensor(seqlens, dtype=torch.int64, device="cpu")

        parity_check_gqa_prompt_with_nonpad_kv_seqlen(
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
#  GQA 4D BNSH Format Tests
# #################################################################################################


def gqa_4d_bnsh_test_cases():
    """Generate test cases for GQA with 4D BNSH input format."""
    return [
        (
            "prompt_nomask",
            AttentionConfig(
                batch_size=2,
                q_sequence_length=16,
                kv_sequence_length=16,
                q_num_heads=8,
                kv_num_heads=2,
                head_size=128,
                is_causal=1,
                use_4d_bnsh=True,
            ),
        ),
        (
            "prompt_smallhead",
            AttentionConfig(
                batch_size=2,
                q_sequence_length=16,
                kv_sequence_length=16,
                q_num_heads=8,
                kv_num_heads=4,
                head_size=64,
                is_causal=1,
                use_4d_bnsh=True,
            ),
        ),
    ]


def gqa_4d_bnsh_past_test_cases():
    """Generate test cases for GQA decode with 4D BNSH input format."""
    return [
        (
            "decode_nomask",
            AttentionConfig(
                batch_size=2,
                q_sequence_length=1,
                kv_sequence_length=1,
                past_kv_sequence_length=32,
                q_num_heads=8,
                kv_num_heads=2,
                head_size=128,
                is_causal=1,
                use_4d_bnsh=True,
            ),
        ),
    ]


@unittest.skipIf(not has_flash_attention(), "Flash Attention is not available, skipping 4D BNSH tests.")
@patch.dict(os.environ, {"ORT_DISABLE_FLASH_ATTENTION": "0"})
class TestONNXAttentionGQA4DBNSH(unittest.TestCase):
    """
    Test GQA with 4D BNSH input format [batch, num_heads, seq, head_size].

    Requires SM80+: tests explicitly force Flash via ORT_DISABLE_FLASH_ATTENTION=0.
    The C++ attention op detects 4D inputs and sets transpose_output=false.
    Flash/MEA always expect BSNH, so the dispatcher transposes Q internally.
    """

    @parameterized.expand(gqa_4d_bnsh_test_cases())
    def test_gqa_4d_bnsh_prompt(self, name, config):
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

    @parameterized.expand(gqa_4d_bnsh_past_test_cases())
    def test_gqa_4d_bnsh_decode(self, name, config):
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


# #################################################################################################
#  GQA Float Additive Mask Tests
# #################################################################################################


@unittest.skipIf(not has_cuda_device(53), "CUDA device not available, skipping float mask tests.")
@patch.dict(os.environ, {"ORT_DISABLE_FLASH_ATTENTION": "1"})
class TestONNXAttentionGQAFloatMask(unittest.TestCase):
    """
    Test GQA with float additive attention mask (not bool) during prompt.

    This exercises MEA's GQA expansion + float bias path. The GQA path converts
    the additive mask to attention bias for MEA cutlass FMHA.
    """

    def test_gqa_prompt_float_mask_4d(self):
        """Test GQA prompt with 4D float additive mask."""
        config = AttentionConfig(
            batch_size=2,
            q_sequence_length=16,
            kv_sequence_length=16,
            q_num_heads=8,
            kv_num_heads=2,
            head_size=128,
            is_causal=0,
            has_attn_mask=True,
            attn_mask_dims=4,
            attn_mask_type="additive",
        )

        torch.manual_seed(0)
        device = "cuda"
        torch_type = torch.float16

        q = torch.randn(2, 16, 8, 128, device=device, dtype=torch_type) * 0.2
        k = torch.randn(2, 16, 2, 128, device=device, dtype=torch_type) * 0.2
        v = torch.randn_like(k) * 0.2

        # Create additive mask with padding pattern: batch 0 has 10 valid, batch 1 full
        seqlens = torch.tensor([10, 16], dtype=torch.int32, device=device)
        attn_mask = create_additive_mask_from_seqlens(
            seqlens=seqlens,
            total_seq_len=16,
            mask_dims=4,
            q_seq_len=16,
            num_heads=8,
            device=device,
            dtype=torch_type,
        )

        # Zero padded KV positions
        k[0, 10:, :, :] = 0
        v[0, 10:, :, :] = 0

        # Reference
        attn_bias_ref = attn_mask
        out_ref, _ = attention_ref(q=q, k=k, v=v, attn_bias=attn_bias_ref, causal=False)

        # ORT path (MEA handles GQA+float mask)
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

        # Zero padded output for comparison
        out_ort[0, 10:, :, :] = 0
        out_ref[0, 10:, :, :] = 0

        out_np = out_ort.float().detach().cpu().numpy()
        out_ref_np = out_ref.float().detach().cpu().numpy()
        print_diff_statistics(torch.tensor(out_np - out_ref_np), "out")
        numpy.testing.assert_allclose(out_np, out_ref_np, rtol=rtol["fp16"], atol=atol["fp16"])


# #################################################################################################
#  Large Head Size Unfused GQA Tests (head_size=512, fixes #28195)
#
#  Flash Attention and Memory-Efficient Attention cap at head_size=256.  For head_size=512 the
#  op falls through to RunGqaUnfusedAttention which writes Q*K^T to an FP32 scratch buffer,
#  eliminating fp16/bf16 overflow that caused NaNs (e.g. Gemma 4 global-attention layers).
#
#  These tests deliberately disable both Flash and MEA to make the unfused fallback explicit
#  and to guard against future changes that might inadvertently route large-head configs
#  away from the FP32-scratch path.
# #################################################################################################


def gqa_large_head_unfused_test_cases():
    """Test cases for GQA with head_size=512 (unfused FP32-QK path, fixes #28195)."""
    # prompt phase
    for b, sq in [(1, 16), (2, 64)]:
        for softcap in [0.0, 50.0]:
            config = AttentionConfig(
                batch_size=b,
                q_sequence_length=sq,
                kv_sequence_length=sq,
                past_kv_sequence_length=0,
                q_num_heads=8,
                kv_num_heads=4,
                head_size=512,
                is_causal=1,
                softcap=softcap,
            )
            yield f"prompt_b{b}_sq{sq}_sc{softcap}", config

    # decode phase (past KV cache)
    for b, past in [(1, 32), (2, 128)]:
        config = AttentionConfig(
            batch_size=b,
            q_sequence_length=1,
            kv_sequence_length=1,
            past_kv_sequence_length=past,
            q_num_heads=8,
            kv_num_heads=4,
            head_size=512,
            is_causal=1,
            softcap=0.0,
        )
        yield f"decode_b{b}_past{past}", config

    # prompt with boolean attn_mask (exercises ConvertAttnMaskToBias + unfused bias path)
    config = AttentionConfig(
        batch_size=2,
        q_sequence_length=32,
        kv_sequence_length=32,
        past_kv_sequence_length=0,
        q_num_heads=8,
        kv_num_heads=4,
        head_size=512,
        is_causal=1,
        has_attn_mask=True,
    )
    yield "prompt_attn_mask", config

    # prompt with nonpad_kv_seqlen (opset 24, exercises seqlens_k path in unfused kernel)
    config = AttentionConfig(
        batch_size=2,
        q_sequence_length=32,
        kv_sequence_length=32,
        past_kv_sequence_length=0,
        q_num_heads=8,
        kv_num_heads=4,
        head_size=512,
        is_causal=1,
        has_nonpad_kv_seqlen=True,
    )
    yield "prompt_nonpad_seqlen", config


@unittest.skipIf(not has_cuda_device(53), "CUDA device not available, skipping large head unfused tests.")
@patch.dict(os.environ, {"ORT_DISABLE_FLASH_ATTENTION": "1", "ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION": "1"})
class TestONNXAttentionGQALargeHeadUnfused(unittest.TestCase):
    """
    Regression tests for GQA with head_size=512 via the unfused FP32-QK path (issue #28195).

    Flash Attention and MEA both cap at head_size=256.  With both disabled the op routes
    to RunGqaUnfusedAttention, which writes Q*K^T to an FP32 scratch buffer to avoid
    fp16/bf16 overflow that produced NaNs for Gemma 4 global-attention layers.

    Validates: no NaNs, numerical parity vs. PyTorch SDPA reference, for fp16 and bf16.
    """

    @parameterized.expand(gqa_large_head_unfused_test_cases())
    def test_gqa_large_head_unfused_fp16(self, name, config):
        func = parity_check_gqa_past if "decode" in name else parity_check_gqa_prompt
        kwargs = dict(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.float16,
            ort_type=TensorProto.FLOAT16,
            causal=True,
            rtol=rtol["fp16"],
            atol=atol["fp16"],
        )
        func(**kwargs)

    @parameterized.expand(gqa_large_head_unfused_test_cases())
    def test_gqa_large_head_unfused_bf16(self, name, config):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BFloat16 not supported on this device")
        func = parity_check_gqa_past if "decode" in name else parity_check_gqa_prompt
        kwargs = dict(
            config=config,
            ep="CUDAExecutionProvider",
            device="cuda",
            torch_type=torch.bfloat16,
            ort_type=TensorProto.BFLOAT16,
            causal=True,
            rtol=rtol["bf16"],
            atol=atol["bf16"],
        )
        func(**kwargs)

    def test_gqa_large_head_unfused_softcap_additive_mask_poison_fp16(self):
        config = AttentionConfig(
            batch_size=1,
            q_sequence_length=1,
            kv_sequence_length=3,
            past_kv_sequence_length=0,
            q_num_heads=8,
            kv_num_heads=4,
            head_size=512,
            is_causal=0,
            softcap=1.0,
            has_attn_mask=True,
            attn_mask_dims=4,
            attn_mask_type="additive",
        )

        device = "cuda"
        torch_type = torch.float16
        q = torch.zeros(
            config.batch_size,
            config.q_sequence_length,
            config.q_num_heads,
            config.head_size,
            device=device,
            dtype=torch_type,
        )
        k = torch.zeros(
            config.batch_size,
            config.kv_sequence_length,
            config.kv_num_heads,
            config.head_size,
            device=device,
            dtype=torch_type,
        )
        v = torch.full_like(k, 0.2)
        v[:, 1, :, :] = 1000.0

        attn_mask = torch.zeros(
            config.batch_size,
            config.q_num_heads,
            config.q_sequence_length,
            config.kv_sequence_length,
            device=device,
            dtype=torch_type,
        )
        attn_mask[:, :, :, 1] = float("-inf")

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

        out = out_ort.reshape(
            config.batch_size,
            config.q_sequence_length,
            config.q_num_heads,
            config.head_size,
        )
        expected = torch.full_like(out, 0.2)
        torch.testing.assert_close(out, expected, rtol=0, atol=2e-2)
        self.assertLess(out.float().max().item(), 1.0)


if __name__ == "__main__":
    unittest.main()
