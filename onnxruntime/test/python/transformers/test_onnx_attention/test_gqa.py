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

import math
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

    # --- Create attn_mask matching the ONNX model's expected shape ---
    attn_mask = None
    key_padding_mask = None
    if config.has_attn_mask:
        total_seq = config.past_kv_sequence_length + config.kv_sequence_length
        # 2D mask shape: [q_seq, total_seq] per ONNX spec (matches create_attention_graph_prompt)
        attn_mask = torch.ones(
            config.q_sequence_length,
            total_seq,
            device=device,
            dtype=torch.bool,
        )
        # key_padding_mask for PyTorch reference: [batch, kv_seq]
        key_padding_mask = torch.ones(
            config.batch_size,
            config.kv_sequence_length,
            device=device,
            dtype=torch.bool,
        )

    # --- Create nonpad_kv_seqlen tensor if needed (opset 24+) ---
    nonpad_kv_seqlen = None
    if config.has_nonpad_kv_seqlen:
        # Each batch element has the full kv_sequence_length as valid (no padding)
        nonpad_kv_seqlen = torch.full(
            (config.batch_size,),
            config.kv_sequence_length,
            device=device,
            dtype=torch.int64,
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
    out, present_k, present_v = attention_prompt_func(
        q=q,
        k=k,
        v=v,
        config=config,
        attn_mask=attn_mask,
        ep=ep,
        device=device,
        ort_type=ort_type,
        nonpad_kv_seqlen=nonpad_kv_seqlen,
    )

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
            config.q_sequence_length,
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
    out, _present_k, _present_v = attention_prompt_func(
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
    out, _present_k, _present_v = attention_past_func(
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
    # NOTE: is_causal=0 because per ONNX spec, is_causal with S_q!=S_kv and no past_key
    # gives upper-left alignment (q[0] sees only kv[0]), which is not meaningful for decode.
    # KV bounds are enforced by the attention mask instead.
    for mask_dims in mask_dims_options:
        config = AttentionConfig(
            batch_size=4,
            q_sequence_length=1,
            kv_sequence_length=32,
            past_kv_sequence_length=0,
            q_num_heads=8,
            kv_num_heads=2,
            head_size=128,
            is_causal=0,
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
    # past=31 + new=1 = total_seq=32, which satisfies MEA's bias alignment
    # requirement (total_seq % 4 == 0) when attn_mask is present.
    seqs = [(1, 31)]
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


@unittest.skipIf(not has_cuda_device(80), "BF16 requires Ampere or higher GPU, skipping tests.")
@patch.dict(os.environ, {"ORT_DISABLE_FLASH_ATTENTION": "1"})
class TestONNXAttentionMemoryEfficientGQABF16(unittest.TestCase):
    """Test ONNX Attention op (opset 23) GQA path with Memory Efficient Attention using BFloat16."""

    @parameterized.expand(gqa_past_test_cases())
    def test_gqa_past_memory_efficient_bf16(self, name, config):
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
class TestONNXAttentionPaddingMaskMEAGQA(unittest.TestCase):
    """
    Test ONNX Attention op (opset 23) GQA path with boolean padding masks.

    GQA + bool attn_mask + past_key uses the MEA decode path (Flash requires
    attn_mask == nullptr). MEA handles bool masks via additive bias conversion.

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

    out, _present_k, _present_v = attention_prompt_func(
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
#  Large Head Size Unfused Tests (head_size=512, fixes #28195)
#
#  Flash Attention and Memory-Efficient Attention cap at head_size=256.  For head_size=512 the
#  op falls through to RunUnfusedAttention which writes Q*K^T to an FP32 scratch buffer,
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
    to RunUnfusedAttention, which writes Q*K^T to an FP32 scratch buffer to avoid
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


class TestONNXAttentionGQALargeHeadUnfusedCPU(unittest.TestCase):
    """CPU twin of TestONNXAttentionGQALargeHeadUnfused.

    CPU's only attention path is the unfused kernel, so head_size > 128 is
    naturally exercised here. fp16 only (CPU does not have a bf16 attention
    instantiation broadly available; the bf16 source method is intentionally
    not twinned). Configs reused verbatim from the CUDA source so the
    spec-property assertions stay paired.
    """

    @parameterized.expand(gqa_large_head_unfused_test_cases())
    def test_gqa_large_head_unfused_cpu_fp16(self, name, config):
        func = parity_check_gqa_past if "decode" in name else parity_check_gqa_prompt
        func(
            config=config,
            ep="CPUExecutionProvider",
            device="cpu",
            torch_type=torch.float16,
            ort_type=TensorProto.FLOAT16,
            causal=True,
            rtol=rtol["fp16"],
            atol=atol["fp16"],
        )

    def test_gqa_large_head_unfused_softcap_additive_mask_poison_cpu_fp16(self):
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

        device = "cpu"
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

        # Use a large finite negative instead of -inf for the additive mask
        # because the CPU softmax expects only finite inputs (see attention.h
        # mask_filter_value<T>()). softmax behaviour is identical via
        # underflow.
        attn_mask = torch.zeros(
            config.batch_size,
            config.q_num_heads,
            config.q_sequence_length,
            config.kv_sequence_length,
            device=device,
            dtype=torch_type,
        )
        attn_mask[:, :, :, 1] = -1.0e4  # fp16-representable large negative

        out_ort, _, _ = attention_prompt_func(
            q=q,
            k=k,
            v=v,
            config=config,
            attn_mask=attn_mask,
            ep="CPUExecutionProvider",
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


@unittest.skipIf(not has_cuda_device(53), "Memory Efficient Attention is not available, skipping tests.")
@patch.dict(os.environ, {"ORT_DISABLE_FLASH_ATTENTION": "1"})
class TestONNXAttentionMemoryEfficientGQAFloatMaskDecode(unittest.TestCase):
    """
    Test GQA with float additive attention mask during decode using MEA.

    This exercises the MEA decode path with float additive masks — a scenario
    that was a HARD ERROR before MEA+decode support (MEA was ineligible
    when past_key was present, so this fell through to no kernel).
    """

    def test_gqa_past_float_mask_4d(self):
        """Test GQA decode with 4D float additive mask via MEA."""
        config = AttentionConfig(
            batch_size=2,
            q_sequence_length=1,
            kv_sequence_length=1,
            past_kv_sequence_length=31,  # 31+1=32, divisible by 4 (CUTLASS bias alignment for MEA)
            q_num_heads=8,
            kv_num_heads=2,
            head_size=128,
            is_causal=1,
            has_attn_mask=True,
            attn_mask_dims=4,
            attn_mask_type="additive",
        )

        torch.manual_seed(0)
        device = "cuda"
        torch_type = torch.float16
        # std=0.2 keeps values in a numerically stable range for fp16 attention
        std = 0.2

        q = torch.randn(2, 1, 8, 128, device=device, dtype=torch_type) * std

        past_k = torch.randn(2, 2, 31, 128, device=device, dtype=torch_type) * std
        past_v = torch.randn_like(past_k) * std

        new_k = torch.randn(2, 1, 2, 128, device=device, dtype=torch_type) * std
        new_v = torch.randn_like(new_k) * std

        total_seq_len = 32  # past(31) + new(1), satisfies MEA bias alignment (32 % 4 == 0)

        # Create additive mask with padding pattern: batch 0 has 28 valid past, batch 1 full
        past_seqlens = torch.tensor([28, 31], dtype=torch.int32, device=device)
        total_seqlens = past_seqlens + config.kv_sequence_length

        attn_mask = create_additive_mask_from_seqlens(
            seqlens=total_seqlens,
            total_seq_len=total_seq_len,
            mask_dims=4,
            q_seq_len=1,
            num_heads=8,
            device=device,
            dtype=torch_type,
        )

        # Zero padded past positions for batch 0
        past_k[0, :, 28:, :] = 0
        past_v[0, :, 28:, :] = 0

        # Reference: concat past + new, then compute attention
        new_k_bnsh = new_k.transpose(1, 2)
        new_v_bnsh = new_v.transpose(1, 2)
        full_k_bnsh = torch.cat([past_k, new_k_bnsh], dim=2)
        full_v_bnsh = torch.cat([past_v, new_v_bnsh], dim=2)
        full_k_bsnh = full_k_bnsh.transpose(1, 2)
        full_v_bsnh = full_v_bnsh.transpose(1, 2)

        # Expand 4D mask to reference attn_bias [batch, heads, q_seq, total_seq]
        attn_bias_ref = attn_mask
        out_ref, _ = attention_ref(q=q, k=full_k_bsnh, v=full_v_bsnh, attn_bias=attn_bias_ref, causal=False)

        # ORT path
        out_ort, present_k, present_v = attention_past_func(
            q=q,
            past_k=past_k,
            past_v=past_v,
            new_k=new_k,
            new_v=new_v,
            config=config,
            attn_mask=attn_mask,
            ep="CUDAExecutionProvider",
            device=device,
            ort_type=TensorProto.FLOAT16,
        )

        out_ort = out_ort.reshape(2, 1, 8, 128)

        # --- Verify present_k/v match concatenated reference ---
        full_k_ref_np = full_k_bnsh.float().detach().cpu().numpy()
        full_v_ref_np = full_v_bnsh.float().detach().cpu().numpy()
        present_k_np = present_k.float().detach().cpu().numpy()
        present_v_np = present_v.float().detach().cpu().numpy()

        print_diff_statistics(torch.tensor(present_k_np - full_k_ref_np), "present_k")
        numpy.testing.assert_allclose(present_k_np, full_k_ref_np, rtol=rtol["fp16"], atol=atol["fp16"])
        print_diff_statistics(torch.tensor(present_v_np - full_v_ref_np), "present_v")
        numpy.testing.assert_allclose(present_v_np, full_v_ref_np, rtol=rtol["fp16"], atol=atol["fp16"])

        # --- Verify output ---
        out_np = out_ort.float().detach().cpu().numpy()
        out_ref_np = out_ref.float().detach().cpu().numpy()
        print_diff_statistics(torch.tensor(out_np - out_ref_np), "out")
        numpy.testing.assert_allclose(out_np, out_ref_np, rtol=rtol["fp16"], atol=atol["fp16"])


@unittest.skipIf(not has_cuda_device(53), "Memory Efficient Attention is not available, skipping tests.")
@patch.dict(os.environ, {"ORT_DISABLE_FLASH_ATTENTION": "1"})
class TestONNXAttentionMEAGQASoftcap(unittest.TestCase):
    """
    Test softcap support for GQA via the Memory Efficient Attention path.

    Disables Flash Attention to force MEA. Verifies softcap with and without
    attention mask for GQA (kv_num_heads != q_num_heads).

    MEA alignment requirement: total_seq % 4 == 0 when attn_mask is present.
    """

    def test_mea_gqa_softcap_with_mask_prompt_fp16(self):
        """MEA GQA softcap + causal mask, prompt phase, fp16."""
        config = AttentionConfig(
            batch_size=2,
            q_sequence_length=8,
            kv_sequence_length=8,  # total_seq=8, divisible by 4
            q_num_heads=8,
            kv_num_heads=4,
            head_size=64,
            is_causal=1,
            softcap=50.0,
            has_attn_mask=True,
        )
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

    def test_mea_gqa_softcap_no_mask_prompt_fp16(self):
        """MEA GQA softcap without explicit mask, prompt phase, fp16."""
        config = AttentionConfig(
            batch_size=2,
            q_sequence_length=8,
            kv_sequence_length=8,
            q_num_heads=8,
            kv_num_heads=4,
            head_size=64,
            is_causal=1,
            softcap=50.0,
        )
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

    def test_mea_gqa_softcap_with_mask_decode_fp16(self):
        """MEA GQA softcap + causal mask, decode phase, fp16."""
        config = AttentionConfig(
            batch_size=2,
            q_sequence_length=1,
            kv_sequence_length=1,
            past_kv_sequence_length=31,  # total_seq=32, divisible by 4
            q_num_heads=8,
            kv_num_heads=4,
            head_size=64,
            is_causal=1,
            softcap=50.0,
            has_attn_mask=True,
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

    def test_mea_gqa_softcap_mask_ordering_no_leakage_prompt_fp16(self):
        """Guard test: verify MEA GQA softcap + mask ordering prevents attention leakage.

        Same poison-value technique as the MHA ordering test, but with GQA
        (kv_num_heads != q_num_heads) forced to MEA path.
        """
        batch_size = 1
        q_seq = 4
        kv_seq = 8  # divisible by 4 for MEA alignment
        q_num_heads = 4
        kv_num_heads = 2
        head_size = 64
        softcap_val = 2.0
        valid_kv_len = 4

        config = AttentionConfig(
            batch_size=batch_size,
            q_sequence_length=q_seq,
            kv_sequence_length=kv_seq,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
            head_size=head_size,
            is_causal=0,
            softcap=softcap_val,
            has_attn_mask=True,
            attn_mask_dims=4,
            attn_mask_type="additive",
        )

        torch.manual_seed(42)
        device = "cuda"
        torch_type = torch.float16

        q = torch.randn(batch_size, q_seq, q_num_heads, head_size, dtype=torch_type, device=device) * 0.2
        k = torch.randn(batch_size, kv_seq, kv_num_heads, head_size, dtype=torch_type, device=device) * 0.2
        v = torch.randn(batch_size, kv_seq, kv_num_heads, head_size, dtype=torch_type, device=device) * 0.2

        # Place poison values in V at masked positions
        poison_value = 1000.0
        v[:, valid_kv_len:, :, :] = poison_value

        # Create additive mask: 0.0 for valid, -inf for masked
        # 4D mask: [batch, q_num_heads, q_seq, kv_seq]
        attn_mask = torch.zeros(batch_size, q_num_heads, q_seq, kv_seq, dtype=torch_type, device=device)
        attn_mask[:, :, :, valid_kv_len:] = float("-inf")

        out, _, _ = attention_prompt_func(
            q=q,
            k=k,
            v=v,
            config=config,
            attn_mask=attn_mask,
            ep="CUDAExecutionProvider",
            device=device,
            ort_type=TensorProto.FLOAT16,
        )

        out_np = out.to(torch.float32).detach().cpu().numpy().flatten()
        max_abs = numpy.max(numpy.abs(out_np))
        self.assertLess(
            max_abs,
            50.0,
            f"MEA GQA attention leakage detected: max |output| = {max_abs:.1f}. "
            f"This likely means MEA applies softcap AFTER mask (wrong ordering). "
            f"Correct ordering: QK → softcap → mask → softmax (per onnx/onnx#7865).",
        )

        # Also verify against reference
        out_ref, _ = attention_ref(q=q, k=k, v=v, attn_bias=attn_mask, softcap=softcap_val)
        out_ref_np = out_ref.to(torch.float32).detach().cpu().numpy()
        out_reshaped = torch.reshape(out, (batch_size, q_seq, q_num_heads, head_size))
        out_reshaped_np = out_reshaped.to(torch.float32).detach().cpu().numpy()
        numpy.testing.assert_allclose(out_reshaped_np, out_ref_np, rtol=0.02, atol=0.02)


@unittest.skipIf(not has_flash_attention(), "Flash Attention is not available, skipping Flash GQA softcap tests.")
@patch.dict(os.environ, {"ORT_DISABLE_FLASH_ATTENTION": "0"})
class TestONNXAttentionFlashGQASoftcap(unittest.TestCase):
    """Test softcap support for GQA via the Flash Attention path.

    Flash does NOT accept explicit attn_mask for GQA — uses nonpad_kv_seqlen
    (padding mask) instead. Tests verify softcap works correctly through Flash
    with and without padding mask.

    Requires SM80+ (Flash Attention hardware requirement).
    """

    def test_flash_gqa_softcap_with_padding_mask_prompt_fp16(self):
        """Flash GQA softcap + padding mask (nonpad_kv_seqlen), prompt phase, fp16."""
        config = AttentionConfig(
            batch_size=2,
            q_sequence_length=8,
            kv_sequence_length=8,
            q_num_heads=8,
            kv_num_heads=4,
            head_size=64,
            is_causal=1,
            softcap=50.0,
            has_nonpad_kv_seqlen=True,
        )
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

    def test_flash_gqa_softcap_no_mask_prompt_fp16(self):
        """Flash GQA softcap without any mask, prompt phase, fp16."""
        config = AttentionConfig(
            batch_size=2,
            q_sequence_length=8,
            kv_sequence_length=8,
            q_num_heads=8,
            kv_num_heads=4,
            head_size=64,
            is_causal=1,
            softcap=50.0,
        )
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

    def test_flash_gqa_softcap_no_mask_decode_fp16(self):
        """Flash GQA softcap, decode phase (past KV), fp16."""
        config = AttentionConfig(
            batch_size=2,
            q_sequence_length=1,
            kv_sequence_length=1,
            past_kv_sequence_length=31,
            q_num_heads=8,
            kv_num_heads=4,
            head_size=64,
            is_causal=1,
            softcap=50.0,
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


class TestONNXAttentionCPUSoftcapMaskOrdering(unittest.TestCase):
    """CPU-EP guard tests for ONNX Attention spec ordering (onnx/onnx#7867).

    The spec mandates: scale*QK -> softcap -> add bias/mask -> softmax.
    With a -inf entry in attn_mask and softcap > 0, the wrong order
    (mask-before-softcap) produces tanh(-inf/softcap)*softcap = -softcap
    (a finite value), which leaks probability through softmax to the masked
    position. Combined with a 'poison' V at the masked position, the wrong
    order produces a dramatically wrong output (~poison_value), while the
    correct order produces ~mean(unmasked V).

    These two tests mirror the CUDA-only guards already in this file
    (test_gqa_large_head_unfused_softcap_additive_mask_poison_fp16 at
    line 1501, and test_mea_gqa_softcap_mask_ordering_no_leakage_prompt_fp16
    at line 1761) but force CPUExecutionProvider with fp32. CPU Attention
    does support fp16 (the kernel is registered for MLFloat16), but fp32 is
    the natural EP-native dtype on CPU and makes the leakage math
    arithmetically obvious.

    Pre-fix: both tests FAIL (max |output| ~= 100..1000 due to leak).
    Post-fix: both tests PASS (max |output| < 1, parity vs attention_ref()).
    """

    def test_cpu_attention_softcap_additive_mask_poison_prompt_fp32(self):
        """CPU mirror of test_gqa_large_head_unfused_softcap_additive_mask_poison_fp16."""
        config = AttentionConfig(
            batch_size=1,
            q_sequence_length=1,
            kv_sequence_length=3,
            past_kv_sequence_length=0,
            q_num_heads=8,
            kv_num_heads=4,
            head_size=64,  # smaller than CUDA variant — CPU is slower; keeps test fast
            is_causal=0,
            softcap=1.0,
            has_attn_mask=True,
            attn_mask_dims=4,
            attn_mask_type="additive",
        )

        device = "cpu"
        torch_type = torch.float32
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
        v[:, 1, :, :] = 1000.0  # poison at the position about to be -inf-masked

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
            ep="CPUExecutionProvider",
            device=device,
            ort_type=TensorProto.FLOAT,
        )

        out = out_ort.reshape(
            config.batch_size,
            config.q_sequence_length,
            config.q_num_heads,
            config.head_size,
        )
        max_abs = float(out.abs().max())
        self.assertLess(
            max_abs,
            50.0,
            "CPU attention leakage detected: max |output| = "
            f"{max_abs:.1f}. This means CPU applies softcap AFTER mask-add "
            "(wrong ordering). Correct ordering per onnx/onnx#7867: "
            "scale*QK -> softcap -> add bias/mask -> softmax.",
        )
        expected = torch.full_like(out, 0.2)
        torch.testing.assert_close(out, expected, rtol=0, atol=2e-2)

        # Also verify against the spec-correct reference.
        out_ref, _ = attention_ref(q=q, k=k, v=v, attn_bias=attn_mask, softcap=config.softcap)
        numpy.testing.assert_allclose(
            out.detach().cpu().numpy(),
            out_ref.detach().cpu().numpy(),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_cpu_attention_softcap_mask_ordering_no_leakage_prompt_fp32(self):
        """CPU mirror of test_mea_gqa_softcap_mask_ordering_no_leakage_prompt_fp16.

        CPU has only one Attention compute path (the unified ComputeAttentionProbs
        loop) — there is no MEA/Flash distinction — so this test exercises the
        same loop the production fix targets.
        """
        batch_size = 1
        q_seq = 4
        kv_seq = 8  # divisible by 4 (kept symmetric with the CUDA MEA variant)
        q_num_heads = 4
        kv_num_heads = 2
        head_size = 64
        softcap_val = 2.0
        valid_kv_len = 4

        config = AttentionConfig(
            batch_size=batch_size,
            q_sequence_length=q_seq,
            kv_sequence_length=kv_seq,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
            head_size=head_size,
            is_causal=0,
            softcap=softcap_val,
            has_attn_mask=True,
            attn_mask_dims=4,
            attn_mask_type="additive",
        )

        torch.manual_seed(42)
        device = "cpu"
        torch_type = torch.float32

        q = torch.randn(batch_size, q_seq, q_num_heads, head_size, dtype=torch_type, device=device) * 0.2
        k = torch.randn(batch_size, kv_seq, kv_num_heads, head_size, dtype=torch_type, device=device) * 0.2
        v = torch.randn(batch_size, kv_seq, kv_num_heads, head_size, dtype=torch_type, device=device) * 0.2

        # Poison values in V at masked positions.
        poison_value = 1000.0
        v[:, valid_kv_len:, :, :] = poison_value

        attn_mask = torch.zeros(batch_size, q_num_heads, q_seq, kv_seq, dtype=torch_type, device=device)
        attn_mask[:, :, :, valid_kv_len:] = float("-inf")

        out, _, _ = attention_prompt_func(
            q=q,
            k=k,
            v=v,
            config=config,
            attn_mask=attn_mask,
            ep="CPUExecutionProvider",
            device=device,
            ort_type=TensorProto.FLOAT,
        )

        out_np = out.detach().cpu().numpy().flatten()
        max_abs = float(numpy.max(numpy.abs(out_np)))
        self.assertLess(
            max_abs,
            50.0,
            "CPU attention leakage detected: max |output| = "
            f"{max_abs:.1f}. This means CPU applies softcap AFTER mask-add "
            "(wrong ordering). Correct ordering per onnx/onnx#7867: "
            "scale*QK -> softcap -> add bias/mask -> softmax.",
        )

        out_ref, _ = attention_ref(q=q, k=k, v=v, attn_bias=attn_mask, softcap=softcap_val)
        out_reshaped = torch.reshape(out, (batch_size, q_seq, q_num_heads, head_size))
        numpy.testing.assert_allclose(
            out_reshaped.detach().cpu().numpy(),
            out_ref.detach().cpu().numpy(),
            rtol=2e-2,
            atol=2e-2,
        )

    def test_cpu_attention_qk_matmul_output_mode_post_softcap_with_softcap_fp32(self):
        """Differentiating test for the qk_matmul_output_mode 1<->2 enum value
        swap per onnx/onnx#7913.

        With softcap > 0 active and qk_matmul_output_mode=1 (kPostSoftCap, the
        post-#7913 numbering used here), the qk snapshot must equal
        ``softcap * tanh(scale * Q @ K^T / softcap)`` and MUST NOT include the
        additive mask. Under the pre-#7913 numbering (where mode 1 meant the
        old kQKMask = scale*Q@K^T + mask), the snapshot would contain the
        large negative mask sentinel at masked positions — drastically
        different. Without softcap > 0, mode 1 (post-softcap) aliases mode 0
        (raw QK), so the swap is observationally indistinguishable.

        Forces CPU EP. Mirror of the C++ test
        Attention_QkMatmulOutputMode_PostSoftCap_WithSoftcap_CPU.
        """
        config = AttentionConfig(
            batch_size=1,
            q_sequence_length=1,
            kv_sequence_length=2,
            past_kv_sequence_length=0,
            q_num_heads=1,
            kv_num_heads=1,
            head_size=4,
            is_causal=0,
            softcap=1.0,
            has_attn_mask=True,
            attn_mask_dims=4,
            attn_mask_type="additive",
        )

        device = "cpu"
        torch_type = torch.float32

        # Q = [1, 0, 0, 0]; K[0] = [1, 0, 0, 0]; K[1] = [2, 0, 0, 0]
        # Raw Q @ K^T = [1, 2]; scale = 1/sqrt(4) = 0.5; scale*QK = [0.5, 1.0].
        q = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]], dtype=torch_type, device=device).transpose(1, 2)
        k = torch.tensor(
            [[[[1.0, 0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0, 0.0]]]],
            dtype=torch_type,
            device=device,
        ).transpose(1, 2)
        # V: position 0 = 0.5 across the head_size dim, position 1 = 100 (poison).
        v = torch.tensor(
            [[[[0.5, 0.5, 0.5, 0.5]], [[100.0, 100.0, 100.0, 100.0]]]],
            dtype=torch_type,
            device=device,
        ).transpose(1, 2)

        # Mask -inf at position 1 — would dominate the snapshot under pre-#7913 numbering.
        attn_mask = torch.zeros(
            config.batch_size,
            config.q_num_heads,
            config.q_sequence_length,
            config.kv_sequence_length,
            dtype=torch_type,
            device=device,
        )
        attn_mask[:, :, :, 1] = float("-inf")

        out_ort, _, _, qk_snapshot = attention_prompt_func(
            q=q,
            k=k,
            v=v,
            config=config,
            attn_mask=attn_mask,
            ep="CPUExecutionProvider",
            device=device,
            ort_type=TensorProto.FLOAT,
            output_qk=1,  # kPostSoftCap (post-#7913 numbering)
        )

        # The snapshot must be softcap * tanh(scale * Q @ K^T / softcap) with NO
        # mask contribution. Under pre-#7913 numbering it would have been
        # [0.5, lowest()] (raw QK + mask sentinel) — wildly different.
        scale = 1.0 / math.sqrt(config.head_size)
        raw_qk = numpy.array([1.0, 2.0]) * scale
        expected_snapshot = config.softcap * numpy.tanh(raw_qk / config.softcap)

        snapshot_np = qk_snapshot.detach().cpu().numpy().reshape(-1)
        numpy.testing.assert_allclose(
            snapshot_np,
            expected_snapshot,
            rtol=1e-5,
            atol=1e-5,
            err_msg=(
                "qk_matmul_output_mode=1 snapshot mismatch. Expected post-softcap "
                "(post-#7913 semantics): softcap*tanh(scale*QK/softcap). If the snapshot "
                "instead contains the mask sentinel, the implementation is using the "
                "pre-#7913 numbering where mode 1 meant post-mask/bias."
            ),
        )

        # Position 1 is -inf-masked AFTER softcap, so output should equal V[0] = 0.5.
        out_reshaped = torch.reshape(
            out_ort,
            (config.batch_size, config.q_sequence_length, config.q_num_heads, config.head_size),
        )
        numpy.testing.assert_allclose(
            out_reshaped.detach().cpu().numpy(),
            numpy.full((1, 1, 1, 4), 0.5, dtype=numpy.float32),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_cpu_attention_softcap_nonpad_kv_seqlen_no_leakage_prompt_fp32(self):
        """Bonus latent fix: with softcap > 0, the nonpad_kv_seqlen sentinel
        is now applied AFTER softcap (per onnx/onnx#7867 ordering).

        Under the pre-fix ordering the nonpad sentinel would be squashed by
        tanh into ~-softcap, leaking probability through softmax to padded
        positions. With poison V at padded positions this would dominate the
        output. Spec-correct ordering keeps the output bounded.

        Forces CPU EP. Mirror of the C++ test
        Attention_NonPadKVSeqLen_WithSoftcap_NoLeakage_CPU.
        """
        config = AttentionConfig(
            batch_size=1,
            q_sequence_length=1,
            kv_sequence_length=4,
            past_kv_sequence_length=0,
            q_num_heads=1,
            kv_num_heads=1,
            head_size=2,
            is_causal=0,
            softcap=1.0,
            has_attn_mask=False,
            has_nonpad_kv_seqlen=True,
        )
        valid_kv_len = 2

        device = "cpu"
        torch_type = torch.float32

        # Uniform Q,K so all 4 raw scores equal -> uniform attention sans nonpad masking.
        q = torch.ones(
            config.batch_size,
            config.q_sequence_length,
            config.q_num_heads,
            config.head_size,
            dtype=torch_type,
            device=device,
        )
        k = torch.ones(
            config.batch_size,
            config.kv_sequence_length,
            config.kv_num_heads,
            config.head_size,
            dtype=torch_type,
            device=device,
        )
        # V: first valid_kv_len positions = 1.0, padded positions = 1000.0 (poison).
        v = torch.full_like(k, 1.0)
        v[:, valid_kv_len:, :, :] = 1000.0

        nonpad_kv_seqlen = torch.tensor([valid_kv_len], dtype=torch.int64, device=device)

        out_ort, _, _ = attention_prompt_func(
            q=q,
            k=k,
            v=v,
            config=config,
            attn_mask=None,
            ep="CPUExecutionProvider",
            device=device,
            ort_type=TensorProto.FLOAT,
            nonpad_kv_seqlen=nonpad_kv_seqlen,
        )

        out_reshaped = torch.reshape(
            out_ort,
            (config.batch_size, config.q_sequence_length, config.q_num_heads, config.head_size),
        )
        max_abs = float(out_reshaped.abs().max())
        self.assertLess(
            max_abs,
            50.0,
            "CPU attention leakage detected: max |output| = "
            f"{max_abs:.1f}. With softcap > 0, the nonpad_kv_seqlen sentinel "
            "must be applied AFTER softcap (onnx/onnx#7867); otherwise tanh "
            "squashes the sentinel into a finite value and poison V at padded "
            "positions leaks through softmax.",
        )
        # Spec-correct: uniform attention over the 2 valid positions, V=1.0.
        numpy.testing.assert_allclose(
            out_reshaped.detach().cpu().numpy(),
            numpy.ones((1, 1, 1, 2), dtype=numpy.float32),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_cpu_attention_qk_matmul_output_mode_post_mask_bias_with_softcap_and_nonpad_fp32(self):
        """Pin the kPostMaskBias x softcap x nonpad_kv_seqlen coverage matrix cell. The kPostMaskBias snapshot (qk_matmul_output_mode == 2 in the
        post-#7913 numbering) is taken AFTER softcap, after `attn_mask` is added,
        AND after `nonpad_kv_seqlen` fills padded positions with the finite
        `mask_filter_value<float>() == std::numeric_limits<float>::lowest()`
        sentinel (CPU softmax expects only finite inputs; see attention.h).

        Forces CPU EP. Mirror of the C++ test
        Attention_QkMatmulOutputMode_PostMaskBias_WithSoftcapAndNonpad_CPU.
        """
        config = AttentionConfig(
            batch_size=1,
            q_sequence_length=1,
            kv_sequence_length=4,
            past_kv_sequence_length=0,
            q_num_heads=1,
            kv_num_heads=1,
            head_size=4,
            is_causal=0,
            softcap=1.0,
            has_attn_mask=True,
            attn_mask_dims=2,
            attn_mask_type="additive",
            has_nonpad_kv_seqlen=True,
        )
        valid_kv_len = 2
        # Mode 2 (post-#7913) = kPostMaskBias = post-mask/bias, pre-softmax.
        output_qk_mode = 2

        device = "cpu"
        torch_type = torch.float32

        # Q = [1, 0, 0, 0]; K[i] = [a_i, 0, 0, 0] -> raw scale*QK = [0.5, 1.0, 0.5, 0.5]
        q = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]], dtype=torch_type, device=device)
        k = torch.zeros(1, 4, 1, 4, dtype=torch_type, device=device)
        k[0, 0, 0, 0] = 1.0
        k[0, 1, 0, 0] = 2.0
        k[0, 2, 0, 0] = 1.0
        k[0, 3, 0, 0] = 1.0
        # V poison at padded positions (indices 2, 3) must NOT leak.
        v = torch.ones(1, 4, 1, 4, dtype=torch_type, device=device)
        v[:, valid_kv_len:, :, :] = 1000.0

        # Mask large finite negative at valid position 1; sidesteps -inf
        # comparison concerns in the snapshot, equivalent in softmax behaviour.
        large_neg = -1.0e9
        attn_mask = torch.tensor([[0.0, large_neg, 0.0, 0.0]], dtype=torch_type, device=device)
        nonpad_kv_seqlen = torch.tensor([valid_kv_len], dtype=torch.int64, device=device)

        out_ort, _, _, qk_ort = attention_prompt_func(
            q=q,
            k=k,
            v=v,
            config=config,
            attn_mask=attn_mask,
            ep="CPUExecutionProvider",
            device=device,
            ort_type=TensorProto.FLOAT,
            nonpad_kv_seqlen=nonpad_kv_seqlen,
            output_qk=output_qk_mode,
        )

        # ---- Snapshot pin (the matrix cell under test) ----
        qk_np = qk_ort.detach().cpu().numpy().reshape(-1)
        self.assertEqual(
            qk_np.shape,
            (4,),
            f"output_qk shape must be [batch, q_num_heads, q_seq, kv_seq] = [1,1,1,4]; got {qk_ort.shape}",
        )
        # Print a few values to evidence the matrix cell is genuinely exercised.
        print(
            f"\n[kPostMaskBias x softcap x nonpad snapshot] "
            f"valid[0]={qk_np[0]:.6f} (expect ~tanh(0.5)={math.tanh(0.5):.6f}), "
            f"valid[1]={qk_np[1]:.3e} (expect ~{large_neg:.0e}), "
            f"padded[2]={qk_np[2]:.3e} (expect lowest()=~{numpy.finfo(numpy.float32).min:.3e}), "
            f"padded[3]={qk_np[3]:.3e}"
        )
        # Position 0 valid: tanh(scale*QK[0]) = tanh(0.5)
        self.assertAlmostEqual(float(qk_np[0]), math.tanh(0.5), places=5)
        # Position 1 valid + masked: tanh(1.0) + large_neg ~= large_neg
        self.assertLess(float(qk_np[1]), -1e8, "position-1 snapshot must contain the additive attn_mask")
        # Positions 2, 3 padded: lowest() sentinel from nonpad_kv_seqlen
        self.assertLess(float(qk_np[2]), -1e30, "position-2 must be the lowest() nonpad sentinel")
        self.assertLess(float(qk_np[3]), -1e30, "position-3 must be the lowest() nonpad sentinel")

        # ---- Final Y bound (no V leakage) ----
        out_reshaped = torch.reshape(out_ort, (1, 1, 1, 4))
        out_np = out_reshaped.detach().cpu().numpy()
        # Only valid position 0 wins softmax; Y should be V[0] = [1, 1, 1, 1].
        numpy.testing.assert_allclose(out_np, numpy.ones((1, 1, 1, 4), dtype=numpy.float32), rtol=1e-4, atol=1e-4)


@unittest.skipIf(not has_cuda_device(53), "CUDA EP is not available, skipping tests.")
@patch.dict(os.environ, {"ORT_DISABLE_FLASH_ATTENTION": "1"})
class TestONNXAttentionGQAAsymmetricHeadSize(unittest.TestCase):
    """
    Regression tests for GQA + asymmetric Q/V head sizes (head_size != v_head_size).

    Guards against the silent-broken-output regression that was fixed by PR #28358
    (microsoft/onnxruntime#28357). Before #28358, the GQA + MEA path's
    LaunchUngroup helper (used by MEA to expand K/V heads before the FMHA kernel)
    ENFORCEd head_size == v_head_size, hard-erroring at runtime, and the MEA
    eligibility predicate did not exclude the asymmetric case, leading to
    NaN / OOB reads when MEA was attempted on an asymmetric V tile.

    These tests pin down the post-fix behaviour by running an asymmetric-GQA
    config (q_num_heads=8, kv_num_heads=1, q/k head_size=32, v_head_size=64,
    self-attention seq_len=4) on both fp16 and bf16 and asserting numerical
    parity with the reference.

    Asymmetric GQA always falls through to the unfused path on CUDA per the
    (!is_gqa || head_size == v_head_size) clause of the MEA eligibility
    predicate at core/providers/cuda/llm/attention.cc.
    """

    def _run_asymmetric_gqa_prompt(self, torch_type, ort_type):
        config = AttentionConfig(
            batch_size=1,
            q_sequence_length=4,
            kv_sequence_length=4,
            q_num_heads=8,
            kv_num_heads=1,  # MQA: kv_num_heads=1, q_num_heads=8
            head_size=32,  # small so the test runs fast on H100
            v_head_size=64,  # asymmetric: V head twice as large as Q/K head
            is_causal=1,
        )

        torch.manual_seed(0)
        device = "cuda"
        std = 0.2

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
        v = (
            torch.randn(
                config.batch_size,
                config.kv_sequence_length,
                config.kv_num_heads,
                config.v_head_size,
                device=device,
                dtype=torch_type,
            )
            * std
        )

        out_ref, _ = attention_ref(q=q, k=k, v=v, causal=True)
        out_ort, _, _ = attention_prompt_func(
            q=q,
            k=k,
            v=v,
            config=config,
            attn_mask=None,
            ep="CUDAExecutionProvider",
            device=device,
            ort_type=ort_type,
        )

        out_ort = torch.reshape(
            out_ort,
            (config.batch_size, config.q_sequence_length, config.q_num_heads, config.v_head_size),
        )

        out_np = out_ort.to(torch.float32).detach().cpu().numpy()
        out_ref_np = out_ref.to(torch.float32).detach().cpu().numpy()
        # Sanity: no NaN propagation from the previously-broken asymmetric path.
        self.assertFalse(numpy.isnan(out_np).any(), "NaN in output — asymmetric GQA path regressed")
        # fp16/bf16 attention has wide tolerance bands when reductions are reordered.
        atol_key = "fp16" if torch_type == torch.float16 else "bf16"
        rtol_key = atol_key
        numpy.testing.assert_allclose(out_np, out_ref_np, rtol=rtol[rtol_key], atol=atol[atol_key])

    def test_gqa_asymmetric_v_head_size_prompt_fp16(self):
        self._run_asymmetric_gqa_prompt(torch.float16, TensorProto.FLOAT16)

    def test_gqa_asymmetric_v_head_size_prompt_bf16(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BFloat16 not supported on this device")
        self._run_asymmetric_gqa_prompt(torch.bfloat16, TensorProto.BFLOAT16)


class TestONNXAttentionGQAAsymmetricHeadSizeCPU(unittest.TestCase):
    """CPU twin of TestONNXAttentionGQAAsymmetricHeadSize.

    Pins post-#28358 behaviour on CPU as well: asymmetric Q/V head sizes
    (head_size != v_head_size) on a GQA shape must produce numerically
    parity-clean output with no NaN propagation. fp16 only (no bf16 twin
    by design — broad CPU bf16 attention is not in scope for these guards).
    """

    def _run_asymmetric_gqa_prompt_cpu(self, torch_type, ort_type, atol_key):
        config = AttentionConfig(
            batch_size=1,
            q_sequence_length=4,
            kv_sequence_length=4,
            q_num_heads=8,
            kv_num_heads=1,
            head_size=32,
            v_head_size=64,
            is_causal=1,
        )

        torch.manual_seed(0)
        device = "cpu"
        std = 0.2

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
        v = (
            torch.randn(
                config.batch_size,
                config.kv_sequence_length,
                config.kv_num_heads,
                config.v_head_size,
                device=device,
                dtype=torch_type,
            )
            * std
        )

        out_ref, _ = attention_ref(q=q, k=k, v=v, causal=True)
        out_ort, _, _ = attention_prompt_func(
            q=q,
            k=k,
            v=v,
            config=config,
            attn_mask=None,
            ep="CPUExecutionProvider",
            device=device,
            ort_type=ort_type,
        )

        out_ort = torch.reshape(
            out_ort,
            (config.batch_size, config.q_sequence_length, config.q_num_heads, config.v_head_size),
        )

        out_np = out_ort.to(torch.float32).detach().cpu().numpy()
        out_ref_np = out_ref.to(torch.float32).detach().cpu().numpy()
        self.assertFalse(numpy.isnan(out_np).any(), "NaN in output - asymmetric GQA path regressed on CPU")
        numpy.testing.assert_allclose(out_np, out_ref_np, rtol=rtol[atol_key], atol=atol[atol_key])

    def test_gqa_asymmetric_v_head_size_prompt_cpu_fp16(self):
        self._run_asymmetric_gqa_prompt_cpu(torch.float16, TensorProto.FLOAT16, "fp16")

    def test_gqa_asymmetric_v_head_size_prompt_cpu_fp32(self):
        self._run_asymmetric_gqa_prompt_cpu(torch.float32, TensorProto.FLOAT, "fp32")


@unittest.skipIf(not has_cuda_device(53), "CUDA EP is not available, skipping tests.")
@patch.dict(os.environ, {"ORT_DISABLE_FLASH_ATTENTION": "1"})
class TestONNXAttentionGQAOutputQK(unittest.TestCase):
    """
    Tests that GQA + qk_matmul_output_mode == 0 (raw QK output) works.

    Issue #28351 sub-item 1c: the output_qk path was implemented in the unfused
    kernel but lacked test coverage for the GQA + raw-QK combination. The
    output_qk shape is [batch, q_num_heads, q_seq, total_seq]; the unfused
    kernel indexes per Q-head, and attention_helper.h infers the shape from
    q_num_heads, so this combination should already work — these tests pin it.

    Note: GQA + MEA on CUDA requires fp16/bf16 because the MEA `LaunchUngroup`
    helper has no fp32 instantiation; the GQA unfused fall-through DOES
    support fp32 (exercised by `TestONNXAttentionGQASoftcapFloat32MaskOrdering`
    below). These tests pin the fp16 + raw-QK + GQA combination on the
    unfused path.
    """

    def test_gqa_output_qk_raw_prompt_fp16(self):
        config = AttentionConfig(
            batch_size=1,
            q_sequence_length=4,
            kv_sequence_length=4,
            q_num_heads=8,
            kv_num_heads=2,
            head_size=32,
            is_causal=1,
        )

        torch.manual_seed(0)
        device = "cuda"
        torch_type = torch.float16
        ort_type = TensorProto.FLOAT16
        std = 0.2

        q = torch.randn(1, 4, 8, 32, device=device, dtype=torch_type) * std
        k = torch.randn(1, 4, 2, 32, device=device, dtype=torch_type) * std
        v = torch.randn(1, 4, 2, 32, device=device, dtype=torch_type) * std

        # Reference output_qk: raw scaled QK (no mask, no softcap, no softmax).
        # Mode kQK == 0 outputs raw Q*K^T / sqrt(d) as the spec defines.
        q_f, k_f = q.float(), k.float()
        # Repeat K heads for GQA (kv_num_heads=2, q_num_heads=8 -> repeat factor 4)
        k_rep = k_f.repeat_interleave(q.shape[2] // k.shape[2], dim=2)
        ref_qk = torch.einsum("bthd,bshd->bhts", q_f, k_rep) / math.sqrt(q.shape[-1])

        # Run ORT with output_qk=0 (kQK in the post-#7913 enum: raw scaled QK).
        _out_ort, _, _, qk_ort = attention_prompt_func(
            q=q,
            k=k,
            v=v,
            config=config,
            attn_mask=None,
            ep="CUDAExecutionProvider",
            device=device,
            ort_type=ort_type,
            output_qk=0,  # kQK: raw scaled QK
        )

        qk_np = qk_ort.to(torch.float32).detach().cpu().numpy()
        ref_qk_np = ref_qk.detach().cpu().numpy()
        self.assertFalse(numpy.isnan(qk_np).any(), "NaN in output_qk")
        self.assertEqual(
            qk_np.shape,
            (1, 8, 4, 4),
            "output_qk shape must be [batch, q_num_heads, q_seq, total_seq]",
        )
        numpy.testing.assert_allclose(qk_np, ref_qk_np, rtol=rtol["fp16"], atol=atol["fp16"])


class TestONNXAttentionGQAOutputQKCPU(unittest.TestCase):
    """CPU twin of TestONNXAttentionGQAOutputQK -- expanded to all 4 modes.

    The CUDA source class only exercises mode 0 (kQK, raw scaled QK). On CPU
    (post-#28379 spec fix and post-#7913 enum swap) all 4 modes are
    supported, so this twin parameterizes over kQK=0, kPostSoftCap=1,
    kPostMaskBias=2, kPostSoftMax=3 and computes the per-mode reference
    snapshot directly from torch ops. fp32 is used so the comparison is
    deterministic and the per-mode arithmetic ladder is checked tightly.

    Pipeline (per attention.cc CPU EP, see SKILL.md SS4):
       raw = scale * Q @ K^T
       mode 0 (kQK)          = raw
       mode 1 (kPostSoftCap) = softcap * tanh(raw / softcap)
       mode 2 (kPostMaskBias)= mode1 + attn_mask
       mode 3 (kPostSoftMax) = softmax(mode2, dim=-1)
    """

    @parameterized.expand([("kQK", 0), ("kPostSoftCap", 1), ("kPostMaskBias", 2), ("kPostSoftMax", 3)])
    def test_gqa_output_qk_all_modes_cpu_fp32(self, _name, mode):
        config = AttentionConfig(
            batch_size=1,
            q_sequence_length=2,
            kv_sequence_length=4,
            past_kv_sequence_length=0,
            q_num_heads=4,
            kv_num_heads=2,
            head_size=8,
            is_causal=0,
            softcap=2.0,
            has_attn_mask=True,
            attn_mask_dims=4,
            attn_mask_type="additive",
        )

        torch.manual_seed(0)
        device = "cpu"
        torch_type = torch.float32
        ort_type = TensorProto.FLOAT
        std = 0.5

        q = torch.randn(1, 2, 4, 8, device=device, dtype=torch_type) * std
        k = torch.randn(1, 4, 2, 8, device=device, dtype=torch_type) * std
        v = torch.randn(1, 4, 2, 8, device=device, dtype=torch_type) * std

        # Mask one valid position with a large finite negative (CPU softmax
        # requires finite inputs; -inf is replaced by mask_filter_value()
        # internally for nonpad, but for explicit attn_mask we keep it
        # finite to keep the per-mode arithmetic check tractable).
        attn_mask = torch.zeros(1, config.q_num_heads, 2, 4, device=device, dtype=torch_type)
        attn_mask[:, :, :, 2] = -1.0e6

        # Per-mode reference, computed directly from torch ops.
        scale = 1.0 / math.sqrt(config.head_size)
        # GQA: repeat KV heads (kv_num_heads=2 -> q_num_heads=4, factor 2).
        k_rep = k.repeat_interleave(config.q_num_heads // config.kv_num_heads, dim=2)
        # raw[b,h,t,s] = scale * sum_d Q[b,t,h,d] * K_rep[b,s,h,d]
        raw = scale * torch.einsum("bthd,bshd->bhts", q, k_rep)
        if mode == 0:
            ref_qk = raw
        else:
            after_softcap = config.softcap * torch.tanh(raw / config.softcap)
            if mode == 1:
                ref_qk = after_softcap
            else:
                after_mask = after_softcap + attn_mask
                if mode == 2:
                    ref_qk = after_mask
                else:  # mode == 3
                    ref_qk = torch.softmax(after_mask, dim=-1)

        _out_ort, _, _, qk_ort = attention_prompt_func(
            q=q,
            k=k,
            v=v,
            config=config,
            attn_mask=attn_mask,
            ep="CPUExecutionProvider",
            device=device,
            ort_type=ort_type,
            output_qk=mode,
        )

        qk_np = qk_ort.to(torch.float32).detach().cpu().numpy()
        ref_qk_np = ref_qk.detach().cpu().numpy()
        self.assertFalse(numpy.isnan(qk_np).any(), f"NaN in output_qk for mode {mode}")
        self.assertEqual(
            qk_np.shape,
            (1, config.q_num_heads, 2, 4),
            f"output_qk shape must be [batch, q_num_heads, q_seq, kv_seq]; got {qk_ort.shape} for mode {mode}",
        )
        numpy.testing.assert_allclose(
            qk_np,
            ref_qk_np,
            rtol=rtol["fp32"],
            atol=atol["fp32"],
            err_msg=f"output_qk parity failed for mode {mode}",
        )


@unittest.skipIf(not has_cuda_device(53), "CUDA EP is not available, skipping tests.")
@patch.dict(os.environ, {"ORT_DISABLE_FLASH_ATTENTION": "1"})
class TestONNXAttentionGQASoftcapFloat32(unittest.TestCase):
    """
    Issue #28351 sub-item 1e: softcap coverage for the fp32 path.

    fp32 GQA on CUDA always falls through to the unfused path (the MEA
    predicate at attention.cc explicitly excludes is_gqa && std::is_same<T,
    float>::value because LaunchUngroup has no fp32 instantiation). Existing
    softcap tests are fp16/bf16; this class pins the fp32 + softcap +
    asymmetric-or-symmetric-GQA combination so future kernel changes can't
    silently break the unfused softcap branch for fp32.

    Sibling class `TestONNXAttentionGQASoftcapFloat32MaskOrdering` below
    additionally pins softcap+mask ORDERING on the fp32 path (poison-V
    pattern). The unmasked tests here cannot detect a wrong order — softcap
    and mask only diverge when both are present.
    """

    def _run_softcap_fp32(self, head_size, v_head_size=None):
        # v_head_size=None means "same as head_size" (symmetric V).
        effective_v_head_size = v_head_size if v_head_size is not None else head_size
        config = AttentionConfig(
            batch_size=1,
            q_sequence_length=4,
            kv_sequence_length=4,  # GQA on CUDA requires self-attention
            q_num_heads=4,
            kv_num_heads=2,
            head_size=head_size,
            # AttentionConfig.v_head_size uses 0 as the "same as head_size" sentinel
            # (defined in common.py); translate from the test-local None convention.
            v_head_size=v_head_size if v_head_size is not None else 0,
            is_causal=1,
            softcap=2.0,  # small softcap exposes ordering / clipping issues
        )

        torch.manual_seed(0)
        device = "cuda"
        torch_type = torch.float32
        ort_type = TensorProto.FLOAT
        std = 0.5

        q = torch.randn(1, 4, 4, head_size, device=device, dtype=torch_type) * std
        k = torch.randn(1, 4, 2, head_size, device=device, dtype=torch_type) * std
        v = torch.randn(1, 4, 2, effective_v_head_size, device=device, dtype=torch_type) * std

        out_ref, _ = attention_ref(q=q, k=k, v=v, causal=True, softcap=2.0)
        out_ort, _, _ = attention_prompt_func(
            q=q,
            k=k,
            v=v,
            config=config,
            attn_mask=None,
            ep="CUDAExecutionProvider",
            device=device,
            ort_type=ort_type,
        )
        out_ort = torch.reshape(out_ort, (1, 4, 4, effective_v_head_size))

        out_np = out_ort.to(torch.float32).detach().cpu().numpy()
        out_ref_np = out_ref.to(torch.float32).detach().cpu().numpy()
        self.assertFalse(numpy.isnan(out_np).any())
        numpy.testing.assert_allclose(out_np, out_ref_np, rtol=rtol["fp32"], atol=atol["fp32"])

    def test_gqa_softcap_fp32_symmetric(self):
        self._run_softcap_fp32(head_size=16, v_head_size=None)

    def test_gqa_softcap_fp32_asymmetric_v_head(self):
        self._run_softcap_fp32(head_size=16, v_head_size=32)


class TestONNXAttentionGQASoftcapFloat32CPU(unittest.TestCase):
    """CPU twin of TestONNXAttentionGQASoftcapFloat32.

    fp32 + softcap + GQA on the CPU EP. Same shapes as the CUDA source so
    the unmasked-softcap parity check stays paired across EPs.
    """

    def _run_softcap_fp32_cpu(self, head_size, v_head_size=None):
        effective_v_head_size = v_head_size if v_head_size is not None else head_size
        config = AttentionConfig(
            batch_size=1,
            q_sequence_length=4,
            kv_sequence_length=4,
            q_num_heads=4,
            kv_num_heads=2,
            head_size=head_size,
            v_head_size=v_head_size if v_head_size is not None else 0,
            is_causal=1,
            softcap=2.0,
        )

        torch.manual_seed(0)
        device = "cpu"
        torch_type = torch.float32
        ort_type = TensorProto.FLOAT
        std = 0.5

        q = torch.randn(1, 4, 4, head_size, device=device, dtype=torch_type) * std
        k = torch.randn(1, 4, 2, head_size, device=device, dtype=torch_type) * std
        v = torch.randn(1, 4, 2, effective_v_head_size, device=device, dtype=torch_type) * std

        out_ref, _ = attention_ref(q=q, k=k, v=v, causal=True, softcap=2.0)
        out_ort, _, _ = attention_prompt_func(
            q=q,
            k=k,
            v=v,
            config=config,
            attn_mask=None,
            ep="CPUExecutionProvider",
            device=device,
            ort_type=ort_type,
        )
        out_ort = torch.reshape(out_ort, (1, 4, 4, effective_v_head_size))

        out_np = out_ort.to(torch.float32).detach().cpu().numpy()
        out_ref_np = out_ref.to(torch.float32).detach().cpu().numpy()
        self.assertFalse(numpy.isnan(out_np).any())
        numpy.testing.assert_allclose(out_np, out_ref_np, rtol=rtol["fp32"], atol=atol["fp32"])

    def test_gqa_softcap_fp32_symmetric_cpu(self):
        self._run_softcap_fp32_cpu(head_size=16, v_head_size=None)

    def test_gqa_softcap_fp32_asymmetric_v_head_cpu(self):
        self._run_softcap_fp32_cpu(head_size=16, v_head_size=32)


@unittest.skipIf(not has_cuda_device(53), "CUDA EP is not available, skipping tests.")
@patch.dict(os.environ, {"ORT_DISABLE_FLASH_ATTENTION": "1"})
class TestONNXAttentionGQASoftcapFloat32MaskOrdering(unittest.TestCase):
    """
    Pin softcap+mask ORDERING on the fp32 unfused GQA path (post-onnx#7867
    spec semantics: scale -> softcap -> +mask -> softmax).

    The unmasked fp32 GQA softcap baseline tests live in the sibling class
    `TestONNXAttentionGQASoftcapFloat32` above — those exercise softcap on
    the unfused fp32 path but cannot detect a wrong ordering of softcap vs
    additive mask, since without a mask the two orderings are arithmetically
    identical. The tests below pin the masked ordering using the same
    poison-V pattern as the fp16/bf16 P1 ordering guards
    (test_gqa_large_head_unfused_softcap_additive_mask_poison_fp16).

    These tests live alongside the CPU spec fix (PR #28379) because they
    semantically depend on the same correct softcap/mask ordering and the
    same post-#7913 enum numbering.
    """

    def _run_softcap_fp32_with_mask(self, head_size, v_head_size=None):
        """
        Pin softcap+mask ORDERING on the fp32 unfused path.

          - Tiny softcap (2.0) so it would clamp very large logits.
          - V values = 1000.0 in the masked KV slot, 0.2 elsewhere.
          - attn_mask = -inf for the masked slot, 0 elsewhere.

        Correct order (QK -> softcap -> +mask -> softmax) zeroes out the
        masked logit via softmax, so output ~= 0.2. Wrong order (mask before
        softcap) would feed -inf through softcap and either clamp it to a
        finite value (allowing the poisoned V to leak) or produce NaN.
        """
        effective_v_head_size = v_head_size if v_head_size is not None else head_size
        config = AttentionConfig(
            batch_size=1,
            q_sequence_length=1,
            kv_sequence_length=3,
            q_num_heads=4,
            kv_num_heads=2,
            head_size=head_size,
            v_head_size=v_head_size if v_head_size is not None else 0,
            is_causal=0,
            softcap=2.0,
            has_attn_mask=True,
            attn_mask_dims=4,
            attn_mask_type="additive",
        )

        device = "cuda"
        torch_type = torch.float32
        ort_type = TensorProto.FLOAT

        q = torch.zeros(1, 1, 4, head_size, device=device, dtype=torch_type)
        k = torch.zeros(1, 3, 2, head_size, device=device, dtype=torch_type)
        v = torch.full((1, 3, 2, effective_v_head_size), 0.2, device=device, dtype=torch_type)
        v[:, 1, :, :] = 1000.0  # poison the masked slot

        attn_mask = torch.zeros(1, 4, 1, 3, device=device, dtype=torch_type)
        attn_mask[:, :, :, 1] = float("-inf")

        out_ref, _ = attention_ref(q=q, k=k, v=v, attn_bias=attn_mask, softcap=2.0)
        out_ort, _, _ = attention_prompt_func(
            q=q,
            k=k,
            v=v,
            config=config,
            attn_mask=attn_mask,
            ep="CUDAExecutionProvider",
            device=device,
            ort_type=ort_type,
        )
        out = out_ort.reshape(1, 1, 4, effective_v_head_size)

        out_np = out.to(torch.float32).detach().cpu().numpy()
        out_ref_np = out_ref.to(torch.float32).detach().cpu().numpy()

        self.assertFalse(
            numpy.isnan(out_np).any(),
            "NaN in fp32 GQA softcap+mask output — wrong softcap/mask ordering on the unfused fp32 path?",
        )
        max_abs = numpy.max(numpy.abs(out_np))
        self.assertLess(
            max_abs,
            1.0,
            f"fp32 GQA softcap+mask leakage: max |output| = {max_abs:.3f}. "
            f"Expected ~0.2 (mask zeroes out the poisoned V=1000 slot via softmax). "
            f"Wrong ordering (mask before softcap) would let the -inf get clamped "
            f"by softcap and the poisoned V to leak through.",
        )
        numpy.testing.assert_allclose(out_np, out_ref_np, rtol=rtol["fp32"], atol=atol["fp32"])

    def test_gqa_softcap_fp32_with_mask_ordering_symmetric(self):
        self._run_softcap_fp32_with_mask(head_size=16, v_head_size=None)

    def test_gqa_softcap_fp32_with_mask_ordering_asymmetric_v_head(self):
        self._run_softcap_fp32_with_mask(head_size=16, v_head_size=32)


if __name__ == "__main__":
    unittest.main()
