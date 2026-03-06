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

    # Zero out padded positions in K, V for proper comparison
    for b in range(config.batch_size):
        valid_len = seqlens[b].item()
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
    for b in range(config.batch_size):
        valid_len = seqlens[b].item()
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
    softcap_opts = [0.0]

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
class TestONNXAttentionFlashGQA(unittest.TestCase):
    """Test ONNX Attention op (opset 23) GQA path with Flash Attention."""

    @parameterized.expand(gqa_prompt_test_cases())
    def test_gqa_prompt_flash(self, name, config):
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "0"
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
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "0"
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
class TestONNXAttentionFlashGQABF16(unittest.TestCase):
    """Test ONNX Attention op (opset 23) GQA path with Flash Attention using BFloat16."""

    @parameterized.expand(gqa_prompt_test_cases())
    def test_gqa_prompt_flash_bf16(self, name, config):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BFloat16 not supported on this device")

        config.kv_cache_type = "bfloat16"
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "0"
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
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "0"
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
class TestONNXAttentionMemoryEfficientGQA(unittest.TestCase):
    """Test ONNX Attention op (opset 23) GQA path with Memory Efficient Attention."""

    @parameterized.expand(gqa_prompt_test_cases())
    def test_gqa_prompt_memory_efficient(self, name, config):
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "1"
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
    def test_gqa_past_memory_efficient(self, name, config):
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "1"
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
class TestONNXAttentionMemoryEfficientGQABF16(unittest.TestCase):
    """Test ONNX Attention op (opset 23) GQA path with Memory Efficient Attention using BFloat16."""

    @parameterized.expand(gqa_past_test_cases())
    def test_gqa_past_memory_efficient_bf16(self, name, config):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BFloat16 not supported on this device")

        config.kv_cache_type = "bfloat16"
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "1"
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
class TestONNXAttentionPaddingMaskGQA(unittest.TestCase):
    """
    Test ONNX Attention op (opset 23) GQA path with boolean padding masks.

    These tests verify that the boolean attn_mask is correctly converted to
    sequence lengths on GPU and that the attention computation respects the
    padding. Tests cover 2D, 3D, and 4D mask shapes.
    """

    @parameterized.expand(gqa_prompt_padding_test_cases())
    def test_gqa_prompt_padding_flash(self, name, config):
        """Test prompt phase with padding mask using Flash Attention."""
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "0"

        seqlens = torch.tensor(
            [config.kv_sequence_length - 6, config.kv_sequence_length],
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

    @parameterized.expand(gqa_past_padding_test_cases())
    def test_gqa_past_padding_flash(self, name, config):
        """Test decoding phase with padding mask using Flash Attention."""
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "0"

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
class TestONNXAttentionPaddingMaskMemoryEfficientGQA(unittest.TestCase):
    """
    Test ONNX Attention op (opset 23) GQA path with boolean padding masks
    using Memory Efficient Attention.
    """

    @parameterized.expand(gqa_prompt_padding_test_cases())
    def test_gqa_prompt_padding_mea(self, name, config):
        """Test prompt phase with padding mask using Memory Efficient Attention."""
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "1"

        seqlens = torch.tensor(
            [config.kv_sequence_length - 6, config.kv_sequence_length],
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

    @parameterized.expand(gqa_past_padding_test_cases())
    def test_gqa_past_padding_mea(self, name, config):
        """Test decoding phase with padding mask using Memory Efficient Attention."""
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "1"

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


if __name__ == "__main__":
    unittest.main()
