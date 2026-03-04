# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Tests for ONNX Attention op (opset 23) — MHA path (kv_num_heads == q_num_heads).

The MHA path in attention.cc is exercised when kv_num_heads == q_num_heads.
It uses the unfused attention kernel and supports:
  - float32, float16, bfloat16
  - 3D inputs (BSNH format)
  - Causal and non-causal attention
  - Self-attention and cross-attention (kv_seq != q_seq)
  - Additive attention bias (NOT boolean masks)
  - Past KV cache
  - 2D, 3D, 4D additive masks with broadcasting
"""

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

    # Create 2D key_padding_mask for reference (per-batch, shape [batch, total_seq])
    key_padding_mask = create_boolean_mask_from_seqlens(
        seqlens=effective_seqlens,
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


if __name__ == "__main__":
    unittest.main()
