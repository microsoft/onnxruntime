# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Test MultiHeadAttention operator for CUDA and CPU.
"""

import concurrent.futures
import itertools
import os
import unittest
from typing import Dict, List, Optional

import numpy
import torch
from benchmark_mha import (
    AttentionMaskFormat,
    InputFormats,
    MultiHeadAttentionConfig,
    OrtMultiHeadAttention,
    SdpaKernel,
    create_ort_session,
)
from einops import rearrange

import onnxruntime


def get_provider_support_info(provider: str, use_kv_cache: bool):
    if provider == "CUDAExecutionProvider":
        if not use_kv_cache:
            formats = [
                InputFormats.Q_K_V_BSNH_BSNH_BSNH,
                InputFormats.Q_KV_BSNH_BSN2H,
                InputFormats.QKV_BSN3H,
                InputFormats.Q_K_V_BSNH_BNSH_BNSH,
            ]
        else:
            formats = [InputFormats.Q_K_V_BSNH_BSNH_BSNH]

        device_id = torch.cuda.current_device()
        device = torch.device("cuda", device_id)
        dtype = torch.float16
    else:
        assert provider == "CPUExecutionProvider"
        formats = [InputFormats.Q_K_V_BSNH_BSNH_BSNH]
        if not use_kv_cache:
            formats.append(InputFormats.Q_K_V_BSNH_BNSH_BNSH)
        device = torch.device("cpu")
        dtype = torch.float
    return device, dtype, formats


def get_bias_support(format: InputFormats):
    if format == InputFormats.Q_K_V_BSNH_BSNH_BSNH:
        return [True, False]

    if format == InputFormats.Q_K_V_BSNH_BNSH_BNSH:
        return [True, False]

    if format == InputFormats.Q_KV_BSNH_BSN2H:
        return [False]

    if format == InputFormats.QKV_BSN3H:
        return [True, False]

    raise RuntimeError(f"Unknown format: {format}")


def get_causal_support(format: InputFormats):
    if format == InputFormats.Q_K_V_BSNH_BSNH_BSNH:
        return [True, False]

    if format == InputFormats.Q_K_V_BSNH_BNSH_BNSH:
        return [True, False]

    if format == InputFormats.Q_KV_BSNH_BSN2H:
        return [True, False]

    if format == InputFormats.QKV_BSN3H:
        return [True, False]

    raise RuntimeError(f"Unknown format: {format}")


def get_atten_bias_support():
    atten_bias_options = [
        # (has_attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1)
        (False, False, False),
        (True, False, False),  # [b, n, s_q, s_kv]
        (True, True, False),  # [1, n, s_q, s_kv]
        (True, False, True),  # [b, 1, s_q, s_kv]
        (True, True, True),  # [1, 1, s_q, s_kv]
    ]
    return atten_bias_options


def attention_reference(
    head_size: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float] = None,
    attn_bias: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Reference implementation of SDPA

    Args:
        head_size (int): dimension per head
        query (torch.Tensor): query in BNSH format
        key (torch.Tensor): key in BNSH format
        value (torch.Tensor): value in BNSH format
        scale (Optional[float], optional): scale applied on QxK'. Defaults to None.
        attn_bias : attention bias tensor added before softmax. Defaults to None.
        masks : attention masks. Defaults to None.

    Returns:
        torch.Tensor: result of SDPA
    """
    if scale is None:
        scale = 1.0 / (head_size**0.5)

    assert query.size(1) == key.size(1) and value.size(1) == key.size(1)
    assert query.dim() == 4
    assert key.dim() == 4
    assert value.dim() == 4

    if verbose:
        torch.set_printoptions(precision=6, linewidth=200, sci_mode=False)
        print("query(ref)", query)
        print("key(ref)", key)
        print("value(ref)", value)
        if mask is not None:
            print("mask", mask)

    # Apply multi-head attention.
    attn = torch.einsum("bhmd,bhnd->bhmn", query, key).float() * scale
    if verbose:
        print("QK(ref)", attn)

    if attn_bias is not None:
        attn = attn + attn_bias
        if verbose:
            print("QK+AttnBias(ref)", attn)

    if mask is not None:
        attn = attn.masked_fill((1 - mask.int()).bool(), float("-inf"))
        if verbose:
            print("masked QK(ref)", attn)

    attn = attn.softmax(-1)
    if verbose:
        print("Softmax(ref)", attn)

    attn_output = torch.einsum("bhmn,bhnd->bhmd", attn.type_as(value), value)

    result = attn_output.transpose(1, 2).contiguous()

    if query.device.type == "cuda":
        torch.cuda.synchronize()

    if verbose:
        print("result(ref)", result)

    return result


def mha_with_past_reference(
    config: MultiHeadAttentionConfig,
    past_k: Optional[torch.Tensor],
    past_v: Optional[torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    attn_bias: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
):
    assert config.kv_sequence_length == config.sequence_length
    assert config.use_kv_cache
    if past_k is not None:
        assert past_k.dim() == 4 and k.dim() == 4 and past_k.size(1) == k.size(1), (
            f"expect BNSH format: {past_k.shape=} {k.shape=}"
        )

    if past_v is not None:
        assert past_v.dim() == 4 and v.dim() == 4 and past_v.size(1) == v.size(1), (
            f"expect BNSH format: {past_v.shape=} {v.shape=}"
        )

    present_k = torch.cat((past_k, k), dim=2) if past_k is not None else k
    present_v = torch.cat((past_v, v), dim=2) if past_v is not None else v
    out = attention_reference(config.head_size, q, present_k, present_v, scale=scale, attn_bias=attn_bias, mask=mask)

    return out, present_k, present_v


def get_compute_capability():
    if torch.cuda.is_available() and "CUDAExecutionProvider" in onnxruntime.get_available_providers():
        major, minor = torch.cuda.get_device_capability()
        sm = major * 10 + minor
        return sm
    return 0


def no_kv_cache_test_cases(provider: str, comprehensive: bool):
    if provider == "CUDAExecutionProvider" and get_compute_capability() < 60:
        return
        yield

    batch_sizes = [1, 2, 3]
    sequence_lengths = [1, 16, 127, 128, 255, 256, 383, 384, 512]
    heads = [1, 3, 4, 16]
    head_sizes = [8, 16, 32, 40, 64, 80, 96, 128, 160, 192, 224, 256]

    mask_formats = [
        AttentionMaskFormat.Mask_None,
        AttentionMaskFormat.Mask_1D_Key_SeqLen,
        AttentionMaskFormat.Mask_2D_Key_PaddingMask,
    ]
    atten_bias_options = get_atten_bias_support()

    device, dtype, formats = get_provider_support_info(provider, False)
    if comprehensive:
        sequence_lengths = [*sequence_lengths, 2048]  # Large sequence length is slow and need a lot of memory
        for batch_size in batch_sizes:
            for sequence_length in sequence_lengths:
                for num_heads in heads:
                    for head_size in head_sizes:
                        for format in formats:
                            for causal in get_causal_support(format):
                                for mask_format in mask_formats:
                                    for has_bias in get_bias_support(format):
                                        for (
                                            has_attn_bias,
                                            broadcast_attn_bias_dim_0,
                                            broadcast_attn_bias_dim_1,
                                        ) in atten_bias_options:
                                            config = MultiHeadAttentionConfig(
                                                batch_size=batch_size,
                                                sequence_length=sequence_length,
                                                num_heads=num_heads,
                                                head_size=head_size,
                                                causal=causal,
                                                past_sequence_length=0,
                                                kv_sequence_length=sequence_length,
                                                max_cache_sequence_length=None,
                                                provider=provider,
                                                device=device,
                                                dtype=dtype,
                                                use_kv_cache=False,
                                                share_past_present_buffer=False,
                                                input_format=format,
                                                has_bias=has_bias,
                                                mask_format=mask_format,
                                                has_attn_bias=has_attn_bias,
                                                broadcast_attn_bias_dim_0=broadcast_attn_bias_dim_0,
                                                broadcast_attn_bias_dim_1=broadcast_attn_bias_dim_1,
                                            )
                                            yield config
    else:
        test_cases = max(len(batch_sizes), len(sequence_lengths), len(heads), len(head_sizes))
        for i in range(test_cases):
            batch_size = batch_sizes[i % len(batch_sizes)]
            sequence_length = sequence_lengths[i % len(sequence_lengths)]
            num_heads = heads[i % len(heads)]
            head_size = head_sizes[i % len(head_sizes)]
            mask_format = mask_formats[i % len(mask_formats)]
            has_attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1 = atten_bias_options[
                i % len(atten_bias_options)
            ]
            for format in formats:
                for causal in get_causal_support(format):
                    for has_bias in get_bias_support(format):
                        config = MultiHeadAttentionConfig(
                            batch_size=batch_size,
                            sequence_length=sequence_length,
                            num_heads=num_heads,
                            head_size=head_size,
                            causal=causal,
                            past_sequence_length=0,
                            kv_sequence_length=sequence_length,
                            max_cache_sequence_length=None,
                            provider=provider,
                            device=device,
                            dtype=dtype,
                            use_kv_cache=False,
                            share_past_present_buffer=False,
                            input_format=format,
                            has_bias=has_bias,
                            mask_format=mask_format,
                            has_attn_bias=has_attn_bias,
                            broadcast_attn_bias_dim_0=broadcast_attn_bias_dim_0,
                            broadcast_attn_bias_dim_1=broadcast_attn_bias_dim_1,
                        )
                        yield config


def kv_cache_test_cases(provider: str, comprehensive: bool):
    if provider == "CUDAExecutionProvider" and get_compute_capability() < 60:
        return
        yield

    batch_sizes = [1, 2, 3]
    sequence_lengths = [1, 15, 16, 255, 256, 512]
    heads = [1, 3, 4, 16]
    head_sizes = [8, 16, 32, 40, 64, 80, 96, 128, 160, 192, 224, 256]
    device, dtype, formats = get_provider_support_info(provider, True)
    mask_formats = [
        AttentionMaskFormat.Mask_None,
        AttentionMaskFormat.Mask_1D_Key_SeqLen,
        AttentionMaskFormat.Mask_2D_Key_PaddingMask,
    ]

    atten_bias_options = get_atten_bias_support()

    if comprehensive:
        sequence_lengths = [*sequence_lengths, 2048]  # Large sequence length is slow and need a lot of memory
        for batch_size in batch_sizes:
            for past_sequence_length in sequence_lengths:
                for num_heads in heads:
                    for head_size in head_sizes:
                        for format in formats:
                            for causal in get_causal_support(format):
                                for has_past_input in [True, False]:
                                    for mask_format in mask_formats:
                                        for has_bias in get_bias_support(format):
                                            for (
                                                has_attn_bias,
                                                broadcast_attn_bias_dim_0,
                                                broadcast_attn_bias_dim_1,
                                            ) in atten_bias_options:
                                                sequence_length = 1 if has_past_input else past_sequence_length
                                                past_seq_len = past_sequence_length if has_past_input else 0
                                                config = MultiHeadAttentionConfig(
                                                    batch_size=batch_size,
                                                    sequence_length=sequence_length,
                                                    num_heads=num_heads,
                                                    head_size=head_size,
                                                    causal=causal,
                                                    past_sequence_length=past_seq_len,
                                                    kv_sequence_length=sequence_length,
                                                    max_cache_sequence_length=None,
                                                    provider=provider,
                                                    device=device,
                                                    dtype=dtype,
                                                    use_kv_cache=True,
                                                    has_past_input=has_past_input,
                                                    share_past_present_buffer=False,
                                                    input_format=format,
                                                    has_bias=has_bias,
                                                    mask_format=mask_format,
                                                    has_attn_bias=has_attn_bias,
                                                    broadcast_attn_bias_dim_0=broadcast_attn_bias_dim_0,
                                                    broadcast_attn_bias_dim_1=broadcast_attn_bias_dim_1,
                                                )
                                                yield config
    else:
        test_cases = max(len(batch_sizes), len(sequence_lengths), len(heads), len(head_sizes))
        for i in range(test_cases):
            batch_size = batch_sizes[i % len(batch_sizes)]
            past_sequence_length = sequence_lengths[i % len(sequence_lengths)]
            num_heads = heads[i % len(heads)]
            head_size = head_sizes[i % len(head_sizes)]
            mask_format = mask_formats[i % len(mask_formats)]
            has_attn_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1 = atten_bias_options[
                i % len(atten_bias_options)
            ]
            for format in formats:
                for causal in get_causal_support(format):
                    for has_past_input in [True, False]:
                        for has_bias in get_bias_support(format):
                            sequence_length = 1 if has_past_input else past_sequence_length
                            past_seq_len = past_sequence_length if has_past_input else 0
                            config = MultiHeadAttentionConfig(
                                batch_size=batch_size,
                                sequence_length=sequence_length,
                                num_heads=num_heads,
                                head_size=head_size,
                                causal=causal,
                                past_sequence_length=past_seq_len,
                                kv_sequence_length=sequence_length,
                                max_cache_sequence_length=None,
                                provider=provider,
                                device=device,
                                dtype=dtype,
                                use_kv_cache=True,
                                has_past_input=has_past_input,
                                share_past_present_buffer=False,
                                input_format=format,
                                has_bias=has_bias,
                                mask_format=mask_format,
                                has_attn_bias=has_attn_bias,
                                broadcast_attn_bias_dim_0=broadcast_attn_bias_dim_0,
                                broadcast_attn_bias_dim_1=broadcast_attn_bias_dim_1,
                            )
                            yield config


def lean_attention_test_cases(provider: str, comprehensive: bool):
    if provider == "CUDAExecutionProvider" and get_compute_capability() < 80:
        return
        yield

    batch_sizes = [1, 2, 3] if comprehensive else [1, 2]
    sequence_lengths = [2, 15, 16, 255, 256, 512, 1024, 2048, 4096, 8192] if comprehensive else [2, 255, 512]
    heads = [1, 4, 16] if comprehensive else [1, 4]
    head_sizes = [64, 128]
    device, dtype, formats = get_provider_support_info(provider, True)
    mask_formats = [AttentionMaskFormat.Mask_None]

    sequence_lengths = [*sequence_lengths, 2048]  # Large sequence length is slow and need a lot of memory
    for batch_size in batch_sizes:
        for total_seq_len in sequence_lengths:
            for num_heads in heads:
                for head_size in head_sizes:
                    for format in formats:
                        for causal in get_causal_support(format):
                            for is_prompt in [False]:
                                for mask_format in mask_formats:
                                    sequence_length = total_seq_len if is_prompt else 1
                                    config = MultiHeadAttentionConfig(
                                        batch_size=batch_size,
                                        sequence_length=sequence_length,
                                        num_heads=num_heads,
                                        head_size=head_size,
                                        causal=causal,
                                        past_sequence_length=total_seq_len - sequence_length,
                                        kv_sequence_length=sequence_length,
                                        max_cache_sequence_length=None,
                                        provider=provider,
                                        device=device,
                                        dtype=dtype,
                                        use_kv_cache=True,
                                        has_past_input=True,
                                        share_past_present_buffer=False,
                                        input_format=format,
                                        mask_format=mask_format,
                                    )
                                    yield config


def no_kv_cache_multi_thread_test_cases(provider: str, comprehensive: bool):
    if provider == "CUDAExecutionProvider" and get_compute_capability() < 60:
        return
        yield

    batch_sizes = [1, 2]
    sequence_lengths = [1, 16, 127, 128, 255, 256, 383, 384, 400] if comprehensive else [1, 64, 128, 256]
    heads = [4]
    head_sizes = [8, 16, 32, 40, 64, 80, 96, 128, 160, 192, 224, 256] if comprehensive else [32, 64]

    device, dtype, formats = get_provider_support_info(provider, False)

    for format in formats:
        for causal in get_causal_support(format):
            for num_heads in heads:
                for head_size in head_sizes:
                    configs = []  # list of configurations to run in parallel
                    for batch_size in batch_sizes:
                        for sequence_length in sequence_lengths:
                            config = MultiHeadAttentionConfig(
                                batch_size=batch_size,
                                sequence_length=sequence_length,
                                num_heads=num_heads,
                                head_size=head_size,
                                causal=causal,
                                past_sequence_length=0,
                                kv_sequence_length=sequence_length,
                                max_cache_sequence_length=None,
                                provider=provider,
                                device=device,
                                dtype=dtype,
                                use_kv_cache=False,
                                share_past_present_buffer=False,
                                input_format=format,
                            )
                            configs.append(config)
                    yield configs


def kv_cache_multi_thread_test_cases(provider: str, comprehensive: bool):
    if provider == "CUDAExecutionProvider" and get_compute_capability() < 60:
        return
        yield

    batch_sizes = [1, 2]
    sequence_lengths = [1, 32, 127, 128, 383, 384, 400] if comprehensive else [1, 32, 127, 128]
    heads = [4]
    head_sizes = [8, 16, 32, 40, 64, 80, 96, 128, 160, 192, 224, 256] if comprehensive else [32, 64]

    sequence_length = 1
    device, dtype, formats = get_provider_support_info(provider, True)

    for format in formats:
        for causal in get_causal_support(format):
            for num_heads in heads:
                for head_size in head_sizes:
                    configs = []
                    for batch_size in batch_sizes:
                        for past_sequence_length in sequence_lengths:
                            config = MultiHeadAttentionConfig(
                                batch_size=batch_size,
                                sequence_length=sequence_length,
                                num_heads=num_heads,
                                head_size=head_size,
                                causal=causal,
                                past_sequence_length=past_sequence_length,
                                kv_sequence_length=sequence_length,
                                max_cache_sequence_length=None,
                                provider=provider,
                                device=device,
                                dtype=dtype,
                                use_kv_cache=True,
                                has_past_input=True,
                                share_past_present_buffer=False,
                                input_format=format,
                            )
                            configs.append(config)
                    yield configs


def causal_mask(seqlen_q, seqlen_k, query_padding_mask=None, key_padding_mask=None, device=None):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    sk = seqlen_k if key_padding_mask is None else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    sq = seqlen_q if query_padding_mask is None else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    return col_idx <= row_idx + sk - sq


def merge_padding_and_causal_masks(config):
    q_mask, k_mask, mask = config.right_side_padding_masks()
    if config.causal:
        query_padding_mask = q_mask.reshape(config.batch_size, config.sequence_length)
        key_padding_mask = k_mask.reshape(config.batch_size, config.total_sequence_length)
        mask = causal_mask(
            config.sequence_length,
            config.total_sequence_length,
            query_padding_mask,
            key_padding_mask,
            device=config.device,
        )

    return mask


def parity_check_mha(
    config: MultiHeadAttentionConfig,
    rtol=1e-3,
    atol=1e-3,
):
    ort_mha = OrtMultiHeadAttention(config, use_tf32=False)
    ort_outputs = ort_mha.infer(synchronize=True)
    out = ort_outputs["output"]
    out = torch.reshape(out, (config.batch_size, config.sequence_length, config.num_heads, config.head_size))

    ort_input_format = config.input_format
    no_bias_k_v = config.input_format == InputFormats.Q_K_V_BSNH_BNSH_BNSH
    config.input_format = InputFormats.Q_K_V_BSNH_BSNH_BSNH
    ref_inputs = config.random_inputs(no_bias_k_v=no_bias_k_v)
    q = ref_inputs["query"].reshape((config.batch_size, config.sequence_length, config.num_heads, config.head_size))
    k = ref_inputs["key"].reshape((config.batch_size, config.kv_sequence_length, config.num_heads, config.head_size))
    v = ref_inputs["value"].reshape((config.batch_size, config.kv_sequence_length, config.num_heads, config.head_size))

    if "bias" in ref_inputs:
        bias = ref_inputs["bias"]
        bias = bias.reshape((3, config.num_heads, config.head_size))
        bias_q = bias[0, :, :].reshape(1, 1, config.num_heads, config.head_size)
        bias_k = bias[1, :, :].reshape(1, 1, config.num_heads, config.head_size)
        bias_v = bias[2, :, :].reshape(1, 1, config.num_heads, config.head_size)
        q = q + bias_q
        k = k + bias_k
        v = v + bias_v

    attn_bias = None
    if config.has_attn_bias:
        attn_bias = ref_inputs["attn_bias"]

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    mask = merge_padding_and_causal_masks(config)
    k_cache = None
    v_cache = None
    if config.use_kv_cache:
        past_k = ref_inputs.get("past_key", None)
        past_v = ref_inputs.get("past_value", None)
        out_ref, k_cache, v_cache = mha_with_past_reference(
            config, past_k, past_v, q, k, v, scale=config.scale, attn_bias=attn_bias, mask=mask
        )
    else:
        out_ref = attention_reference(config.head_size, q, k, v, scale=config.scale, attn_bias=attn_bias, mask=mask)

    # Fill zeros for the padded tokens for comparison.
    if config.mask_index_q is not None:
        for i, m in enumerate(config.mask_index_q):
            out[i, m:, :, :] = 0
            out_ref[i, m:, :, :] = 0

    if config.mask_index_kv is not None and config.use_kv_cache:
        assert k_cache is not None
        assert v_cache is not None
        present_key = ort_outputs["present_key"]
        present_value = ort_outputs["present_value"]
        for i, n in enumerate(config.mask_index_kv):
            k_cache[i, :, n:, :] = 0
            present_key[i, :, n:, :] = 0
            v_cache[i, :, n:, :] = 0
            present_value[i, :, n:, :] = 0

    # Restore the input format so that it shows up in the error message correctly.
    config.input_format = ort_input_format

    numpy.testing.assert_allclose(
        out.detach().cpu().numpy(),
        out_ref.detach().cpu().numpy(),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
        err_msg=f"output not close: {config=}",
    )

    if config.use_kv_cache:
        present_key = ort_outputs["present_key"]
        numpy.testing.assert_allclose(
            k_cache.detach().cpu().numpy(),
            present_key.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
            err_msg=f"present_key not close: {config=}",
        )

        present_value = ort_outputs["present_value"]
        numpy.testing.assert_allclose(
            v_cache.detach().cpu().numpy(),
            present_value.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
            err_msg=f"present_value not close: {config=}",
        )


def parity_check_mha_multi_threading(
    test_inputs: List[Dict],
    rtol: float = 1e-3,
    atol: float = 1e-3,
    attention_kernel=SdpaKernel.DEFAULT,
    max_threads: int = 5,
    verbose: bool = False,
):
    # Use the first config to create a session, which is shared by all configs to run in parallel.
    config = test_inputs[0]["config"]

    # Some kernel does not support certain input format.
    if attention_kernel not in [
        SdpaKernel.DEFAULT,
        SdpaKernel.FLASH_ATTENTION,
        SdpaKernel.EFFICIENT_ATTENTION,
    ] and config.input_format in [InputFormats.Q_KV_BSNH_BSN2H]:
        return None

    ort_session = create_ort_session(config, attention_kernel=attention_kernel, use_symbolic_shape=True, use_tf32=False)

    def convert_to_ort_inputs(feed_dict):
        ort_inputs = {}

        for k, v in feed_dict.items():
            if isinstance(v, numpy.ndarray):
                ort_inputs[k] = v
            else:
                ort_inputs[k] = v.detach().cpu().numpy()
        return ort_inputs

    def check_parity_with_config(i: int):
        config = test_inputs[i]["config"]
        if verbose:
            print(f"Thread {i} with {vars(config)}")

        ort_inputs = test_inputs[i]["ort_inputs"]

        if verbose:
            print(f"Thread {i} ort inputs: {ort_inputs}")
        ort_outputs = ort_session.run(None, convert_to_ort_inputs(ort_inputs))
        out = numpy.reshape(
            ort_outputs[0], (config.batch_size, config.sequence_length, config.num_heads, config.head_size)
        )

        # Create reference inputs
        old_format = config.input_format
        config.input_format = InputFormats.Q_K_V_BSNH_BSNH_BSNH
        ref_inputs = test_inputs[i]["ref_inputs"]
        if verbose:
            print(f"Thread {i} ref inputs: {ref_inputs}")

        q = ref_inputs["query"].reshape((config.batch_size, config.sequence_length, config.num_heads, config.head_size))
        k = ref_inputs["key"].reshape(
            (config.batch_size, config.kv_sequence_length, config.num_heads, config.head_size)
        )
        v = ref_inputs["value"].reshape(
            (config.batch_size, config.kv_sequence_length, config.num_heads, config.head_size)
        )

        if "bias" in ref_inputs:
            bias = ref_inputs["bias"]
            bias = bias.reshape((3, config.num_heads, config.head_size))
            bias_q = bias[0, :, :].reshape(1, 1, config.num_heads, config.head_size)
            bias_k = bias[1, :, :].reshape(1, 1, config.num_heads, config.head_size)
            bias_v = bias[2, :, :].reshape(1, 1, config.num_heads, config.head_size)
            q = q + bias_q
            k = k + bias_k
            v = v + bias_v

        attn_bias = None
        if config.has_attn_bias:
            attn_bias = ref_inputs["attn_bias"]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        mask = merge_padding_and_causal_masks(config)
        k_cache = None
        v_cache = None
        if config.use_kv_cache:
            past_k = ref_inputs.get("past_key", None)
            past_v = ref_inputs.get("past_value", None)
            out_ref, k_cache, v_cache = mha_with_past_reference(
                config, past_k, past_v, q, k, v, scale=config.scale, attn_bias=attn_bias, mask=mask
            )
        else:
            out_ref = attention_reference(config.head_size, q, k, v, scale=config.scale, attn_bias=attn_bias, mask=mask)

        # Fill zeros for the padded tokens for comparison.
        if config.mask_index_q is not None:
            for i, m in enumerate(config.mask_index_q):
                out[i, m:, :, :] = 0
                out_ref[i, m:, :, :] = 0

        if config.mask_index_kv is not None and config.use_kv_cache:
            assert k_cache is not None
            assert v_cache is not None
            present_key = ort_outputs[1]
            present_value = ort_outputs[2]
            for i, n in enumerate(config.mask_index_kv):
                k_cache[i, :, n:, :] = 0
                present_key[i, :, n:, :] = 0
                v_cache[i, :, n:, :] = 0
                present_value[i, :, n:, :] = 0

        # Restore the input format so that it shows up in the error message correctly.
        config.input_format = old_format

        try:
            numpy.testing.assert_allclose(
                out,
                out_ref.detach().cpu().numpy(),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
                err_msg=f"output not close: {config=}",
            )

            if config.use_kv_cache:
                present_key = ort_outputs[1]
                numpy.testing.assert_allclose(
                    k_cache.detach().cpu().numpy(),
                    present_key,
                    rtol=rtol,
                    atol=atol,
                    equal_nan=True,
                    err_msg=f"present_key not close: {config=}",
                )

                present_value = ort_outputs[2]
                numpy.testing.assert_allclose(
                    v_cache.detach().cpu().numpy(),
                    present_value,
                    rtol=rtol,
                    atol=atol,
                    equal_nan=True,
                    err_msg=f"present_value not close: {config=}",
                )
        except AssertionError as e:
            print(f"Failed with {vars(config)}: {e}")
            return e

        if verbose:
            print(f"Passed: {vars(config)}")
        return None

    num_threads = min(max_threads, len(test_inputs))

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_tasks = [executor.submit(check_parity_with_config, i) for i in range(num_threads)]
        for future in concurrent.futures.as_completed(future_tasks):
            result = future.result()
            if result is not None:
                return result

    return None


def mha_test_cases(provider: str, comprehensive: bool):
    return itertools.chain(
        no_kv_cache_test_cases(provider, comprehensive),
        kv_cache_test_cases(provider, comprehensive),
    )


def multi_thread_test_cases(provider: str, comprehensive: bool):
    return itertools.chain(
        no_kv_cache_multi_thread_test_cases(provider, comprehensive),
        kv_cache_multi_thread_test_cases(provider, comprehensive),
    )


# Off by default so that we do not run too many tests in CI pipeline.
comprehensive_mode = False


class TestMultiHeadAttention(unittest.TestCase):
    def run_mha_cuda(self):
        for config in mha_test_cases("CUDAExecutionProvider", comprehensive_mode):
            parity_check_mha(config, rtol=5e-3, atol=5e-3)

    def run_lean_attention(self):
        os.environ["ORT_ENABLE_LEAN_ATTENTION"] = "1"
        for config in lean_attention_test_cases("CUDAExecutionProvider", comprehensive_mode):
            parity_check_mha(config, rtol=5e-3, atol=5e-3 if config.total_sequence_length <= 512 else 5e-2)
        os.environ.pop("ORT_ENABLE_LEAN_ATTENTION", None)

    def run_mha_cpu(self):
        for config in mha_test_cases("CPUExecutionProvider", comprehensive_mode):
            parity_check_mha(config, rtol=5e-3, atol=5e-3)

    def run_mha_cuda_multi_threading(self, attention_kernel):
        for configs in multi_thread_test_cases("CUDAExecutionProvider", comprehensive_mode):
            if configs and configs[0].causal and (SdpaKernel.TRT_CAUSAL_ATTENTION & attention_kernel != 0):
                # TRT fused causal is disabled by default so skip the test of causal for multi-threading.
                continue

            test_inputs = []
            for config in configs:
                ort_inputs = config.random_inputs()

                # Create reference inputs
                old_format = config.input_format
                config.input_format = InputFormats.Q_K_V_BSNH_BSNH_BSNH
                ref_inputs = config.random_inputs()
                config.input_format = old_format
                test_inputs.append({"config": config, "ort_inputs": ort_inputs, "ref_inputs": ref_inputs})

            exception = parity_check_mha_multi_threading(
                test_inputs, attention_kernel=attention_kernel, max_threads=len(configs)
            )
            assert exception is None, f"Multi-threading failed: {attention_kernel=}, {vars(configs[0])}, {exception}"

    def run_mha_cuda_multi_threading_default(self):
        if get_compute_capability() >= 60:
            self.run_mha_cuda_multi_threading(SdpaKernel.DEFAULT)

    def run_mha_cuda_multi_threading_cudnn(self):
        if get_compute_capability() in [80, 86, 89, 90]:
            self.run_mha_cuda_multi_threading(SdpaKernel.CUDNN_FLASH_ATTENTION)

    def run_mha_cuda_multi_threading_efficient(self):
        if comprehensive_mode and get_compute_capability() >= 60:
            self.run_mha_cuda_multi_threading(SdpaKernel.EFFICIENT_ATTENTION)

    def run_mha_cuda_multi_threading_math(self):
        if comprehensive_mode and get_compute_capability() >= 60:
            self.run_mha_cuda_multi_threading(SdpaKernel.MATH)

    def run_mha_cuda_multi_threading_trt(self):
        if get_compute_capability() in [75, 80, 86, 89]:
            self.run_mha_cuda_multi_threading(
                SdpaKernel.TRT_FUSED_ATTENTION
                | SdpaKernel.TRT_FLASH_ATTENTION
                | SdpaKernel.TRT_CAUSAL_ATTENTION
                | SdpaKernel.TRT_CROSS_ATTENTION
            )

    def test_all(self):
        # Run tests sequentially to avoid out of memory issue.
        self.run_mha_cpu()
        self.run_mha_cuda()
        # self.run_lean_attention()
        self.run_mha_cuda_multi_threading_default()
        self.run_mha_cuda_multi_threading_cudnn()
        self.run_mha_cuda_multi_threading_efficient()
        self.run_mha_cuda_multi_threading_math()
        self.run_mha_cuda_multi_threading_trt()


if __name__ == "__main__":
    with torch.no_grad():
        unittest.main()
