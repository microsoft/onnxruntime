# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Test MultiHeadAttention operator for CUDA and CPU.
"""

import itertools
import unittest
from typing import Optional

import numpy
import torch
from benchmark_mha import InputFormats, MultiHeadAttentionConfig, OrtMultiHeadAttention
from einops import rearrange
from parameterized import parameterized

import onnxruntime


def attention_reference(
    head_size: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float] = None,
    mask: Optional[torch.Tensor] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Reference implementation of Dot Product Attention

    Args:
        head_size (int): dimension per head
        query (torch.Tensor): query in BNSH format
        key (torch.Tensor): key in BNSH format
        value (torch.Tensor): value in BNSH format
        scale (Optional[float], optional): scale applied before softmax. Defaults to None.
        mask (Optional[torch.Tensor], optional): attention mask. Defaults to None.

    Returns:
        torch.Tensor: result of dot product attention
    """
    if scale is None:
        scale = 1.0 / (head_size**0.5)

    assert query.size(1) == key.size(1) and value.size(1) == key.size(1)
    assert query.dim() == 4
    assert key.dim() == 4
    assert value.dim() == 4

    if verbose:
        print("query(SDPA)", query)
        print("key(SDPA)", key)
        print("value(SDPA)", value)
        if mask is not None:
            print("mask", mask)

    # Apply multi-head attention.
    attn = torch.einsum("bhmd,bhnd->bhmn", query, key).float() * scale
    if mask is not None:
        attn = attn.masked_fill((1 - mask.int()).bool(), float("-inf"))
    if verbose:
        print("QK(SDPA)", attn)

    attn = attn.softmax(-1)
    if verbose:
        print("Softmax(SDPA)", attn)

    attn_output = torch.einsum("bhmn,bhnd->bhmd", attn.type_as(value), value)

    result = attn_output.transpose(1, 2).contiguous()

    if query.device.type == "cuda":
        torch.cuda.synchronize()

    if verbose:
        print("result(SDPA)", result)

    return result


def mha_with_past_reference(
    config: MultiHeadAttentionConfig,
    past_k: torch.Tensor,
    past_v: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    mask: Optional[torch.Tensor] = None,
):
    assert config.kv_sequence_length == config.sequence_length
    assert config.use_kv_cache
    assert past_k.dim() == 4 and k.dim() == 4 and past_k.size(1) == k.size(1)  # both BNSH format
    assert past_v.dim() == 4 and v.dim() == 4 and past_v.size(1) == v.size(1)  # both BNSH format

    present_k = torch.cat((past_k, k), dim=2)
    present_v = torch.cat((past_v, v), dim=2)
    out = attention_reference(config.head_size, q, present_k, present_v, scale=scale, mask=mask)

    return out, present_k, present_v


def get_provider_support_info(provider: str, use_kv_cache: bool):
    if provider == "CUDAExecutionProvider":
        formats = [InputFormats.Q_K_V_BSNH_BSNH_BSNH, InputFormats.Q_KV_BSNH_BSN2H, InputFormats.QKV_BSN3H]
        if not use_kv_cache:
            formats.append(InputFormats.Q_K_V_BSNH_BSNH_BSNH)
        device_id = torch.cuda.current_device()
        device = torch.device("cuda", device_id)
        dtype = torch.float16
    else:
        assert provider == "CPUExecutionProvider"
        formats = [InputFormats.Q_K_V_BSNH_BSNH_BSNH]
        if not use_kv_cache:
            formats.append(InputFormats.Q_K_V_BSNH_BSNH_BSNH)
        device = torch.device("cpu")
        dtype = torch.float
    return device, dtype, formats


def has_cuda_support():
    if torch.cuda.is_available() and "CUDAExecutionProvider" in onnxruntime.get_available_providers():
        major, _ = torch.cuda.get_device_capability()
        return major >= 6
    return False


def no_kv_cache_test_cases(provider: str, comprehensive: bool):
    if provider == "CUDAExecutionProvider" and not has_cuda_support():
        return
        yield

    batch_sizes = [1, 2, 3]
    sequence_lengths = [1, 16, 127, 128, 255, 256, 383, 384, 2048]
    heads = [1, 3, 4, 16]
    head_sizes = [8, 16, 32, 40, 64, 80, 96, 128, 160, 192, 224, 256]

    device, dtype, formats = get_provider_support_info(provider, False)
    if comprehensive:
        for batch_size in batch_sizes:
            for sequence_length in sequence_lengths:
                for num_heads in heads:
                    for head_size in head_sizes:
                        for format in formats:
                            for causal in [True, False]:
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
                                yield config
    else:
        test_cases = max(len(batch_sizes), len(sequence_lengths), len(heads), len(head_sizes))
        for i in range(test_cases):
            batch_size = batch_sizes[i % len(batch_sizes)]
            sequence_length = sequence_lengths[i % len(sequence_lengths)]
            num_heads = heads[i % len(heads)]
            head_size = head_sizes[i % len(head_sizes)]
            format = formats[i % len(formats)]
            for causal in [True, False]:
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
                yield config


def kv_cache_test_cases(provider: str, comprehensive: bool):
    if provider == "CUDAExecutionProvider" and not has_cuda_support():
        return
        yield

    batch_sizes = [1, 2, 3]
    sequence_lengths = [1, 15, 16, 255, 256, 2048]
    heads = [1, 3, 4, 16]
    head_sizes = [8, 16, 32, 40, 64, 80, 96, 128, 160, 192, 224, 256]

    sequence_length = 1
    device, dtype, formats = get_provider_support_info(provider, True)

    if comprehensive:
        for batch_size in batch_sizes:
            for past_sequence_length in sequence_lengths:
                for num_heads in heads:
                    for head_size in head_sizes:
                        for format in formats:
                            for causal in [True, False]:
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
                                    share_past_present_buffer=False,
                                    input_format=format,
                                )
                                yield config
    else:
        test_cases = max(len(batch_sizes), len(sequence_lengths), len(heads), len(head_sizes))
        for i in range(test_cases):
            batch_size = batch_sizes[i % len(batch_sizes)]
            past_sequence_length = sequence_lengths[i % len(sequence_lengths)]
            num_heads = heads[i % len(heads)]
            head_size = head_sizes[i % len(head_sizes)]
            format = formats[i % len(formats)]
            for causal in [True, False]:
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
                    share_past_present_buffer=False,
                    input_format=format,
                )
                yield config


def mha_test_cases(provider: str, comprehensive: bool):
    return itertools.chain(
        no_kv_cache_test_cases(provider, comprehensive), kv_cache_test_cases(provider, comprehensive)
    )


def causal_mask(seqlen_q, seqlen_k, query_padding_mask=None, key_padding_mask=None, device=None):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    sk = seqlen_k if key_padding_mask is None else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    sq = seqlen_q if query_padding_mask is None else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    return col_idx <= row_idx + sk - sq


def parity_check_mha(
    config: MultiHeadAttentionConfig,
    rtol=1e-3,
    atol=1e-3,
):
    # CUDA kernel does not support causal so skip such test cases.
    if config.causal and config.provider == "CUDAExecutionProvider":
        return

    ort_mha = OrtMultiHeadAttention(config)
    ort_outputs = ort_mha.infer()
    out = ort_outputs["output"]
    out = torch.reshape(out, (config.batch_size, config.sequence_length, config.num_heads, config.head_size))

    config.input_format = InputFormats.Q_K_V_BSNH_BSNH_BSNH
    ref_inputs = config.random_inputs()
    q = (
        ref_inputs["query"]
        .reshape((config.batch_size, config.sequence_length, config.num_heads, config.head_size))
        .transpose(1, 2)
    )
    k = (
        ref_inputs["key"]
        .reshape((config.batch_size, config.kv_sequence_length, config.num_heads, config.head_size))
        .transpose(1, 2)
    )
    v = (
        ref_inputs["value"]
        .reshape((config.batch_size, config.kv_sequence_length, config.num_heads, config.head_size))
        .transpose(1, 2)
    )

    mask = None
    if config.causal:
        mask = causal_mask(config.sequence_length, config.total_sequence_length, device=config.device)

    k_cache = None
    v_cache = None
    if config.use_kv_cache:
        past_k = ref_inputs["past_key"]
        past_v = ref_inputs["past_value"]
        out_ref, k_cache, v_cache = mha_with_past_reference(config, past_k, past_v, q, k, v, mask=mask)
    else:
        out_ref = attention_reference(config.head_size, q, k, v, mask=mask)

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


# Do not run too many tests in CI pipeline. Change it to True to run all combinations in dev machine.
comprehensive_mode = False


class TestMultiHeadAttention(unittest.TestCase):
    # TODO: enable tests on CUDAExecutionProvider after fixing the issue.
    # @parameterized.expand(mha_test_cases("CUDAExecutionProvider", comprehensive_mode), skip_on_empty=True)
    # def test_mha_cuda(self, config):
    #     parity_check_mha(config)

    @parameterized.expand(mha_test_cases("CPUExecutionProvider", comprehensive_mode), skip_on_empty=True)
    def test_mha_cpu(self, config):
        parity_check_mha(config)


if __name__ == "__main__":
    with torch.no_grad():
        unittest.main()
