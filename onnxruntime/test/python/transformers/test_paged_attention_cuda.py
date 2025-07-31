# --------------------------------------------------------------------------
# Copyright 2020 The HuggingFace Inc. team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# -------------------------------------------------------------------------
import math
import platform
import random
import unittest

import numpy
import torch
from einops import rearrange, repeat
from onnx import TensorProto, helper
from packaging import version
from parameterized import parameterized
from test_gqa_cpu import smooth_softmax_ref

from onnxruntime import InferenceSession, OrtValue, SessionOptions, get_available_providers

torch.manual_seed(0)

pipeline_mode = True  # Reduces number of tests so pipeline doesn't time out


class Config:
    batch_size = 0
    sequence_length = 0
    total_sequence_length = 0
    num_heads = 0
    kv_num_heads = 0
    head_size = 0
    paged_kv_block_size = 0
    local = False
    rotary = False
    rotary_interleaved = False
    packed = False
    softcap = 0.0
    ep = "CUDAExecutionProvider"

    def __init__(
        self,
        batch_size,
        sequence_length,
        total_sequence_length,
        num_heads,
        kv_num_heads,
        head_size,
        paged_kv_block_size,
        local,
        rotary,
        rotary_interleaved,
        packed,
        softcap,
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.total_sequence_length = total_sequence_length
        self.num_heads = num_heads
        self.kv_num_heads = kv_num_heads
        self.head_size = head_size
        self.paged_kv_block_size = paged_kv_block_size
        self.local = local
        self.rotary = rotary
        self.rotary_interleaved = rotary_interleaved
        self.packed = packed
        self.softcap = softcap

    def __repr__(self):
        short_ep = self.ep[: -len("ExecutionProvider")].lower()
        return (
            f"Config(batch_size={self.batch_size}, sequence_length={self.sequence_length}, "
            f"total_sequence_length={self.total_sequence_length}, num_heads={self.num_heads}, "
            f"kv_num_heads={self.kv_num_heads}, head_size={self.head_size}, "
            f"paged_kv_block_size={self.paged_kv_block_size} rotary={self.rotary}, "
            f"rotary_interleaved={self.rotary_interleaved}, packed={self.packed}, softcap={self.softcap}, "
            f"ep={short_ep})"
        )


def create_paged_attention_graph(
    config,
    num_tokens,
    num_blocks,
    max_blocks_per_sequence,
    local_window_size=-1,
):
    nodes = [
        helper.make_node(
            "PagedAttention",
            [
                "query",
                "key" if not config.packed else "",
                "value" if not config.packed else "",
                "key_cache",
                "value_cache",
                "cumulative_sequence_length",
                "past_seqlens",
                "block_table",
                "cos_cache" if config.rotary else "",
                "sin_cache" if config.rotary else "",
            ],
            ["output", "key_cache_out", "value_cache_out"],
            "PagedAttention_0",
            num_heads=config.num_heads,
            kv_num_heads=config.kv_num_heads,
            local_window_size=local_window_size,
            do_rotary=config.rotary,
            rotary_interleaved=config.rotary_interleaved,
            softcap=config.softcap,
            domain="com.microsoft",
        ),
    ]

    graph_input = [
        helper.make_tensor_value_info(
            "query",
            TensorProto.FLOAT16,
            [
                num_tokens,
                (config.num_heads * config.head_size)
                if not config.packed
                else (config.num_heads * config.head_size + 2 * config.kv_num_heads * config.head_size),
            ],
        ),
        helper.make_tensor_value_info(
            "key_cache",
            TensorProto.FLOAT16,
            [
                num_blocks,
                config.paged_kv_block_size,
                config.kv_num_heads,
                config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "value_cache",
            TensorProto.FLOAT16,
            [
                num_blocks,
                config.paged_kv_block_size,
                config.kv_num_heads,
                config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "cumulative_sequence_length",
            TensorProto.INT32,
            [config.batch_size + 1],
        ),
        helper.make_tensor_value_info(
            "past_seqlens",
            TensorProto.INT32,
            [config.batch_size],
        ),
        helper.make_tensor_value_info(
            "block_table",
            TensorProto.INT32,
            [config.batch_size, max_blocks_per_sequence],
        ),
    ]
    if not config.packed:
        graph_input += [
            helper.make_tensor_value_info(
                "key",
                TensorProto.FLOAT16,
                [
                    num_tokens,
                    config.kv_num_heads * config.head_size,
                ],
            ),
            helper.make_tensor_value_info(
                "value",
                TensorProto.FLOAT16,
                [
                    num_tokens,
                    config.kv_num_heads * config.head_size,
                ],
            ),
        ]
    if config.rotary:
        graph_input += [
            helper.make_tensor_value_info(
                "cos_cache",
                TensorProto.FLOAT16,
                [
                    config.total_sequence_length,
                    (math.floor(config.head_size / 16) * 16) // 2,
                ],
            ),
            helper.make_tensor_value_info(
                "sin_cache",
                TensorProto.FLOAT16,
                [
                    config.total_sequence_length,
                    (math.floor(config.head_size / 16) * 16) // 2,
                ],
            ),
        ]

    graph_output = [
        helper.make_tensor_value_info(
            "output",
            TensorProto.FLOAT16,
            [num_tokens, config.num_heads * config.head_size],
        ),
        helper.make_tensor_value_info(
            "key_cache_out",
            TensorProto.FLOAT16,
            [
                num_blocks,
                config.paged_kv_block_size,
                config.kv_num_heads,
                config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "value_cache_out",
            TensorProto.FLOAT16,
            [
                num_blocks,
                config.paged_kv_block_size,
                config.kv_num_heads,
                config.head_size,
            ],
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "PagedAttention_Graph",
        graph_input,
        graph_output,
    )

    model = helper.make_model(graph)
    return model.SerializeToString()


def rotary_options_for_current_os():
    # Reference implementation of rotary uses triton, which is not available in Windows.
    # So we only test rotary in Linux right now.
    return [(False, False)] if platform.system() != "Linux" else [(True, False), (True, True), (False, False)]


def paged_attention_func(
    config,
    query,
    key,
    value,
    key_cache,
    value_cache,
    cumulative_sequence_length,
    past_seqlens,
    block_table,
    cos=None,
    sin=None,
    window_size=-1,
):
    num_tokens = cumulative_sequence_length[-1].item()
    num_blocks = key_cache.shape[0]
    max_blocks_per_sequence = block_table.shape[1]
    onnx_model_str = create_paged_attention_graph(
        config,
        num_tokens,
        num_blocks,
        max_blocks_per_sequence,
        local_window_size=window_size,
    )
    ort_inputs = {
        "query": query.detach().cpu().numpy(),
        "key_cache": OrtValue.ortvalue_from_numpy(key_cache.detach().cpu().numpy(), "cuda", 0),
        "value_cache": OrtValue.ortvalue_from_numpy(value_cache.detach().cpu().numpy(), "cuda", 0),
        "cumulative_sequence_length": cumulative_sequence_length.detach().cpu().numpy(),
        "past_seqlens": past_seqlens.detach().cpu().numpy(),
        "block_table": block_table.detach().cpu().numpy(),
    }
    sess_options = SessionOptions()
    ort_session = InferenceSession(onnx_model_str, sess_options, providers=[config.ep])
    io_binding = ort_session.io_binding()
    if key is not None and value is not None:
        ort_inputs["key"] = key.detach().cpu().numpy()
        ort_inputs["value"] = value.detach().cpu().numpy()
        io_binding.bind_cpu_input("key", ort_inputs["key"])
        io_binding.bind_cpu_input("value", ort_inputs["value"])
    if cos is not None and sin is not None:
        ort_inputs["cos_cache"] = cos.detach().cpu().numpy()
        ort_inputs["sin_cache"] = sin.detach().cpu().numpy()
        io_binding.bind_cpu_input("cos_cache", ort_inputs["cos_cache"])
        io_binding.bind_cpu_input("sin_cache", ort_inputs["sin_cache"])
    io_binding.bind_cpu_input("query", ort_inputs["query"])
    io_binding.bind_input(
        "key_cache", "cuda", 0, numpy.float16, ort_inputs["key_cache"].shape(), ort_inputs["key_cache"].data_ptr()
    )
    io_binding.bind_input(
        "value_cache", "cuda", 0, numpy.float16, ort_inputs["value_cache"].shape(), ort_inputs["value_cache"].data_ptr()
    )
    io_binding.bind_cpu_input("cumulative_sequence_length", ort_inputs["cumulative_sequence_length"])
    io_binding.bind_cpu_input("past_seqlens", ort_inputs["past_seqlens"])
    io_binding.bind_cpu_input("block_table", ort_inputs["block_table"])
    io_binding.bind_output("output")
    io_binding.bind_ortvalue_output("key_cache_out", ort_inputs["key_cache"])
    io_binding.bind_ortvalue_output("value_cache_out", ort_inputs["value_cache"])
    ort_session.run_with_iobinding(io_binding)
    output, key_cache_out, value_cache_out = io_binding.copy_outputs_to_cpu()
    output = torch.tensor(numpy.array(output))
    return output, key_cache_out, value_cache_out


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
):
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
            col_idx < row_idx + sk - sq - window_size[0],
        )


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    use_smooth_softmax=False,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling k, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if softcap > 0:
        scores = scores / softcap
        scores = scores.tanh()
        scores = scores * softcap
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
        )
        scores.masked_fill_(local_mask, float("-inf"))

    if use_smooth_softmax:
        head_sink = None
        attention = smooth_softmax_ref(scores, head_sink)
    else:
        attention = torch.softmax(scores, dim=-1)

    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def rotary_embedding(*args, **kwargs):
    # Use local import since triton is not available in Windows.
    from rotary_flash import apply_rotary_emb  # noqa: PLC0415

    return apply_rotary_emb(*args, **kwargs)


def unpad_qkv(config: Config, q, k, v, cum_seqlens):
    token_count = cum_seqlens[-1]
    q_unpad = torch.zeros(
        token_count,
        config.num_heads * config.head_size,
        dtype=torch.float16,
        device="cuda",
    )
    k_unpad = torch.zeros(
        token_count,
        config.kv_num_heads * config.head_size,
        dtype=torch.float16,
        device="cuda",
    )
    v_unpad = torch.zeros(
        token_count,
        config.kv_num_heads * config.head_size,
        dtype=torch.float16,
        device="cuda",
    )
    for i in range(config.batch_size):
        new_seqlen = cum_seqlens[i + 1] - cum_seqlens[i]
        q_unpad[cum_seqlens[i] : cum_seqlens[i + 1]] = rearrange(q[i, :new_seqlen], "s n h -> s (n h)")
        k_unpad[cum_seqlens[i] : cum_seqlens[i + 1]] = rearrange(k[i, :new_seqlen], "s n h -> s (n h)")
        v_unpad[cum_seqlens[i] : cum_seqlens[i + 1]] = rearrange(v[i, :new_seqlen], "s n h -> s (n h)")
    return q_unpad, k_unpad, v_unpad


def generate_block_kvcache(config: Config, device, dtype):
    num_blocks = math.ceil(config.total_sequence_length / config.paged_kv_block_size) * config.batch_size * 3
    k_cache_paged = torch.randn(
        num_blocks, config.paged_kv_block_size, config.kv_num_heads, config.head_size, device=device, dtype=dtype
    )
    v_cache_paged = torch.randn(
        num_blocks, config.paged_kv_block_size, config.kv_num_heads, config.head_size, device=device, dtype=dtype
    )
    block_table = rearrange(
        torch.randperm(num_blocks, dtype=torch.int32, device=device),
        "(b nblocks) -> b nblocks",
        b=config.batch_size,
    )
    k_cache = rearrange(
        # pytorch 1.12 doesn't have indexing with int32
        k_cache_paged[block_table.to(dtype=torch.long).flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=config.batch_size,
    )[:, : config.total_sequence_length]
    v_cache = rearrange(
        v_cache_paged[block_table.to(dtype=torch.long).flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=config.batch_size,
    )[:, : config.total_sequence_length]
    return k_cache, v_cache, block_table, k_cache_paged, v_cache_paged


def parity_check_paged_attention(
    config: Config,
    rtol=1e-3,
    atol=1e-3,
):
    # Generate padded inputs
    q = torch.randn(
        config.batch_size,
        config.sequence_length,
        config.num_heads,
        config.head_size,
        device="cuda",
        dtype=torch.float16,
        requires_grad=False,
    )
    k_new = torch.randn(
        config.batch_size,
        config.sequence_length,
        config.kv_num_heads,
        config.head_size,
        device="cuda",
        dtype=torch.float16,
        requires_grad=False,
    )
    v_new = torch.randn(
        config.batch_size,
        config.sequence_length,
        config.kv_num_heads,
        config.head_size,
        device="cuda",
        dtype=torch.float16,
        requires_grad=False,
    )

    # Generate random sequence lengths
    past_seqlens = torch.randint(
        0,
        config.total_sequence_length - config.sequence_length + 1,  # one above highest integer to be drawn
        (config.batch_size,),
        dtype=torch.int32,
        device="cuda",
    )
    new_seqlens = torch.randint(
        1,
        config.sequence_length + 1,
        (config.batch_size,),
        dtype=torch.int32,
        device="cuda",
    )
    cum_seqlens = torch.cat(
        (torch.tensor([0], dtype=torch.int32, device="cuda"), torch.cumsum(new_seqlens, dim=0))
    ).type(torch.int32)
    total_seqlens = past_seqlens + new_seqlens

    q_unpad, k_unpad, v_unpad = unpad_qkv(config, q, k_new, v_new, cum_seqlens)

    # Generate kv cache and associated block-based data structures
    k_cache, v_cache, block_table, k_cache_paged, v_cache_paged = generate_block_kvcache(config, "cuda", torch.float16)

    # Set window size for local / causal
    window_size = (-1, -1)
    left_window_size = -1
    if config.local:
        left_window_size = random.randint(0, config.total_sequence_length - 1)  # random.randint is inclusive
        window_size = (left_window_size, 0)
    else:
        left_window_size = -1
        window_size = (-1, 0)

    # Apply rotary embedding for reference implementation
    if config.rotary:
        rotary_fraction = 1.0
        rotary_dim = math.floor(int(rotary_fraction * config.head_size) / 16) * 16
        angle = torch.rand(config.total_sequence_length, rotary_dim // 2, device="cuda") * 2 * math.pi
        cos = torch.cos(angle).to(dtype=torch.float16)
        sin = torch.sin(angle).to(dtype=torch.float16)
        q_ro = rotary_embedding(q, cos, sin, seqlen_offsets=past_seqlens, interleaved=config.rotary_interleaved)
        k_ro = rotary_embedding(k_new, cos, sin, seqlen_offsets=past_seqlens, interleaved=config.rotary_interleaved)
    else:
        cos, sin = None, None
        q_ro, k_ro = q, k_new

    # Update reference kv cache
    k_cache_ref = k_cache.clone()
    v_cache_ref = v_cache.clone()
    total_range = rearrange(torch.arange(config.total_sequence_length, device="cuda"), "s -> 1 s")
    past_seqlens_expanded = rearrange(past_seqlens, "b -> b 1")
    update_mask = torch.logical_and(
        past_seqlens_expanded <= total_range, total_range < past_seqlens_expanded + config.sequence_length
    )
    k_cache_ref[update_mask] = rearrange(k_ro, "b s ... -> (b s) ...")
    v_cache_ref[update_mask] = rearrange(v_new, "b s ... -> (b s) ...")
    k_cache_rep = repeat(k_cache_ref, "b s h d -> b s (h g) d", g=config.num_heads // config.kv_num_heads)
    v_cache_rep = repeat(v_cache_ref, "b s h d -> b s (h g) d", g=config.num_heads // config.kv_num_heads)

    # Create padding masks for reference implementation
    total_seqlens_expanded = rearrange(total_seqlens, "b -> b 1")
    key_padding_mask = total_range < total_seqlens_expanded
    query_range = rearrange(torch.arange(config.sequence_length, device="cuda"), "s -> 1 s")
    new_seqlens_expanded = rearrange(new_seqlens, "b -> b 1")
    query_padding_mask = query_range < new_seqlens_expanded

    # Run reference implementation of attention
    out_ref, _ = attention_ref(
        q_ro,
        k_cache_rep,
        v_cache_rep,
        query_padding_mask,
        key_padding_mask,
        0.0,
        None,
        causal=True,
        window_size=window_size,
        softcap=config.softcap,
    )
    out_ref = out_ref.detach().cpu().numpy()

    if config.packed:
        q_unpad = torch.concatenate([q_unpad, k_unpad, v_unpad], dim=1)
        k_unpad = None
        v_unpad = None
    out, updated_k_cache_paged, updated_v_cache_paged = paged_attention_func(
        config,
        q_unpad,
        k_unpad,
        v_unpad,
        k_cache_paged,
        v_cache_paged,
        cum_seqlens,
        past_seqlens,
        block_table,
        cos,
        sin,
        left_window_size,
    )
    num_tokens = q_unpad.shape[0]
    out = torch.reshape(out, (num_tokens, config.num_heads, config.head_size))
    out = out.detach().cpu().numpy()

    err_msg = f" with {config}"
    # Make sure past-present buffer updating correctly
    present_k = rearrange(
        updated_k_cache_paged[block_table.to(dtype=torch.long).flatten().cpu()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=config.batch_size,
    )[:, : config.total_sequence_length]
    present_v = rearrange(
        updated_v_cache_paged[block_table.to(dtype=torch.long).flatten().cpu()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=config.batch_size,
    )[:, : config.total_sequence_length]
    for i in range(config.batch_size):
        numpy.testing.assert_allclose(
            present_k[i, : total_seqlens[i]],
            k_cache_ref[i, : total_seqlens[i]].detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
            err_msg=err_msg,
        )
        numpy.testing.assert_allclose(
            present_v[i, : total_seqlens[i]],
            v_cache_ref[i, : total_seqlens[i]].detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
            err_msg=err_msg,
        )
        new_seqlen = cum_seqlens[i + 1] - cum_seqlens[i]
        out_i = out[cum_seqlens[i] : cum_seqlens[i + 1]]
        out_ref_i = out_ref[i, :new_seqlen]
        numpy.testing.assert_allclose(out_i, out_ref_i, rtol=rtol, atol=atol, equal_nan=True, err_msg=err_msg)


def has_flash_attention():
    if not torch.cuda.is_available():
        return False
    if "CUDAExecutionProvider" not in get_available_providers():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8 and (
        platform.system() == "Linux"
        or (platform.system() == "Windows" and version.parse(torch.version.cuda) >= version.parse("12.0"))
    )


def paged_attention_test_cases():
    batches = [4] if pipeline_mode else [1, 3, 5]
    seqs = (
        [(1025, 2047)]
        if pipeline_mode
        else [
            (3, 1024),
            (1, 339),
            (408, 800),
            (333, 799),
            (64, 2048),
            (837, 4000),
            (17, 49),
            (257, 257),
            (459, 459),
        ]
    )
    num_h = [(32, 8)] if pipeline_mode else [(6, 6), (6, 3), (9, 9), (9, 3)]
    h_sizes = [256] if pipeline_mode else [32, 40, 64, 80, 96, 128, 160, 192, 224, 256]
    block_sizes = [256] if pipeline_mode else [256, 512]

    for b in batches:
        for s, s2 in seqs:
            for n, n2 in num_h:
                for h in h_sizes:
                    for block_size in block_sizes:
                        for local in [False, True]:
                            for rotary, rotary_interleaved in rotary_options_for_current_os():
                                for packed in [False, True]:
                                    for softcap in [0.0, 50.0]:
                                        if rotary and h % 16 > 0:
                                            continue

                                        config = Config(
                                            b,
                                            s,
                                            s2,
                                            n,
                                            n2,
                                            h,
                                            block_size,
                                            local,
                                            rotary,
                                            rotary_interleaved,
                                            packed,
                                            softcap,
                                        )
                                        yield (
                                            str(config),
                                            config,
                                        )


@unittest.skipIf(not has_flash_attention(), reason="Flash Attention is not available, skipping tests.")
class TestPagedAttention(unittest.TestCase):
    @parameterized.expand(paged_attention_test_cases())
    def test_paged_attention(self, _, config):
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
