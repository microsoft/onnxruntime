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
import random
from ctypes import c_int32

import numpy
import torch
from bert_padding import pad_input, unpad_input
from einops import rearrange, repeat
from onnx import TensorProto, helper

from onnxruntime import InferenceSession, SessionOptions

torch.manual_seed(0)


class Config:
    batch_size = 0
    sequence_length = 0
    kv_sequence_length = 0
    past_sequence_length = 0
    num_heads = 0
    kv_num_heads = 0
    head_size = 0

    def __init__(self, b, s, s2, sp, n, n2, h):
        self.batch_size = b
        self.sequence_length = s
        self.kv_sequence_length = s2
        self.past_sequence_length = sp
        self.num_heads = n
        self.kv_num_heads = n2
        self.head_size = h

class ConfigWithPastKV(Config):
    past_sequence_length = 0

def create_packed_multihead_attention_graph(config):
    nodes = [
        helper.make_node(
            "PackedMultiHeadAttention",
            [
                "query",
                "",
                "",
                "",
                "token_offset",
                "cumulative_sequence_length",
            ],
            ["output"],
            "PackedMultiHeadAttention_0",
            num_heads=config.num_heads,
            domain="com.microsoft",
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "PackedMultiHeadAttention_Graph",
        [
            helper.make_tensor_value_info(
                "query",
                TensorProto.FLOAT16,
                [
                    -1,
                    config.num_heads,
                    3,
                    config.head_size,
                ],
            ),
            helper.make_tensor_value_info(
                "token_offset", TensorProto.INT32, [config.batch_size, config.sequence_length]
            ),
            helper.make_tensor_value_info("cumulative_sequence_length", TensorProto.INT32, [config.batch_size + 1]),
        ],
        [
            helper.make_tensor_value_info(
                "output",
                TensorProto.FLOAT16,
                [-1, config.num_heads * config.head_size],
            ),
        ],
    )

    model = helper.make_model(graph)
    return model.SerializeToString()


def create_multihead_attention_graph(config):
    nodes = [
        helper.make_node(
            "MultiHeadAttention",
            [
                "query",
                "key",
                "value",
            ],
            ["output"],
            "MultiHeadAttention_0",
            num_heads=config.num_heads,
            domain="com.microsoft",
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "MultiHeadAttention_Graph",
        [
            helper.make_tensor_value_info(
                "query",
                TensorProto.FLOAT16,
                [
                    config.batch_size,
                    config.sequence_length,
                    config.num_heads * config.head_size,
                ],
            ),
            helper.make_tensor_value_info(
                "key",
                TensorProto.FLOAT16,
                [
                    config.batch_size,
                    config.kv_sequence_length,
                    config.num_heads * config.head_size,
                ],
            ),
            helper.make_tensor_value_info(
                "value",
                TensorProto.FLOAT16,
                [
                    config.batch_size,
                    config.kv_sequence_length,
                    config.num_heads * config.head_size,
                ],
            ),
        ],
        [
            helper.make_tensor_value_info(
                "output",
                TensorProto.FLOAT16,
                [config.batch_size, config.sequence_length, config.num_heads * config.head_size],
            ),
        ],
    )

    model = helper.make_model(graph)
    return model.SerializeToString()


def create_group_query_attention_graph(config, causal=False, new_kv=False):
    nodes = [
        helper.make_node(
            "GroupQueryAttention",
            [
                "query",
                "key",
                "value",
                "past_key" if new_kv else "",
                "past_value" if new_kv else "",
                "past_sequence_length" if new_kv else "",
            ],
            [
                "output"
            ],
            "GroupQueryAttention_0",
            num_heads=config.num_heads,
            kv_num_heads=config.kv_num_heads,
            unidirectional=1 if causal else 0,
            domain="com.microsoft",
        ),
    ]

    graph_input = [
        helper.make_tensor_value_info(
            "query",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                config.sequence_length,
                config.num_heads * config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "key",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                config.kv_sequence_length if not new_kv else config.sequence_length,
                config.kv_num_heads * config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "value",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                config.kv_sequence_length if not new_kv else config.sequence_length,
                config.kv_num_heads * config.head_size,
            ],
        ),
    ]

    if new_kv:
        graph_input += [
            helper.make_tensor_value_info(
                "past_key",
                TensorProto.FLOAT16,
                [
                    config.batch_size,
                    config.kv_sequence_length,
                    config.kv_num_heads,
                    config.head_size,
                ],
            ),
            helper.make_tensor_value_info(
                "past_value",
                TensorProto.FLOAT16,
                [
                    config.batch_size,
                    config.kv_sequence_length,
                    config.kv_num_heads,
                    config.head_size,
                ],
            ),
            helper.make_tensor_value_info("past_sequence_length", TensorProto.INT32, [1],)
        ]

    graph = helper.make_graph(
        nodes,
        "GroupQueryAttention_Graph",
        graph_input,
        [
            helper.make_tensor_value_info(
                "output",
                TensorProto.FLOAT16,
                [config.batch_size, config.sequence_length, config.num_heads * config.head_size],
            ),
            # TODO(aciddelgado): return present key and value?
        ],
    )

    model = helper.make_model(graph)
    return model.SerializeToString()


def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(max(1, max_seqlen - 20), max_seqlen, (batch_size, 1), device=device)
    else:
        lengths = torch.randint(max_seqlen // 3, max_seqlen, (batch_size, 1), device=device)
    padding_mask = repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size) < lengths
    return padding_mask


def generate_qkv(q, k, v, query_padding_mask=None, key_padding_mask=None, kvpacked=False, qkvpacked=False):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    assert k.shape == (batch_size, seqlen_k, nheads_k, d)
    assert v.shape == (batch_size, seqlen_k, nheads_k, d)

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, query_padding_mask)

        def output_pad_fn(output_unpad):
            return pad_input(output_unpad, indices_q, batch_size, seqlen_q)

    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(
            0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q_unpad.device
        )
        max_seqlen_q = seqlen_q

        def output_pad_fn(output_unpad):
            return rearrange(output_unpad, "(b s) h d -> b s h d", b=batch_size)

    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(k, key_padding_mask)
        v_unpad, _, _, _ = unpad_input(v, key_padding_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = torch.arange(
            0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=k_unpad.device
        )
        max_seqlen_k = seqlen_k

    if qkvpacked:
        assert (query_padding_mask == key_padding_mask).all()
        assert nheads == nheads_k
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        qkv = torch.stack([q, k, v], dim=2)
        if query_padding_mask is not None:

            def dqkv_pad_fn(dqkv_unpad):
                return pad_input(dqkv_unpad, indices_q, batch_size, seqlen_q)

        else:

            def dqkv_pad_fn(dqkv_unpad):
                return rearrange(dqkv_unpad, "(b s) t h d -> b s t h d", b=batch_size)

        return (
            qkv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            max_seqlen_q,
            qkv.detach().requires_grad_(),
            output_pad_fn,
            dqkv_pad_fn,
        )
    elif kvpacked:
        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
        kv = torch.stack([k, v], dim=2)
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:

            def dkv_pad_fn(dkv_unpad):
                return pad_input(dkv_unpad, indices_k, batch_size, seqlen_k)

        else:

            def dkv_pad_fn(dkv_unpad):
                return rearrange(dkv_unpad, "(b s) t h d -> b s t h d", b=batch_size)

        return (
            q_unpad.detach().requires_grad_(),
            kv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            kv.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dkv_pad_fn,
        )
    else:
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:

            def dk_pad_fn(dk_unpad):
                return pad_input(dk_unpad, indices_k, batch_size, seqlen_k)

        else:

            def dk_pad_fn(dk_unpad):
                return rearrange(dk_unpad, "(b s) h d -> b s h d", b=batch_size)

        return (
            q_unpad.detach().requires_grad_(),
            k_unpad.detach().requires_grad_(),
            v_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            k.detach().requires_grad_(),
            v.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        )


def create_inputs(config: Config, kv_packed=False, qkv_packed=True):
    qkv = torch.randn(
        config.batch_size,
        config.sequence_length,
        3,
        config.num_heads,
        config.head_size,
        device="cuda",
        dtype=torch.float16,
        requires_grad=False,
    )
    key_padding_mask = generate_random_padding_mask(
        config.sequence_length, config.batch_size, device="cuda", mode="random"
    )
    qkv_unpad, cu_seqlens, max_seqlen, qkv, output_pad_fn, dqkv_pad_fn = generate_qkv(
        *qkv.unbind(dim=2), key_padding_mask, key_padding_mask, kv_packed, qkv_packed
    )
    return qkv_unpad, cu_seqlens, max_seqlen, qkv, output_pad_fn, dqkv_pad_fn, key_padding_mask


def generate_token_offset(cu_seqlens, max_seqlen):
    token_offset = []
    token_padset = []  # These are the indices that contain padding tokens
    for i in range(1, len(cu_seqlens)):
        start = i - 1
        pre_seqlen = cu_seqlens[i - 1]
        seqlen = cu_seqlens[i]
        token_offset += range(start * max_seqlen, (start * max_seqlen) + (seqlen - pre_seqlen))
        token_padset += range((start * max_seqlen) + (seqlen - pre_seqlen), i * max_seqlen)
    return numpy.asarray(token_offset + token_padset, dtype=numpy.int32)


def flash_attn_varlen_qkvpacked_func(qkv_unpad, cu_seqlens, token_offset, config, causal=False):
    onnx_model_str = create_packed_multihead_attention_graph(config)
    qkv_unpad = torch.swapdims(qkv_unpad, 1, 2)
    ort_inputs = {
        "query": qkv_unpad.detach().cpu().numpy(),
        "token_offset": token_offset,
        "cumulative_sequence_length": cu_seqlens.cpu().numpy(),
    }
    sess_options = SessionOptions()
    ort_session = InferenceSession(onnx_model_str, sess_options, providers=["CUDAExecutionProvider"])
    ort_output = ort_session.run(None, ort_inputs)
    output = torch.tensor(ort_output)
    return output


def flash_attn_func(q, k, v, config, gqa=False, causal=False, new_k=None, new_v=None):
    new_kv = new_k is not None
    if gqa:
        onnx_model_str = create_group_query_attention_graph(config, causal, new_kv=new_kv)
    else:
        onnx_model_str = create_multihead_attention_graph(config)
    q = torch.reshape(q, (config.batch_size, config.sequence_length, -1))
    if not new_kv:
        k = torch.reshape(k, (config.batch_size, config.kv_sequence_length, -1))
        v = torch.reshape(v, (config.batch_size, config.kv_sequence_length, -1))
        ort_inputs = {
            "query": q.detach().cpu().numpy(),
            "key": k.detach().cpu().numpy(),
            "value": v.detach().cpu().numpy(),
        }
    elif new_k is not None and new_v is not None:
        past_k = k
        past_v = v
        new_k = torch.reshape(new_k, (config.batch_size, config.sequence_length, -1))
        new_v = torch.reshape(new_v, (config.batch_size, config.sequence_length, -1))
        ort_inputs = {
            "query": q.detach().cpu().numpy(),
            "key": new_k.detach().cpu().numpy(),
            "value": new_v.detach().cpu().numpy(),
            "past_key": past_k.detach().cpu().numpy(),
            "past_value": past_v.detach().cpu().numpy(),
            "past_sequence_length": torch.tensor([config.past_sequence_length], dtype=torch.int32).detach().cpu().numpy(),
        }
    sess_options = SessionOptions()
    ort_session = InferenceSession(onnx_model_str, sess_options, providers=["CUDAExecutionProvider"])
    ort_output = ort_session.run(None, ort_inputs)
    output = torch.tensor(ort_output)
    return output


def construct_causal_mask(seqlen_q, seqlen_k, query_padding_mask=None, key_padding_mask=None,
                          device=None):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    return col_idx > row_idx + sk - sq


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    upcast=True,
    reorder_ops=False,
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
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling k, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
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
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if causal:
        # causal_mask = torch.triu(
        #     torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=q.device), 1
        # )
        causal_mask = construct_causal_mask(
            seqlen_q, seqlen_k, query_padding_mask, key_padding_mask, q.device
        )
        scores.masked_fill_(causal_mask, float("-inf"))
    attention = torch.softmax(scores, dim=-1)
    if causal:  # Some rows are completely masked out so we fill them with zero instead of NaN
        attention = attention.masked_fill(torch.all(causal_mask, dim=-1, keepdim=True), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def attention_qkvpacked_ref(
    qkv, key_padding_mask=None, dropout_p=0.0, dropout_mask=None, causal=False, upcast=True, reorder_ops=False
):
    return attention_ref(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        key_padding_mask,
        key_padding_mask,
        dropout_p,
        dropout_mask,
        upcast=upcast,
        causal=causal,
        reorder_ops=reorder_ops,
    )


def parity_check(
    config,
    packed,
    gqa=False,
    causal=False,
    rtol=1e-3,
    atol=1e-3,
):
    if packed:
        qkv_unpad, cu_seqlens, _, qkv, output_pad_fn, _, key_padding_mask = create_inputs(config)
        token_offset = generate_token_offset(cu_seqlens, config.sequence_length).reshape(
            (config.batch_size, config.sequence_length)
        )
        # ORT Flash
        out_unpad = flash_attn_varlen_qkvpacked_func(qkv_unpad, cu_seqlens, token_offset, config, causal=causal)
        out_unpad = torch.squeeze(out_unpad, 0)
        out = torch.reshape(
            output_pad_fn(out_unpad), (config.batch_size, config.sequence_length, config.num_heads, config.head_size)
        )
        out = out.detach().cpu().numpy()
        # Pytorch to compare
        out_ref, _ = attention_qkvpacked_ref(qkv, key_padding_mask, 0.0, None, causal=causal)
        out_ref = out_ref.detach().cpu().numpy()
    else:
        q = torch.randn(
            config.batch_size,
            config.sequence_length,
            config.num_heads,
            config.head_size,
            device="cuda",
            dtype=torch.float16,
            requires_grad=False,
        )
        k = torch.randn(
            config.batch_size,
            config.kv_sequence_length,
            config.kv_num_heads,
            config.head_size,
            device="cuda",
            dtype=torch.float16,
            requires_grad=False,
        )
        v = torch.randn(
            config.batch_size,
            config.kv_sequence_length,
            config.kv_num_heads,
            config.head_size,
            device="cuda",
            dtype=torch.float16,
            requires_grad=False,
        )
        out = flash_attn_func(q, k, v, config, gqa=gqa, causal=causal)
        out = torch.squeeze(out, 0)
        out = torch.reshape(out, (config.batch_size, config.sequence_length, config.num_heads, config.head_size))
        out = out.detach().cpu().numpy()
        # Pytorch to compare
        out_ref, _ = attention_ref(q, k, v, None, None, 0.0, None, causal=causal)
        out_ref = out_ref.detach().cpu().numpy()

        # print(out[0, 0, 0, :20])
        # print(out_ref[0, 0, 0, :20])
    # Compare results
    print(
        " causal:",
        causal,
        " B:",
        config.batch_size,
        " S:",
        config.sequence_length,
        " N:",
        config.num_heads,
        " kvN:",
        config.kv_num_heads,
        " h:",
        config.head_size,
        " Mean Error:",
        numpy.mean(numpy.abs(out - out_ref)),
        numpy.allclose(
            out,
            out_ref,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        ),
    )


def parity_check_gqa(
    config,
    causal=False,
    new_kv=False,
    rtol=1e-3,
    atol=1e-3,
):
    q = torch.randn(
        config.batch_size,
        config.sequence_length,
        config.num_heads,
        config.head_size,
        device="cuda",
        dtype=torch.float16,
        requires_grad=False,
    )
    k = torch.randn(
        config.batch_size,
        config.kv_sequence_length,
        config.kv_num_heads,
        config.head_size,
        device="cuda",
        dtype=torch.float16,
        requires_grad=False,
    )
    v = torch.randn(
        config.batch_size,
        config.kv_sequence_length,
        config.kv_num_heads,
        config.head_size,
        device="cuda",
        dtype=torch.float16,
        requires_grad=False,
    )
    new_k = None
    new_v = None
    if new_kv:
        new_k = torch.randn(
            config.batch_size,
            config.sequence_length,
            config.kv_num_heads,
            config.head_size,
            device="cuda",
            dtype=torch.float16,
            requires_grad=False,
        )
        new_v = torch.randn(
            config.batch_size,
            config.sequence_length,
            config.kv_num_heads,
            config.head_size,
            device="cuda",
            dtype=torch.float16,
            requires_grad=False,
        )
    out = flash_attn_func(q, k, v, config, gqa=True, causal=causal, new_k=new_k, new_v=new_v)
    out = torch.squeeze(out, 0)
    out = torch.reshape(out, (config.batch_size, config.sequence_length, config.num_heads, config.head_size))
    out = out.detach().cpu().numpy()
    # Pytorch to compare
    if not new_kv:
        out_ref, _ = attention_ref(q, k, v, None, None, 0.0, None, causal=causal)
        out_ref = out_ref.detach().cpu().numpy()
    else:
        k_cache_ref = k.clone()
        v_cache_ref = v.clone()
        cache_seqlens = torch.tensor([config.past_sequence_length], device="cuda").repeat(config.batch_size)
        arange = rearrange(torch.arange(config.kv_sequence_length, device="cuda"), "s -> 1 s")
        cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
        if new_kv:
            update_mask = torch.logical_and(cache_seqlens_expanded <= arange, arange < cache_seqlens_expanded + config.sequence_length)
            k_cache_ref[update_mask] = rearrange(new_k, "b s ... -> (b s) ...")
            v_cache_ref[update_mask] = rearrange(new_v, "b s ... -> (b s) ...")
        k_cache_rep = repeat(k_cache_ref, "b s h d -> b s (h g) d", g=config.num_heads // config.kv_num_heads)
        v_cache_rep = repeat(v_cache_ref, "b s h d -> b s (h g) d", g=config.num_heads // config.kv_num_heads)
        key_padding_mask = arange < cache_seqlens_expanded + (config.sequence_length if new_kv else 0)
        out_ref, _ = attention_ref(q, k_cache_rep, v_cache_rep, None, key_padding_mask, 0.0, None, causal=causal)
        out_ref = out_ref.detach().cpu().numpy()
        # out_pt, _ = attention_ref(q, k_cache_rep, v_cache_rep, None, key_padding_mask, 0.0, None, causal=causal,
        #                           upcast=False, reorder_ops=True)

    # print(out[0, 0, 0, :20])
    # print(out_ref[0, 0, 0, :20])
    # Compare results
    print(
        " causal:",
        causal,
        " B:",
        config.batch_size,
        " S:",
        config.sequence_length,
        " kv S:",
        config.kv_sequence_length,
        " N:",
        config.num_heads,
        " kv N:",
        config.kv_num_heads,
        " h:",
        config.head_size,
        " Mean Error:",
        numpy.mean(numpy.abs(out - out_ref)),
        numpy.allclose(
            out,
            out_ref,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        ),
    )


if __name__ == "__main__":
    # print("-------- TEST PACKED MHA ---------")
    # for b in [5]:
    #     for s in [97, 128, 200, 256, 257, 384, 512, 768, 1024, 1025, 2048]:
    #         for n in [6]:
    #             for h in [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256]:
    #                 config = Config(b, s, s, 0, n, n, h)
    #                 parity_check(config, True)
    # print("-------- TEST MHA ---------")
    # for b in [5]:
    #     for s, s2 in [
    #         (113, 203),
    #         (128, 217),
    #         (113, 211),
    #         (108, 256),
    #         (256, 512),
    #         (512, 256),
    #         (1024, 1024),
    #         (1023, 1024),
    #         (1024, 1023),
    #         (2048, 2048),
    #     ]:
    #         for n in [6]:
    #             for h in [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256]:
    #                 config = Config(b, s, s2, 0, n, n, h)
    #                 parity_check(config, False)
    # print("-------- TEST GQA ---------")
    # for b in [5]:
    #     for s, s2 in [
    #         (113, 203),
    #         (128, 217),
    #         (113, 211),
    #         (108, 256),
    #         (256, 512),
    #         (512, 256),
    #         (1024, 1024),
    #         (1023, 1024),
    #         (1024, 1023),
    #         (2048, 2048),
    #     ]:
    #         for (n, n2) in [(6, 6), (6, 3), (9, 9), (9, 3)]:
    #             for h in [32, 40, 64, 80, 96, 128, 160, 192, 224, 256]:
    #                 for causal in [True, False]:
    #                     config = Config(b, s, s2, 0, n, n2, h)
    #                     parity_check_gqa(config, causal=causal)
    print("-------- TEST GQA PAST ---------")
    for b in [2]:
        for s, s2 in [
            (1, 128),
            (1, 339),
            (3, 1024),
            (64, 800),
            (64, 256),
            (3, 799),
            (64, 2048),
            (16, 20000),
            (1, 128 * 512),
            (16, 128 * 512),
            (128, 128),
        ]:
            for (n, n2) in [(6, 6), (6, 3), (9, 9), (9, 3)]:
                for h in [32, 40, 64, 80, 96, 128, 160, 192, 224, 256]:
                    for causal in [True]:
                        for new_kv in [True]:
                            random.seed(69)
                            sp = random.randint(1, s2-s) if s2-s > 0 else 0
                            config = Config(b, s, s2, sp, n, n2, h)
                            parity_check_gqa(config, causal=causal, new_kv=new_kv, rtol=1e-3, atol=1e-3)
