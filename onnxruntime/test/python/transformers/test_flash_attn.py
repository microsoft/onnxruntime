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

from typing import List, Optional, Tuple

import math

import numpy
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from einops import rearrange, repeat

from bert_padding import unpad_input, pad_input

torch.manual_seed(0)

"""
This is an example of export bart decoder attention with huggingface v3.5.1
def my_bart_attention_forward(
    self,
    query,
    key: Tensor,
    key_padding_mask: Optional[Tensor],
    layer_state: Optional[List[Tensor]],
    attn_mask: Optional[Tensor] = None,
    output_attentions: bool=False,
    use_past=torch.tensor(False),
):
    static_kv: bool = self.encoder_decoder_attention
    q_weight = self.q_proj.weight.transpose(0,1)
    q_weight = q_weight.reshape(self.embed_dim, self.embed_dim)

    kv_weight = torch.stack((self.k_v_proj.k_proj.weight.transpose(0,1), self.k_v_proj.v_proj.weight.transpose(0,1)), dim=1)
    kv_weight = kv_weight.reshape(self.embed_dim, 2 * self.embed_dim)

    bias = torch.stack((self.q_proj.bias, self.k_v_proj.k_proj.bias, self.k_v_proj.v_proj.bias), dim=0)
    bias = bias.reshape(3 * self.embed_dim)

    self_p_k, self_p_v, enc_dec_p_k, enc_dec_p_v = layer_state
    if static_kv:
        key_cache, value_cache = enc_dec_p_k, enc_dec_p_v
    else:
        key_cache, value_cache = self_p_k, self_p_v

    if not static_kv:
        key_padding_mask = torch.tensor(False)

    attn_output, new_key_cache, new_value_cache = torch.ops.onnxruntime.DecoderAttention(
                                                    query,
                                                    key,
                                                    q_weight,
                                                    kv_weight,
                                                    bias,
                                                    key_padding_mask,
                                                    key_cache,
                                                    value_cache,
                                                    torch.tensor(static_kv), #static_kv
                                                    use_past, #use_past
                                                    torch.tensor(True), #has_layer_state
                                                    torch.tensor(static_kv), #has_key_padding_mask
                                                    self.num_heads)

    if not use_past:
        if self.encoder_decoder_attention:
            layer_state[2] = new_key_cache
            layer_state[3] = new_value_cache
        else:
            layer_state[0] = new_key_cache
            layer_state[1] = new_value_cache
    else:
        if not self.encoder_decoder_attention:
            layer_state[0] = new_key_cache
            layer_state[1] = new_value_cache

    attn_output = self.out_proj(attn_output)

    return attn_output, None, layer_state
"""


class Config:
    batch_size = 0
    sequence_length = 0
    num_heads = 0
    head_size = 0
    embed_dim = 0

    def __init__(self, b, s, n, h):
        self.batch_size = b
        self.sequence_length = s
        self.num_heads = n
        self.head_size = h
        self.embed_dim = self.num_heads * self.head_size


def create_decoder_attention_graph(
    query,
    key,
    q_weight,
    kv_weight,
    bias,
    num_heads_,
    static_kv,
    use_past,
    has_layer_state,
    has_key_padding_mask,
):
    from onnx import TensorProto, helper

    S, B, NH = query.size()  # noqa: N806
    S2 = key.size()[0]  # noqa: N806
    N = num_heads_  # noqa: N806
    H = int(NH / N)  # noqa: N806

    nodes = [
        helper.make_node(
            "DecoderAttention",
            [
                "query",
                "key",
                "q_weight",
                "kv_weight",
                "bias",
                "key_padding_mask",
                "key_cache",
                "value_cache",
                "static_kv",
                "use_past",
                "has_layer_state",
                "has_key_padding_mask",
            ],
            ["output", "new_key_cache", "new_value_cache"],
            "DecoderAttention_0",
            num_heads=num_heads_,
            domain="com.microsoft",
        ),
    ]

    initializers = [
        helper.make_tensor("q_weight", TensorProto.FLOAT, [NH, NH], q_weight.flatten().tolist()),
        helper.make_tensor("kv_weight", TensorProto.FLOAT, [NH, 2 * NH], kv_weight.flatten().tolist()),
        helper.make_tensor("bias", TensorProto.FLOAT, [3 * NH], bias.flatten().tolist()),
        helper.make_tensor("static_kv", TensorProto.BOOL, [1], [static_kv]),
        helper.make_tensor("use_past", TensorProto.BOOL, [1], [use_past]),
        helper.make_tensor("has_layer_state", TensorProto.BOOL, [1], [has_layer_state]),
        helper.make_tensor("has_key_padding_mask", TensorProto.BOOL, [1], [has_key_padding_mask]),
    ]

    graph = helper.make_graph(
        nodes,
        "DecoderAttention_Graph",
        [
            helper.make_tensor_value_info("query", TensorProto.FLOAT, [S, B, NH]),
            helper.make_tensor_value_info("key", TensorProto.FLOAT, [S2, B, NH]),
            helper.make_tensor_value_info("key_padding_mask", TensorProto.BOOL, [B, "mask_len"]),
            helper.make_tensor_value_info("key_cache", TensorProto.FLOAT, [B, N, "cache_len", H]),
            helper.make_tensor_value_info("value_cache", TensorProto.FLOAT, [B, N, "cache_len", H]),
        ],
        [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [S, B, NH]),
            helper.make_tensor_value_info("new_key_cache", TensorProto.FLOAT, [B, N, "new_cache_len", H]),
            helper.make_tensor_value_info("new_value_cache", TensorProto.FLOAT, [B, N, "new_cache_len", H]),
        ],
        initializers,
    )

    model = helper.make_model(graph)
    return model.SerializeToString()


def create_packed_multihead_attention_graph(config):
    from onnx import TensorProto, helper, save

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
    save(model, "/home/aciddelgado/pmha_test_model")
    return model.SerializeToString()


def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(max(1, max_seqlen - 20), max_seqlen, (batch_size, 1), device=device)
    elif mode == "third":
        lengths = torch.randint(max_seqlen // 3, max_seqlen, (batch_size, 1), device=device)
    padding_mask = repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size) < lengths
    return padding_mask


# QKV Packed
def generate_qkv(q, k, v, query_padding_mask=None, key_padding_mask=None):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
    batch_size, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    assert k.shape == (batch_size, seqlen_k, nheads_k, d)
    assert v.shape == (batch_size, seqlen_k, nheads_k, d)

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, query_padding_mask)
        output_pad_fn = lambda output_unpad: pad_input(output_unpad, indices_q, batch_size, seqlen_q)
    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(
            0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q_unpad.device
        )
        max_seqlen_q = seqlen_q
        output_pad_fn = lambda output_unpad: rearrange(output_unpad, "(b s) h d -> b s h d", b=batch_size)

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

    assert (query_padding_mask == key_padding_mask).all()
    assert nheads == nheads_k
    qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
    qkv = torch.stack([q, k, v], dim=2)
    if query_padding_mask is not None:
        dqkv_pad_fn = lambda dqkv_unpad: pad_input(dqkv_unpad, indices_q, batch_size, seqlen_q)
    else:
        dqkv_pad_fn = lambda dqkv_unpad: rearrange(dqkv_unpad, "(b s) t h d -> b s t h d", b=batch_size)
    return (
        qkv_unpad.detach().requires_grad_(),
        cu_seqlens_q,
        max_seqlen_q,
        qkv.detach().requires_grad_(),
        output_pad_fn,
        dqkv_pad_fn,
    )


def create_inputs(config: Config):
    qkv = torch.randn(
        config.batch_size,
        config.sequence_length,
        3,
        config.num_heads,
        config.head_size,
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    key_padding_mask = generate_random_padding_mask(
        config.sequence_length, config.batch_size, device="cuda", mode="random"
    )
    qkv_unpad, cu_seqlens, max_seqlen, qkv, output_pad_fn, dqkv_pad_fn = generate_qkv(
        *qkv.unbind(dim=2), key_padding_mask, key_padding_mask
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

    from onnxruntime import InferenceSession, SessionOptions

    sess_options = SessionOptions()
    ort_session = InferenceSession(onnx_model_str, sess_options, providers=["CUDAExecutionProvider"])
    ort_output = ort_session.run(None, ort_inputs)
    output = torch.tensor(ort_output)

    return output


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
        causal_mask = torch.triu(torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=q.device), 1)
        scores.masked_fill_(causal_mask, float("-inf"))
    attention = torch.softmax(scores, dim=-1)
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
    rtol=1e-4,
    atol=1e-4,
):
    qkv_unpad, cu_seqlens, max_seqlen, qkv, output_pad_fn, dqkv_pad_fn, key_padding_mask = create_inputs(config)
    token_offset = generate_token_offset(cu_seqlens, config.sequence_length).reshape(
        (config.batch_size, config.sequence_length)
    )
    # ORT Flash
    out_unpad = flash_attn_varlen_qkvpacked_func(qkv_unpad, cu_seqlens, token_offset, config, causal=False)
    out_unpad = torch.squeeze(out_unpad, 0)
    out = torch.reshape(
        output_pad_fn(out_unpad), (config.batch_size, config.sequence_length, config.num_heads, config.head_size)
    )
    out = out.detach().cpu().numpy()
    # Pytorch to compare
    out_ref, _ = attention_qkvpacked_ref(qkv, key_padding_mask, 0.0, None, causal=False)
    out_ref = out_ref.detach().cpu().numpy()
    print(numpy.mean(out - out_ref))
    # # Compare results
    # print(
    #     " B:",
    #     config.batch_size,
    #     " S:",
    #     config.sequence_length,
    #     " N:",
    #     config.num_heads,
    #     " h:",
    #     config.head_size,
    #     numpy.allclose(
    #         out,
    #         out_ref,
    #         rtol=rtol,
    #         atol=atol,
    #         equal_nan=True,
    #     ),
    # )


if __name__ == "__main__":
    for b in [5]:
        for s in [97, 128, 200, 256, 257, 384, 512, 768, 1024, 1025, 2048]:
            for n in [6]:
                for h in [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256]:
                    config = Config(b, s, n, h)
                    parity_check(config)
                    # parity_check(config)
                    # parity_check(config)
                    # parity_check(config)
                    # parity_check(config)
                    # parity_check(config)
                    # parity_check(config)
                    # parity_check(config)
