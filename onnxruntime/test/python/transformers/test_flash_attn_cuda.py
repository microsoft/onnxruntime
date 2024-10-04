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
import os
import platform
import random
import unittest

import numpy
import torch
from bert_padding import pad_input, unpad_input
from einops import rearrange, repeat
from onnx import TensorProto, helper
from packaging import version
from parameterized import parameterized
from test_gqa_cpu import smooth_softmax_ref

from onnxruntime import InferenceSession, OrtValue, SessionOptions

torch.manual_seed(0)

pipeline_mode = True  # Reduces number of tests so pipeline doesn't time out


class Formats:
    BSNH = 0
    BNSH = 1


class Config:
    batch_size = 0
    sequence_length = 0
    kv_sequence_length = 0
    past_sequence_length = 0
    num_heads = 0
    kv_num_heads = 0
    head_size = 0
    ep = "CUDAExecutionProvider"

    def __init__(
        self, batch_size, sequence_length, kv_sequence_length, past_sequence_length, num_heads, kv_num_heads, head_size
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.kv_sequence_length = kv_sequence_length
        self.past_sequence_length = past_sequence_length
        self.num_heads = num_heads
        self.kv_num_heads = kv_num_heads
        self.head_size = head_size

    def __repr__(self):
        short_ep = self.ep[: -len("ExecutionProvider")].lower()
        return (
            f"Config(batch_size={self.batch_size}, sequence_length={self.sequence_length}, "
            f"kv_sequence_length={self.kv_sequence_length}, past_sequence_length={self.past_sequence_length}, "
            f"num_heads={self.num_heads}, kv_num_heads={self.kv_num_heads}, head_size={self.head_size}, ep={short_ep})"
        )


class PromptConfig:
    batch_size = 0
    q_sequence_length = 0
    kv_sequence_length = 0
    buffer_sequence_length = 0
    num_heads = 0
    kv_num_heads = 0
    head_size = 0
    ep = "CUDAExecutionProvider"

    def __init__(
        self,
        batch_size,
        q_sequence_length,
        kv_sequence_length,
        buffer_sequence_length,
        num_heads,
        kv_num_heads,
        head_size,
    ):
        self.batch_size = batch_size
        self.q_sequence_length = q_sequence_length
        self.kv_sequence_length = kv_sequence_length
        self.buffer_sequence_length = buffer_sequence_length
        self.num_heads = num_heads
        self.kv_num_heads = kv_num_heads
        self.head_size = head_size

    def __repr__(self):
        short_ep = self.ep[: -len("ExecutionProvider")].lower()
        return (
            f"PromptConfig(batch_size={self.batch_size}, q_sequence_length={self.q_sequence_length}, "
            f"kv_sequence_length={self.kv_sequence_length}, buffer_sequence_length={self.buffer_sequence_length}, "
            f"num_heads={self.num_heads}, kv_num_heads={self.kv_num_heads}, head_size={self.head_size}, ep={short_ep})"
        )


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


def create_group_query_attention_graph_prompt(
    config,
    past_kv_format=Formats.BSNH,
    share_buffer=True,
    local_window_size=-1,
    rotary=False,
    rotary_interleaved=False,
    packed=False,
    interactive=False,
    softcap=0.0,
    use_smooth_softmax=False,
):
    past_kv_seqlen = config.buffer_sequence_length if share_buffer else 0
    present_kv_seqlen = config.buffer_sequence_length if share_buffer else config.kv_sequence_length
    nodes = [
        helper.make_node(
            "GroupQueryAttention",
            [
                "query",
                "key" if not packed else "",
                "value" if not packed else "",
                "past_key" if share_buffer else "",
                "past_value" if share_buffer else "",
                "seqlens_k",
                "total_sequence_length",
                "cos_cache" if rotary else "",
                "sin_cache" if rotary else "",
            ],
            ["output", "present_key", "present_value"],
            "GroupQueryAttention_0",
            num_heads=config.num_heads,
            kv_num_heads=config.kv_num_heads,
            local_window_size=local_window_size,
            do_rotary=rotary,
            rotary_interleaved=rotary_interleaved,
            softcap=softcap,
            smooth_softmax=1 if use_smooth_softmax else 0,
            # is_past_bsnh=1 if past_kv_format == Formats.BSNH else 0,
            # kv_share_buffer=1 if share_buffer else 0,
            domain="com.microsoft",
        ),
    ]

    graph_input = [
        helper.make_tensor_value_info(
            "query",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                config.q_sequence_length,
                (
                    (config.num_heads * config.head_size)
                    if not packed
                    else (config.num_heads * config.head_size + 2 * config.kv_num_heads * config.head_size)
                ),
            ],
        ),
        helper.make_tensor_value_info(
            "seqlens_k",
            TensorProto.INT32,
            [config.batch_size],
        ),
        helper.make_tensor_value_info(
            "total_sequence_length",
            TensorProto.INT32,
            [1],
        ),
    ]
    if not packed:
        graph_input += [
            helper.make_tensor_value_info(
                "key",
                TensorProto.FLOAT16,
                [
                    config.batch_size,
                    config.kv_sequence_length,
                    config.kv_num_heads * config.head_size,
                ],
            ),
            helper.make_tensor_value_info(
                "value",
                TensorProto.FLOAT16,
                [
                    config.batch_size,
                    config.kv_sequence_length,
                    config.kv_num_heads * config.head_size,
                ],
            ),
        ]
    if share_buffer:
        graph_input += [
            helper.make_tensor_value_info(
                "past_key",
                TensorProto.FLOAT16,
                [
                    config.batch_size,
                    past_kv_seqlen if past_kv_format == Formats.BSNH else config.kv_num_heads,
                    config.kv_num_heads if past_kv_format == Formats.BSNH else past_kv_seqlen,
                    config.head_size,
                ],
            ),
            helper.make_tensor_value_info(
                "past_value",
                TensorProto.FLOAT16,
                [
                    config.batch_size,
                    past_kv_seqlen if past_kv_format == Formats.BSNH else config.kv_num_heads,
                    config.kv_num_heads if past_kv_format == Formats.BSNH else past_kv_seqlen,
                    config.head_size,
                ],
            ),
        ]
    if rotary:
        graph_input += [
            helper.make_tensor_value_info(
                "cos_cache",
                TensorProto.FLOAT16,
                [
                    config.buffer_sequence_length if share_buffer else config.kv_sequence_length,
                    (math.floor(config.head_size / 16) * 16) // 2,
                ],
            ),
            helper.make_tensor_value_info(
                "sin_cache",
                TensorProto.FLOAT16,
                [
                    config.buffer_sequence_length if share_buffer else config.kv_sequence_length,
                    (math.floor(config.head_size / 16) * 16) // 2,
                ],
            ),
        ]

    graph_output = [
        helper.make_tensor_value_info(
            "output",
            TensorProto.FLOAT16,
            [config.batch_size, config.q_sequence_length, config.num_heads * config.head_size],
        ),
        helper.make_tensor_value_info(
            "present_key",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                present_kv_seqlen if past_kv_format == Formats.BSNH else config.kv_num_heads,
                config.kv_num_heads if past_kv_format == Formats.BSNH else present_kv_seqlen,
                config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "present_value",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                present_kv_seqlen if past_kv_format == Formats.BSNH else config.kv_num_heads,
                config.kv_num_heads if past_kv_format == Formats.BSNH else present_kv_seqlen,
                config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "present_key",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                config.kv_sequence_length if past_kv_format == Formats.BSNH else config.kv_num_heads,
                config.kv_num_heads if past_kv_format == Formats.BSNH else config.kv_sequence_length,
                config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "present_value",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                config.kv_sequence_length if past_kv_format == Formats.BSNH else config.kv_num_heads,
                config.kv_num_heads if past_kv_format == Formats.BSNH else config.kv_sequence_length,
                config.head_size,
            ],
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "GroupQueryAttention_Graph",
        graph_input,
        graph_output,
    )

    model = helper.make_model(graph)
    return model.SerializeToString()


def create_group_query_attention_graph_past(
    config,
    past_kv_format=Formats.BSNH,
    share_buffer=True,
    local_window_size=-1,
    rotary=False,
    rotary_interleaved=False,
    packed=False,
    softcap=0.0,
    use_smooth_softmax=False,
):
    past_kv_seqlen = config.kv_sequence_length
    present_kv_seqlen = (
        config.kv_sequence_length if share_buffer else config.kv_sequence_length + config.sequence_length
    )
    nodes = [
        helper.make_node(
            "GroupQueryAttention",
            [
                "query",
                "key" if not packed else "",
                "value" if not packed else "",
                "past_key",
                "past_value",
                "seqlens_k",
                "total_sequence_length",
                "cos_cache" if rotary else "",
                "sin_cache" if rotary else "",
            ],
            ["output", "present_key", "present_value"],
            "GroupQueryAttention_0",
            num_heads=config.num_heads,
            kv_num_heads=config.kv_num_heads,
            local_window_size=local_window_size,
            do_rotary=rotary,
            rotary_interleaved=rotary_interleaved,
            softcap=softcap,
            smooth_softmax=1 if use_smooth_softmax else 0,
            # is_past_bsnh=1 if past_kv_format == Formats.BSNH else 0,
            # kv_share_buffer=1 if share_buffer else 0,
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
                (
                    (config.num_heads * config.head_size)
                    if not packed
                    else (config.num_heads * config.head_size + 2 * config.kv_num_heads * config.head_size)
                ),
            ],
        ),
        helper.make_tensor_value_info(
            "past_key",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                past_kv_seqlen if past_kv_format == Formats.BSNH else config.kv_num_heads,
                config.kv_num_heads if past_kv_format == Formats.BSNH else past_kv_seqlen,
                config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "past_value",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                past_kv_seqlen if past_kv_format == Formats.BSNH else config.kv_num_heads,
                config.kv_num_heads if past_kv_format == Formats.BSNH else past_kv_seqlen,
                config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "seqlens_k",
            TensorProto.INT32,
            [config.batch_size],
        ),
        helper.make_tensor_value_info(
            "total_sequence_length",
            TensorProto.INT32,
            [1],
        ),
    ]
    if not packed:
        graph_input += [
            helper.make_tensor_value_info(
                "key",
                TensorProto.FLOAT16,
                [
                    config.batch_size,
                    config.sequence_length,
                    config.kv_num_heads * config.head_size,
                ],
            ),
            helper.make_tensor_value_info(
                "value",
                TensorProto.FLOAT16,
                [
                    config.batch_size,
                    config.sequence_length,
                    config.kv_num_heads * config.head_size,
                ],
            ),
        ]
    if rotary:
        graph_input += [
            helper.make_tensor_value_info(
                "cos_cache",
                TensorProto.FLOAT16,
                [
                    config.kv_sequence_length + (0 if share_buffer else config.sequence_length),
                    (math.floor(config.head_size / 16) * 16) // 2,
                ],
            ),
            helper.make_tensor_value_info(
                "sin_cache",
                TensorProto.FLOAT16,
                [
                    config.kv_sequence_length + (0 if share_buffer else config.sequence_length),
                    (math.floor(config.head_size / 16) * 16) // 2,
                ],
            ),
        ]

    graph_output = [
        helper.make_tensor_value_info(
            "output",
            TensorProto.FLOAT16,
            [config.batch_size, config.sequence_length, config.num_heads * config.head_size],
        ),
        helper.make_tensor_value_info(
            "present_key",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                present_kv_seqlen if past_kv_format == Formats.BSNH else config.kv_num_heads,
                config.kv_num_heads if past_kv_format == Formats.BSNH else present_kv_seqlen,
                config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "present_value",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                present_kv_seqlen if past_kv_format == Formats.BSNH else config.kv_num_heads,
                config.kv_num_heads if past_kv_format == Formats.BSNH else present_kv_seqlen,
                config.head_size,
            ],
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "GroupQueryAttention_Graph",
        graph_input,
        graph_output,
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
    ort_session = InferenceSession(onnx_model_str, sess_options, providers=[config.ep])
    ort_output = ort_session.run(None, ort_inputs)
    output = torch.tensor(ort_output)
    return output


def mha_func(q, k, v, config):
    onnx_model_str = create_multihead_attention_graph(config)
    q = torch.reshape(q, (config.batch_size, config.sequence_length, -1))
    k = torch.reshape(k, (config.batch_size, config.kv_sequence_length, -1))
    v = torch.reshape(v, (config.batch_size, config.kv_sequence_length, -1))
    ort_inputs = {
        "query": q.detach().cpu().numpy(),
        "key": k.detach().cpu().numpy(),
        "value": v.detach().cpu().numpy(),
    }
    sess_options = SessionOptions()
    ort_session = InferenceSession(onnx_model_str, sess_options, providers=[config.ep])
    ort_output = ort_session.run(None, ort_inputs)
    ort_output = numpy.array(ort_output)
    output = torch.tensor(ort_output)
    return output


def rotary_options_for_current_os():
    # Reference implementation of rotary uses triton, which is not available in Windows.
    # So we only test rotary in Linux right now.
    return [(False, False)] if platform.system() != "Linux" else [(True, False), (True, True), (False, False)]


def gqa_prompt_func(
    q,
    k,
    v,
    config,
    new_k,
    new_v,
    cos=None,
    sin=None,
    seqlens_k=None,
    window_size=-1,
    past_kv_format=Formats.BSNH,
    share_buffer=True,
    rotary_interleaved=False,
    softcap=0.0,
    use_smooth_softmax=False,
):
    onnx_model_str = create_group_query_attention_graph_prompt(
        config,
        past_kv_format,
        share_buffer,
        local_window_size=window_size,
        rotary=cos is not None,
        rotary_interleaved=rotary_interleaved,
        packed=new_k is None,
        softcap=softcap,
        use_smooth_softmax=use_smooth_softmax,
    )
    q = torch.reshape(q, (config.batch_size, config.q_sequence_length, -1))
    past_k = k.clone() if share_buffer else None
    past_v = v.clone() if share_buffer else None
    if new_k is not None:
        new_k = torch.reshape(new_k, (config.batch_size, config.kv_sequence_length, -1))
        new_v = torch.reshape(new_v, (config.batch_size, config.kv_sequence_length, -1))
    if share_buffer:
        ort_inputs = {
            "query": q.detach().cpu().numpy(),
            "past_key": OrtValue.ortvalue_from_numpy(past_k.detach().cpu().numpy(), "cuda", 0),
            "past_value": OrtValue.ortvalue_from_numpy(past_v.detach().cpu().numpy(), "cuda", 0),
            "seqlens_k": seqlens_k.detach().cpu().numpy().astype(numpy.int32),
            "total_sequence_length": torch.tensor([config.q_sequence_length], dtype=torch.int32).detach().cpu().numpy(),
        }
        sess_options = SessionOptions()
        ort_session = InferenceSession(onnx_model_str, sess_options, providers=[config.ep])
        io_binding = ort_session.io_binding()
        if new_k is not None:
            ort_inputs["key"] = new_k.detach().cpu().numpy()
            ort_inputs["value"] = new_v.detach().cpu().numpy()
            io_binding.bind_cpu_input("key", ort_inputs["key"])
            io_binding.bind_cpu_input("value", ort_inputs["value"])
        if cos is not None:
            ort_inputs["cos_cache"] = cos.detach().cpu().numpy()
            ort_inputs["sin_cache"] = sin.detach().cpu().numpy()
            io_binding.bind_cpu_input("cos_cache", ort_inputs["cos_cache"])
            io_binding.bind_cpu_input("sin_cache", ort_inputs["sin_cache"])
        io_binding.bind_cpu_input("query", ort_inputs["query"])
        io_binding.bind_input(
            "past_key", "cuda", 0, numpy.float16, ort_inputs["past_key"].shape(), ort_inputs["past_key"].data_ptr()
        )
        io_binding.bind_input(
            "past_value",
            "cuda",
            0,
            numpy.float16,
            ort_inputs["past_value"].shape(),
            ort_inputs["past_value"].data_ptr(),
        )
        io_binding.bind_cpu_input("seqlens_k", ort_inputs["seqlens_k"])
        io_binding.bind_cpu_input("total_sequence_length", ort_inputs["total_sequence_length"])
        io_binding.bind_output("output")
        io_binding.bind_ortvalue_output("present_key", ort_inputs["past_key"])
        io_binding.bind_ortvalue_output("present_value", ort_inputs["past_value"])
        ort_session.run_with_iobinding(io_binding)
        ort_output, present_k, present_v = io_binding.copy_outputs_to_cpu()
        ort_output = numpy.array(ort_output)
        output = torch.tensor(ort_output)
        return output, present_k, present_v
    else:
        ort_inputs = {
            "query": q.detach().cpu().numpy(),
            "seqlens_k": seqlens_k.detach().cpu().numpy().astype(numpy.int32),
            "total_sequence_length": torch.tensor([config.q_sequence_length], dtype=torch.int32).detach().cpu().numpy(),
        }
        sess_options = SessionOptions()
        ort_session = InferenceSession(onnx_model_str, sess_options, providers=[config.ep])
        io_binding = ort_session.io_binding()
        if new_k is not None:
            ort_inputs["key"] = new_k.detach().cpu().numpy()
            ort_inputs["value"] = new_v.detach().cpu().numpy()
            io_binding.bind_cpu_input("key", ort_inputs["key"])
            io_binding.bind_cpu_input("value", ort_inputs["value"])
        if cos is not None:
            ort_inputs["cos_cache"] = cos.detach().cpu().numpy()
            ort_inputs["sin_cache"] = sin.detach().cpu().numpy()
            io_binding.bind_cpu_input("cos_cache", ort_inputs["cos_cache"])
            io_binding.bind_cpu_input("sin_cache", ort_inputs["sin_cache"])
        io_binding.bind_cpu_input("query", ort_inputs["query"])
        io_binding.bind_cpu_input("seqlens_k", ort_inputs["seqlens_k"])
        io_binding.bind_cpu_input("total_sequence_length", ort_inputs["total_sequence_length"])
        io_binding.bind_output("output")
        io_binding.bind_output("present_key")
        io_binding.bind_output("present_value")
        ort_session.run_with_iobinding(io_binding)
        ort_output, present_k, present_v = io_binding.copy_outputs_to_cpu()
        ort_output = numpy.array(ort_output)
        output = torch.tensor(ort_output)
        return output, present_k, present_v


def gqa_past_func(
    q,
    k,
    v,
    config,
    new_k,
    new_v,
    cos=None,
    sin=None,
    seqlens_k=None,
    past_kv_format=Formats.BSNH,
    share_buffer=True,
    window_size=-1,
    rotary_interleaved=False,
    softcap=0.0,
    use_smooth_softmax=False,
):
    onnx_model_str = create_group_query_attention_graph_past(
        config,
        past_kv_format,
        share_buffer,
        local_window_size=window_size,
        rotary=cos is not None,
        rotary_interleaved=rotary_interleaved,
        packed=new_k is None,
        softcap=softcap,
        use_smooth_softmax=use_smooth_softmax,
    )
    q = torch.reshape(q, (config.batch_size, config.sequence_length, -1))
    past_k = k.clone()
    past_v = v.clone()
    if new_k is not None:
        new_k = torch.reshape(new_k, (config.batch_size, config.sequence_length, -1))
        new_v = torch.reshape(new_v, (config.batch_size, config.sequence_length, -1))
    if share_buffer:
        ort_inputs = {
            "query": q.detach().cpu().numpy(),
            "past_key": OrtValue.ortvalue_from_numpy(past_k.detach().cpu().numpy(), "cuda", 0),
            "past_value": OrtValue.ortvalue_from_numpy(past_v.detach().cpu().numpy(), "cuda", 0),
            "seqlens_k": seqlens_k.detach().cpu().numpy().astype(numpy.int32),
            "total_sequence_length": torch.tensor([config.kv_sequence_length], dtype=torch.int32)
            .detach()
            .cpu()
            .numpy(),
        }
        sess_options = SessionOptions()
        ort_session = InferenceSession(onnx_model_str, sess_options, providers=[config.ep])
        io_binding = ort_session.io_binding()
        if new_k is not None:
            ort_inputs["key"] = new_k.detach().cpu().numpy()
            ort_inputs["value"] = new_v.detach().cpu().numpy()
            io_binding.bind_cpu_input("key", ort_inputs["key"])
            io_binding.bind_cpu_input("value", ort_inputs["value"])
        if cos is not None:
            ort_inputs["cos_cache"] = cos.detach().cpu().numpy()
            ort_inputs["sin_cache"] = sin.detach().cpu().numpy()
            io_binding.bind_cpu_input("cos_cache", ort_inputs["cos_cache"])
            io_binding.bind_cpu_input("sin_cache", ort_inputs["sin_cache"])
        io_binding.bind_cpu_input("query", ort_inputs["query"])
        io_binding.bind_input(
            "past_key", "cuda", 0, numpy.float16, ort_inputs["past_key"].shape(), ort_inputs["past_key"].data_ptr()
        )
        io_binding.bind_input(
            "past_value",
            "cuda",
            0,
            numpy.float16,
            ort_inputs["past_value"].shape(),
            ort_inputs["past_value"].data_ptr(),
        )
        io_binding.bind_cpu_input("seqlens_k", ort_inputs["seqlens_k"])
        io_binding.bind_cpu_input("total_sequence_length", ort_inputs["total_sequence_length"])
        io_binding.bind_output("output")
        io_binding.bind_ortvalue_output("present_key", ort_inputs["past_key"])
        io_binding.bind_ortvalue_output("present_value", ort_inputs["past_value"])
        ort_session.run_with_iobinding(io_binding)
        ort_output, present_k, present_v = io_binding.copy_outputs_to_cpu()
        ort_output = numpy.array(ort_output)
        output = torch.tensor(ort_output)
        return output, present_k, present_v
    else:
        ort_inputs = {
            "query": q.detach().cpu().numpy(),
            "past_key": past_k.detach().cpu().numpy(),
            "past_value": past_v.detach().cpu().numpy(),
            "seqlens_k": seqlens_k.detach().cpu().numpy().astype(numpy.int32),
            "total_sequence_length": torch.tensor(
                [config.kv_sequence_length + config.sequence_length], dtype=torch.int32
            )
            .detach()
            .cpu()
            .numpy(),
        }
        sess_options = SessionOptions()
        ort_session = InferenceSession(onnx_model_str, sess_options, providers=[config.ep])
        io_binding = ort_session.io_binding()
        if new_k is not None:
            ort_inputs["key"] = new_k.detach().cpu().numpy()
            ort_inputs["value"] = new_v.detach().cpu().numpy()
            io_binding.bind_cpu_input("key", ort_inputs["key"])
            io_binding.bind_cpu_input("value", ort_inputs["value"])
        if cos is not None:
            ort_inputs["cos_cache"] = cos.detach().cpu().numpy()
            ort_inputs["sin_cache"] = sin.detach().cpu().numpy()
            io_binding.bind_cpu_input("cos_cache", ort_inputs["cos_cache"])
            io_binding.bind_cpu_input("sin_cache", ort_inputs["sin_cache"])
        io_binding.bind_cpu_input("query", ort_inputs["query"])
        io_binding.bind_cpu_input("past_key", ort_inputs["past_key"])
        io_binding.bind_cpu_input("past_value", ort_inputs["past_value"])
        io_binding.bind_cpu_input("seqlens_k", ort_inputs["seqlens_k"])
        io_binding.bind_cpu_input("total_sequence_length", ort_inputs["total_sequence_length"])
        io_binding.bind_output("output")
        io_binding.bind_output("present_key")
        io_binding.bind_output("present_value")
        ort_session.run_with_iobinding(io_binding)
        ort_output, present_k, present_v = io_binding.copy_outputs_to_cpu()
        ort_output = numpy.array(ort_output)
        output = torch.tensor(ort_output)
        return output, present_k, present_v


def construct_causal_mask(seqlen_q, seqlen_k, query_padding_mask=None, key_padding_mask=None, device=None):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    sk = seqlen_k if key_padding_mask is None else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    sq = seqlen_q if query_padding_mask is None else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    return col_idx > row_idx + sk - sq


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
        attention = smooth_softmax_ref(scores)
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


def attention_qkvpacked_ref(
    qkv,
    key_padding_mask=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    upcast=True,
    reorder_ops=False,
    use_smooth_softmax=False,
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
        use_smooth_softmax=use_smooth_softmax,
    )


def parity_check_mha(
    config,
    packed,
    rtol=1e-3,
    atol=1e-3,
):
    if packed:
        qkv_unpad, cu_seqlens, _, qkv, output_pad_fn, _, key_padding_mask = create_inputs(config)
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
        out = mha_func(q, k, v, config)
        out = torch.squeeze(out, 0)
        out = torch.reshape(out, (config.batch_size, config.sequence_length, config.num_heads, config.head_size))
        out = out.detach().cpu().numpy()
        # Pytorch to compare
        out_ref, _ = attention_ref(q, k, v, None, None, 0.0, None, causal=False)
        out_ref = out_ref.detach().cpu().numpy()

    numpy.testing.assert_allclose(
        out, out_ref, rtol=rtol, atol=atol, equal_nan=True, err_msg=f" with {config} packed={packed}"
    )


def rotary_embedding(*args, **kwargs):
    # Use local import since triton is not available in Windows.
    from rotary_flash import apply_rotary_emb

    return apply_rotary_emb(*args, **kwargs)


def parity_check_gqa_prompt(
    config,
    causal=True,
    local=False,
    past_format=Formats.BNSH,
    rotary=False,
    rotary_interleaved=False,
    packed=False,
    softcap=0.0,
    use_smooth_softmax=False,
    rtol=1e-3,
    atol=1e-3,
):
    q = torch.randn(
        config.batch_size,
        config.q_sequence_length,
        config.num_heads,
        config.head_size,
        device="cuda",
        dtype=torch.float16,
        requires_grad=False,
    )
    k = torch.randn(
        config.batch_size,
        config.buffer_sequence_length if past_format == Formats.BSNH else config.kv_num_heads,
        config.kv_num_heads if past_format == Formats.BSNH else config.buffer_sequence_length,
        config.head_size,
        device="cuda",
        dtype=torch.float16,
        requires_grad=False,
    )
    v = torch.randn(
        config.batch_size,
        config.buffer_sequence_length if past_format == Formats.BSNH else config.kv_num_heads,
        config.kv_num_heads if past_format == Formats.BSNH else config.buffer_sequence_length,
        config.head_size,
        device="cuda",
        dtype=torch.float16,
        requires_grad=False,
    )
    new_k = torch.randn(
        config.batch_size,
        config.kv_sequence_length,
        config.kv_num_heads,
        config.head_size,
        device="cuda",
        dtype=torch.float16,
        requires_grad=False,
    )
    new_v = torch.randn(
        config.batch_size,
        config.kv_sequence_length,
        config.kv_num_heads,
        config.head_size,
        device="cuda",
        dtype=torch.float16,
        requires_grad=False,
    )

    window_size = (-1, -1)
    left_window_size = -1
    if local:
        left_window_size = random.randint(0, config.kv_sequence_length)
        window_size = (left_window_size, 0)
    elif causal:
        left_window_size = -1
        window_size = (-1, 0)

    # Pytorch to compare
    k_cache_ref = k.clone()
    v_cache_ref = v.clone()
    if past_format == Formats.BNSH:
        k_cache_ref = k_cache_ref.transpose(1, 2)
        v_cache_ref = v_cache_ref.transpose(1, 2)
    cache_seqlens = torch.tensor([config.kv_sequence_length], device="cuda").repeat(config.batch_size)
    # cache_seqlens = torch.randint(
    #     0,
    #     config.kv_sequence_length,
    #     (config.batch_size,),
    #     dtype=torch.int32,
    #     device="cuda",
    # )
    # cache_seqlens[random.randint(0, cache_seqlens.size(dim=0) - 1)] = config.kv_sequence_length
    rotary_seqlens = torch.tensor([0], device="cuda").repeat(config.batch_size)

    if rotary:
        rotary_fraction = 1.0
        rotary_dim = math.floor(int(rotary_fraction * config.head_size) / 16) * 16
        angle = torch.rand(config.buffer_sequence_length, rotary_dim // 2, device="cuda") * 2 * math.pi
        cos = torch.cos(angle).to(dtype=torch.float16)
        sin = torch.sin(angle).to(dtype=torch.float16)

        if causal or local:
            q_ro = rotary_embedding(q, cos, sin, seqlen_offsets=rotary_seqlens, interleaved=rotary_interleaved)
        else:
            q_ro = rearrange(
                rotary_embedding(
                    rearrange(q, "b s h d -> b 1 (s h) d"),
                    cos,
                    sin,
                    seqlen_offsets=rotary_seqlens,
                    interleaved=rotary_interleaved,
                ),
                "b 1 (s h) d -> b s h d",
                s=config.q_sequence_length,
            )
        # q_ro = q
        k_ro = rotary_embedding(new_k, cos, sin, seqlen_offsets=rotary_seqlens, interleaved=rotary_interleaved)
    else:
        cos, sin = None, None
        q_ro, k_ro = q, new_k

    rearrange(torch.arange(config.kv_sequence_length, device="cuda"), "s -> 1 s")
    arange = rearrange(torch.arange(config.buffer_sequence_length, device="cuda"), "s -> 1 s")
    cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
    kv_seqlens = torch.tensor([config.kv_sequence_length], device="cuda").repeat(config.batch_size)
    kv_seqlens_expanded = rearrange(kv_seqlens, "b -> b 1")
    update_mask = arange < kv_seqlens_expanded
    k_cache_ref[update_mask] = rearrange(k_ro, "b s ... -> (b s) ...")
    v_cache_ref[update_mask] = rearrange(new_v, "b s ... -> (b s) ...")
    k_cache_rep = repeat(k_cache_ref, "b s h d -> b s (h g) d", g=config.num_heads // config.kv_num_heads)
    v_cache_rep = repeat(v_cache_ref, "b s h d -> b s (h g) d", g=config.num_heads // config.kv_num_heads)
    key_padding_mask = arange < cache_seqlens_expanded
    out_ref, _ = attention_ref(
        q_ro,
        k_cache_rep,
        v_cache_rep,
        None,
        key_padding_mask,
        0.0,
        None,
        causal=True,
        window_size=window_size,
        softcap=softcap,
        use_smooth_softmax=use_smooth_softmax,
    )
    out_ref = out_ref.detach().cpu().numpy()
    if past_format == Formats.BNSH:
        k_cache_ref = k_cache_ref.transpose(1, 2)
        v_cache_ref = v_cache_ref.transpose(1, 2)

    # Flash function
    if packed:
        packed_qkv = torch.concatenate([q, new_k, new_v], dim=2)
        out, present_k, present_v = gqa_prompt_func(
            packed_qkv,
            k,
            v,
            config,
            None,
            None,
            cos,
            sin,
            cache_seqlens,
            left_window_size,
            past_format,
            True,
            rotary_interleaved,
            softcap,
            use_smooth_softmax=use_smooth_softmax,
        )
    else:
        out, present_k, present_v = gqa_prompt_func(
            q,
            k,
            v,
            config,
            new_k,
            new_v,
            cos,
            sin,
            cache_seqlens,
            left_window_size,
            past_format,
            True,
            rotary_interleaved,
            softcap,
            use_smooth_softmax=use_smooth_softmax,
        )
    out = torch.squeeze(out, 0)
    out = torch.reshape(out, (config.batch_size, config.q_sequence_length, config.num_heads, config.head_size))
    out = out.detach().cpu().numpy()

    err_msg = (
        f" with {config}, causal={causal}, local={local}, past_format={past_format},"
        f" rotary={rotary}, rotary_interleaved={rotary_interleaved}, packed={packed}, softcap={softcap}"
    )
    # Make sure past-present buffer updating correctly
    numpy.testing.assert_allclose(
        present_k, k_cache_ref.detach().cpu().numpy(), rtol=rtol, atol=atol, equal_nan=True, err_msg=err_msg
    )
    numpy.testing.assert_allclose(
        present_v, v_cache_ref.detach().cpu().numpy(), rtol=rtol, atol=atol, equal_nan=True, err_msg=err_msg
    )

    numpy.testing.assert_allclose(out, out_ref, rtol=rtol, atol=atol, equal_nan=True, err_msg=err_msg)


def parity_check_gqa_prompt_no_buff(
    config,
    causal=True,
    local=False,
    past_format=Formats.BNSH,
    rotary=False,
    rotary_interleaved=False,
    packed=False,
    softcap=0.0,
    use_smooth_softmax=False,
    rtol=1e-3,
    atol=1e-3,
):
    q = torch.randn(
        config.batch_size,
        config.q_sequence_length,
        config.num_heads,
        config.head_size,
        device="cuda",
        dtype=torch.float16,
        requires_grad=False,
    )
    new_k = torch.randn(
        config.batch_size,
        config.kv_sequence_length,
        config.kv_num_heads,
        config.head_size,
        device="cuda",
        dtype=torch.float16,
        requires_grad=False,
    )
    new_v = torch.randn(
        config.batch_size,
        config.kv_sequence_length,
        config.kv_num_heads,
        config.head_size,
        device="cuda",
        dtype=torch.float16,
        requires_grad=False,
    )

    window_size = (-1, -1)
    left_window_size = -1
    if local:
        left_window_size = random.randint(0, config.kv_sequence_length)
        window_size = (left_window_size, 0)
    elif causal:
        left_window_size = -1
        window_size = (-1, 0)

    # Pytorch to compare
    k_cache_ref = new_k.clone()
    v_cache_ref = new_v.clone()
    # if past_format == Formats.BNSH:
    #     k_cache_ref = k_cache_ref.transpose(1, 2)
    #     v_cache_ref = v_cache_ref.transpose(1, 2)
    cache_seqlens = torch.tensor([config.kv_sequence_length], device="cuda").repeat(config.batch_size)
    # cache_seqlens = torch.randint(
    #     0,
    #     config.kv_sequence_length,
    #     (config.batch_size,),
    #     dtype=torch.int32,
    #     device="cuda",
    # )
    # cache_seqlens[random.randint(0, cache_seqlens.size(dim=0) - 1)] = config.kv_sequence_length
    rotary_seqlens = torch.tensor([0], device="cuda").repeat(config.batch_size)

    if rotary:
        rotary_fraction = 1.0
        rotary_dim = math.floor(int(rotary_fraction * config.head_size) / 16) * 16
        angle = torch.rand(config.kv_sequence_length, rotary_dim // 2, device="cuda") * 2 * math.pi
        cos = torch.cos(angle).to(dtype=torch.float16)
        sin = torch.sin(angle).to(dtype=torch.float16)

        if causal or local:
            q_ro = rotary_embedding(q, cos, sin, seqlen_offsets=rotary_seqlens, interleaved=rotary_interleaved)
        else:
            q_ro = rearrange(
                rotary_embedding(
                    rearrange(q, "b s h d -> b 1 (s h) d"),
                    cos,
                    sin,
                    seqlen_offsets=rotary_seqlens,
                    interleaved=rotary_interleaved,
                ),
                "b 1 (s h) d -> b s h d",
                s=config.q_sequence_length,
            )
        # q_ro = q
        k_ro = rotary_embedding(k_cache_ref, cos, sin, seqlen_offsets=rotary_seqlens, interleaved=rotary_interleaved)
    else:
        cos, sin = None, None
        q_ro, k_ro = q, k_cache_ref
    k_cache_ref = k_ro

    brange = rearrange(torch.arange(config.kv_sequence_length, device="cuda"), "s -> 1 s")
    cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
    new_mask = brange < cache_seqlens_expanded
    k_cache_rep = repeat(k_cache_ref, "b s h d -> b s (h g) d", g=config.num_heads // config.kv_num_heads)
    v_cache_rep = repeat(v_cache_ref, "b s h d -> b s (h g) d", g=config.num_heads // config.kv_num_heads)
    out_ref, _ = attention_ref(
        q_ro,
        k_cache_rep,
        v_cache_rep,
        None,
        new_mask,
        0.0,
        None,
        causal=True,
        window_size=window_size,
        softcap=softcap,
        use_smooth_softmax=use_smooth_softmax,
    )
    out_ref = out_ref.detach().cpu().numpy()
    if past_format == Formats.BNSH:
        k_cache_ref = k_cache_ref.transpose(1, 2)
        v_cache_ref = v_cache_ref.transpose(1, 2)

    # Flash function
    if packed:
        packed_qkv = torch.concatenate([q, new_k, new_v], dim=2)
        out, present_k, present_v = gqa_prompt_func(
            packed_qkv,
            None,
            None,
            config,
            None,
            None,
            cos,
            sin,
            cache_seqlens,
            left_window_size,
            past_format,
            False,
            rotary_interleaved,
            softcap,
            use_smooth_softmax=use_smooth_softmax,
        )
    else:
        out, present_k, present_v = gqa_prompt_func(
            q,
            None,
            None,
            config,
            new_k,
            new_v,
            cos,
            sin,
            cache_seqlens,
            left_window_size,
            past_format,
            False,
            rotary_interleaved,
            softcap,
            use_smooth_softmax=use_smooth_softmax,
        )
    out = torch.squeeze(out, 0)
    out = torch.reshape(out, (config.batch_size, config.q_sequence_length, config.num_heads, config.head_size))
    out = out.detach().cpu().numpy()

    err_msg = (
        f" with {config}, causal={causal}, local={local}, past_format={past_format},"
        f" rotary={rotary}, rotary_interleaved={rotary_interleaved}, packed={packed}, softcap={softcap}, use_smooth_softmax={use_smooth_softmax}"
    )
    # Make sure past-present buffer updating correctly
    numpy.testing.assert_allclose(
        present_k, k_cache_ref.detach().cpu().numpy(), rtol=rtol, atol=atol, equal_nan=True, err_msg=err_msg
    )
    numpy.testing.assert_allclose(
        present_v, v_cache_ref.detach().cpu().numpy(), rtol=rtol, atol=atol, equal_nan=True, err_msg=err_msg
    )

    numpy.testing.assert_allclose(out, out_ref, rtol=rtol, atol=atol, equal_nan=True, err_msg=err_msg)


def parity_check_gqa_past(
    config,
    causal=True,
    local=False,
    past_format=Formats.BNSH,
    rotary=False,
    rotary_interleaved=False,
    packed=False,
    softcap=0.0,
    use_smooth_softmax=False,
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
        config.kv_sequence_length if past_format == Formats.BSNH else config.kv_num_heads,
        config.kv_num_heads if past_format == Formats.BSNH else config.kv_sequence_length,
        config.head_size,
        device="cuda",
        dtype=torch.float16,
        requires_grad=False,
    )
    v = torch.randn(
        config.batch_size,
        config.kv_sequence_length if past_format == Formats.BSNH else config.kv_num_heads,
        config.kv_num_heads if past_format == Formats.BSNH else config.kv_sequence_length,
        config.head_size,
        device="cuda",
        dtype=torch.float16,
        requires_grad=False,
    )
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

    window_size = (-1, -1)
    left_window_size = -1
    if local:
        left_window_size = random.randint(0, config.kv_sequence_length)
        window_size = (left_window_size, 0)
    elif causal:
        left_window_size = -1
        window_size = (-1, 0)

    # Pytorch to compare
    k_cache_ref = k.clone()
    v_cache_ref = v.clone()
    if past_format == Formats.BNSH:
        k_cache_ref = k_cache_ref.transpose(1, 2)
        v_cache_ref = v_cache_ref.transpose(1, 2)
    cache_seqlens = torch.randint(
        0,
        config.kv_sequence_length - config.sequence_length + 1,
        (config.batch_size,),
        dtype=torch.int32,
        device="cuda",
    )

    if rotary:
        rotary_fraction = 1.0
        rotary_dim = math.floor(int(rotary_fraction * config.head_size) / 16) * 16
        angle = torch.rand(config.kv_sequence_length, rotary_dim // 2, device="cuda") * 2 * math.pi
        cos = torch.cos(angle).to(dtype=torch.float16)
        sin = torch.sin(angle).to(dtype=torch.float16)
        if causal or local:
            q_ro = rotary_embedding(q, cos, sin, seqlen_offsets=cache_seqlens, interleaved=rotary_interleaved)
        else:
            q_ro = rearrange(
                rotary_embedding(
                    rearrange(q, "b s h d -> b 1 (s h) d"),
                    cos,
                    sin,
                    seqlen_offsets=cache_seqlens,
                    interleaved=rotary_interleaved,
                ),
                "b 1 (s h) d -> b s h d",
                s=config.sequence_length,
            )
        k_ro = rotary_embedding(new_k, cos, sin, seqlen_offsets=cache_seqlens, interleaved=rotary_interleaved)
    else:
        cos, sin = None, None
        q_ro, k_ro = q, new_k

    arange = rearrange(torch.arange(config.kv_sequence_length, device="cuda"), "s -> 1 s")
    cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
    update_mask = torch.logical_and(
        cache_seqlens_expanded <= arange, arange < cache_seqlens_expanded + config.sequence_length
    )
    k_cache_ref[update_mask] = rearrange(k_ro, "b s ... -> (b s) ...")
    v_cache_ref[update_mask] = rearrange(new_v, "b s ... -> (b s) ...")
    k_cache_rep = repeat(k_cache_ref, "b s h d -> b s (h g) d", g=config.num_heads // config.kv_num_heads)
    v_cache_rep = repeat(v_cache_ref, "b s h d -> b s (h g) d", g=config.num_heads // config.kv_num_heads)
    key_padding_mask = arange < cache_seqlens_expanded + config.sequence_length
    out_ref, _ = attention_ref(
        q_ro,
        k_cache_rep,
        v_cache_rep,
        None,
        key_padding_mask,
        0.0,
        None,
        causal=True,
        window_size=window_size,
        softcap=softcap,
        use_smooth_softmax=use_smooth_softmax,
    )
    out_ref = out_ref.detach().cpu().numpy()
    if past_format == Formats.BNSH:
        k_cache_ref = k_cache_ref.transpose(1, 2)
        v_cache_ref = v_cache_ref.transpose(1, 2)

    cache_seqlens += config.sequence_length - 1

    # Flash function
    if packed:
        packed_qkv = torch.concatenate([q, new_k, new_v], dim=2)
        out, present_k, present_v = gqa_past_func(
            packed_qkv,
            k,
            v,
            config,
            None,
            None,
            cos,
            sin,
            cache_seqlens,
            past_format,
            True,
            left_window_size,
            rotary_interleaved,
            softcap,
            use_smooth_softmax=use_smooth_softmax,
        )
    else:
        out, present_k, present_v = gqa_past_func(
            q,
            k,
            v,
            config,
            new_k,
            new_v,
            cos,
            sin,
            cache_seqlens,
            past_format,
            True,
            left_window_size,
            rotary_interleaved,
            softcap,
            use_smooth_softmax=use_smooth_softmax,
        )
    out = torch.squeeze(out, 0)
    out = torch.reshape(out, (config.batch_size, config.sequence_length, config.num_heads, config.head_size))
    out = out.detach().cpu().numpy()

    err_msg = (
        f" with {config}, causal={causal}, local={local}, past_format={past_format},"
        f" rotary={rotary}, rotary_interleaved={rotary_interleaved}, packed={packed}, softcap={softcap}"
    )
    # Make sure past-present buffer updating correctly
    numpy.testing.assert_allclose(
        present_k, k_cache_ref.detach().cpu().numpy(), rtol=rtol, atol=atol, equal_nan=True, err_msg=err_msg
    )
    numpy.testing.assert_allclose(
        present_v, v_cache_ref.detach().cpu().numpy(), rtol=rtol, atol=atol, equal_nan=True, err_msg=err_msg
    )
    numpy.testing.assert_allclose(out, out_ref, rtol=rtol, atol=atol, equal_nan=True, err_msg=err_msg)


def parity_check_gqa_past_no_buff(
    config,
    causal=True,
    local=False,
    past_format=Formats.BNSH,
    rotary=False,
    rotary_interleaved=False,
    packed=False,
    softcap=0.0,
    use_smooth_softmax=False,
    rtol=1e-3,
    atol=1e-3,
):
    torch.manual_seed(69)
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
        config.kv_sequence_length if past_format == Formats.BSNH else config.kv_num_heads,
        config.kv_num_heads if past_format == Formats.BSNH else config.kv_sequence_length,
        config.head_size,
        device="cuda",
        dtype=torch.float16,
        requires_grad=False,
    )
    v = torch.randn(
        config.batch_size,
        config.kv_sequence_length if past_format == Formats.BSNH else config.kv_num_heads,
        config.kv_num_heads if past_format == Formats.BSNH else config.kv_sequence_length,
        config.head_size,
        device="cuda",
        dtype=torch.float16,
        requires_grad=False,
    )
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

    window_size = (-1, -1)
    left_window_size = -1
    if local:
        left_window_size = random.randint(0, config.kv_sequence_length)
        window_size = (left_window_size, 0)
    elif causal:
        left_window_size = -1
        window_size = (-1, 0)

    # Pytorch to compare
    k_cache_ref = k.clone()
    v_cache_ref = v.clone()
    if past_format == Formats.BNSH:
        k_cache_ref = k_cache_ref.transpose(1, 2)
        v_cache_ref = v_cache_ref.transpose(1, 2)
    k_cache_ref = torch.cat((k_cache_ref, new_k), 1)
    v_cache_ref = torch.cat((v_cache_ref, new_v), 1)
    cache_seqlens = torch.randint(
        0,
        config.kv_sequence_length,
        (config.batch_size,),
        dtype=torch.int32,
        device="cuda",
    )
    cache_seqlens[random.randint(0, config.batch_size - 1)] = config.kv_sequence_length

    if rotary:
        rotary_fraction = 1.0
        rotary_dim = math.floor(int(rotary_fraction * config.head_size) / 16) * 16
        angle = (
            torch.rand(config.kv_sequence_length + config.sequence_length, rotary_dim // 2, device="cuda") * 2 * math.pi
        )
        cos = torch.cos(angle).to(dtype=torch.float16)
        sin = torch.sin(angle).to(dtype=torch.float16)
        if causal or local:
            q_ro = rotary_embedding(q, cos, sin, seqlen_offsets=cache_seqlens, interleaved=rotary_interleaved)
        else:
            q_ro = rearrange(
                rotary_embedding(
                    rearrange(q, "b s h d -> b 1 (s h) d"),
                    cos,
                    sin,
                    seqlen_offsets=cache_seqlens,
                    interleaved=rotary_interleaved,
                ),
                "b 1 (s h) d -> b s h d",
                s=config.sequence_length,
            )
        k_ro = rotary_embedding(new_k, cos, sin, seqlen_offsets=cache_seqlens, interleaved=rotary_interleaved)
    else:
        cos, sin = None, None
        q_ro, k_ro = q, new_k

    arange = rearrange(torch.arange(config.kv_sequence_length + config.sequence_length, device="cuda"), "s -> 1 s")
    cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
    update_mask = torch.logical_and(
        cache_seqlens_expanded <= arange, arange < cache_seqlens_expanded + config.sequence_length
    )
    k_cache_ref[update_mask] = rearrange(k_ro, "b s ... -> (b s) ...")
    v_cache_ref[update_mask] = rearrange(new_v, "b s ... -> (b s) ...")
    k_cache_rep = repeat(k_cache_ref, "b s h d -> b s (h g) d", g=config.num_heads // config.kv_num_heads)
    v_cache_rep = repeat(v_cache_ref, "b s h d -> b s (h g) d", g=config.num_heads // config.kv_num_heads)
    key_padding_mask = arange < cache_seqlens_expanded + config.sequence_length
    out_ref, _ = attention_ref(
        q_ro,
        k_cache_rep,
        v_cache_rep,
        None,
        key_padding_mask,
        0.0,
        None,
        causal=True,
        window_size=window_size,
        softcap=softcap,
        use_smooth_softmax=use_smooth_softmax,
    )
    out_ref = out_ref.detach().cpu().numpy()
    if past_format == Formats.BNSH:
        k_cache_ref = k_cache_ref.transpose(1, 2)
        v_cache_ref = v_cache_ref.transpose(1, 2)

    cache_seqlens += config.sequence_length - 1

    # Flash function
    if packed:
        packed_qkv = torch.concatenate([q, new_k, new_v], dim=2)
        out, present_k, present_v = gqa_past_func(
            packed_qkv,
            k,
            v,
            config,
            None,
            None,
            cos,
            sin,
            cache_seqlens,
            past_format,
            False,
            window_size=left_window_size,
            rotary_interleaved=rotary_interleaved,
            softcap=softcap,
            use_smooth_softmax=use_smooth_softmax,
        )
    else:
        out, present_k, present_v = gqa_past_func(
            q,
            k,
            v,
            config,
            new_k,
            new_v,
            cos,
            sin,
            cache_seqlens,
            past_format,
            False,
            window_size=left_window_size,
            rotary_interleaved=rotary_interleaved,
            softcap=softcap,
            use_smooth_softmax=use_smooth_softmax,
        )
    out = torch.squeeze(out, 0)
    out = torch.reshape(out, (config.batch_size, config.sequence_length, config.num_heads, config.head_size))
    out = out.detach().cpu().numpy()

    err_msg = (
        f" with {config}, causal={causal}, local={local}, past_format={past_format},"
        f" rotary={rotary}, rotary_interleaved={rotary_interleaved}, packed={packed}, softcap={softcap}"
    )
    for b in range(config.batch_size):
        numpy.testing.assert_allclose(
            present_k[b, :, : (cache_seqlens + 1)[b]],
            k_cache_ref[b, :, : (cache_seqlens + 1)[b]].detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
            err_msg=err_msg,
        )
        numpy.testing.assert_allclose(
            present_v[b, :, : (cache_seqlens + 1)[b]],
            v_cache_ref[b, :, : (cache_seqlens + 1)[b]].detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
            err_msg=err_msg,
        )
    numpy.testing.assert_allclose(out, out_ref, rtol=rtol, atol=atol, equal_nan=True, err_msg=err_msg)


def has_flash_attention():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8 and (
        platform.system() == "Linux"
        or (platform.system() == "Windows" and version.parse(torch.version.cuda) >= version.parse("12.0"))
    )


def has_memory_efficient():
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    if major < 5 or (major == 5 and minor < 3):
        return False
    return True


def packed_mha_test_cases():
    batches = [2] if pipeline_mode else [1, 5]
    seqs = [1024, 1025] if pipeline_mode else [1024, 1025, 2048]
    num_h = [1, 3] if pipeline_mode else [1, 6, 16]
    h_sizes = [16, 256] if pipeline_mode else [32, 40, 64, 80, 96, 128, 160, 192, 224, 256]

    for b in batches:
        for s in seqs:
            for n in num_h:
                for h in h_sizes:
                    config = Config(b, s, s, 0, n, n, h)
                    yield str(config), config


def mha_test_cases():
    batches = [2] if pipeline_mode else [1, 5]
    seqs = (
        [(1, 128), (113, 211), (2048, 2048)]
        if pipeline_mode
        else [
            (113, 203),
            (128, 217),
            (113, 211),
            (108, 256),
            (256, 512),
            (512, 256),
            (1024, 1024),
            (1023, 1024),
            (1024, 1023),
            (2048, 2048),
        ]
    )
    num_h = [1, 3] if pipeline_mode else [1, 6, 16]
    h_sizes = [16, 256] if pipeline_mode else [32, 40, 64, 80, 96, 128, 160, 192, 224, 256]

    for b in batches:
        for s, s2 in seqs:
            for n in num_h:
                for h in h_sizes:
                    config = Config(b, s, s2, 0, n, n, h)
                    yield str(config), config


class TestMHA(unittest.TestCase):
    @parameterized.expand(packed_mha_test_cases())
    def test_packed_mha(self, _, config):
        if not has_flash_attention():
            return
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "0"
        print("-------- TEST PACKED MHA ---------")
        parity_check_mha(config, True)

    @parameterized.expand(mha_test_cases())
    def test_mha(self, _, config):
        if not has_flash_attention():
            return
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "0"
        print("-------- TEST MHA ---------")
        parity_check_mha(config, False)


def gqa_no_past_memory_efficient_test_cases():
    batches = [3] if pipeline_mode else [1, 3, 5]
    seqs = (
        [
            (127, 127),
            (35, 35),
            (2000, 2000),
            (200, 200),
            (240, 240),
        ]
        if pipeline_mode
        else [
            (127, 127),
            (35, 35),
            (2000, 2000),
            (200, 200),
            (240, 240),
        ]
    )
    num_h = [(32, 8), (9, 3), (4, 4)] if pipeline_mode else [(6, 6), (6, 3), (9, 9), (9, 3)]
    h_sizes = [16, 128, 256] if pipeline_mode else [32, 40, 64, 80, 96, 128, 160, 192, 224, 256]
    torch.manual_seed(69)

    for b in batches:
        for sq, skv in seqs:
            for n, n2 in num_h:
                for h in h_sizes:
                    for rotary, rotary_interleaved in rotary_options_for_current_os():
                        for packed in [False, True]:
                            for softcap in [0.0, 50.0]:
                                config = PromptConfig(b, sq, skv, sq + skv + 8, n, n2, h)
                                yield (
                                    str(config) + f"{rotary}_{rotary_interleaved}_{packed}",
                                    config,
                                    rotary,
                                    rotary_interleaved,
                                    packed,
                                    softcap,
                                )


def gqa_no_past_flash_attention_test_cases():
    batches = [3] if pipeline_mode else [1, 3, 5]
    seqs = (
        [
            (127, 127),
            (35, 35),
            (2000, 2000),
            (200, 200),
            (240, 240),
        ]
        if pipeline_mode
        else [
            (127, 127),
            (35, 35),
            (2000, 2000),
            (200, 200),
            (240, 240),
        ]
    )
    num_h = [(32, 8), (9, 3), (4, 4)] if pipeline_mode else [(6, 6), (6, 3), (9, 9), (9, 3)]
    h_sizes = [16, 128, 256] if pipeline_mode else [32, 40, 64, 80, 96, 128, 160, 192, 224, 256]
    torch.manual_seed(69)

    for b in batches:
        for sq, skv in seqs:
            for n, n2 in num_h:
                for h in h_sizes:
                    for local in [False, True]:
                        for rotary, rotary_interleaved in rotary_options_for_current_os():
                            for packed in [False, True]:
                                for softcap in [0.0, 50.0]:
                                    config = PromptConfig(b, sq, skv, sq + skv + 8, n, n2, h)
                                    yield (
                                        str(config) + f"{local}_{rotary}_{rotary_interleaved}_{packed}",
                                        config,
                                        local,
                                        rotary,
                                        rotary_interleaved,
                                        packed,
                                        softcap,
                                    )


def gqa_past_memory_efficient_test_cases():
    batches = [5] if pipeline_mode else [1, 3, 5]
    seqs = (
        [(1, 128), (1, 1024), (1, 2048)]
        if pipeline_mode
        else [
            (1, 128),
            (1, 339),
            (1, 1024),
            (1, 5000),
            (1, 800),
            (1, 256),
            (1, 799),
            (1, 2048),
            # (1, 128 * 512),
            # (16, 128 * 512),
            # (128, 128),
        ]
    )
    num_h = [(32, 8), (9, 3), (4, 4)] if pipeline_mode else [(6, 6), (6, 3), (9, 9), (9, 3)]
    h_sizes = [16, 128, 256] if pipeline_mode else [32, 40, 64, 80, 96, 128, 160, 192, 224, 256]
    random.seed(69)

    for b in batches:
        for s, s2 in seqs:
            for n, n2 in num_h:
                for h in h_sizes:
                    for rotary, rotary_interleaved in rotary_options_for_current_os():
                        for packed in [False, True]:
                            for softcap in [0.0, 50.0]:
                                sp = random.randint(1, s2 - s) if s2 - s > 0 else 0
                                config = Config(b, s, s2, sp, n, n2, h)
                                yield (
                                    str(config) + f"{rotary}_{rotary_interleaved}_{packed}",
                                    config,
                                    rotary,
                                    rotary_interleaved,
                                    packed,
                                    softcap,
                                )


def gqa_past_flash_attention_test_cases():
    batches = [5] if pipeline_mode else [1, 3, 5]
    seqs = (
        [(1, 128), (1, 1024), (1, 2048)]
        if pipeline_mode
        else [
            (1, 128),
            (1, 339),
            (1, 1024),
            (1, 5000),
            (1, 800),
            (1, 256),
            (1, 799),
            (1, 2048),
            # (1, 128 * 512),
            # (16, 128 * 512),
            # (128, 128),
        ]
    )
    num_h = [(32, 8), (9, 3), (4, 4)] if pipeline_mode else [(6, 6), (6, 3), (9, 9), (9, 3)]
    h_sizes = [16, 128, 256] if pipeline_mode else [32, 40, 64, 80, 96, 128, 160, 192, 224, 256]
    random.seed(69)

    for b in batches:
        for s, s2 in seqs:
            for n, n2 in num_h:
                for h in h_sizes:
                    for local in [False, True]:
                        for rotary, rotary_interleaved in rotary_options_for_current_os():
                            for packed in [False, True]:
                                for softcap in [0.0, 50.0]:
                                    sp = random.randint(1, s2 - s) if s2 - s > 0 else 0
                                    config = Config(b, s, s2, sp, n, n2, h)
                                    yield (
                                        str(config) + f"{local}_{rotary}_{rotary_interleaved}_{packed}",
                                        config,
                                        local,
                                        rotary,
                                        rotary_interleaved,
                                        packed,
                                        softcap,
                                    )


def gqa_interactive_one_batch_flash_attention_test_cases():
    batches = [1]
    seqs = (
        [(2, 128), (128, 129), (32, 128), (256, 2048)]
        if pipeline_mode
        else [
            (1, 128),
            (32, 128),
            (128, 2048),
            (1235, 5000),
            (40, 800),
            (1, 256),
            (2, 799),
            (41, 2048),
            # (1, 128 * 512),
            # (16, 128 * 512),
            # (128, 128),
        ]
    )
    num_h = [(32, 8), (9, 3), (4, 4)] if pipeline_mode else [(6, 6), (6, 3), (9, 9), (9, 3)]
    h_sizes = [16, 128, 256] if pipeline_mode else [32, 40, 64, 80, 96, 128, 160, 192, 224, 256]
    random.seed(69)

    for b in batches:
        for s, s2 in seqs:
            for n, n2 in num_h:
                for h in h_sizes:
                    for local in [False, True]:
                        for rotary, rotary_interleaved in rotary_options_for_current_os():
                            for packed in [False, True]:
                                config = Config(b, s, s2, -1, n, n2, h)
                                yield (
                                    str(config) + f"{local}_{rotary}_{rotary_interleaved}_{packed}",
                                    config,
                                    local,
                                    rotary,
                                    rotary_interleaved,
                                    packed,
                                )


def gqa_interactive_one_batch_memory_efficient_attention_test_cases():
    batches = [1]
    seqs = (
        [(2, 128), (128, 129), (32, 128), (256, 2048)]
        if pipeline_mode
        else [
            (1, 128),
            (32, 128),
            (128, 2048),
            (1235, 5000),
            (40, 800),
            (1, 256),
            (2, 799),
            (41, 2048),
            # (1, 128 * 512),
            # (16, 128 * 512),
            # (128, 128),
        ]
    )
    num_h = [(32, 8), (9, 3), (4, 4)] if pipeline_mode else [(6, 6), (6, 3), (9, 9), (9, 3)]
    h_sizes = [16, 128, 256] if pipeline_mode else [32, 40, 64, 80, 96, 128, 160, 192, 224, 256]
    random.seed(69)

    for b in batches:
        for s, s2 in seqs:
            for n, n2 in num_h:
                for h in h_sizes:
                    for rotary, rotary_interleaved in rotary_options_for_current_os():
                        for packed in [False, True]:
                            config = Config(b, s, s2, -1, n, n2, h)
                            yield (
                                str(config) + f"{rotary}_{rotary_interleaved}_{packed}",
                                config,
                                rotary,
                                rotary_interleaved,
                                packed,
                            )


class TestGQA(unittest.TestCase):
    @parameterized.expand(gqa_no_past_memory_efficient_test_cases())
    def test_gqa_no_past_memory_efficient(self, _, config, rotary, rotary_interleaved, packed, softcap):
        if not has_memory_efficient():
            return
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "1"
        print("------- MEMORY EFFICIENT ATTENTION (PROMPT CASE) ---------")

        parity_check_gqa_prompt(
            config,
            rtol=5e-3,
            atol=5e-3,
            past_format=Formats.BNSH,
            rotary=rotary,
            rotary_interleaved=rotary_interleaved,
            packed=packed,
            softcap=softcap,
            use_smooth_softmax=False,
        )
        parity_check_gqa_prompt_no_buff(
            config,
            rtol=5e-3,
            atol=5e-3,
            past_format=Formats.BNSH,
            rotary=rotary,
            rotary_interleaved=rotary_interleaved,
            packed=packed,
            softcap=softcap,
            use_smooth_softmax=True,
        )

    @parameterized.expand(gqa_no_past_flash_attention_test_cases())
    def test_gqa_no_past_flash_attention(self, _, config, local, rotary, rotary_interleaved, packed, softcap):
        if not has_flash_attention():
            return
        print("------- FLASH ATTENTION (PROMPT CASE) --------")
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "0"

        parity_check_gqa_prompt(
            config,
            local=local,
            past_format=Formats.BNSH,
            rotary=rotary,
            rotary_interleaved=rotary_interleaved,
            packed=packed,
            softcap=softcap,
            use_smooth_softmax=True,
        )
        parity_check_gqa_prompt_no_buff(
            config,
            local=local,
            past_format=Formats.BNSH,
            rotary=rotary,
            rotary_interleaved=rotary_interleaved,
            packed=packed,
            softcap=softcap,
            use_smooth_softmax=False,
        )

    @parameterized.expand(gqa_past_memory_efficient_test_cases())
    def test_gqa_past_memory_efficient(self, _, config, rotary, rotary_interleaved, packed, softcap):
        if not has_memory_efficient():
            return
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "1"
        print("-------- MEMORY EFFICIENT (TOKEN GEN) --------")

        parity_check_gqa_past(
            config,
            past_format=Formats.BNSH,
            rtol=1e-3,
            atol=1e-3,
            rotary=rotary,
            rotary_interleaved=rotary_interleaved,
            packed=packed,
            softcap=softcap,
            use_smooth_softmax=True,
        )
        parity_check_gqa_past_no_buff(
            config,
            past_format=Formats.BNSH,
            rtol=1e-3,
            atol=1e-3,
            rotary=rotary,
            rotary_interleaved=rotary_interleaved,
            packed=packed,
            softcap=softcap,
            use_smooth_softmax=False,
        )

    @parameterized.expand(gqa_past_flash_attention_test_cases())
    def test_gqa_past_flash_attention(self, _, config, local, rotary, rotary_interleaved, packed, softcap):
        if not has_flash_attention():
            return
        print("------- FLASH ATTENTION (TOKEN GEN) -------")
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "0"

        parity_check_gqa_past(
            config,
            local=local,
            past_format=Formats.BNSH,
            rtol=1e-3,
            atol=1e-3,
            rotary=rotary,
            rotary_interleaved=rotary_interleaved,
            packed=packed,
            softcap=softcap,
            use_smooth_softmax=False,
        )
        parity_check_gqa_past_no_buff(
            config,
            local=local,
            past_format=Formats.BNSH,
            rtol=1e-3,
            atol=1e-3,
            rotary=rotary,
            rotary_interleaved=rotary_interleaved,
            packed=packed,
            softcap=softcap,
            use_smooth_softmax=True,
        )

    @parameterized.expand(gqa_interactive_one_batch_flash_attention_test_cases())
    def test_gqa_interactive_one_batch_flash_attention(self, _, config, local, rotary, rotary_interleaved, packed):
        if not has_flash_attention():
            return
        print("------- FLASH ATTENTION (INTERACTIVE) -------")
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "0"

        parity_check_gqa_past(
            config,
            local=local,
            past_format=Formats.BNSH,
            rtol=5e-3,
            atol=5e-3,
            rotary=rotary,
            rotary_interleaved=rotary_interleaved,
            packed=packed,
        )
        parity_check_gqa_past_no_buff(
            config,
            local=local,
            past_format=Formats.BNSH,
            rtol=5e-3,
            atol=5e-3,
            rotary=rotary,
            rotary_interleaved=rotary_interleaved,
            packed=packed,
        )

    @parameterized.expand(gqa_interactive_one_batch_memory_efficient_attention_test_cases())
    def test_gqa_interactive_one_batch_memory_efficient_attention(self, _, config, rotary, rotary_interleaved, packed):
        if not has_memory_efficient():
            return
        os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "1"
        print("-------- MEMORY EFFICIENT (INTERACTIVE) --------")

        parity_check_gqa_past(
            config,
            past_format=Formats.BNSH,
            rtol=5e-3,
            atol=5e-3,
            rotary=rotary,
            rotary_interleaved=rotary_interleaved,
            packed=packed,
        )
        parity_check_gqa_past_no_buff(
            config,
            past_format=Formats.BNSH,
            rtol=5e-3,
            atol=5e-3,
            rotary=rotary,
            rotary_interleaved=rotary_interleaved,
            packed=packed,
        )


if __name__ == "__main__":
    unittest.main()
