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
import unittest
from dataclasses import dataclass
from enum import Enum

import numpy
import torch
from bert_padding import pad_input, unpad_input
from einops import rearrange, repeat
from onnx import TensorProto, helper

from onnxruntime import InferenceSession, OrtValue, SessionOptions

torch.manual_seed(0)

pipeline_mode = True  # Reduces number of tests so pipeline doesn't time out

RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"

# These will now be set dynamically in tests
ORT_TYPE = None
TORCH_TYPE = None
NUMPY_TYPE = None
RTOL = None
ATOL = None


class Formats(Enum):
    BSNH = 0
    BNSH = 1


class QKOutputType(Enum):
    NO_OUTPUT = 0
    BEFORE_SOFTMAX = 1
    AFTER_SOFTMAX = 2


@dataclass
class Config:
    batch_size: int = 0
    sequence_length: int = 0
    kv_sequence_length: int = 0
    past_sequence_length: int = 0
    num_heads: int = 0
    kv_num_heads: int = 0
    head_size: int = 0
    has_position_ids: bool = False
    has_attention_bias: bool = False
    has_head_sink: bool = False
    qk_output: QKOutputType = QKOutputType.NO_OUTPUT


@dataclass
class PromptConfig:
    batch_size: int = 0
    q_sequence_length: int = 0
    kv_sequence_length: int = 0
    buffer_sequence_length: int = 0
    num_heads: int = 0
    kv_num_heads: int = 0
    head_size: int = 0
    has_position_ids: bool = False
    has_attention_bias: bool = False
    has_head_sink: bool = False
    qk_output: QKOutputType = QKOutputType.NO_OUTPUT


# LLaMA Microsoft model
class LlamaMSRotaryEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def rotate_tensor(
        self,
        x: torch.Tensor,  # BxSxNxH
        cos: torch.Tensor,  # 1xSx1x(H/2)
        sin: torch.Tensor,  # 1xSx1x(H/2)
        pos: torch.Tensor,
        interleaved: bool,
    ):
        # Dimension of x is [batch_size, seq_len, n_heads, head_dim]
        rot_dim = 2 * cos.shape[3]

        # Dolly requires partial rotation
        x_rot = x[:, :, :, :rot_dim]

        if interleaved:
            x1 = x_rot[:, :, :, 0::2]
            x2 = x_rot[:, :, :, 1::2]
        else:
            half = x_rot.shape[-1] // 2
            x1 = x[:, :, :, 0:half]
            x2 = x[:, :, :, half : 2 * half]

        seq_len = x.shape[1]

        # cos_x: (1, S, 1, H/2)
        # sin_x: (1, S, 1, H/2)
        # x1: (B, S, N, H/2)
        # x2: (B, S, N, H/2)
        if seq_len == 1:
            batch_size = x.shape[0]
            pos_i = pos.unsqueeze(1).unsqueeze(2).unsqueeze(3).long()
            cos_x = cos.expand(batch_size, -1, -1, -1)
            sin_x = sin.expand(batch_size, -1, -1, -1)
            cos_x = cos_x.gather(1, pos_i.expand(-1, -1, cos.shape[2], cos.shape[3]))
            sin_x = sin_x.gather(1, pos_i.expand(-1, -1, sin.shape[2], sin.shape[3]))
            real = cos_x * x1 - sin_x * x2
            imag = sin_x * x1 + cos_x * x2
            if interleaved:
                x_rot[:, :, :, 0::2] = real
                x_rot[:, :, :, 1::2] = imag
            else:
                x_rot = torch.cat((real, imag), dim=-1)
        else:
            batch_size = x.shape[0]
            cos_x = torch.zeros((batch_size, seq_len, 1, cos.shape[3]), device=x.device)
            sin_x = torch.zeros((batch_size, seq_len, 1, sin.shape[3]), device=x.device)
            for b in range(x.shape[0]):
                cos_x[b] = cos[0, pos[b] : pos[b] + seq_len, :, :]
                sin_x[b] = sin[0, pos[b] : pos[b] + seq_len, :, :]
            real = cos_x * x1 - sin_x * x2
            imag = sin_x * x1 + cos_x * x2
            if interleaved:
                x_rot[:, :, :, 0::2] = real
                x_rot[:, :, :, 1::2] = imag
            else:
                x_rot = torch.cat((real, imag), dim=-1)

        return torch.cat((x_rot, x[:, :, :, rot_dim:]), dim=-1)

    def forward(self, x, cos, sin, pos, interleaved):
        return self.rotate_tensor(x, cos, sin, pos, interleaved)


def create_group_query_attention_graph_prompt(
    config,
    ort_type,
    past_kv_format=Formats.BSNH,
    share_buffer=True,
    local_window_size=-1,
    rotary=False,
    rotary_interleaved=False,
    packed=False,
    softcap=0.0,
    use_smooth_softmax=False,
):
    past_kv_seqlen = config.buffer_sequence_length if share_buffer else 0
    present_kv_seqlen = config.buffer_sequence_length if share_buffer else config.kv_sequence_length

    output_names = [
        "output",
        "present_key",
        "present_value",
    ]
    if config.qk_output != QKOutputType.NO_OUTPUT:
        output_names.append("output_qk")

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
                "position_ids" if config.has_position_ids else "",
                "attention_bias" if config.has_attention_bias else "",
                "head_sink" if config.has_head_sink else "",
            ],
            output_names,
            "GroupQueryAttention_0",
            num_heads=config.num_heads,
            kv_num_heads=config.kv_num_heads,
            local_window_size=local_window_size,
            do_rotary=rotary,
            rotary_interleaved=rotary_interleaved,
            softcap=softcap,
            smooth_softmax=1 if use_smooth_softmax else 0,
            qk_output=config.qk_output.value,
            # is_past_bsnh=1 if past_kv_format == Formats.BSNH else 0,
            # kv_share_buffer=1 if share_buffer else 0,
            domain="com.microsoft",
        ),
    ]

    graph_input = [
        helper.make_tensor_value_info(
            "query",
            ort_type,
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
                ort_type,
                [
                    config.batch_size,
                    config.kv_sequence_length,
                    config.kv_num_heads * config.head_size,
                ],
            ),
            helper.make_tensor_value_info(
                "value",
                ort_type,
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
                ort_type,
                [
                    config.batch_size,
                    past_kv_seqlen if past_kv_format == Formats.BSNH else config.kv_num_heads,
                    config.kv_num_heads if past_kv_format == Formats.BSNH else past_kv_seqlen,
                    config.head_size,
                ],
            ),
            helper.make_tensor_value_info(
                "past_value",
                ort_type,
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
                ort_type,
                [
                    config.buffer_sequence_length if share_buffer else config.kv_sequence_length,
                    (math.floor(config.head_size / 16) * 16) // 2,
                ],
            ),
            helper.make_tensor_value_info(
                "sin_cache",
                ort_type,
                [
                    config.buffer_sequence_length if share_buffer else config.kv_sequence_length,
                    (math.floor(config.head_size / 16) * 16) // 2,
                ],
            ),
        ]

    if config.has_position_ids:
        graph_input += [
            helper.make_tensor_value_info(
                "position_ids",
                TensorProto.INT64,
                [config.batch_size, config.kv_sequence_length],
            ),
        ]

    if config.has_attention_bias:
        graph_input += [
            helper.make_tensor_value_info(
                "attention_bias",
                ort_type,
                [config.batch_size, 1, config.kv_sequence_length, config.kv_sequence_length],
            ),
        ]

    if config.has_head_sink:
        graph_input += [
            helper.make_tensor_value_info(
                "head_sink",
                ort_type,
                [config.num_heads],
            ),
        ]

    graph_output = [
        helper.make_tensor_value_info(
            "output",
            ort_type,
            [config.batch_size, config.q_sequence_length, config.num_heads * config.head_size],
        ),
        helper.make_tensor_value_info(
            "present_key",
            ort_type,
            [
                config.batch_size,
                present_kv_seqlen if past_kv_format == Formats.BSNH else config.kv_num_heads,
                config.kv_num_heads if past_kv_format == Formats.BSNH else present_kv_seqlen,
                config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "present_value",
            ort_type,
            [
                config.batch_size,
                present_kv_seqlen if past_kv_format == Formats.BSNH else config.kv_num_heads,
                config.kv_num_heads if past_kv_format == Formats.BSNH else present_kv_seqlen,
                config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "present_key",
            ort_type,
            [
                config.batch_size,
                config.kv_sequence_length if past_kv_format == Formats.BSNH else config.kv_num_heads,
                config.kv_num_heads if past_kv_format == Formats.BSNH else config.kv_sequence_length,
                config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "present_value",
            ort_type,
            [
                config.batch_size,
                config.kv_sequence_length if past_kv_format == Formats.BSNH else config.kv_num_heads,
                config.kv_num_heads if past_kv_format == Formats.BSNH else config.kv_sequence_length,
                config.head_size,
            ],
        ),
    ]

    if config.qk_output != QKOutputType.NO_OUTPUT:
        graph_output += [
            helper.make_tensor_value_info(
                "output_qk",
                ort_type,
                [config.batch_size, config.num_heads, config.kv_sequence_length, config.kv_sequence_length],
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
    ort_type,
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

    output_names = [
        "output",
        "present_key",
        "present_value",
    ]
    if config.qk_output != QKOutputType.NO_OUTPUT:
        output_names.append("output_qk")

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
                "position_ids" if config.has_position_ids else "",
                "attention_bias" if config.has_attention_bias else "",
                "head_sink" if config.has_head_sink else "",
            ],
            output_names,
            "GroupQueryAttention_0",
            num_heads=config.num_heads,
            kv_num_heads=config.kv_num_heads,
            local_window_size=local_window_size,
            do_rotary=rotary,
            rotary_interleaved=rotary_interleaved,
            softcap=softcap,
            smooth_softmax=1 if use_smooth_softmax else 0,
            qk_output=config.qk_output.value,
            # is_past_bsnh=1 if past_kv_format == Formats.BSNH else 0,
            # kv_share_buffer=1 if share_buffer else 0,
            domain="com.microsoft",
        ),
    ]

    graph_input = [
        helper.make_tensor_value_info(
            "query",
            ort_type,
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
            ort_type,
            [
                config.batch_size,
                past_kv_seqlen if past_kv_format == Formats.BSNH else config.kv_num_heads,
                config.kv_num_heads if past_kv_format == Formats.BSNH else past_kv_seqlen,
                config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "past_value",
            ort_type,
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
                ort_type,
                [
                    config.batch_size,
                    config.sequence_length,
                    config.kv_num_heads * config.head_size,
                ],
            ),
            helper.make_tensor_value_info(
                "value",
                ort_type,
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
                ort_type,
                [
                    config.kv_sequence_length + (0 if share_buffer else config.sequence_length),
                    (math.floor(config.head_size / 16) * 16) // 2,
                ],
            ),
            helper.make_tensor_value_info(
                "sin_cache",
                ort_type,
                [
                    config.kv_sequence_length + (0 if share_buffer else config.sequence_length),
                    (math.floor(config.head_size / 16) * 16) // 2,
                ],
            ),
        ]

    if config.has_position_ids:
        graph_input += [
            helper.make_tensor_value_info(
                "position_ids", TensorProto.INT64, [config.batch_size, config.sequence_length]
            ),
        ]

    if config.has_attention_bias:
        graph_input += [
            helper.make_tensor_value_info(
                "attention_bias",
                ort_type,
                [config.batch_size, 1, config.sequence_length, present_kv_seqlen],
            ),
        ]

    if config.has_head_sink:
        graph_input += [
            helper.make_tensor_value_info(
                "head_sink",
                ort_type,
                [config.num_heads],
            ),
        ]

    graph_output = [
        helper.make_tensor_value_info(
            "output",
            ort_type,
            [config.batch_size, config.sequence_length, config.num_heads * config.head_size],
        ),
        helper.make_tensor_value_info(
            "present_key",
            ort_type,
            [
                config.batch_size,
                present_kv_seqlen if past_kv_format == Formats.BSNH else config.kv_num_heads,
                config.kv_num_heads if past_kv_format == Formats.BSNH else present_kv_seqlen,
                config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "present_value",
            ort_type,
            [
                config.batch_size,
                present_kv_seqlen if past_kv_format == Formats.BSNH else config.kv_num_heads,
                config.kv_num_heads if past_kv_format == Formats.BSNH else present_kv_seqlen,
                config.head_size,
            ],
        ),
    ]

    if config.qk_output != QKOutputType.NO_OUTPUT:
        graph_output += [
            helper.make_tensor_value_info(
                "output_qk",
                ort_type,
                [config.batch_size, config.num_heads, config.sequence_length, present_kv_seqlen],
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
        q: (batch_size, seqlen_q, num_heads, d)
        k: (batch_size, seqlen_k, num_heads_k, d)
        v: (batch_size, seqlen_k, num_heads_k, d)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen_q, num_heads, d = q.shape
    _, seqlen_k, num_heads_k, _ = k.shape
    assert k.shape == (batch_size, seqlen_k, num_heads_k, d)
    assert v.shape == (batch_size, seqlen_k, num_heads_k, d)

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
        assert num_heads == num_heads_k
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


def create_inputs(config: Config, torch_type, kv_packed=False, qkv_packed=True):
    qkv = torch.randn(
        config.batch_size,
        config.sequence_length,
        3,
        config.num_heads,
        config.head_size,
        device="cpu",
        dtype=torch_type,
        requires_grad=False,
    )
    key_padding_mask = generate_random_padding_mask(
        config.sequence_length, config.batch_size, device="cpu", mode="random"
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
    position_ids=None,
    attention_bias=None,
    head_sink=None,
    output_qk=None,
    window_size=-1,
    past_kv_format=Formats.BSNH,
    share_buffer=True,
    rotary_interleaved=False,
    softcap=0.0,
    use_smooth_softmax=False,
    ort_type=TensorProto.FLOAT16,
    numpy_type=numpy.float16,
):
    onnx_model_str = create_group_query_attention_graph_prompt(
        config,
        ort_type,
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

    if config.has_position_ids:
        assert position_ids is not None

    if config.has_attention_bias:
        assert attention_bias is not None

    if config.qk_output != QKOutputType.NO_OUTPUT:
        assert output_qk is not None

    if new_k is not None:
        new_k = torch.reshape(new_k, (config.batch_size, config.kv_sequence_length, -1))
        new_v = torch.reshape(new_v, (config.batch_size, config.kv_sequence_length, -1))

    sess_options = SessionOptions()
    ort_session = InferenceSession(onnx_model_str, sess_options, providers=["CPUExecutionProvider"])
    io_binding = ort_session.io_binding()
    ort_outputs = {}

    if share_buffer:
        ort_inputs = {
            "query": q.detach().cpu().numpy(),
            "past_key": OrtValue.ortvalue_from_numpy(past_k.detach().cpu().numpy(), "cpu", 0),
            "past_value": OrtValue.ortvalue_from_numpy(past_v.detach().cpu().numpy(), "cpu", 0),
            "seqlens_k": seqlens_k.detach().cpu().numpy().astype(numpy.int32),
            "total_sequence_length": torch.tensor([config.q_sequence_length], dtype=torch.int32).detach().cpu().numpy(),
        }
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

        if config.has_position_ids:
            ort_inputs["position_ids"] = position_ids.detach().cpu().numpy()
            io_binding.bind_cpu_input("position_ids", ort_inputs["position_ids"])

        if config.has_attention_bias:
            ort_inputs["attention_bias"] = attention_bias.detach().cpu().numpy()
            io_binding.bind_cpu_input("attention_bias", ort_inputs["attention_bias"])

        io_binding.bind_cpu_input("query", ort_inputs["query"])
        io_binding.bind_input(
            "past_key", "cpu", 0, numpy_type, ort_inputs["past_key"].shape(), ort_inputs["past_key"].data_ptr()
        )
        io_binding.bind_input(
            "past_value",
            "cpu",
            0,
            numpy_type,
            ort_inputs["past_value"].shape(),
            ort_inputs["past_value"].data_ptr(),
        )
        io_binding.bind_cpu_input("seqlens_k", ort_inputs["seqlens_k"])
        io_binding.bind_cpu_input("total_sequence_length", ort_inputs["total_sequence_length"])
        io_binding.bind_output("output")
        io_binding.bind_ortvalue_output("present_key", ort_inputs["past_key"])
        io_binding.bind_ortvalue_output("present_value", ort_inputs["past_value"])
    else:
        ort_inputs = {
            "query": q.detach().cpu().numpy(),
            "seqlens_k": seqlens_k.detach().cpu().numpy().astype(numpy.int32),
            "total_sequence_length": torch.tensor([config.q_sequence_length], dtype=torch.int32).detach().cpu().numpy(),
        }
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

        if config.has_position_ids:
            ort_inputs["position_ids"] = position_ids.detach().cpu().numpy()
            io_binding.bind_cpu_input("position_ids", ort_inputs["position_ids"])

        if config.has_attention_bias:
            ort_inputs["attention_bias"] = attention_bias.detach().cpu().numpy()
            io_binding.bind_cpu_input("attention_bias", ort_inputs["attention_bias"])

        io_binding.bind_cpu_input("query", ort_inputs["query"])
        io_binding.bind_cpu_input("seqlens_k", ort_inputs["seqlens_k"])
        io_binding.bind_cpu_input("total_sequence_length", ort_inputs["total_sequence_length"])
        io_binding.bind_output("output")
        io_binding.bind_output("present_key")
        io_binding.bind_output("present_value")

    if config.has_head_sink:
        ort_inputs["head_sink"] = head_sink.detach().cpu().numpy()
        io_binding.bind_cpu_input("head_sink", ort_inputs["head_sink"])

    if config.qk_output != QKOutputType.NO_OUTPUT:
        ort_outputs["output_qk"] = OrtValue.ortvalue_from_numpy(output_qk.detach().cpu().numpy(), "cpu", 0)
        io_binding.bind_ortvalue_output("output_qk", ort_outputs["output_qk"])

    ort_session.run_with_iobinding(io_binding)

    out_qk = None
    if config.qk_output != QKOutputType.NO_OUTPUT:
        ort_output, present_k, present_v, out_qk = io_binding.copy_outputs_to_cpu()
    else:
        ort_output, present_k, present_v = io_binding.copy_outputs_to_cpu()
    ort_output = numpy.array(ort_output)
    output = torch.tensor(ort_output)

    return output, present_k, present_v, out_qk


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
    position_ids=None,
    attention_bias=None,
    head_sink=None,
    output_qk=None,
    past_kv_format=Formats.BSNH,
    share_buffer=True,
    window_size=-1,
    rotary_interleaved=False,
    softcap=0.0,
    use_smooth_softmax=False,
    ort_type=TensorProto.FLOAT16,
    numpy_type=numpy.float16,
):
    assert seqlens_k is not None
    onnx_model_str = create_group_query_attention_graph_past(
        config,
        ort_type,
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

    if config.has_position_ids:
        assert position_ids is not None

    if config.has_attention_bias:
        assert attention_bias is not None

    if config.qk_output != QKOutputType.NO_OUTPUT:
        assert output_qk is not None

    if new_k is not None:
        new_k = torch.reshape(new_k, (config.batch_size, config.sequence_length, -1))
        new_v = torch.reshape(new_v, (config.batch_size, config.sequence_length, -1))

    sess_options = SessionOptions()
    ort_session = InferenceSession(onnx_model_str, sess_options, providers=["CPUExecutionProvider"])
    io_binding = ort_session.io_binding()
    ort_outputs = {}

    if share_buffer:
        ort_inputs = {
            "query": q.detach().cpu().numpy(),
            "past_key": OrtValue.ortvalue_from_numpy(past_k.detach().cpu().numpy(), "cpu", 0),
            "past_value": OrtValue.ortvalue_from_numpy(past_v.detach().cpu().numpy(), "cpu", 0),
            "seqlens_k": seqlens_k.detach().cpu().numpy().astype(numpy.int32),
            "total_sequence_length": torch.tensor([config.kv_sequence_length], dtype=torch.int32)
            .detach()
            .cpu()
            .numpy(),
        }
        if new_k is not None and new_v is not None:
            ort_inputs["key"] = new_k.detach().cpu().numpy()
            ort_inputs["value"] = new_v.detach().cpu().numpy()
            io_binding.bind_cpu_input("key", ort_inputs["key"])
            io_binding.bind_cpu_input("value", ort_inputs["value"])
        if cos is not None and sin is not None:
            ort_inputs["cos_cache"] = cos.detach().cpu().numpy()
            ort_inputs["sin_cache"] = sin.detach().cpu().numpy()
            io_binding.bind_cpu_input("cos_cache", ort_inputs["cos_cache"])
            io_binding.bind_cpu_input("sin_cache", ort_inputs["sin_cache"])

        if config.has_position_ids:
            ort_inputs["position_ids"] = position_ids.detach().cpu().numpy()
            io_binding.bind_cpu_input("position_ids", ort_inputs["position_ids"])

        if config.has_attention_bias:
            ort_inputs["attention_bias"] = attention_bias.detach().cpu().numpy()
            io_binding.bind_cpu_input("attention_bias", ort_inputs["attention_bias"])

        io_binding.bind_cpu_input("query", ort_inputs["query"])
        io_binding.bind_input(
            "past_key", "cpu", 0, numpy_type, ort_inputs["past_key"].shape(), ort_inputs["past_key"].data_ptr()
        )
        io_binding.bind_input(
            "past_value",
            "cpu",
            0,
            numpy_type,
            ort_inputs["past_value"].shape(),
            ort_inputs["past_value"].data_ptr(),
        )
        io_binding.bind_cpu_input("seqlens_k", ort_inputs["seqlens_k"])
        io_binding.bind_cpu_input("total_sequence_length", ort_inputs["total_sequence_length"])
        io_binding.bind_output("output")
        io_binding.bind_ortvalue_output("present_key", ort_inputs["past_key"])
        io_binding.bind_ortvalue_output("present_value", ort_inputs["past_value"])
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
        if new_k is not None and new_v is not None:
            ort_inputs["key"] = new_k.detach().cpu().numpy()
            ort_inputs["value"] = new_v.detach().cpu().numpy()
            io_binding.bind_cpu_input("key", ort_inputs["key"])
            io_binding.bind_cpu_input("value", ort_inputs["value"])
        if cos is not None and sin is not None:
            ort_inputs["cos_cache"] = cos.detach().cpu().numpy()
            ort_inputs["sin_cache"] = sin.detach().cpu().numpy()
            io_binding.bind_cpu_input("cos_cache", ort_inputs["cos_cache"])
            io_binding.bind_cpu_input("sin_cache", ort_inputs["sin_cache"])

        if config.has_position_ids:
            ort_inputs["position_ids"] = position_ids.detach().cpu().numpy()
            io_binding.bind_cpu_input("position_ids", ort_inputs["position_ids"])

        if config.has_attention_bias:
            ort_inputs["attention_bias"] = attention_bias.detach().cpu().numpy()
            io_binding.bind_cpu_input("attention_bias", ort_inputs["attention_bias"])

        io_binding.bind_cpu_input("query", ort_inputs["query"])
        io_binding.bind_cpu_input("past_key", ort_inputs["past_key"])
        io_binding.bind_cpu_input("past_value", ort_inputs["past_value"])
        io_binding.bind_cpu_input("seqlens_k", ort_inputs["seqlens_k"])
        io_binding.bind_cpu_input("total_sequence_length", ort_inputs["total_sequence_length"])
        io_binding.bind_output("output")
        io_binding.bind_output("present_key")
        io_binding.bind_output("present_value")

    if config.has_head_sink:
        ort_inputs["head_sink"] = head_sink.detach().cpu().numpy()
        io_binding.bind_cpu_input("head_sink", ort_inputs["head_sink"])

    if config.qk_output != QKOutputType.NO_OUTPUT:
        ort_outputs["output_qk"] = OrtValue.ortvalue_from_numpy(output_qk.detach().cpu().numpy(), "cpu", 0)
        io_binding.bind_ortvalue_output("output_qk", ort_outputs["output_qk"])

    ort_session.run_with_iobinding(io_binding)

    out_qk = None
    if config.qk_output != QKOutputType.NO_OUTPUT:
        ort_output, present_k, present_v, out_qk = io_binding.copy_outputs_to_cpu()
    else:
        ort_output, present_k, present_v = io_binding.copy_outputs_to_cpu()
    ort_output = numpy.array(ort_output)
    output = torch.tensor(ort_output)

    return output, present_k, present_v, out_qk


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
            col_idx <= row_idx + sk - sq - window_size[0],
        )


def smooth_softmax_ref(x, head_sink):
    """
    Arguments:
        x: (batch_size, num_heads, seqlen_q, seqlen_k)
        head_sink: (num_heads) or None
    Output:
        y: (batch_size, num_heads, seqlen_q, seqlen_k)
    """
    assert len(x.shape) == 4
    b, n, s, t = x.shape

    if head_sink is not None:
        assert len(head_sink.shape) == 1
        assert head_sink.shape[0] == x.shape[1]
        sink = head_sink.reshape(1, n, 1, 1).expand(b, -1, s, -1)
    else:
        sink = torch.zeros(b, n, s, 1, dtype=x.dtype)

    y = torch.cat([x, sink], dim=-1)
    y = torch.softmax(y, dim=-1)
    y = y[..., :-1]
    return y


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
    head_sink=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, num_heads, head_dim)
        k: (batch_size, seqlen_k, num_heads_k, head_dim)
        v: (batch_size, seqlen_k, num_heads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, num_heads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling k, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
        use_smooth_softmax: whether use smooth softmax or not
        head_sink: (num_heads) or None
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        masked_scores: (batch_size, nheads, seqlen_q, seqlen_k), before softmax
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
    masked_scores = scores.clone()
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
        masked_scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), 0)
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
        )
        masked_scores.masked_fill_(local_mask, 0.0)
        scores.masked_fill_(local_mask, float("-inf"))

    if use_smooth_softmax or (head_sink is not None):
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

    return output.to(dtype=dtype_og), masked_scores.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def attention_qkvpacked_ref(
    qkv,
    key_padding_mask=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    upcast=True,
    reorder_ops=False,
    use_smooth_softmax=False,
    head_sink=None,
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
        head_sink=head_sink,
    )


def get_custom_attention_bias(
    batch_size, sequence_length, total_seq_len, seqlens_k=None, past=False, torch_type=torch.float16
):
    if past:
        assert seqlens_k is not None
        attention_bias = torch.zeros((batch_size, 1, sequence_length, total_seq_len), dtype=torch_type)
        for b in range(batch_size):
            total_seq_len = seqlens_k[b] + 1
            past_seq_len = total_seq_len - sequence_length

            # Configure bias
            for i in range(sequence_length):
                for j in range(past_seq_len + i + 1, total_seq_len):
                    attention_bias[b][0][i][j] = -5000
    else:
        attention_bias = torch.rand(batch_size, 1, sequence_length, total_seq_len, dtype=torch_type)
        attention_bias = torch.triu(attention_bias, diagonal=1)

    return attention_bias


def get_custom_position_ids(batch_size, sequence_length, seqlens_k=None, past=False):
    if past:
        assert seqlens_k is not None
        position_ids_data = []
        for b in range(batch_size):
            total_seq_len = seqlens_k[b] + 1
            past_seq_len = total_seq_len - sequence_length
            position_ids_data.append(list(range(past_seq_len, past_seq_len + sequence_length)))

        position_ids = torch.tensor(data=position_ids_data, dtype=torch.int64)
    else:
        position_ids = torch.zeros((batch_size, sequence_length), dtype=torch.int64)

    return position_ids


def get_custom_head_sink(num_heads, torch_type=torch.float16):
    return torch.rand(num_heads, dtype=torch_type)


def parity_check_gqa_prompt(
    config,
    torch_type,
    numpy_type,
    ort_type,
    causal=True,
    local=False,
    past_format=Formats.BSNH,
    rotary=False,
    rotary_interleaved=False,
    packed=False,
    softcap=0.0,
    use_smooth_softmax=False,
    rtol=RTOL,
    atol=ATOL,
):
    q = torch.randn(
        config.batch_size,
        config.q_sequence_length,
        config.num_heads,
        config.head_size,
        device="cpu",
        dtype=torch_type,
        requires_grad=False,
    )
    k = torch.randn(
        config.batch_size,
        config.buffer_sequence_length if past_format == Formats.BSNH else config.kv_num_heads,
        config.kv_num_heads if past_format == Formats.BSNH else config.buffer_sequence_length,
        config.head_size,
        device="cpu",
        dtype=torch_type,
        requires_grad=False,
    )
    v = torch.randn(
        config.batch_size,
        config.buffer_sequence_length if past_format == Formats.BSNH else config.kv_num_heads,
        config.kv_num_heads if past_format == Formats.BSNH else config.buffer_sequence_length,
        config.head_size,
        device="cpu",
        dtype=torch_type,
        requires_grad=False,
    )
    new_k = torch.randn(
        config.batch_size,
        config.kv_sequence_length,
        config.kv_num_heads,
        config.head_size,
        device="cpu",
        dtype=torch_type,
        requires_grad=False,
    )
    new_v = torch.randn(
        config.batch_size,
        config.kv_sequence_length,
        config.kv_num_heads,
        config.head_size,
        device="cpu",
        dtype=torch_type,
        requires_grad=False,
    )

    head_sink = get_custom_head_sink(config.num_heads, torch_type) if config.has_head_sink else None

    window_size = (-1, -1)
    left_window_size = -1
    if local:
        left_window_size = random.randint(1, config.kv_sequence_length)
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
    cache_seqlens = torch.tensor([config.kv_sequence_length], device="cpu").repeat(config.batch_size)
    rotary_seqlens = torch.tensor([0], device="cpu").repeat(config.batch_size)

    if rotary:
        rotary_fraction = 1.0
        rotary_dim = math.floor(int(rotary_fraction * config.head_size) / 16) * 16
        angle = torch.rand(config.buffer_sequence_length, rotary_dim // 2, device="cpu") * 2 * math.pi
        cos = torch.cos(angle).to(dtype=torch_type)
        sin = torch.sin(angle).to(dtype=torch_type)
        rot = LlamaMSRotaryEmbedding()
        q_ro = rot(
            q.clone(), cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2), rotary_seqlens, rotary_interleaved
        )
        k_ro = rot(
            new_k.clone(),
            cos.unsqueeze(0).unsqueeze(2),
            sin.unsqueeze(0).unsqueeze(2),
            rotary_seqlens,
            rotary_interleaved,
        )
    else:
        cos, sin = None, None
        q_ro, k_ro = q, new_k

    position_ids = (
        get_custom_position_ids(config.batch_size, config.kv_sequence_length, seqlens_k=None, past=False)
        if config.has_position_ids
        else None
    )
    attention_bias = (
        get_custom_attention_bias(
            config.batch_size,
            config.kv_sequence_length,
            config.q_sequence_length,
            seqlens_k=None,
            past=False,
            torch_type=torch_type,
        )
        if config.has_attention_bias
        else None
    )

    output_qk = (
        torch.zeros(
            config.batch_size,
            config.num_heads,
            config.kv_sequence_length,
            config.q_sequence_length,
            device="cpu",
            dtype=torch_type,
            requires_grad=False,
        )
        if config.qk_output != QKOutputType.NO_OUTPUT
        else None
    )

    arange = rearrange(torch.arange(config.buffer_sequence_length, device="cpu"), "s -> 1 s")
    cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
    kv_seqlens = torch.tensor([config.kv_sequence_length], device="cpu").repeat(config.batch_size)
    kv_seqlens_expanded = rearrange(kv_seqlens, "b -> b 1")
    update_mask = arange < kv_seqlens_expanded
    k_cache_ref[update_mask] = rearrange(k_ro, "b s ... -> (b s) ...").to(dtype=torch_type)
    v_cache_ref[update_mask] = rearrange(new_v, "b s ... -> (b s) ...").to(dtype=torch_type)
    k_cache_rep = repeat(k_cache_ref, "b s h d -> b s (h g) d", g=config.num_heads // config.kv_num_heads)
    v_cache_rep = repeat(v_cache_ref, "b s h d -> b s (h g) d", g=config.num_heads // config.kv_num_heads)
    key_padding_mask = arange < cache_seqlens_expanded
    out_ref, out_qk_pre_softmax_ref, out_qk_post_softmax_ref = attention_ref(
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
        head_sink=head_sink,
    )
    out_ref = out_ref.detach().cpu().numpy()
    if past_format == Formats.BNSH:
        k_cache_ref = k_cache_ref.transpose(1, 2)
        v_cache_ref = v_cache_ref.transpose(1, 2)

    # Flash function
    # Cache seqlens is reduced by 1 since it is required to be past_seq_len + seq_len - 1
    if packed:
        packed_qkv = torch.concatenate([q, new_k, new_v], dim=2)
        out, present_k, present_v, out_qk = gqa_prompt_func(
            packed_qkv,
            k,
            v,
            config,
            None,
            None,
            cos,
            sin,
            cache_seqlens - 1,
            position_ids,
            attention_bias,
            head_sink,
            output_qk,
            left_window_size,
            past_format,
            True,
            rotary_interleaved,
            softcap,
            use_smooth_softmax=use_smooth_softmax,
            ort_type=ort_type,
            numpy_type=numpy_type,
        )
    else:
        out, present_k, present_v, out_qk = gqa_prompt_func(
            q,
            k,
            v,
            config,
            new_k,
            new_v,
            cos,
            sin,
            cache_seqlens - 1,
            position_ids,
            attention_bias,
            head_sink,
            output_qk,
            left_window_size,
            past_format,
            True,
            rotary_interleaved,
            softcap,
            use_smooth_softmax=use_smooth_softmax,
            ort_type=ort_type,
            numpy_type=numpy_type,
        )
    out = torch.squeeze(out, 0)
    out = torch.reshape(out, (config.batch_size, config.q_sequence_length, config.num_heads, config.head_size))
    out = out.detach().cpu().numpy()

    if config.qk_output != QKOutputType.NO_OUTPUT:
        out_qk_ref = (
            out_qk_post_softmax_ref if config.qk_output == QKOutputType.AFTER_SOFTMAX else out_qk_pre_softmax_ref
        )
        out_qk_ref = out_qk_ref.detach().cpu().numpy()

        for batch_idx in range(config.batch_size):
            total_seqlen = cache_seqlens[batch_idx]
            assert numpy.allclose(
                out_qk[batch_idx, :, :, :total_seqlen],
                out_qk_ref[batch_idx, :, :, :total_seqlen],
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )

    # Make sure past-present buffer updating correctly
    assert numpy.allclose(present_k, k_cache_ref.detach().cpu().numpy(), rtol=rtol, atol=atol, equal_nan=True)
    assert numpy.allclose(present_v, v_cache_ref.detach().cpu().numpy(), rtol=rtol, atol=atol, equal_nan=True)

    # Compare results
    all_close = numpy.allclose(out, out_ref, rtol=rtol, atol=atol, equal_nan=True)
    correct = GREEN + "True" + RESET if all_close else RED + "False" + RESET
    print(
        "KV-buffer",
        " packed:",
        packed,
        " causal:",
        causal,
        " local:",
        local,
        " rotary:",
        rotary,
        " rotary_interleaved:",
        rotary_interleaved,
        " softcap:",
        softcap,
        " smooth_softmax:",
        use_smooth_softmax,
        "past kv format:",
        "BSNH" if past_format == Formats.BSNH else "BNSH",
        " B:",
        config.batch_size,
        " S:",
        config.q_sequence_length,
        " kv S:",
        config.kv_sequence_length,
        " N:",
        config.num_heads,
        " kv N:",
        config.kv_num_heads,
        " h:",
        config.head_size,
        " has_position_ids:",
        config.has_position_ids,
        " has_attention_bias:",
        config.has_attention_bias,
        " qk_output:",
        config.qk_output,
        " Mean Error:",
        numpy.mean(numpy.abs(out - out_ref)),
        correct,
    )
    return all_close


def parity_check_gqa_prompt_no_buff(
    config,
    torch_type,
    numpy_type,
    ort_type,
    causal=True,
    local=False,
    past_format=Formats.BSNH,
    rotary=False,
    rotary_interleaved=False,
    packed=False,
    softcap=0.0,
    use_smooth_softmax=False,
    rtol=RTOL,
    atol=ATOL,
):
    q = torch.randn(
        config.batch_size,
        config.q_sequence_length,
        config.num_heads,
        config.head_size,
        device="cpu",
        dtype=torch_type,
        requires_grad=False,
    )
    new_k = torch.randn(
        config.batch_size,
        config.kv_sequence_length,
        config.kv_num_heads,
        config.head_size,
        device="cpu",
        dtype=torch_type,
        requires_grad=False,
    )
    new_v = torch.randn(
        config.batch_size,
        config.kv_sequence_length,
        config.kv_num_heads,
        config.head_size,
        device="cpu",
        dtype=torch_type,
        requires_grad=False,
    )

    window_size = (-1, -1)
    left_window_size = -1
    if local:
        left_window_size = random.randint(1, config.kv_sequence_length)
        window_size = (left_window_size, 0)
    elif causal:
        left_window_size = -1
        window_size = (-1, 0)

    # Pytorch to compare
    k_cache_ref = new_k.clone()
    v_cache_ref = new_v.clone()
    cache_seqlens = torch.tensor([config.kv_sequence_length], device="cpu").repeat(config.batch_size)
    rotary_seqlens = torch.tensor([0], device="cpu").repeat(config.batch_size)

    if rotary:
        rotary_fraction = 1.0
        rotary_dim = math.floor(int(rotary_fraction * config.head_size) / 16) * 16
        angle = torch.rand(config.kv_sequence_length, rotary_dim // 2, device="cpu") * 2 * math.pi
        cos = torch.cos(angle).to(dtype=torch_type)
        sin = torch.sin(angle).to(dtype=torch_type)
        rot = LlamaMSRotaryEmbedding()
        q_ro = rot(
            q.clone(), cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2), rotary_seqlens, rotary_interleaved
        )
        k_ro = rot(
            k_cache_ref.clone(),
            cos.unsqueeze(0).unsqueeze(2),
            sin.unsqueeze(0).unsqueeze(2),
            rotary_seqlens,
            rotary_interleaved,
        )
    else:
        cos, sin = None, None
        q_ro, k_ro = q, k_cache_ref
    k_cache_ref = k_ro

    position_ids = (
        get_custom_position_ids(config.batch_size, config.kv_sequence_length, seqlens_k=None, past=False)
        if config.has_position_ids
        else None
    )
    attention_bias = (
        get_custom_attention_bias(
            config.batch_size,
            config.kv_sequence_length,
            config.q_sequence_length,
            seqlens_k=None,
            past=False,
            torch_type=torch_type,
        )
        if config.has_attention_bias
        else None
    )

    head_sink = get_custom_head_sink(config.num_heads, torch_type=torch_type) if config.has_head_sink else None

    output_qk = (
        torch.zeros(
            config.batch_size,
            config.num_heads,
            config.kv_sequence_length,
            config.q_sequence_length,
            device="cpu",
            dtype=torch_type,
            requires_grad=False,
        )
        if config.qk_output != QKOutputType.NO_OUTPUT
        else None
    )

    brange = rearrange(torch.arange(config.kv_sequence_length, device="cpu"), "s -> 1 s")
    cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
    new_mask = brange < cache_seqlens_expanded
    k_cache_rep = repeat(k_cache_ref, "b s h d -> b s (h g) d", g=config.num_heads // config.kv_num_heads)
    v_cache_rep = repeat(v_cache_ref, "b s h d -> b s (h g) d", g=config.num_heads // config.kv_num_heads)
    out_ref, out_qk_pre_softmax_ref, out_qk_post_softmax_ref = attention_ref(
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
        head_sink=head_sink,
    )
    out_ref = out_ref.detach().cpu().numpy()
    if past_format == Formats.BNSH:
        k_cache_ref = k_cache_ref.transpose(1, 2)
        v_cache_ref = v_cache_ref.transpose(1, 2)

    # Flash function
    # Cache seqlens is reduced by 1 since it is required to be past_seq_len + seq_len - 1
    if packed:
        packed_qkv = torch.concatenate([q, new_k, new_v], dim=2)
        out, present_k, present_v, out_qk = gqa_prompt_func(
            packed_qkv,
            None,
            None,
            config,
            None,
            None,
            cos,
            sin,
            cache_seqlens - 1,
            position_ids,
            attention_bias,
            head_sink,
            output_qk,
            left_window_size,
            past_format,
            False,
            rotary_interleaved,
            softcap,
            use_smooth_softmax=use_smooth_softmax,
            ort_type=ort_type,
            numpy_type=numpy_type,
        )
    else:
        out, present_k, present_v, out_qk = gqa_prompt_func(
            q,
            None,
            None,
            config,
            new_k,
            new_v,
            cos,
            sin,
            cache_seqlens - 1,
            position_ids,
            attention_bias,
            head_sink,
            output_qk,
            left_window_size,
            past_format,
            False,
            rotary_interleaved,
            softcap,
            use_smooth_softmax=use_smooth_softmax,
            ort_type=ort_type,
            numpy_type=numpy_type,
        )
    out = torch.squeeze(out, 0)
    out = torch.reshape(out, (config.batch_size, config.q_sequence_length, config.num_heads, config.head_size))
    out = out.detach().cpu().numpy()

    if config.qk_output != QKOutputType.NO_OUTPUT:
        out_qk_ref = (
            out_qk_post_softmax_ref if config.qk_output == QKOutputType.AFTER_SOFTMAX else out_qk_pre_softmax_ref
        )
        out_qk_ref = out_qk_ref.detach().cpu().numpy()

        for batch_idx in range(config.batch_size):
            total_seqlen = cache_seqlens[batch_idx]
            assert numpy.allclose(
                out_qk[batch_idx, :, :, :total_seqlen],
                out_qk_ref[batch_idx, :, :, :total_seqlen],
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )

    # Make sure past-present buffer updating correctly
    assert numpy.allclose(present_k, k_cache_ref.detach().cpu().numpy(), rtol=rtol, atol=atol, equal_nan=True)
    assert numpy.allclose(present_v, v_cache_ref.detach().cpu().numpy(), rtol=rtol, atol=atol, equal_nan=True)

    # Compare results
    all_close = numpy.allclose(out, out_ref, rtol=rtol, atol=atol, equal_nan=True)
    correct = GREEN + "True" + RESET if all_close else RED + "False" + RESET
    print(
        "No buff",
        " packed:",
        packed,
        " causal:",
        causal,
        " local:",
        local,
        " rotary:",
        rotary,
        " rotary_interleaved:",
        rotary_interleaved,
        " softcap:",
        softcap,
        " smooth_softmax:",
        use_smooth_softmax,
        "past kv format:",
        "BSNH" if past_format == Formats.BSNH else "BNSH",
        " B:",
        config.batch_size,
        " S:",
        config.q_sequence_length,
        " kv S:",
        config.kv_sequence_length,
        " N:",
        config.num_heads,
        " kv N:",
        config.kv_num_heads,
        " h:",
        config.head_size,
        " has_position_ids:",
        config.has_position_ids,
        " has_attention_bias:",
        config.has_attention_bias,
        " qk_output:",
        config.qk_output,
        " Mean Error:",
        numpy.mean(numpy.abs(out - out_ref)),
        correct,
    )
    return all_close


def parity_check_gqa_past(
    config,
    torch_type,
    numpy_type,
    ort_type,
    causal=True,
    local=False,
    past_format=Formats.BSNH,
    rotary=False,
    rotary_interleaved=False,
    packed=False,
    softcap=0.0,
    use_smooth_softmax=False,
    rtol=RTOL,
    atol=ATOL,
):
    q = torch.randn(
        config.batch_size,
        config.sequence_length,
        config.num_heads,
        config.head_size,
        device="cpu",
        dtype=torch_type,
        requires_grad=False,
    )
    k = torch.randn(
        config.batch_size,
        config.kv_sequence_length if past_format == Formats.BSNH else config.kv_num_heads,
        config.kv_num_heads if past_format == Formats.BSNH else config.kv_sequence_length,
        config.head_size,
        device="cpu",
        dtype=torch_type,
        requires_grad=False,
    )
    v = torch.randn(
        config.batch_size,
        config.kv_sequence_length if past_format == Formats.BSNH else config.kv_num_heads,
        config.kv_num_heads if past_format == Formats.BSNH else config.kv_sequence_length,
        config.head_size,
        device="cpu",
        dtype=torch_type,
        requires_grad=False,
    )
    new_k = torch.randn(
        config.batch_size,
        config.sequence_length,
        config.kv_num_heads,
        config.head_size,
        device="cpu",
        dtype=torch_type,
        requires_grad=False,
    )
    new_v = torch.randn(
        config.batch_size,
        config.sequence_length,
        config.kv_num_heads,
        config.head_size,
        device="cpu",
        dtype=torch_type,
        requires_grad=False,
    )

    window_size = (-1, -1)
    left_window_size = -1
    if local:
        left_window_size = random.randint(1, config.kv_sequence_length)
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
        device="cpu",
    )

    if rotary:
        rotary_fraction = 1.0
        rotary_dim = math.floor(int(rotary_fraction * config.head_size) / 16) * 16
        angle = torch.rand(config.kv_sequence_length, rotary_dim // 2, device="cpu") * 2 * math.pi
        cos = torch.cos(angle).to(dtype=torch_type)
        sin = torch.sin(angle).to(dtype=torch_type)
        rot = LlamaMSRotaryEmbedding()
        q_ro = rot(
            q.clone(), cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2), cache_seqlens, rotary_interleaved
        )
        k_ro = rot(
            new_k.clone(),
            cos.unsqueeze(0).unsqueeze(2),
            sin.unsqueeze(0).unsqueeze(2),
            cache_seqlens,
            rotary_interleaved,
        )
    else:
        cos, sin = None, None
        q_ro, k_ro = q, new_k

    head_sink = get_custom_head_sink(config.num_heads, torch_type=torch_type) if config.has_head_sink else None

    arange = rearrange(torch.arange(config.kv_sequence_length, device="cpu"), "s -> 1 s")
    cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
    update_mask = torch.logical_and(
        cache_seqlens_expanded <= arange, arange < cache_seqlens_expanded + config.sequence_length
    )
    k_cache_ref[update_mask] = rearrange(k_ro, "b s ... -> (b s) ...").to(dtype=torch_type)
    v_cache_ref[update_mask] = rearrange(new_v, "b s ... -> (b s) ...").to(dtype=torch_type)
    k_cache_rep = repeat(k_cache_ref, "b s h d -> b s (h g) d", g=config.num_heads // config.kv_num_heads)
    v_cache_rep = repeat(v_cache_ref, "b s h d -> b s (h g) d", g=config.num_heads // config.kv_num_heads)
    key_padding_mask = arange < cache_seqlens_expanded + config.sequence_length
    out_ref, out_qk_pre_softmax_ref, out_qk_post_softmax_ref = attention_ref(
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
        head_sink=head_sink,
    )
    out_ref = out_ref.detach().cpu().numpy()
    if past_format == Formats.BNSH:
        k_cache_ref = k_cache_ref.transpose(1, 2)
        v_cache_ref = v_cache_ref.transpose(1, 2)

    cache_seqlens += config.sequence_length - 1

    position_ids = (
        get_custom_position_ids(config.batch_size, config.sequence_length, seqlens_k=cache_seqlens, past=True)
        if config.has_position_ids
        else None
    )
    attention_bias = (
        get_custom_attention_bias(
            config.batch_size,
            config.sequence_length,
            config.kv_sequence_length,
            seqlens_k=cache_seqlens,
            past=True,
            torch_type=torch_type,
        )
        if config.has_attention_bias
        else None
    )

    output_qk = (
        torch.zeros(
            config.batch_size,
            config.num_heads,
            config.sequence_length,
            config.kv_sequence_length,
            device="cpu",
            dtype=torch_type,
            requires_grad=False,
        )
        if config.qk_output != QKOutputType.NO_OUTPUT
        else None
    )

    # ORT function
    if packed:
        packed_qkv = torch.concatenate([q, new_k, new_v], dim=2)
        out, present_k, present_v, out_qk = gqa_past_func(
            packed_qkv,
            k,
            v,
            config,
            None,
            None,
            cos,
            sin,
            cache_seqlens,
            position_ids,
            attention_bias,
            head_sink,
            output_qk,
            past_format,
            True,
            left_window_size,
            rotary_interleaved,
            softcap,
            use_smooth_softmax=use_smooth_softmax,
            ort_type=ort_type,
            numpy_type=numpy_type,
        )
    else:
        out, present_k, present_v, out_qk = gqa_past_func(
            q,
            k,
            v,
            config,
            new_k,
            new_v,
            cos,
            sin,
            cache_seqlens,
            position_ids,
            attention_bias,
            head_sink,
            output_qk,
            past_format,
            True,
            left_window_size,
            rotary_interleaved,
            softcap,
            use_smooth_softmax=use_smooth_softmax,
            ort_type=ort_type,
            numpy_type=numpy_type,
        )
    out = torch.squeeze(out, 0)
    out = torch.reshape(out, (config.batch_size, config.sequence_length, config.num_heads, config.head_size))
    out = out.detach().cpu().numpy()

    if config.qk_output != QKOutputType.NO_OUTPUT:
        out_qk_ref = (
            out_qk_post_softmax_ref if config.qk_output == QKOutputType.AFTER_SOFTMAX else out_qk_pre_softmax_ref
        )
        out_qk_ref = out_qk_ref.detach().cpu().numpy()

        for batch_idx in range(config.batch_size):
            total_seqlen = cache_seqlens[batch_idx] + 1
            assert numpy.allclose(
                out_qk[batch_idx, :, :, :total_seqlen],
                out_qk_ref[batch_idx, :, :, :total_seqlen],
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )

    # Make sure past-present buffer updating correctly
    assert numpy.allclose(present_k, k_cache_ref.detach().cpu().numpy(), rtol=rtol, atol=atol, equal_nan=True)
    assert numpy.allclose(present_v, v_cache_ref.detach().cpu().numpy(), rtol=rtol, atol=atol, equal_nan=True)

    # Compare results
    all_close = numpy.allclose(out, out_ref, rtol=rtol, atol=atol, equal_nan=True)
    correct = GREEN + "True" + RESET if all_close else RED + "False" + RESET
    print(
        "KV-buffer",
        "past kv format:",
        "BSNH" if past_format == Formats.BSNH else "BNSH",
        " packed:",
        packed,
        " causal:",
        causal,
        " local:",
        local,
        " rotary:",
        rotary,
        " rotary_interleaved:",
        rotary_interleaved,
        " softcap:",
        softcap,
        " smooth_softmax:",
        use_smooth_softmax,
        " head_sink:",
        config.has_head_sink,
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
        " has_position_ids:",
        config.has_position_ids,
        " has_attention_bias:",
        config.has_attention_bias,
        " qk_output:",
        config.qk_output,
        " Mean Error:",
        numpy.mean(numpy.abs(out - out_ref)),
        correct,
    )
    return all_close


def parity_check_gqa_past_no_buff(
    config,
    torch_type,
    numpy_type,
    ort_type,
    causal=True,
    local=False,
    past_format=Formats.BSNH,
    rotary=False,
    rotary_interleaved=False,
    packed=False,
    softcap=0.0,
    use_smooth_softmax=False,
    rtol=RTOL,
    atol=ATOL,
):
    torch.manual_seed(69)
    q = torch.randn(
        config.batch_size,
        config.sequence_length,
        config.num_heads,
        config.head_size,
        device="cpu",
        dtype=torch_type,
        requires_grad=False,
    )
    k = torch.randn(
        config.batch_size,
        config.kv_sequence_length if past_format == Formats.BSNH else config.kv_num_heads,
        config.kv_num_heads if past_format == Formats.BSNH else config.kv_sequence_length,
        config.head_size,
        device="cpu",
        dtype=torch_type,
        requires_grad=False,
    )
    v = torch.randn(
        config.batch_size,
        config.kv_sequence_length if past_format == Formats.BSNH else config.kv_num_heads,
        config.kv_num_heads if past_format == Formats.BSNH else config.kv_sequence_length,
        config.head_size,
        device="cpu",
        dtype=torch_type,
        requires_grad=False,
    )
    new_k = torch.randn(
        config.batch_size,
        config.sequence_length,
        config.kv_num_heads,
        config.head_size,
        device="cpu",
        dtype=torch_type,
        requires_grad=False,
    )
    new_v = torch.randn(
        config.batch_size,
        config.sequence_length,
        config.kv_num_heads,
        config.head_size,
        device="cpu",
        dtype=torch_type,
        requires_grad=False,
    )

    window_size = (-1, -1)
    left_window_size = -1
    if local:
        left_window_size = random.randint(1, config.kv_sequence_length)
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
        device="cpu",
    )
    cache_seqlens[random.randint(0, config.batch_size - 1)] = config.kv_sequence_length

    if rotary:
        rotary_fraction = 1.0
        rotary_dim = math.floor(int(rotary_fraction * config.head_size) / 16) * 16
        angle = (
            torch.rand(config.kv_sequence_length + config.sequence_length, rotary_dim // 2, device="cpu") * 2 * math.pi
        )
        cos = torch.cos(angle).to(dtype=torch_type)
        sin = torch.sin(angle).to(dtype=torch_type)
        rot = LlamaMSRotaryEmbedding()
        q_ro = rot(
            q.clone(), cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2), cache_seqlens, rotary_interleaved
        )
        k_ro = rot(
            new_k.clone(),
            cos.unsqueeze(0).unsqueeze(2),
            sin.unsqueeze(0).unsqueeze(2),
            cache_seqlens,
            rotary_interleaved,
        )
    else:
        cos, sin = None, None
        q_ro, k_ro = q, new_k

    head_sink = get_custom_head_sink(config.num_heads, torch_type) if config.has_head_sink else None

    arange = rearrange(torch.arange(config.kv_sequence_length + config.sequence_length, device="cpu"), "s -> 1 s")
    cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
    update_mask = torch.logical_and(
        cache_seqlens_expanded <= arange, arange < cache_seqlens_expanded + config.sequence_length
    )
    k_cache_ref[update_mask] = rearrange(k_ro, "b s ... -> (b s) ...").to(dtype=torch_type)
    v_cache_ref[update_mask] = rearrange(new_v, "b s ... -> (b s) ...").to(dtype=torch_type)
    k_cache_rep = repeat(k_cache_ref, "b s h d -> b s (h g) d", g=config.num_heads // config.kv_num_heads)
    v_cache_rep = repeat(v_cache_ref, "b s h d -> b s (h g) d", g=config.num_heads // config.kv_num_heads)
    key_padding_mask = arange < cache_seqlens_expanded + config.sequence_length
    out_ref, out_qk_pre_softmax_ref, out_qk_post_softmax_ref = attention_ref(
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
        head_sink=head_sink,
    )
    out_ref = out_ref.detach().cpu().numpy()
    if past_format == Formats.BNSH:
        k_cache_ref = k_cache_ref.transpose(1, 2)
        v_cache_ref = v_cache_ref.transpose(1, 2)

    cache_seqlens += config.sequence_length - 1

    position_ids = (
        get_custom_position_ids(config.batch_size, config.sequence_length, seqlens_k=cache_seqlens, past=True)
        if config.has_position_ids
        else None
    )
    attention_bias = (
        get_custom_attention_bias(
            config.batch_size,
            config.sequence_length,
            config.kv_sequence_length + config.sequence_length,
            seqlens_k=cache_seqlens,
            past=True,
            torch_type=torch_type,
        )
        if config.has_attention_bias
        else None
    )

    output_qk = (
        torch.zeros(
            config.batch_size,
            config.num_heads,
            config.sequence_length,
            config.kv_sequence_length + config.sequence_length,
            device="cpu",
            dtype=torch_type,
            requires_grad=False,
        )
        if config.qk_output != QKOutputType.NO_OUTPUT
        else None
    )

    # Flash function
    if packed:
        packed_qkv = torch.concatenate([q, new_k, new_v], dim=2)
        out, present_k, present_v, out_qk = gqa_past_func(
            packed_qkv,
            k,
            v,
            config,
            None,
            None,
            cos,
            sin,
            cache_seqlens,
            position_ids,
            attention_bias,
            head_sink,
            output_qk,
            past_format,
            False,
            window_size=left_window_size,
            rotary_interleaved=rotary_interleaved,
            softcap=softcap,
            use_smooth_softmax=use_smooth_softmax,
            ort_type=ort_type,
            numpy_type=numpy_type,
        )
    else:
        out, present_k, present_v, out_qk = gqa_past_func(
            q,
            k,
            v,
            config,
            new_k,
            new_v,
            cos,
            sin,
            cache_seqlens,
            position_ids,
            attention_bias,
            head_sink,
            output_qk,
            past_format,
            False,
            window_size=left_window_size,
            rotary_interleaved=rotary_interleaved,
            softcap=softcap,
            use_smooth_softmax=use_smooth_softmax,
            ort_type=ort_type,
            numpy_type=numpy_type,
        )
    out = torch.squeeze(out, 0)
    out = torch.reshape(out, (config.batch_size, config.sequence_length, config.num_heads, config.head_size))
    out = out.detach().cpu().numpy()

    if config.qk_output != QKOutputType.NO_OUTPUT:
        out_qk_ref = (
            out_qk_post_softmax_ref if config.qk_output == QKOutputType.AFTER_SOFTMAX else out_qk_pre_softmax_ref
        )
        out_qk_ref = out_qk_ref.detach().cpu().numpy()

        for batch_idx in range(config.batch_size):
            total_seqlen = cache_seqlens[batch_idx] + 1
            assert numpy.allclose(
                out_qk[batch_idx, :, :, :total_seqlen],
                out_qk_ref[batch_idx, :, :, :total_seqlen],
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )

    # Compare results
    all_close = numpy.allclose(out, out_ref, rtol=rtol, atol=atol, equal_nan=True)
    correct = GREEN + "True" + RESET if all_close else RED + "False" + RESET
    print(
        "NO buff",
        " packed:",
        packed,
        " causal:",
        causal,
        " local:",
        local,
        " rotary:",
        rotary,
        " rotary_interleaved:",
        rotary_interleaved,
        "softcap",
        softcap,
        " smooth_softmax:",
        use_smooth_softmax,
        " head_sink:",
        config.has_head_sink,
        "past kv format:",
        "BSNH" if past_format == Formats.BSNH else "BNSH",
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
        " has_position_ids:",
        config.has_position_ids,
        " has_attention_bias:",
        config.has_attention_bias,
        " qk_output:",
        config.qk_output,
        " Mean Error:",
        numpy.mean(numpy.abs(out - out_ref)),
        correct,
    )
    return all_close


class TestGQA(unittest.TestCase):
    def setUp(self):
        # Define precision configurations
        self.precision_configs = [
            {
                "ort_type": TensorProto.FLOAT16,
                "torch_type": torch.float16,
                "numpy_type": numpy.float16,
                "rtol": 1e-2,
                "atol": 1e-2,
            },
            {
                "ort_type": TensorProto.FLOAT,
                "torch_type": torch.float32,
                "numpy_type": numpy.float32,
                "rtol": 1e-5,
                "atol": 1e-5,
            },
        ]

    def run_test_config(
        self,
        test_func,
        config_class,
        batches,
        seqs,
        num_h,
        h_sizes,
        pos_ids_attn_bias,
        qk_output,
        additional_params=None,
    ):
        if additional_params is None:
            additional_params = {}

        random.seed(69)
        torch.manual_seed(69)

        for precision in self.precision_configs:
            print(
                f"\nRunning tests with precision: {'FLOAT16' if precision['ort_type'] == TensorProto.FLOAT16 else 'FLOAT32'}"
            )
            for b in batches:
                for s, s2 in seqs:
                    for n, n2 in num_h:
                        for h in h_sizes:
                            for local in [False, True]:
                                for rotary, rotary_interleaved in [(False, False), (True, False), (True, True)]:
                                    for packed in [False, True]:
                                        for softcap in [0.0, 50.0]:
                                            for use_smooth_softmax in [False, True]:
                                                for has_pos, has_attn in pos_ids_attn_bias:
                                                    for head_sink in [False, True]:
                                                        if use_smooth_softmax and head_sink:
                                                            continue
                                                        for output_qk in qk_output:
                                                            if config_class == PromptConfig:
                                                                config = config_class(
                                                                    b,
                                                                    s,
                                                                    s2,
                                                                    s + s2 + 8,
                                                                    n,
                                                                    n2,
                                                                    h,
                                                                    has_pos,
                                                                    has_attn,
                                                                    head_sink,
                                                                    output_qk,
                                                                )
                                                            else:  # Config
                                                                sp = random.randint(1, s2 - s) if s2 - s > 0 else 0
                                                                config = config_class(
                                                                    b,
                                                                    s,
                                                                    s2,
                                                                    sp,
                                                                    n,
                                                                    n2,
                                                                    h,
                                                                    has_pos,
                                                                    has_attn,
                                                                    head_sink,
                                                                    output_qk,
                                                                )

                                                            params = {
                                                                "config": config,
                                                                "torch_type": precision["torch_type"],
                                                                "numpy_type": precision["numpy_type"],
                                                                "ort_type": precision["ort_type"],
                                                                "rtol": precision["rtol"],
                                                                "atol": precision["atol"],
                                                                "local": local,
                                                                "past_format": Formats.BNSH,
                                                                "rotary": rotary,
                                                                "rotary_interleaved": rotary_interleaved,
                                                                "packed": packed,
                                                                "softcap": softcap,
                                                                "use_smooth_softmax": use_smooth_softmax,
                                                            }
                                                            params.update(additional_params)

                                                            all_close = test_func(**params)
                                                            self.assertTrue(all_close)

    def test_gqa_no_past(self):
        print("-------- TEST GQA NO PAST (PROMPT CASE) ---------")
        batches = [3] if pipeline_mode else [1, 3, 5]
        seqs = (
            [(127, 127), (240, 240)]
            if pipeline_mode
            else [(127, 127), (35, 35), (2000, 2000), (200, 200), (240, 240), (8000, 8000)]
        )
        pos_ids_attn_bias = (
            [(False, False), (True, True)]
            if pipeline_mode
            else [(False, False), (True, True), (False, True), (True, False)]
        )
        num_h = [(32, 8)] if pipeline_mode else [(6, 6), (6, 3), (9, 9), (9, 3)]
        h_sizes = [128] if pipeline_mode else [32, 40, 64, 80, 96, 128, 160, 192, 224, 256]
        qk_output = (
            [QKOutputType.NO_OUTPUT]
            if pipeline_mode
            else [QKOutputType.NO_OUTPUT, QKOutputType.BEFORE_SOFTMAX, QKOutputType.AFTER_SOFTMAX]
        )

        # Test with buffer
        self.run_test_config(
            parity_check_gqa_prompt,
            PromptConfig,
            batches,
            seqs,
            num_h,
            h_sizes,
            pos_ids_attn_bias,
            qk_output,
        )
        # Test without buffer
        self.run_test_config(
            parity_check_gqa_prompt_no_buff,
            PromptConfig,
            batches,
            seqs,
            num_h,
            h_sizes,
            pos_ids_attn_bias,
            qk_output,
        )

    def test_gqa_past(self):
        print("-------- TEST GQA PAST (TOKEN GEN) ---------")
        batches = [1] if pipeline_mode else [1, 3, 5]
        seqs = (
            [(1, 128)]
            if pipeline_mode
            else [(1, 128), (1, 339), (1, 1024), (1, 5000), (1, 800), (1, 256), (1, 799), (1, 2048)]
        )
        pos_ids_attn_bias = (
            [(False, False), (True, True)]
            if pipeline_mode
            else [(False, False), (True, True), (False, True), (True, False)]
        )
        num_h = [(9, 3)] if pipeline_mode else [(6, 6), (6, 3), (9, 9), (9, 3)]
        h_sizes = [64] if pipeline_mode else [32, 40, 64, 80, 96, 128, 160, 192, 224, 256]
        qk_output = (
            [QKOutputType.NO_OUTPUT]
            if pipeline_mode
            else [QKOutputType.NO_OUTPUT, QKOutputType.BEFORE_SOFTMAX, QKOutputType.AFTER_SOFTMAX]
        )

        # Test with buffer
        self.run_test_config(parity_check_gqa_past, Config, batches, seqs, num_h, h_sizes, pos_ids_attn_bias, qk_output)
        # Test without buffer
        self.run_test_config(
            parity_check_gqa_past_no_buff,
            Config,
            batches,
            seqs,
            num_h,
            h_sizes,
            pos_ids_attn_bias,
            qk_output,
        )

    def test_gqa_interactive_one_batch(self):
        print("-------- TEST GQA INTERACTIVE ---------")
        batches = [1]
        seqs = (
            [(256, 2048)]
            if pipeline_mode
            else [(1, 128), (1, 339), (1, 1024), (1, 5000), (1, 800), (1, 256), (1, 799), (1, 2048)]
        )
        pos_ids_attn_bias = (
            [(False, False), (True, True)]
            if pipeline_mode
            else [(False, False), (True, True), (False, True), (True, False)]
        )
        qk_output = [QKOutputType.NO_OUTPUT, QKOutputType.BEFORE_SOFTMAX, QKOutputType.AFTER_SOFTMAX]
        num_h = [(32, 8)] if pipeline_mode else [(6, 6), (6, 3), (9, 9), (9, 3)]
        h_sizes = [32] if pipeline_mode else [32, 40, 64, 80, 96, 128, 160, 192, 224, 256]

        # Only test softcap=0.0 for interactive case as per original
        self.run_test_config(
            parity_check_gqa_past,
            Config,
            batches,
            seqs,
            num_h,
            h_sizes,
            pos_ids_attn_bias,
            qk_output,
            additional_params={"softcap": 0.0, "use_smooth_softmax": False},
        )
        self.run_test_config(
            parity_check_gqa_past_no_buff,
            Config,
            batches,
            seqs,
            num_h,
            h_sizes,
            pos_ids_attn_bias,
            qk_output,
            additional_params={"softcap": 0.0, "use_smooth_softmax": False},
        )


if __name__ == "__main__":
    unittest.main()
