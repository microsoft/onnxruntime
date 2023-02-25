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
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import math
import numpy as np
import torch
from torch import nn

torch.set_printoptions(threshold=10000)


def create_t5_mha_graph(
    batch_size,
    seq_len,
    kv_sequence_length,
    head_size,
    num_heads,
):
    from onnx import TensorProto, helper

    nodes = [
        helper.make_node(
            "MultiHeadAttention",
            [
                "query",
                "",  # key
                "",  # value
                "",  # bias
                "key_padding_mask",
                "",  # rel_pos_bias
                "past_key",
                "past_value",
            ],
            ["output"],
            "MHA_0",
            num_heads=num_heads,
            mask_filter_value=-10000.0,
            scale=1.0,
            domain="com.microsoft",
        ),
    ]

    initializers = []

    hidden_size = head_size * num_heads

    graph = helper.make_graph(
        nodes,
        "T5_MHA_Graph",
        [
            helper.make_tensor_value_info("query", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]),
            helper.make_tensor_value_info("key_padding_mask", TensorProto.INT32, [batch_size, kv_sequence_length]),
            helper.make_tensor_value_info(
                "past_key", TensorProto.FLOAT, [batch_size, num_heads, kv_sequence_length, head_size]
            ),
            helper.make_tensor_value_info(
                "past_value", TensorProto.FLOAT, [batch_size, num_heads, kv_sequence_length, head_size]
            ),
        ],
        [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]),
        ],
        initializers,
    )

    model = helper.make_model(graph)
    return model.SerializeToString()


class T5Config:
    def __init__(self, is_decoder, batch_size, seq_len, kv_sequence_length, num_heads, head_size):
        self.is_decoder = is_decoder
        self.relative_attention_num_buckets = 32
        self.relative_attention_max_distance = 128
        self.d_model = num_heads * head_size
        self.key_value_proj_dim = head_size
        self.n_heads = num_heads
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # ORT parameters
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.kv_sequence_length = kv_sequence_length
        self.head_size = head_size
        self.num_heads = num_heads
        self.hidden_size = self.d_model


class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.head_size
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        # self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()

        # ORT parameters
        self.batch_size = config.batch_size
        self.seq_len = config.seq_len
        self.kv_sequence_length = config.kv_sequence_length
        self.head_size = config.head_size
        self.num_heads = config.num_heads
        self.hidden_size = self.d_model

        # Create onnx graph
        self.onnx_graph = create_t5_mha_graph(
            config.batch_size, config.seq_len, config.kv_sequence_length, config.head_size, config.num_heads
        )

    def create_inputs(self):
        hidden_states = torch.normal(mean=0.5, std=0.1, size=(self.batch_size, self.seq_len, self.hidden_size)).to(
            torch.float32
        )
        key_value_states = torch.normal(
            mean=0.5, std=0.1, size=(self.batch_size, self.kv_sequence_length, self.hidden_size)
        ).to(torch.float32)
        past_key = torch.normal(
            mean=0.5, std=0.1, size=(self.batch_size, self.num_heads, self.kv_sequence_length, self.head_size)
        ).to(torch.float32)
        past_value = torch.normal(
            mean=0.5, std=0.1, size=(self.batch_size, self.num_heads, self.kv_sequence_length, self.head_size)
        ).to(torch.float32)
        past_key_value = (past_key, past_value)
        attention_mask = torch.ones((self.batch_size, self.kv_sequence_length)).to(torch.float32)
        return hidden_states, key_value_states, past_key_value, attention_mask

    def torch_forward(
        self,
        hidden_states,
        key_value_states=None,
        past_key_value=None,
        mask=None,
        position_bias=None,
        use_cache=False,
        query_length=None,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
            else:
                assert position_bias is not None

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        # attn_output = self.o(attn_output) # ORT puts this matmul outside of MHA op

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,)

        return outputs

    def ort_forward(
        self,
        hidden_states,
        key_value_states=None,
        past_key_value=None,
        mask=None,
        position_bias=None,
        use_cache=False,
        query_length=None,
    ):
        torch_query = self.q(hidden_states)
        torch_key_padding_mask = mask.to(torch.int32)
        torch_past_key = past_key_value[0]
        torch_past_value = past_key_value[1]

        ort_inputs = {
            "query": np.ascontiguousarray(torch_query.detach().numpy()),
            "key_padding_mask": np.ascontiguousarray(torch_key_padding_mask.detach().numpy()),
            "past_key": np.ascontiguousarray(torch_past_key.detach().numpy()),
            "past_value": np.ascontiguousarray(torch_past_value.detach().numpy()),
        }

        from onnxruntime import InferenceSession, SessionOptions

        sess_options = SessionOptions()
        ort_session = InferenceSession(self.onnx_graph, sess_options, providers=["CUDAExecutionProvider"])
        ort_output = ort_session.run(None, ort_inputs)

        output = torch.tensor(ort_output)

        return output


if __name__ == "__main__":
    batch_size = 1
    seq_len = 2
    num_heads = 2
    head_size = 4
    kv_sequence_length = 3

    config = T5Config(
        is_decoder=True,
        batch_size=batch_size,
        seq_len=seq_len,
        kv_sequence_length=kv_sequence_length,
        num_heads=num_heads,
        head_size=head_size,
    )
    T5CrossAttention = T5Attention(config, has_relative_attention_bias=False)

    hidden_states, key_value_states, past_key_value, attention_mask = T5CrossAttention.create_inputs()
    torch_output = T5CrossAttention.torch_forward(
        hidden_states, key_value_states, past_key_value, attention_mask, position_bias=None, use_cache=False
    )
    ort_output = T5CrossAttention.ort_forward(
        hidden_states, key_value_states, past_key_value, attention_mask, position_bias=None, use_cache=False
    )
    print("torch_output", torch_output)
    print("ort_output", ort_output)
