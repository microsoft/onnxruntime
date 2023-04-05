# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Running this script in Linux like the following
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python multihead_attention_op_test_data_gen.py

import math
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn


# Modified from BertSelfAttention in https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
class Attention(nn.Module):
    def __init__(
        self,
        num_attention_heads,
        hidden_dim,
        qk_head_size,
        v_head_size,
        is_decoder: bool,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.qk_head_size = qk_head_size
        self.v_head_size = v_head_size
        self.qk_hidden_size = self.num_attention_heads * self.qk_head_size
        self.v_hidden_size = self.num_attention_heads * self.v_head_size

        self.query = nn.Linear(hidden_dim, self.qk_hidden_size)
        self.key = nn.Linear(hidden_dim, self.qk_hidden_size)
        self.value = nn.Linear(hidden_dim, self.v_hidden_size)
        self.is_decoder = is_decoder

        # Do not reshape output for pretty print.
        self.reshape_output = False
        self.verbose = False

    def transpose_for_scores(self, x: torch.Tensor, head_size) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def get_extended_attention_mask(self, attention_mask: Tensor, dtype: torch.dtype) -> Tensor:
        assert attention_mask.dim() == 2 or attention_mask.dim() == 3
        extended_attention_mask = (
            attention_mask[:, None, :, :] if attention_mask.dim() == 3 else attention_mask[:, None, None, :]
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)
        if self.verbose:
            print("q", mixed_query_layer)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            k = self.key(encoder_hidden_states)
            if self.verbose:
                print("k", k)
            key_layer = self.transpose_for_scores(k, self.qk_head_size)
            if self.verbose:
                print("transposed key", key_layer)
            v = self.value(encoder_hidden_states)
            if self.verbose:
                print("v", v)
            value_layer = self.transpose_for_scores(v, self.v_head_size)
            if self.verbose:
                print("transposed value", value_layer)
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states), self.qk_head_size)
            value_layer = self.transpose_for_scores(self.value(hidden_states), self.v_head_size)
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states), self.qk_head_size)
            value_layer = self.transpose_for_scores(self.value(hidden_states), self.v_head_size)

        query_layer = self.transpose_for_scores(mixed_query_layer, self.qk_head_size)
        if self.verbose:
            print("transposed query", query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.qk_head_size)
        if self.verbose:
            print("QK", attention_scores)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + self.get_extended_attention_mask(attention_mask, hidden_states.dtype)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        if self.verbose:
            print("softmax", attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        if self.reshape_output:
            new_context_layer_shape = context_layer.size()[:-2] + (self.v_hidden_size,)
            context_layer = context_layer.view(new_context_layer_shape)

        print("output", context_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = (*outputs, past_key_value)

        return outputs


def run_cross_attention(
    hidden_dim,
    q_head_size,
    v_head_size,
    num_heads,
    batch_size,
    sequence_length,
    kv_sequence_length,
    key_padding_mask=None,
    has_bias=True,
):
    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)

    device = torch.device("cuda:0")
    mha = Attention(num_heads, hidden_dim, q_head_size, v_head_size, is_decoder=False).to(device).eval()
    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.to(device)
    torch.nn.init.uniform_(mha.query.weight, -0.5, 0.5)
    torch.nn.init.uniform_(mha.key.weight, -0.5, 0.5)
    torch.nn.init.uniform_(mha.value.weight, -0.5, 0.5)

    if has_bias:
        torch.nn.init.uniform_(mha.query.bias, -0.5, 0.5)
        torch.nn.init.uniform_(mha.key.bias, -0.5, 0.5)
        torch.nn.init.uniform_(mha.value.bias, -0.5, 0.5)
    else:
        torch.nn.init.zeros_(mha.query.bias)
        torch.nn.init.zeros_(mha.key.bias)
        torch.nn.init.zeros_(mha.value.bias)

    # Here we simulate input projection with MatMul but no bias:
    w_q = nn.Linear(hidden_dim, num_heads * q_head_size).to(device).eval()
    w_k = nn.Linear(hidden_dim, num_heads * q_head_size).to(device).eval()
    w_v = nn.Linear(hidden_dim, num_heads * v_head_size).to(device).eval()
    w_q.weight.copy_(mha.query.weight)
    w_k.weight.copy_(mha.key.weight)
    w_v.weight.copy_(mha.value.weight)
    torch.nn.init.zeros_(w_q.bias)
    torch.nn.init.zeros_(w_k.bias)
    torch.nn.init.zeros_(w_v.bias)

    torch.set_printoptions(profile="full", precision=8, linewidth=120, sci_mode=False)

    hidden_states = torch.empty(batch_size, sequence_length, hidden_dim, device="cuda")
    torch.nn.init.normal_(hidden_states)

    encoder_hidden_states = torch.empty(batch_size, kv_sequence_length, hidden_dim, device="cuda")
    torch.nn.init.normal_(encoder_hidden_states)

    input_q = w_q(hidden_states.clone())
    input_k = w_k(encoder_hidden_states.clone())
    input_v = w_v(encoder_hidden_states.clone())
    print("input_q", input_q)
    print("input_k", input_k)
    print("input_v", input_v)

    input_bias = torch.concat([mha.query.bias, mha.key.bias, mha.value.bias])
    print("input_bias", input_bias)
    if not has_bias:
        print("no bias!")

    # packed KV
    if q_head_size == v_head_size:
        packed_kv = torch.dstack(
            (
                input_k.reshape(batch_size * kv_sequence_length, num_heads, q_head_size),
                input_v.reshape(batch_size * kv_sequence_length, num_heads, v_head_size),
            )
        )
        packed_kv = packed_kv.reshape(batch_size, kv_sequence_length, num_heads, 2, q_head_size)
        print("packed_kv_5d", packed_kv)

    mha.forward(
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=key_padding_mask,
        past_key_value=None,
        output_attentions=False,
    )


def run_self_attention(
    hidden_dim,
    q_head_size,
    v_head_size,
    num_heads,
    batch_size,
    sequence_length,
    key_padding_mask=None,
    has_bias=True,
):
    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)

    device = torch.device("cuda:0")
    mha = Attention(num_heads, hidden_dim, q_head_size, v_head_size, is_decoder=False).to(device).eval()
    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.to(device)
    torch.nn.init.uniform_(mha.query.weight, -0.5, 0.5)
    torch.nn.init.uniform_(mha.key.weight, -0.5, 0.5)
    torch.nn.init.uniform_(mha.value.weight, -0.5, 0.5)

    if has_bias:
        torch.nn.init.uniform_(mha.query.bias, -0.5, 0.5)
        torch.nn.init.uniform_(mha.key.bias, -0.5, 0.5)
        torch.nn.init.uniform_(mha.value.bias, -0.5, 0.5)
    else:
        torch.nn.init.zeros_(mha.query.bias)
        torch.nn.init.zeros_(mha.key.bias)
        torch.nn.init.zeros_(mha.value.bias)

    # Here we simulate input projection with MatMul but no bias:
    w_q = nn.Linear(hidden_dim, num_heads * q_head_size).to(device).eval()
    w_k = nn.Linear(hidden_dim, num_heads * q_head_size).to(device).eval()
    w_v = nn.Linear(hidden_dim, num_heads * v_head_size).to(device).eval()
    w_q.weight.copy_(mha.query.weight)
    w_k.weight.copy_(mha.key.weight)
    w_v.weight.copy_(mha.value.weight)
    torch.nn.init.zeros_(w_q.bias)
    torch.nn.init.zeros_(w_k.bias)
    torch.nn.init.zeros_(w_v.bias)

    torch.set_printoptions(profile="full", precision=8, linewidth=120, sci_mode=False)

    hidden_states = torch.empty(batch_size, sequence_length, hidden_dim, device="cuda")
    torch.nn.init.normal_(hidden_states)

    input_q = w_q(hidden_states.clone())
    input_k = w_k(hidden_states.clone())
    input_v = w_v(hidden_states.clone())
    print("input_q", input_q)
    print("input_k", input_k)
    print("input_v", input_v)

    input_bias = torch.concat([mha.query.bias, mha.key.bias, mha.value.bias])
    print("input_bias", input_bias)
    if not has_bias:
        print("no bias!")

    # packed QKV
    if q_head_size == v_head_size:
        packed_qkv = torch.dstack(
            (
                input_q.reshape(batch_size * sequence_length, num_heads, q_head_size),
                input_k.reshape(batch_size * sequence_length, num_heads, q_head_size),
                input_v.reshape(batch_size * sequence_length, num_heads, v_head_size),
            )
        )
        packed_qkv = packed_qkv.reshape(batch_size, sequence_length, num_heads, 3, q_head_size)
        print("packed_qkv_5d", packed_qkv)

    mha.forward(
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=key_padding_mask,
        past_key_value=None,
        output_attentions=False,
    )


def run_cross_batch2_headsize_40():
    hidden_dim = 80
    q_head_size = 40
    v_head_size = 40
    num_heads = 2
    batch_size = 2
    sequence_length = 3
    kv_sequence_length = 5
    run_cross_attention(
        hidden_dim, q_head_size, v_head_size, num_heads, batch_size, sequence_length, kv_sequence_length
    )


def run_cross_batch1_headsize_16():
    hidden_dim = 32
    q_head_size = 16
    v_head_size = 16
    num_heads = 2
    batch_size = 1
    sequence_length = 2
    kv_sequence_length = 3
    run_cross_attention(
        hidden_dim, q_head_size, v_head_size, num_heads, batch_size, sequence_length, kv_sequence_length
    )


def run_cross_batch2_headsize_16_8():
    hidden_dim = 32
    q_head_size = 16
    v_head_size = 8
    num_heads = 3
    batch_size = 2
    sequence_length = 1
    kv_sequence_length = 3
    run_cross_attention(
        hidden_dim, q_head_size, v_head_size, num_heads, batch_size, sequence_length, kv_sequence_length
    )


def run_cross_batch2_headsize_32_right_side_padding():
    hidden_dim = 64
    q_head_size = 32
    v_head_size = 32
    num_heads = 2
    batch_size = 2
    sequence_length = 2
    kv_sequence_length = 3
    key_padding_mask = torch.tensor([[1, 0, 0], [1, 1, 0]], dtype=torch.int32).cuda()

    run_cross_attention(
        hidden_dim,
        q_head_size,
        v_head_size,
        num_heads,
        batch_size,
        sequence_length,
        kv_sequence_length,
        key_padding_mask,
    )


def run_cross_batch1_headsize_32_left_side_padding():
    hidden_dim = 32
    q_head_size = 32
    v_head_size = 32
    num_heads = 1
    batch_size = 2
    sequence_length = 2
    kv_sequence_length = 3
    key_padding_mask = torch.tensor([[0, 1, 1], [0, 0, 1]], dtype=torch.int32).cuda()
    run_cross_attention(
        hidden_dim,
        q_head_size,
        v_head_size,
        num_heads,
        batch_size,
        sequence_length,
        kv_sequence_length,
        key_padding_mask,
    )


def run_cross_batch2_headsize_32_packed_kv():
    hidden_dim = 32
    q_head_size = 32
    v_head_size = 32
    num_heads = 1
    batch_size = 2
    sequence_length = 2
    kv_sequence_length = 3
    key_padding_mask = None
    has_bias = False
    run_cross_attention(
        hidden_dim,
        q_head_size,
        v_head_size,
        num_heads,
        batch_size,
        sequence_length,
        kv_sequence_length,
        key_padding_mask,
        has_bias,
    )


def run_self_batch2_headsize_32_packed_qkv():
    hidden_dim = 32
    q_head_size = 32
    v_head_size = 32
    num_heads = 1
    batch_size = 2
    sequence_length = 2
    key_padding_mask = None
    has_bias = False
    run_self_attention(
        hidden_dim, q_head_size, v_head_size, num_heads, batch_size, sequence_length, key_padding_mask, has_bias
    )


def create_test_data():
    """
    Create test data used in attention_op_test_helper.cc and multihead_attention_op_test.cc
    """
    print("CrossAttention_Batch2_HeadSize40")
    run_cross_batch2_headsize_40()

    print("CrossAttention_Batch1_HeadSize16")
    run_cross_batch1_headsize_16()

    print("CrossAttention_Batch2_HeadSize16_8")
    run_cross_batch2_headsize_16_8()

    print("CrossAttention_Batch2_HeadSize32_RightSidePadding")
    run_cross_batch2_headsize_32_right_side_padding()

    print("CrossAttention_Batch1_HeadSize32_LeftSidePadding")
    run_cross_batch1_headsize_32_left_side_padding()

    print("CrossAttention_Batch2_HeadSize32_PackedKV")
    run_cross_batch2_headsize_32_packed_kv()

    print("SelfAttention_Batch2_HeadSize32_PackedQKV")
    run_self_batch2_headsize_32_packed_qkv()


with torch.no_grad():
    create_test_data()
