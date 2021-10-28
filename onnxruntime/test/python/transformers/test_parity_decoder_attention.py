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
import numpy
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from typing import Dict, List, Optional, Tuple
import os

torch.manual_seed(0)

class Config:
    batch_size = 3
    sequence_length = 1
    kv_sequence_length = 5
    num_heads = 2
    head_size = 4
    embed_dim = num_heads * head_size

class AttentionProjection(nn.Module):
    def __init__(self, num_heads, head_dim, embed_dim, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def shape_state(self, state, batch_size):
        return state.view(batch_size * self.num_heads, -1, self.head_dim)

    def shape_proj(self, proj, batch_size):
        return proj.view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(
        self,
        query,
        key,
        layer_state: Optional[List[Tensor]],
        encoder_decoder_attention: bool,
        use_past=torch.tensor(False),
    ):
        bsz = torch._shape_as_tensor(query)[1]
        if layer_state is None or not use_past:
            if not encoder_decoder_attention:
                k = self.k_proj(query)
                v = self.v_proj(query)
                k = self.shape_proj(k, bsz)
                v = self.shape_proj(v, bsz)
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
                k = self.shape_proj(k, bsz)
                v = self.shape_proj(v, bsz)
        else:
            self_p_k, self_p_v, enc_dec_p_k, enc_dec_p_v = layer_state
            if not encoder_decoder_attention:
                k = self.k_proj(query)
                v = self.v_proj(query)
                k = self.shape_proj(k, bsz)
                v = self.shape_proj(v, bsz)
                k = torch.cat([self.shape_state(self_p_k, bsz), k], dim=1)
                v = torch.cat([self.shape_state(self_p_v, bsz), v], dim=1)
            else:
                k = self.shape_state(enc_dec_p_k, bsz)
                v = self.shape_state(enc_dec_p_v, bsz)

        return k, v

class AttentionForONNX(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        encoder_decoder_attention=False,  # otherwise self_attention
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.encoder_decoder_attention = encoder_decoder_attention
        self.k_v_proj = torch.jit.script(AttentionProjection(num_heads, self.head_dim, embed_dim, bias))
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"

    def _shape(self, tensor, seq_len, bsz):
        return tensor.contiguous().view(seq_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(
        self,
        query,
        key: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        layer_state: Optional[List[Tensor]] = None,
        attn_mask: Optional[Tensor] = None,
        output_attentions: bool=False,
        use_past=torch.tensor(False),
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""
        static_kv: bool = self.encoder_decoder_attention
        tgt_len, bsz, embed_dim = query.size()
        # get here for encoder decoder cause of static_kv
        k, v = self.k_v_proj(query, key, layer_state, self.encoder_decoder_attention, use_past)

        q = self.q_proj(query) * self.scaling
        q = self._shape(q, tgt_len, bsz)

        # Update cache
        if layer_state is not None:
            cached_shape = (bsz, self.num_heads, -1, self.head_dim)  # bsz must be first for reorder_cache
            if static_kv:
                # cross-attn
                layer_state[2] = k.view(*cached_shape)
                layer_state[3] = v.view(*cached_shape)
            else:
                # self-attn
                layer_state[0] = k.view(*cached_shape)
                layer_state[1] = v.view(*cached_shape)

        src_len = k.size(1)
        print(k.size(), key_padding_mask.shape)
        assert key_padding_mask is None or key_padding_mask.shape == (bsz, src_len)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attn_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if key_padding_mask is not None:  # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(reshaped, float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        assert v is not None
        attn_output = torch.bmm(attn_probs, v)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        if output_attentions:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        else:
            attn_weights = None
        return attn_output, attn_weights, layer_state

    def ORT_forward(
        self,
        query,
        key: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        layer_state: Optional[List[Tensor]] = None,
        attn_mask: Optional[Tensor] = None,
        output_attentions: bool=False,
        use_past=torch.tensor(False),
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""
        static_kv = 1 if self.encoder_decoder_attention else 0
        has_layer_state = 1 if layer_state is not None else 0
        use_past_cache = 1 if use_past else 0

        weight = torch.stack((self.q_proj.weight.transpose(0,1), self.k_v_proj.k_proj.weight.transpose(0,1), self.k_v_proj.v_proj.weight.transpose(0,1)), dim=1)
        weight = weight.reshape(self.embed_dim, 3 * self.embed_dim)

        bias = torch.stack((self.q_proj.bias, self.k_v_proj.k_proj.bias, self.k_v_proj.v_proj.bias), dim=0)
        bias = bias.reshape(3 * self.embed_dim)

        onnx_model_str = create_decoder_attention_graph(query, key, weight, bias, self.num_heads, static_kv, use_past_cache, has_layer_state)

        if layer_state is not None and use_past:
            self_p_k, self_p_v, enc_dec_p_k, enc_dec_p_v = layer_state
            if self.encoder_decoder_attention:
                key_cache, value_cache = enc_dec_p_k, enc_dec_p_v
            else:
                key_cache, value_cache = self_p_k, self_p_v

        ort_inputs = {
            'query': numpy.ascontiguousarray(query.cpu().numpy()),
            'key': numpy.ascontiguousarray(key.cpu().numpy()),
            'key_padding_mask': numpy.ascontiguousarray(key_padding_mask.cpu().numpy()),
            'key_cache': numpy.ascontiguousarray(key_cache.cpu().numpy()),
            'value_cache': numpy.ascontiguousarray(value_cache.cpu().numpy())
        }

        from onnxruntime import SessionOptions, InferenceSession
        sess_options = SessionOptions()
        ort_session = InferenceSession(onnx_model_str, sess_options, providers=['CUDAExecutionProvider'])
        ort_output = ort_session.run(None, ort_inputs)
        output, new_key_cache, new_value_cache = ort_output

        # bugbug: need to change to torch.tensor type
        if layer_state is not None:
            if self.encoder_decoder_attention:
                layer_state[2] = new_key_cache
                layer_state[3] = new_value_cache
            else:
                layer_state[0] = new_key_cache
                layer_state[1] = new_value_cache

        attn_output = self.out_proj(output)

        return attn_output, None, layer_state


def create_decoder_attention_graph(query, key, weight, bias, num_heads_, static_kv_, use_past_cache_, has_layer_state_):
    from onnx import helper, TensorProto

    S, B, NH = query.size()
    S2 = key.size()[0]
    N = num_heads_
    H = int(NH / N)

    nodes = [
        helper.make_node("DecoderAttention",
                         ["query", "key", "weight", "bias", "key_padding_mask", "key_cache", "value_cache"],
                         ["output", "new_key_cache", "new_value_cache"],
                         "DecoderAttention_0",
                         num_heads=num_heads_,
                         static_kv = static_kv_,
                         use_past = use_past_cache_,
                         has_layer_state = has_layer_state_,
                         domain="com.microsoft"),
    ]

    initializers = [
        helper.make_tensor('weight', TensorProto.FLOAT, [NH, 3 * NH],
                           weight.flatten().tolist()),
        helper.make_tensor('bias', TensorProto.FLOAT, [3 * NH],
                           bias.flatten().tolist()),
    ]

    graph = helper.make_graph(nodes, "DecoderAttention_Graph", [
        helper.make_tensor_value_info('query', TensorProto.FLOAT, [S, B, NH]),
        helper.make_tensor_value_info('key', TensorProto.FLOAT, [S2, B, NH]),
        helper.make_tensor_value_info('key_padding_mask', TensorProto.BOOL, [B, "mask_len"]),
        helper.make_tensor_value_info('key_cache', TensorProto.FLOAT, [B, N, "cache_len", H]),
        helper.make_tensor_value_info('value_cache', TensorProto.FLOAT, [B, N, "cache_len", H]),
    ], [
        helper.make_tensor_value_info('output', TensorProto.FLOAT, [S, B, NH]),
        helper.make_tensor_value_info('new_key_cache', TensorProto.FLOAT, [B, N, "new_cache_len", H]),
        helper.make_tensor_value_info('new_value_cache', TensorProto.FLOAT, [B, N, "new_cache_len", H]),
    ], initializers)

    model = helper.make_model(graph)
    return model.SerializeToString()


def create_inputs(config: Config, has_layer_state: bool, use_past: bool, has_key_padding_mask: bool):
    query = torch.normal(mean=0.0,
                         std=0.1,
                         size=(config.sequence_length,
                               config.batch_size,
                               config.embed_dim)
                        ).to(torch.float32)
    key = torch.normal(mean=0.0,
                       std=0.1,
                       size=(config.kv_sequence_length,
                             config.batch_size,
                             config.embed_dim)
                       ).to(torch.float32)

    if not has_key_padding_mask:
        key_padding_mask = None
    else:
        #bugbug: lens need to the new key sequence length
        key_padding_mask = torch.normal(mean=0.0,
                                        std=0.1,
                                        size=(config.batch_size,
                                              config.kv_sequence_length)
                                        ) > 0

    if not has_layer_state:
        layer_state = None
    else:
        layer_state = []
        cache = torch.normal(mean=0.0,
                             std=0.1,
                             size=(config.batch_size,
                                   config.num_heads,
                                   config.kv_sequence_length,
                                   config.head_size)
                             ).to(torch.float32)
        layer_state = [cache, cache, cache, cache]

    return query, key, key_padding_mask, layer_state, torch.tensor(use_past)


if __name__ == '__main__':
    torch.manual_seed(0)

    config = Config()
    query, key, key_padding_mask, layer_state, use_past = create_inputs(config,
                                                                        has_layer_state = True,
                                                                        use_past = True,
                                                                        has_key_padding_mask = True)
    attn = AttentionForONNX(config.embed_dim, config.num_heads, encoder_decoder_attention=True)
    attn_output, attn_weights, layer_state = attn.forward(query, key, key_padding_mask, layer_state, None, False, use_past)
    # bugbug: FAIL : Non-zero status code returned while running DecoderAttention node
    attn_output, attn_weights, layer_state = attn.ORT_forward(query, key, key_padding_mask, layer_state, None, False, use_past)
    print(attn_output, attn_weights, layer_state)
    #attn.ORT_forward(input_hidden_states, attention_mask)

    #print(input_hidden_states, attention_mask, output)