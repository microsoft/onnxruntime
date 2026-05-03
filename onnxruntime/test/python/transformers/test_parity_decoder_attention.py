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

import numpy
import torch
from torch import Tensor, nn
from torch.nn import functional as F

torch.manual_seed(0)

def my_bart_attention_forward(
    self,
    query,
    key: Tensor,
    key_padding_mask: Optional[Tensor],
    layer_state: Optional[List[Tensor]],
    attn_mask: Optional[Tensor] = None,
    output_attentions: bool = False,
    use_past=torch.tensor(False),
):
    static_kv: bool = self.encoder_decoder_attention
    q_weight = self.q_proj.weight.transpose(0, 1)
    q_weight = q_weight.reshape(self.embed_dim, self.embed_dim)

    kv_weight = torch.stack((self.k_v_proj.k_proj.weight.transpose(0, 1), self.k_v_proj.v_proj.weight.transpose(0, 1)), dim=1)
    kv_weight = kv_weight.reshape(self.embed_dim, 2 * self.embed_dim)

    bias = torch.stack((self.q_proj.bias, self.k_v_proj.k_proj.bias, self.k_v_proj.v_proj.bias), dim=0)
    bias = bias.reshape(3 * self.embed_dim)

    self_p_k, self_p_v, enc_dec_p_k, enc_dec_p_v = layer_state
    if static_kv:
        key_cache, value_cache = enc_dec_p_k, enc_dec_p_v
    else:
        key_cache, value_cache = self_p_k, self_p_v

    use_cache = True
    len_kv = key.size(2) if static_kv else (key_cache.size(2) + key.size(2))

    q = F.linear(query, q_weight, bias[:self.embed_dim])
    k, v = F.linear(key, kv_weight[:, :self.embed_dim], bias[self.embed_dim:2*self.embed_dim]), \
           F.linear(key, kv_weight[:, self.embed_dim:], bias[2*self.embed_dim:])

    if not static_kv:
        k = torch.cat([key_cache, k], dim=2)
        v = torch.cat([value_cache, v], dim=2)

    q = q.view(q.size(0), q.size(1), -1, self.head_dim).transpose(1, 2)
    k = k.view(k.size(0), k.size(1), -1, self.head_dim).transpose(1, 2)
    v = v.view(v.size(0), v.size(1), -1, self.head_dim).transpose(1, 2)

    attn_weights = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)

    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask

    if key_padding_mask is not None:
        attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, v)

    attn_output = attn_output.transpose(1, 2).contiguous().view(attn_output.size(0), attn_output.size(2), -1)
    attn_output = self.out_proj(attn_output)

    if use_cache:
        if static_kv:
            present_key = key_cache
            present_value = value_cache
        else:
            present_key = k
            present_value = v
    else:
        present_key = present_value = None

    outputs = (attn_output,) + (present_key, present_value) + (attn_weights,) if output_attentions else (attn_output,) + (present_key, present_value)
    return outputs