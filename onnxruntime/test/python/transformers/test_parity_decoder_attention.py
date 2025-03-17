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

import numpy
import torch
from torch import Tensor, nn
from torch.nn import functional as F

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
    kv_sequence_length = 0
    num_heads = 0
    head_size = 0
    embed_dim = 0

    def __init__(self, b, s, s2, n, h):
        self.batch_size = b
        self.sequence_length = s
        self.kv_sequence_length = s2
        self.num_heads = n
        self.head_size = h
        self.embed_dim = self.num_heads * self.head_size


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
        use_past=torch.tensor(False),  # noqa: B008
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
        self.scaling = self.head_dim**-0.5

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
        output_attentions: bool = False,
        use_past=torch.tensor(False),  # noqa: B008
        has_key_padding_mask: bool = False,
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
            cached_shape = (
                bsz,
                self.num_heads,
                -1,
                self.head_dim,
            )  # bsz must be first for reorder_cache
            if static_kv:
                # cross-attn
                new_key_cache = k.view(*cached_shape)
                new_value_cache = v.view(*cached_shape)
            else:
                # self-attn
                new_key_cache = k.view(*cached_shape)
                new_value_cache = v.view(*cached_shape)

        src_len = k.size(1)
        assert key_padding_mask is None or key_padding_mask.shape == (bsz, src_len)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)

        if has_key_padding_mask:  # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(reshaped, float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = attn_weights

        assert v is not None
        attn_output = torch.bmm(attn_probs, v)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, new_key_cache, new_value_cache

    def ort_forward(
        self,
        query,
        key: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        layer_state: Optional[List[Tensor]] = None,
        attn_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
        use_past=torch.tensor(False),  # noqa: B008
        has_key_padding_mask: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""
        # For readability
        static_kv = bool(self.encoder_decoder_attention)
        has_layer_state = layer_state is not None
        use_past_cache = bool(use_past)

        q_weight = self.q_proj.weight.transpose(0, 1)
        q_weight = q_weight.reshape(self.embed_dim, self.embed_dim)

        kv_weight = torch.stack(
            (
                self.k_v_proj.k_proj.weight.transpose(0, 1),
                self.k_v_proj.v_proj.weight.transpose(0, 1),
            ),
            dim=1,
        )
        kv_weight = kv_weight.reshape(self.embed_dim, 2 * self.embed_dim)

        bias = torch.stack(
            (self.q_proj.bias, self.k_v_proj.k_proj.bias, self.k_v_proj.v_proj.bias),
            dim=0,
        )
        bias = bias.reshape(3 * self.embed_dim)

        onnx_model_str = create_decoder_attention_graph(
            query,
            key,
            q_weight,
            kv_weight,
            bias,
            self.num_heads,
            static_kv,
            use_past_cache,
            has_layer_state,
            has_key_padding_mask,
        )

        self_p_k, self_p_v, enc_dec_p_k, enc_dec_p_v = layer_state
        if self.encoder_decoder_attention:
            key_cache, value_cache = enc_dec_p_k, enc_dec_p_v
        else:
            key_cache, value_cache = self_p_k, self_p_v

        ort_inputs = {
            "query": numpy.ascontiguousarray(query.cpu().numpy()),
            "key": numpy.ascontiguousarray(key.cpu().numpy()),
            "key_padding_mask": numpy.ascontiguousarray(key_padding_mask.cpu().numpy()),
            "key_cache": numpy.ascontiguousarray(key_cache.detach().cpu().numpy()),
            "value_cache": numpy.ascontiguousarray(value_cache.detach().cpu().numpy()),
        }

        from onnxruntime import InferenceSession, SessionOptions

        sess_options = SessionOptions()
        ort_session = InferenceSession(onnx_model_str, sess_options, providers=["CUDAExecutionProvider"])
        ort_output = ort_session.run(None, ort_inputs)
        output, new_key_cache, new_value_cache = ort_output

        output = torch.tensor(output)
        attn_output = self.out_proj(output)

        return attn_output, torch.tensor(new_key_cache), torch.tensor(new_value_cache)


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


def create_inputs(
    config: Config,
    has_layer_state: bool,
    use_past: bool,
    encoder_decoder_attention: bool,
):
    query = torch.normal(
        mean=0.0,
        std=0.1,
        size=(config.sequence_length, config.batch_size, config.embed_dim),
    ).to(torch.float32)
    key = torch.normal(
        mean=0.0,
        std=0.1,
        size=(config.kv_sequence_length, config.batch_size, config.embed_dim),
    ).to(torch.float32)

    key_length = None
    if not has_layer_state or not use_past:
        if not encoder_decoder_attention:
            key_length = config.sequence_length
        else:
            key_length = config.kv_sequence_length
    else:
        if not encoder_decoder_attention:
            key_length = config.sequence_length + config.kv_sequence_length
        else:
            key_length = config.kv_sequence_length

    key_padding_mask = torch.normal(mean=0.0, std=0.1, size=(config.batch_size, key_length)) > 0
    # The following line ensure not all the mask are true
    key_padding_mask[0][0] = False

    cache = torch.normal(
        mean=0.0,
        std=0.1,
        size=(
            config.batch_size,
            config.num_heads,
            config.kv_sequence_length,
            config.head_size,
        ),
    ).to(torch.float32)
    layer_state = [cache, cache, cache, cache]

    return query, key, key_padding_mask, layer_state, torch.tensor(use_past)


def parity_check(
    config,
    has_layer_state,
    use_past,
    static_kv,
    has_key_padding_mask,
    rtol=1e-4,
    atol=1e-4,
):
    query, key, key_padding_mask, layer_state, use_past = create_inputs(config, has_layer_state, use_past, static_kv)
    attn = AttentionForONNX(config.embed_dim, config.num_heads, encoder_decoder_attention=static_kv)
    attn_output, new_key_cache, new_value_cache = attn.forward(
        query,
        key,
        key_padding_mask,
        layer_state,
        None,
        False,
        use_past,
        has_key_padding_mask,
    )
    attn_output_ort, new_key_cache_ort, new_value_cache_ort = attn.ort_forward(
        query,
        key,
        key_padding_mask,
        layer_state,
        None,
        False,
        use_past,
        has_key_padding_mask,
    )
    attn_output_ort_1, _, _ = attn.ort_forward(
        query,
        key,
        key_padding_mask,
        layer_state,
        None,
        False,
        use_past,
        has_key_padding_mask,
    )
    print(
        " B:",
        config.batch_size,
        " S:",
        config.sequence_length,
        " S*:",
        config.kv_sequence_length,
        " h:",
        config.embed_dim,
        " has_layer_state:",
        has_layer_state,
        " use_past:",
        use_past,
        " static_kv:",
        static_kv,
        " has_key_padding_mask:",
        has_key_padding_mask,
        "[attn_output, randomness, key, value] parity:",
        numpy.allclose(
            attn_output.detach().numpy(),
            attn_output_ort.detach().numpy(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        ),
        numpy.allclose(
            attn_output_ort_1.detach().numpy(),
            attn_output_ort.detach().numpy(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        ),
        numpy.allclose(
            new_key_cache.detach().numpy(),
            new_key_cache_ort.detach().numpy(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        ),
        numpy.allclose(
            new_value_cache.detach().numpy(),
            new_value_cache_ort.detach().numpy(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        ),
    )


if __name__ == "__main__":
    for b in [1, 32, 128]:
        for s in [1, 2, 128]:
            for s2 in [1, 64, 256]:
                for n in [8]:
                    for h in [64]:
                        config = Config(b, s, s2, n, h)
                        parity_check(
                            config,
                            has_layer_state=True,
                            use_past=True,
                            static_kv=True,
                            has_key_padding_mask=False,
                        )
                        parity_check(
                            config,
                            has_layer_state=True,
                            use_past=True,
                            static_kv=False,
                            has_key_padding_mask=False,
                        )
                        parity_check(
                            config,
                            has_layer_state=True,
                            use_past=False,
                            static_kv=True,
                            has_key_padding_mask=False,
                        )
                        parity_check(
                            config,
                            has_layer_state=True,
                            use_past=False,
                            static_kv=False,
                            has_key_padding_mask=False,
                        )
                        parity_check(
                            config,
                            has_layer_state=True,
                            use_past=True,
                            static_kv=True,
                            has_key_padding_mask=True,
                        )
                        parity_check(
                            config,
                            has_layer_state=True,
                            use_past=True,
                            static_kv=False,
                            has_key_padding_mask=True,
                        )
                        parity_check(
                            config,
                            has_layer_state=True,
                            use_past=False,
                            static_kv=True,
                            has_key_padding_mask=True,
                        )
                        parity_check(
                            config,
                            has_layer_state=True,
                            use_past=False,
                            static_kv=False,
                            has_key_padding_mask=True,
                        )
