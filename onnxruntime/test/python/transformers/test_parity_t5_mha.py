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

import unittest

import numpy as np
import torch
from torch import nn

torch.manual_seed(0)


def create_t5_mha_graph(
    batch_size,
    seq_len,
    kv_sequence_length,
    head_size,
    num_heads,
    use_past,
    is_static_kv,
):
    from onnx import TensorProto, helper

    use_present = not use_past
    if not is_static_kv and use_past:
        use_present = True
    use_rpb = not is_static_kv
    use_mask = not use_rpb

    past_sequence_length = kv_sequence_length
    total_sequence_length = kv_sequence_length if is_static_kv else seq_len

    if not is_static_kv:
        kv_sequence_length = seq_len

    if not is_static_kv and use_past:
        total_sequence_length += past_sequence_length

    rpb_length = total_sequence_length if use_past else seq_len

    nodes = [
        helper.make_node(
            "MultiHeadAttention",
            [
                "query",
                "key" if use_present or is_static_kv else "",
                "value" if use_present or is_static_kv else "",
                "",  # bias
                "key_padding_mask" if use_mask else "",
                "relative_position_bias" if use_rpb else "",
                "past_key" if use_past and not is_static_kv else "",
                "past_value" if use_past and not is_static_kv else "",
            ],
            [
                "output",
                "present_key" if use_present else "",
                "present_value" if use_present else "",
            ],
            "MHA_0",
            num_heads=num_heads,
            mask_filter_value=-10000.0,
            scale=1.0,
            domain="com.microsoft",
        ),
    ]

    initializers = []

    hidden_size = head_size * num_heads

    graph_inputs = [
        helper.make_tensor_value_info("query", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]),
    ]

    graph_outputs = [
        helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]),
    ]

    if use_mask:
        graph_inputs.append(
            helper.make_tensor_value_info("key_padding_mask", TensorProto.INT32, [batch_size, kv_sequence_length])
        )

    if use_rpb:
        graph_inputs.append(
            helper.make_tensor_value_info(
                "relative_position_bias", TensorProto.FLOAT, [1, num_heads, seq_len, rpb_length]
            )
        )

    if use_past and not is_static_kv:
        graph_inputs.append(
            helper.make_tensor_value_info(
                "past_key", TensorProto.FLOAT, [batch_size, num_heads, past_sequence_length, head_size]
            )
        )
        graph_inputs.append(
            helper.make_tensor_value_info(
                "past_value", TensorProto.FLOAT, [batch_size, num_heads, past_sequence_length, head_size]
            )
        )

    if use_present:
        graph_inputs.append(
            helper.make_tensor_value_info("key", TensorProto.FLOAT, [batch_size, kv_sequence_length, hidden_size])
        )
        graph_inputs.append(
            helper.make_tensor_value_info("value", TensorProto.FLOAT, [batch_size, kv_sequence_length, hidden_size])
        )
    elif is_static_kv and use_past:
        graph_inputs.append(
            helper.make_tensor_value_info(
                "key", TensorProto.FLOAT, [batch_size, num_heads, past_sequence_length, head_size]
            )
        )
        graph_inputs.append(
            helper.make_tensor_value_info(
                "value", TensorProto.FLOAT, [batch_size, num_heads, past_sequence_length, head_size]
            )
        )

    if use_present:
        graph_outputs.append(
            helper.make_tensor_value_info(
                "present_key", TensorProto.FLOAT, [batch_size, num_heads, total_sequence_length, head_size]
            )
        )
        graph_outputs.append(
            helper.make_tensor_value_info(
                "present_value", TensorProto.FLOAT, [batch_size, num_heads, total_sequence_length, head_size]
            )
        )

    graph = helper.make_graph(
        nodes,
        "T5_MHA_Graph",
        graph_inputs,
        graph_outputs,
        initializers,
    )

    model = helper.make_model(graph)
    return model.SerializeToString()


# For decoder only (not decoder_init) starting from second iteration
def create_t5_decoder_masked_mha_graph(
    batch_size,
    past_sequence_length,
    kv_sequence_length,
    head_size,
    num_heads,
    is_cross_attention,
):
    from onnx import TensorProto, helper

    nodes = [
        helper.make_node(
            "DecoderMaskedMultiHeadAttention",
            [
                "query",
                "key",
                "value",
                "mask_index" if is_cross_attention else "",
                "relative_position_bias" if not is_cross_attention else "",
                "past_key" if not is_cross_attention else "",
                "past_value" if not is_cross_attention else "",
                "past_sequence_length" if not is_cross_attention else "",
            ],
            [
                "output",
                "present_key" if not is_cross_attention else "",
                "present_value" if not is_cross_attention else "",
            ],
            "DMMHA_0",
            num_heads=num_heads,
            mask_filter_value=-10000.0,
            scale=1.0,
            past_present_share_buffer=0 if is_cross_attention else 1,
            domain="com.microsoft",
        ),
    ]

    initializers = []

    hidden_size = head_size * num_heads

    graph_inputs = [
        helper.make_tensor_value_info("query", TensorProto.FLOAT, [batch_size, 1, hidden_size]),
    ]

    graph_outputs = [
        helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch_size, 1, hidden_size]),
    ]

    if is_cross_attention:
        graph_inputs.append(
            helper.make_tensor_value_info("mask_index", TensorProto.INT32, [batch_size, kv_sequence_length])
        )
        graph_inputs.append(
            helper.make_tensor_value_info(
                "key", TensorProto.FLOAT, [batch_size, num_heads, kv_sequence_length, head_size]
            )
        )
        graph_inputs.append(
            helper.make_tensor_value_info(
                "value", TensorProto.FLOAT, [batch_size, num_heads, kv_sequence_length, head_size]
            )
        )
    else:
        graph_inputs.append(helper.make_tensor_value_info("key", TensorProto.FLOAT, [batch_size, 1, hidden_size]))
        graph_inputs.append(helper.make_tensor_value_info("value", TensorProto.FLOAT, [batch_size, 1, hidden_size]))
        graph_inputs.append(
            helper.make_tensor_value_info(
                "relative_position_bias", TensorProto.FLOAT, [1, num_heads, 1, past_sequence_length + 1]
            )
        )
        # use past_sequence_length + 1 to simulate max_sequence_length
        graph_inputs.append(
            helper.make_tensor_value_info(
                "past_key", TensorProto.FLOAT, [batch_size, num_heads, past_sequence_length + 1, head_size]
            )
        )
        graph_inputs.append(
            helper.make_tensor_value_info(
                "past_value", TensorProto.FLOAT, [batch_size, num_heads, past_sequence_length + 1, head_size]
            )
        )
        graph_inputs.append(helper.make_tensor_value_info("past_sequence_length", TensorProto.INT32, [1]))
        graph_outputs.append(
            helper.make_tensor_value_info(
                "present_key", TensorProto.FLOAT, [batch_size, num_heads, past_sequence_length + 1, head_size]
            )
        )
        graph_outputs.append(
            helper.make_tensor_value_info(
                "present_value", TensorProto.FLOAT, [batch_size, num_heads, past_sequence_length + 1, head_size]
            )
        )

    graph = helper.make_graph(
        nodes,
        "T5_DMMHA_Graph",
        graph_inputs,
        graph_outputs,
        initializers,
    )

    model = helper.make_model(graph)
    return model.SerializeToString()


class T5Config:
    def __init__(self, is_decoder, batch_size, seq_len, kv_sequence_length, num_heads, head_size, use_past):
        self.is_decoder = is_decoder
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
        self.use_past = use_past


class T5Attention(nn.Module):
    def __init__(self, config: T5Config, is_static_kv, use_decoder_masked_kernel: bool = False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.is_static_kv = is_static_kv
        self.has_relative_attention_bias = not self.is_static_kv
        self.d_model = config.d_model
        self.key_value_proj_dim = config.head_size
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        # self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        self.pruned_heads = set()

        # ORT parameters
        self.batch_size = config.batch_size
        self.seq_len = config.seq_len
        self.kv_sequence_length = config.kv_sequence_length
        self.head_size = config.head_size
        self.num_heads = config.num_heads
        self.hidden_size = self.d_model
        self.use_past = config.use_past
        self.use_decoder_masked_kernel = use_decoder_masked_kernel

        # Create onnx graph
        if self.use_decoder_masked_kernel:
            self.onnx_graph = create_t5_decoder_masked_mha_graph(
                self.batch_size,
                self.kv_sequence_length,
                self.kv_sequence_length,
                self.head_size,
                self.num_heads,
                is_static_kv,
            )
        else:
            self.onnx_graph = create_t5_mha_graph(
                self.batch_size,
                self.seq_len,
                self.kv_sequence_length,
                self.head_size,
                self.num_heads,
                self.use_past,
                is_static_kv,
            )

    # Reorder 'K' from [B, N, S, H] to [B, N, H/4, S, 4]
    def reorder_key_cache(self, key_cache, batch_size, num_heads, sequence_length, head_size, max_sequence_length):
        ordered = np.zeros_like(key_cache)

        # assume float
        num_inner_elements = 4
        chunks = int(head_size / num_inner_elements)

        for b in range(batch_size):
            for h in range(num_heads):
                for c in range(chunks):
                    for s in range(sequence_length):
                        base_offset = (b * num_heads * max_sequence_length * head_size) + (
                            h * max_sequence_length * head_size
                        )
                        input_base_offset = base_offset + (s * head_size) + (c * num_inner_elements)
                        output_base_offset = (
                            base_offset + (c * max_sequence_length * num_inner_elements) + (s * num_inner_elements)
                        )
                        for e in range(num_inner_elements):
                            ordered[output_base_offset + e] = key_cache[input_base_offset + e]

        return ordered

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
        position_bias_length = self.seq_len if not self.use_past else self.kv_sequence_length + self.seq_len
        position_bias = torch.normal(
            mean=0.5, std=0.1, size=(1, self.num_heads, position_bias_length, position_bias_length)
        ).to(torch.float32)
        if self.use_decoder_masked_kernel:
            position_bias = torch.normal(mean=5, std=0.1, size=(1, self.num_heads, 1, position_bias_length)).to(
                torch.float32
            )
        return hidden_states, key_value_states, past_key_value, attention_mask, position_bias

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
        if past_key_value is not None and position_bias is not None:
            position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

        if mask is not None:
            # Adjust onnx mask shape
            mask = (1 - mask.unsqueeze(1).unsqueeze(2)) * torch.finfo(torch.float32).min
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
        # attn_output = self.o(attn_output) # ORT places this matmul outside of MHA op

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output, present_key_value_state)

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
        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        cuda_providers = ["CUDAExecutionProvider"]
        if cuda_providers[0] not in onnxruntime.get_available_providers():
            return None
        ort_session = onnxruntime.InferenceSession(self.onnx_graph, sess_options, providers=cuda_providers)

        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            if key_value_states is None:
                # self-attn
                hidden_states = proj_layer(hidden_states)
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = proj_layer(key_value_states)

            return hidden_states

        # get query states
        query_states = self.q(hidden_states)  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        if past_key_value is not None and position_bias is not None:
            position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

        torch_key_padding_mask = mask.to(torch.int32) if mask is not None else None
        torch_position_bias = position_bias if position_bias is not None else None

        ort_inputs = None
        if past_key_value is None:
            ort_inputs = {
                "query": np.ascontiguousarray(query_states.detach().numpy()),
                "key": np.ascontiguousarray(key_states.detach().numpy()),
                "value": np.ascontiguousarray(value_states.detach().numpy()),
            }
            if torch_key_padding_mask is not None:
                ort_inputs["key_padding_mask"] = np.ascontiguousarray(torch_key_padding_mask.detach().numpy())
            if torch_position_bias is not None:
                ort_inputs["relative_position_bias"] = np.ascontiguousarray(torch_position_bias.detach().numpy())
        else:
            torch_past_key = past_key_value[0]
            torch_past_value = past_key_value[1]
            ort_inputs = {
                "query": np.ascontiguousarray(query_states.detach().numpy()),
            }
            torch_past_key = np.ascontiguousarray(torch_past_key.detach().numpy())
            torch_past_value = np.ascontiguousarray(torch_past_value.detach().numpy())
            max_seq_len = torch_past_key.shape[2] + 1
            torch_past_key_padded = np.zeros(
                [torch_past_key.shape[0], torch_past_key.shape[1], max_seq_len, torch_past_key.shape[3]],
                dtype=np.float32,
            )
            torch_past_value_padded = np.zeros(
                [torch_past_value.shape[0], torch_past_value.shape[1], max_seq_len, torch_past_value.shape[3]],
                dtype=np.float32,
            )
            torch_past_key_padded[:, :, : torch_past_key.shape[2], :] = torch_past_key
            torch_past_value_padded[:, :, : torch_past_value.shape[2], :] = torch_past_value
            if self.is_static_kv:
                if self.use_decoder_masked_kernel:
                    reordered_past_key = self.reorder_key_cache(
                        torch_past_key.flatten(),
                        batch_size=batch_size,
                        num_heads=self.num_heads,
                        sequence_length=self.kv_sequence_length,
                        head_size=self.head_size,
                        max_sequence_length=self.kv_sequence_length,
                    )
                    ort_inputs["key"] = reordered_past_key.reshape(torch_past_key.shape)
                    ort_inputs["value"] = torch_past_value
                else:
                    ort_inputs["key"] = np.ascontiguousarray(torch_past_key)
                    ort_inputs["value"] = np.ascontiguousarray(torch_past_value)
            else:
                ort_inputs["key"] = np.ascontiguousarray(key_states.detach().numpy())
                ort_inputs["value"] = np.ascontiguousarray(value_states.detach().numpy())
                if self.use_decoder_masked_kernel:
                    reordered_past_key = self.reorder_key_cache(
                        torch_past_key_padded.flatten(),
                        batch_size=batch_size,
                        num_heads=self.num_heads,
                        sequence_length=self.kv_sequence_length,
                        head_size=self.head_size,
                        max_sequence_length=max_seq_len,
                    )
                    ort_inputs["past_key"] = reordered_past_key.reshape(torch_past_value_padded.shape)
                    ort_inputs["past_value"] = torch_past_value_padded
                    ort_inputs["past_sequence_length"] = np.array([self.kv_sequence_length], dtype=np.int32)
                else:
                    ort_inputs["past_key"] = torch_past_key
                    ort_inputs["past_value"] = torch_past_value
            if torch_key_padding_mask is not None:
                if self.use_decoder_masked_kernel:
                    ort_inputs["mask_index"] = np.ascontiguousarray(torch_key_padding_mask.detach().numpy())
                else:
                    ort_inputs["key_padding_mask"] = np.ascontiguousarray(torch_key_padding_mask.detach().numpy())
            if torch_position_bias is not None:
                ort_inputs["relative_position_bias"] = np.ascontiguousarray(torch_position_bias.detach().numpy())

        ort_output = ort_session.run(None, ort_inputs)

        output = None
        if past_key_value is not None and self.is_static_kv:
            output = torch.tensor(ort_output)
        else:
            output = (torch.tensor(ort_output[0]), (torch.tensor(ort_output[1]), torch.tensor(ort_output[2])))

        return output


def compare_t5_cross_attention_decoder(batch_size, seq_len, num_heads, head_size, kv_sequence_length, use_dmmha=False):
    config = T5Config(
        is_decoder=True,
        batch_size=batch_size,
        seq_len=seq_len,
        kv_sequence_length=kv_sequence_length,
        num_heads=num_heads,
        head_size=head_size,
        use_past=True,
    )

    T5CrossAttention = T5Attention(config, is_static_kv=True, use_decoder_masked_kernel=use_dmmha)  # noqa: N806

    hidden_states, key_value_states, past_key_value, attention_mask, _ = T5CrossAttention.create_inputs()
    torch_output = T5CrossAttention.torch_forward(
        hidden_states, key_value_states, past_key_value, attention_mask, position_bias=None, use_cache=False
    )
    ort_output = T5CrossAttention.ort_forward(
        hidden_states, key_value_states, past_key_value, attention_mask, position_bias=None, use_cache=False
    )

    if ort_output is not None:
        assert torch.allclose(torch_output[0], ort_output[0], atol=1e-4)


def compare_t5_cross_attention_decoder_init(batch_size, seq_len, num_heads, head_size, kv_sequence_length):
    config = T5Config(
        is_decoder=True,
        batch_size=batch_size,
        seq_len=seq_len,
        kv_sequence_length=kv_sequence_length,
        num_heads=num_heads,
        head_size=head_size,
        use_past=False,
    )
    T5CrossAttention = T5Attention(config, is_static_kv=True)  # noqa: N806

    hidden_states, key_value_states, _, attention_mask, _ = T5CrossAttention.create_inputs()
    torch_output = T5CrossAttention.torch_forward(
        hidden_states, key_value_states, None, attention_mask, position_bias=None, use_cache=True
    )
    ort_output = T5CrossAttention.ort_forward(
        hidden_states, key_value_states, None, attention_mask, position_bias=None, use_cache=True
    )

    if ort_output is not None:
        assert torch.allclose(torch_output[0], ort_output[0], atol=1e-4)
        assert torch.allclose(torch_output[1][0], ort_output[1][0], atol=1e-4)
        assert torch.allclose(torch_output[1][1], ort_output[1][1], atol=1e-4)


def compare_t5_self_attention_decoder_init(batch_size, seq_len, num_heads, head_size, kv_sequence_length):
    config = T5Config(
        is_decoder=True,
        batch_size=batch_size,
        seq_len=seq_len,
        kv_sequence_length=kv_sequence_length,
        num_heads=num_heads,
        head_size=head_size,
        use_past=False,
    )
    T5CrossAttention = T5Attention(config, is_static_kv=False)  # noqa: N806

    hidden_states, _, _, _, position_bias = T5CrossAttention.create_inputs()
    torch_output = T5CrossAttention.torch_forward(
        hidden_states, None, None, mask=None, position_bias=position_bias, use_cache=True
    )
    ort_output = T5CrossAttention.ort_forward(
        hidden_states, None, None, mask=None, position_bias=position_bias, use_cache=True
    )

    if ort_output is not None:
        assert torch.allclose(torch_output[0], ort_output[0], atol=1e-4)
        assert torch.allclose(torch_output[1][0], ort_output[1][0], atol=1e-4)
        assert torch.allclose(torch_output[1][1], ort_output[1][1], atol=1e-4)


def compare_t5_self_attention_decoder(batch_size, seq_len, num_heads, head_size, kv_sequence_length, use_dmmha=False):
    config = T5Config(
        is_decoder=True,
        batch_size=batch_size,
        seq_len=seq_len,
        kv_sequence_length=kv_sequence_length,
        num_heads=num_heads,
        head_size=head_size,
        use_past=True,
    )

    T5CrossAttention = T5Attention(config, is_static_kv=False, use_decoder_masked_kernel=use_dmmha)  # noqa: N806

    hidden_states, _, past_key_value, _, position_bias = T5CrossAttention.create_inputs()
    torch_output = T5CrossAttention.torch_forward(
        hidden_states, None, past_key_value, mask=None, position_bias=position_bias, use_cache=True
    )
    ort_output = T5CrossAttention.ort_forward(
        hidden_states, None, past_key_value, mask=None, position_bias=position_bias, use_cache=True
    )

    if ort_output is not None:
        assert torch.allclose(torch_output[0], ort_output[0], atol=1e-4)
        if not use_dmmha:
            assert torch.allclose(torch_output[1][0], ort_output[1][0], atol=1e-4)
            assert torch.allclose(torch_output[1][1], ort_output[1][1], atol=1e-4)


class TestT5MHAParity(unittest.TestCase):
    def setUp(self):
        self.batch_size = 5
        self.seq_len = 2
        self.num_heads = 2
        self.head_size = 4
        self.kv_sequence_length = 3

    def test_t5_cross_attention_decoder_init(self):
        compare_t5_cross_attention_decoder_init(
            self.batch_size, self.seq_len, self.num_heads, self.head_size, self.kv_sequence_length
        )

    def test_t5_self_attention_decoder_init(self):
        compare_t5_self_attention_decoder_init(
            self.batch_size, self.seq_len, self.num_heads, self.head_size, self.kv_sequence_length
        )

    def test_t5_cross_attention_decoder(self):
        compare_t5_cross_attention_decoder(
            self.batch_size, self.seq_len, self.num_heads, self.head_size, self.kv_sequence_length
        )

    def test_t5_self_attention_decoder(self):
        compare_t5_self_attention_decoder(
            self.batch_size, self.seq_len, self.num_heads, self.head_size, self.kv_sequence_length
        )

    def test_t5_cross_attention_decoder_masked_mha(self):
        batch_size = 2
        seq_len = 1
        num_heads = 2
        head_size = 32
        kv_sequence_length = 2
        compare_t5_cross_attention_decoder(
            batch_size, seq_len, num_heads, head_size, kv_sequence_length, use_dmmha=True
        )

    def test_t5_self_attention_decoder_masked_mha(self):
        batch_size = 2
        seq_len = 1
        num_heads = 2
        head_size = 32
        kv_sequence_length = 2
        compare_t5_self_attention_decoder(batch_size, seq_len, num_heads, head_size, kv_sequence_length, use_dmmha=True)


if __name__ == "__main__":
    unittest.main()
