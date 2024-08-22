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
# --------------------------------------------------------------------------
import unittest
from collections import OrderedDict

import numpy
import torch
import torch.nn.functional as F
from onnx import TensorProto, helper
from torch import nn

import onnxruntime

torch.manual_seed(42)
numpy.random.seed(42)

ORT_DTYPE = TensorProto.FLOAT16
NP_TYPE = numpy.float16 if ORT_DTYPE == TensorProto.FLOAT16 else numpy.float32
USE_QUANT = False
THRESHOLD = 3e-1


def value_string_of(numpy_array):
    arr = numpy_array.flatten()
    lines = ["f, ".join([str(v) for v in arr[i : min(i + 8, arr.size)]]) for i in range(0, arr.size, 8)]
    return "{\n    " + "f,\n    ".join(lines) + "f}"


def print_tensor(name, numpy_array):
    print(f"const std::vector<float> {name} = {value_string_of(numpy_array)};")


def quant_dequant(weights, quant_mode: bool = True):
    # use the test version `_symmetric_...` to get the non-interleaved weights
    type = torch.quint4x2 if quant_mode else torch.int8
    import tensorrt_llm

    quant_weights, processed_q_weight, torch_weight_scales = (
        torch.ops.trtllm._symmetric_quantize_last_axis_of_batched_matrix(weights.T.cpu().contiguous(), type)
    )

    # Unpack the int4s int int8s
    if quant_mode:
        upper = quant_weights >> 4
        lower = (quant_weights << 4) >> 4  # Arithmetic right shift sign extends
        quant_weights = torch.stack((lower, upper), dim=2).view(weights.T.shape)

    quant_weights = quant_weights.to(dtype=weights.dtype)
    result = torch.multiply(quant_weights, torch_weight_scales.unsqueeze(0)).T.contiguous()
    return torch_weight_scales.to(torch.float16), processed_q_weight, result.to(device=weights.device)


def create_moe_onnx_graph(
    num_rows,
    num_experts,
    hidden_size,
    inter_size,
    fc1_experts_weights,
    fc2_experts_weights,
    fc3_experts_weights,
    fc1_scales,
    fc2_scales,
    fc3_scales,
    topk,
):
    use_quant = USE_QUANT
    if use_quant:
        assert fc1_experts_weights.dtype == torch.int8
        assert fc2_experts_weights.dtype == torch.int8
        assert fc3_experts_weights.dtype == torch.int8
        assert fc1_scales is not None
        assert fc2_scales is not None
        assert fc3_scales is not None
        assert fc1_scales.dtype == torch.float16
        assert fc2_scales.dtype == torch.float16
        assert fc3_scales.dtype == torch.float16

    nodes = [
        helper.make_node(
            "MoE" if not use_quant else "QMoE8Bits",
            (
                [
                    "input",
                    "router_probs",
                    "fc1_experts_weights",
                    "",
                    "fc2_experts_weights",
                    "",
                    "fc3_experts_weights",
                ]
                if not use_quant
                else [
                    "input",
                    "router_probs",
                    "fc1_experts_weights",
                    "fc1_scales",
                    "",
                    "fc2_experts_weights",
                    "fc2_scales",
                    "",
                    "fc3_experts_weights",
                    "fc3_scales",
                    "",
                ]
            ),
            ["output"],
            "MoE_0",
            k=topk,
            normalize_routing_weights=0,
            use_sparse_mixer=1,
            activation_type="silu",
            domain="com.microsoft",
        ),
    ]

    feature_size_modifier = 1 if not use_quant else 1

    fc1_shape = [num_experts, hidden_size, inter_size // feature_size_modifier]
    fc2_shape = [num_experts, inter_size, hidden_size // feature_size_modifier]
    fc3_shape = [num_experts, hidden_size, inter_size // feature_size_modifier]

    torch_type = torch.float16 if ORT_DTYPE == TensorProto.FLOAT16 else torch.float32
    numpy_type = numpy.float16 if ORT_DTYPE == TensorProto.FLOAT16 else numpy.float32
    if use_quant:
        numpy_type = numpy.uint8

    initializers = [
        helper.make_tensor(
            "fc1_experts_weights",
            ORT_DTYPE if not use_quant else TensorProto.UINT8,
            fc1_shape,
            fc1_experts_weights.flatten().detach().numpy().astype(numpy_type).tolist(),
            raw=False,
        ),
        helper.make_tensor(
            "fc2_experts_weights",
            ORT_DTYPE if not use_quant else TensorProto.UINT8,
            fc2_shape,
            fc2_experts_weights.flatten().detach().numpy().astype(numpy_type).tolist(),
            raw=False,
        ),
        helper.make_tensor(
            "fc3_experts_weights",
            ORT_DTYPE if not use_quant else TensorProto.UINT8,
            fc3_shape,
            fc3_experts_weights.flatten().detach().numpy().astype(numpy_type).tolist(),
            raw=False,
        ),
    ]

    if use_quant:
        fc1_scale_shape = [num_experts, inter_size]
        fc2_scale_shape = [num_experts, hidden_size]
        fc3_scale_shape = [num_experts, inter_size]
        initializers.extend(
            [
                helper.make_tensor(
                    "fc1_scales",
                    ORT_DTYPE,
                    fc1_scale_shape,
                    fc1_scales.to(torch_type).flatten().tolist(),
                    raw=False,
                ),
                helper.make_tensor(
                    "fc2_scales",
                    ORT_DTYPE,
                    fc2_scale_shape,
                    fc2_scales.to(torch_type).flatten().tolist(),
                    raw=False,
                ),
                helper.make_tensor(
                    "fc3_scales",
                    ORT_DTYPE,
                    fc3_scale_shape,
                    fc3_scales.to(torch_type).flatten().tolist(),
                    raw=False,
                ),
            ]
        )

    graph_inputs = [
        helper.make_tensor_value_info("input", ORT_DTYPE, [num_rows, hidden_size]),
    ]

    graph_inputs.append(
        helper.make_tensor_value_info(
            "router_probs",
            ORT_DTYPE,
            [num_rows, num_experts],
        )
    )

    graph_outputs = [
        helper.make_tensor_value_info("output", ORT_DTYPE, [num_rows, hidden_size]),
    ]

    graph = helper.make_graph(
        nodes,
        "MoE_Graph",
        graph_inputs,
        graph_outputs,
        initializers,
    )

    model = helper.make_model(graph)
    return model.SerializeToString()


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {
    "silu": nn.SiLU,
}
ACT2FN = ClassInstantier(ACT2CLS)


class PhiMoEConfig:
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=1e6,
        sliding_window=None,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        expert_dropout=0.0,
        num_experts_per_tok=2,
        num_local_experts=8,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        router_jitter_noise=0.01,
        input_jitter_noise=0.01,
        attention_bias=False,
        lm_head_bias=False,
        drop_reg=0.0,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.expert_dropout = expert_dropout
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise
        self.input_jitter_noise = input_jitter_noise
        self.attention_bias = attention_bias
        self.lm_head_bias = lm_head_bias
        self.drop_reg = drop_reg


class PhiMoEBlockSparseTop2MLP(nn.Module):
    def __init__(self, config: PhiMoEConfig):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.expert_dropout = nn.Dropout(config.expert_dropout)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.expert_dropout(current_hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


def masked_sampling_omp_inference(scores, top_k, jitter_eps, training):
    assert top_k == 2
    assert training == False

    mask_logits_threshold, selected_experts = torch.topk(scores, 2)

    mask_logits_threshold_1 = mask_logits_threshold[:, 0].unsqueeze(-1)

    factor = scores.abs().clamp(min=mask_logits_threshold_1)
    logits_mask = ((mask_logits_threshold_1 - scores) / factor) > (2 * jitter_eps)

    multiplier_1 = torch.softmax(scores.masked_fill(logits_mask, float("-inf")), dim=-1).gather(
        dim=-1, index=selected_experts[:, 0].unsqueeze(-1)
    )

    ################ second expert gating ################

    mask_logits_threshold_2 = mask_logits_threshold[:, 1].unsqueeze(-1)

    factor = scores.abs().clamp(min=mask_logits_threshold_2)
    logits_mask = ((mask_logits_threshold_2 - scores) / factor) > (2 * jitter_eps)

    multiplier_2 = torch.softmax(
        torch.scatter(scores, -1, selected_experts[:, 0].unsqueeze(-1), float("-inf")).masked_fill(
            logits_mask, float("-inf")
        ),
        dim=-1,
    ).gather(dim=-1, index=selected_experts[:, 1].unsqueeze(-1))

    multiplier = torch.concat((multiplier_1, multiplier_2), dim=-1)

    return (
        multiplier,
        selected_experts,
    )


class PhiMoESparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accommodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config, batch_size, sequence_length):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.router_jitter_noise = config.router_jitter_noise

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList([PhiMoEBlockSparseTop2MLP(config) for _ in range(self.num_experts)])

        w1_list = []
        w2_list = []
        w3_list = []
        w1_scale_list = []
        w2_scale_list = []
        w3_scale_list = []
        if not USE_QUANT:
            for i in range(self.num_experts):
                w1_list.append(self.experts[i].w1.weight)
                w2_list.append(self.experts[i].w2.weight)
                w3_list.append(self.experts[i].w3.weight)
        else:
            for i in range(self.num_experts):
                w1_scale, pre_qweight1, w1_qdq = quant_dequant(self.experts[i].w1.weight, False)
                w2_scale, pre_qweight2, w2_qdq = quant_dequant(self.experts[i].w2.weight, False)
                w3_scale, pre_qweight3, w3_qdq = quant_dequant(self.experts[i].w3.weight, False)

                self.experts[i].w1.weight.data = w1_qdq
                self.experts[i].w2.weight.data = w2_qdq
                self.experts[i].w3.weight.data = w3_qdq

                w1_list.append(pre_qweight1)
                w2_list.append(pre_qweight2)
                w3_list.append(pre_qweight3)
                w1_scale_list.append(w1_scale)
                w2_scale_list.append(w2_scale)
                w3_scale_list.append(w3_scale)

        self.moe_experts_weight1 = torch.stack(w1_list, dim=0)
        self.moe_experts_weight2 = torch.stack(w2_list, dim=0)
        self.moe_experts_weight3 = torch.stack(w3_list, dim=0)

        moe_experts_weight_scale1 = torch.stack(w1_scale_list, dim=0) if USE_QUANT else None
        moe_experts_weight_scale2 = torch.stack(w2_scale_list, dim=0) if USE_QUANT else None
        moe_experts_weight_scale3 = torch.stack(w3_scale_list, dim=0) if USE_QUANT else None

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.moe_onnx_graph = create_moe_onnx_graph(
            self.batch_size * self.sequence_length,
            self.num_experts,
            self.hidden_dim,
            self.ffn_dim,
            self.moe_experts_weight1,
            self.moe_experts_weight2,
            self.moe_experts_weight3,
            moe_experts_weight_scale1,
            moe_experts_weight_scale2,
            moe_experts_weight_scale3,
            self.top_k,
        )

        self.ort_sess = self.create_ort_session()

    def create_ort_session(self):
        from onnxruntime import InferenceSession, SessionOptions

        sess_options = SessionOptions()

        cuda_providers = ["CUDAExecutionProvider"]
        if cuda_providers[0] not in onnxruntime.get_available_providers():
            return None

        sess_options.log_severity_level = 2
        ort_session = InferenceSession(self.moe_onnx_graph, sess_options, providers=["CUDAExecutionProvider"])

        return ort_session

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights, selected_experts = masked_sampling_omp_inference(
            router_logits,
            top_k=self.top_k,
            jitter_eps=self.router_jitter_noise,
            training=False,
        )

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        return final_hidden_states  # , router_logits

    def ort_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        ort_inputs = {
            "input": numpy.ascontiguousarray(hidden_states.detach().numpy().astype(NP_TYPE)),
            "router_probs": numpy.ascontiguousarray(router_logits.detach().numpy().astype(NP_TYPE)),
        }

        ort_output = None
        if self.ort_sess is not None:
            ort_output = self.ort_sess.run(None, ort_inputs)
            return torch.tensor(ort_output).reshape(batch_size, sequence_length, -1)  # , router_logits

        # print_tensor("input", ort_inputs["input"])
        # print_tensor("router_probs", ort_inputs["router_probs"])
        # print_tensor("fc1_experts_weights", self.moe_experts_weight1.detach().numpy())
        # print_tensor("fc2_experts_weights", self.moe_experts_weight2.detach().numpy())
        # print_tensor("fc3_experts_weights", self.moe_experts_weight3.detach().numpy())
        # print_tensor("output", ort_output[0])

        return None

    def parity_check(self):
        hidden_state = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim)
        torch_output = self.forward(hidden_state)
        ort_output = self.ort_forward(hidden_state)
        if ort_output is not None:
            assert torch.allclose(torch_output, ort_output.to(torch.float32), rtol=THRESHOLD, atol=THRESHOLD)
            print(
                "batch_size:",
                self.batch_size,
                " sequence_length:",
                self.sequence_length,
                " max_diff:",
                (torch_output - ort_output).abs().max(),
                " parity: OK",
            )


class TestMixtralMoE(unittest.TestCase):
    def test_phi3_moe_parity(self):
        for batch_size in [1, 16]:
            for sequence_length in [128, 512]:
                # use a small sizes to speed up the test
                config = PhiMoEConfig(hidden_size=512, intermediate_size=1024)
                phi3_moe = PhiMoESparseMoeBlock(config, batch_size, sequence_length)
                phi3_moe.parity_check()


if __name__ == "__main__":
    unittest.main()
