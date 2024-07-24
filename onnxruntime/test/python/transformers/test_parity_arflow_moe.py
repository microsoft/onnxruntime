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

ORT_DTYPE = TensorProto.FLOAT
NP_TYPE = numpy.float16 if ORT_DTYPE == TensorProto.FLOAT16 else numpy.float32
THRESHOLD = 3e-2


def value_string_of(numpy_array):
    arr = numpy_array.flatten()
    lines = ["f, ".join([str(v) for v in arr[i : min(i + 8, arr.size)]]) for i in range(0, arr.size, 8)]
    return "{\n    " + "f,\n    ".join(lines) + "f}"


def print_tensor(name, numpy_array):
    print(f"const std::vector<float> {name} = {value_string_of(numpy_array)};")


def create_moe_onnx_graph(
    num_rows,
    num_experts,
    in_features,
    interm_features,
    out_features,
    fc1_experts_weights,
    fc1_experts_bias,
    fc2_experts_weights,
    fc2_experts_bias,
    fc3_experts_weights,
    fc3_experts_bias,
    fc4_experts_weights,
    fc4_experts_bias,
    topk,
):
    nodes = [
        helper.make_node(
            "ArflowMoE",
            [
                "input",
                "router_probs",
                "fc1_experts_weights",
                "fc1_experts_bias",
                "fc2_experts_weights",
                "fc2_experts_bias",
                "fc3_experts_weights",
                "fc3_experts_bias",
                "fc4_experts_weights",
                "fc4_experts_bias",
            ],
            ["output"],
            "MoE_0",
            k=topk,
            normalize_routing_weights=1,
            # activation_type="silu",
            domain="com.microsoft",
        ),
    ]

    fc1_shape = [num_experts, in_features, interm_features]
    fc2_shape = [num_experts, interm_features, interm_features]
    fc3_shape = [num_experts, interm_features, interm_features]
    fc4_shape = [num_experts, interm_features, out_features]

    b1_shape = [num_experts, interm_features]
    b2_shape = [num_experts, interm_features]
    b3_shape = [num_experts, interm_features]
    b4_shape = [num_experts, out_features]

    torch_type = torch.float16 if ORT_DTYPE == TensorProto.FLOAT16 else torch.float32

    initializers = [
        helper.make_tensor(
            "fc1_experts_weights",
            ORT_DTYPE,
            fc1_shape,
            fc1_experts_weights.to(torch_type).flatten().tolist(),
            raw=False,
        ),
        helper.make_tensor(
            "fc2_experts_weights",
            ORT_DTYPE,
            fc2_shape,
            fc2_experts_weights.to(torch_type).flatten().tolist(),
            raw=False,
        ),
        helper.make_tensor(
            "fc3_experts_weights",
            ORT_DTYPE,
            fc3_shape,
            fc3_experts_weights.to(torch_type).flatten().tolist(),
            raw=False,
        ),
        helper.make_tensor(
            "fc4_experts_weights",
            ORT_DTYPE,
            fc4_shape,
            fc4_experts_weights.to(torch_type).flatten().tolist(),
        ),
        helper.make_tensor(
            "fc1_experts_bias",
            ORT_DTYPE,
            b1_shape,
            fc1_experts_bias.to(torch_type).flatten().tolist(),
            raw=False,
        ),
        helper.make_tensor(
            "fc2_experts_bias",
            ORT_DTYPE,
            b2_shape,
            fc2_experts_bias.to(torch_type).flatten().tolist(),
            raw=False,
        ),
        helper.make_tensor(
            "fc3_experts_bias",
            ORT_DTYPE,
            b3_shape,
            fc3_experts_bias.to(torch_type).flatten().tolist(),
            raw=False,
        ),
        helper.make_tensor(
            "fc4_experts_bias",
            ORT_DTYPE,
            b4_shape,
            fc4_experts_bias.to(torch_type).flatten().tolist(),
            raw=False,
        ),
    ]

    graph_inputs = [
        helper.make_tensor_value_info("input", ORT_DTYPE, [num_rows, in_features]),
    ]

    graph_inputs.append(
        helper.make_tensor_value_info(
            "router_probs",
            ORT_DTYPE,
            [num_rows, num_experts],
        )
    )

    graph_outputs = [
        helper.make_tensor_value_info("output", ORT_DTYPE, [num_rows, out_features]),
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


class ArflowConfig:
    def __init__(
        self,
        in_features=12336,
        interm_features=1024,
        out_features=96,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        rope_theta=1e6,
        attention_dropout=0.0,
        num_experts_per_tok=2,
        num_local_experts=8,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
    ):
        self.in_features = in_features
        self.interm_features = interm_features
        self.out_features = out_features
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef


class ArflowBlockSparseTop2MLP(nn.Module):
    def __init__(self, config: ArflowConfig):
        super().__init__()

        fc_model_list = []
        fc_model_list.append(nn.Linear(config.in_features, config.interm_features))
        fc_model_list.append(nn.ELU(True))
        fc_model_list.append(nn.Linear(config.interm_features, config.interm_features))
        fc_model_list.append(nn.ELU(True))
        fc_model_list.append(nn.Linear(config.interm_features, config.interm_features))
        fc_model_list.append(nn.ELU(True))
        fc_model_list.append(nn.Linear(config.interm_features, config.out_features))

        self.fc = nn.Sequential(*fc_model_list)

    def forward(self, hidden_states):
        return self.fc(hidden_states)


class ArflowSparseMoeBlock(nn.Module):
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
        self.in_features = config.in_features
        self.interm_features = config.interm_features
        self.out_features = config.out_features
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # gating
        self.gate = nn.Linear(self.in_features, self.num_experts, bias=False)

        self.experts = nn.ModuleList([ArflowBlockSparseTop2MLP(config) for _ in range(self.num_experts)])

        w1_list = []
        w2_list = []
        w3_list = []
        w4_list = []
        b1_list = []
        b2_list = []
        b3_list = []
        b4_list = []
        for i in range(self.num_experts):
            w1_list.append(self.experts[i].fc[0].weight)
            w2_list.append(self.experts[i].fc[2].weight)
            w3_list.append(self.experts[i].fc[4].weight)
            w4_list.append(self.experts[i].fc[6].weight)
            b1_list.append(self.experts[i].fc[0].bias)
            b2_list.append(self.experts[i].fc[2].bias)
            b3_list.append(self.experts[i].fc[4].bias)
            b4_list.append(self.experts[i].fc[6].bias)

        self.moe_experts_weight1 = torch.stack(w1_list, dim=0)
        self.moe_experts_weight2 = torch.stack(w2_list, dim=0)
        self.moe_experts_weight3 = torch.stack(w3_list, dim=0)
        self.moe_experts_weight4 = torch.stack(w4_list, dim=0)

        self.moe_experts_bias1 = torch.stack(b1_list, dim=0)
        self.moe_experts_bias2 = torch.stack(b2_list, dim=0)
        self.moe_experts_bias3 = torch.stack(b3_list, dim=0)
        self.moe_experts_bias4 = torch.stack(b4_list, dim=0)

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.moe_onnx_graph = create_moe_onnx_graph(
            self.batch_size * self.sequence_length,
            self.num_experts,
            self.in_features,
            self.interm_features,
            self.out_features,
            self.moe_experts_weight1,
            self.moe_experts_bias1,
            self.moe_experts_weight2,
            self.moe_experts_bias2,
            self.moe_experts_weight3,
            self.moe_experts_bias3,
            self.moe_experts_weight4,
            self.moe_experts_bias4,
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
        batch_size, sequence_length, in_features = hidden_states.shape
        hidden_states = hidden_states.view(-1, in_features)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, self.out_features), dtype=hidden_states.dtype, device=hidden_states.device
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

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, in_features)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, self.out_features)
        return final_hidden_states  # , router_logits

    def ort_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, in_features = hidden_states.shape
        hidden_states = hidden_states.view(-1, in_features)
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
        hidden_state = torch.randn(self.batch_size, self.sequence_length, self.in_features)
        torch_output = self.forward(hidden_state)
        ort_output = self.ort_forward(hidden_state)
        if ort_output is not None:
            # print("torch_output", torch_output)
            # print("ort_output", ort_output)
            assert torch.allclose(torch_output, ort_output, rtol=1e-04, atol=1e-04)
            print(
                "batch_size:",
                self.batch_size,
                " sequence_length:",
                self.sequence_length,
                " max_diff:",
                (torch_output - ort_output).abs().max(),
                " parity: OK",
            )


class TestArflowMoE(unittest.TestCase):
    def test_Arflow_moe_parity(self):
        for batch_size in [1, 4]:
            for sequence_length in [1, 128, 1024]:
                # use a small sizes to speed up the test
                config = ArflowConfig(in_features=2048, interm_features=1024, out_features=96)
                Arflow_moe = ArflowSparseMoeBlock(config, batch_size, sequence_length)
                Arflow_moe.parity_check()


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------------------------------------------------------------------------------------
# class MoEBlockForOnnxExport(torch.autograd.Function):
#     @staticmethod
#     def forward(
#         ctx,
#         hidden_states,
#         router_logits,
#         batch_size,
#         sequence_length,
#         hidden_dim,
#         top_k,
#         num_experts,
#         hidden_act,
#         ffn_dim,
#         start_expert_id,
#         expert_weights_1,
#         expert_weights_2,
#         expert_weights_3,
#     ):
#         if get_tensor_model_parallel_world_size() > 1:
#             final_hidden_states = torch.zeros(
#                 (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
#             )
#             return final_hidden_states
#         routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
#         routing_weights, selected_experts = torch.topk(
#             routing_weights, top_k, dim=-1)

#         routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
#         # we cast back to the input dtype
#         routing_weights = routing_weights.to(hidden_states.dtype)

#         final_hidden_states = torch.zeros(
#             (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
#         )

#         # One hot encode the selected experts to create an expert mask
#         # this will be used to easily index which expert is going to be sollicitated
#         expert_mask = torch.nn.functional.one_hot(
#             selected_experts, num_classes=num_experts).permute(2, 1, 0)

#         # Loop over all available experts in the model and perform the computation on each expert
#         for expert_idx in range(num_experts):

#             # expert_layer = self.experts[expert_idx]
#             expert_weight_1 = expert_weights_1[expert_idx]
#             expert_weight_2 = expert_weights_2[expert_idx]
#             expert_weight_3 = expert_weights_3[expert_idx]
#             idx, top_x = torch.where(expert_mask[expert_idx])

#             if top_x.shape[0] == 0:
#                 continue

#             # in torch it is faster to index using lists than torch tensors
#             top_x_list = top_x.tolist()
#             idx_list = idx.tolist()

#             # Index the correct hidden states and compute the expert hidden state for
#             # the current expert. We need to make sure to multiply the output hidden
#             # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
#             current_state = hidden_states[None,
#                                           top_x_list].reshape(-1, hidden_dim)
#             # current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]
#             expert_layer_out = torch.nn.functional.linear(
#                 current_state, expert_weight_1)
#             expert_layer_out = ACT2FN[hidden_act](expert_layer_out)
#             expert_layer_out = expert_layer_out * \
#                 torch.nn.functional.linear(current_state, expert_weight_3)
#             expert_layer_out = torch.nn.functional.linear(
#                 expert_layer_out, expert_weight_2)

#             current_hidden_states = expert_layer_out * \
#                 routing_weights[top_x_list, idx_list, None]

#             # However `index_add_` only support torch tensors for indexing so we'll use
#             # the `top_x` tensor here.
#             final_hidden_states.index_add_(
#                 0, top_x, current_hidden_states.to(hidden_states.dtype))
#         final_hidden_states = final_hidden_states.reshape(
#             batch_size, sequence_length, hidden_dim)
#         return final_hidden_states

#     @staticmethod
#     def symbolic(g: torch.Graph, hidden_states, router_logits, batch_size, sequence_length, hidden_dim, top_k, num_experts, hidden_act, ffn_dim,
#                  start_expert_id,
#                  expert_weights_1, expert_weights_2, expert_weights_3):
#         moe_experts_bias1 = torch.zeros(
#             num_experts, ffn_dim, dtype=hidden_states.type().dtype())
#         moe_experts_bias2 = torch.zeros(
#             get_tensor_model_parallel_world_size()*num_experts, hidden_dim, dtype=hidden_states.type().dtype())
#         moe_experts_bias3 = torch.zeros(
#             num_experts, ffn_dim, dtype=hidden_states.type().dtype())

#         bias1 = g.op("Constant", value_t=moe_experts_bias1)
#         bias2 = g.op("Constant", value_t=moe_experts_bias2)
#         bias3 = g.op("Constant", value_t=moe_experts_bias3)
#         None_value = g.op("Constant", value_t=torch.tensor(
#             [], dtype=torch.float16))

#         if get_tensor_model_parallel_world_size() > 1:
#             final_hidden_states = g.op("com.microsoft::ShardedMoE", hidden_states, router_logits, expert_weights_1, bias1, expert_weights_2,
#                                        bias2, expert_weights_3, activation_type_s="silu", k_i=top_k, normalize_routing_weights_i=1, local_experts_start_index_i=start_expert_id)
#         else:
#             final_hidden_states = g.op("com.microsoft::MoE", hidden_states, router_logits, expert_weights_1, None_value, bias1, expert_weights_2,
#                                        None_value, bias2, expert_weights_3, None_value, bias3, activation_type_s="silu", k_i=top_k, normalize_routing_weights_i=1)
#         final_hidden_states.setType(hidden_states.type())
#         return final_hidden_states


# class MixtralMoE(nn.Module):
#     def __init__(
#         self,
#         config: MixtralConfig,
#         linear_method: Optional[LinearMethodBase] = None,
#     ):
#         super().__init__()
#         self.config = config
#         self.rank = get_tensor_model_parallel_rank()
#         self.tp_size = get_tensor_model_parallel_world_size()
#         self.num_total_experts = config.num_local_experts
#         self.top_k = config.num_experts_per_tok
#         self.hidden_act = config.hidden_act
#         if self.tp_size > self.num_total_experts:
#             raise ValueError(
#                 f"Tensor parallel size {self.tp_size} is greater than "
#                 f"the number of experts {self.num_total_experts}.")
#         # Split experts equally between ranks
#         self.expert_indicies = np.array_split(range(
#             self.num_total_experts), self.tp_size)[self.rank].tolist()
#         if not self.expert_indicies:
#             raise ValueError(
#                 f"Rank {self.rank} has no experts assigned to it.")

#         self.experts = nn.ModuleList([
#             MixtralMLP(self.num_total_experts,
#                        config.hidden_size,
#                        config.intermediate_size,
#                        linear_method=linear_method)
#             if idx in self.expert_indicies else None
#             for idx in range(self.num_total_experts)
#         ])
#         self.gate = ReplicatedLinear(config.hidden_size,
#                                      self.num_total_experts,
#                                      bias=False,
#                                      linear_method=None)

#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         batch_size, sequence_length, hidden_dim = hidden_states.shape
#         hidden_states = hidden_states.view(-1, hidden_dim)
#         # router_logits: (batch * sequence_length, n_experts)
#         router_logits, _ = self.gate(hidden_states)

#         if torch.onnx.is_in_onnx_export():
#             final_hidden_states = MoEBlockForOnnxExport.apply(
#                 hidden_states,
#                 router_logits,
#                 batch_size,
#                 sequence_length,
#                 hidden_dim.item(),
#                 self.top_k,
#                 len(self.expert_indicies),
#                 self.hidden_act,
#                 int(self.experts[self.expert_indicies[0]
#                                  ].w1.weight.shape[0].item()),
#                 self.expert_indicies[0],
#                 torch.stack(
#                     [expert.w1.weight for expert in self.experts if expert is not None], dim=0),
#                 torch.stack(
#                     [expert.w2.weight for expert in self.experts if expert is not None], dim=0),
#                 torch.stack(
#                     [expert.w3.weight for expert in self.experts if expert is not None], dim=0),
#             )
#             return final_hidden_states.view(batch_size, sequence_length, hidden_dim)

#         routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
#         routing_weights, selected_experts = torch.topk(routing_weights,
#                                                        self.top_k,
#                                                        dim=-1)
#         routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

#         final_hidden_states = None
#         for expert_idx in self.expert_indicies:
#             expert_layer = self.experts[expert_idx]
#             expert_mask = (selected_experts == expert_idx)
#             expert_weights = (routing_weights * expert_mask).sum(dim=-1,
#                                                                  keepdim=True)

#             current_hidden_states = expert_layer(hidden_states).mul_(
#                 expert_weights)
#             if final_hidden_states is None:
#                 final_hidden_states = current_hidden_states
#             else:
#                 final_hidden_states.add_(current_hidden_states)

#         return tensor_model_parallel_all_reduce(final_hidden_states).view(
#             batch_size, sequence_length, hidden_dim)
# ----------------------------------------------------------------------------------------------------------------------
