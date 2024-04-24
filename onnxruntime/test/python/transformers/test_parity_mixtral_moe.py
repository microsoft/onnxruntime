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
import time
from torch import nn

import onnx
import onnxruntime

torch.manual_seed(42)
numpy.random.seed(42)

ORT_DTYPE = TensorProto.FLOAT16
NP_TYPE = numpy.float16 if ORT_DTYPE == TensorProto.FLOAT16 else numpy.float32
THRESHOLD = 5e-1


def value_string_of(numpy_array):
    arr = numpy_array.flatten()
    lines = ["f, ".join([str(v) for v in arr[i : min(i + 8, arr.size)]]) for i in range(0, arr.size, 8)]
    return "{\n    " + "f,\n    ".join(lines) + "f}"


def print_tensor(name, numpy_array):
    print(f"const std::vector<float> {name} = {value_string_of(numpy_array)};")


def create_moe_onnx_graph(
    num_experts,
    hidden_size,
    inter_size,
    fc1_experts_weights,
    fc2_experts_weights,
    fc3_experts_weights,
    topk,
):
    nodes = [
        helper.make_node(
            "MoE",
            [
                "input",
                "router_probs",
                "fc1_experts_weights",
                "",
                "fc2_experts_weights",
                "",
                "fc3_experts_weights",
            ],
            ["output"],
            "MoE_0",
            k=topk,
            normalize_routing_weights=1,
            activation_type="silu",
            domain="com.microsoft",
        ),
    ]

    fc1_shape = [num_experts, hidden_size, inter_size]
    fc2_shape = [num_experts, inter_size, hidden_size]
    fc3_shape = [num_experts, hidden_size, inter_size]

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
    ]

    graph_inputs = [
        helper.make_tensor_value_info("input", ORT_DTYPE, ["num_rows", hidden_size]),
    ]

    graph_inputs.append(
        helper.make_tensor_value_info(
            "router_probs",
            ORT_DTYPE,
            ["num_rows", num_experts],
        )
    )

    graph_outputs = [
        helper.make_tensor_value_info("output", ORT_DTYPE, ["num_rows", hidden_size]),
    ]

    graph = helper.make_graph(
        nodes,
        "MoE_Graph",
        graph_inputs,
        graph_outputs,
        initializers,
    )

    model = helper.make_model(graph)

    path = "mixtral_moe.onnx"
    onnx.save(model, path, save_as_external_data=True)
    return path


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {
    "silu": nn.SiLU,
}
ACT2FN = ClassInstantier(ACT2CLS)


class MixtralConfig:
    def __init__(
        self,
        hidden_size=4096,
        intermediate_size=14336,
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
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
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


class MixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states_1 = self.act_fn(self.w1(hidden_states))
        current_hidden_states_3 = self.w3(hidden_states)
        current_hidden_states = current_hidden_states_1 * current_hidden_states_3
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class MixtralSparseMoeBlock(nn.Module):
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

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList([MixtralBlockSparseTop2MLP(config) for _ in range(self.num_experts)])

        w1_list = []
        w2_list = []
        w3_list = []
        for i in range(self.num_experts):
            w1_list.append(self.experts[i].w1.weight)
            w2_list.append(self.experts[i].w2.weight)
            w3_list.append(self.experts[i].w3.weight)

        self.moe_experts_weight1 = torch.stack(w1_list, dim=0)
        self.moe_experts_weight2 = torch.stack(w2_list, dim=0)
        self.moe_experts_weight3 = torch.stack(w3_list, dim=0)

        self.moe_onnx_path = create_moe_onnx_graph(
            self.num_experts,
            self.hidden_dim,
            self.ffn_dim,
            self.moe_experts_weight1,
            self.moe_experts_weight2,
            self.moe_experts_weight3,
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
        ort_session = InferenceSession(
            self.moe_onnx_path, sess_options, providers=["CUDAExecutionProvider"]
        )

        return ort_session

    def ort_run_with_iobinding(self, ort_inputs, repeat=1000):
        iobinding = self.ort_sess.io_binding()
        device_id = torch.cuda.current_device()

        iobinding.bind_input(
            name="input",
            device_type="cuda",
            device_id=device_id,
            element_type=NP_TYPE,
            shape=ort_inputs["input"].shape,
            buffer_ptr=onnxruntime.OrtValue.ortvalue_from_numpy(ort_inputs["input"], "cuda", device_id).data_ptr(),
        )
        iobinding.bind_input(
            name="router_probs",
            device_type="cuda",
            device_id=device_id,
            element_type=NP_TYPE,
            shape=ort_inputs["router_probs"].shape,
            buffer_ptr=onnxruntime.OrtValue.ortvalue_from_numpy(
                ort_inputs["router_probs"], "cuda", device_id
            ).data_ptr(),
        )

        iobinding.bind_output(
            name="output",
            device_type="cuda",
            device_id=device_id,
            element_type=NP_TYPE,
            shape=ort_inputs["input"].shape,
            buffer_ptr=onnxruntime.OrtValue.ortvalue_from_numpy(
                numpy.zeros(ort_inputs["input"].shape), "cuda", device_id
            ).data_ptr(),
        )

        # warm up
        # for _ in range(3):
        #     self.ort_sess.run_with_iobinding(iobinding)

        total_lapse = 0
        for _ in range(repeat):
            iobinding.synchronize_inputs()
            s = time.time()
            self.ort_sess.run_with_iobinding(iobinding)
            e = time.time()
            total_lapse += e - s
            iobinding.synchronize_outputs()

        lapse = total_lapse / repeat
        print(f"MoE cuda kernel time: {lapse * 1000} ms")

        return lapse

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

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

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states  # , router_logits

    def ort_forward(self, hidden_states: torch.Tensor, iobinding=False, repeat=1000) -> torch.Tensor:
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
            if iobinding:
                return self.ort_run_with_iobinding(ort_inputs, repeat=repeat)

            ort_output = self.ort_sess.run(None, ort_inputs)
            return torch.tensor(ort_output).reshape(batch_size, sequence_length, -1).type(torch.float)  # , router_logits

        # print_tensor("input", ort_inputs["input"])
        # print_tensor("router_probs", ort_inputs["router_probs"])
        # print_tensor("fc1_experts_weights", self.moe_experts_weight1.detach().numpy())
        # print_tensor("fc2_experts_weights", self.moe_experts_weight2.detach().numpy())
        # print_tensor("fc3_experts_weights", self.moe_experts_weight3.detach().numpy())
        # print_tensor("output", ort_output[0])

        return None

    def parity_check(self, batch_size, sequence_length):
        hidden_state = torch.randn(batch_size, sequence_length, self.hidden_dim)
        torch_output = self.forward(hidden_state)
        ort_output = self.ort_forward(hidden_state)
        if ort_output is not None:
            assert torch.allclose(torch_output, ort_output, rtol=THRESHOLD, atol=THRESHOLD)
            print(
                "batch_size:",
                batch_size,
                " sequence_length:",
                sequence_length,
                " max_diff:",
                (torch_output - ort_output).abs().max(),
                " parity: OK",
            )

    def benchmark(self, batch_size, sequence_length):
        hidden_state = torch.randn(batch_size, sequence_length, self.hidden_dim)
        lapse = self.ort_forward(hidden_state, iobinding=True, repeat=100)
        return lapse


class TestMixtralMoE(unittest.TestCase):
    def test_mixtral_moe_parity(self):
        config = MixtralConfig(hidden_size=128, intermediate_size=512)
        mixtral_moe = MixtralSparseMoeBlock(config)
        for batch_size in [1, 16]:
            for sequence_length in [128, 1024]:
                # use a small sizes to speed up the test
                mixtral_moe.parity_check(batch_size, sequence_length)

def environ_reset():
    import os

    vars = ["K_FC1_CtaShape16x128x64_WarpShape16x32x64",
            "K_FC1_CtaShape16x256x64_WarpShape16x64x64",
            "K_FC1_CtaShape32x128x64_WarpShape32x32x64",
            "K_FC1_CtaShape64x128x64_WarpShape32x64x64",
            "K_FC1_CtaShape128x128x64_WarpShape64x32x64",
            "K_FC1_Stages_2", "K_FC1_Stages_3", "K_FC1_Stages_4"]

    for var in vars:
        if var in os.environ:
            os.environ.pop(var)

def perf_tuning():
    import os

    config = MixtralConfig(hidden_size=4096, intermediate_size=7168)
    mixtral_moe = MixtralSparseMoeBlock(config)

    tiles = ["K_FC1_CtaShape16x128x64_WarpShape16x32x64",
            "K_FC1_CtaShape16x256x64_WarpShape16x64x64",
            "K_FC1_CtaShape32x128x64_WarpShape32x32x64",
            "K_FC1_CtaShape64x128x64_WarpShape32x64x64",
            "K_FC1_CtaShape128x128x64_WarpShape64x32x64"]

    stages = ["K_FC1_Stages_2", "K_FC1_Stages_3", "K_FC1_Stages_4"]

    environ_reset()

    for batch_size in [1]:
        for sequence_length in range(32, 9128, 8):
            print(f"batch_size: {batch_size}, sequence_length: {sequence_length}")
            for tile in tiles:
                for stage in stages:
                    environ_reset()
                    print(f"tile: {tile}, stage: {stage}")
                    os.environ[tile] = "1"
                    os.environ[stage] = "1"
                    mixtral_moe.benchmark(batch_size, sequence_length)
                    os.environ.pop(tile)
                    os.environ.pop(stage)



if __name__ == "__main__":
    # unittest.main()
    perf_tuning()
