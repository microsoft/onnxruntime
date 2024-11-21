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
from parameterized import parameterized
from torch import nn

import onnxruntime

torch.manual_seed(42)
numpy.random.seed(42)

USE_QUANT = False
ORT_DTYPE = TensorProto.FLOAT16 if USE_QUANT else TensorProto.FLOAT
NP_TYPE = numpy.float16 if ORT_DTYPE == TensorProto.FLOAT16 else numpy.float32
THRESHOLD = 5e-1 if USE_QUANT else 1e-2


def value_string_of(numpy_array):
    arr = numpy_array.flatten()
    lines = ["f, ".join([str(v) for v in arr[i : min(i + 8, arr.size)]]) for i in range(0, arr.size, 8)]
    return "{\n    " + "f,\n    ".join(lines) + "f}"


def print_tensor(name, numpy_array):
    print(f"const std::vector<float> {name} = {value_string_of(numpy_array)};")


def quant_dequant(weights, quant_mode: bool = True):
    # use the test version `_symmetric_...` to get the non-interleaved weights
    type = torch.quint4x2 if quant_mode else torch.int8
    # This import is needed to use torch.ops.trtllm._symmetric_quantize_last_axis_of_batched_matrix()
    # Comment out this line for passing the lintrunner check in the CI.
    # import tensorrt_llm

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
    sequence_length,
    num_experts,
    hidden_size,
    inter_size,
    fc1_experts_weights,
    fc1_experts_bias,
    fc2_experts_weights,
    fc2_experts_bias,
):
    nodes = [
        helper.make_node(
            "MoE",
            [
                "input",
                "router_probs",
                "fc1_experts_weights",
                "fc1_experts_bias",
                "fc2_experts_weights",
                "fc2_experts_bias",
            ],
            ["output"],
            "MoE_0",
            k=1,
            activation_type="gelu",
            domain="com.microsoft",
        ),
    ]

    fc1_shape = [num_experts, hidden_size, inter_size]
    fc2_shape = [num_experts, inter_size, hidden_size]

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
    ]

    fc1_bias_shape = [num_experts, inter_size]
    fc2_bias_shape = [num_experts, hidden_size]
    initializers.extend(
        [
            helper.make_tensor(
                "fc1_experts_bias",
                ORT_DTYPE,
                fc1_bias_shape,
                fc1_experts_bias.to(torch_type).flatten().tolist(),
                raw=False,
            ),
            helper.make_tensor(
                "fc2_experts_bias",
                ORT_DTYPE,
                fc2_bias_shape,
                fc2_experts_bias.to(torch_type).flatten().tolist(),
                raw=False,
            ),
        ]
    )

    graph_inputs = [
        helper.make_tensor_value_info("input", ORT_DTYPE, [sequence_length, hidden_size]),
    ]

    graph_inputs.append(
        helper.make_tensor_value_info(
            "router_probs",
            ORT_DTYPE,
            [sequence_length, num_experts],
        )
    )

    graph_outputs = [
        helper.make_tensor_value_info("output", ORT_DTYPE, [sequence_length, hidden_size]),
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


def create_mixtral_moe_onnx_graph(
    sequence_length,
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
        helper.make_tensor_value_info("input", ORT_DTYPE, [sequence_length, hidden_size]),
    ]

    graph_inputs.append(
        helper.make_tensor_value_info(
            "router_probs",
            ORT_DTYPE,
            [sequence_length, num_experts],
        )
    )

    graph_outputs = [
        helper.make_tensor_value_info("output", ORT_DTYPE, [sequence_length, hidden_size]),
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


def create_phi_moe_onnx_graph(
    sequence_length,
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
            "MoE" if not use_quant else "QMoE",
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

    if use_quant:
        nodes[0].attribute.extend([helper.make_attribute("expert_weight_bits", 8)])

    fc1_shape = [num_experts, hidden_size, inter_size]
    fc2_shape = [num_experts, inter_size, hidden_size]
    fc3_shape = [num_experts, hidden_size, inter_size]

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
        helper.make_tensor_value_info("input", ORT_DTYPE, [sequence_length, hidden_size]),
    ]

    graph_inputs.append(
        helper.make_tensor_value_info(
            "router_probs",
            ORT_DTYPE,
            [sequence_length, num_experts],
        )
    )

    graph_outputs = [
        helper.make_tensor_value_info("output", ORT_DTYPE, [sequence_length, hidden_size]),
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
    "gelu": nn.GELU,
}
ACT2FN = ClassInstantier(ACT2CLS)


class MixtralConfig:
    def __init__(
        self,
        hidden_size=4096,
        intermediate_size=14336,
        hidden_act="silu",
        num_experts_per_tok=2,
        num_local_experts=8,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts


class PhiMoEConfig:
    def __init__(
        self,
        hidden_size=4096,
        intermediate_size=14336,
        hidden_act="silu",
        num_experts_per_tok=2,
        num_local_experts=8,
        router_jitter_noise=0.01,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.router_jitter_noise = router_jitter_noise


class MoEGate(nn.Module):
    def __init__(self, num_experts, in_features):
        super().__init__()
        self.wg_reduction = torch.nn.Linear(in_features, 16, bias=False)

        wg = torch.empty(num_experts, 16)
        torch.nn.init.orthogonal_(wg, gain=0.32)
        self.register_parameter("wg", torch.nn.Parameter(wg))

    def forward(self, input):
        input = self.wg_reduction(input)
        with torch.no_grad():
            wg_norm = self.wg.norm(p=2.0, dim=1, keepdim=True)
            self.wg.mul_(1.5 / wg_norm)
        logits = self._cosine(input, self.wg)
        return logits

    def _cosine(self, mat1, mat2, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2

        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)


class MoERuntimeExperts(nn.Module):
    def __init__(
        self,
        num_experts,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
    ):
        super().__init__()

        self.weight1 = nn.Parameter(torch.rand(num_experts, in_features, hidden_features))
        self.weight2 = nn.Parameter(torch.rand(num_experts, hidden_features, out_features))

        self.bias1 = nn.Parameter(torch.rand(num_experts, hidden_features)) if bias else None
        self.bias2 = nn.Parameter(torch.rand(num_experts, in_features)) if bias else None

        self.act = act_layer()

    def forward(self, x, indices_s):
        x = x.unsqueeze(1)
        x = self.bmm(x, self.weight1, indices_s)
        if self.bias1 is not None:
            x = x + self.bias1[indices_s].unsqueeze(1)  # S x hidden_features
        x = self.act(x)
        x = self.bmm(x, self.weight2, indices_s)
        if self.bias2 is not None:
            x = x + self.bias2[indices_s].unsqueeze(1)  # S x 1 x in_features
        return x

    def bmm(self, x, weight, indices_s):
        x = torch.bmm(x, weight[indices_s])  # S x 1 x hidden_features
        return x


class MoEBlockSparseTop2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class MixtralBlockSparseTop2MLP(MoEBlockSparseTop2MLP):
    def __init__(self, config: MixtralConfig):
        super().__init__(config)


class PhiMoEBlockSparseTop2MLP(MoEBlockSparseTop2MLP):
    def __init__(self, config: PhiMoEConfig):
        super().__init__(config)


class SparseMoeBlockORTHelper(nn.Module):
    def __init__(self):
        super().__init__()

    def create_ort_session(self, moe_onnx_graph):
        from onnxruntime import InferenceSession, SessionOptions

        sess_options = SessionOptions()

        cuda_providers = ["CUDAExecutionProvider"]
        if cuda_providers[0] not in onnxruntime.get_available_providers():
            return None

        sess_options.log_severity_level = 2
        ort_session = InferenceSession(moe_onnx_graph, sess_options, providers=["CUDAExecutionProvider"])

        return ort_session

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pass

    def ort_forward(self, hidden_states: torch.Tensor, iobinding=False) -> torch.Tensor:
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
            if not iobinding:
                ort_output = self.ort_sess.run(None, ort_inputs)
                return torch.tensor(ort_output).reshape(batch_size, sequence_length, -1)  # , router_logits
            else:
                self.ort_run_with_iobinding(ort_inputs)
                return None

        # print_tensor("input", ort_inputs["input"])
        # print_tensor("router_probs", ort_inputs["router_probs"])
        # print_tensor("fc1_experts_weights", self.moe_experts_weight1.detach().numpy())
        # print_tensor("fc2_experts_weights", self.moe_experts_weight2.detach().numpy())
        # print_tensor("fc3_experts_weights", self.moe_experts_weight3.detach().numpy())
        # print_tensor("output", ort_output[0])

        return None

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
        for _ in range(5):
            iobinding.synchronize_inputs()
            self.ort_sess.run_with_iobinding(iobinding)
            iobinding.synchronize_outputs()

        import time

        s = time.time()
        for _ in range(repeat):
            iobinding.synchronize_inputs()
            self.ort_sess.run_with_iobinding(iobinding)
            iobinding.synchronize_outputs()
        e = time.time()
        print(f"MoE cuda kernel time: {(e - s) / repeat * 1000} ms")

    def parity_check(self):
        hidden_state = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim)
        torch_output = self.forward(hidden_state)
        ort_output = self.ort_forward(hidden_state)
        if ort_output is not None:
            print(
                "name:",
                self.__class__.__name__,
                " batch_size:",
                self.batch_size,
                " sequence_length:",
                self.sequence_length,
                " max_diff:",
                (torch_output - ort_output).abs().max(),
            )
            torch.testing.assert_close(ort_output.to(torch.float32), torch_output, rtol=THRESHOLD, atol=THRESHOLD)

    def benchmark_ort(self):
        hidden_state = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim)
        self.ort_forward(hidden_state, iobinding=True)


class SwitchMoE(SparseMoeBlockORTHelper):
    def __init__(
        self,
        batch_size,
        sequence_length,
        num_experts,
        in_features,
        hidden_features=None,
        out_features=None,
        eval_capacity=-1,
        activation="gelu",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_experts = num_experts
        self.hidden_dim = in_features
        self.ffn_dim = hidden_features
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.eval_capacity = eval_capacity  # -1 means we route all tokens

        self.gate = MoEGate(num_experts=num_experts, in_features=in_features)
        self.moe_experts = MoERuntimeExperts(
            num_experts=num_experts,
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            act_layer=ACT2CLS[activation],
            bias=True,
        )

        self.moe_onnx_graph = create_moe_onnx_graph(
            batch_size * sequence_length,
            num_experts,
            in_features,
            hidden_features,
            self.moe_experts.weight1.transpose(1, 2),
            self.moe_experts.bias1,
            self.moe_experts.weight2.transpose(1, 2),
            self.moe_experts.bias2,
        )

        self.ort_sess = self.create_ort_session(self.moe_onnx_graph)

        self.torch_input = torch.randn(batch_size, sequence_length, in_features)

    def forward(self, hidden_states):
        b, t, c = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, c)
        logits = self.gate(hidden_states)
        gates = torch.nn.functional.softmax(logits, dim=1)
        ret = torch.max(gates, dim=1)
        indices_s = ret.indices  # dim: [bs], the index of the expert with highest softmax value
        scores = ret.values.unsqueeze(-1).unsqueeze(-1)  # S
        hidden_states = self.moe_experts(hidden_states, indices_s)

        hidden_states = hidden_states * scores
        hidden_states = hidden_states.reshape(b, t, c)

        return hidden_states


class MixtralSparseMoeBlock(SparseMoeBlockORTHelper):
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

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.moe_onnx_graph = create_mixtral_moe_onnx_graph(
            self.batch_size * self.sequence_length,
            self.num_experts,
            self.hidden_dim,
            self.ffn_dim,
            self.moe_experts_weight1,
            self.moe_experts_weight2,
            self.moe_experts_weight3,
            self.top_k,
        )

        self.ort_sess = self.create_ort_session(self.moe_onnx_graph)

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


def masked_sampling_omp_inference(scores, top_k, jitter_eps, training):
    assert top_k == 2
    assert not training

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


class PhiMoESparseMoeBlock(SparseMoeBlockORTHelper):
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
        self.moe_onnx_graph = create_phi_moe_onnx_graph(
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

        self.ort_sess = self.create_ort_session(self.moe_onnx_graph)

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


def small_test_cases():
    for batch_size in [1, 4, 16]:
        for sequence_length in [128, 512, 1024]:
            yield batch_size, sequence_length


def phi3_test_cases():
    # TODO: phi3 moe failed in long sequence lengths (max diff 0.22 > threshold 0.01), need investigation.
    for batch_size in [1, 4, 16]:
        for sequence_length in [128]:
            yield batch_size, sequence_length


class TestSwitchMoE(unittest.TestCase):
    @parameterized.expand(small_test_cases())
    def test_switch_moe_parity(self, batch_size, sequence_length):
        # if platform.system() == "Windows":
        #     pytest.skip("Skip on Windows")
        switch_moe = SwitchMoE(
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_experts=8,
            in_features=256,
            hidden_features=1024,
            out_features=256,
        )
        switch_moe.parity_check()
        # switch_moe.benchmark_ort()


class TestMixtralMoE(unittest.TestCase):
    @parameterized.expand(small_test_cases())
    def test_mixtral_moe_parity(self, batch_size, sequence_length):
        config = MixtralConfig(hidden_size=256, intermediate_size=1024)
        mixtral_moe = MixtralSparseMoeBlock(config, batch_size, sequence_length)
        mixtral_moe.parity_check()
        # mixtral_moe.benchmark_ort()


class TestPhiMoE(unittest.TestCase):
    @parameterized.expand(phi3_test_cases())
    def test_phi3_moe_parity(self, batch_size, sequence_length):
        config = PhiMoEConfig(hidden_size=256, intermediate_size=1024)
        phi3_moe = PhiMoESparseMoeBlock(config, batch_size, sequence_length)
        phi3_moe.parity_check()
        # phi3_moe.benchmark_ort()


if __name__ == "__main__":
    unittest.main()
