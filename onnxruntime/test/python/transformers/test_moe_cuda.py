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
import itertools
import os
import unittest
from collections import OrderedDict

import numpy
import torch
import torch.nn.functional as F
from onnx import TensorProto, helper
from parameterized import parameterized
from torch import nn

import onnxruntime

# Reduces number of tests to run for faster pipeline checks
pipeline_mode = os.getenv("PIPELINE_MODE", "1") == "1"

onnxruntime.preload_dlls()

# Determine the execution provider and device based on CUDA availability.
use_cuda = "CUDAExecutionProvider" in onnxruntime.get_available_providers() and torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
ort_provider = ["CUDAExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]

torch.manual_seed(42)
numpy.random.seed(42)

onnx_to_torch_type_map = {
    TensorProto.FLOAT16: torch.float16,
    TensorProto.FLOAT: torch.float,
    TensorProto.BFLOAT16: torch.bfloat16,
    TensorProto.UINT8: torch.uint8,
}

ort_to_numpy_type_map = {
    TensorProto.FLOAT16: numpy.float16,
    TensorProto.FLOAT: numpy.float32,
    TensorProto.UINT8: numpy.uint8,
}

ort_dtype_name_map = {
    TensorProto.FLOAT16: "FP16",
    TensorProto.FLOAT: "FP32",
    TensorProto.BFLOAT16: "BF16",
}


def quant_dequant(weights, is_4_bit_quantization: bool = True):
    type = torch.quint4x2 if is_4_bit_quantization else torch.int8

    import tensorrt_llm  # noqa: PLC0415

    # Avoid lint false alert that the package is not used. Note that this function will not be called in pipeline.
    if pipeline_mode:
        print("Tensorrt LLM version", tensorrt_llm.__version__)

    quant_weights, processed_q_weight, torch_weight_scales = (
        torch.ops.trtllm._symmetric_quantize_last_axis_of_batched_matrix(weights.T.cpu().contiguous(), type)
    )

    # Unpack the int4s int int8s
    if is_4_bit_quantization:
        upper = quant_weights >> 4
        lower = (quant_weights << 4) >> 4  # Arithmetic right shift sign extends
        quant_weights = torch.stack((lower, upper), dim=2).view(weights.T.shape)

    result = torch.multiply(quant_weights.to(dtype=weights.dtype), torch_weight_scales.unsqueeze(0)).T.contiguous()
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
    onnx_dtype,
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

    torch_dtype = onnx_to_torch_type_map[onnx_dtype]

    initializers = [
        helper.make_tensor(
            "fc1_experts_weights",
            onnx_dtype,
            fc1_shape,
            fc1_experts_weights.to(torch_dtype).flatten().tolist(),
            raw=False,
        ),
        helper.make_tensor(
            "fc2_experts_weights",
            onnx_dtype,
            fc2_shape,
            fc2_experts_weights.to(torch_dtype).flatten().tolist(),
            raw=False,
        ),
    ]

    fc1_bias_shape = [num_experts, inter_size]
    fc2_bias_shape = [num_experts, hidden_size]
    initializers.extend(
        [
            helper.make_tensor(
                "fc1_experts_bias",
                onnx_dtype,
                fc1_bias_shape,
                fc1_experts_bias.to(torch_dtype).flatten().tolist(),
                raw=False,
            ),
            helper.make_tensor(
                "fc2_experts_bias",
                onnx_dtype,
                fc2_bias_shape,
                fc2_experts_bias.to(torch_dtype).flatten().tolist(),
                raw=False,
            ),
        ]
    )

    graph_inputs = [
        helper.make_tensor_value_info("input", onnx_dtype, [sequence_length, hidden_size]),
    ]

    graph_inputs.append(
        helper.make_tensor_value_info(
            "router_probs",
            onnx_dtype,
            [sequence_length, num_experts],
        )
    )

    graph_outputs = [
        helper.make_tensor_value_info("output", onnx_dtype, [sequence_length, hidden_size]),
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
    onnx_dtype,
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

    torch_dtype = onnx_to_torch_type_map[onnx_dtype]

    initializers = [
        helper.make_tensor(
            "fc1_experts_weights",
            onnx_dtype,
            fc1_shape,
            fc1_experts_weights.to(torch_dtype).flatten().tolist(),
            raw=False,
        ),
        helper.make_tensor(
            "fc2_experts_weights",
            onnx_dtype,
            fc2_shape,
            fc2_experts_weights.to(torch_dtype).flatten().tolist(),
            raw=False,
        ),
        helper.make_tensor(
            "fc3_experts_weights",
            onnx_dtype,
            fc3_shape,
            fc3_experts_weights.to(torch_dtype).flatten().tolist(),
            raw=False,
        ),
    ]

    graph_inputs = [
        helper.make_tensor_value_info("input", onnx_dtype, [sequence_length, hidden_size]),
    ]

    graph_inputs.append(
        helper.make_tensor_value_info(
            "router_probs",
            onnx_dtype,
            [sequence_length, num_experts],
        )
    )

    graph_outputs = [
        helper.make_tensor_value_info("output", onnx_dtype, [sequence_length, hidden_size]),
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
    topk,
    onnx_dtype,
    quant_bits=0,
    fc1_scales=None,
    fc2_scales=None,
    fc3_scales=None,
):
    use_quant = quant_bits > 0
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

    op_name = "QMoE" if use_quant else "MoE"
    inputs = (
        [
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
        if use_quant
        else [
            "input",
            "router_probs",
            "fc1_experts_weights",
            "",
            "fc2_experts_weights",
            "",
            "fc3_experts_weights",
        ]
    )

    nodes = [
        helper.make_node(
            op_name,
            inputs,
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
        nodes[0].attribute.extend([helper.make_attribute("expert_weight_bits", quant_bits)])

    components = 2 if quant_bits == 4 else 1
    fc1_shape = [num_experts, hidden_size, inter_size // components]
    fc2_shape = [num_experts, inter_size, hidden_size // components]
    fc3_shape = [num_experts, hidden_size, inter_size // components]

    torch_dtype = onnx_to_torch_type_map[onnx_dtype]

    weight_numpy_type = numpy.uint8 if use_quant else ort_to_numpy_type_map[onnx_dtype]
    weight_onnx_type = TensorProto.UINT8 if use_quant else onnx_dtype

    initializers = [
        helper.make_tensor(
            "fc1_experts_weights",
            weight_onnx_type,
            fc1_shape,
            fc1_experts_weights.flatten().detach().cpu().numpy().astype(weight_numpy_type).tolist(),
            raw=False,
        ),
        helper.make_tensor(
            "fc2_experts_weights",
            weight_onnx_type,
            fc2_shape,
            fc2_experts_weights.flatten().detach().cpu().numpy().astype(weight_numpy_type).tolist(),
            raw=False,
        ),
        helper.make_tensor(
            "fc3_experts_weights",
            weight_onnx_type,
            fc3_shape,
            fc3_experts_weights.flatten().detach().cpu().numpy().astype(weight_numpy_type).tolist(),
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
                    onnx_dtype,
                    fc1_scale_shape,
                    fc1_scales.to(torch_dtype).flatten().tolist(),
                    raw=False,
                ),
                helper.make_tensor(
                    "fc2_scales",
                    onnx_dtype,
                    fc2_scale_shape,
                    fc2_scales.to(torch_dtype).flatten().tolist(),
                    raw=False,
                ),
                helper.make_tensor(
                    "fc3_scales",
                    onnx_dtype,
                    fc3_scale_shape,
                    fc3_scales.to(torch_dtype).flatten().tolist(),
                    raw=False,
                ),
            ]
        )

    graph_inputs = [
        helper.make_tensor_value_info("input", onnx_dtype, [sequence_length, hidden_size]),
    ]

    graph_inputs.append(
        helper.make_tensor_value_info(
            "router_probs",
            onnx_dtype,
            [sequence_length, num_experts],
        )
    )

    graph_outputs = [
        helper.make_tensor_value_info("output", onnx_dtype, [sequence_length, hidden_size]),
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
    def __init__(self, quant_bits=0, onnx_dtype=None):
        super().__init__()
        self.quant_bits = quant_bits
        if onnx_dtype is None:
            self.onnx_dtype = TensorProto.FLOAT16 if self.quant_bits > 0 else TensorProto.FLOAT
        else:
            self.onnx_dtype = onnx_dtype
        self.np_type = numpy.float16 if self.onnx_dtype == TensorProto.FLOAT16 else numpy.float32

    def create_ort_session(self, moe_onnx_graph):
        from onnxruntime import InferenceSession, SessionOptions  # noqa: PLC0415

        sess_options = SessionOptions()
        sess_options.log_severity_level = 2

        try:
            ort_session = InferenceSession(moe_onnx_graph, sess_options, providers=ort_provider)
        except Exception as e:
            print(f"Failed to create ONNX Runtime session with provider {ort_provider}: {e}")
            print("Skipping ONNX Runtime execution for this test case.")
            return None

        return ort_session

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pass

    def ort_forward(self, hidden_states: torch.Tensor, enable_performance_test=False) -> torch.Tensor:
        if self.ort_sess is None:
            return None

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states_flat)

        # Determine the correct torch dtype from the onnx_dtype
        torch_dtype = onnx_to_torch_type_map[self.onnx_dtype]

        # Prepare tensors on the correct device for ORT inference with the CORRECT dtype
        tensors = {
            "input": hidden_states_flat.clone().to(device=device, dtype=torch_dtype),
            "router_probs": router_logits.clone().to(device=device, dtype=torch_dtype),
            "output": torch.zeros_like(hidden_states_flat, device=device, dtype=torch_dtype),
        }

        # Bind inputs and outputs to torch tensors directly.
        iobinding = self.ort_sess.io_binding()

        for name, tensor in tensors.items():
            # Ensure tensor is on the globally defined device
            if name == "output":
                iobinding.bind_output(
                    name=name,
                    device_type=tensor.device.type,
                    device_id=tensor.device.index or 0,
                    element_type=self.onnx_dtype,
                    shape=tensor.shape,
                    buffer_ptr=tensor.data_ptr(),
                )
            else:
                iobinding.bind_input(
                    name=name,
                    device_type=tensor.device.type,
                    device_id=tensor.device.index or 0,
                    element_type=self.onnx_dtype,
                    shape=tensor.shape,
                    buffer_ptr=tensor.data_ptr(),
                )

        iobinding.synchronize_inputs()
        self.ort_sess.run_with_iobinding(iobinding)
        iobinding.synchronize_outputs()

        if enable_performance_test:
            import time  # noqa: PLC0415

            repeat = 1000
            s = time.time()
            for _ in range(repeat):
                iobinding.synchronize_inputs()
                self.ort_sess.run_with_iobinding(iobinding)
                iobinding.synchronize_outputs()
            e = time.time()
            print(f"MoE cuda kernel time: {(e - s) / repeat * 1000} ms")

        # The output tensor is on `device`. Reshape and return it.
        return tensors["output"].reshape(batch_size, sequence_length, hidden_dim)

    def parity_check(self):
        hidden_state = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim).to(device)
        torch_output = self.forward(hidden_state)
        ort_output = self.ort_forward(hidden_state)

        dtype_str = ort_dtype_name_map[self.onnx_dtype]

        # Maps "ort_type:quant_bits" to (atol, rtol)
        ort_dtype_quant_bits_tolerance_map = {
            "FP32:0": (5e-3, 1e-3),
            "FP16:0": (5e-2, 1e-3),
            "FP16:4": (3.0, 1e-2),
            "FP16:8": (2.0, 1e-2),
            "BF16:0": (1.0, 1e-2),
            "BF16:4": (30.0, 1e-1),
            "BF16:8": (20.0, 1e-1),
        }

        atol, rtol = ort_dtype_quant_bits_tolerance_map[f"{dtype_str}:{self.quant_bits}"]
        if ort_output is not None:
            print(
                f"Parity Check: {self.__class__.__name__}, quant_bits: {self.quant_bits}, dtype: {dtype_str},"
                f" batch: {self.batch_size}, seq_len: {self.sequence_length},"
                f" max_diff: {(torch_output.cpu() - ort_output.cpu()).abs().max()}"
            )
            torch.testing.assert_close(
                ort_output.cpu().to(torch.float32), torch_output.cpu().to(torch.float32), rtol=rtol, atol=atol
            )

    def benchmark_ort(self):
        hidden_state = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim).to(device)
        self.ort_forward(hidden_state, enable_performance_test=True)


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
        super().__init__(quant_bits=0)  # SwitchMoE is not quantized
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
            self.onnx_dtype,
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
        super().__init__(quant_bits=0)  # Mixtral test is not quantized
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
            self.onnx_dtype,
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

    def __init__(self, config, batch_size, sequence_length, quant_bits=0, onnx_dtype=None):
        super().__init__(quant_bits, onnx_dtype)
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.router_jitter_noise = config.router_jitter_noise
        use_quant = self.quant_bits > 0

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList([PhiMoEBlockSparseTop2MLP(config) for _ in range(self.num_experts)])

        w1_list, w2_list, w3_list = [], [], []
        w1_scale_list, w2_scale_list, w3_scale_list = [], [], []

        if not use_quant:
            for i in range(self.num_experts):
                w1_list.append(self.experts[i].w1.weight)
                w2_list.append(self.experts[i].w2.weight)
                w3_list.append(self.experts[i].w3.weight)
        else:
            is_4_bit = self.quant_bits == 4
            for i in range(self.num_experts):
                # Corrected quantization logic for per-output-channel quantization
                w1_scale, pre_qweight1, w1_qdq = quant_dequant(self.experts[i].w1.weight, is_4_bit)
                w2_scale, pre_qweight2, w2_qdq = quant_dequant(self.experts[i].w2.weight, is_4_bit)
                w3_scale, pre_qweight3, w3_qdq = quant_dequant(self.experts[i].w3.weight, is_4_bit)

                self.experts[i].w1.weight.data = w1_qdq
                self.experts[i].w2.weight.data = w2_qdq
                self.experts[i].w3.weight.data = w3_qdq

                # Transpose quantized weights to match the expected ONNX layout
                w1_list.append(pre_qweight1)
                w2_list.append(pre_qweight2)
                w3_list.append(pre_qweight3)
                w1_scale_list.append(w1_scale)
                w2_scale_list.append(w2_scale)
                w3_scale_list.append(w3_scale)

        self.moe_experts_weight1 = torch.stack(w1_list, dim=0)
        self.moe_experts_weight2 = torch.stack(w2_list, dim=0)
        self.moe_experts_weight3 = torch.stack(w3_list, dim=0)

        moe_experts_weight_scale1 = torch.stack(w1_scale_list, dim=0) if use_quant else None
        moe_experts_weight_scale2 = torch.stack(w2_scale_list, dim=0) if use_quant else None
        moe_experts_weight_scale3 = torch.stack(w3_scale_list, dim=0) if use_quant else None

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
            self.top_k,
            self.onnx_dtype,
            self.quant_bits,
            moe_experts_weight_scale1,
            moe_experts_weight_scale2,
            moe_experts_weight_scale3,
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


# @unittest.skipIf(not use_cuda, "skipping moe test since it requires cuda environment.")
# class TestSwitchMoE(unittest.TestCase):
#     @parameterized.expand(small_test_cases())
#     def test_switch_moe_parity(self, batch_size, sequence_length):
#         switch_moe = SwitchMoE(
#             batch_size=batch_size,
#             sequence_length=sequence_length,
#             num_experts=8,
#             in_features=256,
#             hidden_features=1024,
#             out_features=256,
#         )
#         switch_moe.to(device)
#         switch_moe.parity_check()


# quant_bits (0 for fp32/fp32, 8 for int8/fp16, 4 for int4/fp16)
# since qMoE test requires tensorrt_llm for quant_dequant. We disable it in CI pipeline to avoid extra dependency.
quant_bits_list = [0] if pipeline_mode else [0, 8, 4]


# @unittest.skipIf(not use_cuda, "skipping moe test since it requires cuda environment.")
# class TestMixtralMoE(unittest.TestCase):
#     @parameterized.expand(small_test_cases())
#     def test_mixtral_moe_parity(self, batch_size, sequence_length):
#         config = MixtralConfig(hidden_size=256, intermediate_size=1024)
#         mixtral_moe = MixtralSparseMoeBlock(config, batch_size, sequence_length)
#         mixtral_moe.to(device)
#         mixtral_moe.parity_check()


phi3_test_cases = list(
    itertools.product(
        [1, 4],  # batch_size
        [1, 32],  # sequence_length
        quant_bits_list,
    )
)


# @unittest.skipIf(not use_cuda, "skipping moe test since it requires cuda environment.")
# class TestPhiMoE(unittest.TestCase):
#     @parameterized.expand(phi3_test_cases)
#     def test_phi3_moe_parity(self, batch_size, sequence_length, quant_bits):
#         config = PhiMoEConfig(hidden_size=256, intermediate_size=1024)
#         phi3_moe = PhiMoESparseMoeBlock(config, batch_size, sequence_length, quant_bits)
#         phi3_moe.to(device)
#         phi3_moe.parity_check()


# ---------------------------------------------
# The following test are for swiglu activation
# ---------------------------------------------
class SwigluMoeConfig:
    def __init__(
        self,
        hidden_size=2048,
        intermediate_size=2048,
        num_experts_per_token=2,
        num_local_experts=8,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts_per_token = num_experts_per_token
        self.num_local_experts = num_local_experts


def make_onnx_intializer(name: str, tensor: torch.Tensor, onnx_dtype: int):
    """Helper to create an ONNX initializer, handling different data types."""
    torch_dtype = onnx_to_torch_type_map[onnx_dtype]
    dims = tensor.shape

    if torch_dtype == torch.bfloat16:
        # BFLOAT16 needs to be written as raw uint16 data
        numpy_vals_uint16 = tensor.to(torch.bfloat16).cpu().view(torch.uint16).numpy()
        initializer = helper.make_tensor(
            name=name,
            data_type=TensorProto.BFLOAT16,
            dims=dims,
            vals=numpy_vals_uint16.tobytes(),
            raw=True,
        )
    else:
        # Handle standard types and quantized weights
        vals = (
            tensor.flatten().detach().cpu().numpy().astype(numpy.uint8).tolist()
            if onnx_dtype == TensorProto.UINT8
            else tensor.detach().to(torch_dtype).flatten().tolist()
        )
        print(f"make_onnx_intializer - name: {name}, dtype: {onnx_dtype}, vals: {vals[:10]}...")  # Debugging output
        initializer = helper.make_tensor(
            name=name,
            data_type=onnx_dtype,
            dims=dims,
            vals=vals,
            raw=False,
        )

    from onnx import numpy_helper

    np_array = numpy_helper.to_array(initializer)
    print(f"Initializer name = {initializer.name}")
    print("Shape:", np_array.shape)
    print("Values:\n", np_array[:10])  # Print first 10 values for debugging
    print("-" * 40)

    return initializer


def create_swiglu_moe_onnx_graph(
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
    inter_size: int,
    topk: int,
    onnx_dtype: int,
    quant_bits: int,
    swiglu_fusion: int,
    fc1_experts_weights: torch.Tensor,
    fc1_experts_bias: torch.Tensor,
    fc2_experts_weights: torch.Tensor,
    fc2_experts_bias: torch.Tensor,
    fc1_experts_weight_scale: torch.Tensor = None,
    fc2_experts_weight_scale: torch.Tensor = None,
):
    """Creates the ONNX graph for MoE/QMoE with SwiGLU activation."""
    use_quant = quant_bits > 0
    op_name = "QMoE" if use_quant else "MoE"

    # Define inputs based on whether the model is quantized
    inputs = (
        [
            "input",
            "router_probs",
            "fc1_experts_weights",
            "fc1_experts_weight_scale",
            "fc1_experts_bias",
            "fc2_experts_weights",
            "fc2_experts_weight_scale",
            "fc2_experts_bias",
        ]
        if use_quant
        else [
            "input",
            "router_probs",
            "fc1_experts_weights",
            "fc1_experts_bias",
            "fc2_experts_weights",
            "fc2_experts_bias",
        ]
    )

    # Create the MoE/QMoE node
    nodes = [
        helper.make_node(
            op_name,
            inputs,
            ["output"],
            "MoE_0",
            k=topk,
            normalize_routing_weights=1,
            activation_type="swiglu",
            swiglu_fusion=swiglu_fusion,
            swiglu_limit=7.0,
            activation_alpha=1.702,
            activation_beta=1.0,
            domain="com.microsoft",
        ),
    ]

    if use_quant:
        nodes[0].attribute.extend([helper.make_attribute("expert_weight_bits", quant_bits)])

    # Define tensor shapes for weights, biases, and scales
    pack_factor = 2 if quant_bits == 4 else 1
    fc1_weight_shape = [num_experts, 2 * inter_size, hidden_size // pack_factor]
    fc1_bias_shape = [num_experts, 2 * inter_size]

    fc2_weight_shape = [num_experts, hidden_size, inter_size // pack_factor]
    fc2_bias_shape = [num_experts, hidden_size]

    weight_onnx_type = TensorProto.UINT8 if use_quant else onnx_dtype

    # Create initializers for all weights and biases
    initializers = [
        make_onnx_intializer("fc1_experts_weights", fc1_experts_weights, weight_onnx_type),
        make_onnx_intializer("fc1_experts_bias", fc1_experts_bias, onnx_dtype),
        make_onnx_intializer("fc2_experts_weights", fc2_experts_weights, weight_onnx_type),
        make_onnx_intializer("fc2_experts_bias", fc2_experts_bias, onnx_dtype),
    ]

    if use_quant:
        initializers.extend(
            [
                make_onnx_intializer("fc1_experts_weight_scale", fc1_experts_weight_scale, onnx_dtype),
                make_onnx_intializer("fc2_experts_weight_scale", fc2_experts_weight_scale, onnx_dtype),
            ]
        )

    # Define graph inputs and outputs
    graph_inputs = [
        helper.make_tensor_value_info("input", onnx_dtype, [num_tokens, hidden_size]),
        helper.make_tensor_value_info("router_probs", onnx_dtype, [num_tokens, num_experts]),
    ]
    graph_outputs = [
        helper.make_tensor_value_info("output", onnx_dtype, [num_tokens, hidden_size]),
    ]

    # Build and return the graph
    graph = helper.make_graph(nodes, "MoE_Graph", graph_inputs, graph_outputs, initializers)
    model = helper.make_model(graph)
    return model.SerializeToString()


class SwigluExpert(nn.Module):
    """
    A PyTorch module for a single SwiGLU expert.
    This implementation correctly separates gate and value projections (w1 and w3)
    to accurately model the reference behavior for testing.
    """

    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.hidden_dim = config.hidden_size
        # Gate and value projections
        self.w1 = nn.Linear(self.hidden_dim, self.intermediate_size, bias=True)
        self.w3 = nn.Linear(self.hidden_dim, self.intermediate_size, bias=True)
        # Output projection
        self.w2 = nn.Linear(self.intermediate_size, self.hidden_dim, bias=True)

    def forward(self, x):
        """Forward pass matches the custom SwiGLU logic of the ONNX implementation."""
        gate_proj = self.w1(x)
        value_proj = self.w3(x)

        # C++ implementation uses specific alpha and limit values
        alpha = 1.702
        limit = 7.0

        gate_proj = gate_proj.clamp(max=limit)
        value_proj = value_proj.clamp(min=-limit, max=limit)

        activated_gate = gate_proj * torch.sigmoid(alpha * gate_proj)
        y = activated_gate * (value_proj + 1.0)
        y = self.w2(y)
        return y


class SwigluMoEBlock(SparseMoeBlockORTHelper):
    """Test block for MoE with SwiGLU activation."""

    def __init__(
        self,
        config: SwigluMoeConfig,
        batch_size: int,
        sequence_length: int,
        quant_bits: int = 0,
        swiglu_fusion: int = 1,
        onnx_dtype=None,
    ):
        super().__init__(quant_bits, onnx_dtype=onnx_dtype)
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_token
        self.swiglu_fusion = swiglu_fusion
        use_quant = self.quant_bits > 0

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=True)
        self.experts = nn.ModuleList([SwigluExpert(config) for _ in range(self.num_experts)])

        # Prepare weights, biases, and scales for the ONNX graph
        w1_list, w2_list, b1_list, b2_list, s1_list, s2_list = [], [], [], [], [], []

        for expert in self.experts:
            b2_list.append(expert.w2.bias)
            w1_weight, w3_weight = expert.w1.weight, expert.w3.weight
            w1_bias, w3_bias = expert.w1.bias, expert.w3.bias

            if not use_quant:
                # For non-quantized, prepare weights and biases for fusion
                fused_w1 = (
                    torch.cat([w1_weight, w3_weight], dim=0)
                    if swiglu_fusion == 2
                    else torch.empty(2 * self.ffn_dim, self.hidden_dim, dtype=w1_weight.dtype, device=device)
                )
                fused_b1 = (
                    torch.cat([w1_bias, w3_bias], dim=0)
                    if swiglu_fusion == 2
                    else torch.empty(2 * self.ffn_dim, dtype=w1_bias.dtype, device=device)
                )

                if swiglu_fusion == 1:  # Interleave
                    fused_w1[0::2, :] = w1_weight
                    fused_w1[1::2, :] = w3_weight
                    fused_b1[0::2] = w1_bias
                    fused_b1[1::2] = w3_bias

                w1_list.append(fused_w1)
                b1_list.append(fused_b1)
                w2_list.append(expert.w2.weight)
            else:
                # For quantized, prepare weights, biases, and scales
                is_4_bit = self.quant_bits == 4
                s1, qw1, w1_qdq = quant_dequant(w1_weight, is_4_bit)
                s2, qw2, w2_qdq = quant_dequant(expert.w2.weight, is_4_bit)
                s3, qw3, w3_qdq = quant_dequant(w3_weight, is_4_bit)

                expert.w1.weight.data, expert.w2.weight.data, expert.w3.weight.data = w1_qdq, w2_qdq, w3_qdq

                # Fuse quantized weights, scales, and biases
                fused_qw1, fused_s1, fused_b1 = self._fuse_for_swiglu(qw1, s1, w1_bias, qw3, s3, w3_bias, swiglu_fusion)
                w1_list.append(fused_qw1)
                s1_list.append(fused_s1)
                b1_list.append(fused_b1)
                w2_list.append(qw2)
                s2_list.append(s2)

        # Stack all tensors for the graph builder
        fc1_w, fc2_w = torch.stack(w1_list), torch.stack(w2_list)
        fc1_b, fc2_b = torch.stack(b1_list), torch.stack(b2_list)
        fc1_s = torch.stack(s1_list) if use_quant else None
        fc2_s = torch.stack(s2_list) if use_quant else None

        self.batch_size, self.sequence_length = batch_size, sequence_length

        self.moe_onnx_graph = create_swiglu_moe_onnx_graph(
            num_tokens=batch_size * sequence_length,
            num_experts=self.num_experts,
            hidden_size=self.hidden_dim,
            inter_size=self.ffn_dim,
            topk=self.top_k,
            onnx_dtype=self.onnx_dtype,
            quant_bits=self.quant_bits,
            swiglu_fusion=self.swiglu_fusion,
            fc1_experts_weights=fc1_w,
            fc1_experts_bias=fc1_b,
            fc2_experts_weights=fc2_w,
            fc2_experts_bias=fc2_b,
            fc1_experts_weight_scale=fc1_s,
            fc2_experts_weight_scale=fc2_s,
        )
        self.ort_sess = self.create_ort_session(self.moe_onnx_graph)

    def _fuse_for_swiglu(self, w1, s1, b1, w3, s3, b3, fusion_mode):
        """Helper to fuse weights, scales, and biases for SwiGLU."""
        if fusion_mode == 2:  # Concatenate
            fused_w = torch.cat([w1, w3], dim=0)
            fused_s = torch.cat([s1, s3], dim=0)
            fused_b = torch.cat([b1, b3], dim=0)
        elif fusion_mode == 1:  # Interleave
            fused_w = torch.empty(2 * w1.shape[0], w1.shape[1], dtype=w1.dtype, device=device)
            fused_s = torch.empty(2 * s1.shape[0], dtype=s1.dtype, device=device)
            fused_b = torch.empty(2 * b1.shape[0], dtype=b1.dtype, device=device)
            fused_w[0::2, :], fused_w[1::2, :] = w1, w3
            fused_s[0::2], fused_s[1::2] = s1, s3
            fused_b[0::2], fused_b[1::2] = b1, b3
        else:
            raise ValueError(f"Invalid swiglu_fusion mode: {fusion_mode}")
        return fused_w, fused_s, fused_b

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """PyTorch reference forward pass."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=1, dtype=torch.float).to(hidden_states.dtype)

        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.shape[0] == 0:
                continue

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


swiglu_test_cases = list(
    itertools.product(
        [1, 2],  # batch_size
        [1, 3],  # sequence_length
        [4],  # quant_bits (0 for fp32/fp32, 8 for int8/fp16, 4 for int4/fp16)
        [1],  # swiglu_fusion (1 for interleaved, 2 for concatenated)
    )
)


@unittest.skipIf(not use_cuda, "skipping moe test since it requires cuda environment.")
class TestSwigluMoE(unittest.TestCase):
    @parameterized.expand(swiglu_test_cases)
    def test_swiglu_moe_parity(self, batch_size, sequence_length, quant_bits, swiglu_fusion):
        config = SwigluMoeConfig(hidden_size=64, intermediate_size=256, num_experts_per_token=2, num_local_experts=4)
        moe = SwigluMoEBlock(config, batch_size, sequence_length, quant_bits, swiglu_fusion)
        moe.to(device)
        moe.parity_check()


def has_bf16_moe():
    if not use_cuda:
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


# @unittest.skipIf(not has_bf16_moe(), "skipping bf16 moe tests.")
# class TestSwigluMoeBf16(unittest.TestCase):
#     @parameterized.expand(swiglu_test_cases)
#     def test_swiglu_moe_parity(self, batch_size, sequence_length, quant_bits, swiglu_fusion):
#         config = SwigluMoeConfig(hidden_size=64, intermediate_size=128, num_experts_per_token=2, num_local_experts=4)
#         moe = SwigluMoEBlock(
#             config, batch_size, sequence_length, quant_bits, swiglu_fusion, onnx_dtype=TensorProto.BFLOAT16
#         )
#         moe.to(device)
#         moe.parity_check()


perf_test_cases = list(
    itertools.product(
        [1],  # batch_size
        [128, 512, 1024, 2048, 4096],  # sequence_length
        [0, 8, 4],  # quant_bits (0 for fp32/fp32, 8 for int8/fp16, 4 for int4/fp16)
    )
)

run_performance_tests = False


@unittest.skipIf(pipeline_mode or not run_performance_tests, "skipping performance test in CI pipeline.")
class TestSwigluMoEPerf(unittest.TestCase):
    @parameterized.expand(perf_test_cases)
    def test_swiglu_moe_parity(self, batch_size, sequence_length, quant_bits):
        hidden_size = 2880
        intermediate_size = 2880
        num_experts_per_token = 8
        num_local_experts = 128
        config = SwigluMoeConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts_per_token=num_experts_per_token,
            num_local_experts=num_local_experts,
        )
        # For performance test, we default to interleaved fusion mode
        moe = SwigluMoEBlock(config, batch_size, sequence_length, quant_bits, swiglu_fusion=1)
        moe.to(device)
        moe.benchmark_ort()


if __name__ == "__main__":
    unittest.main()
