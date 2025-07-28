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


def value_string_of(numpy_array):
    arr = numpy_array.flatten()
    lines = ["f, ".join([str(v) for v in arr[i : min(i + 8, arr.size)]]) for i in range(0, arr.size, 8)]
    return "{\n    " + "f,\n    ".join(lines) + "f}"


def print_tensor(name, numpy_array):
    print(f"const std::vector<float> {name} = {value_string_of(numpy_array)};")


def quant_dequant(weights: torch.Tensor, is_4_bit_quantization: bool):
    """
    Performs symmetric per-column quantization and dequantization on a weight tensor.

    This implementation is a pure PyTorch replacement for the original function that
    relied on a custom tensorrt_llm operator. It supports both 8-bit (int8) and
    4-bit (quint4x2 style) quantization.

    Args:
        weights (torch.Tensor): The input weight tensor to be quantized.
        is_4_bit_quantization (bool): If True, performs 4-bit quantization. If False,
                           performs 8-bit quantization.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - scales (torch.float16): The quantization scales for each column.
            - processed_q_weight (torch.int8): The packed quantized weights. For
              4-bit mode, two 4-bit values are packed into a single int8. For
              8-bit mode, this is the standard int8 quantized tensor. It is
              transposed relative to the input weights' shape.
            - dequantized_weights (torch.Tensor): The weights after being dequantized,
              restored to the original dtype and device.
    """
    # Determine quantization bits and range based on the mode
    if is_4_bit_quantization:
        # 4-bit symmetric quantization path
        q_bits = 4
        q_max = 2 ** (q_bits - 1) - 1  # 7
        q_min = -(2 ** (q_bits - 1))  # -8

        max_abs_val = torch.max(torch.abs(weights), dim=0, keepdim=True).values
        max_abs_val[max_abs_val == 0] = 1.0
        scales = max_abs_val / q_max

        quant_weights = torch.round(weights / scales).clamp(q_min, q_max).to(torch.int8)

        # Pack two 4-bit integers into a single int8
        q_weights_t = quant_weights.T.contiguous()
        shape = q_weights_t.shape
        q_weights_t_reshaped = q_weights_t.view(shape[0], shape[1] // 2, 2)
        lower_nibble = q_weights_t_reshaped[..., 0]
        upper_nibble = q_weights_t_reshaped[..., 1]
        processed_q_weight = (lower_nibble & 0x0F) | (upper_nibble << 4)

    else:
        # 8-bit symmetric quantization path
        q_bits = 8
        q_max = 2 ** (q_bits - 1) - 1  # 127
        q_min = -(2 ** (q_bits - 1))  # -128

        max_abs_val = torch.max(torch.abs(weights), dim=0, keepdim=True).values
        max_abs_val[max_abs_val == 0] = 1.0
        scales = max_abs_val / q_max

        quant_weights = torch.round(weights / scales).clamp(q_min, q_max).to(torch.int8)

        # For 8-bit, the processed weights are just the transposed quantized weights (no packing)
        processed_q_weight = quant_weights.T.contiguous()

    # Dequantize the weights to verify and return for PyTorch-side parity check
    dequantized_weights = quant_weights.to(weights.dtype) * scales.to(weights.dtype)

    return (scales.squeeze(0).to(torch.float16), processed_q_weight, dequantized_weights.T.to(device=weights.device))


def create_moe_onnx_graph(
    sequence_length,
    num_experts,
    hidden_size,
    inter_size,
    fc1_experts_weights,
    fc1_experts_bias,
    fc2_experts_weights,
    fc2_experts_bias,
    ort_dtype,
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

    torch_type = torch.float16 if ort_dtype == TensorProto.FLOAT16 else torch.float32

    initializers = [
        helper.make_tensor(
            "fc1_experts_weights",
            ort_dtype,
            fc1_shape,
            fc1_experts_weights.to(torch_type).flatten().tolist(),
            raw=False,
        ),
        helper.make_tensor(
            "fc2_experts_weights",
            ort_dtype,
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
                ort_dtype,
                fc1_bias_shape,
                fc1_experts_bias.to(torch_type).flatten().tolist(),
                raw=False,
            ),
            helper.make_tensor(
                "fc2_experts_bias",
                ort_dtype,
                fc2_bias_shape,
                fc2_experts_bias.to(torch_type).flatten().tolist(),
                raw=False,
            ),
        ]
    )

    graph_inputs = [
        helper.make_tensor_value_info("input", ort_dtype, [sequence_length, hidden_size]),
    ]

    graph_inputs.append(
        helper.make_tensor_value_info(
            "router_probs",
            ort_dtype,
            [sequence_length, num_experts],
        )
    )

    graph_outputs = [
        helper.make_tensor_value_info("output", ort_dtype, [sequence_length, hidden_size]),
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
    ort_dtype,
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

    torch_type = torch.float16 if ort_dtype == TensorProto.FLOAT16 else torch.float32

    initializers = [
        helper.make_tensor(
            "fc1_experts_weights",
            ort_dtype,
            fc1_shape,
            fc1_experts_weights.to(torch_type).flatten().tolist(),
            raw=False,
        ),
        helper.make_tensor(
            "fc2_experts_weights",
            ort_dtype,
            fc2_shape,
            fc2_experts_weights.to(torch_type).flatten().tolist(),
            raw=False,
        ),
        helper.make_tensor(
            "fc3_experts_weights",
            ort_dtype,
            fc3_shape,
            fc3_experts_weights.to(torch_type).flatten().tolist(),
            raw=False,
        ),
    ]

    graph_inputs = [
        helper.make_tensor_value_info("input", ort_dtype, [sequence_length, hidden_size]),
    ]

    graph_inputs.append(
        helper.make_tensor_value_info(
            "router_probs",
            ort_dtype,
            [sequence_length, num_experts],
        )
    )

    graph_outputs = [
        helper.make_tensor_value_info("output", ort_dtype, [sequence_length, hidden_size]),
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
    ort_dtype,
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

    torch_type = torch.float16 if ort_dtype == TensorProto.FLOAT16 else torch.float32
    numpy_type = numpy.float16 if ort_dtype == TensorProto.FLOAT16 else numpy.float32
    weight_numpy_type = numpy.uint8 if use_quant else numpy_type
    weight_onnx_type = TensorProto.UINT8 if use_quant else ort_dtype

    initializers = [
        helper.make_tensor(
            "fc1_experts_weights",
            weight_onnx_type,
            fc1_shape,
            fc1_experts_weights.flatten().detach().numpy().astype(weight_numpy_type).tolist(),
            raw=False,
        ),
        helper.make_tensor(
            "fc2_experts_weights",
            weight_onnx_type,
            fc2_shape,
            fc2_experts_weights.flatten().detach().numpy().astype(weight_numpy_type).tolist(),
            raw=False,
        ),
        helper.make_tensor(
            "fc3_experts_weights",
            weight_onnx_type,
            fc3_shape,
            fc3_experts_weights.flatten().detach().numpy().astype(weight_numpy_type).tolist(),
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
                    ort_dtype,
                    fc1_scale_shape,
                    fc1_scales.to(torch_type).flatten().tolist(),
                    raw=False,
                ),
                helper.make_tensor(
                    "fc2_scales",
                    ort_dtype,
                    fc2_scale_shape,
                    fc2_scales.to(torch_type).flatten().tolist(),
                    raw=False,
                ),
                helper.make_tensor(
                    "fc3_scales",
                    ort_dtype,
                    fc3_scale_shape,
                    fc3_scales.to(torch_type).flatten().tolist(),
                    raw=False,
                ),
            ]
        )

    graph_inputs = [
        helper.make_tensor_value_info("input", ort_dtype, [sequence_length, hidden_size]),
    ]

    graph_inputs.append(
        helper.make_tensor_value_info(
            "router_probs",
            ort_dtype,
            [sequence_length, num_experts],
        )
    )

    graph_outputs = [
        helper.make_tensor_value_info("output", ort_dtype, [sequence_length, hidden_size]),
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
    def __init__(self, quant_bits=0):
        super().__init__()
        self.quant_bits = quant_bits
        self.ort_dtype = TensorProto.FLOAT16 if self.quant_bits > 0 else TensorProto.FLOAT
        self.np_type = numpy.float16 if self.ort_dtype == TensorProto.FLOAT16 else numpy.float32

    def create_ort_session(self, moe_onnx_graph):
        from onnxruntime import InferenceSession, SessionOptions  # noqa: PLC0415

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
            "input": numpy.ascontiguousarray(hidden_states.detach().numpy().astype(self.np_type)),
            "router_probs": numpy.ascontiguousarray(router_logits.detach().numpy().astype(self.np_type)),
        }

        ort_output = None
        if self.ort_sess is not None:
            if not iobinding:
                ort_output = self.ort_sess.run(None, ort_inputs)
                return torch.tensor(ort_output).reshape(batch_size, sequence_length, -1)  # , router_logits
            else:
                self.ort_run_with_iobinding(ort_inputs)
                return None

        return None

    def ort_run_with_iobinding(self, ort_inputs, repeat=1000):
        iobinding = self.ort_sess.io_binding()
        device_id = torch.cuda.current_device()

        iobinding.bind_input(
            name="input",
            device_type="cuda",
            device_id=device_id,
            element_type=self.np_type,
            shape=ort_inputs["input"].shape,
            buffer_ptr=onnxruntime.OrtValue.ortvalue_from_numpy(ort_inputs["input"], "cuda", device_id).data_ptr(),
        )

        iobinding.bind_input(
            name="router_probs",
            device_type="cuda",
            device_id=device_id,
            element_type=self.np_type,
            shape=ort_inputs["router_probs"].shape,
            buffer_ptr=onnxruntime.OrtValue.ortvalue_from_numpy(
                ort_inputs["router_probs"], "cuda", device_id
            ).data_ptr(),
        )

        iobinding.bind_output(
            name="output",
            device_type="cuda",
            device_id=device_id,
            element_type=self.np_type,
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

        import time  # noqa: PLC0415

        s = time.time()
        for _ in range(repeat):
            iobinding.synchronize_inputs()
            self.ort_sess.run_with_iobinding(iobinding)
            iobinding.synchronize_outputs()
        e = time.time()
        print(f"MoE cuda kernel time: {(e - s) / repeat * 1000} ms")

    def parity_check(self, atol=None, rtol=None):
        hidden_state = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim)
        torch_output = self.forward(hidden_state)
        ort_output = self.ort_forward(hidden_state)

        if atol is None:
            atol = 1e-2 if self.quant_bits == 0 else 2.0

        if rtol is None:
            rtol = 1e-5 if self.quant_bits == 0 else 1e-3

        if ort_output is not None:
            dtype_str = "FP32" if self.quant_bits == 0 else "FP16"
            print(
                f"name: {self.__class__.__name__}, quant_bits: {self.quant_bits}, dtype: {dtype_str},"
                f" batch: {self.batch_size}, seq_len: {self.sequence_length},"
                f" max_diff: {(torch_output - ort_output).abs().max()}"
            )
            torch.testing.assert_close(
                ort_output.to(torch.float32), torch_output.to(torch.float32), rtol=rtol, atol=atol
            )

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
            self.ort_dtype,
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
            self.ort_dtype,
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

    def __init__(self, config, batch_size, sequence_length, quant_bits=0):
        super().__init__(quant_bits)
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
                w1_scale, pre_qweight1, w1_qdq = quant_dequant(self.experts[i].w1.weight.T, is_4_bit)
                w2_scale, pre_qweight2, w2_qdq = quant_dequant(self.experts[i].w2.weight.T, is_4_bit)
                w3_scale, pre_qweight3, w3_qdq = quant_dequant(self.experts[i].w3.weight.T, is_4_bit)

                self.experts[i].w1.weight.data = w1_qdq
                self.experts[i].w2.weight.data = w2_qdq
                self.experts[i].w3.weight.data = w3_qdq

                # Transpose quantized weights to match the expected ONNX layout
                w1_list.append(pre_qweight1.T)
                w2_list.append(pre_qweight2.T)
                w3_list.append(pre_qweight3.T)
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
            self.ort_dtype,
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
            yield batch_size, sequence_length, 0


# Test cases for Phi-3 MoE.
# We test three modes: no quantization, 8-bit, and 4-bit.
phi3_test_params = list(
    itertools.product(
        [1, 4],  # batch_size
        [1, 32],  # sequence_length
        [0, 8],  # quant_bits (0 for fp32/fp32, 8 for int8/fp16, 4 for int4/fp16)
    )
)


class TestSwitchMoE(unittest.TestCase):
    @parameterized.expand(small_test_cases())
    def test_switch_moe_parity(self, batch_size, sequence_length, quant_bits):
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
    @parameterized.expand([(b, s, q) for b, s, q in small_test_cases() if q == 0])  # only run non-quantized
    def test_mixtral_moe_parity(self, batch_size, sequence_length, quant_bits):
        config = MixtralConfig(hidden_size=256, intermediate_size=1024)
        mixtral_moe = MixtralSparseMoeBlock(config, batch_size, sequence_length)
        mixtral_moe.parity_check()
        # mixtral_moe.benchmark_ort()


class TestPhiMoE(unittest.TestCase):
    @parameterized.expand(phi3_test_params)
    def test_phi3_moe_parity(self, batch_size, sequence_length, quant_bits):
        config = PhiMoEConfig(hidden_size=256, intermediate_size=1024)
        phi3_moe = PhiMoESparseMoeBlock(config, batch_size, sequence_length, quant_bits)
        phi3_moe.parity_check()
        # phi3_moe.benchmark_ort()


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


class SwigluMlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.w1 = nn.Linear(self.hidden_dim, 2 * self.intermediate_size, bias=True)
        self.w2 = nn.Linear(self.intermediate_size, self.hidden_dim, bias=True)

    def swiglu(self, x: torch.Tensor):
        dim = x.shape[-1]
        x = x.view(-1, dim // 2, 2)
        x_glu, x_linear = x[..., 0], x[..., 1]
        y = x_glu * torch.sigmoid(1.702 * x_glu) * (x_linear + 1)
        return y

    def forward(self, x):
        y = self.swiglu(self.w1(x))
        y = self.w2(y)
        return y


def create_swiglu_moe_onnx_graph(
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
    inter_size: int,
    topk: int,
    ort_dtype: int,
    quant_bits: int,
    fc1_experts_weights: torch.Tensor,
    fc1_experts_bias: torch.Tensor,
    fc2_experts_weights: torch.Tensor,
    fc2_experts_bias: torch.Tensor,
    fc1_experts_weight_scale: torch.Tensor = None,
    fc2_experts_weight_scale: torch.Tensor = None,
):
    use_quant = quant_bits > 0
    op_name = "QMoE" if use_quant else "MoE"

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

    nodes = [
        helper.make_node(
            op_name,
            inputs,
            ["output"],
            "MoE_0",
            k=topk,
            normalize_routing_weights=1,
            activation_type="swiglu",
            domain="com.microsoft",
        ),
    ]

    if use_quant:
        nodes[0].attribute.extend([helper.make_attribute("expert_weight_bits", quant_bits)])

    components = 2 if quant_bits == 4 else 1
    fc1_weight_shape = [num_experts, hidden_size, 2 * inter_size // components]
    fc1_bias_shape = [num_experts, 2 * inter_size]
    fc1_experts_weight_scale_shape = [num_experts, 2 * inter_size]

    fc2_weight_shape = [num_experts, inter_size, hidden_size // components]
    fc2_bias_shape = [num_experts, hidden_size]
    fc2_experts_weight_scale_shape = [num_experts, hidden_size]

    torch_type = torch.float16 if ort_dtype == TensorProto.FLOAT16 else torch.float32
    numpy_type = numpy.float16 if ort_dtype == TensorProto.FLOAT16 else numpy.float32
    weight_numpy_type = numpy.uint8 if use_quant else numpy_type
    weight_onnx_type = TensorProto.UINT8 if use_quant else ort_dtype

    initializers = [
        helper.make_tensor(
            "fc1_experts_weights",
            weight_onnx_type,
            fc1_weight_shape,
            fc1_experts_weights.flatten().detach().numpy().astype(weight_numpy_type).tolist()
            if use_quant
            else fc1_experts_weights.to(torch_type).flatten().tolist(),
            raw=False,
        ),
        helper.make_tensor(
            "fc1_experts_bias",
            ort_dtype,
            fc1_bias_shape,
            fc1_experts_bias.to(torch_type).flatten().tolist(),
            raw=False,
        ),
        helper.make_tensor(
            "fc2_experts_weights",
            weight_onnx_type,
            fc2_weight_shape,
            fc2_experts_weights.flatten().detach().numpy().astype(weight_numpy_type).tolist()
            if use_quant
            else fc2_experts_weights.to(torch_type).flatten().tolist(),
            raw=False,
        ),
        helper.make_tensor(
            "fc2_experts_bias",
            ort_dtype,
            fc2_bias_shape,
            fc2_experts_bias.to(torch_type).flatten().tolist(),
            raw=False,
        ),
    ]

    if use_quant:
        initializers.extend(
            [
                helper.make_tensor(
                    "fc1_experts_weight_scale",
                    ort_dtype,
                    fc1_experts_weight_scale_shape,
                    fc1_experts_weight_scale.to(torch_type).flatten().tolist(),
                    raw=False,
                ),
                helper.make_tensor(
                    "fc2_experts_weight_scale",
                    ort_dtype,
                    fc2_experts_weight_scale_shape,
                    fc2_experts_weight_scale.to(torch_type).flatten().tolist(),
                    raw=False,
                ),
            ]
        )

    graph_inputs = [
        helper.make_tensor_value_info("input", ort_dtype, [num_tokens, hidden_size]),
    ]

    graph_inputs.append(
        helper.make_tensor_value_info(
            "router_probs",
            ort_dtype,
            [num_tokens, num_experts],
        )
    )

    graph_outputs = [
        helper.make_tensor_value_info("output", ort_dtype, [num_tokens, hidden_size]),
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


class SwigluMoEBlock(SparseMoeBlockORTHelper):
    def __init__(self, config: SwigluMoeConfig, batch_size: int, sequence_length: int, quant_bits: int = 0):
        super().__init__(quant_bits)
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_token
        use_quant = self.quant_bits > 0

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=True)

        self.experts = nn.ModuleList([SwigluMlp(config) for _ in range(self.num_experts)])

        weight_1_list, weight_2_list = [], []
        bias_1_list, bias_2_list = [], []
        scale_1_list, scale_2_list = [], []

        for i in range(self.num_experts):
            bias_1_list.append(self.experts[i].w1.bias)
            bias_2_list.append(self.experts[i].w2.bias)
            if not use_quant:
                weight_1_list.append(self.experts[i].w1.weight)
                weight_2_list.append(self.experts[i].w2.weight)
            else:
                is_4_bit = self.quant_bits == 4
                # Pass the transposed weight to quant_dequant to get correct scales,
                # then transpose the resulting quantized weight back to the expected layout.
                scale1, pre_qweight1, w1_qdq = quant_dequant(self.experts[i].w1.weight.T, is_4_bit)
                scale2, pre_qweight2, w2_qdq = quant_dequant(self.experts[i].w2.weight.T, is_4_bit)

                self.experts[i].w1.weight.data = w1_qdq
                self.experts[i].w2.weight.data = w2_qdq

                weight_1_list.append(pre_qweight1.T)
                weight_2_list.append(pre_qweight2.T)
                scale_1_list.append(scale1)
                scale_2_list.append(scale2)

        self.moe_experts_weight1 = torch.stack(weight_1_list, dim=0)
        self.moe_experts_weight2 = torch.stack(weight_2_list, dim=0)

        self.moe_experts_bias1 = torch.stack(bias_1_list, dim=0)
        self.moe_experts_bias2 = torch.stack(bias_2_list, dim=0)

        moe_experts_weight_scale1 = torch.stack(scale_1_list, dim=0) if use_quant else None
        moe_experts_weight_scale2 = torch.stack(scale_2_list, dim=0) if use_quant else None

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.moe_onnx_graph = create_swiglu_moe_onnx_graph(
            num_tokens=self.batch_size * self.sequence_length,
            num_experts=self.num_experts,
            hidden_size=self.hidden_dim,
            inter_size=self.ffn_dim,
            topk=self.top_k,
            ort_dtype=self.ort_dtype,
            quant_bits=self.quant_bits,
            fc1_experts_weights=self.moe_experts_weight1,
            fc1_experts_bias=self.moe_experts_bias1,
            fc2_experts_weights=self.moe_experts_weight2,
            fc2_experts_bias=self.moe_experts_bias2,
            fc1_experts_weight_scale=moe_experts_weight_scale1,
            fc2_experts_weight_scale=moe_experts_weight_scale2,
        )

        self.ort_sess = self.create_ort_session(self.moe_onnx_graph)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)  # router_logits shape is (batch * sequence_length, num_experts)
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)

        routing_weights = F.softmax(routing_weights, dim=1, dtype=torch.float)

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
        return final_hidden_states


swiglu_test_params = list(
    itertools.product(
        [1, 4],  # batch_size
        [1, 32],  # sequence_length
        [0, 8],  # quant_bits (0 for fp32/fp32, 8 for int8/fp16, 4 for int4/fp16)
    )
)


class TestSwigluMoE(unittest.TestCase):
    @parameterized.expand(swiglu_test_params)
    def test_swiglu_moe_parity(self, batch_size, sequence_length, quant_bits):
        config = SwigluMoeConfig(hidden_size=128, intermediate_size=512, num_experts_per_token=1, num_local_experts=4)
        moe = SwigluMoEBlock(config, batch_size, sequence_length, quant_bits)
        moe.parity_check()


if __name__ == "__main__":
    unittest.main()
