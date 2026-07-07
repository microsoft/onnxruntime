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
import math
import os
import time
import unittest
from collections import OrderedDict

import numpy
import torch
import torch.nn.functional as F
from cuda_plugin_ep_helper import get_cuda_provider_name
from onnx import TensorProto, helper
from parameterized import parameterized
from torch import nn

import onnxruntime
from onnxruntime import InferenceSession, SessionOptions
from onnxruntime.quantization import CudaQuantizer

# Reduces number of tests to run for faster pipeline checks
pipeline_mode = os.getenv("PIPELINE_MODE", "1") == "1"

onnxruntime.preload_dlls()


# Determine the execution provider and device based on CUDA availability.
cuda_provider = get_cuda_provider_name()
use_cuda = cuda_provider is not None
device = torch.device("cuda:0" if use_cuda else "cpu")


def get_ort_provider():
    if not use_cuda:
        return ["CPUExecutionProvider"]

    return [cuda_provider]


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


def print_diff_statistics(diff_tensor: torch.Tensor, prefix: str = ""):
    """
    Print percentile statistics (75%, 95%, 99%) for a difference tensor.
    This helps assess parity quality beyond just max difference.

    Args:
        diff_tensor: Tensor containing absolute differences between expected and actual outputs.
        prefix: Optional prefix string for the output message.
    """
    diff_flat = diff_tensor.flatten().float()
    if diff_flat.numel() == 0:
        print(f"{prefix}Diff statistics: empty tensor")
        return

    # Compute percentiles
    sorted_diff, _ = torch.sort(diff_flat)
    n = sorted_diff.numel()

    p75_idx = min(int(n * 0.75), n - 1)
    p95_idx = min(int(n * 0.95), n - 1)
    p99_idx = min(int(n * 0.99), n - 1)

    p75 = sorted_diff[p75_idx].item()
    p95 = sorted_diff[p95_idx].item()
    p99 = sorted_diff[p99_idx].item()
    max_val = sorted_diff[-1].item()
    mean_val = diff_flat.mean().item()

    print(
        f"{prefix}Diff stats - mean: {mean_val:.6f}, p75: {p75:.6f}, p95: {p95:.6f}, p99: {p99:.6f}, max: {max_val:.6f}"
    )


def quant_dequant(weights, is_4_bit_quantization: bool = True):
    bits = 4 if is_4_bit_quantization else 8
    n, k = weights.shape
    block_size = min(128, k)
    block_per_k = (k + block_size - 1) // block_size

    q_weight, scales = CudaQuantizer.matmulnbits_blockwise_quantize(
        weights, bits, block_size, abs_scales=True, flatten_qweight=False
    )
    processed_q_weight, _ = CudaQuantizer.qmoe_prepacked_blockwise_quantize(weights, bits, block_size, abs_scales=True)

    q_weight = q_weight.to(weights.device)
    scales = scales.to(weights.device)

    if is_4_bit_quantization:
        q_low = q_weight & 0x0F
        q_high = (q_weight >> 4) & 0x0F
        q_unpacked = torch.stack((q_low, q_high), dim=-1).view(n, block_per_k, -1)[:, :, :block_size]
        q_unpacked = q_unpacked.to(weights.dtype)
        dequantized = (q_unpacked - 8.0) * scales.unsqueeze(-1)
    else:
        q_unpacked = q_weight.to(weights.dtype)
        dequantized = (q_unpacked - 128.0) * scales.unsqueeze(-1)

    scale_flat = scales.mean(dim=1)
    return scale_flat.to(torch.float16), processed_q_weight.to(weights.device), dequantized.view(n, k)


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
    normalize_routing_weights=0,
):
    use_quant = quant_bits > 0
    use_fused_swiglu = fc3_experts_weights is None  # Fused SwiGLU: FC1 contains both gate and value
    if use_quant:
        assert fc1_experts_weights.dtype == torch.uint8
        assert fc2_experts_weights.dtype == torch.uint8
        if not use_fused_swiglu:
            assert fc3_experts_weights.dtype == torch.uint8
            assert fc3_scales is not None
            assert fc3_scales.dtype == torch.float16
        assert fc1_scales is not None
        assert fc2_scales is not None
        assert fc1_scales.dtype == torch.float16
        assert fc2_scales.dtype == torch.float16

    op_name = "QMoE" if use_quant else "MoE"
    if use_fused_swiglu:
        # Fused SwiGLU: FC1 contains both gate and value, no separate FC3
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
            ]
            if use_quant
            else [
                "input",
                "router_probs",
                "fc1_experts_weights",
                "",
                "fc2_experts_weights",
            ]
        )
    else:
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
            normalize_routing_weights=normalize_routing_weights,
            use_sparse_mixer=0,  # Align with Python Reference (Softmax)
            activation_type="silu" if not use_fused_swiglu else "swiglu",
            swiglu_fusion=2 if use_fused_swiglu else 0,  # 2 = fused, not interleaved
            domain="com.microsoft",
        ),
    ]

    if use_quant:
        nodes[0].attribute.extend([helper.make_attribute("expert_weight_bits", quant_bits)])

    # Use actual tensor shapes instead of hardcoding
    fc1_shape = list(fc1_experts_weights.shape)
    fc2_shape = list(fc2_experts_weights.shape)

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
    ]

    # Add FC3 only if not fused
    if not use_fused_swiglu and fc3_experts_weights is not None:
        fc3_shape = list(fc3_experts_weights.shape)
        initializers.append(
            helper.make_tensor(
                "fc3_experts_weights",
                weight_onnx_type,
                fc3_shape,
                fc3_experts_weights.flatten().detach().cpu().numpy().astype(weight_numpy_type).tolist(),
                raw=False,
            )
        )

    if use_quant:
        fc1_scale_shape = list(fc1_scales.shape)
        fc2_scale_shape = list(fc2_scales.shape)
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
            ]
        )
        # Add FC3 scales only if not fused
        if not use_fused_swiglu and fc3_scales is not None:
            fc3_scale_shape = list(fc3_scales.shape)
            initializers.append(
                helper.make_tensor(
                    "fc3_scales",
                    onnx_dtype,
                    fc3_scale_shape,
                    fc3_scales.to(torch_dtype).flatten().tolist(),
                    raw=False,
                )
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
        sess_options = SessionOptions()
        sess_options.log_severity_level = 2
        providers = get_ort_provider()

        try:
            ort_session = InferenceSession(moe_onnx_graph, sess_options, providers=providers)
        except Exception as e:
            print(f"Failed to create ONNX Runtime session with provider {providers}: {e}")
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
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # Determine the correct torch dtype from the onnx_dtype
        torch_dtype = onnx_to_torch_type_map[self.onnx_dtype]

        hidden_state = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim).to(
            device=device, dtype=torch_dtype
        )

        if torch_dtype in [torch.float16, torch.bfloat16]:
            self.to(torch_dtype)

        torch_output = self.forward(hidden_state)
        ort_output = self.ort_forward(hidden_state)

        dtype_str = ort_dtype_name_map[self.onnx_dtype]

        # Maps "ort_type:quant_bits" to (atol, rtol)
        ort_dtype_quant_bits_tolerance_map = {
            "FP32:0": (5e-3, 1e-3),
            "FP16:0": (0.3, 0.05),
            "FP16:4": (0.5, 1e-2),
            "FP16:8": (0.5, 1e-2),
            "BF16:0": (1.0, 1e-2),
            "BF16:4": (30.0, 1e-1),
            "BF16:8": (20.0, 1e-1),
        }

        atol, rtol = ort_dtype_quant_bits_tolerance_map[f"{dtype_str}:{self.quant_bits}"]
        if ort_output is not None:
            diff = (torch_output.cpu() - ort_output.cpu()).abs()
            print(
                f"name: {self.__class__.__name__}, quant_bits: {self.quant_bits}, dtype: {dtype_str},"
                f" batch: {self.batch_size}, seq_len: {self.sequence_length},"
                f" max_diff: {diff.max()}"
            )
            # Print percentile statistics for better parity assessment
            print_diff_statistics(diff, prefix=f"  [{self.__class__.__name__}] ")
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

    def __init__(self, config, batch_size, sequence_length, quant_bits=0, onnx_dtype=None, normalize_routing_weights=0):
        super().__init__(quant_bits, onnx_dtype)
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.router_jitter_noise = config.router_jitter_noise
        self.normalize_routing_weights = normalize_routing_weights
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

        # Combine FC1 (gate) and FC3 (value) for fused SwiGLU to avoid separate FC3 input
        # This triggers swiglu_fusion=2 mode (fused, not interleaved) - concat along N dimension
        # Only apply for quantized weights to avoid separate FC3 scales issue
        if use_quant:
            # Weights: [E, K, N/pack] -> [E, K, 2*N/pack] - concat along dim=2 (N axis)
            self.moe_experts_weight1 = torch.cat([self.moe_experts_weight1, self.moe_experts_weight3], dim=2)
            self.moe_experts_weight3 = None
            # Scales: [E, N] -> [E, 2*N] - concat along dim=1
            moe_experts_weight_scale1 = torch.cat([moe_experts_weight_scale1, moe_experts_weight_scale3], dim=1)
            moe_experts_weight_scale3 = None
        # For non-quant, keep fc1/fc2/fc3 separate

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.moe_onnx_graph = create_phi_moe_onnx_graph(
            self.batch_size * self.sequence_length,
            self.num_experts,
            self.hidden_dim,
            self.ffn_dim,
            self.moe_experts_weight1,
            self.moe_experts_weight2,
            self.moe_experts_weight3,  # Now None, triggering fused SwiGLU path
            self.top_k,
            self.onnx_dtype,
            self.quant_bits,
            moe_experts_weight_scale1,
            moe_experts_weight_scale2,
            moe_experts_weight_scale3,  # Now None
            normalize_routing_weights,
        )

        self.ort_sess = self.create_ort_session(self.moe_onnx_graph)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        if self.normalize_routing_weights:
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            # we cast back to the input dtype
            routing_weights = routing_weights.to(hidden_states.dtype)
        else:
            # ORT LaunchSoftmaxTopK does not support jitter or masked sampling.
            # It performs Softmax -> TopK.
            # To ensure parity, we must match ORT's logic here.
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
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


@unittest.skipIf(not use_cuda, "skipping moe test since it requires cuda environment.")
class TestSwitchMoE(unittest.TestCase):
    @parameterized.expand(small_test_cases())
    def test_switch_moe_parity(self, batch_size, sequence_length):
        switch_moe = SwitchMoE(
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_experts=8,
            in_features=256,
            hidden_features=1024,
            out_features=256,
        )
        switch_moe.to(device)
        switch_moe.parity_check()


# quant_bits (0 for fp32/fp32, 8 for int8/fp16, 4 for int4/fp16)
# since qMoE test requires tensorrt_llm for quant_dequant. We disable it in CI pipeline to avoid extra dependency.
quant_bits_list = [0] if pipeline_mode else [0, 8, 4]


@unittest.skipIf(not use_cuda, "skipping moe test since it requires cuda environment.")
class TestMixtralMoE(unittest.TestCase):
    @parameterized.expand(small_test_cases())
    def test_mixtral_moe_parity(self, batch_size, sequence_length):
        config = MixtralConfig(hidden_size=256, intermediate_size=1024)
        mixtral_moe = MixtralSparseMoeBlock(config, batch_size, sequence_length)
        mixtral_moe.to(device)
        mixtral_moe.parity_check()


phi3_test_cases = list(
    itertools.product(
        [1, 4],  # batch_size
        [1, 32],  # sequence_length
        [0],  # quant_bits (0 for fp32/fp32, 8 for int8/fp16, 4 for int4/fp16)
        [TensorProto.FLOAT, TensorProto.FLOAT16],  # onnx type, None mean fp32 for bits = 0, fp16 for bits > 0
        [True],  # normalize_routing_weights
    )
)


@unittest.skipIf(not use_cuda, "skipping moe test since it requires cuda environment.")
class TestPhiMoE(unittest.TestCase):
    @parameterized.expand(phi3_test_cases)
    def test_phi3_moe_parity(self, batch_size, sequence_length, quant_bits, onnx_type, normalize_routing_weights):
        config = PhiMoEConfig(hidden_size=256, intermediate_size=1024)
        phi3_moe = PhiMoESparseMoeBlock(
            config, batch_size, sequence_length, quant_bits, onnx_type, normalize_routing_weights
        )
        phi3_moe.to(device)
        phi3_moe.parity_check()


phi3_qmoe_test_cases = list(
    itertools.product(
        [1, 4],  # batch_size
        [1, 8],  # sequence_length; keeps expanded rows <= 64 to exercise the INT4 MoE GEMV path
        [TensorProto.FLOAT16],  # onnx type, None mean fp32 for bits = 0, fp16 for bits > 0
        [True],  # normalize_routing_weights
    )
)


@unittest.skipIf(not use_cuda, "skipping moe test since it requires cuda environment.")
class TestPhiQMoE(unittest.TestCase):
    @parameterized.expand(phi3_qmoe_test_cases)
    def test_phi3_qmoe_4bits(self, batch_size, sequence_length, onnx_type, normalize_routing_weights):
        config = PhiMoEConfig(hidden_size=128, intermediate_size=256)
        phi3_moe = PhiMoESparseMoeBlock(config, batch_size, sequence_length, 4, onnx_type, normalize_routing_weights)
        phi3_moe.to(device)
        phi3_moe.parity_check()

    @parameterized.expand(phi3_qmoe_test_cases)
    def test_phi3_qmoe_8bits(self, batch_size, sequence_length, onnx_type, normalize_routing_weights):
        config = PhiMoEConfig(hidden_size=128, intermediate_size=256)
        phi3_moe = PhiMoESparseMoeBlock(config, batch_size, sequence_length, 8, onnx_type, normalize_routing_weights)
        phi3_moe.to(device)
        phi3_moe.parity_check()


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
        swiglu_fusion=1,
        swiglu_limit=7.0,
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts_per_token = num_experts_per_token
        self.num_local_experts = num_local_experts
        self.swiglu_fusion = swiglu_fusion
        self.swiglu_limit = swiglu_limit
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_beta = swiglu_beta


# SwiGLU reference. The general form is:
#     swiglu = G * sigmoid(alpha * G) * (L + beta)
#     G = clamp(g, max=limit), L = clamp(l, min=-limit, max=limit)
# where g (gate) and l (linear/value) are the two halves of the FC1 output.
#
# Two common configurations:
#   - GPT-OSS SwiGLU:  alpha=1.702, beta=1.0, limit=7.0, interleaved layout (swiglu_fusion=1)
#   - Standard SwiGLU: alpha=1.0,   beta=0.0, limit=None (no clamp), concatenated layout
#     (swiglu_fusion=2). This reduces to y = silu(gate) * value, which is what
#     Llama/Gemma-style MoE uses.
#
# When interleaved is True the FC1 output is laid out as [g0, l0, g1, l1, ...];
# otherwise it is concatenated as [g0, g1, ..., l0, l1, ...].
def swiglu(
    x: torch.Tensor,
    alpha: float = 1.702,
    beta: float = 1.0,
    limit: float = 7.0,
    interleaved: bool = True,
):
    dim = x.shape[-1]
    if interleaved:
        x = x.view(-1, dim // 2, 2)
        x_glu, x_linear = x[..., 0], x[..., 1]
    else:
        x_glu, x_linear = x[..., : dim // 2], x[..., dim // 2 :]

    if limit is not None:
        x_glu = x_glu.clamp(max=limit)
        x_linear = x_linear.clamp(min=-limit, max=limit)

    y = x_glu * torch.sigmoid(alpha * x_glu) * (x_linear + beta)
    return y


class SwigluMlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.w1 = nn.Linear(self.hidden_dim, 2 * self.intermediate_size, bias=True)
        self.w2 = nn.Linear(self.intermediate_size, self.hidden_dim, bias=True)
        self.alpha = config.swiglu_alpha
        self.beta = config.swiglu_beta
        self.limit = config.swiglu_limit
        # swiglu_fusion=1 means interleaved gate/value; 2 means concatenated.
        self.interleaved = config.swiglu_fusion == 1

    def forward(self, x):
        x1 = self.w1(x)
        y = swiglu(x1, self.alpha, self.beta, self.limit, self.interleaved)
        y = self.w2(y)
        return y


# Note that the weight shape might not match the tensor shape in legacy operator spec.
def make_onnx_intializer(name: str, tensor: torch.Tensor, shape, onnx_dtype):
    torch_dtype = onnx_to_torch_type_map[onnx_dtype]
    if torch_dtype == torch.bfloat16:
        numpy_vals_uint16 = tensor.to(torch.bfloat16).cpu().view(torch.uint16).numpy()
        initializer = helper.make_tensor(
            name=name,
            data_type=TensorProto.BFLOAT16,
            dims=shape,
            vals=numpy_vals_uint16.tobytes(),
            raw=True,
        )
    else:
        initializer = helper.make_tensor(
            name=name,
            data_type=onnx_dtype,
            dims=shape,
            vals=tensor.flatten().detach().cpu().numpy().astype(numpy.uint8).tolist()
            if onnx_dtype == TensorProto.UINT8
            else tensor.detach().to(torch_dtype).flatten().tolist(),
            raw=False,
        )
    return initializer


def create_swiglu_moe_onnx_graph(
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
    inter_size: int,
    topk: int,
    onnx_dtype: int,
    quant_bits: int,
    fc1_experts_weights: torch.Tensor,
    fc1_experts_bias: torch.Tensor,
    fc2_experts_weights: torch.Tensor,
    fc2_experts_bias: torch.Tensor,
    fc1_experts_weight_scale: torch.Tensor = None,
    fc2_experts_weight_scale: torch.Tensor = None,
    swiglu_fusion: int = 1,
    activation_alpha: float = 1.702,
    activation_beta: float = 1.0,
    swiglu_limit: float = 7.0,
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
            activation_alpha=activation_alpha,
            activation_beta=activation_beta,
            swiglu_fusion=swiglu_fusion,
            domain="com.microsoft",
        ),
    ]

    # Only emit swiglu_limit when a finite clamp limit is requested. Omitting the
    # attribute selects the default (infinity / no clamp), which is required for
    # standard SwiGLU.
    if swiglu_limit is not None and math.isfinite(swiglu_limit):
        nodes[0].attribute.extend([helper.make_attribute("swiglu_limit", float(swiglu_limit))])

    if use_quant:
        nodes[0].attribute.extend([helper.make_attribute("expert_weight_bits", quant_bits)])

    components = 2 if quant_bits == 4 else 1
    fc1_weight_shape = [num_experts, 2 * inter_size, hidden_size // components]
    fc1_bias_shape = [num_experts, 2 * inter_size]
    fc1_experts_weight_scale_shape = [num_experts, 2 * inter_size]

    fc2_weight_shape = [num_experts, hidden_size, inter_size // components]
    fc2_bias_shape = [num_experts, hidden_size]
    fc2_experts_weight_scale_shape = [num_experts, hidden_size]

    weight_onnx_type = TensorProto.UINT8 if use_quant else onnx_dtype

    torch_dtype = onnx_to_torch_type_map[onnx_dtype]
    weight_torch_dtype = onnx_to_torch_type_map[weight_onnx_type]

    initializers = [
        make_onnx_intializer(
            "fc1_experts_weights", fc1_experts_weights.to(weight_torch_dtype), fc1_weight_shape, weight_onnx_type
        ),
        make_onnx_intializer("fc1_experts_bias", fc1_experts_bias.to(torch_dtype), fc1_bias_shape, onnx_dtype),
        make_onnx_intializer(
            "fc2_experts_weights", fc2_experts_weights.to(weight_torch_dtype), fc2_weight_shape, weight_onnx_type
        ),
        make_onnx_intializer("fc2_experts_bias", fc2_experts_bias.to(torch_dtype), fc2_bias_shape, onnx_dtype),
    ]

    if use_quant:
        initializers.extend(
            [
                make_onnx_intializer(
                    "fc1_experts_weight_scale",
                    fc1_experts_weight_scale.to(torch_dtype),
                    fc1_experts_weight_scale_shape,
                    onnx_dtype,
                ),
                make_onnx_intializer(
                    "fc2_experts_weight_scale",
                    fc2_experts_weight_scale.to(torch_dtype),
                    fc2_experts_weight_scale_shape,
                    onnx_dtype,
                ),
            ]
        )

    graph_inputs = [
        helper.make_tensor_value_info("input", onnx_dtype, [num_tokens, hidden_size]),
    ]

    graph_inputs.append(
        helper.make_tensor_value_info(
            "router_probs",
            onnx_dtype,
            [num_tokens, num_experts],
        )
    )

    graph_outputs = [
        helper.make_tensor_value_info("output", onnx_dtype, [num_tokens, hidden_size]),
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
    def __init__(
        self, config: SwigluMoeConfig, batch_size: int, sequence_length: int, quant_bits: int = 0, onnx_dtype=None
    ):
        super().__init__(quant_bits, onnx_dtype=onnx_dtype)
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_token
        use_quant = self.quant_bits > 0

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=True)

        self.experts = nn.ModuleList([SwigluMlp(config) for _ in range(self.num_experts)])

        # For the ONNX MoE operator, weights must be transposed to [In, Out] format.
        # Biases do not require transposition.
        fc1_w_list, fc2_w_list = [], []
        fc1_b_list, fc2_b_list = [], []
        scale_1_list, scale_2_list = [], []

        for expert in self.experts:
            fc1_b_list.append(expert.w1.bias)
            fc2_b_list.append(expert.w2.bias)
            if not use_quant:
                fc1_w_list.append(expert.w1.weight)
                fc2_w_list.append(expert.w2.weight)
            else:
                is_4_bit = self.quant_bits == 4

                # quant_dequant expects [Out, In] format, matching nn.Linear.weight
                scale1, pre_qweight1, w1_qdq = quant_dequant(expert.w1.weight, is_4_bit)
                scale2, pre_qweight2, w2_qdq = quant_dequant(expert.w2.weight, is_4_bit)

                # Update the expert's weight with the dequantized version for the PyTorch reference.
                expert.w1.weight.data = w1_qdq
                expert.w2.weight.data = w2_qdq

                fc1_w_list.append(pre_qweight1)
                fc2_w_list.append(pre_qweight2)
                scale_1_list.append(scale1)
                scale_2_list.append(scale2)

        # Stack the prepared tensors for the graph builder
        fc1_experts_weights = torch.stack(fc1_w_list, dim=0)
        fc2_experts_weights = torch.stack(fc2_w_list, dim=0)
        fc1_experts_bias = torch.stack(fc1_b_list, dim=0)
        fc2_experts_bias = torch.stack(fc2_b_list, dim=0)

        moe_experts_weight_scale1 = torch.stack(scale_1_list, dim=0) if use_quant else None
        moe_experts_weight_scale2 = torch.stack(scale_2_list, dim=0) if use_quant else None

        self.batch_size = batch_size
        self.sequence_length = sequence_length

        # Build the ONNX graph with the correctly shaped tensors
        self.moe_onnx_graph = create_swiglu_moe_onnx_graph(
            num_tokens=self.batch_size * self.sequence_length,
            num_experts=self.num_experts,
            hidden_size=self.hidden_dim,
            inter_size=self.ffn_dim,
            topk=self.top_k,
            onnx_dtype=self.onnx_dtype,
            quant_bits=self.quant_bits,
            fc1_experts_weights=fc1_experts_weights,
            fc1_experts_bias=fc1_experts_bias,
            fc2_experts_weights=fc2_experts_weights,
            fc2_experts_bias=fc2_experts_bias,
            fc1_experts_weight_scale=moe_experts_weight_scale1,
            fc2_experts_weight_scale=moe_experts_weight_scale2,
            swiglu_fusion=config.swiglu_fusion,
            activation_alpha=config.swiglu_alpha,
            activation_beta=config.swiglu_beta,
            swiglu_limit=config.swiglu_limit,
        )

        self.ort_sess = self.create_ort_session(self.moe_onnx_graph)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        This is the robust PyTorch reference implementation. It directly uses the
        nn.Module experts, which is cleaner and less error-prone than manual matmul.
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states


swiglu_test_cases = list(
    itertools.product(
        [1, 2],  # batch_size
        [1, 3],  # sequence_length
        quant_bits_list,  # quant_bits (0 for fp32/fp32, 8 for int8/fp16, 4 for int4/fp16)
    )
)


@unittest.skipIf(not use_cuda, "skipping moe test since it requires cuda environment.")
class TestSwigluMoE(unittest.TestCase):
    @parameterized.expand(swiglu_test_cases)
    def test_swiglu_moe_parity(self, batch_size, sequence_length, quant_bits):
        config = SwigluMoeConfig(
            hidden_size=64,
            intermediate_size=256,
            num_experts_per_token=2,
            num_local_experts=4,
            swiglu_fusion=1,
            swiglu_alpha=1.702,
            swiglu_beta=1.0,
            swiglu_limit=7.0,
        )
        moe = SwigluMoEBlock(config, batch_size, sequence_length, quant_bits)
        moe.to(device)
        moe.parity_check()


@unittest.skipIf(not use_cuda, "skipping moe test since it requires cuda environment.")
class TestStandardSwigluMoE(unittest.TestCase):
    """Standard (Llama/Gemma-style) SwiGLU: y = silu(gate) * value.

    This uses the default SwiGLU parameters (alpha=1.0, beta=0.0, no clamp limit)
    with the concatenated weight layout (swiglu_fusion=2), where the FC1 output is
    [gate; value] rather than the interleaved GPT-OSS layout.
    """

    @parameterized.expand(swiglu_test_cases)
    def test_standard_swiglu_moe_parity(self, batch_size, sequence_length, quant_bits):
        config = SwigluMoeConfig(
            hidden_size=64,
            intermediate_size=256,
            num_experts_per_token=2,
            num_local_experts=4,
            swiglu_fusion=2,
            swiglu_alpha=1.0,
            swiglu_beta=0.0,
            swiglu_limit=None,
        )
        moe = SwigluMoEBlock(config, batch_size, sequence_length, quant_bits)
        moe.to(device)
        moe.parity_check()


def has_bf16_moe():
    if not use_cuda or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


@unittest.skipIf(not has_bf16_moe(), "skipping bf16 moe tests.")
class TestSwigluMoeBf16(unittest.TestCase):
    @parameterized.expand(swiglu_test_cases)
    def test_swiglu_moe_parity(self, batch_size, sequence_length, quant_bits):
        config = SwigluMoeConfig(hidden_size=64, intermediate_size=128, num_experts_per_token=2, num_local_experts=4)
        moe = SwigluMoEBlock(config, batch_size, sequence_length, quant_bits, onnx_dtype=TensorProto.BFLOAT16)
        moe.to(device)
        moe.parity_check()


perf_test_cases = list(
    itertools.product(
        [1],  # batch_size
        [128, 512, 1024, 2048, 4096],  # sequence_length
        [0, 8, 4],  # quant_bits (0 for fp32/fp32, 8 for int8/fp16, 4 for int4/fp16)
    )
)


@unittest.skipIf(pipeline_mode or not use_cuda, "skipping performance test in CI pipeline.")
class TestSwigluMoEPerf(unittest.TestCase):
    @parameterized.expand(perf_test_cases)
    def test_swiglu_moe_performance(self, batch_size, sequence_length, quant_bits):
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
        moe = SwigluMoEBlock(config, batch_size, sequence_length, quant_bits)
        moe.to(device)
        moe.benchmark_ort()


def create_sparse_mixer_onnx_graph(
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
            k=2,
            activation_type="relu",  # Sparse mixer used relu in old code? Actually any activation works with kernel.
            normalize_routing_weights=0,
            use_sparse_mixer=1,
            domain="com.microsoft",
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "MoE_Graph",
        [
            helper.make_tensor_value_info("input", onnx_dtype, [sequence_length, hidden_size]),
            helper.make_tensor_value_info("router_probs", onnx_dtype, [sequence_length, num_experts]),
            helper.make_tensor_value_info("fc1_experts_weights", onnx_dtype, [num_experts, inter_size, hidden_size]),
            helper.make_tensor_value_info("fc1_experts_bias", onnx_dtype, [num_experts, inter_size]),
            helper.make_tensor_value_info("fc2_experts_weights", onnx_dtype, [num_experts, hidden_size, inter_size]),
            helper.make_tensor_value_info("fc2_experts_bias", onnx_dtype, [num_experts, hidden_size]),
        ],
        [
            helper.make_tensor_value_info("output", onnx_dtype, [sequence_length, hidden_size]),
        ],
    )

    return helper.make_model(graph, producer_name="MoE_Model")


class TestSparseMixer(unittest.TestCase):
    @parameterized.expand(
        list(
            itertools.product(
                [TensorProto.FLOAT16],
            )
        )
    )
    def test_sparse_mixer_functional(self, onnx_dtype):
        # Basic regression test for Sparse Mixer integration.
        # k=2, experts=8 (supported size)
        num_rows = 128
        hidden_size = 64
        inter_size = 32
        num_experts = 8

        torch_dtype = onnx_to_torch_type_map[onnx_dtype]

        input_data = torch.randn(num_rows, hidden_size, dtype=torch_dtype, device=device)
        router_probs = torch.randn(num_rows, num_experts, dtype=torch_dtype, device=device)

        fc1_weight = torch.randn(num_experts, hidden_size, inter_size, dtype=torch_dtype, device=device)
        fc1_bias = torch.randn(num_experts, inter_size, dtype=torch_dtype, device=device)
        fc2_weight = torch.randn(num_experts, inter_size, hidden_size, dtype=torch_dtype, device=device)
        fc2_bias = torch.randn(num_experts, hidden_size, dtype=torch_dtype, device=device)

        onnx_model = create_sparse_mixer_onnx_graph(
            num_rows,
            num_experts,
            hidden_size,
            inter_size,
            fc1_weight.transpose(1, 2).contiguous(),
            fc1_bias,
            fc2_weight.transpose(1, 2).contiguous(),
            fc2_bias,
            onnx_dtype,
        )

        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(onnx_model.SerializeToString(), sess_options, providers=get_ort_provider())

        inputs = {
            "input": input_data.cpu().numpy(),
            "router_probs": router_probs.cpu().numpy(),
            "fc1_experts_weights": fc1_weight.transpose(1, 2).contiguous().cpu().numpy(),
            "fc1_experts_bias": fc1_bias.cpu().numpy(),
            "fc2_experts_weights": fc2_weight.transpose(1, 2).contiguous().cpu().numpy(),
            "fc2_experts_bias": fc2_bias.cpu().numpy(),
        }

        # Just ensure it runs without error
        output = sess.run(None, inputs)
        self.assertEqual(output[0].shape, (num_rows, hidden_size))

    @unittest.skipIf(not use_cuda, "Sparse Mixer testing requires CUDAExecutionProvider")
    def test_sparse_mixer_parity(self):
        # Parity test against Python masked_sampling_omp_inference
        # Checks if ORT kernel logic (jitter, OMP) matches Python reference.
        onnx_dtype = TensorProto.FLOAT16
        num_rows = 128
        hidden_size = 64
        inter_size = 32
        num_experts = 8
        k = 2

        torch_dtype = onnx_to_torch_type_map[onnx_dtype]
        jit_eps = 0.01

        # Inputs
        # Use simple ranges to avoid randomness issues if possible, but random is okay for parity check if stable.
        input_data = torch.randn(num_rows, hidden_size, dtype=torch_dtype, device=device)
        # Random logits
        router_logits = torch.randn(num_rows, num_experts, dtype=torch_dtype, device=device)

        fc1_weight = torch.randn(num_experts, hidden_size, inter_size, dtype=torch_dtype, device=device)
        fc1_bias = torch.zeros(num_experts, inter_size, dtype=torch_dtype, device=device)
        fc2_weight = torch.randn(num_experts, inter_size, hidden_size, dtype=torch_dtype, device=device)
        fc2_bias = torch.zeros(num_experts, hidden_size, dtype=torch_dtype, device=device)

        # 1. ORT Execution
        onnx_model = create_sparse_mixer_onnx_graph(
            num_rows,
            num_experts,
            hidden_size,
            inter_size,
            fc1_weight.transpose(1, 2).contiguous(),
            fc1_bias,
            fc2_weight.transpose(1, 2).contiguous(),
            fc2_bias,
            onnx_dtype,
        )
        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession(onnx_model.SerializeToString(), sess_options, providers=get_ort_provider())

        ort_inputs = {
            "input": input_data.cpu().numpy(),
            "router_probs": router_logits.cpu().numpy(),
            "fc1_experts_weights": fc1_weight.transpose(1, 2).contiguous().cpu().numpy(),
            "fc1_experts_bias": fc1_bias.cpu().numpy(),
            "fc2_experts_weights": fc2_weight.transpose(1, 2).contiguous().cpu().numpy(),
            "fc2_experts_bias": fc2_bias.cpu().numpy(),
        }
        ort_output = sess.run(None, ort_inputs)[0]

        # 2. Python Reference Execution
        # Calculate routing weights and indices
        routing_weights, selected_experts = masked_sampling_omp_inference(
            router_logits, top_k=k, jitter_eps=jit_eps, training=False
        )

        final_output = torch.zeros_like(input_data)

        # Manual MoE
        # Loop over experts to mimic expert parallelism / gathering
        for expert_idx in range(num_experts):
            # selected_experts is [B, k]
            # Find which rows selected this expert as 1st choice
            mask1 = selected_experts[:, 0] == expert_idx
            # Find which rows selected this expert as 2nd choice
            mask2 = selected_experts[:, 1] == expert_idx

            # Combine to get all rows processing this expert
            active_mask = mask1 | mask2
            if not active_mask.any():
                continue

            active_indices = torch.nonzero(active_mask, as_tuple=True)[0]

            # Select input rows
            inp_slice = input_data[active_indices]

            # Select weights for these rows for this expert
            # If row selected expert as 1st choice, use weight[:, 0], else weight[:, 1]
            # routing_weights is [B, k]
            w1 = routing_weights[active_indices, 0]
            w2 = routing_weights[active_indices, 1]

            # Construct the weight vector for these rows
            # We need to know for each active row, was it 1st or 2nd choice?
            # It's guaranteed to be one of them (or both? No, expert selection is unique per row in OMP generally, but let's assume unique)

            row_mask1 = mask1[active_indices]
            ex_weights = torch.where(row_mask1, w1, w2).unsqueeze(1)

            # Compute Expert FFN
            # FC1: [B_sub, H] @ [H, I] + [I]
            h = torch.matmul(inp_slice, fc1_weight[expert_idx]) + fc1_bias[expert_idx]
            h = torch.relu(h)

            # FC2: [B_sub, I] @ [I, H] + [H]
            out = torch.matmul(h, fc2_weight[expert_idx]) + fc2_bias[expert_idx]

            # Accumulate
            final_output[active_indices] += out * ex_weights

        # Compare
        ort_output_tensor = torch.from_numpy(ort_output).to(device)

        max_diff = (ort_output_tensor - final_output).abs().max().item()
        print(f"\nTestSparseMixer Parity Max Diff: {max_diff}")

        # Allow some tolerance for float/half and jitter math
        self.assertTrue(
            numpy.allclose(ort_output, final_output.cpu().numpy(), atol=1e-1, rtol=1e-1),
            msg=f"Max Diff {max_diff} exceeds tolerance",
        )


if __name__ == "__main__":
    unittest.main()
