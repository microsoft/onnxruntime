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

# --- Type Maps ---
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


# --- Quantization Helper ---
def quant_dequant(weights, is_4_bit_quantization: bool = True):
    """
    Quantizes and dequantizes weights using the TRT-LLM quantization op.
    This is used to generate reference outputs and prepare weights for the ONNX graph.
    """
    # This function requires tensorrt_llm, which may not be in the CI environment.
    # It is only called when quant_bits > 0.
    type = torch.quint4x2 if is_4_bit_quantization else torch.int8

    import tensorrt_llm  # noqa: PLC0415

    if pipeline_mode:
        print("Tensorrt LLM version", tensorrt_llm.__version__)

    quant_weights, processed_q_weight, torch_weight_scales = (
        torch.ops.trtllm._symmetric_quantize_last_axis_of_batched_matrix(weights.T.cpu().contiguous(), type)
    )

    # Unpack the int4s into int8s for dequantization
    if is_4_bit_quantization:
        upper = quant_weights >> 4
        lower = (quant_weights << 4) >> 4  # Arithmetic right shift sign extends
        quant_weights = torch.stack((lower, upper), dim=2).view(weights.T.shape)

    quant_weights = quant_weights.to(dtype=weights.dtype)
    result = torch.multiply(quant_weights, torch_weight_scales.unsqueeze(0)).T.contiguous()
    return torch_weight_scales.to(torch.float16), processed_q_weight, result.to(device=weights.device)


# --- ONNX Graph Builders ---
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
        initializer = helper.make_tensor(
            name=name,
            data_type=onnx_dtype,
            dims=dims,
            vals=vals,
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
            domain="com.microsoft",
        ),
    ]

    if use_quant:
        nodes[0].attribute.extend([helper.make_attribute("expert_weight_bits", quant_bits)])

    # Define tensor shapes for weights, biases, and scales
    pack_factor = 2 if quant_bits == 4 else 1
    fc1_weight_shape = [num_experts, 2 * inter_size, hidden_size // pack_factor]
    fc1_bias_shape = [num_experts, 2 * inter_size]
    fc1_scale_shape = [num_experts, 2 * inter_size]

    fc2_weight_shape = [num_experts, hidden_size, inter_size // pack_factor]
    fc2_bias_shape = [num_experts, hidden_size]
    fc2_scale_shape = [num_experts, hidden_size]

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


# --- PyTorch Reference Models ---
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


class SparseMoeBlockORTHelper(nn.Module):
    """Base class for MoE test blocks, handling ORT session and parity checks."""

    def __init__(self, quant_bits=0, onnx_dtype=None):
        super().__init__()
        self.quant_bits = quant_bits
        if onnx_dtype is None:
            self.onnx_dtype = TensorProto.FLOAT16 if self.quant_bits > 0 else TensorProto.FLOAT
        else:
            self.onnx_dtype = onnx_dtype
        self.np_type = ort_to_numpy_type_map.get(self.onnx_dtype, numpy.float32)

    def create_ort_session(self, moe_onnx_graph):
        if not moe_onnx_graph:
            return None
        try:
            sess_options = onnxruntime.SessionOptions()
            sess_options.log_severity_level = 2
            return onnxruntime.InferenceSession(moe_onnx_graph, sess_options, providers=ort_provider)
        except Exception as e:
            print(f"Failed to create ONNX Runtime session with provider {ort_provider}: {e}")
            return None

    def ort_forward(self, hidden_states: torch.Tensor, enable_performance_test=False) -> torch.Tensor:
        if self.ort_sess is None:
            return None

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states_flat)

        torch_dtype = onnx_to_torch_type_map[self.onnx_dtype]
        ort_inputs = {
            "input": hidden_states_flat.to(device=device, dtype=torch_dtype).cpu().numpy(),
            "router_probs": router_logits.to(device=device, dtype=torch_dtype).cpu().numpy(),
        }

        if enable_performance_test:
            import time  # noqa: PLC0415

            repeat = 1000
            s = time.time()
            for _ in range(repeat):
                self.ort_sess.run(None, ort_inputs)
            e = time.time()
            print(f"MoE CUDA kernel time: {(e - s) / repeat * 1000:.4f} ms")

        ort_outputs = self.ort_sess.run(None, ort_inputs)
        output_tensor = torch.from_numpy(ort_outputs[0]).to(device)
        return output_tensor.reshape(batch_size, sequence_length, hidden_dim)

    def parity_check(self):
        hidden_state = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim).to(device)
        torch_output = self.forward(hidden_state)
        ort_output = self.ort_forward(hidden_state)

        if ort_output is None:
            self.skipTest("ONNX Runtime execution failed or is not supported.")
            return

        dtype_str = ort_dtype_name_map[self.onnx_dtype]
        print(
            f"Parity Check: {self.__class__.__name__}, quant_bits: {self.quant_bits}, dtype: {dtype_str},"
            f" batch: {self.batch_size}, seq_len: {self.sequence_length},"
            f" max_diff: {(torch_output.cpu() - ort_output.cpu()).abs().max()}"
        )

        ort_dtype_quant_bits_tolerance_map = {
            "FP32:0": (5e-3, 1e-3),
            "FP16:0": (5e-2, 1e-3),
            "FP16:4": (3.0, 1e-2),
            "FP16:8": (2.0, 1e-2),
            "BF16:0": (1.0, 1e-2),
            "BF16:4": (30.0, 1e-1),
            "BF16:8": (20.0, 1e-1),
        }
        tolerance_key = f"{dtype_str}:{self.quant_bits}"
        if tolerance_key not in ort_dtype_quant_bits_tolerance_map:
            self.fail(f"No tolerance defined for {tolerance_key}")

        atol, rtol = ort_dtype_quant_bits_tolerance_map[tolerance_key]
        torch.testing.assert_close(
            ort_output.cpu().to(torch.float32), torch_output.cpu().to(torch.float32), rtol=rtol, atol=atol
        )


class SwigluMoEBlock(SparseMoeBlockORTHelper):
    """Test block for MoE with SwiGLU activation."""

    def __init__(
        self,
        config,
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
                fused_qw1, fused_s1, fused_b1 = self._fuse_for_swiglu(
                    qw1, s1, w1_bias, qw3, s3, w3_bias, swiglu_fusion
                )
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


# --- Test Cases and Classes ---
class SwigluMoeConfig:
    def __init__(
        self, hidden_size=2048, intermediate_size=2048, num_experts_per_token=2, num_local_experts=8
    ):
        self.hidden_size, self.intermediate_size = hidden_size, intermediate_size
        self.num_experts_per_token, self.num_local_experts = num_experts_per_token, num_local_experts


# since qMoE test requires tensorrt_llm for quant_dequant. We disable it in CI pipeline to avoid extra dependency.
quant_bits_list = [0] if pipeline_mode else [0, 8, 4]

swiglu_test_cases = list(
    itertools.product(
        [1, 2],  # batch_size
        [1, 3],  # sequence_length
        quant_bits_list,
        [1, 2],  # swiglu_fusion (1 for interleaved, 2 for concatenated)
    )
)


@unittest.skipIf(not use_cuda, "skipping MoE test since it requires CUDA environment.")
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


@unittest.skipIf(not has_bf16_moe(), "skipping bf16 moe tests.")
class TestSwigluMoeBf16(unittest.TestCase):
    @parameterized.expand(swiglu_test_cases)
    def test_swiglu_moe_parity(self, batch_size, sequence_length, quant_bits, swiglu_fusion):
        config = SwigluMoeConfig(hidden_size=64, intermediate_size=128, num_experts_per_token=2, num_local_experts=4)
        moe = SwigluMoEBlock(
            config, batch_size, sequence_length, quant_bits, swiglu_fusion, onnx_dtype=TensorProto.BFLOAT16
        )
        moe.to(device)
        moe.parity_check()


if __name__ == "__main__":
    unittest.main()
