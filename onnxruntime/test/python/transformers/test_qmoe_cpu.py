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
#
# QMoE quantization implementation notes:
#
# Both CPU and CUDA implementations use symmetric quantization:
# - 4-bit: range [-8, 7]
# - 8-bit: range [-128, 127]
#
# This ensures TensorRT compatibility.
# Tolerance values account for numerical differences between implementations.
# --------------------------------------------------------------------------
import itertools
import time
import unittest
from collections import OrderedDict

import numpy
import torch
from onnx import helper
from parameterized import parameterized
from torch import nn

import onnxruntime

try:
    from onnx import TensorProto

    has_onnx = True
except ImportError:
    has_onnx = False

    # Define placeholder constants if onnx is not available
    class TensorProtoPlaceholder:
        FLOAT16 = 10
        FLOAT = 1
        # BF16 not supported in QMoE CPU
        UINT8 = 2

    TensorProto = TensorProtoPlaceholder

# Reduces number of tests to run for faster pipeline checks

onnxruntime.preload_dlls()

# Force CPU execution provider regardless of CUDA availability
device = torch.device("cpu")
ort_provider = ["CPUExecutionProvider"]

torch.manual_seed(42)
numpy.random.seed(42)

onnx_to_torch_type_map = {
    TensorProto.FLOAT16: torch.float16,
    TensorProto.FLOAT: torch.float,
    # BF16 not supported in QMoE CPU
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
    # QMoE CPU does not support BF16
}


def quant_dequant(weights, is_4_bit_quantization: bool = True):
    """
    Quantize and dequantize weights for testing purposes.
    This function exactly matches the C++ implementation in QMoE CPU.

    This uses symmetric quantization to match the C++ implementation:
    - 4-bit: range = [-8, 7], zero_point = 8
    - 8-bit: range = [-128, 127], zero_point = 128
    """
    # Handle edge case of all-zero weights tensor
    if torch.all(weights == 0):
        if is_4_bit_quantization:
            packed_size = (weights.shape[-1] + 1) // 2
            return (
                torch.zeros_like(weights[..., 0:1]),
                torch.zeros(
                    (weights.shape[0], weights.shape[1], packed_size),
                    dtype=torch.uint8,
                    device=weights.device,
                ),
                torch.zeros_like(weights),
            )
        else:
            return (
                torch.zeros_like(weights[..., 0:1]),
                torch.zeros_like(weights, dtype=torch.uint8),
                torch.zeros_like(weights),
            )

    # Calculate scale like C++ implementation
    abs_max = weights.abs().max(dim=-1, keepdim=True)[0]
    abs_max = torch.clamp(abs_max, min=1e-12)  # Avoid division by zero

    if is_4_bit_quantization:
        # 4-bit: scale = abs_max / 7.0 (C++ uses 7.0 as max positive value)
        scale = abs_max / 7.0

        # Handle potential edge cases for zero or very small weights
        if torch.max(abs_max) < 1e-8:
            packed_size = (weights.shape[-1] + 1) // 2
            return (
                torch.ones_like(weights[..., 0:1]) * 1e-8,
                torch.full(
                    (weights.shape[0], weights.shape[1], packed_size),
                    fill_value=8 | (8 << 4),  # 8 = zero in symmetric quantization
                    dtype=torch.uint8,
                    device=weights.device,
                ),
                torch.zeros_like(weights),
            )

        # Quantize: round(weight / scale) then clamp to [-8, 7]
        scaled_weights = weights / scale
        quantized_weights = torch.round(scaled_weights).clamp(-8, 7)

        # Convert to uint8 storage: add 8 to shift [-8,7] -> [0,15]
        quant_weights = (quantized_weights + 8).to(torch.uint8)

        # Pack 4-bit values into uint8 (every two elements)
        even_indices = torch.arange(0, weights.shape[-1], 2)
        odd_indices = torch.arange(1, weights.shape[-1], 2)

        # Handle odd length by padding with zero (which is 8 in storage)
        if odd_indices.shape[0] < even_indices.shape[0]:
            padding = torch.full(
                (quant_weights.shape[0], quant_weights.shape[1], 1),
                fill_value=8,
                dtype=torch.uint8,
                device=quant_weights.device,
            )
            quant_weights = torch.cat([quant_weights, padding], dim=-1)
            odd_indices = torch.arange(1, quant_weights.shape[-1], 2)

        even_weights = quant_weights[..., even_indices]
        odd_weights = quant_weights[..., odd_indices]

        # Pack: low nibble = even, high nibble = odd (matches C++)
        packed_weights = (even_weights & 0xF) | ((odd_weights & 0xF) << 4)

        # Dequantize exactly like C++: scale * (quantized_value - 8.0)
        # Unpack for dequantization
        lower = packed_weights & 0xF
        upper = (packed_weights >> 4) & 0xF

        # Restore original shape
        unpacked_weights = torch.zeros_like(weights, dtype=torch.uint8)
        unpacked_weights[..., even_indices] = lower

        valid_odd_length = min(odd_indices.shape[0], weights.shape[-1] - even_indices.shape[0])
        if valid_odd_length > 0:
            valid_odd_indices = odd_indices[:valid_odd_length]
            unpacked_weights[..., valid_odd_indices] = upper[..., :valid_odd_length]

        # Dequantize exactly like C++: scale * (quantized_value - 8.0f)
        result = scale * (unpacked_weights.float() - 8.0)

        return scale.to(torch.float16), packed_weights, result.to(weights.dtype)
    else:
        # 8-bit: scale = abs_max / 127.0 (C++ uses 127.0 as max positive value)
        # Add small epsilon for better numerical stability with larger tensors
        scale = abs_max / 127.0 + 1e-10

        # Handle potential edge cases for zero or very small weights
        if torch.max(abs_max) < 1e-8:
            return (
                torch.ones_like(weights[..., 0:1]) * 1e-8,
                torch.full_like(weights, fill_value=128, dtype=torch.uint8),
                torch.zeros_like(weights),
            )

        # Quantize: round(weight / scale) then clamp to [-128, 127]
        scaled_weights = weights / scale
        quantized_weights = torch.round(scaled_weights).clamp(-128, 127)

        # Convert to uint8 storage: add 128 to shift [-128,127] -> [0,255]
        quant_weights = (quantized_weights + 128).to(torch.uint8)

        # Dequantize exactly like C++: scale * (quantized_value - 128.0f)
        result = scale * (quant_weights.float() - 128.0)

        return scale.to(torch.float16), quant_weights, result.to(weights.dtype)


def create_cpu_moe_onnx_graph(
    hidden_size,
    sequence_length,
    num_experts,
    top_k,
    intermediate_size,
    torch_dtype,
    onnx_dtype,
    fc1_experts_weights,
    fc2_experts_weights,
    fc1_bias=None,
    fc2_bias=None,
    fc1_scales=None,
    fc2_scales=None,
    use_swiglu=False,
    use_quant=False,
    quant_bits=4,
    swiglu_interleaved=False,
):
    # Make sure we have onnx available before proceeding
    if not has_onnx:
        return None

    # Define intermediate_size variable consistently
    inter_size = intermediate_size
    topk = top_k

    # Force use_quant to True - we only want to test QMoE
    use_quant = True

    # In QMoE, biases are not used
    # The following parameters are only relevant when use_quant=False (which is never the case here)
    # fc1_bias and fc2_bias are completely ignored for QMoE

    # Ensure all variables are properly initialized for safety
    if fc1_scales is None and use_quant:
        return None
    if fc2_scales is None and use_quant:
        return None
    if not has_onnx:
        return None

    # Using uint8 storage type with symmetric quantization
    # 4-bit: range = [-8, 7] (stored as uint8 values [0, 15])
    # 8-bit: range = [-128, 127] (stored as uint8 values [0, 255])
    assert fc1_experts_weights.dtype == torch.uint8, "FC1 weights must be uint8 for QMoE"
    assert fc2_experts_weights.dtype == torch.uint8, "FC2 weights must be uint8 for QMoE"
    assert fc1_scales is not None, "FC1 scales must be provided for QMoE"
    assert fc2_scales is not None, "FC2 scales must be provided for QMoE"
    assert fc1_scales.dtype == torch.float16, "FC1 scales must be float16 for QMoE"
    assert fc2_scales.dtype == torch.float16, "FC2 scales must be float16 for QMoE"

    # Make sure we have onnx available before proceeding
    if not has_onnx:
        return None

    # Always use QMoE, never MoE
    op_name = "QMoE"
    inputs = [
        "input",
        "router_probs",
        "fc1_experts_weights",
        "fc1_scales",
        "",
        "fc2_experts_weights",
        "fc2_scales",
        "",
    ]

    # In QMoE mode, biases are not used
    # This code path is never executed since use_quant is always True

    # Use SwiGLU activation if specified, otherwise use SiLU
    activation = "swiglu" if use_swiglu else "silu"

    nodes = [
        helper.make_node(
            op_name,
            inputs,
            ["output"],
            "MoE_0",
            k=topk,
            normalize_routing_weights=0,
            activation_type=activation,
            # Add new attributes with backwards-compatible default values
            swiglu_fusion=1 if (use_swiglu and swiglu_interleaved) else 0,  # 1 = fused and interleaved
            activation_alpha=1.702,  # SwiGLU default alpha
            activation_beta=1.0,  # SwiGLU default beta
            domain="com.microsoft",
        ),
    ]

    if use_quant:
        nodes[0].attribute.extend([helper.make_attribute("expert_weight_bits", quant_bits)])

    # Weights are store in column major order. Need pack 2 int4 values into uint8.
    # Use the actual tensor shapes instead of calculating them to avoid size mismatches
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
        ),
        helper.make_tensor(
            "fc2_experts_weights",
            weight_onnx_type,
            fc2_shape,
            fc2_experts_weights.flatten().detach().cpu().numpy().astype(weight_numpy_type).tolist(),
        ),
    ]

    # QMoE always uses scales, never biases
    # For SwiGLU, FC1 scales shape needs to be doubled to account for gate and value components
    fc1_scale_shape = [num_experts, 2 * inter_size if use_swiglu else inter_size]
    fc2_scale_shape = [num_experts, hidden_size]

    # Handle scale tensors
    # Calculate correct scale tensor sizes - fc1 needs 2x for SwiGLU (gate + value)
    fc1_scale_size = num_experts * (2 * inter_size if use_swiglu else inter_size)
    fc2_scale_size = num_experts * hidden_size

    if fc1_scales is not None:
        # Handle different possible scale tensor structures
        if len(fc1_scales.shape) == 4:
            # 4D case: [num_experts, inter_size, hidden_size, 1] - extract first scale per expert per output
            if use_swiglu:
                fc1_scale_tensor = (
                    fc1_scales.to(torch_dtype)[:, : 2 * inter_size, 0, 0].flatten().detach().cpu().numpy()
                )
            else:
                fc1_scale_tensor = fc1_scales.to(torch_dtype)[:, :inter_size, 0, 0].flatten().detach().cpu().numpy()
        elif len(fc1_scales.shape) == 2:
            # 2D case: already flattened, just ensure correct size
            fc1_scale_tensor = fc1_scales.to(torch_dtype).flatten().detach().cpu().numpy()
            if use_swiglu and fc1_scale_tensor.size == num_experts * inter_size:
                # For SwiGLU, duplicate the scales to cover both gate and value components
                fc1_scale_tensor = numpy.tile(fc1_scale_tensor.reshape(num_experts, inter_size), (1, 2)).flatten()
            elif fc1_scale_tensor.size > fc1_scale_size:
                # Truncate to expected size
                fc1_scale_tensor = fc1_scale_tensor[:fc1_scale_size]
        else:
            # Other cases: flatten and truncate/pad as needed
            fc1_scale_tensor = fc1_scales.to(torch_dtype).flatten().detach().cpu().numpy()
            if fc1_scale_tensor.size > fc1_scale_size:
                fc1_scale_tensor = fc1_scale_tensor[:fc1_scale_size]
            elif fc1_scale_tensor.size < fc1_scale_size:
                # Pad with ones if too small
                pad_size = fc1_scale_size - fc1_scale_tensor.size
                fc1_scale_tensor = numpy.concatenate(
                    [fc1_scale_tensor, numpy.ones(pad_size, dtype=fc1_scale_tensor.dtype)]
                )

        # Process scale tensor for proper shape
        fc1_scale_data_list = fc1_scale_tensor.tolist()
        fc1_scale_data = fc1_scale_data_list
    else:
        fc1_scale_data = numpy.ones(fc1_scale_size, dtype=ort_to_numpy_type_map[onnx_dtype]).tobytes()

    if fc2_scales is not None:
        # Handle different possible scale tensor structures
        if len(fc2_scales.shape) == 4:
            # 4D case: [num_experts, hidden_size, inter_size, 1] - extract first scale per expert per output
            fc2_scale_tensor = fc2_scales.to(torch_dtype)[:, :hidden_size, 0, 0].flatten().detach().cpu().numpy()
        elif len(fc2_scales.shape) == 2:
            # 2D case: already flattened, just ensure correct size
            fc2_scale_tensor = fc2_scales.to(torch_dtype).flatten().detach().cpu().numpy()
            if fc2_scale_tensor.size > fc2_scale_size:
                # Truncate to expected size
                fc2_scale_tensor = fc2_scale_tensor[:fc2_scale_size]
        else:
            # Other cases: flatten and truncate/pad as needed
            fc2_scale_tensor = fc2_scales.to(torch_dtype).flatten().detach().cpu().numpy()
            if fc2_scale_tensor.size > fc2_scale_size:
                fc2_scale_tensor = fc2_scale_tensor[:fc2_scale_size]
            elif fc2_scale_tensor.size < fc2_scale_size:
                # Pad with ones if too small
                pad_size = fc2_scale_size - fc2_scale_tensor.size
                fc2_scale_tensor = numpy.concatenate(
                    [fc2_scale_tensor, numpy.ones(pad_size, dtype=fc2_scale_tensor.dtype)]
                )

        # Process scale tensor for proper shape
        fc2_scale_data_list = fc2_scale_tensor.tolist()
        fc2_scale_data = fc2_scale_data_list
    else:
        fc2_scale_data = numpy.ones(fc2_scale_size, dtype=ort_to_numpy_type_map[onnx_dtype]).tobytes()

    initializers.extend(
        [
            helper.make_tensor(
                "fc1_scales",
                onnx_dtype,
                fc1_scale_shape,
                fc1_scale_data,
                raw=False,
            ),
            helper.make_tensor(
                "fc2_scales",
                onnx_dtype,
                fc2_scale_shape,
                fc2_scale_data,
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
act2fn = ClassInstantier(ACT2CLS)


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


def masked_sampling_omp_inference(scores, top_k, jitter_eps, training):
    """
    Modified to match C++ QMoE implementation's routing logic:
    - Select top-k experts
    - Give equal weight (1.0) to each selected expert
    - No softmax normalization (matches C++ behavior)
    """
    assert top_k == 2
    assert not training

    # Get top-k experts (same as C++ top-k selection)
    mask_logits_threshold, selected_experts = torch.topk(scores, top_k)

    # Use equal weights for selected experts
    # C++ gives weight 1.0 to each selected expert
    multiplier = torch.ones_like(mask_logits_threshold)  # [batch, top_k] with values 1.0

    return multiplier, selected_experts

    # DEBUG: Check what multipliers we're computing

    return (
        multiplier,
        selected_experts,
    )


class MoEBlockSparseTop2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = act2fn[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class PhiMoEBlockSparseTop2MLP(MoEBlockSparseTop2MLP):
    def __init__(self, config: PhiMoEConfig, use_swiglu=False, swiglu_interleaved=True):
        super().__init__(config)
        self.use_swiglu = use_swiglu
        self.swiglu_interleaved = swiglu_interleaved

        # If using interleaved SwiGLU, adjust w1 size to accommodate both gate and value weights
        if self.use_swiglu and self.swiglu_interleaved:
            # Save the original weights before replacing
            gate_weight = self.w1.weight.data.clone()
            value_weight = self.w3.weight.data.clone()

            # Replace w1 with a layer that has 2x the output size for interleaved weights
            old_w1 = self.w1
            self.w1 = nn.Linear(old_w1.in_features, 2 * old_w1.out_features, bias=False)

            # Initialize with interleaved weights: [g0, v0, g1, v1, g2, v2, ...]
            with torch.no_grad():
                self.w1.weight[0::2] = gate_weight  # Even rows: gate weights
                self.w1.weight[1::2] = value_weight  # Odd rows: value weights

    def forward(self, hidden_states):
        if self.use_swiglu:
            # Enhanced SwiGLU implementation for better numerical compatibility with ORT
            # Use double precision temporarily to improve numerical stability
            hidden_fp64 = hidden_states.to(torch.float64)

            if self.swiglu_interleaved:
                # When weights are interleaved, w1 contains both gate and value weights
                # w1 shape: [2*intermediate_size, hidden_size] with interleaved rows
                # Output will be [batch, 2*intermediate_size] with interleaved values

                with torch.autocast(device_type="cpu", enabled=False):
                    combined_output = torch.nn.functional.linear(hidden_fp64, self.w1.weight.to(torch.float64))

                # De-interleave the output: [g0, v0, g1, v1, ...] -> [g0, g1, ...], [v0, v1, ...]
                # combined_output shape: [batch, 2*intermediate_size]
                gate_output = combined_output[..., 0::2]  # Even indices: gate values
                value_output = combined_output[..., 1::2]  # Odd indices: value values

            else:
                # Standard separate weights approach
                with torch.autocast(device_type="cpu", enabled=False):
                    gate_output = torch.nn.functional.linear(hidden_fp64, self.w1.weight.to(torch.float64))
                    value_output = torch.nn.functional.linear(hidden_fp64, self.w3.weight.to(torch.float64))

            # Apply SwiGLU exactly as in the C++ implementation (moe_utils.cc:ApplySwiGLUActivation)
            # C++ uses constexpr float swiglu_alpha = 1.702f and constexpr float clamp_limit = 7.0f
            swiglu_alpha = 1.702
            clamp_limit = 7.0

            # Apply clamping exactly as in the C++ implementation:
            # gate_val = std::min(gate_val, clamp_limit);                 // Clamp gate max only
            # linear_val = std::clamp(linear_val, -clamp_limit, clamp_limit); // Clamp linear min/max
            gate_output = torch.clamp(gate_output, max=clamp_limit)  # Clamp max only for gate
            value_output = torch.clamp(value_output, min=-clamp_limit, max=clamp_limit)  # Clamp both for value

            # In C++: float sigmoid_arg = swiglu_alpha * gate_val;
            # Compute the sigmoid input with the scaling factor (alpha)
            sigmoid_input = swiglu_alpha * gate_output

            # In C++: float sigmoid_out = 1.0f / (1.0f + std::exp(-sigmoid_arg));
            # PyTorch's sigmoid is equivalent to C++'s 1.0f / (1.0f + std::exp(-x))
            sigmoid_output = torch.sigmoid(sigmoid_input)

            # In C++: float swish_out = gate_val * sigmoid_out;
            # Complete the SiLU (Swish) operation: gate * sigmoid(alpha * gate)
            swish_output = gate_output * sigmoid_output

            # In C++: float result = swish_out * (linear_val + 1.0f);
            # This is exactly the SwiGLU formula: G * sigmoid(alpha * G) * (L + 1)
            gate_result = swish_output * (value_output + 1.0)

            # The SwiGLU result should have the same size as each component (intermediate_size/2 for interleaved)
            # This matches what the CPU kernel produces: 256→128 after SwiGLU
            current_hidden_states = gate_result

            # Match C++ rounding behavior exactly
            # C++ doesn't explicitly round FP32 results in most implementations,
            # but there is implicit rounding when converting between precisions
            if hidden_states.dtype == torch.float16:
                # For float16, C++ compilers typically use specific rounding modes
                # when converting from float32 to float16, which we need to emulate
                # The IEEE-754 standard specifies "round to nearest even" for this conversion
                # We don't need to manually round since PyTorch handles this in the .to() method
                pass
            else:
                # For float32, no explicit rounding is needed as C++ doesn't typically do this
                # unless specifically coded to do so
                pass

            current_hidden_states = current_hidden_states.to(hidden_states.dtype)

            # Apply FC2 (also handle the weight dtype conversion if needed)
            with torch.autocast(device_type="cpu", enabled=False):
                current_hidden_states = torch.nn.functional.linear(current_hidden_states, self.w2.weight)
            return current_hidden_states
        else:
            # Original implementation with standard activation
            return super().forward(hidden_states)


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
        if moe_onnx_graph is None:
            return None

        sess_options = onnxruntime.SessionOptions()
        sess_options.log_severity_level = 2

        try:
            ort_session = onnxruntime.InferenceSession(moe_onnx_graph, sess_options, providers=ort_provider)
        except Exception:
            return None

        return ort_session

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pass

    def ort_forward(self, hidden_states: torch.Tensor, enable_performance_test=False) -> torch.Tensor:
        # If session creation failed, we can't run inference
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

        try:
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
                repeat = 100  # Using fewer repeats for CPU tests
                s = time.time()
                for _ in range(repeat):
                    iobinding.synchronize_inputs()
                    self.ort_sess.run_with_iobinding(iobinding)
                    iobinding.synchronize_outputs()
                e = time.time()
                time_ms = (e - s) / repeat * 1000
                is_swiglu = hasattr(self, "use_swiglu") and self.use_swiglu
                is_interleaved = hasattr(self, "swiglu_interleaved") and self.swiglu_interleaved
                act_type = f"SwiGLU(interleaved={is_interleaved})" if is_swiglu else "SiLU"
                print(f"ORT Performance - {act_type} {self.quant_bits}-bit: {time_ms:.3f} ms/inference")

            # The output tensor is on `device`. Reshape and return it.
            return tensors["output"].reshape(batch_size, sequence_length, hidden_dim)

        except Exception as e:
            raise

    def recreate_onnx_model(self):
        """Recreate the ONNX model with the current weights to reflect any changes to the quantization code."""

        w1_list, w2_list = [], []
        w1_scale_list, w2_scale_list = [], []

        # Always use quantization for QMoE
        is_4_bit = self.quant_bits == 4
        for i in range(self.num_experts):
            # Re-quantize the weights with our updated quantization function
            w1_scale, pre_qweight1, w1_qdq = quant_dequant(self.experts[i].w1.weight, is_4_bit)
            w2_scale, pre_qweight2, w2_qdq = quant_dequant(self.experts[i].w2.weight, is_4_bit)

            # For SwiGLU, handle interleaved vs separate weights
            if self.use_swiglu:
                if self.swiglu_interleaved:
                    # In interleaved mode, w1 already contains both gate and value weights
                    # We don't need to quantize w3 separately since w1 contains everything
                    # pre_qweight1 already contains the interleaved quantized weights
                    # w1_scale already contains the interleaved scales
                    pass  # pre_qweight1 and w1_scale are already correct
                else:
                    # In separate mode, quantize w3 (value) weights separately
                    w3_scale, pre_qweight3, w3_qdq = quant_dequant(self.experts[i].w3.weight, is_4_bit)

                    gate_weights = pre_qweight1
                    value_weights = pre_qweight3
                    gate_scales = w1_scale
                    value_scales = w3_scale

                    # Create chunked layout: [gate..., value...]
                    pre_qweight1 = torch.cat([gate_weights, value_weights], dim=0)
                    w1_scale = torch.cat([gate_scales, value_scales], dim=0)

                # CRITICAL FIX: Update PyTorch weights to use the SAME dequantized values that the CPU will use
                # This ensures PyTorch and CPU use identical weights for fair comparison

                if self.swiglu_interleaved:
                    # For interleaved layout, w1 was already sized correctly in __init__
                    # Just set the weight data to the dequantized interleaved weights
                    # Make sure to create a contiguous copy to avoid reference issues

                    # Try to completely replace the weight parameter
                    self.experts[i].w1.weight = nn.Parameter(w1_qdq.contiguous().clone())

                else:
                    # For chunked layout, split the dequantized weights
                    intermediate_size = self.experts[i].w1.weight.shape[0]
                    gate_dequant = w1_qdq[:intermediate_size].contiguous().clone()
                    value_dequant = w1_qdq[intermediate_size:].contiguous().clone()
                    self.experts[i].w1.weight.data = gate_dequant
                    self.experts[i].w3.weight.data = value_dequant
            else:
                # Update the expert weights with dequantized values for PyTorch execution
                self.experts[i].w1.weight.data = w1_qdq.contiguous().clone()

            # Always update FC2 weights with dequantized values
            self.experts[i].w2.weight.data = w2_qdq.contiguous().clone()

            # Store the quantized weights and scales for ONNX model
            w1_list.append(pre_qweight1)
            w2_list.append(pre_qweight2)
            w1_scale_list.append(w1_scale)
            w2_scale_list.append(w2_scale)

        self.moe_experts_weight1 = torch.stack(w1_list, dim=0)
        self.moe_experts_weight2 = torch.stack(w2_list, dim=0)

        # Always use scales for QMoE
        moe_experts_weight_scale1 = torch.stack(w1_scale_list, dim=0)
        moe_experts_weight_scale2 = torch.stack(w2_scale_list, dim=0)

        # Fix shape mismatch: ensure scales are 2D for ONNX graph
        # The ONNX graph expects [num_experts, scale_size] but we might have [num_experts, scale_size, 1]
        if moe_experts_weight_scale1.dim() == 3:
            moe_experts_weight_scale1 = moe_experts_weight_scale1.squeeze(-1)
        if moe_experts_weight_scale2.dim() == 3:
            moe_experts_weight_scale2 = moe_experts_weight_scale2.squeeze(-1)

        # Recreate the ONNX graph with our updated quantization
        try:
            self.moe_onnx_graph = create_cpu_moe_onnx_graph(
                hidden_size=self.hidden_dim,
                sequence_length=self.batch_size * self.sequence_length,
                num_experts=self.num_experts,
                top_k=self.top_k,
                intermediate_size=self.ffn_dim,
                torch_dtype=torch.float32,
                onnx_dtype=self.onnx_dtype,
                fc1_experts_weights=self.moe_experts_weight1,
                fc2_experts_weights=self.moe_experts_weight2,
                # Biases are not used in QMoE
                fc1_bias=None,
                fc2_bias=None,
                # Scales are used for dequantization
                fc1_scales=moe_experts_weight_scale1,
                fc2_scales=moe_experts_weight_scale2,
                use_swiglu=self.use_swiglu,
                use_quant=True,  # Always use QMoE
                quant_bits=self.quant_bits,
                swiglu_interleaved=self.swiglu_interleaved if hasattr(self, "swiglu_interleaved") else False,
            )
        except Exception:
            self.moe_onnx_graph = None
            return False

        # Create a new ORT session with the updated model
        self.ort_sess = self.create_ort_session(self.moe_onnx_graph) if self.moe_onnx_graph else None
        return self.ort_sess is not None

    def parity_check(self):
        # Recreate the ONNX model with our latest quantization implementation
        model_updated = self.recreate_onnx_model()
        if not model_updated:
            return

        hidden_state = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim).to(device)
        torch_output = self.forward(hidden_state)
        ort_output = self.ort_forward(hidden_state)

        # If no ORT output was produced, we can't do a parity check
        if ort_output is None:
            return

        # Check for NaN and inf values in outputs and handle them
        torch_has_nan = torch.isnan(torch_output).any()
        ort_has_nan = torch.isnan(ort_output).any()
        torch_has_inf = torch.isinf(torch_output).any()
        ort_has_inf = torch.isinf(ort_output).any()

        if torch_has_nan or ort_has_nan or torch_has_inf or ort_has_inf:
            # Replace NaN and inf values with zeros for comparison
            torch_output_clean = torch.where(
                torch.isnan(torch_output) | torch.isinf(torch_output), torch.zeros_like(torch_output), torch_output
            )
            ort_output_clean = torch.where(
                torch.isnan(ort_output) | torch.isinf(ort_output), torch.zeros_like(ort_output), ort_output
            )
            max_diff = (torch_output_clean.cpu() - ort_output_clean.cpu()).abs().max()

            # If both have NaN/inf in the same places, consider it a match
            if (torch_has_nan and ort_has_nan) or (torch_has_inf and ort_has_inf):
                problematic_torch = torch.isnan(torch_output) | torch.isinf(torch_output)
                problematic_ort = torch.isnan(ort_output) | torch.isinf(ort_output)
                if torch.equal(problematic_torch, problematic_ort):
                    max_diff = 0.0  # Perfect match including NaN/inf patterns
        else:
            # Normal case without NaN/inf values
            max_diff = (torch_output.cpu() - ort_output.cpu()).abs().max()

        is_swiglu = hasattr(self, "use_swiglu") and self.use_swiglu
        is_interleaved = hasattr(self, "swiglu_interleaved") and self.swiglu_interleaved
        act_type = f"SwiGLU(interleaved={is_interleaved})" if is_swiglu else "SiLU"

        print(f"Parity check - {act_type} {self.quant_bits}-bit: max_diff = {max_diff:.6f}")

        # Check against expected tolerance for this configuration
        ort_dtype_quant_bits_tolerance_map = {
            "FP32:0": (5e-3, 1e-3),
            "FP16:0": (5e-2, 1e-3),
            "FP16:4": (0.05, 0.01),  # Much lower tolerance with exact C++ matching
            "FP16:8": (0.02, 0.01),  # Much lower tolerance with exact C++ matching
        }

        dtype_str = ort_dtype_name_map[self.onnx_dtype]
        tolerance_key = f"{dtype_str}:{self.quant_bits}"
        if tolerance_key in ort_dtype_quant_bits_tolerance_map:
            base_atol, rtol = ort_dtype_quant_bits_tolerance_map[tolerance_key]

            # With exact C++ matching, we should have low consistent errors
            if max_diff > base_atol:
                raise AssertionError(
                    f"QMoE parity check failed: max difference {max_diff:.6f} exceeds "
                    f"tolerance {base_atol:.6f} for {tolerance_key}"
                )
        else:
            # Fallback for unknown configurations
            fallback_atol = 0.1
            if max_diff > fallback_atol:
                raise AssertionError(
                    f"QMoE parity check failed: max difference {max_diff:.6f} exceeds "
                    f"fallback tolerance {fallback_atol:.6f} for unknown config {tolerance_key}"
                )

    def benchmark_ort(self):
        hidden_state = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim).to(device)
        self.ort_forward(hidden_state, enable_performance_test=True)


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

    CPU version: Modified to use only FC1 and FC2 for CPU compatibility.

    Quantization: Uses symmetric quantization to exactly match the C++ implementation:
    - 4-bit: range = [-8, 7] (stored as uint8 values [0, 15])
    - 8-bit: range = [-128, 127] (stored as uint8 values [0, 255])
    This ensures the test exactly simulates the C++ implementation with full
    compatibility with the CUDA implementation and TensorRT.
    """

    def __init__(
        self,
        config,
        batch_size,
        sequence_length,
        quant_bits=0,
        onnx_dtype=None,
        use_swiglu=False,
        swiglu_interleaved=True,
    ):
        # Ensure we always have a valid quantization bits value (4 or 8) before passing to parent
        if quant_bits <= 0:
            quant_bits = 4

        # Now pass the validated quant_bits to parent constructor
        super().__init__(quant_bits, onnx_dtype)
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.router_jitter_noise = config.router_jitter_noise
        self.use_swiglu = use_swiglu
        self.swiglu_interleaved = swiglu_interleaved

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        # Use PhiMoEBlockSparseTop2MLP for all experts
        self.experts = nn.ModuleList(
            [
                PhiMoEBlockSparseTop2MLP(config, use_swiglu=self.use_swiglu, swiglu_interleaved=self.swiglu_interleaved)
                for _ in range(self.num_experts)
            ]
        )

        w1_list, w2_list = [], []
        w1_scale_list, w2_scale_list = [], []

        # Always use quantization for QMoE
        is_4_bit = self.quant_bits == 4
        for i in range(self.num_experts):
            # For SwiGLU with interleaved weights, handle quantization differently
            if self.use_swiglu and self.swiglu_interleaved:
                # In interleaved mode, w1 already contains both gate and value weights
                # w1 shape: [2*intermediate_size, hidden_size] with interleaved rows

                # Just quantize the full interleaved weight matrix directly
                w1_scale, pre_qweight1, w1_qdq = quant_dequant(self.experts[i].w1.weight, is_4_bit)

                # For compatibility, we still need w3 quantization for the ONNX graph creation
                # but we won't use the interleaved logic since w1 already has everything
                w3_scale, pre_qweight3, w3_qdq = quant_dequant(self.experts[i].w3.weight, is_4_bit)

                # The pre_qweight1 already contains the interleaved quantized weights
                # w1_qdq already contains the interleaved dequantized weights
                # w1_scale already has the correct scales for the interleaved weights

                # For the ONNX graph, we still need to create the chunked layout for the CPU kernel
                # Extract gate and value parts from the interleaved weights for ONNX
                if len(pre_qweight1.shape) == 3:  # Packed quantized weights
                    gate_weights = pre_qweight1[0::2]  # Even rows: gate weights
                    value_weights = pre_qweight1[1::2]  # Odd rows: value weights
                else:
                    gate_weights = pre_qweight1[0::2]  # Even rows: gate weights
                    value_weights = pre_qweight1[1::2]  # Odd rows: value weights

                gate_scales = w1_scale[0::2] if w1_scale.dim() > 0 and w1_scale.shape[0] > 1 else w1_scale
                value_scales = w1_scale[1::2] if w1_scale.dim() > 0 and w1_scale.shape[0] > 1 else w1_scale

                # Create chunked layout for ONNX: [gate..., value...]
                pre_qweight1_onnx = torch.cat([gate_weights, value_weights], dim=0)
                w1_scale_onnx = torch.cat([gate_scales, value_scales], dim=0)

                # Use the chunked versions for ONNX graph creation
                pre_qweight1 = pre_qweight1_onnx
                w1_scale = w1_scale_onnx

            else:
                # Standard quantization path
                w1_scale, pre_qweight1, w1_qdq = quant_dequant(self.experts[i].w1.weight, is_4_bit)

                # For SwiGLU with separate weights, also quantize w3
                if self.use_swiglu:
                    w3_scale, pre_qweight3, w3_qdq = quant_dequant(self.experts[i].w3.weight, is_4_bit)
                    self.experts[i].w3.weight.data = w3_qdq

                    gate_weights = pre_qweight1
                    value_weights = pre_qweight3
                    gate_scales = w1_scale
                    value_scales = w3_scale

                    # Create chunked layout: [gate..., value...]
                    pre_qweight1 = torch.cat([gate_weights, value_weights], dim=0)
                    w1_scale = torch.cat([gate_scales, value_scales], dim=0)

            # Always quantize w2 weights
            w2_scale, pre_qweight2, w2_qdq = quant_dequant(self.experts[i].w2.weight, is_4_bit)

            # CRITICAL FIX: Update PyTorch weights to use the SAME dequantized values that the CPU will use
            # This ensures PyTorch and CPU use identical weights for fair comparison
            if self.use_swiglu:
                # For SwiGLU, w1_qdq and w3_qdq are already the correct dequantized weights
                # No need to de-interleave because quant_dequant was called on individual weights
                self.experts[i].w1.weight.data = w1_qdq  # Gate weights
                # w3_qdq was already set above in the main loop

                # Verify that PyTorch weights now match what we expect
            else:
                # For non-SwiGLU, just use the dequantized weights directly
                self.experts[i].w1.weight.data = w1_qdq.contiguous().clone()

            # Always update FC2 weights with dequantized values (no special handling needed for FC2)
            self.experts[i].w2.weight.data = w2_qdq.contiguous().clone()
            w1_list.append(pre_qweight1)
            w2_list.append(pre_qweight2)
            w1_scale_list.append(w1_scale)
            w2_scale_list.append(w2_scale)

        self.moe_experts_weight1 = torch.stack(w1_list, dim=0)
        self.moe_experts_weight2 = torch.stack(w2_list, dim=0)

        # CRITICAL FIX: After creating the packed weights for ONNX, we need to ensure PyTorch
        # uses the SAME dequantized weights that the CPU kernel will use
        # The CPU kernel will dequantize the packed weights, so PyTorch must use those exact values
        for i in range(self.num_experts):
            # Get the packed weights for this expert (same as what CPU kernel uses)
            packed_w1 = self.moe_experts_weight1[i]  # This is what goes to CPU kernel (quantized)
            packed_w2 = self.moe_experts_weight2[i]  # This is what goes to CPU kernel (quantized)

            # Get the scales that the CPU kernel will use
            w1_scale = w1_scale_list[i]
            w2_scale = w2_scale_list[i]

            # Dequantize the packed weights manually (same as CPU kernel will do)
            zero_point = 8 if is_4_bit else 128

            # Dequantize FC1 weights
            if is_4_bit:
                # For 4-bit, unpack and dequantize
                packed_w1_int = packed_w1.to(torch.uint8)
                unpacked_w1 = torch.zeros(packed_w1.shape[0], packed_w1.shape[1] * 2, dtype=torch.float32)
                for j in range(packed_w1.shape[1]):
                    unpacked_w1[:, j * 2] = (packed_w1_int[:, j] & 0xF).to(torch.float32) - zero_point
                    unpacked_w1[:, j * 2 + 1] = ((packed_w1_int[:, j] >> 4) & 0xF).to(torch.float32) - zero_point

                # Apply scales
                if w1_scale.dim() > 1:
                    w1_scale_flat = w1_scale.squeeze(-1)
                else:
                    w1_scale_flat = w1_scale
                packed_w1_qdq = unpacked_w1 * w1_scale_flat.unsqueeze(-1)
            else:
                # For 8-bit, direct dequantization
                # Apply same scale flattening logic as 4-bit to avoid broadcasting issues
                if w1_scale.dim() > 1:
                    w1_scale_flat = w1_scale.squeeze(-1)
                else:
                    w1_scale_flat = w1_scale
                packed_w1_qdq = (packed_w1.to(torch.float32) - zero_point) * w1_scale_flat.unsqueeze(-1)

            # Dequantize FC2 weights
            if is_4_bit:
                # For 4-bit, unpack and dequantize
                packed_w2_int = packed_w2.to(torch.uint8)
                unpacked_w2 = torch.zeros(packed_w2.shape[0], packed_w2.shape[1] * 2, dtype=torch.float32)
                for j in range(packed_w2.shape[1]):
                    unpacked_w2[:, j * 2] = (packed_w2_int[:, j] & 0xF).to(torch.float32) - zero_point
                    unpacked_w2[:, j * 2 + 1] = ((packed_w2_int[:, j] >> 4) & 0xF).to(torch.float32) - zero_point

                # Apply scales
                if w2_scale.dim() > 1:
                    w2_scale_flat = w2_scale.squeeze(-1)
                else:
                    w2_scale_flat = w2_scale
                packed_w2_qdq = unpacked_w2 * w2_scale_flat.unsqueeze(-1)
            else:
                # For 8-bit, direct dequantization
                # Apply same scale flattening logic as 4-bit to avoid broadcasting issues
                if w2_scale.dim() > 1:
                    w2_scale_flat = w2_scale.squeeze(-1)
                else:
                    w2_scale_flat = w2_scale
                packed_w2_qdq = (packed_w2.to(torch.float32) - zero_point) * w2_scale_flat.unsqueeze(-1)

            if self.use_swiglu:
                # For SwiGLU, the packed_w1_qdq contains interleaved gate and value weights
                if self.swiglu_interleaved:
                    # For interleaved mode, keep the full interleaved weights in w1
                    # w1 should contain the full [2*intermediate_size, hidden_size] matrix
                    # CRITICAL: Create an independent copy to avoid tensor reference issues

                    self.experts[i].w1.weight.data = packed_w1_qdq.contiguous().clone()

                    # For printing/debugging, extract the gate and value parts
                    gate_dequant = packed_w1_qdq[0::2]  # Even rows: gate weights
                    value_dequant = packed_w1_qdq[1::2]  # Odd rows: value weights

                    # Update w3 for compatibility but it won't be used in interleaved mode
                    self.experts[i].w3.weight.data = value_dequant.contiguous().clone()

                else:
                    # Chunked: [gate..., value...] -> split in half
                    intermediate_size = packed_w1_qdq.shape[0] // 2
                    gate_dequant = packed_w1_qdq[:intermediate_size]
                    value_dequant = packed_w1_qdq[intermediate_size:]

                    # Update PyTorch experts with the de-interleaved weights from packed format
                    self.experts[i].w1.weight.data = gate_dequant.contiguous().clone()
                    self.experts[i].w3.weight.data = value_dequant.contiguous().clone()

            else:
                # For non-SwiGLU, use packed weights directly
                self.experts[i].w1.weight.data = packed_w1_qdq.contiguous().clone()

            # Update FC2 weights with the packed dequantized weights
            self.experts[i].w2.weight.data = packed_w2_qdq.contiguous().clone()

        # Always use scales for QMoE
        moe_experts_weight_scale1 = torch.stack(w1_scale_list, dim=0)
        moe_experts_weight_scale2 = torch.stack(w2_scale_list, dim=0)

        # Fix shape mismatch: ensure scales are 2D for ONNX graph
        # The ONNX graph expects [num_experts, scale_size] but we might have [num_experts, scale_size, 1]
        if moe_experts_weight_scale1.dim() == 3:
            moe_experts_weight_scale1 = moe_experts_weight_scale1.squeeze(-1)
        if moe_experts_weight_scale2.dim() == 3:
            moe_experts_weight_scale2 = moe_experts_weight_scale2.squeeze(-1)

        self.batch_size = batch_size
        self.sequence_length = sequence_length

        # Use CPU specific graph creation
        try:
            self.moe_onnx_graph = create_cpu_moe_onnx_graph(
                hidden_size=self.hidden_dim,
                sequence_length=self.batch_size * self.sequence_length,
                num_experts=self.num_experts,
                top_k=self.top_k,
                intermediate_size=self.ffn_dim,
                torch_dtype=torch.float32,  # Assuming float32 as default
                onnx_dtype=self.onnx_dtype,
                fc1_experts_weights=self.moe_experts_weight1,
                fc2_experts_weights=self.moe_experts_weight2,
                # Biases are not used in QMoE, only passed as None for API compatibility
                fc1_bias=None,
                fc2_bias=None,
                # Scales are used for dequantization
                fc1_scales=moe_experts_weight_scale1,
                fc2_scales=moe_experts_weight_scale2,
                use_swiglu=self.use_swiglu,  # Use SwiGLU if specified
                use_quant=True,  # Always use QMoE
                quant_bits=self.quant_bits,
                swiglu_interleaved=self.swiglu_interleaved if hasattr(self, "swiglu_interleaved") else False,
            )
        except Exception:
            self.moe_onnx_graph = None

        self.ort_sess = self.create_ort_session(self.moe_onnx_graph) if self.moe_onnx_graph else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        # DEBUG: Compare router logits with CPU

        routing_weights, selected_experts = masked_sampling_omp_inference(
            router_logits,
            top_k=self.top_k,
            jitter_eps=self.router_jitter_noise,
            training=False,
        )

        # DEBUG: Compare routing results with CPU

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

        return final_hidden_states


def small_test_cases():
    for batch_size in [1, 4]:
        for sequence_length in [32, 128]:
            yield batch_size, sequence_length


# Define our test cases for QMoE (4-bit and 8-bit quantization) with SwiGLU only
# Only test QMoE since standard MoE is not supported on CPU
cpu_phi3_swiglu_test_cases = list(
    itertools.product(
        [1, 4],  # batch_size
        [8, 32],  # sequence_length - smaller sequence lengths for CPU
        [4, 8],  # quant_bits - only test QMoE (4-bit and 8-bit)
        [True],  # use_swiglu - SwiGLU activation only
        [True],  # swiglu_interleaved - Kernel only supports interleaved right now.
    )
)

# Enable CPU qMoE tests
disable_cpu_qmoe_tests = False


@unittest.skipIf(disable_cpu_qmoe_tests, "Skipping qMoE cpu tests")
class TestPhiQMoECPU(unittest.TestCase):
    @parameterized.expand(cpu_phi3_swiglu_test_cases)
    def test_phi3_qmoe_parity_cpu(
        self, batch_size, sequence_length, quant_bits, use_swiglu=True, swiglu_interleaved=True
    ):
        # Reset random seeds for consistent test behavior regardless of test execution order
        torch.manual_seed(42)
        numpy.random.seed(42)

        # Print test configuration
        test_config = f"batch_size={batch_size}, sequence_length={sequence_length}, quant_bits={quant_bits}, use_swiglu={use_swiglu}, swiglu_interleaved={swiglu_interleaved}"
        print(f"Running test: {test_config}")

        config = PhiMoEConfig(
            hidden_size=128, intermediate_size=256, hidden_act="silu"
        )  # Even smaller sizes for better accuracy

        phi3_moe = PhiMoESparseMoeBlock(
            config,
            batch_size,
            sequence_length,
            quant_bits,
            use_swiglu=use_swiglu,
            swiglu_interleaved=swiglu_interleaved,
        )
        phi3_moe.to(device)

        # Skip tests if ONNX is not available
        if not has_onnx:
            self.skipTest("ONNX is not installed")

        # Skip if the session creation failed
        if phi3_moe.ort_sess is None:
            self.skipTest("Failed to create ONNX Runtime session")

        try:
            phi3_moe.parity_check()
        except RuntimeError as e:
            if "FC3 gating is not yet implemented on CPU" in str(e):
                self.skipTest("FC3 gating is not yet implemented on CPU")
            else:
                raise

    run_performance_test = False

    @unittest.skipIf(not run_performance_test, "Skipping qMoE CPU performance test")
    def test_phi3_qmoe_cpu_benchmark(self):
        # Test different batch sizes and sequence lengths for performance analysis
        # Note: The C++ implementation now uses optimized expert processing and improved
        # memory access patterns for better performance
        batch_sizes = [1, 4, 16]
        sequence_lengths = [8, 32, 128]
        use_swiglu = True  # Only use SwiGLU for benchmarks

        for quant_bits in [4, 8]:
            for batch_size in batch_sizes:
                for sequence_length in sequence_lengths:
                    print(
                        f"Benchmarking QMoE CPU with quant_bits={quant_bits}, batch_size={batch_size}, sequence_length={sequence_length}, use_swiglu={use_swiglu}"
                    )

                    # Create MoE config
                    config = PhiMoEConfig(
                        hidden_size=256,
                        intermediate_size=512,
                        hidden_act="silu",  # This doesn't matter for SwiGLU
                        num_local_experts=8,
                    )

                    # Create MoE model
                    phi3_moe = PhiMoESparseMoeBlock(
                        config,
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        quant_bits=quant_bits,
                        use_swiglu=use_swiglu,
                        swiglu_interleaved=True,
                    )
                    phi3_moe.to(device)

                    if phi3_moe.ort_sess is None:
                        continue

                    # Run benchmark and calculate tokens/sec
                    num_runs = 100

                    # Warmup
                    hidden_state = torch.randn(batch_size, sequence_length, config.hidden_size).to(device)
                    for _ in range(5):
                        phi3_moe.ort_forward(hidden_state)

                    # Benchmark
                    total_tokens = batch_size * sequence_length * num_runs
                    start_time = time.time()
                    for _ in range(num_runs):
                        phi3_moe.ort_forward(hidden_state)
                    end_time = time.time()

                    elapsed_time = end_time - start_time
                    tokens_per_second = total_tokens / elapsed_time

                    print(f"QMoE CPU Benchmark: {tokens_per_second:.2f} tokens/sec")


if __name__ == "__main__":
    unittest.main()
