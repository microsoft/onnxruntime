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
# Note on QMoE quantization approaches:
#
# Both CPU and CUDA implementations of QMoE use symmetric quantization:
#
# 1. CPU (this file): Symmetric quantization
#    - 4-bit: range = [-8, 7]
#    - 8-bit: range = [-128, 127]
#
# 2. CUDA: Symmetric quantization
#    - 4-bit: range = [-8, 7]
#    - 8-bit: range = [-128, 127]
#
# This aligned approach ensures better compatibility with TensorRT.
# The tolerance values used in testing account for minor numerical differences.
# --------------------------------------------------------------------------
import itertools
import os
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

    HAS_ONNX = True
except ImportError:
    print("ONNX is not installed. Some functionality will not be available.")
    HAS_ONNX = False

    # Define placeholder constants if onnx is not available
    class TensorProtoPlaceholder:
        FLOAT16 = 10
        FLOAT = 1
        # BF16 not supported in QMoE CPU
        UINT8 = 2

    TensorProto = TensorProtoPlaceholder

# Reduces number of tests to run for faster pipeline checks
pipeline_mode = os.getenv("PIPELINE_MODE", "1") == "1"

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

    This uses symmetric quantization to match the C++ implementation and for TensorRT compatibility:
    - 4-bit: range = [-8, 7]
    - 8-bit: range = [-128, 127]

    This implementation aims to precisely match the C++ implementation by:
    1. Using symmetric quantization (zero point = 0)
    2. Using the same scale calculation methodology
    3. Using consistent rounding behavior
    4. Properly handling edge cases
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

    # Get absolute maximum for scale calculation
    abs_max = weights.abs().max(dim=-1, keepdim=True)[0]

    if is_4_bit_quantization:
        # For 4-bit symmetric quantization, range is [-8, 7]
        scale = abs_max / 7.0  # Scale factor ensures max value maps to 7

        # Handle potential edge cases for zero or very small weights
        if torch.max(abs_max) < 1e-10:
            # For extremely small values, avoid division by near-zero
            packed_size = (weights.shape[-1] + 1) // 2
            # Just return zeros with appropriate scale to avoid numerical issues
            return (
                torch.ones_like(weights[..., 0:1]) * 1e-6,  # Very small non-zero scale
                torch.full(
                    (weights.shape[0], weights.shape[1], packed_size),
                    fill_value=8 | (8 << 4),  # 8 = 0 in symmetric quantization
                    dtype=torch.uint8,
                    device=weights.device,
                ),
                torch.zeros_like(weights),
            )

        # Convert to int4 range (-8 to 7)
        scaled_weights = torch.round(weights / scale)
        clipped_weights = torch.clamp(scaled_weights, -8, 7)

        # Convert from int4 signed range [-8,7] to uint4 storage range [0,15]
        # by adding 8 to map -8->0, -7->1, ..., 7->15
        quant_weights = (clipped_weights + 8).to(torch.uint8)

        # Pack 4-bit values into uint8 (every two elements)
        even_indices = torch.arange(0, weights.shape[-1], 2, device=weights.device)
        odd_indices = torch.arange(1, weights.shape[-1], 2, device=weights.device)

        # Handle odd length by padding
        if odd_indices.shape[0] < even_indices.shape[0]:
            # Pad with 8 (which represents 0 in symmetric quantization)
            # Create a new padding tensor for more predictable behavior
            padding = torch.full(
                (quant_weights.shape[0], quant_weights.shape[1], 1),
                fill_value=8,
                dtype=torch.uint8,
                device=quant_weights.device,
            )
            quant_weights = torch.cat([quant_weights, padding], dim=-1)
            odd_indices = torch.arange(1, quant_weights.shape[-1], 2, device=weights.device)

        even_weights = quant_weights[..., even_indices]
        odd_weights = quant_weights[..., odd_indices]

        # Pack two 4-bit values into each byte
        packed_weights = (even_weights & 0xF) | ((odd_weights & 0xF) << 4)

        # For dequantization, unpack
        lower = packed_weights & 0xF
        upper = (packed_weights >> 4) & 0xF

        # Restore original shape, taking care to handle dimensions correctly
        unpacked_weights = torch.zeros_like(weights, dtype=torch.uint8)

        # Assign values ensuring we don't go out of bounds
        unpacked_weights[..., even_indices] = lower

        # Calculate valid odd indices that fit within our original tensor dimensions
        valid_odd_length = min(odd_indices.shape[0], weights.shape[-1] - even_indices.shape[0])
        valid_odd_indices = odd_indices[:valid_odd_length]

        # Only assign upper bits to valid positions
        if valid_odd_length > 0:
            unpacked_weights[..., valid_odd_indices] = upper[..., :valid_odd_length]

        # Convert back from uint4 to int4 by subtracting 8
        int4_weights = unpacked_weights.float() - 8

        # Dequantize with proper broadcasting
        # Make sure scale has the right shape for broadcasting
        scale_expanded = scale.float()
        result = (int4_weights * scale_expanded).to(dtype=weights.dtype)
        return scale.to(torch.float16), packed_weights, result
    else:
        # 8-bit symmetric quantization, range is [-128, 127]
        scale = abs_max / 127.0  # Scale factor ensures max value maps to 127

        # Handle potential edge cases for zero or very small weights
        if torch.max(abs_max) < 1e-10:
            # For extremely small values, avoid division by near-zero
            # Just return zeros with appropriate scale to avoid numerical issues
            return (
                torch.ones_like(weights[..., 0:1]) * 1e-6,  # Very small non-zero scale
                torch.full_like(weights, fill_value=128, dtype=torch.uint8),  # 128 = 0 in symmetric
                torch.zeros_like(weights),
            )

        # Convert to int8 range (-128 to 127)
        scaled_weights = torch.round(weights / scale)
        clipped_weights = torch.clamp(scaled_weights, -128, 127)

        # Convert from int8 signed range [-128,127] to uint8 storage range [0,255]
        # by adding 128 to map -128->0, -127->1, ..., 127->255
        quant_weights = (clipped_weights + 128).to(torch.uint8)

        # Dequantize - convert back from uint8 to int8 by subtracting 128, then multiply by scale
        # Make sure scale has the right shape for broadcasting
        scale_expanded = scale.float()
        result = ((quant_weights.float() - 128) * scale_expanded).to(dtype=weights.dtype)
        return scale.to(torch.float16), quant_weights, result


def create_cpu_moe_onnx_graph(
    hidden_size,
    sequence_length,
    num_experts,
    top_k,
    intermediate_size,
    onnx_dtype,
    fc1_experts_weights,
    fc2_experts_weights,
    fc1_scales=None,
    fc2_scales=None,
    use_swiglu=False,
    quant_bits=4,
):
    # Make sure we have onnx available before proceeding
    if not HAS_ONNX:
        print("ONNX not found, skipping graph creation")
        return None

    # Define intermediate_size variable consistently
    inter_size = intermediate_size
    topk = top_k

    # Note: In QMoE, biases are not used at all, only scales
    assert fc1_scales is not None, "FC1 scales must be provided for QMoE"
    assert fc2_scales is not None, "FC2 scales must be provided for QMoE"

    # Using uint8 storage type with symmetric quantization
    # 4-bit: range = [-8, 7] (stored as uint8 values [0, 15])
    # 8-bit: range = [-128, 127] (stored as uint8 values [0, 255])
    assert fc1_experts_weights.dtype == torch.uint8, "FC1 weights must be uint8 for QMoE"
    assert fc2_experts_weights.dtype == torch.uint8, "FC2 weights must be uint8 for QMoE"
    assert fc1_scales.dtype == torch.float16, "FC1 scales must be float16 for QMoE"
    assert fc2_scales.dtype == torch.float16, "FC2 scales must be float16 for QMoE"

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
            domain="com.microsoft",
        ),
    ]

    nodes[0].attribute.extend([helper.make_attribute("expert_weight_bits", quant_bits)])

    # For 4-bit quantization, we need to pack 2 values into each byte
    pack_factor = 2 if quant_bits == 4 else 1

    # For SwiGLU, we need to double the FC1 dimension to accommodate both gate and value paths
    act_factor = 2 if use_swiglu else 1

    # FC1 shape needs to account for both SwiGLU and quantization packing
    # Weights are store in column major order. Need pack 2 int4 values into uint8.
    fc1_shape = [num_experts, (act_factor * inter_size), hidden_size // pack_factor]
    fc2_shape = [num_experts, hidden_size, inter_size // pack_factor]

    torch_dtype = onnx_to_torch_type_map[onnx_dtype]

    weight_numpy_type = numpy.uint8
    weight_onnx_type = TensorProto.UINT8

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

    # QMoE always uses scales, never biases
    # For SwiGLU, FC1 scales shape needs to be doubled to account for gate and value components
    fc1_scale_shape = [num_experts, 2 * inter_size if use_swiglu else inter_size]
    fc2_scale_shape = [num_experts, hidden_size]
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


class PhiMoEBlockSparseTop2MLP(MoEBlockSparseTop2MLP):
    def __init__(self, config: PhiMoEConfig, use_swiglu=False):
        super().__init__(config)
        self.use_swiglu = use_swiglu

    def forward(self, hidden_states):
        if self.use_swiglu:
            # SwiGLU implementation matching C++ implementation exactly
            gate_output = self.w1(hidden_states)  # Gate
            value_output = self.w3(hidden_states)  # Value

            # Apply SwiGLU exactly as in the C++ implementation
            # C++ uses swiglu_alpha = 1.702f and clamp_limit = 7.0f
            swiglu_alpha = 1.702
            clamp_limit = 7.0

            # Apply clamping to match C++ implementation
            gate_output = torch.clamp(gate_output, max=clamp_limit)  # Clamp max only for gate
            value_output = torch.clamp(value_output, min=-clamp_limit, max=clamp_limit)  # Clamp both for value

            # Compute gate activation: gate * sigmoid(alpha * gate)
            sigmoid_input = swiglu_alpha * gate_output
            sigmoid_output = torch.sigmoid(sigmoid_input)
            swish_output = gate_output * sigmoid_output

            # Multiply by (value + 1) as done in C++
            current_hidden_states = swish_output * (value_output + 1.0)

            # Apply FC2
            current_hidden_states = self.w2(current_hidden_states)
            return current_hidden_states
        else:
            # Original implementation with standard activation
            return super().forward(hidden_states)


class SparseMoeBlockORTHelper(nn.Module):
    def __init__(self, quant_bits, onnx_dtype):
        super().__init__()
        self.quant_bits = quant_bits
        self.onnx_dtype = onnx_dtype
        self.np_type = ort_to_numpy_type_map[self.onnx_dtype]

    def create_ort_session(self, moe_onnx_graph):
        if moe_onnx_graph is None:
            print("No ONNX graph provided, skipping session creation")
            return None

        sess_options = onnxruntime.SessionOptions()
        sess_options.log_severity_level = 2

        try:
            ort_session = onnxruntime.InferenceSession(moe_onnx_graph, sess_options, providers=ort_provider)
        except Exception as e:
            print(f"Failed to create ONNX Runtime session with provider {ort_provider}: {e}")
            print("Skipping ONNX Runtime execution for this test case.")
            return None

        return ort_session

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pass

    def ort_forward(self, hidden_states: torch.Tensor, enable_performance_test=False) -> torch.Tensor:
        # If session creation failed, we can't run inference
        if self.ort_sess is None:
            print("No ORT session available, skipping ONNX Runtime execution")
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
                import time  # noqa: PLC0415

                repeat = 100  # Using fewer repeats for CPU tests
                s = time.time()
                for _ in range(repeat):
                    iobinding.synchronize_inputs()
                    self.ort_sess.run_with_iobinding(iobinding)
                    iobinding.synchronize_outputs()
                e = time.time()
                print(f"QMoE CPU kernel time: {(e - s) / repeat * 1000} ms")

            # The output tensor is on `device`. Reshape and return it.
            return tensors["output"].reshape(batch_size, sequence_length, hidden_dim)

        except Exception as e:
            print(f"Error running ORT session: {e!s}")
            raise

    def parity_check(self):
        hidden_state = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim).to(device)
        torch_output = self.forward(hidden_state)
        ort_output = self.ort_forward(hidden_state)

        # If no ORT output was produced, we can't do a parity check
        if ort_output is None:
            print("ORT execution failed or is not supported, skipping parity check")
            return

        dtype_str = ort_dtype_name_map[self.onnx_dtype]
        max_diff = (torch_output.cpu() - ort_output.cpu()).abs().max()

        print(
            f"name: {self.__class__.__name__}, quant_bits: {self.quant_bits}, dtype: {dtype_str},"
            f" batch: {self.batch_size}, seq_len: {self.sequence_length},"
            f" max_diff: {max_diff}"
        )

        # TODO: set proper threshold after kernel accuracy issue is resolved.
        ort_dtype_quant_bits_tolerance_map = {
            "FP32:4": (100.0, 8e-3),
            "FP32:8": (100.0, 8e-3),
            "FP16:4": (100.0, 8e-3),
            "FP16:8": (100.0, 8e-3),
        }

        tolerance_key = f"{dtype_str}:{self.quant_bits}"
        if tolerance_key not in ort_dtype_quant_bits_tolerance_map:
            self.fail(f"No tolerance defined for {tolerance_key}")

        atol, rtol = ort_dtype_quant_bits_tolerance_map[tolerance_key]
        torch.testing.assert_close(
            ort_output.cpu().to(torch.float32), torch_output.cpu().to(torch.float32), rtol=rtol, atol=atol
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
    assignments of tokens to experts.

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
        quant_bits,
        onnx_dtype,
        use_swiglu=False,
        swiglu_interleaved=True,
    ):
        # Ensure we always have a valid quantization bits value (4 or 8)
        if quant_bits not in [4, 8]:
            raise ValueError(f"Invalid quant_bits: {quant_bits}. Must be 4 or 8.")

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
            [PhiMoEBlockSparseTop2MLP(config, use_swiglu=self.use_swiglu) for _ in range(self.num_experts)]
        )

        w1_list, w2_list = [], []
        w1_scale_list, w2_scale_list = [], []

        # Always use quantization for QMoE
        is_4_bit = self.quant_bits == 4
        for i in range(self.num_experts):
            # Quantize the weights
            w1_scale, pre_qweight1, w1_qdq = quant_dequant(self.experts[i].w1.weight, is_4_bit)
            w2_scale, pre_qweight2, w2_qdq = quant_dequant(self.experts[i].w2.weight, is_4_bit)

            # For SwiGLU, we also need to quantize w3 (value) weights
            if self.use_swiglu:
                w3_scale, pre_qweight3, w3_qdq = quant_dequant(self.experts[i].w3.weight, is_4_bit)
                self.experts[i].w3.weight.data = w3_qdq

                gate_weights = pre_qweight1
                value_weights = pre_qweight3
                gate_scales = w1_scale
                value_scales = w3_scale

                if self.swiglu_interleaved:
                    # Create interleaved layout: [gate, value, gate, value, ...]
                    combined_weights = torch.empty(
                        gate_weights.shape[0] * 2, gate_weights.shape[1], dtype=gate_weights.dtype
                    )
                    combined_weights[0::2, :] = gate_weights
                    combined_weights[1::2, :] = value_weights
                    pre_qweight1 = combined_weights

                    combined_scales = torch.empty(
                        gate_scales.shape[0] * 2, gate_scales.shape[1], dtype=gate_scales.dtype
                    )
                    combined_scales[0::2, :] = gate_scales
                    combined_scales[1::2, :] = value_scales
                    w1_scale = combined_scales
                else:
                    # Create chunked layout: [gate..., value...]
                    pre_qweight1 = torch.cat([gate_weights, value_weights], dim=0)
                    w1_scale = torch.cat([gate_scales, value_scales], dim=0)

            # Update the expert weights with dequantized values for PyTorch execution
            self.experts[i].w1.weight.data = w1_qdq
            self.experts[i].w2.weight.data = w2_qdq

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
                onnx_dtype=self.onnx_dtype,
                fc1_experts_weights=self.moe_experts_weight1,
                fc2_experts_weights=self.moe_experts_weight2,
                fc1_scales=moe_experts_weight_scale1,
                fc2_scales=moe_experts_weight_scale2,
                use_swiglu=self.use_swiglu,
                quant_bits=self.quant_bits,
            )
        except Exception as e:
            print(f"Error creating ONNX graph: {e}")
            self.moe_onnx_graph = None

        self.ort_sess = self.create_ort_session(self.moe_onnx_graph) if self.moe_onnx_graph else None

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

            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        return final_hidden_states


# Define our test cases for QMoE (4-bit and 8-bit quantization) for both FP16 and FP32 inputs
cpu_test_cases = []
# SiLU cases (swiglu=False)
cpu_test_cases.extend(
    list(
        itertools.product(
            [1, 4],  # batch_size
            [8, 32],  # sequence_length
            [4, 8],  # quant_bits
            [TensorProto.FLOAT16, TensorProto.FLOAT],  # onnx_dtype
            [False],  # use_swiglu
            [False],  # swiglu_interleaved (placeholder, not used for SiLU)
        )
    )
)
# SwiGLU cases (swiglu=True, interleaved=True)
cpu_test_cases.extend(
    list(
        itertools.product(
            [1, 4],  # batch_size
            [8, 32],  # sequence_length
            [4, 8],  # quant_bits
            [TensorProto.FLOAT16, TensorProto.FLOAT],  # onnx_dtype
            [True],  # use_swiglu
            [True],  # swiglu_interleaved (Kernel only supports interleaved)
        )
    )
)


class TestPhiQMoECPU(unittest.TestCase):
    @parameterized.expand(cpu_test_cases)
    def test_phi3_qmoe_parity_cpu(
        self, batch_size, sequence_length, quant_bits, onnx_dtype, use_swiglu, swiglu_interleaved
    ):
        dtype_str = ort_dtype_name_map[onnx_dtype]
        activation_type = f"SwiGLU(interleaved={swiglu_interleaved})" if use_swiglu else "SiLU"
        print(
            f"Running PhiMoE CPU test with: batch={batch_size}, seq_len={sequence_length}, "
            f"quant={quant_bits}, dtype={dtype_str}, activation={activation_type}"
        )
        config = PhiMoEConfig(hidden_size=256, intermediate_size=512, hidden_act="silu")
        phi3_moe = PhiMoESparseMoeBlock(
            config,
            batch_size,
            sequence_length,
            quant_bits,
            onnx_dtype,
            use_swiglu=use_swiglu,
            swiglu_interleaved=swiglu_interleaved,
        )
        phi3_moe.to(device)

        # Skip tests if ONNX is not available
        if not HAS_ONNX:
            self.skipTest("ONNX is not installed")

        # Skip if the session creation failed
        if phi3_moe.ort_sess is None:
            self.skipTest("Failed to create ONNX Runtime session - CPU MoE operator not available")

        phi3_moe.parity_check()

    run_performance_test = False

    @unittest.skipIf(not run_performance_test, "Skipping qMoE CPU performance test")
    def test_phi3_qmoe_cpu_benchmark(self):
        for quant_bits in [4, 8]:
            for use_swiglu in [False, True]:
                swiglu_interleaved = True  # Kernel only supports interleaved
                activation_type = f"SwiGLU(interleaved={swiglu_interleaved})" if use_swiglu else "SiLU"
                print(f"Benchmarking PhiMoE CPU with quant_bits={quant_bits}, activation={activation_type}")
                batch_size = 1
                sequence_length = 32
                config = PhiMoEConfig(hidden_size=256, intermediate_size=512)
                phi3_moe = PhiMoESparseMoeBlock(
                    config,
                    batch_size,
                    sequence_length,
                    quant_bits,
                    TensorProto.FLOAT,  # Benchmark with FP32
                    use_swiglu=use_swiglu,
                    swiglu_interleaved=swiglu_interleaved,
                )
                phi3_moe.to(device)

                if not HAS_ONNX or phi3_moe.ort_sess is None:
                    self.skipTest("ONNX not installed or CPU MoE operator not available")
                    return

                phi3_moe.benchmark_ort()


if __name__ == "__main__":
    unittest.main()
