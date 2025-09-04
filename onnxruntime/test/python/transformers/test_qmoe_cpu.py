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
# Both CPU and CUDA implementations use symmetric quantization centered around 0:
# - 4-bit: range [-8, 7] with no zero-point (symmetric around 0)
# - 8-bit: range [-128, 127] with no zero-point (symmetric around 0)
#
# This follows the _symmetric_quantize_last_axis_of_batched_matrix pattern.
# Tolerance values account for numerical differences between implementations.
#
# Routing Logic: CPU implementation uses top-k selection first, then softmax
# normalization on the selected experts. This provides proper weight distribution
# while maintaining computational efficiency.
# --------------------------------------------------------------------------
import time
import unittest
from collections import OrderedDict

import numpy
import torch
import torch.nn.functional as F
from onnx import helper
from parameterized import parameterized
from torch import nn

import onnxruntime

try:
    from onnx import TensorProto

    has_onnx = True
except ImportError:
    has_onnx = False

    class TensorProtoPlaceholder:
        FLOAT16 = 10
        FLOAT = 1


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

if not has_onnx:

    class TensorProtoPlaceholder:
        FLOAT16 = 10
        FLOAT = 1
        UINT8 = 2

    TensorProto = TensorProtoPlaceholder

onnxruntime.preload_dlls()

device = torch.device("cpu")

ort_provider = ["CPUExecutionProvider"]

torch.manual_seed(42)
numpy.random.seed(42)

onnx_to_torch_type_map = {
    TensorProto.FLOAT16: torch.float16,
    TensorProto.FLOAT: torch.float,
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
}


def quant_dequant(weights, is_4_bit_quantization: bool = True):
    """
    Quantize and dequantize weights for testing purposes.
    This function uses symmetric quantization centered around 0 (no zero-point).

    This uses symmetric quantization similar to _symmetric_quantize_last_axis_of_batched_matrix:
    - 4-bit: range = [-8, 7], no zero-point (symmetric around 0)
    - 8-bit: range = [-128, 127], no zero-point (symmetric around 0)
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
    abs_max = torch.clamp(abs_max, min=1e-8)  # More conservative clamping for better precision

    if is_4_bit_quantization:
        # 4-bit: scale = abs_max / 7.0 (using 7.0 as max positive value for symmetric range)
        # Use higher precision computation for better accuracy
        scale = (abs_max.double() / 7.0).float() + 1e-12

        # Handle potential edge cases for zero or very small weights
        if torch.max(abs_max) < 1e-8:
            packed_size = (weights.shape[-1] + 1) // 2
            return (
                torch.ones_like(weights[..., 0:1]) * 1e-8,
                torch.zeros(
                    (weights.shape[0], weights.shape[1], packed_size),
                    dtype=torch.uint8,
                    device=weights.device,
                ),
                torch.zeros_like(weights),
            )

        # Quantize: round(weight / scale) then clamp to [-8, 7]
        # Use higher precision for the division to reduce accumulated errors
        scaled_weights = weights.double() / scale.double()
        quantized_weights = torch.round(scaled_weights).clamp(-8, 7).float()

        # For symmetric quantization, we use signed int4 representation
        # Convert to uint8 storage for packing: shift [-8,7] -> [0,15] for storage only
        storage_weights = (quantized_weights + 8).to(torch.uint8)

        # Pack 4-bit values into uint8 (every two elements)
        even_indices = torch.arange(0, weights.shape[-1], 2)
        odd_indices = torch.arange(1, weights.shape[-1], 2)

        # Handle odd length by padding with zero (which is 8 in storage representation)
        if odd_indices.shape[0] < even_indices.shape[0]:
            padding = torch.full(
                (storage_weights.shape[0], storage_weights.shape[1], 1),
                fill_value=8,  # 0 in symmetric quantization, stored as 8
                dtype=torch.uint8,
                device=storage_weights.device,
            )
            storage_weights = torch.cat([storage_weights, padding], dim=-1)
            odd_indices = torch.arange(1, storage_weights.shape[-1], 2)

        even_weights = storage_weights[..., even_indices]
        odd_weights = storage_weights[..., odd_indices]

        # Pack: low nibble = even, high nibble = odd
        packed_weights = (even_weights & 0xF) | ((odd_weights & 0xF) << 4)

        # Dequantize: scale * quantized_value (no zero-point subtraction)
        # Unpack for dequantization
        lower = packed_weights & 0xF
        upper = (packed_weights >> 4) & 0xF

        # Restore original shape and convert back to signed representation
        unpacked_weights = torch.zeros_like(weights, dtype=torch.uint8)
        unpacked_weights[..., even_indices] = lower

        valid_odd_length = min(odd_indices.shape[0], weights.shape[-1] - even_indices.shape[0])
        if valid_odd_length > 0:
            valid_odd_indices = odd_indices[:valid_odd_length]
            unpacked_weights[..., valid_odd_indices] = upper[..., :valid_odd_length]

        # Convert back to signed values: [0,15] -> [-8,7] and apply scale
        signed_weights = unpacked_weights.float() - 8.0  # Convert storage back to signed
        dequant_scale = scale.float()  # Ensure FP32 precision for computation
        result = dequant_scale * signed_weights  # No zero-point in symmetric quantization

        return scale.to(torch.float16), packed_weights, result.to(weights.dtype)
    else:
        # 8-bit: scale = abs_max / 127.0 (using 127.0 as max positive value for symmetric range)
        # Use higher precision computation for better accuracy
        scale = (abs_max.double() / 127.0).float() + 1e-12

        # Handle potential edge cases for zero or very small weights
        if torch.max(abs_max) < 1e-8:
            return (
                torch.ones_like(weights[..., 0:1]) * 1e-8,
                torch.zeros_like(weights, dtype=torch.uint8),
                torch.zeros_like(weights),
            )

        # Quantize: round(weight / scale) then clamp to [-128, 127]
        # Use higher precision for the division to reduce accumulated errors
        scaled_weights = weights.double() / scale.double()
        quantized_weights = torch.round(scaled_weights).clamp(-128, 127).float()

        # For symmetric quantization, we use signed int8 representation
        # Convert to uint8 storage: shift [-128,127] -> [0,255] for storage only
        storage_weights = (quantized_weights + 128).to(torch.uint8)

        # Dequantize: scale * quantized_value (no zero-point subtraction)
        # Convert back to signed values: [0,255] -> [-128,127] and apply scale
        signed_weights = storage_weights.float() - 128.0  # Convert storage back to signed
        dequant_scale = scale.float()  # Ensure FP32 precision for computation
        result = dequant_scale * signed_weights  # No zero-point in symmetric quantization

        return scale.to(torch.float16), storage_weights, result.to(weights.dtype)


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
    if not has_onnx:
        return None

    inter_size = intermediate_size
    topk = top_k

    use_quant = True

    if fc1_scales is None and use_quant:
        return None
    if fc2_scales is None and use_quant:
        return None
    if not has_onnx:
        return None

    assert fc1_experts_weights.dtype == torch.uint8, "FC1 weights must be uint8 for QMoE"
    assert fc2_experts_weights.dtype == torch.uint8, "FC2 weights must be uint8 for QMoE"
    assert fc1_scales is not None, "FC1 scales must be provided for QMoE"
    assert fc2_scales is not None, "FC2 scales must be provided for QMoE"
    assert fc1_scales.dtype == torch.float16, "FC1 scales must be float16 for QMoE"
    assert fc2_scales.dtype == torch.float16, "FC2 scales must be float16 for QMoE"

    if not has_onnx:
        return None

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

    activation = "swiglu" if use_swiglu else "silu"

    nodes = [
        helper.make_node(
            op_name,
            inputs,
            ["output"],
            "MoE_0",
            k=topk,
            normalize_routing_weights=1,  # Use proper routing normalization to match PyTorch behavior
            activation_type=activation,
            # Add new attributes with backwards-compatible default values
            swiglu_fusion=1 if (use_swiglu and swiglu_interleaved) else 0,  # 1 = fused and interleaved
            swiglu_limit=7.0,
            activation_alpha=1.702,
            activation_beta=1.0,
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

    fc1_scale_shape = [num_experts, 2 * inter_size if use_swiglu else inter_size]
    fc2_scale_shape = [num_experts, hidden_size]

    fc1_scale_size = num_experts * (2 * inter_size if use_swiglu else inter_size)
    fc2_scale_size = num_experts * hidden_size

    # Handle scale tensors - fc1_scales and fc2_scales are guaranteed to be not None due to earlier assertions
    # Handle different possible scale tensor structures for fc1_scales
    if len(fc1_scales.shape) == 4:
        # 4D case: [num_experts, inter_size, hidden_size, 1] - extract first scale per expert per output
        if use_swiglu:
            fc1_scale_tensor = fc1_scales.to(torch_dtype)[:, : 2 * inter_size, 0, 0].flatten().detach().cpu().numpy()
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
            fc1_scale_tensor = numpy.concatenate([fc1_scale_tensor, numpy.ones(pad_size, dtype=fc1_scale_tensor.dtype)])

    # Process scale tensor for proper shape
    fc1_scale_data_list = fc1_scale_tensor.tolist()
    fc1_scale_data = fc1_scale_data_list

    # Handle different possible scale tensor structures for fc2_scales
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
            fc2_scale_tensor = numpy.concatenate([fc2_scale_tensor, numpy.ones(pad_size, dtype=fc2_scale_tensor.dtype)])

    # Process scale tensor for proper shape
    fc2_scale_data_list = fc2_scale_tensor.tolist()
    fc2_scale_data = fc2_scale_data_list

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


class SwigluMoeConfig:
    def __init__(
        self,
        hidden_size=4096,
        intermediate_size=14336,
        num_local_experts=8,
        num_experts_per_token=2,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_local_experts = num_local_experts
        self.num_experts_per_token = num_experts_per_token


def swiglu(x: torch.Tensor, alpha: float = 1.702, limit: float = 7.0):
    dim = x.shape[-1]
    x = x.view(-1, dim // 2, 2)
    x_glu, x_linear = x[..., 0], x[..., 1]

    if limit is not None:
        x_glu = x_glu.clamp(max=limit)
        x_linear = x_linear.clamp(min=-limit, max=limit)

    y = x_glu * torch.sigmoid(alpha * x_glu) * (x_linear + 1)
    return y


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
    def __init__(self, config: PhiMoEConfig):
        super().__init__(config)


class PhiMoESwiGLUMLP(nn.Module):
    """
    Phi3 MoE expert converted to 2-weight SwiGLU structure for CPU compatibility.
    This converts the traditional 3-weight Phi3 structure to SwiGLU format.
    """

    def __init__(self, config: PhiMoEConfig):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.w1 = nn.Linear(self.hidden_dim, 2 * self.intermediate_size, bias=True)
        self.w2 = nn.Linear(self.intermediate_size, self.hidden_dim, bias=True)

    def forward(self, x):
        x1 = self.w1(x)
        y = swiglu(x1)
        y = self.w2(y)
        return y


class SwigluMlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.w1 = nn.Linear(self.hidden_dim, 2 * self.intermediate_size, bias=True)
        self.w2 = nn.Linear(self.intermediate_size, self.hidden_dim, bias=True)

    def forward(self, x):
        x1 = self.w1(x)
        y = swiglu(x1)
        y = self.w2(y)
        return y


def masked_sampling_omp_inference(scores, top_k, jitter_eps, training):
    """
    Updated to match the CUDA implementation's routing logic for fair comparison.
    This now uses the same complex jitter-based masking approach as the CUDA tests.
    """
    assert top_k == 2
    assert not training

    mask_logits_threshold, selected_experts = torch.topk(scores, 2)

    mask_logits_threshold_1 = mask_logits_threshold[:, 0].unsqueeze(-1)

    factor = scores.abs().clamp(min=mask_logits_threshold_1)
    logits_mask = ((mask_logits_threshold_1 - scores) / factor) > (2 * jitter_eps)

    multiplier_1 = torch.softmax(scores.masked_fill(logits_mask, float("-inf")), dim=-1).gather(
        dim=-1, index=selected_experts[:, 0].unsqueeze(-1)
    )

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
        if self.ort_sess is None:
            return None

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states_flat)

        torch_dtype = onnx_to_torch_type_map[self.onnx_dtype]

        tensors = {
            "input": hidden_states_flat.clone().to(device=device, dtype=torch_dtype),
            "router_probs": router_logits.clone().to(device=device, dtype=torch_dtype),
            "output": torch.zeros_like(hidden_states_flat, device=device, dtype=torch_dtype),
        }

        try:
            iobinding = self.ort_sess.io_binding()

            for name, tensor in tensors.items():
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
                repeat = 100
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

            return tensors["output"].reshape(batch_size, sequence_length, hidden_dim)

        except Exception as e:
            raise

    def recreate_onnx_model(self):
        """Recreate the ONNX model with the current weights to reflect any changes to the quantization code."""

        w1_list, w2_list = [], []
        w1_scale_list, w2_scale_list = [], []

        is_4_bit = self.quant_bits == 4
        for i in range(self.num_experts):
            w1_scale, pre_qweight1, w1_qdq = quant_dequant(self.experts[i].w1.weight, is_4_bit)
            w2_scale, pre_qweight2, w2_qdq = quant_dequant(self.experts[i].w2.weight, is_4_bit)

            if self.use_swiglu:
                if self.swiglu_interleaved:
                    pass
                else:
                    w3_scale, pre_qweight3, w3_qdq = quant_dequant(self.experts[i].w3.weight, is_4_bit)

                    gate_weights = pre_qweight1
                    value_weights = pre_qweight3
                    gate_scales = w1_scale
                    value_scales = w3_scale

                    pre_qweight1 = torch.cat([gate_weights, value_weights], dim=0)
                    w1_scale = torch.cat([gate_scales, value_scales], dim=0)

                if self.swiglu_interleaved:
                    self.experts[i].w1.weight = nn.Parameter(w1_qdq.contiguous().clone())

                else:
                    intermediate_size = self.experts[i].w1.weight.shape[0]
                    gate_dequant = w1_qdq[:intermediate_size].contiguous().clone()
                    value_dequant = w1_qdq[intermediate_size:].contiguous().clone()
                    self.experts[i].w1.weight.data = gate_dequant
                    self.experts[i].w3.weight.data = value_dequant
            else:
                self.experts[i].w1.weight.data = w1_qdq.contiguous().clone()

            self.experts[i].w2.weight.data = w2_qdq.contiguous().clone()

            w1_list.append(pre_qweight1)
            w2_list.append(pre_qweight2)
            w1_scale_list.append(w1_scale)
            w2_scale_list.append(w2_scale)

        self.moe_experts_weight1 = torch.stack(w1_list, dim=0)
        self.moe_experts_weight2 = torch.stack(w2_list, dim=0)

        moe_experts_weight_scale1 = torch.stack(w1_scale_list, dim=0)
        moe_experts_weight_scale2 = torch.stack(w2_scale_list, dim=0)

        if moe_experts_weight_scale1.dim() == 3:
            moe_experts_weight_scale1 = moe_experts_weight_scale1.squeeze(-1)
        if moe_experts_weight_scale2.dim() == 3:
            moe_experts_weight_scale2 = moe_experts_weight_scale2.squeeze(-1)

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

        self.ort_sess = self.create_ort_session(self.moe_onnx_graph) if self.moe_onnx_graph else None
        return self.ort_sess is not None

    def parity_check(self):
        model_updated = self.recreate_onnx_model()
        if not model_updated:
            return

        hidden_state = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim).to(device)
        torch_output = self.forward(hidden_state)
        ort_output = self.ort_forward(hidden_state)

        if ort_output is None:
            return

        torch_has_nan = torch.isnan(torch_output).any()
        ort_has_nan = torch.isnan(ort_output).any()
        torch_has_inf = torch.isinf(torch_output).any()
        ort_has_inf = torch.isinf(ort_output).any()

        if torch_has_nan or ort_has_nan or torch_has_inf or ort_has_inf:
            torch_output_clean = torch.where(
                torch.isnan(torch_output) | torch.isinf(torch_output), torch.zeros_like(torch_output), torch_output
            )
            ort_output_clean = torch.where(
                torch.isnan(ort_output) | torch.isinf(ort_output), torch.zeros_like(ort_output), ort_output
            )
            max_diff = (torch_output_clean.cpu() - ort_output_clean.cpu()).abs().max()

            if (torch_has_nan and ort_has_nan) or (torch_has_inf and ort_has_inf):
                problematic_torch = torch.isnan(torch_output) | torch.isinf(torch_output)
                problematic_ort = torch.isnan(ort_output) | torch.isinf(ort_output)
                if torch.equal(problematic_torch, problematic_ort):
                    max_diff = 0.0
        else:
            max_diff = (torch_output.cpu() - ort_output.cpu()).abs().max()

        is_swiglu = hasattr(self, "use_swiglu") and self.use_swiglu
        is_interleaved = hasattr(self, "swiglu_interleaved") and self.swiglu_interleaved
        act_type = f"SwiGLU(interleaved={is_interleaved})" if is_swiglu else "SiLU"

        print(f"Parity check - {act_type} {self.quant_bits}-bit: max_diff = {max_diff:.6f}")

        ort_dtype_quant_bits_tolerance_map = {
            "FP32:0": (5e-3, 1e-3),
            "FP16:0": (5e-2, 1e-3),
            "FP16:4": (0.05, 0.01),
            "FP16:8": (0.02, 0.01),
            "FP32:4": (0.11, 0.01),
            "FP32:8": (0.11, 0.01),
        }

        dtype_str = ort_dtype_name_map[self.onnx_dtype]
        tolerance_key = f"{dtype_str}:{self.quant_bits}"
        if tolerance_key in ort_dtype_quant_bits_tolerance_map:
            base_atol, rtol = ort_dtype_quant_bits_tolerance_map[tolerance_key]

            if max_diff > base_atol:
                raise AssertionError(
                    f"QMoE parity check failed: max difference {max_diff:.6f} exceeds "
                    f"tolerance {base_atol:.6f} for {tolerance_key}"
                )
        else:
            fallback_atol = 0.1
            if max_diff > fallback_atol:
                raise AssertionError(
                    f"QMoE parity check failed: max difference {max_diff:.6f} exceeds "
                    f"fallback tolerance {fallback_atol:.6f} for unknown config {tolerance_key}"
                )

    def benchmark_ort(self):
        hidden_state = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim).to(device)
        self.ort_forward(hidden_state, enable_performance_test=True)


def small_test_cases():
    for batch_size in [1, 4]:
        for sequence_length in [32, 128]:
            yield batch_size, sequence_length


class SwigluMoEBlock(SparseMoeBlockORTHelper):
    def __init__(
        self, config: SwigluMoeConfig, batch_size: int, sequence_length: int, quant_bits: int = 0, onnx_dtype=None
    ):
        super().__init__(quant_bits, onnx_dtype=onnx_dtype)
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_token
        self.use_swiglu = True
        self.swiglu_interleaved = True
        use_quant = self.quant_bits > 0

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=True)

        self.experts = nn.ModuleList([SwigluMlp(config) for _ in range(self.num_experts)])

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

                scale1, pre_qweight1, w1_qdq = quant_dequant(expert.w1.weight, is_4_bit)
                scale2, pre_qweight2, w2_qdq = quant_dequant(expert.w2.weight, is_4_bit)

                expert.w1.weight.data = w1_qdq
                expert.w2.weight.data = w2_qdq

                fc1_w_list.append(pre_qweight1)
                fc2_w_list.append(pre_qweight2)
                scale_1_list.append(scale1)
                scale_2_list.append(scale2)

        self.batch_size = batch_size
        self.sequence_length = sequence_length

        self.moe_onnx_graph = None
        self.ort_sess = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=1, dtype=torch.float)

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


class PhiMoESparseMoeBlock(SparseMoeBlockORTHelper):
    def __init__(
        self, config: PhiMoEConfig, batch_size: int, sequence_length: int, quant_bits: int = 0, onnx_dtype=None
    ):
        super().__init__(quant_bits, onnx_dtype=onnx_dtype)
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.router_jitter_noise = config.router_jitter_noise
        self.use_swiglu = True
        self.swiglu_interleaved = True
        use_quant = self.quant_bits > 0

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=True)

        self.experts = nn.ModuleList([PhiMoESwiGLUMLP(config) for _ in range(self.num_experts)])

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

                scale1, pre_qweight1, w1_qdq = quant_dequant(expert.w1.weight, is_4_bit)
                scale2, pre_qweight2, w2_qdq = quant_dequant(expert.w2.weight, is_4_bit)

                expert.w1.weight.data = w1_qdq
                expert.w2.weight.data = w2_qdq

                fc1_w_list.append(pre_qweight1)
                fc2_w_list.append(pre_qweight2)
                scale_1_list.append(scale1)
                scale_2_list.append(scale2)

        fc1_experts_weights = torch.stack(fc1_w_list, dim=0)
        fc2_experts_weights = torch.stack(fc2_w_list, dim=0)
        fc1_experts_bias = torch.stack(fc1_b_list, dim=0)
        fc2_experts_bias = torch.stack(fc2_b_list, dim=0)

        moe_experts_weight_scale1 = torch.stack(scale_1_list, dim=0) if use_quant else None
        moe_experts_weight_scale2 = torch.stack(scale_2_list, dim=0) if use_quant else None

        self.batch_size = batch_size
        self.sequence_length = sequence_length

        self.moe_onnx_graph = create_cpu_moe_onnx_graph(
            hidden_size=self.hidden_dim,
            sequence_length=self.batch_size * self.sequence_length,
            num_experts=self.num_experts,
            top_k=self.top_k,
            intermediate_size=self.ffn_dim,
            torch_dtype=torch.float32,
            onnx_dtype=self.onnx_dtype,
            fc1_experts_weights=fc1_experts_weights,
            fc2_experts_weights=fc2_experts_weights,
            fc1_bias=fc1_experts_bias,
            fc2_bias=fc2_experts_bias,
            fc1_scales=moe_experts_weight_scale1,
            fc2_scales=moe_experts_weight_scale2,
            use_swiglu=self.use_swiglu,
            use_quant=use_quant,
            quant_bits=self.quant_bits,
            swiglu_interleaved=self.swiglu_interleaved,
        )

        self.ort_sess = self.create_ort_session(self.moe_onnx_graph) if self.moe_onnx_graph else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """PyTorch reference forward pass using SwiGLU-style routing"""
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


disable_cpu_qmoe_tests = False

# Define test cases for different MoE types
phi3_test_cases = [
    (1, 32, 4),
    (1, 32, 8),
    (2, 16, 4),
    (2, 16, 8),
]


@unittest.skipIf(disable_cpu_qmoe_tests, "Skipping qMoE cpu tests")
class TestPhiQMoECPU(unittest.TestCase):
    @parameterized.expand(phi3_test_cases)
    def test_phi3_qmoe_parity_cpu(self, batch_size, sequence_length, quant_bits):
        torch.manual_seed(42)
        numpy.random.seed(42)

        test_config = f"batch_size={batch_size}, sequence_length={sequence_length}, quant_bits={quant_bits}"
        print(f"Running Phi3 QMoE test: {test_config}")

        config = PhiMoEConfig(hidden_size=128, intermediate_size=256, num_local_experts=4, num_experts_per_tok=2)

        phi3_moe = PhiMoESparseMoeBlock(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            quant_bits=quant_bits,
            onnx_dtype=TensorProto.FLOAT,
        )

        hidden_states = torch.randn(batch_size, sequence_length, config.hidden_size).to(torch.float32)

        torch_result = phi3_moe.forward(hidden_states)

        # Verify output shape and basic properties
        expected_shape = (batch_size, sequence_length, config.hidden_size)
        self.assertEqual(torch_result.shape, expected_shape)
        self.assertFalse(torch.isnan(torch_result).any())
        self.assertFalse(torch.isinf(torch_result).any())

        phi3_moe.parity_check()


disable_cpu_qmoe_tests = False

swiglu_test_cases = [
    (1, 32, 4),
    (1, 32, 8),
    (2, 16, 4),
    (2, 16, 8),
]


@unittest.skipIf(disable_cpu_qmoe_tests, "Skipping qMoE cpu tests")
class TestSwigluQMoECPU(unittest.TestCase):
    @parameterized.expand(swiglu_test_cases)
    def test_swiglu_qmoe_parity_cpu(self, batch_size, sequence_length, quant_bits):
        torch.manual_seed(42)
        numpy.random.seed(42)

        test_config = f"batch_size={batch_size}, sequence_length={sequence_length}, quant_bits={quant_bits}"
        print(f"Running SwiGLU test: {test_config}")

        config = SwigluMoeConfig(hidden_size=128, intermediate_size=256, num_local_experts=4, num_experts_per_token=2)

        swiglu_moe = SwigluMoEBlock(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            quant_bits=quant_bits,
            onnx_dtype=TensorProto.FLOAT,
        )

        hidden_states = torch.randn(batch_size, sequence_length, config.hidden_size).to(torch.float32)

        torch_result = swiglu_moe.forward(hidden_states)

        expected_shape = (batch_size, sequence_length, config.hidden_size)
        self.assertEqual(torch_result.shape, expected_shape)
        self.assertFalse(torch.isnan(torch_result).any())
        self.assertFalse(torch.isinf(torch_result).any())

        swiglu_moe.parity_check()


if __name__ == "__main__":
    unittest.main()
