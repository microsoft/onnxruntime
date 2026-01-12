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


def quant_dequant(weights, is_4_bit_quantization: bool = True, asymmetric: bool = False):
    """
    Quantize and dequantize weights for testing purposes.
    Supports symmetric (default) and asymmetric quantization.

    Returns:
        scale, quantized_storage, dequantized, zero_point_storage
    """
    rows, cols = weights.shape

    # Handle edge case of all-zero weights tensor
    if torch.all(weights == 0):
        scale = torch.zeros((rows), dtype=torch.float32, device=weights.device)
        if is_4_bit_quantization:
            packed_size = (cols + 1) // 2
            quantized_storage = torch.zeros((rows, packed_size), dtype=torch.uint8, device=weights.device)
            zp_packed_size = (rows + 1) // 2
            zero_point_storage = (
                torch.zeros(zp_packed_size, dtype=torch.uint8, device=weights.device) if asymmetric else None
            )
        else:
            quantized_storage = torch.zeros_like(weights, dtype=torch.uint8)
            zero_point_storage = torch.zeros((rows), dtype=torch.uint8, device=weights.device) if asymmetric else None

        return scale, quantized_storage, torch.zeros_like(weights), zero_point_storage

    if is_4_bit_quantization:
        qmin_val, qmax_val = (0, 15) if asymmetric else (-8, 7)
        storage_qmin, storage_qmax = (0, 15)
    else:
        qmin_val, qmax_val = (0, 255) if asymmetric else (-128, 127)
        storage_qmin, storage_qmax = (0, 255)

    # Calculate scale and zero point
    if asymmetric:
        min_val = weights.min(dim=-1, keepdim=True)[0]
        max_val = weights.max(dim=-1, keepdim=True)[0]
        scale = (max_val - min_val) / (qmax_val - qmin_val)
        scale = torch.clamp(scale, min=1e-8)
        zero_point_float = qmin_val - min_val.double() / scale.double()
        zero_point_int = (
            torch.round(zero_point_float).clamp(storage_qmin, storage_qmax).to(torch.int32)
        )  # Shape [rows, 1]
    else:
        abs_max = weights.abs().max(dim=-1, keepdim=True)[0]
        scale = abs_max / qmax_val
        scale = torch.clamp(scale, min=1e-8)
        # Symmetric zero point (storage offset)
        zero_point_int = torch.full_like(
            scale, (storage_qmax + 1) // 2, dtype=torch.int32
        )  # 8 for 4-bit, 128 for 8-bit

    # Quantize
    quantized_float = weights.double() / scale.double()
    if asymmetric:
        quantized_float += zero_point_float.double()
    else:
        quantized_float += zero_point_int.double()  # Add storage offset for symmetric

    clamped_quantized = torch.round(quantized_float).clamp(storage_qmin, storage_qmax).to(torch.uint8)  # [rows, cols]

    # Dequantize for verification
    if asymmetric:
        dequantized = (clamped_quantized.float() - zero_point_int.float()) * scale.float()
    else:
        # Symmetric: convert storage [0, 15] back to [-8, 7]
        signed_vals = clamped_quantized.float() - zero_point_int.float()
        dequantized = signed_vals * scale.float()

    # Pack quantized weights and zero points
    if is_4_bit_quantization:
        # Pack weights
        packed_size = (cols + 1) // 2
        quantized_storage = torch.zeros((rows, packed_size), dtype=torch.uint8, device=weights.device)
        for i in range(0, cols, 2):
            val1 = clamped_quantized[..., i]
            val2 = clamped_quantized[..., i + 1] if i + 1 < cols else torch.zeros_like(val1)
            quantized_storage[..., i // 2] = (val1 & 0xF) | ((val2 & 0xF) << 4)

        # Pack zero points (if asymmetric)
        if asymmetric:
            zp_vals = zero_point_int.squeeze(-1).to(torch.uint8)  # Shape [rows]
            zp_packed_size = (rows + 1) // 2
            zero_point_storage = torch.zeros(zp_packed_size, dtype=torch.uint8, device=weights.device)
            for i in range(0, rows, 2):
                val1 = zp_vals[i] & 0x0F
                val2 = (zp_vals[i + 1] & 0x0F) << 4 if i + 1 < rows else 0
                zero_point_storage[i // 2] = val1 | val2
        else:
            zero_point_storage = None  # Symmetric, no ZP tensor

    else:  # 8-bit
        quantized_storage = clamped_quantized
        if asymmetric:
            zero_point_storage = zero_point_int.squeeze(-1).to(torch.uint8)  # Shape [rows]
        else:
            zero_point_storage = None  # Symmetric, no ZP tensor

    return scale.squeeze(-1).to(torch.float32), quantized_storage, dequantized, zero_point_storage


def quant_dequant_blockwise(weights, block_size, is_4_bit_quantization: bool = True, asymmetric: bool = False):
    """
    Block-wise quantization and dequantization for testing purposes.
    Supports symmetric (default) and asymmetric quantization.

    Args:
        weights: Input tensor of shape [rows, cols]
        block_size: Size of each quantization block
        is_4_bit_quantization: Whether to use 4-bit (True) or 8-bit (False) quantization
        asymmetric: Whether to use asymmetric (True) or symmetric (False) quantization

    Returns:
        scales: Scale tensor of shape [rows, num_blocks]
        quantized: Quantized tensor
        dequantized: Dequantized tensor for verification
        zero_points_storage: Packed zero-point tensor
    """
    rows, cols = weights.shape
    num_blocks = (cols + block_size - 1) // block_size

    # Handle edge case of all-zero weights tensor
    if torch.all(weights == 0):
        scales = torch.zeros((rows, num_blocks), dtype=torch.float32, device=weights.device)
        dequantized = torch.zeros_like(weights)
        if is_4_bit_quantization:
            packed_size = (cols + 1) // 2
            quantized = torch.zeros((rows, packed_size), dtype=torch.uint8, device=weights.device)
            zp_packed_size = (num_blocks + 1) // 2
            zero_points_storage = (
                torch.zeros((rows, zp_packed_size), dtype=torch.uint8, device=weights.device) if asymmetric else None
            )
        else:
            quantized = torch.zeros((rows, cols), dtype=torch.uint8, device=weights.device)
            zero_points_storage = (
                torch.zeros((rows, num_blocks), dtype=torch.uint8, device=weights.device) if asymmetric else None
            )
        return scales, quantized, dequantized, zero_points_storage

    # Initialize output tensors
    scales = torch.zeros((rows, num_blocks), dtype=torch.float32, device=weights.device)
    zero_points_tensor = (
        torch.zeros((rows, num_blocks), dtype=torch.int32, device=weights.device) if asymmetric else None
    )
    dequantized = torch.zeros_like(weights)

    # Quantization ranges
    if is_4_bit_quantization:
        qmin_val, qmax_val = (0, 15) if asymmetric else (-8, 7)
        storage_qmin, storage_qmax = (0, 15)
        sym_zp_offset = 8
        packed_size = (cols + 1) // 2
        quantized = torch.zeros((rows, packed_size), dtype=torch.uint8, device=weights.device)
    else:
        qmin_val, qmax_val = (0, 255) if asymmetric else (-128, 127)
        storage_qmin, storage_qmax = (0, 255)
        sym_zp_offset = 128
        quantized = torch.zeros((rows, cols), dtype=torch.uint8, device=weights.device)

    # Process each block
    for row in range(rows):
        for block_idx in range(num_blocks):
            start_col = block_idx * block_size
            end_col = min(start_col + block_size, cols)
            block_data = weights[row, start_col:end_col]

            if torch.all(block_data == 0):
                scales[row, block_idx] = 1e-8
                if asymmetric:
                    zero_points_tensor[row, block_idx] = storage_qmin
                # Quantized block will be 0 (for asymmetric) or sym_zp_offset (for symmetric)
                # Dequantized block will be 0
                quantized_block_uint8 = torch.full_like(
                    block_data, storage_qmin if asymmetric else sym_zp_offset, dtype=torch.uint8
                )
                dequantized[row, start_col:end_col] = 0.0

            else:
                if asymmetric:
                    min_val = block_data.min()
                    max_val = block_data.max()
                    scale = (max_val - min_val) / (qmax_val - qmin_val)
                    scale = torch.clamp(scale, min=1e-8)
                    zero_point_float = qmin_val - min_val.double() / scale.double()
                    zero_point_int = torch.round(zero_point_float).clamp(storage_qmin, storage_qmax).to(torch.int32)

                    scales[row, block_idx] = scale.to(torch.float32)
                    zero_points_tensor[row, block_idx] = zero_point_int

                    quantized_block_float = (block_data.double() / scale.double()) + zero_point_float.double()
                    quantized_block_uint8 = (
                        torch.round(quantized_block_float).clamp(storage_qmin, storage_qmax).to(torch.uint8)
                    )

                    dequantized[row, start_col:end_col] = (
                        quantized_block_uint8.float() - zero_point_int.float()
                    ) * scale.float()

                else:  # Symmetric
                    abs_max = block_data.abs().max()
                    scale = abs_max / qmax_val
                    scale = torch.clamp(scale, min=1e-8)
                    scales[row, block_idx] = scale.to(torch.float32)

                    quantized_block_signed = (
                        torch.round(block_data.double() / scale.double()).clamp(qmin_val, qmax_val).to(torch.int32)
                    )
                    quantized_block_uint8 = (quantized_block_signed + sym_zp_offset).to(torch.uint8)

                    dequantized[row, start_col:end_col] = quantized_block_signed.float() * scale.float()

            # Pack quantized data
            if is_4_bit_quantization:
                for i in range(0, end_col - start_col, 2):
                    col_idx = start_col + i
                    packed_idx = col_idx // 2

                    val1 = int(quantized_block_uint8[i])
                    val2 = (
                        int(quantized_block_uint8[i + 1])
                        if i + 1 < len(quantized_block_uint8)
                        else storage_qmin
                        if asymmetric
                        else sym_zp_offset
                    )

                    packed_val = (val1 & 0xF) | ((val2 & 0xF) << 4)
                    quantized[row, packed_idx] = packed_val
            else:
                quantized[row, start_col:end_col] = quantized_block_uint8

    # Pack zero points
    zero_points_storage = None
    if asymmetric:
        if is_4_bit_quantization:
            zp_packed_size = (num_blocks + 1) // 2
            zero_points_storage = torch.zeros((rows, zp_packed_size), dtype=torch.uint8, device=weights.device)
            zp_vals_uint8 = zero_points_tensor.to(torch.uint8)
            for j in range(0, num_blocks, 2):
                val1 = zp_vals_uint8[:, j] & 0x0F
                val2 = (zp_vals_uint8[:, j + 1] & 0x0F) << 4 if j + 1 < num_blocks else 0
                zero_points_storage[:, j // 2] = val1 | val2
        else:  # 8-bit
            zero_points_storage = zero_points_tensor.to(torch.uint8)

    return scales, quantized, dequantized, zero_points_storage


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
    fc1_zero_points=None,
    fc2_zero_points=None,
    use_swiglu=False,
    use_quant=False,
    quant_bits=4,
    swiglu_interleaved=False,
    block_size=0,
):
    if not has_onnx:
        return None

    inter_size = intermediate_size
    topk = top_k

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
    # Accept float16 or float32 scales; tests may produce float32 for better precision
    assert fc1_scales.dtype in (torch.float16, torch.float32), "FC1 scales must be float16 or float32 for QMoE"
    assert fc2_scales.dtype in (torch.float16, torch.float32), "FC2 scales must be float16 or float32 for QMoE"

    if not has_onnx:
        return None

    # Set operator name and inputs based on quantization mode
    if use_quant:
        op_name = "QMoE"
        # Match the 14-input schema
        inputs = [
            "input",  # 0
            "router_probs",  # 1
            "fc1_experts_weights",  # 2
            "fc1_scales",  # 3
            "",  # 4: fc1_bias
            "fc2_experts_weights",  # 5
            "fc2_scales",  # 6
            "",  # 7: fc2_bias
            "",  # 8: fc3_weights
            "",  # 9: fc3_scales
            "",  # 10: fc3_bias
            "fc1_zero_points" if fc1_zero_points is not None else "",  # 11
            "fc2_zero_points" if fc2_zero_points is not None else "",  # 12
            "",  # 13: fc3_zero_points
        ]
    else:
        # For regular (non-quantized) MoE, use different operator and input layout
        op_name = "MoE"  # Regular MoE operator
        inputs = [
            "input",
            "router_probs",
            "fc1_experts_weights",
            "fc1_experts_bias" if fc1_bias is not None else "",  # fc1_bias as input 3
            "fc2_experts_weights",
            "fc2_experts_bias" if fc2_bias is not None else "",  # fc2_bias as input 5
            "",  # fc3_experts_weights (not used)
            "",  # fc3_experts_bias (not used)
        ]

    activation = "swiglu" if use_swiglu else "silu"

    # Set normalization behavior based on operator type:
    # - QMoE: Raw logits passed, needs normalization in C++ kernel
    # - Regular MoE: Pre-computed probabilities passed, no additional normalization needed
    normalize_routing = 1 if use_quant else 0

    nodes = [
        helper.make_node(
            op_name,
            inputs,
            ["output"],
            "MoE_0",
            k=topk,
            normalize_routing_weights=normalize_routing,
            activation_type=activation,
            # Add new attributes with backwards-compatible default values
            swiglu_fusion=1 if use_swiglu else 0,  # 1 if using SwiGLU activation
            swiglu_limit=7.0,
            activation_alpha=1.702,
            activation_beta=1.0,
            swiglu_interleaved=1 if swiglu_interleaved else 0,  # Enable this attribute
            domain="com.microsoft",
        ),
    ]

    if use_quant:
        nodes[0].attribute.extend([helper.make_attribute("expert_weight_bits", quant_bits)])

    # Add block_size attribute for block-wise quantization
    if block_size > 0:
        nodes[0].attribute.extend([helper.make_attribute("block_size", block_size)])

    # Weights are store in column major order. Need pack 2 int4 values into uint8.
    # Use the actual tensor shapes instead of calculating them to avoid size mismatches
    fc1_shape = list(fc1_experts_weights.shape)
    fc2_shape = list(fc2_experts_weights.shape)

    torch_dtype = onnx_to_torch_type_map[onnx_dtype]

    weight_numpy_type = numpy.uint8 if use_quant else ort_to_numpy_type_map[onnx_dtype]
    weight_onnx_type = TensorProto.UINT8 if use_quant else onnx_dtype

    # Use raw bytes from C-contiguous numpy arrays to ensure the exact memory layout
    # of the packed uint8 weight tensors is preserved when writing the ONNX initializer.
    fc1_np = fc1_experts_weights.detach().cpu().numpy().astype(weight_numpy_type)
    fc2_np = fc2_experts_weights.detach().cpu().numpy().astype(weight_numpy_type)
    fc1_np = numpy.ascontiguousarray(fc1_np)
    fc2_np = numpy.ascontiguousarray(fc2_np)

    initializers = [
        helper.make_tensor(
            "fc1_experts_weights",
            weight_onnx_type,
            fc1_shape,
            fc1_np.tobytes(),
            raw=True,
        ),
        helper.make_tensor(
            "fc2_experts_weights",
            weight_onnx_type,
            fc2_shape,
            fc2_np.tobytes(),
            raw=True,
        ),
    ]

    # Calculate scale tensor shapes based on block_size
    if block_size > 0:
        # Block-wise quantization: 3D scale tensors
        fc1_blocks_per_row = (hidden_size + block_size - 1) // block_size
        fc2_blocks_per_row = (inter_size + block_size - 1) // block_size

        fc1_scale_shape = [num_experts, 2 * inter_size if use_swiglu else inter_size, fc1_blocks_per_row]
        fc2_scale_shape = [num_experts, hidden_size, fc2_blocks_per_row]
    else:
        # Row-wise quantization: 2D scale tensors
        fc1_scale_shape = [num_experts, 2 * inter_size if use_swiglu else inter_size]
        fc2_scale_shape = [num_experts, hidden_size]

    # Handle scale tensors
    fc1_scale_tensor = fc1_scales.to(torch_dtype).flatten().detach().cpu().numpy()
    fc2_scale_tensor = fc2_scales.to(torch_dtype).flatten().detach().cpu().numpy()

    # Process scale tensors for proper data format
    fc1_scale_data = fc1_scale_tensor.tolist()
    fc2_scale_data = fc2_scale_tensor.tolist()

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

    # Add zero-point initializers if provided
    if fc1_zero_points is not None:
        fc1_zp_np = fc1_zero_points.detach().cpu().numpy().astype(numpy.uint8)
        fc1_zp_np = numpy.ascontiguousarray(fc1_zp_np)
        initializers.append(
            helper.make_tensor(
                "fc1_zero_points",
                TensorProto.UINT8,
                list(fc1_zero_points.shape),
                fc1_zp_np.tobytes(),
                raw=True,
            )
        )

    if fc2_zero_points is not None:
        fc2_zp_np = fc2_zero_points.detach().cpu().numpy().astype(numpy.uint8)
        fc2_zp_np = numpy.ascontiguousarray(fc2_zp_np)
        initializers.append(
            helper.make_tensor(
                "fc2_zero_points",
                TensorProto.UINT8,
                list(fc2_zero_points.shape),
                fc2_zp_np.tobytes(),
                raw=True,
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
    def __init__(self, quant_bits=0, onnx_dtype=None, use_asymmetric_quant: bool = False):
        super().__init__()
        self.quant_bits = quant_bits
        self.onnx_dtype = onnx_dtype
        self.np_type = numpy.float16 if self.onnx_dtype == TensorProto.FLOAT16 else numpy.float32
        self.use_asymmetric_quant = use_asymmetric_quant

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
            print(f"ERROR: ORT session is None for {self.__class__.__name__}")
            return None

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states_flat)

        # Different routing logic for QMoE vs regular MoE:
        # - QMoE expects raw logits (does its own softmax internally)
        # - Regular MoE expects pre-computed routing probabilities
        if hasattr(self, "quant_bits") and self.quant_bits > 0:
            # QMoE: Pass raw logits directly (QMoE does softmax internally)
            router_input = router_logits
            # print("DEBUG: Using QMoE routing (raw logits)")
        else:
            # Regular MoE: Apply the same routing logic as PyTorch reference
            # This converts raw logits to proper routing probabilities
            routing_weights, selected_experts = masked_sampling_omp_inference(
                router_logits,
                top_k=self.top_k,
                jitter_eps=self.router_jitter_noise,
                training=False,
            )

            # IMPORTANT: The routing weights from masked_sampling_omp_inference sum to top_k,
            # but ONNX Runtime expects normalized probabilities that sum to 1.0
            # Normalize the routing weights per token
            routing_weights = routing_weights / routing_weights.sum(dim=1, keepdim=True)

            # Create proper router probabilities tensor that matches PyTorch routing
            router_input = torch.zeros_like(router_logits)
            for i in range(router_logits.shape[0]):  # For each token
                for j in range(self.top_k):  # For each top-k expert
                    expert_idx = selected_experts[i, j]
                    router_input[i, expert_idx] = routing_weights[i, j]

        #     print("DEBUG: Using regular MoE routing (processed probabilities)")

        # print(f"DEBUG: router_input stats: mean={router_input.mean():.6f}, std={router_input.std():.6f}")
        # print(
        #     f"DEBUG: hidden_states_flat stats: mean={hidden_states_flat.mean():.6f}, std={hidden_states_flat.std():.6f}"
        # )

        torch_dtype = onnx_to_torch_type_map[self.onnx_dtype]

        tensors = {
            "input": hidden_states_flat.clone().to(device=device, dtype=torch_dtype),
            "router_probs": router_logits.clone().to(device=device, dtype=torch_dtype),
            "output": torch.zeros((batch_size * sequence_length, hidden_dim), device=device, dtype=torch_dtype),
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

            # print("DEBUG: About to run ORT inference...")

            iobinding.synchronize_inputs()
            self.ort_sess.run_with_iobinding(iobinding)
            iobinding.synchronize_outputs()

            # print("DEBUG: ORT inference completed successfully")

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
        w1_zp_list, w2_zp_list = [], []

        is_4_bit = self.quant_bits == 4
        for i in range(self.num_experts):
            if self.block_size > 0:
                # Use block-wise quantization
                w1_scale, pre_qweight1, w1_qdq, w1_zp = quant_dequant_blockwise(
                    self.experts[i].w1.weight, self.block_size, is_4_bit, asymmetric=self.use_asymmetric_quant
                )
                w2_scale, pre_qweight2, w2_qdq, w2_zp = quant_dequant_blockwise(
                    self.experts[i].w2.weight, self.block_size, is_4_bit, asymmetric=self.use_asymmetric_quant
                )
            else:
                # Use row-wise quantization
                w1_scale, pre_qweight1, w1_qdq, w1_zp = quant_dequant(
                    self.experts[i].w1.weight, is_4_bit, asymmetric=self.use_asymmetric_quant
                )
                w2_scale, pre_qweight2, w2_qdq, w2_zp = quant_dequant(
                    self.experts[i].w2.weight, is_4_bit, asymmetric=self.use_asymmetric_quant
                )

            if self.use_swiglu:
                if self.swiglu_interleaved:
                    pass
                else:
                    if self.block_size > 0:
                        w3_scale, pre_qweight3, w3_qdq, w3_zp = quant_dequant_blockwise(
                            self.experts[i].w3.weight, self.block_size, is_4_bit, asymmetric=self.use_asymmetric_quant
                        )
                    else:
                        w3_scale, pre_qweight3, w3_qdq, w3_zp = quant_dequant(
                            self.experts[i].w3.weight, is_4_bit, asymmetric=self.use_asymmetric_quant
                        )

                    gate_weights = pre_qweight1
                    value_weights = pre_qweight3
                    gate_scales = w1_scale
                    value_scales = w3_scale
                    gate_zp = w1_zp
                    value_zp = w3_zp

                    pre_qweight1 = torch.cat([gate_weights, value_weights], dim=0)
                    w1_scale = torch.cat([gate_scales, value_scales], dim=0)
                    if w1_zp is not None and w3_zp is not None:
                        w1_zp = torch.cat([gate_zp, value_zp], dim=0)

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
            if w1_zp is not None:
                w1_zp_list.append(w1_zp)
            if w2_zp is not None:
                w2_zp_list.append(w2_zp)

        self.moe_experts_weight1 = torch.stack(w1_list, dim=0)
        self.moe_experts_weight2 = torch.stack(w2_list, dim=0)

        moe_experts_weight_scale1 = torch.stack(w1_scale_list, dim=0)
        moe_experts_weight_scale2 = torch.stack(w2_scale_list, dim=0)

        moe_experts_zp1 = torch.stack(w1_zp_list, dim=0) if len(w1_zp_list) > 0 else None
        moe_experts_zp2 = torch.stack(w2_zp_list, dim=0) if len(w2_zp_list) > 0 else None

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
                # Zero points
                fc1_zero_points=moe_experts_zp1,
                fc2_zero_points=moe_experts_zp2,
                use_swiglu=self.use_swiglu,
                use_quant=True,  # Always use QMoE
                quant_bits=self.quant_bits,
                swiglu_interleaved=self.swiglu_interleaved if hasattr(self, "swiglu_interleaved") else False,
                block_size=self.block_size,  # Add block_size for block-wise quantization
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
        quant_type = "Asymmetric" if self.use_asymmetric_quant else "Symmetric"
        block_type = f"Block({self.block_size})" if self.block_size > 0 else "Row"

        print(f"Parity check - {act_type} {self.quant_bits}-bit {quant_type} {block_type}: max_diff = {max_diff:.6f}")

        # Diagnostic dump: when differences are large, show the index and nearby values
        if max_diff > 1e-3:
            diff = (torch_output.cpu() - ort_output.cpu()).abs()
            idx = torch.argmax(diff)
            flat_idx = int(idx)
            # Derive coordinates (batch, seq, hidden) from flattened index
            total_elems = torch_output.numel()
            # Work in flattened [batch, seq, hidden] ordering
            hidden_dim = self.hidden_dim
            seq = self.sequence_length
            # Clamp to safe bounds
            flat_idx = min(flat_idx, total_elems - 1)
            i = flat_idx // (hidden_dim)
            j = i // seq
            k = flat_idx % hidden_dim
            print(
                f"Diagnostic - max diff at flat_idx={flat_idx} -> sample (batch_idx={j}, seq_idx={i % seq}, hidden_idx={k})"
            )
            print("Torch sample:", torch_output.cpu().reshape(-1, hidden_dim)[i, k].item())
            print("ORT  sample:", ort_output.cpu().reshape(-1, hidden_dim)[i, k].item())
            # Print routing and per-expert contributions for this token from the PyTorch reference
            try:
                hidden_states_flat = hidden_state.view(-1, hidden_dim)
                token_vec = hidden_states_flat[i : i + 1]
                gate_logits = self.gate(token_vec)
                topk_vals, topk_experts = torch.topk(gate_logits, self.top_k, dim=-1)
                topk_soft = F.softmax(topk_vals, dim=1)
                print("Gate logits:", gate_logits.detach().cpu().numpy())
                print("Selected experts:", topk_experts.detach().cpu().numpy())
                print("Routing weights:", topk_soft.detach().cpu().numpy())
                # Compute per-expert contributions for selected experts
                for idx_e, e in enumerate(topk_experts[0].tolist()):
                    expert_layer = self.experts[e]
                    expert_out = expert_layer(token_vec)
                    contrib = expert_out[0, k].item() * topk_soft[0, idx_e].item()
                    print(f"Expert {e} contrib at hidden {k}: {contrib}")
            except Exception as _:
                pass

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

            # Increase tolerance for asymmetric quantization due to different computation path
            if self.use_asymmetric_quant:
                base_atol *= 1.5

            if max_diff > base_atol:
                raise AssertionError(
                    f"QMoE parity check failed: max difference {max_diff:.6f} exceeds "
                    f"tolerance {base_atol:.6f} for {tolerance_key} ({quant_type})"
                )
        else:
            fallback_atol = 0.1
            if self.use_asymmetric_quant:
                fallback_atol = 0.15

            if max_diff > fallback_atol:
                raise AssertionError(
                    f"QMoE parity check failed: max difference {max_diff:.6f} exceeds "
                    f"fallback tolerance {fallback_atol:.6f} for unknown config {tolerance_key} ({quant_type})"
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
        self,
        config: SwigluMoeConfig,
        batch_size: int,
        sequence_length: int,
        quant_bits: int = 0,
        onnx_dtype=None,
        block_size: int = 0,
        use_asymmetric_quant: bool = False,
    ):
        super().__init__(quant_bits, onnx_dtype=onnx_dtype, use_asymmetric_quant=use_asymmetric_quant)
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_token
        self.use_swiglu = True
        self.swiglu_interleaved = True
        self.block_size = block_size
        use_quant = self.quant_bits > 0

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=True)

        self.experts = nn.ModuleList([SwigluMlp(config) for _ in range(self.num_experts)])

        fc1_w_list, fc2_w_list = [], []
        fc1_b_list, fc2_b_list = [], []
        scale_1_list, scale_2_list = [], []
        zp_1_list, zp_2_list = [], []

        for expert in self.experts:
            fc1_b_list.append(expert.w1.bias)
            fc2_b_list.append(expert.w2.bias)
            if not use_quant:
                fc1_w_list.append(expert.w1.weight)
                fc2_w_list.append(expert.w2.weight)
            else:
                is_4_bit = self.quant_bits == 4

                if self.block_size > 0:
                    scale1, pre_qweight1, w1_qdq, zp1 = quant_dequant_blockwise(
                        expert.w1.weight, self.block_size, is_4_bit, asymmetric=self.use_asymmetric_quant
                    )
                    scale2, pre_qweight2, w2_qdq, zp2 = quant_dequant_blockwise(
                        expert.w2.weight, self.block_size, is_4_bit, asymmetric=self.use_asymmetric_quant
                    )
                else:
                    scale1, pre_qweight1, w1_qdq, zp1 = quant_dequant(
                        expert.w1.weight, is_4_bit, asymmetric=self.use_asymmetric_quant
                    )
                    scale2, pre_qweight2, w2_qdq, zp2 = quant_dequant(
                        expert.w2.weight, is_4_bit, asymmetric=self.use_asymmetric_quant
                    )

                expert.w1.weight.data = w1_qdq
                expert.w2.weight.data = w2_qdq

                fc1_w_list.append(pre_qweight1)
                fc2_w_list.append(pre_qweight2)
                scale_1_list.append(scale1)
                scale_2_list.append(scale2)
                if zp1 is not None:
                    zp_1_list.append(zp1)
                if zp2 is not None:
                    zp_2_list.append(zp2)

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
        self,
        config: PhiMoEConfig,
        batch_size: int,
        sequence_length: int,
        quant_bits: int = 0,
        onnx_dtype=None,
        block_size: int = 0,
        use_asymmetric_quant: bool = False,
    ):
        super().__init__(quant_bits, onnx_dtype=onnx_dtype, use_asymmetric_quant=use_asymmetric_quant)
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.router_jitter_noise = config.router_jitter_noise
        self.use_swiglu = True
        self.swiglu_interleaved = True
        self.block_size = block_size
        use_quant = self.quant_bits > 0

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=True)

        self.experts = nn.ModuleList([PhiMoESwiGLUMLP(config) for _ in range(self.num_experts)])

        fc1_w_list, fc2_w_list = [], []
        fc1_b_list, fc2_b_list = [], []
        scale_1_list, scale_2_list = [], []
        zp_1_list, zp_2_list = [], []

        for expert in self.experts:
            fc1_b_list.append(expert.w1.bias)
            fc2_b_list.append(expert.w2.bias)
            if not use_quant:
                fc1_w_list.append(expert.w1.weight)
                fc2_w_list.append(expert.w2.weight)
            else:
                is_4_bit = self.quant_bits == 4

                if self.block_size > 0:
                    scale1, pre_qweight1, w1_qdq, zp1 = quant_dequant_blockwise(
                        expert.w1.weight, self.block_size, is_4_bit, asymmetric=self.use_asymmetric_quant
                    )
                    scale2, pre_qweight2, w2_qdq, zp2 = quant_dequant_blockwise(
                        expert.w2.weight, self.block_size, is_4_bit, asymmetric=self.use_asymmetric_quant
                    )
                else:
                    scale1, pre_qweight1, w1_qdq, zp1 = quant_dequant(
                        expert.w1.weight, is_4_bit, asymmetric=self.use_asymmetric_quant
                    )
                    scale2, pre_qweight2, w2_qdq, zp2 = quant_dequant(
                        expert.w2.weight, is_4_bit, asymmetric=self.use_asymmetric_quant
                    )

                expert.w1.weight.data = w1_qdq
                expert.w2.weight.data = w2_qdq

                fc1_w_list.append(pre_qweight1)
                fc2_w_list.append(pre_qweight2)
                scale_1_list.append(scale1)
                scale_2_list.append(scale2)
                if zp1 is not None:
                    zp_1_list.append(zp1)
                if zp2 is not None:
                    zp_2_list.append(zp2)

        fc1_experts_weights = torch.stack(fc1_w_list, dim=0)
        fc2_experts_weights = torch.stack(fc2_w_list, dim=0)
        fc1_experts_bias = torch.stack(fc1_b_list, dim=0)
        fc2_experts_bias = torch.stack(fc2_b_list, dim=0)

        moe_experts_weight_scale1 = torch.stack(scale_1_list, dim=0) if use_quant else None
        moe_experts_weight_scale2 = torch.stack(scale_2_list, dim=0) if use_quant else None

        moe_experts_zp1 = torch.stack(zp_1_list, dim=0) if len(zp_1_list) > 0 else None
        moe_experts_zp2 = torch.stack(zp_2_list, dim=0) if len(zp_2_list) > 0 else None

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
            fc1_zero_points=moe_experts_zp1,
            fc2_zero_points=moe_experts_zp2,
            use_swiglu=self.use_swiglu,
            use_quant=use_quant,
            quant_bits=self.quant_bits,
            swiglu_interleaved=self.swiglu_interleaved,
            block_size=self.block_size,
        )

        self.ort_sess = self.create_ort_session(self.moe_onnx_graph) if self.moe_onnx_graph else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """PyTorch reference forward pass using SwiGLU-style routing"""
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        # Match CPU implementation: select top-k experts by logits, then softmax over those logits
        routing_weights_vals, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights_vals, dim=1, dtype=torch.float)
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

# Define test cases for block-wise quantization
phi3_blockwise_test_cases = [
    (1, 32, 4, 32),  # batch_size, sequence_length, quant_bits, block_size
    (1, 32, 8, 64),
    (2, 16, 4, 32),
    (2, 16, 8, 64),
]


@unittest.skipIf(disable_cpu_qmoe_tests, "Skipping qMoE cpu tests")
class TestPhiQMoECPU(unittest.TestCase):
    @parameterized.expand(phi3_test_cases)
    def test_phi3_qmoe_parity_cpu(self, batch_size, sequence_length, quant_bits):
        # Create unique seed based on test parameters to ensure different inputs for each test
        base_seed = 2000  # Different base seed from other tests
        param_hash = hash((batch_size, sequence_length, quant_bits))
        unique_seed = base_seed + abs(param_hash) % 1000

        torch.manual_seed(unique_seed)
        numpy.random.seed(unique_seed)

        test_config = (
            f"batch_size={batch_size}, sequence_length={sequence_length}, quant_bits={quant_bits}, seed={unique_seed}"
        )
        print(f"Running Phi3 QMoE test: {test_config}")

        config = PhiMoEConfig(hidden_size=128, intermediate_size=256, num_local_experts=4, num_experts_per_tok=2)

        phi3_moe = PhiMoESparseMoeBlock(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            quant_bits=quant_bits,
            onnx_dtype=TensorProto.FLOAT,
            use_asymmetric_quant=False,
        )

        hidden_states = torch.randn(batch_size, sequence_length, config.hidden_size).to(torch.float32)

        torch_result = phi3_moe.forward(hidden_states)

        # Verify output shape and basic properties
        expected_shape = (batch_size, sequence_length, config.hidden_size)
        self.assertEqual(torch_result.shape, expected_shape)
        self.assertFalse(torch.isnan(torch_result).any())
        self.assertFalse(torch.isinf(torch_result).any())

        phi3_moe.parity_check()

    @parameterized.expand(phi3_test_cases)
    def test_phi3_qmoe_asymmetric_parity_cpu(self, batch_size, sequence_length, quant_bits):
        base_seed = 3000
        param_hash = hash((batch_size, sequence_length, quant_bits))
        unique_seed = base_seed + abs(param_hash) % 1000
        torch.manual_seed(unique_seed)
        numpy.random.seed(unique_seed)

        test_config = (
            f"batch_size={batch_size}, sequence_length={sequence_length}, quant_bits={quant_bits}, seed={unique_seed}"
        )
        print(f"Running Phi3 QMoE Asymmetric test: {test_config}")

        config = PhiMoEConfig(hidden_size=128, intermediate_size=256, num_local_experts=4, num_experts_per_tok=2)

        phi3_moe = PhiMoESparseMoeBlock(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            quant_bits=quant_bits,
            onnx_dtype=TensorProto.FLOAT,
            use_asymmetric_quant=True,
        )
        phi3_moe.parity_check()

    @parameterized.expand(phi3_blockwise_test_cases)
    def test_phi3_qmoe_blockwise_parity_cpu(self, batch_size, sequence_length, quant_bits, block_size):
        torch.manual_seed(42)
        numpy.random.seed(42)

        test_config = f"batch_size={batch_size}, sequence_length={sequence_length}, quant_bits={quant_bits}, block_size={block_size}"
        print(f"Running Phi3 QMoE block-wise test: {test_config}")

        config = PhiMoEConfig(hidden_size=128, intermediate_size=256, num_local_experts=4, num_experts_per_tok=2)

        phi3_moe = PhiMoESparseMoeBlock(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            quant_bits=quant_bits,
            onnx_dtype=TensorProto.FLOAT,
            block_size=block_size,
            use_asymmetric_quant=False,
        )

        hidden_states = torch.randn(batch_size, sequence_length, config.hidden_size).to(torch.float32)

        torch_result = phi3_moe.forward(hidden_states)

        # Verify output shape and basic properties
        expected_shape = (batch_size, sequence_length, config.hidden_size)
        self.assertEqual(torch_result.shape, expected_shape)
        self.assertFalse(torch.isnan(torch_result).any())
        self.assertFalse(torch.isinf(torch_result).any())

        phi3_moe.parity_check()

    @parameterized.expand(phi3_blockwise_test_cases)
    def test_phi3_qmoe_blockwise_asymmetric_parity_cpu(self, batch_size, sequence_length, quant_bits, block_size):
        torch.manual_seed(43)
        numpy.random.seed(43)

        test_config = f"batch_size={batch_size}, sequence_length={sequence_length}, quant_bits={quant_bits}, block_size={block_size}"
        print(f"Running Phi3 QMoE block-wise Asymmetric test: {test_config}")

        config = PhiMoEConfig(hidden_size=128, intermediate_size=256, num_local_experts=4, num_experts_per_tok=2)

        phi3_moe = PhiMoESparseMoeBlock(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            quant_bits=quant_bits,
            onnx_dtype=TensorProto.FLOAT,
            block_size=block_size,
            use_asymmetric_quant=True,
        )
        phi3_moe.parity_check()


disable_cpu_qmoe_tests = False

swiglu_test_cases = [
    (1, 32, 4),
    (1, 32, 8),
    (2, 16, 4),
    (2, 16, 8),
]

# Define test cases for block-wise quantization
swiglu_blockwise_test_cases = [
    (1, 32, 4, 32),  # batch_size, sequence_length, quant_bits, block_size
    (1, 32, 8, 64),
    (2, 16, 4, 32),
    (2, 16, 8, 64),
]


@unittest.skipIf(disable_cpu_qmoe_tests, "Skipping qMoE cpu tests")
class TestSwigluQMoECPU(unittest.TestCase):
    @parameterized.expand(swiglu_test_cases)
    def test_swiglu_qmoe_parity_cpu(self, batch_size, sequence_length, quant_bits):
        # Create unique seed based on test parameters to ensure different inputs for each test
        base_seed = 1000  # Different base seed from regular MoE tests
        param_hash = hash((batch_size, sequence_length, quant_bits))
        unique_seed = base_seed + abs(param_hash) % 1000

        torch.manual_seed(unique_seed)
        numpy.random.seed(unique_seed)

        test_config = (
            f"batch_size={batch_size}, sequence_length={sequence_length}, quant_bits={quant_bits}, seed={unique_seed}"
        )
        print(f"Running SwiGLU test: {test_config}")

        config = SwigluMoeConfig(hidden_size=128, intermediate_size=256, num_local_experts=4, num_experts_per_token=2)

        swiglu_moe = SwigluMoEBlock(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            quant_bits=quant_bits,
            onnx_dtype=TensorProto.FLOAT,
            use_asymmetric_quant=False,
        )

        hidden_states = torch.randn(batch_size, sequence_length, config.hidden_size).to(torch.float32)

        torch_result = swiglu_moe.forward(hidden_states)

        expected_shape = (batch_size, sequence_length, config.hidden_size)
        self.assertEqual(torch_result.shape, expected_shape)
        self.assertFalse(torch.isnan(torch_result).any())
        self.assertFalse(torch.isinf(torch_result).any())

        swiglu_moe.parity_check()

    @parameterized.expand(swiglu_test_cases)
    def test_swiglu_qmoe_asymmetric_parity_cpu(self, batch_size, sequence_length, quant_bits):
        base_seed = 1100
        param_hash = hash((batch_size, sequence_length, quant_bits))
        unique_seed = base_seed + abs(param_hash) % 1000
        torch.manual_seed(unique_seed)
        numpy.random.seed(unique_seed)

        test_config = (
            f"batch_size={batch_size}, sequence_length={sequence_length}, quant_bits={quant_bits}, seed={unique_seed}"
        )
        print(f"Running SwiGLU Asymmetric test: {test_config}")

        config = SwigluMoeConfig(hidden_size=128, intermediate_size=256, num_local_experts=4, num_experts_per_token=2)

        swiglu_moe = SwigluMoEBlock(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            quant_bits=quant_bits,
            onnx_dtype=TensorProto.FLOAT,
            use_asymmetric_quant=True,
        )
        swiglu_moe.parity_check()

    @parameterized.expand(swiglu_blockwise_test_cases)
    def test_swiglu_qmoe_blockwise_parity_cpu(self, batch_size, sequence_length, quant_bits, block_size):
        torch.manual_seed(42)
        numpy.random.seed(42)

        test_config = f"batch_size={batch_size}, sequence_length={sequence_length}, quant_bits={quant_bits}, block_size={block_size}"
        print(f"Running SwiGLU block-wise test: {test_config}")

        config = SwigluMoeConfig(hidden_size=128, intermediate_size=256, num_local_experts=4, num_experts_per_token=2)

        swiglu_moe = SwigluMoEBlock(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            quant_bits=quant_bits,
            onnx_dtype=TensorProto.FLOAT,
            block_size=block_size,
            use_asymmetric_quant=False,
        )

        hidden_states = torch.randn(batch_size, sequence_length, config.hidden_size).to(torch.float32)

        torch_result = swiglu_moe.forward(hidden_states)

        expected_shape = (batch_size, sequence_length, config.hidden_size)
        self.assertEqual(torch_result.shape, expected_shape)
        self.assertFalse(torch.isnan(torch_result).any())
        self.assertFalse(torch.isinf(torch_result).any())

        swiglu_moe.parity_check()

    @parameterized.expand(swiglu_blockwise_test_cases)
    def test_swiglu_qmoe_blockwise_asymmetric_parity_cpu(self, batch_size, sequence_length, quant_bits, block_size):
        torch.manual_seed(43)
        numpy.random.seed(43)

        test_config = f"batch_size={batch_size}, sequence_length={sequence_length}, quant_bits={quant_bits}, block_size={block_size}"
        print(f"Running SwiGLU block-wise Asymmetric test: {test_config}")

        config = SwigluMoeConfig(hidden_size=128, intermediate_size=256, num_local_experts=4, num_experts_per_token=2)

        swiglu_moe = SwigluMoEBlock(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            quant_bits=quant_bits,
            onnx_dtype=TensorProto.FLOAT,
            block_size=block_size,
            use_asymmetric_quant=True,
        )
        swiglu_moe.parity_check()


@unittest.skipIf(True, "Skipping QMoE CPU benchmark tests")
class TestQMoESwiGLUBenchmark(unittest.TestCase):
    """Benchmark tests for QMoE SwiGLU performance measurement."""

    def test_qmoe_swiglu_throughput_benchmark(self):
        """Comprehensive throughput benchmark for QMoE SwiGLU across different configurations."""
        if disable_cpu_qmoe_tests:
            self.skipTest("QMoE CPU tests disabled")

        print("\n=== QMoE SwiGLU Throughput Benchmark ===")

        # Test configurations: (name, hidden_size, intermediate_size, num_experts, top_k, quant_bits)
        configs = [
            ("Medium-4bit", 2880, 2880, 32, 4, 4),
            ("Medium-8bit", 2880, 2880, 32, 4, 8),
        ]

        batch_size = 1
        sequence_length = 512
        num_runs = 30

        results = []

        for config_name, hidden_size, intermediate_size, num_experts, top_k, quant_bits in configs:
            torch.manual_seed(42)
            numpy.random.seed(42)

            print(f"\nTesting {config_name}:")
            print(f"  Hidden: {hidden_size}, Intermediate: {intermediate_size}")
            print(f"  Experts: {num_experts}, Top-K: {top_k}, Quant: {quant_bits}-bit")

            try:
                # Create config and model
                config = PhiMoEConfig(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_local_experts=num_experts,
                    num_experts_per_tok=top_k,
                )

                qmoe_swiglu = PhiMoESparseMoeBlock(
                    config,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    quant_bits=quant_bits,
                    onnx_dtype=TensorProto.FLOAT,
                )

                # Create test input with fixed sequence length to match ONNX model
                full_hidden_states = torch.randn(batch_size, sequence_length, hidden_size).to(torch.float32)

                # For TTFT simulation, we'll measure single forward pass time
                # This represents the time to process one token in autoregressive generation

                # Initialize variables
                torch_output = None
                ort_output = None

                # Warm up with full context
                for _ in range(3):
                    _ = qmoe_swiglu.forward(full_hidden_states)

                # Benchmark PyTorch TTFT (Time to First Token)
                # Measure time for a single forward pass (represents token generation time)
                torch.manual_seed(42)

                start_time = time.time()
                for _ in range(num_runs):
                    torch_output = qmoe_swiglu.forward(full_hidden_states)
                end_time = time.time()
                torch_ttft_ms = (end_time - start_time) / num_runs * 1000

                # Calculate tokens per second (throughput)
                # For sequence generation, this represents the rate at which we can generate tokens
                torch_tokens_per_sec = 1000.0 / torch_ttft_ms  # 1 token / (time_ms / 1000)

                print(f"  PyTorch TTFT: {torch_ttft_ms:.3f} ms (per token generation time)")
                print(f"  PyTorch Throughput: {torch_tokens_per_sec:.1f} tokens/sec")

                # Benchmark ONNX Runtime
                ort_ttft_ms = 0
                ort_tokens_per_sec = 0
                speedup = 0
                throughput_ratio = 0
                max_diff = 0

                model_updated = qmoe_swiglu.recreate_onnx_model()
                if model_updated and qmoe_swiglu.ort_sess is not None:
                    # Warm up ORT with full context
                    for _ in range(3):
                        _ = qmoe_swiglu.ort_forward(full_hidden_states)

                    torch.manual_seed(42)

                    # Measure ONNX Runtime TTFT (Time to First Token)
                    start_time = time.time()
                    for _ in range(num_runs):
                        ort_output = qmoe_swiglu.ort_forward(full_hidden_states)
                    end_time = time.time()
                    ort_ttft_ms = (end_time - start_time) / num_runs * 1000

                    # Calculate tokens per second for ONNX Runtime
                    ort_tokens_per_sec = 1000.0 / ort_ttft_ms  # 1 token / (time_ms / 1000)

                    speedup = torch_ttft_ms / ort_ttft_ms if ort_ttft_ms > 0 else 0
                    throughput_ratio = ort_tokens_per_sec / torch_tokens_per_sec if torch_tokens_per_sec > 0 else 0

                    print(f"  ONNX RT TTFT: {ort_ttft_ms:.3f} ms (per token generation time)")
                    print(f"  ONNX RT Throughput: {ort_tokens_per_sec:.1f} tokens/sec")
                    print(f"  TTFT Speedup: {speedup:.2f}x")
                    print(f"  Throughput Gain: {throughput_ratio:.2f}x")
                else:
                    print("  ONNX RT: Not available")

                # Calculate max difference if both outputs available
                if torch_output is not None and ort_output is not None:
                    max_diff = (torch_output.cpu() - ort_output.cpu()).abs().max().item()
                    print(f"  Max diff: {max_diff:.6f}")

                results.append(
                    {
                        "config": config_name,
                        "torch_ttft_ms": torch_ttft_ms,
                        "torch_tokens_per_sec": torch_tokens_per_sec,
                        "ort_ttft_ms": ort_ttft_ms,
                        "ort_tokens_per_sec": ort_tokens_per_sec,
                        "speedup": speedup,
                        "throughput_ratio": throughput_ratio,
                        "max_diff": max_diff,
                    }
                )

            except Exception as e:
                print(f"  Error: {e}")
                continue

        # Summary
        print("\n=== Token Generation Time & Throughput Summary ===")
        print(
            f"{'Config':<15} {'PT Time':<10} {'PT tok/s':<10} {'ORT Time':<11} {'ORT tok/s':<11} {'Time Gain':<10} {'Throughput':<11} {'Max Diff':<10}"
        )
        print("-" * 105)
        for result in results:
            config = result["config"]
            torch_ttft = result["torch_ttft_ms"]
            torch_tps = result["torch_tokens_per_sec"]
            ort_ttft = result["ort_ttft_ms"]
            ort_tps = result["ort_tokens_per_sec"]
            speedup = result["speedup"]
            throughput_ratio = result["throughput_ratio"]
            max_diff = result["max_diff"]

            ort_ttft_str = f"{ort_ttft:.3f}" if ort_ttft > 0 else "N/A"
            ort_tps_str = f"{ort_tps:.1f}" if ort_tps > 0 else "N/A"
            speedup_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"
            throughput_str = f"{throughput_ratio:.2f}x" if throughput_ratio > 0 else "N/A"

            print(
                f"{config:<15} {torch_ttft:<10.3f} {torch_tps:<10.1f} {ort_ttft_str:<11} {ort_tps_str:<11} {speedup_str:<10} {throughput_str:<11} {max_diff:<10.6f}"
            )

        print("\nNotes:")
        print("- Time: Token generation time in ms (lower is better)")
        print("- tok/s: Tokens per second throughput (higher is better)")
        print("- Time Gain: ORT speedup for latency (higher is better)")
        print("- Throughput: ORT throughput improvement (higher is better)")


if __name__ == "__main__":
    unittest.main()
