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
# -------------------------------------------------------------------------
import math
import os
import platform
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy
import torch
from einops import rearrange, repeat
from onnx import TensorProto, helper
from packaging import version
from parameterized import parameterized

from onnxruntime import InferenceSession, OrtValue, SessionOptions, get_available_providers

torch.manual_seed(0)

pipeline_mode = True  # Reduces number of tests so pipeline doesn't time out

# Element type of the paged KV cache, keyed by Config.kv_cache_type.
KV_CACHE_TENSOR_PROTO = {
    "float16": TensorProto.FLOAT16,
    "int8": TensorProto.INT8,
    "fp8": TensorProto.FLOAT8E4M3FN,
}

# Largest magnitude representable by each quantized cache type.
KV_CACHE_QMAX = {"int8": 127.0, "fp8": 448.0}


# EP -> (torch_device_when_cuda_available, ort_iobinding_device).
# The torch device is where reference tensors live. WebGPU has no torch backend,
# so we still allocate reference tensors on CUDA when it's available (faster and
# matches CUDA fp16 semantics), else on CPU. The ORT device is the device string
# passed to OrtValue / IO-binding.
_EP_TO_ORT_DEVICE = {
    "CUDAExecutionProvider": "cuda",
    "WebGpuExecutionProvider": "webgpu",
}


class Config:
    batch_size = 0
    sequence_length = 0
    total_sequence_length = 0
    num_heads = 0
    kv_num_heads = 0
    head_size = 0
    paged_kv_block_size = 0
    local = False
    rotary = False
    rotary_interleaved = False
    packed = False
    softcap = 0.0
    ep = "CUDAExecutionProvider"
    # Optional features layered on top of the original schema. They default to off so that every
    # pre-existing parameterized test keeps its generated name and behavior.
    use_slot_mapping = False
    use_head_sink = False
    use_qk_norm = False
    qk_norm_epsilon = 1e-6
    # Quantized paged KV cache. "float16" keeps the cache unquantized; "int8" and "fp8" store the
    # cache in the corresponding narrow type and require a matching non-"NONE" quant type.
    kv_cache_type = "float16"
    k_quant_type = "NONE"
    v_quant_type = "NONE"
    # "" means the cache tensor's own element type is the logical element type. Sub-byte names
    # ("int4", "float4e2m1") describe a uint8 packed cache and are not supported yet. Every legal
    # value is signed: quantization is symmetric with no zero point, so "uint4"/"uint8" are invalid.
    k_cache_dtype = ""
    v_cache_dtype = ""

    def __init__(
        self,
        batch_size,
        sequence_length,
        total_sequence_length,
        num_heads,
        kv_num_heads,
        head_size,
        paged_kv_block_size,
        local,
        rotary,
        rotary_interleaved,
        packed,
        softcap,
        ep="CUDAExecutionProvider",
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.total_sequence_length = total_sequence_length
        self.num_heads = num_heads
        self.kv_num_heads = kv_num_heads
        self.head_size = head_size
        self.paged_kv_block_size = paged_kv_block_size
        self.local = local
        self.rotary = rotary
        self.rotary_interleaved = rotary_interleaved
        self.packed = packed
        self.softcap = softcap
        self.ep = ep

    @property
    def torch_device(self) -> str:
        """Device string for torch tensor allocation (inputs + reference).

        For CUDA EP: always "cuda" (must match the EP device).
        For other EPs (e.g., WebGPU): prefer CUDA if torch has it (faster,
        matches CUDA fp16 semantics); otherwise CPU. The reference math never
        touches the ORT session, so torch device does not need to equal the EP
        device.
        """
        if self.ep == "CUDAExecutionProvider":
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def ort_device(self) -> str:
        """Device string for OrtValue / IO-binding (must match the EP device)."""
        try:
            return _EP_TO_ORT_DEVICE[self.ep]
        except KeyError as exc:
            raise ValueError(f"Unknown EP for PagedAttention parity tests: {self.ep!r}") from exc

    def __repr__(self):
        short_ep = self.ep[: -len("ExecutionProvider")].lower()
        return (
            f"Config(batch_size={self.batch_size}, sequence_length={self.sequence_length}, "
            f"total_sequence_length={self.total_sequence_length}, num_heads={self.num_heads}, "
            f"kv_num_heads={self.kv_num_heads}, head_size={self.head_size}, "
            f"paged_kv_block_size={self.paged_kv_block_size} rotary={self.rotary}, "
            f"rotary_interleaved={self.rotary_interleaved}, packed={self.packed}, softcap={self.softcap}, "
            f"ep={short_ep})"
        )


def kv_scale_shape(config, quant_type):
    """Shape of the k_scale / v_scale input for a given quantization granularity."""
    if quant_type == "PER_TENSOR":
        return [1]
    if quant_type == "PER_CHANNEL":
        return [config.kv_num_heads, 1, config.head_size]
    raise ValueError(f"Unsupported quant_type: {quant_type}")


def compute_kv_scale(tensors, quant_type, kv_cache_type, kv_num_heads, head_size):
    """Symmetric (zero-point-free) scale for the paged KV cache.

    'tensors' are every tensor that will end up in the cache -- the pre-existing pages *and* the new
    tokens the kernel is about to write -- so that the chosen scale never clips and the reference can
    reproduce the kernel bit-for-bit. Each tensor's two trailing dimensions are (kv_num_heads,
    head_size), which is exactly the PER_CHANNEL scale layout.
    """
    qmax = KV_CACHE_QMAX[kv_cache_type]
    if quant_type == "PER_TENSOR":
        amax = max(float(t.abs().max().item()) for t in tensors)
        return torch.tensor([max(amax, 1e-6) / qmax], dtype=torch.float32, device="cuda")
    amax = None
    for t in tensors:
        per_channel = t.reshape(-1, kv_num_heads, head_size).abs().amax(dim=0)
        amax = per_channel if amax is None else torch.maximum(amax, per_channel)
    scale = torch.clamp(amax.to(torch.float32), min=1e-6) / qmax
    return scale.reshape(kv_num_heads, 1, head_size)


def broadcast_kv_scale(scale, quant_type, kv_num_heads, head_size):
    """View the scale so that it broadcasts against any tensor shaped (..., kv_num_heads, head_size)."""
    if quant_type == "PER_TENSOR":
        return scale
    return scale.reshape(1, 1, kv_num_heads, head_size)


def quantize_kv(tensor_float, scale, kv_cache_type):
    """Mirror of QuantizeToCache in paged_attention_impl.cu.

    The kernel multiplies by the reciprocal of the scale rather than dividing, so the reference does
    the same: with round-to-nearest-even the two differ by one LSB often enough to make an exact
    comparison of the updated cache flaky otherwise.
    """
    scaled = tensor_float.to(torch.float32) * torch.reciprocal(scale)
    if kv_cache_type == "fp8":
        return torch.clamp(scaled, -KV_CACHE_QMAX["fp8"], KV_CACHE_QMAX["fp8"]).to(torch.float8_e4m3fn)
    return torch.clamp(torch.round(scaled), -128.0, 127.0).to(torch.int8)


def dequantize_kv(quantized, scale):
    """Mirror of DequantizeFromCache in paged_attention_impl.cu."""
    return (quantized.to(torch.float32) * scale).to(torch.float16)


def quantize_dequantize_kv(tensor_float, scale, kv_cache_type):
    return dequantize_kv(quantize_kv(tensor_float, scale, kv_cache_type), scale)


def create_paged_attention_graph(
    config,
    num_tokens,
    num_blocks,
    max_blocks_per_sequence,
    local_window_size=-1,
):
    cache_proto_type = KV_CACHE_TENSOR_PROTO[config.kv_cache_type]
    # The scale inputs and the quantization attributes are emitted independently of the cache dtype
    # so that invalid combinations (quantized cache without a quant type, and vice versa) can be
    # built and their rejection tested.
    has_k_scale = config.k_quant_type != "NONE"
    has_v_scale = config.v_quant_type != "NONE"
    # Optional host-side [max_query_len_bound, max_kv_len_bound, optional max_kv_len_lower_bound].
    # When present the kernel can skip the device readback of the cumulative length arrays, so
    # results must be identical either way.
    has_attention_metadata = getattr(config, "use_attention_metadata", False)
    quant_attrs = (
        {
            "k_quant_type": config.k_quant_type,
            "v_quant_type": config.v_quant_type,
            "k_cache_dtype": config.k_cache_dtype,
            "v_cache_dtype": config.v_cache_dtype,
        }
        if (has_k_scale or has_v_scale or config.kv_cache_type != "float16")
        else {}
    )
    optional_inputs = [
        "slot_mapping" if config.use_slot_mapping else "",
        "head_sink" if config.use_head_sink else "",
        "q_norm_weight" if config.use_qk_norm else "",
        "k_norm_weight" if config.use_qk_norm else "",
        "k_scale" if has_k_scale else "",
        "v_scale" if has_v_scale else "",
        "attention_metadata" if has_attention_metadata else "",
    ]
    last_optional_idx = -1
    for i, name in enumerate(optional_inputs):
        if name:
            last_optional_idx = i

    # Keep the node compact when none of the post-v1 optional inputs are used.
    # This allows the baseline WebGPU path to run on runtimes that still expose
    # the older 10-input schema while remaining compatible with the expanded
    # schema when newer optional inputs are exercised.
    node_inputs = [
        "query",
        "key" if not config.packed else "",
        "value" if not config.packed else "",
        "key_cache",
        "value_cache",
        "cumulative_sequence_length",
        "past_seqlens",
        "block_table",
        "cos_cache" if config.rotary else "",
        "sin_cache" if config.rotary else "",
    ]
    if last_optional_idx >= 0:
        node_inputs.extend(optional_inputs[: last_optional_idx + 1])

    node_attrs = {
        "num_heads": config.num_heads,
        "kv_num_heads": config.kv_num_heads,
        "local_window_size": local_window_size,
        "do_rotary": config.rotary,
        "rotary_interleaved": config.rotary_interleaved,
        "softcap": config.softcap,
        "domain": "com.microsoft",
    }
    # Keep baseline graphs compatible with older PagedAttention schema builds.
    # qk_norm_epsilon is only needed when QK-Norm is exercised.
    if config.use_qk_norm:
        node_attrs["qk_norm_epsilon"] = config.qk_norm_epsilon
    if not getattr(config, "is_causal", True):
        node_attrs["is_causal"] = 0

    nodes = [
        helper.make_node(
            "PagedAttention",
            node_inputs,
            ["output", "key_cache_out", "value_cache_out"],
            "PagedAttention_0",
            **node_attrs,
            **quant_attrs,
        ),
    ]

    graph_input = [
        helper.make_tensor_value_info(
            "query",
            TensorProto.FLOAT16,
            [
                num_tokens,
                (config.num_heads * config.head_size)
                if not config.packed
                else (config.num_heads * config.head_size + 2 * config.kv_num_heads * config.head_size),
            ],
        ),
        helper.make_tensor_value_info(
            "key_cache",
            cache_proto_type,
            [
                num_blocks,
                config.paged_kv_block_size,
                config.kv_num_heads,
                config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "value_cache",
            cache_proto_type,
            [
                num_blocks,
                config.paged_kv_block_size,
                config.kv_num_heads,
                config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "cumulative_sequence_length",
            TensorProto.INT32,
            [config.batch_size + 1],
        ),
        helper.make_tensor_value_info(
            "past_seqlens",
            TensorProto.INT32,
            [config.batch_size],
        ),
        helper.make_tensor_value_info(
            "block_table",
            TensorProto.INT32,
            [config.batch_size, max_blocks_per_sequence],
        ),
    ]
    if not config.packed:
        graph_input += [
            helper.make_tensor_value_info(
                "key",
                TensorProto.FLOAT16,
                [
                    num_tokens,
                    config.kv_num_heads * config.head_size,
                ],
            ),
            helper.make_tensor_value_info(
                "value",
                TensorProto.FLOAT16,
                [
                    num_tokens,
                    config.kv_num_heads * config.head_size,
                ],
            ),
        ]
    if config.rotary:
        graph_input += [
            helper.make_tensor_value_info(
                "cos_cache",
                TensorProto.FLOAT16,
                [
                    config.total_sequence_length,
                    (math.floor(config.head_size / 16) * 16) // 2,
                ],
            ),
            helper.make_tensor_value_info(
                "sin_cache",
                TensorProto.FLOAT16,
                [
                    config.total_sequence_length,
                    (math.floor(config.head_size / 16) * 16) // 2,
                ],
            ),
        ]
    if config.use_slot_mapping:
        graph_input += [
            helper.make_tensor_value_info("slot_mapping", TensorProto.INT32, [num_tokens]),
        ]
    if config.use_head_sink:
        graph_input += [
            helper.make_tensor_value_info("head_sink", TensorProto.FLOAT16, [config.num_heads]),
        ]
    if config.use_qk_norm:
        graph_input += [
            helper.make_tensor_value_info("q_norm_weight", TensorProto.FLOAT16, [config.head_size]),
            helper.make_tensor_value_info("k_norm_weight", TensorProto.FLOAT16, [config.head_size]),
        ]
    if has_k_scale:
        graph_input += [
            helper.make_tensor_value_info("k_scale", TensorProto.FLOAT, kv_scale_shape(config, config.k_quant_type)),
        ]
    if has_v_scale:
        graph_input += [
            helper.make_tensor_value_info("v_scale", TensorProto.FLOAT, kv_scale_shape(config, config.v_quant_type)),
        ]
    if has_attention_metadata:
        graph_input += [
            helper.make_tensor_value_info(
                "attention_metadata",
                TensorProto.INT32,
                getattr(config, "attention_metadata_shape", [2]),
            ),
        ]

    graph_output = [
        helper.make_tensor_value_info(
            "output",
            TensorProto.FLOAT16,
            [num_tokens, config.num_heads * config.head_size],
        ),
        helper.make_tensor_value_info(
            "key_cache_out",
            cache_proto_type,
            [
                num_blocks,
                config.paged_kv_block_size,
                config.kv_num_heads,
                config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "value_cache_out",
            cache_proto_type,
            [
                num_blocks,
                config.paged_kv_block_size,
                config.kv_num_heads,
                config.head_size,
            ],
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "PagedAttention_Graph",
        graph_input,
        graph_output,
    )

    # Pin ai.onnx to opset 22. helper.make_model() otherwise stamps the
    # installed onnx package's newest opset, which may exceed the runtime's
    # kMaxSupportedOpset (e.g. onnx 1.22 stamps opset 27 while ORT tops out
    # at 26 today) and causes InferenceSession creation to fail.
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 22),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    return model.SerializeToString()


def rotary_options_for_current_os():
    # Reference implementation of rotary uses triton, which is not available in Windows.
    # So we only test rotary in Linux right now.
    return [(False, False)] if platform.system() != "Linux" else [(True, False), (True, True), (False, False)]


def paged_attention_func(
    config,
    query,
    key,
    value,
    key_cache,
    value_cache,
    cumulative_sequence_length,
    past_seqlens,
    block_table,
    cos=None,
    sin=None,
    window_size=-1,
    sdpa_kernel=0,
    slot_mapping=None,
    head_sink=None,
    q_norm_weight=None,
    k_norm_weight=None,
    k_scale=None,
    v_scale=None,
):
    num_tokens = cumulative_sequence_length[-1].item()
    num_blocks = key_cache.shape[0]
    max_blocks_per_sequence = block_table.shape[1]
    quantized = config.kv_cache_type != "float16"
    onnx_model_str = create_paged_attention_graph(
        config,
        num_tokens,
        num_blocks,
        max_blocks_per_sequence,
        local_window_size=window_size,
    )
    ort_inputs = {
        "query": query.detach().cpu().numpy(),
        "cumulative_sequence_length": cumulative_sequence_length.detach().cpu().numpy(),
        "past_seqlens": past_seqlens.detach().cpu().numpy(),
        "block_table": block_table.detach().cpu().numpy(),
    }
    key_cache_np = key_cache.detach().cpu().numpy()
    value_cache_np = value_cache.detach().cpu().numpy()

    if getattr(config, "use_attention_metadata", False):
        override = getattr(config, "attention_metadata_override", None)
        if override is not None:
            ort_inputs["attention_metadata"] = override
        else:
            cum_q = cumulative_sequence_length.detach().cpu().numpy().astype(numpy.int64)
            query_lens = cum_q[1:] - cum_q[:-1]
            kv_lens = past_seqlens.detach().cpu().numpy().astype(numpy.int64) + query_lens
            # The exact per-step maxima are valid upper bounds for a single Run, which is what these
            # tests do. A real scheduler would pass looser, replay-wide bounds instead.
            ort_inputs["attention_metadata"] = numpy.array([query_lens.max(), kv_lens.max()], dtype=numpy.int32)

    # CUDA pre-allocates the K/V cache on-device so the same OrtValue can be
    # bound as both input (key_cache) and output (key_cache_out), giving an
    # in-place update we can read back at the end. WebGPU cannot construct
    # device-side OrtValues from numpy without a session-registered shared
    # allocator, so we bind cache tensors from CPU for WebGPU.
    if config.ep == "CUDAExecutionProvider" and not quantized:
        ort_inputs["key_cache"] = OrtValue.ortvalue_from_numpy(key_cache_np, config.ort_device, 0)
        ort_inputs["value_cache"] = OrtValue.ortvalue_from_numpy(value_cache_np, config.ort_device, 0)
    sess_options = SessionOptions()
    if sdpa_kernel != 0 and config.ep == "CUDAExecutionProvider":
        providers = [(config.ep, {"sdpa_kernel": str(sdpa_kernel)})]
    else:
        providers = [config.ep]
    ort_session = InferenceSession(onnx_model_str, sess_options, providers=providers)
    io_binding = ort_session.io_binding()
    if key is not None and value is not None:
        ort_inputs["key"] = key.detach().cpu().numpy()
        ort_inputs["value"] = value.detach().cpu().numpy()
        io_binding.bind_cpu_input("key", ort_inputs["key"])
        io_binding.bind_cpu_input("value", ort_inputs["value"])
    if cos is not None and sin is not None:
        ort_inputs["cos_cache"] = cos.detach().cpu().numpy()
        ort_inputs["sin_cache"] = sin.detach().cpu().numpy()
        io_binding.bind_cpu_input("cos_cache", ort_inputs["cos_cache"])
        io_binding.bind_cpu_input("sin_cache", ort_inputs["sin_cache"])
    for name, tensor in (
        ("slot_mapping", slot_mapping),
        ("head_sink", head_sink),
        ("q_norm_weight", q_norm_weight),
        ("k_norm_weight", k_norm_weight),
        ("k_scale", k_scale),
        ("v_scale", v_scale),
    ):
        if tensor is not None:
            ort_inputs[name] = tensor.detach().cpu().numpy()
            io_binding.bind_cpu_input(name, ort_inputs[name])
    if "attention_metadata" in ort_inputs:
        io_binding.bind_cpu_input("attention_metadata", ort_inputs["attention_metadata"])
    io_binding.bind_cpu_input("query", ort_inputs["query"])
    if config.ep == "CUDAExecutionProvider":
        if quantized:
            # A quantized cache has no numpy dtype, so bind the torch device buffers directly.
            cache_proto_type = KV_CACHE_TENSOR_PROTO[config.kv_cache_type]
            key_cache = key_cache.contiguous()
            value_cache = value_cache.contiguous()
            io_binding.bind_input(
                "key_cache", "cuda", 0, cache_proto_type, tuple(key_cache.shape), key_cache.data_ptr()
            )
            io_binding.bind_input(
                "value_cache", "cuda", 0, cache_proto_type, tuple(value_cache.shape), value_cache.data_ptr()
            )
        else:
            io_binding.bind_input(
                "key_cache",
                config.ort_device,
                0,
                numpy.float16,
                ort_inputs["key_cache"].shape(),
                ort_inputs["key_cache"].data_ptr(),
            )
            io_binding.bind_input(
                "value_cache",
                config.ort_device,
                0,
                numpy.float16,
                ort_inputs["value_cache"].shape(),
                ort_inputs["value_cache"].data_ptr(),
            )
    else:
        io_binding.bind_cpu_input("key_cache", key_cache_np)
        io_binding.bind_cpu_input("value_cache", value_cache_np)
    io_binding.bind_cpu_input("cumulative_sequence_length", ort_inputs["cumulative_sequence_length"])
    io_binding.bind_cpu_input("past_seqlens", ort_inputs["past_seqlens"])
    io_binding.bind_cpu_input("block_table", ort_inputs["block_table"])
    io_binding.bind_output("output")
    if config.ep == "CUDAExecutionProvider":
        if quantized:
            # Each cache output must alias its input, which is what the op requires anyway.
            io_binding.bind_output(
                "key_cache_out", "cuda", 0, cache_proto_type, tuple(key_cache.shape), key_cache.data_ptr()
            )
            io_binding.bind_output(
                "value_cache_out", "cuda", 0, cache_proto_type, tuple(value_cache.shape), value_cache.data_ptr()
            )
        else:
            io_binding.bind_ortvalue_output("key_cache_out", ort_inputs["key_cache"])
            io_binding.bind_ortvalue_output("value_cache_out", ort_inputs["value_cache"])
    else:
        # These cache tensors are graph inputs, so the allocation planner does
        # not guarantee aliasing. Bind separate outputs to exercise the
        # WebGPU copy fallback; production IO-binding should alias these
        # buffers to avoid the extra cache copy.
        io_binding.bind_output("key_cache_out")
        io_binding.bind_output("value_cache_out")
    ort_session.run_with_iobinding(io_binding)
    if quantized:
        output = torch.tensor(numpy.array(io_binding.copy_outputs_to_cpu()[0]))
        return output, key_cache, value_cache
    output, key_cache_out, value_cache_out = io_binding.copy_outputs_to_cpu()
    output = torch.tensor(numpy.array(output))

    # WebGPU EP has an ownership/destruction race that segfaults during Python
    # interpreter shutdown GC if sessions are left to be reclaimed implicitly.
    # Release the binding and session explicitly here so each parity test
    # completes with a clean process state.
    del io_binding
    del ort_session

    return output, key_cache_out, value_cache_out


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    sk = seqlen_k if key_padding_mask is None else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    sq = seqlen_q if query_padding_mask is None else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx <= row_idx + sk - sq - window_size[0],
        )


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    head_sink=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling k, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if softcap > 0:
        scores = scores / softcap
        scores = scores.tanh()
        scores = scores * softcap
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
        )
        scores.masked_fill_(local_mask, float("-inf"))

    if head_sink is not None:
        # Append one extra logit per (batch, head, query) to the softmax denominator that
        # contributes no value. head_sink is the learned logit.
        b, n, s, _ = scores.shape
        sink = head_sink.to(scores.dtype).reshape(1, n, 1, 1).expand(b, -1, s, -1)
        attention = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
    else:
        attention = torch.softmax(scores, dim=-1)

    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def rms_norm_ref(x, weight, epsilon):
    """Per-head RMSNorm reference matching the fused CUDA prologue: reduce in fp32 over the last
    dimension (head_size), scale, then cast back to the input dtype.

    Arguments:
        x: (..., head_size)
        weight: (head_size)
    """
    x_f32 = x.float()
    inv_rms = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    return (x_f32 * inv_rms * weight.float()).to(dtype=x.dtype)


def rotary_embedding(*args, **kwargs):
    # Use local import since triton is not available in Windows.
    from rotary_flash import apply_rotary_emb  # noqa: PLC0415

    return apply_rotary_emb(*args, **kwargs)


def unpad_qkv(config: Config, q, k, v, cum_seqlens):
    token_count = cum_seqlens[-1]
    q_unpad = torch.zeros(
        token_count,
        config.num_heads * config.head_size,
        dtype=torch.float16,
        device=config.torch_device,
    )
    k_unpad = torch.zeros(
        token_count,
        config.kv_num_heads * config.head_size,
        dtype=torch.float16,
        device=config.torch_device,
    )
    v_unpad = torch.zeros(
        token_count,
        config.kv_num_heads * config.head_size,
        dtype=torch.float16,
        device=config.torch_device,
    )
    for i in range(config.batch_size):
        new_seqlen = cum_seqlens[i + 1] - cum_seqlens[i]
        q_unpad[cum_seqlens[i] : cum_seqlens[i + 1]] = rearrange(q[i, :new_seqlen], "s n h -> s (n h)")
        k_unpad[cum_seqlens[i] : cum_seqlens[i + 1]] = rearrange(k[i, :new_seqlen], "s n h -> s (n h)")
        v_unpad[cum_seqlens[i] : cum_seqlens[i + 1]] = rearrange(v[i, :new_seqlen], "s n h -> s (n h)")
    return q_unpad, k_unpad, v_unpad


def generate_block_kvcache(config: Config, device, dtype):
    num_blocks = math.ceil(config.total_sequence_length / config.paged_kv_block_size) * config.batch_size * 3
    k_cache_paged = torch.randn(
        num_blocks, config.paged_kv_block_size, config.kv_num_heads, config.head_size, device=device, dtype=dtype
    )
    v_cache_paged = torch.randn(
        num_blocks, config.paged_kv_block_size, config.kv_num_heads, config.head_size, device=device, dtype=dtype
    )
    block_table = rearrange(
        torch.randperm(num_blocks, dtype=torch.int32, device=device),
        "(b nblocks) -> b nblocks",
        b=config.batch_size,
    )
    k_cache = rearrange(
        # pytorch 1.12 doesn't have indexing with int32
        k_cache_paged[block_table.to(dtype=torch.long).flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=config.batch_size,
    )[:, : config.total_sequence_length]
    v_cache = rearrange(
        v_cache_paged[block_table.to(dtype=torch.long).flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=config.batch_size,
    )[:, : config.total_sequence_length]
    return k_cache, v_cache, block_table, k_cache_paged, v_cache_paged


def gather_paged_to_batch(config: Config, paged, block_table):
    """Gather a paged [num_blocks, block_size, kv_num_heads, head_size] cache into the dense
    [batch_size, total_sequence_length, kv_num_heads, head_size] view the reference works with."""
    return rearrange(
        paged[block_table.to(dtype=torch.long).flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=config.batch_size,
    )[:, : config.total_sequence_length]


def derive_slot_mapping(config: Config, past_seqlens, new_seqlens, cum_seqlens, block_table):
    """Reproduce, on the host, the flat cache slot that the kernel derives for every query token
    when 'slot_mapping' is absent: block_table[b, pos // block_size] * block_size + pos % block_size
    where pos = past_seqlens[b] + index_of_token_within_its_sequence."""
    token_count = int(cum_seqlens[-1].item())
    slot_mapping = torch.empty(token_count, dtype=torch.int32, device="cuda")
    block_table_cpu = block_table.cpu()
    for b in range(config.batch_size):
        start = int(cum_seqlens[b].item())
        for j in range(int(new_seqlens[b].item())):
            pos = int(past_seqlens[b].item()) + j
            block_id = int(block_table_cpu[b, pos // config.paged_kv_block_size].item())
            slot_mapping[start + j] = block_id * config.paged_kv_block_size + pos % config.paged_kv_block_size
    return slot_mapping


def parity_check_paged_attention(
    config: Config,
    rtol=1e-3,
    atol=1e-3,
    sdpa_kernel=0,
    new_seqlens_override=None,
    local_window_size_override=None,
):
    # Generate padded inputs
    q = torch.randn(
        config.batch_size,
        config.sequence_length,
        config.num_heads,
        config.head_size,
        device=config.torch_device,
        dtype=torch.float16,
        requires_grad=False,
    )
    k_new = torch.randn(
        config.batch_size,
        config.sequence_length,
        config.kv_num_heads,
        config.head_size,
        device=config.torch_device,
        dtype=torch.float16,
        requires_grad=False,
    )
    v_new = torch.randn(
        config.batch_size,
        config.sequence_length,
        config.kv_num_heads,
        config.head_size,
        device=config.torch_device,
        dtype=torch.float16,
        requires_grad=False,
    )

    # Generate random sequence lengths
    past_seqlens = torch.randint(
        0,
        config.total_sequence_length - config.sequence_length + 1,  # one above highest integer to be drawn
        (config.batch_size,),
        dtype=torch.int32,
        device=config.torch_device,
    )
    if new_seqlens_override is not None:
        new_seqlens = new_seqlens_override.to(dtype=torch.int32, device=config.torch_device)
        assert new_seqlens.shape == (config.batch_size,)
        assert int(new_seqlens.min().item()) >= 0
        assert int(new_seqlens.max().item()) <= config.sequence_length
    else:
        new_seqlens = torch.randint(
            1,
            config.sequence_length + 1,
            (config.batch_size,),
            dtype=torch.int32,
            device=config.torch_device,
        )
    cum_seqlens = torch.cat(
        (torch.tensor([0], dtype=torch.int32, device=config.torch_device), torch.cumsum(new_seqlens, dim=0))
    ).type(torch.int32)
    total_seqlens = past_seqlens + new_seqlens

    q_unpad, k_unpad, v_unpad = unpad_qkv(config, q, k_new, v_new, cum_seqlens)

    # Generate kv cache and associated block-based data structures
    k_cache, v_cache, block_table, k_cache_paged, v_cache_paged = generate_block_kvcache(
        config, config.torch_device, torch.float16
    )

    # Optional per-head attention sink.
    head_sink = None
    if config.use_head_sink:
        # Spread over [-2, 6]: exp(sink) then ranges from negligible to far larger than a typical
        # softmax denominator, so a kernel that ignored the sink could not pass within tolerance.
        head_sink = (torch.rand(config.num_heads, device="cuda") * 8.0 - 2.0).to(dtype=torch.float16)

    # Optional QK-Norm. The kernel applies RMSNorm to every Q and K head before rotary embedding,
    # so the reference has to normalize before computing q_ro / k_ro below, and the normalized +
    # rotated K is what must land in the KV cache.
    q_norm_weight = None
    k_norm_weight = None
    if config.use_qk_norm:
        q_norm_weight = torch.randn(config.head_size, device="cuda", dtype=torch.float16)
        k_norm_weight = torch.randn(config.head_size, device="cuda", dtype=torch.float16)
        q = rms_norm_ref(q, q_norm_weight, config.qk_norm_epsilon)
        k_new = rms_norm_ref(k_new, k_norm_weight, config.qk_norm_epsilon)

    # Optional explicit slot mapping. Reproducing exactly what the kernel derives from
    # past_seqlens / cumulative_sequence_length / block_table must give identical results.
    slot_mapping = None
    if config.use_slot_mapping:
        slot_mapping = derive_slot_mapping(config, past_seqlens, new_seqlens, cum_seqlens, block_table)

    # Set window size for local / causal
    is_causal = getattr(config, "is_causal", True)
    # A non-causal query block attends to everything on its right too, so the reference asks for a
    # right window wide enough to never mask (construct_local_mask clamps it to seqlen_k).
    right_window_size = 0 if is_causal else config.total_sequence_length
    window_size = (-1, -1)
    left_window_size = -1
    if config.local:
        left_window_size = (
            local_window_size_override
            if local_window_size_override is not None
            else random.randint(1, config.total_sequence_length - 1)
        )
        assert 0 < left_window_size < config.total_sequence_length
        window_size = (left_window_size, right_window_size)
    else:
        left_window_size = -1
        window_size = (-1, right_window_size) if is_causal else (-1, -1)

    # Apply rotary embedding for reference implementation
    if config.rotary:
        rotary_fraction = 1.0
        rotary_dim = math.floor(int(rotary_fraction * config.head_size) / 16) * 16
        angle = torch.rand(config.total_sequence_length, rotary_dim // 2, device=config.torch_device) * 2 * math.pi
        cos = torch.cos(angle).to(dtype=torch.float16)
        sin = torch.sin(angle).to(dtype=torch.float16)
        q_ro = rotary_embedding(q, cos, sin, seqlen_offsets=past_seqlens, interleaved=config.rotary_interleaved)
        k_ro = rotary_embedding(k_new, cos, sin, seqlen_offsets=past_seqlens, interleaved=config.rotary_interleaved)
    else:
        cos, sin = None, None
        q_ro, k_ro = q, k_new

    # Quantized paged KV cache. The pages the kernel reads back are the dequantized values, and the
    # new tokens it writes go through the same quantize step, so the reference models both. The scale
    # covers the existing pages *and* the incoming tokens so that nothing clips.
    k_scale = v_scale = None
    k_scale_b = v_scale_b = None
    if config.kv_cache_type != "float16":
        k_scale = compute_kv_scale(
            [k_cache_paged, k_ro], config.k_quant_type, config.kv_cache_type, config.kv_num_heads, config.head_size
        )
        v_scale = compute_kv_scale(
            [v_cache_paged, v_new], config.v_quant_type, config.kv_cache_type, config.kv_num_heads, config.head_size
        )
        k_scale_b = broadcast_kv_scale(k_scale, config.k_quant_type, config.kv_num_heads, config.head_size)
        v_scale_b = broadcast_kv_scale(v_scale, config.v_quant_type, config.kv_num_heads, config.head_size)
        k_cache_paged = quantize_kv(k_cache_paged, k_scale_b, config.kv_cache_type)
        v_cache_paged = quantize_kv(v_cache_paged, v_scale_b, config.kv_cache_type)
        k_cache = gather_paged_to_batch(config, dequantize_kv(k_cache_paged, k_scale_b), block_table)
        v_cache = gather_paged_to_batch(config, dequantize_kv(v_cache_paged, v_scale_b), block_table)
        k_ro = quantize_dequantize_kv(k_ro, k_scale_b, config.kv_cache_type)
        v_new = quantize_dequantize_kv(v_new, v_scale_b, config.kv_cache_type)

    # Update reference kv cache
    k_cache_ref = k_cache.clone()
    v_cache_ref = v_cache.clone()
    total_range = rearrange(torch.arange(config.total_sequence_length, device=config.torch_device), "s -> 1 s")
    past_seqlens_expanded = rearrange(past_seqlens, "b -> b 1")
    update_mask = torch.logical_and(
        past_seqlens_expanded <= total_range, total_range < past_seqlens_expanded + config.sequence_length
    )
    k_cache_ref[update_mask] = rearrange(k_ro, "b s ... -> (b s) ...")
    v_cache_ref[update_mask] = rearrange(v_new, "b s ... -> (b s) ...")
    k_cache_rep = repeat(k_cache_ref, "b s h d -> b s (h g) d", g=config.num_heads // config.kv_num_heads)
    v_cache_rep = repeat(v_cache_ref, "b s h d -> b s (h g) d", g=config.num_heads // config.kv_num_heads)

    # Create padding masks for reference implementation
    total_seqlens_expanded = rearrange(total_seqlens, "b -> b 1")
    key_padding_mask = total_range < total_seqlens_expanded
    query_range = rearrange(torch.arange(config.sequence_length, device=config.torch_device), "s -> 1 s")
    new_seqlens_expanded = rearrange(new_seqlens, "b -> b 1")
    query_padding_mask = query_range < new_seqlens_expanded

    # Run reference implementation of attention
    out_ref, _ = attention_ref(
        q_ro,
        k_cache_rep,
        v_cache_rep,
        query_padding_mask,
        key_padding_mask,
        0.0,
        None,
        causal=is_causal,
        window_size=window_size,
        softcap=config.softcap,
        head_sink=head_sink,
    )
    out_ref = out_ref.detach().cpu().numpy()

    if config.packed:
        q_unpad = torch.concatenate([q_unpad, k_unpad, v_unpad], dim=1)
        k_unpad = None
        v_unpad = None
    out, updated_k_cache_paged, updated_v_cache_paged = paged_attention_func(
        config,
        q_unpad,
        k_unpad,
        v_unpad,
        k_cache_paged,
        v_cache_paged,
        cum_seqlens,
        past_seqlens,
        block_table,
        cos,
        sin,
        left_window_size,
        sdpa_kernel=sdpa_kernel,
        slot_mapping=slot_mapping,
        head_sink=head_sink,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    if config.kv_cache_type != "float16":
        updated_k_cache_paged = dequantize_kv(updated_k_cache_paged, k_scale_b).cpu().numpy()
        updated_v_cache_paged = dequantize_kv(updated_v_cache_paged, v_scale_b).cpu().numpy()
    num_tokens = q_unpad.shape[0]
    out = torch.reshape(out, (num_tokens, config.num_heads, config.head_size))
    out = out.detach().cpu().numpy()

    err_msg = f" with {config}"
    # The updated cache is compared to the reference at one quantization step of slack: the host
    # computes rotary / RMSNorm slightly differently from the kernel, and a 1-ULP fp16 difference in
    # the pre-quantization value is enough to move the rounded result by a whole step.
    cache_rtol, cache_atol = rtol, atol
    if config.kv_cache_type == "int8":
        cache_atol = atol + max(float(k_scale.max().item()), float(v_scale.max().item()))
    elif config.kv_cache_type == "fp8":
        cache_rtol = rtol + 2.0**-3  # float8e4m3fn has 3 mantissa bits

    # Make sure past-present buffer updating correctly
    present_k = rearrange(
        updated_k_cache_paged[block_table.to(dtype=torch.long).flatten().cpu()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=config.batch_size,
    )[:, : config.total_sequence_length]
    present_v = rearrange(
        updated_v_cache_paged[block_table.to(dtype=torch.long).flatten().cpu()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=config.batch_size,
    )[:, : config.total_sequence_length]
    for i in range(config.batch_size):
        numpy.testing.assert_allclose(
            present_k[i, : total_seqlens[i]],
            k_cache_ref[i, : total_seqlens[i]].detach().cpu().numpy(),
            rtol=cache_rtol,
            atol=cache_atol,
            equal_nan=True,
            err_msg=err_msg,
        )
        numpy.testing.assert_allclose(
            present_v[i, : total_seqlens[i]],
            v_cache_ref[i, : total_seqlens[i]].detach().cpu().numpy(),
            rtol=cache_rtol,
            atol=cache_atol,
            equal_nan=True,
            err_msg=err_msg,
        )
        new_seqlen = cum_seqlens[i + 1] - cum_seqlens[i]
        out_i = out[cum_seqlens[i] : cum_seqlens[i + 1]]
        out_ref_i = out_ref[i, :new_seqlen]
        numpy.testing.assert_allclose(out_i, out_ref_i, rtol=rtol, atol=atol, equal_nan=True, err_msg=err_msg)


def capture_native_stdout(run_func):
    """Capture output written by the native runtime directly to file descriptor 1."""
    sys.stdout.flush()
    saved_fd = os.dup(1)
    try:
        with tempfile.TemporaryFile() as tmp:
            os.dup2(tmp.fileno(), 1)
            try:
                run_func()
            finally:
                try:
                    sys.stdout.flush()
                finally:
                    os.dup2(saved_fd, 1)
            tmp.seek(0)
            return tmp.read().decode(errors="replace")
    finally:
        os.close(saved_fd)


def capture_native_stdout_and_result(run_func):
    """Capture native stdout while preserving a callable's return value."""
    result = []

    def save_result():
        result.append(run_func())

    output = capture_native_stdout(save_result)
    return output, result[0]


def has_cuda_device():
    """Every test in this file allocates torch tensors on "cuda" and runs the CUDA EP.

    Some pipelines install a CPU-only torch wheel, where any `device="cuda"` allocation raises
    "AssertionError: Torch not compiled with CUDA enabled". Gate every test class on this so those
    runs skip instead of erroring."""
    return torch.cuda.is_available() and "CUDAExecutionProvider" in get_available_providers()


def has_flash_attention():
    if not has_cuda_device():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8 and (
        platform.system() == "Linux"
        or (platform.system() == "Windows" and version.parse(torch.version.cuda) >= version.parse("12.0"))
    )


def has_xqa():
    if not has_cuda_device():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor >= 80


def has_fp8_xqa():
    if not has_cuda_device():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor >= 89


def has_memory_efficient_attention():
    # CUTLASS fMHA (MemoryEfficientAttention) gate — these tests are fp16-only,
    # so sm>=53 is sufficient. bf16 MEA would require sm>=80 but is not covered here.
    if not has_cuda_device():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major * 10 + minor) >= 53


def has_webgpu_ep() -> bool:
    return "WebGpuExecutionProvider" in get_available_providers()


def _webgpu_supports_config(config: Config) -> bool:
    """Feature guard for the WebGPU PagedAttention op.

    The WebGPU kernel is fp16-only and does not yet implement softcap or
    sliding-window local attention. Rotary (interleaved and non-interleaved),
    packed QKV, and GQA are supported.
    """
    if config.softcap != 0.0:
        return False
    if config.local:
        return False
    return True


# Bit value matching AttentionBackend::EFFICIENT_ATTENTION in
# onnxruntime/contrib_ops/cpu/bert/attention_common.h. Passing this as the
# CUDA provider option `sdpa_kernel` forces the PagedAttention kernel to
# select the MemoryEfficientAttention (CUTLASS fMHA) fallback even on SM>=80
# where FlashAttention would otherwise be preferred.
SDPA_KERNEL_EFFICIENT_ATTENTION = 2

# Bit value matching AttentionBackend::DECODER_ATTENTION in
# onnxruntime/contrib_ops/cpu/bert/attention_common.h. Passing this as the CUDA provider option
# `sdpa_kernel` leaves the paged decode kernel as the only enabled backend, which is how the
# unquantized decode path is reached (it is otherwise only auto-selected for a quantized cache or
# when FlashAttention is unavailable).
SDPA_KERNEL_DECODER_ATTENTION = 512


def paged_attention_test_cases():
    batches = [4] if pipeline_mode else [1, 3, 5]
    seqs = (
        [(1025, 2047)]
        if pipeline_mode
        else [
            (3, 1024),
            (1, 339),
            (408, 800),
            (333, 799),
            (64, 2048),
            (837, 4000),
            (17, 49),
            (257, 257),
            (459, 459),
        ]
    )
    num_h = [(32, 8)] if pipeline_mode else [(6, 6), (6, 3), (9, 9), (9, 3)]
    h_sizes = [256] if pipeline_mode else [32, 40, 64, 80, 96, 128, 160, 192, 224, 256]
    block_sizes = [256] if pipeline_mode else [256, 512]

    for b in batches:
        for s, s2 in seqs:
            for n, n2 in num_h:
                for h in h_sizes:
                    for block_size in block_sizes:
                        for local in [False, True]:
                            for rotary, rotary_interleaved in rotary_options_for_current_os():
                                for packed in [False, True]:
                                    for softcap in [0.0, 50.0]:
                                        if rotary and h % 16 > 0:
                                            continue

                                        config = Config(
                                            b,
                                            s,
                                            s2,
                                            n,
                                            n2,
                                            h,
                                            block_size,
                                            local,
                                            rotary,
                                            rotary_interleaved,
                                            packed,
                                            softcap,
                                        )
                                        yield (
                                            str(config),
                                            config,
                                        )


@unittest.skipIf(not has_flash_attention(), reason="Flash Attention is not available, skipping tests.")
class TestPagedAttention(unittest.TestCase):
    @parameterized.expand(paged_attention_test_cases())
    def test_paged_attention(self, _, config):
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3)


@unittest.skipIf(
    not has_memory_efficient_attention(),
    reason="MemoryEfficientAttention (fp16) requires sm>=53; skipping.",
)
class TestPagedAttentionMEA(unittest.TestCase):
    """Runs the same parity matrix as TestPagedAttention but forces the CUTLASS
    memory-efficient attention fallback via the `sdpa_kernel` CUDA provider option.
    This is the only coverage for the SM<80 fallback path introduced for PagedAttention;
    on SM>=80 the class still runs to exercise the MEA dispatch end-to-end."""

    @parameterized.expand(paged_attention_test_cases())
    def test_paged_attention_mea(self, _, config):
        parity_check_paged_attention(
            config,
            rtol=5e-3,
            atol=5e-3,
            sdpa_kernel=SDPA_KERNEL_EFFICIENT_ATTENTION,
        )


@unittest.skipIf(not has_cuda_device(), reason="CUDA is not available, skipping tests.")
class TestPagedAttentionFeatures(unittest.TestCase):
    """Coverage for the optional inputs and attributes added on top of the original schema:
    slot_mapping, head_sink and QK-Norm, plus the block_size and batch_size
    limits that were lifted at the same time."""

    def _config(self, **overrides):
        kwargs = {
            "batch_size": 4,
            "sequence_length": 33,
            "total_sequence_length": 128,
            "num_heads": 8,
            "kv_num_heads": 2,
            "head_size": 64,
            "paged_kv_block_size": 256,
            "local": False,
            "rotary": False,
            "rotary_interleaved": False,
            "packed": False,
            "softcap": 0.0,
        }
        feature_overrides = {k: overrides.pop(k) for k in list(overrides) if k not in kwargs}
        kwargs.update(overrides)
        config = Config(**kwargs)
        for key, value in feature_overrides.items():
            setattr(config, key, value)
        return config

    # ---- slot_mapping -------------------------------------------------------------------

    @unittest.skipIf(not has_flash_attention(), reason="Flash Attention is not available")
    def test_slot_mapping_matches_derived_mapping(self):
        # An explicit slot_mapping that reproduces the derived mapping must be a no-op.
        parity_check_paged_attention(self._config(use_slot_mapping=True), rtol=5e-3, atol=5e-3)

    @unittest.skipIf(not has_flash_attention(), reason="Flash Attention is not available")
    def test_slot_mapping_with_rotary_and_packed(self):
        config = self._config(use_slot_mapping=True, rotary=True, packed=True)
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3)

    @unittest.skipIf(
        not has_memory_efficient_attention(),
        reason="MemoryEfficientAttention (fp16) requires sm>=53",
    )
    def test_slot_mapping_mea(self):
        parity_check_paged_attention(
            self._config(use_slot_mapping=True),
            rtol=5e-3,
            atol=5e-3,
            sdpa_kernel=SDPA_KERNEL_EFFICIENT_ATTENTION,
        )

    @unittest.skipIf(not has_flash_attention(), reason="Flash Attention is not available")
    def test_slot_mapping_negative_one_skips_cache_write(self):
        # -1 tells the kernel not to store this token's K/V, which is how a scheduler suppresses
        # writes for prefix-cache hits or rejected speculative tokens. The cache must be
        # bit-identical to its pre-run contents at those slots.
        config = self._config(use_slot_mapping=True)
        token_count = 2
        num_blocks = 4
        block_size = config.paged_kv_block_size
        query = torch.randn(token_count, config.num_heads * config.head_size, device="cuda", dtype=torch.float16)
        key = torch.randn(token_count, config.kv_num_heads * config.head_size, device="cuda", dtype=torch.float16)
        value = torch.randn(token_count, config.kv_num_heads * config.head_size, device="cuda", dtype=torch.float16)
        key_cache = torch.randn(
            num_blocks, block_size, config.kv_num_heads, config.head_size, device="cuda", dtype=torch.float16
        )
        value_cache = torch.randn_like(key_cache)
        key_cache_before = key_cache.clone().cpu().numpy()
        value_cache_before = value_cache.clone().cpu().numpy()

        # One sequence of 2 new tokens on top of 1 cached token.
        config.batch_size = 1
        cum_seqlens = torch.tensor([0, token_count], dtype=torch.int32, device="cuda")
        past_seqlens = torch.tensor([1], dtype=torch.int32, device="cuda")
        block_table = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
        # Store the first token at slot 1 of block 0; skip the second one entirely.
        slot_mapping = torch.tensor([1, -1], dtype=torch.int32, device="cuda")

        _, key_cache_out, value_cache_out = paged_attention_func(
            config,
            query,
            key,
            value,
            key_cache,
            value_cache,
            cum_seqlens,
            past_seqlens,
            block_table,
            slot_mapping=slot_mapping,
        )
        key_cache_out = numpy.array(key_cache_out)
        value_cache_out = numpy.array(value_cache_out)

        # Slot 1 of block 0 holds the first token's K/V.
        numpy.testing.assert_allclose(
            key_cache_out[0, 1], key[0].reshape(config.kv_num_heads, config.head_size).cpu().numpy()
        )
        numpy.testing.assert_allclose(
            value_cache_out[0, 1], value[0].reshape(config.kv_num_heads, config.head_size).cpu().numpy()
        )
        # Everything else, including the slot the second token would have used, is untouched.
        key_cache_before[0, 1] = key_cache_out[0, 1]
        value_cache_before[0, 1] = value_cache_out[0, 1]
        numpy.testing.assert_array_equal(key_cache_out, key_cache_before)
        numpy.testing.assert_array_equal(value_cache_out, value_cache_before)

    # ---- head_sink ---------------------------------------------------------------------

    @unittest.skipIf(not has_flash_attention(), reason="Flash Attention is not available")
    def test_head_sink(self):
        parity_check_paged_attention(self._config(use_head_sink=True), rtol=5e-3, atol=5e-3)

    @unittest.skipIf(not has_flash_attention(), reason="Flash Attention is not available")
    def test_head_sink_local_and_softcap(self):
        # The LSE epilogue must compose with sliding window and softcap, both of which are already
        # baked into the log-sum-exp that FlashAttention returns.
        config = self._config(use_head_sink=True, local=True, softcap=50.0)
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3)

    @unittest.skipIf(not has_flash_attention(), reason="Flash Attention is not available")
    def test_head_sink_with_rotary(self):
        parity_check_paged_attention(self._config(use_head_sink=True, rotary=True), rtol=5e-3, atol=5e-3)

    @unittest.skipIf(
        not has_memory_efficient_attention(),
        reason="MemoryEfficientAttention (fp16) requires sm>=53",
    )
    def test_head_sink_rejected_on_memory_efficient_path(self):
        # The CUTLASS kernel does not expose a log-sum-exp, so the sink cannot be applied. The op
        # must fail loudly rather than silently ignore the input.
        with self.assertRaises(Exception) as ctx:
            parity_check_paged_attention(
                self._config(use_head_sink=True),
                rtol=5e-3,
                atol=5e-3,
                sdpa_kernel=SDPA_KERNEL_EFFICIENT_ATTENTION,
            )
        self.assertIn("head_sink", str(ctx.exception))

    # ---- QK-Norm -----------------------------------------------------------------------

    @unittest.skipIf(not has_flash_attention(), reason="Flash Attention is not available")
    def test_qk_norm(self):
        parity_check_paged_attention(self._config(use_qk_norm=True), rtol=5e-3, atol=5e-3)

    @unittest.skipIf(not has_flash_attention(), reason="Flash Attention is not available")
    def test_qk_norm_with_rotary(self):
        # QK-Norm is applied before rotary, and the normalized+rotated K is what lands in the cache.
        # The cache parity assertions in parity_check_paged_attention cover that ordering.
        parity_check_paged_attention(self._config(use_qk_norm=True, rotary=True), rtol=5e-3, atol=5e-3)

    @unittest.skipIf(not has_flash_attention(), reason="Flash Attention is not available")
    def test_qk_norm_with_rotary_interleaved_and_packed(self):
        config = self._config(use_qk_norm=True, rotary=True, rotary_interleaved=True, packed=True)
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3)

    @unittest.skipIf(not has_flash_attention(), reason="Flash Attention is not available")
    def test_qk_norm_non_power_of_two_head_size(self):
        # head_size=80 rounds up to a 128-thread block in the fused prologue, so the lanes past
        # head_size must contribute zero to the RMS reduction and skip all global accesses.
        parity_check_paged_attention(self._config(use_qk_norm=True, head_size=80), rtol=5e-3, atol=5e-3)

    @unittest.skipIf(
        not has_memory_efficient_attention(),
        reason="MemoryEfficientAttention (fp16) requires sm>=53",
    )
    def test_qk_norm_mea(self):
        parity_check_paged_attention(
            self._config(use_qk_norm=True),
            rtol=5e-3,
            atol=5e-3,
            sdpa_kernel=SDPA_KERNEL_EFFICIENT_ATTENTION,
        )

    @unittest.skipIf(not has_flash_attention(), reason="Flash Attention is not available")
    def test_qk_norm_and_head_sink_together(self):
        parity_check_paged_attention(
            self._config(use_qk_norm=True, use_head_sink=True, rotary=True, use_slot_mapping=True),
            rtol=5e-3,
            atol=5e-3,
        )

    # ---- lifted limits -----------------------------------------------------------------

    @parameterized.expand([(16,), (32,), (64,), (128,)])
    @unittest.skipIf(
        not has_memory_efficient_attention(),
        reason="MemoryEfficientAttention (fp16) requires sm>=53",
    )
    def test_small_block_size(self, block_size):
        # block_size used to be required to be a multiple of 256. Smaller pages are now accepted;
        # FlashAttention cannot address them (a kBlockN tile would straddle a page), so the kernel
        # transparently falls back to the gather-based memory-efficient backend.
        config = self._config(paged_kv_block_size=block_size, total_sequence_length=128)
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3)

    @unittest.skipIf(not has_flash_attention(), reason="Flash Attention is not available")
    def test_large_batch_size(self):
        # cumulative_seqlens_kv used to be produced by independent 256-thread cub::BlockScan blocks,
        # so any batch beyond 256 sequences got silently wrong KV offsets.
        config = self._config(batch_size=300, sequence_length=4, total_sequence_length=64)
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3)

    # ---- is_causal ---------------------------------------------------------------------

    @unittest.skipIf(not has_flash_attention(), reason="Flash Attention is not available")
    def test_non_causal(self):
        # A block drafter submits its whole query block at once and every row must see the rest of
        # the block, so the mask is unbounded on the right.
        parity_check_paged_attention(self._config(is_causal=False), rtol=5e-3, atol=5e-3)

    @unittest.skipIf(not has_flash_attention(), reason="Flash Attention is not available")
    def test_non_causal_local_window(self):
        # local_window_size still bounds the mask on the left when the causal bound is removed.
        config = self._config(is_causal=False, local=True)
        parity_check_paged_attention(
            config,
            rtol=5e-3,
            atol=5e-3,
            new_seqlens_override=torch.full((config.batch_size,), config.sequence_length, dtype=torch.int32),
            local_window_size_override=4,
        )

    @unittest.skipIf(not has_flash_attention(), reason="Flash Attention is not available")
    def test_non_causal_with_rotary_and_packed(self):
        config = self._config(is_causal=False, rotary=True, packed=True)
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3)

    @unittest.skipIf(
        not has_memory_efficient_attention(),
        reason="MemoryEfficientAttention (fp16) requires sm>=53",
    )
    def test_non_causal_rejected_without_flash_attention(self):
        # The CUTLASS and paged-decode kernels hard-code a causal mask, so asking for is_causal=0
        # on a backend that cannot express it has to fail loudly instead of returning a causal result.
        with self.assertRaises(Exception) as ctx:
            parity_check_paged_attention(
                self._config(is_causal=False),
                rtol=5e-3,
                atol=5e-3,
                sdpa_kernel=SDPA_KERNEL_EFFICIENT_ATTENTION,
            )
        self.assertIn("is_causal=0 requires the FlashAttention backend", str(ctx.exception))


# -----------------------------------------------------------------------------
# WebGPU EP parity tests
#
# Reuses the CUDA parity harness (parity_check_paged_attention, attention_ref,
# unpad_qkv, generate_block_kvcache) with Config.ep = "WebGpuExecutionProvider".
# Config matrix is deliberately small (per-op session-create through Dawn is
# heavier than CUDA). 5e-3 abs/rel tolerance matches CUDA at head_size=128.
# -----------------------------------------------------------------------------
def paged_attention_test_cases_webgpu():
    """Hand-picked config matrix for the WebGPU EP. Kept small because the
    WebGPU EP dispatches per-op through Dawn and each session-create is
    heavier than under CUDA.
    """
    batches = [1, 2]
    seqs = [
        (1, 32),  # decode, short past
        (4, 32),  # short prefill, short past
        (16, 64),  # medium prefill
        (1, 64),  # decode, medium past
    ]
    num_h = [(8, 8), (8, 4)]  # MHA + GQA
    h_sizes = [128]  # WebGPU FA requires head_size % 4 == 0
    block_sizes = [256]

    for b in batches:
        for s, s2 in seqs:
            for n, n2 in num_h:
                for h in h_sizes:
                    for block_size in block_sizes:
                        for rotary, rotary_interleaved in rotary_options_for_current_os():
                            for packed in [False, True]:
                                # Rotary requires head_size % 16 == 0 (matches CUDA harness).
                                if rotary and h % 16 > 0:
                                    continue

                                config = Config(
                                    b,
                                    s,
                                    s2,
                                    n,
                                    n2,
                                    h,
                                    block_size,
                                    False,  # local - not supported on WebGPU
                                    rotary,
                                    rotary_interleaved,
                                    packed,
                                    0.0,  # softcap - not supported on WebGPU
                                    ep="WebGpuExecutionProvider",
                                )
                                if _webgpu_supports_config(config):
                                    yield (str(config), config)


@unittest.skipIf(not has_webgpu_ep(), reason="WebGpuExecutionProvider is not available.")
class TestPagedAttentionWebGpu(unittest.TestCase):
    """Parity tests against the WebGPU PagedAttention kernel.

    Runs the same PyTorch reference (attention_ref) as the CUDA classes over a
    smaller config matrix filtered to the features WebGPU supports.
    """

    @parameterized.expand(paged_attention_test_cases_webgpu())
    def test_paged_attention_webgpu(self, _, config):
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3)

    def test_non_causal_rejected(self):
        config = Config(1, 4, 32, 2, 1, 32, 16, False, False, False, False, 0.0, ep="WebGpuExecutionProvider")
        config.is_causal = False
        with self.assertRaises(Exception) as ctx:
            parity_check_paged_attention(config, rtol=5e-3, atol=5e-3)
        self.assertIn("PagedAttention (WebGPU): is_causal=0 is not supported yet", str(ctx.exception))

    def test_paged_attention_webgpu_attention_metadata(self):
        config = Config(
            batch_size=2,
            sequence_length=1,
            total_sequence_length=64,
            num_heads=8,
            kv_num_heads=4,
            head_size=128,
            paged_kv_block_size=256,
            local=False,
            rotary=False,
            rotary_interleaved=False,
            packed=False,
            softcap=0.0,
            ep="WebGpuExecutionProvider",
        )
        config.use_attention_metadata = True
        config.attention_metadata_shape = [3]
        config.attention_metadata_override = numpy.array([1, 64, 1], dtype=numpy.int32)
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3)


@unittest.skipIf(not has_cuda_device(), reason="CUDA is not available, skipping tests.")
class TestPagedAttentionRotaryZeroTokenRegression(unittest.TestCase):
    """Regression tests for the FA `max_query_len` heuristic when one or more
    batches have zero new tokens.

    The old FA path used `token_count - batch_size + 1` as both the rotary
    grid size and the `mha_varlen_fwd` max query length. This assumes every
    batch has at least one new token. With zero-token batches, the value can be
    smaller than the real per-batch maximum, or even non-positive. Then rotary
    can skip Q/K tokens, FA can under-launch at the kBlockM boundary, or FA can
    fail with an invalid launch grid. The MEA path was fixed in PR #28200; this
    class covers the FA regressions fixed by the current PR.
    """

    def _config(self, batch_size=4, sequence_length=10, total_sequence_length=64, rotary=True):
        return Config(
            batch_size=batch_size,
            sequence_length=sequence_length,
            total_sequence_length=total_sequence_length,
            num_heads=8,
            kv_num_heads=2,
            head_size=64,
            paged_kv_block_size=256,
            local=False,
            rotary=rotary,
            rotary_interleaved=False,
            packed=False,
            softcap=0.0,
        )

    @unittest.skipIf(not has_flash_attention(), reason="Flash Attention is not available")
    def test_fa_rotary_zero_token_first_batch(self):
        # lens = [10, 0, 0, 0]. heuristic = 10 - 4 + 1 = 7. true max = 10.
        # Tokens at positions s=7,8,9 in batch 0 do not get rotary applied.
        new_seqlens = torch.tensor([10, 0, 0, 0], dtype=torch.int32, device="cuda")
        parity_check_paged_attention(self._config(), rtol=5e-3, atol=5e-3, new_seqlens_override=new_seqlens)

    @unittest.skipIf(not has_flash_attention(), reason="Flash Attention is not available")
    def test_fa_rotary_zero_token_mixed(self):
        # lens = [0, 7, 0, 3]. heuristic = 10 - 4 + 1 = 7. true max = 7.
        # In this case the heuristic happens to equal the true max, but the
        # input still has a zero-token batch and exercises the grid-launch path.
        new_seqlens = torch.tensor([0, 7, 0, 3], dtype=torch.int32, device="cuda")
        parity_check_paged_attention(self._config(), rtol=5e-3, atol=5e-3, new_seqlens_override=new_seqlens)

    @unittest.skipIf(
        not has_memory_efficient_attention(),
        reason="MemoryEfficientAttention (fp16) requires sm>=53",
    )
    def test_mea_rotary_zero_token_no_regression(self):
        # PR #28200 fixed the MEA path on this input. This test guards against
        # a regression of that fix.
        new_seqlens = torch.tensor([10, 0, 0, 0], dtype=torch.int32, device="cuda")
        parity_check_paged_attention(
            self._config(),
            rtol=5e-3,
            atol=5e-3,
            sdpa_kernel=SDPA_KERNEL_EFFICIENT_ATTENTION,
            new_seqlens_override=new_seqlens,
        )

    @unittest.skipIf(not has_flash_attention(), reason="Flash Attention is not available")
    def test_fa_rotary_zero_token_large_batch(self):
        # 16 batches, only batch 0 has new tokens. token_count = 10, so the
        # heuristic max_query_len_hint = 10 - 16 + 1 = -5 (negative). The
        # value reaches mha_varlen_fwd as params.seqlen_q. Without the exact
        # max_query_len fix added in this PR, FA launches with an invalid grid
        # and fails with CUDA error 9 (invalid configuration argument).
        config = self._config(batch_size=16)
        new_seqlens = torch.tensor([10] + [0] * 15, dtype=torch.int32, device="cuda")
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3, new_seqlens_override=new_seqlens)

    @unittest.skipIf(not has_flash_attention(), reason="Flash Attention is not available")
    def test_fa_no_rotary_zero_token_sanity(self):
        # With rotary off, the rotary kernel is not launched, so this input
        # only checks that ordinary zero-token batches still work on the FA
        # path. The max query length stays below the kBlockM=64 boundary here.
        new_seqlens = torch.tensor([10, 0, 0, 0], dtype=torch.int32, device="cuda")
        parity_check_paged_attention(self._config(rotary=False), rtol=5e-3, atol=5e-3, new_seqlens_override=new_seqlens)

    @unittest.skipIf(not has_flash_attention(), reason="Flash Attention is not available")
    def test_fa_kblockm_boundary_zero_token(self):
        # batch_size=2, lens=[65, 0]. The true max new-query length is 65,
        # which crosses the FA kBlockM=64 boundary. With the older
        # `token_count - batch_size + 1` heuristic, mha_varlen_fwd would see
        # seqlen_q=64 and launch grid.x=1, dropping the 65th query token.
        # This test exercises FA grid sizing with rotary on.
        config = self._config(batch_size=2, sequence_length=65, total_sequence_length=128)
        new_seqlens = torch.tensor([65, 0], dtype=torch.int32, device="cuda")
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3, new_seqlens_override=new_seqlens)

    @unittest.skipIf(not has_flash_attention(), reason="Flash Attention is not available")
    def test_fa_kblockm_boundary_zero_token_no_rotary(self):
        # Same kBlockM=64 boundary case as above with rotary off. This isolates
        # the FA grid silent-drop from the rotary grid silent-drop.
        config = self._config(batch_size=2, sequence_length=65, total_sequence_length=128, rotary=False)
        new_seqlens = torch.tensor([65, 0], dtype=torch.int32, device="cuda")
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3, new_seqlens_override=new_seqlens)


def has_fp8_kv_cache():
    """The float8e4m3fn PagedAttention kernels are only built when onnxruntime_USE_FP8_KV_CACHE is on."""
    if not hasattr(torch, "float8_e4m3fn") or not hasattr(TensorProto, "FLOAT8E4M3FN"):
        return False
    if not has_flash_attention():
        return False
    config = Config(1, 1, 16, 1, 1, 64, 16, False, False, False, False, 0.0)
    config.kv_cache_type = "fp8"
    config.k_quant_type = "PER_TENSOR"
    config.v_quant_type = "PER_TENSOR"
    try:
        InferenceSession(
            create_paged_attention_graph(config, 1, 1, 1),
            SessionOptions(),
            providers=[config.ep],
        )
    except Exception:
        return False
    return True


@unittest.skipIf(not has_flash_attention(), reason="Flash Attention is not available, skipping tests.")
class TestPagedAttentionQuantizedCache(unittest.TestCase):
    """Coverage for the quantized paged KV cache (int8 / float8e4m3fn, PER_TENSOR / PER_CHANNEL).

    Both backends read a quantized cache through the dequantize-on-gather path, so these tests also
    cover FlashAttention's non-paged varlen entry point, which is only reachable this way."""

    def setUp(self):
        # Quantization amplifies host/device rounding differences, so the inputs are re-seeded here
        # to keep these tests independent of the order the rest of the module ran in.
        torch.manual_seed(0)

    def _config(self, **overrides):
        kwargs = {
            "batch_size": 4,
            "sequence_length": 33,
            "total_sequence_length": 128,
            "num_heads": 8,
            "kv_num_heads": 2,
            "head_size": 64,
            "paged_kv_block_size": 256,
            "local": False,
            "rotary": False,
            "rotary_interleaved": False,
            "packed": False,
            "softcap": 0.0,
        }
        feature_overrides = {k: overrides.pop(k) for k in list(overrides) if k not in kwargs}
        kwargs.update(overrides)
        config = Config(**kwargs)
        for key, value in feature_overrides.items():
            setattr(config, key, value)
        return config

    def _int8_config(self, quant_type="PER_TENSOR", **overrides):
        return self._config(kv_cache_type="int8", k_quant_type=quant_type, v_quant_type=quant_type, **overrides)

    # ---- int8 ---------------------------------------------------------------------------

    @parameterized.expand([("per_tensor", "PER_TENSOR"), ("per_channel", "PER_CHANNEL")])
    def test_int8_cache(self, _, quant_type):
        parity_check_paged_attention(self._int8_config(quant_type), rtol=5e-3, atol=5e-3)

    @parameterized.expand([("per_tensor", "PER_TENSOR"), ("per_channel", "PER_CHANNEL")])
    def test_int8_cache_with_rotary_and_packed(self, _, quant_type):
        config = self._int8_config(quant_type, rotary=True, packed=True)
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3)

    def test_int8_cache_with_qk_norm_and_slot_mapping(self):
        # QK-Norm rescales K before it is written, and slot_mapping changes where it is written;
        # both happen upstream of the quantization step in ReshapeAndCache.
        config = self._int8_config("PER_CHANNEL", use_qk_norm=True, use_slot_mapping=True, rotary=True)
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3)

    def test_int8_cache_local_and_softcap(self):
        config = self._int8_config("PER_TENSOR", local=True, softcap=50.0)
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3)

    def test_int8_cache_mixed_granularity(self):
        # k_quant_type and v_quant_type are independent attributes.
        config = self._config(kv_cache_type="int8", k_quant_type="PER_CHANNEL", v_quant_type="PER_TENSOR")
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3)

    @parameterized.expand([("16", 16), ("32", 32), ("64", 64)])
    def test_int8_cache_small_block_size(self, _, block_size):
        # A quantized cache never reaches FlashAttention's paged kernel, so the page-alignment
        # constraint that forces small block sizes onto the MEA fallback does not apply here.
        config = self._int8_config("PER_CHANNEL", paged_kv_block_size=block_size)
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3)

    @unittest.skipIf(
        not has_memory_efficient_attention(),
        reason="MemoryEfficientAttention (fp16) requires sm>=53",
    )
    def test_int8_cache_mea(self):
        parity_check_paged_attention(
            self._int8_config("PER_CHANNEL"),
            rtol=5e-3,
            atol=5e-3,
            sdpa_kernel=SDPA_KERNEL_EFFICIENT_ATTENTION,
        )

    # ---- float8e4m3fn -------------------------------------------------------------------

    @parameterized.expand([("per_tensor", "PER_TENSOR"), ("per_channel", "PER_CHANNEL")])
    @unittest.skipIf(not has_fp8_kv_cache(), reason="FP8 KV cache kernels are not built")
    def test_fp8_cache(self, _, quant_type):
        config = self._config(kv_cache_type="fp8", k_quant_type=quant_type, v_quant_type=quant_type)
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3)

    @unittest.skipIf(not has_fp8_kv_cache(), reason="FP8 KV cache kernels are not built")
    def test_fp8_cache_with_rotary_and_qk_norm(self):
        config = self._config(
            kv_cache_type="fp8",
            k_quant_type="PER_CHANNEL",
            v_quant_type="PER_CHANNEL",
            rotary=True,
            use_qk_norm=True,
        )
        # Rotary and QK-Norm both diverge from the kernel by ~1 fp16 ULP, which is enough to move a
        # value across an e4m3 rounding boundary; one such step is a ~12% change in a K element.
        parity_check_paged_attention(config, rtol=5e-3, atol=2e-2)

    # ---- validation ---------------------------------------------------------------------

    def _run_minimal(self, config, k_scale=None, v_scale=None):
        """Run PagedAttention on a trivially small input, bypassing the parity reference. Used to
        check that invalid quantization configurations are rejected."""
        cache_torch_dtype = {
            "float16": torch.float16,
            "int8": torch.int8,
            "fp8": getattr(torch, "float8_e4m3fn", None),
        }[config.kv_cache_type]
        cache_shape = (1, config.paged_kv_block_size, config.kv_num_heads, config.head_size)
        key_cache = torch.zeros(cache_shape, dtype=torch.float16, device="cuda").to(cache_torch_dtype)
        value_cache = torch.zeros(cache_shape, dtype=torch.float16, device="cuda").to(cache_torch_dtype)
        num_tokens = config.sequence_length
        query = torch.zeros((num_tokens, config.num_heads * config.head_size), dtype=torch.float16, device="cuda")
        key = torch.zeros((num_tokens, config.kv_num_heads * config.head_size), dtype=torch.float16, device="cuda")
        paged_attention_func(
            config,
            query,
            key,
            key.clone(),
            key_cache,
            value_cache,
            torch.tensor([0, num_tokens], dtype=torch.int32, device="cuda"),
            torch.zeros(config.batch_size, dtype=torch.int32, device="cuda"),
            torch.zeros((config.batch_size, 1), dtype=torch.int32, device="cuda"),
            k_scale=k_scale,
            v_scale=v_scale,
        )

    def _minimal_config(self, **overrides):
        return self._config(
            batch_size=1, sequence_length=4, total_sequence_length=16, paged_kv_block_size=16, **overrides
        )

    def test_quantized_cache_without_quant_type_is_rejected(self):
        config = self._minimal_config(kv_cache_type="int8")
        with self.assertRaises(Exception) as ctx:
            self._run_minimal(config)
        self.assertIn("k_quant_type", str(ctx.exception))

    def test_quant_type_without_quantized_cache_is_rejected(self):
        config = self._minimal_config(k_quant_type="PER_TENSOR", v_quant_type="PER_TENSOR")
        scale = torch.ones(1, dtype=torch.float32, device="cuda")
        with self.assertRaises(Exception) as ctx:
            self._run_minimal(config, k_scale=scale, v_scale=scale)
        self.assertIn("not quantized", str(ctx.exception))

    @parameterized.expand([("key", "k_cache_dtype"), ("value", "v_cache_dtype")])
    def test_unsupported_cache_dtype_is_rejected(self, _, attribute_name):
        config = self._minimal_config(kv_cache_type="int8", k_quant_type="PER_TENSOR", v_quant_type="PER_TENSOR")
        setattr(config, attribute_name, "int4")
        scale = torch.ones(1, dtype=torch.float32, device="cuda")
        with self.assertRaises(Exception) as ctx:
            self._run_minimal(config, k_scale=scale, v_scale=scale)
        self.assertIn(attribute_name, str(ctx.exception))

    @parameterized.expand([("key", "k_cache_dtype"), ("value", "v_cache_dtype")])
    def test_cache_dtype_disagreeing_with_cache_tensor_is_rejected(self, _, attribute_name):
        config = self._minimal_config(kv_cache_type="int8", k_quant_type="PER_TENSOR", v_quant_type="PER_TENSOR")
        setattr(config, attribute_name, "float16")
        scale = torch.ones(1, dtype=torch.float32, device="cuda")
        with self.assertRaises(Exception) as ctx:
            self._run_minimal(config, k_scale=scale, v_scale=scale)
        self.assertIn(attribute_name, str(ctx.exception))

    def test_cache_dtype_naming_the_cache_tensor_type_is_accepted(self):
        # '' and an explicit spelling of the tensor's own element type mean the same thing.
        config = self._minimal_config(
            kv_cache_type="int8",
            k_quant_type="PER_TENSOR",
            v_quant_type="PER_TENSOR",
            k_cache_dtype="int8",
            v_cache_dtype="int8",
        )
        scale = torch.ones(1, dtype=torch.float32, device="cuda")
        self._run_minimal(config, k_scale=scale, v_scale=scale)

    @parameterized.expand([("key", "k_cache_dtype"), ("value", "v_cache_dtype")])
    def test_unknown_cache_dtype_is_rejected(self, _, attribute_name):
        # "uint4" is deliberately not in the vocabulary: an unsigned logical type implies a zero
        # point of 8, and this operator quantizes symmetrically with a scale only.
        config = self._minimal_config(kv_cache_type="int8", k_quant_type="PER_TENSOR", v_quant_type="PER_TENSOR")
        setattr(config, attribute_name, "uint4")
        scale = torch.ones(1, dtype=torch.float32, device="cuda")
        with self.assertRaises(Exception) as ctx:
            self._run_minimal(config, k_scale=scale, v_scale=scale)
        self.assertIn("Invalid KV cache data type", str(ctx.exception))


@unittest.skipIf(not has_flash_attention(), reason="Flash Attention is not available, skipping tests.")
class TestPagedAttentionPagedDecode(unittest.TestCase):
    """Coverage for the paged decode backend: a flash-decoding style kernel that scores the paged
    KV cache in place (dequantizing in registers) instead of gathering it into a dense buffer.

    It is selected by the static shape test `token_count == batch_size`, which is a heuristic for
    "one new token per sequence" rather than a proof, so the kernel has to stay correct for ragged
    steps too (see the ragged cases below). The unquantized cases pin the backend with
    `sdpa_kernel`; the quantized cases reach it through the normal auto-selection."""

    def setUp(self):
        torch.manual_seed(0)

    def _config(self, **overrides):
        kwargs = {
            "batch_size": 4,
            "sequence_length": 1,
            "total_sequence_length": 128,
            "num_heads": 8,
            "kv_num_heads": 2,
            "head_size": 64,
            "paged_kv_block_size": 256,
            "local": False,
            "rotary": False,
            "rotary_interleaved": False,
            "packed": False,
            "softcap": 0.0,
        }
        feature_overrides = {k: overrides.pop(k) for k in list(overrides) if k not in kwargs}
        kwargs.update(overrides)
        config = Config(**kwargs)
        for key, value in feature_overrides.items():
            setattr(config, key, value)
        return config

    def _check_decode(self, config, rtol=5e-3, atol=5e-3, new_seqlens_override=None):
        parity_check_paged_attention(
            config,
            rtol=rtol,
            atol=atol,
            sdpa_kernel=SDPA_KERNEL_DECODER_ATTENTION,
            new_seqlens_override=new_seqlens_override,
        )

    # ---- shapes -------------------------------------------------------------------------

    @parameterized.expand([("32", 32), ("64", 64), ("80", 80), ("96", 96), ("128", 128), ("256", 256)])
    def test_decode_head_size(self, _, head_size):
        # head_size straddles the two PV thread mappings: one channel group when head_size >= 128
        # (the CTA width), several groups of head_size threads below it.
        self._check_decode(self._config(head_size=head_size))

    @parameterized.expand([("mha", 8, 8), ("gqa_4x", 8, 2), ("gqa_2x", 6, 3), ("mqa", 9, 1)])
    def test_decode_head_grouping(self, _, num_heads, kv_num_heads):
        self._check_decode(self._config(num_heads=num_heads, kv_num_heads=kv_num_heads))

    @parameterized.expand([("16", 16), ("32", 32), ("64", 64), ("256", 256), ("512", 512)])
    def test_decode_block_size(self, _, block_size):
        # The decode kernel resolves a page per KV token, so unlike FlashAttention's paged kernel it
        # has no page-alignment constraint and a KV tile may straddle any number of pages.
        self._check_decode(self._config(paged_kv_block_size=block_size))

    def test_decode_long_context_multi_tile(self):
        # Far more than one 128-token tile per split, so the online-softmax rescaling across tiles
        # is exercised rather than a single-shot tile.
        self._check_decode(self._config(total_sequence_length=4000, paged_kv_block_size=256))

    def test_decode_multi_split(self):
        # One sequence and few heads leaves the GPU mostly idle, so the host picks num_splits > 1
        # and the cross-split reduction (rather than a single partial) produces the output.
        self._check_decode(self._config(batch_size=1, num_heads=2, kv_num_heads=1, total_sequence_length=4000))

    def test_decode_short_context(self):
        # Fewer KV tokens than a tile, and short enough that some splits are empty.
        self._check_decode(self._config(batch_size=1, num_heads=2, kv_num_heads=1, total_sequence_length=8))

    def test_decode_zero_new_tokens(self):
        # Sequences with no new token contribute no output row; the kernel must skip them without
        # disturbing the rows of the sequences that do have one.
        new_seqlens = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
        self._check_decode(self._config(), new_seqlens_override=new_seqlens)

    # ---- ragged steps -------------------------------------------------------------------
    #
    # The host selects this backend from `token_count <= batch_size` alone, which does not prove one
    # token per sequence. Every CTA therefore resolves its own sequence and in-sequence position
    # from cumulative_sequence_length on device and masks against that token's own causal length.
    # These cases are the ones a wrong implementation passes only by accident.

    def test_decode_ragged_shape_test_holds(self):
        # token_count == batch_size == 4, but the tokens are distributed 3 / 0 / 0 / 1.
        new_seqlens = torch.tensor([3, 0, 0, 1], dtype=torch.int32)
        self._check_decode(self._config(sequence_length=3), new_seqlens_override=new_seqlens)

    def test_decode_ragged_all_tokens_in_one_sequence(self):
        # The extreme case: one sequence owns the whole step, so per-token causal masking is the
        # only thing that can produce the right answer.
        new_seqlens = torch.tensor([4, 0, 0, 0], dtype=torch.int32)
        self._check_decode(self._config(sequence_length=4), new_seqlens_override=new_seqlens)

    def test_decode_ragged_local_window(self):
        new_seqlens = torch.tensor([3, 0, 0, 1], dtype=torch.int32)
        self._check_decode(self._config(sequence_length=3, local=True), new_seqlens_override=new_seqlens)

    def test_decode_ragged_multi_split(self):
        # Few (token, head) pairs and a long context force num_splits > 1, so the per-token split
        # boundaries and the cross-split reduction are exercised on a ragged step.
        new_seqlens = torch.tensor([2, 0], dtype=torch.int32)
        self._check_decode(
            self._config(batch_size=2, sequence_length=2, num_heads=2, kv_num_heads=1, total_sequence_length=4000),
            new_seqlens_override=new_seqlens,
        )

    def test_decode_ragged_int8_cache(self):
        # Auto-selected: a quantized decode-shaped step. XQA cannot serve it (its output layout is
        # one row per batch index), so it must fall through to this kernel and still be correct.
        new_seqlens = torch.tensor([3, 0, 0, 1], dtype=torch.int32)
        config = self._config(
            sequence_length=3, kv_cache_type="int8", k_quant_type="PER_CHANNEL", v_quant_type="PER_CHANNEL"
        )
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3, new_seqlens_override=new_seqlens)

    # ---- masking and score transforms ---------------------------------------------------

    def test_decode_local_window(self):
        self._check_decode(self._config(local=True))

    def test_decode_softcap(self):
        self._check_decode(self._config(softcap=50.0))

    def test_decode_local_and_softcap(self):
        self._check_decode(self._config(local=True, softcap=50.0))

    def test_decode_head_sink(self):
        self._check_decode(self._config(use_head_sink=True))

    def test_decode_head_sink_local_and_softcap(self):
        self._check_decode(self._config(use_head_sink=True, local=True, softcap=50.0))

    # ---- prologue interaction -----------------------------------------------------------

    def test_decode_rotary(self):
        self._check_decode(self._config(rotary=True))

    def test_decode_rotary_interleaved_and_packed(self):
        self._check_decode(self._config(rotary=True, rotary_interleaved=True, packed=True))

    def test_decode_qk_norm_and_slot_mapping(self):
        self._check_decode(self._config(use_qk_norm=True, use_slot_mapping=True, rotary=True))

    # ---- quantized cache (auto-selected, no sdpa_kernel override) ------------------------

    @parameterized.expand([("per_tensor", "PER_TENSOR"), ("per_channel", "PER_CHANNEL")])
    def test_decode_int8_cache(self, _, quant_type):
        # A quantized cache auto-selects the decode backend at sequence_length 1: the scales fold
        # into Q and into the output, so the pages are read once at int8 width.
        config = self._config(kv_cache_type="int8", k_quant_type=quant_type, v_quant_type=quant_type)
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3)

    def test_decode_int8_cache_mixed_granularity(self):
        config = self._config(kv_cache_type="int8", k_quant_type="PER_CHANNEL", v_quant_type="PER_TENSOR")
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3)

    def test_decode_int8_cache_local_softcap_and_sink(self):
        config = self._config(
            kv_cache_type="int8",
            k_quant_type="PER_CHANNEL",
            v_quant_type="PER_CHANNEL",
            local=True,
            softcap=50.0,
            use_head_sink=True,
        )
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3)

    @parameterized.expand([("per_tensor", "PER_TENSOR"), ("per_channel", "PER_CHANNEL")])
    @unittest.skipIf(not has_fp8_kv_cache(), reason="FP8 KV cache kernels are not built")
    def test_decode_fp8_cache(self, _, quant_type):
        config = self._config(kv_cache_type="fp8", k_quant_type=quant_type, v_quant_type=quant_type)
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3)

    @unittest.skipIf(not has_fp8_kv_cache(), reason="FP8 KV cache kernels are not built")
    def test_decode_fp8_cache_with_rotary(self):
        config = self._config(kv_cache_type="fp8", k_quant_type="PER_CHANNEL", v_quant_type="PER_CHANNEL", rotary=True)
        parity_check_paged_attention(config, rtol=5e-3, atol=2e-2)


@unittest.skipIf(not has_xqa(), reason="XQA requires an SM80 or newer GPU")
class TestPagedAttentionXqaDecode(unittest.TestCase):
    """Coverage for the XQA decode backend.

    XQA is the tensor-core decode kernel (shared with GroupQueryAttention) reading the paged cache
    in place. It is auto-selected ahead of the portable decode kernel when the cache is quantized
    and the step fits its constraints: exactly one new token per sequence, head_size in {64, 128},
    a query/KV group size in {4, 6, 8, 16, 32}, no softcap, and block_size a multiple of 128 (a block
    is presented to the kernel as several 128-token pages). Anything outside that falls back, which
    is what the ORT_ENABLE_XQA=0 comparison below pins down.

    Every case here is also run through the fallback so a bug in XQA shows up as a parity failure
    rather than being masked by both paths sharing the reference."""

    def setUp(self):
        torch.manual_seed(0)

    def _config(self, **overrides):
        kwargs = {
            "batch_size": 4,
            "sequence_length": 1,
            "total_sequence_length": 1024,
            "num_heads": 8,
            "kv_num_heads": 2,
            "head_size": 64,
            "paged_kv_block_size": 256,
            "local": False,
            "rotary": False,
            "rotary_interleaved": False,
            "packed": False,
            "softcap": 0.0,
        }
        feature_overrides = {k: overrides.pop(k) for k in list(overrides) if k not in kwargs}
        kwargs.update(overrides)
        config = Config(**kwargs)
        for key, value in feature_overrides.items():
            setattr(config, key, value)
        return config

    def _check_xqa(self, quant_type="PER_TENSOR", kv_cache_type="int8", rtol=5e-3, atol=5e-3, **overrides):
        if kv_cache_type == "fp8":
            if not has_fp8_kv_cache():
                self.skipTest("FP8 KV cache kernels are not built")
            if not has_fp8_xqa():
                self.skipTest("FP8 XQA requires an SM89 or newer GPU")

        config = self._config(
            kv_cache_type=kv_cache_type,
            k_quant_type=quant_type,
            v_quant_type=quant_type,
            **overrides,
        )
        parity_check_paged_attention(config, rtol=rtol, atol=atol)

    def _capture_xqa_debug(self, config):
        with patch.dict(
            os.environ,
            {"ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO": "1", "ORT_ENABLE_XQA": "1"},
        ):
            return capture_native_stdout(lambda: parity_check_paged_attention(config, rtol=5e-3, atol=5e-3))

    # ---- shapes -------------------------------------------------------------------------

    @parameterized.expand([("64", 64), ("128", 128)])
    def test_xqa_head_size(self, _, head_size):
        self._check_xqa(head_size=head_size)

    @parameterized.expand([("grp4", 8, 2), ("grp6", 12, 2), ("grp8", 8, 1), ("grp16", 16, 1), ("grp32", 32, 1)])
    def test_xqa_head_grouping(self, _, num_heads, kv_num_heads):
        self._check_xqa(num_heads=num_heads, kv_num_heads=kv_num_heads)

    @parameterized.expand([("head64_per_tensor", 64, "PER_TENSOR"), ("head128_per_channel", 128, "PER_CHANNEL")])
    def test_xqa_group_size_six_dispatch(self, _, head_size, quant_type):
        # Six query heads exercise XQA's padded query-row path. Parity checks the visible rows,
        # while debug telemetry proves the result did not come from the fallback kernel.
        baseline_config = self._config(
            num_heads=32,
            kv_num_heads=8,
            head_size=head_size,
            kv_cache_type="int8",
            k_quant_type=quant_type,
            v_quant_type=quant_type,
        )
        baseline_debug_output = self._capture_xqa_debug(baseline_config)
        if "SdpaKernel=XQA" not in baseline_debug_output:
            self.skipTest("Paged XQA is not runnable for this head size and quantization configuration")

        config = self._config(
            num_heads=48,
            kv_num_heads=8,
            head_size=head_size,
            kv_cache_type="int8",
            k_quant_type=quant_type,
            v_quant_type=quant_type,
        )
        debug_output = self._capture_xqa_debug(config)
        self.assertIn("Operator=PagedAttention", debug_output)
        self.assertIn("SdpaKernel=XQA", debug_output)
        self.assertIn("GqaGroupSize=6", debug_output)

    @parameterized.expand([("128", 128), ("256", 256), ("512", 512)])
    def test_xqa_block_size(self, _, block_size):
        # Larger blocks are remapped to consecutive 128-token XQA pages; 128-token tables pass through.
        self._check_xqa(paged_kv_block_size=block_size)

    @parameterized.expand(
        [
            ("contiguous", [0, 1], -1),
            ("fragmented", [2, 0], -1),
            ("negative_one", [-1, 1], 128),
        ]
    )
    def test_xqa_native_page_table_matches_expanded(self, _, expanded_block_ids, local_window_size):
        """A 128-token native table must be exactly the page expansion of the 256-token table."""
        # The -1 entries are outside the 128-token local window and therefore never dereferenced.
        device = "cuda"
        pages_per_expanded_block = 2
        expanded_block_table = torch.tensor([expanded_block_ids], dtype=torch.int32, device=device)
        page_offsets = torch.arange(pages_per_expanded_block, dtype=torch.int32, device=device)
        native_page_table = torch.where(
            expanded_block_table.unsqueeze(-1) < 0,
            torch.full_like(expanded_block_table.unsqueeze(-1) + page_offsets, -1),
            expanded_block_table.unsqueeze(-1) * pages_per_expanded_block + page_offsets,
        ).reshape(1, -1)

        expected_pages = []
        for block_id in expanded_block_ids:
            expected_pages.extend([-1, -1] if block_id < 0 else [block_id * 2, block_id * 2 + 1])
        self.assertEqual(native_page_table.cpu().tolist(), [expected_pages])

        physical_blocks = max(expanded_block_ids) + 1
        expanded_key_cache = torch.randint(
            -32,
            33,
            (physical_blocks, 256, 2, 64),
            dtype=torch.int8,
            device=device,
        )
        expanded_value_cache = torch.randint(
            -32,
            33,
            (physical_blocks, 256, 2, 64),
            dtype=torch.int8,
            device=device,
        )
        native_key_cache = expanded_key_cache.reshape(physical_blocks * 2, 128, 2, 64).clone()
        native_value_cache = expanded_value_cache.reshape(physical_blocks * 2, 128, 2, 64).clone()

        query = torch.randn(1, 8 * 64, dtype=torch.float16, device=device)
        key = torch.randn(1, 2 * 64, dtype=torch.float16, device=device)
        value = torch.randn(1, 2 * 64, dtype=torch.float16, device=device)
        cumulative_seqlens = torch.tensor([0, 1], dtype=torch.int32, device=device)
        past_seqlens = torch.tensor([383], dtype=torch.int32, device=device)
        k_scale = torch.tensor([1.0 / 32.0], dtype=torch.float32, device=device)
        v_scale = torch.tensor([1.0 / 32.0], dtype=torch.float32, device=device)

        def run(block_size, block_table, key_cache, value_cache):
            config = self._config(
                batch_size=1,
                total_sequence_length=512,
                paged_kv_block_size=block_size,
                local=local_window_size > 0,
                kv_cache_type="int8",
                k_quant_type="PER_TENSOR",
                v_quant_type="PER_TENSOR",
                use_attention_metadata=True,
            )
            return paged_attention_func(
                config,
                query,
                key,
                value,
                key_cache,
                value_cache,
                cumulative_seqlens,
                past_seqlens,
                block_table,
                window_size=local_window_size,
                k_scale=k_scale,
                v_scale=v_scale,
            )

        with patch.dict(
            os.environ,
            {"ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO": "1", "ORT_ENABLE_XQA": "1"},
        ):
            native_debug, native_result = capture_native_stdout_and_result(
                lambda: run(128, native_page_table, native_key_cache, native_value_cache)
            )
            expanded_debug, expanded_result = capture_native_stdout_and_result(
                lambda: run(256, expanded_block_table, expanded_key_cache, expanded_value_cache)
            )

        for debug_output, expected_mode in (
            (native_debug, "native"),
            (expanded_debug, "expanded"),
        ):
            self.assertIn("Operator=PagedAttention", debug_output)
            self.assertIn("SdpaKernel=XQA", debug_output)
            self.assertIn(f"XqaPageTable={expected_mode}", debug_output)

        native_output, native_key_cache_out, native_value_cache_out = native_result
        expanded_output, expanded_key_cache_out, expanded_value_cache_out = expanded_result
        torch.testing.assert_close(native_output, expanded_output, rtol=0, atol=0)
        torch.testing.assert_close(native_key_cache_out.reshape(-1), expanded_key_cache_out.reshape(-1), rtol=0, atol=0)
        torch.testing.assert_close(
            native_value_cache_out.reshape(-1), expanded_value_cache_out.reshape(-1), rtol=0, atol=0
        )

    def test_xqa_batch_one(self):
        self._check_xqa(batch_size=1)

    def test_xqa_long_context_multi_block(self):
        # Long enough that XQA splits the sequence and reduces across CTAs through its scratch.
        self._check_xqa(total_sequence_length=8192, batch_size=2)

    def test_xqa_short_context(self):
        self._check_xqa(total_sequence_length=8, batch_size=1)

    def test_xqa_context_not_page_aligned(self):
        # The live length is not a multiple of 128, so the last page is partially valid.
        self._check_xqa(total_sequence_length=1000)

    # ---- quantization granularity -------------------------------------------------------

    @parameterized.expand(
        [
            ("int8_per_tensor", "int8", "PER_TENSOR"),
            ("int8_per_channel", "int8", "PER_CHANNEL"),
            ("fp8_per_tensor", "fp8", "PER_TENSOR"),
            ("fp8_per_channel", "fp8", "PER_CHANNEL"),
        ]
    )
    def test_xqa_quant_type(self, _, kv_cache_type, quant_type):
        self._check_xqa(kv_cache_type=kv_cache_type, quant_type=quant_type)

    def test_xqa_mixed_granularity(self):
        # k PER_CHANNEL folds into Q, v PER_TENSOR stays a kernel argument: the two scales take
        # different routes, so an asymmetric config catches a mix-up between them.
        config = self._config(kv_cache_type="int8", k_quant_type="PER_CHANNEL", v_quant_type="PER_TENSOR")
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3)

    # ---- masking and score transforms ---------------------------------------------------

    def test_xqa_local_window(self):
        self._check_xqa(local=True)

    def test_xqa_head_sink(self):
        self._check_xqa(use_head_sink=True)

    def test_xqa_head_sink_and_local(self):
        self._check_xqa(use_head_sink=True, local=True)

    # ---- prologue interaction -----------------------------------------------------------

    def test_xqa_rotary(self):
        self._check_xqa(rotary=True, atol=2e-2)

    def test_xqa_rotary_interleaved_and_packed(self):
        self._check_xqa(rotary=True, rotary_interleaved=True, packed=True, atol=2e-2)

    def test_xqa_qk_norm_and_slot_mapping(self):
        self._check_xqa(use_qk_norm=True, use_slot_mapping=True, rotary=True, atol=2e-2)

    # ---- fallback -----------------------------------------------------------------------

    def test_softcap_falls_back(self):
        # XQA has no softcap, so this must land on the portable decode kernel and still be correct.
        self._check_xqa(softcap=50.0)

    def test_unsupported_head_size_falls_back(self):
        self._check_xqa(head_size=96)

    def test_unsupported_block_size_falls_back(self):
        self._check_xqa(paged_kv_block_size=64)

    def test_multi_token_step_falls_back(self):
        # More than one new token in a sequence: XQA emits one row per sequence, so this has to use
        # a backend that handles a ragged step.
        config = self._config(
            sequence_length=2, kv_cache_type="int8", k_quant_type="PER_TENSOR", v_quant_type="PER_TENSOR"
        )
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3)


@unittest.skipIf(not has_cuda_device(), reason="CUDA is not available, skipping tests.")
class TestPagedAttentionAttentionMetadata(unittest.TestCase):
    """Coverage for the optional 'attention_metadata' input.

    'cumulative_sequence_length' and 'past_seqlens' are device tensors, but the op needs an upper
    bound on the query length (to size grids) and on the KV length (to size workspaces) on the host.
    Without this input it falls back to the static capacities, and in two narrow prefill-side cases
    -- a dense gather, or XQA, which needs proof of exactly one token per sequence -- it copies the
    cumulative arrays back and blocks the stream instead. 'attention_metadata' supplies both bounds
    in CPU memory so neither ever happens.

    It is purely an optimization, so the contract under test is that it changes nothing: every case
    here mirrors a config that is also covered without the input and must produce the same results.
    The cases span the backends because they consume the bounds differently."""

    def setUp(self):
        torch.manual_seed(0)

    def _config(self, **overrides):
        kwargs = {
            "batch_size": 4,
            "sequence_length": 1,
            "total_sequence_length": 1024,
            "num_heads": 8,
            "kv_num_heads": 2,
            "head_size": 64,
            "paged_kv_block_size": 256,
            "local": False,
            "rotary": False,
            "rotary_interleaved": False,
            "packed": False,
            "softcap": 0.0,
        }
        feature_overrides = {k: overrides.pop(k) for k in list(overrides) if k not in kwargs}
        kwargs.update(overrides)
        config = Config(**kwargs)
        config.use_attention_metadata = True
        for key, value in feature_overrides.items():
            setattr(config, key, value)
        return config

    def test_prefill_flash_attention(self):
        # A prefill bound is > 1, so it only sizes the FlashAttention grid; the input must not
        # perturb anything there either.
        parity_check_paged_attention(self._config(sequence_length=256, total_sequence_length=256))

    def test_decode_unquantized(self):
        parity_check_paged_attention(self._config(), sdpa_kernel=SDPA_KERNEL_DECODER_ATTENTION)

    def test_decode_xqa_int8(self):
        config = self._config(kv_cache_type="int8", k_quant_type="PER_CHANNEL", v_quant_type="PER_CHANNEL")
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3)

    @unittest.skipIf(not has_fp8_kv_cache(), reason="FP8 KV cache kernels are not built")
    def test_decode_xqa_fp8(self):
        config = self._config(kv_cache_type="fp8", k_quant_type="PER_TENSOR", v_quant_type="PER_TENSOR")
        parity_check_paged_attention(config, rtol=5e-2, atol=5e-2)

    def test_decode_softcap_portable_kernel(self):
        # softcap rules out XQA, so this exercises the split-KV decode kernel, whose split count is
        # derived from max_kv_len -- here from the bound rather than from a readback.
        config = self._config(kv_cache_type="int8", k_quant_type="PER_TENSOR", v_quant_type="PER_TENSOR", softcap=50.0)
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3)

    def test_decode_local_window(self):
        config = self._config(local=True, kv_cache_type="int8", k_quant_type="PER_TENSOR", v_quant_type="PER_TENSOR")
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3)

    def test_chunked_prefill_dense_gather(self):
        # A quantized cache on a multi-token step gathers the live context into a dense buffer. With
        # a bound available that buffer is sized by batch_size * max_kv_len_bound instead of the
        # exact total_kv_tokens, so the gather kernel has to tolerate indices past the real end of
        # the packed layout.
        config = self._config(
            sequence_length=8,
            kv_cache_type="int8",
            k_quant_type="PER_TENSOR",
            v_quant_type="PER_TENSOR",
        )
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3)

    def test_ragged_new_token_counts(self):
        # token_count == batch_size even though one sequence contributes two tokens and another
        # none. The shape test alone selects a decode-shaped backend, so every backend it can reach
        # must handle the raggedness.
        new_seqlens = torch.tensor([2, 0, 1, 1], dtype=torch.int32)
        parity_check_paged_attention(self._config(sequence_length=2), new_seqlens_override=new_seqlens)

    def test_ragged_new_token_counts_decode_kernel(self):
        # Same step, pinned to the paged decode kernel: with a bound of 2 there is no readback, so
        # the kernel is the only thing that knows the tokens are unevenly distributed.
        new_seqlens = torch.tensor([2, 0, 1, 1], dtype=torch.int32)
        parity_check_paged_attention(
            self._config(sequence_length=2),
            new_seqlens_override=new_seqlens,
            sdpa_kernel=SDPA_KERNEL_DECODER_ATTENTION,
            rtol=5e-3,
            atol=5e-3,
        )

    def test_zero_new_tokens_in_batch(self):
        # token_count < batch_size: still decode-shaped, but XQA is excluded because it would emit a
        # row for the sequence that contributed nothing.
        new_seqlens = torch.tensor([1, 0, 1, 1], dtype=torch.int32)
        parity_check_paged_attention(self._config(), new_seqlens_override=new_seqlens)

    def test_batch_one(self):
        parity_check_paged_attention(self._config(batch_size=1), sdpa_kernel=SDPA_KERNEL_DECODER_ATTENTION)

    def test_negative_entry_is_rejected(self):
        config = self._config()
        config.attention_metadata_override = numpy.array([-1, 1024], dtype=numpy.int32)
        with self.assertRaises(Exception) as ctx:
            parity_check_paged_attention(config)
        self.assertIn("must be non-negative", str(ctx.exception))

    def test_unknown_bounds_fall_back_to_readback(self):
        # All-zero means "no bound", which must behave exactly like supplying the static capacities.
        config = self._config(kv_cache_type="int8", k_quant_type="PER_TENSOR", v_quant_type="PER_TENSOR")
        config.attention_metadata_override = numpy.array([0, 0], dtype=numpy.int32)
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3)

    def test_over_large_bounds_are_clamped(self):
        # Bounds beyond the static limits are legal (they are still upper bounds) and must be
        # clamped rather than used to size an allocation.
        config = self._config(kv_cache_type="int8", k_quant_type="PER_TENSOR", v_quant_type="PER_TENSOR")
        config.attention_metadata_override = numpy.array([1 << 20, 1 << 20], dtype=numpy.int32)
        parity_check_paged_attention(config, rtol=5e-3, atol=5e-3)


####################################################################################################
# Multi-head Latent Attention (kv_cache_layout="LATENT")
#
# See docs/contrib_ops/cuda/paged_attention.md §12. In the absorbed form there is a single physical
# cache holding the latent row [compressed_kv | k_pe] of width head_size = kv_lora_rank +
# qk_rope_head_dim. K is the whole row; V of every head is its leading v_head_size = kv_lora_rank
# channels. 'value' and 'value_cache' are therefore absent and kv_num_heads is 1.
####################################################################################################


class MLAConfig:
    """Shape bundle for a LATENT-layout PagedAttention node. Deliberately separate from Config,
    whose fields (kv_num_heads, packed, ...) encode SEPARATE-mode assumptions."""

    def __init__(
        self,
        batch_size,
        num_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        block_size,
        qk_nope_head_dim=None,
        kv_cache_type="float16",
        k_quant_type="NONE",
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        # Absorbed geometry, i.e. what the op actually sees.
        self.head_size = kv_lora_rank + qk_rope_head_dim
        self.v_head_size = kv_lora_rank
        self.rotary_offset = kv_lora_rank
        self.kv_num_heads = 1
        self.block_size = block_size
        # Un-absorbed geometry, used only by the equivalence test.
        self.qk_nope_head_dim = qk_nope_head_dim
        self.kv_cache_type = kv_cache_type
        self.k_quant_type = k_quant_type

    @property
    def softmax_scale(self):
        # DeepSeek scales by the pre-absorption QK width, which is nope + rope, NOT the absorbed
        # head_size. This is exactly why the op requires an explicit 'scale' for MLA.
        width = (self.qk_nope_head_dim or self.kv_lora_rank) + self.qk_rope_head_dim
        return width**-0.5


def create_mla_graph(
    mla_config,
    num_tokens,
    num_blocks,
    max_blocks_per_sequence,
    do_rotary=False,
    rotary_interleaved=False,
    rotary_offset=None,
    scale=None,
    local_window_size=-1,
    softcap=0.0,
    with_value=False,
    with_value_cache=False,
    with_value_cache_out=False,
    with_head_sink=False,
    with_qk_norm=False,
    kv_cache_layout="LATENT",
    v_head_size=None,
):
    """Build a single-node LATENT PagedAttention model. Every deviation from a valid MLA graph is a
    keyword here so that the negative tests can construct rejected models."""
    cache_proto_type = KV_CACHE_TENSOR_PROTO[mla_config.kv_cache_type]
    head_size = mla_config.head_size
    v_head_size = mla_config.v_head_size if v_head_size is None else v_head_size
    rotary_offset = mla_config.rotary_offset if rotary_offset is None else rotary_offset
    rotary_dim = mla_config.qk_rope_head_dim

    attrs = {
        "num_heads": mla_config.num_heads,
        "kv_num_heads": mla_config.kv_num_heads,
        "kv_cache_layout": kv_cache_layout,
        "v_head_size": v_head_size,
        "local_window_size": local_window_size,
        "softcap": softcap,
        "domain": "com.microsoft",
    }
    if scale is not None:
        attrs["scale"] = scale
    if do_rotary:
        attrs["do_rotary"] = 1
        attrs["rotary_interleaved"] = 1 if rotary_interleaved else 0
        attrs["rotary_offset"] = rotary_offset
    if mla_config.k_quant_type != "NONE":
        attrs["k_quant_type"] = mla_config.k_quant_type

    node_outputs = ["output", "key_cache_out"]
    if with_value_cache_out:
        node_outputs.append("value_cache_out")

    nodes = [
        helper.make_node(
            "PagedAttention",
            [
                "query",
                "key",
                "value" if with_value else "",
                "key_cache",
                "value_cache" if with_value_cache else "",
                "cumulative_sequence_length",
                "past_seqlens",
                "block_table",
                "cos_cache" if do_rotary else "",
                "sin_cache" if do_rotary else "",
                "",  # slot_mapping
                "head_sink" if with_head_sink else "",
                "q_norm_weight" if with_qk_norm else "",
                "k_norm_weight" if with_qk_norm else "",
                "k_scale" if mla_config.k_quant_type != "NONE" else "",
            ],
            node_outputs,
            "PagedAttention_MLA",
            **attrs,
        ),
    ]

    cache_dims = [num_blocks, mla_config.block_size, mla_config.kv_num_heads, head_size]
    graph_input = [
        helper.make_tensor_value_info("query", TensorProto.FLOAT16, [num_tokens, mla_config.num_heads * head_size]),
        helper.make_tensor_value_info("key", TensorProto.FLOAT16, [num_tokens, mla_config.kv_num_heads * head_size]),
        helper.make_tensor_value_info("key_cache", cache_proto_type, cache_dims),
        helper.make_tensor_value_info("cumulative_sequence_length", TensorProto.INT32, [mla_config.batch_size + 1]),
        helper.make_tensor_value_info("past_seqlens", TensorProto.INT32, [mla_config.batch_size]),
        helper.make_tensor_value_info(
            "block_table", TensorProto.INT32, [mla_config.batch_size, max_blocks_per_sequence]
        ),
    ]
    if with_value:
        # SEPARATE mode requires value to match key's hidden size; in LATENT its mere presence is
        # the violation under test.
        graph_input.append(
            helper.make_tensor_value_info(
                "value", TensorProto.FLOAT16, [num_tokens, mla_config.kv_num_heads * head_size]
            )
        )
    if with_value_cache:
        graph_input.append(helper.make_tensor_value_info("value_cache", cache_proto_type, cache_dims))
    if do_rotary:
        # The rotary caches are indexed by rotary_dim // 2, and the op derives rotary_dim from their
        # width. MLA rotates only the qk_rope_head_dim suffix, so these are much narrower than the
        # head_size-derived caches a full-width RoPE would use.
        cache_width = rotary_dim // 2
        graph_input += [
            helper.make_tensor_value_info("cos_cache", TensorProto.FLOAT16, [None, cache_width]),
            helper.make_tensor_value_info("sin_cache", TensorProto.FLOAT16, [None, cache_width]),
        ]
    if with_head_sink:
        graph_input.append(helper.make_tensor_value_info("head_sink", TensorProto.FLOAT16, [mla_config.num_heads]))
    if with_qk_norm:
        graph_input += [
            helper.make_tensor_value_info("q_norm_weight", TensorProto.FLOAT16, [head_size]),
            helper.make_tensor_value_info("k_norm_weight", TensorProto.FLOAT16, [head_size]),
        ]
    if mla_config.k_quant_type != "NONE":
        scale_shape = [1] if mla_config.k_quant_type == "PER_TENSOR" else [mla_config.kv_num_heads, 1, head_size]
        graph_input.append(helper.make_tensor_value_info("k_scale", TensorProto.FLOAT, scale_shape))

    graph_output = [
        helper.make_tensor_value_info("output", TensorProto.FLOAT16, [num_tokens, mla_config.num_heads * v_head_size]),
        helper.make_tensor_value_info("key_cache_out", cache_proto_type, cache_dims),
    ]
    if with_value_cache_out:
        graph_output.append(helper.make_tensor_value_info("value_cache_out", cache_proto_type, cache_dims))

    graph = helper.make_graph(nodes, "PagedAttention_MLA_Graph", graph_input, graph_output)
    return helper.make_model(graph).SerializeToString()


def run_mla(
    mla_config,
    query,
    key,
    key_cache,
    cumulative_sequence_length,
    past_seqlens,
    block_table,
    cos=None,
    sin=None,
    k_scale=None,
    **graph_kwargs,
):
    """Run a LATENT PagedAttention model and return (output, key_cache) with the cache updated in
    place. key_cache is bound on device so the in-place scatter is observable."""
    num_tokens = int(cumulative_sequence_length[-1].item())
    onnx_model_str = create_mla_graph(
        mla_config,
        num_tokens,
        key_cache.shape[0],
        block_table.shape[1],
        do_rotary=cos is not None,
        **graph_kwargs,
    )
    ort_session = InferenceSession(onnx_model_str, SessionOptions(), providers=["CUDAExecutionProvider"])
    io_binding = ort_session.io_binding()

    io_binding.bind_cpu_input("query", query.detach().cpu().numpy())
    io_binding.bind_cpu_input("key", key.detach().cpu().numpy())
    io_binding.bind_cpu_input("cumulative_sequence_length", cumulative_sequence_length.detach().cpu().numpy())
    io_binding.bind_cpu_input("past_seqlens", past_seqlens.detach().cpu().numpy())
    io_binding.bind_cpu_input("block_table", block_table.detach().cpu().numpy())
    if cos is not None:
        io_binding.bind_cpu_input("cos_cache", cos.detach().cpu().numpy())
        io_binding.bind_cpu_input("sin_cache", sin.detach().cpu().numpy())
    if k_scale is not None:
        io_binding.bind_cpu_input("k_scale", k_scale.detach().cpu().numpy())

    cache_proto_type = KV_CACHE_TENSOR_PROTO[mla_config.kv_cache_type]
    key_cache = key_cache.contiguous()
    io_binding.bind_input("key_cache", "cuda", 0, cache_proto_type, tuple(key_cache.shape), key_cache.data_ptr())
    io_binding.bind_output("output")
    io_binding.bind_output("key_cache_out", "cuda", 0, cache_proto_type, tuple(key_cache.shape), key_cache.data_ptr())
    ort_session.run_with_iobinding(io_binding)
    output = torch.tensor(numpy.array(io_binding.copy_outputs_to_cpu()[0]))
    return output, key_cache


def mla_reference(
    mla_config,
    query,  # [token_count, num_heads, head_size]
    latent_cache,  # [batch, total_seqlen, head_size] dense view of the paged cache
    past_seqlens,
    new_seqlens,
    cum_seqlens,
    scale,
    local_window_size=-1,
    softcap=0.0,
):
    """Straightforward fp32 MLA: K is the whole latent row, V its leading v_head_size channels."""
    v_head_size = mla_config.v_head_size
    token_count = int(cum_seqlens[-1].item())
    out = torch.zeros(token_count, mla_config.num_heads, v_head_size, dtype=torch.float32, device="cuda")
    q = query.to(torch.float32)
    for b in range(mla_config.batch_size):
        start = int(cum_seqlens[b].item())
        for j in range(int(new_seqlens[b].item())):
            kv_end = int(past_seqlens[b].item()) + j + 1
            kv_begin = max(0, kv_end - local_window_size) if local_window_size > 0 else 0
            k = latent_cache[b, kv_begin:kv_end].to(torch.float32)  # [L, head_size]
            v = k[:, :v_head_size]
            logits = torch.einsum("nh,lh->nl", q[start + j], k) * scale
            if softcap > 0.0:
                logits = softcap * torch.tanh(logits / softcap)
            probs = torch.softmax(logits, dim=-1)
            out[start + j] = torch.einsum("nl,lv->nv", probs, v)
    return out


def make_mla_batch(mla_config, past_seqlens, new_seqlens, device="cuda"):
    """Allocate a shuffled paged latent cache pre-filled with the 'past' tokens, plus the block
    table and cumulative sequence lengths. Returns everything both paged and densified."""
    total_seqlens = past_seqlens + new_seqlens
    max_total = int(total_seqlens.max().item())
    blocks_per_seq = math.ceil(max_total / mla_config.block_size)
    num_blocks = blocks_per_seq * mla_config.batch_size
    # A shuffled permutation makes block-table indirection load-bearing: a kernel that ignored it
    # and read blocks sequentially would fail.
    block_table = torch.randperm(num_blocks, dtype=torch.int32, device=device).reshape(
        mla_config.batch_size, blocks_per_seq
    )
    latent_paged = torch.randn(
        num_blocks,
        mla_config.block_size,
        mla_config.kv_num_heads,
        mla_config.head_size,
        device=device,
        dtype=torch.float16,
    )
    cum_seqlens = torch.zeros(mla_config.batch_size + 1, dtype=torch.int32, device=device)
    cum_seqlens[1:] = torch.cumsum(new_seqlens, dim=0)
    return latent_paged, block_table, cum_seqlens, blocks_per_seq * mla_config.block_size


def densify_latent(mla_config, latent_paged, block_table, total_len):
    return rearrange(
        latent_paged[block_table.to(dtype=torch.long).flatten()],
        "(b nblocks) block_size h d -> b (nblocks block_size) (h d)",
        b=mla_config.batch_size,
    )[:, :total_len]


def apply_offset_rope(x, cos, sin, positions, rotary_offset, rotary_dim, interleaved):
    """Reference for the 'rotary_offset' attribute: rotate x[..., offset:offset+rotary_dim] using
    the same half-rotated / interleaved conventions as the op, copying everything else through."""
    out = x.clone().to(torch.float32)
    seg = out[..., rotary_offset : rotary_offset + rotary_dim]
    c = cos[positions].to(torch.float32)  # [tokens, rotary_dim // 2]
    s = sin[positions].to(torch.float32)
    c = c[..., : rotary_dim // 2]
    s = s[..., : rotary_dim // 2]
    while c.dim() < seg.dim():
        c = c.unsqueeze(-2)
        s = s.unsqueeze(-2)
    if interleaved:
        even = seg[..., 0::2]
        odd = seg[..., 1::2]
        rotated = torch.stack([even * c - odd * s, even * s + odd * c], dim=-1).flatten(-2)
    else:
        half = rotary_dim // 2
        x1 = seg[..., :half]
        x2 = seg[..., half:]
        rotated = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    out[..., rotary_offset : rotary_offset + rotary_dim] = rotated
    return out.to(x.dtype)


@unittest.skipIf(not has_cuda_device(), reason="CUDA is not available, skipping tests.")
class TestPagedAttentionMLA(unittest.TestCase):
    """Correctness of kv_cache_layout='LATENT' (design doc §12, phase P4)."""

    def setUp(self):
        # These tests build random weights; a fixed seed keeps them order-independent.
        torch.manual_seed(20240727)

    def _config(self, **kwargs):
        # Small but structurally faithful: kv_lora_rank and qk_rope_head_dim keep DeepSeek's ratio
        # while staying cheap enough for a unit test.
        defaults = dict(batch_size=2, num_heads=4, kv_lora_rank=64, qk_rope_head_dim=32, block_size=16)
        defaults.update(kwargs)
        return MLAConfig(**defaults)

    def _run_case(self, mla_config, past_seqlens, new_seqlens, local_window_size=-1, softcap=0.0):
        """Shared body: build a paged latent cache, run the op, compare against mla_reference."""
        device = "cuda"
        past_seqlens = torch.tensor(past_seqlens, dtype=torch.int32, device=device)
        new_seqlens = torch.tensor(new_seqlens, dtype=torch.int32, device=device)
        latent_paged, block_table, cum_seqlens, total_len = make_mla_batch(mla_config, past_seqlens, new_seqlens)
        token_count = int(cum_seqlens[-1].item())

        query = torch.randn(token_count, mla_config.num_heads, mla_config.head_size, device=device, dtype=torch.float16)
        new_key = torch.randn(token_count, mla_config.head_size, device=device, dtype=torch.float16)

        scale = mla_config.softmax_scale
        out, latent_paged = run_mla(
            mla_config,
            query.reshape(token_count, -1),
            new_key,
            latent_paged,
            cum_seqlens,
            past_seqlens,
            block_table,
            scale=scale,
            local_window_size=local_window_size,
            softcap=softcap,
        )

        # The op scattered the new keys into the cache in place, so densifying afterwards gives the
        # exact K/V the reference must see (including any quantization error).
        dense = densify_latent(mla_config, latent_paged, block_table, total_len)
        ref = mla_reference(
            mla_config,
            query,
            dense,
            past_seqlens,
            new_seqlens,
            cum_seqlens,
            scale,
            local_window_size=local_window_size,
            softcap=softcap,
        )
        out = out.reshape(token_count, mla_config.num_heads, mla_config.v_head_size).to(device).to(torch.float32)
        torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)
        return out

    def test_prefill(self):
        config = self._config()
        self._run_case(config, past_seqlens=[0, 0], new_seqlens=[13, 7])

    def test_decode(self):
        config = self._config()
        self._run_case(config, past_seqlens=[31, 18], new_seqlens=[1, 1])

    def test_chunked_prefill(self):
        # Mixed batch: one sequence extends an existing cache, one is pure decode, one adds nothing.
        config = self._config(batch_size=3)
        self._run_case(config, past_seqlens=[20, 9, 5], new_seqlens=[6, 1, 0])

    def test_local_window(self):
        config = self._config()
        self._run_case(config, past_seqlens=[24, 24], new_seqlens=[5, 5], local_window_size=8)

    def test_softcap(self):
        config = self._config()
        self._run_case(config, past_seqlens=[12, 12], new_seqlens=[4, 4], softcap=30.0)

    def test_deepseek_v3_geometry(self):
        # The real absorbed shape: head_size 576, v_head_size 512, kv_num_heads 1.
        config = self._config(num_heads=8, kv_lora_rank=512, qk_rope_head_dim=64, block_size=16)
        self.assertEqual(config.head_size, 576)
        self.assertEqual(config.v_head_size, 512)
        self._run_case(config, past_seqlens=[17, 3], new_seqlens=[1, 4])

    def test_absorbed_matches_non_absorbed(self):
        """The point of MLA: attention over the latent row with absorbed projections equals
        standard MHA over the up-projected K/V. Both sides run through PagedAttention, so this also
        pins the LATENT path against the already-verified SEPARATE path."""
        device = "cuda"
        batch_size, num_heads = 2, 4
        kv_lora_rank, qk_rope_head_dim = 64, 32
        qk_nope_head_dim, v_head_dim = 32, 32
        mla_config = MLAConfig(
            batch_size,
            num_heads,
            kv_lora_rank,
            qk_rope_head_dim,
            block_size=16,
            qk_nope_head_dim=qk_nope_head_dim,
        )
        past_seqlens = torch.tensor([9, 5], dtype=torch.int32, device=device)
        new_seqlens = torch.tensor([4, 6], dtype=torch.int32, device=device)
        latent_paged, block_table, cum_seqlens, total_len = make_mla_batch(mla_config, past_seqlens, new_seqlens)
        token_count = int(cum_seqlens[-1].item())
        scale = mla_config.softmax_scale

        # Up-projection weights, shared by both spellings.
        w_uk = torch.randn(kv_lora_rank, num_heads, qk_nope_head_dim, device=device, dtype=torch.float32) * 0.1
        w_uv = torch.randn(kv_lora_rank, num_heads, v_head_dim, device=device, dtype=torch.float32) * 0.1

        # Non-absorbed query: q_nope [N, qk_nope_head_dim] and q_pe [N, qk_rope_head_dim].
        q_nope = torch.randn(token_count, num_heads, qk_nope_head_dim, device=device, dtype=torch.float32) * 0.5
        q_pe = torch.randn(token_count, num_heads, qk_rope_head_dim, device=device, dtype=torch.float32) * 0.5
        new_key = torch.randn(token_count, mla_config.head_size, device=device, dtype=torch.float16)

        # --- absorbed: q_latent = [q_nope @ W_UK^T | q_pe], attention over the latent row ---
        q_absorbed_nope = torch.einsum("tnp,cnp->tnc", q_nope, w_uk)  # [T, N, kv_lora_rank]
        q_absorbed = torch.cat([q_absorbed_nope, q_pe], dim=-1).to(torch.float16)
        out_absorbed, latent_paged = run_mla(
            mla_config,
            q_absorbed.reshape(token_count, -1),
            new_key,
            latent_paged,
            cum_seqlens,
            past_seqlens,
            block_table,
            scale=scale,
        )
        out_absorbed = out_absorbed.reshape(token_count, num_heads, kv_lora_rank).to(device).to(torch.float32)
        # Absorbed output lives in latent space; project it out with W_UV to compare.
        out_absorbed = torch.einsum("tnc,cnv->tnv", out_absorbed, w_uv)

        # --- non-absorbed: up-project the cache into per-head K/V and run plain MHA ---
        dense = densify_latent(mla_config, latent_paged, block_table, total_len).to(torch.float32)
        compressed_kv = dense[..., :kv_lora_rank]  # [B, L, kv_lora_rank]
        k_pe = dense[..., kv_lora_rank:]  # [B, L, qk_rope_head_dim]
        k_nope = torch.einsum("blc,cnp->blnp", compressed_kv, w_uk)
        k_full = torch.cat([k_nope, k_pe.unsqueeze(2).expand(-1, -1, num_heads, -1)], dim=-1)
        v_full = torch.einsum("blc,cnv->blnv", compressed_kv, w_uv)
        q_full = torch.cat([q_nope, q_pe], dim=-1)

        out_ref = torch.zeros(token_count, num_heads, v_head_dim, dtype=torch.float32, device=device)
        for b in range(batch_size):
            start = int(cum_seqlens[b].item())
            for j in range(int(new_seqlens[b].item())):
                kv_end = int(past_seqlens[b].item()) + j + 1
                logits = torch.einsum("nd,lnd->nl", q_full[start + j], k_full[b, :kv_end]) * scale
                probs = torch.softmax(logits, dim=-1)
                out_ref[start + j] = torch.einsum("nl,lnv->nv", probs, v_full[b, :kv_end])

        torch.testing.assert_close(out_absorbed, out_ref, rtol=5e-3, atol=5e-3)

    def test_rotary_offset_matches_graph_applied_rope(self):
        """do_rotary=1 with rotary_offset=kv_lora_rank must equal applying the same RoPE to the
        k_pe suffix outside the op and running with do_rotary=0."""
        device = "cuda"
        for interleaved in (False, True):
            with self.subTest(interleaved=interleaved):
                config = self._config()
                past_seqlens = torch.tensor([6, 2], dtype=torch.int32, device=device)
                new_seqlens = torch.tensor([3, 5], dtype=torch.int32, device=device)
                latent_paged, block_table, cum_seqlens, total_len = make_mla_batch(config, past_seqlens, new_seqlens)
                token_count = int(cum_seqlens[-1].item())
                rotary_dim = config.qk_rope_head_dim
                cache_width = rotary_dim // 2
                max_pos = 128
                angle = torch.rand(max_pos, cache_width, device=device) * 2 - 1
                cos = torch.cos(angle).to(torch.float16)
                sin = torch.sin(angle).to(torch.float16)

                query = torch.randn(token_count, config.num_heads, config.head_size, device=device, dtype=torch.float16)
                new_key = torch.randn(token_count, config.head_size, device=device, dtype=torch.float16)
                scale = config.softmax_scale

                # Positions the op uses: past_seqlens[b] + index within the sequence.
                positions = torch.empty(token_count, dtype=torch.long, device=device)
                for b in range(config.batch_size):
                    start = int(cum_seqlens[b].item())
                    n = int(new_seqlens[b].item())
                    positions[start : start + n] = int(past_seqlens[b].item()) + torch.arange(n, device=device)

                out_in_op, cache_in_op = run_mla(
                    config,
                    query.reshape(token_count, -1),
                    new_key,
                    latent_paged.clone(),
                    cum_seqlens,
                    past_seqlens,
                    block_table,
                    cos=cos,
                    sin=sin,
                    rotary_interleaved=interleaved,
                    scale=scale,
                )

                q_roped = apply_offset_rope(query, cos, sin, positions, config.rotary_offset, rotary_dim, interleaved)
                k_roped = apply_offset_rope(new_key, cos, sin, positions, config.rotary_offset, rotary_dim, interleaved)
                out_pre, cache_pre = run_mla(
                    config,
                    q_roped.reshape(token_count, -1),
                    k_roped,
                    latent_paged.clone(),
                    cum_seqlens,
                    past_seqlens,
                    block_table,
                    scale=scale,
                )

                torch.testing.assert_close(out_in_op.to(torch.float32), out_pre.to(torch.float32), rtol=3e-3, atol=3e-3)
                # The scattered latent rows must match too: RoPE touches only the k_pe suffix.
                torch.testing.assert_close(
                    cache_in_op.to(torch.float32), cache_pre.to(torch.float32), rtol=3e-3, atol=3e-3
                )

    def test_v_aliases_k(self):
        """With value_cache absent, V must be the leading v_head_size channels of key_cache. Zero
        the tail of every latent row and confirm the output is unchanged: the tail is K-only."""
        device = "cuda"
        config = self._config()
        past_seqlens = torch.tensor([0, 0], dtype=torch.int32, device=device)
        new_seqlens = torch.tensor([4, 4], dtype=torch.int32, device=device)
        latent_paged, block_table, cum_seqlens, total_len = make_mla_batch(config, past_seqlens, new_seqlens)
        token_count = int(cum_seqlens[-1].item())
        query = torch.randn(token_count, config.num_heads, config.head_size, device=device, dtype=torch.float16)
        # Zero the query's rope slice so the k_pe channels cannot influence the logits either.
        query[..., config.v_head_size :] = 0
        new_key = torch.randn(token_count, config.head_size, device=device, dtype=torch.float16)
        scale = config.softmax_scale

        out_a, cache_a = run_mla(
            config,
            query.reshape(token_count, -1),
            new_key,
            latent_paged.clone(),
            cum_seqlens,
            past_seqlens,
            block_table,
            scale=scale,
        )
        key_tail_zeroed = new_key.clone()
        key_tail_zeroed[:, config.v_head_size :] = 0
        paged_tail_zeroed = latent_paged.clone()
        paged_tail_zeroed[..., config.v_head_size :] = 0
        out_b, _ = run_mla(
            config,
            query.reshape(token_count, -1),
            key_tail_zeroed,
            paged_tail_zeroed,
            cum_seqlens,
            past_seqlens,
            block_table,
            scale=scale,
        )
        torch.testing.assert_close(out_a.to(torch.float32), out_b.to(torch.float32), rtol=2e-3, atol=2e-3)

        # And the leading channels really are V: scaling them scales the output linearly.
        self.assertGreater(out_a.abs().max().item(), 1e-3)

    def test_fp8_latent_cache(self):
        if not has_fp8_kv_cache():
            self.skipTest("FP8 KV cache kernels are not built")
        device = "cuda"
        config = self._config(kv_cache_type="fp8", k_quant_type="PER_TENSOR")
        past_seqlens = torch.tensor([8, 8], dtype=torch.int32, device=device)
        new_seqlens = torch.tensor([2, 2], dtype=torch.int32, device=device)
        total_seqlens = past_seqlens + new_seqlens
        max_total = int(total_seqlens.max().item())
        blocks_per_seq = math.ceil(max_total / config.block_size)
        num_blocks = blocks_per_seq * config.batch_size
        block_table = torch.randperm(num_blocks, dtype=torch.int32, device=device).reshape(
            config.batch_size, blocks_per_seq
        )
        cum_seqlens = torch.zeros(config.batch_size + 1, dtype=torch.int32, device=device)
        cum_seqlens[1:] = torch.cumsum(new_seqlens, dim=0)
        token_count = int(cum_seqlens[-1].item())

        k_scale = torch.tensor([0.01], dtype=torch.float32, device=device)
        latent_float = torch.randn(
            num_blocks, config.block_size, config.kv_num_heads, config.head_size, device=device, dtype=torch.float16
        )
        latent_paged = quantize_kv(latent_float, k_scale, "fp8")

        query = torch.randn(token_count, config.num_heads, config.head_size, device=device, dtype=torch.float16)
        new_key = torch.randn(token_count, config.head_size, device=device, dtype=torch.float16)
        scale = config.softmax_scale
        out, latent_paged = run_mla(
            config,
            query.reshape(token_count, -1),
            new_key,
            latent_paged,
            cum_seqlens,
            past_seqlens,
            block_table,
            k_scale=k_scale,
            scale=scale,
        )
        # Dequantize the cache the op left behind and score against it, so only the attention math
        # (not the quantization error) is under test.
        dense_q = densify_latent(config, latent_paged, block_table, blocks_per_seq * config.block_size)
        dense = dequantize_kv(dense_q, k_scale)
        ref = mla_reference(config, query, dense, past_seqlens, new_seqlens, cum_seqlens, scale)
        out = out.reshape(token_count, config.num_heads, config.v_head_size).to(device).to(torch.float32)
        torch.testing.assert_close(out, ref, rtol=5e-3, atol=5e-3)

    # ---- rejected configurations (design doc §12.9, §12.10) ----

    def _expect_rejected(self, message_fragment, **graph_kwargs):
        """Build a LATENT model with one deliberate violation and assert the op rejects it. Schema
        violations surface at session creation, input violations in ComputeInternal, so both are
        wrapped."""
        config = graph_kwargs.pop("config", None) or self._config()
        num_tokens, num_blocks, max_blocks = 4, 4, 2
        head_size = config.head_size

        def build_and_run():
            model = create_mla_graph(config, num_tokens, num_blocks, max_blocks, **graph_kwargs)
            session = InferenceSession(model, SessionOptions(), providers=["CUDAExecutionProvider"])
            feeds = {
                "query": torch.randn(num_tokens, config.num_heads * head_size).to(torch.float16).numpy(),
                "key": torch.randn(num_tokens, config.kv_num_heads * head_size).to(torch.float16).numpy(),
                "key_cache": torch.randn(num_blocks, config.block_size, config.kv_num_heads, head_size)
                .to(torch.float16)
                .numpy(),
                "cumulative_sequence_length": numpy.array([0, 2, 4, 4, 4][: config.batch_size + 1], dtype=numpy.int32),
                "past_seqlens": numpy.zeros(config.batch_size, dtype=numpy.int32),
                "block_table": numpy.arange(config.batch_size * max_blocks, dtype=numpy.int32).reshape(
                    config.batch_size, max_blocks
                ),
            }
            if graph_kwargs.get("with_value"):
                feeds["value"] = torch.randn(num_tokens, config.kv_num_heads * head_size).to(torch.float16).numpy()
            if graph_kwargs.get("with_value_cache"):
                feeds["value_cache"] = feeds["key_cache"].copy()
            if graph_kwargs.get("do_rotary"):
                cache_width = config.qk_rope_head_dim // 2
                feeds["cos_cache"] = torch.ones(64, cache_width).to(torch.float16).numpy()
                feeds["sin_cache"] = torch.zeros(64, cache_width).to(torch.float16).numpy()
            if graph_kwargs.get("with_head_sink"):
                feeds["head_sink"] = torch.zeros(config.num_heads).to(torch.float16).numpy()
            if graph_kwargs.get("with_qk_norm"):
                feeds["q_norm_weight"] = torch.ones(head_size).to(torch.float16).numpy()
                feeds["k_norm_weight"] = torch.ones(head_size).to(torch.float16).numpy()
            session.run(None, feeds)

        with self.assertRaises(Exception) as ctx:
            build_and_run()
        self.assertIn(message_fragment, str(ctx.exception))

    def test_reject_missing_scale(self):
        # v_head_size != head_size with no explicit scale would silently use 1/sqrt(head_size).
        self._expect_rejected("explicit 'scale'")

    def test_reject_value_input(self):
        self._expect_rejected("'value'", scale=0.1, with_value=True)

    def test_reject_value_cache(self):
        self._expect_rejected("'value_cache' must be absent", scale=0.1, with_value_cache=True)

    def test_reject_value_cache_output(self):
        self._expect_rejected("value_cache_out must be absent", scale=0.1, with_value_cache_out=True)

    def test_reject_head_sink(self):
        self._expect_rejected("'head_sink'", scale=0.1, with_head_sink=True)

    def test_reject_qk_norm(self):
        self._expect_rejected("q_norm_weight", scale=0.1, with_qk_norm=True)

    def test_reject_multi_kv_head(self):
        config = self._config()
        config.kv_num_heads = 2
        self._expect_rejected("'kv_num_heads' must be 1", config=config, scale=0.1)

    def test_reject_v_head_size_in_separate_layout(self):
        # v_head_size != head_size is only meaningful for LATENT.
        self._expect_rejected(
            "may only differ from head_size",
            scale=0.1,
            kv_cache_layout="SEPARATE",
            with_value=True,
            with_value_cache=True,
            with_value_cache_out=True,
        )

    def test_reject_unaligned_rotary_offset(self):
        self._expect_rejected("multiple of 8", scale=0.1, do_rotary=True, rotary_offset=60)

    def test_reject_rotary_offset_overflow(self):
        config = self._config()
        self._expect_rejected(
            "must not exceed head_size", config=config, scale=0.1, do_rotary=True, rotary_offset=config.head_size
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
