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
import platform
import random
import unittest

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
    k_cache_bit_width = 8
    v_cache_bit_width = 8

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
    quant_attrs = (
        {
            "k_quant_type": config.k_quant_type,
            "v_quant_type": config.v_quant_type,
            "k_cache_bit_width": config.k_cache_bit_width,
            "v_cache_bit_width": config.v_cache_bit_width,
        }
        if (has_k_scale or has_v_scale or config.kv_cache_type != "float16")
        else {}
    )
    nodes = [
        helper.make_node(
            "PagedAttention",
            [
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
                "slot_mapping" if config.use_slot_mapping else "",
                "head_sink" if config.use_head_sink else "",
                "q_norm_weight" if config.use_qk_norm else "",
                "k_norm_weight" if config.use_qk_norm else "",
                "k_scale" if has_k_scale else "",
                "v_scale" if has_v_scale else "",
            ],
            ["output", "key_cache_out", "value_cache_out"],
            "PagedAttention_0",
            num_heads=config.num_heads,
            kv_num_heads=config.kv_num_heads,
            local_window_size=local_window_size,
            do_rotary=config.rotary,
            rotary_interleaved=config.rotary_interleaved,
            softcap=config.softcap,
            qk_norm_epsilon=config.qk_norm_epsilon,
            domain="com.microsoft",
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

    model = helper.make_model(graph)
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
    if not quantized:
        ort_inputs["key_cache"] = OrtValue.ortvalue_from_numpy(key_cache.detach().cpu().numpy(), "cuda", 0)
        ort_inputs["value_cache"] = OrtValue.ortvalue_from_numpy(value_cache.detach().cpu().numpy(), "cuda", 0)
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
    io_binding.bind_cpu_input("query", ort_inputs["query"])
    if quantized:
        # A quantized cache has no numpy dtype, so bind the torch device buffers directly.
        cache_proto_type = KV_CACHE_TENSOR_PROTO[config.kv_cache_type]
        key_cache = key_cache.contiguous()
        value_cache = value_cache.contiguous()
        io_binding.bind_input("key_cache", "cuda", 0, cache_proto_type, tuple(key_cache.shape), key_cache.data_ptr())
        io_binding.bind_input(
            "value_cache", "cuda", 0, cache_proto_type, tuple(value_cache.shape), value_cache.data_ptr()
        )
    else:
        io_binding.bind_input(
            "key_cache", "cuda", 0, numpy.float16, ort_inputs["key_cache"].shape(), ort_inputs["key_cache"].data_ptr()
        )
        io_binding.bind_input(
            "value_cache",
            "cuda",
            0,
            numpy.float16,
            ort_inputs["value_cache"].shape(),
            ort_inputs["value_cache"].data_ptr(),
        )
    io_binding.bind_cpu_input("cumulative_sequence_length", ort_inputs["cumulative_sequence_length"])
    io_binding.bind_cpu_input("past_seqlens", ort_inputs["past_seqlens"])
    io_binding.bind_cpu_input("block_table", ort_inputs["block_table"])
    io_binding.bind_output("output")
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
    ort_session.run_with_iobinding(io_binding)
    if quantized:
        output = torch.tensor(numpy.array(io_binding.copy_outputs_to_cpu()[0]))
        return output, key_cache, value_cache
    output, key_cache_out, value_cache_out = io_binding.copy_outputs_to_cpu()
    output = torch.tensor(numpy.array(output))
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
        device="cuda",
    )
    k_unpad = torch.zeros(
        token_count,
        config.kv_num_heads * config.head_size,
        dtype=torch.float16,
        device="cuda",
    )
    v_unpad = torch.zeros(
        token_count,
        config.kv_num_heads * config.head_size,
        dtype=torch.float16,
        device="cuda",
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
):
    # Generate padded inputs
    q = torch.randn(
        config.batch_size,
        config.sequence_length,
        config.num_heads,
        config.head_size,
        device="cuda",
        dtype=torch.float16,
        requires_grad=False,
    )
    k_new = torch.randn(
        config.batch_size,
        config.sequence_length,
        config.kv_num_heads,
        config.head_size,
        device="cuda",
        dtype=torch.float16,
        requires_grad=False,
    )
    v_new = torch.randn(
        config.batch_size,
        config.sequence_length,
        config.kv_num_heads,
        config.head_size,
        device="cuda",
        dtype=torch.float16,
        requires_grad=False,
    )

    # Generate random sequence lengths
    past_seqlens = torch.randint(
        0,
        config.total_sequence_length - config.sequence_length + 1,  # one above highest integer to be drawn
        (config.batch_size,),
        dtype=torch.int32,
        device="cuda",
    )
    if new_seqlens_override is not None:
        new_seqlens = new_seqlens_override.to(dtype=torch.int32, device="cuda")
        assert new_seqlens.shape == (config.batch_size,)
        assert int(new_seqlens.min().item()) >= 0
        assert int(new_seqlens.max().item()) <= config.sequence_length
    else:
        new_seqlens = torch.randint(
            1,
            config.sequence_length + 1,
            (config.batch_size,),
            dtype=torch.int32,
            device="cuda",
        )
    cum_seqlens = torch.cat(
        (torch.tensor([0], dtype=torch.int32, device="cuda"), torch.cumsum(new_seqlens, dim=0))
    ).type(torch.int32)
    total_seqlens = past_seqlens + new_seqlens

    q_unpad, k_unpad, v_unpad = unpad_qkv(config, q, k_new, v_new, cum_seqlens)

    # Generate kv cache and associated block-based data structures
    k_cache, v_cache, block_table, k_cache_paged, v_cache_paged = generate_block_kvcache(config, "cuda", torch.float16)

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
    window_size = (-1, -1)
    left_window_size = -1
    if config.local:
        left_window_size = random.randint(0, config.total_sequence_length - 1)  # random.randint is inclusive
        window_size = (left_window_size, 0)
    else:
        left_window_size = -1
        window_size = (-1, 0)

    # Apply rotary embedding for reference implementation
    if config.rotary:
        rotary_fraction = 1.0
        rotary_dim = math.floor(int(rotary_fraction * config.head_size) / 16) * 16
        angle = torch.rand(config.total_sequence_length, rotary_dim // 2, device="cuda") * 2 * math.pi
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
    total_range = rearrange(torch.arange(config.total_sequence_length, device="cuda"), "s -> 1 s")
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
    query_range = rearrange(torch.arange(config.sequence_length, device="cuda"), "s -> 1 s")
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
        causal=True,
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


def has_flash_attention():
    if not torch.cuda.is_available():
        return False
    if "CUDAExecutionProvider" not in get_available_providers():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8 and (
        platform.system() == "Linux"
        or (platform.system() == "Windows" and version.parse(torch.version.cuda) >= version.parse("12.0"))
    )


def has_memory_efficient_attention():
    # CUTLASS fMHA (MemoryEfficientAttention) gate — these tests are fp16-only,
    # so sm>=53 is sufficient. bf16 MEA would require sm>=80 but is not covered here.
    if not torch.cuda.is_available():
        return False
    if "CUDAExecutionProvider" not in get_available_providers():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major * 10 + minor) >= 53


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

    @parameterized.expand([("key", "k_cache_bit_width"), ("value", "v_cache_bit_width")])
    def test_invalid_cache_bit_width_is_rejected(self, _, attribute_name):
        config = self._minimal_config(kv_cache_type="int8", k_quant_type="PER_TENSOR", v_quant_type="PER_TENSOR")
        setattr(config, attribute_name, 4)
        scale = torch.ones(1, dtype=torch.float32, device="cuda")
        with self.assertRaises(Exception) as ctx:
            self._run_minimal(config, k_scale=scale, v_scale=scale)
        self.assertIn(attribute_name, str(ctx.exception))


@unittest.skipIf(not has_flash_attention(), reason="Flash Attention is not available, skipping tests.")
class TestPagedAttentionPagedDecode(unittest.TestCase):
    """Coverage for the paged decode backend: a flash-decoding style kernel that scores the paged
    KV cache in place (dequantizing in registers) instead of gathering it into a dense buffer.

    It only handles steps where every sequence contributes at most one new token, so every config
    here uses sequence_length=1. The unquantized cases pin the backend with `sdpa_kernel`; the
    quantized cases reach it through the normal auto-selection."""

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
