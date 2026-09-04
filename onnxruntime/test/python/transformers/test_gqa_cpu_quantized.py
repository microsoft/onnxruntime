# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for CPU GroupQueryAttention with quantized KV cache (INT8/INT4)."""

import math
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper

from onnxruntime import InferenceSession, SessionOptions
from onnxruntime.capi.onnxruntime_pybind11_state import Fail

# Whether to run the full matrix of tests or a subset for CI.
pipeline_mode = True


# ---- Quantization helpers ----


def quantize_int8_per_tensor(data_fp32):
    """Quantize float32 BNSH data to int8 per-tensor. Returns (quantized_int8, scale)."""
    amax = np.max(np.abs(data_fp32))
    scale = float(amax / 127.0) if amax > 1e-6 else 1.0
    quantized = np.clip(np.round(data_fp32 / scale), -128, 127).astype(np.int8)
    return quantized, np.array([scale], dtype=np.float32)


def dequantize_int8_per_tensor(quantized_int8, scale):
    """Dequantize int8 per-tensor back to float32."""
    return quantized_int8.astype(np.float32) * scale


def quantize_int8_per_channel(data_fp32):
    """Quantize float32 BNSH data to int8 per-channel. Returns (quantized_int8, scale).
    Scale shape: [kv_num_heads * head_size] (flat over N*H dims).
    """
    _, n, _, h = data_fp32.shape
    # Per-channel: one scale per (n, h) channel across all batch and seq positions
    reshaped = data_fp32.transpose(0, 2, 1, 3).reshape(-1, n * h)  # [B*S, N*H]
    amax = np.max(np.abs(reshaped), axis=0)  # [N*H]
    scale = np.where(amax > 1e-6, amax / 127.0, 1.0).astype(np.float32)  # [N*H]
    quantized = np.clip(np.round(data_fp32 / scale.reshape(1, n, 1, h)), -128, 127).astype(np.int8)
    return quantized, scale


def dequantize_int8_per_channel(quantized_int8, scale, kv_num_heads, head_size):
    """Dequantize int8 per-channel back to float32."""
    _, n, _, h = quantized_int8.shape
    return quantized_int8.astype(np.float32) * scale.reshape(1, n, 1, h)


def pack_int4(data_int8):
    """Pack int8 values into int4 format (2 per byte). data_int8 must have even last dim."""
    assert data_int8.shape[-1] % 2 == 0
    even = (data_int8[..., 0::2].astype(np.int16) + 8) & 0x0F
    odd = (data_int8[..., 1::2].astype(np.int16) + 8) & 0x0F
    packed = (even | (odd << 4)).astype(np.uint8)
    return packed


def unpack_int4(packed_uint8):
    """Unpack int4 packed format to int8 values."""
    even = (packed_uint8.astype(np.int16) & 0x0F) - 8
    odd = (packed_uint8.astype(np.int16) >> 4) - 8
    shape = list(packed_uint8.shape)
    shape[-1] *= 2
    unpacked = np.empty(shape, dtype=np.int8)
    unpacked[..., 0::2] = even.astype(np.int8)
    unpacked[..., 1::2] = odd.astype(np.int8)
    return unpacked


def quantize_int4_per_tensor(data_fp32):
    """Quantize float32 to int4 per-tensor. Returns (packed_uint8, scale)."""
    amax = np.max(np.abs(data_fp32))
    scale = float(amax / 7.0) if amax > 1e-6 else 1.0
    quantized = np.clip(np.round(data_fp32 / scale), -8, 7).astype(np.int8)
    packed = pack_int4(quantized)
    return packed, np.array([scale], dtype=np.float32)


def dequantize_int4_per_tensor(packed_uint8, scale):
    """Dequantize int4 per-tensor back to float32."""
    unpacked = unpack_int4(packed_uint8)
    return unpacked.astype(np.float32) * scale


def quantize_int4_per_channel(data_fp32):
    """Quantize float32 BNSH to int4 per-channel. Returns (packed_uint8, scale)."""
    _, n, _, h = data_fp32.shape
    reshaped = data_fp32.transpose(0, 2, 1, 3).reshape(-1, n * h)
    amax = np.max(np.abs(reshaped), axis=0)
    scale = np.where(amax > 1e-6, amax / 7.0, 1.0).astype(np.float32)
    quantized = np.clip(np.round(data_fp32 / scale.reshape(1, n, 1, h)), -8, 7).astype(np.int8)
    packed = pack_int4(quantized)
    return packed, scale


def dequantize_int4_per_channel(packed_uint8, scale, kv_num_heads, head_size):
    """Dequantize int4 per-channel back to float32."""
    unpacked = unpack_int4(packed_uint8)
    return unpacked.astype(np.float32) * scale.reshape(1, kv_num_heads, 1, head_size)


# ---- OSCAR 2-bit PER_GROUP quantization helpers ----
#
# These replicate the CPU kernel codec in
# contrib_ops/cpu/bert/gqa_attention_base.h (Oscar2BitQuantizeRow /
# Oscar2BitDequantizeRow). The 2-bit path stores scales and zero-points inline in
# each packed cache row instead of via separate k_scale/v_scale inputs, so the
# packed head dimension is  head_size/4 code bytes + num_groups*(scale,zero) fp32.

OSCAR2BIT_Q_MAX = 3


def oscar2bit_packed_head_size(head_size, group_size, meta_fp16=False):
    """Packed last-dim (uint8 count) of a 2-bit PER_GROUP KV cache row."""
    num_groups = head_size // group_size
    meta_bytes = 2 if meta_fp16 else 4
    return head_size // 4 + num_groups * 2 * meta_bytes


def _oscar2bit_clip_threshold(row_vals, rho):
    """OSCAR per-row clip threshold: sort |x| over the full row and pick the value at the
    discrete index int(rho * n), clamped to [0, n - 1]. rho <= 0 disables the clip.
    Matches _clip_index / _ref_threshold in the OSCAR reference (no interpolation)."""
    if rho <= 0.0:
        return np.inf
    a = np.sort(np.abs(row_vals.astype(np.float32)))
    n = a.shape[0]
    idx = int(rho * n)
    if idx >= n:
        idx = n - 1
    if idx < 0:
        idx = 0
    return float(a[idx])


def quantize_dequantize_oscar2bit(data_bnsh, group_size, rho, meta_fp16=False):
    """Replicate the C++ OSCAR 2-bit per-group asymmetric codec (quantize+dequantize).

    Per token row:
      * optional clip tau = |row| sorted, indexed at int(rho * head_size) over the full row
        (rho <= 0 disables it; rho >= 1 selects the row max, a no-op clip),
    Per group (contiguous head_size/group_size channels, after the shared row clip):
      * scale = (max - min) / 3, zero = min (over the clipped values),
      * code  = round((clip(x) - min) / scale) clamped to [0, 3],
      * dequant = code * scale + min.

    When meta_fp16 is set, the per-group scale and zero (min) are round-tripped
    through fp16 before dequant, matching the kernel's fp16 metadata storage.
    """
    b, n, s, h = data_bnsh.shape
    assert h % group_size == 0, "head_size must be divisible by group_size"
    num_groups = h // group_size
    x = data_bnsh.astype(np.float32)
    out = np.empty_like(x)
    for bi in range(b):
        for ni in range(n):
            for si in range(s):
                row = x[bi, ni, si]
                drow = out[bi, ni, si]
                # OSCAR clips per row over the full head_size before groupwise quant.
                tau = _oscar2bit_clip_threshold(row, rho)
                for g in range(num_groups):
                    base = g * group_size
                    gp = row[base : base + group_size]
                    clipped = np.clip(gp, -tau, tau)
                    gmin = float(clipped.min())
                    gmax = float(clipped.max())
                    scale = (gmax - gmin) / OSCAR2BIT_Q_MAX
                    if not (scale > 0.0):
                        scale = 1.0
                    codes = np.clip(np.round((clipped - gmin) / scale), 0, OSCAR2BIT_Q_MAX)
                    if meta_fp16:
                        scale = float(np.float16(scale))
                        gmin = float(np.float16(gmin))
                    drow[base : base + group_size] = codes * scale + gmin
    return out


# ---- Reference attention ----


def reference_gqa(q_input, k_input, v_input, num_heads, kv_num_heads, head_size, causal=True, attention_bias=None):
    """Reference FP32 GQA: q[B,S,num_heads*H], k[B,N,S_kv,H], v[B,N,S_kv,H] -> out[B,S,num_heads*H].
    attention_bias: [B|1, num_heads|1, S, S_kv] or None.
    """
    batch, seq, _ = q_input.shape
    s_kv = k_input.shape[2]
    groups = num_heads // kv_num_heads
    scale = 1.0 / math.sqrt(head_size)

    # Reshape Q to BNSH
    q_bnsh = q_input.reshape(batch, seq, num_heads, head_size).transpose(0, 2, 1, 3)

    output = np.zeros((batch, num_heads, seq, head_size), dtype=np.float32)

    for b in range(batch):
        for h in range(num_heads):
            kv_h = h // groups
            for q_s in range(seq):
                # QK^T
                logits = np.zeros(s_kv, dtype=np.float32)
                for k_s in range(s_kv):
                    logits[k_s] = np.dot(q_bnsh[b, h, q_s], k_input[b, kv_h, k_s]) * scale
                # Attention bias
                if attention_bias is not None:
                    bias_b = 0 if attention_bias.shape[0] == 1 else b
                    bias_h = 0 if attention_bias.shape[1] == 1 else h
                    logits[:s_kv] += attention_bias[bias_b, bias_h, q_s, :s_kv]
                # Causal mask
                if causal:
                    for k_s in range(q_s + 1, s_kv):
                        logits[k_s] = -np.inf
                # Softmax
                max_val = np.max(logits)
                exp_logits = np.exp(logits - max_val)
                sum_exp = np.sum(exp_logits)
                probs = exp_logits / sum_exp
                # Output
                output[b, h, q_s] = np.dot(probs, v_input[b, kv_h])

    # Transpose back to [B, S, num_heads * H]
    return output.transpose(0, 2, 1, 3).reshape(batch, seq, num_heads * head_size)


# ---- ONNX graph construction ----


def create_quantized_gqa_graph(
    batch_size,
    seq_len,
    num_heads,
    kv_num_heads,
    head_size,
    quant_type,
    bit_width,
    buffer_seq_len=None,
    is_past=False,
    packed_qkv=False,
):
    """Create an ONNX graph for GroupQueryAttention with quantized KV cache."""
    if buffer_seq_len is None:
        buffer_seq_len = seq_len

    hidden_size = num_heads * head_size
    kv_hidden_size = kv_num_heads * head_size
    query_hidden_size = (num_heads + 2 * kv_num_heads) * head_size if packed_qkv else hidden_size
    packed_head_size = head_size // 2 if bit_width == 4 else head_size

    cache_ort_type = TensorProto.UINT8 if bit_width == 4 else TensorProto.INT8

    # Determine present sequence length
    if is_past:
        past_kv_seqlen = buffer_seq_len
        present_kv_seqlen = buffer_seq_len
    else:
        past_kv_seqlen = buffer_seq_len
        present_kv_seqlen = buffer_seq_len

    # Inputs
    inputs = [
        "query",
        "" if packed_qkv else "key",
        "" if packed_qkv else "value",
        "past_key",
        "past_value",
        "seqlens_k",
        "total_sequence_length",
        "",  # cos_cache
        "",  # sin_cache
        "",  # position_ids
        "",  # attention_bias
        "",  # head_sink
        "k_scale",
        "v_scale",
    ]

    # Remove trailing empty strings
    while inputs and inputs[-1] == "":
        inputs.pop()

    node = helper.make_node(
        op_type="GroupQueryAttention",
        inputs=inputs,
        outputs=["output", "present_key", "present_value"],
        name="GroupQueryAttention_0",
        num_heads=num_heads,
        kv_num_heads=kv_num_heads,
        k_quant_type=quant_type,
        v_quant_type=quant_type,
        kv_cache_bit_width=bit_width,
        domain="com.microsoft",
    )

    # Graph inputs
    graph_input = [
        helper.make_tensor_value_info("query", TensorProto.FLOAT, [batch_size, seq_len, query_hidden_size]),
    ]
    if not packed_qkv:
        graph_input.extend(
            [
                helper.make_tensor_value_info("key", TensorProto.FLOAT, [batch_size, seq_len, kv_hidden_size]),
                helper.make_tensor_value_info("value", TensorProto.FLOAT, [batch_size, seq_len, kv_hidden_size]),
            ]
        )
    graph_input.extend(
        [
            helper.make_tensor_value_info(
                "past_key", cache_ort_type, [batch_size, kv_num_heads, past_kv_seqlen, packed_head_size]
            ),
            helper.make_tensor_value_info(
                "past_value", cache_ort_type, [batch_size, kv_num_heads, past_kv_seqlen, packed_head_size]
            ),
            helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [batch_size]),
            helper.make_tensor_value_info("total_sequence_length", TensorProto.INT32, [1]),
            helper.make_tensor_value_info("k_scale", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("v_scale", TensorProto.FLOAT, None),
        ]
    )

    # Graph outputs
    graph_output = [
        helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]),
        helper.make_tensor_value_info(
            "present_key", cache_ort_type, [batch_size, kv_num_heads, present_kv_seqlen, packed_head_size]
        ),
        helper.make_tensor_value_info(
            "present_value", cache_ort_type, [batch_size, kv_num_heads, present_kv_seqlen, packed_head_size]
        ),
    ]

    graph = helper.make_graph([node], "QuantizedGQA_Graph", graph_input, graph_output)
    model = helper.make_model(graph)
    return model.SerializeToString()


def create_quantized_gqa_graph_with_bias(
    batch_size,
    seq_len,
    num_heads,
    kv_num_heads,
    head_size,
    quant_type,
    bit_width,
    bias_batch_size,
    bias_num_heads,
    total_seqlen,
    buffer_seq_len=None,
):
    """Create an ONNX graph for GroupQueryAttention with quantized KV cache and attention bias."""
    if buffer_seq_len is None:
        buffer_seq_len = seq_len

    hidden_size = num_heads * head_size
    kv_hidden_size = kv_num_heads * head_size
    packed_head_size = head_size // 2 if bit_width == 4 else head_size

    cache_ort_type = TensorProto.UINT8 if bit_width == 4 else TensorProto.INT8

    past_kv_seqlen = buffer_seq_len
    present_kv_seqlen = buffer_seq_len

    # Inputs (attention_bias at index 10)
    inputs = [
        "query",
        "key",
        "value",
        "past_key",
        "past_value",
        "seqlens_k",
        "total_sequence_length",
        "",  # cos_cache
        "",  # sin_cache
        "",  # position_ids
        "attention_bias",
        "",  # head_sink
        "k_scale",
        "v_scale",
    ]

    # Remove trailing empty strings
    while inputs and inputs[-1] == "":
        inputs.pop()

    node = helper.make_node(
        op_type="GroupQueryAttention",
        inputs=inputs,
        outputs=["output", "present_key", "present_value"],
        name="GroupQueryAttention_0",
        num_heads=num_heads,
        kv_num_heads=kv_num_heads,
        k_quant_type=quant_type,
        v_quant_type=quant_type,
        kv_cache_bit_width=bit_width,
        domain="com.microsoft",
    )

    # Graph inputs
    graph_input = [
        helper.make_tensor_value_info("query", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]),
        helper.make_tensor_value_info("key", TensorProto.FLOAT, [batch_size, seq_len, kv_hidden_size]),
        helper.make_tensor_value_info("value", TensorProto.FLOAT, [batch_size, seq_len, kv_hidden_size]),
        helper.make_tensor_value_info(
            "past_key", cache_ort_type, [batch_size, kv_num_heads, past_kv_seqlen, packed_head_size]
        ),
        helper.make_tensor_value_info(
            "past_value", cache_ort_type, [batch_size, kv_num_heads, past_kv_seqlen, packed_head_size]
        ),
        helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [batch_size]),
        helper.make_tensor_value_info("total_sequence_length", TensorProto.INT32, [1]),
        helper.make_tensor_value_info(
            "attention_bias", TensorProto.FLOAT, [bias_batch_size, bias_num_heads, seq_len, total_seqlen]
        ),
        helper.make_tensor_value_info("k_scale", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("v_scale", TensorProto.FLOAT, None),
    ]

    # Graph outputs
    graph_output = [
        helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]),
        helper.make_tensor_value_info(
            "present_key", cache_ort_type, [batch_size, kv_num_heads, present_kv_seqlen, packed_head_size]
        ),
        helper.make_tensor_value_info(
            "present_value", cache_ort_type, [batch_size, kv_num_heads, present_kv_seqlen, packed_head_size]
        ),
    ]

    graph = helper.make_graph([node], "QuantizedGQA_Bias_Graph", graph_input, graph_output)
    model = helper.make_model(graph)
    return model.SerializeToString()


# ---- Test runner ----


def select_quant_atol(bit_width, quant_type, batch_size, seq_len, head_size, atol=None):
    """Select the comparison tolerance for quantized GQA outputs.

    INT4 caches need a looser bound. INT8 per-tensor caches drift slightly more on
    multi-batch or long-sequence CI runs while remaining within expected quantization
    behavior. An explicit ``atol`` always overrides the heuristic.
    """
    if atol is not None:
        return atol
    if bit_width == 4:
        return 0.15
    if bit_width == 8 and quant_type == "PER_TENSOR":
        if batch_size > 1 or seq_len >= 32:
            return 0.08
        if head_size >= 64:
            return 0.06
        return 0.05
    return 0.05


def run_quantized_gqa_prompt_test(
    batch_size, seq_len, num_heads, kv_num_heads, head_size, quant_type, bit_width, atol=None
):
    """Run a quantized GQA prompt test and compare against FP32 reference with quantization noise."""
    np.random.seed(42)

    hidden_size = num_heads * head_size
    kv_hidden_size = kv_num_heads * head_size

    # Generate random input data (small magnitude)
    query = np.random.uniform(-0.5, 0.5, (batch_size, seq_len, hidden_size)).astype(np.float32)
    key_input = np.random.uniform(-0.5, 0.5, (batch_size, seq_len, kv_hidden_size)).astype(np.float32)
    value_input = np.random.uniform(-0.5, 0.5, (batch_size, seq_len, kv_hidden_size)).astype(np.float32)

    # Reshape K/V to BNSH for quantization
    k_bnsh = key_input.reshape(batch_size, seq_len, kv_num_heads, head_size).transpose(0, 2, 1, 3)
    v_bnsh = value_input.reshape(batch_size, seq_len, kv_num_heads, head_size).transpose(0, 2, 1, 3)

    # Compute scales from the data
    if bit_width == 8:
        if quant_type == "PER_TENSOR":
            _, k_scale = quantize_int8_per_tensor(k_bnsh)
            _, v_scale = quantize_int8_per_tensor(v_bnsh)
        else:
            _, k_scale = quantize_int8_per_channel(k_bnsh)
            _, v_scale = quantize_int8_per_channel(v_bnsh)
    else:
        if quant_type == "PER_TENSOR":
            _, k_scale = quantize_int4_per_tensor(k_bnsh)
            _, v_scale = quantize_int4_per_tensor(v_bnsh)
        else:
            _, k_scale = quantize_int4_per_channel(k_bnsh)
            _, v_scale = quantize_int4_per_channel(v_bnsh)

    # Create empty past cache (prompt phase)
    packed_head_size = head_size // 2 if bit_width == 4 else head_size
    if bit_width == 4:
        past_k = np.zeros((batch_size, kv_num_heads, seq_len, packed_head_size), dtype=np.uint8)
        past_v = np.zeros((batch_size, kv_num_heads, seq_len, packed_head_size), dtype=np.uint8)
    else:
        past_k = np.zeros((batch_size, kv_num_heads, seq_len, packed_head_size), dtype=np.int8)
        past_v = np.zeros((batch_size, kv_num_heads, seq_len, packed_head_size), dtype=np.int8)

    seqlens_k = np.array([seq_len - 1] * batch_size, dtype=np.int32)
    total_seq = np.array([seq_len], dtype=np.int32)

    # Build and run ONNX model
    onnx_model_str = create_quantized_gqa_graph(
        batch_size, seq_len, num_heads, kv_num_heads, head_size, quant_type, bit_width
    )
    sess_options = SessionOptions()
    sess = InferenceSession(onnx_model_str, sess_options, providers=["CPUExecutionProvider"])

    feeds = {
        "query": query,
        "key": key_input,
        "value": value_input,
        "past_key": past_k,
        "past_value": past_v,
        "seqlens_k": seqlens_k,
        "total_sequence_length": total_seq,
        "k_scale": k_scale,
        "v_scale": v_scale,
    }

    outputs = sess.run(None, feeds)
    out_ort = outputs[0]

    # Compute reference: quantize + dequantize K/V, then FP32 attention
    if bit_width == 8 and quant_type == "PER_TENSOR":
        k_q, _ = quantize_int8_per_tensor(k_bnsh)
        v_q, _ = quantize_int8_per_tensor(v_bnsh)
        # Re-quantize with provided scale
        k_q = np.clip(np.round(k_bnsh / k_scale[0]), -128, 127).astype(np.int8)
        v_q = np.clip(np.round(v_bnsh / v_scale[0]), -128, 127).astype(np.int8)
        k_deq = dequantize_int8_per_tensor(k_q, k_scale[0])
        v_deq = dequantize_int8_per_tensor(v_q, v_scale[0])
    elif bit_width == 8 and quant_type == "PER_CHANNEL":
        k_q = np.clip(np.round(k_bnsh / k_scale.reshape(1, kv_num_heads, 1, head_size)), -128, 127).astype(np.int8)
        v_q = np.clip(np.round(v_bnsh / v_scale.reshape(1, kv_num_heads, 1, head_size)), -128, 127).astype(np.int8)
        k_deq = dequantize_int8_per_channel(k_q, k_scale, kv_num_heads, head_size)
        v_deq = dequantize_int8_per_channel(v_q, v_scale, kv_num_heads, head_size)
    elif bit_width == 4 and quant_type == "PER_TENSOR":
        k_q = np.clip(np.round(k_bnsh / k_scale[0]), -8, 7).astype(np.int8)
        v_q = np.clip(np.round(v_bnsh / v_scale[0]), -8, 7).astype(np.int8)
        k_deq = k_q.astype(np.float32) * k_scale[0]
        v_deq = v_q.astype(np.float32) * v_scale[0]
    elif bit_width == 4 and quant_type == "PER_CHANNEL":
        k_q = np.clip(np.round(k_bnsh / k_scale.reshape(1, kv_num_heads, 1, head_size)), -8, 7).astype(np.int8)
        v_q = np.clip(np.round(v_bnsh / v_scale.reshape(1, kv_num_heads, 1, head_size)), -8, 7).astype(np.int8)
        k_deq = k_q.astype(np.float32) * k_scale.reshape(1, kv_num_heads, 1, head_size)
        v_deq = v_q.astype(np.float32) * v_scale.reshape(1, kv_num_heads, 1, head_size)
    else:
        raise ValueError(f"Unsupported config: bit_width={bit_width}, quant_type={quant_type}")

    out_ref = reference_gqa(query, k_deq, v_deq, num_heads, kv_num_heads, head_size, causal=True)

    # Compare
    atol = select_quant_atol(bit_width, quant_type, batch_size, seq_len, head_size, atol)

    # Check for NaN
    if np.any(np.isnan(out_ort)):
        raise AssertionError(f"NaN in output (quant={quant_type}, bit={bit_width})")
    # Check non-zero
    if np.allclose(out_ort, 0.0):
        raise AssertionError(f"Output is all zeros (quant={quant_type}, bit={bit_width})")

    np.testing.assert_allclose(
        out_ort,
        out_ref,
        atol=atol,
        rtol=0.1,
        err_msg=f"Quantized GQA output mismatch (quant={quant_type}, bit={bit_width})",
    )


def run_quantized_gqa_packed_qkv_test(
    batch_size, seq_len, num_heads, kv_num_heads, head_size, quant_type, bit_width, atol=None
):
    """Run a packed-QKV quantized GQA prompt test and compare against FP32 reference with quantization noise."""
    np.random.seed(43)

    hidden_size = num_heads * head_size
    kv_hidden_size = kv_num_heads * head_size

    query = np.random.uniform(-0.5, 0.5, (batch_size, seq_len, hidden_size)).astype(np.float32)
    key_input = np.random.uniform(-0.5, 0.5, (batch_size, seq_len, kv_hidden_size)).astype(np.float32)
    value_input = np.random.uniform(-0.5, 0.5, (batch_size, seq_len, kv_hidden_size)).astype(np.float32)
    packed_qkv = np.concatenate([query, key_input, value_input], axis=2)

    k_bnsh = key_input.reshape(batch_size, seq_len, kv_num_heads, head_size).transpose(0, 2, 1, 3)
    v_bnsh = value_input.reshape(batch_size, seq_len, kv_num_heads, head_size).transpose(0, 2, 1, 3)

    if bit_width == 8:
        if quant_type == "PER_TENSOR":
            _, k_scale = quantize_int8_per_tensor(k_bnsh)
            _, v_scale = quantize_int8_per_tensor(v_bnsh)
        else:
            _, k_scale = quantize_int8_per_channel(k_bnsh)
            _, v_scale = quantize_int8_per_channel(v_bnsh)
    else:
        if quant_type == "PER_TENSOR":
            _, k_scale = quantize_int4_per_tensor(k_bnsh)
            _, v_scale = quantize_int4_per_tensor(v_bnsh)
        else:
            _, k_scale = quantize_int4_per_channel(k_bnsh)
            _, v_scale = quantize_int4_per_channel(v_bnsh)

    packed_head_size = head_size // 2 if bit_width == 4 else head_size
    if bit_width == 4:
        past_k = np.zeros((batch_size, kv_num_heads, seq_len, packed_head_size), dtype=np.uint8)
        past_v = np.zeros((batch_size, kv_num_heads, seq_len, packed_head_size), dtype=np.uint8)
    else:
        past_k = np.zeros((batch_size, kv_num_heads, seq_len, packed_head_size), dtype=np.int8)
        past_v = np.zeros((batch_size, kv_num_heads, seq_len, packed_head_size), dtype=np.int8)

    seqlens_k = np.array([seq_len - 1] * batch_size, dtype=np.int32)
    total_seq = np.array([seq_len], dtype=np.int32)

    onnx_model_str = create_quantized_gqa_graph(
        batch_size, seq_len, num_heads, kv_num_heads, head_size, quant_type, bit_width, packed_qkv=True
    )
    sess_options = SessionOptions()
    sess = InferenceSession(onnx_model_str, sess_options, providers=["CPUExecutionProvider"])

    feeds = {
        "query": packed_qkv,
        "past_key": past_k,
        "past_value": past_v,
        "seqlens_k": seqlens_k,
        "total_sequence_length": total_seq,
        "k_scale": k_scale,
        "v_scale": v_scale,
    }

    outputs = sess.run(None, feeds)
    out_ort = outputs[0]

    if bit_width == 8 and quant_type == "PER_TENSOR":
        k_q = np.clip(np.round(k_bnsh / k_scale[0]), -128, 127).astype(np.int8)
        v_q = np.clip(np.round(v_bnsh / v_scale[0]), -128, 127).astype(np.int8)
        k_deq = dequantize_int8_per_tensor(k_q, k_scale[0])
        v_deq = dequantize_int8_per_tensor(v_q, v_scale[0])
    elif bit_width == 8 and quant_type == "PER_CHANNEL":
        k_q = np.clip(np.round(k_bnsh / k_scale.reshape(1, kv_num_heads, 1, head_size)), -128, 127).astype(np.int8)
        v_q = np.clip(np.round(v_bnsh / v_scale.reshape(1, kv_num_heads, 1, head_size)), -128, 127).astype(np.int8)
        k_deq = dequantize_int8_per_channel(k_q, k_scale, kv_num_heads, head_size)
        v_deq = dequantize_int8_per_channel(v_q, v_scale, kv_num_heads, head_size)
    elif bit_width == 4 and quant_type == "PER_TENSOR":
        k_q = np.clip(np.round(k_bnsh / k_scale[0]), -8, 7).astype(np.int8)
        v_q = np.clip(np.round(v_bnsh / v_scale[0]), -8, 7).astype(np.int8)
        k_deq = k_q.astype(np.float32) * k_scale[0]
        v_deq = v_q.astype(np.float32) * v_scale[0]
    elif bit_width == 4 and quant_type == "PER_CHANNEL":
        k_q = np.clip(np.round(k_bnsh / k_scale.reshape(1, kv_num_heads, 1, head_size)), -8, 7).astype(np.int8)
        v_q = np.clip(np.round(v_bnsh / v_scale.reshape(1, kv_num_heads, 1, head_size)), -8, 7).astype(np.int8)
        k_deq = k_q.astype(np.float32) * k_scale.reshape(1, kv_num_heads, 1, head_size)
        v_deq = v_q.astype(np.float32) * v_scale.reshape(1, kv_num_heads, 1, head_size)
    else:
        raise ValueError(f"Unsupported config: bit_width={bit_width}, quant_type={quant_type}")

    out_ref = reference_gqa(query, k_deq, v_deq, num_heads, kv_num_heads, head_size, causal=True)

    atol = select_quant_atol(bit_width, quant_type, batch_size, seq_len, head_size, atol)

    if np.any(np.isnan(out_ort)):
        raise AssertionError(f"NaN in output (quant={quant_type}, bit={bit_width}, packed QKV)")
    if np.allclose(out_ort, 0.0):
        raise AssertionError(f"Output is all zeros (quant={quant_type}, bit={bit_width}, packed QKV)")

    np.testing.assert_allclose(
        out_ort,
        out_ref,
        atol=atol,
        rtol=0.1,
        err_msg=f"Packed-QKV quantized GQA output mismatch (quant={quant_type}, bit={bit_width})",
    )


# ---- Test class ----


class TestGQACPUQuantizedKV(unittest.TestCase):
    """Test CPU GroupQueryAttention with quantized KV cache."""

    def test_int8_per_tensor_basic(self):
        run_quantized_gqa_prompt_test(
            batch_size=1,
            seq_len=4,
            num_heads=2,
            kv_num_heads=1,
            head_size=8,
            quant_type="PER_TENSOR",
            bit_width=8,
        )

    def test_int8_per_channel_basic(self):
        run_quantized_gqa_prompt_test(
            batch_size=1,
            seq_len=4,
            num_heads=2,
            kv_num_heads=1,
            head_size=8,
            quant_type="PER_CHANNEL",
            bit_width=8,
        )

    def test_int4_per_tensor_basic(self):
        run_quantized_gqa_prompt_test(
            batch_size=1,
            seq_len=4,
            num_heads=2,
            kv_num_heads=1,
            head_size=8,
            quant_type="PER_TENSOR",
            bit_width=4,
        )

    def test_int4_per_channel_basic(self):
        run_quantized_gqa_prompt_test(
            batch_size=1,
            seq_len=4,
            num_heads=2,
            kv_num_heads=1,
            head_size=8,
            quant_type="PER_CHANNEL",
            bit_width=4,
        )

    def test_int8_multi_batch(self):
        run_quantized_gqa_prompt_test(
            batch_size=2,
            seq_len=4,
            num_heads=4,
            kv_num_heads=2,
            head_size=16,
            quant_type="PER_TENSOR",
            bit_width=8,
        )

    def test_int8_packed_qkv_multi_batch(self):
        run_quantized_gqa_packed_qkv_test(
            batch_size=3,
            seq_len=8,
            num_heads=4,
            kv_num_heads=2,
            head_size=16,
            quant_type="PER_TENSOR",
            bit_width=8,
        )

    def test_int4_multi_batch(self):
        run_quantized_gqa_prompt_test(
            batch_size=2,
            seq_len=4,
            num_heads=4,
            kv_num_heads=2,
            head_size=16,
            quant_type="PER_TENSOR",
            bit_width=4,
        )

    def test_int8_large_head(self):
        run_quantized_gqa_prompt_test(
            batch_size=1,
            seq_len=8,
            num_heads=2,
            kv_num_heads=1,
            head_size=64,
            quant_type="PER_TENSOR",
            bit_width=8,
        )

    def test_int4_large_head(self):
        run_quantized_gqa_prompt_test(
            batch_size=1,
            seq_len=8,
            num_heads=2,
            kv_num_heads=1,
            head_size=64,
            quant_type="PER_TENSOR",
            bit_width=4,
        )

    def test_int8_gqa_ratio_4(self):
        """num_heads=4, kv_num_heads=1: GQA ratio 4:1."""
        run_quantized_gqa_prompt_test(
            batch_size=1,
            seq_len=4,
            num_heads=4,
            kv_num_heads=1,
            head_size=16,
            quant_type="PER_TENSOR",
            bit_width=8,
        )

    def test_int8_per_channel_large(self):
        run_quantized_gqa_prompt_test(
            batch_size=1,
            seq_len=16,
            num_heads=4,
            kv_num_heads=2,
            head_size=32,
            quant_type="PER_CHANNEL",
            bit_width=8,
        )

    def test_int4_per_channel_large(self):
        run_quantized_gqa_prompt_test(
            batch_size=1,
            seq_len=16,
            num_heads=4,
            kv_num_heads=2,
            head_size=32,
            quant_type="PER_CHANNEL",
            bit_width=4,
        )

    @unittest.skipIf(pipeline_mode, "Extended tests disabled in pipeline mode")
    def test_int8_long_sequence(self):
        run_quantized_gqa_prompt_test(
            batch_size=1,
            seq_len=128,
            num_heads=8,
            kv_num_heads=2,
            head_size=64,
            quant_type="PER_TENSOR",
            bit_width=8,
        )

    @unittest.skipIf(pipeline_mode, "Extended tests disabled in pipeline mode")
    def test_int4_long_sequence(self):
        run_quantized_gqa_prompt_test(
            batch_size=1,
            seq_len=128,
            num_heads=8,
            kv_num_heads=2,
            head_size=64,
            quant_type="PER_TENSOR",
            bit_width=4,
        )


def run_quantized_gqa_bias_test(
    batch_size,
    seq_len,
    num_heads,
    kv_num_heads,
    head_size,
    quant_type,
    bit_width,
    bias_broadcast_batch=False,
    bias_broadcast_head=False,
    atol=None,
):
    """Run a quantized GQA test with attention bias and compare against reference."""
    np.random.seed(123)

    hidden_size = num_heads * head_size
    kv_hidden_size = kv_num_heads * head_size

    query = np.random.uniform(-0.5, 0.5, (batch_size, seq_len, hidden_size)).astype(np.float32)
    key_input = np.random.uniform(-0.5, 0.5, (batch_size, seq_len, kv_hidden_size)).astype(np.float32)
    value_input = np.random.uniform(-0.5, 0.5, (batch_size, seq_len, kv_hidden_size)).astype(np.float32)

    # Reshape K/V to BNSH for quantization
    k_bnsh = key_input.reshape(batch_size, seq_len, kv_num_heads, head_size).transpose(0, 2, 1, 3)
    v_bnsh = value_input.reshape(batch_size, seq_len, kv_num_heads, head_size).transpose(0, 2, 1, 3)

    # Compute scales
    if bit_width == 8:
        if quant_type == "PER_TENSOR":
            _, k_scale = quantize_int8_per_tensor(k_bnsh)
            _, v_scale = quantize_int8_per_tensor(v_bnsh)
        else:
            _, k_scale = quantize_int8_per_channel(k_bnsh)
            _, v_scale = quantize_int8_per_channel(v_bnsh)
    else:
        if quant_type == "PER_TENSOR":
            _, k_scale = quantize_int4_per_tensor(k_bnsh)
            _, v_scale = quantize_int4_per_tensor(v_bnsh)
        else:
            _, k_scale = quantize_int4_per_channel(k_bnsh)
            _, v_scale = quantize_int4_per_channel(v_bnsh)

    # Empty past (prompt)
    packed_head_size = head_size // 2 if bit_width == 4 else head_size
    if bit_width == 4:
        past_k = np.zeros((batch_size, kv_num_heads, seq_len, packed_head_size), dtype=np.uint8)
        past_v = np.zeros((batch_size, kv_num_heads, seq_len, packed_head_size), dtype=np.uint8)
    else:
        past_k = np.zeros((batch_size, kv_num_heads, seq_len, packed_head_size), dtype=np.int8)
        past_v = np.zeros((batch_size, kv_num_heads, seq_len, packed_head_size), dtype=np.int8)

    seqlens_k = np.array([seq_len - 1] * batch_size, dtype=np.int32)
    total_seq = np.array([seq_len], dtype=np.int32)

    # Generate attention bias
    bias_batch = 1 if bias_broadcast_batch else batch_size
    bias_heads = 1 if bias_broadcast_head else num_heads
    attention_bias = np.random.uniform(-1.0, 1.0, (bias_batch, bias_heads, seq_len, seq_len)).astype(np.float32)

    # Build and run ONNX model
    onnx_model_str = create_quantized_gqa_graph_with_bias(
        batch_size,
        seq_len,
        num_heads,
        kv_num_heads,
        head_size,
        quant_type,
        bit_width,
        bias_batch_size=bias_batch,
        bias_num_heads=bias_heads,
        total_seqlen=seq_len,
    )
    sess_options = SessionOptions()
    sess = InferenceSession(onnx_model_str, sess_options, providers=["CPUExecutionProvider"])

    feeds = {
        "query": query,
        "key": key_input,
        "value": value_input,
        "past_key": past_k,
        "past_value": past_v,
        "seqlens_k": seqlens_k,
        "total_sequence_length": total_seq,
        "attention_bias": attention_bias,
        "k_scale": k_scale,
        "v_scale": v_scale,
    }

    outputs = sess.run(None, feeds)
    out_ort = outputs[0]

    # Compute reference with quantized K/V
    if bit_width == 8 and quant_type == "PER_TENSOR":
        k_q = np.clip(np.round(k_bnsh / k_scale[0]), -128, 127).astype(np.int8)
        v_q = np.clip(np.round(v_bnsh / v_scale[0]), -128, 127).astype(np.int8)
        k_deq = dequantize_int8_per_tensor(k_q, k_scale[0])
        v_deq = dequantize_int8_per_tensor(v_q, v_scale[0])
    elif bit_width == 8 and quant_type == "PER_CHANNEL":
        k_q = np.clip(np.round(k_bnsh / k_scale.reshape(1, kv_num_heads, 1, head_size)), -128, 127).astype(np.int8)
        v_q = np.clip(np.round(v_bnsh / v_scale.reshape(1, kv_num_heads, 1, head_size)), -128, 127).astype(np.int8)
        k_deq = dequantize_int8_per_channel(k_q, k_scale, kv_num_heads, head_size)
        v_deq = dequantize_int8_per_channel(v_q, v_scale, kv_num_heads, head_size)
    elif bit_width == 4 and quant_type == "PER_TENSOR":
        k_q = np.clip(np.round(k_bnsh / k_scale[0]), -8, 7).astype(np.int8)
        v_q = np.clip(np.round(v_bnsh / v_scale[0]), -8, 7).astype(np.int8)
        k_deq = k_q.astype(np.float32) * k_scale[0]
        v_deq = v_q.astype(np.float32) * v_scale[0]
    elif bit_width == 4 and quant_type == "PER_CHANNEL":
        k_q = np.clip(np.round(k_bnsh / k_scale.reshape(1, kv_num_heads, 1, head_size)), -8, 7).astype(np.int8)
        v_q = np.clip(np.round(v_bnsh / v_scale.reshape(1, kv_num_heads, 1, head_size)), -8, 7).astype(np.int8)
        k_deq = k_q.astype(np.float32) * k_scale.reshape(1, kv_num_heads, 1, head_size)
        v_deq = v_q.astype(np.float32) * v_scale.reshape(1, kv_num_heads, 1, head_size)
    else:
        raise ValueError(f"Unsupported config: bit_width={bit_width}, quant_type={quant_type}")

    out_ref = reference_gqa(
        query, k_deq, v_deq, num_heads, kv_num_heads, head_size, causal=True, attention_bias=attention_bias
    )

    atol = select_quant_atol(bit_width, quant_type, batch_size, seq_len, head_size, atol)

    if np.any(np.isnan(out_ort)):
        raise AssertionError(f"NaN in output (quant={quant_type}, bit={bit_width}, bias test)")
    if np.allclose(out_ort, 0.0):
        raise AssertionError(f"Output is all zeros (quant={quant_type}, bit={bit_width}, bias test)")

    np.testing.assert_allclose(
        out_ort,
        out_ref,
        atol=atol,
        rtol=0.1,
        err_msg=f"Quantized GQA + bias mismatch (quant={quant_type}, bit={bit_width})",
    )


class TestGQACPUQuantizedKVWithBias(unittest.TestCase):
    """Test CPU GroupQueryAttention with quantized KV cache and attention bias."""

    def test_int8_per_tensor_bias(self):
        run_quantized_gqa_bias_test(
            batch_size=1,
            seq_len=8,
            num_heads=2,
            kv_num_heads=1,
            head_size=16,
            quant_type="PER_TENSOR",
            bit_width=8,
        )

    def test_int8_per_channel_bias(self):
        run_quantized_gqa_bias_test(
            batch_size=1,
            seq_len=8,
            num_heads=2,
            kv_num_heads=1,
            head_size=16,
            quant_type="PER_CHANNEL",
            bit_width=8,
        )

    def test_int4_per_tensor_bias(self):
        run_quantized_gqa_bias_test(
            batch_size=1,
            seq_len=8,
            num_heads=2,
            kv_num_heads=1,
            head_size=16,
            quant_type="PER_TENSOR",
            bit_width=4,
        )

    def test_int4_per_channel_bias(self):
        run_quantized_gqa_bias_test(
            batch_size=1,
            seq_len=8,
            num_heads=2,
            kv_num_heads=1,
            head_size=16,
            quant_type="PER_CHANNEL",
            bit_width=4,
        )

    def test_int8_bias_broadcast_batch(self):
        """Bias shape [1, N, S, T] with batch_size > 1."""
        run_quantized_gqa_bias_test(
            batch_size=2,
            seq_len=8,
            num_heads=4,
            kv_num_heads=2,
            head_size=16,
            quant_type="PER_TENSOR",
            bit_width=8,
            bias_broadcast_batch=True,
        )

    def test_int8_bias_broadcast_head(self):
        """Bias shape [B, 1, S, T] with num_heads > 1."""
        run_quantized_gqa_bias_test(
            batch_size=1,
            seq_len=8,
            num_heads=4,
            kv_num_heads=2,
            head_size=16,
            quant_type="PER_TENSOR",
            bit_width=8,
            bias_broadcast_head=True,
        )

    def test_int8_bias_broadcast_head_multi_batch(self):
        """Bias shape [B, 1, S, T] with batch_size > 1 and num_heads > 1.

        Regression test: the bias batch stride must use the head extent (1 when the
        head dimension is broadcast), not num_heads. With batch_size == 1 the bug is
        masked because batch_idx is always 0.
        """
        run_quantized_gqa_bias_test(
            batch_size=3,
            seq_len=8,
            num_heads=4,
            kv_num_heads=2,
            head_size=16,
            quant_type="PER_TENSOR",
            bit_width=8,
            bias_broadcast_head=True,
        )

    def test_int8_bias_broadcast_both(self):
        """Bias shape [1, 1, S, T] with batch_size > 1 and num_heads > 1."""
        run_quantized_gqa_bias_test(
            batch_size=2,
            seq_len=8,
            num_heads=4,
            kv_num_heads=2,
            head_size=16,
            quant_type="PER_TENSOR",
            bit_width=8,
            bias_broadcast_batch=True,
            bias_broadcast_head=True,
        )

    def test_int8_bias_large(self):
        """Larger test to exercise flash attention path with bias."""
        run_quantized_gqa_bias_test(
            batch_size=2,
            seq_len=32,
            num_heads=4,
            kv_num_heads=2,
            head_size=64,
            quant_type="PER_TENSOR",
            bit_width=8,
        )

    def test_int4_bias_large(self):
        """Larger test with INT4 to exercise flash attention path with bias."""
        run_quantized_gqa_bias_test(
            batch_size=2,
            seq_len=32,
            num_heads=4,
            kv_num_heads=2,
            head_size=64,
            quant_type="PER_CHANNEL",
            bit_width=4,
        )


# ---- OSCAR 2-bit PER_GROUP graph / runners ----


def create_oscar2bit_gqa_graph(
    batch_size,
    q_len,
    past_len,
    present_len,
    num_heads,
    kv_num_heads,
    head_size,
    group_size,
    k_rho,
    v_rho,
    meta_fp16=False,
    io_fp16=False,
):
    """ONNX graph for MixedPrecisionGroupQueryAttention with the 2-bit PER_GROUP (OSCAR) KV cache
    and no high-precision sink/recent window (sink_size == recent_size == 0).

    Unlike the INT8/INT4 path, scales/zeros are packed inline in the cache rows, so
    there are no k_scale/v_scale inputs and the cache dtype is UINT8. Independent
    past_len/present_len allow driving an incremental decode step.

    io_fp16 selects the Q/K/V/output compute dtype (FLOAT16 vs FLOAT); the packed
    2-bit KV cache is UINT8 in either case. The fp16 kernel path bridges half<->float
    at the boundary and reuses the same float codec.
    """
    hidden_size = num_heads * head_size
    kv_hidden_size = kv_num_heads * head_size
    phs = oscar2bit_packed_head_size(head_size, group_size, meta_fp16)
    io_dtype = TensorProto.FLOAT16 if io_fp16 else TensorProto.FLOAT

    node = helper.make_node(
        op_type="MixedPrecisionGroupQueryAttention",
        inputs=["query", "key", "value", "past_key", "past_value", "seqlens_k", "total_sequence_length"],
        outputs=["output", "present_key", "present_value"],
        name="MixedPrecisionGroupQueryAttention_0",
        num_heads=num_heads,
        kv_num_heads=kv_num_heads,
        kv_quant_group_size=group_size,
        k_quant_rho=float(k_rho),
        v_quant_rho=float(v_rho),
        metadata_type="fp16" if meta_fp16 else "fp32",
        cache_format_version=1,
        domain="com.microsoft",
    )

    graph_input = [
        helper.make_tensor_value_info("query", io_dtype, [batch_size, q_len, hidden_size]),
        helper.make_tensor_value_info("key", io_dtype, [batch_size, q_len, kv_hidden_size]),
        helper.make_tensor_value_info("value", io_dtype, [batch_size, q_len, kv_hidden_size]),
        helper.make_tensor_value_info("past_key", TensorProto.UINT8, [batch_size, kv_num_heads, past_len, phs]),
        helper.make_tensor_value_info("past_value", TensorProto.UINT8, [batch_size, kv_num_heads, past_len, phs]),
        helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [batch_size]),
        helper.make_tensor_value_info("total_sequence_length", TensorProto.INT32, [1]),
    ]
    graph_output = [
        helper.make_tensor_value_info("output", io_dtype, [batch_size, q_len, hidden_size]),
        helper.make_tensor_value_info("present_key", TensorProto.UINT8, [batch_size, kv_num_heads, present_len, phs]),
        helper.make_tensor_value_info("present_value", TensorProto.UINT8, [batch_size, kv_num_heads, present_len, phs]),
    ]

    graph = helper.make_graph([node], "Oscar2BitGQA_Graph", graph_input, graph_output)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
    )
    return model.SerializeToString()


def run_oscar2bit_gqa_prompt_test(
    batch_size,
    seq_len,
    num_heads,
    kv_num_heads,
    head_size,
    group_size,
    k_rho=1.0,
    v_rho=1.0,
    atol=2e-3,
    meta_fp16=False,
    io_fp16=False,
):
    """Prompt-phase parity: ORT 2-bit PER_GROUP GQA vs the codec-matched FP32 reference."""
    np.random.seed(42)
    hidden_size = num_heads * head_size
    kv_hidden_size = kv_num_heads * head_size
    io_np = np.float16 if io_fp16 else np.float32

    query = np.random.uniform(-0.5, 0.5, (batch_size, seq_len, hidden_size)).astype(io_np)
    key_input = np.random.uniform(-0.5, 0.5, (batch_size, seq_len, kv_hidden_size)).astype(io_np)
    value_input = np.random.uniform(-0.5, 0.5, (batch_size, seq_len, kv_hidden_size)).astype(io_np)

    # The kernel bridges fp16 -> float at the boundary, so the reference sees the
    # fp16-rounded values promoted back to float32.
    query_ref = query.astype(np.float32)
    k_bnsh = key_input.astype(np.float32).reshape(batch_size, seq_len, kv_num_heads, head_size).transpose(0, 2, 1, 3)
    v_bnsh = value_input.astype(np.float32).reshape(batch_size, seq_len, kv_num_heads, head_size).transpose(0, 2, 1, 3)

    phs = oscar2bit_packed_head_size(head_size, group_size, meta_fp16)
    past_k = np.zeros((batch_size, kv_num_heads, seq_len, phs), dtype=np.uint8)
    past_v = np.zeros((batch_size, kv_num_heads, seq_len, phs), dtype=np.uint8)
    seqlens_k = np.array([seq_len - 1] * batch_size, dtype=np.int32)
    total_seq = np.array([seq_len], dtype=np.int32)

    onnx_model_str = create_oscar2bit_gqa_graph(
        batch_size,
        seq_len,
        seq_len,
        seq_len,
        num_heads,
        kv_num_heads,
        head_size,
        group_size,
        k_rho,
        v_rho,
        meta_fp16=meta_fp16,
        io_fp16=io_fp16,
    )
    sess = InferenceSession(onnx_model_str, SessionOptions(), providers=["CPUExecutionProvider"])
    feeds = {
        "query": query,
        "key": key_input,
        "value": value_input,
        "past_key": past_k,
        "past_value": past_v,
        "seqlens_k": seqlens_k,
        "total_sequence_length": total_seq,
    }
    out_ort = sess.run(None, feeds)[0].astype(np.float32)

    k_deq = quantize_dequantize_oscar2bit(k_bnsh, group_size, k_rho, meta_fp16)
    v_deq = quantize_dequantize_oscar2bit(v_bnsh, group_size, v_rho, meta_fp16)
    out_ref = reference_gqa(query_ref, k_deq, v_deq, num_heads, kv_num_heads, head_size, causal=True)

    if np.any(np.isnan(out_ort)):
        raise AssertionError(f"NaN in output (oscar 2-bit, group_size={group_size})")
    if np.allclose(out_ort, 0.0):
        raise AssertionError(f"Output is all zeros (oscar 2-bit, group_size={group_size})")

    np.testing.assert_allclose(
        out_ort,
        out_ref,
        atol=atol,
        rtol=0.1,
        err_msg=f"OSCAR 2-bit GQA output mismatch (group_size={group_size}, rho=({k_rho},{v_rho}))",
    )


def run_oscar2bit_gqa_decode_test(
    batch_size,
    prompt_len,
    num_heads,
    kv_num_heads,
    head_size,
    group_size,
    k_rho=1.0,
    v_rho=1.0,
    atol=2e-3,
    meta_fp16=False,
    io_fp16=False,
):
    """Incremental-decode parity: prompt, then one decode step that reuses the prompt's
    present cache as past. Exercises ConcatQuant2BitStateChunkGQA with a populated
    (non-shared) past buffer, which the prompt-only cases never reach."""
    np.random.seed(7)
    hidden_size = num_heads * head_size
    kv_hidden_size = kv_num_heads * head_size
    s = prompt_len
    phs = oscar2bit_packed_head_size(head_size, group_size, meta_fp16)
    io_np = np.float16 if io_fp16 else np.float32

    q1 = np.random.uniform(-0.5, 0.5, (batch_size, s, hidden_size)).astype(io_np)
    k1 = np.random.uniform(-0.5, 0.5, (batch_size, s, kv_hidden_size)).astype(io_np)
    v1 = np.random.uniform(-0.5, 0.5, (batch_size, s, kv_hidden_size)).astype(io_np)
    past0 = np.zeros((batch_size, kv_num_heads, s, phs), dtype=np.uint8)

    model1 = create_oscar2bit_gqa_graph(
        batch_size,
        s,
        s,
        s,
        num_heads,
        kv_num_heads,
        head_size,
        group_size,
        k_rho,
        v_rho,
        meta_fp16=meta_fp16,
        io_fp16=io_fp16,
    )
    sess1 = InferenceSession(model1, SessionOptions(), providers=["CPUExecutionProvider"])
    out1 = sess1.run(
        None,
        {
            "query": q1,
            "key": k1,
            "value": v1,
            "past_key": past0,
            "past_value": past0,
            "seqlens_k": np.array([s - 1] * batch_size, dtype=np.int32),
            "total_sequence_length": np.array([s], dtype=np.int32),
        },
    )
    present_k, present_v = out1[1], out1[2]

    q2 = np.random.uniform(-0.5, 0.5, (batch_size, 1, hidden_size)).astype(io_np)
    k2 = np.random.uniform(-0.5, 0.5, (batch_size, 1, kv_hidden_size)).astype(io_np)
    v2 = np.random.uniform(-0.5, 0.5, (batch_size, 1, kv_hidden_size)).astype(io_np)

    model2 = create_oscar2bit_gqa_graph(
        batch_size,
        1,
        s,
        s + 1,
        num_heads,
        kv_num_heads,
        head_size,
        group_size,
        k_rho,
        v_rho,
        meta_fp16=meta_fp16,
        io_fp16=io_fp16,
    )
    sess2 = InferenceSession(model2, SessionOptions(), providers=["CPUExecutionProvider"])
    out2 = sess2.run(
        None,
        {
            "query": q2,
            "key": k2,
            "value": v2,
            "past_key": present_k,
            "past_value": present_v,
            "seqlens_k": np.array([s] * batch_size, dtype=np.int32),
            "total_sequence_length": np.array([s + 1], dtype=np.int32),
        },
    )
    out_ort = out2[0].astype(np.float32)

    # Reference sees the fp16-rounded feeds promoted back to float32 (the kernel bridges).
    k_full = np.concatenate([k1, k2], axis=1).astype(np.float32)
    v_full = np.concatenate([v1, v2], axis=1).astype(np.float32)
    k_bnsh = k_full.reshape(batch_size, s + 1, kv_num_heads, head_size).transpose(0, 2, 1, 3)
    v_bnsh = v_full.reshape(batch_size, s + 1, kv_num_heads, head_size).transpose(0, 2, 1, 3)
    k_deq = quantize_dequantize_oscar2bit(k_bnsh, group_size, k_rho, meta_fp16)
    v_deq = quantize_dequantize_oscar2bit(v_bnsh, group_size, v_rho, meta_fp16)
    # The single decode query is at the last position and attends to all cached tokens,
    # so no causal masking is applied for a length-1 query.
    out_ref = reference_gqa(q2.astype(np.float32), k_deq, v_deq, num_heads, kv_num_heads, head_size, causal=False)

    if np.any(np.isnan(out_ort)):
        raise AssertionError("NaN in output (oscar 2-bit decode)")
    np.testing.assert_allclose(
        out_ort,
        out_ref,
        atol=atol,
        rtol=0.1,
        err_msg=f"OSCAR 2-bit GQA decode mismatch (group_size={group_size}, rho=({k_rho},{v_rho}))",
    )


class TestGQACPUQuantizedKVOscar2Bit(unittest.TestCase):
    """Test CPU MixedPrecisionGroupQueryAttention with the 2-bit PER_GROUP (OSCAR) KV cache
    and no high-precision sink/recent window."""

    def test_2bit_basic(self):
        run_oscar2bit_gqa_prompt_test(1, 8, 2, 2, 16, 8)

    def test_2bit_gqa_ratio(self):
        """GQA ratio 2:1 (num_heads=2, kv_num_heads=1)."""
        run_oscar2bit_gqa_prompt_test(1, 8, 2, 1, 16, 8)

    def test_2bit_multi_batch(self):
        run_oscar2bit_gqa_prompt_test(2, 12, 4, 2, 16, 8)

    def test_2bit_one_group(self):
        """num_groups=1 (group_size == head_size)."""
        run_oscar2bit_gqa_prompt_test(1, 8, 2, 2, 16, 16)

    def test_2bit_four_groups(self):
        """num_groups=4."""
        run_oscar2bit_gqa_prompt_test(1, 8, 2, 2, 32, 8)

    def test_2bit_eight_groups(self):
        """num_groups=8."""
        run_oscar2bit_gqa_prompt_test(2, 10, 4, 2, 64, 8)

    def test_2bit_rho_clip(self):
        """Percentile clipping (rho < 1.0) enabled."""
        run_oscar2bit_gqa_prompt_test(1, 10, 2, 2, 16, 8, k_rho=0.96, v_rho=0.92)

    def test_2bit_large_head(self):
        run_oscar2bit_gqa_prompt_test(1, 6, 2, 2, 128, 64, k_rho=0.96, v_rho=0.92)

    def test_2bit_decode_step(self):
        run_oscar2bit_gqa_decode_test(1, 6, 2, 2, 16, 8)

    def test_2bit_decode_gqa_rho(self):
        run_oscar2bit_gqa_decode_test(2, 8, 4, 2, 128, 64, k_rho=0.96, v_rho=0.92)

    @unittest.skipIf(pipeline_mode, "Extended tests disabled in pipeline mode")
    def test_2bit_long_sequence(self):
        run_oscar2bit_gqa_prompt_test(1, 128, 8, 2, 64, 8, k_rho=0.96, v_rho=0.92)

    # ---- fp16 metadata (metadata_type="fp16") variants ----
    # Scale/zero stored as fp16 inline in each packed row (40B vs 48B for head_size=64,
    # group_size=8). atol is loosened slightly to absorb the fp16 rounding of scale/zero.

    def test_2bit_fp16meta_basic(self):
        run_oscar2bit_gqa_prompt_test(1, 8, 2, 2, 16, 8, atol=3e-3, meta_fp16=True)

    def test_2bit_fp16meta_gqa_ratio(self):
        run_oscar2bit_gqa_prompt_test(1, 8, 2, 1, 16, 8, atol=3e-3, meta_fp16=True)

    def test_2bit_fp16meta_multi_batch(self):
        run_oscar2bit_gqa_prompt_test(2, 12, 4, 2, 16, 8, atol=3e-3, meta_fp16=True)

    def test_2bit_fp16meta_one_group(self):
        run_oscar2bit_gqa_prompt_test(1, 8, 2, 2, 16, 16, atol=3e-3, meta_fp16=True)

    def test_2bit_fp16meta_eight_groups(self):
        run_oscar2bit_gqa_prompt_test(2, 10, 4, 2, 64, 8, atol=3e-3, meta_fp16=True)

    def test_2bit_fp16meta_rho_clip(self):
        run_oscar2bit_gqa_prompt_test(1, 10, 2, 2, 16, 8, k_rho=0.96, v_rho=0.92, atol=3e-3, meta_fp16=True)

    def test_2bit_fp16meta_decode_step(self):
        run_oscar2bit_gqa_decode_test(1, 6, 2, 2, 16, 8, atol=3e-3, meta_fp16=True)

    def test_2bit_fp16meta_decode_gqa_rho(self):
        run_oscar2bit_gqa_decode_test(2, 8, 4, 2, 128, 64, k_rho=0.96, v_rho=0.92, atol=3e-3, meta_fp16=True)

    # ---- fp16 compute dtype (io_fp16=True) variants ----
    # Q/K/V/output are MLFloat16; the kernel bridges half<->float around the float codec.
    # atol is loosened to absorb the fp16 rounding of the inputs and the output.

    def test_2bit_fp16_basic(self):
        run_oscar2bit_gqa_prompt_test(1, 8, 2, 2, 16, 8, atol=2e-2, io_fp16=True)

    def test_2bit_fp16_gqa_ratio(self):
        run_oscar2bit_gqa_prompt_test(1, 8, 2, 1, 16, 8, atol=2e-2, io_fp16=True)

    def test_2bit_fp16_multi_batch(self):
        run_oscar2bit_gqa_prompt_test(2, 12, 4, 2, 16, 8, atol=2e-2, io_fp16=True)

    def test_2bit_fp16_one_group(self):
        run_oscar2bit_gqa_prompt_test(1, 8, 2, 2, 16, 16, atol=2e-2, io_fp16=True)

    def test_2bit_fp16_eight_groups(self):
        run_oscar2bit_gqa_prompt_test(2, 10, 4, 2, 64, 8, atol=2e-2, io_fp16=True)

    def test_2bit_fp16_rho_clip(self):
        run_oscar2bit_gqa_prompt_test(1, 10, 2, 2, 16, 8, k_rho=0.96, v_rho=0.92, atol=2e-2, io_fp16=True)

    def test_2bit_fp16_large_head(self):
        run_oscar2bit_gqa_prompt_test(1, 6, 2, 2, 128, 64, k_rho=0.96, v_rho=0.92, atol=2e-2, io_fp16=True)

    def test_2bit_fp16_decode_step(self):
        run_oscar2bit_gqa_decode_test(1, 6, 2, 2, 16, 8, atol=2e-2, io_fp16=True)

    def test_2bit_fp16_decode_gqa_rho(self):
        # rho clipping + fp16 input rounding can nudge a value onto a 2-bit quantization
        # boundary where numpy's round-half-to-even and the kernel's rounding disagree by
        # one (large) level, so a couple of elements need a wider tolerance.
        run_oscar2bit_gqa_decode_test(2, 8, 4, 2, 128, 64, k_rho=0.96, v_rho=0.92, atol=6e-2, io_fp16=True)

    def test_2bit_fp16_and_fp16meta(self):
        """fp16 compute dtype combined with fp16 scale/zero metadata storage."""
        run_oscar2bit_gqa_prompt_test(2, 10, 4, 2, 64, 8, atol=2e-2, meta_fp16=True, io_fp16=True)

    def test_2bit_fp16_attention_bias_rejected(self):
        """The fp16 OSCAR path bridges half<->float and does not carry attention_bias, so the
        kernel must reject an fp16 graph that wires it rather than silently ignoring it."""
        batch_size, seq_len, num_heads, kv_num_heads, head_size, group_size = 1, 8, 2, 2, 16, 8
        hidden_size = num_heads * head_size
        kv_hidden_size = kv_num_heads * head_size
        phs = oscar2bit_packed_head_size(head_size, group_size)

        node = helper.make_node(
            op_type="MixedPrecisionGroupQueryAttention",
            inputs=[
                "query",
                "key",
                "value",
                "past_key",
                "past_value",
                "seqlens_k",
                "total_sequence_length",
                "",
                "",
                "",
                "attention_bias",  # attention_bias at index 10
            ],
            outputs=["output", "present_key", "present_value"],
            name="MixedPrecisionGroupQueryAttention_0",
            num_heads=num_heads,
            kv_num_heads=kv_num_heads,
            kv_quant_group_size=group_size,
            k_quant_rho=1.0,
            v_quant_rho=1.0,
            cache_format_version=1,
            domain="com.microsoft",
        )
        graph_input = [
            helper.make_tensor_value_info("query", TensorProto.FLOAT16, [batch_size, seq_len, hidden_size]),
            helper.make_tensor_value_info("key", TensorProto.FLOAT16, [batch_size, seq_len, kv_hidden_size]),
            helper.make_tensor_value_info("value", TensorProto.FLOAT16, [batch_size, seq_len, kv_hidden_size]),
            helper.make_tensor_value_info("past_key", TensorProto.UINT8, [batch_size, kv_num_heads, seq_len, phs]),
            helper.make_tensor_value_info("past_value", TensorProto.UINT8, [batch_size, kv_num_heads, seq_len, phs]),
            helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [batch_size]),
            helper.make_tensor_value_info("total_sequence_length", TensorProto.INT32, [1]),
            helper.make_tensor_value_info(
                "attention_bias", TensorProto.FLOAT16, [batch_size, num_heads, seq_len, seq_len]
            ),
        ]
        graph_output = [
            helper.make_tensor_value_info("output", TensorProto.FLOAT16, [batch_size, seq_len, hidden_size]),
            helper.make_tensor_value_info("present_key", TensorProto.UINT8, [batch_size, kv_num_heads, seq_len, phs]),
            helper.make_tensor_value_info("present_value", TensorProto.UINT8, [batch_size, kv_num_heads, seq_len, phs]),
        ]
        graph = helper.make_graph([node], "Oscar2BitGQAFp16Bias", graph_input, graph_output)
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
        )
        sess = InferenceSession(model.SerializeToString(), SessionOptions(), providers=["CPUExecutionProvider"])
        feeds = {
            "query": np.zeros((batch_size, seq_len, hidden_size), dtype=np.float16),
            "key": np.zeros((batch_size, seq_len, kv_hidden_size), dtype=np.float16),
            "value": np.zeros((batch_size, seq_len, kv_hidden_size), dtype=np.float16),
            "past_key": np.zeros((batch_size, kv_num_heads, seq_len, phs), dtype=np.uint8),
            "past_value": np.zeros((batch_size, kv_num_heads, seq_len, phs), dtype=np.uint8),
            "seqlens_k": np.array([seq_len - 1] * batch_size, dtype=np.int32),
            "total_sequence_length": np.array([seq_len], dtype=np.int32),
            "attention_bias": np.zeros((batch_size, num_heads, seq_len, seq_len), dtype=np.float16),
        }
        with self.assertRaises(Exception) as ctx:
            sess.run(None, feeds)
        self.assertIn("attention_bias", str(ctx.exception))


def _mixed_kv_ref(kv_bnsh, group_size, rho, sink, recent):
    """Reference for the OSCAR mixed-precision cache: sink (first) + recent (last) tokens
    stay exact FP; only the middle history is 2-bit quantize/dequantized."""
    deq = quantize_dequantize_oscar2bit(kv_bnsh, group_size, rho)
    t = kv_bnsh.shape[2]
    n_sink = min(sink, t)
    n_recent = min(recent, t - n_sink)
    out = deq.copy()
    if n_sink > 0:
        out[:, :, :n_sink, :] = kv_bnsh[:, :, :n_sink, :]
    if n_recent > 0:
        out[:, :, t - n_recent :, :] = kv_bnsh[:, :, t - n_recent :, :]
    return out


def run_oscar2bit_mixed_prompt_test(
    batch_size,
    seq_len,
    num_heads,
    kv_num_heads,
    head_size,
    group_size,
    sink,
    recent,
    k_rho=1.0,
    v_rho=1.0,
    atol=2e-3,
    io_fp16=False,
):
    """Prompt-phase parity for the mixed-precision cache: sink+recent exact, middle 2-bit."""
    np.random.seed(42)
    hidden_size = num_heads * head_size
    kv_hidden_size = kv_num_heads * head_size
    io_np = np.float16 if io_fp16 else np.float32

    query = np.random.uniform(-0.5, 0.5, (batch_size, seq_len, hidden_size)).astype(io_np)
    key_input = np.random.uniform(-0.5, 0.5, (batch_size, seq_len, kv_hidden_size)).astype(io_np)
    value_input = np.random.uniform(-0.5, 0.5, (batch_size, seq_len, kv_hidden_size)).astype(io_np)

    # The kernel bridges fp16 -> float, so the reference uses the fp16-rounded values.
    query_ref = query.astype(np.float32)
    k_bnsh = key_input.astype(np.float32).reshape(batch_size, seq_len, kv_num_heads, head_size).transpose(0, 2, 1, 3)
    v_bnsh = value_input.astype(np.float32).reshape(batch_size, seq_len, kv_num_heads, head_size).transpose(0, 2, 1, 3)

    phs = oscar2bit_packed_head_size(head_size, group_size)
    hp_present_len = min(seq_len, sink + recent)
    past_k = np.zeros((batch_size, kv_num_heads, seq_len, phs), dtype=np.uint8)
    past_v = np.zeros((batch_size, kv_num_heads, seq_len, phs), dtype=np.uint8)
    hp_empty = np.zeros((batch_size, kv_num_heads, 0, head_size), dtype=io_np)

    model = create_mixed_precision_gqa_graph(
        batch_size,
        seq_len,
        seq_len,
        seq_len,
        0,
        hp_present_len,
        num_heads,
        kv_num_heads,
        head_size,
        group_size,
        sink,
        recent,
        k_rho,
        v_rho,
        io_fp16=io_fp16,
    )
    sess = InferenceSession(model, SessionOptions(), providers=["CPUExecutionProvider"])
    out_ort = sess.run(
        None,
        {
            "query": query,
            "key": key_input,
            "value": value_input,
            "past_key": past_k,
            "past_value": past_v,
            "seqlens_k": np.array([seq_len - 1] * batch_size, dtype=np.int32),
            "total_sequence_length": np.array([seq_len], dtype=np.int32),
            "past_hp_key": hp_empty,
            "past_hp_value": hp_empty.copy(),
        },
    )[0].astype(np.float32)

    k_deq = _mixed_kv_ref(k_bnsh, group_size, k_rho, sink, recent)
    v_deq = _mixed_kv_ref(v_bnsh, group_size, v_rho, sink, recent)
    out_ref = reference_gqa(query_ref, k_deq, v_deq, num_heads, kv_num_heads, head_size, causal=True)

    if np.any(np.isnan(out_ort)):
        raise AssertionError(f"NaN in output (mixed 2-bit, sink={sink}, recent={recent})")
    np.testing.assert_allclose(
        out_ort,
        out_ref,
        atol=atol,
        rtol=0.1,
        err_msg=f"mixed 2-bit prompt mismatch (sink={sink}, recent={recent}, group_size={group_size})",
    )


def run_oscar2bit_mixed_decode_test(
    batch_size,
    prompt_len,
    num_heads,
    kv_num_heads,
    head_size,
    group_size,
    sink,
    recent,
    k_rho=1.0,
    v_rho=1.0,
    atol=2e-3,
    io_fp16=False,
):
    """Incremental-decode parity for the mixed cache: prompt then one decode step, feeding
    present + present_hp back as past + past_hp. Exercises the recent-window slide and the
    age-out re-quantization of the token demoted from the FP window into 2-bit history."""
    np.random.seed(7)
    hidden_size = num_heads * head_size
    kv_hidden_size = kv_num_heads * head_size
    s = prompt_len
    phs = oscar2bit_packed_head_size(head_size, group_size)
    hp_prompt_len = min(s, sink + recent)
    hp_decode_len = min(s + 1, sink + recent)
    io_np = np.float16 if io_fp16 else np.float32

    q1 = np.random.uniform(-0.5, 0.5, (batch_size, s, hidden_size)).astype(io_np)
    k1 = np.random.uniform(-0.5, 0.5, (batch_size, s, kv_hidden_size)).astype(io_np)
    v1 = np.random.uniform(-0.5, 0.5, (batch_size, s, kv_hidden_size)).astype(io_np)
    hp_empty = np.zeros((batch_size, kv_num_heads, 0, head_size), dtype=io_np)

    model1 = create_mixed_precision_gqa_graph(
        batch_size,
        s,
        s,
        s,
        0,
        hp_prompt_len,
        num_heads,
        kv_num_heads,
        head_size,
        group_size,
        sink,
        recent,
        k_rho,
        v_rho,
        io_fp16=io_fp16,
    )
    sess1 = InferenceSession(model1, SessionOptions(), providers=["CPUExecutionProvider"])
    out1 = sess1.run(
        None,
        {
            "query": q1,
            "key": k1,
            "value": v1,
            "past_key": np.zeros((batch_size, kv_num_heads, s, phs), dtype=np.uint8),
            "past_value": np.zeros((batch_size, kv_num_heads, s, phs), dtype=np.uint8),
            "seqlens_k": np.array([s - 1] * batch_size, dtype=np.int32),
            "total_sequence_length": np.array([s], dtype=np.int32),
            "past_hp_key": hp_empty,
            "past_hp_value": hp_empty.copy(),
        },
    )
    present_k, present_v, present_hp_k, present_hp_v = out1[1], out1[2], out1[3], out1[4]

    q2 = np.random.uniform(-0.5, 0.5, (batch_size, 1, hidden_size)).astype(io_np)
    k2 = np.random.uniform(-0.5, 0.5, (batch_size, 1, kv_hidden_size)).astype(io_np)
    v2 = np.random.uniform(-0.5, 0.5, (batch_size, 1, kv_hidden_size)).astype(io_np)

    model2 = create_mixed_precision_gqa_graph(
        batch_size,
        1,
        s,
        s + 1,
        hp_prompt_len,
        hp_decode_len,
        num_heads,
        kv_num_heads,
        head_size,
        group_size,
        sink,
        recent,
        k_rho,
        v_rho,
        io_fp16=io_fp16,
    )
    sess2 = InferenceSession(model2, SessionOptions(), providers=["CPUExecutionProvider"])
    out_ort = sess2.run(
        None,
        {
            "query": q2,
            "key": k2,
            "value": v2,
            "past_key": present_k,
            "past_value": present_v,
            "seqlens_k": np.array([s] * batch_size, dtype=np.int32),
            "total_sequence_length": np.array([s + 1], dtype=np.int32),
            "past_hp_key": present_hp_k,
            "past_hp_value": present_hp_v,
        },
    )[0].astype(np.float32)

    # Reference sees the fp16-rounded feeds promoted back to float32.
    k_full = np.concatenate([k1, k2], axis=1).astype(np.float32)
    v_full = np.concatenate([v1, v2], axis=1).astype(np.float32)
    k_bnsh = k_full.reshape(batch_size, s + 1, kv_num_heads, head_size).transpose(0, 2, 1, 3)
    v_bnsh = v_full.reshape(batch_size, s + 1, kv_num_heads, head_size).transpose(0, 2, 1, 3)
    k_deq = _mixed_kv_ref(k_bnsh, group_size, k_rho, sink, recent)
    v_deq = _mixed_kv_ref(v_bnsh, group_size, v_rho, sink, recent)
    out_ref = reference_gqa(q2.astype(np.float32), k_deq, v_deq, num_heads, kv_num_heads, head_size, causal=False)

    if np.any(np.isnan(out_ort)):
        raise AssertionError("NaN in output (mixed 2-bit decode)")
    np.testing.assert_allclose(
        out_ort,
        out_ref,
        atol=atol,
        rtol=0.1,
        err_msg=f"mixed 2-bit decode mismatch (sink={sink}, recent={recent}, group_size={group_size})",
    )


def _make_rotation(kv_num_heads, head_size, seed):
    """Per-kv-head random orthogonal matrices R [kv_num_heads, head_size, head_size] (float32)."""
    rng = np.random.default_rng(seed)
    rot = np.zeros((kv_num_heads, head_size, head_size), dtype=np.float32)
    for h in range(kv_num_heads):
        q, r = np.linalg.qr(rng.standard_normal((head_size, head_size)))
        q = q * np.sign(np.diag(r))  # fix sign ambiguity for a deterministic orthogonal matrix
        rot[h] = q.astype(np.float32)
    return rot


def _rotate_bnsh(x_bnsh, R):
    """x_bnsh [B, N, S, D] rotated per kv-head: x_hat[..., e] = sum_d x[..., d] R[n, d, e] = x @ R."""
    return np.einsum("bnsd,nde->bnse", x_bnsh, R).astype(np.float32)


def _rotate_q(q_input, R, num_heads, kv_num_heads, head_size):
    """q_input [B, S, num_heads*H] -> each query head rotated by its kv-head R (q @ R)."""
    b, s, _ = q_input.shape
    groups = num_heads // kv_num_heads
    q_bnsh = q_input.reshape(b, s, num_heads, head_size).transpose(0, 2, 1, 3)
    out = np.zeros_like(q_bnsh)
    for h in range(num_heads):
        out[:, h] = np.einsum("bsd,de->bse", q_bnsh[:, h], R[h // groups])
    return out.transpose(0, 2, 1, 3).reshape(b, s, num_heads * head_size).astype(np.float32)


def _unrotate_out(out_input, R_V, num_heads, kv_num_heads, head_size):
    """out_input [B, S, num_heads*H] (rotated basis) -> un-rotate per query head: out @ R_V^T."""
    b, s, _ = out_input.shape
    groups = num_heads // kv_num_heads
    o_bnsh = out_input.reshape(b, s, num_heads, head_size).transpose(0, 2, 1, 3)
    res = np.zeros_like(o_bnsh)
    for h in range(num_heads):
        res[:, h] = np.einsum("bse,de->bsd", o_bnsh[:, h], R_V[h // groups])
    return res.transpose(0, 2, 1, 3).reshape(b, s, num_heads * head_size).astype(np.float32)


def _canonical_mixed_rot_reference(
    q_input,
    k_bnsh,
    v_bnsh,
    R_K,
    R_V,
    num_heads,
    kv_num_heads,
    head_size,
    group_size,
    k_rho,
    v_rho,
    sink,
    recent,
):
    """Canonical OSCAR two-basis attention reference.

    Sink/recent rows remain in the original FP basis. Only the 2-bit history is
    rotated and quantized; its QK scores use q @ R_K and its value contribution
    is un-rotated by R_V.T after the shared softmax.
    """
    batch, seq_len, _ = q_input.shape
    groups = num_heads // kv_num_heads
    scale = 1.0 / math.sqrt(head_size)
    q_bnsh = q_input.reshape(batch, seq_len, num_heads, head_size).transpose(0, 2, 1, 3)
    output = np.zeros((batch, num_heads, seq_len, head_size), dtype=np.float32)

    for batch_index in range(batch):
        for head_index in range(num_heads):
            kv_head = head_index // groups
            k_rot = _rotate_bnsh(
                k_bnsh[batch_index : batch_index + 1, kv_head : kv_head + 1], R_K[kv_head : kv_head + 1]
            )[0, 0]
            v_rot = _rotate_bnsh(
                v_bnsh[batch_index : batch_index + 1, kv_head : kv_head + 1], R_V[kv_head : kv_head + 1]
            )[0, 0]
            k_hist = quantize_dequantize_oscar2bit(k_rot[None, None], group_size, k_rho)[0, 0]
            v_hist = quantize_dequantize_oscar2bit(v_rot[None, None], group_size, v_rho)[0, 0]
            # GroupQueryAttention builds the complete prompt cache once; each causal query
            # sees a prefix of that fixed sink/history/recent partition.
            n_sink = min(sink, seq_len)
            n_recent = min(recent, seq_len - n_sink)
            history_start = n_sink
            history_end = seq_len - n_recent

            for query_index in range(seq_len):
                visible = query_index + 1
                q = q_bnsh[batch_index, head_index, query_index]
                q_rot = q @ R_K[kv_head]
                logits = np.empty(visible, dtype=np.float32)

                sink_visible = min(n_sink, visible)
                history_visible_end = min(history_end, visible)
                if sink_visible:
                    logits[:sink_visible] = k_bnsh[batch_index, kv_head, :sink_visible] @ q * scale
                if history_visible_end > history_start:
                    logits[history_start:history_visible_end] = (
                        k_hist[history_start:history_visible_end] @ q_rot * scale
                    )
                if visible > history_end:
                    logits[history_end:visible] = k_bnsh[batch_index, kv_head, history_end:visible] @ q * scale

                probs = np.exp(logits - np.max(logits))
                probs /= np.sum(probs)
                context = np.zeros(head_size, dtype=np.float32)
                if sink_visible:
                    context += probs[:sink_visible] @ v_bnsh[batch_index, kv_head, :sink_visible]
                if history_visible_end > history_start:
                    context_rot = probs[history_start:history_visible_end] @ v_hist[history_start:history_visible_end]
                    context += context_rot @ R_V[kv_head].T
                if visible > history_end:
                    context += probs[history_end:visible] @ v_bnsh[batch_index, kv_head, history_end:visible]
                output[batch_index, head_index, query_index] = context

    return output.transpose(0, 2, 1, 3).reshape(batch, seq_len, num_heads * head_size)


def create_oscar2bit_mixed_rot_gqa_graph(
    batch_size,
    q_len,
    past_len,
    present_len,
    hp_past_len,
    hp_present_len,
    num_heads,
    kv_num_heads,
    head_size,
    group_size,
    sink,
    recent,
    k_rho,
    v_rho,
):
    """Like create_mixed_precision_gqa_graph but also wires oscar_rotation_k/v at inputs 18/19."""
    hidden_size = num_heads * head_size
    kv_hidden_size = kv_num_heads * head_size
    phs = oscar2bit_packed_head_size(head_size, group_size)

    node = helper.make_node(
        op_type="MixedPrecisionGroupQueryAttention",
        inputs=[
            "query",
            "key",
            "value",
            "past_key",
            "past_value",
            "seqlens_k",
            "total_sequence_length",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",  # indices 7..15 (unused optional inputs)
            "past_hp_key",
            "past_hp_value",  # indices 16, 17
            "oscar_rotation_k",
            "oscar_rotation_v",  # indices 18, 19
        ],
        outputs=["output", "present_key", "present_value", "", "present_hp_key", "present_hp_value"],
        name="MixedPrecisionGroupQueryAttention_0",
        num_heads=num_heads,
        kv_num_heads=kv_num_heads,
        kv_quant_group_size=group_size,
        k_quant_rho=float(k_rho),
        v_quant_rho=float(v_rho),
        sink_size=int(sink),
        recent_size=int(recent),
        cache_format_version=1,
        domain="com.microsoft",
    )

    graph_input = [
        helper.make_tensor_value_info("query", TensorProto.FLOAT, [batch_size, q_len, hidden_size]),
        helper.make_tensor_value_info("key", TensorProto.FLOAT, [batch_size, q_len, kv_hidden_size]),
        helper.make_tensor_value_info("value", TensorProto.FLOAT, [batch_size, q_len, kv_hidden_size]),
        helper.make_tensor_value_info("past_key", TensorProto.UINT8, [batch_size, kv_num_heads, past_len, phs]),
        helper.make_tensor_value_info("past_value", TensorProto.UINT8, [batch_size, kv_num_heads, past_len, phs]),
        helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [batch_size]),
        helper.make_tensor_value_info("total_sequence_length", TensorProto.INT32, [1]),
        helper.make_tensor_value_info(
            "past_hp_key", TensorProto.FLOAT, [batch_size, kv_num_heads, hp_past_len, head_size]
        ),
        helper.make_tensor_value_info(
            "past_hp_value", TensorProto.FLOAT, [batch_size, kv_num_heads, hp_past_len, head_size]
        ),
        helper.make_tensor_value_info("oscar_rotation_k", TensorProto.FLOAT, [kv_num_heads, head_size, head_size]),
        helper.make_tensor_value_info("oscar_rotation_v", TensorProto.FLOAT, [kv_num_heads, head_size, head_size]),
    ]
    graph_output = [
        helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch_size, q_len, hidden_size]),
        helper.make_tensor_value_info("present_key", TensorProto.UINT8, [batch_size, kv_num_heads, present_len, phs]),
        helper.make_tensor_value_info("present_value", TensorProto.UINT8, [batch_size, kv_num_heads, present_len, phs]),
        helper.make_tensor_value_info(
            "present_hp_key", TensorProto.FLOAT, [batch_size, kv_num_heads, hp_present_len, head_size]
        ),
        helper.make_tensor_value_info(
            "present_hp_value", TensorProto.FLOAT, [batch_size, kv_num_heads, hp_present_len, head_size]
        ),
    ]

    graph = helper.make_graph([node], "Oscar2BitMixedRotGQA_Graph", graph_input, graph_output)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
    )
    return model.SerializeToString()


def run_oscar2bit_mixed_rot_prompt_test(
    batch_size,
    seq_len,
    num_heads,
    kv_num_heads,
    head_size,
    group_size,
    sink,
    recent,
    k_rho=1.0,
    v_rho=1.0,
    atol=2e-3,
):
    """Prompt-phase parity for the mixed cache WITH the OSCAR spectral rotation (R_K/R_V).

    Fused reference: rotate q/k/v by R (post-RoPE), quantize the sink/recent+history cache in the
    rotated basis, run attention, then un-rotate the V-side output by R_V^T. Because R_K is
    orthogonal the QK scores are basis-invariant; the rotation only reshapes the per-group
    quantization error of the 2-bit history."""
    np.random.seed(123)
    hidden_size = num_heads * head_size
    kv_hidden_size = kv_num_heads * head_size

    query = np.random.uniform(-0.5, 0.5, (batch_size, seq_len, hidden_size)).astype(np.float32)
    key_input = np.random.uniform(-0.5, 0.5, (batch_size, seq_len, kv_hidden_size)).astype(np.float32)
    value_input = np.random.uniform(-0.5, 0.5, (batch_size, seq_len, kv_hidden_size)).astype(np.float32)

    k_bnsh = key_input.reshape(batch_size, seq_len, kv_num_heads, head_size).transpose(0, 2, 1, 3)
    v_bnsh = value_input.reshape(batch_size, seq_len, kv_num_heads, head_size).transpose(0, 2, 1, 3)

    r_k = _make_rotation(kv_num_heads, head_size, seed=1)
    r_v = _make_rotation(kv_num_heads, head_size, seed=2)

    phs = oscar2bit_packed_head_size(head_size, group_size)
    hp_present_len = min(seq_len, sink + recent)
    past_k = np.zeros((batch_size, kv_num_heads, seq_len, phs), dtype=np.uint8)
    past_v = np.zeros((batch_size, kv_num_heads, seq_len, phs), dtype=np.uint8)
    hp_empty = np.zeros((batch_size, kv_num_heads, 0, head_size), dtype=np.float32)

    model = create_oscar2bit_mixed_rot_gqa_graph(
        batch_size,
        seq_len,
        seq_len,
        seq_len,
        0,
        hp_present_len,
        num_heads,
        kv_num_heads,
        head_size,
        group_size,
        sink,
        recent,
        k_rho,
        v_rho,
    )
    sess = InferenceSession(model, SessionOptions(), providers=["CPUExecutionProvider"])
    out_ort = sess.run(
        None,
        {
            "query": query,
            "key": key_input,
            "value": value_input,
            "past_key": past_k,
            "past_value": past_v,
            "seqlens_k": np.array([seq_len - 1] * batch_size, dtype=np.int32),
            "total_sequence_length": np.array([seq_len], dtype=np.int32),
            "past_hp_key": hp_empty,
            "past_hp_value": hp_empty.copy(),
            "oscar_rotation_k": r_k,
            "oscar_rotation_v": r_v,
        },
    )[0]

    out_ref = _canonical_mixed_rot_reference(
        query,
        k_bnsh,
        v_bnsh,
        r_k,
        r_v,
        num_heads,
        kv_num_heads,
        head_size,
        group_size,
        k_rho,
        v_rho,
        sink,
        recent,
    )

    if np.any(np.isnan(out_ort)):
        raise AssertionError(f"NaN in output (mixed 2-bit rotated, sink={sink}, recent={recent})")
    np.testing.assert_allclose(
        out_ort,
        out_ref,
        atol=atol,
        rtol=0.1,
        err_msg=f"mixed 2-bit rotated prompt mismatch (sink={sink}, recent={recent}, group_size={group_size})",
    )


def run_oscar2bit_mixed_rot_identity_test(
    batch_size,
    seq_len,
    num_heads,
    kv_num_heads,
    head_size,
    group_size,
    sink,
    recent,
    k_rho=1.0,
    v_rho=1.0,
    atol=1e-5,
):
    """Identity-rotation sanity: passing R = I must reproduce the no-rotation mixed output exactly."""
    np.random.seed(321)
    hidden_size = num_heads * head_size
    kv_hidden_size = kv_num_heads * head_size

    query = np.random.uniform(-0.5, 0.5, (batch_size, seq_len, hidden_size)).astype(np.float32)
    key_input = np.random.uniform(-0.5, 0.5, (batch_size, seq_len, kv_hidden_size)).astype(np.float32)
    value_input = np.random.uniform(-0.5, 0.5, (batch_size, seq_len, kv_hidden_size)).astype(np.float32)

    eye = np.broadcast_to(np.eye(head_size, dtype=np.float32), (kv_num_heads, head_size, head_size)).copy()
    phs = oscar2bit_packed_head_size(head_size, group_size)
    hp_present_len = min(seq_len, sink + recent)
    past_k = np.zeros((batch_size, kv_num_heads, seq_len, phs), dtype=np.uint8)
    past_v = np.zeros((batch_size, kv_num_heads, seq_len, phs), dtype=np.uint8)
    hp_empty = np.zeros((batch_size, kv_num_heads, 0, head_size), dtype=np.float32)

    inputs_common = {
        "query": query,
        "key": key_input,
        "value": value_input,
        "past_key": past_k,
        "past_value": past_v,
        "seqlens_k": np.array([seq_len - 1] * batch_size, dtype=np.int32),
        "total_sequence_length": np.array([seq_len], dtype=np.int32),
        "past_hp_key": hp_empty,
        "past_hp_value": hp_empty.copy(),
    }

    rot_model = create_oscar2bit_mixed_rot_gqa_graph(
        batch_size,
        seq_len,
        seq_len,
        seq_len,
        0,
        hp_present_len,
        num_heads,
        kv_num_heads,
        head_size,
        group_size,
        sink,
        recent,
        k_rho,
        v_rho,
    )
    out_rot = InferenceSession(rot_model, SessionOptions(), providers=["CPUExecutionProvider"]).run(
        None, {**inputs_common, "oscar_rotation_k": eye, "oscar_rotation_v": eye}
    )[0]

    plain_model = create_mixed_precision_gqa_graph(
        batch_size,
        seq_len,
        seq_len,
        seq_len,
        0,
        hp_present_len,
        num_heads,
        kv_num_heads,
        head_size,
        group_size,
        sink,
        recent,
        k_rho,
        v_rho,
    )
    out_plain = InferenceSession(plain_model, SessionOptions(), providers=["CPUExecutionProvider"]).run(
        None, inputs_common
    )[0]

    np.testing.assert_allclose(
        out_rot,
        out_plain,
        atol=atol,
        rtol=0,
        err_msg="identity rotation must match the no-rotation mixed output",
    )


def run_oscar2bit_mixed_rot_decode_test(
    batch_size,
    prompt_len,
    num_heads,
    kv_num_heads,
    head_size,
    group_size,
    sink,
    recent,
    k_rho=1.0,
    v_rho=1.0,
    atol=2e-3,
):
    """Incremental-decode parity WITH the OSCAR spectral rotation: prompt then one decode step,
    feeding present + present_hp AND both rotation matrices back into the next step. Unlike the
    prompt-only rotation tests, this exercises reuse of the already-rotated 2-bit history plus the
    rotation of the recent token that ages into the 2-bit cache when the window slides -- the
    stateful BuildMixedHeadCache branches the prompt-only tests never reach."""
    np.random.seed(11)
    hidden_size = num_heads * head_size
    kv_hidden_size = kv_num_heads * head_size
    s = prompt_len
    phs = oscar2bit_packed_head_size(head_size, group_size)
    hp_prompt_len = min(s, sink + recent)
    hp_decode_len = min(s + 1, sink + recent)

    r_k = _make_rotation(kv_num_heads, head_size, seed=1)
    r_v = _make_rotation(kv_num_heads, head_size, seed=2)

    q1 = np.random.uniform(-0.5, 0.5, (batch_size, s, hidden_size)).astype(np.float32)
    k1 = np.random.uniform(-0.5, 0.5, (batch_size, s, kv_hidden_size)).astype(np.float32)
    v1 = np.random.uniform(-0.5, 0.5, (batch_size, s, kv_hidden_size)).astype(np.float32)
    hp_empty = np.zeros((batch_size, kv_num_heads, 0, head_size), dtype=np.float32)

    # Step 1: prompt. Build the rotated 2-bit history + FP sink/recent window in one call.
    model1 = create_oscar2bit_mixed_rot_gqa_graph(
        batch_size,
        s,
        s,
        s,
        0,
        hp_prompt_len,
        num_heads,
        kv_num_heads,
        head_size,
        group_size,
        sink,
        recent,
        k_rho,
        v_rho,
    )
    sess1 = InferenceSession(model1, SessionOptions(), providers=["CPUExecutionProvider"])
    out1 = sess1.run(
        None,
        {
            "query": q1,
            "key": k1,
            "value": v1,
            "past_key": np.zeros((batch_size, kv_num_heads, s, phs), dtype=np.uint8),
            "past_value": np.zeros((batch_size, kv_num_heads, s, phs), dtype=np.uint8),
            "seqlens_k": np.array([s - 1] * batch_size, dtype=np.int32),
            "total_sequence_length": np.array([s], dtype=np.int32),
            "past_hp_key": hp_empty,
            "past_hp_value": hp_empty.copy(),
            "oscar_rotation_k": r_k,
            "oscar_rotation_v": r_v,
        },
    )
    present_k, present_v, present_hp_k, present_hp_v = out1[1], out1[2], out1[3], out1[4]

    q2 = np.random.uniform(-0.5, 0.5, (batch_size, 1, hidden_size)).astype(np.float32)
    k2 = np.random.uniform(-0.5, 0.5, (batch_size, 1, kv_hidden_size)).astype(np.float32)
    v2 = np.random.uniform(-0.5, 0.5, (batch_size, 1, kv_hidden_size)).astype(np.float32)

    # Step 2: decode. Feed present + present_hp back as past + past_hp, with the same rotations.
    model2 = create_oscar2bit_mixed_rot_gqa_graph(
        batch_size,
        1,
        s,
        s + 1,
        hp_prompt_len,
        hp_decode_len,
        num_heads,
        kv_num_heads,
        head_size,
        group_size,
        sink,
        recent,
        k_rho,
        v_rho,
    )
    sess2 = InferenceSession(model2, SessionOptions(), providers=["CPUExecutionProvider"])
    out_ort = sess2.run(
        None,
        {
            "query": q2,
            "key": k2,
            "value": v2,
            "past_key": present_k,
            "past_value": present_v,
            "seqlens_k": np.array([s] * batch_size, dtype=np.int32),
            "total_sequence_length": np.array([s + 1], dtype=np.int32),
            "past_hp_key": present_hp_k,
            "past_hp_value": present_hp_v,
            "oscar_rotation_k": r_k,
            "oscar_rotation_v": r_v,
        },
    )[0]

    # Reference: build the full (s + 1) rotated/quantized cache once and read the last (decode)
    # query. Re-quantizing an aged-out recent token is deterministic, so the prompt-then-decode
    # cache equals a single fixed s+1 partition -- attention is causal and per-query independent.
    q_full = np.concatenate([q1, q2], axis=1)
    k_full = np.concatenate([k1, k2], axis=1)
    v_full = np.concatenate([v1, v2], axis=1)
    k_bnsh = k_full.reshape(batch_size, s + 1, kv_num_heads, head_size).transpose(0, 2, 1, 3)
    v_bnsh = v_full.reshape(batch_size, s + 1, kv_num_heads, head_size).transpose(0, 2, 1, 3)
    out_ref = _canonical_mixed_rot_reference(
        q_full, k_bnsh, v_bnsh, r_k, r_v, num_heads, kv_num_heads, head_size, group_size, k_rho, v_rho, sink, recent
    )[:, -1:, :]

    if np.any(np.isnan(out_ort)):
        raise AssertionError("NaN in output (mixed 2-bit rotated decode)")
    np.testing.assert_allclose(
        out_ort,
        out_ref,
        atol=atol,
        rtol=0.1,
        err_msg=f"mixed 2-bit rotated decode mismatch (sink={sink}, recent={recent}, group_size={group_size})",
    )


class TestMixedPrecisionGQAOscar2Bit(unittest.TestCase):
    """com.microsoft.MixedPrecisionGroupQueryAttention: OSCAR 2-bit KV cache with the
    mixed-precision sink/recent window (sink_size / recent_size attributes)."""

    def test_mixed_middle_history(self):
        """seq > sink+recent so a 2-bit middle exists."""
        run_oscar2bit_mixed_prompt_test(1, 16, 2, 2, 128, 64, sink=4, recent=4)

    def test_mixed_all_high_precision(self):
        """seq <= sink+recent: every token is FP, so the output matches unquantized attention."""
        run_oscar2bit_mixed_prompt_test(1, 6, 2, 2, 128, 64, sink=4, recent=4)

    def test_mixed_sink_only(self):
        run_oscar2bit_mixed_prompt_test(1, 12, 2, 2, 128, 64, sink=4, recent=0)

    def test_mixed_recent_only(self):
        run_oscar2bit_mixed_prompt_test(1, 12, 2, 2, 128, 64, sink=0, recent=4)

    def test_mixed_gqa_ratio_rho(self):
        run_oscar2bit_mixed_prompt_test(2, 20, 4, 2, 128, 64, sink=4, recent=8, k_rho=0.96, v_rho=0.92)

    def test_mixed_two_groups(self):
        run_oscar2bit_mixed_prompt_test(1, 16, 2, 2, 32, 16, sink=2, recent=4)

    def test_mixed_decode_step(self):
        run_oscar2bit_mixed_decode_test(1, 10, 2, 2, 128, 64, sink=2, recent=4)

    def test_mixed_decode_gqa_rho(self):
        run_oscar2bit_mixed_decode_test(2, 12, 4, 2, 128, 64, sink=4, recent=4, k_rho=0.96, v_rho=0.92)

    def test_mixed_decode_all_hp(self):
        """Decode while still inside the FP window (no history yet)."""
        run_oscar2bit_mixed_decode_test(1, 5, 2, 2, 128, 64, sink=4, recent=4)

    # ---- fp16 compute dtype (io_fp16=True) variants ----
    # Q/K/V/output AND the high-precision sink/recent window are MLFloat16; the kernel
    # bridges the hp window (past_hp half->float, present_hp float->half) around the codec.

    def test_mixed_fp16_middle_history(self):
        run_oscar2bit_mixed_prompt_test(1, 16, 2, 2, 128, 64, sink=4, recent=4, atol=2e-2, io_fp16=True)

    def test_mixed_fp16_all_high_precision(self):
        run_oscar2bit_mixed_prompt_test(1, 6, 2, 2, 128, 64, sink=4, recent=4, atol=2e-2, io_fp16=True)

    def test_mixed_fp16_sink_only(self):
        run_oscar2bit_mixed_prompt_test(1, 12, 2, 2, 128, 64, sink=4, recent=0, atol=2e-2, io_fp16=True)

    def test_mixed_fp16_recent_only(self):
        run_oscar2bit_mixed_prompt_test(1, 12, 2, 2, 128, 64, sink=0, recent=4, atol=2e-2, io_fp16=True)

    def test_mixed_fp16_gqa_ratio_rho(self):
        run_oscar2bit_mixed_prompt_test(
            2, 20, 4, 2, 128, 64, sink=4, recent=8, k_rho=0.96, v_rho=0.92, atol=2e-2, io_fp16=True
        )

    def test_mixed_fp16_decode_step(self):
        run_oscar2bit_mixed_decode_test(1, 10, 2, 2, 128, 64, sink=2, recent=4, atol=2e-2, io_fp16=True)

    def test_mixed_fp16_decode_gqa_rho(self):
        run_oscar2bit_mixed_decode_test(
            2, 12, 4, 2, 128, 64, sink=4, recent=4, k_rho=0.96, v_rho=0.92, atol=2e-2, io_fp16=True
        )

    def test_mixed_fp16_decode_all_hp(self):
        run_oscar2bit_mixed_decode_test(1, 5, 2, 2, 128, 64, sink=4, recent=4, atol=2e-2, io_fp16=True)


class TestMixedPrecisionGQAOscar2BitRotation(unittest.TestCase):
    """com.microsoft.MixedPrecisionGroupQueryAttention: OSCAR 2-bit KV cache with in-kernel
    post-RoPE R_K/R_V rotation."""

    def test_rotation_identity_matches_plain(self):
        """R = I must reproduce the un-rotated mixed output bit-for-bit."""
        run_oscar2bit_mixed_rot_identity_test(1, 16, 2, 2, 128, 64, sink=4, recent=4)

    def test_rotation_identity_two_groups(self):
        run_oscar2bit_mixed_rot_identity_test(1, 12, 2, 2, 32, 16, sink=2, recent=4)

    def test_rotation_middle_history(self):
        """seq > sink+recent so a rotated 2-bit middle exists."""
        run_oscar2bit_mixed_rot_prompt_test(1, 16, 2, 2, 128, 64, sink=4, recent=4)

    def test_rotation_gqa_ratio_rho(self):
        run_oscar2bit_mixed_rot_prompt_test(2, 20, 4, 2, 128, 64, sink=4, recent=8, k_rho=0.96, v_rho=0.92)

    def test_rotation_sink_only(self):
        run_oscar2bit_mixed_rot_prompt_test(1, 12, 2, 2, 128, 64, sink=4, recent=0)

    def test_rotation_recent_only(self):
        run_oscar2bit_mixed_rot_prompt_test(1, 12, 2, 2, 128, 64, sink=0, recent=4)

    def test_rotation_two_groups(self):
        run_oscar2bit_mixed_rot_prompt_test(1, 16, 2, 2, 32, 16, sink=2, recent=4)

    def test_rotation_decode_step(self):
        """Decode after a prompt, feeding present/present_hp + both rotations into the next step."""
        run_oscar2bit_mixed_rot_decode_test(1, 10, 2, 2, 128, 64, sink=2, recent=4)

    def test_rotation_decode_gqa_rho(self):
        run_oscar2bit_mixed_rot_decode_test(2, 12, 4, 2, 128, 64, sink=4, recent=4, k_rho=0.96, v_rho=0.92)

    def test_rotation_decode_two_groups(self):
        run_oscar2bit_mixed_rot_decode_test(1, 12, 2, 2, 32, 16, sink=2, recent=4)


def create_mixed_precision_gqa_graph(
    batch_size,
    q_len,
    past_len,
    present_len,
    hp_past_len,
    hp_present_len,
    num_heads,
    kv_num_heads,
    head_size,
    group_size,
    sink,
    recent,
    k_rho=1.0,
    v_rho=1.0,
    metadata_type="fp32",
    io_fp16=False,
):
    """com.microsoft.MixedPrecisionGroupQueryAttention graph for the OSCAR mixed-precision cache:
    2-bit history in present_{key,value} plus a high-precision FP window in present_hp_{key,value}
    (I/O at node input indices 16/17 and output indices 4/5). INT2/PER_GROUP is inherent and the
    sink/recent window sizes are node attributes (sink_size / recent_size)."""
    hidden_size = num_heads * head_size
    kv_hidden_size = kv_num_heads * head_size
    phs = oscar2bit_packed_head_size(head_size, group_size)
    io_dtype = TensorProto.FLOAT16 if io_fp16 else TensorProto.FLOAT

    node = helper.make_node(
        op_type="MixedPrecisionGroupQueryAttention",
        inputs=[
            "query",
            "key",
            "value",
            "past_key",
            "past_value",
            "seqlens_k",
            "total_sequence_length",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",  # indices 7..15 (unused optional inputs)
            "past_hp_key",
            "past_hp_value",  # indices 16, 17
        ],
        outputs=["output", "present_key", "present_value", "", "present_hp_key", "present_hp_value"],
        name="MixedPrecisionGroupQueryAttention_0",
        num_heads=num_heads,
        kv_num_heads=kv_num_heads,
        kv_quant_group_size=group_size,
        k_quant_rho=float(k_rho),
        v_quant_rho=float(v_rho),
        sink_size=int(sink),
        recent_size=int(recent),
        metadata_type=metadata_type,
        cache_format_version=1,
        domain="com.microsoft",
    )

    graph_input = [
        helper.make_tensor_value_info("query", io_dtype, [batch_size, q_len, hidden_size]),
        helper.make_tensor_value_info("key", io_dtype, [batch_size, q_len, kv_hidden_size]),
        helper.make_tensor_value_info("value", io_dtype, [batch_size, q_len, kv_hidden_size]),
        helper.make_tensor_value_info("past_key", TensorProto.UINT8, [batch_size, kv_num_heads, past_len, phs]),
        helper.make_tensor_value_info("past_value", TensorProto.UINT8, [batch_size, kv_num_heads, past_len, phs]),
        helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [batch_size]),
        helper.make_tensor_value_info("total_sequence_length", TensorProto.INT32, [1]),
        helper.make_tensor_value_info("past_hp_key", io_dtype, [batch_size, kv_num_heads, hp_past_len, head_size]),
        helper.make_tensor_value_info("past_hp_value", io_dtype, [batch_size, kv_num_heads, hp_past_len, head_size]),
    ]
    graph_output = [
        helper.make_tensor_value_info("output", io_dtype, [batch_size, q_len, hidden_size]),
        helper.make_tensor_value_info("present_key", TensorProto.UINT8, [batch_size, kv_num_heads, present_len, phs]),
        helper.make_tensor_value_info("present_value", TensorProto.UINT8, [batch_size, kv_num_heads, present_len, phs]),
        helper.make_tensor_value_info(
            "present_hp_key", io_dtype, [batch_size, kv_num_heads, hp_present_len, head_size]
        ),
        helper.make_tensor_value_info(
            "present_hp_value", io_dtype, [batch_size, kv_num_heads, hp_present_len, head_size]
        ),
    ]

    graph = helper.make_graph([node], "MixedPrecisionGQA_Graph", graph_input, graph_output)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
    )
    return model.SerializeToString()


class TestMixedPrecisionGroupQueryAttention(unittest.TestCase):
    """com.microsoft.MixedPrecisionGroupQueryAttention operator-config validation.

    Numeric correctness of the attribute-driven OSCAR 2-bit cache is covered by
    TestMixedPrecisionGQAOscar2Bit and TestMixedPrecisionGQAOscar2BitRotation; this class checks
    the op-specific attribute enforcement (cache_format_version / metadata_type)."""

    def test_bad_cache_format_version_rejected(self):
        model = create_mixed_precision_gqa_graph(1, 8, 8, 8, 0, 4, 2, 2, 128, 64, sink=2, recent=2)
        # Patch cache_format_version to an unsupported value.
        m = onnx.load_model_from_string(model)
        for attr in m.graph.node[0].attribute:
            if attr.name == "cache_format_version":
                attr.i = 2
        with self.assertRaises(Fail):
            InferenceSession(m.SerializeToString(), SessionOptions(), providers=["CPUExecutionProvider"])

    def test_bad_metadata_type_rejected(self):
        model = create_mixed_precision_gqa_graph(
            1, 8, 8, 8, 0, 4, 2, 2, 128, 64, sink=2, recent=2, metadata_type="bf16"
        )
        with self.assertRaises(Fail):
            InferenceSession(model, SessionOptions(), providers=["CPUExecutionProvider"])


if __name__ == "__main__":
    unittest.main()
