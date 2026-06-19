# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for CPU GroupQueryAttention with quantized KV cache (INT8/INT4)."""

import math
import unittest

import numpy as np
from onnx import TensorProto, helper

from onnxruntime import InferenceSession, SessionOptions

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
    if atol is None:
        atol = 0.15 if bit_width == 4 else 0.05

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

    if atol is None:
        atol = 0.15 if bit_width == 4 else 0.05

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

    if atol is None:
        atol = 0.15 if bit_width == 4 else 0.05

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


if __name__ == "__main__":
    unittest.main()
