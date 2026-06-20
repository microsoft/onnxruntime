# GroupQueryAttention — Operator Documentation

This document describes the `com.microsoft::GroupQueryAttention` (GQA) contrib operator: its schema,
the CUDA kernel backends and how one is selected, and the attention-sink (`head_sink`) decode path
that is accelerated by the XQA kernel.

For CPU-specific implementation details (including the quantized KV-cache flash path), see
[cpu/gqa.md](cpu/gqa.md).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Operator Schema](#2-operator-schema)
3. [KV Cache and Quantization](#3-kv-cache-and-quantization)
4. [Attention Sink (`head_sink`) and Smooth Softmax](#4-attention-sink-head_sink-and-smooth-softmax)
5. [CUDA Kernel Backends and Dispatch](#5-cuda-kernel-backends-and-dispatch)
6. [XQA Decode Path](#6-xqa-decode-path)
7. [XQA `head_sink` PrePack](#7-xqa-head_sink-prepack)
8. [Environment Variables](#8-environment-variables)
9. [Testing](#9-testing)

---

## 1. Overview

GroupQueryAttention implements causal grouped-query attention with KV-cache (past/present) support.
Grouped-query attention uses fewer key/value heads than query heads: each KV head is shared by a
group of `num_heads / kv_num_heads` query heads. The operator also supports:

- Rotary positional embeddings (RoPE)
- Past/present KV cache with optional in-place (shared) buffer
- Quantized KV cache (int4 / int8 / float8e4m3fn) to reduce memory footprint
- Optional attention bias and local (sliding) window attention
- Smooth softmax, including a per-head attention sink (`head_sink`)

The operator schema is defined in
[onnxruntime/core/graph/contrib_ops/bert_defs.cc](../../onnxruntime/core/graph/contrib_ops/bert_defs.cc).
The CUDA kernel is implemented in
[onnxruntime/contrib_ops/cuda/bert/group_query_attention.cc](../../onnxruntime/contrib_ops/cuda/bert/group_query_attention.cc)
and [group_query_attention_impl.cu](../../onnxruntime/contrib_ops/cuda/bert/group_query_attention_impl.cu).

## 2. Operator Schema

Selected attributes:

| Attribute | Description |
|-----------|-------------|
| `num_heads` | Number of query heads. |
| `kv_num_heads` | Number of key/value heads. `num_heads % kv_num_heads == 0`. |
| `scale` | Softmax scale. Defaults to `1/sqrt(head_size)`. |
| `softcap` | Optional logit soft-capping value. `0` disables it. |
| `local_window_size` | Left window size for local attention. `-1` means global attention. |
| `do_rotary` / `rotary_interleaved` | Enable RoPE and select interleaved vs. half-rotary layout. |
| `smooth_softmax` | Add a smooth factor to the softmax denominator. |
| `k_quant_type` / `v_quant_type` | KV cache quantization mode: `NONE`, `PER_TENSOR`, or `PER_CHANNEL`. |
| `kv_cache_bit_width` | Bit width of the quantized KV cache (`8` or `4`). |

Selected inputs (see the schema for the full list and shapes):

| Index | Name | Notes |
|-------|------|-------|
| 0 | `query` | `(batch, seq, hidden)`, or packed QKV. |
| 1, 2 | `key`, `value` | Optional when QKV is packed into `query`. |
| 3, 4 | `past_key`, `past_value` | BNSH cache. Shares the buffer with `present_*` when in-place. |
| 5 | `seqlens_k` | `total_sequence_lengths - 1` per batch entry. |
| 6 | `total_sequence_length` | Scalar used to distinguish prompt vs. decode. |
| 7, 8 | `cos_cache`, `sin_cache` | RoPE caches. |
| 11 | `head_sink` | `(num_heads,)` per-head attention sink (see §4). |
| 12, 13 | `k_scale`, `v_scale` | FP32 dequant scales for the quantized KV cache. |

Outputs are `output`, `present_key`, `present_value`, and optional `output_qk`.

## 3. KV Cache and Quantization

The past/present KV cache uses BNSH layout `(batch_size, kv_num_heads, cache_sequence_length, head_size)`.
When `past_present_share_buffer` holds (the past and present tensors alias the same memory), the cache
length is the maximum sequence length and new keys/values are appended in place. This shared-buffer mode
is required by the XQA decode path.

When quantization is enabled, `k_quant_type` and `v_quant_type` select `PER_TENSOR` or `PER_CHANNEL`
scaling, and `kv_cache_bit_width` selects 8-bit or 4-bit storage. The `k_scale` / `v_scale` inputs are
always FP32.

## 4. Attention Sink (`head_sink`) and Smooth Softmax

An attention sink adds a learned per-head bias term to the softmax denominator. With sink value `s_h`
for head `h`, the attention weights over `T` cached positions become:

$$
\text{softmax}_i = \frac{e^{x_i - m}}{e^{s_h - m} + \sum_{j} e^{x_j - m}}, \quad m = \max\left(s_h, \max_j x_j\right)
$$

This is equivalent to appending a single extra logit `s_h` (whose value contributes nothing to the
output, only to normalization). GPT-OSS style models use this to let a head attend to "nothing".

In the kernel, providing the `head_sink` input is treated as smooth softmax:
`parameters.use_smooth_softmax = use_smooth_softmax_ || head_sink != nullptr`. The `head_sink` tensor is
1D of shape `(num_heads,)` and matches the operator's floating-point type (`float16` or `bfloat16` on
the XQA path).

## 5. CUDA Kernel Backends and Dispatch

The CUDA EP can route a GQA node to several backends. At runtime it selects the first eligible one:

| Backend | Typical use |
|---------|-------------|
| **XQA** | Single-token global decode (`seq_len == 1`), shared KV buffer. Fastest decode path. |
| **Flash Attention / Flash Decoding** | General prompt and decode, including local window and softcap. |
| **cuDNN SDPA** | Preferred on SM≥90 for non-quantized FP16/BF16 causal attention. |
| **Memory Efficient Attention** | Fallback for FP16/FP32 (and BF16 on SM80+). |
| **Unfused** | Last-resort fallback (e.g. `head_size > 256` with past KV). |

The selected backend is reported in the kernel debug info as `SdpaKernel=...` when debug info is enabled.

## 6. XQA Decode Path

XQA (a highly optimized cross/decode attention kernel) is used only when **all** of the following hold:

1. Compute capability SM 8.0+ (Ampere or newer).
2. Decoding phase (not the first prompt) with `sequence_length == 1`.
3. `kv_sequence_length > 0` (there is a new K/V to append).
4. Past and present KV cache share the same buffer.
5. No softcap.
6. Standard softmax, **or** smooth softmax expressed via a `head_sink` tensor (non-quantized KV cache).
7. No local (sliding) window attention — global attention only.
8. Supported `head_size` (64, 128, or 256) and group size.

`head_sink` (attention sink) is supported on the non-quantized XQA path only. Quantized KV cache
(int8 / fp8) paths explicitly reject a non-null attention sink, so a GQA node with both `head_sink`
and a quantized cache falls back to Flash/Flash-Decoding.

XQA selection defaults are:

- **Quantized KV cache (int8 / fp8):** on by default.
- **Non-quantized with a `head_sink` input:** on by default (GPT-OSS style decode).
- **Non-quantized without `head_sink`:** opt-in via `ORT_ENABLE_XQA=1`.

Setting `ORT_ENABLE_XQA=0` disables XQA for the non-quantized path regardless of `head_sink`.

## 7. XQA `head_sink` PrePack

XQA consumes the attention sink as an FP32 buffer, while the model stores `head_sink` as FP16/BF16. To
avoid converting on every decode step, `GroupQueryAttention::PrePack` converts a **constant-initializer**
`head_sink` once into a cached FP32 device buffer (`xqa_head_sink_`):

- The cached buffer is reused for every launch when XQA is eligible.
- A dynamic / non-initializer `head_sink` is **not** prepacked; the kernel instead reserves a small FP32
  scratch buffer and converts the sink per launch (`xqa_head_sink_needs_conversion = true`).
- `PrePack` keeps `is_packed = false` so the original FP16/BF16 `head_sink` is still delivered to the
  Flash/fallback paths when XQA is disabled or ineligible.

## 8. Environment Variables

| Variable | Effect |
|----------|--------|
| `ORT_ENABLE_XQA` | `1` enables the XQA decode path for the non-quantized KV cache (default off; default on for quantized). |
| `ORT_DISABLE_FLASH_DECODE` | `1` disables the Flash Decoding split-KV optimization. |

These are read once when the kernel is constructed.

## 9. Testing

CUDA parity tests live in
[onnxruntime/test/python/transformers/test_gqa.py](../../onnxruntime/test/python/transformers/test_gqa.py):

- `TestXQAQuantizedParity` — XQA per-tensor int8 quantized decode parity.
- `TestXQAHeadSinkParity` — non-quantized XQA decode parity with a `head_sink` (attention sink) input.

`TestXQAQuantizedParity` sets `ORT_ENABLE_XQA=1` to force the XQA path. `TestXQAHeadSinkParity`
instead clears `ORT_ENABLE_XQA` to validate that XQA is enabled by default when a `head_sink` input
is present. Both compare against a PyTorch reference (`attention_ref` with `smooth_softmax_ref`).
