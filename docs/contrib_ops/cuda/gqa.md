# GroupQueryAttention — Operator Documentation

This document describes the `com.microsoft::GroupQueryAttention` (GQA) contrib operator: its schema,
the CUDA kernel backends and how one is selected, and the attention-sink (`head_sink`) decode path
that is accelerated by the XQA kernel.

For CPU-specific implementation details (including the quantized KV-cache flash path), see
[cpu/gqa.md](../cpu/gqa.md).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Operator Schema](#2-operator-schema)
3. [Input Formats](#3-input-formats)
4. [KV Cache and Quantization](#4-kv-cache-and-quantization)
5. [Attention Sink (`head_sink`) and Smooth Softmax](#5-attention-sink-head_sink-and-smooth-softmax)
6. [CUDA Kernel Backends and Dispatch](#6-cuda-kernel-backends-and-dispatch)
7. [XQA Decode Path](#7-xqa-decode-path)
8. [XQA `head_sink` PrePack](#8-xqa-head_sink-prepack)
9. [Selecting a Kernel: Provider Option and Environment Variables](#9-selecting-a-kernel-provider-option-and-environment-variables)
10. [Profiling and Benchmarking](#10-profiling-and-benchmarking)
11. [Fast Build Options](#11-fast-build-options)
12. [Testing](#12-testing)
13. [Future Work and Known Limitations](#13-future-work-and-known-limitations)

---

## 1. Overview

GroupQueryAttention implements grouped-query attention with KV-cache (past/present) support.
The `causal` attribute must be `0` or `1` and defaults to `1`; set it to `0` for bidirectional
attention when the selected backend supports it. Bidirectional attention requires
`local_window_size=-1`; local windows are defined only for causal attention.
Grouped-query attention uses fewer key/value heads than query heads: each KV head is shared by a
group of `num_heads / kv_num_heads` query heads. The operator also supports:

- Rotary positional embeddings (RoPE)
- Past/present KV cache with optional in-place (shared) buffer
- Quantized KV cache (int4 / int8 / float8e4m3fn) to reduce memory footprint
- Optional attention bias and local (sliding) window attention
- Smooth softmax, including a per-head attention sink (`head_sink`)

The operator schema is defined in
[onnxruntime/core/graph/contrib_ops/bert_defs.cc](../../../onnxruntime/core/graph/contrib_ops/bert_defs.cc).
The CUDA kernel is implemented in
[onnxruntime/contrib_ops/cuda/bert/group_query_attention.cc](../../../onnxruntime/contrib_ops/cuda/bert/group_query_attention.cc)
and [group_query_attention_impl.cu](../../../onnxruntime/contrib_ops/cuda/bert/group_query_attention_impl.cu).

## 2. Operator Schema

Selected attributes:

| Attribute | Description |
|-----------|-------------|
| `num_heads` | Number of query heads. |
| `kv_num_heads` | Number of key/value heads. `num_heads % kv_num_heads == 0`. |
| `scale` | Softmax scale. Defaults to `1/sqrt(head_size)`. |
| `causal` | Apply the causal mask. Must be `0` or `1` and defaults to `1`; `0` enables bidirectional attention. |
| `softcap` | Optional logit soft-capping value. `0` disables it. |
| `local_window_size` | Left window size for causal local attention. `-1` means global attention and is required when `causal=0`. |
| `sliding_window_cache` | Set to `1` when using a windowed (sliding-window) KV cache instead of full-length. When enabled, the operator keeps the most recent positions contiguously at cache rows `[0, L)` and evicts internally. Requires `local_window_size > 0`; the CUDA kernel additionally requires the cache capacity to equal `local_window_size`. Defaults to `0` (full-length cache). See [the CPU notes](../cpu/gqa.md#windowed-sliding-window-kv-cache) for the normative layout, eviction and rollback contract. |
| `do_rotary` / `rotary_interleaved` | Enable RoPE and select interleaved vs. half-rotary layout. |
| `smooth_softmax` | Add a smooth factor to the softmax denominator. |
| `qk_norm_epsilon` | Epsilon for the fused per-head Q/K RMSNorm (QK-Norm) prologue. Defaults to `1e-6`. |
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
| 11 | `head_sink` | `(num_heads,)` per-head attention sink (see §5). |
| 12, 13 | `k_scale`, `v_scale` | FP32 dequant scales for the quantized KV cache. |
| 14, 15 | `q_norm_weight`, `k_norm_weight` | `(head_size,)` per-head Q/K RMSNorm weights (QK-Norm, see §3). Both must be present together. |

Outputs are `output`, `present_key`, `present_value`, and optional `output_qk`.

## 3. Input Formats

GQA accepts query/key/value in two layouts. The layout is inferred from whether `key` (input 1)
is present.

### Unpacked Q, K, V (`Q_K_V_BSNH`)

`key` and `value` are both provided:

| Tensor | Shape |
|--------|-------|
| `query` | `(batch_size, sequence_length, num_heads * head_size)` |
| `key`   | `(batch_size, sequence_length, kv_num_heads * head_size)` |
| `value` | `(batch_size, sequence_length, kv_num_heads * head_size)` |

### Packed QKV (`QKV_BS3NH`)

`key` and `value` are omitted (null) and Q, K, V are concatenated along the last dimension of
`query`:

| Tensor | Shape |
|--------|-------|
| `query` | `(batch_size, sequence_length, (num_heads + 2 * kv_num_heads) * head_size)` |

`head_size` is derived as `hidden_size / (num_heads + 2 * kv_num_heads)`.

### KV cache layout

`past_key` / `past_value` / `present_key` / `present_value` always use BNSH:
`(batch_size, kv_num_heads, cache_sequence_length, head_size)`. For a 4-bit quantized cache the
last dimension is `(head_size + 1) / 2` because two nibbles are packed per byte.

### Constraints

- `num_heads % kv_num_heads == 0` (each KV head is shared by `num_heads / kv_num_heads` query heads).
- `head_size == v_head_size` (Q and V share the head size).
- Q and K/V must have the same `sequence_length` (cross-attention is not supported). The exception
  is the shared-buffer decode case where `kv_sequence_length == 0` (no new K/V to append — the past
  buffer already holds the full KV cache).
- RoPE, packed-QKV unpacking, and KV-head expansion are handled internally (`PrepareQKV`) before the
  selected backend runs, so every backend sees a consistent layout.

### Fused QK-Norm (per-head Q/K RMSNorm)

When the optional `q_norm_weight` (input 14) and `k_norm_weight` (input 15) tensors are provided, the
CUDA kernel applies a fused per-head RMS normalization to Q and K **before** RoPE. This matches the
QK-Norm used by **Qwen3, Gemma 2/3, OLMo2, SmolLM3**, etc. For each head, over the `head_size`
channels:

$$
x_\text{norm}[c] = x[c] \cdot \frac{1}{\sqrt{\frac{1}{H}\sum_{j} x[j]^2 + \epsilon}} \cdot w[c]
$$

where `H = head_size`, `w` is the per-head weight vector (`q_norm_weight` for Q, `k_norm_weight` for
K), and `epsilon = qk_norm_epsilon` (default `1e-6`). The sum of squares is reduced in FP32 for
numerical stability and the result is cast back to the operator type `T`.

- Both weights are 1D tensors of shape `(head_size,)`, share the operator's element type `T`
  (`float16`/`bfloat16`), and are **shared across all heads**. They must be supplied together —
  providing only one is rejected.
- The normalization is fused into the `PrepareQKV` prologue (`UnpackRoPEAppend` for the new-KV path,
  or a standalone per-head RMSNorm kernel for the shared-buffer Q-only decode case), so it composes
  with packed QKV, RoPE, KV-head expansion, and the quantized KV cache.
- Because the Flash-Decoding fast path does its own RoPE/append internally and bypasses `PrepareQKV`,
  it is disabled when QK-Norm is present (see §6). The non-quantized XQA decode path can still run
  with QK-Norm: CUDA normalizes Q/K in the `UnpackRoPEAppend` preprocess before launching XQA.

## 4. KV Cache and Quantization

### Layout and shared buffer

The past/present KV cache uses BNSH layout
`(batch_size, kv_num_heads, cache_sequence_length, head_size)`. When `past_present_share_buffer`
holds (the past and present tensors alias the same memory), the cache length is the maximum
sequence length and new keys/values are appended in place. This shared-buffer mode is required by
the XQA decode path and by the Flash-Decoding fast path.

### Sliding Window (Windowed) KV Cache

When the `sliding_window_cache` attribute is set to `1`, the KV cache operates in a windowed
(sliding-window) mode, keeping only the most recent `local_window_size` tokens instead of growing
to the full sequence length. This significantly reduces memory usage for long-context models that
employ local attention (e.g., GPT-OSS with layer-wise sliding windows).

The CPU documentation's **Windowed (Sliding-Window) KV Cache** section is the normative source for
the shared layout semantics. This page documents the CUDA-supported subset and its additional
restrictions.

**Key behaviors:**

- **Cache capacity:** The CUDA kernel requires the cache buffer's sequence dimension to be exactly
  `local_window_size`. It evicts the minimum number of rows on every step, which reproduces the
  documented layout only when there is no slack above the window, so a larger capacity is rejected
  with `INVALID_ARGUMENT` rather than silently producing a different resident range. Slack is
  useful on CPU, where attention scans every resident entry and a drifting append point amortizes
  compaction; on CUDA attention costs the same regardless of the capacity, so the extra rows buy
  nothing.
- **Cache-relative indexing:** New keys/values are appended at cache-relative positions, not global
  positions. Because the capacity equals the window, rows `[0, L)` hold the `L = min(T, C)` most
  recent positions in order, where `T` is `seqlens_k[b] + 1` for that batch entry; row `i` holds
  absolute position `T - L + i`. This is the `G == 1` case of the general contract — the exact
  formula, the chunk-invariance property that multi-token (speculative) steps rely on, and the
  rollback rule are specified in [the CPU notes](../cpu/gqa.md#windowed-sliding-window-kv-cache).
- **RoPE position bounds:** The `rotary_max_position` parameter controls the upper bound (exclusive)
  for RoPE position indices, decoupling absolute sequence positions from cache buffer indices.
- **Multi-token staging:** Because the capacity equals the window, a step of `S > 1` tokens can need
  more entries than the cache holds (its earliest queries still read keys its last ones evict). Such
  a step runs against an internal staging buffer of `min(T - S, C) + S` entries and only the
  surviving tail is written back, so any `S >= 1` is accepted.
- **Present shape:** When using windowed cache with `past_present_share_buffer`, the `present_key`
  and `present_value` shapes remain bounded by `kv_cache_capacity` in the sequence dimension,
  rather than growing with `total_sequence_length`.
- **Constraints:** `sliding_window_cache=1` requires `local_window_size > 0` and a capacity equal to
  it. Windowed caches are incompatible with the Flash-Attention fast-decode path and instead use the
  XQA or standard attention backends.
- **CPU support:** The CPU kernel implements the same contract and additionally accepts a capacity
  larger than the window. Entries live at `[0, end)` and are appended at `end`, so a step only moves
  memory when the append would run past the capacity; at that point the surviving entries are
  compacted to the front in one go. Multi-token steps that would drop entries the step itself still
  reads are staged instead, so results match a full-length cache exactly. On CPU the `qk_output`
  attribute and a shared KV layout (`key`/`value` folded into `query`) are not supported together
  with `sliding_window_cache=1`.

### Quantized KV cache

To reduce the KV-cache memory footprint, the cache may be stored quantized while `query` stays
FP16/BF16. Quantization is **symmetric** and configured by three attributes:

| Attribute | Values |
|-----------|--------|
| `k_quant_type` / `v_quant_type` | `NONE`, `PER_TENSOR`, `PER_CHANNEL` |
| `kv_cache_bit_width` | `8` (INT8 / FP8) or `4` (INT4) |

Supported storage types (`T_CACHE`) and their formula:

| Type | Range | Quantize |
|------|-------|----------|
| INT8 | `[-128, 127]` | `q = clamp(round(x / scale), -128, 127)` |
| INT4 | `[-8, 7]`, two nibbles packed per byte | `q = clamp(round(x / scale), -8, 7)` |
| FP8 E4M3 | `[-448, 448]` | `q = clamp(x / scale, -448, 448)` (SM89+/Ada or SM90+) |

- `k_scale` / `v_scale` (inputs 12, 13) are **always FP32**. For `PER_TENSOR` they are scalars; for
  `PER_CHANNEL` they have shape `(kv_num_heads, 1, head_size)`.
- New keys/values are quantized as they are appended to the present cache. For `PER_TENSOR`, XQA
  applies the K and V dequantization scales inside the attention kernel. For `PER_CHANNEL`, the
  decode path folds the K scale into Q before attention and applies the V scale to the output
  afterward, avoiding full-cache dequantization.
- Registered type combinations are `T ∈ {float16, bfloat16}` × `T_CACHE ∈ {same as T, int8, FP8E4M3, uint8 (int4)}`.

### How quantized decode is served

The INT8 and FP8 quantized KV-cache paths are handled by the **XQA** decode kernel when its other
decode constraints are met (see §7). K and V may independently use `PER_TENSOR` or `PER_CHANNEL`
quantization, and `k_scale` and `v_scale` may be distinct tensors. Quantized XQA also requires
`head_size ∈ {64, 128, 256}` and a query/KV group size in `{4, 8, 16, 32}`. FP8 additionally
requires SM89+ (Ada) or SM90+.

For per-tensor scaling, XQA applies `k_scale` to the QK scores before softmax and `v_scale` to the
attention-value accumulator. For per-channel scaling, dequantization remains linear but cannot be
represented by scalar kernel parameters: the decode path multiplies Q by the per-channel K scale
and multiplies the attention output by the per-channel V scale. These two
`O(num_heads * head_size)` passes avoid dequantizing the entire cache on every decode step.

INT4 caches are not supported by XQA. Quantized configurations that are ineligible for XQA use the
dequantize-then-Flash-Attention fallback when available.

INT8 cache kernels are always built; FP8 (`onnxruntime_USE_FP8_KV_CACHE`, default ON) and INT4
(`onnxruntime_USE_INT4_KV_CACHE`, default OFF) are gated by build options (see §11).

## 5. Attention Sink (`head_sink`) and Smooth Softmax

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

## 6. CUDA Kernel Backends and Dispatch

The CUDA EP can route a GQA node to one of five backends. They are evaluated in a fixed priority
order and the first eligible backend wins:

**XQA → cuDNN SDPA → Flash Attention → Memory Efficient Attention (MEA) → Unfused**

| Priority | Backend | Selected when (summary) |
|----------|---------|-------------------------|
| 1 | **XQA** | Causal single-token decode (`seq_len == 1`), shared KV buffer. Supports sliding-window attention and attention sinks on both the non-quantized and quantized (INT8/FP8) paths. Fastest decode path; supports per-tensor and per-channel quantized caches. |
| 2 | **cuDNN SDPA** | Non-quantized FP16/BF16 causal or bidirectional attention. Auto-preferred on SM≥90 (Hopper/Blackwell). |
| 3 | **Flash Attention** | General FP16/BF16 causal or bidirectional prompt and decode, including softcap and packed QKV. Local windows are supported for causal attention. |
| 4 | **Memory Efficient Attention (MEA)** | Non-quantized causal or bidirectional fallback for FP16/FP32 (and BF16 on SM80+). |
| 5 | **Unfused** | Non-quantized causal or bidirectional last-resort fallback (e.g. `head_size > 256`). Any head size, GQA, and softcap; sliding windows are causal-only. |

The selected backend is reported in the kernel debug info as `SdpaKernel=...` when debug info is
enabled (see §10).

> **QK-Norm interaction.** When `q_norm_weight` / `k_norm_weight` are present (see §3), the
> Flash-Decoding fast path is disabled so the QK-Norm prologue always runs. Non-quantized XQA decode
> remains eligible for supported shapes: the `UnpackRoPEAppend` preprocess normalizes Q/K, applies
> RoPE, appends K/V, and then XQA consumes the normalized Q and cache. Quantized-cache QK-Norm decode
> still falls back to Flash Attention (or cuDNN SDPA / MEA / Unfused) until normalized-K scale
> handling is validated for XQA.

### 6.1 XQA

Checked first. Used only for single-token decode under the conditions detailed in §7. Global and
sliding-window decode, with or without an attention sink, are supported on both the non-quantized
and quantized paths. When XQA is selected, no other backend is considered. An XQA-ineligible
quantized cache uses the dequantize-then-Flash-Attention fallback when available.

### 6.2 cuDNN SDPA

Eligible when **all** of the following hold:

- not already selected for XQA;
- KV cache is **not** quantized (`T_CACHE == T`);
- `softcap == 0`, no smooth softmax, and no `head_sink`;
- no local (sliding) window (`local_window_size == -1`);
- past/present KV in BNSH (`Q_K_V_BNSH`);
- cuDNN SDPA is enabled — either explicitly (`ORT_ENABLE_CUDNN_FLASH_ATTENTION=1` or the cuDNN bit of
  `sdpa_kernel`), or auto-preferred on SM≥90 when no kernel is explicitly pinned;
- cuDNN ≥ 9.3 (stable) and `is_supported` returns true for the shape.

### 6.3 Flash Attention

Eligible when:

- not XQA and not cuDNN SDPA;
- FP16/BF16 (`sizeof(T) == 2`) and Flash is enabled (not `ORT_DISABLE_FLASH_ATTENTION`, not disabled
  via `sdpa_kernel`, and built with `USE_FLASH_ATTENTION`);
- `flash::is_supported` is true for `head_size` / `num_heads` / `kv_num_heads`.

Flash supports local window, softcap, RoPE, and packed QKV. For decode it additionally uses a
**Flash-Decoding** split-KV fast path (`seq_len == 1`, shared buffer, non-quantized), unless
`ORT_DISABLE_FLASH_DECODE=1`.

### 6.4 Memory Efficient Attention (MEA)

Fallback when XQA, cuDNN SDPA, and Flash are all ineligible:

- MEA enabled (not `ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION`, built with `USE_MEMORY_EFFICIENT_ATTENTION`);
- `has_memory_efficient_attention(sm, is_fp16, is_bf16, head_size)` is true — FP16/FP32 broadly,
  BF16 on SM80+.

When the query/KV head counts differ, the KV heads are expanded to `num_heads` into a scratch buffer.

### 6.5 Unfused

Last-resort path, activated when XQA / cuDNN / Flash / MEA are all ineligible **and**:

- KV cache is not quantized;
- no smooth softmax and no `head_sink`;
- past/present KV in BNSH.

It supports any `head_size` (FP32 QK accumulation), GQA, sliding window, and softcap — for example
`head_size > 256` with past KV. The unfused (math) path can never be turned off and is always
available as a fallback.

## 7. XQA Decode Path

XQA (a highly optimized cross/decode attention kernel) is used only when **all** of the following hold:

1. Compute capability SM 8.0+ (Ampere or newer).
2. Decoding phase (not the first prompt) with `sequence_length == 1`.
3. `kv_sequence_length > 0` (there is a new K/V to append).
4. Past and present KV cache share the same buffer.
5. No softcap.
6. Standard softmax, **or** smooth softmax expressed via a `head_sink` tensor.
7. Global attention, **or** local (sliding) window attention (`local_window_size > 0`), supported on
   both the non-quantized and quantized (INT8/FP8) paths.
8. Supported `head_size` (64, 128, or 256) and query/KV group size:
   - Non-quantized: `{1, 2, 4, 5, 8, 16, 32}`.
   - INT8/FP8: `{4, 8, 16, 32}`, with both K and V using `PER_TENSOR` or `PER_CHANNEL`
     quantization. K and V may use different modes and distinct scale tensors.
9. For FP8, SM89+ (Ada) or SM90+ and an FP8-enabled build.
10. The selected XQA kernel's dynamic shared-memory requirement fits the device limit.

`head_sink` (attention sink) is supported on both the non-quantized and quantized INT8/FP8 XQA
paths, including local-window and multi-block decode. INT4 caches and quantized-cache QK-Norm are
not supported by XQA.

XQA selection is on by default. Setting `ORT_ENABLE_XQA=0` disables XQA.

## 8. XQA `head_sink` PrePack

XQA consumes the attention sink as an FP32 buffer, while the model stores `head_sink` as FP16/BF16. To
avoid converting on every decode step, `GroupQueryAttention::PrePack` converts a **constant-initializer**
`head_sink` once into a cached FP32 device buffer (`xqa_head_sink_`):

- The cached buffer is reused for every launch when XQA is eligible.
- A dynamic / non-initializer `head_sink` is **not** prepacked; the kernel instead reserves a small FP32
  scratch buffer and converts the sink per launch (`xqa_head_sink_needs_conversion = true`).
- `PrePack` keeps `is_packed = false` so the original FP16/BF16 `head_sink` is still delivered to the
  Flash/fallback paths when XQA is disabled or ineligible.

## 9. Selecting a Kernel: Provider Option and Environment Variables

### `sdpa_kernel` provider option

The CUDA EP exposes a `sdpa_kernel` provider option (a bitmask defined by `AttentionBackend`) that
pins which fused attention backends are allowed. It applies to GroupQueryAttention,
MultiHeadAttention, and Attention nodes.

| Bit value | Backend |
|-----------|---------|
| `0` | Default — selection follows heuristics / environment variables (auto-prefers cuDNN SDPA on SM≥90). |
| `1` | Flash Attention |
| `2` | Memory Efficient Attention |
| `8` | cuDNN SDPA |
| `16` | Unfused (math) — note the unfused fallback can never actually be turned off |

Bits can be OR-ed together. Any positive value is treated as an **explicit** selection: only the
listed backends are enabled and the automatic cuDNN-on-SM≥90 preference is disabled. **XQA is not
part of this bitmask** — it is controlled separately by `ORT_ENABLE_XQA`.

```python
import onnxruntime as ort

sess = ort.InferenceSession(
    "model.onnx",
    providers=[("CUDAExecutionProvider", {"sdpa_kernel": "1"})],  # 1 = Flash Attention only
)
```

### Environment variables

| Variable | Effect |
|----------|--------|
| `ORT_ENABLE_XQA` | `1` enables the XQA decode; `0` disables XQA entirely. Unset: on by default. |
| `ORT_DISABLE_FLASH_ATTENTION` | `1` disables Flash Attention. |
| `ORT_DISABLE_FLASH_DECODE` | `1` disables the Flash-Decoding split-KV optimization. |
| `ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION` | `1` disables Memory Efficient Attention. |
| `ORT_ENABLE_CUDNN_FLASH_ATTENTION` | `1` enables cuDNN SDPA; `0` disables it and also disables the SM≥90 auto-preference. |
| `ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO` | `1` prints the selected backend (`SdpaKernel=...`) per node (see §10). |

A positive `sdpa_kernel` value takes precedence over these environment defaults. Environment
variables are read once when the kernel is constructed.

## 10. Profiling and Benchmarking

### Verify which backend ran

Set `ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO=1`. For each GQA node the kernel prints a line such as:

```
Operator=GroupQueryAttention Node=<name> DataType=fp16 SdpaKernel=XQA
```

`SdpaKernel` is one of `XQA`, `FLASH_ATTENTION`, `EFFICIENT_ATTENTION`, `CUDNN_FLASH_ATTENTION`, or
`MATH` (unfused). Use this to confirm that an env var / `sdpa_kernel` choice took effect.

### Benchmark and profiling scripts

Located in `onnxruntime/test/python/transformers/`:

| Script | Purpose |
|--------|---------|
| [profile_gqa.py](../../../onnxruntime/test/python/transformers/profile_gqa.py) | Profile GQA (incl. quantized KV cache) with NVTX markers; examples for Nsight Compute (`ncu`) and Nsight Systems (`nsys`). |
| [benchmark_gqa.py](../../../onnxruntime/test/python/transformers/benchmark_gqa.py) | Triton-based throughput comparison across dense / local / packed-QKV and INT4/INT8/FP8 variants. |
| [benchmark_gqa_windows.py](../../../onnxruntime/test/python/transformers/benchmark_gqa_windows.py) | GQA benchmark variant for Windows. |
| [benchmark_gqa_cpu_flash.py](../../../onnxruntime/test/python/transformers/benchmark_gqa_cpu_flash.py) | CPU flash-vs-naive GQA benchmark. |

Example kernel-level and timeline profiling:

```bash
cd onnxruntime/test/python/transformers

# Kernel-level analysis with Nsight Compute
ncu --set full -o gqa_int8 python profile_gqa.py --mode int8 --warmup 5 --repeat 1

# Timeline with Nsight Systems, then parse kernel timings
nsys profile -o gqa_int8 --export=sqlite python profile_gqa.py --mode int8 --warmup 5 --repeat 10
python parse_nsys.py gqa_int8.sqlite
```

ONNX Runtime's built-in profiler (`SessionOptions.enable_profiling = True`) also emits a JSON
timeline with per-node durations.

## 11. Fast Build Options

These CMake options speed up CUDA builds during development. Pass them through
`--cmake_extra_defines` (see the `ort-build` skill).

| Option | Default | Effect |
|--------|---------|--------|
| `onnxruntime_QUICK_BUILD` | `OFF` | Builds only the `hdim128` FP16/BF16 Flash Attention kernels. Greatly reduces compile time, but **changes dispatch**: shapes with `head_size != 128` fall back to Memory Efficient Attention because Flash is no longer compiled for them. Do not use it to characterize Flash-vs-arch behavior. |
| `onnxruntime_USE_FP8_KV_CACHE` | `ON` | Builds the FP8 (E4M3) quantized KV-cache kernels (`-DUSE_FP8_KV_CACHE=1`). |
| `onnxruntime_USE_INT4_KV_CACHE` | `OFF` | Builds the INT4 quantized KV-cache kernels (`-DUSE_INT4_KV_CACHE=1`). A `kv_cache_bit_width == 4` node errors out if this is off. |

Other ways to shorten the iteration loop:

- Restrict GPU architectures with `CMAKE_CUDA_ARCHITECTURES` (e.g.
  `--cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=80`) so kernels are not compiled for unused SMs.
- Build only the CUDA provider target:
  `./build.sh --config Release --build --parallel --target onnxruntime_providers_cuda`.
- Skip `--update` when you only edited existing `.cc` / `.h` / `.cu` files.

```bash
./build.sh --config Release --parallel --use_cuda \
  --cuda_home /usr/local/cuda --cudnn_home /usr/local/cuda \
  --cmake_extra_defines onnxruntime_QUICK_BUILD=ON onnxruntime_USE_INT4_KV_CACHE=ON
```

## 12. Testing

CUDA parity tests live in
[onnxruntime/test/python/transformers/test_gqa.py](../../../onnxruntime/test/python/transformers/test_gqa.py):

- `TestXQAQuantizedParity` — XQA per-tensor int8 quantized decode parity.
- `TestXQAHeadSinkParity` — non-quantized XQA decode parity with a `head_sink` (attention sink) input.
- `TestXQAQuantizedHeadSinkParity` — INT8/FP8 XQA decode with runtime or prepacked attention sinks,
  including global, local-window, and multi-block decode.
- `TestXQASeparateKVScaleParity` — INT8/FP8 XQA parity with independently calibrated per-tensor K
  and V scales.
- `TestXQAPerChannelScaleParity` — INT8/FP8 XQA parity with per-channel K/V scales folded into Q
  and the output.
- `TestGQAQKNorm` — fused per-head Q/K RMSNorm (QK-Norm) parity for prompt and decode (past), FP16 and
  BF16, across packed/unpacked Q/K/V and with/without RoPE.

`TestXQAQuantizedParity` sets `ORT_ENABLE_XQA=1` to force the XQA path. `TestXQAHeadSinkParity`
instead clears `ORT_ENABLE_XQA` to validate that XQA is enabled by default when a `head_sink` input
is present. Both compare against a PyTorch reference (`attention_ref` with `smooth_softmax_ref`).
`TestGQAQKNorm` applies the RMSNorm-before-RoPE reference to Q and K and compares against the CUDA
output.

The feature-interaction tests in `TestFlashGQA` cover batch size greater than one with:

- causal and bidirectional attention;
- RoPE with both half-rotary and interleaved layouts;
- per-head Q/K RMSNorm before RoPE;
- softcap with and without attention sinks;
- quantized KV cache with distinct K/V quantization modes and attention sinks; and
- attention-bias shapes with batch and head broadcasting.

CUDA dispatches `attention_bias` to an unfused fallback. It cannot be combined with a quantized KV
cache, `head_sink`, or smooth softmax. Softcap is supported by the Flash, MEA, and unfused paths but
disables XQA and cuDNN SDPA. QK-Norm is supported with non-quantized RoPE paths; quantized-cache
QK-Norm is not eligible for XQA.

## 13. Future Work and Known Limitations

The following features are missing or limited in the CUDA GQA kernel and would broaden coverage of
popular LLMs. Listed roughly by impact.

### High impact

1. **Fused QK-Norm (per-head Q/K RMSNorm prologue).** *Implemented.* The CUDA kernel applies the
   fused per-head RMSNorm to Q and K before RoPE when `q_norm_weight` / `k_norm_weight` are provided
    (see §3), matching **Qwen3, Gemma 2/3, OLMo2, SmolLM3**, etc. Remaining limitation: QK-Norm
    disables Flash-Decoding, and quantized-cache QK-Norm does not yet get the XQA fast path.
2. **Sliding-window and attention sinks on the quantized fused decode path.** *Implemented.* The
  XQA decode path serves sliding-window attention (`local_window_size > 0`) and `head_sink` on both
  the non-quantized and quantized (INT8/FP8) paths, including when both features are enabled.
3. **Softcap on the fastest kernels.** Logit soft-capping (**Gemma 2**) disables both XQA and cuDNN
   SDPA, forcing the Flash / MEA / unfused paths. Adding softcap support to XQA and cuDNN would
   recover decode throughput.
4. **Attention bias / ALiBi.** `attention_bias` is supported by the unfused CUDA fallback, but it
  cannot currently be combined with quantized KV cache, `head_sink`, or smooth softmax.

### Medium impact

5. **Quantized KV cache coverage.** INT8/FP8 XQA decode supports independent `PER_TENSOR` or
  `PER_CHANNEL` K/V scales, `head_sink`, `head_size ∈ {64, 128, 256}`, and group size
  `{4, 8, 16, 32}`. Remaining gaps include fused prompt-phase quantized attention, INT4 XQA,
  INT4 enabled by default, and quantized-cache QK-Norm on XQA.
6. **Paged KV cache / continuous batching.** GQA uses a contiguous shared buffer; there is a
   separate `PagedAttention` op, but GQA itself has no paged-cache path. Paged KV is what
   high-throughput serving (vLLM-style) needs.
7. **MLA (Multi-head Latent Attention).** **DeepSeek-V2/V3** use latent KV compression with a
   `v_head_size` that differs from `head_size`; GQA assumes `head_size == v_head_size`. This needs a
   distinct kernel/op rather than a GQA tweak.

### Lower impact / niche

8. **Returning attention weights (`output_qk`).** Never supported by the CUDA fused kernels. Only
   relevant for interpretability or speculative-decode scoring.
9. **Cross-attention (different Q vs KV sequence lengths).** Rejected by the input checker.
   Encoder-decoder / multimodal cross-attention is not covered by GQA.
