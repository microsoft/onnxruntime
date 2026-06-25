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

GroupQueryAttention implements causal grouped-query attention with KV-cache (past/present) support.
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
| `softcap` | Optional logit soft-capping value. `0` disables it. |
| `local_window_size` | Left window size for local attention. `-1` means global attention. |
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
- New keys/values are quantized as they are appended to the present cache; the attention kernel
  dequantizes on the fly while computing scores.
- Registered type combinations are `T ∈ {float16, bfloat16}` × `T_CACHE ∈ {same as T, int8, FP8E4M3, uint8 (int4)}`.

### How quantized decode is served

The quantized KV-cache path is handled by the **XQA** decode kernel (see §7). XQA requires
`PER_TENSOR` scaling with `k_scale` and `v_scale` pointing to the **same** FP32 tensor,
`head_size ∈ {64, 128, 256}`, and a query/KV group size in `{4, 8, 16, 32}`. FP8 additionally
requires SM89+ (Ada) or SM90+.

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
| 1 | **XQA** | Single-token decode (`seq_len == 1`), shared KV buffer. Supports sliding-window attention on both the non-quantized and quantized (INT8/FP8) paths (attention sink remains non-quantized-only). Fastest decode path; the only backend that serves a quantized KV cache. |
| 2 | **cuDNN SDPA** | Non-quantized FP16/BF16 causal attention. Auto-preferred on SM≥90 (Hopper/Blackwell). |
| 3 | **Flash Attention** | General FP16/BF16 prompt and decode, including local window, softcap, and packed QKV. |
| 4 | **Memory Efficient Attention (MEA)** | Fallback for FP16/FP32 (and BF16 on SM80+). |
| 5 | **Unfused** | Last-resort fallback (e.g. `head_size > 256`). Any head size, GQA, sliding window, softcap. |

The selected backend is reported in the kernel debug info as `SdpaKernel=...` when debug info is
enabled (see §10).

> **QK-Norm interaction.** When `q_norm_weight` / `k_norm_weight` are present (see §3), the
> Flash-Decoding fast path is disabled so the QK-Norm prologue always runs. Non-quantized XQA decode
> remains eligible for supported shapes: the `UnpackRoPEAppend` preprocess normalizes Q/K, applies
> RoPE, appends K/V, and then XQA consumes the normalized Q and cache. Quantized-cache QK-Norm decode
> still falls back to Flash Attention (or cuDNN SDPA / MEA / Unfused) until normalized-K scale
> handling is validated for XQA.

### 6.1 XQA

Checked first. Used only for single-token decode under the conditions detailed in §7 (global or
sliding-window decode; sliding window is supported on both the non-quantized and quantized paths).
When XQA is selected, no other backend is considered.

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
6. Standard softmax, **or** smooth softmax expressed via a `head_sink` tensor (non-quantized KV cache).
7. Global attention, **or** local (sliding) window attention (`local_window_size > 0`), supported on
   both the non-quantized and quantized (INT8/FP8) paths.
8. Supported `head_size` (64, 128, or 256) and group size.

`head_sink` (attention sink) is supported on the non-quantized XQA path only. Quantized KV cache
(int8 / fp8) paths explicitly reject a non-null attention sink, so a GQA node with both `head_sink`
and a quantized cache falls back to Flash/Flash-Decoding.

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
- `TestGQAQKNorm` — fused per-head Q/K RMSNorm (QK-Norm) parity for prompt and decode (past), FP16 and
  BF16, across packed/unpacked Q/K/V and with/without RoPE.

`TestXQAQuantizedParity` sets `ORT_ENABLE_XQA=1` to force the XQA path. `TestXQAHeadSinkParity`
instead clears `ORT_ENABLE_XQA` to validate that XQA is enabled by default when a `head_sink` input
is present. Both compare against a PyTorch reference (`attention_ref` with `smooth_softmax_ref`).
`TestGQAQKNorm` applies the RMSNorm-before-RoPE reference to Q and K and compares against the CUDA
output.

## 13. Future Work and Known Limitations

The following features are missing or limited in the CUDA GQA kernel and would broaden coverage of
popular LLMs. Listed roughly by impact.

### High impact

1. **Fused QK-Norm (per-head Q/K RMSNorm prologue).** *Implemented.* The CUDA kernel applies the
   fused per-head RMSNorm to Q and K before RoPE when `q_norm_weight` / `k_norm_weight` are provided
    (see §3), matching **Qwen3, Gemma 2/3, OLMo2, SmolLM3**, etc. Remaining limitation: QK-Norm
    disables Flash-Decoding, and quantized-cache QK-Norm does not yet get the XQA fast path.
2. **Sliding-window on the quantized fused decode path.** *Implemented.* The XQA decode path now
   serves sliding-window attention (`local_window_size > 0`) on both the non-quantized and the
   quantized (INT8/FP8) paths. Remaining limitation: the attention sink (`head_sink`) is still
   non-quantized-only, so quantized **GPT-OSS / Mistral / Gemma 2** sliding-window layers that also
   use an attention sink fall back to Flash / Flash-Decoding.
3. **Softcap on the fastest kernels.** Logit soft-capping (**Gemma 2**) disables both XQA and cuDNN
   SDPA, forcing the Flash / MEA / unfused paths. Adding softcap support to XQA and cuDNN would
   recover decode throughput.
4. **Attention bias / ALiBi.** `attention_bias` is rejected outright. Needed for ALiBi-style models
   and additive-mask use cases, though less commonly used in current popular decoder-only LLMs.

### Medium impact

5. **Quantized KV cache coverage.** Quantized decode is XQA-only and narrow: `PER_TENSOR` with
   `k_scale == v_scale`, `head_size ∈ {64, 128, 256}`, group size `{4, 8, 16, 32}`. Gaps worth
   filling: `PER_CHANNEL` serving, prompt-phase quantized attention, INT4 enabled by default, and
   `head_sink` combined with a quantized cache (currently rejected).
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
