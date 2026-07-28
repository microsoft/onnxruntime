# PagedAttention — CUDA Design Document

Status: **Draft / proposal**
Scope: `com.microsoft::PagedAttention`, CUDA Execution Provider
Related: [gqa.md](gqa.md) (`com.microsoft::GroupQueryAttention`)

## Table of Contents

- [1. Purpose and Scope](#1-purpose-and-scope)
- [2. Current State](#2-current-state)
- [3. Design Principles](#3-design-principles)
- [4. Proposed Schema Evolution](#4-proposed-schema-evolution)
- [5. Feature: `slot_mapping`](#5-feature-slot_mapping)
- [6. Feature: Attention Sink (`head_sink`) and Smooth Softmax](#6-feature-attention-sink-head_sink-and-smooth-softmax)
- [7. Feature: Fused QK-Norm](#7-feature-fused-qk-norm)
- [8. Feature: Quantized Paged KV Cache](#8-feature-quantized-paged-kv-cache)
- [9. Feature: Sliding Window Attention](#9-feature-sliding-window-attention)
- [10. Feature: `attention_bias`](#10-feature-attention_bias)
- [11. Feature: `output_qk`](#11-feature-output_qk)
- [12. Feature: Multi-head Latent Attention (MLA)](#12-feature-multi-head-latent-attention-mla)
- [13. Kernel Dispatch and Backend Plan](#13-kernel-dispatch-and-backend-plan)
- [14. Shared Code with GroupQueryAttention](#14-shared-code-with-groupqueryattention)
- [15. Validation Rules](#15-validation-rules)
- [16. Shape Inference and Tooling](#16-shape-inference-and-tooling)
- [17. Testing Plan](#17-testing-plan)
- [18. Known Defects to Fix First](#18-known-defects-to-fix-first)
- [19. Phasing](#19-phasing)
- [20. Open Questions](#20-open-questions)

---

## 1. Purpose and Scope

`PagedAttention` is the **server / continuous-batching** attention operator for ONNX Runtime.
`GroupQueryAttention` (GQA) remains the **padded-batch** operator used by edge and single-stream
scenarios. The two ops are deliberately separate:

| | `GroupQueryAttention` | `PagedAttention` |
|---|---|---|
| Query layout | `(batch_size, sequence_length, hidden_size)` — padded | `(token_count, hidden_size)` — packed varlen |
| KV cache | `(batch, kv_num_heads, max_seq, head_size)` BNSH, one contiguous buffer per sequence | `(num_blocks, block_size, kv_num_heads, head_size)` + `block_table` |
| Batching | Static, padded | Continuous / in-flight, ragged |
| Target | Edge, on-device, single stream | Multi-tenant serving, high throughput |

Rank and semantics of `query` differ between the two, and the KV aliasing contract differs
(GQA shares one past/present buffer per sequence; PagedAttention mutates a global block pool in
place). Overloading GQA with a `block_table` input is rejected as a design: ORT kernel matching is
by op type and type constraints, not by optional-input presence, so every EP that registers GQA
would claim a paged node and fail at run time instead of at partition time.

**This document specifies the feature work required to bring `PagedAttention` to parity with GQA
for popular LLMs** (GPT-OSS, Qwen3, Gemma 2/3, DeepSeek, Llama, Mistral, Phi), plus the paging
primitives (`slot_mapping`) that serving frameworks require, plus **Multi-head Latent Attention**
(§12) — which is not a GQA feature at all, but is native to this operator because MLA's entire
value proposition is a smaller paged KV cache.

## 2. Current State

Implemented today in `onnxruntime/contrib_ops/cuda/bert/paged_attention{.cc,.h,_impl.cu,_impl.h,_helper.h}`:

| Capability | State |
|---|---|
| Packed varlen Q/K/V and packed QKV | Done |
| Block KV cache, in-place update (`ReshapeAndCache`) | Done |
| `block_table` gather for reads | Done |
| RoPE (`do_rotary`, `rotary_interleaved`, `cos_cache`/`sin_cache`) | Done |
| `scale`, `softcap` | Done |
| `local_window_size` | Wired to Flash varlen (`local_window_size - 1`) and MEA; not validated end-to-end |
| Backends | FlashAttention varlen → Memory-Efficient Attention (CUTLASS fMHA) fallback |
| dtypes | `float16`, `bfloat16` only |
| EPs | CUDA only |

Not implemented: `slot_mapping` (a `slot_mappings` field exists in `PagedAttentionData` but is
never populated or consumed), `head_sink` / smooth softmax, QK-Norm, quantized cache,
`attention_bias`, `output_qk`, a dedicated paged decode kernel.

Structural limits in the current implementation:

- `batch_size <= 256` — `LaunchGetCumulativeSeqlensKV` uses a per-block `cub::BlockScan` with 256
  threads and independent blocks.
- `block_size % 256 == 0` — see [§18](#18-known-defects-to-fix-first); this is almost certainly a bug.
  *(Partly true. It is a genuine FlashAttention tiling constraint, but a head-size-dependent one. See
  the implementation note in §18.1: validation now accepts any power-of-two `block_size >= 16` and
  the constraint has been moved into Flash backend eligibility, with fallback to the
  memory-efficient backend.)*
- MEA fallback materializes a **tight gathered, GQA-expanded** KV buffer of
  `[total_kv_tokens, num_heads, head_size]`; memory scales with context × query heads.
- A device→host sync per step to obtain `max_query_len` (and `total_kv_tokens` for MEA).

## 3. Design Principles

1. **Additive schema only.** `PagedAttention` is already shipped as contrib opset 1. All new inputs
   are appended at the end as `OpSchema::Optional`, all new attributes have defaults matching current
   behavior, and existing type constraints are only *widened*. No existing model breaks.
2. **Semantic parity with GQA, not schema parity.** A feature that exists in both ops must have
   identical math, identical attribute names, identical defaults, and identical scale-tensor
   layouts. Inputs that only make sense for padded batching (`past_key`, `past_value`,
   `total_sequence_length`, padded `position_ids`, `present_*`) are **never** mirrored.
3. **Share the math, not the schema.** New features are implemented once in a shared kernel layer
   parameterized by a KV *accessor* (contiguous BNSH vs. block table), so GQA and PagedAttention
   cannot drift. See [§14](#14-shared-code-with-groupqueryattention).
4. **Fail at partition time, not run time.** No stub kernels that return `NOT_IMPLEMENTED`. An EP
   either registers `PagedAttention` or does not.
5. **Prefer exact epilogues over new fused kernels** where an existing kernel already returns the
   quantities needed (see the `head_sink` LSE rescale in [§6](#6-feature-attention-sink-head_sink-and-smooth-softmax)).

## 4. Proposed Schema Evolution

### 4.1 Inputs

Indices 0–9 are unchanged. Indices 10–16 are new and optional. The index order matches the
landing order in [§19](#19-phasing), so the schema grows monotonically per phase.

| Idx | Name | Type | Shape | Status |
|-----|------|------|-------|--------|
| 0 | `query` | `T` | `(token_count, hidden_size)` or packed `(token_count, (num_heads + 2*kv_num_heads)*head_size)` | existing |
| 1 | `key` | `T` (opt) | `(token_count, kv_hidden_size)` | existing |
| 2 | `value` | `T` (opt) | `(token_count, kv_hidden_size)` | existing |
| 3 | `key_cache` | **`T_CACHE`** | `(num_blocks, block_size, kv_num_heads, head_size)` | **widened** |
| 4 | `value_cache` | **`T_CACHE`** (opt) | same as `key_cache` | **widened + optional (§12)** |
| 5 | `cumulative_sequence_length` | `S` | `(batch_size + 1,)` | existing |
| 6 | `past_seqlens` | `S` | `(batch_size,)` | existing |
| 7 | `block_table` | `S` | `(batch_size, max_num_blocks_per_seq)` | existing |
| 8 | `cos_cache` | `T` (opt) | `(max_seq_len, rotary_dim/2)` | existing |
| 9 | `sin_cache` | `T` (opt) | `(max_seq_len, rotary_dim/2)` | existing |
| 10 | `slot_mapping` | `S` (opt) | `(token_count,)` | **new — §5** |
| 11 | `head_sink` | `T` (opt) | `(num_heads,)` | **new — §6** |
| 12 | `q_norm_weight` | `T` (opt) | `(head_size,)` | **new — §7** |
| 13 | `k_norm_weight` | `T` (opt) | `(head_size,)` | **new — §7** |
| 14 | `k_scale` | `T_KV_SCALE` (opt) | scalar or `(kv_num_heads, 1, head_size)` | **new — §8** |
| 15 | `v_scale` | `T_KV_SCALE` (opt) | scalar or `(kv_num_heads, 1, head_size)` | **new — §8** |
| 16 | `attention_bias` | `T` (opt) | `(1 or num_heads, token_count, max_context_len)` | **new — §10** |

### 4.2 Outputs

| Idx | Name | Type | Shape | Status |
|-----|------|------|-------|--------|
| 0 | `output` | `T` | `(token_count, num_heads * v_head_size)` | existing (shape generalized by §12) |
| 1 | `key_cache_out` | `T_CACHE` (opt) | aliases `key_cache` | existing |
| 2 | `value_cache_out` | `T_CACHE` (opt) | aliases `value_cache` | existing |
| 3 | `output_qk` | `T` (opt) | `(num_heads, token_count, max_context_len)` | **new — §11** |

### 4.3 Attributes

| Name | Type | Default | Status |
|---|---|---|---|
| `num_heads` | INT | required | existing |
| `kv_num_heads` | INT | required | existing |
| `scale` | FLOAT | `1/sqrt(head_size)` | existing |
| `softcap` | FLOAT | `0.0` | existing |
| `local_window_size` | INT | `-1` | existing — §9 |
| `do_rotary` | INT | `0` | existing |
| `rotary_interleaved` | INT | `0` | existing |
| `smooth_softmax` | INT | `0` | **new — §6** |
| `qk_norm_epsilon` | FLOAT | `1e-6` | **new — §7** |
| `k_quant_type` | STRING | `"NONE"` | **new — §8** |
| `v_quant_type` | STRING | `"NONE"` | **new — §8** |
| `kv_cache_bit_width` | INT | unset | **new — §8** |
| `qk_output` | INT | `0` | **new — §11** |
| `v_head_size` | INT | `0` (= `head_size`) | **new — §12** |
| `rotary_offset` | INT | `0` | **new — §12** |

Names, defaults, and value sets are copied verbatim from GQA so that a graph rewriter can move an
attribute between the two ops without translation.

### 4.4 Type constraints

| Name | Allowed | Change |
|---|---|---|
| `T` | `float16`, `bfloat16` | unchanged |
| `T_CACHE` | `float16`, `bfloat16`, `int8`, `uint8`, `float8e4m3fn` | **new** (split out of `T`) |
| `T_KV_SCALE` | `float` | **new** |
| `S` | `int32` | unchanged |

Splitting `T_CACHE` out of `T` is backward compatible: every previously valid model has
`T_CACHE == T`.

## 5. Feature: `slot_mapping`

### 5.1 Problem

Today the write slot for each token is *derived* inside `ReshapeAndCache`:

```
batch_id      = binary_search(cumulative_seqlens_q, token_id)     // which sequence owns this token
token_offset  = token_id - cumulative_seqlens_q[batch_id]
position      = past_seqlens[batch_id] + token_offset
block_id      = block_table[batch_id * max_num_blocks_per_seq + position / block_size]
slot          = block_id * block_size + position % block_size
```

This hard-codes "append `n_b` contiguous tokens at the end of sequence `b`". It cannot express:

- **Prefix caching** — a prefill whose first *k* tokens hit an already-populated shared block must
  *not* rewrite those slots (and must not have its KV recomputed into them).
- **Speculative / tree decoding** — draft tokens are written speculatively and rejected tokens must
  be discarded; positions are not a contiguous run.
- **Chunked prefill with out-of-order scheduling**, where the scheduler owns slot assignment.
- **Cache-aware scheduling / block migration**, where the runtime, not the kernel, decides placement.

It also concentrates a whole class of out-of-bounds bugs in the kernel (see [§18](#18-known-defects-to-fix-first)).

### 5.2 Design

Add optional input 10, `slot_mapping`, `int32`, shape `(token_count,)`.

Each element is a **flat slot index** into the cache viewed as
`[num_blocks * block_size, kv_num_heads, head_size]`:

```
slot_mapping[t] = block_id * block_size + offset_in_block      // 0 <= slot < num_blocks * block_size
slot_mapping[t] = -1                                            // do not write this token's K/V
```

Semantics:

- **When present**, `slot_mapping` is authoritative for the *write* path. `ReshapeAndCache` performs
  no binary search and no derivation; it reads `slot_mapping[token_id]` directly. Tokens with `-1`
  skip the K/V store entirely (their Q still participates in attention). This is the prefix-cache
  and rejected-speculative-token case.
- **When absent**, behavior is exactly today's derivation — full backward compatibility.
- `block_table` remains **required** in both cases: it drives the *read* path (which blocks make up
  each sequence's context). `slot_mapping` only controls writes.
- `past_seqlens` and `cumulative_sequence_length` remain required; they still define the causal mask
  and context length per sequence.

### 5.3 Implementation

`ReshapeAndCache` becomes templated on a slot resolver:

```cpp
struct DerivedSlotResolver { /* current binary-search + block_table derivation */ };
struct ExplicitSlotResolver { const int* slot_mapping;
                              __device__ int operator()(int token_id) const { return slot_mapping[token_id]; } };
```

The explicit resolver removes the per-thread binary search over `cumulative_seqlens_q`, which is a
measurable win for large batches and eliminates the OOB failure mode. The kernel guards
`slot < 0 → return` and (debug builds) `slot < num_blocks * block_size`.

`PagedAttentionData<T>::slot_mappings` already exists as an unused field; it is renamed to
`slot_mapping` and wired.

### 5.4 Validation

- Rank 1, `shape[0] == token_count`.
- Element type `int32`.
- Range checking on device is a debug-build assertion only; host-side range checking would require a
  D→H copy per step. The op documents that out-of-range values are undefined behavior, consistent
  with `block_table` today.

### 5.5 Why not derive-only

Deriving slots keeps the graph smaller but pushes scheduling policy into the kernel. Every serving
framework that matters (vLLM, TRT-LLM, SGLang) passes an explicit slot mapping precisely because
the scheduler — not the attention kernel — owns block allocation. Keeping the derived path as the
default preserves the simple single-turn case; the explicit path unlocks serving.

## 6. Feature: Attention Sink (`head_sink`) and Smooth Softmax

### 6.1 Math

Identical to GQA. With per-head sink value $s_h$ over $T$ attended positions:

$$
\text{softmax}_i = \frac{e^{x_i - m}}{e^{s_h - m} + \sum_{j} e^{x_j - m}}, \qquad m = \max\!\left(s_h, \max_j x_j\right)
$$

Equivalent to appending one extra logit $s_h$ that contributes to the denominator only. Used by
GPT-OSS. Providing `head_sink` implies smooth softmax:
`parameters.use_smooth_softmax = smooth_softmax_attr || head_sink != nullptr`.

### 6.2 Design: exact LSE epilogue (preferred)

FlashAttention varlen already produces `softmax_lse` and the buffer is already allocated in
`paged_attention.cc`. For query token $t$ and head $h$, Flash returns

$$
\ell_{t,h} = \log\!\sum_j e^{x_j}, \qquad o_{t,h} = \frac{\sum_j e^{x_j} v_j}{\sum_j e^{x_j}}
$$

Adding the sink changes only the denominator, so the corrected output is an **exact** elementwise
rescale:

$$
o^{\text{sink}}_{t,h} = o_{t,h} \cdot \frac{e^{\ell_{t,h}}}{e^{\ell_{t,h}} + e^{s_h}}
                      = o_{t,h} \cdot \frac{1}{1 + e^{\,s_h - \ell_{t,h}}}
$$

This is a single `token_count × num_heads × head_size` pass over the output, computed in FP32 and
cast back to `T`. It requires **no change to the Flash kernel**, composes with sliding window,
softcap, GQA grouping, and packed QKV, and is numerically exact (the $1/(1+e^{s-\ell})$ form is
stable for both signs of $s - \ell$).

Implementation: `LaunchApplyHeadSink(output, softmax_lse, head_sink, token_count, num_heads, head_size, stream)`
in `paged_attention_impl.cu`, invoked after `mha_varlen_fwd` when `data.head_sink != nullptr`.

### 6.3 MEA fallback

The CUTLASS fMHA path does not expose an LSE output in the current ORT wrapper. Options, in order of
preference:

1. Enable the fMHA LSE output (`output_accum` / `logsumexp` variant) and reuse the same epilogue.
2. Reject `head_sink` on the MEA path and require Flash (SM80+, FP16/BF16). Since PagedAttention is
   a server op and already requires Flash *or* MEA, and GPT-OSS deployments are SM80+, this is an
   acceptable Phase-1 restriction — but it must be a clear `INVALID_ARGUMENT`, not silent wrong math.

Phase 1 ships option 2 with option 1 as follow-up.

### 6.4 Validation

- Rank 1, `shape[0] == num_heads`, element type `T`.
- `smooth_softmax == 1` without `head_sink` is legal (sink value 0, i.e. an extra logit of 1.0 in the
  denominator), matching GQA.

## 7. Feature: Fused QK-Norm

### 7.1 Math

Identical to GQA §3. Per head, over `head_size` channels, applied to Q and K **before** RoPE:

$$
x_\text{norm}[c] = x[c] \cdot \frac{1}{\sqrt{\frac{1}{H}\sum_{j} x[j]^2 + \epsilon}} \cdot w[c]
$$

`H = head_size`; `w` is `q_norm_weight` for Q and `k_norm_weight` for K, both 1D `(head_size,)` of
type `T`, shared across heads; `epsilon = qk_norm_epsilon` (default `1e-6`). Sum of squares is
reduced in FP32.

Required by **Qwen3, Gemma 2/3, OLMo2, SmolLM3**.

### 7.2 Design

PagedAttention already has a prologue kernel that unpacks packed QKV, applies RoPE, and calls
`ReshapeAndCache`. QK-Norm is fused into that prologue:

```
[packed QKV split] → [QK-Norm on Q and K] → [RoPE on Q and K] → [ReshapeAndCache writes K,V]
```

Critically, **the normalized-and-RoPE'd K is what lands in the block cache**, matching GQA's
`UnpackRoPEAppend` ordering. Cached K is therefore directly consumable by attention on later steps —
QK-Norm is never re-applied to cached K.

### 7.3 Workspace implication

Today `workspace_buffer` is allocated only when `do_rotary_` or `is_packed_qkv`. QK-Norm requires a
writable Q (and K) buffer even when neither holds. The allocation condition becomes:

```cpp
const bool needs_prologue = do_rotary_ || parameters.is_packed_qkv || has_qk_norm;
if (needs_prologue) {
  workspace_buffer_bytes = sizeof(T) * token_count * (hidden_size + kv_hidden_size);
}
```

(When only QK-Norm is active, K must also be staged because its normalized form is what is cached.)

### 7.4 Validation

- `q_norm_weight` and `k_norm_weight` must be provided **together**; one alone is `INVALID_ARGUMENT`.
- Rank 1, `shape[0] == head_size`, element type `T`.
- `qk_norm_epsilon > 0`.

## 8. Feature: Quantized Paged KV Cache

### 8.1 Goal

Store the block cache in INT8 or FP8 E4M3 while `query` remains FP16/BF16, halving (or better) the
dominant memory consumer in a serving deployment and proportionally reducing HBM traffic on the
decode path. Scope for this phase: **`PER_TENSOR` and `PER_CHANNEL`**, `kv_cache_bit_width == 8`.
INT4 is deferred ([§19](#19-phasing)).

### 8.2 Schema

- `key_cache` / `value_cache` move from `T` to `T_CACHE ∈ {float16, bfloat16, int8, uint8, float8e4m3fn}`.
- `k_scale` / `v_scale` (inputs 14, 15), **always FP32**, matching GQA.
- Attributes `k_quant_type`, `v_quant_type` ∈ `{"NONE", "PER_TENSOR", "PER_CHANNEL"}`, and
  `kv_cache_bit_width`.
- Kernel becomes `PagedAttention<T, T_CACHE>`, registered for the same combinations GQA uses:
  `{MLFloat16, BFloat16} × {same as T, int8_t, Float8E4M3FN}` (plus `uint8_t` when INT4 lands).

### 8.3 Scale layout under the block layout

The block cache is `(num_blocks, block_size, kv_num_heads, head_size)`. A `PER_CHANNEL` scale of
shape `(kv_num_heads, 1, head_size)` — the same shape GQA uses — broadcasts naturally over the
leading `(num_blocks, block_size)` dims. **This is why the GQA scale shape is reused verbatim**: the
quantize/dequantize helpers index only on `(kv_head, channel)` and are layout-agnostic.

| Mode | Scale shape | Indexing |
|---|---|---|
| `PER_TENSOR` | `(1,)` | scalar |
| `PER_CHANNEL` | `(kv_num_heads, 1, head_size)` | `scale[h * head_size + c]` |

Symmetric quantization, same formulas as GQA:

| Type | Range | Quantize |
|---|---|---|
| INT8 | `[-128, 127]` | `q = clamp(round(x / scale), -128, 127)` |
| FP8 E4M3 | `[-448, 448]` | `q = clamp(x / scale, -448, 448)` — SM89+/SM90+ |
| INT4 (deferred) | `[-8, 7]`, 2/byte | last cache dim becomes `(head_size + 1) / 2` |

### 8.4 Write path

`ReshapeAndCache` gains the quantization step. After QK-Norm and RoPE, K/V are quantized as they are
scattered to their slots. This is a natural fit: the kernel is already elementwise over
`(token, kv_head, channel)`, which is exactly the `PER_CHANNEL` scale index.

### 8.5 Read path

Phase 2 (correctness first): **dequantize-on-gather**. The MEA fallback already materializes a
gathered `[total_kv_tokens, num_heads, head_size]` KV buffer via `GatherAndExpandPagedKVCache`; that
kernel is extended to dequantize while gathering, and the Flash varlen path is routed through the
same gather when the cache is quantized. Cost: one extra pass over the live context per step, and
the gather buffer is FP16/BF16-sized. Correct, low-risk, and reuses existing code.

Phase 3 (performance): a **paged decode kernel with in-kernel dequantization**, following the GQA
XQA approach — for `PER_TENSOR`, fold `k_scale` into the QK score scale and `v_scale` into the output
accumulator; for `PER_CHANNEL`, fold the K scale into Q before attention and apply the V scale to the
output afterward. Both are `O(num_heads * head_size)` passes and avoid touching the full cache. This
is the path that makes a quantized cache actually pay off; the gather-based Phase 2 mostly buys
memory capacity, not bandwidth.

### 8.6 Build gating

Mirror GQA: `onnxruntime_USE_FP8_KV_CACHE` (default ON), `onnxruntime_USE_INT4_KV_CACHE`
(default OFF). INT8 always built.

### 8.7 Validation

- `k_quant_type != "NONE"` requires `k_scale`; likewise for V. Conversely, a `k_scale` with
  `k_quant_type == "NONE"` is `INVALID_ARGUMENT`.
- `T_CACHE != T` requires a non-`NONE` quant type.
- `kv_cache_bit_width ∈ {8}` for this phase (`{4, 8}` later); must be consistent with `T_CACHE`.
- FP8 requires SM89+ (Ada) or SM90+; otherwise `INVALID_ARGUMENT` naming the required arch.
- `PER_CHANNEL` scale shape must be exactly `(kv_num_heads, 1, head_size)`.

> **Implementation note (P3, implemented).** Delivered: `int8` and `float8e4m3fn` caches with
> `PER_TENSOR` / `PER_CHANNEL` granularity, independently selectable for K and V, via
> `PagedAttention<T, T_CACHE>` registered for `{MLFloat16, BFloat16} × {int8_t, Float8E4M3FN}` in
> addition to the unquantized pairs. Deviations from the text above:
>
> - **Read path is §8.5 Phase 2 only.** `GatherAndExpandPagedKVCache` dequantizes while gathering,
>   and the Flash varlen path is routed through the same gather when the cache is quantized (using
>   `num_heads = kv_num_heads` so no GQA expansion happens, which keeps Flash's grouped layout). The
>   §8.5 Phase 3 paged decode kernel with in-kernel dequantization is **not** implemented.
> - Because a quantized cache never reaches Flash's *paged* kernel, the `block_size` tiling
>   constraint of §18.1 does not apply to it; Flash eligibility skips that check when the cache is
>   quantized. Any power-of-two `block_size >= 16` works with a quantized cache on either backend.
> - **`uint8` / INT4 not added.** `T_CACHE` is `{float16, bfloat16, int8, float8e4m3fn}`;
>   `kv_cache_bit_width` must be `8` for a quantized cache and `0` or `16` otherwise.
> - **No SM89/SM90 gate for FP8.** `Float8E4M3FN`'s converting constructor uses
>   `__nv_cvt_float_to_fp8`, which is available on every architecture ORT builds for from CUDA 11.8
>   onward, so the arch check in §8.7 would reject working configurations. FP8 remains gated at
>   *build* time by `onnxruntime_USE_FP8_KV_CACHE`.
> - Parity tests compare the updated cache at one quantization step of slack. Rotary and RMSNorm are
>   computed ~1 fp16 ULP differently on the host, which is enough to move a value across a rounding
>   boundary and flip the stored code by one LSB.

## 9. Feature: Sliding Window Attention

### 9.1 State

`local_window_size` is already an attribute and is already passed to Flash varlen as
`local_window_size - 1` (Flash's `window_size_left` excludes the current token; ORT's convention
includes it) and to the MEA params. Required by **Mistral, Gemma 2/3, Phi-3, GPT-OSS** (which
alternates full and sliding layers).

### 9.2 Work items

1. **Convention verification.** Assert that PagedAttention's window semantics match GQA's exactly —
   "the window includes the new token and only extends to the left". Add a direct GQA↔PagedAttention
   parity test at several window sizes, including `window >= context` (must equal full attention) and
   `window == 1`.
2. **Block-table pruning.** With a window of $W$ and context length $L_b$ for sequence $b$, all KV
   blocks whose highest position is below $L_b - W$ are unreachable. The read path should start at

   ```
   first_block = max(0, (L_b - W) / block_size)
   ```

   and only walk `block_table[b][first_block ...]`. For a 128K context with a 4K window this reduces
   both the gather volume and the number of block-table indirections by ~30×. This is the single
   largest win available for long-context sliding-window models and is *only* expressible in the
   paged layout.
3. **Freeing pruned blocks is a runtime concern**, not a kernel concern; the kernel must tolerate a
   `block_table` whose leading entries are stale or `-1`. Define `-1` in `block_table` as "block not
   mapped, treat as masked out", consistent with `slot_mapping`'s `-1`.
4. **Composition with `head_sink`.** GPT-OSS uses sinks *and* sliding windows in the same layer. The
   LSE epilogue in [§6](#6-feature-attention-sink-head_sink-and-smooth-softmax) composes trivially
   because Flash's LSE already reflects the window mask.
5. **Composition with `softcap`.** Both are already handled inside Flash varlen; add a combined test.

## 10. Feature: `attention_bias`

### 10.1 Difficulty

In a padded batch, `attention_bias` is naturally `(batch, num_heads, q_len, kv_len)`. In a packed
varlen batch with a paged cache there is no single `kv_len`, and a dense bias over the union of all
contexts would be enormous. This feature is therefore the **lowest priority** and the most likely to
be replaced by a narrower input.

### 10.2 Design

Input 16, `attention_bias`, type `T`, shape `(1 or num_heads, token_count, max_context_len)` where
`max_context_len = max_b (past_seqlens[b] + q_len_b)`. Row `t` is indexed by the **absolute KV
position within token `t`'s own sequence**; entries beyond that sequence's context are ignored.

- Rejected on the Flash varlen path (Flash has no additive-bias entry point in ORT's wrapper).
  Supported on the MEA / unfused paths only.
- Bias is applied after `scale` and before `softcap`, matching GQA.
- Emit an explicit `INVALID_ARGUMENT` when `attention_bias` is combined with a backend that cannot
  serve it, naming the backend and the reason.

### 10.3 Preferred alternative

For the dominant real use case — **ALiBi** — a dense bias is wasteful. FlashAttention supports
`alibi_slopes` of shape `(num_heads,)` natively. Recommendation: land `alibi_slopes` first (cheap,
fused, works on the fast path) and treat the general dense `attention_bias` as a correctness-only
fallback for models that genuinely need arbitrary additive masks. Decision deferred to
[§20](#20-open-questions).

## 11. Feature: `output_qk`

### 11.1 Design

- Attribute `qk_output`: `0` = no output (default), `1` = pre-softmax scores, `2` = post-softmax
  probabilities. Same encoding as GQA's `QKOutputType`.
- Output 3, `output_qk`, type `T`, shape `(num_heads, token_count, max_context_len)`.
- Use cases: interpretability, speculative-decode scoring, kernel debugging and parity triage.

### 11.2 Constraints

- Supported on the **unfused / MEA** paths only. Fused Flash kernels never materialize the score
  matrix; requesting `output_qk` forces a fallback backend and is documented as such.
- Memory is `O(num_heads × token_count × max_context_len)` and can dwarf the KV cache itself.
  The op must reject configurations where the output tensor would exceed a configurable byte
  threshold rather than attempting the allocation.
- `qk_output != 0` with fewer than 4 outputs, or 4 outputs with `qk_output == 0`, is
  `INVALID_ARGUMENT`. Shape inference must only touch output 3 when `ctx.getNumOutputs() > 3`.

## 12. Feature: Multi-head Latent Attention (MLA)

### 12.1 What MLA is

DeepSeek-V2 / V3 / R1 replace the per-head K/V cache with a single low-rank **latent** vector per
token:

- `compressed_kv` of width `kv_lora_rank` (512 in V3), shared by all heads;
- `k_pe` of width `qk_rope_head_dim` (64), shared by all heads (MQA-style), carrying RoPE.

The cached footprint is `512 + 64 = 576` elements per token, against
`2 × num_heads × head_size = 2 × 128 × 128 = 32768` for a comparable MHA model — a ~57× reduction.
That reduction is the entire point of MLA, and it is a **paged-serving** feature: the win lands in
decode, where the KV cache dominates both capacity and bandwidth. MLA therefore belongs in
`PagedAttention` rather than in GQA.

MLA has two mathematically equivalent evaluation forms:

| Form | What attention sees as K/V | `head_size` | `v_head_size` | `kv_num_heads` | Used for |
|---|---|---|---|---|---|
| **Non-absorbed** | per-head `k = [k_nope(128); k_pe(64)]`, `v(128)`, produced by applying `W_UK` / `W_UV` to the latent | 192 | 128 | `num_heads` | Prefill with no cached prefix |
| **Absorbed** | the latent itself: `k = [compressed_kv(512); k_pe(64)]`, `v = compressed_kv(512)` | 576 | 512 | 1 | Decode, chunked prefill, prefix caching |

Absorption folds `W_UK` into Q (`q_nope' = q_nope @ W_UKᵀ`, 128 → 512) and `W_UV` into the output
projection. Both are ordinary **MatMuls in the graph**; the attention op never sees a projection
weight. This keeps the operator an attention operator.

### 12.2 Design: MLA is absorbed-form MQA with `v_head_size < head_size`

Absorbed MLA is **already** the operator `PagedAttention` is, except for three properties:

1. `v_head_size != head_size` (512 vs. 576).
2. V is not a separate tensor — it is the **leading `v_head_size` channels of K**. There is one
   cache, not two.
3. RoPE applies to a **suffix** of the head dimension (channels 512–575), not a prefix.

Everything else — packed varlen Q, `block_table`, `slot_mapping`, in-place cache update, causal
masking over ragged sequences, quantized cache — is unchanged. MLA is therefore added as a **mode of
`PagedAttention`**, expressed entirely through two new attributes, and not as a new operator. This is
the same conclusion FlashMLA, vLLM's MLA backend, and SGLang reached: absorbed MLA decode is MQA with
a wide head and a shared K/V buffer.

### 12.3 Schema additions

| Addition | Meaning |
|---|---|
| Attribute `v_head_size` (INT, default `0`) | Head width of V and of each output head. `0` means "same as `head_size`" — i.e. every non-MLA model is unaffected. |
| Attribute `rotary_offset` (INT, default `0`) | First channel within `head_size` covered by RoPE (§12.5). |
| Input 4 `value_cache` becomes `Optional` | When absent, V is the leading `v_head_size` channels of `key_cache`. |
| Input 2 `value` absent while `key` is present becomes legal | Previously an error; now signals the V-aliases-K mode. `key` absent still means packed QKV. |
| Output 0 shape generalized to `(token_count, num_heads * v_head_size)` | Identical to today when `v_head_size == head_size`. |

All five are backward compatible: existing models set neither attribute, supply `value_cache`, and
get byte-identical behavior.

### 12.4 Concrete node contract (DeepSeek-V3, absorbed)

| Tensor / attribute | Value |
|---|---|
| `query` | `(token_count, 128 * 576)` — per head `[q_nope @ W_UKᵀ (512); q_rope (64)]` |
| `key` | `(token_count, 1 * 576)` — `[compressed_kv (512); k_pe (64)]` after `kv_a_layernorm` |
| `value` | absent |
| `key_cache` | `(num_blocks, block_size, 1, 576)` |
| `value_cache` | absent (aliases `key_cache`) |
| `block_table`, `slot_mapping`, `cumulative_sequence_length`, `past_seqlens` | as for any paged node |
| `output` | `(token_count, 128 * 512)`, consumed by the `W_UV`-absorbed output projection |
| `num_heads` / `kv_num_heads` | `128` / `1` |
| `head_size` (derived) | `576` |
| `v_head_size` | `512` |
| `rotary_offset` | `512` |
| `scale` | **must be set explicitly** (§12.6) |

### 12.5 Offset (partial) RoPE

`rotary_dim` is derived from `cos_cache` as today (`2 × cos_cache.shape[1]`). `rotary_offset` selects
where it starts: RoPE covers channels `[rotary_offset, rotary_offset + rotary_dim)` of each head and
channels outside that range are copied through unchanged. Default `0` reproduces current behavior
exactly. For absorbed MLA, `rotary_offset = kv_lora_rank = 512` and `rotary_dim = 64`.

RoPE must be applied to K **before** the latent is written to the cache, so that the cache holds the
already-rotated `k_pe`. This matches the DeepSeek reference implementation and means RoPE is never
re-applied on cache reads. Exporters may equivalently apply RoPE in the graph and set `do_rotary=0`;
both spellings must produce identical results and both are covered by tests (§17).

### 12.6 The softmax-scale trap

DeepSeek's softmax scale is derived from the **pre-absorption** head width:
`scale = mscale² / sqrt(qk_nope_head_dim + qk_rope_head_dim)` = based on `192`, with an additional
YaRN `mscale` factor. It is *not* `1/sqrt(576)`.

The operator's default (`scale == 0 → 1/sqrt(head_size)`) would therefore silently compute
`1/sqrt(576)` and produce plausible-but-wrong logits — the worst possible failure mode. Rule:
**when `v_head_size` is set and differs from `head_size`, an explicit `scale` attribute is
required**; omitting it is `INVALID_ARGUMENT`. The op refuses to guess.

### 12.7 Kernel backend

MLA is the one feature in this document that **cannot** be served by the existing backends:

- ORT's vendored FlashAttention caps `head_size` at 256 and requires `v_head_size == head_size`, so
  it cannot run 576/512.
- The CUTLASS fMHA (MEA) wrapper in ORT is constrained the same way.

An MLA-capable backend is required. Candidates in order of preference:

1. **FlashMLA** (DeepSeek, SM90) — purpose-built paged MLA decode with 576/512 and a shared K/V
   buffer; it consumes precisely the cache layout proposed in §12.4.
2. **FlashInfer MLA** (SM80+) — wider architecture coverage, also paged.
3. **TensorRT-LLM MLA / XQA-MLA** kernels.
4. **cuDNN SDPA**, where it exposes asymmetric head dimensions over a paged cache.
5. **Unfused reference** — needed regardless, as the correctness oracle and for architectures with no
   fused kernel.

Recommended sequencing: land the unfused reference first (correct on any architecture, unblocks the
test matrix), then integrate exactly one fused backend chosen by target hardware. Adding a second
fused backend before the reference exists makes parity failures undebuggable.

### 12.8 Non-absorbed prefill

A prefill with **no cached prefix** does not read the paged cache at all: K and V come entirely from
the current chunk. Running it in absorbed form is correct but inflates QK FLOPs ~4× (512-wide instead
of 128-wide `q_nope`), so the recommended graph shape there is ordinary varlen attention
(`head_size = 192`, `v_head_size = 128`, `kv_num_heads = num_heads`) over the decompressed tensors,
plus a cache write of the **latent** form.

Because the attended tensors and the cached tensor differ in that case, the write must be decoupled
from attention. Two supported spellings, neither of which needs a new operator:

1. **Graph-level `ScatterND`** — view the cache as `(num_blocks * block_size, 1, 576)` and scatter
   the latent rows using `slot_mapping` as indices. Pure standard ONNX.
2. **`PagedAttention` with `slot_mapping` entirely `-1`** (writes suppressed, §5) alongside the
   scatter of option 1.

Chunked prefill and prefix caching *do* read cached history and must therefore use the absorbed form,
exactly as decode does. The operator itself does not distinguish the three cases — the graph does.

### 12.9 Interaction with the other features

| Feature | With MLA | Rationale |
|---|---|---|
| `slot_mapping` (§5) | **Supported** | Orthogonal — write path only, and the latent is written exactly like any K/V row. |
| Quantized cache (§8) | **Supported** | An FP8 latent cache is standard in DeepSeek deployments. `PER_CHANNEL` scale shape becomes `(1, 1, head_size)` since `kv_num_heads == 1`. |
| RoPE | **Supported** via `rotary_offset` (§12.5) | |
| `softcap` | Allowed, unused by DeepSeek | |
| Sliding window (§9) | Allowed but untested | No MLA model uses it; semantics are well defined (window is over positions, not channels). |
| `head_sink` (§6) | **Rejected** | No MLA model uses sinks. The LSE epilogue is valid math here, but shipping an untested combination invites silent errors. Revisit if a model needs it. |
| QK-Norm (§7) | **Rejected** | DeepSeek's `q_a_layernorm` / `kv_a_layernorm` act on the *latent* projections in the graph, before absorption. A `head_size`-wide RMSNorm in absorbed space is a different operation; accepting it would let an exporter produce silently wrong math. |
| `attention_bias` / `output_qk` (§10, §11) | Supported on the unfused path | `output_qk` shape is unchanged: `(num_heads, token_count, max_context_len)`. |

### 12.10 Validation

- `v_head_size ∈ [1, head_size]`; `0` means "equal to `head_size`" (non-MLA).
- `v_head_size != head_size` requires an explicit `scale` attribute (§12.6).
- `v_head_size != head_size` requires `value` and `value_cache` to be **absent**, or `value_cache` to
  be the identical tensor to `key_cache`. A distinct V cache with a different head width is not
  supported and is `INVALID_ARGUMENT`.
- `key` present + `value` absent ⇒ MLA mode. `key` absent + `value` absent ⇒ packed QKV (unchanged).
  `key` absent + `value` present remains an error.
- `rotary_offset >= 0`, `rotary_offset % 8 == 0`, `rotary_offset + rotary_dim <= head_size`.
- In MLA mode `head_size` may exceed 256, but the selected backend must accept it; otherwise return
  `INVALID_ARGUMENT` naming the backend and the supported head widths.
- `kv_num_heads` is `1` for DeepSeek but any divisor of `num_heads` is accepted.
- `head_sink` or QK-Norm weights combined with `v_head_size != head_size` ⇒ `INVALID_ARGUMENT`.

## 13. Kernel Dispatch and Backend Plan

Target dispatch order once all phases land, first eligible wins:

| Priority | Backend | Eligible when |
|---|---|---|
| 0 | **MLA backend** (new, §12.7) | `v_head_size != head_size` — FlashMLA / FlashInfer MLA / TRT-LLM MLA where available, unfused MLA reference otherwise |
| 1 | **Paged decode kernel** (new, Phase 4) | All sequences contribute exactly 1 new token; non-quantized or `PER_TENSOR`/`PER_CHANNEL` INT8/FP8; `head_size ∈ {64, 128, 256}`; sliding window and `head_sink` supported |
| 2 | **FlashAttention varlen** | FP16/BF16, SM80+, non-quantized cache (or quantized via dequant-gather); supports sliding window, softcap, packed QKV, `head_sink` via LSE epilogue |
| 3 | **Memory-Efficient Attention (CUTLASS fMHA)** | Fallback; needed for `attention_bias`, `output_qk`, and pre-SM80 |
| 4 | **Unfused** | Last resort — arbitrary `head_size`, `output_qk` |

Feature × backend matrix (target state):

| Feature | Paged decode | Flash varlen | MEA | Unfused |
|---|---|---|---|---|
| Sliding window | Yes | Yes | Yes | Yes |
| `softcap` | Planned | Yes | Yes | Yes |
| `head_sink` | Yes (native) | Yes (LSE epilogue) | After fMHA LSE | Yes |
| QK-Norm | Yes (prologue) | Yes (prologue) | Yes (prologue) | Yes |
| Quantized cache | Yes (in-kernel) | Via dequant-gather | Via dequant-gather | Via dequant-gather |
| `attention_bias` | No | No | Yes | Yes |
| `output_qk` | No | No | Yes | Yes |
| `slot_mapping` | Yes (write path — backend independent) | Yes | Yes | Yes |
| MLA (`v_head_size != head_size`) | No | No | No | Yes — plus the dedicated MLA backend (§12.7) |

QK-Norm, RoPE, offset RoPE, packed-QKV unpacking, quantized writes, and `slot_mapping` all live in
the **prologue** and are therefore backend independent. Only the attention math itself varies by
backend.

The selected backend must be reported through `AttentionKernelDebugInfo` (`SdpaKernel=...`) exactly
as GQA does, so that `ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO=1` works uniformly across both ops.

## 14. Shared Code with GroupQueryAttention

Feature drift between the two ops is the principal long-term risk of keeping them separate. The
mitigation is structural, not procedural.

1. **Common parameter base.** `PagedAttentionParameters` and `GroupQueryAttentionParameters` already
   derive from `AttentionParameters` in `contrib_ops/cpu/bert/attention_parameters.h`. Move the
   fields that are genuinely shared — `local_window_size`, `softcap`, `use_smooth_softmax`,
   `qk_norm_epsilon`, `k_quant_type`, `v_quant_type`, `kv_cache_bit_width`, `rotary_interleaved` —
   into the base so a feature has one definition. `v_head_size` and `rotary_offset` (§12) start in
   `PagedAttentionParameters`; promote them if and when a second op needs asymmetric head widths.

2. **Shared validation helpers.** `paged_attention_helper.h` already calls
   `group_query_attention_helper::CheckRotaryCaches`. Extend the same pattern for
   `CheckQKNormWeights`, `CheckHeadSink`, and `CheckKVQuantScales`, so both ops enforce identical
   shapes and identical error text.

3. **KV accessor abstraction (the important one).** Introduce a device-side accessor concept:

   ```cpp
   struct ContiguousKVAccessor { /* BNSH, (b, h, s, d) -> offset */ };
   struct PagedKVAccessor      { /* block_table + block_size, (b, h, s, d) -> offset */ };
   ```

   Prologue kernels (QK-Norm → RoPE → quantize → store) and any future fused attention kernel are
   templated on the accessor. A new feature is then written once and both ops receive it. This is the
   only mechanism that makes "separate ops, shared math" hold over time.

4. **Shared epilogues.** `LaunchApplyHeadSink`, per-channel dequant scaling, and softcap helpers live
   in a common `attention_epilogues.cuh` used by both.

## 15. Validation Rules

Consolidated, to be implemented in `paged_attention_helper::CheckInputs`. Every violation returns
`INVALID_ARGUMENT` with a message naming the offending tensor and the expected value.

**Existing (retain):**
- `num_heads % kv_num_heads == 0`; `num_heads <= max_threads_per_block`.
- `query` rank 2; `head_size % 8 == 0`.
- `key`/`value` both present or both absent; `key`/`value` rank 2 with dim 0 == `token_count`.
- Packed QKV: `hidden_size % (num_heads + 2 * kv_num_heads) == 0`.
- `key_cache`/`value_cache` rank 4, identical shapes, dim 2 == `kv_num_heads`, dim 3 == `head_size`.
- `cumulative_sequence_length` rank 1 with `dim0 >= 2`; `past_seqlens` rank 1 with `dim0 == batch_size`.
- `block_table` rank 2 with `dim0 == batch_size`.
- `cos_cache`/`sin_cache` both present or both absent; required when `do_rotary == 1`.
- `key_cache_out` must alias `key_cache`; same for value.
- `batch_size <= 256` (BlockScan limitation — to be lifted, see §18).

**Corrected:**
- `block_size` must be a power of two in `{16, 32, 64, 128, 256}` (**replaces** `block_size % 256 == 0`; see §18).
  *Implemented as: any power of two `>= 16`. Values that FlashAttention cannot address for the given
  `head_size` select the memory-efficient backend rather than failing — see the note in §18.1.*

**New:**
- `slot_mapping`: rank 1, `dim0 == token_count`, `int32`.
- `head_sink`: rank 1, `dim0 == num_heads`, type `T`.
- `q_norm_weight` / `k_norm_weight`: both-or-neither; rank 1, `dim0 == head_size`, type `T`.
- `k_scale` / `v_scale`: FP32; scalar for `PER_TENSOR`, `(kv_num_heads, 1, head_size)` for `PER_CHANNEL`;
  present iff the corresponding quant type is not `NONE`.
- `T_CACHE != T` iff a quant type is not `NONE`.
- `kv_cache_bit_width` consistent with `T_CACHE`; FP8 requires SM89+/SM90+.
- `attention_bias`: rank 3, `dim0 ∈ {1, num_heads}`, `dim1 == token_count`; rejected on Flash.
- `qk_output != 0` iff the node has 4 outputs; `output_qk` rejected on Flash.
- MLA (§12.10): `v_head_size ∈ [1, head_size]`; `v_head_size != head_size` requires an explicit
  `scale`, absent `value` / `value_cache` (or `value_cache` identical to `key_cache`), and no
  `head_sink` or QK-Norm weights. `rotary_offset % 8 == 0` and
  `rotary_offset + rotary_dim <= head_size`.
- Backend-incompatible feature combinations must be reported at `Compute` entry with the reason,
  never silently ignored.

## 16. Shape Inference and Tooling

- `PagedAttentionTypeAndShapeInference` in `onnxruntime/core/graph/contrib_ops/bert_defs.cc`:
  - Output 0 dim 1 becomes `num_heads * v_head_size`, where `v_head_size` defaults to the derived
    `head_size` when the attribute is unset. Both the unpacked and packed-QKV branches must apply
    this, otherwise every MLA graph gets a wrong — and silently propagated — output width.
  - Outputs 1/2 must propagate **type from inputs 3/4** (not from input 0) once `T_CACHE` can differ
    from `T`. The current code propagates elem type from input 0 to outputs 1/2 first and then from
    3/4 — the first propagation becomes wrong under quantization and must be removed.
  - `value_cache` (input 4) is now optional: guard `getInputShape(ctx, 4)` and the output-2
    propagation on `ctx.hasInput(4)` before any write.
  - Output 3 (`output_qk`) is written only when `ctx.getNumOutputs() > 3`, guarded before any write,
    consistent with the contrib-op shape-inference memory-safety rules.
- `onnxruntime/python/tools/symbolic_shape_infer.py` — `_infer_PagedAttention` must handle the new
  optional inputs and the fourth output.
- Regenerate `docs/ContribOperators.md` and `docs/OperatorKernels.md`.
- Update `onnxruntime/python/tools/transformers/fusion_options.py` and any fusion that emits
  `PagedAttention` so new attributes get explicit values.

## 17. Testing Plan

### 17.1 GQA ↔ PagedAttention cross-parity harness

The highest-value test. For a given configuration, construct a block table that maps each sequence's
blocks contiguously and in order, build the equivalent padded GQA inputs, and assert
`PagedAttention` output == `GroupQueryAttention` output within tolerance. This makes GQA the
reference oracle for every shared feature and catches drift automatically. Run it across:

- `head_sink` on/off × sliding window on/off × `softcap` on/off
- QK-Norm on/off × RoPE on/off × interleaved on/off
- packed QKV vs. unpacked
- quantized cache: `NONE`/`PER_TENSOR`/`PER_CHANNEL` × INT8/FP8

### 17.2 Paging-specific tests (no GQA equivalent)

- `slot_mapping`: explicit mapping equals derived mapping when the mapping is the derived one.
- `slot_mapping` with `-1` entries: skipped tokens leave the cache byte-identical, and attention
  output matches a run where those positions were pre-populated (prefix-cache simulation).
- Non-contiguous / shuffled / shared block tables — including two sequences sharing prefix blocks.
- `block_table` entries of `-1` (unmapped) are masked out.
- Ragged batches including sequences with **zero** new tokens.
- `token_count == 0` early-out.
- `batch_size` at and just above the BlockScan limit (must error, not corrupt).
- Sliding-window block pruning: pruned run bit-matches the unpruned run.

### 17.3 MLA tests

GQA cannot be the oracle here — it has no MLA mode — so MLA needs its own reference chain:

- **Absorbed ↔ non-absorbed equivalence.** For a random `W_UK` / `W_UV`, assert that absorbed-form
  `PagedAttention` (576/512, `kv_num_heads=1`) matches a non-absorbed reference (192/128,
  `kv_num_heads=num_heads`) built from the decompressed latent. This is the single test that proves
  the whole design and must run before any fused MLA kernel is integrated.
- **HuggingFace parity.** Compare one DeepSeek-V2-Lite decoder layer against the reference PyTorch
  implementation, including `mscale`/YaRN scaling, over prefill and several decode steps.
- **`rotary_offset`.** In-op offset RoPE (`do_rotary=1`, `rotary_offset=512`) must bit-match the
  graph-applied spelling (`do_rotary=0`, RoPE fused into the producer).
- **Scale guard.** Omitting `scale` while `v_head_size != head_size` must fail with a clear error —
  a regression here produces plausible-looking but wrong logits (§12.6).
- **V-aliases-K.** `value_cache` absent and `value_cache` passed as the same tensor as `key_cache`
  must produce identical results; a distinct V cache must be rejected.
- **MLA × paging.** Shuffled block tables, `slot_mapping` with `-1`, and an FP8 latent cache, each
  combined with MLA.
- **Rejected combinations.** MLA + `head_sink` and MLA + QK-Norm must fail with the documented
  message, not silently compute something.

### 17.4 Reference implementations

Extend `onnxruntime/test/python/transformers/test_paged_attention_cuda.py` with a PyTorch reference
covering sinks (`smooth_softmax_ref`), QK-Norm (RMSNorm-before-RoPE), per-channel dequantization,
windowing, and MLA absorption — reusing the GQA test helpers rather than duplicating them.

### 17.5 Negative tests

One test per validation rule in [§15](#15-validation-rules), asserting the error *message*, not just
failure — so that backend-incompatible combinations cannot regress into silent wrong results.

## 18. Known Defects to Fix First

These block the feature work and should land ahead of it.

1. **`block_size % 256 == 0` is wrong.** `paged_attention_helper.h::CheckKVCache` requires the block
   size *in tokens* to be a multiple of 256. Conventional paged block sizes are 16/32/64/128 tokens;
   256-token blocks defeat the fine-grained allocation that paging exists to provide, and a 16-token
   block table (the vLLM default) is rejected outright. The in-code `TODO(aciddelgado): block size
   multiple of 8` suggests the intent was an alignment constraint on the *innermost* dimension.
   Replace with `block_size ∈ {16, 32, 64, 128, 256}` and, if a byte-alignment constraint is truly
   needed, express it on `block_size * head_size * sizeof(T_CACHE)`.

   > **Implementation note (correction).** The constraint is *not* purely a validation bug. The
   > vendored FlashAttention split-KV kernel builds each `gK`/`gV` tile as a single contiguous
   > `kBlockN × head_size` region addressed by one `(block_table_idx, block_table_offset)` pair, so a
   > tile must never straddle a page. That requires `block_size % kBlockN == 0`, where
   > `kBlockN = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64)`
   > (`flash_fwd_launch_template.h::run_mha_fwd_splitkv_dispatch`). Relaxing the *validation* alone
   > would produce silent garbage on the Flash path.
   >
   > What was implemented instead: `CheckKVCache` accepts any power-of-two `block_size >= 16` (a
   > superset of `{16, 32, 64, 128, 256}` that also keeps the existing `block_size = 512` test case
   > valid), and `paged_attention.cc` treats `block_size % kBlockN == 0` as part of Flash backend
   > *eligibility*. When a model uses a smaller page than Flash can address, the op transparently
   > falls back to the memory-efficient backend, which gathers pages into a dense buffer first and
   > therefore accepts any block size. The op only errors when neither backend is eligible.
   > Lifting this properly requires teaching the Flash paged loader to split a tile across pages.
2. **Out-of-bounds binary search.** The binary search over `cumulative_seqlens_q` in
   `ReshapeAndCache` and `GatherAndExpandPagedKVCache` can yield `batch_id == batch_size` when
   `token_id >= cumulative_seqlens_q[batch_size]`, producing OOB reads of `past_seqlens` and
   `block_table`. Guard with an early `return` when
   `token_id >= cumulative_seqlens_q[batch_size]`. (`slot_mapping` removes the search entirely on the
   write path, but the gather path still needs the fix.)
3. **`batch_size <= 256`.** Replace the per-block `cub::BlockScan` with a grid-wide scan (or a single
   256-thread block doing a strided serial scan) so continuous batching is not capped at 256
   concurrent sequences — a real limit for a serving op.
4. **Per-step D→H synchronization.** `max_query_len` (and `total_kv_tokens` for MEA) are obtained via
   `cudaStreamSynchronize` every step, serializing the pipeline. Either accept these as optional
   host-side scalar inputs supplied by the scheduler (which already knows them), or compute an upper
   bound on device. This is a throughput bug for the op's primary use case.

## 19. Phasing

| Phase | Contents | Schema delta |
|---|---|---|
| **P0 — Foundation** | §18 defect fixes; sliding-window semantic verification and GQA parity harness (§17.1) | none |
| **P1 — Paging primitives** | `slot_mapping` (§5); sliding-window block pruning (§9.2) | input 10 |
| **P2 — Model coverage** | `head_sink` + `smooth_softmax` via LSE epilogue (§6); fused QK-Norm (§7) | inputs 11–13, attrs `smooth_softmax`, `qk_norm_epsilon` |
| **P3 — Memory** | Quantized cache INT8/FP8, `PER_TENSOR` + `PER_CHANNEL`, dequant-on-gather read path (§8) | `T_CACHE`, inputs 14–15, 3 attrs |
| **P4 — MLA (correctness)** | `v_head_size`, `rotary_offset`, V-aliases-K, optional `value_cache`, unfused MLA reference kernel, absorbed↔non-absorbed equivalence tests (§12) | attrs `v_head_size`, `rotary_offset`; input 4 optional |
| **P5 — Performance** | Paged decode kernel with in-kernel dequant; fused MLA backend (FlashMLA / FlashInfer MLA, §12.7); `softcap` on decode; lift D→H sync | none |
| **P6 — Completeness** | `attention_bias` / `alibi_slopes` (§10); `output_qk` (§11) | input 16, output 3, attr `qk_output` |
| **Later** | INT4 cache; MLA quantized latent cache tuning; non-CUDA EPs | — |

P0–P2 cover GPT-OSS, Qwen3, Gemma 2/3, Llama, Mistral and Phi. P3 and P5 are what make the op
competitive for throughput-oriented serving. P4–P5 add DeepSeek-V2/V3/R1.

P4 is deliberately split from P5: the schema work and the unfused reference are small, low-risk, and
unblock the test matrix, whereas integrating a fused MLA kernel is a large dependency decision
(§12.7, §20). Shipping the reference first means the fused kernel arrives with an oracle already in
place.

Note that MLA is folded into `PagedAttention` rather than given its own operator precisely because
the *only* differences are two attributes and a K/V aliasing rule (§12.2) — the layout, batching
model, and cache management are identical. That is the opposite of the GQA-vs-PagedAttention case in
§1, where the query rank and the cache contract genuinely differ.

## 20. Open Questions

1. **`alibi_slopes` vs. dense `attention_bias` (§10).** Should P6 ship the narrow, fused
   `alibi_slopes` input instead of, or in addition to, the general dense bias?
2. **Host-scalar inputs for `max_query_len` / `total_kv_tokens` (§18.4).** Adding them as optional
   inputs removes a per-step sync but leaks scheduler state into the graph. Is that acceptable?
3. **`block_table == -1` semantics (§9.2).** Confirm "unmapped, masked out" rather than "invalid,
   error" — the former is required for sliding-window block eviction.
4. **Non-CUDA EPs.** Is `PagedAttention` a server-GPU-only op (CUDA, ROCm, TensorRT), or is a CPU
   reference implementation required? A CPU reference has real value as a test oracle even if it is
   never used in production. Stub kernels returning `NOT_IMPLEMENTED` are not an acceptable middle
   ground (§3.4).
5. **ORT-GenAI commitment.** Does the continuous-batching path in `onnxruntime-genai` commit to
   emitting `PagedAttention` (as opposed to a `block_table`-extended GQA)? This determines the
   priority of everything above.
6. **Ownership of the parity matrix (§13).** Who keeps the feature × backend table current, and is
   the KV-accessor refactor (§14.3) in scope for P2 or a follow-up?
7. **Which fused MLA backend (§12.7)?** FlashMLA is the closest fit but is SM90-only and adds a
   third-party dependency; FlashInfer MLA covers SM80+; TRT-LLM has its own. The choice determines
   the hardware matrix ORT can serve DeepSeek on and should be made before P5 starts.
8. **Does MLA need a non-absorbed *paged* path (§12.8)?** The proposal handles prefill-with-prefix by
   absorbing. If a workload shows the absorbed prefill FLOP inflation to be material, an in-op
   decompression path would require passing `W_UK` / `W_UV` into the operator — which this design
   deliberately avoids. Needs a measurement before it is reconsidered.
9. **MLA + quantized latent cache (§12.9).** FP8 on a 576-wide latent shared by 128 query heads has
   a different error profile from FP8 on a per-head cache. Accuracy validation is required before
   the combination is recommended, even though the schema supports it on day one.
