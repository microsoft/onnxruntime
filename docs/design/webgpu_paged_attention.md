# Design: WebGPU PagedAttention

**Status**: Draft (v1 in progress)
**Target**: WebGPU EP, `com.microsoft::PagedAttention` v1
**Owner**: TBD
**Precision**: `MLFloat16` only in v1

---

## 1. Motivation

ONNX Runtime GenAI's continuous-batching engine relies on the `PagedAttention`
contrib op to pack multiple in-flight sequences into a single ONNX graph
step. It stores the KV cache in a shared paged pool (`[num_blocks, block_size,
num_kv_heads, head_size]`) indexed per-request by a `block_table`, and packs
all requests' query tokens into a single 1-D `input_ids` axis.

Today only the CUDA EP implements this op. That gates GenAI's continuous
batching to CUDA-only. This doc designs the WebGPU EP implementation so the
same model artifact (produced by `builder.py --extra_options
use_paged_attention=true`) can run in WebGPU-backed deployments (Chromium
tabs, Electron desktop apps, native WebGPU on Windows/macOS via Dawn).

### 1.1 Related PRs

- **[microsoft/onnxruntime#29867](https://github.com/microsoft/onnxruntime/pull/29867)** — draft CPU + WebGPU stubs (`Compute()` returns `NOT_IMPLEMENTED`) and helper cleanups.
- **[microsoft/onnxruntime#29912](https://github.com/microsoft/onnxruntime/pull/29912)** — schema extension: `T_CACHE`/`T_KV_SCALE` type constraints; new optional inputs `slot_mapping`, `head_sink`, `q_norm_weight`, `k_norm_weight`, `k_scale`, `v_scale`, `attention_metadata`; new attributes `k_cache_dtype`, `v_cache_dtype`, `k_quant_type`, `v_quant_type`, `kv_cache_layout` (`SEPARATE`/`LATENT`), `v_head_size`, `rotary_offset`, `qk_norm_epsilon`, `use_smooth_softmax`. Also adds portable split-KV decode + XQA quantized decode kernels on CUDA.
- **[microsoft/onnxruntime-genai#2330](https://github.com/microsoft/onnxruntime-genai/pull/2330)** — model builder emits `PagedAttention` in place of GQA when `use_paged_attention=true`; scheduler feeds `block_table` / `cumulative_sequence_lengths` / `past_sequence_lengths`. Currently gated to `-e cuda` and fp16/bf16.
- **[microsoft/onnxruntime-genai#2333](https://github.com/microsoft/onnxruntime-genai/pull/2333)** — follow-up: quantized KV cache in the builder, `attention_metadata` input wiring (index 16), CUDA-graph capture in the continuous-batching engine, block-accounting bug fix.

### 1.2 Non-goals for v1

- **Quantized KV cache** (`T_CACHE ∈ {int8, fp8e4m3fn}`). Deferred to Phase 3. WebGPU doesn't have an fp8 storage type at all; int8 is doable but not on the v1 critical path.
- **LATENT / MLA layout.** Deferred to Phase 4. No customer need on WebGPU yet.
- **QK-Norm and head-sink** (schema additions in #29912). Deferred to Phase 2.
- **Speculative-decoding `slot_mapping = -1` semantics.** Accepted-but-ignored in v1 (the input is validated, the sentinel branch is a one-line follow-up).

---

## 2. Design constraints (locked in)

| Question | Answer |
|---|---|
| `block_size` | Mirror CUDA: `block_size % 256 == 0`. Same model artifact GenAI produces today. Revisit only if a WebGPU adapter proves a smaller page is worth the model-side divergence. |
| Precision | `MLFloat16` only. Registered as a single typed kernel. |
| Schema baseline | Build v1 against the **current-on-main** schema (10 inputs / 3 outputs). Adopt PR #29912's expanded surface once it merges. |
| `slot_mapping = -1` | v1 accepts `slot_mapping` when present and uses it as an authoritative override of the derived slot, but leaves the "skip write on negative" branch out. GenAI does not emit `< 0` today. Adding it later is a one-line shader change. |

---

## 3. Op contract

### 3.1 Current-on-main schema (v1 target)

| # | Name | Kind | Shape | Type | Notes |
|---|---|---|---|---|---|
| 0 | `query` | Input | `(num_tokens, hidden_size)` or packed `(num_tokens, num_heads*head_size + 2*kv_num_heads*head_size)` | `T` | Packed layout when `key`/`value` are absent. |
| 1 | `key` | Input (opt) | `(num_tokens, kv_hidden_size)` | `T` | Absent iff Q is packed QKV. |
| 2 | `value` | Input (opt) | `(num_tokens, kv_hidden_size)` | `T` | Absent iff Q is packed QKV. |
| 3 | `key_cache` | Input | `(num_blocks, block_size, kv_num_heads, head_size)` | `T` | Updated **in-place**. |
| 4 | `value_cache` | Input | `(num_blocks, block_size, kv_num_heads, head_size)` | `T` | Updated **in-place**. |
| 5 | `cumulative_sequence_length` | Input | `(batch_size + 1)` | `S=int32` | Prefix-sum of Q lengths across batch. |
| 6 | `past_seqlens` | Input | `(batch_size)` | `S=int32` | Per-request length of cached tokens. |
| 7 | `block_table` | Input | `(batch_size, max_blocks_per_seq)` | `S=int32` | Row-per-request, block indices into `key_cache`/`value_cache`. |
| 8 | `cos_cache` | Input (opt) | `(max_total_seqlen, head_size/2)` | `T` | Required when `do_rotary=1`. |
| 9 | `sin_cache` | Input (opt) | `(max_total_seqlen, head_size/2)` | `T` | Required when `do_rotary=1`. |
| 0 | `output` | Output | `(num_tokens, hidden_size)` | `T` | |
| 1 | `key_cache_out` | Output (opt) | same as `key_cache` | `T` | **Must alias** `key_cache`. |
| 2 | `value_cache_out` | Output (opt) | same as `value_cache` | `T` | **Must alias** `value_cache`. |

**Attributes** (all `INT` unless noted): `num_heads`, `kv_num_heads`, `scale`
(`FLOAT`, opt), `softcap` (`FLOAT`, opt), `local_window_size` (default `-1`),
`do_rotary` (default `0`), `rotary_interleaved` (default `0`).

### 3.2 Forward-looking additions (#29912, for later phases)

Extra optional inputs, indices 10–16: `slot_mapping`, `head_sink`,
`q_norm_weight`, `k_norm_weight`, `k_scale`, `v_scale`, `attention_metadata`.
Extra attributes: `k_cache_dtype`, `v_cache_dtype`, `k_quant_type`,
`v_quant_type`, `kv_cache_layout` (`SEPARATE`/`LATENT`), `v_head_size`,
`rotary_offset`, `qk_norm_epsilon`, `use_smooth_softmax`. New type constraints
`T_CACHE ∈ {fp16, bf16, int8, fp8e4m3fn}`, `T_KV_SCALE ∈ {fp32}`.

v1 registers only against the current schema. Phase 2+ expands as those inputs
land.

### 3.3 GenAI runtime contract

The paged decoder step provides:

- `input_ids: int32[num_tokens]` — packed across all in-flight requests.
- `block_table: int32[batch_size, max_blocks_per_seq]` — one row per request.
- `cumulative_sequence_lengths: int32[batch_size + 1]` — prefix sum.
- `past_sequence_lengths: int32[batch_size]` — cached-token count per request.
- Per-layer `key_cache` and `value_cache` shared across the whole engine.
- (Phase 2) `attention_metadata: int32[2]` on CPU = `[max_query_len_bound, max_kv_len_bound]`, produced by the engine each step.

---

## 4. Reuse strategy

The core WebGPU FlashAttention kernel is doing the attention math in **every**
paged path. We never rewrite the online softmax, the tiling, the split-K
reduce, or the head_sink apply. What we add is a **paged-aware addressing
skin** around it.

### 4.1 Decode path (max_query_len == 1) — full reuse

The existing WebGPU flash-decode split-K kernels
([`ComputeFlashAttentionDecodeQKV` + `ComputeFlashAttentionDecodeVxReduce`](../../onnxruntime/contrib_ops/webgpu/bert/flash_attention.cc)),
including the reduce shader, indirect dispatch plumbing, and per-vendor
cache-hint variants, are mathematically identical to what paged decode needs.
The only site that changes is the K/V load:

```wgsl
// Existing (dense present):
let k = present_key[((batch * num_heads + head) * total_seq_len + t) * head_size + h];

// Paged (selected by cache-hint):
let block_row = block_table[batch * max_blocks_per_seq + (t / block_size)];
let in_block  = t % block_size;
let k = key_cache[((block_row * block_size + in_block) * kv_num_heads + head) * head_size + h];
```

Per-batch `total_kv_len` — GQA already reads this per-batch from `seqlens_k`
when present ([`use_seqlen_k` branch](../../onnxruntime/contrib_ops/webgpu/bert/flash_attention.cc)).
PA gets it from `past_seqlens[b] + 1` (decode) or `past_seqlens[b] +
(cumulative_seqlens_q[b+1] - cumulative_seqlens_q[b])` (prefill). We bind
`past_seqlens` in the same slot the shader expects for the per-batch length
input and the code path is reused unchanged.

### 4.2 Prefill path (max_query_len > 1) — two levels

- **Option A — Gather-then-flash (v1).** Materialize a dense per-batch K/V
  from the paged cache into `[B, kv_num_heads, max_kv_len, head_size]`
  scratch, un-pack Q into `[B, num_heads, max_q_len, head_size]` (padded),
  call `ApplyFlashAttention` verbatim. Two scratch allocs per layer per step;
  throwaway perf but zero shader rewrite. **Recommended for v1.**
- **Option B — Paged prefill (Phase 2).** Take the FlashAttention prefill
  shader, apply the same paged-K/V-load parameterization we did for decode,
  teach it to slice Q from packed `[num_tokens, num_heads, head_size]` using
  `cumulative_seqlens_q`. Reuses the online-softmax math and tiling; rewrites
  the K/V load site and Q addressing. About half the shader is reused.

### 4.3 What can't be absorbed by the FlashAttention kernel

- **`ReshapeAndCache`** (K/V writer into the paged cache). New program, ~40
  lines of WGSL. Uses `slot_mapping` when present else derives the slot from
  `past_seqlens + block_table`.
- **RoPE + QK-Norm prologue.** GQA already has
  [`SplitPackedQKVWithRotaryEmbeddingAndCopyKVProgram`](../../onnxruntime/contrib_ops/webgpu/bert/flash_attention.cc);
  the RoPE math and shape are the same, only the position lookup differs
  (`past_seqlens[batch] + token_offset` instead of `seqlens_k[batch] - 1`).
  Reuse with a cache-hint variant.
- **Per-batch varlen bookkeeping** — binary search over `cumulative_seqlens_q`
  to map a token index to a batch index. Shared WGSL helper reused across all
  three paged programs (rotary prologue, `ReshapeAndCache`, paged flash
  prefill).

---

## 5. Phased delivery plan

### Phase 0 — Skeleton (this PR)

- Add `contrib_ops/webgpu/bert/paged_attention.{h,cc}` with the kernel class
  and `ComputeInternal` returning `NOT_IMPLEMENTED`. Same shape as the CPU
  stub in #29867. Purpose: register the op with the WebGPU EP so a model
  containing `PagedAttention` no longer fails at kernel-matching time, and
  reserve the file for the real implementation.
- Register in `webgpu_contrib_kernels.cc` for `MLFloat16`.
- Reuse the CUDA `paged_attention_helper.h` for input validation (once we
  wire it in). The helper is pure host code; no CUDA deps.

**Ships:** a kernel that says "not implemented" gracefully rather than
"missing." Unblocks the file layout for Phase 1.

### Phase 1 — Functional decode + gather-then-flash prefill (unquantized, SEPARATE)

1. `PagedAttentionValidateInputs` — reuse the CUDA helper. Refactor its
   location to `contrib_ops/cpu/bert/paged_attention_helper.h` (or a similar
   provider-neutral spot) so CPU/CUDA/WebGPU share one copy. Zero CUDA deps
   already — no code change needed to the helper itself.

2. `SplitPackedQKVWithRotaryProgram` (paged variant) — only runs when
   `is_packed_qkv || do_rotary`. Uses `past_seqlens + cumulative_seqlens_q`
   for the position lookup.

3. `ReshapeAndCacheProgram` — writes K/V into paged cache. Standalone (not
   fused with rotary) in v1 so we're ready for `slot_mapping = -1` later.
   Dispatch shape `[total_tokens * kv_hidden_size / vec]`, workgroup 256.

4. **Decode** (`max_query_len == 1`): parameterize the existing
   `flash_attention_decode_qkv.wgsl.template` and
   `flash_attention_decode_vx_reduce.wgsl.template` on a `paged` cache-hint
   bool. Thread `block_table`/`block_size`/`max_num_blocks_per_seq` through
   the uniform block. Bind `past_seqlens` where the shader expects `seqlens_k`.

5. **Prefill** (`max_query_len > 1`): `GatherAndExpandPagedKVCache` — new
   program that scatters paged K/V into a dense `[B, kv_num_heads, max_kv_len,
   head_size]` scratch. `UnpackVarlenQuery` — pads Q from
   `[num_tokens, num_heads, head_size]` to `[B, num_heads, max_q_len,
   head_size]`. Then call `ApplyFlashAttention` unchanged.

6. **Cache-output aliasing.** Emit `key_cache_out`/`value_cache_out` and
   verify `MutableData<T>() == input->Data<T>()`. Fail INVALID_ARGUMENT
   otherwise, exactly as CUDA does. WebGPU EP allocator reuse should make
   this straightforward.

7. **Empty-input fast path.** `parameters.token_count == 0` returns OK with a
   zero-sized output. Exercised by GenAI engine on graph-capture warmups.

**Ships:** a functional PagedAttention on WebGPU sufficient for GenAI
continuous-batching decode + prefill, MLFloat16, unquantized, SEPARATE
layout, with GenAI's builder gate flipped to allow `-e webgpu`.

### Phase 2 — Perf and forward-looking schema

- **Paged prefill kernel.** Fused prefill that indexes the paged cache
  directly, eliminating the gather-then-flash scratch alloc.
- **Fused split-packed-QKV + rotary + reshape-and-cache**, mirroring
  `SplitPackedQKVWithRotaryEmbeddingAndCopyKVProgram`. Saves one full-tokens
  read of Q/K/V.
- **`attention_metadata` consumption.** Read the CPU-side `int32[2]` when
  the model provides it (input 16 under #29912's schema); use it for
  workspace sizing and to skip D→H sync in decode. Enables WebGPU graph
  capture with PagedAttention.
- **Indirect dispatch for decode**, mirroring GQA's graph-capture-friendly
  path.
- **`head_sink` + `use_smooth_softmax`**. Small change: extra add of
  `exp(sink_logit[h])` in the softmax denominator. The existing decode-reduce
  shader already accepts a `head_sink` tensor in the GQA path — reuse.
- **`q_norm_weight` + `k_norm_weight` (QK-Norm).** Same fused prologue GQA
  already implements.

### Phase 3 — Quantized KV cache (`T_CACHE = int8`)

- Add the `T` × `T_CACHE` template axis to the kernel registration.
- Simplest first cut: dequant-in-place inside the K/V load in decode+prefill
  shaders using a `fp32` scale bound as a uniform (per-tensor) or as a
  `[kv_num_heads, 1, head_size]` buffer (per-channel).
- No fp8 on WebGPU (no shader type). No int4 either — schema `T_CACHE` doesn't
  include sub-byte types.
- No XQA analog on WebGPU (XQA is TensorRT-LLM cutlass, sm80+).

### Phase 4 — MLA / LATENT layout

Deferred until a customer needs DeepSeek-V3-class models on WebGPU. Requires
wider head_size (576 for DSV3), which most WebGPU adapters can't fit in
shared memory. Also reworks the split-K decode kernel's cache indexing
(V is a slice of the K row).

### Phase 5 — GenAI engine WebGPU graph capture

Not an ORT change — an ORT-GenAI change. Mirror the pattern in ORT-GenAI PR
#2333 §3 (persistent oversized buffers, static device block table, shape
bucketing) with `wgpuGraph` in place of `cudaGraph`. Prerequisite: Phase 2's
`attention_metadata` consumption on the ORT side.

---

## 6. File layout

```
onnxruntime/contrib_ops/webgpu/bert/
  paged_attention.h                                     # kernel class decl (Phase 0)
  paged_attention.cc                                    # kernel class impl + dispatch (Phase 0 stub, filled Phase 1)
  paged_attention_common.h                              # WebgpuAttentionParameters overload + program uniforms (Phase 1)
  paged_reshape_and_cache.wgsl.template                 # K/V paged writer (Phase 1)
  paged_flash_attention_decode.wgsl.template            # paged variant of flash_attention_decode_qkv (Phase 1)
  paged_flash_attention_decode_reduce.wgsl.template     # paged variant of the reduce (Phase 1)
  paged_gather_kv.wgsl.template                         # gather-then-flash prefill scratch (Phase 1)
  paged_flash_attention_prefill.wgsl.template           # fused paged prefill (Phase 2)
```

Shared helper (already exists on CUDA, refactor location in Phase 1):

```
onnxruntime/contrib_ops/cpu/bert/paged_attention_helper.h   # provider-neutral shape checks
```

---

## 7. Dispatch cascade (Phase 1)

```
ComputeInternal:
    ValidateInputs (shared helper)
    if token_count == 0:
        emit zero-sized output; return OK
    if is_packed_qkv or do_rotary:
        RunSplitPackedQKVAndRotary()
    RunReshapeAndCache()
    if max_query_len == 1:
        RunPagedFlashDecodeQKV()
        RunPagedFlashDecodeReduce()
    else:
        RunGatherAndExpandPagedKVCache()   # -> [B, kv_num_heads, max_kv_len, head_size]
        RunUnpackVarlenQuery()             # -> [B, num_heads, max_q_len, head_size]
        ApplyFlashAttention(...)
    return OK
```

`max_query_len` — v1 derives it from `cumulative_seqlens_q` (D→H sync once
per node); Phase 2 uses `attention_metadata` when available.

Feature guards (v1 rejects with `NOT_IMPLEMENTED` and a specific message):

- Any `T_CACHE != T` (quantized).
- `kv_cache_layout == LATENT`.
- Non-null `head_sink`, `q_norm_weight`, `k_norm_weight`, `k_scale`, `v_scale`.
- `slot_mapping` containing negative entries.

---

## 8. WebGPU-specific pitfalls to plan around

| Pitfall | Mitigation |
|---|---|
| WGSL has no dynamic 4-D array indexing into storage buffers. | Linearize the cache addressing in the shader: `((block_row * block_size + in_block) * kv_num_heads + head) * head_size + c`. Pass `block_size`, `kv_num_heads`, `head_size`, `max_num_blocks_per_seq` as uniforms. |
| Storage buffer binding max is adapter-dependent (128 MiB on some). | Paged representation naturally partitions the cache into per-layer bindings; never bind the full pool. |
| WebGPU graph capture forbids host-visible reads mid-graph. | Consume `attention_metadata` (Phase 2) instead of D→H syncing `cumulative_seqlens_q`. |
| Subgroup width varies (16 Intel, 32 NV/AMD, 64 Qualcomm/Apple). | Copy the `is_qualcomm`/`is_nvidia`/`is_apple`/`has_subgroups` cache-hint knobs from `FlashAttentionProgram`. |

---

## 9. Testing plan

- **Correctness (host-side enforce paths, lavapipe-compatible)**: input
  validation tests. See the [`webgpu-local-testing`](../../.agents/skills/webgpu-local-testing/SKILL.md)
  skill for lavapipe details.
- **Numerical correctness against GQA**: port [`test_paged_attention_cuda.py`](../../onnxruntime/test/python/transformers/test_paged_attention_cuda.py)
  into `test_paged_attention_webgpu.py`; parameterize the EP. Cross-check
  output against the WebGPU GQA op run on the same K/V materialized into a
  dense past_key/past_value. Because lavapipe crashes on MatMul, the
  numerical tests must run on **macOS-arm64 Metal** as the source of truth
  (same policy as the expanded-Attention tests).
- **GenAI E2E**: run Phi-3-mini or Llama-3.2-1B through the GenAI continuous
  batching engine on WebGPU once GenAI PR #2330's `-e webgpu` gate is
  flipped.

---

## 10. Open items (do not block v1)

1. **Block-size waiver for non-CUDA EPs.** Are we allowed to relax the model-side `paged_block_size % 256 == 0` constraint in the GenAI builder for WebGPU? Decision: **keep the CUDA constraint in v1**; revisit if perf data shows it matters.
2. **`slot_mapping = -1` semantics.** GenAI doesn't emit `-1` today; v1 accepts the input and uses it as an override but doesn't implement the "skip on negative" branch. One-line follow-up when a customer needs speculative decoding on WebGPU.
3. **`bf16` support on WebGPU.** Adapter-dependent, gated by Dawn feature flag. v1 registers only `MLFloat16`. Add `BFloat16` when Dawn ships stable bf16 on the target adapters.
