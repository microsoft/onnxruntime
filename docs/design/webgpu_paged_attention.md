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
- **[microsoft/onnxruntime#29912](https://github.com/microsoft/onnxruntime/pull/29912)** (merged) — schema extension: `T_CACHE`/`T_KV_SCALE` type constraints; new optional inputs `slot_mapping`, `head_sink`, `q_norm_weight`, `k_norm_weight`, `k_scale`, `v_scale`, `attention_metadata`; new attributes `k_cache_dtype`, `v_cache_dtype`, `k_quant_type`, `v_quant_type`, `kv_cache_layout` (`SEPARATE`/`LATENT`), `v_head_size`, `rotary_offset`, `qk_norm_epsilon`, `use_smooth_softmax`. Also adds portable split-KV decode + XQA quantized decode kernels on CUDA.
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
| Schema baseline | Build v1 against the **merged expanded schema** (inputs 0-16). WebGPU v1 implements the pre-existing subset and rejects unsupported new inputs/attrs with explicit `NOT_IMPLEMENTED` errors. |
| `slot_mapping = -1` | v1 accepts `slot_mapping` when present and uses it as an authoritative override of the derived slot, but leaves the "skip write on negative" branch out. GenAI does not emit `< 0` today. Adding it later is a one-line shader change. |
| `softcap != 0` | v1 rejects with `ORT_NOT_IMPLEMENTED`. FlashAttention has no softcap today; adding it is a Phase 2 change. |
| `local_window_size != -1` | v1 rejects with `ORT_NOT_IMPLEMENTED`. Sliding-window attention lands in Phase 2 (port from GQA). |
| `T = bfloat16` | v1 rejects (registers `MLFloat16` only). FA has no `bf16` path yet either; both add together in Phase 2 when Dawn's `bf16` support on target adapters stabilizes. |

---

## 3. Op contract

### 3.1 Baseline v1 contract used by WebGPU

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
| 1 | `key_cache_out` | Output (opt) | same as `key_cache` | `T` | Must be paired with `value_cache_out`; may alias `key_cache`. |
| 2 | `value_cache_out` | Output (opt) | same as `value_cache` | `T` | Must be paired with `key_cache_out`; may alias `value_cache`. |

**WebGPU implementation limitation:** `key_cache_out` and `value_cache_out` must
either both be present or both be omitted. When omitted, the input KV-cache
buffers are updated in place. When present but not aliased with the input
buffers, WebGPU copies the input caches to the output buffers before scattering
new tokens; this preserves correctness but adds per-run device-copy overhead.
Production callers should use IO-binding to alias each cache input with its
corresponding output.

**Attributes** (all `INT` unless noted): `num_heads`, `kv_num_heads`, `scale`
(`FLOAT`, opt), `softcap` (`FLOAT`, opt), `local_window_size` (default `-1`),
`do_rotary` (default `0`), `rotary_interleaved` (default `0`).

### 3.2 Expanded merged schema surface (#29912)

Extra optional inputs, indices 10–16: `slot_mapping`, `head_sink`,
`q_norm_weight`, `k_norm_weight`, `k_scale`, `v_scale`, `attention_metadata`.
Extra attributes: `k_cache_dtype`, `v_cache_dtype`, `k_quant_type`,
`v_quant_type`, `kv_cache_layout` (`SEPARATE`/`LATENT`), `v_head_size`,
`rotary_offset`, `qk_norm_epsilon`, `use_smooth_softmax`. New type constraints
`T_CACHE ∈ {fp16, bf16, int8, fp8e4m3fn}`, `T_KV_SCALE ∈ {fp32}`.

v1 is schema-compatible with this merged surface, but only a subset is
implemented. Unsupported new inputs/attributes are validated and rejected with
clear `NOT_IMPLEMENTED` errors.

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

v1 unifies decode and prefill into a **single gather-then-flash** code path
that reuses [`ApplyFlashAttention`](../../onnxruntime/contrib_ops/webgpu/bert/flash_attention.cc)
verbatim. No new attention math is written. The paged specifics are contained
in shape-transform kernels around the FA call.

### 4.1 What FA already provides

`ApplyFlashAttention` gives us everything the attention math needs:

- **Per-batch causal mask** via the optional `seqlen_k` input
  (`seq_causal_length = past_sequence_length + q_idx_global + 1`), which
  handles the per-token cutoff prefill needs and degenerates to the decode
  case when `seqlens_q[b] = 1`.
- **Automatic decode-vs-prefill tier dispatch** based on Q sequence length
  (`use_split_reduce = sequence_length < 32`) — decode picks the split-K
  decode + reduce kernels; prefill picks the flash-prefill kernel. Both
  tiers are reused as-is.
- **GQA head grouping** via `n_reps = num_heads / kv_num_heads`.
- **Custom `scale`**.

FA does **not** apply rotary, does **not** write present K/V (it only reads
them), does **not** support softcap, and does **not** yet support `bf16` —
all of which line up with our v1 constraints and pre-passes.

### 4.2 Paged-aware skin around FA (v1)

What we add for v1 is a paged-aware skin consisting of the Phase 1b passes
(already done) plus three new shape-transform programs:

| Existing (Phase 1b, packed varlen throughout) | Purpose |
|---|---|
| `PagedAttentionSplitPackedQKVProgram` | Split packed QKV → separate Q, K, V (varlen). |
| `PagedAttentionRotaryProgram` | RoPE on packed varlen Q or K. |
| `ScatterKVToPagedCacheProgram` | Write K, V into the paged cache. |

| New (Phase 1, this PR) | Purpose |
|---|---|
| `PagedAttentionGatherKVProgram` | Un-page `key_cache` / `value_cache` through `block_table` into padded contiguous scratch tensors `(B, kv_num_heads, max_kv_len, head_size)`. |
| `PagedAttentionUnpackQueryProgram` | Expand packed varlen Q from `(token_count, num_heads * head_size)` to padded BSNH `(B, max_seqlen_q, num_heads, head_size)` using `cumulative_sequence_length`. Padding slots are zero-filled; their outputs are dropped in the repack. |
| `PagedAttentionRepackOutputProgram` | Inverse of unpack: gather valid slots of the padded FA output back to `(token_count, hidden_size)`. |

Plus a tiny computation of `seqlen_k[b] = past_seqlens[b] + seqlens_q[b] - 1`
(FA's last-valid-index convention) to drive FA's per-batch causal mask.

The scatter kernel updates the paged cache first. FA then receives gathered
K/V scratch through `past_key` / `past_value` and `nullptr` for the present
K/V parameters, so it reads the contiguous scratch and does not touch the
paged cache.

### 4.3 Cost of the fallback path

Two full paged K/V reads plus two contiguous K/V writes per layer per step
(gather), Q/output shuffles proportional to `token_count`, and a scratch
allocation of `2 * B * kv_num_heads * max_kv_len * head_size + B *
max_seqlen_q * hidden_size` bytes. Fusing all of these into a paged-aware FA
kernel is the Phase 2 optimization — see §5 Phase 2.

### 4.4 Host-visible values and graph capture (deferred to Phase 2)

The v1 op performs **one blocking D→H metadata download per node per Run**.
It packs `cumulative_seqlens_q` and `past_seqlens` into a small GPU buffer,
then reads it on the CPU to build `seqlen_k_cpu` and compute
`max_seqlen_q` / `max_kv_len`. Those two scalars drive:

- **Dispatch dims** of `PagedAttentionGatherKVProgram`,
  `PagedAttentionUnpackQueryProgram`, `FlashAttentionProgram` /
  `FlashAttentionDecodeQKVProgram`, and `PagedAttentionRepackOutputProgram`.
- **Scratch tensor sizes** for `k_padded`, `v_padded`, `q_padded`, and
  `output_padded`.

The download ends the current compute pass, flushes the queue, allocates a
staging buffer, and waits for the result. It is therefore a v1 latency
limitation and unsuitable for browser-main-thread decode at many transformer
layers, not only a graph-capture limitation.

The host-derived values are captured as literals when a WebGPU graph is recorded, so any
subsequent step that presents different per-batch lengths would replay with
wrong grids and undersized scratch. This is the exact same class of blocker
that keeps the CUDA PagedAttention op out of CUDA Graphs — see the
`cudaMemcpyAsync(cumulative_seqlens_q → host)` + `cudaStreamSynchronize`
pair in [`onnxruntime/contrib_ops/cuda/bert/paged_attention.cc`][cuda-pa-sync]
that computes `data.max_query_len` from a D→H sync.

GQA/FA-decode escape the blocker via `use_indirect_dispatch` +
`PrepareIndirectDispatchProgram`, but they only had **one** host-visible
scalar to hide (`total_sequence_length`) and got static scratch for free
from `past_present_share_buffer=true`. Paged has four (`q_len_b`,
`total_kv_b`, `max_seqlen_q`, `max_kv_len`) and no free scratch — the
lift-and-shift plan is spelled out under §5 Phase 2 "Graph-capture support".

[cuda-pa-sync]: ../../onnxruntime/contrib_ops/cuda/bert/paged_attention.cc

---

## 5. Phased delivery plan

### Phase 0 — Skeleton (early commits in this PR)

- Add `contrib_ops/webgpu/bert/paged_attention.{h,cc}` with the kernel class
  and `ComputeInternal` returning `NOT_IMPLEMENTED`. Same shape as the CPU
  stub in #29867. Purpose: register the op with the WebGPU EP so a model
  containing `PagedAttention` no longer fails at kernel-matching time, and
  reserve the file for the real implementation.
- Register in `webgpu_contrib_kernels.cc` for `MLFloat16`.
- Reuse the CUDA `paged_attention_helper.h` for input validation (once we
  wire it in). The helper is pure host code; no CUDA deps.

**Ships:** a kernel that says "not implemented" gracefully rather than
"missing." Unblocks the file layout for Phase 1. Included in this PR only as
the first two commits of the branch history; the final state delivered by
this PR is Phase 1 below.

### Phase 1 — Functional decode + gather-then-flash prefill (unquantized, SEPARATE) — this PR

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

6. **Cache-output handling.** `key_cache_out`/`value_cache_out` are optional,
  but must be emitted as a pair when present. Omitted outputs update the input
  caches in place. Non-aliased output buffers are supported through a device
  copy, although IO-binding aliasing is preferred to avoid that overhead.

7. **Empty-input fast path.** `parameters.token_count == 0` returns OK with a
   zero-sized output. Exercised by GenAI engine on graph-capture warmups.

**Ships:** a functional PagedAttention on WebGPU sufficient for GenAI
continuous-batching decode + prefill, MLFloat16, unquantized, SEPARATE
layout, with GenAI's builder gate flipped to allow `-e webgpu`.

### Phase 2 — Perf and forward-looking schema

The following five items define the Phase 2 implementation scope. Packed-QKV
support itself is already functional in Phase 1; the fourth item is an
optimization of that existing path.

1. **Validate and tune direct paged decode.** The current implementation uses
  the direct paged split-reduce kernels when `max_seqlen_q < 32` and retains
  gather-then-flash for longer requests. Benchmark both paths at the boundary
  (`1`, `4`, `16`, `31`, `32`, `64`) across MHA, GQA, batch sizes, past lengths,
  packed QKV, rotary, and cache aliasing. Keep correctness tests for every
  route and replace the cutoff with a measured or explicitly documented
  policy if adapter-specific results require it. Use
  `test/python/transformers/benchmark_paged_attention_webgpu.py` for both
  `decode` (one new token per request) and `prompt` (multiple new tokens per
  request) so the same matrix remains valid when fused prefill lands. Direct
  paged decode is the production route for short Q sequences; prompt/prefill
  remains on the gathered fallback until a dedicated paged prefill kernel is
  validated.

2. **Complete deferred Phase 1 feature support.** Add and test the features
  currently rejected by WebGPU: `softcap`, `local_window_size`, `head_sink`,
  `use_smooth_softmax`, `q_norm_weight`, and `k_norm_weight`. Evaluate
  `slot_mapping` including negative-slot skip-write semantics, plus
  `rotary_offset` and non-default `v_head_size` when model compatibility
  requires them. Add `bfloat16` only when target WebGPU adapters provide a
  reliable storage and shader path. Each feature should have a support test
  and an explicit unsupported-path test until implemented.

3. **Implement fused paged prefill.** Replace the prefill gather-then-flash
  sequence with a dedicated `FlashAttentionPagedPrefillProgram` that indexes
  `key_cache` and `value_cache` directly through `block_table`. It should
  retain the single-kernel prefill algorithm and workgroup Q tiling of
  `FlashAttentionProgram`; only the K/V tile loads become page-table-aware.
  The first implementation supports fp16, no attention bias, no head sink,
  no TurboQuant, and no graph capture. It must preserve variable-Q-length
  causal masking via `seqlen_k` and `seqlens_q`. The goal is to eliminate the
  dense K/V scratch allocation and gather bandwidth. Compare latency, peak
  scratch memory, and output parity against the current fallback for MHA,
  GQA, mixed batch lengths, and rotary inputs.

4. **Fuse packed-QKV preprocessing.** Combine the existing split-packed-QKV,
  rotary, reshape, and cache-scatter stages, mirroring
  `SplitPackedQKVWithRotaryEmbeddingAndCopyKVProgram`. This should remove
  intermediate Q/K/V tensors and avoid an extra full-token read/write cycle.
  Preserve the Phase 1 packed-QKV behavior and add parity tests for packed
  non-rotary, packed rotary, interleaved rotary, MHA, and GQA cases.

5. **Make PagedAttention graph-capture-safe.** Consume
  `attention_metadata: int32[2]` (input 16 under the merged schema) as
  `[max_query_len_bound, max_kv_len_bound]`, so scratch buffers can be sized
  once from stable bounds. Move `seqlen_k` and per-batch Q-length derivation
  to the GPU, then write indirect dispatch dimensions for
  `PagedAttentionGatherKVProgram`, `PagedAttentionUnpackQueryProgram`, the
  FlashAttention prefill/decode programs, and
  `PagedAttentionRepackOutputProgram`. This removes the current GPU-to-CPU
  metadata copy and per-step shape-dependent allocation, the two blockers to
  graph capture. The GenAI integration belongs in Phase 5.

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
  paged_attention.h                                      # kernel and program declarations
  paged_attention.cc                                     # host dispatch and validation
  paged_attention_pack_metadata.wgsl.template            # pack metadata for one D→H readback
  paged_attention_split_packed_qkv.wgsl.template         # split packed QKV input
  paged_attention_rotary.wgsl.template                  # rotary embedding for Q or K
  paged_attention_scatter_kv.wgsl.template               # scatter K/V into paged cache
  paged_attention_gather_kv.wgsl.template                # gather paged K/V into padded scratch
  paged_attention_unpack_query.wgsl.template             # unpack packed Q into LEFT-aligned BSNH
  paged_attention_repack_output.wgsl.template            # repack padded output to packed output
```

Shared helper (already exists on CUDA, refactor location in Phase 1):

```
onnxruntime/contrib_ops/cpu/bert/paged_attention_helper.h   # provider-neutral shape checks
```

Benchmark harness:

```
onnxruntime/test/python/transformers/benchmark_paged_attention_webgpu.py
```

The harness constructs each session before timing and reports end-to-end
latency with output synchronization. It intentionally exposes MHA and GQA as
separate cases: MHA uses one KV head per query head, while GQA shares fewer KV
heads across groups of query heads. Both `decode` and `prompt` modes use the
same cache and packed-sequence layout, allowing the fallback and future fused
prefill paths to be compared directly.

---

## 7. Dispatch cascade (Phase 1)

```
ComputeInternal:
  ValidateInputs (shared helper)
  copy non-aliased cache inputs to cache outputs
  if token_count == 0:
    return OK
  if is_packed_qkv:
    RunSplitPackedQKV()
  read and validate cumulative_sequence_length / past_seqlens once
  if max_seqlen_q == 0:
    fill output with zeros; return OK
  if do_rotary:
    RunRotaryEmbedding() for Q and K
  RunScatterKVToPagedCache()
  RunGatherKV()          # -> [B, kv_num_heads, max_kv_len, head_size]
  RunUnpackQuery()       # -> [B, max_seqlen_q, num_heads, head_size]
  ApplyFlashAttention()  # decode and prefill tiers selected internally
  RunRepackOutput()      # -> [token_count, hidden_size]
    return OK
```

`max_seqlen_q` and `max_kv_len` are derived from one packed metadata D→H
readback per node. `ApplyFlashAttention` selects its decode or prefill tier
internally based on the padded Q sequence length. Phase 2 uses
`attention_metadata` and GPU-side metadata preparation to remove this host
readback and make the path graph-capture-safe.

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
| Storage buffer binding max is adapter-dependent (128 MiB on some). | The paged cache is addressed through per-layer bindings, and the gather-then-flash scratch buffers are checked against `maxStorageBufferBindingSize` before allocation. |
| WebGPU graph capture forbids host-visible reads mid-graph. | Consume `attention_metadata` (Phase 2) instead of D→H syncing `cumulative_seqlens_q`. |
| Subgroup width varies (16 Intel, 32 NV/AMD, 64 Qualcomm/Apple). | Copy the `is_qualcomm`/`is_nvidia`/`is_apple`/`has_subgroups` cache-hint knobs from `FlashAttentionProgram`. |

---

## 9. Testing plan

- **Correctness (host-side enforce paths, lavapipe-compatible)**: input
  validation tests. See the [`webgpu-local-testing`](../../.agents/skills/webgpu-local-testing/SKILL.md)
  skill for lavapipe details.
- **Numerical correctness against GQA**: [`test_paged_attention.py`](../../onnxruntime/test/python/transformers/test_paged_attention.py)
  is EP-parametrized on `Config.ep`. The CUDA classes
  (`TestPagedAttention`, `TestPagedAttentionMEA`,
  `TestPagedAttentionRotaryZeroTokenRegression`) remain the CUDA source of
  truth. `TestPagedAttentionWebGpu` runs the same PyTorch reference
  (`attention_ref`) over a WebGPU-scoped config matrix (rotary + packed QKV +
  GQA), filtered by `_webgpu_supports_config` to skip `softcap != 0` and
  `local_window_size != -1` until the WebGPU kernel implements them. Because
  lavapipe crashes on MatMul, the numerical tests must run on
  **macOS-arm64 Metal** or on a discrete Windows/Linux WebGPU adapter as the
  source of truth (same policy as the expanded-Attention tests).
- **GenAI E2E**: run Phi-3-mini or Llama-3.2-1B through the GenAI continuous
  batching engine on WebGPU once GenAI PR #2330's `-e webgpu` gate is
  flipped.

---

## 10. Open items (do not block v1)

1. **Block-size waiver for non-CUDA EPs.** Are we allowed to relax the model-side `paged_block_size % 256 == 0` constraint in the GenAI builder for WebGPU? Decision: **keep the CUDA constraint in v1**; revisit if perf data shows it matters.
2. **`slot_mapping = -1` semantics.** GenAI doesn't emit `-1` today; v1 accepts the input and uses it as an override but doesn't implement the "skip on negative" branch. One-line follow-up when a customer needs speculative decoding on WebGPU.
3. **`bf16` support on WebGPU.** Adapter-dependent, gated by Dawn feature flag. v1 registers only `MLFloat16`. Add `BFloat16` when Dawn ships stable bf16 on the target adapters.
