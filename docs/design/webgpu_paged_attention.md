# Design: WebGPU PagedAttention

**Status**: v1 landed. Phase 2 partially landed (direct paged decode + fused paged prefill + Unpack/Repack skip fast paths).
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
- **[microsoft/onnxruntime#31727](https://github.com/microsoft/onnxruntime/pull/31727)** — WebGPU Phase 2 (partial): direct paged split-reduce decode, fused paged prefill, and Unpack/Repack skip fast paths for uniform-batch and packed-varlen callers.
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
| `block_size` | Any power-of-two >= 16 (enforced by `paged_attention_helper`). GenAI currently emits `block_size % 256 == 0`; the shared fused paged-prefill selection predicate additionally requires `block_size >= max_k_step` (16 or 32 for fp16). Configs that fail the alignment fall back to gather-then-flash. |
| Precision | `MLFloat16` only. Registered as a single typed kernel. |
| Schema baseline | Build v1 against the **merged expanded schema** (inputs 0-16). WebGPU v1 implements the pre-existing subset and rejects unsupported new inputs/attrs with explicit `NOT_IMPLEMENTED` errors. |
| `slot_mapping` | v1 rejects any non-null `slot_mapping` input with `ORT_NOT_IMPLEMENTED`. GenAI does not emit this input today. Adding it (and the negative-slot skip-write semantics) is Phase 2 work. |
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

v1 shipped a gather-then-flash fallback so `ApplyFlashAttention` could be
reused verbatim. It composes the Phase 1b passes (already done) with three
new shape-transform programs:

| Existing (Phase 1b, packed varlen throughout) | Purpose |
|---|---|
| `PagedAttentionSplitPackedQKVProgram` | Split packed QKV → separate Q, K, V (varlen). |
| `PagedAttentionRotaryProgram` | RoPE on packed varlen Q or K. |
| `ScatterKVToPagedCacheProgram` | Write K, V into the paged cache. |

| New (Phase 1) | Purpose |
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

### 4.3 Direct paged paths (Phase 2, this PR)

Phase 2 keeps the Phase-1b preprocessing untouched but replaces the gather
step with two paged-aware FA programs that address `key_cache` and
`value_cache` directly through `block_table`. This eliminates the dense K/V
scratch and its bandwidth. It also collapses the padded-BSNH Q/output round
trip on the common uniform-batch and packed-varlen cases (see
"Unpack/Repack skip" below).

**Two direct paths**, selected by `max_seqlen_q` (mirrors the dense-FA
split-reduce threshold):

- **Direct paged decode** (`max_seqlen_q < 32`): dispatch
  `FlashAttentionPagedDecodeQKV` + `FlashAttentionPagedDecodeVxReduce` on the
  paged cache. No adapter gating — the split-reduce path already sizes its
  workgroup budget for every WebGPU adapter class.
- **Fused paged prefill** (`max_seqlen_q >= 32`): dispatch
  `FlashAttentionPagedPrefillProgram`. Straight port of the dense FA
  prefill shader's shared-memory algorithm. The shader uses no subgroup
  intrinsics, so it runs on every WebGPU adapter that satisfies the
  dtype / shm-budget / block-alignment gates — no adapter-class fallback.
  (A paged subgroup-shuffle variant could be added later if benchmarks on
  subgroup adapters justify it.)

**Selection helper.** All fast-path decisions in `PagedAttention` and
`ApplyFlashAttention` consult one shared predicate:

```cpp
bool ShouldRunFusedPagedPrefill(context, is_fp16, max_seqlen_q,
                                head_size, block_size);
```

It rejects when: `!is_fp16`,
`max_seqlen_q < 32`, the workgroup shared-memory budget can't fit a K/V tile
(fp16: `head_size > 256`), or `block_size < max_k_step` (the tile-per-lookup
invariant, see §8 pitfalls). Because the same predicate gates the
"skip `RunGatherKV`", "skip `q_padded` scratch", and "select fused shader"
decisions, the three cannot drift.

The exact three call sites the anti-drift invariant covers are:

1. `PagedAttention::ComputeInternal` — `use_direct_paged_prefill` (drives
   whether `RunGatherKV` runs and whether `k_padded`/`v_padded` are allocated);
   see [`onnxruntime/contrib_ops/webgpu/bert/paged_attention.cc`](../../onnxruntime/contrib_ops/webgpu/bert/paged_attention.cc).
2. `PagedAttention::ComputeInternal` — `varlen_mode` inside the
   `skip_unpack_repack` decision (drives whether the Q buffer is handed to FA
   as a rank-4 `[token_count, 1, N, H]` view or a padded BSNH scratch).
3. `ApplyFlashAttention` — `use_paged_prefill` (drives whether
   `FlashAttentionPagedPrefillProgram` runs vs. the dense
   `FlashAttentionProgram`); see [`onnxruntime/contrib_ops/webgpu/bert/flash_attention.cc`](../../onnxruntime/contrib_ops/webgpu/bert/flash_attention.cc).

`ApplyFlashAttention` also refuses to fall through to the dense prefill
program when `use_paged_kv_cache == true` (a `NOT_IMPLEMENTED` guard),
so a future feature added to `PagedAttention` without matching support in the
fused shader fails loud instead of silently corrupting the output.

**Unpack/Repack skip.** When direct paged attention is used, the packed
varlen Q buffer can be handed to FA as a rank-4 view without materializing
a padded BSNH scratch:

- **Uniform-batch mode** (`B * max_seqlen_q == token_count`): the packed
  buffer is byte-identical to `[B, max_seqlen_q, N, H]`. Applies to decode
  (`max_seqlen_q == 1`), `B == 1` prefill, and equal-length batched prefill
  (the common continuous-batching case).
- **Varlen-Q mode**: for non-uniform prefill, FA gets a
  `[token_count, 1, N, H]` view plus `cumulative_seqlens_q`; the fused
  paged-prefill shader reads Q rows through `q_row = cumulative_seqlens_q[b]
  + q_idx` (`q_varlen` template variant). Only safe when
  `ShouldRunFusedPagedPrefill` returns true.

Skipping unpack + repack removes two dispatches (~300–500 µs of CPU
dispatch cost per Run on D3D12) plus the padded scratch allocation
(`B * max_seqlen_q * hidden * 2 B`, tens of MB at long prefill).

### 4.4 Cost of the fallback path

Two full paged K/V reads plus two contiguous K/V writes per layer per step
(gather), Q/output shuffles proportional to `token_count`, and a scratch
allocation of `2 * B * kv_num_heads * max_kv_len * head_size + B *
max_seqlen_q * hidden_size` bytes. Fallback is exercised on configurations
that `ShouldRunFusedPagedPrefill` rejects (fp32, `head_size > 256`, or
`block_size < max_k_step`); direct paged paths cover the common case on
every WebGPU adapter.

### 4.5 Host-visible values and graph capture (deferred to Phase 2)

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

Phase 2 originally covered five items. Three landed in
[#31727](https://github.com/microsoft/onnxruntime/pull/31727); the other two
remain future work. Numbering is kept for cross-reference.

#### Phase 2 items landed in this PR

**1. Validate and tune direct paged decode.** ✅ Direct paged split-reduce
kernels (`FlashAttentionPagedDecodeQKV` + `FlashAttentionPagedDecodeVxReduce`)
are the production route when `max_seqlen_q < 32`. Both paths (direct paged
and gather-then-flash fallback) are tested end-to-end in
`onnxruntime/test/contrib_ops/paged_attention_op_test.cc` for MHA, GQA,
mixed batch lengths, packed QKV, rotary, and cache-aliasing configs, and are
benchmarked in
`onnxruntime/test/onnx/microbenchmark/paged_attention.cc`. The `< 32` cutoff
is the same threshold FA uses internally between its split-reduce decode and
single-kernel prefill tiers; no adapter-specific override has been needed.

**3. Fused paged prefill.** ✅ `FlashAttentionPagedPrefillProgram` (in
`flash_attention.{h,cc}` and `flash_attention_paged_prefill.wgsl.template`)
indexes `key_cache` / `value_cache` directly through `block_table`,
preserving the single-kernel prefill algorithm and workgroup Q tiling of
`FlashAttentionProgram`. It supports fp16, both BSNH Q and packed varlen Q
(`q_varlen` template variant), variable-Q-length causal masking via
`seqlen_k` + `seqlens_q`, and no attention bias / head sink / TurboQuant.
`ShouldRunFusedPagedPrefill` (see §4.3) is the shared selection gate.
Measured wins on the microbench matrix vs. the #31611 gather-then-flash
cascade: **1.01×–1.25×** on uniform prefill (geomean ~1.13×) and
**1.14×–1.73×** on varlen prefill (geomean ~1.29×).

**Skip Unpack/Repack fast paths.** ✅ New in this PR, not in the original
Phase 2 list. When direct paged attention is active, the packed varlen Q
buffer is handed to FA as a rank-4 view without materializing padded BSNH
Q/output scratch. Uniform-batch mode (`B * max_seqlen_q == token_count`)
covers decode, `B == 1` prefill, and equal-length batched prefill; varlen-Q
mode covers non-uniform prefill through the fused shader's `q_varlen`
template variant. Regression tested by
`EndToEnd_Prefill_MultiBatch_Varlen_Fused` in
`paged_attention_op_test.cc`. See §4.3 for the correctness invariant.

#### Phase 2 items remaining (future work)

**2. Complete deferred Phase 1 feature support.** Add and test the features
currently rejected by WebGPU: `softcap`, `local_window_size`, `head_sink`,
`use_smooth_softmax`, `q_norm_weight`, and `k_norm_weight`. Evaluate
`slot_mapping` including negative-slot skip-write semantics, plus
`rotary_offset` and non-default `v_head_size` when model compatibility
requires them. Add `bfloat16` only when target WebGPU adapters provide a
reliable storage and shader path. Each feature should have a support test
and an explicit unsupported-path test until implemented.

**4. Fuse packed-QKV preprocessing.** Combine the existing split-packed-QKV,
rotary, reshape, and cache-scatter stages, mirroring
`SplitPackedQKVWithRotaryEmbeddingAndCopyKVProgram`. This should remove
intermediate Q/K/V tensors and avoid an extra full-token read/write cycle.
Preserve the Phase 1 packed-QKV behavior and add parity tests for packed
non-rotary, packed rotary, interleaved rotary, MHA, and GQA cases.

**5. Make PagedAttention graph-capture-safe.** Consume
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
  paged_attention_rotary.wgsl.template                   # rotary embedding for Q or K
  paged_attention_scatter_kv.wgsl.template               # scatter K/V into paged cache
  paged_attention_gather_kv.wgsl.template                # gather paged K/V into padded scratch (fallback)
  paged_attention_unpack_query.wgsl.template             # unpack packed Q into LEFT-aligned BSNH (fallback)
  paged_attention_repack_output.wgsl.template            # repack padded output to packed output (fallback)
  # Paged-aware FA programs (Phase 2, in flash_attention.{h,cc}):
  flash_attention_paged_decode_qkv.wgsl.template         # direct paged split-K decode QKV
  flash_attention_paged_decode_vx_reduce.wgsl.template   # direct paged split-K decode reduce
  flash_attention_paged_prefill.wgsl.template            # fused paged prefill (q_varlen variant)
```

The direct paged decode / fused paged prefill programs are declared in
`flash_attention.h` alongside the dense-FA programs so they can share the
same `ApplyFlashAttention` dispatch surface, adapter cache hints, and
`ShouldRunFusedPagedPrefill` selection helper.

Shared helper (already exists on CUDA, refactor location in Phase 1):

```
onnxruntime/contrib_ops/cpu/bert/paged_attention_helper.h   # provider-neutral shape checks
```

Benchmark harness:

```
onnxruntime/test/onnx/microbenchmark/paged_attention.cc
```

The C++ harness (Google Benchmark) constructs each session before timing and
reports end-to-end latency with output synchronization. It intentionally
exposes MHA and GQA as separate cases: MHA uses one KV head per query head,
while GQA shares fewer KV heads across groups of query heads. Both `decode`
(one new token per request) and prefill (uniform and varlen Q lengths) modes
use the same cache and packed-sequence layout. The direct paged / fused
prefill programs are selected automatically by shape and adapter (no
runtime toggle).

---

## 7. Dispatch cascade

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

  # Route selection:
  use_direct_paged_decode   = (max_seqlen_q  < 32)
  use_direct_paged_prefill  = ShouldRunFusedPagedPrefill(context, is_fp16,
                                                        max_seqlen_q,
                                                        head_size, block_size)
  use_direct_paged_attention = use_direct_paged_decode or use_direct_paged_prefill

  # Unpack/Repack skip:
  uniform_q_lens   = (B * max_seqlen_q == token_count)
  varlen_mode      = (not uniform_q_lens) and use_direct_paged_prefill
  skip_unpack_repack = uniform_q_lens or varlen_mode

  # (a) Direct-paged fast path (Phase 2, common case):
  if use_direct_paged_attention:
    if skip_unpack_repack:
      q_view      = view over packed Q as [B, max_seqlen_q, N, H]
                    (uniform) or [token_count, 1, N, H] (varlen)
      output_view = matching view over the packed output buffer
      ApplyFlashAttention(q_view, key_cache, value_cache, block_table,
                          seqlens_q, seqlen_k, output_view,
                          cumulative_seqlens_q=varlen ? ptr : nullptr)
                          # picks FlashAttentionPagedPrefillProgram or
                          # FlashAttentionPagedDecode{QKV,VxReduce}
    else:
      RunUnpackQuery()      # padded BSNH Q
      ApplyFlashAttention() on paged cache directly (no gather)
      RunRepackOutput()

  # (b) Fallback (rejected by ShouldRunFusedPagedPrefill, e.g. fp32,
  #     head_size > 256, or block_size < max_k_step):
  else:
    RunGatherKV()          # -> [B, kv_num_heads, max_kv_len, head_size]
    RunUnpackQuery()       # -> [B, max_seqlen_q, num_heads, head_size]
    ApplyFlashAttention()  # dense FA reads gathered K/V scratch
    RunRepackOutput()      # -> [token_count, hidden_size]

  return OK
```

`max_seqlen_q` and `max_kv_len` are derived from one packed metadata D→H
readback per node. Phase 2 direct paths remove the gather step and, on
uniform-batch and packed-varlen callers, the padded Q/output round trip.
The residual D→H readback is a graph-capture blocker addressed by
outstanding Phase 2 item 5 (`attention_metadata` + indirect dispatch).

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
| Storage buffer binding max is adapter-dependent (128 MiB on some). | The paged cache is addressed through per-layer bindings, and the gather-then-flash scratch buffers are checked against `maxStorageBufferBindingSize` before allocation. On the direct-paged fast paths, the padded Q/output scratch check is skipped (see `skip_unpack_repack` in §4.3) so sparse varlen batches whose padded-Q size would exceed the limit but whose actual `token_count * hidden_size` Q buffer fits are not rejected. |
| WebGPU graph capture forbids host-visible reads mid-graph. | Consume `attention_metadata` (Phase 2) instead of D→H syncing `cumulative_seqlens_q`. |
| Subgroup width varies (16 Intel, 32 NV/AMD, 64 Qualcomm/Apple). | For the dense `FlashAttentionProgram` the `is_qualcomm`/`is_nvidia`/`is_apple`/`has_subgroups` cache-hint knobs select between shm and subgroup-shuffle variants. The paged prefill / paged decode shaders use no subgroup intrinsics (they run on every adapter that satisfies the dtype/shm-budget/alignment gates), so they do not need those knobs. |
| WGSL forbids mixed `i32 + u32` arithmetic. `cumulative_seqlens_q` is stored as `array<i32>` but the row index is `u32`; tint reports the resolution failure as an opaque `absl::…raw_hash_map<>::at` at runtime. | Cast the storage-buffer element to `u32` before the addition (`u32(cumulative_seqlens_q[b]) + q_idx`). Applies in both the fused prefill shader's `loadq` and `writeo`. |
| Fused paged prefill assumes one K/V tile lives entirely inside a single paged block, so both `loadk` and `loadv` do one `block_table` lookup per tile. `paged_attention_helper` only enforces `block_size >= 16` (power-of-two), so `block_size = 16` combined with fp16 `head_size <= 128` (`max_k_step = 32`) would splice into a physically-adjacent block that need not be the next entry in `block_table`. | `ShouldRunFusedPagedPrefill` rejects `block_size < max_k_step`; callers fall back to gather-then-flash. Both values are powers of two ≥ 16, so `>=` implies alignment. |

---

## 9. Testing plan

- **Correctness (host-side enforce paths, lavapipe-compatible)**: input
  validation tests. See the [`webgpu-local-testing`](../../.agents/skills/webgpu-local-testing/SKILL.md)
  skill for lavapipe details.
- **End-to-end op tests** (`onnxruntime/test/contrib_ops/paged_attention_op_test.cc`,
  `PagedAttention.EndToEnd_*`): cover MHA, GQA, single/multi-batch,
  variable past lengths, empty tokens (`token_count == 0` cache-only
  copy), packed QKV, rotary, mixed prefill+decode batches, and cache
  aliasing (via IO-binding). `EndToEnd_Prefill_MultiBatch_Varlen_Fused`
  is the regression test for the fused paged-prefill varlen path
  (§4.3).
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
- **Micro-benchmark** (`onnxruntime/test/onnx/microbenchmark/paged_attention.cc`,
  Google Benchmark): decode and prefill (uniform + varlen) shape matrix.
  The direct paged decode + fused paged prefill programs are the production
  routes on every fp16 adapter that meets the shm-budget and block-alignment
  gates. Peak scratch memory can be inspected in adapter tooling since the
  direct path skips the `k_padded`/`v_padded` and (for `skip_unpack_repack`)
  `q_padded`/`output_padded` allocations.
- **GenAI E2E**: run Phi-3-mini or Llama-3.2-1B through the GenAI continuous
  batching engine on WebGPU once GenAI PR #2330's `-e webgpu` gate is
  flipped.

---

## 10. Open items (do not block v1)

1. **Block-size waiver for non-CUDA EPs.** Are we allowed to relax the model-side `paged_block_size % 256 == 0` constraint in the GenAI builder for WebGPU? Decision: **keep the CUDA constraint in v1**; revisit if perf data shows it matters.
2. **`slot_mapping = -1` semantics.** GenAI doesn't emit `-1` today; v1 accepts the input and uses it as an override but doesn't implement the "skip on negative" branch. One-line follow-up when a customer needs speculative decoding on WebGPU.
3. **`bf16` support on WebGPU.** Adapter-dependent, gated by Dawn feature flag. v1 registers only `MLFloat16`. Add `BFloat16` when Dawn ships stable bf16 on the target adapters.
