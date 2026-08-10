# PagedAttention — CUDA Design Document

Status: **Draft / proposal**
Scope: `com.microsoft::PagedAttention`, CUDA Execution Provider
Related: [gqa.md](gqa.md) (`com.microsoft::GroupQueryAttention`)

> **Compatibility decision.** `PagedAttention` is already serialized as `com.microsoft::PagedAttention`
> opset 1. [§4](#4-schema-the-compatible-contract) is the single normative contract for that opset and
> evolves it **additively only**. [§21](#21-deferred-breaking-contract) records the breaking
> cache-layout ideas that were considered and rejected for opset 1; they are deferred to a separately
> versioned schema and must not replace the shipped contract in place.

## Table of Contents

- [1. Purpose and Scope](#1-purpose-and-scope)
- [2. Current State](#2-current-state)
- [3. Design Principles](#3-design-principles)
- [4. Schema — The Compatible Contract](#4-schema--the-compatible-contract)
- [5. Feature: `slot_mapping`](#5-feature-slot_mapping)
- [6. Feature: Attention Sink (`head_sink`)](#6-feature-attention-sink-head_sink)
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
- [21. Deferred Breaking Contract](#21-deferred-breaking-contract)

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
  *(Fixed. Dispatch and workspace sizing now use static shapes and upper bounds; see §4.7. The sync
  remains only as a prefill-side fallback that no capturable step reaches.)*

## 3. Design Principles

1. **Additive schema only.** `PagedAttention` is already shipped as contrib opset 1. All new inputs
   are appended at the end as `OpSchema::Optional`, all new attributes have defaults matching current
   behavior, and existing type constraints are only *widened*. No existing model breaks. This is
   binding: [§4.2](#42-compatibility-invariant) states the invariant, and the breaking alternatives
   that cannot satisfy it are deferred to [§21](#21-deferred-breaking-contract).
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
  quantities needed (see the `head_sink` LSE rescale in [§6](#6-feature-attention-sink-head_sink)).

## 4. Schema — The Compatible Contract

This section is the **single normative index table** for `com.microsoft::PagedAttention` opset 1.
Every other section in this document refers to these indices.

### 4.1 Decision

Evolve the operator additively. Preserve inputs 0–9, outputs 0–2, and every existing attribute with
its current meaning. Add model coverage and serving features through trailing optional inputs,
optional attributes whose defaults reproduce current behavior, a widened cache type constraint, and
`value_cache` becoming optional only for an explicitly selected latent-cache mode.

Do **not** merge `key_cache` and `value_cache`, replace the existing cache outputs, remove
`kv_num_heads`, rename `local_window_size`, or reinterpret an existing input combination. Those
options are recorded and deferred in [§21](#21-deferred-breaking-contract).

### 4.2 Compatibility Invariant

Every model valid under the shipped opset-1 schema remains valid and behaves identically when all
new inputs and attributes are absent. For such a model:

```text
T_CACHE == T
value_cache is present
key and value are either both present or both absent
kv_cache_layout == "SEPARATE"
v_head_size == 0
all quantization attributes select NONE
all trailing optional inputs are absent
```

The baseline this evolves from is the schema on ONNX Runtime `main`
(`onnxruntime/core/graph/contrib_ops/bert_defs.cc`): inputs 0–9 with `T`-typed caches, outputs 0–2
under a both-or-neither rule, and the seven attributes `num_heads`, `kv_num_heads`, `scale`,
`softcap`, `local_window_size`, `do_rotary`, `rotary_interleaved`.

### 4.3 Inputs

Indices 0–9 are unchanged from the shipped schema. Indices 10–18 are new and optional. The order
matches the landing order in [§19](#19-phasing), so the schema grows monotonically per phase.

| Idx | Name | Type | Shape | Status |
|-----|------|------|-------|--------|
| 0 | `query` | `T` | `(token_count, hidden_size)` or packed `(token_count, (num_heads + 2*kv_num_heads)*head_size)` | existing |
| 1 | `key` | `T` (opt) | `(token_count, kv_num_heads * head_size)` | existing — required in `LATENT` (§12) |
| 2 | `value` | `T` (opt) | `(token_count, kv_num_heads * head_size)` | existing — absent in `LATENT` (§12) |
| 3 | `key_cache` | **`T_CACHE`** | `(num_blocks, block_size, kv_num_heads, head_size)` | **type widened — §8** |
| 4 | `value_cache` | **`T_CACHE`** (opt) | `(num_blocks, block_size, kv_num_heads, head_size)` | **type widened; absent in `LATENT` — §12** |
| 5 | `cumulative_sequence_length` | `S` | `(batch_size + 1,)` | existing |
| 6 | `past_seqlens` | `S` | `(batch_size,)` | existing |
| 7 | `block_table` | `S` | `(batch_size, max_num_blocks_per_seq)` | existing — `-1` = unmapped (§9.2) |
| 8 | `cos_cache` | `T` (opt) | `(max_seq_len, rotary_dim/2)` | existing |
| 9 | `sin_cache` | `T` (opt) | `(max_seq_len, rotary_dim/2)` | existing |
| 10 | `slot_mapping` | `S` (opt) | `(token_count,)` | **new — §5** |
| 11 | `head_sink` | `T` (opt) | `(num_heads,)` | **new — §6** |
| 12 | `q_norm_weight` | `T` (opt) | `(head_size,)` | **new — §7** |
| 13 | `k_norm_weight` | `T` (opt) | `(head_size,)` | **new — §7** |
| 14 | `k_scale` | `T_KV_SCALE` (opt) | `(1,)` or `(kv_num_heads, 1, head_size)` | **new — §8** |
| 15 | `v_scale` | `T_KV_SCALE` (opt) | `(1,)` or `(kv_num_heads, 1, head_size)` | **new — §8**; absent in `LATENT` |
| 16 | `attention_metadata` | `S` (opt, **CPU**) | `(2,)` | **new — trusted bounds only, §4.7** |
| 17 | `query_positions` | `S` (opt) | `(token_count,)` | **new — §4.8** |
| 18 | `attention_bias` | `T` (opt) | `(batch_size or 1, num_heads or 1, query_length_capacity, context_length_capacity)` | **new — §10** |

`max_context_len` is the largest per-sequence total KV length in the batch, bounded above by
`block_table.shape[1] * block_size`.

`attention_bias` is deliberately last: it is a correctness fallback with potentially large memory
cost that disqualifies every fused backend. The rank and broadcast dimensions match GQA; only the
two sequence extents differ because PagedAttention is packed and ragged (§10).

**Why this table says `head_size` and not `v_head_size`.** `v_head_size` may differ from `head_size`
**only** in `"LATENT"` mode (§12.3), and a `"LATENT"` node has no `value`, no `value_cache`, and no
`v_scale` — V is a *view* of the leading `v_head_size` channels of `key_cache`, so there is nothing
left to give a separate width to. Every V-carrying tensor above therefore exists only in
`"SEPARATE"` mode, where `effective_v_head_size == head_size` by definition. The document spells
`effective_v_head_size` only in the two places where the width genuinely varies: output 0 (§4.4) and
the V view of `key_cache` (§12.3).

### 4.4 Outputs

| Idx | Name | Type | Shape | Status |
|-----|------|------|-------|--------|
| 0 | `output` | `T` | `(token_count, num_heads * effective_v_head_size)` | existing (shape generalized by §12) |
| 1 | `key_cache_out` | `T_CACHE` (opt) | aliases `key_cache` | existing |
| 2 | `value_cache_out` | `T_CACHE` (opt) | aliases `value_cache` | existing |
| 3 | `output_qk` | `QK` (opt) | `(num_heads, token_count, max_context_len)` | **new — §11** |

In `SEPARATE` mode outputs 1 and 2 are both present or both absent, preserving the shipped rule. In
`LATENT` mode output 1 may be present and output 2 must be absent.

Requesting `output_qk` requires a four-entry ONNX output list. Unused optional output positions are
represented by an **empty output name**; indices are never compacted. A `LATENT` node requesting
`output_qk` therefore writes `[output, key_cache_out-or-empty, "", output_qk]`.

`output_qk` uses its own `QK` type constraint rather than `T`, matching GQA, so the QK matrix can be
emitted in `float32` independently of the activation type.

**Aliasing.** The shipped implementation requires the cache outputs to be the *same buffer* as the
corresponding inputs, but enforces it only as a runtime pointer-equality check in
`paged_attention.cc` that returns `INVALID_ARGUMENT`. There is no `.Alias()` in the kernel definition,
so a graph in which the allocation planner does not happen to reuse the input buffer fails at run
time rather than at partition time — a violation of design principle 4 (§3). Registering
`.Alias(3, 1).Alias(4, 2)` on the kernel def is fully compatible and belongs in P0 (§19). A
functional-output contract that stays correct when the planner cannot reuse the buffer is a
different, breaking contract and is deferred (§21).

### 4.5 Attributes

Names, defaults, and value sets follow GQA so a graph rewriter can move an attribute between the two
ops without translation.

| Name | Type | Default | Status |
|---|---|---|---|
| `num_heads` | INT | required | existing |
| `kv_num_heads` | INT | required | existing |
| `scale` | FLOAT | `1/sqrt(head_size)` | existing — mandatory in `LATENT` (§12.6) |
| `softcap` | FLOAT | `0.0` | existing |
| `local_window_size` | INT | `-1` | existing — §9 |
| `do_rotary` | INT | `0` | existing |
| `rotary_interleaved` | INT | `0` | existing |
| `qk_norm_epsilon` | FLOAT | `1e-6` | **new — §7** |
| `k_quant_type` | STRING | `"NONE"` | **new — §8**; `NONE` \| `PER_TENSOR` \| `PER_CHANNEL` |
| `v_quant_type` | STRING | `"NONE"` | **new — §8**; same value set |
| `k_cache_dtype` | STRING | `""` | **new — §8**; `""` = the K cache tensor's own element type |
| `v_cache_dtype` | STRING | `""` | **new — §8**; `""` = the V cache tensor's own element type |
| `v_head_size` | INT | `0` | **new — §12**; `0` = `head_size`. Non-zero and `!= head_size` is legal **only** in `LATENT` |
| `rotary_offset` | INT | `0` | **new — §12.5** |
| `kv_cache_layout` | STRING | `"SEPARATE"` | **new — §12**; `SEPARATE` \| `LATENT` |
| `qk_output` | INT | `0` | **new — §11**; `0` none, `1` pre-softmax, `2` post-softmax |

`k_cache_dtype` and `v_cache_dtype` name the *logical* element type of each cache. Every value is
spelled as the ONNX element type it denotes. `""` — the default — means the cache tensor's own
element type is also the logical type; `"float16"`, `"bfloat16"`, `"int8"` and `"float8e4m3fn"` name
that same type explicitly and must agree with the tensor. The reserved values `"int4"` and
`"float4e2m1"` describe sub-byte types packed two per byte into a `uint8` cache (§21.4), which
no ONNX tensor type can express here; they are rejected until a sub-byte backend exists. Every
value is a signed, zero-symmetric type — there is no zero-point input, so `uint4` / `uint8` are
deliberately not in the vocabulary (§8.3.1).

A *dtype*, not a bit width, is the right unit here. Bit width alone cannot distinguish `int4` from
`float4e2m1` — same width, unrelated decode math — and for every non-packed format it is pure
redundancy against the cache tensor's element type, i.e. a second source of truth that can only ever
agree or disagree. K and V stay independent so future formats may use different precisions for each
(for example, TurboQuant-style mixed-precision caches). The current schema still binds both cache
tensors to `T_CACHE`; a future format whose K and V tensors have different ONNX element types must
additionally split that type constraint into `T_K_CACHE` and `T_V_CACHE`. Byte-backed sub-byte
formats can differ in logical width while both tensors remain `uint8`.

Do **not** add `window_size_left` / `window_size_right` to opset 1. `local_window_size = W` has the
established meaning of admitting `W` positions *including* the current token; its Flash parameters
are `window_size_left = W - 1`, `window_size_right = 0`.

### 4.6 Input-Mode State Machine

Input mode is determined by `kv_cache_layout` and Q/K/V presence. No separate format attribute is
needed.

| `kv_cache_layout` | `query` | `key` | `value` | `value_cache` | Mode |
|---|---|---|---|---|---|
| `SEPARATE` | Q | present | present | required | separate Q/K/V |
| `SEPARATE` | packed QKV | absent | absent | required | packed QKV |
| `LATENT` | absorbed Q | present (latent row) | absent | absent | absorbed MLA |

Every other presence pattern is `INVALID_ARGUMENT`. In particular, `SEPARATE` preserves the shipped
rule exactly: K and V are packed inside `query` iff both `key` and `value` are absent. `LATENT` is
explicitly selected by `kv_cache_layout`, so its key-present/value-absent pattern cannot be confused
with packed QKV.

Shape inference and runtime validation must inspect `kv_cache_layout` **before** applying the shipped
`value absent => packed QKV` branch. In `LATENT`, `query.shape[1] / num_heads` determines
`head_size`; in packed `SEPARATE`, the existing
`query.shape[1] / (num_heads + 2 * kv_num_heads)` formula remains unchanged.

### 4.7 CUDA Graph Contract and `attention_metadata`

Decode is launch-bound, so CUDA graph replay is the main performance lever and the schema must not
obstruct it. Two properties of the ORT implementation determine the whole contract:

1. `cudaStreamSynchronize` on a capturing stream is illegal, so the unconditional per-step D→H sync
   in `paged_attention.cc` (§18.4) makes the operator **uncapturable** today.
2. On replay, `InferenceSession::Run` short-circuits to
   `cached_execution_provider_for_graph_replay_.ReplayGraph(...)` and never builds an
   `ExecutionFrame`. **No kernel's `ComputeInternal` runs again.** Backend selection, grid
   dimensions and workspace extents are all frozen at capture.

Property 2 is the one that constrains the schema. A CPU-resident input is read exactly once, at
capture. `max_kv_len` and `total_kv_tokens` grow with every decode step, so a host input carrying
their *exact* values would leave a captured graph attending over the capture-step's KV length for the
rest of the sequence — silently wrong, and undetectable by the producer. "Recapture when the metadata
changes" is not a mitigation for decode, where it changes every token.

> **Replay-invariance rule.** No host-visible value may determine a loop trip count, a mask boundary,
> or a memory extent that varies per step. Host values may only *select* the kernel and *size* the
> launch, and must stay valid for every step the captured graph will serve. Every per-step quantity
> is read on device from `cumulative_sequence_length`, `past_seqlens` and `block_table`.

Three derivations satisfy the rule, none of which needs a synchronization:

| Quantity | Source | Replay-safe because |
|---|---|---|
| decode vs. prefill dispatch | static shapes: `query.shape[0] == cumulative_sequence_length.shape[0] - 1` | shapes are fixed for a captured graph |
| grid size, split count, gather/workspace extents | static capacity bound `max_kv_len_bound = block_table.shape[1] * block_size` | independent of step |
| per-sequence KV length, causal and window masking, gather trip counts | device `past_seqlens` / `cumulative_sequence_length` | re-read from device memory on every replay |

The shape test is a **performance heuristic only**. `token_count <= batch_size` does not prove that
every sequence contributes exactly one query token — one sequence may contribute two while another
contributes none. The paged decode kernel must therefore derive each token's sequence and position
from `cumulative_sequence_length` on device and stay correct for ragged input. Correctness never
depends on the heuristic being right, only speed.

This removes the D→H sync **unconditionally** and without any new input, which is a stronger result
than making the sync conditional on a host hint. Sizing by the capacity bound instead of the exact
length costs empty split blocks that exit after a single device read — far cheaper than a per-layer,
per-step sync.

`attention_metadata` is consequently demoted to optional **replay-wide bounds**:

```text
attention_metadata : (2,) int32, OrtMemTypeCPUInput
  [0] max_query_len_bound   # 0 = unknown. Replay-wide upper bound on tokens from any one sequence.
  [1] max_kv_len_bound      # 0 = unknown. Replay-wide upper bound on total KV length of any sequence.
```

- Both entries are **upper bounds, never exact values**, and must hold for *every* step the node —
  or the captured graph containing it — will serve.
- `0` means "no bound"; the implementation falls back to `token_count` for query length and to
  `block_table.shape[1] * block_size` for KV length.
- The implementation clamps an over-large bound to the corresponding static limit. A non-zero
  bound smaller than an actual per-step value violates the input contract and may omit work and
  produce an incorrect result. The kernel cannot detect that violation without reading the device
  length tensors back to the host.
- A valid bound may only shrink launch dimensions and workspace sizes. It must not enter a mask
  comparison. Device loops use the current device lengths, additionally bounded by the trusted
  launch/workspace extent.
- Even for an invalid bound, every device read must remain memory-safe: device lengths and static
  tensor capacities guard all accesses. This safety property does not imply a correct result when
  the producer violates the upper-bound contract.
- It must be a **graph input fed from the host**, never the output of an in-graph node. An in-graph
  producer would be placed on the CPU EP and trip the `AreAllNodesInMainGraphAssignedToOneEp` check
  that gates graph capture.

`total_kv_tokens` is **not** part of the input. It was only ever used to size the MEA gather buffer,
which must now be sized by `batch_size * max_kv_len_bound` so that the allocation is replay-invariant.

> **Implementation note (implemented).** `attention_metadata` exists as input 16 with the semantics
> above, and the D→H readback is gone from every configuration a CUDA graph can reach.
>
> Backend selection is now the static shape test of the table above: `token_count <= batch_size`
> selects the paged decode backend (when the cache is quantized or FlashAttention is ineligible),
> with no host knowledge of the actual per-sequence query lengths. That is safe because
> `PagedDecodeSplitKV` is indexed by **global query token** — it resolves each token's sequence and
> position from `cumulative_seqlens_q` on device and applies per-token causal and window masks — so
> a wrong heuristic costs speed, never correctness. The fused QK-Norm / rotary prologue was made
> token-indexed for the same reason, which removed its dependence on `max_query_len` entirely.
>
> Everything the host still needs is an upper bound, and each bound has a static fallback used when
> no metadata is supplied:
>
> | Host quantity | Consumer | Bound used |
> |---|---|---|
> | `max_query_len` | `params.seqlen_q` (Flash), `p.sequence_length` (MEA) — grid extent only | `max_query_len_bound`, else `token_count` |
> | `max_kv_len` | quantized-Flash `max_seqlen_k`, decode split count | `max_kv_len_bound`, else `block_table.shape[1] * block_size` |
> | `total_kv_tokens` | gather staging buffer extent | `batch_size * max_kv_len_bound` |
>
> Two narrow cases still take the readback, and only when the caller supplied **no** metadata at all:
> a gather backend (MEA, or FlashAttention on a quantized cache), because with no bound the
> capacity-based staging allocation would be a large over-allocation for a short prefill; and XQA,
> whose one-output-row-per-batch-index layout needs *proof* of exactly one token per sequence rather
> than the shape heuristic. Neither case can occur on a capturable step — a captured step is
> decode-shaped over a paged cache, so no gather runs, and a producer that captures must supply
> `attention_metadata` anyway, since replay-wide bounds are the only replay-safe host input. Both
> cases are prefill-side, and prefill is never captured.
>
> Measured with nsys on a single-node decode graph, this removes one `cudaStreamSynchronize` and two
> D→H `cudaMemcpyAsync` per node per `Run`, leaving zero of either — for the **unquantized** cache as
> well as the quantized ones, which is what makes `enable_cuda_graph=true` work for every KV type.

Producers additionally owe the usual CUDA-graph obligations, which are outside this operator's
contract: fixed device addresses for `key_cache`, `value_cache`, `block_table`, `past_seqlens` and
`cumulative_sequence_length` across replays, and separate captures for prefill and decode.

### 4.8 `query_positions`

When absent, token `j` of sequence `b` has position `past_seqlens[b] + j` — the shipped behavior.
When present, element `j` supplies the linear logical position used for in-op RoPE and for backends
that support arbitrary-position causal and window masking.

This covers linear position overrides and chunked prefill. It does **not** by itself encode tree
ancestry: Medusa and general tree attention additionally require a branch/ancestor mask, which is
deferred (§21). Backends that derive positions solely from packed row alignment are ineligible
whenever the supplied positions differ from the legacy sequence.

Cached K is stored **after** RoPE, so reusing a cached block at a different logical position is
invalid unless the producer guarantees the block was rotated for that position, or the backend
re-rotates on read. `query_positions` does not make an already-rotated prefix relocatable.

`query_positions` is `int32` and 1-D over `token_count`, unlike GQA's `position_ids`, which is
`int64` and 2-D over `(batch_size, sequence_length)`. The divergence is deliberate — the packed
varlen layout has no `(batch, seq)` grid — but it does mean a GQA↔PagedAttention rewriter needs a
`Cast` plus a reshape here.

### 4.9 Type Constraints

| Name | Allowed | Change |
|---|---|---|
| `T` | `float16`, `bfloat16` | unchanged |
| `T_CACHE` | `float16`, `bfloat16`, `int8`, `float8e4m3fn` | **new** (split out of `T`) |
| `T_KV_SCALE` | `float` | **new** |
| `QK` | `float`, `float16`, `bfloat16` | **new** — §11 |
| `S` | `int32` | unchanged |

Splitting `T_CACHE` out of `T` is backward compatible: every previously valid model has
`T_CACHE == T`. The constraint name is `T_KV_SCALE`, matching GQA and the registration already in
`paged_attention.cc`.

`uint8` is **intentionally** omitted from `T_CACHE`, even though GQA's `T_CACHE` already admits it
for packed INT4. There is no unsigned or sub-byte logical cache format specified for this operator
yet (§21), and widening a type constraint later is itself a compatible change, so nothing is lost by
waiting. This is a deliberate divergence from GQA, not an oversight.

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

## 6. Feature: Attention Sink (`head_sink`)

### 6.1 Math

Identical to GQA. With per-head sink value $s_h$ over $T$ attended positions:

$$
\text{softmax}_i = \frac{e^{x_i - m}}{e^{s_h - m} + \sum_{j} e^{x_j - m}}, \qquad m = \max\!\left(s_h, \max_j x_j\right)
$$

Equivalent to appending one extra logit $s_h$ that contributes to the denominator only. Used by
GPT-OSS. The internal sink-enabled softmax path is selected whenever `head_sink` is present; there
is no standalone attribute for an unlearned zero-valued sink.

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
decode path. Scope for this phase: **`PER_TENSOR` and `PER_CHANNEL`**, with `k_cache_dtype` and
`v_cache_dtype` left at `""` (or naming the cache tensor's own element type).
INT4 is deferred ([§19](#19-phasing)).

### 8.2 Schema

- `key_cache` / `value_cache` move from `T` to `T_CACHE ∈ {float16, bfloat16, int8, float8e4m3fn}`.
  `uint8` is intentionally excluded until a sub-byte format is specified (§4.9, §21).
- `k_scale` / `v_scale` (inputs 14, 15), type `T_KV_SCALE` = **always FP32**, matching GQA.
- Attributes `k_quant_type`, `v_quant_type` ∈ `{"NONE", "PER_TENSOR", "PER_CHANNEL"}`, plus
  independent `k_cache_dtype` and `v_cache_dtype` attributes, which stay `""` while every logical
  type is expressible as an ONNX element type.
- Kernel becomes `PagedAttention<T, T_CACHE>`, registered for the same combinations GQA uses:
  `{MLFloat16, BFloat16} × {same as T, int8_t, Float8E4M3FN}` (plus `uint8_t` if and when INT4 lands).

### 8.3 Scale layout under the block layout

The block cache is `(num_blocks, block_size, kv_num_heads, head_size)`. A `PER_CHANNEL` scale of
shape `(kv_num_heads, 1, head_size)` — the same shape GQA uses — broadcasts naturally over the
leading `(num_blocks, block_size)` dims. **This is why the GQA scale shape is reused verbatim**: the
quantize/dequantize helpers index only on `(kv_head, channel)` and are layout-agnostic.

| Mode | Scale shape | Indexing |
|---|---|---|
| `PER_TENSOR` | `(1,)` | scalar |
| `PER_CHANNEL` | `(kv_num_heads, 1, head_size)` | `scale[h * head_size + c]` |

`v_scale` uses the same last dimension, `head_size`: it exists only where a `value_cache` exists,
which is `"SEPARATE"` mode, and `effective_v_head_size == head_size` there (§12.3). Under
`"LATENT"` there is one physical cache and `k_scale` alone describes it, so `v_scale` is rejected
(§8.7, §12.9); the V dequant of that cache indexes `k_scale` with the `head_size` stride and reads
only its leading `v_head_size` entries.

Symmetric quantization, same formulas as GQA:

| Type | Range | Quantize |
|---|---|---|
| INT8 | `[-128, 127]` | `q = clamp(round(x / scale), -128, 127)` |
| FP8 E4M3 | `[-448, 448]` | `q = clamp(x / scale, -448, 448)` |
| INT4 (deferred) | `[-8, 7]`, 2/byte | last cache dim becomes `(head_size + 1) / 2` |

#### 8.3.1 Zero point: always 0, and why the vocabulary is signed-only

There is **no zero-point input and no zero-point attribute**. Dequantization is exactly
`x = q * scale`, with an implied zero point of `0`. Consequently every value `k_cache_dtype` /
`v_cache_dtype` may name is a *signed, zero-symmetric* type: `int8`, `float8e4m3fn`, `int4`,
`float4e2m1`, plus the unquantized floats. `uint4` and `uint8` are deliberately **absent**: an
unsigned logical type under symmetric quantization implies an offset of `2^(bits-1)` (128 for uint8,
8 for uint4) that nothing in the contract carries, so admitting the name would let two
implementations disagree about whether the stored code is biased. Unsigned logical types must arrive
together with the optional zero-point inputs sketched in §21.3, not before.

> **Storage bias is not a zero point.** INT4 is *stored* in an unsigned nibble biased by `+8`
> (`store = q + 8`, `load = nibble - 8`), which is how ORT's MLAS KV path already packs its `S4`
> modes (`kInt4Bias` in `mlas/lib/qkv_quant_common.h`). That bias is a storage encoding removed
> before the value is used; the *logical* value stays signed in `[-8, 7]` and the dequantization is
> still `q_signed * scale`. Naming that format `uint4` would be wrong.

**The kernels depend on this, not merely comply with it.** The §8.5 Phase 3 decode kernel folds
`k_scale` into Q at load time and `v_scale` into the reduce epilogue. Those foldings are exact only
because the zero point is 0:

$$\sum_c q_c \cdot (k_c \cdot s_k) = \sum_c (q_c \cdot s_k) \cdot k_c$$

With a non-zero zero point $z$ the score becomes
$\sum_c q_c (k_c - z) s_k = s_k \sum_c q_c k_c - z\,s_k \sum_c q_c$, so a per-(token, head)
correction term appears in the QK product and a second one in the PV product. That is a structural
change to the kernel, not an extra subtraction — which is the concrete reason zero points are a
versioned-successor topic rather than a late addition.

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
- `T_CACHE != T` requires a non-`NONE` quant type; `T_CACHE == T` requires both to be `NONE` and both
  scales to be absent.
- `k_cache_dtype` and `v_cache_dtype` are `""` for every cache this operator stores, quantized or
  not: the cache tensor's element type is the logical element type. Naming that type explicitly
  (`"float16"`, `"bfloat16"`, `"int8"`, `"float8e4m3fn"`) is accepted but must agree with the tensor.
  `"int4"` and `"float4e2m1"` are reserved for a `uint8` packed cache and are rejected
  until one exists. Unsigned logical types (`uint4`, `uint8`) are rejected outright: quantization
  here has no zero point (§8.3.1).
- In `"LATENT"` mode only K storage exists: `k_quant_type` and `k_scale` describe the latent row,
  `v_quant_type` and `v_cache_dtype` must be unset, and `v_scale` must be
  absent because V is a view of K.
- FP8 is available when ORT is built with `onnxruntime_USE_FP8_KV_CACHE`; no additional runtime
  architecture gate is required for the conversion path used by this operator.
- `PER_CHANNEL` scale shape must be exactly `(kv_num_heads, 1, head_size)` for both K and V. There
  is no `v_head_size`-shaped scale: `v_scale` only exists alongside a `value_cache`, i.e. in
  `"SEPARATE"` mode, where `effective_v_head_size == head_size`.

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
> - **`uint8` / INT4 not added.** `T_CACHE` is `{float16, bfloat16, int8, float8e4m3fn}`, so
>   `k_cache_dtype` and `v_cache_dtype` must be `""` or name the cache tensor's own element type.
> - **No SM89/SM90 gate for FP8.** `Float8E4M3FN`'s converting constructor uses
>   `__nv_cvt_float_to_fp8`, which is available on every architecture ORT builds for from CUDA 11.8
>   onward, so the arch check in §8.7 would reject working configurations. FP8 remains gated at
>   *build* time by `onnxruntime_USE_FP8_KV_CACHE`.
> - Parity tests compare the updated cache at one quantization step of slack. Rotary and RMSNorm are
>   computed ~1 fp16 ULP differently on the host, which is enough to move a value across a rounding
>   boundary and flip the stored code by one LSB.

> **Implementation note (P5, implemented).** §8.5 Phase 3 is now in place as a purpose-built
> flash-decoding kernel (`PagedDecodeSplitKV` + `PagedDecodeReduce` in `paged_attention_impl.cu`),
> not by reusing the vendored XQA kernel. XQA does have paged-KV support, but its page list is
> `[batch][beam][2][max_pages]` over a *single* K/V pool whereas PagedAttention has two pools and one
> shared `block_table`, and it restricts `head_size ∈ {64, 128, 256}` and `group_size ∈ {4, 8, 16,
> 32}` while multiplying instantiations across page size × group size × head size × dtype × quant
> type. A ~250-line kernel covers the whole schema instead.
>
> - **Both scale foldings are exact and granularity-agnostic.** K folds into Q at load time
>   (`q_sh[c] = float(q[c]) * GetCacheScale(k_scale, kv_head * head_size + c, k_per_channel)`), so
>   `PER_TENSOR` is just the `per_channel == false` branch of the same expression rather than a
>   separate "fold into the softmax scale" path. V folds into the epilogue: `v_scale_c` does not
>   depend on the KV position, so it factors out of the accumulation entirely and never enters the
>   softmax denominator.
> - The kernel reads pages in place at their stored width, so a decode step touches the KV cache once
>   at `int8`/`fp8` bandwidth instead of gathering and dequantizing the whole live context.
> - `softcap` matches FlashAttention bit-for-bit: `softcap * tanh(qk_raw * scale / softcap)`, which is
>   what `flash_api.cc` produces from `params.softcap = softmax_scale / softcap` and
>   `params.scale_softmax = softcap`.
> - The attention sink enters as one extra `exp(sink - m_final)` term in the final denominator, which
>   is algebraically identical to §6.2's `factor = 1 / (1 + exp(sink - lse))` epilogue but does not
>   need FlashAttention's log-sum-exp output. §6.3's MEA restriction therefore applies only to MEA.
> - Sliding window uses the token's own causal position `q_pos = kv_len - 1`, admitting
>   `t ∈ [kv_len - local_window_size, kv_len)`, matching Flash's `window_size_left = local_window_size - 1`.
> - **Backend gating.** The kernel is selected by the static shape test of §4.7
>   (`token_count <= batch_size`) when the cache is quantized *or* FlashAttention is unavailable;
>   unquantized FlashAttention-eligible shapes keep using FlashAttention. `sdpa_kernel = 512`
>   (`AttentionBackend::DECODER_ATTENTION`) forces it, which is how the unquantized path is tested.
>   The shape test is a heuristic: one CTA owns one **global query token**, resolves its sequence and
>   in-sequence position from `cumulative_seqlens_q` on device, and masks against
>   `past_seqlens[b] + q_index + 1`, so the kernel is correct for arbitrary ragged input (including
>   full prefill) and a wrong heuristic only costs speed. That is what removes the D→H sync.
> - Split-KV: `ComputePagedDecodeSplits` splits the KV range across up to 32 CTAs only when
>   `token_count * num_heads` would leave the device under-occupied. `max_kv_len` may be an upper
>   bound. Empty splits publish `(max = -FLT_MAX, denom = 0)` and the reduce kernel skips them, so
>   their accumulator slice is never read.
> - The `FlashAttention` / `EfficientAttention` prologue (packed-QKV unpack, fused QK-Norm + rotary,
>   `ReshapeAndCache`) was factored into a shared `PrepareQueryAndCache`, which the decode backend
>   reuses. It lives outside the `USE_FLASH_ATTENTION` / `USE_MEMORY_EFFICIENT_ATTENTION` guards
>   because the decode backend needs neither.
> - **Still deferred from P5:** the fused MLA decode backend (§12.7).

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

### 9.3 Bounded-capacity (rolling) sliding-window cache

§9.2 is about not *reading* out-of-window KV. This sub-section is about not *storing* it: when
`local_window_size = W > 0`, a sequence of context length $L$ only ever needs $O(W)$ cached tokens,
not $O(L)$. For a 128K context with a 4K window that is a ~32× reduction in KV memory, which is what
decides how many sequences fit on the device.

**Yes — it is expressed through `slot_mapping`, but not through `slot_mapping` alone.** The
capability splits exactly along the write/read line of [§5](#5-feature-slot_mapping):

| Concern | Mechanism | Owner |
|---|---|---|
| Reuse a physical block for a later token | `slot_mapping` (§5) — the *write* path | Runtime / scheduler |
| Stop reading an evicted block | `block_table[b][i] = -1` — the *read* path | Runtime / scheduler |
| Not attending out-of-window positions | `local_window_size` mask inside the backend | Kernel |
| Not *fetching* out-of-window blocks | window-clamped block walk (§9.2 item 2) | Kernel |

The derived slot resolver cannot do this: it computes
`slot = block_table[b][position / block_size] * block_size + position % block_size`, so a block is
implicitly owned by one absolute position range forever and the allocation grows with $L$. An
explicit `slot_mapping` lets the same physical block be rewritten by a token $W$ positions later,
which is the whole trick.

#### 9.3.1 Invariant: the block table stays indexed by absolute position

`block_table[b][i]` must keep meaning "the block holding positions
`[i * block_size, (i+1) * block_size)` of sequence `b`". Do **not** compact or shift the row so that
entry 0 becomes the window start. Every backend derives its mask from absolute positions —
FlashAttention from `seqlen_k` (`n_block_min` is computed from `actual_seqlen_k`), the paged decode
kernel from `kv_len - 1`, the latent kernel from `past_seqlens[b] + s` — and RoPE has already been
applied at the absolute position when the row was written. Rotating the row would silently decouple
position from mask and from RoPE phase.

So an evicting runtime presents a **sparse row**: entries below the window are `-1`, entries inside
it point into a small recycled pool. `-1` is defined as *"block not mapped — treat every position in
it as masked out"*, consistent with `slot_mapping`'s `-1`, and is already honoured by the slot
resolver, the gather kernel, the paged decode kernel and the latent kernel.

#### 9.3.2 Allocation policies

Let $N = \lceil W / \text{block\_size} \rceil + 1$. The `+1` absorbs the partially filled boundary
block, so the oldest still-in-window position is never evicted early.

1. **Ring buffer** (TRT-LLM's *cyclic* KV cache, vLLM's sliding-window allocator). The runtime holds
   $N$ blocks `ring[0 .. N-1]` per sequence and fills, for absolute position $p$ with
   $i = \lfloor p / \text{block\_size} \rfloor$:

   ```
   slot_mapping[t]    = ring[i % N] * block_size + (p % block_size)
   block_table[b][i]  = ring[i % N]                       // i inside the live range
   block_table[b][j]  = -1                                // j below the live range
   ```

   Steady-state KV memory per sequence is exactly `N * block_size` tokens, independent of $L$.
2. **Free-list eviction.** After a step, return every block whose highest position falls below the
   window start to the global allocator and write `-1` into its entry. Same asymptotic memory, more
   allocator traffic, but it interleaves with prefix caching and with layers that are *not* sliding
   (GPT-OSS alternates), which a per-sequence ring does not.

**Eviction condition.** The earliest query position a sequence will use in the current step is
$p_{\min} = \texttt{past\_seqlens}[b]$, and its window admits positions $\ge p_{\min} - W + 1$. So
after the step, blocks with

$$i < \left\lfloor \frac{\max(0,\ \texttt{past\_seqlens}[b] - W + 1)}{\text{block\_size}} \right\rfloor$$

are unreachable forever (query positions only move forward), and may be recycled. Eviction is
therefore a *post-step* runtime action; the kernel never frees anything.

#### 9.3.3 Backend obligations

| Backend | Behaviour on an out-of-window `-1` entry |
|---|---|
| Paged decode | `kv_begin = max(kv_begin, kv_len - W)` already skips the range; `block_id < 0` additionally forces `-FLT_MAX`. Safe. |
| Latent / MLA | `kv_begin = max(0, kv_end - W)`. Safe. |
| FlashAttention varlen (paged) | `n_block_min = max(0, (m_block * kBlockM + seqlen_k - seqlen_q - window_size_left) / kBlockN)` — out-of-window pages are never dereferenced. Requires `block_size % kBlockN == 0`, which is already a Flash *eligibility* condition (§13). Safe. |
| Gather path (MEA, quantized cache) | The gather **zero-fills** unmapped rows. A zero key yields logit $0$, not $-\infty$, so correctness relies on MEA's own window mask covering exactly the same positions. It does, because both derive from the same $W$ — but this makes the invariant below load-bearing. |

> **Invariant (load-bearing).** A `-1` entry must never overlap a position the mask admits. The
> runtime may only unmap blocks strictly below the window start. Violating it is silently wrong on
> the gather path (zeros contribute weight) rather than loudly wrong. Add a debug-build assertion in
> the gather kernel: `block_id >= 0 || pos < kv_len - W`.

#### 9.3.4 Follow-on kernel work this exposes

1. **Window-clamped gather.** `GatherAndExpandPagedKVCache` still materializes $L$ tokens of
   workspace even when only $W$ are readable. Starting the gather at
   `first_block = max(0, (L_b - W) / block_size)` shrinks both the workspace and the gather traffic
   to $O(W)$ — this is §9.2 item 2 applied to the gather path, and it is the difference between the
   quantized/MEA backends being $O(L)$ and $O(W)$ per step.
2. **Split-KV layout.** `ComputePagedDecodeSplits` lays splits out over the full `kv_len` and each
   CTA then clamps to `kv_len - W`, so with $W \ll L$ most CTAs launch only to exit immediately. The
   split range should be computed over the clamped interval `[max(0, kv_len - W), kv_len)`.
3. **No schema change is required.** The block-table *row* stays $O(L / \text{block\_size})$, but at
   4 bytes per `block_size` tokens versus `2 * kv_num_heads * head_size * sizeof(T)` bytes per token
   of KV, the row is ~0.01% of what it indexes — not worth a breaking change. Shrinking the row too
   would need a per-sequence window origin (a rotated block table plus a `kv_start_positions` input);
   that is listed with the other deferred contract changes in [§21](#21-deferred-breaking-contract).

#### 9.3.5 Validation

- Ring-buffer run (only $N$ blocks allocated per sequence, `slot_mapping` recycling them, leading
  `block_table` entries `-1`) must be **bit-identical** to a full-cache run with the same window, for
  every decode step past $L > W$.
- `W >= L` with recycling enabled must still equal full attention (nothing is ever evicted).
- Mixed batch: some sequences past the window, some not, in the same step.
- Composition: ring buffer × quantized cache (the gather path is the risky one) × `head_sink`.

## 10. Feature: `attention_bias`

### 10.1 GQA-compatible shape

GQA defines `attention_bias` as
`(batch_size or 1, num_heads or 1, sequence_length, total_sequence_length)`. PagedAttention keeps
the same rank, dimension roles, and independent batch/head broadcasting. Its packed ragged batch has
no single query or context length, so the two sequence dimensions use batch maxima:

```text
attention_bias : (batch_size or 1, num_heads or 1, query_length_capacity, context_length_capacity)
```

The last two dimensions are replay-wide capacities satisfying
`query_length_capacity >= max_b q_len_b` and
`context_length_capacity >= max_b (past_seqlens[b] + q_len_b)` for every step the node or captured
graph will serve. They may equal the exact maxima in an uncaptured run. The tensor may broadcast
across all sequences, all heads, or both, exactly as GQA does.

### 10.2 Design

Input 18, `attention_bias`, type `T`. For packed token `t` belonging to sequence `b`, let
`j = t - cumulative_sequence_length[b]` be its sequence-local query offset. Query head `h` and
logical KV position `k` use:

```text
attention_bias[b_or_0, h_or_0, j, k]
```

The first two indices select `0` when that dimension broadcasts. Entries with `j >= q_len_b` or
`k >= past_seqlens[b] + q_len_b` are padding and are ignored. This is the direct packed equivalent
of GQA's indexing contract; flattening batch and query into `(num_heads, token_count, ...)` is
rejected because it loses GQA's batch-broadcast dimension and prevents schema-level parity.

The bias capacities are host-visible shapes and therefore frozen during CUDA graph replay. Device
code must bounds-check `j` and `k` against them before reading the bias. An under-sized capacity is
a producer contract violation that may omit attention work, but it must not cause an out-of-bounds
read. This mirrors the trusted-bound rule for `attention_metadata` (§4.7).

- Rejected on the Flash varlen, paged-decode, and current MEA paths. Initially supported only on the
  unfused fallback, matching GQA. MEA may become eligible after its wrapper supports the rank-4
  batch/head broadcasting and ragged sequence-local row strides.
- Bias is applied after `scale` and before `softcap`, matching GQA.
- Emit an explicit `INVALID_ARGUMENT` when `attention_bias` is combined with a backend that cannot
  serve it, naming the backend and the reason.

### 10.3 No dedicated ALiBi input

Do not add `alibi_slopes` in this schema revision. GQA has no corresponding input, and the general
`attention_bias` contract already expresses ALiBi when an exporter is willing to materialize it.
A dedicated fused representation remains an additive future extension if a concrete model or
performance requirement justifies adding it to both attention operators.

## 11. Feature: `output_qk`

### 11.1 Design

- Attribute `qk_output`: `0` = no output (default), `1` = pre-softmax scores, `2` = post-softmax
  probabilities. Same encoding as GQA's `QKOutputType`.
- Output 3, `output_qk`, type `QK`, shape `(num_heads, token_count, max_context_len)`. `QK` is a
  separate type constraint (`float`, `float16`, `bfloat16`) so the score matrix can be emitted in
  `float32` regardless of the activation type, exactly as in GQA.
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
`PagedAttention`**, expressed through three MLA-specific attributes, and not as a new operator. This is
the same conclusion FlashMLA, vLLM's MLA backend, and SGLang reached: absorbed MLA decode is MQA with
a wide head and a shared K/V buffer.

### 12.3 Schema additions

MLA is selected **explicitly** by `kv_cache_layout="LATENT"`, never inferred from input presence.

| Addition | Meaning |
|---|---|
| Attribute `kv_cache_layout` (STRING, default `"SEPARATE"`) | `"LATENT"` selects absorbed MLA: one physical cache, V aliasing K. |
| Attribute `v_head_size` (INT, default `0`) | Head width of V and of each output head. `0` means "same as `head_size`". A value differing from `head_size` is legal **only** in `"LATENT"` mode. |
| Attribute `rotary_offset` (INT, default `0`) | First channel within `head_size` covered by RoPE (§12.5). |
| Input 4 `value_cache` becomes `Optional` | Absent in `"LATENT"`, where V is the leading `v_head_size` channels of `key_cache`. Still required in `"SEPARATE"`. |
| Input 2 `value` absent while `key` is present becomes legal | Only under `"LATENT"` (§4.6). In `SEPARATE`, the shipped rule stands: `key` **and** `value` absent means packed QKV. |
| Output 0 shape generalized to `(token_count, num_heads * effective_v_head_size)` | Identical to today whenever `v_head_size == 0`. |

All six are backward compatible: existing models set none of the attributes, supply `value_cache`,
and get byte-identical behavior.

**Why `v_head_size != head_size` is confined to `LATENT`.** Asymmetric K/V widths in `SEPARATE` mode
would be a second, independent feature: it needs a `value_cache` whose last dimension differs from
`key_cache`, which the shipped implementation contradicts (it builds `value_cache_out_shape[3]` from
`head_size` unconditionally and validates the two cache shapes as identical), and no available
backend supports it — ORT's Flash and CUTLASS fMHA both require `v_head_size == head_size`. Allowing
it in the schema without backend coverage would only create a validation surface with nothing behind
it. `effective_v_head_size` is therefore defined as:

```text
effective_v_head_size = (kv_cache_layout == "LATENT" && v_head_size != 0) ? v_head_size : head_size
```

and `v_head_size != 0 && v_head_size != head_size` in `SEPARATE` mode is `INVALID_ARGUMENT`.

A consequence worth stating explicitly, because it is why the rest of this document writes
`head_size` rather than `v_head_size` almost everywhere: the only tensors whose last dimension can
ever be `effective_v_head_size` are **output 0** and the **V view of `key_cache`**. `value`,
`value_cache`, and `v_scale` are all absent in `"LATENT"` (§4.6, §12.9), and in `"SEPARATE"` their
width is `head_size` by the rule above. Asymmetric K/V widths for a real `value_cache` arrive only
with the `"KV_CONCAT"` layout deferred in §21.

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
| `kv_cache_layout` | `"LATENT"` |
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
| Quantized cache (§8) | **Supported** | An FP8 latent cache is standard in DeepSeek deployments. `PER_CHANNEL` scale shape becomes `(1, 1, head_size)` since `kv_num_heads == 1`. Because V *is* K, the same bytes are written once with `k_scale` and read back as both, so `k_scale` alone describes the cache and `v_scale` / `v_quant_type` must be unset — a second scale for the same bytes could only disagree. A `PER_CHANNEL` V dequant therefore indexes the scale with the `head_size` stride, reading only its leading `v_head_size` entries. |
| RoPE | **Supported** via `rotary_offset` (§12.5) | |
| `softcap` | Allowed, unused by DeepSeek | |
| Sliding window (§9) | Allowed but untested | No MLA model uses it; semantics are well defined (window is over positions, not channels). |
| `head_sink` (§6) | **Rejected** | No MLA model uses sinks. The LSE epilogue is valid math here, but shipping an untested combination invites silent errors. Revisit if a model needs it. |
| QK-Norm (§7) | **Rejected** | DeepSeek's `q_a_layernorm` / `kv_a_layernorm` act on the *latent* projections in the graph, before absorption. A `head_size`-wide RMSNorm in absorbed space is a different operation; accepting it would let an exporter produce silently wrong math. |
| `attention_bias` / `output_qk` (§10, §11) | Supported on the unfused path | `output_qk` shape is unchanged: `(num_heads, token_count, max_context_len)`. |

### 12.10 Validation

- `v_head_size != 0 && v_head_size != head_size` requires `kv_cache_layout == "LATENT"`; otherwise
  `INVALID_ARGUMENT`. In `"SEPARATE"` mode `effective_v_head_size == head_size` always.
- `kv_cache_layout == "LATENT"` requires `key` present and `value` and `value_cache` absent (§4.6).
- `v_head_size ∈ [1, head_size]` when set; `0` means "equal to `head_size`".
- `v_head_size != head_size` requires an explicit `scale` attribute (§12.6).
- Initially require `kv_num_heads == 1`; widen only alongside a backend and tests for grouped latent
  heads. `kv_num_heads` is `1` for DeepSeek, and any divisor of `num_heads` is accepted once a
  backend exists.
- `rotary_offset >= 0`, `rotary_offset % 8 == 0`, `rotary_offset + rotary_dim <= head_size`.
- In `"LATENT"` mode `head_size` may exceed 256, but the selected backend must accept it; otherwise
  return `INVALID_ARGUMENT` naming the backend and the supported head widths.
- `head_sink` or QK-Norm weights combined with `"LATENT"` ⇒ `INVALID_ARGUMENT` (§12.9).
- `v_scale` and a non-`"NONE"` `v_quant_type` combined with `"LATENT"` ⇒ `INVALID_ARGUMENT`: there is
  one physical cache and `k_scale` already describes it (§12.9).
- Dispatch only to an MLA-capable backend or the unfused reference; never silently ignore an input.

## 13. Kernel Dispatch and Backend Plan

Target dispatch order once all phases land, first eligible wins:

| Priority | Backend | Eligible when |
|---|---|---|
| 0 | **MLA backend** (new, §12.7) | `kv_cache_layout == "LATENT"` — FlashMLA / FlashInfer MLA / TRT-LLM MLA where available, unfused MLA reference otherwise |
| 1 | **Paged decode kernel** (new, Phase 4) | Decode-shaped batch per the static shape test in §4.7 (`token_count <= batch_size`); non-quantized or `PER_TENSOR`/`PER_CHANNEL` INT8/FP8; sliding window and `head_sink` supported |
| 2 | **FlashAttention varlen** | FP16/BF16, SM80+, non-quantized cache (or quantized via dequant-gather); supports sliding window, softcap, packed QKV, `head_sink` via LSE epilogue |
| 3 | **Memory-Efficient Attention (CUTLASS fMHA)** | Fallback for supported combinations and pre-SM80 |
| 4 | **Unfused** | Last resort — arbitrary `head_size`, `attention_bias`, `output_qk` |

Implementation status: priority 0 is currently served by `PagedLatentAttentionKernel`, the unfused
MLA reference in `paged_attention_impl.cu`. It is selected unconditionally when
`kv_cache_layout == "LATENT"` — the other three backends cannot express `v_head_size != head_size`
over a single aliased cache, so there is no eligibility test to run. One CTA owns one
(query token, query head) pair and streams the KV range in tiles with an online-softmax state,
which makes prefill, chunked prefill and decode take the same path. A fused MLA backend (P5) will
slot in ahead of it under the same priority.

Backend selection must depend only on values that are constant for a captured CUDA graph (§4.7):
shapes, attributes, and type constraints. It must never depend on a device-resident sequence length.

Feature × backend matrix (target state):

| Feature | Paged decode | Flash varlen | MEA | Unfused |
|---|---|---|---|---|
| Sliding window | Yes | Yes | Yes | Yes |
| `softcap` | Planned | Yes | Yes | Yes |
| `head_sink` | Yes (native) | Yes (LSE epilogue) | After fMHA LSE | Yes |
| QK-Norm | Yes (prologue) | Yes (prologue) | Yes (prologue) | Yes |
| Quantized cache | Yes (in-kernel) | Via dequant-gather | Via dequant-gather | Via dequant-gather |
| `query_positions` (§4.8) | Backend work required | Fallback unless legacy-equivalent | Backend work required | Yes |
| `attention_bias` | No | No | No initially | Yes |
| `output_qk` | No | No | Yes | Yes |
| `slot_mapping` | Yes (write path — backend independent) | Yes | Yes | Yes |
| `LATENT` MLA | No | No | No | Yes — plus the dedicated MLA backend (§12.7) |

Feature acceptance and backend eligibility are separate concerns. Schema validation (§15) decides
whether the *request* is well formed; dispatch then selects a backend that implements the requested
combination. An unsupported combination must produce an `INVALID_ARGUMENT` naming both the feature
and the backend limitation — never a silently ignored input or attribute.

QK-Norm, RoPE, offset RoPE, packed-QKV unpacking, quantized writes, and `slot_mapping` all live in
the **prologue** and are therefore backend independent. Only the attention math itself varies by
backend.

The selected backend must be reported through `AttentionKernelDebugInfo` (`SdpaKernel=...`) exactly
as GQA does, so that `ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO=1` works uniformly across both ops.

> **Implementation note (P5, implemented).** Priority 1 is in place and reports
> `SdpaKernel=DECODER_ATTENTION`. Actual eligibility is broader than the table: any `head_size` whose
> working set fits `sharedMemPerBlock` (roughly `2 * head_size + 256` floats plus 512 B, so every
> head size the op supports on current hardware), any `block_size`, any GQA ratio, `softcap`, and
> arbitrary ragged query lengths are all supported. It is only *preferred* over FlashAttention when
> the cache is quantized or FlashAttention is ineligible, since Flash's tensor-core decode path is
> faster on an unquantized cache. Priority 0 (MLA) is still unimplemented.
>
> The gate is the static shape test of §4.7 (`token_count <= batch_size`), so no device-resident
> length reaches the host and the D→H sync is gone. The XQA fast path inside this backend keeps a
> stricter gate — it needs `token_count == batch_size` *and* `max_query_len == 1`, the latter from
> `attention_metadata` or, failing that, from the readback — because its output layout is one row
> per batch index rather than per query token.

## 14. Shared Code with GroupQueryAttention

Feature drift between the two ops is the principal long-term risk of keeping them separate. The
mitigation is structural, not procedural.

1. **Common parameter base.** `PagedAttentionParameters` and `GroupQueryAttentionParameters` already
   derive from `AttentionParameters` in `contrib_ops/cpu/bert/attention_parameters.h`. Move the
  fields that are genuinely shared — `local_window_size`, `softcap`,
  `qk_norm_epsilon`, `k_quant_type`, `v_quant_type`,
  `rotary_interleaved` —
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
- `k_scale` / `v_scale`: FP32; `(1,)` for `PER_TENSOR`, `(kv_num_heads, 1, head_size)` for
  `PER_CHANNEL` (both K and V — `v_scale` only exists alongside a `value_cache`, so its last
  dimension is always `head_size`); present iff the corresponding quant type is not `NONE`.
- `T_CACHE != T` iff a quant type is not `NONE`.
- `k_cache_dtype` and `v_cache_dtype` must be `""` or name the cache tensor's own element type:
  every logical element type this operator stores is expressible as an ONNX element type. The
  reserved sub-byte values are rejected until a `uint8` packed cache exists. FP8 availability is
  controlled by `onnxruntime_USE_FP8_KV_CACHE`, without an additional runtime architecture gate.
- `attention_metadata`: rank 1, `dim0 == 2`, `int32`, CPU-resident; entries `>= 0`; each non-zero
  entry is clamped to its static limit before use and may only size launch dimensions and workspace
  (§4.7). It must never enter a mask comparison. Each value is a trusted upper bound for every step
  served by the node or captured graph.
- `query_positions`: rank 1, `dim0 == token_count`, `int32`, entries `>= 0`.
- `attention_bias`: rank 4; `dim0 ∈ {1, batch_size}`, `dim1 ∈ {1, num_heads}`,
  `dim2 == query_length_capacity`, and `dim3 == context_length_capacity`; both capacities must cover
  every actual length served by the node or captured graph. Indexing uses the sequence-local query
  offset and logical KV position, matching GQA (§10.2). Device code bounds-checks both indices;
  fused backends reject the input and the initial implementation uses the unfused fallback.
- `qk_output != 0` iff the node has 4 outputs; `output_qk` rejected on Flash.
- `kv_cache_layout` must be one of its documented values, and the input presence pattern must match
  a row in [§4.6](#46-input-mode-state-machine).
- MLA (§12.10): `kv_cache_layout == "LATENT"` requires `key` present, `value` and `value_cache`
  absent, an explicit `scale`, and no `head_sink` or QK-Norm weights.
  `v_head_size ∈ [1, head_size]`; `v_head_size != head_size` outside `"LATENT"` is
  `INVALID_ARGUMENT`. `rotary_offset % 8 == 0` and `rotary_offset + rotary_dim <= head_size`.
- Backend-incompatible feature combinations must be reported at `Compute` entry with the reason,
  never silently ignored.

**Trust boundary.** `attention_metadata` is host-supplied and cannot be checked against the device
tensors without reintroducing the synchronization it exists to remove. It is therefore a *trusted
bound*: an under-sized value violates the contract and may omit valid attention work. Independently,
the kernel must keep every device read within the static tensor capacities even when the bound is
invalid. No other host-supplied value is permitted to bound a launch or workspace extent.

## 16. Shape Inference and Tooling

- `PagedAttentionTypeAndShapeInference` in `onnxruntime/core/graph/contrib_ops/bert_defs.cc`:
  - Output 0 dim 1 becomes `num_heads * effective_v_head_size`. **Only compute the product when
    `effective_v_head_size != head_size`** (i.e. in `"LATENT"` mode). In every `SEPARATE`-mode model
    keep the shipped `propagateShapeFromInputToOutput(ctx, 0, 0)`, which needs no numeric dimension;
    deriving `head_size = query.shape[1] / num_heads` unconditionally would make a model with a
    symbolic hidden dimension — which infers fine today — start failing. Both the unpacked and
    packed-QKV branches must apply the generalization when it does apply, otherwise every MLA graph
    gets a wrong — and silently propagated — output width.
  - Read `kv_cache_layout` before branching on Q/K/V presence. `LATENT` uses the unpacked-query
    formula even though `value` is absent; only `SEPARATE` with both K and V absent uses the packed
    formula. Unknown symbolic dimensions must stay symbolic.
  - Outputs 1/2 must propagate **type from inputs 3/4** (not from input 0) once `T_CACHE` can differ
    from `T`. The current code propagates elem type from input 0 to outputs 1/2 first and then from
    3/4 — the first propagation becomes wrong under quantization and must be removed. Element-type
    propagation must run *before* shape propagation for the same output, or ONNX reports
    `Mismatch between inferred and declared type`.
  - `value_cache` (input 4) is now optional: guard `getInputShape(ctx, 4)` and the output-2
    propagation on `ctx.hasInput(4)` before any write.
  - The shipped `getNumOutputs() > 1 ⇒ getNumOutputs() == 3` rule must be relaxed to admit a
    four-entry output list with empty names at unused optional cache positions (§4.4), while keeping
    the both-or-neither rule for `SEPARATE` mode.
  - Output 3 (`output_qk`) is written only when `ctx.getNumOutputs() > 3`, guarded before any write,
    consistent with the contrib-op shape-inference memory-safety rules.
- Kernel definition: register `.Alias(3, 1).Alias(4, 2)` and `.InputMemoryType(OrtMemTypeCPUInput, 16)`
  (§4.4, §4.7).
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
- Rolling sliding-window cache (§9.3): a run holding only `ceil(W / block_size) + 1` blocks per
  sequence, recycled through `slot_mapping` with the evicted `block_table` entries set to `-1`,
  bit-matches the full-cache run at every decode step.

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
- **V-aliases-K.** With `value_cache` absent, verify that the leading `v_head_size` channels of
  `key_cache` supply V. Any present `value_cache` — including the same tensor as `key_cache` — must
  be rejected in `LATENT` mode.
- **MLA × paging.** Shuffled block tables, `slot_mapping` with `-1`, and an FP8 latent cache, each
  combined with MLA.
- **Rejected combinations.** MLA + `head_sink` and MLA + QK-Norm must fail with the documented
  message, not silently compute something.

### 17.4 Reference implementations

Extend `onnxruntime/test/python/transformers/test_paged_attention_cuda.py` with a PyTorch reference
covering attention sinks, QK-Norm (RMSNorm-before-RoPE), per-channel dequantization,
windowing, and MLA absorption — reusing the GQA test helpers rather than duplicating them.

### 17.5 Negative tests

One test per validation rule in [§15](#15-validation-rules), asserting the error *message*, not just
failure — so that backend-incompatible combinations cannot regress into silent wrong results.

### 17.6 Compatibility and CUDA graph regression

These two guard the contract itself rather than a feature.

- **Opset-1 compatibility.** Serialize a model using *only* the shipped contract — inputs 0–9,
  outputs 0–2, the seven original attributes — and run it against the extended kernel **without
  rewriting the graph**. Assert byte-identical output against a run on the pre-extension build. This
  must be re-run in every phase (§19), not just the phase that adds an input.
- **CUDA graph decode replay.** Capture a decode graph, then replay it for `N` steps while the KV
  length grows, and assert the result matches an uncaptured run step for step. This is the test that
  would have caught the frozen-`max_kv_len` failure mode in §4.7: a naive implementation passes step
  1 and diverges from step 2 onward, so the test must check **every** step, not just the last.
- **Metadata-bound invariance.** The same decode sequence must produce identical results with
  `attention_metadata` absent, with the tightest replay-wide valid bounds, and with deliberately
  loose (over-large) bounds.
  Any difference means a host value affected device-side logical attention bounds rather than only
  launch or workspace capacity (§4.7).
- **Under-sized bound safety.** A deliberately too-small `max_kv_len_bound` violates the contract,
  so its numerical result is unspecified, but it must not read out of bounds; run it under
  `compute-sanitizer`.

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
4. **Per-step D→H synchronization — and it blocks CUDA graph capture.** ~~`max_query_len` (and
   `total_kv_tokens` for MEA) are obtained via `cudaStreamSynchronize` every step, once per layer.
   This is not only a throughput bug for the op's primary use case: `cudaStreamSynchronize` on a
   capturing stream is **illegal**, so the operator cannot be captured into a CUDA graph at all —
   precisely the optimization decode needs most.~~ **Fixed.** Dispatch now derives from static
   shapes, extents from the static capacity bound `block_table.shape[1] * block_size` or the
   optional `attention_metadata` bounds, and every per-step quantity from device memory, as
   specified in [§4.7](#47-cuda-graph-contract-and-attention_metadata). The sync survives only as a
   prefill-side fallback for a dense gather or for XQA when no metadata is supplied, neither of
   which a capturable step can reach.

## 19. Phasing

| Phase | Contents | Schema delta |
|---|---|---|
| **P0 — Foundation** | §18 defect fixes; `.Alias(3,1).Alias(4,2)` on the kernel def (§4.4); sliding-window semantic verification and GQA parity harness (§17.1) | none |
| **P1 — Paging primitives** | `slot_mapping` (§5); sliding-window block pruning (§9.2); rolling sliding-window cache (§9.3) | input 10 |
| **P2 — Model coverage** | `head_sink` via LSE epilogue (§6); fused QK-Norm (§7) | inputs 11–13, attr `qk_norm_epsilon` |
| **P3 — Memory** | Quantized cache INT8/FP8, `PER_TENSOR` + `PER_CHANNEL`, dequant-on-gather read path (§8) | `T_CACHE`, `T_KV_SCALE`, inputs 14–15, 4 attrs |
| **P4 — MLA (correctness)** | `kv_cache_layout="LATENT"`, `v_head_size`, `rotary_offset`, V-aliases-K, optional `value_cache`, unfused MLA reference kernel, absorbed↔non-absorbed equivalence tests (§12) | attrs `kv_cache_layout`, `v_head_size`, `rotary_offset`; input 4 optional |
| **P5 — Performance** | Paged decode kernel with in-kernel dequant; fused MLA backend (FlashMLA / FlashInfer MLA, §12.7); `softcap` on decode; **remove the D→H sync and make the op CUDA-graph-capturable (§4.7)**; optional `attention_metadata` replay-wide bounds | input 16 |
| **P6 — Completeness** | `query_positions` (§4.8); `attention_bias` (§10); `output_qk` (§11) | inputs 17–18, output 3, attr `qk_output` |
| **Later** | INT4 cache; MLA quantized latent cache tuning; non-CUDA EPs | — |

Status: P0–P4 are implemented, except the `.Alias` registration. P5 is partially
implemented — the paged decode kernel with in-kernel dequantization (including `softcap`, sliding
window and `head_sink`) has landed (§8.5, §13), as has the sync removal and CUDA graph capture
(§4.7); the fused MLA backend has not.

Every phase must re-run the old-model tests with all new inputs absent, in addition to the
feature-specific tests. The compatibility regression test should serialize a model using only the
shipped opset-1 contract and run it against the extended kernel **without rewriting the graph**
(§4.2).

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

1. ~~**Host-scalar inputs for `max_query_len` / `total_kv_tokens` (§18.4).** Adding them as optional
   inputs removes a per-step sync but leaks scheduler state into the graph. Is that acceptable?~~
   **Resolved in [§4.7](#47-cuda-graph-contract-and-attention_metadata):** no host input is needed to
   remove the sync, and an *exact* one would be actively unsafe — under CUDA graph replay no kernel
   `Compute` runs, so a host value is frozen at capture while `max_kv_len` grows every step. Dispatch
   comes from static shapes, extents from the static capacity bound, and every per-step quantity from
    device memory. `attention_metadata` survives only as optional replay-wide bounds: valid bounds
    preserve correctness, while an under-sized bound is a producer contract violation.
2. **`block_table == -1` semantics (§9.2).** Confirm "unmapped, masked out" rather than "invalid,
   error" — the former is required for sliding-window block eviction.
3. **Non-CUDA EPs.** Is `PagedAttention` a server-GPU-only op (CUDA, ROCm, TensorRT), or is a CPU
   reference implementation required? A CPU reference has real value as a test oracle even if it is
   never used in production. Stub kernels returning `NOT_IMPLEMENTED` are not an acceptable middle
   ground (§3.4).
4. **ORT-GenAI commitment.** Does the continuous-batching path in `onnxruntime-genai` commit to
   emitting `PagedAttention` (as opposed to a `block_table`-extended GQA)? This determines the
   priority of everything above.
5. **Ownership of the parity matrix (§13).** Who keeps the feature × backend table current, and is
   the KV-accessor refactor (§14.3) in scope for P2 or a follow-up?
6. **Which fused MLA backend (§12.7)?** FlashMLA is the closest fit but is SM90-only and adds a
   third-party dependency; FlashInfer MLA covers SM80+; TRT-LLM has its own. The choice determines
   the hardware matrix ORT can serve DeepSeek on and should be made before P5 starts.
7. **Does MLA need a non-absorbed *paged* path (§12.8)?** The proposal handles prefill-with-prefix by
   absorbing. If a workload shows the absorbed prefill FLOP inflation to be material, an in-op
   decompression path would require passing `W_UK` / `W_UV` into the operator — which this design
   deliberately avoids. Needs a measurement before it is reconsidered.
8. **MLA + quantized latent cache (§12.9).** FP8 on a 576-wide latent shared by 128 query heads has
   a different error profile from FP8 on a per-head cache. Accuracy validation is required before
   the combination is recommended, even though the schema supports it on day one.

## 21. Deferred Breaking Contract

Status: **deferred, not adopted.** [§4](#4-schema--the-compatible-contract) is the only normative
contract. This section records the breaking cache-format ideas that were evaluated, states why each
was rejected for opset 1, and preserves the analysis so it does not have to be redone if a merged or
sub-byte cache ever becomes a concrete requirement.

### 21.1 Why they are deferred

The original argument was that three of these changes cannot be expressed additively at any later
date, so they must land before the operator acquires an external emitter or never:

| Change | Why it cannot be additive |
|---|---|
| Merge `key_cache` + `value_cache` → `kv_cache` | Changes input arity and the aliasing contract |
| `kv_cache_out` becomes required | Changes output arity |
| Positions no longer implicit in `past_seqlens[b] + j` | Changes the meaning of an existing graph |

The first two hold. The third does not: [§4.8](#48-query_positions) shows that an optional
`query_positions` input expresses explicit positions additively, because absence keeps the legacy
derivation. That removes the only *model-coverage* item from the "now or never" list, and what
remains is physical cache-format cleanup with no feature behind it.

Weighed against that, the operator is already serialized as `com.microsoft::PagedAttention` opset 1,
and P0–P3 plus half of P5 are implemented and tested against that contract. Breaking it buys
expressibility for formats no ORT model uses today, at the cost of invalidating every already
serialized graph and every test. The decision is therefore:

> Treat the separate-cache representation as **permanent** for `com.microsoft::PagedAttention`
> opset 1. If a merged or sub-byte cache becomes a real requirement, introduce a separately versioned
> schema or a new operator name with a migration tool — do not change the meaning of inputs, outputs
> or attributes in place.

The complete deferred list:

- one merged `kv_cache` input containing K and V (§21.2);
- one required functional `kv_cache_out` instead of two optional aliasing outputs;
- removal of `kv_num_heads` in favor of `kv_cache.shape[2]`;
- quantization granularity inferred from scale shape, and zero points (§21.3);
- sub-byte logical types stored in `uint8` tensors — the `k_cache_dtype` / `v_cache_dtype` attributes
  that name them are adopted in §4.5, but no backend decodes a packed cache yet (§21.4);
- inline scales or zero points packed into cache rows (§21.3, note);
- a physical `HND` cache layout (§21.6);
- renaming `local_window_size` to `window_size_left` / `window_size_right`, and the
  lookahead window (`window_size_right > 0`) that tree speculative decoding needs;
- a tree-attention ancestry mask, which `query_positions` deliberately does **not** provide (§4.8).

Everything else that was in the v2 proposal — `attention_metadata`, `query_positions`,
`kv_cache_layout`, `v_head_size`, `rotary_offset`, explicit quantization attributes — is expressible
additively and has been folded into §4.

---

### 21.2 Single `kv_cache` with a layout attribute

```
kv_cache : (num_blocks, block_size, kv_num_heads, kv_pack_dim)
```

| `kv_cache_layout` | `kv_num_heads` | `kv_pack_dim` | K slice | V slice |
|---|---|---|---|---|
| `"KV_CONCAT"` | `H` | `head_size + v_head_size` | `[0, head_size)` | `[head_size, +v_head_size)` |
| `"LATENT"` (MLA) | `1` | `kv_lora_rank + qk_rope_head_dim` (576) | `[0, kv_pack_dim)` | **`[0, v_head_size)`** |

`"KV_CONCAT"` covers MHA, GQA, and the asymmetric-head-size case (`v_head_size != head_size`) in one
rule. `"LATENT"` is the MLA case where the V view *overlaps* the K view rather than following it.

This is the layout vLLM's FlashAttention, FlashInfer and Triton backends converged on
(`(num_blocks, num_kv_heads, block_size, 2 * head_size)` in their head-major ordering), and
TensorRT-LLM parameterizes the same axis as `kv_factor ∈ {1, 2}`
(`kv_cache_manager_v2.py`, `CacheType::kSELFKONLY`). A hard-coded `2` axis — the shape this document
previously implied — cannot express either MLA or asymmetric K/V and is rejected.

**Derived, not attributes.** `k_head_size` is always `query.shape[-1] / num_heads`, including for
MLA where `query` is the 576-wide nope‖rope concatenation. `v_head_size` is derivable as
`kv_pack_dim - k_head_size` under `"KV_CONCAT"` but **must** be given under `"LATENT"` (it is
`kv_lora_rank`; 512 cannot be recovered from `kv_pack_dim = 576` and `k_head_size = 576`).

**No explicit `k_offset` / `v_offset`.** They add nothing a named layout does not already determine,
and they cannot express the part that actually matters: the *write* path differs between the two
layouts. Under `"KV_CONCAT"` both `key` and `value` are supplied and scattered into disjoint regions
of the row. Under `"LATENT"` there is no separate V to write — `value` is **absent** and the V region
aliases bytes already written by the K store. With raw offsets an implementation would have to infer
"the ranges overlap, therefore skip the V write," which is implicit, easy to get wrong, and gives
shape inference no way to reject `value` being present. Named layouts also require no range/overlap
validation, and can grow new members later (e.g. an alignment-padded or scale-interleaved variant)
without having locked the operator into a byte-layout contract.

This makes §12.2's "V aliases K" a layout selection rather than a schema fork. §4 achieves the same
effect additively with `kv_cache_layout="LATENT"` over the *separate* caches, at the cost of not
covering the asymmetric-K/V and packed-quant cases — which is why those are restricted to `LATENT`
in §12.3.

**Performance is not the motivation.** There is no meaningful kernel-level gain from merging the two
pools for non-MLA models — vLLM splits the merged tensor back into two strided views before every
kernel call. The merge buys *expressibility* (MLA, asymmetric K/V, packed quant formats) and one
fewer aliased graph edge. The KV-transfer and allocator-packing benefits that motivate it in vLLM
and TensorRT-LLM do not apply to ORT, which does not do disaggregated prefill and does not own the
block pool. That thin margin is what makes the merge deferrable.

### 21.3 Quantization: granularity from scale shape

`k_quant_type` / `v_quant_type` would be removed; granularity read off the scale tensor instead:

| `k_scale` shape | Granularity |
|---|---|
| absent | not quantized |
| `()` or `(1,)` | per-tensor |
| `(kv_num_heads, head_size)` | per-channel |
| `(num_blocks, block_size, kv_num_heads)` | per-token |
| `(num_blocks, kv_num_heads)` | per-block |

One source of truth instead of two that can disagree, and it extends to per-token/per-block without
new enum values. This also drops the vestigial middle `1` in v1's `(kv_num_heads, 1, head_size)`.
K and V may use different granularities independently.

`k_zero_point` / `v_zero_point` are new and optional, same shape as the corresponding scale, for
asymmetric integer quantization. Absent ⇒ symmetric (the current behavior). These are the inputs an
**unsigned** logical cache type would need: §8.3.1 keeps `k_cache_dtype` / `v_cache_dtype` restricted
to signed, zero-symmetric types precisely because they do not exist yet, and §8.3.1 also shows that
adding them is not free — a non-zero zero point breaks the decode kernel's scale folding and
introduces correction terms in both the QK and PV products.

> Note: vLLM's DeepSeek V3.2/V4 formats store scales **inline in the cache row** rather than in a
> separate tensor (`fp8_ds_mla` = 512 NoPE + 16 scale + 128 RoPE = 656 B; DeepSeek-V4 = 448 + 128 +
> 8 = 584 B), so one page read fetches data and scale together. The shape-derived rule above covers
> the separate-tensor case only. If an inline format is ever needed it should be added as a
> `kv_cache_layout` member, not by overloading the scale inputs.

### 21.4 `k_cache_dtype` / `v_cache_dtype` for sub-byte caches

**The attributes themselves are adopted in §4.5**; only their sub-byte *values* are deferred, because
no backend decodes a packed cache yet. They were adopted rather than deferred because the obvious
alternative — a `k_cache_bit_width` / `v_cache_bit_width` pair — is redundant against the cache
tensor's element type for every format that exists today and still insufficient for the format it
was meant to describe.

For `int8` and `float8e4m3fn` each cache tensor's own element type is the logical type and its
corresponding cache-dtype attribute stays `""`. Sub-byte needs more:

- ORT's CUDA EP has no usable 4-bit tensor element type, so storage must be `uint8`.
- `kv_pack_dim` then counts **storage bytes**, and the logical head width is unrecoverable.
- Bit width alone is insufficient: `int4` and `float4e2m1` are both 4 bits with entirely different
  decode math. Keeping K and V independent supports formats where they use different precisions.

| `k_cache_dtype` / `v_cache_dtype` | Storage elem type | Packed logical width |
|---|---|---|
| `""` | corresponding cache tensor type | unchanged |
| `"float16"`, `"bfloat16"`, `"int8"`, `"float8e4m3fn"` | the same type, named explicitly | unchanged |
| `"int4"`, `"float4e2m1"` | `uint8` | logical width / 2 |
| `"int2"` | `uint8` | logical width / 4 |

where `E = head_size + v_head_size` under `"KV_CONCAT"`, or `kv_pack_dim`'s logical width under
`"LATENT"`. Packing order must be specified or implementations will diverge: **logical element `2i`
occupies the low-order bits of byte `i`**, element `2i+1` the high-order bits. The storage type is
`uint8` for all of these, but the *logical* values stay signed: a 4-bit code is written as `q + 8`
and read back as `nibble - 8`, matching MLAS's `S4` packing, so the dequantization remains
`q_signed * scale` with no zero point (§8.3.1). `"uint4"` / `"uint2"` are not members and cannot be
until `k_zero_point` / `v_zero_point` (§21.3) exist.

### 21.5 Tightened / clarified

- **`kv_cache_out` is required.** As an optional output it is a dead-code-elimination hazard: the
  cache mutation is a side effect, so a graph in which the output is unconsumed is legal but the node
  is not removable. Making it required removes the ambiguity. §4 keeps the optional outputs and
  instead recommends registering the alias on the kernel def (§4.4).
- **`window_size_left` / `window_size_right`** replace `local_window_size`, matching FlashAttention's
  own parameter names and removing the current off-by-one ambiguity (`local_window_size - 1` is what
  actually reaches the kernel). `window_size_right = 0` expresses causal; `> 0` expresses the
  bidirectional-lookahead window that speculative decoding needs. §4.5 keeps `local_window_size`
  with its established "W positions including the current token" semantic; the lookahead window is
  deferred with the rename.
- **`block_table == -1`** means unmapped and fully masked, not an error (§9.2, §20.3).
  *Adopted in §4.3, not breaking.*

### 21.6 `kv_layout`

`"NHD"` (default) is `(num_blocks, block_size, kv_num_heads, kv_pack_dim)`; `"HND"` is
`(num_blocks, kv_num_heads, block_size, kv_pack_dim)`.

NHD is the default and the only value implemented, because ORT's Flash backend requires it:
`flash_api.h` documents the paged `kcache`/`vcache` as
`num_blocks x page_block_size x num_heads_k x head_size`, and `paged_attention_impl.cu` hands the
cache straight to `flash::mha_varlen_fwd`. NHD also keeps the per-channel scale index trivial (the
innermost dimension is the channel).

HND has real but modest advantages on the read path — a head's page tile is one contiguous run,
which helps alignment/vectorization when `head_size * sizeof(T)` is not a multiple of 16, and suits
1-D TMA bulk copies. It is worse on the write path, where NHD gives one contiguous store per token.
Some kernels require one or the other (vLLM's AITER assembly path needs independently contiguous K
and V; trtllm-gen backends report `get_required_kv_cache_layout() == "HND"`), which is why the
attribute exists at all. **For MLA the two are identical**, since `kv_num_heads == 1`.

### 21.7 Migration sketch, if a versioned successor is ever needed

| opset 1 (§4) | Successor |
|---|---|
| `key_cache`, `value_cache` (inputs 3, 4) | single `kv_cache` (input 3) with `kv_pack_dim = 2 * head_size` |
| `key_cache_out`, `value_cache_out` | single required `kv_cache_out` |
| `kv_num_heads` attr | drop — read `kv_cache.shape[2]` |
| `local_window_size = w` | `window_size_left = w`, `window_size_right = 0` |
| `k_quant_type` / `v_quant_type` | drop — infer from `k_scale` / `v_scale` shape |
| `k_scale` shape `(kv_num_heads, 1, head_size)` | `(kv_num_heads, head_size)` |

Implementation impact is concentrated in `ReshapeAndCache` (one packed row per token instead of two
scattered writes), the cache-read stride arithmetic in the P5 decode kernel (innermost stride becomes
`kv_pack_dim`, and the V base gains `k_head_size`), and the two K/V pointer derivations handed to
Flash and MEA, which become strided views over one tensor.

Every row above is mechanical and scriptable, which is the other reason none of it is urgent: a
migration tool over serialized graphs is cheap compared with breaking a shipped contract in place.

### 21.8 Disposition

| Change | Compatible extension? | Disposition |
|---|---|---|
| §21.2 merged `kv_cache` | **No** | Deferred to a versioned successor |
| §21.5 required `kv_cache_out` | **No** | Deferred; §4.4 registers the alias instead |
| §21.3 scale-shape granularity, zero points | Yes | Deferred — explicit attributes in §4.5 are preferred while only two granularities exist |
| §21.4 `k_cache_dtype` / `v_cache_dtype` | Yes | **Adopted** — §4.5; only the sub-byte *values* wait for a packed-cache backend |
| §21.6 `kv_layout` | Yes | Deferred until a backend requires `HND` |
| §21.5 `window_size_*` rename | Yes, with deprecated aliases | Deferred with the lookahead window |
| `attention_metadata` | Yes | **Adopted, redesigned** — §4.7 |
| `query_positions` | Yes | **Adopted** — §4.8 |
| `kv_cache_layout`, `v_head_size`, `rotary_offset` | Yes | **Adopted** — §4.5, §12 |
