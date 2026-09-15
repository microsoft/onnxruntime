# SparseAttentionIndexer — Operator Documentation

This document describes the `com.microsoft::SparseAttentionIndexer` contrib operator: the two
indexer policies it implements, the schema and state contract, the CUDA kernel pipeline, and the
known limitations and performance follow-ups.

The operator answers a single question for every query token: *which* keys is the following
attention operator allowed to read. It does not compute attention itself. Two policies are
supported, selected by the `policy_mode` attribute:

| `policy_mode` | Reference implementation | Selection granularity |
|---|---|---|
| `qsa` | `Qwen4ExpTextQSAIndexer` | token indices, grouped in blocks of `compress_ratio` |
| `csa` | `DeepseekV4Indexer` / `DeepseekV4IndexerScorer` | compressed-entry indices |

Source:
[bert_defs.cc](../../../onnxruntime/core/graph/contrib_ops/bert_defs.cc) (schema),
[sparse_attention_indexer_common.h](../../../onnxruntime/contrib_ops/cpu/sparse/sparse_attention_indexer_common.h)
(slot map, capacity and window plan shared by the schema, the kernel and the tests),
[sparse_attention_indexer.cc](../../../onnxruntime/contrib_ops/cuda/sparse/sparse_attention_indexer.cc),
[sparse_attention_indexer_impl.cu](../../../onnxruntime/contrib_ops/cuda/sparse/sparse_attention_indexer_impl.cu).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Operator Schema](#2-operator-schema)
3. [State Contract](#3-state-contract)
4. [Policy `qsa`](#4-policy-qsa)
5. [Policy `csa`](#5-policy-csa)
6. [Rotary Embeddings](#6-rotary-embeddings)
7. [CUDA Kernel Pipeline](#7-cuda-kernel-pipeline)
8. [Numerics and Determinism](#8-numerics-and-determinism)
9. [Validation Rules](#9-validation-rules)
10. [Testing](#10-testing)
11. [Known Limitations and Performance Follow-ups](#11-known-limitations-and-performance-follow-ups)

---

## 1. Overview

Sparse-attention decoders run a small, cheap "indexer" attention next to the real attention. The
indexer has its own low-rank query/key projections, its own RMSNorm and its own rotary embedding,
and its only product is a set of indices. The real attention then reads just those keys.

`SparseAttentionIndexer` implements that stage as one operator with three properties that matter
for a runtime:

- **Fixed output capacity.** `selected_indices` always has shape
  `(batch_size, sequence_length, capacity)` where `capacity` is a pure function of the attributes
  (`token_budget + compress_ratio - 1` for `qsa`, `index_topk` for `csa`). Entries that are not
  used are `-1`. No output size depends on tensor *data*, so the kernel never copies a count back
  to the host and never allocates from a device-computed size.
- **Explicit, graph-visible state.** Everything that survives between calls — the concatenated
  indexer key cache, the compressed-key cache and the partial-window buffers — is an input/output
  pair. The operator caches nothing internally.
- **A single versioned schema.** Both policies share one schema; the policy-specific inputs and
  outputs are optional slots at fixed indices and are validated strictly (see
  [§9](#9-validation-rules)).

## 2. Operator Schema

### Attributes

| Attribute | Type | Required | Description |
|---|---|---|---|
| `policy_mode` | string | yes | Exactly `qsa` or `csa`. Any other value is rejected. |
| `compress_ratio` | int | yes | Tokens folded into one block/entry. Must be `> 0`. |
| `token_budget` | int | `qsa` only | Maximum number of tokens taken from complete blocks. Must be `> 0` and divisible by `compress_ratio`. Must be absent for `csa`. |
| `index_topk` | int | `csa` only | Number of compressed entries selected per query. Must be `> 0`. Must be absent for `qsa`. |
| `epsilon` | float | no | RMSNorm epsilon of the compressed key. Default `1e-6`. |
| `scale` | float | no | Scale of the per-head ReLU score. Default `1/sqrt(head_size)`. |
| `head_weight_scale` | float | `csa` only | Scale applied to `head_weights`. Default `1/sqrt(num_heads)`. Must be absent for `qsa`. |

### Inputs

`B` = batch size, `S` = sequence length, `N` = `num_heads`, `D` = `head_size`,
`r` = `compress_ratio`, `P` = past sequence length, `T = P + S`, `R` = rotary width.

| # | Name | Policy | Type | Shape |
|---|---|---|---|---|
| 0 | `query` | both | `T` | `(B, S, N, D)` |
| 1 | `key` | both | `T` | `(B, S, D)` for `qsa`, `(B, S, 2D)` for `csa` |
| 2 | `key_norm_weight` | both | `T` | `(D)` |
| 3 | `cos_cache` | both | `T` | `(B, max_rotary_sequence_length, R)` |
| 4 | `sin_cache` | both | `T` | same as `cos_cache` |
| 5 | `mask` | `qsa` | `TB` | `(B, 1, S, T)` or `(B, S, T)` |
| 6 | `past_key` | `qsa` | `T` | `(B, P, D)` |
| 7 | `gate` | `csa` | `T` | `(B, S, 2D)` |
| 8 | `position_bias` | `csa` | `T` | `(r, 2D)` |
| 9 | `head_weights` | `csa` | `T` | `(B, S, N)` |
| 10 | `position_ids` | `csa` | `I` | `(B, S)` |
| 11 | `past_compressed_key` | `csa` | `T` | `(B, Pc, D)` |
| 12 | `past_kv_buffer` | `csa` | `T` | `(B, Lb, 2D)`, `Lb` in `[0, 2r)` |
| 13 | `past_gate_buffer` | `csa` | `T` | same as `past_kv_buffer` |

### Outputs

| # | Name | Policy | Type | Shape |
|---|---|---|---|---|
| 0 | `selected_indices` | both | `M` | `(B, S, capacity)` |
| 1 | `present_key` | `qsa` | `T` | `(B, T, D)` |
| 2 | `present_compressed_key` | `csa` | `T` | `(B, Pc + W, D)` |
| 3 | `present_kv_buffer` | `csa` | `T` | `(B, Lb', 2D)` |
| 4 | `present_gate_buffer` | `csa` | `T` | same as `present_kv_buffer` |

`W` is the number of complete windows closed by this call and `Lb'` the new buffer length; both
follow from `Lb`, `S` and `r` alone (see [§5](#5-policy-csa)).

### Type constraints

| Name | Allowed types |
|---|---|
| `T` | `tensor(float)`, `tensor(float16)`, `tensor(bfloat16)` |
| `TB` | `tensor(bool)` |
| `I` | `tensor(int64)` |
| `M` | `tensor(int32)` |

Only the CUDA execution provider registers a kernel. There is no CPU kernel; the header under
`contrib_ops/cpu/sparse/` only holds the CUDA-free constants that the schema, the kernel and the
tests must agree on.

### Output slot discipline

A `qsa` node declares exactly 2 outputs (`selected_indices`, `present_key`). A `csa` node declares
exactly 5, leaving slot 1 as a missing optional so that the three `csa` state outputs keep their
fixed indices. Shape inference verifies the declared output count *before* touching any output, so
`getOutputType` is never called past the declared range.

## 3. State Contract

`SparseAttentionIndexer` is a pure function of its inputs. Every value it needs on the next call is
returned as an output, so the caller (or the ORT session binding) owns the buffers:

| Policy | State pair |
|---|---|
| `qsa` | `past_key` → `present_key` |
| `csa` | `past_compressed_key` → `present_compressed_key` |
| `csa` | `past_kv_buffer` → `present_kv_buffer` |
| `csa` | `past_gate_buffer` → `present_gate_buffer` |

`present_key` and `present_compressed_key` grow by a number of positions that is known from the
input shapes and the attributes, so the graph can pre-allocate them. The two `csa` buffers stay
bounded by `2r - 1` positions.

## 4. Policy `qsa`

For every `(b, s)`:

1. **Visible set.** `visible = [t for t in range(T) if mask[b, s, t]]`, in ascending `t`.
2. **Complete blocks.** `nblocks = len(visible) // r`. Block `j` covers
   `visible[j*r : (j+1)*r]`.
3. **Pooled key.** `k_j = mean` of the `r` raw `present_key` rows of block `j`, then
   `RMSNorm(k_j) * key_norm_weight`, then rotary at absolute position `visible[j*r]`.
4. **Score.** `score_j = scale * sum_h ReLU(q_h · k_j)` where `q_h` is the rotated query head
   at absolute position `P + s`.
5. **Selection.** `topk = min(token_budget // r, nblocks)` blocks, ordered by decreasing score.
   Their tokens are emitted in that block order, `r` token indices per block.
6. **Tail.** The `len(visible) % r` tokens of the trailing incomplete block,
   `visible[nblocks*r:]`, are appended unconditionally.
7. Remaining entries up to `capacity = token_budget + r - 1` are `-1`.

The capacity is exactly `token_budget` selected tokens plus at most `r - 1` tail tokens.

## 5. Policy `csa`

### Window bookkeeping

Let `ext = concat(past_kv_buffer, key)` along the sequence axis (and likewise for the gates).
The buffer is split as

```
overlap = (Lb >= r) ? r : 0        # previous complete window, the "Ca" source
leftover = Lb - overlap            # tokens of the still-open window, always < r
pending  = leftover + S
W        = pending / r             # complete windows closed by this call
```

Window `w` (`0 <= w < W`) covers `ext[overlap + w*r : overlap + (w+1)*r]`. Its `Ca` partner is the
preceding `r` tokens, which exist iff `w >= 1 || overlap == r`. Because `leftover < r` always holds,
a buffer length `>= r` unambiguously means "the previous complete window is present" — no extra
state tensor is needed to disambiguate.

The new buffer starts at `overlap + (W-1)*r` when `W > 0` (the last complete window followed by the
new leftover) and at `0` otherwise, giving

```
Lb' = (W > 0) ? r + pending % r : Lb + S
```

`Lb'` is again in `[0, 2r)`.

### Compression

For window `w`, a `2r`-slot pooling window is built from `[B, 2r, D]` values and `[B, 2r, D]` gates:

- slots `[r, 2r)` take the **`Cb`** half (channels `[D, 2D)`) of the window's own `r` tokens;
- slots `[0, r)` take the **`Ca`** half (channels `[0, D)`) of the preceding `r` tokens, or are
  masked out (value `0`, gate `-inf`) when that window does not exist.

`position_bias[slot % r]` is added to the raw gate (the buffers store the **raw** projection, so
the bias is re-applied at use time and never accumulates). A softmax over the `2r` slots is taken
**per channel `d`**, the values are weighted and summed, the result is RMS-normalized with
`key_norm_weight`, and finally rotated at absolute position `(Pc + w) * r`.

### Scoring and selection

```
scores[b, s, e] = head_weight_scale * sum_h head_weights[b, s, h] * scale * ReLU(q_h · k_e)
threshold[b, s] = (position_ids[b, s] + 1) / r        # integer division
scores[b, s, e] = -inf  for  e >= threshold[b, s]
```

The top `min(index_topk, Pc + W)` entries are emitted in decreasing score order; any emitted entry
whose index is `>= threshold` (which only happens when fewer than `index_topk` entries are
visible) is written as `-1`, and the remaining capacity is `-1`.

## 6. Rotary Embeddings

Both policies reuse the model's precomputed `cos_cache` / `sin_cache`, indexed by absolute
position. This keeps the exact rotary variant of each reference model without re-deriving
frequencies inside the kernel:

- **`qsa` (leading, split-half).** `R = cos_cache.shape[2]` channels are rotated, and the rotation
  is the split-half form
  `out[i] = x[i]*cos[i] - x[i + R/2]*sin[i]`, `out[i + R/2] = x[i + R/2]*cos[i + R/2] + x[i]*sin[i + R/2]`,
  applied to channels `[0, R)`. Channels `[R, D)` pass through. Feeding an MRoPE-expanded
  `cos_cache`/`sin_cache` therefore reproduces the model's MRoPE exactly, because the operator
  never recomputes the position→angle mapping.
- **`csa` (trailing, interleaved).** `2R` channels are rotated and they are the **last** `2R`
  channels of the head. `cos`/`sin` are half-width, so entry `j >> 1` covers the channel pair
  `(j, j+1)`, and the rotation is the interleaved form
  `out[2i] = x[2i]*cos[i] - x[2i+1]*sin[i]`, `out[2i+1] = x[2i+1]*cos[i] + x[2i]*sin[i]`.

Positions are clamped into `[0, max_rotary_sequence_length - 1]` inside the kernel, so an
out-of-range `position_ids` value cannot read out of bounds.

### `key_norm_weight` and zero-centered gamma

`key_norm_weight` is the **effective** multiplier: the kernel computes
`normalized * key_norm_weight`. Qwen's `Qwen4ExpTextRMSNorm` multiplies by `1 + gamma`. Exporters
targeting that model must fold the addition into the initializer (`key_norm_weight = 1 + gamma`).
DeepSeek's `DeepseekV4RMSNorm` uses a plain `weight *`, so its tensor is passed through unchanged.

## 7. CUDA Kernel Pipeline

All kernels use a fixed block of 128 threads (a power of two, required by the shared-memory block
reductions). Elementwise kernels clamp the grid to 65535 blocks and use grid-stride loops; kernels
that assign one block to each work item clamp the grid to the CUDA `gridDim.x` limit.

### `qsa`

| Stage | Kernel | Parallelism |
|---|---|---|
| 1 | `ConcatPastKeyKernel` | element |
| 2 | `RotateQueryKernel<T, /*leading=*/true>` | one block per `(b, s, h)` |
| 3 | `CompactVisibleKernel` | one block per `(b, s)`; Hillis–Steele scan in shared memory |
| 4 | `QsaBlockScoreKernel` | one block per `(b, s, block)`; mean-pool → RMSNorm → rotary → score |
| 5 | `QsaSelectKernel` | one block per `(b, s)`; iterated block arg-max |

### `csa`

| Stage | Kernel | Parallelism |
|---|---|---|
| 1 | `CsaCopyPastCompressedKernel` | element |
| 2 | `CsaCompressKernel` | one block per `(b, window)`; per-channel softmax over `2r` slots |
| 3 | `CsaCopyBufferKernel` | element |
| 4 | `RotateQueryKernel<T, /*leading=*/false>` | one block per `(b, s, h)` |
| 5 | `CsaScoreKernel` | one thread per `(b, s, entry)` |
| 6 | `CsaSelectKernel` | one block per `(b, s)`; iterated block arg-max |

### Workspaces

| Policy | Buffer | Elements |
|---|---|---|
| `qsa` | float | `B*S*N*D` (rotated query) + `B*S*max_block_count` (block scores) |
| `qsa` | int32 | `B*S*T` (visible indices) + `B*S` (visible counts) |
| `csa` | float | `B*S*N*D` (rotated query) + `B*S*present_compressed_length` (scores) |

Every size is derived from shapes and attributes only.

### Selection without a visited bitmap

`QsaSelectKernel` and `CsaSelectKernel` repeatedly scan for the next element in the total order
"score descending, index ascending", using the previously emitted `(score, index)` pair as the
cursor. That avoids an `O(entries)` bitmap in shared memory, keeps the selection deterministic and
makes ties resolve to the smaller index. The cost is `O(topk * entries)` per query row, which is
the main performance follow-up below.

## 8. Numerics and Determinism

- Pooling, softmax, RMS normalization, rotary and scoring are all done in `float32`; only the final
  store is rounded to the tensor element type. This is a deliberate deviation from the reference
  implementations, which for `qsa` run in the model dtype — the operator is strictly more accurate,
  never less.
- The `2r`-slot softmax is a two-pass (max-subtracted) formulation, so a fully masked slot column
  cannot produce `NaN`.
- Selection order is a total order, so the output is bitwise reproducible for a given input.
- `float16` and `bfloat16` are supported for all `T` tensors. `bfloat16` uses the conversion
  helpers in `cu_inc/cuda_type_helper.cuh`, which are emulated on pre-`sm_80` devices, so no
  architecture guard is required.

## 9. Validation Rules

Shape inference and the kernel both reject:

- a `policy_mode` other than `qsa` / `csa`;
- `compress_ratio <= 0`;
- a `qsa` node that provides any `csa`-only input (slots 7–13) or attribute, and vice versa;
- a missing required input for the active policy;
- `token_budget` absent, `<= 0`, or not divisible by `compress_ratio` for `qsa`;
- `index_topk` absent or `<= 0` for `csa`;
- an output count other than 2 (`qsa`) or 5 (`csa`);
- `past_kv_buffer` / `past_gate_buffer` lengths outside `[0, 2 * compress_ratio)` or differing from
  each other;
- rank or dimension mismatches between `query`, `key`, the caches and the buffers.

Because the checks live in shape inference, most misuse fails at `Graph::Resolve()` with a clear
message rather than at kernel launch.

## 10. Testing

[`onnxruntime/test/contrib_ops/sparse_attention_indexer_op_test.cc`](../../../onnxruntime/test/contrib_ops/sparse_attention_indexer_op_test.cc)
contains:

- **Shape-inference tests** that build a real `Model`, run `Graph::Resolve()` and assert the
  inferred element type and shape of every output for both policies, including the buffer-only
  `csa` case (`W == 0`).
- **Negative tests** that assert the exact validation message for each rule in
  [§9](#9-validation-rules), including a `csa` node that declares too few outputs — the case that
  would otherwise write past the declared output range.
- **Numeric tests** (`float`, `float16`, `bfloat16`) that run the CUDA kernel against a float
  reference implementation of both policies written directly from the reference semantics. Inputs
  are round-tripped through the tested element type before the reference runs, so the reference
  sees exactly the values the kernel reads. These tests skip when no CUDA EP is available.

Run them with:

```bash
./build/Linux/Release/onnxruntime_provider_test --gtest_filter='SparseAttentionIndexer*'
```

## 11. Known Limitations and Performance Follow-ups

The implementation is correctness-first. The following are known and deliberate:

1. **Selection is `O(topk * entries)` per query row.** Each emitted index costs a full block-wide
   scan. A radix-select or a per-row bitonic top-k would reduce this to roughly one pass, and is the
   single biggest win for large `index_topk` / `token_budget`.
2. **Scoring is not tensor-core accelerated.** `QsaBlockScoreKernel` and `CsaScoreKernel` compute
   `q · k` with a shared-memory block reduction, one dot product per block. A tiled GEMM (or a
   fused `ReLU`+reduce epilogue) would be far better once shapes grow.
3. **The rotated query is materialized in `float32`.** That costs `B*S*N*D` floats of workspace.
   Fusing the rotation into the scoring kernels removes the traffic at the cost of recomputing the
   rotation per block.
4. **`CompactVisibleKernel` is `O(T)` per query row** and re-reads the mask for every `s`. For long
   contexts a batched exclusive scan over the whole `(B, S, T)` mask would be cheaper.
5. **`present_key` / `present_compressed_key` are copied every call.** An in-place cache with a
   `past_sequence_length` input (as `GroupQueryAttention` does) would avoid the copy, at the cost of
   a less explicit state contract.
6. **`compress_ratio` and `head_size` are not specialized.** Templating the hot kernels on a small
   set of common values would remove the dynamic loop bounds.
7. **No CPU kernel.** The operator is CUDA-only today.
