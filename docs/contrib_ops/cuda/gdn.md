# GatedDeltaNet — Operator Documentation

This document describes the `com.microsoft::GatedDeltaNet` contrib operator: its schema and
state contract, the chunked and warp-specialised CUDA engines and how one is selected, the
consumer-Blackwell shared-memory constraint, and measured comparisons against
`LinearAttention`, `VarlenLinearAttention` and flash-linear-attention (FLA).

The operator implements gated linear attention with an explicit recurrent state — the
"gated delta net" family used by Qwen3-Next / Qwen3.5 / Qwen3.8 style hybrid models.

Source:
[bert_defs.cc](../../../onnxruntime/core/graph/contrib_ops/bert_defs.cc) (schema),
[gated_delta_net.cc](../../../onnxruntime/contrib_ops/cuda/bert/gated_delta_net.cc),
[gated_delta_net_impl.cu](../../../onnxruntime/contrib_ops/cuda/bert/gated_delta_net_impl.cu),
[gated_delta_net_plan.h](../../../onnxruntime/contrib_ops/cuda/bert/gated_delta_net_plan.h).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Operator Schema](#2-operator-schema)
3. [Layout and State Contract](#3-layout-and-state-contract)
4. [Recurrence and the Chunked Form](#4-recurrence-and-the-chunked-form)
5. [CUDA Engines and Plan Selection](#5-cuda-engines-and-plan-selection)
6. [Shared Memory and Consumer Blackwell (SM120)](#6-shared-memory-and-consumer-blackwell-sm120)
7. [Benchmarks](#7-benchmarks)
8. [Accuracy](#8-accuracy)
9. [Environment Variables](#9-environment-variables)
10. [Testing](#10-testing)
11. [Known Limitations and Future Work](#11-known-limitations-and-future-work)

---

## 1. Overview

`GatedDeltaNet` is a packed, token-major operator. It differs from the existing
`LinearAttention` operator in three ways that matter for both performance and correctness:

- **Token-major (THD) inputs.** Head counts are derived from rank-3 shapes rather than from
  attributes, and an optional `cu_seqlens` allows variable-length requests in one call.
- **V-major float32 state, split into `initial_state` / `final_state`.** The recurrence
  boundary is where reduced precision hurts most.
- **Compact speculative-state capture.** A single packed `state_update` output records only
  the transitions needed to replay an accepted prefix. It avoids materializing a dense copy
  of the recurrent state for every draft token.
- **A chunked, tensor-core prefill engine.** `LinearAttention` runs a sequential per-token
  recurrence with no tensor-core use at any sequence length; `GatedDeltaNet` uses a chunked
  (WY / UT-transform) algorithm for prefill and keeps a sequential engine only for decode.

## 2. Operator Schema

### Attributes

| Attribute | Type | Default | Description |
|---|---|---|---|
| `update_rule` | string | `gated_delta` | One of `linear`, `gated`, `delta`, `gated_delta`. Selects which of the decay and delta-retrieval terms are present. |
| `scale` | float | `0.0` | Output scale. `0.0` means `1/sqrt(head_size_qk)`. |
| `gate_activation` | string | `none` | `none` treats `decay` as the effective log-space decay. `qwen` computes `-exp(a_log) * Softplus(decay + dt_bias)` in float32. |
| `beta_activation` | string | `none` | `none` treats `beta` as the effective update rate; `sigmoid` applies a sigmoid. |
| `qk_l2_norm` | int | `0` | When `1`, L2-normalizes each query and key head vector in-kernel. |
| `chunk_size` | int | `64` | Chunk length for the chunked engine. `32` pins the reduced shared-memory configuration (see §6). |
| `state_update_capacity` | int | `0` | Maximum number of leading per-request transitions captured in output 2, in `[0, 8]`. `capture_count` is required exactly when this is positive. |

### Inputs

| # | Name | Shape | Type | Required |
|---|---|---|---|---|
| 0 | `query` | `(total_tokens, num_heads_q, head_size_qk)` | T | yes |
| 1 | `key` | `(total_tokens, num_heads_k, head_size_qk)` | T | yes |
| 2 | `value` | `(total_tokens, num_heads_v, head_size_v)` | T | yes |
| 3 | `cu_seqlens` | `(batch_size + 1)` | int32 | no |
| 4 | `decay` | `(total_tokens, num_heads_v)` or `(total_tokens, num_heads_v, head_size_qk)` | float | no |
| 5 | `beta` | `(total_tokens, num_heads_v)` | float | no |
| 6 | `initial_state` | `(batch_size, num_heads_v, head_size_v, head_size_qk)` | float | no |
| 7 | `a_log` | `(num_heads_v)` | float | no |
| 8 | `dt_bias` | `(num_heads_v)` | float | no |
| 9 | `capture_count` | `(batch_size)` | int32 | when `state_update_capacity > 0` |
| 10 | `state_update_active` | `(1)` in CPU memory | int32 | no |

Input slot 7 contains `A_log`. The operator evaluates the fused gate, beta, and normalization
arithmetic in float32 and applies `scale` to the recurrence output. The float16 D=128 Qwen
shape is supported by recurrent, fused chunked, and split chunked engines; non-float16 inputs
remain on the recurrent engine.

### Outputs

| # | Name | Shape | Type | Required |
|---|---|---|---|---|
| 0 | `output` | `(total_tokens, max(num_heads_q, num_heads_v), head_size_v)` | T | yes |
| 1 | `final_state` | `(batch_size, num_heads_v, head_size_v, head_size_qk)` | float | no |
| 2 | `state_update` | `(batch_size, C * (H_v + H_k * D_k + H_v * D_v))` | float | no |

Type constraints: `T` is `float`, `float16`, or `bfloat16`; state, decay and beta are always
`float`, independent of `T`.

Head-count rules: `num_heads_q == num_heads_k`, and `num_heads_v` must be a positive multiple
of `num_heads_q`. This is *inverse* grouped-query attention — each query/key head is shared
by `num_heads_v / num_heads_q` value heads, which is the Qwen3.8 arrangement
(`num_heads_q = num_heads_k = 16`, `num_heads_v = 48`).

## 3. Layout and State Contract

**Rank 3 or rank 4.** Query, key and value are token-major. The leading token axis may be
packed (`[total_tokens, H, D]`) or spelled out as `[batch_size, sequence_length, H, D]`, in
which case decay and beta gain the same extra axis and so does the output. The two forms have
identical memory layouts. The rank-4 form exists so an exporter can go from a `(B, S, H*D)`
activation and back with static `Reshape` targets (`[0, 0, H, D]` and `[0, 0, H*D]`) instead
of Shape-derived ones, which would be placed on CPU and prevent CUDA graph capture. Ragged
packing needs the rank-3 form because `cu_seqlens` cannot describe a rectangular batch.

**Sequence packing.** With `cu_seqlens` present, it holds the exclusive prefix sums of the
per-request token counts and requests may have different lengths. With it absent the packing
is uniform and the batch size is taken from the rank-4 shape, or from `initial_state` when
the inputs are rank 3.

**`cu_seqlens` is device data.** Offsets are clamped to `[0, total_tokens]` inside the
kernels (`SequenceRange`), so a malformed producer cannot steer an out-of-bounds access.
There is no host synchronization and no device error flag; values outside the contract yield
unspecified results but remain memory-safe.

**Aliasing.** `initial_state` and `final_state` may be the same allocation. Both engines read
the entire incoming state into registers or shared memory before writing any of it.

**Compact transition layout.** Let `C = state_update_capacity`, `H_v = num_heads_v`,
`H_k = num_heads_k`, `D_k = head_size_qk`, and `D_v = head_size_v`. Each output row is one
struct-of-arrays capsule containing all `C * H_v` effective decay values, followed by all
`C * H_k * D_k` normalized keys, followed by all `C * H_v * D_v` state-dependent deltas.
For a scalar decay, replay applies one transition as `S *= decay; S += outer(key, delta)`.
Only the prefix selected by `capture_count[b]` is defined; the suffix is intentionally not
cleared. The shared-key representation is retained instead of expanding keys to value heads.

**Speculative rollback.** The runtime keeps one committed state per request. For partial
acceptance it starts from that state and replays the accepted compact transitions into the
inactive persistent bank, then atomically publishes the bank. Full acceptance uses
`final_state` directly. At the Qwen3.8 shape, a dense four-slot recurrent checkpoint window
would consume 576 MiB across 48 layers; compact capture allocates only the transition payload
and does not multiply persistent state by the speculative width.

When `state_update_active[0]` is zero, transition capture is disabled: `capture_count` is ignored,
`state_update` is zero-filled, and the planner may select a chunked engine that does not emit
transitions. When capture is active, chunked processing may handle non-capturing rows while a
recurrent tail pass emits the requested capsules. This keeps mixed prefill/decode batches on their
normal arithmetic paths.

## 4. Recurrence and the Chunked Form

Per value head, with `S` the `[head_size_qk x head_size_v]` state:

```
S_t = exp(g_t) S_{t-1} + k_t (beta_t (v_t - exp(g_t) S_{t-1}^T k_t))^T
o_t = scale * S_t^T q_t
```

Over a chunk of `BT` tokens, with `gc` the within-chunk cumulative log-decay:

```
M[t,s] = beta_t (k_t . k_s) exp(gc_t - gc_s)          strictly lower
U      = (I + M)^-1 (Ubar - Wbar S0)                  Ubar = beta v, Wbar[t] = beta_t exp(gc_t) k_t
P[t,s] = (q_t . k_s) exp(gc_t - gc_s)                 inclusive lower
o      = scale (P U + Qg S0)                          Qg[t] = q_t exp(gc_t)
S1     = exp(gc_BT) S0 + Kd^T U                       Kd[t] = k_t exp(gc_BT - gc_t)
```

Two properties of that form are used directly by the kernel:

1. **The `W` solve is removable.** `W` is only ever used as `W S0`, and `W = (I+M)^-1 Wbar`,
   so `U = (I+M)^-1 Ubar - W S0 = (I+M)^-1 (Ubar - Wbar S0)`. The `[BT x head_size_qk]`
   triangular solve disappears. Because `Wbar` is only a row scaling of `k`, it is never
  materialized either — the scaling folds into the epilogue of `(k S0)`.
2. **`(I+M)^-1` has an exact closed form.** With `Dinv` the block diagonal of the four
   `16x16` inverses and `N = Dinv M` (whose own `16x16` diagonal blocks are exactly zero),
   `N` is strictly block-lower over `BT/16` levels, so `N^(BT/16) = 0` and
   `(I+M)^-1 = (I - N + N^2 - ...) Dinv` terminates **exactly**. Evaluated by Horner this is
   a handful of full `BT x BT x BT` GEMMs, each using every warp, in place of a blocked
   forward substitution's sequence of tiny serial ones.

`k * exp(-gc)` is never formed; the decay ratio is applied to the `[BT x BT]` gram matrices
so the exponent stays bounded within a chunk.

**The delta family requires L2-normalized keys.** Without them `(I + M)` is arbitrarily
ill-conditioned and the recurrence diverges — in reduced precision the Neumann intermediates
overflow. Either normalize upstream or set `qk_l2_norm=1`.

## 5. CUDA Engines and Plan Selection

Selection follows the cuDNN frontend's execution flow: a heuristic (`SelectPlan`, the analog
of `heur_mode::A`) evaluates the problem `Descriptor` and the device shared-memory opt-in
limit to pick an engine and its tile parameters, the plan reports its workspace and
shared-memory requirement, and execution binds a `VariantPack` of device pointers.

| Engine | Used for | Shape |
|---|---|---|
| `chunked` | prefill | one CTA per `(sequence, v-head, v-block)`, 512 threads, state resident in shared memory for the whole walk, `mma.sync.m16n8k16` GEMMs |
| `chunked_split` | prefill, where the fused grid underfills the device | a token-parallel prepare launch, one CTA per `(sequence, v-head, chunk)`, followed by a scan that carries only the state |
| `recurrent` (warp-specialised) | decode / MTP verify | one **warp** per `(sequence, v-head, v-column)`, lanes spanning K |
| `recurrent` (generic) | head geometries the warp kernel cannot take | one CTA per `(sequence, v-head)`, state in shared memory |
| `cudnn` | reserved, not implemented | see §11 |

`SelectPlan` requires all of the following for the chunked engine, and falls back to the
sequential engine otherwise: `total_tokens >= 32 * batch_size`,
`head_size_qk == head_size_v == 128`, `num_heads_q == num_heads_k`,
`num_heads_v % num_heads_q == 0`, scalar (not per-key-dimension) decay, `float16` input, SM80
or newer, and enough shared memory. Active compact capture does not disqualify the chunked
engine: rows with positive `capture_count` are skipped by the chunked pass and handled by a
recurrent tail pass. Ragged rows shorter than the crossover use the same tail mechanism.

The **32-token threshold is measured, not assumed**: below roughly 30 tokens the chunked
engine still pays for a full 64-token chunk, so a single token costs about what 64 do
(46.5 us against 17.9 us for a sequential recurrence at the Qwen3.8 shape).

### Choosing between the fused and split chunked engines

The split engine exists to fill a machine the fused one leaves idle. Its measured advantage
tracks the fused engine's own **wave-quantisation efficiency** rather than a sequence-length
shape guess.
The fused grid is one CTA per `(sequence, v-head, 64 v-columns)`, so it occupies
`waves = batch * num_heads_v * (head_size_v / 64) / sm_count` waves of a device but costs
`ceil(waves)`. Measured on H200 (132 SMs) at the Qwen3.8 geometry, the split/fused runtime
ratio tracks that efficiency and nothing else:

| batch | fused CTAs | waves | efficiency | `T=256` | `T=1024` | `T=4096` |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 96 | 0.73 | 73% | 0.90 | **0.80** | 0.83 |
| 2 | 192 | 1.45 | 73% | 0.80 | **0.88** | 0.88 |
| 3 | 288 | 2.18 | 73% | 0.85 | **0.91** | 0.90 |
| 4 | 384 | 2.91 | 97% | 0.94 | 1.12 | 1.13 |
| 8 | 768 | 5.82 | 97% | 1.22 | 1.31 | 1.32 |

The fused per-token cost jumps from 0.392 us to 0.298 us across that same boundary, so once
the fused engine stops wasting a wave the split engine's extra launch and its round-trip
through the workspace are pure overhead. The split engine is currently an explicit opt-in
path selected with `ORT_GDN_PLAN=chunked_split` (the fused engine is the default production
prefill path). It also requires at least two chunks and enough shared-memory opt-in capacity
for both kernels, which excludes SM120.

Because the split engine keeps its state in float16 where the fused engine uses float32, the
two are **not** numerically identical, so this choice is deliberately a deterministic
function of the descriptor rather than a runtime autotune: an autotuned choice would make a
model's output depend on timing noise across processes. See §8 and §11 for the accuracy
evidence behind allowing it by default.

### Why the decode engine is warp-specialised

The V-major state makes a warp-per-column mapping natural, and it removes every barrier from
the token loop:

- Lanes span K, and the state is `[..., V, K]`, so lane accesses are fully coalesced. A
  CTA-per-head mapping instead walks `state[c * head_size_qk + r]` with consecutive threads
  on `c`, striding every access.
- Both reductions (`S^T k` and `S^T q`) are over K, so they are `__shfl_xor` butterflies.
  The token loop uses **no `__syncthreads()` and no shared memory**.
- The grid becomes `(sequences, v-heads, head_size_v / warps)` instead of
  `(sequences, v-heads)` — 768 CTAs instead of 48 at the Qwen3.8 decode shape.

Each lane takes `k = lane, lane + 32, ...` rather than a contiguous run. That is deliberate:
each of the resulting accesses is its own fully coalesced warp transaction, and the extra
requests in flight beat the single wide `float4` a contiguous run would allow — measured
5.86 us against 6.37 us.

## 6. Shared Memory and Consumer Blackwell (SM120)

The chunked engine's footprint depends on the chunk length. Consumer Blackwell allows only
99 KB of opt-in shared memory per block, against 227 KB on SM90 and SM100 — CUTLASS records
these as `sm120_smem_capacity_bytes = 101376` and `sm100_smem_capacity_bytes = 232448` in
`cutlass/arch/arch.h`.

| `chunk_size` x v-block | shared memory | fits SM120 (99 KB) | fits SM90 (227 KB) |
|---|---:|---|---|
| 64 x 64 | 157 KB | no | yes |
| 64 x 32 | 141 KB | no | yes |
| 32 x 64 | 96 KB | **yes** | yes |

`SelectPlan` therefore takes the widest chunk the device's `sharedMemPerBlockOptin` can
actually hold. `chunk_size = 64` is the faster configuration where it fits; `chunk_size = 32`
costs about 10% (127.2 / 440.2 / 3503.7 us at `T = 256 / 1024 / 8192` against
117.2 / 398.8 / 3090.1) at identical accuracy, and needs *fewer* Neumann terms — with two
diagonal blocks instead of four, `N^2 = 0`, so `(I+M)^-1 = (I - N) Dinv`.

Setting the `chunk_size` attribute to `32` pins the narrow configuration so the SM120 code
path can be exercised on other architectures. The kernel uses only `mma.sync.m16n8k16`
(sm_80 and newer PTX) with no wgmma, TMA or cluster features, so nothing else blocks SM120.

## 7. Benchmarks

**Hardware / build.** NVIDIA H200 (SM90, 132 SMs, driver 580.105.08), CUDA 13.0,
ONNX Runtime 1.30.0.

**Shape.** The Qwen3.8-27B linear-attention geometry: `num_heads_q = num_heads_k = 16`,
`num_heads_v = 48`, `head_size_qk = head_size_v = 128`, float16 input. One operator call is
all 48 value heads of **one** layer; the model has 48 such layers, so multiplying any figure
below by 48 gives the cost across a whole forward pass.

**Method.** `LinearAttention`, `VarlenLinearAttention` (PR #32166) and `GatedDeltaNet` are
compiled into **one provider build** and timed in **one process** under IOBinding, so every
tensor is device-resident and the timed region contains only `Run()`. Median of 30 after 10
warmup iterations.

**FLA is measured separately** in a PyTorch/Triton process (`torch 2.13.0+cu130`,
`triton 3.7.1`, flash-linear-attention `3da0a0c6`) at the same shapes and the same
median-of-30 protocol. It is *not* the same binary, so treat it as a reference point rather
than an exactly controlled comparison. `fla_chunk` is `chunk_gated_delta_rule`;
`fla_recurrent` is `fused_recurrent_gated_delta_rule`.

### Prefill, batch 1 (us per call, lower is better)

| T | LinearAttention | Varlen (#32166) | FLA chunk | **GatedDeltaNet** | vs LA | vs #32166 | vs FLA |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 399.0 | 585.4 | 422.0 | **126.4** | 3.16x | 4.63x | 3.34x |
| 1024 | 1841.1 | 2682.7 | 427.5 | **405.3** | 4.54x | 6.62x | 1.05x |
| 2048 | 4012.4 | 5761.9 | **454.9** | 797.6 | 5.03x | 7.22x | 0.57x |
| 8192 | 15942.8 | 22935.7 | **1014.3** | 3058.3 | 5.21x | 7.50x | 0.33x |

`T = 1024` is the production operating point (the validated prefill chunk for this model), so
**4.54x against the current operator and 6.62x against PR #32166** is the figure that matters
for end-to-end prefill; the default chunk of 256 gives 3.16x / 4.63x.

FLA is nearly flat in `T` because it parallelises across chunks in separate kernels, so it
overtakes this single-launch design beyond about `T = 1024`. That trade is deliberate: the
same multi-kernel structure gives FLA a **~375 us floor at batch 1** that makes it lose to
every other implementation below `T = 256` and by an order of magnitude at decode.

### Decode / MTP verify, batch 1 (us per call)

| T | LinearAttention | Varlen (#32166) | FLA recurrent | **GatedDeltaNet** |
|---:|---:|---:|---:|---:|
| 1 | **18.3** | 21.9 | 83.1 | 18.5 |
| 2 | 21.9 | 25.1 | 82.7 | **19.9** |
| 4 | 23.0 | 31.1 | 81.9 | **22.7** |

These end-to-end numbers include roughly 13 us of ORT `Run()` host overhead, which dominates
at decode shapes and compresses the differences. Pure kernel time from Nsight Compute
(`gpu__time_duration.sum`, p50) at `B = 1, T = 1`:

| kernel | grid | time | DRAM traffic | state dtype |
|---|---|---:|---:|---|
| `LinearAttentionDecodeColSplitKernel<half,128,8>` | 192 | 5.40 us | 1.61 MB | float16 |
| `VarlenLinearAttentionColKernel<half,128,1>` | 192 | 7.30 us | 1.63 MB | float16 |
| **`GatedDeltaNetDecodeWarpKernel<half,128,8>`** | 768 | **5.82 us** | 3.18 MB | float32 |

`GatedDeltaNet` reaches parity with the tuned `LinearAttention` decode kernel while moving
**twice the bytes**, because its state is float32 by contract where the others are float16 —
546 GB/s against 298 GB/s, about 1.8x the bandwidth efficiency per byte. Splitting by token
count gives a fixed cost of about 4.46 us and a per-token cost of about 1.36 us
(`T = 1` 5.82 us, `T = 4` 9.89 us), so the fixed state round-trip dominates decode.

### Batch 4 (us per call)

| T | LinearAttention | Varlen (#32166) | FLA chunk | **GatedDeltaNet** |
|---:|---:|---:|---:|---:|
| 256 | 2836.0 | 757.6 | 423.4 | **361.9** |
| 1024 | 11152.1 | 2920.7 | **572.8** | 1217.2 |
| 2048 | 22233.0 | 5814.3 | **955.1** | 2357.3 |
| 8192 | 88749.7 | 23213.2 | **3348.8** | 9305.5 |

`LinearAttention` degrades sharply with batch because `batch * num_heads_v` crosses its
grid-size heuristic and it falls back to a slower recurrent kernel; `VarlenLinearAttention`
removed that guard and is 3.7x faster there. `GatedDeltaNet` is a further 2.0x-2.5x faster
than `VarlenLinearAttention` across these shapes.

### End to end: Qwen3.5-27B-A3B (Qwen3.8) NVFP4, H200, batch 1

Two 64-layer models built from the same checkpoint with the same options, differing only in
which operator the 48 linear-attention layers use (`onnxruntime-genai` model builder,
`--extra_options linear_attn_op=gated_delta_net`). Prefill chunk 1024, 128 generated tokens.

| context | TTFT LinearAttention | TTFT GatedDeltaNet | prefill tok/s | decode ms/token |
|---:|---:|---:|---|---|
| 1024 | 234.3 ms | **183.9 ms** (-21.5%) | 4370 -> 5568 (+27%) | 12.38 -> 12.22 |
| 8192 | 1672.9 ms | **1239.2 ms** (-25.9%) | 4897 -> 6611 (+35%) | 13.29 -> 13.08 |

Greedy decoding agrees token for token (64/64) between the two models, with
`max|delta logits| = 0.075` on the 248320-wide fp16 logit vector at the first decode step.

### End to end with compact speculative-state replay

The compact-only Qwen3.8-27B NVFP4 packages were measured at context 2048 and generation 256
with five fresh-process repetitions. Before removing the compatibility candidate, each
native/compatibility pair ran on the same H200 and arm order was alternated. MTP uses three
draft tokens; DFlash2 uses seven. The table reports median decode throughput and the median
paired native/compatibility ratio.

| drafter | batch | native tok/s | compatibility tok/s | native speedup | native wins |
|---|---:|---:|---:|---:|---:|
| MTP | 1 | **195.76** | 188.38 | 1.0393x | 5/5 |
| MTP | 4 | **550.86** | 527.47 | 1.0481x | 5/5 |
| MTP | 8 | **760.02** | 721.01 | 1.0565x | 5/5 |
| MTP | 16 | **838.96** | 792.63 | 1.0609x | 5/5 |
| DFlash2 | 1 | **351.38** | 336.43 | 1.0467x | 5/5 |
| DFlash2 | 4 | **794.79** | 754.50 | 1.0521x | 5/5 |
| DFlash2 | 8 | **900.60** | 855.48 | 1.0554x | 5/5 |
| DFlash2 | 16 | **1644.98** | 1544.34 | 1.0659x | 5/5 |

Native won all 40 paired batch/repetition points. Both modes generated 256 tokens and
evaluated the same draft budget, and each individual configuration was deterministic across
its five repetitions. Native and compatibility are not token-identical to each other: their
different rounding boundaries change some draft decisions, so acceptance, Engine steps and
replay counts also differ. The throughput comparison therefore measures the complete serving
behavior of each arithmetic mode rather than an artificially fixed acceptance trace.

### Packed capsule versus separate factor outputs

The retired three-output representation and the packed capsule carry the same FP32 payload,
11,074,560 bytes per batch row across 48 GDN layers. The separate representation also needs
three graph outputs and three full-capacity clears per invocation; the capsule is one output
and writes only the valid prefix. Paired context-2048, generation-256 results were:

| drafter | batch | capsule tok/s | factors tok/s | paired capsule speedup | capsule wins |
|---|---:|---:|---:|---:|---:|
| MTP (5 reps) | 1 | **196.77** | 191.92 | 1.0279x | 5/5 |
| MTP (5 reps) | 4 | **546.10** | 535.24 | 1.0242x | 5/5 |
| MTP (5 reps) | 8 | **742.03** | 735.95 | 1.0112x | 5/5 |
| MTP (5 reps) | 16 | **820.31** | 812.11 | 1.0079x | 5/5 |
| DFlash2 (4 reps) | 1 | **352.32** | 346.52 | 1.0185x | 4/4 |
| DFlash2 (4 reps) | 4 | **783.96** | 777.88 | 1.0088x | 4/4 |
| DFlash2 (4 reps) | 8 | **883.45** | 879.95 | 1.0052x | 4/4 |
| DFlash2 (4 reps) | 16 | 1577.13 | **1578.77** | 0.9999x | 2/4 |

Tokens, Engine steps, draft counters and replay work matched exactly between capsule and
factor arms. MTP favored the capsule in all 20 pairs; DFlash2 shows the same small-batch
benefit and parity at batch 16. Together with the simpler graph/runtime contract and removal
of redundant clears, this is the basis for exposing only the packed output.

### Reproducing

```bash
# ORT three-way (same binary, same process)
unset PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python bench_three_way.py

# FLA reference
PYTHONPATH=/path/to/flash-linear-attention CUDA_VISIBLE_DEVICES=0 python bench_fla.py

# pure kernel time
ncu -k regex:"GatedDeltaNetDecodeWarpKernel" --metrics gpu__time_duration.sum,dram__bytes.sum \
    --csv python bench_three_way.py
```

## 8. Accuracy

Maximum relative error of `output` / `final_state` against a float64 sequential reference at
the shape above:

| Implementation | output | state |
|---|---|---|
| `LinearAttention` (float16) | 4.5e-4 | 2.9e-4 |
| FLA `fused_recurrent_gated_delta_rule` (float32) | 1.9e-7 | 3.0e-7 |
| FLA `chunk_gated_delta_rule` (float32) | 2.0e-3 | 1.3e-3 |
| **`GatedDeltaNet` (float16 io, float32 state)** | **8.7e-4** | **5.6e-4** |

`VarlenLinearAttention` was not measured against this reference; it shares
`LinearAttention`'s sequential recurrence, so its error is expected to be comparable.

Verified at `T = 1, 4, 63, 64, 65, 130, 256` so both chunk boundaries and partial chunks are
covered. The chunked engine is more accurate than FLA's chunked path, whose error is a TF32
artifact of `tl.dot`, and is within a small factor of the sequential float16 operators
despite replacing the recurrence with matrix algebra.

Because the chunked algorithm reassociates the recurrence, its results are **not** bitwise
identical to a sequential implementation. Any model-level change adopting this operator
should be gated on a task-level quality evaluation rather than on output hashes.

Native and compatibility arithmetic were evaluated on the same compact-only Qwen3.8-27B MTP
model pair before the compatibility candidate was removed. Scores are paired per question;
`native only` / `compat only` count discordant questions. Positive deltas favor native.

| task | n | native | compatibility | delta | discordant (native only / compat only) | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| GPQA | 198 | **81.82%** | 78.28% | +3.54pp | 25 (16 / 9) | 0.23 |
| MMLU-Pro | 800 | **83.75%** | 83.38% | +0.38pp | 33 (18 / 15) | 0.73 |

Neither difference is statistically significant at the 0.05 level. Compatibility showed no
aggregate quality advantage, while native won all 40 paired speculative-throughput runs by
3.9%-6.6%. The unpublished compatibility attribute and its reduced-precision kernel paths
were therefore removed; the operator exposes only the native fused arithmetic.

## 9. Environment Variables

| Variable | Values | Description |
|---|---|---|
| `ORT_GDN_PLAN` | `chunked`, `chunked_split`, `recurrent`, `cudnn` | Pins an engine, bypassing the heuristic, in either direction: `chunked` keeps the fused engine where the heuristic would split, and `chunked_split` forces the split engine where it would not. Either chunked override reports an error when the descriptor cannot use that engine. The analog of cuDNN's `select_plan(name)`. Intended for benchmarking and bisection. `cudnn` is reserved and returns an error. |

## 10. Testing

```bash
./onnxruntime_provider_test --gtest_filter='GatedDeltaNet*'
```

Note that contrib op tests link into `onnxruntime_provider_test`, not
`onnxruntime_test_all`. Coverage in
[gated_delta_net_op_test.cc](../../../onnxruntime/test/contrib_ops/gated_delta_net_op_test.cc)
includes all four update rules, inverse GQA and equal head counts, chunk boundaries and
partial chunks, ragged and uniform packing, the fused activations, and the reduced
shared-memory chunk configuration. Three cases are worth calling out:

- **`Ragged_CompactReplayMatchesSequentialPrefixes`** replays each captured capsule prefix
  and compares it with repeated one-token recurrence, including mixed prefill/decode rows.
- **`TwoCallContinuationMatchesSingleRun`** feeds one call's state output back as the next
  call's input and requires it to reproduce a single long run.
- **`MalformedCuSeqlensIsClamped`** drives negative, decreasing and oversized device offsets
  and requires the run to remain memory-safe.
- **`GatedDeltaNetPlanTest.PicksNarrowChunkOnConsumerBlackwellSharedMemory`** exercises the
  plan heuristic at SM120's 101376-byte budget on the host, so the architecture-dependent
  choice is covered without SM120 hardware.

## 11. Known Limitations and Future Work

- **Chunked engine coverage.** float16 input only, `head_size_qk == head_size_v == 128`, and
  scalar decay. bfloat16 needs bf16 mma fragments; per-key-dimension decay (KDA) and other
  head sizes fall back to the sequential engine.
- **FLA overtakes beyond `T ~ 1024`.** The single-launch design trades long-sequence
  throughput for a low fixed cost. Recovering it needs chunk-parallelism without FLA's
  per-chunk state materialization.
- **Decode is bounded by the float32 state round-trip.** A float16 state would halve decode
  state traffic. The type constraint already keeps the state dtype independent of `T`, so
  this is a contract decision rather than a kernel rewrite — but float32 was chosen
  deliberately for recurrence stability at chunk and request boundaries. The compact ABI
  keeps only two persistent state banks and replays accepted transitions, avoiding the
  retired dense window's 144 MiB-per-draft-token cost at the Qwen3.8 geometry. The companion
  convolution state and compact appended values remain in the activation type.
- **The chunked engine is bound by its own serial chain, not by grid size.** Measured on
  H200 at `T=8192`, batch 1 (`ncu`): grid 96 CTAs, **0.73 waves per SM**, 25.0% warp
  occupancy, **7.1%** tensor-pipe utilization, **1.8%** DRAM throughput. It is tempting to
  read that as "too few CTAs", but widening the grid measurably loses: with a `v_block` of
  32 (192 CTAs) the same work takes 657.6 us instead of 404.8 us at `T=1024`, and
  `BT=32, v_block=32` (the configuration that maximizes CTA residency) takes 793.0 us.
  The reason is that 157 KB of shared memory pins the kernel to one CTA per SM, so extra
  CTAs only add waves. Batch scaling confirms it: batch 4 at `T=1024` costs 1217.2 us,
  which is exactly `ceil(384/132) = 3` waves times the 405 us single-wave time, so
  per-CTA time is fixed and co-residency contributes nothing.
  Consequently the only way to speed this engine up is to **shorten the per-CTA critical
  path**. That is what `Engine::kChunkedSplit` does; see below.
- **The split engine (`Engine::kChunkedSplit`) is 20% faster where the fused grid underfills
  the device.** Pushing
  `(I+M)^-1` through the subtraction gives `U = Uv - W S0` with `Uv = (I+M)^-1 (beta v)` and
  `W = (I+M)^-1 diag(beta e^gc) k`, so every state-independent term factorizes out. A
  prepare kernel owns one `(sequence, v-head, chunk)` — 768 CTAs instead of 96 at `T=1024` —
  and emits `W`, `Qg`, `Uv`, `P`, `Kd`; the scan kernel is then left with `U = Uv - W S`,
  `o = scale (P U + Qg S)`, `S = decay S + Kd^T U`. Measured on H200 at batch 1: `T=1024`
  406 us to **325 us**, `T=8192` 3069 us to **2579 us**. By `nsys` the scan is 200 us and
  the prepare 110 us, so the scan **alone** is 2.0x faster than the whole fused kernel —
  close to the 2.3x cut in serial FLOPs the split predicts. Three things fall out that the
  fused engine could not have:
  - Every one of the scan's GEMMs scales with `v_block`, so narrowing it now divides the
    work instead of duplicating it. `v_block=32` measures 325 us against 341 us for 64, the
    reverse of the fused engine's 1.6x loss.
  - `W` and `Qg` are stored adjacently, so `W S` and `Qg S` are one `[2 BT x DVB]` GEMM
    whose `Qg S` half stays live while `P U` overwrites the other, and the state update
    accumulates straight into the float16 state instead of staging through float32 scratch.
    Together those take the chunk body from seven barriers to four.
  - The state is float16 rather than float32, which is what brings the scan under the
    113 KB that lets two CTAs share an SM. The prepare kernel reaches the same limit by
    aliasing the Neumann iterate onto `M`, worth 143 us to 110 us on its own.
  Both kernels remain latency-bound (`ncu`: tensor 7.0% / 5.1%, DRAM 10.4% / 7.0%). What is
  left in the prepare kernel is **not** worth chasing: short-circuiting the whole `(I+M)^-1`
  construction saves only 28 us of its 110 us, and the 16x16 forward substitution inside it
  — the part that occupies just 64 of 512 threads, and so looks like the obvious target —
  accounts for 4.7 us, about 1.5% of the operator.
- **float16, not bfloat16, for the on-chip state.** The operator's q/k/v/output are already
  float16, `qk_l2_norm` bounds `|k| <= 1` and `beta = sigmoid(.) < 1`, so the state stays
  around `O(10)` against float16's 65504 ceiling; entries that underflow are ones the decay
  has already made irrelevant. float16 then carries three more mantissa bits than bfloat16
  on what is an accumulator running the length of the sequence. The reference
  implementations use bfloat16 only because their surrounding stacks are bfloat16-native.
  Scored against a float64 reference at the Qwen3.8 geometry, the float16 state costs
  **1.09x** the output relative RMS of the float32 state (5.5e-4 against 5.1e-4) and 1.22x
  on the returned state, and the ratio does not move between `T=1024` and `T=4096`, so the
  per-chunk rounding is not accumulating. Both figures sit at float16's own machine epsilon
  of 4.9e-4, which is the floor imposed by storing the output as float16 at all. At model
  level, running the same Qwen3.8-27B MTP model under both engines and pairing per question:

  | task | n | float32 state | float16 state | delta | discordant | McNemar p |
  |---|---:|---:|---:|---:|---|---:|
  | GPQA | 198 | 81.31% | 82.32% | +1.01pp | 18 (8/10) | 0.81 |
  | MMLU-Pro | 800 | 83.38% | 84.38% | +1.00pp | 46 (19/27) | 0.30 |

  Long chain-of-thought is the sensitive probe for a prefill-only numerics change, so GPQA is
  the one that matters here; neither moves. The split engine is currently selected via
  `ORT_GDN_PLAN=chunked_split`.
- **cuDNN backend.** `Engine::kCudnn` is reserved and unimplemented. As of cudnn-frontend
  v1.27.0 — the version this build pins, and the release that introduced GDN — the operation
  exists **only** in the Python package (`python/cudnn/linear_attention/`); there are no GDN
  symbols in the C++ headers, so the CUDA EP cannot call it. Its FROST engine is additionally
  restricted to SM100-SM103, excluding both SM90 and SM120, and its cuTile fallback requires
  the `cuda.tile` Python JIT and does not support compact state-update capture. The seam is kept so the
  decision can be revisited if a C++ entry point appears.
- **Model migration.** Packages exported with `state_checkpoints`, dense checkpoint outputs,
  or separate decay/key/delta update outputs are not accepted by the compact-only runtime.
  Re-export them with `state_update_capacity` and the single packed `state_update` output.
- **Prefill chunk parallelism, `cp.async` staging of the next chunk's q/k/v, and merging the
  `K K^T` and `Q K^T` GEMMs** are the ranked next steps for the chunked engine, which
  profiling shows to be latency-bound rather than compute- or bandwidth-bound.
