# PagedAttention - Backward-Compatible Schema Evolution

Status: **Proposal**

Scope: backward-compatible evolution of `com.microsoft::PagedAttention` opset 1 from the schema
currently shipped on the ONNX Runtime main branch.

Related documents:

- [paged_attention.md](paged_attention.md) describes the implementation work, feature semantics,
  and the alternative breaking cache-layout proposal.
- [gqa.md](gqa.md) describes `com.microsoft::GroupQueryAttention` and the shared attention features.

## 1. Decision

Evolve the existing operator additively. Preserve inputs 0-9, outputs 0-2, and every existing
attribute with its current meaning. Add model coverage and serving features through:

- trailing optional inputs;
- optional attributes whose defaults reproduce current behavior;
- widening the cache type constraint from `T` to `T_CACHE`;
- making `value_cache` optional only for an explicitly selected latent-cache mode; and
- extending output shape inference for asymmetric K/V head widths.

Do **not** merge `key_cache` and `value_cache`, replace the existing cache outputs, remove
`kv_num_heads`, rename `local_window_size`, or reinterpret an existing input combination.

This design provides slot mapping, attention sinks, QK-Norm, INT8/FP8 cache, paged decode,
absorbed MLA, scheduler metadata, explicit linear query positions, ALiBi, arbitrary attention bias,
and optional QK output without invalidating a model accepted by the main-branch schema.

## 2. Compatibility Invariant

Every model valid under the main-branch schema remains valid and has identical behavior when all
new inputs and attributes are absent.

For such a model:

```text
T_CACHE == T
value_cache is present
key and value are either both present or both absent
qkv_format == "AUTO"
kv_cache_layout == "SEPARATE"
v_head_size == 0
all quantization attributes select NONE
all trailing optional inputs are absent
```

The existing operator is already serialized as `com.microsoft::PagedAttention` opset 1. A future
contract that changes the meaning or position of existing inputs and outputs must use a separately
versioned schema or a new operator name; it must not replace this schema in place.

## 3. Main-Branch Baseline

The shipped schema has inputs 0-9:

| Idx | Name | Type | Shape | Presence |
|-----|------|------|-------|----------|
| 0 | `query` | `T` | `(token_count, hidden_size)` or packed QKV | required |
| 1 | `key` | `T` | `(token_count, kv_hidden_size)` | optional |
| 2 | `value` | `T` | `(token_count, kv_hidden_size)` | optional |
| 3 | `key_cache` | `T` | `(num_blocks, block_size, kv_num_heads, head_size)` | required |
| 4 | `value_cache` | `T` | same as `key_cache` | required |
| 5 | `cumulative_sequence_length` | `S` | `(batch_size + 1,)` | required |
| 6 | `past_seqlens` | `S` | `(batch_size,)` | required |
| 7 | `block_table` | `S` | `(batch_size, max_num_blocks_per_seq)` | required |
| 8 | `cos_cache` | `T` | `(max_seq_len, rotary_dim / 2)` | optional |
| 9 | `sin_cache` | `T` | `(max_seq_len, rotary_dim / 2)` | optional |

The shipped outputs are:

| Idx | Name | Type | Shape | Presence |
|-----|------|------|-------|----------|
| 0 | `output` | `T` | `(token_count, hidden_size)` | required |
| 1 | `key_cache_out` | `T` | same as `key_cache` | optional |
| 2 | `value_cache_out` | `T` | same as `value_cache` | optional |

The shipped attributes are `num_heads`, `kv_num_heads`, `scale`, `softcap`,
`local_window_size`, `do_rotary`, and `rotary_interleaved`. Their names, defaults, and semantics
remain unchanged.

## 4. Consolidated Compatible Contract

### 4.1 Inputs

Inputs 0-9 retain their current indices. `key_cache` and `value_cache` move to the widened
`T_CACHE` constraint. `value_cache` becomes optional, but may be absent only in `LATENT` mode.

| Idx | Name | Type | Shape | Change |
|-----|------|------|-------|--------|
| 0 | `query` | `T` | `(token_count, hidden_size)` or packed QKV | unchanged |
| 1 | `key` | `T` (opt) | `(token_count, kv_hidden_size)` | widened validation for `LATENT` |
| 2 | `value` | `T` (opt) | `(token_count, kv_value_hidden_size)` | optional in `LATENT` |
| 3 | `key_cache` | `T_CACHE` | `(num_blocks, block_size, kv_num_heads, head_size)` | type widened |
| 4 | `value_cache` | `T_CACHE` (opt) | `(num_blocks, block_size, kv_num_heads, v_head_size)` | optional in `LATENT` |
| 5 | `cumulative_sequence_length` | `S` | `(batch_size + 1,)` | unchanged |
| 6 | `past_seqlens` | `S` | `(batch_size,)` | unchanged |
| 7 | `block_table` | `S` | `(batch_size, max_num_blocks_per_seq)` | unchanged |
| 8 | `cos_cache` | `T` (opt) | `(max_seq_len, rotary_dim / 2)` | unchanged |
| 9 | `sin_cache` | `T` (opt) | `(max_seq_len, rotary_dim / 2)` | unchanged |
| 10 | `slot_mapping` | `S` (opt) | `(token_count,)` | scheduler-owned cache write locations |
| 11 | `head_sink` | `T` (opt) | `(num_heads,)` | attention-sink logits |
| 12 | `q_norm_weight` | `T` (opt) | `(head_size,)` | per-head Q RMSNorm gain |
| 13 | `k_norm_weight` | `T` (opt) | `(head_size,)` | per-head K RMSNorm gain |
| 14 | `k_scale` | `T_SCALE` (opt) | scalar or `(kv_num_heads, 1, head_size)` | K-cache scale |
| 15 | `v_scale` | `T_SCALE` (opt) | scalar or `(kv_num_heads, 1, v_head_size)` | V-cache scale |
| 16 | `attention_metadata` | `S` (opt, CPU) | `(3,)` | dispatch and workspace metadata |
| 17 | `query_positions` | `S` (opt) | `(token_count,)` | linear logical positions |
| 18 | `alibi_slopes` | `T_SCALE` (opt) | `(num_heads,)` | ALiBi slopes |
| 19 | `attention_bias` | `T` (opt) | `(1 or num_heads, token_count, max_context_len)` | dense additive bias |

Input 16 uses `OrtMemTypeCPUInput`. It contains:

```text
attention_metadata[0] = max_query_len
attention_metadata[1] = max_kv_len
attention_metadata[2] = total_kv_tokens
```

When it is absent, the implementation retains the existing device-to-host fallback. When it is
present, all values must be non-negative and `total_kv_tokens` must be positive for a non-empty
query. The producer is responsible for consistency with the device sequence-length tensors; the
kernel must not add a synchronizing validation copy.

`attention_bias` remains last because it is a correctness fallback with substantial memory cost.
`alibi_slopes` is preferred for the common position-bias case and can remain on fused backends.

### 4.2 Outputs

The existing outputs keep their indices and optionality:

| Idx | Name | Type | Shape | Presence |
|-----|------|------|-------|----------|
| 0 | `output` | `T` | `(token_count, num_heads * effective_v_head_size)` | required |
| 1 | `key_cache_out` | `T_CACHE` | same as `key_cache` | optional |
| 2 | `value_cache_out` | `T_CACHE` | same as `value_cache` | optional |
| 3 | `output_qk` | `T` | `(num_heads, token_count, max_context_len)` | optional |

In `SEPARATE` mode, cache outputs 1 and 2 are both present or both absent, preserving the current
rule. In `LATENT` mode, output 1 may be present and output 2 must be absent.

Requesting `output_qk` at index 3 requires a four-entry ONNX output list. Any unused cache-output
position is represented by an empty output name; output indices are never compacted. For example,
a `LATENT` node requesting `output_qk` uses `[output, key_cache_out-or-empty, empty, output_qk]`.

The existing implementation requires cache outputs to alias their corresponding inputs. That rule
is retained for compatibility. A future functional-output contract that remains correct when the
planner cannot reuse the input buffer should be introduced separately rather than silently changing
the side-effect contract of opset 1.

### 4.3 Attributes

| Name | Type | Default | Meaning |
|------|------|---------|---------|
| `num_heads` | INT | required | existing Q head count |
| `kv_num_heads` | INT | required | existing KV head count |
| `scale` | FLOAT | `1/sqrt(head_size)` | existing score scale |
| `softcap` | FLOAT | `0.0` | existing logit softcap |
| `local_window_size` | INT | `-1` | existing left window including current token |
| `do_rotary` | INT | `0` | existing in-op RoPE switch |
| `rotary_interleaved` | INT | `0` | existing RoPE convention |
| `smooth_softmax` | INT | `0` | add one denominator-only zero logit |
| `qk_norm_epsilon` | FLOAT | `1e-6` | Q/K RMSNorm epsilon |
| `k_quant_type` | STRING | `"NONE"` | `NONE`, `PER_TENSOR`, or `PER_CHANNEL` |
| `v_quant_type` | STRING | `"NONE"` | `NONE`, `PER_TENSOR`, or `PER_CHANNEL` |
| `kv_cache_bit_width` | INT | unset | `8` for an 8-bit cache; `0` or `16` otherwise |
| `v_head_size` | INT | `0` | `0` means equal to `head_size` |
| `rotary_offset` | INT | `0` | first head channel covered by RoPE |
| `kv_cache_layout` | STRING | `"SEPARATE"` | `SEPARATE` or `LATENT` |
| `qkv_format` | STRING | `"AUTO"` | `AUTO`, `Q_K_V`, or `QKV_PACKED` |
| `qk_output` | INT | `0` | `0` none, `1` pre-softmax, `2` post-softmax |

`qkv_format="AUTO"` preserves main-branch inference: K/V are packed in `query` exactly when
`key` and `value` are absent. An explicit format is recommended for new exporters but is not
required for old models.

Do not add `window_size_left` or `window_size_right` to opset 1. `local_window_size=W` has the
established semantic of admitting W positions including the current token. Its effective Flash
parameters are `window_size_left=W-1` and `window_size_right=0`.

### 4.4 Type Constraints

| Name | Allowed |
|------|---------|
| `T` | `float16`, `bfloat16` |
| `T_CACHE` | `float16`, `bfloat16`, `int8`, `float8e4m3fn` |
| `T_SCALE` | `float` |
| `S` | `int32` |

`uint8` is intentionally omitted until a concrete unsigned or sub-byte logical cache format is
specified. Widening inputs 3/4 and outputs 1/2 from `T` to `T_CACHE` is compatible: an old model
continues to bind `T_CACHE == T`.

## 5. Input-Mode State Machine

Input presence must be interpreted jointly with `qkv_format` and `kv_cache_layout`; no new mode is
inferred from an old input combination.

| `qkv_format` | `kv_cache_layout` | `query` | `key` | `value` | `value_cache` |
|--------------|-------------------|---------|-------|---------|---------------|
| `AUTO` | `SEPARATE` | Q or packed QKV | both present or both absent | same as K | required |
| `Q_K_V` | `SEPARATE` | Q | required | required | required |
| `QKV_PACKED` | `SEPARATE` | packed QKV | absent | absent | required |
| `Q_K_V` | `LATENT` | absorbed Q | required latent K | absent | absent |
| `AUTO` | `LATENT` | absorbed Q | required latent K | absent | absent |
| `QKV_PACKED` | `LATENT` | - | - | - | rejected |

`key` is required in `LATENT` mode because it carries the new latent row
`[compressed_kv; k_pe]`. Only V aliases K; K is not absent.

## 6. Quantized Cache

Quantized-cache support is an additive type widening from the main branch, not part of its existing
contract. Keep explicit quantization attributes so scale shape and declared semantics cannot
disagree silently.

### 6.1 Supported Modes

| Quantization | Cache type | Scale shape |
|--------------|------------|-------------|
| none | same type as `T` | absent |
| per-tensor INT8/FP8 | `int8` or `float8e4m3fn` | scalar or `(1,)` |
| per-channel INT8/FP8 | `int8` or `float8e4m3fn` | `(kv_num_heads, 1, head_size)` |

For V, replace `head_size` with `effective_v_head_size`. Quantization is symmetric:

$$
q = \operatorname{clamp}(\operatorname{round}(x / s)), \qquad x = q s.
$$

K and V may select different supported granularities, but both caches have the same element type
because inputs 3 and 4 share `T_CACHE`.

### 6.2 Validation

- `T_CACHE == T` requires both quant types to be `NONE` and both scale inputs to be absent.
- An INT8/FP8 cache requires both quant types to be non-`NONE` and both scales to be present in
  `SEPARATE` mode.
- In `LATENT` mode, only K storage exists. `k_quant_type` and `k_scale` describe the latent row;
  `v_quant_type` must equal `k_quant_type` and `v_scale` must be absent because V is a view of K.
- `kv_cache_bit_width` is `8` for INT8/FP8 and `0` or `16` for an unquantized cache.
- FP8 availability is controlled by the existing build gate. Do not add a runtime SM89/SM90 check
  solely for the converting constructor.

Per-token, per-block, asymmetric, INT4, INT2, FP4, and inline-scale formats are deferred. They need
state ownership and byte-packing contracts that cannot be expressed safely by merely accepting
additional scale shapes in this operator.

## 7. Absorbed MLA

`kv_cache_layout="LATENT"` selects absorbed Multi-head Latent Attention. It uses one physical
cache without changing the ordinary separate-cache representation:

```text
query     : (token_count, num_heads * head_size)
key       : (token_count, kv_num_heads * head_size)
value     : absent
key_cache : (num_blocks, block_size, kv_num_heads, head_size)
value_cache: absent
V view    : key_cache[..., 0:effective_v_head_size]
output    : (token_count, num_heads * effective_v_head_size)
```

For DeepSeek-V3 absorbed decode, `head_size=576`, `effective_v_head_size=512`,
`kv_num_heads=1`, and `rotary_offset=512`.

Validation:

- `key` is present; `value` and `value_cache` are absent.
- `v_head_size` is explicitly set and satisfies `0 < v_head_size <= head_size`.
- `scale` is explicitly set; the operator must not guess `1/sqrt(576)` for DeepSeek.
- `rotary_offset >= 0`, `rotary_offset % 8 == 0`, and
  `rotary_offset + rotary_dim <= head_size`.
- Initially require `kv_num_heads == 1`; widen only with a backend and tests for grouped latent
  heads.
- Reject `head_sink` and QK-Norm until a model requires and validates those combinations.
- Dispatch only to an MLA-capable backend or the unfused reference implementation.

The operator does not take `W_UK` or `W_UV`; absorption remains graph-level MatMul work. A
non-absorbed prefill that writes a latent cache should use an explicit graph-level cache-write path,
as described in the main design document.

## 8. Explicit Positions and Speculative Decoding

When `query_positions` is absent, token `j` of sequence `b` uses the existing position
`past_seqlens[b] + j`. When present, its element supplies the linear logical position used for
in-op RoPE and for backends that support arbitrary-position causal/window masking.

This input supports linear position overrides and chunked prefill. It does not by itself encode
tree ancestry: Medusa or general tree attention additionally requires a branch/ancestor mask.
Backends that derive positions solely from packed row alignment are ineligible when explicit
positions differ from the legacy sequence.

Cached K is stored after RoPE. Reusing a cached block at a different logical position is invalid
unless the producer guarantees the cached K was rotated for that position or a backend explicitly
rerotates it. `query_positions` does not make an already-rotated prefix relocatable.

`window_size_right > 0` is not added here. Supporting speculative lookahead beyond ordinary causal
attention requires an explicit mask/window contract and backend coverage; it should not change the
meaning of `local_window_size`.

## 9. CUDA Graph and Metadata Contract

`attention_metadata` removes the unconditional D2H synchronization in ordinary execution. CUDA
graph replay has an additional constraint: host-side backend selection and launch dimensions are
captured, not recomputed dynamically on replay.

Producers using CUDA graphs must therefore use one of these strategies:

- capture separate graphs for decode/prefill and relevant batch/length buckets;
- keep the captured launch topology fixed and supply dynamic device lengths within its bounds; or
- recapture when metadata changes in a way that changes backend or workspace selection.

The metadata input is a dispatch/workspace hint with required correctness. It is not a mechanism
for dynamically changing a captured graph's topology.

## 10. Shape Inference

Shape inference must preserve the existing packed-QKV behavior and then apply these extensions:

1. Propagate output 0 element type from input 0.
2. Derive `head_size` from unpacked `query.shape[1] / num_heads`, or use the legacy packed-QKV
   formula when `qkv_format` resolves to packed.
3. Set `effective_v_head_size = v_head_size == 0 ? head_size : v_head_size`.
4. Set output 0 shape to `(token_count, num_heads * effective_v_head_size)`.
5. Propagate output 1 type and shape from input 3, not input 0.
6. Propagate output 2 only when it exists and input 4 exists; use input 4 as its source.
7. Without `output_qk`, permit one output or the legacy cache-output forms described in section
  4.2. With `output_qk`, require a four-entry output list and permit empty names at unused optional
  cache-output indices. Validate `qk_output` without writing beyond `ctx.getNumOutputs()`.

Unknown symbolic dimensions should remain symbolic rather than causing shape inference to assume a
mode inconsistent with the explicit attributes.

## 11. Backend Eligibility

Feature acceptance and backend eligibility are separate. Schema validation establishes a valid
operator request; dispatch selects only a backend that implements the requested combination.

| Feature | Paged decode | Flash varlen | MEA | MLA/reference |
|---------|--------------|--------------|-----|---------------|
| Legacy FP16/BF16 | yes | yes | yes | n/a |
| Slot mapping | prologue | prologue | prologue | prologue |
| QK-Norm | prologue | prologue | prologue | rejected initially for MLA |
| INT8/FP8 cache | in-kernel | dequant-gather | dequant-gather | backend-specific |
| Head sink | native | LSE epilogue | requires LSE support | rejected initially for MLA |
| Explicit positions | backend work required | fallback unless legacy-equivalent | backend work required | backend-specific |
| ALiBi | planned | native wrapper support | planned | backend-specific |
| Dense bias / QK output | no | no | yes where implemented | reference |
| LATENT MLA | no | no | no | yes |

An unsupported combination must produce a clear `INVALID_ARGUMENT` naming the selected feature and
backend limitation. It must never silently ignore an input or attribute.

## 12. Deferred Breaking Contract

The following ideas remain useful but belong in a separately versioned schema or operator:

- one merged `kv_cache` containing K and V;
- one required functional `kv_cache_out`;
- removal of `kv_num_heads`;
- physical `HND` cache layout;
- logical sub-byte types stored in a `uint8` tensor;
- inline scales or zero points in cache rows;
- shape-only quantization-granularity inference; and
- a tree-attention mask or ancestry representation.

These are cache-format or state-contract changes, not prerequisites for the model and serving
features in this proposal.

## 13. Implementation Phases

| Phase | Contents | Schema delta |
|-------|----------|--------------|
| C0 | Main-branch defect fixes and parity tests | none |
| C1 | `slot_mapping`, sink, QK-Norm | inputs 10-13; two attributes |
| C2 | INT8/FP8 cache and dequant-gather | `T_CACHE`, inputs 14-15; quant attributes |
| C3 | Paged decode with in-kernel dequantization | none |
| C4 | Optional `attention_metadata`; remove unconditional D2H sync | input 16 |
| C5 | LATENT MLA reference, output shape, offset RoPE | input 4 optional; three attributes |
| C6 | Fused MLA backend | none |
| C7 | Explicit linear positions and ALiBi | inputs 17-18 |
| C8 | Dense bias and optional QK output | input 19, output 3, `qk_output` |

Each phase must rerun old-model tests with every new input absent, in addition to feature-specific
tests. The compatibility regression test should serialize a model using only the main-branch
contract and run it against the extended kernel without rewriting the graph.

## 14. Recommendation

Adopt this compatible contract for `com.microsoft::PagedAttention` opset 1. Treat the existing
separate-cache representation as permanent for that operator. It captures nearly all model coverage
and serving performance value; the deferred work is primarily physical cache-format cleanup and
specialized state semantics.

If a merged or sub-byte cache becomes a concrete requirement, introduce a separately versioned
contract with a migration tool instead of changing the meaning of already serialized opset-1
models.