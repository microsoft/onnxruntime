# PackedSparseAttentionIndexer on WebGPU

The WebGPU execution provider implements version 1 of
`com.microsoft.PackedSparseAttentionIndexer` for the `qsa` and `csa` policies. It uses the
provider-neutral schema and generic state ABI described in the
[operator documentation](../packed_sparse_attention_indexer.md).

## Supported subset

- packed (`total_tokens`-major) inputs, driven by device-resident
  `cumulative_sequence_lengths` / `past_sequence_lengths`;
- `qsa` and `csa` policy modes;
- `float32` and `float16`;
- shared cos/sin rotary cache (`(max_position, rotary_width)`) or request-specific cache
  (`(batch_size, max_position, rotary_width)`);
- generic fixed-capacity `past_key_state` / `past_kv_buffer` / `past_gate_buffer` /
  `past_state_lengths` state, including input/output aliasing;
- deterministic score-descending, index-ascending top-k ties;
- `selected_counts`, the exact active-entry count per query.

BF16 is not registered by the WebGPU kernel (CUDA only). Unknown policies and
policy-incompatible inputs or attributes are rejected.

## Execution

Every program is one invocation per row (one active thread per workgroup; the rest of the
workgroup is idle), exactly like the dense `SparseAttentionIndexer` WebGPU kernel: state-update
programs dispatch one row per **request**, and the score/select programs dispatch one row per
**query token**. As in the dense kernel, intermediate values (pooled/normalized/rotated keys,
per-candidate scores) are recomputed by small WGSL helper functions on demand rather than staged
into workgroup-shared arrays, both to keep every kernel correct without relying on WGSL arrays
sized by a runtime (uniform) `head_size`, and to keep the packed kernel's structure directly
comparable to the CUDA implementation's per-request/per-token update and score/select stages. All
reductions and softmax calculations accumulate in FP32, including for FP16 inputs.
Raw flattened queries are RMS-normalized per logical head with `query_norm_weight` before the
policy-specific rotary embedding, matching the CUDA implementation.

Per-request quantities (`cumulative_sequence_lengths`, `past_sequence_lengths`,
`past_state_lengths`) are read directly from device buffers inside the shaders — never on the
host — and are clamped into their valid ranges before use, so malformed packed metadata can never
cause an out-of-bounds buffer access (see the main document's device-side safety section).

## Follow-up work

- specialized large-candidate top-k;
- subgroup-optimized reductions;
- fused projection, pooling, and scoring;
- reduced recomputation and temporary-buffer use;
- BF16 support;
- WebGPU `SparsePagedAttention` end-to-end integration.
