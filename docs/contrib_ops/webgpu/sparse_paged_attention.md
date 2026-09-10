# WebGPU SparsePagedAttention

`com.microsoft.SparsePagedAttention` executes attention over externally selected
request-local indices. It does not perform selection or own indexer state. The
WebGPU implementation mirrors the numerics of the CUDA kernel documented in
[cuda/sparse_paged_attention.md](../cuda/sparse_paged_attention.md).

## Modes

| `attention_mode` | `selected_kv_source` | Reads |
|---|---|---|
| `selected_only` | `main` | Selected logical positions resolved through the main `block_table` |
| `selected_only` | `auxiliary` | Selected contiguous auxiliary rows |
| `local_plus_selected` | `main` | Union of the main-cache local window and the selected main-cache positions, de-duplicated |
| `local_plus_selected` | `auxiliary` | Main paged local window and selected contiguous auxiliary rows in one softmax |

Main-cache selection is causal when `is_causal=1`. Invalid, out-of-range,
non-causal, and unmapped selected positions are ignored rather than clamped, so a
caller may pad `selected_indices` with `-1` freely.

`selected_indices` has shape `[token_count, max_selected_entries]`;
`selected_counts` has shape `[token_count]`. Indices are request-local and unused
entries are `-1`.

The contiguous auxiliary cache has shape
`[batch_size, auxiliary_capacity, kv_heads_or_one, head_dim]`, with
`auxiliary_lengths[batch_size]` defining request boundaries. When
`auxiliary_kv_shared=1`, `auxiliary_key` supplies both K and V and
`auxiliary_value` must be absent. Auxiliary reads are never filtered by
`is_causal` or by the main-cache length; only `auxiliary_lengths` and the
auxiliary capacity bound them.

## Execution plan

The kernel is a bounded staged pipeline. Every stage is a separate compute pass,
and no stage truncates its candidate list.

1. **Prologue (reused from the WebGPU `PagedAttention` kernel).** Packed-QKV
   split, then rotary embedding of the current Q and K.
2. **Main-cache store.** When `slot_mapping` is present, a dedicated shader
   scatters the current K/V through it; otherwise the `PagedAttention`
   `block_table`-derived scatter is reused. A negative `slot_mapping` entry, or
   one beyond the cache, suppresses the write for that token. The `i32` value is
   range-checked before it is converted to `u32`.
3. **Token metadata.** A shader resolves
   `token_meta[t] = (batch_id, query_position, main_length, auxiliary_length)`
   on device from `cumulative_sequence_length`, `past_seqlens`, and
   `auxiliary_lengths`. Values are sanitized here (negatives clamped, ranges
   ordered, main lengths clamped to `max_num_blocks_per_seq * block_size`,
   auxiliary length clamped to capacity) so later stages can convert to `u32`
   without further checks. Clamping happens before any addition and the bounds
   stay below 2^30, so no caller-supplied length can overflow the `i32`
   arithmetic or turn into a candidate count large enough to hang the device.
   Positions at or beyond the block table's reach have no physical block, so
   clamping to it drops nothing the attention stages would have used.
4. **Partial attention.** One workgroup per `(token, head)` walks its candidate
   list and produces an FP32 online-softmax state
   `(accumulator[head_size], running_max, running_sum)`. Up to two partial states
   are produced: one over the paged main cache, one over the contiguous auxiliary
   cache.
5. **Finalize.** The partial states are merged with the standard online-softmax
   rule, which is exactly equivalent to one softmax over the union of their
   candidates, and the optional head sink seeds the merge with
   `(max = sink, sum = 1, accumulator = 0)`. A token with no valid candidate and
   no sink produces a zero row rather than `NaN`.

Selection data, block tables, and per-request lengths are read only by shaders.
The kernel never copies them to the host, so the op is usable under graph capture
and in continuous-batching runtimes. The only host-visible input is the optional
CPU `attention_metadata` tensor, which is declared `OrtMemTypeCPUInput`.

Cache outputs follow the WebGPU `PagedAttention` contract: `key_cache_out` and
`value_cache_out` must be both present or both absent. When present but not
aliased onto the cache inputs by the allocation planner, the input caches are
copied into them first (with a warning) so untouched slots survive the run.
Configure IO-binding to alias the cache buffers in production.

## Support matrix

| Feature | WebGPU support |
|---|---|
| Activation (`T`) | FP16 |
| Main cache (`T_CACHE`) | FP16 |
| Auxiliary cache (`T_AUX`) | Contiguous FP16 |
| Index/length type (`S`) | INT32 |
| `attention_mode` | `selected_only`, `local_plus_selected` |
| `selected_kv_source` | `main`, `auxiliary` |
| `auxiliary_cache_layout` | `contiguous` |
| Auxiliary K=V (`auxiliary_kv_shared=1`) | Supported |
| `slot_mapping`, including `-1` suppression | Supported |
| `is_causal`, `local_window_size` | Supported |
| `softcap` | Supported |
| RoPE (`do_rotary`, `rotary_interleaved`) | Supported, `rotary_offset` must be 0 |
| Attention sink (`head_sink`) | Supported in the joint softmax |
| Device-resident selection (no readback) | Yes |
| Joint FP32 softmax across both sources | Yes |

## Not supported

Each of these is rejected with an explicit error; none silently changes numerics.

| Feature | Status |
|---|---|
| BFloat16 activations or cache | Not registered; fails kernel lookup |
| INT8 main cache, `k_quant_type`/`v_quant_type` != `NONE`, `k_scale`/`v_scale` | `NOT_IMPLEMENTED` |
| Session-level KV cache quantization | `NOT_IMPLEMENTED` |
| QK-Norm (`q_norm_weight`/`k_norm_weight`) | `NOT_IMPLEMENTED` |
| `rotary_offset != 0` | `NOT_IMPLEMENTED` |
| `auxiliary_cache_layout` other than `contiguous` | Rejected when the kernel is created |
| `head_size > 512`, or a head size whose workgroup storage exceeds the device limit | `NOT_IMPLEMENTED` |
| Configurations needing more storage buffer bindings per stage than the device reports | `NOT_IMPLEMENTED` |
| An optional (raw-indexed) binding larger than `maxStorageBufferBindingSize` | `NOT_IMPLEMENTED` |

## Limitations

- The implementation is staged rather than fused: it writes FP32 partial softmax
  states to a scratch tensor of
  `token_count * num_heads * (head_size + 2)` floats and merges them in a second
  pass. The scratch size is validated against `maxStorageBufferBindingSize`.
- Storage-binding budgets are validated per stage, from the actual binding
  segments rather than from a tensor count: the WebGPU EP binds any buffer larger
  than `maxStorageBufferBindingSize` as several consecutive segments, so a large
  KV cache consumes more than one binding. The main-cache stage binds eight
  tensors, which is the guaranteed WebGPU minimum for
  `maxStorageBuffersPerShaderStage`, so any segmented buffer in that stage is
  rejected up front instead of failing during shader creation. Optional bindings
  (`selected_indices`, `selected_counts`, `auxiliary_value`, `auxiliary_lengths`,
  the partial scratch, `head_sink`) are read with raw indexing, which cannot
  address past the first segment, so they must each fit one binding.
- `cos_cache`/`sin_cache` lengths are not validated against the largest KV
  position actually reached, because that would require reading device-resident
  lengths on the host. WebGPU's robust buffer access makes an out-of-range read
  safe (it returns zero) rather than undefined.
- One workgroup handles one `(token, head)` pair and iterates candidates
  serially, so throughput is bounded by the longest candidate list. There is no
  decode-specialized split-K variant yet.
