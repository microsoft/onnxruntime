# CUDA DynamicSparseAttention grouped decode

`UseGroupedDecodeAttention` enables the split-K grouped decode attention kernel
for decode and multi-row speculative verification (`1 <= sequence_length <= 8`)
across both floating-point (`FP16`/`BF16`) and quantized (`INT8`) KV caches,
with `max_splits = 32` for quantized KV caches.

The INT8 grouped-decode and vector-staging reference tests cover one through
four query rows (`sequence_length in {1, 2, 3, 4}`).

## INT8 KV cache encoding

The INT8 main cache is `tensor(int8)`: symmetric signed storage with no zero point,
`q = clamp(rint(x / scale), -128, 127)` and `x = q * scale`, with one float32 scale per
`kv_head * head_size` channel (`key_scale`, `value_scale`). This is the same encoding as the 8-bit
KV cache of `GroupQueryAttention` (`group_query_attention_qdq.cuh`) and `PagedAttention` /
`SparsePagedAttention` (`paged_attention_impl.cu`), which also use `int8`. `uint8` is used by those
operators only for packed 4-bit caches (`kv_cache_bit_width=4`), so `DynamicSparseAttention` does not
accept `tensor(uint8)`.

## Validation status codes

`validation_status` (and the CUDA strict-validation host check enabled with
`ORT_DYNAMIC_SPARSE_ATTENTION_STRICT_VALIDATION=1`) uses these codes; the first error found wins,
except that a cursor error (9) is replaced by any selection/sequence/position/scale error.

| Code | Enumerator | Meaning |
|---|---|---|
| 0 | `kDynamicSparseAttentionValidationOk` | Valid. |
| 1 | `kDynamicSparseAttentionInvalidCount` | `selected_counts` outside `[0, max_selected]`. |
| 2 | `kDynamicSparseAttentionInvalidPadding` | Entries after `selected_counts` are not `-1`. |
| 3 | `kDynamicSparseAttentionNegativeIndex` | Active selected index is negative. |
| 4 | `kDynamicSparseAttentionIndexOutOfBounds` | Selected index outside the source sequence. |
| 5 | `kDynamicSparseAttentionDuplicateIndex` | Duplicate active selected index. |
| 6 | `kDynamicSparseAttentionNonCausalIndex` | Main selected index refers to a future key. |
| 7 | `kDynamicSparseAttentionInvalidSequenceLength` | `seqlens_k` incompatible with the sequence and cache capacity. |
| 8 | `kDynamicSparseAttentionInvalidPosition` | Rotary position outside the rotary cache. |
| 9 | `kDynamicSparseAttentionInvalidCommittedLength` | `committed_length` disagrees with `seqlens_k`, `sequence_length` or `cache_capacity`. |
| 10 | `kDynamicSparseAttentionInvalidCacheScale` | INT8 scale is not positive and finite. |

With strict validation enabled every non-zero code, including 9, fails the kernel with an
`INVALID_ARGUMENT` status even when `validation_status` is not connected. Without strict validation
the code is only reported through `validation_status`.

## CUDA dispatch switches

`ORT_DSA_INT8_VECTOR_STAGING` and `ORT_DSA_SHARED_SCALE_MERGE` are read once per process (cached in
function-local statics) when the grouped decode path is first launched, so set them before the first
operator invocation. Only `0` and `1` are accepted. Neither changes schemas or cache layouts.
`ORT_DSA_PREFILL_KERNEL` (default `1`) is read on every BF16 call.

Grouped decode is used when `1 <= sequence_length <= 8`, `num_heads / kv_num_heads >= 4` (and
divisible), `head_size == 256`, 16-bit `T` (FP16/BF16), more than 256 candidates
(`max_selected` plus the local window for `local_plus_selected`) and the grouped shared-memory
footprint fits the per-block limit. The kernel uses 12 query-head warps when the ratio is exactly 12
and 8 otherwise.

| Variable | Default | Applies to | Value | Kernels selected |
|---|---|---|---|---|
| `ORT_DSA_INT8_VECTOR_STAGING` | `1` | Grouped decode with an INT8 main cache | `1` | `GroupedDecodeDynamicSparseAttentionKernel<..., kQuantizedCache=true, kVectorizedStaging=true>`: 8-byte `uint2` cache loads with `float4` scale loads. |
| | | | `0` | Same kernel with `kVectorizedStaging=false`: scalar per-channel dequantization into shared memory. |
| `ORT_DSA_SHARED_SCALE_MERGE` | `1` | Grouped decode (INT8 and floating-point caches) when `split_count <= 32` (always for INT8, whose `max_splits` is 32; for floating-point caches `split_count = min(ceil(candidates / 64), ceil(128 / grouped_blocks))`) | `1` | `MergeGroupedDecodeDynamicSparseAttentionKernel<T, true>`: one `expf` per split shared across channels; a split is active when its partial sum is non-zero. |
| | | | `0` | `MergeGroupedDecodeDynamicSparseAttentionKernel<T, false>`: per-channel `expf` over all splits. |
| `ORT_DSA_PREFILL_KERNEL` | `1` | BF16, not grouped decode, `sequence_length >= 8`, `head_size == 256`, `selected_only` from the main cache, `smooth_softmax = 0`, SM80+ with enough opt-in shared memory | `1` | `DynamicSparseAttentionPrefillKernel<quantized>` tensor-core prefill. |
| | | | `0` | Generic fused/split or grouped prefill kernels. |

When `split_count > 32` (floating-point caches only) the generic
`MergeDynamicSparseAttentionKernel` is used regardless of `ORT_DSA_SHARED_SCALE_MERGE`. The tests
`Int8CacheGroupedDecodeAlternateSwitchesInFreshProcess_CUDA` re-run the INT8 grouped-decode and merge
cases in fresh processes with each non-default combination.
