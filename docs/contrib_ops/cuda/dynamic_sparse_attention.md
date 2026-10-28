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
`ORT_DSA_PREFILL_KERNEL` and `ORT_DSA_PREFILL_INT8_TC` (both default `1`) are read on every BF16 call.

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
| `ORT_DSA_PREFILL_INT8_TC` | `1` | The `ORT_DSA_PREFILL_KERNEL` conditions with an INT8 main cache, 16-byte-aligned `present_key` and `present_value`, an 8-byte-aligned `output` and enough opt-in shared memory for `dsa_prefill::kInt8SharedBytes` | `1` | `DynamicSparseAttentionPrefillInt8Kernel` ([dynamic_sparse_attention_prefill_int8.cuh](../../../onnxruntime/contrib_ops/cuda/bert/dynamic_sparse_attention_prefill_int8.cuh)): the gathered K/V rows stay INT8 in shared memory and are converted exactly to FP16 in registers for FP16 `mma` with FP32 accumulation. `key_scale` is folded into the query once per block and `value_scale` is applied to the output once. |
| | | | `0` | `DynamicSparseAttentionPrefillKernel<true>`: dequantizes every gathered tile to BF16 in shared memory. |

When `split_count > 32` (floating-point caches only) the generic
`MergeDynamicSparseAttentionKernel` is used regardless of `ORT_DSA_SHARED_SCALE_MERGE`. The tests
`Int8CacheGroupedDecodeAlternateSwitchesInFreshProcess_CUDA` re-run the INT8 grouped-decode and merge
cases in fresh processes with each non-default combination.

### Buffer alignment

The wide-load fast paths assume more alignment than a tensor's element type guarantees, and tensors
bound through IOBinding can start at any element offset of a larger allocation. The launcher therefore
checks the caller-provided buffers and uses an equivalent narrower path, with identical results, when a
buffer is misaligned:

| Fast path | Buffers that must be aligned | Fallback |
|---|---|---|
| INT8 grouped decode vector staging (`ORT_DSA_INT8_VECTOR_STAGING=1`) | `key_scale`, `value_scale`, `auxiliary_key`, `auxiliary_value`: 16 bytes; `present_key`, `present_value`: 8 bytes | Scalar per-channel staging (`kVectorizedStaging=false`). |
| Floating-point grouped decode and grouped prefill | Each staged K/V row: 16 bytes (checked per row in the kernel) | Element-wise copy into shared memory. |
| Tensor-core BF16 prefill (`ORT_DSA_PREFILL_KERNEL=1`) | `present_key`, `present_value`: 16 bytes (8 for INT8); `key_scale`, `value_scale`: 16 bytes; `output`: 4 bytes | Generic fused/split or grouped prefill kernels. |
| Tensor-core INT8-cache prefill (`ORT_DSA_PREFILL_INT8_TC=1`) | In addition to the row above: `present_key`, `present_value`: 16 bytes; `output`: 8 bytes | `DynamicSparseAttentionPrefillKernel<true>`. |

Buffers allocated by ONNX Runtime are 256-byte aligned, so only IOBinding with offset buffers takes a
fallback. The `*MisalignedBuffers_CUDA` tests cover each case.

### Benchmark

rc3 model (INT4), H200, clean machine, 5 repetitions, baseline noise about +-0.03 ms, DFlash2 S=8
decode ms/cycle at context 2K/8K/32K. Both defaults are the faster setting, so the switches remain
only as fallbacks:

| Setting | Decode vs default |
|---|---|
| `ORT_DSA_INT8_VECTOR_STAGING=0` | about +0.3 ms (+2%) slower |
| `ORT_DSA_SHARED_SCALE_MERGE=0` | about +0.05 to +0.1 ms (+0.3 to 0.5%) slower |

Prefill with the INT8 KV cache (24 query / 2 KV heads, head size 256, 12 full-attention layers), H200:
`DynamicSparseAttentionPrefillInt8Kernel` takes 2.15 ms per launch instead of 6.1 ms in a standalone
harness (8192 rows, 2048 candidates, 2.83x) and 2.8 ms instead of 7.9 ms per 8192-row launch in the
model. The previous kernel spent 2.8% of its instructions on `mma` and the rest on in-kernel
dequantization and index handling. Time to first token in the Qwen3.8-Flash-Next packages (one round)
drops by 4.0 to 4.8% at 2048, 8.4 to 9.5% at 8192 and 7.6 to 9.4% at 32768 prompt tokens; decode is
unchanged. The RMS error against an FP64 reference is 1.6e-3 (previous kernel 1.7e-3).
`ORT_DSA_PREFILL_INT8_TC=0` restores the previous kernel. The test
`DynamicSparseAttentionTest.Int8CachePrefillTensorCoreKernel_CUDA` compares both kernels with a
reference over head counts, candidate counts and invalid indices.

The corresponding indexer switch results are in
[../packed_sparse_attention_indexer.md](../packed_sparse_attention_indexer.md#benchmark).
