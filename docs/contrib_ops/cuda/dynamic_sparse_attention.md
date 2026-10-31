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
`ORT_DSA_PREFILL_KERNEL`, `ORT_DSA_PREFILL_INT8_TC` and `ORT_DSA_DECODE_INT8_TC` are read once at kernel
construction. `ORT_DSA_PREFILL_KERNEL` defaults to `1`. The two INT8 tensor-core switches default to `0`: they are
opt-in, and without them the operator runs the BF16-dequantizing prefill and the CUDA-core grouped decode, bit for
bit as before (see [Rollout](#rollout)). Session options `ep.cuda.dsa_prefill_kernel`, `ep.cuda.dsa_prefill_int8_tc`
and `ep.cuda.dsa_decode_int8_tc` take precedence over their corresponding environment variables. Each of them
accepts exactly the one-character string `0` or `1` (no whitespace, sign or leading zeros: `01`, `-0` and `true` are
rejected); changing a value requires a new session. The loaded prefill and decode specializations must have a PTX
target of SM80+; otherwise dispatch uses the generic fallback.
`ORT_DSA_DECODE_INT8_TC_TILES_PER_SPLIT` (session option `ep.cuda.dsa_decode_int8_tc_tiles_per_split`) is a
tuning and test knob with the same precedence: an integer in `[0, INT_MAX]`, `0` keeps the automatic split
heuristic. Invalid values of any of these options fail session initialization.

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
| `ORT_DSA_PREFILL_INT8_TC` | `0` | The `ORT_DSA_PREFILL_KERNEL` conditions with an INT8 main cache, 16-byte-aligned `present_key` and `present_value`, an 8-byte-aligned `output`, a softmax `scale <= 1` and enough opt-in shared memory for `dsa_prefill::kInt8SharedBytes` | `1` | `DynamicSparseAttentionInt8TcKernel<false>` ([dynamic_sparse_attention_prefill_int8.cuh](../../../onnxruntime/contrib_ops/cuda/bert/dynamic_sparse_attention_prefill_int8.cuh)): the gathered K/V rows stay INT8 in shared memory and are converted exactly to FP16 in registers for FP16 `mma` with FP32 accumulation. `key_scale` is folded into the query once per block (`Q' = Q * key_scale`, rounded once to FP16) and `value_scale` is applied to the output once. Scores and the running maximum are kept in the natural-log domain. A softmax `scale > 1` (the default is `1 / sqrt(head_size)`) uses the kernel below, because the rounding bound grows with it. The FP16 probability tile holds `2^15 * p`, so a candidate keeps its contribution down to `p = 2^-40` (plain FP16 flushes below `2^-25`); the row sum stays unscaled FP32. Prefill keeps a single FP16 probability tile, as the BF16-dequantizing kernel does with a coarser BF16 one. A block whose query or value scales leave the validated range is evaluated by an FP32 path (`AttendRowFp32`) instead of being clipped or rounded, which would break the cancellation between large terms; the conditions, their cost and how likely they are are described under [Numerics and fallbacks](#numerics-and-fallbacks). |
| | | | `0` | `DynamicSparseAttentionPrefillKernel<true>`: dequantizes every gathered tile to BF16 in shared memory. |
| `ORT_DSA_DECODE_INT8_TC` | `0` | Grouped decode (conditions above) with BF16, an INT8 main cache, `head_size == 256`, `selected_only` from the main cache (no `local_plus_selected`), `smooth_softmax = 0`, a softmax `scale <= 1`, at most `kMaxGroup` query heads per KV head, 16-byte-aligned `prepared_query`, `present_key`, `present_value`, `key_scale`, `value_scale` and attention workspace, SM80+ and enough opt-in shared memory for `dsa_prefill::kInt8SplitKSharedBytes` (75,264 B) | `1` | `DynamicSparseAttentionInt8TcKernel<true>`: the tensor-core pipeline of the row above, split over the 64-candidate tiles of each (row, KV head). Each CTA of the (row, KV head, split) grid writes an un-normalized partial output, the natural-log row maximum and the row sum in the layout of the grouped decode merge kernel, which is reused unchanged. The probability tile is an FP16 high part plus an FP16 residual (about 22 significant bits) and both multiply the converted V fragments, because the signed sum of large value codes can cancel to a value far below the rounding of one FP16 probability. A partial is written as `(acc * 2^-15) * value_scale`. The split count follows `ORT_DSA_DECODE_INT8_TC_TILES_PER_SPLIT` below. A CTA that meets one of the conditions of [Numerics and fallbacks](#numerics-and-fallbacks) evaluates its candidates in FP32 (`AttendRowFp32<true>`) and publishes an FP32 partial in the same layout, so the fallback also holds for the splits. |
| | | | `0` | The CUDA-core `GroupedDecodeDynamicSparseAttentionKernel`, bit for bit as before. |
| `ORT_DSA_DECODE_INT8_TC_TILES_PER_SPLIT` | `0` | `ORT_DSA_DECODE_INT8_TC=1` and the conditions above | `0` | Automatic: the smallest power-of-two number of 64-candidate tiles per split for which all `ceil(max_selected / 64) * rows * kv_heads` tiles fit one wave of resident CTAs (`rows = batch_size * sequence_length`; resident CTAs = SM count times the CTAs per SM that the kernel's shared-memory and register footprint allow, queried with `cudaOccupancyMaxActiveBlocksPerMultiprocessor`: 396 on an H200). The split count is `ceil(tiles / tiles_per_split)`, at most 32 and never more than the grouped decode workspace was sized for. |
| | | | `N > 0` | A requested `N` tiles per split instead of the automatic choice. The 32-split cap, the workspace size and the even distribution of the tiles over the splits still apply, so the number of tiles of a split can exceed `N` (8192 candidates are 128 tiles: `N = 1` asks for 128 splits and runs 32 splits of 4 tiles). Used by the tests to force many or few splits. |

The decode time is not flat in the tiles per split: with too few splits the CTAs are long and SMs stay idle, with too
many the grid spills into a second, partly filled wave. On an H200, 8 rows and 2048 candidates (16 pairs of row and KV
head with 32 tiles each), kernel plus merge,
`benchmark_dynamic_sparse_attention.py --rows 8 --tiles-per-split 0 1 2 4 8 16`:

| Tiles per split | 1 | 2 (automatic) | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Splits, CTAs | 32, 512 | 16, 256 | 8, 128 | 4, 64 | 2, 32 |
| Time | 18.8 us | 13.2 us | 12.7 us | 17.7 us | 29.2 us |

The automatic rule is a heuristic, not the optimum. Over 20 shapes on the H200 (1, 2, 4 and 8 rows, 512 to 8192
candidates, context 8192, each swept over 1, 2, 3, 4, 6, 8 and 16 tiles per split) it is on average 1.8% and at most
14.8% slower than the best swept value (4 rows and 2048 candidates: 11.9 us against 10.4 us at 2 tiles per split). The
previous target of about 384 CTAs is 2.7% and 14.2% slower on the same points: with 396 resident CTAs the two rules mostly
agree on an H200, and the larger shapes gain from the power-of-two choice (8 rows, 8192 candidates: 0.2% against 6.7%).
The rule matters when fewer CTAs are resident. With the kernel's shared memory padded on the H200 so that one or two CTAs
fit an SM (132 or 264 resident CTAs; an ad hoc harness that is not part of this change; 1 to 32 rows, 2048 candidates)
the rule is on average 0.4% (at most 2.4%) and 3.1% (13.4%) slower than the best swept value, where the 384-CTA target is
23.8% (44.9%) and 14.4% (38.4%) slower. A device with 108 SMs and two resident CTAs per SM (the 75,264 B of shared memory
allow two on an A100) is expected to choose 4 tiles per split, 8 splits, for 8 rows and 2048 candidates, the faster
setting in an A100 measurement reported for this change (22.1 us against 25.9 us at 16 splits). The A100 has not been
measured with this document's script; `ORT_DSA_DECODE_INT8_TC_TILES_PER_SPLIT` tunes other devices.

When `split_count > 32` (floating-point caches only) the generic
`MergeDynamicSparseAttentionKernel` is used regardless of `ORT_DSA_SHARED_SCALE_MERGE`. The tests
`Int8CacheGroupedDecodeAlternateSwitchesInFreshProcess_CUDA` re-run the INT8 grouped-decode and merge
cases in fresh processes with each non-default combination.

### Numerics and fallbacks

The INT8 tensor-core kernels multiply `Q' = fp16(Q * key_scale)` with the exactly converted INT8 K rows and the FP16
probability tile with the exactly converted INT8 V rows, in FP16 `mma` with FP32 accumulation. They therefore differ from
the CUDA-core kernels by the rounding of `Q'` and of the probability tile and by the FP32 summation order, so outputs are
close but not bit-identical to the previous kernels. A CTA (one query row and KV head, that is a group of up to 16 query
heads over the row's candidates) that cannot bound that rounding does not clip or round: it evaluates its heads in FP32
on the CUDA cores (`AttendRowFp32`), which is exact. The decision is uniform across the CTA and the FP32 path starts
from the inputs, so the result of a CTA is either the tensor-core result within the bounds below or the FP32 result.

| A CTA takes the FP32 path when | Bound that it enforces | RC model (Qwen3.8-Flash-Next DSA layers) |
|---|---|---|
| An element of `Q * key_scale` is not finite or its magnitude is above 65504 | `Q'` fits FP16 | Does not happen |
| For one of its query heads, `eps = scale * 128 * sum_d abs(Q'_d - fp16(Q'_d))` is above `kInt8TcQueryErrorBudget` (0.125) | Rounding `Q'` moves any logit of that head by at most `eps` nats (`abs(K_int8) <= 128`), so a probability by at most a factor `exp(eps)`. A query whose large terms cancel to a small logit exceeds the budget by orders of magnitude, an ordinary query does not | `eps` is 0.015 on average and 0.022 at most per head (table below) |
| A `value_scale` is not finite | The FP16 accumulator result is meaningful | Does not happen |
| Some channel has `abs(value_scale) > 2^16` and a valid candidate of one head is more than 20 nats (`kInt8TcTailDepth`) below the running maximum | A probability below `2^-29` is a subnormal of the `2^15 * p` tile and below `2^-40` it is zero. For `abs(value_scale) <= 2^16` the lost contribution is below `2^-17` of one value code; above it the check decides per CTA instead of sending every CTA to the FP32 path | The largest `value_scale` is 0.285: the check is never armed |

The scores and the running maximum are kept in the natural-log domain and only the difference to the new maximum is
scaled by `log2(e)` for `ex2`, so the difference of two large, nearby logits is not rounded away. Prefill keeps a single
FP16 probability tile holding `2^15 * p`, as the BF16-dequantizing kernel does with a coarser BF16 tile; decode adds an
FP16 residual tile, because the signed sum of large value codes can cancel to a value far below the `2^-11` rounding of
one FP16 probability.

The bound `eps` is proportional to the magnitude of the query. Monte Carlo over the 12 DSA layers of the RC model
(their `q_norm` weights and key scales, 4000 rows per layer and KV head, scale `1/16`, unit-RMS Gaussian queries with
partial rotary embedding rounded to BF16, then multiplied by `m`; `m = 1` is what the model produces):

| `m` | mean `eps` of a head | largest `eps` of a head | CTAs over the budget (12 heads per CTA) |
|---|---|---|---|
| 1 | 0.015 | 0.022 | 0 |
| 2 | 0.031 | 0.045 | 0 |
| 4 | 0.061 | 0.088 | 0 |
| 8 | 0.122 | 0.179 | 69% |
| 16 | 0.245 | 0.356 | 100% |

The budget is therefore about 8 times the magnitude the model produces. Heavy-tailed queries (Student-t with 4 degrees
of freedom) reach 48% of the CTAs at `m = 8`, queries whose channel variances differ by a log-normal factor (sigma 1)
about 1%. Workloads whose queries are 8 times larger than those of the RC model, or whose cache has
`abs(value_scale) > 2^16` and wide logit spreads, should leave the switches at `0`: the FP32 path is slower than the
previous kernels, as the measurements below show. It stays accurate: with queries 8 times larger the relative RMS error
against FP64 is 1.7e-3 (the previous kernels: 8.5e-3 for prefill, 1.7e-3 for decode).

Cost of the FP32 path on an H200 (`benchmark_dynamic_sparse_attention.py`; 24 query / 2 KV heads, head size 256,
2048 selected candidates, context 8192; attention-kernel time, previous kernel: BF16-dequantizing prefill / CUDA-core
decode):

| Data | Rows | Previous kernel | Tensor-core kernel | Tensor core vs previous |
|---|---|---|---|---|
| Queries like the RC model | 128 | 0.173 ms | 0.063 ms | 2.7x faster |
| | 8192 | 7.82 ms | 2.83 ms | 2.8x faster |
| Queries 8 times larger (`--query-std 11.2`, every CTA over the budget) | 128 | 0.172 ms | 3.68 ms | 21x slower |
| | 8192 | 7.82 ms | 135.6 ms | 17x slower |
| | 8 (decode) | 73.6 us | 240 us | 3.3x slower |
| `abs(value_scale)` 24k to 360k (`--value-scale-multiplier 1.2e7`), logit standard deviation about 2.4 nats: 10 to 12% of the CTAs have a candidate deeper than 20 nats | 128 | 0.172 ms | 3.61 ms | 21x slower |
| | 512 | 0.577 ms | 3.90 ms | 6.8x slower |
| | 2048 | 2.288 ms | 6.17 ms | 2.7x slower |
| The same value scales, logit standard deviation about 1.3 nats (the cache holds only rows appended by the call): no CTA is deeper than 20 nats | 8192 | 7.78 ms | 2.86 ms | 2.7x faster |
| The same value scales | 8 (decode) | 73.6 us | 13.3 us | 5.6x faster |

A CTA on the FP32 path needs about 3.6 ms for 2048 candidates, so the kernel time is set by the slowest CTA of a launch
with one wave of CTAs and grows with the fraction of such CTAs for larger launches. A decode CTA covers fewer candidates
(64 to 128 at the automatic split on an H200 for 2048 candidates and 1 to 8 rows), so a candidate 20 nats below the
running maximum is rarer and the decode row of the last group shows no slowdown. With
`ORT_DSA_DECODE_INT8_TC_TILES_PER_SPLIT=16` (1024 candidates per CTA) the same data takes 1.8 ms instead of 13 us, with
the same RMS error (1.7e-3). The previous design of this kernel sent every CTA to the FP32 path as soon as one channel
had `abs(value_scale) > 2^16` (8192 rows: 2.76 ms became 136 ms), which the check above replaces; it still applies to
queries above the budget, as the second group of rows shows. The unit tests exercise every condition of the table
(`Int8TensorCoreQueryFoldingRoundingError_CUDA`, `Int8TensorCoreDecodeSignedValueCancellation_CUDA`,
`Int8TensorCoreLargeCommonLogit_CUDA`, `Int8TensorCoreTinyValueScale_CUDA`,
`Int8TensorCoreLargeValueScaleRandom_CUDA`, `Int8TensorCoreRandomQueryFolding_CUDA`,
`Int8PrefillQueryBeyondHalfRange_CUDA`, `Int8DecodeQueryBeyondHalfRange_CUDA`) against an FP64 host reference.

### Rollout

`ORT_DSA_PREFILL_INT8_TC` and `ORT_DSA_DECODE_INT8_TC` are opt-in and independent, so prefill and decode can be enabled
one at a time. The end-to-end accuracy measured for an earlier revision of these kernels (MMLU-Pro, 2400 items,
pooled difference +0.08 percentage points, 95% confidence interval -0.60 to +0.77) predates the numerics described above
and does not cover the final revision. The end-to-end timings in [Benchmark](#benchmark) are preliminary for the same
reason. Making a switch the default needs paired accuracy runs of the final revision with that switch toggled alone,
including long contexts, and a stated bound on the degradation. Packages that qualify the kernels enable them in their
session options (`ep.cuda.dsa_prefill_int8_tc` and `ep.cuda.dsa_decode_int8_tc` set to `1`).

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
| Tensor-core INT8-cache prefill (opt-in, `ORT_DSA_PREFILL_INT8_TC=1`) | In addition to the row above: `present_key`, `present_value`: 16 bytes; `output`: 8 bytes | `DynamicSparseAttentionPrefillKernel<true>`. |
| Tensor-core INT8-cache decode (opt-in, `ORT_DSA_DECODE_INT8_TC=1`) | `prepared_query`, `present_key`, `present_value`, `key_scale`, `value_scale`, attention workspace: 16 bytes | The CUDA-core grouped decode kernel. |

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

The INT8 tensor-core kernels (opt-in, see [Rollout](#rollout)) are measured at the operator level with
[benchmark_dynamic_sparse_attention.py](../../../onnxruntime/test/python/transformers/benchmark_dynamic_sparse_attention.py)
against the kernels that they replace (the BF16-dequantizing prefill and the CUDA-core grouped decode). The script builds
a single-node `DynamicSparseAttention` model per case (24 query / 2 KV heads, head size 256, INT8 cache, scale 1/16,
2048 selected candidates per row: the shapes of the 12 full-attention layers of Qwen3.8-Flash-Next) and one session per
kernel configuration, because the `ep.cuda.dsa_*` options are read when the kernel is created. The time is the GPU time
of the kernels that one `Run` launches, taken from the CUPTI-based torch profiler, so host launch overhead does not
enter: "attention" sums the `DynamicSparseAttention` kernels including the decode merge kernel, "node" all GPU activity
of the node (cache append, query preparation, ...). Each round measures every configuration once, the order alternates
between rounds, the tables show the median of 7 rounds, and the speedup is the median of the paired ratios of the rounds.
The error is the RMS error of the BF16 output relative to the RMS of an FP64 reference over 64 evenly spaced rows. The
data are synthetic: BF16 queries with standard deviation 1.4 (the queries of the RC model after `q_norm` have a per-head
RMS of 1.3 to 1.6), INT8 cache codes with standard deviation 40, and key and value scales drawn from the ranges of that
model (`--rc-model` reads the scales of a model's node instead). H200, CUDA 13.0, the automatic tiles per split:

```bash
python benchmark_dynamic_sparse_attention.py --rows 1 8 --context 2048 8192 32768 --tiles-per-split 0 1 2 4 8 16
python benchmark_dynamic_sparse_attention.py --rows 128 2048 8192 --context 8192 32768
```

Decode (`sequence_length <= 8`):

| Rows | Context | Attention, previous | Attention, tensor core | Speedup | Node, previous | Node, tensor core | RMS error, previous / tensor core |
|---|---|---|---|---|---|---|---|
| 1 | 2048 | 31.1 us | 9.5 us | 3.30x | 34.1 us | 12.5 us | 1.59e-3 / 1.62e-3 |
| 1 | 8192 | 31.2 us | 9.5 us | 3.30x | 34.2 us | 12.5 us | 1.55e-3 / 1.57e-3 |
| 1 | 32768 | 31.2 us | 9.5 us | 3.30x | 34.1 us | 12.5 us | 1.67e-3 / 1.72e-3 |
| 8 | 2048 | 73.6 us | 13.2 us | 5.58x | 76.7 us | 16.5 us | 1.64e-3 / 1.69e-3 |
| 8 | 8192 | 73.6 us | 13.3 us | 5.57x | 76.8 us | 16.6 us | 1.62e-3 / 1.66e-3 |
| 8 | 32768 | 73.7 us | 13.2 us | 5.57x | 77.0 us | 16.6 us | 1.68e-3 / 1.73e-3 |

Prefill:

| Rows | Context | Attention, previous | Attention, tensor core | Speedup | Node, previous | Node, tensor core | RMS error, previous / tensor core |
|---|---|---|---|---|---|---|---|
| 128 | 8192 | 0.173 ms | 0.063 ms | 2.72x | 0.179 ms | 0.070 ms | 4.19e-3 / 1.71e-3 |
| 128 | 32768 | 0.191 ms | 0.064 ms | 2.99x | 0.199 ms | 0.072 ms | 4.25e-3 / 1.70e-3 |
| 2048 | 8192 | 2.287 ms | 0.832 ms | 2.75x | 2.350 ms | 0.894 ms | 4.25e-3 / 1.71e-3 |
| 2048 | 32768 | 2.294 ms | 0.832 ms | 2.75x | 2.356 ms | 0.895 ms | 4.21e-3 / 1.70e-3 |
| 8192 | 8192 | 7.818 ms | 2.830 ms | 2.76x | 8.057 ms | 3.069 ms | 2.69e-3 / 1.66e-3 |
| 8192 | 32768 | 8.877 ms | 3.200 ms | 2.77x | 9.116 ms | 3.439 ms | 4.29e-3 / 1.71e-3 |

The previous prefill kernel spent 2.8% of its instructions on `mma` and the rest on in-kernel dequantization and index
handling, and the previous grouped decode kernel was instruction-issue and shared-load bound on its per-candidate
serial chain. The decode time does not depend on the context length (the selected candidates are gathered). The error
includes the rounding of the BF16 output, whose floor is about 1.5e-3 relative to the RMS: both decode kernels and the
tensor-core prefill are near it, and the BF16-dequantizing prefill, which rounds the dequantized keys and values and the
probabilities to BF16, has 1.6 to 2.5 times the error of the tensor-core prefill. The tensor-core kernels accumulate Q.K
and P.V in a different FP32 order, so their outputs are close but not bit-identical to the previous kernels.

End-to-end numbers are preliminary. In the Qwen3.8-Flash-Next packages, an earlier revision of these kernels
(one round of consecutive A/B runs, not alternated) shortened the time to first token by 4.0 to 4.8% at 2048, 8.4 to 9.5%
at 8192 and 7.6 to 9.4% at 32768 prompt tokens, the speculative-decoding cycle (S = 8) by 4.6 to 6.8% at 2K, 8K and 32K
context, and the 8192-row prefill launch from 7.9 to 2.8 ms. They have not been reproduced for the current revision of
the kernels, whose numerics changed (see [Rollout](#rollout)).

`ORT_DSA_PREFILL_INT8_TC=0` and `ORT_DSA_DECODE_INT8_TC=0` (the defaults) run the previous kernels exactly. The tests
`Int8CachePrefillTensorCoreKernel_CUDA` and `Int8CacheDecodeTensorCoreKernel_CUDA` compare the tensor-core kernels with a
reference over head counts, candidate counts, splits and invalid indices; `Int8DecodeQueryBeyondHalfRange_CUDA` covers
the FP32 fallback of the splits, `Int8TensorCoreDecodeTilesPerSplitCaps_CUDA` the split caps, and
`PrefillSessionOptions_CUDA` and `DecodeSessionOptions_CUDA` the switches, the tuning knob and invalid values.
`Int8TensorCoreLargeSoftmaxScale_CUDA` (the `scale > 1` fallback) and
`Int8TensorCoreTinyProbabilityLargeValueScale_CUDA` (the FP16 probability range) cover prefill and decode, and the tests
listed under [Numerics and fallbacks](#numerics-and-fallbacks) the guards.

The corresponding indexer switch results are in
[../packed_sparse_attention_indexer.md](../packed_sparse_attention_indexer.md#benchmark).
