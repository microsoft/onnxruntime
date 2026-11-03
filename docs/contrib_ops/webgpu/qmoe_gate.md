# QMoE single-token routing on WebGPU

The WebGPU implementation of `com.microsoft::QMoE` selects the top `k`
experts for each token. See the [operator documentation](../../ContribOperators.md)
for the input, output, quantization, and routing contracts. This page describes
the single-token gate implementation; the operator schema is unchanged.

## Selection and normalization

When there is one token and `normalize_routing_weights=1`, the gate uses a
workgroup-wide reduction to select one expert per round. It ranks higher
router logits first, breaking ties by lower expert index, and invalidates
each selected lane before the next round. A lane that has already been
selected cannot win again, even if every remaining logit is `-inf`.
The gate writes selected indices in rank order and leaves unselected routing
weights at zero.

With optional `router_weights`, the selected weights are normalized by their
sum. Without them, the gate applies a stable softmax over the selected logits.
When `normalize_routing_weights=0`, the existing full-sort path is retained:
the softmax denominator includes **all** experts, not only the selected `k`.
Changing that denominator would change the operator's results.

One workgroup has one lane per expert. The expert count must fit both the
adapter's `maxComputeWorkgroupSizeX` and
`maxComputeInvocationsPerWorkgroup` limits; larger nonempty inputs are
rejected rather than dispatched with invalid workgroup dimensions. The
WebGPU baseline for both limits is 256, so a 512-expert configuration
requires an adapter that reports higher limits. Multi-token routing uses
the existing gate implementation and is not changed by this optimization.

The gate supports both FP16 and FP32 router logits. It selects experts
independently of whether their downstream weights are integer-quantized or
block-FP8. This change does not introduce native FP8 matrix arithmetic or
alter the expert matmul kernels.

## Validation and performance

Focused GPU tests cover odd expert counts, ties, `k` equal to the expert
count, negative-infinity logits, nonnormalized routing, FP16/FP32 gate
indices, and adapters with smaller workgroup limits. In particular,
`[0, -inf, -inf]` with `k=2` selects experts `[0, 1]`, not an invalid
sentinel index.

On H200/Vulkan, a synthetic single-token block-FP8 QMoE graph with 256
experts and top-4 routing measured 7.09 ms/run on the merged FP8 QMoE
baseline and 5.56 ms/run on an integrated branch containing this gate
change (medians of three 100-run trial averages). Trial ranges overlapped:
5.71-7.12 ms versus 5.46-5.89 ms. These are whole-graph timings, not
isolated gate-kernel timings or a full-model speedup claim.

## Local H200 performance experiments

### INT4 RTN prefill routing

The primary H200 target is the original INT4 RTN export, not NVFP4.
`ORT_WEBGPU_QMOE_INT4_PREFILL=1` is a default-off experiment that keeps
routing GPU-resident for up to 16 tokens and 256 token-expert pairs.
It requires normalized routing, implicit zero points, and 4-bit FC1/FC2
and optional FC3 weights. Other cases retain the existing general path.
Original weights, scales, and quantization are unchanged.

The integer matmul helper expects one activation row per routed expert.
An input-expansion kernel therefore replicates each token for its selected
experts. Selected expert indices are sorted after routing weights are
computed, retaining the general path's ascending-expert accumulation order.
Single-token behavior and the FP8/NVFP4 routing paths are unchanged.
Matmul dispatch choices can still change rounding and require numerical
and model-quality validation.

The comparison uses Generator without MTP and non-paged attention, with
the original explicit CPU embedding layer. This is not a reproduction of
Tianlei's NVFP4 CUDA Engine package. GPU profiling and unprofiled
2K/8K/32K context measurements are separate stages.

On H200, 20 independent reference fixtures passed 720 executions with
full WebGPU validation and CPU fallback forbidden. Coverage includes
FP16/FP32, fused SwiGLU, separate FC3, ties, the 16-token/256-pair limits,
and larger-batch, nonnormalized-routing and explicit-zero-point fallbacks.
Maximum absolute errors were 0.000183105 for FP16 and below 1.5e-8 for FP32.
The FP32 fixtures use random exactly INT8-quantizable activations, avoiding
the unchanged general path's DP4A activation-quantization error without
relaxing reference tolerances. FP16 fixtures use arbitrary random inputs.
These are operator checks, not full-model throughput or quality results.

The original layer-0 INT4 payload (512 experts, top-10, 16 tokens,
hidden size 2560 and intermediate size 640) also passed 36 reference
executions without changing weight bytes or canonical tolerances.
Maximum absolute error was 3.05176e-5. Relative RMSE was 0.1686% for
the control and 0.1683% for the prefill candidate. An exploratory
2e-6 absolute bound failed near zero even on the unchanged control;
it is not the canonical FP16 acceptance bound. Timings collected during
model loading are diagnostic only.

The legacy INT4 graph combines growing indexer caches, with no scalar
`past_sequence_length` input, and fixed-capacity attention KV caches.
Disabling `past_present_share_buffer` globally is not valid: attention then
rejects the zero-capacity past cache during prefill. The local GenAI fix
instead retains shared attention KV and selects growing indexer buffers
when the scalar indexer-length contract is absent. Fixed-shape indexers
still require that scalar and retain the existing error when it is absent.
A CPU-only graph preserving the original cache metadata validates both
legacy search settings and the fixed-indexer rejection against an isolated
GenAI library. Full-model execution remains a separate qualification step.
The isolated decoder also uses one intra-op and one inter-op host thread
to avoid unnecessary CPU thread-pool contention. The explicit CPU
embedding session remains separate.

### Cooperative INT4 projection experiment

`ORT_WEBGPU_QMOE_INT4_COOPERATIVE=1` is a separate default-off experiment
for symmetric, implicit-zero-point INT4 block-32 weights. It reuses the
software block-quantized projection's 64-lane K reduction with float32
accumulation, but decodes signed integer values and ordinary floating-point
block scales, not NVFP4 codes or FP8 scales. It does not change weight bytes,
require native FP4/FP8 support, or modify Dawn.

It applies only to the GPU-resident indirect routing path: single-token
decode, or eligible prefill when the INT4 prefill flag is also enabled.
Dimensions must be divisible by 32, weight offsets must fit uint32, and
output columns must fit the adapter's dispatch limit. Other cases retain
the existing integer matmul path. Token rows are addressed directly rather
than expanded for this kernel. Float32 reduction changes arithmetic relative
to the existing FP16/DP4A kernels; numerical, device-timing, and full-model
qualification are required before claiming an improvement.

The integrated candidate passed 1,440 reference executions: 20 synthetic
INT4 fixtures, 18 preserved NVFP4 fixtures, and original-weight INT4
single-token/16-token fixtures, each in four modes over nine executions.
For the actual weights, cooperative reduction lowered relative RMSE
from 0.1597% to 0.0684% for one token and from 0.1686% to 0.0582% for
16 tokens. It did not outperform the existing indirect integer prefill
path in the full-validation host diagnostic, so it remains disabled in
the primary comparison. The harness enables it for a candidate only when
explicitly requested through its environment, and records paired output
agreement separately from throughput and model-quality qualification.

### Dense projection N-tail experiment

The partial original-INT4 prefill profile also identifies 324-column
hyper-connection projections with K=10240 on the scalar wide-tile path.
`ORT_WEBGPU_MATMUL_NBITS_SGM_N_TAIL=1` is a default-off experiment allowing
FP16 block-32 dense projections with N divisible by four, but not by 64,
to use an available 16x16x16 FP16-input/FP32-result subgroup-matrix configuration. Existing
batch, K-alignment, accuracy and minimum-M requirements still apply.
Other configurations retain the existing fallback. Output strides not
divisible by 16 use the bounds-checked staging path. FP32 matrix results
are staged and converted to FP16 outputs; FP16 accumulation failed the
strict long-K oracle and is not used for this experiment.

This does not change quantized weights or enable native FP8. It targets
prefill, not single-token decode. Independent numerical fixtures and
unprofiled device measurements are required before enabling it for a
model benchmark; it is not enabled in the current full-model profile.

### Small-upload transport experiment and CPU residency

The first completed diagnostic profile separates substantial host-side
stalls from device work: even a 16-byte host-to-device metadata copy can
take several seconds. `ORT_WEBGPU_SMALL_UPLOAD_WRITE_BUFFER=1` is a
default-off experiment using Dawn queue writes for uploads up to 1 MiB,
instead of allocating a mapped staging buffer per copy. It submits
previously encoded work before writing and zero-pads unaligned copies.
Capture recordings and larger uploads retain the existing path.
Dynamic-input ordering passed 220 exact-output GPU executions, alternating
both modes, including odd FP16 element counts and FP32 transfers below,
at and above the size threshold. Profiles confirm WebGPU execution.
Full-model timing remains to be measured.

The unchanged original model also has a 51.2 GB CPU FP8 PLE table; this is
not native WebGPU FP8. Random single-token CPU lookups initially took
63.87 ms, versus 0.111 ms after controlled read-only file-cache warming.
The corresponding 16-token lookup changed from 917.12 ms to 1.429 ms.
These are CPU operator diagnostics, not model TPS or a thread-count
comparison. Warm measurements had no major page faults. Future model
timings must disclose CPU file-cache residency and exclude warming.

Original-weight dense fixtures use synthetic activations, not captured
model activations. The unchanged baseline exceeds the existing reference
bound for some K=10240 inputs drawn from [-0.25, 0.25]; those failures
are retained separately. FP32-accumulating tensor-core qualification
restores that input range with unchanged numerical tolerances.
The unchanged M1 fallback uses [-1/32, 1/32]; it is not a new tensor-core
path. No fixture uses captured model activations or establishes model quality.

### Small synchronous readback experiment

`ORT_WEBGPU_SMALL_READBACK_CACHE=1` separately enables bounded reuse of one
unmapped staging buffer for synchronous readbacks up to 1 MiB. It targets
allocation/free overhead rather than shader execution. Capture and larger
readbacks retain the existing path. A dedicated mutex protects the staging
buffer through copy, map, host read and unmap; failed operations do not return
the buffer to the cache. Smaller subsequent reads map and access only their
requested aligned range. This default-off experiment requires exact-output
and reuse-boundary qualification before full-model use. The standalone
qualification passed 300 exact-output executions across all four upload/readback
flag combinations, including grow/shrink, odd FP16 lengths and the 1 MiB FP32
boundary, plus 16 exact concurrent-Session executions. Profiling confirms
WebGPU Add/Cast execution. These are correctness checks, not a full-model
speedup result.

### Legacy QSA long-context score-cache experiment

`ORT_WEBGPU_QSA_GLOBAL_SCORE_CACHE=1` is a separate, default-off experiment
for legacy `SparseAttentionIndexer`, not the packed indexer or Dawn FP8.
Once the existing 768-score or 3072-token workgroup cache is exceeded,
eligible requests use bounded GPU scratch buffers for the full visible
token map and one score per compressed block. The candidate also
selects this path within those caches for at least two query heads once
the total sequence contains a full compressed block. This avoids rescanning the
mask during each pooled-key load and recomputing every score for each
selected rank. Score arithmetic, stable rank order, mask handling and
tail-token semantics are unchanged.

The current candidate uses three ordered dispatches: stable visible-token
compaction; one workgroup per compressed block to reuse pooled, normalized,
rotated keys across query heads; and stable top-k selection. RMS,
dot-product and head-summation order remain scalar and unchanged. Each
visible-map row reserves one extra uint32 for its compacted count.

Storage/index/binding and dispatch limits are checked before selecting the
scratch path. It currently supports head sizes up to 128 and at most 64 query
heads. Requests outside these eligibility conditions retain the existing path.
An earlier combined-workgroup version passed 144 exact reference executions,
but realistic 32K scoring still took about 874 ms on H200, motivating the
staged workgroup-reuse implementation. Full validation covers mask holes,
high-word INT64 masks, empty and omitted masks, stable ties, packed QK,
FP16/FP32, partial/full rotary caches and 8K/32K cases: 22 fixtures passed
198 exact reference executions.

A subsequent same-H200 control/candidate/candidate/control comparison passed
another 72 exact reference executions. With the first execution of each
fixture in each process excluded, pooled device-time medians across 16
executions per variant were:

| Context | Previous combined-workgroup experiment | Staged experiment |
| --- | ---: | ---: |
| 8K | 163.5285 ms | 0.337 ms |
| 32K | 656.512 ms | 0.9935 ms |

These are sums of all indexer GPU dispatches, including concatenation, for
synthetic batch-2, single-query, 16-head, width-128, compress-32 fixtures
with mask holes and budget 2048. The control is the previous default-off
experiment, not upstream main. These are not full-model speedups or token
rates; throughput and quality remain separate qualification gates.

The original INT4 model uses a different contract: batch 1, four query heads,
head width 128, rotary width 32, compression ratio 4, and token budget 2048.
Extending scratch eligibility to its short contexts passed 351 exact-reference
executions across 39 fixtures. Selection remained the dominant device cost.

The subsequent candidate incrementally merges sorted tiles, retaining a
power-of-two top-k prefix in workgroup memory. Ordering remains score-descending,
then original-index-ascending. Padding, mask holes, tails, and the existing
first-score-NaN behavior are preserved. Sorting requires top-k at most 1024,
sufficient workgroup storage, and a lower estimated barrier cost than rank
reduction; otherwise parallel rank reduction is retained.

This candidate passed 369 exact-reference executions across 41 fixtures with
full GPU validation, including non-power-of-two budgets and stable ties.
Complete related test translation units passed syntax checks; the full GTest
binary was not executed. Actual-contract device medians, excluding the first
execution per fixture, were:

| Context | Query tokens | Rank-reduction total | Sorted-selection total | Sorted-selection phase |
| --- | ---: | ---: | ---: | ---: |
| 2K | 1 | 1.379 ms | 0.1585 ms | 0.086 ms |
| 2K | 16 | Not measured here | 0.318 ms | 0.157 ms |
| 8K | 1 | 3.2095 ms | 0.503 ms | 0.322 ms |
| 32K | 1 | 9.839 ms | 1.8935 ms | 1.263 ms |

These are sequential operator diagnostics, not a new counterbalanced model
comparison. The earlier staged full-model candidate reached approximately
2.75 decode tok/s at 2K but generated severely repetitive output; it is not
quality-qualified. The sorted candidate's separate unprofiled model run used
the original INT4 graph and weights, Generator without MTP, chunk 16, and
256 output tokens. One warmup per context was excluded:

| Context | Completed measured requests | Median TTFT | Median decode tok/s |
| --- | ---: | ---: | ---: |
| 2K | 3 | 21.6435 s | 15.1938 |
| 8K | 3 | 93.3773 s | 14.2612 |
| 32K | 0 | Not available | Not available |

The 32K warmup did not produce its first token before the whole run's
3600-second cap; the owned container was stopped and GPU memory released.
All completed requests emitted 256 tokens, but output was severely repetitive,
so these are timing diagnostics, not quality-qualified throughput. Both
prompts shared with the previous 2K run preserved all output token IDs exactly.
For its single shared measured prompt, decode increased from 2.7477 to
15.1938 tok/s and TTFT decreased from 156.5254 to 21.6756 s; this is not a
counterbalanced comparison or evidence of correct generation. Neither the
100-tok/s target nor half-CUDA performance has been demonstrated.

The 32K investigation found a dispatch eligibility bug: at 16 query tokens
and compression ratio 4, the score workgroup count reaches 65536 at a
total context of 16384. Comparing that flattened count with the per-dimension
limit incorrectly disabled scratch scoring and restored the slow legacy path.
The candidate now relies on ORT's existing multidimensional dispatch
normalization, retaining uint32 and binding-size checks. The score shader
uniformly exits padded workgroups before accessing scratch or reaching barriers.

Eight additional independent fixtures cover 16-query FP16/FP32, masked and
maskless, packed and unpacked cases at total contexts 16380, 16384, 16388,
and 32784. All 49 fixtures passed 441 exact-reference executions with GPU
validation. The corresponding packed-FP16 total device medians were 1.697,
1.6995, 1.7695, and 3.453 ms. Profiles confirm map/score/sort dispatches,
including padded multidimensional grids. These are operator diagnostics;
the corrected full-model run completed all 12 requests normally in 2501.7 s,
using the same original-INT4 protocol with 32K measured first:

| Context | Measured requests | Median TTFT | Median decode tok/s |
| --- | ---: | ---: | ---: |
| 2K | 3 | 21.5163 s | 15.3106 |
| 8K | 3 | 93.1930 s | 14.1491 |
| 32K | 3 | 417.3638 s | 11.1631 |

All requests emitted 256 tokens. All eight 2K/8K warmup and measured requests
preserved the previous candidate's output token IDs exactly. Repetition
persists at every context; completing 32K resolves the dispatch performance
cliff, not model-quality qualification.

Short natural-EOS sanity checks also use the export's actual
`chat_template.jinja`, without forced minimum lengths or repetition penalties.
Both chunk-16 and chunk-1 chat runs answered the France-capital prompt
correctly, but answered `2 + 2` as `2` and failed an exact word-copy prompt.
Thus chat formatting alone does not resolve correctness, and these three
prompts do not constitute model-quality qualification.
The same six chat answers occur with all six experimental flags disabled.
Enabling the float32-reducing cooperative INT4 expert path also does not fix
the arithmetic or word-copy checks. These controls narrow the investigation
but do not establish correctness of the underlying graph or provider.

A separate native-GPU diagnostic at context 32 isolated seven measured decode
forwards: 2410 kernel events per step, including 653 `MatMulNBits` calls
accounting for 13.09 ms of summed device time per step. Hundreds of sigmoid,
split, multiply, and normalization dispatches remain. This motivates targeted
GEMV and graph-fusion work, not a projected TPS claim. A later bounded 2K
profiling attempt timed out after model loading and during warmup; the short
profile must not be relabeled as a 2K result.

### Native NVFP4 weight-only path

The local candidate also accepts block-16 NVFP4 expert weights, decoding
E2M1 nibbles and E4M3FN block scales in WGSL with float32 accumulation.
This is software weight decoding, not native FP4 matrix arithmetic and
not a Dawn change. Per-expert float32 global scales are required.

Logical weight shapes remain `[E,K,N/2]`. `weights_prepacked=-1` or `0`
selects N-packed payloads; `1` selects K-packed payloads with the same
logical dimensions, matching the CUDA schema. No requantization, expanded
resident weight copy, or offline weight rewrite is required. Scale shape
is `[E,N,K/16]`. Input dimensions must be divisible by 16. Separate FC3
weights, zero points, and activation scales are rejected.

Normalized routing without explicit router weights stays GPU-resident
for up to eight input tokens and 128 token-expert pairs. The default-off
`ORT_WEBGPU_QMOE_NVFP4_PREFILL=1` experiment extends this to 128 input tokens
and 4096 token-expert pairs, avoiding routing-count readback and per-expert
host dispatch for eligible prefill chunks. Larger or ineligible batches
retain the general path. This flag does not change FP8 routing.
K-packed projections distribute K across 64 lanes; N-packed projections
use eight column lanes and eight K lanes for coalesced reads.

On H200, 18 independent NumPy-reference fixtures passed 648 executions
with full WebGPU validation and CPU fallback disabled. Coverage includes
both packing layouts, FP16/FP32, fused SwiGLU with clipping and bias,
multiple experts, and the 128-token boundary and 129-token fallback.
The 13 preserved FP8 fixtures also passed 468 executions against the
extended plugin. These establish operator-level coverage, not model
quality or full-model throughput.

The plugin profiling bridge now exposes the existing native WebGPU
profiler to ORT. A diagnostic run produced real GPU `Kernel` events,
including NVFP4 expert projections. Earlier host-only node traces must
not be interpreted as isolated device timings. Profiling runs are kept
separate from throughput measurements.

NVFP4 qualification is secondary to the INT4 target. Both use a
2K/8K/32K-input, 256-output protocol with excluded warmups and three
distinct natural-text measurements. His original text pool is unavailable,
so a new frozen shared pool cannot be described as an exact reproduction.
The existing local Kunal NVFP4 package is a separate qualification vehicle,
not Tianlei's custom-op v12b package. Full-model loading, MTP correctness,
cache capacity, and throughput remain unqualified for this new path.

### Earlier FP8 diagnostics

Two default-off experiments use existing WebGPU compute shaders, without
changes to Dawn, Tint, or the quantization contract:

- `ORT_WEBGPU_QMOE_FP8_COOPERATIVE=1` distributes the inner dimension of
  block-FP8 expert projections across a 64-lane workgroup. It applies when
  there are at most 128 routed rows, the inner dimension is at least 512,
  the output dimension is at least 256,
  and one workgroup per output column fits the adapter's dispatch limit.
  Other shapes retain the existing scalar or subgroup-matrix path.
  FP32 reduction order changes, so numerical qualification is required.
- `ORT_WEBGPU_QMOE_SMALL_BATCH=1` extends GPU-resident indirect expert
  routing to block-FP8 inputs with at most eight tokens and 128 routed
  token-expert pairs. It requires normalized routing without optional
  router weights. Each token has independent TopK indices, input-row
  addressing, and final accumulation. This avoids the general path's
  routing-count readback and per-expert host dispatch loop for eligible
  small verification batches. Final accumulation follows selected-expert
  rank order rather than the general path's expert-index order, which can
  change FP16 rounding. Integer quantization is unchanged.

These experiments are part of local Flash Next performance work, not the
separate Dawn FP8 feature effort. Synthetic operator improvements do not
establish full-model throughput or equivalence to a different CUDA export.

### Initial matched full-model result

An H200/Vulkan run used the preserved checkpoint with FP8 expert weights,
INT4 dense weights, and separate CPU Engram. It used 45 prompt tokens,
32 generated tokens, MTP width 7, a bounded 16-block cache, no graph capture,
and no profiling. A single candidate-plugin model load was reused with both
flags off or on. One warmup per mode was excluded, followed by three
counterbalanced pairs of fresh requests. Decode TPS uses the interval
between the first and last emitted token, not request completion.

| Metric | Flags off | Both flags on |
|--------|-----------|---------------|
| Median decode tok/s | 1.1417 | 1.6347 |
| Pooled decode tok/s | 1.1407 | 1.6011 |
| Pooled end-to-end tok/s | 0.8238 | 1.3439 |
| Mean time to first token, seconds | 11.6697 | 4.4490 |

All measured requests emitted the same 32 tokens. Candidate requests
accepted 26 of 28 proposed draft tokens versus 27 of 27 for the control,
and used one additional target forward pass. This is a 1.43x median decode
improvement on a short diagnostic, not full quality qualification, a
100-tok/s result, or a matched comparison with Tianlei's CUDA package.

The synthetic 512-expert, top-10, eight-token QMoE test showed a much larger
whole-operator gain, from 127.82 to 3.27 ms. The full-model result above
must take precedence over extrapolating that microbenchmark. ORT host
traces do not isolate device time: removing a routing readback can move
the visible wait for queued GPU work into a later operator.

### Tianlei-inspired narrow INT4 decode GEMV

The local `ORT_WEBGPU_MATMUL_NBITS_SKINNY=1` experiment adapts the
few-row, K-parallel GEMV strategy from Tianlei's
[`skinny_gemm.cuh`](https://github.com/tianleiwu/qwen38-flash-ops/blob/76075ac4e62d275ba9d838e70ded9c2594c550a2/src/fused_gemm/skinny_gemm.cuh)
to the original model's symmetric RTN INT4 layout. It is a new WGSL
implementation, not a literal port of CUDA clusters, PDL, BF16/FP8
weights, or the fused hyperconnection graph. One workgroup reduces
each output column in FP32, with FP16 dequantized weights and final
output. Direct activation reads avoid the generic kernel's repeated
activation-tile barriers.

The default-off path requires NVIDIA, a single batch, one to four rows,
K at least 2048 and divisible by 32, N at most 1024, block32 INT4,
FP16 activations/scales, and no bias, explicit zero points, or expert
weight indexing. All other cases retain existing kernel selection.

An isolated strict-link build passed complete MoE and MatMulNBits test
translation-unit syntax checks. Full native GPU validation passed
35 fixtures, nine executions each, against independently computed
references without relaxing tolerances. Coverage includes actual
hyperconnection weights at M=1/2/4 and M=5/16 fallbacks, irregular N,
M=4/5 and N=1024/1025 boundaries, K=2016/2048/2080, bias, explicit
zero points, INT8, and FP32-input fallbacks. The full GTest binary was
not run.

An initial interleaved original-weight M=1, K=10240, N=324 diagnostic
measured a warmed GPU median of 24 us for the generic kernel and
12 us for the new path (seven samples each, full validation enabled).
Maximum absolute error decreased from 0.000244141 to 0.0000610352.
These are device kernel timings, not host Run latency or model TPS.
The same interleaved attempt stopped on the unchanged M=2 baseline's
FP16 accuracy failure at the wider activation range; the candidate
subsequently passed that fixture independently. Do not describe the
failed baseline comparison as a complete performance qualification.

Full-model chat sanity checks completed nine samples, all with the same
output tokens as the previous library. Arithmetic and exact-copy checks
still fail; the new GEMV does not resolve model correctness. A
counterbalanced 2K comparison is being evaluated separately. It changes only the skinny
flag, retaining the qualified QSA, dense-tail, upload, and readback
settings in both modes. This experiment does not establish model
quality, 100 tok/s, or half of Tianlei's CUDA throughput.

The bounded 2K/128-output run subsequently hit its 1800-second cap after
833.032 seconds of model loading, both excluded warmups, one measured
pair, and one unpaired candidate request. The matched pair measured
14.7968 tok/s with the skinny path off versus 15.5581 tok/s on
(5.14% higher); the unpaired second candidate measured 15.7953 tok/s.
This is not a three-pair median. Paired output tokens first differ at
zero-based index 13, so neither output equivalence nor quality is
established. The short chat suite's unchanged outputs do not override
this longer-context divergence.

Measured TTFT was 149.167 seconds off and 147.962 seconds on, substantially
worse in both modes than earlier v10 runs. The cause has not been
isolated; the narrow decode flag alone does not explain the common
regression. An independent GPU1 control overlapped part of loading and
warmup. Preserve this as a timed-out, headline-ineligible diagnostic, not
as a qualified full-model improvement. All GPUs were verified released
after the owned containers stopped.

Follow-up isolation found identical graph/configuration, GenAI/core
library hashes, and measured prompt hashes versus v10. The new library's
control also preserves v10's first 128 output tokens. CPU-only original
PLE lookups remain warm: changing-index M=1/M=16 medians were
0.110347/1.45187 ms with zero measured major faults. An original-weight
M=16 dense projection comparison passed nine reference checks per
library; old/new warmed GPU medians were 414/413 us for matmul and
11/11 us for layout. These checks rule out those specific changes as
an explanation, but do not isolate the full-model prefill slowdown.

### Experimental uniform upload batching

`ORT_WEBGPU_UNIFORM_ARENA=1` enables a default-off submission experiment
for ordinary, non-captured command recordings. Each recording owns a
64 KiB uniform arena. Dispatches bind individual, device-aligned ranges;
the packed CPU data is uploaded once before the recording is submitted.
An arena-capacity flush bounds storage independently of
`maxNumPendingDispatches`. Queue ordering preserves previous submissions
before the arena is overwritten. Graph capture and external capture
retain dedicated uniform buffers, so captured bindings never reference
reusable arena storage. Oversized uniforms retain the original path.

This targets host overhead, not different quantization or GPU arithmetic.
A strict native build recompiles all affected context-header dependents.
An independent FP32 Slice/Add chain with 640 nominal compute nodes
passed 18 executions with zero absolute and relative tolerance, including
arena-capacity flushes at a configured submission window of 1024.
Four original-weight FP16 dependency chains passed another 72
reference-checked executions with full validation. These are operator
diagnostics, not end-to-end throughput or model-quality qualification.
The initial large-window diagnostic improved the M=1 chains but regressed
the M=16 chains; this is not a universal tuning recommendation.

With profiling off and validation disabled (but every output checked
against the independent reference), two interleaved window-16 repeats of
the long M=1 chain measured 11.3698/11.3484 ms without the arena and
9.02541/9.02248 ms with it: approximately 20.6% lower host latency.
At window 64 the same comparison measured 11.0620 versus 8.38341 ms.
The short M=16 chain improved only approximately 2.6% at window 16.
These synthetic repeated-weight results do not establish a corresponding
full-model TPS improvement.

The varying-uniform chain also passed nine capture/replay executions with
the arena flag enabled, full validation, and zero numerical tolerance.
The native harness must prebind its output tensor for capture/replay:
an initial attempt with lazily constructed Run outputs failed explicitly
before comparison, then passed with a persistent bound output.
The queue-ordering and cold-capacity-flush regressions in the context test
translation unit pass native syntax/type checks; the full GTest binary
has not been run.

Broader profiling exposed a lifetime bug in the first arena candidate:
capacity-triggered flushing could finish and discard an in-flight
pipeline build while the current dispatch still held a pointer to its
bind-group layout. The corrected candidate retains owning layout and
ready-pipeline handles across that flush. The failing initial candidate
is not qualified; an exact-reference, cold-cache profiled test with two
independent session recordings passes after the fix. Model comparisons
must change only the arena flag and retain all other kernel and
transport settings.

The corrected build subsequently passed all 315 INT4 fixture checks,
18 exact profiled executions across two independent sessions, and nine
exact capture/replay executions with full validation. Its unprofiled
window-16 repeat retained the improvement: 11.3623 to 9.02871 ms for
the long M=1 chain, versus 14.2995 to 13.9004 ms for the short M=16
chain. Full-model validation is separate; these are not model TPS.

A subsequent single-load, unprofiled full-model diagnostic at a
32-token context completed both excluded warmups and three
counterbalanced pairs, generating 128 tokens per request. Changing only
the arena flag preserved all paired output tokens. Candidate decode
rates were 37.3925, 35.3915, and 36.3526 tok/s (median 36.3526);
control rates were 29.8268, 13.4780, and 13.2086 tok/s. Control
variability means the median ratio is not an established general
speedup. Mean TTFT was 5.3542 seconds on versus 2.7379 seconds off, and
aggregate end-to-end rates were 14.4682 versus 12.1868 tok/s. Model
loading took 496.345 seconds. This is not a 2K/8K/32K comparison and
does not resolve the existing quality failures. Preserve it as a
completed, quality-unqualified short-context diagnostic; longer-context
qualification is separate.

The completed 2K subset of the longer-context protocol contains three
counterbalanced measured pairs, with 256 output tokens per request.
Median decode rates were 15.1811 tok/s with the arena off and
16.2910 tok/s with it on, a 7.31% increase. All three pairs generated
identical tokens. This does not reproduce the short-context gain.
Prefill remained variable: control TTFTs were 87.1849, 21.4281, and
21.5748 seconds; candidate TTFTs were 20.8537, 20.9238, and 134.1701
seconds. Do not claim a stable prefill improvement from their medians.
The parent protocol was stopped after its first measured 8K request:
8K TTFT was 815.4670 seconds during warmup and 1139.4466 seconds
during that measured request, whose decode rate was 11.0875 tok/s.
These long-context observations overlap a separate GPU1 model load;
they are not a clean long-context comparison or a three-run median.
32K was not reached. Preserve the completed 2K subset independently.
Existing model quality failures remain unresolved.

For a decode-only diagnostic, the benchmark accepts
`--profile-decode-after 32` together with `--profile-prefix`. It leaves
session profiling disabled and enables GenAI run-level profiling only
after 32 generated tokens, avoiding a large prefill trace. The harness
requires new profiles containing GPU `Kernel` events for every requested
profiled token. Profiled timings are diagnostic, not unprofiled
throughput results; counterbalanced performance comparisons reject
profiling.

Increasing the submission window alone is not a large established gain:
a counterbalanced 16/64/64/16 repeat of the long M=1 chain measured
11.3514/11.0822/11.0724/11.3721 ms, respectively. The earlier 267 ms
window-16 outlier was not reproduced. All measurements include host Run
and readback; profiling and validation settings must be reported separately.

### Dynamic sparse-attention decode partitioning experiment

Actual 2K decode-only profiling identifies a different bottleneck from
the short-context diagnostic. Sixteen excluded-warmup traces each
contain 2410 GPU dispatches. Median instrumented GPU work was
75.848 ms per step, including 35.3295 ms across 12
`DynamicSparseAttention` calls and 14.7365 ms across 653
`MatMulNBits` calls. Instrumentation changes timings; these are not
unprofiled throughput measurements.

The original attention shader assigns one workgroup to each query head
and serially scans selected keys, performing a shared-memory reduction
with barriers for every key. A default-off single-subgroup replacement
passed 378 independent-reference executions, but did not improve the
model-shaped 2K device diagnostic: 8686 versus 8790 us. It is not a
qualified performance improvement.

`ORT_WEBGPU_DYNAMIC_SPARSE_ATTENTION_SPLIT_KV=1` instead applies
CUDA/Flash-Decoding-style key partitioning: one 32-thread workgroup
per head and 64-key partition, then a second dispatch merges FP32
partial maxima, denominators and unnormalized value accumulators.
Explicit or implicit softmax sinks are added once during the merge,
not once per partition. This requires NVIDIA, subgroup support with
both reported subgroup bounds equal to 32, batch one and single-token
decode. Other configurations retain the original attention path.
Weights and model precision are unchanged; this is not Dawn FP8 work.

An isolated strict-link candidate passed 27 fixtures, 18 executions
each, against independent references without relaxing tolerances.
Coverage includes FP16/FP32, head sizes 8/32/64/128/256, GQA cache
append, invalid and empty selections, zero-width selections, shared
and distinct auxiliary values, joint local/selected softmax, explicit
and implicit sinks, multiple partitions and prefill/batch fallbacks.
The added long-selection GTest source passes complete translation-unit
syntax/type checking; the GTest binary was not run.

The counterbalanced, full-validation GPU diagnostic at B=1, H=24,
KVH=2, D=256 and 2051 selected slots measured 8833 us for the
original attention kernel, versus 312 us for the partition stage and
86 us for merging. These are instrumented model-shaped operator
measurements with synthetic activations, not model TPS or an
end-to-end speedup. A validation-disabled, unprofiled preflight of the
same fixture measured median host Run including readback of 2.00352 ms
for the original path and 0.225495 ms for split-KV, with seven measured
samples per mode. Nine capture-mode executions also passed the independent
reference with full validation. Neither preflight establishes model TPS.
The unprofiled 2K model comparison completed with clean process exit and
released its GPU. It reused one model load, generated 256 tokens per
request, excluded one warmup per mode and changed only the split-KV flag:

| Measured pair | Original attention decode tok/s | Split-KV decode tok/s |
| --- | ---: | ---: |
| 1 | 16.0207 | 36.0501 |
| 2 | 16.1949 | 34.8088 |
| 3 | 16.1684 | 36.0343 |
| Median | 16.1684 | 36.0343 |

This is an observed 2.2287x throughput ratio, not an identical-token
workload comparison: all three paired sequences differ, first at generated
token indices 11, 4 and 4. Different tokens can change expert routing.
The six actual-template sanity checks still pass only two questions;
seven of nine total QA token sequences match the prior arena-only run,
with two raw-prompt completions differing. Language quality remains
unqualified despite passing operator references.

TTFT is unstable: control 72.7716/154.2743/134.9386 s versus candidate
146.0045/151.0191/129.6032 s. There is no qualified prefill gain.
A separate old/new/new/old frozen-library diagnostic passed all 72
independent-reference executions at sequence 16, including main-cache
selections and local plus auxiliary selections. Main-cache host medians
were 2.20908/2.20888/2.20801/2.18523 ms; this does not explain the
full-model TTFT variation. These synthetic operator timings do not
establish model prefill performance. Both attention experiments remain
default-off.

The follow-up, split-enabled 2K skinny-GEMV comparison completed its
three measured pairs in the same model load as the longer-context
requests. Skinny disabled measured 36.7020/36.7066/36.2922 tok/s;
enabled measured 36.0885/37.0666/37.9015 tok/s. The medians are
36.7020 versus 37.0666 tok/s, an observed 0.99% difference with variable
pair results, not an established improvement. All paired output
sequences differ, first at token indices 11/12/66. The excluded
39.5647 tok/s warmup is not a measured result. Longer contexts retain
skinny disabled, with split-KV and the uniform arena enabled.

A later default-off `ORT_WEBGPU_MATMUL_NBITS_HYPER_SKINNY=1` experiment
restricts the same K-parallel kernel to K=10240, N=324 hyperconnection
projections, retaining the original M=1..4, precision and layout guards.
The original broad skinny flag retains its previous behavior. This avoids
changing router and other narrow projections merely to optimize this shape.
An isolated provider reproduces the qualified command-state baseline before
replacing only the MatMulNBits object. All 35 candidate native fixtures
passed full-validation references. The unchanged flag-off actual-weight
M=2 fixture still fails its existing FP16 tolerance; that failed diagnostic
is retained, not represented as a passing baseline.

On GPU5, three same-load counterbalanced 2K/256-output pairs plus an
excluded warmup measured median 45.6540 tok/s off versus 48.8419 on
(aggregate 45.2232 versus 48.5238). The three measured candidate rates
were 47.8254, 48.9197 and 48.8419 tok/s. Generated sequences differ
between modes, so this is not an identical-token-workload speedup.
Only two of nine strict answer checks passed; whole-model accuracy remains
unqualified and the experiment is not promoted. An earlier GPU0 attempt
yielded to a foreign process after excluded warmups; no measured result
was taken from that interrupted attempt.

The earlier split-enabled model completed all three measured requests at each context.
The following fixed-configuration table uses the 2K skinny-disabled
control requests and the 8K/32K configured requests: original INT4 weights,
split-KV and uniform arena enabled, skinny disabled, chunk 16 and
256 generated tokens. Warmups are excluded; no profiling is enabled.

| Context | Measured requests | Median TTFT (s) | Median decode tok/s |
| --- | ---: | ---: | ---: |
| 2K | 3 | 147.5350 | 36.7020 |
| 8K | 3 | 301.3388 | 31.0883 |
| 32K | 3 | 409.7541 | 18.8908 |

The 8K measured rates were 31.1275/30.7938/31.0883 tok/s, with TTFT
285.7503/301.3388/313.5746 s. Its excluded warmup took 857.2141 s
to the first token and decoded at 28.9179 tok/s. These are warm-request
medians, not cold-start latency or an established prefill improvement.
The 32K measured rates were 18.5048/18.9173/18.8908 tok/s, with TTFT
441.0645/407.6648/409.7541 s. Its excluded warmup decoded at 18.6199
tok/s with TTFT 423.4445 s. The whole protocol exited cleanly in
5730.2287 s, including one 534.5478 s model load. Model quality remains
unqualified. Concurrent foreign work on other GPUs was observed; GPU 0
remained isolated, but shared-host effects are not ruled out.

### Norm-free INT4 MLP fusion qualification

The existing `MatMulNBitsMlp` schema and WebGPU kernel support an absent
normalization input. However, the graph optimizer requires a supported
normalization producer. The original Flash Next graph has 48 eligible,
bias-free standard-SiLU gate/up pairs consuming `HyperConnectionPreMix`,
so that optimizer does not fuse them. This is not a missing-schema
problem, and the pinned core library has not been replaced.

An opt-in offline conversion experiment recognizes these shared-input
INT4/block32 pairs, checks exact packed-weight and FP16-scale layouts,
and protects intermediate graph outputs and additional consumers.
It preserves all initializer bytes, graph interfaces and opsets, rejects
subgraphs requiring implicit-input analysis, and leaves the source model
unchanged. Fourteen structural/CLI tests pass, including failure-safe,
non-overwriting publication. A separate candidate passes the ONNX
checker with exactly 48 fusions and 2314 -> 2122 graph nodes. This is a
graph-node count, not a measured GPU-dispatch reduction.

Native qualification completed cleanly with 144 independent-reference
executions at M=1/2/16, K=32/N=8 and model-shaped K=2560/N=640,
without relaxing tolerances: 108 with profiling/full validation and 36
unprofiled. Profiling confirms one `MatMulNBitsMlpDecode` dispatch at
M=1; M=2/16 retain projection-plus-activation fallbacks. At the
model-shaped M=1 fixture, the last-seven median summed instrumented
GPU work was 149 us unfused versus 56 us fused. Counterbalanced
unprofiled host Run/readback medians were 1.33776/1.84635/1.56740/
2.10736 ms in unfused/fused/fused/unfused order, with substantial
variation. These are synthetic operator measurements, not model TPS.
Three additional fixtures produced by the actual offline converter pass
27 full-validation native reference executions at model-shaped
K=2560/N=640 and M=1/2/16. Both qualification guards exited cleanly and
released GPU 3.
The derived model subsequently completed full-model sanity checks and
three measured 2K requests after an excluded warmup, with the original
pinned provider/core and no parallel-rank experiment. Its external assets
are contained hard links to the original files; no weight bytes were
changed and external-data containment checks remain enabled.
Decode rates were 36.7119/36.6910/36.7021 tok/s, median 36.7021,
versus the serial-load original-graph median 36.7020 tok/s. This does
not establish a decode improvement, so the derived graph is not adopted.
Measured TTFT was 20.9921/21.0199/20.9789 s; the control's large
latency variation and non-counterbalanced model loads prevent attributing
that difference to fusion. All three paired generated sequences differ,
first at indices 17/4/4. Model quality remains unqualified.

### Bounded parallel QSA ranking experiment

The private, default-off `ORT_WEBGPU_QSA_PARALLEL_RANK=1` experiment
replaces single-workgroup sorted selection with one 64-lane workgroup
per candidate block. Each workgroup counts greater scores and earlier
tied indices, then scatters its selected block. Score computation is
unchanged. In the first qualified candidate, a separate fill initializes
unused output slots; mask
compaction, stable ties, partial-block tails and the existing NaN
selection handling are retained.

The quadratic comparison work is deliberately bounded: NVIDIA only,
global score cache enabled, one logical query row, 1..1024 selected
blocks and score stride 512..16384. This includes 32K plus 256 decode
tokens. Prefill, batched queries and out-of-range cases keep the existing
path.

All 49 independent-reference fixtures passed both flag modes, 882
executions with full validation/profiling. Seven fixtures actually
exercise parallel rank, including FP16/FP32, periodic score ties, empty
visibility, maskless inputs, rotary inputs and an irregular budget;
the other 42 exercise unchanged fallback eligibility. The first guard
yielded to a foreign training process after 43 completed fixtures.
Only the remaining six were rerun on a freshly checked idle GPU; that
guard exited cleanly and also completed 54 validation-disabled,
unprofiled reference executions. These are completed combined results,
not a claim that the interrupted guard completed normally. Related
GTest source typechecks but its executable was not run.

Last-seven median instrumented selection costs for the actual model
contract were:

| Context | Old sort | New fill | New parallel rank |
| --- | ---: | ---: | ---: |
| 2K | 88 us | 10 us | 13 us |
| 8K | 323 us | 7 us | 12 us |
| 32K | 1262 us | 10 us | 39 us |

Counterbalanced unprofiled host Run/readback medians were respectively
1.60071 -> 1.61518 ms, 1.95234 -> 1.95284 ms, and
13.6089 -> 6.60495 ms. They include full cache-output readback and
are not isolated shader timings or model token rates. A serialized original-graph model run attempted the candidate at 2K/8K
and three rank-only counterbalanced pairs at 32K, with all other flags
fixed. No model-level rank speedup or quality qualification is established.

The model protocol completed its configured 2K/8K subsets:
three measured requests each, medians 35.5070 and 34.1420 tok/s,
respectively, versus the earlier serial-load medians 36.7020 and
31.0883 tok/s. All six corresponding output sequences match exactly.
These are not same-load paired comparisons. Original-graph TTFT
medians are again 21.0528 s and 95.7061 s, reinforcing that the earlier
large latency variation must not be attributed to MLP fusion or to a
decode-only selection change. The whole protocol subsequently hit its
10800-second cap during 32K repetition 1. One measured 32K pair
completed: control 18.1746 tok/s with TTFT 410.9489 s, candidate
0.6455 tok/s with TTFT 3777.7405 s, with exactly matching generated
tokens. The candidate's excluded warmup was 26.0113 tok/s; it must not
replace that slow measured result. Three-pair medians are unavailable,
the severe collapse is unresolved, and the experiment is not promoted.
The owned container was stopped; no foreign process was found on its
GPU at termination.

A follow-up single-dispatch variant was built in a separate build,
without changing the provider used by that model run. Candidate zero
counts non-NaN scores and initializes only output slots that no selected
candidate can write, removing the extra fill dispatch. Two CPU property
tests cover 1060 finite/nonfinite, tie, empty and randomized partitions
against an independent sort and require every output slot to have exactly
one writer. Twenty additional checked native fixtures cover first/late
NaNs, infinities, 32K plus 256 tokens and both sides of the score-stride
cap. After the previous model released GPU 0, all 69 fixtures passed
1242 full-validation reference executions, followed by 54 unprofiled
reference executions, with a clean guard exit. This includes all 20
new boundary/nonfinite fixtures. Actual model-shaped profiles confirm
there is no separate fill dispatch: old sort versus rank medians are
85 versus 10 us at 2K, 323 versus 12 us at 8K, and 1259 versus
47 us at 32K. Unprofiled host Run/readback medians are
1.61452 -> 1.60041, 2.01186 -> 2.00847, and 13.9959 -> 7.01586 ms.
Neither native correctness nor removing the fill establishes a fix for
the full-model collapse.

The bounded `--compare-qsa-rank-windows` diagnostic avoids repeating
eight expensive prefills. It uses 32 excluded decode warmup tokens,
warming each flag, then eight equal decode windows in ABBA/BAAB order.
A second identical-prompt request complements every window's flag,
giving matched logical context/token positions. Full generated
sequences must match before a window summary is published. Window
timings exclude boundary flag changes and evidence writes; this is not
the standard 256-output table. GPU memory/clocks and only the owned
process's RSS/page-fault counters are recorded separately. The current
32K diagnostic uses 32-token windows, two requests and an 1800-second
guard, with no other owned GPU work. Twenty-six focused Python tests
pass, including prior benchmark behavior, callback timing, complementary
pairing, interrupted evidence and output-mismatch rejection. The summary
uses the geometric mean of matched throughput ratios, rather than an
arithmetic median that can be inflated by a multiplicative slowdown
between complementary requests. Per-request absolute throughput medians
remain visible; cancellation in a relative statistic does not establish
stable end-to-end performance. A regression test covers a 100-fold
between-request slowdown.

The bounded 32K attempt subsequently timed out without reaching decode:
loading took 814.2447 s and the first rank-disabled prefill never completed
within the remaining runtime. No measured windows or token-rate summary
exist for that attempt. Peak recorded GPU memory was 69587 MiB, with no
foreign compute PID on GPU 0. The last owned-process sample had approximately
6.51 GiB RSS, zero swap and five cumulative major faults. Thus a severe
slowdown can occur before parallel ranking is enabled; the earlier
rank-enabled failure does not establish ranking as its cause.

### Persistent monitoring and matched 2K follow-up

The observer's nominal five-second cadence actually had a 15.7511-second
median interval. Separate idle `nvidia-smi` launches took 9.13-9.22 s.
Holding one read-only NVML client open reduced subsequent direct API
queries to approximately 0.03-0.40 ms and a CLI query to 0.099 s.
The guard now retains that client for its lifetime instead of repeatedly
initializing the driver. Foreign-PID ownership checks and the minimum
130000 MiB free-memory requirement remain enforced, conservatively rounding
free memory down. No clocks, power limits or driver settings are changed.
A real guarded smoke test observed its owned Vulkan PID and passed
54 independent-reference executions across 2K/8K/32K fixtures.

The subsequent original-graph 2K diagnostic completed cleanly with a
58.2275-second load, 16 windows and eight complementary pairs. All
288 output token IDs matched between requests:

| Diagnostic only | Rank disabled | Single-dispatch rank |
|---|---:|---:|
| Median 32-token window throughput | 36.9127 tok/s | 39.2912 tok/s |

The geometric mean of matched throughput ratios was 1.06583. Per-request
medians were also consistent (36.8829/39.2907 and 36.9127/39.3429 tok/s).
These are diagnostic windows, not a replacement for the standard
256-output table or a model-quality qualification. The much shorter load
is an observed sequential comparison, not proof that polling caused every
earlier slowdown. The 8K/32K follow-up completed in 1087.5131 s, including
a 57.8110-second model load, under the same persistent monitoring:

| Context | Rank disabled, median window tok/s | Rank enabled, median window tok/s | Geometric mean matched ratio |
|---|---:|---:|---:|
| 2K | 36.9127 | 39.2912 | 1.06583 |
| 8K | 31.6037 | 36.8831 | 1.16709 |
| 32K | 19.7847 | 28.6431 | 1.44559 |

Each context has eight complementary pairs and two requests producing
identical 288-token sequences. Both new 32K sequences also match the
first 256 token IDs from both the earlier measured control and the
earlier collapsed candidate. The two 32K TTFTs were 400.7405 and
400.5598 s. Recorded GPU memory peaked at 69928 MiB; median observer CLI
query latency was 0.1320 s. This is a completed decode-window diagnostic,
not the standard table, a quality qualification or proof that every
historical stall has been explained. The 100 tok/s goal remains unmet.

### Experimental dense-visible QSA map

`ORT_WEBGPU_QSA_DENSE_MAP=1` is a separate default-off experiment for
NVIDIA, one logical query row and 512-65536 total tokens on the existing
global-cache path. It first counts visibility using the existing causal
and full-width INT64 mask predicate. If every token is visible, it writes
the identity map without a prefix barrier for every 64-token chunk; empty
visibility only writes a zero count. Masks containing holes retain the
original compaction. `workgroupUniformLoad` makes the fast-path decision
uniform before an early return, and the shader-cache hint distinguishes
the experiment.

The isolated provider compiles with the existing strict link recipe.
CPU checks cover 158 mask/past cases and single-writer output invariants;
the WebGPU-enabled GTest source also passes its syntax check. After the
model released GPU 0, both flag modes passed all 69 native fixtures:
2484 full-validation reference executions plus 108 unprofiled references,
with a clean guard exit. Full-visible map medians changed from
37 to 12 us at 2K, 474 to 75 us at 33025 tokens, and 931 to 142 us at
65536 tokens. The synthetic model-shaped 32K fixture contains seven mask
holes and instead regressed from 472 to 534 us because it takes the
fallback after the extra visibility scan. This distinction must not be
hidden or treated as a universal operator gain.

The `--compare-qsa-dense-map-windows` full-model follow-up holds parallel
ranking enabled and varies only the map flag, using the same matched
window protocol with correctly named map-specific evidence fields.
Forty-two focused Python tests pass. The full-model comparison completed
with a clean guard exit in 1137.3986 s, including a 58.3076-second load:

| Context | Map disabled, median window tok/s | Map enabled, median window tok/s | Geometric mean matched ratio |
|---|---:|---:|---:|
| 2K | 39.2186 | 39.6544 | 1.01252 |
| 8K | 36.9705 | 38.6188 | 1.04708 |
| 32K | 28.5583 | 33.6174 | 1.17959 |

All 48 windows form 24 complementary pairs. Each of the six requests
produced 288 token IDs identical to both earlier rank-diagnostic requests
at the same context. These are diagnostic windows, not the standard
256-output table or a model-quality qualification. The seven-hole
operator regression remains relevant. Previously qualified binaries and
their exact source snapshots remain preserved.

### Experimental N96 FP16 GEMV projection

The current split-KV/rank-enabled 2K single-step profile contains 2422
GPU kernels with 34.637 ms summed duration. The 36 linear-attention
`a_b_proj` MatMuls are the largest individual category at 2.556 ms.
Profiling changes timing, so these durations are attribution evidence,
not unprofiled token rates or CPU/GPU time fractions.

`ORT_WEBGPU_MATMUL_GEMV_N96=1` experimentally reuses the existing
f32-accumulating FP16 GEMV shader for NVIDIA, batch one, M=1, K=2560,
N=96, without bias or activation. The existing N=16-64 eligibility is
unchanged. Rebuilding the unchanged MatMul and relinking first reproduced
the qualified dense-map provider byte for byte.

The isolated candidate passed 100 native candidate executions against
independent float64 products rounded to the output type. Another 68
control executions passed exact structured references; 32 control
executions characterize existing FP16 accumulation error, including
overflow in constructed cancellation cases, and are not counted as
passing numerical references. All 200 actual GPU dispatches were checked:
eligible shapes select GEMV only with the flag enabled, existing N=64
behavior is preserved, and float32, multiple rows/batches, K boundaries
and neighboring widths retain their old paths. This deliberately changes
accumulation precision, unlike the integer-exact map optimization.
The added WebGPU GTest source passes its enabled-provider syntax check
but its executable was not run. Forty-three focused Python tests pass.

Three counterbalanced, full-256-output 2K request pairs completed with
a clean guard exit in 279.9136 s, including a 57.8724-second load.
Median decode throughput was 39.8174 tok/s for control and 43.4859 tok/s
for the candidate, an observed 9.21% ratio-of-medians increase. However,
all three measured pairs and the excluded warmup pair generated different
token sequences; measured first differences were at indices 19, 1 and 4.
This is not a same-token optimization-equivalence result, and changed
generation/routing can influence timing. Independent operator precision
checks do not establish model quality. The candidate remains default-off
and quality-unqualified.

The separate standard run completed with a clean guard exit in
2203.1780 s, including a 58.0211-second model load. Each context has
one excluded warmup and three measured requests producing 256 outputs;
decode rates use the 255 intervals after the first token. This is the
standard protocol, not the complementary-window diagnostic:

| Context | Median TTFT (s) | Median decode (tok/s) | Aggregate decode (tok/s) |
|---|---:|---:|---:|
| 2K | 20.6160 | 43.4781 | 43.5069 |
| 8K | 89.4949 | 42.4118 | 42.4019 |
| 32K | 400.4215 | 36.0463 | 35.8981 |

All 12 request counts, output hashes and rate calculations were checked.
All four 2K sequences exactly reproduce the earlier N96-enabled
requests. This remains a quality-unqualified experimental stack, not
a same-token causal comparison with older precision paths. Brief
accuracy-only GPU 1 work and separate Dawn GPU 4 correctness work may
overlap on the shared host; fully idle-host isolation is not claimed.
Late-start telemetry covers the remaining long-context work, not the
whole run: observed GPU memory peaked at 69928 MiB, owned-process swap
and major faults were zero, and median query latency was 0.1325 s.
The owned GPU 0 process was released. The 100 tok/s goal remains unmet.

A separate accuracy-only control on another idle H200 kept the same
graph, core, GenAI, provider, inputs and flags except N96. All six
chat-templated sanity sequences match between N96 off and on, including
the failures: both answer `2 + 2` with `2` and emit an empty response
for the word-repetition prompt, with either chunk 1 or chunk 16.
These particular failures are therefore not newly introduced by N96;
they remain an unresolved full-pipeline quality problem. No cross-GPU
performance comparison is made in this accuracy check.

### Full-validation control and real-activation diagnostics

A further nine-case sanity control disabled all 13 recorded experimental
flags and explicitly selected WebGPU `validationMode=full`, retaining the
same graph, configuration, provider, core and GenAI identities. The guard
completed successfully in 87.7792 s without a reported validation failure.
All six chat-templated output sequences exactly matched the earlier
control: arithmetic still returned `2` instead of `4`, word repetition
returned an empty answer, and the capital question returned `Paris`.
Two of nine strict exact-string checks passed; raw reasoning responses
also face formatting restrictions, so that count is not a general
semantic-accuracy metric. These observed failures persist without the
13 recorded experiments; this does not identify an export defect.

Separate derived graphs exposed existing intermediate tensors while
preserving the original graph, configuration and external weights.
The runtime's external-data containment checks remained enabled; derived
graphs were placed beside the original data rather than using an
out-of-directory symbolic-link clone. Both full-validation capture guards
completed successfully, and the first two generated token IDs matched
the uninstrumented control. These are correctness diagnostics, not
throughput measurements.

The first QMoE's actual settings are 512 experts, top-k 10, normalized
routing, interleaved fused SwiGLU and both expert biases. Independently
dequantizing its real selected weights and evaluating float64 products
with FP16 dequantization and intermediate boundaries produced maximum
absolute differences of 0.0001220703125 for both the captured last
prefill row and the next decode row. RMS differences were 0.0000188302
and 0.0000194474 across 2560 outputs each. This is a numerical
characterization, not exact equality or full QMoE qualification.
An additional independent reconstruction of its INT4 router projection
had maximum absolute differences of 0.03125 and 0.0234375 across 512
logits; both reconstructed top-10 expert rankings exactly matched the
captured rankings. This also does not establish later-layer routing parity.

The first GatedDeltaNet's real inputs were also evaluated with an
independent float64 V-major recurrence, including Qwen decay, sigmoid
beta, Q/K normalization and the actual contiguous 16-to-48 head
expansion. The 14-token prefill compared all 86016 outputs and 786432
state elements; maximum absolute differences were 0.0000038147 and
0.0000056624 respectively. The next decode compared all 6144 outputs
and 786432 state elements, with maxima 0.0000019073 and 0.0000033379.
The prefill reference started with zero recurrent state; decode used
the preceding captured native state. Three analytic reference tests
passed. These two first-layer snapshots do not cover later layers,
long-context behavior or the whole-model quality failure.

### Experimental subgroup gate selection

`ORT_WEBGPU_QMOE_SUBGROUP_GATE=1` selects a default-off NVIDIA subgroup
gate for a single token, 512 experts and top-k 10, with FP16 symmetric
INT4 experts, normalized routing and no separate router weights.
Unsupported devices and other configurations retain their existing path.
The shader uses ordered integer float keys, canonicalizes signed zero
and breaks equal-value ties by lower expert index. It replaces each
full-workgroup reduction tree with subgroup winners and two workgroup
barriers per selected expert. Negative-infinity candidates remain valid.

An isolated provider build reproduced the preceding qualified provider
byte for byte when linked with unchanged objects. Four alternating
off/on rounds across 14 full-QMoE fixtures produced 112 GPU executions.
All outputs met independent references within
`0.001 + 0.001 * abs(reference)` and were byte-identical between modes.
Seven eligible fixtures cover random values, equal ties, plateaus,
large magnitudes, minimum finite FP16, negative infinity and signed
zeros. Seven fallback fixtures cover other expert counts, top-k counts,
two rows, FP32 and unnormalized routing.

After excluding each variant's first execution, profiled gate medians
were 27-29 us for the preceding path and 20-21 us for this candidate.
These are operator measurements, not an end-to-end throughput gain.
The enabled native WebGPU GTest translation unit, including direct
index and signed-zero regressions, passed syntax compilation; its full
test executable was not run.

A subsequent same-load, counterbalanced 2K test fixed the submission
window at 64 and compared three measured 256-output request pairs,
excluding a warmup pair. All four paired output sequences were identical,
and all nine candidate sanity sequences matched the earlier fast-stack
control. Median decode rates were 46.7029 tok/s off and 46.7006 tok/s on.
Thus the operator improvement did not improve the full-model median
in this experiment. The candidate is not promoted, and the known model
quality failures remain unresolved.

### Actual-model submission-window experiment

The provider's existing `maxNumPendingDispatches` option was tested on
the current original-INT4 fast stack without changing the graph, provider,
core, GenAI or other experimental settings. Separate same-GPU loads ran
windows 64, 16 and 256, in that order. Each used one excluded warmup and
three measured 2K requests with 256 output tokens and 255 decode intervals:

| Pending dispatch window | Median TTFT (s) | Median decode (tok/s) |
|---|---:|---:|
| 16 | 20.6248 | 43.4151 |
| 64 | 20.3747 | 46.5507 |
| 256 | 20.3261 | 45.6707 |

All four generated sequences per window were byte-identical, including
the excluded warmup, and the 64-dispatch run reproduced all nine earlier
fast-stack sanity sequences exactly. The known quality failures persist.
These are serial-load observations, not counterbalanced or isolated-host
causal measurements. The default remains 16; larger windows are not
universally faster, and these runs establish no long-context gain.

The subsequent complete 64-dispatch standard run finished with guard
exit zero in 2187.0644 s, including a 60.2439-second model load. It used
the unchanged fast provider with the subgroup-gate candidate disabled.
Each context again has one excluded warmup and three measured requests,
256 outputs per request and 255 decode intervals:

| Context | Median TTFT (s) | Median decode (tok/s) | Aggregate decode (tok/s) |
|---|---:|---:|---:|
| 2K | 20.3742 | 45.7262 | 45.7836 |
| 8K | 88.5374 | 45.2896 | 45.0189 |
| 32K | 404.8132 | 37.9328 | 37.9265 |

All twelve generated sequences, including warmups, exactly matched the
preceding standard table, as did all nine sanity sequences. Graph,
configuration, provider, core and GenAI identities were unchanged; all
rate formulas and output counts were checked. The earlier standalone
2K median of 46.5507 remains a separate result and is not substituted
for this complete run's 45.7262.

This is a separate-load comparison, not counterbalanced or isolated-host
causal attribution. Brief Dawn work and a separate GPU 1 CPU-profile
diagnostic overlap parts of the shared-host run. Telemetry starts after
the first two requests: 405 active samples observed a 69928-MiB GPU-memory
peak, zero owned-process major faults and zero swap. Known model sanity
failures persist, and the 100 tok/s goal remains unmet.

### Experimental bind-group reuse

An actual-model decode CPU profile identified bind-group creation at
13.42% of inclusive active CPU samples. This is not a wall-time fraction
or a predicted throughput gain. `ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES`
enables a default-off NVIDIA-only experiment, accepting capacities from
0 (disabled) through 8192. The full key includes the bind-group layout
and every binding number, buffer handle, offset and size. Retained bind
groups keep their referenced resources alive. Capture uses the original
creation path.

The cache belongs to the context factory and is destroyed before its
context/device. It skips entries referencing more than 1 MiB of whole
buffers and clears on capacity or 256-MiB aggregate referenced-buffer
pressure. The current revision counts each retained whole buffer once
across entries; the preceding revision counted shared buffers repeatedly.
These bounds do not measure exact driver allocation overhead.
`ORT_WEBGPU_BIND_GROUP_CACHE_STATS_FILE` optionally appends JSONL
hit, miss, large-binding fallback, clear and peak statistics at cache
destruction; write failures are logged explicitly.

The hit path avoids querying buffer sizes on cache hits.
Fifteen independent full-QMoE fixtures, each run eight times with both
8192-entry and one-entry capacities, passed all 240 GPU reference and
exact off/on comparisons. Real hits, one-entry eviction and large-binding
fallback were observed. A further 220 simultaneously retained sessions
passed 1760 executions and forced a genuine byte-pressure clear at
1011 entries, below the entry limit, with a peak accounted size of
268428096 bytes, below 256 MiB, using the earlier conservative accounting.
The unique-buffer revision passed another 240 ordinary executions and
9600 executions across 1200 simultaneously retained sessions. It forced
four byte-pressure clears, peaking at 1278 entries, 2307 unique buffers
and 268394624 accounted bytes. These are native checks, not whole-model
quality qualification.

The preceding cache revision did not improve a same-load counterbalanced
2K model comparison: three measured 256-output pairs gave median decode
rates of 46.3784 tok/s off and 46.2583 tok/s on. All four paired sequences,
including warmups, were identical; all nine candidate sanity sequences
matched the prior fast stack.

The subsequent hit-path revision measured medians of 45.2722 tok/s off
and 45.8702 tok/s on, but aggregate decode rates were 45.7334 off and
45.4353 on: no consistent gain. Its process-wide statistics showed only
28972 hits against 4311607 misses and 2186 clears. Unique-buffer
accounting increased hits to 988165 against 3352414 misses, with 224
clears, but the full-model medians regressed from 45.9781 off to
44.2010 on. That run reached the 8192-entry limit, with a peak byte
count of 224091248, below the byte budget. These statistics include
cache-enabled sanity, prefill, warmup and measured decode, not decode
alone. Both revisions retained exact agreement for all four paired
sequences and all nine sanity sequences. The cache is not promoted.

Invalid capacities emit explicit errors. An earlier negative-capacity
diagnostic also crashed during process shutdown. Validation now occurs
before pipeline/uniform work, and all twelve repeated negative tests
(`-1`, `8193`, malformed text; four each) produced the expected message
and exit code one. This does not establish that every asynchronous
shutdown race has been eliminated.

`ORT_WEBGPU_DYNAMIC_UNIFORMS=1` is a further default-off NVIDIA experiment
targeting the observed entry churn. It uses a dynamic uniform binding
and stores the offset with each recorded command instead of including
the arena offset in a cached bind group. The arena is still reused
between submissions; this is not a fix for an arena recreated on every
flush. Pipeline keys distinguish dynamic layouts, and dedicated uniform
buffers, including capture/replay, use dynamic offset zero. Adapters
without a dynamic-uniform binding limit retain the original path.
The eleven revised command-state consumers were rebuilt together. Native
qualification passed 360 independent-reference/exact-parity QMoE executions,
35 dense fixtures, exact varying-uniform/interleaved-session and capture/replay
checks, and twelve explicit invalid-flag errors with exit code one.

With dynamic uniforms enabled in both modes, a same-load, counterbalanced
cache-off/cache-on comparison measured median decode rates of 44.9895 and
46.4082 tok/s, respectively (aggregate 45.4057 and 46.3970). All four
paired generated sequences and all nine sanity sequences matched exactly.
The cache recorded 2748562 hits, 1592017 misses, 1510410 large-binding skips
and nine clears: 97.1165% eligible hits. These are whole-process statistics,
not decode-only statistics. This comparison does not establish a whole-feature
gain against static uniforms with no cache. Both experiments remain default-off,
and whole-model accuracy is still unqualified.

A subsequent same-provider whole-bundle comparison toggled static
uniforms/cache zero against dynamic uniforms/cache 8192. Three measured,
counterbalanced 2K pairs plus an excluded warmup measured median
45.0091 versus 45.8758 tok/s (aggregate 44.9404 versus 45.7806).
All four paired output sequences matched exactly. Precomparison sanity
used the static/cache-off load configuration; the prior separate
dynamic-uniform sanity run also matched the original fast-stack sequences.
This approximately 1.9% observation is on a shared host, not an
original-A9-binary head-to-head or an accuracy-qualified promotion.

### Original-graph runtime recheck and CPU embedding residency

A fresh one-token profile of the dynamic-uniform/cache-8192 stack retained
2422 GPU kernels, with 31.897 ms summed instrumented device time. Its CPU
FP8 `GatherBlockQuantized` event took 48.260 ms for just 5120 output bytes.
The separate 51,200,245,760-byte `engram.data` table had only 6,208,815,104
bytes resident, even after warming the original INT4 weight file.
An earlier cold model-load attempt reached its explicit 600-second cap;
that failure is retained rather than counted as a completed measurement.

Read-only sequential reads warmed both original data files in RAM cache
without making persistent weight copies or changing the graph. An
unprofiled 2K run with an excluded warmup and three measured 256-token
requests then reached 46.2693 / 45.4830 / 45.5654 tok/s:
median 45.5654 and aggregate 45.7699 tok/s. All four sequences exactly
match the original complete-table 2K sequences. This is a runtime recheck,
not a new kernel speedup, a replacement context table, or quality promotion.

A subsequent same-workload profile measured the CPU gather at 99 us;
both profiled generated sequences match the corresponding unprofiled
32-token prefixes. New Linux harness metadata separates prefill and decode
process major/minor page faults, CPU time and context switches. Both
phases recorded zero major page faults in both requests of that hot-table
diagnostic. These counters are process-wide, not per-operator attribution;
profiles remain ineligible for headline throughput. Cache residency is
not memory pinning and can change under shared-host memory pressure.

### Opt-in FP16 hyper-connection pointwise fusion

An explicit model-local-function derivative fuses `Sigmoid` into 97
`HyperConnectionPreMix` calls and `Div -> Sigmoid -> Mul` into 96
`HyperConnectionPostMix` calls. The original graph, configuration and
external weight descriptors remain unchanged; the derivative uses
same-inode weight hardlinks rather than additional weight copies.
The new FP16-only kernels are selected only by the derivative's function
calls. Existing operator names retain the unfused behavior. Shader-cache
hints distinguish the fused arithmetic, including its FP16 intermediate
storage boundaries, from the ordinary programs.

Eight fused/unfused WebGPU fixture pairs at M=1/2/16, including hidden
dimensions 4/7/2560, passed three runs each under full validation with CPU
fallback disabled: 653,070 finite FP16 output elements matched bitwise.
Independent ONNX function expansion also matches the original CPU reference
body exactly. This is not a CPU/GPU numerical-parity qualification: an
earlier CPU-oracle fixture run failed at `hc_post_m2_h7`; its tolerances
were not relaxed. Unsupported scalar-parameter shape cases are prepared,
but their runtime execution was blocked by subsequent foreign GPU work.

The full model executed exactly **2037 kernels/token**, down from 2422:
97 fused pre-mix kernels and 96 fused post-mix kernels replaced 385 separate
pointwise dispatches. Both bounded decode profiles confirm these counts.
Profiled timing is not used as headline throughput.

An unprofiled GPU5 2K run, with one excluded warmup and three measured
256-token requests, reached **47.5889 / 48.2435 / 49.4195 tok/s**:
median **48.2435**, aggregate **48.4055 tok/s**. Measured TTFTs were
20.3698 / 20.3634 / 20.2813 seconds. All four 256-token sequences match
the preceding original-graph control exactly, and all prefill/decode
phases recorded zero process major page faults. The two profiled 32-token
requests also match the corresponding unprofiled prefixes.

This is not a counterbalanced same-GPU comparison, a replacement complete
context table, or whole-model quality promotion. The original complete
45.7262 / 45.2896 / 37.9328 tok/s table remains unchanged, 100 tok/s remains
unmet, and no new fused-graph 8K/32K result is claimed.

The retained `build_hc_fusion.py` checks that unchanged link inputs reproduce
the exact dynamic-uniform provider before replacing only the two HC
translation units and the contrib registry. `fuse_hyper_connections.py`
requires FP16 metadata, exact supported operator/attribute chains,
single-consumer non-output intermediates, scalar parameters and unchanged
initializer descriptors. The original graph has 97/96 eligible chains;
its standalone invocation checks those exact counts before staging.
`hc-fusion-qualification.json` records the resulting graph/provider hashes,
full-sequence parity, dispatch evidence and remaining limitations.

### Exact FP16 split/activation fusion on the HC stack

The next opt-in model-local function combines last-axis `Split` and
`ScaledSiLU` into one two-output kernel. The activated prefix preserves
the FP16 scale, sigmoid and final multiplication rounding of the existing
`ScaledSiLU`; the second split output is copied unchanged. The function
is FP16-only, accepts exactly two positive split widths, and retains the
ordinary unfused operators for other models. The rewrite requires a
single-consumer, non-output prefix and no optional SiLU scale input.

Exactly 96 block chains match; the final HC mixer has no matching Split.
The new graph retains the preceding HC fusions and all original initializer
descriptors. Original weights and configuration are same-inode hardlinks,
not new weight copies. The provider build first reproduces the preceding
HC provider byte-for-byte, then replaces only `scaled_silu.cc` and the
contrib registry.

Six two-output native fixture pairs, three executions each, passed full
WebGPU validation with CPU fallback disabled: **19,506 finite FP16 elements
matched bitwise**. Coverage includes M=1/2/16, the actual 320/4 split,
odd 7/3 widths, 1/1 widths, and positive/zero/negative alpha. Profiling
requires the actual `SplitScaledSiLUF16` dispatch, not successful function
inlining. Earlier registration attempts logged capability failures and
inlined the reference body; they are retained but excluded from fused
qualification. Invalid split totals are rejected by function shape
inference; an empty prefix reaches the fused kernel's explicit positive-size
error. Both negative executions return exit1.

Two bounded full-model decode profiles confirm **1941 kernels/token**:
96 fewer than the preceding HC-fused graph's 2037, and 481 fewer than the
original graph's 2422. One unfused ScaledSiLU remains at the final mixer.
Both profiled 32-token outputs match their unprofiled prefixes.

An unprofiled GPU0 2K run with one excluded warmup and three measured
256-token requests reached **47.9716 / 49.0061 / 49.1821 tok/s**:
median **49.0061**, aggregate **48.7141 tok/s**. All four full sequences
match the original-graph control exactly; all prefill/decode phases
recorded zero process major page faults. Measured TTFTs were
20.3743 / 23.3631 / 25.8236 seconds; the variation is retained without
attributing a cause on the shared host.

Submission-window retuning did not improve the result: window128 measured
47.8393 tok/s median and window32 measured 47.3069, each with three
256-token samples and all four original sequences exact. Window64 remains
selected. These are sequential checks, not a counterbalanced causal
speedup qualification. No new 8K/32K table or whole-model quality
promotion is claimed, and 100 tok/s remains unmet.

The retained `fuse_split_silu.py`, multi-output `hc_fusion_compare.cpp`
and `qualify_split_silu.py` reproduce the rewrite, strict native checks
and persisted identity/dispatch/token-parity gates.
`split-silu-qualification.json` records every submission-window row and
the excluded registration attempts.

### Exact GDN projection/preparation fusion

The next opt-in old-model derivative adds `GdnProjPrepF16` after the HC and
Split/ScaledSiLU stages. It replaces 36 chains, each containing two Splits,
two FP16-to-FP32 Casts, two Transposes and `CausalConvWithState`, with one
WebGPU dispatch. The reference local function retains those original
operations; no original-model graph, weights or configuration is modified.

The narrow kernel consumes FP16 packed `[B,L,16384]` QKV/Z and `[B,L,96]`
A/B projections, convolution weights and bias, and channels-first
`[B,10240,3]` convolution history. It produces channels-last convolution
output and Z, FP32 A/B gates, and channels-first updated history.
`GatedDeltaNet` itself, its packed-input route and recurrent cache layout
remain unchanged. Convolution accumulation and SiLU preserve the original
FP16 arithmetic boundaries. This is projection **preparation**, not a
fusion of either projection matrix multiplication.

Seven five-output native fixture pairs, each run three times with full
WebGPU validation and CPU fallback disabled, passed **2,121,600 finite,
bitwise-exact FP16/FP32 element comparisons** against the unfused GPU path.
They cover batch sizes 1/2 and sequence lengths 0/1/2/3/4/16, including
unchanged history for empty input. Profiles prove the fused dispatch
actually executes instead of local-function inlining. Two invalid
history layouts reject explicitly in the fused kernel. Thirteen focused
rewrite/reference tests pass, and the combined HC/Split/GDN set has 39
passing tests. This is strict GPU arithmetic preservation, not a passed
CPU/GPU numerical-quality gate.

Two bounded full-model decode profiles contain **1,725 kernels/token**,
down from 1,941 at the previous stage and 2,422 on the original graph:
216 fewer at this stage, 697 fewer in total. Both contain 36 actual
`GdnProjPrepF16` kernels and retain the qualified HC and Split fusions.

At 2K, window64, one excluded warmup and three measured 256-output
requests produced **51.2377 / 51.5463 / 51.4557 tok/s**, median
**51.4557**, aggregate **51.4129**. All four complete sequences exactly
match the original-graph control. This is a standalone sequential timing
check, not a counterbalanced causal speedup. It does not replace the
complete original-graph 2K/8K/32K context table, establish whole-model
quality, or meet the 100-tok/s goal.

The subsequent 8K check on the identical provider and graph measured
**48.8962 / 50.2211 / 48.3506 tok/s**, median **48.8962**, aggregate
**49.1435**, again with all four original-control 256-token sequences
exact and no prefill/decode process major page faults. The 2K and 8K
rows are recorded in `gdn-prep-context-qualification.json`.

The completed 32K check measured **41.1131 / 41.2940 / 41.0024 tok/s**,
median **41.1131**, aggregate **41.1361**. All four original-control
256-token sequences are exact, with zero prefill/decode process major
page faults. Measured TTFTs were 400.1242 / 401.4000 / 417.2473 seconds.
The complete GDN-only fused-provider median table is therefore
**51.4557 / 48.8962 / 41.1131 tok/s at 2K / 8K / 32K**. All twelve
original sequences, including the three excluded warmups, match.
This table excludes the later exact-dense optimization described below;
it is a separately qualified derivative, not a whole-model quality
promotion or a counterbalanced comparison.

`fuse_gdn_prep.py`, `make_gdn_prep_fixtures.py`, the mixed-output
`hc_fusion_compare.cpp`, and `qualify_gdn_prep.py` retain reproducible
rewrite, native validation and identity/dispatch/token-parity gates in
the local evidence directory. `gdn-prep-qualification.json` records the
validated provider, graph and unchanged external-data identities.

### Exact HC dense projection on the GDN-fused stack

The opt-in `ORT_WEBGPU_HC_EXACT_DENSE=1` selector retains the FP16
arithmetic and reduction order of the original HC projection path while
using register partials to eliminate repeated workgroup barriers. Only
the MatMulNBits translation unit changes in this stage. It does not use
the earlier, sequence-changing hyper-skinny experiment.

The combined provider was built over the qualified GDN preparation
provider with a byte-identical baseline relink gate. **131 finite,
bitwise-exact native fixtures** pass, including all 96 actual HC weight
tensors. A full-model profile proves 96 actual `HCExactDense` dispatches
and 1,725 total kernels/token, with no capability fallback.

On the identical old-model GDN-fused graph, an ordered control/candidate
comparison at 2K with three measured 256-token requests per provider
gave **51.4475 -> 54.0633 tok/s median** (+5.08%) and
**51.0097 -> 54.1313 aggregate**. All eight complete sequences, including
both excluded warmups, exactly match the original graph; no major
process page faults were observed. This is a provider-block ordered
comparison, not a counterbalanced causal speedup. It does not establish
whole-model numerical quality or a new longer-context table.

The combined provider SHA-256 is
`77e49325eb0bb7f55b46f25f570e42c8cdb5d1c332ae0d49b6a62f9b184ba0ca`.
The isolated dense experiment's `combined/qualification.json` retains
fixture, selector, graph/config/weight identity and full-sequence checks.
The selector remains opt-in, and 100 tok/s remains unmet.

The combined provider's subsequent 8K run measured
**52.7309 / 52.5172 / 52.6642 tok/s**, median **52.6642**, aggregate
**52.6373**, with one excluded warmup and three measured 256-token
requests. All four complete sequences match the original graph exactly;
all prefill/decode process major-page-fault counts are zero.
`gdn-exact-dense-8k-qualification.json` records this separate, sequential
longer-context check. It is not a paired causal comparison against the
earlier GDN-only 8K row or a complete 2K/8K/32K table.

The same exact-HC provider's completed 32K check measured
**43.2768 / 43.1579 / 43.1084 tok/s**, median **43.1579**, aggregate
**43.1809**. All four original-control 256-token sequences are exact,
with zero prefill/decode process major page faults. Measured TTFTs were
400.7256 / 400.8127 / 391.4454 seconds.
`gdn-exact-dense-32k-qualification.json` records the run. Together with
the qualified 2K/8K rows, this provider has a complete median table of
**54.0633 / 52.6642 / 43.1579 tok/s at 2K / 8K / 32K**. It excludes
the later exact-mid optimization, whose 32K throughput is unmeasured;
do not combine its 2K/8K rows with this 32K result as a same-stack table.
Whole-model quality and counterbalanced causal speedup remain unclaimed.

### Exact mid-width dense projection follow-up

A further opt-in `ORT_WEBGPU_EXACT_DENSE_MID=1` selector, used together
with `ORT_WEBGPU_HC_EXACT_DENSE=1`, extends exact register-partial
execution to 156 additional real projection dispatches. HC-up remains
unchanged. The combined native set has **287 finite, bitwise-exact
fixtures**; a full-model profile proves the 156 new dispatches and
reduces their summed instrumented time from 2,338 to 1,821 microseconds.
Those profiled timings are diagnostic, not unprofiled throughput.

On the identical old-model GDN-fused graph, an ordered control/candidate
2K comparison with three measured 256-token requests per provider gave
**54.3672 -> 56.1799 tok/s median** (+3.33%) and
**54.2138 -> 56.1610 aggregate** (+3.59%). All eight original-control
256-token sequences, including both excluded warmups, are exact.
No capability fallback or major process page faults were observed.
This remains a provider-block ordered comparison, not a counterbalanced
causal speedup or a whole-model quality qualification.

Only the main MatMulNBits translation unit was updated, preserving the
existing dirty changes and previous exact-HC route. Integration rebuilt
the prior provider byte-identically, then recompiled the updated object
and provider byte-identically to the isolated, qualified candidate:
SHA-256 `2a92e92a66dee76a83ceee23e834c5d0e2281a932a424f6ca62f6b6503b27f4c`.
No redundant GPU or TPS rerun was used to claim integration parity.
The isolated experiment's `followup2560/qualification.json` and
`followup2560/integration/stage-snapshot.json` retain the validation and
integration provenance. Both selectors remain opt-in; 100 tok/s is
still unmet.

The exact-mid provider's separate 8K run measured
**52.8911 / 53.9262 / 53.5942 tok/s**, median **53.5942**, aggregate
**53.4670**. All four original-control 256-token sequences, including
the excluded warmup, are exact, with zero prefill/decode process major
page faults. `gdn-exact-mid-8k-qualification.json` records this
standalone sequential check; it is not a counterbalanced comparison
against the earlier exact-HC-only 8K row or a complete context table.

The completed exact-mid 32K check subsequently measured **42.8232 /
43.4076 / 40.9736 tok/s**, median **42.8232**. All four original
256-token sequences are exact, with zero process major page faults.
Its same-stack complete median table is **56.1799 / 53.5942 / 42.8232
tok/s at 2K / 8K / 32K**. The 32K observation is not an improvement
over the earlier exact-HC-only median of 43.1579.

### Opt-in shared-expert sigmoid/add/post-mix fusion

`HyperConnectionSharedPostMixF16` additionally combines 48 shared-gate
`Sigmoid -> GatedAdd -> HyperConnectionPostMixGateF16` chains in an
explicit model-local derivative. Original initializer descriptors,
external-data inodes and configuration are unchanged. The retained
scalar 64-thread provider is SHA-256
`f65f28b1521f0106f14b49dea79047387f451ef4d3c3294b75bc87f7d946913d`.
Its build first reproduces the retained GDN provider byte-identically,
then compiles the HC-post, qualified exact-dense and registry objects.

Sixteen native fixture pairs, three runs each under full validation
with CPU fallback disabled, passed **1,107,510 finite bitwise-exact
FP16 elements**. Actual fused-only fixture dispatches are verified,
including empty tensors, tails, multiple rows, branch-singleton and
feature-grouped gates. Two invalid shared-output/gate shapes explicitly
reject in the fused kernel. Earlier WGSL/fixture failures and a blocked
GPU2 attempt are retained; successful final negatives used GPU1.

Two full-model decode profiles prove **1,629 kernels/token**, down
from 1,725: 48 new fused kernels, no `GatedAdd`, and 12 remaining
`Sigmoid` kernels. With both exact-dense selectors enabled in both
modes, same-provider candidate-before-control 2K blocks measured
**56.0490 versus 53.4495 tok/s median** (aggregates **56.0350 versus
53.5296**). All eight complete original-control 256-token sequences,
including excluded warmups, are exact, with zero process major faults.
This is not a counterbalanced causal speedup. Scalar256 and vec4
alternatives and their variable shared-host timings remain in evidence;
scalar64 is retained. No shared-fusion 8K/32K table, whole-model quality
promotion or 100 tok/s claim is made.

`qualify_shared_post.py` reproducibly checks hashes, original data/config
inodes, native negatives, actual dispatches, all original sequences,
submission flags and resource counters, and writes
`shared-post-qualification.json` in the local performance evidence directory.

### Current-stack MLP fusion follow-up

The existing derived graph with 48 norm-free MLP fusions was retested with
the current A9 fast stack and submission window 64. Three actual-converter
M=1/2/16 native fixtures passed full validation against their references.
A standalone 2K run with one excluded warmup and three measured 256-token
requests reached median 47.1318 tok/s (aggregate 47.1380). Measured TTFTs
were 20.5415, 20.5170 and 35.6598 seconds; the third request was anomalous
on the shared host. This was not a counterbalanced graph comparison.
Generated sanity sequences differ from the earlier fused run, and only
two of nine strict answer checks passed. It is not an accuracy-qualified
promotion or a replacement for the complete original-graph context table.

A separate GPU-less CPU control could not load the original graph: the pinned
ORT core has no CPU kernel for `com.microsoft.GatedDeltaNet(1)`. No CPU answers
were generated, so this does not locate the whole-model quality defect in
either the export or the WebGPU implementation.

### CUDA-to-WebGPU transfer boundary

The local Tianlei source package, `qwen38-flash-ops`, is not yet a completely
ported WebGPU stack. Its CUDA libraries target BF16/FP8/NVFP4 layouts and
include fused graph operators as well as kernel optimizations. The original
WebGPU model instead uses RTN INT4/FP16, so equivalent transformations must
preserve that graph's own rounding, weight layout and cache semantics.

| Area | WebGPU status |
| --- | --- |
| Few-row, K-parallel GEMV | Adapted in WGSL, with guarded selectors and native fixtures; not a literal CUDA kernel port. |
| HC pointwise chains | Opt-in FP16 pre-mix sigmoid and post-mix Div/Sigmoid/Mul fusion qualified as described above. |
| Full `HcBlock` / `HcPostBlock`, including norm and projection phases | Not transferred end-to-end. The existing HC fusion removes pointwise dispatches, not whole HC blocks. |
| GDN convolution/projection preparation and QSA projection preparation | The opt-in narrow GDN preparation fusion described above is qualified on the original INT4 graph. Projection matrix multiplies and QSA projection preparation are not merged; this is not the complete CUDA `GdnConvPrep` / `GdnProjPrep` / `QsaProjPrep` implementation. |
| QSA selection and sparse attention | WebGPU-specific score/rank/map and attention work exists; it is not the identical CUDA CUB/capture implementation or a matched quality-qualified reproduction. |
| Command recording and graph replay | Batching, dynamic uniforms and binding reuse are implemented. Whole-model capture remains blocked by fixed-cache/sequence-length semantics, including the QSA indexer restriction. |
| CUDA PDL, cluster/co-resident-CTA handoffs | No direct WebGPU equivalent. Do not copy software grid spin barriers into WGSL: workgroup co-residency/progress is not guaranteed. |
| Folded speculative/MTP cycle | Separate experimental work, not the current non-MTP throughput workload or a completed transfer. |

Tianlei's CUDA README explicitly requires all 168 CTAs of the fused HC
kernel to be co-resident and prohibits another GPU process during that
execution. Replacing that mechanism requires a safe staged WebGPU design,
not merely translating the arithmetic or removing dispatch boundaries.

### RC14 native packed INT3 bring-up

The RC14 export is a different architecture, not another precision of the
earlier Flash Next graph: 42 attention layers, 21 QMoE blocks, 512 experts
and top-8 routing. Its BF16 floating tensors and compact sliding-window
KV-cache mode are not supported by the retained WebGPU runtime.

The 17.1970 tok/s 2K result already enables dynamic uniforms, an 8192-entry
binding cache, the uniform arena, small-transfer improvements, native
packed-INT3 QMoE and the dedicated FP32 dense GEMV path. Static reshape
specialization also removes 42 host shape-transfer events per decode.
This is not an unoptimized run: the successive explicit FP32/full-cache
experiments reached 9.3220, 14.2506 and then 17.1970 tok/s.
However, INT4/FP16-specialized GEMV and Flash Next HC/QSA graph fusions
cannot simply be enabled on RC14's different operators and FP32 tensors.
Setting an INT4 optimization flag does not imply a matching INT3 path ran.

Only the routed expert weights are native INT3 here; the floating
activations, dense weights and caches in the working derivative are FP32.
The earlier pre-dense-GEMV profile measured 295 dense MatMul kernels at
37.483 ms summed device time versus 42 packed-INT3 expert MatMul kernels
at 3.485 ms. Those are instrumented diagnostic sums, not current
unprofiled fractions, but they identify why reducing expert bit-width
alone does not make the entire different model faster. Original BF16
arithmetic/compact-cache parity and whole-model quality remain unqualified.

The experimental QMoE implementation accepts symmetric, contiguous
little-endian INT3 weights with block size 32. It reads cross-word 3-bit
codes directly, uses implicit zero point 4, and accumulates in FP32.
FP16 and FP32 activations/scales are supported. Explicit zero points,
other block sizes, and unsupported dispatch dimensions return errors.
This is internal QMoE support; public `MatMulNBits(bits=3)` support is
not claimed. Existing INT4/NVFP4 selectors retain their original guards.

`ORT_WEBGPU_QMOE_INT3_PREFILL=1` is default-off and enables the existing
indirect small-batch route for symmetric INT3, up to 16 tokens and 256
routed rows. The default expert-by-expert route is also tested.
The isolated provider is
`76fd46df68030324a911bc1c5e07bd0407eb93a9d7f2a9329e995fa2e1ff8698`.
Full-validation independent-reference tests cover FP16/FP32, M=1/2/4/16,
realistic K=1536/intermediate=2688 expert dimensions, and 13 existing
INT4 fixtures. Separate local-attention tests cover rotary, the
511-visible-key boundary, and head width 128 with capacity 2304.
CUDA's implementation passes `local_window_size - 1` as FlashAttention's
left window; the initial 512-visible-key reference was corrected to this
511-key convention without changing test tolerances.

Whole-model diagnostics use explicitly labeled floating-precision and
full-cache derivatives; packed INT3 initializers are unchanged. FP16
2K runs failed the finite-logit check, including with FP32 MatMul
accumulation. Intermediate-output diagnostics first observed nonfinite
final-layer QKV at processed=304/total=320, also before the final FP16
cast; a cast-overflow root cause is therefore not established.
BF16-to-FP32 conversion preserves all initializer values exactly but
still changes activation arithmetic and is not original-model parity.
No standard-context performance or whole-model quality qualification
follows from operator-level tests or the short FP16 smoke result.
The retained CUDA EPv2 plugin explicitly rejects 3-bit QMoE weights;
no native CUDA INT3 throughput is available. Full-cache/compact-cache
whole-model equivalence and tokenizer-based quality remain unqualified.

The first completed RC14 2K/256-output table uses the FP32/full-cache
derivative, raw ORT IOBinding and deterministic synthetic prompt IDs.
Three measured samples after an excluded warmup reached
9.32203 / 9.90221 / 8.56009 tok/s, median 9.32203 tok/s.
Each sample generated exactly 256 tokens over 255 timed decode intervals
and passed the last-position finite-logit checks. TTFTs were
24.7653 / 24.9623 / 23.6440 seconds. This is an initial unprofiled baseline,
not a performance target achievement, original-BF16 parity or a
precision-only comparison against the earlier INT4 export. All 338
converted nonempty BF16 initializers were byte-verified as exact FP32
values, with zero weight-conversion error. Native coverage additionally
includes FP16/FP32 bias and specific unsupported-zero-point errors.

#### Eliminating RC14 shape round trips

A bounded one-decode profile found 21 uploads of 24-byte shape tensors
and 21 readbacks of 16-byte shape tensors. Their host events accounted
for 85.747 ms of a 129.594 ms instrumented step; these are diagnostic
timings, not an unprofiled time attribution. Each MLP reshaped its
rank-3 input with `[-1, Shape(input)[2]]`, although dimension 2 was
already statically declared as 3584.

A guarded execution-graph derivative replaces those 21 shape inputs
with `[-1, 3584]` and removes only their 84 otherwise-unused
Shape/Gather/Unsqueeze/Concat nodes. All original initializer descriptors
and all other computation nodes are byte-identical. CPU fallback remains
disabled. Separate before/after profiles prove 42 transfer nodes become
zero while retaining 295 MatMul, 42 GQA and 21 QMoE GPU nodes.

Three unprofiled 2K/256-output samples reached
14.2324 / 14.2506 / 14.2543 tok/s, median 14.2506 tok/s.
TTFTs were 18.9616 / 19.6663 / 18.9451 seconds.
All three measured sequences and the excluded warmup exactly match the
FP32/full-cache baseline. This standalone observation was not
counterbalanced against that earlier process and does not establish
whole-model quality.

#### Default-off dense FP32 decode GEMV

`ORT_WEBGPU_MATMUL_F32_DECODE_GEMV=1` reuses the existing FP32-reducing
GEMV template for NVIDIA FP32 inputs/weights/output, one batch and one
row, K=512..8192, N>=128 divisible by four, and no bias or fused
activation. Workgroups use four adjacent vec4 columns and 32 K lanes.
The existing FP16 small-N/N96 selectors and all flag-off paths are
unchanged. The isolated experimental provider is
`0080431394e1d87447927b1d6f65e0f7e3691d817624a3c89d4a14ce89330bfe`.

Ten independent NumPy dense fixtures passed in both flag modes with
full validation, including actual 3584x201088 LM-head weights, tail
columns, M=2 fallback and an exactly representable 640000 output above
FP16's range. Nineteen existing INT3/INT4 reference fixtures also pass
with the new dense object linked. Native evidence alone is not a
whole-model speedup or quality qualification; model timing uses
same-provider counterbalanced off/on pairs with identical prompt IDs.

The completed 2K comparison used three measured off/on pairs plus an
excluded warmup pair, alternating order within one loaded provider and
the same static-shape graph. Control median/aggregate were
14.5760/14.5763 tok/s; candidate median/aggregate were
17.1970/17.1919 tok/s. Candidate samples were
17.1970 / 17.0009 / 17.3822 tok/s, about 18% higher median.
All four paired 256-token sequences were exact. The first measured
control TTFT was anomalous at 34.1833 seconds versus about 18.8 seconds
for the other measured TTFTs; it was retained and its cause is not
established. Decode results are unprofiled and do not promote the
default-off flag or establish original-BF16/whole-model accuracy.

The subsequent 8K three-measurement attempt and exploratory 32K
single-measurement attempt each hit the external 900-second run cap.
Both owned containers were stopped and their final guard status is
`run_timed_out`. The 8K attempt completed only its excluded warmup
(TTFT 312.135 seconds, decode 159.446 seconds); the 32K attempt produced
no complete sample. Neither supplies headline throughput or a completed
standard-context row. Partial records remain in the qualification report
with explicit timeout errors; they are not silently omitted or averaged
into the completed 2K result.

### October 29: original INT4 repro and targeted transfer experiments

The original INT4 model remains the throughput target; the newer INT3
model and Dawn FP8 work are separate. GPU 7 is the primary authorized
adapter. Small native fixtures use idle GPU 0; a foreign GPU 1 process
is left untouched. All new provider candidates are isolated, default-off
experiments, not changes to the enabled baseline.

Fresh 2K runs use chunk 16, pending-dispatch limit 64, one excluded
warmup and three measured 256-output requests (255 timed decode
intervals). Both complete original sequences, including warmup, and
zero prefill/decode major page faults are required.

| Provider | Median decode tok/s | Aggregate tok/s | Qualification |
|---|---:|---:|---|
| Current QuickGelu baseline | 55.6137 | 55.5708 | All four sequences exact; all major faults zero |
| Filtered exact-wide projection | 55.1912 | 55.0311 | All four sequences exact; all major faults zero |

These are separate model loads on GPU 7, not same-session matched
controls. They do not establish a filtered-route improvement or improve
the historical approximately 57.33 tok/s best 2K row. A separate
complementary-window trial suggested a 1.0207 geometric ratio, but its
first prefill incurred five major faults. That trial failed strict
cleanliness and is not qualified throughput evidence.

The CUDA transfer audit inspected Tianlei's operation inventory and the
actual small-N GEMV, sparse-attention, quantized-MoE and IndexShare changes.
Existing WGSL K-parallel GEMV, HC pointwise stages, GDN preparation and
sparse-attention work already cover parts of these ideas. Complete HC
block/projection-preparation fusion and safe whole-model graph capture
remain gaps. CUDA-specific BF16/FP8/NVFP4 layouts, PDL and co-resident-CTA
synchronization cannot simply replace this model's INT4/FP16 WGSL paths.
Do not introduce grid-wide spin barriers to imitate CUDA coordination.

Native experiments rejected before expensive full-model timing:

- Exact-order expert register partials: actual original-weight QMoE
  fixtures across layers 0, 3, 7, 23, 35 and 47. Four-column and
  eight-column candidates each proved 96 actual candidate dispatches
  and finite bitwise outputs. They were slower than baseline GPU
  kernels (approximately 30/24 and 31/25 microseconds for FC1/FC2,
  versus 27/22), so neither was promoted. An earlier no-bias selector
  never selected the candidate on biased model fixtures; its apparent
  parity result is explicitly fallback-only, not candidate evidence.
- Exact-order branch normalization: 60 tail/scale fixtures passed.
  The 32-thread reconstruction was slower (approximately 33 versus
  15 microseconds for hidden 2560); the 256-thread subgroup variant
  remained approximately tied. Neither supplies a TPS improvement.

A bounded large-weight bind-group cache passed six actual-weight QMoE
fixtures with six off/on timing samples per mode and finite bitwise
outputs. Cache statistics prove hits and about 1.42 GB of referenced
buffers, including larger weights excluded by the old 256 MiB budget.
The experimental bounds are 4 GiB per entry, 96 GiB unique referenced
buffers and 8192 entries; no weight copies are created. Clearing when
switching back to the small budget is necessary. This H200-specific,
default-off experiment is not a consumer-GPU default; microfixture host
timings are not model TPS.

The clean GPU 7 full-model follow-up completed four complementary
288-output requests, with all complete sequences equal, all original
256-token prefixes exact and all major faults zero. Across 16 matched
windows the old-cache median was 56.1030 tok/s versus 55.7429 for the
large cache, with geometric matched ratio 0.99723. Statistics prove
1,576,950 hits, 8192 peak entries and 71,394,993,008 peak referenced
bytes, within the 96 GiB bound. GPU 7 allocations were released at
process exit. The larger cache did not improve throughput and is not
promoted. More binding reuse alone does not establish a model speedup.

The staged HC post-mix/normalization rewrite targets 95 adjacent chains
while preserving both the post-mix and normalized outputs. Native
qualification must prove actual fused dispatches, not local-function
fallback. An initial native candidate reproduced the post-mix output
but changed normalized FP16 values and was rejected. Kernel-count
reduction and full-model benefit remain unproven until a corrected
candidate passes. The corrected candidate makes its FP16 storage
boundary explicit before the RMS reduction, and passed all 32 native
fixture pairs: 5,287,680 finite bitwise element comparisons and 96
actual fused dispatches. Native hidden-2560 single-row timings were
approximately 15/16 microseconds for ordinary/shared fused kernels,
versus 25/27 for the respective two-kernel reference chains.
Those are native kernel timings, not full-model TPS. GPU 0 acquired
a new foreign workload before this validation; its guard refused
the launch and work moved to idle GPU 7 without disturbing that job.
No experiment here establishes 100 tok/s or whole-model quality
qualification.

The corrected HC candidate also passed full-model route qualification:
95 actual fused dispatches and 1486 total kernels per profiled decode
step, versus the previous 1581. Both profiled 32-token prefixes matched
the originals with zero major faults. Its subsequent unprofiled 2K
standard row was 54.2015 median / 53.0766 aggregate tok/s (measured
samples 54.2015, 50.7429 and 54.4509), with all four original complete
256-token sequences exact and all major faults zero. This did not
improve the separately loaded current baseline; fewer dispatches and
faster native microfixtures alone are not end-to-end success. The
candidate remains isolated rather than replacing the baseline.

A subsequent GDN decode experiment targets an identity copy rather
than changing the recurrence arithmetic. With exactly one total packed
QKV token, token-major packed Q/K/V is byte-identical to the temporary
concatenated Q/K/V layout. `ORT_WEBGPU_GDN_PACKED_DECODE_VIEW=1`
therefore binds read-only views of the original input instead of
allocating and dispatching an unpack copy. More than one total token,
including two batches each containing one token, still uses the
original allocation/copy path. Writable outputs always refer to the
owned temporary, never the borrowed input.

The isolated original-baseline provider passed 36 FP16/FP32 native
fixtures, including tail dimensions, actual 16-Q/48-V-head dimensions,
Q/K normalization on/off, and both multi-token fallback layouts.
Both recurrence outputs and FP32 states were finite and bitwise exact:
38,233,128 element comparisons. Profiles proved the unpack dispatch
was removed in all 12 eligible cases and retained in all 24 fallback
cases.

The subsequent full-model profile proved 1545 kernels per decode step
(36 real unpack dispatches removed from 1581), with both original
32-token prefixes exact and all major faults zero. Clean, same-provider
complementary 2K windows then measured:

| Packed GDN decode view | Median window tok/s |
|---|---:|
| Off | 54.1222 |
| On | 54.7897 |

The geometric ratio across 16 matched pairs was 1.01111. All four
complete 288-token sequences were equal, their original 256-token
prefixes matched, and all prefill/decode major faults were zero.
This is a small observed matched-window gain, not a standard-context
record or a statistically established stable speedup. Whole-model
quality remains unqualified. Dispatch-limit tuning must use fresh
standard rows and retain endpoint controls rather than comparing
selected samples with old results.

The completed GDN-view pending-dispatch sweep retained the original
64-dispatch limit:

| Sweep order | Pending dispatches | Median decode tok/s | Aggregate decode tok/s |
|---|---:|---:|---:|
| Initial control | 64 | 55.7804 | 55.1288 |
| Candidate | 128 | 54.8274 | 55.1317 |
| Candidate | 256 | 51.0662 | 51.0881 |
| Candidate | 96 | 53.6422 | 53.6740 |
| Candidate | 32 | 54.7889 | 54.2535 |
| Final control | 64 | 56.3679 | 56.4315 |

Every row passed all four original 256-token sequences and the
zero-major-fault checks. These are separate-load tuning rows with
repeated endpoint controls, not matched causal speedup measurements.

Post-mix/RMSNorm fusion revisions v6 and v7 were rejected: conversion
intrinsics did not reproduce the original FP16 storage boundary
exactly. In particular, the v6 `quantizeToF16` path flushed an observed
FP16 subnormal to zero. Revision v8 instead stores intermediate
post-mix values in actual FP16 workgroup memory, synchronizes, and
reloads them for the norm. It checks the device workgroup-memory limit
and includes hidden size in the program cache key. At hidden size 2560,
the staging and FP32 reduction arrays require 6144 bytes.

The v8 native qualification passed 32 ordinary pairs and 16 explicit
subnormal/rounding-tie pairs: 9,519,456 finite bitwise element
comparisons and 144 actual fused dispatches. Native timing improvements
are not a model TPS result.

The combined HCv8/GDN provider was assembled only after both component
relinks reproduced their original library bytes. An initial
different-basename relink gate failed and was retained as failed
evidence; the successful verification used the original library
basename in separate directories. The full-model profile then proved
95 actual HC post/norm fusions, 36 GDN kernels without the unpack
dispatches, and **1450 kernels per decode step**, versus the original
1581. Both original 32-token prefixes matched, with zero major faults.

Both stacks subsequently completed the required standard context table:

| Isolated stack | GPU | 2K median tok/s | 8K median tok/s | 32K median tok/s |
|---|---:|---:|---:|---:|
| QuickGelu + GDN decode view | 7 | 56.3679 | 52.8166 | 43.4302 |
| QuickGelu + HCv8 post/norm + GDN decode view | 3 | 57.4162 | 54.8458 | 44.1205 |

Each context used an excluded warmup plus three measured 256-output
requests, 255 timed decode intervals, chunk 16 and pending 64. Every
warmup/measured prompt hash and complete output sequence matched the
frozen original control, and every prefill/decode major-fault count
was zero. The first 2K row is the final sweep control above. The
evidence prefixes are `gdn-decode-view-standard8k-20261029`,
`gdn-decode-view-standard32k-20261029`, and
`hc-gdn-combined-gpu3-standard{2k,8k,32k}-20261029`.

These are different-GPU, separate-load observations, not a matched
comparison between stacks. Neither stack beats the earlier clean
32K median of 44.4925 tok/s, and **100 tok/s remains unmet**. Token
parity does not qualify whole-model quality or logits; all reports
retain `quality_qualified=false` and `headline_eligible=false`.
Promising future candidates require 2K, 8K and 32K coverage before
promotion; a 2K-only screen is insufficient.

### Scoped QMoE command reuse experiment

The original INT4 decode trace contains five QMoE programs per layer:
gate, FC1, activation, FC2 and weighted mix, for 240 dispatches across
48 layers. An isolated default-off
`ORT_WEBGPU_QMOE_COMMAND_REPLAY=1` experiment reuses their prepared
commands, rather than attempting to capture dynamic attention or the
whole model.

The cache key includes every input/output buffer identity, type and
shape, relevant kernel-selection flags and profiling state. Temporary
QMoE tensors have retained ownership, and captured uniform buffers
never return to a reusable uniform allocator. Hits append commands
to the existing ordered deferred queue and preserve the ordinary
pending-dispatch limit; they do not force a submission per QMoE.
Each kernel retains at most eight command entries with LRU replacement.
A changed tensor layout flushes pending work and invalidates the
entries and owned workspace. Whole-graph capture and active profiling
use the original uncached compute path.

The v5 native qualification passed 18 real-weight fixtures with
changing activations and routing, 32 rounds each, for 9,338,880
finite bitwise comparisons. Six single-token fixtures proved real
reuse and LRU eviction after kernel-selection flag changes. Twelve
multi-token fixtures retained the original path. The tests also
covered pending limits 1 and 64 and explicit rejection of invalid
flag value 2. These are native correctness results, not model TPS.

An earlier v3 model profile failed its dispatch-count gate: capture
suppressed timestamps for the 240 QMoE dispatches. This was not a
kernel-count improvement, and that run was not qualified. Revision v5
explicitly uses uncached execution while profiling. Its model profile
retained all 1545 expected kernels, matched both original 32-token
prefixes, and had zero major faults. The surrounding unprofiled
decode calls recorded 2304 hits from 2880 eligible node calls (80%),
with 576 captures and 192 evictions; all 48 kernels stayed within
eight entries. The retained evidence prefix is
`qmoe-node-replay-v5-model-profile-20261029`. End-to-end matched-window
and full-context performance remain separate gates.

The subsequent clean same-provider GPU7 complementary windows passed:

| QMoE command reuse | Median window tok/s |
|---|---:|
| Off | 55.9632 |
| On | 58.5814 |

Across 16 matched pairs, the geometric ratio was 1.0482806 (+4.83%).
All four 288-token sequences were equal, every original 256-token
prefix matched, and all prefill/decode major faults were zero.
Counters recorded 26,304 hits from 27,648 eligible calls (95.14%),
1344 captures and 960 evictions, bounded to eight entries per kernel.
This is a warmed complementary-window gain, not a standard-context
record or statistical stability claim.

The subsequent standard 2K run measured median 57.3954 tok/s and
aggregate 57.4113 tok/s. All four original sequences and all fault
checks passed. Its 47,808 hits from 48,960 eligible calls (97.65%)
include cold command setup, which the window warmup excludes.
The standard 8K/32K checks and a combined HCv8/replay experiment are
separate promotion gates. A further native changing-input comparison
against scalar-gate execution also passed 9,338,880 bitwise finite
checks with the subgroup-gate flag exercised; this alone does not
establish a subgroup-gate model speedup.

The standard replay 8K run subsequently passed at median 57.0303 tok/s
(aggregate 57.0646), with all four original sequences exact, zero major
faults and the same 97.65% command-cache hit fraction. Its 32K run is
still a separate gate.

The combined HCv8/GDN/replay provider
`f1b79c2c947a18b8373bed7a96d970f22eb8f25d249cce2cf51cba6cadbd88fa`
passed a GPU3 model profile with all 1450 expected kernels, exact
original prefixes and zero major faults. Complementary warmed 2K
windows measured 57.5237 tok/s with replay off and 59.9212 with it on;
the geometric gain across 16 matched pairs was 4.4870%. All four full
288-token sequences matched, including their original 256-token
prefixes. Command counters recorded 26,448 hits from 27,648 calls.

The combined standard 2K run then passed at median 59.8693 tok/s and
aggregate 60.0672 tok/s, again with all original sequences exact and
zero major faults. Combined standard 8K/32K remain promotion gates.
These separate-load GPU3 observations are not causal comparisons
against the GPU7 results, and none establishes the 100 tok/s target
or whole-model quality qualification.

### Broader static-node reuse: rejected screens and binding-aware follow-up

A separate default-off `ORT_WEBGPU_STATIC_NODE_REPLAY=1` experiment
targets single-token FP16 SplitScaledSiLUF16, GatedRMSNorm and
QuickGeluMulF16. The initial implementation kept per-kernel prepared
commands, immutable uniforms and an I/O-identity LRU. It did not change
the base kernel or command-recording object layouts, and did not retain
borrowed intermediate scratch.

The eight-entry version passed 9,826,432 finite bitwise comparisons
across 11 changing-input fixtures, including changing CPU split sizes,
multi-token fallback, pending limits 1/64 and invalid-flag rejection.
The first fixture launch lacked the existing custom-op function
schemas; that failed run was retained, then corrected by reusing the
qualified fixture schemas rather than bypassing model validation.
Its model profile retained all 1450 kernels and original prefixes.
However, its 16-pair GPU0 screen regressed: 60.4007 tok/s off versus
41.0033 on, geometric ratio 0.7026065. Only 37.82% of eligible calls
hit the cache. It was rejected, not promoted to full-context testing.

A bounded 64-entry revision passed 19,652,864 changing-input bitwise
checks over 128 rounds per fixture, including more than 64 CPU split
variants to prove eviction. It improved the hit fraction to 79.39%
but still regressed in matched windows: 59.1937 tok/s off versus
55.0474 on, geometric ratio 0.9225280. It was also rejected.
Both screens matched every original token prefix and had zero major
faults; numerical correctness alone was insufficient for promotion.

The follow-up separates shape/CPU-value-dependent prepared state from
changing GPU buffer identities. A scoped capture records the original
binding layout, segmentation and immutable uniform buffer. Hits
rebind current parent inputs/outputs through the existing bind-group
builder/cache and append the ready dispatch without an extra
submission. Explicit per-operator binding roles avoid ambiguity when
two parent inputs alias. Initial capture validates that every shader
binding is exactly a declared parent tensor or the captured uniform
buffer, with no hidden temporary.

The same isolated follow-up includes the existing exact dense
MatMulNBits routes only: symmetric block32 INT4, NVIDIA, FP16
activations/scales, no bias/zero points and FP32 accumulation disabled,
for K=2560/N=512 or 640 and K=10240/N=324 with their existing selector
flags enabled. Multi-token, other routes, whole-graph capture and
profiling keep the original execution path. These changes do not
alter the shaders or claim a speedup before native and model tests.

The QMoE-only replay stack completed the GPU7 standard-context gates:

| Context | Median decode tok/s | Aggregate decode tok/s |
|---|---:|---:|
| 2K | 57.3954 | 57.4113 |
| 8K | 57.0303 | 57.0646 |
| 32K | 46.2284 | 44.8523 |

Every full 256-token sequence, including warmup, matched its frozen
original reference, and all prefill/decode major faults were zero.
Every context recorded 47,808 cache hits from 48,960 eligible calls.
The 32K median/aggregate difference is retained rather than hiding
run-to-run variation. This is a separate-load full-context table,
not a controlled attribution of each context's gain.

The combined HCv8/GDN/replay 8K standard run passed at median 56.3395
and aggregate 56.0216 tok/s; its 32K run is still pending.

Binding-aware revision v4 initially rejected a valid captured dynamic
uniform offset of zero. Immutable captured uniform allocation does
not require disabling dynamic-offset binding layouts. That failed
native run was retained; v5 preserves the original zero offset and
requires a dedicated non-arena uniform buffer. It subsequently passed
23,242,496 finite bitwise comparisons on GPU7 across 20 fixtures,
128 changing rounds each. Twelve multi-token fixtures retained the
original path; eight single-token fixtures proved actual rebinding.
The tests also exercised exact-dense selector changes, CPU split
changes/LRU eviction, invalid flag rejection and pending limits 1/64.
This is native evidence, not a model performance result.

The combined HCv8/GDN/replay standard table subsequently completed
on GPU3, with every original sequence exact and zero major faults:

| Context | Median decode tok/s | Aggregate decode tok/s |
|---|---:|---:|
| 2K | 59.8693 | 60.0672 |
| 8K | 56.3395 | 56.0216 |
| 32K | 46.6720 | 46.7183 |

These are separate-load observations, not causal comparisons against
the GPU7 table. The 100 tok/s target remains unmet.

Binding-aware v5 passed the actual-model 1450-kernel profile with all
432 intended nodes exercised, exact original prefixes and zero faults.
Its matched GPU7 windows were effectively neutral: 59.7700 tok/s off,
59.6942 on, geometric matched ratio 1.0031568. Counters recorded
248,400 metadata hits/rebindings from 248,832 calls, with just 432
initial captures. A high metadata hit fraction therefore did not
establish a useful speedup, and no standard-context promotion followed.

Revision v6 added explicit default-memory/WebGPU-buffer checks to
the rebound parent tensors and passed the same 23,242,496 finite
bitwise comparisons. Revision v7 additionally keeps a per-kernel
64-entry LRU of ready bound commands. A changed I/O identity prepares
a new binding without capture or an extra submission; a repeated
identity appends the existing ready command. Bound entries carry
monotonic prototype IDs, and prototype eviction removes its bound
entries. Both caches are independently bounded to 64 per kernel,
not 64 bindings for each of 64 prototypes. Native changing-input,
CPU-value, selector, eviction and fallback tests passed; actual-model
tests must prove both the rebind and bound-command-hit paths before
performance promotion.

The v7 GPU7 profile subsequently exercised all 432 intended nodes
and both binding paths, with 1450 kernels, exact original prefixes
and zero faults. Its complementary windows were marginal: 61.1146
tok/s off, 61.3501 on, geometric matched ratio 1.0046803.
The 248,400 metadata hits comprised 241,416 bound-command hits and
6,984 rebindings, with no binding eviction. A standard full-context
follow-up is recorded separately rather than treating this sub-1%
screen as a substantial improvement.

Revision v8 extended the same mechanism to 97 HC pre-mix and 95
HC post/norm programs. Eligibility uses the operator's validated
logical row size (branches times hidden), supporting both flattened
and grouped stream layouts. An initial native fixture incorrectly
combined grouped stream dimensions with the flattened-layout
attribute; the existing shape validator rejected it. That failed
evidence was retained. Corrected flattened fixtures plus a grouped
mixed-FP32-gate fixture passed 129,410,816 finite bitwise comparisons.
The actual-model profile exercised all 624 intended nodes and both
binding paths, retaining 1450 kernels and original prefixes.
GPU7 windows again showed no useful gain: 59.5090 off, 58.9478 on,
geometric ratio 0.9997051. No substantial speedup is claimed.

A separate default-off HC projection/activation fusion prototype
targets the 96 K10240/N324 projections feeding SplitScaledSiLUF16.
It preserves the qualified HC projection's FP16 dot/reduction
order and the activation's explicit FP16 roundings, storing the
two outputs directly. Unsupported decode routes and multi-token
inputs execute the original ApplyMatMulNBits and Split program,
with a normal temporary tensor; no hidden temporary is replayed.
The model-local rewrite requires symmetric block32 INT4, exactly
three matrix inputs, a single consumer, and no graph output of
the removed intermediate. Weights/config remain unchanged.
Native changing-input and changing-split tests passed 2,363,904
finite bitwise comparisons, including multi-token fallback,
selector toggles, invalid flags and bounded cache eviction.
This is not yet a whole-model quality qualification or a speed claim.

The fused candidate's actual-model profile then passed both original
32-token prefixes with zero major faults and exactly 1354 kernels
per profiled token (96 projection/activation programs, no separate
SplitScaledSiLUF16 program), versus 1450 in the retained stack.
All 528 intended replay nodes and both binding paths were observed.
Its GPU7 complementary windows produced 58.4443 tok/s with fusion
disabled and 61.7712 enabled; the geometric ratio across 16 matched
pairs was 1.0561333. All four 288-token sequences matched each other,
their original 256-token prefixes matched the frozen control, and
all page-fault checks were zero. Every request favored fusion.
This compares the new node's two-pass fallback against its fused
and prepared-command path within the same rewritten graph; it is
not a 5.61% improvement over the best historical standard table.
Binding counters included 12,902 evictions, so near-perfect metadata
hits should not be confused with perfect binding reuse.
Real-weight validation and standard 2K/8K/32K follow-ups are separate
gates; warmed windows do not establish standard decode TPS or
whole-model/logit quality.

The binding-only v7 follow-up completed on GPU3 with exact original
sequences and zero faults. Medians were 59.7767 / 56.2029 / 45.4134
tok/s at 2K / 8K / 32K; aggregates were 60.0009 / 56.6027 / 45.6391.
These observations do not improve the retained combined table.

The projection/SiLU candidate additionally passed all 96 actual
projection weight sets at activation scales 0.125, 1 and 4:
288 fixture pairs, three repeats, 279,936 finite bitwise comparisons
against the original two-node graph. Graph optimizations and CPU
fallback were disabled; command-capture counters proved execution
of all 288 fused cases and their original reference programs.
Standard GPU7 results available so far are:

| Context | Median decode tok/s | Aggregate decode tok/s |
|---|---:|---:|
| 2K | 62.1572 | 62.2407 |
| 8K | 58.1495 | 58.1513 |
| 32K | 46.5430 | 46.5536 |

All warmup/measured 256-token sequences match the frozen
original and all major faults are zero. Do not compare these
separate-load rows as a controlled percentage gain. The 32K result
does not beat the retained 46.6720 GPU3 observation; the short-context
improvement should not be generalized to long contexts.

The next isolated default-off prototype fuses 97 K320/N10240
up-projections into their sigmoid/pre-mix reduction. It preserves
FP16 block dots and the ordered 32-partial reduction, rounds the
gate in FP16, and retains the original ordered FP32 four-branch
mix followed by FP16 output. Unsupported/multi-token paths use
the original matrix and pre-mix programs with an ordinary
temporary, outside capture. Nine changing-input fixtures passed
18,677,760 finite bitwise comparisons across 128 rounds, pending
limits 1/64, three reduction scales and multi-token fallbacks.
Actual-model dispatch and performance checks are separate gates;
native success alone does not establish a model speedup.

The up/pre-mix actual-model profile subsequently proved 1257
kernels per profiled token, including all 97 new programs, with
exact original prefixes and zero faults. GPU3 complementary
windows measured 59.6569 tok/s off and 61.6925 on, geometric matched
ratio 1.0349144 across 16 pairs. All four requests favored the
new path; full 288-token sequences matched each other and original
256-token prefixes matched the frozen control. Major faults were
zero. This is the fused/prepared path versus the same new node's
original-program fallback, not a 3.49% causal improvement over
the separate-load 62.1572 standard result.

Revision v1 subsequently failed the strict real-weight gate on
`projection_31_stress2`, output element 1117. An exact-bit probe
confirmed negative zero versus positive zero (`0x8000` versus
`0x0000`), not a decimal-printing ambiguity or a subnormal.
Its standard runs were blocked, and its window result is not
a numerically qualified candidate result.

Revision v2 replaces the compile-time four-branch mix loop bound
with the original shader's runtime-uniform bound, initialized
to four. This preserves the original zero-seeded reduction
behavior without canonicalizing zeros or relaxing comparisons.
It passed all nine changing-input fixtures (18,677,760 finite
bitwise comparisons), then all 97 actual projection weight sets
at three activation ranges, three repeats (2,234,880 finite
bitwise comparisons). The failing v1 case is included. Original
graph optimizations and CPU fallback were disabled for the
real-weight comparisons; capture counters verified actual fused
and reference execution. Provider SHA-256:
`69522d588470eba8eb84043fcf91c523c99c025a2e2216c67ac4f6150fc50f00`.
Model profile/window and standard 2K/8K/32K tests must be rerun
for this changed binary; the v1 performance evidence does not
qualify v2. Whole-model/logit quality and the 100 tok/s target
remain unmet.

The next default-off isolated overlay, `ORT_WEBGPU_QSA_PARALLEL_MAP=1`,
targets the serialized visibility-map stage. The completed up/pre-mix
v1 profile showed 1,926 microseconds summed over 12 map dispatches
at 32K versus 256 at 2K. These are profiled GPU durations from an
unqualified v1 binary, useful for triage, not headline token rates.
Eligibility retains the existing NVIDIA/dense-map/global-cache,
single-row gates and further requires a one-token sequence and
8,192--65,536 total tokens. Other cases retain the original map.

One workgroup per 512-token chunk first counts visible tokens;
a second dispatch computes exclusive chunk prefixes and stably
scatters visible token IDs. Dense and empty chunks take direct
paths; hole masks retain exact ordered compaction. Only integer
mapping is changed: score computation, ranking, attention and
their floating-point arithmetic are untouched. The extra count
dispatch adds 12 kernels per eligible model token; dispatch count
alone is not the performance objective.

The original qualified QSA comparator/runtime passed 69 fixtures
with outer map0/map1 and inner rank0/rank1: 2,484 exact CPU-reference
executions, full validation and no CPU fallback. Profiles proved
both new programs on all seven eligible fixtures, including
8K/32K model-shaped hole masks, irregular budgets, FP16/FP32 ties,
33,025 and 65,536 total tokens. Unchanged fallback cases also ran.
The first attempt using an alternate SDK-linked comparator failed
on an inactive 2K FP16 NaN edge fixture; that evidence is retained.
An isolated parent-provider probe passed it, and the original
previously-qualified harness passed the complete matrix without
changing the provider or comparison tolerances. The alternate
harness discrepancy is not claimed resolved.

Model dispatch, complementary windows and standard full-context
checks remain required before promoting this parallel-map overlay.

The up/pre-mix v2 actual-model profile passed 1,257 kernels/token.
GPU7 complementary windows measured 59.7543 tok/s off and 62.2337
on, geometric paired ratio 1.0447495, with all four requests
favoring on, exact frozen prefixes and zero faults. This still
compares the new node against its own fallback, not the best
historical standard row. Completed v2 standard results are:

| Context | Median decode tok/s | Aggregate decode tok/s |
|---|---:|---:|
| 2K | 62.1290 | 61.9906 |
| 8K | 59.6357 | 59.7877 |
| 32K | 31.0363 (performance rejected) | 33.2141 |

All complete original sequences and zero-fault checks pass.
The 2K row does not improve the prior 62.1572 observation.
The 32K run passed original sequences and zero-major-fault checks,
but its performance is not promoted: warmup/measured request rates
were 26.0932/27.0230/31.0363/47.3995 tok/s. Early requests had
0.89--0.98 system CPU seconds and 6,426--6,812 voluntary switches,
versus 0.2752 seconds and 5,445 switches in the final request.
The prior down-fusion 32K observation remains 46.5430 tok/s.
The recovery suggests investigating environmental interference;
it does not establish a cause or justify discarding slow samples.
Separately, the parallel
map actual-model profiles passed the expected 1,257 kernels at
2K (unchanged fallback) and 1,269 at 8K, including exactly 12
count and 12 scatter programs with no old map programs at 8K.
Its model windows and full-context standards remain separate gates.

A third independent, default-off overlay,
`ORT_WEBGPU_GENERIC_MATMUL_REPLAY=1`, extends the existing private
per-node binding replay to nine original INT4 GEMV shapes. It
changes no shader arithmetic or class layout. It retains NVIDIA,
FP16 input/scale, symmetric block32, no-bias/no-zero-point and
FP32-accumulation-disabled gates. The skinny and hyper-skinny
selectors must be off; the shape set excludes the switchable
exact-mid and HC-exact shapes, avoiding stale command reuse when
those math selectors change. Unsupported/multi-token paths retain
the original route and existing capture validation.

Twenty-seven changing-input fixtures (nine shapes, rows 1/2/16)
passed 116,582,784 finite bitwise comparisons over 128 rounds,
pending limits 1/64 and explicit invalid-flag rejection. Counters
proved actual bound-command hits on all nine single-row cases.
The baseline provider relink was byte-identical to up/pre-mix v2;
the new provider SHA-256 is
`f892029dffcd37d01f206bf46c11a078b6290cc1f6311e5dbddb0fa238bd5ebb`.
This candidate is independent of the parallel-map overlay, not
silently stacked with it. Model capture/window checks are required
before deciding whether its command-reuse scope is worthwhile.

The generic-replay actual-model profile passed all 735 intended
nodes (363 under the inherited `MatMulNBitsExact` statistics label),
both rebinding and bound-command hits, no 2K binding evictions,
unchanged 1,257 kernels and exact frozen prefixes. GPU0 complementary
2K windows measured 60.1239 tok/s off and 60.6780 on, geometric
paired ratio 1.0083631. All four requests favored on, all complete
288-token sequences matched and faults were zero. This is a
marginal screen, not progress toward 100 tok/s by itself.
Its standard results were 62.0136 median / 61.9467 aggregate at 2K
and 59.1515 / 59.3238 at 8K. No cross-GPU causal percentage
comparison is implied.

Parallel-map GPU3 standard results completed so far are 60.9694
median / 61.0119 aggregate at 2K and 60.2645 / 60.5119 at 8K.
The map is disabled by its shape gate throughout the 2K case.
At 8K, complementary windows recorded 59.7758 off and 60.3857 on,
geometric paired ratio 1.0479783, 16 pairs, with all four requests
favoring on. The geometric paired ratio is not the ratio of the
two reported medians. Original full sequences and fault checks
pass. Both independent GPU0/GPU3 32K runs were intentionally stopped
before completion to serialize performance testing on GPU7. Their
partial evidence is retained and is not full-context qualification.

Recorded resource counters for one up/pre-mix v2 GPU7 2K measured
request show 2.3901 CPU seconds during 4.0687 decode wall seconds.
This does not by itself distinguish useful CPU work from driver
polling. A separate GPU4 diagnostic samples only its exact owned
model process during decode; its instrumented window rates must
not be promoted as performance evidence. The first attempt failed
before container creation because the container launcher had not
yet enabled GPU4; that startup evidence is retained.

The corrected diagnostic completed successfully. Its exact owned
PID was verified against GPU4 and sampled for two seconds during
the same decode request (639 samples, 13 lost). Inclusive CPU
samples included Flush 34.59%, AppendNodeCommands 26.29%, deferred
encoding 11.11%, dispatch 10.64%, malloc 10.33%, environment parsing
about 5%, and bind-group creation 4.23%. These overlap and must not
be added. CPU greedy selection was 3.13% self. Instrumented rates
are excluded from performance claims.

The next isolated screen retests submission batching after expanded
immutable command reuse: generic replay, GPU7, pending limits
64/128/256 followed by a repeated 64 control. Promising settings
still require 8K/32K original-sequence and fault checks. Non-64
settings are explicitly separate from the retained pending64 table.
Other users' GPU1/GPU2 jobs are untouched; serializing owned jobs
does not eliminate all shared-host interference.

The serialized GPU7 batching screen passed frozen original sequences
and zero-major-fault checks at every setting. Median 2K tok/s were
61.8298 (pending64 first), 61.8640 (128), 59.0663 (256), and 62.1283
(64 repeated). Larger batches are not promoted; pending64 is retained.

A separate default-off direct-activation INT4 overlay,
`ORT_WEBGPU_MATMUL_NBITS_DIRECT_A=1`, removes cooperative activation
staging and its per-K-tile workgroup barriers from the generic
eight-column kernel. Each lane instead loads the identical FP16
activation values directly. Packed weight decoding, FP16 dot-product
and block accumulation order, zero-initialized shared partials,
serial 32-lane final sum and optional bias addition are retained.
Only the final partial-sum barrier remains. This generalizes the
direct-load approach already used by the exact dense kernels.

The selector retains earlier specialized routes and is restricted
to NVIDIA, one batch, one-row or explicitly overridden ten-row
expert dispatch, symmetric block32 INT4, vector4 operands, FP16
activations/scales and FP16 accumulation. Other cases fall back.
Both leaf binding-replay and QMoE command-replay metadata include
the new selector, preventing stale command reuse across toggles.
No core/base class layout changes are introduced.

The isolated provider passed 27 generic changing-input fixtures
and 18 routed/bias/fallback QMoE fixtures: 125,921,664 finite
bitwise comparisons against the original route, pending1/64,
explicit invalid-flag rejection and changing routes. Cache counters
proved both leaf variants captured and reused within each session.
Baseline relink was byte-identical to generic replay. Provider
SHA-256:
`e2b884752295cdc035a528d26b7136fe5be3244d83f2e9cd76b4810f5f712a1c`.
Actual-model dispatch, controlled windows and full-context standards
remain required; this is not yet a TPS gain or quality qualification.

The v1 model profile passed prefixes/fault checks but failed its
requested dispatch-coverage assertion: 255 direct programs, not
303. The missing 48 were FC2 expert matmuls with ten input rows
and indirect indices but no explicit row override. That failed
coverage evidence is retained. V2 admits ten-row indirect dispatch
without requiring the override, rebuilt the same isolated scope
and repeated all 125,921,664 finite bitwise comparisons successfully.
Its model profile proved all 303 direct programs, one retained
INT8 lm_head program and unchanged 1,257 total kernels/token.

GPU7 complementary windows then rejected the direct-load candidate
for performance: 61.2472 tok/s off versus 60.4255 on, geometric
paired ratio 0.9839184; all four requests favored off. All complete
288-token sequences matched and faults were zero. It stays off;
fewer workgroup barriers did not produce an end-to-end gain.
The next full-context timing run is the independent parallel-map
candidate on GPU7, serialized across 2K/8K/32K, pending64.

The direct-load v2 provider SHA-256 is
`1aa3fedcc4951fa07603cb4eec4d04a9f30b0fa2786f70b69a5160915f101a5c`.
The serialized parallel-map GPU7 run has completed 2K at 61.7428
median / 61.7093 aggregate and 8K at 62.2409 / 61.3095. Full
original sequences and zero-major-fault checks pass. These are
separate-run observations, not causal percentage comparisons.
The completed 32K run measured 52.5585 median / 52.1572 aggregate
tok/s. Measured requests were 52.5585, 52.6891 and 51.2485 tok/s;
the excluded warmup was 49.2760. All four original full sequences
matched and all prefill/decode major fault counters were zero.
This serialized attention-map table is 61.7428/62.2409/52.5585
tok/s at 2K/8K/32K, versus the earlier retained projection/SiLU
table of 62.1572/58.1495/46.5430. These separate runs do not
isolate a causal speedup. The 100 tok/s target remains unmet.

An independent QMoE FC1/SwiGLU epilogue overlay has built against
the generic-replay parent, with a byte-identical baseline relink.
Its default-off selector is `ORT_WEBGPU_QMOE_FC1_ACTIVATION=1`.
It retains the original eight-column/128-thread projection mapping,
cooperative activation staging, FP16 dot/block/serial reduction order
and bias addition. Each final projection is materialized in FP16
shared memory before the unchanged FP32 interleaved SwiGLU
expression, replacing the FC1 and activation pair with one program.
This is intended to remove 48 dispatches per model token, not to
change accumulation precision.

Eligibility is restricted to the original one-token, hidden2560,
top10, symmetric block32 INT4 NVIDIA path, FP16 input/scales,
FP16 accumulation, interleaved SwiGLU (including automatic mode),
no FC3/FC1 zero points/router weights, and cooperative/skinny/
hyper-skinny selectors off. Both selector values enter the QMoE prototype key.
The five-buffer workspace layout is retained, including the unused
raw FC1 scratch on the fused path, to preserve existing workspace
and binding lifetimes. Capture requires four commands when eligible
and enabled, five otherwise.

Unqualified v1/v2 builds and failed evidence are preserved. The
v2 native attempt exposed reversed scale/bias input indices in
the eligibility predicate and no-bias fixture. V3 uses QMoE input3
for scales and input4 for bias, matching the original operator.
It additionally restricts fusion to canonical UINT8 packed weights
and matching three-dimensional block scales with 80 blocks per
projection row. Row-wise scales retain the general QMoE path.
Root-owned external model data is hardlinked using the existing
preparation scripts' checked `sudo ln` fallback, never copied.

The v3 provider SHA-256 is
`30ee82eca3e8fd456ee781ac579e38e8637d0524f4c6654c43f4f8c932bbe632`.
Native validation passed 9,584,640 finite bitwise comparisons over
21 fixtures: 18 original real-weight cases plus no-bias,
automatic-interleaving and concatenated-layout fallback cases.
Both selector values, changing inputs/routes, pending1/64 and
four/five-command capture contracts pass. Actual model profiles
confirm 48 fused programs, no separate MoEActivation, 256 remaining
MatMulNBits programs and 1,209 kernels per token.

Controlled complementary GPU7 2K windows measured 60.5669 tok/s
off and 61.6357 on, matched geometric ratio 1.0132109. All four
requests favored fusion; complete 288-token sequences matched and
fault counters were zero. This is a window comparison, not the
standard 256-output median. The serialized standard 2K/8K/32K
table completed at 62.0042/62.0785/47.8201 median tok/s
(aggregate 62.5965/62.0755/48.1038). All original full sequences
and zero-major-fault checks pass. The measured 32K requests were
47.1631/47.8201/49.3817, with excluded warmup 46.6278.
This does not beat the independent attention-map run at 32K.

### Immediate encoding of ready replay commands

A further default-off runtime overlay is built on FC1/SwiGLU v3,
with a byte-identical parent relink and no C++ object-layout changes.
`ORT_WEBGPU_NODE_REPLAY_IMMEDIATE=1` first encodes earlier deferred
work, then records ready cached commands directly into the current
compute pass, preserving dispatch order and pending-dispatch
submission limits. This avoids copying ready commands and their
program-key strings into the deferred vector. Uniform uploads,
buffer retirement and queue submission still use the existing
Flush path. Profiling retains the original deferred route.
Every submission still uses a fresh WebGPU command buffer; this
does not reuse a single-use command buffer or change shader math.

The built provider SHA-256 is
`9baa6b40e12e9b9c5a7d0154381327116ee56fa0e0ab5693e9e503ea75ca38c8`.
Native tests passed 255,578,240 finite bitwise comparisons over
78 QMoE, generic MatMul and static replay fixtures, including
changing routes, CPU split values, binding reuse, LRU eviction and
explicit invalid-flag rejection. The initial result validator
incorrectly omitted the rejected trial from request counters;
the completed immutable guard was revalidated with exact command,
provider and fixture provenance, counting one rejected request
per eligible kernel separately from successful captures/hits.
No numerical gate was relaxed.

Actual model profiles retain 1,209 kernels. Controlled 2K windows
were effectively flat: off 63.2379, on 63.2237 tok/s, matched
geometric ratio 1.0012243. Only one of four request medians favored
on. Full tokens and zero faults passed, but this overlay is not
retained for performance and is not escalated to a full-context
table.

### FC2 projection and final expert mix fusion

The next independent overlay is built on FC1/SwiGLU v3, not on the
rejected immediate-encoding route or the attention-map provider.
`ORT_WEBGPU_QMOE_FC2_FINAL_MIX=1` assigns ten 32-lane expert
groups to each 320-thread workgroup and computes two output
columns. It preserves the original FP16 block-dot and serial32
projection reduction, optional bias and materialized FP16 FC2
values. Two threads then perform the original rank-ordered
**FP16** weighted mix. The original final-mix accumulator is FP16,
not FP32; this experiment does not change it.

Eligibility additionally checks canonical INT4 FC2/scales shapes,
hidden2560/intermediate640/top10, absent FC2 zero points and
adapter limits for 320 threads and 14,080 bytes of shared storage.
The existing five scratch tensors remain allocated for stable
binding lifetimes. Both fusion selectors participate in the QMoE
cache key; capture checks account for three, four or five commands.

V1 failed bitwise parity immediately. A diagnostic using one-hot
routing passed while ordinary routing failed. V2 restored the
original uniform-bounded FP16 final-mix loop instead of the
constant-bound rewrite and passed 9,666,560 finite bitwise
comparisons over 22 fixtures, including separate no-FC1-bias and
no-FC2-bias cases, automatic interleaving, concatenated fallback,
changing routes and pending1/64. This establishes the successful
rewrite, not a proven compiler-level cause for the V1 mismatch.
Failed V1 evidence is retained.

V2 provider SHA-256:
`8e674e8a3bcd6833a7e95343b88ae6171db8130dbcdcb2a16a94f4bc03f98d13`.
Actual model profiles confirm 48 FC1 fusions, 48 FC2/mix fusions,
208 remaining MatMulNBits programs, no standalone final mix, and
1,161 kernels per token. Controlled 2K windows measured 63.1362
off and 63.4695 on, matched geometric ratio 1.0035657. All four
request medians favored fusion; full sequences and zero faults
passed. The gain is small. The completed standard GPU7 table is:

| Context | Median decode tok/s | Aggregate decode tok/s |
|---|---:|---:|
| 2K | 64.2501 | 63.4123 |
| 8K | 60.0257 | 59.6614 |
| 32K | 48.7320 | 48.6933 |

All original full sequences and zero-major-fault checks pass.
These separate-run observations do not establish causal speedups.
The independent attention-map candidate remains faster at 8K/32K
(62.2409/52.5585); its rows must not be presented as one configuration
with the expert-fusion 2K result.

### Combining the expert fusions and attention map

The isolated combined build-v2 replaces only the QSA object in
the FC2-v2 link. Both component baseline relinks are byte-identical.
The original library basename must be preserved even for scratch
relinks because ELF's BASE version definition embeds that name;
the failed differently named scratch relink is retained.

Combined provider SHA-256:
`611b020b3a5277a7b3dc3c82b6ef3e5736836899d0d9452ca0cea36f9b46a783`.
Native qualification passes 9,666,560 finite bitwise comparisons
across 22 QMoE fixtures and 2,484 reference executions across
69 QSA fixtures, with 138 profile files. Actual-model profile,
controlled map windows and a complete standard context table
remain required before claiming combined performance. Actual-model
8K profiling subsequently confirmed 48 FC1 and 48 FC2/mix programs,
12 parallel-map count/scatter pairs, no original map kernel and
1,173 total kernels per token. Controlled 8K map windows, keeping
both expert fusions fixed on, measured 62.7398 off / 63.8186 on
with matched geometric ratio 1.0165923. All four request medians
favor the map; original sequences and zero faults pass. The
combined standard table completed serially on GPU7:

| Context | Median decode tok/s | Aggregate decode tok/s |
|---|---:|---:|
| 2K | 64.9301 | 63.9641 |
| 8K | 61.2015 | 62.1664 |
| 32K | 52.9948 | 53.3133 |

Every original sequence and major-fault gate passes. The 32K
median is slightly above the separate map-only 52.5585 run;
that difference is not a controlled causal speedup. The combined
8K median does not beat the separate map-only 62.2409 run.

### FC2 shared partial layout experiment

The independent padding build-v1 descends from FC2-v2, not the
combined map provider. `ORT_WEBGPU_QMOE_FC2_PADDED_PARTIALS=1`
changes each 32-element partial row to a 33-element shared row.
The extra element is unused. All arithmetic, reduction bounds,
FP16 materialization and uniform-bounded expert mix are unchanged.
This tests whether a different shared-memory bank access pattern
helps; a bank-conflict bottleneck has not been measured.

Padded programs require 14,120 shared bytes, otherwise retaining
the original layout. The padding selector is validated and included
in replay and program keys. Native comparison with FC1/FC2 fusions
fixed on passes 9,666,560 finite bitwise comparisons over the same
22 immutable fixtures. Actual-model profiling confirms all 48
padded programs and 1,161 kernels per token. Controlled 2K windows
measured 61.6798 off / 61.7438 on, matched ratio 1.0011452, with
three of four request medians favoring off. This is effectively
flat: leave padding off and do not escalate to a full table.

### Shared-activation HC projection experiment

The tiled projection build-v2 descends from the combined provider.
It stages the 10,240 FP16 activations once per workgroup and
computes eight instead of two columns with 256 instead of 64
threads. Block dot order, ten-tile FP16 accumulation, serial
32-lane reduction and materialized SiLU rounding are unchanged.
The final four out-of-range columns are predicated without
skipping workgroup barriers. Each block's eight activation vectors
occupy a nine-vector shared stride to distribute strided accesses
across banks; actual bank-conflict counters have not been measured.
The path requires 23,552 shared
bytes and 256 threads; unsupported devices retain the old shader.

Old and tiled variants use separate static replay caches so
within-process selector changes cannot replay the wrong shader.
Only the private HC kernel gains a cache member; no runtime
context/header object layout changes. The unpadded-stride build-v1 is retained as CPU-only evidence.
Build-v2's CPU build and exact parent relink pass. Native
qualification passes 2,363,904 finite bitwise comparisons over
nine immutable random-weight fixtures, 128 changing-input/split
rounds each, both pending windows, explicit invalid-flag rejection,
LRU eviction and three old plus three tiled replay-cache rows.
Actual-model profiling verifies 96 tiled projections and 1,161
kernels per token. Controlled 2K windows measured 62.3346 off /
62.5506 on, matched ratio 1.0013555, with split request medians.
This is effectively flat: leave tiling off, with no full-table
escalation. All tokens and faults pass. The first validator
incorrectly expected only tiled-cache statistics even though
windows intentionally exercise both independent caches. Corrected
validation requires both 96-row families. Explicit completed-run
resumption checks exact command/GPU/provider/model provenance,
preserves guard/log/results, and records missing historical host
snapshots as unavailable rather than fabricating them.
Whole-model quality qualification and the 100 tok/s target remain
unmet.

### Subgroup gate retest on the combined stack

The existing default-off subgroup gate, previously flat on the
older 46.7 tok/s stack, was retested without changing the provider
binary. A standalone comparator fixes FC1 and FC2 fusions on while
comparing original and subgroup gates: 9,666,560 finite bitwise
comparisons across the same 22 immutable fixtures pass. Actual
model profiling verifies 48 gate programs whose cache hints select
subgroups, with 1,161 kernels per token.

Controlled 2K windows measured 61.4076 off / 62.8370 on, matched
geometric ratio 1.0245965. All four request medians favor on;
original sequences and zero faults pass. This does not establish
why the outcome differs from the earlier stack. A new complete
2K/8K/32K standard table completed serially on GPU7:

| Context | Median decode tok/s | Aggregate decode tok/s |
|---|---:|---:|
| 2K | 66.3362 | 66.3912 |
| 8K | 65.1457 | 63.7651 |
| 32K | 55.3042 | 54.8573 |

Every original sequence and major-fault gate passes. These are
the best completed medians observed so far, not controlled causal
percentages relative to preceding separate tables.

### Padded QMoE activation tiles

The independent activation-padding build-v1 descends from the
combined provider and keeps projection arithmetic and workgroup
geometry unchanged. `ORT_WEBGPU_QMOE_PADDED_ACTIVATIONS=1`
maps each block's eight activation vectors into a nine-vector
shared stride in both FC1 and FC2. This is separate from the
rejected FC2 partial-row padding experiment.

FC1 uses 288 instead of 256 shared vectors; FC2 uses 1,800 instead
of 1,600. The path requires 15,680 shared bytes and retains the
old layout on smaller devices. The selector is validated and
included in the QMoE replay key and both program keys. Zero-fill
and staged/read cell-coverage CPU tests pass; the array layout
is intended to distribute shared-memory bank addresses, not a
measured hardware-counter claim.

Provider SHA-256:
`7cfa100b170fa4f24e58bfcbc159bb1e588f3df7d93a697486b402e59f1424e5`.
The CPU build and exact parent relink pass. Native qualification
passes 9,666,560 finite bitwise comparisons over 22 immutable
fixtures, with both expert fusions fixed on and gate modes varied.
Actual-model profiling verifies 48 padded FC1 and 48 padded FC2
programs, 1,161 kernels/token and the subgroup gate enabled.
Controlled 2K windows measured 65.0123 tok/s off versus 65.1734
on, with a matched geometric ratio of 1.0009846 and request
medians split two/two. This is effectively flat: activation
padding remains off and was not escalated to full-context timing.

### Post-normalization and projection graph fusion

The independent post-normalization/projection build-v2 descends
from the combined provider. Its graph rewrite merges 94 chains:
48 regular and 46 shared post-normalizations followed by the
canonical K10240/N324 projection. All four outputs remain
available, including normalized streams consumed by the
following up/pre-mix. Initializer serialization is unchanged.

`ORT_WEBGPU_HC_POST_NORM_PROJECTION=1` enables the single-token
NVIDIA FP16 path when exact-dense projection is enabled and
device limits permit 256 threads, 27,648 shared bytes and
10/12 storage bindings for regular/shared variants. Each
workgroup recomputes post-normalization locally; only group
zero writes the two retained normalization outputs. FP16
materialization, FP32 RMS reduction order and the original
FP16 projection reduction/SiLU rounding are preserved.

Provider SHA-256:
`e2f816b44f938d86b816e441b36f025d45f7a219ba5c82afe5f03453eb25530f`.
The CPU build and exact parent relink pass. Native qualification
passes 607,143,936 finite bitwise comparisons over 36 immutable
random-weight fixtures, comparing all four outputs. Coverage
includes regular/shared branches, two scale layouts, rows1/2/16,
three alpha values, changing GPU inputs/CPU split sizes,
pending1/64, selector fallback, invalid-selector rejection and
replay-cache LRU eviction. The 77 focused CPU tests pass.

Actual 2K profiling verifies 94 fused programs, one remaining
post-normalization, two remaining projection programs and
1,067 kernels/token, down from 1,161. The completed standard
table preserves every frozen token and zero major faults:

| Context | Median decode tok/s | Aggregate decode tok/s |
|---|---:|---:|
| 2K | 67.2606 | 67.2279 |
| 8K | 65.7851 | 65.7395 |
| 32K | 55.9173 | 55.8256 |

These are separate-run observations, not causal percentages
against the preceding subgroup-gate table. Whole-model quality
remains unqualified and the 100 tok/s target remains unmet.

Only the single fused dispatch is cached. The original
normalization/projection fallback uses multiple commands and
is deliberately uncached; within-process off/on windows
against that fallback are rejected as an unfair baseline.

The subsequent four-column build-v3 changes only projection
work distribution: 81 groups with four active column warps,
versus 41 groups with eight. All 256 threads still participate
in normalization and barriers. Inactive matrix lanes are
predicated, and eight/four-column paths have separate replay
caches and program keys. Runtime/common object layouts are
unchanged. Provider SHA-256:
`1dee92be2abface69414e3828ee779d3c68d940904507a970bd45609537c6b15`.

Native qualification again passes 607,143,936 finite bitwise
comparisons over the same 36 random-weight fixtures, including
explicit rejection of an invalid column selector. Actual-model
profiling verifies 94 four-column programs and 1,067 kernels.
Fair within-process windows hold the graph fusion enabled and
toggle only `ORT_WEBGPU_HC_POST_NORM_FOUR_COLUMNS`. Both paths
are warmed and both cache families participate. Controlled 2K
windows measure 64.3362 tok/s for eight columns and 64.5700 for
four, with matched geometric ratio 1.0025056. Although all four
request medians favor four columns, the effect is effectively
flat; it remains off and was not escalated to full-context timing.

### Hierarchical subgroup routing

The independent hierarchical-gate build-v2 descends from the qualified
eight-column post-normalization/projection provider. It selects each
subgroup's local top ten, then uses the first subgroup to merge those
sorted lists. This removes workgroup-wide synchronization from each of
the ten global selections. Ordered keys, expert-ID tie-breaking,
rank-ordered FP32 softmax and FP16 routing-weight storage are preserved.

`ORT_WEBGPU_QMOE_HIERARCHICAL_GATE=1` requires the existing subgroup-gate
eligibility checks and an adapter minimum subgroup size of at least 32.
Unsupported cases retain the original gate. The selector is validated
and included in the QMoE replay key; existing object layouts are unchanged.
Provider SHA-256:
`75a87ad85fd51b12be39f1474d0408282c63ffc5886562d005687deb4219ed96`.

The exact parent relink and CPU build pass. GPU7 native qualification
passes 9,748,480 finite bitwise comparisons over 23 fixtures, including
the original changing-input/replay/fallback cases. A synthetic sparse
identity fixture additionally exposes selected expert membership and
exact routing weights in output channels. Its six routing patterns
cover ties, signed zeros, negative-infinity non-winners, large logits,
and winners concentrated within or spread across subgroups. The fixture
uses 5,521,408 allocated bytes for 1,416,888,320 logical external-data
bytes; it is not a copy of real model weights.

Actual-model 2K profiling verifies 48 `QmoeHierarchicalGate1Token`
dispatches and 1,067 total kernels/token. Counterbalanced 2K windows
hold the original subgroup gate and every graph/expert fusion enabled,
toggle only the hierarchical selector, and warm both variants.
Previous/hierarchical medians are 65.3346/67.0860 tok/s, with matched
geometric ratio 1.0267791 across 16 window pairs. All four request
medians favor the candidate and output tokens match.

The completed standard table preserves all frozen tokens and zero
major faults:

| Context | Median decode tok/s | Aggregate decode tok/s |
|---|---:|---:|
| 2K | 68.9082 | 68.8352 |
| 8K | 67.3642 | 67.3767 |
| 32K | 55.7294 | 55.6697 |

These are separate-run observations, not the controlled-window gain.
The 32K result is effectively flat/slightly below the preceding
55.9173 median; it is not evidence of a long-context improvement.
The combined focused CPU suite, including the subsequent projection
fusion's graph tests, passes 86 tests. Whole-model quality remains
unqualified; the 100 tok/s target remains unmet.

### MLP input-projection and activation fusion

The independent MLP input-projection build-v2 descends from the
hierarchical-gate provider. Each of 48 rewritten chains combines four
symmetric block32 INT4 projections sharing K2560 inputs: the N512 router,
N640 shared gate/up projections and N1 shared-expert gate. It also
absorbs the following `QuickGeluMulF16`. Three outputs retain the
router logits, activated shared-expert input and shared-expert gate.
The rewrite rejects observable gate/up intermediates, ambiguous
projections, mismatching types/shapes/attributes and unavailable inputs.
Initializer serialization and external-data hardlinks are unchanged.

`ORT_WEBGPU_MLP_INPUT_PROJECTIONS=1` enables the single-token NVIDIA
path with exact-dense-mid enabled, FP16 accumulation and device limits
supporting 256 threads and 12 storage buffers. One 145-group dispatch
uses the original FP16 block/tile accumulation and serial 32-lane
reduction. Shared-expert activation preserves the original FP16 stable
sigmoid, FP16 QuickGelu storage boundary and final FP16 multiplication;
it does not substitute the QMoE kernel's FP32 SwiGLU expression.
Other cases execute the original four projections and activation.
That compound fallback is deliberately uncached and is not accepted
as a within-process performance-window baseline.

Provider SHA-256:
`e2518e015c5d84a52be523926641d68d58c3ce8742f78c5c8abdcbfef3a89abf`.
Graph SHA-256:
`8037bd0a14a3ae5307cd2d626e90d3be26a6a7b8893062246ee23487599400f9`.
The exact parent relink passes. Native GPU7 qualification passes
5,608,192 finite bitwise comparisons over eight random-weight
fixtures, all three outputs, 128 changing-input rounds, rows0/1/2/16,
two scale ranges, pending1/64, exact-selector fallback, explicit invalid
selector rejection and replay binding reuse. The failed build-v1 GPU
proof is retained; build-v2 corrects the WGSL reserved output identifier.

Actual-model 2K profiling verifies 48 fused programs and 875 total
kernels/token, down from 1,067: four dispatches saved per layer.
The full standard table preserves all original frozen tokens and zero
prefill/decode major faults:

| Context | Median decode tok/s | Aggregate decode tok/s |
|---|---:|---:|
| 2K | 72.9906 | 72.3973 |
| 8K | 71.5611 | 71.2471 |
| 32K | 57.6763 | 58.2649 |

These are separate-run observations, not causal speedup percentages.
All previous best-stack optimizations remain enabled, while rejected
four-column, activation-padding and direct-A variants remain off.
Whole-model quality remains unqualified and 100 tok/s remains unmet.

### CUDA-inspired INT4 bit-pattern conversion experiments

These isolated experiments adapt the FP16 mantissa-insertion technique
used by CUDA's `Convert8xInt4To8xHalfs`, not its assembly or accumulation
order. For each half lane, inserting a four-bit code into the mantissa
of 1024 and subtracting 1032 gives the exact symmetric value `code - 8`.
WGSL uses `bitcast<vec2<f16>>`, masks and shifts instead of integer-to-half
conversions. Original scale multiplication, dot-product grouping, FP16
materialization and accumulation remain unchanged. CPU tests compare
1,572,864 half results bitwise across all two-byte patterns in both
word halves plus randomized words.

The MLP-only build-v1 uses `ORT_WEBGPU_MLP_BITCAST_UNPACK`, separate shader
names and independent replay caches while keeping graph fusion enabled
on both sides. Native qualification passes the same 5,608,192 finite
bitwise comparisons; actual-model profiling confirms 48 bitcast programs
and 875 total kernels. Fair 2K windows measure 73.3882/73.8904 tok/s,
with matched geometric ratio 1.0054097. This small result is not promoted
or escalated to a full standard table; the flag remains off.

The separate expert build-v2 instead applies the shared decoder to both
fused FC1/SwiGLU and FC2/final-mix kernels. It preserves the MLP projection
parent unchanged and uses `ORT_WEBGPU_EXPERT_BITCAST_UNPACK`, distinct
shader families and an additional QMoE prototype-replay key component.
The selector is validated both before replay and in workspace execution.
The QMoE runtime/header layout is unchanged.

Expert provider SHA-256:
`fc5335db2dc1b4924473126d69a5e5a3a478f79d0cea467fc3ee3e9ff58d5f20`.
The CPU build and exact parent relink pass. Native qualification passes
9,748,480 finite bitwise comparisons over 23 changing-input/routing
fixtures, including optional biases, unsupported-layout fallback,
pending1/64, replay variants and direct identity-based routing checks.
Actual-model profiling verifies 48 bitcast FC1 and 48 bitcast FC2 programs
with the same 875 total dispatches.

Expert-only 2K windows keep both expert fusions and every parent
optimization enabled. Original/bitcast medians are 71.5120/72.7109 tok/s,
with matched geometric ratio 1.0170242 across 16 window pairs. All four
request medians favor the candidate and tokens match. The completed
standard table preserves all frozen original sequences and zero
prefill/decode major faults:

| Context | Median decode tok/s | Aggregate decode tok/s |
|---|---:|---:|
| 2K | 74.2136 | 74.1332 |
| 8K | 72.8242 | 72.0678 |
| 32K | 60.4944 | 60.4834 |

These are separate-run observations. In particular, the larger 32K
change relative to the preceding standard table is not a causal
percentage. Controlled 32K windows measure 58.9122/60.4493 tok/s
with expert unpacking off/on, with matched geometric ratio 1.0184843
across 16 window pairs. All four request medians favor the candidate,
all 288 output tokens match across requests, the original frozen
prefixes match, and all prefill/decode major faults are zero.
The focused CPU suite passes 97 tests. Whole-model quality remains
unqualified and the 100 tok/s target remains unmet.

### Standalone dense-projection unpacking experiment

The separate dense build-v1 descends from expert build-v2 and leaves its
expert and fused-projection shaders unchanged. The default-off
`ORT_WEBGPU_DENSE_BITCAST_UNPACK` selector applies the same exact nibble
conversion to eligible standalone single-token FP16/block32 INT4
projections. Unsupported shapes, precisions, optional inputs and
multi-token requests retain the original path.

The experiment derives a private shader family from the frozen compiled
WGSL template, not the independently modified working-tree template.
Only four unpack expressions change; scale multiplication, accumulation
and output rounding remain unchanged. The original shared
`ApplyMatMulNBits` implementation is preserved byte-for-byte in source,
so callers inside other fused operators cannot change shader families
without updating their replay keys. Standalone replay keys explicitly
include the selected variant; no existing runtime object layout changes.

Native GPU7 qualification passes 116,582,784 finite bitwise comparisons
over 27 changing-input fixtures: nine projection shapes, rows1/2/16,
128 rounds, pending1/64, both selector values, invalid-selector rejection,
and two independently cached single-token variants. This is correctness
evidence, not a whole-model quality claim. Actual-model profiling
verifies 159 `MatMulNBitsBitcastF16` programs, one unchanged INT8
`MatMulNBits` program and 875 total dispatches. Controlled 2K windows
measure 74.2114/75.5346 tok/s with dense unpacking off/on, with matched
geometric ratio 1.0189829 across 16 window pairs. All four request
medians favor the candidate and all tokens match.

The clean standard 2K median/aggregate is 75.0453/75.0052 tok/s;
the clean 8K retry is 70.9459/71.0938 tok/s. The initial 8K attempt
failed because the shared disk filled up and is excluded. The 32K
retry completed but failed the zero-major-fault gate in every request:
prefill faults were 6/7525/8/8 and decode faults were 17/31/46/48.
Its throughput is not qualified or promoted. The dense flag remains
off in the retained complete 74.2136/72.8242/60.4944 tok/s stack;
the isolated positive 2K window result is not sufficient full-context
evidence. Failed runs are preserved, not overwritten.

A subsequent controlled 8K window run also fails the fault gate:
request 0 has four prefill and 23 decode major faults; the other
three requests have zero. All request outputs match, but the
entire timing run is excluded rather than dropping the failing
request. This does not establish a page-fault cause or a
long-context performance regression.

A separate four-column FC2 experiment initially descended from this
dense candidate in CPU-only build-v1. Build-v2 instead rebases on the
clean, fully qualified expert-bitcast parent, leaving dense unpacking
off. It reuses each 12,800-byte activation tile for four
output columns instead of two, halving workgroups from 1,280 to 640
without changing per-column FP16 arithmetic or rank-ordered mixing.
It requires 15,360 bytes of workgroup storage and retains the original
two-column path below that limit. The validated default-off selector
`ORT_WEBGPU_QMOE_FC2_FOUR_COLUMNS` is present in the QMoE replay key;
shader names distinguish both column widths and unpacking variants.
Native GPU7 qualification passes 9,748,480 finite bitwise comparisons
over the same 23 changing-input/routing fixtures, including optional
biases, fallback layouts, pending1/64 and both column widths in replay.
The actual-model raw profile contains 48 four-column programs and
875 total dispatches, but fails the zero-major-fault gate in both
requests (prefill3/3, decode49/35). Its timings are unqualified.
The complementary 2K screen also fails that gate: its first request
records three decode major faults, while the other three are clean.
The unqualified diagnostic matched ratio is 1.0011634, with two request
medians favoring each side. This is not convincing improvement
evidence; the four-column flag stays off and is not escalated to
8K/32K. All compared output tokens match.

### Eight-output hyperconnection pre-mixing experiment

The separate HC-up-tiled build-v1 descends directly from the clean
expert-bitcast parent, not either held candidate above. The default-off
`ORT_WEBGPU_HC_UP_TILED` selector keeps the existing HC-up fusion enabled
on both sides and batches eight output columns per 128-thread workgroup,
reducing workgroups from 2,560 to 320 in each of 97 nodes.

Each column retains its original ten valid block products, ordered
32-lane FP16 sum, FP16 sigmoid and FP32 branch mixing/output cast.
The wider workgroup uses 1,024 FP16 partials and 32 FP16 gates.
Unsupported layouts and multi-token requests retain the original
fallback. Separate shader names and a replay-variant key prevent
cross-variant reuse without changing existing runtime object layouts.

Native GPU7 qualification passes 18,677,760 finite bitwise comparisons
over nine fixtures, 128 changing-input rounds, rows1/2/16,
three reduction scales, pending1/64, both selector values and invalid
selector rejection. Both single-token variants occupy separate replay
entries. Actual-model profiling passes the original-prefix and
zero-major-fault gates and verifies 97 `HCUpPreMixTiledF16` programs
with 875 total dispatches. The combined focused CPU suite passes
87 tests. The clean complementary 2K screen measures 72.90598 tok/s
for the original versus 71.12005 for tiled pre-mixing, with a matched
geometric ratio of 0.976423 (-2.36%). All four request medians favor
the original. Token and zero-major-fault gates pass. This candidate
is rejected, remains off, and is not escalated to 8K/32K.

The distinct, default-off `ORT_WEBGPU_HC_UP_PARALLEL2` candidate instead
maps two columns concurrently onto 128 threads: two columns, four
branches and 16 lanes per branch. It halves workgroups to 1,280 without
the rejected candidate's sequential eight-column loop. Every column
retains ten computed block partials and 22 zero partials, the ordered
32-slot FP16 reduction/sigmoid, and FP32 branch mixing. A private shader
name and replay variant preserve the original shader and fallback.
Four new CPU tests pass, covering complete/disjoint partial writes,
all output/projection columns and preserved arithmetic/fallbacks.
The immutable provider is built on the standard-qualified score16-v5
parent; its unmodified-parent relink is byte-identical. Native GPU7
qualification passes 18,677,760 finite bitwise comparisons over nine
fixtures and 128 changing-input rounds, including rows1/2/16,
pending1/64, replay variants and invalid-selector rejection.
Full-validation 2K profiling verifies 97 new programs and no original
HC-up programs in the measured decode step. Six controlled 2K/8K/32K
comparisons and independent joint qualification pass 6,144 frozen-prefix
IDs, repeated 288-token sequences, zero faults and evidence identities:

| Context | Validation | Original median window tok/s | Parallel2 median window tok/s | Matched ratio |
|---|---|---:|---:|---:|
| 2K | Full | 74.970845 | 75.426239 | 1.007774 |
| 2K | Retained | 73.731515 | 74.892767 | 1.008003 |
| 8K | Full | 75.225875 | 75.776447 | 1.007715 |
| 8K | Retained | 75.770727 | 76.331587 | 1.007181 |
| 32K | Full | 64.288377 | 64.630658 | 1.004994 |
| 32K | Retained | 66.864149 | 67.261683 | 1.006832 |

All four request medians favor parallel2 in all six cases. The matched
gain is about 0.5-0.8%; differences between pooled medians are not the
causal estimate. The focused shared/new CPU suite passes 76 tests.
Five original score16-v5 paired datasets and all six original standard
datasets retain exact verification results after shared-helper reuse.
Six separate standard full/retained-validation runs and independent
joint TTFT/decode qualification also pass all 6,144 original generated
IDs and zero faults:

| Context | Retained median decode tok/s | Full median decode tok/s | Retained mean TTFT s | Retained end-to-end output tok/s |
|---|---:|---:|---:|---:|
| 2K | 75.844065 | 75.010666 | 4.457613 | 32.741172 |
| 8K | 75.885723 | 75.554971 | 19.369472 | 11.241526 |
| 32K | 65.585698 | 64.024963 | 93.814096 | 2.620225 |

Full mean TTFT is 4.476064/19.325200/93.692861 s. These separate-run
observations do not establish causal changes: the lower retained
32K result versus the preceding 66.370386 tok/s is kept, despite the
controlled +0.683% paired gain. Its cause is not established.
This is not an unconditional 32K standard performance upgrade.
The selector remains default-off, below 100 tok/s, and unqualified
for whole-model quality.

### Packed-key subgroup routing experiment

The separate packed-gate build-v2 descends directly from the retained
expert-bitcast build-v2. Default-off `ORT_WEBGPU_QMOE_PACKED_GATE`
requires the existing hierarchical gate eligibility and FP16 router
logits. Other inputs retain the original router.

FP16 values promoted to FP32 have 13 redundant low mantissa bits.
Compressing their ordered keys leaves room to encode the expert ID
in a single integer, preserving minimum-ID tie-breaking and signed-zero
canonicalization. A subgroup bitonic sort supplies ten candidates per
subgroup; a single-key k-way merge selects the global top ten. The
original rank-ordered FP32 softmax and FP16 routing weights remain
unchanged. Separate shader names and prototype keys prevent cross-variant
reuse without changing existing runtime object layouts.

CPU tests exhaust all 65,536 half bit patterns for key ordering and
exercise subgroup sizes 32/64/128, ties, zeros and nonfinite patterns.
Native GPU7 validation passes 23 fixtures and 9,748,480 finite bitwise
comparisons, including direct identity-routing outputs, changing inputs
and routes, pending1/64 and variant replay separation. Expert conversion
and both expert fusions stay enabled on both sides. The focused CPU
suite passes 79 tests. Actual-model profiling passes the original-prefix
and zero-major-fault gates and verifies 48 `QmoePackedGate1Token`
programs with 875 total dispatches.

The clean complementary 2K screen has a matched geometric ratio of
0.997058 (-0.29%). All four per-request medians slightly favor the
candidate, but the paired ratio does not show a gain. This is
insufficient evidence to promote it; the flag remains off and no
8K/32K routing throughput or new standard-context result is claimed.
Whole-model quality remains unqualified.

Packed-gate build-v3 instead uses one packed integer `subgroupMax`
per local rank, eliminating the bitonic shuffle loops and the
original separate expert-ID minimum reductions. The global merge
and arithmetic contract are unchanged. Its native proof again
passes 23 fixtures and 9,748,480 finite bitwise comparisons; all
79 focused CPU tests pass. Clean actual-model profiling verifies
48 packed routers and 875 total dispatches.

The clean complementary 2K screen measures 72.89540 tok/s off and
73.15563 on, with a matched geometric ratio of 1.007492 (+0.75%).
All four request medians favor it. Original tokens and zero-major-fault
gates pass. This is a small promising screen, not a new complete
standard-context stack; long-context testing is still required.

The subsequent 32K paired run completes and matches every output prefix,
but request 0 records 7,183 prefill and 61 decode major faults. The
other three measured requests have zero. The entire timing run is
excluded, and packed routing remains unpromoted.

### Cache-residency preflight and prefill priority

An unprivileged cache probe reported both data files fully resident,
but these files are root-owned and read-only to the benchmark user.
Linux can conceal residency from such callers by reporting every page
resident. A privileged probe instead found only 18,854,830,080 of
51,200,245,760 bytes resident in the CPU PLE embedding table, and
71,158,423,552 of 72,206,712,832 bytes in the INT4 data file.
Controlled read-only warming restored full residency in both.
This identifies a misleading cache diagnostic, not per-operator
attribution for every previously observed process-wide fault.

The benchmark harness now records privileged cache snapshots before
and after execution, warming the exact model files only when needed.
Warming has a 600-second limit, checks available host memory, remains
outside request timing, and does not pin memory. All original-token
and zero-major-fault gates remain unchanged. The affected CPU suite
passes 86 tests. The private Docker runtime was subsequently restored
with isolated containerd namespaces; the system Docker daemon and
foreign workloads were left untouched. Real chunk16/chunk64 screens
and short profiles passed the zero-major-fault gate with the preflight.

Historical clean 32K runs show that prefill has barely improved:

| Stack | Median decode tok/s | Mean TTFT seconds | End-to-end output tok/s |
|---|---:|---:|---:|
| FC2/final-mix | 48.7320 | 389.6842 | 0.64823 |
| Expert bit-pattern conversion | 60.4944 | 387.6879 | 0.65322 |

These are separate-run observations for 256 generated tokens, not
isolated causal percentages. Decode improvements are not a comparable
end-to-end improvement for this long-prompt workload. Both TTFT and
decode must be tracked. The retained protocol uses 16-token prompt
chunks; the new diagnostic screens chunks16/64 with full validation,
the same retained model/provider, privileged cache preflight, frozen
32-token output-prefix checks and zero-fault gates. It is not a new
standard protocol or whole-model quality qualification.

The completed 2K diagnostic (32 generated tokens, one excluded warmup
and one measured request) rejected chunk64:

| Prompt chunk | Measured TTFT seconds | Original 32-token prefixes | Major faults |
|---|---:|---|---:|
| 16 | 20.0680 | Match | 0 |
| 64 | 49.1465 | Mismatch | 0 |

These are diagnostic timings, not replacements for the retained standard
results. Increasing the chunk alone neither preserves outputs nor
improves prefill. The retained INT4 indirect QMoE path is limited to
16 input tokens / 256 routed rows. Above that limit, the generic path
reads expert counts back to the CPU and launches per-expert work.

Bounded 128-input-token profiles confirmed extensive generic QMoE work
with chunk64: the instrumented QMoE host-node duration sum was
8,181,851 us versus 174,045 us for chunk16. These host-node sums include
profiling overhead and are not steady-state percentages. The chunk16
profile instead identified dense subgroup-matrix work as a prefill
target: its current layout pads M16 to M128 and computes a 128-by-128
tile. A private, default-off M16 tile experiment is being investigated.
It must preserve the selected accumulation precision and pass finite
bitwise native comparisons, frozen model-prefix checks and zero-fault
gates before any performance claim. The retained 2K/8K/32K results and
whole-model quality status remain unchanged.

The private M16 build-v3 subsequently passed 52 true-M fixtures with
274,881,024 finite bitwise comparisons over 128 changing-input rounds.
Native profiles verified dispatch of both the FP16-result path and
the FP32-result N324 tail path, plus unchanged M1/M2/M17 fallbacks.
The original subgroup-matrix configuration selection is preserved.
The candidate uses a 16-by-128 workgroup tile instead of 128-by-128,
without changing each output's K-reduction order.

Initial paired full-validation screens (original chunk16, one excluded
warmup plus one measured request, 32 generated tokens) found:

| Context | Build-v3 flag off TTFT seconds | Flag on TTFT seconds | Reduction |
|---|---:|---:|---:|
| 2K | 20.0892 | 19.2600 | 4.13% |
| 8K | 87.2487 | 84.0669 | 3.65% |
| 32K | 389.9686 | 377.1326 | 3.29% |

Both modes matched the frozen original 32-token prefixes and had zero
prefill/decode major faults at all three contexts.
These short-output diagnostic decode rates are not replacements for
the retained 256-output-token standard results. The feature remains
default-off, unpromoted and not whole-model-quality qualified.

The M16-only prepacked-activation variant (build-v4) also passed the
52-fixture native proof with 274,881,024 finite bitwise comparisons.
It is not yet actual-model qualified.

The original-shape INT4 QMoE indirect-prefill extension (build-v1),
capped at 128 tokens / 1280 routed rows, passed six original-layer
dynamic fixtures with 33,454,080 finite bitwise comparisons. M16/32/64/128
match the retained 16-token-shard arithmetic, while M129 preserves
the generic fallback. Native profiles verified the indirect gate
dispatch count; the existing generic large-batch path was not assumed
to be the desired bitwise reference.

A subsequent original-model 2K/chunk64 screen matched all frozen
32-token prefixes with zero faults and measured 7.4032-second TTFT
(one excluded warmup plus one measured request, full validation).
The earlier retained chunk16 diagnostic measured 20.0680 seconds;
this is an initial cross-run observation, not a controlled percentage
or a 256-output-token standard result.

Full-validation runs then passed all original 256-token sequences for
the excluded warmup and three measured requests at each context:

| Context | Chunk64 mean TTFT seconds | Median decode tok/s | End-to-end output tok/s |
|---|---:|---:|---:|
| 2K | 7.4852 | 69.4378 | 22.9317 |
| 8K | 31.8365 | 70.6655 | 7.2198 |
| 32K | 147.9244 | 58.1183 | 1.6808 |

Every recorded prefill/decode major-fault counter was zero. These
runs explicitly enable full WebGPU validation, unlike the retained
benchmark invocation, so their decode rates must not replace the
retained headline or be used to attribute a decode regression.
Fresh runs retaining the original validation setting were
gated on the full-validation proofs at all three contexts.

Those retained-validation runs subsequently completed successfully,
again matching all four original 256-token sequences per context with
zero recorded major faults:

| Context | Previous / candidate TTFT seconds | Previous / candidate median decode tok/s | Previous / candidate end-to-end output tok/s |
|---|---:|---:|---:|
| 2K | 19.8739 / 7.4020 | 74.2136 / 73.5531 | 10.9807 / 23.4984 |
| 8K | 86.5425 / 31.8676 | 72.8242 / 71.9942 | 2.8419 / 7.2287 |
| 32K | 387.6879 / 147.7936 | 60.4944 / 58.8188 | 0.6532 / 1.6832 |

These are separate-run observations, not an isolated causal pair.
The approximately 62–63% TTFT reduction is the meaningful improvement;
no decode speedup is claimed. Candidate decode medians are slightly
below the retained measurements, so the retained decode headline is
not replaced. The 100-decode-tok/s goal remains unmet.

The combined qualification records 33,454,080 native finite bitwise
comparisons and 6,144 original generated-token ID comparisons across
both validation settings. Seventeen focused CPU tests pass, including
a regression test proving that a divergence after token32 cannot pass
the 256-token gate.

The prefill extension remains explicit opt-in via
`ORT_WEBGPU_QMOE_EXTENDED_PREFILL=1`, with no model/weight changes,
and no whole-model-quality qualification. No existing benchmark
defaults or retained provider/model artifacts were replaced.

### Combining faster prefill with exact dense unpacking

The separate prefill/dense build-v1 combines the qualified extended
QMoE prefill provider with the standalone dense-unpacking leaf object.
The unchanged parent first relinks byte-identically; no existing kernel
object layout changes. Native tests pass 150,036,864 finite bitwise
comparisons across both component proof families.

Controlled decode windows retain chunk64 prefill, the original validation
setting, pending64, and every parent optimization. Each context uses four
288-token requests, 32 excluded decode warmup tokens and 16 complementary
matched window pairs:

| Context | Dense unpacking off / on, median window tok/s | Matched geometric gain |
|---|---:|---:|
| 2K | 73.8842 / 75.1572 | 1.74% |
| 8K | 70.2486 / 71.3078 | 1.47% |
| 32K | 59.4097 / 60.7433 | 1.95% |

All 288 generated tokens agree across repeated requests, all original
256-token prefixes match, and every prefill/decode major-fault counter
is zero. Unlike the earlier faulted dense experiment, these complete
long-context timing runs pass the cache and correctness gates.

These are paired-window results, not a new standard 256-output headline
or a fresh standard TTFT table. The retained complete decode table is
unchanged pending standard qualification of the combined stack.
The 100 tok/s goal and whole-model-quality qualification remain unmet.

The separate default-off `ORT_WEBGPU_HC_POST_NORM_BITCAST` experiment
applies the same nibble conversion only inside the fused HC post-norm
projection. It preserves normalization, dot grouping, serial column
reduction and all four outputs. The selector has distinct shader and
replay keys without changing the existing operator object layout.
Native tests pass 607,143,936 finite bitwise comparisons over 36 fixtures,
including changing inputs/splits, fallback rows, both scale layouts,
LRU eviction, pending1/64 and invalid-selector rejection. Actual-model
full-validation profiling confirms 94 candidate dispatches.

Its full-validation 2K matched window ratio is 1.0061048; the retained
validation ratio is only 1.0029240 with two request medians favoring each
mode. This noisy result is held, not promoted or described as a decode
breakthrough. The parent combined stack remains unchanged.

A separate default-off `ORT_WEBGPU_HC_POST_NORM_FUSED_SUM` candidate
instead accumulates RMS squared values immediately from the rounded
FP16 post-mix values, avoiding one shared-memory reload pass/barrier.
Its parent relinks byte-identically and the leaf compiles, but native
build-v1 fails bitwise parity in the normalized output of the first
M1 fixture (one FP16 ULP). Explicit `quantizeToF16` in build-v2 delays
the first mismatch to round7; half packing/unpacking in build-v3 still
fails round0. The fallback M16 fixtures pass, but the M1 failures reject
all three candidates. The precise compiler-level cause is not
established. No model timing or quality claim is made for this rejected
experiment, and it is not part of the qualified stack.

### Qualified 128-token prefill chunks with dense unpacking

The same prefill/dense provider also supports chunk128 within the
already native-tested 128-token indirect-routing cap. No additional
kernel, model or weight change is needed for this chunk-size experiment.
Full validation passes the excluded warmup and three measured
256-output requests at 2K, 8K and 32K:

| Context | Mean TTFT seconds | Median decode tok/s | End-to-end output tok/s |
|---|---:|---:|---:|
| 2K | 5.3783 | 73.2979 | 28.8660 |
| 8K | 23.2166 | 72.1574 | 9.5681 |
| 32K | 110.2813 | 60.2570 | 2.2351 |

Fresh runs retaining the original validation setting then also pass:

| Context | Chunk64 / chunk128 TTFT seconds | Chunk64 / chunk128 median decode tok/s | Chunk64 / chunk128 end-to-end output tok/s |
|---|---:|---:|---:|
| 2K | 7.4020 / 5.3725 | 73.5531 / 73.8795 | 23.4984 / 28.9210 |
| 8K | 31.8676 / 23.2337 | 71.9942 / 73.8640 | 7.2287 / 9.5923 |
| 32K | 147.7936 / 110.2346 | 58.8188 / 60.1600 | 1.6832 / 2.2364 |

All 6,144 original generated-token IDs match across both validation
settings; every prefill/decode major-fault counter is zero. The
qualification independently recomputes median and pooled decode,
pooled end-to-end throughput and mean TTFT, excluding warmups, and
records hashes of the raw, guard and diagnostic evidence.
The focused CPU suite passes 103 tests.

These are separate-run observations. TTFT is a further 25--27% lower
than chunk64, with 23--33% higher end-to-end output throughput.
The candidate also enables dense unpacking, so neither the decode
differences nor every part of the end-to-end change is attributed
solely to chunk size. The earlier controlled dense windows remain the
isolated decode evidence. The retained best complete decode headline,
whole-model-quality status and 100 tok/s goal remain unchanged.

### Subsequent decode screens on the chunk128 stack

Retuning submission batching does not establish a new gain. The 2K
standard sweep uses pending64/128/256/512/64 with repeated endpoint
controls, matching all frozen 256-token outputs and recording zero
major faults:

| Pending dispatches | Median decode tok/s | Aggregate decode tok/s |
|---|---:|---:|
| 64, first control | 73.3316 | 73.7178 |
| 128 | 74.0700 | 73.4127 |
| 256 | 69.9506 | 69.1389 |
| 512 | 61.5290 | 61.5140 |
| 64, repeated control | 72.8706 | 73.1806 |

The small pending128 median difference is not a consistent aggregate
improvement; pending256/512 regress. Pending64 remains selected.
These separate-load tuning rows are not paired causal evidence.

The separate default-off `ORT_WEBGPU_HC_POST_NORM_SUBGROUP` experiment
retains FP16 materialization and the original per-thread RMS arithmetic.
It replaces the final five shared-memory reduction stages with the
same ordered FP32 shuffle tree. Eligibility requires NVIDIA with
subgroups fixed at 32 lanes. V2 uses subgroup0 only, addresses logical
values by subgroup lane rather than assuming a workgroup-lane mapping,
and writes the result to a disjoint shared-memory slot.

V1 and v2 each pass 607,143,936 finite native bitwise comparisons over
36 fixtures. V2 actual-model profiling verifies 94 candidate dispatches.
However, its 2K matched window ratios are 0.9952663 with full validation
and 0.9889891 with the retained setting, with every request median
favoring off. Outputs and zero-fault gates pass. This candidate is
rejected for H200 performance, not included in the qualified stack.

The default-off `ORT_WEBGPU_HC_UP_BITCAST_UNPACK` experiment applies
the exact nibble decoder to the private HC up-projection shader while
retaining its dot grouping, serial FP16 reduction, gates and pre-mix.
Nine native fixtures pass 18,677,760 finite bitwise comparisons,
including replay, invalid-selector and M2/M16 fallback checks.
Full-validation 2K profiling verifies 97 candidate dispatches.

Its matched 2K window ratio is 1.0034503 with full validation but
0.9957845 with the retained setting. All original 256-token prefixes,
all repeated 288-token outputs and zero-fault gates pass. This is not
a reliable gain, so the candidate remains off and is not promoted.
The same instrumented variant can diagnose context-dependent kernel
costs; those profiles are not throughput measurements or promotion.

### Long-context decode diagnosis and vector cache copy

Single measured decode-step profiles at 2K and 32K match the original
36-token prefixes and pass full validation and zero-fault gates.
They use the same instrumented HC up-bitcast variant, not a promoted
stack. GPU-timestamp sums for the main context-dependent programs are:

| Program, 12 dispatches each | 2K GPU us | 32K GPU us |
|---|---:|---:|
| SparseAttentionIndexerQsaScore | 370 | 1404 |
| SparseAttentionIndexerQsaConcat | 174 | 632 |
| SparseAttentionIndexerQsaParallelRank | 163 | 602 |
| DynamicSparseAttention | 1416 | 1418 |

This points to indexer scoring, cache copying and ranking, rather than
the selected-token attention calculation, for further long-context
work. These are instrumented diagnostic sums, not steady-state latency
fractions or causal throughput estimates.

A default-off `ORT_WEBGPU_QSA_CONCAT_VECTOR` candidate copies four
elements per invocation for NVIDIA and head sizes divisible by four.
All addresses and dispatch bounds use vector units consistently,
including packed QK offsets, batch strides and the current-key row
stride. Other cases retain the frozen scalar path. It changes neither
cache ownership/lifetime nor numerical arithmetic.

The combined chunk128 parent relinks byte-identically before replacing
only the frozen indexer leaf. All 69 native fixtures pass both flag
settings against the CPU reference: 2484 executions with bitwise checks
of both selected indices and present keys. Profiling verifies the new
copy on all 69 fixtures; invalid flag2 is explicitly rejected.
A representative FP16 32K copy falls from 52 to 22 instrumented us
(2K: 15 to 13 us). These native copy timings are not a model TPS claim.

Actual-model full-validation 2K profiling verifies 12 vector-copy
dispatches. Full-validation 2K windows pass with a 1.0005755 matched
ratio. Retained-setting matched windows then qualify through 32K:

| Context | Scalar median window tok/s | Vector median window tok/s | Matched geometric ratio |
|---|---:|---:|---:|
| 2048 | 74.0152 | 74.4789 | 1.0030427 |
| 8192 | 72.2584 | 72.6193 | 1.0071215 |
| 32768 | 61.5163 | 62.7942 | 1.0205934 |

Every original 256-token prefix and every repeated 288-token output
matches, with zero major faults. All four 32K request medians favor
the candidate. Short-context differences are small; the useful signal
is the 2.06% matched 32K gain. These paired-window rows are not standard
three-prompt decode headlines, and do not replace the qualified joint
TTFT table. Whole-model quality and 100 tok/s remain unqualified.

The next separate default-off `ORT_WEBGPU_QSA_PACKED_SCORE` experiment
packs four independent candidate blocks into 64 threads, with 16 lanes
per candidate. It retains each serial key/query RMS sum, dot product,
head sum and all barriers; shared arrays use disjoint candidate slots.
V1 passes all 69 native references in both flag modes (2484 executions),
with 27 eligible fixtures exercising the new program.
Representative instrumented scoring times are 117 to 71 us at 32K
and 41 to 35 us at 8K, but 28 to 34 us at 2K.
Therefore v2 raises eligibility from 512 to 2048 score positions,
preserving the scalar-scoring path at 2K on the original model.
V2 also passes all 69 native references in both modes (2484 executions),
with exactly 11 eligible fixtures exercising packed scoring and
explicit invalid-selector rejection. Full-validation 8K profiling
verifies 12 packed-score dispatches, and full-validation 8K windows
pass with a matched ratio of 1.0060227. Retained-setting windows:

| Context | Original median window tok/s | Packed median window tok/s | Matched geometric ratio |
|---|---:|---:|---:|
| 2048 | 75.6057 | 75.7318 | 0.9997899 |
| 8192 | 74.3696 | 74.8734 | 1.0071906 |
| 32768 | 61.3721 | 63.5219 | 1.0352147 |

All frozen 256-token prefixes, repeated 288-token outputs and zero-fault
gates pass. Every 8K and 32K request median favors packed scoring.
At 2K the shader remains unchanged; the timing difference is noise.
These independent packed-scoring rows do not include vector cache copy,
are not additive speedup estimates, and are not standard decode
headlines.

The combined indexer candidate exposes default-off
`ORT_WEBGPU_QSA_INDEXER_STACK`, enabling both optimizations while keeping
their eligibility gates and individual selector validation. Explicit
child flags retain their independent behavior when the stack flag is
off. It composes both frozen transformations on the same parent leaf,
without changing existing operator layouts or numerical reductions.

Combined native qualification passes 69 fixtures in five states:
both off, each individually on, both explicitly on, and stack on with
both child flags off. All 6210 executions match both CPU-reference
outputs bitwise. Profiles verify 3726 vector-copy and 594 packed-score
dispatches with exact eligibility coverage. Invalid stack2 is rejected
before GPU dispatch. Full-validation 8K model profiling verifies both
new programs, 12 dispatches each. Full-validation 8K windows pass with
a matched ratio of 1.0127311. Retained-setting combined windows:

| Context | Both off median window tok/s | Stack on median window tok/s | Matched geometric ratio |
|---|---:|---:|---:|
| 2048 | 75.7041 | 75.6283 | 0.9996789 |
| 8192 | 74.2668 | 75.3034 | 1.0141649 |
| 32768 | 61.5588 | 65.2856 | 1.0604607 |

All frozen 256-token prefixes, repeated 288-token outputs and zero-fault
gates pass. Every 8K and 32K request median favors the combined stack.
The measured 6.05% matched 32K gain is from the actual combination, not
an assumption that independent gains add. The 2K difference is
effectively flat. These are paired windows, not standard decode
headlines. Six standard runs (full and retained validation at all three
contexts) and independent joint TTFT/decode qualification have passed:

| Context | Previous retained median decode tok/s | New retained median decode tok/s | New mean TTFT s | New pooled end-to-end output tok/s |
|---|---:|---:|---:|---:|
| 2048 | 73.8795 | 72.1415 | 5.3830 | 28.8359 |
| 8192 | 73.8640 | 75.0755 | 23.1397 | 9.6471 |
| 32768 | 60.1600 | 64.6329 | 110.0515 | 2.2457 |

These standard comparisons are separate-run observations, not isolated
causal pairs. In particular, do not discard the lower retained 2K
result: full-validation 2K measured 74.5476 tok/s, while the paired
comparison was neutral. No cause for the run variation is established.
Full-validation 8K/32K measured 71.8419/64.4897 tok/s and
23.2409/110.2550 s TTFT. TTFT is essentially unchanged by the
decode-indexer optimizations.

The independent aggregate verifies all 6144 frozen generated token IDs,
zero major page faults, original graph/config hashes, provider/native
proof identities, exact selectors, and recomputed timing metrics.
Evidence is `qsa-indexer-stack-v1-contexts-20261029.qualification.json`.
No model weights or graph changed. Whole-model quality remains
unqualified; no context meets the 100 decode tok/s target.

The next TTFT experiment adds default-off `ORT_WEBGPU_QMOE_PREFILL256`
on the qualified combined-indexer parent. It extends indirect INT4
prefill only for exactly 256 tokens, retaining the existing canonical
geometry/FP16/NVIDIA/accumulation guards and requiring extended prefill
on and cooperative INT4 off. All existing <=128 and 129/255/257 paths
remain unchanged. The new path permits 2560 routed rows and requires
`maxComputeWorkgroupsPerDimension >= 12800`: the unfused SwiGLU
activation needs ceil(2560 * 640 / 128) workgroups, not just the
matrix-projection dispatch limit. Existing operator layouts and shader
arithmetic are unchanged.

The frozen parent relinks byte-identically before replacing only the
QMoE leaf. Native full-validation tests pass 68,843,520 finite bitwise
comparisons on six original expert layers: 16/128/256-token results
against retained 16-token shards, and 129/255/257-token results against
their unchanged full-batch fallback. Both pending1/pending64 are
covered, invalid selector2 is rejected, and profiles verify exactly
eight 256-token indirect gates per layer. Actual-model full-validation
2K profiling verifies 768 such gates across warmup and measured
requests; its instrumented timings are not performance claims.
Uninstrumented full/retained-validation 2K screens match the original
36-token prefixes with zero major page faults and observe
4.4472/4.4579 s TTFT. Each screen contains one measured request, not
the three-request standard qualification. The six full/retained
standard runs and an independent aggregate through 32K have passed:

| Context | Chunk128 retained mean TTFT s | Chunk256 retained mean TTFT s | Retained median decode tok/s | Pooled end-to-end output tok/s |
|---|---:|---:|---:|---:|
| 2048 | 5.3830 | 4.4703 | 71.8095 | 31.9342 |
| 8192 | 23.1397 | 19.3879 | 75.0368 | 11.2329 |
| 32768 | 110.0515 | 93.7860 | 64.3465 | 2.6189 |

Observed TTFT reductions are 16.95%/16.21%/14.78%; retained median
decode differs by less than 0.5% at each context. These are separate-run
observations, not isolated causal pairs. Full-validation TTFT is
4.4669/19.3473/93.7379 s, with median decode
71.1882/74.4962/62.0636 tok/s; the lower full-validation 32K decode
result is retained in the evidence, not filtered out.
The independent aggregate rechecks all 6144 frozen output IDs, zero
major faults, graph/config hashes, native/source identities, selectors,
and timing formulas. Evidence is
`prefill256-v1-contexts-20261029.qualification.json`.
This is the current fully qualified TTFT/decode experimental stack,
not whole-model quality qualification or a 100 tok/s result.

The next decode experiment uses default-off `ORT_WEBGPU_QSA_SCORE16`
on this chunk256 parent. The supplied model has four indexer query
heads. A new private `QsaPackedScore16` program assigns 16 independent
candidates to a 64-thread workgroup, four lanes each. It preserves
serial key/query RMS sums, dot products, head sums, all workgroup
barriers and the single visible-count writer. Candidate shared slots
are disjoint; head-score slots are compacted to four per candidate.
The original four-candidate shader and every existing fallback remain
unchanged. Eligibility additionally requires the original packed-score
or indexer-stack selector, global cache, one row, at most four heads,
NVIDIA and at least 8516 bytes of workgroup storage.

V1 native testing passes 69 fixtures in both modes (2484 CPU-reference
executions), with 198 candidate dispatches. Representative
instrumented 32K scoring falls from 70 to 56 us, but 8K regresses from
35 to 49 us. V1 is not promoted. Immutable v2 therefore raises the
minimum score stride from 2048 to 8192, leaving existing 2K/8K scoring
untouched. V2 independently passes the same 2484 reference executions,
with exactly nine eligible fixtures and 162 candidate dispatches.
All other program counts match between modes, and invalid selector2
is rejected before GPU dispatch. The 32K native timing remains
70 versus 56 us; this is a shader diagnostic, not a whole-model
speedup claim. Actual-model full-validation 32K profiling verifies
12 candidate dispatches. Full-validation 32K paired windows pass
with a matched ratio of 1.0092327. Retained-setting windows:

| Context | Packed4 median window tok/s | Packed16 median window tok/s | Matched geometric ratio |
|---|---:|---:|---:|
| 2048 | 75.8572 | 75.8826 | 0.9974012 |
| 8192 | 75.2837 | 75.2539 | 0.9995611 |
| 32768 | 65.0206 | 65.5177 | 1.0073800 |

All four 32K request medians favor packed16 in both validation modes.
2K/8K keep the same shaders; their small timing differences are
retained, not filtered out. All original 256-token prefixes and
repeated 288-token outputs match, with zero major page faults.
The independent aggregate rechecks medians (including each request),
matched ratios, native/profile identities, graph/config hashes and
4096 window-reference prefix IDs. Evidence is
`qsa-score16-v2-windows-contexts-20261029.qualification.json`.
This is a small controlled 32K gain, not a standard headline replacement
or a 100 tok/s result.

Further native-only screens retain the same default-off selector and
frozen chunk256 parent:

| Version | Change | Model-shaped 8K packed4/candidate us | Model-shaped 32K packed4/candidate us |
|---|---|---:|---:|
| v3 | Shared key stride132, otherwise v1 | 35 / 48 | 71 / 51 |
| v4 | v3 plus per-workgroup FP32 query normalization/rotary cache | 35 / 42 | 70 / 44.5 |
| v5 | v4 with 256 threads, key stride144 | 35 / 29 | 70.5 / 35 |

Each version independently passes 69 fixtures in both modes (2484
CPU-reference executions), 198 candidate dispatches, all unchanged
fallback program counts and invalid-selector rejection. These compare
the selected indices and present keys exactly; they are not whole-model
quality qualification. V3/v4 retain the 8K regression and are not
promoted. V5 improves both model-shaped native cases and is selected
for model testing, not yet as a new decode headline.

Query RMS sums remain serial per head and use the original
`query_value` implementation; each workgroup reuses FP32 cached query
values across its 16 candidates. V5 maps 16 lanes to each candidate
in a 256-thread workgroup. Shared query stride129 and key stride144
give disjoint slots. It adds two unconditional barriers and requires
11620 bytes of workgroup storage, workgroup sizeX/invocations >=256,
one row, <=4 heads, NVIDIA, global cache, and score stride >=2048.
Original packed4 and short-context paths remain unchanged.
Native timing reductions must not be interpreted as whole-model
TPS gains; full-validation 8K/32K active-path profiles and paired
windows through 32K have passed independent aggregate verification.
Each active-path profile verifies 12 `QsaPackedScore16QueryWide`
dispatches. Full-validation matched ratios are 1.0041477 at 8K and
1.0265357 at 32K. Retained-setting results:

| Context | Packed4 median window tok/s | Cached wide median window tok/s | Matched geometric ratio |
|---|---:|---:|---:|
| 2048 | 75.5277 | 75.5232 | 0.9997264 |
| 8192 | 75.2934 | 74.4176 | 0.9998600 |
| 32768 | 64.0179 | 66.6825 | 1.0286785 |

All four 32K request medians favor the candidate in both validation
modes. The 8K aggregate median and matched-window results differ:
retain both, and do not claim a retained 8K speedup from its favorable
per-request medians. No timing windows were excluded; no cause for
the variation is established. The 2K shader remains unchanged.
All 5120 frozen window-prefix IDs and repeated 288-token outputs
match, with zero major page faults, original graph/config hashes
and verified native/profile/source identities.
Evidence is `qsa-score16-v5-windows-contexts-20261029.qualification.json`.
Six standard full/retained-validation runs and independent joint
TTFT/decode qualification have now passed all 6,144 original generated
IDs, zero-major-fault gates and provider/model identities. The focused
CPU suite passes 99 tests, including exact per-context 100 tok/s
thresholds and exclusion of instrumented/comparison runs.

| Context | Previous retained decode tok/s | V5 retained decode tok/s | V5 full-validation decode tok/s | Retained mean TTFT s | Retained end-to-end output tok/s |
|---|---:|---:|---:|---:|---:|
| 2K | 71.809470 | 75.290476 | 74.750960 | 4.466031 | 32.586310 |
| 8K | 75.036811 | 71.780574 | 75.114396 | 19.398040 | 11.169198 |
| 32K | 64.346466 | 66.370386 | 65.979152 | 93.661739 | 2.625350 |

These are separate standard-run observations, not isolated causal
comparisons. In particular, the 2K score shader is unchanged, and the
lower retained 8K result is not discarded or explained away. Full
mean TTFT is 4.485433/19.384238/93.642888 s at 2K/8K/32K.
The stronger evidence for the 32K scoring improvement remains the
controlled +2.87% paired result above; 8K paired timing is neutral.
This is not an unconditional all-context performance upgrade.
The candidate remains default-off, below 100 tok/s, and unqualified
for whole-model quality.

### Post-warmup CPU diagnosis and exact postnorm padding candidate

A separate retained-validation 2K CPU capture samples only the owned
GPU7 container, after its excluded warmup. Its 12-second scope includes
both prefill and decode; it is not a decode-only or throughput result.
All 1,024 original generated IDs and zero-major-fault gates pass.
The actual recorded event is `cpu-clock:uH`, with 521 samples and zero
lost samples. Container symbols were resolved offline using the frozen
provider/ORT binaries and matching image libraries, without another GPU
run; initial unresolved reports and the original capture remain intact.

Inclusive active-CPU samples include environment parsing at 9.79%,
`getenv` at 8.45%, context `Run` at 15.16%, tensor copying at 11.32%,
`Flush` at 10.56% and bind-group creation at 4.99%. These nested
percentages overlap, are not wall-time fractions and do not predict
equivalent tok/s gains. No environment-caching change is implemented.

The existing instrumented 32K decode trace identifies 94 fused postnorm
projection dispatches totaling 2,960 us. The distinct default-off
`ORT_WEBGPU_HC_POST_NORM_PAD34` experiment preserves their arithmetic,
FP16 materialization, reduction order, 256 threads and 41 workgroups,
but changes the private shared-memory stride from 36 to 34 half
elements. Storage falls from 11,520 to 10,880 half elements. For a
32-lane block-start address calculation, this changes 16 distinct
32-bit bank indices to 32; it does not prove actual bank stalls or
a speedup, especially given compiler packing and hardware ports.

Four new CPU tests cover all logical addresses/projection reads,
unchanged arithmetic, original shader/fallback and replay identity.
The frozen leaf builds on the qualified parallel2 parent with a
byte-identical unmodified-parent relink. Native four-output,
changing-input qualification passed 607,143,936 finite bitwise
comparisons over 36 fixtures, including CPU-split changes, pending
windows 1/64, replay eviction, 24 fallback fixtures and invalid-selector
rejection. The provider SHA256 is
`77e89372952c07600e494f0ee6d9f213d2de30fecbbcaa2f3c0451860e4ae877`.
Whole-model metadata and complementary-window routing are wired;
74 targeted regression tests and Python compilation pass. The model
runner preserves the qualified parallel2 and score16 selectors on both
comparison sides. Native correctness is not a performance result.

The subsequent full-validation 2K active-path profile confirms exactly
94 `HCPostNormProjectionPad34F16` dispatches and no original fused
postnorm programs, with all original generated IDs and zero faults.
Complementary 2K windows pass all original prefixes and repeated
288-token sequences, but do not establish a gain:

| Validation | Stride36 median window tok/s | Stride34 median window tok/s | Matched ratio |
|---|---:|---:|---:|
| Full | 72.219224 | 72.027055 | 0.9993128381 |
| Retained | 76.549501 | 76.287772 | 0.9994795524 |

These are controlled windows, not replacement standard headlines.
Leave PAD34 off; no 8K/32K or TTFT gain is claimed, and no six-run
standard batch is justified by this screen.

The next distinct experiment, default-off `ORT_WEBGPU_DENSE_TILE4`,
clones only the standalone bitcast INT4 family and changes its output
column tile from eight to four. It retains 128 threads, the frozen
unpacking template, per-column arithmetic/reductions and eligibility;
its replay variant is distinct from the original bitcast program.
This tests fewer serial columns/registers per warp against twice as
many workgroups, not a claimed speedup. It builds independently on the
qualified parallel2 parent, with PAD34 off.

The dense tile4 leaf has a byte-identical unmodified-parent relink
and provider SHA256
`7122cd597e37899c1f7a15570916e351fd6fa9408c1020956abed1bfa44c3090`.
Native qualification passes 116,582,784 finite bitwise comparisons
over 27 fixtures (nine shapes, rows1/2/16), changing inputs, pending1/64,
both replay variants and invalid-selector rejection. A full-validation
2K profile verifies 159 `MatMulNBitsTile4F16` dispatches and no original
standalone bitcast programs; all original generated IDs and zero faults
pass. The 79-test focused regression suite also passes.

| 2K validation | Tile8 median window tok/s | Tile4 median window tok/s | Matched ratio |
|---|---:|---:|---:|
| Full | 76.156558 | 77.021140 | 1.0121266129 |
| Retained | 76.319812 | 77.158335 | 1.0146596326 |

All eight request-median comparisons favor tile4. Frozen original
prefixes, complete repeated 288-token sequences and zero-major-fault
gates pass. These are controlled-window gains, not a new standard
headline or TTFT claim. All six controlled comparisons have now
completed and independently revalidated:

| Context | Full matched ratio | Retained tile8/tile4 median window tok/s | Retained matched ratio |
|---|---:|---:|---:|
| 2K | 1.0121266129 | 76.319812 / 77.158335 | 1.0146596326 |
| 8K | 1.0146393291 | 75.093006 / 76.931277 | 1.0140419599 |
| 32K | 1.0128755519 | 67.296007 / 68.197531 | 1.0157684493 |

All 24 per-request median comparisons favor tile4. The aggregate
revalidates 6,144 original prefix IDs, complete repeated sequences,
native fixtures/dispatches, provider/model hashes and zero faults.
The frozen template is checked against its original build manifest.
The focused suite now passes 86 tests. Reusing the qualification
helpers reproduces both original parallel2 paired and standard
aggregates exactly, without changing their evidence files.
The six standard tile4 runs and independent aggregate have completed:

| Context | Retained median decode tok/s | Full median decode tok/s | Retained mean TTFT s | Retained E2E output tok/s |
|---|---:|---:|---:|---:|
| 2K | 75.992064 | 73.414878 | 4.462924 | 32.705284 |
| 8K | 77.523379 | 76.371554 | 19.381020 | 11.288043 |
| 32K | 65.215279 | 64.737294 | 93.766136 | 2.622117 |

All 6,144 original generated IDs, zero faults and provider/model
identities pass. These are separate-run observations. In particular,
retained 32K is below the preceding 65.585698 tok/s, despite its
favorable controlled comparison; neither that decrease nor the lower
full-validation 2K median is discarded. No unconditional standard
32K gain, material TTFT improvement, whole-model quality qualification
or 100 tok/s result is established.

A separate next TTFT experiment, default-off
`ORT_WEBGPU_QMOE_PREFILL512`, extends only exact 512-token QMoE
eligibility on the existing extended INT4 path, requires a 25,600
per-dimension workgroup limit, and caps routed rows at 5,120.
The original 256-token route, shaders and floating-point arithmetic
remain intact. Its selector is validated before replay and included
in the existing command-cache key. Boundary tests cover original
16/128/256-token cases, 511/513 fallback cases and the intended
512-token case. CPU-heavy builds waited until standard timing completed.
The initial build-v1 failed on a missing quoted shader include and is
preserved. Build-v2 pins all original component sources and adds their
local include directory; native qualification passes 143,523,840 finite
bitwise comparisons over six original-weight layers, including
512-token dispatch coverage, pending1/64 and invalid-selector rejection.

The first full-model chunk512 attempt fails explicitly at the existing
DynamicSparseAttention 256-token bound; it is not a timing result.
Build-v3 extends only that host admission check for the same default-off
selector, NVIDIA, exactly 512 tokens, one batch and query-workgroup counts
within the actual device limit. All other shapes retain the original
256-token bound and all tensor/storage limits remain unchanged.
Before modifying admission, the original attention object was recompiled
byte-identically against the frozen linked object; its WGSL, reductions,
normalization and absolute cache addressing are unchanged.
The provider SHA256 is
`c1f259b3a26275c58098fa159082cf5e69ebed68916799ff747a300a2037e110`.

Independent attention qualification passes 83,238,912 finite CPU-reference
comparisons, checking attention and both cache outputs, FP16/FP32,
selected-main and local-plus-auxiliary modes, sink/smooth/empty cases,
and pending1/64. It uses existing FP16 atol0.0005/rtol0.005 and FP32
atol0.000002/rtol0.0005, not a bitwise or whole-model quality claim.
Twelve negative checks verify flag0, invalid selectors, 511/513 tokens,
batch2 and over-limit dispatch remain explicitly rejected.
An earlier synthetic fixture accidentally combined unsupported local
attention with main selection; its failed evidence is preserved, and
fresh fixtures use the operator's supported local-plus-auxiliary mode.
Combined build-v3 QMoE requalification also passes 143,523,840 finite
bitwise comparisons. Full-validation 2K profiling verifies 384 M512
gate dispatches, original output prefixes and zero faults. Separate
full/retained 36-output screens pass, with TTFT 4.029593/4.026349 s;
these short screens are not standard headline measurements.
All six standard runs and their independent aggregate through 32K pass
6,144 original generated IDs, zero major faults and provider/model
identities:

| Context | Chunk256 / chunk512 retained TTFT s | Chunk512 retained median decode tok/s | Chunk512 full median decode tok/s | Chunk512 retained E2E output tok/s |
|---|---:|---:|---:|---:|
| 2K | 4.462924 / 4.041451 | 75.212415 | 73.370475 | 34.597373 |
| 8K | 19.381020 / 17.455671 | 73.159225 | 74.990760 | 12.246067 |
| 32K | 93.766136 / 84.971422 | 65.916389 | 65.760269 | 2.882307 |

Observed retained TTFT reductions are 9.44%/9.93%/9.38%, and full
TTFT is 4.044866/17.514047/84.924790 s. These are separate-run
observations, not isolated causal pairs. Retained decode at 2K/8K
is below the preceding 75.992064/77.523379 tok/s; the 8K decrease
is 5.63% and is not filtered out or assigned an unproven cause.
This is a TTFT/end-to-end improvement, not an unconditional decode
upgrade. Evidence is `prefill512-v3-contexts-20261030.qualification.json`.
The 100 tok/s target and whole-model-quality qualification remain unmet.

A distinct default-off `ORT_WEBGPU_HC_PROJECTION_SHUFFLE` experiment
builds on this provider with a byte-identical unmodified-parent relink.
Its provider SHA256 is
`970e5100d455ef1220c6c6e79866a5bb90fa5d4d570f61bd8701ed3dcd6a0d68`.
GPU-native qualification passes 607,143,936 finite bitwise comparisons
over 36 fixtures, all four outputs, changing inputs/CPU splits,
pending1/64, fallback rows, replay eviction and invalid selectors.
Full-validation 2K profiling verifies 94 candidate dispatches and
zero original fused postnorm programs, with all original prefixes
and zero faults. Complementary 2K timing windows pass correctness
but reject the candidate for performance:

| Validation | Shared / shuffle median window tok/s | Matched ratio |
|---|---:|---:|
| Full | 76.773510 / 76.730017 | 1.0000142 |
| Retained | 76.930212 / 76.107262 | 0.9938546 |

All four retained request medians favor off. The selector remains
off; no long-context or standard timing escalation is justified.
It preserves the fused postnorm shader's normalization and per-column
dot products, replacing only its final serial FP16 shared-memory
reduction with the same 32 ordered FP16 additions fed by FP32 subgroup
shuffles. Every finite FP16 value survives FP32 transport exactly.
Subgroup IDs/lanes define projection columns independently of local
invocation ordering. Eligibility requires NVIDIA, subgroup support
and fixed 32-lane subgroups; other cases retain the original shader.
The candidate has a private program identity and replay variant.
Unlike the rejected subgroup RMS experiment, its normalization tree
is unchanged. The model runner accepts explicit parent evidence
date, provider path and chunk size, preserving existing defaults.
The 76 focused CPU tests pass; original parallel2 six-standard
aggregation is independently reproduced exactly after helper reuse.
There is no performance gain for this experiment.

The next default-off `ORT_WEBGPU_DENSE_TILE2` experiment instead
clones the qualified standalone tile4 family with two output columns
and 64 threads, retaining 32 lanes per column and its original dot
products/serial FP16 reductions. It doubles workgroups while keeping
the total column warps unchanged, at the cost of more activation
loads and workgroup overhead. Original tile8/tile4 shaders and
eligibility remain intact; replay variants 0/1/2 are unchanged and
variant3 is reserved for tile2. This candidate descends from the
qualified chunk512 stack, not the rejected projection shuffle.
Build-v1 links but fails its comparator-source transformation because
the inherited comparator already contains the requested diagnostics;
its output is preserved. Fresh build-v2 validates/reuses those existing
diagnostics and passes the byte-identical parent relink. Provider
SHA256 is `e87fedf6e3ec7cad978d6bb82b5d4bbbdacb89a1069a618de7da0e76ecb12294`.
Native qualification passes 116,582,784 finite bitwise comparisons
over 27 fixtures, changing inputs, pending1/64, fallback rows,
both replay variants and invalid selectors. Full 2K profiling verifies
159 candidate dense programs and zero tile4 programs. Both complementary
2K comparisons pass original/repeated outputs and zero faults, but regress:

| Validation | Tile4 / tile2 median window tok/s | Matched ratio |
|---|---:|---:|
| Full | 72.591457 / 71.697746 | 0.9875787 |
| Retained | 73.843085 / 71.005776 | 0.9818511 |

Every request median favors tile4. Tile2 remains off and is not
escalated to long-context or standard timing.

The distinct default-off `ORT_WEBGPU_DENSE_WIDE8` candidate tests the
opposite tradeoff: eight columns with 256 threads, still 32 lanes per
column, all columns concurrent rather than the original tile8's two
serial columns per warp. It halves workgroups relative to tile4 and
reuses each loaded activation tile across twice as many columns.
The original per-column arithmetic/reduction remains unchanged.
Eligibility also requires sizeX/invocation limits of at least256;
unsupported devices retain tile4/tile8 and their original replay keys.
The byte-identical parent relink passes. Provider SHA256 is
`5c33c5475a77eb91a9a1e29d938dda1e9b118247ed80bdc12f9b060680262546`.
Native qualification passes 116,582,784 finite bitwise comparisons
over the same 27 fixtures and replay/fallback/invalid-selector cases.
Full 2K profiling verifies 159 candidate programs and zero tile4
programs. The 88 focused CPU tests pass. Both 2K complementary-window
comparisons preserve original/repeated outputs and zero faults:

| Validation | Tile4 / wide8 median window tok/s | Matched ratio |
|---|---:|---:|
| Full | 71.288431 / 71.230706 | 1.0015918 |
| Retained | 74.041442 / 73.749903 | 1.0029996 |

These are effectively flat, mixed results: candidate request medians
win three of four full comparisons but only one of four retained
comparisons, and both pooled medians favor tile4. The matched ratio
and pooled medians measure different things; retain both. Wide8
remains off, without long-context/standard escalation or a speed claim.
The fully standard-qualified chunk512 parent remains unchanged.

Three additional default-off screens also do not establish decode gains.
All model windows below preserve frozen/repeated generated IDs and have
zero major faults. They use the qualified chunk512 parent and the same
complementary 2K window protocol, not new standard headline measurements.

`ORT_WEBGPU_HC_STAGED_PROJECTION` computes normalization once in a
four-workgroup shader and consumes it in a separate 162-workgroup projection
shader. Both preserve the original arithmetic and have independent
single-command replay caches. A private replay helper keys additional GPU
input shapes and validates their binding order; common runtime layouts
remain unchanged. The exact parent relink passes, as do 607,143,936 finite
bitwise comparisons over 36 fixtures and all 36 replay-stat records.
Actual full-model profiling verifies 94 normalization and 94 projection
programs, with no original fused programs. Their instrumented GPU sum is
3,709 microseconds versus 2,990 for the original fused path; neither number
is steady-state model latency.

`ORT_WEBGPU_HC_UP_PARALLEL4` packs four concurrent HC-up columns into
256 threads, versus the qualified parallel2 path's two columns. It retains
the 16-lane branch mapping, 32-slot partials and ordered FP16 reduction.
Device guards require 256 threads and 1,056 bytes of shared storage.
The exact parent relink and 18,677,760 finite bitwise comparisons over nine
fixtures pass; full-model profiling verifies 97 candidate dispatches.

| Experiment | Validation | Parent / candidate median window tok/s | Matched ratio |
|---|---|---:|---:|
| Staged HC projection | Full | 75.976732 / 75.309042 | 0.9876117 |
| Staged HC projection | Retained | 73.693031 / 73.260301 | 0.9861865 |
| HC-up parallel4 | Full | 73.549388 / 72.755363 | 0.9955505 |
| HC-up parallel4 | Retained | 73.904932 / 73.790385 | 0.9995067 |

Both selectors remain off. Staged projection loses every request-median
comparison. Parallel4 is effectively flat/slower. Neither is escalated to
long-context or standard timing.

The host-only `ORT_WEBGPU_CPU_LOCAL_WINDOWS` screen moves only the benchmark
main thread from CPUs 0-95 to GPU7's NUMA-local CPUs 48-95 at excluded window
boundaries. Worker affinities are unchanged. The harness verifies requested
and restored masks and restores the original environment even if affinity
restoration fails. GPU topology and provider/model identities are recorded;
no clocks, power settings or foreign-process affinities are changed.
The four implementation/helper sources are frozen alongside the CPU-local
evidence before subsequent benchmark selector additions; their hashes match
both original qualification records.

| Validation | Default / local median window tok/s | Matched ratio |
|---|---:|---:|
| Full | 76.439215 / 76.479608 | 1.0030882 |
| Retained | 77.010411 / 76.943389 | 1.0022191 |

Only three of four full and two of four retained request medians favor
local placement. These mixed sub-percent observations do not establish a
reliable gain; locality is not enabled in the qualified stack. The current
six-standard chunk512 result and its 2K/8K decode caveats remain unchanged.

The default-off `ORT_WEBGPU_DENSE_PRELOAD` screen preloads the entire
single-row activation into workgroup memory once, retaining shared reuse
and the original K-loop, dot products and ordered reductions. This differs
from the previously rejected direct-global activation loads. A private
shader template uses full-K shared offsets and two uniform barriers per
workgroup; device shared-storage admission accounts for both activation
and reduction arrays. Full K is included in the program cache hint.
The original tile4/tile8 programs, fallbacks and replay variants remain.
The exact parent relink passes. Native qualification passes 116,582,784
finite bitwise comparisons over 27 fixtures, pending1/64, changing inputs,
multi-row fallbacks, replay variants and invalid selectors. Full-model
profiling verifies 159 candidate programs and zero tile4 programs.

| Validation | Tile4 / preload median window tok/s | Matched ratio |
|---|---:|---:|
| Full | 70.336766 / 67.612581 | 0.9474538 |
| Retained | 73.503555 / 69.144166 | 0.9416185 |

Both runs preserve original/repeated generated IDs and zero major faults,
but every request median favors tile4. Preload is rejected and remains
off; it is not escalated to longer contexts or standard qualification.

The default-off `ORT_WEBGPU_QMOE_FAST_FLAGS` screen targets host overhead
rather than GPU arithmetic. Each compute invocation selects a local parser
which directly reads exact empty/unset/`0`/`1` values; all other strings
use the original integer parser, preserving its errors and accepted values.
There is no persistent environment cache, shader change, replay-key change
or kernel-object layout change. The original mode remains available within
each complementary timing window. The exact chunk512 parent relink passes.
Native qualification passes 9,748,480 finite bitwise comparisons over
23 changing-input/router fixtures, direct identity-routing checks and
pending1/64. A further 196 parser cases cover 14 selectors, including unset,
empty, whitespace, signed/leading-zero, nonbinary and invalid values.
Errors agree exactly between modes. The verifier explicitly reconciles
72 expected failed key-parse requests with replay counters, rather than
treating them as successful captures/hits. Full-model profiling verifies
48 each of the unchanged hierarchical gate, fused FC1 and fused FC2 programs.

| Validation | Original / fast parser median window tok/s | Matched ratio |
|---|---:|---:|
| Full | 75.380828 / 75.440279 | 0.9909288 |
| Retained | 74.903277 / 75.065449 | 0.9938452 |

Original/repeated generated IDs and zero major faults pass. Pooled medians
are marginally higher but matched ratios favor the original parser; request
medians are mixed (two of four full, one of four retained favor candidate).
This does not establish a reliable gain. The candidate remains off and is
not escalated. The complete standard-qualified headline remains chunk512.

The default-off `ORT_WEBGPU_DSA_PARTITION32` screen splits sparse attention
into 32-key partitions rather than 64, retaining the existing shader
expressions and stable merge. A private program identity and dispatch-budget
guard preserve original fallbacks. This intentionally changes FP32 softmax
association across partitions, unlike the exact-arithmetic dense screens.
The exact chunk512 parent relink passes. Native qualification passes
972 independent CPU-reference executions over 27 FP16/FP32 fixtures,
pending1/64, 18 fixed-selector capture executions and invalid-selector
rejection. Profiles verify 207 candidate/36 fallback dispatches per pending
mode. On the model-shaped operator fixture, attention plus merge medians
are 77+28 versus 46+42 microseconds (pending1) and 77+27 versus 45+43
(pending64). These are instrumented operator diagnostics, not model TPS.

Despite passing the existing CPU tolerances, the actual full-2K model
profile changes the frozen generated IDs: warmup first differs at token
index21 (15 of36 positions), and the measured request first differs at
index19 (17 of36). An independent rejection report checks both prompt
identities, zero faults and 12 active candidate programs/zero original
programs. The candidate is rejected, remains off and is not timed in
model windows or escalated. No throughput gain is claimed from this result.
The initial runner import failure and subsequent per-fixture-profile
verification failure are preserved separately; the final native proof
uses the complete r3 run without changing tolerances.

### 1024-token prefill and subsequent exact decode screens (October 30)

The default-off `ORT_WEBGPU_QMOE_PREFILL1024` extends the existing
extended-prefill path to exactly 1024 input rows and 10240 routed rows
under the canonical INT4 geometry, cooperative INT4 disabled and a device
workgroup-dispatch limit of at least 51200. Existing 16/128/256/512 paths
and 1023/1025 fallbacks remain unchanged. The joint build-v2 additionally
admits exactly 1024 rows in NVIDIA batch-one sparse attention, with its
sequence-times-heads dispatch bound checked independently.

Both component builds reproduce their frozen parent provider exactly
before replacement. Expert qualification passes 300,810,240 finite bitwise
comparisons over six original layers. Attention qualification passes
168,256,512 CPU-reference comparisons, pending1/64, 12 expected rejections
and legacy256/512 fixtures. Full-model profiling verifies 192 active
M1024 expert gates. All six full/retained 36-token screens and all six
256-output standard runs preserve frozen generated IDs, including warmup,
with zero major faults. The independent standard aggregate passes.

| Context | Chunk512 retained median decode tok/s | Chunk1024 full / retained median decode tok/s | Chunk512 retained TTFT s | Chunk1024 full / retained TTFT s |
|---|---:|---:|---:|---:|
| 2K | 75.212415 | 74.263652 / 75.464409 | 4.041451 | 3.943610 / 3.953578 |
| 8K | 73.159225 | 70.281738 / 71.757639 | 17.455671 | 16.856544 / 16.782570 |
| 32K | 65.916389 | 65.610065 / 63.610607 | 84.971422 | 81.816165 / 81.884637 |

These are separate-run observations, not isolated causal decode comparisons.
Retained TTFT decreases 2.17%, 3.86% and 3.63%; retained decode at 8K and
32K is lower. Chunk1024 is a correctness-qualified TTFT-oriented candidate,
not evidence of a decode breakthrough or attainment of 100 tok/s.
The unchanged 256-output protocol uses median request decode throughput,
255 transitions divided by decode time; it does not average prefill and
decode rates. End-to-end retained output throughput is 34.889538,
12.606994 and 2.981352 tok/s respectively. Whole-model quality remains
unqualified beyond frozen-ID regression checks.

Two subsequent default-off decode candidates use this joint build-v2 as
their frozen parent:

- `ORT_WEBGPU_DENSE_DOUBLE_BUFFER` alternates two shared activation tiles
  while preserving tile dimensions, K-loop arithmetic, packed decoding
  and ordered reductions. The previous iteration's initial load barrier
  protects the buffer reused two iterations later. The per-tile trailing
  barrier is replaced by one final barrier before reading partial sums.
  Storage admission accounts for both activation tiles and the accumulation
  type. Randomized four-warp scheduling tests check buffer reuse.
  Native qualification passes 116,582,784 finite bitwise comparisons over
  27 changing-input fixtures, pending1/64 and replay-key separation.
  An additional 116,582,784 comparisons verify the existing FP32-accumulation
  fallback; that mode deliberately bypasses the optimized route and replay,
  so it does not qualify an FP32 double-buffer shader. The initial verifier
  incorrectly expected replay statistics in this fallback; its failed
  attempt and the original successful GPU execution are preserved, and
  independent re-verification requires no statistics rather than inventing
  replay activity. Full-2K profiling preserves frozen IDs and verifies
  159 candidate programs and zero original tile4 programs.
- `ORT_WEBGPU_DSA_QUERY_CACHE` retains 64-key partitions and the original
  merge, caching eight query values per lane and emitting dot updates in
  their original address order. Build-v1 fails strict bitwise comparison
  on the FP32 D256 fixture despite passing the preceding FP16 fixtures;
  it is rejected without relaxing tolerances. Build-v2 restricts admission
  to FP16 and preserves the original FP32 fallback. Its exact parent
  relink passes, followed by 4,870,656 finite bitwise comparisons across
  all 27 fixtures, 486 CPU-reference executions, nine fixed-selector
  capture executions and invalid-selector rejection. Profiles verify
  18 candidate and 225 fallback dispatches per pending mode. This native
  result alone is not a full-model throughput or quality claim.

The double-buffer full/retained 2K complementary-window screen completes
with frozen/repeated IDs and zero faults:

| Validation | Tile4 / double-buffer median window tok/s | Matched ratio |
|---|---:|---:|
| Full | 73.370205 / 74.845155 | 1.0007860 |
| Retained | 77.151662 / 76.962389 | 0.9971814 |

Only one of four full-validation request medians favors double buffering;
all four retained request medians regress. The higher pooled full median
does not establish a reliable gain. The candidate is rejected, remains
off and is not escalated to longer contexts.

Query-cache build-v2 completes all six full/retained complementary-window
runs and independent aggregation. All 6144 frozen prefix IDs, complete
repeated 288-token sequences and zero-major-fault checks pass:

| Context | Full matched ratio | Retained original / query-cache median window tok/s | Retained matched ratio |
|---|---:|---:|---:|
| 2K | 1.0269745 | 74.663612 / 77.502460 | 1.0293709 |
| 8K | 1.0230897 | 75.031067 / 77.412599 | 1.0284438 |
| 32K | 1.0250502 | 68.016680 / 69.601200 | 1.0246085 |

All 12 retained request medians favor query caching; 11 of 12 full
request medians favor it. These are controlled-window gains, not new
standard decode headlines. All six full/retained 256-output standards
also pass independent qualification, frozen IDs including warmup, and
zero major faults:

| Context | Full median decode tok/s | Retained median decode tok/s | Full TTFT s | Retained TTFT s | Retained E2E output tok/s |
|---|---:|---:|---:|---:|---:|
| 2K | 75.122107 | 76.821994 | 3.943251 | 3.950710 | 35.219110 |
| 8K | 73.150985 | 77.117301 | 16.803089 | 16.807248 | 12.728347 |
| 32K | 67.306263 | 65.914297 | 81.886629 | 81.859820 | 2.984172 |

The frozen provider SHA is
`c473e6c890afd7988c29d895f09189288c5fa54d77b8cea96a2cd0ea5f70e75a`.
Retained median ratios against the separately measured prefill1024
parent are 1.017990, 1.074691 and 1.036216. These separate-run differences
are not isolated causal gains; the controlled evidence above supports
roughly 2.5-2.9%. TTFT is essentially unchanged by this decode optimization.
The 100 tok/s goal and whole-model quality qualification remain unmet.
The generalized standard collector retains
its chunk256 default, and its existing dense-tile4 standard rows are
independently reproduced exactly without rewriting evidence.

The independent default-off `ORT_WEBGPU_GDN_VALUE2` prototype uses two
rather than four value channels per recurrent workgroup, leaving
per-channel arithmetic, reductions and original shader unchanged.
Admission is limited to NVIDIA batch-one, single-token, packed FP16
QKV with Dk/Dv128 and a checked dispatch bound. The original GDN object
is recompiled byte-identically before replacement, and the complete
parent relink is also exact. Native pending1/64 runs pass 229,398,768
finite bitwise comparisons over 36 fixtures, checking both output and
final state, invalid selectors, and batch/multitoken/dtype/odd-shape
fallbacks. Profiles verify 18 candidate and 630 original dispatches
per pending mode. The first fixed-capture harness used ordinary unbound
output allocation and failed with an unconstructed-tensor error.
A separately built comparator using the existing explicit CPU-output
IoBinding pattern passes fixed-capture execution for both outputs,
bringing the independently verified total to 236,531,952 finite bitwise
comparisons. The failed harness evidence is preserved. These native
checks do not establish a model-throughput gain.

The full 2K profile verifies 36 `GatedDeltaNetValue2` programs and
zero original recurrent programs. Both complementary 2K screens pass
frozen/repeated outputs and zero faults, but reject this tiling for
performance:

| Validation | Value4 / value2 median window tok/s | Matched ratio |
|---|---:|---:|
| Full | 75.906171 / 76.211374 | 0.9951467 |
| Retained | 75.607805 / 74.136829 | 0.9959618 |

The pooled full-window median and matched estimator disagree; the
predeclared matched estimator is below one in both modes. Value2
remains off and is not escalated to longer contexts.

The independent default-off `ORT_WEBGPU_DENSE_BLOCK32` prototype
specializes integer addressing in the already canonical single-row,
batch-one, block32 tile4 route. The runtime block divisor becomes 32,
batch and A-row indices become zero, and the column-base coordinate
becomes `workgroup_idx * tile_size`. Floating-point dot/update/reduction
source and the original shader/fallbacks remain unchanged. The original
object and complete parent relink reproduce exactly. Native pending1/64
execution passes 116,582,784 finite bitwise comparisons over 27 fixtures,
including changing inputs, replay and fallbacks. Model timing remains
separate from this correctness proof.

The full 2K profile verifies 159 `MatMulNBitsBlock32F16` programs and
zero original tile4 programs. Both complementary screens pass frozen
and complete repeated outputs and zero faults:

| Validation | Tile4 / block32 median window tok/s | Matched ratio |
|---|---:|---:|
| Full | 73.287777 / 74.392074 | 1.0143484 |
| Retained | 77.465186 / 78.037194 | 1.0081201 |

All eight 2K request median comparisons favor block32. Independent
aggregation now also verifies the remaining four windows through 32K,
including all 6144 frozen prefix IDs, complete repeated 288-token
sequences and zero faults:

| Context | Full matched ratio | Retained tile4 / block32 median window tok/s | Retained matched ratio |
|---|---:|---:|---:|
| 2K | 1.0143484 | 77.465186 / 78.037194 | 1.0081201 |
| 8K | 1.0045745 | 77.427370 / 78.208448 | 1.0112666 |
| 32K | 1.0037505 | 68.496990 / 68.922444 | 1.0082330 |

All 12 retained request medians favor block32; nine of 12 full
request medians do. The full 8K pooled medians favor the original
despite the positive matched estimate; both statistics are retained.
These are small controlled effects, not a new standard decode headline.
Both experiments above are independent on prefill1024-v2, not stacked
with the qualified query-cache provider.

A separate block32 build-v2 combines the qualified block32 object with
the qualified query-cache-v2 provider, leaving the attention object
unchanged. Both original providers reproduce byte-identically by
relinking before combination. No shader or arithmetic is changed by
this step. Fresh native qualification of the combined provider passes
116,582,784 finite bitwise comparisons, changing inputs, pending1/64,
fallbacks and both dense replay variants. Model qualification remains
pending; component gains are not assumed to stack.

The combined provider's 2K active-path profile verifies 159 block32
and 12 query-cache programs, with neither original dense nor original
attention programs. All six controlled windows and independent
aggregation pass frozen/repeated sequences and zero faults:

| Context | Full matched ratio | Retained query-cache / combined median window tok/s | Retained matched ratio |
|---|---:|---:|---:|
| 2K | 1.0098213 | 78.635026 / 79.433905 | 1.0103477 |
| 8K | 1.0099277 | 79.699724 / 80.512928 | 1.0106959 |
| 32K | 1.0088034 | 70.019176 / 70.313900 | 1.0061122 |

23 of 24 per-request medians favor block32 on top of query caching;
one retained 32K request favors the original.
The combined provider SHA is
`85bac6cad418c6263a08b3b7a0e2a9ab712690a74fd46028e543557b9dccef0a`.
All six subsequent standard runs pass 6,144 frozen output comparisons
including warmups and zero prefill/decode major faults:

| Context | Full median decode tok/s | Retained median decode tok/s | Full TTFT s | Retained TTFT s | Retained E2E output tok/s |
|---|---:|---:|---:|---:|---:|
| 2K | 78.214649 | 77.987318 | 3.915971 | 3.947348 | 35.481636 |
| 8K | 77.806094 | 78.662909 | 16.638274 | 16.654273 | 12.870911 |
| 32K | 65.113411 | 68.786350 | 81.269856 | 81.557079 | 3.002445 |

These are median decode headlines, not window or aggregate rates.
Against the prior query-cache retained standards of
76.821994/77.117301/65.914297 tok/s, the observed separate-run ratios
are 1.0151691/1.0200423/1.0435725. They do not isolate causality;
the controlled matched ratios above are the incremental evidence.
Full and retained 32K results differ noticeably, so neither the
retained increase nor that cross-mode gap is attributed to a specific
cause. The independent standard aggregate pins the provider, original
graph/config and all source evidence. The 100 tok/s target and
whole-model quality qualification remain unmet.

The independent default-off `ORT_WEBGPU_QMOE_PREFILL1280` prototype
targets TTFT with exact 1280-token chunks and their 768-token remainder.
The canonical routed dispatch budgets are 64000 and 38400 workgroups,
both within 65535; 1536-token chunks would exceed that limit and are
not admitted. Original 256/512/1024 paths remain, as do the cooperative
INT4 exclusion and device-limit checks. Attention admission is extended
only for batch-one NVIDIA sequences of exactly 768 or 1280 whose
query dispatch fits the actual device limit. Neither expert nor
attention shaders or floating-point reductions change. The original
attention object is recompiled byte-identically and both build stages
reproduce their parents before replacement. The parent is query-cache
v2, without block32. Native and model qualification are still required;
there is no TTFT claim yet.

The bounded prefill native proof now passes 568,197,120 finite bitwise
comparisons over six original-weight layers, including exact 768/1280
admission, 1279/1281 fallbacks, invalid selectors and pending1/64.
Independent attention qualification passes 338,789,376 finite
CPU-reference comparisons over 30 fixture/reference files, including
both new sequence lengths, existing 256/512/1024 paths, FP16/FP32,
cache outputs and all 12 expected admission rejections. The 2K full
profile passes frozen prefixes/zero faults and verifies 96 exact-1280
gate programs plus 96 remainder-768 programs. This is an instrumented
correctness profile, not a TTFT or decode-speed result. The focused
113-test CPU suite also reproduces the previous prefill1024 standard
qualification exactly without rewriting evidence.

Larger-prefill full/retained screens pass frozen output prefixes and
zero faults at 2K and 8K, but do not establish a timing benefit:

| Context | Full screen TTFT s | Retained screen TTFT s | Qualified parent retained standard TTFT s |
|---|---:|---:|---:|
| 2K | 4.166666 | 4.522048 | 3.950710 |
| 8K | 16.863163 | 16.861349 | 16.807248 |

The short screens and three-request standards are different protocols,
not causal pairs. Even at 8K, where the candidate reduces the number
of prefill chunks, no TTFT improvement is observed. It remains off
and is not escalated to expensive 32K or standard timing.

The next default-off `ORT_WEBGPU_DSA_PAIRED` prototype packs two unchanged logical
64-key partitions into one 64-thread workgroup, one fixed 32-lane
subgroup each. It retains query caching, per-lane dot/update order,
partition output indexing and the original merge. Integer sequence
length/count values are read directly rather than communicated through
cross-subgroup shared variables; there is no floating-point reduction
change. Admission requires the existing FP16 query-cache route,
batch-one/single-token input, even query-head count and actual
64-thread device limits. Original query-cache and fallback shaders
remain unchanged. The CPU suite reproduces both old query-cache
window and six-standard aggregates exactly without writing evidence;
it also checks logical partition/lane coverage and selector restoration.
The isolated provider SHA is
`da0d0c8fe7e1fe420d270121fe3e062ddd001a2794eb4494bf7ae081432b2e60`;
its original combined parent relinks byte-identically. Native pending1/64
qualification passes 4,870,656 finite bitwise comparisons over 27
fixtures, 486 CPU-reference executions and nine fixed-flag capture
executions, including explicit invalid selector rejection and original
FP32/fallback routes. Each pending mode verifies 18 paired and 225
original attention programs, with the original merge. The actual
full-validation 2K model profile passes frozen prefixes/zero faults
and verifies 12 paired plus 159 block32 programs, with neither original
attention nor tile4 dense routes. Controlled 2K windows are the next
performance gate; no performance claim is made for this prototype yet.

The completed paired-attention 2K windows preserve frozen/repeated
tokens and zero faults, but do not establish a performance benefit:

| Validation | Original / paired median window tok/s | Matched ratio |
|---|---:|---:|
| Full | 79.706811 / 79.732952 | 0.9992679 |
| Retained | 80.322555 / 80.138546 | 0.9965685 |

All four retained request medians favor the original. The full pooled
median slightly favors pairing while its matched estimator does not;
both are retained. The selector remains off, without long-context
or standard escalation.

The next default-off `ORT_WEBGPU_HC_POST_NORM_VEC4` candidate targets
the original fused normalization/projection shader instead. Its padded
workgroup storage changes from 11,520 scalar FP16 values to 2,880
four-component FP16 vectors, preserving the same 27,648-byte total
workgroup footprint. Normalization addresses retain the same scalar
component mapping and barriers; the projection loads two aligned
vectors instead of constructing them from eight scalar reads. RMS,
dot/update order, FP16 sequential projection reduction, dispatch,
original shaders and fallback admission remain unchanged. This is
a source-level vector-load hypothesis, not a verified ISA or speed
claim. 107 focused CPU tests pass, including the complete logical
address/vector-lane mapping, selector restoration and previous
query-cache qualification reproduction. The provider
`18adc19b6cff74d31025d78c086ae6e2830ddaec230d01543e1f72e450971a32`
builds with a byte-identical combined-parent relink. Native qualification
passes 607,143,936 finite bitwise comparisons over 36 fixtures and
all four outputs, changing inputs, pending1/64, keyed replay/LRU
eviction, explicit invalid selectors and 24 fallback fixtures.
The actual 2K profile verifies 94 vector programs, 12 query-cache
attention and 159 block32 dense programs with original routes absent.

| Validation | Original / vector median window tok/s | Matched ratio |
|---|---:|---:|
| Full | 79.756491 / 79.803211 | 1.0006345 |
| Retained, fresh attempt 2 | 79.561116 / 79.807445 | 1.0018390 |

These are flat/tiny gains with mixed request medians (two of four
retained medians favor each mode), not a robust promotion. Both
completed screens pass frozen/repeated sequences and zero faults.
The original retained attempt produced four raw requests and ratio
0.9994780, but its process-lifetime guard did not finalize after an
interruption. No owned process/container remained. That attempt is
preserved unchanged and not qualified; the fresh run uses a distinct
`-attempt2` evidence suffix, not overwritten guard/cache/measurement
files. No long-context escalation is justified; the selector stays off.

The next default-off `ORT_WEBGPU_GDN_VALUE8` candidate packs eight
independent recurrent value channels per workgroup instead of four.
Unlike the rejected value2 experiment, it reduces workgroup count.
It preserves the per-channel floating-point/state update order and
original four-channel shader. Admission remains NVIDIA packed-FP16,
batch-one/single-token, dk128/dv128 with checked dispatch limits;
all other shapes use original fallbacks. The parent is the qualified
query-cache/block32 stack. 117 focused CPU tests pass, including
complete key/value state-coordinate ownership, unchanged original
shader, preserved value2 CLI defaults and selector restoration.
The isolated provider and fixed-output capture comparator build
successfully; strict native/profile/2K controlled gates are pending.

Value8 native qualification passes 236,531,952 finite bitwise
comparisons, both recurrent outputs, pending1/64, fixed capture and
original fallbacks. The actual 2K profile verifies 36 value8 programs
and no original recurrent program, while retaining query-cache and
block32 routes. Full 2K windows pass all token/fault gates but regress:
original/candidate median window rates 79.707920/78.921308 tok/s,
matched ratio 0.9916405, all four request medians favoring original.
The retained run fails the zero-major-fault gate (first request:
47 prefill and two decode faults), so it is not qualified and supplies
no performance claim. The clean negative full screen already
justifies leaving the selector off without long-context escalation.

The next default-off `ORT_WEBGPU_HC_POST_NORM_PREDECODE` experiment
caches FP16 decoded projection weights on GPU only when both weights
and scales are constant initializers. It preserves original model
files and normalization/dot/sequential-reduction expressions. Each
eligible kernel retains 6,635,520 extra bytes (about 595 MiB for 94
kernels), with a one-time decode dispatch and explicit initialization
errors. Cache ownership and `call_once` follow the existing constant-B
padding pattern. Cold initialization and extra memory are real costs;
no latency or memory benefit is assumed.

The first build correctly fails the original static-node replay guard:
the cached weight buffer is not a parent input. It is not a numerical
failure or qualified model result. Build-v1 and its failed native
evidence remain untouched. Build-v2 reuses the already tested staged
leaf's explicit-extra-input replay helper, under a private class/header,
with the cached weight tensor declared as an extra shader input.
Binding order, WebGPU buffer provenance, metadata keys, exact captured
bindings and LRU validation remain enforced. The original replay
header and original shader stay unchanged. Separate original/cached
replay instances prevent cross-mode binding reuse; the native verifier
requires 24 original/cached statistics rows, 96 requests per row,
binding hits, more than 64 captured keys and eviction.
Build-v2 passes 607,143,936 finite bitwise comparisons over all four
outputs, replay/LRU checks and original fallbacks. The actual 2K
profile verifies 94 cached-weight programs with query-cache/block32
routes retained. The first full window run fails the fault gate
(48 prefill and two decode faults in its first request); its evidence
is preserved and supplies no qualified timing claim.

A fresh full-validation attempt 2 passes frozen/repeated sequences
and zero faults: original/cached median window rates are
78.892780/78.295737 tok/s, matched ratio 0.9918559503, with all four
request medians favoring original. The exact cache is slower, so it
stays off without retained or long-context escalation.

The next default-off `ORT_WEBGPU_QMOE_PREFILL2048` experiment admits
exactly 2,048 tokens under the original canonical NVIDIA symmetric
block32 FP16 expert guards, with cooperative override excluded.
Its activation dispatch is 51,200 by two workgroups instead of
102,400 by one, using ShaderHelper's existing flattened `global_idx`.
The original activation template, floating-point expressions and
other routes remain unchanged. The replay variant includes the
new selector. Attention admission likewise extends only to batch-one
sequence 2,048 with checked query-head dispatch limits; existing
256/512/1024 admission is preserved.

Both providers build against the qualified query-cache/block32
parent, with a byte-identical parent relink and prior byte-identical
original attention object proof. 113 focused CPU tests pass, including
complete flattened workgroup coverage, native boundary/comparison
budgets, required stacked flags and old qualification reproduction.
Native expert qualification passes 615,383,040 finite bitwise
comparisons, including 2047/2048/2049 boundaries, pending1/64 and
original fallbacks. Attention passes 338,208,768 CPU-reference output
comparisons, legacy admission and all 12 expected rejections. The
actual full 2K profile verifies 96 prefill2048 gate programs.
Full/retained short 2K screens pass original output prefixes and zero
faults, with TTFT 4.351833/4.342591 seconds. These do not establish an
improvement over the qualified parent's 3.915971/3.947348-second
standards; different request counts prevent a causal timing claim.
The selector stays off without long-context or standard escalation.
No new decode or whole-model quality benefit is claimed.

The next default-off `ORT_WEBGPU_DENSE_STATIC_SHAPE` candidate
specializes the existing canonical single-row block32 dense route.
Only five integer dimension uniforms become shader constants:
N, K, activation/weight K strides and blocks per column. All values
are passed from the original computed dimensions and included in
the program cache hint. Floating-point expressions, dispatch,
shared storage, original shader/admission and FP32/multi-token
fallbacks remain unchanged. The new route requires both the existing
block32 selector and its own selector, and has a distinct replay
variant. This targets compiler constant propagation, not a claimed
native matrix instruction or measured speedup.

117 focused CPU tests pass, including exact template restoration,
unchanged original admission/programs, dimension cache keys,
selector restoration/interruption and preserved old comparator
defaults. The provider builds against the qualified parent with a
byte-identical parent relink. Native and FP32 fallback paths each
pass 116,582,784 finite bitwise
comparisons, including changing inputs, invalid selectors and
multi-token fallbacks. The actual 2K profile verifies 159 candidate
programs, no original block32/tile4 programs and retained query-cache
attention. Controlled full/retained 2K matched ratios are
0.9922495/0.9940081; median window rates are respectively
77.264714/76.677746 and 80.202640/79.836615 tok/s
(original/candidate). All four retained request medians favor
original. Both screens pass frozen/repeated sequences and zero
faults. The selector stays off without long-context escalation.

The next default-off `ORT_WEBGPU_QSA_PRUNED_RANK` experiment preserves
the original floating-point score programs and exact integer ranking
order. A private 32-thread/one-subgroup ranking program accumulates
the same greater-score/lower-index comparisons and can reject a
nonzero candidate once its partial rank already reaches top-k.
Candidate zero always completes valid-score counting, preserving
disjoint invalid-slot initialization, NaN-first behavior and tails.
Admission retains the original bounded parallel-rank route and
requires NVIDIA subgroup support with both subgroup bounds equal
to 32. Other hardware and routes retain the original rank program.

Build-v1 is rejected by WGSL uniformity validation before numerical
qualification; its build and failed evidence are preserved. Build-v2
uses an explicitly workgroup-uniform header and rejection decision,
checking the monotonic rank bound after 512 comparisons. It does
not disable diagnostics or validation. 123 focused CPU tests pass,
including ties, NaNs, signed zero, infinity, partial subgroup/chunk
boundaries, candidate-zero handling, selector restoration and
legacy defaults. The provider builds against the qualified parent
with a byte-identical parent relink. The 69-fixture GPU proof passes
2,484 exact reference executions; the actual 2K model profile verifies
12 new rank programs with query-cache/block32 retained. Full/retained
controlled matched ratios are 0.9966247/0.9981599, with median window
rates 79.452414/79.335673 and 77.615646/77.466112 tok/s
(original/candidate). Frozen/repeated sequences and zero faults pass.
The selector remains off without long-context escalation.

The default-off `ORT_WEBGPU_QSA_SUBGROUP_RANK` follow-up retains
the original 64-thread comparison ownership and replaces the six-stage
workgroup reduction with two fixed 32-lane integer subgroup sums,
two shared partials and one workgroup barrier. It does not prune
candidates or require a per-chunk uniform header. NVIDIA subgroup
bounds must both equal 32; other adapters retain the original route.
127 focused CPU tests pass, the provider builds with a byte-identical
parent relink, and 69 native fixtures pass 2,484 exact reference
executions. Actual 2K coverage verifies 12 new rank programs with the
qualified query-cache/block32 stack retained.

Full/retained 2K controlled matched ratios are 0.9955796/0.9985353.
Median window rates are 79.846937/79.769350 and
80.031641/79.601626 tok/s (original/candidate); three of four
request medians in each mode favor original. Frozen/repeated
sequences and zero faults pass. The selector remains off and is not
escalated to longer contexts. These are controlled-window screens,
not a new standard TPS result or a whole-model quality qualification.

An unchanged query-cache/block32 provider full-validation 32K session
profile passes the warmup/measured frozen two-token prefixes and zero
faults. Instrumented GPU category totals are 69.153 s for expert
`MatMulNBits` (44.72%), 38.399 s for indexer scoring (24.83%),
17.407 s for GDN (11.26%) and 16.209 s for dynamic sparse attention
(10.48%). These totals include two requests, profiling overhead and
decode; they are diagnostic, not request TTFT or throughput.

The default-off `ORT_WEBGPU_QSA_PREFILL_SCORE16` experiment extends
the existing 16-candidate wide query-cache scoring shader to canonical
FP16 prefill rather than changing its floating-point shader code.
Admission requires NVIDIA, batch one, 512 or 1024 rows, four
128-dimensional heads, packed width 640, compress ratio four and
token budget 2048. Score stride is bounded to 128..16384; original
workgroup/storage guards and per-dimension dispatch limits remain.
The new route uses a two-dimensional packed-score/row dispatch and
an explicit separate shader-cache hint. The original single-row
decode dispatch and all other fallbacks remain unchanged.

139 focused tests pass. The provider builds against the qualified
parent with a byte-identical relink and unchanged existing kernel
object layouts. 76 native fixtures pass 2,736 exact reference
executions, including newly admitted 512/1024-row cases through
32K past context, masked/rotary cases, row-boundary and FP32
fallbacks, and explicit invalid-selector rejection. The actual full
2K model profile verifies 48 prefill wide-score programs, frozen
two-token prefixes and zero faults. The four-load full-validation
2K ABBA screen reduces median TTFT from 3.912200 to 3.843237 s;
both matched pairs improve, with a geometric matched reduction of
1.7628%. The retained 2K screen does not reproduce the gain:
3.905761 to 3.920688 s, a matched reduction of -0.3620%.
Each screen passes 288 frozen output comparisons, repeated
sequences and zero faults. The full-validation 8K ABBA screen
reduces median TTFT from 16.708461 to 15.672872 s, with both
matched pairs improving and a geometric matched reduction of
6.1981%; its 288 frozen comparisons, repeated sequences and
zero-fault gates pass. The full-validation 32K ABBA screen
reduces median TTFT from 81.571267 to 65.746850 s, with both
matched pairs improving and a geometric matched reduction of
19.3993%; its 288 frozen comparisons, repeated sequences and
zero-fault gates pass. Retained 8K TTFT is 16.718530/15.701413 s
(original/candidate), a matched reduction of 6.0836%; retained
32K is 81.462597/65.727178 s, a matched reduction of 19.3160%.
Both matched pairs improve at each context, with frozen/repeated
sequences and zero faults. All six screens total 1,728 frozen
output comparisons including excluded warmups.
These 36-output screens are not standard decode throughput or
whole-model quality qualification; the selector is not promoted.

A separate standard runner requires all six controlled screens,
actual prefill coverage and native qualification before preserving
the existing 256-output/three-measured-request protocol. Long
contexts must improve in both validation modes; the neutral retained
2K screen is explicitly recorded rather than relabeled a win.
The completed full/retained 2K standards pass frozen outputs and
zero faults: median decode is 77.498160/78.687705 tok/s, mean
TTFT is 3.871673/3.863676 s, and E2E is 35.775576/36.046232
tok/s. All six standards now pass 6,144 frozen output comparisons
including excluded warmups and zero faults:

| Context | Full decode tok/s | Retained decode tok/s | Full TTFT s | Retained TTFT s |
| --- | ---: | ---: | ---: | ---: |
| 2K | 77.498160 | 78.687705 | 3.871673 | 3.863676 |
| 8K | 78.052956 | 78.400867 | 15.681233 | 15.730460 |
| 32K | 68.315920 | 66.698585 | 65.861096 | 65.677081 |

Retained 32K E2E is 3.683619 tok/s versus the prior 3.002445.
The separate-run decode shifts are not isolated causal gains
or regressions. The 100-decode-tok/s goal remains unmet, and
the provider is default-off/unpromoted pending independent
quality review and further performance investigation.
Existing completed
2K evidence was verified without rerunning after correcting a
verification-only assumption: inactive experimental selectors
can be explicitly `"0"` rather than unset.

A separate default-off `ORT_WEBGPU_DENSE_DIRECT_A` experiment is
prepared against the qualified query-cache/block32 parent. For
the existing canonical single-row FP16 block32 route, a private
program replaces shared activation staging with direct vector
loads, removes K-loop barriers and retains one final workgroup
barrier. Shared FP16 partial writes and the original serial
32-lane fold remain unchanged. FP32 accumulation, other input
types and all parent fallbacks retain their original routes;
the selector has explicit validation and a distinct replay key.
26 focused dense/prefill CPU tests pass. The provider builds with
a byte-identical qualified parent relink and unchanged existing
kernel object layouts. Separate native FP16/replay and FP32
accumulation-fallback runs each pass 27 changing-input fixtures
and 116,582,784 finite bitwise comparisons, including multi-row
fallbacks and explicit invalid-selector rejection. The actual
2K model profile verifies 159 direct-A programs, retained
query-cache attention, frozen outputs and zero faults.
The full-validation 2K controlled-window ratio is 0.9728783;
median window rates are 79.781153/77.624561 tok/s
(original/candidate), and all four request medians favor
original. The retained ratio is 0.9682080, with median window
rates 80.411973/78.169914 tok/s; all four request medians also
favor original. Both modes pass frozen/repeated sequences and
zero faults. The candidate remains off and is not escalated
to longer contexts or promoted as a new standard TPS result.

The next default-off `ORT_WEBGPU_DENSE_PRIVATE_PARTIALS`
experiment is prepared against the unchanged qualified parent.
It retains shared activation staging and both existing K-loop
barriers, replacing repeated shared partial reads/writes with
a typed FP16 scalar partial per thread. The canonical workgroup
has 128 threads: 32 K lanes for each of four output columns.
Each K tile still performs the same ordered typed addition;
only the last tile publishes each thread's own partial to
shared memory before the original serial fold.
The canonical FP16/non-FP32-accumulation route, parent
fallbacks, explicit selector validation and a separate replay
key are retained. Build-v1's native checks catch missing
publication from inactive final-K-tile lanes; build-v2 catches
cross-column writes after the ownership fix was incomplete.
Both failed versions and evidence remain preserved. Build-v3
publishes outside the K-tail guard to each thread's unique
`[idy][idx]` slot. Separate FP16/replay and FP32-fallback runs
each pass 27 fixtures and 116,582,784 finite bitwise comparisons.
36 related CPU tests pass, including K-tail publication,
128-thread ownership, window wiring and standard gates.
Actual 2K model coverage verifies 159 private-partial programs.
Full/retained controlled ratios are 1.0028359/1.0003295;
median window rates are 80.444712/80.359059 and
80.828068/80.839004 tok/s (original/candidate). Frozen/repeated
outputs and zero faults pass. This is effectively neutral,
not a new standard TPS gain; the selector remains off without
long-context escalation.

The decode-only private-partial profile attributes 2,698 us
of instrumented GPU time to 94 `HCPostNormProjectionF16`
dispatches (17.32% of 15,575 us total). This is diagnostic,
not uninstrumented decode timing. Prior four-column, vector
packing, staged/predecoded, fused-sum and subgroup/shuffle
experiments are not retried unchanged.

The next default-off `ORT_WEBGPU_HC_POST_NORM_WIDE16`
experiment reuses one unchanged 256-thread normalization
stage across two eight-column projection slots. It halves
canonical dispatch groups from 41 to 21 while preserving
all FP16 dot operations, ordered K-tile additions, the serial
32-lane fold and all four outputs. The private program uses
512 half partials, requiring 28,160 workgroup-storage bytes;
adapters below that limit retain the original fused route.
The selector has explicit validation and a distinct replay
key without changing existing kernel object layouts.
20 focused tests pass. Native validation passes 36 fixtures
and 607,143,936 finite bitwise comparisons, changing CPU
splits, all four outputs, pending windows 1/64 and LRU replay.
The actual full 2K profile verifies 94 wide programs, frozen
outputs and zero faults. Completed full/retained controlled-window
ratios are 0.9462543583/0.9499741595. Window rates fall from
79.848909 to 75.567623 and from 79.819861 to 75.797214 tok/s,
respectively. All four request medians favor the original in
both modes; frozen/repeated outputs and zero faults pass.
The wide tile is rejected and remains off without long-context
escalation.

The improved full 32K prefill profile passes 768 actual large-row
score16 dispatches, frozen prefixes and zero faults. Instrumented
GPU totals across warmup and measured requests, including decode,
fall from 154.629147 to 123.235703 seconds; indexer scoring falls
from approximately 38.399 to 7.014767 seconds. This explains much
of the controlled approximately 19.3% TTFT reduction, not a new
decode TPS gain. Expert-indirect prefill matmul now dominates:
FC1/FC2 total 39.745641/29.421165 seconds across 3,072 calls each.
These rows are token-times-top10 expanded and are not grouped by
expert; grouped matrix multiplication remains unimplemented.

### Short-stride query caching and global HC normalization screens

The default-off `ORT_WEBGPU_QSA_SMALL_QUERY_CACHE` experiment
reuses the existing wide query-cache scorer with four candidates
and 64 threads for single-row score strides 128 through 2047.
Per-candidate 16-lane ownership is unchanged; shared storage is
4,468 bytes. Existing prefill and long-decode code remain unchanged.
The first build duplicated the following class declaration; the
corrected build-v2 retains the failed evidence and passes 69 immutable
fixtures, 2,484 reference executions, explicit invalid-selector
rejection and actual 12-dispatch model profiling. The first collector
incorrectly assumed the short baseline used packed4: it actually uses
the generic scorer below stride 2048. The collector was corrected and
qualified the already completed evidence without rerunning it.

Full/retained 2K matched window ratios are 0.9940405509/1.0020371879.
Median window rates are 80.047533/78.826152 and
80.573215/80.714945 tok/s (original/candidate). Frozen/repeated
IDs and zero faults pass. This is not a consistent gain: keep off
without long escalation or a new standard headline.

The default-off `ORT_WEBGPU_HC_CACHED_NORM` experiment computes
the original normalization once, then loads the existing normalized
output into the original padded, eight-column projection. All FP16
dot and serial fold operations remain unchanged. Build-v1 correctly
failed the frozen static replay invariant: that cache permits only
one command, not two. Build-v2 uses ordinary compound dispatch,
matching the original fallback design, and leaves the frozen replay
header and original fused replay path unchanged. Candidate replay
is explicitly not claimed. Native validation passes 36 fixtures,
607,143,936 finite bitwise comparisons, changing CPU splits and
pending windows 1/64; original single-command replay/LRU stats pass.

The actual model profile proves 94 new projections. Instrumented
GPU time is 1,712 us for projection plus 1,582 us across 95 norm
programs, within 16,314 us total; this is diagnostic, not an isolated
causal comparison. The extra stage does not improve end-to-end decode:
full/retained matched ratios are 0.9816339215/0.9770691709.
Median window rates fall from 79.500827 to 78.121791 and
80.567184 to 78.814504 tok/s. All four request medians favor the
original in both modes. Frozen/repeated IDs and zero faults pass.
Keep this candidate off without long-context escalation.

The default-off `ORT_WEBGPU_HC_QUAD_FOLD` experiment folds four
independent columns per thread using `vec4<f16>`, retaining the same
32 ordered additions per column, all normalization/dot expressions
and one-command replay. This is distinct from the prior shared
activation vec4-storage experiment. Its shared factory extraction
produces byte-identical code for that existing experiment. Native
validation passes 36 fixtures and 607,143,936 finite bitwise
comparisons, changing CPU splits and replay/LRU gates. The actual
model profile proves 94 new programs.

Full/retained matched 2K ratios are 0.9907785619/0.9890966010.
Median window rates are 77.903279/77.126621 and
80.198597/79.296421 tok/s (original/candidate). All four request
medians favor the original in both modes; frozen/repeated IDs and
zero faults pass. Reject this candidate, keep it off, and do not
escalate it to long contexts.

The next default-off `ORT_WEBGPU_DENSE_PAD_A` experiment ports
the shared-memory bank-padding layout idea to canonical dense
block32 activations: groups of eight vec4 values have physical
pitch nine. Every activation store and read uses the same padded
address; original floating operations, staged loads, barriers and
serial partial fold remain unchanged. The conventional four-byte
bank-address model reduces repeated starting-bank aliases from
16 to two; this is a layout hypothesis, not a measured speed claim.
Parent c1 retains the prefill-score16 work. Native FP16/replay and
FP32-fallback runs each pass 27 fixtures and 116,582,784 finite
bitwise comparisons. Actual model profiling proves 159 padded
programs, frozen outputs and zero faults. First full/retained 2K
matched ratios are 1.0015878014/1.0058423949. A fresh repeat gives
1.0015300656/1.0002525553: the retained signal nearly disappears,
and request medians are mixed. All four screens pass frozen/repeated
IDs and zero faults. These tiny signals do not establish a robust
decode win. Keep the selector off without long-context or standard
escalation; no headline improvement is claimed.

The next default-off `ORT_WEBGPU_HC_NARROW4` experiment uses
128 physical threads and four output columns (81 workgroups).
Unlike the earlier four-column experiment, it does not retain
256 physical normalization threads with idle projection warps.
Each thread computes two independent original 256-lane RMS
partials, retaining each partial's ordered FP32 additions and
the original 128-to-one reduction tree. All FP16 projection dot
and serial-fold expressions remain unchanged, with one-command
variant-keyed replay. This is an occupancy hypothesis, not a
measured speed claim. Native qualification passes 36 fixtures and
607,143,936 finite bitwise comparisons over all four outputs, with
pending1/64 and replay/LRU checks. Actual profiling proves 94 new
programs. Full/retained matched 2K ratios are 0.9441347128/0.9470033654;
all request medians favor the original. Frozen/repeated IDs and
zero faults pass, but this narrower route is slower. Keep it off
without long-context escalation.

The default-off `ORT_WEBGPU_QSA_HIST_RANK` experiment leaves
the 2K route unchanged and first targets score strides at least
2,048 (8K/32K contexts). A 256-thread histogram computes a
conservative top-k bucket bound. Only candidates strictly below
the bound skip the original quadratic rank loop; candidate zero
still counts all valid scores, and every workgroup reaches the
original barriers. The original float scores, rank comparisons,
tie ordering and NaN/tail rules are unchanged. New floating-point
operations calculate only the bound, not scores or output ranks.
Negative/infinite maxima and zero maxima disable pruning.
Native references and actual 8K dispatches are prerequisites
for controlled 8K/32K timing; long-only standard qualification
is explicitly unavailable until the screens qualify.

Histogram-rank v1 passes all 69 immutable fixtures/2,484 reference
executions and proves 12 histograms plus 12 bounded-rank programs
in the actual 8K model. Full/retained matched 8K ratios are
0.9844956548/0.9806678199; every request median favors original.
Frozen/repeated IDs and zero faults pass, but v1 stays off and
is not escalated.

V2 targets the remaining workgroup overhead, not another run of
unchanged v1: `workgroupUniformLoad` synchronizes a singleton
pruning decision so a discarded workgroup can leave before the
seven original rank barriers. Admitted candidates still execute
the unchanged rank/reduction logic; candidate zero never exits
early. Admission starts at stride 8,192, leaving 2K/8K routes
unchanged and targeting 32K. V2 passes all 69 immutable native
fixtures/2,484 reference executions and proves 12 histogram plus
12 uniform-rank dispatches in the actual 32K model. Full/retained
matched window ratios are 1.0073985480/1.0047153012; all eight
request medians favor the candidate. Window medians are
70.0841 -> 70.4130 and 70.4804 -> 70.7047 tok/s respectively.
Frozen/repeated IDs and zero faults pass. This is a small initial
controlled signal, not a new standard headline or quality claim.
V2 stays off/unpromoted pending repetition and proper short-context
and standard qualification.

Fresh full/retained 32K attempt-2 controls preserve all frozen/
repeated IDs and zero faults. Matched ratios are
1.0051543188/1.0033119005, with window medians
70.2675 -> 70.5943 and 70.2268 -> 70.6775 tok/s.
All four full request medians and three of four retained medians
favor the candidate. The repeated average signal is small;
it does not remove the short-context/standard qualification
requirements or establish a new headline. Retained retries now
require the full-validation proof from the matching fresh attempt.

Dedicated v2 qualification now admits 2K/8K regression controls
without weakening the generic long-only standard blocker. It first
requires both independent full/retained 32K repetitions to pass IDs,
faults, source hashes, positive matched ratios and at least three
positive request medians. Short profiles must prove zero candidate
dispatches and 12 original rank dispatches. Short controls use a
1% regression-screen bound (not a claim of statistical equivalence);
all four short full/retained controls and both short profiles are
prerequisites for standard 256-output benchmarks. These new
qualification runs are pending; the provider remains unpromoted.

The dedicated 2K/8K profiles now pass with zero candidate and 12
original rank dispatches. Full/retained short matched ratios are
1.0005051797/1.0002063193 (2K) and
0.9977584324/1.0010028177 (8K), within the regression screen.
All frozen/repeated IDs and zero-fault gates pass. Because these
routes are inactive, the small timing differences are noise, not
short-context improvements. Six full/retained standard benchmarks
through 32K are now queued serially after the GDN controls.

All six standards subsequently pass their per-run gates. The aggregate
initially fails because its reused verifier defaults to chunk 256
instead of the actual 1024. A histogram-specific verifier now explicitly
checks the selector and chunk 1024, preserving the historical shared
default. Reverification of the existing immutable runs passes all
6,144 frozen comparisons and zero faults; no GPU reruns were needed.

| Context | Full median decode tok/s | Retained median decode tok/s | Full mean TTFT s | Retained mean TTFT s |
| --- | ---: | ---: | ---: | ---: |
| 2K | 77.247140 | 77.908588 | 3.864213 | 3.866091 |
| 8K | 77.657346 | 78.261133 | 15.710668 | 15.693557 |
| 32K | 67.349535 | 69.218384 | 65.636274 | 65.639613 |

The retained 32K standard differs from the parent's 66.698585 tok/s,
but separate-run differences are observational, not isolated causal
gains. Repeated matched ranking gains remain approximately 0.33%/
0.47% retained, not 3.78%. Original baseline/model files are unchanged;
the regression-qualified candidate remains unpromoted and does not
reach 100 tok/s or constitute whole-model quality qualification.

The default-off `ORT_WEBGPU_QMOE_GROUPED_PREFILL` experiment
targets expert-indirect prefill locality, not decode arithmetic.
It admits only the existing canonical extended INT4 512/1024-row
routes with the cooperative override disabled. Integer count,
exclusive-offset and permutation programs group routed rows by
expert. Hidden inputs are gathered into that order; the original
FC1/activation/FC2 programs are reused without arithmetic changes.
An exact FP16 scatter restores FC2 rows before the original final
mix, preserving token/rank accumulation order. Within-expert
permutation order need not be deterministic because every row is
restored to its original location.

Both selector parsing sites reject invalid values, and the replay
key includes the selector. The new multi-command prefill route
does not claim static replay. Native qualification requires all
six immutable QMoE fixtures, retained chunk-16 bitwise references,
pending windows 1/64 and exact dispatch coverage for all five new
programs. Actual-model profiling must prove grouping is active
before complementary 0/1/1/0 TTFT screens, with full/retained
validation and frozen IDs through 2K/8K/32K. The isolated leaf build
passes with byte-identical parent relink. Model qualification
is pending; no TTFT or decode gain is claimed.

Grouped-prefill native qualification now passes all six immutable
fixtures and 300,810,240 finite bitwise comparisons, including
pending 1/64, fallback rows and invalid-selector rejection. Each
fixture proves 16 dispatches of every new grouping program.
The first model runner exposed a selector-label shadowing bug;
that pipeline was stopped and its old artifacts retained without
qualification. The corrected runner has an exact written-report
regression test for both score16 and grouping, explicit selector
gates and fresh attempt suffixes. Attempt 1 proves 192 dispatches
per grouping program in full 2K profiling, with frozen IDs and
zero faults. Corrected full/retained 2K TTFT controls preserve
all frozen/repeated IDs and zero faults, but do not qualify a win:
full matched ratio 1.0009212311 has mixed pairs; retained ratio
0.9964177127 has both pairs slower. Median TTFT is
3.882303 -> 3.878717 s (full) and 3.864573 -> 3.878466 s
(retained). Grouping v1 stays off/rejected without long-context
or standard escalation.

The separate default-off `ORT_WEBGPU_QMOE_FC1_WARP_FOLD`
candidate preserves the original single-row fused FC1 dots,
shared FP16 partials, 32-step ordered folds and SwiGLU.
It moves the eight final column folds from local IDs 0..7
of one warp to local IDs 0/32/64/96, each folding two columns.
Unlike the previously examined HC fold, these original FC1
readers really do occupy the same warp. This is an ownership
experiment, not another FC2-padding test or an arithmetic/tree
change. Workgroup size 128, all barriers and the original
admission/fallback rules remain unchanged. Both selector sites
validate 0/1, and the replay key includes the selector.
The isolated build passes with byte-identical parent relink.
Native 23-fixture changing-input/route qualification passes
9,748,480 finite bitwise comparisons, pending 1/64 and replay
variant keys/LRU eviction. Actual full 2K profiling proves 48
new FC1 dispatches with frozen IDs and zero faults.
Full/retained 2K controlled windows preserve frozen/repeated IDs
and zero faults, but matched ratios 0.9959422472/0.9947293008
favor the original in all eight request medians. V1 stays off/
rejected without long-context or standard escalation.

V2 keeps the same FC1 workgroup and admission but uses one
`vec2<f16>` accumulator for two adjacent independent column
folds at each warp leader. Each component still receives the
original 32 additions in order; there is no cross-column
reduction. Both components are stored into the original shared
FP16 slots before unchanged SwiGLU reads. This separately tests
packed FP16 arithmetic rather than the two scalar folds of v1.
The default v1 transform remains byte-identical to its built
source. Initial packed-fold build v2 failed because a program-name
replacement changed offsets before the source splice. The fix
applies name replacement after splicing, adds a stream-syntax
regression test, and preserves the failed build/log. Fresh v3
build passes with byte-identical parent relink; native 23-fixture
bitwise/replay proof passes 9,748,480 comparisons. Actual-model
full 2K profiling proves 48 packed-fold dispatches and preserves
frozen IDs/zero faults. Full/retained matched decode-window ratios
are 0.9978975512/0.9965364359; the full requests split two/two
and all retained request medians favor original. The packed-fold
candidate stays off/rejected without long or standard escalation.

The default-off `ORT_WEBGPU_QMOE_FC2_PAIRED_DOT` follow-up
keeps two output columns and 320 threads/1,280 workgroups.
It computes the original two independent four-step FP16 dot
sums together, sharing each staged activation read and using
a `vec2<f16>` accumulator. It retains original weight/scaling
expressions, per-column shared FP16 updates, all three barriers,
the ordered 32-lane folds and the rank-order final mix. This is
not the rejected four-column geometry or activation-padding
experiment. The old two-loop shader remains the default-off
fallback. Both selector sites/replay key, immutable native
fixture runner and actual-model window selectors are wired.
Source/ownership/window regression tests and the isolated build pass
with byte-identical parent relink. Native 23-fixture bitwise/replay
qualification passes 9,748,480 comparisons. Actual full 2K profiling
proves 48 paired FC2 dispatches. Full/retained controlled window
ratios are 0.9973416683/0.9985993343: all four request medians favor
the original in each mode. Frozen/repeated IDs and zero faults pass.
Rejected/default-off; no expensive long-context escalation.

The same actual decode profile has 36 `GatedDeltaNet` dispatches
and zero `GatedDeltaNetParams` dispatches. Although the source can
pack dynamic parameters when storage bindings require it, eliminating
that preparation dispatch cannot benefit this measured decode path:
it is already absent. No constant-packing experiment was implemented.

### Exact GDN hybrid tree experiment (build/native qualification pending)

Default-off `ORT_WEBGPU_GDN_TREE32` instead targets barriers in the
existing recurrent shader. On canonical single-token packed FP16
128-wide decode with fixed 32-lane NVIDIA subgroups, the original
128-lane FP32 norm/dot trees retain their first 64/32 shared stages.
Only the final 16/8/4/2/1 stages move into subgroup zero, followed by
original-root shared stores and a workgroup barrier. Four independent
value channels, the full addition association, normalization,
state update, gate/beta arithmetic and all output coordinates remain.
Other shapes, dtypes and devices retain the original program.

The private shader uses the repository's static C++ literal template
generator; original templates, kernel object layouts and frozen
model/config/weights are unchanged. Expression-tree/FP32 tests,
template generation, comparison metadata and interruption/flag
restoration pass with the focused 136-test suite. Isolated C++
build passes with byte-identical parent relink and original GDN object
reproduction. Native 36-fixture qualification passes 236,531,952
finite bitwise comparisons, output/final-state checks, pending 1/64,
original fallbacks, invalid selector and fixed-I/O capture.
Actual full 2K profiling proves 36 private dispatches and preserved
frozen prefixes/zero faults. Full/retained 2K controlled ratios are
1.0039202816/1.0048692671, with all eight request medians favoring
the candidate. Window medians are 79.741775 -> 80.045473 and
80.677825 -> 80.734281 tok/s. Frozen/repeated IDs and zero faults
pass. This is a small controlled signal, not new standard headline
performance. Full/retained 8K/32K validation is now queued serially.

A separate norm-only `ORT_WEBGPU_GDN_NORM32` follow-up is prepared.
It retains the entire original dot/retrieval tree and changes only
the final five norm-tree stages, with admission additionally requiring
Q/K normalization. This isolates the two effects: instrumented native
medians for both-tree mode are 18 -> 23 us without normalization and
20 -> 18 us with normalization. Those fixture timings motivate the
follow-up but do not establish model gains or causal attribution.
Source association/original-dot preservation, default tree32
byte-identical reproduction, metadata and interruption tests pass
in the expanded 140-test suite. Norm-only build/native/model work is
pending; original/default paths and frozen model files remain unchanged.

Norm-only build-v1 subsequently passes exact parent/object reproduction.
The first native report preserves exact outputs and correct 9-private/
639-original dispatch counts but incorrectly labels two optimized
fixtures instead of one. Model preflight rejects that metadata before
GPU/model execution. The report writer now derives eligibility from
the selected fixtures. Old evidence is preserved unused; a fresh
attempt-2 proof passes 236,531,952 comparisons, one optimized fixture,
both pending windows and fixed-I/O capture. Model preflight explicitly
uses the fresh proof, and the actual full 2K profile proves 36 norm-only
dispatches. Full/retained 2K timing is active. The expanded 143-test
suite covers chunk/selector verification, fresh native metadata and
actual model preflight without weakening the existing gates.

The norm-only candidate fails the controlled 2K performance screen:
full/retained matched ratios 0.9883196368/0.9927336861,
or -1.1680%/-0.7266%. All four full request medians and three of
four retained request medians favor the original. Frozen/repeated
IDs and zero faults pass. Rejected/default-off without long-context
escalation; the earlier native fixture timing was not a reliable
model-performance predictor.

The full-tree candidate instead remains positive through 32K:

| Context | Full matched gain | Retained matched gain | Retained original/candidate window tok/s |
| --- | ---: | ---: | ---: |
| 2K | +0.3920% | +0.4869% | 80.677825 / 80.734281 |
| 8K | +0.1855% | +0.3456% | 80.281929 / 80.619834 |
| 32K | +0.5353% | +0.5957% | 70.551267 / 70.919783 |

All six cases pass IDs/fault gates; 22 of 24 request medians favor
the candidate, with one exception each in retained 8K and full 32K.
These are complementary controlled windows, not standard headline
TPS or proof of whole-model quality. All-context full/retained
repetition is now running on GPU7 with fresh attempt-2 artifacts.

The dedicated GDN qualifier rechecks both attempts, native/source/
capture hashes, original sequences and actual profile dispatches.
Standard escalation requires positive matched ratios and at least
three positive request medians in every case of both repetitions.
Its standard verifier explicitly preserves chunk 1024 and the
GDN selector; the original shared chunk-256 default is unchanged.
The 147-test suite covers actual initial controls, output shapes,
strict gain thresholds, malformed/nonfinite data and verifier wiring.
Repeat aggregation completed successfully and persisted
`gdn-tree32-v1-windows-contexts-20261031.qualification.json`.
The fresh full/retained matched gains are +0.2839%/+0.2838% at
2K, +0.3565%/+0.3048% at 8K and +0.3940%/+0.4252% at 32K.
Twenty-three of 24 fresh request medians favor the candidate;
both repetitions meet every escalation gate. The aggregate
revalidates 12,288 frozen-prefix comparisons, repeated IDs,
zero faults and immutable build/native/source/profile evidence.
Six serialized standard runs are next, including TTFT through
32K. No standard result or 100 tok/s claim is implied.

All six GDN standards and the final aggregate now pass 6,144
frozen comparisons and zero faults:

| Context | Full median decode tok/s | Retained median decode tok/s | Full mean TTFT s | Retained mean TTFT s |
| --- | ---: | ---: | ---: | ---: |
| 2K | 76.113899 | 78.758290 | 3.870838 | 3.866660 |
| 8K | 78.108189 | 78.300308 | 15.700468 | 15.693262 |
| 32K | 66.795322 | 68.895069 | 65.673104 | 65.893358 |

The retained 32K parent observation was 66.698585 tok/s, but its
separate-run difference is not an isolated causal +3.29% gain.
The repeated matched signal remains +0.43% at 32K. TTFT is
essentially unchanged, and 100 tok/s is still unmet at every
context. This candidate is regression-qualified, default-off
and not whole-model quality qualified.

The next isolated stack keeps the histogram-v2 parent and changes
only its GDN object to the already-qualified full-tree object.
Link inputs must differ exactly in the disjoint GDN/ranking
objects, with a shared c1 ancestor and byte-identical parent
relink. Fresh native/capture and model comparisons use
histogram-v2 as the original; standalone percentages are not
added. Profiles check both GDN and histogram programs, including
the original short-context ranking fallback. Stack performance
is not yet measured.

The isolated stack builds successfully with provider SHA-256
`ecd79fa5fe60420bdb1ea801d5d3d984648118d3b0939fd30d57f0bdb7e73800`
and histogram parent
`b071a3f62c45f591f0251fb0c0771db866e1d81f0a607dfca34f8b9e61812e36`.
The parent relink is byte-identical; GDN object and comparator
bytes are reused unchanged, and a fresh fixed-I/O capture
comparator is built. All 150 focused tests pass, including
rejection of unqualified extra objects, missing/reordered link
inputs, mixed ancestors and incorrect active/fallback dispatch
counts. Fresh native and 2K model admission are next.

The shared GDN aggregate now also admits only this explicitly
named stack with histogram-v2 as its parent. It additionally
requires the actual 32K histogram profile and both selectors
in every model proof and standard verifier, while preserving
the original GDN aggregate shape and parent defaults. All 152
focused tests and Python compilation pass. The stacked native/
capture, 2K profile and full/retained controlled pipeline is
running on GPU7. Longer model escalation waits for the fresh
2K performance screen.

Stack native validation passes 236,531,952 finite bitwise comparisons
and fixed-I/O capture. Its actual 2K profile has 36 tree programs,
zero histogram programs and the original short-ranking fallback.
Initial full/retained matched gains are +0.3736%/+0.3557%, but
only two of four full request medians favor the candidate
(retained: three of four). A fresh diagnostic 2K repetition
returns +0.6011%/+0.4130% with all eight request medians positive.
All output-ID/fault gates pass. The initial full case still
fails the established three-of-four support gate: hold/default-off,
no 8K/32K or standard escalation, and no pooled-median gain claim.

### Persistent dense weight predecode experiment

A larger, separate candidate caches FP16-expanded weights for
the nine existing canonical dense shapes, not the previously
rejected HC post-norm cache. The observed 159 dense nodes would
require approximately 5.194 GiB of additional GPU storage.
This exchanges extra bandwidth/storage for removing repeated
INT4 unpacking and scale multiplication; speed is a hypothesis,
not a result.

`ORT_WEBGPU_DENSE_PREDECODE` remains default-off. A private kernel
owns each decoded tensor and initializes it once, admitting only
constant weights/scales on the existing single-token NVIDIA FP16
block32 route. The original kernel class, original compute body,
dot order, activation staging, partial updates and final serial
fold remain unchanged. Unsupported shapes, dynamic weights,
multi-token inputs and explicit FP32 accumulation keep original
behavior. Owned decoded buffers are explicit extra replay inputs,
not raw pointers in a global cache.

The first build is preserved unqualified. A fresh v2 adds the
explicit FP32-fallback comparator and builds successfully with
byte-identical c1 parent relink:
`30646cacd87f06245c865d1e9862e32a954ab0b47fde86804204e307c356614f`.
All 158 focused tests and Python compilation pass. Native
changing-input/replay and FP32-fallback validation are running
on GPU7. The replay gate requires nine original and nine private
instances with real binding reuse, rather than accepting absent
candidate coverage.

The model runner additionally requires both native proofs and
their source/build hashes. Its 2K profile must show 159
`MatMulNBitsPredecodedF16` programs, zero original block32 programs
and zero repeated weight-decode dispatches after warmup. Full/
retained windows must precede any long-context escalation.
No native/model correctness qualification or speed claim has
yet been established for this candidate.

Both native passes have now completed successfully: 27 fixtures
and 116,582,784 finite bitwise comparisons each, including the
original FP32 fallback, invalid selector rejection and multi-token
fallback. The default-precision pass proves real reuse in all
nine private and nine original replay instances. Native success
does not yet establish full-model correctness or performance.

The actual-model 2K profile and full/retained controlled windows
are running on GPU7. All 159 focused tests pass, including a
functional preflight using the fresh native/FP32 evidence. This
also handles the historical native helper's repository-relative
`build-manifest.json` source-hash key explicitly, without changing
the existing helper's report shape or weakening hash checks.

The actual 2K dense-cache profile passes the 159-private/
zero-original/zero-repeated-decode gates, with frozen IDs and
zero faults. Performance is decisively worse:
full/retained matched ratios 0.8866092133/0.8852973934
(-11.3391%/-11.4703%). All eight request medians favor the
original; pooled retained windows are 80.271102 versus
71.138493 tok/s. Rejected/default-off without long-context
or standard escalation. The expanded-weight bandwidth/storage
tradeoff did not help this measured decode path.

### Native FP16 tensor-matrix dot4 compatibility probe

The next bounded investigation targets existing FP16 subgroup-matrix
primitives, not Dawn FP8 or altered model precision. The deployed
prefill path already uses subgroup matrices, but decode's original
four-product dot/FP16-rounding order cannot simply be replaced by
a full-K matrix accumulation.

A private probe admits only isolated FP16 K2560/N2560 M1 fixtures
and requires the native FP16-result 16x16x16 configuration. Three
fixtures have four potentially nonzero INT4 weights at K offsets
0, 4 and 12; all other quantized weights are the implicit zero.
This compares the original decode dot arithmetic with native
matrix arithmetic over changing inputs, while leaving the supplied
model untouched. An arithmetic mismatch is an explicit rejected
result, not a success-shaped fallback or a quality qualification.
Passing sparse fixtures would still not qualify general GEMV.

The probe builds with byte-identical c1 parent relink and provider
SHA-256
`5cad3751c62bd1f70a446097bb0cb7e767327bf541ad64c58d46328a9d07caf7`.
All 161 focused tests and Python compilation pass, including checks
that fixture weights outside the four-product group are zero and
that original routes remain intact. GPU7 native compatibility
testing is running. This is not a performance benchmark and implies
no decode TPS gain.

The FP16-result native matrix probe rejects immediately on
offset 0, round 0, output element 2: original half bits 5774
versus matrix half bits 5773. The guard correctly records a
failed numerical comparison. The aggregate initially confused
that terminal failure with a guard error; its classifier now
distinguishes numerical rejection from infrastructure/partial
failure and revalidates the original immutable run without
rerunning the GPU test.

A fresh FP16-input/FP32-fragment-result variant is testing the
same FP16 output bits. This changes only the isolated matrix
fragment, not the supplied model or original scalar baseline.
Its build passes byte-identical c1 parent relink:
`1044604c61b4339690757ad125afda466ec12e23752c4793b5601675901131d4`.
All 162 focused tests pass. The FP32-result GPU7 probe also rejects
at exactly the same element and bits, 5774 versus 5773. Neither
variant is a qualified decoder or a throughput result.

A host diagnostic of this first mismatch finds that separately
rounding the four products to FP16 before their pairwise additions
produces 5774, whereas summing their FP32 products produces 5773.
This is only a hypothesis for the original dot's rounding, not
general compatibility evidence. A distinct diagonal-input matrix
probe will materialize each independent product in FP16 shared
memory before reconstructing the four-product dot. It remains
restricted to the three sparse fixtures, not a general MatMulNBits
implementation or a model-performance candidate.

The diagonal-product build-v3 was retained after a WGSL validation
error: this Dawn uses the `row_major`/`col_major` template enums,
not boolean majorness. Corrected build-v4 passes C++ build and
byte-identical c1 relink, but its actual native comparison rejects
offset 0, round 0, element 4: 36084 versus 36083. Materializing
products alone therefore does not reproduce the original dot.
All three matrix arithmetic substitutions remain rejected/off.

The next exact prefill experiment differs from the previously
neutral/slower expert sorting alone. Its private FP16 kernel
processes four routed rows together, reusing each unpacked weight
for consecutive equal expert IDs. Mixed-expert tile boundaries
reload the correct weights. It retains each row's original
eight-column tile, 32 K lanes, vectorized 1024-element K staging, paired
four-product dots, FP16 partial updates and serial 32-lane fold.
Only canonical 512/1024-token expanded expert projections are
admitted; decode, FP32 accumulation and other shapes stay original.
It inherits the existing default-off grouped-prefill selector
and its replay key. Native bitwise qualification is pending;
no throughput or TTFT improvement is claimed.

Row-tiling build-v1 fails the native 512-token gate and is
retained unqualified. Investigation finds that profile input
shapes describe vectorized buffers: FC1's displayed width 640
means K2560 FP16 elements, and original A/B use four-component
loads. The corrected build-v2 preserves each lane's original
32-element block, four ordered dot-pair additions, then the
typed partial update. FC1 K2560/N1280 and FC2 K640/N2560 are
admitted separately. It does not substitute a 256-element K tile.

Build-v2 passes byte-identical grouped-parent relink:
`0955e1d6cc30cac13044450f692100ffa5f33361aaef229e25ff0efa20a8c8d4`.
Native GPU7 qualification passes six frozen fixtures and
300,810,240 finite bitwise comparisons against retained chunk16,
changing inputs/routes, pending windows 1/64, fallback sizes and
invalid selector rejection. Each fixture proves 32 actual new
projection dispatches alongside the original five grouping programs.
The actual-model collector now verifies both vectorized projection
shapes and requires full/retained positive 2K TTFT controls before
long-context escalation. All 27 focused tests pass.
The actual full 2K model profile now passes frozen prefixes and
zero faults, proving 192 instances of each grouping program and
384 new projections, evenly split between the correct FC1 and
FC2 shapes. Complementary 0/1/1/0 controlled 2K TTFT screens
pass both matched pairs in each validation mode:

| Validation | Original median TTFT s | Candidate median TTFT s | Matched TTFT reduction |
| --- | ---: | ---: | ---: |
| Full | 3.865837 | 3.492322 | 9.6619% |
| Retained | 3.867635 | 3.491932 | 9.7140% |
| Full 8K | 15.742978 | 14.189621 | 9.8662% |
| Retained 8K | 15.704442 | 14.192541 | 9.6271% |
| Full 32K | 65.701004 | 59.699609 | 9.1345% |
| Retained 32K | 65.830744 | 59.694025 | 9.3223% |

Both modes pass frozen/repeated prefixes and zero faults. These
are controlled short-output TTFT screens, not standard decode
TPS or whole-model quality qualification. The candidate remains
default-off/unpromoted. Equivalent 8K and 32K profiles and controls
now pass, including both 8K escalation gates before 32K.
All twelve matched pairs favor the candidate, with 1,728 frozen
prefix comparisons and zero faults. Profile coverage is 1,536
new projections at 8K and 6,144 at 32K. A dedicated aggregate
preserves these as controlled TTFT results, not standard decode
qualification. The 100 decode-TPS goal is unmet.

While the long-context GPU timings run, a separate default-off
FC2 decode fold was prepared without overlapping CPU compilation.
It preserves the original dot prefix and shared FP16 partial
materialization. Each NVIDIA 32-lane subgroup packs two existing
half partials into a u32, broadcasts source lanes 0 through 31
in original order, and independently folds the two columns in
FP16. Integer broadcasts avoid requiring FP16 subgroup builtins.
Bias, shared final stores and the original routed final mix stay
ordered. The selector is validated and included in the replay
key; unsupported adapters keep the original fold. After the
timing batch ended, its build passed byte-identical row-tiling
parent relink:
`bcd5c7dfcd577de4b0e0919d8eb155f1b16be9deec6159056d82e4c5fe37711e`.
Native GPU7 checks pass 23 fixtures and 9,748,480 finite bitwise
comparisons, changing routes, optional-bias/layout fallbacks,
pending windows 1/64 and replay/LRU keys. The actual full 2K decode
profile passes frozen prefixes and zero faults, with 48 subgroup-fold
dispatches and no original FC2 final-mix dispatches. All 129 focused
integration/regression tests pass, and the row4 controlled-context
aggregate verifies the immutable all-context TTFT evidence.
Both control and candidate retain the grouped-row prefill selector
so the earlier TTFT improvement is not lost.

Full/retained controlled 2K decode ratios are 0.9977138312 and
0.9973222795. Median window rates are 76.489506/76.337529 and
80.803018/80.655567 tok/s (original/candidate), respectively.
All eight request medians favor the original, while frozen/repeated
IDs and zero-fault gates pass. Reject this fold, keep its selector
off, and do not escalate to long-context or standard decode timing.
The shared-weight prefill improvement remains independently qualified.

Natural row4 standards now use 256 outputs, three measured requests
and one excluded warmup, with the rejected FC2 fold explicitly off.
The collector retains the original c1 command protocol, uses distinct
benchmark and both replay-statistics paths, checks the immutable
all-context TTFT/native/profile closure, and verifies all frozen IDs
and zero faults. Full 2K precedes retained 2K; both precede 8K, and
both 8K modes precede 32K. Preparation and aggregate tests pass.
These standalone standards are not isolated causal decode comparisons.

The next default-off `ORT_WEBGPU_QMOE_FC2_COMPACT20` experiment packs
the original ten experts' twenty active K-block lanes into 256 physical
threads instead of 320. It retains all 200 original rank/lane tasks,
their complete 32-element dot blocks, shared FP16 updates, the ordered
32-lane folds, optional bias and routed mixing. Inactive threads do
not read expert indices; all threads stage activations and execute
all three barriers. The original shader remains the selector-off
fallback, and existing admission limits are unchanged. This is an
occupancy/layout hypothesis, not a measured gain or another reduction
substitution. Eleven focused tests pass. CPU compilation is deferred
until the active natural-standard timing batch ends.

All six natural row4 standards and their immutable aggregate now pass
6,144 frozen comparisons and zero faults:

| Context | Full median decode tok/s | Retained median decode tok/s | Full mean TTFT s | Retained mean TTFT s |
| --- | ---: | ---: | ---: | ---: |
| 2K | 77.677232 | 75.956958 | 3.488250 | 3.488692 |
| 8K | 75.070543 | 78.352449 | 14.180252 | 14.241192 |
| 32K | 68.338374 | 69.249888 | 59.911473 | 59.672971 |

These separate runs do not isolate causal decode gains. The earlier
controlled TTFT reductions remain the matched performance evidence.
Whole-model quality and the over-100 decode-TPS goal remain unmet.

Compact FC2 subsequently passes byte-identical row4 parent relink,
23 native fixtures and 9,748,480 finite bitwise comparisons, replay
and optional-bias/layout fallback checks, plus actual 48-dispatch
full 2K model profiling. All 135 integration tests pass. Full/retained
2K matched ratios are 0.9961241602/0.9962033046; median window rates
are 79.622222/79.384265 and 80.395599/80.151072 tok/s
(original/candidate). All eight request medians favor original,
with frozen/repeated IDs and zero faults. Reject and keep off,
without repeats or long-context escalation.

The next default-off `ORT_WEBGPU_DENSE_QKV_WIDE16` targets only
single-row FP16 K2560/N16384 projections, not all dense shapes.
It uses 512 threads for sixteen concurrent columns and preserves
each original 32-lane K partition, dot expression, per-tile FP16
partial store and ordered final fold. The block32 template is
unchanged except its private function name; device limits and
3,072 workgroup-storage bytes are required. Other shapes/devices,
FP32 accumulation and row4 prefill keep their original paths.
This differs from the previous general eight-column bitcast route.

Its build passes byte-identical row4 parent relink:
`610f3d104406d6aa4218bf41c759f7f6b5a176bf9f52ea383d451bf84b5d0fbb`.
Native normal and explicit FP32-fallback runs each pass 27 fixtures
and 116,582,784 finite bitwise comparisons. Replay proves one new
variant only for the target shape, with all eight other shape keys
unchanged. All 141 integration/regression tests pass. Actual full
2K decode profiling passes frozen prefixes/zero faults and proves
36 new projections plus 123 retained block32 projections.
Controlled 2K windows are running; no TPS gain is claimed yet.
Any long escalation requires positive, at-least-three-of-four
request support in both validation modes and a fresh 2K repetition;
32K additionally requires both positive 8K controls.

Wide16 controlled 2K timing completes with full/retained matched
ratios 0.9977417705/0.9964191995. Median window rates are
79.646537/79.496499 and 80.164516/79.890085 tok/s
(original/candidate). All eight request medians favor original;
frozen/repeated IDs and zero faults pass. Reject and keep off
without repeats, long-context or standard escalation.

The next default-off `ORT_WEBGPU_QMOE_FC1_EXPERTS10` instead
shares one original activation tile across all ten selected experts.
One 320-thread workgroup computes one interleaved gate/up channel
for each expert, replacing 1,600 four-channel/single-expert groups
with 640 groups. Total dot work, 32-lane K partitions, per-tile
shared FP16 updates, serial folds, bias and FP32 SwiGLU expressions
are preserved. Activation staging drops between groups; its actual
performance is not yet measured. Original admission is unchanged,
with extra 320-thread/3,328-byte limits only for the new route.
The selector is explicitly validated and replay-keyed. The row4
prefill parent is retained; no arithmetic/tree substitution is used.

Ten-expert FC1 passes byte-identical parent relink, 23 native fixtures
and 9,748,480 finite bitwise comparisons, 147 integration tests, and
actual 48-dispatch full 2K profiling. Full/retained matched ratios
are 1.0017251999/0.9990020104, with median window rates
79.582921/79.638270 and 77.120823/76.979586 tok/s
(original/candidate). Full has four positive request medians;
retained has three, but its matched ratio fails the positive gate.
Hold/off, with frozen IDs and zero faults, without repeats or
long-context escalation. This is not a qualified performance gain.

The independent default-off `ORT_WEBGPU_QMOE_FC1_FULL_A` retains
the original 128-thread/four-channel geometry and stages all fixed
K2560 activations once. Shared storage is 5,632 bytes; runtime
barriers fall from seven to three. Original per-K-tile FP16 partial
updates, dot order, serial fold, bias and FP32 SwiGLU remain intact.
Unsupported devices retain the original route; the selector is
validated at both parsing sites and replay-keyed.
Its provider is
`2d68677a78b89910db04cc9b3a31e2d626e359294a3b4bb4ffd3da92aac32e77`.
Byte-identical parent relink, 23 native fixtures and 9,748,480 finite
bitwise comparisons pass, including replay and layout/bias fallbacks.
All 135 focused integration tests pass. Actual full 2K profiling
proves 48 new FC1 programs, zero originals, frozen prefixes and zero
faults. Full 2K controlled windows regress: matched ratio
0.9690265790, median 79.713588/77.221544 tok/s. All four request
medians favor original. Reject/off; retained timing completes for
evidence, but no repeat or long escalation is permitted.

Retained full-A timing also regresses: matched ratio 0.9690546510,
median 80.038789/77.522584 tok/s. All four retained request medians
favor original; frozen/repeated IDs and zero faults pass. Both-mode
evidence confirms rejection.

The next independent default-off `ORT_WEBGPU_QMOE_FC1_COLUMNS8`
tests eight activated channels per single-expert group, instead of
four, using 256 threads and 3,072 shared bytes. It halves group count
from 1,600 to 800, sharing each original activation tile across
twice as many columns. Each projected column keeps its original
32-lane K partition, two-column-per-thread work, per-tile FP16
updates, ordered fold, bias and SwiGLU. This is distinct from the
ten-expert layout and general dense-wide8 experiments. No speedup
or native arithmetic qualification is claimed during preparation.

Eight-channel FC1 subsequently passes byte-identical parent relink,
23 fixtures/9,748,480 finite bitwise comparisons, 141 integration
tests and actual 48-dispatch profiling. Its provider is
`b6be385fffb3119c0b4832691094ed474092b7c9fb1a8fb2f593e8973f485db5`.
Full/retained 2K matched ratios are 0.9956424479/0.9992645334;
median rates are 79.429462/79.189448 and 80.160229/80.190419 tok/s
(original/candidate). Request support is zero/four and one/four;
the slightly higher pooled retained median does not override its
negative matched ratio. Frozen/repeated IDs and zero faults pass.
Reject/off without repeats or long escalation.

### Exact eight-row expert-prefill experiment

The default-off/replay-keyed `ORT_WEBGPU_QMOE_PREFILL_ROWS8` retains
grouped expert ordering and both original four-row fallback routes.
On admitted NVIDIA FP16 M5120/M10240 expert projections, it shares
dequantized weights across eight rows instead of four, with the same
128 threads and 20,512 shared bytes. Original dot4 order, 32-lane K
partition, per-tile FP16 partial updates, ordered folds and bias
are unchanged. Unsupported devices/shapes keep row4 or original
paths; invalid selectors are rejected in both QMoE parsing sites
and MatMulNBits admission. Both QMoE replay and MatMulNBits objects
are rebuilt without changing existing kernel-object layouts.

Its provider is
`a3706eaa59f48abe7d7431e67ffe4f8a9ec8aff92019649eb5b081fd6ae274ff`.
Byte-identical row4 parent relink and 300,810,240 finite bitwise
comparisons across six original-layer fixtures pass, covering
retained chunk16 reference, indirect inputs, pending1/64, rejected
selectors and fallback rows. Native dispatch verification initially
expected row4 in the candidate-only profiled session; that metadata
expectation was corrected to require 32 row8 programs and the five
grouping programs, each 16 times. Existing immutable successful raw
evidence was reverified without rerunning GPU arithmetic or changing
the bitwise gate. All 22 related tests pass. Actual full 2K model
profiling proves 384 row8 projections, 192 occurrences of each
grouping program and zero row4 projections, with frozen prefixes
and zero faults. Controlled TTFT timing is next; no gain is claimed.

Eight-row controlled 2K TTFT fails both modes. Full/retained
matched changes are 12.7507213%/12.5619015% slower, with median
four-row/eight-row TTFT 3.487250/3.931890 and 3.491764/3.930399
seconds. All eight requests retain the frozen output prefixes,
including excluded warmups, and zero faults. Reject/off without
8K/32K escalation or repeats; the qualified row4 stack is unchanged.
These are short-output TTFT controls, not natural decode standards.

### Independent paired-dot and half-K-tile FC1 experiments

`ORT_WEBGPU_QMOE_FC1_PAIRED_DOT` pairs each lane's original projected
columns `group` and `group+4` in one private vec2 FP16 accumulation
loop, sharing the two activation vec4 loads across both columns.
Original 128-thread/four-activated-channel geometry, per-tile FP16
stores, serial folds, bias and SwiGLU remain. Admission additionally
requires the fixed 640-channel shape so no partial column pair can
be dropped. Its provider is
`4d4ed3d18537c60b93056d576b40fbcb633bd7c1504ca4804610e97e91ee33ae`.

`ORT_WEBGPU_QMOE_FC1_K16_TILE` keeps the same workgroup and output
geometry, but maps 128 threads to sixteen K lanes and eight projected
columns. Five 512-element activation tiles replace three 1024-element
tiles, avoiding idle half-warps in the final K2560 tile. Each half
tile writes its matching original 32-lane partial slots, preserving
every lane's sequence of FP16 updates and the original final fold;
no private-partial or reduction-tree substitution is used. Shared
storage drops from 2,560 to 1,536 bytes, while barriers rise from
seven to eleven. Its provider is
`657ac9da5b1758c68b1b32fc513471d2fbe723cb0065acecd7e0d4226310f541`.

Both selectors are default-off, validated at both QMoE parsing sites
and replay-keyed; original fallback shaders remain. Both builds
relink the unchanged row4 parent byte-identically. Each passes
23 native fixtures/9,748,480 finite bitwise comparisons, optional
bias/layout fallbacks, changing inputs/routes and replay tests.
All 156 combined integration/regression tests pass. Paired-dot actual
full 2K profiling proves 48 candidate FC1 programs and zero originals,
with frozen prefixes and zero faults. Both-mode controlled 2K
measurements are pending; no gain or long-context qualification is
claimed. Long escalation requires both initial and fresh-repeat 2K
positive controls with at least three/four request support in both
modes; 32K additionally requires both positive 8K controls.

Initial paired-dot full/retained 2K matched ratios are
1.0002781737/1.0016353128, with four/four and three/four positive
request medians. Fresh full/retained ratios are
1.0018627779/0.9998574261, with four/four and three/four support.
The fresh retained ratio fails the both-repeat gate. Hold/off,
without further retries or long escalation; a favorable pooled
retained median does not override the negative matched ratio.

Half-K-tile initial full/retained ratios are
1.0038740616/1.0003484741; fresh repeat ratios are
1.0036340199/1.0035875793. All sixteen request medians favor the
candidate, with frozen/repeated IDs and zero faults. Both 2K
repetitions pass; 8K/32K controls are next. These small controlled
gains are not natural-standard TPS or an over-100 claim.

The next independent default-off `ORT_WEBGPU_QMOE_FC1_K8_TILE`
uses eight K lanes and sixteen projected columns per 128-thread
group, with ten 256-element activation tiles and 1,536 shared bytes.
It halves group count to 800, while preserving each original
32-lane partial slot, its FP16 update sequence, bias and final fold.
Barriers rise to twenty-one. The original selector-off shader and
geometry remain intact. This is distinct from the slower
256-thread/eight-channel and full-activation layouts.
Its provider is
`23b949bd2df05fb5f1914bf3db1ed550fb17da31858a59e398f49bf5aff0bcf6`.
Byte-identical row4 parent relink and 23 native fixtures/9,748,480
finite bitwise comparisons pass, including replay and bias/layout
fallbacks. Model profile and controlled 2K windows are pending.

After narrower-tile integration, all 160 combined tests pass. The
half-K-tile 8K full/retained ratios are 1.0020293991/1.0038955942,
with all eight request medians positive, frozen IDs and zero faults.
Both 8K gates pass; 32K controlled measurements are running.

The current instrumented 2K decode profile still attributes about
2.728 ms to 94 HC post-norm projections and 2.665 ms to 159 dense
block32 projections; FC1/FC2 each account for about 1.08 ms.
These timestamp-instrumented categories guide experiments, not
normal TPS claims. The previously held HC bitcast experiment is
not retried unchanged.

The next default-off `ORT_WEBGPU_DENSE_QKV_K16` targets only canonical
NVIDIA FP16 M1/K2560/N16384 QKV projections. It uses 128 threads,
sixteen K lanes and eight output columns, with 512-element activation
tiles and 1,536 shared bytes. The original 32 partial slots remain:
each half tile writes its corresponding original lane slot, preserving
the sequence of typed FP16 updates and the ordered final 32-lane fold.
All original dot expressions and non-target/FP32/row4-prefill fallbacks
remain. The selector is validated and separately replay-keyed.
This differs from previously rejected wide16 and generic wide8 routes.
Three preparation tests pass; build/native/model qualification is
pending and no performance gain is claimed.

Half-K-tile FC1 completes 32K full/retained timing with matched
ratios 1.0006843453/1.0039613151. Request support is one/four and
three/four. Median windows are 70.005912/70.048733 and
70.128345/70.310934 tok/s (original/candidate). Frozen/repeated IDs
and zero faults pass, but full support fails. The aggregate correctly
rejects all-context qualification. Hold/off without natural standards,
promotion or further diagnostic retries; the positive 2K/8K results
are not relabeled as an all-context gain.

Dense QKV half tiles build successfully with byte-identical row4
parent relink:
`2d3c326e0f5bd9e2ca18af759e8e0ae47423125058f8dfe8620dcaec3fa70864`.
Native normal and explicit FP32-fallback runs each pass 27 fixtures
and 116,582,784 finite bitwise comparisons. Replay proves a second
variant entry only for the target shape; all eight other shapes
retain one entry. All 168 combined integration/regression tests
pass, including flag restoration, interruption and strict-support
gates. Actual-model profiling must prove 36 new QKV programs and
123 retained block32 programs before timing. No TPS gain is claimed.

K8 FC1 actual-model profiling proves 48 new programs. Full/retained
2K matched ratios are 0.9942334152/0.9932344510; all eight request
medians favor the original. Rejected/off without fresh repeats or
long-context escalation.

Dense QKV K16 actual profiling proves 36 new QKV programs and
123 retained block32 programs. Full/retained 2K matched ratios are
0.9961477285/0.9971586124; all eight request medians favor the
original. Frozen IDs and zero faults pass. Rejected/off without
fresh repeats or long-context escalation.

An independent original-row4 CPU diagnostic on GPU7 collects
owned-process user-mode call stacks, without system/hardware changes.
Two 768-output requests preserve all 512 checked original-prefix IDs
with zero major faults. This is instrumented diagnostic evidence,
not a natural standard or TPS gain. Initial whole-process sampling
is dominated by runtime thread-pool `SpinPause`/worker loops and
model-load copying; it cannot identify decode bottlenecks alone.
A separate phase-marked diagnostic uses the existing decode-step
callback and monotonic timestamps to exclude model loading and
prefill from subsequent CPU conclusions. Frozen model and retained
row4 provider are unchanged; rejected experiments remain off.

The phase-marked diagnostic subsequently resolves 1,136 main-thread
user-mode samples during the measured decode interval, with zero lost
samples. Inclusive CPU percentages include Flush 11.44%, deferred
encoding 11.18%, BufferManager allocation 9.24%, and environment
parsing 8.54%; greedy CPU selection is 3.35% self. These overlap and
are not additive or equivalent to wall-time savings. The resolved
report and monotonic phase boundaries are retained separately from
performance standards. No new TPS gain is claimed.

A new independent, default-off `ORT_WEBGPU_QMOE_FC1_U32_PARTIALS`
experiment keeps original FC1 geometry, dot expressions, typed FP16
updates, serial 32-lane folds, bias and SwiGLU. Each shared half value
occupies the low half of a private u32 slot; the high half is zero.
This avoids adjacent lanes sharing a 32-bit storage word without
introducing FP32 arithmetic. Shared storage rises from 2,560 to
3,072 bytes. Admission is NVIDIA-only with sufficient workgroup
storage; selector-off shaders and existing layout fallbacks remain.
The selector is validated at both parsing sites and replay-keyed.
Build/native/model qualification is pending.

The u32-partial candidate builds with byte-identical retained-parent
relink and provider
`bb6d3d5511d7ed7f610d76f22184d39a6b7daff4fa3d714bdcf1fea9e735f27f`.
All 23 native fixtures/9,748,480 finite bitwise comparisons pass,
including changing inputs/routes, replay, optional bias and existing
layout fallbacks. The new benchmark selector, metadata, exact
environment restoration and conservative all-context prerequisites
are wired; 144 focused regression tests pass. Actual profiling and
full/retained 2K controls remain required before any speed claim.

The u32-slot actual profile proves 48 candidate programs and zero
original FC1 programs. Full/retained 2K matched ratios are
0.9984004547/0.9985600709; all eight request medians favor the
original. Frozen/repeated IDs and zero faults pass. Rejected/off
without fresh repeats or 8K/32K escalation. The higher pooled full
median does not override the negative matched ratio.

The next independent default-off `ORT_WEBGPU_QMOE_FC1_EXPERT_PAIRS`
shares each activation tile across two routed experts in a
256-thread workgroup. Each expert retains the original four
projected-column groups, 32 K lanes, typed half updates and ordered
folds. Group count falls from 1,600 to 800, with 3,072 shared bytes.
This differs from the earlier ten-expert/two-projected-column
320-thread layout and eight-channel/single-expert layout. Admission
requires the original ten routed rows and sufficient NVIDIA limits;
original shaders and existing fallbacks remain. Build/native/model
qualification is pending.

The paired-expert candidate builds with byte-identical row4-parent
relink and provider
`397e9099795568ad0d5140e3a250ffcbd538979289013682c5e45b319d383265`.
All 23 native fixtures/9,748,480 finite bitwise comparisons and
147 focused tests pass. Actual profiling proves 48 candidate FC1
programs and zero originals. Full/retained 2K matched ratios are
0.9986104378/0.9959997797, with all eight request medians favoring
the original. Frozen/repeated IDs and zero faults pass. Rejected/off
without fresh repeats or long-context escalation.

The next tuning screen uses the existing public maximum-pending
dispatch option on the unchanged row4 provider. It changes no shader,
arithmetic, model, system setting or allocator behavior. Both cached
and non-cached dispatch paths in the exact frozen context source use
this configured limit. Fresh 2K separate-load observations at smaller
limits are bracketed by pending64 controls, with three measured
256-output requests and one excluded warmup per load. These are not
matched causal gains or all-context qualification. Any promising
setting requires subsequent both-mode controlled qualification before
8K/32K escalation; original standards and pending64 defaults remain.

The completed full-2K separate-load screen passes frozen IDs and zero
major faults in every load:

| Load order | Maximum pending dispatches | Median decode tok/s | Mean TTFT (s) |
|---|---:|---:|---:|
| Before control | 64 | 78.137791 | 3.482643 |
| Candidate | 16 | 69.900640 | 3.479137 |
| Candidate | 32 | 77.719563 | 3.479680 |
| After control | 64 | 77.603457 | 3.484737 |

Pending16 is decisively slower. Pending32 falls between the controls:
there is no convincing gain, and these observations are not matched
causal evidence. Keep pending64; do not escalate either setting to
retained-mode or long-context qualification.

The independent default-off `ORT_WEBGPU_DENSE_QKV_PREDECODE8`
limits persistent decoded FP16 weights to constant K2560/N16384
single-token QKV projections. Eight columns/256 threads retain the
original 32 K lanes and dot/update/fold ordering, unlike the rejected
all-nine-shape/four-column cache. Each initializer adds 80 MiB.
Original classes, multi-token/dynamic-weight/FP32 fallbacks and row4
prefill remain. NVIDIA/storage/dispatch admission is explicit.
The row4-parent relink is byte-identical; candidate provider is
`93c0f37b62b82c7b6d8c796f2537a5b0e2552d1579d5016af4d3badfd4618910`.
Normal and explicit FP32-fallback runs each pass 27 native fixtures
and 116,582,784 finite bitwise comparisons. Actual model profiling
proves 36 candidate and 123 original dense dispatches, with zero
repeated dequantization after warmup. All 148 focused tests pass.

| 2K validation | Original pooled window tok/s | Candidate pooled window tok/s | Matched ratio |
|---|---:|---:|---:|
| Full | 79.500833 | 74.420294 | 0.9344909547 |
| Retained | 80.219310 | 75.164074 | 0.9373337281 |

Frozen/repeated IDs and zero faults pass, but all eight request medians
favor the original. Reject/off: no fresh repeat, long-context
escalation or promotion. This is a performance rejection, not a
quality qualification or a new headline standard.

The next default-off `ORT_WEBGPU_DENSE_QKV_THREADS64` retains compressed
INT4 weights, four output columns/group and 32 K lanes, but uses
64 threads with two columns/thread instead of 128 threads with one.
The original template arithmetic, row4 prefill and fallbacks remain.
Provider `bbab7169b35fcb7e603bbe92d2ed3cc02819cd77a34bc05f5b9a6ed5f8ad7f97`
passes byte-identical row4-parent relink, normal/FP32-fallback native
proofs of 27 fixtures/116,582,784 comparisons each, 153 focused tests,
and actual coverage of 36 candidates/123 originals. Full/retained
initial 2K matched ratios are 0.9915283788/0.9939289468. All eight
request medians favor the original; IDs and zero faults pass.
Reject/default-off without repeats or long-context escalation.

The independently scoped `ORT_WEBGPU_DENSE_QKV_PAD34` preserves the
original 128 threads, four columns and compressed weights, padding
each 32-element FP16 partial row to 34 elements. Under a 32-bank,
four-byte-bank model, the four serial-fold writer lanes address four
distinct banks rather than two. Arithmetic, logical slots, input
staging and fold order are unchanged; shared storage adds only
16 bytes (2,304 to 2,320). This bank model is a hypothesis, not
measured proof of a bottleneck.
Provider `07f0587ecedbac71d6df9807dfb75197876f6e5544c3b6dfad47633a96f33616`
passes byte-identical row4 relink, normal/FP32-fallback native
proofs of 27 fixtures/116,582,784 comparisons each, 158 focused tests,
and actual 36-candidate/123-original coverage. Initial full/retained
2K ratios are 0.9959863799/0.9992577813, with 0/4 and 1/4 positive
request medians respectively. IDs and zero faults pass.
Reject/default-off; no repeat or long-context escalation.

A distinct isolated FP16 tensor-product probe replaces the
rounded-products `dot(ones, p)` with explicitly half-rounded pair
sums before the final half addition. This changes only arithmetic
research fixtures, not the supplied model. The byte-identical c1
relink and six focused tests pass, but native offset0/round0/element4
rejects again: original bits 36084, candidate 36082. Provider
`970663e5f88595f2f3658240e51eef51871fb4ca1c2d155c092a494c8a959e40`
is not dot4-compatible or performance-qualified.

New single-weight fixtures at K0/K4/K12 isolate native tensor
multiplication from multi-term association. Original four-weight
fixture generation remains byte-identical. Both existing FP16-result
and FP32-result matrix variants pass 983,040 finite bitwise comparisons
on these single-product fixtures. This does not qualify dot4.

A host hypothesis fits two known dot4 mismatch values using cross
pairs (0,2)/(1,3): half-round the first products, then fuse each
remaining product into its pair before the final half addition.
Three distinct staged native probes test this hypothesis:
FP16 accumulators, FP16 accumulators with an explicit half-store/reload,
and FP32 accumulators with explicit half-rounding boundaries.
All build with byte-identical c1 relinks, but all reject offset0,
round0, element1: original bits 4584 versus 4585.
The two-case host fit does not generalize, and explicit intermediate
materialization does not repair it. These remain isolated arithmetic
rejections, not qualified model kernels or throughput results.
All 12 focused tensor tests pass.

### Canonical 2,048-token grouped row4 prefill extension

The default-off `ORT_WEBGPU_QMOE_PREFILL2048_ROWS4` experiment extends
the qualified grouped row4 projections from 5,120/10,240 routed rows
to exactly 20,480 routed rows (2,048 input tokens). Existing short,
decode and neighboring fallback routes remain intact. The QMoE
selector is validated in compute/capture and included in the replay
key. The projection shader, four-row/eight-column tile, 128-thread
workgroup and arithmetic are unchanged.

The 102,400 logical SwiGLU activation groups exceed the one-dimensional
WebGPU limit. Existing direct-dispatch normalization maps this exact
count to 320 x 320 x 1 groups; existing shader workgroup/global-index
flattening covers every logical group once. No new dispatch shortcut
or removal of buffer initialization is used. This differs from the
earlier non-grouped 2,048-token experiment.

V1 provider
`ac3b4b0b814be4d7ac47f9939160f39ea4fa6945388109561b6cd97efbb9bce5`
passes a byte-identical qualified row4 parent relink and six original
expert fixtures: **678,297,600 finite bitwise comparisons**, pending
windows 1/64, retained chunk16 references, original shorter shapes,
2,047/2,049-token fallbacks, invalid-selector rejection and actual
large projection/grouped-program coverage.

V2 provider
`c65de0fddbe9a7281108c36b24672821321b51f86619a3ca215bbc90e18a6b51`
adds the prior 2,048-token attention admission guard under the new
selector, preserving default bounds and attention shader arithmetic.
The original attention object is verified against the existing
byte-identical rebuild proof. Integrated QMoE passes the same 678,297,600 finite bitwise comparisons;
attention passes 338,208,768 finite CPU-reference comparisons and
12 expected admission/invalid-selector rejections. All 45 focused and
related prefill tests pass. The actual full 2K model profile proves
96 grouped programs of each kind, 96 large gates and 192 large
row4 projections split equally between both original projection
shapes.

| 2K validation | Control / candidate decode tok/s | Control / candidate TTFT s |
|---|---:|---:|
| Full | 78.129618 / 74.837054 | 3.483815 / 3.653716 |
| Retained | 78.108639 / 75.256178 | 3.489629 / 3.649646 |

Both natural 256-output standards pass frozen-ID and zero-major-fault
checks, but TTFT is worse in both modes and there is no decode gain.
Reject/default-off; no unchanged repeat or long-context escalation.
Candidate chunk2048 versus qualified chunk1024 is a separate-load
comparison, not an isolated flag-only route gain.

### Startup CPU locality screen

A separate process-only placement experiment keeps the qualified row4
provider and frozen model/arithmetic unchanged. GPU7's read-only topology
reports NUMA node1/local CPUs48-95, while the unrestricted process permits
CPUs0-95. Startup `taskset` compares unrestricted placement against CPUs48-95
and CPUs48-55, without changing memory policy, clocks, power limits or any
foreign process.

This is distinct from the earlier rejected `CPU_LOCAL_WINDOWS` experiment:
that changed only the benchmark main thread after workers had been
created. The startup screen records the actual main-thread mask and all
surviving worker masks after natural requests, outside measured request
timing; qualification requires every worker to remain within the selected
mask. Three natural 256-output requests plus excluded warmup are bracketed
by unrestricted controls in both full/retained modes. Long contexts remain
blocked until both modes show supported decode improvement without TTFT
regression. These are separate-load placement screens, not isolated shader
gains.

The first baseline completed but lacked natural-path affinity metadata;
that output remains preserved and unqualified. Metadata capture was moved
from the window-only completion path to the natural-standard path, with
an explicit placement regression test. A fresh immutable v2 evidence family
was run. Its local case failed the worker-scope gate: the runtime
explicitly repinned workers outside the inherited startup mask.
The excluded output is preserved; `taskset` alone is not proof that
every runtime worker is local.

V3 adds default-off `--inherit-worker-cpu-affinity`. After model loading
and before warmup/timing, it assigns every existing worker the main-thread
mask, verifies each assignment and records predecessor masks. Future
workers inherit the main-thread mask; the final snapshot must still pass
the all-worker gate. The operation affects only the owned benchmark
process. Existing main-thread-only window behavior remains unchanged;
136 helper/window tests pass.

This v3 diagnostic applies inheritance to unrestricted controls too,
removing their original individual worker pins. Therefore even a positive
v3 placement ratio would require comparison against the qualified default
worker policy before promotion. No locality performance gain is claimed yet.

V3 completes all eight natural 2K loads with frozen IDs and zero faults.
The 48-CPU candidate fails retained support (0/3 positive requests);
the eight-CPU candidate passes 3/3 request support in both modes against
the diagnostic unpinned controls:

| Validation | Before / local8 / after decode tok/s | Bracketed request ratio |
|---|---:|---:|
| Full | 74.199047 / 78.124735 / 77.989892 | 1.0271393 |
| Retained | 78.039135 / 78.791721 / 78.222779 | 1.0142499 |

These are not gains over the qualified default worker policy. A separate
frozen-source wrapper preserves individual worker pins in its controls and
changes only the candidate. It requires explicit control/candidate mask
verification, both initial 2K modes and a successful fresh bracket before
8K/32K escalation. Initial and fresh default-policy 2K comparisons pass:

| Bracket | Validation | Before / local8 / after decode tok/s | Bracketed request gain | Local8 TTFT s |
|---|---|---:|---:|---:|
| Initial | Full | 77.672273 / 78.460314 / 77.972978 | +0.684807% | 3.485248 |
| Initial | Retained | 76.636527 / 79.009187 / 75.788054 | +3.433382% | 3.484584 |
| Fresh | Full | 77.972978 / 78.371129 / 77.896297 | +1.006075% | 3.482069 |
| Fresh | Retained | 75.788054 / 78.882947 / 78.190515 | +2.077868% | 3.484489 |

Every bracket has 3/3 positive measured requests in each validation mode.
Frozen IDs, zero major faults, source/provider hashes and worker-mask
checks pass. Gains are geometric means of matched request ratios against
the surrounding controls, not ratios of the displayed median TPS.
These are separate-load placement screens, not same-load shader causality
or whole-model quality qualification. The repeated 2K proof permits an
initial 8K bracket, now in progress; fresh 8K success is required before
32K. The qualified stack remains unchanged and 100 tok/s remains unmet.

The initial 8K bracket also passes both modes with 3/3 positive requests:

| Validation | Before / local8 / after decode tok/s | Bracketed request gain | Local8 TTFT s |
|---|---:|---:|---:|
| Full | 77.808927 / 78.472084 / 76.892521 | +2.121102% | 14.170228 |
| Retained | 78.535404 / 78.956525 / 77.264744 | +1.599506% | 14.146851 |

Frozen IDs, zero faults and worker-mask gates pass. Fresh 8K confirmation
also passes:

| Validation | Before / local8 / after decode tok/s | Bracketed request gain | Local8 TTFT s |
|---|---:|---:|---:|
| Full | 76.892521 / 78.084840 / 78.137154 | +2.091510% | 14.177243 |
| Retained | 77.264744 / 78.787510 / 78.493315 | +1.487034% | 14.159214 |

All 24 matched request comparisons across initial/fresh 2K and 8K brackets
favor local8. TTFT remains essentially unchanged. Repeated 8K success
permits the initial 32K bracket, now running on the unchanged qualified
provider/runtime. No all-context promotion or 100 tok/s claim.

The initial 32K bracket passes:

| Validation | Before / local8 / after decode tok/s | Bracketed request gain | Local8 TTFT s |
|---|---:|---:|---:|
| Full | 67.646439 / 69.078555 / 68.779720 | +1.542818% | 59.585655 |
| Retained | 69.213692 / 69.477840 / 68.869333 | +0.670386% | 59.547205 |

All six measured requests favor local8, with frozen IDs, zero faults and
worker-mask checks passing. Fresh 32K confirmation is running without
intervening builds or other GPU work. This small initial gain does not
establish repeated all-context locality support or reach 100 tok/s.

A separate all-context aggregator is prepared to verify all six immutable
initial/fresh screens, request support, TTFT gates, source/provider hashes
and the exact 2K/8K/32K full/retained output shape. It reports insufficient
support explicitly rather than repairing a failed screen. Three tests
cover rejected support, misreported ratios and the strict requirement that
all six fresh candidate medians exceed 100 tok/s. It has not run while the
fresh 32K evidence is incomplete.

Fresh 32K confirmation passes:

| Validation | Before / local8 / after decode tok/s | Bracketed request gain | Local8 TTFT s |
|---|---:|---:|---:|
| Full | 68.779720 / 69.115199 / 68.718434 | +0.848195% | 59.586137 |
| Retained | 68.869333 / 69.786429 / 68.935767 | +0.975975% | 59.610205 |

The all-context aggregator subsequently passes with 36/36 positive request
comparisons and all original-prefix, fault, worker, source and provider
gates. Status is
`repeated_default_policy_cpu_locality_supported_all_contexts`.
The strict all-context 100 tok/s flag is false. This qualifies the
default-off GPU7 process-placement experiment, not a new shader/provider
stack or whole-model quality. TTFT is essentially unchanged.

### Unrounded independent tensor-product arithmetic

A distinct arithmetic-only probe materializes independent FP16-input
tensor products as FP32, sums four products with an FP32 dot and converts
the result to FP16. This removes the premature product rounding from the
earlier FP16-product probe, without changing the supplied model.
V1 provider
`75f0c15a3b002495a3ae116103d68f1292b231fcd3be48d550a8e17a0ccad476`
passes the byte-identical c1 relink but rejects offset0/round0/element2:
original bits 5774, candidate 5773. FP32 product materialization does not
repair dot4 compatibility; no model/performance escalation.

V2 source corrects capability selection to require FP16 inputs with
FP32 matrix results, matching its generated shader. This admission-only
correction does not change the rejected arithmetic. Its CPU build and
byte-identical baseline relink pass, with provider
`edf9482fd38af6f5f533ef25cf9cdd9c1e6811fc372ec16639f106db1336c4bb`.
It will not receive an unchanged numerical GPU repeat.

A separate reversed cross-FMA prototype instead first materializes
products at K2/K3, then accumulates K0/K1 into their original cross pairs.
Both activation and packed-weight mappings reverse within each quartet.
Focused tests and its byte-identical baseline relink pass, with provider
`26a6ff02fa03c8ab7c0d6c0c409a012735a1a1d64d114067fc4fd8b960840a7c`.
The native GPU7 screen rejects offset0/round0/element1: original bits
4584, candidate 4585. This distinct ordering also fails compatibility;
it remains default-off with no model/performance escalation.
Neither prototype is general-matmul or throughput qualified.

### Allocator-session CPU worker investigation

Decoder intra-op/inter-op counts already equal one, and the native CPU
embedding session also defaults to intra-op one. A separate source trace
finds `EnsureDeviceOrtInit` creates a persistent trivial allocator session
without an explicit intra-op limit. This is a candidate explanation for
the 95 individually pinned workers, not yet a measured attribution.

An isolated source transformation adds default-off
`ORT_GENAI_WEBGPU_ALLOCATOR_SINGLE_THREAD`, using the existing boolean
`GetEnv` helper and applying intra-op one only to the WebGPU allocator
session. Model, decoder, CPU embedding, provider and arithmetic remain
unchanged. Three source tests pass. The builder requires both a
byte-identical parent relink and a byte-identical rebuild of the original
model source before producing the candidate runtime. Both byte-identical
checks and the CPU build pass, with runtime
`04c60710104f76fb1e0b94d2aa363b0c93c6130f4c542f59d16f69d39cea34c2`.
Worker-count/model/performance validation remains pending; no gain or
promotion is claimed.

The prepared worker probe uses the same candidate runtime with the flag
off/on, unchanged WebGPU provider and main affinity, full validation,
36 outputs and one measured request plus excluded warmup. It verifies the
actual mapped runtime hash and frozen prefixes, then requires the default
95 individual worker pins versus zero under the single-thread allocator
setting. Three tests cover commands, masks and rejected overrides. This
is a resource diagnostic, not a standard-TPS result; no GPU worker probe
has run yet.

### Physical dense lane-layout preparation

A separate default-off `ORT_WEBGPU_DENSE_LANE8` experiment keeps 128
threads, four output columns and 32 logical K lanes per column. Unlike
the rejected wider-column tiling experiment, it only permutes physical
lanes: each warp covers four columns with eight K lanes each. Original
activation staging, dot order, FP16 partial updates and serial final fold
remain unchanged. A bijection test enumerates all 128 logical lanes and
each warp's exact geometry.

The private program and replay key are distinct, with NVIDIA/FP16
canonical block32 admission and unchanged original/FP32 fallbacks.
Four source/wiring tests pass. V1 compilation succeeded but its link
replacement assertion exposed that the qualified parent owns a newer
`matmul_nbits` object containing grouped row4 prefill. V1 remains
unqualified. V2 derives the actual parent source, preserving that path,
and passes the byte-identical parent relink with provider
`1f5814650629ca1db68c1f78053f249f543c0b4bd96fe251adeeea0c0b8e9d1f`.

Native GPU7 validation passes 27 fixtures and 116,582,784 finite bitwise
comparisons, including pending1/64, real replay binding reuse, invalid
selector rejection and M2/M17 fallbacks. Separate explicit FP32 fallback
validation passes another 116,582,784 comparisons. Builds and native
screens ran between completed 8K and unstarted 32K placement brackets,
not inside a matched timing bracket. Model profile/performance validation
remains pending; no throughput gain is claimed.

The model screen uses the frozen natural benchmark without adding new
window options or changing existing CPU-proof hashes. Separate loads use
the same candidate provider with lane8 off/on/off, unchanged default worker
policy, three measured 256-output requests and excluded warmup. Guard
commands explicitly record the new flag; an actual decode profile must
first show all 159 lane8 programs and zero old block32/tile4 programs.
The unchanged benchmark does not include this new flag in its own
environment metadata, so the dedicated qualification records its value
alongside the exact guard command and source/native/profile hashes.

Both initial and fresh full/retained brackets must have 3/3 positive
request ratios and TTFT ratio at most 1.02 before the next context.
This is explicitly a separate-load shader-flag screen, not the original
same-load window protocol. Four model command/profile tests pass; no
model screen has run yet.

The actual full 2K decode profile now passes: 159
`MatMulNBitsLane8F16`, zero original block32/tile4 programs, 12
`DynamicSparseAttentionQueryCache` and zero original attention programs.
Frozen 36-output prefixes, zero faults, default workers and the unchanged
GenAI runtime pass. The instrumented profile is not a speed claim.
An initial off/on/off natural 2K bracket is now running.

The initial and fresh separate-load 2K screens now pass both validation
modes, with all three requests positive per mode and bracket. Initial
full/retained candidate medians are 78.410195/78.994280 TPS, with
matched-request gains of 2.932396%/0.587876%. Fresh results are:

| Validation | Before / candidate / after median decode TPS | Matched-request gain | Candidate TTFT |
|---|---:|---:|---:|
| Full | 74.350149 / 78.448088 / 77.913814 | 3.110795% | 3.490443 s |
| Retained | 78.624023 / 79.383050 / 78.108784 | 1.177672% | 3.484440 s |

The shared slow full-mode control affects both brackets, so the larger
full gain is not an isolated causal 3% claim. The retained signal is
smaller but positive in both screens. Frozen prefixes, zero faults,
source/runtime hashes and unchanged default worker masks pass; TTFT
ratios remain at most 1.02. The fresh status is
`repeated_lane8_decode_gain_both_modes`, permitting 8K validation.
CPU-local8 is not stacked, and 100 TPS remains unmet.

### Allocator worker attribution

The actual same-runtime off/on GPU7 resource probe now passes. The
default allocator session produces 208 observed threads, including 95
individually pinned workers; the single-thread setting produces 113
observed threads and zero individually pinned workers. Frozen
36-output prefixes, mapped runtime/provider hashes and fault gates pass.
This attributes the additional 95 workers to the persistent allocator
session rather than the already single-threaded decoder or embedding.
These short diagnostic requests are not standard TPS evidence; the
flag remains default-off pending performance validation.

### Packed INT4 four-column locality preparation

Default-off `ORT_WEBGPU_DENSE_PACKED4` interleaves complete 16-byte INT4
blocks from four columns and their FP16 scales, retaining original
storage datatypes and sizes. It does not expand weights into FP16 or
change dequantization, dot order, partial updates or final fold.
The private kernel owns constant-only cached weight/scale copies,
includes both as explicit replay inputs and admits only canonical
NVIDIA/FP16 block32 shapes with N divisible by four. N1 and multi-row
requests preserve original fallback behavior.

V1 compiles and passes a byte-identical parent relink, producing provider
`388255125977f7b71e1bb2cf2a4e1742bee3d743418439d01a4c0bd88e67430b`,
but its native run fails shader creation before numerical validation:
the producer uses WGSL's reserved identifier `target`. V1 evidence is
preserved unqualified. V2 renames that identifier to `packed_offset`;
a regression test protects the correction. V2 build/native/model
validation is pending; no gain is claimed.

V2 builds with byte-identical parent relink and provider
`d68007c2913a18aea976eb8d4332a06c86322b5a807430670292b9df394dad61`.
The producer shader now compiles, but native execution rejects its
replay binding order before numerical qualification: the reused helper
prepends both cached inputs, whereas the consumer binds cached weights,
activation, then cached scales. V2 remains unqualified. V3 specializes
the private helper to that exact order, validates two cached inputs and
activation input0, and retains strict captured-buffer equality checks.
A source test protects ordering; build/native validation is pending.

V3 builds, but its pre-run source-hash gate rejects a stale V1 include
path in the manifest; no V3 GPU work starts. V4 derives manifest paths
from the actual include directory and builds with byte-identical parent
relink, provider
`89fcb3edc6e1547627aedad72294dc78fa95c8961148d94b5a28986aa7714fc6`.
Normal native validation passes 27 fixtures and 116,582,784 finite
bitwise comparisons. The N1 fallback uses the same original replay
variant under both flag values, correctly producing 256 requests and
one entry, not the initially assumed two. The verifier correction
qualifies the unchanged successful GPU evidence without rerunning it.

All eight admitted shapes have separate original/packed replay rows,
128 requests each and real binding-cache reuse. N1 has only its original
row. These fixtures reuse GPU addresses, so they prove changing-content
replay and binding hits, not different-address rebinding; that additional
check remains pending. Separate explicit FP32 original-fallback
validation passes another 116,582,784 comparisons. Pending1/64,
invalid-selector rejection and multi-row fallbacks pass. Seven tests
cover packing bijection, retained arithmetic, admission, private
binding order, manifest paths and N1 statistics. No packed4 model
throughput or quality claim is made.

A separate comparator-only rebinding proof is prepared against the
unchanged V4 provider. It keeps three distinct GPU activation tensors
alive, verifies their GPU memory and different buffer handles, copies
each round's CPU activation into the selected bank, then compares both
flag modes using that bank. Its verifier requires at least two actual
rebinds, three binding entries and positive binding hits for every
original/packed replay row before writing qualification. Two tests
pass. Compilation and GPU execution are deferred until the ongoing
lane8 8K timing bracket completes.

The lane8 8K bracket has completed. Initial full/retained medians are
78.467105/78.735534 TPS, with gains of 1.009567%/0.681953% and 3/3
positive requests each; one retained ratio is essentially tied at
1.000008901. Fresh full is 78.561571 median TPS and 14.197841 s TTFT,
but only 2/3 requests improve: ratios are
1.012479087, 1.011951937 and 0.980753211. Fresh retained is 79.368714
TPS and 14.152952 s TTFT, with 1.408363% gain and 3/3 positive requests.
The fresh aggregate therefore reports `no_supported_decode_gain`.
Lane8 remains default-off with limited 2K support, not all-context
qualification; no unchanged retry or 32K escalation is performed.

The comparator-only packed4 rebinding proof now compiles with SHA
`fb1254c200b0206f3b38759c0edc72f356d4106b16db9f6a2647a3ba5dffc29e`.
GPU7 passes another 27 fixtures and 116,582,784 finite bitwise
comparisons against the unchanged V4 provider, with three distinct live
GPU activation buffers and at least two actual rebinds, three binding
entries and positive binding hits in every replay row.

The packed4 full 2K last-token model profile passes: all 159
`MatMulNBitsPacked4F16` programs, zero original block32/tile4 programs,
zero cache-packing programs in the measured decode profile, 12
query-cache attention programs and zero original attention. Original
36-token prefixes, fault gates, runtime/provider/source hashes and
default workers pass. This instrumented profile is not throughput
evidence. Three model command/profile tests and two rebinding tests
pass; an initial separate-load off/on/off natural 2K model screen is
running without stacking CPU-local8 or allocator changes.

The completed packed4 initial 2K screen rejects promotion: full validation
76.580928 tok/s, matched ratio 0.980229735 and 0/3 positive requests;
retained validation 78.688829 tok/s, ratio 1.000042313 and 2/3 positive.
The flag remains off, with no unchanged retry or long-context escalation.

Removing allocator workers also fails the standard 2K performance gate:
full 74.972390 tok/s, ratio 0.966925103 and 0/3 positive; retained
78.746238 tok/s, ratio 1.012247101 and 3/3 positive. The retained result
shares a slow after-control. The 95-worker attribution remains proved,
but resource reduction alone does not establish a throughput benefit.

The new immediate-node-replay provider passes 116,582,784 finite bitwise
comparisons and another 13,107,200 mixed Add/MatMul/Add/MatMul comparisons,
including deferred-prefix ordering and pending limits 1/64. Both model
coverage attempts remain unqualified. The second verifier accidentally
read QMoE's statistics: its prepended static-stat export was overridden
by the inherited command template. The actual separate static-stat file
contains 171 dense replay instances, 11,799 hits and 10,714 binding hits;
the earlier inference of zero dense replay instances was incorrect.

Prior art already implements effectively the same unprofiled immediate
encoding route. Its October 29 window ratio was 1.0012243345, with only
one of four request medians positive, on a different model graph.
No new immediate-replay throughput claim is made or unchanged timing
retry started. The next distinct default-off experiment targets dense
replay-key vector allocation, without changing arithmetic or commands.

The default-off dense replay scratch-key candidate uses a private helper
type and preserves kernel storage size (checked during compilation).
Only the actual qualified row4 MatMul object is rebuilt. Keys are cleared
before reconstruction; misses transfer key ownership before executing
the compute callback. CPU-value limits, binding checks, command ordering
and shader arithmetic remain unchanged.

Five focused tests pass. A compiled extraction of the actual key-building
fragment verifies changing key contents and miss ownership, then measures
exactly 1,000 allocations with reuse off versus zero after warmup with it
on. This isolated allocation result is not an end-to-end speedup.
The parent relink is byte-identical; candidate provider SHA-256 is
`e62216392b3321f97932222e7ee08f5a00591f9457444f17bf2e810f5716f71a`.
Native validation passes 116,582,784 finite bitwise comparisons across
27 fixtures, pending limits 1/64, and invalid selector rejection.
Unprofiled full 2K coverage passes original 36-token prefixes, faults,
default workers and hashes, with all 171 private dense replay instances
and 11,799 replay hits. Initial full/retained 2K timing is pending;
the qualified provider and selected settings remain unchanged.

The completed scratch-key screen is held: full 2K 77.485259 tok/s,
3.488371 s mean TTFT, matched ratio 1.009442925 and 2/3 positive;
retained 78.396782 tok/s, 3.486326 s, ratio 1.018342196 and 3/3.
Both ratios include slow after-controls (75.771535 and 76.283830 tok/s).
No unchanged repeat or long-context escalation is started.
The next separate default-off experiment checks the most recent binding
before constructing a temporary buffer vector or searching the cache.
It requires prototype identity, exact buffer count, every validated
input/output handle, and the uniform-buffer handle to match; all misses
retain the original cache path. Scratch-key reuse is not stacked.

The recent-binding provider builds with a byte-identical parent relink
and private kernel/helper names. SHA-256:
`0c066e9d24d32131f6896f03fba0b116000648a49572a19b3204cbfe90176e84`.
Eleven focused tests pass. Matching-binding native validation passes
116,582,784 finite bitwise comparisons with observed recent-binding
hits. A separate proof passes another 116,582,784 comparisons using
three concurrently live GPU activation buffers: the candidate uses a
different bank than the control, requires at least two actual rebinds
per shape, and records zero recent-binding hits as expected for forced
misses. Both preserve pending limits 1/64 and invalid-selector rejection.

Full 2K model coverage passes the original 36-token prefixes, faults,
workers and hashes. All 171 private dense replay instances are observed:
11,799 replay hits, 10,714 binding-cache hits and 3,772 recent-binding
hits (35.2063% of binding-cache hits). This coverage is not a throughput
claim; an initial paired 2K screen is pending.

The completed recent-binding screen is held. Full 2K: 77.662958 tok/s,
3.485655 s mean TTFT, matched ratio 1.011806668 and 3/3 positive,
including a slow 74.783626 tok/s before-control. Retained:
78.389766 tok/s, 3.486293 s, ratio 1.000418501 and only 2/3 positive.
No unchanged repeat or long-context escalation is started.

The next distinct default-off CPU candidate is
`ORT_WEBGPU_DENSE_OUTPUT_SHAPE_FAST`. The existing MatMul helper's
flattened case allocates three offset vectors that this caller never
uses. For positive K, rank at least two, exactly one input row and a
matching trailing K dimension, the candidate copies the input shape
and replaces its final dimension with N. Rank-one, multi-row, empty
and mismatched inputs retain the original helper. No shader, weight,
binding or accumulation changes are made; neither previous candidate
is stacked.

V1 fails compilation because a logging macro needs explicit braces
under the existing dangling-else warning gate; that directory is
preserved. V2 builds with a byte-identical parent relink, provider
SHA-256 `6aa1c4f90d139752c96ffda1e9219493adb9841be06ce19d4893b16d813ac36a`.
Three focused shape/source/model tests pass. Original native fixtures
pass 116,582,784 finite bitwise comparisons, including observed
single-row branch counters and unchanged multi-row fallback.
Another 36 synthetic fixtures pass 24,543,744 comparisons:
rank-one fallback and rank-two/three/four fast outputs, pending limits
1/64 and 36 invalid-selector rejections.

Full 2K coverage passes original 36-token prefixes, workers, faults
and hashes. It observes all 172 shape kernels taking the fast path,
12,040 fast output-shape calls, and 171 dense replay instances with
11,799 hits. Initial paired full/retained 2K timing is pending.
The qualified provider and all-context throughput claims remain
unchanged; 100 tok/s has not been established.

Output-shape V2 passes initial and fresh paired 2K gates:

| Screen | Validation | Decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---|---:|---:|---:|---:|
| Initial | Full | 77.769053 | 3.485248 | 1.011652577 | 3/3 |
| Initial | Retained | 78.427379 | 3.485911 | 1.016982643 | 3/3 |
| Fresh | Full | 77.467010 | 3.492543 | 1.013057057 | 3/3 |
| Fresh | Retained | 78.162412 | 3.487321 | 1.009537799 | 3/3 |

These are matched per-request ratios, not ratios of displayed medians.
The fresh full bracket includes a slow 75.643215 tok/s after-control.
The fresh brackets share the initial ending controls; they are not
independent estimates of a fixed percentage improvement.

Before escalation, an additional full/retained 2K net screen compares
the candidate directly with the unchanged original qualified provider
`0955e1d6cc30cac13044450f692100ffa5f33361aaef229e25ff0efa20a8c8d4`.
That screen is pending. Same-provider flag-off/on support alone does not
establish net benefit over the original provider or any 8K/32K gain.

The original-provider net check rejects V2 promotion:

| Validation | Candidate decode tok/s | Mean TTFT s | Net matched ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 77.679583 | 3.488985 | 0.991577548 | 1/3 |
| Retained | 78.534415 | 3.480994 | 1.003504183 | 3/3 |

Both full original-provider controls are healthy (77.820053 and
77.748050 tok/s). V2 stays off; its paired flag-off/on support is not
promoted as net throughput improvement. No 8K/32K escalation is started.

A separate refinement uses `ORT_WEBGPU_DENSE_OUTPUT_SHAPE_STATIC`.
This new flag is validated and read only when the kernel/session is
created; changing it requires a new model/session. Its disabled branch
retains the original helper, output-shape copy and allocation scope;
the enabled single-row branch directly copy-constructs the shape.
This removes the experimental hot flag parsing without changing
arithmetic, bindings or the old V2 flag's behavior.

The private session-static provider builds with byte-identical parent
relink, SHA-256
`67e3e2dd87495e8f2c23c831f9e8ceff8e647d8008f9e0c0498765252e10ee57`.
Four focused tests pass. A dual-session native comparator constructs
separate flag-zero and flag-one sessions and rejects invalid flag-two
at construction. Original fixtures pass 116,582,784 finite bitwise
comparisons; the reused 36 rank fixtures pass another 24,543,744,
including rank-one and multi-row fallback with pending limits 1/64.
Full 2K frozen-prefix coverage again observes 172 fast shape kernels,
12,040 fast calls and 171 replay instances with 11,799 hits.
Its first full/retained performance screen uses original-provider
controls directly, rather than an experimental flag-zero baseline.
The initial net screen fails support in both modes:

| Validation | Candidate decode tok/s | Mean TTFT s | Net matched ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 77.467396 | 3.494948 | 1.004934079 | 2/3 |
| Retained | 77.928474 | 3.489219 | 0.988839414 | 1/3 |

The full ratio includes a slower ending control (76.568774 tok/s).
Session-static output shape stays off; no unchanged repeat or 8K/32K
escalation is started. The qualified provider remains unchanged and
100 tok/s remains unestablished.

### Ready node-replay key copies (experimental, default off)

The native adapter's per-operator input/output tensor vectors are a
broader allocation lead, but changing their storage requires rebuilding
all 152 linked provider objects. Only 28 exact compile commands were
recoverable from their manifests and the current CMake database; that
layout-changing experiment was held before implementation. No partial
ABI rebuild or symbol-interposition shortcut was used.

The exact active V8 context exposes a narrower allocation: node replay
copies `CapturedCommandInfo::program_key` into deferred storage, despite
already requiring a ready pipeline and an absent pending build.
`WaitForDeferredPipelineBuilds` skips such ready pipelines before looking
up a key. `ORT_WEBGPU_READY_REPLAY_KEY_TRIM=1` now constructs an unkeyed
copy only for unprofiled node replay. It preserves pipeline/bind-group
references, uniform offset, dispatch dimensions, indirect buffer,
dispatch ordering and the original pending64 batching. Profiling,
unresolved pipeline builds, graph capture and cached prototypes retain
their original paths and keys.

This selector defaults to zero and is process-static: export it before
the first GPU dispatch and use a fresh process to change it. Invalid
values are rejected on that first dispatch. The isolated experiment
rebuilds only the context object, with no header/layout changes and a
byte-identical parent relink.

Sixteen focused mechanism, model-wiring and existing-builder tests pass.
A compiled extraction of the actual copy helper preserves every dispatch
field and reduces 1,000 long-key copy allocations to zero. V1's first
native run completed its comparisons but failed the explicit FP32-option
log verifier; it was preserved unqualified. V2 uses the existing explicit
FP32-off comparator transform, while retaining its original block32
comparison selector. V2 provider SHA-256:
`e96ea8215f08485ec4d0f449f0e7ae8f7e5496aea08f8635572d6f5560de0b04`.

Separate process-zero and process-one runs pass 233,165,568 finite
bitwise comparisons across the 27 changing-input fixtures, pending1/64,
multi-row fallback and original invalid block32 rejection. A fresh
process also explicitly rejects the new selector's invalid value two.
Actual full 2K frozen-prefix coverage observes 171 dense replay instances,
11,799 hits and 10,714 binding hits; the new context records 39,909
unkeyed ready commands and 3,498,975 key bytes not copied. Coverage is
not standard throughput or whole-model quality evidence.

Initial full/retained natural 2K timing against the original qualified
provider completed:

| Validation | Candidate median decode tok/s | Mean TTFT s | Matched ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 77.566390 | 3.492876 | 1.008959598 | 3/3 |
| Retained | 78.150815 | 3.489962 | 1.005041405 | 2/3 |

The full before-control was slower (76.592477 median tok/s) than its
ending control (77.918865); retained controls were 77.386597 and
78.378724. Retained per-request ratios were 0.998046648, 1.007148918
and 1.009967341. The required both-mode 3/3 support failed, so this
candidate remains held/default off, without unchanged repetition or
8K/32K escalation. Same-provider flag-off timing is not substituted
for the original-provider net check.

### Adapter tensor-vector reuse (held before GPU build)

A separate default-off `ORT_WEBGPU_ADAPTER_TENSOR_VECTOR_REUSE`
prototype preserves the context's vector fields and layout. It swaps
empty capacity from a thread-local cache into the constructor, then
clears every tensor wrapper before returning capacity in the destructor.
Compiled actual-constructor/destructor tests cover variable counts,
nested contexts, stale-slot clearing, error cleanup, invalid selectors
and unchanged context size. After warmup, 2,000 allocations become zero;
the combined related test suite passes 19 tests. This is CPU mechanism
evidence, not measured model throughput.

Compiler provenance was recovered for all 152 linked objects, including
ancestor singular compile-command fields and retained generated Make
rules. The old generated PCH explicitly includes the FP8-checkpoint
SDK adapters, not the main H200 SDK adapters. Prioritizing both the
canonical SDK root and its `core/session` include directory avoids
duplicate C API definitions. A fresh canonical preinclude reproduces
the original control-flow object byte-for-byte, but not all objects.

The complete unchanged-source rebuild gate passes the first allocator
and buffer-manager objects, then fails on the compute-context object:
rebuilt `a1550b69d1f0c68067656f432b07586a2421c87636e33f88a721980c6f198117`,
original `ba555d7fba5e908527ce28cb22149f19c2744d1d5e3ee2d83554b8328f04ff81`.
Its current and frozen source hashes still match:
`89f980f68d6758a905edb6557fb4db31c14b0f1ec854ecceb5623ecd74888e31`;
the buffer-manager sources also match:
`062cdae5766de4a0fe3d08b9c6ec22dbe4fcc809a7cc0b7499b796b393c6cc51`.
SDK/PCH equivalence remains unresolved. The gate was not weakened:
no changed adapter header or candidate provider was produced, no mixed
inline definitions or interposition were used, and no GPU timing ran.

### GDN projection-preparation replay (experimental, default off)

`ORT_WEBGPU_GDN_PREP_NODE_REPLAY` is an independent constructor-scoped
0/1 selector. A private kernel implementation preserves the public
`GdnProjPrepF16` registration and the existing fused program, uniforms,
dispatch and arithmetic. Only B1/L1 wraps the original program in a
private copy of the qualified node-replay helper, binding all five
inputs and five mixed FP16/FP32 outputs. Zero-length state copying,
batch/multi-row execution and profiling retain their original paths.
The existing global node-replay selector must also be enabled.

V1 lacked the active replay API include path and failed compilation;
it was preserved without timing. V2 uses the exact hash-checked V8
replay API headers and the original causal-convolution compile recipe.
The unchanged original object rebuild and original-provider relink are
byte-identical. V2 provider SHA-256:
`464b476bfc25311494a866e494d8a2eb6e98a6f5923278c540869b1c576d8d75`.
Native qualification passes 181,043,200 finite bitwise comparisons
across seven changing-input fixtures, all five mixed-type outputs and
both pending1/64 windows for every fixture. Three live GPU buffer banks
keep all five inputs at distinct nonempty addresses; reference and
candidate use different banks while candidate activations/state alternate
between two live banks. Each single-row session records 128 requests,
one capture, 127 hits, one rebind and 126 binding hits. Lengths
0/2/3/4/16 and B2/L1 retain their fallback paths. Constructor value two
is rejected; separate fresh-process checks also reject it and preserve
the original bad-state-width/channel errors in both selector modes.

Full 2K frozen-prefix coverage observes all 36 private GDN preparation
replay instances, 2,484 hits and 2,048 binding hits, while preserving
171 dense replay instances, 11,799 hits and 10,714 binding hits.
These 36-output coverage requests are not standard TPS or whole-model
quality evidence. The combined related suite passes 25 tests, including
numeric 3/3 request-support and 2% TTFT thresholds. Original-provider
full/retained 2K timing completed without the required request support:

| Validation | Candidate median decode tok/s | Mean TTFT s | Matched ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 76.777507 | 3.485244 | 1.003503908 | 2/3 |
| Retained | 78.351951 | 3.479828 | 1.005324383 | 2/3 |

Full control medians drifted from 77.821392 to 74.550722; retained
controls were 78.460401 and 78.239262. A CPU provenance probe overlapped
the screen, so this is not promotion evidence even aside from the failed
3/3 criterion. This family remains held/default off without unchanged
repetition or long-context escalation.

### Complete adapter rebuild provenance (continuing)

The initial compute-context mismatch was diagnostic `__FILE__` strings,
not an arithmetic change. A canonical-SDK macro-path mapping reproduces
its original object byte-for-byte. A later complete pass identifies one
additional SDK difference: the main SDK adds only the existing nonvirtual
`ProfilerImpl()` accessor. A scoped complete SDK copy retains that exact
accessor; copying the full header tree also ensures relative includes
actually reach the changed adapter header rather than bypassing an
include-only overlay.

The original N96 matmul source is recovered without editing the worktree
by removing later FP32-decode-only additions in a scoped copy. Its exact
manifest hash matches:
`8bd085a685d1284235b8a7bc33ad53f29403ad906817363c526d3272783b93a1`.
SDK diagnostic paths differ by original source family, so mappings retain
the original main or FP8-checkpoint path for each object. These changes
reproduce the first 144 linked objects exactly.

The remaining subgroup object also requires the active uniform-arena
WGSL-generated header, not the older fallback-generated header selected
by the recovered base recipe. An isolated probe using that exact header
and the current source reproduces original object
`260271f9c54024d52aa00a411939a4359af951ffdc403544c349cbcf93c4f168`
byte-for-byte. No source or shader simplification was accepted.
V7 now passes the complete 152-object byte-identical baseline gate and
original-provider relink before writing the changed header. All 152
candidate objects are then rebuilt using a fully wired SDK copy.
Provider SHA-256:
`4f75b29bebb3b78f71aa8a73dbe3dee5140d937328089165e4878c8cc09c43ee`.

Separate process-zero/process-one native tests pass 237,408,768 finite
bitwise comparisons, including changing-input dense fixtures and the
original expanded GDN reference with all five mixed FP16/FP32 outputs.
A fresh process explicitly rejects selector two. Enabled adapter
statistics require balanced requests/returns, positive capacity reuse
and zero retained tensor elements at exit.

Full 2K frozen-prefix coverage records 60,902 requests and returns,
with 60,901 nonempty-capacity reuses. Existing 171 dense replay instances,
11,799 hits and 10,714 binding hits remain intact. Thirty related tests
pass. Original-provider full/retained natural 2K brackets completed;
no CPU compilation ran during this screen. Native/coverage/allocation
proofs are not TPS or whole-model quality qualification.

| 2K validation | Candidate median decode tok/s | Mean TTFT s | Matched ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 77.826771 | 3.481544 | 1.010966290 | 3/3 |
| Retained | 77.946637 | 3.494252 | 0.995429545 | 0/3 |

The retained mode fails support, so pooled vectors remain held/default-off.
There is no unchanged repeat or 8K/32K escalation.

### Inline adapter tensors (independent, default-off)

`ORT_WEBGPU_ADAPTER_INLINE_TENSORS` selects existing ORT Abseil
`InlinedVector<Tensor, 8>` inputs and `InlinedVector<Tensor, 5>` outputs.
Larger counts retain normal heap fallback. Mode zero keeps the original
vector access, constant-prepacked input paths and optional/required guards.
Unlike pooling, this expands `OpKernelContext` layout: all 152 candidate
objects must rebuild coherently after the byte-identical original-object
and original-provider relink gates. There is no thread-local tensor pool.
Selector two is rejected; mode one also explicitly rejects builds with
`DISABLE_ABSEIL`, whose container alias lacks inline storage.

CPU tests use the actual transformed constructor and real inline containers.
Within capacity, 1,000 contexts use zero allocations instead of 2,000;
9 inputs/6 outputs use exactly two allocations. Variable and zero counts,
nested contexts, stale-slot isolation and compute-error cleanup pass.
Ten focused allocation/provenance/wiring tests pass, including selector
defaults and the no-Abseil guard.

V1 reproduces all 152 original objects but candidate compilation rejects
an unbraced logging macro under `-Werror=dangling-else`. V2 corrects the
repository-standard logging use and rebuilds independently; the failed
V1 output remains separate. V2 passes all original-object/relink gates and
the complete expanded-layout candidate rebuild. Provider SHA-256:
`2832ed1736ee780e2308ed35f2889e68b19b5ce349eb7c6fa9fbf64947535698`.
Separate mode-zero/mode-one native runs pass 237,408,768 finite bitwise
comparisons, including changing inputs, pending1/64 and all five GDN
FP16/FP32 outputs against the expanded reference. A fresh process rejects
selector two.

Actual full 2K frozen-prefix coverage records 60,902 enabled contexts,
14,060 input heap fallbacks and zero output heap fallbacks. Existing
171 dense replay instances, 11,799 hits and 10,714 binding hits remain
intact. This proves the inline path runs, not a TPS or whole-model quality
gain. Original-provider full/retained 2K brackets are active.
Optional `ORT_WEBGPU_ADAPTER_INLINE_STATS_FILE` counters
are enabled only for path coverage and native proofs, never standard
timed model commands. Original-provider controls and repeated 2K/8K gates
remain mandatory before 32K escalation.

The initial 2K inline screen passes both modes:

| Validation | Candidate median decode tok/s | Mean TTFT s | Matched ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 77.873541 | 3.483322 | 1.025155684 | 3/3 |
| Retained | 78.430154 | 3.484825 | 1.005236352 | 3/3 |

Full original-provider controls drift from 77.734251 to 74.339897 median
tok/s, inflating its apparent gain. Retained controls are
78.134575/78.347810. Fresh confirmation is active and reuses these ending
controls as its beginning controls; the two brackets are not independent.
Fresh confirmation fails support:

| Validation | Candidate median decode tok/s | Mean TTFT s | Matched ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 77.547466 | 3.487610 | 1.010982765 | 2/3 |
| Retained | 75.675140 | 3.482192 | 0.966827601 | 0/3 |

The capacity-eight provider stays held/default-off. There is no unchanged
repeat or 8K/32K escalation.

### Inline input capacity 17 (measured fallback follow-up)

Read-only frozen-graph inspection finds 851 nodes, maximum 17 inputs and
five outputs. Above-capacity input operators are 12 DynamicSparseAttention
nodes (17), 48 QMoE nodes (14), 48 MlpInputProjectionsF16 nodes (9),
46 HCSharedPostNormProjectionF16 nodes (9), and 36 GatedDeltaNet nodes (9).
These 190 nodes explain the 14,060 observed input heap fallbacks.

A separate V3 provider raises only input inline capacity from eight to
17; output capacity remains five. This tests eliminating the remaining
input allocation rather than repeating the failed provider unchanged.
Counts above capacity still use normal heap allocation. CPU tests require
zero allocations for 1,000 contexts within the new capacity and exactly
two allocations for 18 inputs/six outputs, with the same nesting, guards
and error cleanup checks. Complete layout-consistent rebuild, separate
native modes, coverage counters and original-provider performance gates
remain required; no V3 TPS or whole-model quality claim is made.

V3 passes the complete original-object/relink proof and isolated candidate
build, then 237,408,768 finite bitwise native comparisons. Full 2K coverage
records 60,902 enabled contexts with exactly zero input/output heap
fallbacks and unchanged dense171/11,799-hit/10,714-binding-hit coverage.

| 2K validation | Candidate median decode tok/s | Mean TTFT s | Matched ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 77.750149 | 3.480388 | 1.002854765 | 2/3 |
| Retained | 78.485956 | 3.482040 | 1.013729181 | 3/3 |

Full-mode consistency fails; V3 remains held/default-off, with no
unchanged repeat or long-context escalation. Eliminating these allocations
does not establish a reliable throughput gain.

### Dense runtime flag reads (profile-derived, default-off)

The original-row4 CPU profile attributes 8.54% inclusive samples to integer
environment parsing (overlapping CPU samples, not wall-time shares).
The exact qualified MatMulNBits compute path reads eight distinct integer
selectors repeatedly. `ORT_WEBGPU_DENSE_FLAG_CACHE` tests a private
implementation with lazy, per-kernel, thread-safe `std::call_once` values.
Mode zero preserves dynamic reads. Mode one freezes each selector after
its first use for that kernel; unused selectors remain unparsed and
parse exceptions retry normally. Change settings using a fresh session.
The public class, original compute implementation, registration identity,
shader arithmetic and replay names remain unchanged.

CPU validation checks dynamic reads, concurrent first reads, independent
kernel values and parse-error retry. Native comparison must use separate
reference/candidate sessions and a fresh invalid-selector session, rather
than falsely expecting frozen flags to refresh within one kernel. Optional
flag-read statistics are diagnostic only and must remain off during timing.
V1 checks all 152 retained original objects against their byte-identical
rebuilds, then freshly reproduces the original dense object and original
provider relink before replacing that single translation unit. The first
preflight exposed relative object paths; resolving them against the
original link working directory fixes the check without weakening it.
Provider SHA-256:
`dcbce3a8213bb275a4671c30da96fd0858619121a56a5f2e3a230f263dc6e090`.
Four CPU/source/wiring tests pass.

Separate process-zero/process-one native tests pass 237,408,768 finite
bitwise comparisons. Each fixed-flag dense session requires one replay
entry/capture and 127 hits over 128 changing-input rounds. Fresh sessions
reject invalid dense selectors, and a fresh process rejects cache selector
two. Full 2K frozen-prefix coverage preserves dense171/11,799 hits/10,714
binding hits. Across 172 kernels, 75,332 flag requests require 1,018 parses.
These counters prove the mechanism, not TPS or whole-model quality.
Original-provider full/retained 2K brackets complete with statistics
disabled and no CPU compilation overlap:

| Validation | Candidate median decode tok/s | Mean TTFT s | Matched ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 77.269983 | 3.503022 | 1.004775007 | 2/3 |
| Retained | 78.478740 | 3.483906 | 1.001921135 | 2/3 |

Both consistency gates fail. V1 remains held/default-off, with no
unchanged repeat or 8K/32K escalation. The full controls drift from
74.991156 to 77.858253; retained controls are 78.202566/78.346066.

### Steady-state 32K attribution (diagnostic, not TPS)

Caller attribution requires the recorded container paths and exact libc/
Conda libraries; unresolved host-only reports are excluded. The restored
original decode trace places 2.37% inclusive CPU samples under dense
integer parsing and 2.02% under QMoE integer parsing. These are overlapping
CPU sample percentages, not wall-time or expected throughput gains.
Dense replay hits return before invoking the helper callback, so simply
caching helper-body flags would not remove repeated work on that route.

An independent original-provider diagnostic profiles exactly output 256,
after 255 outputs, at 32K context rather than profiling prefill or initial
decode setup. Warmup and measured requests must each retain all 256 frozen
IDs, zero major faults and exactly one profiled decode step. Two tests
validate command shape and rejection of incorrect profile boundaries or
token counts. Profiling remains explicitly instrumented/not a TPS or
whole-model quality claim; no rejected optimization is promoted.

The completed original-provider 2K and 32K measured-request profiles have
512 checked frozen IDs per context (warmup plus measured). Selected totals
for the final decode step are below; these are sums of instrumented GPU
kernel durations, not natural decode wall time or a throughput result.

| Program family | Kernels per decode | 2K us | 32K us |
|---|---:|---:|---:|
| HCPostNormProjectionF16 | 94 | 2726 | 2745 |
| MatMulNBitsBlock32F16 | 159 | 2667 | 2664 |
| QmoeFC1SwiGLUBitcastF16 | 48 | 1140 | 1130 |
| QmoeFC2FinalMixBitcastF16 | 48 | 1101 | 1090 |
| HCUpPreMixPacked2F16 | 97 | 1166 | 1163 |
| DynamicSparseAttentionQueryCache | 12 | 1044 | 1058 |
| MlpInputProjectionsF16 | 48 | 795 | 799 |
| GatedDeltaNet | 36 | 729 | 722 |
| QmoeHierarchicalGate1Token | 48 | 652 | 648 |
| QSA parallel rank | 12 | 133 | 566 |

### Ordered dense K-tile unrolling (experimental, default off)

`ORT_WEBGPU_DENSE_ORDERED_UNROLL2=1` selects an isolated shader experiment.
The selector is frozen on first block32 program construction in a process;
use a fresh process to change it. Zero preserves the original block32
template and program identity. Other values fail explicitly when that
program is constructed. Existing compute eligibility and fallback logic
are unchanged.

The active shader comes from the retained custom
`dense_block32_template.inc`, not the normal generated WGSL template.
Mode one emits two consecutive K-tile bodies per outer iteration, retaining
each original body verbatim, including its FP16 accumulation boundaries
and both workgroup barriers. The second tile is guarded only by uniform K,
so odd tile counts and partial tiles retain their original bounds checks.
The distinct `MatMulNBitsBlock32F16Unroll2` program identity separates shader
cache entries. Optional shader-generation statistics must be disabled
during timing; they count generated shaders, not all dispatches or TPS.

The isolated single-translation-unit build rechecks all 152 original
objects and the canonical SDK proof, freshly reproduces the original dense
object and original provider relink, and verifies the retained custom
template hash before building the candidate. Four CPU tests pass,
including compilation of the actual template emitter and exact comparison
of both emitted tile bodies and 2K/8K/32K timing-command wiring.
Separate process-zero/process-one native runs pass 237,408,768 finite
bitwise comparisons, including multi-row fallback, GDN regression and fresh
invalid-selector rejection. Full 2K frozen-prefix coverage preserves dense
171/11,799 hits/10,714 binding hits with zero major faults. It records one
actual new shader generation, with K-tile size 1024; this is not a dispatch
count. Provider SHA-256:
`2bf0730840c06023fd2b17a3956785bf9f66cadf2444540160d3a1e31a484830`.
Original-provider full/retained 2K timing brackets are required before any
repeat or long-context escalation; no throughput gain is established yet.

The initial original-provider brackets complete with shader-generation
statistics off and no CPU compilation overlap:

| Validation | Candidate median decode tok/s | Mean TTFT s | Matched ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 77.510438 | 3.487405 | 1.014131484 | 3/3 |
| Retained | 77.865144 | 3.521872 | 0.995415809 | 0/3 |

Retained validation fails consistency. The candidate remains held/default
off, with no unchanged repeat or 8K/32K escalation. Full controls drift
75.662474 to 77.376293; retained controls are 78.425257/77.969577. These
results do not establish a throughput improvement or whole-model quality.

### HC projection component unrolling (experimental, default off)

`ORT_WEBGPU_HC_PROJECTION_INNER_UNROLL=1` explicitly emits the four packed
INT4 component steps in the fused HC projection shader. Each step retains
the original nibble decoding, FP16 scale multiplication and ordered
`sum += dot(x0, w0) + dot(x1, w1)` statement. Separate scopes provide
constant component indices without changing the ten-tile loop, FP16
tile accumulation, padding, RMS reduction tree, barriers or output split.
The selector is frozen at first fused HC program construction per process;
fresh processes are required to change it. Invalid values fail at that
construction point. Zero retains the original shader text and identity;
one uses `HCPostNormProjectionInnerUnrollF16`. There is no additional
per-decode environment parsing before replay hits.

The isolated build checks all 152 original objects and SDK hashes, then
freshly reproduces the original HC translation unit and original0955
provider relink before changing that single translation unit. Public and
private object layouts remain unchanged. Provider SHA-256:
`6213408d13df8afdc48ff12108af2e5d4f419b81ed118a00c040a2d458bd817c`.
Four tests pass, including compilation of the actual shader renderer for
all shared/nonshared scale combinations and exact default-mode WGSL
equality. Native proof generation initially rejected a premature wrapper
entrypoint before any GPU work; the generator/entrypoint were corrected and
an AST-equivalence/order regression test added.

Separate process-zero/process-one native runs pass 1,214,287,872 finite
bitwise comparisons across all four outputs and fallback fixtures,
including changing inputs/splits, pending1/64, replay eviction and fresh
invalid-selector rejection. There are 96 fused replay requests per
eligible fixture; the original projection selector still controls the
baseline reference, rather than switching the process-static new selector
inside one session.

Full 2K frozen-prefix coverage passes with zero major faults and two
actual new shader generations. All 94 HC replay instances are checked:
6,580 requests, 6,486 hits and 4,888 binding hits. Dense coverage remains
171/11,799 hits/10,714 binding hits. Shader-generation counts are not
dispatch counts or TPS. Original-provider full/retained 2K brackets are
running with the new counters disabled and no CPU compilation overlap.
No throughput gain or whole-model quality claim is established.

The completed original-provider 2K brackets reject component unrolling:

| Validation | Candidate median decode tok/s | Mean TTFT s | Matched ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 77.810239 | 3.487228 | 1.001037309 | 2/3 |
| Retained | 75.166896 | 3.487220 | 0.982179958 | 1/3 |

Both consistency gates fail. Full controls are 77.778438/77.496513;
retained controls drift 78.252133 to 74.918190. The candidate remains
held/default-off, with no unchanged repeat or 8K/32K escalation.

### Static HC RMS-tree emission (experimental, default off)

`ORT_WEBGPU_HC_RMS_TREE_UNROLL=1` emits the original eight width levels
(128, 64, 32, 16, 8, 4, 2, 1) as explicit constant-width scopes. Each scope
preserves the four stream additions, FP32 shared-memory stores and
workgroup barrier. The projection loop, RMS input accumulation,
normalization, output split and fallback paths remain unchanged.
Zero retains the original shader text/identity; one uses
`HCPostNormProjectionRmsTreeUnrollF16`. The selector is frozen on first
fused program construction per process, with explicit invalid-value
rejection. Optional shader-generation counters must remain off during
timing.

A shared single-translation-unit builder reuses the exact 152-object/SDK
proof, freshly rebuilds the original HC object and original0955 relink,
and prevents experiment metadata from overriding provenance assertions.
Prior completed builders/runners remain unchanged. Four RMS-specific
tests pass, including actual renderer/default-mode equality, exact
eight-level body/barrier order, 2K/8K/32K command wiring and rejection of
missing HC replay nodes or binding hits. The model coverage gate now
explicitly requires all 94 HC replay instances as well as dense171.
Native/model/performance qualification remains required; no speedup is
established.

Separate process-zero/process-one native runs pass 1,214,287,872 finite
bitwise comparisons, four-output/fallback checks, replay eviction and
fresh invalid-selector rejection. Full 2K frozen-prefix coverage passes
with zero major faults, two actual new shader generations and explicit
HC94/6,486 hits/4,888 binding hits plus dense171/11,799/10,714.
Provider SHA-256:
`4cf077e99f53bfaf8e6b9451170b8e7ab43104fc5d3f17c714e1871296ae6b92`.

The initial original-provider 2K brackets pass both request-consistency
and TTFT gates, with shader counters off and no CPU compilation overlap:

| Validation | Candidate median decode tok/s | Mean TTFT s | Matched ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 78.061178 | 3.482675 | 1.022557399 | 3/3 |
| Retained | 78.485074 | 3.483610 | 1.017709190 | 3/3 |

These results are provisional. Both ending controls slow: full
77.783743 to 75.001557, retained 78.494548 to 75.725188. Fresh confirmation
reuses those ending controls, so its brackets are not independent.
No promotion, 8K/32K candidate result or 100 tok/s claim follows from the
initial screen alone.

Fresh 2K confirmation also passes both consistency/TTFT gates:

| Validation | Candidate median decode tok/s | Mean TTFT s | Matched ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 77.636117 | 3.489893 | 1.016216591 | 3/3 |
| Retained | 78.485080 | 3.480969 | 1.012609161 | 3/3 |

The shared slow controls remain material: fresh full brackets are
75.001557/77.717733, retained 75.725188/78.097383. One retained matched
request is only 1.000324573. The almost-equal initial/fresh retained
medians come from distinct raw-file hashes and distinct measured decode
durations, not reused candidate samples. This is 2K evidence only.
Initial/fresh 8K qualification is now eligible; 32K remains gated on
repeated 8K support, and 100 tok/s/whole-model quality remain unestablished.

The initial 8K screen does not support escalation:

| Validation | Candidate median decode tok/s | Mean TTFT s | Matched ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 77.707704 | 14.194691 | 1.003950404 | 2/3 |
| Retained | 78.779078 | 14.169179 | 0.997369214 | 2/3 |

Full control medians are 77.722550/77.675086; retained are
78.631074/78.168731. Both modes miss three-of-three request support;
retained matched decode also regresses. TTFT gates pass. No unchanged
8K repetition or 32K escalation was run. The candidate remains held and
default-off; the original qualified provider is unchanged.

### HC projection shared padding34 (experimental, default off)

`ORT_WEBGPU_HC_PROJECTION_PADDING34=1` changes only the shared-memory
address stride from 36 to 34 half-elements per 32 logical values and
the corresponding array size from 11520 to 10880. Original arithmetic,
rounding, reductions, barriers, output splits, admission and fallback
paths remain unchanged. The separate shader identity is
`HCPostNormProjectionPadding34F16`; zero preserves original shader text.
The process-static selector explicitly rejects values other than zero
and one. This candidate does not stack RMS-tree or component unrolling.

For scalar half-element projection loads, the predicted 32-bank word
mapping changes from 18 times lane to 17 times lane modulo 32, reducing
two-way conflicts to a bijection. This is a layout hypothesis, not proof
of generated instructions or a performance gain; reduced vector
alignment could offset it. Full address-range/mapping and actual C++
renderer tests, original object/SDK/relink provenance, separate-process
native comparisons and frozen-model/performance gates remain required.

Four targeted tests pass: actual shader text differs only in array
length/address stride for all four shared/scale variants, exhaustive
logical/projection address and predicted bank-word mappings, unchanged
Compute/admission/fallbacks, and native/model/2K-32K proof wiring with
HC94/dense171 coverage gates. The isolated build rechecks all 152
original objects and canonical SDK hashes, freshly reproduces the
original HC object and original0955 relink, and replaces only that
translation unit. Candidate provider SHA-256:
`6fbc20a8a12ac0364a44faff3440277d5c926369cf09e25e7eab29cb4a1036c0`.
Native qualification followed by frozen 2K coverage and gated timing is
running serially on GPU7; no performance gain is established.

Qualification completed: separate process0/process1 and fresh invalid2
pass 1,214,287,872 finite bitwise comparisons. Full 2K frozen coverage
passes HC94/6,486 hits/4,888 binding hits and dense171/11,799/10,714 with
zero major faults. Initial timing rejects escalation:

| Validation | Candidate median decode tok/s | Mean TTFT s | Matched ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 77.814846 | 3.486235 | 1.003069813 | 3/3 |
| Retained | 78.062277 | 3.487585 | 0.993595254 | 0/3 |

Both TTFT gates pass; retained decode regresses. No unchanged repetition
or 8K/32K escalation was run; keep off. Follow-up prior-art inspection
finds the earlier `ORT_WEBGPU_HC_POST_NORM_PAD34` tested the same shared
layout on the parallel2 parent with neutral/negative controlled windows.
This source/SDK-qualified process-static implementation on original0955
reconfirms a held layout; it is not a novel bank-layout optimization.

### HC RMS vector layout (experimental, default off)

`ORT_WEBGPU_HC_RMS_VECTOR_LAYOUT=1` packs the four independent FP32
RMS reduction streams into `array<vec4<f32>, 256>` instead of four
256-element scalar regions. Each vector component preserves its original
sum, eight-level tree and FP32 addition order; all barriers are unchanged.
Storage remains 4096 bytes. Scalar FP16 post-values, their stride36,
normalization expressions, projection, output splits, admission and
fallbacks are unchanged. This differs from the rejected FP16 activation
vec4-storage, subgroup RMS, shuffle-fold and fused-sum experiments.

Zero preserves the original shader text and identity; one uses
`HCPostNormProjectionRmsVectorLayoutF16`. The selector is process-static
and explicitly rejects invalid values. The intended improvement is
vector shared-memory instruction generation, not a promised bandwidth
or tok/s gain. Actual renderer/tree-layout, native bitwise, frozen-model
and gated natural-performance qualification remain required.

The initial renderer compile check caught a missing C++ stream insertion
after the conditional array declaration, before any build/GPU execution.
After correcting it, all four targeted tests pass: actual default shader
equality and component-wise tree transformation across all shared/scale
variants, full 4096-byte component mapping/tree edges, unchanged
projection/Compute/admission/fallbacks, and native/model/2K-32K wiring.
The isolated single-source build passes all original 152-object and SDK
hash checks, fresh original HC object reproduction and original0955
relink. Candidate provider SHA-256:
`365192cc1fc30361f761ffddacde477666dfad54caed62fdf2f8ea345f636cc8`.
Native0/1/invalid2, full frozen 2K coverage and conditional initial/fresh
2K timing are running serially on GPU7. No speedup is established.

Native0/1/fresh-invalid2 qualification passes 1,214,287,872 finite bitwise
comparisons. Full frozen 2K coverage passes HC94/6,486 hits/4,888 binding
hits and dense171/11,799/10,714 with zero major faults. Both initial
screens pass, but fresh confirmation fails:

| Screen | Validation | Median decode tok/s | Mean TTFT s | Matched ratio | Positive requests |
|---|---|---:|---:|---:|---:|
| Initial | Full | 78.117486 | 3.484589 | 1.005383144 | 3/3 |
| Initial | Retained | 78.722393 | 3.488711 | 1.003343385 | 3/3 |
| Fresh | Full | 74.685679 | 3.481057 | 0.964577102 | 0/3 |
| Fresh | Retained | 77.515463 | 3.481668 | 0.989501938 | 0/3 |

Fresh brackets reuse initial ending controls: full 77.517343/77.321473,
retained 78.450249/77.944740. All TTFT gates pass. The failed fresh
decode results are retained; their cause is not established. Keep off
without 8K/32K escalation or another unchanged repetition.

### Static decoded weights in grouped row4 prefill (experimental, default off)

Expert sorting and four-row reuse are already implemented and active in
the qualified original0955 provider; earlier statements that grouping
remained unimplemented are historical, not its current state.
`ORT_WEBGPU_QMOE_ROW4_STATIC_WEIGHTS=1` instead replaces the private
eight-vector decoded-weight array with eight named FP16 vector variables
and emits both four-word loops statically. It preserves expert-change
caching, row order, 1024-element K tiles, every FP16 dot/update, serial
fold, bias, barriers and all admission/Compute/fallback code.

Zero retains original shader text/identity; one uses
`MatMulNBitsGroupedRows4StaticWeightsF16`. The selector is process-static
and rejects invalid values. The private shader emitter receives the
unchanged tensor/uniform metadata. Native comparator changes are limited
to a new profiling prefix so prior evidence is not overwritten.
This targets prefill TTFT and possible compiler scalarization, not a
decode arithmetic change or a promised decode-TPS gain. Native, frozen
coverage, consistent TTFT gains and decode-regression gates remain
required before long-context escalation.

Twenty source/shared-renderer tests pass, including exact default shader
equality, all eight weight/index pairs, original ordered dot updates and
barriers, unchanged Compute/admission/fallbacks, comparator-prefix-only
changes and all four existing HC renderer consumers. The shared CPU
renderer accepts an optional program class/constructor argument list;
its original defaults are unchanged. The isolated build passes original
152-object/SDK checks, fresh original matmul object reproduction and
original0955 relink. Provider SHA-256:
`31116693bd9c53d26ff39fa23d285b70405daeba194d2ba7785aa09b4e299fa6`.

Separate process0/process1 native comparison and fresh invalid2 checks
are running on GPU7, preserving the six immutable fixture suite,
chunk16 references, pending1/64 and actual grouping/row4 dispatch gates.
Each native mode must prove 300,810,240 finite bitwise comparisons;
these are pending, not claimed successes. Profile filenames are
checked disjoint across modes.

Four additional model/gate tests pass. Prefill support requires all
three matched-request TTFT ratios below one in both validation modes
and a matched natural-decode ratio at least 0.99, using the existing
one-percent inactive-route regression-screen bound. Exact equality,
the decode threshold boundary, prompt/output mismatch and non-finite
metrics are tested. It does not require or infer a decode speedup.
Full 256-output natural runs, excluded warmup, original0955 controls,
frozen HC94/dense171 coverage and source/worker/major-fault gates are
preserved. Initial/fresh 2K precede 8K and then 32K; failure stops
unchanged repetition/escalation. No new performance result is established.

Both native modes now pass 300,810,240 finite bitwise comparisons each,
with six original layers, pending1/64, retained chunk16 references and
32 actual row4 plus 16 of each grouping program per fixture.
The first fresh invalid-selector probe omitted the qualified native
prefill/fusion/bitcast flags and hit a rows128 reference mismatch before
reaching the eligible row4 constructor. That failed probe is preserved
and is not counted as invalid-selector success.

Recovery revalidates the completed mode proofs, fixture metadata,
provider/comparator/source hashes, gate shapes and actual dispatch
counts without rerunning either successful GPU comparison. The corrected
negative probe derives its environment from the successful mode1
command, changes only the new selector to two and unsets shader stats
before execution to protect previous evidence. It explicitly rejects the
new selector in a fresh process. Six model/recovery tests pass, including
the real mode0/mode1 proof revalidation and qualified-environment
preservation. Aggregate finite bitwise comparisons are 601,620,480.
Frozen full 2K coverage and gated initial/fresh TTFT timing are now
running on GPU7; no model performance gain is established.

The completed frozen full 2K coverage passed HC94/dense171, but the initial
TTFT screen did not support the candidate:

| Validation | Median natural decode tok/s | Mean TTFT s | Matched TTFT ratio | Faster TTFT requests | Matched decode ratio |
|---|---:|---:|---:|---:|---:|
| Full | 77.512775 | 3.488798536 | 0.996987837 | 2/3 | 0.996540391 |
| Retained | 78.257377 | 3.484606698 | 1.000105745 | 1/3 | 0.983596360 |

Neither mode met the three-of-three TTFT gate; retained decode also failed
the 0.99 floor. All measured samples are retained. The selector stays
default-off, with no unchanged fresh 2K repetition or 8K/32K escalation.
The qualified original0955 provider is unchanged.

### Diagnostic FP16 four-term matrix feasibility

A separate default-off `ORT_WEBGPU_FP16_DOT4_MATRIX_PROBE` adds only a
canonical M16/K4/N32 diagnostic branch. It zero-pads four real products
into K16 native FP16-input/FP16-result subgroup matrices and compares
their outputs with WGSL `dot(vec4<f16>, vec4<f16>)` using the same decoded
weights. Its flag-zero production paths are unchanged. This uses the
existing experimental Dawn matrix feature, not the paused Dawn FP8 work.

Two actual-emitter/admission/bounds tests and the exact 152-object/SDK,
fresh original-object and original0955-relink build passed. GPU7 executed
the native matrix shader successfully: eight immutable seeded fixtures,
four rounds, 8,192 finite comparisons, **3,000 bitwise mismatches**, exit2.
Both signed-zero and nonzero differences occurred; for example, matrix
-0.936035 versus scalar -0.935547. This demonstrates native FP16 matrix
availability, not an exact dot substitution or a throughput improvement.
The strategy is held, with no model timing or provider promotion.

CPU inference against five printed nonzero results found no matching
serial accumulation among 24 orders and eight rounded/fused masks.
Four mixed-precision pairwise `(0,3)/(1,2)` patterns remain hypotheses,
not an established dot implementation. A new isolated diagnostic compares
all four against the original WGSL dot over the complete eight-fixture
suite, including signed zeros. It materializes FP16 products through
workgroup storage before testing rounded and fused pair sums. It is not
model eligible, does not relax bitwise gates, and makes no performance
claim.

The completed tree diagnostic rejects all four hypotheses: 8,192 finite
comparisons per pattern, with 1,548/752/1,872/1,128 bitwise mismatches.
Three CPU tests, the exact provenance build and a fresh explicit
invalid-selector rejection pass. The source/fixture/provider/guard-hashed
outcome is preserved; signed zeros remain included. No tree is selected,
and no model benchmark or production arithmetic change follows.

### HC projection physical lane8 (experimental, default off)

`ORT_WEBGPU_HC_PROJECTION_LANE8` changes only the physical projection-lane
mapping of the fused HC shader. Each 32-thread warp covers four columns
with eight K lanes each. All 256 physical normalization threads, eight
columns per workgroup, 32 logical K lanes per column, RMS trees, padded
post-values, original FP16 dots/updates, barriers and serial ordered folds
remain unchanged. Partials are written/read through the original logical
column/lane index rather than the permuted physical index.

This is distinct from rejected narrow-column HC variants, which changed
workgroup geometry or normalization work, and from the held standalone
dense lane8 experiment. It tests shared-activation multicast/locality
without asserting a bank-conflict or performance improvement. The
selector is process-static, rejects invalid values, preserves default
shader text/identity, and uses `HCPostNormProjectionLane8F16` when enabled.
Native four-output/replay and frozen model gates remain mandatory;
initial/fresh 2K and repeated 8K precede 32K. No gain is established.

Four tests pass: actual C++ renderer/default equality for all shared-scale
combinations, exhaustive physical/logical bijection and warp geometry,
unchanged surrounding Compute/fallback code, and generated native/model
proof wiring through 32K. The exact 152-object/SDK, fresh original-HC
object and original0955-relink build passes. Provider SHA-256:
`e9be40c3ba257d203103c3a6eb3c694290f9e91d25d6a7d6f8c9f3278ec92743`.

Separate process-zero/process-one native tests pass 607,143,936 finite
bitwise comparisons each, plus fresh explicit invalid-selector rejection:
1,214,287,872 comparisons across all four outputs, changing inputs/splits,
pending1/64, replay binding/LRU checks and original fallback fixtures.
Frozen full 2K HC94/dense171 coverage and original-provider full/retained
initial timing are running on GPU7. CPU builds are complete and do not
overlap timing. These correctness results are not throughput or
whole-model quality qualification.

Frozen full 2K HC94/dense171 coverage passes, but the initial original0955
timing screen rejects the physical HC lane8 mapping:

| Validation | Median decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 77.370123 | 3.480105 | 0.997468431 | 1/3 |
| Retained | 78.267212 | 3.483905 | 0.999441897 | 1/3 |

TTFT gates pass; decode consistency fails in both modes. No samples are
discarded and no unchanged repeat or 8K/32K escalation follows. The
selector stays default-off; original0955 is unchanged.

### CPU greedy eight-lane argmax (experimental, default off)

Earlier CPU profiles placed about three percent of self samples in greedy
selection; this is not a wall-time prediction. Fresh rebuilding of the
original GenAI search translation unit and relinking reproduces the
frozen legacy runtime byte-for-byte, as does the original parent relink.
The isolated experiment changes only CPU greedy float32 token selection:
`ORT_GENAI_CPU_GREEDY_ARGMAX_LANES8` uses eight independent strided maxima
and an index-aware final merge. Original strict-greater comparisons,
lower-index ties, signed zeros, sticky first-NaN behavior and ignored later
NaNs are preserved. Logits, model/provider, sampling, beam search and
public object layouts are unchanged.

The boolean selector reuses the existing throwing environment helper,
defaults off and is process-static. CPU oracle/input-preservation tests
and CPU-only timing at the actual 248,320 vocabulary precede frozen-model
timing. No CPU build overlaps GPU timing; no decode or TTFT gain is
established.

Three source/command tests pass, including original-runtime (not
candidate-flag-zero) controls and 2K/8K/32K wiring. The original search
rebuild/relink reproduces runtime
`0697af07edc986545f6a54c3649be02af116aba616dc71292cd46ae121a7faa9`.
The actual original and candidate C++ reducers select identical IDs over
127,233 cases and 21,358,629 input elements, with input preservation,
empty-input rejection, NaN/infinity/signed-zero and tie cases.
Five CPU-only brackets at vocabulary248320/200 calls show
3.408-3.446 times speedup; these are not model throughput measurements.

The first negative probe used one output token and was rejected by the
existing benchmark CLI before reaching token selection. It is preserved,
not counted as selector validation. A fresh attempt with the supported
two-output request reaches greedy selection and explicitly rejects
selector two. Frozen 256-output initial 2K full/retained timing is running
on GPU7 with original0955 provider, original-runtime controls and unchanged
95 pinned allocator workers. Initial/fresh 2K and repeated 8K gates still
precede 32K; no gain or promotion is established yet.

The initial original-runtime 2K screen passes both validation modes:

| Validation | Median natural decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 79.109435 | 3.482949 | 1.015938930 | 3/3 |
| Retained | 79.500802 | 3.483292 | 1.029892438 | 3/3 |

Full before/after medians are 77.725140/77.986396; retained controls are
75.505597/78.356473. The slow retained before-control inflates that
three-percent matched result; it is not an isolated causal three-percent
claim. Frozen outputs, original0955 provider, mapped runtime hashes,
unchanged 95 allocator worker pins, zero major faults and TTFT gates pass.
Fresh 2K confirmation is running, reusing the initial ending controls;
the windows are therefore dependent. No long-context qualification,
provider/runtime promotion, whole-model quality or 100-TPS claim follows
from this initial screen.

Fresh confirmation does not support promotion:

| Validation | Median natural decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 78.940876 | 3.497864 | 1.036844666 | 3/3 |
| Retained | 75.811198 | 3.489842 | 0.966713811 | 0/3 |

The slow full after-control (74.634531) inflates its matched gain;
retained before/after are 78.356473/78.341250. All samples remain included.
No slowdown cause is established. Frozen IDs/fault/worker and TTFT gates
pass, but retained decode fails. The helper stays default-off, with no
unchanged retry or 8K/32K escalation.

A distinct combination now tests the unchanged exact eight-lane helper
with the separately all-context-qualified local8 placement: process and
all workers use CPUs48-55 in **both** controls and candidate. Controls
still load the original legacy runtime, not the candidate with flag zero.
This is not an assertion that CPU migration caused the rejected screen,
and not an unchanged repetition of its default-worker configuration.
The wrapper verifies the immutable all-context locality proof and exact
worker masks, retains natural 256-output/request/TTFT gates, and requires
initial/fresh 2K before repeated 8K and 32K. No gain is established.

Three generated-source, matched-command/all-context and exact worker-mask
tests pass. The initial matched-local8 screen passes:

| Validation | Median natural decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 79.545623 | 3.488727 | 1.015749041 | 3/3 |
| Retained | 79.871652 | 3.487272 | 1.012884075 | 3/3 |

Before/after controls are 78.386779/78.212394 full and
78.822887/78.985950 retained. These controls are relatively stable, unlike
the preceding rejected default-worker screen. Original-runtime controls,
original0955 provider, exact local8 process/worker masks, frozen 256-output
sequences, source/runtime hashes, zero major faults and TTFT gates pass.
Fresh 2K confirmation is running on GPU7 and reuses the initial ending
controls; it is not an independent bracket. No 8K/32K qualification,
promotion or 100-TPS claim follows from this initial screen.

Fresh matched-local8 2K confirmation passes:

| Validation | Median natural decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 79.385789 | 3.485952 | 1.013647245 | 3/3 |
| Retained | 80.125513 | 3.476601 | 1.015410361 | 3/3 |

Full controls are 78.212394/78.333604; retained controls are
78.985950/79.006387. All 12 initial/fresh matched-request comparisons
are positive; shared ending controls remain disclosed. Frozen outputs,
exact placement masks, source/runtime/provider hashes, fault and TTFT
gates pass. Initial 8K qualification is now running on GPU7. Repeated
8K support is still mandatory before 32K; no all-context promotion or
100-TPS claim is established.

Initial matched-local8 8K qualification passes:

| Validation | Median natural decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 79.381836 | 14.178650 | 1.013050314 | 3/3 |
| Retained | 79.894744 | 14.194766 | 1.014384861 | 3/3 |

Full before/after medians are 78.538692/78.344343; retained controls are
78.863851/79.001864. Frozen 256-output sequences, source/runtime/provider
hashes, exact local8 masks, zero major faults and TTFT gates pass.
TTFT is essentially unchanged. Fresh 8K confirmation is running on GPU7,
reusing initial ending controls. Repeated 8K success remains mandatory
before 32K; no all-context promotion or 100-TPS claim follows.

Fresh matched-local8 8K confirmation passes:

| Validation | Median natural decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 79.572839 | 14.171234 | 1.017275083 | 3/3 |
| Retained | 79.787643 | 14.190260 | 1.012753282 | 3/3 |

Full controls are 78.344343/78.171462; retained controls are
79.001864/78.967954. All 24 initial/fresh 2K/8K matched-request comparisons
are positive, with shared controls disclosed. Frozen IDs, exact CPU/worker
masks, source/runtime/provider hashes, zero major faults and TTFT gates
pass. Initial 32K qualification is now running on GPU7. Fresh 32K
confirmation remains required for repeated all-context support; the
original stack stays unchanged and 100 TPS remains unmet.

An all-context aggregator is prepared to verify all six immutable
initial/fresh 2K/8K/32K screens, exact runtime/provider/source hashes,
request support, matched placement, TTFT gates and ordered full/retained
output shape. It explicitly reports insufficient support rather than
repairing failed screens. It reuses the existing strict target helper:
all six fresh candidate medians must exceed 100 tok/s, not merely their
average. Three tests pass for valid/rejected support, misreported
ratios/policy/output shapes and the exact 100-TPS boundary. It has not run
while initial and fresh 32K qualification remain incomplete.

Initial matched-local8 32K qualification passes:

| Validation | Median natural decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 70.107415 | 59.543156 | 1.013427816 | 3/3 |
| Retained | 70.493944 | 59.578765 | 1.011882990 | 3/3 |

Full before/after medians are 69.155810/69.123666; retained controls are
69.416067/69.872892. Frozen outputs, mapped runtime/provider hashes,
exact CPU/worker masks, zero major faults and TTFT gates pass. This is
a small natural-decode gain; TTFT remains essentially unchanged.
Fresh 32K confirmation is running on GPU7, reusing the initial ending
controls, followed by the all-context aggregator once complete. No
repeated all-context promotion or 100-TPS claim is established yet.

Fresh matched-local8 32K confirmation and the all-context aggregator pass:

| Validation | Median natural decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 70.009307 | 59.607459 | 1.011500761 | 3/3 |
| Retained | 70.614092 | 59.624631 | 1.013220497 | 3/3 |

Full before/after medians are 69.123666/69.209686; retained controls are
69.872892/69.466292. All 36 measured request comparisons across initial
and fresh 2K/8K/32K brackets are positive. The aggregate status is
`repeated_cpu_greedy_local8_supported_all_contexts`; frozen output
regressions, source/runtime/provider hashes, exact matched CPU and worker
masks, zero major faults and TTFT gates pass. Fresh brackets reuse initial
ending controls, so these measurements are not fully independent.

The candidate runtime SHA256 is
`ae960fdff1476630f91f798abb5ccb57ad9bbb4f40feee2c5a15328062b7fbc9`.
Controls retain the original `0697af07` runtime and both sides use the
unchanged `0955e1d6` WebGPU provider with CPUs 48-55. The optimization
remains default-off. This establishes a small CPU greedy-selection
improvement with matched local8 placement through 32K, not whole-model
quality qualification or headline eligibility. All six fresh medians
remain below 100 tok/s, and TTFT remains essentially unchanged, including
the still-poor approximately 59.6-second 32K result.

### Paired FP16 projection streams: default-off exactness experiment

`ORT_WEBGPU_HC_PROJECTION_PACKED16` pairs original logical K lanes `l`
and `l + 16` into independent components of `vec2<f16>` accumulators.
This is a portable half2-style packing experiment, not a new summation
tree or a repeat of narrow-column/lane8 permutations. Projection uses
128 active threads for the same eight columns; all 256 threads still
perform the original normalization and participate in every barrier.
The 41 dispatch groups, padded shared post-values, original 256 RMS
partials and tree, four ordered dot-pair updates per block, ten ordered
tile updates, and final serial fold of all 32 original logical partials
are preserved. Threads in the inactive projection half do not write
partials or return before the workgroup barrier.

Three tests pass, including actual C++ shader rendering in all four
shared/shared-scale combinations, exact default-shader equality,
unchanged normalization/output suffixes and barriers, a bijection over
all 256 logical projection streams, padded-column coverage and generated
native/model/screen wiring. The isolated build passes the recovered
152-object/SDK provenance gate, exact original translation-unit rebuild
and byte-identical original `0955e1d6` provider relink. The original
operator sources and completed evidence builders remain unchanged.
Native separate-process 0/1 and fresh invalid-selector checks are next;
no model timing, exactness qualification or speedup is claimed yet.

Paired projection native qualification now passes in separate selector
0/1 processes: 607,143,936 finite bitwise comparisons per mode,
1,214,287,872 total, covering all four outputs, changing inputs and CPU
split sizes, eligible and fallback fixtures, pending windows 1/64 and
replay eviction. Fresh selector 2 is explicitly rejected. The provider
SHA256 is
`69e6ac44d7dcd585f2557dc596d664b302fb7e9c45fec63808f69bf3eb17e389`.
This is arithmetic/replay qualification only. Frozen full-2K coverage
and initial full/retained original-provider before/candidate/after
performance brackets are next on GPU7, with unchanged worker placement
and original runtime on both sides. No speedup or long-context support
is established yet.

Initial paired projection full-2K coverage passes with 94 HC and 171
dense replay instances, preserved frozen output sequences and all
runtime/provider/source/fault gates. Initial performance is rejected:

| Validation | Median natural decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 74.715944 | 3.485990 | 0.972898027 | 0/3 |
| Retained | 76.799841 | 3.495041 | 0.978588294 | 0/3 |

Both modes are slower than matched original-provider controls; TTFT is
essentially unchanged (matched ratios 1.000391253/1.003079955).
The candidate stays default-off/held, without an unchanged repeat or
8K/32K escalation. Exact arithmetic does not establish useful packing
performance; no slowdown mechanism is established by this screen.

### Dense paired columns: activation-sharing FP16 experiment

`ORT_WEBGPU_DENSE_PACKED_COLUMNS` pairs output columns `c` and `c + 2`
in independent `vec2<f16>` accumulators, reusing the two activation
vectors across both columns. Unlike the held HC K-lane pairing, all
64 dense workgroup threads participate in projection. The original four
columns/group, 32 logical K lanes/column, bitcast nibble decoding,
FP16 weight scaling, four ordered dot-pair updates per block, shared
FP16 partial update after every 1024-element K tile, and final serial
32-partial fold are retained. N=1 and K-tail reads are individually
guarded; unsupported shapes, FP32 accumulation and all original fallback
paths remain untouched. The private program has a distinct identity and
variant-keyed replay. This is a new activation-sharing layout, not an
unchanged retry of lane8 permutations or HC packing.

The first isolated build passed original provenance rebuilding but
failed compilation on four now-unused template parameters under
`-Werror`. That failed build and log are preserved. Build-v2 adds an
explicit canonical-parameter guard rather than suppressing warnings or
changing arithmetic. The recovered 152-object/SDK checks, byte-identical
original translation-unit rebuild and original `0955e1d6` provider
relink pass; the candidate and FP32-aware comparator build successfully.
Renderer/mapping/reversible-source/generated-runner tests pass.
Native changing-input/replay and FP32-fallback checks are next, before
any model profiling or timing. Model controls explicitly retain the
original provider and runtime; no performance or quality claim follows.

Four tests now pass, including the explicit canonical-parameter rejection
and all 2K/8K/32K full/retained command shapes. Dense paired-column native
qualification passes 116,582,784 finite bitwise comparisons across
27 fixtures, nine single-token shapes, 128 changing rounds, pending
windows 1/64, on/off replay variants, multi-token fallbacks and explicit
invalid-selector rejection. Separate FP32-accumulation fallback
qualification passes another 116,582,784 comparisons with no candidate
replay path. Frozen full-2K active-program profiling is next; only if
its expected 159 paired dense programs and original output regressions
pass will initial full/retained original-provider brackets run.

Dense paired-column full-2K active-path profiling passes the frozen output
regressions and confirms exactly 159 `MatMulNBitsPackedColumnsF16`
programs, zero original block32/tile4 programs, 94 original fused HC
programs and 12 cached attention programs. Instrumented kernel durations
are not standard request timings.

Initial original-provider full/retained 2K performance is mixed and held:

| Validation | Median natural decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 77.853433 | 3.485623 | 1.014677970 | 3/3 |
| Retained | 78.152178 | 3.488189 | 0.997240523 | 1/3 |

Frozen sequences, original-runtime controls, provider/source hashes,
worker placement, fault and TTFT gates pass. The retained screen does
not support a gain, so the candidate stays default-off/held, with no
unchanged repeat or long-context escalation. The positive full result
alone is not sufficient evidence for promotion or a headline speedup.

### Single FP32-result native matrix dot4 diagnostic

`ORT_WEBGPU_FP16_DOT4_MATRIX_F32_PROBE` tests one native subgroup-matrix
accumulation of four FP16 products, padded to K=16, with an FP32 result
and conversion to FP16 only at the final write. Unlike the rejected
independent FP32-product materialization followed by a WGSL FP32 dot,
all four products are accumulated inside one native matrix operation.
Unlike the earlier F16-result matrix probe, both capability admission
and the generated result/storage types require F32. Activations,
FP16-scaled INT4 coefficients, matrix load layout, scalar reference and
eight immutable seeded fixtures are otherwise unchanged.

Three tests pass, including actual C++ shader rendering for both M8/M16
configurations, exact result-only shader changes, typed capability
selection, diagnostic-only shape admission, unchanged default/fallback
paths and complete signed-zero/nonzero mismatch classification. The
isolated build passes the recovered 152-object/SDK checks, exact original
translation-unit rebuild and byte-identical `0955e1d6` provider relink.
GPU7 arithmetic diagnostics and fresh invalid-selector rejection are
next. Coverage is 2,048 unique fixture elements repeated four times
(8,192 comparisons), not whole-model quality or performance evidence.
No model is eligible for this diagnostic, and no throughput claim is made.

The single F32-result matrix diagnostic executes natively on GPU7 but
fails exact arithmetic: 8,192 comparisons, 3,000 mismatches, partitioned
into 100 signed-zero mismatches and 2,900 other numerical mismatches.
Fresh selector 2 is explicitly rejected. Merely retaining an FP32
matrix result does not repair the dot4 substitution; a signed-zero-only
special case would not repair its nonzero errors. The diagnostic is
held/default-off, without model timing or an unchanged retry.

### Expanded paired dot4 rounding characterization

The original four mixed-rounding tests covered only the `(0,3)/(1,2)`
pairing and selected rounding orientations. A distinct expanded
diagnostic now covers all three pairings, four pair forms (rounded
products/addition, full FP32 product sum converted to FP16, and both
FMA orientations), and both expression-only and explicitly workgroup-
materialized FP16 pairs: 96 patterns. Products are materialized in
FP16 before testing rounded-product/FMA hypotheses. The prior four
expressions are tracked as subset indices 36, 39, 40 and 43, not
presented as newly untested arithmetic.

Three renderer/mapping/fixture/comparator tests pass. Every pattern has
its own 16-column output block with no overlap with the final scalar
dot4 reference. Eight diagnostic fixtures expand to N1552, preserving
parent activation bytes and first-16 INT4 weights/scales exactly. The
isolated build passes recovered 152-object/SDK checks, exact original
translation-unit rebuilding and byte-identical `0955e1d6` provider
relink. GPU7 testing will report all 96 counts (8,192 comparisons each,
2,048 unique fixture elements repeated four times) and fresh invalid-
selector rejection. The supplied model remains unchanged and ineligible
for this diagnostic; no exact pattern or speedup is claimed yet.

Expanded characterization finds two exact variants, indices 47 and 95:
both use `fma(x.w, w.w, rounded(x.x * w.x))` plus
`fma(x.z, w.z, rounded(x.y * w.y))`, with FP16 products and pair results.
The expression and workgroup-materialized pair variants each match all
8,192 finite comparisons. The other 94 variants fail. Prior subset
counts reproduce exactly: 36/39/40/43 have
1,548/752/1,872/1,128 mismatches, respectively. Fresh selector 2 is
explicitly rejected. This identifies a supported diagnostic rounding
direction, not universal dot4 semantics or whole-model qualification.
It differs from the held native cross-FMA prototype's `(0,2)/(1,3)`
pairing and the held reversed-product prototype.

### Rounding-aware native pair-matrix diagnostic

A new default-off diagnostic uses the supported `(0,3)/(1,2)` FMA
direction, materializing products 0/1 in FP16 RHS storage before a
single F16-input/F32-result matrix operation. It reports raw matrix
pair sums, guarded pair sums, the original dot4 and a correction mask
in four separate 16-column blocks (M16/K4/N64). The guard recomputes
only an affected scalar FMA pair for zero/subnormal results, normal
FP16 rounding boundaries within one FP32 mantissa step, or near the
FP16 overflow range. This is a candidate precision safeguard, not a
claim that all hardware matrix accumulation matches scalar FMA.
The mask records native-vs-corrected pair coverage and correction
reasons, so agreement cannot conceal an all-scalar output path.

Three tests pass for actual M8/M16 shader rendering, correct product/
FMA direction and FP32 admission, unchanged default paths, output-plane
bijection, mask consistency and exact parent fixture bytes. The
recovered 152-object/SDK checks, exact original translation-unit
rebuild and byte-identical `0955e1d6` provider relink pass, and the
isolated comparator builds. GPU7 diagnostics and fresh invalid-selector
rejection are next. The supplied model remains unchanged/ineligible;
no exactness or throughput claim follows from the build.

The first native pair diagnostic is held: raw sums have 2,940/8,192
mismatches, guarded sums 2,840/8,192. Both pairs use native results in
6,672 comparisons; there are 2,992 corrected pair results (2,944
zero/subnormal reasons and 48 boundary reasons). Fresh selector 2
is explicitly rejected. Correct pairing plus these safeguards alone
does not establish compatibility. This version casts FP32 matrix
results to FP16 variables but does not explicitly store both final
pairs in FP16 before adding them; the next distinct diagnostic will
test that materialization boundary without changing the native MMA,
pairing, guard or input fixtures. No model timing follows from V1.

The distinct materialized-pair diagnostic also fails, with exactly the
same mismatch and path counts on byte-identical fixtures. Five renderer,
admission and inherited fixture tests and the exact original-object/SDK
build passed; fresh invalid selector rejection passed. Explicit FP16
pair storage is therefore not a demonstrated fix or enabler here.
This variant is held without model timing. A component diagnostic is
needed to distinguish native-pair disagreement from scalar-tree
disagreement within the same shader.

The component diagnostic completes 8,192 comparisons for each reported
variant. Native/scalar FP16 pair disagreements are 368 for pair 0 and
364 for pair 1. More importantly, the entirely scalar FP16-FMA paired
sum itself disagrees with this shader's `dot` in 2,840 cases; the explicit
FP32-FMA paired sum also has 2,840 disagreements. Thus the earlier
96-pattern shader's exact 47/95 results do not transfer to this shader.
Four renderer/admission/comparator/fixture tests, exact original-object
and SDK build, and fresh invalid-selector rejection pass. The evidence
does not identify a driver lowering mechanism or universal dot tree.
An all-scalar paired-tree failure rules out matrix guard widening as a
complete fix here. No model timing or performance claim is made.

### Ordered HC projection tile unrolling (experimental, default off)

`ORT_WEBGPU_HC_PROJECTION_TILE_UNROLL=1` emits ten explicitly ordered
projection tiles with constant K-block offsets. Unlike the held
component-unroll experiment, the original four-component loop stays
intact. Each tile still starts a fresh FP16 `sum`, performs the original
ordered `dot(x0, w0) + dot(x1, w1)` updates and updates FP16 `total`
before the next tile. All normalization code, padded shared addresses,
256 threads, 41 groups, barriers, tail handling and the final serial
32-partial fold remain unchanged. This tests loop/address generation,
not a new arithmetic decomposition or a proven throughput improvement.

The selector is process-static at program construction; zero emits
byte-identical original shader text and retains its program identity.
One uses `HCPostNormProjectionTileUnrollF16`; optional generation
statistics are off during natural timing. Three tests pass, including
actual C++ rendering across all shared/scale combinations, ordered tile
mapping, unchanged arithmetic/control suffix and proof-runner wiring.
The isolated build reproduces all 152 original objects, original SDK
hashes, the original HC translation unit and original0955 relink.
Native exactness/replay qualification on GPU7 is running; no model
timing, promotion, TTFT or throughput claim follows from readiness.

Native process modes 0/1 now pass 1,214,287,872 finite bitwise
comparisons in total, including all four outputs, changing inputs and
CPU split values, pending windows 1/64, replay/LRU eviction and fallback
fixtures. Fresh invalid selector 2 is explicitly rejected. Full 2K
model coverage passes frozen prefixes and zero-fault gates, with 94
HC replay instances (6,486 hits, 4,888 binding hits) and the original
171 dense instances (11,799 hits, 10,714 binding hits). Actual new
shader generation is verified independently of replay counters.
Original0955-provider full/retained natural 2K timing brackets are
running without generation counters or concurrent CPU builds. This
is readiness for performance qualification, not a measured gain.

The completed 2K timing brackets reject tile unrolling:

| Validation | Candidate median decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 74.364011 | 3.478631 | 0.957022150 | 0/3 |
| Retained | 73.987242 | 3.487032 | 0.941926480 | 0/3 |

Frozen outputs and zero-fault gates pass, but there is no supported
decode gain. Keep the selector off, with no unchanged repeat or
8K/32K escalation. A causal slowdown mechanism has not been established.

### HC packed-block lookahead (experimental, default off)

`ORT_WEBGPU_HC_PROJECTION_LOOKAHEAD=1` retains the original ten-tile
loop and original four-component loop. It loads tile 0 once, then
requests the next packed block and its FP16 scale before decoding and
computing the current tile. Current-block registers advance only after
the original ordered FP16 `total += sum`. A `tile < 9` guard prevents
out-of-range lookahead; padded columns still skip all projection loads.
The RMS tree, padded shared addresses, every dot and FP16 update,
geometry, barriers, final serial32 fold, output split and fallbacks
stay intact. This is a load-scheduling experiment, not an asynchronous
copy API or a guarantee of compiler scheduling or higher throughput.

The process-static selector preserves original shader text/identity
at zero; one uses `HCPostNormProjectionLookaheadF16`. Generation
statistics remain optional and disabled for natural timing. Three
tests pass for actual rendering in all shared/scale combinations,
unchanged normalization/fold/arithmetic statements, bounded block
mapping and generated native/model/all-context screen wiring. The
exact 152-object/SDK/original-translation-unit/0955-relink build
passes. GPU7 native arithmetic/replay qualification is running.

Native process-zero/process-one qualification now passes 1,214,287,872
finite bitwise comparisons, all four outputs, changing inputs and
splits, pending1/64, fallback fixtures and replay/LRU gates. Fresh
invalid selector rejection passes. Full 2K frozen model coverage
verifies new lookahead shaders and the expected 94 HC/171 dense
replay instances, with 6,486/11,799 replay hits and 4,888/10,714
binding hits respectively. Frozen prefixes and zero-fault gates pass.
Original-provider full/retained 2K natural timing brackets are running
with counters disabled and no concurrent CPU builds. No throughput
or TTFT improvement is established yet.

Completed lookahead timing is mixed and held:

| Validation | Candidate median decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 78.914989 | 3.483924 | 1.014892443 | 3/3 |
| Retained | 76.096207 | 3.484154 | 0.994506492 | 1/3 |

The retained gate fails; keep off without an unchanged repeat or
long-context escalation. The positive full result alone is not a
qualified improvement. An independent cache-capacity hypothesis is
not supported by this short coverage sample: all 94 HC binding
caches contain 18 entries of capacity 64, not a saturated cache.

The next arithmetic check compares the paired-matrix outputs against
captured `dot` results from the completed, unchanged enumeration
shader in a separate process. All previous paired-matrix diagnostics
compare against `dot` within their own changed shader. The component
results justify checking cross-shader transfer; they do not prove
that compiler lowering changed or that native matrix results match
the original production kernels. No model eligibility follows from
this additional diagnostic.

The serialized cross-shader diagnostic succeeds against the unchanged
enumeration shader's independently captured `dot` output:

| Candidate field | Bitwise mismatches / 8,192 |
|---|---:|
| Raw native matrix sum | 100 |
| Guarded native matrix sum | 0 |
| Scalar FP16 paired-FMA sum | 0 |
| Scalar FP32 paired-FMA sum | 0 |
| Modified shader's local `dot` | 2,840 |

Original enumeration patterns 47/95 and reference repeatability are
reverified during capture. Both native pairs remain uncorrected in
6,672 comparisons; 2,992 pair results are corrected. Qualified fixture
hashes and operand bytes match across the two processes; the 4,096-byte
reference capture is checked unchanged during comparison. Three
mapping/contract/provenance tests and SDK-verified comparator build
pass. Neither provider is edited for this check.

This reverses the inference that the paired native arithmetic cannot
match the prior separate shader: local `dot` was not a transferable
reference in these diagnostics. It does not establish why the local
result differs, a universal rounding rule, production-kernel agreement,
whole-model quality or a TPS gain. Expanded finite/boundary/actual-weight
qualification and original-provider full-kernel gates are still needed.
The next distinct probe seeds native FP32 accumulator matrices from
materialized FP16 products, then adds one remaining FP16 product per
pair. This tests a packing prerequisite without production eligibility.

The seeded-accumulator probe builds with exact 152-object/SDK/original
TU/0955 relink provenance and passes two actual-renderer/admission
tests. Native M16 accumulator-matrix loads execute on GPU7. Against
the sealed independent reference, guarded outputs, scalar FP16 trees
and scalar FP32 trees each have zero mismatches in 8,192 comparisons;
raw outputs have 100 and local `dot` has 2,840. Both native pairs are
retained in 6,672 comparisons, with 2,992 corrected pair results.
Fresh invalid selector 2 is explicitly rejected. This validates the
seeded-C primitive on the original small fixture suite only.

The next qualification sweeps all 63,488 finite FP16 bit patterns
through input component slots with bounded INT4 coefficients, plus
separate signed-zero, cancellation, subnormal and near-overflow cases.
Each changed operand set gets a fresh independent enumeration-shader
capture. The ±1-FP32-mantissa-step safeguard remains experimental;
expanded success, if obtained, still would not be full-kernel or
whole-model qualification.

The first expanded harness times out during independent-reference
capture at its inherited 180-second process budget. The read-only
GPU guard reports `run_timed_out`, no foreign process and completed
container cleanup. No expanded arithmetic verdict or completed
coverage is recorded. The dataset and failed attempt are preserved.
A distinct reusable-session harness will upload changing operands
instead of creating 1,000 separate sessions. Its small-suite reference
must first match the sealed original capture byte for byte, so this
efficiency change cannot silently change the arithmetic reference.

The reusable-session replacement passes three binary-layout, dynamic-graph
and calibration-gate tests, and its SDK-verified C++ build succeeds. Each
phase uses one session with changing CPU-uploaded activation, packed INT4
and scale tensors; CPU EP fallback is disabled. GPU memory is checked
before every phase (all eight GPUs initially have 143,156 MiB free), and
all four guarded phases complete on GPU7. The eight-fixture dynamic
reference is byte-identical to the sealed original 4,096-byte capture.

The full sweep completes 1,024,000 bitwise comparisons per output kind
across 1,000 operand frames and four repeats:

| Output against separately compiled original reference | Mismatches |
|---|---:|
| Raw seeded native matrix | 32,292 |
| Guarded seeded native matrix | 0 |
| Scalar FP16 paired tree | 0 |
| Scalar FP32 paired tree | 0 |
| Modified matrix shader's local `dot` | 305,236 |

Both pairs remain native in 651,204 comparisons; 655,516 pair results
take correction. Original enumeration patterns 47/95 and reference
repeatability pass throughout. The aggregate reference is 512,000
bytes. Source, fixture, provider, SDK build, capture and guard hashes
are recorded in `fp16-dot4-dynamic-sweep-v1-outcome.json`.

This covers every finite FP16 input bit pattern with bounded INT4
coefficients and the separate edge cases, not all FMA operand
combinations or actual model weights. It supports the guarded primitive
on this dataset only: no default enablement, full-kernel eligibility,
whole-model quality or TPS improvement is claimed. Denser packing must
still pass comparisons against the actual original production kernel.

### Packed seeded-matrix dense production-kernel experiment

`ORT_WEBGPU_DENSE_PAIRED_MATRIX` is a separate default-off single-TU
candidate. Native M16/K16/N16 FP16-input/FP32-result matrices pack eight
original logical K streams and sixteen output columns per subgroup;
four 32-lane subgroups retain all 32 streams. FP16 first products are
materialized before FP32 accumulator loads. The original four ordered
word updates, paired-dot addition, block sum, tile update and final
serial 32-stream fold are retained. Shared staging occupies 11,264
bytes. Admission requires the existing NVIDIA single-row dense
bitcast/block32 route, exact tensor storage shapes, native matrix
configuration, adequate shared memory and N divisible by 16. Other
cases fall back; native replay gets a distinct variant key.

Three actual-renderer, mapping and admission tests pass, and the exact
152-object/SDK/original-TU/0955-relink build succeeds. Separate original
provider captures and candidate profiling prove 64 real native
dispatches per synthetic shape, not comparison with a modified
shader's local dot. K640/N2560 passes 163,840 finite bitwise comparisons.
K2560/N2560 fails all 163,840 comparisons. The guard's `failed` status
here means a completed numerical-mismatch child exit2, not a timeout.
The runner initially raises on that guard status; a separate sealing
script preserves the completed-child numerical verdict and hashes.
The original experiment remains held, with no model timing.

A distinct `ORT_WEBGPU_DENSE_PAIRED_MATRIX_RESET` candidate explicitly
clears private totals at invocation start and private block-sum arrays
inside every K-tile iteration. A CPU scalar-tree diagnostic reproduces
the original first eight outputs bitwise; carrying previous sums gives
outputs close to the failed candidate. Explicit reset already passes
the separately compiled production K640/N2560 and K2560/N2560 gates.
This does not establish a compiler/driver cause or universal WGSL
initialization rule. Its remaining shape checks are still in progress.
Three reset-renderer, admission and guard-status tests and the exact
provenance build pass. The new harness accepts only proven finished
exit0 or numerical exit2 children and explicitly rejects timeouts.

The explicit-reset candidate finishes all nine synthetic dense shapes:
3,067,968 candidate bitwise comparisons, zero mismatches, 512 profiled
native dispatches across the eight eligible shapes, and exact N1
fallback. This establishes production-kernel agreement on these
synthetic operands. Actual frozen-model weights and compiled-default
processes are being checked separately; they are not inferred from
the synthetic result.

The isolated, instrumented kernel profiles reject this matrix packing
as a performance candidate:

| K / N | Original median GPU us | Seeded matrix median GPU us |
|---|---:|---:|
| 640 / 2560 | 11 | 24 |
| 2560 / 2560 | 13 | 58 |
| 2560 / 7168 | 18 | 77 |
| 6144 / 2560 | 16 | 116 |
| 10240 / 320 | 17 | 154 |
| 2560 / 16384 | 26 | 141 |
| 2560 / 6144 | 16 | 67 |
| 2560 / 10240 | 20 | 113 |

These are medians of 64 instrumented dispatches per phase, not natural
model TPS or bracketed steady-state evidence. N1 fallback is 12 us in
both phases. The candidate remains held/default-off; no unchanged
natural-model timing or long-context escalation is planned.

The next distinct `ORT_WEBGPU_DENSE_COLUMNS16` candidate keeps ordinary
FP16 dot arithmetic and expands the output tile from four to sixteen.
Each of 128 threads computes four columns for one of the original 32
logical K streams. Packed weights/scales are loaded once per block;
the same two activation vectors serve four columns per word. A shared
activation tile is reused fourfold without native matrix staging.
The original bitcast decoding, FP16 scaling, ordered block updates,
shared FP16 tile updates and serial 32-stream fold remain. Shared
storage is 3,072 bytes; unsupported and N1 cases fall back, with a
distinct selected replay key. Three renderer/mapping/admission tests
and the exact provenance build pass. Synthetic and actual-weight
original-provider GPU gates are prepared, not yet run.

The native explicit-reset candidate subsequently passes eight actual
frozen-model weight shapes: 3,067,904 finite bitwise comparisons in each
original-capture, compiled-default and selected-native phase, all zero
mismatches. Each eligible shape executes 64 native dispatches. Actual
weight payloads, their external-file offsets, frozen graph/config and
independent captures are sealed. This extends arithmetic support but
does not overturn the isolated performance rejection.

Columns16 finishes all nine synthetic shapes with zero mismatches in
3,067,968 candidate comparisons. Its isolated timings are mixed:
K2560/N16384 improves 26→23 us and K2560/N10240 improves 20→18 us, while
K640/N2560 is 11→12 us, K2560/N2560 is 13→15 us, K6144/N2560 is 16→20 us
and K10240/N320 is 17→24 us. No unchanged model escalation is performed.

A distinct `ORT_WEBGPU_DENSE_COLUMNS8` candidate reduces private
packed-weight/register state to two columns per thread and eight
outputs per 128-thread group, retaining all 32 K streams. Shared
storage is 2,560 bytes. Admission requires N≥2560 and N divisible by
8; N1 and the final K10240/N320 projection remain original. Replay
uses a distinct key6. Two renderer/mapping/batch tests, three
actual-weight/model-gate tests and the exact provenance build pass.

The new runner batches nine separate child processes into one guarded
container per original/default/candidate phase, instead of launching
one container per shape. Process-static selectors still get separate
processes, and every batch checks all GPUs' memory and uses monitored,
serialized GPU7 work. Synthetic comparisons total 3,067,968 per phase.
Actual-weight comparisons total 3,067,904 per phase across eight
frozen-model tensors. All original, compiled-default and selected
comparisons are finite and bitwise equal; exact capture repeatability
and real selected/fallback dispatch counts are checked.

| Actual-weight shape | Original instrumented GPU median us | Columns8 median us |
|---|---:|---:|
| 2560 / 7168 | 18 | 15 |
| 2560 / 16384 | 26 | 22 |
| 2560 / 6144 | 16 | 15 |
| 2560 / 10240 | 20 | 18 |

These kernel medians are not natural TPS measurements; other shapes
are unchanged or have small profiling differences. A fresh invalid
selector2 is explicitly rejected. Full frozen-model 2K profiling
passes original output prefixes, unchanged graph/config, original
runtime/worker policy and zero major faults. It proves 158
`DenseColumns8` dispatches, one original block32 final projection,
94 HC post-normalization projections and 12 query-cache attention
instances. The original-provider bracketed full/retained 2K natural
decode/TTFT screen is now running. Longer-context escalation requires
both modes and the repeated support gates; no >100-TPS, quality or
default-promotion claim is made.

The initial Columns8 natural 2K original-provider screen supports
both validation modes:

| Validation | Candidate median decode TPS | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 77.959823 | 3.485199 | 1.007602959 | 3/3 |
| Retained | 78.700107 | 3.487510 | 1.006762190 | 3/3 |

TTFT ratios are 0.996463/0.999291, respectively. Original runtime and
CPU/worker mask0–95 are matched in these brackets, so these are GPU-only
incremental measurements, not replacements for the qualified ~80-TPS
CPU-greedy/local8 combination. Frozen outputs, hashes and zero faults
pass. The 2K repeat is running, with initial ending controls reused
as its starting controls; the brackets are therefore partly dependent.

Two command/placement/metadata tests also pass for a subsequent
combination: both control and candidate use the already qualified
CPU-greedy-eight-lane runtime with flag1 and exact CPU/worker mask48–55,
while only the GPU provider changes from original0955 to Columns8.
This directly targets the best existing configuration without mixing
CPU runtimes or affinity policies in one comparison. It requires the
successful GPU-only 2K repeat and the existing repeated all-context
CPU qualification before running. The combined 2K→8K→32K initial/repeat
gates will assess that configuration directly rather than duplicate
the entire long-context GPU-only and combined qualification ladders.
No combined result or all-context support is yet claimed.

The GPU-only 2K repeat has completed with both validation modes supported:

| Validation | Fresh median natural decode tok/s | TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 78.030234 | 3.498456 | 1.009979324 | 3/3 |
| Retained | 78.885860 | 3.479305 | 1.026873414 | 3/3 |

The corresponding TTFT ratios are 0.999608 and 0.997218. All twelve
initial/repeat request comparisons are positive, but the repeat shares
the initial ending controls. The retained matched gain partly reflects
control drift: candidate medians change only from 78.700107 to 78.885860.
These original-runtime numbers do not replace the previously qualified
CPU-local8 results. Columns8 remains default-off; 100 tok/s, long-context
support for Columns8 and whole-model quality are not established.

The combined CPU-local8/Columns8 initial 2K screen is now running, with
the same qualified CPU runtime, greedy flag and exact 48-55 worker mask
in both control and candidate. Its outcome is still pending.

The combined initial 2K screen passes both validation modes:

| Validation | Median natural decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 79.667819 | 3.480793 | 1.031978576 | 3/3 |
| Retained | 80.403407 | 3.487195 | 1.008240665 | 3/3 |

Matched TTFT ratios are 0.997022880 and 0.998855805. Frozen prefixes,
provider/runtime hashes, identical CPU/worker placement and zero faults
pass. The full-mode controls are lower than earlier CPU-local8 controls;
the 3.20% matched result must not be presented as a 3.20% increase over
the earlier qualified absolute throughput. A fresh 2K repeat is required
before any 8K/32K escalation. No all-context or 100 tok/s claim follows.

A distinct Columns8 double-buffer experiment is prepared but not built
or GPU-tested. It combines the new eight-column/two-column-per-thread
mapping with the earlier alternating-activation-buffer design, removing
the trailing barrier from each K tile and adding one final barrier before
the serial partial fold. Storage increases from 2,560 to 4,608 bytes.
Two tests pass: rendered shader normalization verifies that arithmetic,
indices outside activation staging, and fallbacks remain unchanged;
randomized four-warp schedules check buffer reuse for up to 32 tiles.
The earlier four-column double-buffer candidate failed performance
support. Its result is not support for this new combination, which
requires independent original-provider arithmetic and performance gates.

The combined CPU-local8/Columns8 2K repeat supports both modes:

| Validation | Fresh median natural decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 80.122168 | 3.481829 | 1.022666999 | 3/3 |
| Retained | 80.392949 | 3.486272 | 1.008646072 | 3/3 |

TTFT ratios are 0.996777770 and 0.998475889. All twelve initial/repeat
comparisons are positive, with frozen prefixes, identical CPU/runtime/
worker policy, provider hashes and zero faults verified. The repeat
shares the initial ending controls; those comparisons are not independent.
Full controls recover toward the earlier CPU-local8 level, while candidate
throughput remains approximately 80 tok/s. The next gate is initial 8K,
not an all-context, 100 tok/s or whole-model quality claim.

The initial combined 8K screen is running on GPU7 after checking all
eight GPUs: each had 143,156 MiB free and no compute applications.
No other owned GPU work or CPU build overlaps these timing brackets.

The distinct Columns8 double-buffer provider now builds successfully
between timing runs. Provider
`7cf1bddfe5c0b7196c29d38dd14ad017bc02cca6a311a9edddf58a0b466940db`
passes the SDK checks, all 152 original source-object checks, original
translation-unit reconstruction and byte-identical `0955e1d6` relink.
Separate synthetic/actual batch drivers are generated; their additional
test verifies fresh original captures, compiled-default comparisons,
candidate comparisons, process isolation and separate evidence paths.
No double-buffer GPU arithmetic, model profile or throughput result
exists yet. The active combined 8K screen must finish before GPU testing
this candidate.

The combined initial 8K screen is mixed and held:

| Validation | Median natural decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 80.296255 | 14.169338 | 1.009729045 | 3/3 |
| Retained | 80.351317 | 14.178578 | 0.994015429 | 2/3 |

TTFT ratios are 0.999283550 and 1.001399474. Frozen prefixes, runtime/
provider provenance, worker placement and zero-fault checks pass, but
retained performance does not meet the support gate. Its candidate median
is higher than both control medians while its matched request aggregate
regresses: one request ratio is 0.969856576. The median must not conceal
that failure. Columns8/CPU-local8 remains default-off/held with no unchanged
8K repeat or 32K escalation. The earlier CPU-local8 baseline remains the
best repeatedly supported all-context configuration. Continue with the
substantively distinct double-buffer candidate, starting with independent
synthetic and actual-weight arithmetic rather than assuming its speedup.

The new double-buffer kernel passes nine synthetic shapes and eight
representative actual-weight shapes against fresh separate-process
original-provider captures. Each capture/default/candidate phase has
3,067,968 synthetic or 3,067,904 actual comparisons and zero mismatches.
Seven eligible shapes show 64 selected dispatches each; N320 and N1
remain on the original route.

| Actual shape | Original instrumented GPU median us | Double-buffer median us |
|---|---:|---:|
| K2560/N7168 | 18 | 16 |
| K2560/N16384 | 26 | 22 |
| K2560/N6144 | 16 | 15 |
| K2560/N10240 | 20 | 17 |

These isolated GPU timings are not natural decode TPS or matched
improvements over the held Columns8 candidate. Fresh selector-two
rejection passes. The full frozen-model 2K active-path profile verifies
158 `DenseColumns8DoubleBuffer` dispatches, one original final projection,
94 original HC postnorm projections and 12 cached-attention instances,
with original prefixes and zero faults.

The new combined double-buffer/CPU-local8 initial 2K natural screen is
running, comparing original0955 and new GPU providers with the same
qualified CPU runtime, greedy flag and 48-55 worker placement. Its
prerequisites require the new native/profile proofs and existing CPU
all-context qualification, not a repeated GPU-only timing ladder.
Two model/provenance/placement tests pass. No natural throughput gain is
yet established for double buffering.

The fresh decode-only profile totals 15,589 instrumented kernel us.
The largest totals are original HC postnorm projection (94 calls,
2,715 us), new dense GEMV (158, 2,572 us), HC up/premix (97, 1,153 us),
MoE FC1 (48, 1,144 us), MoE FC2 (48, 1,084 us) and cached attention
(12, 1,040 us). Profiling overhead prevents interpreting their sum as
natural token latency or a predicted causal speedup.

The distinct double-buffer/CPU-local8 initial 2K screen supports both modes:

| Validation | Median natural decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 80.412395 | 3.483676 | 1.011526299 | 3/3 |
| Retained | 80.576024 | 3.483744 | 1.008741943 | 3/3 |

TTFT ratios are 0.999315179 and 0.998925392. Frozen prefixes, identical
CPU/runtime/worker policy, provider hashes and zero faults pass.
This is modest initial support, not attainment of 100 tok/s, an isolated
improvement over Columns8, or all-context qualification. Repeat 2K is next.

A second distinct candidate targets original HC postnorm projection.
`ORT_WEBGPU_HC_NARROW_RMS_BROADCAST` combines 128 physical threads/four
projection columns with all-subgroup register RMS result broadcasting.
It retains all 256 original logical RMS partials and their descending
FP32 addition tree: shared widths 128/64/32, then subgroup widths
16/8/4/2/1. Each subgroup independently transports the same shared
32 values and broadcasts lane zero, avoiding a subgroup-zero/shared-memory
handoff barrier. This removes five runtime workgroup barriers, not five
floating additions. Admission requires NVIDIA, subgroup support and
fixed 32-lane subgroups; default and unsupported paths stay original.
Earlier narrow4 and subgroup-zero/shared-handoff candidates were held;
neither supplies performance support for this combination.

Two tests pass: generated shaders for all shared-input/scale combinations,
geometry/capability/default checks, and exact FP32 tree/result broadcast
comparison over 200 seeded four-stream inputs. GPU arithmetic is untested.
The first build exposes the reused legacy helper's five-argument replay
call, incompatible with the frozen replay header. The corrected candidate
retains the original call/API: selector and device features cannot switch
kernel variants within a process-static replay instance. A second build
compiles but metadata sealing rejects a duplicate provenance field.
Those attempts are preserved. Separate build-v3 successfully seals provider
`6813c8f107a080bed73a358089809f3729afbf45d701aff07cdc7be83ebd6648`
after the SDK, all 152 objects, original HC translation unit and exact
`0955e1d6` relink checks. No headers/layouts change and no GPU speedup or
whole-model-quality claim follows.

The double-buffer/CPU-local8 2K repeat supports both modes:

| Validation | Fresh median natural decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 80.041869 | 3.484843 | 1.021800281 | 3/3 |
| Retained | 80.303058 | 3.484441 | 1.004341438 | 3/3 |

TTFT ratios are 0.998811835 and 0.999888864. All twelve initial/repeat
comparisons are positive, but ending controls are shared. The full ending
control slows to median 78.420552 while candidate throughput is below
its initial 80.412395; the larger matched ratio is not an absolute TPS
increase. The retained gain is only 0.43%. Frozen outputs, provenance,
CPU/worker placement and zero faults pass. Initial combined 8K is running
on GPU7; no repeat-8K, 32K, 100 tok/s or quality qualification exists yet.

An independent HC-reference comparator is staged for the next timing gap.
It retains the inherited 36-fixture/128-round changing-input, split-value
and pending1/64 schedule, and adds separate original-provider binary
captures with explicit output type/shape headers and exact payload checks
for all four outputs. Its source-wiring test passes. C++ compilation and
the float16/float32, signed-zero, metadata and truncation I/O tests remain
pending; no reference outputs have been captured yet. The subsequent
original/default/candidate native phases will require real replay,
binding hits and LRU activity, and still need active-model kernel coverage
before any throughput claim. No CPU build overlaps active 8K timing.

The double-buffer/CPU-local8 initial 8K screen supports both modes:

| Validation | Median natural decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 80.056875 | 14.169231 | 1.007891280 | 3/3 |
| Retained | 80.385577 | 14.178749 | 1.004224011 | 3/3 |

TTFT ratios are 1.000068200 and 1.000258289. All six matched requests,
frozen prefixes, provider/runtime provenance, worker placement and
zero-fault checks pass. Gains remain modest; repeat 8K precedes 32K.

The independent HC-reference comparator now compiles successfully in
the timing gap. Its C++ tests pass float16/float32 roundtrips, signed-zero
payload mismatch rejection, output-shape mismatch rejection and truncated
metadata rejection. Two source/phase/replay-gate tests also pass. No HC
GPU reference capture has run yet; active kernel coverage and natural
throughput remain unqualified.

The double-buffer/CPU-local8 8K repeat supports both modes:

| Validation | Fresh median natural decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 80.456694 | 14.145512 | 1.012027004 | 3/3 |
| Retained | 80.847676 | 14.161045 | 1.008716156 | 3/3 |

TTFT ratios are 0.998312416 and 0.999526465. All 24 initial/repeat
2K/8K comparisons are positive, with shared ending controls disclosed.
Frozen prefixes, provider/runtime hashes, CPU/worker placement and
zero faults pass. Initial full/retained 32K testing is now running on
GPU7 after checking all eight GPUs free at 143,156 MiB each.

The queued HC candidate's model and CPU-local8 timing controllers are
generated and two tests pass. They require the independent native
reference/replay proof and fresh selector rejection, then real full-model
coverage of 94 new HC programs, zero original HC programs, 159 original
dense programs and 12 cached-attention programs. Both natural-timing
modes use the same qualified CPU runtime/greedy flag and exact 48-55
worker placement. No HC GPU work overlaps 32K timing, and no HC
throughput or quality qualification is claimed. A later six-test
invocation included short C++ shader-renderer builds while the initial
32K driver was active (the whole invocation took 2.080 s). Natural
request overlap is unresolved; initial 32K timing carries this caveat.
The 32K repeat is kept free of owned CPU builds.

Initial double-buffer/CPU-local8 full-validation 32K has completed:
70.797522 median natural decode tok/s, 59.575552 s mean TTFT,
matched decode ratio 1.011363263 and TTFT ratio 0.999450897.
All three request comparisons are positive and frozen-prefix/fault
gates pass. Retained validation is still running; this one completed
mode is not initial-both-mode or repeated-all-context qualification.
The all-context aggregate is prepared, with four passing tests
covering actual completed 2K/8K screens, invalid/mixed provenance,
missing modes, nonfinite metrics and the strict requirement that
all six fresh context/mode medians exceed 100 tok/s.

Initial retained 32K also passes: 70.944846 median natural decode
tok/s, 59.613986 s mean TTFT, matched decode ratio 1.007508343
and TTFT ratio 1.000172339, with all three requests positive.
Both initial modes preserve frozen outputs and zero faults.
The required 32K repeat has started on GPU7, after all eight GPUs
were checked free at 143,156 MiB with no compute applications.
It reuses initial ending controls, as did the short-context repeats.
No all-context promotion, whole-model quality or 100 tok/s claim
is made before its completed outcome and aggregate verification.

The queued `ORT_WEBGPU_QMOE_FC1_DOUBLE_BUFFER` candidate applies
two-buffer activation staging to the original fused FC1/SwiGLU
program. It retains 128 threads, eight columns, 32 K streams,
ordered FP16 dots/block/tile updates, optional bias, serial folds
and SwiGLU materialization. Alternating two 1024-element activation
tiles removes the per-tile trailing barrier and inserts one barrier
before the final fold. Shared storage is 4,608 bytes. The existing
FC1 admission remains unchanged; the new path additionally requires
bitcast unpacking and sufficient shared memory. Its process-static
selector is checked in both compute surfaces and included in command
replay and program keys.

Two pure-Python selector/geometry/protocol tests pass, including the
randomized four-warp two-buffer schedule. Actual C++ rendering,
152-object/SDK/original-TU relinking, independent original-provider
arithmetic, dispatch profiling and natural timing remain pending.
These are deferred until the current 32K repeat completes; no FC1
speedup is claimed. The older compact-20 FC2 candidate was already
rejected and is not retried unchanged.

Its independent QMoE comparator is now prepared: the inherited
23-fixture, 32-round input/route schedule and pending-1/64 parity
are preserved. Local comparisons toggle the original FC1 fusion,
while the new experiment selector stays process-static. Each
actual output is additionally captured/compared against a separate
0955 provider process using the existing type/shape-aware raw
FP16/FP32 reference codec. Fresh captures refuse existing files;
numeric differences return 2 and operational failures return 1.
The three-phase runner requires the original eleven command-replay
instances, 64 requests each and capacity eight, and explicitly does
not claim active candidate dispatch coverage. These replay assumptions
remain GPU-unvalidated. Five pure-Python checks and script compilation
pass; comparator/provider compilation and all GPU phases are pending.

The build-free double-buffer/CPU-local8 32K repeat passes both modes:

| Validation | Fresh median decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 70.365059 | 59.573479 | 1.006988621 | 3/3 |
| Retained | 70.996028 | 59.568784 | 1.007023627 | 3/3 |

TTFT ratios are 0.999111638 and 1.000347644. The immutable
all-context aggregate passes all 36 initial/repeat matched requests,
frozen outputs and zero faults. Fresh 2K/8K medians are
80.041869/80.303058 and 80.456694/80.847676 (full/retained).
Ending controls are shared; the initial 32K renderer-build timing
caveat remains explicitly recorded. No defaults are promoted.
All six fresh medians remain below 100 tok/s, and TTFT is essentially
unchanged. Whole-model quality is still unqualified.

The HC independent original capture phase has completed all
607,143,936 comparisons with zero mismatches and the required
replay activity; compiled-default and candidate phases are running
serially on GPU7. This is not yet candidate arithmetic qualification.

During this non-natural-timing gap, the FC1 candidate passed all six
renderer/protocol/native-source tests. Default shaders render exactly
as the originals; candidate normalization of staging/barriers renders
exactly as the originals for both bias and unpacking combinations.
Its build passes the exact 152-object/SDK/original-TU checks and
byte-identical 0955 relink. Provider SHA-256:
`3557fdb0963427b8cb1cc580d9fb16dbb78be2ea829efdbba23e26fc048a880a`.
The independent comparator and typed reference-I/O self-tests compile
and pass. Three model-driver tests also pass; generated drivers require
23 independent captures, all three native phases, real replay activity,
fresh selector rejection and an actual profile containing 48 new FC1,
48 original FC2, 94 original HC, 159 original dense and 12 cached
attention programs. No FC1 GPU work overlaps HC arithmetic, and no
FC1 performance or model-quality qualification is claimed.

HC independent arithmetic now passes all three original/default/
candidate phases: 607,143,936 comparisons per phase, zero mismatches,
all four outputs, 36 fixtures and 128 changing-input/split rounds.
Every phase has 12 replay instances, 1,152 requests, 372 hits and
372 binding hits, with the original binding/LRU invariants verified.
Fresh selector-2 rejection also passes. The full frozen 2K model
profile proves 94 new HC and zero original HC programs, 159 original
dense programs and 12 cached-attention programs, with matching
prefixes and zero faults. This is arithmetic/route admission, not
model-quality or throughput qualification.

Initial full/retained HC/CPU-local8 natural 2K timing is now running
on GPU7. Both providers use the same qualified CPU runtime, greedy
flag and exact 48-55 CPU/worker mask; the GPU control is original
0955, not a silently combined dense candidate. All GPUs were checked
free at 143,156 MiB before launch. No owned CPU builds or FC1 GPU
batch overlaps this natural timing. Long-context escalation requires
both modes and a fresh 2K repeat, then initial/repeat 8K before 32K.

The narrow-RMS HC candidate fails natural 2K timing decisively:
full/retained median decode is 75.147705/75.533137 tok/s,
with matched ratios 0.944711110/0.932919149 and zero of six
positive requests. TTFT is 3.482655/3.488272 s; frozen outputs
and faults still pass. Keep it off, with no unchanged repeat or
long-context escalation.

The fused FC1 double-buffer candidate passes all three independent
original/default/candidate phases: 9,748,480 comparisons each,
23 fixtures and 32 changing-input/route rounds, zero mismatches.
Every phase has eleven original command-replay instances, 704
requests and 528 hits. Fresh selector rejection passes. Full frozen
2K profiling proves 48 new FC1 and zero original FC1 programs,
48 original FC2, 94 original HC, 159 original dense and twelve
cached-attention programs, with matching prefixes and zero faults.
Its initial full/retained CPU-local8 natural 2K timing is running
alone on GPU7, after checking every GPU free at 143,156 MiB.

A distinct `ORT_WEBGPU_HC_WIDE_RMS_BROADCAST` prototype retains
the original 256 threads, eight columns and 41 workgroups instead
of the rejected narrow candidate's 128/four/81 geometry. It changes
only the final five RMS reduction levels to the exact ordered
all-subgroup register broadcast; post-mix/normalization staging,
projection arithmetic and output code retain original geometry.
Shared storage remains 27,648 bytes. This isolates the RMS change
without doubling normalization workgroups and differs from the old
subgroup-only experiment's shared-memory final handoff. Performance
is a hypothesis, not a claim.

Four pure-Python geometry/projection/proof-controller checks pass.
The actual C++ renderer and build are deferred until FC1 natural
timing ends. Native drivers will reuse the already-built immutable
HC reference comparator but create distinct fresh original captures.
No wide-HC GPU arithmetic, profile or throughput result exists yet.

The FC1 double-buffer natural 2K screen is complete and fails both
performance gates, despite bitwise/profile qualification:

| Validation | Median decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 78.719985 | 3.482198 | 0.990300681 | 0/3 |
| Retained | 79.270819 | 3.489738 | 0.986864512 | 0/3 |

Frozen prefixes and zero faults pass. Keep this candidate off; no
unchanged repeat or 8K/32K escalation. There were no owned CPU builds
or other owned GPU jobs during this natural timing.

After that screen ended, all five wide-HC tests passed, including the
actual C++ shader renderer. The scoped provider build also passes the
152-object/SDK, byte-identical original-TU and exact 0955 relink gates.
The independent original/default/candidate native qualification is now
running alone on GPU7 with distinct fresh original output captures.
All eight GPUs were checked at 143,156 MiB free, zero used and no
compute applications immediately before launch. No throughput gain
is claimed before numerical, active-model and natural-timing gates.

Wide-HC original/default/candidate phases each pass 607,143,936
bitwise comparisons over 36 fixtures and 128 changing rounds, all
four outputs and zero mismatches. Each phase exercises twelve replay
instances and 1,152 requests, with binding/LRU invariants passing.
Fresh invalid-selector rejection passes. Full frozen-model profiling
proves 94 wide-HC and zero original HC programs, 159 original dense
and twelve cached-attention programs, matching prefixes and zero faults.
Provider:
`47770ecf5d683eb3fa73f86358cb6191e1feb302ade1c3f2d94aa4d1e89548c0`.

The wide-HC CPU-local8 natural 2K screen is mixed and held:

| Validation | Median decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 79.615791 | 3.490947 | 1.002024793 | 2/3 |
| Retained | 79.764739 | 3.486082 | 0.996471918 | 0/3 |

Frozen prefixes and zero faults pass, but neither mode meets its
performance support gate. No owned CPU builds or other owned GPU jobs
overlap this timing. Keep the variant off without an unchanged repeat,
8K/32K escalation or stacking it into the qualified dense configuration.

### Grouped rows4 prefill double buffering (experimental, default off)

`ORT_WEBGPU_QMOE_PREFILL_ROWS4_DOUBLE_BUFFER` targets only the qualified
M5120/M10240, K2560/N1280 grouped-prefill FC1 projection. Four rows,
eight columns, 128 threads, 32 K streams, cached expert weights, original
FP16 dots, per-1024K partial updates, serial folds and optional bias are
unchanged. Two four-row activation tiles alternate; the trailing tile
barrier is replaced by a final barrier before folding. Shared storage
is 18,448 bytes. Unsupported limits, K640/N2560 FC2 and all decode
routes retain original admissions and shaders.

This is distinct from rejected rows8 and single-token FC1 double
buffering. It targets TTFT, not a hypothetical decode speedup.
The selector is process-static, validated at the matrix and both QMoE
surfaces, and included in the QMoE replay key. The private program has
a distinct identity; no frozen model, shared runtime layout or replay
API changes.

Three geometry/actual-renderer/buffer-scheduling tests pass, as do
the independent-comparator schedule test and two all-context model
controller/benchmark tests. The original benchmark remains sealed;
a generated copy adds only the new environment-flag recording entry.
The first build failed before creating provider output because its
comparator came from a different component manifest. The failure log
is preserved; build-v2 uses the rows4 component's pinned comparator.
The corrected build passes SDK/all-152-original-object checks, exact
0955 relink, and byte-identical original rebuilds of both modified
translation units. No numerical or TTFT gain is established yet.

Native qualification is running alone on GPU7 against distinct fresh
original-provider captures, compiled-default and selected variants,
preserving all six actual-weight fixtures, changing inputs/routes,
chunk16 controls and pending1/64 schedules. All eight GPUs were checked
at 143,156 MiB free and zero used with no compute applications before
launch. Full active-prefill profiling and controlled TTFT screens must
follow numerical and selector/replay qualification before escalation.

All three grouped-prefill phases now pass 300,810,240 finite bitwise
comparisons each against chunk16 controls and the fresh separate-original
captures. All six actual-weight fixtures cover rows16/128/256/512/1023/
1024/1025 with the inherited input/route schedule and pending1/64.
Each selected fixture profile has sixteen new FC1 and sixteen original
FC2 rows4 programs; compiled-default/original profiles retain thirty-two
original programs. Fresh invalid-selector rejection passes.

An additional 9,748,480 exact comparisons preserve the original
single-token/multi-token fallbacks with eleven replay instances, 704
requests and 528 hits. That fallback check deliberately reuses the
previously sealed original0955 captures; it is not a new original capture
set. The affected grouped-prefill paths have their own fresh captures.

The first full-model profile verification incorrectly expected two
profile files. The configured profiler is session-scoped: one trace
contains both the excluded warmup and measured request. Its actual
counts are correct: 192 new FC1 and 192 original FC2 programs at 2K,
preserved frozen prefixes and zero faults. A strict existing-evidence
verification corrects only the file-count assumption, validates session
scope and all expected counts, and seals that original run without a
GPU rerun or rewriting its raw/cache evidence. The failed validation log
is retained. All six tests pass again before natural timing begins.

The complementary original-provider/candidate/candidate/original-provider
2K TTFT brackets are now running in both validation modes, with identical
original GenAI runtime and CPU/worker policy. Each load has an excluded
warmup and one measured 36-token request; these are TTFT screens, not
natural 256-output decode headline measurements. All GPUs were again
checked at 143,156 MiB free and no compute applications before launch.
No owned CPU builds or other owned GPU jobs overlap these brackets.

The completed grouped-prefill screen rejects this candidate:

| Validation | Control median TTFT s | Candidate median TTFT s | Matched TTFT reduction |
|---|---:|---:|---:|
| Full | 3.488113479 | 3.978105600 | -14.047560% |
| Retained | 3.485305511 | 3.980152536 | -14.198113% |

All four matched pairs favor the original. Frozen/repeated 36-output
prefixes and fault checks pass, but arithmetic correctness does not
overrule this roughly 14% TTFT regression. The candidate stays held and
default off, without unchanged retries or 8K/32K escalation.

### INT8 LM-head full activation staging (experimental, default off)

The supplied INT4 model's final `/lm_head/MatMul_Q8` is actually INT8:
K2560, N248320, block size 32, UINT8 weights and FP16 scales, with no zero
points or bias. A distinct `ORT_WEBGPU_INT8_LM_HEAD_FULL_A` experiment
stages all 640 vec4 FP16 activation entries once instead of five separate
512K tiles. `Int8LmHeadFullAF16` retains the original 128 threads, eight
output columns, 32 K streams, four ordered FP16 dots per 16-byte packed
chunk, five shared-FP16 partial updates and the final serial 32-lane
fold. Barriers decrease from ten to two; shared storage is 5632 bytes.

Admission is restricted to NVIDIA, FP16 inputs/scales without FP32
accumulation, M1/K2560, complete aligned blocks, exact tensor sizes and
N8/16/32 synthetic or N248320 actual-head shapes. Unsupported shapes and
types retain their original paths. The selector is immutable, validated
as 0/1 in a fresh process, and encoded as replay variant 8. The original
`ApplyMatMulNBits` implementation is unchanged.

All three source/mapping/actual C++ renderer tests pass. The scoped
build verifies the SDK, all 152 original objects, exact original-TU
rebuild and original-provider relink. Candidate provider SHA256:
`ea86b426d4344f45d16bbc0209e6c5d6017bb7bd5bb4b0709531146efe256a65`.

Independent native qualification uses twelve changing synthetic
fixtures and the complete actual head, sixteen rounds and four repeats,
with typed FP16/FP32 references, signed-zero inputs, finite outputs and
repeatability checks. Original capture, candidate flag-off and candidate
flag-on each pass 15,906,816 bitwise comparisons with zero mismatches;
the complete actual head contributes 15,892,480 per phase. All thirteen
fixture profiles pass, including the original shape/FP32 fallbacks.

Infrastructure failures are preserved rather than relabeled successes:
the first external-data symlink was unavailable inside the container;
the second relative link escaped the fixture directory and ORT rejected
it. The valid third fixture set copies precisely 675,430,400 original
weight/scale bytes into fixture-local data, verifying both payload
hashes without modifying the frozen graph/config/data. The original
capture then exposed an incorrect FP32 fallback checker assumption:
the original H200 path is `DP4AMatMulQuantize` plus
`DP4AMatMulNBitsSmallMProgram`, not `MatMulNBits`. Version 4 changes only
that expected path and uses distinct fresh original references; earlier
logs and captures remain intact.

Fresh selector 2 rejection passes with the required diagnostic. The
frozen-model full-validation 2K decode-only profile preserves prefixes
and zero faults: one new INT8 head, zero original `MatMulNBits`, 94
original HC projections, 159 original dense programs and twelve cached
attention programs. It is instrumented evidence, not a speed claim.
Three additional benchmark/controller/all-context CPU-placement tests
pass. The benchmark copy changes only recording of the new selector.

Natural 256-output full/retained 2K brackets are now running with the
same qualified CPU-local8 GenAI runtime, CPU selector 1, taskset 48-55
and inherited worker affinity for both original-provider control and
candidate. All eight GPUs showed 143,156 MiB free and no compute apps
before launch; GPU7 remains isolated. No owned CPU builds overlap
timing. Repeated 2K, then initial/repeated 8K and 32K, require success
in both modes. No dense double-buffer stacking, 100 tok/s achievement,
whole-model quality qualification or default promotion is claimed.

The completed natural INT8-head screen is mixed and therefore held:

| Validation | Candidate median decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 79.660060 | 3.486193 | 1.003002516 | 3/3 |
| Retained | 79.978089 | 3.488362 | 0.997885571 | 0/3 |

Frozen/repeated outputs and major-fault checks pass, but retained
validation fails the gain gate. No unchanged repeat, 8K/32K escalation
or stacking follows. The head remains default off and quality
unqualified. Instrumented attribution reports 229 us for its single
dispatch versus 15,711 us aggregate kernel time; this is not natural
model timing. The largest attributed family is the 94 original HC
projections at 2729 us, motivating a distinct experiment there.

### HC owned-vector staging (experimental, default off)

A new `ORT_WEBGPU_HC_OWNED_VECTOR` candidate retains 256 physical
threads, 41 groups, eight columns, the original 256-stream FP32 RMS
partials/tree, FP16 projection schedule and serial 32-lane output fold.
Only post-mix and normalization staging group four adjacent hidden
elements under one invocation. That invocation writes the complete
padded vec4 once, permitting vector projection reads without
component-level writes by different invocations. The prior vec4,
narrow-four, wide-sixteen and RMS-broadcast candidates are not retried.
Shared storage remains 27,648 bytes. This candidate is prepared, not
yet built, native-qualified, profiled or timed.

All six actual-renderer/ownership/arithmetic/selector/replay/fallback/
benchmark/all-context CPU-placement tests now pass. The exact SDK,
152 original objects, original-TU and original-provider relink build
passes; candidate provider SHA256:
`841ae44f10fd9c2efb2bf9d2f8296bea80b7c3b330ef068b424cf1b0a05ce4d3`.

Fresh separate original capture, candidate default and candidate
selected each pass 607,143,936 finite bitwise comparisons across all
36 fixtures, 128 changing rounds and all four outputs. The shared typed
comparator's I/O proof and original fixtures are revalidated; original
output captures are fresh for this experiment. All three phases retain
the required twelve replay instances and binding/LRU invariants.

Fresh selector 2 rejection and full-validation frozen 2K decode-only
profiling pass. The latter proves 94 owned-vector HC programs and zero
original HC projections, while preserving 159 original dense, twelve
cached attention and one original INT8 head program, frozen prefixes
and zero faults. No held GPU candidate is stacked.

Initial natural 256-output full/retained 2K brackets are now running on
GPU7 with identical CPU-local8 runtime/flag/48-55 worker policy for
control and candidate. All GPUs again showed 143,156 MiB free and zero
compute apps before launch. No owned CPU builds overlap these timings.
No performance, longer-context or quality claim follows from the
completed arithmetic and routing gates alone.

The initial owned-vector CPU-local8 natural 2K screen now supports
both modes:

| Validation | Candidate median decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 80.770964 | 3.487228 | 1.023780935 | 3/3 |
| Retained | 81.225202 | 3.489798 | 1.019261953 | 3/3 |

Frozen/repeated output and major-fault checks pass; matched TTFT ratios
are 1.001166 and 1.000841. The full ending-control load has measurable
request drift, so a fresh repeat is required rather than treating
these initial ratios as final all-context evidence. That 2K repeat is
now running with new candidate/ending-control loads and the initial
ending controls shared as its starting reference; this dependence is
explicit. Successful repeats gate 8K and then 32K. All GPU memory/apps
checks are repeated before batches, and no CPU builds overlap timing.
No median reaches 100 tok/s yet.

The fresh owned-vector repeat fails retained validation and is held:

| Validation | Repeat median decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 80.660363 | 3.482425 | 1.022096014 | 3/3 |
| Retained | 78.065151 | 3.490111 | 0.978016815 | 0/3 |

Frozen/repeated outputs and fault checks pass. The initial positive
screen is not sufficient for promotion; no unchanged repeat, 8K/32K
escalation or stacking follows. No cause for the retained regression is
established.

A distinct `ORT_WEBGPU_HC_OWNED_VECTOR_UNROLL` candidate is prepared.
It emits four constant component offsets for each owned post-mix and
normalization vector instead of dynamic component loops. It retains
the same vector ownership, original RMS streams/tree, physical
geometry, shared bytes and projection/fold arithmetic. It is not an
unchanged retry, and the preceding evidence remains sealed.

The constant-component build and all six renderer/ownership/selector/
replay/fallback/benchmark/all-context placement tests pass. Provider
SHA256 is
`3c262529dcff6c0e2d7040dde4d605cdcff31d89d834cdef29d793bf2eb8a314`;
the original SDK, 152 objects, original-TU rebuild and 0955 relink gates
remain intact. Fresh original capture, default and selected phases
each pass 607,143,936 finite bitwise comparisons with zero mismatches.

The subsequent selector/profile chain was interrupted before creating
selector guard evidence. On November 2 the private Docker daemon was
restored using its existing image/storage, without changing system
Docker or reinstalling dependencies. Original completed native evidence
is revalidated rather than repeated. Fresh selector 2 rejection now
passes. Full frozen 2K decode-only profiling verifies 94 new HC programs,
zero original HC, 159 original dense, twelve cached attention and one
original INT8 head, with preserved prefixes and zero faults.

Initial full/retained natural 256-output 2K brackets are running on
isolated GPU7 with the same CPU-local8 runtime, selector 1 and 48-55
inherited worker placement on both sides. Pre-launch checks show
1.7 TiB host RAM available, 440 GiB disk free and all eight GPUs empty.
No CPU builds or parallel owned GPU jobs overlap natural timing.
Obsolete INT3, NVFP4 and raw download-cache payloads were removed at the
user's request; active INT4/FP8 model identities were verified intact.
No throughput or whole-model quality claim is made for this candidate
until its remaining gates complete.

The constant-component initial natural 2K screen supports both modes:

| Validation | Median decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 81.410898 | 3.485148 | 1.022809099 | 3/3 |
| Retained | 82.101585 | 3.475456 | 1.024022124 | 3/3 |

Frozen/repeated prefixes and major-fault checks pass. Matched TTFT
ratios are 1.000349 and 0.999101. A fresh 2K repeat is now running before
8K/32K escalation; its initial ending controls are reused as starting
references, so the two screens are not fully independent. These
controllers retain their prepared `20261101` filename stamp; the
resumed guard timestamps establish the actual November 2 execution.
No 100 tok/s or whole-model quality claim follows from initial support.

The fresh constant-component 2K repeat supports both modes:

| Validation | Repeat median decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 81.495129 | 3.485195 | 1.026467925 | 3/3 |
| Retained | 82.084050 | 3.469675 | 1.026143990 | 3/3 |

Frozen/repeated prefixes, zero major faults and matched TTFT gates pass
(ratios 1.000741 and 0.997757). All twelve initial/repeat matched
requests are positive. Reused starting controls remain disclosed.
Initial 8K qualification is now running. GPU7 is empty at launch;
a foreign workload on GPU0 uses approximately 24 GiB and is untouched.
Host memory remains approximately 1.7 TiB available and disk 440 GiB
free. This candidate is still default off, not stacked with the earlier
dense double-buffer provider, and not yet supported through 32K.

Initial constant-component 8K qualification also supports both modes:

| Validation | Median decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 81.531034 | 14.140547 | 1.024829493 | 3/3 |
| Retained | 82.443154 | 14.156023 | 1.028059856 | 3/3 |

Frozen/repeated prefixes and major-fault checks pass. Matched TTFT
ratios are 0.998916 and 1.001671; these are small changes, not a material
TTFT improvement. A fresh 8K repeat is running with new candidate and
ending-control loads, reusing initial ending controls as its starting
references. At repeat launch all GPUs are empty, RAM remains 1.7 TiB
available and disk 440 GiB free. No owned CPU builds overlap timing.
Repeated 8K support is required before 32K testing.

Fresh constant-component 8K qualification supports both modes:

| Validation | Repeat median decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 81.729967 | 14.116667 | 1.028095162 | 3/3 |
| Retained | 82.336293 | 14.154444 | 1.027878707 | 3/3 |

Matched TTFT ratios are 0.996942 and 1.000650; frozen/repeated prefixes
and major-fault checks pass. All 24 initial/repeat matched requests
through 8K improve, with shared-control dependence disclosed. Initial
32K qualification is now running on isolated GPU7. All GPUs are empty
before launch, RAM available remains 1.7 TiB and disk free 440 GiB.
No owned CPU builds or parallel GPU jobs overlap natural timing.
The candidate remains default off and quality unqualified.

Initial constant-component 32K qualification supports both modes:

| Validation | Median decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 71.836317 | 59.438558 | 1.041223223 | 3/3 |
| Retained | 71.968650 | 59.444609 | 1.020684270 | 3/3 |

Frozen/repeated prefixes and major-fault checks pass. Matched TTFT
ratios are 0.999835 and 0.998834. The full control bracket exhibits
drift, so its initial 4.12% ratio is not treated as a settled gain.
TTFT remains essentially unchanged. A fresh full/retained 32K repeat
is running with new candidate/ending-control loads and shared initial
ending-control starting references. All GPUs are empty at launch;
host RAM remains approximately 1.7 TiB available and disk 440 GiB free.
No CPU builds or parallel owned GPU work overlap timing. Repeated
32K support remains required for all-context qualification.

The constant-component 32K repeat does not meet the strict retained
request-consistency gate:

| Validation | Repeat median decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 71.951699 | 59.464714 | 1.042854996 | 3/3 |
| Retained | 71.519384 | 59.519651 | 1.008772976 | 2/3 |

One retained request ratio is 0.994014. Frozen/repeated prefixes and
major-fault checks pass, as do TTFT ratios 0.998688 and 0.998980.
Repeated short-context support remains context-specific, not an
all-context promotion. Keep this candidate held/default off without
unchanged retries or stacking. The cause of request variation is not
established. The earlier repeatedly supported dense/CPU-local8
all-context configuration remains unchanged.

Prior-art review finds that scalar-to-vec4 RMS shared layout already
failed its repeat, so it is not reimplemented. A distinct
`ORT_WEBGPU_DSA_FOUR_SUBGROUPS` attention experiment is prepared instead:
four independent original 32-lane logical attention groups share one
128-thread workgroup, retaining 64-candidate partitions, cached query
dot order, per-partition softmax and original merge. This differs from
the earlier rejected two-group/64-thread pairing. Admission requires
single-token batch-one NVIDIA, fixed 32-lane subgroups and head count
divisible by four; the selector is immutable. No arithmetic, model,
throughput or quality qualification is claimed for the new candidate.

All three four-subgroup source/mapping/admission/unchanged shader and
merge tests pass. Exact SDK/152-object/original-DSA-TU/0955 relink build
passes; provider SHA256:
`7512b21176cec798aef164a264b2b878819ef314be5940c264308504c6069449`.
A new typed constant-input original-output comparator compiles using
the existing FP16/FP32 reference codec. Fresh original/default/candidate
phases across 27 preserved fixtures and pending windows 1/64 are now
running serially on GPU7, including profile route checks. Each fixture
has nine fixed-input repeats; changing-input coverage is not claimed.
All GPUs are empty before launch, RAM remains 1.7 TiB available and
disk free 440 GiB. Full-model, fresh invalid-selector and natural timing
gates are still pending; no gain or quality claim is made.

The four-subgroup native batch now passes 2,435,328 finite bitwise
comparisons per complete original/default/candidate phase at each
pending window (six phases total), using 27 fixed fixtures and nine
repeats. Each selected phase profiles 18 new attention programs, 225
original fallback programs and 63 original merge programs; original
and flag-off phases retain 18 cached-query programs instead.

All six source/mapping/admission/benchmark/model-route/all-context CPU
placement tests pass. Fresh selector 2 rejection passes. Full frozen
2K decode-only profiling verifies twelve new attention programs, zero
old cached-query programs, 94 original HC, 159 original dense and one
original INT8 head, with preserved output prefixes and zero faults.
Original/default shaders and fallback routes are not promoted by these
checks; changing-input and whole-model quality coverage remain limited.

Initial natural 256-output full/retained 2K brackets are running with
identical CPU-local8 runtime/flag/48-55 inherited workers on both sides.
All GPUs are empty at launch, RAM available is 1.7 TiB and disk free
439 GiB. Only one owned GPU job runs; CPU builds completed before
timing. Longer contexts require the usual initial and repeat gates.

The four-subgroup attention natural 2K result is mixed and held:

| Validation | Median decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 79.357255 | 3.471924 | 1.002676654 | 3/3 |
| Retained | 79.861218 | 3.471799 | 0.996997071 | 0/3 |

Frozen/repeated prefixes and major-fault checks pass. Matched TTFT
ratios are 0.998948 and 1.000437. Retained validation does not support
a gain; keep default off without unchanged repeat, longer-context
escalation or stacking. No slowdown mechanism is established.
All-context supported performance therefore remains the earlier
dense-double-buffer/CPU-local8 result, not either newly held candidate.

### Expanded natural-request metrics (measurement qualification)

The new metrics-only benchmark and guard copies leave sealed experiments
unchanged. The full-validation reporting batch uses the previously
all-context-qualified dense Columns8 double buffer and CPU greedy lanes8
runtime, with the same local8 worker placement, frozen INT4 graph/config,
256 generated tokens and pending64/chunk1024 policy. It requests ten
measured natural prompts plus one excluded warmup at each of 2K/8K/32K;
this is a descriptive measurement batch, not a new isolated speedup claim.

- TTFT is request start through first emitted token. P50/P95 use linear
  interpolation over measured requests; ten samples are still limited
  evidence for tail latency.
- Prefill is separately timed `append_tokens` host wall time. The pinned
  GenAI implementation calls `ComputeLogits` there; it includes prompt
  processing through logits readiness, but excludes first-token selection.
  Prefill throughput is total input tokens divided by total measured
  prefill time, not input tokens divided by TTFT or GPU-only kernel time.
  The inherited resource-usage phase called `prefill` still includes the
  first generated token and is not this new timing boundary.
- Decode throughput remains total `(output_tokens - 1)` divided by total
  decode wall time, excluding first-token time.
- Peak GPU is NVML device-used bytes converted to GiB, sampled by the
  existing host guard at nominal 0.5-second intervals after idle admission.
  It includes model loading, warmup and measured requests, plus driver
  allocations; it is a sampled lower bound, not an allocator-exact peak.
  Foreign GPU7 work still aborts the owned container only.
- First three requests and the warmup retain frozen-reference checks.
  Seven additional prompts have no frozen output references. Neither
  output-ID agreement nor these metrics imply whole-model quality
  qualification. Historical runs have no recoverable measured GPU peak.

Five CPU-only tests passed: actual generated request timing/unchanged IDs,
warmup exclusion and quantile/aggregate/unit arithmetic, explicit rejection
of invalid/incomplete measurements, and mocked NVML used-memory access
with peak initialization after idle admission, plus recomputing summaries
for the three-request reference subset. All 158 prerequisite
source/evidence hashes and inherited frozen-output checks passed before
launch. The result is persisted progressively with exact source and
runtime/provider/evidence hashes.

The reporting batch completed all 30 measured requests and three excluded
warmups. First-three/warmup frozen references, CPU placement, exact
provider/runtime and measured-request major-fault checks passed at each
context. The initial post-run validator incorrectly compared the
three-request subset against the ten-request summary; it was fixed and
regression-tested. The already completed 2K run was revalidated and reused,
not rerun. Original failure evidence is retained.

| Context | Requests done | TTFT P50 s | TTFT P95 s | Prefill tok/s | Decode tok/s | Sampled peak GPU GiB |
|---|---:|---:|---:|---:|---:|---:|
| 2K | 10/10 | 3.463 | 3.475 | 591.265 | 78.653 | 68.979 |
| 8K | 10/10 | 14.067 | 14.195 | 581.252 | 79.801 | 69.325 |
| 32K | 10/10 | 59.605 | 59.803 | 549.353 | 70.264 | 70.671 |

These are fresh full-validation aggregate decode rates, not the
three-request medians used by the optimization gates. They do not replace
matched before/after candidate qualification. No >100 tok/s claim or new
isolated improvement is supported. Retained-validation metrics have not
been newly measured by this batch. All GPUs were empty afterward; host
available RAM remained approximately 1.7 TiB and disk free space 439 GB.

### AVX512 greedy argmax build and frozen-prefix gate (not performance-qualified)

The distinct default-off `ORT_GENAI_CPU_GREEDY_ARGMAX_AVX512` runtime
build completed with SHA256
`74cc5589de32f8a913a58887eb73f3bf4bbb1c3cd66320c6bc98ddcd5aab1e9f`.
The actual-C++ standalone test passed 270337 cases / 32271653 elements
against both original and lanes8 selected IDs. Five CPU-only brackets
showed roughly 7x faster vocabulary scans, approximately 85 to 12
microseconds; this is not a model-throughput gain. Fresh attempt2 subsequently
passed real invalid-selector rejection and the full frozen 2K output-prefix
gate on GPU7, using the previously supported GPU double buffer and local8
worker policy. Attempt1 was interrupted after writing the expected selector
error but before its guard finalized; its raw running guard is not accepted
as completed evidence and is preserved. The sealed CPU proof and both
attempt2 proofs were revalidated against their source/runtime hashes.

Two focused matched-controller tests pass for all three contexts and both
validation modes. The completed natural screen used the same AVX512
runtime on both sides, with lanes8 enabled in both and only AVX512 changing
from 0 to 1. It does not compare the new runtime against the old original
runtime. Fresh repeat consistency failed as recorded in the final verdict
below. The metrics batch explicitly
disables this selector and does not combine an unqualified candidate with
the supported GPU configuration.

### Parallel16 HC projection build (not model-qualified)

The distinct default-off `ORT_WEBGPU_HC_PARALLEL16` candidate uses 512 physical
threads, 16 concurrently projected columns, and 21 groups. This is not the
previously rejected 256-thread wide16 implementation with two sequential
column slots. The first 256 invocations retain the original post-mix,
normalization, and 256-stream RMS accumulation/tree; all 512 participate in
uniform barriers and the original ordered FP16 projection and serial32 fold.
Admission requires NVIDIA, 512-thread limits, and 28160 shared bytes.

Five actual-C++ shader-renderer, ownership, lane/tail mapping, selector, and
fallback tests pass. The exact SDK/152-object/original-HC-TU/0955 relink build
passed, producing provider SHA256
`0429a4b42f10ae17535a78a8f323e7a16b2dd487b2f9e431b98cb35030edfe18`.
Independent changing-input native arithmetic completed on GPU4: fresh
original captures, candidate default-off, and selected candidate each passed
607143936 finite bitwise comparisons across 36 fixtures / 128 changing
rounds, with zero mismatches in all four outputs. Each phase also passed
replay/binding/LRU activity checks. The completed outcome and every captured
reference/source hash were revalidated. Selected full-model shader routing
subsequently passed as recorded in the final verdict below.
The first setup attempt had an incorrectly renamed inherited proof path;
the second was blocked before GPU child creation because the sealed
container wrapper excludes GPU6. Both failure logs are retained. The fresh
GPU4 attempt uses separate captures and prefixes without modifying that
wrapper. Arithmetic and routing agreement are not a throughput gain; the
final performance verdict is recorded below, with no held-candidate stacking.

The generated model controllers retain exact frozen-output/profile gates,
including rejection of a fallback-only candidate profile. All 12 natural
context/mode/validation command-policy checks pass, with lanes8/local8 kept
on both sides and AVX512 explicitly disabled for HC experiments. Their
benchmark copy carries the separately timed prefill and sampled peak-GPU
instrumentation without changing the sealed ten-request benchmark.

The completed bounded progression waited for native proof, ran fresh
HC selector rejection and full frozen 2K routing, and then serialized AVX512
and HC natural screens. Three progression tests verify initial/repeat order,
halt without unchanged retry after failed consistency, and no longer-context
escalation after a failed repeat. Every screen checks persisted evidence
hashes; full and retained runs use three measured requests plus an excluded
warmup. Native GPU work and CPU builds are not overlapped with natural timing,
and no automatic default promotion or candidate stacking is performed.
At admission, all eight GPUs were free; during the small GPU4 native proof,
host RAM remained approximately 1.7 TiB available and disk free space 438 GB.

### Final AVX512 and parallel16 performance verdicts

The completed progression and every natural-screen evidence hash were
revalidated. Parallel16 passed fresh selector rejection and frozen 2K
routing: 94 new HC dispatches, zero original fused HC dispatches; original
dense/attention/head routes and output references remained intact.

| Candidate / screen, 2K | Validation | Median decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---|---:|---:|---:|---:|
| AVX512 initial | Full | 80.496928 | 3.469385 | 1.004363960 | 3/3 |
| AVX512 initial | Retained | 80.887665 | 3.464763 | 1.005191544 | 3/3 |
| AVX512 repeat | Full | 80.291514 | 3.467429 | 1.003947177 | 2/3 |
| AVX512 repeat | Retained | 80.931627 | 3.467745 | 1.006507271 | 3/3 |
| Parallel16 initial | Full | 78.328261 | 3.467101 | 0.985578252 | 0/3 |
| Parallel16 initial | Retained | 78.705197 | 3.468445 | 0.986631235 | 0/3 |

AVX512 fails fresh full-mode consistency despite positive aggregate ratios;
parallel16 regresses both modes. Both are held/default-off, with no unchanged
retry, longer-context escalation, or stacking. TTFT/output/fault checks do
not override the failed decode gates. The previously supported GPU double
buffer plus CPU lanes8/local8 configuration and ten-request metrics remain
the best all-context baseline; 100 tok/s remains unmet.

### Separate normalization with eight-column projection (not qualified)

The next distinct `ORT_WEBGPU_HC_SEPARATE8` candidate retains the original
256-stream RMS/tree but computes normalization once in four workgroups
instead of repeating it in every fused projection workgroup. It then runs
the original ordered FP16 projection with eight columns / 256 threads /
41 groups and 512 shared bytes. This differs from the old held staged
projection's two columns / 64 threads / 162 groups.

Both stages are captured together by the original replay instance using all
four original output buffers, including the normalized activation used by
projection. No extra persistent buffers, replay API changes, or existing
kernel object layout changes are introduced. Four actual-C++ normalization
renderer, projection mapping/tail, original fallback, and replay tests pass;
the exact 152-object/SDK/original-TU/0955 relink build passed, producing provider
SHA256 `03cf13d6fd7df5f0104a153002289d1ef2f6579e8467c4a5b65014b4c40210e5`.
All 12 context/mode/validation command-policy checks pass, including fixed
lanes8/local8 and explicit AVX512-off. Profile checks require both new stages
at 94 dispatches, not merely the projection stage. Fresh independent native
arithmetic is active on GPU7; selected frozen routing and isolated natural
initial/repeat progression are queued behind that prerequisite.
The candidate remains default-off and is not combined with held candidates.

### Separate8 replay constraint and corrected per-stage replay

The first separate8 build passed fresh original and default-off phases,
607143936 exact comparisons each. Its selected eligible path failed before
arithmetic qualification: the frozen `StaticNodeReplay` explicitly requires
one captured command and exact parent-buffer binding order, but the combined
normalization/projection capture contained two. This is an implementation
failure, not a numerical pass or performance result. The original failure
log/guard/captures remain sealed and its waiting progression was stopped.

The corrected distinct `ORT_WEBGPU_HC_SEPARATE8_STAGE_REPLAY` candidate keeps
both shader schedules unchanged but uses one replay per stage. Normalization
captures only its two outputs. Projection captures its two outputs and uses
the existing private stage-replay mechanism to include normalized activation
in both the key and the first shader-input binding. The original replay
implementation and original fused/fallback paths are unchanged.

This necessarily adds private leaf-kernel replay metadata and changes that
leaf's object layout; the build manifest explicitly records that deviation.
Runtime object layouts, headers used by other translation units, and all
152 original source-object/SDK provenance checks remain unchanged. No
additional persistent GPU buffers are introduced. Seven original-renderer
and single-command/binding-order tests pass, and the exact original-TU/0955
relink rebuild passed. Fresh corrected native/model/performance qualification
is still required; the failed version is not retried unchanged.

The corrected provider SHA256 is
`3d862f71764ad0860ff8df33d668043884154c23646d1ede1d2cd37048395d34`.
All 12 context/mode/validation command-policy checks pass. Corrected native
qualification requires 24 selected replay instances (six each for shared
and non-shared normalization and projection), versus 12 original/default
instances, with the same per-instance binding/rebind/LRU checks. Frozen
profiling must still observe 94 of each new shader and no original fused HC.
The fresh GPU7 arithmetic job is active; model and isolated natural screens
are queued behind successful completion. The first failed build is not
accepted as arithmetic evidence for the corrected implementation.

The corrected GPU7 native batch subsequently completed: original captures,
default-off, and selected candidate each passed 607143936 finite bitwise
comparisons across all 36 fixtures / 128 changing rounds. The selected phase
also passed all 24 stage-replay instances with 2304 requests and 744
binding hits; original/default each passed 12 instances with 1152 requests
and 372 binding hits. Completed capture/reference/source hashes were
revalidated. Fresh corrected selector rejection passed. Full frozen 2K
routing is now active, followed only on success by isolated natural timing.
No model-throughput or whole-model quality gain is claimed by this proof.

### Final separate8 stage-replay performance verdict

The corrected candidate passed frozen 2K routing with 94 normalization and
94 projection dispatches, and no original fused HC dispatches. Its initial
natural screen was negative:

| Validation | Median decode tok/s | Mean TTFT s | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|
| Full | 78.582819 | 3.472595 | 0.986419514 | 0/3 |
| Retained | 76.478170 | 3.464578 | 0.964606747 | 0/3 |

The completed screen/source hashes were revalidated. Both modes regress, so
the candidate is held/default-off with no unchanged retry, repeat,
longer-context escalation, or stacking. Arithmetic/replay/output/fault
qualification does not override this measured throughput failure.

### Concurrent word/half native matrix staging (not qualified)

The next distinct default-off `ORT_WEBGPU_DENSE_PAIRED_MATRIX_BATCH8` candidate
spatially separates the eight original word/half slots across 32 fixed
32-lane subgroups / 1024 threads instead of serializing those slots inside
four subgroups. It retains the calibrated materialized-first-product,
guarded native pair arithmetic, original serial four-word FP16 accumulation,
original tile accumulation, and final serial32 fold. Four owner subgroups
reduce the half-dot staging in the original order.

Shared storage is 91136 bytes; admission explicitly checks that storage and
1024-thread limits. Four actual-C++ renderer, unique-storage/owner mapping,
original reduction-order, and default/fallback tests pass. The exact
SDK/152-object/original-TU/0955 relink build passed. A small K640/N2560 probe
will require 64 actual selected native dispatches before any broader suite:
exact original-output agreement through fallback alone is not sufficient.
The completed small probe found zero selected native dispatches despite
163840 exact fallback comparisons. It therefore failed the selected-kernel
gate, not arithmetic qualification; no broader suite or model timing ran.

An arithmetic-unchanged, default-off device-limit diagnostic then observed
49152 workgroup-storage bytes, 1024 X threads / total invocations, and an
available paired F16/F32 16x16x16 fixed32 matrix configuration on GPU7.
All 64 diagnostic observations agreed and all 163840 default-kernel
comparisons remained exact. Thus the batch8 storage requirement exceeds the
actual WebGPU limit, irrespective of CUDA's larger shared-memory capacity.
The candidate is held/default-off; fallback matches are not native proof.

### Two batches of four concurrent native matrix slots (not qualified)

The distinct `ORT_WEBGPU_DENSE_PAIRED_MATRIX_BATCH4` candidate uses 16 fixed32
subgroups / 512 threads and 46080 shared bytes, below the observed 49152-byte
limit. Two uniform batches each stage two words / both halves concurrently.
Owner subgroups accumulate all four individual word-pairs in the original
FP16 order; they do not add separately rounded batch aggregates. A uniform
barrier after owner reads protects shared storage before the next batch.
Original tile accumulation, calibrated guarded native-pair arithmetic,
private zero initialization, and final serial32 fold remain unchanged.

Four actual-renderer, per-batch unique-writer, original rounding-order,
storage-reuse barrier and admission/fallback tests pass. Exact
SDK/152-object/original-TU/0955 relink build passed. A fresh one-shape probe
must still prove 64 actual selected native dispatches and exact outputs;
observed device limits alone do not qualify shader execution or performance.
No model timing or TPS gain is claimed.

The fresh probe completed: 64 new native dispatches and 163840 finite
bitwise comparisons with zero mismatches at K640/N2560. Isolated
instrumented kernel medians were 11 us original versus 22 us selected
batch4 (2.0x slower). This is one shape, not broad arithmetic coverage or
natural model TPS. The candidate is held/default-off with no unchanged
retry, broad escalation, or stacking. Spatial staging reduced synchronization
but did not turn the sparse paired-matrix construction into a GPU speedup.

### Constant-word scheduling on supported dense double buffering (held)

`ORT_WEBGPU_DENSE_COLUMNS8_WORD_UNROLL` replaces the dynamic four-word
loop and packed-vector indices with four ordered constant-component
steps. It retains eight columns, 128 threads, two activation buffers,
original per-word/tile FP16 additions and the serial32 fold. The supported
double-buffer program remains byte-identical in the provider, with
independent private replay variants for selected and parent paths.
This is not the older two-K-tile ordered-unroll experiment.

Two renderer/default/selector tests and exact provenance build passed.
Fresh original/default/candidate phases each passed 3067968 synthetic
and 3067904 actual-weight finite bitwise comparisons. Each selected phase
proved 64 new dispatches at each of seven eligible shapes; N320/N1
fallbacks remained original. Compiled default phases explicitly proved
the supported double-buffer route and matched the independent original.

Actual-weight isolated medians were tied at five eligible shapes and
regressed at K2560/N16384 (22 to 23 us) and K2560/N6144 (14 to 15 us).
There is no consistent isolated gain. The candidate is held/default-off
without a natural model screen, unchanged retry or stacking.

### XOR-swizzled supported dense activation storage (not qualified)

The distinct `ORT_WEBGPU_DENSE_COLUMNS8_SWIZZLE` changes only activation
storage addresses inside the supported eight-column double-buffer shader.
For logical vector index `i`, stream `s = i / 8`, it uses
`8*s + ((i%8) ^ ((s>>1)&7))`, independently within each 256-vector buffer.
The mapping is bijective and leaves global input ownership, all FP16
dot/word/tile arithmetic, partials, final fold, storage size and barriers
unchanged. It does not stack the held constant-word scheduling.

For the modeled NVIDIA 32 four-byte banks and eight-byte f16 vectors,
32-lane fixed-word bank-start multiplicity drops from 16 to 2; staging
writes retain multiplicity 2. This is a bank-address model, not a measured
hardware speedup. Selection is default-off and NVIDIA-gated; the original
supported parent program remains intact.

Four bijection/repeated-buffer roundtrip, bank-address-model, actual-renderer
arithmetic/barrier and default/admission tests and the exact scoped
provenance build pass. Fresh selected synthetic/actual-weight proofs and
timings remain required before any model screen or throughput claim.

Both fresh swizzle suites completed with zero mismatches in all three
phases: 3067968 synthetic and 3067904 actual-weight comparisons per phase.
Selected and compiled-parent dispatch coverage passed. Actual-weight
medians tied at five eligible shapes and regressed at K2560/N16384
(22 to 23 us) and K2560/N10240 (17 to 18 us). The bank-address model did
not produce an isolated gain. Held/default-off; no natural model screen,
unchanged retry or stacking. The additional generated-controller test
also passes, for five tests in total.

### Paired private totals on supported dense double buffering (not qualified)

`ORT_WEBGPU_DENSE_COLUMNS8_PRIVATE_TOTALS` retains the supported
eight-column / 128-thread / two-activation-buffer schedule, but keeps two
explicitly zero-initialized `vec2<f16>` totals per invocation. It performs
the same per-word sums and per-tile FP16 additions and writes each of
the 256 final shared partials once, before the existing final barrier
and original serial32 fold. This removes repeated shared partial
read/modify/write operations; it does not change weights or quantization.

The prior private-partial template used four columns and one scalar
partial per invocation. This is a distinct paired-vector/eight-column/
double-buffer schedule; held word-unroll and swizzle arithmetic are not
stacked. The supported parent program remains intact, with separate
private replay variants. Shared storage remains 4608 bytes.

Three renderer, unique-final-writer and default/fallback tests and the
exact scoped build pass. A fourth generated-controller test is prepared.
Private-register FP16 arithmetic still requires fresh GPU bitwise
qualification; neither source-level typing nor prior scalar-partial
evidence establishes exactness or throughput here.

Both fresh private-total suites completed with zero mismatches in all
three phases: 3067968 synthetic and 3067904 actual-weight comparisons
per phase. Selected/default-parent and fallback coverage passed. All four
tests pass. Actual isolated medians tied at four eligible shapes and
regressed at K2560/N7168 (15 to 16 us), N6144 (14 to 15 us) and N10240
(17 to 18 us). Held/default-off without a model screen, unchanged retry
or stacking. These remain instrumented kernel screens, not natural TPS.

### Spatial 16-column dense double buffering (not qualified)

The distinct `ORT_WEBGPU_DENSE_COLUMNS16_SPATIAL_DOUBLE_BUFFER` uses 512
threads and 16 simultaneous columns, one column per invocation. Earlier
Columns16 used 128 threads and four sequential columns per invocation;
the supported Columns8 double buffer uses 128 threads and two columns.
This tests greater physical column concurrency on eligible wide dense
shapes, not the previously held HC parallel16 geometry with only 21 groups.

Both activation buffers, original four-word FP16 additions, per-tile
shared partial updates and final serial32 fold remain. The first 256
threads each stage one activation vector; all 512 threads participate
in barriers and uniquely own the 512 partials. Storage is 5120 bytes.
Admission requires NVIDIA, 512-thread/5120-byte limits and N divisible
by 16, in addition to the supported Columns8 route. Unsupported/default
paths retain the supported parent or original fallback. No held private
totals, constant-word or XOR-swizzle arithmetic is stacked.

Four renderer, unique-writer, exact-admission/default and timing-comparator
tests and the exact scoped provider/comparator build pass. A fifth
generated-controller/matched-ratio test is prepared. Fresh GPU arithmetic
remains required.

This build also adds an unprofiled isolated comparator. It disables
profiling, excludes eight warmup runs, checks every finite output against
the independent original reference, and separately times synchronous
host Run including upload/readback. A prepared controller uses the same
provider with flag0/flag1, CPU48, three before/candidate/after triplets
per actual eligible shape and 1024 measured runs per process. These
latencies are not GPU-only times or natural model TPS. They cannot
qualify 100 tok/s; full frozen-model routing and natural screens remain
required after consistent primitive support.

Spatial16 synthetic/actual suites completed exactly: 3067968/3067904
finite bitwise comparisons per original/default/candidate phase, with
selected/default/fallback coverage. All five tests pass. The three
unprofiled triplets at each of seven actual eligible shapes failed
all-shape consistency:

| K / N | Matched host Run/readback speed ratio | Positive triplets |
|---|---:|---:|
| 640 / 2560 | 1.000216247 | 2/3 |
| 2560 / 2560 | 1.003459355 | 3/3 |
| 2560 / 7168 | 0.996175547 | 0/3 |
| 6144 / 2560 | 1.002244692 | 2/3 |
| 2560 / 16384 | 1.000612558 | 2/3 |
| 2560 / 6144 | 1.001005464 | 3/3 |
| 2560 / 10240 | 0.999794146 | 2/3 |

This primitive run left the optional small-upload/readback-cache and
uniform-arena flags at their fixture defaults; it is not the fully
configured model workload. The latencies include synchronous CPU
transfers/readback and must not be relabeled as GPU-only or natural
TPS gains. Held/default-off with no model screen or unchanged retry.

### Spatial eight-column dense double buffering (not qualified)

The orthogonal `ORT_WEBGPU_DENSE_COLUMNS8_SPATIAL_DOUBLE_BUFFER` keeps
the supported eight-column workgroup count and 4608-byte storage but
uses 256 threads / one column per thread, versus the supported 128
threads / two columns. It separates physical column parallelism from
the group-count reduction in Spatial16. Activation writers and final
partials are unique; original FP16 word/tile/shared-partial arithmetic,
two buffers and barriers are unchanged.

Three renderer/writer/admission/default tests and the exact scoped
build pass. A fourth GPU4 UUID/guard-controller test passes. Fresh
GPU4 original/default/candidate suites each passed 3067968 synthetic
and 3067904 actual-weight bitwise comparisons. Eligible selected
programs and supported-parent defaults have full coverage. This is
not stacking Spatial16 or earlier held arithmetic variants.

Actual isolated GPU medians improve at K640/N2560 (11 to 10 us), tie
at three eligible shapes and regress at K6144/N2560 (16 to 17 us),
K2560/N16384 (22 to 24 us) and K2560/N10240 (17 to 20 us). These
instrumented results do not establish natural throughput.
A fresh same-provider, CPU48, three-triplet unprofiled controller is
prepared for GPU4, explicitly enabling the qualified small-upload,
small-readback-cache and uniform-arena flags on both sides. Its
host Run/readback timings still cannot qualify model TPS or TTFT.

The fresh GPU4 unprofiled comparison completed with all outputs checked
against independently captured originals. Each process excluded eight
warmups and measured 1024 synchronous runs; three before/candidate/after
triplets used the same provider and transfer/cache flags.

| K / N | Matched host Run/readback speed ratio | Positive triplets |
|---|---:|---:|
| 640 / 2560 | 0.998474310 | 1/3 |
| 2560 / 2560 | 0.996006313 | 1/3 |
| 2560 / 7168 | 0.997253305 | 1/3 |
| 6144 / 2560 | 0.997885463 | 0/3 |
| 2560 / 16384 | 0.996019946 | 0/3 |
| 2560 / 6144 | 0.989917953 | 0/3 |
| 2560 / 10240 | 0.999784758 | 1/3 |

All seven ratios are below one. Spatial8 remains held/default-off;
there is no natural model screen, unchanged retry, stacking or new
qualified model-throughput claim. The evidence outcome is
`dense-columns8-spatial-double-buffer-unprofiled-v1-outcome.json`;
provider SHA256 is
`19364937a7bb38f0683ff54d6ad099065ef19bdc2e2c190a4dbd602523d6ef6c`.

### Integer-key parallel GPU ArgMax primitive (not model-qualified)

The current GenAI model path allocates WebGPU logits on the CPU when
graph capture is disabled. Its CPU greedy search eagerly reads logits
before selecting a token. The existing WebGPU ArgMax is a serial scan
per output row, not the CPU greedy path.

A default-off two-stage primitive first reduces 2048-element chunks
with 128 threads, then folds their winning keys/indices. An initial
floating-comparison probe exposed a pre-existing GPU-versus-CPU
subnormal difference: the original GPU returned index0 rather than
index16 for the smallest positive f32 subnormal at the last position
of a 17-element row. That prototype was not promoted.

The distinct `ORT_WEBGPU_ARGMAX_PARALLEL_BITWISE` compares integer
IEEE-order keys, canonicalizes signed zeros, explicitly skips later
NaNs and preserves sticky first-NaN behavior and earliest ties. It
uses f16 bitcasts rather than f16 arithmetic to build keys. No public
header/object layout or frozen model changed.

Five arithmetic/renderer tests and the exact 152-object, original
reduction-TU and 0955-relink provenance checks pass. Ten f16/f32
fixtures cover rows1/3 and lengths17/32/2049/248320. Each
original/default/selected phase passed 1152 independent CPU-reference
token comparisons, with 64 selected dispatches of each stage per
eligible shape and unchanged small-row fallback coverage. Selected
eligible shapes include subnormal inputs. Original/default and
length17 fallback checks explicitly omit the known baseline
subnormal-divergence pattern; this is not a claim that the old GPU
ArgMax has complete CPU semantics.

At one row / 248320 entries, isolated instrumented GPU medians are
7000us for serial f16 and 8995us for serial f32, versus 13us partial
plus 9us finish for either selected type. These compare against the
serial GPU operator, **not the current model's CPU greedy search**,
and do not establish model decode TPS or TTFT.

Provider SHA256:
`28b6d89853d3239d57aa6544847eecd6882607410cdf7ab4b6fb7daedabb4dbc`.
Native outcome: `argmax-parallel-bitwise-v4-native-outcome.json`.
F16-to-F32 conversion/ArgMax compatibility, invalid-selector rejection
and fresh unprofiled primitive timing are being checked separately.
Runtime ownership/search wiring and natural model gates remain pending;
the primitive is default-off and not model-eligible.

The primitive follow-up completed. A GPU F16->F32 Cast followed by
integer-key ArgMax preserved all64 CPU-reference IDs, including the
smallest f16 subnormal and NaN/tie cases; the invalid selector was
explicitly rejected. Three unprofiled before/candidate/after triplets
per type measured1024 runs/process after eight excluded warmups:

| Input | Matched host Run/readback speed ratio | Positive triplets |
|---|---:|---:|
| F16 | 46.410297146 | 3/3 |
| F32 | 45.706116624 | 3/3 |

Again, the control is the serial **GPU** ArgMax operator, not current
CPU greedy model execution. These large operator ratios must not be
presented as model-throughput gains.

### Private-leaf GPU greedy runtime integration (2K routing admitted)

`ORT_GENAI_WEBGPU_GREEDY_ARGMAX=1` requires the integer-key shader flag.
It changes only the gated WebGPU logits/scoring placement, leaving
CPU input/embedding/Engram placement untouched. A private greedy
search subclass retains GPU scores until CPU access is required.
A private lazy DeviceBuffer materializes scores on any CpuSpan/copy
access, including existing sampling, penalty and API paths. Once
materialized, CPU scores are authoritative for that search step;
readback must not overwrite CPU penalties or implicitly upload stale
scores. The next SetLogits starts a fresh lazy search-step view.

The device ArgMax uses the existing GraphExecutor and returns only
int64 token IDs to the CPU. Gated GPU helper sessions use one CPU
thread rather than creating default-sized pools for Cast and ArgMax.
An optional default-off trace reports GPU selection versus CPU fallback;
it is for routing admission, not natural performance measurements.

Three lazy-lifecycle/default-placement tests pass. The scoped build
reproduced the original Model, WebGPU interface and GraphExecutor
objects and relinked the supported lanes8 GenAI and dense double-buffer
provider byte-identically before substitutions. Public headers and
object layouts are unchanged; only new private search/buffer leaves
are introduced. Original 0955/152-object and integer-key primitive
provenance remain prerequisites.

The v3 frozen2K admission preserved generated prefixes but exposed
1,024 CPU fallbacks: the benchmark's required minimum length accesses
CPU scores to suppress EOS. The setting was not removed. The v4 private
search leaf implements the same suppression with GPU Where and float32
lowest, not negative infinity. Invalid EOS IDs are skipped as in the
CPU helper; later CPU access sees the masked logical scores.

Eight native F16/F32 Cast/Where/ArgMax fixtures pass 1,024 independent
CPU-reference ID comparisons, including ties, NaNs and an unmasked
subnormal. Fresh v4 invalid/default/selected frozen2K admission passes:
the selected run matches original generated prefixes and executes
1,024 GPU ArgMax calls and 1,024 minimum-length masks with zero CPU
fallbacks. This is routing/prefix admission, not whole-model quality
or natural performance qualification. The runtime/provider hashes are
`e9cc90725af6db49045a1611b248e13831d639c744066059f559ac3e4dcf40fd`
and `5f818217939e459f5b98e4824cd77fd010b7f4d90ef6200e34254076f4d2b5c4`.

The first untraced screen copies raw logits to CPU to synchronize
prefill. Although the API avoids materializing lazy search scores,
the raw logits shape includes the full sequence, so the copy introduces
artificial TTFT and memory overhead. Its results must not be used as
natural application TTFT or escalated to longer contexts.
A fresh benchmark instead timestamps prefill after first-token
selection, whose token-ID readback guarantees GPU readiness without
copying scores. Its explicit scope is
`append_tokens_through_first_token_selection`: an inclusive prefill
upper bound, not isolated prefill kernel time. TTFT remains request
start through first emission; decode excludes the first token.
Corrected initial/repeat screens support GPU greedy decode at 2K and
8K in both full-validation and retained-validation modes. Repeats
reuse the initial ending controls, so they are not wholly independent.
The repeat full-validation results, three measured requests plus an
excluded warmup, are:

| Context | Aggregate decode tok/s | Prefill tok/s | TTFT P50/P95 s | Sampled peak GPU GiB |
|---|---:|---:|---:|---:|
| 2K | 82.644709 | 586.958845 | 3.487797 / 3.492673 | 68.956055 |
| 8K | 82.226357 | 587.748095 | 13.937964 / 13.951037 | 69.298096 |

Prefill includes first-token selection. Peak GPU memory is sampled
every 0.5 seconds over container lifetime, including load and warmup.
These differ in scope/statistic/request count from the earlier
ten-request CPU-greedy table. The progressive 32K screen finished
with a performance hold, not a repeat-qualified gain. Its initial
full candidate measured 71.922762 decode tok/s, 528.585496 inclusive
prefill tok/s, TTFT P50/P95 62.025164/62.055126 s, and sampled peak
70.683044 GiB. Matched decode ratios were 1.019367 full and 1.034948
retained, each positive in all three requests, but matched TTFT
ratios 1.041516/1.041117 exceeded the 1.02 gate. No unchanged 32K
repeat was launched; GPU greedy is not supported at 32K as a
standalone performance configuration.
CPU builds and small GPU4 native probes overlapped
portions of the v4 pipeline; pristine isolation is not claimed.
No 100 tok/s or whole-model quality claim is made.
INT4 model and Dawn/FP8 assets remain unchanged.

### Private coalesced minimum-length helper (native only)

Runtime v5 optionally executes an ordinary two-node Where/ArgMax graph
under `ORT_GENAI_WEBGPU_GREEDY_ARGMAX_FUSED_MIN_LENGTH`. Masked F32
scores remain materialized on GPU and visible through the existing
lazy API; search-owned INT64 token IDs are cached for greedy selection.
CPU score access, penalties, API replacement and sampling retain CPU
authority. SetLogits invalidates cached selection. Public headers,
layouts, provider shaders and frozen models are unchanged.

Native F16/F32 fixtures at vocabularies 32, 2049 and 248320 pass 216
ID/behavior checks and 14,022,456 API score-element checks across
flags 0/1. Real fixed-seed topK2 sampling agrees across flags.
Flag2 rejects. Public GetLogits computes next-step logits after the
previous logits have been consumed; it does not return the previous
masked scores. Native checks therefore compare the correct next step.

All non-NaN F32 bits and NaN locations match the independent CPU
fixtures. Existing GPU F16-to-F32 Cast canonicalizes NaN payloads
differently from the CPU conversion, also with fused flag0; F16 NaN
payload CPU identity is explicitly unqualified. F32 payload changes
still fail. Frozen-model and natural v5 admission remain pending.
The fused helper is not stacked into the sliced-copy experiment.

### Private aligned GPU sliced copy (repeated 2K/8K TTFT admitted)

Logits::Get extracts the final non-pad row using DeviceSpan::CopyFrom.
The original WebGPUMemory nonzero-offset path calls CopyThroughCpu,
which downloads the entire source allocation before uploading the
requested row. A private default-off alternative,
`ORT_GENAI_WEBGPU_GPU_SLICE_COPY`, uses ordinary ONNX Slice on
UInt32 words, preserving raw bits without changing model arithmetic.

The source must be an actual WebGPUMemory allocation, not a lazy
CPU-authoritative search buffer. Allocations must be distinct,
destination offset zero, source offset nonzero, all relevant sizes
four-byte aligned, and full-source word count within UInt32 range.
Other cases retain the original fallback. Slice starts/ends are
explicit CPU INT64 inputs; data/output remain GPU tensors. CPU EP
fallback is disabled for this helper, preventing a success-shaped
full-buffer CPU transfer if GPU UInt32 support is missing.

The provider's default-off `ORT_WEBGPU_SLICE_UINT32_BITCOPY` extends
only Slice13's type registration with UInt32; floating registrations
and shader code are unchanged. Original Slice object and composite
provider relink reproduce byte-identically before substitution.
Runtime v6-r2 reproduces the original four GenAI translation units
and relinks v5 byte-identically before its private changes.
Public headers/layouts and frozen model assets are unchanged.

Across flags 0/1, native two-token F16/F32 fixtures pass 216 behavior
checks and 28,044,912 initial/next API score-element checks. Selected
routing executes 160 GPU slices, including the appropriate fallback
for unaligned F16 vocabulary2049. Penalties, API replacement and
fixed-seed sampling agree; flag2 rejects. Non-NaN bits and NaN
positions are exact; the same F16 NaN-payload limitation remains.

A separate native test copies from a 16,273,899,520-byte segmented
buffer with starting word index 4,068,226,560, above signed INT32.
Four cases verify 993,280 F32 score elements bitwise, including
NaNs, signed zeros, ties and subnormals. Only the 993,280-byte
requested row is copied; sampled peak GPU memory is 15.857910 GiB.
This qualifies large unsigned indexing, not frozen-model TTFT.

The runtime/provider hashes are
`d061518f7523417dacff57a6e69e77a0558fe6ec237b35433edac9ce8776b8d9`
and `487e11724e576b27212b022a46f4e2b8258d63227ac28d5692a8fca0677a0ef1`.
Frozen2K control/selected routing admission passed against the original
generated prefixes. Selected routing performs four copies, each taking
496,640 bytes from offset 508,062,720 in a 508,559,360-byte source.
That source is the final 1024-token prefill chunk, not the entire
2048-token prompt. Do not extrapolate its size or the cause of the
held 32K TTFT regression without profiling.

Initial and repeat natural 2K/8K TTFT screens pass in both validation modes.
The full-validation repeats report:

| Context | Decode tok/s | Prefill tok/s | TTFT P50 / P95 (s) | Sampled peak GPU (GiB) |
|---|---:|---:|---:|---:|
| 2K | 82.170830 | 650.539557 | 3.147457 / 3.151668 | 68.956055 |
| 8K | 81.627116 | 600.204393 | 13.629189 / 13.728585 | 69.297241 |

These are three measured requests plus excluded warmup, 256 outputs
per request, and 255 divided by measured decode seconds. Prefill includes
append_tokens through the first token selection; it is not isolated
kernel throughput. Peak GPU is sampled every 0.5 seconds over container
lifetime including load/warmup, not an instantaneous maximum.
Full-repeat matched TTFT ratio is 0.900759 and decode ratio 0.995244.
Retained-repeat ratios are 0.909909 and 0.991869 respectively.
At 8K, full-repeat matched TTFT/decode ratios are 0.979236/0.993789;
retained-repeat ratios are 0.969633/1.004267.
All three requests improve TTFT in each mode. Sampled peak is unchanged;
neither peak-memory reduction nor a decode speedup is claimed.

The TTFT-specific gate requires greater than 1% matched improvement,
all three requests faster, and aggregate matched decode ratio at
least 0.98 in both validation modes. Repeat controls are reused and
disclosed. Longer contexts require repeated support at the preceding
context and v4 GPU-greedy support at their own context. Standalone v4
was held at 32K, so this existing controller cannot admit 32K against
an unsupported GPU-greedy control. A separate combined comparison was
run with CPU-greedy/slice-off control versus GPU-greedy/slice-on
candidate, keeping runtime/provider fixed and private FC2 disabled.
The old controller terminated with FileNotFoundError for the absent
standalone v4 32K repeat proof, before launching a 32K GPU job.
Its repeated 2K/8K support remains valid; this is a prerequisite block,
not a new 32K candidate performance failure.
The 100 tok/s target and independent whole-model quality qualification
remain unmet.

### Private FC2 activation cache (native exact, model repeat held)

`ORT_WEBGPU_QMOE_FC2_PRIVATE_A` is a validated, process-static,
default-off selector on the actual fused bitcast FC2 kernel.
Each active lane caches eight activation vectors in function-local
storage and reuses them for the original two output columns.
Declared workgroup storage falls from 14,080 to 1,280 bytes by removing
the 12,800-byte shared activation tile and its staging barrier.
FP16 dot/add order, the serial 32-lane fold, bias, and serial rank
mixing are unchanged. Existing conservative eligibility limits and
the nonbitcast fallback remain. Physical register residency has not
been verified.

The provider inherits the qualified UInt32 Slice composite parent,
reproduces its relink and original QMoE object byte-identically before
substitution, and has hash
`3a283ecfb892613b19a6f49d15b726e928b41027c8b3d838bd8cd1dfed9939d3`.
Flags 0/1 each pass 9,748,480 finite bitwise comparisons across 23
native fixtures; flag2 explicitly rejects. Shader dumps confirm
`QmoeFC2PrivateABitcastF16`, `cached_a`, and no shared `tile_a`.
The reference is original unfused GPU FP16 arithmetic, not an
independent whole-model CPU oracle.

A separate one-token QMoE benchmark uses 16 excluded warmups and 128
measured changing-input runs per fresh process. Before/candidate/after
unprofiled runs and separate control/candidate profiles have identical
output-byte hashes and zero measured major faults. The matched whole
operator wall-time speed ratio is only 1.003224, with 98/128 positive
runs; median wall times are 3311.629, 3252.725, and 3259.102 microseconds.
Separate named GPU Kernel events contain exactly 144 FC2 invocations
per profile. Excluding 16 warmups, FC2 medians are 21 versus 17
microseconds (means 20.898438 versus 16.859375). Those separate profiles
are not a bracketed model comparison and do not establish a decode
speedup. Frozen2K control/selected original-prefix and shader-routing
admission subsequently passed on GPU7 with sliced copy enabled on both
sides. Initial natural 2K full/retained before/candidate/after screens
pass: matched decode ratios 1.017753/1.016534 with all three requests
positive in each mode; matched TTFT ratios 1.000526/1.016295.
Full-validation candidate aggregate decode is 83.442701 tok/s,
prefill 649.986980 tok/s, TTFT P50/P95 3.150211/3.152105 seconds,
and sampled peak 68.956055 GiB. Retained aggregate decode is
83.682023 tok/s, TTFT P50/P95 3.153518/3.298332 seconds; one
retained TTFT request regresses 4.905867%, although the matched
aggregate remains within the predefined 2% gate.

The repeat fails full-validation consistency: matched decode ratio
1.004259 with only 2/3 requests positive. The third candidate request
is 78.190818 tok/s, with matched ratio 0.986200; full candidate aggregate
is 81.377175 tok/s. Retained mode passes with ratio 1.024120 and 3/3
positive, but cannot override the full-mode failure. The private cache
is held/default-off, without unchanged retry or longer-context admission.
The provisional 83.442701 tok/s initial result is not a stable deployment
gain. All measured major faults are zero; CPU-time observations alone
do not establish the cause of the inconsistent timing.

CPU builds use NUMA0 CPUs8-15 with a 32 GiB cap and no network.
Small native/operator jobs use GPU0/CPUs0-7 while separate GPU7 model
jobs use CPUs48-55. All-GPU checks and own-container interference
guards remain enabled; this overlap is disclosed rather than claimed
to be pristine isolation.

### Private FC2 rank-wave scheduling (native exact, performance held)

A distinct default-off `ORT_WEBGPU_QMOE_FC2_RANK_WAVES` experiment
uses 128 threads and three waves of four rank warps instead of 320
threads and ten simultaneous rank warps. It requires private FC2
activation caching, keeps 1,280 bytes of shared partials, and retains
the original 32-lane FP16 fold and serial rank mixing. No intermediate
wave barriers or arithmetic regrouping are introduced. This differs
from the held compact20 lane remapping.

Exact parent relink/object proof and native flags0/1 each pass
9,748,480 finite bitwise comparisons across 23 fixtures; flag2 rejects.
The provider hash is
`7bc7b3be6478e85c2e6c7b835eca8eb273bc642b581b78be016a02a9d8fde568`.
Fresh before/candidate/after GPU profiles each exclude 16 warmups and
retain 128 named kernel events, verifying the selected 128-thread
cache key. All three FC2 medians are 17 microseconds, with means
16.960938, 17.195313, and 16.914062. Matched GPU speed ratio is 0.985132,
with only 2/128 positive runs; whole-operator wall ratio is 0.678608.
The candidate is held/default-off, has no model admission, and will
not be retried unchanged.

The original standalone private-FC2 benchmark sets hierarchical gate
permission but leaves subgroup gate selection disabled; its actual
gate event is QmoeGate1Token. Its FC2 attribution remains an isolated
kernel measurement, not a full frozen-model workload profile.
A separate private-FC1 experiment explicitly enables subgroup gate
selection in operator profiles and requires 144 actual
QmoeHierarchicalGate1Token events, preventing that proxy mismatch.

### Private FC1 per-tile activations (native exact, performance held)

The distinct default-off `ORT_WEBGPU_QMOE_FC1_PRIVATE_A` caches eight
activation vectors per lane for each original 1024-element K tile.
It preserves the last 512-element K tail, original word/tile FP16
additions, serial 32-lane fold, bias, and SwiGLU expressions. It
removes the shared activation tile but retains the final shared
partial synchronization; declared workgroup storage is 512 versus
2,560 bytes. Physical register residency remains unverified.

Exact composite-parent relink/object proof passes. The native
comparator explicitly switches FC1 activation fusion off/on, using
original unfused FC1 GPU FP16 as reference, not the FC2-only
comparator (which would share the new FC1 implementation on both
sides). Flags0/1 each pass 9,748,480 finite bitwise comparisons over
23 fixtures; flag2 rejects. The provider hash is
`5eb62b4c8ae43220b7a90969cb2d917f374e5ad95a63b596fad9b7a11c641ff9`.

Bracketed GPU profiles each exclude 16 warmups and retain 128 FC1
events, with private FC2 fixed off and actual hierarchical gate
routing confirmed. FC1 medians are 22, 24, and 22 microseconds;
means are 22.039063, 24.531250, and 22.375000. Matched GPU speed ratio
is 0.905266 with 0/128 positive runs. The 1.007166 operator wall
ratio does not override this GPU regression. Held/default-off, no
model admission or unchanged retry.

### Coalesced greedy helper plus sliced copy (combined native admitted)

Runtime v6-r2 and the private-FC2 composite provider, with private FC2
explicitly disabled, now qualify coalesced minimum-length processing
and sliced copy together. Flags0/1 each pass 108 behavior checks and
14,022,456 initial/next score-element checks across F16/F32 vocabularies
32, 2049, and 248320. Both execute 160 GPU slices; selected routing
executes 108 fused helper calls, control zero. Original token IDs,
CPU penalties, API replacement, and fixed-seed sampling agree; selector2
rejects. Non-NaN score bits and NaN positions agree, while the inherited
F16-to-F32 CPU NaN-payload limitation remains explicitly unqualified.

A separate bounded frozen/natural 2K-to-8K pipeline runs on GPU1,
UUID `GPU-dbecbd7a-fcd4-7a20-89f1-a897e52339b7`, using NUMA0 CPUs16-23.
The same runtime/provider, GPU, eight-CPU inherited worker mask, and
sliced-copy configuration are used for both control and candidate.
Every observed worker mask and actual adapter identity must match.
The existing GPU7-only metadata verifier is not spoofed; the new
runner explicitly validates GPU1 identity plus the same frozen
model, execution settings, prefill flags, prefixes, and page faults.
The request function is reused without edits. Concurrent GPU7 work
is disclosed; absolute cross-GPU performance comparisons and 32K
eligibility are not claimed. Frozen2K control/selected original-prefix
and complete ordinary/fused/cached-ID/Slice routing admission passed.
The initial full-validation natural screen fails: aggregate decode
81.044613 tok/s, matched ratio 0.994124, and only 1/3 requests positive.
Retained aggregate is 82.704537 tok/s with ratio 1.003062 and 3/3
positive, which does not override the full-mode hold. The helper stays
disabled, with no unchanged repeat or 8K escalation. Native/routing
qualifications are not independent whole-model quality or a 100 tok/s
result.

### FC2 named-vector static cache (native exact, operator held)

A distinct default-off `ORT_WEBGPU_QMOE_FC2_STATIC_CACHE` requires
private FC2 caching and replaces dynamically indexed private vectors
with eight named vectors and four explicit ordered word updates.
Original FP16 accumulation order, 32-lane fold, and serial rank mix
remain unchanged. Physical register residency remains unverified.

Exact parent relink/object proof passes. Flags0/1 each pass 9,748,480
finite bitwise comparisons over 23 native fixtures against original
unfused GPU FP16; selector2 rejects. The provider hash is
`48582f790498afa1dec847249b3ca232f9d55f8cf57f6ee44e5aa78848ead6e9`.
Fresh before/candidate/after GPU profiles require actual hierarchical
gate routing, exclude 16 warmups, and retain 128 FC2 events each.
All medians are 17 microseconds; matched GPU speed ratio is 0.996166,
with only 7/128 positive runs. Whole-operator wall ratio is 1.002973,
not a model-decode result. The candidate is held/default-off, without
model admission or unchanged retry.

### Current GPU-greedy decode CPU attribution (diagnostic only)

A separate GPU1 diagnostic uses the supported ordinary GPU-greedy plus
sliced-copy configuration, with private FC2 and fused minimum-length
processing disabled. The original request implementation is unchanged;
its existing decode-step callback publishes sampling boundaries after
outputs 32 and 224 of the first measured 256-output request. The warmup
is excluded. Only the verified owned container's Python process is
sampled, using user-only cpu-clock at 199 Hz and 8 KiB DWARF stacks.
No profiler permissions, system settings, or hardware settings change.

The completed v2 capture has 239 samples strictly between the markers,
zero lost samples, all original generated prefixes matching, and sampled
peak GPU memory 68.956055 GiB. Diagnostic timings are explicitly not
headline-eligible. Symbol resolution uses the exact existing-image libc
libraries and scoped container-to-repository path mappings, not host
libc substitutions.

Inclusive CPU samples place integer environment parsing at 10.88%.
Direct parser callers include MatMulNBits Compute at 3.35%, WebGpuContext
Run at 2.51%, and QMoE Compute at 2.09%. Command append and context Run
each appear at 12.55% inclusive; context Flush is 14.64%. These overlap
and are CPU sample shares, not wall-time shares or predicted tok/s gains.
Concurrent GPU7 work is disclosed. The earlier held dense-selector cache
is not promoted or retried unchanged.

### Context selector cache (native exact, initial model screen held)

A distinct default-off `ORT_WEBGPU_STATIC_CONTEXT_FLAGS` caches only
bind-group cache capacity and dynamic-uniform selection on first use
in a process. It covers the two capacity reads in context Run and
PrepareNodeBinding and the dynamic-uniform read in Run. Other selectors
and all existing range checks, Status-versus-exception error paths,
public layouts, shader arithmetic, and dispatch thresholds are unchanged.
Mode zero retains dynamic environment reads. Mode one intentionally
requires a fresh process to change the two captured settings; unused
settings remain unparsed, and failed integer parsing retries normally.

Compiled tests exercise the actual generated helpers for dynamic reads,
frozen independent settings, concurrent reads, parse-error retry, and
invalid private-selector rejection. The original context object and
the current composite-parent relink both reproduce byte-identically.
The new provider hash is
`e6e390d4ee8ab27791fc8974dba7e06b1c67d220ca5c18496528faad4e062f4a`.
GPU0 flags0/1 each pass 9,748,480 finite bitwise comparisons over 23
fixtures against original unfused GPU FP16; selector2 rejects. Actual
hierarchical-gate routing is required and private FC2 remains disabled.

A fresh GPU1/CPUs16-23 pipeline kept runtime/provider identical between
flags, recorded settings before model load, and checked unchanged frozen
prefixes, ordinary greedy/Slice routing, affinity, and zero major faults.
Initial and repeat 2K full/retained natural brackets must pass before 8K.
Unsupported results stop without unchanged retry. There is no 32K
admission against an unsupported GPU-greedy control.

The initial 2K screen is held. Full candidate aggregate decode is
82.470701 tok/s, prefill 651.890854 tok/s, TTFT P50/P95
3.141744/3.143744 seconds, and sampled peak 68.956055 GiB.
Matched decode ratio is 1.005320, but only 2/3 requests improve:
the third ratio is 0.999991639. Retained aggregate is 83.233710 tok/s,
with matched ratio 1.017695 and 3/3 positive, which cannot override
the full-mode hold. No unchanged repeat or 8K escalation occurs.
NUMA0 CPU builds and small GPU0 native jobs overlap portions of these
GPU1 brackets on different CPU masks; pristine isolation is not claimed
and does not excuse the failed gate. No model speedup or independent
whole-model quality qualification is established.

### Fresh combined 32K greedy/Slice comparison (initial screen held)

The fresh CPU-greedy/slice-off versus GPU-greedy/slice-on comparison
keeps runtime/provider fixed, private FC2 disabled, and GPU7/CPUs48-55
placement unchanged. Frozen-prefix and routing admission passes.
Initial natural results are:

| Validation | Decode tok/s | Prefill tok/s | TTFT P50 / P95 (s) | Sampled peak GPU (GiB) |
|---|---:|---:|---:|---:|
| Full | 71.951384 | 531.489685 | 61.664798 / 61.673615 | 70.682800 |
| Retained | 47.081107 | 503.227232 | 65.926411 / 66.283300 | 70.814636 |

Full matched decode ratio is 1.203356, but matched TTFT ratio 1.025319
exceeds the 2% gate. Retained matched decode/TTFT ratios are
0.932831/1.064130, with only 1/3 decode requests improving. Both modes
fail. No repeat, 32K promotion, or 100 tok/s claim follows. These are
three measured 256-output requests plus excluded warmup, not the
earlier ten-request CPU-greedy baseline or a stable new deployment result.

Measured major faults are zero. Full decode uses roughly 2.00-2.03 user
CPU seconds and 0.21-0.24 system seconds per 255 intervals; retained uses
1.75-1.89 user and 0.48-1.10 system seconds, with more voluntary context
switches. These observations do not establish the slowdown's cause.
A separate owned-GPU7 diagnostic captures user-only CPU stacks and
read-only NVML clocks/power/temperature/throttle flags strictly during
decode. It uses one warmup and one measured request, requiring all 512
original IDs and zero major faults; its timing is not headline-eligible.
No profiler permissions or hardware settings change.

The diagnostic completes with 224 strict-window user CPU samples and
zero lost samples. Exact-image symbolization attributes 12.50% inclusive
CPU samples to integer environment parsing and 4.46% to bind-group
lookup, not percentages of end-to-end wall time. Allocation appears in
8.04% inclusive samples, but those callchains truncate before a caller;
they cannot establish that the binding key is responsible.

The measured diagnostic request takes 3.634772 decode seconds, with
1.789556 user and 0.256935 system CPU seconds. Its excluded warmup takes
6.569982 decode seconds. Thus the slow retained *measured* request is not
reproduced. All 26 in-window clock samples report SM 1980 MHz, memory
3201 MHz, temperature 39-40 C, and throttle mask1 (GPU idle), without
thermal/power-cap bits. This does not rule out different conditions in
the earlier slow requests or establish their cause.

### Inline bind-group lookup keys (native exact, initial model screen held)

The distinct default-off `ORT_WEBGPU_BIND_GROUP_INLINE_KEY` uses 129
stack words for cache probes with up to 32 bindings. Larger descriptors
retain the original heap-vector fallback. Transparent span hashing and
equality preserve the original hash algorithm, layout handle, and every
binding number, buffer handle, offset, and size. Cache misses materialize
an owned vector before insertion; no cached key refers to stack storage.
Only the context translation unit and its private cache implementation
change. Public layouts, shaders, command order, mutex ownership, and
resource destruction order are unchanged.

Cache entry limits, 1 MiB whole-buffer per-entry eligibility, and the
256 MiB aggregate referenced-buffer bound are unchanged. Compiled actual
helper tests verify zero probe allocations through 32 bindings, heap
fallback at 33, original hashes, full-field distinctions, allocation-free
heterogeneous lookup, and owned-key lifetime after scratch reuse.

V1 passes valid-input tests but is not native-admitted: selector2 logs
the explicit rejection and then exits139 during teardown. V2 validates
the private selector at context Run entry before command work; a fresh
smoke test and complete native invalid-selector phase both reject with
clean exit1. V1 evidence and source remain unchanged.

V2 reproduces the original context object and composite-parent relink
byte-identically. Provider hash:
`7edf9533230fc383400eb398f0358df49321a6aea8c17b677a85b09c30f4e633`.
Each of flag0/capacity8192, flag1/capacity8192, and flag1/capacity1
passes 9,748,480 finite bitwise comparisons over 23 fixtures against
original unfused GPU FP16. Both normal-capacity cases record 2,256
actual hits and peak 886,784 referenced bytes. The one-entry selected
case records 2,292 evictions and peaks at 851,968 referenced bytes.
Actual hierarchical gate routing is required; private FC2 stays off.

The fresh GPU1 model pipeline changes only this selector between
matched control/candidate configurations, not the held context cache.
Frozen admission precedes full/retained natural brackets and repeat
gates. Unsupported results stop without unchanged retry or longer-context
escalation. Native allocation/correctness evidence is not a model TPS
gain or independent whole-model quality qualification.

The initial 2K full candidate measures 81.984477 decode tok/s,
651.023525 prefill tok/s, 3.147207/3.153003-second TTFT P50/P95,
and 68.956055 GiB sampled peak. Matched decode ratio 0.996872 and
only 1/3 improving requests fail the gate. Retained measures 83.262171
tok/s and matched ratio 1.001103 with 3/3 positive, which does not
override full-mode failure. The pipeline holds without repeat or 8K.
The selector remains default-off; no supported model improvement follows.

A separate bounded GPU7 diagnostic profiles only the final four
decode tokens of one excluded warmup and one measured 2K/256-output
request using the supported ordinary GPU-greedy/Slice configuration,
private FC2 off, and full validation. It requires all 512 original IDs,
zero major faults, and native `Kernel` events with pid/tid -1. GPU event
durations guide the next distinct shader experiment; profiling
changes command handling and is not natural throughput or necessarily
inclusive of greedy-helper kernels. No new model copies are created.

The diagnostic completes: all 512 IDs match, faults are zero, sampled
peak is 68.952148 GiB, and all eight profiles contain 875 native GPU
events each. Four excluded-warmup profiles are separate from four
measured-tail profiles. The measured instrumented averages reconfirm
the previously identified HC/dense shader costs, not a new TPS gain:

| Shader family | Instrumented GPU us/token | Share of instrumented GPU sum |
|---|---:|---:|
| HC post-normalization/projection | 2720.75 | 17.31% |
| Dense columns8 double-buffer | 2594.50 | 16.51% |
| HC up/pre-mix | 1162.25 | 7.40% |
| QMoE FC1 | 1132.25 | 7.21% |
| QMoE FC2 | 1099.50 | 7.00% |
| Sparse-attention query cache | 1023.75 | 6.51% |

### Reduced-thread HC normalization/projection (native exact, operator screen held)

The distinct default-off `ORT_WEBGPU_HC_POST_NORM_SMALL_CTA` uses 128
physical threads and four projection columns instead of 256 threads and
eight columns: 81 CTAs instead of 41. This is not the earlier four-column
experiment, which kept 256 physical threads and left half idle during
projection. Each new physical thread computes two original logical RMS
lanes; each lane retains its original h-stride256 serial sum and the
original 256-value ordered tree. Elementwise post/normalization writes
use h-stride128 with complete, disjoint ownership. Dot grouping, ordered
FP16 tile additions, serial 32-lane fold, all four outputs, original
program and fallback eligibility remain unchanged. Shared storage is
27,392 bytes. Extra duplicated normalization work can offset occupancy
benefits; no speedup is assumed.

V1 correctly stops before GPU admission when its exact unchanged-object
proof detects the generic builder's stale replay-header precedence.
Its source and failure evidence remain sealed. V2 uses the original
compile/include order, reproduces the original HC object and complete
parent relink byte-identically, and isolates the variant-key extension
in a uniquely named private replay type derived from the exact v8
header. A compile-time size assertion and restoration tests guard that
private type; no shared header or public layout changes.

V2 provider hash:
`31c3c56405d3bed0298792c7ac605c4eb03afbf5a104ca6ae0a7ed5286623596`.
Native qualification passes 607,143,936 finite bitwise comparisons
across 36 fixtures, all four outputs, changing inputs/CPU splits,
pending1/64, keyed replay/LRU eviction, 24 fallback fixtures, and
explicit invalid-selector rejection. Six focused mapping/order/private
replay tests pass. A separate four-output benchmark builds with strict
warnings and two command/event-filter tests pass.

Fresh GPU0/CPUs0-7 operator before/candidate/after brackets use 16
excluded warmups and 128 measured calls, followed by matched actual
native GPU timestamp brackets. All output bytes, finite results, zero
major faults, program names, event counts and 128/256-thread keys are
checked. Operator admission requires more than 1% matched GPU gain and
at least 120/128 improving calls; these are not model decode ratios.
Model timing remains unqualified and the selector remains off.

The completed operator screen rejects the candidate. Original-before,
candidate, and original-after GPU medians are 22.5/28.0/22.0 us.
Matched GPU speed ratio is 0.801807 with 0/128 improving calls;
matched operator wall-speed ratio is 0.976833. All four output-byte
hashes and zero-fault gates pass. Reducing threads while duplicating
normalization does not improve this workload. No model admission,
unchanged retry, or promotion follows.

The separate user-CPU captures still truncate every malloc-bearing
callchain before its caller. They do not prove which allocation site
dominates; the recorded perf files also do not supply build IDs.
A fresh owned GPU1 diagnostic changes only its evidence prefix and
DWARF stack payload from 8 KiB to 32 KiB, retaining the existing
request function, strict decode markers, frozen prefixes, CPU mask,
user-only permissions and resource guard. This tests whether deeper
stacks resolve the caller, not whether an unchanged held optimization
can pass another timing screen. The capture is rejected: perf reports
9.49% lost samples, failing the unchanged zero-loss gate. Structural
inspection still finds truncated malloc callers despite the larger
stack payload; no timing or sampled CPU shares from this capture are
qualified. Cleanup explicitly stopped only the owned container.

CPU-only investigation finds the immediate resolution failure: the
offline symbol root omitted `/opt/conda/lib/libstdc++.so.6.0.34`.
Verbose libdw output reaches malloc's return address but cannot open
that mapped Conda library. A fresh symbol root adds only the small
libstdc++/libgcc files copied and hash-checked from the pinned approved
image; the old root and raw captures remain unchanged. No host-library
substitution, installation, permission change, or new model run occurs.

Fresh strict-window decoding of the original zero-loss captures recovers
all 12 allocation callers in 239 user-CPU samples at 2K and all 18 in
224 samples at 32K. Four parser tests and raw-capture/library hashes,
marker bounds, sample counts and regenerated-callchain checks pass.
Five malloc-bearing stacks in each capture have a Dawn ancestor
(2.09%/2.23% of sampled user CPU), including synchronization-scope maps
and output-vector creation. Other sites include Tensor/view creation
and host vectors. These are sampled user-CPU fractions, not wall-time
shares or predicted TPS gains. Deep ancestors can still exceed the
original 8-KiB snapshot, and mapped-DSO build IDs are not independently
verified; the capture records a vDSO build ID but not these libraries.
The lossy 32-KiB capture remains rejected.

### Dawn compute synchronization-scope scratch reuse (isolated experiment)

The recovered callers motivate a distinct allocation experiment, not
another unchanged bind-group or environment-cache retry. Ordinary direct
dispatch currently constructs and destroys a scope tracker every time,
although acquiring its result already clears all buffer, texture and
external-texture maps and the resource-table pointer. The private candidate
reuses that cleared tracker per thread. Explicit indirect scopes remain
owned by value, and a reentrancy guard uses the original owned path for
nested calls. The synchronization/validation body and resource-reference
ownership are unchanged; no tracker fields or object layouts change.
The existing Abseil clear implementation releases backing arrays with
capacity at least 128 instead of retaining arbitrary large maps.

This is compile-time gated by `ORT_WEBGPU_DAWN_REUSE_COMPUTE_SCOPE`,
default zero, in a private source/header overlay and private Dawn archive.
Only the isolated candidate enables it; the supported carrier and shared
Dawn/FP8 sources and archives remain unchanged. The build reproduces
the original Dawn compute-pass object and original combined-provider
relink byte-identically before replacing that one archive member.
Provider SHA256:
`af5985a0d6182c4d6155457ad6ef750a2fa185e357cdb1b685e1d4538d445525`.

Nine focused tests pass, including compiling the actual new C++ wrapper
in both selector modes with strict warnings and checking reentrancy,
indirect input preservation and eight-thread isolation. The separate
native comparators also compile with `-Wall -Wextra -Werror`.
Fresh original/candidate native runs must match whole-output fingerprints
across the 36 existing arithmetic/replay/fallback fixtures. Only then may
a CPU-sensitive 128-dispatch batch screen run, using 16 excluded warmups
and 128 measured calls in original/candidate/original brackets. Admission
requires more than 1% matched wall-speed gain and 120/128 improving calls;
these are operator criteria, not GPU-kernel or model-decode gains.
The completed original and candidate native runs each pass 607,143,936
finite bitwise comparisons, changing inputs/CPU splits, replay eviction,
pending1/64 and 24 fallback fixtures. All 36 whole-output fingerprints
also agree across independently run original/candidate providers.

The CPU-sensitive batch screen is held. Original-before, candidate and
original-after median wall times are 5.620946/5.558924/5.592654 ms.
The matched wall-speed ratio is 1.007605, with 93/128 improving calls;
both the 1% and 120/128 admission gates fail. All output-byte hashes
agree, measured major faults are zero and sampled peak GPU memory is
0.649109 GiB in each bracket. This small operator observation is not a
model-decode improvement. No model admission, promotion, or unchanged
retry follows; the supported carrier remains unchanged.

### Omitting unused fused QMoE scratch (isolated experiment)

A separate source/provenance check finds that the supported QMoE path
allocates FC1-output and FC2-output scratch even when their corresponding
fused programs never consume those tensors. The exact allocator sources
match the carrier manifest and encode zero clears when cached buffers
are allocated. Removing genuinely unused allocations can therefore
avoid both host allocation work and clear/pass boundaries without
weakening initialization of any live tensor.

The default-off `ORT_WEBGPU_QMOE_LAZY_UNUSED_SCRATCH` selector is frozen
at the first indirect-path execution; changing it requires a fresh
process. It omits only these two temporaries when their own fusion
is selected, for a single token and no QMoE command replay workspace.
Unfused paths still allocate their required tensors. Prefill and command
replay preserve their original allocation order and workspace slots.
All shader text, arithmetic, initialization of live buffers, object
layouts and the supported carrier are unchanged.

The first builder stopped before creating an output directory or GPU
admission because the carrier records one `compile_command`, not a
`compile_commands` list. Its failure evidence remains untouched. Fresh
v2 uses the actual manifest schema and reproduces the original QMoE
object and combined-provider relink byte-identically. Provider SHA256:
`0ee4ea98e85bd788596d8745e46dd57146a58956efdf78df61e16097078a179a`.
Six focused tests pass, including compiling the actual transformed C++
allocation snippets across both selectors, both fusion decisions,
single/multiple tokens and replay/no-replay combinations. The
whole-output fingerprint comparator builds with strict warnings.

Fresh independent original/default-off/selected and replay-fallback
native phases each pass 9,748,480 finite bitwise comparisons over 23
fixtures. Independent whole-output fingerprints agree, and invalid
selector 2 is explicitly rejected. Command replay and private FC2
remain off in operator timing. The matched wall ratio is 0.979067,
with only 32/128 improving calls; before/candidate/after medians are
3.349550/3.378568/3.324688 ms. Output hashes agree, no measured major
faults occur, and sampled peak GPU usage is approximately 1.968 GiB.
Both operator gates fail: held/default-off, no model admission and no
unchanged retry. These operator ratios are not model decode ratios.

The inherited operator template uses `QmoeGate1Token`, whereas the
supported model uses `QmoeHierarchicalGate1Token`. An existing trace
confirms both fused projection programs were exercised, but the
operator configuration is not the deployed gate configuration.
Subsequent distinct scheduling experiments must explicitly select
and verify the deployed hierarchical gate, rather than silently
inherit the older template.

### Pre-dispatch QMoE scratch scheduling (isolated experiment)

A distinct default-off `ORT_WEBGPU_QMOE_EARLY_SCRATCH` candidate starts
from the supported original carrier, not the held lazy-scratch provider.
For a single token, both fused projections and no replay workspace,
it acquires scratch slots 2/3/4 before queuing the gate. All five
allocations and their zero initialization remain intact; the existing
temporary helper reuses those exact tensors later. This tests whether
moving clear/pass boundaries ahead of dispatch helps, rather than
omitting unused tensors. Default-off, prefill, replay and unfused paths
preserve their original allocation order. All shader text and arithmetic
are unchanged. The selector is process-static and rejects values other
than 0/1.

The original QMoE object and full provider relink reproduce
byte-identically. Candidate provider SHA256:
`5b7d1e2d522da6ecef11bdc5255efb0f01b64db5add78f81b569b48b957f0af1`.
Five focused tests pass, including compiling the actual allocation
snippets across all selector/token/replay/fusion combinations, source
restoration, safe launches and strict actual-GPU routing validation.

The new operator fixture uses 32 independent QMoE nodes sharing the
original weights, inputs and routes, followed by Concat. It amortizes
per-Run readback overhead and fingerprints every output byte, requires
finite values and verifies all 32 independent outputs agree exactly.
It is a CPU-sensitive microbenchmark, not model latency: shared weights
and routes have different cache behavior from 32 distinct model layers.
No weight bytes are copied. The benchmark compiles with strict warnings. Both
subgroup and hierarchical gate selectors are explicit, and separate
diagnostic traces must show all 32 hierarchical gates and both fused
projections per call with their deployed workgroups before unprofiled
timing. Before/after controls use the original provider.

Original/default-off/selected and replay-fallback phases each pass
9,748,480 finite bitwise comparisons over 23 fixtures, with independent
whole-output fingerprints equal. Invalid selector 2 is rejected.
The first operator launch stops before timing because ORT correctly
rejects an external-weight symlink escaping the model directory.
Failure evidence remains unchanged. A fresh v2 fixture lives beside
the existing external data, needs no new weight alias or copied bytes,
and reuses completed native qualification without repeating it.
Six focused tests pass.

Both original and selected diagnostic traces contain 4,608 executions
of each deployed hierarchical gate/FC1/FC2 program, with verified
workgroups, and 432 Concat executions. All batch outputs are finite,
agree within each batch, and have identical original/candidate hashes.
The unprofiled original/candidate/original screen measures a matched
wall ratio of 1.014105, but only 115/128 improving calls, below the
120/128 consistency requirement. Median wall times are
5.018439/4.919490/4.961886 ms; sampled peak GPU usage is 1.975830 GiB
in all brackets, with no measured major faults. Held/default-off:
no model admission or unchanged retry. The small operator gain is
not a model decode gain.

### Matched 2K/32K native GPU-tail diagnostics

A fresh GPU7/CPUs48-55 diagnostic retains the original frozen model,
runtime, provider and load flags, with one excluded warmup and one
measured 32K request. All 512 generated IDs match their frozen prefixes;
major faults remain zero. Only the last four decode tokens are
instrumented, after output 252, with full validation. Sampled peak GPU
usage is 70.675049 GiB. Two launch/event-filter tests pass, and the
bounded container releases GPU7 after completion.

A separate two-test parser revalidates both outcomes and raw profile
hashes, requires exactly four measured token profiles per context,
checks the complete 48-layer hierarchical/FC1/FC2 routes and their
512/128/320-thread keys, and confirms identical model/runtime/provider/
load flags. The 2K/32K traces have 875/887 actual native GPU events per
token and summed instrumented kernel durations of 15,714.5/16,322.5 us.
These sums are not natural latency, throughput or predicted gains.

| Program family | 2K instrumented us/token | 32K instrumented us/token |
|---|---:|---:|
| HCPostNormProjectionF16 | 2720.75 | 2734.00 |
| DenseColumns8DoubleBuffer | 2594.50 | 2602.25 |
| HCUpPreMixPacked2F16 | 1162.25 | 1173.50 |
| QmoeFC1SwiGLUBitcastF16 | 1132.25 | 1143.75 |
| QmoeFC2FinalMixBitcastF16 | 1099.50 | 1084.00 |
| DynamicSparseAttentionQueryCache | 1023.75 | 1058.00 |
| SparseAttentionIndexerQsaParallelRank | 133.50 | 571.50 |

The largest positive context deltas include rank (+438 us), the 32K
wide score path (393 us, replacing another score path), concatenation
(+125.75 us), and parallel map count/scatter (117.5/105.25 us). Program
replacement and instrumentation prevent interpreting positive deltas
alone as the total context cost. The traces do not establish the cause
of the earlier unstable retained natural 32K result. No stable 32K
promotion or new headline decode/TTFT gain is claimed.

### HC owned-pair staging with 34-half block pitch (isolated experiment)

The context traces motivate a distinct default-off
`ORT_WEBGPU_HC_OWNED_PAIR34` candidate. It combines single-owner
`vec2<f16>` stores with a 34-half/17-word block pitch, unlike the held
scalar pad34 or owned-vec4/pad36 layouts. Projection reconstructs its
original vec4 operands from pairs. The 256-stream FP32 RMS tree,
ordered FP16 dots/tile updates/serial fold, 256 threads, eight columns,
41 groups and all four outputs remain unchanged. Shared storage is
26,368 bytes. A different shared access pattern is a hypothesis, not a
measured bank-conflict or model-speed claim.

Five focused tests pass: actual C++ shader rendering for shared/scale
variants, unique pair ownership and complete padding/index coverage,
unchanged default program/fallbacks/geometry/replay, safe original
provider controls, and strict actual-GPU program/workgroup filtering.
The exact original HC object rebuild and supported-provider relink
reproduce byte-identically. The new provider starts from the supported
original carrier, not a held provider; SHA256:
`843f11d89323e74f5aa67be6042f02f83d326f5a88b71043d9cfd9949c6d677a`.
No model weights are copied and no shared Dawn/FP8 source is changed.

Independent original/default-off/selected native parity, invalid
selector rejection and original/candidate/original operator brackets
complete. Each non-invalid phase passes 607,143,936 finite bitwise
comparisons over 36 fixtures, all four independent output fingerprints
agree, and replay/binding/LRU invariants pass. Invalid selector 2 is
explicitly rejected. Selected native peak usage is 0.658936 GiB.

The actual-GPU operator medians are 23.0/22.0/23.0 us for original/
selected/original, with a matched GPU ratio of 1.033903. Only 102/128
GPU calls improve, below the unchanged 120/128 consistency gate.
The matched wall ratio is 0.978319, with wall medians
7.832301/7.983771/7.813883 ms. All four finite output-byte hashes,
zero measured major faults, program counts and 256-thread keys agree.
The candidate is held/default-off: no model admission or unchanged
retry. These isolated operator results are not model decode gains.

### One-time HC gate staging (isolated experiment)

A distinct default-off `ORT_WEBGPU_HC_GATE_STAGING` candidate starts
from the supported original HC program, not a held layout. Invocations
0-3 compute the four original FP16 gate sigmoids; the shared variant
uses invocation 4 for its shared sigmoid. FP32 workgroup scalars hold
the exact FP16 values, which are converted back to FP16 after one new
barrier. All rendered shader text after the gate prologue, including
post-mix, the original RMS tree, normalization and ordered projection,
remains identical. Geometry stays 256 threads/eight columns/41 groups.
Declared shared storage is 27,668 bytes. The process-static selector
requires a fresh process to change and explicitly rejects invalid
values.

Three focused tests pass, including actual C++ rendering across shared/
scale variants, exact unchanged shader suffix/default/fallback checks,
and fresh generated builder/controller provenance. Generated launch
and strict actual-GPU route smoke checks also pass. The original HC
object and supported original provider reproduce byte-identically.
Candidate SHA256:
`2a70fe2c6e8ebef2a02adf09cda9c537bbf21339d07c08f55a379e0debedc359`.
No model weights are copied. Native arithmetic and matched actual-GPU
operator qualification complete. Original/default-off/selected each
pass 607,143,936 finite bitwise comparisons over 36 fixtures, with
independent four-output fingerprints equal and replay/binding/LRU
invariants intact. Invalid selector 2 is rejected.

Original/candidate/original actual-GPU medians are 22.0/23.0/23.0 us.
The matched GPU ratio is 0.982792, with only 14/128 improving calls;
the matched wall ratio is 1.002494, with wall medians
7.902212/7.858021/7.864611 ms. All finite output hashes, zero measured
major faults, program counts and 256-thread keys pass, but the GPU
performance gates fail. Held/default-off, without model admission or
unchanged retry. No model gain is claimed.

### Further offline CPU stack repair and libc attribution

The zero-loss 2K/32K captures originally contain 36/37 header-only
samples. Verbose offline unwinding identifies the missing runtime-injected
`libnvidia-eglcore.so.580.159.04`, not a missing host libc.
A guarded helper, without GPU computation, records and copies the
35,076,592-byte driver DSO into a fresh private symbol root.
Its observed SHA-256 is
`3c9340e2af8c764d8bd024cc895d41b5bb244103d2879501081f7677bfe3d557`
and current build ID is `701a056de23ed91e89c48916a3f5847dd8da607e`.
This recovers 36 stacks at each context without changing the strict
239/224 sample denominators. Driver-bearing stacks primarily lead to
Dawn Vulkan compute-pass recording, queue submission and barriers.

The final empty 32K stack instead lacks `/opt/conda/bin/python3.11`.
A CPU-only pinned-image helper adds that executable to a fresh root,
hardlinking inherited symbol files instead of duplicating their data.
The Conda executable has no ELF build ID: the first probe's contrary
assumption fails explicitly and remains preserved. The fresh corrected
probe records the absence and verifies copied bytes by SHA-256.
Both original strict captures now have zero header-only stacks.
No new model load, installs, profiler-permission changes or GPU
computation are involved.

The 49/40 originally decoded unknown libc leaves are a separate issue.
Exact-image IFUNC resolution and recorded executable mappings assign
only instruction addresses inside unique FDE ranges beginning exactly
at a selected implementation entry. This identifies 30/24 leaves;
16/17 belong to the selected `strncmp` implementation. Shared
`memcpy`/`memmove` and `bcmp`/`memcmp` implementation names remain aliases,
not claims about the public API invoked. The remaining 19/16 addresses
remain unnamed; 18/16 have observed allocator ancestors.
Five parser/FDE tests pass.

Observed `getenv` stacks total 21/239 samples at 2K and 28/224 at
32K. Largest first non-parser callers include MatMulNBits (7/9),
context Run (5/5), and QMoE ComputeInternal (3/6).
These are overlapping sampled user-CPU observations, not wall-time
shares or predicted TPS gains. Current image/driver hashes and observed
dispatch choices do not establish historical mapped-DSO build IDs.
The original 8-KiB stack depth limit and exclusion of the lossy 32-KiB
capture remain unchanged.

### QMoE invocation-local selector reuse (native exact, operator held)

A distinct default-off `ORT_WEBGPU_QMOE_INVOCATION_FLAGS` passes the
nine already validated ComputeInternal selector values into
ComputeWithWorkspace. Unlike the earlier dense/context caches and
QMoE fast parser, there is no persistent value cache: selectors refresh
on every invocation. Original entry parse order, range-error behavior,
replay keys, object fields, shaders and arithmetic are unchanged.
Only a private nested value type and private helper signature are added.

Three compiled/source/launch tests pass, as do the exact original
QMoE-object and current-provider relink proofs. Provider SHA-256 is
`d14c45fe6992d3f54560e9edf30d391a6cf20292b0806ba322f359ab2970bcfd`.
Original/default-off/selected/replay each pass 9,748,480 finite bitwise
comparisons over 23 changing-input/router fixtures, with independent
fingerprints equal; invalid selector 2 rejects.

The shared-weight 32-node operator verifies the deployed hierarchical
gate, FC1 and FC2 programs and 512/128/320-thread keys. Matched wall
ratio is 1.021389, but only 117/128 calls improve, below the required
120. Original/candidate/original medians are
5.074277/4.899572/4.940109 ms. Finite output hashes agree, measured
major faults are zero and sampled peak is 1.975830 GiB.
Held/default-off, without model admission or unchanged retry.

This operator disables QMoE command replay; the supported natural
model enables it. The original strict CPU windows have 14/8
QMoE ComputeInternal-bearing stacks but no ComputeWithWorkspace-bearing
stacks. Consequently this cold/capture-path reduction would not itself
establish a steady-state model gain even if its operator gate passed.

### Shared literal-selector snapshots (native exact, operator held)

The distinct default-off `ORT_WEBGPU_LITERAL_SNAPSHOT` targets actual
QMoE replay-entry, dense ComputeInternal and context Run selectors.
Only literal `ORT_WEBGPU_*` integer calls in those three translation
units are rewritten: 29/26/6 call sites. Shader strings, comments,
headers, variable-name reads, noninteger parsing, public layouts and
kernel fields remain unchanged.

A compile-time flag-name helper shares one lazily parsed optional
integer per process and flag name. There is no map, per-kernel
allocation or persistent mutex. Unset values stay optional so caller
defaults can still differ; parse exceptions leave initialization
retryable. Mode zero keeps dynamic reads. Mode one intentionally
requires a fresh process for changed captured selectors, including
settings previously unset. It must not be validated using a benchmark
that silently mutates those flags between reference/candidate runs.

Five tests cover scoped source rewriting, preservation of shader
strings/comments and digit separators, actual helper behavior across
translation units and threads, distinct caller defaults, parse-error
retry, invalid private-selector rejection, fixed-process native
golden comparison, startup launch wiring and replay-hit admission.
The three exact original objects and current carrier relink reproduce
byte-identically. Candidate provider SHA-256 is
`6759761dab76901abc934a95227d78930b78023fea71ee8d226739c5ea2bd61b`.

A fresh native comparator never changes flags inside its process.
The original unfused FC2 process writes an exclusive, finite FP16
golden; separate original-fused/default-off/selected/replay processes
must compare every byte and rank/shape/count against it. Fresh
processes test invalid private and legacy selectors. The initial
launcher preflight explicitly fails on the SDK runtime's absolute
container-path symlink; a fresh controller maps that recorded target
before verifying its unchanged library hash. Failed evidence is
preserved.

The following operator gate enables QMoE command replay, unlike the
held invocation-local experiment. It requires the deployed GPU program
counts/workgroups and actual high-hit, 32-node replay counters before
matched original/candidate/original wall timing can admit model work.
Operator ratios are not natural model gains.

The completed original-unfused reference checks 9,748,480 finite
elements. Fresh original-fused/default-off/selected/replay processes
each compare 9,748,480 elements bitwise against that independent
golden over 23 changing-input/router fixtures; all rank/shape/count
and output fingerprints agree. Fresh private selector 2 and legacy
FC2 selector 2 both reject explicitly.

All three replay-enabled operator brackets have exactly 144 requests,
143 hits and one capture in each of 32 distinct kernels. Original and
selected GPU diagnostics verify identical hierarchical-gate/FC1/FC2
counts and 512/128/320-thread keys. Matched wall ratio is 1.007956,
with only 97/128 improving calls; both the greater-than-1% gain and
120/128 consistency gates fail. Original/candidate/original medians
are 4.166361/4.136280/4.166286 ms, with identical finite hashes,
zero measured major faults and sampled peaks at most 1.975830 GiB.
Held/default-off, without model admission, unchanged retry or
promotion of earlier held dense/context caches.

### HC tile-parallel projection (four workers native exact, operator held)

The distinct default-off `ORT_WEBGPU_HC_TILE_PARALLEL4` splits the ten
independent projection tile sums across four workers per original logical
lane. It uses 1024 physical threads, retaining eight columns/41 groups.
Each original four-component FP16 dot sequence stays unchanged. Full-word
FP32 scratch stores each FP16 tile result; the first 256 threads convert
back to FP16 and fold tiles in original order, followed by the unchanged
serial 32-lane fold. Every scratch word has one writer. All finite FP16
values survive the storage conversion exactly; actual shader arithmetic
still requires native qualification. Existing post/RMS/normalization
writers are guarded to the original 256 threads, with uniform barriers.
Shared storage is 37,888 bytes. NVIDIA device admission checks both
1024-thread limits and shared capacity; unsupported devices stay original.
Original pad36 staging, fallback shaders and public layouts are unchanged.
No previously held layout/cache is stacked.

Six focused actual-renderer/order/ownership/finite-conversion/hardware/
launch-filter tests pass. The original HC object and supported current
carrier relink reproduce byte-identically. Provider SHA-256:
`8518d43961fdf2cc3e4509f56df966d6797c94c4101c4f1230309074224e7675`.
Fresh original/default-off/selected processes each pass 607,143,936 finite
bitwise comparisons over 36 changing-input fixtures, all four outputs,
identical independent fingerprints and replay/LRU invariants. Selector 2
rejects explicitly.

GPU0 operator brackets verify original 256-thread and selected 1024-thread
programs, 16 excluded warmups/128 measured calls, identical output hashes
and zero measured major faults. Original/candidate/original median native
GPU times are 22/22/22 us. Matched GPU ratio is 1.015506559, but only
66/128 calls improve, failing the required 120/128 consistency gate.
Matched wall ratio is 1.004922469, with medians
7.872632/7.877728/7.957145 ms. Sampled peaks are at most 0.649170 GiB.
Held/default-off, without model admission or an unchanged retry. These
operator ratios do not establish natural decode gains.

A separate two-worker `ORT_WEBGPU_HC_TILE_PARALLEL2` experiment reduces
physical threads to 512 and assigns exactly five tiles to each worker.
It targets idle normalization warps and four-worker scheduling overhead
after the near-flat result, not a claimed speedup. Full-word scratch,
shared capacity, original arithmetic order and all admission gates remain;
both thread capability checks become 512. It requires fresh original,
default-off and selected native/operator qualification before model work.

The two-worker build and nine combined focused tests pass. Original HC
object/current carrier relink proofs reproduce byte-identically; provider
SHA-256 is
`b9d5cc37b9c35568c91cfc55b94015f5e7eb00e262e4124fd102ac22970de167`.
Fresh original/default-off/selected each pass 607,143,936 finite bitwise
comparisons/36 fixtures, all four fingerprints and replay invariants.
Invalid selector 2 rejects. Actual original/selected thread keys are
256/512. GPU medians are 23/22/23 us and matched GPU ratio is 1.042250121,
but 109/128 improving calls fail the unchanged 120/128 consistency gate.
Matched wall ratio is 1.002730268, with medians
7.867800/7.886672/7.963141 ms; finite output hashes agree, measured major
faults are zero and sampled peak is at most 0.649170 GiB. Held/default-off,
without model admission or an unchanged retry. No natural TPS gain follows.

### Three-worker dense tiles with packed scratch (broad route held)

The distinct default-off `ORT_WEBGPU_DENSE_TILE_PARALLEL3` targets the
supported Columns8 double-buffer shader at single-row K2560/6144.
Three workers per original logical lane use 384 physical threads and
eight columns/group. Three independent double-buffered activation tiles
use 12,288 bytes; 768 full-word UINT32 scratch slots hold pairs of original
FP16 tile sums in another 3072 bytes. Each word has one writer. The first
128 threads fold each stream's tiles in original FP16 order and reuse
their own first-tile word; no other fold reads that word. A uniform barrier
precedes the original serial32 output fold. Invalid last-tile blocks
retain the original skipped-update behavior, including signed zeros.
Both original four-component dot sequences are unchanged. Shared storage
is 15,360 bytes, with explicit NVIDIA/thread/shared-limit admission.
K640, multi-row/FP32 cases, original default shaders and public
layouts remain unchanged. The private selected replay variant is 8;
original Columns8 keeps variant 7. No held arithmetic/cache is stacked.

Seven actual-C++-renderer, ownership, packed-reuse/FP16-order, fallback,
native-scope, CLI and GPU-event/gate tests pass. Original dense object and
supported current-carrier relink reproduce byte-identically. The strict
warning-clean native comparator builds. Provider SHA-256:
`81fcbcf6882f5156dad1e97ca3ee8903aeac3fe99e398879d00277607d0e8a92`.

The faster operator-first screen uses seven existing actual-weight
fixtures, without copying weights. Separate original-capture, candidate
default-off, selected and original-after processes each compare 6,856,704
finite elements exactly across 36 changing inputs/four repeats. Fresh
invalid selector 2 rejects. Every bracket checks 144 actual GPU events,
excludes 16 warmups and measures 128; original/selected keys are 128/384
threads. Native source enforces finite rank/shape/type, every reference
byte, original repeatability and zero measured major faults. Peaks are at
most 0.668641 GiB. Successful screening would still require the complete
native/fallback suite before natural-model admission.

| K / N | Before / candidate / after median GPU us | Matched GPU ratio | Improving calls |
|---|---:|---:|---:|
| 640 / 2560, original fallback | 12 / 12 / 12 | 1.003992 | 23/128 |
| 2560 / 2560 | 14 / 13 / 14 | 1.080043 | 121/128 |
| 2560 / 7168 | 17 / 18 / 17 | 0.938789 | 0/128 |
| 6144 / 2560 | 17 / 16 / 17 | 1.069504 | 121/128 |
| 2560 / 16384 | 23 / 25 / 23 | 0.927697 | 0/128 |
| 2560 / 6144 | 15.5 / 16 / 15 | 0.958710 | 4/128 |
| 2560 / 10240 | 19 / 20 / 19 | 0.936006 | 0/128 |

Both N2560 shapes individually pass the unchanged greater-than-1% and
120/128 gates, but all four wider shapes regress. The broad route is
held/default-off without model work or unchanged retry. These are native
GPU operator times, not synchronous host Run/readback times or model TPS.

A separate `ORT_WEBGPU_DENSE_TILE_PARALLEL3_NARROW` candidate changes only
private selection: N must also equal 2560. The GPU shader stays identical;
wider projections retain the supported original program. This is measured
shape-specific routing, not promotion of the rejected broad route. It
requires fresh route/default/fallback proof and gated qualification.

The narrow candidate builds with exact original-object/current-carrier
proofs and nine passing tests. Provider SHA-256:
`a8256b2adefb41c7169c2a37a8f74e0090bba9dbf786e3387de618cb34fce0bc`.
Fresh original/default-off/selected/original-after each compare 6,856,704
finite elements over the same seven actual-weight fixtures; invalid 2
rejects, all actual program/thread keys pass and major faults are zero.
Peak remains 0.668641 GiB. Wider and K640 cases correctly retain original
128-thread shaders and 12/17/23/16/19-us medians. K2560/N2560 still measures
14/13/14 us, but matched ratio 1.062494057 with 106/128 improving calls
fails consistency. K6144/N2560 measures 17/16/16 us, matched ratio
1.056199881 with 122/128 improving calls. Narrow route held/default-off,
without model preparation or unchanged retry.

The only shape with two independent passing screens is K6144/N2560:
121/128 and 122/128 calls improve, with matched ratios 1.069503525 and
1.056199881. A separate `ORT_WEBGPU_DENSE_TILE_PARALLEL3_LONG_K` keeps only
that six-tile shape; the three-tile K2560 case also stays original. The GPU
program is unchanged. Prior operator proofs apply only to the identical
K6144 shader/geometry, not to a new model throughput claim. Fresh native
route/default/fallback and fixed-process original-golden proof remain
required. The historical in-process double-buffer comparator mutates
legacy flags and cannot independently qualify this process-static private
selection. No additional unchanged synthetic timing retry is planned.

Long-K provider `0fa7cf2b34fb21809df6c0e0fd2e495618db2e7d9d1d72f8ab31b071facffd71`
passes the fresh original-object/current-carrier relink proof. Its new
fixed-process comparator builds warning-clean with owning TypeInfo lifetimes,
streamed independent FP16 goldens, fixed private selection and explicit FP32
accumulation fallback. Frozen fixture interfaces are rank3 `[1,rows,K]` /
`[1,rows,N]`; the failed v1 rank2-only preflight is retained, not counted as
numerical qualification.

V2 original capture checks 116,582,784 finite elements across all 27 fixtures,
including original repeatability. Its controller then correctly refuses
admission for missing replay evidence. The sealed replay helper intentionally
bypasses profiled calls and multi-row inputs: expecting 27 real replay rows
from this profiled phase was incorrect. V3 keeps the GPU-profile proof and
adds separate unprofiled original/default-off/selected/original-after and
FP32-fallback comparisons against the same independent streamed goldens.
Every process must check all 27 fixtures; normal unprofiled processes must
prove nine single-row replay instances with actual requests/hits/binding hits,
while profiled and FP32 processes must have none. Profiling mode is explicit
and checked in every PASS row. V3 passes all 12 independent processes:
1,398,993,408 finite bitwise comparisons, nine real replay instances in each
normal unprofiled process, zero in profiled/FP32 processes, unchanged independent
output fingerprints and explicit invalid-2 rejection. Peak is 0.668701 GiB.
Native model admission is eligible; natural throughput and whole-model quality
remain unqualified. The fresh model pipeline uses the unchanged original 3a
provider for both controls, not merely candidate flag0. Late full-validation
decode profiles must show exactly 48 selected long-K dispatches and unchanged
remaining program counts before matched natural 2K full/retained timing.

Fresh 2K and 8K late decode profiles both verify exactly 48 selected long-K
programs and identical remaining program counts after mapping only that
substitution back to original Columns8. Frozen output prefixes and resource
fault checks pass. Initial and repeated 2K matched brackets both pass all
three positive decode requests in both modes:

| 2K run / mode | Decode tok/s | Prefill tok/s | TTFT P50 / P95 s | Sampled peak GiB | Matched decode ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial full validation | 83.002610 | 652.013475 | 3.140855 / 3.143274 | 68.956543 | 1.007463782 |
| Repeated full validation | 82.996081 | 652.019333 | 3.140560 / 3.144302 | 68.956543 | 1.007557970 |
| Initial retained settings | 83.900186 | 651.934084 | 3.140790 / 3.146610 | 68.956482 | 1.009272589 |
| Repeated retained settings | 84.109799 | 651.666330 | 3.144275 / 3.146155 | 68.956543 | 1.009610835 |

TTFT matched ratios are 0.999974041, 0.999988492, 0.999878547 and
1.000127485 respectively. Original 3a before/after controls are unchanged.
8K initial timing passed both modes, but repeated full validation failed
the all-three-request gate. These results do not establish 100 tok/s or
independently qualify whole-model quality:

| 8K run / mode | Decode tok/s | Prefill tok/s | TTFT P50 / P95 s | Sampled peak GiB | Matched decode ratio | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Initial full validation | 82.263101 | 603.459695 | 13.575563 / 13.584403 | 69.297241 | 1.004183938 | Pass |
| Initial retained settings | 83.078967 | 603.566915 | 13.570549 / 13.581301 | 69.297241 | 1.005796735 | Pass |
| Repeated full validation | 82.064722 | 603.336443 | 13.577983 / 13.581188 | 69.297241 | 1.001726771 | Hold |
| Repeated retained settings | 83.292626 | 603.611542 | 13.574010 / 13.577122 | 69.297241 | 1.007548053 | Pass |

The failing repeated full-validation request ratios are 1.004098975,
1.003102074 and 0.997990000. A positive average does not override the
failed consistency gate. The completed pipeline is held at 8K, with no
unchanged retry or 32K escalation. Repeated 2K support remains scoped to
that context.

A distinct default-off `ORT_WEBGPU_DENSE_TILE_PARALLEL2` now tests 256
physical threads, two tile workers, 1,024 activation vec4s and the same 768
full-word packed scratch slots: 11,264 shared bytes. K2560's fourth scheduled
tile is entirely invalid and never read by the ordered fold; its writes
remain inside scratch. K6144 has three waves and safely reuses the two-wave
activation ring. Four geometry/ownership/original-fallback/launch/threshold
tests pass, and the fresh original-object/3a provider build succeeds.
The seven actual-weight operator screen completed, but the broad route is
held/default-off, not admitted to model testing:

| K / N | Before / default-off / selected / after GPU median us | Matched ratio | Improving / 128 |
| --- | --- | ---: | ---: |
| 640 / 2560, unchanged fallback | 12 / 12 / 12 / 12 | 1.013721229 | 33 |
| 2560 / 2560 | 14 / 14 / 14 / 14 | 1.032230987 | 57 |
| 2560 / 7168 | 17 / 17 / 18 / 17 | 0.949939292 | 4 |
| 6144 / 2560 | 17 / 17 / 16 / 17 | 1.093109657 | 126 |
| 2560 / 16384 | 23 / 23 / 25 / 23 | 0.918621347 | 0 |
| 2560 / 6144 | 15 / 16 / 17 / 15 | 0.893300234 | 1 |
| 2560 / 10240 | 19 / 19 / 20 / 19 | 0.935265829 | 2 |

Only K6144/N2560 passes both the greater-than-1% and 120/128 gates.
Each original/default-off/selected/original-after phase passes 6,856,704
finite bitwise comparisons; sampled peak GPU is 0.668640 GiB. Invalid
selector 2 rejects, and actual selected/default program keys confirm
256/128 physical threads. Provider SHA256 is
`b587fc50d699b89ce5a112223ce3a9cc5f9d7a335601976e8a0b35b7bfaf7eb5`.
The sealed outcome inherited a stale `parallel3` status label; its explicit
flag, program, manifest, SHA and event geometry identify this as the
two-worker experiment. Sealed evidence is not renamed after execution.

It ran in an explicit clean gap: only the owned model controller was paused,
its already-running child was allowed to finish normally, then CPU/native work
started with all GPUs empty; the controller was resumed in a finally block
and verified running with a new guard child. No builds/native GPU tests
overlapped natural model request timing.

A separate HC-up investigation rules out the proposed pointwise four-gate
fusion: `HCUpPreMixPacked2F16` computes another K320/N10240 INT4 GEMV from
all 320 projected activations. Its current workgroup is 128 threads with
16 lanes per branch and two output columns, not a simple linear 128-lane
output map that can safely be changed to 32 threads.

The new default-off `ORT_WEBGPU_HC_UP_ZERO_TAIL` instead preserves the
existing workgroup, shared stores, barriers, ten active FP16 partials,
ordered dots, sigmoid rounding and final stream mix. Only the 22 explicitly
positive-zero additions at the end of each ordered 32-partial fold are
collapsed to one positive-zero addition. Keeping that one operation
preserves the tail's signed-zero effect rather than simply removing it.
All 63,488 finite half patterns, including both zeros and subnormals, pass
the CPU idempotence proof; eleven focused shader/ownership/fallback/ABI/
launcher/scope/threshold tests pass. Runtime bitwise proof is still required.
The selector is process-static, invalid values reject, and selected replay
uses variant 2 rather than original packed/single variants 1/0.

The exact original HC-up object and unchanged original 3a provider reproduce
before compiling the candidate; its native comparator is warning-clean.
Nine fresh diagnostics reference first/middle/last frozen-model HC-up
weights at rows 1/2/16 through one read-only relative shard alias. Original
graph, tensor metadata, byte ranges and aliases are checked; no model
weight copies or frozen-model edits are made. The initial operator screen
uses the first actual-weight single-row fixture, independent original
goldens, original/default-off/selected/original-after processes, 16 excluded
warmup pairs and 128 measured second calls with identical two-call scope
in every phase. No model gain or admission is claimed from preparation.

The v1 actual-weight launch stopped before math: ORT correctly rejected
the external-data symlink because its resolved path escaped the diagnostic
model directory. Its failed log/guard and fixture generation are retained;
this is neither a shader correctness failure nor a performance measurement.
A fresh v2 in-directory hardlink preparation then hit Linux's protection
on root-owned read-only source weights before any GPU launch. Fresh v3 uses
the existing repository `sudo -n ln` fallback only for `PermissionError`,
with explicit failure propagation and original-inode/range verification.
Path validation is not disabled, weight bytes are not copied or changed,
and shader/provider/native scope are unchanged. Four focused packaging and
fresh-name equivalence tests pass. The v3 guarded actual-weight screen is
complete: every original/default-off/selected/original-after phase passes
737,280 finite bitwise comparisons, with matching fingerprints, explicit
invalid-selector rejection, zero measured major faults and sampled peak
0.649109 GiB. GPU medians are 13/12/13/12 us; matched ratio 1.003912417 and
49/128 improving calls fail both operator gates. Zero-tail pruning remains
held/default-off, with no full-native/model admission or unchanged retry.
Provider SHA256 is
`24f4d0c18ae47cc5ff6944f116579ae6a02754c842d07b16ee156d390ff61308`.

The next distinct default-off `ORT_WEBGPU_HC_UP_PACKED80` does not inherit
zero-tail pruning: it retains every original 32-term FP16 fold. Ten lanes
per branch and two columns give 80 physical threads instead of 128. Each
active lane writes its original partial plus uniquely assigned positive-zero
padding. Padded 33-half rows avoid introducing same-bank branch-fold reads
within a 32-lane warp; all 256 read locations have one original-valued
writer, and eight unused padding locations are never read. Shared memory
is 544 bytes. Dot/sigmoid/final-mix arithmetic and uniform barriers remain
unchanged; default shaders, fallback and kernel object layout are preserved.
Four ownership/bank-layout/exact-source/fallback tests pass, and the exact
original object/provider and warning-clean native build reproduce. The fresh
screen reuses v3's validated original-inode actual-weight diagnostics and
independent original 3a controls. No model admission or gain is assumed.

Packed80's actual screen completed: all four valid phases pass 737,280
finite bitwise comparisons apiece, with invalid selector rejection, matching
fingerprints and zero measured major faults. Sampled peak GPU is 0.649109 GiB.
Before/default-off/selected/after medians are 13/12.5/12/13 us. Matched
ratio 1.033405169 is positive, but only 78/128 calls improve; the strict
consistency gate fails. Packed80 remains held/default-off, without unchanged
retry or full-native/model escalation. Provider SHA256 is
`969bbef53c09df4dfe8544ea0b956d40f1a8a349dc3cc5f1d6ad2954626a4f36`.

The distinct `ORT_WEBGPU_HC_UP_PACKED160` uses four columns / 160 threads /
640 groups, versus two columns / 80 threads / 1,280 groups. Four outputs now
occupy five fully active 32-lane warps instead of six partly occupied warps
across two Packed80 groups. This is also distinct from the older held
four-column / 256-thread route: it retains ten active lanes per branch and
the padded 33-half rows, all original 32 FP16 additions, dot order, barriers,
sigmoid and stream mixing. It uses 1,088 shared bytes, with 512 uniquely
written/read partials, 16 unused padding slots and 16 gate writers.
Three geometry/ownership/bank-map/exact-source/fallback tests pass; the
original HC-up object and unchanged 3a carrier reproduce before the new
warning-clean native build. A fresh actual-weight original-provider bracket
must pass the same independent-golden and 120/128 gates before admission.

Further HC-post research identifies the actual supported source as the
original fused eight-column / 256-thread / 41-group shader, not the earlier
two-column staged fallback. The tempting 36-to-34-half shared pitch is
already held prior art, including the later owned-pair form, and is not
recreated as a new experiment.

Packed160's actual bracket is also held: original/default-off/selected/after
each pass 737,280 finite bitwise comparisons, with matching fingerprints,
invalid-selector rejection and zero measured major faults. Peak GPU is
0.649109 GiB; medians are 13/12/12/13 us. Matched ratio 1.052357923 does
not override only 87/128 improving calls. No unchanged retry, full-native
or model admission follows. Provider SHA256 is
`07b4a39807747a18f9f5a24aac30cd405858c291aab2ec552794f174a4675754`.

A fresh `ORT_WEBGPU_DENSE_TILE_PARALLEL2_LONG_K` now limits the already
measured two-worker shader to K6144/N2560, preserving its byte-identical
GPU dot/tile/fold body and all other original fallbacks. The prior broad
two-worker long shape passed at 1.093109657 / 126 of 128 calls, unlike its
wide K2560 regressions. Five focused exact-shader/admission/replay/default/
geometry/gate tests pass; fresh original-object/3a reproduction and native
build succeed. The seven actual-weight original/default-off/selected/after
screen completed: every phase passes 6,856,704 finite bitwise comparisons,
with sampled peak 0.668640 GiB. Only the exact long shape counts toward
admission; its matched ratio 1.044352723 and 118/128 positive calls fail the
120-call consistency gate. The candidate remains held/default-off, without
full-native/model escalation or an unchanged retry. Provider SHA256 is
`63276f3da0c7b4feb2b2d184e9c93efcf5850a01d56995c6a951f33e5088c814`.
The old copied `parallel3` outcome label is corrected in this fresh
generator, not retroactively in sealed evidence.

Independent timing-resolution inspection finds all 288 selected Packed160
GPU events use integer microsecond durations. The actual frozen
`static-node-replay/build-v8/overlay` context rounds resolved Dawn
nanosecond differences with `round(delta/1000)`. This limits discrimination
for roughly 12-us kernels; it is not permission to waive a failed gate.
A qualified additive implementation adds `args.gpu_duration_ns` while preserving every
existing event field and all compute code. Its consumer requires a positive
explicit uint64 decimal and agreement with the unchanged rounded duration;
it does not silently substitute legacy durations. The exact original
context object and original 3a relink reproduce before the metadata-only
build. Two independent actual-weight phases, both using original packed2
compute, each pass 737,280 finite bitwise comparisons and emit 288 GPU
events. The precise phase has 276 genuinely fractional-microsecond events,
ranging from 10.784 to 14.048 us, with sampled peak 0.649109 GiB. Four
focused source/consumer/actual-scope tests pass. Provider SHA256 is
`02a7693a1fab6cebaf092ceb8eb4b17c589a0e84b7bb9fd99af50131243d843a`.
The separate qualification outcome, not the sealed build manifest, records
precision support. This qualifies the observation, not a held shader,
whole-model quality or decode performance; no global settings changed.

A fresh two-worker long-K experiment now uses that same qualified precise
context object for both the original 3a compute control and the unchanged
candidate compute. It preserves the old 118/128 hold and does not weaken
the >1% / 120-of-128 gates, 144 calls, 16 excluded warmups, exact fixtures,
finite golden comparisons or shader/workgroup checks. Every GPU event must
supply valid raw nanoseconds; integer legacy fallback is prohibited.
V1 stopped before GPU launch because proof outputs used different library
basenames, changing GNU ELF base-version dynamic strings. Sealed failure
evidence is retained. V2 keeps the exact original library basename in
independent proof subdirectories: precise original-control and unmodified
candidate relinks both reproduce byte-for-byte before replacing only the
candidate's context object. Six focused precision/scope/link-packaging tests
pass. The fresh v2 actual-weight bracket completed: all four valid phases
pass 6,856,704 finite bitwise comparisons, with zero measured major faults
and sampled peak 0.668640 GiB. The unchanged long-K shader passes the same
operator gate at ratio 1.052262403 and 124/128 improving calls; nonselected
shapes remain original fallback diagnostics, not candidate gains.
Provider SHA256 is
`46dc00d59f6dc497c2bf380f8e7bf3ddda089ed1f52acb86189077160ce24d3a`.
The old rounded-timing 118/128 hold remains sealed, and this fresh
observation does not retroactively promote it.

Full-native validation completed for the supported precise observation:
the fixed-process comparator changes only the private selector identity.
It retains 27 frozen fixtures, twelve original/default/selected/after,
FP32-fallback and separately unprofiled replay processes, pending1/64,
116,582,784 finite elements per phase and nine real single-row replay
instances where applicable. Original-compute controls use the qualified
02a precision context; every profiled GPU event must supply valid raw
nanoseconds, and selected workgroups must be 256 threads. The original
broad two-worker long-K operator proof and the new precise exact-shape
proof both meet unchanged gates. Nine focused tests and the warning-clean
comparator build pass. All twelve phases pass 116,582,784 finite bitwise
comparisons apiece, totaling 1,398,993,408, with explicit invalid-selector
rejection and matching independent fingerprints. Each of the four
unprofiled FP16 phases verifies nine real single-row replay instances;
FP32 fallback correctly has none. The sampled full-suite peak is
0.676453 GiB.

The fresh natural model pipeline is now active. It uses the unchanged
63276 compute library against original 3a, not the nanosecond-profile
library for headline timing. The exact unmodified-candidate relink and
metadata-only context provenance connect that library to the completed
46dc native proof. Ten scoped timing/model-wiring tests pass. Late-decode
admission must observe all 48 selected 256-thread dispatches and unchanged
normalized program counts before the usual full/retained, three-request,
256-output initial/repeat comparisons. Both modes and repeats must pass
before context support; 32K is not automatically escalated. Current model
headlines remain unchanged until this pipeline's outcomes are read.

A separate fresh Packed160 observation also relinks its unchanged compute
against the same qualified precision context and exact original 3a compute
control. Original/default-off/selected/after each pass 737,280 finite bitwise
comparisons with matching fingerprints, explicit invalid-selector rejection,
zero measured major faults and sampled peak 0.649109 GiB. GPU medians are
12.800/12.416/11.616/12.416 us. Ratio 1.066694312 does not override
117/128 improving calls: this precise observation remains held/default-off,
without unchanged retry or full-native/model escalation. The older rounded
87/128 hold is preserved. Provider SHA256 is
`201ea456150cd77d272f4b84e10ae829937eaec8aa4e1db99573f33d9f1fa02e`.

The two-worker natural model pipeline completed and is held at its initial
2K both-mode gate. Actual late-decode routing passed, but the kernel gain
did not establish a full-mode model gain:

| 2K candidate | Decode tok/s | Prefill tok/s | TTFT P50/P95 s | Sampled peak GiB | Matched decode ratio | Positive requests |
|---|---:|---:|---:|---:|---:|---:|
| Full validation | 82.382742 | 651.501743 | 3.144588 / 3.145482 | 68.956543 | 0.999084385 | 0/3 |
| Retained validation | 83.579231 | 651.479866 | 3.145359 / 3.146223 | 68.956543 | 1.003698991 | 3/3 |

Full request ratios are 0.999182731, 0.998073614 and 0.999997744;
the positive retained mean does not override the both-mode failure.
No unchanged retry, repeat or 8K escalation follows. Previously supported
headlines remain unchanged, and 100 tok/s is still unmet.

The next distinct default-off `ORT_WEBGPU_DENSE_TILE_PARALLEL6_LONG_K`
uses six 128-thread logical workers in a 768-thread group. K6144 has
192 block32 blocks / six independent tiles, so every tile now executes
in a single wave, rather than the three-worker shader's two waves.
Eight output columns and all per-block dot/scale operations, packed FP16
tile storage, original ordered tile accumulation and final ordered
32-stream fold remain unchanged. All 768 packed tile locations have one
writer. Shared memory is 27,648 bytes, and hardware eligibility explicitly
requires 768 invocations and that storage limit; only K6144/N2560 selects it.
Other original fallbacks, kernel object layouts and process-static selector
validation are retained.

Six source/ownership/geometry/launch/precision tests pass. The exact
original dense object, original 3a carrier and qualified original-compute
02a nanosecond control reproduce before the new shader is compiled with
the same precision context. The warning-clean native comparator retains
144 calls, 16 excluded warmups, 128 measured calls, independent finite
goldens, strict raw-nanosecond consumption and unchanged >1% / 120-call
admission thresholds. The fresh seven-shape actual bracket completed as
reported below; preparation itself establishes no operator/model gain.

The six-worker bracket completed and is held: original/default-off/
selected/after each pass 6,856,704 finite bitwise comparisons, with sampled
peak 0.668640 GiB. The selected long-K ratio is 0.915669771 and only
1/128 calls improves. Exact arithmetic is insufficient for admission;
no unchanged retry, full-native or model escalation follows. Provider
SHA256 is
`585654c3765bacdb1a47a7448af886dac266780430d085f8b415e7e04daba6a0`.

A distinct four-worker `ORT_WEBGPU_DENSE_TILE_PARALLEL4_LONG_K` retains
the original ordered arithmetic but uses 512 threads / 19,456 shared bytes.
Six tiles execute as four in the first wave and two in the second. The
two inactive final-wave workers explicitly skip packed tile writes, so
all 768 read locations still have exactly one valid writer and no writes
extend past the scratch array. Uniform workgroup barriers and original
FP16 dot/scale/tile-fold/stream-fold order are unchanged. Only K6144/N2560
selects it, with explicit hardware limits and original fallbacks.

Eight focused arithmetic-equivalence, tail-ownership, launch and precision
tests pass; exact original dense/3a/qualified-Ns-control relinks and the
warning-clean native build succeed. The fresh seven-shape raw-nanosecond
original/default-off/selected/after bracket completed with unchanged
144-call / 16-warmup / 128-measured and >1% / 120-positive gates.
Every valid phase passes 6,856,704 finite bitwise comparisons, with sampled
peak 0.668640 GiB. The selected long-K ratio 0.970812729 and 16/128 positive
calls fail admission; it remains held/default-off, without unchanged
retry or full-native/model escalation. Provider SHA256 is
`43d4692738f2843cde074cc1a0ca68159b675eaf73352ec27ad0dbeebf828f6a`.

A distinct CPU dispatch lead, `ORT_WEBGPU_READY_PIPELINE_STATE`, uses
pointer-only state owned by an unprofiled/non-capturing deferred-encoding
scope to omit redundant SetPipeline calls. It owns no GPU references and
changes no header or object layout. Scoped restoration handles exceptions,
foreign recordings and nested/disabled scopes; state resets when a compute
pass ends or is absent, including recycled encoder handles. Bind groups,
offsets, dispatches, timestamps and original pending64 submission remain
unchanged. Two actual-C++ helper/source tests and exact original context/
3a relink proofs pass; built provider SHA256 is
`f3875a5d76279114b16dac1e88951a14cf107356e2ac446991b7c5d67c98fd8b`.

This idea is held before GPU/model admission, not promoted from its CPU
mechanism test: an existing original-3a late 2K profile has only two
adjacent identical pipeline cache keys among 875 GPU events, one Dense
and one Add. That observed coverage does not justify spending a natural
model performance screen on this lead. The same diagnostic's GPU timestamp
sum is profiling-only, not the natural 255/decode_seconds denominator.

The next fresh observation targets the previously held two-worker HC
postnorm projection. The older 1.042250121 / 109-of-128 rounded-timing hold
remains sealed. New control and candidate use the same qualified 02a
nanosecond context: exact relinks reproduce both that original-3a control
and unchanged b9 two-worker compute before replacing only its context.
The 512-thread / 41-group shader, original FP32 RMS tree and ordered FP16
projection arithmetic are not altered or stacked with held layouts.
Six precise-consumer/geometry/native-scope/byte-proof tests and the fresh
build pass. Original/default-off/selected 36-fixture finite/replay
qualification and original/candidate/original operator brackets are active
on guarded GPU0. Every profiled event must provide valid raw nanoseconds;
the >1% / 120-of-128 gates, 16 warmups, all four outputs and zero-fault
requirements remain unchanged. No operator, quality or model gain is
claimed from preparation.
