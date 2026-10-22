# INT3 Per-Block Padding Experiment

## Scope

Review request: [PR #33394, padding performance](https://github.com/microsoft/onnxruntime/pull/33394#discussion_r4340859601).
Work is isolated on `jiafa/int3-qmoe-review-experiments-20261021`, based on
`eda735324e7075a225e7538c00b4712bcc545cd0`. No change to the portable model layout
or the original PR branch is required by this experiment.

For INT3 block sizes 32/64/128, the payload is 12/24/48 bytes per block.
All are already byte-aligned: padding only to the next byte changes nothing.
This experiment interprets additional per-block padding as rounding each block's
payload up to a 16- or 32-byte boundary. Scales remain separate and unchanged.

| Block Size | Raw Bytes | 16-Byte Aligned | Payload Increase | 32-Byte Aligned | Payload Increase |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 | 12 | 16 | 33.33% | 32 | 166.67% |
| 64 | 24 | 32 | 33.33% | 32 | 33.33% |
| 128 | 48 | 48 | 0% | 64 | 33.33% |

These percentages apply to packed expert weights, not scales or whole models.

## Method

- Hardware: NVIDIA A100-SXM4-80GB, SM80, 40 MiB L2. GPU 1 was idle before the runs.
- Toolchain: container `jiafa-dev`, CUDA compiler/runtime 13.0, driver API 13.0.
  This is independent of the earlier CUDA12.8 cached ORT model experiments.
- Synthetic deterministic INT3 codes, FP16 scales/input/output, FP32 projection
  accumulation; identical logical data for every layout. Padding is filled with
  `0xa5`, so numerical checks can detect accidental consumption.
- Qwen3-30B-A3B dimensions: FC1 gate/up `[128,1536,2048]`, FC2 down
  `[128,2048,768]`; eight selected experts per call, block32/64/128.
- Operations: packed-to-FP16 dequantization, one-token linear projection, and
  32-token linear projection. These are individual projections, not complete
  SwiGLU/QMoE layers or routing-inclusive model inference.
- Access patterns: fixed first eight experts and rotation through all 128.
  Decode/dequantization trials execute 32 calls; prefill trials execute 16 calls.
  The rotating start is `(iteration * 8) % 128`, so every full trial covers all
  experts (twice for 32 calls, once for 16 calls).
- Seven trials per layout after three warmup calls; layout order alternates
  forward/reverse between trials. CUDA events time the call batch, divided by
  call count. Median of seven trials is reported. There is no explicit L2 flush.
- Two separate full benchmark invocations, each producing 108 result rows.
  Every row passed full output equality across layouts for the first/last
  expert groups and independent scalar samples for first/last rows and tokens.
  Scalar sampling is not an exhaustive independent oracle for every element.
- Raw-to-layout conversion times are recorded separately in the JSON. They are
  not included in steady-state kernel times. Alignment=1 conversion is an
  identity-copy control, not a required production prepack cost.

The original preliminary JSON used only four prefill iterations, covering 32
experts per trial. It has been replaced by the corrected run below; no conclusion
here uses that preliminary measurement.

## Block64 Results

Rotation through all 128 experts; milliseconds per call. The padded layout
uses 32 bytes/block instead of 24. Both 16/32-byte alignment requests produce
the same physical layout for block64; the table uses the 32-byte request.

| Projection | Operation | Raw Run 1 | Padded Run 1 | Time Increase | Raw Run 2 | Padded Run 2 | Time Increase |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FC1 gate/up | Dequantize | 0.164928 | 0.172384 | 4.52% | 0.164352 | 0.171744 | 4.50% |
| FC1 gate/up | Decode projection | 0.054272 | 0.060352 | 11.20% | 0.053952 | 0.059968 | 11.15% |
| FC1 gate/up | 32-token projection | 1.442560 | 1.588864 | 10.14% | 1.442496 | 1.588736 | 10.14% |
| FC2 down | Dequantize | 0.084064 | 0.087808 | 4.45% | 0.083936 | 0.087648 | 4.42% |
| FC2 down | Decode projection | 0.029856 | 0.033216 | 11.25% | 0.029664 | 0.032928 | 11.00% |
| FC2 down | 32-token projection | 0.784576 | 0.867648 | 10.59% | 0.784448 | 0.867456 | 10.58% |

All block sizes, both access patterns, per-trial times, conversion samples,
payload and scale byte counts are retained in
[run 1](int3-qmoe-padding-experiment.json) and
[run 2](int3-qmoe-padding-experiment-run2.json), not only these selected block64 rows.

## Interpretation and Limits

For this scalar byte-load/warp-reduction implementation, per-block alignment
padding does not buy a block64 speedup: it increases payload by 33.33% and makes
the measured kernels slower in both runs. Retain compact canonical model bytes;
consider padded layouts only as provider-private candidates with separate evidence.

This is not proof that padding is always harmful. Padded loads also execute
different address arithmetic; the experiment measures that combined layout and
implementation cost, not a pure bandwidth effect. It does not implement vectorized
loads, Tensor Core GEMM, shared-memory tiles, interleaved scale headers or a
2+1-plane prepack. A kernel designed around aligned vector loads could change the
trade-off. No profiler counters establish the bottleneck here.

No whole-model TTFT/TPS, BF16 qualification, allocator peak, sanitizer pass,
or production dispatch claim follows from these measurements. Fixed-expert tests
can be cache-resident; rotating expert coverage reduces that bias but is not a
forced cold-cache guarantee. Trial variation and the second run are in the raw
records; these are two runs on one GPU, not statistical hardware qualification.

## Reproduction

From the repository root in a CUDA13 development environment:

```bash
nvcc -O3 -std=c++17 -arch=sm_80 \
  onnxruntime/test/contrib_ops/cuda_kernels/qmoe_int3_padding_benchmark.cu \
  -o /tmp/qmoe_int3_padding_review
CUDA_VISIBLE_DEVICES=1 /tmp/qmoe_int3_padding_review > /tmp/padding-run1.json
CUDA_VISIBLE_DEVICES=1 /tmp/qmoe_int3_padding_review > /tmp/padding-run2.json
```

`--quick` runs smaller correctness fixtures, not the reported workload. The
standalone program is not registered as a normal ORT unit-test target.

## Other Review Fixes and Validation

- Guard all eight throwing ByteHelpers assertions under `#ifndef
  ORT_NO_EXCEPTIONS`; retain successful cases in both configurations.
- Accumulate concrete projection row constraints across weights, scales,
  zero points and bias even when weight rows are symbolic. New cases reject
  scale/bias and scale/zero-point contradictions.
- Clarify that each unpacked INT3 zero-point code must be 4 (`100` binary),
  not that every packed byte is 4; omission implies the same offset.
- Current-source normal schema/format regression: 17/17 tests passed.
- Format test translation unit compiled with `ORT_NO_EXCEPTIONS`: 6/6 passed,
  including ByteHelpers without aborting. The real `core/common/helper.cc`
  was also compiled with that flag to supply `PrintFinalMessage`.
- These native checks use cached Ninja flags/libraries and isolated rebuilt
  source objects. They are not a clean full ORT build or a globally
  exception-disabled build. Existing MLAS GNU-stack linker warning remains.