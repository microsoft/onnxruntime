# SM80 INT2 Grouped GEMM Foundation

This is the initial kernel prototype for Workstream 4, PR 4a of the
[mixed-width QMoE delivery plan](https://github.com/microsoft/onnxruntime/pull/32657).
It does not change QMoE dispatch or enable model-level native prefill.

## Supported Configuration

- SM80, FP16 activations/output, symmetric INT2 weights, and block size 64.
- Default CTA shape 32x128x64, warp shape 32x32x64, three pipeline stages.
- Explicit `tile_rows=64` selects CTA 64x128x64 and warp 32x64x64, also three
  stages. There is no automatic tactic selection or QMoE dispatch change.
- FP32 accumulation with FP16 output; no bias or activation epilogue.
- Positive N and K divisible by 64. Partial M and N tiles are supported.
- Positive total grouped row count fitting in a signed 32-bit integer.
- One or more experts, including experts receiving zero rows.

`IsInt2GroupedGemmSupported` checks metadata only. `RunInt2GroupedGemm` also
requires non-null device buffers and applies CUTLASS operand-alignment checks.
The caller must supply the current device's SM and multiprocessor count and
keep buffers alive until work on the supplied stream completes.

## Buffer Contract

For E experts, P total routed rows, N output columns, and K reduction elements:

| Parameter | Layout |
| --- | --- |
| `activations` | FP16 `[P, K]`, rows grouped contiguously by expert |
| `packed_weights` | E consecutive SM80 fpA_intB prepacked INT2 matrices, each K*N/4 bytes |
| `block_scales` | FP16 `[E, K/64, N]`, contiguous N dimension |
| `expert_row_ends` | E device int64 cumulative end offsets, exclusive |
| `output` | FP16 `[P, N]`, in the same grouped row order |

Offsets must be nonnegative, nondecreasing, and end at P. Repeated offsets
represent empty experts. These device-resident invariants are the caller's
responsibility; the launcher does not copy routing data to the host to validate
them. Buffer capacities must match the metadata, and output must not overlap
inputs.

The input model's raw unsigned codes use values 0 through 3, representing
`(code - 2) * scale`. Convert raw `[E, N, K/4]` weights with the existing
`unpack_uint2_transposed_to_int8_direct_cuda` and
`preprocess_weights_for_mixed_gemm_cuda` helpers, using SM80 and `W2_A16`.
The preprocessor mutates its scratch input. The direct tests exercise this
same packing pipeline with different scales across experts, columns, and K blocks.

The GEMM converts weight tiles internally; it does not allocate or write a
complete global-memory FP16 expert-weight buffer. It performs no routing or
finalization and requires no caller-supplied GEMM workspace.

## Validation

`Int2GroupedGemmTest.*` directly invokes this entry point and compares against
a scalar reference using the same quantized weights. Tests cover one expert,
empty/skewed experts, partial tiles, minimum alignment, and P=128/512/2048.
`Int2GroupedGemmValidationTest.*` covers unsupported metadata and null buffers.
Enable CUDA EP internal tests. In the plugin build, use the `Int2GroupedGemm*`
gtest filter. In the legacy build, `CUDA_EP_Unittest.All` loads the internal
test module and executes its registered suites. Set the environment variable
`GTEST_FILTER=Int2GroupedGemm*` for the inner module, and pass
`--gtest_filter=CUDA_EP_Unittest.All` to the outer provider-test executable.
The tests are registered for both builds.

P here is the sum of the per-expert GEMM row counts, not the model input token
count T. Future QMoE integration must benchmark T=128/512/2048 with the actual
top-k routing distribution; these small synthetic kernel tests do not qualify
model TTFT or throughput.

Local validation on 2026-09-29 used A100-SXM4-80GB, CUDA 12.8.61, GCC 13.3,
Release, SM80, internal tests enabled, and QUICK_BUILD disabled. The legacy
build started in an empty `build/int2-sm80-clean-20260929` directory, without
old objects or compiler caches; compilation errors were repaired and the same
isolated tree rebuilt after each edit.

- Normal CUDA provider, CUDA internal-test module, `onnxruntime_provider_test`,
  and `onnxruntime_test_all` targets built. The full test_all suite was not run.
- 15 direct tests passed; Compute Sanitizer memcheck also passed all 15 with
  zero errors. Three opt-in benchmarks are excluded from these counts.
- Dense-reference parity and the 64-row candidate include exhaustive small
  cases with empty experts, partial tiles, and minimum N/K alignment.
- Production W4 and W8 block-64 runners passed scalar-reference regressions.
- GPT-OSS-20B-shaped cases use E=32, top-k=4, K=2880, FC1 N=5760,
  FC2 N=2880, and T=128/512/2048 (P=512/2048/8192). All output values must
  be finite. Large cases compare sampled first/middle/last expert rows and
  columns on both sides of 64-column boundaries; small cases compare every
  output. Balanced and feasible skewed/empty-expert routing are covered.
- The existing `*QMoE*` operator suite passed 53 tests and skipped 11, with no
  failures. Mixtral INT4 skips because its dimensions are too small; it is not
  counted as CUDA INT4 coverage. Direct W4/W8 tests supply kernel-level coverage.
- On 2026-09-30, the separate `build/int2-sm80-plugin-clean-20260929` tree
  completed full provider-test linking and execution. The filter
  `Int2GroupedGemm*:*QMoE*` ran 82 tests: 70 passed, 12 skipped, zero failed.
  All 15 grouped tests passed, and a separate plugin Compute Sanitizer memcheck
  run passed those 15 with zero errors. The additional QMoE skip concerns routing
  statistics, which the plugin EP does not support.
- Plugin validation used local
  `CMAKE_CUDA_FLAGS="--diag-suppress=970 --diag-suppress=2189"` to handle
  CUDA 12.8 diagnostics from Abseil/Protobuf in production plugin sources.
  This is not evidence that the default plugin configuration or CI passes.
  The test target also includes the five existing FP16 SM80 fused launchers
  referenced by the dense runner's full template instantiation.

From the legacy build directory:

```bash
GTEST_FILTER='Int2GroupedGemm*' ./onnxruntime_provider_test --gtest_filter=CUDA_EP_Unittest.All
./onnxruntime_provider_test --gtest_filter='*QMoE*'
GTEST_FILTER='Int2GroupedGemm*' compute-sanitizer --tool memcheck --error-exitcode 99 \
  ./onnxruntime_provider_test --gtest_filter=CUDA_EP_Unittest.All
```

From the plugin build directory, direct tests do not need the legacy loader:

```bash
CUDA_VISIBLE_DEVICES=1 ./onnxruntime_provider_test --gtest_filter='Int2GroupedGemm*:*QMoE*'
CUDA_VISIBLE_DEVICES=1 compute-sanitizer --tool memcheck --error-exitcode 99 \
  ./onnxruntime_provider_test --gtest_filter='Int2GroupedGemm*'
```

## Initial Kernel Measurements

The opt-in `DISABLED_GptOssKernelBenchmark` uses the same scalar oracle, five
warmups after an initial launch, and three trials of 30 launches timed with CUDA
events on a nonblocking stream. Values below are median per-launch milliseconds.
These are stream elapsed times around host-submitted launches, not isolated
profiler kernel durations. Packing, allocation, H2D/D2H copies, and reference
computation are outside the GEMM timing. GPU 1 had no other compute clients at
the start; clocks were not locked. This is one local run, not a performance gate.

| T | P | W2 FC1 | W4 FC1, same shape | W4 FC2 |
| --- | --- | --- | --- | --- |
| 128 | 512 | 0.403695 | 0.391612 | 0.212548 |
| 512 | 2048 | 0.760764 | 0.766020 | 0.323072 |
| 2048 | 8192 | 2.476240 | 2.570210 | 1.253720 |

Both FC1 paths use the same 32x128x64, three-stage tactic. W4 is a production
grouped-kernel comparison with a different quantized weight distribution, not
an equal-quality model baseline. W2 is slightly slower at T=128 and similar at
larger T. The T=2048 W2 trial range was 2.47624-2.98469 ms, so these data do not
establish a robust speedup. No routing, SwiGLU, finalization, TTFT, or full
prefill time is measured here.

Separate one-pass event timings for prepack were 23.1-24.6 ms (W2 FC1),
24.1-24.3 ms (W4 FC1), and 10.2-12.5 ms (W4 FC2). These include transpose and
preprocessing launches and may include host submission gaps; they are not
steady-state GEMM overhead.

| Buffer | W2 FC1 bytes | W4 FC1 bytes | W4 FC2 bytes |
| --- | --- | --- | --- |
| Packed weights | 132710400 | 265420800 | 132710400 |
| FP16 scales | 16588800 | 16588800 | 8294400 |
| Prepack scratch | 4147456 | 8294656 | 4147456 |

The test also retains a raw GPU weight copy equal in size to the packed weights,
plus activations, output, and routing offsets. The table is allocation accounting,
not measured peak process memory. GEMM itself requests no external workspace or
full dequantized A16 weight buffer. W2 halves the FC1 packed-weight bytes, not
total model memory.

```bash
CUDA_VISIBLE_DEVICES=1 \
GTEST_FILTER='Int2GroupedGemmTest.DISABLED_GptOssKernelBenchmark' \
GTEST_ALSO_RUN_DISABLED_TESTS=1 \
./onnxruntime_provider_test --gtest_filter=CUDA_EP_Unittest.All
```

## Dense Baseline and Tactic Comparison

The opt-in `DISABLED_GptOssTacticBenchmark` compares identical INT2 codes,
activations, scales, expert offsets and FP32 accumulation with FP16 outputs.
The dense path transposes the test scales to the raw model layout and calls
the production `DequantizeNBits<half, uint8_t>` and `MoeGemmRunner<half, half, half>`.
Mode 1 dequantizes once before timing. Mode 2 dequantizes all experts before
every GEMM, like the full-weight fallback; allocations remain outside timing.
This is not an active-experts-only or tiled-dequantization baseline.

FC1 N=5760, K=2880, E=32; event median milliseconds from a second local run:

| Routing | T | W2 tile32 | W2 tile64 | Cached dense tile32 | Cached dense tile64 | Dequant+dense tile32 | Dequant+dense tile64 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Balanced | 128 | 0.404 | 0.602 | 0.651 | 0.663 | 1.538 | 1.548 |
| Balanced | 512 | 0.761 | 0.605 | 0.884 | 0.702 | 1.765 | 1.586 |
| Balanced | 2048 | 2.728 | 1.905 | 2.375 | 1.905 | 3.678 | 2.765 |
| Skewed | 128 | 0.233 | 0.228 | 0.217 | 0.189 | 1.097 | 1.071 |
| Skewed | 512 | 0.798 | 0.641 | 0.737 | 0.488 | 1.510 | 1.353 |
| Skewed | 2048 | 2.819 | 1.979 | 2.532 | 1.700 | 3.139 | 2.585 |

Balanced experts receive T/8 rows each. Skewed expert row counts are
1, 31, T-32, T, T, T with the remaining experts empty; their sum is 4T.
Every benchmark case passes the sampled scalar oracle and finite-output check.
The run used GPU 1; CPU plugin compilation ran concurrently. Clocks were not
locked and execution order was fixed, so repeat controlled measurements before
setting dispatch thresholds or asserting a stable speedup.

The 64-row W2 candidate helps larger per-expert M but is slower for balanced
T=128. The default remains 32. W2 avoids the per-call full dequantization cost,
but cached dense GEMM can be faster, particularly for skewed experts. The dense
FC1 buffer alone adds 1,061,683,200 bytes (1012.5 MiB); this is capacity accounting,
not a measured process peak. A 128-row W2 candidate was rejected at compile time
by CUTLASS thread-map assertions and is not exposed as supported.

```bash
CUDA_VISIBLE_DEVICES=1 \
GTEST_FILTER='Int2GroupedGemmTest.DISABLED_GptOssTacticBenchmark' \
GTEST_ALSO_RUN_DISABLED_TESTS=1 \
./onnxruntime_provider_test --gtest_filter=CUDA_EP_Unittest.All
```

## Remaining Acceptance Work

- Complete supported-platform CI, including the default plugin toolchain
  configuration. Local full plugin linking/execution is now validated with
  the CUDA diagnostic workaround recorded above.
- Repeat the dense/tactic measurements with controlled clocks and execution
  order before defining automatic tactic selection or performance gates.
- Collect profiler evidence and peak allocation measurements before claiming
  end-to-end memory or prefill improvements.

Mixed FC1/FC2 execution, SwiGLU, finalization, runtime dispatch, and model-level
qualification remain PR 4b work. BF16, additional block sizes, and other GPU
architectures are not enabled by this prototype.