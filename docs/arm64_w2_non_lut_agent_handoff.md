# ARM64 W2 Non-LUT: Implementation Handoff for Coding Agent

## Purpose

This document is an actionable handoff for implementing ARM64 W2 native non-LUT kernels in ONNX Runtime MLAS.

Primary reference:

- `docs/arm64_w2_non_lut_kernel_plan.md`

This file focuses on concrete implementation steps, files, and verification commands.


## Target Outcome

Add ARM64 W2 native non-LUT support for:

- Compute type: `SQNBIT_CompInt8`
- Block lengths: `BlkLen` in `{32, 64, 128}`
- Backends:
  - NEON dot-product backend (fallback)
  - i8mm backend (preferred when available)

Keep existing behavior unchanged on other platforms.


## Ground Truth Constraints

- W2 availability contract must remain: only `{32,64,128}`.
- ARM64 implementation must not require ARMv9-only features (SME/SME2).
- Runtime dispatch should pick i8mm when available, otherwise DotProd, otherwise fallback.


## Current Relevant Files

### Variant and API plumbing

- `onnxruntime/core/mlas/lib/qnbitgemm.cpp`
- `onnxruntime/core/mlas/lib/qnbitgemm.h`

### ARM64 dispatch and kernels

- `onnxruntime/core/mlas/lib/platform.cpp`
- `onnxruntime/core/mlas/lib/mlasi.h`
- `onnxruntime/core/mlas/lib/qnbitgemm_kernel_neon.cpp`
- `onnxruntime/core/mlas/lib/qnbitgemm_kernel_neon.h`
- `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_neon_int8.cpp`
- `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_neon_int8_i8mm.cpp`
- `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_neon_fp32.cpp`

### Existing W2 x86 reference implementation (for parity)

- `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_avx512_2bit.h`
- `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_avx512_2bit.cpp`
- `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_avx512_2bit_blklen32.h`
- `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_avx512_2bit_blklen64.h`
- `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_avx512_2bit_blklen128.h`

### Tests and benches

- `onnxruntime/test/mlas/unittest/test_sqnbitgemm_2bit_gemm.cpp`
- `onnxruntime/test/mlas/unittest/test_sqnbitgemm_2bit.cpp`
- `onnxruntime/test/contrib_ops/matmul_2bits_test.cc`
- `onnxruntime/test/mlas/bench/bench_qnbitgemm.cpp`
- `onnxruntime/test/mlas/bench/bench_lutgemm.cpp`


## Implementation Plan (Concrete)

## Step 1: Add ARM64 W2 dispatch hooks

In `GetMlasQNBitGemmDispatchNeon(...)` (`qnbitgemm_kernel_neon.cpp`), add function pointers:

- `Q2BitGemmPackQuantBDataSize`
- `SQ2BitGemmPackQuantBDataAndBlkSum`
- `SQ2BitGemmKernel_BlkSum_CompInt8`
- `Q2BitGemmEffectiveBlockCountK`

Notes:

- Keep hooks null when unsupported by selected backend.
- Ensure availability behavior in `MlasIsQNBitGemmAvailable` remains truthful.

## Step 2: Implement W2 pack-size and pack+blksum on ARM64

Add W2 pack helpers under NEON path (prefer in `qnbitgemm_kernel_neon.cpp` and/or new dedicated source if cleaner):

- `Q2BitGemmPackQuantBDataSize_*`
- `SQ2BitGemmPackQuantBDataAndBlkSum_*`

Design choice:

- Keep K-block-grouping logic aligned with AVX512 W2 model where practical.
- Define and return effective K-block stride via `Q2BitGemmEffectiveBlockCountK`.

## Step 3: DotProd backend kernel (first)

Implement first working W2 kernel for DotProd backend:

- `BlkLen=64` first
- then extend to `BlkLen=32` and `BlkLen=128`

Kernel API should align with existing dispatch signature:

- `SQ2BitGemmKernel_BlkSum_CompInt8(...)`

Keep correction path identical to current int8 flow:

- core int8 GEMM accumulation
- block-sum correction add (ABlockSum/QuantBBlkSum)

## Step 4: i8mm backend kernel

Implement i8mm backend for same three block lengths.

- Reuse same external signature
- Keep packed layout contract compatible with DotProd path where possible
- Use runtime gating already present (`HasArmNeon_I8MM`) to select this backend in dispatch initialization

## Step 5: Tests (must-have)

Extend/add tests in `test_sqnbitgemm_2bit_gemm.cpp`:

- For each `BlkLen` in `{32,64,128}`:
  - scalar vs dot backend
  - scalar vs i8mm backend
  - with/without zero-points
  - with/without bias
  - K-tail and N-tail shapes

Add/extend availability contract checks for ARM:

- true for W2 + `{32,64,128}` only when backend hooks are valid
- false for W2 + `{16,256}`

If backend forcing cannot be done by host feature alone, add explicit backend test entrypoints.

## Step 6: Benchmarks and validation

Run `bench_qnbitgemm` and compare:

- W2 native ARM vs W4 native ARM
- W2 native ARM vs fallback path

Collect at minimum:

- latency mean/median/p95
- throughput
- shape-level deltas for representative `(M,N,K)` set


## Suggested Incremental PR Split

### PR A (plumbing + first kernel)

- W2 ARM dispatch hooks
- W2 pack/size
- DotProd BlkLen=64
- basic tests for BlkLen=64

### PR B (complete block lengths)

- DotProd BlkLen=32/128
- full shape/tail tests

### PR C (i8mm acceleration)

- i8mm backend for 32/64/128
- explicit backend tests
- benchmark updates


## Commands: Build and Test (Examples)

Use existing ONNX Runtime build flow on ARM64 host. Example patterns:

```powershell
# Build MLAS/unit tests (adjust build dir/preset to local setup)
cmake --build <build_dir> --config Release --target onnxruntime_mlas_test --parallel
```

```powershell
# Run W2-focused MLAS tests
aarch64\Release\onnxruntime_mlas_test --gtest_filter=MlasSq2BitTest.*
```

```powershell
# Run MatMul 2-bit contrib op tests (if built in this config)
ctest -C Release -R matmul_2bits --output-on-failure
```

```powershell
# Run QNBit benchmark binary (shape args depend on harness)
aarch64\Release\onnxruntime_mlas_bench --benchmark_filter=QNBITGEMM
```


## Definition of Done

- ARM64 W2 `SQNBIT_CompInt8` available for `BlkLen` 32/64/128 when dot/i8mm backend is present.
- No false-positive availability for unsupported block lengths.
- Unit tests pass for scalar + dot + i8mm paths (or skips are feature-truthful).
- Benchmark evidence recorded for at least one ARM64 host.
- No regressions on existing W4/W8 paths.


## Risks to Watch

- Pack layout mismatch between kernel and correction path.
- K-tail handling for non-multiple-of-block-group K.
- Host-dependent implicit coverage masking backend-specific bugs.
- Performance regressions from suboptimal tile choice on ARM.


## Quick Notes for the Next Agent

- Mirror AVX512 W2 structure conceptually, but avoid copying x86-specific assumptions directly.
- Keep public API behavior stable; changes should primarily be in ARM dispatch and kernel implementation.
- Prefer correctness-first bring-up (dot BlkLen64) then optimize/expand.
