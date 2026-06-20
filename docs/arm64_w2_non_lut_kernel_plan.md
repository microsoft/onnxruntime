# ARM64 W2 Non-LUT Kernel: Knowledge Dump and Development Plan

## Scope and Target

This document captures:

1. What was implemented for the AVX512 W2 native non-LUT kernel.
2. Which MLAS tests to run (and add) once W2 is supported on ARM64.
3. Which benchmarks to run on ARM64 hosts.
4. A practical initial development plan for ARM64 W2 non-LUT kernels.

Deployment assumptions:

- Target is ARMv8.x (for example, ARMv8.7 class devices such as recent Surface ARM64 devices).
- No dependence on ARMv9-only features (SME/SME2).
- Prefer two backends where available:
  - i8mm fast path
  - NEON dot-product fallback


## Current Status Snapshot (ARM64)

### Support matrix (today)

- LUT W2: not wired on ARM64 dispatch in current MLAS platform initialization.
- Native W2 non-LUT: not present on ARM64 today.
- Native W4:
  - SQNBIT_CompFp32 path exists on NEON.
  - SQNBIT_CompInt8 requires dot-product/i8mm capability for practical fast path.
- Native W8:
  - SQNBIT_CompInt8 path exists (again feature-dependent for fast SIMD path).

### Public block-length variant gating

At the API/variant layer (`qnbitgemm.cpp`):

- W4 and W8: BlkLen in {16, 32, 64, 128, 256}
- W2: BlkLen in {32, 64, 128}

For ARM64, the W2 variant is currently not backed by ARM dispatch hooks.


## ARM ISA Notes Relevant to This Work

- NEON dot-product (SDOT/UDOT): FEAT_DotProd (introduced as Armv8.2-A extension, optional by implementation).
- i8mm (SMMLA/UMMLA family): FEAT_I8MM (introduced in Armv8.6-A).

Implication for engineering:

- Baseline NEON-only machines should still have a correct fallback route, but native int8 throughput depends heavily on DotProd/i8mm.
- i8mm is the preferred fast backend where available.


## 1) What We Did for the AVX512 W2 Kernel

### Core enablement

- Added W2 variant support in variant selection (`SQ2BitGemmVariant_CompInt8`) and tied availability to dispatch function pointer presence.
- Enforced W2 availability contract to BlkLen {32, 64, 128}.

### Dispatch wiring

- Added AVX512 non-VNNI dispatch entrypoint for W2.
- Added AVX512-VNNI dispatch entrypoint for W2.
- Added `Q2BitGemmEffectiveBlockCountK` plumbing for effective/padded K-block stride.

### Packing and execution model

- Introduced/used W2 packed-B + block-sum path consistent with MLAS int8 correction flow.
- Kept SGEMM-style block-sum correction integration in the wrapper path.

### Test additions and hardening

- Added broad W2 unit coverage in `test_sqnbitgemm_2bit_gemm.cpp`:
  - scalar reference checks
  - SIMD checks for non-VNNI and VNNI
  - with and without zero-points
  - with and without bias
  - K-tail and N-tail coverage
  - BlkLen 32/64/128 coverage
- Added explicit availability contract test:
  - true for W2 + BlkLen {32,64,128}
  - false for W2 + BlkLen {16,256}
  - false for unsupported W2 compute types

### Review follow-up outcomes

- Kept cleanup changes that are compiler-safe:
  - zero-addend FMA init changed to MUL where equivalent
  - comments clarifying dpbusd operand order
  - comments clarifying tail-block asymmetry
- Reverted `_mm256_reduce_add_ps` usage due to MSVC intrinsic availability issues in this environment.


## 2) MLAS Tests to Run for ARM64 W2 (When Implemented)

## Existing tests to run

- `onnxruntime/test/mlas/unittest/test_sqnbitgemm_2bit_gemm.cpp`
- `onnxruntime/test/mlas/unittest/test_sqnbitgemm_2bit.cpp`

These provide a template for W2 correctness coverage and should be extended with ARM-specific backend dispatch hooks.

## New/extended ARM64 W2 test plan

### A. Kernel correctness parity

For each BlkLen in {32, 64, 128}:

- Scalar baseline vs ARM dot-product backend
- Scalar baseline vs ARM i8mm backend
- With/without zero-points
- With/without bias
- K-tail cases (BlockCountK not multiple of 4)
- N-tail cases (N not multiple of main tile width)
- M coverage: decode-like and prefill-like (e.g., M=1,4,128)

### B. Explicit backend forcing

To avoid host-dependent accidental coverage gaps:

- Add tests that explicitly call ARM dot backend entrypoint.
- Add tests that explicitly call ARM i8mm backend entrypoint.

This mirrors AVX512 explicit non-VNNI/VNNI path testing.

### C. Availability contract tests on ARM64

- Verify `MlasIsQNBitGemmAvailable(2, 32/64/128, SQNBIT_CompInt8)` behavior matches dispatch hooks and host feature availability.
- Verify W2 false for unsupported BlkLen 16/256.
- Verify unsupported W2 compute types remain false.

### D. Operator-level regression coverage

- `onnxruntime/test/contrib_ops/matmul_2bits_test.cc`
- Ensure W2 operator paths continue to pass for accuracy levels and fallback modes.


## 3) Benchmarks to Run on ARM64 Host

Run both microbench and end-to-end.

### A. MLAS microbench (primary)

File:

- `onnxruntime/test/mlas/bench/bench_qnbitgemm.cpp`

Compare:

- W2 native ARM (new)
- W4 native ARM int8 path
- Existing fallback paths for context

Recommended representative shape set:

- M in {1, 128}
- (K, N) in:
  - (384, 1024)
  - (1024, 192)
  - (1024, 384)
  - (1024, 4096)
  - (4096, 1024)
- BlkLen in {32, 64, 128}

Collect:

- mean/median/p95 latency
- throughput
- relative speedups vs fallback and vs W4

### B. LUT-related benchmark context

File:

- `onnxruntime/test/mlas/bench/bench_lutgemm.cpp`

Useful for comparing historical LUT and fallback baselines; ARM64 W2 non-LUT target should be compared against whichever ARM paths are active.

### C. End-to-end model benchmark

Use representative model workloads (decode + prefill-like lengths):

- seq_len in {32, 64, 128}
- warmup and steady-state runs
- report mean/median/p95/min/max and tokens/s

Include host feature stamp in report:

- DotProd present?
- i8mm present?

### D. Host classes to benchmark (if available)

- ARMv8.x + DotProd + i8mm
- ARMv8.x + DotProd only
- Baseline NEON-only


## 4) ARM64 W2 Non-LUT Initial Kernel Development Plan

## Phase 0: Design decisions

- Keep AVX512-aligned BlkLen support: 32, 64, 128.
- Reuse existing MLAS block-sum correction model.
- Keep dispatch surface symmetric with existing W4/W8 patterns.

## Phase 1: Dispatch and API plumbing

In ARM NEON dispatch (`GetMlasQNBitGemmDispatchNeon`), add W2 hooks:

- `Q2BitGemmPackQuantBDataSize`
- `SQ2BitGemmPackQuantBDataAndBlkSum`
- `SQ2BitGemmKernel_BlkSum_CompInt8`
- `Q2BitGemmEffectiveBlockCountK`

Ensure availability contracts in `MlasIsQNBitGemmAvailable` are satisfied only when these pointers are valid.

## Phase 2: First functional backend (DotProd-first)

- Implement W2 dot-product backend first for BlkLen=64.
- Then extend to BlkLen=32 and 128.
- Validate correctness against scalar reference before tuning.

Rationale:

- DotProd backend is simpler than i8mm bring-up and establishes correctness and packing contracts.

## Phase 3: i8mm backend

- Add i8mm-specialized kernel implementation for same BlkLen set.
- Keep entry signature identical to dot backend.
- Select backend using runtime feature detection in dispatch init.

## Phase 4: Test hardening

- Add explicit dot backend tests.
- Add explicit i8mm backend tests.
- Add availability contract tests for ARM64 feature combinations.
- Keep scalar reference path as oracle.

## Phase 5: Performance tuning

Tune in this order:

1. BlkLen=64 (most representative)
2. BlkLen=32
3. BlkLen=128

Focus areas:

- unpack strategy
- tile shape and register pressure
- prefetch and memory layout
- correction-step overhead minimization

## Phase 6: Rollout and guardrails

- Enable by default where feature-gated dispatch pointers are valid.
- Preserve fallback behavior when unavailable.
- Add benchmark snapshots to PR for ARM hosts.


## Suggested Execution Order (Practical)

1. ARM W2 DotProd BlkLen=64 end-to-end (pack + kernel + tests).
2. Extend DotProd to BlkLen=32 and 128.
3. Add i8mm path for same three BlkLen values.
4. Add explicit dual-backend tests.
5. Run full MLAS + operator + benchmark suite on ARM64 host(s).


## Deliverables Checklist

- [ ] ARM64 W2 dispatch hooks added
- [ ] W2 pack-size + pack + kernel wired on ARM64
- [ ] BlkLen 32/64/128 support complete
- [ ] DotProd backend functional
- [ ] i8mm backend functional
- [ ] Explicit tests for dot and i8mm paths
- [ ] Availability contract tests updated
- [ ] Bench results captured and compared against W4/fallback


## Notes

- Keep PRs incremental (for example: DotProd first, then i8mm) to simplify review and bisectability.
- Keep test names and structure parallel to AVX512 W2 tests for readability and maintenance.
