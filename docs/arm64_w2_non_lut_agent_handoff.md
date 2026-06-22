# ARM64 W2 Non-LUT: Implementation Handoff for Coding Agent

## Purpose

Actionable handoff for adding ARM64 W2 (2-bit) native non-LUT kernels in
ONNX Runtime MLAS, building on the already-landed AVX-512 W2 implementation.

Primary reference (design / rationale / benchmarks):

- `docs/arm64_w2_non_lut_kernel_plan.md`

This file focuses on concrete steps, files, gotchas, and verification.


## Target Outcome

Add ARM64 W2 native non-LUT support for:

- Compute type: `SQNBIT_CompInt8`
- Block lengths: `BlkLen` in `{32, 64, 128}`
- Backends:
  - NEON dot-product (FEAT_DotProd) -- correctness-first baseline
  - i8mm (FEAT_I8MM) -- preferred fast path when available

Keep existing behavior unchanged on other platforms (x86 AVX-512 and
fallback paths must not regress).


## Ground Truth Constraints

- W2 availability contract must remain: only `BlkLen` in `{32, 64, 128}`.
- ARM64 implementation must not require ARMv9-only features (no SME/SME2).
- Runtime dispatch picks i8mm when available, else DotProd, else fallback
  (today there is no W2 fallback path; absence of i8mm and DotProd should
  leave `MlasIsQNBitGemmAvailable(2, ...)` returning false on ARM64).


## Already Landed (Reference Implementation)

The AVX-512 W2 path is the canonical reference for layout, packing, and the
block-sum correction contract:

- Public variant gate: `SQ2BitGemmVariant_CompInt8` in
  `onnxruntime/core/mlas/lib/qnbitgemm.{h,cpp}` -- W2 allowed BlkLen is
  `{32, 64, 128}` only.
- Dispatch entries: `MlasSQNBitGemmDispatchAvx512` and
  `MlasSQNBitGemmDispatchAvx512vnni` in
  `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_avx512.cpp` and
  `sqnbitgemm_kernel_avx512vnni.cpp`.
- Scalar oracle / pack helper TU:
  `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_avx512_2bit.cpp` (compiled
  WITHOUT `-mavx512vnni` -- see "Gotchas" below).
- SIMD inner loops:
  - `sqnbitgemm_kernel_avx512_2bit_blklen32.h`
  - `sqnbitgemm_kernel_avx512_2bit_blklen64.h`
  - `sqnbitgemm_kernel_avx512_2bit_blklen128.h`
- Tests (already include scalar/SIMD/VNNI variants and availability contract
  checks): `onnxruntime/test/mlas/unittest/test_sqnbitgemm_2bit_gemm.cpp`.

The ARM64 implementation should mirror the dispatch wiring pattern and reuse
the same external function signatures so that the same tests can run with an
ARM backend swapped in for the kernel pointer.


## Files To Touch on ARM64

### Variant and API plumbing (no changes expected, just verify)

- `onnxruntime/core/mlas/lib/qnbitgemm.cpp`
- `onnxruntime/core/mlas/lib/qnbitgemm.h`

### ARM64 dispatch and kernels (this is where the work happens)

- `onnxruntime/core/mlas/lib/platform.cpp`
- `onnxruntime/core/mlas/lib/mlasi.h`
- `onnxruntime/core/mlas/lib/qnbitgemm_kernel_neon.cpp`
- `onnxruntime/core/mlas/lib/qnbitgemm_kernel_neon.h`
- `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_neon_int8.cpp`
- `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_neon_int8_i8mm.cpp`
- `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_neon_fp32.cpp`

Prefer adding new TUs (e.g. `sqnbitgemm_kernel_neon_2bit.cpp`,
`sqnbitgemm_kernel_neon_2bit_i8mm.cpp`) rather than mega-files. Wire each
TU's compile flags in `cmake/onnxruntime_mlas.cmake` (the existing
`set_source_files_properties(... COMPILE_FLAGS "-march=armv8.2-a+i8mm")`
patterns for the W4/W8 i8mm files are the template).

### Existing W2 x86 reference (for parity reading)

- `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_avx512_2bit.h`
- `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_avx512_2bit.cpp`
- `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_avx512_2bit_blklen{32,64,128}.h`

### Tests and benches

- `onnxruntime/test/mlas/unittest/test_sqnbitgemm_2bit_gemm.cpp`
- `onnxruntime/test/mlas/unittest/test_sqnbitgemm_2bit.cpp`
- `onnxruntime/test/contrib_ops/matmul_2bits_test.cc`
- `onnxruntime/test/mlas/bench/bench_qnbitgemm.cpp`
- `onnxruntime/test/mlas/bench/bench_lutgemm.cpp`


## Implementation Plan (Concrete)

### Step 1: ARM64 W2 dispatch hooks

In `GetMlasQNBitGemmDispatchNeon(HasDotProductInstructions, HasI8MMInstructions)`
(`qnbitgemm_kernel_neon.cpp`), add W2 function pointers when either
`HasDotProductInstructions` or `HasI8MMInstructions` is true:

- `Q2BitGemmPackQuantBDataSize`
- `SQ2BitGemmPackQuantBDataAndBlkSum`
- `SQ2BitGemmKernel_BlkSum_CompInt8`
- `Q2BitGemmEffectiveBlockCountK`

Notes:

- Leave hooks `nullptr` on hosts where no backend is available -- the
  availability contract is dispatch-pointer-driven.
- Confirm `MlasIsQNBitGemmAvailable(2, BlkLen, SQNBIT_CompInt8)` returns
  true iff `BlkLen` in `{32, 64, 128}` AND the dispatch pointers are set.

### Step 2: Pack-size and pack+blksum on ARM64

Add W2 pack helpers in a new TU under the NEON path. These can be backend-
agnostic (same packed layout consumable by both DotProd and i8mm kernels):

- `Q2BitGemmPackQuantBDataSize_*`
- `SQ2BitGemmPackQuantBDataAndBlkSum_*`

Design choice:

- Reuse the AVX-512 K-block-grouping layout where it makes sense; deviate
  only if NEON / i8mm tile shapes demand it.
- Define and return effective K-block stride via
  `Q2BitGemmEffectiveBlockCountK`.

### Step 3: DotProd backend kernel (correctness first)

Implement first working W2 kernel for the DotProd backend in this order:

1. `BlkLen=64`
2. `BlkLen=32`
3. `BlkLen=128`

Kernel signature must match the existing dispatch:

- `SQ2BitGemmKernel_BlkSum_CompInt8(...)`

Keep the correction path identical to existing int8 flow:

- core int8 GEMM accumulation
- block-sum correction add (`ABlockSum` / `QuantBBlkSum`)

### Step 4: i8mm backend kernel

Implement i8mm backend for the same three block lengths.

- Reuse the same external signature.
- Keep the packed-B layout contract compatible with the DotProd path if
  possible; if not, document the divergence in code comments.
- Use runtime gating already present (`HasArmNeon_I8MM`) to install the
  i8mm dispatch pointers in `platform.cpp` initialization.

### Step 5: Tests (must-have)

Extend `test_sqnbitgemm_2bit_gemm.cpp`:

- For each `BlkLen` in `{32, 64, 128}`:
  - scalar oracle vs DotProd backend
  - scalar oracle vs i8mm backend
  - with/without zero-points
  - with/without bias
  - K-tail and N-tail shapes
  - M coverage: `{1, 4, 128}` (decode-like and prefill-like)
- Gate ARM-specific tests on host features (mirror the existing
  `Avx512Supported_` / dispatch-pointer-equality guard pattern that the
  x86 SIMD tests already use).
- Extend the availability-contract test:
  - true for W2 + `{32, 64, 128}` when backend hooks installed
  - false for W2 + `{16, 256}`
  - false for unsupported W2 compute types

If backend forcing cannot be done by host feature detection alone (e.g.,
the host has both DotProd and i8mm), add explicit backend test entrypoints
that call the kernel function pointer directly, the way the x86 W2 tests
call `SQ2BitGemmKernel_BlkSum_CompInt8_Avx512_Dispatch` and
`...Avx512Vnni_Dispatch` separately.

### Step 6: Benchmarks and validation

Run `bench_qnbitgemm` and compare:

- W2 native ARM (new) vs W4 native ARM
- W2 native ARM vs fallback path (if/when one exists)

Collect at minimum:

- mean/median/p95 latency
- throughput
- shape-level deltas for the representative `(M, N, K)` set in the plan doc


## Concrete Contracts (Layout, K-tail, Tile Shapes, Pack Sharing, Smoke)

These are explicit pins so the agent does not have to reverse-engineer
them from the AVX-512 sources.

### Packed-B layout (target contract for ARM)

Mirror AVX-512 W2 (canonical reference: the BlkLen-specific headers in
`onnxruntime/core/mlas/lib/sqnbitgemm_kernel_avx512_2bit_blklen{32,64,128}.h`):

- Source weights are 2-bit, stored 4 weights per byte.
- A "block" is `BlkLen` K-elements. Per N-col, a block occupies
  `BlkLen/4` bytes of packed B.
- A "block-group" is 4 consecutive blocks along K (so 4 * `BlkLen`
  elements = 4 * `BlkLen/4` bytes per N-col per group).
- The packed buffer is grouped along K in chunks of 4 blocks so that the
  inner SIMD kernel consumes one group per inner-K step.
- Per-block float scale and an int32 block-sum live alongside the packed
  weights. Block-sum is consumed by the SGEMM-style correction step
  outside the int8 inner loop, identical to W4/W8.
- N-tile shape: keep the existing W4/W8 convention (`kNCols4 = 4`
  N-columns per packed N-tile) so block-sum addressing can be reused.

### K-tail handling (must match AVX-512 contract)

When `BlockCountK` is not a multiple of 4 (the block-group size):

- Round `BlockCountK` up to a multiple of 4 for the packed buffer.
- Zero-fill the trailing 1-3 padding blocks (both weights and per-block
  scale). They must contribute 0 to the dot product.
- `Q2BitGemmEffectiveBlockCountK` returns the padded count so callers
  walk the packed buffer correctly.
- Kernel inner-K loop runs on padded blocks; the block-sum correction
  step still runs only on the logical `BlockCountK`.

### Suggested initial tile shapes

Start with these and tune in Phase 5:

- DotProd backend (SDOT/UDOT):
  - `BlkLen=64`: R2xC4 main tile (M=2, N=4), R1xC4 odd-M tail. Use one
    SDOT per K-quad per (M,N) lane.
  - `BlkLen=32`: same shape, half the K-quads per block.
  - `BlkLen=128`: same shape, twice the K-quads per block.
- i8mm backend (SMMLA / UMMLA):
  - `BlkLen=64`: R2xC4 main tile, but consume 8 K-elements per SMMLA
    (i8mm is an 8x8x4 mma -> 4-elem K tile). Plan a 2:1 unroll on K to
    keep register pressure flat with DotProd.
  - `BlkLen=32` / `BlkLen=128`: same shape, adjust K-unroll.

These are starting points, not requirements. Validate correctness first.

### Pack-helper sharing

Implement ONE pack helper TU that produces a packed-B layout consumed by
both backends. The packed bytes must be identical on output regardless
of which backend will later consume them. The DotProd and i8mm inner
kernels then load and interpret the same bytes through different SIMD
intrinsics. Document any deviation in code comments.

Rationale: a single layout removes a class of bugs and lets the dispatch
table install either backend without re-packing weights.

### `MlasIsQNBitGemmAvailable` truthfulness

Today, x86 gates W2 availability strictly by the AVX-512 dispatch
pointers and the `BlkLen` set. On ARM, do the same:

- W2 + `BlkLen` `{32,64,128}` + `SQNBIT_CompInt8`: true iff ARM W2
  dispatch pointers are installed (DotProd or i8mm backend present).
- W2 + `BlkLen` `{16,256}`: false.
- W2 + unsupported compute types: false.

The op-level tests in
`onnxruntime/test/contrib_ops/matmul_2bits_test.cc` already key off
`MlasIsQNBitGemmAvailable`, so they should pass on ARM hosts where the
ARM dispatch is installed without per-architecture branching in the
tests themselves.

### Test guard style (match the x86 file)

In `test_sqnbitgemm_2bit_gemm.cpp`, the x86 file uses two patterns:

1. `if (!GetMlasPlatform().Avx512Supported_) GTEST_SKIP();` for
   scalar/AVX-512-baseline tests.
2. `if (GetMlasPlatform().QNBitGemmDispatch != &MlasSQNBitGemmDispatchAvx512vnni) GTEST_SKIP();`
   for VNNI-specific paths.

For ARM, mirror pattern (2) -- skip on dispatch-pointer identity rather
than raw feature flags -- so adding a new backend later doesn't silently
break the gate.

### Build-smoke after the per-file flag override

If you add a per-file CMake compile-flag override on the ARM W2 scalar /
pack TU (mirroring the AVX-512 case that strips `-mavx512vnni`), verify
post-build that the desired instructions are absent. Example, after a
release build on Linux ARM64:

```bash
# Scalar / pack TU should NOT contain i8mm encodings.
aarch64-linux-gnu-objdump -d build/Release/libonnxruntime_mlas.a \
    | grep -E '\b(smmla|ummla|usmmla)\b' | head
```

A clean smoke (no hits) confirms the per-file flag pin took effect.


## Suggested Incremental PR Split

### PR A (plumbing + first kernel)

- W2 ARM dispatch hooks in `platform.cpp` / `qnbitgemm_kernel_neon.cpp`.
- W2 pack-size + pack+blksum on ARM.
- DotProd kernel for `BlkLen=64`.
- Tests for `BlkLen=64` (scalar vs DotProd, availability contract).

### PR B (complete block lengths)

- DotProd kernels for `BlkLen=32` and `BlkLen=128`.
- Full shape / K-tail / N-tail tests.

### PR C (i8mm acceleration)

- i8mm backend for `BlkLen` in `{32, 64, 128}`.
- Explicit-backend tests.
- Benchmark snapshots.
- Verify the post-build smoke check from the "Concrete Contracts"
  section still shows no `smmla` / `ummla` / `usmmla` in the scalar /
  pack TU even after i8mm intrinsics are added in the dedicated i8mm
  TU.


## Build and Test Commands (Examples)

ARM64 host (Linux):

```bash
# Configure (one-time per build dir)
python tools/ci_build/build.py \
    --build_dir build/Release \
    --config Release \
    --skip_submodule_sync --skip_tests --parallel \
    --build_shared_lib

# Incremental rebuild of the MLAS test target
cmake --build build/Release --config Release --target onnxruntime_mlas_test --parallel
```

Run W2-focused MLAS tests:

```bash
./build/Release/onnxruntime_mlas_test --gtest_filter='MlasSq2BitTest.*'
```

Run W2 operator tests in `onnxruntime_test_all`:

```bash
./build/Release/onnxruntime_test_all \
    --gtest_filter='MatMul2Bits.*:MatMulNBitsLutGemm.*'
```

Run the QNBit microbench:

```bash
./build/Release/onnxruntime_mlas_bench --benchmark_filter=QNBITGEMM
```

ARM64 host (Windows):

```powershell
# Equivalent paths under build\Release on Windows ARM64
.\build\Release\Release\onnxruntime_mlas_test.exe --gtest_filter=MlasSq2BitTest.*
```


## Gotchas From the AVX-512 W2 Bring-Up (Read Before You Code)

### G1: Per-file compile flags can be silently widened under LTO

The AVX-512 W2 scalar oracle / pack helper TU
(`sqnbitgemm_kernel_avx512_2bit.cpp`) is intentionally pure C++ but lives
in the AVX-512 compile group. Under `--enable_lto`, the compiler is free
to re-codegen its loops at link time using the union of feature flags in
the binary, which would autovectorize the int8 dot-product loops to
`vpdpbusd` (AVX-512 VNNI) and SIGILL on AVX-512-only (non-VNNI) hosts.

This is fixed today by an explicit
`set_source_files_properties(... COMPILE_FLAGS "-mfma -mavx512bw -mavx512dq -mavx512vl")`
override on that TU in `cmake/onnxruntime_mlas.cmake`. Per-file flags are
hints under LTO; if the override is ever insufficient, escalate to a
`#pragma GCC target("...,no-avx512vnni")` block around the TU.

**ARM analogue to watch for:** do not give the scalar oracle / pack
helper TU `-march=...+i8mm`. Production hosts without i8mm reach that
TU through the AVX-512-equivalent dispatch (the DotProd dispatch) and
would SIGILL on autovectorized SMMLA/UMMLA encodings. Keep the scalar /
pack TU on a baseline NEON+DotProd target, and put i8mm intrinsics only
in the dedicated i8mm TU.

### G2: Scalar tests run scalar-named functions, not scalar-codegen functions

The `Scalar_*` tests in `test_sqnbitgemm_2bit_gemm.cpp` call internal
scalar reference symbols directly, bypassing `MlasIsQNBitGemmAvailable`.
That means the function may still be autovectorized by the compiler --
"scalar" is the source-level intent, not a codegen guarantee. Gate
those tests on the appropriate host-feature flag (the x86 ones gate on
`Avx512Supported_`); the ARM equivalent should gate on
`HasArmNeonDot()` or `HasArmNeon_I8MM()` as appropriate, OR rely on
proper feature gating in `platform.cpp` plus the dispatch-pointer-
equality guard pattern.

### G3: Don't trust "PassedUnchanged" reads of stale CI artifacts

If you split build and test into two pipelines, double-check that the
test pipeline consumes a freshly built artifact from your branch SHA --
not the last green artifact built from `main`. The AVX-512 W2 PR hit
this multiple times before we noticed.

### G4: Production vs test pack-helper reach

`SQ2BitGemmPackQuantBDataAndBlkSum_Scalar` is installed in the AVX-512
W2 dispatch table and runs at model load on AVX-512 hosts in production.
The ARM equivalent must be safe to run on every host where ARM W2
dispatch is installed -- not just on the "fast" hosts.


## Definition of Done

- ARM64 W2 `SQNBIT_CompInt8` available for `BlkLen` `{32, 64, 128}` when
  DotProd and/or i8mm backend is present.
- No false-positive availability for unsupported block lengths or
  compute types.
- Unit tests pass for scalar + DotProd + i8mm paths (or skips are
  feature-truthful).
- Benchmark evidence recorded for at least one ARM64 host.
- No regressions on existing W4 / W8 ARM paths.
- No regressions on x86 paths.


## Risks to Watch

- Pack layout mismatch between kernel and block-sum correction path.
- K-tail handling for K not a multiple of block-group size.
- Host-dependent implicit coverage masking backend-specific bugs.
- Performance regressions from suboptimal tile choice on ARM.
- LTO widening per-file ISA flags (see G1).


## Quick Notes for the Next Agent

- Mirror AVX-512 W2 structure conceptually but do not copy x86-specific
  assumptions directly.
- Keep the public API stable. Changes should be confined to ARM dispatch
  registration and ARM kernel sources, plus narrow CMake updates.
- Prefer correctness-first bring-up (DotProd `BlkLen=64`) then expand and
  optimize.
- Keep PRs small (PR A / B / C above).
- The scalar oracle in
  `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_avx512_2bit.cpp` is
  architecturally portable; tests already use it as the cross-arch
  reference. You should not need to add a separate ARM scalar oracle.

