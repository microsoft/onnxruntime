/*++

Copyright 2025 FUJITSU LIMITED

Module Name:

    sgemm_sve.h

Abstract:

    Prototypes for the SVE FP32 GEMM (SGEMM) compute and packing kernels.

    The symbols are extern "C" so the two interchangeable implementations
    link identically: the SVE intrinsics reference translation unit
    (sgemm_sve.cpp) and the generated KleidiAI-style machine-code variant
    (aarch64/sgemm_sve_asm.S). The build links exactly one of the two.

    This header is includable from translation units compiled WITHOUT SVE
    support (the SGEMM driver, sgemm.cpp), on any platform, and it pulls in
    nothing from mlasi.h. sve/gen_sve_asm.py compiles sgemm_sve.cpp on its own
    to freeze it, so that translation unit must not depend on the wider ONNX
    Runtime include tree.

--*/

#pragma once

#include <cstddef>

//
// Repeated from mlas.h/mlasi.h so sgemm_sve.cpp stays self-contained for the
// generator. The guards let the real definitions win when mlasi.h was seen
// first; a static_assert in sgemm.cpp checks the value that matters.
//

#if !defined(MLASCALL)
#if (_MSC_VER >= 800) || defined(_STDCALL_SUPPORTED)
#define MLASCALL __stdcall
#else
#define MLASCALL
#endif
#endif

#if !defined(MLAS_SGEMM_STRIDEN_THREAD_ALIGN)
#define MLAS_SGEMM_STRIDEN_THREAD_ALIGN 16
#endif

#if !defined(MLAS_UNREFERENCED_PARAMETER)
#define MLAS_UNREFERENCED_PARAMETER(parameter) ((void)(parameter))
#endif

//
// Width in floats of one packed-B block produced by the SVE packing routines.
//
#if !defined(kMlasSvePackedBBlockWidth)
#define kMlasSvePackedBBlockWidth 16
#endif

//
// Redundant when the TU is already built for SVE, and on some compilers it
// conflicts with the command-line -march.
//
#if !defined(MLAS_SVE_TARGET)
#if defined(__ARM_FEATURE_SVE)
#define MLAS_SVE_TARGET
#else
#define MLAS_SVE_TARGET __attribute__((target("arch=armv8.2-a+sve")))
#endif
#endif

extern "C" {

//
// Compute kernels. Both return the number of rows handled.
//

MLAS_SVE_TARGET
MLASCALL
size_t
MlasSgemmKernelZero_sve(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
);

MLAS_SVE_TARGET
MLASCALL
size_t
MlasSgemmKernelAdd_sve(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
);

//
// Packing helpers used by MlasSgemmTransposePackB / MlasSgemmCopyPackB.
//

void MLAS_SVE_TARGET MLASCALL
MlasSveTranspose(float*& D, const float*& b, size_t ldb, size_t& x);

void MLAS_SVE_TARGET MLASCALL
MlasSveScatterStore(float* d, const float* b);

void MLAS_SVE_TARGET MLASCALL
MlasSveLoadStore(float* D, const float* b);

void MLAS_SVE_TARGET MLASCALL
MlasSveZeroInitialize(float* d);

//
// The N = 4 and N = 8 shapes of the SVE transposing packer, as concrete
// symbols: a template instantiation emits a weak one, which cannot be frozen.
//

void MLAS_SVE_TARGET MLASCALL
MlasSveTransposePackB4x4(float* D, const float* B, size_t ldb);

void MLAS_SVE_TARGET MLASCALL
MlasSveTransposePackB8x4(float* D, const float* B, size_t ldb);

//
// Largest K routed to MlasSgemmSmallKKernel_sve. Above this the packed B
// block starts being reused across rows of A and the packed kernels win.
//
#define MLAS_SGEMM_SVE_SMALLK_MAX 8

//
// Rank-K update for a small K, reading B directly rather than through the
// packed 16-column buffer. Row k of B starts at B + k * ldb and is contiguous
// across N. Computes alpha * A * B, adding into C unless ZeroMode.
//
void MLAS_SVE_TARGET MLASCALL
MlasSgemmSmallKKernel_sve(
    const float* A,
    size_t lda,
    const float* B,
    size_t ldb,
    float* C,
    size_t ldc,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    float alpha,
    bool ZeroMode
);

//
// Vector length in 32-bit words. sgemm.cpp uses this to keep 128-bit SVE on the
// NEON kernels, a choice that used to be made inside the SVE kernels
// themselves; a frozen kernel cannot call out to the NEON one because
// gen_sve_asm.py forbids bl/blr.
//

MLAS_SVE_TARGET
MLASCALL
size_t
MlasSveVectorLengthWords(void);

}  // extern "C"
