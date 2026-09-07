/*++

Copyright (c) Microsoft Corporation. All rights reserved.
Copyright 2025 FUJITSU LIMITED

Licensed under the MIT License.

Module Name:

    hgemm.cpp

Abstract:

    This module implements the half precision (FP16 / MLFloat16) matrix/matrix
    multiply operation (HGEMM). It is the FP16 analogue of sgemm.cpp and
    provides the operation / packing / threading / batch driver layer that
    sits ABOVE the SVE compute kernels in sve/halfgemm_kernel_sve.cpp.

    This driver owns the public HGEMM entry points declared in mlas.h:

        MlasHGemmSupported   - capability probe used by the EP layer
        MlasGemmBatch(...)   - the MLAS_HGEMM_DATA_PARAMS overload

    The compute is SVE-only. On a target without SVE, MlasHGemmSupported
    returns false and the EP layer routes to the legacy MlasHalfGemmBatch
    path or the Eigen fallback (see gemm.cc / matmul.cc).

    Element type: half precision is carried as MLAS_FP16 (a 16-bit storage
    type). alpha/beta arrive in MLAS_HGEMM_DATA_PARAMS as FP16 bit encodings
    (uint16) and are widened to float at the arithmetic boundary.

--*/

#include "mlasi.h"
#include "mlas_float16.h"

#include <algorithm>
#include <exception>

using MLAS_HGEMM_FP16 = MLAS_FP16;

//
// ============================================================================
//  Forward declarations of the SVE compute kernels implemented in
//  sve/halfgemm_kernel_sve.cpp. These are the functions this driver links
//  against. The element type at the kernel boundary is the raw fp16 storage
//  word (_mlas_fp16_ == uint16_t == MLAS_FP16::val).
// ============================================================================
//
#if defined(MLAS_USE_SVE)

//
// The SVE compute kernels. Two interchangeable implementations satisfy these
// symbols: the intrinsics reference in sve/halfgemm_kernel_sve.cpp and the
// frozen machine code in aarch64/halfgemm_sve_asm.S (see
// onnxruntime_SVE_HGEMM_ASM in onnxruntime_mlas.cmake).
//
#include "sve/halfgemm_sve.h"
#include "sve/halfgemv_sve.h"

static_assert(PACKED_B_BLOCK_WIDTH_FP16 == MLAS_HGEMM_STRIDEN_THREAD_ALIGN,
              "packed-B block width must match the driver's N thread alignment");

#endif // MLAS_USE_SVE

//
// Number of rows from matrix A to transpose to a local buffer when TransA is
// requested. Tuned to the widest row tile (8).
//
#define MLAS_HGEMM_TRANSA_ROWS              12

//
// Physical width (in fp16 elements) of one packed B block. MUST match
// PACKED_B_BLOCK_WIDTH_FP16 in sve/halfgemm_kernel_sve.cpp.
//
#ifndef PACKED_B_BLOCK_WIDTH_FP16
#define PACKED_B_BLOCK_WIDTH_FP16           MLAS_HGEMM_STRIDEN_THREAD_ALIGN
#endif  // set by sve/halfgemm_sve.h when MLAS_USE_SVE


// ============================================================================
//  beta scaling   (analogue of MlasSgemmMultiplyBeta)
//  Multiplies every element of the CountM x CountN output tile by beta.
//  Needed for the general C = alpha*A*B + beta*C form; the SVE kernels only
//  implement beta in {0,1} via ZeroMode, so arbitrary beta routes here first.
// ============================================================================
static void
MlasHgemmMultiplyBeta(
    MLAS_HGEMM_FP16* C,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    float beta
    )
{
    if (beta == 1.0f) {
        return;
    }
    auto* c = reinterpret_cast<_mlas_fp16_*>(C);
    for (size_t m = 0; m < CountM; ++m) {
        _mlas_fp16_* row = c + m * ldc;
        for (size_t n = 0; n < CountN; ++n) {
            row[n] = MLAS_Float2Half(MLAS_Half2Float(row[n]) * beta);
        }
    }
}


// ============================================================================
//  A transpose into local panel   (analogue of MlasSgemmTransposeA)
//  Transposes a CountX(rows of stored A) x CountY(cols of stored A) block of A
//  into D (row-major CountY x CountX) so the NoTrans kernels can consume it.
// ============================================================================
static void
MlasHgemmTransposeA(
    MLAS_HGEMM_FP16* D,
    const MLAS_HGEMM_FP16* A,
    size_t lda,
    size_t CountY,
    size_t CountX
    )
{
#if defined(MLAS_USE_SVE)
    if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArmSve()) {
        MlasHgemmTransposeA_sve(reinterpret_cast<_mlas_fp16_*>(D),
                                reinterpret_cast<const _mlas_fp16_*>(A),
                                lda, CountY, CountX);
        return;
    }
#endif
    auto* d = reinterpret_cast<_mlas_fp16_*>(D);
    const auto* a = reinterpret_cast<const _mlas_fp16_*>(A);
    // D[y, x] = A_stored[x, y]; D is CountY x CountX row-major.
    for (size_t y = 0; y < CountY; ++y) {
        for (size_t x = 0; x < CountX; ++x) {
            d[y * CountX + x] = a[x * lda + y];
        }
    }
}


// ============================================================================
//  B packing, NoTrans   (analogue of MlasSgemmCopyPackB)
//  Wrapper selecting SVE vs scalar fallback.
// ============================================================================
static void
MlasHgemmCopyPackB(
    MLAS_HGEMM_FP16* D,
    const MLAS_HGEMM_FP16* B,
    size_t ldb,
    size_t CountX,
    size_t CountY
    )
{
#if defined(MLAS_USE_SVE)
    if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArmSve()) {
        MlasHgemmCopyPackB_sve(reinterpret_cast<_mlas_fp16_*>(D),
                               reinterpret_cast<const _mlas_fp16_*>(B),
                               ldb, CountX, CountY);
        return;
    }
#endif
    // Scalar fp16 fallback copy-pack: CountX columns x CountY rows of
    // row-major B (row stride ldb) into PACKED_B_BLOCK_WIDTH_FP16-wide blocks,
    // zero-padding the tail of each block.
    auto* d = reinterpret_cast<_mlas_fp16_*>(D);
    const auto* b = reinterpret_cast<const _mlas_fp16_*>(B);
    for (size_t x0 = 0; x0 < CountX; x0 += PACKED_B_BLOCK_WIDTH_FP16) {
        const size_t block = std::min((size_t)PACKED_B_BLOCK_WIDTH_FP16, CountX - x0);
        for (size_t y = 0; y < CountY; ++y) {
            _mlas_fp16_* dd = d + y * PACKED_B_BLOCK_WIDTH_FP16;
            for (size_t j = 0; j < block; ++j) {
                dd[j] = b[y * ldb + x0 + j];
            }
            for (size_t j = block; j < PACKED_B_BLOCK_WIDTH_FP16; ++j) {
                dd[j] = (_mlas_fp16_)0;
            }
        }
        d += (size_t)PACKED_B_BLOCK_WIDTH_FP16 * CountY;
    }
}


// ============================================================================
//  B packing, Trans   (analogue of MlasSgemmTransposePackB)
//  Wrapper selecting SVE vs scalar fallback.
// ============================================================================
static void
MlasHgemmTransposePackB(
    MLAS_HGEMM_FP16* D,
    const MLAS_HGEMM_FP16* B,
    size_t ldb,
    size_t CountY,   // CountN
    size_t CountX    // CountK
    )
{
#if defined(MLAS_USE_SVE)
    if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArmSve()) {
        MlasHgemmTransposePackB_sve(reinterpret_cast<_mlas_fp16_*>(D),
                                    reinterpret_cast<const _mlas_fp16_*>(B),
                                    ldb, CountY, CountX);
        return;
    }
#endif
    // Scalar fp16 fallback transpose-pack. B is N x K row-major (row stride
    // ldb): logical element (k, n) == B[n*ldb + k].
    auto* d = reinterpret_cast<_mlas_fp16_*>(D);
    const auto* b = reinterpret_cast<const _mlas_fp16_*>(B);
    const size_t CountN = CountY;
    const size_t CountK = CountX;
    for (size_t n0 = 0; n0 < CountN; n0 += PACKED_B_BLOCK_WIDTH_FP16) {
        const size_t block = std::min((size_t)PACKED_B_BLOCK_WIDTH_FP16, CountN - n0);
        for (size_t k = 0; k < CountK; ++k) {
            _mlas_fp16_* dd = d + k * PACKED_B_BLOCK_WIDTH_FP16;
            for (size_t j = 0; j < block; ++j) {
                dd[j] = b[(n0 + j) * ldb + k];
            }
            for (size_t j = block; j < PACKED_B_BLOCK_WIDTH_FP16; ++j) {
                dd[j] = (_mlas_fp16_)0;
            }
        }
        d += (size_t)PACKED_B_BLOCK_WIDTH_FP16 * CountK;
    }
}


// ============================================================================
//  Kernel row driver   (analogue of MlasSgemmKernelLoop)
//  Steps through M rows, calling the SVE Zero/Add kernels, which return the
//  number of rows handled so we can advance A and C.
// ============================================================================
static MLAS_HGEMM_FP16*
MlasHgemmKernelLoop(
    const MLAS_HGEMM_FP16* A,
    const MLAS_HGEMM_FP16* B,
    MLAS_HGEMM_FP16* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha,
    bool ZeroMode
    )
{
    const auto* a = reinterpret_cast<const _mlas_fp16_*>(A);
    const auto* b = reinterpret_cast<const _mlas_fp16_*>(B);
    auto* c = reinterpret_cast<_mlas_fp16_*>(C);

    while (CountM > 0) {
        size_t RowsHandled = 0;

#if defined(MLAS_USE_SVE)
        if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArmSve()) {
            if (ZeroMode) {
                RowsHandled = MlasHgemmKernelZero_sve(a, b, c, CountK, CountM, CountN, lda, ldc, alpha);
            } else {
                RowsHandled = MlasHgemmKernelAdd_sve(a, b, c, CountK, CountM, CountN, lda, ldc, alpha);
            }
        } else
#endif
        {
            // Unreachable in practice: MlasHGemmSupported() returns false
            // without SVE, so the EP layer never routes here. Without
            // MLAS_USE_SVE the block above is preprocessed away and these
            // would otherwise trip -Werror=unused-parameter.
            MLAS_UNREFERENCED_PARAMETER(b);
            MLAS_UNREFERENCED_PARAMETER(CountK);
            MLAS_UNREFERENCED_PARAMETER(CountN);
            MLAS_UNREFERENCED_PARAMETER(alpha);
            MLAS_UNREFERENCED_PARAMETER(ZeroMode);
            MLAS_THROW_EX(std::runtime_error, "HGEMM requires SVE in this build");
        }

        c += ldc * RowsHandled;
        a += lda * RowsHandled;
        CountM -= RowsHandled;
    }

    return reinterpret_cast<MLAS_HGEMM_FP16*>(c);
}


// ============================================================================
//  Core operation   (analogue of MlasSgemmOperation)
//  Owns: M=1 gemv fast path, strideN/strideK blocking, the TransB pack choice,
//  and the TransA upstream transpose.
// ============================================================================
static void
MlasHgemmOperation(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    const MLAS_HGEMM_FP16* A,
    size_t lda,
    const MLAS_HGEMM_FP16* B,
    size_t ldb,
    float beta,
    MLAS_HGEMM_FP16* C,
    size_t ldc
    )
{
    // --- K == 0: just apply beta and exit (mirror SGEMM) ---
    if (K == 0) {
        MlasHgemmMultiplyBeta(C, M, N, ldc, beta);
        return;
    }

    // --- M == 1 NoTrans/NoTrans fast path -> gemv ---
    if (M == 1 && TransA == CblasNoTrans && TransB == CblasNoTrans &&
        alpha == 1.0f && (beta == 0.0f || beta == 1.0f)) {
#if defined(MLAS_USE_SVE)
        if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArmSve()) {
            MlasHgemvFloat16Kernel_sve(
                reinterpret_cast<const _mlas_fp16_*>(A),
                reinterpret_cast<const _mlas_fp16_*>(B),
                reinterpret_cast<_mlas_fp16_*>(C),
                K, N, ldb, (beta == 0.0f));
            return;
        }
#endif
        // fall through to general path if gemv unavailable
    }

    //
    // --- N == 1 matrix-vector fast path ---
    //
    // The general path packs B into PACKED_B_BLOCK_WIDTH_FP16 (32) wide
    // blocks, so a K x 1 operand pays for a whole block: measured on
    // Graviton3, N = 1 through N = 16 all cost the same (~404 us at
    // M = K = 1023), i.e. 21x the per-column rate reached at N >= 32. Skip the
    // pack entirely for a single column.
    //
    // B must be contiguous: it is when TransB (B is 1 x K), and when NoTrans
    // with ldb == 1. SVE has no 16-bit gather, so a strided B stays on the
    // general path.
    //
    //
    // --- M == 1 with TransB: the same matrix-vector kernel, operands swapped ---
    //
    // With TransB, B is N x K row-major, so C[n] = dot(B[n, :], A) -- a
    // matrix-vector product with B as the matrix and A as the length-K vector.
    // MlasHgemvNKernel_sve already computes exactly that, so pass B where it
    // expects A (row stride ldb) and A where it expects the vector. Without
    // this the row-vector case falls into the packed path, which was measured
    // up to 1.9x slower than SGEMM at small N and K.
    //
    if (M == 1 && TransA == CblasNoTrans && TransB == CblasTrans &&
        (beta == 0.0f || beta == 1.0f)) {
#if defined(MLAS_USE_SVE)
        if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArmSve()) {
            MlasHgemvNKernel_sve(
                reinterpret_cast<const _mlas_fp16_*>(B),   // matrix: N x K, row stride ldb
                reinterpret_cast<const _mlas_fp16_*>(A),   // vector: length K, contiguous
                reinterpret_cast<_mlas_fp16_*>(C),
                N, K, ldb, 1, alpha, (beta == 0.0f));
            return;
        }
#endif
    }

    if (N == 1 && TransA == CblasNoTrans && (beta == 0.0f || beta == 1.0f)) {
        const bool b_contiguous = (TransB == CblasTrans) || (ldb == 1);
#if defined(MLAS_USE_SVE)
        if (b_contiguous && MLAS_CPUIDINFO::GetCPUIDInfo().HasArmSve()) {
            MlasHgemvNKernel_sve(
                reinterpret_cast<const _mlas_fp16_*>(A),
                reinterpret_cast<const _mlas_fp16_*>(B),
                reinterpret_cast<_mlas_fp16_*>(C),
                M, K, lda, ldc, alpha, (beta == 0.0f));
            return;
        }
#else
        MLAS_UNREFERENCED_PARAMETER(b_contiguous);
#endif
    }

    // --- general path: stride blocking + pack + kernel loop ---
    MLAS_HGEMM_FP16 PanelA[MLAS_HGEMM_TRANSA_ROWS * MLAS_HGEMM_STRIDEK];
    MLAS_DECLSPEC_ALIGN(MLAS_HGEMM_FP16 PanelB[MLAS_HGEMM_STRIDEN * MLAS_HGEMM_STRIDEK],
                        MLAS_HGEMM_STRIDEN_THREAD_ALIGN * sizeof(MLAS_HGEMM_FP16));

    // Expand the N stride if K is small, or expand the K stride if N is small,
    // for better utilization of the B panel. The StrideN*StrideK product is
    // invariant under these halve/double steps, so PanelB sizing is preserved.
    // Avoid growing StrideK when A must be transposed (TransA != NoTrans): the
    // K-expansion branch is gated on NoTrans because PanelA is sized for
    // MLAS_HGEMM_STRIDEK and is only used on the TransA path. Mirror of the
    // SGEMM heuristic in MlasSgemmOperation.
    size_t StrideN = MLAS_HGEMM_STRIDEN;
    size_t StrideK = MLAS_HGEMM_STRIDEK;

    if (N >= K) {
        while (StrideK / 2 >= K) {
            StrideN *= 2;
            StrideK /= 2;
        }
    } else if (TransA == CblasNoTrans) {
        while (StrideN > MLAS_HGEMM_STRIDEN_THREAD_ALIGN && StrideN / 2 >= N) {
            StrideK *= 2;
            StrideN /= 2;
        }
    }

    size_t CountN;
    for (size_t n = 0; n < N; n += CountN) {
        CountN = std::min(N - n, StrideN);

        // Apply arbitrary beta up front; the kernel then accumulates with
        // ZeroMode=false (beta becomes 1 after this scaling).
        bool ZeroMode = (beta == 0.0f);
        if (!ZeroMode && beta != 1.0f) {
            MlasHgemmMultiplyBeta(C + n, M, CountN, ldc, beta);
        }

        size_t CountK;
        for (size_t k = 0; k < K; k += CountK) {
            CountK = std::min(K - k, StrideK);

            // Pack a panel of B (copy if NoTrans, transpose if Trans).
            if (TransB == CblasNoTrans) {
                MlasHgemmCopyPackB(PanelB, B + n + k * ldb, ldb, CountN, CountK);
            } else {
                MlasHgemmTransposePackB(PanelB, B + k + n * ldb, ldb, CountN, CountK);
            }

            MLAS_HGEMM_FP16* c = C + n;

            if (TransA == CblasNoTrans) {
                MlasHgemmKernelLoop(A + k, PanelB, c, CountK, M, CountN, lda, ldc, alpha, ZeroMode);
            } else {
                const MLAS_HGEMM_FP16* a = A + k * lda;
                size_t RowsRemaining = M;
                MLAS_HGEMM_FP16* cc = c;
                while (RowsRemaining > 0) {
                    size_t RowsTransposed =
                        std::min(RowsRemaining, size_t(MLAS_HGEMM_TRANSA_ROWS));
                    // PanelA[row, kk] = A_stored[k + kk, RangeRow] -> NoTrans panel.
                    MlasHgemmTransposeA(PanelA, a, lda, RowsTransposed, CountK);
                    RowsRemaining -= RowsTransposed;
                    a += RowsTransposed;
                    cc = MlasHgemmKernelLoop(PanelA, PanelB, cc, CountK, RowsTransposed,
                                             CountN, CountK, ldc, alpha, ZeroMode);
                }
            }
            ZeroMode = false;
        }
    }
    (void)PanelA;
}


// ============================================================================
//  Public capability probe   (declared in mlas.h)
// ============================================================================
bool
MLASCALL
MlasHGemmSupported(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB
    )
{
    // Carried over from the MlasHGemmSupported() that used to live in
    // halfgemm.cpp: refuse FP16 GEMM outright when the CPU has no FP16 vector
    // acceleration, regardless of SVE.
    if (!MlasFp16AccelerationSupported()) {
        MLAS_UNREFERENCED_PARAMETER(TransA);
        MLAS_UNREFERENCED_PARAMETER(TransB);
        return false;
    }
#if defined(MLAS_USE_SVE)
    if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArmSve()) {
        // The SVE driver supports both NoTrans/Trans for A and B.
        return (TransA == CblasNoTrans || TransA == CblasTrans) &&
               (TransB == CblasNoTrans || TransB == CblasTrans);
    }
#endif
    MLAS_UNREFERENCED_PARAMETER(TransA);
    MLAS_UNREFERENCED_PARAMETER(TransB);
    return false;
}


// ============================================================================
//  Threaded segment entry   (analogue of MlasSgemmThreaded)
// ============================================================================
static void
MlasHgemmThreaded(
    const ptrdiff_t ThreadCountM,
    const ptrdiff_t ThreadCountN,
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const size_t M,
    const size_t N,
    const size_t K,
    const MLAS_HGEMM_DATA_PARAMS* DataParams,
    ptrdiff_t ThreadId
    )
{
    const ptrdiff_t ThreadIdM = ThreadId / ThreadCountN;
    const ptrdiff_t ThreadIdN = ThreadId % ThreadCountN;

    size_t RangeStartM, RangeCountM;
    MlasPartitionWork(ThreadIdM, ThreadCountM, M, &RangeStartM, &RangeCountM);

    size_t RangeStartN, RangeCountN;
    const size_t BlockedN =
        (N + MLAS_HGEMM_STRIDEN_THREAD_ALIGN - 1) / MLAS_HGEMM_STRIDEN_THREAD_ALIGN;
    MlasPartitionWork(ThreadIdN, ThreadCountN, BlockedN, &RangeStartN, &RangeCountN);
    RangeStartN *= MLAS_HGEMM_STRIDEN_THREAD_ALIGN;
    RangeCountN *= MLAS_HGEMM_STRIDEN_THREAD_ALIGN;
    RangeCountN = std::min(N - RangeStartN, RangeCountN);

    if (RangeCountM == 0 || RangeCountN == 0) {
        return;
    }

    const size_t lda = DataParams->lda;
    const size_t ldb = DataParams->ldb;
    const size_t ldc = DataParams->ldc;
    const float alpha = MLAS_Half2Float(DataParams->alpha);
    const float beta = MLAS_Half2Float(DataParams->beta);

    const MLAS_HGEMM_FP16* A =
        DataParams->A + RangeStartM * ((TransA == CblasNoTrans) ? lda : 1);
    const MLAS_HGEMM_FP16* B =
        DataParams->B + RangeStartN * ((TransB == CblasNoTrans) ? 1 : ldb);
    MLAS_HGEMM_FP16* C = DataParams->C + RangeStartM * ldc + RangeStartN;

    MlasHgemmOperation(TransA, TransB, RangeCountM, RangeCountN, K,
                       alpha, A, lda, B, ldb, beta, C, ldc);
}


// ============================================================================
//  Public batch entry   (declared in mlas.h as the MLAS_HGEMM_DATA_PARAMS
//  overload of MlasGemmBatch)
// ============================================================================
void
MLASCALL
MlasGemmBatch(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    const MLAS_HGEMM_DATA_PARAMS* Data,
    size_t BatchSize,
    MLAS_THREADPOOL* ThreadPool
    )
{
    const double Complexity = double(M) * double(N) * double(K) * double(BatchSize);
    ptrdiff_t TargetThreadCount =
        ptrdiff_t(Complexity / double(MLAS_HGEMM_THREAD_COMPLEXITY)) + 1;
    ptrdiff_t MaximumThreadCount = MlasGetMaximumThreadCount(ThreadPool);
    if (TargetThreadCount >= MaximumThreadCount) {
        TargetThreadCount = MaximumThreadCount;
    }

    ptrdiff_t ThreadsPerGemm = (TargetThreadCount + ptrdiff_t(BatchSize) - 1) / ptrdiff_t(BatchSize);
    if (ThreadsPerGemm < 1) {
        ThreadsPerGemm = 1;
    }

    ptrdiff_t ThreadCountM, ThreadCountN;
    if (N > M) {
        const size_t BlockedN =
            (N + MLAS_HGEMM_STRIDEN_THREAD_ALIGN - 1) / MLAS_HGEMM_STRIDEN_THREAD_ALIGN;
        if (size_t(ThreadsPerGemm) > BlockedN) {
            ThreadsPerGemm = ptrdiff_t(BlockedN);
        }
        ThreadCountM = 1;
        ThreadCountN = ThreadsPerGemm;
    } else {
        if (size_t(ThreadsPerGemm) > M) {
            ThreadsPerGemm = ptrdiff_t(M);
        }
        ThreadCountM = ThreadsPerGemm;
        ThreadCountN = 1;
    }
    if (ThreadsPerGemm < 1) {
        ThreadsPerGemm = 1;
        ThreadCountM = 1;
        ThreadCountN = 1;
    }

    MlasTrySimpleParallel(
        ThreadPool, ThreadsPerGemm * static_cast<ptrdiff_t>(BatchSize),
        [=](ptrdiff_t tid) {
            ptrdiff_t GemmIdx = tid / ThreadsPerGemm;
            ptrdiff_t ThreadIdx = tid % ThreadsPerGemm;
            MlasHgemmThreaded(ThreadCountM, ThreadCountN, TransA, TransB, M, N, K,
                              &(Data[GemmIdx]), ThreadIdx);
        });
}
