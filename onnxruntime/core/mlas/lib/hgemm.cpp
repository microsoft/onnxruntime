/*++

Copyright (c) Microsoft Corporation. All rights reserved.
Copyright 2025 FUJITSU LIMITED

Licensed under the MIT License.

Module Name:

    hgemm.cpp

Abstract:

    This module implements the half precision floating point matrix/matrix
    multiply operation (HGEMM) on top of the SVE kernels.

--*/

#include "mlasi.h"
#include "mlas_float16.h"

#include <algorithm>
#include <exception>

using MLAS_HGEMM_FP16 = MLAS_FP16;

#if defined(MLAS_USE_SVE)

#include "sve/halfgemm_sve.h"
#include "sve/halfgemv_sve.h"

static_assert(PACKED_B_BLOCK_WIDTH_FP16 == MLAS_HGEMM_STRIDEN_THREAD_ALIGN,
              "packed-B block width must match the driver's N thread alignment");

#endif // MLAS_USE_SVE

//
// Define the number of rows from matrix A to transpose to a local buffer.
//
#define MLAS_HGEMM_TRANSA_ROWS              12

#ifndef PACKED_B_BLOCK_WIDTH_FP16
#define PACKED_B_BLOCK_WIDTH_FP16           MLAS_HGEMM_STRIDEN_THREAD_ALIGN
#endif  // set by sve/halfgemm_sve.h when MLAS_USE_SVE


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
    for (size_t y = 0; y < CountY; ++y) {
        for (size_t x = 0; x < CountX; ++x) {
            d[y * CountX + x] = a[x * lda + y];
        }
    }
}


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


static void
MlasHgemmTransposePackB(
    MLAS_HGEMM_FP16* D,
    const MLAS_HGEMM_FP16* B,
    size_t ldb,
    size_t CountY,
    size_t CountX
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
            // Not reached, MlasHGemmSupported() is false without SVE.
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
    if (K == 0) {
        MlasHgemmMultiplyBeta(C, M, N, ldc, beta);
        return;
    }

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
    }

    if (M == 2 && TransA == CblasNoTrans && TransB == CblasNoTrans &&
        alpha == 1.0f && (beta == 0.0f || beta == 1.0f)) {
#if defined(MLAS_USE_SVE)
        if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArmSve()) {
            MlasHgemv2Float16Kernel_sve(
                reinterpret_cast<const _mlas_fp16_*>(A), lda,
                reinterpret_cast<const _mlas_fp16_*>(B),
                reinterpret_cast<_mlas_fp16_*>(C), ldc,
                K, N, ldb, (beta == 0.0f));
            return;
        }
#endif
    }

    // With TransB, C[n] = dot(B[n, :], A), so B is passed as the matrix and A as the vector.
    if (M == 1 && TransA == CblasNoTrans && TransB == CblasTrans &&
        (beta == 0.0f || beta == 1.0f)) {
#if defined(MLAS_USE_SVE)
        if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArmSve()) {
            MlasHgemvNKernel_sve(
                reinterpret_cast<const _mlas_fp16_*>(B),
                reinterpret_cast<const _mlas_fp16_*>(A),
                reinterpret_cast<_mlas_fp16_*>(C),
                N, K, ldb, 1, alpha, (beta == 0.0f));
            return;
        }
#endif
    }

    // SVE has no 16-bit gather, so B has to be contiguous.
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

    MLAS_HGEMM_FP16 PanelA[MLAS_HGEMM_TRANSA_ROWS * MLAS_HGEMM_STRIDEK];
    MLAS_DECLSPEC_ALIGN(MLAS_HGEMM_FP16 PanelB[MLAS_HGEMM_STRIDEN * MLAS_HGEMM_STRIDEK],
                        MLAS_HGEMM_STRIDEN_THREAD_ALIGN * sizeof(MLAS_HGEMM_FP16));

    // Expand the N stride if K is small or expand the K stride if N is small
    // for better utilization of the B panel. Avoid changing the K stride if
    // the A panel needs to be used for transposing.
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

        bool ZeroMode = (beta == 0.0f);
        if (!ZeroMode && beta != 1.0f) {
            MlasHgemmMultiplyBeta(C + n, M, CountN, ldc, beta);
        }

        size_t CountK;
        for (size_t k = 0; k < K; k += CountK) {
            CountK = std::min(K - k, StrideK);

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


//
// PackedB holds, for each K slice, AlignedN columns in the MlasHgemmCopyPackB
// layout, so the panel for columns [n, n + CountN) of slice k starts at
// PackedB + AlignedN * k + CountK * n.
//
#define MLAS_HGEMM_PACKED_STRIDEN           MLAS_HGEMM_STRIDEN
#define MLAS_HGEMM_PACKED_STRIDEK           MLAS_HGEMM_STRIDEK

static void
MlasHgemmPackedOperation(
    CBLAS_TRANSPOSE TransA,
    size_t M,
    size_t RangeStartN,
    size_t RangeCountN,
    size_t K,
    float alpha,
    const MLAS_HGEMM_FP16* A,
    size_t lda,
    const MLAS_HGEMM_FP16* PackedB,
    size_t AlignedN,
    float beta,
    MLAS_HGEMM_FP16* C,
    size_t ldc
    )
{
    if (K == 0) {
        MlasHgemmMultiplyBeta(C, M, RangeCountN, ldc, beta);
        return;
    }

    MLAS_HGEMM_FP16 PanelA[MLAS_HGEMM_TRANSA_ROWS * MLAS_HGEMM_PACKED_STRIDEK];

    size_t CountN;
    for (size_t n = 0; n < RangeCountN; n += CountN) {
        const size_t SliceStartN = RangeStartN + n;
        CountN = std::min(RangeCountN - n, size_t(MLAS_HGEMM_PACKED_STRIDEN));

        bool ZeroMode = (beta == 0.0f);
        if (!ZeroMode && beta != 1.0f) {
            MlasHgemmMultiplyBeta(C + n, M, CountN, ldc, beta);
        }

        size_t CountK;
        for (size_t k = 0; k < K; k += CountK) {
            CountK = std::min(K - k, size_t(MLAS_HGEMM_PACKED_STRIDEK));

            const MLAS_HGEMM_FP16* pb = PackedB + AlignedN * k + CountK * SliceStartN;
            MLAS_HGEMM_FP16* c = C + n;

            if (TransA == CblasNoTrans) {
                MlasHgemmKernelLoop(A + k, pb, c, CountK, M, CountN, lda, ldc, alpha, ZeroMode);
            } else {
                const MLAS_HGEMM_FP16* a = A + k * lda;
                size_t RowsRemaining = M;
                while (RowsRemaining > 0) {
                    size_t RowsTransposed =
                        std::min(RowsRemaining, size_t(MLAS_HGEMM_TRANSA_ROWS));
                    MlasHgemmTransposeA(PanelA, a, lda, RowsTransposed, CountK);
                    RowsRemaining -= RowsTransposed;
                    a += RowsTransposed;
                    c = MlasHgemmKernelLoop(PanelA, pb, c, CountK, RowsTransposed,
                                            CountN, CountK, ldc, alpha, ZeroMode);
                }
            }
            ZeroMode = false;
        }
    }
    (void)PanelA;
}


bool
MLASCALL
MlasHGemmSupported(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB
    )
{
    if (!MlasFp16AccelerationSupported()) {
        MLAS_UNREFERENCED_PARAMETER(TransA);
        MLAS_UNREFERENCED_PARAMETER(TransB);
        return false;
    }
#if defined(MLAS_USE_SVE)
    if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArmSve()) {
        return (TransA == CblasNoTrans || TransA == CblasTrans) &&
               (TransB == CblasNoTrans || TransB == CblasTrans);
    }
#endif
    MLAS_UNREFERENCED_PARAMETER(TransA);
    MLAS_UNREFERENCED_PARAMETER(TransB);
    return false;
}


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
    MLAS_HGEMM_FP16* C = DataParams->C + RangeStartM * ldc + RangeStartN;

    if (DataParams->BIsPacked) {
        const size_t AlignedN =
            (N + MLAS_HGEMM_STRIDEN_THREAD_ALIGN - 1) & ~size_t(MLAS_HGEMM_STRIDEN_THREAD_ALIGN - 1);
        MlasHgemmPackedOperation(TransA, RangeCountM, RangeStartN, RangeCountN, K,
                                 alpha, A, lda, DataParams->B, AlignedN, beta, C, ldc);
        return;
    }

    const MLAS_HGEMM_FP16* B =
        DataParams->B + RangeStartN * ((TransB == CblasNoTrans) ? 1 : ldb);

    MlasHgemmOperation(TransA, TransB, RangeCountM, RangeCountN, K,
                       alpha, A, lda, B, ldb, beta, C, ldc);
}


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


size_t
MLASCALL
MlasHGemmPackBSize(
    CBLAS_TRANSPOSE TransB,
    size_t N,
    size_t K
    )
{
    if (!MlasHGemmSupported(CblasNoTrans, TransB)) {
        return 0;
    }

    const size_t AlignedN =
        (N + MLAS_HGEMM_STRIDEN_THREAD_ALIGN - 1) & ~size_t(MLAS_HGEMM_STRIDEN_THREAD_ALIGN - 1);
    const size_t BytesRequired = AlignedN * K * sizeof(MLAS_HGEMM_FP16);
    const size_t BufferAlignment = MlasGetPreferredBufferAlignment();
    return (BytesRequired + BufferAlignment - 1) & ~(BufferAlignment - 1);
}

void
MLASCALL
MlasHGemmPackB(
    CBLAS_TRANSPOSE TransB,
    size_t N,
    size_t K,
    const MLAS_FP16* B,
    size_t ldb,
    void* PackedB
    )
{
    const size_t AlignedN =
        (N + MLAS_HGEMM_STRIDEN_THREAD_ALIGN - 1) & ~size_t(MLAS_HGEMM_STRIDEN_THREAD_ALIGN - 1);
    auto* D = static_cast<MLAS_HGEMM_FP16*>(PackedB);

    size_t CountK;
    for (size_t k = 0; k < K; k += CountK) {
        CountK = std::min(K - k, size_t(MLAS_HGEMM_PACKED_STRIDEK));
        if (TransB == CblasNoTrans) {
            MlasHgemmCopyPackB(D, B + k * ldb, ldb, N, CountK);
        } else {
            MlasHgemmTransposePackB(D, B + k, ldb, N, CountK);
        }
        D += AlignedN * CountK;
    }
}
