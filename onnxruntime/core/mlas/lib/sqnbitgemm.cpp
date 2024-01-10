/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm.cpp

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication hardware agnostic entrypoint, MlasSQNBitGemmBatch,
    as well as some SQNBitGemm-related query functions.
--*/

#include "sqnbitgemm.h"

#include <cassert>

#ifdef MLAS_JBLAS
#include "jblas_gemm.h"
#endif

namespace
{

enum SQNBitGemmVariant {
    SQNBitGemmVariantInvalid = -1,

    // Valid variants

    SQNBitGemmVariant_BitWidth4_CompFp32 = 0,
    SQNBitGemmVariant_BitWidth4_CompInt8,

    // End of valid variants

    // Keep this element last and ensure that its value is the number of valid SQNBitGemmVariant values.
    // Its value is used as an array size.
    SQNBitGemmVariantCount,
};

bool
IsSupportedBlkLen(size_t BlkBitWidth, size_t BlkLen)
{
    if (BlkBitWidth == 4) {
        return BlkLen == 16 ||
               BlkLen == 32 ||
               BlkLen == 64 ||
               BlkLen == 128 ||
               BlkLen == 256;
    }
    return false;
}

SQNBitGemmVariant
GetSQNBitGemmVariant(
    size_t M,
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    MLAS_UNREFERENCED_PARAMETER(N);
    MLAS_UNREFERENCED_PARAMETER(K);

    if (BlkBitWidth == 4 && IsSupportedBlkLen(BlkBitWidth, BlkLen)) {
        if (ComputeType == CompFp32 ||
            ComputeType == CompUndef) {  // treat CompUndef (undefined) as CompFp32
            return SQNBitGemmVariant_BitWidth4_CompFp32;
        } else if (ComputeType == CompInt8 && M == 1) {
            return SQNBitGemmVariant_BitWidth4_CompInt8;
        }
    }

    return SQNBitGemmVariantInvalid;
}

}  // namespace

bool MLASCALL
MlasIsSQNBitGemmAvailable(
    size_t M,
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    const auto* Dispatch = GetMlasPlatform().SQNBitGemmDispatch;
    if (Dispatch == nullptr) {
        return false;
    }

    const auto Variant = GetSQNBitGemmVariant(M, N, K, BlkBitWidth, BlkLen, ComputeType);

    switch (Variant) {
        case SQNBitGemmVariant_BitWidth4_CompFp32: {
            return Dispatch->SQNBitGemmM1Kernel_BlkBitWidth4_CompFp32 != nullptr &&
                   Dispatch->QNBitBlkDequantBForSgemm_BlkBitWidth4_CompFp32 != nullptr;
        }
        case SQNBitGemmVariant_BitWidth4_CompInt8: {
            return Dispatch->SQNBitGemmM1Kernel_BlkBitWidth4_CompInt8 != nullptr &&
                   Dispatch->QuantizeARow_CompInt8 != nullptr;
        }
        default: {
            return false;
        }
    }
}

namespace
{

size_t
SQNBitGemmPerGemmWorkspaceSize(
    SQNBitGemmVariant Variant,
    size_t M,
    size_t N,
    size_t K,
    size_t BlkLen
)
{
    MLAS_UNREFERENCED_PARAMETER(N);

    switch (Variant) {
        case SQNBitGemmVariant_BitWidth4_CompInt8: {
            // workspace buffer is used for block quantization of A to int8
            const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
            const size_t PerGemmWorkspaceSize = M * BlockCountK * Q8BlkSize(BlkLen);
            return PerGemmWorkspaceSize;
        }
        default: {
            return 0;
        }
    }
}

size_t
SQNBitGemmPerGemmWorkspaceStride(
    SQNBitGemmVariant Variant,
    size_t M,
    size_t N,
    size_t K,
    size_t BlkLen
)
{
    const auto Size = SQNBitGemmPerGemmWorkspaceSize(Variant, M, N, K, BlkLen);
    return Size;
}

}  // namespace

size_t MLASCALL
MlasSQNBitGemmBatchWorkspaceSize(
    size_t M,
    size_t N,
    size_t K,
    size_t BatchN,
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    const auto Variant = GetSQNBitGemmVariant(M, N, K, BlkBitWidth, BlkLen, ComputeType);

    const size_t PerGemmWorkspaceStride = SQNBitGemmPerGemmWorkspaceStride(Variant, M, N, K, BlkLen);
    if (PerGemmWorkspaceStride == 0) {
        return 0;
    }

    const size_t WorkspaceSize = BatchN * PerGemmWorkspaceStride;

    return WorkspaceSize;
}

namespace
{

typedef void(SQNBitGemmFn)(
    size_t BlkLen,
    size_t K,
    const MLAS_SQNBIT_GEMM_DATA_PARAMS* DataParams,
    void* PerGemmWorkspace,
    size_t RangeStartM,
    size_t RangeCountM,
    size_t RangeStartN,
    size_t RangeCountN
);

void
SQNBitGemm_BlkBitWidth4_CompFp32(
    const size_t BlkLen,
    const size_t K,
    const MLAS_SQNBIT_GEMM_DATA_PARAMS* const DataParams,
    void* const PerGemmWorkspace,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
)
{
    MLAS_UNREFERENCED_PARAMETER(PerGemmWorkspace);

    const size_t lda = DataParams->lda;
    const size_t ldc = DataParams->ldc;

    const size_t k_blks = MlasDivRoundup(K, BlkLen);
    const size_t ldb = k_blks * Q4BlkSize(BlkLen);

    const float* A = DataParams->A + RangeStartM * lda;

    const std::byte* QuantB = static_cast<const std::byte*>(DataParams->QuantB) + RangeStartN * ldb;

    float* C = DataParams->C + RangeStartM * ldc + RangeStartN;

    if (RangeCountM == 1) {
        size_t CountN;
        for (size_t n = 0; n < RangeCountN; n += CountN) {
            CountN = std::min(RangeCountN - n, size_t{128});

            const float* a_row = A;
            const std::byte* b_col = QuantB + n * ldb;
            float* c_blk = C + n;

            GetMlasPlatform().SQNBitGemmDispatch->SQNBitGemmM1Kernel_BlkBitWidth4_CompFp32(
                BlkLen,
                a_row, b_col, c_blk, CountN, K, k_blks
            );
        }
        return;
    }

    constexpr size_t StrideN = 32;
    size_t bufsize = k_blks * BlkLen * StrideN * sizeof(float);
    MlasThreadedBufAlloc(bufsize);
    auto* dequant_b = reinterpret_cast<float*>(ThreadedBufHolder.get());

    //
    // Step through each slice of matrix B along the N dimension.
    //
    size_t CountN;
    for (size_t n = 0; n < RangeCountN; n += CountN) {
        CountN = std::min(RangeCountN - n, StrideN);

        //
        // Step through each slice of matrix A along the M dimension.
        //
        const float* a_row = A;
        const std::byte* b_col = QuantB + n * ldb;
        float* c_blk = C + n;

        GetMlasPlatform().SQNBitGemmDispatch->QNBitBlkDequantBForSgemm_BlkBitWidth4_CompFp32(
            BlkLen,
            dequant_b, b_col, CountN, K, k_blks
        );

        size_t RowsRemaining = RangeCountM;
        while (RowsRemaining > 0) {
#if defined(MLAS_TARGET_AMD64_IX86) || defined(MLAS_TARGET_POWER)
            auto RowsHandled = GetMlasPlatform().GemmFloatKernel(
                a_row, dequant_b, c_blk, K, RowsRemaining, CountN, lda, ldc, 1.f, true
            );
#else
            auto RowsHandled = MlasSgemmKernelZero(a_row, dequant_b, c_blk, K, RowsRemaining, CountN, lda, ldc, 1.f);
#endif

            c_blk += ldc * RowsHandled;
            a_row += lda * RowsHandled;
            RowsRemaining -= RowsHandled;
        }
    }
}

void
SQNBitGemm_BlkBitWidth4_CompInt8(
    const size_t BlkLen,
    const size_t K,
    const MLAS_SQNBIT_GEMM_DATA_PARAMS* const DataParams,
    void* const PerGemmWorkspace,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
)
{
    const size_t k_blks = MlasDivRoundup(K, BlkLen);

    const size_t lda = k_blks * Q8BlkSize(BlkLen);
    const size_t ldc = DataParams->ldc;
    const size_t ldb = k_blks * Q4BlkSize(BlkLen);

    const std::byte* QuantA = static_cast<const std::byte*>(PerGemmWorkspace) + RangeStartM * lda;

    const std::byte* QuantB = static_cast<const std::byte*>(DataParams->QuantB) + RangeStartN * ldb;

    float* C = DataParams->C + RangeStartM * ldc + RangeStartN;

    if (RangeCountM == 1) {
        size_t CountN;
        for (size_t n = 0; n < RangeCountN; n += CountN) {
            CountN = std::min(RangeCountN - n, size_t{128});

            const std::byte* a_row = QuantA;
            const std::byte* b_col = QuantB + n * ldb;
            float* c_blk = C + n;

            GetMlasPlatform().SQNBitGemmDispatch->SQNBitGemmM1Kernel_BlkBitWidth4_CompInt8(
                BlkLen,
                a_row, b_col, c_blk, CountN, K, k_blks
            );
        }
        return;
    }

    assert(false && "not implemented for M > 1");
}

typedef void(InitializeWorkspaceFn)(
    size_t M,
    size_t N,
    size_t K,
    size_t BatchN,
    size_t BlkLen,
    const MLAS_SQNBIT_GEMM_DATA_PARAMS* DataParams,
    void* Workspace,
    size_t PerGemmWorkspaceStride,
    MLAS_THREADPOOL* ThreadPool
);

void
InitializeWorkspace_CompInt8(
    size_t M,
    size_t N,
    size_t K,
    size_t BatchN,
    size_t BlkLen,
    const MLAS_SQNBIT_GEMM_DATA_PARAMS* DataParams,
    void* Workspace,
    size_t PerGemmWorkspaceStride,
    MLAS_THREADPOOL* ThreadPool
)
{
    MLAS_UNREFERENCED_PARAMETER(N);

    const auto QuantizeARow = GetMlasPlatform().SQNBitGemmDispatch->QuantizeARow_CompInt8;

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t QuantAStride = BlockCountK * Q8BlkSize(BlkLen);

    MlasTrySimpleParallel(ThreadPool, BatchN, [&](ptrdiff_t gemm_idx) {
        const auto& data = DataParams[gemm_idx];

        const float* ARowPtr = data.A;
        std::byte* QuantARowPtr = static_cast<std::byte*>(Workspace) + gemm_idx * PerGemmWorkspaceStride;

        for (size_t m = 0; m < M; ++m) {
            QuantizeARow(BlkLen, ARowPtr, K, QuantARowPtr);

            ARowPtr += data.lda;
            QuantARowPtr += QuantAStride;
        }
    });
}

struct Operations {
    InitializeWorkspaceFn* InitializeWorkspace = nullptr;
    SQNBitGemmFn* SQNBitGemm = nullptr;
};

constexpr auto OperationMap = []() {
    std::array<Operations, SQNBitGemmVariantCount> ops;

    ops[SQNBitGemmVariant_BitWidth4_CompFp32].SQNBitGemm = SQNBitGemm_BlkBitWidth4_CompFp32;

    ops[SQNBitGemmVariant_BitWidth4_CompInt8].InitializeWorkspace = InitializeWorkspace_CompInt8;
    ops[SQNBitGemmVariant_BitWidth4_CompInt8].SQNBitGemm = SQNBitGemm_BlkBitWidth4_CompInt8;

    return ops;
}();

}  // namespace

void MLASCALL
MlasSQNBitGemmBatch(
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t BatchN,
    const size_t BlkBitWidth,
    const size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const MLAS_SQNBIT_GEMM_DATA_PARAMS* DataParams,
    void* Workspace,
    MLAS_THREADPOOL* ThreadPool
)
{
    const auto Variant = GetSQNBitGemmVariant(M, N, K, BlkBitWidth, BlkLen, ComputeType);
    assert(Variant != SQNBitGemmVariantInvalid);

    const size_t PerGemmWorkspaceStride = SQNBitGemmPerGemmWorkspaceStride(Variant, M, N, K, BlkLen);

    if (const auto InitializeWorkspaceOperation = OperationMap[Variant].InitializeWorkspace;
        InitializeWorkspaceOperation != nullptr) {
        InitializeWorkspaceOperation(
            M, N, K, BatchN, BlkLen, DataParams, Workspace, PerGemmWorkspaceStride, ThreadPool
        );
    }

    const auto ComputeOperation = OperationMap[Variant].SQNBitGemm;

    if (ThreadPool == nullptr) {
        for (size_t gemm_i = 0; gemm_i < BatchN; gemm_i++) {
            const auto* Data = &DataParams[gemm_i];
            void* PerGemmWorkspace =
                reinterpret_cast<std::byte*>(Workspace) + gemm_i * PerGemmWorkspaceStride;
            ComputeOperation(BlkLen, K, Data, PerGemmWorkspace, 0, M, 0, N);
        }
        return;
    }

    //
    // Compute the number of target threads given the complexity of the SGEMM
    // operation. Small requests should run using the single threaded path.
    //

    const double Complexity = double(M) * double(N) * double(K) * double(BatchN);

    ptrdiff_t TargetThreadCount = ptrdiff_t(Complexity / double(MLAS_QGEMM_THREAD_COMPLEXITY)) + 1;

    ptrdiff_t MaximumThreadCount = MlasGetMaximumThreadCount(ThreadPool) * 8;

    if (TargetThreadCount >= MaximumThreadCount) {
        TargetThreadCount = MaximumThreadCount;
    }

    ptrdiff_t ThreadsPerGemm = TargetThreadCount / BatchN;
    if (ThreadsPerGemm < 1) {
        ThreadsPerGemm = 1;
    }

    constexpr size_t StrideM = 128;

    size_t nc = N;
    if (ThreadsPerGemm > 1) {
        // more than one thread per GEMM

        const size_t BlockedM = MlasDivRoundup(M, StrideM);
        const size_t max_nc = MlasDivRoundup(N * BlockedM, ThreadsPerGemm);
        if (max_nc < nc) {
            nc = std::min(
                nc, MlasDivRoundup(max_nc, MLAS_QGEMM_STRIDEN_THREAD_ALIGN) *
                        MLAS_QGEMM_STRIDEN_THREAD_ALIGN
            );
        }
    }
    const size_t StrideN = nc;

    const size_t ThreadCountM = MlasDivRoundup(M, StrideM);
    const size_t ThreadCountN = MlasDivRoundup(N, StrideN);
    ThreadsPerGemm = ThreadCountM * ThreadCountN;

    MlasTrySimpleParallel(ThreadPool, ThreadsPerGemm * BatchN, [&](ptrdiff_t tid) {
        const auto gemm_i = tid / ThreadsPerGemm;
        const auto blk_i = tid % ThreadsPerGemm;
        const auto* Data = &DataParams[gemm_i];
        void* PerGemmWorkspace = reinterpret_cast<void*>(
            reinterpret_cast<std::byte*>(Workspace) + gemm_i * PerGemmWorkspaceStride
        );

        const ptrdiff_t ThreadIdN = blk_i / ThreadCountM;
        const ptrdiff_t ThreadIdM = blk_i % ThreadCountM;

        const size_t RangeStartM = ThreadIdM * StrideM;
        const size_t RangeCountM = std::min(M - RangeStartM, (size_t)StrideM);

        const size_t RangeStartN = ThreadIdN * StrideN;
        const size_t RangeCountN = std::min(N - RangeStartN, (size_t)StrideN);

        ComputeOperation(BlkLen, K, Data, PerGemmWorkspace, RangeStartM, RangeCountM, RangeStartN, RangeCountN);
    });
}

namespace
{
void
SQNBitGemmPackBData_BlkBitWidth4(
    size_t N,
    size_t BlockCountK,
    size_t BlkLen,
    const std::byte* QuantBData,
    std::byte* PackedQuantB
)
{
    assert(BlkLen % 16 == 0);

    for (size_t n = 0; n < N; ++n) {
        for (size_t k_blk = 0; k_blk < BlockCountK; ++k_blk) {
            std::byte* BlkData = Q4BlkMutableData(PackedQuantB);

            //
            // Pack 16 4-bit values (8 bytes) at a time like this:
            //
            // src: | v0 v1 | v2 v3 | v4 v5 | v6 v7 | v8 v9 | vA vB | vC vD | vE vF |
            //   =>
            // dst: | v0 v8 | v1 v9 | v2 vA | v3 vB | v4 vC | v5 vD | v6 vE | v7 vF |
            //
            for (size_t kk = 0; kk < BlkLen; kk += 16) {
                for (size_t byte_i = 0; byte_i < 4; ++byte_i) {
                    const std::byte src0 = QuantBData[byte_i];
                    const std::byte src1 = QuantBData[byte_i + 4];

                    std::byte& dst0 = BlkData[2 * byte_i];
                    std::byte& dst1 = BlkData[2 * byte_i + 1];

                    dst0 = (src0 & std::byte{0x0F}) | ((src1 & std::byte{0x0F}) << 4);
                    dst1 = (src0 >> 4) | ((src1 >> 4) << 4);
                }

                QuantBData += 8;
                BlkData += 8;
            }

            PackedQuantB += Q4BlkSize(BlkLen);
        }
    }
}

void
SQNBitGemmPackBScale_BlkBitWidth4(
    size_t N,
    size_t BlockCountK,
    size_t BlkLen,
    const float* QuantBScale,
    std::byte* PackedQuantB
)
{
    for (size_t n = 0; n < N; ++n) {
        for (size_t k_blk = 0; k_blk < BlockCountK; ++k_blk) {
            const float Scale = *QuantBScale;

            Q4BlkSetScale(PackedQuantB, Scale);

            QuantBScale += 1;
            PackedQuantB += Q4BlkSize(BlkLen);
        }
    }
}

void
SQNBitGemmPackBZeroPoint_BlkBitWidth4(
    size_t N,
    size_t BlockCountK,
    size_t BlkLen,
    const std::byte* QuantBZeroPoint,
    std::byte* PackedQuantB
)
{
    if (QuantBZeroPoint != nullptr) {
        const std::byte* QuantBZeroPointCol = QuantBZeroPoint;
        const size_t QuantBZeroPointStride = MlasQNBitZeroPointsForBlksSizeInBytes<4>(BlockCountK);

        for (size_t n = 0; n < N; ++n) {
            for (size_t k_blk = 0; k_blk < BlockCountK; ++k_blk) {
                const std::byte zp_byte = QuantBZeroPointCol[k_blk / 2];
                const int8_t zp =
                    ((k_blk & 1) != 0)
                        ? std::to_integer<int8_t>(zp_byte >> 4)
                        : std::to_integer<int8_t>(zp_byte & std::byte{0x0F});

                Q4BlkSetZeroPoint(PackedQuantB, zp);

                PackedQuantB += Q4BlkSize(BlkLen);
            }
            QuantBZeroPointCol += QuantBZeroPointStride;
        }
    } else {
        for (size_t n = 0; n < N; ++n) {
            for (size_t k_blk = 0; k_blk < BlockCountK; ++k_blk) {
                Q4BlkSetZeroPoint(PackedQuantB, 8);

                PackedQuantB += Q4BlkSize(BlkLen);
            }
        }
    }
}
}  // namespace

size_t MLASCALL
MlasSQNBitGemmPackBSize2(
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    bool IsAsymmetric
)
{
    MLAS_UNREFERENCED_PARAMETER(IsAsymmetric);

    if (BlkBitWidth == 4 && IsSupportedBlkLen(BlkBitWidth, BlkLen)) {
        const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
        const size_t PackBSize = N * BlockCountK * Q4BlkSize(BlkLen);
        return PackBSize;
    }

    return 0;
}

void MLASCALL
MlasSQNBitGemmPackBData(
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    const void* QuantBData,
    void* PackedQuantB
)
{
    if (BlkBitWidth == 4) {
        const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
        SQNBitGemmPackBData_BlkBitWidth4(
            N,
            BlockCountK,
            BlkLen,
            static_cast<const std::byte*>(QuantBData),
            static_cast<std::byte*>(PackedQuantB)
        );
    }
}

void MLASCALL
MlasSQNBitGemmPackBScale(
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    const float* QuantBScale,
    void* PackedQuantB
)
{
    if (BlkBitWidth == 4) {
        const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
        SQNBitGemmPackBScale_BlkBitWidth4(
            N,
            BlockCountK,
            BlkLen,
            QuantBScale,
            static_cast<std::byte*>(PackedQuantB)
        );
    }
}

void MLASCALL
MlasSQNBitGemmPackBZeroPoint(
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    const void* QuantBZeroPoint,
    void* PackedQuantB
)
{
    if (BlkBitWidth == 4) {
        const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
        SQNBitGemmPackBZeroPoint_BlkBitWidth4(
            N,
            BlockCountK,
            BlkLen,
            static_cast<const std::byte*>(QuantBZeroPoint),
            static_cast<std::byte*>(PackedQuantB)
        );
    }
}

size_t MLASCALL
MlasSQNBitGemmPackBSize(
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    bool IsAsymmetric,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
#ifdef MLAS_JBLAS
    if (BlkBitWidth == 4) {
        auto jsize = JblasQ4GemmPackBSize(N, K, BlkLen, IsAsymmetric, ComputeType);
        if (jsize) {
            return jsize;
        }
    }
#endif
    (void)(N);
    (void)(K);
    (void)(BlkBitWidth);
    (void)(BlkLen);
    (void)(IsAsymmetric);
    (void)(ComputeType);
    return 0;
}

void MLASCALL
MlasSQNBitGemmPackB(
    void* PackedBuf,
    const void* QData,
    const float* Scale,
    const void* Zp,
    size_t N,
    size_t K,
    size_t ldb,
    size_t BlkBitWidth,
    size_t BlkLen,
    bool IsAsymmetric,
    bool IsLastCall,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType,
    MLAS_THREADPOOL* ThreadPool
)
{
#ifdef MLAS_JBLAS
    if (BlkBitWidth == 4) {
        if (JblasQ4GemmPackB(
                PackedBuf, static_cast<const uint8_t*>(QData), Scale, static_cast<const uint8_t*>(Zp), N, K, ldb,
                BlkLen, IsAsymmetric, IsLastCall, ComputeType, ThreadPool
            )) {
            return;
        }
    }
#endif
    (void)(PackedBuf);
    (void)(QData);
    (void)(Scale);
    (void)(Zp);
    (void)(N);
    (void)(K);
    (void)(ldb);
    (void)(BlkBitWidth);
    (void)(BlkLen);
    (void)(IsAsymmetric);
    (void)(IsLastCall);
    (void)(ComputeType);
    (void)(ThreadPool);
}

void MLASCALL
MlasNBitsGemmUnPackB(float* FpData, const void* PackedBuf, size_t N, size_t K, size_t ldb, MLAS_THREADPOOL* ThreadPool)
{
#ifdef MLAS_JBLAS
    if (JblasQ4GemmUnPackB(FpData, PackedBuf, N, K, ldb, ThreadPool)) {
        return;
    }
#endif
    (void)(FpData);
    (void)(PackedBuf);
    (void)(N);
    (void)(K);
    (void)(ldb);
    (void)(ThreadPool);
}

size_t MLASCALL
MlasSQNBitsGemmBatchPackedBWorkspaceSize(
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t BatchN,
    const MLAS_SQNBITS_GEMM_DATA_PACKED_PARAMS* DataParams
)
{
#ifdef MLAS_JBLAS
    return JblasSQ4GemmBatchWorkspaceSize(M, N, K, BatchN, DataParams);
#endif
    (void)(M);
    (void)(N);
    (void)(K);
    (void)(BatchN);
    (void)(DataParams);
    return 0;
}

void MLASCALL
MlasSQNBitsGemmBatchPackedB(
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t BatchN,
    const MLAS_SQNBITS_GEMM_DATA_PACKED_PARAMS* DataParams,
    void* WorkSpace,
    MLAS_THREADPOOL* ThreadPool
)
{
    GetMlasPlatform();
#ifdef MLAS_JBLAS
    if (JblasSQ4GemmBatchDriver(M, N, K, BatchN, DataParams, reinterpret_cast<int8_t*>(WorkSpace), ThreadPool)) {
        // PackedWeight is created by jblas
        return;
    }
#endif
    (void)(M);
    (void)(N);
    (void)(K);
    (void)(BatchN);
    (void)(DataParams);
    (void)(WorkSpace);
    (void)(ThreadPool);
}
