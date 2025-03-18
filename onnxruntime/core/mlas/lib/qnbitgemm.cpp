/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qnbitgemm.cpp

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication hardware agnostic entrypoint, MlasQNBitGemmBatch,
    as well as some SQNBitGemm-related query functions.
--*/

#include "qnbitgemm.h"
#include "sqnbitgemm_q8_block.h"

#include <cassert>

namespace
{

enum QNBitGemmVariant {
    SQNBitGemmVariantInvalid = -1,

    // Valid variants

    SQNBitGemmVariant_BitWidth4_CompFp32 = 0,
    SQNBitGemmVariant_BitWidth4_CompInt8,
    HQNBitGemmVariant_BitWidth4_CompFp16,
    HQNBitGemmVariant_BitWidth4_CompInt8,

    // End of valid variants

    // Keep this element last and ensure that its value is the number of valid QNBitGemmVariant values.
    // Its value is used as an array size.
    SQNBitGemmVariantCount,
};

QNBitGemmVariant
GetQNBitGemmVariant(
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    if (BlkBitWidth == 4 &&
        (BlkLen == 16 || BlkLen == 32 || BlkLen == 64 || BlkLen == 128 || BlkLen == 256)) {
        if (ComputeType == SQNBIT_CompFp32) {
            return SQNBitGemmVariant_BitWidth4_CompFp32;
        } else if (ComputeType == HQNBIT_CompFp16) {
            return HQNBitGemmVariant_BitWidth4_CompFp16;
        } else if (ComputeType == SQNBIT_CompInt8) {
            return SQNBitGemmVariant_BitWidth4_CompInt8;
        } else if (ComputeType == HQNBIT_CompInt8) {
            return HQNBitGemmVariant_BitWidth4_CompInt8;
        }
    }

    return SQNBitGemmVariantInvalid;
}

}  // namespace

bool MLASCALL
MlasIsQNBitGemmAvailable(
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    const auto* Dispatch = GetMlasPlatform().QNBitGemmDispatch;
    if (Dispatch == nullptr) {
        return false;
    }

    const auto Variant = GetQNBitGemmVariant(BlkBitWidth, BlkLen, ComputeType);

    switch (Variant) {
        case SQNBitGemmVariant_BitWidth4_CompFp32: {
            return Dispatch->SQ4BitGemmM1Kernel_CompFp32 != nullptr &&
                   Dispatch->SQ4BitBlkDequantBForSgemm_CompFp32 != nullptr;
        }
        case HQNBitGemmVariant_BitWidth4_CompFp16: {
            return Dispatch->HQ4BitGemmPackQuantBData != nullptr &&
                   Dispatch->HQ4BitGemmKernel_CompFp16 != nullptr &&
                   Dispatch->HQ4BitBlkDequantBForHgemm_CompFp16 != nullptr;
        }
        case SQNBitGemmVariant_BitWidth4_CompInt8: { // SQ4BitGemmKernel_BlkSum_CompInt8
            return
              (Dispatch->SQ4BitGemmKernel_CompInt8 != nullptr && Dispatch->QuantizeARow_CompInt8 != nullptr) ||
              (Dispatch->SQ4BitGemmKernel_BlkSum_CompInt8 != nullptr && Dispatch->QuantizeARowComputeBlkSum_CompInt8 != nullptr);
        }
        default: {
            return false;
        }
    }
}

namespace
{

size_t
QNBitGemmPerGemmWorkspaceSize(
    size_t M,
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    const auto* Dispatch = GetMlasPlatform().QNBitGemmDispatch;
    if (Dispatch == nullptr) {
        return 0;
    }

    if (BlkBitWidth == 4 && Dispatch->Q4BitGemmPerGemmWorkspaceSize != nullptr) {
        return Dispatch->Q4BitGemmPerGemmWorkspaceSize(M, N, K, BlkLen, ComputeType);
    }

    return 0;
}

size_t
QNBitGemmPerGemmWorkspaceAlignment(
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    const auto* Dispatch = GetMlasPlatform().QNBitGemmDispatch;
    if (Dispatch == nullptr) {
        return 1;
    }

    if (BlkBitWidth == 4 && Dispatch->Q4BitGemmPerGemmWorkspaceAlignment != nullptr) {
        return Dispatch->Q4BitGemmPerGemmWorkspaceAlignment(BlkLen, ComputeType);
    }

    return 1;
}

size_t
QNBitGemmPerGemmWorkspaceStride(
    size_t M,
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    const auto Size = QNBitGemmPerGemmWorkspaceSize(M, N, K, BlkBitWidth, BlkLen, ComputeType);
    const auto Alignment = QNBitGemmPerGemmWorkspaceAlignment(BlkBitWidth, BlkLen, ComputeType);
    return MlasDivRoundup(Size, Alignment) * Alignment;
}

}  // namespace

size_t MLASCALL
MlasQNBitGemmBatchWorkspaceSize(
    size_t M,
    size_t N,
    size_t K,
    size_t BatchN,
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    const size_t PerGemmWorkspaceStride = QNBitGemmPerGemmWorkspaceStride(M, N, K, BlkBitWidth, BlkLen, ComputeType);
    if (PerGemmWorkspaceStride == 0) {
        return 0;
    }

    const size_t Alignment = QNBitGemmPerGemmWorkspaceAlignment(BlkBitWidth, BlkLen, ComputeType);

    const size_t WorkspaceSize = BatchN * PerGemmWorkspaceStride;

    return WorkspaceSize + Alignment - 1;
}

size_t MLASCALL
MlasQNBitGemmPackQuantBDataSize(
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    const auto* Dispatch = GetMlasPlatform().QNBitGemmDispatch;
    if (Dispatch == nullptr) {
        return 0;
    }

    if (BlkBitWidth == 4 && Dispatch->Q4BitGemmPackQuantBDataSize != nullptr) {
        return Dispatch->Q4BitGemmPackQuantBDataSize(
            N, K, BlkLen, ComputeType
        );
    }

    return 0;
}

struct PerGemmQuantAWorkspace {
    PerGemmQuantAWorkspace(void* PerGemmWorkspace, size_t M, size_t BlockCountK, size_t BlkLen)
        : PerGemmWorkspace_(PerGemmWorkspace), M_(M), BlockCountK_(BlockCountK), BlkLen_(BlkLen)
    {
        QuantData = (std::byte*)PerGemmWorkspace;
        QuantScale = (float*)(QuantData + M * BlockCountK * BlkLen);
        BlockSum = QuantScale + M * BlockCountK;
    }
    std::byte* QuantData;     // NxBlockCountKxBlkLen
    float* QuantScale;        // NxBlockCountK
    float* BlockSum;          // NxBlockCountK
    void* PerGemmWorkspace_;  // memory for above data
    size_t M_, BlockCountK_, BlkLen_;
};

void MLASCALL
MlasQNBitGemmPackQuantBData(
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const void* QuantBData,
    void* PackedQuantBDataAndOrBlkSumWorkspace,
    const void* QuantBScale,
    bool has_zp_input,
    const void* QuantBZeroPoint,
    MLAS_THREADPOOL* ThreadPool
)
{
    const auto* Dispatch = GetMlasPlatform().QNBitGemmDispatch;
    if (Dispatch == nullptr) {
        return;
    }

    if (BlkBitWidth == 4) {
        if (ComputeType == SQNBIT_CompInt8 && Dispatch->SQ4BitGemmPackQuantBDataAndBlkSum != nullptr) {
            const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
            PackedQuantBDataStruct<float> packed_quant_b(PackedQuantBDataAndOrBlkSumWorkspace, N, BlockCountK, BlkLen);
            Dispatch->SQ4BitGemmPackQuantBDataAndBlkSum(
                N,
                K,
                BlkLen,
                ComputeType,
                static_cast<const std::byte*>(QuantBData),
                static_cast<const float*>(QuantBScale),
                has_zp_input,
                static_cast<const std::byte*>(QuantBZeroPoint),
                packed_quant_b,
                ThreadPool
            );
        } else if (ComputeType == HQNBIT_CompFp16 && Dispatch->HQ4BitGemmPackQuantBData != nullptr) {
            Dispatch->HQ4BitGemmPackQuantBData(
                N,
                K,
                BlkLen,
                ComputeType,
                static_cast<const std::byte*>(QuantBData),
                static_cast<std::byte*>(PackedQuantBDataAndOrBlkSumWorkspace),
                ThreadPool
            );
        } else if (Dispatch->SQ4BitGemmPackQuantBData != nullptr) {
          // TODO: these assertions are true if called from matmul_nbits kernel but not from mlas tests.
            //assert(QuantBScale == nullptr);
            //assert(QuantBZeroPoint == nullptr);
            Dispatch->SQ4BitGemmPackQuantBData(
                N,
                K,
                BlkLen,
                ComputeType,
                static_cast<const std::byte*>(QuantBData),
                static_cast<std::byte*>(PackedQuantBDataAndOrBlkSumWorkspace),
                ThreadPool
            );
            return;
        }
    }
}

namespace
{

MLAS_FORCEINLINE void
AddBiasForGemm(const float* Bias, float* C, size_t CountM, size_t CountN, size_t ldc)
{
    for (size_t m = 0; m < CountM; m++) {
        const float* bias = Bias;
        float* sum = C;
        for (size_t n = 0; n < CountN; n += 4) {
            if (CountN - n < 4) {
                for (size_t nn = n; nn < CountN; nn++) {
                    *sum += *bias;
                    sum++;
                    bias++;
                }
                break;
            }

            MLAS_FLOAT32X4 acc_x = MlasLoadFloat32x4(sum);
            acc_x = MlasAddFloat32x4(acc_x, MlasLoadFloat32x4(bias));
            MlasStoreFloat32x4(sum, acc_x);
            bias += 4;
            sum += 4;
        }
        C += ldc;
    }
}

void
SQ4BitGemm_CompFp32(
    const size_t BlkLen,
    const size_t K,
    const MLAS_QNBIT_GEMM_DATA_PARAMS<float>* const DataParams,
    void* const PerGemmWorkspace,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
)
{
    constexpr size_t BlkBitWidth = 4;

    MLAS_UNREFERENCED_PARAMETER(PerGemmWorkspace);

    const size_t lda = DataParams->lda;
    const size_t ldc = DataParams->ldc;

    const size_t k_blks = MlasDivRoundup(K, BlkLen);
    const size_t ldb = k_blks * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t k_blks_zp_bytes = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(k_blks);

    const float* A = DataParams->A + RangeStartM * lda;

    const std::byte* QuantBData = static_cast<const std::byte*>(DataParams->PackedQuantBData) + RangeStartN * ldb;
    const float* QuantBScale = DataParams->QuantBScale + RangeStartN * k_blks;
    const std::byte* QuantBZeroPoint =
        (DataParams->QuantBZeroPoint == nullptr)
            ? nullptr
            : static_cast<const std::byte*>(DataParams->QuantBZeroPoint) + RangeStartN * k_blks_zp_bytes;

    float* C = DataParams->C + RangeStartM * ldc + RangeStartN;

    const float* Bias = (DataParams->Bias == nullptr) ? nullptr : DataParams->Bias + RangeStartN;

    if (RangeCountM == 1) {
        size_t CountN;
        for (size_t n = 0; n < RangeCountN; n += CountN) {
            CountN = std::min(RangeCountN - n, size_t{128});

            const float* a_row = A;
            const std::byte* b_col = QuantBData + n * ldb;
            const float* b_col_scale = QuantBScale + n * k_blks;
            const std::byte* b_col_zp =
                (QuantBZeroPoint == nullptr) ? nullptr : QuantBZeroPoint + n * k_blks_zp_bytes;
            float* c_blk = C + n;
            const float* bias = (Bias == nullptr) ? nullptr : Bias + n;

            GetMlasPlatform().QNBitGemmDispatch->SQ4BitGemmM1Kernel_CompFp32(
                BlkLen,
                a_row, b_col, b_col_scale, b_col_zp, c_blk, CountN, K, k_blks, bias
            );

            if (DataParams->PostProcessor != nullptr) {
                DataParams->PostProcessor->Process(
                    DataParams->C, RangeStartM, RangeStartN + n,
                    RangeCountM, CountN, ldc
                );
            }
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
        const std::byte* b_col = QuantBData + n * ldb;
        const float* b_col_scale = QuantBScale + n * k_blks;
        const std::byte* b_col_zp =
            (QuantBZeroPoint == nullptr) ? nullptr : QuantBZeroPoint + n * k_blks_zp_bytes;
        float* c_blk = C + n;
        const float* bias = (Bias == nullptr) ? nullptr : Bias + n;

        GetMlasPlatform().QNBitGemmDispatch->SQ4BitBlkDequantBForSgemm_CompFp32(
            BlkLen,
            dequant_b, b_col, b_col_scale, b_col_zp, CountN, K, k_blks
        );

        size_t RowsRemaining = RangeCountM;
        while (RowsRemaining > 0) {
#if defined(MLAS_TARGET_AMD64_IX86) || defined(MLAS_TARGET_POWER) || defined(MLAS_TARGET_LARCH64)
            auto RowsHandled = GetMlasPlatform().GemmFloatKernel(
                a_row, dequant_b, c_blk, K, RowsRemaining, CountN, lda, ldc, 1.f, true
            );
#else
            auto RowsHandled = MlasSgemmKernelZero(a_row, dequant_b, c_blk, K, RowsRemaining, CountN, lda, ldc, 1.f);
#endif

            if (bias) {
                AddBiasForGemm(bias, c_blk, RowsHandled, CountN, ldc);
            }
            if (DataParams->PostProcessor != nullptr) {
                DataParams->PostProcessor->Process(
                    DataParams->C, RangeStartM + RangeCountM - RowsRemaining, RangeStartN + n,
                    RowsHandled, CountN, ldc
                );
            }

            c_blk += ldc * RowsHandled;
            a_row += lda * RowsHandled;
            RowsRemaining -= RowsHandled;
        }
    }
}

void
HQ4BitGemm_CompFp16(
    const size_t BlkLen,
    const size_t K,
    const MLAS_QNBIT_GEMM_DATA_PARAMS<MLAS_FP16>* const DataParams,
    void* const PerGemmWorkspace,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
)
{
    constexpr size_t BlkBitWidth = 4;
    MLAS_UNREFERENCED_PARAMETER(PerGemmWorkspace);

    const size_t lda = DataParams->lda;
    const size_t ldc = DataParams->ldc;
    const size_t k_blk_num = MlasDivRoundup(K, BlkLen);
    const size_t qldb = k_blk_num * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t ldb = k_blk_num * BlkLen;
    const size_t k_zp_bytes = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(k_blk_num);

    const MLAS_FP16* A = DataParams->A + RangeStartM * lda;
    MLAS_FP16* C = DataParams->C + RangeStartM * ldc + RangeStartN;
    const std::byte* QuantBData = static_cast<const std::byte*>(DataParams->PackedQuantBData) + RangeStartN * qldb;
    const MLAS_FP16* QuantBScale = DataParams->QuantBScale + RangeStartN * k_blk_num;
    const std::byte* QuantBZeroPoint =
        (DataParams->QuantBZeroPoint == nullptr)
            ? nullptr
            : static_cast<const std::byte*>(DataParams->QuantBZeroPoint) + RangeStartN * k_zp_bytes;
    const MLAS_FP16* Bias = (DataParams->Bias == nullptr) ? nullptr : DataParams->Bias;

    // 32N is the sweet spot of cache utilization. It is machine dependent though.
    constexpr size_t StrideM = 2;
    constexpr size_t StrideN = 32;

    // TODO(fajin): move allocation up to the op.
    size_t bufsize = ldb * StrideN * sizeof(MLAS_FP16);
    MlasThreadedBufAlloc(bufsize);
    auto* dequant_b = reinterpret_cast<MLAS_FP16*>(ThreadedBufHolder.get());

    for (size_t n = 0, countN; n < RangeCountN; n += countN) {
        countN = std::min(StrideN, RangeCountN - n);
        GetMlasPlatform().QNBitGemmDispatch->HQ4BitBlkDequantBForHgemm_CompFp16(
            BlkLen, dequant_b, QuantBData, QuantBScale, QuantBZeroPoint, countN, K, k_blk_num
        );

        const MLAS_FP16* a = A;
        MLAS_FP16* c = C;
        for (size_t m = 0, countM; m < RangeCountM; m += countM) {
            countM = std::min(StrideM, RangeCountM - m);
            GetMlasPlatform().QNBitGemmDispatch->HQ4BitGemmKernel_CompFp16(
                a, dequant_b, Bias, c, countM, countN, K, lda, ldb, ldc
            );

            if (DataParams->PostProcessor != nullptr) {
                DataParams->PostProcessor->Process(
                    DataParams->C, RangeStartM + m, RangeStartN + n, countM, countN, ldc
                );
            }

            a += countM * lda;
            c += countM * ldc;
        }

        QuantBData += countN * qldb;
        QuantBScale += countN * k_blk_num;
        QuantBZeroPoint = QuantBZeroPoint ? QuantBZeroPoint + countN * k_zp_bytes : nullptr;
        Bias = Bias ? Bias + countN : nullptr;
        C += countN;
    }
}

void
SQ4BitGemm_CompInt8(
    const size_t BlkLen,
    const size_t K,
    const MLAS_QNBIT_GEMM_DATA_PARAMS<float>* const DataParams,
    void* const PerGemmWorkspace,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
)
{
#ifdef MLAS_TARGET_AMD64_IX86
    PerGemmQuantAWorkspace* const per_gemm_quant_a_workspace = static_cast<PerGemmQuantAWorkspace*>(PerGemmWorkspace);
    constexpr size_t BlkBitWidth = 4;

    const size_t k_blks = MlasDivRoundup(K, BlkLen);

    // quant A scale is embedded in QuantData if QuantScale is nullptr.
    const size_t lda = k_blks * (per_gemm_quant_a_workspace->QuantScale ? BlkLen : Q8BlkSize(BlkLen));
    const size_t ldc = DataParams->ldc;
    const size_t ldb = k_blks * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t k_blks_zp_bytes = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(k_blks);

    const std::byte* QuantA = per_gemm_quant_a_workspace->QuantData + RangeStartM * lda;
    const float* QuantAScale = per_gemm_quant_a_workspace->QuantScale + RangeStartM * k_blks;

    assert(RangeStartN % 4 == 0);
    const std::byte* QuantBData = static_cast<const std::byte*>(DataParams->PackedQuantBData) + RangeStartN * ldb;
    const float* QuantBScale = DataParams->QuantBScale + RangeStartN * k_blks;
    const std::byte* QuantBZeroPoint =
        (DataParams->QuantBZeroPoint == nullptr)
            ? nullptr
            : static_cast<const std::byte*>(DataParams->QuantBZeroPoint) + RangeStartN * k_blks_zp_bytes;
    const float* ABlockSum = per_gemm_quant_a_workspace->BlockSum + RangeStartM * k_blks;
    const float* QuantBBlkSum = DataParams->QuantBBlkSum + RangeStartN * k_blks;
    float* C = DataParams->C + RangeStartM * ldc + RangeStartN;

    const float* Bias = (DataParams->Bias == nullptr) ? nullptr : DataParams->Bias + RangeStartN;
#else
    constexpr size_t BlkBitWidth = 4;

    const size_t k_blks = MlasDivRoundup(K, BlkLen);

    const size_t lda = k_blks * Q8BlkSize(BlkLen);
    const size_t ldc = DataParams->ldc;
    const size_t ldb = k_blks * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t k_blks_zp_bytes = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(k_blks);

    const std::byte* QuantA = static_cast<const std::byte*>(PerGemmWorkspace) + RangeStartM * lda;

    const std::byte* QuantBData = static_cast<const std::byte*>(DataParams->PackedQuantBData) + RangeStartN * ldb;
    const float* QuantBScale = DataParams->QuantBScale + RangeStartN * k_blks;
    const std::byte* QuantBZeroPoint =
        (DataParams->QuantBZeroPoint == nullptr)
            ? nullptr
            : static_cast<const std::byte*>(DataParams->QuantBZeroPoint) + RangeStartN * k_blks_zp_bytes;

    float* C = DataParams->C + RangeStartM * ldc + RangeStartN;

    const float* Bias = (DataParams->Bias == nullptr) ? nullptr : DataParams->Bias + RangeStartN;
#endif

    size_t CountN;
    for (size_t n = 0; n < RangeCountN; n += CountN) {
        CountN = std::min(RangeCountN - n, size_t{128});

        const std::byte* a_row = QuantA;
        const std::byte* b_col = QuantBData + n * ldb;
        const float* b_col_scale = QuantBScale + n * k_blks;
        const std::byte* b_col_zp =
            (QuantBZeroPoint == nullptr) ? nullptr : QuantBZeroPoint + n * k_blks_zp_bytes;
        float* c_blk = C + n;
        const float* bias = (Bias == nullptr) ? nullptr : Bias + n;

        if (GetMlasPlatform().QNBitGemmDispatch->SQ4BitGemmKernel_CompInt8 != nullptr) {
            size_t RowsRemaining = RangeCountM;
            while (RowsRemaining > 0) {
                const auto RowsHandled = GetMlasPlatform().QNBitGemmDispatch->SQ4BitGemmKernel_CompInt8(
                    BlkLen,
                    a_row, b_col, b_col_scale, b_col_zp, c_blk, RowsRemaining, CountN, K, k_blks, ldc, bias
                );

                if (DataParams->PostProcessor != nullptr) {
                    DataParams->PostProcessor->Process(
                        DataParams->C, RangeStartM + RangeCountM - RowsRemaining, RangeStartN + n,
                        RowsHandled, CountN, ldc
                    );
                }

                c_blk += RowsHandled * ldc;
                a_row += RowsHandled * lda;

                RowsRemaining -= RowsHandled;
            }
        }
#ifdef MLAS_TARGET_AMD64_IX86
        else if (GetMlasPlatform().QNBitGemmDispatch->SQ4BitGemmKernel_BlkSum_CompInt8 != nullptr)
        {
            const float* b_blk_sum = QuantBBlkSum + n * k_blks;
            GetMlasPlatform().QNBitGemmDispatch->SQ4BitGemmKernel_BlkSum_CompInt8(
                BlkLen,
                QuantA,
                QuantAScale,
                b_col,
                b_col_scale,
                b_col_zp,
                c_blk,
                RangeCountM,
                CountN,
                K,
                k_blks,
                bias,
                ldc,
                ABlockSum,
                b_blk_sum
            );

            if (DataParams->PostProcessor != nullptr) {
                DataParams->PostProcessor->Process(
                    DataParams->C, RangeStartM, RangeStartN + n,
                    RangeCountM, CountN, ldc
                );
            }
        }
#endif
    }
}

template <typename T>
void
InitializeWorkspace_CompInt8(
    size_t M,
    size_t N,
    size_t K,
    size_t BatchN,
    size_t BlkLen,
    const MLAS_QNBIT_GEMM_DATA_PARAMS<T>* DataParams,
    void* Workspace,
    size_t PerGemmWorkspaceStride,
    MLAS_THREADPOOL* ThreadPool
);

template <>
void
InitializeWorkspace_CompInt8<float>(
    size_t M,
    size_t N,
    size_t K,
    size_t BatchN,
    size_t BlkLen,
    const MLAS_QNBIT_GEMM_DATA_PARAMS<float>* DataParams,
    void* Workspace,
    size_t PerGemmWorkspaceStride,
    MLAS_THREADPOOL* ThreadPool
)
{
    MLAS_UNREFERENCED_PARAMETER(N);

    const auto QuantizeARow = GetMlasPlatform().QNBitGemmDispatch->QuantizeARow_CompInt8;
    const auto QuantizeARow2 = GetMlasPlatform().QNBitGemmDispatch->QuantizeARowComputeBlkSum_CompInt8;

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t QuantAStride = BlockCountK * Q8BlkSize(BlkLen);

    // TODO: try parallel on BatchN * M threads because BatchN is usually 1.
    if (QuantizeARow) {
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
    } else {
        MlasTrySimpleParallel(ThreadPool, BatchN, [&](ptrdiff_t gemm_idx) {
            const auto& data = DataParams[gemm_idx];
            const float* ARowPtr = data.A;

            void* PerGemmWorkspace = static_cast<std::byte*>(Workspace) + gemm_idx * PerGemmWorkspaceStride;
            PerGemmQuantAWorkspace quant_a_data(PerGemmWorkspace, M, BlockCountK, BlkLen);
            std::byte* QuantARowPtr = quant_a_data.QuantData;
            float* QuantARowScalePtr = quant_a_data.QuantScale;
            float* QuantARowBlkSum = quant_a_data.BlockSum;
            for (size_t m = 0; m < M; ++m) {
                QuantizeARow2(BlkLen, ARowPtr, K, QuantARowPtr, QuantARowScalePtr, QuantARowBlkSum);
                ARowPtr += data.lda;
                QuantARowPtr += BlockCountK * BlkLen;
                QuantARowScalePtr += BlockCountK;
                QuantARowBlkSum += BlockCountK;
            }
        });
    }
}

template <>
void
InitializeWorkspace_CompInt8<MLAS_FP16>(
    size_t M,
    size_t N,
    size_t K,
    size_t BatchN,
    size_t BlkLen,
    const MLAS_QNBIT_GEMM_DATA_PARAMS<MLAS_FP16>* DataParams,
    void* Workspace,
    size_t PerGemmWorkspaceStride,
    MLAS_THREADPOOL* ThreadPool
) {
    MLAS_UNREFERENCED_PARAMETER(M);
    MLAS_UNREFERENCED_PARAMETER(N);
    MLAS_UNREFERENCED_PARAMETER(K);
    MLAS_UNREFERENCED_PARAMETER(BatchN);
    MLAS_UNREFERENCED_PARAMETER(BlkLen);
    MLAS_UNREFERENCED_PARAMETER(DataParams);
    MLAS_UNREFERENCED_PARAMETER(Workspace);
    MLAS_UNREFERENCED_PARAMETER(PerGemmWorkspaceStride);
    MLAS_UNREFERENCED_PARAMETER(ThreadPool);
}

template <typename T>
using InitializeWorkspaceFn = std::function<void(
    size_t M,
    size_t N,
    size_t K,
    size_t BatchN,
    size_t BlkLen,
    const MLAS_QNBIT_GEMM_DATA_PARAMS<T>* DataParams,
    void* Workspace,
    size_t PerGemmWorkspaceStride,
    MLAS_THREADPOOL* ThreadPool
)>;

template <typename T>
InitializeWorkspaceFn<T>
GetInitializeWorkspace(QNBitGemmVariant variant);

template <>
InitializeWorkspaceFn<float>
GetInitializeWorkspace(QNBitGemmVariant variant)
{
    switch (variant) {
        case SQNBitGemmVariant_BitWidth4_CompInt8:
            return InitializeWorkspace_CompInt8<float>;
        default:
            return nullptr;
    }
}

template <>
InitializeWorkspaceFn<MLAS_FP16>
GetInitializeWorkspace(QNBitGemmVariant variant)
{
    switch (variant) {
        case HQNBitGemmVariant_BitWidth4_CompInt8:
            return InitializeWorkspace_CompInt8<MLAS_FP16>;
        default:
            return nullptr;
    }
}

template <typename T>
using QNBitGemmFn = std::function<void(
    const size_t BlkLen,
    const size_t K,
    const MLAS_QNBIT_GEMM_DATA_PARAMS<T>* const DataParams,
    void* const PerGemmWorkspace,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
)>;

template <typename T>
QNBitGemmFn<T>
GetQNBitGemm(QNBitGemmVariant variant);

template <>
QNBitGemmFn<float>
GetQNBitGemm(QNBitGemmVariant variant)
{
    switch (variant) {
        case SQNBitGemmVariant_BitWidth4_CompFp32:
            return SQ4BitGemm_CompFp32;
        case SQNBitGemmVariant_BitWidth4_CompInt8:
            return SQ4BitGemm_CompInt8;
        default:
            return nullptr;
    }
}

template <>
QNBitGemmFn<MLAS_FP16>
GetQNBitGemm(QNBitGemmVariant variant)
{
    switch (variant) {
        case HQNBitGemmVariant_BitWidth4_CompFp16:
            return HQ4BitGemm_CompFp16;
        default:
            return nullptr;
    }
}
}  // namespace

template <typename T>
void MLASCALL
MlasQNBitGemmBatch(
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t BatchN,
    const size_t BlkBitWidth,
    const size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const MLAS_QNBIT_GEMM_DATA_PARAMS<T>* DataParams,
    void* Workspace,
    MLAS_THREADPOOL* ThreadPool
)
{
    const auto Variant = GetQNBitGemmVariant(BlkBitWidth, BlkLen, ComputeType);
    assert(Variant != SQNBitGemmVariantInvalid);

    //
    // Ensure `Workspace` has correct alignment.
    //
    if (Workspace != nullptr) {
        const size_t Alignment = QNBitGemmPerGemmWorkspaceAlignment(BlkBitWidth, BlkLen, ComputeType);
        const uintptr_t WorkspaceAddress = reinterpret_cast<uintptr_t>(Workspace);
        Workspace = reinterpret_cast<void*>(
            (WorkspaceAddress + Alignment - 1) & (~(Alignment - 1))
        );
    }

    const size_t PerGemmWorkspaceStride = QNBitGemmPerGemmWorkspaceStride(M, N, K, BlkBitWidth, BlkLen, ComputeType);

    if (const auto InitializeWorkspaceOperation = GetInitializeWorkspace<T>(Variant);
        InitializeWorkspaceOperation != nullptr) {
        InitializeWorkspaceOperation(
            M, N, K, BatchN, BlkLen, DataParams, Workspace, PerGemmWorkspaceStride, ThreadPool
        );
    }

    const auto ComputeOperation = GetQNBitGemm<T>(Variant);

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);

    if (ThreadPool == nullptr) {
        for (size_t gemm_i = 0; gemm_i < BatchN; gemm_i++) {
            const auto* Data = &DataParams[gemm_i];
            void* PerGemmWorkspace =
                reinterpret_cast<std::byte*>(Workspace) + gemm_i * PerGemmWorkspaceStride;
            if (ComputeType == SQNBIT_CompInt8 && GetMlasPlatform().QNBitGemmDispatch->SQ4BitGemmPackQuantBDataAndBlkSum != nullptr) {
                PackedQuantBDataStruct<T> packed_quant_b(const_cast<void*>(Data->QuantBDataWorkspace), N, BlockCountK, BlkLen);
                const_cast<MLAS_QNBIT_GEMM_DATA_PARAMS<T>*>(Data)->PackedQuantBData = packed_quant_b.PackedQuantBData;
                const_cast<MLAS_QNBIT_GEMM_DATA_PARAMS<T>*>(Data)->QuantBBlkSum = packed_quant_b.QuantBBlkSum;
                const_cast<MLAS_QNBIT_GEMM_DATA_PARAMS<T>*>(Data)->QuantBScale = packed_quant_b.PackedQuantBScale;
                PerGemmQuantAWorkspace per_gemm_quant_a_workspace(PerGemmWorkspace, M, BlockCountK, BlkLen);
                ComputeOperation(BlkLen, K, Data, &per_gemm_quant_a_workspace, 0, M, 0, N);
            } else {
                ComputeOperation(BlkLen, K, Data, PerGemmWorkspace, 0, M, 0, N);
            }
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

        const ptrdiff_t ThreadIdN = blk_i / ThreadCountM;
        const ptrdiff_t ThreadIdM = blk_i % ThreadCountM;

        const size_t RangeStartM = ThreadIdM * StrideM;
        const size_t RangeCountM = std::min(M - RangeStartM, (size_t)StrideM);

        const size_t RangeStartN = ThreadIdN * StrideN;
        const size_t RangeCountN = std::min(N - RangeStartN, (size_t)StrideN);

        void* PerGemmWorkspace =
            reinterpret_cast<std::byte*>(Workspace) + gemm_i * PerGemmWorkspaceStride;
        if (ComputeType == SQNBIT_CompInt8 && GetMlasPlatform().QNBitGemmDispatch->SQ4BitGemmPackQuantBDataAndBlkSum != nullptr) {
            PackedQuantBDataStruct<T> packed_quant_b(const_cast<void*>(Data->QuantBDataWorkspace), N, BlockCountK, BlkLen);
            const_cast<MLAS_QNBIT_GEMM_DATA_PARAMS<T>*>(Data)->PackedQuantBData = packed_quant_b.PackedQuantBData;
            const_cast<MLAS_QNBIT_GEMM_DATA_PARAMS<T>*>(Data)->QuantBBlkSum = packed_quant_b.QuantBBlkSum;
            const_cast<MLAS_QNBIT_GEMM_DATA_PARAMS<T>*>(Data)->QuantBScale = packed_quant_b.PackedQuantBScale;

            PerGemmQuantAWorkspace per_gemm_quant_a_workspace(PerGemmWorkspace, M, BlockCountK, BlkLen);
            ComputeOperation(BlkLen, K, Data, &per_gemm_quant_a_workspace, RangeStartM, RangeCountM, RangeStartN, RangeCountN);
        } else {
            ComputeOperation(BlkLen, K, Data, PerGemmWorkspace, RangeStartM, RangeCountM, RangeStartN, RangeCountN);
        }
    });
}

template
void MLASCALL
MlasQNBitGemmBatch(
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t BatchN,
    const size_t BlkBitWidth,
    const size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const MLAS_QNBIT_GEMM_DATA_PARAMS<float>* DataParams,
    void* Workspace,
    MLAS_THREADPOOL* ThreadPool
);

template
void MLASCALL
MlasQNBitGemmBatch(
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t BatchN,
    const size_t BlkBitWidth,
    const size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const MLAS_QNBIT_GEMM_DATA_PARAMS<MLAS_FP16>* DataParams,
    void* Workspace,
    MLAS_THREADPOOL* ThreadPool
);
