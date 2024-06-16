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
#include "sqnbitgemm_q8_block.h"

#include <cassert>
#include <iostream>
#include <chrono>
#include "core/common/profiler.h"

class ProfilerWrapper
{
   public:
    ProfilerWrapper()
    {
        profiler_ = std::make_unique<onnxruntime::profiling::Profiler>();
        //profiler_->StartProfiling<char>("profile.json");
    }

    ~ProfilerWrapper()
    {
        if (profiler_) {
            //profiler_->EndProfiling();
        }
    }

    onnxruntime::profiling::Profiler* operator->()
    {
        return profiler_.get();
    }

   private:
    std::unique_ptr<onnxruntime::profiling::Profiler> profiler_;
};

static ProfilerWrapper profiler_;

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

SQNBitGemmVariant
GetSQNBitGemmVariant(
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    if (BlkBitWidth == 4 &&
        (BlkLen == 16 || BlkLen == 32 || BlkLen == 64 || BlkLen == 128 || BlkLen == 256)) {
        if (ComputeType == CompFp32 ||
            ComputeType == CompUndef) {  // treat CompUndef (undefined) as CompFp32
            return SQNBitGemmVariant_BitWidth4_CompFp32;
        } else if (ComputeType == CompInt8) {
            return SQNBitGemmVariant_BitWidth4_CompInt8;
        }
    }

    return SQNBitGemmVariantInvalid;
}

}  // namespace

bool MLASCALL
MlasIsSQNBitGemmAvailable(
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    const auto* Dispatch = GetMlasPlatform().SQNBitGemmDispatch;
    if (Dispatch == nullptr) {
        return false;
    }

    const auto Variant = GetSQNBitGemmVariant(BlkBitWidth, BlkLen, ComputeType);

    switch (Variant) {
        case SQNBitGemmVariant_BitWidth4_CompFp32: {
            return Dispatch->SQ4BitGemmM1Kernel_CompFp32 != nullptr &&
                   Dispatch->Q4BitBlkDequantBForSgemm_CompFp32 != nullptr;
        }
        case SQNBitGemmVariant_BitWidth4_CompInt8: {
            return
              (Dispatch->SQ4BitGemmM1Kernel_CompInt8 != nullptr && Dispatch->QuantizeARow_CompInt8 != nullptr) ||
              (Dispatch->SQ4BitGemmKernel_CompInt8 != nullptr && Dispatch->QuantizeARow_CompInt8_2 != nullptr);
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
    size_t M,
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    MLAS_UNREFERENCED_PARAMETER(N);

    const auto* Dispatch = GetMlasPlatform().SQNBitGemmDispatch;
    if (Dispatch == nullptr) {
        return 0;
    }

    if (BlkBitWidth == 4 && Dispatch->SQ4BitGemmPerGemmWorkspaceSize != nullptr) {
        return Dispatch->SQ4BitGemmPerGemmWorkspaceSize(M, N, K, BlkLen, ComputeType);
    }

    return 0;
}

size_t
SQNBitGemmPerGemmWorkspaceAlignment(
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    const auto* Dispatch = GetMlasPlatform().SQNBitGemmDispatch;
    if (Dispatch == nullptr) {
        return 1;
    }

    if (BlkBitWidth == 4 && Dispatch->SQ4BitGemmPerGemmWorkspaceAlignment != nullptr) {
        return Dispatch->SQ4BitGemmPerGemmWorkspaceAlignment(BlkLen, ComputeType);
    }

    return 1;
}

size_t
SQNBitGemmPerGemmWorkspaceStride(
    size_t M,
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    const auto Size = SQNBitGemmPerGemmWorkspaceSize(M, N, K, BlkBitWidth, BlkLen, ComputeType);
    const auto Alignment = SQNBitGemmPerGemmWorkspaceAlignment(BlkBitWidth, BlkLen, ComputeType);
    return MlasDivRoundup(Size, Alignment) * Alignment;
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
    const size_t PerGemmWorkspaceStride = SQNBitGemmPerGemmWorkspaceStride(M, N, K, BlkBitWidth, BlkLen, ComputeType);
    if (PerGemmWorkspaceStride == 0) {
        return 0;
    }

    const size_t Alignment = SQNBitGemmPerGemmWorkspaceAlignment(BlkBitWidth, BlkLen, ComputeType);

    const size_t WorkspaceSize = BatchN * PerGemmWorkspaceStride;

    return WorkspaceSize + Alignment - 1;
}

size_t MLASCALL
MlasSQNBitGemmPackQuantBDataSize(
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    const auto* Dispatch = GetMlasPlatform().SQNBitGemmDispatch;
    if (Dispatch == nullptr) {
        return 0;
    }

    if (BlkBitWidth == 4 && Dispatch->SQ4BitGemmPackQuantBDataSize != nullptr) {
        return Dispatch->SQ4BitGemmPackQuantBDataSize(
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
MlasSQNBitGemmPackQuantBData(
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const void* QuantBData,
    void* PackedQuantBDataAndOrBlkSum,
    const void* QuantBScale,
    bool has_zp_input,
    const void* QuantBZeroPoint,
    MLAS_THREADPOOL* ThreadPool
)
{
    const auto* Dispatch = GetMlasPlatform().SQNBitGemmDispatch;
    if (Dispatch == nullptr) {
        return;
    }

    if (BlkBitWidth == 4) {
        if (Dispatch->SQ4BitGemmPackQuantBData != nullptr) {
            assert(QuantBScale == nullptr);
            assert(QuantBZeroPoint == nullptr);
            Dispatch->SQ4BitGemmPackQuantBData(
                N,
                K,
                BlkLen,
                ComputeType,
                static_cast<const std::byte*>(QuantBData),
                static_cast<std::byte*>(PackedQuantBDataAndOrBlkSum),
                ThreadPool
            );
            return;
        } else if (Dispatch->SQ4BitGemmPackQuantBDataAndBlkSum != nullptr) {
            const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
            PackedQuantBDataStruct packed_quant_b(PackedQuantBDataAndOrBlkSum, N, BlockCountK, BlkLen);
            Dispatch->SQ4BitGemmPackQuantBDataAndBlkSum(
                N,
                K,
                BlkLen,
                ComputeType,
                static_cast<const std::byte*>(QuantBData),
                packed_quant_b.PackedQuantBData,
                static_cast<const float*>(QuantBScale),
                has_zp_input,
                static_cast<const std::byte*>(QuantBZeroPoint),
                packed_quant_b.QuantBBlkSum,
                ThreadPool
            );
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

typedef void(SQNBitGemmFn)(
    size_t BlkLen,
    size_t K,
    const MLAS_SQNBIT_GEMM_DATA_PARAMS* DataParams,
    PerGemmQuantAWorkspace* PerGemmWorkspace,
    size_t RangeStartM,
    size_t RangeCountM,
    size_t RangeStartN,
    size_t RangeCountN
);

void
SQ4BitGemm_CompFp32(
    const size_t BlkLen,
    const size_t K,
    const MLAS_SQNBIT_GEMM_DATA_PARAMS* const DataParams,
    PerGemmQuantAWorkspace* const PerGemmWorkspace,
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

    const std::byte* QuantBData = static_cast<const std::byte*>(DataParams->QuantBData) + RangeStartN * ldb;
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

            GetMlasPlatform().SQNBitGemmDispatch->SQ4BitGemmM1Kernel_CompFp32(
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

        GetMlasPlatform().SQNBitGemmDispatch->Q4BitBlkDequantBForSgemm_CompFp32(
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
                    DataParams->C, RangeStartM + RangeCountM - RowsRemaining, RangeStartN,
                    RowsHandled, CountN, ldc
                );
            }

            c_blk += ldc * RowsHandled;
            a_row += lda * RowsHandled;
            RowsRemaining -= RowsHandled;
        }
    }
}

//#define BlockSumM1Layout 1
//#define CALL_SGEMM_SEPARATELY 1
#if defined(CALL_SGEMM_SEPARATELY)
void
SQ4BitGemm_CompInt8_0(
    const size_t BlkLen,
    const size_t K,
    const MLAS_SQNBIT_GEMM_DATA_PARAMS* const DataParams,
    PerGemmQuantAWorkspace* const per_gemm_quant_a_workspace,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
)
{
    const size_t k_blks = MlasDivRoundup(K, BlkLen);

    // quant A scale is embedded in QuantData if QuantScale is nullptr.
    const size_t ldc = DataParams->ldc;

    const float* ABlockSum = per_gemm_quant_a_workspace->BlockSum + RangeStartM * k_blks;

#if defined(BlockSumM1Layout)
    const float* QuantBBlkSum = DataParams->QuantBBlkSum + RangeStartN;
#else
    const float* QuantBBlkSum = DataParams->QuantBBlkSum + RangeStartN * k_blks;
#endif


    float* C = DataParams->C + RangeStartM * ldc + RangeStartN;

    if (RangeCountM == 1) {
        // auto start = std::chrono::high_resolution_clock::now();  // Start timing here

        size_t CountN;
        for (size_t n = 0; n < RangeCountN; n += CountN) {
            CountN = std::min(RangeCountN - n, size_t{128});

            float* c_blk = C + n;
            //std::chrono::high_resolution_clock::time_point tp;
            //if (profiler_->IsEnabled()) {
            //    tp = profiler_->Start();
            //}
#if defined(BlockSumM1Layout)
             const float* b_blk_sum = QuantBBlkSum + n;
             GetMlasPlatform().KernelM1Routine(ABlockSum, b_blk_sum, c_blk, k_blks, CountN, ldc, 0.0f);
            //  GetMlasPlatform().KernelM1TransposeBRoutine(ABlockSum, b_blk_sum, c_blk, k_blks, CountN, ldc, 0.0f);
#else
            const float* b_blk_sum = QuantBBlkSum + n * k_blks;
            GetMlasPlatform().GemmFloatKernel(
                ABlockSum, b_blk_sum, c_blk, k_blks, RangeCountM, CountN, k_blks, ldc, 1.f, true
            );
#endif
            //if (profiler_->IsEnabled()) {
            //    std::string eventName = DataParams->node_name + "Sep GemmFloatKernel_" + std::to_string(RangeCountM) + "_" + std::to_string(CountN) + "_" + std::to_string(k_blks);
            //    profiler_->EndTimeAndRecordEvent(onnxruntime::profiling::KERNEL_EVENT, eventName, tp);
            //}
        }
        // auto end = std::chrono::high_resolution_clock::now();  // End timing here
        //// Calculate and print the duration in nanoseconds
        // std::chrono::duration<double, std::nano> elapsed = end - start;
        // std::cout << "Duration_M" << RangeCountM << "xN" << RangeCountN << "xK" << K << ": " << elapsed.count() << " ns\n";
        return;
    } else {
        size_t CountN;
        for (size_t n = 0; n < RangeCountN; n += CountN) {
            CountN = std::min(RangeCountN - n, size_t{128});

            const float* b_blk_sum = QuantBBlkSum + n * k_blks;

            float* c_blk = C + n;

            if (GetMlasPlatform().SQNBitGemmDispatch->SQ4BitGemmPackQuantBDataAndBlkSum) {
                size_t RowsRemaining = RangeCountM;
                const float* a_blksum_row = ABlockSum;
                while (RowsRemaining > 0) {
                    auto RowsHandled = GetMlasPlatform().GemmFloatKernel(
                        a_blksum_row, b_blk_sum, c_blk, k_blks, RowsRemaining, CountN, k_blks, ldc, 1.f, true
                    );

                    c_blk += ldc * RowsHandled;
                    a_blksum_row += k_blks * RowsHandled;
                    RowsRemaining -= RowsHandled;
                }
            }
        }
    }
}
#endif

void
SQ4BitGemm_CompInt8(
    const size_t BlkLen,
    const size_t K,
    const MLAS_SQNBIT_GEMM_DATA_PARAMS* const DataParams,
    PerGemmQuantAWorkspace* const per_gemm_quant_a_workspace,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
)
{
//#ifdef MLAS_TARGET_AMD64_IX86
//    if (RangeCountM != 1) {
//        // perf experiment shows fp32 is faster than int8 in M > 1 cases.
//        // route to fp32 compute before int8 compute is improved.
//        SQ4BitGemm_CompFp32(
//            BlkLen,
//            K, DataParams, PerGemmWorkspace, RangeStartM, RangeCountM, RangeStartN, RangeCountN
//        );
//        return;
//    }
//#endif
    constexpr size_t BlkBitWidth = 4;

    const size_t k_blks = MlasDivRoundup(K, BlkLen);

    // quant A scale is embedded in QuantData if QuantScale is nullptr.
    const size_t lda = k_blks * (per_gemm_quant_a_workspace->QuantScale ? BlkLen : Q8BlkSize(BlkLen));
    const size_t ldc = DataParams->ldc;
    const size_t ldb = k_blks * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t k_blks_zp_bytes = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(k_blks);

    const std::byte* QuantA = per_gemm_quant_a_workspace->QuantData + RangeStartM * lda;
    const float* QuantAScale = per_gemm_quant_a_workspace->QuantScale + RangeStartM * k_blks;

    const std::byte* QuantBData = static_cast<const std::byte*>(DataParams->QuantBData) + RangeStartN * ldb;
    const float* QuantBScale = DataParams->QuantBScale + RangeStartN * k_blks;
    const std::byte* QuantBZeroPoint =
        (DataParams->QuantBZeroPoint == nullptr)
            ? nullptr
            : static_cast<const std::byte*>(DataParams->QuantBZeroPoint) + RangeStartN * k_blks_zp_bytes;
#ifndef CALL_SGEMM_SEPARATELY
    const float* ABlockSum = per_gemm_quant_a_workspace->BlockSum + RangeStartM * k_blks;
#if defined(BlockSumM1Layout)
    const float* QuantBBlkSum = DataParams->QuantBBlkSum + RangeStartN;
#else
    const float* QuantBBlkSum = DataParams->QuantBBlkSum + RangeStartN * k_blks;
#endif
#endif
    float* C = DataParams->C + RangeStartM * ldc + RangeStartN;

    const float* Bias = (DataParams->Bias == nullptr) ? nullptr : DataParams->Bias + RangeStartN;

    if (RangeCountM == 1) {
        if (GetMlasPlatform().SQNBitGemmDispatch->SQ4BitGemmKernel_CompInt8)
        {
            //auto start = std::chrono::high_resolution_clock::now();  // Start timing here

            size_t CountN;
            for (size_t n = 0; n < RangeCountN; n += CountN) {
                CountN = std::min(RangeCountN - n, size_t{128});

                const std::byte* b_col = QuantBData + n * ldb;
                const float* b_col_scale = QuantBScale + n * k_blks;
                float* c_blk = C + n;
                const float* bias = (Bias == nullptr) ? nullptr : Bias + n;
                //std::chrono::high_resolution_clock::time_point tp;
                //if (profiler_->IsEnabled()) {
                //    tp = profiler_->Start();
                //}
                GetMlasPlatform().SQNBitGemmDispatch->SQ4BitGemmKernel_CompInt8(
                    BlkLen,
                    QuantA,
                    QuantAScale,
                    b_col,
                    b_col_scale,
                    c_blk,
                    RangeCountM,
                    CountN,
                    K,
                    k_blks,
                    bias,
                    lda,
                    ldc
                );
                //if (profiler_->IsEnabled()) {
                //    std::string eventName = DataParams->node_name + "SQ4BitGemmKernel_CompInt_" + std::to_string(RangeCountM) + "_" + std::to_string(CountN) + "_" + std::to_string(K);
                //    profiler_->EndTimeAndRecordEvent(onnxruntime::profiling::KERNEL_EVENT, eventName, tp);
                //}

// #ifndef CALL_SGEMM_SEPARATELY
//                 if (profiler_->IsEnabled()) {
//                     tp = profiler_->Start();
//                 }
#if defined(BlockSumM1Layout)
                const float* b_blk_sum = QuantBBlkSum + n;
                GetMlasPlatform().KernelM1Routine(ABlockSum, b_blk_sum, c_blk, k_blks, CountN, ldc, 0.0f);
                // GetMlasPlatform().KernelM1TransposeBRoutine(ABlockSum, b_blk_sum, c_blk, k_blks, CountN, ldc, 0.0f);
#else
                const float* b_blk_sum = QuantBBlkSum + n * k_blks;
                GetMlasPlatform().GemmFloatKernel(
                    ABlockSum, b_blk_sum, c_blk, k_blks, RangeCountM, CountN, k_blks, ldc, 1.f, false
                );
#endif
                //                if (profiler_->IsEnabled()) {
                //                    std::string eventName = DataParams->node_name + "GemmFloatKernel_" + std::to_string(RangeCountM) + "_" + std::to_string(CountN) + "_" + std::to_string(k_blks);
                //                    profiler_->EndTimeAndRecordEvent(onnxruntime::profiling::KERNEL_EVENT, eventName, tp);
                //                }
// #endif
                if (DataParams->PostProcessor != nullptr) {
                    DataParams->PostProcessor->Process(
                        DataParams->C, RangeStartM, RangeStartN + n,
                        RangeCountM, CountN, ldc
                    );
                }
            }
            //auto end = std::chrono::high_resolution_clock::now();  // End timing here
            //// Calculate and print the duration in nanoseconds
            //std::chrono::duration<double, std::nano> elapsed = end - start;
            //std::cout << "Duration_M" << RangeCountM << "xN" << RangeCountN << "xK" << K << ": " << elapsed.count() << " ns\n";
            return;
        } else {
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

                GetMlasPlatform().SQNBitGemmDispatch->SQ4BitGemmM1Kernel_CompInt8(
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
        }
        return;
    }

    if (GetMlasPlatform().SQNBitGemmDispatch->SQ4BitGemmKernel_CompInt8)
    {
        size_t CountN;
        for (size_t n = 0; n < RangeCountN; n += CountN) {
            CountN = std::min(RangeCountN - n, size_t{128});

            const std::byte* b_col = QuantBData + n * ldb;
            const float* b_col_scale = QuantBScale + n * k_blks;
            float* c_blk = C + n;
            const float* bias = (Bias == nullptr) ? nullptr : Bias + n;
            //std::chrono::high_resolution_clock::time_point tp;
            //if (profiler_->IsEnabled()) {
            //    tp = profiler_->Start();
            //}
            GetMlasPlatform().SQNBitGemmDispatch->SQ4BitGemmKernel_CompInt8(
                BlkLen,
                QuantA,
                QuantAScale,
                b_col,
                b_col_scale,
                c_blk,
                RangeCountM,
                CountN,
                K,
                k_blks,
                bias,
                lda,
                ldc
            );
            //if (profiler_->IsEnabled()) {
            //    std::string eventName = DataParams->node_name + "SQ4BitGemmKernel_CompInt8_" + std::to_string(RangeCountM) + "_" + std::to_string(CountN) + "_" + std::to_string(K);
            //    profiler_->EndTimeAndRecordEvent(onnxruntime::profiling::KERNEL_EVENT, eventName, tp);
            //}

#ifndef CALL_SGEMM_SEPARATELY
            //if (profiler_->IsEnabled()) {
            //    tp = profiler_->Start();
            //}

            const float* b_blk_sum = QuantBBlkSum + n * k_blks;
            if (GetMlasPlatform().SQNBitGemmDispatch->SQ4BitGemmPackQuantBDataAndBlkSum) {
                size_t RowsRemaining = RangeCountM;
                const float* a_blksum_row = ABlockSum;
                while (RowsRemaining > 0) {
                    auto RowsHandled = GetMlasPlatform().GemmFloatKernel(
                        a_blksum_row, b_blk_sum, c_blk, k_blks, RowsRemaining, CountN, k_blks, ldc, 1.f, false
                    );

                    c_blk += ldc * RowsHandled;
                    a_blksum_row += k_blks * RowsHandled;
                    RowsRemaining -= RowsHandled;
                }
            }
            //if (profiler_->IsEnabled()) {
            //    std::string eventName = DataParams->node_name + "GemmFloatKernel_" + std::to_string(RangeCountM) + "_" + std::to_string(CountN) + "_" + std::to_string(k_blks);
            //    profiler_->EndTimeAndRecordEvent(onnxruntime::profiling::KERNEL_EVENT, eventName, tp);
            //}
#endif
            if (DataParams->PostProcessor != nullptr) {
                DataParams->PostProcessor->Process(
                    DataParams->C, RangeStartM, RangeStartN + n,
                    RangeCountM, CountN, ldc
                );
            }

        }
        return;
    }

    // This is a naive M > 1 implementation that repeatedly calls the M=1 kernel.
    // TODO Replace it with an optimized implementation.
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
        for (size_t m = 0; m < RangeCountM; ++m) {
            GetMlasPlatform().SQNBitGemmDispatch->SQ4BitGemmM1Kernel_CompInt8(
                BlkLen,
                a_row, b_col, b_col_scale, b_col_zp, c_blk, CountN, K, k_blks, bias
            );
            // GetMlasPlatform().SQNBitGemmDispatch->SQ4BitGemmKernel_CompInt8(
            //     BlkLen,
            //     a_row, b_col, b_col_scale, b_col_zp, c_blk, /*RangeCountM*/1, CountN,
            //     K, k_blks, bias, lda, ldc
            //);

            // TODO: shall be processed outsize the loop
            if (DataParams->PostProcessor != nullptr) {
                DataParams->PostProcessor->Process(
                    DataParams->C, RangeStartM, RangeStartN + n,
                    RangeCountM, CountN, ldc
                );
            }

            c_blk += ldc;
            a_row += lda;
        }
    }
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
    const auto QuantizeARow2 = GetMlasPlatform().SQNBitGemmDispatch->QuantizeARow_CompInt8_2;

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t QuantAStride = BlockCountK * Q8BlkSize(BlkLen);

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

struct Operations {
    InitializeWorkspaceFn* InitializeWorkspace = nullptr;
    SQNBitGemmFn* SQNBitGemm = nullptr;
};

constexpr auto OperationMap = []() {
    std::array<Operations, SQNBitGemmVariantCount> ops;

    ops[SQNBitGemmVariant_BitWidth4_CompFp32].SQNBitGemm = SQ4BitGemm_CompFp32;

    ops[SQNBitGemmVariant_BitWidth4_CompInt8].InitializeWorkspace = InitializeWorkspace_CompInt8;
    ops[SQNBitGemmVariant_BitWidth4_CompInt8].SQNBitGemm = SQ4BitGemm_CompInt8;

    return ops;
}();

void
ComputeParallelTasksSGemm(const size_t M, const size_t N, const size_t CountK, const size_t BatchN,
  MLAS_THREADPOOL* ThreadPool,
  size_t& ThreadCountM, size_t& ThreadCountN, size_t& ThreadsPerGemm)
{
    const double Complexity = double(M) * double(N) * double(CountK);

    ptrdiff_t TargetThreadCount;

    if (Complexity < double(MLAS_SGEMM_THREAD_COMPLEXITY * GetMlasPlatform().MaximumThreadCount)) {
        TargetThreadCount = ptrdiff_t(Complexity / double(MLAS_SGEMM_THREAD_COMPLEXITY)) + 1;
    } else {
        TargetThreadCount = GetMlasPlatform().MaximumThreadCount;
    }

    ptrdiff_t MaximumThreadCount = MlasGetMaximumThreadCount(ThreadPool) * 8;

    if (TargetThreadCount >= MaximumThreadCount) {
        TargetThreadCount = MaximumThreadCount;
    }

    //
    // Segment the operation across multiple threads.
    //
    // N.B. Currently, the operation is segmented as a 1D partition, which
    // works okay for operations involving skinny matrices.
    //

    ThreadsPerGemm = (TargetThreadCount + BatchN - 1) / BatchN;
    if (N > M) {
        const size_t BlockedN = (N + MLAS_SGEMM_STRIDEN_THREAD_ALIGN - 1) /
                                MLAS_SGEMM_STRIDEN_THREAD_ALIGN;

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
}
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
    //auto start_batch = std::chrono::high_resolution_clock::now();  // Start timing here

    const auto Variant = GetSQNBitGemmVariant(BlkBitWidth, BlkLen, ComputeType);
    assert(Variant != SQNBitGemmVariantInvalid);

    //
    // Ensure `Workspace` has correct alignment.
    //
    if (Workspace != nullptr) {
        const size_t Alignment = SQNBitGemmPerGemmWorkspaceAlignment(BlkBitWidth, BlkLen, ComputeType);
        const uintptr_t WorkspaceAddress = reinterpret_cast<uintptr_t>(Workspace);
        Workspace = reinterpret_cast<void*>(
            (WorkspaceAddress + Alignment - 1) & (~(Alignment - 1))
        );
    }

    const size_t PerGemmWorkspaceStride = SQNBitGemmPerGemmWorkspaceStride(M, N, K, BlkBitWidth, BlkLen, ComputeType);

    if (const auto InitializeWorkspaceOperation = OperationMap[Variant].InitializeWorkspace;
        InitializeWorkspaceOperation != nullptr) {
        //auto start = std::chrono::high_resolution_clock::now();  // Start timing here
        InitializeWorkspaceOperation(
            M, N, K, BatchN, BlkLen, DataParams, Workspace, PerGemmWorkspaceStride, ThreadPool
        );
        //auto end = std::chrono::high_resolution_clock::now();  // End timing here
        //// Calculate and print the duration in nanoseconds
        //std::chrono::duration<double, std::nano> elapsed = end - start;
        //std::cout << "InitializeWorkspaceOperation: " << elapsed.count() << " ns\n";
    }

    const auto ComputeOperation = OperationMap[Variant].SQNBitGemm;

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);

    if (/*true || */ThreadPool == nullptr) {
        //auto start = std::chrono::high_resolution_clock::now();  // Start timing here
        for (size_t gemm_i = 0; gemm_i < BatchN; gemm_i++) {
            const auto* Data = &DataParams[gemm_i];
            if (ComputeType == CompInt8) {
              // TODO: shall sepqrate QuantBBlkSum from QuantBData
                PackedQuantBDataStruct packed_quant_b(const_cast<void*>(Data->QuantBData), N, BlockCountK, BlkLen);
                const_cast<MLAS_SQNBIT_GEMM_DATA_PARAMS*>(Data)->QuantBBlkSum = packed_quant_b.QuantBBlkSum;
                void* PerGemmWorkspace =
                    reinterpret_cast<std::byte*>(Workspace) + gemm_i * PerGemmWorkspaceStride;
                PerGemmQuantAWorkspace per_gemm_quant_a_workspace(PerGemmWorkspace, M, BlockCountK, BlkLen);
#if defined(CALL_SGEMM_SEPARATELY)
                //SQ4BitGemm_CompInt8_0(BlkLen, K, Data, &per_gemm_quant_a_workspace, 0, M, 0, N);
#endif
                ComputeOperation(BlkLen, K, Data, &per_gemm_quant_a_workspace, 0, M, 0, N);
            } else {
                ComputeOperation(BlkLen, K, Data, nullptr, 0, M, 0, N);
            }
        }
        //auto end = std::chrono::high_resolution_clock::now();  // End timing here
        //// Calculate and print the duration in nanoseconds
        //std::chrono::duration<double, std::nano> elapsed = end - start;
        //std::cout << "ThreadPool == nullptr: " << elapsed.count() << " ns\n";

        return;
    }

    //
    // Compute the number of target threads given the complexity of the SGEMM
    // operation. Small requests should run using the single threaded path.
    //
    //auto start = std::chrono::high_resolution_clock::now();  // Start timing here

#if defined(CALL_SGEMM_SEPARATELY)
    if (ComputeType == CompInt8) {
        size_t ThreadCountM, ThreadCountN, ThreadsPerGemm;
        ComputeParallelTasksSGemm(M, N, BlockCountK, BatchN, ThreadPool,
          ThreadCountM, ThreadCountN, ThreadsPerGemm);

        //std::cout << "ThreadsPerGemm: " << ThreadsPerGemm << "\t"
        //          << "ThreadCountM: " << ThreadCountM << "\t"
        //          << "ThreadCountN: " << ThreadCountN << "\n";
        //auto start = std::chrono::high_resolution_clock::now();  // Start timing here
        MlasTrySimpleParallel(ThreadPool, ThreadsPerGemm * BatchN, [&](ptrdiff_t tid) {
            ptrdiff_t GemmIdx = tid / ThreadsPerGemm;
            ptrdiff_t ThreadIdx = tid % ThreadsPerGemm;

            // MlasSgemmThreaded
            ptrdiff_t ThreadId = ThreadIdx;
            const ptrdiff_t ThreadIdM = ThreadId / ThreadCountN;
            const ptrdiff_t ThreadIdN = ThreadId % ThreadCountN;

            //
            // Partition the operation along the M dimension.
            //

            size_t RangeStartM;
            size_t RangeCountM;

            MlasPartitionWork(ThreadIdM, ThreadCountM, M, &RangeStartM, &RangeCountM);

            //
            // Partition the operation along the N dimension.
            //

            size_t RangeStartN;
            size_t RangeCountN;

            const size_t BlockedN = (N + MLAS_SGEMM_STRIDEN_THREAD_ALIGN - 1) /
                                    MLAS_SGEMM_STRIDEN_THREAD_ALIGN;

            MlasPartitionWork(ThreadIdN, ThreadCountN, BlockedN, &RangeStartN, &RangeCountN);

            RangeStartN *= MLAS_SGEMM_STRIDEN_THREAD_ALIGN;
            RangeCountN *= MLAS_SGEMM_STRIDEN_THREAD_ALIGN;

            RangeCountN = std::min(N - RangeStartN, RangeCountN);

            const auto* Data = &DataParams[GemmIdx];

            PackedQuantBDataStruct packed_quant_b(const_cast<void*>(Data->QuantBData), N, BlockCountK, BlkLen);
            const_cast<MLAS_SQNBIT_GEMM_DATA_PARAMS*>(Data)->QuantBBlkSum = packed_quant_b.QuantBBlkSum;

            void* PerGemmWorkspace =
                reinterpret_cast<std::byte*>(Workspace) + GemmIdx * PerGemmWorkspaceStride;
            PerGemmQuantAWorkspace per_gemm_quant_a_workspace(PerGemmWorkspace, M, BlockCountK, BlkLen);
            //SQ4BitGemm_CompInt8_0(BlkLen, K, Data, &per_gemm_quant_a_workspace, RangeStartM, RangeCountM, RangeStartN, RangeCountN);
            //ComputeOperation(BlkLen, K, Data, &per_gemm_quant_a_workspace, RangeStartM, RangeCountM, RangeStartN, RangeCountN);
        });
    }
#endif
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

      //std::cout << "ThreadsPerGemm: " << ThreadsPerGemm << "\t"
      //          << "ThreadCountM: " << ThreadCountM << "\t"
      //          << "ThreadCountN: " << ThreadCountN << "\n";
    std::chrono::high_resolution_clock::time_point tp;
    if (profiler_->IsEnabled()) {
        tp = profiler_->Start();
    }

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

        if (ComputeType == CompInt8) {
            PackedQuantBDataStruct packed_quant_b(const_cast<void*>(Data->QuantBData), N, BlockCountK, BlkLen);
            const_cast<MLAS_SQNBIT_GEMM_DATA_PARAMS*>(Data)->QuantBBlkSum = packed_quant_b.QuantBBlkSum;

            void* PerGemmWorkspace =
                reinterpret_cast<std::byte*>(Workspace) + gemm_i * PerGemmWorkspaceStride;
            PerGemmQuantAWorkspace per_gemm_quant_a_workspace(PerGemmWorkspace, M, BlockCountK, BlkLen);
            ComputeOperation(BlkLen, K, Data, &per_gemm_quant_a_workspace, RangeStartM, RangeCountM, RangeStartN, RangeCountN);
        } else {
            ComputeOperation(BlkLen, K, Data, nullptr, RangeStartM, RangeCountM, RangeStartN, RangeCountN);
        }
    });
    if (profiler_->IsEnabled()) {
        std::string eventName = DataParams->node_name + "MlasTrySimpleParallel" + std::to_string(ThreadsPerGemm) + "-" + std::to_string(M) + "_" + std::to_string(N) + "_" + std::to_string(K);
        profiler_->EndTimeAndRecordEvent(onnxruntime::profiling::KERNEL_EVENT, eventName, tp);
    }

    // auto end = std::chrono::high_resolution_clock::now();  // End timing here
    //// Calculate and print the duration in nanoseconds
    // std::chrono::duration<double, std::nano> elapsed = end - start;
    // std::chrono::duration<double, std::nano> elapsed_batch = end - start_batch;

    // std::cout << "ThreadPool kernel: " << elapsed.count() << " ns\n";
    // std::cout << "Batch Internal: " << elapsed_batch.count() << " ns\n";
}
