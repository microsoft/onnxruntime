/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm.cpp

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication hardware agnostic entrypoint, MlasSQNBitGemmBatch.
--*/

#include "sqnbitgemm.h"

namespace
{

// Get quantization variant based on `BlkBitWidth` and `BlkLen`.
// Return -1 if the input values are unsupported.
int32_t
GetDispatchQuantVariant(size_t BlkBitWidth, size_t BlkLen)
{
    int32_t type = -1;
    if (BlkBitWidth == 4 && BlkLen == 16) {
        type = QuantVariant_BitWidth4_BlockSize16;
    } else if (BlkBitWidth == 4 && BlkLen == 32) {
        type = QuantVariant_BitWidth4_BlockSize32;
    } else if (BlkBitWidth == 4 && BlkLen == 64) {
        type = QuantVariant_BitWidth4_BlockSize64;
    } else if (BlkBitWidth == 4 && BlkLen == 128) {
        type = QuantVariant_BitWidth4_BlockSize128;
    }

    return type;
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
    const MLAS_SQNBIT_GEMM_DATA_PARAMS* DataParams,
    MLAS_THREADPOOL* ThreadPool
)
{
    const int32_t QuantVariant = GetDispatchQuantVariant(BlkBitWidth, BlkLen);
    MLAS_SQNBIT_GEMM_OPERATION* const Operation = GetMlasPlatform().SQNBitGemmDispatch->Operations[QuantVariant];

    if (ThreadPool == nullptr) {
        for (size_t gemm_i = 0; gemm_i < BatchN; gemm_i++) {
            auto Data = &DataParams[gemm_i];
            Operation(K, Data, 0, M, 0, N);
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
        auto Data = &DataParams[gemm_i];

        const ptrdiff_t ThreadIdN = blk_i / ThreadCountM;
        const ptrdiff_t ThreadIdM = blk_i % ThreadCountM;

        const size_t RangeStartM = ThreadIdM * StrideM;
        const size_t RangeCountM = std::min(M - RangeStartM, (size_t)StrideM);

        const size_t RangeStartN = ThreadIdN * StrideN;
        const size_t RangeCountN = std::min(N - RangeStartN, (size_t)StrideN);

        Operation(K, Data, RangeStartM, RangeCountM, RangeStartN, RangeCountN);
    });
}

bool MLASCALL
MlasIsSQNBitGemmAvailable(
    size_t BlkBitWidth,
    size_t BlkLen
)
{
    const int32_t QuantVariant = GetDispatchQuantVariant(BlkBitWidth, BlkLen);
    if (QuantVariant == -1) {
        return false;
    }

    if (GetMlasPlatform().SQNBitGemmDispatch == nullptr ||
        GetMlasPlatform().SQNBitGemmDispatch->Operations[QuantVariant] == nullptr) {
        return false;
    }

    return true;
}
