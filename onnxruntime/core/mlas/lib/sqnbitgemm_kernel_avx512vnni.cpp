/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx512.cpp.h

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication kernels for x64 avx512vnni.

--*/

#include <algorithm>
#include <cassert>
#include <utility>

#include "sqnbitgemm.h"
#include "sqnbitgemm_kernel_avx_common.h"
#include "sqnbitgemm_kernel_avx_common_fp32.h"
#include "sqnbitgemm_kernel_avx_common_int8.h"

MLAS_FORCEINLINE void
SQ4BitGemmM1Kernel_CompFp32(
    size_t BlkLen,
    const float* A,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t CountK,
    size_t BlockStrideQuantB,
    const float* Bias
)
{
    if (BlkLen == 16) {
        if (QuantBZeroPoint != nullptr) {
            MlasQ4GemmKernelBlkLen16Avx512f<true>(
                A,
                QuantBData,
                QuantBScale,
                QuantBZeroPoint,
                C,
                1,
                CountN,
                CountK,
                BlockStrideQuantB,
                Bias,
                0,
                0
            );
        } else {
            MlasQ4GemmKernelBlkLen16Avx512f<false>(
                A,
                QuantBData,
                QuantBScale,
                QuantBZeroPoint,
                C,
                1,
                CountN,
                CountK,
                BlockStrideQuantB,
                Bias,
                0,
                0
            );
        }
    } else if (BlkLen == 32) {
        if (QuantBZeroPoint != nullptr) {
            MlasQ4GemmKernelBlkLen32PlusAvx512f<true, false>(
                BlkLen,
                A,
                QuantBData,
                QuantBScale,
                QuantBZeroPoint,
                C,
                1,
                CountN,
                CountK,
                BlockStrideQuantB,
                Bias,
                0,
                0
            );
        } else {
            MlasQ4GemmKernelBlkLen32PlusAvx512f<false, false>(
                BlkLen,
                A,
                QuantBData,
                QuantBScale,
                QuantBZeroPoint,
                C,
                1,
                CountN,
                CountK,
                BlockStrideQuantB,
                Bias,
                0,
                0
            );
        }
    } else /*if (BlkLen >= 64)*/ {
        if (QuantBZeroPoint != nullptr) {
            MlasQ4GemmKernelBlkLen32PlusAvx512f<true, true>(
                BlkLen,
                A,
                QuantBData,
                QuantBScale,
                QuantBZeroPoint,
                C,
                1,
                CountN,
                CountK,
                BlockStrideQuantB,
                Bias,
                0,
                0
            );
        } else {
            MlasQ4GemmKernelBlkLen32PlusAvx512f<false, true>(
                BlkLen,
                A,
                QuantBData,
                QuantBScale,
                QuantBZeroPoint,
                C,
                1,
                CountN,
                CountK,
                BlockStrideQuantB,
                Bias,
                0,
                0
            );
        }
    }
}

MLAS_FORCEINLINE
void
SQ4BitGemmM1Kernel_CompInt8_avx512vnni(
    size_t BlkLen,
    const std::byte* QuantA,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t CountK,
    size_t BlockStrideQuantB,
    const float* Bias
)
{
    if (QuantBZeroPoint != nullptr) {
        constexpr bool HasZeroPoint = true;
        if (BlkLen == 16) {
            SQ4BitGemmM1Kernel_BlkLen16_CompInt8_Impl<HasZeroPoint>(
                QuantA,
                QuantBData,
                QuantBScale,
                QuantBZeroPoint,
                C,
                CountN,
                CountK,
                BlockStrideQuantB,
                Bias
            );
        } else if (BlkLen == 32) {
            SQ4BitGemmM1Kernel_BlkLen32_CompInt8_Impl<HasZeroPoint, accumulate_mul_sum_avx512vnni<HasZeroPoint>>(
                QuantA,
                QuantBData,
                QuantBScale,
                QuantBZeroPoint,
                C,
                CountN,
                BlockStrideQuantB,
                Bias
            );
        } else {
            SQ4BitGemmM1Kernel_BlkLen64Plus_CompInt8_Impl<HasZeroPoint, dot_quad_avx512vnni>(
                BlkLen,
                QuantA,
                QuantBData,
                QuantBScale,
                QuantBZeroPoint,
                C,
                CountN,
                CountK,
                BlockStrideQuantB,
                Bias
            );
        }
    } else {
        constexpr bool HasZeroPoint = false;
        if (BlkLen == 16) {
            SQ4BitGemmM1Kernel_BlkLen16_CompInt8_Impl<HasZeroPoint>(
                QuantA,
                QuantBData,
                QuantBScale,
                QuantBZeroPoint,
                C,
                CountN,
                CountK,
                BlockStrideQuantB,
                Bias
            );
        } else if (BlkLen == 32) {
            SQ4BitGemmM1Kernel_BlkLen32_CompInt8_Impl<HasZeroPoint, accumulate_mul_sum_avx512vnni<HasZeroPoint>>(
                QuantA,
                QuantBData,
                QuantBScale,
                QuantBZeroPoint,
                C,
                CountN,
                BlockStrideQuantB,
                Bias
            );
        } else {
            SQ4BitGemmM1Kernel_BlkLen64Plus_CompInt8_Impl<HasZeroPoint, dot_quad_avx512vnni>(
                BlkLen,
                QuantA,
                QuantBData,
                QuantBScale,
                QuantBZeroPoint,
                C,
                CountN,
                CountK,
                BlockStrideQuantB,
                Bias
            );
        }
    }
}

void MLASCALL
QuantizeARow_CompInt8_avx512(
    size_t BlkLen,
    const float* A,
    size_t CountK,
    std::byte* QuantA,
    float* QuantAScale,
    float* AScaledBlkSum  // scale_k * Sum_blklen(a_i)
);

static void
SQ4BitGemmPackQuantBDataAndBlkSum(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const std::byte* QuantBDataBegin,
    std::byte* PackedQuantBDataBegin,
    const float* QuantBScaleBegin,
    bool /*has_zp_input*/,
    const std::byte* QuantBZPBegin,
    float* BlockSumBegin,
    MLAS_THREADPOOL* ThreadPool
)
{
    assert(BlkLen >= 16 && BlkLen % 16 == 0);

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);

    size_t SubBlkLen = (BlkLen == 16) ? 16 : (BlkLen == 32 ? 32 : 64);
    if (BlkLen == 32 && ComputeType == CompInt8) {
        SubBlkLen = 64;
    }

    PackQuantB(QuantBDataBegin, PackedQuantBDataBegin, ThreadPool, N, BlockCountK, BlkLen, SubBlkLen);

    if (QuantBScaleBegin) {
        ComputePackBlkSum(N, QuantBScaleBegin, QuantBZPBegin, BlockSumBegin, ThreadPool, BlockCountK);
    }
}

//
// Kernel dispatch structure definition.
//
const MLAS_SQNBIT_GEMM_DISPATCH MlasSQNBitGemmDispatchAvx512vnni = []() {
    MLAS_SQNBIT_GEMM_DISPATCH d;

    d.SQ4BitGemmPackQuantBDataSize = SQ4BitGemmPackQuantBDataSize;
    d.SQ4BitGemmPackQuantBData = nullptr;
    d.SQ4BitGemmPackQuantBDataAndBlkSum = SQ4BitGemmPackQuantBDataAndBlkSum;

    d.SQ4BitGemmPerGemmWorkspaceSize = SQ4BitGemmPerGemmWorkspaceSize;
    d.SQ4BitGemmPerGemmWorkspaceAlignment = SQ4BitGemmPerGemmWorkspaceAlignment;

    d.SQ4BitGemmM1Kernel_CompFp32 = SQ4BitGemmM1Kernel_CompFp32;
    d.Q4BitBlkDequantBForSgemm_CompFp32 = Q4BitBlkDequantBForSgemm_CompFp32_avx2;

    d.SQ4BitGemmM1Kernel_CompInt8 = SQ4BitGemmM1Kernel_CompInt8_avx512vnni;
    d.QuantizeARow_CompInt8 = nullptr;
    d.QuantizeARow_CompInt8_2 = QuantizeARow_CompInt8_avx512;

    return d;
}();
