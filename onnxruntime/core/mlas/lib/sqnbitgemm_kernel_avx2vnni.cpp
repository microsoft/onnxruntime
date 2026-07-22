/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx2vnni.cpp

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication kernels for x64 avx2+avxvnni.

    AVX-VNNI-specific kernels and dispatch table.  This translation unit is
    compiled with -mavxvnni (GCC/Clang) so the auto-vectorizer may emit
    AVX-VNNI instructions throughout.  It must only be loaded on CPUs that
    support AVX-VNNI; the AVX2-only fallback lives in
    sqnbitgemm_kernel_avx2.cpp which is compiled without -mavxvnni.

--*/

#include <algorithm>
#include <cassert>
#include <utility>

#include "qnbitgemm.h"
#include "sqnbitgemm_kernel_avx_common.h"
#include "sqnbitgemm_kernel_avx_common_int8.h"
#include "sqnbitgemm_kernel_avx2_int8_blklen16.h"
#include "sqnbitgemm_kernel_avx2_int8_blklen32.h"
#include "sqnbitgemm_kernel_avx2_int8_blklen64.h"

#include "sqnbitgemm_kernel_avx2_2bit.h"
#include "sqnbitgemm_kernel_avx2_2bit_blklen32.h"
#include "sqnbitgemm_kernel_avx2_2bit_blklen64.h"
#include "sqnbitgemm_kernel_avx2_2bit_blklen128.h"

#include "sqnbitgemm_m1_sym_kernel_avx2_int8_blklen32.h"
#include "sqnbitgemm_m1_sym_kernel_avx2_int8_blklen64.h"

//
// Forward declarations for functions defined in sqnbitgemm_kernel_avx2.cpp
// that are referenced from the AVX2-VNNI dispatch table.
//
void
SQ4BitGemmM1Kernel_CompFp32_avx2(
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
);

void
Q4BitBlkDequantBForSgemm_CompFp32_avx2(
    const size_t BlkLen,
    float* FpData,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    const size_t CountN,
    const size_t CountK,
    const size_t BlockStrideQuantB
);

void MLASCALL
QuantizeARow_CompInt8_avx2(
    size_t BlkLen,
    const float* A,
    size_t CountK,
    std::byte* QuantA,
    float* QuantAScale,
    float* AScaledBlkSum
);

void MLASCALL
QuantizeARow_CompInt8_Fp16_avx2(
    size_t BlkLen,
    const MLAS_FP16* A,
    size_t CountK,
    std::byte* QuantA,
    float* QuantAScale,
    float* AScaledBlkSum
);

//
// Local packer functions (mirrors of the static functions in
// sqnbitgemm_kernel_avx2.cpp; identical implementation, separate static copy).
//

static void
SQ4BitGemmPackQuantBDataAndBlkSumVnni(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const std::byte* QuantBDataBegin,
    const float* QuantBScaleBegin,
    bool HasZeroPoint,
    const std::byte* QuantBZPBegin,
    PackedQuantBDataStruct<float, 4>& PackedQuantB,
    MLAS_THREADPOOL* ThreadPool,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*BackendKernelSelectorConfig*/
)
{
    assert(BlkLen >= 16 && BlkLen % 16 == 0);

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);

    size_t SubBlkLen = (BlkLen == 16) ? 16 : (BlkLen == 32 ? 32 : 64);
    if (BlkLen == 32 && ComputeType == SQNBIT_CompInt8) {
        SubBlkLen = 64;
    }
    PackQuantBDataAndBlkSum(N, BlockCountK, BlkLen, SubBlkLen, QuantBDataBegin, QuantBScaleBegin,
        HasZeroPoint, QuantBZPBegin, PackedQuantB, ThreadPool);
}

static void
SQ8BitGemmPackQuantBDataAndBlkSumVnni(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const std::byte* QuantBDataBegin,
    const float* QuantBScaleBegin,
    bool HasZeroPoint,
    const std::byte* QuantBZPBegin,
    PackedQuantBDataStruct<float, 8>& PackedQuantB,
    MLAS_THREADPOOL* ThreadPool,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*BackendKernelSelectorConfig*/
)
{
    assert(BlkLen >= 16 && BlkLen % 16 == 0);

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);

    size_t SubBlkLen = (BlkLen == 16) ? 16 : (BlkLen == 32 ? 32 : 64);
    if (ComputeType == SQNBIT_CompInt8) {
        SubBlkLen = 64;
    }
    Q8PackQuantBDataAndBlkSum(N, BlockCountK, BlkLen, SubBlkLen, QuantBDataBegin, QuantBScaleBegin,
        HasZeroPoint, QuantBZPBegin, PackedQuantB, ThreadPool);
}

//
// AVX2-VNNI kernel implementations.
//

size_t
SQ4BitGemmKernel_BlkSum_CompInt8_avx2vnni(
    const size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ABlockSum,
    const float* QuantBBlkSum
)
{
    if (BlkLen >= 32 && CountM == 1) {
        // Inline SQ4BitGemmM1Kernel_CompInt8_avx2<true>
        if (QuantBZeroPoint) {
            if (BlkLen == 32) {
                MlasQ4Int8GemmM1KernelBlkLen32Avx2<true, true>(
                    QuantA, QuantAScale, QuantBData, QuantBScale,
                    QuantBZeroPoint, C, CountN, BlockCountK, Bias);
            } else {
                MlasQ4Int8GemmKernelBlkLen64Avx2<true>(
                    BlkLen, QuantA, QuantAScale, QuantBData, QuantBScale,
                    QuantBZeroPoint, C, CountN, BlockCountK, Bias);
            }
        } else {
            if (BlkLen == 32) {
                MlasQ4Int8GemmM1KernelBlkLen32Avx2<false, true>(
                    QuantA, QuantAScale, QuantBData, QuantBScale,
                    QuantBZeroPoint, C, CountN, BlockCountK, Bias);
            } else {
                MlasQ4Int8GemmKernelBlkLen64Avx2<false>(
                    BlkLen, QuantA, QuantAScale, QuantBData, QuantBScale,
                    QuantBZeroPoint, C, CountN, BlockCountK, Bias);
            }
        }
        return CountM;
    }

    // Inline SQ4BitGemmKernel_CompInt8_avx2<true>
    if (BlkLen == 16) {
        MlasQ4Int8GemmKernelBlkLen16Avx2(
            QuantA, QuantAScale, QuantBData, QuantBScale,
            C, CountM, CountN, CountK, BlockCountK, Bias, ldc);
    } else if (BlkLen == 32) {
        MlasQ4Int8GemmKernelBlkLen32Avx2<true>(
            QuantA, QuantAScale, QuantBData, QuantBScale,
            C, CountM, CountN, CountK, BlockCountK, Bias, ldc);
    } else {
        MlasQ4Int8GemmKernelBlkLen64Avx2<true>(
            BlkLen, QuantA, QuantAScale, QuantBData, QuantBScale,
            C, CountM, CountN, BlockCountK, Bias, ldc);
    }

    float* c_blk = C;
    const float* b_blk_sum = QuantBBlkSum;

    size_t RowsRemaining = CountM;
    const float* a_blksum_row = ABlockSum;
    while (RowsRemaining > 0) {
        auto RowsHandled = GetMlasPlatform().GemmFloatKernel(
            a_blksum_row, b_blk_sum, c_blk, BlockCountK, RowsRemaining, CountN, BlockCountK, ldc, 1.f, false
        );

        c_blk += ldc * RowsHandled;
        a_blksum_row += BlockCountK * RowsHandled;
        RowsRemaining -= RowsHandled;
    }
    return CountM;
}

size_t
SQ8BitGemmKernel_BlkSum_CompInt8_avx2vnni(
    const size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* /*QuantBZeroPoint*/,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ABlockSum,
    const float* QuantBBlkSum,
    const float* /*QuantBBlkSum2*/
)
{
    if (BlkLen == 16) {
        MlasQ8Int8GemmKernelBlkLen16Avx2<true>(
            QuantA, QuantAScale, QuantBData, QuantBScale,
            C, CountM, CountN, CountK, BlockCountK, Bias, ldc);
    } else if (BlkLen == 32) {
        MlasQ8Int8GemmKernelBlkLen32Avx2<true>(
            QuantA, QuantAScale, QuantBData, QuantBScale,
            C, CountM, CountN, CountK, BlockCountK, Bias, ldc);
    } else {
        MlasQ8Int8GemmKernelBlkLen64Avx2<true>(
            BlkLen, QuantA, QuantAScale, QuantBData, QuantBScale,
            C, CountM, CountN, BlockCountK, Bias, ldc);
    }

    float* c_blk = C;
    const float* b_blk_sum = QuantBBlkSum;

    size_t RowsRemaining = CountM;
    const float* a_blksum_row = ABlockSum;
    while (RowsRemaining > 0) {
        auto RowsHandled = GetMlasPlatform().GemmFloatKernel(
            a_blksum_row, b_blk_sum, c_blk, BlockCountK, RowsRemaining, CountN, BlockCountK, ldc, 1.f, false
        );

        c_blk += ldc * RowsHandled;
        a_blksum_row += BlockCountK * RowsHandled;
        RowsRemaining -= RowsHandled;
    }
    return CountM;
}

//
// BlkLen-routing wrapper for the W2 CompInt8 AVX2-VNNI dispatch entry.
// Sibling of SQ2BitGemmKernel_BlkSum_CompInt8_Avx2_Dispatch in
// sqnbitgemm_kernel_avx2.cpp.
//
namespace onnxruntime::mlas::sq2bit_avx2 {
size_t MLASCALL
SQ2BitGemmKernel_BlkSum_CompInt8_Avx2Vnni_Dispatch(
    size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ABlockSum,
    const float* QuantBBlkSum)
{
    if (BlkLen == 128) {
        return SQ2BitGemmKernel_BlkSum_CompInt8_BlkLen128_Avx2Vnni(
            QuantA, QuantAScale, QuantBData, QuantBScale,
            C, CountM, CountN, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum);
    }
    if (BlkLen == 32) {
        return SQ2BitGemmKernel_BlkSum_CompInt8_BlkLen32_Avx2Vnni(
            QuantA, QuantAScale, QuantBData, QuantBScale,
            C, CountM, CountN, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum);
    }
    return SQ2BitGemmKernel_BlkSum_CompInt8_Avx2Vnni(
        BlkLen, QuantA, QuantAScale, QuantBData, QuantBScale, QuantBZeroPoint,
        C, CountM, CountN, CountK, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum);
}
}  // namespace onnxruntime::mlas::sq2bit_avx2

//
// Kernel dispatch structure definition.
//
const MLAS_QNBIT_GEMM_DISPATCH MlasSQNBitGemmDispatchAvx2vnni = []() {
    MLAS_QNBIT_GEMM_DISPATCH d;

    d.Q4BitGemmPackQuantBDataSize = QNBitGemmPackQuantBDataSize<4>;
    d.Q8BitGemmPackQuantBDataSize = QNBitGemmPackQuantBDataSize<8>;
    d.SQ4BitGemmPackQuantBData = SQ4BitGemmPackQuantBData;
    d.SQ4BitGemmPackQuantBDataAndBlkSum = SQ4BitGemmPackQuantBDataAndBlkSumVnni;
    d.SQ8BitGemmPackQuantBDataAndBlkSum = SQ8BitGemmPackQuantBDataAndBlkSumVnni;

    d.QNBitGemmPerGemmWorkspaceSize = QNBitGemmPerGemmWorkspaceSize;
    d.QNBitGemmPerGemmWorkspaceAlignment = QNBitGemmPerGemmWorkspaceAlignment;

    d.SQ4BitGemmM1Kernel_CompFp32 = SQ4BitGemmM1Kernel_CompFp32_avx2;
    d.SQ4BitBlkDequantBForSgemm_CompFp32 = Q4BitBlkDequantBForSgemm_CompFp32_avx2;

    d.SQ4BitGemmKernel_BlkSum_CompInt8 = SQ4BitGemmKernel_BlkSum_CompInt8_avx2vnni;
    d.SQ8BitGemmKernel_BlkSum_CompInt8 = SQ8BitGemmKernel_BlkSum_CompInt8_avx2vnni;
    d.QuantizeARowComputeBlkSum_CompInt8 = QuantizeARow_CompInt8_avx2;
    d.QuantizeARowComputeBlkSum_CompInt8_Fp16 = QuantizeARow_CompInt8_Fp16_avx2;

    // 2-bit native CompInt8 path (AVX-VNNI compute).
    static_assert(
        onnxruntime::mlas::sq2bit_avx512::kBlockGroupBlks == kSq2BitAvx512WeightKBlockGroup,
        "kBlockGroupBlks (kernel-internal) must match kSq2BitAvx512WeightKBlockGroup (qnbitgemm.h).");
    d.Q2BitGemmPackQuantBDataSize       = onnxruntime::mlas::sq2bit_avx512::Q2BitGemmPackQuantBDataSize_Avx512;
    d.SQ2BitGemmPackQuantBDataAndBlkSum = onnxruntime::mlas::sq2bit_avx512::SQ2BitGemmPackQuantBDataAndBlkSum_Scalar;
    d.SQ2BitGemmKernel_BlkSum_CompInt8  = onnxruntime::mlas::sq2bit_avx2::SQ2BitGemmKernel_BlkSum_CompInt8_Avx2Vnni_Dispatch;
    d.Q2BitGemmEffectiveBlockCountK     = onnxruntime::mlas::sq2bit_avx512::Q2BitGemmEffectiveBlockCountK;

    return d;
}();
