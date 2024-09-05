/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx512.cpp.h

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication kernels for x64 avx512.

--*/

#include <algorithm>
#include <cassert>
#include <utility>

#include "sqnbitgemm.h"
#include "sqnbitgemm_kernel_avx_common.h"
#include "sqnbitgemm_kernel_avx_common_int8.h"
#include "sqnbitgemm_kernel_avx512_int8_blklen16.h"
#include "sqnbitgemm_kernel_avx512_int8_blklen32.h"
#include "sqnbitgemm_kernel_avx512_int8_blklen64.h"
#include "sqnbitgemm_kernel_avx512_int8_blklen128.h"

//
// CompFp32 kernel implementation.
//

#include "sqnbitgemm_kernel_avx_common_fp32.h"

MLAS_FORCEINLINE void
SQ4BitGemmM1Kernel_CompFp32_avx512(
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

//
// CompInt8 kernel implementation.
//

MLAS_FORCEINLINE
size_t
SQ4BitGemmKernel_BlkSum_CompInt8_avx512(
    const size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* /*QuantBZeroPoint*/,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t /*CountK*/,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ABlockSum,
    const float* QuantBBlkSum
)
{
    if (BlkLen == 16) {
        MlasQ4Int8GemmKernelBlkLen16Avx512<false>(
            QuantA,
            QuantAScale,
            QuantBData,
            QuantBScale,
            C,
            CountM,
            CountN,
            BlockCountK,
            Bias,
            ldc
        );
    } else if (BlkLen == 32) {
        MlasQ4Int8GemmKernelBlkLen32Avx512<false>(
            QuantA,
            QuantAScale,
            QuantBData,
            QuantBScale,
            C,
            CountM,
            CountN,
            BlockCountK,
            Bias,
            ldc
        );
    } else if (BlkLen == 64) {
        MlasQ4Int8GemmKernelBlkLen64Avx512<false>(
            BlkLen,
            QuantA,
            QuantAScale,
            QuantBData,
            QuantBScale,
            C,
            CountM,
            CountN,
            BlockCountK,
            Bias,
            ldc
        );
    } else {
        MlasQ4Int8GemmKernelBlkLen128Avx512<false>(
            BlkLen,
            QuantA,
            QuantAScale,
            QuantBData,
            QuantBScale,
            C,
            CountM,
            CountN,
            BlockCountK,
            Bias,
            ldc
        );
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

void MLASCALL
QuantizeARow_CompInt8_avx512(
    size_t BlkLen,
    const float* A,
    size_t CountK,
    std::byte* QuantA,
    float* QuantAScale,
    float* AScaledBlkSum  // scale_k * Sum_blklen(a_i)
)
{
    // port from MlasQ80BlkQuantRow
    assert(BlkLen % 16 == 0);
    const __m512 signBit = _mm512_set1_ps(-0.0f);
    const __m256i one_16_epi16 = _mm256_set1_epi16(1);
    int8_t* blob = reinterpret_cast<int8_t*>(QuantA);
    float* scale_ptr = QuantAScale;
    for (size_t k = 0; k < CountK; k += BlkLen) {
        const size_t step = std::min(BlkLen, CountK - k);

        __m512 maxAbs = _mm512_setzero_ps();
        for (size_t kk = 0; kk < step; kk += 16) {
            const size_t klen = std::min(size_t(16), step - kk);

            uint32_t mask = 0xffff >> (16 - klen);
            __m512 v0 = _mm512_maskz_loadu_ps(__mmask16(mask), A + k + kk);

            // Compute max(abs(e)) for the block
            maxAbs = _mm512_max_ps(maxAbs, _mm512_andnot_ps(signBit, v0));
        }

        __m256 max8 =
            _mm256_max_ps(_mm512_extractf32x8_ps(maxAbs, 1), _mm512_extractf32x8_ps(maxAbs, 0));
        __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(max8, 1), _mm256_castps256_ps128(max8));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
        const float maxScalar = _mm_cvtss_f32(max4);

        // Quantize these floats
        const float scale = maxScalar / 127.f;
        *scale_ptr = scale;
        scale_ptr++;

        const float inverse_scale = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
        const __m512 mul = _mm512_set1_ps(inverse_scale);
        __m128i* dst = reinterpret_cast<__m128i*>(blob);

        __m256i sum_16_epi16 = _mm256_setzero_si256();
        for (size_t kk = 0; kk < step; kk += 16) {
            const size_t klen = std::min(size_t(16), step - kk);

            uint32_t mask = 0xffff >> (16 - klen);
            __m512 v0 = _mm512_maskz_loadu_ps(__mmask16(mask), A + k + kk);
            v0 = _mm512_mul_ps(v0, mul);

            // Round to nearest integer
            v0 = _mm512_roundscale_ps(v0, _MM_ROUND_NEAREST);

            // Convert floats to integers
            __m512i i0 = _mm512_cvtps_epi32(v0);

            // Convert int32 to int8
            __m128i i0_8 = _mm512_cvtepi32_epi8(i0);
            _mm_storeu_si128(dst++, i0_8);

            // accumulate Sum(a_i)
            __m256i i_16_epi16 = _mm256_cvtepi8_epi16(i0_8);
            sum_16_epi16 = _mm256_hadds_epi16(sum_16_epi16, i_16_epi16);

        }
        if (step < BlkLen) {
            memset(blob + step, 0, BlkLen - step);
        }

        const __m256i sum_8_epi32 = _mm256_madd_epi16(one_16_epi16, sum_16_epi16);
        *AScaledBlkSum = scale * hsum_8_epi32(sum_8_epi32);
        AScaledBlkSum++;
        blob += BlkLen;
    }
}

static void
SQ4BitGemmPackQuantBDataAndBlkSum512(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const std::byte* QuantBDataBegin,
    const float* QuantBScaleBegin,
    bool has_zp_input,
    const std::byte* QuantBZPBegin,
    PackedQuantBDataStruct& packed_quant_b,
    MLAS_THREADPOOL* ThreadPool
)
{
    assert(BlkLen >= 16 && BlkLen % 16 == 0);

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);

    size_t SubBlkLen = (BlkLen == 16) ? 16 : (BlkLen == 32 ? 32 : 64);
    if (ComputeType == CompInt8) {
        SubBlkLen = 128;
    }
    PackQuantBDataAndBlkSum(N, BlockCountK, BlkLen, SubBlkLen, QuantBDataBegin, QuantBScaleBegin, has_zp_input, QuantBZPBegin, packed_quant_b, ThreadPool);
}

const MLAS_SQNBIT_GEMM_DISPATCH MlasSQNBitGemmDispatchAvx512 = []() {
    MLAS_SQNBIT_GEMM_DISPATCH d;

    d.SQ4BitGemmPackQuantBDataSize = SQ4BitGemmPackQuantBDataSize;
    d.SQ4BitGemmPackQuantBData = SQ4BitGemmPackQuantBData;
    d.SQ4BitGemmPackQuantBDataAndBlkSum = SQ4BitGemmPackQuantBDataAndBlkSum512;

    d.SQ4BitGemmPerGemmWorkspaceSize = SQ4BitGemmPerGemmWorkspaceSize;
    d.SQ4BitGemmPerGemmWorkspaceAlignment = SQ4BitGemmPerGemmWorkspaceAlignment;

    d.SQ4BitGemmM1Kernel_CompFp32 = SQ4BitGemmM1Kernel_CompFp32_avx512;
    d.Q4BitBlkDequantBForSgemm_CompFp32 = Q4BitBlkDequantBForSgemm_CompFp32_avx2;

    d.SQ4BitGemmKernel_BlkSum_CompInt8 = SQ4BitGemmKernel_BlkSum_CompInt8_avx512;
    d.QuantizeARowComputeBlkSum_CompInt8 = QuantizeARow_CompInt8_avx512;

    d.ConvertFp32ToFp16 = ConvertFp32ToFp16Avx;
    d.ConvertFp16ToFp32 = ConvertFp16ToFp32Avx;

    return d;
}();
