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

__m512
ComputeMulScal(const float* a_ptr, size_t step, float& scale)
{
    const __m512 signBit = _mm512_set1_ps(-0.0f);
    __m512 maxAbs = _mm512_setzero_ps();

    for (size_t kk = 0; kk < step; kk += 16) {
        const size_t klen = std::min(size_t(16), step - kk);

        uint32_t mask = 0xffff >> (16 - klen);
        __m512 v0 = _mm512_maskz_loadu_ps(__mmask16(mask), a_ptr + kk);

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
    scale = maxScalar / 127.f;

    const float inverse_scale = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
    return _mm512_set1_ps(inverse_scale);
}

void
QuantizeInt8ComputeBlksum(const float* a_ptr, size_t step, __m512& mul, float scale, __m256i& i0_32_epi8, float& blksum)
{
    const __m256i one_16_epi16 = _mm256_set1_epi16(1);
    __m256i sum_16_epi16 = _mm256_setzero_si256();
    __m128i i_16_epi8[2] = {_mm_setzero_si128(), _mm_setzero_si128()};
    int index = 0;
    for (size_t kk = 0; kk < step; kk += 16, index++) {
        const size_t klen = std::min(size_t(16), step - kk);

        uint32_t mask = 0xffff >> (16 - klen);
        __m512 v0 = _mm512_maskz_loadu_ps(__mmask16(mask), a_ptr + kk);
        v0 = _mm512_mul_ps(v0, mul);

        // Round to nearest integer
        v0 = _mm512_roundscale_ps(v0, _MM_ROUND_NEAREST);

        // Convert floats to integers
        __m512i i0 = _mm512_cvtps_epi32(v0);

        // Convert int32 to int8
        i_16_epi8[index] = _mm512_cvtepi32_epi8(i0);
        //_mm_storeu_si128(dst++, i0_8);

        // accumulate Sum(a_i)
        __m256i i_16_epi16 = _mm256_cvtepi8_epi16(i_16_epi8[index]);
        sum_16_epi16 = _mm256_hadds_epi16(sum_16_epi16, i_16_epi16);
    }
    i0_32_epi8 = _mm256_set_m128i(i_16_epi8[1], i_16_epi8[0]);
    const __m256i sum_8_epi32 = _mm256_madd_epi16(one_16_epi16, sum_16_epi16);
    blksum = scale * hsum_8_epi32(sum_8_epi32);
}

void
Quantize1BlkBlkLen32(const float* a_ptr, size_t step, __m256i& i_32_epi8, float& scale, float& blksum)
{
    // 32 float to 32 epi8s in i0_32_epi8
    __m512 mul = ComputeMulScal(a_ptr, step, scale);
    QuantizeInt8ComputeBlksum(a_ptr, step, mul, scale, i_32_epi8, blksum);
}

void
store_4blk_blklen32_interleaved(__m256i i_32_epi8[4], int8_t* blob)
{
    // 0   1   2   3  32  33  34  35  64  65  66  67  96  97  98  99
    // 4   5   6   7  36  37  38  39  68  69  70  71 100 101 102 103
    // 8   9  10  11  40  41  42  43  72  73  74  75 104 105 106 107
    // 12  13  14  15  44  45  46  47  76  77  78  79 108 109 110 111
    //
    // 16  17  18  19  48  49  50  51  80  81  82  83 112 113 114 115
    // 20  21  22  23  52  53  54  55  84  85  86  87 116 117 118 119
    // 24  25  26  27  56  57  58  59  88  89  90  91 120 121 122 123
    // 28  29  30  31  60  61  62  63  92  93  94  95 124 125 126 127

    // Interleave and store i_32_epi8[4] in the specified layout
    __m256i a0_lower = _mm256_permute2x128_si256(i_32_epi8[0], i_32_epi8[1], 0x20);
    __m256i a0_higher = _mm256_permute2x128_si256(i_32_epi8[0], i_32_epi8[1], 0x31);
    __m256i a1_lower = _mm256_permute2x128_si256(i_32_epi8[2], i_32_epi8[3], 0x20);
    __m256i a1_higher = _mm256_permute2x128_si256(i_32_epi8[2], i_32_epi8[3], 0x31);

    __m512i a_lower = _mm512_inserti64x4(_mm512_castsi256_si512(a0_lower), a1_lower, 1);
    __m512i a_higher = _mm512_inserti64x4(_mm512_castsi256_si512(a0_higher), a1_higher, 1);

    __m512i idx = _mm512_set_epi32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
    __m512i a_lower_interleaved = _mm512_permutexvar_epi32(idx, a_lower);
    __m512i a_higher_interleaved = _mm512_permutexvar_epi32(idx, a_higher);

    _mm512_storeu_si512(reinterpret_cast<__m512i*>(blob + 0 * 64), a_lower_interleaved);
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(blob + 1 * 64), a_higher_interleaved);
}

void MLASCALL
QuantizeARow_CompInt8_avx512_blklen32(
    const float* A,
    size_t CountK,
    std::byte* QuantA,
    float* QuantAScale,
    float* AScaledBlkSum  // scale_k * Sum_blklen(a_i)
)
{
    const size_t BlkLen = 32;
    const int64_t SubBlkLen = 4 * BlkLen;  // process 128 weights at a time and then process the remaining weights

    const float* a_ptr = A;
    int8_t* quant_a_ptr = reinterpret_cast<int8_t*>(QuantA);
    float* scale_ptr = QuantAScale;
    float* blksum_ptr = AScaledBlkSum;

    int k_remaining = (int)CountK;

    for (; k_remaining >= SubBlkLen; k_remaining -= SubBlkLen) {
        __m256i i_32_epi8[4];
        float scale[4];
        float blksum[4];
        for (int i = 0; i < 4; i++) {
            Quantize1BlkBlkLen32(a_ptr, BlkLen, i_32_epi8[i], scale[i], blksum[i]);
            a_ptr += BlkLen;
        }
        store_4blk_blklen32_interleaved(i_32_epi8, quant_a_ptr);
        quant_a_ptr += BlkLen * 4;
        std::copy(scale, scale + 4, scale_ptr);
        scale_ptr += 4;
        std::copy(blksum, blksum + 4, blksum_ptr);
        blksum_ptr += 4;
    }

    while (k_remaining > 0) {
        // for (size_t k = 0; k < CountK; k += BlkLen) {
        __m256i i_32_epi8;
        float scale;
        float blksum;
        const size_t step = std::min(BlkLen, (size_t)k_remaining);
        Quantize1BlkBlkLen32(a_ptr, step, i_32_epi8, scale, blksum);
        _mm256_storeu_epi8(quant_a_ptr, i_32_epi8);
        a_ptr += BlkLen;
        quant_a_ptr += BlkLen;
        *scale_ptr = scale;
        scale_ptr++;
        *blksum_ptr = blksum;
        blksum_ptr++;
        k_remaining -= BlkLen;
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
)
{
    if (BlkLen == 32) {
        QuantizeARow_CompInt8_avx512_blklen32(A, CountK, QuantA, QuantAScale, AScaledBlkSum);
        return;
    }
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

    return d;
}();
