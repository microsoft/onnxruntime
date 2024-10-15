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

void MLASCALL
MlasQ80BlkQuantRow_avx512(
    size_t BlkLen,
    const float* A,
    size_t CountK,
    std::byte* QuantA
)
{
    // port from MlasQ80BlkQuantRow
    assert(BlkLen % 16 == 0);
    const __m512 signBit = _mm512_set1_ps(-0.0f);
    int8_t* blob = reinterpret_cast<int8_t*>(QuantA);
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
        *reinterpret_cast<float*>(blob) = scale;
        blob += sizeof(float);

        const float inverse_scale = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
        const __m512 mul = _mm512_set1_ps(inverse_scale);
        __m128i* dst = reinterpret_cast<__m128i*>(blob);

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
        }
        if (step < BlkLen) {
            memset(blob + step, 0, BlkLen - step);
        }
        blob += BlkLen;
    }
}

void MLASCALL
QuantizeARow_CompInt8_avx512(
    size_t BlkLen,
    const float* A,
    size_t CountK,
    std::byte* QuantA
)
{
    MlasQ80BlkQuantRow_avx512(BlkLen, A, CountK, QuantA);
}

const MLAS_SQNBIT_GEMM_DISPATCH MlasSQNBitGemmDispatchAvx512 = []() {
    MLAS_SQNBIT_GEMM_DISPATCH d;

    d.SQ4BitGemmPackQuantBDataSize = SQ4BitGemmPackQuantBDataSize;
    d.SQ4BitGemmPackQuantBData = SQ4BitGemmPackQuantBData;

    d.SQ4BitGemmPerGemmWorkspaceSize = SQ4BitGemmPerGemmWorkspaceSize;
    d.SQ4BitGemmPerGemmWorkspaceAlignment = SQ4BitGemmPerGemmWorkspaceAlignment;

    d.SQ4BitGemmM1Kernel_CompFp32 = SQ4BitGemmM1Kernel_CompFp32_avx512;
    d.Q4BitBlkDequantBForSgemm_CompFp32 = Q4BitBlkDequantBForSgemm_CompFp32_avx2;

    d.SQ4BitGemmM1Kernel_CompInt8 = SQ4BitGemmM1Kernel_CompInt8_avx2;
    d.QuantizeARow_CompInt8 = QuantizeARow_CompInt8_avx512;

    return d;
}();
