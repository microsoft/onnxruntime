/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx2.cpp.h

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication kernels for x64 avx2.

--*/

#include <algorithm>
#include <cassert>
#include <utility>

#include "qnbitgemm.h"

static MLAS_FORCEINLINE float
hsum_float_8(const __m256 x)
{
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static inline int
nearest_int(float fval)
{
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4244 4267)
#endif

void
quantize_row_q8_K_ref(const float* x, block_q8_K* y, int64_t k)
{
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        float max = 0;
        float amax = 0;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax;
                max = x[j];
            }
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, QK_K);
            x += QK_K;
            continue;
        }
        // const float iscale = -128.f/max;
        //  We need this change for IQ2_XXS, else the AVX implementation becomes very awkward
        const float iscale = -127.f / max;
        for (int j = 0; j < QK_K; ++j) {
            int v = nearest_int(iscale * x[j]);
            y[i].qs[j] = MIN(127, v);
        }
        for (int j = 0; j < QK_K / 16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += y[i].qs[j * 16 + ii];
            }
            y[i].bsums[j] = sum;
        }
        y[i].d = 1 / iscale;
        x += QK_K;
    }
}

void
dequantize_row_q8_K(const block_q8_K* x, float* y, int64_t k)
{
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < QK_K; ++j) {
            *y++ = x[i].d * x[i].qs[j];
        }
    }
}

void
QuantizeARow_Q8_K(
    size_t /*BlkLen*/,
    const float* A,
    size_t CountK,
    std::byte* QuantA
)
{
    block_q8_K* y = reinterpret_cast<block_q8_K*>(QuantA);
    // QuantA point to M * CountKBlk of block_q8_K
    quantize_row_q8_K_ref(A, y, CountK);
}

void
Quantize_Q8_K(
    size_t BlkLen,
    const float* A,
    size_t M,
    size_t K,
    size_t lda,
    std::byte* QuantA
)
{
    const float* ARowPtr = A;
    std::byte* QuantARowPtr = static_cast<std::byte*>(QuantA);

    size_t QuantAStride = ((K + BlkLen - 1) / BlkLen) * sizeof(block_q8_K);
    for (size_t m = 0; m < M; ++m) {
        QuantizeARow_Q8_K(BlkLen, ARowPtr, K, QuantARowPtr);
        ARowPtr += lda;
        QuantARowPtr += QuantAStride;
    }
}

void
DequantizeARow_Q8_K(
    size_t /*BlkLen*/,
    float* A,
    size_t CountK,
    const std::byte* QuantA
)
{
    const block_q8_K* x = reinterpret_cast<const block_q8_K*>(QuantA);
    // QuantA point to M * CountKBlk of block_q8_K
    dequantize_row_q8_K(x, A, CountK);
}

void
Dequantize_Q8_K(
    size_t BlkLen,
    float* A,
    size_t M,
    size_t K,
    size_t lda,
    const std::byte* QuantA
)
{
    float* ARowPtr = A;
    const std::byte* QuantARowPtr = static_cast<const std::byte*>(QuantA);

    size_t QuantAStride = ((K + BlkLen - 1) / BlkLen) * sizeof(block_q8_K);
    for (size_t m = 0; m < M; ++m) {
        DequantizeARow_Q8_K(BlkLen, ARowPtr, K, QuantARowPtr);
        ARowPtr += lda;
        QuantARowPtr += QuantAStride;
    }
}

void
ggml_vec_dot_tq1_0_q8_K(int n, float* s, const void* vx, const void* vy)
{
    const block_tq1_0* x = reinterpret_cast<const block_tq1_0*>(vx);
    const block_q8_K* y = reinterpret_cast<const block_q8_K*>(vy);

    const int nb = n / QK_K;
    __m256 sumf = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        // 16-bit sums
        __m256i sumi0 = _mm256_setzero_si256();
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();

        // first 32 bytes of 5 elements
        {
            __m256i qx0 = _mm256_loadu_si256((const __m256i *)(x[i].qs));
            // 8-bit multiplies with shifts, masks and adds
            __m256i qx1 = _mm256_add_epi8(qx0, _mm256_add_epi8(qx0, qx0));                                          // 1 * 3
            __m256i qx2 = _mm256_add_epi8(_mm256_and_si256(_mm256_slli_epi16(qx0, 3), _mm256_set1_epi8(-8)), qx0);  // 1 * 9
            __m256i qx3 = _mm256_add_epi8(_mm256_and_si256(_mm256_slli_epi16(qx1, 3), _mm256_set1_epi8(-8)), qx1);  // 3 * 9
            __m256i qx4 = _mm256_add_epi8(_mm256_and_si256(_mm256_slli_epi16(qx2, 3), _mm256_set1_epi8(-8)), qx2);  // 9 * 9

            // TODO: can _mm256_mulhi_epu16 be faster even if 16-bits?

            // Cancel the +1 from avg so that it behaves like a halving add
            qx0 = _mm256_subs_epu8(qx0, _mm256_set1_epi8(1));
            qx1 = _mm256_subs_epu8(qx1, _mm256_set1_epi8(1));
            qx2 = _mm256_subs_epu8(qx2, _mm256_set1_epi8(1));
            qx3 = _mm256_subs_epu8(qx3, _mm256_set1_epi8(1));
            qx4 = _mm256_subs_epu8(qx4, _mm256_set1_epi8(1));
            // Multiply by 3 and get the top 2 bits
            qx0 = _mm256_avg_epu8(qx0, _mm256_avg_epu8(qx0, _mm256_setzero_si256()));
            qx1 = _mm256_avg_epu8(qx1, _mm256_avg_epu8(qx1, _mm256_setzero_si256()));
            qx2 = _mm256_avg_epu8(qx2, _mm256_avg_epu8(qx2, _mm256_setzero_si256()));
            qx3 = _mm256_avg_epu8(qx3, _mm256_avg_epu8(qx3, _mm256_setzero_si256()));
            qx4 = _mm256_avg_epu8(qx4, _mm256_avg_epu8(qx4, _mm256_setzero_si256()));
            qx0 = _mm256_and_si256(_mm256_srli_epi16(qx0, 6), _mm256_set1_epi8(3));
            qx1 = _mm256_and_si256(_mm256_srli_epi16(qx1, 6), _mm256_set1_epi8(3));
            qx2 = _mm256_and_si256(_mm256_srli_epi16(qx2, 6), _mm256_set1_epi8(3));
            qx3 = _mm256_and_si256(_mm256_srli_epi16(qx3, 6), _mm256_set1_epi8(3));
            qx4 = _mm256_and_si256(_mm256_srli_epi16(qx4, 6), _mm256_set1_epi8(3));

            const __m256i qy0 = _mm256_loadu_si256((const __m256i *)(y[i].qs + 0));
            const __m256i qy1 = _mm256_loadu_si256((const __m256i *)(y[i].qs + 32));
            const __m256i qy2 = _mm256_loadu_si256((const __m256i *)(y[i].qs + 64));
            const __m256i qy3 = _mm256_loadu_si256((const __m256i *)(y[i].qs + 96));
            const __m256i qy4 = _mm256_loadu_si256((const __m256i *)(y[i].qs + 128));

            qx0 = _mm256_maddubs_epi16(qx0, qy0);
            qx1 = _mm256_maddubs_epi16(qx1, qy1);
            qx2 = _mm256_maddubs_epi16(qx2, qy2);
            qx3 = _mm256_maddubs_epi16(qx3, qy3);
            qx4 = _mm256_maddubs_epi16(qx4, qy4);

            sumi0 = _mm256_add_epi16(sumi0, _mm256_add_epi16(qx0, qx1));
            sumi1 = _mm256_add_epi16(sumi1, _mm256_add_epi16(qx2, qx3));
            sumi2 = _mm256_add_epi16(sumi2, qx4);
        }

        // last 16 bytes of 5-element, along with the 4 bytes of 4 elements
        {
            __m128i qx0 = _mm_loadu_si128((const __m128i *)(x[i].qs + 32));
            uint32_t qh;
            memcpy(&qh, x[i].qh, sizeof(qh));  // potentially unaligned
            __m256i qx5_l = _mm256_cvtepu8_epi16(_mm_set1_epi32(qh));
            __m128i qx1 = _mm_add_epi8(qx0, _mm_add_epi8(qx0, qx0));                                    // 1 * 3
            __m128i qx2 = _mm_add_epi8(_mm_and_si128(_mm_slli_epi16(qx0, 3), _mm_set1_epi8(-8)), qx0);  // 1 * 9
            __m128i qx3 = _mm_add_epi8(_mm_and_si128(_mm_slli_epi16(qx1, 3), _mm_set1_epi8(-8)), qx1);  // 3 * 9
            __m128i qx4 = _mm_add_epi8(_mm_and_si128(_mm_slli_epi16(qx2, 3), _mm_set1_epi8(-8)), qx2);  // 9 * 9
            __m256i qx01 = MM256_SET_M128I(qx1, qx0);
            __m256i qx23 = MM256_SET_M128I(qx3, qx2);

            // avx2 does not have 8-bit multiplies, so 16-bit it is.
            qx5_l = _mm256_mullo_epi16(qx5_l, _mm256_set_epi16(27, 27, 27, 27, 9, 9, 9, 9, 3, 3, 3, 3, 1, 1, 1, 1));
            qx5_l = _mm256_and_si256(qx5_l, _mm256_set1_epi16(0xFF));
            __m128i qx5 = _mm_packus_epi16(_mm256_castsi256_si128(qx5_l), _mm256_extracti128_si256(qx5_l, 1));

            __m256i qx45 = MM256_SET_M128I(qx5, qx4);

            // Cancel the +1 from avg so that it behaves like a halving add
            qx01 = _mm256_subs_epu8(qx01, _mm256_set1_epi8(1));
            qx23 = _mm256_subs_epu8(qx23, _mm256_set1_epi8(1));
            qx45 = _mm256_subs_epu8(qx45, _mm256_set1_epi8(1));
            // Multiply by 3 and get the top 2 bits
            qx01 = _mm256_avg_epu8(qx01, _mm256_avg_epu8(qx01, _mm256_setzero_si256()));
            qx23 = _mm256_avg_epu8(qx23, _mm256_avg_epu8(qx23, _mm256_setzero_si256()));
            qx45 = _mm256_avg_epu8(qx45, _mm256_avg_epu8(qx45, _mm256_setzero_si256()));
            qx01 = _mm256_and_si256(_mm256_srli_epi16(qx01, 6), _mm256_set1_epi8(3));
            qx23 = _mm256_and_si256(_mm256_srli_epi16(qx23, 6), _mm256_set1_epi8(3));
            qx45 = _mm256_and_si256(_mm256_srli_epi16(qx45, 6), _mm256_set1_epi8(3));

            const __m256i qy01 = _mm256_loadu_si256((const __m256i *)(y[i].qs + 160));
            const __m256i qy23 = _mm256_loadu_si256((const __m256i *)(y[i].qs + 192));
            const __m256i qy45 = _mm256_loadu_si256((const __m256i *)(y[i].qs + 224));

            qx01 = _mm256_maddubs_epi16(qx01, qy01);
            qx23 = _mm256_maddubs_epi16(qx23, qy23);
            qx45 = _mm256_maddubs_epi16(qx45, qy45);

            sumi0 = _mm256_add_epi16(sumi0, qx01);
            sumi1 = _mm256_add_epi16(sumi1, qx23);
            sumi2 = _mm256_add_epi16(sumi2, qx45);
        }

        const __m256i ysum = _mm256_loadu_si256((const __m256i *)y[i].bsums);
        const __m256 d = _mm256_set1_ps(y[i].d * GGML_FP16_TO_FP32(x[i].d));

        sumi0 = _mm256_sub_epi16(sumi0, ysum);
        sumi0 = _mm256_add_epi16(sumi0, _mm256_add_epi16(sumi1, sumi2));
        sumi0 = _mm256_madd_epi16(sumi0, _mm256_set1_epi16(1));

        sumf = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(sumi0), d), sumf);
    }

    *s = hsum_float_8(sumf);
}

size_t
SQTernaryBitGemmKernel_TQ1_0_Q8_K(
    size_t /*BlkLen*/,
    const std::byte* QuantA,
    const std::byte* QuantB,
    const float* /*QuantBScale*/,
    const std::byte* /*QuantBZeroPoint*/,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t /*BlockCountK*/,
    size_t /*ldc*/,
    const float* /*Bias*/
)
{
    size_t BlkCountK = (CountK + QK_K - 1) / QK_K;
    float* s = C;
    const std::byte* vy = QuantA;
    size_t QuantASizeInBytes = BlkCountK * sizeof(block_q8_K);
    size_t QuantBSizeInBytes = BlkCountK * sizeof(block_tq1_0);
    for (size_t m = 0; m < CountM; m++) {
      const std::byte* vx = QuantB;
      for (size_t n = 0; n < CountN; n++) {
        ggml_vec_dot_tq1_0_q8_K(CountK, s, vx, vy);
        s++;
        vx += QuantBSizeInBytes;
      }
      vy += QuantASizeInBytes;
    }
    return CountM;
}

size_t
QTernaryBitGemmPackQuantBDataSize(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    assert(ComputeType == SQNBIT_CompInt8);
    assert(BlkLen == QK_K);
    size_t BlkCountK = (K + QK_K - 1) / QK_K;
    return BlkCountK * N * sizeof(block_tq1_0);
}

size_t
QTernaryBitGemmPerGemmWorkspaceSize(
    size_t M,
    size_t /*N*/,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    assert(ComputeType == SQNBIT_CompInt8);
    assert(BlkLen == QK_K);

    // workspace buffer is used for block quantization of A to int8
    const size_t BlockCountK = (K + QK_K - 1) / QK_K;
    // QuantData + Scale + bsums
    //size_t per_blk_bsums_size_in_bytes = 256 / 16;
    //size_t per_blk_scale_size_in_bytes = sizeof(float);  // TODO: use fp16
    const size_t PerGemmWorkspaceSize = M * BlockCountK * sizeof(block_q8_K);
    return PerGemmWorkspaceSize;
}

//
// Kernel dispatch structure definition.
//
const MLAS_QNBIT_GEMM_DISPATCH MlasSQTernaryBitGemmDispatchAvx2 = []() {
    MLAS_QNBIT_GEMM_DISPATCH d;

    d.Q2BitGemmPackQuantBDataSize = QTernaryBitGemmPackQuantBDataSize;
    d.SQ2BitGemmPackQuantBData = nullptr;

    d.Q2BitGemmPerGemmWorkspaceSize = QTernaryBitGemmPerGemmWorkspaceSize;

    d.SQ2BitGemmKernel_CompInt8 = SQTernaryBitGemmKernel_TQ1_0_Q8_K;
    d.QuantizeARow_CompInt8 = QuantizeARow_Q8_K;

    return d;
}();

#ifdef _WIN32
#pragma warning(pop)
#endif
