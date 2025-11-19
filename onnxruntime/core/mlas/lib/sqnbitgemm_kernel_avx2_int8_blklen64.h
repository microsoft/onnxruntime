#pragma once
#include <algorithm>
#include <cassert>
#include <utility>

#include "qnbitgemm.h"
#include "sqnbitgemm_kernel_avx_common.h"

MLAS_DECLSPEC_ALIGN(static const uint32_t MasksAvx2BlkLen64[24], 32) = {
    0x00ff00ff, 0x00ff00ff, 0x00ff00ff, 0x00ff00ff, 0x00ff00ff, 0x00ff00ff, 0x00ff00ff, 0x00ff00ff,
    0xff00ff00, 0xff00ff00, 0xff00ff00, 0xff00ff00, 0xff00ff00, 0xff00ff00, 0xff00ff00, 0xff00ff00,
    0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001, 0x00010001
};

template<bool vnni>
static MLAS_FORCEINLINE void
accumulate_blklen64_r2c1blk1_avx2(
    const __m256i& av00_32_epi8,
    const __m256i& av01_32_epi8,
    const __m256i& av10_32_epi8,
    const __m256i& av11_32_epi8,
    const std::byte* QuantBDataPtr,
    const float* scale_a0,
    const float* scale_a1,
    const float* scale_b,
    __m256& acc0,
    __m256& acc1
)
{
    const __m256i bv_packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr));
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    __m256i bv0_32_epi8 = _mm256_and_si256(bv_packed, low_mask);  // 0, 1,...30, 31
    __m256i bv1_32_epi8 = _mm256_srli_epi16(_mm256_sub_epi8(bv_packed, bv0_32_epi8), 4);  // 32, 33,...62, 63

#if !defined(__GNUC__) || (__GNUC__ > 10)
    if constexpr (vnni) {
        __m256i sum_8_epi32 = _mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), bv0_32_epi8, av00_32_epi8);
        sum_8_epi32 = _mm256_dpbusds_avx_epi32(sum_8_epi32, bv1_32_epi8, av01_32_epi8);
        __m256 sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);

        __m256 scale_a0_ps = _mm256_broadcast_ss(scale_a0);
        __m256 scale_b_ps = _mm256_broadcast_ss(scale_b);

        acc0 = _mm256_fmadd_ps(sum_ps, _mm256_mul_ps(scale_a0_ps, scale_b_ps), acc0);

        sum_8_epi32 = _mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), bv0_32_epi8, av10_32_epi8);
        sum_8_epi32 = _mm256_dpbusds_avx_epi32(sum_8_epi32, bv1_32_epi8, av11_32_epi8);
        sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);

        __m256 scale_a1_ps = _mm256_broadcast_ss(scale_a1);

        acc1 = _mm256_fmadd_ps(sum_ps, _mm256_mul_ps(scale_a1_ps, scale_b_ps), acc1);

    } else {
#endif
        __m256i dot0_16_epi16 = _mm256_maddubs_epi16(bv0_32_epi8, av00_32_epi8);
        __m256i dot1_16_epi16 = _mm256_maddubs_epi16(bv1_32_epi8, av01_32_epi8);
        __m256i sum_16_epi16 = _mm256_hadd_epi16(dot0_16_epi16, dot1_16_epi16);

        const __m256i one_16_epi16 = _mm256_srli_epi16(_mm256_cmpeq_epi16(bv0_32_epi8, bv0_32_epi8), 15);

        __m256i sum_8_epi32 = _mm256_madd_epi16(one_16_epi16, sum_16_epi16);
        __m256 sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);

        __m256 scale_a0_ps = _mm256_broadcast_ss(scale_a0);
        __m256 scale_b_ps = _mm256_broadcast_ss(scale_b);

        acc0 = _mm256_fmadd_ps(sum_ps, _mm256_mul_ps(scale_a0_ps, scale_b_ps), acc0);

        dot0_16_epi16 = _mm256_maddubs_epi16(bv0_32_epi8, av10_32_epi8);
        dot1_16_epi16 = _mm256_maddubs_epi16(bv1_32_epi8, av11_32_epi8);
        sum_16_epi16 = _mm256_hadd_epi16(dot0_16_epi16, dot1_16_epi16);

        sum_8_epi32 = _mm256_madd_epi16(one_16_epi16, sum_16_epi16);
        sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);

        __m256 scale_a1_ps = _mm256_broadcast_ss(scale_a1);

        acc1 = _mm256_fmadd_ps(sum_ps, _mm256_mul_ps(scale_a1_ps, scale_b_ps), acc1);
#if !defined(__GNUC__) || (__GNUC__ > 10)
    }
#endif
}

template<bool vnni>
static MLAS_FORCEINLINE void
accumulate_q8_blklen64_r1c1blk1_avx2(
    const __m256i& av00_32_epi8,
    const __m256i& av01_32_epi8,
    const __m256i& bv0_32_epi8,
    const __m256i& bv1_32_epi8,
    float scale_a0b,
    __m256& acc0
)
{
    __m256 scale_8_ps = _mm256_set1_ps(scale_a0b);

#if !defined(__GNUC__) || (__GNUC__ > 10)
    if constexpr (vnni)
    {
        __m256i sum_8_epi32 = _mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), bv0_32_epi8, av00_32_epi8);
        sum_8_epi32 = _mm256_dpbusds_avx_epi32(sum_8_epi32, bv1_32_epi8, av01_32_epi8);
        __m256 sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);
        acc0 = _mm256_fmadd_ps(sum_ps, scale_8_ps, acc0);
    }
    else
#endif
    {
        // 2 x i8 x i8 may be larger than i16
        const __m256i low_mask = _mm256_load_si256(reinterpret_cast<const __m256i*>(MasksAvx2BlkLen64));
        const __m256i high_mask = _mm256_load_si256(reinterpret_cast<const __m256i*>(MasksAvx2BlkLen64 + 8));
        const __m256i one_mask = _mm256_load_si256(reinterpret_cast<const __m256i*>(MasksAvx2BlkLen64 + 16));

        const __m256i bv0_low_32_epi8 = _mm256_and_si256(bv0_32_epi8, low_mask);
        const __m256i bv0_high_32_epi8 = _mm256_and_si256(bv0_32_epi8, high_mask);
        const __m256i bv1_low_32_epi8 = _mm256_and_si256(bv1_32_epi8, low_mask);
        const __m256i bv1_high_32_epi8 = _mm256_and_si256(bv1_32_epi8, high_mask);

        const __m256i dot00_low_16_epi16 = _mm256_maddubs_epi16(bv0_low_32_epi8, av00_32_epi8);
        const __m256i dot00_high_16_epi16 = _mm256_maddubs_epi16(bv0_high_32_epi8, av00_32_epi8);
        const __m256i dot01_low_16_epi16 = _mm256_maddubs_epi16(bv1_low_32_epi8, av01_32_epi8);
        const __m256i dot01_high_16_epi16 = _mm256_maddubs_epi16(bv1_high_32_epi8, av01_32_epi8);

        const __m256i dot00_low_8_epi32 = _mm256_madd_epi16(one_mask, dot00_low_16_epi16);
        const __m256i dot00_high_8_epi32 = _mm256_madd_epi16(one_mask, dot00_high_16_epi16);
        const __m256i dot00_8_epi32 = _mm256_add_epi32(dot00_low_8_epi32, dot00_high_8_epi32);

        const __m256i dot01_low_8_epi32 = _mm256_madd_epi16(one_mask, dot01_low_16_epi16);
        const __m256i dot01_high_8_epi32 = _mm256_madd_epi16(one_mask, dot01_high_16_epi16);
        const __m256i dot01_8_epi32 = _mm256_add_epi32(dot01_low_8_epi32, dot01_high_8_epi32);

        const __m256i sum0_8_epi32 = _mm256_add_epi32(dot00_8_epi32, dot01_8_epi32);
        __m256 sum0_8_ps = _mm256_cvtepi32_ps(sum0_8_epi32);
        acc0 = _mm256_fmadd_ps(sum0_8_ps, scale_8_ps, acc0);
    }
}

template<bool vnni>
static MLAS_FORCEINLINE void
accumulate_q8_blklen64_r2c1blk1_avx2(
    const __m256i& av00_32_epi8,
    const __m256i& av01_32_epi8,
    const __m256i& av10_32_epi8,
    const __m256i& av11_32_epi8,
    const __m256i& bv0_32_epi8,
    const __m256i& bv1_32_epi8,
    float scale_a0b,
    float scale_a1b,
    __m256& acc0,
    __m256& acc1
)
{
    __m256 scale0_8_ps = _mm256_set1_ps(scale_a0b);
    __m256 scale1_8_ps = _mm256_set1_ps(scale_a1b);

#if !defined(__GNUC__) || (__GNUC__ > 10)
    if constexpr (vnni)
    {
        __m256i sum0_8_epi32 = _mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), bv0_32_epi8, av00_32_epi8);
        sum0_8_epi32 = _mm256_dpbusds_avx_epi32(sum0_8_epi32, bv1_32_epi8, av01_32_epi8);
        __m256 sum0_ps = _mm256_cvtepi32_ps(sum0_8_epi32);
        acc0 = _mm256_fmadd_ps(sum0_ps, scale0_8_ps, acc0);

        __m256i sum1_8_epi32 = _mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), bv0_32_epi8, av10_32_epi8);
        sum1_8_epi32 = _mm256_dpbusds_avx_epi32(sum1_8_epi32, bv1_32_epi8, av11_32_epi8);
        __m256 sum1_ps = _mm256_cvtepi32_ps(sum1_8_epi32);
        acc1 = _mm256_fmadd_ps(sum1_ps, scale1_8_ps, acc1);
    }
    else
    #endif
    {
        // 2 x i8 x i8 may be larger than i16
        const __m256i low_mask = _mm256_load_si256(reinterpret_cast<const __m256i*>(MasksAvx2BlkLen64));
        const __m256i high_mask = _mm256_load_si256(reinterpret_cast<const __m256i*>(MasksAvx2BlkLen64 + 8));
        const __m256i one_mask = _mm256_load_si256(reinterpret_cast<const __m256i*>(MasksAvx2BlkLen64 + 16));

        const __m256i bv0_low_32_epi8 = _mm256_and_si256(bv0_32_epi8, low_mask);
        const __m256i bv0_high_32_epi8 = _mm256_and_si256(bv0_32_epi8, high_mask);
        const __m256i bv1_low_32_epi8 = _mm256_and_si256(bv1_32_epi8, low_mask);
        const __m256i bv1_high_32_epi8 = _mm256_and_si256(bv1_32_epi8, high_mask);

        // row 0
        const __m256i dot00_low_16_epi16 = _mm256_maddubs_epi16(bv0_low_32_epi8, av00_32_epi8);
        const __m256i dot00_high_16_epi16 = _mm256_maddubs_epi16(bv0_high_32_epi8, av00_32_epi8);
        const __m256i dot01_low_16_epi16 = _mm256_maddubs_epi16(bv1_low_32_epi8, av01_32_epi8);
        const __m256i dot01_high_16_epi16 = _mm256_maddubs_epi16(bv1_high_32_epi8, av01_32_epi8);

        const __m256i dot00_low_8_epi32 = _mm256_madd_epi16(one_mask, dot00_low_16_epi16);
        const __m256i dot00_high_8_epi32 = _mm256_madd_epi16(one_mask, dot00_high_16_epi16);
        const __m256i dot00_8_epi32 = _mm256_add_epi32(dot00_low_8_epi32, dot00_high_8_epi32);

        const __m256i dot01_low_8_epi32 = _mm256_madd_epi16(one_mask, dot01_low_16_epi16);
        const __m256i dot01_high_8_epi32 = _mm256_madd_epi16(one_mask, dot01_high_16_epi16);
        const __m256i dot01_8_epi32 = _mm256_add_epi32(dot01_low_8_epi32, dot01_high_8_epi32);

        const __m256i sum0_8_epi32 = _mm256_add_epi32(dot00_8_epi32, dot01_8_epi32);
        __m256 sum0_8_ps = _mm256_cvtepi32_ps(sum0_8_epi32);
        acc0 = _mm256_fmadd_ps(sum0_8_ps, scale0_8_ps, acc0);

        // row 1
        const __m256i dot10_low_16_epi16 = _mm256_maddubs_epi16(bv0_low_32_epi8, av10_32_epi8);
        const __m256i dot10_high_16_epi16 = _mm256_maddubs_epi16(bv0_high_32_epi8, av10_32_epi8);
        const __m256i dot11_low_16_epi16 = _mm256_maddubs_epi16(bv1_low_32_epi8, av11_32_epi8);
        const __m256i dot11_high_16_epi16 = _mm256_maddubs_epi16(bv1_high_32_epi8, av11_32_epi8);

        const __m256i dot10_low_8_epi32 = _mm256_madd_epi16(one_mask, dot10_low_16_epi16);
        const __m256i dot10_high_8_epi32 = _mm256_madd_epi16(one_mask, dot10_high_16_epi16);
        const __m256i dot10_8_epi32 = _mm256_add_epi32(dot10_low_8_epi32, dot10_high_8_epi32);

        const __m256i dot11_low_8_epi32 = _mm256_madd_epi16(one_mask, dot11_low_16_epi16);
        const __m256i dot11_high_8_epi32 = _mm256_madd_epi16(one_mask, dot11_high_16_epi16);
        const __m256i dot11_8_epi32 = _mm256_add_epi32(dot11_low_8_epi32, dot11_high_8_epi32);

        const __m256i sum1_8_epi32 = _mm256_add_epi32(dot10_8_epi32, dot11_8_epi32);
        __m256 sum1_8_ps = _mm256_cvtepi32_ps(sum1_8_epi32);
        acc1 = _mm256_fmadd_ps(sum1_8_ps, scale1_8_ps, acc1);
    }
}

template <bool vnni>
static MLAS_FORCEINLINE void
accumulate_blklen64_r1c1blk1_avx2(
    const __m256i& av00_32_epi8,
    const __m256i& av01_32_epi8,
    const std::byte* QuantBDataPtr,
    const float* scale_a,
    const float* scale_b,
    __m256& acc0
)
{
    // | v0  v32 | v1  v33 | ... | v30 v62 | v31 v63 |
    const __m256i bv_packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr));
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    __m256i bv0_32_epi8 = _mm256_and_si256(bv_packed, low_mask);                          // 0, 1,...30, 31
    __m256i bv1_32_epi8 = _mm256_srli_epi16(_mm256_sub_epi8(bv_packed, bv0_32_epi8), 4);  // 32, 33,...62, 63

#if !defined(__GNUC__) || (__GNUC__ > 10)
    if constexpr (vnni) {
        __m256i sum_8_epi32 = _mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), bv0_32_epi8, av00_32_epi8);
        sum_8_epi32 = _mm256_dpbusds_avx_epi32(sum_8_epi32, bv1_32_epi8, av01_32_epi8);
        const __m256 sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);

        __m256 scale_a_8_ps = _mm256_broadcast_ss(scale_a);
        __m256 scale_b_8_ps = _mm256_broadcast_ss(scale_b);

        acc0 = _mm256_fmadd_ps(sum_ps, _mm256_mul_ps(scale_a_8_ps, scale_b_8_ps), acc0);
    } else {
#endif
        const __m256i dot0_16_epi16 = _mm256_maddubs_epi16(bv0_32_epi8, av00_32_epi8);
        const __m256i dot1_16_epi16 = _mm256_maddubs_epi16(bv1_32_epi8, av01_32_epi8);
        const __m256i sum_16_epi16 = _mm256_hadd_epi16(dot0_16_epi16, dot1_16_epi16);

        __m256i one_16_epi16 = _mm256_srli_epi16(_mm256_cmpeq_epi16(bv0_32_epi8, bv0_32_epi8), 15);
        const __m256i sum_8_epi32 = _mm256_madd_epi16(one_16_epi16, sum_16_epi16);
        const __m256 sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);

        __m256 scale_a_8_ps = _mm256_broadcast_ss(scale_a);
        __m256 scale_b_8_ps = _mm256_broadcast_ss(scale_b);

        acc0 = _mm256_fmadd_ps(sum_ps, _mm256_mul_ps(scale_a_8_ps, scale_b_8_ps), acc0);
#if !defined(__GNUC__) || (__GNUC__ > 10)
    }
#endif
}

template <bool vnni>
MLAS_FORCEINLINE void
Q4Int8GemmR2xC4BlkLen64Avx2(
    const size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc
)
{
    constexpr size_t BlkBitWidth4 = 4;
    constexpr size_t NCols4 = 4;
    constexpr size_t NRows2 = 2;
    constexpr size_t SubblkLen = 64;

    const size_t BlkDataSizeInBytes = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
    const size_t PerBlkSubblkCount = BlkLen / SubblkLen;
    const size_t SubblkDataSizeInBytes = BlkDataSizeInBytes / PerBlkSubblkCount;

    const size_t lda = BlockCountK * BlkLen;
    const size_t StrideQuantBData = BlockCountK * BlkDataSizeInBytes;
    //const size_t StrideQuantBScale = BlockCountK;

    assert(CountM % NRows2 == 0);
    assert(CountN % NCols4 == 0);

    for (size_t m = 0; m < CountM; m += NRows2) {
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const float* BiasPtr = Bias;
        auto* SumPtr = C + m * ldc;

        for (size_t n = 0; n < CountN; n += NCols4) {
            const std::byte* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;

            const std::byte* QuantBDataPtr = QuantBDataColPtr;
            const float* QuantBScalePtr = QuantBScaleColPtr;

            __m256 acc[NCols4 * NRows2] = {
                _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()
            };

            // process 1 blks of 64 4b weights a time
            for (size_t k = 0; k < BlockCountK; ++k) {
                for (size_t kk = 0; kk < PerBlkSubblkCount; kk++) {
                    const __m256i av_00_epi8 = _mm256_loadu_si256((const __m256i*)QuantAPtr);
                    const __m256i av_01_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr + 32));
                    const __m256i av_10_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr + lda));
                    const __m256i av_11_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr + lda + 32));

                    accumulate_blklen64_r2c1blk1_avx2<vnni>(av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, QuantBDataPtr, QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr, acc[0], acc[NCols4]);
                    accumulate_blklen64_r2c1blk1_avx2<vnni>(av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, QuantBDataPtr + SubblkDataSizeInBytes, QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr + 1, acc[1], acc[NCols4 + 1]);
                    accumulate_blklen64_r2c1blk1_avx2<vnni>(av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, QuantBDataPtr + 2 * SubblkDataSizeInBytes, QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr + 2, acc[2], acc[NCols4 + 2]);
                    accumulate_blklen64_r2c1blk1_avx2<vnni>(av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, QuantBDataPtr + 3 * SubblkDataSizeInBytes, QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr + 3, acc[3], acc[NCols4 + 3]);

                    // increment block pointers
                    QuantAPtr += SubblkLen;
                    QuantBDataPtr += NCols4 * SubblkDataSizeInBytes;
                }
                QuantAScalePtr++;
                QuantBScalePtr += NCols4;
            }  // k_blks_remaining

            __m128 acc_r0 = FoldAccumulators(acc[0], acc[1], acc[2], acc[3]);
            __m128 acc_r1 = FoldAccumulators(acc[NCols4 + 0], acc[NCols4 + 1], acc[NCols4 + 2], acc[NCols4 + 3]);
            if (BiasPtr != nullptr) {
                const __m128 bias_4_ps = _mm_loadu_ps(BiasPtr);
                acc_r0 = _mm_add_ps(acc_r0, bias_4_ps);
                acc_r1 = _mm_add_ps(acc_r1, bias_4_ps);
            }
            _mm_storeu_ps(SumPtr, acc_r0);
            _mm_storeu_ps(SumPtr + ldc, acc_r1);

            // move to next NCols columns
            QuantBDataColPtr += NCols4 * StrideQuantBData;
            QuantBScaleColPtr += NCols4 * BlockCountK;
            BiasPtr += BiasPtr != nullptr ? NCols4 : 0;
            SumPtr += NCols4;
        }
    }
}

template <bool vnni>
MLAS_FORCEINLINE void
Q8Int8GemmR2xC4BlkLen64Avx2(
    const size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc
)
{
    constexpr size_t BlkBitWidth = 8;
    constexpr size_t NCols4 = 4;
    constexpr size_t NRows2 = 2;
    constexpr size_t SubblkLen = 64;

    const size_t BlkDataSizeInBytes = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t PerBlkSubblkCount = BlkLen / SubblkLen;
    const size_t SubblkDataSizeInBytes = BlkDataSizeInBytes / PerBlkSubblkCount;

    const size_t lda = BlockCountK * BlkLen;
    const size_t StrideQuantBData = BlockCountK * BlkDataSizeInBytes;

    assert(CountM % NRows2 == 0);
    assert(CountN % NCols4 == 0);

    for (size_t m = 0; m < CountM; m += NRows2) {
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const float* BiasPtr = Bias;
        auto* SumPtr = C + m * ldc;

        for (size_t n = 0; n < CountN; n += NCols4) {
            const std::byte* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;

            const std::byte* QuantBDataPtr = QuantBDataColPtr;
            const float* QuantBScalePtr = QuantBScaleColPtr;

            __m256 acc[NCols4 * NRows2] = {
                _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()
            };

            for (size_t k = 0; k < BlockCountK; ++k) {
                const float scale_a0 = *QuantAScalePtr;
                const float scale_a1 = *(QuantAScalePtr + BlockCountK);
                const float scale_a0b0 = (*QuantBScalePtr) * scale_a0;
                const float scale_a1b0 = (*QuantBScalePtr) * scale_a1;
                const float scale_a0b1 = (*(QuantBScalePtr + 1)) * scale_a0;
                const float scale_a1b1 = (*(QuantBScalePtr + 1)) * scale_a1;
                const float scale_a0b2 = (*(QuantBScalePtr + 2)) * scale_a0;
                const float scale_a1b2 = (*(QuantBScalePtr + 2)) * scale_a1;
                const float scale_a0b3 = (*(QuantBScalePtr + 3)) * scale_a0;
                const float scale_a1b3 = (*(QuantBScalePtr + 3)) * scale_a1;

                __m256i av_00_epi8 = _mm256_load_si256((const __m256i*)QuantAPtr);
                __m256i av_01_epi8 = _mm256_load_si256((const __m256i*)(QuantAPtr + 32));
                __m256i av_10_epi8 = _mm256_load_si256((const __m256i*)(QuantAPtr + lda));
                __m256i av_11_epi8 = _mm256_load_si256((const __m256i*)(QuantAPtr + lda + 32));

                __m256i bv00_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr));
                __m256i bv01_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 32));
                __m256i bv10_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + SubblkDataSizeInBytes));
                __m256i bv11_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + SubblkDataSizeInBytes + 32));
                __m256i bv20_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 2 * SubblkDataSizeInBytes));
                __m256i bv21_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 2 * SubblkDataSizeInBytes + 32));
                __m256i bv30_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 3 * SubblkDataSizeInBytes));
                __m256i bv31_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 3 * SubblkDataSizeInBytes + 32));

                for (size_t kk = 0; kk < PerBlkSubblkCount - 1; kk++) {
                    accumulate_q8_blklen64_r2c1blk1_avx2<vnni>(av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, bv00_32_epi8, bv01_32_epi8, scale_a0b0, scale_a1b0, acc[0], acc[NCols4]);
                    accumulate_q8_blklen64_r2c1blk1_avx2<vnni>(av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, bv10_32_epi8, bv11_32_epi8, scale_a0b1, scale_a1b1, acc[1], acc[NCols4 + 1]);
                    accumulate_q8_blklen64_r2c1blk1_avx2<vnni>(av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, bv20_32_epi8, bv21_32_epi8, scale_a0b2, scale_a1b2, acc[2], acc[NCols4 + 2]);
                    accumulate_q8_blklen64_r2c1blk1_avx2<vnni>(av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, bv30_32_epi8, bv31_32_epi8, scale_a0b3, scale_a1b3, acc[3], acc[NCols4 + 3]);

                    // increment block pointers
                    QuantAPtr += SubblkLen;
                    QuantBDataPtr += NCols4 * SubblkDataSizeInBytes;

                    av_00_epi8 = _mm256_load_si256((const __m256i*)QuantAPtr);
                    av_01_epi8 = _mm256_load_si256((const __m256i*)(QuantAPtr + 32));
                    av_10_epi8 = _mm256_load_si256((const __m256i*)(QuantAPtr + lda));
                    av_11_epi8 = _mm256_load_si256((const __m256i*)(QuantAPtr + lda + 32));

                    bv00_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr));
                    bv01_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 32));
                    bv10_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + SubblkDataSizeInBytes));
                    bv11_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + SubblkDataSizeInBytes + 32));
                    bv20_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 2 * SubblkDataSizeInBytes));
                    bv21_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 2 * SubblkDataSizeInBytes + 32));
                    bv30_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 3 * SubblkDataSizeInBytes));
                    bv31_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 3 * SubblkDataSizeInBytes + 32));
                }

                accumulate_q8_blklen64_r2c1blk1_avx2<vnni>(av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, bv00_32_epi8, bv01_32_epi8, scale_a0b0, scale_a1b0, acc[0], acc[NCols4]);
                accumulate_q8_blklen64_r2c1blk1_avx2<vnni>(av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, bv10_32_epi8, bv11_32_epi8, scale_a0b1, scale_a1b1, acc[1], acc[NCols4 + 1]);
                accumulate_q8_blklen64_r2c1blk1_avx2<vnni>(av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, bv20_32_epi8, bv21_32_epi8, scale_a0b2, scale_a1b2, acc[2], acc[NCols4 + 2]);
                accumulate_q8_blklen64_r2c1blk1_avx2<vnni>(av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, bv30_32_epi8, bv31_32_epi8, scale_a0b3, scale_a1b3, acc[3], acc[NCols4 + 3]);

                // increment block pointers
                QuantAPtr += SubblkLen;
                QuantBDataPtr += NCols4 * SubblkDataSizeInBytes;

                QuantAScalePtr++;
                QuantBScalePtr += NCols4;
            }  // k_blks_remaining

            __m128 acc_r0 = FoldAccumulators(acc[0], acc[1], acc[2], acc[3]);
            __m128 acc_r1 = FoldAccumulators(acc[NCols4 + 0], acc[NCols4 + 1], acc[NCols4 + 2], acc[NCols4 + 3]);
            if (BiasPtr != nullptr) {
                const __m128 bias_4_ps = _mm_loadu_ps(BiasPtr);
                acc_r0 = _mm_add_ps(acc_r0, bias_4_ps);
                acc_r1 = _mm_add_ps(acc_r1, bias_4_ps);
            }
            _mm_storeu_ps(SumPtr, acc_r0);
            _mm_storeu_ps(SumPtr + ldc, acc_r1);

            // move to next NCols columns
            QuantBDataColPtr += NCols4 * StrideQuantBData;
            QuantBScaleColPtr += NCols4 * BlockCountK;
            BiasPtr += BiasPtr != nullptr ? NCols4 : 0;
            SumPtr += NCols4;
        }
    }
}

template<bool vnni>
void MLAS_FORCEINLINE
Q4Int8GemmR2xC1BlkLen64Avx2(
    const size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc
)
{
    constexpr size_t BlkBitWidth4 = 4;
    [[maybe_unused]] constexpr size_t NCols4 = 4;
    constexpr size_t NRows2 = 2;
    constexpr size_t SubblkLen = 64;

    const size_t BlkDataSizeInBytes = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
    const size_t PerBlkSubblkCount = BlkLen / SubblkLen;
    const size_t SubblkDataSizeInBytes = BlkDataSizeInBytes / PerBlkSubblkCount;

    const size_t lda = BlockCountK * BlkLen;
    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
    const size_t StrideQuantBScale = BlockCountK;

    assert(CountM % NRows2 == 0);
    assert(CountN < NCols4);

    for (size_t m = 0; m < CountM; m += NRows2) {
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const float* BiasPtr = Bias;
        float* SumPtr = C + m * ldc;

        for (size_t n = 0; n < CountN; n++) {
            const std::byte* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;

            const std::byte* QuantBDataPtr = QuantBDataColPtr;
            const float* QuantBScalePtr = QuantBScaleColPtr;

            __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();

            for (size_t k = 0; k < BlockCountK; ++k) {
                for (size_t kk = 0; kk < PerBlkSubblkCount; kk++) {
                    const __m256i av_00_epi8 = _mm256_loadu_si256((const __m256i*)QuantAPtr);
                    const __m256i av_01_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr + 32));
                    const __m256i av_10_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr + lda));
                    const __m256i av_11_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr + lda + 32));

                    accumulate_blklen64_r2c1blk1_avx2<vnni>(av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, QuantBDataPtr, QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr, acc0, acc1);

                    // increment block pointers
                    QuantAPtr += SubblkLen;
                    QuantBDataPtr += SubblkDataSizeInBytes;
                }
                QuantAScalePtr++;
                QuantBScalePtr++;
            }

            *SumPtr = hsum_float_8(acc0);
            *(SumPtr + ldc) = hsum_float_8(acc1);
            if (BiasPtr) {
                *SumPtr += *BiasPtr;
                *(SumPtr + ldc) += *BiasPtr;
            }

            // move to next column
            QuantBDataColPtr += StrideQuantBData;
            QuantBScaleColPtr += StrideQuantBScale;
            BiasPtr += BiasPtr != nullptr ? 1 : 0;
            SumPtr += 1;
        }
    }
}

template<bool vnni>
void MLAS_FORCEINLINE
Q8Int8GemmR2xC1BlkLen64Avx2(
    const size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc
)
{
    constexpr size_t BlkBitWidth = 8;
    [[maybe_unused]] constexpr size_t NCols4 = 4;
    constexpr size_t NRows2 = 2;
    constexpr size_t SubblkLen = 64;

    const size_t BlkDataSizeInBytes = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t PerBlkSubblkCount = BlkLen / SubblkLen;
    const size_t SubblkDataSizeInBytes = BlkDataSizeInBytes / PerBlkSubblkCount;

    const size_t lda = BlockCountK * BlkLen;
    const size_t StrideQuantBData = BlockCountK * BlkDataSizeInBytes;
    const size_t StrideQuantBScale = BlockCountK;

    assert(CountM % NRows2 == 0);
    assert(CountN < NCols4);

    for (size_t m = 0; m < CountM; m += NRows2) {
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const float* BiasPtr = Bias;
        float* SumPtr = C + m * ldc;

        for (size_t n = 0; n < CountN; n++) {
            const std::byte* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;

            const std::byte* QuantBDataPtr = QuantBDataColPtr;
            const float* QuantBScalePtr = QuantBScaleColPtr;

            __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();

            for (size_t k = 0; k < BlockCountK; ++k) {
                const float scale_a0 = *QuantAScalePtr;
                const float scale_a1 = *(QuantAScalePtr + BlockCountK);
                const float scale_a0b0 = (*QuantBScalePtr) * scale_a0;
                const float scale_a1b0 = (*QuantBScalePtr) * scale_a1;

                __m256i av_00_epi8 = _mm256_load_si256((const __m256i*)QuantAPtr);
                __m256i av_01_epi8 = _mm256_load_si256((const __m256i*)(QuantAPtr + 32));
                __m256i av_10_epi8 = _mm256_load_si256((const __m256i*)(QuantAPtr + lda));
                __m256i av_11_epi8 = _mm256_load_si256((const __m256i*)(QuantAPtr + lda + 32));

                __m256i bv00_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr));
                __m256i bv01_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 32));

                for (size_t kk = 0; kk < PerBlkSubblkCount - 1; kk++) {
                    accumulate_q8_blklen64_r2c1blk1_avx2<vnni>(av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, bv00_32_epi8, bv01_32_epi8, scale_a0b0, scale_a1b0, acc0, acc1);

                    // increment block pointers
                    QuantAPtr += SubblkLen;
                    QuantBDataPtr += SubblkDataSizeInBytes;

                    av_00_epi8 = _mm256_load_si256((const __m256i*)QuantAPtr);
                    av_01_epi8 = _mm256_load_si256((const __m256i*)(QuantAPtr + 32));
                    av_10_epi8 = _mm256_load_si256((const __m256i*)(QuantAPtr + lda));
                    av_11_epi8 = _mm256_load_si256((const __m256i*)(QuantAPtr + lda + 32));

                    bv00_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr));
                    bv01_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 32));
                }

                accumulate_q8_blklen64_r2c1blk1_avx2<vnni>(av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, bv00_32_epi8, bv01_32_epi8, scale_a0b0, scale_a1b0, acc0, acc1);

                // increment block pointers
                QuantAPtr += SubblkLen;
                QuantBDataPtr += SubblkDataSizeInBytes;

                QuantAScalePtr++;
                QuantBScalePtr++;
            }

            *SumPtr = hsum_float_8(acc0);
            *(SumPtr + ldc) = hsum_float_8(acc1);
            if (BiasPtr) {
                *SumPtr += *BiasPtr;
                *(SumPtr + ldc) += *BiasPtr;
            }

            // move to next column
            QuantBDataColPtr += StrideQuantBData;
            QuantBScaleColPtr += StrideQuantBScale;
            BiasPtr += BiasPtr != nullptr ? 1 : 0;
            SumPtr += 1;
        }
    }
}

template <bool vnni>
MLAS_FORCEINLINE void
Q4Int8GemmR1xC4BlkLen64Avx2(
    const size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc
)
{
    constexpr size_t BlkBitWidth4 = 4;
    constexpr size_t NCols4 = 4;
    [[maybe_unused]] constexpr size_t NRows2 = 2;
    constexpr size_t SubblkLen = 64;

    const size_t BlkDataSizeInBytes = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
    const size_t PerBlkSubblkCount = BlkLen / SubblkLen;
    const size_t SubblkDataSizeInBytes = BlkDataSizeInBytes / PerBlkSubblkCount;

    const size_t lda = BlockCountK * BlkLen;
    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
    //const size_t StrideQuantBScale = BlockCountK;

    assert(CountM < NRows2);
    assert(CountN % NCols4 == 0);

    for (size_t m = 0; m < CountM; m++) {
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const float* BiasPtr = Bias;
        auto* SumPtr = C + m * ldc;

        for (size_t n = 0; n < CountN; n += NCols4) {
            const std::byte* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;

            const std::byte* QuantBDataPtr = QuantBDataColPtr;
            const float* QuantBScalePtr = QuantBScaleColPtr;

            __m256 acc[NCols4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
            for (size_t k = 0; k < BlockCountK; ++k) {
                for (size_t kk = 0; kk < PerBlkSubblkCount; kk++) {
                    const __m256i av_00_epi8 = _mm256_loadu_si256((const __m256i*)QuantAPtr);
                    const __m256i av_01_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr + 32));
                    accumulate_blklen64_r1c1blk1_avx2<vnni>(av_00_epi8, av_01_epi8, QuantBDataPtr, QuantAScalePtr, QuantBScalePtr, acc[0]);
                    accumulate_blklen64_r1c1blk1_avx2<vnni>(av_00_epi8, av_01_epi8, QuantBDataPtr + SubblkDataSizeInBytes, QuantAScalePtr, QuantBScalePtr + 1, acc[1]);
                    accumulate_blklen64_r1c1blk1_avx2<vnni>(av_00_epi8, av_01_epi8, QuantBDataPtr + 2 * SubblkDataSizeInBytes, QuantAScalePtr, QuantBScalePtr + 2, acc[2]);
                    accumulate_blklen64_r1c1blk1_avx2<vnni>(av_00_epi8, av_01_epi8, QuantBDataPtr + 3 * SubblkDataSizeInBytes, QuantAScalePtr, QuantBScalePtr + 3, acc[3]);

                    // increment block pointers
                    QuantAPtr += SubblkLen;
                    QuantBDataPtr += NCols4 * SubblkDataSizeInBytes;
                }
                QuantAScalePtr++;
                QuantBScalePtr += NCols4;
            }

            __m128 acc_r0 = FoldAccumulators(acc[0], acc[1], acc[2], acc[3]);
            if (BiasPtr != nullptr) {
                acc_r0 = _mm_add_ps(acc_r0, _mm_loadu_ps(BiasPtr));
            }

            _mm_storeu_ps(SumPtr, acc_r0);

            // move to next NCols columns
            QuantBDataColPtr += NCols4 * StrideQuantBData;
            QuantBScaleColPtr += NCols4 * BlockCountK;
            BiasPtr += BiasPtr != nullptr ? NCols4 : 0;
            SumPtr += NCols4;
        }
    }
}

template <bool vnni>
MLAS_FORCEINLINE void
Q8Int8GemmR1xC4BlkLen64Avx2(
    const size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc
)
{
    constexpr size_t BlkBitWidth = 8;
    constexpr size_t NCols4 = 4;
    [[maybe_unused]] constexpr size_t NRows2 = 2;
    constexpr size_t SubblkLen = 64;

    const size_t BlkDataSizeInBytes = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t PerBlkSubblkCount = BlkLen / SubblkLen;
    const size_t SubblkDataSizeInBytes = BlkDataSizeInBytes / PerBlkSubblkCount;

    const size_t lda = BlockCountK * BlkLen;
    const size_t StrideQuantBData = BlockCountK * BlkDataSizeInBytes;
    //const size_t StrideQuantBScale = BlockCountK;

    assert(CountM < NRows2);
    assert(CountN % NCols4 == 0);

    for (size_t m = 0; m < CountM; m++) {
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const float* BiasPtr = Bias;
        auto* SumPtr = C + m * ldc;

        for (size_t n = 0; n < CountN; n += NCols4) {
            const std::byte* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;

            const std::byte* QuantBDataPtr = QuantBDataColPtr;
            const float* QuantBScalePtr = QuantBScaleColPtr;

            __m256 acc[NCols4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
            for (size_t k = 0; k < BlockCountK; ++k) {
                const float scale_a0 = *QuantAScalePtr;
                const float scale_a0b0 = (*QuantBScalePtr) * scale_a0;
                const float scale_a0b1 = (*(QuantBScalePtr + 1)) * scale_a0;
                const float scale_a0b2 = (*(QuantBScalePtr + 2)) * scale_a0;
                const float scale_a0b3 = (*(QuantBScalePtr + 3)) * scale_a0;

                __m256i av_00_epi8 = _mm256_load_si256((const __m256i*)QuantAPtr);
                __m256i av_01_epi8 = _mm256_load_si256((const __m256i*)(QuantAPtr + 32));

                __m256i bv00_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr));
                __m256i bv01_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 32));
                __m256i bv10_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + SubblkDataSizeInBytes));
                __m256i bv11_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + SubblkDataSizeInBytes + 32));
                __m256i bv20_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 2 * SubblkDataSizeInBytes));
                __m256i bv21_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 2 * SubblkDataSizeInBytes + 32));
                __m256i bv30_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 3 * SubblkDataSizeInBytes));
                __m256i bv31_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 3 * SubblkDataSizeInBytes + 32));

                for (size_t kk = 0; kk < PerBlkSubblkCount - 1; kk++) {
                    accumulate_q8_blklen64_r1c1blk1_avx2<vnni>(av_00_epi8, av_01_epi8, bv00_32_epi8, bv01_32_epi8, scale_a0b0, acc[0]);
                    accumulate_q8_blklen64_r1c1blk1_avx2<vnni>(av_00_epi8, av_01_epi8, bv10_32_epi8, bv11_32_epi8, scale_a0b1, acc[1]);
                    accumulate_q8_blklen64_r1c1blk1_avx2<vnni>(av_00_epi8, av_01_epi8, bv20_32_epi8, bv21_32_epi8, scale_a0b2, acc[2]);
                    accumulate_q8_blklen64_r1c1blk1_avx2<vnni>(av_00_epi8, av_01_epi8, bv30_32_epi8, bv31_32_epi8, scale_a0b3, acc[3]);
                    // increment block pointers
                    QuantAPtr += SubblkLen;
                    QuantBDataPtr += NCols4 * SubblkDataSizeInBytes;

                    av_00_epi8 = _mm256_load_si256((const __m256i*)QuantAPtr);
                    av_01_epi8 = _mm256_load_si256((const __m256i*)(QuantAPtr + 32));

                    bv00_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr));
                    bv01_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 32));
                    bv10_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + SubblkDataSizeInBytes));
                    bv11_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + SubblkDataSizeInBytes + 32));
                    bv20_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 2 * SubblkDataSizeInBytes));
                    bv21_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 2 * SubblkDataSizeInBytes + 32));
                    bv30_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 3 * SubblkDataSizeInBytes));
                    bv31_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 3 * SubblkDataSizeInBytes + 32));
                }

                accumulate_q8_blklen64_r1c1blk1_avx2<vnni>(av_00_epi8, av_01_epi8, bv00_32_epi8, bv01_32_epi8, scale_a0b0, acc[0]);
                accumulate_q8_blklen64_r1c1blk1_avx2<vnni>(av_00_epi8, av_01_epi8, bv10_32_epi8, bv11_32_epi8, scale_a0b1, acc[1]);
                accumulate_q8_blklen64_r1c1blk1_avx2<vnni>(av_00_epi8, av_01_epi8, bv20_32_epi8, bv21_32_epi8, scale_a0b2, acc[2]);
                accumulate_q8_blklen64_r1c1blk1_avx2<vnni>(av_00_epi8, av_01_epi8, bv30_32_epi8, bv31_32_epi8, scale_a0b3, acc[3]);
                QuantAPtr += SubblkLen;
                QuantBDataPtr += NCols4 * SubblkDataSizeInBytes;

                QuantAScalePtr++;
                QuantBScalePtr += NCols4;
            }

            __m128 acc_r0 = FoldAccumulators(acc[0], acc[1], acc[2], acc[3]);
            if (BiasPtr != nullptr) {
                acc_r0 = _mm_add_ps(acc_r0, _mm_loadu_ps(BiasPtr));
            }

            _mm_storeu_ps(SumPtr, acc_r0);

            // move to next NCols columns
            QuantBDataColPtr += NCols4 * StrideQuantBData;
            QuantBScaleColPtr += NCols4 * BlockCountK;
            BiasPtr += BiasPtr != nullptr ? NCols4 : 0;
            SumPtr += NCols4;
        }
    }
}

template <bool vnni>
MLAS_FORCEINLINE void
Q4Int8GemmR1xC1BlkLen64Avx2(
    const size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc
)
{
    constexpr size_t BlkBitWidth4 = 4;
    [[maybe_unused]] constexpr size_t NCols4 = 4;
    [[maybe_unused]] constexpr size_t NRows2 = 2;
    constexpr size_t SubblkLen = 64;

    const size_t BlkDataSizeInBytes = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
    const size_t PerBlkSubblkCount = BlkLen / SubblkLen;
    const size_t SubblkDataSizeInBytes = BlkDataSizeInBytes / PerBlkSubblkCount;

    const size_t lda = BlockCountK * BlkLen;
    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
    const size_t StrideQuantBScale = BlockCountK;

    assert(CountM < NRows2);
    assert(CountN < NCols4);

    for (size_t m = 0; m < CountM; m++) {
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const float* BiasPtr = Bias;
        auto* SumPtr = C + m * ldc;

        for (size_t n = 0; n < CountN; n++) {
            const std::byte* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;
            const std::byte* QuantBDataPtr = QuantBDataColPtr;
            const float* QuantBScalePtr = QuantBScaleColPtr;

            __m256 acc0 = _mm256_setzero_ps();
            for (size_t k = 0; k < BlockCountK; ++k) {
                for (size_t kk = 0; kk < PerBlkSubblkCount; kk++) {
                    const __m256i av_00_epi8 = _mm256_loadu_si256((const __m256i*)QuantAPtr);
                    const __m256i av_01_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr + 32));

                    accumulate_blklen64_r1c1blk1_avx2<vnni>(
                        av_00_epi8, av_01_epi8, QuantBDataPtr, QuantAScalePtr, QuantBScalePtr, acc0
                    );

                    // increment block pointers
                    QuantAPtr += SubblkLen;
                    QuantBDataPtr += SubblkDataSizeInBytes;
                }
                QuantAScalePtr++;
                QuantBScalePtr++;
            }

            *SumPtr = hsum_float_8(acc0);
            if (BiasPtr) {
                *SumPtr += *BiasPtr;
            }

            // move to next column
            QuantBDataColPtr += StrideQuantBData;
            QuantBScaleColPtr += StrideQuantBScale;
            BiasPtr += BiasPtr != nullptr ? 1 : 0;
            SumPtr += 1;
        }
    }
}

template <bool vnni>
MLAS_FORCEINLINE void
Q8Int8GemmR1xC1BlkLen64Avx2(
    const size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc
)
{
    constexpr size_t BlkBitWidth = 8;
    [[maybe_unused]] constexpr size_t NCols4 = 4;
    [[maybe_unused]] constexpr size_t NRows2 = 2;
    constexpr size_t SubblkLen = 64;

    const size_t BlkDataSizeInBytes = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t PerBlkSubblkCount = BlkLen / SubblkLen;
    const size_t SubblkDataSizeInBytes = BlkDataSizeInBytes / PerBlkSubblkCount;

    const size_t lda = BlockCountK * BlkLen;
    const size_t StrideQuantBData = BlockCountK * BlkDataSizeInBytes;
    const size_t StrideQuantBScale = BlockCountK;

    assert(CountM < NRows2);
    assert(CountN < NCols4);

    for (size_t m = 0; m < CountM; m++) {
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const float* BiasPtr = Bias;
        auto* SumPtr = C + m * ldc;

        for (size_t n = 0; n < CountN; n++) {
            const std::byte* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;
            const std::byte* QuantBDataPtr = QuantBDataColPtr;
            const float* QuantBScalePtr = QuantBScaleColPtr;

            __m256 acc0 = _mm256_setzero_ps();
            for (size_t k = 0; k < BlockCountK; ++k) {
                const float scale_a0 = *QuantAScalePtr;
                const float scale_a0b0 = (*QuantBScalePtr) * scale_a0;

                __m256i av_00_epi8 = _mm256_load_si256((const __m256i*)QuantAPtr);
                __m256i av_01_epi8 = _mm256_load_si256((const __m256i*)(QuantAPtr + 32));

                __m256i bv00_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr));
                __m256i bv01_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 32));

                for (size_t kk = 0; kk < PerBlkSubblkCount - 1; kk++) {
                    accumulate_q8_blklen64_r1c1blk1_avx2<vnni>(av_00_epi8, av_01_epi8, bv00_32_epi8, bv01_32_epi8, scale_a0b0, acc0);

                    // increment block pointers
                    QuantAPtr += SubblkLen;
                    QuantBDataPtr += SubblkDataSizeInBytes;

                    av_00_epi8 = _mm256_load_si256((const __m256i*)QuantAPtr);
                    av_01_epi8 = _mm256_load_si256((const __m256i*)(QuantAPtr + 32));

                    bv00_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr));
                    bv01_32_epi8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr + 32));
                }

                accumulate_q8_blklen64_r1c1blk1_avx2<vnni>(av_00_epi8, av_01_epi8, bv00_32_epi8, bv01_32_epi8, scale_a0b0, acc0);
                QuantAPtr += SubblkLen;
                QuantBDataPtr += SubblkDataSizeInBytes;

                QuantAScalePtr++;
                QuantBScalePtr++;
            }

            *SumPtr = hsum_float_8(acc0);
            if (BiasPtr) {
                *SumPtr += *BiasPtr;
            }

            // move to next column
            QuantBDataColPtr += StrideQuantBData;
            QuantBScaleColPtr += StrideQuantBScale;
            BiasPtr += BiasPtr != nullptr ? 1 : 0;
            SumPtr += 1;
        }
    }
}

template <bool vnni>
MLAS_FORCEINLINE size_t
MlasQ4Int8GemmKernelBlkLen64Avx2(
    const size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc
)
{
    constexpr size_t BlkBitWidth4 = 4;
    constexpr size_t NCols4 = 4;
    constexpr size_t NRows2 = 2;

    const size_t lda = BlockCountK * BlkLen * sizeof(int8_t);
    const size_t lda_scale = BlockCountK;
    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
    const size_t StrideQuantBScale = BlockCountK;

    size_t remainingRows = CountM % NRows2;
    size_t multipleRows = CountM - remainingRows;
    size_t remainingCols = CountN % NCols4;
    size_t multipleCols = CountN - remainingCols;

    if (multipleRows > 0 && multipleCols > 0) {
        Q4Int8GemmR2xC4BlkLen64Avx2<vnni>(
            BlkLen,
            QuantA,
            QuantAScale,
            QuantBData,
            QuantBScale,
            C,
            multipleRows,
            multipleCols,
            BlockCountK,
            Bias,
            ldc
        );
    }
    if (remainingCols > 0 && multipleRows > 0) {
        Q4Int8GemmR2xC1BlkLen64Avx2<vnni>(
            BlkLen,
            QuantA,
            QuantAScale,
            QuantBData + multipleCols * StrideQuantBData,
            QuantBScale + multipleCols * StrideQuantBScale,
            C + multipleCols,
            multipleRows,
            remainingCols,
            BlockCountK,
            Bias ? Bias + multipleCols : nullptr,
            ldc);
    }

    if (remainingRows > 0 && multipleCols > 0) {
        Q4Int8GemmR1xC4BlkLen64Avx2<vnni>(
            BlkLen,
            QuantA + multipleRows * lda,
            QuantAScale + multipleRows * lda_scale,
            QuantBData,
            QuantBScale,
            C + multipleRows * ldc,
            remainingRows,
            multipleCols,
            BlockCountK,
            Bias,
            ldc);
    }

    if (remainingCols > 0 && remainingRows > 0) {
        Q4Int8GemmR1xC1BlkLen64Avx2<vnni>(
            BlkLen,
            QuantA + multipleRows * lda,
            QuantAScale + multipleRows * lda_scale,
            QuantBData + multipleCols * StrideQuantBData,
            QuantBScale + multipleCols * StrideQuantBScale,
            C + multipleRows * ldc + multipleCols,
            remainingRows,
            remainingCols,
            BlockCountK,
            Bias ? Bias + multipleCols : nullptr,
            ldc);
    }

    return CountM;
}

template <bool vnni>
MLAS_FORCEINLINE size_t
MlasQ8Int8GemmKernelBlkLen64Avx2(
    const size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc
)
{
    constexpr size_t BlkBitWidth = 8;
    constexpr size_t NCols4 = 4;
    constexpr size_t NRows2 = 2;

    const size_t lda = BlockCountK * BlkLen * sizeof(int8_t);
    const size_t lda_scale = BlockCountK;
    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t StrideQuantBScale = BlockCountK;

    size_t remainingRows = CountM % NRows2;
    size_t multipleRows = CountM - remainingRows;
    size_t remainingCols = CountN % NCols4;
    size_t multipleCols = CountN - remainingCols;

    if (multipleRows > 0 && multipleCols > 0) {
        Q8Int8GemmR2xC4BlkLen64Avx2<vnni>(
            BlkLen,
            QuantA,
            QuantAScale,
            QuantBData,
            QuantBScale,
            C,
            multipleRows,
            multipleCols,
            BlockCountK,
            Bias,
            ldc
        );
    }
    if (remainingCols > 0 && multipleRows > 0) {
        Q8Int8GemmR2xC1BlkLen64Avx2<vnni>(
            BlkLen,
            QuantA,
            QuantAScale,
            QuantBData + multipleCols * StrideQuantBData,
            QuantBScale + multipleCols * StrideQuantBScale,
            C + multipleCols,
            multipleRows,
            remainingCols,
            BlockCountK,
            Bias ? Bias + multipleCols : nullptr,
            ldc);
    }

    if (remainingRows > 0 && multipleCols > 0) {
        Q8Int8GemmR1xC4BlkLen64Avx2<vnni>(
            BlkLen,
            QuantA + multipleRows * lda,
            QuantAScale + multipleRows * lda_scale,
            QuantBData,
            QuantBScale,
            C + multipleRows * ldc,
            remainingRows,
            multipleCols,
            BlockCountK,
            Bias,
            ldc);
    }

    if (remainingCols > 0 && remainingRows > 0) {
        Q8Int8GemmR1xC1BlkLen64Avx2<vnni>(
            BlkLen,
            QuantA + multipleRows * lda,
            QuantAScale + multipleRows * lda_scale,
            QuantBData + multipleCols * StrideQuantBData,
            QuantBScale + multipleCols * StrideQuantBScale,
            C + multipleRows * ldc + multipleCols,
            remainingRows,
            remainingCols,
            BlockCountK,
            Bias ? Bias + multipleCols : nullptr,
            ldc);
    }

    return CountM;
}
