#pragma once
#include <algorithm>
#include <cassert>
#include <utility>

#include "qnbitgemm.h"
#include "sqnbitgemm_kernel_avx_common.h"

template <bool HasZeroPoint, bool vnni>
static MLAS_FORCEINLINE void
accumulate_blklen32_r1c1blk1_zp_avx2(
    const __m256i& av_32_epi8,
    const std::byte* QuantBDataPtr,
    const float& combined_scale,
    const std::byte* QuantBZeroPointPtr,
    __m256& acc,
    const __m256i& low_mask
)
{
    // | v0  v16 | v1  v17 | ... | v14 v30 | v15 v31 |
    const __m128i bv_packed0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(QuantBDataPtr));
    __m256i bv_32_epi8 = _mm256_set_m128i(_mm_srli_epi16(bv_packed0, 4), bv_packed0);
    bv_32_epi8 = _mm256_and_si256(low_mask, bv_32_epi8);

    bv_32_epi8 = _mm256_sub_epi8(bv_32_epi8, _mm256_set1_epi8(get_zp<HasZeroPoint>(true, QuantBZeroPointPtr)));

#if !defined(__GNUC__) || (__GNUC__ > 10)
    if constexpr (vnni) {
        const __m256i sum_8_epi32 = _mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), _mm256_sign_epi8(bv_32_epi8, bv_32_epi8), _mm256_sign_epi8(av_32_epi8, bv_32_epi8));
        const __m256 sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);
        acc = _mm256_fmadd_ps(sum_ps, _mm256_set1_ps(combined_scale), acc);
    } else {
#endif
        __m256i one_16_epi16 = _mm256_srli_epi16(_mm256_cmpeq_epi16(bv_32_epi8, bv_32_epi8), 15);
        const __m256i dot_16_epi16 = _mm256_maddubs_epi16(_mm256_sign_epi8(bv_32_epi8, bv_32_epi8), _mm256_sign_epi8(av_32_epi8, bv_32_epi8));
        const __m256i sum_8_epi32 = _mm256_madd_epi16(one_16_epi16, dot_16_epi16);
        const __m256 sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);
        acc = _mm256_fmadd_ps(sum_ps, _mm256_set1_ps(combined_scale), acc);
#if !defined(__GNUC__) || (__GNUC__ > 10)
    }
#endif
}

template<bool vnni>
static MLAS_FORCEINLINE void
accumulate_blklen32_r1c1blk2_zp_avx2(
    const __m256i& av0_32_epi8,
    const __m256i& av1_32_epi8,
    const std::byte* QuantBDataPtr,
    const float* scale_a,
    const float* scale_b,
    const std::byte* QuantBZeroPointPtr,
    __m256& acc0,
    const __m256i& low_mask
)
{
    const __m256i bv_packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr));
    __m256i bv0_32_epi8 = _mm256_and_si256(bv_packed, low_mask);                        // 0~31
    __m256i bv1_32_epi8 = _mm256_and_si256(_mm256_srli_epi16(bv_packed, 4), low_mask);  // 32~63

#if !defined(__GNUC__) || (__GNUC__ > 10)
    if constexpr (vnni) {
        {
            bv0_32_epi8 = _mm256_sub_epi8(bv0_32_epi8, _mm256_set1_epi8(get_zp<true>(true, QuantBZeroPointPtr)));
            __m256i sum_8_epi32 = _mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), _mm256_sign_epi8(bv0_32_epi8, bv0_32_epi8), _mm256_sign_epi8(av0_32_epi8, bv0_32_epi8));
            const __m256 sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);
            const __m256 scale = _mm256_set1_ps(*(scale_a) * *(scale_b));
            acc0 = _mm256_fmadd_ps(sum_ps, scale, acc0);
        }

        {
            bv1_32_epi8 = _mm256_sub_epi8(bv1_32_epi8, _mm256_set1_epi8(get_zp<true>(false, QuantBZeroPointPtr)));
            const __m256 scale = _mm256_set1_ps(*(scale_a + 1) * *(scale_b + 1));
            __m256i sum_8_epi32 = _mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), _mm256_sign_epi8(bv1_32_epi8, bv1_32_epi8), _mm256_sign_epi8(av1_32_epi8, bv1_32_epi8));
            const __m256 sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);
            acc0 = _mm256_fmadd_ps(sum_ps, scale, acc0);
        }
    } else {
#endif
        {
            bv0_32_epi8 = _mm256_sub_epi8(bv0_32_epi8, _mm256_set1_epi8(get_zp<true>(true, QuantBZeroPointPtr)));
            const __m256 scale = _mm256_set1_ps(*(scale_a) * *(scale_b));
            __m256i dot_16_epi16 = _mm256_maddubs_epi16(
                _mm256_sign_epi8(bv0_32_epi8, bv0_32_epi8), _mm256_sign_epi8(av0_32_epi8, bv0_32_epi8)
            );
            __m256i sum_8_epi32 = _mm256_madd_epi16(_mm256_set1_epi16(1), dot_16_epi16);
            const __m256 sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);
            acc0 = _mm256_fmadd_ps(sum_ps, scale, acc0);
        }

        {
            bv1_32_epi8 = _mm256_sub_epi8(bv1_32_epi8, _mm256_set1_epi8(get_zp<true>(false, QuantBZeroPointPtr)));
            const __m256 scale = _mm256_set1_ps(*(scale_a + 1) * *(scale_b + 1));
            __m256i dot_16_epi16 = _mm256_maddubs_epi16(
                _mm256_sign_epi8(bv1_32_epi8, bv1_32_epi8), _mm256_sign_epi8(av1_32_epi8, bv1_32_epi8)
            );
            __m256i sum_8_epi32 = _mm256_madd_epi16(_mm256_set1_epi16(1), dot_16_epi16);
            const __m256 sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);
            acc0 = _mm256_fmadd_ps(sum_ps, scale, acc0);
        }
#if !defined(__GNUC__) || (__GNUC__ > 10)
    }
#endif
}

template<bool vnni>
static MLAS_FORCEINLINE void
accumulate_blklen32_r1c1blk2_zp_is_8_avx2(
    const __m256i& av0_32_epi8,
    const __m256i& av1_32_epi8,
    const std::byte* QuantBDataPtr,
    const float* scale_a,
    const float* scale_b,
    __m256& acc0,
    const __m256i& low_mask,
    const __m256i& bzp8
)
{
  // accumulate_blklen32_r1c1blk2_zp_is_8_avx2 is much faster than
  // accumulate_blklen32_r1c1blk2_zp_is_8_no_bc_avx2:
  // BlkBitWidth:4/BlkLen:32/M:1/N:2560/K:2560/Threads:8/Symmetric:1/HasBias:0/ComputeType:4
  // 36591 vs 40270 ns (the main is 51836 ns). both are not as good as main with genai.
    // TODO: consolidate with accumulate_blklen32_r1c1blk2_avx2 using a zp8 template option
    // | v0 v32 | v1 v33 | ... | v30 v62 | v31 v63 |

    const __m256i bv_packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr));
    __m256i bv0_32_epi8 = _mm256_and_si256(bv_packed, low_mask);                          // 0~31
    __m256i bv1_32_epi8 = _mm256_and_si256(_mm256_srli_epi16(bv_packed, 4), low_mask);    // 32~63

    bv0_32_epi8 = _mm256_sub_epi8(bv0_32_epi8, bzp8);
    bv1_32_epi8 = _mm256_sub_epi8(bv1_32_epi8, bzp8);

#if !defined(__GNUC__) || (__GNUC__ > 10)
    if constexpr (vnni) {
        __m256i dot0_8_epi32 = _mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), _mm256_sign_epi8(bv0_32_epi8, bv0_32_epi8), _mm256_sign_epi8(av0_32_epi8, bv0_32_epi8));
        __m256i dot1_8_epi32 = _mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), _mm256_sign_epi8(bv1_32_epi8, bv1_32_epi8), _mm256_sign_epi8(av1_32_epi8, bv1_32_epi8));
        const __m256i sum_8_epi32 = _mm256_hadd_epi32(dot0_8_epi32, dot1_8_epi32);
        const __m256 sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);

        __m256 scale_a0_2_ps = _mm256_castpd_ps(_mm256_broadcast_sd((double*)scale_a));
        __m256 scale_b_2_ps = _mm256_castpd_ps(_mm256_broadcast_sd((double*)scale_b));
        // 1 0 1 0 1 0 1 0 -> 1 1 0 0 1 1 0 0
        __m256 scale_8_ps = _mm256_permute_ps(_mm256_mul_ps(scale_a0_2_ps, scale_b_2_ps), _MM_SHUFFLE(1, 1, 0, 0));

        acc0 = _mm256_fmadd_ps(sum_ps, scale_8_ps, acc0);
    } else {
#endif
        __m256i dot0_16_epi16 = _mm256_maddubs_epi16(_mm256_sign_epi8(bv0_32_epi8, bv0_32_epi8), _mm256_sign_epi8(av0_32_epi8, bv0_32_epi8));
        __m256i dot1_16_epi16 = _mm256_maddubs_epi16(_mm256_sign_epi8(bv1_32_epi8, bv1_32_epi8), _mm256_sign_epi8(av1_32_epi8, bv1_32_epi8));
        const __m256i sum_16_epi16 = _mm256_hadd_epi16(dot0_16_epi16, dot1_16_epi16);

        const __m256i one_16_epi16 = _mm256_srli_epi16(_mm256_cmpeq_epi16(low_mask, low_mask), 15);
        const __m256i sum_8_epi32 = _mm256_madd_epi16(one_16_epi16, sum_16_epi16);
        const __m256 sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);

        __m256 scale_a0_2_ps = _mm256_castpd_ps(_mm256_broadcast_sd((double*)scale_a));
        __m256 scale_b_2_ps = _mm256_castpd_ps(_mm256_broadcast_sd((double*)scale_b));
        // 1 0 1 0 1 0 1 0 -> 1 1 0 0 1 1 0 0
        __m256 scale_8_ps = _mm256_permute_ps(
            _mm256_mul_ps(scale_a0_2_ps, scale_b_2_ps), _MM_SHUFFLE(1, 1, 0, 0)
        );

        acc0 = _mm256_fmadd_ps(sum_ps, scale_8_ps, acc0);
#if !defined(__GNUC__) || (__GNUC__ > 10)
    }
#endif
}

template <bool vnni>
static MLAS_FORCEINLINE void
accumulate_blklen32_r1c1blk2_zp_is_8_no_bc_avx2(
    const __m256i& av0_32_epi8,
    const __m256i& av1_32_epi8,
    const __m256& scale_a0_8_ps,
    const __m256& scale_a1_8_ps,
    const std::byte* QuantBDataPtr,
    const float* scale_b,
    __m256& acc0,
    const __m256i& low_mask,
    const __m256i& bzp8
)
{
    // TODO: consolidate with accumulate_blklen32_r1c1blk2_avx2 using a zp8 template option
    // | v0 v32 | v1 v33 | ... | v30 v62 | v31 v63 |
    const __m256i bv_packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr));
    __m256i bv0_32_epi8 = _mm256_and_si256(bv_packed, low_mask);                          // 0~31
    __m256i bv1_32_epi8 = _mm256_srli_epi16(_mm256_sub_epi8(bv_packed, bv0_32_epi8), 4);  // 32~63

    bv0_32_epi8 = _mm256_sub_epi8(bv0_32_epi8, bzp8);
    bv1_32_epi8 = _mm256_sub_epi8(bv1_32_epi8, bzp8);

#if !defined(__GNUC__) || (__GNUC__ > 10)
    if constexpr (vnni) {
        {
            __m256i sum_8_epi32 = _mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), _mm256_sign_epi8(bv0_32_epi8, bv0_32_epi8), _mm256_sign_epi8(av0_32_epi8, bv0_32_epi8));
            const __m256 sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);
            const __m256 scale = _mm256_mul_ps(_mm256_set1_ps(*scale_b), scale_a0_8_ps);
            acc0 = _mm256_fmadd_ps(sum_ps, scale, acc0);
        }
        {
            __m256i sum_8_epi32 = _mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), _mm256_sign_epi8(bv1_32_epi8, bv1_32_epi8), _mm256_sign_epi8(av1_32_epi8, bv1_32_epi8));
            const __m256 sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);
            const __m256 scale = _mm256_mul_ps(_mm256_set1_ps(*(scale_b + 1)), scale_a1_8_ps);
            acc0 = _mm256_fmadd_ps(sum_ps, scale, acc0);
        }
    } else {
#endif
        {
            __m256i dot0_16_epi16 = _mm256_maddubs_epi16(_mm256_sign_epi8(bv0_32_epi8, bv0_32_epi8), _mm256_sign_epi8(av0_32_epi8, bv0_32_epi8));
            __m256i sum_8_epi32 = _mm256_madd_epi16(_mm256_set1_epi16(1), dot0_16_epi16);
            const __m256 sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);

            const __m256 scale = _mm256_mul_ps(_mm256_set1_ps(*scale_b), scale_a0_8_ps);
            acc0 = _mm256_fmadd_ps(sum_ps, scale, acc0);
        }
        {
            __m256i dot0_16_epi16 = _mm256_maddubs_epi16(_mm256_sign_epi8(bv1_32_epi8, bv1_32_epi8), _mm256_sign_epi8(av1_32_epi8, bv1_32_epi8));
            __m256i sum_8_epi32 = _mm256_madd_epi16(_mm256_set1_epi16(1), dot0_16_epi16);
            const __m256 sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);

            const __m256 scale = _mm256_mul_ps(_mm256_set1_ps(*(scale_b + 1)), scale_a1_8_ps);
            acc0 = _mm256_fmadd_ps(sum_ps, scale, acc0);
        }
#if !defined(__GNUC__) || (__GNUC__ > 10)
    }
#endif
}

template <bool HasZeroPoint, bool vnni>
MLAS_FORCEINLINE void
Q4Int8GemmM1C4BlkLen32Avx2(
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias)
{
    constexpr size_t BlkLen32 = 32;
    constexpr size_t BlkBitWidth4 = 4;
    constexpr size_t NCols4 = 4;
    constexpr size_t BlkDataSizeInBytes16 = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);

    // process 2 blks of 64 4b weights a time
    constexpr size_t PerAccuBlk2 = 2;

    //const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);
    //const size_t StrideQuantBScale = BlockCountK;

    assert(CountN % NCols4 == 0);

    const std::byte* QuantBDataColPtr = QuantBData;
    const float* QuantBScaleColPtr = QuantBScale;
    const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;
    const float* BiasPtr = Bias;
    auto* SumPtr = C;

    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    //const __m256i one_16_epi16 = _mm256_srli_epi16(_mm256_cmpeq_epi16(low_mask, low_mask), 15);
    const size_t StrideQuantBDataCol = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);
    const size_t StrideQuantBData2 = 2 * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);
    const size_t StrideQuantBData1 = 1 * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);
    const size_t StrideQuantBScale2 = 2;
    const size_t StrideQuantBScale1 = 1;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth4>(BlockCountK);


    for (size_t n = 0; n < CountN; n += NCols4) {
        const std::byte* QuantAPtr = QuantA;
        const float* QuantAScalePtr = QuantAScale;

        const std::byte* QuantBDataPtr = QuantBDataColPtr;
        const float* QuantBScalePtr = QuantBScaleColPtr;
        const std::byte* QuantBZeroPointPtr = QuantBZeroPointColPtr;

        __m256 acc[NCols4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
        size_t k_blks_remaining = BlockCountK;
        for (; k_blks_remaining > 1; k_blks_remaining -= PerAccuBlk2) {
            const __m256i av_00_epi8 = _mm256_loadu_si256((const __m256i*)QuantAPtr);
            const __m256i av_01_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr + BlkLen32));
            //const __m256 scale_a0_8_ps = _mm256_set1_ps(Q8BlkScale(QuantAPtr));
            //const __m256 scale_a1_8_ps = _mm256_set1_ps(Q8BlkScale(QuantAPtr + Q8BlkSize(BlkLen32)));

            //accumulate_blklen32_r1c1blk2_zp_is_8_no_bc_avx2(av_00_epi8, av_01_epi8, scale_a0_8_ps, scale_a1_8_ps, QuantBDataPtr, QuantBScalePtr, acc[0], low_mask, bzp8);
            //accumulate_blklen32_r1c1blk2_zp_is_8_no_bc_avx2(av_00_epi8, av_01_epi8, scale_a0_8_ps, scale_a1_8_ps, QuantBDataPtr + StrideQuantBData, QuantBScalePtr + StrideQuantBScale, acc[1], low_mask, bzp8);
            //accumulate_blklen32_r1c1blk2_zp_is_8_no_bc_avx2(av_00_epi8, av_01_epi8, scale_a0_8_ps, scale_a1_8_ps, QuantBDataPtr + 2 * StrideQuantBData, QuantBScalePtr + 2 * StrideQuantBScale, acc[2], low_mask, bzp8);
            //accumulate_blklen32_r1c1blk2_zp_is_8_no_bc_avx2(av_00_epi8, av_01_epi8, scale_a0_8_ps, scale_a1_8_ps, QuantBDataPtr + 3 * StrideQuantBData, QuantBScalePtr + 3 * StrideQuantBScale, acc[3], low_mask, bzp8);
            if constexpr (HasZeroPoint) {
                accumulate_blklen32_r1c1blk2_zp_avx2<vnni>(av_00_epi8, av_01_epi8, QuantBDataPtr, QuantAScalePtr, QuantBScalePtr, QuantBZeroPointPtr, acc[0], low_mask);
                accumulate_blklen32_r1c1blk2_zp_avx2<vnni>(av_00_epi8, av_01_epi8, QuantBDataPtr + StrideQuantBData2, QuantAScalePtr, QuantBScalePtr + StrideQuantBScale2, QuantBZeroPointPtr + StrideQuantBZeroPoint, acc[1], low_mask);
                accumulate_blklen32_r1c1blk2_zp_avx2<vnni>(av_00_epi8, av_01_epi8, QuantBDataPtr + 2 * StrideQuantBData2, QuantAScalePtr, QuantBScalePtr + 2 * StrideQuantBScale2, QuantBZeroPointPtr + 2 * StrideQuantBZeroPoint, acc[2], low_mask);
                accumulate_blklen32_r1c1blk2_zp_avx2<vnni>(av_00_epi8, av_01_epi8, QuantBDataPtr + 3 * StrideQuantBData2, QuantAScalePtr, QuantBScalePtr + 3 * StrideQuantBScale2, QuantBZeroPointPtr + 3 * StrideQuantBZeroPoint, acc[3], low_mask);

            } else {
                const __m256i bzp8 = _mm256_set1_epi8(8);
                accumulate_blklen32_r1c1blk2_zp_is_8_avx2<vnni>(av_00_epi8, av_01_epi8, QuantBDataPtr, QuantAScalePtr, QuantBScalePtr, acc[0], low_mask, bzp8);
                accumulate_blklen32_r1c1blk2_zp_is_8_avx2<vnni>(av_00_epi8, av_01_epi8, QuantBDataPtr + StrideQuantBData2, QuantAScalePtr, QuantBScalePtr + StrideQuantBScale2, acc[1], low_mask, bzp8);
                accumulate_blklen32_r1c1blk2_zp_is_8_avx2<vnni>(av_00_epi8, av_01_epi8, QuantBDataPtr + 2 * StrideQuantBData2, QuantAScalePtr, QuantBScalePtr + 2 * StrideQuantBScale2, acc[2], low_mask, bzp8);
                accumulate_blklen32_r1c1blk2_zp_is_8_avx2<vnni>(av_00_epi8, av_01_epi8, QuantBDataPtr + 3 * StrideQuantBData2, QuantAScalePtr, QuantBScalePtr + 3 * StrideQuantBScale2, acc[3], low_mask, bzp8);
            }
            // increment block pointers
            QuantAPtr += BlkLen32 * PerAccuBlk2;
            QuantAScalePtr += PerAccuBlk2;
            QuantBDataPtr += BlkDataSizeInBytes16 * PerAccuBlk2 * NCols4;
            QuantBScalePtr += PerAccuBlk2 * NCols4;
            if constexpr (HasZeroPoint) {
                QuantBZeroPointPtr += 1;
            }
        }

        // TODO: use a loop in case PerAccuBlk2 is not 2.
        if (k_blks_remaining > 0) {
            // load A
            const std::byte* QuantABlk0 = QuantAPtr;
            const __m256i av_00_epi8 = _mm256_loadu_si256((const __m256i*)QuantABlk0);
            const float& scale_a00 = *QuantAScalePtr;
            {
                const float& scale_00 = scale_a00 * (QuantBScalePtr)[0];
                accumulate_blklen32_r1c1blk1_zp_avx2<HasZeroPoint, vnni>(av_00_epi8, QuantBDataPtr, scale_00, QuantBZeroPointPtr, acc[0], low_mask);
            }
            {
                const float& scale_00 = scale_a00 * (QuantBScalePtr + StrideQuantBScale1)[0];
                accumulate_blklen32_r1c1blk1_zp_avx2<HasZeroPoint, vnni>(av_00_epi8, QuantBDataPtr + StrideQuantBData1, scale_00, QuantBZeroPointPtr + StrideQuantBZeroPoint, acc[1], low_mask);
            }
            {
                const float& scale_00 = scale_a00 * (QuantBScalePtr + 2 * StrideQuantBScale1)[0];
                accumulate_blklen32_r1c1blk1_zp_avx2<HasZeroPoint, vnni>(av_00_epi8, QuantBDataPtr + 2 * StrideQuantBData1, scale_00, QuantBZeroPointPtr + 2 * StrideQuantBZeroPoint, acc[2], low_mask);
            }
            {
                const float& scale_00 = scale_a00 * (QuantBScalePtr + 3 * StrideQuantBScale1)[0];
                accumulate_blklen32_r1c1blk1_zp_avx2<HasZeroPoint, vnni>(av_00_epi8, QuantBDataPtr + 3 * StrideQuantBData1, scale_00, QuantBZeroPointPtr + 3 * StrideQuantBZeroPoint, acc[3], low_mask);
            }
        }

        __m128 acc_r0 = FoldAccumulators(acc[0], acc[1], acc[2], acc[3]);
        if (BiasPtr != nullptr) {
            acc_r0 = _mm_add_ps(acc_r0, _mm_loadu_ps(BiasPtr));
        }

        _mm_storeu_ps(SumPtr, acc_r0);

        // move to next NCols columns
        QuantBDataColPtr += NCols4 * StrideQuantBDataCol;
        QuantBScaleColPtr += NCols4 * BlockCountK;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointColPtr += NCols4 * StrideQuantBZeroPoint;
        }

        BiasPtr += BiasPtr != nullptr ? NCols4 : 0;
        SumPtr += NCols4;
    }
}

template <bool HasZeroPoint, bool vnni>
MLAS_FORCEINLINE void
Q4Int8GemmM1C1BlkLen32Avx2(
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias
)
{
    constexpr size_t BlkLen32 = 32;
    constexpr size_t BlkBitWidth4 = 4;
    [[maybe_unused]] constexpr size_t NCols4 = 4;
    constexpr size_t BlkDataSizeInBytes16 = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);

    // process 2 blks of 64 4b weights a time
    constexpr size_t PerAccuBlk2 = 2;

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);
    const size_t StrideQuantBScale = BlockCountK;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth4>(BlockCountK);

    [[maybe_unused]] size_t QuantBZeroPointIdx = 0;  // track half byte increments with this index instead of a pointer
    assert(CountN < NCols4);

    const std::byte* QuantBDataColPtr = QuantBData;
    const float* QuantBScaleColPtr = QuantBScale;
    const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;
    const float* BiasPtr = Bias;
    auto* SumPtr = C;

    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    [[maybe_unused]] const __m256i bzp8 = _mm256_set1_epi8(8);
    for (size_t n = 0; n < CountN; n++) {
        const std::byte* QuantAPtr = QuantA;
        const float* QuantAScalePtr = QuantAScale;
        const std::byte* QuantBDataPtr = QuantBDataColPtr;
        const float* QuantBScalePtr = QuantBScaleColPtr;
        const std::byte* QuantBZeroPointPtr = QuantBZeroPointColPtr;

        __m256 acc0 = _mm256_setzero_ps();
        size_t k_blks_remaining = BlockCountK;
        for (; k_blks_remaining > 1; k_blks_remaining -= PerAccuBlk2) {
            const __m256i av_00_epi8 = _mm256_loadu_si256((const __m256i*)QuantAPtr);
            const __m256i av_01_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr + BlkLen32));
            //const __m256 scale_a0_8_ps = _mm256_set1_ps(Q8BlkScale(QuantAPtr));
            //const __m256 scale_a1_8_ps = _mm256_set1_ps(Q8BlkScale(QuantAPtr + Q8BlkSize(BlkLen32)));

            if constexpr (HasZeroPoint) {
                accumulate_blklen32_r1c1blk2_zp_avx2<vnni>(av_00_epi8, av_01_epi8, QuantBDataPtr, QuantAScalePtr, QuantBScalePtr, QuantBZeroPointPtr, acc0, low_mask);
            } else {
                accumulate_blklen32_r1c1blk2_zp_is_8_avx2<vnni>(av_00_epi8, av_01_epi8, QuantBDataPtr, QuantAScalePtr, QuantBScalePtr, acc0, low_mask, bzp8);
            }

            // increment block pointers
            QuantAPtr += BlkLen32 * PerAccuBlk2;
            QuantAScalePtr += PerAccuBlk2;
            QuantBDataPtr += BlkDataSizeInBytes16 * PerAccuBlk2;
            QuantBScalePtr += PerAccuBlk2;
            if constexpr (HasZeroPoint) {
                QuantBZeroPointPtr += 1;
            }
        }

        // TODO: use a loop in case PerAccuBlk2 is not 2.
        if (k_blks_remaining > 0) {
            const __m256i av_00_epi8 = _mm256_loadu_si256((const __m256i*)QuantAPtr);
            const float& scale_a00 = *QuantAScalePtr;
            const float& scale_00 = scale_a00 * (QuantBScalePtr)[0];
            accumulate_blklen32_r1c1blk1_zp_avx2<HasZeroPoint, vnni>(av_00_epi8, QuantBDataPtr, scale_00, QuantBZeroPointPtr, acc0, low_mask);
        }

        *SumPtr = hsum_float_8(acc0);
        if (BiasPtr) {
            *SumPtr += *BiasPtr;
        }

        // move to next column
        QuantBDataColPtr += StrideQuantBData;
        QuantBScaleColPtr += StrideQuantBScale;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointColPtr += StrideQuantBZeroPoint;
        }

        BiasPtr += BiasPtr != nullptr ? 1 : 0;
        SumPtr += 1;
    }
}

template <bool HasZeroPoint, bool vnni>
MLAS_FORCEINLINE
void
MlasQ4Int8GemmM1KernelBlkLen32Avx2(
        const std::byte* QuantA,
        const float* QuantAScale,
        const std::byte* QuantBData,
        const float* QuantBScale,
        const std::byte* QuantBZeroPoint,
        float* C,
        size_t CountN,
        size_t BlockCountK,
        const float* Bias
    )
{
    constexpr size_t BlkLen32 = 32;
    constexpr size_t BlkBitWidth4 = 4;
    constexpr size_t NCols4 = 4;

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);
    const size_t StrideQuantBScale = BlockCountK;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth4>(BlockCountK);

    size_t remainingCols = CountN % NCols4;
    size_t multipleCols = CountN - remainingCols;

    if (multipleCols > 0) {
        Q4Int8GemmM1C4BlkLen32Avx2<HasZeroPoint, vnni>(
            QuantA,
            QuantAScale,
            QuantBData,
            QuantBScale,
            QuantBZeroPoint,
            C,
            multipleCols,
            BlockCountK,
            Bias);
    }

    if (remainingCols > 0) {
        Q4Int8GemmM1C1BlkLen32Avx2<HasZeroPoint, vnni>(
            QuantA,
            QuantAScale,
            QuantBData + multipleCols * StrideQuantBData,
            QuantBScale + multipleCols * StrideQuantBScale,
            QuantBZeroPoint + multipleCols * StrideQuantBZeroPoint,
            C + multipleCols,
            remainingCols,
            BlockCountK,
            Bias ? Bias + multipleCols : nullptr);
    }
}

//#define SQ4BitGemmM1Kernel_BlkLen32_CompInt8_NewLayout 1
void SQ4BitGemmM1Kernel_BlkLen32_CompInt8_Impl2(
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias
)
{
    // port from neon implementation
    constexpr size_t BlkBitWidth = 4;
    constexpr size_t BlkLen = 32;
#if defined SQ4BitGemmM1Kernel_BlkLen32_CompInt8_NewLayout
#else
    constexpr bool HasZeroPoint = false;
#endif

    float* CRowPtr = C;

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);
    //const size_t StrideQuantBScale = BlockCountK;
    const float* BiasPtr = Bias;

    const std::byte* QuantBDataColPtr = QuantBData;
    const float* QuantBScaleColPtr = QuantBScale;
    const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

    float* SumPtr = CRowPtr;

#if defined SQ4BitGemmM1Kernel_BlkLen32_CompInt8_NewLayout
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i bzp8 = _mm256_set1_epi8(8);
    const __m256i one_16_epi16 = _mm256_srli_epi16(_mm256_cmpeq_epi16(low_mask, low_mask), 15);
    (void)StrideQuantBZeroPoint;
#else
    const __m256i zero = _mm256_setzero_si256();
    const __m128i low_mask = _mm_set1_epi8(0xF);
#endif
    const size_t NCols = 4;
    constexpr size_t StrideQuantBScale2 = 2;
    constexpr size_t StrideQuantBScale1 = 1;

    int64_t nblk = (int64_t)(CountN)-4;
    while (nblk >= 0) {
        const std::byte* QuantAPtr = QuantA;
        const float* QuantAScalePtr = QuantAScale;
        const std::byte* QuantBDataPtr = QuantBDataColPtr;
        const float* QuantBScalePtr = QuantBScaleColPtr;
        const std::byte* QuantBZeroPointPtr = QuantBZeroPointColPtr;

#if defined SQ4BitGemmM1Kernel_BlkLen32_CompInt8_NewLayout
        (void)QuantBZeroPointPtr;
#endif
        __m256
            acc0 = _mm256_setzero_ps(),
            acc1 = _mm256_setzero_ps(),
            acc2 = _mm256_setzero_ps(),
            acc3 = _mm256_setzero_ps();

        size_t k_blks_remaining = BlockCountK;
        for (; k_blks_remaining > 1; k_blks_remaining -= 2) {
            const std::byte* QuantABlk0 = QuantAPtr;
            const std::byte* QuantABlk1 = QuantABlk0 + BlkLen;

            // load A:
            const __m256i av_0_epi8 = _mm256_loadu_si256((const __m256i*)QuantABlk0);
            const __m256i av_1_epi8 = _mm256_loadu_si256((const __m256i*)QuantABlk1);
#if defined SQ4BitGemmM1Kernel_BlkLen32_CompInt8_NewLayout
            const __m256 scale_a0_8_ps = _mm256_set1_ps(Q8BlkScale(QuantAPtr));
            const __m256 scale_a1_8_ps = _mm256_set1_ps(Q8BlkScale(QuantAPtr + Q8BlkSize(BlkLen)));
#else
            const float& scale_a0 = QuantAScalePtr[0];
            const float& scale_a1 = QuantAScalePtr[1];
#endif

            // Col0
#if defined SQ4BitGemmM1Kernel_BlkLen32_CompInt8_NewLayout
            accumulate_blklen32_r1c1blk2_zp_is_8_no_bc_avx2(av_0_epi8, av_1_epi8, scale_a0_8_ps, scale_a1_8_ps, QuantBDataPtr, QuantBScalePtr, acc0, low_mask, bzp8);
#else
            const float& scale_00 = scale_a0 * QuantBScalePtr[0];
            const float& scale_01 = scale_a1 * QuantBScalePtr[1];
            accumulate_mul_sum_avx2<HasZeroPoint>(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr), low_mask, zero, QuantBZeroPointPtr, true, scale_00, acc0);
            accumulate_mul_sum_avx2<HasZeroPoint>(av_1_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 16), low_mask, zero, QuantBZeroPointPtr, false, scale_01, acc0);
#endif

            // Col1
#if defined SQ4BitGemmM1Kernel_BlkLen32_CompInt8_NewLayout
            accumulate_blklen32_r1c1blk2_zp_is_8_no_bc_avx2(av_0_epi8, av_1_epi8, scale_a0_8_ps, scale_a1_8_ps, QuantBDataPtr + StrideQuantBData, QuantBScalePtr + StrideQuantBScale2, acc1, low_mask, bzp8);
#else
            const float& scale_10 = scale_a0 * (QuantBScalePtr + StrideQuantBScale2)[0];
            const float& scale_11 = scale_a1 * (QuantBScalePtr + StrideQuantBScale2)[1];
            accumulate_mul_sum_avx2<HasZeroPoint>(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + StrideQuantBData), low_mask, zero, QuantBZeroPointPtr + StrideQuantBZeroPoint, true, scale_10, acc1);
            accumulate_mul_sum_avx2<HasZeroPoint>(av_1_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + StrideQuantBData + 16), low_mask, zero, QuantBZeroPointPtr + StrideQuantBZeroPoint, false, scale_11, acc1);
#endif

            // Col2
#if defined SQ4BitGemmM1Kernel_BlkLen32_CompInt8_NewLayout
            accumulate_blklen32_r1c1blk2_zp_is_8_no_bc_avx2(av_0_epi8, av_1_epi8, scale_a0_8_ps, scale_a1_8_ps, QuantBDataPtr + 2 * StrideQuantBData, QuantBScalePtr + 2 * StrideQuantBScale2, acc2, low_mask, bzp8);
#else
            const float& scale_20 = scale_a0 * (QuantBScalePtr + 2 * StrideQuantBScale2)[0];
            const float& scale_21 = scale_a1 * (QuantBScalePtr + 2 * StrideQuantBScale2)[1];
            accumulate_mul_sum_avx2<HasZeroPoint>(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 2 * StrideQuantBData), low_mask, zero, QuantBZeroPointPtr + 2 * StrideQuantBZeroPoint, true, scale_20, acc2);
            accumulate_mul_sum_avx2<HasZeroPoint>(av_1_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 2 * StrideQuantBData + 16), low_mask, zero, QuantBZeroPointPtr + 2 * StrideQuantBZeroPoint, false, scale_21, acc2);
#endif
            // Col3
#if defined SQ4BitGemmM1Kernel_BlkLen32_CompInt8_NewLayout
            accumulate_blklen32_r1c1blk2_zp_is_8_no_bc_avx2(av_0_epi8, av_1_epi8, scale_a0_8_ps, scale_a1_8_ps, QuantBDataPtr + 3 * StrideQuantBData, QuantBScalePtr + 3 * StrideQuantBScale2, acc3, low_mask, bzp8);
#else
            const float& scale_30 = scale_a0 * (QuantBScalePtr + 3 * StrideQuantBScale2)[0];
            const float& scale_31 = scale_a1 * (QuantBScalePtr + 3 * StrideQuantBScale2)[1];
            accumulate_mul_sum_avx2<HasZeroPoint>(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 3 * StrideQuantBData), low_mask, zero, QuantBZeroPointPtr + 3 * StrideQuantBZeroPoint, true, scale_30, acc3);
            accumulate_mul_sum_avx2<HasZeroPoint>(av_1_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 3 * StrideQuantBData + 16), low_mask, zero, QuantBZeroPointPtr + 3 * StrideQuantBZeroPoint, false, scale_31, acc3);
#endif
            // increment block pointers
            QuantAPtr += BlkLen * 2;
            QuantAScalePtr += 2;
            QuantBDataPtr += 16 * 2;
            QuantBScalePtr += 2 * NCols;
        }

        if (k_blks_remaining > 0) {
            // load A
            const std::byte* QuantABlk0 = QuantAPtr;
            const __m256i av_0_epi8 = _mm256_loadu_si256((const __m256i*)QuantABlk0);

            const float& scale_a0 = *QuantAScalePtr;

            // Col0
            const float& scale_0 = scale_a0 * QuantBScalePtr[0];
#if defined SQ4BitGemmM1Kernel_BlkLen32_CompInt8_NewLayout
            accumulate_blklen32_r1c1blk1_zp_avx2(av_0_epi8, QuantBDataPtr, scale_0, acc0, low_mask, bzp8);
#else
            accumulate_mul_sum_avx2<HasZeroPoint>(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr), low_mask, zero, QuantBZeroPointPtr, true, scale_0, acc0);
#endif

            // Col1
            const float& scale_1 = scale_a0 * (QuantBScalePtr + StrideQuantBScale1)[0];
#if defined SQ4BitGemmM1Kernel_BlkLen32_CompInt8_NewLayout
            accumulate_blklen32_r1c1blk1_zp_avx2(av_0_epi8, QuantBDataPtr + StrideQuantBData, scale_1, acc1, low_mask, bzp8);
#else
            accumulate_mul_sum_avx2<HasZeroPoint>(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + StrideQuantBData), low_mask, zero, QuantBZeroPointPtr + StrideQuantBZeroPoint, true, scale_1, acc1);
#endif

            // Col2
            const float& scale_2 = scale_a0 * (QuantBScalePtr + 2 * StrideQuantBScale1)[0];
#if defined SQ4BitGemmM1Kernel_BlkLen32_CompInt8_NewLayout
            accumulate_blklen32_r1c1blk1_zp_avx2(av_0_epi8, QuantBDataPtr + 2 * StrideQuantBData, scale_2, acc2, low_mask, bzp8);
#else
            accumulate_mul_sum_avx2<HasZeroPoint>(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 2 * StrideQuantBData), low_mask, zero, QuantBZeroPointPtr + 2 * StrideQuantBZeroPoint, true, scale_2, acc2);
#endif

            // Col3
            const float& scale_3 = scale_a0 * (QuantBScalePtr + 3 * StrideQuantBScale1)[0];
#if defined SQ4BitGemmM1Kernel_BlkLen32_CompInt8_NewLayout
            accumulate_blklen32_r1c1blk1_zp_avx2(av_0_epi8, QuantBDataPtr + 3 * StrideQuantBData, scale_3, acc3, low_mask, bzp8);
#else
            accumulate_mul_sum_avx2<HasZeroPoint>(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 3 * StrideQuantBData), low_mask, zero, QuantBZeroPointPtr + 3 * StrideQuantBZeroPoint, true, scale_3, acc3);
#endif
        }

        __m128 acc_x = FoldAccumulators(acc0, acc1, acc2, acc3);
        if (BiasPtr != nullptr) {
            acc_x = _mm_add_ps(acc_x, _mm_loadu_ps(BiasPtr));
        }
        _mm_storeu_ps(SumPtr, acc_x);

        // move to next NCols columns

        QuantBDataColPtr += NCols * StrideQuantBData;
        QuantBScaleColPtr += NCols * BlockCountK;

        BiasPtr += BiasPtr != nullptr ? NCols : 0;
        SumPtr += NCols;
        nblk -= NCols;
    }

    nblk += NCols;
    for (int64_t n = 0; n < nblk; n++) {
        const std::byte* QuantAPtr = QuantA;
        const float* QuantAScalePtr = QuantAScale;
        const std::byte* QuantBDataPtr = QuantBDataColPtr;
        const float* QuantBScalePtr = QuantBScaleColPtr;
        const std::byte* QuantBZeroPointPtr = QuantBZeroPointColPtr;

#if defined SQ4BitGemmM1Kernel_BlkLen32_CompInt8_NewLayout
        (void)QuantBZeroPointPtr;
#endif
        __m256 acc0 = _mm256_setzero_ps();

        size_t k_blks_remaining = BlockCountK;
        for (; k_blks_remaining > 1; k_blks_remaining -= 2) {
            const std::byte* QuantABlk0 = QuantAPtr;
            const std::byte* QuantABlk1 = QuantABlk0 + BlkLen;

            // load A:
            const __m256i av_0_epi8 = _mm256_loadu_si256((const __m256i*)QuantABlk0);
            const __m256i av_1_epi8 = _mm256_loadu_si256((const __m256i*)QuantABlk1);

#if defined SQ4BitGemmM1Kernel_BlkLen32_CompInt8_NewLayout
            const __m256 scale_a0_8_ps = _mm256_set1_ps(Q8BlkScale(QuantABlk0));
            const __m256 scale_a1_8_ps = _mm256_set1_ps(Q8BlkScale(QuantABlk1));
#else
            const float& scale_a0 = QuantAScalePtr[0];
            const float& scale_a1 = QuantAScalePtr[1];
#endif

            // Col0
#if defined SQ4BitGemmM1Kernel_BlkLen32_CompInt8_NewLayout
            accumulate_blklen32_r1c1blk2_zp_is_8_no_bc_avx2(av_0_epi8, av_1_epi8, scale_a0_8_ps, scale_a1_8_ps, QuantBDataPtr, QuantBScalePtr, acc0, low_mask, bzp8);
#else
            const float& scale_00 = scale_a0 * QuantBScalePtr[0];
            const float& scale_01 = scale_a1 * QuantBScalePtr[1];
            accumulate_mul_sum_avx2<HasZeroPoint>(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr), low_mask, zero, QuantBZeroPointPtr, true, scale_00, acc0);
            accumulate_mul_sum_avx2<HasZeroPoint>(av_1_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 16), low_mask, zero, QuantBZeroPointPtr, false, scale_01, acc0);
#endif
            // increment block pointers
            QuantAPtr += BlkLen * 2;
            QuantAScalePtr += 2;
            QuantBDataPtr += 16 * 2;
            QuantBScalePtr += 2;
        }

        if (k_blks_remaining > 0) {
            // load A
            const std::byte* QuantABlk0 = QuantAPtr;
            const __m256i av_0_epi8 = _mm256_loadu_si256((const __m256i*)QuantABlk0);

            const float& scale_a0 = *QuantAScalePtr;

            // Col0
            const float& scale_00 = scale_a0 * QuantBScalePtr[0];
#if defined SQ4BitGemmM1Kernel_BlkLen32_CompInt8_NewLayout
            accumulate_blklen32_r1c1blk1_zp_avx2(av_0_epi8, QuantBDataPtr, scale_00, acc0, low_mask, bzp8);
#else
            accumulate_mul_sum_avx2<HasZeroPoint>(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr), low_mask, zero, QuantBZeroPointPtr, true, scale_00, acc0);
#endif
        }

        *SumPtr = hsum_float_8(acc0);
        if (BiasPtr) {
            *SumPtr += *BiasPtr;
        }

        // move to next column

        QuantBDataColPtr += StrideQuantBData;
        QuantBScaleColPtr += BlockCountK;

        BiasPtr += BiasPtr != nullptr ? 1 : 0;
        SumPtr += 1;
    }
}
