#pragma once
#include <algorithm>
#include <cassert>
#include <utility>

#include "qnbitgemm.h"
#include "sqnbitgemm_kernel_avx_common.h"


MLAS_FORCEINLINE void
accumulate_1blk_dot(const __m256i& av_32_epi8, const __m256i& bv_32_epi8,
  const float& combined_scale, const __m256i& one_16_epi16, __m256& acc)
{
    const __m256i dot_16_epi16 = _mm256_maddubs_epi16(
        bv_32_epi8, av_32_epi8
    );
    const __m256i sum_8_epi32 = _mm256_madd_epi16(one_16_epi16, dot_16_epi16);
    const __m256 sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);
    acc = _mm256_fmadd_ps(sum_ps, _mm256_set1_ps(combined_scale), acc);
}

#if !defined(__GNUC__) || (__GNUC__ > 10)
MLAS_FORCEINLINE void
accumulate_1blk_dot_vnni(const __m256i& av_32_epi8, const __m256i& bv_32_epi8, const float& combined_scale, __m256& acc)
{
    __m256i sum_8_epi32 = _mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), bv_32_epi8, av_32_epi8);
    const __m256 sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);
    acc = _mm256_fmadd_ps(sum_ps, _mm256_set1_ps(combined_scale), acc);
}
#endif

template <bool vnni>
static MLAS_FORCEINLINE void
accumulate_blklen32_r2c1blk2_avx2(
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
    // | v0  v32 | v1  v33 | ... | v30 v62 | v31 v63 |
    const __m256i bv_packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr));

    // generating low_mask of 0x0Fs is not as fast as just calling _mm256_set1_epi8(0x0F).
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    //__m256i low_mask = _mm256_srli_epi16(_mm256_cmpeq_epi16(bv_packed, bv_packed), 12);
    // low_mask = _mm256_packus_epi16(low_mask, low_mask);
    __m256i bv0_32_epi8 = _mm256_and_si256(bv_packed, low_mask);  // 0~31
    // TODO: this (the second line below) is faster and does not keep low_mask in use.
    // const __m256i bv1_32_epi8 = _mm256_and_si256(_mm256_srli_epi16(bv_packed, 4), low_mask);
    __m256i bv1_32_epi8 = _mm256_srli_epi16(_mm256_sub_epi8(bv_packed, bv0_32_epi8), 4);  // 32~63

#if !defined(__GNUC__) || (__GNUC__ > 10)
    if constexpr (vnni) {
        __m256 scale_b_2_ps = _mm256_castpd_ps(_mm256_broadcast_sd((double*)scale_b));
        {
            const __m256i dot0_8_epi32 = _mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), bv0_32_epi8, av00_32_epi8);
            const __m256i dot1_8_epi32 = _mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), bv1_32_epi8, av01_32_epi8);
            const __m256i sum_8_epi32 = _mm256_hadd_epi32(dot0_8_epi32, dot1_8_epi32);
            const __m256 sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);

            __m256 scale_a0_2_ps = _mm256_castpd_ps(_mm256_broadcast_sd((double*)scale_a0));
            // 1 0 1 0 1 0 1 0 -> 1 1 0 0 1 1 0 0
            __m256 scale_8_ps = _mm256_permute_ps(_mm256_mul_ps(scale_a0_2_ps, scale_b_2_ps), _MM_SHUFFLE(1, 1, 0, 0));
            acc0 = _mm256_fmadd_ps(sum_ps, scale_8_ps, acc0);
        }
        {
            const __m256i dot0_8_epi32 = _mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), bv0_32_epi8, av10_32_epi8);
            const __m256i dot1_8_epi32 = _mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), bv1_32_epi8, av11_32_epi8);
            const __m256i sum_8_epi32 = _mm256_hadd_epi32(dot0_8_epi32, dot1_8_epi32);
            const __m256 sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);

            __m256 scale_a1_2_ps = _mm256_castpd_ps(_mm256_broadcast_sd((double*)scale_a1));
            __m256 scale_8_ps = _mm256_permute_ps(_mm256_mul_ps(scale_a1_2_ps, scale_b_2_ps), _MM_SHUFFLE(1, 1, 0, 0));
            acc1 = _mm256_fmadd_ps(sum_ps, scale_8_ps, acc1);
        }
    } else {
#endif
        //{
            const __m256i dot0_16_epi16 = _mm256_maddubs_epi16(bv0_32_epi8, av00_32_epi8);
            const __m256i dot1_16_epi16 = _mm256_maddubs_epi16(bv1_32_epi8, av01_32_epi8);
            const __m256i sum_16_epi16 = _mm256_hadd_epi16(dot0_16_epi16, dot1_16_epi16);

            // generating constant 1s is faster here.
            // __m256i one = _mm256_set1_epi16(1);
            __m256i one_16_epi16 = _mm256_srli_epi16(_mm256_cmpeq_epi16(bv0_32_epi8, bv0_32_epi8), 15);
            const __m256i sum_8_epi32 = _mm256_madd_epi16(one_16_epi16, sum_16_epi16);
            const __m256 sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);

            __m256 scale_a0_2_ps = _mm256_castpd_ps(_mm256_broadcast_sd((double*)scale_a0));
            __m256 scale_b_2_ps = _mm256_castpd_ps(_mm256_broadcast_sd((double*)scale_b));
            // 1 0 1 0 1 0 1 0 -> 1 1 0 0 1 1 0 0
            __m256 scale_8_ps = _mm256_permute_ps(_mm256_mul_ps(scale_a0_2_ps, scale_b_2_ps), _MM_SHUFFLE(1, 1, 0, 0));
            acc0 = _mm256_fmadd_ps(sum_ps, scale_8_ps, acc0);
        //}
        //{
            const __m256i dot0_16_epi16_ = _mm256_maddubs_epi16(bv0_32_epi8, av10_32_epi8);
            const __m256i dot1_16_epi16_ = _mm256_maddubs_epi16(bv1_32_epi8, av11_32_epi8);
            const __m256i sum_16_epi16_ = _mm256_hadd_epi16(dot0_16_epi16_, dot1_16_epi16_);
            const __m256i sum_8_epi32_ = _mm256_madd_epi16(one_16_epi16, sum_16_epi16_);
            const __m256 sum_ps_ = _mm256_cvtepi32_ps(sum_8_epi32_);

            __m256 scale_a1_2_ps = _mm256_castpd_ps(_mm256_broadcast_sd((double*)scale_a1));
            __m256 scale_8_ps_ = _mm256_permute_ps(_mm256_mul_ps(scale_a1_2_ps, scale_b_2_ps), _MM_SHUFFLE(1, 1, 0, 0));
            acc1 = _mm256_fmadd_ps(sum_ps_, scale_8_ps_, acc1);
        //}
#if !defined(__GNUC__) || (__GNUC__ > 10)
    }
#endif
}

template <bool vnni>
static MLAS_FORCEINLINE void
accumulate_blklen32_r1c1blk2_avx2(
    const __m256i& av00_32_epi8,
    const __m256i& av01_32_epi8,
    const std::byte* QuantBDataPtr,
    const float* scale_a0,
    const float* scale_b,
    __m256& acc0
)
{
    // | v0  v32 | v1  v33 | ... | v30 v62 | v31 v63 |
    const __m256i bv_packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr));
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    __m256i bv0_32_epi8 = _mm256_and_si256(bv_packed, low_mask);  // 0~31
    __m256i bv1_32_epi8 = _mm256_srli_epi16(_mm256_sub_epi8(bv_packed, bv0_32_epi8), 4);  // 32~63

#if !defined(__GNUC__) || (__GNUC__ > 10)
    if constexpr (vnni) {
        const __m256i dot0_8_epi32 = _mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), bv0_32_epi8, av00_32_epi8);
        const __m256i dot1_8_epi32 = _mm256_dpbusds_avx_epi32(_mm256_setzero_si256(), bv1_32_epi8, av01_32_epi8);
        const __m256i sum_8_epi32 = _mm256_hadd_epi32(dot0_8_epi32, dot1_8_epi32);  // 00110011

        const __m256 sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);

        __m256 scale_a0_2_ps = _mm256_castpd_ps(_mm256_broadcast_sd((double*)scale_a0));
        __m256 scale_b_2_ps = _mm256_castpd_ps(_mm256_broadcast_sd((double*)scale_b));
        // 1 0 1 0 1 0 1 0 -> 1 1 0 0 1 1 0 0
        __m256 scale_8_ps = _mm256_permute_ps(_mm256_mul_ps(scale_a0_2_ps, scale_b_2_ps), _MM_SHUFFLE(1, 1, 0, 0));
        acc0 = _mm256_fmadd_ps(sum_ps, scale_8_ps, acc0);
    } else {
#endif
        const __m256i dot0_16_epi16 = _mm256_maddubs_epi16(bv0_32_epi8, av00_32_epi8);
        const __m256i dot1_16_epi16 = _mm256_maddubs_epi16(bv1_32_epi8, av01_32_epi8);
        const __m256i sum_16_epi16 = _mm256_hadd_epi16(dot0_16_epi16, dot1_16_epi16);

        __m256i one_16_epi16 = _mm256_srli_epi16(_mm256_cmpeq_epi16(bv0_32_epi8, bv0_32_epi8), 15);
        const __m256i sum_8_epi32 = _mm256_madd_epi16(one_16_epi16, sum_16_epi16);
        const __m256 sum_ps = _mm256_cvtepi32_ps(sum_8_epi32);

        __m256 scale_a0_2_ps = _mm256_castpd_ps(_mm256_broadcast_sd((double*)scale_a0));
        __m256 scale_b_2_ps = _mm256_castpd_ps(_mm256_broadcast_sd((double*)scale_b));
        // 1 0 1 0 1 0 1 0 -> 1 1 0 0 1 1 0 0
        __m256 scale_8_ps = _mm256_permute_ps(_mm256_mul_ps(scale_a0_2_ps, scale_b_2_ps), _MM_SHUFFLE(1, 1, 0, 0));
        acc0 = _mm256_fmadd_ps(sum_ps, scale_8_ps, acc0);
#if !defined(__GNUC__) || (__GNUC__ > 10)
    }
#endif
}

template<bool vnni>
static MLAS_FORCEINLINE void
accumulate_blklen32_r2c1blk1_avx2(
    const __m256i& av00_32_epi8,
    const __m256i& av10_32_epi8,
    const std::byte* QuantBDataPtr,
    const float& combined_scale00,
    const float& combined_scale10,
    __m256& acc0,
    __m256& acc1
)
{
    // | v0  v16 | v1  v17 | ... | v14 v30 | v15 v31 |
    const __m128i bv_packed0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(QuantBDataPtr));
    __m256i bv_32_epi8 = _mm256_set_m128i(_mm_srli_epi16(bv_packed0, 4), bv_packed0);
    bv_32_epi8 = _mm256_and_si256(_mm256_set1_epi8(0x0F), bv_32_epi8);

#if !defined(__GNUC__) || (__GNUC__ > 10)
    if constexpr (vnni) {
        accumulate_1blk_dot_vnni(av00_32_epi8, bv_32_epi8, combined_scale00, acc0);
        accumulate_1blk_dot_vnni(av10_32_epi8, bv_32_epi8, combined_scale10, acc1);
    } else {
#endif
        __m256i one_16_epi16 = _mm256_srli_epi16(_mm256_cmpeq_epi16(bv_32_epi8, bv_32_epi8), 15);
        accumulate_1blk_dot(av00_32_epi8, bv_32_epi8, combined_scale00, one_16_epi16, acc0);
        accumulate_1blk_dot(av10_32_epi8, bv_32_epi8, combined_scale10, one_16_epi16, acc1);
#if !defined(__GNUC__) || (__GNUC__ > 10)
    }
#endif
}

template <bool vnni>
static MLAS_FORCEINLINE void
accumulate_blklen32_r1c1blk1_avx2(
    const __m256i& av00_32_epi8,
    const std::byte* QuantBDataPtr,
    const float& combined_scale00,
    __m256& acc0
)
{
    // | v0  v16 | v1  v17 | ... | v14 v30 | v15 v31 |
    const __m128i bv_packed0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(QuantBDataPtr));
    __m256i bv_32_epi8 = _mm256_set_m128i(_mm_srli_epi16(bv_packed0, 4), bv_packed0);
    bv_32_epi8 = _mm256_and_si256(_mm256_set1_epi8(0x0F), bv_32_epi8);

#if !defined(__GNUC__) || (__GNUC__ > 10)
    if constexpr (vnni) {
        accumulate_1blk_dot_vnni(av00_32_epi8, bv_32_epi8, combined_scale00, acc0);
    } else {
#endif
        __m256i one_16_epi16 = _mm256_srli_epi16(_mm256_cmpeq_epi16(bv_32_epi8, bv_32_epi8), 15);
        accumulate_1blk_dot(av00_32_epi8, bv_32_epi8, combined_scale00, one_16_epi16, acc0);
#if !defined(__GNUC__) || (__GNUC__ > 10)
    }
#endif
}

template <bool vnni>
MLAS_FORCEINLINE void
Q4Int8Gemm2x4x2BlkLen32Avx2(
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
    constexpr size_t BlkLen32 = 32;
    constexpr size_t BlkBitWidth4 = 4;
    constexpr size_t NCols4 = 4;
    constexpr size_t NRows2 = 2;
    constexpr size_t BlkDataSizeInBytes16 = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);

    // process 2 blks of 64 4b weights a time
    constexpr size_t PerAccuBlk2 = 2;

    const size_t lda = BlockCountK * BlkLen32;
    //const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);
    //const size_t StrideQuantBScale = BlockCountK;

    [[maybe_unused]] size_t QuantBZeroPointIdx = 0;  // track half byte increments with this index instead of a pointer
    assert(CountM % NRows2 == 0);
    assert(CountN % NCols4 == 0);
    const size_t StrideQuantBDataCol = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);
    const size_t StrideQuantBData2 = 2 * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);
    const size_t StrideQuantBData1 = 1 * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);
    const size_t StrideQuantBScale2 = 2;
    const size_t StrideQuantBScale1 = 1;

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

            size_t k_blks_remaining = BlockCountK;
            // process 2 blks of 64 4b weights a time
            for (; k_blks_remaining > 1; k_blks_remaining -= PerAccuBlk2) {
                // load A:
                const __m256i av_00_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr));
                const __m256i av_01_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr + BlkLen32));
                const __m256i av_10_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr + lda));
                const __m256i av_11_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr + lda + BlkLen32));

                {
                    accumulate_blklen32_r2c1blk2_avx2<vnni>(av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, QuantBDataPtr, QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr, acc[0], acc[NCols4]);
                }
                {
                    accumulate_blklen32_r2c1blk2_avx2<vnni>(av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, QuantBDataPtr + StrideQuantBData2, QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr + StrideQuantBScale2, acc[1], acc[NCols4 + 1]);
                }

                {
                    accumulate_blklen32_r2c1blk2_avx2<vnni>(av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, QuantBDataPtr + 2 * StrideQuantBData2, QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr + 2 * StrideQuantBScale2, acc[2], acc[NCols4 + 2]);
                }

                {
                    accumulate_blklen32_r2c1blk2_avx2<vnni>(av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, QuantBDataPtr + 3 * StrideQuantBData2, QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr + 3 * StrideQuantBScale2, acc[3], acc[NCols4 + 3]);
                }

                // increment block pointers
                QuantAPtr += BlkLen32 * PerAccuBlk2;
                QuantAScalePtr += PerAccuBlk2;
                QuantBDataPtr += BlkDataSizeInBytes16 * PerAccuBlk2 * NCols4;
                QuantBScalePtr += PerAccuBlk2 * NCols4;
            }  // k_blks_remaining

            // TODO: use a loop in case PerAccuBlk2 is not 2.
            if (k_blks_remaining > 0) {
                // load A
                const __m256i av_00_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr));
                const __m256i av_10_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr + lda));

                const float& scale_a00 = *QuantAScalePtr;
                const float& scale_a10 = *(QuantAScalePtr + BlockCountK);

                {
                    // Col0
                    const float scale_00 = scale_a00 * (QuantBScalePtr)[0];
                    const float scale_10 = scale_a10 * (QuantBScalePtr)[0];
                    accumulate_blklen32_r2c1blk1_avx2<vnni>(av_00_epi8, av_10_epi8, QuantBDataPtr, scale_00, scale_10, acc[0], acc[NCols4]);
                }

                {
                    // Col1
                    const float scale_00 = scale_a00 * (QuantBScalePtr + StrideQuantBScale1)[0];
                    const float scale_10 = scale_a10 * (QuantBScalePtr + StrideQuantBScale1)[0];
                    accumulate_blklen32_r2c1blk1_avx2<vnni>(av_00_epi8, av_10_epi8, QuantBDataPtr + StrideQuantBData1, scale_00, scale_10, acc[1], acc[NCols4 + 1]);
                }

                {
                    // Col2
                    const float scale_00 = scale_a00 * (QuantBScalePtr + 2 * StrideQuantBScale1)[0];
                    const float scale_10 = scale_a10 * (QuantBScalePtr + 2 * StrideQuantBScale1)[0];
                    accumulate_blklen32_r2c1blk1_avx2<vnni>(av_00_epi8, av_10_epi8, QuantBDataPtr + 2 * StrideQuantBData1, scale_00, scale_10, acc[2], acc[NCols4 + 2]);
                }

                {
                    // Col3
                    const float& scale_00 = scale_a00 * (QuantBScalePtr + 3 * StrideQuantBScale1)[0];
                    const float& scale_10 = scale_a10 * (QuantBScalePtr + 3 * StrideQuantBScale1)[0];
                    accumulate_blklen32_r2c1blk1_avx2<vnni>(av_00_epi8, av_10_epi8, QuantBDataPtr + 3 * StrideQuantBData1, scale_00, scale_10, acc[3], acc[NCols4 + 3]);
                }
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
            QuantBDataColPtr += NCols4 * StrideQuantBDataCol;
            QuantBScaleColPtr += NCols4 * BlockCountK;

            BiasPtr += BiasPtr != nullptr ? NCols4 : 0;
            SumPtr += NCols4;
        }
    }
}

template <bool vnni>
void MLAS_FORCEINLINE Q4Int8Gemm2xXBlkLen32Avx2(
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc)
{
    constexpr size_t BlkLen32 = 32;
    constexpr size_t BlkBitWidth4 = 4;
    [[maybe_unused]] constexpr size_t NCols4 = 4;
    constexpr size_t NRows2 = 2;
    constexpr size_t BlkDataSizeInBytes16 = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);

    // process 2 blks of 64 4b weights a time
    constexpr size_t PerAccuBlk2 = 2;

    const size_t lda = BlockCountK * BlkLen32;
    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);
    //const size_t StrideQuantBScale = BlockCountK;

    [[maybe_unused]] size_t QuantBZeroPointIdx = 0;  // track half byte increments with this index instead of a pointer
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

            size_t k_blks_remaining = BlockCountK;
            // process 2 blks of 64 4b weights a time
            for (; k_blks_remaining > 1; k_blks_remaining -= PerAccuBlk2) {
                const __m256i av_00_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr));
                const __m256i av_01_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr + BlkLen32));
                const __m256i av_10_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr + lda));
                const __m256i av_11_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr + lda + BlkLen32));

                accumulate_blklen32_r2c1blk2_avx2<vnni>(
                    av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, QuantBDataPtr,
                    QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr, acc0, acc1);

                // increment block pointers
                QuantAPtr += BlkLen32 * PerAccuBlk2;
                QuantAScalePtr += PerAccuBlk2;
                QuantBDataPtr += BlkDataSizeInBytes16 * PerAccuBlk2;
                QuantBScalePtr += PerAccuBlk2;
            }

            if (k_blks_remaining > 0) {
                const __m256i av_00_epi8 = _mm256_loadu_si256((const __m256i*)QuantAPtr);
                const __m256i av_10_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr + lda));

                const float& scale_a00 = *QuantAScalePtr;
                const float& scale_a10 = *(QuantAScalePtr + BlockCountK);

                const float& scale_00 = scale_a00 * (QuantBScalePtr)[0];
                const float& scale_10 = scale_a10 * (QuantBScalePtr)[0];
                accumulate_blklen32_r2c1blk1_avx2<vnni>(av_00_epi8, av_10_epi8, QuantBDataPtr, scale_00, scale_10, acc0, acc1);
            }

            *SumPtr = hsum_float_8(acc0);
            *(SumPtr + ldc) = hsum_float_8(acc1);
            if (BiasPtr) {
                *SumPtr += *BiasPtr;
                *(SumPtr + ldc) += *BiasPtr;
            }

            // move to next column
            QuantBDataColPtr += StrideQuantBData;
            QuantBScaleColPtr += BlockCountK;

            BiasPtr += BiasPtr != nullptr ? 1 : 0;
            SumPtr += 1;
        }
    }
}

template <bool vnni>
MLAS_FORCEINLINE void
Q4Int8GemmXx4BlkLen32Avx2(
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
    constexpr size_t BlkLen32 = 32;
    constexpr size_t BlkBitWidth4 = 4;
    constexpr size_t NCols4 = 4;
    [[maybe_unused]] constexpr size_t NRows2 = 2;
    constexpr size_t BlkDataSizeInBytes16 = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);

    // process 2 blks of 64 4b weights a time
    constexpr size_t PerAccuBlk2 = 2;

    const size_t lda = BlockCountK * BlkLen32;
    //const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);
    //const size_t StrideQuantBScale = BlockCountK;

    assert(CountM < NRows2);
    assert(CountN % NCols4 == 0);
    const size_t StrideQuantBDataCol = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);
    const size_t StrideQuantBData2 = 2 * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);
    const size_t StrideQuantBData1 = 1 * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);
    const size_t StrideQuantBScale2 = 2;
    const size_t StrideQuantBScale1 = 1;

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
            size_t k_blks_remaining = BlockCountK;
            for (; k_blks_remaining > 1; k_blks_remaining -= PerAccuBlk2) {
                const __m256i av_00_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr));
                const __m256i av_01_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr + BlkLen32));

                {
                    accumulate_blklen32_r1c1blk2_avx2<vnni>(av_00_epi8, av_01_epi8, QuantBDataPtr,
                      QuantAScalePtr, QuantBScalePtr, acc[0]);
                }
                {
                    accumulate_blklen32_r1c1blk2_avx2<vnni>(
                        av_00_epi8, av_01_epi8, QuantBDataPtr + StrideQuantBData2,
                        QuantAScalePtr, QuantBScalePtr + StrideQuantBScale2, acc[1]
                    );
                }
                {
                    accumulate_blklen32_r1c1blk2_avx2<vnni>(
                        av_00_epi8, av_01_epi8, QuantBDataPtr + 2 * StrideQuantBData2,
                        QuantAScalePtr, QuantBScalePtr + 2 * StrideQuantBScale2, acc[2]
                    );
                }
                {
                    accumulate_blklen32_r1c1blk2_avx2<vnni>(
                        av_00_epi8, av_01_epi8, QuantBDataPtr + 3 * StrideQuantBData2,
                        QuantAScalePtr, QuantBScalePtr + 3 * StrideQuantBScale2, acc[3]
                    );
                }
                // increment block pointers
                QuantAPtr += BlkLen32 * PerAccuBlk2;
                QuantAScalePtr += PerAccuBlk2;
                QuantBDataPtr += BlkDataSizeInBytes16 * PerAccuBlk2 * NCols4;
                QuantBScalePtr += PerAccuBlk2 * NCols4;
            }

            // TODO: use a loop in case PerAccuBlk2 is not 2.
            if (k_blks_remaining > 0) {
                // load A
                const std::byte* QuantABlk0 = QuantAPtr;
                const __m256i av_00_epi8 = _mm256_loadu_si256((const __m256i*)QuantABlk0);

                const float& scale_a00 = *QuantAScalePtr;
                {
                    // Col0
                    const float& scale_00 = scale_a00 * (QuantBScalePtr)[0];
                    accumulate_blklen32_r1c1blk1_avx2<vnni>(av_00_epi8, QuantBDataPtr, scale_00, acc[0]);
                }
                {
                    // Col1
                    const float& scale_00 = scale_a00 * (QuantBScalePtr + StrideQuantBScale1)[0];
                    accumulate_blklen32_r1c1blk1_avx2<vnni>(av_00_epi8, QuantBDataPtr + StrideQuantBData1, scale_00, acc[1]);
                }
                {
                    // Col2
                    const float& scale_00 = scale_a00 * (QuantBScalePtr + 2 * StrideQuantBScale1)[0];
                    accumulate_blklen32_r1c1blk1_avx2<vnni>(av_00_epi8, QuantBDataPtr + 2 * StrideQuantBData1, scale_00, acc[2]);
                }
                {
                    // Col3
                    const float& scale_00 = scale_a00 * (QuantBScalePtr + 3 * StrideQuantBScale1)[0];
                    accumulate_blklen32_r1c1blk1_avx2<vnni>(av_00_epi8, QuantBDataPtr + 3 * StrideQuantBData1, scale_00, acc[3]);
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
            BiasPtr += BiasPtr != nullptr ? NCols4 : 0;
            SumPtr += NCols4;
        }
    }
}

template <bool vnni>
MLAS_FORCEINLINE void
Q4Int8GemmXxXBlkLen32Avx2(
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
    constexpr size_t BlkLen32 = 32;
    constexpr size_t BlkBitWidth4 = 4;
    [[maybe_unused]] constexpr size_t NCols4 = 4;
    [[maybe_unused]] constexpr size_t NRows2 = 2;
    constexpr size_t BlkDataSizeInBytes16 = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);

    // process 2 blks of 64 4b weights a time
    constexpr size_t PerAccuBlk2 = 2;

    const size_t lda = BlockCountK * BlkLen32;
    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);
    //const size_t StrideQuantBScale = BlockCountK;

    [[maybe_unused]] size_t QuantBZeroPointIdx = 0;  // track half byte increments with this index instead of a pointer
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
            size_t k_blks_remaining = BlockCountK;
            for (; k_blks_remaining > 1; k_blks_remaining -= PerAccuBlk2) {
                const __m256i av_00_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr));
                const __m256i av_01_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr + BlkLen32));
                accumulate_blklen32_r1c1blk2_avx2<vnni>(
                    av_00_epi8, av_01_epi8, QuantBDataPtr,
                    QuantAScalePtr, QuantBScalePtr, acc0
                );

                // increment block pointers
                QuantAPtr += BlkLen32 * PerAccuBlk2;
                QuantAScalePtr += PerAccuBlk2;
                QuantBDataPtr += BlkDataSizeInBytes16 * PerAccuBlk2;
                QuantBScalePtr += PerAccuBlk2;
            }

            // TODO: use a loop in case PerAccuBlk2 is not 2.
            if (k_blks_remaining > 0) {
                const __m256i av_00_epi8 = _mm256_loadu_si256((const __m256i*)QuantAPtr);
                const float& scale_a00 = *QuantAScalePtr;
                const float& scale_00 = scale_a00 * (QuantBScalePtr)[0];
                accumulate_blklen32_r1c1blk1_avx2<vnni>(av_00_epi8, QuantBDataPtr, scale_00, acc0);
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
}

template <bool vnni>
MLAS_FORCEINLINE
    size_t
    MlasQ4Int8GemmKernelBlkLen32Avx2(
        const std::byte* QuantA,
        const float* QuantAScale,
        const std::byte* QuantBData,
        const float* QuantBScale,
        float* C,
        size_t CountM,
        size_t CountN,
        size_t /*CountK*/,
        size_t BlockCountK,
        const float* Bias,
        size_t ldc
    )
{
    constexpr size_t BlkLen32 = 32;
    constexpr size_t BlkBitWidth4 = 4;
    constexpr size_t NCols4 = 4;
    constexpr size_t NRows2 = 2;

    const size_t lda = BlockCountK * BlkLen32 * sizeof(int8_t);
    const size_t lda_scale = BlockCountK;
    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);
    const size_t StrideQuantBScale = BlockCountK;

    [[maybe_unused]] size_t QuantBZeroPointIdx = 0;  // track half byte increments with this index instead of a pointer

    size_t remainingRows = CountM % NRows2;
    size_t multipleRows = CountM - remainingRows;
    size_t remainingCols = CountN % NCols4;
    size_t multipleCols = CountN - remainingCols;

    if (multipleRows > 0 && multipleCols > 0) {
        Q4Int8Gemm2x4x2BlkLen32Avx2<vnni>(
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
        Q4Int8Gemm2xXBlkLen32Avx2<vnni>(
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
        Q4Int8GemmXx4BlkLen32Avx2<vnni>(
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
        Q4Int8GemmXxXBlkLen32Avx2<vnni>(
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

// this function is to explore larger NCols. With Avx2 it does not improve performance.
// Leave it here until the same is implemented in avx512.
template <bool HasZeroPoint, AccumulateFunctionType<HasZeroPoint> accumulator>
MLAS_FORCEINLINE
size_t
MlasQ4Int8TileGemmKernelBlkLen32Avx2(
    const std::byte* QuantA,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t /*CountK*/,
    size_t BlockCountK,
    const float* Bias,
    size_t lda,
    size_t ldc
)
{
    // We process 32 quantized values in a batch.
    constexpr size_t BlkLen32 = 32;
    constexpr size_t BlkBitWidth4 = 4;
    constexpr size_t NCols4 = 4;
    constexpr size_t BlkDataSizeInBytes16 = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);

    // process 2 blks of 64 4b weights a time
    constexpr size_t PerAccuBlk2 = 2;

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);
    const size_t StrideQuantBScale = BlockCountK;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth4>(BlockCountK);

    const __m256i zero = _mm256_setzero_si256();
    const __m128i low_mask = _mm_set1_epi8(0xF);

    [[maybe_unused]] size_t QuantBZeroPointIdx = 0;  // track half byte increments with this index instead of a pointer

    for (size_t m = 0; m < CountM; m++) {
        // for each row of A, reset B pointers
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;
        const float* BiasPtr = Bias;
        auto* SumPtr = C + m * ldc;

        int64_t nblk = (int64_t)(CountN)-NCols4;
        while (nblk >= 0) {
            const std::byte* QuantAPtr = QuantA + m * lda;

            const std::byte* QuantBDataPtr = QuantBDataColPtr;
            const float* QuantBScalePtr = QuantBScaleColPtr;
            const std::byte* QuantBZeroPointPtr = QuantBZeroPointColPtr;

            __m256 acc[NCols4];

            acc[0] = _mm256_setzero_ps();
            acc[1] = _mm256_setzero_ps();
            acc[2] = _mm256_setzero_ps();
            acc[3] = _mm256_setzero_ps();

            if constexpr (NCols4 == 8) {
                acc[4] = _mm256_setzero_ps();
                acc[5] = _mm256_setzero_ps();
                acc[6] = _mm256_setzero_ps();
                acc[7] = _mm256_setzero_ps();
            }

            size_t k_blks_remaining = BlockCountK;

            // process 2 blks of 64 4b weights a time
            for (; k_blks_remaining > 1; k_blks_remaining -= PerAccuBlk2) {
                const std::byte* QuantABlk0 = QuantAPtr;
                const std::byte* QuantABlk1 = QuantABlk0 + Q8BlkSize(BlkLen32);

                // load A:
                const __m256i av_0_epi8 = _mm256_loadu_si256((const __m256i*)Q8BlkData(QuantABlk0));
                const __m256i av_1_epi8 = _mm256_loadu_si256((const __m256i*)Q8BlkData(QuantABlk1));

                const float& scale_a0 = Q8BlkScale(QuantABlk0);
                const float& scale_a1 = Q8BlkScale(QuantABlk1);

                // Col0
                const float& scale_00 = scale_a0 * QuantBScalePtr[0];
                const float& scale_01 = scale_a1 * QuantBScalePtr[1];
                accumulator(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr), low_mask, zero, QuantBZeroPointPtr, true, scale_00, acc[0]);
                accumulator(av_1_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 16), low_mask, zero, QuantBZeroPointPtr, false, scale_01, acc[0]);

                // Col1
                const float& scale_10 = scale_a0 * (QuantBScalePtr + StrideQuantBScale)[0];
                const float& scale_11 = scale_a1 * (QuantBScalePtr + StrideQuantBScale)[1];
                accumulator(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + StrideQuantBData), low_mask, zero, QuantBZeroPointPtr + StrideQuantBZeroPoint, true, scale_10, acc[1]);
                accumulator(av_1_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + StrideQuantBData + 16), low_mask, zero, QuantBZeroPointPtr + StrideQuantBZeroPoint, false, scale_11, acc[1]);

                // Col2
                const float& scale_20 = scale_a0 * (QuantBScalePtr + 2 * StrideQuantBScale)[0];
                const float& scale_21 = scale_a1 * (QuantBScalePtr + 2 * StrideQuantBScale)[1];
                accumulator(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 2 * StrideQuantBData), low_mask, zero, QuantBZeroPointPtr + 2 * StrideQuantBZeroPoint, true, scale_20, acc[2]);
                accumulator(av_1_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 2 * StrideQuantBData + 16), low_mask, zero, QuantBZeroPointPtr + 2 * StrideQuantBZeroPoint, false, scale_21, acc[2]);

                // Col3
                const float& scale_30 = scale_a0 * (QuantBScalePtr + 3 * StrideQuantBScale)[0];
                const float& scale_31 = scale_a1 * (QuantBScalePtr + 3 * StrideQuantBScale)[1];
                accumulator(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 3 * StrideQuantBData), low_mask, zero, QuantBZeroPointPtr + 3 * StrideQuantBZeroPoint, true, scale_30, acc[3]);
                accumulator(av_1_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 3 * StrideQuantBData + 16), low_mask, zero, QuantBZeroPointPtr + 3 * StrideQuantBZeroPoint, false, scale_31, acc[3]);

                if constexpr (NCols4 == 8) {
                    // Col4
                    const float& scale_40 = scale_a0 * (QuantBScalePtr + 4 * StrideQuantBScale)[0];
                    const float& scale_41 = scale_a1 * (QuantBScalePtr + 4 * StrideQuantBScale)[1];
                    accumulator(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 4 * StrideQuantBData), low_mask, zero, QuantBZeroPointPtr, true, scale_40, acc[4]);
                    accumulator(av_1_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 4 * StrideQuantBData + 16), low_mask, zero, QuantBZeroPointPtr, false, scale_41, acc[4]);

                    // Col5
                    const float& scale_50 = scale_a0 * (QuantBScalePtr + 5 * StrideQuantBScale)[0];
                    const float& scale_51 = scale_a1 * (QuantBScalePtr + 5 * StrideQuantBScale)[1];
                    accumulator(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 5 * StrideQuantBData), low_mask, zero, QuantBZeroPointPtr + StrideQuantBZeroPoint, true, scale_50, acc[5]);
                    accumulator(av_1_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 5 * StrideQuantBData + 16), low_mask, zero, QuantBZeroPointPtr + StrideQuantBZeroPoint, false, scale_51, acc[5]);

                    // Col6
                    const float& scale_60 = scale_a0 * (QuantBScalePtr + 6 * StrideQuantBScale)[0];
                    const float& scale_61 = scale_a1 * (QuantBScalePtr + 6 * StrideQuantBScale)[1];
                    accumulator(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 6 * StrideQuantBData), low_mask, zero, QuantBZeroPointPtr + 6 * StrideQuantBZeroPoint, true, scale_60, acc[6]);
                    accumulator(av_1_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 6 * StrideQuantBData + 16), low_mask, zero, QuantBZeroPointPtr + 6 * StrideQuantBZeroPoint, false, scale_61, acc[6]);

                    // Col7
                    const float& scale_70 = scale_a0 * (QuantBScalePtr + 7 * StrideQuantBScale)[0];
                    const float& scale_71 = scale_a1 * (QuantBScalePtr + 7 * StrideQuantBScale)[1];
                    accumulator(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 7 * StrideQuantBData), low_mask, zero, QuantBZeroPointPtr + 7 * StrideQuantBZeroPoint, true, scale_70, acc[7]);
                    accumulator(av_1_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 7 * StrideQuantBData + 16), low_mask, zero, QuantBZeroPointPtr + 7 * StrideQuantBZeroPoint, false, scale_71, acc[7]);
                }

                // increment block pointers
                QuantAPtr += Q8BlkSize(BlkLen32) * PerAccuBlk2;
                QuantBDataPtr += BlkDataSizeInBytes16 * PerAccuBlk2;
                QuantBScalePtr += PerAccuBlk2;
                if constexpr (HasZeroPoint) {
                    QuantBZeroPointPtr += 1;
                }
            } // k_blks_remaining

            // TODO: use a loop in case PerAccuBlk2 is not 2.
            if (k_blks_remaining > 0) {
                // load A
                const std::byte* QuantABlk0 = QuantAPtr;
                const __m256i av_0_epi8 = _mm256_loadu_si256((const __m256i*)Q8BlkData(QuantABlk0));

                const float& scale_a0 = Q8BlkScale(QuantABlk0);

                // Col0
                const float& scale_00 = scale_a0 * QuantBScalePtr[0];
                accumulator(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr), low_mask, zero, QuantBZeroPointPtr, true, scale_00, acc[0]);

                // Col1
                const float& scale_10 = scale_a0 * (QuantBScalePtr + StrideQuantBScale)[0];
                accumulator(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + StrideQuantBData), low_mask, zero, QuantBZeroPointPtr + StrideQuantBZeroPoint, true, scale_10, acc[1]);

                // Col2
                const float& scale_20 = scale_a0 * (QuantBScalePtr + 2 * StrideQuantBScale)[0];
                accumulator(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 2 * StrideQuantBData), low_mask, zero, QuantBZeroPointPtr + 2 * StrideQuantBZeroPoint, true, scale_20, acc[2]);

                // Col3
                const float& scale_30 = scale_a0 * (QuantBScalePtr + 3 * StrideQuantBScale)[0];
                accumulator(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 3 * StrideQuantBData), low_mask, zero, QuantBZeroPointPtr + 3 * StrideQuantBZeroPoint, true, scale_30, acc[3]);

                if constexpr (NCols4 == 8) {
                    // Col4
                    const float& scale_40 = scale_a0 * (QuantBScalePtr + 4 * StrideQuantBScale)[0];
                    accumulator(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 4 * StrideQuantBData), low_mask, zero, QuantBZeroPointPtr + 4 * StrideQuantBZeroPoint, true, scale_40, acc[4]);

                    // Col5
                    const float& scale_50 = scale_a0 * (QuantBScalePtr + 5 * StrideQuantBScale)[0];
                    accumulator(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 5 * StrideQuantBData), low_mask, zero, QuantBZeroPointPtr + 5 * StrideQuantBZeroPoint, true, scale_50, acc[5]);

                    // Col6
                    const float& scale_60 = scale_a0 * (QuantBScalePtr + 6 * StrideQuantBScale)[0];
                    accumulator(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 6 * StrideQuantBData), low_mask, zero, QuantBZeroPointPtr + 6 * StrideQuantBZeroPoint, true, scale_60, acc[6]);

                    // Col7
                    const float& scale_70 = scale_a0 * (QuantBScalePtr + 7 * StrideQuantBScale)[0];
                    accumulator(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 7 * StrideQuantBData), low_mask, zero, QuantBZeroPointPtr + 7 * StrideQuantBZeroPoint, true, scale_70, acc[7]);
                }
            }  // k_blks_remaining

            if constexpr (NCols4 == 8) {
                __m128 acc_0 = FoldAccumulators(acc[0], acc[1], acc[2], acc[3]);
                __m128 acc_1 = FoldAccumulators(acc[4], acc[5], acc[6], acc[7]);
                if (BiasPtr != nullptr) {
                    acc_0 = _mm_add_ps(acc_0, _mm_loadu_ps(BiasPtr));
                    acc_1 = _mm_add_ps(acc_1, _mm_loadu_ps(BiasPtr + 4));
                }
                _mm_storeu_ps(SumPtr, acc_0);
                _mm_storeu_ps(SumPtr+4, acc_1);
            } else {
                __m128 acc_x = FoldAccumulators(acc[0], acc[1], acc[2], acc[3]);
                if (BiasPtr != nullptr) {
                    acc_x = _mm_add_ps(acc_x, _mm_loadu_ps(BiasPtr));
                }
                _mm_storeu_ps(SumPtr, acc_x);
            }

            // move to next NCols columns

            QuantBDataColPtr += NCols4 * StrideQuantBData;
            QuantBScaleColPtr += NCols4 * StrideQuantBScale;
            if constexpr (HasZeroPoint) {
                QuantBZeroPointColPtr += NCols4 * StrideQuantBZeroPoint;
            }

            BiasPtr += BiasPtr != nullptr ? NCols4 : 0;
            SumPtr += NCols4;
            nblk -= NCols4;
        } // while (nblk >= 0)

        nblk += NCols4;
        for (int64_t n = 0; n < nblk; n++) {
            const std::byte* QuantAPtr = QuantA + m * lda;
            const std::byte* QuantBDataPtr = QuantBDataColPtr;
            const float* QuantBScalePtr = QuantBScaleColPtr;
            const std::byte* QuantBZeroPointPtr = QuantBZeroPointColPtr;

            __m256 acc0 = _mm256_setzero_ps();

            size_t k_blks_remaining = BlockCountK;
            for (; k_blks_remaining > 1; k_blks_remaining -= PerAccuBlk2) {
                const std::byte* QuantABlk0 = QuantAPtr;
                const std::byte* QuantABlk1 = QuantABlk0 + Q8BlkSize(BlkLen32);

                // load A:
                const __m256i av_0_epi8 = _mm256_loadu_si256((const __m256i*)Q8BlkData(QuantABlk0));
                const __m256i av_1_epi8 = _mm256_loadu_si256((const __m256i*)Q8BlkData(QuantABlk1));

                const float& scale_a0 = Q8BlkScale(QuantABlk0);
                const float& scale_a1 = Q8BlkScale(QuantABlk1);

                // Col0
                const float& scale_00 = scale_a0 * QuantBScalePtr[0];
                const float& scale_01 = scale_a1 * QuantBScalePtr[1];
                accumulator(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr), low_mask, zero, QuantBZeroPointPtr, true, scale_00, acc0);
                accumulator(av_1_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 16), low_mask, zero, QuantBZeroPointPtr, false, scale_01, acc0);

                // increment block pointers
                QuantAPtr += Q8BlkSize(BlkLen32) * PerAccuBlk2;
                QuantBDataPtr += BlkDataSizeInBytes16 * PerAccuBlk2;
                QuantBScalePtr += PerAccuBlk2;
                if constexpr (HasZeroPoint) {
                    QuantBZeroPointPtr += 1;
                }
            }

            // TODO: use a loop in case PerAccuBlk2 is not 2.
            if (k_blks_remaining > 0) {
                // load A
                const std::byte* QuantABlk0 = QuantAPtr;
                const __m256i av_0_epi8 = _mm256_loadu_si256((const __m256i*)Q8BlkData(QuantABlk0));

                const float& scale_a0 = Q8BlkScale(QuantABlk0);

                // Col0
                const float& scale_00 = scale_a0 * QuantBScalePtr[0];
                accumulator(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr), low_mask, zero, QuantBZeroPointPtr, true, scale_00, acc0);
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
    } // m
    return CountM;
}
