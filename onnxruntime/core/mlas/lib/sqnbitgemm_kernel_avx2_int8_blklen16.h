#pragma once
#include <algorithm>
#include <cassert>
#include <utility>

#include "qnbitgemm.h"
#include "sqnbitgemm_kernel_avx_common.h"


MLAS_FORCEINLINE __m256
load_and_broadcast_4_scale_2(const float* scale)
{
    // 3 2 1 0 3 2 1 0 (7)
    __m256 scale_2_4_ps = _mm256_broadcast_ps((__m128 const*)scale);

    // 2 1 0 0 2 1 0 0 (1)
    __m256 scale_2_4_ps_shifted = _mm256_castsi256_ps(
        _mm256_bslli_epi128(_mm256_castps_si256(scale_2_4_ps), 4)
    );

    // 3 2 1 0 2 1 0 0: (3) cross lane
    __m256 scale_2_4_ps_permutted = _mm256_permute2f128_ps(
        scale_2_4_ps_shifted, scale_2_4_ps, 0b00110000
    );

    // in accumulate_r1_4blk_dot and accumulate_r2_4blk_dot
    // _mm256_hadd_epi16 inter leaved dot sum, resulting:
    // a31b31|a30b30|a11b11|a10b10|a21b21|a20b20|a01b01|a00b00
    // therefore we need weight to be:
    // 3 3 1 1 2 2 0 0 (1)
    return _mm256_permute_ps(scale_2_4_ps_permutted, 0b11110101);
}

MLAS_FORCEINLINE
__m256i
load_16_epi8_as_epi16(const std::byte* ablob)
{
    const __m128i av_epi8 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(ablob));
    __m256i av_epi16 = _mm256_cvtepi8_epi16(av_epi8);
    return av_epi16;
}

MLAS_FORCEINLINE void
accumulate_r1_4blk_dot(
  const __m256i& av0_32_epi8, const __m256i& av1_32_epi8,
  const __m256i& bv0_32_epi8, const __m256i& bv1_32_epi8,
  const float* scale_a, const float* scale_b,
  __m256& acc)
{
    const __m256i dot0_16_epi16 = _mm256_maddubs_epi16(bv0_32_epi8, av0_32_epi8);
    const __m256i dot1_16_epi16 = _mm256_maddubs_epi16(bv1_32_epi8, av1_32_epi8);
    const __m256i sum_16_inter_leaved_epi16 = _mm256_hadd_epi16(dot0_16_epi16, dot1_16_epi16);

    __m256i one_16_epi16 = _mm256_srli_epi16(_mm256_cmpeq_epi16(bv0_32_epi8, bv0_32_epi8), 15);
    const __m256i sum_8_inter_leaved_epi32 = _mm256_madd_epi16(one_16_epi16, sum_16_inter_leaved_epi16);
    const __m256 sum_8_inter_leaved_ps = _mm256_cvtepi32_ps(sum_8_inter_leaved_epi32);

    // load 4 scales
    __m256 scale_a_4_ps = load_and_broadcast_4_scale_2(scale_a);
    __m256 scale_b_4_ps = load_and_broadcast_4_scale_2(scale_b);
    __m256 scale_8_ps = _mm256_mul_ps(scale_a_4_ps, scale_b_4_ps);
    acc = _mm256_fmadd_ps(sum_8_inter_leaved_ps, scale_8_ps, acc);
}

MLAS_FORCEINLINE void
accumulate_r2_4blk_dot(
    const __m256i& av00_32_epi8, const __m256i& av01_32_epi8, const __m256i& av10_32_epi8, const __m256i& av11_32_epi8,
    const __m256i& bv0_32_epi8, const __m256i& bv1_32_epi8,
    const float* scale_a0, const float* scale_a1, const float* scale_b,
    __m256& acc0, __m256& acc1
)
{
    const __m256i dot0_16_epi16 = _mm256_maddubs_epi16(bv0_32_epi8, av00_32_epi8);
    const __m256i dot1_16_epi16 = _mm256_maddubs_epi16(bv1_32_epi8, av01_32_epi8);
    const __m256i sum_16_inter_leaved_epi16 = _mm256_hadd_epi16(dot0_16_epi16, dot1_16_epi16);

    __m256i one_16_epi16 = _mm256_srli_epi16(_mm256_cmpeq_epi16(bv0_32_epi8, bv0_32_epi8), 15);
    const __m256i sum_8_inter_leaved_epi32 = _mm256_madd_epi16(one_16_epi16, sum_16_inter_leaved_epi16);
    const __m256 sum_8_inter_leaved_ps = _mm256_cvtepi32_ps(sum_8_inter_leaved_epi32);

    // load 4 scales
    __m256 scale_a0_4_ps = load_and_broadcast_4_scale_2(scale_a0);
    __m256 scale_b_4_ps = load_and_broadcast_4_scale_2(scale_b);
    __m256 scale_8_ps = _mm256_mul_ps(scale_a0_4_ps, scale_b_4_ps);
    acc0 = _mm256_fmadd_ps(sum_8_inter_leaved_ps, scale_8_ps, acc0);

    const __m256i dot0_16_epi16_ = _mm256_maddubs_epi16(bv0_32_epi8, av10_32_epi8);
    const __m256i dot1_16_epi16_ = _mm256_maddubs_epi16(bv1_32_epi8, av11_32_epi8);
    const __m256i sum_16_inter_leaved_epi16_ = _mm256_hadd_epi16(dot0_16_epi16_, dot1_16_epi16_);
    const __m256i sum_8_inter_leaved_epi32_ = _mm256_madd_epi16(one_16_epi16, sum_16_inter_leaved_epi16_);
    const __m256 sum_inter_leaved_ps_ = _mm256_cvtepi32_ps(sum_8_inter_leaved_epi32_);

    __m256 scale_a1_4_ps = load_and_broadcast_4_scale_2(scale_a1);
    scale_8_ps = _mm256_mul_ps(scale_a1_4_ps, scale_b_4_ps);
    acc1 = _mm256_fmadd_ps(sum_inter_leaved_ps_, scale_8_ps, acc1);
}

static MLAS_FORCEINLINE __m256i
load_4b_packed_1blk_blklen16(const std::byte* QuantBDataPtr)
{
    // | 0 8 |...| 7 15 |
    const __m128i bv_packed_64 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(QuantBDataPtr));
    const __m128i low_mask = _mm_set1_epi8(0xF);
    const __m128i lower_8_epu8 = _mm_and_si128(bv_packed_64, low_mask);                                         // 0~7
    const __m128i upper_8_epu8 = _mm_bslli_si128(_mm_and_si128(_mm_srli_epi16(bv_packed_64, 4), low_mask), 8);  // 8~15
    const __m256i bv_16_epu16 = _mm256_cvtepi8_epi16(_mm_add_epi8(upper_8_epu8, lower_8_epu8));                 // 0~15
    return bv_16_epu16;
}

static MLAS_FORCEINLINE void
load_4b_packed_4blk_blklen16(const std::byte* QuantBDataPtr, __m256i& bv0_32_epi8, __m256i& bv1_32_epi8)
{
    // | 0 8 |...| 7 15 | 16 24 |...| 23 31 ||| 32 40 |...| 39 47 | 48 56 |...| 55 63 |
    const __m256i bv_packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(QuantBDataPtr));
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    // 0~7, 16~22, 32~39, 48~55
    __m256i bv0_32_epi8_ = _mm256_and_si256(bv_packed, low_mask);
    // 8~15, 24~31, 40~47, 56~63: (1)
    __m256i bv1_32_epi8_ = _mm256_srli_epi16(_mm256_sub_epi8(bv_packed, bv0_32_epi8_), 4);
    // 0~7, 32~39, 16~22, 48~55 <- cross lane (3)
    bv0_32_epi8_ = _mm256_permute4x64_epi64(bv0_32_epi8_, 0b11011000);
    // 40~47, 8~15, 56~63, 24~31 <- cross lane (3)
    bv1_32_epi8_ = _mm256_permute4x64_epi64(bv1_32_epi8_, 0b01110010);

    // 0~7, 8~15, 16~22, 24~31: (1)
    bv0_32_epi8 = _mm256_blend_epi32(bv0_32_epi8_, bv1_32_epi8_, 0b11001100);

    // 40~47, 32~39, 56~63, 48~55: (1)
    bv1_32_epi8 = _mm256_blend_epi32(bv0_32_epi8_, bv1_32_epi8_, 0b00110011);

    // 32~39, 40~47, 48~55, 56~63: (1)
    bv1_32_epi8 = _mm256_shuffle_epi32(bv1_32_epi8, 0b01001110);
}

static MLAS_FORCEINLINE void
accumulate_blklen16_r2c1blk4_avx2(
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
    __m256i bv0_32_epi8, bv1_32_epi8;
    load_4b_packed_4blk_blklen16(QuantBDataPtr, bv0_32_epi8, bv1_32_epi8);
    accumulate_r2_4blk_dot(av00_32_epi8, av01_32_epi8, av10_32_epi8, av11_32_epi8, bv0_32_epi8, bv1_32_epi8,
      scale_a0, scale_a1, scale_b, acc0, acc1);
}

static MLAS_FORCEINLINE void
accumulate_blklen16_r1c1blk4_avx2(
    const __m256i& av0_32_epi8,
    const __m256i& av1_32_epi8,
    const std::byte* QuantBDataPtr,
    const float* scale_a,
    const float* scale_b,
    __m256& acc
)
{
    __m256i bv0_32_epi8, bv1_32_epi8;
    load_4b_packed_4blk_blklen16(QuantBDataPtr, bv0_32_epi8, bv1_32_epi8);
    accumulate_r1_4blk_dot(av0_32_epi8, av1_32_epi8, bv0_32_epi8, bv1_32_epi8, scale_a, scale_b, acc);
}

static MLAS_FORCEINLINE void
accumulate_blklen16_r2c1blk1_avx2(
    const __m256i& av0_32_epi8,
    const __m256i& av1_32_epi8,
    const std::byte* QuantBDataPtr,
    const float& combined_scale0,
    const float& combined_scale1,
    __m256& acc0,
    __m256& acc1
)
{
    const __m256i bv_16_epu16 = load_4b_packed_1blk_blklen16(QuantBDataPtr);

    __m256i prod_8_epi32 = _mm256_madd_epi16(bv_16_epu16, av0_32_epi8);
    __m256 prod_8_ps = _mm256_cvtepi32_ps(prod_8_epi32);
    acc0 = _mm256_fmadd_ps(_mm256_set1_ps(combined_scale0), prod_8_ps, acc0);

    prod_8_epi32 = _mm256_madd_epi16(bv_16_epu16, av1_32_epi8);
    prod_8_ps = _mm256_cvtepi32_ps(prod_8_epi32);
    acc1 = _mm256_fmadd_ps(_mm256_set1_ps(combined_scale1), prod_8_ps, acc1);
}

static MLAS_FORCEINLINE void
accumulate_blklen16_r1c1blk1_avx2(
    const __m256i& av_16_epi8,
    const std::byte* QuantBDataPtr,
    const float& combined_scale,
    __m256& acc
)
{
    // | v0  v16 | v1  v17 | ... | v14 v30 | v15 v31 |
    const __m256i bv_16_epu16 = load_4b_packed_1blk_blklen16(QuantBDataPtr);

    __m256i prod_8_epi32 = _mm256_madd_epi16(bv_16_epu16, av_16_epi8);
    __m256 prod_8_ps = _mm256_cvtepi32_ps(prod_8_epi32);
    acc = _mm256_fmadd_ps(_mm256_set1_ps(combined_scale), prod_8_ps, acc);
}

MLAS_FORCEINLINE void
Q4Int8GemmR2xC4BlkLen16Avx2(
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
    constexpr size_t BlkLen16 = 16;
    constexpr size_t BlkBitWidth4 = 4;
    constexpr size_t NCols4 = 4;
    constexpr size_t NRows2 = 2;
    constexpr size_t BlkDataSizeInBytes8 = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen16);

    // process 2 blks of 64 4b weights a time
    constexpr size_t PerAccuBlk4 = 4;

    const size_t lda = BlockCountK * BlkLen16;
    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen16);
    const size_t StrideQuantBScale = BlockCountK;

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

            // process 4 blks of 64 4b weights a time
            size_t k_blks_remaining = BlockCountK;
            for (; k_blks_remaining > 3; k_blks_remaining -= PerAccuBlk4) {
                const __m256i av_00_epi8 = _mm256_loadu_si256((const __m256i*)QuantAPtr);
                const __m256i av_01_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr + 32));
                const __m256i av_10_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr + lda));
                const __m256i av_11_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr + lda + 32));

                accumulate_blklen16_r2c1blk4_avx2(av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, QuantBDataPtr,
                  QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr, acc[0], acc[NCols4]);
                accumulate_blklen16_r2c1blk4_avx2(av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, QuantBDataPtr + StrideQuantBData,
                  QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr + StrideQuantBScale, acc[1], acc[NCols4 + 1]);
                accumulate_blklen16_r2c1blk4_avx2(av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, QuantBDataPtr + 2 * StrideQuantBData,
                  QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr + 2 * StrideQuantBScale, acc[2], acc[NCols4 + 2]);
                accumulate_blklen16_r2c1blk4_avx2(av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, QuantBDataPtr + 3 * StrideQuantBData,
                  QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr + 3 * StrideQuantBScale, acc[3], acc[NCols4 + 3]);

                QuantAPtr += BlkLen16 * PerAccuBlk4;
                QuantAScalePtr += PerAccuBlk4;
                QuantBDataPtr += BlkDataSizeInBytes8 * PerAccuBlk4;
                QuantBScalePtr += PerAccuBlk4;
            }

            while (k_blks_remaining-- > 0) {
                const std::byte* QuantABlk0 = QuantAPtr;
                const __m256i av0_16_epi16 = load_16_epi8_as_epi16(QuantABlk0);
                const __m256i av1_16_epi16 = load_16_epi8_as_epi16(QuantABlk0 + lda);

                const float& scale_a00 = *QuantAScalePtr;
                const float& scale_a10 = *(QuantAScalePtr + BlockCountK);

                {
                    const float scale_00 = scale_a00 * (QuantBScalePtr)[0];
                    const float scale_10 = scale_a10 * (QuantBScalePtr)[0];
                    accumulate_blklen16_r2c1blk1_avx2(av0_16_epi16, av1_16_epi16, QuantBDataPtr, scale_00, scale_10, acc[0], acc[NCols4]);
                }

                {
                    const float scale_00 = scale_a00 * (QuantBScalePtr + StrideQuantBScale)[0];
                    const float scale_10 = scale_a10 * (QuantBScalePtr + StrideQuantBScale)[0];
                    accumulate_blklen16_r2c1blk1_avx2(av0_16_epi16, av1_16_epi16, QuantBDataPtr + StrideQuantBData, scale_00, scale_10, acc[1], acc[NCols4 + 1]);
                }

                {
                    const float scale_00 = scale_a00 * (QuantBScalePtr + 2 * StrideQuantBScale)[0];
                    const float scale_10 = scale_a10 * (QuantBScalePtr + 2 * StrideQuantBScale)[0];
                    accumulate_blklen16_r2c1blk1_avx2(av0_16_epi16, av1_16_epi16, QuantBDataPtr + 2 * StrideQuantBData, scale_00, scale_10, acc[2], acc[NCols4 + 2]);
                }

                {
                    const float& scale_00 = scale_a00 * (QuantBScalePtr + 3 * StrideQuantBScale)[0];
                    const float& scale_10 = scale_a10 * (QuantBScalePtr + 3 * StrideQuantBScale)[0];
                    accumulate_blklen16_r2c1blk1_avx2(av0_16_epi16, av1_16_epi16, QuantBDataPtr + 3 * StrideQuantBData, scale_00, scale_10, acc[3], acc[NCols4 + 3]);
                }
                QuantAPtr += BlkLen16;
                QuantAScalePtr++;
                QuantBDataPtr += BlkDataSizeInBytes8;
                QuantBScalePtr++;
            }

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
            QuantBScaleColPtr += NCols4 * StrideQuantBScale;

            BiasPtr += BiasPtr != nullptr ? NCols4 : 0;
            SumPtr += NCols4;
        }
    }
}

void MLAS_FORCEINLINE Q4Int8GemmR2xC1BlkLen16Avx2(
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
    constexpr size_t BlkLen16 = 16;
    constexpr size_t BlkBitWidth4 = 4;
    [[maybe_unused]] constexpr size_t NCols4 = 4;
    constexpr size_t NRows2 = 2;
    constexpr size_t BlkDataSizeInBytes8 = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen16);

    // process 4 blks of 64 4b weights a time
    constexpr size_t PerAccuBlk4 = 4;

    const size_t lda = BlockCountK * BlkLen16;
    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen16);
    const size_t StrideQuantBScale = BlockCountK;

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

            // process 4 blks of 64 4b weights a time
            size_t k_blks_remaining = BlockCountK;
            for (; k_blks_remaining >= PerAccuBlk4; k_blks_remaining -= PerAccuBlk4) {
                const std::byte* QuantABlk00 = QuantAPtr;
                const std::byte* QuantABlk01 = QuantABlk00 + 32;
                const std::byte* QuantABlk10 = QuantAPtr + lda;
                const std::byte* QuantABlk11 = QuantABlk10 + 32;

                // load A:
                const __m256i av_00_epi8 = _mm256_loadu_si256((const __m256i*)QuantABlk00);
                const __m256i av_01_epi8 = _mm256_loadu_si256((const __m256i*)QuantABlk01);
                const __m256i av_10_epi8 = _mm256_loadu_si256((const __m256i*)QuantABlk10);
                const __m256i av_11_epi8 = _mm256_loadu_si256((const __m256i*)QuantABlk11);

                accumulate_blklen16_r2c1blk4_avx2(
                    av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, QuantBDataPtr,
                    QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr, acc0, acc1);

                // increment block pointers
                QuantAPtr += BlkLen16 * PerAccuBlk4;
                QuantAScalePtr += PerAccuBlk4;
                QuantBDataPtr += BlkDataSizeInBytes8 * PerAccuBlk4;
                QuantBScalePtr += PerAccuBlk4;
            }

            while (k_blks_remaining-- > 0) {
                // load A
                const std::byte* QuantABlk0 = QuantAPtr;
                const __m256i av0_16_epi16 = load_16_epi8_as_epi16(QuantABlk0);
                const __m256i av1_16_epi16 = load_16_epi8_as_epi16(QuantABlk0 + lda);

                const float& scale_a00 = *QuantAScalePtr;
                const float& scale_a10 = *(QuantAScalePtr + BlockCountK);

                const float& scale_00 = scale_a00 * (QuantBScalePtr)[0];
                const float& scale_10 = scale_a10 * (QuantBScalePtr)[0];
                accumulate_blklen16_r2c1blk1_avx2(av0_16_epi16, av1_16_epi16, QuantBDataPtr, scale_00, scale_10, acc0, acc1);

                QuantAPtr += BlkLen16;
                QuantAScalePtr++;
                QuantBDataPtr += BlkDataSizeInBytes8;
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

MLAS_FORCEINLINE void
Q4Int8GemmR1xC4BlkLen16Avx2(
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
    constexpr size_t BlkLen16 = 16;
    constexpr size_t BlkBitWidth4 = 4;
    constexpr size_t NCols4 = 4;
    [[maybe_unused]] constexpr size_t NRows2 = 2;
    constexpr size_t BlkDataSizeInBytes8 = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen16);

    // process 2 blks of 64 4b weights a time
    constexpr size_t PerAccuBlk4 = 4;

    const size_t lda = BlockCountK * BlkLen16;
    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen16);
    const size_t StrideQuantBScale = BlockCountK;

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

            size_t k_blks_remaining = BlockCountK;
            for (; k_blks_remaining >= PerAccuBlk4; k_blks_remaining -= PerAccuBlk4) {
                const __m256i av_00_epi8 = _mm256_loadu_si256((const __m256i*)QuantAPtr);
                const __m256i av_01_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr + 32));

                accumulate_blklen16_r1c1blk4_avx2(av_00_epi8, av_01_epi8, QuantBDataPtr,
                    QuantAScalePtr, QuantBScalePtr, acc[0]);
                accumulate_blklen16_r1c1blk4_avx2(av_00_epi8, av_01_epi8, QuantBDataPtr + StrideQuantBData,
                    QuantAScalePtr, QuantBScalePtr + StrideQuantBScale, acc[1]);
                accumulate_blklen16_r1c1blk4_avx2(av_00_epi8, av_01_epi8, QuantBDataPtr + 2 * StrideQuantBData,
                    QuantAScalePtr, QuantBScalePtr + 2 * StrideQuantBScale, acc[2]);
                accumulate_blklen16_r1c1blk4_avx2(av_00_epi8, av_01_epi8, QuantBDataPtr + 3 * StrideQuantBData,
                    QuantAScalePtr, QuantBScalePtr + 3 * StrideQuantBScale, acc[3]);
                // increment block pointers
                QuantAPtr += BlkLen16 * PerAccuBlk4;
                QuantAScalePtr += PerAccuBlk4;
                QuantBDataPtr += BlkDataSizeInBytes8 * PerAccuBlk4;
                QuantBScalePtr += PerAccuBlk4;
            }

            while (k_blks_remaining-- > 0) {
                const std::byte* QuantABlk0 = QuantAPtr;
                const __m256i av_00_epi8 = load_16_epi8_as_epi16(QuantABlk0);

                const float& scale_a00 = *QuantAScalePtr;
                {
                  // Col0
                    const float& scale_00 = scale_a00 * (QuantBScalePtr)[0];
                  accumulate_blklen16_r1c1blk1_avx2(av_00_epi8, QuantBDataPtr, scale_00, acc[0]);
                }
                {
                    // Col1
                    const float& scale_00 = scale_a00 * (QuantBScalePtr + StrideQuantBScale)[0];
                    accumulate_blklen16_r1c1blk1_avx2(av_00_epi8, QuantBDataPtr + StrideQuantBData, scale_00, acc[1]);
                }
                {
                    // Col2
                    const float& scale_00 = scale_a00 * (QuantBScalePtr + 2 * StrideQuantBScale)[0];
                    accumulate_blklen16_r1c1blk1_avx2(av_00_epi8, QuantBDataPtr + 2 * StrideQuantBData, scale_00, acc[2]);
                }
                {
                    // Col3
                    const float& scale_00 = scale_a00 * (QuantBScalePtr + 3 * StrideQuantBScale)[0];
                    accumulate_blklen16_r1c1blk1_avx2(av_00_epi8, QuantBDataPtr + 3 * StrideQuantBData, scale_00, acc[3]);
                }
                QuantAPtr += BlkLen16;
                QuantAScalePtr++;
                QuantBDataPtr += BlkDataSizeInBytes8;
                QuantBScalePtr++;
            }

            __m128 acc_r0 = FoldAccumulators(acc[0], acc[1], acc[2], acc[3]);
            if (BiasPtr != nullptr) {
                acc_r0 = _mm_add_ps(acc_r0, _mm_loadu_ps(BiasPtr));
            }

            _mm_storeu_ps(SumPtr, acc_r0);

            // move to next NCols columns
            QuantBDataColPtr += NCols4 * StrideQuantBData;
            QuantBScaleColPtr += NCols4 * StrideQuantBScale;
            BiasPtr += BiasPtr != nullptr ? NCols4 : 0;
            SumPtr += NCols4;
        }
    }
}

MLAS_FORCEINLINE void
Q4Int8GemmR1xC1BlkLen16Avx2(
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
    constexpr size_t BlkLen16 = 16;
    constexpr size_t BlkBitWidth4 = 4;
    [[maybe_unused]] constexpr size_t NCols4 = 4;
    [[maybe_unused]] constexpr size_t NRows2 = 2;
    constexpr size_t BlkDataSizeInBytes8 = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen16);

    // process 4 blks of 64 4b weights a time
    constexpr size_t PerAccuBlk4 = 4;

    const size_t lda = BlockCountK * BlkLen16;
    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen16);
    const size_t StrideQuantBScale = BlockCountK;

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
            for (; k_blks_remaining >= PerAccuBlk4; k_blks_remaining -= PerAccuBlk4) {
                const __m256i av_00_epi8 = _mm256_loadu_si256((const __m256i*)QuantAPtr);
                const __m256i av_01_epi8 = _mm256_loadu_si256((const __m256i*)(QuantAPtr + 32));

                accumulate_blklen16_r1c1blk4_avx2(
                    av_00_epi8, av_01_epi8, QuantBDataPtr,
                    QuantAScalePtr, QuantBScalePtr, acc0);

                // increment block pointers
                QuantAPtr += BlkLen16 * PerAccuBlk4;
                QuantAScalePtr += PerAccuBlk4;
                QuantBDataPtr += BlkDataSizeInBytes8 * PerAccuBlk4;
                QuantBScalePtr += PerAccuBlk4;
            }

            while (k_blks_remaining-- > 0) {
                const __m256i av_16_epi16 = load_16_epi8_as_epi16(QuantAPtr);
                const float& scale_a00 = *QuantAScalePtr;
                const float& scale_00 = scale_a00 * (QuantBScalePtr)[0];
                accumulate_blklen16_r1c1blk1_avx2(av_16_epi16, QuantBDataPtr, scale_00, acc0);

                QuantAPtr += BlkLen16;
                QuantAScalePtr++;
                QuantBDataPtr += BlkDataSizeInBytes8;
                QuantBScalePtr++;
            }

             *SumPtr = hsum_float_8(acc0);
            if (BiasPtr) {
                *SumPtr += *BiasPtr;
            }

            QuantBDataColPtr += StrideQuantBData;
            QuantBScaleColPtr += StrideQuantBScale;
            BiasPtr += BiasPtr != nullptr ? 1 : 0;
            SumPtr += 1;
        }
    }
}

MLAS_FORCEINLINE
    size_t
    MlasQ4Int8GemmKernelBlkLen16Avx2(
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
    constexpr size_t BlkLen16 = 16;
    constexpr size_t BlkBitWidth4 = 4;
    constexpr size_t NCols4 = 4;
    constexpr size_t NRows2 = 2;

    const size_t lda = BlockCountK * BlkLen16 * sizeof(int8_t);
    const size_t lda_scale = BlockCountK;
    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen16);
    const size_t StrideQuantBScale = BlockCountK;

    [[maybe_unused]] size_t QuantBZeroPointIdx = 0;  // track half byte increments with this index instead of a pointer

    size_t remainingRows = CountM % NRows2;
    size_t multipleRows = CountM - remainingRows;
    size_t remainingCols = CountN % NCols4;
    size_t multipleCols = CountN - remainingCols;

    if (multipleRows > 0 && multipleCols > 0) {
        Q4Int8GemmR2xC4BlkLen16Avx2(
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
        Q4Int8GemmR2xC1BlkLen16Avx2(
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
        Q4Int8GemmR1xC4BlkLen16Avx2(
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
        Q4Int8GemmR1xC1BlkLen16Avx2(
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
