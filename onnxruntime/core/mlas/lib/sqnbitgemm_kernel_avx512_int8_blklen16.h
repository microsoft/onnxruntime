#pragma once
#include <algorithm>
#include <cassert>
#include <utility>

#include "qnbitgemm.h"
#include "sqnbitgemm_kernel_avx_common.h"
#include "sqnbitgemm_kernel_avx2_int8_blklen16.h"
#include "sqnbitgemm_kernel_avx512_int8_blklen32.h"
#include "sqnbitgemm_kernel_avx512_int8_blklen64.h"



static MLAS_FORCEINLINE void
load_4blk_4b_packed_blklen16(const std::byte* QuantBDataPtr, __m512i& bv0_64_epi8, __m512i& bv1_64_epi8)
{
    // | v0 v64 | v1 v65 | ... | v62 v126 | v63 v127 |
    const __m512i bv_packed = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(QuantBDataPtr));
    const __m512i low_mask = _mm512_set1_epi8(0x0F);
    bv0_64_epi8 = _mm512_and_si512(bv_packed, low_mask);                          // 0~63
    bv1_64_epi8 = _mm512_srli_epi16(_mm512_sub_epi8(bv_packed, bv0_64_epi8), 4);  // 64~127
}

static MLAS_FORCEINLINE void
accumulate_blklen16_r1c1blk8_avx512(
  const __m512i& av0_64_epi8,
  const __m512i& av1_64_epi8,
  const std::byte* QuantBDataPtr,
  const float* scale_a,
  const float* scale_b,
  __m512& acc0)
{
    __m512i bv0_64_epi8, bv1_64_epi8;
    load_4blk_4b_packed_blklen16(QuantBDataPtr, bv0_64_epi8, bv1_64_epi8);

    const __m256 scale_b_ps = _mm256_loadu_ps(scale_b);  // 01234567
    {
        const __m256 scale_a0_ps = _mm256_loadu_ps(scale_a);  // 01234567
        const __m256 scale_a0b_ps = _mm256_mul_ps(scale_b_ps, scale_a0_ps);
        __m512 scale_a0b_16_ps = _mm512_castsi512_ps(
            _mm512_broadcast_i64x4(_mm256_castps_si256(scale_a0b_ps))
        ); // 0123456701234567

        __m512i idx = _mm512_set_epi32(7, 7, 3, 3, 6, 6, 2, 2, 5, 5, 1, 1, 4, 4, 0, 0);
        scale_a0b_16_ps = _mm512_permutexvar_ps(idx, scale_a0b_16_ps);  // 0044115522663377

        const __m512i dot0_32_epi16 = _mm512_maddubs_epi16(bv0_64_epi8, av0_64_epi8);  // 0~0,1~1,2~2,3~3
        const __m512i dot1_32_epi16 = _mm512_maddubs_epi16(bv1_64_epi8, av1_64_epi8);  // 4~4,5~5,6~6,7~7

        const __m512i t1 = _mm512_unpacklo_epi64(dot0_32_epi16, dot1_32_epi16);  // 00004444111155552222666633337777
        const __m512i t2 = _mm512_unpackhi_epi64(dot0_32_epi16, dot1_32_epi16);  // 00004444111155552222666633337777
        const __m512i sum_32_epi16 = _mm512_add_epi16(t1, t2);
        const __m512i one_32_epi16 = generate_ones_32_epi16();
        const __m512i sum_16_epi32 = _mm512_madd_epi16(one_32_epi16, sum_32_epi16);  // 0044115522663377
        const __m512 sum_16_ps = _mm512_cvtepi32_ps(sum_16_epi32);
        acc0 = _mm512_fmadd_ps(sum_16_ps, scale_a0b_16_ps, acc0);
    }
}

static MLAS_FORCEINLINE void
accumulate_blklen16_r2c1blk4_avx512(
    const __m512i& av00_64_epi8,
    const __m512i& av01_64_epi8,
    const __m512i& av10_64_epi8,
    const __m512i& av11_64_epi8,
    const std::byte* QuantBDataPtr,
    const float* scale_a0,
    const float* scale_a1,
    const float* scale_b,
    __m512& acc0,
    __m512& acc1
)
{
    __m512i bv0_64_epi8, bv1_64_epi8;
    load_2blk_4b_packed_blklen64(QuantBDataPtr, bv0_64_epi8, bv1_64_epi8);

    const __m256 scale_b_ps = _mm256_loadu_ps(scale_b);  // 01234567
    {
        const __m256 scale_a0_ps = _mm256_loadu_ps(scale_a0);  // 01234567
        const __m256 scale_a0b_ps = _mm256_mul_ps(scale_b_ps, scale_a0_ps);
        __m512 scale_a0b_16_ps = _mm512_castsi512_ps(
            _mm512_broadcast_i64x4(_mm256_castps_si256(scale_a0b_ps))
        );  // 0123456701234567

        // TODO: load from memory
        __m512i idx = _mm512_set_epi32(7, 7, 3, 3, 6, 6, 2, 2, 5, 5, 1, 1, 4, 4, 0, 0);
        scale_a0b_16_ps = _mm512_permutexvar_ps(idx, scale_a0b_16_ps);

        const __m512i dot0_32_epi16 = _mm512_maddubs_epi16(bv0_64_epi8, av00_64_epi8);
        const __m512i dot1_32_epi16 = _mm512_maddubs_epi16(bv1_64_epi8, av01_64_epi8);

        const __m512i t1 = _mm512_unpacklo_epi64(dot0_32_epi16, dot1_32_epi16);
        const __m512i t2 = _mm512_unpackhi_epi64(dot0_32_epi16, dot1_32_epi16);
        const __m512i sum_32_epi16 = _mm512_add_epi16(t1, t2);
        const __m512i one_32_epi16 = generate_ones_32_epi16();
        const __m512i sum_16_epi32 = _mm512_madd_epi16(one_32_epi16, sum_32_epi16);
        const __m512 sum_16_ps = _mm512_cvtepi32_ps(sum_16_epi32);
        acc0 = _mm512_fmadd_ps(sum_16_ps, scale_a0b_16_ps, acc0);
    }
    {
        const __m256 scale_a1_ps = _mm256_loadu_ps(scale_a1);  // 01234567
        const __m256 scale_a1b_ps = _mm256_mul_ps(scale_b_ps, scale_a1_ps);
        __m512 scale_a1b_16_ps = _mm512_castsi512_ps(
            _mm512_broadcast_i64x4(_mm256_castps_si256(scale_a1b_ps))
        );  // 0123456701234567

        __m512i idx = _mm512_set_epi32(7, 7, 3, 3, 6, 6, 2, 2, 5, 5, 1, 1, 4, 4, 0, 0);
        scale_a1b_16_ps = _mm512_permutexvar_ps(idx, scale_a1b_16_ps);

        const __m512i dot0_32_epi16 = _mm512_maddubs_epi16(bv0_64_epi8, av10_64_epi8);
        const __m512i dot1_32_epi16 = _mm512_maddubs_epi16(bv1_64_epi8, av11_64_epi8);

        const __m512i t1 = _mm512_unpacklo_epi64(dot0_32_epi16, dot1_32_epi16);
        const __m512i t2 = _mm512_unpackhi_epi64(dot0_32_epi16, dot1_32_epi16);
        const __m512i sum_32_epi16 = _mm512_add_epi16(t1, t2);
        const __m512i one_32_epi16 = generate_ones_32_epi16();
        const __m512i sum_16_epi32 = _mm512_madd_epi16(one_32_epi16, sum_32_epi16);
        const __m512 sum_16_ps = _mm512_cvtepi32_ps(sum_16_epi32);
        acc1 = _mm512_fmadd_ps(sum_16_ps, scale_a1b_16_ps, acc1);
    }
}

static MLAS_FORCEINLINE void
accumulate_blklen16_r1c1blk8_avx512vnni(
    const __m512i& av0_64_epi8,
    const __m512i& av1_64_epi8,
    const std::byte* QuantBDataPtr,
    const float* scale_a,
    const float* scale_b,
    __m512& acc0
)
{
    __m512i bv0_64_epi8, bv1_64_epi8;
    load_4blk_4b_packed_blklen16(QuantBDataPtr, bv0_64_epi8, bv1_64_epi8);

    const __m256 scale_b_ps = _mm256_loadu_ps(scale_b);  // 01234567
    {
        const __m256 scale_a0_ps = _mm256_loadu_ps(scale_a);  // 01234567
        const __m256 scale_a0b_ps = _mm256_mul_ps(scale_b_ps, scale_a0_ps);
        __m512 scale_a0b_16_ps = _mm512_castsi512_ps(
            _mm512_broadcast_i64x4(_mm256_castps_si256(scale_a0b_ps))
        );  // 0123456701234567

        __m512i idx = _mm512_set_epi32(7, 7, 3, 3, 6, 6, 2, 2, 5, 5, 1, 1, 4, 4, 0, 0);
        scale_a0b_16_ps = _mm512_permutexvar_ps(idx, scale_a0b_16_ps);  // 0044115522663377

        const __m512i dot0_16_epi32 = _mm512_dpbusd_epi32(_mm512_setzero_epi32(), bv0_64_epi8, av0_64_epi8);  // 0000111122223333
        const __m512i dot1_16_epi32 = _mm512_dpbusd_epi32(_mm512_setzero_epi32(), bv1_64_epi8, av1_64_epi8);  // 4444555566667777

        const __m512i t1_16_epi32 = _mm512_unpacklo_epi64(dot0_16_epi32, dot1_16_epi32);  // 0044115522663377
        const __m512i t2_16_epi32 = _mm512_unpackhi_epi64(dot0_16_epi32, dot1_16_epi32);  // 0044115522663377
        const __m512i sum_16_epi32 = _mm512_add_epi32(t1_16_epi32, t2_16_epi32);
        const __m512 sum_16_ps = _mm512_cvtepi32_ps(sum_16_epi32);
        acc0 = _mm512_fmadd_ps(sum_16_ps, scale_a0b_16_ps, acc0);
    }
}

static MLAS_FORCEINLINE void
accumulate_blklen16_r2c1blk4_avx512vnni(
    const __m512i& av00_64_epi8,
    const __m512i& av01_64_epi8,
    const __m512i& av10_64_epi8,
    const __m512i& av11_64_epi8,
    const std::byte* QuantBDataPtr,
    const float* scale_a0,
    const float* scale_a1,
    const float* scale_b,
    __m512& acc0,
    __m512& acc1
)
{
    __m512i bv0_64_epi8, bv1_64_epi8;
    load_2blk_4b_packed_blklen64(QuantBDataPtr, bv0_64_epi8, bv1_64_epi8);

    const __m256 scale_b_ps = _mm256_loadu_ps(scale_b);  // 01234567
    {
        const __m256 scale_a0_ps = _mm256_loadu_ps(scale_a0);  // 01234567
        const __m256 scale_a0b_ps = _mm256_mul_ps(scale_b_ps, scale_a0_ps);
        __m512 scale_a0b_16_ps = _mm512_castsi512_ps(
            _mm512_broadcast_i64x4(_mm256_castps_si256(scale_a0b_ps))
        );  // 0123456701234567

        // TODO: load from memory
        __m512i idx = _mm512_set_epi32(7, 7, 3, 3, 6, 6, 2, 2, 5, 5, 1, 1, 4, 4, 0, 0);
        scale_a0b_16_ps = _mm512_permutexvar_ps(idx, scale_a0b_16_ps);

        const __m512i dot0_16_epi32 = _mm512_dpbusd_epi32(_mm512_setzero_epi32(), bv0_64_epi8, av00_64_epi8);  // 0000111122223333
        const __m512i dot1_16_epi32 = _mm512_dpbusd_epi32(_mm512_setzero_epi32(), bv1_64_epi8, av01_64_epi8);  // 4444555566667777

        const __m512i t1_16_epi32 = _mm512_unpacklo_epi64(dot0_16_epi32, dot1_16_epi32);
        const __m512i t2_16_epi32 = _mm512_unpackhi_epi64(dot0_16_epi32, dot1_16_epi32);
        const __m512i sum_16_epi32 = _mm512_add_epi32(t1_16_epi32, t2_16_epi32);
        const __m512 sum_16_ps = _mm512_cvtepi32_ps(sum_16_epi32);
        acc0 = _mm512_fmadd_ps(sum_16_ps, scale_a0b_16_ps, acc0);
    }
    {
        const __m256 scale_a1_ps = _mm256_loadu_ps(scale_a1);  // 01234567
        const __m256 scale_a1b_ps = _mm256_mul_ps(scale_b_ps, scale_a1_ps);
        __m512 scale_a1b_16_ps = _mm512_castsi512_ps(
            _mm512_broadcast_i64x4(_mm256_castps_si256(scale_a1b_ps))
        );  // 0123456701234567

        __m512i idx = _mm512_set_epi32(7, 7, 3, 3, 6, 6, 2, 2, 5, 5, 1, 1, 4, 4, 0, 0);
        scale_a1b_16_ps = _mm512_permutexvar_ps(idx, scale_a1b_16_ps);

        const __m512i dot0_16_epi32 = _mm512_dpbusd_epi32(_mm512_setzero_epi32(), bv0_64_epi8, av10_64_epi8);  // 0000111122223333
        const __m512i dot1_16_epi32 = _mm512_dpbusd_epi32(_mm512_setzero_epi32(), bv1_64_epi8, av11_64_epi8);  // 4444555566667777

        const __m512i t1_16_epi32 = _mm512_unpacklo_epi64(dot0_16_epi32, dot1_16_epi32);
        const __m512i t2_16_epi32 = _mm512_unpackhi_epi64(dot0_16_epi32, dot1_16_epi32);
        const __m512i sum_16_epi32 = _mm512_add_epi32(t1_16_epi32, t2_16_epi32);
        const __m512 sum_16_ps = _mm512_cvtepi32_ps(sum_16_epi32);
        acc1 = _mm512_fmadd_ps(sum_16_ps, scale_a1b_16_ps, acc1);
    }
}

template <bool vnni>
MLAS_FORCEINLINE void
Q4Int8GemmR2xC4BlkLen16Avx512(
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
    constexpr size_t BlkDataSizeInBytes = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen16);

    // process 2 blks of 64 4b weights a time
    constexpr size_t PerAccuBlk8 = 8;

    const size_t lda = BlockCountK * BlkLen16;
    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen16);
    const size_t StrideQuantBScale = BlockCountK;

    [[maybe_unused]] size_t QuantBZeroPointIdx = 0;  // track half byte increments with this index instead of a pointer
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

            __m512 acc[NCols4 * NRows2] = {
                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()
            };

            size_t k_blks_remaining = BlockCountK;
            // process 2 blks of 64 4b weights a time
            for (; k_blks_remaining >= PerAccuBlk8; k_blks_remaining -= PerAccuBlk8) {
                const __m512i av_00_epi8 = _mm512_loadu_si512((const __m512i*)QuantAPtr);
                const __m512i av_01_epi8 = _mm512_loadu_si512((const __m512i*)(QuantAPtr + 64));
                const __m512i av_10_epi8 = _mm512_loadu_si512((const __m512i*)(QuantAPtr + lda));
                const __m512i av_11_epi8 = _mm512_loadu_si512((const __m512i*)(QuantAPtr + lda + 64));

                if constexpr (vnni) {
                    accumulate_blklen16_r2c1blk4_avx512vnni(
                        av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8,
                        QuantBDataPtr, QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr,
                        acc[0], acc[NCols4]
                    );
                    accumulate_blklen16_r2c1blk4_avx512vnni(
                        av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, QuantBDataPtr + StrideQuantBData,
                        QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr + StrideQuantBScale,
                        acc[1], acc[NCols4 + 1]
                    );
                    accumulate_blklen16_r2c1blk4_avx512vnni(
                        av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8,
                        QuantBDataPtr + 2 * StrideQuantBData, QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr + 2 * StrideQuantBScale,
                        acc[2], acc[NCols4 + 2]
                    );
                    accumulate_blklen16_r2c1blk4_avx512vnni(
                        av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8,
                        QuantBDataPtr + 3 * StrideQuantBData, QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr + 3 * StrideQuantBScale,
                        acc[3], acc[NCols4 + 3]
                    );
                } else {
                    accumulate_blklen16_r2c1blk4_avx512(
                        av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8,
                        QuantBDataPtr, QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr,
                        acc[0], acc[NCols4]
                    );
                    accumulate_blklen16_r2c1blk4_avx512(
                        av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, QuantBDataPtr + StrideQuantBData,
                        QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr + StrideQuantBScale,
                        acc[1], acc[NCols4 + 1]
                    );
                    accumulate_blklen16_r2c1blk4_avx512(
                        av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8,
                        QuantBDataPtr + 2 * StrideQuantBData, QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr + 2 * StrideQuantBScale,
                        acc[2], acc[NCols4 + 2]
                    );
                    accumulate_blklen16_r2c1blk4_avx512(
                        av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8,
                        QuantBDataPtr + 3 * StrideQuantBData, QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr + 3 * StrideQuantBScale,
                        acc[3], acc[NCols4 + 3]
                    );
                }

                // increment block pointers
                QuantAPtr += BlkLen16 * PerAccuBlk8;
                QuantAScalePtr += PerAccuBlk8;
                QuantBDataPtr += BlkDataSizeInBytes * PerAccuBlk8;
                QuantBScalePtr += PerAccuBlk8;
            }  // k_blks_remaining

            __m256 acc2[NCols4 * NRows2] = {
                h_add_512(acc[0]),
                h_add_512(acc[1]),
                h_add_512(acc[2]),
                h_add_512(acc[3]),
                h_add_512(acc[4]),
                h_add_512(acc[5]),
                h_add_512(acc[6]),
                h_add_512(acc[7])
            };

            while (k_blks_remaining-- > 0) {
                const std::byte* QuantABlk0 = QuantAPtr;
                const __m256i av0_16_epi16 = load_16_epi8_as_epi16(QuantABlk0);
                const __m256i av1_16_epi16 = load_16_epi8_as_epi16(QuantABlk0 + lda);

                const float& scale_a00 = *QuantAScalePtr;
                const float& scale_a10 = *(QuantAScalePtr + BlockCountK);

                {
                    // Col0
                    const float scale_00 = scale_a00 * (QuantBScalePtr)[0];
                    const float scale_10 = scale_a10 * (QuantBScalePtr)[0];
                    accumulate_blklen16_r2c1blk1_avx2(av0_16_epi16, av1_16_epi16, QuantBDataPtr, scale_00, scale_10, acc2[0], acc2[NCols4]);
                }

                {
                    // Col1
                    const float scale_00 = scale_a00 * (QuantBScalePtr + StrideQuantBScale)[0];
                    const float scale_10 = scale_a10 * (QuantBScalePtr + StrideQuantBScale)[0];
                    accumulate_blklen16_r2c1blk1_avx2(av0_16_epi16, av1_16_epi16, QuantBDataPtr + StrideQuantBData, scale_00, scale_10,
                      acc2[1], acc2[NCols4 + 1]);
                }

                {
                    // Col2
                    const float scale_00 = scale_a00 * (QuantBScalePtr + 2 * StrideQuantBScale)[0];
                    const float scale_10 = scale_a10 * (QuantBScalePtr + 2 * StrideQuantBScale)[0];
                    accumulate_blklen16_r2c1blk1_avx2(av0_16_epi16, av1_16_epi16, QuantBDataPtr + 2 * StrideQuantBData, scale_00, scale_10,
                      acc2[2], acc2[NCols4 + 2]);
                }

                {
                    // Col3
                    const float& scale_00 = scale_a00 * (QuantBScalePtr + 3 * StrideQuantBScale)[0];
                    const float& scale_10 = scale_a10 * (QuantBScalePtr + 3 * StrideQuantBScale)[0];
                    accumulate_blklen16_r2c1blk1_avx2(
                        av0_16_epi16, av1_16_epi16, QuantBDataPtr + 3 * StrideQuantBData, scale_00, scale_10,
                      acc2[3], acc2[NCols4 + 3]);
                }
                QuantAPtr += BlkLen16;
                QuantAScalePtr++;
                QuantBDataPtr += BlkDataSizeInBytes;
                QuantBScalePtr++;
            }  // k_blks_remaining

            __m128 acc_r0 = FoldAccumulators(acc2[0], acc2[1], acc2[2], acc2[3]);
            __m128 acc_r1 = FoldAccumulators(acc2[NCols4 + 0], acc2[NCols4 + 1], acc2[NCols4 + 2], acc2[NCols4 + 3]);
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

template <bool vnni>
void MLAS_FORCEINLINE
Q4Int8GemmR2C1BlkLen16Avx512(
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
    constexpr size_t BlkDataSizeInBytes = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen16);

    // process 2 blks of 64 4b weights a time
    constexpr size_t PerAccuBlk8 = 8;

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

            __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();

            size_t k_blks_remaining = BlockCountK;
            // process 2 blks of 64 4b weights a time
            for (; k_blks_remaining >= PerAccuBlk8; k_blks_remaining -= PerAccuBlk8) {
                const __m512i av_00_epi8 = _mm512_loadu_si512((const __m512i*)QuantAPtr);
                const __m512i av_01_epi8 = _mm512_loadu_si512((const __m512i*)(QuantAPtr + 64));
                const __m512i av_10_epi8 = _mm512_loadu_si512((const __m512i*)(QuantAPtr + lda));
                const __m512i av_11_epi8 = _mm512_loadu_si512((const __m512i*)(QuantAPtr + lda + 64));

                if constexpr (vnni) {
                    accumulate_blklen16_r2c1blk4_avx512vnni(
                        av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, QuantBDataPtr,
                        QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr, acc0, acc1
                    );
                } else {
                    accumulate_blklen16_r2c1blk4_avx512(
                        av_00_epi8, av_01_epi8, av_10_epi8, av_11_epi8, QuantBDataPtr,
                        QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr, acc0, acc1
                    );
                }

                // increment block pointers
                QuantAPtr += BlkLen16 * PerAccuBlk8;
                QuantAScalePtr += PerAccuBlk8;
                QuantBDataPtr += BlkDataSizeInBytes * PerAccuBlk8;
                QuantBScalePtr += PerAccuBlk8;
            }

            __m256 acc20 = h_add_512(acc0);
            __m256 acc21 = h_add_512(acc1);
            while (k_blks_remaining-- > 0) {
                const std::byte* QuantABlk0 = QuantAPtr;
                const __m256i av0_16_epi16 = load_16_epi8_as_epi16(QuantABlk0);
                const __m256i av1_16_epi16 = load_16_epi8_as_epi16(QuantABlk0 + lda);

                const float& scale_a00 = *QuantAScalePtr;
                const float& scale_a10 = *(QuantAScalePtr + BlockCountK);

                const float& scale_00 = scale_a00 * (QuantBScalePtr)[0];
                const float& scale_10 = scale_a10 * (QuantBScalePtr)[0];
                accumulate_blklen16_r2c1blk1_avx2(av0_16_epi16, av1_16_epi16, QuantBDataPtr, scale_00, scale_10, acc20, acc21);

                QuantAPtr += BlkLen16;
                QuantAScalePtr++;
                QuantBDataPtr += BlkDataSizeInBytes;
                QuantBScalePtr++;
            }

            *SumPtr = hsum_float_8(acc20);
            *(SumPtr + ldc) = hsum_float_8(acc21);
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
Q4Int8GemmR1xC4BlkLen16Avx512(
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
    constexpr size_t BlkDataSizeInBytes = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen16);

    // process 2 blks of 64 4b weights a time
    constexpr size_t PerAccuBlk8 = 8;

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

            __m512 acc[NCols4] = {
              _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()
            };
            size_t k_blks_remaining = BlockCountK;
            for (; k_blks_remaining >= PerAccuBlk8; k_blks_remaining -= PerAccuBlk8) {
                const __m512i av_00_epi8 = _mm512_loadu_si512((const __m512i*)QuantAPtr);
                const __m512i av_01_epi8 = _mm512_loadu_si512((const __m512i*)(QuantAPtr + 64));

                if constexpr (vnni) {
                    accumulate_blklen16_r1c1blk8_avx512vnni(av_00_epi8, av_01_epi8, QuantBDataPtr, QuantAScalePtr, QuantBScalePtr, acc[0]);
                    accumulate_blklen16_r1c1blk8_avx512vnni(av_00_epi8, av_01_epi8, QuantBDataPtr + StrideQuantBData, QuantAScalePtr, QuantBScalePtr + StrideQuantBScale, acc[1]);
                    accumulate_blklen16_r1c1blk8_avx512vnni(av_00_epi8, av_01_epi8, QuantBDataPtr + 2 * StrideQuantBData, QuantAScalePtr, QuantBScalePtr + 2 * StrideQuantBScale, acc[2]);
                    accumulate_blklen16_r1c1blk8_avx512vnni(av_00_epi8, av_01_epi8, QuantBDataPtr + 3 * StrideQuantBData, QuantAScalePtr, QuantBScalePtr + 3 * StrideQuantBScale, acc[3]);
                } else {
                    accumulate_blklen16_r1c1blk8_avx512(av_00_epi8, av_01_epi8, QuantBDataPtr, QuantAScalePtr, QuantBScalePtr, acc[0]);
                    accumulate_blklen16_r1c1blk8_avx512(av_00_epi8, av_01_epi8, QuantBDataPtr + StrideQuantBData, QuantAScalePtr, QuantBScalePtr + StrideQuantBScale, acc[1]);
                    accumulate_blklen16_r1c1blk8_avx512(av_00_epi8, av_01_epi8, QuantBDataPtr + 2 * StrideQuantBData, QuantAScalePtr, QuantBScalePtr + 2 * StrideQuantBScale, acc[2]);
                    accumulate_blklen16_r1c1blk8_avx512(av_00_epi8, av_01_epi8, QuantBDataPtr + 3 * StrideQuantBData, QuantAScalePtr, QuantBScalePtr + 3 * StrideQuantBScale, acc[3]);
                }

                QuantAPtr += BlkLen16 * PerAccuBlk8;
                QuantAScalePtr += PerAccuBlk8;
                QuantBDataPtr += BlkDataSizeInBytes * PerAccuBlk8;
                QuantBScalePtr += PerAccuBlk8;
            }

            __m256 acc2[NCols4] = {
                h_add_512(acc[0]), h_add_512(acc[1]), h_add_512(acc[2]), h_add_512(acc[3])
            };

            while (k_blks_remaining-- > 0) {
                const std::byte* QuantABlk0 = QuantAPtr;
                const __m256i av_00_epi8 = load_16_epi8_as_epi16(QuantABlk0);

                const float& scale_a00 = *QuantAScalePtr;
                {
                    const float& scale_00 = scale_a00 * (QuantBScalePtr)[0];
                    accumulate_blklen16_r1c1blk1_avx2(av_00_epi8, QuantBDataPtr, scale_00, acc2[0]);
                }
                {
                    const float& scale_00 = scale_a00 * (QuantBScalePtr + StrideQuantBScale)[0];
                    accumulate_blklen16_r1c1blk1_avx2(av_00_epi8, QuantBDataPtr + StrideQuantBData, scale_00, acc2[1]);
                }
                {
                    const float& scale_00 = scale_a00 * (QuantBScalePtr + 2 * StrideQuantBScale)[0];
                    accumulate_blklen16_r1c1blk1_avx2(av_00_epi8, QuantBDataPtr + 2 * StrideQuantBData, scale_00, acc2[2]);
                }
                {
                    const float& scale_00 = scale_a00 * (QuantBScalePtr + 3 * StrideQuantBScale)[0];
                    accumulate_blklen16_r1c1blk1_avx2(av_00_epi8, QuantBDataPtr + 3 * StrideQuantBData, scale_00, acc2[3]);
                }

                QuantAPtr += BlkLen16;
                QuantAScalePtr++;
                QuantBDataPtr += BlkDataSizeInBytes;
                QuantBScalePtr++;

            }

            __m128 acc_r0 = FoldAccumulators(acc2[0], acc2[1], acc2[2], acc2[3]);
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

template <bool vnni>
MLAS_FORCEINLINE void
Q4Int8GemmR1xC1BlkLen16Avx512(
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
    constexpr size_t BlkDataSizeInBytes = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen16);

    // process 2 blks of 64 4b weights a time
    constexpr size_t PerAccuBlk8 = 8;

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

            __m512 acc0 = _mm512_setzero_ps();
            size_t k_blks_remaining = BlockCountK;
            for (; k_blks_remaining >= PerAccuBlk8; k_blks_remaining -= PerAccuBlk8) {
                const __m512i av_00_epi8 = _mm512_loadu_si512((const __m512i*)QuantAPtr);
                const __m512i av_01_epi8 = _mm512_loadu_si512((const __m512i*)(QuantAPtr + 64));

                if constexpr (vnni) {
                    accumulate_blklen16_r1c1blk8_avx512vnni(av_00_epi8, av_01_epi8, QuantBDataPtr, QuantAScalePtr, QuantBScalePtr, acc0);
                } else {
                    accumulate_blklen16_r1c1blk8_avx512(av_00_epi8, av_01_epi8, QuantBDataPtr, QuantAScalePtr, QuantBScalePtr, acc0);
                }

                QuantAPtr += BlkLen16 * PerAccuBlk8;
                QuantAScalePtr += PerAccuBlk8;
                QuantBDataPtr += BlkDataSizeInBytes * PerAccuBlk8;
                QuantBScalePtr += PerAccuBlk8;
            }

            __m256 acc2 = h_add_512(acc0);
            while (k_blks_remaining-- > 0) {
                const __m256i av_00_epi8 = load_16_epi8_as_epi16(QuantAPtr);

                const float& scale_a00 = *QuantAScalePtr;
                const float& scale_00 = scale_a00 * (QuantBScalePtr)[0];
                accumulate_blklen16_r1c1blk1_avx2(av_00_epi8, QuantBDataPtr, scale_00, acc2);

                QuantAPtr += BlkLen16;
                QuantAScalePtr++;
                QuantBDataPtr += BlkDataSizeInBytes;
                QuantBScalePtr++;
            }

            *SumPtr = hsum_float_8(acc2);
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

template<bool vnni>
MLAS_FORCEINLINE
    size_t
MlasQ4Int8GemmKernelBlkLen16Avx512(
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
        Q4Int8GemmR2xC4BlkLen16Avx512<vnni>(
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
        Q4Int8GemmR2C1BlkLen16Avx512<vnni>(
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
        Q4Int8GemmR1xC4BlkLen16Avx512<vnni>(
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
        Q4Int8GemmR1xC1BlkLen16Avx512<vnni>(
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
