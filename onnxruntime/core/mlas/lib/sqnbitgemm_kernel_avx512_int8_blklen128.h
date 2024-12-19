#pragma once
#include <algorithm>
#include <cassert>
#include <utility>

#include "qnbitgemm.h"
#include "sqnbitgemm_kernel_avx_common.h"
#include "sqnbitgemm_kernel_avx512_int8_blklen64.h"

//static MLAS_FORCEINLINE __m512i
//combine_two_m256i_to_m512i(const __m256i& a, const __m256i& b)
//{
//    __m512i result = _mm512_castsi256_si512(a);
//    result = _mm512_inserti64x4(result, b, 1);
//    return result;
//}

//static MLAS_FORCEINLINE void
//load_2blk_4b_packed_blklen64(const std::byte* QuantBDataPtr, __m512i& bv0_64_epi8, __m512i& bv1_64_epi8)
//{
//    // | v0 v32 | v1 v33 | ... | v30 v62 | v31 v63 | v64 v96 | ... | v95 v127 |
//    const __m512i bv_packed = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(QuantBDataPtr));
//    const __m512i low_mask = _mm512_set1_epi8(0x0F);
//    __m512i bv0_64_epi8_ = _mm512_and_si512(bv_packed, low_mask); // 0~31, 64~95
//    __m512i bv1_64_epi8_ = _mm512_srli_epi16(_mm512_sub_epi8(bv_packed, bv0_64_epi8), 4);  // 32~63, 96~127
//
//    // Extract lower and higher 256 bits from bv0_64_epi8 and bv1_64_epi8
//    __m256i bv0_lower = _mm512_castsi512_si256(bv0_64_epi8_);
//    __m256i bv0_higher = _mm512_extracti64x4_epi64(bv0_64_epi8_, 1);
//    __m256i bv1_lower = _mm512_castsi512_si256(bv1_64_epi8_);
//    __m256i bv1_higher = _mm512_extracti64x4_epi64(bv1_64_epi8_, 1);
//
//    // Compose new __m512i variables
//    bv0_64_epi8 = _mm512_inserti64x4(_mm512_castsi256_si512(bv0_lower), bv1_lower, 1);
//    bv1_64_epi8 = _mm512_inserti64x4(_mm512_castsi256_si512(bv0_higher), bv1_higher, 1);
//}

static MLAS_FORCEINLINE void
dot_accumulate_1blk(
    const __m512i& bv0_64_epi8,
    const __m512i& bv1_64_epi8,
    const __m512i& av0_64_epi8,
    const __m512i& av1_64_epi8,
    const float combined_scale,
    __m512& acc
)
{
    __m512i dot0_32_epi16 = _mm512_maddubs_epi16(bv0_64_epi8, av0_64_epi8);
    __m512i dot1_32_epi16 = _mm512_maddubs_epi16(bv1_64_epi8, av1_64_epi8);
    __m512i t1 = _mm512_unpacklo_epi32(dot0_32_epi16, dot1_32_epi16);
    __m512i t2 = _mm512_unpackhi_epi32(dot0_32_epi16, dot1_32_epi16);
    __m512i sum_32_epi16 = _mm512_add_epi16(t1, t2);
    const __m512i zeros = _mm512_setzero_si512();
    const __m512i one_32_epi16 = _mm512_srli_epi16(_mm512_ternarylogic_epi32(zeros, zeros, zeros, 1), 15);
    __m512i sum_16_epi32 = _mm512_madd_epi16(one_32_epi16, sum_32_epi16);
    __m512 sum_16_ps = _mm512_cvtepi32_ps(sum_16_epi32);
    acc = _mm512_fmadd_ps(sum_16_ps, _mm512_set1_ps(combined_scale), acc);
}

static MLAS_FORCEINLINE void
dot_accumulate_1blkvnni(
    const __m512i& bv0_64_epi8,
    const __m512i& bv1_64_epi8,
    const __m512i& av0_64_epi8,
    const __m512i& av1_64_epi8,
    const float combined_scale,
    __m512& acc
)
{
    __m512i dot0_16_epi32 = _mm512_dpbusd_epi32(_mm512_setzero_epi32(), bv0_64_epi8, av0_64_epi8);
    __m512i dot1_16_epi32 = _mm512_dpbusd_epi32(dot0_16_epi32, bv1_64_epi8, av1_64_epi8);
    __m512 sum_16_ps = _mm512_cvtepi32_ps(dot1_16_epi32);
    acc = _mm512_fmadd_ps(sum_16_ps, _mm512_set1_ps(combined_scale), acc);
}

template<bool vnni>
static MLAS_FORCEINLINE void
accumulate_blklen128_r1c1blk1_avx512(
    const __m512i& av00_64_epi8,
    const __m512i& av01_64_epi8,
    const std::byte* QuantBDataPtr,
    const float* scale_a,
    const float* scale_b,
    __m512& acc
)
{
    __m512i bv0_64_epi8, bv1_64_epi8;
    load_2blk_4b_packed_blklen64(QuantBDataPtr, bv0_64_epi8, bv1_64_epi8);

    if constexpr (vnni) {
        dot_accumulate_1blkvnni(
            bv0_64_epi8, bv1_64_epi8, av00_64_epi8, av01_64_epi8,
            (*scale_a) * (*scale_b), acc
        );
    } else {
        dot_accumulate_1blk(
            bv0_64_epi8, bv1_64_epi8, av00_64_epi8, av01_64_epi8,
            (*scale_a) * (*scale_b), acc
        );
    }
}

template <bool vnni>
static MLAS_FORCEINLINE void
accumulate_blklen128_r2c1blk1_avx512(
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

    if constexpr (vnni) {
        dot_accumulate_1blkvnni(
            bv0_64_epi8, bv1_64_epi8, av00_64_epi8, av01_64_epi8,
            (*scale_a0) * (*scale_b), acc0
        );
        dot_accumulate_1blkvnni(
            bv0_64_epi8, bv1_64_epi8, av10_64_epi8, av11_64_epi8,
            (*scale_a1) * (*scale_b), acc1
        );
    } else {
        dot_accumulate_1blk(
            bv0_64_epi8, bv1_64_epi8, av00_64_epi8, av01_64_epi8,
            (*scale_a0) * (*scale_b), acc0
        );
        dot_accumulate_1blk(
            bv0_64_epi8, bv1_64_epi8, av10_64_epi8, av11_64_epi8,
            (*scale_a1) * (*scale_b), acc1
        );
    }
}

template <bool vnni>
MLAS_FORCEINLINE void
Q4Int8GemmR2xC4BlkLen128Avx512(
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
    constexpr size_t SubblkLen = 128;
    const size_t PerBlkSubblkCount = BlkLen / SubblkLen;
    const size_t BlkDataSizeInBytes = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen);
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

            __m512 acc[NCols4 * NRows2] = {
                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()
            };

            // process 1 blks of 64 4b weights a time
            for (size_t k = 0; k < BlockCountK; ++k) {
                for (size_t kk = 0; kk < PerBlkSubblkCount; kk++) {
                    const __m512i av00_64_epi8 = _mm512_loadu_si512((const __m512i*)QuantAPtr);
                    const __m512i av01_64_epi8 = _mm512_loadu_si512((const __m512i*)(QuantAPtr + SubblkLen / 2));
                    const __m512i av10_64_epi8 = _mm512_loadu_si512((const __m512i*)(QuantAPtr + lda));
                    const __m512i av11_64_epi8 = _mm512_loadu_si512((const __m512i*)(QuantAPtr + lda + SubblkLen / 2));

                    accumulate_blklen128_r2c1blk1_avx512<vnni>(av00_64_epi8, av01_64_epi8, av10_64_epi8, av11_64_epi8, QuantBDataPtr, QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr, acc[0], acc[NCols4]);
                    accumulate_blklen128_r2c1blk1_avx512<vnni>(av00_64_epi8, av01_64_epi8, av10_64_epi8, av11_64_epi8, QuantBDataPtr + SubblkDataSizeInBytes, QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr + 1, acc[1], acc[NCols4 + 1]);
                    accumulate_blklen128_r2c1blk1_avx512<vnni>(av00_64_epi8, av01_64_epi8, av10_64_epi8, av11_64_epi8, QuantBDataPtr + 2 * SubblkDataSizeInBytes, QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr + 2, acc[2], acc[NCols4 + 2]);
                    accumulate_blklen128_r2c1blk1_avx512<vnni>(av00_64_epi8, av01_64_epi8, av10_64_epi8, av11_64_epi8, QuantBDataPtr + 3 * SubblkDataSizeInBytes, QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr + 3, acc[3], acc[NCols4 + 3]);

                    // increment block pointers
                    QuantAPtr += SubblkLen;
                    QuantBDataPtr += NCols4 * SubblkDataSizeInBytes;
                }
                QuantAScalePtr++;
                QuantBScalePtr += NCols4;
            }  // k_blks_remaining

#if 1
            *SumPtr = _mm512_reduce_add_ps(acc[0]);
            *(SumPtr + 1) = _mm512_reduce_add_ps(acc[1]);
            *(SumPtr + 2) = _mm512_reduce_add_ps(acc[2]);
            *(SumPtr + 3) = _mm512_reduce_add_ps(acc[3]);
            *(SumPtr + ldc) = _mm512_reduce_add_ps(acc[NCols4]);
            *(SumPtr + ldc + 1) = _mm512_reduce_add_ps(acc[NCols4 + 1]);
            *(SumPtr + ldc + 2) = _mm512_reduce_add_ps(acc[NCols4 + 2]);
            *(SumPtr + ldc + 3) = _mm512_reduce_add_ps(acc[NCols4 + 3]);
            if (BiasPtr != nullptr) {
                *SumPtr += *BiasPtr;
                *(SumPtr + 1) += *(BiasPtr + 1);
                *(SumPtr + 2) += *(BiasPtr + 2);
                *(SumPtr + 3) += *(BiasPtr + 3);
                *(SumPtr + ldc) += *BiasPtr;
                *(SumPtr + ldc + 1) += *(BiasPtr + 1);
                *(SumPtr + ldc + 2) += *(BiasPtr + 2);
                *(SumPtr + ldc + 3) += *(BiasPtr + 3);
            }
#else
            __m128 acc_r0 = FoldAccumulators(acc[0], acc[1], acc[2], acc[3]);
            __m128 acc_r1 = FoldAccumulators(acc[NCols4 + 0], acc[NCols4 + 1], acc[NCols4 + 2], acc[NCols4 + 3]);
            if (BiasPtr != nullptr) {
                const __m128 bias_4_ps = _mm_loadu_ps(BiasPtr);
                acc_r0 = _mm_add_ps(acc_r0, bias_4_ps);
                acc_r1 = _mm_add_ps(acc_r1, bias_4_ps);
            }
            const __m128 level_r0 = _mm_loadu_ps(SumPtr);
            _mm_storeu_ps(SumPtr, _mm_sub_ps(acc_r0, level_r0));

            const __m128 level_r1 = _mm_loadu_ps(SumPtr + ldc);
            _mm_storeu_ps(SumPtr + ldc, _mm_sub_ps(acc_r1, level_r1));
#endif
            // move to next NCols columns
            QuantBDataColPtr += NCols4 * StrideQuantBData;
            QuantBScaleColPtr += NCols4 * BlockCountK;
            BiasPtr += BiasPtr != nullptr ? NCols4 : 0;
            SumPtr += NCols4;
        }
    }
}

template <bool vnni>
void MLAS_FORCEINLINE
Q4Int8GemmR2xC1BlkLen128Avx512(
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
    constexpr size_t SubblkLen = 128;

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

            __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();

            for (size_t k = 0; k < BlockCountK; ++k) {
                for (size_t kk = 0; kk < PerBlkSubblkCount; kk++) {
                    const __m512i av00_64_epi8 = _mm512_loadu_si512((const __m512i*)QuantAPtr);
                    const __m512i av01_64_epi8 = _mm512_loadu_si512((const __m512i*)(QuantAPtr + SubblkLen / 2));
                    const __m512i av10_64_epi8 = _mm512_loadu_si512((const __m512i*)(QuantAPtr + lda));
                    const __m512i av11_64_epi8 = _mm512_loadu_si512((const __m512i*)(QuantAPtr + lda + SubblkLen / 2));

                    accumulate_blklen128_r2c1blk1_avx512<vnni>(av00_64_epi8, av01_64_epi8, av10_64_epi8, av11_64_epi8,
                      QuantBDataPtr, QuantAScalePtr, QuantAScalePtr + BlockCountK, QuantBScalePtr, acc0, acc1);

                    // increment block pointers
                    QuantAPtr += SubblkLen;
                    QuantBDataPtr += SubblkDataSizeInBytes;
                }
                QuantAScalePtr++;
                QuantBScalePtr++;
            }

            *SumPtr = hsum_float_16(acc0);
            *(SumPtr + ldc) = hsum_float_16(acc1);
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
Q4Int8GemmR1xC4BlkLen128Avx512(
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
    constexpr size_t SubblkLen = 128;

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

            __m512 acc[NCols4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()};
            for (size_t k = 0; k < BlockCountK; ++k) {
                for (size_t kk = 0; kk < PerBlkSubblkCount; kk++) {
                    const __m512i av0_64_epi8 = _mm512_loadu_si512((const __m512i*)QuantAPtr);
                    const __m512i av1_64_epi8 = _mm512_loadu_si512((const __m512i*)(QuantAPtr + SubblkLen / 2));
                    accumulate_blklen128_r1c1blk1_avx512<vnni>(av0_64_epi8, av1_64_epi8, QuantBDataPtr, QuantAScalePtr, QuantBScalePtr, acc[0]);
                    accumulate_blklen128_r1c1blk1_avx512<vnni>(av0_64_epi8, av1_64_epi8, QuantBDataPtr + SubblkDataSizeInBytes, QuantAScalePtr, QuantBScalePtr + 1, acc[1]);
                    accumulate_blklen128_r1c1blk1_avx512<vnni>(av0_64_epi8, av1_64_epi8, QuantBDataPtr + 2 * SubblkDataSizeInBytes, QuantAScalePtr, QuantBScalePtr + 2, acc[2]);
                    accumulate_blklen128_r1c1blk1_avx512<vnni>(av0_64_epi8, av1_64_epi8, QuantBDataPtr + 3 * SubblkDataSizeInBytes, QuantAScalePtr, QuantBScalePtr + 3, acc[3]);

                    // increment block pointers
                    QuantAPtr += SubblkLen;
                    QuantBDataPtr += NCols4 * SubblkDataSizeInBytes;
                }
                QuantAScalePtr++;
                QuantBScalePtr +=NCols4;
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
Q4Int8GemmR1xC1BlkLen128Avx512(
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
    constexpr size_t SubblkLen = 128;

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

            __m512 acc0 = _mm512_setzero_ps();
            for (size_t k = 0; k < BlockCountK; ++k) {
                for (size_t kk = 0; kk < PerBlkSubblkCount; kk++) {
                    const __m512i av0_64_epi8 = _mm512_loadu_si512((const __m512i*)QuantAPtr);
                    const __m512i av1_64_epi8 = _mm512_loadu_si512((const __m512i*)(QuantAPtr + SubblkLen / 2));

                    accumulate_blklen128_r1c1blk1_avx512<vnni>(
                        av0_64_epi8, av1_64_epi8, QuantBDataPtr, QuantAScalePtr, QuantBScalePtr, acc0
                    );

                    // increment block pointers
                    QuantAPtr += SubblkLen;
                    QuantBDataPtr += SubblkDataSizeInBytes;
                }
                QuantAScalePtr++;
                QuantBScalePtr++;
            }

            *SumPtr = hsum_float_16(acc0);
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
MLAS_FORCEINLINE size_t
MlasQ4Int8GemmKernelBlkLen128Avx512(
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
        Q4Int8GemmR2xC4BlkLen128Avx512<vnni>(
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
        Q4Int8GemmR2xC1BlkLen128Avx512<vnni>(
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
        Q4Int8GemmR1xC4BlkLen128Avx512<vnni>(
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
        Q4Int8GemmR1xC1BlkLen128Avx512<vnni>(
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
