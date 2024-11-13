#pragma once
#include <algorithm>
#include <cassert>
#include <utility>

#include "sqnbitgemm.h"
#include "sqnbitgemm_kernel_avx_common.h"

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
