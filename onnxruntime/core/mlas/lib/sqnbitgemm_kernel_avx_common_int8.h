#pragma once
#include <algorithm>
#include <cassert>
#include <utility>

#include "qnbitgemm.h"
#include "sqnbitgemm_kernel_avx_common.h"
#include "sqnbitgemm_q8_block.h"

template <size_t NCols, bool HasZeroPoint>
MLAS_FORCEINLINE void
ComputeDotProducts_BlkBitWidth4_CompInt8_SubBlkLen16(
    size_t BlkLen,
    const std::byte* QuantARowPtr,
    const std::byte* QuantBDataColPtr,
    const float* QuantBScaleColPtr,
    const std::byte* QuantBZeroPointColPtr,
    float* SumPtr,
    size_t CountK,
    size_t StrideQuantBData,
    size_t StrideQuantBScale,
    size_t StrideQuantBZeroPoint,
    const float* BiasPtr
)
{
    if constexpr (!HasZeroPoint) {
        // Suppress unused variable warnings
        (void)QuantBZeroPointColPtr;
        (void)StrideQuantBZeroPoint;
    }

    assert(BlkLen == 16);
    constexpr size_t SubBlkLen = 16;
    const __m128i low_mask = _mm_set1_epi8(0xF);

    constexpr size_t BlkBitWidth = 4;
    constexpr size_t SubBlkStep = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, SubBlkLen);

    __m256 acc[NCols];
    UnrolledLoop<NCols>([&](size_t i) {
        acc[i] = _mm256_setzero_ps();
    });

    const std::byte* ablob = QuantARowPtr;
    const auto* b = QuantBDataColPtr;
    const float* s = QuantBScaleColPtr;

    [[maybe_unused]] size_t QuantBZeroPointIdx = 0;  // track half byte increments with this index instead of a pointer
    // only used if HasZeroPoint == true

    for (size_t k = 0; k < CountK; k += BlkLen) {
        const float a_scale = Q8BlkScale(ablob);
        ablob += sizeof(float);

        float scale_v[NCols];
        UnrolledLoop<NCols>([&](size_t i) {
            scale_v[i] = (*(s + StrideQuantBScale * i)) * a_scale;
        });

        std::byte* bptr[NCols];
        UnrolledLoop<NCols>([&](size_t i) {
            bptr[i] = (std::byte*)(b + StrideQuantBData * i);
        });

        [[maybe_unused]] uint8_t offset[NCols];
        // not ready for "Manual conversion to float" in neon yet. following neon to unpack to uint8_t.
        if constexpr (HasZeroPoint) {
            UnrolledLoop<NCols>([&](size_t i) {
                const std::byte zp_packed =
                    QuantBZeroPointColPtr[i * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
                const std::byte zp = ((QuantBZeroPointIdx & 1) == 1)
                                         ? (zp_packed >> 4)
                                         : (zp_packed & std::byte{0x0F});
                offset[i] = std::to_integer<uint8_t>(zp);
            });
        }

        // Load A row vector
        const __m128i av_epi8 = _mm_lddqu_si128((const __m128i*)ablob);
        __m256i av_epi16 = _mm256_cvtepi8_epi16(av_epi8);
        ablob += BlkLen;

        // Load 4 B column vectors (quantized to int4 blobs)
        __m128i bvi[NCols];
        UnrolledLoop<NCols>([&](size_t i) {
            bvi[i] = _mm_loadl_epi64((__m128i const*)bptr[i]);
            bptr[i] += SubBlkStep;
        });

        // expand 4b into byte array
        __m256i bv_epi16[NCols];
        UnrolledLoop<NCols>([&](size_t i) {
            const __m128i lower = _mm_and_si128(bvi[i], low_mask);
            const __m128i upper = _mm_bslli_si128(_mm_and_si128(_mm_srli_epi16(bvi[i], 4), low_mask), 8);
            bv_epi16[i] = _mm256_cvtepi8_epi16(_mm_add_epi8(upper, lower));
        });

        // Subtract zero-point from the integers
        if constexpr (HasZeroPoint) {
            UnrolledLoop<NCols>([&](size_t i) {
                bv_epi16[i] = _mm256_sub_epi16(bv_epi16[i], _mm256_set1_epi16(offset[i]));
            });
        } else {
            const __m256i eight = _mm256_set1_epi16(8);
            UnrolledLoop<NCols>([&](size_t i) {
                bv_epi16[i] = _mm256_sub_epi16(bv_epi16[i], eight);
            });
        }

        UnrolledLoop<NCols>([&](size_t i) {
            __m256i prod_8_epi32 = _mm256_madd_epi16(bv_epi16[i], av_epi16);

            const __m256 prod_8_ps = _mm256_cvtepi32_ps(prod_8_epi32);
            acc[i] = _mm256_fmadd_ps(_mm256_set1_ps(scale_v[i]), prod_8_ps, acc[i]);
        });

        b += MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
        s++;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointIdx += 1;
        }
    }

    if constexpr (NCols == 4) {
        __m128 acc_x = FoldAccumulators(acc[0], acc[1], acc[2], acc[3]);
        if (BiasPtr != nullptr) {
            acc_x = _mm_add_ps(acc_x, _mm_loadu_ps(BiasPtr));
        }
        _mm_storeu_ps(SumPtr, acc_x);
    } else {
        UnrolledLoop<NCols>([&](size_t i) {
            __m128 vlow = _mm256_castps256_ps128(acc[i]);
            __m128 vhigh = _mm256_extractf128_ps(acc[i], 1);  // Extract high 128 bit

            // Add the two 128-bit vectors together
            __m128 vsum = _mm_add_ps(vlow, vhigh);
            // Horizontally add the elements of the resulting 128-bit vector
            vsum = _mm_hadd_ps(vsum, vsum);
            vsum = _mm_hadd_ps(vsum, vsum);

            _mm_store_ss(&SumPtr[i], vsum);
            SumPtr[i] += BiasPtr == nullptr ? 0.0f : BiasPtr[i];
        });
    }
}

template <bool HasZeroPoint>
void
SQ4BitGemmM1Kernel_BlkLen16_CompInt8_Impl(
    const std::byte* QuantA,
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
    constexpr size_t NCols4 = 4;
    constexpr size_t BlkBitWidth4 = 4;
    constexpr size_t BlkLen16 = 16;

    const std::byte* QuantARowPtr = QuantA;
    float* CRowPtr = C;

    const size_t BlockCountK = BlockStrideQuantB;

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen16);
    const size_t StrideQuantBScale = BlockCountK;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth4>(BlockCountK);

    const float* BiasPtr = Bias;

    const std::byte* QuantBDataColPtr = QuantBData;
    const float* QuantBScaleColPtr = QuantBScale;
    const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

    float* SumPtr = CRowPtr;

    int64_t nblk = static_cast<int64_t>(CountN) - NCols4;

    while (nblk >= 0) {
        ComputeDotProducts_BlkBitWidth4_CompInt8_SubBlkLen16<NCols4, HasZeroPoint>(
            BlkLen16, QuantARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr,
            SumPtr, CountK, StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint, BiasPtr
        );

        // move to next `NCols` columns

        QuantBDataColPtr += NCols4 * StrideQuantBData;
        QuantBScaleColPtr += NCols4 * StrideQuantBScale;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointColPtr += NCols4 * StrideQuantBZeroPoint;
        }

        BiasPtr += BiasPtr != nullptr ? NCols4 : 0;
        SumPtr += NCols4;

        nblk -= NCols4;
    }

    // left over columns less than `NCols`?
    nblk += NCols4;
    for (int64_t n = 0; n < nblk; ++n) {
        ComputeDotProducts_BlkBitWidth4_CompInt8_SubBlkLen16<1, HasZeroPoint>(
            BlkLen16, QuantARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr,
            SumPtr, CountK, StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint, BiasPtr
        );

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

template <bool HasZeroPoint, AccumulateFunctionType<HasZeroPoint> accumulator>
void
SQ4BitGemmM1Kernel_BlkLen32_CompInt8_Impl(
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

    float* CRowPtr = C;

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t StrideQuantBScale = BlockCountK;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);

    const float* BiasPtr = Bias;

    const std::byte* QuantBDataColPtr = QuantBData;
    const float* QuantBScaleColPtr = QuantBScale;
    const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

    float* SumPtr = CRowPtr;

    const __m256i zero = _mm256_setzero_si256();
    const __m128i low_mask = _mm_set1_epi8(0xF);
    const size_t NCols = 4;
    int64_t nblk = (int64_t)(CountN)-4;
    while (nblk >= 0) {
        const std::byte* QuantAPtr = QuantA;
        const float* QuantAScalePtr = QuantAScale;
        const std::byte* QuantBDataPtr = QuantBDataColPtr;
        const float* QuantBScalePtr = QuantBScaleColPtr;
        const std::byte* QuantBZeroPointPtr = QuantBZeroPointColPtr;

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

            const float& scale_a0 = *QuantAScalePtr;
            const float& scale_a1 = *(QuantAScalePtr + 1);

            // Col0
            const float& scale_00 = scale_a0 * QuantBScalePtr[0];
            const float& scale_01 = scale_a1 * QuantBScalePtr[1];
            accumulator(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr), low_mask, zero, QuantBZeroPointPtr, true, scale_00, acc0);
            accumulator(av_1_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 16), low_mask, zero, QuantBZeroPointPtr, false, scale_01, acc0);

            // Col1
            const float& scale_10 = scale_a0 * (QuantBScalePtr + StrideQuantBScale)[0];
            const float& scale_11 = scale_a1 * (QuantBScalePtr + StrideQuantBScale)[1];
            accumulator(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + StrideQuantBData), low_mask, zero, QuantBZeroPointPtr + StrideQuantBZeroPoint, true, scale_10, acc1);
            accumulator(av_1_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + StrideQuantBData + 16), low_mask, zero, QuantBZeroPointPtr + StrideQuantBZeroPoint, false, scale_11, acc1);

            // Col2
            const float& scale_20 = scale_a0 * (QuantBScalePtr + 2 * StrideQuantBScale)[0];
            const float& scale_21 = scale_a1 * (QuantBScalePtr + 2 * StrideQuantBScale)[1];
            accumulator(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 2 * StrideQuantBData), low_mask, zero, QuantBZeroPointPtr + 2 * StrideQuantBZeroPoint, true, scale_20, acc2);
            accumulator(av_1_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 2 * StrideQuantBData + 16), low_mask, zero, QuantBZeroPointPtr + 2 * StrideQuantBZeroPoint, false, scale_21, acc2);

            // Col3
            const float& scale_30 = scale_a0 * (QuantBScalePtr + 3 * StrideQuantBScale)[0];
            const float& scale_31 = scale_a1 * (QuantBScalePtr + 3 * StrideQuantBScale)[1];
            accumulator(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 3 * StrideQuantBData), low_mask, zero, QuantBZeroPointPtr + 3 * StrideQuantBZeroPoint, true, scale_30, acc3);
            accumulator(av_1_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 3 * StrideQuantBData + 16), low_mask, zero, QuantBZeroPointPtr + 3 * StrideQuantBZeroPoint, false, scale_31, acc3);

            // increment block pointers
            QuantAPtr += BlkLen * 2;
            QuantAScalePtr += 2;
            QuantBDataPtr += 16 * 2;
            QuantBScalePtr += 2;
            if constexpr (HasZeroPoint) {
                QuantBZeroPointPtr += 1;
            }
        }

        if (k_blks_remaining > 0) {
            // load A
            const std::byte* QuantABlk0 = QuantAPtr;
            const __m256i av_0_epi8 = _mm256_loadu_si256((const __m256i*)QuantABlk0);

            const float& scale_a0 = *QuantAScalePtr;

            // Col0
            const float& scale_00 = scale_a0 * QuantBScalePtr[0];
            accumulator(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr), low_mask, zero, QuantBZeroPointPtr, true, scale_00, acc0);

            // Col1
            const float& scale_10 = scale_a0 * (QuantBScalePtr + StrideQuantBScale)[0];
            accumulator(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + StrideQuantBData), low_mask, zero, QuantBZeroPointPtr + StrideQuantBZeroPoint, true, scale_10, acc1);

            // Col2
            const float& scale_20 = scale_a0 * (QuantBScalePtr + 2 * StrideQuantBScale)[0];
            accumulator(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 2 * StrideQuantBData), low_mask, zero, QuantBZeroPointPtr + 2 * StrideQuantBZeroPoint, true, scale_20, acc2);

            // Col3
            const float& scale_30 = scale_a0 * (QuantBScalePtr + 3 * StrideQuantBScale)[0];
            accumulator(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 3 * StrideQuantBData), low_mask, zero, QuantBZeroPointPtr + 3 * StrideQuantBZeroPoint, true, scale_30, acc3);
        }

        __m128 acc_x = FoldAccumulators(acc0, acc1, acc2, acc3);
        if (BiasPtr != nullptr) {
            acc_x = _mm_add_ps(acc_x, _mm_loadu_ps(BiasPtr));
        }
        _mm_storeu_ps(SumPtr, acc_x);

        // move to next NCols columns

        QuantBDataColPtr += NCols * StrideQuantBData;
        QuantBScaleColPtr += NCols * StrideQuantBScale;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointColPtr += NCols * StrideQuantBZeroPoint;
        }

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

        __m256 acc0 = _mm256_setzero_ps();

        size_t k_blks_remaining = BlockCountK;
        for (; k_blks_remaining > 1; k_blks_remaining -= 2) {
            const std::byte* QuantABlk0 = QuantAPtr;
            const std::byte* QuantABlk1 = QuantABlk0 + BlkLen;

            // load A:
            const __m256i av_0_epi8 = _mm256_loadu_si256((const __m256i*)QuantABlk0);
            const __m256i av_1_epi8 = _mm256_loadu_si256((const __m256i*)QuantABlk1);

            const float& scale_a0 = *QuantAScalePtr;
            const float& scale_a1 = *(QuantAScalePtr + 1);

            // Col0
            const float& scale_00 = scale_a0 * QuantBScalePtr[0];
            const float& scale_01 = scale_a1 * QuantBScalePtr[1];
            accumulator(av_0_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr), low_mask, zero, QuantBZeroPointPtr, true, scale_00, acc0);
            accumulator(av_1_epi8, reinterpret_cast<const __m128i*>(QuantBDataPtr + 16), low_mask, zero, QuantBZeroPointPtr, false, scale_01, acc0);

            // increment block pointers
            QuantAPtr += BlkLen * 2;
            QuantAScalePtr += 2;
            QuantBDataPtr += 16 * 2;
            QuantBScalePtr += 2;
            if constexpr (HasZeroPoint) {
                QuantBZeroPointPtr += 1;
            }
        }

        if (k_blks_remaining > 0) {
            // load A
            const std::byte* QuantABlk0 = QuantAPtr;
            const __m256i av_0_epi8 = _mm256_loadu_si256((const __m256i*)QuantABlk0);

            const float& scale_a0 = *QuantAScalePtr;

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
}

using DotQuadFunctionType = __m256 (*)(
    const __m256i, const __m256i, const __m256i, const __m256i
);

template <bool HasZeroPoint, DotQuadFunctionType dot_quad>
MLAS_FORCEINLINE void
ComputeDotProducts_BlkBitWidth4_CompInt8_SubBlkLen64_NCols4(
    size_t BlkLen,
    const std::byte* QuantARowPtr,
    const std::byte* QuantBDataColPtr,
    const float* QuantBScaleColPtr,
    const std::byte* QuantBZeroPointColPtr,
    float* SumPtr,
    size_t CountK,
    size_t StrideQuantBData,
    size_t StrideQuantBScale,
    size_t StrideQuantBZeroPoint,
    const float* BiasPtr
)
{
    // TODO: make it work with all BlkLens
    assert(BlkLen >= 64);
    constexpr size_t SubBlkLen64 = 64;
    // const __m256i zero = _mm256_setzero_si256();
    const __m256i low_mask = _mm256_set1_epi8(0xF);

    constexpr size_t BlkBitWidth = 4;
    constexpr size_t SubBlkStep = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, SubBlkLen64);

    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps(), acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();

    const std::byte* ablob = QuantARowPtr;
    const std::byte* b_blk_data_ptr = QuantBDataColPtr;
    const float* blk_scale_ptr = QuantBScaleColPtr;

    [[maybe_unused]] size_t QuantBZeroPointIdx = 0;  // track half byte increments with this index instead of a pointer
    // only used if HasZeroPoint == true

    for (size_t k = 0; k < CountK; k += BlkLen) {
        size_t ck = std::min(CountK - k, BlkLen);

        const float a_scale = Q8BlkScale(ablob);
        ablob += sizeof(float);

        float
            scale_v0 = (*(blk_scale_ptr + StrideQuantBScale * 0)) * a_scale,
            scale_v1 = (*(blk_scale_ptr + StrideQuantBScale * 1)) * a_scale,
            scale_v2 = (*(blk_scale_ptr + StrideQuantBScale * 2)) * a_scale,
            scale_v3 = (*(blk_scale_ptr + StrideQuantBScale * 3)) * a_scale;

        const std::byte* bptr0 = (b_blk_data_ptr + StrideQuantBData * 0);
        const std::byte* bptr1 = (b_blk_data_ptr + StrideQuantBData * 1);
        const std::byte* bptr2 = (b_blk_data_ptr + StrideQuantBData * 2);
        const std::byte* bptr3 = (b_blk_data_ptr + StrideQuantBData * 3);

        uint8_t zp0, zp1, zp2, zp3;
        if constexpr (HasZeroPoint) {
            // TODO: this block causes near 30% of the computation.
            bool is_lower = (QuantBZeroPointIdx & 1) == 0;
            std::byte zp_packed = QuantBZeroPointColPtr[0 * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
            zp0 = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{0x0F}) : (zp_packed >> 4));
            zp_packed = QuantBZeroPointColPtr[1 * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
            zp1 = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{0x0F}) : (zp_packed >> 4));
            zp_packed = QuantBZeroPointColPtr[2 * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
            zp2 = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{0x0F}) : (zp_packed >> 4));
            zp_packed = QuantBZeroPointColPtr[3 * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
            zp3 = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{0x0F}) : (zp_packed >> 4));
        } else {
            zp0 = 8;
            zp1 = 8;
            zp2 = 8;
            zp3 = 8;
        }

        for (size_t kk = 0; kk < ck; kk += SubBlkLen64) {
            // Load A row vector
            const __m256i av0_32_epi8 = _mm256_loadu_si256((const __m256i*)ablob);
            ablob += 32;
            const __m256i av1_32_epi8 = _mm256_loadu_si256((const __m256i*)ablob);
            ablob += 32;

            // Load B column vectors (quantized to int4 blobs)
            // dst: | v0  v32 | v1  v33 | ... | v30 v62 | v31 v63 |
            __m256i bv = _mm256_loadu_si256((__m256i const*)bptr0);
            bptr0 += SubBlkStep;
            __m256i bv0_32_epi8 = _mm256_and_si256(bv, low_mask);
            __m256i bv1_32_epi8 = _mm256_and_si256(_mm256_srli_epi16(bv, 4), low_mask);
            __m256i zp_epi8 = _mm256_set1_epi8(zp0);
            bv0_32_epi8 = _mm256_sub_epi8(bv0_32_epi8, zp_epi8);
            bv1_32_epi8 = _mm256_sub_epi8(bv1_32_epi8, zp_epi8);
            __m256 sum_ps = dot_quad(bv0_32_epi8, bv1_32_epi8, av0_32_epi8, av1_32_epi8);
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(scale_v0), sum_ps, acc0);

            bv = _mm256_loadu_si256((__m256i const*)bptr1);
            bptr1 += SubBlkStep;
            bv0_32_epi8 = _mm256_and_si256(bv, low_mask);
            bv1_32_epi8 = _mm256_and_si256(_mm256_srli_epi16(bv, 4), low_mask);
            zp_epi8 = _mm256_set1_epi8(zp1);
            bv0_32_epi8 = _mm256_sub_epi8(bv0_32_epi8, zp_epi8);
            bv1_32_epi8 = _mm256_sub_epi8(bv1_32_epi8, zp_epi8);
            sum_ps = dot_quad(bv0_32_epi8, bv1_32_epi8, av0_32_epi8, av1_32_epi8);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(scale_v1), sum_ps, acc1);

            bv = _mm256_loadu_si256((__m256i const*)bptr2);
            bptr2 += SubBlkStep;
            bv0_32_epi8 = _mm256_and_si256(bv, low_mask);
            bv1_32_epi8 = _mm256_and_si256(_mm256_srli_epi16(bv, 4), low_mask);
            zp_epi8 = _mm256_set1_epi8(zp2);
            bv0_32_epi8 = _mm256_sub_epi8(bv0_32_epi8, zp_epi8);
            bv1_32_epi8 = _mm256_sub_epi8(bv1_32_epi8, zp_epi8);
            sum_ps = dot_quad(bv0_32_epi8, bv1_32_epi8, av0_32_epi8, av1_32_epi8);
            acc2 = _mm256_fmadd_ps(_mm256_set1_ps(scale_v2), sum_ps, acc2);

            bv = _mm256_loadu_si256((__m256i const*)bptr3);
            bptr3 += SubBlkStep;
            bv0_32_epi8 = _mm256_and_si256(bv, low_mask);
            bv1_32_epi8 = _mm256_and_si256(_mm256_srli_epi16(bv, 4), low_mask);
            zp_epi8 = _mm256_set1_epi8(zp3);
            bv0_32_epi8 = _mm256_sub_epi8(bv0_32_epi8, zp_epi8);
            bv1_32_epi8 = _mm256_sub_epi8(bv1_32_epi8, zp_epi8);
            sum_ps = dot_quad(bv0_32_epi8, bv1_32_epi8, av0_32_epi8, av1_32_epi8);
            acc3 = _mm256_fmadd_ps(_mm256_set1_ps(scale_v3), sum_ps, acc3);
        }  // kk

        b_blk_data_ptr += MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
        blk_scale_ptr++;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointIdx += 1;
        }
    }  // k

    __m128 acc_x = FoldAccumulators(acc0, acc1, acc2, acc3);
    if (BiasPtr != nullptr) {
        acc_x = _mm_add_ps(acc_x, _mm_loadu_ps(BiasPtr));
    }
    _mm_storeu_ps(SumPtr, acc_x);
}

// TODO: is this able to be inlined if DotQuadFunctionType is a function pointer?
template <bool HasZeroPoint, DotQuadFunctionType dot_quad>
MLAS_FORCEINLINE void
ComputeDotProducts_BlkBitWidth4_CompInt8_SubBlkLen64_NCols1(
    size_t BlkLen,
    const std::byte* QuantARowPtr,
    const std::byte* QuantBDataColPtr,
    const float* QuantBScaleColPtr,
    const std::byte* QuantBZeroPointColPtr,
    float* SumPtr,
    size_t CountK,
    size_t StrideQuantBData,
    size_t StrideQuantBScale,
    size_t StrideQuantBZeroPoint,
    const float* BiasPtr
)
{
    // TODO: make it work with all BlkLens
    assert(BlkLen >= 64);
    constexpr size_t SubBlkLen64 = 64;
    // const __m256i zero = _mm256_setzero_si256();
    const __m256i low_mask = _mm256_set1_epi8(0xF);

    constexpr size_t BlkBitWidth = 4;
    constexpr size_t SubBlkStep = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, SubBlkLen64);

    __m256 acc0 = _mm256_setzero_ps();

    const std::byte* ablob = QuantARowPtr;
    const std::byte* b_blk_data_ptr = QuantBDataColPtr;
    const float* blk_scale_ptr = QuantBScaleColPtr;

    [[maybe_unused]] size_t QuantBZeroPointIdx = 0;  // track half byte increments with this index instead of a pointer
    // only used if HasZeroPoint == true

    for (size_t k = 0; k < CountK; k += BlkLen) {
        size_t ck = std::min(CountK - k, BlkLen);

        const float a_scale = Q8BlkScale(ablob);
        ablob += sizeof(float);

        float scale_v0 = (*(blk_scale_ptr + StrideQuantBScale * 0)) * a_scale;

        const std::byte* bptr0 = (b_blk_data_ptr + StrideQuantBData * 0);

        uint8_t zp0;
        if constexpr (HasZeroPoint) {
            // TODO: this block causes near 30% of the computation.
            // The solution proposed by @yufenglee is to factor out scaleB * zp
            // while packing A. Will do this in next PR.
            bool is_lower = (QuantBZeroPointIdx & 1) == 0;
            std::byte zp_packed = QuantBZeroPointColPtr[0 * StrideQuantBZeroPoint + QuantBZeroPointIdx / 2];
            zp0 = std::to_integer<int8_t>(is_lower ? (zp_packed & std::byte{0x0F}) : (zp_packed >> 4));
        } else {
            zp0 = 8;
        }

        for (size_t kk = 0; kk < ck; kk += SubBlkLen64) {
            // Load A row vector
            const __m256i a_byte_lo = _mm256_loadu_si256((const __m256i*)ablob);
            ablob += 32;
            const __m256i a_byte_hi = _mm256_loadu_si256((const __m256i*)ablob);
            ablob += 32;

            // Load B column vectors (quantized to int4 blobs)
            // dst: | v0  v32 | v1  v33 | ... | v30 v62 | v31 v63 |
            __m256i bv = _mm256_loadu_si256((__m256i const*)bptr0);
            bptr0 += SubBlkStep;
            __m256i bv_lo_epi8 = _mm256_and_si256(bv, low_mask);
            __m256i bv_hi_epi8 = _mm256_and_si256(_mm256_srli_epi16(bv, 4), low_mask);
            __m256i zp_epi8 = _mm256_set1_epi8(zp0);
            bv_lo_epi8 = _mm256_sub_epi8(bv_lo_epi8, zp_epi8);
            bv_hi_epi8 = _mm256_sub_epi8(bv_hi_epi8, zp_epi8);
            __m256 sum_ps = dot_quad(bv_lo_epi8, bv_hi_epi8, a_byte_lo, a_byte_hi);
            //__m256 sum_ps = mul_sum_s8_quads_float_avx2(bv_lo_epi8, bv_hi_epi8, a_byte_lo, a_byte_hi);
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(scale_v0), sum_ps, acc0);
        }  // kk

        b_blk_data_ptr += MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
        blk_scale_ptr++;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointIdx += 1;
        }
    }  // k

    *SumPtr = hsum_float_8(acc0);
    *SumPtr += BiasPtr == nullptr ? 0.0f : *BiasPtr;
}

template <bool HasZeroPoint, DotQuadFunctionType dot_quad>
void
SQ4BitGemmM1Kernel_BlkLen64Plus_CompInt8_Impl(
    size_t BlkLen,
    const std::byte* QuantA,
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
    constexpr size_t BlkBitWidth = 4;

    const std::byte* QuantARowPtr = QuantA;
    float* CRowPtr = C;

    const size_t BlockCountK = BlockStrideQuantB;

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t StrideQuantBScale = BlockCountK;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);

    const float* BiasPtr = Bias;

    const std::byte* QuantBDataColPtr = QuantBData;
    const float* QuantBScaleColPtr = QuantBScale;
    const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

    float* SumPtr = CRowPtr;

    const size_t NCols = 4;
    int64_t nblk = static_cast<int64_t>(CountN) - NCols;

    while (nblk >= 0) {
        ComputeDotProducts_BlkBitWidth4_CompInt8_SubBlkLen64_NCols4<HasZeroPoint, dot_quad>(
            BlkLen, QuantARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr,
            SumPtr, CountK, StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint, BiasPtr
        );

        // move to next `NCols` columns

        QuantBDataColPtr += NCols * StrideQuantBData;
        QuantBScaleColPtr += NCols * StrideQuantBScale;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointColPtr += NCols * StrideQuantBZeroPoint;
        }

        BiasPtr += BiasPtr != nullptr ? NCols : 0;
        SumPtr += NCols;

        nblk -= NCols;
    }

    // left over columns less than `NCols`?
    nblk += NCols;
    for (int64_t n = 0; n < nblk; ++n) {
        ComputeDotProducts_BlkBitWidth4_CompInt8_SubBlkLen64_NCols1<HasZeroPoint, dot_quad>(
            BlkLen,
            QuantARowPtr, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr, CountK,
            StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
            BiasPtr
        );

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
