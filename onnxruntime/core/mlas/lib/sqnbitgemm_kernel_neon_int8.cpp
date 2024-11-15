/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_neon_int8.cpp

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication kernels for ARM NEON specific to
    input type T1 as float32 and
    MLAS_QNBIT_GEMM_COMPUTE_TYPE SQNBIT_CompInt8.

--*/

#include <arm_neon.h>

#include <cassert>

#include "qnbitgemm.h"
#include "qnbitgemm_kernel_neon.h"
#include "sqnbitgemm_q8_block.h"

namespace sqnbitgemm_neon
{

//
// SQNBIT_CompInt8 kernel implementation.
//

namespace
{

template <size_t SubBlkLen>
MLAS_FORCEINLINE void
QuantizeBlock(
    size_t BlkLen,
    const float* A,
    size_t ElementCount,
    std::byte* QuantA
)
{
    static_assert(SubBlkLen >= 16 && SubBlkLen % 16 == 0);

    assert(BlkLen % SubBlkLen == 0);

    //
    // Scan block values first to determine scale.
    //

    float amax = 0.0f;  // max of absolute values of A block

    size_t k;
    for (k = 0; k < ElementCount; k += SubBlkLen) {
        const size_t SubBlkElementCount = std::min(ElementCount - k, SubBlkLen);

        float32x4_t a[SubBlkLen / 4]{};
        LoadFloatData<SubBlkLen>(A + k, SubBlkElementCount, a);

        float32x4_t abs_a[SubBlkLen / 4];
        UnrolledLoop<SubBlkLen / 4>([&](size_t i) {
            abs_a[i] = vabsq_f32(a[i]);
        });

        // find amax of SubBlkLen elements
        for (size_t interval = SubBlkLen / 4 / 2; interval > 0; interval /= 2) {
            for (size_t i = 0; i < interval; ++i) {
                abs_a[i] = vmaxq_f32(abs_a[i], abs_a[i + interval]);
            }
        }

        // update existing amax
        amax = std::max(amax, vmaxvq_f32(abs_a[0]));
    }

    constexpr float range_max = (1 << 7) - 1;
    const float scale = amax / range_max;
    const float scale_reciprocal = scale != 0.0f ? 1.0f / scale : 0.0f;

    Q8BlkScale(QuantA) = scale;

    //
    // Compute quantized block values.
    //

    int8_t* QuantAData = Q8BlkData(QuantA);

    for (k = 0; k < ElementCount; k += SubBlkLen) {
        const size_t SubBlkElementCount = std::min(ElementCount - k, SubBlkLen);

        float32x4_t a[SubBlkLen / 4]{};
        LoadFloatData<SubBlkLen>(A + k, SubBlkElementCount, a);

        UnrolledLoop<SubBlkLen / 4>([&](size_t i) {
            a[i] = vmulq_n_f32(a[i], scale_reciprocal);
        });

        int32x4_t a_s32[SubBlkLen / 4];
        UnrolledLoop<SubBlkLen / 4>([&](size_t i) {
            a_s32[i] = vcvtaq_s32_f32(a[i]);
        });

        UnrolledLoop<SubBlkLen / 4>([&](size_t i) {
            QuantAData[k + i * 4 + 0] = static_cast<int8_t>(vgetq_lane_s32(a_s32[i], 0));
            QuantAData[k + i * 4 + 1] = static_cast<int8_t>(vgetq_lane_s32(a_s32[i], 1));
            QuantAData[k + i * 4 + 2] = static_cast<int8_t>(vgetq_lane_s32(a_s32[i], 2));
            QuantAData[k + i * 4 + 3] = static_cast<int8_t>(vgetq_lane_s32(a_s32[i], 3));
        });
    }

    //
    // Zero out any remaining sub-block elements.
    //

    for (; k < BlkLen; k += SubBlkLen) {
        const int8x16_t Zeros = vdupq_n_s8(0);
        UnrolledLoop<SubBlkLen / 16>([&](size_t i) {
            vst1q_s8(QuantAData + k + i * 16, Zeros);
        });
    }
}

}  // namespace

void
QuantizeARow_CompInt8(
    size_t BlkLen,
    const float* A,
    size_t CountK,
    std::byte* QuantA
)
{
    const float* ADataBlkPtr = A;
    std::byte* QuantABlkPtr = QuantA;

    for (size_t k = 0; k < CountK; k += BlkLen) {
        const size_t k_blk_len = std::min(CountK - k, BlkLen);

        QuantizeBlock<16>(BlkLen, ADataBlkPtr, k_blk_len, QuantABlkPtr);

        ADataBlkPtr += BlkLen;
        QuantABlkPtr += Q8BlkSize(BlkLen);
    }
}

namespace
{

//
// The ComputeRxC functions compute an R row by C column tile of the output matrix.
//

template <bool HasZeroPoint>
MLAS_FORCEINLINE void
SQ4BitGemm_CompInt8_Compute4x2_BlkLen16(
    const std::byte* QuantARowPtr,
    const std::byte* QuantBDataColPtr,
    const float* QuantBScaleColPtr,
    const std::byte* QuantBZeroPointColPtr,
    const float* BiasPtr,
    float* SumPtr,
    size_t BlockCountK,
    size_t StrideQuantA,
    size_t StrideQuantBData,
    size_t StrideQuantBScale,
    size_t StrideQuantBZeroPoint,
    size_t ldc
)
{
    constexpr size_t BlkLen = 16;

    const std::byte* QuantAPtr = QuantARowPtr;
    const std::byte* QuantBDataPtr = QuantBDataColPtr;
    const float* QuantBScalePtr = QuantBScaleColPtr;
    const std::byte* QuantBZeroPointPtr = QuantBZeroPointColPtr;

    float32x4_t acc00{}, acc01{}, acc10{}, acc11{}, acc20{}, acc21{}, acc30{}, acc31{};

    for (size_t k_blk_idx = 0; k_blk_idx < BlockCountK; ++k_blk_idx) {
        const std::byte* QuantABlkRow0 = QuantAPtr;
        const std::byte* QuantABlkRow1 = QuantAPtr + StrideQuantA;
        const std::byte* QuantABlkRow2 = QuantAPtr + StrideQuantA * 2;
        const std::byte* QuantABlkRow3 = QuantAPtr + StrideQuantA * 3;

        const float QuantBScaleCol0 = *QuantBScalePtr;
        const float QuantBScaleCol1 = *(QuantBScalePtr + StrideQuantBScale);

        // compute combined scales
        const float scale00 = Q8BlkScale(QuantABlkRow0) * QuantBScaleCol0;
        const float scale01 = Q8BlkScale(QuantABlkRow0) * QuantBScaleCol1;
        const float scale10 = Q8BlkScale(QuantABlkRow1) * QuantBScaleCol0;
        const float scale11 = Q8BlkScale(QuantABlkRow1) * QuantBScaleCol1;
        const float scale20 = Q8BlkScale(QuantABlkRow2) * QuantBScaleCol0;
        const float scale21 = Q8BlkScale(QuantABlkRow2) * QuantBScaleCol1;
        const float scale30 = Q8BlkScale(QuantABlkRow3) * QuantBScaleCol0;
        const float scale31 = Q8BlkScale(QuantABlkRow3) * QuantBScaleCol1;

        // load B zero point
        int8_t bzp_col0;
        int8_t bzp_col1;
        if constexpr (HasZeroPoint) {
            const std::byte QuantBZeroPointByteCol0 = *QuantBZeroPointPtr;
            const std::byte QuantBZeroPointByteCol1 = *(QuantBZeroPointPtr + StrideQuantBZeroPoint);
            if ((k_blk_idx & 1) == 0) {
                bzp_col0 = std::to_integer<int8_t>(QuantBZeroPointByteCol0 & std::byte{0x0F});
                bzp_col1 = std::to_integer<int8_t>(QuantBZeroPointByteCol1 & std::byte{0x0F});
            } else {
                bzp_col0 = std::to_integer<int8_t>(QuantBZeroPointByteCol0 >> 4);
                bzp_col1 = std::to_integer<int8_t>(QuantBZeroPointByteCol1 >> 4);
            }
        } else {
            bzp_col0 = 8;
            bzp_col1 = 8;
        }

        const int8_t* QuantADataPtrRow0 = Q8BlkData(QuantABlkRow0);
        const int8_t* QuantADataPtrRow1 = Q8BlkData(QuantABlkRow1);
        const int8_t* QuantADataPtrRow2 = Q8BlkData(QuantABlkRow2);
        const int8_t* QuantADataPtrRow3 = Q8BlkData(QuantABlkRow3);

        // TODO handling only 16 elements per accumulator at a time here, probably can do better
        {
            // load B
            const uint8x8_t bv_packed_col0 = vld1_u8(reinterpret_cast<const uint8_t*>(QuantBDataPtr));
            const uint8x8_t bv_packed_col1 = vld1_u8(reinterpret_cast<const uint8_t*>(QuantBDataPtr) + StrideQuantBData);

            const uint8x8_t LowMaskU8x8 = vdup_n_u8(0x0F);

            int8x16_t bv_col0 = vreinterpretq_s8_u8(
                vcombine_u8(
                    vand_u8(bv_packed_col0, LowMaskU8x8),
                    vshr_n_u8(bv_packed_col0, 4)
                )
            );
            int8x16_t bv_col1 = vreinterpretq_s8_u8(
                vcombine_u8(
                    vand_u8(bv_packed_col1, LowMaskU8x8),
                    vshr_n_u8(bv_packed_col1, 4)
                )
            );

            // subtract B zero point
            bv_col0 = vsubq_s8(bv_col0, vdupq_n_s8(bzp_col0));
            bv_col1 = vsubq_s8(bv_col1, vdupq_n_s8(bzp_col1));

            // rows 0 and 1 of A
            {
                // load A
                const int8x16_t av_row0 = vld1q_s8(QuantADataPtrRow0 + 0);
                const int8x16_t av_row1 = vld1q_s8(QuantADataPtrRow1 + 0);

                // quantized dot product
                const int32x4_t dot00 = vdotq_s32(int32x4_t{}, av_row0, bv_col0);
                const int32x4_t dot01 = vdotq_s32(int32x4_t{}, av_row0, bv_col1);
                const int32x4_t dot10 = vdotq_s32(int32x4_t{}, av_row1, bv_col0);
                const int32x4_t dot11 = vdotq_s32(int32x4_t{}, av_row1, bv_col1);

                // convert to float
                const float32x4_t dot_f32_00 = vcvtq_f32_s32(dot00);
                const float32x4_t dot_f32_01 = vcvtq_f32_s32(dot01);
                const float32x4_t dot_f32_10 = vcvtq_f32_s32(dot10);
                const float32x4_t dot_f32_11 = vcvtq_f32_s32(dot11);

                // multiply by scale and update accumulator
                acc00 = vfmaq_f32(acc00, dot_f32_00, vdupq_n_f32(scale00));
                acc01 = vfmaq_f32(acc01, dot_f32_01, vdupq_n_f32(scale01));
                acc10 = vfmaq_f32(acc10, dot_f32_10, vdupq_n_f32(scale10));
                acc11 = vfmaq_f32(acc11, dot_f32_11, vdupq_n_f32(scale11));
            }

            // rows 2 and 3 of A
            {
                // load A
                const int8x16_t av_row2 = vld1q_s8(QuantADataPtrRow2 + 0);
                const int8x16_t av_row3 = vld1q_s8(QuantADataPtrRow3 + 0);

                // quantized dot product
                const int32x4_t dot20 = vdotq_s32(int32x4_t{}, av_row2, bv_col0);
                const int32x4_t dot21 = vdotq_s32(int32x4_t{}, av_row2, bv_col1);
                const int32x4_t dot30 = vdotq_s32(int32x4_t{}, av_row3, bv_col0);
                const int32x4_t dot31 = vdotq_s32(int32x4_t{}, av_row3, bv_col1);

                // convert to float
                const float32x4_t dot_f32_20 = vcvtq_f32_s32(dot20);
                const float32x4_t dot_f32_21 = vcvtq_f32_s32(dot21);
                const float32x4_t dot_f32_30 = vcvtq_f32_s32(dot30);
                const float32x4_t dot_f32_31 = vcvtq_f32_s32(dot31);

                // multiply by scale and update accumulator
                acc20 = vfmaq_f32(acc20, dot_f32_20, vdupq_n_f32(scale20));
                acc21 = vfmaq_f32(acc21, dot_f32_21, vdupq_n_f32(scale21));
                acc30 = vfmaq_f32(acc30, dot_f32_30, vdupq_n_f32(scale30));
                acc31 = vfmaq_f32(acc31, dot_f32_31, vdupq_n_f32(scale31));
            }
        }

        // increment block pointers

        QuantAPtr += Q8BlkSize(BlkLen);
        QuantBDataPtr += 8;
        QuantBScalePtr += 1;

        if constexpr (HasZeroPoint) {
            QuantBZeroPointPtr += ((k_blk_idx & 1) == 0) ? 0 : 1;
        }
    }

    SumPtr[ldc * 0 + 0] = vaddvq_f32(acc00);
    SumPtr[ldc * 0 + 1] = vaddvq_f32(acc01);
    SumPtr[ldc * 1 + 0] = vaddvq_f32(acc10);
    SumPtr[ldc * 1 + 1] = vaddvq_f32(acc11);
    SumPtr[ldc * 2 + 0] = vaddvq_f32(acc20);
    SumPtr[ldc * 2 + 1] = vaddvq_f32(acc21);
    SumPtr[ldc * 3 + 0] = vaddvq_f32(acc30);
    SumPtr[ldc * 3 + 1] = vaddvq_f32(acc31);

    if (BiasPtr != nullptr) {
        SumPtr[ldc * 0 + 0] += BiasPtr[0];
        SumPtr[ldc * 0 + 1] += BiasPtr[1];
        SumPtr[ldc * 1 + 0] += BiasPtr[0];
        SumPtr[ldc * 1 + 1] += BiasPtr[1];
        SumPtr[ldc * 2 + 0] += BiasPtr[0];
        SumPtr[ldc * 2 + 1] += BiasPtr[1];
        SumPtr[ldc * 3 + 0] += BiasPtr[0];
        SumPtr[ldc * 3 + 1] += BiasPtr[1];
    }
}

template <bool HasZeroPoint>
MLAS_FORCEINLINE void
SQ4BitGemm_CompInt8_Compute4x2_BlkLenGreaterThan16(
    size_t BlkLen,
    const std::byte* QuantARowPtr,
    const std::byte* QuantBDataColPtr,
    const float* QuantBScaleColPtr,
    const std::byte* QuantBZeroPointColPtr,
    const float* BiasPtr,
    float* SumPtr,
    size_t BlockCountK,
    size_t StrideQuantA,
    size_t StrideQuantBData,
    size_t StrideQuantBScale,
    size_t StrideQuantBZeroPoint,
    size_t ldc
)
{
    // process blocks in 32-element sub-blocks
    const size_t SubBlksPerBlk = BlkLen / 32;

    const std::byte* QuantAPtr = QuantARowPtr;
    const std::byte* QuantBDataPtr = QuantBDataColPtr;
    const float* QuantBScalePtr = QuantBScaleColPtr;
    const std::byte* QuantBZeroPointPtr = QuantBZeroPointColPtr;

    float32x4_t acc00{}, acc01{}, acc10{}, acc11{}, acc20{}, acc21{}, acc30{}, acc31{};

    for (size_t k_blk_idx = 0; k_blk_idx < BlockCountK; ++k_blk_idx) {
        const std::byte* QuantABlkRow0 = QuantAPtr;
        const std::byte* QuantABlkRow1 = QuantAPtr + StrideQuantA;
        const std::byte* QuantABlkRow2 = QuantAPtr + StrideQuantA * 2;
        const std::byte* QuantABlkRow3 = QuantAPtr + StrideQuantA * 3;

        const float QuantBScaleCol0 = *QuantBScalePtr;
        const float QuantBScaleCol1 = *(QuantBScalePtr + StrideQuantBScale);

        // compute combined scales
        const float scale00 = Q8BlkScale(QuantABlkRow0) * QuantBScaleCol0;
        const float scale01 = Q8BlkScale(QuantABlkRow0) * QuantBScaleCol1;
        const float scale10 = Q8BlkScale(QuantABlkRow1) * QuantBScaleCol0;
        const float scale11 = Q8BlkScale(QuantABlkRow1) * QuantBScaleCol1;
        const float scale20 = Q8BlkScale(QuantABlkRow2) * QuantBScaleCol0;
        const float scale21 = Q8BlkScale(QuantABlkRow2) * QuantBScaleCol1;
        const float scale30 = Q8BlkScale(QuantABlkRow3) * QuantBScaleCol0;
        const float scale31 = Q8BlkScale(QuantABlkRow3) * QuantBScaleCol1;

        // load B zero point
        int8_t bzp_col0;
        int8_t bzp_col1;
        if constexpr (HasZeroPoint) {
            const std::byte QuantBZeroPointByteCol0 = *QuantBZeroPointPtr;
            const std::byte QuantBZeroPointByteCol1 = *(QuantBZeroPointPtr + StrideQuantBZeroPoint);
            if ((k_blk_idx & 1) == 0) {
                bzp_col0 = std::to_integer<int8_t>(QuantBZeroPointByteCol0 & std::byte{0x0F});
                bzp_col1 = std::to_integer<int8_t>(QuantBZeroPointByteCol1 & std::byte{0x0F});
            } else {
                bzp_col0 = std::to_integer<int8_t>(QuantBZeroPointByteCol0 >> 4);
                bzp_col1 = std::to_integer<int8_t>(QuantBZeroPointByteCol1 >> 4);
            }
        } else {
            bzp_col0 = 8;
            bzp_col1 = 8;
        }

        const int8_t* QuantADataPtrRow0 = Q8BlkData(QuantABlkRow0);
        const int8_t* QuantADataPtrRow1 = Q8BlkData(QuantABlkRow1);
        const int8_t* QuantADataPtrRow2 = Q8BlkData(QuantABlkRow2);
        const int8_t* QuantADataPtrRow3 = Q8BlkData(QuantABlkRow3);

        for (size_t sub_blk_idx = 0; sub_blk_idx < SubBlksPerBlk; ++sub_blk_idx) {
            // load B
            const uint8x16_t bv_packed_col0 = vld1q_u8(reinterpret_cast<const uint8_t*>(QuantBDataPtr));
            const uint8x16_t bv_packed_col1 = vld1q_u8(reinterpret_cast<const uint8_t*>(QuantBDataPtr) + StrideQuantBData);

            const uint8x16_t LowMaskU8x16 = vdupq_n_u8(0x0F);

            int8x16_t bv_col0_0 = vreinterpretq_s8_u8(vandq_u8(bv_packed_col0, LowMaskU8x16));
            int8x16_t bv_col0_1 = vreinterpretq_s8_u8(vshrq_n_u8(bv_packed_col0, 4));
            int8x16_t bv_col1_0 = vreinterpretq_s8_u8(vandq_u8(bv_packed_col1, LowMaskU8x16));
            int8x16_t bv_col1_1 = vreinterpretq_s8_u8(vshrq_n_u8(bv_packed_col1, 4));

            // subtract B zero point
            bv_col0_0 = vsubq_s8(bv_col0_0, vdupq_n_s8(bzp_col0));
            bv_col0_1 = vsubq_s8(bv_col0_1, vdupq_n_s8(bzp_col0));
            bv_col1_0 = vsubq_s8(bv_col1_0, vdupq_n_s8(bzp_col1));
            bv_col1_1 = vsubq_s8(bv_col1_1, vdupq_n_s8(bzp_col1));

            // rows 0 and 1 of A
            {
                // load A
                const int8x16_t av_row0_0 = vld1q_s8(QuantADataPtrRow0 + 0);
                const int8x16_t av_row0_1 = vld1q_s8(QuantADataPtrRow0 + 16);
                const int8x16_t av_row1_0 = vld1q_s8(QuantADataPtrRow1 + 0);
                const int8x16_t av_row1_1 = vld1q_s8(QuantADataPtrRow1 + 16);

                // quantized dot product
                const int32x4_t dot00 = vdotq_s32(vdotq_s32(int32x4_t{}, av_row0_0, bv_col0_0), av_row0_1, bv_col0_1);
                const int32x4_t dot01 = vdotq_s32(vdotq_s32(int32x4_t{}, av_row0_0, bv_col1_0), av_row0_1, bv_col1_1);
                const int32x4_t dot10 = vdotq_s32(vdotq_s32(int32x4_t{}, av_row1_0, bv_col0_0), av_row1_1, bv_col0_1);
                const int32x4_t dot11 = vdotq_s32(vdotq_s32(int32x4_t{}, av_row1_0, bv_col1_0), av_row1_1, bv_col1_1);

                // convert to float
                const float32x4_t dot_f32_00 = vcvtq_f32_s32(dot00);
                const float32x4_t dot_f32_01 = vcvtq_f32_s32(dot01);
                const float32x4_t dot_f32_10 = vcvtq_f32_s32(dot10);
                const float32x4_t dot_f32_11 = vcvtq_f32_s32(dot11);

                // multiply by scale and update accumulator
                acc00 = vfmaq_f32(acc00, dot_f32_00, vdupq_n_f32(scale00));
                acc01 = vfmaq_f32(acc01, dot_f32_01, vdupq_n_f32(scale01));
                acc10 = vfmaq_f32(acc10, dot_f32_10, vdupq_n_f32(scale10));
                acc11 = vfmaq_f32(acc11, dot_f32_11, vdupq_n_f32(scale11));
            }

            // rows 2 and 3 of A
            {
                // load A
                const int8x16_t av_row2_0 = vld1q_s8(QuantADataPtrRow2 + 0);
                const int8x16_t av_row2_1 = vld1q_s8(QuantADataPtrRow2 + 16);
                const int8x16_t av_row3_0 = vld1q_s8(QuantADataPtrRow3 + 0);
                const int8x16_t av_row3_1 = vld1q_s8(QuantADataPtrRow3 + 16);

                // quantized dot product
                const int32x4_t dot20 = vdotq_s32(vdotq_s32(int32x4_t{}, av_row2_0, bv_col0_0), av_row2_1, bv_col0_1);
                const int32x4_t dot21 = vdotq_s32(vdotq_s32(int32x4_t{}, av_row2_0, bv_col1_0), av_row2_1, bv_col1_1);
                const int32x4_t dot30 = vdotq_s32(vdotq_s32(int32x4_t{}, av_row3_0, bv_col0_0), av_row3_1, bv_col0_1);
                const int32x4_t dot31 = vdotq_s32(vdotq_s32(int32x4_t{}, av_row3_0, bv_col1_0), av_row3_1, bv_col1_1);

                // convert to float
                const float32x4_t dot_f32_20 = vcvtq_f32_s32(dot20);
                const float32x4_t dot_f32_21 = vcvtq_f32_s32(dot21);
                const float32x4_t dot_f32_30 = vcvtq_f32_s32(dot30);
                const float32x4_t dot_f32_31 = vcvtq_f32_s32(dot31);

                // multiply by scale and update accumulator
                acc20 = vfmaq_f32(acc20, dot_f32_20, vdupq_n_f32(scale20));
                acc21 = vfmaq_f32(acc21, dot_f32_21, vdupq_n_f32(scale21));
                acc30 = vfmaq_f32(acc30, dot_f32_30, vdupq_n_f32(scale30));
                acc31 = vfmaq_f32(acc31, dot_f32_31, vdupq_n_f32(scale31));
            }

            // increment block data pointers to next sub-block
            QuantADataPtrRow0 += 32;
            QuantADataPtrRow1 += 32;
            QuantADataPtrRow2 += 32;
            QuantADataPtrRow3 += 32;
            QuantBDataPtr += 16;
        }

        // increment other block pointers

        QuantAPtr += Q8BlkSize(BlkLen);
        QuantBScalePtr += 1;

        if constexpr (HasZeroPoint) {
            QuantBZeroPointPtr += ((k_blk_idx & 1) == 0) ? 0 : 1;
        }
    }

    SumPtr[ldc * 0 + 0] = vaddvq_f32(acc00);
    SumPtr[ldc * 0 + 1] = vaddvq_f32(acc01);
    SumPtr[ldc * 1 + 0] = vaddvq_f32(acc10);
    SumPtr[ldc * 1 + 1] = vaddvq_f32(acc11);
    SumPtr[ldc * 2 + 0] = vaddvq_f32(acc20);
    SumPtr[ldc * 2 + 1] = vaddvq_f32(acc21);
    SumPtr[ldc * 3 + 0] = vaddvq_f32(acc30);
    SumPtr[ldc * 3 + 1] = vaddvq_f32(acc31);

    if (BiasPtr != nullptr) {
        SumPtr[ldc * 0 + 0] += BiasPtr[0];
        SumPtr[ldc * 0 + 1] += BiasPtr[1];
        SumPtr[ldc * 1 + 0] += BiasPtr[0];
        SumPtr[ldc * 1 + 1] += BiasPtr[1];
        SumPtr[ldc * 2 + 0] += BiasPtr[0];
        SumPtr[ldc * 2 + 1] += BiasPtr[1];
        SumPtr[ldc * 3 + 0] += BiasPtr[0];
        SumPtr[ldc * 3 + 1] += BiasPtr[1];
    }
}

template <bool HasZeroPoint>
MLAS_FORCEINLINE void
SQ4BitGemm_CompInt8_Compute1x1_BlkLen16(
    const std::byte* QuantARowPtr,
    const std::byte* QuantBDataColPtr,
    const float* QuantBScaleColPtr,
    const std::byte* QuantBZeroPointColPtr,
    const float* BiasPtr,
    float* SumPtr,
    size_t BlockCountK
)
{
    constexpr size_t BlkLen = 16;

    const std::byte* QuantAPtr = QuantARowPtr;
    const std::byte* QuantBDataPtr = QuantBDataColPtr;
    const float* QuantBScalePtr = QuantBScaleColPtr;
    const std::byte* QuantBZeroPointPtr = QuantBZeroPointColPtr;

    float32x4_t acc0{}, acc1{};

    size_t k_blks_remaining = BlockCountK;
    for (; k_blks_remaining > 1; k_blks_remaining -= 2) {
        const std::byte* QuantABlk0 = QuantAPtr;
        const std::byte* QuantABlk1 = QuantABlk0 + Q8BlkSize(BlkLen);

        // compute combined scale
        const float32x4_t scale0 = vdupq_n_f32(Q8BlkScale(QuantABlk0) * QuantBScalePtr[0]);
        const float32x4_t scale1 = vdupq_n_f32(Q8BlkScale(QuantABlk1) * QuantBScalePtr[1]);

        // load B zero point
        const int8x16_t bzp0 = vdupq_n_s8(
            HasZeroPoint ? std::to_integer<int8_t>((*QuantBZeroPointPtr) & std::byte{0x0F}) : 8
        );
        const int8x16_t bzp1 = vdupq_n_s8(
            HasZeroPoint ? std::to_integer<int8_t>((*QuantBZeroPointPtr) >> 4) : 8
        );

        // load A
        const int8x16_t av0 = vld1q_s8(Q8BlkData(QuantABlk0));
        const int8x16_t av1 = vld1q_s8(Q8BlkData(QuantABlk1));

        // load B
        const uint8x16_t bv_packed01 = vld1q_u8(reinterpret_cast<const uint8_t*>(QuantBDataPtr));

        const uint8x16_t LowMaskU8x16 = vdupq_n_u8(0x0F);

        const uint8x16_t bv_lo01 = vandq_u8(bv_packed01, LowMaskU8x16);
        const uint8x16_t bv_hi01 = vshrq_n_u8(bv_packed01, 4);

        int8x16_t bv0 = vreinterpretq_s8_u8(vcombine_u8(vget_low_u8(bv_lo01), vget_low_u8(bv_hi01)));
        int8x16_t bv1 = vreinterpretq_s8_u8(vcombine_u8(vget_high_u8(bv_lo01), vget_high_u8(bv_hi01)));

        // subtract B zero point
        bv0 = vsubq_s8(bv0, bzp0);
        bv1 = vsubq_s8(bv1, bzp1);

        // quantized dot product
        const int32x4_t dot0 = vdotq_s32(int32x4_t{}, av0, bv0);
        const int32x4_t dot1 = vdotq_s32(int32x4_t{}, av1, bv1);

        // convert to float
        const float32x4_t dot_f32_0 = vcvtq_f32_s32(dot0);
        const float32x4_t dot_f32_1 = vcvtq_f32_s32(dot1);

        // multiply by scale and update accumulator
        acc0 = vfmaq_f32(acc0, dot_f32_0, scale0);
        acc1 = vfmaq_f32(acc1, dot_f32_1, scale1);

        // increment block pointers

        QuantAPtr += Q8BlkSize(BlkLen) * 2;
        QuantBDataPtr += 8 * 2;
        QuantBScalePtr += 2;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointPtr += 1;
        }
    }

    if (k_blks_remaining > 0) {
        const std::byte* QuantABlk0 = QuantAPtr;

        // compute combined scale
        const float32x4_t scale0 = vdupq_n_f32(Q8BlkScale(QuantABlk0) * (*QuantBScalePtr));

        // load B zero point
        const int8x16_t bzp0 = vdupq_n_s8(
            HasZeroPoint ? std::to_integer<int8_t>((*QuantBZeroPointPtr) & std::byte{0x0F}) : 8
        );

        // load A
        const int8x16_t av0 = vld1q_s8(Q8BlkData(QuantABlk0));

        // load B
        const uint8x8_t bv_packed0 = vld1_u8(reinterpret_cast<const uint8_t*>(QuantBDataPtr));

        const uint8x8_t LowMaskU8x8 = vdup_n_u8(0x0F);

        const uint8x8_t bv_lo0 = vand_u8(bv_packed0, LowMaskU8x8);
        const uint8x8_t bv_hi0 = vshr_n_u8(bv_packed0, 4);

        int8x16_t bv0 = vreinterpretq_s8_u8(vcombine_u8(bv_lo0, bv_hi0));

        // subtract B zero point
        bv0 = vsubq_s8(bv0, bzp0);

        // quantized dot product
        const int32x4_t dot0 = vdotq_s32(int32x4_t{}, av0, bv0);

        // convert to float
        const float32x4_t dot_f32_0 = vcvtq_f32_s32(dot0);

        // multiply by scale and update accumulator
        acc0 = vfmaq_f32(acc0, dot_f32_0, scale0);
    }

    *SumPtr = vaddvq_f32(acc0) + vaddvq_f32(acc1);
    if (BiasPtr) {
        *SumPtr += *BiasPtr;
    }
}

template <bool HasZeroPoint>
MLAS_FORCEINLINE void
SQ4BitGemm_CompInt8_Compute1x1_BlkLen32(
    const std::byte* QuantARowPtr,
    const std::byte* QuantBDataColPtr,
    const float* QuantBScaleColPtr,
    const std::byte* QuantBZeroPointColPtr,
    const float* BiasPtr,
    float* SumPtr,
    size_t BlockCountK
)
{
    constexpr size_t BlkLen = 32;

    const uint8x16_t LowMaskU8x16 = vdupq_n_u8(0x0F);

    const std::byte* QuantAPtr = QuantARowPtr;
    const std::byte* QuantBDataPtr = QuantBDataColPtr;
    const float* QuantBScalePtr = QuantBScaleColPtr;
    const std::byte* QuantBZeroPointPtr = QuantBZeroPointColPtr;

    float32x4_t acc0{}, acc1{};

    size_t k_blks_remaining = BlockCountK;
    for (; k_blks_remaining > 1; k_blks_remaining -= 2) {
        const std::byte* QuantABlk0 = QuantAPtr;
        const std::byte* QuantABlk1 = QuantABlk0 + Q8BlkSize(BlkLen);

        // compute combined scale
        const float32x4_t scale0 = vdupq_n_f32(Q8BlkScale(QuantABlk0) * QuantBScalePtr[0]);
        const float32x4_t scale1 = vdupq_n_f32(Q8BlkScale(QuantABlk1) * QuantBScalePtr[1]);

        // load B zero point
        const int8x16_t bzp0 = vdupq_n_s8(
            HasZeroPoint ? std::to_integer<int8_t>((*QuantBZeroPointPtr) & std::byte{0x0F}) : 8
        );
        const int8x16_t bzp1 = vdupq_n_s8(
            HasZeroPoint ? std::to_integer<int8_t>((*QuantBZeroPointPtr) >> 4) : 8
        );

        // load A
        const int8x16_t av_lo0 = vld1q_s8(Q8BlkData(QuantABlk0));
        const int8x16_t av_hi0 = vld1q_s8(Q8BlkData(QuantABlk0) + 16);
        const int8x16_t av_lo1 = vld1q_s8(Q8BlkData(QuantABlk1));
        const int8x16_t av_hi1 = vld1q_s8(Q8BlkData(QuantABlk1) + 16);

        // load B
        const uint8x16_t bv_packed0 = vld1q_u8(reinterpret_cast<const uint8_t*>(QuantBDataPtr));
        const uint8x16_t bv_packed1 = vld1q_u8(reinterpret_cast<const uint8_t*>(QuantBDataPtr) + 16);

        int8x16_t bv_lo0 = vreinterpretq_s8_u8(vandq_u8(bv_packed0, LowMaskU8x16));
        int8x16_t bv_hi0 = vreinterpretq_s8_u8(vshrq_n_u8(bv_packed0, 4));
        int8x16_t bv_lo1 = vreinterpretq_s8_u8(vandq_u8(bv_packed1, LowMaskU8x16));
        int8x16_t bv_hi1 = vreinterpretq_s8_u8(vshrq_n_u8(bv_packed1, 4));

        // subtract B zero point
        bv_lo0 = vsubq_s8(bv_lo0, bzp0);
        bv_hi0 = vsubq_s8(bv_hi0, bzp0);
        bv_lo1 = vsubq_s8(bv_lo1, bzp1);
        bv_hi1 = vsubq_s8(bv_hi1, bzp1);

        // quantized dot product
        const int32x4_t dot0 = vdotq_s32(vdotq_s32(int32x4_t{}, av_lo0, bv_lo0), av_hi0, bv_hi0);
        const int32x4_t dot1 = vdotq_s32(vdotq_s32(int32x4_t{}, av_lo1, bv_lo1), av_hi1, bv_hi1);

        // convert to float
        const float32x4_t dot_f32_0 = vcvtq_f32_s32(dot0);
        const float32x4_t dot_f32_1 = vcvtq_f32_s32(dot1);

        // multiply by scale and update accumulator
        acc0 = vfmaq_f32(acc0, dot_f32_0, scale0);
        acc1 = vfmaq_f32(acc1, dot_f32_1, scale1);

        // increment block pointers

        QuantAPtr += Q8BlkSize(BlkLen) * 2;
        QuantBDataPtr += 16 * 2;
        QuantBScalePtr += 2;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointPtr += 1;
        }
    }

    if (k_blks_remaining > 0) {
        const std::byte* QuantABlk0 = QuantAPtr;

        // compute combined scale
        const float32x4_t scale0 = vdupq_n_f32(Q8BlkScale(QuantABlk0) * (*QuantBScalePtr));

        // load B zero point
        const int8x16_t bzp0 = vdupq_n_s8(
            HasZeroPoint ? std::to_integer<int8_t>((*QuantBZeroPointPtr) & std::byte{0x0F}) : 8
        );

        // load A
        const int8x16_t av_lo0 = vld1q_s8(Q8BlkData(QuantABlk0));
        const int8x16_t av_hi0 = vld1q_s8(Q8BlkData(QuantABlk0) + 16);

        // load B
        const uint8x16_t bv_packed0 = vld1q_u8(reinterpret_cast<const uint8_t*>(QuantBDataPtr));

        int8x16_t bv_lo0 = vreinterpretq_s8_u8(vandq_u8(bv_packed0, LowMaskU8x16));
        int8x16_t bv_hi0 = vreinterpretq_s8_u8(vshrq_n_u8(bv_packed0, 4));

        // subtract B zero point
        bv_lo0 = vsubq_s8(bv_lo0, bzp0);
        bv_hi0 = vsubq_s8(bv_hi0, bzp0);

        // quantized dot product
        const int32x4_t dot0 = vdotq_s32(vdotq_s32(int32x4_t{}, av_lo0, bv_lo0), av_hi0, bv_hi0);

        // convert to float
        const float32x4_t dot_f32_0 = vcvtq_f32_s32(dot0);

        // multiply by scale and update accumulator
        acc0 = vfmaq_f32(acc0, dot_f32_0, scale0);
    }

    *SumPtr = vaddvq_f32(acc0) + vaddvq_f32(acc1);
    if (BiasPtr) {
        *SumPtr += *BiasPtr;
    }
}

template <bool HasZeroPoint>
MLAS_FORCEINLINE void
SQ4BitGemm_CompInt8_Compute1x1_BlkLenGreaterThan32(
    size_t BlkLen,
    const std::byte* QuantARowPtr,
    const std::byte* QuantBDataColPtr,
    const float* QuantBScaleColPtr,
    const std::byte* QuantBZeroPointColPtr,
    const float* BiasPtr,
    float* SumPtr,
    size_t BlockCountK
)
{
    const uint8x16_t LowMaskU8x16 = vdupq_n_u8(0x0F);

    // process blocks in 32-element sub-blocks
    const size_t SubBlksPerBlk = BlkLen / 32;

    const std::byte* QuantAPtr = QuantARowPtr;
    const std::byte* QuantBDataPtr = QuantBDataColPtr;
    const float* QuantBScalePtr = QuantBScaleColPtr;
    const std::byte* QuantBZeroPointPtr = QuantBZeroPointColPtr;

    float32x4_t acc0{}, acc1{};

    for (size_t k_blk_idx = 0; k_blk_idx < BlockCountK; ++k_blk_idx) {
        const std::byte* QuantABlk0 = QuantAPtr;

        // compute combined scale
        const float32x4_t scale = vdupq_n_f32(Q8BlkScale(QuantABlk0) * QuantBScalePtr[0]);

        // load B zero point
        const int8x16_t bzp = [&]() -> int8x16_t {
            if constexpr (HasZeroPoint) {
                return vdupq_n_s8(
                    ((k_blk_idx & 1) == 0) ? std::to_integer<int8_t>((*QuantBZeroPointPtr) & std::byte{0x0F})
                                           : std::to_integer<int8_t>((*QuantBZeroPointPtr) >> 4)
                );
            } else {
                return vdupq_n_s8(8);
            }
        }();

        const int8_t* QuantADataPtr = Q8BlkData(QuantAPtr);

        for (size_t sub_blk_idx = 0; sub_blk_idx < SubBlksPerBlk; sub_blk_idx += 2) {
            // load A
            const int8x16_t av0 = vld1q_s8(QuantADataPtr + 0);
            const int8x16_t av1 = vld1q_s8(QuantADataPtr + 16);
            const int8x16_t av2 = vld1q_s8(QuantADataPtr + 32);
            const int8x16_t av3 = vld1q_s8(QuantADataPtr + 48);

            // load B
            const uint8x16_t bv_packed0 = vld1q_u8(reinterpret_cast<const uint8_t*>(QuantBDataPtr));
            const uint8x16_t bv_packed1 = vld1q_u8(reinterpret_cast<const uint8_t*>(QuantBDataPtr) + 16);

            int8x16_t bv0 = vreinterpretq_s8_u8(vandq_u8(bv_packed0, LowMaskU8x16));
            int8x16_t bv1 = vreinterpretq_s8_u8(vshrq_n_u8(bv_packed0, 4));
            int8x16_t bv2 = vreinterpretq_s8_u8(vandq_u8(bv_packed1, LowMaskU8x16));
            int8x16_t bv3 = vreinterpretq_s8_u8(vshrq_n_u8(bv_packed1, 4));

            // subtract B zero point
            bv0 = vsubq_s8(bv0, bzp);
            bv1 = vsubq_s8(bv1, bzp);
            bv2 = vsubq_s8(bv2, bzp);
            bv3 = vsubq_s8(bv3, bzp);

            // quantized dot product
            const int32x4_t dot0 = vdotq_s32(vdotq_s32(int32x4_t{}, av0, bv0), av1, bv1);
            const int32x4_t dot1 = vdotq_s32(vdotq_s32(int32x4_t{}, av2, bv2), av3, bv3);

            // convert to float
            const float32x4_t dot_f32_0 = vcvtq_f32_s32(dot0);
            const float32x4_t dot_f32_1 = vcvtq_f32_s32(dot1);

            // multiply by scale and update accumulator
            acc0 = vfmaq_f32(acc0, dot_f32_0, scale);
            acc1 = vfmaq_f32(acc1, dot_f32_1, scale);

            // increment block data pointers to next sub-block
            QuantADataPtr += 16 * 4;
            QuantBDataPtr += 16 * 2;
        }

        // increment block pointers

        QuantAPtr += Q8BlkSize(BlkLen);
        QuantBScalePtr += 1;

        if constexpr (HasZeroPoint) {
            QuantBZeroPointPtr += ((k_blk_idx & 1) == 0) ? 0 : 1;
        }
    }

    *SumPtr = vaddvq_f32(acc0) + vaddvq_f32(acc1);
    if (BiasPtr) {
        *SumPtr += *BiasPtr;
    }
}

template <size_t NumCols, bool HasZeroPoint>
MLAS_FORCEINLINE void
AdvanceColPtrs(
    size_t StrideQuantBData,
    size_t StrideQuantBScale,
    size_t StrideQuantBZeroPoint,
    const std::byte*& QuantBDataColPtr,
    const float*& QuantBScaleColPtr,
    const std::byte*& QuantBZeroPointColPtr,
    const float*& BiasPtr,
    float*& SumPtr
)
{
    QuantBDataColPtr += NumCols * StrideQuantBData;
    QuantBScaleColPtr += NumCols * StrideQuantBScale;
    if constexpr (HasZeroPoint) {
        QuantBZeroPointColPtr += NumCols * StrideQuantBZeroPoint;
    }

    BiasPtr += BiasPtr != nullptr ? NumCols : 0;
    SumPtr += NumCols;
}

template <size_t NumRows>
MLAS_FORCEINLINE void
AdvanceRowPtrs(
    size_t StrideQuantA,
    size_t ldc,
    const std::byte*& QuantARowPtr,
    float*& SumRowPtr
)
{
    QuantARowPtr += NumRows * StrideQuantA;
    SumRowPtr += NumRows * ldc;
}

template <bool HasZeroPoint>
void
SQ4BitGemmKernel_CompInt8_BlkLen16(
    const std::byte* QuantA,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    size_t ldc,
    const float* Bias
)
{
    constexpr size_t BlkBitWidth = 4;
    constexpr size_t BlkLen = 16;

    const size_t StrideQuantA = BlockCountK * Q8BlkSize(BlkLen);

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t StrideQuantBScale = BlockCountK;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);

    const std::byte* QuantARowPtr = QuantA;

    float* SumRowPtr = C;

    size_t m_remaining = CountM;
    while (m_remaining > 3) {
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

        const float* BiasPtr = Bias;

        float* SumPtr = SumRowPtr;

        size_t n_remaining = CountN;
        while (n_remaining > 1) {
            // Compute 4x2 tiles of output
            SQ4BitGemm_CompInt8_Compute4x2_BlkLen16<HasZeroPoint>(
                QuantARowPtr,
                QuantBDataColPtr,
                QuantBScaleColPtr,
                QuantBZeroPointColPtr,
                BiasPtr,
                SumPtr,
                BlockCountK,
                StrideQuantA,
                StrideQuantBData,
                StrideQuantBScale,
                StrideQuantBZeroPoint,
                ldc
            );

            // Move to next 2 columns
            AdvanceColPtrs<2, HasZeroPoint>(
                StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
                QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, BiasPtr, SumPtr
            );

            n_remaining -= 2;
        }

        if (n_remaining > 0) {
            // Compute last 4x1 tile of output
            for (size_t i = 0; i < 4; ++i) {
                SQ4BitGemm_CompInt8_Compute1x1_BlkLen16<HasZeroPoint>(
                    QuantARowPtr + StrideQuantA * i,
                    QuantBDataColPtr,
                    QuantBScaleColPtr,
                    QuantBZeroPointColPtr,
                    BiasPtr,
                    SumPtr + ldc * i,
                    BlockCountK
                );
            }
        }

        // Move to next 4 rows
        AdvanceRowPtrs<4>(
            StrideQuantA, ldc,
            QuantARowPtr, SumRowPtr
        );

        m_remaining -= 4;
    }

    while (m_remaining > 0) {
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

        const float* BiasPtr = Bias;

        float* SumPtr = SumRowPtr;

        size_t n_remaining = CountN;
        while (n_remaining > 0) {
            // Compute 1x1 tiles of output
            SQ4BitGemm_CompInt8_Compute1x1_BlkLen16<HasZeroPoint>(
                QuantARowPtr,
                QuantBDataColPtr,
                QuantBScaleColPtr,
                QuantBZeroPointColPtr,
                BiasPtr,
                SumPtr,
                BlockCountK
            );

            // Move to next column
            AdvanceColPtrs<1, HasZeroPoint>(
                StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
                QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, BiasPtr, SumPtr
            );

            n_remaining -= 1;
        }

        // Move to next row
        AdvanceRowPtrs<1>(
            StrideQuantA, ldc,
            QuantARowPtr, SumRowPtr
        );

        m_remaining -= 1;
    }
}

template <bool HasZeroPoint>
void
SQ4BitGemmKernel_CompInt8_BlkLen32(
    const std::byte* QuantA,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    size_t ldc,
    const float* Bias
)
{
    constexpr size_t BlkBitWidth = 4;
    constexpr size_t BlkLen = 32;

    const size_t StrideQuantA = BlockCountK * Q8BlkSize(BlkLen);

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t StrideQuantBScale = BlockCountK;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);

    const std::byte* QuantARowPtr = QuantA;

    float* SumRowPtr = C;

    size_t m_remaining = CountM;
    while (m_remaining > 3) {
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

        const float* BiasPtr = Bias;

        float* SumPtr = SumRowPtr;

        size_t n_remaining = CountN;
        while (n_remaining > 1) {
            // Compute 4x2 tiles of output
            SQ4BitGemm_CompInt8_Compute4x2_BlkLenGreaterThan16<HasZeroPoint>(
                BlkLen,
                QuantARowPtr,
                QuantBDataColPtr,
                QuantBScaleColPtr,
                QuantBZeroPointColPtr,
                BiasPtr,
                SumPtr,
                BlockCountK,
                StrideQuantA,
                StrideQuantBData,
                StrideQuantBScale,
                StrideQuantBZeroPoint,
                ldc
            );

            // Move to next 2 columns
            AdvanceColPtrs<2, HasZeroPoint>(
                StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
                QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, BiasPtr, SumPtr
            );

            n_remaining -= 2;
        }

        if (n_remaining > 0) {
            // Compute last 4x1 tile of output
            for (size_t i = 0; i < 4; ++i) {
                SQ4BitGemm_CompInt8_Compute1x1_BlkLen32<HasZeroPoint>(
                    QuantARowPtr + StrideQuantA * i,
                    QuantBDataColPtr,
                    QuantBScaleColPtr,
                    QuantBZeroPointColPtr,
                    BiasPtr,
                    SumPtr + ldc * i,
                    BlockCountK
                );
            }
        }

        // Move to next 4 rows
        AdvanceRowPtrs<4>(
            StrideQuantA, ldc,
            QuantARowPtr, SumRowPtr
        );

        m_remaining -= 4;
    }

    while (m_remaining > 0) {
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

        const float* BiasPtr = Bias;

        float* SumPtr = SumRowPtr;

        size_t n_remaining = CountN;
        while (n_remaining > 0) {
            // Compute 1x1 tiles of output
            SQ4BitGemm_CompInt8_Compute1x1_BlkLen32<HasZeroPoint>(
                QuantARowPtr,
                QuantBDataColPtr,
                QuantBScaleColPtr,
                QuantBZeroPointColPtr,
                BiasPtr,
                SumPtr,
                BlockCountK
            );

            // Move to next column
            AdvanceColPtrs<1, HasZeroPoint>(
                StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
                QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, BiasPtr, SumPtr
            );

            n_remaining -= 1;
        }

        // Move to next row
        AdvanceRowPtrs<1>(
            StrideQuantA, ldc,
            QuantARowPtr, SumRowPtr
        );

        m_remaining -= 1;
    }
}

template <bool HasZeroPoint>
void
SQ4BitGemmKernel_CompInt8_BlkLenGreaterThan32(
    size_t BlkLen,
    const std::byte* QuantA,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    size_t ldc,
    const float* Bias
)
{
    constexpr size_t BlkBitWidth = 4;

    const size_t StrideQuantA = BlockCountK * Q8BlkSize(BlkLen);

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t StrideQuantBScale = BlockCountK;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);

    const std::byte* QuantARowPtr = QuantA;

    float* SumRowPtr = C;

    size_t m_remaining = CountM;
    while (m_remaining > 3) {
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

        const float* BiasPtr = Bias;

        float* SumPtr = SumRowPtr;

        size_t n_remaining = CountN;
        while (n_remaining > 1) {
            // Compute 4x2 tiles of output
            SQ4BitGemm_CompInt8_Compute4x2_BlkLenGreaterThan16<HasZeroPoint>(
                BlkLen,
                QuantARowPtr,
                QuantBDataColPtr,
                QuantBScaleColPtr,
                QuantBZeroPointColPtr,
                BiasPtr,
                SumPtr,
                BlockCountK,
                StrideQuantA,
                StrideQuantBData,
                StrideQuantBScale,
                StrideQuantBZeroPoint,
                ldc
            );

            // Move to next 2 columns
            AdvanceColPtrs<2, HasZeroPoint>(
                StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
                QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, BiasPtr, SumPtr
            );

            n_remaining -= 2;
        }

        if (n_remaining > 0) {
            // Compute last 4x1 tile of output
            for (size_t i = 0; i < 4; ++i) {
                SQ4BitGemm_CompInt8_Compute1x1_BlkLenGreaterThan32<HasZeroPoint>(
                    BlkLen,
                    QuantARowPtr + StrideQuantA * i,
                    QuantBDataColPtr,
                    QuantBScaleColPtr,
                    QuantBZeroPointColPtr,
                    BiasPtr,
                    SumPtr + ldc * i,
                    BlockCountK
                );
            }
        }

        // Move to next 4 rows
        AdvanceRowPtrs<4>(
            StrideQuantA, ldc,
            QuantARowPtr, SumRowPtr
        );

        m_remaining -= 4;
    }

    while (m_remaining > 0) {
        const std::byte* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

        const float* BiasPtr = Bias;

        float* SumPtr = SumRowPtr;

        size_t n_remaining = CountN;
        while (n_remaining > 0) {
            // Compute 1x1 tiles of output
            SQ4BitGemm_CompInt8_Compute1x1_BlkLenGreaterThan32<HasZeroPoint>(
                BlkLen,
                QuantARowPtr,
                QuantBDataColPtr,
                QuantBScaleColPtr,
                QuantBZeroPointColPtr,
                BiasPtr,
                SumPtr,
                BlockCountK
            );

            // Move to next column
            AdvanceColPtrs<1, HasZeroPoint>(
                StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
                QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, BiasPtr, SumPtr
            );

            n_remaining -= 1;
        }

        // Move to next row
        AdvanceRowPtrs<1>(
            StrideQuantA, ldc,
            QuantARowPtr, SumRowPtr
        );

        m_remaining -= 1;
    }
}

template <bool HasZeroPoint>
void
SQ4BitGemmKernel_CompInt8_DispatchOnBlkLen(
    size_t BlkLen,
    const std::byte* QuantA,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    size_t ldc,
    const float* Bias
)
{
    if (BlkLen == 16) {
        SQ4BitGemmKernel_CompInt8_BlkLen16<HasZeroPoint>(
            QuantA,
            QuantBData,
            QuantBScale,
            QuantBZeroPoint,
            C,
            CountM,
            CountN,
            BlockCountK,
            ldc,
            Bias
        );
    } else if (BlkLen == 32) {
        SQ4BitGemmKernel_CompInt8_BlkLen32<HasZeroPoint>(
            QuantA,
            QuantBData,
            QuantBScale,
            QuantBZeroPoint,
            C,
            CountM,
            CountN,
            BlockCountK,
            ldc,
            Bias
        );
    } else {
        SQ4BitGemmKernel_CompInt8_BlkLenGreaterThan32<HasZeroPoint>(
            BlkLen,
            QuantA,
            QuantBData,
            QuantBScale,
            QuantBZeroPoint,
            C,
            CountM,
            CountN,
            BlockCountK,
            ldc,
            Bias
        );
    }
}

}  // namespace

size_t
SQ4BitGemmKernel_CompInt8(
    size_t BlkLen,
    const std::byte* QuantA,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t /*CountK*/,
    size_t BlockCountK,
    size_t ldc,
    const float* Bias
)
{
    if (QuantBZeroPoint != nullptr) {
        constexpr bool HasZeroPoint = true;
        SQ4BitGemmKernel_CompInt8_DispatchOnBlkLen<HasZeroPoint>(
            BlkLen,
            QuantA,
            QuantBData,
            QuantBScale,
            QuantBZeroPoint,
            C,
            CountM,
            CountN,
            BlockCountK,
            ldc,
            Bias
        );
    } else {
        constexpr bool HasZeroPoint = false;
        SQ4BitGemmKernel_CompInt8_DispatchOnBlkLen<HasZeroPoint>(
            BlkLen,
            QuantA,
            QuantBData,
            QuantBScale,
            QuantBZeroPoint,
            C,
            CountM,
            CountN,
            BlockCountK,
            ldc,
            Bias
        );
    }

    return CountM;
}

}  // namespace sqnbitgemm_neon
