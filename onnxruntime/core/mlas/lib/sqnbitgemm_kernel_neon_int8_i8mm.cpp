/*++

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_neon_int8_i8mm.cpp

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication kernels for ARM NEON specific to
    input type T1 as float32 and
    MLAS_QNBIT_GEMM_COMPUTE_TYPE SQNBIT_CompInt8
    using i8mm instructions.

--*/

#include "qnbitgemm.h"
#include "qnbitgemm_kernel_neon.h"

namespace sqnbitgemm_neon
{

MLAS_FORCEINLINE void
Q8Int8GemmR2xC8I8MM(
    const size_t BlkLen,
    const int8_t* QuantA,
    const float* QuantAScale,
    const uint8_t* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc
)
{
    constexpr size_t NCols4 = 4;
    constexpr size_t NCols8 = 8;
    constexpr size_t NRows2 = 2;
    constexpr size_t KStep16 = 16;

    const size_t lda = BlockCountK * BlkLen;
    const size_t StrideQuantBDataCol8 = BlockCountK * BlkLen * NCols8;

    assert(CountM % NRows2 == 0);
    assert(CountN % NCols8 == 0);

    for (size_t m = 0; m < CountM; m += NRows2) {
        const uint8_t* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const float* BiasPtr = Bias;
        auto* SumPtr = C + m * ldc;

        for (size_t n = 0; n < CountN; n += NCols8) {
            const int8_t* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;

            const uint8_t* QuantBDataPtr = QuantBDataColPtr;
            const float* QuantBScalePtr = QuantBScaleColPtr;

            float32x4_t accf0_03 = vdupq_n_f32(0.0f);
            float32x4_t accf0_47 = vdupq_n_f32(0.0f);
            float32x4_t accf1_03 = vdupq_n_f32(0.0f);
            float32x4_t accf1_47 = vdupq_n_f32(0.0f);

            for (size_t i = 0; i < BlockCountK; ++i) {
                const float scaleA0 = *QuantAScalePtr;
                const float scaleA1 = *(QuantAScalePtr + BlockCountK);
                const float32x4_t scaleB03 = vld1q_f32(QuantBScalePtr);
                const float32x4_t scaleB47 = vld1q_f32(QuantBScalePtr + NCols4);

                const float32x4_t scaleA0B03 = vmulq_n_f32(scaleB03, scaleA0);
                const float32x4_t scaleA0B47 = vmulq_n_f32(scaleB47, scaleA0);
                const float32x4_t scaleA1B03 = vmulq_n_f32(scaleB03, scaleA1);
                const float32x4_t scaleA1B47 = vmulq_n_f32(scaleB47, scaleA1);

                int32x4_t acc0_03 = vdupq_n_s32(0);
                int32x4_t acc0_47 = vdupq_n_s32(0);
                int32x4_t acc1_03 = vdupq_n_s32(0);
                int32x4_t acc1_47 = vdupq_n_s32(0);

                for (size_t k = 0; k < BlkLen; k += KStep16) {
                    const int8x16_t av0_16_i8 = vld1q_s8(QuantAPtr);
                    const int8x16_t av1_16_i8 = vld1q_s8(QuantAPtr + lda);

                    uint8x16_t bv_packed_0_03 = vld1q_u8(QuantBDataPtr);
                    uint8x16_t bv_packed_0_47 = vld1q_u8(QuantBDataPtr + 16);
                    uint8x16_t bv_packed_1_03 = vld1q_u8(QuantBDataPtr + 32);
                    uint8x16_t bv_packed_1_47 = vld1q_u8(QuantBDataPtr + 48);
                    uint8x16_t bv_packed_2_03 = vld1q_u8(QuantBDataPtr + 64);
                    uint8x16_t bv_packed_2_47 = vld1q_u8(QuantBDataPtr + 80);
                    uint8x16_t bv_packed_3_03 = vld1q_u8(QuantBDataPtr + 96);
                    uint8x16_t bv_packed_3_47 = vld1q_u8(QuantBDataPtr + 112);

                    acc0_03 = vusdotq_laneq_s32(acc0_03, bv_packed_0_03, av0_16_i8, 0);
                    acc0_03 = vusdotq_laneq_s32(acc0_03, bv_packed_1_03, av0_16_i8, 1);
                    acc0_03 = vusdotq_laneq_s32(acc0_03, bv_packed_2_03, av0_16_i8, 2);
                    acc0_03 = vusdotq_laneq_s32(acc0_03, bv_packed_3_03, av0_16_i8, 3);

                    acc0_47 = vusdotq_laneq_s32(acc0_47, bv_packed_0_47, av0_16_i8, 0);
                    acc0_47 = vusdotq_laneq_s32(acc0_47, bv_packed_1_47, av0_16_i8, 1);
                    acc0_47 = vusdotq_laneq_s32(acc0_47, bv_packed_2_47, av0_16_i8, 2);
                    acc0_47 = vusdotq_laneq_s32(acc0_47, bv_packed_3_47, av0_16_i8, 3);

                    acc1_03 = vusdotq_laneq_s32(acc1_03, bv_packed_0_03, av1_16_i8, 0);
                    acc1_03 = vusdotq_laneq_s32(acc1_03, bv_packed_1_03, av1_16_i8, 1);
                    acc1_03 = vusdotq_laneq_s32(acc1_03, bv_packed_2_03, av1_16_i8, 2);
                    acc1_03 = vusdotq_laneq_s32(acc1_03, bv_packed_3_03, av1_16_i8, 3);

                    acc1_47 = vusdotq_laneq_s32(acc1_47, bv_packed_0_47, av1_16_i8, 0);
                    acc1_47 = vusdotq_laneq_s32(acc1_47, bv_packed_1_47, av1_16_i8, 1);
                    acc1_47 = vusdotq_laneq_s32(acc1_47, bv_packed_2_47, av1_16_i8, 2);
                    acc1_47 = vusdotq_laneq_s32(acc1_47, bv_packed_3_47, av1_16_i8, 3);

                    QuantAPtr += KStep16;
                    QuantBDataPtr += NCols8 * KStep16;
                }

                accf0_03 = vmlaq_f32(accf0_03, scaleA0B03, vcvtq_f32_s32(acc0_03));
                accf0_47 = vmlaq_f32(accf0_47, scaleA0B47, vcvtq_f32_s32(acc0_47));
                accf1_03 = vmlaq_f32(accf1_03, scaleA1B03, vcvtq_f32_s32(acc1_03));
                accf1_47 = vmlaq_f32(accf1_47, scaleA1B47, vcvtq_f32_s32(acc1_47));

                ++QuantAScalePtr;
                QuantBScalePtr += NCols8;
            }

            if (BiasPtr != nullptr) {
                const float32x4_t bias_4_f32_03 = vld1q_f32(BiasPtr);
                const float32x4_t bias_4_f32_47 = vld1q_f32(BiasPtr + 4);

                accf0_03 = vaddq_f32(accf0_03, bias_4_f32_03);
                accf0_47 = vaddq_f32(accf0_47, bias_4_f32_47);
                accf1_03 = vaddq_f32(accf1_03, bias_4_f32_03);
                accf1_47 = vaddq_f32(accf1_47, bias_4_f32_47);
            }

            vst1q_f32(SumPtr, accf0_03);
            vst1q_f32(SumPtr + 4, accf0_47);
            vst1q_f32(SumPtr + ldc, accf1_03);
            vst1q_f32(SumPtr + ldc + 4, accf1_47);

            // move to next NCols columns
            QuantBDataColPtr += StrideQuantBDataCol8;
            QuantBScaleColPtr += NCols8 * BlockCountK;

            BiasPtr += BiasPtr != nullptr ? NCols8 : 0;
            SumPtr += NCols8;
        }
    }
}

MLAS_FORCEINLINE void
Q8Int8GemmR1xC8I8MM(
    const size_t BlkLen,
    const int8_t* QuantA,
    const float* QuantAScale,
    const uint8_t* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc
)
{
    constexpr size_t NCols4 = 4;
    constexpr size_t NCols8 = 8;
    constexpr size_t KStep16 = 16;

    const size_t lda = BlockCountK * BlkLen;
    const size_t StrideQuantBDataCol8 = BlockCountK * BlkLen * NCols8;

    assert(CountN % NCols8 == 0);

    for (size_t m = 0; m < CountM; ++m) {
        const uint8_t* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const float* BiasPtr = Bias;
        auto* SumPtr = C + m * ldc;

        for (size_t n = 0; n < CountN; n += NCols8) {
            const int8_t* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;

            const uint8_t* QuantBDataPtr = QuantBDataColPtr;
            const float* QuantBScalePtr = QuantBScaleColPtr;

            float32x4_t accf0_03 = vdupq_n_f32(0.0f);
            float32x4_t accf0_47 = vdupq_n_f32(0.0f);

            for (size_t i = 0; i < BlockCountK; ++i) {
                const float scaleA0 = *QuantAScalePtr;
                const float32x4_t scaleB03 = vld1q_f32(QuantBScalePtr);
                const float32x4_t scaleB47 = vld1q_f32(QuantBScalePtr + NCols4);

                const float32x4_t scaleA0B03 = vmulq_n_f32(scaleB03, scaleA0);
                const float32x4_t scaleA0B47 = vmulq_n_f32(scaleB47, scaleA0);

                int32x4_t acc0_03 = vdupq_n_s32(0);
                int32x4_t acc0_47 = vdupq_n_s32(0);

                for (size_t k = 0; k < BlkLen; k += KStep16) {
                    const int8x16_t av0_16_i8 = vld1q_s8(QuantAPtr);

                    uint8x16_t bv_packed_0_03 = vld1q_u8(QuantBDataPtr);
                    uint8x16_t bv_packed_0_47 = vld1q_u8(QuantBDataPtr + 16);
                    uint8x16_t bv_packed_1_03 = vld1q_u8(QuantBDataPtr + 32);
                    uint8x16_t bv_packed_1_47 = vld1q_u8(QuantBDataPtr + 48);
                    uint8x16_t bv_packed_2_03 = vld1q_u8(QuantBDataPtr + 64);
                    uint8x16_t bv_packed_2_47 = vld1q_u8(QuantBDataPtr + 80);
                    uint8x16_t bv_packed_3_03 = vld1q_u8(QuantBDataPtr + 96);
                    uint8x16_t bv_packed_3_47 = vld1q_u8(QuantBDataPtr + 112);

                    acc0_03 = vusdotq_laneq_s32(acc0_03, bv_packed_0_03, av0_16_i8, 0);
                    acc0_03 = vusdotq_laneq_s32(acc0_03, bv_packed_1_03, av0_16_i8, 1);
                    acc0_03 = vusdotq_laneq_s32(acc0_03, bv_packed_2_03, av0_16_i8, 2);
                    acc0_03 = vusdotq_laneq_s32(acc0_03, bv_packed_3_03, av0_16_i8, 3);

                    acc0_47 = vusdotq_laneq_s32(acc0_47, bv_packed_0_47, av0_16_i8, 0);
                    acc0_47 = vusdotq_laneq_s32(acc0_47, bv_packed_1_47, av0_16_i8, 1);
                    acc0_47 = vusdotq_laneq_s32(acc0_47, bv_packed_2_47, av0_16_i8, 2);
                    acc0_47 = vusdotq_laneq_s32(acc0_47, bv_packed_3_47, av0_16_i8, 3);

                    QuantAPtr += KStep16;
                    QuantBDataPtr += NCols8 * KStep16;
                }

                accf0_03 = vmlaq_f32(accf0_03, scaleA0B03, vcvtq_f32_s32(acc0_03));
                accf0_47 = vmlaq_f32(accf0_47, scaleA0B47, vcvtq_f32_s32(acc0_47));

                ++QuantAScalePtr;
                QuantBScalePtr += NCols8;
            }

            if (BiasPtr != nullptr) {
                const float32x4_t bias_4_f32_03 = vld1q_f32(BiasPtr);
                const float32x4_t bias_4_f32_47 = vld1q_f32(BiasPtr + 4);
                accf0_03 = vaddq_f32(accf0_03, bias_4_f32_03);
                accf0_47 = vaddq_f32(accf0_47, bias_4_f32_47);
            }

            vst1q_f32(SumPtr, accf0_03);
            vst1q_f32(SumPtr + 4, accf0_47);

            // move to next NCols columns
            QuantBDataColPtr += StrideQuantBDataCol8;
            QuantBScaleColPtr += NCols8 * BlockCountK;

            BiasPtr += BiasPtr != nullptr ? NCols8 : 0;
            SumPtr += NCols8;
        }
    }
}

MLAS_FORCEINLINE void
Q8Int8GemmR2xC4I8MM(
    const size_t BlkLen,
    const int8_t* QuantA,
    const float* QuantAScale,
    const uint8_t* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc
)
{
    constexpr size_t NCols4 = 4;
    constexpr size_t NRows2 = 2;
    constexpr size_t KStep16 = 16;

    const size_t lda = BlockCountK * BlkLen;
    const size_t StrideQuantBDataCol4 = BlockCountK * BlkLen * NCols4;

    assert(CountM % NRows2 == 0);
    assert(CountN % NCols4 == 0);

    for (size_t m = 0; m < CountM; m += NRows2) {
        const uint8_t* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const float* BiasPtr = Bias;
        auto* SumPtr = C + m * ldc;

        for (size_t n = 0; n < CountN; n += NCols4) {
            const int8_t* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;

            const uint8_t* QuantBDataPtr = QuantBDataColPtr;
            const float* QuantBScalePtr = QuantBScaleColPtr;

            float32x4_t accf0_03 = vdupq_n_f32(0.0f);
            float32x4_t accf1_03 = vdupq_n_f32(0.0f);

            for (size_t i = 0; i < BlockCountK; ++i) {
                const float scaleA0 = *QuantAScalePtr;
                const float scaleA1 = *(QuantAScalePtr + BlockCountK);
                const float32x4_t scaleB = vld1q_f32(QuantBScalePtr);
                const float32x4_t scaleA0B03 = vmulq_n_f32(scaleB, scaleA0);
                const float32x4_t scaleA1B03 = vmulq_n_f32(scaleB, scaleA1);

                int32x4_t acc0_03 = vdupq_n_s32(0);
                int32x4_t acc1_03 = vdupq_n_s32(0);

                for (size_t k = 0; k < BlkLen; k += KStep16) {
                    const int8x16_t av0_16_i8 = vld1q_s8(QuantAPtr);
                    const int8x16_t av1_16_i8 = vld1q_s8(QuantAPtr + lda);

                    uint8x16_t bv_packed_0_03 = vld1q_u8(QuantBDataPtr);
                    uint8x16_t bv_packed_1_03 = vld1q_u8(QuantBDataPtr + 16);
                    uint8x16_t bv_packed_2_03 = vld1q_u8(QuantBDataPtr + 32);
                    uint8x16_t bv_packed_3_03 = vld1q_u8(QuantBDataPtr + 48);

                    acc0_03 = vusdotq_laneq_s32(acc0_03, bv_packed_0_03, av0_16_i8, 0);
                    acc0_03 = vusdotq_laneq_s32(acc0_03, bv_packed_1_03, av0_16_i8, 1);
                    acc0_03 = vusdotq_laneq_s32(acc0_03, bv_packed_2_03, av0_16_i8, 2);
                    acc0_03 = vusdotq_laneq_s32(acc0_03, bv_packed_3_03, av0_16_i8, 3);

                    acc1_03 = vusdotq_laneq_s32(acc1_03, bv_packed_0_03, av1_16_i8, 0);
                    acc1_03 = vusdotq_laneq_s32(acc1_03, bv_packed_1_03, av1_16_i8, 1);
                    acc1_03 = vusdotq_laneq_s32(acc1_03, bv_packed_2_03, av1_16_i8, 2);
                    acc1_03 = vusdotq_laneq_s32(acc1_03, bv_packed_3_03, av1_16_i8, 3);

                    QuantAPtr += KStep16;
                    QuantBDataPtr += NCols4 * KStep16;
                }

                accf0_03 = vmlaq_f32(accf0_03, scaleA0B03, vcvtq_f32_s32(acc0_03));
                accf1_03 = vmlaq_f32(accf1_03, scaleA1B03, vcvtq_f32_s32(acc1_03));

                ++QuantAScalePtr;
                QuantBScalePtr += NCols4;
            }

            if (BiasPtr != nullptr) {
                const float32x4_t bias_4_f32 = vld1q_f32(BiasPtr);
                accf0_03 = vaddq_f32(accf0_03, bias_4_f32);
                accf1_03 = vaddq_f32(accf1_03, bias_4_f32);
            }

            vst1q_f32(SumPtr, accf0_03);
            vst1q_f32(SumPtr + ldc, accf1_03);

            // move to next NCols columns
            QuantBDataColPtr += StrideQuantBDataCol4;
            QuantBScaleColPtr += NCols4 * BlockCountK;

            BiasPtr += BiasPtr != nullptr ? NCols4 : 0;
            SumPtr += NCols4;
        }
    }
}

MLAS_FORCEINLINE void
Q8Int8GemmR1xC4I8MM(
    const size_t BlkLen,
    const int8_t* QuantA,
    const float* QuantAScale,
    const uint8_t* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc
)
{
    constexpr size_t NCols4 = 4;
    constexpr size_t KStep16 = 16;

    const size_t lda = BlockCountK * BlkLen;
    const size_t StrideQuantBDataCol4 = BlockCountK * BlkLen * NCols4;

    assert(CountN % NCols4 == 0);

    for (size_t m = 0; m < CountM; ++m) {
        const uint8_t* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const float* BiasPtr = Bias;
        auto* SumPtr = C + m * ldc;

        for (size_t n = 0; n < CountN; n += NCols4) {
            const int8_t* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;

            const uint8_t* QuantBDataPtr = QuantBDataColPtr;
            const float* QuantBScalePtr = QuantBScaleColPtr;

            float32x4_t accf0_03 = vdupq_n_f32(0.0f);

            for (size_t i = 0; i < BlockCountK; ++i) {
                const float scaleA0 = *QuantAScalePtr;
                const float32x4_t scaleB = vld1q_f32(QuantBScalePtr);
                const float32x4_t scaleA0B03 = vmulq_n_f32(scaleB, scaleA0);

                int32x4_t acc0_03 = vdupq_n_s32(0);

                for (size_t k = 0; k < BlkLen; k += KStep16) {
                    const int8x16_t av0_16_i8 = vld1q_s8(QuantAPtr);

                    uint8x16_t bv_packed_0_03 = vld1q_u8(QuantBDataPtr);
                    uint8x16_t bv_packed_1_03 = vld1q_u8(QuantBDataPtr + 16);
                    uint8x16_t bv_packed_2_03 = vld1q_u8(QuantBDataPtr + 32);
                    uint8x16_t bv_packed_3_03 = vld1q_u8(QuantBDataPtr + 48);

                    acc0_03 = vusdotq_laneq_s32(acc0_03, bv_packed_0_03, av0_16_i8, 0);
                    acc0_03 = vusdotq_laneq_s32(acc0_03, bv_packed_1_03, av0_16_i8, 1);
                    acc0_03 = vusdotq_laneq_s32(acc0_03, bv_packed_2_03, av0_16_i8, 2);
                    acc0_03 = vusdotq_laneq_s32(acc0_03, bv_packed_3_03, av0_16_i8, 3);

                    QuantAPtr += KStep16;
                    QuantBDataPtr += NCols4 * KStep16;
                }

                accf0_03 = vmlaq_f32(accf0_03, scaleA0B03, vcvtq_f32_s32(acc0_03));

                ++QuantAScalePtr;
                QuantBScalePtr += NCols4;
            }

            if (BiasPtr != nullptr) {
                const float32x4_t bias_4_f32 = vld1q_f32(BiasPtr);
                accf0_03 = vaddq_f32(accf0_03, bias_4_f32);
            }

            vst1q_f32(SumPtr, accf0_03);

            // move to next NCols columns
            QuantBDataColPtr += StrideQuantBDataCol4;
            QuantBScaleColPtr += NCols4 * BlockCountK;

            BiasPtr += BiasPtr != nullptr ? NCols4 : 0;
            SumPtr += NCols4;
        }
    }
}

MLAS_FORCEINLINE void
Q8Int8GemmR2xC1I8MM(
    const size_t BlkLen,
    const int8_t* QuantA,
    const float* QuantAScale,
    const uint8_t* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc
)
{
    constexpr size_t NRows2 = 2;
    constexpr size_t KStep16 = 16;

    const size_t lda = BlockCountK * BlkLen;
    const size_t StrideQuantBDataCol = BlockCountK * BlkLen;

    assert(CountM % NRows2 == 0);

    for (size_t m = 0; m < CountM; m += NRows2) {
        const uint8_t* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const float* BiasPtr = Bias;
        auto* SumPtr = C + m * ldc;

        for (size_t n = 0; n < CountN; ++n) {
            const int8_t* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;

            const uint8_t* QuantBDataPtr = QuantBDataColPtr;
            const float* QuantBScalePtr = QuantBScaleColPtr;

            float32x4_t accf0 = vdupq_n_f32(0.0f);
            float32x4_t accf1 = vdupq_n_f32(0.0f);

            for (size_t i = 0; i < BlockCountK; ++i) {
                const float scaleA0 = *QuantAScalePtr;
                const float scaleA1 = *(QuantAScalePtr + BlockCountK);
                const float scaleB = *QuantBScalePtr;
                const float scaleA0B = scaleB * scaleA0;
                const float scaleA1B = scaleB * scaleA1;

                int32x4_t acc0 = vdupq_n_s32(0);
                int32x4_t acc1 = vdupq_n_s32(0);

                for (size_t k = 0; k < BlkLen; k += KStep16) {
                    const int8x16_t av0_16_i8 = vld1q_s8(QuantAPtr);
                    const int8x16_t av1_16_i8 = vld1q_s8(QuantAPtr + lda);

                    uint8x16_t bv_packed = vld1q_u8(QuantBDataPtr);

                    acc0 = vusdotq_s32(acc0, bv_packed, av0_16_i8);
                    acc1 = vusdotq_s32(acc1, bv_packed, av1_16_i8);

                    QuantAPtr += KStep16;
                    QuantBDataPtr += KStep16;
                }

                accf0 = vmlaq_n_f32(accf0, vcvtq_f32_s32(acc0), scaleA0B);
                accf1 = vmlaq_n_f32(accf1, vcvtq_f32_s32(acc1), scaleA1B);

                ++QuantAScalePtr;
                ++QuantBScalePtr;
            }

            float32_t accf0v = vaddvq_f32(accf0);
            float32_t accf1v = vaddvq_f32(accf1);

            if (BiasPtr != nullptr) {
                const float bias = *BiasPtr;
                accf0v += bias;
                accf1v += bias;
            }

            *SumPtr = accf0v;
            *(SumPtr + ldc) = accf1v;

            // move to next NCols columns
            QuantBDataColPtr += StrideQuantBDataCol;
            QuantBScaleColPtr += BlockCountK;

            BiasPtr += BiasPtr ? 1 : 0;
            ++SumPtr;
        }
    }
}

MLAS_FORCEINLINE void
Q8Int8GemmR1xC1I8MM(
    const size_t BlkLen,
    const int8_t* QuantA,
    const float* QuantAScale,
    const uint8_t* QuantBData,
    const float* QuantBScale,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc
)
{
    constexpr size_t KStep16 = 16;

    const size_t lda = BlockCountK * BlkLen;
    const size_t StrideQuantBDataCol = BlockCountK * BlkLen;

    for (size_t m = 0; m < CountM; ++m) {
        const uint8_t* QuantBDataColPtr = QuantBData;
        const float* QuantBScaleColPtr = QuantBScale;
        const float* BiasPtr = Bias;
        auto* SumPtr = C + m * ldc;

        for (size_t n = 0; n < CountN; ++n) {
            const int8_t* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;

            const uint8_t* QuantBDataPtr = QuantBDataColPtr;
            const float* QuantBScalePtr = QuantBScaleColPtr;

            float32x4_t accf0 = vdupq_n_f32(0.0f);

            for (size_t i = 0; i < BlockCountK; ++i) {
                const float scaleA0 = *QuantAScalePtr;
                const float scaleB = *QuantBScalePtr;
                const float scaleA0B = scaleB * scaleA0;

                int32x4_t acc0 = vdupq_n_s32(0);

                for (size_t k = 0; k < BlkLen; k += KStep16) {
                    const int8x16_t av0_16_i8 = vld1q_s8(QuantAPtr);

                    uint8x16_t bv_packed = vld1q_u8(QuantBDataPtr);

                    acc0 = vusdotq_s32(acc0, bv_packed, av0_16_i8);

                    QuantAPtr += KStep16;
                    QuantBDataPtr += KStep16;
                }

                accf0 = vmlaq_n_f32(accf0, vcvtq_f32_s32(acc0), scaleA0B);

                ++QuantAScalePtr;
                ++QuantBScalePtr;
            }

            float32_t accf0v = vaddvq_f32(accf0);

            if (BiasPtr != nullptr) {
                const float bias = *BiasPtr;
                accf0v += bias;
            }

            *SumPtr = accf0v;

            // move to next NCols columns
            QuantBDataColPtr += StrideQuantBDataCol;
            QuantBScaleColPtr += BlockCountK;

            BiasPtr += BiasPtr ? 1 : 0;
            ++SumPtr;
        }
    }
}

template <>
size_t
MlasQ8Int8GemmKernelNeon<false>(
    const size_t BlkLen,
    const int8_t* QuantA,
    const float* QuantAScale,
    const uint8_t* QuantBData,
    const float * QuantBScale,
    float* C,
    const size_t CountM,
    const size_t CountN,
    const size_t CountK,
    const float* Bias,
    const size_t ldc
) {
    constexpr size_t BlkBitWidth = 8;
    constexpr size_t NCols8 = 8;
    constexpr size_t NCols4 = 4;
    constexpr size_t NRows2 = 2;
    const size_t BlockCountK = MlasDivRoundup(CountK, BlkLen);

    const size_t lda = BlockCountK * BlkLen;
    const size_t lda_scale = BlockCountK;
    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t StrideQuantBScale = BlockCountK;

    size_t remainingRows = CountM % NRows2;
    size_t multipleRows = CountM - remainingRows;
    size_t multipleCols8 = CountN & (~(NCols8 - 1));
    size_t multipleCols4 = CountN & (~(NCols4 - 1));
    size_t remainingCols4 = CountN % NCols4;

    if (multipleRows > 0 && multipleCols8 > 0) {
        Q8Int8GemmR2xC8I8MM(
            BlkLen,
            QuantA,
            QuantAScale,
            QuantBData,
            QuantBScale,
            C,
            multipleRows,
            multipleCols8,
            BlockCountK,
            Bias,
            ldc
        );
    }

    if (multipleRows > 0 && multipleCols4 > multipleCols8) {
        Q8Int8GemmR2xC4I8MM(
            BlkLen,
            QuantA,
            QuantAScale,
            QuantBData + multipleCols8 * StrideQuantBData,
            QuantBScale + multipleCols8 * StrideQuantBScale,
            C + multipleCols8,
            multipleRows,
            multipleCols4 - multipleCols8,
            BlockCountK,
            Bias ? Bias + multipleCols8 : nullptr,
            ldc
        );
    }

    if (multipleRows > 0 && remainingCols4 > 0) {
        Q8Int8GemmR2xC1I8MM(
            BlkLen,
            QuantA,
            QuantAScale,
            QuantBData + multipleCols4 * StrideQuantBData,
            QuantBScale + multipleCols4 * StrideQuantBScale,
            C + multipleCols4,
            multipleRows,
            remainingCols4,
            BlockCountK,
            Bias ? Bias + multipleCols4 : nullptr,
            ldc
        );
    }

    if (remainingRows > 0 && multipleCols8 > 0) {
        Q8Int8GemmR1xC8I8MM(
            BlkLen,
            QuantA + multipleRows * lda,
            QuantAScale + multipleRows * lda_scale,
            QuantBData,
            QuantBScale,
            C + multipleRows * ldc,
            remainingRows,
            multipleCols8,
            BlockCountK,
            Bias,
            ldc);
    }

    if (remainingRows > 0 && multipleCols4 > multipleCols8) {
        Q8Int8GemmR1xC4I8MM(
            BlkLen,
            QuantA + multipleRows * lda,
            QuantAScale + multipleRows * lda_scale,
            QuantBData + multipleCols8 * StrideQuantBData,
            QuantBScale + multipleCols8 * StrideQuantBScale,
            C + multipleRows * ldc + multipleCols8,
            remainingRows,
            multipleCols4 - multipleCols8,
            BlockCountK,
            Bias ? Bias + multipleCols8 : nullptr,
            ldc);
    }

    if (remainingRows > 0 && remainingCols4 > 0) {
        Q8Int8GemmR1xC1I8MM(
            BlkLen,
            QuantA + multipleRows * lda,
            QuantAScale + multipleRows * lda_scale,
            QuantBData + multipleCols4 * StrideQuantBData,
            QuantBScale + multipleCols4 * StrideQuantBScale,
            C + multipleRows * ldc + multipleCols4,
            remainingRows,
            remainingCols4,
            BlockCountK,
            Bias ? Bias + multipleCols4 : nullptr,
            ldc);
    }

    return CountM;
}

}  // namespace sqnbitgemm_neon
