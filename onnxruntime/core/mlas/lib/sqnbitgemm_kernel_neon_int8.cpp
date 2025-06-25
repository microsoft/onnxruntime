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
#include <limits>

#include "qnbitgemm.h"
#include "qnbitgemm_kernel_neon.h"
#include "sqnbitgemm_q8_block.h"

#ifdef USE_KLEIDIAI
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai_ukernel_interface.h"
#endif

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

bool
UsePacked_CompInt8(size_t K, size_t BlkLen, bool HasZp)
{
    return UseKleidiAI(K, BlkLen, HasZp);
}

#ifdef USE_KLEIDIAI
void
QuantizeA_Packed_CompInt8(
    size_t,
    const float* A,
    size_t CountM,
    size_t CountK,
    std::byte* QuantA
)
{
    const kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel& ukernel =
        CountM == 1? GetKleidiAIGemvUKernel() : GetKleidiAIGemmUKernel();

    const size_t mr = ukernel.get_mr();
    const size_t kr = ukernel.get_kr();
    const size_t sr = ukernel.get_sr();

    const size_t src_stride = CountK * sizeof(float);
    const size_t lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32(0, src_stride);
    const size_t lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(
                            0, CountK, mr, kr, sr);

    const float* src_ptr = reinterpret_cast<const float*>(reinterpret_cast<const std::byte*>(A) + lhs_offset);
    void* dst_ptr = QuantA + lhs_packed_offset;

    kai_run_lhs_quant_pack_qai8dxp_f32(CountM, CountK, mr, kr, sr, 0, src_ptr, src_stride, dst_ptr);
}
#endif

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

MLAS_FORCEINLINE
float32x4_t LoadFloat32x4(const float* src, size_t count)
{
    if (count == 4) {
        return vld1q_f32(src);
    } else if (count == 3) {
        float32x4_t v = vdupq_n_f32(0.0f);
        v = vld1q_lane_f32(src, v, 0);
        v = vld1q_lane_f32(src + 1, v, 1);
        v = vld1q_lane_f32(src + 2, v, 2);
        return v;
    } else if (count == 2) {
        float32x4_t v = vdupq_n_f32(0.0f);
        v = vld1q_lane_f32(src, v, 0);
        v = vld1q_lane_f32(src + 1, v, 1);
        return v;
    } else {
        assert(count == 1);
        float32x4_t v = vdupq_n_f32(0.0f);
        v = vld1q_lane_f32(src, v, 0);
        return v;
    }
}

template <bool QuantAUnsigned>
using I16VecType = typename std::conditional<QuantAUnsigned, uint16x8_t, int16x8_t>::type;

template <bool QuantAUnsigned>
I16VecType<QuantAUnsigned> MLAS_FORCEINLINE
PrepareZeroI16()
{
    if constexpr (QuantAUnsigned) {
        return vdupq_n_u16(0);
    } else {
        return vdupq_n_s16(0);
    }
}

template <bool QuantAUnsigned>
void MLASCALL
QuantizeARowComputeBlkSum_CompInt8(
    size_t BlkLen,
    const float* A,
    size_t CountK,
    std::byte* QuantA,
    float* QuantAScale,
    float* AScaledBlkSum // scale_k * Sum_blklen(a_i)
)
{
    // First use i8 to quantize A. range [-128, 127]
    // If convert to u8, +128. Range [0, 255]
    assert(BlkLen % 16 == 0);
    assert(BlkLen <= 256);
    MLAS_DECLSPEC_ALIGN(static const uint8_t MASK[16], 16) = {
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    };
    const int16x8_t v128 = vdupq_n_s16(128);
    QuantAType<QuantAUnsigned>* blob = reinterpret_cast<QuantAType<QuantAUnsigned>*>(QuantA);
    float* scale_ptr = QuantAScale;
    size_t k = 0;
    for (; k + BlkLen <= CountK; k += BlkLen) {
        float32x4_t absMax0 = vdupq_n_f32(0.0f);
        float32x4_t absMax1 = vdupq_n_f32(0.0f);
        float32x4_t absMax2 = vdupq_n_f32(0.0f);
        float32x4_t absMax3 = vdupq_n_f32(0.0f);

        for (size_t kk = 0; kk < BlkLen; kk += 16) {
            const float32x4x4_t v0 = vld4q_f32(A + k + kk);
            absMax0 = vmaxq_f32(absMax0, vabsq_f32(v0.val[0]));
            absMax1 = vmaxq_f32(absMax1, vabsq_f32(v0.val[1]));
            absMax2 = vmaxq_f32(absMax2, vabsq_f32(v0.val[2]));
            absMax3 = vmaxq_f32(absMax3, vabsq_f32(v0.val[3]));
        }

        const float32x4_t max01 = vmaxq_f32(absMax0, absMax1);
        const float32x4_t max23 = vmaxq_f32(absMax2, absMax3);
        const float32x4_t max0123 = vmaxq_f32(max01, max23);
        const float maxScalar = vmaxvq_f32(max0123);

        // Quantize these floats
        const float scale = maxScalar / 127.f;
        *scale_ptr = scale;
        scale_ptr++;

        const float inverse_scale = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
        const float32x4_t mul = vdupq_n_f32(inverse_scale);

        I16VecType<QuantAUnsigned> sum_8_i16_0 = PrepareZeroI16<QuantAUnsigned>();
        I16VecType<QuantAUnsigned> sum_8_i16_1 = PrepareZeroI16<QuantAUnsigned>();

        for (size_t kk = 0; kk < BlkLen; kk += 16) {
            const float32x4_t vfp32_0 = LoadFloat32x4(A + k + kk, 4);
            const float32x4_t vfp32_1 = LoadFloat32x4(A + k + kk + 4, 4);
            const float32x4_t vfp32_2 = LoadFloat32x4(A + k + kk + 8, 4);
            const float32x4_t vfp32_3 = LoadFloat32x4(A + k + kk + 12, 4);

            const float32x4_t v0 = vmulq_f32(vfp32_0, mul);
            const float32x4_t v1 = vmulq_f32(vfp32_1, mul);
            const float32x4_t v2 = vmulq_f32(vfp32_2, mul);
            const float32x4_t v3 = vmulq_f32(vfp32_3, mul);

            const int32x4_t i0 = vcvtnq_s32_f32(v0);
            const int32x4_t i1 = vcvtnq_s32_f32(v1);
            const int32x4_t i2 = vcvtnq_s32_f32(v2);
            const int32x4_t i3 = vcvtnq_s32_f32(v3);

            const int16x8_t v_8_i16_0 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
            const int16x8_t v_8_i16_1 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));

            if constexpr (QuantAUnsigned) {
                const uint16x8_t v_8_u16_0 = vreinterpretq_u16_s16(vaddq_s16(v_8_i16_0, v128));
                const uint16x8_t v_8_u16_1 = vreinterpretq_u16_s16(vaddq_s16(v_8_i16_1, v128));
                const uint8x16_t v_16_u8 = vcombine_u8(vqmovn_u16(v_8_u16_0), vqmovn_u16(v_8_u16_1));
                vst1q_u8(blob + k + kk, v_16_u8);

                // accumulate Sum(a_i)
                const uint16x8_t i_8_u16_0 = vmovl_u8(vget_low_u8(v_16_u8));
                const uint16x8_t i_8_u16_1 = vmovl_high_u8(v_16_u8);
                sum_8_i16_0 = vaddq_u16(sum_8_i16_0, i_8_u16_0);
                sum_8_i16_1 = vaddq_u16(sum_8_i16_1, i_8_u16_1);
            } else {
                const int8x16_t v_16_i8 = vcombine_s8(vqmovn_s16(v_8_i16_0), vqmovn_s16(v_8_i16_1));
                vst1q_s8(blob + k + kk, v_16_i8);

                // accumulate Sum(a_i)
                const int16x8_t i_8_i16_0 = vmovl_s8(vget_low_s8(v_16_i8));
                const int16x8_t i_8_i16_1 = vmovl_high_s8(v_16_i8);
                sum_8_i16_0 = vaddq_s16(sum_8_i16_0, i_8_i16_0);
                sum_8_i16_1 = vaddq_s16(sum_8_i16_1, i_8_i16_1);
            }
        }

        float qsum;

        if constexpr (QuantAUnsigned) {
            const uint16x8_t sum_8_u16 = vaddq_u16(sum_8_i16_0, sum_8_i16_1);
            qsum = static_cast<float>(vaddvq_u16(sum_8_u16));
        } else {
            const int16x8_t sum_8_i16 = vaddq_s16(sum_8_i16_0, sum_8_i16_1);
            qsum = static_cast<float>(vaddvq_s16(sum_8_i16));
        }

        *AScaledBlkSum = scale * qsum;
        AScaledBlkSum++;
    }

    if (k < CountK) {
        float32x4_t absMax = vdupq_n_f32(0.0f);

        for (size_t kk = k; kk < CountK; kk += 4) {
            size_t step = std::min(static_cast<size_t>(4), CountK - kk);
            const float32x4_t v0 = LoadFloat32x4(A + kk, step);
            absMax = vmaxq_f32(absMax, vabsq_f32(v0));
        }

        const float maxScalar = vmaxvq_f32(absMax);
        const float scale = maxScalar / 127.f;
        *scale_ptr = scale;

        const float inverse_scale = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
        const float32x4_t mul = vdupq_n_f32(inverse_scale);

        I16VecType<QuantAUnsigned> sum_8_i16 = PrepareZeroI16<QuantAUnsigned>();

        for (size_t kk = k; kk < CountK; kk += 4) {
            size_t step = std::min(static_cast<size_t>(4), CountK - kk);
            const float32x4_t vfp32 = LoadFloat32x4(A + kk, step);
            const float32x4_t v_f32 = vmulq_f32(vfp32, mul);
            const int32x4_t v_i32 = vcvtnq_s32_f32(v_f32);
            const int16x8_t v_8_i16 = vcombine_s16(vqmovn_s32(v_i32), vdup_n_s16(0));

            if constexpr (QuantAUnsigned) {
                const uint16x8_t v_8_u16 = vreinterpretq_u16_s16(vaddq_s16(v_8_i16, v128));
                uint8x8_t v_8_u8 = vqmovn_u16(v_8_u16);
                vst1_lane_s32(reinterpret_cast<int32_t*>(blob + kk), vreinterpret_s32_u8(v_8_u8), 0);

                // accumulate Sum(a_i)
                v_8_u8 = vand_u8(v_8_u8, vld1_u8(MASK + 8 - step));
                const uint16x8_t i_8_u16 = vmovl_u8(v_8_u8);
                sum_8_i16 = vaddq_u16(sum_8_i16, i_8_u16);
            } else {
                const int8x8_t v_8_i8 = vqmovn_s16(v_8_i16);
                vst1_lane_s32(reinterpret_cast<int32_t*>(blob + kk), vreinterpret_s32_s8(v_8_i8), 0);

                // accumulate Sum(a_i)
                const int16x8_t i_8_i16 = vmovl_s8(v_8_i8);
                sum_8_i16 = vaddq_s16(sum_8_i16, i_8_i16);
            }
        }

        float qsum;

        if constexpr (QuantAUnsigned) {
            qsum = static_cast<float>(vaddvq_u16(sum_8_i16));
        } else {
            qsum = static_cast<float>(vaddvq_s16(sum_8_i16));
        }

        *AScaledBlkSum = scale * qsum;

        memset(blob + CountK, 0, BlkLen - (CountK % BlkLen));
    }
}

template
void MLASCALL
QuantizeARowComputeBlkSum_CompInt8<true>(
    size_t BlkLen,
    const float* A,
    size_t CountK,
    std::byte* QuantA,
    float* QuantAScale,
    float* AScaledBlkSum // scale_k * Sum_blklen(a_i)
);

template
void MLASCALL
QuantizeARowComputeBlkSum_CompInt8<false>(
    size_t BlkLen,
    const float* A,
    size_t CountK,
    std::byte* QuantA,
    float* QuantAScale,
    float* AScaledBlkSum // scale_k * Sum_blklen(a_i)
);

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

MLAS_FORCEINLINE void
Q8Int8GemmR2xC8DotProd(
    const size_t BlkLen,
    const uint8_t* QuantA,
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
            const uint8_t* QuantAPtr = QuantA + m * lda;
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
                const float32x4x2_t scaleB = vld2q_f32(QuantBScalePtr);
                const float32x4_t scaleA0B03 = vmulq_n_f32(scaleB.val[0], scaleA0);
                const float32x4_t scaleA0B47 = vmulq_n_f32(scaleB.val[1], scaleA0);
                const float32x4_t scaleA1B03 = vmulq_n_f32(scaleB.val[0], scaleA1);
                const float32x4_t scaleA1B47 = vmulq_n_f32(scaleB.val[1], scaleA1);

                uint32x4_t acc0_03 = vdupq_n_u32(0U);
                uint32x4_t acc0_47 = vdupq_n_u32(0U);
                uint32x4_t acc1_03 = vdupq_n_u32(0U);
                uint32x4_t acc1_47 = vdupq_n_u32(0U);

                for (size_t k = 0; k < BlkLen; k += KStep16) {
                    const uint8x16_t av0_16_i8 = vld1q_u8(QuantAPtr);
                    const uint8x16_t av1_16_i8 = vld1q_u8(QuantAPtr + lda);

                    uint8x16_t bv_packed_0_03 = vld1q_u8(QuantBDataPtr);
                    uint8x16_t bv_packed_0_47 = vld1q_u8(QuantBDataPtr + 16);
                    uint8x16_t bv_packed_1_03 = vld1q_u8(QuantBDataPtr + 32);
                    uint8x16_t bv_packed_1_47 = vld1q_u8(QuantBDataPtr + 48);
                    uint8x16_t bv_packed_2_03 = vld1q_u8(QuantBDataPtr + 64);
                    uint8x16_t bv_packed_2_47 = vld1q_u8(QuantBDataPtr + 80);
                    uint8x16_t bv_packed_3_03 = vld1q_u8(QuantBDataPtr + 96);
                    uint8x16_t bv_packed_3_47 = vld1q_u8(QuantBDataPtr + 112);

                    acc0_03 = vdotq_laneq_u32(acc0_03, bv_packed_0_03, av0_16_i8, 0);
                    acc0_03 = vdotq_laneq_u32(acc0_03, bv_packed_1_03, av0_16_i8, 1);
                    acc0_03 = vdotq_laneq_u32(acc0_03, bv_packed_2_03, av0_16_i8, 2);
                    acc0_03 = vdotq_laneq_u32(acc0_03, bv_packed_3_03, av0_16_i8, 3);

                    acc0_47 = vdotq_laneq_u32(acc0_47, bv_packed_0_47, av0_16_i8, 0);
                    acc0_47 = vdotq_laneq_u32(acc0_47, bv_packed_1_47, av0_16_i8, 1);
                    acc0_47 = vdotq_laneq_u32(acc0_47, bv_packed_2_47, av0_16_i8, 2);
                    acc0_47 = vdotq_laneq_u32(acc0_47, bv_packed_3_47, av0_16_i8, 3);

                    acc1_03 = vdotq_laneq_u32(acc1_03, bv_packed_0_03, av1_16_i8, 0);
                    acc1_03 = vdotq_laneq_u32(acc1_03, bv_packed_1_03, av1_16_i8, 1);
                    acc1_03 = vdotq_laneq_u32(acc1_03, bv_packed_2_03, av1_16_i8, 2);
                    acc1_03 = vdotq_laneq_u32(acc1_03, bv_packed_3_03, av1_16_i8, 3);

                    acc1_47 = vdotq_laneq_u32(acc1_47, bv_packed_0_47, av1_16_i8, 0);
                    acc1_47 = vdotq_laneq_u32(acc1_47, bv_packed_1_47, av1_16_i8, 1);
                    acc1_47 = vdotq_laneq_u32(acc1_47, bv_packed_2_47, av1_16_i8, 2);
                    acc1_47 = vdotq_laneq_u32(acc1_47, bv_packed_3_47, av1_16_i8, 3);

                    QuantAPtr += KStep16;
                    QuantBDataPtr += NCols8 * KStep16;
                }

                accf0_03 = vmlaq_f32(accf0_03, scaleA0B03, vcvtq_f32_u32(acc0_03));
                accf0_47 = vmlaq_f32(accf0_47, scaleA0B47, vcvtq_f32_u32(acc0_47));
                accf1_03 = vmlaq_f32(accf1_03, scaleA1B03, vcvtq_f32_u32(acc1_03));
                accf1_47 = vmlaq_f32(accf1_47, scaleA1B47, vcvtq_f32_u32(acc1_47));

                ++QuantAScalePtr;
                QuantBScalePtr += NCols8;
            }

            if (BiasPtr != nullptr) {
                const float32x4x2_t bias_4x2_f32 = vld2q_f32(BiasPtr);
                accf0_03 = vaddq_f32(accf0_03, bias_4x2_f32.val[0]);
                accf0_47 = vaddq_f32(accf0_47, bias_4x2_f32.val[1]);
                accf1_03 = vaddq_f32(accf1_03, bias_4x2_f32.val[0]);
                accf1_47 = vaddq_f32(accf1_47, bias_4x2_f32.val[1]);
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
Q8Int8GemmR1xC8DotProd(
    const size_t BlkLen,
    const uint8_t* QuantA,
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
            const uint8_t* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;

            const uint8_t* QuantBDataPtr = QuantBDataColPtr;
            const float* QuantBScalePtr = QuantBScaleColPtr;

            float32x4_t accf0_03 = vdupq_n_f32(0.0f);
            float32x4_t accf0_47 = vdupq_n_f32(0.0f);

            for (size_t i = 0; i < BlockCountK; ++i) {
                const float scaleA0 = *QuantAScalePtr;
                const float32x4x2_t scaleB = vld2q_f32(QuantBScalePtr);
                const float32x4_t scaleA0B03 = vmulq_n_f32(scaleB.val[0], scaleA0);
                const float32x4_t scaleA0B47 = vmulq_n_f32(scaleB.val[1], scaleA0);

                uint32x4_t acc0_03 = vdupq_n_u32(0U);
                uint32x4_t acc0_47 = vdupq_n_u32(0U);

                for (size_t k = 0; k < BlkLen; k += KStep16) {
                    const uint8x16_t av0_16_i8 = vld1q_u8(QuantAPtr);

                    uint8x16_t bv_packed_0_03 = vld1q_u8(QuantBDataPtr);
                    uint8x16_t bv_packed_0_47 = vld1q_u8(QuantBDataPtr + 16);
                    uint8x16_t bv_packed_1_03 = vld1q_u8(QuantBDataPtr + 32);
                    uint8x16_t bv_packed_1_47 = vld1q_u8(QuantBDataPtr + 48);
                    uint8x16_t bv_packed_2_03 = vld1q_u8(QuantBDataPtr + 64);
                    uint8x16_t bv_packed_2_47 = vld1q_u8(QuantBDataPtr + 80);
                    uint8x16_t bv_packed_3_03 = vld1q_u8(QuantBDataPtr + 96);
                    uint8x16_t bv_packed_3_47 = vld1q_u8(QuantBDataPtr + 112);

                    acc0_03 = vdotq_laneq_u32(acc0_03, bv_packed_0_03, av0_16_i8, 0);
                    acc0_03 = vdotq_laneq_u32(acc0_03, bv_packed_1_03, av0_16_i8, 1);
                    acc0_03 = vdotq_laneq_u32(acc0_03, bv_packed_2_03, av0_16_i8, 2);
                    acc0_03 = vdotq_laneq_u32(acc0_03, bv_packed_3_03, av0_16_i8, 3);

                    acc0_47 = vdotq_laneq_u32(acc0_47, bv_packed_0_47, av0_16_i8, 0);
                    acc0_47 = vdotq_laneq_u32(acc0_47, bv_packed_1_47, av0_16_i8, 1);
                    acc0_47 = vdotq_laneq_u32(acc0_47, bv_packed_2_47, av0_16_i8, 2);
                    acc0_47 = vdotq_laneq_u32(acc0_47, bv_packed_3_47, av0_16_i8, 3);

                    QuantAPtr += KStep16;
                    QuantBDataPtr += NCols8 * KStep16;
                }

                accf0_03 = vmlaq_f32(accf0_03, scaleA0B03, vcvtq_f32_u32(acc0_03));
                accf0_47 = vmlaq_f32(accf0_47, scaleA0B47, vcvtq_f32_u32(acc0_47));

                ++QuantAScalePtr;
                QuantBScalePtr += NCols8;
            }

            if (BiasPtr != nullptr) {
                const float32x4x2_t bias_4x2_f32 = vld2q_f32(BiasPtr);
                accf0_03 = vaddq_f32(accf0_03, bias_4x2_f32.val[0]);
                accf0_47 = vaddq_f32(accf0_47, bias_4x2_f32.val[1]);
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
Q8Int8GemmR2xC4DotProd(
    const size_t BlkLen,
    const uint8_t* QuantA,
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
            const uint8_t* QuantAPtr = QuantA + m * lda;
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

                uint32x4_t acc0_03 = vdupq_n_u32(0U);
                uint32x4_t acc1_03 = vdupq_n_u32(0U);

                for (size_t k = 0; k < BlkLen; k += KStep16) {
                    const uint8x16_t av0_16_i8 = vld1q_u8(QuantAPtr);
                    const uint8x16_t av1_16_i8 = vld1q_u8(QuantAPtr + lda);

                    uint8x16_t bv_packed_0_03 = vld1q_u8(QuantBDataPtr);
                    uint8x16_t bv_packed_1_03 = vld1q_u8(QuantBDataPtr + 16);
                    uint8x16_t bv_packed_2_03 = vld1q_u8(QuantBDataPtr + 32);
                    uint8x16_t bv_packed_3_03 = vld1q_u8(QuantBDataPtr + 48);

                    acc0_03 = vdotq_laneq_u32(acc0_03, bv_packed_0_03, av0_16_i8, 0);
                    acc0_03 = vdotq_laneq_u32(acc0_03, bv_packed_1_03, av0_16_i8, 1);
                    acc0_03 = vdotq_laneq_u32(acc0_03, bv_packed_2_03, av0_16_i8, 2);
                    acc0_03 = vdotq_laneq_u32(acc0_03, bv_packed_3_03, av0_16_i8, 3);

                    acc1_03 = vdotq_laneq_u32(acc1_03, bv_packed_0_03, av1_16_i8, 0);
                    acc1_03 = vdotq_laneq_u32(acc1_03, bv_packed_1_03, av1_16_i8, 1);
                    acc1_03 = vdotq_laneq_u32(acc1_03, bv_packed_2_03, av1_16_i8, 2);
                    acc1_03 = vdotq_laneq_u32(acc1_03, bv_packed_3_03, av1_16_i8, 3);

                    QuantAPtr += KStep16;
                    QuantBDataPtr += NCols4 * KStep16;
                }

                accf0_03 = vmlaq_f32(accf0_03, scaleA0B03, vcvtq_f32_u32(acc0_03));
                accf1_03 = vmlaq_f32(accf1_03, scaleA1B03, vcvtq_f32_u32(acc1_03));

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
Q8Int8GemmR1xC4DotProd(
    const size_t BlkLen,
    const uint8_t* QuantA,
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
            const uint8_t* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;

            const uint8_t* QuantBDataPtr = QuantBDataColPtr;
            const float* QuantBScalePtr = QuantBScaleColPtr;

            float32x4_t accf0_03 = vdupq_n_f32(0.0f);

            for (size_t i = 0; i < BlockCountK; ++i) {
                const float scaleA0 = *QuantAScalePtr;
                const float32x4_t scaleB = vld1q_f32(QuantBScalePtr);
                const float32x4_t scaleA0B03 = vmulq_n_f32(scaleB, scaleA0);

                uint32x4_t acc0_03 = vdupq_n_u32(0U);

                for (size_t k = 0; k < BlkLen; k += KStep16) {
                    const uint8x16_t av0_16_i8 = vld1q_u8(QuantAPtr);

                    uint8x16_t bv_packed_0_03 = vld1q_u8(QuantBDataPtr);
                    uint8x16_t bv_packed_1_03 = vld1q_u8(QuantBDataPtr + 16);
                    uint8x16_t bv_packed_2_03 = vld1q_u8(QuantBDataPtr + 32);
                    uint8x16_t bv_packed_3_03 = vld1q_u8(QuantBDataPtr + 48);

                    acc0_03 = vdotq_laneq_u32(acc0_03, bv_packed_0_03, av0_16_i8, 0);
                    acc0_03 = vdotq_laneq_u32(acc0_03, bv_packed_1_03, av0_16_i8, 1);
                    acc0_03 = vdotq_laneq_u32(acc0_03, bv_packed_2_03, av0_16_i8, 2);
                    acc0_03 = vdotq_laneq_u32(acc0_03, bv_packed_3_03, av0_16_i8, 3);

                    QuantAPtr += KStep16;
                    QuantBDataPtr += NCols4 * KStep16;
                }

                accf0_03 = vmlaq_f32(accf0_03, scaleA0B03, vcvtq_f32_u32(acc0_03));

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
Q8Int8GemmR2xC1DotProd(
    const size_t BlkLen,
    const uint8_t* QuantA,
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
            const uint8_t* QuantAPtr = QuantA + m * lda;
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

                uint32x4_t acc0 = vdupq_n_u32(0U);
                uint32x4_t acc1 = vdupq_n_u32(0U);

                for (size_t k = 0; k < BlkLen; k += KStep16) {
                    const uint8x16_t av0_16_i8 = vld1q_u8(QuantAPtr);
                    const uint8x16_t av1_16_i8 = vld1q_u8(QuantAPtr + lda);

                    uint8x16_t bv_packed = vld1q_u8(QuantBDataPtr);

                    acc0 = vdotq_u32(acc0, bv_packed, av0_16_i8);
                    acc1 = vdotq_u32(acc1, bv_packed, av1_16_i8);

                    QuantAPtr += KStep16;
                    QuantBDataPtr += KStep16;
                }

                accf0 = vmlaq_n_f32(accf0, vcvtq_f32_u32(acc0), scaleA0B);
                accf1 = vmlaq_n_f32(accf1, vcvtq_f32_u32(acc1), scaleA1B);

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
Q8Int8GemmR1xC1DotProd(
    const size_t BlkLen,
    const uint8_t* QuantA,
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
            const uint8_t* QuantAPtr = QuantA + m * lda;
            const float* QuantAScalePtr = QuantAScale + m * BlockCountK;

            const uint8_t* QuantBDataPtr = QuantBDataColPtr;
            const float* QuantBScalePtr = QuantBScaleColPtr;

            float32x4_t accf0 = vdupq_n_f32(0.0f);

            for (size_t i = 0; i < BlockCountK; ++i) {
                const float scaleA0 = *QuantAScalePtr;
                const float scaleB = *QuantBScalePtr;
                const float scaleA0B = scaleB * scaleA0;

                uint32x4_t acc0 = vdupq_n_u32(0U);

                for (size_t k = 0; k < BlkLen; k += KStep16) {
                    const uint8x16_t av0_16_i8 = vld1q_u8(QuantAPtr);

                    uint8x16_t bv_packed = vld1q_u8(QuantBDataPtr);

                    acc0 = vdotq_u32(acc0, bv_packed, av0_16_i8);

                    QuantAPtr += KStep16;
                    QuantBDataPtr += KStep16;
                }

                accf0 = vmlaq_n_f32(accf0, vcvtq_f32_u32(acc0), scaleA0B);

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
MlasQ8Int8GemmKernelNeon<true>(
    const size_t BlkLen,
    const uint8_t* QuantA,
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
        Q8Int8GemmR2xC8DotProd(
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
        Q8Int8GemmR2xC4DotProd(
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
        Q8Int8GemmR2xC1DotProd(
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
        Q8Int8GemmR1xC8DotProd(
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
        Q8Int8GemmR1xC4DotProd(
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
        Q8Int8GemmR1xC1DotProd(
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

#ifdef USE_KLEIDIAI
void
SQ4BitGemmKernel_Packed_CompInt8(
    size_t BlkLen,
    const std::byte* QuantA,
    const std::byte* PackedQuantBData,
    float* C,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN,
    size_t CountK,
    size_t ldc,
    const float* Bias
)
{
    const kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel ukernel =
        RangeCountM == 1 && RangeStartM == 0? GetKleidiAIGemvUKernel() : GetKleidiAIGemmUKernel();

    const size_t dst_stride = ldc * sizeof(float);

    const size_t lhs_packed_offset = ukernel.get_lhs_packed_offset(RangeStartM, CountK);
    const size_t rhs_packed_offset = ukernel.get_rhs_packed_offset(RangeStartN, CountK, BlkLen);
    const size_t dst_offset = ukernel.get_dst_offset(RangeStartM, RangeStartN, dst_stride);

    const void* lhs_ptr = QuantA + lhs_packed_offset;
    const void* rhs_ptr = PackedQuantBData + rhs_packed_offset;
    float* dst_ptr = reinterpret_cast<float*>(reinterpret_cast<std::byte*>(C) + dst_offset);

    ukernel.run_matmul(
        RangeCountM, RangeCountN, CountK, BlkLen, lhs_ptr, rhs_ptr, dst_ptr, dst_stride, sizeof(float),
        -std::numeric_limits<float>::max(), std::numeric_limits<float>::max());

    if (Bias != nullptr) {
        for (size_t m = RangeStartM; m < RangeStartM + RangeCountM; m++) {
            for (size_t n = RangeStartN; n < RangeStartN + RangeCountN; n++) {
                C[m * ldc + n] += Bias[n];
            }
        }
    }
}
#endif

}  // namespace sqnbitgemm_neon
