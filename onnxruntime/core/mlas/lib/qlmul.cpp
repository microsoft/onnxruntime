/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qlmul.cpp

Abstract:

    This module implements routines to quantize linear mul.

    For quantization formula as specified in the ONNX operator documentation is:

        Output = Saturate(RoundToEven(Input / Scale) + ZeroPoint)

--*/

#include "qladd.h"

#if defined(MLAS_NEON_INTRINSICS)

template<typename DataType, bool IsScalarB>
static
void
MlasQLinearMulKernel(
    const DataType* InputA,
    float ScaleA,
    int32_t ZeroPointA,
    const DataType* InputB,
    float ScaleB,
    int32_t ZeroPointB,
    float ScaleC,
    int32_t ZeroPointC,
    DataType* OutputC,
    size_t N
    )
{
    typedef MLAS_SignedUnsignedIntOps<DataType> SUI;

    const float ScaleRatio = ScaleA * ScaleB / ScaleC;
    const float32x4_t VectorScaleRatio = vld1q_dup_f32(&ScaleRatio);
    const typename SUI::i8x8_t VectorZeroPointA = SUI::vmov_n_i8((DataType)ZeroPointA);
    const typename SUI::i8x8_t VectorZeroPointB = SUI::vmov_n_i8((DataType)ZeroPointB);
    const int16x8_t VectorZeroPointC = vmovq_n_s16((int16_t)ZeroPointC);

    int16x8_t vb0_s16x8, vb1_s16x8;
    if (IsScalarB) {
        const typename SUI::i8x8_t VectorB0 = SUI::vmov_n_i8(*InputB);
        vb0_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(VectorB0, VectorZeroPointB));
        vb1_s16x8 = vb0_s16x8;
    }

    while (N >= 16) {
        const typename SUI::i8x16_t VectorA0 = SUI::vld1q_i8(InputA);
        InputA += 16;
        const int16x8_t va0_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_low_i8(VectorA0), VectorZeroPointA));
        const int16x8_t va1_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_high_i8(VectorA0), VectorZeroPointA));

        if (!IsScalarB) {
            const typename SUI::i8x16_t VectorB0 = SUI::vld1q_i8(InputB);
            InputB += 16;
            vb0_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_low_i8(VectorB0), VectorZeroPointB));
            vb1_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_high_i8(VectorB0), VectorZeroPointB));
        }

        int32x4_t vacc0_lo = vmull_s16(vget_low_s16(va0_s16x8), vget_low_s16(vb0_s16x8));
        int32x4_t vacc0_hi = vmull_s16(vget_high_s16(va0_s16x8), vget_high_s16(vb0_s16x8));
        int32x4_t vacc1_lo = vmull_s16(vget_low_s16(va1_s16x8), vget_low_s16(vb1_s16x8));
        int32x4_t vacc1_hi = vmull_s16(vget_high_s16(va1_s16x8), vget_high_s16(vb1_s16x8));
        vacc0_lo = vcvtq_s32_f32(vmulq_f32(VectorScaleRatio, vcvtq_f32_s32(vacc0_lo)));
        vacc0_hi = vcvtq_s32_f32(vmulq_f32(VectorScaleRatio, vcvtq_f32_s32(vacc0_hi)));
        vacc1_lo = vcvtq_s32_f32(vmulq_f32(VectorScaleRatio, vcvtq_f32_s32(vacc1_lo)));
        vacc1_hi = vcvtq_s32_f32(vmulq_f32(VectorScaleRatio, vcvtq_f32_s32(vacc1_hi)));

        // Pack, saturate, and add output zero point.
        const int16x8_t vacc0 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0_lo), vqmovn_s32(vacc0_hi)), VectorZeroPointC);
        const int16x8_t vacc1 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1_lo), vqmovn_s32(vacc1_hi)), VectorZeroPointC);
        typename SUI::i8x16_t vc = SUI::combine_i8_s16(vacc0, vacc1);

        N -= 16;
        SUI::vst1q_i8(OutputC, vc);
        OutputC += 16;
    }

    if (N > 0) {
        typename SUI::T TailDataA[16] = { 0 };
        typename SUI::T TailDataB[16] = { 0 };

        MlasCopyTailBytes((uint8_t*)TailDataA, (const uint8_t*)InputA, N);
        if (!IsScalarB) {
            MlasCopyTailBytes((uint8_t*)TailDataB, (const uint8_t*)InputB, N);
        }

        const typename SUI::i8x16_t VectorA0 = SUI::vld1q_i8(TailDataA);
        const int16x8_t va0_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_low_i8(VectorA0), VectorZeroPointA));
        const int16x8_t va1_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_high_i8(VectorA0), VectorZeroPointA));

        if (!IsScalarB) {
            const typename SUI::i8x16_t VectorB0 = SUI::vld1q_i8(TailDataB);
            vb0_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_low_i8(VectorB0), VectorZeroPointB));
            vb1_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_high_i8(VectorB0), VectorZeroPointB));
        }

        int32x4_t vacc0_lo = vmull_s16(vget_low_s16(va0_s16x8), vget_low_s16(vb0_s16x8));
        int32x4_t vacc0_hi = vmull_s16(vget_high_s16(va0_s16x8), vget_high_s16(vb0_s16x8));
        int32x4_t vacc1_lo = vmull_s16(vget_low_s16(va1_s16x8), vget_low_s16(vb1_s16x8));
        int32x4_t vacc1_hi = vmull_s16(vget_high_s16(va1_s16x8), vget_high_s16(vb1_s16x8));
        vacc0_lo = vcvtq_s32_f32(vmulq_f32(VectorScaleRatio, vcvtq_f32_s32(vacc0_lo)));
        vacc0_hi = vcvtq_s32_f32(vmulq_f32(VectorScaleRatio, vcvtq_f32_s32(vacc0_hi)));
        vacc1_lo = vcvtq_s32_f32(vmulq_f32(VectorScaleRatio, vcvtq_f32_s32(vacc1_lo)));
        vacc1_hi = vcvtq_s32_f32(vmulq_f32(VectorScaleRatio, vcvtq_f32_s32(vacc1_hi)));

        // Pack, saturate, and add output zero point.
        const int16x8_t vacc0 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0_lo), vqmovn_s32(vacc0_hi)), VectorZeroPointC);
        const int16x8_t vacc1 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1_lo), vqmovn_s32(vacc1_hi)), VectorZeroPointC);
        typename SUI::i8x16_t vc = SUI::combine_i8_s16(vacc0, vacc1);

        typename SUI::i8x8_t i8x8 = SUI::vget_low_i8(vc);
        if (N & 8) {
            SUI::vst1_i8(OutputC, i8x8);
            OutputC += 8;
            i8x8 = SUI::vget_high_i8(vc);
        }
        if (N & 4) {
            vst1_lane_u32_ex((uint32_t*)OutputC, SUI::vreinterpret_u32_i8(i8x8), 0, 8);
            OutputC += 4;
            i8x8 = SUI::template vext_i8<4>(i8x8, i8x8);
        }
        if (N & 2) {
            vst1_lane_u16_ex((uint16_t*)OutputC, SUI::vreinterpret_u16_i8(i8x8), 0, 8);
            OutputC += 2;
            i8x8 = SUI::template vext_i8<2>(i8x8, i8x8);
        }
        if (N & 1) {
            SUI::template vst1_lane_i8<0>(OutputC, i8x8);
        }
    }
}

#elif defined(MLAS_SSE2_INTRINSICS)

template<typename DataType, bool IsScalarB>
static
void
MlasQLinearMulKernel(
    const DataType* InputA,
    float ScaleA,
    int32_t ZeroPointA,
    const DataType* InputB,
    float ScaleB,
    int32_t ZeroPointB,
    float ScaleC,
    int32_t ZeroPointC,
    DataType* OutputC,
    size_t N
    )
{
    const auto VectorZeroPointA = _mm_set1_epi16((int16_t)ZeroPointA);
    const auto VectorZeroPointB = _mm_set1_epi16((int16_t)ZeroPointB);
    const auto VectorZeroPointC = MlasBroadcastInt32x4(ZeroPointC);
    const auto VectorScaleRatio = MlasBroadcastFloat32x4(ScaleA * ScaleB / ScaleC);

    MLAS_INT32X4 vb_lo_i16x8, vb_hi_i16x8;
    if (IsScalarB) {
        vb_lo_i16x8 = _mm_sub_epi16(_mm_set1_epi16((int16_t)*InputB), VectorZeroPointB);
        vb_hi_i16x8 = vb_lo_i16x8;
    }

    while (N >= 16) {
        const auto va_i8x16 = _mm_loadu_si128((const MLAS_INT32X4*)InputA);
        InputA += 16;
        const auto va_lo_i16x8 = _mm_sub_epi16(MlasShiftRightInt16<DataType>(_mm_unpacklo_epi8(va_i8x16, va_i8x16), 8), VectorZeroPointA);
        const auto va_hi_i16x8 = _mm_sub_epi16(MlasShiftRightInt16<DataType>(_mm_unpackhi_epi8(va_i8x16, va_i8x16), 8), VectorZeroPointA);

        if (!IsScalarB) {
            const auto vb_i8x16 = _mm_loadu_si128((const MLAS_INT32X4*)InputB);
            InputB += 16;
            vb_lo_i16x8 = _mm_sub_epi16(MlasShiftRightInt16<DataType>(_mm_unpacklo_epi8(vb_i8x16, vb_i8x16), 8), VectorZeroPointB);
            vb_hi_i16x8 = _mm_sub_epi16(MlasShiftRightInt16<DataType>(_mm_unpackhi_epi8(vb_i8x16, vb_i8x16), 8), VectorZeroPointB);
        }

        auto ab_lo_lo = _mm_mullo_epi16(va_lo_i16x8, vb_lo_i16x8);
        auto ab_lo_hi = _mm_mulhi_epi16(va_lo_i16x8, vb_lo_i16x8);
        auto r_lo_lo = _mm_add_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(ab_lo_lo, ab_lo_hi)), VectorScaleRatio)), VectorZeroPointC);
        auto r_lo_hi = _mm_add_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(ab_lo_lo, ab_lo_hi)), VectorScaleRatio)), VectorZeroPointC);
        const auto vc_lo_i16x8 = _mm_packs_epi32(r_lo_lo, r_lo_hi);

        auto ab_hi_lo = _mm_mullo_epi16(va_hi_i16x8, vb_hi_i16x8);
        auto ab_hi_hi = _mm_mulhi_epi16(va_hi_i16x8, vb_hi_i16x8);
        auto r_hi_lo = _mm_add_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(ab_hi_lo, ab_hi_hi)), VectorScaleRatio)), VectorZeroPointC);
        auto r_hi_hi = _mm_add_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(ab_hi_lo, ab_hi_hi)), VectorScaleRatio)), VectorZeroPointC);
        const auto vc_hi_i16x8 = _mm_packs_epi32(r_hi_lo, r_hi_hi);

        auto vc = MlasPackS16_128<DataType>(vc_lo_i16x8, vc_hi_i16x8);
        _mm_storeu_si128((MLAS_INT32X4*)OutputC, vc);
        OutputC += 16;
        N -= 16;
    }

    if (N > 0) {
        uint8_t TailData[16] = { 0 };

        MlasCopyTailBytes(TailData, (const uint8_t*)InputA, N);
        const auto va_i8x16 = _mm_loadu_si128((const MLAS_INT32X4*)TailData);
        const auto va_lo_i16x8 = _mm_sub_epi16(MlasShiftRightInt16<DataType>(_mm_unpacklo_epi8(va_i8x16, va_i8x16), 8), VectorZeroPointA);
        const auto va_hi_i16x8 = _mm_sub_epi16(MlasShiftRightInt16<DataType>(_mm_unpackhi_epi8(va_i8x16, va_i8x16), 8), VectorZeroPointA);

        if (!IsScalarB) {
            MlasCopyTailBytes(TailData, (const uint8_t*)InputB, N);
            const auto vb_i8x16 = _mm_loadu_si128((const MLAS_INT32X4*)TailData);
            vb_lo_i16x8 = _mm_sub_epi16(MlasShiftRightInt16<DataType>(_mm_unpacklo_epi8(vb_i8x16, vb_i8x16), 8), VectorZeroPointB);
            vb_hi_i16x8 = _mm_sub_epi16(MlasShiftRightInt16<DataType>(_mm_unpackhi_epi8(vb_i8x16, vb_i8x16), 8), VectorZeroPointB);
        }

        auto ab_lo_lo = _mm_mullo_epi16(va_lo_i16x8, vb_lo_i16x8);
        auto ab_lo_hi = _mm_mulhi_epi16(va_lo_i16x8, vb_lo_i16x8);
        auto r_lo_lo = _mm_add_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(ab_lo_lo, ab_lo_hi)), VectorScaleRatio)), VectorZeroPointC);
        auto r_lo_hi = _mm_add_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(ab_lo_lo, ab_lo_hi)), VectorScaleRatio)), VectorZeroPointC);
        const auto vc_lo_i16x8 = _mm_packs_epi32(r_lo_lo, r_lo_hi);

        auto ab_hi_lo = _mm_mullo_epi16(va_hi_i16x8, vb_hi_i16x8);
        auto ab_hi_hi = _mm_mulhi_epi16(va_hi_i16x8, vb_hi_i16x8);
        auto r_hi_lo = _mm_add_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(ab_hi_lo, ab_hi_hi)), VectorScaleRatio)), VectorZeroPointC);
        auto r_hi_hi = _mm_add_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(ab_hi_lo, ab_hi_hi)), VectorScaleRatio)), VectorZeroPointC);
        const auto vc_hi_i16x8 = _mm_packs_epi32(r_hi_lo, r_hi_hi);

        auto vc = MlasPackS16_128<DataType>(vc_lo_i16x8, vc_hi_i16x8);

        while (N >= 4) {
            *(int*)OutputC = _mm_cvtsi128_si32(vc);
            N -= 4;
            OutputC += 4;
            vc = _mm_shuffle_epi32(vc, _MM_SHUFFLE(0, 3, 2, 1));
        }

        uint32_t PackedValueC = (uint32_t)_mm_cvtsi128_si32(vc);
        for (size_t i = 0; i < N; ++i) {
            *((uint8_t*)OutputC + i) = (uint8_t)PackedValueC;
            PackedValueC >>= 8;
        }
    }
}

#else

// Pure C++ implementation.
template<typename DataType, bool IsScalarB>
static
void
MlasQLinearMulKernel(
    const DataType* InputA,
    float ScaleA,
    int32_t ZeroPointA,
    const DataType* InputB,
    float ScaleB,
    int32_t ZeroPointB,
    float ScaleC,
    int32_t ZeroPointC,
    DataType* OutputC,
    size_t N
    )
{
    const float MinimumValue = (float)((int)std::numeric_limits<DataType>::min() - ZeroPointC);
    const float MaximumValue = (float)((int)std::numeric_limits<DataType>::max() - ZeroPointC);

    float ValueB;

    if (IsScalarB) {
        ValueB = ScaleB * (int32_t(InputB[0]) - ZeroPointB);
    }

    for (size_t n = 0; n < N; n++) {
        float ValueA = ScaleA * (int32_t(InputA[n]) - ZeroPointA);
        if (!IsScalarB) {
            ValueB = ScaleB * (int32_t(InputB[n]) - ZeroPointB);
        }
        float ValueC = (ValueA * ValueB) / ScaleC;
        ValueC = std::min(std::max(ValueC, MinimumValue), MaximumValue);
        OutputC[n] = (DataType)(int32_t)std::nearbyintf(ValueC + ZeroPointC);
    }
}

#endif

template <typename DataType>
void
MLASCALL
MlasQLinearMul(
    const DataType* InputA,
    float ScaleA,
    int32_t ZeroPointA,
    const DataType* InputB,
    float ScaleB,
    int32_t ZeroPointB,
    float ScaleC,
    int32_t ZeroPointC,
    DataType* OutputC,
    size_t N,
    bool IsScalarB
    )
{
    if (IsScalarB) {
        MlasQLinearMulKernel<DataType, true>(
            InputA, ScaleA, ZeroPointA, InputB, ScaleB, ZeroPointB, ScaleC, ZeroPointC, OutputC, N);
    } else {
        MlasQLinearMulKernel<DataType, false>(
            InputA, ScaleA, ZeroPointA, InputB, ScaleB, ZeroPointB, ScaleC, ZeroPointC, OutputC, N);
    }
}

// Explicit instantiation
template
void
MlasQLinearMul<uint8_t>(
    const uint8_t* InputA,
    float ScaleA,
    int32_t ZeroPointA,
    const uint8_t* InputB,
    float ScaleB,
    int32_t ZeroPointB,
    float ScaleC,
    int32_t ZeroPointC,
    uint8_t* OutputC,
    size_t N,
    bool IsScalarB
    );

template
void
MlasQLinearMul<int8_t>(
    const int8_t* InputA,
    float ScaleA,
    int32_t ZeroPointA,
    const int8_t* InputB,
    float ScaleB,
    int32_t ZeroPointB,
    float ScaleC,
    int32_t ZeroPointC,
    int8_t* OutputC,
    size_t N,
    bool IsScalarB
    );
