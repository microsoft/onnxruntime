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

#if defined(MLAS_NEON64_INTRINSICS)

template <typename SUI, bool IsLow>
MLAS_FORCEINLINE
static
int16x8_t
MlasExtendToS16Debias(
    typename SUI::i8x16_t Int8Vector,
    typename SUI::i8x8_t VectorBias
    )
{
    auto HalfVector = IsLow ? SUI::vget_low_i8(Int8Vector) : SUI::vget_high_i8(Int8Vector);
    return SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(HalfVector, VectorBias));
}

MLAS_FORCEINLINE
static
int16x8_t
MlasQLinearMulVectorS16(
    int16x8_t va_s16x8,
    int16x8_t vb_s16x8,
    float32x4_t VectorScaleRatio,
    float32x4_t VectorZeroPointC
    )
{
    int32x4_t vacc0_lo = vmull_s16(vget_low_s16(va_s16x8), vget_low_s16(vb_s16x8));
    int32x4_t vacc0_hi = vmull_s16(vget_high_s16(va_s16x8), vget_high_s16(vb_s16x8));
    auto vacc0_lo_f32 = vaddq_f32(VectorZeroPointC, vmulq_f32(VectorScaleRatio, vcvtq_f32_s32(vacc0_lo)));
    auto vacc0_hi_f32 = vaddq_f32(VectorZeroPointC, vmulq_f32(VectorScaleRatio, vcvtq_f32_s32(vacc0_hi)));
    // using rounding to nearst, ties to even
    vacc0_lo = vcvtnq_s32_f32(vacc0_lo_f32);
    vacc0_hi = vcvtnq_s32_f32(vacc0_hi_f32);
    // Pack and saturate.
    return vcombine_s16(vqmovn_s32(vacc0_lo), vqmovn_s32(vacc0_hi));
}

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

    const float32x4_t VectorScaleRatio = vmovq_n_f32(ScaleA * ScaleB / ScaleC);
    const typename SUI::i8x8_t VectorZeroPointA = SUI::vmov_n_i8((DataType)ZeroPointA);
    const typename SUI::i8x8_t VectorZeroPointB = SUI::vmov_n_i8((DataType)ZeroPointB);
    const float32x4_t VectorZeroPointC = vmovq_n_f32((float)ZeroPointC);

    typename SUI::T TailDataA[16] = { 0 };
    typename SUI::T TailDataB[16] = { 0 };
    int16x8_t vb0_s16x8, vb1_s16x8;
    if (IsScalarB) {
        const typename SUI::i8x8_t VectorB0 = SUI::vmov_n_i8(*InputB);
        vb0_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(VectorB0, VectorZeroPointB));
        vb1_s16x8 = vb0_s16x8;
    }

    while (N > 0) {
        if (N < 16) {
            MlasCopyTailBytes((uint8_t*)TailDataA, (const uint8_t*)InputA, N);
            InputA = (const DataType*)TailDataA;
            if (!IsScalarB) {
                MlasCopyTailBytes((uint8_t*)TailDataB, (const uint8_t*)InputB, N);
                InputB = (const DataType*)TailDataB;
            }
        }

        const typename SUI::i8x16_t VectorA0 = SUI::vld1q_i8(InputA);
        InputA += 16;
        const int16x8_t va0_s16x8 = MlasExtendToS16Debias<SUI, /* IsLow = */ true>(VectorA0, VectorZeroPointA);
        const int16x8_t va1_s16x8 = MlasExtendToS16Debias<SUI, /* IsLow = */ false>(VectorA0, VectorZeroPointA);;

        if (!IsScalarB) {
            const typename SUI::i8x16_t VectorB0 = SUI::vld1q_i8(InputB);
            InputB += 16;
            vb0_s16x8 = MlasExtendToS16Debias<SUI, /* IsLow = */ true>(VectorB0, VectorZeroPointB);
            vb1_s16x8 = MlasExtendToS16Debias<SUI, /* IsLow = */ false>(VectorB0, VectorZeroPointB);
        }

        const int16x8_t vacc0 = MlasQLinearMulVectorS16(va0_s16x8, vb0_s16x8, VectorScaleRatio, VectorZeroPointC);
        const int16x8_t vacc1 = MlasQLinearMulVectorS16(va1_s16x8, vb1_s16x8, VectorScaleRatio, VectorZeroPointC);
        typename SUI::i8x16_t vc = SUI::combine_i8_s16(vacc0, vacc1);

        if (N >= 16) {
            N -= 16;
            SUI::vst1q_i8(OutputC, vc);
            OutputC += 16;
        } else {
            SUI::vst1q_i8(TailDataA, vc);
            MlasCopyTailBytes((uint8_t*)OutputC, (const uint8_t*)TailDataA, N);
            N = 0;
        }
    }
}

#elif defined(MLAS_SSE2_INTRINSICS)

template <class DataType, bool IsLow>
MLAS_FORCEINLINE
static
__m128i
MlasExtendToS16(
    __m128i Int8Vector,
    __m128i ZeroVector
    );

template <>
MLAS_FORCEINLINE
__m128i
MlasExtendToS16<uint8_t, /* bool IsLow = */ true>(
    __m128i Int8Vector,
    __m128i ZeroVector
    )
{
    return _mm_unpacklo_epi8(Int8Vector, ZeroVector);
}

template <>
MLAS_FORCEINLINE
__m128i
MlasExtendToS16<uint8_t, /* bool IsLow = */ false>(
    __m128i Int8Vector,
    __m128i ZeroVector
    )
{
    return _mm_unpackhi_epi8(Int8Vector, ZeroVector);
}

template <>
MLAS_FORCEINLINE
__m128i
MlasExtendToS16<int8_t, /* bool IsLow = */ true>(
    __m128i Int8Vector,
    __m128i ZeroVector
    )
{
    MLAS_UNREFERENCED_PARAMETER(ZeroVector);
    return _mm_srai_epi16(_mm_unpacklo_epi8(Int8Vector, Int8Vector), 8);
}

template <>
MLAS_FORCEINLINE
__m128i
MlasExtendToS16<int8_t, /* bool IsLow = */ false>(
    __m128i Int8Vector,
    __m128i ZeroVector
    )
{
    MLAS_UNREFERENCED_PARAMETER(ZeroVector);
    return _mm_srai_epi16(_mm_unpackhi_epi8(Int8Vector, Int8Vector), 8);
}

template <class DataType, bool IsLow>
MLAS_FORCEINLINE
static
__m128i
MlasExtendToS16Debias(
    __m128i Int8Vector,
    __m128i ZeroVector,
    __m128i VectorBias
    )
{
    return _mm_sub_epi16(MlasExtendToS16<DataType, IsLow>(Int8Vector, ZeroVector), VectorBias);
}

MLAS_FORCEINLINE
static
__m128i
MlasQLinearMulVectorS16(
    __m128i va_s16x8,
    __m128i vb_s16x8,
    __m128 VectorScaleRatio,
    __m128 VectorZeroPointC
    )
{
    const auto ab_lo = _mm_mullo_epi16(va_s16x8, vb_s16x8);
    const auto ab_hi = _mm_mulhi_epi16(va_s16x8, vb_s16x8);
    auto r_lo = _mm_unpacklo_epi16(ab_lo, ab_hi);
    auto r_hi = _mm_unpackhi_epi16(ab_lo, ab_hi);
    r_lo = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(r_lo), VectorScaleRatio), VectorZeroPointC));
    r_hi = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(r_hi), VectorScaleRatio), VectorZeroPointC));
    return _mm_packs_epi32(r_lo, r_hi);
}

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
    const auto VectorZeroPointC = MlasBroadcastFloat32x4((float)ZeroPointC);
    const auto VectorScaleRatio = MlasBroadcastFloat32x4(ScaleA * ScaleB / ScaleC);
    const auto ZeroVector = _mm_setzero_si128();

    uint8_t TailDataA[16] = { 0 };
    uint8_t TailDataB[16] = { 0 };
    __m128i vb_lo_s16x8, vb_hi_s16x8;

    if (IsScalarB) {
        vb_lo_s16x8 = _mm_sub_epi16(_mm_set1_epi16((int16_t)*InputB), VectorZeroPointB);
        vb_hi_s16x8 = vb_lo_s16x8;
    }

    while (N > 0) {
        if (N < 16) {
            MlasCopyTailBytes(TailDataA, (const uint8_t*)InputA, N);
            InputA = (const DataType*)TailDataA;
            if (!IsScalarB) {
                MlasCopyTailBytes(TailDataB, (const uint8_t*)InputB, N);
                InputB = (const DataType*)TailDataB;
            }
        }

        const auto va_i8x16 = _mm_loadu_si128((const MLAS_INT32X4*)InputA);
        InputA += 16;
        const auto va_lo_s16x8 = MlasExtendToS16Debias<DataType, true>(va_i8x16, ZeroVector, VectorZeroPointA);
        const auto va_hi_s16x8 = MlasExtendToS16Debias<DataType, false>(va_i8x16, ZeroVector, VectorZeroPointA);

        if (!IsScalarB) {
            const auto vb_i8x16 = _mm_loadu_si128((const MLAS_INT32X4*)InputB);
            InputB += 16;
            vb_lo_s16x8 = MlasExtendToS16Debias<DataType, true>(vb_i8x16, ZeroVector, VectorZeroPointB);
            vb_hi_s16x8 = MlasExtendToS16Debias<DataType, false>(vb_i8x16, ZeroVector, VectorZeroPointB);
        }

        const auto vc_lo_s16x8 = MlasQLinearMulVectorS16(va_lo_s16x8, vb_lo_s16x8, VectorScaleRatio, VectorZeroPointC);
        const auto vc_hi_s16x8 = MlasQLinearMulVectorS16(va_hi_s16x8, vb_hi_s16x8, VectorScaleRatio, VectorZeroPointC);
        auto vc = MlasPackS16_128<DataType>(vc_lo_s16x8, vc_hi_s16x8);

        if (N >= 16) {
            _mm_storeu_si128((__m128i*)OutputC, vc);
            OutputC += 16;
            N -= 16;
        } else {
            _mm_storeu_si128((__m128i*)TailDataA, vc);
            MlasCopyTailBytes((uint8_t*)OutputC, TailDataA, N);
            N = 0;
        }
    }
}

#elif defined(MLAS_VSX_INTRINSICS)

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

    auto ZeroPointAVector = vec_splats(int32_t(ZeroPointA));
    auto ZeroPointBVector = vec_splats(int32_t(ZeroPointB));
    auto ZeroPointCVector = vec_splats(float(ZeroPointC));

    auto ScaleAVector = vec_splats(ScaleA);
    auto ScaleBVector = vec_splats(ScaleB);
    auto ScaleCVector = vec_splats(ScaleC);

    auto MinimumVector = vec_splats(MinimumValue);
    auto MaximumVector = vec_splats(MaximumValue);

    float ValueB;
    __vector float ValueBVector;

    if (IsScalarB) {
        ValueB = ScaleB * (int32_t(InputB[0]) - ZeroPointB);
        ValueBVector = vec_splats(ValueB);
    }

    while (N >= 4) {
        __vector int32_t IntegerAVector {InputA[0], InputA[1], InputA[2], InputA[3]};
        auto IntegerVector = vec_sub(IntegerAVector, ZeroPointAVector);
        auto ValueAVector = vec_mul(ScaleAVector, vec_ctf(IntegerVector, 0));

        if (!IsScalarB) {
            __vector int32_t IntegerBVector {InputB[0], InputB[1], InputB[2], InputB[3]};
            IntegerVector = vec_sub(IntegerBVector, ZeroPointBVector);
            ValueBVector = vec_mul(ScaleBVector, vec_ctf(IntegerVector, 0));
        }

        auto ValueCVector = vec_div(vec_mul(ValueAVector, ValueBVector), ScaleCVector);
        ValueCVector = vec_min(vec_max(ValueCVector, MinimumVector), MaximumVector);
        ValueCVector = vec_nearbyint(vec_add(ValueCVector, ZeroPointCVector));

        auto IntegerValueCVector = vec_signed(ValueCVector);
        OutputC[0] = (DataType) IntegerValueCVector[0];
        OutputC[1] = (DataType) IntegerValueCVector[1];
        OutputC[2] = (DataType) IntegerValueCVector[2];
        OutputC[3] = (DataType) IntegerValueCVector[3];

        OutputC += 4;
        InputA += 4;
        InputB += 4;

        N -= 4;

        // Suppress wrong GCC warnings
        MLAS_UNREFERENCED_PARAMETER(ValueAVector);
    }

    while (N > 0) {
        float ValueA = ScaleA * (int32_t(*InputA) - ZeroPointA);
        if (!IsScalarB) {
            ValueB = ScaleB * (int32_t(*InputB) - ZeroPointB);
        }
        float ValueC = (ValueA * ValueB) / ScaleC;
        ValueC = std::min(std::max(ValueC, MinimumValue), MaximumValue);

        *OutputC = (DataType)(int32_t)std::nearbyintf(ValueC + ZeroPointC);

        InputA++;
        InputB++;
        OutputC++;
        N--;
    }

    // Suppress wrong GCC warnings
    MLAS_UNREFERENCED_PARAMETER(ScaleAVector);
    MLAS_UNREFERENCED_PARAMETER(ScaleBVector);
    MLAS_UNREFERENCED_PARAMETER(ValueBVector);
}

#elif defined(MLAS_LSX_INTRINSICS)

template <class DataType, bool IsLow>
MLAS_FORCEINLINE
static
__m128i
MlasExtendToS16(
    __m128i Int8Vector,
    __m128i ZeroVector
    );

template <>
MLAS_FORCEINLINE
__m128i
MlasExtendToS16<uint8_t, /* bool IsLow = */ true>(
    __m128i Int8Vector,
    __m128i ZeroVector
    )
{
    return __lsx_vilvl_b(ZeroVector, Int8Vector);
}

template <>
MLAS_FORCEINLINE
__m128i
MlasExtendToS16<uint8_t, /* bool IsLow = */ false>(
    __m128i Int8Vector,
    __m128i ZeroVector
    )
{
    return __lsx_vilvh_b(ZeroVector, Int8Vector);
}

template <>
MLAS_FORCEINLINE
__m128i
MlasExtendToS16<int8_t, /* bool IsLow = */ true>(
    __m128i Int8Vector,
    __m128i ZeroVector
    )
{
    MLAS_UNREFERENCED_PARAMETER(ZeroVector);
    return __lsx_vsrai_h(__lsx_vilvl_b(Int8Vector, Int8Vector), 8);
}

template <>
MLAS_FORCEINLINE
__m128i
MlasExtendToS16<int8_t, /* bool IsLow = */ false>(
    __m128i Int8Vector,
    __m128i ZeroVector
    )
{
    MLAS_UNREFERENCED_PARAMETER(ZeroVector);
    return __lsx_vsrai_h(__lsx_vilvh_b(Int8Vector, Int8Vector), 8);
}

template <class DataType, bool IsLow>
MLAS_FORCEINLINE
static
__m128i
MlasExtendToS16Debias(
    __m128i Int8Vector,
    __m128i ZeroVector,
    __m128i VectorBias
    )
{
    return __lsx_vsub_h(MlasExtendToS16<DataType, IsLow>(Int8Vector, ZeroVector), VectorBias);
}

MLAS_FORCEINLINE
static
__m128i
MlasQLinearMulVectorS16(
    __m128i va_s16x8,
    __m128i vb_s16x8,
    __m128 VectorScaleRatio,
    __m128 VectorZeroPointC
    )
{
    __m128i tmp, tmp1;

    const auto ab_lo = __lsx_vmul_h(va_s16x8, vb_s16x8);
    const auto ab_hi = __lsx_vmuh_h(va_s16x8, vb_s16x8);
    auto r_lo = __lsx_vilvl_h(ab_hi, ab_lo);
    auto r_hi = __lsx_vilvh_h(ab_hi, ab_lo);
    r_lo = __lsx_vftint_w_s(__lsx_vfmadd_s(__lsx_vffint_s_w(r_lo), VectorScaleRatio, VectorZeroPointC));
    r_hi = __lsx_vftint_w_s(__lsx_vfmadd_s(__lsx_vffint_s_w(r_hi), VectorScaleRatio, VectorZeroPointC));

    tmp = __lsx_vsat_w(r_lo, 15);
    tmp1 = __lsx_vsat_w(r_hi, 15);
    return __lsx_vpickev_h(tmp1, tmp);
}

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
    const auto VectorZeroPointA = __lsx_vreplgr2vr_h((int16_t)ZeroPointA);
    const auto VectorZeroPointB = __lsx_vreplgr2vr_h((int16_t)ZeroPointB);
    const auto VectorZeroPointC = MlasBroadcastFloat32x4((float)ZeroPointC);
    const auto VectorScaleRatio = MlasBroadcastFloat32x4(ScaleA * ScaleB / ScaleC);
    const auto ZeroVector = __lsx_vldi(0);

    uint8_t TailDataA[16] = { 0 };
    uint8_t TailDataB[16] = { 0 };
    __m128i vb_lo_s16x8, vb_hi_s16x8;

    if (IsScalarB) {
        vb_lo_s16x8 = __lsx_vsub_h(__lsx_vreplgr2vr_h((int16_t)*InputB), VectorZeroPointB);
        vb_hi_s16x8 = vb_lo_s16x8;
    }

    while (N > 0) {
        if (N < 16) {
            MlasCopyTailBytes(TailDataA, (const uint8_t*)InputA, N);
            InputA = (const DataType*)TailDataA;
            if (!IsScalarB) {
                MlasCopyTailBytes(TailDataB, (const uint8_t*)InputB, N);
                InputB = (const DataType*)TailDataB;
            }
        }

        const auto va_i8x16 = __lsx_vld((const MLAS_INT32X4*)InputA, 0);
        InputA += 16;
        const auto va_lo_s16x8 = MlasExtendToS16Debias<DataType, true>(va_i8x16, ZeroVector, VectorZeroPointA);
        const auto va_hi_s16x8 = MlasExtendToS16Debias<DataType, false>(va_i8x16, ZeroVector, VectorZeroPointA);

        if (!IsScalarB) {
            const auto vb_i8x16 = __lsx_vld((const MLAS_INT32X4*)InputB, 0);
            InputB += 16;
            vb_lo_s16x8 = MlasExtendToS16Debias<DataType, true>(vb_i8x16, ZeroVector, VectorZeroPointB);
            vb_hi_s16x8 = MlasExtendToS16Debias<DataType, false>(vb_i8x16, ZeroVector, VectorZeroPointB);
        }

        const auto vc_lo_s16x8 = MlasQLinearMulVectorS16(va_lo_s16x8, vb_lo_s16x8, VectorScaleRatio, VectorZeroPointC);
        const auto vc_hi_s16x8 = MlasQLinearMulVectorS16(va_hi_s16x8, vb_hi_s16x8, VectorScaleRatio, VectorZeroPointC);
        auto vc = MlasPackS16_128<DataType>(vc_lo_s16x8, vc_hi_s16x8);

        if (N >= 16) {
            __lsx_vst(vc, (__m128i*)OutputC, 0);
            OutputC += 16;
            N -= 16;
        } else {
            __lsx_vst(vc, (__m128i*)TailDataA, 0);
            MlasCopyTailBytes((uint8_t*)OutputC, TailDataA, N);
            N = 0;
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
