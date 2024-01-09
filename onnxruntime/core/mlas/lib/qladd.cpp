/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qladd.cpp

Abstract:

    This module implements routines to quantize linear add.

    For quantization formula as specified in the ONNX operator documentation is:

        Output = Saturate(RoundToEven(Input / Scale) + ZeroPoint)

--*/

#include "qladd.h"

// Pure C++ helper, back off here in rare case.
template<typename DataType, bool IsScalarB>
MLAS_FORCEINLINE
static
void
MlasQLinearAddKernelRawHelper(
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
        float ValueC = (ValueA + ValueB) / ScaleC;
        ValueC = std::min(std::max(ValueC, MinimumValue), MaximumValue);
        OutputC[n] = (DataType)(int32_t)std::nearbyintf(ValueC + ZeroPointC);
    }
}

#if defined(MLAS_NEON_INTRINSICS)

bool MlasCalcQLinearAddParameters(
    float ScaleRatio_AC,
    float ScaleRatio_BC,
    int32_t& Shift,
    int32_t& MultiplierA,
    int32_t& MultiplierB) {
  constexpr float MinScaleRatio = 6.103515625e-05f;  // std::stof("0x1.0p-14f");
  constexpr float MaxScaleRatio = 256.0f;            //std::stof("0x1.0p+8f");
  if (ScaleRatio_AC < MinScaleRatio || ScaleRatio_AC >= MaxScaleRatio ||
      ScaleRatio_BC < MinScaleRatio || ScaleRatio_BC >= MaxScaleRatio) {
    return false;
  }

  const float GreaterScaleRatio = std::max(ScaleRatio_AC, ScaleRatio_BC);
  const int32_t GreaterExponent = (int32_t)(MlasBitsOfFp32(GreaterScaleRatio) >> 23) - 127;
  Shift = 21 - GreaterExponent;
  if (Shift > 31 || Shift < 13) return false;

  const float MultiplierFloatValue = MlasFp32FromBits((uint32_t)(21 - GreaterExponent + 127) << 23);
  MultiplierA = (int32_t)lrintf(ScaleRatio_AC * MultiplierFloatValue);
  MultiplierB = (int32_t)lrintf(ScaleRatio_BC * MultiplierFloatValue);
  return ((MultiplierA < 0x00400000 && MultiplierB < 0x00400000) &&
          (MultiplierA >= 0x00200000 || MultiplierB >= 0x00200000));  // the greater one must fullfil this check
}

template<typename DataType, bool IsScalarB>
static
void
MlasQLinearAddKernelHelper(
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

    int32_t Shift, MultiplierA, MultiplierB;
    const float ScaleRatio_AC = ScaleA / ScaleC;
    const float ScaleRatio_BC = ScaleB / ScaleC;
    if (!MlasCalcQLinearAddParameters(ScaleRatio_AC, ScaleRatio_BC, Shift, MultiplierA, MultiplierB)) {
        MlasQLinearAddKernelRawHelper<DataType, IsScalarB>(
            InputA, ScaleA, ZeroPointA, InputB, ScaleB, ZeroPointB, ScaleC, ZeroPointC, OutputC, N);
        return;
    }

    const int32x4_t VectorMultiplierA = vld1q_dup_s32(&MultiplierA);
    const int32x4_t VectorMultiplierB = vld1q_dup_s32(&MultiplierB);
    const typename SUI::i8x8_t VectorZeroPointA = SUI::vmov_n_i8((DataType)ZeroPointA);
    const typename SUI::i8x8_t VectorZeroPointB = SUI::vmov_n_i8((DataType)ZeroPointB);
    const int16x8_t VectorZeroPointC = vmovq_n_s16((int16_t)ZeroPointC);
    const int32x4_t vright_shift = vmovq_n_s32(-Shift); // vld1q_dup_s32(&right_shift);
    const int32x4_t vzero_shift_mask = vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));

    int32x4_t vscalar;
    if (IsScalarB) {
        const typename SUI::i8x8_t VectorB0 = SUI::vmov_n_i8(*InputB);
        const int16x8_t vb_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(VectorB0, VectorZeroPointB));
        vscalar = vmulq_s32(vmovl_s16(vget_low_s16(vb_s16x8)), VectorMultiplierB);
    }

#if defined(MLAS_NEON64_INTRINSICS)

    while (N >= 32) {
        int32x4_t vacc0_lo, vacc0_hi, vacc1_lo, vacc1_hi, vacc2_lo, vacc2_hi, vacc3_lo, vacc3_hi;
        if (IsScalarB) {
            const typename SUI::i8x16_t VectorA0 = SUI::vld1q_i8(InputA);
            const typename SUI::i8x16_t VectorA1 = SUI::vld1q_i8(InputA + 16);
            InputA += 32;
            const int16x8_t va0_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_low_i8(VectorA0), VectorZeroPointA));
            const int16x8_t va1_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_high_i8(VectorA0), VectorZeroPointA));
            const int16x8_t va2_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_low_i8(VectorA1), VectorZeroPointA));
            const int16x8_t va3_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_high_i8(VectorA1), VectorZeroPointA));

            vacc0_lo = vmlaq_s32(vscalar, vmovl_s16(vget_low_s16(va0_s16x8)), VectorMultiplierA);
            vacc1_lo = vmlaq_s32(vscalar, vmovl_s16(vget_low_s16(va1_s16x8)), VectorMultiplierA);
            vacc2_lo = vmlaq_s32(vscalar, vmovl_s16(vget_low_s16(va2_s16x8)), VectorMultiplierA);
            vacc3_lo = vmlaq_s32(vscalar, vmovl_s16(vget_low_s16(va3_s16x8)), VectorMultiplierA);
            vacc0_hi = vmlaq_s32(vscalar, MlasMoveHighS16S32(va0_s16x8), VectorMultiplierA);
            vacc1_hi = vmlaq_s32(vscalar, MlasMoveHighS16S32(va1_s16x8), VectorMultiplierA);
            vacc2_hi = vmlaq_s32(vscalar, MlasMoveHighS16S32(va2_s16x8), VectorMultiplierA);
            vacc3_hi = vmlaq_s32(vscalar, MlasMoveHighS16S32(va3_s16x8), VectorMultiplierA);
        } else  {
            const typename SUI::i8x16_t VectorA0 = SUI::vld1q_i8(InputA);
            const typename SUI::i8x16_t VectorB0 = SUI::vld1q_i8(InputB);
            const typename SUI::i8x16_t VectorA1 = SUI::vld1q_i8(InputA + 16);
            const typename SUI::i8x16_t VectorB1 = SUI::vld1q_i8(InputB + 16);
            InputA += 32;
            InputB += 32;
            const int16x8_t va0_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_low_i8(VectorA0), VectorZeroPointA));
            const int16x8_t vb0_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_low_i8(VectorB0), VectorZeroPointB));
            const int16x8_t va1_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_high_i8(VectorA0), VectorZeroPointA));
            const int16x8_t vb1_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_high_i8(VectorB0), VectorZeroPointB));
            const int16x8_t va2_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_low_i8(VectorA1), VectorZeroPointA));
            const int16x8_t vb2_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_low_i8(VectorB1), VectorZeroPointB));
            const int16x8_t va3_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_high_i8(VectorA1), VectorZeroPointA));
            const int16x8_t vb3_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_high_i8(VectorB1), VectorZeroPointB));

            vacc0_lo = vmulq_s32(vmovl_s16(vget_low_s16(va0_s16x8)), VectorMultiplierA);
            vacc1_lo = vmulq_s32(vmovl_s16(vget_low_s16(va1_s16x8)), VectorMultiplierA);
            vacc2_lo = vmulq_s32(vmovl_s16(vget_low_s16(va2_s16x8)), VectorMultiplierA);
            vacc3_lo = vmulq_s32(vmovl_s16(vget_low_s16(va3_s16x8)), VectorMultiplierA);
            vacc0_hi = vmulq_s32(MlasMoveHighS16S32(va0_s16x8), VectorMultiplierA);
            vacc1_hi = vmulq_s32(MlasMoveHighS16S32(va1_s16x8), VectorMultiplierA);
            vacc2_hi = vmulq_s32(MlasMoveHighS16S32(va2_s16x8), VectorMultiplierA);
            vacc3_hi = vmulq_s32(MlasMoveHighS16S32(va3_s16x8), VectorMultiplierA);

            vacc0_lo = vmlaq_s32(vacc0_lo, vmovl_s16(vget_low_s16(vb0_s16x8)), VectorMultiplierB);
            vacc1_lo = vmlaq_s32(vacc1_lo, vmovl_s16(vget_low_s16(vb1_s16x8)), VectorMultiplierB);
            vacc2_lo = vmlaq_s32(vacc2_lo, vmovl_s16(vget_low_s16(vb2_s16x8)), VectorMultiplierB);
            vacc3_lo = vmlaq_s32(vacc3_lo, vmovl_s16(vget_low_s16(vb3_s16x8)), VectorMultiplierB);
            vacc0_hi = vmlaq_s32(vacc0_hi, MlasMoveHighS16S32(vb0_s16x8), VectorMultiplierB);
            vacc1_hi = vmlaq_s32(vacc1_hi, MlasMoveHighS16S32(vb1_s16x8), VectorMultiplierB);
            vacc2_hi = vmlaq_s32(vacc2_hi, MlasMoveHighS16S32(vb2_s16x8), VectorMultiplierB);
            vacc3_hi = vmlaq_s32(vacc3_hi, MlasMoveHighS16S32(vb3_s16x8), VectorMultiplierB);
        }

        vacc0_lo = vsraq_n_s32(vacc0_lo, vbicq_s32(vacc0_lo, vzero_shift_mask), 31);
        vacc1_lo = vsraq_n_s32(vacc1_lo, vbicq_s32(vacc1_lo, vzero_shift_mask), 31);
        vacc2_lo = vsraq_n_s32(vacc2_lo, vbicq_s32(vacc2_lo, vzero_shift_mask), 31);
        vacc3_lo = vsraq_n_s32(vacc3_lo, vbicq_s32(vacc3_lo, vzero_shift_mask), 31);
        vacc0_hi = vsraq_n_s32(vacc0_hi, vbicq_s32(vacc0_hi, vzero_shift_mask), 31);
        vacc1_hi = vsraq_n_s32(vacc1_hi, vbicq_s32(vacc1_hi, vzero_shift_mask), 31);
        vacc2_hi = vsraq_n_s32(vacc2_hi, vbicq_s32(vacc2_hi, vzero_shift_mask), 31);
        vacc3_hi = vsraq_n_s32(vacc3_hi, vbicq_s32(vacc3_hi, vzero_shift_mask), 31);

        vacc0_lo = vrshlq_s32(vacc0_lo, vright_shift);
        vacc1_lo = vrshlq_s32(vacc1_lo, vright_shift);
        vacc2_lo = vrshlq_s32(vacc2_lo, vright_shift);
        vacc3_lo = vrshlq_s32(vacc3_lo, vright_shift);
        vacc0_hi = vrshlq_s32(vacc0_hi, vright_shift);
        vacc1_hi = vrshlq_s32(vacc1_hi, vright_shift);
        vacc2_hi = vrshlq_s32(vacc2_hi, vright_shift);
        vacc3_hi = vrshlq_s32(vacc3_hi, vright_shift);

        // Pack, saturate, and add output zero point.
        const int16x8_t vacc0 = vqaddq_s16(MlasCombineS16S32(vacc0_lo, vacc0_hi), VectorZeroPointC);
        const int16x8_t vacc1 = vqaddq_s16(MlasCombineS16S32(vacc1_lo, vacc1_hi), VectorZeroPointC);
        const int16x8_t vacc2 = vqaddq_s16(MlasCombineS16S32(vacc2_lo, vacc2_hi), VectorZeroPointC);
        const int16x8_t vacc3 = vqaddq_s16(MlasCombineS16S32(vacc3_lo, vacc3_hi), VectorZeroPointC);

        const typename SUI::i8x16_t vc0 = SUI::combine_i8_s16(vacc0, vacc1);
        const typename SUI::i8x16_t vc1 = SUI::combine_i8_s16(vacc2, vacc3);

        SUI::vst1q_i8(OutputC, vc0);
        SUI::vst1q_i8(OutputC + 16, vc1);
        N -= 32;
        OutputC += 32;
    }

#endif

    while (N >= 16) {
        int32x4_t vacc0_lo, vacc1_lo, vacc0_hi, vacc1_hi;
        if (IsScalarB) {
            const typename SUI::i8x16_t VectorA0 = SUI::vld1q_i8(InputA);
            InputA += 16;
            const int16x8_t va0_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_low_i8(VectorA0), VectorZeroPointA));
            const int16x8_t va1_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_high_i8(VectorA0), VectorZeroPointA));

            vacc0_lo = vmlaq_s32(vscalar, vmovl_s16(vget_low_s16(va0_s16x8)), VectorMultiplierA);
            vacc1_lo = vmlaq_s32(vscalar, vmovl_s16(vget_low_s16(va1_s16x8)), VectorMultiplierA);
            vacc0_hi = vmlaq_s32(vscalar, MlasMoveHighS16S32(va0_s16x8), VectorMultiplierA);
            vacc1_hi = vmlaq_s32(vscalar, MlasMoveHighS16S32(va1_s16x8), VectorMultiplierA);
        } else  {
            const typename SUI::i8x16_t VectorA0 = SUI::vld1q_i8(InputA);
            const typename SUI::i8x16_t VectorB0 = SUI::vld1q_i8(InputB);
            InputA += 16;
            InputB += 16;
            const int16x8_t va0_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_low_i8(VectorA0), VectorZeroPointA));
            const int16x8_t vb0_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_low_i8(VectorB0), VectorZeroPointB));
            const int16x8_t va1_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_high_i8(VectorA0), VectorZeroPointA));
            const int16x8_t vb1_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_high_i8(VectorB0), VectorZeroPointB));

            vacc0_lo = vmulq_s32(vmovl_s16(vget_low_s16(va0_s16x8)), VectorMultiplierA);
            vacc1_lo = vmulq_s32(vmovl_s16(vget_low_s16(va1_s16x8)), VectorMultiplierA);
            vacc0_hi = vmulq_s32(MlasMoveHighS16S32(va0_s16x8), VectorMultiplierA);
            vacc1_hi = vmulq_s32(MlasMoveHighS16S32(va1_s16x8), VectorMultiplierA);

            vacc0_lo = vmlaq_s32(vacc0_lo, vmovl_s16(vget_low_s16(vb0_s16x8)), VectorMultiplierB);
            vacc1_lo = vmlaq_s32(vacc1_lo, vmovl_s16(vget_low_s16(vb1_s16x8)), VectorMultiplierB);
            vacc0_hi = vmlaq_s32(vacc0_hi, MlasMoveHighS16S32(vb0_s16x8), VectorMultiplierB);
            vacc1_hi = vmlaq_s32(vacc1_hi, MlasMoveHighS16S32(vb1_s16x8), VectorMultiplierB);
        }

        vacc0_lo = vsraq_n_s32(vacc0_lo, vbicq_s32(vacc0_lo, vzero_shift_mask), 31);
        vacc1_lo = vsraq_n_s32(vacc1_lo, vbicq_s32(vacc1_lo, vzero_shift_mask), 31);
        vacc0_hi = vsraq_n_s32(vacc0_hi, vbicq_s32(vacc0_hi, vzero_shift_mask), 31);
        vacc1_hi = vsraq_n_s32(vacc1_hi, vbicq_s32(vacc1_hi, vzero_shift_mask), 31);

        vacc0_lo = vrshlq_s32(vacc0_lo, vright_shift);
        vacc1_lo = vrshlq_s32(vacc1_lo, vright_shift);
        vacc0_hi = vrshlq_s32(vacc0_hi, vright_shift);
        vacc1_hi = vrshlq_s32(vacc1_hi, vright_shift);

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

        int32x4_t vacc0_lo, vacc1_lo, vacc0_hi, vacc1_hi;
        if (IsScalarB) {
            const typename SUI::i8x16_t VectorA0 = SUI::vld1q_i8(TailDataA);
            const int16x8_t va0_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_low_i8(VectorA0), VectorZeroPointA));
            const int16x8_t va1_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_high_i8(VectorA0), VectorZeroPointA));

            vacc0_lo = vmlaq_s32(vscalar, vmovl_s16(vget_low_s16(va0_s16x8)), VectorMultiplierA);
            vacc1_lo = vmlaq_s32(vscalar, vmovl_s16(vget_low_s16(va1_s16x8)), VectorMultiplierA);
            vacc0_hi = vmlaq_s32(vscalar, MlasMoveHighS16S32(va0_s16x8), VectorMultiplierA);
            vacc1_hi = vmlaq_s32(vscalar, MlasMoveHighS16S32(va1_s16x8), VectorMultiplierA);
        } else  {
            const typename SUI::i8x16_t VectorA0 = SUI::vld1q_i8(TailDataA);
            const typename SUI::i8x16_t VectorB0 = SUI::vld1q_i8(TailDataB);
            const int16x8_t va0_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_low_i8(VectorA0), VectorZeroPointA));
            const int16x8_t vb0_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_low_i8(VectorB0), VectorZeroPointB));
            const int16x8_t va1_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_high_i8(VectorA0), VectorZeroPointA));
            const int16x8_t vb1_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_high_i8(VectorB0), VectorZeroPointB));

            vacc0_lo = vmulq_s32(vmovl_s16(vget_low_s16(va0_s16x8)), VectorMultiplierA);
            vacc1_lo = vmulq_s32(vmovl_s16(vget_low_s16(va1_s16x8)), VectorMultiplierA);
            vacc0_hi = vmulq_s32(MlasMoveHighS16S32(va0_s16x8), VectorMultiplierA);
            vacc1_hi = vmulq_s32(MlasMoveHighS16S32(va1_s16x8), VectorMultiplierA);

            vacc0_lo = vmlaq_s32(vacc0_lo, vmovl_s16(vget_low_s16(vb0_s16x8)), VectorMultiplierB);
            vacc1_lo = vmlaq_s32(vacc1_lo, vmovl_s16(vget_low_s16(vb1_s16x8)), VectorMultiplierB);
            vacc0_hi = vmlaq_s32(vacc0_hi, MlasMoveHighS16S32(vb0_s16x8), VectorMultiplierB);
            vacc1_hi = vmlaq_s32(vacc1_hi, MlasMoveHighS16S32(vb1_s16x8), VectorMultiplierB);
        }

        vacc0_lo = vsraq_n_s32(vacc0_lo, vbicq_s32(vacc0_lo, vzero_shift_mask), 31);
        vacc1_lo = vsraq_n_s32(vacc1_lo, vbicq_s32(vacc1_lo, vzero_shift_mask), 31);
        vacc0_hi = vsraq_n_s32(vacc0_hi, vbicq_s32(vacc0_hi, vzero_shift_mask), 31);
        vacc1_hi = vsraq_n_s32(vacc1_hi, vbicq_s32(vacc1_hi, vzero_shift_mask), 31);

        vacc0_lo = vrshlq_s32(vacc0_lo, vright_shift);
        vacc1_lo = vrshlq_s32(vacc1_lo, vright_shift);
        vacc0_hi = vrshlq_s32(vacc0_hi, vright_shift);
        vacc1_hi = vrshlq_s32(vacc1_hi, vright_shift);

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
MlasQLinearAddKernelHelper(
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
    const float ScaleRatio_AC = ScaleA / ScaleC;
    const float ScaleRatio_BC = ScaleB / ScaleC;
    const auto VectorScaleRatio_AC = MlasBroadcastFloat32x4(ScaleRatio_AC);
    const auto VectorScaleRatio_BC = MlasBroadcastFloat32x4(ScaleRatio_BC);
    auto VectorFixedPart = MlasBroadcastFloat32x4((float)ZeroPointC - (ScaleRatio_AC * ZeroPointA + ScaleRatio_BC * ZeroPointB));

    MLAS_FLOAT32X4 va_lo, va_hi, vb_lo, vb_hi;
    if (IsScalarB) {
        vb_lo = _mm_set1_ps((float)*InputB);
        VectorFixedPart = _mm_add_ps(VectorFixedPart, _mm_mul_ps(vb_lo, VectorScaleRatio_BC));
    }

    while (N >= 8) {
        const auto va_low_half = _mm_loadl_epi64((const MLAS_INT32X4*)InputA);
        const auto va_i16x8 = _mm_unpacklo_epi8(va_low_half, va_low_half);
        InputA += 8;
        va_lo = _mm_cvtepi32_ps(MlasShiftRightInt32<DataType>(_mm_unpacklo_epi16(va_i16x8, va_i16x8), 24));
        va_hi = _mm_cvtepi32_ps(MlasShiftRightInt32<DataType>(_mm_unpackhi_epi16(va_i16x8, va_i16x8), 24));

        if (!IsScalarB) {
            const auto vb_low_half = _mm_loadl_epi64((const MLAS_INT32X4*)InputB);
            const auto vb_i16x8 = _mm_unpacklo_epi8(vb_low_half, vb_low_half);
            InputB += 8;
            vb_lo = _mm_cvtepi32_ps(MlasShiftRightInt32<DataType>(_mm_unpacklo_epi16(vb_i16x8, vb_i16x8), 24));
            vb_hi = _mm_cvtepi32_ps(MlasShiftRightInt32<DataType>(_mm_unpackhi_epi16(vb_i16x8, vb_i16x8), 24));
        }

        MLAS_INT32X4 r_lo, r_hi;
        if (IsScalarB) {
            r_lo = _mm_cvtps_epi32(_mm_add_ps(VectorFixedPart, _mm_mul_ps(va_lo, VectorScaleRatio_AC)));
            r_hi = _mm_cvtps_epi32(_mm_add_ps(VectorFixedPart, _mm_mul_ps(va_hi, VectorScaleRatio_AC)));
        } else {
            r_lo = _mm_cvtps_epi32(_mm_add_ps(_mm_add_ps(VectorFixedPart, _mm_mul_ps(va_lo, VectorScaleRatio_AC)), _mm_mul_ps(vb_lo, VectorScaleRatio_BC)));
            r_hi = _mm_cvtps_epi32(_mm_add_ps(_mm_add_ps(VectorFixedPart, _mm_mul_ps(va_hi, VectorScaleRatio_AC)), _mm_mul_ps(vb_hi, VectorScaleRatio_BC)));
        }
        const auto vc_i16x8 = _mm_packs_epi32(r_lo, r_hi);
        MLAS_INT32X4 vc = MlasPackS16_128<DataType>(vc_i16x8, vc_i16x8);

        N -= 8;
        _mm_storel_epi64((MLAS_INT32X4*)OutputC, vc);
        OutputC += 8;
    }

    if (N > 0) {
        uint8_t TailData[8] = { 0 };

        MlasCopyTailBytes(TailData, (const uint8_t*)InputA, N);
        const auto va_low_half = _mm_loadl_epi64((const MLAS_INT32X4*)TailData);
        const auto va_i16x8 = _mm_unpacklo_epi8(va_low_half, va_low_half);
        va_lo = _mm_cvtepi32_ps(MlasShiftRightInt32<DataType>(_mm_unpacklo_epi16(va_i16x8, va_i16x8), 24));
        va_hi = _mm_cvtepi32_ps(MlasShiftRightInt32<DataType>(_mm_unpackhi_epi16(va_i16x8, va_i16x8), 24));

        if (!IsScalarB) {
            MlasCopyTailBytes(TailData, (const uint8_t*)InputB, N);
            const auto vb_low_half = _mm_loadl_epi64((const MLAS_INT32X4*)TailData);
            const auto vb_i16x8 = _mm_unpacklo_epi8(vb_low_half, vb_low_half);
            vb_lo = _mm_cvtepi32_ps(MlasShiftRightInt32<DataType>(_mm_unpacklo_epi16(vb_i16x8, vb_i16x8), 24));
            vb_hi = _mm_cvtepi32_ps(MlasShiftRightInt32<DataType>(_mm_unpackhi_epi16(vb_i16x8, vb_i16x8), 24));
        }

        MLAS_INT32X4 r_lo, r_hi;
        if (IsScalarB) {
            r_lo = _mm_cvtps_epi32(_mm_add_ps(VectorFixedPart, _mm_mul_ps(va_lo, VectorScaleRatio_AC)));
            r_hi = _mm_cvtps_epi32(_mm_add_ps(VectorFixedPart, _mm_mul_ps(va_hi, VectorScaleRatio_AC)));
        } else {
            r_lo = _mm_cvtps_epi32(_mm_add_ps(_mm_add_ps(VectorFixedPart, _mm_mul_ps(va_lo, VectorScaleRatio_AC)), _mm_mul_ps(vb_lo, VectorScaleRatio_BC)));
            r_hi = _mm_cvtps_epi32(_mm_add_ps(_mm_add_ps(VectorFixedPart, _mm_mul_ps(va_hi, VectorScaleRatio_AC)), _mm_mul_ps(vb_hi, VectorScaleRatio_BC)));
        }
        const auto vc_i16x8 = _mm_packs_epi32(r_lo, r_hi);
        MLAS_INT32X4 vc = MlasPackS16_128<DataType>(vc_i16x8, vc_i16x8);

        if (N & 4) {
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
#elif defined(MLAS_TARGET_POWER)
template<typename DataType, bool IsScalarB>
static
void
MlasQLinearAddKernelHelper(
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
    if (N >= 16) {
        float ScaleRatio_AC = ScaleA / ScaleC;
        float ScaleRatio_BC = ScaleB / ScaleC;
        MLAS_FLOAT32X4 VectorScaleRatio_AC = MlasBroadcastFloat32x4(ScaleRatio_AC);
        MLAS_FLOAT32X4 VectorScaleRatio_BC = MlasBroadcastFloat32x4(ScaleRatio_BC);
        MLAS_FLOAT32X4 VectorFixedPart = MlasBroadcastFloat32x4((float)ZeroPointC - (ScaleRatio_AC * ZeroPointA + ScaleRatio_BC * ZeroPointB));
        MLAS_FLOAT32X4 vb0_lo, vb0_hi, vb1_lo, vb1_hi;
        const uint8_t flip = 128;
        MLAS_UNREFERENCED_PARAMETER(flip);
        __vector unsigned char vmask = reinterpret_cast<__vector unsigned char>(vec_splats(flip));
        __vector signed short vmask1 = reinterpret_cast<__vector signed short>(vec_splats((short)flip));

        if (IsScalarB) {
            vb0_lo = MlasBroadcastFloat32x4((float)*InputB);
            VectorFixedPart = vec_add(VectorFixedPart, vec_mul(vb0_lo, VectorScaleRatio_BC));
        }
        while (N >= 16) {
            MLAS_INT32X4 r_lo, r_hi;
            MLAS_FLOAT32X4 va_lo, va_hi;
            MLAS_UNREFERENCED_PARAMETER(VectorScaleRatio_AC);
            MLAS_UNREFERENCED_PARAMETER(VectorScaleRatio_BC);
            auto va = MlasPackL8<DataType>(InputA, vmask);
            auto vshort = vec_unpackh(va);
            vshort = MlasPackS16<DataType>(vshort, vmask1);
            auto va1 = vec_unpackl(vshort);
            auto va0 = vec_unpackh(vshort);
            va_lo = vec_ctf(va0, 0);
            va_hi = vec_ctf(va1, 0);
            if (!IsScalarB) {
                auto vb = MlasPackL8<DataType>(InputB, vmask);
                vshort = vec_unpackh(vb);
                vshort = MlasPackS16<DataType>(vshort, vmask1);
                auto vb1 = vec_unpackl(vshort);
                auto vb0 = vec_unpackh(vshort);
                vb0_lo = vec_ctf(vb0, 0);
                vb0_hi= vec_ctf(vb1, 0);
                vshort = vec_unpackl(vb);
                vshort = MlasPackS16<DataType>(vshort, vmask1);
                vb1 = vec_unpackl(vshort);
                vb0 = vec_unpackh(vshort);
                vb1_lo = vec_ctf(vb0, 0);
                vb1_hi= vec_ctf(vb1, 0);
                InputB += 16;
            }
            va_lo = vec_mul(va_lo, VectorScaleRatio_AC);
            va_hi = vec_mul(va_hi, VectorScaleRatio_AC);
            if (IsScalarB) {
                r_lo = vec_cts(vec_round(vec_add(VectorFixedPart, va_lo)), 0);
                r_hi = vec_cts(vec_round(vec_add(VectorFixedPart, va_hi)), 0);
            } else {
                vb0_lo = vec_mul(vb0_lo, VectorScaleRatio_BC);
                vb0_hi = vec_mul(vb0_hi, VectorScaleRatio_BC);
                r_lo = vec_cts(vec_round(vec_add(vec_add(VectorFixedPart, va_lo), vb0_lo)), 0);
                r_hi = vec_cts(vec_round(vec_add(vec_add(VectorFixedPart, va_hi), vb0_hi)), 0);
            }
            const auto vc0 = vec_packs(r_lo, r_hi);
            vshort = vec_unpackl(va);
            vshort = MlasPackS16<DataType>(vshort, vmask1);
            va1 = vec_unpackl(vshort);
            va0 = vec_unpackh(vshort);
            va_lo = vec_ctf(va0, 0);
            va_hi = vec_ctf(va1, 0);
            va_lo = vec_mul(va_lo, VectorScaleRatio_AC);
            va_hi = vec_mul(va_hi, VectorScaleRatio_AC);
            if (IsScalarB) {
                r_lo = vec_cts(vec_round(vec_add(VectorFixedPart, va_lo)), 0);
                r_hi = vec_cts(vec_round(vec_add(VectorFixedPart, va_hi)), 0);
            } else {
                vb1_lo = vec_mul(vb1_lo, VectorScaleRatio_BC);
                vb1_hi = vec_mul(vb1_hi, VectorScaleRatio_BC);
                r_lo = vec_cts(vec_round(vec_add(vec_add(VectorFixedPart, va_lo), vb1_lo)), 0);
                r_hi = vec_cts(vec_round(vec_add(vec_add(VectorFixedPart, va_hi), vb1_hi)), 0);
            }
            const auto vc1 = vec_packs(r_lo, r_hi);
            MLAS_INT32X4 vc = MlasPackS16_128<DataType>(vc0, vc1);
            vec_vsx_st(vc, 0, reinterpret_cast<MLAS_INT32X4*>(OutputC));
            N -= 16;
            InputA += 16;
            OutputC += 16;
        }
    }
    if (N > 0) {
        MlasQLinearAddKernelRawHelper<DataType, IsScalarB>(
          InputA, ScaleA, ZeroPointA, InputB, ScaleB, ZeroPointB, ScaleC, ZeroPointC, OutputC, N);
    }
}
#elif defined(MLAS_LSX_INTRINSICS)

template<typename DataType, bool IsScalarB>
static
void
MlasQLinearAddKernelHelper(
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
    const float ScaleRatio_AC = ScaleA / ScaleC;
    const float ScaleRatio_BC = ScaleB / ScaleC;
    const auto VectorScaleRatio_AC = MlasBroadcastFloat32x4(ScaleRatio_AC);
    const auto VectorScaleRatio_BC = MlasBroadcastFloat32x4(ScaleRatio_BC);
    auto VectorFixedPart = MlasBroadcastFloat32x4((float)ZeroPointC - (ScaleRatio_AC * ZeroPointA + ScaleRatio_BC * ZeroPointB));

    MLAS_FLOAT32X4 va_lo, va_hi, vb_lo, vb_hi;
    if (IsScalarB) {
        float tmp_f = (float)*InputB;
        uint32_t *tmp_p = (uint32_t *)&tmp_f;
        vb_lo = MlasReinterpretAsFloat32x4(__lsx_vreplgr2vr_w(*tmp_p));
        VectorFixedPart = __lsx_vfmadd_s(vb_lo, VectorScaleRatio_BC, VectorFixedPart);
    }

    __m128i tmp, tmp1;

    while (N >= 8) {
        const auto va_low_half = __lsx_vinsgr2vr_d(__lsx_vld((const MLAS_INT32X4*)InputA, 0), 0 ,1);
        const auto va_i16x8 = __lsx_vilvl_b(va_low_half, va_low_half);
        InputA += 8;
        va_lo = __lsx_vffint_s_w(MlasShiftRightInt32<DataType>(__lsx_vilvl_h(va_i16x8, va_i16x8), 24));
        va_hi = __lsx_vffint_s_w(MlasShiftRightInt32<DataType>(__lsx_vilvh_h(va_i16x8, va_i16x8), 24));

        if (!IsScalarB) {
            const auto vb_low_half = __lsx_vinsgr2vr_d(__lsx_vld((const MLAS_INT32X4*)InputB, 0), 0 ,1);
            const auto vb_i16x8 = __lsx_vilvl_b(vb_low_half, vb_low_half);
            InputB += 8;
            vb_lo = __lsx_vffint_s_w(MlasShiftRightInt32<DataType>(__lsx_vilvl_h(vb_i16x8, vb_i16x8), 24));
            vb_hi = __lsx_vffint_s_w(MlasShiftRightInt32<DataType>(__lsx_vilvh_h(vb_i16x8, vb_i16x8), 24));
        }

        MLAS_INT32X4 r_lo, r_hi;
        if (IsScalarB) {
            r_lo = __lsx_vftint_w_s(__lsx_vfmadd_s(va_lo, VectorScaleRatio_AC, VectorFixedPart));
            r_hi = __lsx_vftint_w_s(__lsx_vfmadd_s(va_hi, VectorScaleRatio_AC, VectorFixedPart));
        } else {
            r_lo = __lsx_vftint_w_s(__lsx_vfadd_s(__lsx_vfmadd_s(va_lo, VectorScaleRatio_AC, VectorFixedPart), __lsx_vfmul_s(vb_lo, VectorScaleRatio_BC)));
            r_hi = __lsx_vftint_w_s(__lsx_vfadd_s(__lsx_vfmadd_s(va_hi, VectorScaleRatio_AC, VectorFixedPart), __lsx_vfmul_s(vb_hi, VectorScaleRatio_BC)));
        }
        tmp = __lsx_vsat_w(r_lo, 15);
        tmp1 = __lsx_vsat_w(r_hi, 15);
         const auto vc_i16x8 = __lsx_vpickev_h(tmp1, tmp);

        MLAS_INT32X4 vc = MlasPackS16_128<DataType>(vc_i16x8, vc_i16x8);

        N -= 8;
        __lsx_vst(__lsx_vinsgr2vr_d(__lsx_vld((MLAS_INT32X4*)OutputC, 0), __lsx_vpickve2gr_d(vc, 0), 0), (MLAS_INT32X4*)OutputC, 0);
        OutputC += 8;
    }

    if (N > 0) {
        uint8_t TailData[8] = { 0 };

        MlasCopyTailBytes(TailData, (const uint8_t*)InputA, N);
        const auto va_low_half = __lsx_vinsgr2vr_d(__lsx_vld((const MLAS_INT32X4*)TailData, 0), 0 ,1);
        const auto va_i16x8 = __lsx_vilvl_b(va_low_half, va_low_half);
        va_lo = __lsx_vffint_s_w(MlasShiftRightInt32<DataType>(__lsx_vilvl_h(va_i16x8, va_i16x8), 24));
        va_hi = __lsx_vffint_s_w(MlasShiftRightInt32<DataType>(__lsx_vilvh_h(va_i16x8, va_i16x8), 24));

        if (!IsScalarB) {
            MlasCopyTailBytes(TailData, (const uint8_t*)InputB, N);
            const auto vb_low_half = __lsx_vinsgr2vr_d(__lsx_vld((const MLAS_INT32X4*)TailData, 0), 0 ,1);
            const auto vb_i16x8 = __lsx_vilvl_b(vb_low_half, vb_low_half);
            vb_lo = __lsx_vffint_s_w(MlasShiftRightInt32<DataType>(__lsx_vilvl_h(vb_i16x8, vb_i16x8), 24));
            vb_hi = __lsx_vffint_s_w(MlasShiftRightInt32<DataType>(__lsx_vilvh_h(vb_i16x8, vb_i16x8), 24));
        }

        MLAS_INT32X4 r_lo, r_hi;
        if (IsScalarB) {
            r_lo = __lsx_vftint_w_s(__lsx_vfmadd_s(va_lo, VectorScaleRatio_AC, VectorFixedPart));
            r_hi = __lsx_vftint_w_s(__lsx_vfmadd_s(va_hi, VectorScaleRatio_AC, VectorFixedPart));
        } else {
            r_lo = __lsx_vftint_w_s(__lsx_vfadd_s(__lsx_vfmadd_s(va_lo, VectorScaleRatio_AC, VectorFixedPart), __lsx_vfmul_s(vb_lo, VectorScaleRatio_BC)));
            r_hi = __lsx_vftint_w_s(__lsx_vfadd_s(__lsx_vfmadd_s(va_hi, VectorScaleRatio_AC, VectorFixedPart), __lsx_vfmul_s(vb_hi, VectorScaleRatio_BC)));
        }
        tmp = __lsx_vsat_w(r_lo, 15);
        tmp1 = __lsx_vsat_w(r_hi, 15);
        const auto vc_i16x8 = __lsx_vpickev_h(tmp1, tmp);

        MLAS_INT32X4 vc = MlasPackS16_128<DataType>(vc_i16x8, vc_i16x8);

        if (N & 4) {
            __lsx_vstelm_w(vc, (int*)OutputC, 0, 0);
            N -= 4;
            OutputC += 4;
            vc = __lsx_vshuf4i_w(vc, 0x39); //_MM_SHUFFLE(0, 3, 2, 1)
        }

        uint32_t PackedValueC = (uint32_t)__lsx_vpickve2gr_w(vc, 0);
        for (size_t i = 0; i < N; ++i) {
            *((uint8_t*)OutputC + i) = (uint8_t)PackedValueC;
            PackedValueC >>= 8;
        }
    }
}
#else

template<typename DataType, bool IsScalarB>
static
void
MlasQLinearAddKernelHelper(
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
    // Pure C++ implementation.
    MlasQLinearAddKernelRawHelper<DataType, IsScalarB>(
        InputA, ScaleA, ZeroPointA, InputB, ScaleB, ZeroPointB, ScaleC, ZeroPointC, OutputC, N);
}

#endif

template<typename DataType>
static
void
MLASCALL
MlasQLinearAddKernel(
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
        MlasQLinearAddKernelHelper<DataType, true>(
            InputA, ScaleA, ZeroPointA, InputB, ScaleB, ZeroPointB, ScaleC, ZeroPointC, OutputC, N);
    } else {
        MlasQLinearAddKernelHelper<DataType, false>(
            InputA, ScaleA, ZeroPointA, InputB, ScaleB, ZeroPointB, ScaleC, ZeroPointC, OutputC, N);
    }
}

template<>
void
MLASCALL
MlasQLinearAdd<int8_t>(
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
    )
{
#if defined(MLAS_TARGET_AMD64)
        GetMlasPlatform().QLinearAddS8Kernel(
#else
        MlasQLinearAddKernel<int8_t>(
#endif
            InputA, ScaleA, ZeroPointA, InputB, ScaleB, ZeroPointB, ScaleC, ZeroPointC, OutputC, N, IsScalarB);
}

template<>
void
MLASCALL
MlasQLinearAdd<uint8_t>(
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
    )
{
#if defined(MLAS_TARGET_AMD64)
        GetMlasPlatform().QLinearAddU8Kernel(
#else
        MlasQLinearAddKernel<uint8_t>(
#endif
            InputA, ScaleA, ZeroPointA, InputB, ScaleB, ZeroPointB, ScaleC, ZeroPointC, OutputC, N, IsScalarB);
}

//
// Function definition for platform usage
//

void
MLASCALL
MlasQLinearAddS8Kernel(
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
    )
{
    MlasQLinearAddKernel<int8_t>(
        InputA, ScaleA, ZeroPointA, InputB, ScaleB, ZeroPointB, ScaleC, ZeroPointC, OutputC, N, IsScalarB);
}

void
MLASCALL
MlasQLinearAddU8Kernel(
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
    )
{
    MlasQLinearAddKernel<uint8_t>(
        InputA, ScaleA, ZeroPointA, InputB, ScaleB, ZeroPointB, ScaleC, ZeroPointC, OutputC, N, IsScalarB);
}
