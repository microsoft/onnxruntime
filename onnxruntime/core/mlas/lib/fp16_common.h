/*++

Copyright (c) Microsoft Corporation.  All rights reserved.

Licensed under the MIT License.

Module Name:

    fp16_common.h

Abstract:

    Intrinsic and inline functions for fp16 processing.

--*/

#pragma once

#include "mlas_float16.h"
#include "mlasi.h"

#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED

// TODO!! Add intel fp16 implementations

typedef float16x8_t MLAS_FLOAT16X8;
typedef float16x4_t MLAS_FLOAT16X4;
typedef uint16x8_t MLAS_UINT16X8;
typedef uint16x4_t MLAS_UINT16X4;

MLAS_FORCEINLINE
MLAS_FLOAT16X8
MlasReinterpretAsFloat16x8(MLAS_INT32X4 Vector) { return vreinterpretq_f16_s32(Vector); }

MLAS_FORCEINLINE
MLAS_FLOAT16X8
MlasBroadcastFloat16x8(_mlas_fp16_ Value) { return vreinterpretq_f16_p16(vdupq_n_p16(Value)); }

MLAS_FORCEINLINE
MLAS_FLOAT16X4
MlasBroadcastFloat16x4(_mlas_fp16_ Value) { return vreinterpret_f16_p16(vdup_n_p16(Value)); }

MLAS_FORCEINLINE
MLAS_FLOAT16X8
MlasBroadcastFloat16x8(const _mlas_fp16_* Value) { return vreinterpretq_f16_u16(vld1q_dup_u16(Value)); }

MLAS_FORCEINLINE
MLAS_FLOAT16X4
MlasBroadcastFloat16x4(const _mlas_fp16_* Value) { return vreinterpret_f16_u16(vld1_dup_u16(Value)); }

MLAS_FORCEINLINE
MLAS_FLOAT16X8
MlasZeroFloat16x8(void) { return vreinterpretq_f16_f32(vdupq_n_f32(0.0f)); }

MLAS_FORCEINLINE
MLAS_FLOAT16X4
MlasZeroFloat16x4(void) { return vreinterpret_f16_f32(vdup_n_f32(0.0f)); }

MLAS_FORCEINLINE
MLAS_FLOAT16X8
MlasLoadFloat16x8(const _mlas_fp16_* Buffer) { return vreinterpretq_f16_u16(vld1q_u16(Buffer)); }

MLAS_FORCEINLINE
MLAS_FLOAT16X4
MlasLoadFloat16x4(const _mlas_fp16_* Buffer) { return vreinterpret_f16_u16(vld1_u16(Buffer)); }

MLAS_FORCEINLINE
void
MlasStoreFloat16x8(_mlas_fp16_* Buffer, MLAS_FLOAT16X8 Vector)
{
    vst1q_u16(Buffer, vreinterpretq_u16_f16(Vector));
}

MLAS_FORCEINLINE
void
MlasStoreFloat16x4(_mlas_fp16_* Buffer, MLAS_FLOAT16X4 Vector)
{
    vst1_u16(Buffer, vreinterpret_u16_f16(Vector));
}

MLAS_FORCEINLINE
void
MlasStorePartialFloat16x4(_mlas_fp16_* Buffer, MLAS_FLOAT16X4 Vector, size_t len)
{
    if ((len & 2) != 0) {
        vst1_lane_f32(reinterpret_cast<float*>(Buffer), vreinterpret_f32_f16(Vector), 0);
        Vector = vreinterpret_f16_f32(vdup_lane_f32(vreinterpret_f32_f16(Vector), 1));
        Buffer += 2;
    }
    if ((len & 1) != 0) {
        vst1_lane_u16(Buffer, vreinterpret_u16_f16(Vector), 0);
    }
}

template <unsigned Lane>
MLAS_FORCEINLINE void
MlasStoreLaneFloat16x8(_mlas_fp16_* Buffer, MLAS_FLOAT16X8 Vector)
{
    vst1q_lane_u16(Buffer, vreinterpretq_u16_f16(Vector), Lane);
}

MLAS_FORCEINLINE MLAS_FLOAT16X4
MlasToLowHalfFloat16x4(MLAS_FLOAT16X8 V)
{
    // vget_low should be compiled to nothing
    return vget_low_f16(V);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X8
MlasAddFloat16x8(MLAS_FLOAT16X8 Vector1, MLAS_FLOAT16X8 Vector2)
{
    return vaddq_f16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X4
MlasAddFloat16x4(MLAS_FLOAT16X4 Vector1, MLAS_FLOAT16X4 Vector2)
{
    return vadd_f16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X8
MlasSubtractFloat16x8(MLAS_FLOAT16X8 Vector1, MLAS_FLOAT16X8 Vector2)
{
    return vsubq_f16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X4
MlasSubtractFloat16x4(MLAS_FLOAT16X4 Vector1, MLAS_FLOAT16X4 Vector2)
{
    return vsub_f16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X8
MlasMultiplyFloat16x8(MLAS_FLOAT16X8 Vector1, MLAS_FLOAT16X8 Vector2)
{
    return vmulq_f16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X4
MlasMultiplyFloat16x4(MLAS_FLOAT16X4 Vector1, MLAS_FLOAT16X4 Vector2)
{
    return vmul_f16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X8
MlasDivFloat16x8(MLAS_FLOAT16X8 Vector1, MLAS_FLOAT16X8 Vector2)
{
    return vdivq_f16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X4
MlasDivFloat16x4(MLAS_FLOAT16X4 Vector1, MLAS_FLOAT16X4 Vector2)
{
    return vdiv_f16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X8
MlasMultiplyAddFloat16x8(MLAS_FLOAT16X8 Vector1, MLAS_FLOAT16X8 Vector2, MLAS_FLOAT16X8 Vector3)
{
    return vfmaq_f16(Vector3, Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X4
MlasMultiplyAddFloat16x4(MLAS_FLOAT16X4 Vector1, MLAS_FLOAT16X4 Vector2, MLAS_FLOAT16X4 Vector3)
{
    return vfma_f16(Vector3, Vector1, Vector2);
}


MLAS_FORCEINLINE
void
MlasMultiplyAddFloat16x8(MLAS_FLOAT16X8 Vector1, _mlas_fp16_ Scalar2, MLAS_FLOAT16X8 Vector3)
{
    MlasMultiplyAddFloat16x8(Vector1, MlasBroadcastFloat16x8(Scalar2), Vector3);
}

MLAS_FORCEINLINE
void
MlasMultiplyAddFloat16x8(MLAS_FLOAT16X8 Vector1, MLAS_FLOAT16X8 Vector2, _mlas_fp16_ Scalar3)
{
    MlasMultiplyAddFloat16x8(Vector1, Vector2, MlasBroadcastFloat16x8(Scalar3));
}

MLAS_FORCEINLINE
MLAS_FLOAT16X8
MlasDivideFloat16x8(MLAS_FLOAT16X8 Vector1, MLAS_FLOAT16X8 Vector2)
{
    return vdivq_f16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X8
MlasGreaterThanFloat16x8(MLAS_FLOAT16X8 Vector1, MLAS_FLOAT16X8 Vector2)
{
    return vreinterpretq_f16_u16(vcgtq_f16(Vector1, Vector2));
}

MLAS_FORCEINLINE
MLAS_FLOAT16X8
MlasAndFloat16x8(MLAS_FLOAT16X8 Vector1, MLAS_FLOAT16X8 Vector2)
{
    return vreinterpretq_f16_s64(vandq_s64(vreinterpretq_s64_f16(Vector1), vreinterpretq_s64_f16(Vector2)));
}

MLAS_FORCEINLINE
MLAS_FLOAT16X8
MlasOrFloat16x8(MLAS_FLOAT16X8 Vector1, MLAS_FLOAT16X8 Vector2)
{
    return vreinterpretq_f16_s64(vorrq_s64(vreinterpretq_s64_f16(Vector1), vreinterpretq_s64_f16(Vector2)));
}

MLAS_FORCEINLINE
MLAS_FLOAT16X8
MlasAndNotFloat16x8(MLAS_FLOAT16X8 VectorNot, MLAS_FLOAT16X8 Vector)
{
    return vreinterpretq_f16_s32(vandq_s32(vmvnq_s32(vreinterpretq_s32_f16(VectorNot)), vreinterpretq_s32_f16(Vector)));
}

MLAS_FORCEINLINE
MLAS_FLOAT16X8
MlasXorFloat16x8(MLAS_FLOAT16X8 Vector1, MLAS_FLOAT16X8 Vector2)
{
    return vreinterpretq_f16_s32(veorq_s32(vreinterpretq_s32_f16(Vector1), vreinterpretq_s32_f16(Vector2)));
}

MLAS_FORCEINLINE
MLAS_FLOAT16X8
MlasBlendFloat16x8(MLAS_FLOAT16X8 Vector1, MLAS_FLOAT16X8 Vector2, MLAS_FLOAT16X8 Selection)
{
    return MlasOrFloat16x8(MlasAndFloat16x8(Vector2, Selection),
                           MlasAndNotFloat16x8(Selection, Vector1));
}

MLAS_FORCEINLINE
MLAS_FLOAT16X8
MlasMaximumFloat16x8(MLAS_FLOAT16X8 Vector1, MLAS_FLOAT16X8 Vector2)
{
    return vmaxq_f16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X4
MlasMaximumFloat16x4(MLAS_FLOAT16X4 Vector1, MLAS_FLOAT16X4 Vector2)
{
    return vmax_f16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X8
MlasMinimumFloat16x8(MLAS_FLOAT16X8 Vector1, MLAS_FLOAT16X8 Vector2)
{
    return vminq_f16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X4
MlasMinimumFloat16x4(MLAS_FLOAT16X4 Vector1, MLAS_FLOAT16X4 Vector2)
{
    return vmin_f16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X8
MlasClampFloat16x8(MLAS_FLOAT16X8 Value, _mlas_fp16_ LowerRange, _mlas_fp16_ UpperRange)
{
    Value = MlasMaximumFloat16x8(MlasBroadcastFloat16x8(LowerRange), Value);
    Value = MlasMinimumFloat16x8(MlasBroadcastFloat16x8(UpperRange), Value);
    return Value;
}

MLAS_FORCEINLINE
_mlas_fp16_
MlasReduceAddFloat16x8(MLAS_FLOAT16X8 Vector)
{
    Vector = vpaddq_f16(Vector, Vector);
    Vector = vpaddq_f16(Vector, Vector);
    return vgetq_lane_u16(vreinterpretq_u16_f16(Vector), 0);
}

MLAS_FORCEINLINE
MLAS_UINT16X8
MlasCmpLessEqualFloat16x8(MLAS_FLOAT16X8 left, MLAS_FLOAT16X8 right)
{
    return vcleq_f16(left, right);
}

MLAS_FORCEINLINE
MLAS_UINT16X4
MlasCmpLessEqualFloat16x4(MLAS_FLOAT16X4 left, MLAS_FLOAT16X4 right)
{
    return vcle_f16(left, right);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X8
MlasBitwiseSelectFloat16x8(MLAS_UINT16X8 select, MLAS_FLOAT16X8 ones, MLAS_FLOAT16X8 zeros)
{
    return vbslq_f16(select, ones, zeros);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X4
MlasBitwiseSelectFloat16x4(MLAS_UINT16X4 select, MLAS_FLOAT16X4 ones, MLAS_FLOAT16X4 zeros)
{
    return vbsl_f16(select, ones, zeros);
}

#endif  // fp16 vector intrinsic supported
