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
typedef int16x8_t MLAS_INT16X8;
typedef int16x4_t MLAS_INT16X4;

MLAS_FORCEINLINE
MLAS_FLOAT16X8
MlasReinterpretInt32AsFloat16(MLAS_INT32X4 Vector) { return vreinterpretq_f16_s32(Vector); }

MLAS_FORCEINLINE
MLAS_FLOAT16X8
MlasReinterpretInt16AsFloat16(MLAS_INT16X8 Vector) { return vreinterpretq_f16_s16(Vector); }

MLAS_FORCEINLINE
MLAS_FLOAT16X4
MlasReinterpretInt16AsFloat16(MLAS_INT16X4 Vector) { return vreinterpret_f16_s16(Vector); }

MLAS_FORCEINLINE
MLAS_INT16X8
MlasReinterpretFloat16AsInt16(MLAS_FLOAT16X8 Vector) { return vreinterpretq_s16_f16(Vector); }

MLAS_FORCEINLINE
MLAS_INT16X4
MlasReinterpretFloat16AsInt16(MLAS_FLOAT16X4 Vector) { return vreinterpret_s16_f16(Vector); }

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

template <int lane>
MLAS_FORCEINLINE
MLAS_FLOAT16X4
MlasLoadLaneFloat16x4(const _mlas_fp16_* Buffer, MLAS_FLOAT16X4 vec) {
    return vreinterpret_f16_u16(
        vld1_lane_u16(Buffer, vreinterpret_u16_f16(vec), lane)
    );
}

MLAS_FORCEINLINE
MLAS_FLOAT16X4
MlasLoadPartialFloat16x4(const _mlas_fp16_* Buffer, size_t len)
{
    MLAS_FLOAT16X4 Vector = MlasZeroFloat16x4();
    if ((len & 1) != 0) {
        Vector = vreinterpret_f16_u16(vld1_lane_u16(Buffer + (len - 1), vreinterpret_u16_f16(Vector), 0));
    }
    if ((len & 2) != 0) {
        Vector = vreinterpret_f16_f32(vdup_lane_f32(vreinterpret_f32_f16(Vector), 0));
        Vector = vreinterpret_f16_f32(
            vld1_lane_f32(reinterpret_cast<const float*>(Buffer), vreinterpret_f32_f16(Vector), 0)
        );
    }
    return Vector;
}

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

template <int lane>
MLAS_FORCEINLINE
void
MlasStoreLaneFloat16x4(_mlas_fp16_* Buffer, MLAS_FLOAT16X4 Vector)
{
    vst1_lane_u16(Buffer, vreinterpret_u16_f16(Vector), lane);
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
MlasAddFloat16(MLAS_FLOAT16X8 Vector1, MLAS_FLOAT16X8 Vector2)
{
    return vaddq_f16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X4
MlasAddFloat16(MLAS_FLOAT16X4 Vector1, MLAS_FLOAT16X4 Vector2)
{
    return vadd_f16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_INT16X8
MlasAddInt16(MLAS_INT16X8 Vector1, MLAS_INT16X8 Vector2)
{
    return vaddq_s16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_INT16X4
MlasAddInt16(MLAS_INT16X4 Vector1, MLAS_INT16X4 Vector2)
{
    return vadd_s16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X8
MlasSubtractFloat16(MLAS_FLOAT16X8 Vector1, MLAS_FLOAT16X8 Vector2)
{
    return vsubq_f16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X4
MlasSubtractFloat16(MLAS_FLOAT16X4 Vector1, MLAS_FLOAT16X4 Vector2)
{
    return vsub_f16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_INT16X8
MlasSubtractInt16(MLAS_INT16X8 Vector1, MLAS_INT16X8 Vector2)
{
    return vsubq_s16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_INT16X4
MlasSubtractInt16(MLAS_INT16X4 Vector1, MLAS_INT16X4 Vector2)
{
    return vsub_s16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X8
MlasMultiplyFloat16(MLAS_FLOAT16X8 Vector1, MLAS_FLOAT16X8 Vector2)
{
    return vmulq_f16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X4
MlasMultiplyFloat16(MLAS_FLOAT16X4 Vector1, MLAS_FLOAT16X4 Vector2)
{
    return vmul_f16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X8
MlasDivideFloat16(MLAS_FLOAT16X8 Vector1, MLAS_FLOAT16X8 Vector2)
{
    return vdivq_f16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X4
MlasDivideFloat16(MLAS_FLOAT16X4 Vector1, MLAS_FLOAT16X4 Vector2)
{
    return vdiv_f16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X8
MlasMultiplyAddFloat16(MLAS_FLOAT16X8 Vector1, MLAS_FLOAT16X8 Vector2, MLAS_FLOAT16X8 Dest)
{
    return vfmaq_f16(Dest, Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X4
MlasMultiplyAddFloat16(MLAS_FLOAT16X4 Vector1, MLAS_FLOAT16X4 Vector2, MLAS_FLOAT16X4 Dest)
{
    return vfma_f16(Dest, Vector1, Vector2);
}

MLAS_FORCEINLINE
void
MlasMultiplyAddFloat16x8(MLAS_FLOAT16X8 Vector1, _mlas_fp16_ Scalar2, MLAS_FLOAT16X8 Vector3)
{
    MlasMultiplyAddFloat16(Vector1, MlasBroadcastFloat16x8(Scalar2), Vector3);
}

MLAS_FORCEINLINE
void
MlasMultiplyAddFloat16x8(MLAS_FLOAT16X8 Vector1, MLAS_FLOAT16X8 Vector2, _mlas_fp16_ Scalar3)
{
    MlasMultiplyAddFloat16(Vector1, Vector2, MlasBroadcastFloat16x8(Scalar3));
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
MlasMaximumFloat16(MLAS_FLOAT16X8 Vector1, MLAS_FLOAT16X8 Vector2)
{
    return vmaxq_f16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X4
MlasMaximumFloat16(MLAS_FLOAT16X4 Vector1, MLAS_FLOAT16X4 Vector2)
{
    return vmax_f16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_INT16X8
MlasMaximumInt16(MLAS_INT16X8 Vector1, MLAS_INT16X8 Vector2)
{
    return vmaxq_s16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_INT16X4
MlasMaximumInt16(MLAS_INT16X4 Vector1, MLAS_INT16X4 Vector2)
{
    return vmax_s16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X8
MlasMinimumFloat16(MLAS_FLOAT16X8 Vector1, MLAS_FLOAT16X8 Vector2)
{
    return vminq_f16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X4
MlasMinimumFloat16(MLAS_FLOAT16X4 Vector1, MLAS_FLOAT16X4 Vector2)
{
    return vmin_f16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_INT16X8
MlasMinimumInt16(MLAS_INT16X8 Vector1, MLAS_INT16X8 Vector2)
{
    return vminq_s16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_INT16X4
MlasMinimumInt16(MLAS_INT16X4 Vector1, MLAS_INT16X4 Vector2)
{
    return vmin_s16(Vector1, Vector2);
}

MLAS_FORCEINLINE
MLAS_FLOAT16X8
MlasClampFloat16x8(MLAS_FLOAT16X8 Value, _mlas_fp16_ LowerRange, _mlas_fp16_ UpperRange)
{
    Value = MlasMaximumFloat16(MlasBroadcastFloat16x8(LowerRange), Value);
    Value = MlasMaximumFloat16(MlasBroadcastFloat16x8(UpperRange), Value);
    return Value;
}

template <typename T>
MLAS_FORCEINLINE
T
MlasClampFloat16(T Value, T LowerRange, T UpperRange)
{
    Value = MlasMaximumFloat16(LowerRange, Value);
    Value = MlasMinimumFloat16(UpperRange, Value);
    return Value;
}

template <typename T>
MLAS_FORCEINLINE
T
MlasClampInt16(T Value, T LowerRange, T UpperRange)
{
    Value = MlasMaximumInt16(LowerRange, Value);
    Value = MlasMinimumInt16(UpperRange, Value);
    return Value;
}

MLAS_FORCEINLINE
_mlas_fp16_
MlasReduceAddFloat16(MLAS_FLOAT16X8 Vector)
{
    Vector = vpaddq_f16(Vector, Vector);
    Vector = vpaddq_f16(Vector, Vector);
    Vector = vpaddq_f16(Vector, Vector);
    return vgetq_lane_u16(vreinterpretq_u16_f16(Vector), 0);
}

MLAS_FORCEINLINE
_mlas_fp16_
MlasReduceAddFloat16(MLAS_FLOAT16X4 Vector)
{
    Vector = vpadd_f16(Vector, Vector);
    Vector = vpadd_f16(Vector, Vector);
    return vget_lane_u16(vreinterpret_u16_f16(Vector), 0);
}

MLAS_FORCEINLINE
_mlas_fp16_
MlasReduceMaximumFloat16(MLAS_FLOAT16X8 Vector)
{
    Vector = vpmaxq_f16(Vector, Vector);
    Vector = vpmaxq_f16(Vector, Vector);
    Vector = vpmaxq_f16(Vector, Vector);
    return vgetq_lane_u16(vreinterpretq_u16_f16(Vector), 0);
}

MLAS_FORCEINLINE
_mlas_fp16_
MlasReduceMaximumFloat16(MLAS_FLOAT16X4 Vector)
{
    Vector = vpmax_f16(Vector, Vector);
    Vector = vpmax_f16(Vector, Vector);
    return vget_lane_u16(vreinterpret_u16_f16(Vector), 0);
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

MLAS_FORCEINLINE
void
Transpose8x8(MLAS_FLOAT16X8& v0, MLAS_FLOAT16X8& v1, MLAS_FLOAT16X8& v2, MLAS_FLOAT16X8& v3,
             MLAS_FLOAT16X8& v4, MLAS_FLOAT16X8& v5, MLAS_FLOAT16X8& v6, MLAS_FLOAT16X8& v7)
{
    // |v00|v01|v02|v03|v04|v05|v06|v07|
    // |v10|v11|v12|v13|v14|v15|v16|v17|
    // |v20|v21|v22|v23|v24|v25|v26|v27|
    // |v30|v31|v32|v33|v34|v35|v36|v37|
    // |v40|v41|v42|v43|v44|v45|v46|v47|
    // |v50|v51|v52|v53|v54|v55|v56|v57|
    // |v60|v61|v62|v63|v64|v65|v66|v67|
    // |v70|v71|v72|v73|v74|v75|v76|v77|
    float16x8x2_t t01 = vtrnq_f16(v0, v1);
    float16x8x2_t t23 = vtrnq_f16(v2, v3);
    float16x8x2_t t45 = vtrnq_f16(v4, v5);
    float16x8x2_t t67 = vtrnq_f16(v6, v7);
    // |v00|v10|v02|v12|v04|v14|v06|v16|
    // |v01|v11|v03|v13|v05|v15|v07|v17|
    // |v20|v30|v22|v32|v24|v34|v26|v36|
    // |v21|v31|v23|v33|v25|v35|v27|v37|
    // |v40|v50|v42|v52|v44|v54|v46|v56|
    // |v41|v51|v43|v53|v45|v55|v47|v57|
    // |v60|v70|v62|v72|v64|v74|v66|v76|
    // |v61|v71|v63|v73|v65|v75|v67|v77|
    float32x4x2_t t02 = vtrnq_f32(vreinterpretq_f32_f16(t01.val[0]), vreinterpretq_f32_f16(t23.val[0]));
    float32x4x2_t t13 = vtrnq_f32(vreinterpretq_f32_f16(t01.val[1]), vreinterpretq_f32_f16(t23.val[1]));
    float32x4x2_t t46 = vtrnq_f32(vreinterpretq_f32_f16(t45.val[0]), vreinterpretq_f32_f16(t67.val[0]));
    float32x4x2_t t57 = vtrnq_f32(vreinterpretq_f32_f16(t45.val[1]), vreinterpretq_f32_f16(t67.val[1]));
    // |v00|v10|v20|v30|v04|v14|v24|v34|
    // |v01|v11|v21|v31|v05|v15|v25|v35|
    // |v02|v12|v22|v32|v06|v16|v26|v36|
    // |v03|v13|v23|v33|v07|v17|v27|v37|
    // |v40|v50|v60|v70|v44|v54|v64|v74|
    // |v41|v51|v61|v71|v45|v55|v65|v75|
    // |v42|v52|v62|v72|v46|v56|v66|v76|
    // |v43|v53|v63|v73|v47|v57|v67|v77|
    v0 = vreinterpretq_f16_f64(vtrn1q_f64(vreinterpretq_f64_f32(t02.val[0]), vreinterpretq_f64_f32(t46.val[0])));
    v4 = vreinterpretq_f16_f64(vtrn2q_f64(vreinterpretq_f64_f32(t02.val[0]), vreinterpretq_f64_f32(t46.val[0])));
    v2 = vreinterpretq_f16_f64(vtrn1q_f64(vreinterpretq_f64_f32(t02.val[1]), vreinterpretq_f64_f32(t46.val[1])));
    v6 = vreinterpretq_f16_f64(vtrn2q_f64(vreinterpretq_f64_f32(t02.val[1]), vreinterpretq_f64_f32(t46.val[1])));
    v1 = vreinterpretq_f16_f64(vtrn1q_f64(vreinterpretq_f64_f32(t13.val[0]), vreinterpretq_f64_f32(t57.val[0])));
    v5 = vreinterpretq_f16_f64(vtrn2q_f64(vreinterpretq_f64_f32(t13.val[0]), vreinterpretq_f64_f32(t57.val[0])));
    v3 = vreinterpretq_f16_f64(vtrn1q_f64(vreinterpretq_f64_f32(t13.val[1]), vreinterpretq_f64_f32(t57.val[1])));
    v7 = vreinterpretq_f16_f64(vtrn2q_f64(vreinterpretq_f64_f32(t13.val[1]), vreinterpretq_f64_f32(t57.val[1])));
    // |v00|v10|v20|v30|v40|v50|v60|v70|
    // |v01|v11|v21|v31|v41|v51|v61|v71|
    // |v02|v12|v22|v32|v42|v52|v62|v72|
    // |v03|v13|v23|v33|v43|v53|v63|v73|
    // |v04|v14|v24|v34|v44|v54|v64|v74|
    // |v05|v15|v25|v35|v45|v55|v65|v75|
    // |v06|v16|v26|v36|v46|v56|v66|v76|
    // |v07|v17|v27|v37|v47|v57|v67|v77|
}

MLAS_FORCEINLINE
void
Transpose4x8(MLAS_FLOAT16X8& v0, MLAS_FLOAT16X8& v1, MLAS_FLOAT16X8& v2, MLAS_FLOAT16X8& v3)
{
    // |v00|v01|v02|v03|v04|v05|v06|v07|
    // |v10|v11|v12|v13|v14|v15|v16|v17|
    // |v20|v21|v22|v23|v24|v25|v26|v27|
    // |v30|v31|v32|v33|v34|v35|v36|v37|
    //  =>
    // |v00|v10|v20|v30|v04|v14|v24|v34|
    // |v01|v11|v21|v31|v05|v15|v25|v35|
    // |v02|v12|v22|v32|v06|v16|v26|v36|
    // |v03|v13|v23|v33|v07|v17|v27|v37|
    float16x8x2_t t01 = vtrnq_f16(v0, v1);
    float16x8x2_t t23 = vtrnq_f16(v2, v3);

    v0 = vreinterpretq_f16_f32(vtrn1q_f32(vreinterpretq_f32_f16(t01.val[0]), vreinterpretq_f32_f16(t23.val[0])));
    v2 = vreinterpretq_f16_f32(vtrn2q_f32(vreinterpretq_f32_f16(t01.val[0]), vreinterpretq_f32_f16(t23.val[0])));
    v1 = vreinterpretq_f16_f32(vtrn1q_f32(vreinterpretq_f32_f16(t01.val[1]), vreinterpretq_f32_f16(t23.val[1])));
    v3 = vreinterpretq_f16_f32(vtrn2q_f32(vreinterpretq_f32_f16(t01.val[1]), vreinterpretq_f32_f16(t23.val[1])));
}

MLAS_FORCEINLINE
void
Transpose4x4(MLAS_FLOAT16X4& v0, MLAS_FLOAT16X4& v1, MLAS_FLOAT16X4& v2, MLAS_FLOAT16X4& v3)
{
    // |v00|v01|v02|v03|
    // |v10|v11|v12|v13|
    // |v20|v21|v22|v23|
    // |v30|v31|v32|v33|
    //  =>
    // |v00|v10|v20|v30|
    // |v01|v11|v21|v31|
    // |v02|v12|v22|v32|
    // |v03|v13|v23|v33|
    float16x4x2_t t01 = vtrn_f16(v0, v1);
    float16x4x2_t t23 = vtrn_f16(v2, v3);

    v0 = vreinterpret_f16_f32(vtrn1_f32(vreinterpret_f32_f16(t01.val[0]), vreinterpret_f32_f16(t23.val[0])));
    v1 = vreinterpret_f16_f32(vtrn1_f32(vreinterpret_f32_f16(t01.val[1]), vreinterpret_f32_f16(t23.val[1])));
    v2 = vreinterpret_f16_f32(vtrn2_f32(vreinterpret_f32_f16(t01.val[0]), vreinterpret_f32_f16(t23.val[0])));
    v3 = vreinterpret_f16_f32(vtrn2_f32(vreinterpret_f32_f16(t01.val[1]), vreinterpret_f32_f16(t23.val[1])));
}

template<unsigned ShiftCount>
MLAS_FORCEINLINE
MLAS_INT16X8
MlasShiftLeftInt16(MLAS_INT16X8 Vector)
{
    return vshlq_n_s16(Vector, ShiftCount);
}

template<unsigned ShiftCount>
MLAS_FORCEINLINE
MLAS_INT16X4
MlasShiftLeftInt16(MLAS_INT16X4 Vector)
{
    return vshl_n_s16(Vector, ShiftCount);
}

#endif  // fp16 vector intrinsic supported
