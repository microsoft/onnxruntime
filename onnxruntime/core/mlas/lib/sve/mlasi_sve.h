/*++
Copyright 2025 FUJITSU LIMITED

Module Name:

    mlasi_sve.h

Abstract:

    This module contains the procedure prototypes for the SVE intrinsics.

--*/

#pragma once

#include "../mlasi.h"
#include <arm_sve.h>  // SVE intrinsic header

#ifndef __clang__
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")

// Use Clang-specific per-function attribute
#ifdef __clang__
#define MLAS_SVE_TARGET __attribute__((target("arch=armv8.2-a+sve")))
#else
#define MLAS_SVE_TARGET
#endif

#define PACKED_B_BLOCK_WIDTH 16
typedef svfloat32_t MLAS_SVFLOAT32;
typedef svint32_t MLAS_SVINT32;
typedef svuint32_t MLAS_SVUINT32;
typedef svbool_t MLAS_SVBOOL;

// function decarations
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveComputeExpVector(
    MLAS_SVBOOL Pred,
    MLAS_SVFLOAT32 Vector
);

void
MLASCALL
MlasSveComputeExpF32Kernel(
    const float* Input,
    float* Output,
    size_t N
);

MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveComputeSumExpVector(
    MLAS_SVBOOL Pred,
    MLAS_SVFLOAT32 Vector,
    MLAS_SVFLOAT32 NegativeMaximumVector
);

float
MLASCALL
MlasSveComputeSumExpF32Kernel(
    const float* Input,
    float* Output,
    size_t N,
    const float* NegativeMaximum
);

float MLASCALL
MlasSveReduceMaximumF32Kernel(
    const float* Input,
    size_t N
);

void
MLASCALL
MlasSveReduceMinimumMaximumF32Kernel(
    const float* Input,
    float* Min,
    float* Max,
    size_t N
);

void
MLASCALL
MlasSveComputeSoftmaxOutputF32Kernel(
    float* Output,
    size_t N,
    const float* Parameters
);

void
MLASCALL
MlasSveComputeLogSoftmaxOutputF32Kernel(
    const float* Input,
    float* Output,
    size_t N,
    const float* Parameters
);

void
MLASCALL
MlasSveErfKernel(
    const float* Input,
    float* Output,
    size_t N
);

void 
MLASCALL
MlasSveLogisticKernel(
    const float* Input,
    float* Output,
    size_t N
);

//MLAS API for SVE intrinsics
size_t MLASCALL
MlasSgemmKernelAdd_sve(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
);

size_t MLASCALL
MlasSgemmKernelZero_sve(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
);

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLASCALL
void
SCATTER_STORE(float* d, const float* b);

MLAS_SVE_TARGET
inline int
VL()
{
    static int fp32Lanes = svcntw();  // evaluated only once, the first time it's called
    return fp32Lanes;
}

// MLAS API for SVE intrinsics

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVINT32
MlasSveReinterpretAsInt32(MLAS_SVFLOAT32 Vector)
{
    return svreinterpret_s32_f32(Vector);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVUINT32
MlasSveReinterpretAsUInt32(MLAS_SVFLOAT32 Vector)
{
    return svreinterpret_u32_f32(Vector);
}

// Reinterprets an unsigned 32-bit vector as a 32-bit floating-point vector.
MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveReinterpretAsFLOAT32(MLAS_SVUINT32 Vector)
{
    return svreinterpret_f32_u32(Vector);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVINT32
MlasSveCastToInt32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector)
{
    return svcvt_s32_f32_z(Pred, Vector);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveCastToFloat32(MLAS_SVBOOL Pred, MLAS_SVINT32 Vector)
{
    return svcvt_f32_s32_z(Pred, Vector);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVINT32
MlasSveBroadcastInt32(int32_t Value)
{
    return svdup_n_s32(Value);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVINT32
MlasSveLoadInt32(MLAS_SVBOOL Pred, const int32_t* Buffer)
{
    return svld1_s32(Pred, Buffer);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
void
MlasSveStoreInt32(MLAS_SVBOOL Pred, int32_t* Buffer, MLAS_SVINT32 Vector)
{
    svst1_s32(Pred, Buffer, Vector);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVINT32
MlasSveAddInt32(MLAS_SVBOOL Pred, MLAS_SVINT32 Vector1, MLAS_SVINT32 Vector2)
{   
    return svadd_s32_m(Pred, Vector1, Vector2);  
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVINT32
MlasSveSubtractInt32(MLAS_SVBOOL Pred, MLAS_SVINT32 Vector1, MLAS_SVINT32 Vector2)
{
    return svsub_s32_m(Pred, Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVINT32
MlasSveAndInt32(MLAS_SVBOOL Pred, MLAS_SVINT32 Vector1, MLAS_SVINT32 Vector2)
{
    return svand_s32_m(Pred, Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVUINT32
MlasSveAndUInt32(MLAS_SVBOOL Pred, MLAS_SVUINT32 Vector1, MLAS_SVUINT32 Vector2)
{
    return svand_u32_m(Pred, Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVINT32
MlasSveOrInt32(MLAS_SVBOOL Pred, MLAS_SVINT32 Vector1, MLAS_SVINT32 Vector2)
{
    return svorr_s32_m(Pred, Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVINT32
MlasSveAndNotInt32(MLAS_SVBOOL Pred, MLAS_SVINT32 VectorNot, MLAS_SVINT32 Vector)
{
    return svand_s32_m(Pred, svnot_s32_z(Pred, VectorNot), Vector);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVINT32
MlasSveXorInt32(MLAS_SVBOOL Pred, MLAS_SVINT32 Vector1, MLAS_SVINT32 Vector2)
{
    return sveor_s32_m(Pred, Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVINT32
MlasSveBlendInt32(MLAS_SVBOOL Pred, MLAS_SVINT32 Vector1, MLAS_SVINT32 Vector2, MLAS_SVINT32 Selection)
{
    return MlasSveOrInt32(
        Pred, 
        MlasSveAndInt32(Pred, Vector2, Selection), 
        MlasSveAndNotInt32(Pred, Selection, Vector1)
    );
}

template<unsigned ShiftCount>
MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVUINT32
MlasSveShiftLeftUInt32(MLAS_SVBOOL Pred, MLAS_SVUINT32 Vector)
{
    return svlsl_n_u32_z(Pred, Vector, ShiftCount);
}

template<unsigned ShiftCount>
MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVINT32
MlasSveShiftLeftInt32(MLAS_SVBOOL Pred, MLAS_SVINT32 Vector)
{
    return svlsl_n_s32_z(Pred, Vector, ShiftCount);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVUINT32
MlasSveShiftRightInt32(MLAS_SVBOOL Pred, MLAS_SVUINT32 Vector, uint ShiftCount)
{
    return svlsr_n_u32_m(Pred, Vector, ShiftCount);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVINT32
MlasSveMaximumInt32(MLAS_SVBOOL Pred, MLAS_SVINT32 Vector1, MLAS_SVINT32 Vector2)
{
    return svmax_s32_m(Pred, Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVINT32
MlasSveMinimumInt32(MLAS_SVBOOL Pred, MLAS_SVINT32 Vector1, MLAS_SVINT32 Vector2)
{
    return svmin_s32_m(Pred, Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveReinterpretAsFloat32(MLAS_SVINT32 Vector)
{
    return svreinterpret_f32_s32(Vector);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveBroadcastFloat32(float Value)
{
    return svdup_n_f32(Value);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVUINT32
MlasSveBroadcastUINT32(uint Value)
{
    return svdup_n_u32(Value);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveBroadcastFloat32(const float* Value)
{
    return svld1_f32(svptrue_b32(), Value);
}

MLAS_SVBOOL
    MLAS_FORCEINLINE
    MlasSveptrue(void)
{
    return svptrue_b32();
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveZeroFloat32(void)
{
    return svdup_n_f32(0.0f);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveLoadFloat32(MLAS_SVBOOL Pred, const float* Buffer)
{
    return svld1_f32(Pred, Buffer);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
void
MlasSveStoreFloat32(MLAS_SVBOOL Pred, float* Buffer, MLAS_SVFLOAT32 Vector)
{
    svst1_f32(Pred, Buffer, Vector);
}

template<unsigned Lane>
MLAS_SVE_TARGET
MLAS_FORCEINLINE
void
MlasSveStoreLaneFloat32(float* Buffer, MLAS_SVFLOAT32 Vector)
{
    svbool_t Pred = svwhilelt_b32(Lane, Lane + 1);
    svst1_f32(Pred, Buffer, Vector);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
void
MlasSveStoreLowHalfFloat32(float* Buffer, MLAS_SVFLOAT32 Vector)
{
    svbool_t Pred = svwhilelt_b32(0, (int32_t)svcntw() / 2);
    svst1_f32(Pred, Buffer, Vector);
}

template<unsigned Lane>
MLAS_SVE_TARGET
MLAS_FORCEINLINE
float
MlasSveExtractLaneFloat32(MLAS_SVFLOAT32 Vector)
{
    float TmpBuffer[1];
    svbool_t Pred = svwhilelt_b32(Lane, Lane + 1);
    svst1_f32(Pred, TmpBuffer, Vector);
    return TmpBuffer[0];
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveInterleaveLowFloat32(MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2)
{
    return svzip1_f32(Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveInterleaveHighFloat32(MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2)
{
    return svzip2_f32(Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveAddFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2)
{
    return svadd_f32_m(Pred, Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveSubtractFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2)
{
    return svsub_f32_m(Pred, Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveMultiplyFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2)
{
    return svmul_f32_m(Pred, Vector1, Vector2);  
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveExpFloat32(MLAS_SVUINT32 Vector)
{
    return svexpa_f32(Vector);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveScaleFloat32(MLAS_SVBOOL Pred,  MLAS_SVFLOAT32 Vector1, MLAS_SVINT32 Vector2)
{
    return svscale_f32_m(Pred, Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveRoundINTFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector)
{
   return svrintm_f32_z(Pred, Vector);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveMultiplyAddFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2, MLAS_SVFLOAT32 Vector3)
{
    return svmla_f32_m(Pred, Vector3, Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveMultiplyAddFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, float Scalar2, MLAS_SVFLOAT32 Vector3)
{
    return MlasSveMultiplyAddFloat32(Pred, Vector1, MlasSveBroadcastFloat32(Scalar2), Vector3);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveMultiplyAddFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2, float Scalar3)
{
    return MlasSveMultiplyAddFloat32(Pred, Vector1, Vector2, MlasSveBroadcastFloat32(Scalar3));
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveDivideFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2)
{
    return svdiv_f32_m(Pred, Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveGreaterThanFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2)
{
    // Compare Vector1 and Vector2, return a predicate vector
    svbool_t cmp_mask = svcmpgt_f32(Pred, Vector1, Vector2);

    //Convert predicate to uint32_t mask
    svuint32_t mask_bits = svdup_u32_z(cmp_mask, 0xFFFFFFFF);

    //Reinterpret to float32
    return svreinterpret_f32_u32(mask_bits);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveAndFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2)
{
    return MlasSveReinterpretAsFloat32(
        MlasSveAndInt32(
            Pred, 
            MlasSveReinterpretAsInt32(Vector1),
            MlasSveReinterpretAsInt32(Vector2)
        )
    );
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveOrFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2)
{
    return MlasSveReinterpretAsFloat32(
        MlasSveOrInt32(
            Pred,
            MlasSveReinterpretAsInt32(Vector1),
            MlasSveReinterpretAsInt32(Vector2)
        )
    );
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveAndNotFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2)
{
    return MlasSveReinterpretAsFloat32(
        MlasSveAndNotInt32(
            Pred,
            MlasSveReinterpretAsInt32(Vector1),
            MlasSveReinterpretAsInt32(Vector2)
        )
    );
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveXorFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2)
{
    return MlasSveReinterpretAsFloat32(
        MlasSveXorInt32(
            Pred,
            MlasSveReinterpretAsInt32(Vector1),
            MlasSveReinterpretAsInt32(Vector2)
        )
    );
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveBlendFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2, MLAS_SVFLOAT32 Selection)
{
    return MlasSveOrFloat32(
        Pred, 
        MlasSveAndFloat32(Pred, Vector2, Selection),
        MlasSveAndFloat32(Pred, Vector1, Selection)
    );
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveMaximumFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2)
{
    return svmax_f32_m(Pred, Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveMinimumFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector1, MLAS_SVFLOAT32 Vector2)
{
    return svmin_f32_m(Pred, Vector1, Vector2);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveClampFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Value, float LowerRange, float UpperRange)
{
    Value = MlasSveMaximumFloat32(Pred, MlasSveBroadcastFloat32(LowerRange), Value);
    Value = MlasSveMinimumFloat32(Pred, MlasSveBroadcastFloat32(UpperRange), Value);
    return Value;
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
float
MlasSveReduceAddFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector)
{
    return svaddv_f32(Pred, Vector);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
float
MlasSveReduceMaximumFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector)
{
    return svmaxv_f32(Pred, Vector);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
float
MlasSveReduceMinimumFloat32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector)
{
    return svminv_f32(Pred, Vector);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSvePowerOf2Float32(MLAS_SVBOOL Pred, MLAS_SVFLOAT32 Vector)
{
    MLAS_SVINT32 emm0 = MlasSveAddInt32(
        Pred, 
        MlasSveCastToInt32(Pred, Vector), 
        MlasSveBroadcastInt32(127)
    );
    return MlasSveReinterpretAsFloat32(MlasSveShiftLeftInt32<23>(Pred, emm0));
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT32
MlasSveSelect(svbool_t Pred, MLAS_SVFLOAT32 TrueValue, MLAS_SVFLOAT32 FalseValue)
{
    return svsel_f32(Pred, TrueValue, FalseValue);
}
svfloat32_t
MlasSvedupFloat32(float Vector)
{
    return svdup_f32(Vector);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVBOOL
MlasSveCompareLessThan(svbool_t Pred, MLAS_SVFLOAT32 A, MLAS_SVFLOAT32 B)
{
    return svcmplt_f32(Pred, A, B);
}
void
Transpose_SVE512_4x4(float* D, const float* B, size_t ldb)
{
    const static int VL = svcntw();
    MLAS_SVBOOL p = svwhilelt_b32(0, VL / 4);
    MLAS_SVBOOL p3 = svwhilelt_b32(0, VL / 2);
    MLAS_SVBOOL p1 = svnot_b_z(svwhilelt_b32(0, VL), p);
    p1 = svand_b_z(p3, p3, p1);
    p3 = svrev_b32(p1);
    MLAS_SVBOOL p4 = svrev_b32(p);

    MLAS_SVFLOAT32 t0 = MlasSveLoadFloat32(p, &B[ldb * 0]);
    MLAS_SVFLOAT32 t1 = MlasSveLoadFloat32(p, &B[ldb * 1]);
    MLAS_SVFLOAT32 t2 = MlasSveLoadFloat32(p, &B[ldb * 2]);
    MLAS_SVFLOAT32 t3 = MlasSveLoadFloat32(p, &B[ldb * 3]);

    MLAS_SVFLOAT32 t02 = MlasSveInterleaveLowFloat32(t0, t2);
    MLAS_SVFLOAT32 t13 = MlasSveInterleaveLowFloat32(t1, t3);
    MLAS_SVFLOAT32 t0123 = MlasSveInterleaveLowFloat32(t02, t13);  // This zips the first half together

    MlasSveStoreFloat32(p, D, t0123);
    MlasSveStoreFloat32(p1, &D[12], t0123);
    MlasSveStoreFloat32(p3, &D[24], t0123);
    MlasSveStoreFloat32(p4, &D[36], t0123);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVBOOL
MlasSveCompareGreaterThan(svbool_t Pred, MLAS_SVFLOAT32 A, MLAS_SVFLOAT32 B)
{
    return svcmpgt_f32(Pred, A, B);
}
void
Transpose_SVE256_4x4(float* D, const float* B, size_t ldb)
{
    const static int VL = svcntw();
    MLAS_SVBOOL p = svwhilelt_b32(0, VL / 2);

    MLAS_SVFLOAT32 t0 = MlasSveLoadFloat32(p, &B[ldb * 0]);
    MLAS_SVFLOAT32 t1 = MlasSveLoadFloat32(p, &B[ldb * 1]);
    MLAS_SVFLOAT32 t2 = MlasSveLoadFloat32(p, &B[ldb * 2]);
    MLAS_SVFLOAT32 t3 = MlasSveLoadFloat32(p, &B[ldb * 3]);

    MLAS_SVBOOL p1 = svnot_b_z(svwhilelt_b32((int)0, VL), p);
    MLAS_SVFLOAT32 t02 = MlasSveInterleaveLowFloat32(t0, t2);
    MLAS_SVFLOAT32 t13 = MlasSveInterleaveLowFloat32(t1, t3);
    MLAS_SVFLOAT32 first_t0123 = MlasSveInterleaveLowFloat32(t02, t13);    // This zips the first half together
    MLAS_SVFLOAT32 second_t0123 = MlasSveInterleaveHighFloat32(t02, t13);  // This zips the second half together

    MlasSveStoreFloat32(p, D, first_t0123);
    MlasSveStoreFloat32(p1, &D[12], first_t0123);
    MlasSveStoreFloat32(p, &D[32], second_t0123);
    MlasSveStoreFloat32(p1, &D[44], second_t0123);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
void
Transpose_SVE128_4x4(float* D, const float* B, size_t ldb)
{
    const static int VL = svcntw();
    MLAS_SVBOOL p = svwhilelt_b32((int)0, VL);

    MLAS_SVFLOAT32 v1 = MlasSveLoadFloat32(p, &B[ldb * 0]);
    MLAS_SVFLOAT32 v2 = MlasSveLoadFloat32(p, &B[ldb * 1]);
    MLAS_SVFLOAT32 v4 = MlasSveLoadFloat32(p, &B[ldb * 2]);
    MLAS_SVFLOAT32 v5 = MlasSveLoadFloat32(p, &B[ldb * 3]);

    MLAS_SVFLOAT32 v3 = MlasSveInterleaveLowFloat32(v1, v4);
    v1 = MlasSveInterleaveHighFloat32(v1, v4);

    v4 = MlasSveInterleaveLowFloat32(v2, v5);
    v2 = MlasSveInterleaveHighFloat32(v2, v5);

    v5 = MlasSveInterleaveLowFloat32(v3, v4);
    v3 = MlasSveInterleaveHighFloat32(v3, v4);

    v4 = MlasSveInterleaveLowFloat32(v1, v2);
    v1 = MlasSveInterleaveHighFloat32(v1, v2);

    MlasSveStoreFloat32(p, &D[0], v5);
    MlasSveStoreFloat32(p, &D[16], v3);
    MlasSveStoreFloat32(p, &D[32], v4);
    MlasSveStoreFloat32(p, &D[48], v1);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
void
Transpose_SVE256_8x8(float* D, const float* B, size_t ldb)
{
    const static int VL = svcntw();

    MLAS_SVBOOL p = svwhilelt_b32((int)0, VL);

    MLAS_SVFLOAT32 v1 = MlasSveLoadFloat32(p, &B[ldb * 0]);
    MLAS_SVFLOAT32 v2 = MlasSveLoadFloat32(p, &B[ldb * 1]);
    MLAS_SVFLOAT32 v4 = MlasSveLoadFloat32(p, &B[ldb * 2]);
    MLAS_SVFLOAT32 v5 = MlasSveLoadFloat32(p, &B[ldb * 3]);

    MLAS_SVFLOAT32 v6 = MlasSveLoadFloat32(p, &B[ldb * 4]);
    MLAS_SVFLOAT32 v7 = MlasSveLoadFloat32(p, &B[ldb * 5]);
    MLAS_SVFLOAT32 v8 = MlasSveLoadFloat32(p, &B[ldb * 6]);
    MLAS_SVFLOAT32 v9 = MlasSveLoadFloat32(p, &B[ldb * 7]);

    // First mix
    MLAS_SVFLOAT32 v3 = MlasSveInterleaveLowFloat32(v1, v6);
    v1 = MlasSveInterleaveHighFloat32(v1, v6);

    v6 = MlasSveInterleaveLowFloat32(v2, v7);
    v2 = MlasSveInterleaveHighFloat32(v2, v7);

    v7 = MlasSveInterleaveLowFloat32(v4, v8);
    v4 = MlasSveInterleaveHighFloat32(v4, v8);

    v8 = MlasSveInterleaveLowFloat32(v5, v9);

    v5 = MlasSveInterleaveHighFloat32(v5, v9);

    // Second mix

    v9 = MlasSveInterleaveLowFloat32(v3, v7);
    v3 = MlasSveInterleaveHighFloat32(v3, v7);

    v7 = MlasSveInterleaveLowFloat32(v6, v8);
    v6 = MlasSveInterleaveHighFloat32(v6, v8);

    v8 = MlasSveInterleaveLowFloat32(v1, v4);
    v1 = MlasSveInterleaveHighFloat32(v1, v4);

    v4 = MlasSveInterleaveLowFloat32(v2, v5);
    v2 = MlasSveInterleaveHighFloat32(v2, v5);

    // Third mix
    v5 = MlasSveInterleaveLowFloat32(v9, v7);
    v9 = MlasSveInterleaveHighFloat32(v9, v7);

    v7 = MlasSveInterleaveLowFloat32(v8, v4);
    v8 = MlasSveInterleaveHighFloat32(v8, v4);

    v4 = MlasSveInterleaveLowFloat32(v3, v6);
    v3 = MlasSveInterleaveHighFloat32(v3, v6);

    v6 = MlasSveInterleaveLowFloat32(v1, v2);
    v1 = MlasSveInterleaveHighFloat32(v1, v2);

    // Store the results

    MlasSveStoreFloat32(p, &D[0], v5);
    MlasSveStoreFloat32(p, &D[16], v9);
    MlasSveStoreFloat32(p, &D[32], v4);
    MlasSveStoreFloat32(p, &D[48], v3);
    MlasSveStoreFloat32(p, &D[64], v7);
    MlasSveStoreFloat32(p, &D[80], v8);
    MlasSveStoreFloat32(p, &D[96], v6);
    MlasSveStoreFloat32(p, &D[112], v1);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
void
Transpose_SVE512_16x16(float* D, const float* B, size_t ldb)
{
    const static int VL = svcntw();
    MLAS_SVBOOL p = svwhilelt_b32((int)0, VL);

    MLAS_SVFLOAT32 v1 = MlasSveLoadFloat32(p, &B[ldb * 0]);
    MLAS_SVFLOAT32 v2 = MlasSveLoadFloat32(p, &B[ldb * 1]);
    MLAS_SVFLOAT32 v3 = MlasSveLoadFloat32(p, &B[ldb * 2]);
    MLAS_SVFLOAT32 v4 = MlasSveLoadFloat32(p, &B[ldb * 3]);

    MLAS_SVFLOAT32 v5 = MlasSveLoadFloat32(p, &B[ldb * 4]);
    MLAS_SVFLOAT32 v6 = MlasSveLoadFloat32(p, &B[ldb * 5]);
    MLAS_SVFLOAT32 v7 = MlasSveLoadFloat32(p, &B[ldb * 6]);
    MLAS_SVFLOAT32 v8 = MlasSveLoadFloat32(p, &B[ldb * 7]);

    MLAS_SVFLOAT32 v9 = MlasSveLoadFloat32(p, &B[ldb * 8]);
    MLAS_SVFLOAT32 v10 = MlasSveLoadFloat32(p, &B[ldb * 9]);
    MLAS_SVFLOAT32 v11 = MlasSveLoadFloat32(p, &B[ldb * 10]);
    MLAS_SVFLOAT32 v12 = MlasSveLoadFloat32(p, &B[ldb * 11]);

    MLAS_SVFLOAT32 v13 = MlasSveLoadFloat32(p, &B[ldb * 12]);
    MLAS_SVFLOAT32 v14 = MlasSveLoadFloat32(p, &B[ldb * 13]);
    MLAS_SVFLOAT32 v15 = MlasSveLoadFloat32(p, &B[ldb * 14]);
    MLAS_SVFLOAT32 v16 = MlasSveLoadFloat32(p, &B[ldb * 15]);

    /*========= FIRST MIX ==============*/

    MLAS_SVFLOAT32 v17 = MlasSveInterleaveLowFloat32(v1, v9);
    MLAS_SVFLOAT32 v18 = MlasSveInterleaveHighFloat32(v1, v9);

    MLAS_SVFLOAT32 v19 = MlasSveInterleaveLowFloat32(v2, v10);
    MLAS_SVFLOAT32 v20 = MlasSveInterleaveHighFloat32(v2, v10);

    MLAS_SVFLOAT32 v21 = MlasSveInterleaveLowFloat32(v3, v11);
    MLAS_SVFLOAT32 v22 = MlasSveInterleaveHighFloat32(v3, v11);

    MLAS_SVFLOAT32 v23 = MlasSveInterleaveLowFloat32(v4, v12);
    MLAS_SVFLOAT32 v24 = MlasSveInterleaveHighFloat32(v4, v12);

    //

    MLAS_SVFLOAT32 v25 = MlasSveInterleaveLowFloat32(v5, v13);
    MLAS_SVFLOAT32 v26 = MlasSveInterleaveHighFloat32(v5, v13);

    MLAS_SVFLOAT32 v27 = MlasSveInterleaveLowFloat32(v6, v14);
    MLAS_SVFLOAT32 v28 = MlasSveInterleaveHighFloat32(v6, v14);

    MLAS_SVFLOAT32 v29 = MlasSveInterleaveLowFloat32(v7, v15);
    MLAS_SVFLOAT32 v30 = MlasSveInterleaveHighFloat32(v7, v15);

    MLAS_SVFLOAT32 v31 = MlasSveInterleaveLowFloat32(v8, v16);
    MLAS_SVFLOAT32 v32 = MlasSveInterleaveHighFloat32(v8, v16);

    /*========= SECOND MIX ==============*/

    v1 = MlasSveInterleaveLowFloat32(v17, v25);
    v9 = MlasSveInterleaveHighFloat32(v17, v25);

    v2 = MlasSveInterleaveLowFloat32(v18, v26);
    v10 = MlasSveInterleaveHighFloat32(v18, v26);

    v3 = MlasSveInterleaveLowFloat32(v19, v27);
    v11 = MlasSveInterleaveHighFloat32(v19, v27);

    v4 = MlasSveInterleaveLowFloat32(v20, v28);
    v12 = MlasSveInterleaveHighFloat32(v20, v28);

    //
    v5 = MlasSveInterleaveLowFloat32(v21, v29);
    v13 = MlasSveInterleaveHighFloat32(v21, v29);

    v6 = MlasSveInterleaveLowFloat32(v22, v30);
    v14 = MlasSveInterleaveHighFloat32(v22, v30);

    v7 = MlasSveInterleaveLowFloat32(v23, v31);
    v15 = MlasSveInterleaveHighFloat32(v23, v31);

    v8 = MlasSveInterleaveLowFloat32(v24, v32);
    v16 = MlasSveInterleaveHighFloat32(v24, v32);

    /*======= Third Mix =================*/

    v17 = MlasSveInterleaveLowFloat32(v1, v5);
    v25 = MlasSveInterleaveHighFloat32(v1, v5);

    v18 = MlasSveInterleaveLowFloat32(v9, v13);
    v26 = MlasSveInterleaveHighFloat32(v9, v13);

    v19 = MlasSveInterleaveLowFloat32(v2, v6);
    v27 = MlasSveInterleaveHighFloat32(v2, v6);

    v20 = MlasSveInterleaveLowFloat32(v10, v14);
    v28 = MlasSveInterleaveHighFloat32(v10, v14);

    v21 = MlasSveInterleaveLowFloat32(v3, v7);
    v29 = MlasSveInterleaveHighFloat32(v3, v7);

    v22 = MlasSveInterleaveLowFloat32(v11, v15);
    v30 = MlasSveInterleaveHighFloat32(v11, v15);

    v23 = MlasSveInterleaveLowFloat32(v4, v8);
    v31 = MlasSveInterleaveHighFloat32(v4, v8);

    v24 = MlasSveInterleaveLowFloat32(v12, v16);
    v32 = MlasSveInterleaveHighFloat32(v12, v16);

    /*======== Final Mix ================*/

    v1 = MlasSveInterleaveLowFloat32(v17, v21);
    v9 = MlasSveInterleaveHighFloat32(v17, v21);

    v2 = MlasSveInterleaveLowFloat32(v25, v29);
    v10 = MlasSveInterleaveHighFloat32(v25, v29);

    v3 = MlasSveInterleaveLowFloat32(v18, v22);
    v11 = MlasSveInterleaveHighFloat32(v18, v22);

    v4 = MlasSveInterleaveLowFloat32(v26, v30);
    v12 = MlasSveInterleaveHighFloat32(v26, v30);

    v5 = MlasSveInterleaveLowFloat32(v19, v23);
    v13 = MlasSveInterleaveHighFloat32(v19, v23);

    v6 = MlasSveInterleaveLowFloat32(v27, v31);
    v14 = MlasSveInterleaveHighFloat32(v27, v31);

    v7 = MlasSveInterleaveLowFloat32(v20, v24);
    v15 = MlasSveInterleaveHighFloat32(v20, v24);

    v8 = MlasSveInterleaveLowFloat32(v28, v32);
    v16 = MlasSveInterleaveHighFloat32(v28, v32);

    // store the result.

    MlasSveStoreFloat32(p, &D[0], v1);
    MlasSveStoreFloat32(p, &D[16], v9);
    MlasSveStoreFloat32(p, &D[32], v2);
    MlasSveStoreFloat32(p, &D[48], v10);
    //
    MlasSveStoreFloat32(p, &D[64], v3);
    MlasSveStoreFloat32(p, &D[80], v11);
    MlasSveStoreFloat32(p, &D[96], v4);
    MlasSveStoreFloat32(p, &D[112], v12);
    //
    MlasSveStoreFloat32(p, &D[128], v5);
    MlasSveStoreFloat32(p, &D[144], v13);
    MlasSveStoreFloat32(p, &D[160], v6);
    MlasSveStoreFloat32(p, &D[176], v14);
    //
    MlasSveStoreFloat32(p, &D[192], v7);
    MlasSveStoreFloat32(p, &D[208], v15);
    MlasSveStoreFloat32(p, &D[224], v8);
    MlasSveStoreFloat32(p, &D[240], v16);
}

template <unsigned N>
MLAS_FORCEINLINE void
TransposePackBNx8(
    float* D,
    const float* B,
    size_t ldb
)
{
    for (unsigned n = 0; n < N / 8; n++) {
        Transpose_SVE256_8x8(D, B, ldb);
        D += 8;
        B += ldb * 8;
    }
}

MLAS_SVE_TARGET
template <unsigned N>
void
MlasSveTransposePackBNx4(
    float* D,
    const float* B,
    size_t ldb
)
{
    for (unsigned n = 0; n < N / 4; n++) {
        if (VL() == 16) {
            Transpose_SVE512_4x4(&D[0], &B[0], ldb);
        } else if (VL() == 8) {
            Transpose_SVE256_4x4(&D[0], &B[0], ldb);
        } else if (VL() == 4) {
            Transpose_SVE128_4x4(&D[0], &B[0], ldb);
        }

        D += 4;
        B += ldb * 4;
    }
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
void
SVE_TRANSPOSE(
    float*& D,
    const float*& b,
    size_t ldb,
    size_t& x
)
{
    const static int VL = svcntw();
    if (VL == 16) {
        while (x >= 16) {
            Transpose_SVE512_16x16(&D[0], &b[0], ldb);
            D += 256;
            b += 16;
            x = x - 16;
        }
    } else if (VL == 8) {
        while (x >= 8) {
            TransposePackBNx8<16>(&D[0], &b[0], ldb);
            D += 128;
            b += 8;
            x = x - 8;
        }
    }

    while (x >= 4) {
        MlasSveTransposePackBNx4<16>(&D[0], &b[0], ldb);

        D += 16 * 4;
        b += 4;
        x = x - 4;
    }
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
void
SCATTER_STORE(float* d, const float* b)
{
    MLAS_SVBOOL pb = svwhilelt_b32((int)0, 4);        // MSB 4 bits
    MLAS_SVFLOAT32 vec0 = MlasSveLoadFloat32(pb, b);  // Load a set of 4 elements

    svuint32_t idx = svindex_u32(0, 1);
    MLAS_SVBOOL pb_first_half = svcmpeq_u32(pb, idx, svdup_n_u32(0));
    MLAS_SVBOOL pb_second_half = svcmpeq_u32(pb, idx, svdup_n_u32(1));
    MLAS_SVBOOL pb_third_half = svcmpeq_u32(pb, idx, svdup_n_u32(2));
    MLAS_SVBOOL pb_fourth_half = svcmpeq_u32(pb, idx, svdup_n_u32(3));

    MlasSveStoreFloat32(pb_first_half, &d[0], vec0);
    MlasSveStoreFloat32(pb_second_half, &d[15], vec0);
    MlasSveStoreFloat32(pb_third_half, &d[30], vec0);
    MlasSveStoreFloat32(pb_fourth_half, &d[45], vec0);
}

void MLAS_SVE_TARGET
    MLAS_FORCEINLINE MLASCALL
    SVE_LOAD_STORE(float* D, const float* b)
{
    for (int i = 0; i < MLAS_SGEMM_STRIDEN_THREAD_ALIGN; i += VL()) {
        svfloat32_t vec0 = MlasSveLoadFloat32(svptrue_b32(), b + i);
        MlasSveStoreFloat32(svptrue_b32(), D + i, vec0);
    }
}

void MLAS_SVE_TARGET
    MLAS_FORCEINLINE MLASCALL
    SVE_ZERO_INITIALIZE(float* d)
{
    if (VL() == PACKED_B_BLOCK_WIDTH) {
        MlasSveStoreFloat32(svptrue_b32(), d, svdup_f32(0));
    } else {
        MlasSveStoreFloat32(svptrue_b32(), d, svdup_f32(0));
        MlasSveStoreFloat32(svptrue_b32(), d + VL(), svdup_f32(0));
    }
}

// GCC: Pop options after SVE-specific functions
#ifndef __clang__
#pragma GCC pop_options
#endif

#endif
<<<<<<< HEAD

=======
>>>>>>> 62945f3a2 (ADD support for SVE sgemm)
