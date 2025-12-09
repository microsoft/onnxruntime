/*++

Copyright 2025 FUJITSU LIMITED

Module Name:

  mlas_sve_fp16.h

Abstract:

    This module contains the procedure prototypes for the SVE FP16 intrinsics.

--*/

#pragma once
#include <arm_fp16.h>
#include <math.h>  // for isnan if needed
#include <stddef.h>

#include "mlasi_sve.h"

#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT16
MlasSveBroadcastfloat16(__fp16 Value)
{
    return svdup_f16(Value);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT16
MlasSveMinfloat16(MLAS_SVBOOL pg, MLAS_SVFLOAT16 x, MLAS_SVFLOAT16 range)
{
    return svmin_f16_m(pg, x, range);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT16
MlasSveMaxfloat16(MLAS_SVBOOL pg, MLAS_SVFLOAT16 x, MLAS_SVFLOAT16 range)
{
    return svmax_f16_m(pg, x, range);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT16
MlasSveMulfloat16(MLAS_SVBOOL pg, MLAS_SVFLOAT16 x, MLAS_SVFLOAT16 y)
{
    return svmul_f16_m(pg, x, y);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT16
MlasSveMLAfloat16(MLAS_SVBOOL pg, MLAS_SVFLOAT16 x, MLAS_SVFLOAT16 y, MLAS_SVFLOAT16 z)
{
    return svmla_f16_m(pg, x, y, z);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT16
MlasSveDivfloat16(MLAS_SVBOOL pg, MLAS_SVFLOAT16 x, MLAS_SVFLOAT16 y)
{
    return svdiv_f16_m(pg, x, y);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVBOOL
MlasSveSelPredictefloat16(size_t x, size_t y)
{
    return svwhilelt_b16(x, y);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT16
MlasSvereinterpretf16_u16(MLAS_SVUINT16 x)
{
    return svreinterpret_f16_u16(x);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVUINT16
MlasSveLoadUint16(MLAS_SVBOOL pg, const uint16_t* x)
{
    return svld1_u16(pg, x);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT16
MlasSveLoadFloat16(MLAS_SVBOOL pg, const MLAS_FP16* x)
{
    return svld1_f16(pg, reinterpret_cast<const __fp16*>(x));
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
void
MlasSveStoreUint16(MLAS_SVBOOL pg, uint16_t* Buffer, MLAS_SVUINT16 Vector)
{
    return svst1_u16(pg, Buffer, Vector);
}
MLAS_SVE_TARGET
MLAS_FORCEINLINE
void
MlasSveStoreF16(MLAS_SVBOOL pg, MLAS_FP16* Buffer, MLAS_SVFLOAT16 Vector)
{
    return svst1_f16(pg, reinterpret_cast<__fp16*>(Buffer), Vector);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVUINT16
MlasSvereinterpretu16_f16(MLAS_SVFLOAT16 x)
{
    return svreinterpret_u16_f16(x);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT16
MlasSveReciprocalfloat16(MLAS_SVFLOAT16 x)
{
    return svrecpe_f16(x);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT16
MlasSveReciprocalStepfloat16(MLAS_SVFLOAT16 x, MLAS_SVFLOAT16 y)
{
    return svrecps_f16(x, y);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT16
MlasSveSelectfloat16(MLAS_SVBOOL Pred, MLAS_SVFLOAT16 x, MLAS_SVFLOAT16 y)
{
    return svsel_f16(Pred, x, y);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT16
MlasSveSubtractfloat16(MLAS_SVBOOL Pred, MLAS_SVFLOAT16 x, MLAS_SVFLOAT16 y)
{
    return svsub_f16_m(Pred, x, y);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVBOOL
MlasSveComparelessthanfloat16(MLAS_SVBOOL Pred, MLAS_SVFLOAT16 x, MLAS_SVFLOAT16 y)
{
    return svcmplt_f16(Pred, x, y);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT16
MlasSveAbsolutefloat16(MLAS_SVFLOAT16 inactive, MLAS_SVBOOL Pred, MLAS_SVFLOAT16 y)
{
    return svabs_f16_m(inactive, Pred, y);
}

MLAS_SVE_TARGET
MLAS_FORCEINLINE
MLAS_SVFLOAT16
MlasSveAddfloat16(MLAS_SVBOOL Pred, MLAS_SVFLOAT16 x, MLAS_SVFLOAT16 y)
{
    return svadd_f16_m(Pred, x, y);
}
#endif
