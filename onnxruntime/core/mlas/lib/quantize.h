/*++

Copyright (c) Microsoft Corporation.  All rights reserved.

Licensed under the MIT License.

Module Name:

    quantize.h

Abstract:

    This module contains the private data structures and procedure prototypes
    shared by general quantization operations.

--*/

#pragma once

#include "mlasi.h"

union MLAS_FLOAT32BITS {
    uint32_t u32;
    float fp32;
};

MLAS_FORCEINLINE
static uint32_t
MlasBitsOfFp32(
    float f
    )
{
    MLAS_FLOAT32BITS uf;
    uf.fp32 = f;
    return uf.u32;
}

MLAS_FORCEINLINE
static float
MlasFp32FromBits(
    uint32_t u
    )
{
    MLAS_FLOAT32BITS uf = {u};
    return uf.fp32;
}

MLAS_FORCEINLINE
static uint8_t*
MlasCopyTailBytes(
    uint8_t* target,
    const uint8_t* src,
    size_t N
    )
{
    uint8_t* dst = target;
    while (N >= sizeof(uint32_t)) {
        *(uint32_t*)dst = *(uint32_t*)src;
        N -= sizeof(uint32_t);
        dst += sizeof(uint32_t);
        src += sizeof(uint32_t);
    }
    while (N > 0) {
        *dst++ = *src++;
        --N;
    }
    return target;
}

#if defined(MLAS_SSE2_INTRINSICS)

MLAS_FORCEINLINE
MLAS_INT32X4
MlasRequantizeOutputVector(
    MLAS_INT32X4 IntegerVector,
    MLAS_INT32X4 BiasVector,
    MLAS_FLOAT32X4 ScaleVector,
    MLAS_FLOAT32X4 MinimumValueVector,
    MLAS_FLOAT32X4 MaximumValueVector,
    MLAS_INT32X4 ZeroPointVector
    )
{
    IntegerVector = _mm_add_epi32(IntegerVector, BiasVector);
    MLAS_FLOAT32X4 FloatVector = _mm_cvtepi32_ps(IntegerVector);

    //
    // Scale the input vector and clamp the values to the minimum and maximum
    // range (adjusted by the zero point value).
    //

    FloatVector = MlasMultiplyFloat32x4(FloatVector, ScaleVector);

    // N.B. MINPS and MAXPS returns the value from the second vector if the
    // value from the first vector is a NaN.
    FloatVector = _mm_max_ps(FloatVector, MinimumValueVector);
    FloatVector = _mm_min_ps(FloatVector, MaximumValueVector);

    //
    // Convert the float values to integer using "round to nearest even" and
    // then shift the output range using the zero point value.
    //

    // N.B. Assumes MXCSR has been configured with the default rounding mode of
    // "round to nearest even".
    IntegerVector = _mm_cvtps_epi32(FloatVector);
    IntegerVector = _mm_add_epi32(IntegerVector, ZeroPointVector);

    return IntegerVector;
}

#endif
