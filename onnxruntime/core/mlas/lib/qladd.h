/*++

Copyright (c) Microsoft Corporation.  All rights reserved.

Licensed under the MIT License.

Module Name:

    qladd.h

Abstract:

    This module contains the private data structures and procedure prototypes
    for QLinearAdd function usage .

--*/

#pragma once

#include "mlasi.h"

union MLAS_FLOAT32BITS
{
    uint32_t u32;
    float    fp32;
};

MLAS_FORCEINLINE
static
uint32_t
MlasBitsOfFp32(
    float f
    )
{
    MLAS_FLOAT32BITS uf;
    uf.fp32 = f;
    return uf.u32;
}

MLAS_FORCEINLINE
static
float
MlasFp32FromBits(
    uint32_t u
    )
{
    MLAS_FLOAT32BITS uf = { u };
    return uf.fp32;
}

MLAS_FORCEINLINE
static
void
MlasCopyTailBytes(
    uint8_t* target,
    const uint8_t* src,
    size_t N)
{
    while (N >= sizeof(uint32_t)) {
        *(uint32_t*)(target) = *(uint32_t*)(src);
        N -= sizeof(uint32_t);
        target += sizeof(uint32_t);
        src += sizeof(uint32_t);
    }
    while (N > 0) {
        *target++ = *src++;
        --N;
    }
}

bool
MlasCalcQLinearAddParameters(
    float ScaleRatio_AC,
    float ScaleRatio_BC,
    int32_t& Shift,
    int32_t& MultiplierA,
    int32_t& MultiplierB
    );
