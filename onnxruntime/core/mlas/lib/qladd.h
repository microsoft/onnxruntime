/*++

Copyright (c) Microsoft Corporation.  All rights reserved.

Licensed under the MIT License.

Module Name:

    qladd.h

Abstract:

    This module contains the private data structures and procedure prototypes
    for QLinearAdd function usage.

--*/

#pragma once

#include "mlasi.h"

union Float32Bits {
    uint32_t u32;
    float    fp32;
};

MLAS_FORCEINLINE
uint32_t
BitsOfFp32(float f)
{
    Float32Bits uf;
    uf.fp32 = f;
    return uf.u32;
}

MLAS_FORCEINLINE
float
Fp32FromBits(uint32_t u)
{
    Float32Bits uf = { u };
    return uf.fp32;
}

bool
CalcQLinearAddParameters(
    float ScaleRatio_AC, float ScaleRatio_BC,
    int32_t& Shift, int32_t& MultiplierA, int32_t& MultiplierB
    );
