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
    float ScaleRatio_AC, float ScaleRatio_BC, int32_t& Shift, int32_t& MultiplierA, int32_t& MultiplierB)
{
    constexpr float MinScaleRatio = 6.103515625e-05f; // std::stof("0x1.0p-14f");
    constexpr float MaxScaleRatio = 256.0f; //std::stof("0x1.0p+8f");
    if (ScaleRatio_AC < MinScaleRatio || ScaleRatio_AC >= MaxScaleRatio ||
            ScaleRatio_BC < MinScaleRatio || ScaleRatio_BC >= MaxScaleRatio) {
        return false;
    }

    const float GreaterScaleRatio = std::max(ScaleRatio_AC, ScaleRatio_BC);
    const int32_t GreaterExponent = (int32_t)(BitsOfFp32(GreaterScaleRatio) >> 23) - 127;
    Shift = 21 - GreaterExponent;
    if (Shift > 31 || Shift < 13) return false;

    const float MultiplierFloatValue = Fp32FromBits((uint32_t)(21 - GreaterExponent + 127) << 23);
    MultiplierA = (int32_t) lrintf(ScaleRatio_AC * MultiplierFloatValue);
    MultiplierB = (int32_t) lrintf(ScaleRatio_BC * MultiplierFloatValue);
    return ((MultiplierA < 0x00400000 && MultiplierB < 0x00400000) && 
           (MultiplierA >= 0x00200000 || MultiplierB >= 0x00200000)); // the greater one must fullfil this check

}
