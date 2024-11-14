/*++

Copyright (c) Intel Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    cast.cpp

Abstract:

    This module implements Half (F16) to Single (F32) precision casting.

--*/
#include "mlasi.h"

void
MLASCALL
MlasConvertHalfToFloatBuffer(
    const MLAS_FP16* Source,
    float* Destination,
    size_t Count
)
{
    if (GetMlasPlatform().CastF16ToF32Kernel == nullptr) {
        for (size_t i = 0; i < Count; ++i) {
            Destination[i] = Source[i].ToFloat();
        }
    } else {
        // If the kernel is available, use it to perform the conversion.
        GetMlasPlatform().CastF16ToF32Kernel(reinterpret_cast<const unsigned short*>(Source), Destination, Count);
    }
}

void
MLASCALL
MlasConvertFloatToHalfBuffer(
    const float* Source,
    MLAS_FP16* Destination,
    size_t Count
)
{
    if (GetMlasPlatform().CastF32ToF16Kernel == nullptr) {
        for (size_t i = 0; i < Count; ++i) {
            Destination[i] = MLAS_FP16(Source[i]);
        }
    } else {
        // If the kernel is available, use it to perform the conversion.
        GetMlasPlatform().CastF32ToF16Kernel(Source, reinterpret_cast<unsigned short*>(Destination), Count);
    }
}
