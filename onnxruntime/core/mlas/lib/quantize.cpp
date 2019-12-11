/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    quantize.cpp

Abstract:

    This module implements routines to quantize buffers.

--*/

#include "mlasi.h"

void
MLASCALL
MlasQuantizeLinear(
    const float* Input,
    uint8_t* Output,
    size_t N,
    float Scale,
    uint8_t ZeroPoint
    )
{
    auto ScaleInvertedPacket = MlasPacketBroadcast<MLAS_FLOAT32X4>(1.0f / Scale);
    auto ZeroPointPacket = MlasPacketBroadcast<MLAS_FLOAT32X4>(float(ZeroPoint));
    auto MinimumValuePacket = MlasPacketBroadcast<MLAS_FLOAT32X4>(0.0f);
    auto MaximumValuePacket = MlasPacketBroadcast<MLAS_FLOAT32X4>(255.0f);

    while (N >= 4) {

        auto Value = MlasPacketLoad<MLAS_FLOAT32X4>(Input);

        Value = MlasPacketMultiply(Value, ScaleInvertedPacket);
        Value = _mm_round_ps(Value, 0);
        Value = MlasPacketAdd(Value, ZeroPointPacket);

        Value = MlasPacketMaximum(Value, MinimumValuePacket);
        Value = MlasPacketMinimum(Value, MaximumValuePacket);

        __m128i ValueI = _mm_cvtps_epi32(Value);

        ValueI = _mm_packus_epi32(ValueI, ValueI);
        ValueI = _mm_packus_epi16(ValueI, ValueI);

        *((int32_t*)Output) = _mm_cvtsi128_si32(ValueI);

        Input += 4;
        Output += 4;
        N -= 4;
    }

    while (N > 0) {
        __debugbreak();
    }
}

void
MLASCALL
MlasQuantizeLinear(
    const float* Input,
    int8_t* Output,
    size_t N,
    float Scale,
    int8_t ZeroPoint
    )
{
    MLAS_UNREFERENCED_PARAMETER(Input);
    MLAS_UNREFERENCED_PARAMETER(Output);
    MLAS_UNREFERENCED_PARAMETER(N);
    MLAS_UNREFERENCED_PARAMETER(Scale);
    MLAS_UNREFERENCED_PARAMETER(ZeroPoint);

    __debugbreak();
}
