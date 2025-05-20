/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    dequantize.cpp

Abstract:

    This module implements routines to dequantize buffers.

    The dequantization formula as specified in the ONNX operator documentation is:

        Output = (Input - ZeroPoint) * Scale

--*/

#include "mlasi.h"

//
// DequantizeLinear reference implementation using the C++ runtime.
//

template<typename InputType>
static
MLAS_FORCEINLINE
void
MlasDequantizeLinearRefImpl(
    const InputType* Input,
    float* Output,
    size_t N,
    float Scale,
    InputType ZeroPoint
    )
/*++

Routine Description:

    This routine quantizes the input buffer using the supplied quantization
    parameters.

Arguments:

    Input - Supplies the input buffer with quantized data.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

    Scale - Supplies the quantization scale.

    ZeroPoint - Supplies the quantization zero point value.

Return Value:

    None.

--*/
{
    int32_t ZeroPointS32 = static_cast<int32_t>(ZeroPoint);

    for (size_t n = 0; n < N; n++) {
        Output[n] = static_cast<float>(static_cast<int32_t>(Input[n]) - ZeroPointS32) * Scale;
    }
}

#if defined(MLAS_SSE2_INTRINSICS)
static
MLAS_FORCEINLINE
MLAS_INT32X4
MlasLoad4CharAsInt32x4(
    const int8_t* bytes
    )
{
    // Loads 4 int8s in an array into an INT32x4 where each int8 is sign-exteded to int32.
    auto Packed8 = _mm_loadu_si32(bytes);
    auto Zero = _mm_setzero_si128();
    auto SignMask8 = _mm_cmpgt_epi8(Zero, Packed8);
    auto Packed16 = _mm_unpacklo_epi8(Packed8, SignMask8);
    auto SignMask16 = _mm_cmpgt_epi16(Zero, Packed16);
    auto Packed32 = _mm_unpacklo_epi16(Packed16, SignMask16);
    return Packed32;
}

static
MLAS_FORCEINLINE
MLAS_INT32X4
MlasLoad4UCharAsInt32x4(
    const uint8_t* bytes
    )
{
    // Loads 4 uint8s in an array into an INT32x4 where each int8 is zero-exteded to int32.
    auto Packed8 = _mm_loadu_si32(bytes);
    auto Zero = _mm_setzero_si128();
    auto Packed16 = _mm_unpacklo_epi8(Packed8, Zero);
    auto Packed32 = _mm_unpacklo_epi16(Packed16, Zero);
    return Packed32;
}

void
MLASCALL
MlasDequantizeLinearS8Kernel(
    const int8_t* Input,
    float* Output,
    size_t N,
    float Scale,
    int8_t ZeroPoint
    )
{
    auto ZeroPointVector = MlasBroadcastInt32x4(ZeroPoint);
    auto ScaleVector = MlasBroadcastFloat32x4(Scale);

    while (N >= 4) {
        auto IntegerVector = MlasLoad4CharAsInt32x4(Input);
        auto NormIntegerVector = MlasSubtractInt32x4(IntegerVector, ZeroPointVector);
        auto NormFloatVector = MlasCastToFloat32x4(NormIntegerVector);
        auto ResultFloatVector = MlasMultiplyFloat32x4(NormFloatVector, ScaleVector);
        MlasStoreFloat32x4(Output, ResultFloatVector);

        Input += 4;
        Output += 4;
        N -= 4;
    }

    MlasDequantizeLinearRefImpl(Input, Output, N, Scale, ZeroPoint);
}

void
MLASCALL
MlasDequantizeLinearU8Kernel(
    const uint8_t* Input,
    float* Output,
    size_t N,
    float Scale,
    uint8_t ZeroPoint
    )
{
    auto ZeroPointVector = MlasBroadcastInt32x4(ZeroPoint);
    auto ScaleVector = MlasBroadcastFloat32x4(Scale);

    while (N >= 4) {
        auto IntegerVector = MlasLoad4UCharAsInt32x4(Input);
        auto NormIntegerVector = MlasSubtractInt32x4(IntegerVector, ZeroPointVector);
        auto NormFloatVector = MlasCastToFloat32x4(NormIntegerVector);
        auto ResultFloatVector = MlasMultiplyFloat32x4(NormFloatVector, ScaleVector);
        MlasStoreFloat32x4(Output, ResultFloatVector);

        Input += 4;
        Output += 4;
        N -= 4;
    }

    MlasDequantizeLinearRefImpl(Input, Output, N, Scale, ZeroPoint);
}

template<>
void
MLASCALL
MlasDequantizeLinear<int8_t>(
    const int8_t* Input,
    float* Output,
    size_t N,
    float Scale,
    int8_t ZeroPoint
    )
{
#if defined(MLAS_TARGET_AMD64)
    GetMlasPlatform().DequantizeLinearS8Kernel(
#else
    MlasDequantizeLinearS8Kernel(
#endif
        Input, Output, N, Scale, ZeroPoint);
}

template<>
void
MLASCALL
MlasDequantizeLinear<uint8_t>(
    const uint8_t* Input,
    float* Output,
    size_t N,
    float Scale,
    uint8_t ZeroPoint
    )
{
#if defined(MLAS_TARGET_AMD64)
    GetMlasPlatform().DequantizeLinearU8Kernel(
#else
    MlasDequantizeLinearU8Kernel(
#endif
        Input, Output, N, Scale, ZeroPoint);
}

#else
template<typename InputType>
void
MLASCALL
MlasDequantizeLinear(
    const InputType* Input,
    float* Output,
    size_t N,
    float Scale,
    InputType ZeroPoint
    )
{
    MlasDequantizeLinearRefImpl(Input, Output, N, Scale, ZeroPoint);
}

template
void
MLASCALL
MlasDequantizeLinear<int8_t>(
    const int8_t* Input,
    float* Output,
    size_t N,
    float Scale,
    int8_t ZeroPoint
    );

template
void
MLASCALL
MlasDequantizeLinear<uint8_t>(
    const uint8_t* Input,
    float* Output,
    size_t N,
    float Scale,
    uint8_t ZeroPoint
    );

#endif
