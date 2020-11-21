/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    requantize.cpp

Abstract:

    This module implements routines to requantize buffers.

--*/

#include "mlasi.h"

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

void
MLASCALL
MlasRequantizeOutput(
    const int32_t* Input,
    uint8_t* Output,
    const int32_t* Bias,
    size_t M,
    size_t N,
    float Scale,
    uint8_t ZeroPoint
    )
/*++

Routine Description:

    This routine requantizes the intermediate buffer to the output buffer
    optionally adding the supplied bias.

Arguments:

    Input - Supplies the input matrix.

    Output - Supplies the output matrix.

    Bias - Supplies the optional bias vector to be added to the input buffer
        before requantization.

    Buffer - Supplies the output matrix.

    M - Supplies the number of elements of the bias vector and the number of
        rows in the output matrix.

    N - Supplies the number of columns of the output matrix.

    Scale - Supplies the quantization scale.

    ZeroPoint - Supplies the quantization zero point value.

Return Value:

    None.

--*/
{
    MLAS_FLOAT32X4 ScaleVector = MlasBroadcastFloat32x4(Scale);
    MLAS_FLOAT32X4 MinimumValueVector = MlasBroadcastFloat32x4(float(0 - ZeroPoint));
    MLAS_FLOAT32X4 MaximumValueVector = MlasBroadcastFloat32x4(float(255 - ZeroPoint));
    MLAS_INT32X4 ZeroPointVector = MlasBroadcastInt32x4(ZeroPoint);
    MLAS_INT32X4 BiasVector = _mm_setzero_si128();

    //
    // Step through each row of the output matrix.
    //

    while (M-- > 0) {

        if (Bias != nullptr) {
            BiasVector = MlasBroadcastInt32x4(*Bias++);
        }

        size_t n = N;

        while (n >= 4) {

            MLAS_INT32X4 IntegerVector = _mm_loadu_si128((const __m128i *)Input);
            IntegerVector = MlasRequantizeOutputVector(IntegerVector, BiasVector,
                ScaleVector, MinimumValueVector, MaximumValueVector, ZeroPointVector);

            IntegerVector = _mm_packus_epi16(IntegerVector, IntegerVector);
            IntegerVector = _mm_packus_epi16(IntegerVector, IntegerVector);

            *((int32_t*)Output) = _mm_cvtsi128_si32(IntegerVector);

            Input += 4;
            Output += 4;
            n -= 4;
        }

        while (n > 0) {

            MLAS_INT32X4 IntegerVector = _mm_cvtsi32_si128(*Input);
            IntegerVector = MlasRequantizeOutputVector(IntegerVector, BiasVector,
                ScaleVector, MinimumValueVector, MaximumValueVector, ZeroPointVector);

            *Output = (uint8_t)_mm_cvtsi128_si32(IntegerVector);

            Input += 1;
            Output += 1;
            n -= 1;
        }
    }
}

void
MLASCALL
MlasRequantizeOutputColumn(
    const int32_t* Input,
    uint8_t* Output,
    const int32_t* Bias,
    size_t M,
    size_t N,
    float Scale,
    uint8_t ZeroPoint
    )
/*++

Routine Description:

    This routine requantizes the intermediate buffer to the output buffer
    optionally adding the supplied bias.

Arguments:

    Input - Supplies the input matrix.

    Output - Supplies the output matrix.

    Bias - Supplies the optional bias vector to be added to the input buffer
        before requantization.

    Buffer - Supplies the output matrix.

    M - Supplies the number of elements of the bias vector and the number of
        rows in the output matrix.

    N - Supplies the number of columns of the output matrix.

    Scale - Supplies the quantization scale.

    ZeroPoint - Supplies the quantization zero point value.

Return Value:

    None.

--*/
{
    MLAS_FLOAT32X4 ScaleVector = MlasBroadcastFloat32x4(Scale);
    MLAS_FLOAT32X4 MinimumValueVector = MlasBroadcastFloat32x4(float(0 - ZeroPoint));
    MLAS_FLOAT32X4 MaximumValueVector = MlasBroadcastFloat32x4(float(255 - ZeroPoint));
    MLAS_INT32X4 ZeroPointVector = MlasBroadcastInt32x4(ZeroPoint);
    MLAS_INT32X4 BiasVector = _mm_setzero_si128();

    //
    // Step through each row of the output matrix.
    //

    while (M-- > 0) {

        const int32_t* bias = Bias;

        size_t n = N;

        while (n >= 4) {

            MLAS_INT32X4 IntegerVector = _mm_loadu_si128((const __m128i *)Input);

            if (bias != nullptr) {
                BiasVector = _mm_loadu_si128((const __m128i*)bias);
                bias += 4;
            }

            IntegerVector = MlasRequantizeOutputVector(IntegerVector, BiasVector,
                ScaleVector, MinimumValueVector, MaximumValueVector, ZeroPointVector);

            IntegerVector = _mm_packus_epi16(IntegerVector, IntegerVector);
            IntegerVector = _mm_packus_epi16(IntegerVector, IntegerVector);

            *((int32_t*)Output) = _mm_cvtsi128_si32(IntegerVector);

            Input += 4;
            Output += 4;
            n -= 4;
        }

        while (n > 0) {

            MLAS_INT32X4 IntegerVector = _mm_cvtsi32_si128(*Input);

            if (bias != nullptr) {
                BiasVector = _mm_cvtsi32_si128(*bias);
                bias += 1;
            }

            IntegerVector = MlasRequantizeOutputVector(IntegerVector, BiasVector,
                ScaleVector, MinimumValueVector, MaximumValueVector, ZeroPointVector);

            *Output = (uint8_t)_mm_cvtsi128_si32(IntegerVector);

            Input += 1;
            Output += 1;
            n -= 1;
        }
    }
}

void
MLASCALL
MlasRequantizeOutputColumn(
    const int32_t* Input,
    uint8_t* Output,
    const int32_t* Bias,
    size_t M,
    size_t N,
    const float* Scale,
    uint8_t ZeroPoint
    )
/*++

Routine Description:

    This routine requantizes the intermediate buffer to the output buffer
    optionally adding the supplied bias.

Arguments:

    Input - Supplies the input matrix.

    Output - Supplies the output matrix.

    Bias - Supplies the optional bias vector to be added to the input buffer
        before requantization.

    Buffer - Supplies the output matrix.

    M - Supplies the number of elements of the bias vector and the number of
        rows in the output matrix.

    N - Supplies the number of columns of the output matrix.

    Scale - Supplies the quantization scale vector.

    ZeroPoint - Supplies the quantization zero point value.

Return Value:

    None.

--*/
{
    MLAS_FLOAT32X4 MinimumValueVector = MlasBroadcastFloat32x4(float(0 - ZeroPoint));
    MLAS_FLOAT32X4 MaximumValueVector = MlasBroadcastFloat32x4(float(255 - ZeroPoint));
    MLAS_INT32X4 ZeroPointVector = MlasBroadcastInt32x4(ZeroPoint);
    MLAS_INT32X4 BiasVector = _mm_setzero_si128();

    //
    // Step through each row of the output matrix.
    //

    while (M-- > 0) {

        const int32_t* bias = Bias;
        const float* scale = Scale;

        size_t n = N;

        while (n >= 4) {

            MLAS_INT32X4 IntegerVector = _mm_loadu_si128((const __m128i *)Input);

            if (bias != nullptr) {
                BiasVector = _mm_loadu_si128((const __m128i*)bias);
                bias += 4;
            }

            MLAS_FLOAT32X4 ScaleVector = MlasLoadFloat32x4(scale);
            scale += 4;

            IntegerVector = MlasRequantizeOutputVector(IntegerVector, BiasVector,
                ScaleVector, MinimumValueVector, MaximumValueVector, ZeroPointVector);

            IntegerVector = _mm_packus_epi16(IntegerVector, IntegerVector);
            IntegerVector = _mm_packus_epi16(IntegerVector, IntegerVector);

            *((int32_t*)Output) = _mm_cvtsi128_si32(IntegerVector);

            Input += 4;
            Output += 4;
            n -= 4;
        }

        while (n > 0) {

            MLAS_INT32X4 IntegerVector = _mm_cvtsi32_si128(*Input);

            if (bias != nullptr) {
                BiasVector = _mm_cvtsi32_si128(*bias);
                bias += 1;
            }

            MLAS_FLOAT32X4 ScaleVector = _mm_load_ss(scale);
            scale += 1;

            IntegerVector = MlasRequantizeOutputVector(IntegerVector, BiasVector,
                ScaleVector, MinimumValueVector, MaximumValueVector, ZeroPointVector);

            *Output = (uint8_t)_mm_cvtsi128_si32(IntegerVector);

            Input += 1;
            Output += 1;
            n -= 1;
        }
    }
}

#endif
