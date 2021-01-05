/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    quantize_avx512.cpp

Abstract:

    This module implements routines to quantize buffers with AVX512.

    For quantization formula as specified in the ONNX operator documentation is:

        Output = Saturate(RoundToEven(Input / Scale) + ZeroPoint)

--*/

#include "mlasi.h"

template<typename OutputType>
void
MLASCALL
MlasQuantizeLinearKernel(
    const float* Input,
    OutputType* Output,
    size_t N,
    float Scale,
    OutputType ZeroPoint
);


//
// QuantizeLinear implementation using AVX512 intrinsics.
//

template <typename OutputType>
void
MLASCALL
MlasQuantizeLinearAVX512(
    const float* Input,
    OutputType* Output,
    size_t N,
    float Scale,
    OutputType ZeroPoint)
/*++

Routine Description:

    This routine quantizes the input buffer using the supplied quantization
    parameters.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

    Scale - Supplies the quantization scale.

    ZeroPoint - Supplies the quantization zero point value.

Return Value:

    None.

--*/
{
  constexpr int32_t MinimumValue = std::numeric_limits<OutputType>::min();
  constexpr int32_t MaximumValue = std::numeric_limits<OutputType>::max();

  auto ScaleVector = _mm512_set1_ps(Scale);
  auto MinimumValueVector = _mm512_set1_ps(float(MinimumValue - ZeroPoint));
  auto MaximumValueVector = _mm512_set1_ps(float(MaximumValue - ZeroPoint));
  auto ZeroPointVector = _mm512_set1_epi32(ZeroPoint);

  while (N >= 64) {
    auto FloatVector0 = _mm512_loadu_ps(Input);
    auto FloatVector1 = _mm512_loadu_ps(Input + 16);
    auto FloatVector2 = _mm512_loadu_ps(Input + 32);
    auto FloatVector3 = _mm512_loadu_ps(Input + 48);

    FloatVector0 = _mm512_div_ps(FloatVector0, ScaleVector);
    FloatVector1 = _mm512_div_ps(FloatVector1, ScaleVector);
    FloatVector2 = _mm512_div_ps(FloatVector2, ScaleVector);
    FloatVector3 = _mm512_div_ps(FloatVector3, ScaleVector);

    FloatVector0 = _mm512_max_ps(FloatVector0, MinimumValueVector);
    FloatVector1 = _mm512_max_ps(FloatVector1, MinimumValueVector);
    FloatVector2 = _mm512_max_ps(FloatVector2, MinimumValueVector);
    FloatVector3 = _mm512_max_ps(FloatVector3, MinimumValueVector);

    FloatVector0 = _mm512_min_ps(FloatVector0, MaximumValueVector);
    FloatVector1 = _mm512_min_ps(FloatVector1, MaximumValueVector);
    FloatVector2 = _mm512_min_ps(FloatVector2, MaximumValueVector);
    FloatVector3 = _mm512_min_ps(FloatVector3, MaximumValueVector);

    auto IntegerVector0 = _mm512_cvtps_epi32(FloatVector0);
    auto IntegerVector1 = _mm512_cvtps_epi32(FloatVector1);
    auto IntegerVector2 = _mm512_cvtps_epi32(FloatVector2);
    auto IntegerVector3 = _mm512_cvtps_epi32(FloatVector3);

    IntegerVector0 = _mm512_add_epi32(IntegerVector0, ZeroPointVector);
    IntegerVector1 = _mm512_add_epi32(IntegerVector1, ZeroPointVector);
    IntegerVector2 = _mm512_add_epi32(IntegerVector2, ZeroPointVector);
    IntegerVector3 = _mm512_add_epi32(IntegerVector3, ZeroPointVector);

    auto ByteVector0 = _mm512_cvtepi32_epi8(IntegerVector0);
    auto ByteVector1 = _mm512_cvtepi32_epi8(IntegerVector1);
    auto ByteVector2 = _mm512_cvtepi32_epi8(IntegerVector2);
    auto ByteVector3 = _mm512_cvtepi32_epi8(IntegerVector3);
    _mm_storeu_si128((__m128i*)Output, ByteVector0);
    _mm_storeu_si128((__m128i*)(Output+16), ByteVector1);
    _mm_storeu_si128((__m128i*)(Output+32), ByteVector2);
    _mm_storeu_si128((__m128i*)(Output+48), ByteVector3);

    Input += 64;
    Output += 64;
    N -= 64;
  }

  while (N >= 16) {
      auto FloatVector = _mm512_loadu_ps(Input);
      FloatVector = _mm512_div_ps(FloatVector, ScaleVector);
      FloatVector = _mm512_max_ps(FloatVector, MinimumValueVector);
      FloatVector = _mm512_min_ps(FloatVector, MaximumValueVector);

      auto IntegerVector = _mm512_cvtps_epi32(FloatVector);
      IntegerVector = _mm512_add_epi32(IntegerVector, ZeroPointVector);

      auto ByteVector = _mm512_cvtepi32_epi8(IntegerVector);
      _mm_storeu_si128((__m128i*)Output, ByteVector);

      Input += 16;
      Output += 16;
      N -= 16;
  }

  MlasQuantizeLinearKernel(Input, Output, N, Scale, ZeroPoint);
}

void
    MLASCALL
    MlasQuantizeLinearU8KernalAVX512(
        const float* Input,
        uint8_t* Output,
        size_t N,
        float Scale,
        uint8_t ZeroPoint) {
  MlasQuantizeLinearAVX512<uint8_t>(Input, Output, N, Scale, ZeroPoint);
}

void
    MLASCALL
    MlasQuantizeLinearS8KernalAVX512(
        const float* Input,
        int8_t* Output,
        size_t N,
        float Scale,
        int8_t ZeroPoint) {
  MlasQuantizeLinearAVX512<int8_t>(Input, Output, N, Scale, ZeroPoint);
}
