/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    quantize.cpp

Abstract:

    This module implements routines to quantize buffers.

    For quantization formula as specified in the ONNX operator documentation is:

        Output = Saturate(RoundToEven(Input / Scale) + ZeroPoint)

--*/

#include "mlasi.h"

#if defined(MLAS_NEON64_INTRINSICS) || defined(MLAS_SSE2_INTRINSICS)

//
// QuantizeLinear implementation using NEON or SSE2 intrinsics.
//

MLAS_FORCEINLINE
MLAS_INT32X4
MlasQuantizeLinearVector(
    MLAS_FLOAT32X4 FloatVector,
    MLAS_FLOAT32X4 ScaleVector,
    MLAS_FLOAT32X4 MinimumValueVector,
    MLAS_FLOAT32X4 MaximumValueVector,
    MLAS_INT32X4 ZeroPointVector
    )
{
    //
    // Scale the input vector and clamp the values to the minimum and maximum
    // range (adjusted by the zero point value).
    //

    FloatVector = MlasDivideFloat32x4(FloatVector, ScaleVector);

#if defined(MLAS_NEON64_INTRINSICS)
    // N.B. FMINNM and FMAXNM returns the numeric value if either of the values
    // is a NaN.
    FloatVector = vmaxnmq_f32(FloatVector, MinimumValueVector);
    FloatVector = vminnmq_f32(FloatVector, MaximumValueVector);
#else
    // N.B. MINPS and MAXPS returns the value from the second vector if the
    // value from the first vector is a NaN.
    FloatVector = _mm_max_ps(FloatVector, MinimumValueVector);
    FloatVector = _mm_min_ps(FloatVector, MaximumValueVector);
#endif

    //
    // Convert the float values to integer using "round to nearest even" and
    // then shift the output range using the zero point value.
    //

#if defined(MLAS_NEON64_INTRINSICS)
    auto IntegerVector = vcvtnq_s32_f32(FloatVector);
    IntegerVector = vaddq_s32(IntegerVector, ZeroPointVector);
#else
    // N.B. Assumes MXCSR has been configured with the default rounding mode of
    // "round to nearest even".
    auto IntegerVector = _mm_cvtps_epi32(FloatVector);
    IntegerVector = _mm_add_epi32(IntegerVector, ZeroPointVector);
#endif

    return IntegerVector;
}

MLAS_FORCEINLINE
__m256i
MlasQuantizeLinearVector(
   __m256 FloatVector,
   __m256 ScaleVector,
   __m256 MinimumValueVector,
   __m256 MaximumValueVector,
   __m256i ZeroPointVector
)
{
   //
   // Scale the input vector and clamp the values to the minimum and maximum
   // range (adjusted by the zero point value).
   //

   FloatVector = _mm256_div_ps( FloatVector, ScaleVector );

   // N.B. MINPS and MAXPS returns the value from the second vector if the
   // value from the first vector is a NaN.
   FloatVector = _mm256_max_ps( FloatVector, MinimumValueVector );
   FloatVector = _mm256_min_ps( FloatVector, MaximumValueVector );

   //
   // Convert the float values to integer using "round to nearest even" and
   // then shift the output range using the zero point value.
   //

    // N.B. Assumes MXCSR has been configured with the default rounding mode of
    // "round to nearest even".
   auto IntegerVector = _mm256_cvtps_epi32( FloatVector );
   IntegerVector = _mm256_add_epi32( IntegerVector, ZeroPointVector );

   return IntegerVector;
}


template<typename OutputType>
MLAS_INT32X4
MlasQuantizeLinearPackBytes(
    MLAS_INT32X4 IntegerVector
    );

template<typename OutputType>
__m128i
MlasQuantizeLinearPackBytes(
   __m256i IntegerVector
);

#if defined(MLAS_NEON64_INTRINSICS)

template<typename OutputType>
MLAS_INT32X4
MlasQuantizeLinearPackBytes(
    MLAS_INT32X4 IntegerVector
    )
{
    //
    // Swizzle the least significant byte from each int32_t element to the
    // bottom four bytes of the vector register.
    //

    uint16x8_t WordVector = vreinterpretq_u16_s32(IntegerVector);
    WordVector = vuzp1q_u16(WordVector, WordVector);
    uint8x16_t ByteVector = vreinterpretq_u8_u16(WordVector);
    ByteVector = vuzp1q_u8(ByteVector, ByteVector);

    return vreinterpretq_s32_u8(ByteVector);
}

#else

template<>
MLAS_FORCEINLINE
MLAS_INT32X4
MlasQuantizeLinearPackBytes<uint8_t>(
    MLAS_INT32X4 IntegerVector
    )
{
    IntegerVector = _mm_packus_epi16(IntegerVector, IntegerVector);
    IntegerVector = _mm_packus_epi16(IntegerVector, IntegerVector);

    return IntegerVector;
}

template<>
MLAS_FORCEINLINE
MLAS_INT32X4
MlasQuantizeLinearPackBytes<int8_t>(
    MLAS_INT32X4 IntegerVector
    )
{
    IntegerVector = _mm_packs_epi16(IntegerVector, IntegerVector);
    IntegerVector = _mm_packs_epi16(IntegerVector, IntegerVector);

    return IntegerVector;
}

template<>
MLAS_FORCEINLINE
__m128i
MlasQuantizeLinearPackBytes<uint8_t>(
   __m256i IntegerVector
   )
{
   __m128i low = _mm256_castsi256_si128( IntegerVector );
   __m128i high = _mm256_extractf128_si256( IntegerVector, 1 );
   low = _mm_packus_epi16( low, high );
   low = _mm_packus_epi16( low, low );

   return low;
}

template<>
MLAS_FORCEINLINE
__m128i
MlasQuantizeLinearPackBytes<int8_t>(
   __m256i IntegerVector
   )
{
   __m128i low = _mm256_castsi256_si128( IntegerVector );
   __m128i high = _mm256_extractf128_si256( IntegerVector, 1 );
   low = _mm_packs_epi16( low, high );
   low = _mm_packs_epi16( low, low );

   return low;
}

#endif

#if 0
template<typename OutputType>
void
MLASCALL
MlasQuantizeLinear(
    const float* Input,
    OutputType* Output,
    size_t N,
    float Scale,
    OutputType ZeroPoint
    )
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

    auto ScaleVector = MlasBroadcastFloat32x4(Scale);
    auto MinimumValueVector = MlasBroadcastFloat32x4(float(MinimumValue - ZeroPoint));
    auto MaximumValueVector = MlasBroadcastFloat32x4(float(MaximumValue - ZeroPoint));
    auto ZeroPointVector = MlasBroadcastInt32x4(ZeroPoint);

    while (N >= 4) {

        auto FloatVector = MlasLoadFloat32x4(Input);
        auto IntegerVector = MlasQuantizeLinearVector(FloatVector, ScaleVector,
            MinimumValueVector, MaximumValueVector, ZeroPointVector);

        IntegerVector = MlasQuantizeLinearPackBytes<OutputType>(IntegerVector);

#if defined(MLAS_NEON64_INTRINSICS)
        vst1q_lane_s32((int32_t*)Output, IntegerVector, 0);
#else
        *((int32_t*)Output) = _mm_cvtsi128_si32(IntegerVector);
#endif

        Input += 4;
        Output += 4;
        N -= 4;
    }

    for (size_t n = 0; n < N; n++) {

#if defined(MLAS_NEON64_INTRINSICS)
        auto FloatVector = vld1q_dup_f32(Input + n);
#else
        auto FloatVector = _mm_load_ss(Input + n);
#endif
        auto IntegerVector = MlasQuantizeLinearVector(FloatVector, ScaleVector,
            MinimumValueVector, MaximumValueVector, ZeroPointVector);

#if defined(MLAS_NEON64_INTRINSICS)
        vst1q_lane_u8((uint8_t*)Output + n, vreinterpretq_u8_s32(IntegerVector), 0);
#else
        *((uint8_t*)Output + n) = (uint8_t)_mm_cvtsi128_si32(IntegerVector);
#endif
    }
}
#endif

template<typename OutputType>
void
MLASCALL
MlasQuantizeLinear(
   const float* Input,
   OutputType* Output,
   size_t N,
   float Scale,
   OutputType ZeroPoint
)
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

   auto ScaleVector = _mm256_set1_ps( Scale );
   auto MinimumValueVector = _mm256_set1_ps( float( MinimumValue - ZeroPoint ) );
   auto MaximumValueVector = _mm256_set1_ps( float( MaximumValue - ZeroPoint ) );
   auto ZeroPointVector = _mm256_set1_epi32( ZeroPoint );

   while( N >= 8 )
   {

      auto FloatVector = _mm256_loadu_ps( Input );
      auto IntegerVector = MlasQuantizeLinearVector( FloatVector, ScaleVector,
         MinimumValueVector, MaximumValueVector, ZeroPointVector );

      auto PackedVector = MlasQuantizeLinearPackBytes<OutputType>( IntegerVector );
      * ( ( int64_t* ) Output ) = _mm_cvtsi128_si64( PackedVector );

      Output += 8;
      Input += 8;
      N -= 8;
   }

   for( size_t n = 0; n < N; n++ )
   {
      auto FloatVector = _mm256_loadu_ps( Input + n );
      auto IntegerVector = MlasQuantizeLinearVector( FloatVector, ScaleVector,
         MinimumValueVector, MaximumValueVector, ZeroPointVector );

      * ( ( uint8_t* ) Output + n ) = ( uint8_t ) _mm256_cvtsi256_si32( IntegerVector );
   }
}

#else

//
// QuantizeLinear implementation using the C++ runtime.
//

template<typename OutputType>
void
MLASCALL
MlasQuantizeLinear(
    const float* Input,
    OutputType* Output,
    size_t N,
    float Scale,
    OutputType ZeroPoint
    )
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

    for (size_t n = 0; n < N; n++) {

        float FloatValue = std::nearbyintf(Input[n] / Scale) + float(ZeroPoint);
        FloatValue = std::max(FloatValue, float(MinimumValue));
        FloatValue = std::min(FloatValue, float(MaximumValue));
        Output[n] = (OutputType)(int32_t)FloatValue;
    }
}

#endif

template
void
MLASCALL
MlasQuantizeLinear<int8_t>(
    const float* Input,
    int8_t* Output,
    size_t N,
    float Scale,
    int8_t ZeroPoint
    );

template
void
MLASCALL
MlasQuantizeLinear<uint8_t>(
    const float* Input,
    uint8_t* Output,
    size_t N,
    float Scale,
    uint8_t ZeroPoint
    );

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

#endif

//void
//MLASCALL
//MlasMinMaxElement(
//   const float* a,
//   float* min,
//   float* max,
//   size_t N)
//{
//   if( N <= 0 )
//   {
//      *min = 0.0f;
//      *max = 0.0f;
//      return;
//   }
//
//   float temp_min = *a, temp_max = *a;
//   int i = 0;
//
////#ifdef __AVX__
//   __m256 min_v = _mm256_set1_ps( *a );
//   __m256 max_v = _mm256_set1_ps( *a );
//   constexpr int VLEN = 8;
//   if( N >= VLEN )
//   {
//      for( ; i < N / VLEN * VLEN; i += VLEN )
//      {
//         min_v = _mm256_min_ps( min_v, _mm256_loadu_ps( a + i ) );
//         max_v = _mm256_max_ps( max_v, _mm256_loadu_ps( a + i ) );
//      }
//
//      float min_buf[ VLEN ], max_buf[ VLEN ];
//      _mm256_storeu_ps( min_buf, min_v );
//      _mm256_storeu_ps( max_buf, max_v );
//      for( int j = 0; j < VLEN; ++j )
//      {
//         temp_min = std::min( temp_min, min_buf[ j ] );
//         temp_max = std::max( temp_max, max_buf[ j ] );
//      }
//   }
////#endif
//
//   for( ; i < N; i++ )
//   {
//      temp_min = std::min( temp_min, a[ i ] );
//      temp_max = std::max( temp_max, a[ i ] );
//   }
//   *min = temp_min;
//   *max = temp_max;
//}


void
MLASCALL
MlasMinMaxElement(
   const float* Input,
   float* min,
   float* max,
   size_t N )
/*++

Routine Description:

    This routine implements the generic kernel to find the maximum value of
    the supplied buffer.

Arguments:

    Input - Supplies the input buffer.

    N - Supplies the number of elements to process.

Return Value:

    Returns the maximum value of the supplied buffer.

--*/
{
   if( N <= 0 )
   {
      *min = 0.0f;
      *max = 0.0f;
      return;
   }

   if( N >= 8 )
   {

      __m256 MaximumVector0 = _mm256_set1_ps( *Input );
      __m256 MinimumVector0 = _mm256_set1_ps( *Input );

      if( N >= 32 )
      {

         __m256 MaximumVector1 = MaximumVector0;
         __m256 MaximumVector2 = MaximumVector0;
         __m256 MaximumVector3 = MaximumVector0;

         __m256 MinimumVector1 = MinimumVector0;
         __m256 MinimumVector2 = MinimumVector0;
         __m256 MinimumVector3 = MinimumVector0;

         while( N >= 32 )
         {

            __m256 InputVector0 = _mm256_loadu_ps( Input );
            __m256 InputVector1 = _mm256_loadu_ps( Input + 8 );
            __m256 InputVector2 = _mm256_loadu_ps( Input + 16);
            __m256 InputVector3 = _mm256_loadu_ps( Input + 24);

            MaximumVector0 = _mm256_max_ps( MaximumVector0, InputVector0 );
            MaximumVector1 = _mm256_max_ps( MaximumVector1, InputVector1 );
            MaximumVector2 = _mm256_max_ps( MaximumVector2, InputVector2 );
            MaximumVector3 = _mm256_max_ps( MaximumVector3, InputVector3);

            MinimumVector0 = _mm256_min_ps( MinimumVector0, InputVector0 );
            MinimumVector1 = _mm256_min_ps( MinimumVector1, InputVector1 );
            MinimumVector2 = _mm256_min_ps( MinimumVector2, InputVector2 );
            MinimumVector3 = _mm256_min_ps( MinimumVector3, InputVector3 );

            Input += 32;
            N -= 32;
         }

         MaximumVector0 = _mm256_max_ps( MaximumVector0, MaximumVector1 );
         MaximumVector2 = _mm256_max_ps( MaximumVector2, MaximumVector3 );
         MaximumVector0 = _mm256_max_ps( MaximumVector0, MaximumVector2 );

         MinimumVector0 = _mm256_min_ps( MinimumVector0, MinimumVector1 );
         MinimumVector2 = _mm256_min_ps( MinimumVector2, MinimumVector3 );
         MinimumVector0 = _mm256_min_ps( MinimumVector0, MinimumVector2 );
      }

      while( N >= 8 )
      {
         __m256 InputVector0 = _mm256_loadu_ps( Input );
         MaximumVector0 = _mm256_max_ps( MaximumVector0, InputVector0 );
         MinimumVector0 = _mm256_min_ps( MinimumVector0, InputVector0 );

         Input += 8;
         N -= 8;
      }

      float min_buf[ 8 ], max_buf[ 8 ];
      _mm256_storeu_ps( min_buf, MinimumVector0 );
      _mm256_storeu_ps( max_buf, MaximumVector0 );
      for( int j = 0; j < 8; ++j )
      {
         *min = std::min( *min, min_buf[ j ] );
         *max = std::max( *max, max_buf[ j ] );
      }
   }

   while( N > 0 )
   {

      *max = std::max( *max, *Input );
      *min = std::min( *min, *Input );

      Input += 1;
      N -= 1;
   }
}
