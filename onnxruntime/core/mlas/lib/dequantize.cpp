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
// Implementation for Intel SSE 2. Refer to the Intel Intrisics Guide:
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html

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
    const __m128 ScaleVector = MlasBroadcastFloat32x4(Scale);
    const __m128i ZeroPointS16Vector = _mm_set1_epi16(static_cast<int16_t>(ZeroPoint)); // Broadcast zp to 8 int16s
    const __m128i Zeros = _mm_setzero_si128();

    while (N >= 16) {
        // Load a vector of 16 int8s: [0 ... 15]
        __m128i VectorS8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(Input));

        // Sign-extend into 2 vectors of 8 int16s
        __m128i SignMaskS8 = _mm_cmpgt_epi8(Zeros, VectorS8); // 0xFF for every negative byte in VectorS8
        __m128i VectorS16_0 = _mm_unpacklo_epi8(VectorS8, SignMaskS8); // [0 ... 7]
        __m128i VectorS16_1 = _mm_unpackhi_epi8(VectorS8, SignMaskS8); // [8 ... 15]

        // Subtract the zero-points in int16 domain.
        VectorS16_0 = _mm_sub_epi16(VectorS16_0, ZeroPointS16Vector);
        VectorS16_1 = _mm_sub_epi16(VectorS16_1, ZeroPointS16Vector);

        // Sign-extend into 4 vectors of 4 int32s
        __m128i SignMaskS16_0 = _mm_cmpgt_epi16(Zeros, VectorS16_0);
        __m128i VectorS32_0 = _mm_unpacklo_epi16(VectorS16_0, SignMaskS16_0); // [0 ... 3]
        __m128i VectorS32_1 = _mm_unpackhi_epi16(VectorS16_0, SignMaskS16_0); // [4 ... 7]

        __m128i SignMaskS16_1 = _mm_cmpgt_epi16(Zeros, VectorS16_1);
        __m128i VectorS32_2 = _mm_unpacklo_epi16(VectorS16_1, SignMaskS16_1); // [8 ... 11]
        __m128i VectorS32_3 = _mm_unpackhi_epi16(VectorS16_1, SignMaskS16_1); // [12 ... 15]

        // Cast each int32x4 to float and multiply by the scale vector.
        __m128 VectorF32_0 = _mm_mul_ps(_mm_cvtepi32_ps(VectorS32_0), ScaleVector);
        __m128 VectorF32_1 = _mm_mul_ps(_mm_cvtepi32_ps(VectorS32_1), ScaleVector);
        __m128 VectorF32_2 = _mm_mul_ps(_mm_cvtepi32_ps(VectorS32_2), ScaleVector);
        __m128 VectorF32_3 = _mm_mul_ps(_mm_cvtepi32_ps(VectorS32_3), ScaleVector);

        // Store each int32x4 into the output.
        _mm_storeu_ps(Output + 0, VectorF32_0);
        _mm_storeu_ps(Output + 4, VectorF32_1);
        _mm_storeu_ps(Output + 8, VectorF32_2);
        _mm_storeu_ps(Output + 12, VectorF32_3);

        Input += 16;
        Output += 16;
        N -= 16;
    }

    // Handle leftover elements (< 16) with the scalar reference implementation.
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
    const __m128 ScaleVector = MlasBroadcastFloat32x4(Scale);
    const __m128i ZeroPointS16Vector = _mm_set1_epi16(static_cast<int16_t>(ZeroPoint)); // Broadcast zp to 8 int16s
    const __m128i Zeros = _mm_setzero_si128();

    while (N >= 16) {
        // Load a vector of 16 uint8s: [0 ... 15]
        __m128i VectorU8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(Input));

        // Zero-extend into 2 vectors of 8 uint16s
        __m128i VectorU16_0 = _mm_unpacklo_epi8(VectorU8, Zeros); // [0 ... 7]
        __m128i VectorU16_1 = _mm_unpackhi_epi8(VectorU8, Zeros); // [8 ... 15]

        // Subtract the zero-points as uint16s. Due to two's compliment, negative results can be reinterpreted as int16
        __m128i VectorS16_0 = _mm_sub_epi16(VectorU16_0, ZeroPointS16Vector);
        __m128i VectorS16_1 = _mm_sub_epi16(VectorU16_1, ZeroPointS16Vector);

        // Sign-extend into 4 vectors of 4 int32s
        __m128i SignMaskS16_0 = _mm_cmpgt_epi16(Zeros, VectorS16_0);
        __m128i VectorS32_0 = _mm_unpacklo_epi16(VectorS16_0, SignMaskS16_0); // [0 ... 3]
        __m128i VectorS32_1 = _mm_unpackhi_epi16(VectorS16_0, SignMaskS16_0); // [4 ... 7]

        __m128i SignMaskS16_1 = _mm_cmpgt_epi16(Zeros, VectorS16_1);
        __m128i VectorS32_2 = _mm_unpacklo_epi16(VectorS16_1, SignMaskS16_1); // [8 ... 11]
        __m128i VectorS32_3 = _mm_unpackhi_epi16(VectorS16_1, SignMaskS16_1); // [12 ... 15]

        // Cast each int32x4 to float and multiply by the scale vector.
        __m128 VectorF32_0 = _mm_mul_ps(_mm_cvtepi32_ps(VectorS32_0), ScaleVector);
        __m128 VectorF32_1 = _mm_mul_ps(_mm_cvtepi32_ps(VectorS32_1), ScaleVector);
        __m128 VectorF32_2 = _mm_mul_ps(_mm_cvtepi32_ps(VectorS32_2), ScaleVector);
        __m128 VectorF32_3 = _mm_mul_ps(_mm_cvtepi32_ps(VectorS32_3), ScaleVector);

        // Store each int32x4 into the output.
        _mm_storeu_ps(Output + 0, VectorF32_0);
        _mm_storeu_ps(Output + 4, VectorF32_1);
        _mm_storeu_ps(Output + 8, VectorF32_2);
        _mm_storeu_ps(Output + 12, VectorF32_3);

        Input += 16;
        Output += 16;
        N -= 16;
    }

    // Handle leftover elements (< 16) with the scalar reference implementation.
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
#elif defined(MLAS_NEON64_INTRINSICS)
// Implementation for ARM64 NEON. Refer to the ARM instrinsics guide:
// https://developer.arm.com/architectures/instruction-sets/intrinsics/

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
    const float32x4_t ScaleVector = MlasBroadcastFloat32x4(Scale);
    const int16x8_t ZeroPointVector = vdupq_n_s16(ZeroPoint); // Broadcast ZeroPoint (sign-extended to 16bits)

    while (N >= 16) {
        // Load a vector of 16 int8s: [0 ... 15]
        int8x16_t VectorS8 = vld1q_s8(Input);

        // Sign-extend into 2 vectors of 8 int16s
        int16x8_t VectorS16_0 = vmovl_s8(vget_low_s8(VectorS8));  // [0 ... 7]
        int16x8_t VectorS16_1 = vmovl_s8(vget_high_s8(VectorS8)); // [8 ... 15]

        // Subtract the zero-points in int16 domain.
        VectorS16_0 = vsubq_s16(VectorS16_0, ZeroPointVector);
        VectorS16_1 = vsubq_s16(VectorS16_1, ZeroPointVector);

        // Sign-extend into 4 vectors of 4 int32s
        int32x4_t VectorS32_0 = vmovl_s16(vget_low_s16(VectorS16_0));  // [0 ... 3]
        int32x4_t VectorS32_1 = vmovl_s16(vget_high_s16(VectorS16_0)); // [4 ... 7]
        int32x4_t VectorS32_2 = vmovl_s16(vget_low_s16(VectorS16_1));  // [8 ... 11]
        int32x4_t VectorS32_3 = vmovl_s16(vget_high_s16(VectorS16_1)); // [12 ... 15]

        // Cast each int32x4 to float and multiply by the scale vector.
        float32x4_t VectorF32_0 = vmulq_f32(vcvtq_f32_s32(VectorS32_0), ScaleVector);
        float32x4_t VectorF32_1 = vmulq_f32(vcvtq_f32_s32(VectorS32_1), ScaleVector);
        float32x4_t VectorF32_2 = vmulq_f32(vcvtq_f32_s32(VectorS32_2), ScaleVector);
        float32x4_t VectorF32_3 = vmulq_f32(vcvtq_f32_s32(VectorS32_3), ScaleVector);

        // Store each int32x4 into the output.
        vst1q_f32(Output + 0, VectorF32_0);
        vst1q_f32(Output + 4, VectorF32_1);
        vst1q_f32(Output + 8, VectorF32_2);
        vst1q_f32(Output + 12, VectorF32_3);

        N -= 16;
        Input += 16;
        Output += 16;
    }

    // Handle leftover elements (< 16) with the scalar reference implementation.
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
    const float32x4_t ScaleVector = MlasBroadcastFloat32x4(Scale);
    const uint8x8_t ZeroPointVector = vdup_n_u8(ZeroPoint); // Broadcast ZeroPoint to 8 uint8s

    while (N >= 16) {
        // Load a vector of 16 uint8s: [0 ... 15]
        uint8x16_t VectorU8 = vld1q_u8(Input);

        // Subtract zero-point. The vsubl_u8 instruction zero-extends its arguments to uint16 first.
        // The reinterpret from uint16x8 to int16x8 is actually a NOP.
        int16x8_t VectorS16_0 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(VectorU8), ZeroPointVector));  // [0 ... 7]
        int16x8_t VectorS16_1 = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(VectorU8), ZeroPointVector)); // [8 ... 15]

        // Sign-extend into 4 vectors of 4 int32s
        int32x4_t VectorS32_0 = vmovl_s16(vget_low_s16(VectorS16_0));  // [0 ... 3]
        int32x4_t VectorS32_1 = vmovl_s16(vget_high_s16(VectorS16_0)); // [4 ... 7]
        int32x4_t VectorS32_2 = vmovl_s16(vget_low_s16(VectorS16_1));  // [8 ... 11]
        int32x4_t VectorS32_3 = vmovl_s16(vget_high_s16(VectorS16_1)); // [12 ... 15]

        // Cast each int32x4 to float and multiply by the scale vector.
        float32x4_t VectorF32_0 = vmulq_f32(vcvtq_f32_s32(VectorS32_0), ScaleVector);
        float32x4_t VectorF32_1 = vmulq_f32(vcvtq_f32_s32(VectorS32_1), ScaleVector);
        float32x4_t VectorF32_2 = vmulq_f32(vcvtq_f32_s32(VectorS32_2), ScaleVector);
        float32x4_t VectorF32_3 = vmulq_f32(vcvtq_f32_s32(VectorS32_3), ScaleVector);

        // Store each int32x4 into the output.
        vst1q_f32(Output + 0, VectorF32_0);
        vst1q_f32(Output + 4, VectorF32_1);
        vst1q_f32(Output + 8, VectorF32_2);
        vst1q_f32(Output + 12, VectorF32_3);

        N -= 16;
        Input += 16;
        Output += 16;
    }

    // Handle leftover elements (< 16) with the scalar reference implementation.
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
    MlasDequantizeLinearS8Kernel(Input, Output, N, Scale, ZeroPoint);
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
    MlasDequantizeLinearU8Kernel(Input, Output, N, Scale, ZeroPoint);
}
#else
// Implementation that uses the scalar reference implementation.

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
