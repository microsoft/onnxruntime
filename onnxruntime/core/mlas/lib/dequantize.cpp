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
#elif defined(MLAS_NEON64_INTRINSICS)
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
        // Vector of 16 int8s: [0 ... 15]
        int8x16_t VectorS8 = vld1q_s8(Input);

        // Sign-extend into 2 vectors of 8 int16s
        int16x8_t VectorS16_0 = vmovl_s8(vget_low_s8(VectorS8));  // [0 ... 7]
        int16x8_t VectorS16_1 = vmovl_s8(vget_high_s8(VectorS8)); // [8 ... 15]

        // Subtract the zero-points now.
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
        // Vector of 16 uint8s: [0 ... 15]
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
