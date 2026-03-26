/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sconv_nchwc_kernel_neon.cpp

Abstract:

    This module implements the single precision NCHWC convolution kernels for ARM NEON.

--*/

#if defined(MLAS_USE_ARM_NEON_NCHWC)

#include <cstddef>
#include <cstdint>

#include "mlasi.h"
#include "sconv_nchwc_kernel_neon.h"

constexpr size_t BlockSize = MLAS_PLATFORM::MLAS_NEON_NCHWC_BLOCK_SIZE;

#if defined(__aarch64__) && !defined(_WIN32)

struct MLAS_NCHW_BF16_MMLA_PARAMS {
    const float* Input;
    const uint16_t* PackedFilter;
    float* Output;
    const float* Bias;
    size_t StrideWidthElements;
    size_t DilatedInputWidthElements;
    size_t OutputCount;
    size_t FilterCount;
    size_t OutputStrideElements;
    unsigned KernelFlags;
    unsigned Reserved;
};

#define MLAS_NCHW_BF16_MMLA_PARAM_OFFSET_ASSERT(Field, Offset) \
    static_assert(offsetof(MLAS_NCHW_BF16_MMLA_PARAMS, Field) == Offset, #Field " offset mismatch")

MLAS_NCHW_BF16_MMLA_PARAM_OFFSET_ASSERT(Input, 0);
MLAS_NCHW_BF16_MMLA_PARAM_OFFSET_ASSERT(PackedFilter, 8);
MLAS_NCHW_BF16_MMLA_PARAM_OFFSET_ASSERT(Output, 16);
MLAS_NCHW_BF16_MMLA_PARAM_OFFSET_ASSERT(Bias, 24);
MLAS_NCHW_BF16_MMLA_PARAM_OFFSET_ASSERT(StrideWidthElements, 32);
MLAS_NCHW_BF16_MMLA_PARAM_OFFSET_ASSERT(DilatedInputWidthElements, 40);
MLAS_NCHW_BF16_MMLA_PARAM_OFFSET_ASSERT(OutputCount, 48);
MLAS_NCHW_BF16_MMLA_PARAM_OFFSET_ASSERT(FilterCount, 56);
MLAS_NCHW_BF16_MMLA_PARAM_OFFSET_ASSERT(OutputStrideElements, 64);
MLAS_NCHW_BF16_MMLA_PARAM_OFFSET_ASSERT(KernelFlags, 72);

#undef MLAS_NCHW_BF16_MMLA_PARAM_OFFSET_ASSERT

extern "C" void
MLASCALL
MlasConvNchwBf16KernelNeonAsm(const MLAS_NCHW_BF16_MMLA_PARAMS* Params);

extern "C" void
MLASCALL
MlasConvNchwBf16PackFilterNeonAsm(
    const float* Filter,
    size_t FilterStrideElements,
    size_t FilterCount,
    uint16_t* PackedFilter
    );

#endif

#if defined(__aarch64__) && defined(__linux__)
extern "C" void
MLASCALL
MlasConvBf16OutputPostProcessNeonAsm(
    float* Output,
    size_t OutputCount,
    const float* Bias,
    unsigned KernelFlags
    );
#endif

static inline unsigned
MlasConvBf16SemanticKernelFlags(
    unsigned KernelFlags
    )
{
    return KernelFlags & ~MLAS_CONV_KERNEL_MLAS_ARM_USE_KLEIDIAI;
}

static inline const float*
MlasConvBf16BiasData(
    const float* Bias,
    unsigned KernelFlags
    )
{
    return ((KernelFlags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION) != 0) ? Bias : nullptr;
}

static inline void
MlasConvBf16PostProcessOutputs(
    float* Output,
    size_t OutputCount,
    const float* Bias,
    unsigned KernelFlags
    )
{
    const unsigned PostProcessFlags =
        KernelFlags & (MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION | MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION);

    if (OutputCount == 0 || PostProcessFlags == 0) {
        return;
    }

#if defined(__aarch64__) && defined(__linux__)
    MlasConvBf16OutputPostProcessNeonAsm(
        Output,
        OutputCount,
        MlasConvBf16BiasData(Bias, PostProcessFlags),
        PostProcessFlags
    );
    return;
#endif

    const bool BiasAddition = (PostProcessFlags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION) != 0;
    const bool ReluActivation = (PostProcessFlags & MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION) != 0;
    const float32x4_t ZeroVector = MlasBroadcastFloat32x4(0.0f);

    float32x4_t BiasVector0 = ZeroVector;
    float32x4_t BiasVector1 = ZeroVector;
    float32x4_t BiasVector2 = ZeroVector;
    float32x4_t BiasVector3 = ZeroVector;
    if (BiasAddition) {
        BiasVector0 = MlasLoadFloat32x4(Bias);
        BiasVector1 = MlasLoadFloat32x4(Bias + 4);
        BiasVector2 = MlasLoadFloat32x4(Bias + 8);
        BiasVector3 = MlasLoadFloat32x4(Bias + 12);
    }

    for (size_t OutputIndex = 0; OutputIndex < OutputCount; ++OutputIndex) {
        float* OutputRow = Output + OutputIndex * BlockSize;

        float32x4_t Accumulator0 = MlasLoadFloat32x4(OutputRow);
        float32x4_t Accumulator1 = MlasLoadFloat32x4(OutputRow + 4);
        float32x4_t Accumulator2 = MlasLoadFloat32x4(OutputRow + 8);
        float32x4_t Accumulator3 = MlasLoadFloat32x4(OutputRow + 12);

        if (BiasAddition) {
            Accumulator0 = MlasAddFloat32x4(Accumulator0, BiasVector0);
            Accumulator1 = MlasAddFloat32x4(Accumulator1, BiasVector1);
            Accumulator2 = MlasAddFloat32x4(Accumulator2, BiasVector2);
            Accumulator3 = MlasAddFloat32x4(Accumulator3, BiasVector3);
        }

        if (ReluActivation) {
            Accumulator0 = MlasMaximumFloat32x4(Accumulator0, ZeroVector);
            Accumulator1 = MlasMaximumFloat32x4(Accumulator1, ZeroVector);
            Accumulator2 = MlasMaximumFloat32x4(Accumulator2, ZeroVector);
            Accumulator3 = MlasMaximumFloat32x4(Accumulator3, ZeroVector);
        }

        MlasStoreFloat32x4(OutputRow, Accumulator0);
        MlasStoreFloat32x4(OutputRow + 4, Accumulator1);
        MlasStoreFloat32x4(OutputRow + 8, Accumulator2);
        MlasStoreFloat32x4(OutputRow + 12, Accumulator3);
    }
}

#if defined(__aarch64__) && defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)

static inline void
MlasPackPointwiseBf16InputPairNeon(
    const float* Input0,
    const float* Input1,
    uint16_t* PackedInput
    )
{
    for (size_t KGroup = 0; KGroup < BlockSize / 4; ++KGroup) {
        bfloat16x8_t InputPair = vcvtq_low_bf16_f32(vld1q_f32(Input0 + KGroup * 4));
        InputPair = vcvtq_high_bf16_f32(InputPair, vld1q_f32(Input1 + KGroup * 4));
        vst1q_u16(PackedInput + KGroup * 8, vreinterpretq_u16_bf16(InputPair));
    }
}

static inline void
MlasPackPointwiseBf16InputQuartetNeon(
    const float* Input0,
    const float* Input1,
    const float* Input2,
    const float* Input3,
    uint16_t* PackedInput
    )
{
    MlasPackPointwiseBf16InputPairNeon(Input0, Input1, PackedInput);
    MlasPackPointwiseBf16InputPairNeon(Input2, Input3, PackedInput + BlockSize * 2);
}

static inline void
MlasPackPointwiseBf16InputPairsNeon(
    const float* Input,
    uint16_t* PackedInput,
    size_t StrideWidthElements,
    size_t InputStrideElements,
    size_t InputChannels,
    size_t OutputPairCount
    )
{
    constexpr size_t PointwisePackedInputPairSizeBf16 = BlockSize * 2;

    for (size_t OutputPairIndex = 0; OutputPairIndex < OutputPairCount; ++OutputPairIndex) {
        const float* Input0 = Input + OutputPairIndex * 2 * StrideWidthElements;
        const float* Input1 = Input0 + StrideWidthElements;

        for (size_t InputChannelIndex = 0; InputChannelIndex < InputChannels; ++InputChannelIndex) {
            MlasPackPointwiseBf16InputPairNeon(
                Input0 + InputChannelIndex * InputStrideElements,
                Input1 + InputChannelIndex * InputStrideElements,
                PackedInput
            );
            PackedInput += PointwisePackedInputPairSizeBf16;
        }
    }
}

static inline void
MlasPackPointwiseBf16InputQuartetsNeon(
    const float* Input,
    uint16_t* PackedInput,
    size_t StrideWidthElements,
    size_t InputStrideElements,
    size_t InputChannels,
    size_t OutputQuartetCount
    )
{
    constexpr size_t PointwisePackedInputQuartetSizeBf16 = BlockSize * 4;

    for (size_t OutputQuartetIndex = 0; OutputQuartetIndex < OutputQuartetCount; ++OutputQuartetIndex) {
        const float* Input0 = Input + OutputQuartetIndex * 4 * StrideWidthElements;
        const float* Input1 = Input0 + StrideWidthElements;
        const float* Input2 = Input1 + StrideWidthElements;
        const float* Input3 = Input2 + StrideWidthElements;

        for (size_t InputChannelIndex = 0; InputChannelIndex < InputChannels; ++InputChannelIndex) {
            MlasPackPointwiseBf16InputQuartetNeon(
                Input0 + InputChannelIndex * InputStrideElements,
                Input1 + InputChannelIndex * InputStrideElements,
                Input2 + InputChannelIndex * InputStrideElements,
                Input3 + InputChannelIndex * InputStrideElements,
                PackedInput
            );
            PackedInput += PointwisePackedInputQuartetSizeBf16;
        }
    }
}

static inline void
MlasMergePointwiseBf16OutputsNeon(
    const float* PartialOutput,
    float* Output,
    size_t OutputCount,
    const float* Bias,
    unsigned KernelFlags
    )
{
    const float32x4_t ZeroVector = MlasBroadcastFloat32x4(0.0f);
    const float32x4_t BiasMask = vreinterpretq_f32_s32(MlasBroadcastInt32x4(-(KernelFlags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION)));
    const float32x4_t ReluMask = vreinterpretq_f32_s32(MlasBroadcastInt32x4(-(KernelFlags & MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION)));

    float32x4_t BiasVector0 = ZeroVector;
    float32x4_t BiasVector1 = ZeroVector;
    float32x4_t BiasVector2 = ZeroVector;
    float32x4_t BiasVector3 = ZeroVector;

    if ((KernelFlags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION) != 0 && Bias != nullptr) {
        BiasVector0 = MlasLoadFloat32x4(Bias);
        BiasVector1 = MlasLoadFloat32x4(Bias + 4);
        BiasVector2 = MlasLoadFloat32x4(Bias + 8);
        BiasVector3 = MlasLoadFloat32x4(Bias + 12);
    }

    for (size_t OutputIndex = 0; OutputIndex < OutputCount; ++OutputIndex) {
        const float* PartialRow = PartialOutput + OutputIndex * BlockSize;
        float* OutputRow = Output + OutputIndex * BlockSize;

        float32x4_t Accumulator0 = MlasAddFloat32x4(MlasLoadFloat32x4(PartialRow), MlasLoadFloat32x4(OutputRow));
        float32x4_t Accumulator1 = MlasAddFloat32x4(MlasLoadFloat32x4(PartialRow + 4), MlasLoadFloat32x4(OutputRow + 4));
        float32x4_t Accumulator2 = MlasAddFloat32x4(MlasLoadFloat32x4(PartialRow + 8), MlasLoadFloat32x4(OutputRow + 8));
        float32x4_t Accumulator3 = MlasAddFloat32x4(MlasLoadFloat32x4(PartialRow + 12), MlasLoadFloat32x4(OutputRow + 12));

        Accumulator0 = MlasAddFloat32x4(Accumulator0, MlasAndFloat32x4(BiasVector0, BiasMask));
        Accumulator1 = MlasAddFloat32x4(Accumulator1, MlasAndFloat32x4(BiasVector1, BiasMask));
        Accumulator2 = MlasAddFloat32x4(Accumulator2, MlasAndFloat32x4(BiasVector2, BiasMask));
        Accumulator3 = MlasAddFloat32x4(Accumulator3, MlasAndFloat32x4(BiasVector3, BiasMask));

        float32x4_t Relu0 = MlasMaximumFloat32x4(Accumulator0, ZeroVector);
        float32x4_t Relu1 = MlasMaximumFloat32x4(Accumulator1, ZeroVector);
        float32x4_t Relu2 = MlasMaximumFloat32x4(Accumulator2, ZeroVector);
        float32x4_t Relu3 = MlasMaximumFloat32x4(Accumulator3, ZeroVector);

        Accumulator0 = MlasBlendFloat32x4(Accumulator0, Relu0, ReluMask);
        Accumulator1 = MlasBlendFloat32x4(Accumulator1, Relu1, ReluMask);
        Accumulator2 = MlasBlendFloat32x4(Accumulator2, Relu2, ReluMask);
        Accumulator3 = MlasBlendFloat32x4(Accumulator3, Relu3, ReluMask);

        MlasStoreFloat32x4(OutputRow, Accumulator0);
        MlasStoreFloat32x4(OutputRow + 4, Accumulator1);
        MlasStoreFloat32x4(OutputRow + 8, Accumulator2);
        MlasStoreFloat32x4(OutputRow + 12, Accumulator3);
    }
}

#endif

// Common implementation for NCHW and NCHWC convolution kernels
template <bool IsNchwcFormat>
void
    MLASCALL
    MlasConvFloatKernelNeonImpl(
        const float* Input,
        const float* Filter,
        float* Output,
        size_t StrideWidth,
        size_t DilationWidth,
        size_t FilterCount,
        size_t InputStride,
        size_t FilterStride,
        size_t OutputStride,
        size_t KernelHeight,
        size_t KernelWidth,
        const float* InputBase,
        size_t InputWidth,
        size_t DilatedInputWidth,
        size_t OutputCountLeftPad,
        size_t OutputCount,
        size_t OutputCountRightPad,
        const float* Bias,
        unsigned KernelFlags
    )
{
    const float32x4_t ZeroVector = MlasBroadcastFloat32x4(0.0f);
    const float32x4_t AccumulateMask = vreinterpretq_f32_s32(MlasBroadcastInt32x4(-(KernelFlags & MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT)));
    const bool BiasAddition = (KernelFlags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION) != 0;
    const float32x4_t BiasMask = vreinterpretq_f32_s32(MlasBroadcastInt32x4(BiasAddition ? -1 : 0));
    const float32x4_t ReluMask = vreinterpretq_f32_s32(MlasBroadcastInt32x4(-(KernelFlags & MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION)));

    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t FilterStrideElements = FilterStride / sizeof(float);
    const size_t OutputStrideElements = OutputStride / sizeof(float);
    const size_t InputWidthElements = InputWidth / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);

    MLAS_UNREFERENCED_PARAMETER(InputStride);

    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;

    for (size_t output_idx = 0; output_idx < TotalOutputCount; output_idx++) {

        for (size_t filterSetBlock = 0; filterSetBlock < FilterCount; filterSetBlock++) {
            const float* filter = Filter + filterSetBlock * FilterStrideElements;
            float* output = Output + filterSetBlock * OutputStrideElements;

            float32x4_t OldOutput0 = MlasLoadFloat32x4(&output[output_idx * BlockSize]);
            float32x4_t OldOutput1 = MlasLoadFloat32x4(&output[output_idx * BlockSize + 4]);
            float32x4_t OldOutput2 = MlasLoadFloat32x4(&output[output_idx * BlockSize + 8]);
            float32x4_t OldOutput3 = MlasLoadFloat32x4(&output[output_idx * BlockSize + 12]);

            float32x4_t Accumulator0 = MlasBroadcastFloat32x4(0.0f);
            float32x4_t Accumulator1 = MlasBroadcastFloat32x4(0.0f);
            float32x4_t Accumulator2 = MlasBroadcastFloat32x4(0.0f);
            float32x4_t Accumulator3 = MlasBroadcastFloat32x4(0.0f);

            float32x4_t BiasVector0 = ZeroVector;
            float32x4_t BiasVector1 = ZeroVector;
            float32x4_t BiasVector2 = ZeroVector;
            float32x4_t BiasVector3 = ZeroVector;
            if (BiasAddition) {
                BiasVector0 = MlasLoadFloat32x4(&Bias[filterSetBlock * BlockSize]);
                BiasVector1 = MlasLoadFloat32x4(&Bias[filterSetBlock * BlockSize + 4]);
                BiasVector2 = MlasLoadFloat32x4(&Bias[filterSetBlock * BlockSize + 8]);
                BiasVector3 = MlasLoadFloat32x4(&Bias[filterSetBlock * BlockSize + 12]);
            }

            for (size_t kernel_pos = 0; kernel_pos < KernelHeight * KernelWidth; kernel_pos++) {
                size_t kh = kernel_pos / KernelWidth;
                size_t kw = kernel_pos % KernelWidth;

                const float* input_base = Input + output_idx * StrideWidthElements +
                                          kh * DilatedInputWidthElements + kw * DilationWidthElements;

                if constexpr (IsNchwcFormat) {
                    for (size_t filterBlock = 0; filterBlock < BlockSize; filterBlock++) {
                        const float* input_element = input_base + filterBlock;
                        const float* input_row_start = InputBase + kh * DilatedInputWidthElements;
                        const float* input_row_end = input_row_start + InputWidthElements;

                        float input_value;
                        if (input_element >= input_row_start && input_element < input_row_end) {
                            input_value = *input_element;
                        } else {
                            input_value = 0.0f;
                        }

                        const float32x4_t InputVector = MlasBroadcastFloat32x4(input_value);

                        size_t kernel_base_pos = kh * (KernelWidth * BlockSize * BlockSize) +
                                                 kw * (BlockSize * BlockSize) +
                                                 filterBlock * BlockSize;

                        const float32x4_t FilterVector0 = MlasLoadFloat32x4(&filter[kernel_base_pos]);
                        const float32x4_t FilterVector1 = MlasLoadFloat32x4(&filter[kernel_base_pos + 4]);
                        const float32x4_t FilterVector2 = MlasLoadFloat32x4(&filter[kernel_base_pos + 8]);
                        const float32x4_t FilterVector3 = MlasLoadFloat32x4(&filter[kernel_base_pos + 12]);

                        Accumulator0 = MlasMultiplyAddFloat32x4(InputVector, FilterVector0, Accumulator0);
                        Accumulator1 = MlasMultiplyAddFloat32x4(InputVector, FilterVector1, Accumulator1);
                        Accumulator2 = MlasMultiplyAddFloat32x4(InputVector, FilterVector2, Accumulator2);
                        Accumulator3 = MlasMultiplyAddFloat32x4(InputVector, FilterVector3, Accumulator3);
                    }
                } else {
                    const float* input_row_start = InputBase + kh * DilatedInputWidthElements;
                    const float* input_row_end = input_row_start + InputWidthElements;

                    float input_value;
                    if (input_base >= input_row_start && input_base < input_row_end) {
                        input_value = *input_base;
                    } else {
                        input_value = 0.0f;
                    }

                    const float32x4_t InputVector = MlasBroadcastFloat32x4(input_value);

                    size_t kernel_base_pos = kh * KernelWidth + kw;

                    const float32x4_t FilterVector0 = MlasLoadFloat32x4(&filter[kernel_base_pos * BlockSize]);
                    const float32x4_t FilterVector1 = MlasLoadFloat32x4(&filter[kernel_base_pos * BlockSize + 4]);
                    const float32x4_t FilterVector2 = MlasLoadFloat32x4(&filter[kernel_base_pos * BlockSize + 8]);
                    const float32x4_t FilterVector3 = MlasLoadFloat32x4(&filter[kernel_base_pos * BlockSize + 12]);

                    Accumulator0 = MlasMultiplyAddFloat32x4(InputVector, FilterVector0, Accumulator0);
                    Accumulator1 = MlasMultiplyAddFloat32x4(InputVector, FilterVector1, Accumulator1);
                    Accumulator2 = MlasMultiplyAddFloat32x4(InputVector, FilterVector2, Accumulator2);
                    Accumulator3 = MlasMultiplyAddFloat32x4(InputVector, FilterVector3, Accumulator3);
                }
            }

            Accumulator0 = MlasAddFloat32x4(Accumulator0, MlasAndFloat32x4(OldOutput0, AccumulateMask));
            Accumulator1 = MlasAddFloat32x4(Accumulator1, MlasAndFloat32x4(OldOutput1, AccumulateMask));
            Accumulator2 = MlasAddFloat32x4(Accumulator2, MlasAndFloat32x4(OldOutput2, AccumulateMask));
            Accumulator3 = MlasAddFloat32x4(Accumulator3, MlasAndFloat32x4(OldOutput3, AccumulateMask));

            Accumulator0 = MlasAddFloat32x4(Accumulator0, MlasAndFloat32x4(BiasVector0, BiasMask));
            Accumulator1 = MlasAddFloat32x4(Accumulator1, MlasAndFloat32x4(BiasVector1, BiasMask));
            Accumulator2 = MlasAddFloat32x4(Accumulator2, MlasAndFloat32x4(BiasVector2, BiasMask));
            Accumulator3 = MlasAddFloat32x4(Accumulator3, MlasAndFloat32x4(BiasVector3, BiasMask));

            float32x4_t Relu0 = MlasMaximumFloat32x4(Accumulator0, ZeroVector);
            float32x4_t Relu1 = MlasMaximumFloat32x4(Accumulator1, ZeroVector);
            float32x4_t Relu2 = MlasMaximumFloat32x4(Accumulator2, ZeroVector);
            float32x4_t Relu3 = MlasMaximumFloat32x4(Accumulator3, ZeroVector);

            Accumulator0 = MlasBlendFloat32x4(Accumulator0, Relu0, ReluMask);
            Accumulator1 = MlasBlendFloat32x4(Accumulator1, Relu1, ReluMask);
            Accumulator2 = MlasBlendFloat32x4(Accumulator2, Relu2, ReluMask);
            Accumulator3 = MlasBlendFloat32x4(Accumulator3, Relu3, ReluMask);

            MlasStoreFloat32x4(&output[output_idx * BlockSize], Accumulator0);
            MlasStoreFloat32x4(&output[output_idx * BlockSize + 4], Accumulator1);
            MlasStoreFloat32x4(&output[output_idx * BlockSize + 8], Accumulator2);
            MlasStoreFloat32x4(&output[output_idx * BlockSize + 12], Accumulator3);
        }
    }
}


//
// Implementation of MlasConvNchwBf16KernelNeon
//

#if defined(__aarch64__) && defined(__linux__)
void
    MLASCALL
    MlasConvNchwBf16KernelNeon(
        const float* Input,
        const float* Filter,
        float* Output,
        size_t StrideWidth,
        size_t DilationWidth,
        size_t FilterCount,
        size_t InputStride,
        size_t FilterStride,
        size_t OutputStride,
        size_t KernelHeight,
        size_t KernelWidth,
        const float* InputBase,
        size_t InputWidth,
        size_t DilatedInputWidth,
        size_t OutputCountLeftPad,
        size_t OutputCount,
        size_t OutputCountRightPad,
        const float* Bias,
        unsigned KernelFlags
    )
{
#if defined(__aarch64__) && !defined(_WIN32)
    constexpr size_t Bf16MmlaKernelHeight = 3;
    constexpr size_t Bf16MmlaKernelWidth = 3;
    constexpr size_t Bf16MmlaMaxFilterCount = 4;
    constexpr size_t Bf16MmlaPackedFilterStrideBf16 = 192;

    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t FilterStrideElements = FilterStride / sizeof(float);
    const size_t OutputStrideElements = OutputStride / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);

    const bool UseBf16MmlaKernel =
        KernelHeight == Bf16MmlaKernelHeight &&
        KernelWidth == Bf16MmlaKernelWidth &&
        DilationWidthElements == 1 &&
        FilterCount > 0 && FilterCount <= Bf16MmlaMaxFilterCount &&
        OutputCount >= 2;

    if (UseBf16MmlaKernel) {
        auto ConvFallbackSegment = [&](size_t outputOffset, size_t segmentCount) {
            if (segmentCount == 0) {
                return;
            }

            const float* segmentInput = Input + outputOffset * StrideWidthElements;
            float* segmentOutput = Output + outputOffset * BlockSize;

            MlasConvFloatKernelNeonImpl<false>(
                segmentInput,
                Filter,
                segmentOutput,
                StrideWidth,
                DilationWidth,
                FilterCount,
                InputStride,
                FilterStride,
                OutputStride,
                KernelHeight,
                KernelWidth,
                InputBase,
                InputWidth,
                DilatedInputWidth,
                0,
                segmentCount,
                0,
                Bias,
                KernelFlags);
        };

        size_t outputOffset = 0;

        // Left padding outputs still require bounds checks.
        ConvFallbackSegment(outputOffset, OutputCountLeftPad);
        outputOffset += OutputCountLeftPad;

        const size_t OutputCountEven = OutputCount & ~size_t{1};

        if (OutputCountEven != 0) {
            alignas(16) uint16_t PackedFilter[Bf16MmlaMaxFilterCount * Bf16MmlaPackedFilterStrideBf16];
            MlasConvNchwBf16PackFilterNeonAsm(Filter, FilterStrideElements, FilterCount, PackedFilter);

            const float* midInput = Input + outputOffset * StrideWidthElements;
            float* midOutput = Output + outputOffset * BlockSize;

            MLAS_NCHW_BF16_MMLA_PARAMS params;
            params.Input = midInput;
            params.PackedFilter = PackedFilter;
            params.Output = midOutput;
            params.Bias = Bias;
            params.StrideWidthElements = StrideWidthElements;
            params.DilatedInputWidthElements = DilatedInputWidthElements;
            params.OutputCount = OutputCountEven;
            params.FilterCount = FilterCount;
            params.OutputStrideElements = OutputStrideElements;
            params.KernelFlags = KernelFlags;
            params.Reserved = 0;

            MlasConvNchwBf16KernelNeonAsm(&params);

            outputOffset += OutputCountEven;
        }

        // Handle a remaining single mid output, if present.
        ConvFallbackSegment(outputOffset, OutputCount - OutputCountEven);
        outputOffset += OutputCount - OutputCountEven;

        // Right padding outputs require bounds checks.
        ConvFallbackSegment(outputOffset, OutputCountRightPad);
        return;
    }
#endif

    MlasConvFloatKernelNeonImpl<false>(
        Input,
        Filter,
        Output,
        StrideWidth,
        DilationWidth,
        FilterCount,
        InputStride,
        FilterStride,
        OutputStride,
        KernelHeight,
        KernelWidth,
        InputBase,
        InputWidth,
        DilatedInputWidth,
        OutputCountLeftPad,
        OutputCount,
        OutputCountRightPad,
        Bias,
        KernelFlags
    );
}
#endif

//
// Implementation of MlasConvNchwFloatKernelNeon
//

void
    MLASCALL
    MlasConvNchwFloatKernelNeon(
        const float* Input,
        const float* Filter,
        float* Output,
        size_t StrideWidth,
        size_t DilationWidth,
        size_t FilterCount,
        size_t InputStride,
        size_t FilterStride,
        size_t OutputStride,
        size_t KernelHeight,
        size_t KernelWidth,
        const float* InputBase,
        size_t InputWidth,
        size_t DilatedInputWidth,
        size_t OutputCountLeftPad,
        size_t OutputCount,
        size_t OutputCountRightPad,
        const float* Bias,
        unsigned KernelFlags
    )
{
    MlasConvFloatKernelNeonImpl<false>(
        Input,
        Filter,
        Output,
        StrideWidth,
        DilationWidth,
        FilterCount,
        InputStride,
        FilterStride,
        OutputStride,
        KernelHeight,
        KernelWidth,
        InputBase,
        InputWidth,
        DilatedInputWidth,
        OutputCountLeftPad,
        OutputCount,
        OutputCountRightPad,
        Bias,
        KernelFlags
    );
}


//
// Implementation of MlasConvNchwcFloatKernelNeon
//

void
    MLASCALL
    MlasConvNchwcFloatKernelNeon(
        const float* Input,
        const float* Filter,
        float* Output,
        size_t StrideWidth,
        size_t DilationWidth,
        size_t FilterCount,
        size_t InputStride,
        size_t FilterStride,
        size_t OutputStride,
        size_t KernelHeight,
        size_t KernelWidth,
        const float* InputBase,
        size_t InputWidth,
        size_t DilatedInputWidth,
        size_t OutputCountLeftPad,
        size_t OutputCount,
        size_t OutputCountRightPad,
        const float* Bias,
        unsigned KernelFlags
    )
{
#if defined(MLAS_TARGET_ARM64) && defined(MLAS_USE_ARM_NEON_NCHWC) && !defined(_WIN32)
    MlasConvNchwcFloatKernelNeonAsm(
        Input,
        Filter,
        Output,
        StrideWidth,
        DilationWidth,
        FilterCount,
        InputStride,
        FilterStride,
        OutputStride,
        KernelHeight,
        KernelWidth,
        InputBase,
        InputWidth,
        DilatedInputWidth,
        OutputCountLeftPad,
        OutputCount,
        OutputCountRightPad,
        Bias,
        KernelFlags
    );
#else
    MlasConvFloatKernelNeonImpl<true>(
        Input,
        Filter,
        Output,
        StrideWidth,
        DilationWidth,
        FilterCount,
        InputStride,
        FilterStride,
        OutputStride,
        KernelHeight,
        KernelWidth,
        InputBase,
        InputWidth,
        DilatedInputWidth,
        OutputCountLeftPad,
        OutputCount,
        OutputCountRightPad,
        Bias,
        KernelFlags
    );
#endif
}

//
// Implementation of MlasConvDepthwiseFloatKernelNeon
//
// This kernel performs depthwise separable convolution where each input channel
// is convolved with its own filter. 
//

#if defined(__aarch64__) && defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
extern "C" void
    MLASCALL
    MlasConvDepthwiseBf16KernelNeon3x3MidAsm(
        const float* Input,
        const float* Filter,
        float* Output,
        size_t OutputCount,
        size_t StrideWidthBytes,
        size_t DilationWidthBytes,
        size_t DilatedInputWidthBytes
    );

extern "C" void
    MLASCALL
    MlasConvDepthwiseBf16KernelNeon3x3MidStride1Asm(
        const float* Input,
        const float* Filter,
        float* Output,
        size_t OutputCount,
        size_t StrideWidthBytes,
        size_t DilationWidthBytes,
        size_t DilatedInputWidthBytes
    );

extern "C" void
    MLASCALL
    MlasConvDepthwiseBf16KernelNeon3x3MidStride2Asm(
        const float* Input,
        const float* Filter,
        float* Output,
        size_t OutputCount,
        size_t StrideWidthBytes,
        size_t DilationWidthBytes,
        size_t DilatedInputWidthBytes
    );

extern "C" void
    MLASCALL
    MlasConvDepthwiseBf16KernelNeon3x3DispatchAsm(
        const float* Input,
        const float* Filter,
        float* Output,
        size_t OutputCount,
        size_t StrideWidthBytes,
        size_t DilationWidthBytes,
        size_t DilatedInputWidthBytes,
        const float* Bias,
        unsigned KernelFlags
    );
#endif

static void
MlasConvDepthwiseFloatKernelNeonImpl(
        const float* Input,
        const float* Filter,
        float* Output,
        size_t StrideWidth,
        size_t DilationWidth,
        size_t InputStride,
        size_t KernelHeight,
        size_t KernelWidth,
        const float* InputBase,
        size_t InputWidth,
        size_t DilatedInputWidth,
        size_t OutputCountLeftPad,
        size_t OutputCount,
        size_t OutputCountRightPad,
        const float* Bias,
        unsigned KernelFlags,
        bool UseBf16FastPath
    )
{
    const float32x4_t ZeroVector = MlasBroadcastFloat32x4(0.0f);
    const float32x4_t AccumulateMask = vreinterpretq_f32_s32(MlasBroadcastInt32x4(-(KernelFlags & MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT)));
    const float32x4_t BiasMask = vreinterpretq_f32_s32(MlasBroadcastInt32x4(-(KernelFlags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION)));
    const float32x4_t ReluMask = vreinterpretq_f32_s32(MlasBroadcastInt32x4(-(KernelFlags & MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION)));

    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);

    MLAS_UNREFERENCED_PARAMETER(InputStride);

    const size_t InputWidthElements = InputWidth / sizeof(float);

    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;

    const size_t KernelSize = KernelHeight * KernelWidth;

    auto ComputeDepthwiseOutput = [&](size_t output_idx) {

        float32x4_t OldOutput0 = MlasLoadFloat32x4(&Output[output_idx * BlockSize]);
        float32x4_t OldOutput1 = MlasLoadFloat32x4(&Output[output_idx * BlockSize + 4]);
        float32x4_t OldOutput2 = MlasLoadFloat32x4(&Output[output_idx * BlockSize + 8]);
        float32x4_t OldOutput3 = MlasLoadFloat32x4(&Output[output_idx * BlockSize + 12]);

        float32x4_t Accumulator0 = MlasBroadcastFloat32x4(0.0f);
        float32x4_t Accumulator1 = MlasBroadcastFloat32x4(0.0f);
        float32x4_t Accumulator2 = MlasBroadcastFloat32x4(0.0f);
        float32x4_t Accumulator3 = MlasBroadcastFloat32x4(0.0f);

        const bool BiasAddition = (KernelFlags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION) != 0;
        float32x4_t BiasVector0 = ZeroVector;
        float32x4_t BiasVector1 = ZeroVector;
        float32x4_t BiasVector2 = ZeroVector;
        float32x4_t BiasVector3 = ZeroVector;
        if (BiasAddition) {
            BiasVector0 = MlasLoadFloat32x4(Bias);
            BiasVector1 = MlasLoadFloat32x4(Bias + 4);
            BiasVector2 = MlasLoadFloat32x4(Bias + 8);
            BiasVector3 = MlasLoadFloat32x4(Bias + 12);
        }

        for (size_t kernel_pos = 0; kernel_pos < KernelSize; kernel_pos++) {
            size_t kh = kernel_pos / KernelWidth;
            size_t kw = kernel_pos % KernelWidth;

            const float* input_base = Input + output_idx * StrideWidthElements +
                                      kh * DilatedInputWidthElements + kw * DilationWidthElements;

            float32x4_t InputVector0 = LoadInputVectorWithBounds(input_base + 0, InputBase + kh * DilatedInputWidthElements, InputBase + kh * DilatedInputWidthElements + InputWidthElements);
            float32x4_t InputVector1 = LoadInputVectorWithBounds(input_base + 4, InputBase + kh * DilatedInputWidthElements, InputBase + kh * DilatedInputWidthElements + InputWidthElements);
            float32x4_t InputVector2 = LoadInputVectorWithBounds(input_base + 8, InputBase + kh * DilatedInputWidthElements, InputBase + kh * DilatedInputWidthElements + InputWidthElements);
            float32x4_t InputVector3 = LoadInputVectorWithBounds(input_base + 12, InputBase + kh * DilatedInputWidthElements, InputBase + kh * DilatedInputWidthElements + InputWidthElements);

            const float32x4_t FilterVector0 = MlasLoadFloat32x4(&Filter[kernel_pos * BlockSize]);
            const float32x4_t FilterVector1 = MlasLoadFloat32x4(&Filter[kernel_pos * BlockSize + 4]);
            const float32x4_t FilterVector2 = MlasLoadFloat32x4(&Filter[kernel_pos * BlockSize + 8]);
            const float32x4_t FilterVector3 = MlasLoadFloat32x4(&Filter[kernel_pos * BlockSize + 12]);

            Accumulator0 = MlasMultiplyAddFloat32x4(InputVector0, FilterVector0, Accumulator0);
            Accumulator1 = MlasMultiplyAddFloat32x4(InputVector1, FilterVector1, Accumulator1);
            Accumulator2 = MlasMultiplyAddFloat32x4(InputVector2, FilterVector2, Accumulator2);
            Accumulator3 = MlasMultiplyAddFloat32x4(InputVector3, FilterVector3, Accumulator3);
        }

        Accumulator0 = MlasAddFloat32x4(Accumulator0, MlasAndFloat32x4(OldOutput0, AccumulateMask));
        Accumulator1 = MlasAddFloat32x4(Accumulator1, MlasAndFloat32x4(OldOutput1, AccumulateMask));
        Accumulator2 = MlasAddFloat32x4(Accumulator2, MlasAndFloat32x4(OldOutput2, AccumulateMask));
        Accumulator3 = MlasAddFloat32x4(Accumulator3, MlasAndFloat32x4(OldOutput3, AccumulateMask));

        Accumulator0 = MlasAddFloat32x4(Accumulator0, MlasAndFloat32x4(BiasVector0, BiasMask));
        Accumulator1 = MlasAddFloat32x4(Accumulator1, MlasAndFloat32x4(BiasVector1, BiasMask));
        Accumulator2 = MlasAddFloat32x4(Accumulator2, MlasAndFloat32x4(BiasVector2, BiasMask));
        Accumulator3 = MlasAddFloat32x4(Accumulator3, MlasAndFloat32x4(BiasVector3, BiasMask));

        float32x4_t Relu0 = MlasMaximumFloat32x4(Accumulator0, ZeroVector);
        float32x4_t Relu1 = MlasMaximumFloat32x4(Accumulator1, ZeroVector);
        float32x4_t Relu2 = MlasMaximumFloat32x4(Accumulator2, ZeroVector);
        float32x4_t Relu3 = MlasMaximumFloat32x4(Accumulator3, ZeroVector);

        Accumulator0 = MlasBlendFloat32x4(Accumulator0, Relu0, ReluMask);
        Accumulator1 = MlasBlendFloat32x4(Accumulator1, Relu1, ReluMask);
        Accumulator2 = MlasBlendFloat32x4(Accumulator2, Relu2, ReluMask);
        Accumulator3 = MlasBlendFloat32x4(Accumulator3, Relu3, ReluMask);

        MlasStoreFloat32x4(&Output[output_idx * BlockSize], Accumulator0);
        MlasStoreFloat32x4(&Output[output_idx * BlockSize + 4], Accumulator1);
        MlasStoreFloat32x4(&Output[output_idx * BlockSize + 8], Accumulator2);
        MlasStoreFloat32x4(&Output[output_idx * BlockSize + 12], Accumulator3);

    };

#if defined(__aarch64__) && defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    if (UseBf16FastPath) {
        const unsigned SemanticKernelFlags = KernelFlags & ~MLAS_CONV_KERNEL_MLAS_ARM_USE_KLEIDIAI;
        const bool AccumulateOutput = (SemanticKernelFlags & MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT) != 0;

        //
        // Fast path: depthwise 3x3 without padding on the processed range.
        //
        // This is only enabled when the kernel flags are clear so that the 
        // assembly micro-kernel can focus on the raw convolution accumulation. 
        // Boundary outputs and other flag combinations fall back to the 
        // generic implementation.
        //
        constexpr bool BlockSizeSupported = (BlockSize == 16);
        const bool CanUseAsmHotPath =
            BlockSizeSupported &&
            !AccumulateOutput &&
            KernelHeight == 3 &&
            KernelWidth == 3 &&
            DilationWidthElements == BlockSize &&
            DilatedInputWidth > 2 * DilationWidth &&
            OutputCount > 0;

        if (CanUseAsmHotPath) {
            // Left padded outputs.
            for (size_t output_idx = 0; output_idx < OutputCountLeftPad; output_idx++) {
                ComputeDepthwiseOutput(output_idx);
            }

            // Interior outputs without padding.
            const size_t MidOutputStart = OutputCountLeftPad;
            const float* MidInput = Input + MidOutputStart * StrideWidthElements;
            float* MidOutput = Output + MidOutputStart * BlockSize;
            MlasConvDepthwiseBf16KernelNeon3x3DispatchAsm(
                MidInput,
                Filter,
                MidOutput,
                OutputCount,
                StrideWidth,
                DilationWidth,
                DilatedInputWidth,
                MlasConvBf16BiasData(Bias, SemanticKernelFlags),
                SemanticKernelFlags
            );

            // Right padded outputs.
            const size_t RightOutputStart = MidOutputStart + OutputCount;
            for (size_t output_idx = RightOutputStart; output_idx < TotalOutputCount; output_idx++) {
                ComputeDepthwiseOutput(output_idx);
            }

            return;
        }
    }
#else
    MLAS_UNREFERENCED_PARAMETER(UseBf16FastPath);
#endif

    for (size_t output_idx = 0; output_idx < TotalOutputCount; output_idx++) {
        ComputeDepthwiseOutput(output_idx);
    }
}

void
    MLASCALL
    MlasConvDepthwiseFloatKernelNeon(
        const float* Input,
        const float* Filter,
        float* Output,
        size_t StrideWidth,
        size_t DilationWidth,
        size_t InputStride,
        size_t KernelHeight,
        size_t KernelWidth,
        const float* InputBase,
        size_t InputWidth,
        size_t DilatedInputWidth,
        size_t OutputCountLeftPad,
        size_t OutputCount,
        size_t OutputCountRightPad,
        const float* Bias,
        unsigned KernelFlags
    )
{
    MlasConvDepthwiseFloatKernelNeonImpl(
        Input,
        Filter,
        Output,
        StrideWidth,
        DilationWidth,
        InputStride,
        KernelHeight,
        KernelWidth,
        InputBase,
        InputWidth,
        DilatedInputWidth,
        OutputCountLeftPad,
        OutputCount,
        OutputCountRightPad,
        Bias,
        KernelFlags,
        false);
}

#if defined(__aarch64__) && defined(__linux__)
void
    MLASCALL
    MlasConvDepthwiseBf16KernelNeon(
        const float* Input,
        const float* Filter,
        float* Output,
        size_t StrideWidth,
        size_t DilationWidth,
        size_t InputStride,
        size_t KernelHeight,
        size_t KernelWidth,
        const float* InputBase,
        size_t InputWidth,
        size_t DilatedInputWidth,
        size_t OutputCountLeftPad,
        size_t OutputCount,
        size_t OutputCountRightPad,
        const float* Bias,
        unsigned KernelFlags
    )
{
    MlasConvDepthwiseFloatKernelNeonImpl(
        Input,
        Filter,
        Output,
        StrideWidth,
        DilationWidth,
        InputStride,
        KernelHeight,
        KernelWidth,
        InputBase,
        InputWidth,
        DilatedInputWidth,
        OutputCountLeftPad,
        OutputCount,
        OutputCountRightPad,
        Bias,
        KernelFlags,
        true);
}
#endif

//
// Pointwise convolution helpers. The generic float kernel stays on the
// established batched-GEMM path, and the BF16-specific helper below routes
// fast-math workloads to the BFMMLA assembly kernel.
//

namespace {

#if defined(__aarch64__) && defined(__ARM_FEATURE_BF16)

extern "C" void
MLASCALL
MlasConvPointwiseBf16PackFilterNeonAsm(
    const float* Filter,
    size_t InputChannels,
    uint16_t* PackedFilter
    );

extern "C" void
MLASCALL
MlasConvPointwiseBf16KernelNeonAsm(
    const float* Input,
    const uint16_t* PackedFilter,
    float* Output,
    size_t StrideWidthBytes,
    size_t InputStrideBytes,
    size_t InputChannels,
    size_t OutputCountEven
    );

extern "C" void
MLASCALL
MlasConvPointwiseBf16KernelNeonSingleOutputAsm(
    const float* Input,
    const uint16_t* PackedFilter,
    float* Output,
    size_t StrideWidthBytes,
    size_t InputStrideBytes,
    size_t InputChannels,
    size_t OutputCount
    );

extern "C" void
MLASCALL
MlasConvPointwiseBf16PackedInputKernelNeon4xAsm(
    const uint16_t* PackedInput,
    const uint16_t* PackedFilter,
    float* Output,
    size_t OutputQuartetCount,
    size_t InputChannels
    );

extern "C" void
MLASCALL
MlasConvPointwiseBf16PackedInputKernelNeon2xAsm(
    const uint16_t* PackedInput,
    const uint16_t* PackedFilter,
    float* Output,
    size_t OutputPairCount,
    size_t InputChannels
    );

static void
MlasConvPointwiseFloatKernelNeonBf16Mmla(
    const float* Input,
    const float* Filter,
    float* Output,
    size_t StrideWidth,
    size_t InputChannels,
    size_t FilterCount,
    size_t InputStride,
    size_t FilterStride,
    size_t OutputStride,
    size_t OutputCount
    )
{
    constexpr size_t PointwiseInputChannelsMax = 8;
    constexpr size_t PointwiseFilterCountMax = 4;
    constexpr size_t PointwisePackedFilterStrideBf16 = 256;
    constexpr size_t PointwisePackedFilterSizeBf16 = PointwiseInputChannelsMax * PointwisePackedFilterStrideBf16;
    constexpr size_t PointwisePackedInputQuartetSizeBf16 = BlockSize * 4;
    constexpr size_t PointwisePackedInputPairSizeBf16 = BlockSize * 2;
    constexpr size_t PointwiseOutputQuartetBatchMax = 8;
    constexpr size_t PointwiseOutputPairBatchMax = 16;

    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t InputStrideElements = InputStride / sizeof(float);
    const size_t FilterStrideElements = FilterStride / sizeof(float);
    const size_t OutputStrideElements = OutputStride / sizeof(float);

    const size_t OutputCountQuartet = OutputCount & ~size_t{3};
    const size_t OutputCountEven = OutputCount & ~size_t{1};
    const size_t OutputCountPairTail = OutputCountEven - OutputCountQuartet;
    const size_t OutputCountRemainder = OutputCount - OutputCountEven;
    const float* InputRemainder = Input + OutputCountEven * StrideWidthElements;

    alignas(16) uint16_t PackedFilters[PointwiseFilterCountMax * PointwisePackedFilterSizeBf16];

    for (size_t f = 0; f < FilterCount; ++f) {
        const float* filter = Filter + f * FilterStrideElements;
        uint16_t* packed_filter = PackedFilters + f * PointwisePackedFilterSizeBf16;
        MlasConvPointwiseBf16PackFilterNeonAsm(filter, InputChannels, packed_filter);
    }

    if (FilterCount > 1) {
        if (OutputCountQuartet != 0) {
            alignas(16) uint16_t PackedInput[PointwiseOutputQuartetBatchMax * PointwiseInputChannelsMax * PointwisePackedInputQuartetSizeBf16];

            size_t OutputIndex = 0;
            while (OutputIndex < OutputCountQuartet) {
                const size_t OutputQuartetCount = std::min(
                    (OutputCountQuartet - OutputIndex) / 4,
                    PointwiseOutputQuartetBatchMax
                );

                MlasPackPointwiseBf16InputQuartetsNeon(
                    Input + OutputIndex * StrideWidthElements,
                    PackedInput,
                    StrideWidthElements,
                    InputStrideElements,
                    InputChannels,
                    OutputQuartetCount
                );

                for (size_t f = 0; f < FilterCount; ++f) {
                    MlasConvPointwiseBf16PackedInputKernelNeon4xAsm(
                        PackedInput,
                        PackedFilters + f * PointwisePackedFilterSizeBf16,
                        Output + f * OutputStrideElements + OutputIndex * BlockSize,
                        OutputQuartetCount,
                        InputChannels
                    );
                }

                OutputIndex += OutputQuartetCount * 4;
            }
        }

        if (OutputCountPairTail != 0) {
            alignas(16) uint16_t PackedInput[PointwiseOutputPairBatchMax * PointwiseInputChannelsMax * PointwisePackedInputPairSizeBf16];

            MlasPackPointwiseBf16InputPairsNeon(
                Input + OutputCountQuartet * StrideWidthElements,
                PackedInput,
                StrideWidthElements,
                InputStrideElements,
                InputChannels,
                OutputCountPairTail / 2
            );

            for (size_t f = 0; f < FilterCount; ++f) {
                MlasConvPointwiseBf16PackedInputKernelNeon2xAsm(
                    PackedInput,
                    PackedFilters + f * PointwisePackedFilterSizeBf16,
                    Output + f * OutputStrideElements + OutputCountQuartet * BlockSize,
                    OutputCountPairTail / 2,
                    InputChannels
                );
            }
        }
    } else {
        for (size_t f = 0; f < FilterCount; ++f) {
            float* output = Output + f * OutputStrideElements;

            if (OutputCountEven != 0) {
                MlasConvPointwiseBf16KernelNeonAsm(
                    Input,
                    PackedFilters + f * PointwisePackedFilterSizeBf16,
                    output,
                    StrideWidth,
                    InputStride,
                    InputChannels,
                    OutputCountEven);
            }
        }
    }

    if (OutputCountRemainder != 0) {
        for (size_t f = 0; f < FilterCount; ++f) {
            float* output_remainder = Output + f * OutputStrideElements + OutputCountEven * BlockSize;
            MlasConvPointwiseBf16KernelNeonSingleOutputAsm(
                InputRemainder,
                PackedFilters + f * PointwisePackedFilterSizeBf16,
                output_remainder,
                StrideWidth,
                InputStride,
                InputChannels,
                OutputCountRemainder);
        }
    }
}

#endif // __aarch64__ && __ARM_FEATURE_BF16

static void
MlasConvPointwiseFloatKernelNeonFallback(
    const float* Input,
    const float* Filter,
    float* Output,
    size_t StrideWidth,
    size_t InputChannels,
    size_t FilterCount,
    size_t InputStride,
    size_t FilterStride,
    size_t OutputStride,
    size_t OutputCount,
    const float* Bias,
    unsigned KernelFlags
    )
{
    const bool AccumulateOutput = (KernelFlags & MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT) != 0;
    const bool BiasAddition = (KernelFlags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION) != 0;
    const bool ReluActivation = (KernelFlags & MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION) != 0;
    const float firstBeta = (AccumulateOutput || BiasAddition) ? 1.0f : 0.0f;

    MLAS_BACKEND_KERNEL_SELECTOR_CONFIG mlas_backend_kernel_selector_config;
    mlas_backend_kernel_selector_config.use_kleidiai = ((KernelFlags & MLAS_CONV_KERNEL_MLAS_ARM_USE_KLEIDIAI) != 0);

    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t InputStrideElements = InputStride / sizeof(float);
    const size_t FilterStrideElements = FilterStride / sizeof(float);
    const size_t OutputStrideElements = OutputStride / sizeof(float);

    // FilterCount <= 4 (FilterSetSize), InputChannels <= 8 (MaximumInputChannelBatch/BlockSize)
    MLAS_SGEMM_DATA_PARAMS gemm_params[32];

    if (BiasAddition) {
        if (AccumulateOutput) {
            for (size_t f = 0; f < FilterCount; f++) {
                float* output = Output + f * OutputStrideElements;
                const float32x4_t BiasVector0 = MlasLoadFloat32x4(&Bias[f * BlockSize]);
                const float32x4_t BiasVector1 = MlasLoadFloat32x4(&Bias[f * BlockSize + 4]);
                const float32x4_t BiasVector2 = MlasLoadFloat32x4(&Bias[f * BlockSize + 8]);
                const float32x4_t BiasVector3 = MlasLoadFloat32x4(&Bias[f * BlockSize + 12]);
                for (size_t output_idx = 0; output_idx < OutputCount; output_idx++) {
                    MlasStoreFloat32x4(&output[output_idx * BlockSize], MlasAddFloat32x4(BiasVector0, MlasLoadFloat32x4(&output[output_idx * BlockSize])));
                    MlasStoreFloat32x4(&output[output_idx * BlockSize + 4], MlasAddFloat32x4(BiasVector1, MlasLoadFloat32x4(&output[output_idx * BlockSize + 4])));
                    MlasStoreFloat32x4(&output[output_idx * BlockSize + 8], MlasAddFloat32x4(BiasVector2, MlasLoadFloat32x4(&output[output_idx * BlockSize + 8])));
                    MlasStoreFloat32x4(&output[output_idx * BlockSize + 12], MlasAddFloat32x4(BiasVector3, MlasLoadFloat32x4(&output[output_idx * BlockSize + 12])));
                }
            }
        } else {
            for (size_t f = 0; f < FilterCount; f++) {
                float* output = Output + f * OutputStrideElements;
                const float32x4_t BiasVector0 = MlasLoadFloat32x4(&Bias[f * BlockSize]);
                const float32x4_t BiasVector1 = MlasLoadFloat32x4(&Bias[f * BlockSize + 4]);
                const float32x4_t BiasVector2 = MlasLoadFloat32x4(&Bias[f * BlockSize + 8]);
                const float32x4_t BiasVector3 = MlasLoadFloat32x4(&Bias[f * BlockSize + 12]);
                for (size_t output_idx = 0; output_idx < OutputCount; output_idx++) {
                    MlasStoreFloat32x4(&output[output_idx * BlockSize], BiasVector0);
                    MlasStoreFloat32x4(&output[output_idx * BlockSize + 4], BiasVector1);
                    MlasStoreFloat32x4(&output[output_idx * BlockSize + 8], BiasVector2);
                    MlasStoreFloat32x4(&output[output_idx * BlockSize + 12], BiasVector3);
                }
            }
        }
    }

    size_t idx = 0;
    for (size_t f = 0; f < FilterCount; f++) {
        const float* filter = Filter + f * FilterStrideElements;
        float* output = Output + f * OutputStrideElements;
        for (size_t ic = 0; ic < InputChannels; ic++, idx++) {
            gemm_params[idx].A = Input + ic * InputStrideElements;
            gemm_params[idx].B = filter + ic * BlockSize * BlockSize;
            gemm_params[idx].C = output;
            gemm_params[idx].lda = StrideWidthElements;
            gemm_params[idx].ldb = BlockSize;
            gemm_params[idx].ldc = BlockSize;
            gemm_params[idx].alpha = 1.0f;
            gemm_params[idx].beta = (ic == 0) ? firstBeta : 1.0f;
            gemm_params[idx].BIsPacked = false;
        }
    }

    MlasGemmBatch(CblasNoTrans, CblasNoTrans, OutputCount, BlockSize, BlockSize,
                  gemm_params, idx, nullptr, &mlas_backend_kernel_selector_config);

    if (ReluActivation) {
        const float32x4_t ZeroVector = MlasBroadcastFloat32x4(0.0f);
        for (size_t f = 0; f < FilterCount; f++) {
            float* output = Output + f * OutputStrideElements;
            for (size_t output_idx = 0; output_idx < OutputCount; output_idx++) {
                MlasStoreFloat32x4(&output[output_idx * BlockSize], MlasMaximumFloat32x4(MlasLoadFloat32x4(&output[output_idx * BlockSize]), ZeroVector));
                MlasStoreFloat32x4(&output[output_idx * BlockSize + 4], MlasMaximumFloat32x4(MlasLoadFloat32x4(&output[output_idx * BlockSize + 4]), ZeroVector));
                MlasStoreFloat32x4(&output[output_idx * BlockSize + 8], MlasMaximumFloat32x4(MlasLoadFloat32x4(&output[output_idx * BlockSize + 8]), ZeroVector));
                MlasStoreFloat32x4(&output[output_idx * BlockSize + 12], MlasMaximumFloat32x4(MlasLoadFloat32x4(&output[output_idx * BlockSize + 12]), ZeroVector));
            }
        }
    }
}

} // namespace

bool
MLASCALL
MlasTryConvPointwiseBf16KernelNeonAsm(
    const float* Input,
    const float* Filter,
    float* Output,
    size_t StrideWidth,
    size_t InputChannels,
    size_t FilterCount,
    size_t InputStride,
    size_t FilterStride,
    size_t OutputStride,
    size_t OutputCount,
    const float* Bias,
    unsigned KernelFlags
    )
{
#if defined(__aarch64__) && defined(__ARM_FEATURE_BF16)
    constexpr size_t PointwiseFilterCountMax = 4;
    constexpr size_t PointwiseInputChannelsMax = 8;

    const unsigned SemanticKernelFlags = MlasConvBf16SemanticKernelFlags(KernelFlags);
    const bool AccumulateOutput = (SemanticKernelFlags & MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT) != 0;

    if (BlockSize != 16 ||
        OutputCount == 0 ||
        InputChannels == 0 || InputChannels > PointwiseInputChannelsMax ||
        FilterCount == 0 || FilterCount > PointwiseFilterCountMax) {
        return false;
    }

    float* KernelOutput = Output;
    size_t KernelOutputStride = OutputStride;
    size_t ScratchOutputStrideElements = 0;

    if (AccumulateOutput) {
        ScratchOutputStrideElements = OutputCount * BlockSize;
        const size_t ScratchOutputBytes = UpAlignSize(FilterCount * ScratchOutputStrideElements * sizeof(float));
        MlasThreadedBufAlloc(ScratchOutputBytes);

        if (ThreadedBufHolder.get() == nullptr) {
            return false;
        }

        KernelOutput = reinterpret_cast<float*>(ThreadedBufHolder.get());
        KernelOutputStride = ScratchOutputStrideElements * sizeof(float);
    }

    MlasConvPointwiseFloatKernelNeonBf16Mmla(
        Input,
        Filter,
        KernelOutput,
        StrideWidth,
        InputChannels,
        FilterCount,
        InputStride,
        FilterStride,
        KernelOutputStride,
        OutputCount);

    const unsigned PostProcessFlags =
        SemanticKernelFlags & (MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION | MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION);

    if (AccumulateOutput) {
        const size_t OutputStrideElements = OutputStride / sizeof(float);
        const float* BiasData = MlasConvBf16BiasData(Bias, PostProcessFlags);

        for (size_t f = 0; f < FilterCount; ++f) {
            MlasMergePointwiseBf16OutputsNeon(
                KernelOutput + f * ScratchOutputStrideElements,
                Output + f * OutputStrideElements,
                OutputCount,
                BiasData == nullptr ? nullptr : BiasData + f * BlockSize,
                PostProcessFlags
            );
        }
    } else if (PostProcessFlags != 0) {
        const size_t OutputStrideElements = OutputStride / sizeof(float);
        const float* BiasData = MlasConvBf16BiasData(Bias, PostProcessFlags);

        for (size_t f = 0; f < FilterCount; ++f) {
            MlasConvBf16PostProcessOutputs(
                Output + f * OutputStrideElements,
                OutputCount,
                BiasData == nullptr ? nullptr : BiasData + f * BlockSize,
                PostProcessFlags
            );
        }
    }

    return true;
#else
    MLAS_UNREFERENCED_PARAMETER(Input);
    MLAS_UNREFERENCED_PARAMETER(Filter);
    MLAS_UNREFERENCED_PARAMETER(Output);
    MLAS_UNREFERENCED_PARAMETER(StrideWidth);
    MLAS_UNREFERENCED_PARAMETER(InputChannels);
    MLAS_UNREFERENCED_PARAMETER(FilterCount);
    MLAS_UNREFERENCED_PARAMETER(InputStride);
    MLAS_UNREFERENCED_PARAMETER(FilterStride);
    MLAS_UNREFERENCED_PARAMETER(OutputStride);
    MLAS_UNREFERENCED_PARAMETER(OutputCount);
    MLAS_UNREFERENCED_PARAMETER(Bias);
    MLAS_UNREFERENCED_PARAMETER(KernelFlags);
    return false;
#endif
}

void
    MLASCALL
    MlasConvPointwiseFloatKernelNeon(
        const float* Input,
        const float* Filter,
        float* Output,
        size_t StrideWidth,
        size_t InputChannels,
        size_t FilterCount,
        size_t InputStride,
        size_t FilterStride,
        size_t OutputStride,
        size_t OutputCount,
        const float* Bias,
        unsigned KernelFlags
    )
{
    MlasConvPointwiseFloatKernelNeonFallback(
        Input,
        Filter,
        Output,
        StrideWidth,
        InputChannels,
        FilterCount,
        InputStride,
        FilterStride,
        OutputStride,
        OutputCount,
        Bias,
        KernelFlags);
}

#endif
