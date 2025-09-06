/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sconv_kernel_neon.cpp

Abstract:

    This module implements the single precision convolution kernels for ARM NEON.

--*/

#include "sconv.h"

#if defined(__aarch64__) || defined(_M_ARM64)

#include <algorithm>
#include <cstddef>

#include "mlasi.h"

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
    const bool AccumulateOutput = (KernelFlags & MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT) != 0;
    const bool BiasAddition = (KernelFlags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION) != 0;
    const bool ReluActivation = (KernelFlags & MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION) != 0;

    const size_t BlockSize = MlasNchwcGetBlockSize();
    const float32x4_t ZeroVector = MlasBroadcastFloat32x4(0.0f);

    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t FilterStrideElements = FilterStride / sizeof(float);
    const size_t OutputStrideElements = OutputStride / sizeof(float);
    const size_t InputWidthElements = InputWidth / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);

    (void)InputStride;

    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;

    for (size_t output_idx = 0; output_idx < TotalOutputCount; output_idx++) {
        bool is_main_region = (output_idx >= OutputCountLeftPad && output_idx < OutputCountLeftPad + OutputCount);

        for (size_t filterSetBlock = 0; filterSetBlock < FilterCount; filterSetBlock++) {
            const float* filter = Filter + filterSetBlock * FilterStrideElements;
            float* output = Output + filterSetBlock * OutputStrideElements;

            float32x4_t Accumulator0, Accumulator1, Accumulator2, Accumulator3;

            if (AccumulateOutput) {
                Accumulator0 = MlasLoadFloat32x4(&output[output_idx * BlockSize]);
                Accumulator1 = MlasLoadFloat32x4(&output[output_idx * BlockSize + 4]);
                Accumulator2 = MlasLoadFloat32x4(&output[output_idx * BlockSize + 8]);
                Accumulator3 = MlasLoadFloat32x4(&output[output_idx * BlockSize + 12]);
            } else {
                Accumulator0 = MlasBroadcastFloat32x4(0.0f);
                Accumulator1 = MlasBroadcastFloat32x4(0.0f);
                Accumulator2 = MlasBroadcastFloat32x4(0.0f);
                Accumulator3 = MlasBroadcastFloat32x4(0.0f);
            }

            if (BiasAddition) {
                const float32x4_t BiasVector0 = MlasLoadFloat32x4(&Bias[filterSetBlock * BlockSize]);
                const float32x4_t BiasVector1 = MlasLoadFloat32x4(&Bias[filterSetBlock * BlockSize + 4]);
                const float32x4_t BiasVector2 = MlasLoadFloat32x4(&Bias[filterSetBlock * BlockSize + 8]);
                const float32x4_t BiasVector3 = MlasLoadFloat32x4(&Bias[filterSetBlock * BlockSize + 12]);

                Accumulator0 = MlasAddFloat32x4(Accumulator0, BiasVector0);
                Accumulator1 = MlasAddFloat32x4(Accumulator1, BiasVector1);
                Accumulator2 = MlasAddFloat32x4(Accumulator2, BiasVector2);
                Accumulator3 = MlasAddFloat32x4(Accumulator3, BiasVector3);
            }

            for (size_t kh = 0; kh < KernelHeight; kh++) {
                for (size_t kw = 0; kw < KernelWidth; kw++) {
                    const float* input_base = Input + output_idx * StrideWidthElements +
                                              kh * DilatedInputWidthElements + kw * DilationWidthElements;

                    if (IsNchwcFormat) {
                        for (size_t filterBlock = 0; filterBlock < BlockSize; filterBlock++) {
                            const float* input_element = input_base + filterBlock;
                            const float* input_row_start = InputBase + kh * DilatedInputWidthElements;
                            const float* input_row_end = input_row_start + InputWidthElements;

                            float input_value;
                            if (is_main_region || (input_element >= input_row_start && input_element < input_row_end)) {
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
                        if (is_main_region || (input_base >= input_row_start && input_base < input_row_end)) {
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
            }

            if (ReluActivation) {
                Accumulator0 = MlasMaximumFloat32x4(Accumulator0, ZeroVector);
                Accumulator1 = MlasMaximumFloat32x4(Accumulator1, ZeroVector);
                Accumulator2 = MlasMaximumFloat32x4(Accumulator2, ZeroVector);
                Accumulator3 = MlasMaximumFloat32x4(Accumulator3, ZeroVector);
            }

            MlasStoreFloat32x4(&output[output_idx * BlockSize], Accumulator0);
            MlasStoreFloat32x4(&output[output_idx * BlockSize + 4], Accumulator1);
            MlasStoreFloat32x4(&output[output_idx * BlockSize + 8], Accumulator2);
            MlasStoreFloat32x4(&output[output_idx * BlockSize + 12], Accumulator3);
        }
    }
}

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
}

//
// Implementation of MlasConvDepthwiseFloatKernelNeon
//
// This kernel performs depthwise separable convolution where each input channel
// is convolved with its own filter. This is more efficient than standard convolution
// for certain network architectures like MobileNets.
//

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
    const bool AccumulateOutput = (KernelFlags & MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT) != 0;
    const bool BiasAddition = (KernelFlags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION) != 0;
    const bool ReluActivation = (KernelFlags & MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION) != 0;

    const size_t BlockSize = MlasNchwcGetBlockSize();
    const float32x4_t ZeroVector = MlasBroadcastFloat32x4(0.0f);

    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t InputStrideElements = InputStride / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);

    (void)InputStrideElements;

    const size_t InputWidthElements = InputWidth / sizeof(float);

    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;

    for (size_t output_idx = 0; output_idx < TotalOutputCount; output_idx++) {
        bool is_main_region = (output_idx >= OutputCountLeftPad && output_idx < OutputCountLeftPad + OutputCount);

        float32x4_t Accumulator0, Accumulator1, Accumulator2, Accumulator3;

        if (AccumulateOutput) {
            Accumulator0 = MlasLoadFloat32x4(&Output[output_idx * BlockSize]);
            Accumulator1 = MlasLoadFloat32x4(&Output[output_idx * BlockSize + 4]);
            Accumulator2 = MlasLoadFloat32x4(&Output[output_idx * BlockSize + 8]);
            Accumulator3 = MlasLoadFloat32x4(&Output[output_idx * BlockSize + 12]);
        } else {
            Accumulator0 = MlasBroadcastFloat32x4(0.0f);
            Accumulator1 = MlasBroadcastFloat32x4(0.0f);
            Accumulator2 = MlasBroadcastFloat32x4(0.0f);
            Accumulator3 = MlasBroadcastFloat32x4(0.0f);
        }

        if (BiasAddition) {
            const float32x4_t BiasVector0 = MlasLoadFloat32x4(Bias);
            const float32x4_t BiasVector1 = MlasLoadFloat32x4(Bias + 4);
            const float32x4_t BiasVector2 = MlasLoadFloat32x4(Bias + 8);
            const float32x4_t BiasVector3 = MlasLoadFloat32x4(Bias + 12);

            Accumulator0 = MlasAddFloat32x4(Accumulator0, BiasVector0);
            Accumulator1 = MlasAddFloat32x4(Accumulator1, BiasVector1);
            Accumulator2 = MlasAddFloat32x4(Accumulator2, BiasVector2);
            Accumulator3 = MlasAddFloat32x4(Accumulator3, BiasVector3);
        }

        for (size_t kh = 0; kh < KernelHeight; kh++) {
            for (size_t kw = 0; kw < KernelWidth; kw++) {
                size_t kernel_pos = kh * KernelWidth + kw;

                const float* input_base = Input + output_idx * StrideWidthElements +
                                          kh * DilatedInputWidthElements + kw * DilationWidthElements;

                float32x4_t InputVector0;
                if (is_main_region) {
                    InputVector0 = MlasLoadFloat32x4(input_base);
                } else {
                    float input_values[4];
                    for (size_t i = 0; i < 4; i++) {
                        const float* input_element = input_base + i;
                        const float* input_row_start = InputBase + kh * DilatedInputWidthElements;
                        const float* input_row_end = input_row_start + InputWidthElements;

                        if (input_element >= input_row_start && input_element < input_row_end) {
                            input_values[i] = *input_element;
                        } else {
                            input_values[i] = 0.0f;
                        }
                    }
                    InputVector0 = MlasLoadFloat32x4(input_values);
                }

                float32x4_t InputVector1;
                if (is_main_region) {
                    InputVector1 = MlasLoadFloat32x4(input_base + 4);
                } else {
                    float input_values[4];
                    for (size_t i = 0; i < 4; i++) {
                        const float* input_element = input_base + 4 + i;
                        const float* input_row_start = InputBase + kh * DilatedInputWidthElements;
                        const float* input_row_end = input_row_start + InputWidthElements;

                        if (input_element >= input_row_start && input_element < input_row_end) {
                            input_values[i] = *input_element;
                        } else {
                            input_values[i] = 0.0f;
                        }
                    }
                    InputVector1 = MlasLoadFloat32x4(input_values);
                }

                float32x4_t InputVector2;
                if (is_main_region) {
                    InputVector2 = MlasLoadFloat32x4(input_base + 8);
                } else {
                    float input_values[4];
                    for (size_t i = 0; i < 4; i++) {
                        const float* input_element = input_base + 8 + i;
                        const float* input_row_start = InputBase + kh * DilatedInputWidthElements;
                        const float* input_row_end = input_row_start + InputWidthElements;

                        if (input_element >= input_row_start && input_element < input_row_end) {
                            input_values[i] = *input_element;
                        } else {
                            input_values[i] = 0.0f;
                        }
                    }
                    InputVector2 = MlasLoadFloat32x4(input_values);
                }

                float32x4_t InputVector3;
                if (is_main_region) {
                    InputVector3 = MlasLoadFloat32x4(input_base + 12);
                } else {
                    float input_values[4];
                    for (size_t i = 0; i < 4; i++) {
                        const float* input_element = input_base + 12 + i;
                        const float* input_row_start = InputBase + kh * DilatedInputWidthElements;
                        const float* input_row_end = input_row_start + InputWidthElements;

                        if (input_element >= input_row_start && input_element < input_row_end) {
                            input_values[i] = *input_element;
                        } else {
                            input_values[i] = 0.0f;
                        }
                    }
                    InputVector3 = MlasLoadFloat32x4(input_values);
                }

                const float32x4_t FilterVector0 = MlasLoadFloat32x4(&Filter[kernel_pos * BlockSize]);
                const float32x4_t FilterVector1 = MlasLoadFloat32x4(&Filter[kernel_pos * BlockSize + 4]);
                const float32x4_t FilterVector2 = MlasLoadFloat32x4(&Filter[kernel_pos * BlockSize + 8]);
                const float32x4_t FilterVector3 = MlasLoadFloat32x4(&Filter[kernel_pos * BlockSize + 12]);

                Accumulator0 = MlasMultiplyAddFloat32x4(InputVector0, FilterVector0, Accumulator0);
                Accumulator1 = MlasMultiplyAddFloat32x4(InputVector1, FilterVector1, Accumulator1);
                Accumulator2 = MlasMultiplyAddFloat32x4(InputVector2, FilterVector2, Accumulator2);
                Accumulator3 = MlasMultiplyAddFloat32x4(InputVector3, FilterVector3, Accumulator3);
            }
        }

        if (ReluActivation) {
            Accumulator0 = MlasMaximumFloat32x4(Accumulator0, ZeroVector);
            Accumulator1 = MlasMaximumFloat32x4(Accumulator1, ZeroVector);
            Accumulator2 = MlasMaximumFloat32x4(Accumulator2, ZeroVector);
            Accumulator3 = MlasMaximumFloat32x4(Accumulator3, ZeroVector);
        }

        MlasStoreFloat32x4(&Output[output_idx * BlockSize], Accumulator0);
        MlasStoreFloat32x4(&Output[output_idx * BlockSize + 4], Accumulator1);
        MlasStoreFloat32x4(&Output[output_idx * BlockSize + 8], Accumulator2);
        MlasStoreFloat32x4(&Output[output_idx * BlockSize + 12], Accumulator3);
    }
}

//
// Implementation of MlasConvPointwiseFloatKernelNeon
//
// This kernel performs pointwise (1x1) convolution which is essentially
// a matrix multiplication across the channel dimension. It's optimized
// for cases where the kernel size is 1x1.
//

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
    const bool AccumulateOutput = (KernelFlags & MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT) != 0;
    const bool BiasAddition = (KernelFlags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION) != 0;
    const bool ReluActivation = (KernelFlags & MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION) != 0;

    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t InputStrideElements = InputStride / sizeof(float);
    const size_t FilterStrideElements = FilterStride / sizeof(float);
    const size_t OutputStrideElements = OutputStride / sizeof(float);

    const size_t BlockSize = MlasNchwcGetBlockSize();
    const float32x4_t ZeroVector = MlasBroadcastFloat32x4(0.0f);

    for (size_t output_idx = 0; output_idx < OutputCount; output_idx++) {
        for (size_t f = 0; f < FilterCount; f++) {
            const float* filter = Filter + f * FilterStrideElements;
            float* output = Output + f * OutputStrideElements;

            float32x4_t Accumulator0, Accumulator1, Accumulator2, Accumulator3;

            if (AccumulateOutput) {
                Accumulator0 = MlasLoadFloat32x4(&output[output_idx * BlockSize]);
                Accumulator1 = MlasLoadFloat32x4(&output[output_idx * BlockSize + 4]);
                Accumulator2 = MlasLoadFloat32x4(&output[output_idx * BlockSize + 8]);
                Accumulator3 = MlasLoadFloat32x4(&output[output_idx * BlockSize + 12]);
            } else {
                Accumulator0 = MlasBroadcastFloat32x4(0.0f);
                Accumulator1 = MlasBroadcastFloat32x4(0.0f);
                Accumulator2 = MlasBroadcastFloat32x4(0.0f);
                Accumulator3 = MlasBroadcastFloat32x4(0.0f);
            }

            if (BiasAddition) {
                const float32x4_t BiasVector0 = MlasLoadFloat32x4(&Bias[f * BlockSize]);
                const float32x4_t BiasVector1 = MlasLoadFloat32x4(&Bias[f * BlockSize + 4]);
                const float32x4_t BiasVector2 = MlasLoadFloat32x4(&Bias[f * BlockSize + 8]);
                const float32x4_t BiasVector3 = MlasLoadFloat32x4(&Bias[f * BlockSize + 12]);

                Accumulator0 = MlasAddFloat32x4(Accumulator0, BiasVector0);
                Accumulator1 = MlasAddFloat32x4(Accumulator1, BiasVector1);
                Accumulator2 = MlasAddFloat32x4(Accumulator2, BiasVector2);
                Accumulator3 = MlasAddFloat32x4(Accumulator3, BiasVector3);
            }

            for (size_t c = 0; c < InputChannels; c++) {
                const float* input_ptr = Input + c * InputStrideElements + output_idx * StrideWidthElements;

                for (size_t input_b = 0; input_b < BlockSize; input_b++) {
                    const float input_value = input_ptr[input_b];
                    const float32x4_t InputVector = MlasBroadcastFloat32x4(input_value);

                    const float* filter_ptr = filter + (c * BlockSize + input_b) * BlockSize;

                    const float32x4_t FilterVector0 = MlasLoadFloat32x4(filter_ptr);
                    const float32x4_t FilterVector1 = MlasLoadFloat32x4(filter_ptr + 4);
                    const float32x4_t FilterVector2 = MlasLoadFloat32x4(filter_ptr + 8);
                    const float32x4_t FilterVector3 = MlasLoadFloat32x4(filter_ptr + 12);

                    Accumulator0 = MlasMultiplyAddFloat32x4(InputVector, FilterVector0, Accumulator0);
                    Accumulator1 = MlasMultiplyAddFloat32x4(InputVector, FilterVector1, Accumulator1);
                    Accumulator2 = MlasMultiplyAddFloat32x4(InputVector, FilterVector2, Accumulator2);
                    Accumulator3 = MlasMultiplyAddFloat32x4(InputVector, FilterVector3, Accumulator3);
                }
            }

            if (ReluActivation) {
                Accumulator0 = MlasMaximumFloat32x4(Accumulator0, ZeroVector);
                Accumulator1 = MlasMaximumFloat32x4(Accumulator1, ZeroVector);
                Accumulator2 = MlasMaximumFloat32x4(Accumulator2, ZeroVector);
                Accumulator3 = MlasMaximumFloat32x4(Accumulator3, ZeroVector);
            }

            MlasStoreFloat32x4(&output[output_idx * BlockSize], Accumulator0);
            MlasStoreFloat32x4(&output[output_idx * BlockSize + 4], Accumulator1);
            MlasStoreFloat32x4(&output[output_idx * BlockSize + 8], Accumulator2);
            MlasStoreFloat32x4(&output[output_idx * BlockSize + 12], Accumulator3);
        }
    }
}

#endif  // __aarch64__ || _M_ARM64