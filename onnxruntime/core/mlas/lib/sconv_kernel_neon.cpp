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

#include "arm_neon.h"
#include "mlasi.h"

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
    
}

//
// Implementation of MlasConvNchwcFloatKernelNeon
//
// This kernel performs 2D convolution optimized for NCHWc format.
// It processes multiple filters in blocks to improve cache efficiency.
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
    const float32x4_t ZeroVector = vdupq_n_f32(0.0f);

    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t InputStrideElements = InputStride / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);

    (void)InputStrideElements;

    const size_t InputWidthElements = InputWidth / sizeof(float);

    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;

    for (size_t output_idx = 0; output_idx < TotalOutputCount; output_idx++) {
        bool is_main_region = (output_idx >= OutputCountLeftPad && output_idx < OutputCountLeftPad + OutputCount);

        float32x4_t Accumulator;

        if (AccumulateOutput) {
            Accumulator = MlasLoadFloat32x4(&Output[output_idx * BlockSize]);
        } else if (BiasAddition) {
            Accumulator = MlasLoadFloat32x4(Bias);
        } else {
            Accumulator = vdupq_n_f32(0.0f);
        }

        for (size_t kh = 0; kh < KernelHeight; kh++) {
            for (size_t kw = 0; kw < KernelWidth; kw++) {
                size_t kernel_pos = kh * KernelWidth + kw;

                const float* input_base = Input + output_idx * StrideWidthElements +
                                          kh * DilatedInputWidthElements + kw * DilationWidthElements;

                float32x4_t InputVector;

                if (is_main_region) {
                    InputVector = MlasLoadFloat32x4(input_base);
                } else {
                    float input_values[4];
                    for (size_t i = 0; i < BlockSize; i++) {
                        const float* input_element = input_base + i;
                        const float* input_row_start = InputBase + kh * DilatedInputWidthElements;
                        const float* input_row_end = input_row_start + InputWidthElements;

                        if (input_element >= input_row_start && input_element < input_row_end) {
                            input_values[i] = *input_element;
                        } else {
                            input_values[i] = 0.0f;
                        }
                    }
                    InputVector = MlasLoadFloat32x4(input_values);
                }

                const float32x4_t FilterVector = MlasLoadFloat32x4(&Filter[kernel_pos * BlockSize]);

                Accumulator = MlasMultiplyAddFloat32x4(InputVector, FilterVector, Accumulator);
            }
        }

        if (ReluActivation) {
            Accumulator = MlasMaximumFloat32x4(Accumulator, ZeroVector);
        }

        MlasStoreFloat32x4(&Output[output_idx * BlockSize], Accumulator);
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
    const float32x4_t ZeroVector = vdupq_n_f32(0.0f);

    for (size_t i = 0; i < OutputCount; i++) {
        for (size_t f = 0; f < FilterCount; f++) {
            const float* filter = Filter + f * FilterStrideElements;
            float* output = Output + f * OutputStrideElements;
            float32x4_t Accumulator;
            if (AccumulateOutput) {
                Accumulator = MlasLoadFloat32x4(&output[i * BlockSize]);
            } else if (BiasAddition) {
                Accumulator = MlasLoadFloat32x4(&Bias[f * BlockSize]);
            } else {
                Accumulator = vdupq_n_f32(0.0f);
            }
            for (size_t c = 0; c < InputChannels; c++) {
                const float* input_ptr = Input + c * InputStrideElements + i * StrideWidthElements;
                for (size_t input_b = 0; input_b < BlockSize; input_b++) {
                    const float input_value = input_ptr[input_b];
                    const float32x4_t InputVector = vdupq_n_f32(input_value);
                    const float* filter_ptr = filter + (c * BlockSize + input_b) * BlockSize;
                    const float32x4_t FilterVector = MlasLoadFloat32x4(filter_ptr);
                    Accumulator = MlasMultiplyAddFloat32x4(InputVector, FilterVector, Accumulator);
                }
            }
            if (ReluActivation) {
                Accumulator = MlasMaximumFloat32x4(Accumulator, ZeroVector);
            }
            MlasStoreFloat32x4(&output[i * BlockSize], Accumulator);
        }
    }
}

#endif  // __aarch64__ || _M_ARM64