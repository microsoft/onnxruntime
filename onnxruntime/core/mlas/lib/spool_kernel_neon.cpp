/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    spool_kernel_neon.cpp

Abstract:

    This module implements the single precision pooling kernels for ARM NEON.

--*/

// #include "spool.h"

#if defined(__aarch64__) || defined(_M_ARM64)

#include <algorithm>
#include <cstddef>
#include <limits>

#include "arm_neon.h"
#include "mlasi.h"

void
    MLASCALL
    MlasPoolMaximumFloatKernelNeon(
        const float* Input,
        float* Output,
        size_t StrideWidth,
        size_t DilationWidth,
        size_t InputStride,
        size_t ActualKernelSize,
        size_t KernelHeight,
        size_t KernelWidth,
        const float* InputBase,
        size_t InputWidth,
        size_t DilatedInputWidth,
        size_t OutputCountLeftPad,
        size_t OutputCount,
        size_t OutputCountRightPad
    )
{
    MLAS_UNREFERENCED_PARAMETER(ActualKernelSize);
    MLAS_UNREFERENCED_PARAMETER(InputStride);
    
    constexpr size_t BlockSize = 4;
    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t InputWidthElements = InputWidth / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);

    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;

    for (size_t output_idx = 0; output_idx < TotalOutputCount; output_idx++) {
        // Initialize maximum values to negative infinity
        float max_values[BlockSize];
        for (size_t i = 0; i < BlockSize; i++) {
            max_values[i] = std::numeric_limits<float>::lowest();
        }

        for (size_t kh = 0; kh < KernelHeight; kh++) {
            for (size_t kw = 0; kw < KernelWidth; kw++) {
                const float* input_base = Input + output_idx * StrideWidthElements +
                                          kh * DilatedInputWidthElements + kw * DilationWidthElements;

                const float* input_row_start = InputBase + kh * DilatedInputWidthElements;
                const float* input_row_end = input_row_start + InputWidthElements;

                for (size_t block_idx = 0; block_idx < BlockSize; block_idx++) {
                    const float* input_element = input_base + block_idx;
                    
                    float input_value;
                    if (input_element >= input_row_start && input_element < input_row_end) {
                        input_value = *input_element;
                    } else {
                        input_value = std::numeric_limits<float>::lowest();
                    }

                    max_values[block_idx] = std::max(max_values[block_idx], input_value);
                }
            }
        }

        // Store the results
        for (size_t i = 0; i < BlockSize; i++) {
            Output[output_idx * BlockSize + i] = max_values[i];
        }
    }
}

void
    MLASCALL
    MlasPoolAverageExcludePadFloatKernelNeon(
        const float* Input,
        float* Output,
        size_t StrideWidth,
        size_t DilationWidth,
        size_t InputStride,
        size_t ActualKernelSize,
        size_t KernelHeight,
        size_t KernelWidth,
        const float* InputBase,
        size_t InputWidth,
        size_t DilatedInputWidth,
        size_t OutputCountLeftPad,
        size_t OutputCount,
        size_t OutputCountRightPad
    )
{
    MLAS_UNREFERENCED_PARAMETER(ActualKernelSize);
    MLAS_UNREFERENCED_PARAMETER(InputStride);
    
    constexpr size_t BlockSize = 4;
    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t InputWidthElements = InputWidth / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);

    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;

    for (size_t output_idx = 0; output_idx < TotalOutputCount; output_idx++) {
        // Initialize sum values and count
        float sum_values[BlockSize];
        size_t valid_count[BlockSize];
        for (size_t i = 0; i < BlockSize; i++) {
            sum_values[i] = 0.0f;
            valid_count[i] = 0;
        }

        for (size_t kh = 0; kh < KernelHeight; kh++) {
            for (size_t kw = 0; kw < KernelWidth; kw++) {
                const float* input_base = Input + output_idx * StrideWidthElements +
                                          kh * DilatedInputWidthElements + kw * DilationWidthElements;

                const float* input_row_start = InputBase + kh * DilatedInputWidthElements;
                const float* input_row_end = input_row_start + InputWidthElements;

                for (size_t block_idx = 0; block_idx < BlockSize; block_idx++) {
                    const float* input_element = input_base + block_idx;
                    
                    if (input_element >= input_row_start && input_element < input_row_end) {
                        float input_value = *input_element;
                        sum_values[block_idx] += input_value;
                        valid_count[block_idx]++;
                    }
                }
            }
        }

        // Store the results (average excluding padding)
        for (size_t i = 0; i < BlockSize; i++) {
            if (valid_count[i] > 0) {
                Output[output_idx * BlockSize + i] = sum_values[i] / static_cast<float>(valid_count[i]);
            } else {
                Output[output_idx * BlockSize + i] = 0.0f;
            }
        }
    }
}

void
    MLASCALL
    MlasPoolAverageIncludePadFloatKernelNeon(
        const float* Input,
        float* Output,
        size_t StrideWidth,
        size_t DilationWidth,
        size_t InputStride,
        size_t ActualKernelSize,
        size_t KernelHeight,
        size_t KernelWidth,
        const float* InputBase,
        size_t InputWidth,
        size_t DilatedInputWidth,
        size_t OutputCountLeftPad,
        size_t OutputCount,
        size_t OutputCountRightPad
    )
{
    MLAS_UNREFERENCED_PARAMETER(InputStride);
    
    constexpr size_t BlockSize = 4;
    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t InputWidthElements = InputWidth / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);

    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;
    
    // Use ActualKernelSize if provided, otherwise compute from dimensions
    const float KernelSize = (ActualKernelSize > 0) ? 
        static_cast<float>(ActualKernelSize) : 
        static_cast<float>(KernelHeight * KernelWidth);

    for (size_t output_idx = 0; output_idx < TotalOutputCount; output_idx++) {
        bool is_main_region = (output_idx >= OutputCountLeftPad && output_idx < OutputCountLeftPad + OutputCount);

        // Initialize sum values
        float sum_values[BlockSize];
        for (size_t i = 0; i < BlockSize; i++) {
            sum_values[i] = 0.0f;
        }

        for (size_t kh = 0; kh < KernelHeight; kh++) {
            for (size_t kw = 0; kw < KernelWidth; kw++) {
                const float* input_ptr = Input + output_idx * StrideWidthElements +
                                         kh * DilatedInputWidthElements + kw * DilationWidthElements;

                // Check bounds for this kernel position
                const float* row_start = InputBase + kh * DilatedInputWidthElements;
                const float* row_end = row_start + InputWidthElements;

                for (size_t block_idx = 0; block_idx < BlockSize; block_idx++) {
                    const float* element_ptr = input_ptr + block_idx;
                    
                    float value;
                    if (is_main_region || (element_ptr >= row_start && element_ptr < row_end)) {
                        value = *element_ptr;
                    } else {
                        value = 0.0f;  // Padding values are treated as 0
                    }

                    sum_values[block_idx] += value;
                }
            }
        }

        // Store the results (divide by total kernel size for include pad)
        for (size_t i = 0; i < BlockSize; i++) {
            Output[output_idx * BlockSize + i] = sum_values[i] / KernelSize;
        }
    }
}

#endif  // __aarch64__ || _M_ARM64