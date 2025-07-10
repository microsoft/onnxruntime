/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    spool_kernel_neon.cpp

Abstract:

    This module implements the single precision pooling kernels for ARM NEON.

--*/

#if defined(__aarch64__) || defined(_M_ARM64)

#include <cstddef>
#include <limits>
#include <vector>

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

    const size_t BlockSize = MlasNchwcGetBlockSize();
    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t InputWidthElements = InputWidth / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);

    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;

    // Initialize negative infinity vector for out-of-bounds values using MLAS intrinsics
    const MLAS_FLOAT32X4 NegInfVector = MlasBroadcastFloat32x4(-std::numeric_limits<float>::infinity());

    for (size_t output_idx = 0; output_idx < TotalOutputCount; output_idx++) {
        // Initialize maximum values to negative infinity using MLAS intrinsics
        MLAS_FLOAT32X4 MaxVector = NegInfVector;

        for (size_t kh = 0; kh < KernelHeight; kh++) {
            for (size_t kw = 0; kw < KernelWidth; kw++) {
                const float* input_ptr = Input + output_idx * StrideWidthElements +
                                         kh * DilatedInputWidthElements + kw * DilationWidthElements;

                const float* row_start = InputBase + kh * DilatedInputWidthElements;
                const float* row_end = row_start + InputWidthElements;

                MLAS_FLOAT32X4 InputVector;
                if (input_ptr >= row_start && (input_ptr + BlockSize - 1) < row_end) {
                    // All elements are within bounds, load directly using MLAS intrinsics
                    InputVector = MlasLoadFloat32x4(input_ptr);
                } else {
                    // Some elements might be out of bounds, load individually
                    std::vector<float> values(BlockSize);
                    for (size_t i = 0; i < BlockSize; i++) {
                        const float* element_ptr = input_ptr + i;
                        if (element_ptr >= row_start && element_ptr < row_end) {
                            values[i] = *element_ptr;
                        } else {
                            values[i] = -std::numeric_limits<float>::infinity();
                        }
                    }
                    InputVector = MlasLoadFloat32x4(values.data());
                }

                // Update maximum using MLAS intrinsics
                MaxVector = MlasMaximumFloat32x4(MaxVector, InputVector);
            }
        }

        // Store the results using MLAS intrinsics
        MlasStoreFloat32x4(&Output[output_idx * BlockSize], MaxVector);
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

    const size_t BlockSize = MlasNchwcGetBlockSize();
    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t InputWidthElements = InputWidth / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);

    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;

    // Initialize zero vector using MLAS intrinsics
    const MLAS_FLOAT32X4 ZeroVector = MlasZeroFloat32x4();

    for (size_t output_idx = 0; output_idx < TotalOutputCount; output_idx++) {
        // Initialize sum vector using MLAS intrinsics
        MLAS_FLOAT32X4 SumVector = ZeroVector;

        // Track valid count for each element (allocate dynamically for variable block size)
        std::vector<uint32_t> valid_count(BlockSize, 0);

        for (size_t kh = 0; kh < KernelHeight; kh++) {
            for (size_t kw = 0; kw < KernelWidth; kw++) {
                const float* input_ptr = Input + output_idx * StrideWidthElements +
                                         kh * DilatedInputWidthElements + kw * DilationWidthElements;

                const float* row_start = InputBase + kh * DilatedInputWidthElements;
                const float* row_end = row_start + InputWidthElements;

                MLAS_FLOAT32X4 InputVector;

                if (input_ptr >= row_start && (input_ptr + BlockSize - 1) < row_end) {
                    // All elements are within bounds, load directly using MLAS intrinsics
                    InputVector = MlasLoadFloat32x4(input_ptr);
                    // All elements are valid
                    for (size_t i = 0; i < BlockSize; i++) {
                        valid_count[i]++;
                    }
                } else {
                    // Some elements might be out of bounds, handle individually
                    std::vector<float> values(BlockSize);
                    for (size_t i = 0; i < BlockSize; i++) {
                        const float* element_ptr = input_ptr + i;
                        if (element_ptr >= row_start && element_ptr < row_end) {
                            values[i] = *element_ptr;
                            valid_count[i]++;
                        } else {
                            values[i] = 0.0f;
                        }
                    }
                    InputVector = MlasLoadFloat32x4(values.data());
                }

                // Add to sum using MLAS intrinsics
                SumVector = MlasAddFloat32x4(SumVector, InputVector);
            }
        }

        // Compute average by dividing by valid count for each element
        std::vector<float> results(BlockSize);
        MlasStoreFloat32x4(results.data(), SumVector);

        for (size_t i = 0; i < BlockSize; i++) {
            if (valid_count[i] > 0) {
                results[i] = results[i] / static_cast<float>(valid_count[i]);
            } else {
                results[i] = 0.0f;
            }
        }

        // Store the results using MLAS intrinsics
        MLAS_FLOAT32X4 ResultVector = MlasLoadFloat32x4(results.data());
        MlasStoreFloat32x4(&Output[output_idx * BlockSize], ResultVector);
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

    const size_t BlockSize = MlasNchwcGetBlockSize();
    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t InputWidthElements = InputWidth / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);

    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;

    // Use ActualKernelSize as provided by the caller
    const float KernelSize = static_cast<float>(ActualKernelSize);

    // Initialize vectors using MLAS intrinsics
    const MLAS_FLOAT32X4 ZeroVector = MlasZeroFloat32x4();
    const MLAS_FLOAT32X4 KernelSizeVector = MlasBroadcastFloat32x4(KernelSize);

    for (size_t output_idx = 0; output_idx < TotalOutputCount; output_idx++) {
        bool is_main_region = (output_idx >= OutputCountLeftPad && output_idx < OutputCountLeftPad + OutputCount);

        // Initialize sum vector using MLAS intrinsics
        MLAS_FLOAT32X4 SumVector = ZeroVector;

        for (size_t kh = 0; kh < KernelHeight; kh++) {
            for (size_t kw = 0; kw < KernelWidth; kw++) {
                const float* input_ptr = Input + output_idx * StrideWidthElements +
                                         kh * DilatedInputWidthElements + kw * DilationWidthElements;

                const float* row_start = InputBase + kh * DilatedInputWidthElements;
                const float* row_end = row_start + InputWidthElements;

                MLAS_FLOAT32X4 InputVector;

                if (is_main_region || (input_ptr >= row_start && (input_ptr + BlockSize - 1) < row_end)) {
                    // All elements are within bounds or in main region, load directly using MLAS intrinsics
                    InputVector = MlasLoadFloat32x4(input_ptr);
                } else {
                    // Some elements might be out of bounds, handle individually
                    std::vector<float> values(BlockSize);
                    for (size_t i = 0; i < BlockSize; i++) {
                        const float* element_ptr = input_ptr + i;
                        if (is_main_region || (element_ptr >= row_start && element_ptr < row_end)) {
                            values[i] = *element_ptr;
                        } else {
                            values[i] = 0.0f;  // Padding values are treated as 0
                        }
                    }
                    InputVector = MlasLoadFloat32x4(values.data());
                }

                // Add to sum using MLAS intrinsics
                SumVector = MlasAddFloat32x4(SumVector, InputVector);
            }
        }

        // Compute average by dividing by kernel size using MLAS intrinsics
        MLAS_FLOAT32X4 ResultVector = MlasDivideFloat32x4(SumVector, KernelSizeVector);

        // Store the results using MLAS intrinsics
        MlasStoreFloat32x4(&Output[output_idx * BlockSize], ResultVector);
    }
}

#endif  // __aarch64__ || _M_ARM64