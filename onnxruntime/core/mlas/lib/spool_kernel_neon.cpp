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

    const float MaxPaddingValue = std::numeric_limits<float>::lowest();
    const MLAS_FLOAT32X4 MaxPaddingVector = MlasBroadcastFloat32x4(MaxPaddingValue);

    for (size_t output_idx = 0; output_idx < TotalOutputCount; output_idx++) {
        MLAS_FLOAT32X4 MaxVector = MaxPaddingVector;

        for (size_t kh = 0; kh < KernelHeight; kh++) {
            const float* row_start = InputBase + kh * DilatedInputWidthElements;
            const float* row_end = row_start + InputWidthElements;

            for (size_t kw = 0; kw < KernelWidth; kw++) {
                const float* input_ptr = Input + output_idx * StrideWidthElements +
                                         kh * DilatedInputWidthElements + kw * DilationWidthElements;

                MLAS_FLOAT32X4 InputVector;

                if (input_ptr >= row_start && (input_ptr + BlockSize) <= row_end) {
                    InputVector = MlasLoadFloat32x4(input_ptr);
                } else {
                    std::vector<float> values(BlockSize);
                    for (size_t i = 0; i < BlockSize; i++) {
                        const float* element_ptr = input_ptr + i;
                        if (element_ptr >= row_start && element_ptr < row_end) {
                            values[i] = *element_ptr;
                        } else {
                            values[i] = MaxPaddingValue;
                        }
                    }
                    InputVector = MlasLoadFloat32x4(values.data());
                }

                MaxVector = MlasMaximumFloat32x4(MaxVector, InputVector);
            }
        }

        MlasStoreFloat32x4(&Output[output_idx * BlockSize], MaxVector);
    }
}

static void
MlasPoolAverageFloatKernelNeonImpl(
    const float* Input,
    float* Output,
    size_t StrideWidth,
    size_t DilationWidth,
    size_t ActualKernelSize,
    size_t KernelHeight,
    size_t KernelWidth,
    const float* InputBase,
    size_t InputWidth,
    size_t DilatedInputWidth,
    size_t OutputCountLeftPad,
    size_t OutputCount,
    size_t OutputCountRightPad,
    bool ExcludePad
)
{
    const size_t BlockSize = MlasNchwcGetBlockSize();
    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t InputWidthElements = InputWidth / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);
    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;

    const MLAS_FLOAT32X4 ZeroVector = MlasZeroFloat32x4();

    for (size_t output_idx = 0; output_idx < TotalOutputCount; output_idx++) {
        MLAS_FLOAT32X4 SumVector = ZeroVector;

        std::vector<uint32_t> valid_count;
        if (ExcludePad) {
            valid_count.resize(BlockSize, 0);
        }

        for (size_t kh = 0; kh < KernelHeight; kh++) {
            const float* row_start = InputBase + kh * DilatedInputWidthElements;
            const float* row_end = row_start + InputWidthElements;

            for (size_t kw = 0; kw < KernelWidth; kw++) {
                const float* input_ptr = Input + output_idx * StrideWidthElements +
                                         kh * DilatedInputWidthElements + kw * DilationWidthElements;

                MLAS_FLOAT32X4 InputVector;

                if (input_ptr >= row_start && (input_ptr + BlockSize) <= row_end) {
                    InputVector = MlasLoadFloat32x4(input_ptr);

                    if (ExcludePad) {
                        for (size_t i = 0; i < BlockSize; i++) {
                            valid_count[i]++;
                        }
                    }
                } else {
                    std::vector<float> values(BlockSize);
                    for (size_t i = 0; i < BlockSize; i++) {
                        const float* element_ptr = input_ptr + i;
                        if (element_ptr >= row_start && element_ptr < row_end) {
                            values[i] = *element_ptr;
                            if (ExcludePad) {
                                valid_count[i]++;
                            }
                        } else {
                            values[i] = 0.0f;
                        }
                    }
                    InputVector = MlasLoadFloat32x4(values.data());
                }

                SumVector = MlasAddFloat32x4(SumVector, InputVector);
            }
        }

        MLAS_FLOAT32X4 ResultVector;
        if (ExcludePad) {
            std::vector<float> results(BlockSize);
            MlasStoreFloat32x4(results.data(), SumVector);

            for (size_t i = 0; i < BlockSize; i++) {
                results[i] = results[i] / static_cast<float>(valid_count[i]);
            }

            ResultVector = MlasLoadFloat32x4(results.data());
        } else {
            const float KernelSize = static_cast<float>(ActualKernelSize);
            const MLAS_FLOAT32X4 KernelSizeVector = MlasBroadcastFloat32x4(KernelSize);
            ResultVector = MlasDivideFloat32x4(SumVector, KernelSizeVector);
        }

        MlasStoreFloat32x4(&Output[output_idx * BlockSize], ResultVector);
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
    MLAS_UNREFERENCED_PARAMETER(InputStride);

    MlasPoolAverageFloatKernelNeonImpl(
        Input, Output, StrideWidth, DilationWidth, ActualKernelSize,
        KernelHeight, KernelWidth, InputBase, InputWidth, DilatedInputWidth,
        OutputCountLeftPad, OutputCount, OutputCountRightPad,
        true  // ExcludePad = true
    );
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

    MlasPoolAverageFloatKernelNeonImpl(
        Input, Output, StrideWidth, DilationWidth, ActualKernelSize,
        KernelHeight, KernelWidth, InputBase, InputWidth, DilatedInputWidth,
        OutputCountLeftPad, OutputCount, OutputCountRightPad,
        false  // ExcludePad = false
    );
}

#endif  // __aarch64__ || _M_ARM64