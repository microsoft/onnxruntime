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
    MLAS_UNREFERENCED_PARAMETER(Input);
    MLAS_UNREFERENCED_PARAMETER(Output);
    MLAS_UNREFERENCED_PARAMETER(StrideWidth);
    MLAS_UNREFERENCED_PARAMETER(DilationWidth);
    MLAS_UNREFERENCED_PARAMETER(InputStride);
    MLAS_UNREFERENCED_PARAMETER(ActualKernelSize);
    MLAS_UNREFERENCED_PARAMETER(KernelHeight);
    MLAS_UNREFERENCED_PARAMETER(KernelWidth);
    MLAS_UNREFERENCED_PARAMETER(InputBase);
    MLAS_UNREFERENCED_PARAMETER(InputWidth);
    MLAS_UNREFERENCED_PARAMETER(DilatedInputWidth);
    MLAS_UNREFERENCED_PARAMETER(OutputCountLeftPad);
    MLAS_UNREFERENCED_PARAMETER(OutputCount);
    MLAS_UNREFERENCED_PARAMETER(OutputCountRightPad);
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
    MLAS_UNREFERENCED_PARAMETER(Input);
    MLAS_UNREFERENCED_PARAMETER(Output);
    MLAS_UNREFERENCED_PARAMETER(StrideWidth);
    MLAS_UNREFERENCED_PARAMETER(DilationWidth);
    MLAS_UNREFERENCED_PARAMETER(InputStride);
    MLAS_UNREFERENCED_PARAMETER(ActualKernelSize);
    MLAS_UNREFERENCED_PARAMETER(KernelHeight);
    MLAS_UNREFERENCED_PARAMETER(KernelWidth);
    MLAS_UNREFERENCED_PARAMETER(InputBase);
    MLAS_UNREFERENCED_PARAMETER(InputWidth);
    MLAS_UNREFERENCED_PARAMETER(DilatedInputWidth);
    MLAS_UNREFERENCED_PARAMETER(OutputCountLeftPad);
    MLAS_UNREFERENCED_PARAMETER(OutputCount);
    MLAS_UNREFERENCED_PARAMETER(OutputCountRightPad);
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
    MLAS_UNREFERENCED_PARAMETER(Input);
    MLAS_UNREFERENCED_PARAMETER(Output);
    MLAS_UNREFERENCED_PARAMETER(StrideWidth);
    MLAS_UNREFERENCED_PARAMETER(DilationWidth);
    MLAS_UNREFERENCED_PARAMETER(InputStride);
    MLAS_UNREFERENCED_PARAMETER(ActualKernelSize);
    MLAS_UNREFERENCED_PARAMETER(KernelHeight);
    MLAS_UNREFERENCED_PARAMETER(KernelWidth);
    MLAS_UNREFERENCED_PARAMETER(InputBase);
    MLAS_UNREFERENCED_PARAMETER(InputWidth);
    MLAS_UNREFERENCED_PARAMETER(DilatedInputWidth);
    MLAS_UNREFERENCED_PARAMETER(OutputCountLeftPad);
    MLAS_UNREFERENCED_PARAMETER(OutputCount);
    MLAS_UNREFERENCED_PARAMETER(OutputCountRightPad);
}

#endif  // __aarch64__ || _M_ARM64