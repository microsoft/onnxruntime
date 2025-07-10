/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    spool.h

Abstract:

    This module contains the private data structures and procedure prototypes
    for the single precision pooling operation.

--*/

#pragma once

#include <cstddef>

//
// Define the calling convention for MLAS functions.
//

#ifndef MLASCALL
#if defined(_WIN32) && !defined(_WIN64)
#define MLASCALL __stdcall
#else
#define MLASCALL
#endif
#endif

//
// Define the prototypes of the NEON convolution kernels.
//

#if defined(__aarch64__) || defined(_M_ARM64)

extern "C" {

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
    );

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
    );

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
    );

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
    );

}

#endif
