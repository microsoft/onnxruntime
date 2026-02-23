/*++

Module Name:

    SconvNchwcKernelNeon.S

Abstract:

    Experimental AArch64 NEON entry point for NCHWc activations + NCHWc
    weights convolution micro-kernel work.

    This file is intentionally separate from SconvKernelNeon.S so the NCHW
    path remains unchanged while NCHWc assembly work iterates independently.

    Current behavior is a functional shim that tail-calls the existing
    C++ NEON NCHWc kernel implementation.

--*/

#include "asmmacro.h"

        .text

// Prototype matches MLAS_CONV_FLOAT_KERNEL.
//
// void
// MlasConvNchwcFloatKernelNeonAsmExperimental(
//     const float* Input,
//     const float* Filter,
//     float* Output,
//     size_t StrideWidth,
//     size_t DilationWidth,
//     size_t FilterCount,
//     size_t InputStride,
//     size_t FilterStride,
//     size_t OutputStride,
//     size_t KernelHeight,
//     size_t KernelWidth,
//     const float* InputBase,
//     size_t InputWidth,
//     size_t DilatedInputWidth,
//     size_t OutputCountLeftPad,
//     size_t OutputCount,
//     size_t OutputCountRightPad,
//     const float* Bias,
//     unsigned KernelFlags);
//

        FUNCTION_ENTRY MlasConvNchwcFloatKernelNeonAsmExperimental

        // Temporary shim:
        // tail-call the existing NCHWc NEON C++ implementation.
        b       MlasConvNchwcFloatKernelNeon

        .end
