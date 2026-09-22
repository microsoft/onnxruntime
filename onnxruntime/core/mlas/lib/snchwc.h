/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    snchwc.h

Abstract:

    This module contains private definitions shared by the single precision
    NCHWc convolution implementations.

--*/

#pragma once

//
// Define the convolution kernel flags.
//

#define MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT     0x00000001
#define MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION         0x00000002
#define MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION       0x00000004
#define MLAS_CONV_KERNEL_FLAG_OTHER_ACTIVATION      0x00000008
#define MLAS_CONV_KERNEL_MLAS_ARM_USE_KLEIDIAI      0x00000010

#if defined(MLAS_TARGET_AMD64)

//
// Returns the depthwise kernel to use in place of the platform kernel. The
// sliding window AVX-512 kernel is bitwise identical to
// MlasConvDepthwiseFloatKernelAvx512F and forwards unsupported geometries to
// it.
//

inline
MLAS_CONV_DEPTHWISE_FLOAT_KERNEL*
MlasNchwcSelectDepthwiseKernel(
    MLAS_CONV_DEPTHWISE_FLOAT_KERNEL* PlatformKernel,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
    )
{
    if (PlatformKernel == MlasConvDepthwiseFloatKernelAvx512F &&
        (BackendKernelSelectorConfig == nullptr ||
         BackendKernelSelectorConfig->nchwc_depthwise_sliding_kernel)) {
        return MlasConvDepthwiseFloatKernelAvx512FSliding;
    }

    return PlatformKernel;
}

#endif
