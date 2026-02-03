/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sconv_nchwc_kernel_neon.h

Abstract:

    This module defines convolution kernel flags for configuring convolution
    operations including output accumulation, bias addition, and activations.

--*/

//
// Define the convolution kernel flags.
//

#if defined(MLAS_USE_ARM_NEON_NCHWC)

#define MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT 0x00000001
#define MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION 0x00000002
#define MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION 0x00000004
#define MLAS_CONV_KERNEL_FLAG_OTHER_ACTIVATION 0x00000008

//
// Helper function to load input vector with bounds checking
//
static inline float32x4_t
LoadInputVectorWithBounds(
    const float* ptr,
    const float* row_start,
    const float* row_end
)
{
    if (ptr >= row_start && (ptr + 3) < row_end) {
        return MlasLoadFloat32x4(ptr);
    }
    return MlasBroadcastFloat32x4(0.0f);
}

#endif
