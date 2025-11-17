/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sconv.h

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
    const float* input_base,
    size_t offset,
    bool is_main_region,
    const float* InputBase,
    size_t kh,
    size_t DilatedInputWidthElements,
    size_t InputWidthElements
)
{
    if (is_main_region) {
        return MlasLoadFloat32x4(input_base + offset);
    } else {
        float input_values[4];
        const float* row_start = InputBase + kh * DilatedInputWidthElements;
        const float* row_end   = row_start + InputWidthElements;

        for (size_t i = 0; i < 4; ++i) {
            const float* elem = input_base + offset + i;
            input_values[i] = (elem >= row_start && elem < row_end) ? *elem : 0.0f;
        }

        return MlasLoadFloat32x4(input_values);
    }
}

#endif
