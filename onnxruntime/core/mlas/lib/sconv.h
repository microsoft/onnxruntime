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

#define MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT 0x00000001
#define MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION 0x00000002
#define MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION 0x00000004
#define MLAS_CONV_KERNEL_FLAG_OTHER_ACTIVATION 0x00000008