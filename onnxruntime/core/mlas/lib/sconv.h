/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sconv.h

Abstract:

    This module contains the private data structures and procedure prototypes
    for the single precision convolution operation.

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
// Define the convolution kernel flags.
//

#define MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT 0x00000001
#define MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION 0x00000002
#define MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION 0x00000004
#define MLAS_CONV_KERNEL_FLAG_OTHER_ACTIVATION 0x00000008