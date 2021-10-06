/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    convsym.cpp

Abstract:

    This module implements the symmetric quantized integer convolution
    operation.

--*/

#include "mlasi.h"

//
//
//

#define MLAS_CONV_SYM_FLAG_INPUT_DIRECT             0x00000001
#define MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE        0x00000002

//
//
//

struct MLAS_CONV_SYM_POST_PROCESS_PARAMS {
    const int32_t* Bias;
    const float* Scale;
    float MinimumValue;
    float MaximumValue;
    int32_t OutputZeroPoint;
};

//
// Define the prototypes of the platform optimized routines.
//

typedef
void
(MLASCALL MLAS_CONV_SYM_KERNEL)(
    const void* Input,
    const void* Filter,
    uint8_t* Output,
    size_t KernelSize,
    size_t InputChannels,
    size_t OutputChannels,
    unsigned ChannelCount,
    unsigned OutputCount,
    const struct MLAS_CONV_SYM_POST_PROCESS_PARAMS* PostProcessParams,
    unsigned KernelFlags
    );

typedef
void
(MLASCALL MLAS_CONV_SYM_DEPTHWISE_KERNEL)(
    const void* Input,
    const void* Filter,
    uint8_t* Output,
    size_t KernelSize,
    size_t Channels,
    size_t ChannelOffset,
    unsigned ChannelCount,
    unsigned OutputCount,
    const struct MLAS_CONV_SYM_POST_PROCESS_PARAMS* PostProcessParams,
    unsigned KernelFlags
    );

struct MLAS_CONV_SYM_DISPATCH {
    MLAS_CONV_SYM_KERNEL* Kernel;
    MLAS_CONV_SYM_DEPTHWISE_KERNEL* DepthwiseKernel;
    uint8_t KernelChannelCount;
    uint8_t KernelOutputCount;
    uint8_t KernelDepthwiseChannelCount;
    uint8_t KernelDepthwiseOutputCount;
};