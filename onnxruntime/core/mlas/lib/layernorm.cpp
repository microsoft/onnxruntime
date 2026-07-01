/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    layernorm.cpp

Abstract:

    This module implements the dispatch for platform-optimized
    LayerNorm/RMSNorm kernels.

--*/

#include "mlasi.h"

bool
    MLASCALL
    MlasLayerNormF32(
        const float* Input,
        const float* Scale,
        const float* Bias,
        float* Output,
        float* MeanOut,
        float* InvStdDevOut,
        size_t NormSize,
        float Epsilon,
        bool Simplified
    )
{
    auto kernel = GetMlasPlatform().LayerNormF32Kernel;
    if (kernel == nullptr) {
        return false;
    }

    kernel(Input, Scale, Bias, Output, MeanOut, InvStdDevOut, NormSize, Epsilon, Simplified);
    return true;
}
