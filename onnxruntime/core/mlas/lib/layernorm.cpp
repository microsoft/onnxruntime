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

#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_IX86)
    //
    // Skip the AVX2 kernel for short rows where it cannot win.
    //
    // LayerNorm performs vector work from 8 elements onward. RMSNorm needs
    // at least 16 elements to recover its additional setup costs.
    //
    // This threshold is x86-specific. Other platforms (e.g. RISC-V RVV)
    // use variable-length vectors and handle short rows natively, so they
    // must not be gated here.
    //
    // Keep in sync: test/mlas/unittest/test_layernorm.cpp kKernelDispatchThreshold.
    //
    //
    const size_t dispatch_threshold = Simplified ? 16 : 8;
    if (NormSize < dispatch_threshold) {
        return false;
    }
#endif

    kernel(Input, Scale, Bias, Output, MeanOut, InvStdDevOut, NormSize, Epsilon, Simplified);
    return true;
}
