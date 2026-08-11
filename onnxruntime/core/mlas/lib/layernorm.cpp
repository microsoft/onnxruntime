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

    //
    // Skip the SIMD kernel for very short rows where it cannot win.
    //
    // Measured on AMD EPYC 9V74 (AVX2/FMA, no AVX-512): for NormSize < 8
    // the AVX2 kernel performs zero 256-bit iterations and falls entirely
    // into its scalar tail, yet still pays vector register setup and
    // horizontal reduction overhead. RMSNorm regresses 5-22% for N=1..7;
    // full LayerNorm regresses 6-29% for N=1..2 (the Welford scalar path's
    // per-element division makes it expensive enough that the AVX2 two-pass
    // tail wins from N >= 3, but below 8 there is no SIMD benefit by
    // definition). The threshold of 8 (== one ymm register width) is the
    // natural boundary: below it, no vectorization is possible.
    //
    if (NormSize < 8) {
        return false;
    }

    kernel(Input, Scale, Bias, Output, MeanOut, InvStdDevOut, NormSize, Epsilon, Simplified);
    return true;
}
