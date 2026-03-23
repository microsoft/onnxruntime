/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    gelu.cpp

Abstract:

    This module implements routines to compute the exact Gelu function.

--*/

#include "mlasi.h"

namespace {

constexpr float kInvSqrt2 = 0.70710678118654752440f;

}  // namespace


void
MLASCALL
MlasGeluErfKernel(
    const float* Input,
    float* Output,
    size_t N
    )
{
    // This kernel is not buffer alias safe because it is implemented in
    // multiple passes: first scale Input into Output, then apply erf in place,
    // and finally combine that intermediate with the original Input values.
    // Callers must guarantee that Input and Output do not overlap (see mlas.h for aliasing requirements).
    for (size_t i = 0; i < N; ++i) {
        Output[i] = Input[i] * kInvSqrt2;
    }

    MlasComputeErf(Output, Output, N);

    for (size_t i = 0; i < N; ++i) {
        Output[i] = 0.5f * Input[i] * (Output[i] + 1.0f);
    }
}

void
MLASCALL
MlasComputeGeluErf(
    const float* Input,
    float* Output,
    size_t N
    )
{
#if defined(MLAS_TARGET_AMD64)
    // TODO: Add an intermediate fused AVX2/FMA3 GELU(erf) path on AMD64.
    // Today the dispatch jumps from the generic multi-pass implementation to
    // AVX512F, so non-AVX512 x64 machines fall back to the generic kernel.
    GetMlasPlatform().GeluErfKernelRoutine(Input, Output, N);
#else
    MlasGeluErfKernel(Input, Output, N);
#endif
}
