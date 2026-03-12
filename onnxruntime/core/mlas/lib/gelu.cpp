/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    gelu.cpp

Abstract:

    This module implements routines to compute the exact Gelu function.

--*/

#include <cmath>

#include "mlasi.h"


void
MLASCALL
MlasGeluKernel(
    const float* Input,
    float* Output,
    size_t N
    )
{
    // This kernel is not buffer alias safe, as the computation is not elementwise.
    // The caller should guarantee Input and Output do not overlap.
    // The current CPU EP kernel where we call this from guarantees that.
    for (size_t i = 0; i < N; ++i) {
        Output[i] = Input[i] * static_cast<float>(M_SQRT1_2);
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
    size_t N,
    MLAS_GELU_ERF_MODE Mode
    )
{
#if defined(MLAS_TARGET_AMD64)
    if (Mode == MlasGeluErfModeMinimaxApproximation && GetMlasPlatform().GeluErfMinimaxKernelRoutine != nullptr) {
        GetMlasPlatform().GeluErfMinimaxKernelRoutine(Input, Output, N);
        return;
    }
#endif

#if defined(MLAS_TARGET_AMD64) || defined(MLAS_USE_SVE)
    GetMlasPlatform().GeluKernelRoutine(Input, Output, N);
#else
    MLAS_UNREFERENCED_PARAMETER(Mode);
    MlasGeluKernel(Input, Output, N);
#endif
}
