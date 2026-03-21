/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    silu.cpp

Abstract:

    This module implements routines to compute the SiLU function.

--*/

#include <limits>

#include "mlasi.h"

void
MLASCALL
MlasSiluKernel(
    const float* Input,
    float* Output,
    size_t N
    )
{
    // This kernel is not buffer alias safe, as the computation is not elementwise.
    // Callers must guarantee that Input and Output do not overlap (see mlas.h for aliasing requirements).
    const float PositiveInfinity = std::numeric_limits<float>::infinity();
    const float NegativeInfinity = -std::numeric_limits<float>::infinity();

    MlasComputeLogistic(Input, Output, N);
    MlasEltwiseMul<float>(Input, Output, Output, N);

    for (size_t i = 0; i < N; ++i) {
        if (Input[i] == PositiveInfinity) {
            Output[i] = PositiveInfinity;
        } else if (Input[i] == NegativeInfinity) {
            Output[i] = -0.0f;
        }
    }
}

void
MLASCALL
MlasComputeSilu(
    const float* Input,
    float* Output,
    size_t N
    )
{
#if defined(MLAS_TARGET_AMD64)
    GetMlasPlatform().SiluKernelRoutine(Input, Output, N);
#else
    MlasSiluKernel(Input, Output, N);
#endif
}
