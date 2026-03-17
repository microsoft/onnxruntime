/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    silu.cpp

Abstract:

    This module implements routines to compute the SiLU function.

--*/

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
    // The caller should guarantee Input and Output do not overlap.
    // The current CPU EP kernel where we call this from guarantees that.
    MlasComputeLogistic(Input, Output, N);
    MlasEltwiseMul<float>(Input, Output, Output, N);
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
