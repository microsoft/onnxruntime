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
    // This kernel is not buffer alias safe because it is implemented in two
    // passes: first compute logistic(Input) into Output, then multiply that
    // intermediate by the original Input values. Callers must guarantee that
    // Input and Output do not overlap (see mlas.h for aliasing requirements).
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
    // TODO: Add an intermediate fused AVX2/FMA3 SiLU path on AMD64. Today the
    // dispatch jumps from the generic two-pass implementation to AVX512F, so
    // non-AVX512 x64 machines fall back to the generic kernel.
    GetMlasPlatform().SiluKernelRoutine(Input, Output, N);
#else
    MlasSiluKernel(Input, Output, N);
#endif
}
