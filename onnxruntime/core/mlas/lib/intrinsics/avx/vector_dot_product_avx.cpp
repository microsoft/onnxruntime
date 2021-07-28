/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm_kernel_avx2.cpp

Abstract:

    This module implements QGEMM kernels for avx2.

--*/

#include "mlasi.h"


#pragma warning(disable: 4100)
// This needs to move into the AVX directory I think.
void
MLASCALL
MlasVectorDotProductF32KernelAVX(
    const float* A,
    const float* B,
    float* C,
    size_t M,
    size_t N
    )
{
    // TODO - this needs to move into the folder.
    // TODO(kreeger): Handle size steps for |N|.
}
#pragma warning(default: 4100)
