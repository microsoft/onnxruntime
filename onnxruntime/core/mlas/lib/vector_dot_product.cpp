/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    vector_dot_prod.cpp

Abstract:

    This module implements the vector dot product routine.

--*/


#include "mlasi.h"

#include <iostream>

#pragma warning(disable: 4100)

void
MLASCALL
MlasVectorDotProductF32KernelSSE(
    const float* A,
    const float* B,
    float* C,
    size_t M,
    size_t N
    )
{
    // TODO(kreeger): Handle size steps for |N|.
}

void 
MLASCALL
MlasVectorDotProduct(
    const float* A,
    const float* B,
    float* C,
    size_t M, size_t N)
{
    std::cerr << "Hi from MlasVectorDotProduct\n";

    // This thing needs to loop over inputs and call into a kernel!
    //
    // TODO(kreeger): LEFT OFF RIGHT HERE!
    //
}
#pragma warning(default: 4100)
