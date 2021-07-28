/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    vector_dot_prod.cpp

Abstract:

    This module implements the vector dot product routine.

--*/


#include "mlasi.h"

void
MLASCALL
MlasVectorDotProduct(const float* A, const float* B, float* C, size_t M, size_t N)
{
/*++

Routine Description:

    This routine TODO DOC ME.

Arguments:

    Input - Supplies the input buffer.

    Min - Returns the minimum value of the supplied buffer.

    Max - Returns the maximum value of the supplied buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
#if defined(MLAS_TARGET_AMD64)
    MlasPlatform.VectorDotProductF32Kernel(A, B, C, M, N);
#else
    VectorDotProductF32Kernel(A, B, C, M, N);
#endif
}


