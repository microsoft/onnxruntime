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

    This routine implements the vector dot product routine.

Arguments:

    A - Supplies the A input vector of M length.

    TransB - Supplies the transposed B input vector of M * N length.

    C - Supplies the C output vector of N length.

    M - Supplies the number of values in the A vector.

    N - Supplies the number of values in the C vector.

Return Value:

    None.

--*/
#if defined(MLAS_TARGET_AMD64)
    MlasPlatform.VectorDotProductF32Kernel(A, B, C, M, N);
#else
    MlasVectorDotProductF32Kernel(A, B, C, M, N);
#endif
}


