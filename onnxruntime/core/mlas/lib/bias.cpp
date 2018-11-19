/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    bias.cpp

Abstract:

    This module implements the bias add operation.

--*/

#include "mlasi.h"

void
MLASCALL
MlasBiasAdd(
    const float* Bias,
    size_t M,
    float* Output,
    size_t N,
    size_t ldc
    )
/*++

Routine Description:

    This routine adds a bias vector to the output matrix.

Arguments:

    Bias - Supplies the bias vector.

    M - Supplies the number of elements of the bias vector and the number of
        rows in the output matrix.

    Output - Supplies the output matrix.

    N - Supplies the number of columns of the output matrix.

    ldc - Supplies the number of elements per row of the output matrix.

Return Value:

    None.

--*/
{
    //
    // Step through each row of the output matrix.
    //

    while (M-- > 0) {

        float* buffer = Output;
        size_t n = N;

        float BiasValue = *Bias++;

        if (n >= 4) {

            MLAS_FLOAT32X4 BiasBroadcast = MlasBroadcastFloat32x4(BiasValue);

            do {
                MlasStoreFloat32x4(buffer, MlasAddFloat32x4(BiasBroadcast, MlasLoadFloat32x4(buffer)));
                buffer += 4;
                n -= 4;
            } while (n >= 4);
        }

        while (n > 0) {
            *buffer++ += BiasValue;
            n -= 1;
        }

        Output += ldc;
    }
}
