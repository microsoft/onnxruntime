/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    DgemmKernelpower.h

Abstract:

    This module implements the kernels for the double precision matrix/matrix
    multiply operation (DGEMM).

--*/

#include "FgemmKernelpower.h"

template<size_t RowCount>
MLAS_FORCEINLINE
size_t
MlasDgemmProcessCount(
    const double* A,
    const double* B,
    double* C,
    size_t CountK,
    size_t CountN,
    size_t lda,
    size_t ldc,
    MLAS_FLOAT64X2 AlphaBroadcast,
    bool ZeroMode
    )
{
    do {

        const double* a = A;
        size_t k = CountK;

        MLAS_FLOAT64X2 Accumulators[RowCount][4];
        MLAS_FLOAT64X2 AElements[RowCount];
        MLAS_FLOAT64X2 ABroadcast[RowCount];

        //
        // Clear the block accumulators.
        //

        MlasLoopUnroll<RowCount, MlasFgemmZeroAccumulators>()(Accumulators);

        //
        // Compute the output block.
        //
        while (k >= 2) {

            MlasLoopUnroll<RowCount, MlasFgemmLoadAElements>()(AElements, a, lda);

            MlasLoopUnroll<RowCount, MlasFgemmSplatAElements<0>>()(AElements, ABroadcast);
            MlasFgemmComputeBlock<RowCount>(Accumulators, ABroadcast, B);

            MlasLoopUnroll<RowCount, MlasFgemmSplatAElements<1>>()(AElements, ABroadcast);
            MlasFgemmComputeBlock<RowCount>(Accumulators, ABroadcast, B + 8);

            a += 2;
            B += 8 * 2;
            k -= 2;
        }
        if (k > 0) {

            MlasLoopUnroll<RowCount, MlasFgemmBroadcastAElements>()(ABroadcast, a, lda);
            MlasFgemmComputeBlock<RowCount>(Accumulators, ABroadcast, B);

            a += 1;
            B += 8;
            k -= 1;
        }

        if (CountN >= 8) {

            //
            // Store the entire output block.
            //

            MlasLoopUnroll<RowCount, MlasFgemmStoreVector<4>>()(Accumulators, C, ldc, AlphaBroadcast, ZeroMode);

        } else {

            //
            // Store the partial output block.
            //

            //
            if (CountN >= 6) {
                MlasLoopUnroll<RowCount, MlasFgemmStoreVector<3>>()(Accumulators, C, ldc, AlphaBroadcast, ZeroMode);
            } else if (CountN >= 4) {
                MlasLoopUnroll<RowCount, MlasFgemmStoreVector<2>>()(Accumulators, C, ldc, AlphaBroadcast, ZeroMode);
            } else if (CountN >= 2) {
                MlasLoopUnroll<RowCount, MlasFgemmStoreVector<1>>()(Accumulators, C, ldc, AlphaBroadcast, ZeroMode);
            }
            //
            // Store the remaining unaligned columns.
            //
            C += (CountN & ~1);
            CountN &= 1;

            if (CountN > 0) {

                MlasLoopUnroll<RowCount, MlasFgemmMultiplyAlphaTrailing>()(Accumulators, AlphaBroadcast);

                MlasLoopUnroll<RowCount, MlasFgemmStoreScalar<0>>()(Accumulators, C, ldc, ZeroMode);
            }

            break;
        }

        C += 8;
        CountN -= 8;

    } while (CountN > 0);

    return RowCount;
}

