/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    DgemmKernelPower.cpp

Abstract:

    This module implements the kernels for the single precision matrix/matrix
    multiply operation (DGEMM).

--*/

#include "mlasi.h"

//
// Templates to ensure that a loop is unrolled.
//

template<size_t Count, size_t Index>
struct MlasLoopUnrollStep
{
    template<typename IterationType, typename... IterationArgs>
    MLAS_FORCEINLINE
    static
    void
    Step(
        IterationArgs&&... Arguments
        )
    {
        IterationType::template Iteration<Count, Index>(Arguments...);
        MlasLoopUnrollStep<Count, Index + 1>::template Step<IterationType>(Arguments...);
    }
};

template<size_t Count>
struct MlasLoopUnrollStep<Count, Count>
{
    template<typename IterationType, typename... IterationArgs>
    MLAS_FORCEINLINE
    static
    void
    Step(
        IterationArgs&&...
        )
    {
        // Terminate the loop.
    }
};

template<size_t Count, typename IteratorType>
struct MlasLoopUnroll
{
    template<typename... IterationArgs>
    MLAS_FORCEINLINE
    void
    operator()(
        IterationArgs&&... Arguments
        )
    {
        MlasLoopUnrollStep<Count, 0>::template Step<IteratorType>(Arguments...);
    }
};

//
// Templates used with loop unrolling to perform an action on one row of the
// output.
//

struct MlasDgemmZeroAccumulators
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOAT64X2 Accumulators[RowCount][4]
        )
    {
        Accumulators[Row][0] = MlasZeroFloat64x2();
        Accumulators[Row][1] = MlasZeroFloat64x2();
        Accumulators[Row][2] = MlasZeroFloat64x2();
        Accumulators[Row][3] = MlasZeroFloat64x2();
    }
};

struct MlasDgemmLoadAElements
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOAT64X2 AElements[RowCount],
        const double* A,
        size_t lda
        )
    {
        AElements[Row] = MlasLoadFloat64x2(A + Row * lda);
    }
};

struct MlasDgemmBroadcastAElements
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOAT64X2 ABroadcast[RowCount],
        const double* A,
        size_t lda
        )
    {
        ABroadcast[Row] = MlasBroadcastFloat64x2(A + Row * lda);
    }
};

template<unsigned Lane>
struct MlasDgemmSplatAElements
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOAT64X2 AElements[RowCount],
        MLAS_FLOAT64X2 ABroadcast[RowCount]
        )
    {
        ABroadcast[Row] = vec_splat(AElements[Row], Lane);
    }
};

struct MlasDgemmMultiplyAddRow
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOAT64X2 Accumulators[RowCount][4],
        MLAS_FLOAT64X2 ABroadcast[RowCount],
        MLAS_FLOAT64X2 BElements[4]
        )
    {
        Accumulators[Row][0] = MlasMultiplyAddFloat64x2(ABroadcast[Row], BElements[0], Accumulators[Row][0]);
        Accumulators[Row][1] = MlasMultiplyAddFloat64x2(ABroadcast[Row], BElements[1], Accumulators[Row][1]);
        Accumulators[Row][2] = MlasMultiplyAddFloat64x2(ABroadcast[Row], BElements[2], Accumulators[Row][2]);
        Accumulators[Row][3] = MlasMultiplyAddFloat64x2(ABroadcast[Row], BElements[3], Accumulators[Row][3]);
    }
};

template<size_t RowCount>
MLAS_FORCEINLINE
void
MlasDgemmComputeBlock(
    MLAS_FLOAT64X2 Accumulators[RowCount][4],
    MLAS_FLOAT64X2 ABroadcast[RowCount],
    const double* B
    )
{
    MLAS_FLOAT64X2 BElements[4];

    BElements[0] = MlasLoadFloat64x2(B);
    BElements[1] = MlasLoadFloat64x2(B + 2);
    BElements[2] = MlasLoadFloat64x2(B + 4);
    BElements[3] = MlasLoadFloat64x2(B + 6);

    MlasLoopUnroll<RowCount, MlasDgemmMultiplyAddRow>()(Accumulators, ABroadcast, BElements);
}

struct MlasDgemmMultiplyAlphaRow
{
    template<size_t Count, size_t Index>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOAT64X2 Accumulators[4],
        MLAS_FLOAT64X2 AlphaBroadcast
        )
    {
        Accumulators[Index] = MlasMultiplyFloat64x2(Accumulators[Index], AlphaBroadcast);
    }
};

struct MlasDgemmMultiplyAlphaAddRow
{
    template<size_t Count, size_t Index>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOAT64X2 Accumulators[4],
        MLAS_FLOAT64X2 AlphaBroadcast,
        const double* C
        )
    {
        Accumulators[Index] = MlasMultiplyAddFloat64x2(Accumulators[Index],
            AlphaBroadcast, MlasLoadFloat64x2(C + Index * 2));
    }
};

struct MlasDgemmStoreRow
{
    template<size_t Count, size_t Index>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOAT64X2 Accumulators[4],
        double* C
        )
    {
        MlasStoreFloat64x2(C + Index * 2, Accumulators[Index]);
    }
};

template<size_t VectorCount>
struct MlasDgemmStoreVector
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOAT64X2 Accumulators[RowCount][4],
        double* C,
        size_t ldc,
        MLAS_FLOAT64X2 AlphaBroadcast,
        bool ZeroMode
        )
    {
        double* c = C + Row * ldc;
        if (ZeroMode) {
            MlasLoopUnroll<VectorCount, MlasDgemmMultiplyAlphaRow>()(Accumulators[Row], AlphaBroadcast);
        } else {
            MlasLoopUnroll<VectorCount, MlasDgemmMultiplyAlphaAddRow>()(Accumulators[Row], AlphaBroadcast, c);
        }
        MlasLoopUnroll<VectorCount, MlasDgemmStoreRow>()(Accumulators[Row], c);

        //
        // Shift down any unaligned elements to the bottom for further processing.
        //

        if (VectorCount < 4) {
            Accumulators[Row][0] = Accumulators[Row][VectorCount];
        }
    }
};

struct MlasDgemmMultiplyAlphaTrailing
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOAT64X2 Accumulators[RowCount][4],
        MLAS_FLOAT64X2 AlphaBroadcast
        )
    {
        Accumulators[Row][0] = MlasMultiplyFloat64x2(Accumulators[Row][0], AlphaBroadcast);
    }
};

template<unsigned Lane>
struct MlasDgemmStoreScalar
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOAT64X2 Accumulators[RowCount][4],
        double* C,
        size_t ldc,
        bool ZeroMode
        )
    {
        double* c = C + Row * ldc + Lane;
        double Value = MlasExtractLaneFloat64x2<Lane>(Accumulators[Row][0]);

        if (!ZeroMode) {
            Value += *c;
        }

        *c = Value;
    }
};

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

        MlasLoopUnroll<RowCount, MlasDgemmZeroAccumulators>()(Accumulators);

        //
        // Compute the output block.
        //
        while (k >= 2) {

            MlasLoopUnroll<RowCount, MlasDgemmLoadAElements>()(AElements, a, lda);

            MlasLoopUnroll<RowCount, MlasDgemmSplatAElements<0>>()(AElements, ABroadcast);
            MlasDgemmComputeBlock<RowCount>(Accumulators, ABroadcast, B);

            MlasLoopUnroll<RowCount, MlasDgemmSplatAElements<1>>()(AElements, ABroadcast);
            MlasDgemmComputeBlock<RowCount>(Accumulators, ABroadcast, B + 8);

            a += 2;
            B += 8 * 2;
            k -= 2;
        }
        if (k > 0) {

            MlasLoopUnroll<RowCount, MlasDgemmBroadcastAElements>()(ABroadcast, a, lda);
            MlasDgemmComputeBlock<RowCount>(Accumulators, ABroadcast, B);

            a += 1;
            B += 8;
            k -= 1;
        }

        if (CountN >= 8) {

            //
            // Store the entire output block.
            //

            MlasLoopUnroll<RowCount, MlasDgemmStoreVector<4>>()(Accumulators, C, ldc, AlphaBroadcast, ZeroMode);

        } else {

            //
            // Store the partial output block.
            //

            //
            if (CountN >= 6) {
                MlasLoopUnroll<RowCount, MlasDgemmStoreVector<3>>()(Accumulators, C, ldc, AlphaBroadcast, ZeroMode);
            } else if (CountN >= 4) {
                MlasLoopUnroll<RowCount, MlasDgemmStoreVector<2>>()(Accumulators, C, ldc, AlphaBroadcast, ZeroMode);
            } else if (CountN >= 2) {
                MlasLoopUnroll<RowCount, MlasDgemmStoreVector<1>>()(Accumulators, C, ldc, AlphaBroadcast, ZeroMode);
            }
            //
            // Store the remaining unaligned columns.
            //
            C += (CountN & ~1);
            CountN &= 1;

            if (CountN > 0) {

                MlasLoopUnroll<RowCount, MlasDgemmMultiplyAlphaTrailing>()(Accumulators, AlphaBroadcast);

                MlasLoopUnroll<RowCount, MlasDgemmStoreScalar<0>>()(Accumulators, C, ldc, ZeroMode);
            }

            break;
        }

        C += 8;
        CountN -= 8;

    } while (CountN > 0);

    return RowCount;
}

