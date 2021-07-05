/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmKernelPower.cpp

Abstract:

    This module implements the kernels for the single precision matrix/matrix
    multiply operation (SGEMM).

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

struct MlasSgemmZeroAccumulators
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOAT32X4 Accumulators[RowCount][4]
        )
    {
        Accumulators[Row][0] = MlasZeroFloat32x4();
        Accumulators[Row][1] = MlasZeroFloat32x4();
        Accumulators[Row][2] = MlasZeroFloat32x4();
        Accumulators[Row][3] = MlasZeroFloat32x4();
    }
};

struct MlasSgemmLoadAElements
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOAT32X4 AElements[RowCount],
        const float* A,
        size_t lda
        )
    {
        AElements[Row] = MlasLoadFloat32x4(A + Row * lda);
    }
};

struct MlasSgemmBroadcastAElements
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOAT32X4 ABroadcast[RowCount],
        const float* A,
        size_t lda
        )
    {
        ABroadcast[Row] = MlasBroadcastFloat32x4(A + Row * lda);
    }
};

template<unsigned Lane>
struct MlasSgemmSplatAElements
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOAT32X4 AElements[RowCount],
        MLAS_FLOAT32X4 ABroadcast[RowCount]
        )
    {
        ABroadcast[Row] = vec_splat(AElements[Row], Lane);
    }
};

struct MlasSgemmMultiplyAddRow
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOAT32X4 Accumulators[RowCount][4],
        MLAS_FLOAT32X4 ABroadcast[RowCount],
        MLAS_FLOAT32X4 BElements[4]
        )
    {
        Accumulators[Row][0] = MlasMultiplyAddFloat32x4(ABroadcast[Row], BElements[0], Accumulators[Row][0]);
        Accumulators[Row][1] = MlasMultiplyAddFloat32x4(ABroadcast[Row], BElements[1], Accumulators[Row][1]);
        Accumulators[Row][2] = MlasMultiplyAddFloat32x4(ABroadcast[Row], BElements[2], Accumulators[Row][2]);
        Accumulators[Row][3] = MlasMultiplyAddFloat32x4(ABroadcast[Row], BElements[3], Accumulators[Row][3]);
    }
};

template<size_t RowCount>
MLAS_FORCEINLINE
void
MlasSgemmComputeBlock(
    MLAS_FLOAT32X4 Accumulators[RowCount][4],
    MLAS_FLOAT32X4 ABroadcast[RowCount],
    const float* B
    )
{
    MLAS_FLOAT32X4 BElements[4];

    BElements[0] = MlasLoadFloat32x4(B);
    BElements[1] = MlasLoadFloat32x4(B + 4);
    BElements[2] = MlasLoadFloat32x4(B + 8);
    BElements[3] = MlasLoadFloat32x4(B + 12);

    MlasLoopUnroll<RowCount, MlasSgemmMultiplyAddRow>()(Accumulators, ABroadcast, BElements);
}

struct MlasSgemmMultiplyAlphaRow
{
    template<size_t Count, size_t Index>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOAT32X4 Accumulators[4],
        MLAS_FLOAT32X4 AlphaBroadcast
        )
    {
        Accumulators[Index] = MlasMultiplyFloat32x4(Accumulators[Index], AlphaBroadcast);
    }
};

struct MlasSgemmMultiplyAlphaAddRow
{
    template<size_t Count, size_t Index>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOAT32X4 Accumulators[4],
        MLAS_FLOAT32X4 AlphaBroadcast,
        const float* C
        )
    {
        Accumulators[Index] = MlasMultiplyAddFloat32x4(Accumulators[Index],
            AlphaBroadcast, MlasLoadFloat32x4(C + Index * 4));
    }
};

struct MlasSgemmStoreRow
{
    template<size_t Count, size_t Index>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOAT32X4 Accumulators[4],
        float* C
        )
    {
        MlasStoreFloat32x4(C + Index * 4, Accumulators[Index]);
    }
};

template<size_t VectorCount>
struct MlasSgemmStoreVector
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOAT32X4 Accumulators[RowCount][4],
        float* C,
        size_t ldc,
        MLAS_FLOAT32X4 AlphaBroadcast,
        bool ZeroMode
        )
    {
        float* c = C + Row * ldc;

        if (ZeroMode) {
            MlasLoopUnroll<VectorCount, MlasSgemmMultiplyAlphaRow>()(Accumulators[Row], AlphaBroadcast);
        } else {
            MlasLoopUnroll<VectorCount, MlasSgemmMultiplyAlphaAddRow>()(Accumulators[Row], AlphaBroadcast, c);
        }

        MlasLoopUnroll<VectorCount, MlasSgemmStoreRow>()(Accumulators[Row], c);

        //
        // Shift down any unaligned elements to the bottom for further processing.
        //

        if (VectorCount < 4) {
            Accumulators[Row][0] = Accumulators[Row][VectorCount];
        }
    }
};

struct MlasSgemmMultiplyAlphaTrailing
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOAT32X4 Accumulators[RowCount][4],
        MLAS_FLOAT32X4 AlphaBroadcast
        )
    {
        Accumulators[Row][0] = MlasMultiplyFloat32x4(Accumulators[Row][0], AlphaBroadcast);
    }
};

template<unsigned Lane>
struct MlasSgemmStoreScalar
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOAT32X4 Accumulators[RowCount][4],
        float* C,
        size_t ldc,
        bool ZeroMode
        )
    {
        float* c = C + Row * ldc + Lane;
        float Value = MlasExtractLaneFloat32x4<Lane>(Accumulators[Row][0]);

        if (!ZeroMode) {
            Value += *c;
        }

        *c = Value;
    }
};

template<size_t RowCount>
MLAS_FORCEINLINE
size_t
MlasSgemmProcessCount(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountN,
    size_t lda,
    size_t ldc,
    MLAS_FLOAT32X4 AlphaBroadcast,
    bool ZeroMode
    )
{
    do {

        const float* a = A;
        size_t k = CountK;

        MLAS_FLOAT32X4 Accumulators[RowCount][4];
        MLAS_FLOAT32X4 AElements[RowCount];
        MLAS_FLOAT32X4 ABroadcast[RowCount];

        //
        // Clear the block accumulators.
        //

        MlasLoopUnroll<RowCount, MlasSgemmZeroAccumulators>()(Accumulators);

        //
        // Compute the output block.
        //

        while (k >= 4) {

            MlasLoopUnroll<RowCount, MlasSgemmLoadAElements>()(AElements, a, lda);

            MlasLoopUnroll<RowCount, MlasSgemmSplatAElements<0>>()(AElements, ABroadcast);
            MlasSgemmComputeBlock<RowCount>(Accumulators, ABroadcast, B);

            MlasLoopUnroll<RowCount, MlasSgemmSplatAElements<1>>()(AElements, ABroadcast);
            MlasSgemmComputeBlock<RowCount>(Accumulators, ABroadcast, B + 16);

            MlasLoopUnroll<RowCount, MlasSgemmSplatAElements<2>>()(AElements, ABroadcast);
            MlasSgemmComputeBlock<RowCount>(Accumulators, ABroadcast, B + 32);

            MlasLoopUnroll<RowCount, MlasSgemmSplatAElements<3>>()(AElements, ABroadcast);
            MlasSgemmComputeBlock<RowCount>(Accumulators, ABroadcast, B + 48);

            a += 4;
            B += 16 * 4;
            k -= 4;
        }

        while (k > 0) {

            MlasLoopUnroll<RowCount, MlasSgemmBroadcastAElements>()(ABroadcast, a, lda);
            MlasSgemmComputeBlock<RowCount>(Accumulators, ABroadcast, B);

            a += 1;
            B += 16;
            k -= 1;
        }

        if (CountN >= 16) {

            //
            // Store the entire output block.
            //

            MlasLoopUnroll<RowCount, MlasSgemmStoreVector<4>>()(Accumulators, C, ldc, AlphaBroadcast, ZeroMode);

        } else {

            //
            // Store the partial output block.
            //

            if (CountN >= 12) {
                MlasLoopUnroll<RowCount, MlasSgemmStoreVector<3>>()(Accumulators, C, ldc, AlphaBroadcast, ZeroMode);
            } else if (CountN >= 8) {
                MlasLoopUnroll<RowCount, MlasSgemmStoreVector<2>>()(Accumulators, C, ldc, AlphaBroadcast, ZeroMode);
            } else if (CountN >= 4) {
                MlasLoopUnroll<RowCount, MlasSgemmStoreVector<1>>()(Accumulators, C, ldc, AlphaBroadcast, ZeroMode);
            }

            //
            // Store the remaining unaligned columns.
            //

            C += (CountN & ~3);
            CountN &= 3;

            if (CountN > 0) {

                MlasLoopUnroll<RowCount, MlasSgemmMultiplyAlphaTrailing>()(Accumulators, AlphaBroadcast);

                MlasLoopUnroll<RowCount, MlasSgemmStoreScalar<0>>()(Accumulators, C, ldc, ZeroMode);

                if (CountN >= 2) {
                    MlasLoopUnroll<RowCount, MlasSgemmStoreScalar<1>>()(Accumulators, C, ldc, ZeroMode);
                }

                if (CountN >= 3) {
                    MlasLoopUnroll<RowCount, MlasSgemmStoreScalar<2>>()(Accumulators, C, ldc, ZeroMode);
                }
            }

            break;
        }

        C += 16;
        CountN -= 16;

    } while (CountN > 0);

    return RowCount;
}

