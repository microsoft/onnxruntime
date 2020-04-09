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

template<size_t RowCount, size_t Row, size_t ColumnCount>
struct MLAS_SGEMM_OUTPUT
{
    static
    void
    StoreVector(
        MLAS_FLOAT32X4 Accumulators[RowCount][4],
        float* C,
        size_t ldc,
        MLAS_FLOAT32X4 AlphaBroadcast,
        bool ZeroMode
        );
};

template<size_t RowCount, size_t Row>
struct MLAS_SGEMM_OUTPUT<RowCount, Row, 4>
{
    MLAS_FORCEINLINE
    static
    void
    StoreVector(
        MLAS_FLOAT32X4 Accumulators[RowCount][4],
        float* C,
        size_t ldc,
        MLAS_FLOAT32X4 AlphaBroadcast,
        bool ZeroMode
        )
    {
        if (ZeroMode) {
            Accumulators[Row][0] = MlasMultiplyFloat32x4(Accumulators[Row][0], AlphaBroadcast);
        } else {
            Accumulators[Row][0] = MlasMultiplyAddFloat32x4(Accumulators[Row][0], AlphaBroadcast,
                MlasLoadFloat32x4(C + Row * ldc));
        }

        MlasStoreFloat32x4(C + Row * ldc, Accumulators[Row][0]);

        Accumulators[Row][0] = Accumulators[Row][1];
    }
};

template<size_t RowCount, size_t Row>
struct MLAS_SGEMM_OUTPUT<RowCount, Row, 8>
{
    MLAS_FORCEINLINE
    static
    void
    StoreVector(
        MLAS_FLOAT32X4 Accumulators[RowCount][4],
        float* C,
        size_t ldc,
        MLAS_FLOAT32X4 AlphaBroadcast,
        bool ZeroMode
        )
    {
        if (ZeroMode) {
            Accumulators[Row][0] = MlasMultiplyFloat32x4(Accumulators[Row][0], AlphaBroadcast);
            Accumulators[Row][1] = MlasMultiplyFloat32x4(Accumulators[Row][1], AlphaBroadcast);
        } else {
            Accumulators[Row][0] = MlasMultiplyAddFloat32x4(Accumulators[Row][0], AlphaBroadcast,
                MlasLoadFloat32x4(C + Row * ldc));
            Accumulators[Row][1] = MlasMultiplyAddFloat32x4(Accumulators[Row][1], AlphaBroadcast,
                MlasLoadFloat32x4(C + Row * ldc + 4));
        }

        MlasStoreFloat32x4(C + Row * ldc, Accumulators[Row][0]);
        MlasStoreFloat32x4(C + Row * ldc + 4, Accumulators[Row][1]);

        Accumulators[Row][0] = Accumulators[Row][2];
    }
};

template<size_t RowCount, size_t Row>
struct MLAS_SGEMM_OUTPUT<RowCount, Row, 12>
{
    MLAS_FORCEINLINE
    static
    void
    StoreVector(
        MLAS_FLOAT32X4 Accumulators[RowCount][4],
        float* C,
        size_t ldc,
        MLAS_FLOAT32X4 AlphaBroadcast,
        bool ZeroMode
        )
    {
        if (ZeroMode) {
            Accumulators[Row][0] = MlasMultiplyFloat32x4(Accumulators[Row][0], AlphaBroadcast);
            Accumulators[Row][1] = MlasMultiplyFloat32x4(Accumulators[Row][1], AlphaBroadcast);
            Accumulators[Row][2] = MlasMultiplyFloat32x4(Accumulators[Row][2], AlphaBroadcast);
        } else {
            Accumulators[Row][0] = MlasMultiplyAddFloat32x4(Accumulators[Row][0], AlphaBroadcast,
                MlasLoadFloat32x4(C + Row * ldc));
            Accumulators[Row][1] = MlasMultiplyAddFloat32x4(Accumulators[Row][1], AlphaBroadcast,
                MlasLoadFloat32x4(C + Row * ldc + 4));
            Accumulators[Row][2] = MlasMultiplyAddFloat32x4(Accumulators[Row][2], AlphaBroadcast,
                MlasLoadFloat32x4(C + Row * ldc + 8));
        }

        MlasStoreFloat32x4(C + Row * ldc, Accumulators[Row][0]);
        MlasStoreFloat32x4(C + Row * ldc + 4, Accumulators[Row][1]);
        MlasStoreFloat32x4(C + Row * ldc + 8, Accumulators[Row][2]);

        Accumulators[Row][0] = Accumulators[Row][3];
    }
};

template<size_t RowCount, size_t Row>
struct MLAS_SGEMM_OUTPUT<RowCount, Row, 16>
{
    MLAS_FORCEINLINE
    static
    void
    StoreVector(
        MLAS_FLOAT32X4 Accumulators[RowCount][4],
        float* C,
        size_t ldc,
        MLAS_FLOAT32X4 AlphaBroadcast,
        bool ZeroMode
        )
    {
        if (ZeroMode) {
            Accumulators[Row][0] = MlasMultiplyFloat32x4(Accumulators[Row][0], AlphaBroadcast);
            Accumulators[Row][1] = MlasMultiplyFloat32x4(Accumulators[Row][1], AlphaBroadcast);
            Accumulators[Row][2] = MlasMultiplyFloat32x4(Accumulators[Row][2], AlphaBroadcast);
            Accumulators[Row][3] = MlasMultiplyFloat32x4(Accumulators[Row][3], AlphaBroadcast);
        } else {
            Accumulators[Row][0] = MlasMultiplyAddFloat32x4(Accumulators[Row][0], AlphaBroadcast,
                MlasLoadFloat32x4(C + Row * ldc));
            Accumulators[Row][1] = MlasMultiplyAddFloat32x4(Accumulators[Row][1], AlphaBroadcast,
                MlasLoadFloat32x4(C + Row * ldc + 4));
            Accumulators[Row][2] = MlasMultiplyAddFloat32x4(Accumulators[Row][2], AlphaBroadcast,
                MlasLoadFloat32x4(C + Row * ldc + 8));
            Accumulators[Row][3] = MlasMultiplyAddFloat32x4(Accumulators[Row][3], AlphaBroadcast,
                MlasLoadFloat32x4(C + Row * ldc + 12));
        }

        MlasStoreFloat32x4(C + Row * ldc, Accumulators[Row][0]);
        MlasStoreFloat32x4(C + Row * ldc + 4, Accumulators[Row][1]);
        MlasStoreFloat32x4(C + Row * ldc + 8, Accumulators[Row][2]);
        MlasStoreFloat32x4(C + Row * ldc + 12, Accumulators[Row][3]);
    }
};

template<size_t RowCount, size_t Row>
struct MLAS_SGEMM_ROW_LOOP_STEP
{
    using MLAS_SGEMM_ROW_LOOP_NEXT = MLAS_SGEMM_ROW_LOOP_STEP<RowCount, Row + 1>;

    MLAS_FORCEINLINE
    static
    void
    ZeroAccumulators(
        MLAS_FLOAT32X4 Accumulators[RowCount][4]
        )
    {
        Accumulators[Row][0] = MlasZeroFloat32x4();
        Accumulators[Row][1] = MlasZeroFloat32x4();
        Accumulators[Row][2] = MlasZeroFloat32x4();
        Accumulators[Row][3] = MlasZeroFloat32x4();

        MLAS_SGEMM_ROW_LOOP_NEXT::ZeroAccumulators(Accumulators);
    }

    MLAS_FORCEINLINE
    static
    void
    LoadAElements(
        MLAS_FLOAT32X4 AElements[RowCount],
        const float* A,
        size_t lda
        )
    {
        AElements[Row] = MlasLoadFloat32x4(A + Row * lda);

        MLAS_SGEMM_ROW_LOOP_NEXT::LoadAElements(AElements, A, lda);
    }

    MLAS_FORCEINLINE
    static
    void
    BroadcastAElements(
        MLAS_FLOAT32X4 ABroadcast[RowCount],
        const float* A,
        size_t lda
        )
    {
        ABroadcast[Row] = MlasBroadcastFloat32x4(A + Row * lda);

        MLAS_SGEMM_ROW_LOOP_NEXT::BroadcastAElements(ABroadcast, A, lda);
    }

    template<unsigned Lane>
    MLAS_FORCEINLINE
    static
    void
    BroadcastAElements(
        MLAS_FLOAT32X4 AElements[RowCount],
        MLAS_FLOAT32X4 ABroadcast[RowCount]
        )
    {
        ABroadcast[Row] = vec_splat(AElements[Row], Lane);

        MLAS_SGEMM_ROW_LOOP_NEXT::template BroadcastAElements<Lane>(AElements, ABroadcast);
    }

    MLAS_FORCEINLINE
    static
    void
    MultiplyAdd(
        MLAS_FLOAT32X4 Accumulators[RowCount][4],
        MLAS_FLOAT32X4 ABroadcast[RowCount],
        MLAS_FLOAT32X4 BElements[4]
        )
    {
        Accumulators[Row][0] = MlasMultiplyAddFloat32x4(ABroadcast[Row], BElements[0], Accumulators[Row][0]);
        Accumulators[Row][1] = MlasMultiplyAddFloat32x4(ABroadcast[Row], BElements[1], Accumulators[Row][1]);
        Accumulators[Row][2] = MlasMultiplyAddFloat32x4(ABroadcast[Row], BElements[2], Accumulators[Row][2]);
        Accumulators[Row][3] = MlasMultiplyAddFloat32x4(ABroadcast[Row], BElements[3], Accumulators[Row][3]);

        MLAS_SGEMM_ROW_LOOP_NEXT::MultiplyAdd(Accumulators, ABroadcast, BElements);
    }

    template<unsigned ColumnCount>
    MLAS_FORCEINLINE
    static
    void
    StoreVector(
        MLAS_FLOAT32X4 Accumulators[RowCount][4],
        float* C,
        size_t ldc,
        MLAS_FLOAT32X4 AlphaBroadcast,
        bool ZeroMode
        )
    {
        MLAS_SGEMM_OUTPUT<RowCount, Row, ColumnCount>::StoreVector(Accumulators, C, ldc, AlphaBroadcast, ZeroMode);
        MLAS_SGEMM_ROW_LOOP_NEXT::template StoreVector<ColumnCount>(Accumulators, C, ldc, AlphaBroadcast, ZeroMode);
    }

    MLAS_FORCEINLINE
    static
    void
    MultiplyAlphaTrailing(
        MLAS_FLOAT32X4 Accumulators[RowCount][4],
        MLAS_FLOAT32X4 AlphaBroadcast
        )
    {
        Accumulators[Row][0] = MlasMultiplyFloat32x4(Accumulators[Row][0], AlphaBroadcast);

        MLAS_SGEMM_ROW_LOOP_NEXT::MultiplyAlphaTrailing(Accumulators, AlphaBroadcast);
    }

    template<unsigned Lane>
    MLAS_FORCEINLINE
    static
    void
    StoreFloat(
        MLAS_FLOAT32X4 Accumulators[RowCount][4],
        float* C,
        size_t ldc,
        bool ZeroMode
        )
    {
        float Value = MlasExtractLaneFloat32x4<Lane>(Accumulators[Row][0]);

        if (!ZeroMode) {
            Value += *(C + Row * ldc + Lane);
        }

        *(C + Row * ldc + Lane) = Value;

        MLAS_SGEMM_ROW_LOOP_NEXT::template StoreFloat<Lane>(Accumulators, C, ldc, ZeroMode);
    }
};

template<size_t RowCount>
struct MLAS_SGEMM_ROW_LOOP_STEP<RowCount, RowCount>
{
    MLAS_FORCEINLINE
    static
    void
    ZeroAccumulators(
        MLAS_FLOAT32X4 Accumulators[RowCount][4]
        )
    {
        MLAS_UNREFERENCED_PARAMETER(Accumulators);
    }

    MLAS_FORCEINLINE
    static
    void
    LoadAElements(
        MLAS_FLOAT32X4 AElements[RowCount],
        const float* A,
        size_t lda
        )
    {
        MLAS_UNREFERENCED_PARAMETER(AElements);
        MLAS_UNREFERENCED_PARAMETER(A);
        MLAS_UNREFERENCED_PARAMETER(lda);
    }

    MLAS_FORCEINLINE
    static
    void
    BroadcastAElements(
        MLAS_FLOAT32X4 ABroadcast[RowCount],
        const float* A,
        size_t lda
        )
    {
        MLAS_UNREFERENCED_PARAMETER(ABroadcast);
        MLAS_UNREFERENCED_PARAMETER(A);
        MLAS_UNREFERENCED_PARAMETER(lda);
    }

    template<unsigned Lane>
    MLAS_FORCEINLINE
    static
    void
    BroadcastAElements(
        MLAS_FLOAT32X4 AElements[RowCount],
        MLAS_FLOAT32X4 ABroadcast[RowCount]
        )
    {
        MLAS_UNREFERENCED_PARAMETER(AElements);
        MLAS_UNREFERENCED_PARAMETER(ABroadcast);
    }

    MLAS_FORCEINLINE
    static
    void
    MultiplyAdd(
        MLAS_FLOAT32X4 Accumulators[RowCount][4],
        MLAS_FLOAT32X4 ABroadcast[RowCount],
        MLAS_FLOAT32X4 BElements[4]
        )
    {
        MLAS_UNREFERENCED_PARAMETER(Accumulators);
        MLAS_UNREFERENCED_PARAMETER(ABroadcast);
        MLAS_UNREFERENCED_PARAMETER(BElements);
    }

    template<unsigned ColumnCount>
    MLAS_FORCEINLINE
    static
    void
    StoreVector(
        MLAS_FLOAT32X4 Accumulators[RowCount][4],
        float* C,
        size_t ldc,
        MLAS_FLOAT32X4 AlphaBroadcast,
        bool ZeroMode
        )
    {
        MLAS_UNREFERENCED_PARAMETER(Accumulators);
        MLAS_UNREFERENCED_PARAMETER(C);
        MLAS_UNREFERENCED_PARAMETER(ldc);
        MLAS_UNREFERENCED_PARAMETER(AlphaBroadcast);
        MLAS_UNREFERENCED_PARAMETER(ZeroMode);
    }

    MLAS_FORCEINLINE
    static
    void
    MultiplyAlphaTrailing(
        MLAS_FLOAT32X4 Accumulators[RowCount][4],
        MLAS_FLOAT32X4 AlphaBroadcast
        )
    {
        MLAS_UNREFERENCED_PARAMETER(Accumulators);
        MLAS_UNREFERENCED_PARAMETER(AlphaBroadcast);
    }

    template<unsigned Lane>
    MLAS_FORCEINLINE
    static
    void
    StoreFloat(
        MLAS_FLOAT32X4 Accumulators[RowCount][4],
        float* C,
        size_t ldc,
        bool ZeroMode
        )
    {
        MLAS_UNREFERENCED_PARAMETER(Accumulators);
        MLAS_UNREFERENCED_PARAMETER(C);
        MLAS_UNREFERENCED_PARAMETER(ldc);
        MLAS_UNREFERENCED_PARAMETER(ZeroMode);
    }
};

template<size_t RowCount>
struct MLAS_SGEMM_ROW_LOOP : MLAS_SGEMM_ROW_LOOP_STEP<RowCount, 0>
{
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

    MLAS_SGEMM_ROW_LOOP<RowCount>::MultiplyAdd(Accumulators, ABroadcast, BElements);
}

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

        MLAS_SGEMM_ROW_LOOP<RowCount>::ZeroAccumulators(Accumulators);

        while (k >= 4) {

            MLAS_SGEMM_ROW_LOOP<RowCount>::LoadAElements(AElements, a, lda);

            MLAS_SGEMM_ROW_LOOP<RowCount>::template BroadcastAElements<0>(AElements, ABroadcast);
            MlasSgemmComputeBlock<RowCount>(Accumulators, ABroadcast, B);

            MLAS_SGEMM_ROW_LOOP<RowCount>::template BroadcastAElements<1>(AElements, ABroadcast);
            MlasSgemmComputeBlock<RowCount>(Accumulators, ABroadcast, B + 16);

            MLAS_SGEMM_ROW_LOOP<RowCount>::template BroadcastAElements<2>(AElements, ABroadcast);
            MlasSgemmComputeBlock<RowCount>(Accumulators, ABroadcast, B + 32);

            MLAS_SGEMM_ROW_LOOP<RowCount>::template BroadcastAElements<3>(AElements, ABroadcast);
            MlasSgemmComputeBlock<RowCount>(Accumulators, ABroadcast, B + 48);

            a += 4;
            B += 16 * 4;
            k -= 4;
        }

        while (k > 0) {

            MLAS_SGEMM_ROW_LOOP<RowCount>::BroadcastAElements(ABroadcast, a, lda);
            MlasSgemmComputeBlock<RowCount>(Accumulators, ABroadcast, B);

            a += 1;
            B += 16;
            k -= 1;
        }

        if (CountN >= 16) {

            //
            // Store the entire output block.
            //

            MLAS_SGEMM_ROW_LOOP<RowCount>::template StoreVector<16>(Accumulators, C, ldc, AlphaBroadcast, ZeroMode);

        } else {

            //
            // Store the partial output block.
            //

            if (CountN >= 12) {
                MLAS_SGEMM_ROW_LOOP<RowCount>::template StoreVector<12>(Accumulators, C, ldc, AlphaBroadcast, ZeroMode);
            } else if (CountN >= 8) {
                MLAS_SGEMM_ROW_LOOP<RowCount>::template StoreVector<8>(Accumulators, C, ldc, AlphaBroadcast, ZeroMode);
            } else if (CountN >= 4) {
                MLAS_SGEMM_ROW_LOOP<RowCount>::template StoreVector<4>(Accumulators, C, ldc, AlphaBroadcast, ZeroMode);
            }

            //
            // Store the remaining unaligned columns.
            //

            C += (CountN & ~3);
            CountN &= 3;

            if (CountN > 0) {

                MLAS_SGEMM_ROW_LOOP<RowCount>::MultiplyAlphaTrailing(Accumulators, AlphaBroadcast);

                MLAS_SGEMM_ROW_LOOP<RowCount>::template StoreFloat<0>(Accumulators, C, ldc, ZeroMode);

                if (CountN >= 2) {
                    MLAS_SGEMM_ROW_LOOP<RowCount>::template StoreFloat<1>(Accumulators, C, ldc, ZeroMode);
                }

                if (CountN >= 3) {
                    MLAS_SGEMM_ROW_LOOP<RowCount>::template StoreFloat<2>(Accumulators, C, ldc, ZeroMode);
                }
            }

            break;
        }

        C += 16;
        CountN -= 16;

    } while (CountN > 0);

    return RowCount;
}

size_t
MLASCALL
MlasSgemmKernel(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha,
    bool ZeroMode
    )
/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A - Supplies the address of matrix A.

    B - Supplies the address of matrix B. The matrix data has been packed using
        MlasSgemmCopyPackB or MlasSgemmTransposePackB.

    C - Supplies the address of matrix C.

    CountK - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    CountM - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.

    CountN - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    lda - Supplies the first dimension of matrix A.

    ldc - Supplies the first dimension of matrix C.

    alpha - Supplies the scaler multiplier (see SGEMM definition).

    ZeroMode - Supplies true if the output matrix must be zero initialized,
        else false if the output matrix is accumulated into.

Return Value:

    Returns the number of rows handled.

--*/
{
    size_t RowsHandled;

    MLAS_FLOAT32X4 AlphaBroadcast = MlasBroadcastFloat32x4(alpha);

    if (CountM >= 4) {
        RowsHandled = MlasSgemmProcessCount<4>(A, B, C, CountK, CountN, lda, ldc, AlphaBroadcast, ZeroMode);
    } else if (CountM >= 2) {
        RowsHandled = MlasSgemmProcessCount<2>(A, B, C, CountK, CountN, lda, ldc, AlphaBroadcast, ZeroMode);
    } else {
        RowsHandled = MlasSgemmProcessCount<1>(A, B, C, CountK, CountN, lda, ldc, AlphaBroadcast, ZeroMode);
    }

    return RowsHandled;
}
