/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    DgemmKernelPower.cpp

Abstract:

    This module implements the kernels for the double precision matrix/matrix
    multiply operation (DGEMM).

--*/

#include "DgemmKernelpower.h"
struct MlasDgemmBroadcastAElementsMMA
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        double ARow[RowCount],
        const double* A,
        size_t lda
        )
    {
        ARow[Row] = A [Row * lda];
    }
};

template<size_t RowCount>
MLAS_FORCEINLINE
void
MlasDgemmComputeAElements(
    MLAS_FLOAT64X2 AElements[RowCount],
    MLAS_FLOAT64X2 ABroadcast[RowCount]
    )
{
    ABroadcast[0] = vec_mergee (AElements[0], AElements[1]);
    ABroadcast[1] = vec_mergee (AElements[2], AElements[3]);
    ABroadcast[2] = vec_mergeo (AElements[0], AElements[1]);
    ABroadcast[3] = vec_mergeo (AElements[2], AElements[3]);
}

template<size_t RowCount>
MLAS_FORCEINLINE
void
MlasDgemmComputeBlockMMA(
    __vector_quad acc[8],
    MLAS_FLOAT64X2 ABroadcast[RowCount],
    MLAS_FLOAT64X2 A2Broadcast[RowCount],
    const double* B,
    size_t CountM
    )
{
    MLAS_FLOAT64X2 BElements[4];
    typedef __vector unsigned char vec_t;
    __vector_pair A2pair, Apair;
#if (defined(__GNUC__) && (__GNUC__ == 10 && __GNUC_MINOR__ <= 3))
#if (__BYTE_ORDER__ != __ORDER_BIG_ENDIAN__)
    __builtin_mma_assemble_pair (&Apair, reinterpret_cast<vec_t>(ABroadcast[1]), reinterpret_cast<vec_t>(ABroadcast[0]));
    if (CountM == 8)  {
      __builtin_mma_assemble_pair (&A2pair, reinterpret_cast<vec_t>(A2Broadcast[1]), reinterpret_cast<vec_t>(A2Broadcast[0]));
    }
#else
    __builtin_mma_assemble_pair (&Apair, reinterpret_cast<vec_t>(ABroadcast[0]), reinterpret_cast<vec_t>(ABroadcast[1]));
    if (CountM == 8)  {
      __builtin_mma_assemble_pair (&A2pair, reinterpret_cast<vec_t>(A2Broadcast[0]), reinterpret_cast<vec_t>(A2Broadcast[1]));
    }
#endif
#elif (defined(__GNUC__) && (__GNUC__ == 11 && __GNUC_MINOR__ <= 2))
    Apair = *reinterpret_cast<__vector_pair *>(&ABroadcast[0]);
    if (CountM == 8)  {
      A2pair = *reinterpret_cast<__vector_pair *>(&A2Broadcast[0]);
    }
#else
    __builtin_vsx_build_pair (&Apair, reinterpret_cast<vec_t>(ABroadcast[0]), reinterpret_cast<vec_t>(ABroadcast[1]));
    if (CountM == 8)  {
      __builtin_vsx_build_pair (&A2pair, reinterpret_cast<vec_t>(A2Broadcast[0]), reinterpret_cast<vec_t>(A2Broadcast[1]));
    }
#endif
    BElements[0] = MlasLoadFloat64x2(B);
    BElements[1] = MlasLoadFloat64x2(B + 2);
    BElements[2] = MlasLoadFloat64x2(B + 4);
    BElements[3] = MlasLoadFloat64x2(B + 6);
   __builtin_mma_xvf64gerpp (&acc[0], Apair, reinterpret_cast<vec_t>(BElements[0]));
   __builtin_mma_xvf64gerpp (&acc[1], Apair, reinterpret_cast<vec_t>(BElements[1]));
   __builtin_mma_xvf64gerpp (&acc[2], Apair, reinterpret_cast<vec_t>(BElements[2]));
   __builtin_mma_xvf64gerpp (&acc[3], Apair, reinterpret_cast<vec_t>(BElements[3]));
   if (CountM == 8) {
       __builtin_mma_xvf64gerpp (&acc[4], A2pair, reinterpret_cast<vec_t>(BElements[0]));
       __builtin_mma_xvf64gerpp (&acc[5], A2pair, reinterpret_cast<vec_t>(BElements[1]));
       __builtin_mma_xvf64gerpp (&acc[6], A2pair, reinterpret_cast<vec_t>(BElements[2]));
       __builtin_mma_xvf64gerpp (&acc[7], A2pair, reinterpret_cast<vec_t>(BElements[3]));
   }
}
template<size_t VectorCount>
struct MlasDgemmStoreVectorMMA
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOAT64X2 Result[4],
        double* C,
        size_t ldc,
        MLAS_FLOAT64X2 AlphaBroadcast,
        bool ZeroMode
        )
    {
        MLAS_FLOAT64X2 *rowC;
        if (ZeroMode) {
            rowC = reinterpret_cast<MLAS_FLOAT64X2 *>(&C[Row * ldc + VectorCount]);
            rowC[0] = Result[Row] * AlphaBroadcast;
        } else {
            rowC = reinterpret_cast<MLAS_FLOAT64X2 *>(&C[Row * ldc + VectorCount]);
            rowC[0] += Result[Row] * AlphaBroadcast;
        }
    }
};

struct MlasDgemmMultiplyAlphaTrailingMMA
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOAT64X2 Accumulators[RowCount],
        MLAS_FLOAT64X2 AlphaBroadcast
        )
    {
        Accumulators[Row] = MlasMultiplyFloat64x2(Accumulators[Row], AlphaBroadcast);
    }
};
template<unsigned Lane>
struct MlasDgemmStoreScalarMMA
{
    template<size_t RowCount, size_t Row>
    MLAS_FORCEINLINE
    static
    void
    Iteration(
        MLAS_FLOAT64X2 Accumulators[RowCount],
        double* C,
        size_t ldc,
        bool ZeroMode
        )
    {
        double* c = C + Row * ldc + Lane;
        double Value = Accumulators[Row][Lane];
        if (!ZeroMode) {
            Value += *c;
        }

        *c = Value;
    }
};

template<size_t RowCount>
MLAS_FORCEINLINE
size_t
MlasDgemmMMAProcessCount(
    const double* A,
    const double* B,
    double* C,
    size_t CountM,
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

        MLAS_FLOAT64X2 Accumulators[2][RowCount] = {{ 0 }};
        MLAS_FLOAT64X2 Result[RowCount];
        MLAS_FLOAT64X2 AElements[RowCount];
        MLAS_FLOAT64X2 ABroadcast[RowCount] = { 0 };
        MLAS_FLOAT64X2 A2Broadcast[RowCount] = { 0 };
        MLAS_FLOAT64X2 A3Broadcast[RowCount] = { 0 };
        MLAS_FLOAT64X2 A4Broadcast[RowCount] = { 0 };
        double ARow[RowCount] = { 0 };
        double A2Row[RowCount] = { 0 };
        __vector_quad acc[8];

        //
        // Clear the block accumulators.
        //
        __builtin_mma_xxsetaccz(&acc[0]);
        __builtin_mma_xxsetaccz(&acc[1]);
        __builtin_mma_xxsetaccz(&acc[2]);
        __builtin_mma_xxsetaccz(&acc[3]);
        __builtin_mma_xxsetaccz(&acc[4]);
        __builtin_mma_xxsetaccz(&acc[5]);
        __builtin_mma_xxsetaccz(&acc[6]);
        __builtin_mma_xxsetaccz(&acc[7]);

        //
        // Compute the output block.
        //
        while (k >= 4) {

            MlasLoopUnroll<RowCount, MlasFgemmLoadAElements>()(AElements, a, lda);
            MlasDgemmComputeAElements<RowCount>(AElements, ABroadcast);
            MlasLoopUnroll<RowCount, MlasFgemmLoadAElements>()(AElements, a+2, lda);
            MlasDgemmComputeAElements<RowCount>(AElements, A3Broadcast);
            if (CountM == 8) {
                MlasLoopUnroll<RowCount, MlasFgemmLoadAElements>()(AElements, a + ( lda * 4), lda);
                MlasDgemmComputeAElements<RowCount>(AElements, A2Broadcast);
                MlasLoopUnroll<RowCount, MlasFgemmLoadAElements>()(AElements, (a+2) + ( lda * 4), lda);
                MlasDgemmComputeAElements<RowCount>(AElements, A4Broadcast);
            }
            MlasDgemmComputeBlockMMA<RowCount>(&acc[0], &ABroadcast[0], &A2Broadcast[0], B, CountM);
            MlasDgemmComputeBlockMMA<RowCount>(&acc[0], &ABroadcast[2], &A2Broadcast[2], B+8, CountM);
            MlasDgemmComputeBlockMMA<RowCount>(&acc[0], &A3Broadcast[0], &A4Broadcast[0], B+16, CountM);
            MlasDgemmComputeBlockMMA<RowCount>(&acc[0], &A3Broadcast[2], &A4Broadcast[2], B+24, CountM);
            B += 8 * 4;
            a += 4;
            k -= 4;
        }
        while (k > 0) {
            MlasLoopUnroll<RowCount, MlasDgemmBroadcastAElementsMMA>()(ARow, a, lda);
            if (CountM == 8)  {
                MlasLoopUnroll<RowCount, MlasDgemmBroadcastAElementsMMA>()(A2Row, a + (lda * 4), lda);
            }

            MlasDgemmComputeBlockMMA<RowCount>(&acc[0], (MLAS_FLOAT64X2 *)ARow, (MLAS_FLOAT64X2 *)A2Row, B, CountM);
            a += 1;
            B += 8;
            k -= 1;
        }
        if (CountN >= 8) {

            //
            // Store the entire output block.
            //
            __builtin_mma_disassemble_acc (Result, &acc[0]);
            MlasLoopUnroll<RowCount, MlasDgemmStoreVectorMMA<0>>()(Result, C, ldc, AlphaBroadcast, ZeroMode);
            __builtin_mma_disassemble_acc (Result, &acc[1]);
            MlasLoopUnroll<RowCount, MlasDgemmStoreVectorMMA<2>>()(Result, C, ldc, AlphaBroadcast, ZeroMode);
            __builtin_mma_disassemble_acc (Result, &acc[2]);
            MlasLoopUnroll<RowCount, MlasDgemmStoreVectorMMA<4>>()(Result, C, ldc, AlphaBroadcast, ZeroMode);
            __builtin_mma_disassemble_acc (Result, &acc[3]);
            MlasLoopUnroll<RowCount, MlasDgemmStoreVectorMMA<6>>()(Result, C, ldc, AlphaBroadcast, ZeroMode);
            if (CountM == 8) {
                __builtin_mma_disassemble_acc (Result, &acc[4]);
                MlasLoopUnroll<RowCount, MlasDgemmStoreVectorMMA<0>>()(Result, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
                __builtin_mma_disassemble_acc (Result, &acc[5]);
                MlasLoopUnroll<RowCount, MlasDgemmStoreVectorMMA<2>>()(Result, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
                __builtin_mma_disassemble_acc (Result, &acc[6]);
                MlasLoopUnroll<RowCount, MlasDgemmStoreVectorMMA<4>>()(Result, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
                __builtin_mma_disassemble_acc (Result, &acc[7]);
                MlasLoopUnroll<RowCount, MlasDgemmStoreVectorMMA<6>>()(Result, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
            }
        } else {

            //
            // Store the partial output block.
            //

            if (CountN >= 6) {
                __builtin_mma_disassemble_acc (Result, &acc[0]);
                MlasLoopUnroll<RowCount, MlasDgemmStoreVectorMMA<0>>()(Result, C, ldc, AlphaBroadcast, ZeroMode);
                __builtin_mma_disassemble_acc (Result, &acc[1]);
                MlasLoopUnroll<RowCount, MlasDgemmStoreVectorMMA<2>>()(Result, C, ldc, AlphaBroadcast, ZeroMode);
                __builtin_mma_disassemble_acc (Result, &acc[2]);
                MlasLoopUnroll<RowCount, MlasDgemmStoreVectorMMA<4>>()(Result, C, ldc, AlphaBroadcast, ZeroMode);
                if (CountM == 8) {
                    __builtin_mma_disassemble_acc (Result, &acc[4]);
                    MlasLoopUnroll<RowCount, MlasDgemmStoreVectorMMA<0>>()(Result, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
                    __builtin_mma_disassemble_acc (Result, &acc[5]);
                    MlasLoopUnroll<RowCount, MlasDgemmStoreVectorMMA<2>>()(Result, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
                    __builtin_mma_disassemble_acc (Result, &acc[6]);
                    MlasLoopUnroll<RowCount, MlasDgemmStoreVectorMMA<4>>()(Result, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
                    if (CountN - 6 > 0) {
                        __builtin_mma_disassemble_acc (Accumulators[1], &acc[7]);
                    }
                }
                if (CountN - 6 > 0) {
                    __builtin_mma_disassemble_acc (Accumulators[0], &acc[3]);
                }
            } else if (CountN >= 4) {
                __builtin_mma_disassemble_acc (Result, &acc[0]);
                MlasLoopUnroll<RowCount, MlasDgemmStoreVectorMMA<0>>()(Result, C, ldc, AlphaBroadcast, ZeroMode);
                __builtin_mma_disassemble_acc (Result, &acc[1]);
                MlasLoopUnroll<RowCount, MlasDgemmStoreVectorMMA<2>>()(Result, C, ldc, AlphaBroadcast, ZeroMode);
                if (CountM == 8) {
                    __builtin_mma_disassemble_acc (Result, &acc[4]);
                    MlasLoopUnroll<RowCount, MlasDgemmStoreVectorMMA<0>>()(Result, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
                    __builtin_mma_disassemble_acc (Result, &acc[5]);
                    MlasLoopUnroll<RowCount, MlasDgemmStoreVectorMMA<2>>()(Result, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
                    if (CountN - 4 > 0) {
                        __builtin_mma_disassemble_acc (Accumulators[1], &acc[6]);
                    }
                }
                if (CountN - 4 > 0) {
                    __builtin_mma_disassemble_acc (Accumulators[0], &acc[2]);
                }
            } else if (CountN >= 2) {
                __builtin_mma_disassemble_acc (Result, &acc[0]);
                MlasLoopUnroll<RowCount, MlasDgemmStoreVectorMMA<0>>()(Result, C, ldc, AlphaBroadcast, ZeroMode);
                if (CountM == 8) {
                    __builtin_mma_disassemble_acc (Result, &acc[4]);
                    MlasLoopUnroll<RowCount, MlasDgemmStoreVectorMMA<0>>()(Result, C + (ldc*4), ldc, AlphaBroadcast, ZeroMode);
                    if (CountN - 2 > 0) {
                        __builtin_mma_disassemble_acc (Accumulators[1], &acc[5]);
                    }
                }
                if (CountN - 2 > 0) {
                    __builtin_mma_disassemble_acc (Accumulators[0], &acc[1]);
                }
            } else {
                __builtin_mma_disassemble_acc (Accumulators[0], &acc[0]);
                if (CountM == 8) {
                    __builtin_mma_disassemble_acc (Accumulators[1], &acc[4]);
                }
           }

            //
            // Store the remaining unaligned columns.
            //
            C += (CountN & ~1);
            CountN &= 1;

            if (CountN > 0) {

                MlasLoopUnroll<RowCount, MlasDgemmMultiplyAlphaTrailingMMA>()(Accumulators[0], AlphaBroadcast);
                MlasLoopUnroll<RowCount, MlasDgemmStoreScalarMMA<0>>()(Accumulators[0], C, ldc, ZeroMode);
                if (CountM == 8) {
                    MlasLoopUnroll<RowCount, MlasDgemmMultiplyAlphaTrailingMMA>()(Accumulators[1], AlphaBroadcast);
                    MlasLoopUnroll<RowCount, MlasDgemmStoreScalarMMA<0>>()(Accumulators[1], C + (ldc*4), ldc, ZeroMode);
                }
            }

            break;
        }

        C += 8; 
        CountN -= 8;

    } while (CountN > 0);

    return CountM;
}

size_t
MLASCALL
MlasDgemmKernelPOWER10(
    const double* A,
    const double* B,
    double* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    double alpha,
    bool ZeroMode
    )
/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A - Supplies the address of matrix A.

    B - Supplies the address of matrix B. The matrix data has been packed using
        MlasDgemmCopyPackB or MlasDgemmTransposePackB.

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

    alpha - Supplies the scalar multiplier (see DGEMM definition).

    ZeroMode - Supplies true if the output matrix must be zero initialized,
        else false if the output matrix is accumulated into.

Return Value:

    Returns the number of rows handled.

--*/
{
    size_t RowsHandled;
    MLAS_FLOAT64X2 AlphaBroadcast = MlasBroadcastFloat64x2(alpha);
    if (CountM >= 8) {
        RowsHandled = MlasDgemmMMAProcessCount<4>(A, B, C, 8 ,CountK, CountN, lda, ldc, AlphaBroadcast, ZeroMode);
    } else if (CountM >= 4) {
        RowsHandled = MlasDgemmMMAProcessCount<4>(A, B, C, 4, CountK, CountN, lda, ldc, AlphaBroadcast, ZeroMode);
    } else if (CountM >= 2) {
        RowsHandled = MlasDgemmProcessCount<2>(A, B, C, CountK, CountN, lda, ldc, AlphaBroadcast, ZeroMode);
    } else {
        RowsHandled = MlasDgemmProcessCount<1>(A, B, C, CountK, CountN, lda, ldc, AlphaBroadcast, ZeroMode);
    }

    return RowsHandled;
}
