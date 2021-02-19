/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sgemmc.cpp

Abstract:

    This module implements the kernels for the single precision matrix/matrix
    multiply operation (SGEMM).

--*/

#include "mlasi.h"

// #include <wasm_simd128.h>

template<bool ZeroMode, bool ProcessTwoRows>
size_t
MlasSgemmKernel(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
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

    CountN - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    lda - Supplies the first dimension of matrix A.

    ldc - Supplies the first dimension of matrix C.

    alpha - Supplies the scaler multiplier (see SGEMM definition).

Return Value:

    Returns the number of rows handled.

--*/
{
    MLAS_FLOAT32X4 Row0Block0;
    MLAS_FLOAT32X4 Row0Block1;
    MLAS_FLOAT32X4 Row0Block2;
    MLAS_FLOAT32X4 Row0Block3;

    MLAS_FLOAT32X4 Row1Block0;
    MLAS_FLOAT32X4 Row1Block1;
    MLAS_FLOAT32X4 Row1Block2;
    MLAS_FLOAT32X4 Row1Block3;

#if defined(_WIN32)

    if (!ProcessTwoRows) {
        UNREFERENCED_PARAMETER(lda);
        UNREFERENCED_PARAMETER(ldc);
    }

#endif

    MLAS_FLOAT32X4 Alpha = MlasBroadcastFloat32x4(alpha);

    do {

        MLAS_FLOAT32X4 BElements0;
        MLAS_FLOAT32X4 BElements1;
        MLAS_FLOAT32X4 BElements2;
        MLAS_FLOAT32X4 BElements3;

        float Row0AElements0;
        float Row0AElements1;
        float Row1AElements0;
        float Row1AElements1;

        //
        // Clear the block accumulators.
        //

        Row0Block0 = MlasZeroFloat32x4();
        Row0Block1 = MlasZeroFloat32x4();
        Row0Block2 = MlasZeroFloat32x4();
        Row0Block3 = MlasZeroFloat32x4();

        if (ProcessTwoRows) {
            Row1Block0 = MlasZeroFloat32x4();
            Row1Block1 = MlasZeroFloat32x4();
            Row1Block2 = MlasZeroFloat32x4();
            Row1Block3 = MlasZeroFloat32x4();
        }

        //
        // Compute the 16x1 or 16x2 output block.
        //

        const float* a = A;
        size_t k = CountK;

        while (k >= 2) {

            Row0AElements0 = a[0];
            Row0AElements1 = a[1];

            if (ProcessTwoRows) {
                Row1AElements0 = a[lda];
                Row1AElements1 = a[lda + 1];
            }

            BElements0 = MlasLoadFloat32x4(B + 0);
            BElements1 = MlasLoadFloat32x4(B + 4);
            BElements2 = MlasLoadFloat32x4(B + 8);
            BElements3 = MlasLoadFloat32x4(B + 12);

            Row0Block0 = MlasMultiplyAddFloat32x4(BElements0, Row0AElements0, Row0Block0);
            Row0Block1 = MlasMultiplyAddFloat32x4(BElements1, Row0AElements0, Row0Block1);
            Row0Block2 = MlasMultiplyAddFloat32x4(BElements2, Row0AElements0, Row0Block2);
            Row0Block3 = MlasMultiplyAddFloat32x4(BElements3, Row0AElements0, Row0Block3);

            if (ProcessTwoRows) {
                Row1Block0 = MlasMultiplyAddFloat32x4(BElements0, Row1AElements0, Row1Block0);
                Row1Block1 = MlasMultiplyAddFloat32x4(BElements1, Row1AElements0, Row1Block1);
                Row1Block2 = MlasMultiplyAddFloat32x4(BElements2, Row1AElements0, Row1Block2);
                Row1Block3 = MlasMultiplyAddFloat32x4(BElements3, Row1AElements0, Row1Block3);
            }

            BElements0 = MlasLoadFloat32x4(B + 16);
            BElements1 = MlasLoadFloat32x4(B + 20);
            BElements2 = MlasLoadFloat32x4(B + 24);
            BElements3 = MlasLoadFloat32x4(B + 28);

            Row0Block0 = MlasMultiplyAddFloat32x4(BElements0, Row0AElements1, Row0Block0);
            Row0Block1 = MlasMultiplyAddFloat32x4(BElements1, Row0AElements1, Row0Block1);
            Row0Block2 = MlasMultiplyAddFloat32x4(BElements2, Row0AElements1, Row0Block2);
            Row0Block3 = MlasMultiplyAddFloat32x4(BElements3, Row0AElements1, Row0Block3);

            if (ProcessTwoRows) {
                Row1Block0 = MlasMultiplyAddFloat32x4(BElements0, Row1AElements1, Row1Block0);
                Row1Block1 = MlasMultiplyAddFloat32x4(BElements1, Row1AElements1, Row1Block1);
                Row1Block2 = MlasMultiplyAddFloat32x4(BElements2, Row1AElements1, Row1Block2);
                Row1Block3 = MlasMultiplyAddFloat32x4(BElements3, Row1AElements1, Row1Block3);
            }

            a += 2;
            B += 32;
            k -= 2;
        }

        if (k > 0) {

            Row0AElements0 = a[0];
            Row0AElements1 = a[1];

            if (ProcessTwoRows) {
                Row1AElements0 = a[lda];
                Row1AElements1 = a[lda + 1];
            }

            BElements0 = MlasLoadFloat32x4(B + 0);
            BElements1 = MlasLoadFloat32x4(B + 4);
            BElements2 = MlasLoadFloat32x4(B + 8);
            BElements3 = MlasLoadFloat32x4(B + 12);

            Row0Block0 = MlasMultiplyAddFloat32x4(BElements0, Row0AElements0, Row0Block0);
            Row0Block1 = MlasMultiplyAddFloat32x4(BElements1, Row0AElements0, Row0Block1);
            Row0Block2 = MlasMultiplyAddFloat32x4(BElements2, Row0AElements0, Row0Block2);
            Row0Block3 = MlasMultiplyAddFloat32x4(BElements3, Row0AElements0, Row0Block3);

            if (ProcessTwoRows) {
                Row1Block0 = MlasMultiplyAddFloat32x4(BElements0, Row1AElements0, Row1Block0);
                Row1Block1 = MlasMultiplyAddFloat32x4(BElements1, Row1AElements0, Row1Block1);
                Row1Block2 = MlasMultiplyAddFloat32x4(BElements2, Row1AElements0, Row1Block2);
                Row1Block3 = MlasMultiplyAddFloat32x4(BElements3, Row1AElements0, Row1Block3);
            }

            B += 16;
        }

        //
        // Multiply by the alpha value.
        //

        Row0Block0 = MlasMultiplyFloat32x4(Row0Block0, Alpha);
        Row0Block1 = MlasMultiplyFloat32x4(Row0Block1, Alpha);
        Row0Block2 = MlasMultiplyFloat32x4(Row0Block2, Alpha);
        Row0Block3 = MlasMultiplyFloat32x4(Row0Block3, Alpha);

        if (ProcessTwoRows) {
            Row1Block0 = MlasMultiplyFloat32x4(Row1Block0, Alpha);
            Row1Block1 = MlasMultiplyFloat32x4(Row1Block1, Alpha);
            Row1Block2 = MlasMultiplyFloat32x4(Row1Block2, Alpha);
            Row1Block3 = MlasMultiplyFloat32x4(Row1Block3, Alpha);
        }

        if (CountN >= 16) {

            //
            // Store the entire output block.
            //

            if (!ZeroMode) {
                Row0Block0 = MlasAddFloat32x4(Row0Block0, MlasLoadFloat32x4(C));
                Row0Block1 = MlasAddFloat32x4(Row0Block1, MlasLoadFloat32x4(C + 4));
                Row0Block2 = MlasAddFloat32x4(Row0Block2, MlasLoadFloat32x4(C + 8));
                Row0Block3 = MlasAddFloat32x4(Row0Block3, MlasLoadFloat32x4(C + 12));
            }

            MlasStoreFloat32x4(C, Row0Block0);
            MlasStoreFloat32x4(C + 4, Row0Block1);
            MlasStoreFloat32x4(C + 8, Row0Block2);
            MlasStoreFloat32x4(C + 12, Row0Block3);

            if (ProcessTwoRows) {

                if (!ZeroMode) {
                    Row1Block0 = MlasAddFloat32x4(Row1Block0, MlasLoadFloat32x4(C + ldc));
                    Row1Block1 = MlasAddFloat32x4(Row1Block1, MlasLoadFloat32x4(C + ldc + 4));
                    Row1Block2 = MlasAddFloat32x4(Row1Block2, MlasLoadFloat32x4(C + ldc + 8));
                    Row1Block3 = MlasAddFloat32x4(Row1Block3, MlasLoadFloat32x4(C + ldc + 12));
                }

                MlasStoreFloat32x4(C + ldc, Row1Block0);
                MlasStoreFloat32x4(C + ldc + 4, Row1Block1);
                MlasStoreFloat32x4(C + ldc + 8, Row1Block2);
                MlasStoreFloat32x4(C + ldc + 12, Row1Block3);
            }

        } else {

            //
            // Store the partial output block.
            //

            if ((CountN & 8) != 0) {

                if (!ZeroMode) {
                    Row0Block0 = MlasAddFloat32x4(Row0Block0, MlasLoadFloat32x4(C));
                    Row0Block1 = MlasAddFloat32x4(Row0Block1, MlasLoadFloat32x4(C + 4));
                }

                MlasStoreFloat32x4(C, Row0Block0);
                MlasStoreFloat32x4(C + 4, Row0Block1);
                Row0Block0 = Row0Block2;
                Row0Block1 = Row0Block3;

                if (ProcessTwoRows) {

                    if (!ZeroMode) {
                        Row1Block0 = MlasAddFloat32x4(Row1Block0, MlasLoadFloat32x4(C + ldc));
                        Row1Block1 = MlasAddFloat32x4(Row1Block1, MlasLoadFloat32x4(C + ldc + 4));
                    }

                    MlasStoreFloat32x4(C + ldc, Row1Block0);
                    MlasStoreFloat32x4(C + ldc + 4, Row1Block1);
                    Row1Block0 = Row1Block2;
                    Row1Block1 = Row1Block3;
                }

                C += 8;
            }

            if ((CountN & 4) != 0) {

                if (!ZeroMode) {
                    Row0Block0 = MlasAddFloat32x4(Row0Block0, MlasLoadFloat32x4(C));
                }

                MlasStoreFloat32x4(C, Row0Block0);
                Row0Block0 = Row0Block1;

                if (ProcessTwoRows) {

                    if (!ZeroMode) {
                        Row1Block0 = MlasAddFloat32x4(Row1Block0, MlasLoadFloat32x4(C + ldc));
                    }

                    MlasStoreFloat32x4(C + ldc, Row1Block0);
                    Row1Block0 = Row1Block1;
                }

                C += 4;
            }

            float Row0Block00 = MlasExtractLaneFloat32x4<0>(Row0Block0);
            float Row0Block01 = MlasExtractLaneFloat32x4<1>(Row0Block0);
            float Row1Block00;
            float Row1Block01;

            if (ProcessTwoRows) {
                Row1Block00 = MlasExtractLaneFloat32x4<0>(Row1Block0);
                Row1Block01 = MlasExtractLaneFloat32x4<1>(Row1Block0);
            }

            if ((CountN & 2) != 0) {

                if (!ZeroMode) {
                    Row0Block00 = Row0Block00 + C[0];
                    Row0Block01 = Row0Block01 + C[1];
                }

                
                C[0] = Row0Block00;
                C[1] = Row0Block01;
                Row0Block00 = MlasExtractLaneFloat32x4<2>(Row0Block0);
                Row0Block01 = MlasExtractLaneFloat32x4<3>(Row0Block0);

                if (ProcessTwoRows) {

                    if (!ZeroMode) {
                        Row1Block00 = Row1Block00 + C[ldc];
                        Row1Block01 = Row1Block01 + C[ldc + 1];
                    }

                    C[ldc] = Row1Block00;
                    C[ldc + 1] = Row1Block01;
                    Row1Block00 = MlasExtractLaneFloat32x4<2>(Row1Block0);
                    Row1Block01 = MlasExtractLaneFloat32x4<3>(Row1Block0);
                }

                C += 2;
            }

            if ((CountN & 1) != 0) {

                if (!ZeroMode) {
                    Row0Block00 = Row0Block00 + C[0];
                }

                C[0] = Row0Block00;

                if (ProcessTwoRows) {

                    if (!ZeroMode) {
                        Row1Block00 = Row1Block00 + C[ldc];
                    }

                    C[ldc] = Row1Block00;
                }
            }

            break;
        }

        C += 16;
        CountN -= 16;

    } while (CountN > 0);

    return ProcessTwoRows ? 2 : 1;
}

template<bool ZeroMode>
size_t
MlasSgemmKernel(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
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

Return Value:

    Returns the number of rows handled.

--*/
{
    size_t RowsHandled;
    
    if (CountM >= 2) {
        RowsHandled = MlasSgemmKernel<ZeroMode, true>(A, B, C, CountK, CountN, lda, ldc, alpha);
    } else {
        RowsHandled = MlasSgemmKernel<ZeroMode, false>(A, B, C, CountK, CountN, lda, ldc, alpha);
    }

    return RowsHandled;
}

size_t
MLASCALL
MlasSgemmKernelZero(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
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

Return Value:

    Returns the number of rows handled.

--*/
{
    return MlasSgemmKernel<true>(A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
}

size_t
MLASCALL
MlasSgemmKernelAdd(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
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

Return Value:

    Returns the number of rows handled.

--*/
{
    return MlasSgemmKernel<false>(A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
}
