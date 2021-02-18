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
    float Row0Block00;
    float Row0Block01;
    float Row0Block02;
    float Row0Block03;
    float Row0Block10;
    float Row0Block11;
    float Row0Block12;
    float Row0Block13;
    float Row0Block20;
    float Row0Block21;
    float Row0Block22;
    float Row0Block23;
    float Row0Block30;
    float Row0Block31;
    float Row0Block32;
    float Row0Block33;

    float Row1Block00;
    float Row1Block01;
    float Row1Block02;
    float Row1Block03;
    float Row1Block10;
    float Row1Block11;
    float Row1Block12;
    float Row1Block13;
    float Row1Block20;
    float Row1Block21;
    float Row1Block22;
    float Row1Block23;
    float Row1Block30;
    float Row1Block31;
    float Row1Block32;
    float Row1Block33;

#if defined(_WIN32)

    if (!ProcessTwoRows) {
        UNREFERENCED_PARAMETER(lda);
        UNREFERENCED_PARAMETER(ldc);
    }

#endif

    do {

        float BElements00;
        float BElements01;
        float BElements02;
        float BElements03;
        float BElements10;
        float BElements11;
        float BElements12;
        float BElements13;
        float BElements20;
        float BElements21;
        float BElements22;
        float BElements23;
        float BElements30;
        float BElements31;
        float BElements32;
        float BElements33;

        float Row0AElements0;
        float Row0AElements1;
        float Row1AElements0;
        float Row1AElements1;

        //
        // Clear the block accumulators.
        //

        Row0Block00 = 0.0f;
        Row0Block01 = 0.0f;
        Row0Block02 = 0.0f;
        Row0Block03 = 0.0f;
        Row0Block10 = 0.0f;
        Row0Block11 = 0.0f;
        Row0Block12 = 0.0f;
        Row0Block13 = 0.0f;
        Row0Block20 = 0.0f;
        Row0Block21 = 0.0f;
        Row0Block22 = 0.0f;
        Row0Block23 = 0.0f;
        Row0Block30 = 0.0f;
        Row0Block31 = 0.0f;
        Row0Block32 = 0.0f;
        Row0Block33 = 0.0f;

        if (ProcessTwoRows) {
            Row1Block00 = 0.0f;
            Row1Block01 = 0.0f;
            Row1Block02 = 0.0f;
            Row1Block03 = 0.0f;
            Row1Block10 = 0.0f;
            Row1Block11 = 0.0f;
            Row1Block12 = 0.0f;
            Row1Block13 = 0.0f;
            Row1Block20 = 0.0f;
            Row1Block21 = 0.0f;
            Row1Block22 = 0.0f;
            Row1Block23 = 0.0f;
            Row1Block30 = 0.0f;
            Row1Block31 = 0.0f;
            Row1Block32 = 0.0f;
            Row1Block33 = 0.0f;
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

            BElements00 = B[0];
            BElements01 = B[1];
            BElements02 = B[2];
            BElements03 = B[3];
            BElements10 = B[4];
            BElements11 = B[5];
            BElements12 = B[6];
            BElements13 = B[7];
            BElements20 = B[8];
            BElements21 = B[9];
            BElements22 = B[10];
            BElements23 = B[11];
            BElements30 = B[12];
            BElements31 = B[13];
            BElements32 = B[14];
            BElements33 = B[15];

            Row0Block00 = Row0Block00 + BElements00 * Row0AElements0;
            Row0Block01 = Row0Block01 + BElements01 * Row0AElements0;
            Row0Block02 = Row0Block02 + BElements02 * Row0AElements0;
            Row0Block03 = Row0Block03 + BElements03 * Row0AElements0;
            Row0Block10 = Row0Block10 + BElements10 * Row0AElements0;
            Row0Block11 = Row0Block11 + BElements11 * Row0AElements0;
            Row0Block12 = Row0Block12 + BElements12 * Row0AElements0;
            Row0Block13 = Row0Block13 + BElements13 * Row0AElements0;
            Row0Block20 = Row0Block20 + BElements20 * Row0AElements0;
            Row0Block21 = Row0Block21 + BElements21 * Row0AElements0;
            Row0Block22 = Row0Block22 + BElements22 * Row0AElements0;
            Row0Block23 = Row0Block23 + BElements23 * Row0AElements0;
            Row0Block30 = Row0Block30 + BElements30 * Row0AElements0;
            Row0Block31 = Row0Block31 + BElements31 * Row0AElements0;
            Row0Block32 = Row0Block32 + BElements32 * Row0AElements0;
            Row0Block33 = Row0Block33 + BElements33 * Row0AElements0;

            if (ProcessTwoRows) {
                Row1Block00 = Row1Block00 + BElements00 * Row1AElements0;
                Row1Block01 = Row1Block01 + BElements01 * Row1AElements0;
                Row1Block02 = Row1Block02 + BElements02 * Row1AElements0;
                Row1Block03 = Row1Block03 + BElements03 * Row1AElements0;
                Row1Block10 = Row1Block10 + BElements10 * Row1AElements0;
                Row1Block11 = Row1Block11 + BElements11 * Row1AElements0;
                Row1Block12 = Row1Block12 + BElements12 * Row1AElements0;
                Row1Block13 = Row1Block13 + BElements13 * Row1AElements0;
                Row1Block20 = Row1Block20 + BElements20 * Row1AElements0;
                Row1Block21 = Row1Block21 + BElements21 * Row1AElements0;
                Row1Block22 = Row1Block22 + BElements22 * Row1AElements0;
                Row1Block23 = Row1Block23 + BElements23 * Row1AElements0;
                Row1Block30 = Row1Block30 + BElements30 * Row1AElements0;
                Row1Block31 = Row1Block31 + BElements31 * Row1AElements0;
                Row1Block32 = Row1Block32 + BElements32 * Row1AElements0;
                Row1Block33 = Row1Block33 + BElements33 * Row1AElements0;
            }

            BElements00 = B[16];
            BElements01 = B[17];
            BElements02 = B[18];
            BElements03 = B[19];
            BElements10 = B[20];
            BElements11 = B[21];
            BElements12 = B[22];
            BElements13 = B[23];
            BElements20 = B[24];
            BElements21 = B[25];
            BElements22 = B[26];
            BElements23 = B[27];
            BElements30 = B[28];
            BElements31 = B[29];
            BElements32 = B[30];
            BElements33 = B[31];

            Row0Block00 = Row0Block00 + BElements00 * Row0AElements1;
            Row0Block01 = Row0Block01 + BElements01 * Row0AElements1;
            Row0Block02 = Row0Block02 + BElements02 * Row0AElements1;
            Row0Block03 = Row0Block03 + BElements03 * Row0AElements1;
            Row0Block10 = Row0Block10 + BElements10 * Row0AElements1;
            Row0Block11 = Row0Block11 + BElements11 * Row0AElements1;
            Row0Block12 = Row0Block12 + BElements12 * Row0AElements1;
            Row0Block13 = Row0Block13 + BElements13 * Row0AElements1;
            Row0Block20 = Row0Block20 + BElements20 * Row0AElements1;
            Row0Block21 = Row0Block21 + BElements21 * Row0AElements1;
            Row0Block22 = Row0Block22 + BElements22 * Row0AElements1;
            Row0Block23 = Row0Block23 + BElements23 * Row0AElements1;
            Row0Block30 = Row0Block30 + BElements30 * Row0AElements1;
            Row0Block31 = Row0Block31 + BElements31 * Row0AElements1;
            Row0Block32 = Row0Block32 + BElements32 * Row0AElements1;
            Row0Block33 = Row0Block33 + BElements33 * Row0AElements1;

            if (ProcessTwoRows) {
                Row1Block00 = Row1Block00 + BElements00 * Row1AElements1;
                Row1Block01 = Row1Block01 + BElements01 * Row1AElements1;
                Row1Block02 = Row1Block02 + BElements02 * Row1AElements1;
                Row1Block03 = Row1Block03 + BElements03 * Row1AElements1;
                Row1Block10 = Row1Block10 + BElements10 * Row1AElements1;
                Row1Block11 = Row1Block11 + BElements11 * Row1AElements1;
                Row1Block12 = Row1Block12 + BElements12 * Row1AElements1;
                Row1Block13 = Row1Block13 + BElements13 * Row1AElements1;
                Row1Block20 = Row1Block20 + BElements20 * Row1AElements1;
                Row1Block21 = Row1Block21 + BElements21 * Row1AElements1;
                Row1Block22 = Row1Block22 + BElements22 * Row1AElements1;
                Row1Block23 = Row1Block23 + BElements23 * Row1AElements1;
                Row1Block30 = Row1Block30 + BElements30 * Row1AElements1;
                Row1Block31 = Row1Block31 + BElements31 * Row1AElements1;
                Row1Block32 = Row1Block32 + BElements32 * Row1AElements1;
                Row1Block33 = Row1Block33 + BElements33 * Row1AElements1;
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

            BElements00 = B[0];
            BElements01 = B[1];
            BElements02 = B[2];
            BElements03 = B[3];
            BElements10 = B[4];
            BElements11 = B[5];
            BElements12 = B[6];
            BElements13 = B[7];
            BElements20 = B[8];
            BElements21 = B[9];
            BElements22 = B[10];
            BElements23 = B[11];
            BElements30 = B[12];
            BElements31 = B[13];
            BElements32 = B[14];
            BElements33 = B[15];

            Row0Block00 = Row0Block00 + BElements00 * Row0AElements0;
            Row0Block01 = Row0Block01 + BElements01 * Row0AElements0;
            Row0Block02 = Row0Block02 + BElements02 * Row0AElements0;
            Row0Block03 = Row0Block03 + BElements03 * Row0AElements0;
            Row0Block10 = Row0Block10 + BElements10 * Row0AElements0;
            Row0Block11 = Row0Block11 + BElements11 * Row0AElements0;
            Row0Block12 = Row0Block12 + BElements12 * Row0AElements0;
            Row0Block13 = Row0Block13 + BElements13 * Row0AElements0;
            Row0Block20 = Row0Block20 + BElements20 * Row0AElements0;
            Row0Block21 = Row0Block21 + BElements21 * Row0AElements0;
            Row0Block22 = Row0Block22 + BElements22 * Row0AElements0;
            Row0Block23 = Row0Block23 + BElements23 * Row0AElements0;
            Row0Block30 = Row0Block30 + BElements30 * Row0AElements0;
            Row0Block31 = Row0Block31 + BElements31 * Row0AElements0;
            Row0Block32 = Row0Block32 + BElements32 * Row0AElements0;
            Row0Block33 = Row0Block33 + BElements33 * Row0AElements0;

            if (ProcessTwoRows) {
                Row1Block00 = Row1Block00 + BElements00 * Row1AElements0;
                Row1Block01 = Row1Block01 + BElements01 * Row1AElements0;
                Row1Block02 = Row1Block02 + BElements02 * Row1AElements0;
                Row1Block03 = Row1Block03 + BElements03 * Row1AElements0;
                Row1Block10 = Row1Block10 + BElements10 * Row1AElements0;
                Row1Block11 = Row1Block11 + BElements11 * Row1AElements0;
                Row1Block12 = Row1Block12 + BElements12 * Row1AElements0;
                Row1Block13 = Row1Block13 + BElements13 * Row1AElements0;
                Row1Block20 = Row1Block20 + BElements20 * Row1AElements0;
                Row1Block21 = Row1Block21 + BElements21 * Row1AElements0;
                Row1Block22 = Row1Block22 + BElements22 * Row1AElements0;
                Row1Block23 = Row1Block23 + BElements23 * Row1AElements0;
                Row1Block30 = Row1Block30 + BElements30 * Row1AElements0;
                Row1Block31 = Row1Block31 + BElements31 * Row1AElements0;
                Row1Block32 = Row1Block32 + BElements32 * Row1AElements0;
                Row1Block33 = Row1Block33 + BElements33 * Row1AElements0;
            }

            B += 16;
        }

        //
        // Multiply by the alpha value.
        //

        Row0Block00 = Row0Block00 * alpha;
        Row0Block01 = Row0Block01 * alpha;
        Row0Block02 = Row0Block02 * alpha;
        Row0Block03 = Row0Block03 * alpha;
        Row0Block10 = Row0Block10 * alpha;
        Row0Block11 = Row0Block11 * alpha;
        Row0Block12 = Row0Block12 * alpha;
        Row0Block13 = Row0Block13 * alpha;
        Row0Block20 = Row0Block20 * alpha;
        Row0Block21 = Row0Block21 * alpha;
        Row0Block22 = Row0Block22 * alpha;
        Row0Block23 = Row0Block23 * alpha;
        Row0Block30 = Row0Block30 * alpha;
        Row0Block31 = Row0Block31 * alpha;
        Row0Block32 = Row0Block32 * alpha;
        Row0Block33 = Row0Block33 * alpha;

        if (ProcessTwoRows) {
            Row1Block00 = Row1Block00 * alpha;
            Row1Block01 = Row1Block01 * alpha;
            Row1Block02 = Row1Block02 * alpha;
            Row1Block03 = Row1Block03 * alpha;
            Row1Block10 = Row1Block10 * alpha;
            Row1Block11 = Row1Block11 * alpha;
            Row1Block12 = Row1Block12 * alpha;
            Row1Block13 = Row1Block13 * alpha;
            Row1Block20 = Row1Block20 * alpha;
            Row1Block21 = Row1Block21 * alpha;
            Row1Block22 = Row1Block22 * alpha;
            Row1Block23 = Row1Block23 * alpha;
            Row1Block30 = Row1Block30 * alpha;
            Row1Block31 = Row1Block31 * alpha;
            Row1Block32 = Row1Block32 * alpha;
            Row1Block33 = Row1Block33 * alpha;
        }

        if (CountN >= 16) {

            //
            // Store the entire output block.
            //

            if (!ZeroMode) {
                Row0Block00 = Row0Block00 + C[0];
                Row0Block01 = Row0Block01 + C[1];
                Row0Block02 = Row0Block02 + C[2];
                Row0Block03 = Row0Block03 + C[3];
                Row0Block10 = Row0Block10 + C[4];
                Row0Block11 = Row0Block11 + C[5];
                Row0Block12 = Row0Block12 + C[6];
                Row0Block13 = Row0Block13 + C[7];
                Row0Block20 = Row0Block20 + C[8];
                Row0Block21 = Row0Block21 + C[9];
                Row0Block22 = Row0Block22 + C[10];
                Row0Block23 = Row0Block23 + C[11];
                Row0Block30 = Row0Block30 + C[12];
                Row0Block31 = Row0Block31 + C[13];
                Row0Block32 = Row0Block32 + C[14];
                Row0Block33 = Row0Block33 + C[15];
            }

            C[0] = Row0Block00;
            C[1] = Row0Block01;
            C[2] = Row0Block02;
            C[3] = Row0Block03;
            C[4] = Row0Block10;
            C[5] = Row0Block11;
            C[6] = Row0Block12;
            C[7] = Row0Block13;
            C[8] = Row0Block20;
            C[9] = Row0Block21;
            C[10] = Row0Block22;
            C[11] = Row0Block23;
            C[12] = Row0Block30;
            C[13] = Row0Block31;
            C[14] = Row0Block32;
            C[15] = Row0Block33;

            if (ProcessTwoRows) {

                if (!ZeroMode) {
                    Row1Block00 = Row1Block00 + C[ldc];
                    Row1Block01 = Row1Block01 + C[ldc + 1];
                    Row1Block02 = Row1Block02 + C[ldc + 2];
                    Row1Block03 = Row1Block03 + C[ldc + 3];
                    Row1Block10 = Row1Block10 + C[ldc + 4];
                    Row1Block11 = Row1Block11 + C[ldc + 5];
                    Row1Block12 = Row1Block12 + C[ldc + 6];
                    Row1Block13 = Row1Block13 + C[ldc + 7];
                    Row1Block20 = Row1Block20 + C[ldc + 8];
                    Row1Block21 = Row1Block21 + C[ldc + 9];
                    Row1Block22 = Row1Block22 + C[ldc + 10];
                    Row1Block23 = Row1Block23 + C[ldc + 11];
                    Row1Block30 = Row1Block30 + C[ldc + 12];
                    Row1Block31 = Row1Block31 + C[ldc + 13];
                    Row1Block32 = Row1Block32 + C[ldc + 14];
                    Row1Block33 = Row1Block33 + C[ldc + 15];
                }

                C[ldc] = Row1Block00;
                C[ldc + 1] = Row1Block01;
                C[ldc + 2] = Row1Block02;
                C[ldc + 3] = Row1Block03;
                C[ldc + 4] = Row1Block10;
                C[ldc + 5] = Row1Block11;
                C[ldc + 6] = Row1Block12;
                C[ldc + 7] = Row1Block13;
                C[ldc + 8] = Row1Block20;
                C[ldc + 9] = Row1Block21;
                C[ldc + 10] = Row1Block22;
                C[ldc + 11] = Row1Block23;
                C[ldc + 12] = Row1Block30;
                C[ldc + 13] = Row1Block31;
                C[ldc + 14] = Row1Block32;
                C[ldc + 15] = Row1Block33;
            }

        } else {

            //
            // Store the partial output block.
            //

            if ((CountN & 8) != 0) {

                if (!ZeroMode) {
                    Row0Block00 = Row0Block00 + C[0];
                    Row0Block01 = Row0Block01 + C[1];
                    Row0Block02 = Row0Block02 + C[2];
                    Row0Block03 = Row0Block03 + C[3];
                    Row0Block10 = Row0Block10 + C[4];
                    Row0Block11 = Row0Block11 + C[5];
                    Row0Block12 = Row0Block12 + C[6];
                    Row0Block13 = Row0Block13 + C[7];
                }

                C[0] = Row0Block00;
                C[1] = Row0Block01;
                C[2] = Row0Block02;
                C[3] = Row0Block03;
                C[4] = Row0Block10;
                C[5] = Row0Block11;
                C[6] = Row0Block12;
                C[7] = Row0Block13;

                Row0Block00 = Row0Block20;
                Row0Block01 = Row0Block21;
                Row0Block02 = Row0Block22;
                Row0Block03 = Row0Block23;
                Row0Block10 = Row0Block30;
                Row0Block11 = Row0Block31;
                Row0Block12 = Row0Block32;
                Row0Block13 = Row0Block33;

                if (ProcessTwoRows) {

                    if (!ZeroMode) {
                        Row1Block00 = Row1Block00 + C[ldc];
                        Row1Block01 = Row1Block01 + C[ldc + 1];
                        Row1Block02 = Row1Block02 + C[ldc + 2];
                        Row1Block03 = Row1Block03 + C[ldc + 3];
                        Row1Block10 = Row1Block10 + C[ldc + 4];
                        Row1Block11 = Row1Block11 + C[ldc + 5];
                        Row1Block12 = Row1Block12 + C[ldc + 6];
                        Row1Block13 = Row1Block13 + C[ldc + 7];
                    }

                    C[ldc] = Row1Block00;
                    C[ldc + 1] = Row1Block01;
                    C[ldc + 2] = Row1Block02;
                    C[ldc + 3] = Row1Block03;
                    C[ldc + 4] = Row1Block10;
                    C[ldc + 5] = Row1Block11;
                    C[ldc + 6] = Row1Block12;
                    C[ldc + 7] = Row1Block13;
                    Row1Block00 = Row1Block20;
                    Row1Block01 = Row1Block21;
                    Row1Block02 = Row1Block22;
                    Row1Block03 = Row1Block23;
                    Row1Block10 = Row1Block30;
                    Row1Block11 = Row1Block31;
                    Row1Block12 = Row1Block32;
                    Row1Block13 = Row1Block33;
                }

                C += 8;
            }

            if ((CountN & 4) != 0) {

                if (!ZeroMode) {
                    Row0Block00 = Row0Block00 + C[0];
                    Row0Block01 = Row0Block01 + C[1];
                    Row0Block02 = Row0Block02 + C[2];
                    Row0Block03 = Row0Block03 + C[3];
                }

                C[0] = Row0Block00;
                C[1] = Row0Block01;
                C[2] = Row0Block02;
                C[3] = Row0Block03;
                Row0Block00 = Row0Block10;
                Row0Block01 = Row0Block11;
                Row0Block02 = Row0Block12;
                Row0Block03 = Row0Block13;

                if (ProcessTwoRows) {

                    if (!ZeroMode) {
                        Row1Block00 = Row1Block00 + C[ldc];
                        Row1Block01 = Row1Block01 + C[ldc + 1];
                        Row1Block02 = Row1Block02 + C[ldc + 2];
                        Row1Block03 = Row1Block03 + C[ldc + 3];
                    }

                    C[ldc] = Row1Block00;
                    C[ldc + 1] = Row1Block01;
                    C[ldc + 2] = Row1Block02;
                    C[ldc + 3] = Row1Block03;
                    Row1Block00 = Row1Block10;
                    Row1Block01 = Row1Block11;
                    Row1Block02 = Row1Block12;
                    Row1Block03 = Row1Block13;
                }

                C += 4;
            }

            if ((CountN & 2) != 0) {

                if (!ZeroMode) {
                    Row0Block00 = Row0Block00 + C[0];
                    Row0Block01 = Row0Block01 + C[1];
                }

                C[0] = Row0Block00;
                C[1] = Row0Block01;
                Row0Block00 = Row0Block02;
                Row0Block01 = Row0Block03;

                if (ProcessTwoRows) {

                    if (!ZeroMode) {
                        Row1Block00 = Row1Block00 + C[ldc];
                        Row1Block01 = Row1Block01 + C[ldc + 1];
                    }

                    C[ldc] = Row1Block00;
                    C[ldc + 1] = Row1Block01;
                    Row1Block00 = Row1Block02;
                    Row1Block01 = Row1Block03;
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
