/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemmKernelScalar.cpp

Abstract:

    This module implements the kernels for the single precision matrix/matrix
    multiply operation (SGEMM).

--*/

#include "mlasi.h"

#define MLAS_FLOAT32X4(name) \
    float name##0; \
    float name##1; \
    float name##2; \
    float name##3;

#define MLAS_SET_ZERO_FLOAT32X4(name) \
    name##0 = 0.0f; \
    name##1 = 0.0f; \
    name##2 = 0.0f; \
    name##3 = 0.0f;

#define MLAS_SET_VECTOR_FLOAT32X4(name, name_b) \
    name##0 = name_b##0; \
    name##1 = name_b##1; \
    name##2 = name_b##2; \
    name##3 = name_b##3;

#define MLAS_SET_VALUE_FLOAT32X4(name, address, index) \
    name##0 = address[index]; \
    name##1 = address[index+1]; \
    name##2 = address[index+2]; \
    name##3 = address[index+3];

#define MLAS_ADD_MUL_FLOAT32x4(a, b, c)\
    a##0 = a##0 + b##0 * c; \
    a##1 = a##1 + b##1 * c; \
    a##2 = a##2 + b##2 * c; \
    a##3 = a##3 + b##3 * c;

#define MLAS_ADD_FLOAT32x4(a, c, index)\
    a##0 = a##0 + c[index]; \
    a##1 = a##1 + c[index+1]; \
    a##2 = a##2 + c[index+2]; \
    a##3 = a##3 + c[index+3];

#define MLAS_SET_RESULT_FLOAT32x4(c, index, a)\
    c[index] = a##0; \
    c[index+1] = a##1; \
    c[index+2] = a##2; \
    c[index+3] = a##3;

#define MLAS_MUL_FLOAT32x4(a, c)\
    a##0 = a##0 * c; \
    a##1 = a##1 * c; \
    a##2 = a##2 * c; \
    a##3 = a##3 * c;

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
        MlasSgemmCopyPackB or MlasSgemmTransposePackB. Note that in scalar,
        the packing wide is 4 for wasm taget and 16 for other taget.

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
    MLAS_FLOAT32X4(Row0Block0)
    MLAS_FLOAT32X4(Row1Block0)

#if !defined(MLAS_TARGET_WASM_SCALAR)

    MLAS_FLOAT32X4(Row0Block1)
    MLAS_FLOAT32X4(Row0Block2)
    MLAS_FLOAT32X4(Row0Block3)

    MLAS_FLOAT32X4(Row1Block1)
    MLAS_FLOAT32X4(Row1Block2)
    MLAS_FLOAT32X4(Row1Block3)

#endif

#if defined(_WIN32)

    if (!ProcessTwoRows) {
        UNREFERENCED_PARAMETER(lda);
        UNREFERENCED_PARAMETER(ldc);
    }

#endif

    do {

        MLAS_FLOAT32X4(BElements0)

#if !defined(MLAS_TARGET_WASM_SCALAR)

        MLAS_FLOAT32X4(BElements1)
        MLAS_FLOAT32X4(BElements2)
        MLAS_FLOAT32X4(BElements3)

#endif

        float Row0AElements0;
        float Row0AElements1;
        float Row1AElements0;
        float Row1AElements1;

        //
        // Clear the block accumulators.
        //

        MLAS_SET_ZERO_FLOAT32X4(Row0Block0)

#if !defined(MLAS_TARGET_WASM_SCALAR)

        MLAS_SET_ZERO_FLOAT32X4(Row0Block1)
        MLAS_SET_ZERO_FLOAT32X4(Row0Block2)
        MLAS_SET_ZERO_FLOAT32X4(Row0Block3)

#endif

        if (ProcessTwoRows) {
            MLAS_SET_ZERO_FLOAT32X4(Row1Block0)

#if !defined(MLAS_TARGET_WASM_SCALAR)

            MLAS_SET_ZERO_FLOAT32X4(Row1Block1)
            MLAS_SET_ZERO_FLOAT32X4(Row1Block2)
            MLAS_SET_ZERO_FLOAT32X4(Row1Block3)

#endif
        }

        //
        // Compute the 4x1 or 4x2 output block for wasm taeget and 16x1 or 16x2 output block for other target.
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

            MLAS_SET_VALUE_FLOAT32X4(BElements0, B, 0)
            MLAS_ADD_MUL_FLOAT32x4(Row0Block0, BElements0, Row0AElements0)

#if !defined(MLAS_TARGET_WASM_SCALAR)

            MLAS_SET_VALUE_FLOAT32X4(BElements1, B, 4)
            MLAS_SET_VALUE_FLOAT32X4(BElements2, B, 8)
            MLAS_SET_VALUE_FLOAT32X4(BElements3, B, 12)

            MLAS_ADD_MUL_FLOAT32x4(Row0Block1, BElements1, Row0AElements0)
            MLAS_ADD_MUL_FLOAT32x4(Row0Block2, BElements2, Row0AElements0)
            MLAS_ADD_MUL_FLOAT32x4(Row0Block3, BElements3, Row0AElements0)

#endif

            if (ProcessTwoRows) {
                MLAS_ADD_MUL_FLOAT32x4(Row1Block0, BElements0, Row1AElements0)

#if !defined(MLAS_TARGET_WASM_SCALAR)

                MLAS_ADD_MUL_FLOAT32x4(Row1Block1, BElements1, Row1AElements0)
                MLAS_ADD_MUL_FLOAT32x4(Row1Block2, BElements2, Row1AElements0)
                MLAS_ADD_MUL_FLOAT32x4(Row1Block3, BElements3, Row1AElements0)

#endif

            }

#if !defined(MLAS_TARGET_WASM_SCALAR)

            MLAS_SET_VALUE_FLOAT32X4(BElements0, B, 16)
            MLAS_SET_VALUE_FLOAT32X4(BElements1, B, 20)
            MLAS_SET_VALUE_FLOAT32X4(BElements2, B, 24)
            MLAS_SET_VALUE_FLOAT32X4(BElements3, B, 28)
            MLAS_ADD_MUL_FLOAT32x4(Row0Block0, BElements0, Row0AElements1)
            MLAS_ADD_MUL_FLOAT32x4(Row0Block1, BElements1, Row0AElements1)
            MLAS_ADD_MUL_FLOAT32x4(Row0Block2, BElements2, Row0AElements1)
            MLAS_ADD_MUL_FLOAT32x4(Row0Block3, BElements3, Row0AElements1)

#else //defined(MLAS_TARGET_WASM_SCALAR)

            MLAS_SET_VALUE_FLOAT32X4(BElements0, B, 4)
            MLAS_ADD_MUL_FLOAT32x4(Row0Block0, BElements0, Row0AElements1)

#endif

            if (ProcessTwoRows) {
                MLAS_ADD_MUL_FLOAT32x4(Row1Block0, BElements0, Row1AElements1)

#if !defined(MLAS_TARGET_WASM_SCALAR)

                MLAS_ADD_MUL_FLOAT32x4(Row1Block1, BElements1, Row1AElements1)
                MLAS_ADD_MUL_FLOAT32x4(Row1Block2, BElements2, Row1AElements1)
                MLAS_ADD_MUL_FLOAT32x4(Row1Block3, BElements3, Row1AElements1)

#endif

            }

            a += 2;

#if !defined(MLAS_TARGET_WASM_SCALAR)
        
            B += 32;

#else //defined(MLAS_TARGET_WASM_SCALAR)

            B += 8;

#endif

            k -= 2;
        }

        if (k > 0) {

            Row0AElements0 = a[0];

            if (ProcessTwoRows) {
                Row1AElements0 = a[lda];
            }

            MLAS_SET_VALUE_FLOAT32X4(BElements0, B, 0)
            MLAS_ADD_MUL_FLOAT32x4(Row0Block0, BElements0, Row0AElements0)

#if !defined(MLAS_TARGET_WASM_SCALAR)

            MLAS_SET_VALUE_FLOAT32X4(BElements1, B, 4)
            MLAS_SET_VALUE_FLOAT32X4(BElements2, B, 8)
            MLAS_SET_VALUE_FLOAT32X4(BElements3, B, 12)
            MLAS_ADD_MUL_FLOAT32x4(Row0Block1, BElements1, Row0AElements0)
            MLAS_ADD_MUL_FLOAT32x4(Row0Block2, BElements2, Row0AElements0)
            MLAS_ADD_MUL_FLOAT32x4(Row0Block3, BElements3, Row0AElements0)

#endif

            if (ProcessTwoRows) {
                MLAS_ADD_MUL_FLOAT32x4(Row1Block0, BElements0, Row1AElements0)

#if !defined(MLAS_TARGET_WASM_SCALAR)

                MLAS_ADD_MUL_FLOAT32x4(Row1Block1, BElements1, Row1AElements0)
                MLAS_ADD_MUL_FLOAT32x4(Row1Block2, BElements2, Row1AElements0)
                MLAS_ADD_MUL_FLOAT32x4(Row1Block3, BElements3, Row1AElements0)

#endif

            }

#if !defined(MLAS_TARGET_WASM_SCALAR)

            B += 16;

#else

            B += 4;

#endif
        }

        //
        // Multiply by the alpha value.
        //

        MLAS_MUL_FLOAT32x4(Row0Block0, alpha)

#if !defined(MLAS_TARGET_WASM_SCALAR)

        MLAS_MUL_FLOAT32x4(Row0Block1, alpha)
        MLAS_MUL_FLOAT32x4(Row0Block2, alpha)
        MLAS_MUL_FLOAT32x4(Row0Block3, alpha)

#endif

        if (ProcessTwoRows) {
            MLAS_MUL_FLOAT32x4(Row1Block0, alpha)

#if !defined(MLAS_TARGET_WASM_SCALAR)

            MLAS_MUL_FLOAT32x4(Row1Block1, alpha)
            MLAS_MUL_FLOAT32x4(Row1Block2, alpha)
            MLAS_MUL_FLOAT32x4(Row1Block3, alpha)

#endif
        }

#if !defined(MLAS_TARGET_WASM_SCALAR)

        if (CountN >= 16) {

#else

        if (CountN >= 4) {

#endif

            //
            // Store the entire output block.
            //

            if (!ZeroMode) {
                MLAS_ADD_FLOAT32x4(Row0Block0, C, 0)

#if !defined(MLAS_TARGET_WASM_SCALAR)

                MLAS_ADD_FLOAT32x4(Row0Block1, C, 4)
                MLAS_ADD_FLOAT32x4(Row0Block2, C, 8)
                MLAS_ADD_FLOAT32x4(Row0Block3, C, 12)

#endif

            }

            MLAS_SET_RESULT_FLOAT32x4(C, 0, Row0Block0)

#if !defined(MLAS_TARGET_WASM_SCALAR)

            MLAS_SET_RESULT_FLOAT32x4(C, 4, Row0Block1)
            MLAS_SET_RESULT_FLOAT32x4(C, 8, Row0Block2)
            MLAS_SET_RESULT_FLOAT32x4(C, 12, Row0Block3)

#endif

            if (ProcessTwoRows) {

                if (!ZeroMode) {
                    MLAS_ADD_FLOAT32x4(Row1Block0, C, ldc)

#if !defined(MLAS_TARGET_WASM_SCALAR)

                    MLAS_ADD_FLOAT32x4(Row1Block1, C, ldc + 4)
                    MLAS_ADD_FLOAT32x4(Row1Block2, C, ldc + 8)
                    MLAS_ADD_FLOAT32x4(Row1Block3, C, ldc + 12)
    
#endif

                }

                MLAS_SET_RESULT_FLOAT32x4(C, ldc, Row1Block0)

#if !defined(MLAS_TARGET_WASM_SCALAR)

                MLAS_SET_RESULT_FLOAT32x4(C, ldc + 4, Row1Block1)
                MLAS_SET_RESULT_FLOAT32x4(C, ldc + 8, Row1Block2)
                MLAS_SET_RESULT_FLOAT32x4(C, ldc + 12, Row1Block3)

#endif

            }

        } else {

            //
            // Store the partial output block.
            //

#if !defined(MLAS_TARGET_WASM_SCALAR)

            if ((CountN & 8) != 0) {

                if (!ZeroMode) {

                    MLAS_ADD_FLOAT32x4(Row0Block0, C, 0)
                    MLAS_ADD_FLOAT32x4(Row0Block1, C, 4)
                }

                MLAS_SET_RESULT_FLOAT32x4(C, 0, Row0Block0)
                MLAS_SET_RESULT_FLOAT32x4(C, 4, Row0Block1)
                MLAS_SET_VECTOR_FLOAT32X4(Row0Block0, Row0Block2)
                MLAS_SET_VECTOR_FLOAT32X4(Row0Block1, Row0Block3)

                if (ProcessTwoRows) {

                    if (!ZeroMode) {
                        MLAS_ADD_FLOAT32x4(Row1Block0, C, ldc)
                        MLAS_ADD_FLOAT32x4(Row1Block1, C, ldc+4)
                    }

                    MLAS_SET_RESULT_FLOAT32x4(C, ldc, Row1Block0)
                    MLAS_SET_RESULT_FLOAT32x4(C, ldc+4, Row1Block1)
                    MLAS_SET_VECTOR_FLOAT32X4(Row1Block0, Row1Block2)
                    MLAS_SET_VECTOR_FLOAT32X4(Row1Block1, Row1Block3)
                }

                C += 8;
            }
            if ((CountN & 4) != 0) {

                if (!ZeroMode) {
                    MLAS_ADD_FLOAT32x4(Row0Block0, C, 0)
                }

                MLAS_SET_RESULT_FLOAT32x4(C, 0, Row0Block0)
                MLAS_SET_VECTOR_FLOAT32X4(Row0Block0, Row0Block1)

                if (ProcessTwoRows) {

                    if (!ZeroMode) {
                        MLAS_ADD_FLOAT32x4(Row1Block0, C, ldc)

                    }

                    MLAS_SET_RESULT_FLOAT32x4(C, ldc, Row1Block0)
                    MLAS_SET_VECTOR_FLOAT32X4(Row1Block0, Row1Block1)
                }

                C += 4;
            }

#endif

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

#if !defined(MLAS_TARGET_WASM_SCALAR)

        C += 16;
        CountN -= 16;

#else
        C += 4;
        CountN -= 4;

#endif

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
