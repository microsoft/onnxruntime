/*++

Copyright (c) Microsoft Corporation. All rights reserved.
Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the MIT License.

Module Name:

    sbgemm.cpp

Abstract:

    This module implements the bfloat16 half precision matrix/matrix multiply
    operation (SBGEMM).

--*/

#include "mlasi.h"
//
// Define the number of rows from matrix A to transpose to a local buffer.
//
// N.B. aarch64 sbgemm kernel processes 8 rows in one loop
//

#define MLAS_SGEMM_TRANSA_ROWS              8

//
// Define the parameters to execute segments of a SBGEMM operation on worker
// threads.
//

void
MlasSbgemmMultiplyBeta(
    float* C,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    float beta
    )
/*++

Routine Description:

    This routine multiplies all elements of the output matrix by the beta
    scalar value.

Arguments:

    C - Supplies the address of matrix C.

    CountM - Supplies the number of rows from matrix C.

    CountN - Supplies the number of columns from matrix C.

    ldc - Supplies the first dimension of matrix C.

    beta - Supplies the scalar beta multiplier (see SGEMM definition).

Return Value:

    None.

--*/
{

    MLAS_FLOAT32X4 BetaBroadcast = MlasBroadcastFloat32x4(beta);

    while (CountM-- > 0) {

        float* c = C;
        size_t n = CountN;

        while (n >= 4) {
            MlasStoreFloat32x4(c, MlasMultiplyFloat32x4(MlasLoadFloat32x4(c), BetaBroadcast));
            c += 4;
            n -= 4;
        }

        while (n > 0) {
            *c = *c * beta;
            c += 1;
            n -= 1;
        }

        C += ldc;
    }
}

void
MlasSbgemmTransposeA(
    float* D,
    const float* A,
    size_t lda,
    size_t CountY,
    size_t CountX
    )
/*++

Routine Description:

    This routine transposes elements from the source matrix to the destination
    buffer.

Arguments:

    D - Supplies the address of the destination buffer.

    A - Supplies the address of the source matrix.

    lda - Supplies the number of elements per row of the source matrix.

    CountY - Supplies the number of columns of the source matrix to transpose.

    CountX - Supplies the number of rows of the source matrix to transpose.

Return Value:

    None.

--*/
{

    size_t ldd = CountX;

    //
    // Transpose elements from matrix A into the destination buffer 4 columns
    // at a time.
    //

    while (CountX >= 4) {

        float* d = D;
        const float* a = A;
        size_t y = CountY;

        do {

            float t0 = a[0];
            float t1 = a[lda];
            float t2 = a[lda * 2];
            float t3 = a[lda * 3];

            d[0] = t0;
            d[1] = t1;
            d[2] = t2;
            d[3] = t3;

            d += ldd;
            a += 1;
            y--;

        } while (y > 0);

        D += 4;
        A += lda * 4;
        CountX -= 4;
    }

    //
    // Transpose elements from matrix A into the destination buffer for the
    // remaining columns.
    //

    if (CountX >= 2) {

        float* d = D;
        const float* a = A;
        size_t y = CountY;

        do {

            float t0 = a[0];
            float t1 = a[lda];

            d[0] = t0;
            d[1] = t1;

            d += ldd;
            a += 1;
            y--;

        } while (y > 0);

        D += 2;
        A += lda * 2;
        CountX -= 2;
    }

    if (CountX >= 1) {

        float* d = D;
        const float* a = A;
        size_t y = CountY;

        do {

            d[0] = a[0];

            d += ldd;
            a += 1;
            y--;

        } while (y > 0);
    }
}

void
MlasSbgemmCopyPackB(
    bfloat16_t* D,
    const float* B,
    size_t ldb,
    size_t CountX,
    size_t CountY
    )
/*++

Routine Description:

    This routine copies elements from the source matrix to the destination
    packed buffer.

    4x2 elements from the source matrix are unrolled to be physically
    contiguous for better locality inside the SBGEMM kernels. The remaining
    rows and columns are padded to 4 and 2 alignment.

Arguments:

    D - Supplies the address of the destination packed buffer.

    B - Supplies the address of the source matrix.

    ldb - Supplies the number of elements per row of the source matrix.

    CountX - Supplies the number of columns of the source matrix to copy.

    CountY - Supplies the number of rows of the source matrix to copy.

Return Value:

    None.

--*/
{
    //
    // Copy data from matrix B into the destination buffer 4x2 blocks at a
    // time.
    //
    //
    while (CountX >= 8) {
        const float* b = B;
        int y = CountY;

        while (y > 0) {
            MLAS_FLOAT32X4 t0_l = MlasZeroFloat32x4();
            MLAS_FLOAT32X4 t0_h = MlasZeroFloat32x4();
            MLAS_FLOAT32X4 t1_l = MlasZeroFloat32x4();
            MLAS_FLOAT32X4 t1_h = MlasZeroFloat32x4();
            MLAS_FLOAT32X4 t2_l = MlasZeroFloat32x4();
            MLAS_FLOAT32X4 t2_h = MlasZeroFloat32x4();
            MLAS_FLOAT32X4 t3_l = MlasZeroFloat32x4();
            MLAS_FLOAT32X4 t3_h = MlasZeroFloat32x4();

            if (y >= 4) {
                t0_l = MlasLoadFloat32x4(&b[ldb * 0]);
                t0_h = MlasLoadFloat32x4(&b[ldb * 0 + 4]);
                t1_l = MlasLoadFloat32x4(&b[ldb * 1]);
                t1_h = MlasLoadFloat32x4(&b[ldb * 1 + 4]);
                t2_l = MlasLoadFloat32x4(&b[ldb * 2]);
                t2_h = MlasLoadFloat32x4(&b[ldb * 2 + 4]);
                t3_l = MlasLoadFloat32x4(&b[ldb * 3]);
                t3_h = MlasLoadFloat32x4(&b[ldb * 3 + 4]);
            } else {
                switch (y) {
                    case 3:
                        t0_l = MlasLoadFloat32x4(&b[ldb * 0]);
                        t0_h = MlasLoadFloat32x4(&b[ldb * 0 + 4]);
                        t1_l = MlasLoadFloat32x4(&b[ldb * 1]);
                        t1_h = MlasLoadFloat32x4(&b[ldb * 1 + 4]);
                        t2_l = MlasLoadFloat32x4(&b[ldb * 2]);
                        t2_h = MlasLoadFloat32x4(&b[ldb * 2 + 4]);
                    break;
                    case 2:
                        t0_l = MlasLoadFloat32x4(&b[ldb * 0]);
                        t0_h = MlasLoadFloat32x4(&b[ldb * 0 + 4]);
                        t1_l = MlasLoadFloat32x4(&b[ldb * 1]);
                        t1_h = MlasLoadFloat32x4(&b[ldb * 1 + 4]);
                    break;
                    case 1:
                        t0_l = MlasLoadFloat32x4(&b[ldb * 0]);
                        t0_h = MlasLoadFloat32x4(&b[ldb * 0 + 4]);
                    break;
                }
            }

            float32x4x2_t z0_l = vzipq_f32(t0_l, t2_l);
            float32x4x2_t z1_l = vzipq_f32(t1_l, t3_l);
            float32x4x2_t o0_l = vzipq_f32(z0_l.val[0], z1_l.val[0]);
            float32x4x2_t o1_l = vzipq_f32(z0_l.val[1], z1_l.val[1]);
            t0_l = o0_l.val[0];
            t1_l = o0_l.val[1];
            t2_l = o1_l.val[0];
            t3_l = o1_l.val[1];

            bfloat16x8_t t0t1_l_4h = vcvtq_low_bf16_f32(t0_l);
            bfloat16x8_t t0t1_l_8h = vcvtq_high_bf16_f32(t0t1_l_4h, t1_l);

            bfloat16x8_t t2t3_l_4h = vcvtq_low_bf16_f32(t2_l);
            bfloat16x8_t t2t3_l_8h = vcvtq_high_bf16_f32(t2t3_l_4h, t3_l);

            vst1q_bf16(&D[0], t0t1_l_8h);
            vst1q_bf16(&D[8], t2t3_l_8h);

            float32x4x2_t z0_h = vzipq_f32(t0_h, t2_h);
            float32x4x2_t z1_h = vzipq_f32(t1_h, t3_h);
            float32x4x2_t o0_h = vzipq_f32(z0_h.val[0], z1_h.val[0]);
            float32x4x2_t o1_h = vzipq_f32(z0_h.val[1], z1_h.val[1]);
            t0_h = o0_h.val[0];
            t1_h = o0_h.val[1];
            t2_h = o1_h.val[0];
            t3_h = o1_h.val[1];

            bfloat16x8_t t0t1_h_4h = vcvtq_low_bf16_f32(t0_h);
            bfloat16x8_t t0t1_h_8h = vcvtq_high_bf16_f32(t0t1_h_4h, t1_h);

            bfloat16x8_t t2t3_h_4h = vcvtq_low_bf16_f32(t2_h);
            bfloat16x8_t t2t3_h_8h = vcvtq_high_bf16_f32(t2t3_h_4h, t3_h);

            vst1q_bf16(&D[16], t0t1_h_8h);
            vst1q_bf16(&D[24], t2t3_h_8h);

            D += 32;
            b += ldb*4;
            y -= 4;
        };
        B += 8;
        CountX -= 8;
    }

    //
    // Special case the handling of the remaining columns less than 8 elements
    // wide.
    //
    if (CountX > 0) {
        int y = CountY;
        while (y > 0) {
            const float* b = B;
	    size_t b_inc = 0;

            if ((CountX & 4) != 0) {
                MLAS_FLOAT32X4 t0 = MlasZeroFloat32x4();
                MLAS_FLOAT32X4 t1 = MlasZeroFloat32x4();
                MLAS_FLOAT32X4 t2 = MlasZeroFloat32x4();
                MLAS_FLOAT32X4 t3 = MlasZeroFloat32x4();
                if ( y >= 4) {
                    t0 = MlasLoadFloat32x4(&b[ldb * 0]);
                    t1 = MlasLoadFloat32x4(&b[ldb * 1]);
                    t2 = MlasLoadFloat32x4(&b[ldb * 2]);
                    t3 = MlasLoadFloat32x4(&b[ldb * 3]);
                } else {
                    switch(y) {
                        case 3:
                            t0 = MlasLoadFloat32x4(&b[ldb * 0]);
                            t1 = MlasLoadFloat32x4(&b[ldb * 1]);
                            t2 = MlasLoadFloat32x4(&b[ldb * 2]);
                        break;
                        case 2:
                            t0 = MlasLoadFloat32x4(&b[ldb * 0]);
                            t1 = MlasLoadFloat32x4(&b[ldb * 1]);
                        break;
                        case 1:
                            t0 = MlasLoadFloat32x4(&b[ldb * 0]);
                        break;
                    }
                }

                float32x4x2_t z0 = vzipq_f32(t0, t2);
                float32x4x2_t z1 = vzipq_f32(t1, t3);
                float32x4x2_t o0 = vzipq_f32(z0.val[0], z1.val[0]);
                float32x4x2_t o1 = vzipq_f32(z0.val[1], z1.val[1]);

                t0 = o0.val[0];
                t1 = o0.val[1];
                t2 = o1.val[0];
                t3 = o1.val[1];

                bfloat16x8_t t0t1_4h = vcvtq_low_bf16_f32(t0);
                bfloat16x8_t t0t1_8h = vcvtq_high_bf16_f32(t0t1_4h, t1);

                bfloat16x8_t t2t3_4h = vcvtq_low_bf16_f32(t2);
                bfloat16x8_t t2t3_8h = vcvtq_high_bf16_f32(t2t3_4h, t3);

                vst1q_bf16(&D[0], t0t1_8h);
                vst1q_bf16(&D[8], t2t3_8h);

                D += 16;
                b += 4;
                b_inc += 4;
	    }
	    if ((CountX & 2) != 0) {
                float32x2_t t0 = {0x0, 0x0};
                float32x2_t t1 = {0x0, 0x0};
                float32x2_t t2 = {0x0, 0x0};
                float32x2_t t3 = {0x0, 0x0};

                if (y >= 4) {
                    t0 = vld1_f32(&b[ldb * 0]);
                    t1 = vld1_f32(&b[ldb * 1]);
                    t2 = vld1_f32(&b[ldb * 2]);
                    t3 = vld1_f32(&b[ldb * 3]);
                } else {
                    switch(y) {
                        case 3:
                            t0 = vld1_f32(&b[ldb * 0]);
                            t1 = vld1_f32(&b[ldb * 1]);
                            t2 = vld1_f32(&b[ldb * 2]);
                        break;
                        case 2:
                            t0 = vld1_f32(&b[ldb * 0]);
                            t1 = vld1_f32(&b[ldb * 1]);
                        break;
                        case 1:
                            t0 = vld1_f32(&b[ldb * 0]);
                        break;
                    }
                }

                float32x2x2_t z0 = vzip_f32(t0, t2);
                float32x2x2_t z1 = vzip_f32(t1, t3);
                float32x2x2_t o0 = vzip_f32(z0.val[0], z1.val[0]);
                float32x2x2_t o1 = vzip_f32(z0.val[1], z1.val[1]);

                float32x4_t tt0 = vcombine_f32(o0.val[0], o0.val[1]);
                float32x4_t tt1 = vcombine_f32(o1.val[0], o1.val[1]);

                bfloat16x8_t t_4h = vcvtq_low_bf16_f32(tt0);
                bfloat16x8_t t_8h = vcvtq_high_bf16_f32(t_4h, tt1);

                vst1q_bf16(&D[0], t_8h);

                D += 8;
                b += 2;
		b_inc += 2;
            }
            if ((CountX & 1) != 0) {
                float  a = 0.0f;
                float  b = 0.0f;
                float  c = 0.0f;
                float  d = 0.0f;

                if (y >=4 ) {
                    a = *(float*)(&B[ldb * 0 + b_inc]);
                    b = *(float*)(&B[ldb * 1 + b_inc]);
                    c = *(float*)(&B[ldb * 2 + b_inc]);
                    d = *(float*)(&B[ldb * 3 + b_inc]);
                 } else {
                    switch(y) {
                        case 3:
                            a = *(float*)(&B[ldb * 0 + b_inc]);
                            b = *(float*)(&B[ldb * 1 + b_inc]);
                            c = *(float*)(&B[ldb * 2 + b_inc]);
                        break;
                        case 2:
                            a = *(float*)(&B[ldb * 0 + b_inc]);
                            b = *(float*)(&B[ldb * 1 + b_inc]);
                        break;
                        case 1:
                            a = *(float*)(&B[ldb * 0 + b_inc]);
                        break;
                    }
                }
                float32x2_t t0 = {a, 0x0};
                float32x2_t t1 = {b, 0x0};
                float32x2_t t2 = {c, 0x0};
                float32x2_t t3 = {d, 0x0};

                float32x2x2_t z0 = vzip_f32(t0, t2);
                float32x2x2_t z1 = vzip_f32(t1, t3);
                float32x2x2_t o0 = vzip_f32(z0.val[0], z1.val[0]);
                float32x2x2_t o1 = vzip_f32(z0.val[1], z1.val[1]);

                float32x4_t tt0 = vcombine_f32(o0.val[0], o0.val[1]);
                float32x4_t tt1 = vcombine_f32(o1.val[0], o1.val[1]);

                bfloat16x8_t t_4h = vcvtq_low_bf16_f32(tt0);
                bfloat16x8_t t_8h = vcvtq_high_bf16_f32(t_4h, tt1);

                vst1q_bf16(&D[0], t_8h);

                D += 8;
                b += 1;
		b_inc += 1;
            }
            B += 4*ldb;
            y -= 4;
            }
       }
}

MLAS_FORCEINLINE
float*
MlasSbgemmKernelLoop(
    const float* A,
    const bfloat16_t* B,
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

    This routine steps through the rows of the input and output matrices calling
    the kernel until all rows have been processed.

Arguments:

    A - Supplies the address of matrix A.

    B - Supplies the address of matrix B. The matrix data has been packed using
        MlasSgemmCopyPackB or MlasSgemmTransposePackB.

    C - Supplies the address of matrix C.

    CountK - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    CountM - Supplies the number of rows from matrix A and matrix C to iterate
        over.

    CountN - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    lda - Supplies the first dimension of matrix A.

    ldc - Supplies the first dimension of matrix C.

    alpha - Supplies the scalar alpha multiplier (see SGEMM definition).

    ZeroMode - Supplies true if the output matrix must be zero initialized,
        else false if the output matrix is accumulated into.

Return Value:

    Returns the next address of matrix C.

--*/
{
    while (CountM > 0) {
        size_t RowsHandled;
        if (ZeroMode) {
            RowsHandled = MlasSbgemmKernelZero(A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
        } else {
            RowsHandled = MlasSbgemmKernelAdd(A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
        }
        C += ldc * RowsHandled;
        A += lda * RowsHandled;
        CountM -= RowsHandled;
    }

return C;
}

void
MlasSbgemmOperation(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    const float* A,
    size_t lda,
    const float* B,
    size_t ldb,
    float beta,
    float* C,
    size_t ldc
    )
/*++

Routine Description:

    This routine implements the bfloat16 half precision matrix/matrix multiply
    operation (SBGEMM).

Arguments:

    TransA - Supplies the transpose operation for matrix A.

    TransB - Supplies the transpose operation for matrix B.

    M - Supplies the number of rows of matrix A and matrix C.

    N - Supplies the number of columns of matrix B and matrix C.

    K - Supplies the number of columns of matrix A and the number of rows of
        matrix B.

    alpha - Supplies the scalar alpha multiplier (see SGEMM definition).

    A - Supplies the address of matrix A.

    lda - Supplies the first dimension of matrix A.

    B - Supplies the address of matrix B.

    ldb - Supplies the first dimension of matrix B.

    beta - Supplies the scalar beta multiplier (see SGEMM definition).

    C - Supplies the address of matrix C.

    ldc - Supplies the first dimension of matrix C.

Return Value:

    None.

--*/
{
    float PanelA[MLAS_SGEMM_TRANSA_ROWS * MLAS_SGEMM_STRIDEK];
    MLAS_DECLSPEC_ALIGN(bfloat16_t PanelB[MLAS_SGEMM_STRIDEN * MLAS_SGEMM_STRIDEK], 16 * sizeof(bfloat16_t));

    //
    // Compute the strides to step through slices of the input matrices.
    //
    // Expand the N stride if K is small or expand the K stride if N is small
    // for better utilization of the B panel. Avoid changing the K stride if
    // the A panel needs to be used for transposing.
    //

    size_t StrideN = MLAS_SGEMM_STRIDEN;
    size_t StrideK = MLAS_SGEMM_STRIDEK;

    if (N >= K) {
        while (StrideK / 2 >= K) {
            StrideN *= 2;
            StrideK /= 2;
        }
    } else if (TransA == CblasNoTrans) {
        while (StrideN > 16 && StrideN / 2 >= N) {
            StrideK *= 2;
            StrideN /= 2;
        }
    }

    //
    // Step through each slice of matrix B along the N dimension.
    //

    size_t CountN;

    for (size_t n = 0; n < N; n += CountN) {

        CountN = std::min(N - n, StrideN);

        //
        // Multiply the output matrix by beta as needed.
        //

        if (beta != 0.0f && beta != 1.0f) {
            MlasSbgemmMultiplyBeta(C + n, M, CountN, ldc, beta);
        }

        //
        // Step through each slice of matrix B along the K dimension.
        //

        size_t CountK;
        bool ZeroMode = (beta == 0.0f);

        for (size_t k = 0; k < K; k += CountK) {

            CountK = std::min(K - k, StrideK);

            //
            // Copy or transpose a panel of matrix B to a local packed buffer.
            //

            if (TransB == CblasNoTrans) {
                MlasSbgemmCopyPackB(PanelB, B + n + k * ldb, ldb, CountN, CountK);
            }

            //
            // Step through each slice of matrix A along the M dimension.
            //

            float* c = C + n;

            if (TransA == CblasNoTrans) {
                MlasSbgemmKernelLoop(A + k, PanelB, c, CountK, M, CountN, lda, ldc, alpha, ZeroMode);
            } else {

                const float* a = A + k * lda;
                size_t RowsRemaining = M;

                while (RowsRemaining > 0) {

                    //
                    // Transpose elements from matrix A into a local buffer.
                    //

                    size_t RowsTransposed = std::min(RowsRemaining, size_t(MLAS_SGEMM_TRANSA_ROWS));

                    MlasSbgemmTransposeA(PanelA, a, lda, RowsTransposed, CountK);

                    RowsRemaining -= RowsTransposed;
                    a += RowsTransposed;

                    //
                    // Step through the rows of the local buffer.
                    //
                    c = MlasSbgemmKernelLoop(PanelA, PanelB, c, CountK, RowsTransposed, CountN, CountK, ldc, alpha, ZeroMode);
                }
            }

            ZeroMode = false;
        }
    }
}


void
MlasSbgemmPackedOperation(
    CBLAS_TRANSPOSE TransA,
    size_t M,
    size_t RangeStartN,
    size_t RangeCountN,
    size_t K,
    float alpha,
    const float* A,
    size_t lda,
    const void* PackedB,
    size_t AlignedN,
    float beta,
    float* C,
    size_t ldc
    )
/*++

Routine Description:

    This routine implements the bfloat16 half precision matrix/matrix multiply
    operation (SBGEMM).

Arguments:

    TransA - Supplies the transpose operation for matrix A.

    M - Supplies the number of rows of matrix A and matrix C.

    RangeStartN - Supplies the starting column from packed matrix B.

    RangeCountN - Supplies the number of columns of matrix B and matrix C.

    K - Supplies the number of columns of matrix A and the number of rows of
        matrix B.

    alpha - Supplies the scalar alpha multiplier (see SGEMM definition).

    A - Supplies the address of matrix A.

    lda - Supplies the first dimension of matrix A.

    PackedB - Supplies the address of packed matrix B.

    AlignedN - Supplies the total number of aligned columns for packed matrix B.

    ldb - Supplies the first dimension of matrix B.

    beta - Supplies the scalar beta multiplier (see SGEMM definition).

    C - Supplies the address of matrix C.

    ldc - Supplies the first dimension of matrix C.

Return Value:

    None.

--*/

{
    float PanelA[MLAS_SGEMM_TRANSA_ROWS * MLAS_SGEMM_PACKED_STRIDEK];

    //
    // Step through each slice of matrix B along the N dimension.
    //

    size_t CountN;

    for (size_t n = 0; n < RangeCountN; n += CountN) {

        const size_t SliceStartN = RangeStartN + n;

        CountN = std::min(RangeCountN - n, size_t(MLAS_SGEMM_PACKED_STRIDEN));

        //
        // Multiply the output matrix by beta as needed.
        //

        if (beta != 0.0f && beta != 1.0f) {
            MlasSbgemmMultiplyBeta(C + n, M, CountN, ldc, beta);
        }

        //
        // Step through each slice of matrix B along the K dimension.
        //

        size_t CountK;
        bool ZeroMode = (beta == 0.0f);

        for (size_t k = 0; k < K; k += CountK) {

            CountK = std::min(K - k, size_t(MLAS_SGEMM_PACKED_STRIDEK));

            //
            // Step through each slice of matrix A along the M dimension.
            //

            const bfloat16_t* pb = (const bfloat16_t*)PackedB + AlignedN * k + CountK * SliceStartN;
            float* c = C + n;

            if (TransA == CblasNoTrans) {

                MlasSbgemmKernelLoop(A + k, pb, c, CountK, M, CountN, lda, ldc, alpha, ZeroMode);
            } else {

                const float* a = A + k * lda;
                size_t RowsRemaining = M;

                while (RowsRemaining > 0) {

                    //
                    // Transpose elements from matrix A into a local buffer.
                    //

                    size_t RowsTransposed = std::min(RowsRemaining, size_t(MLAS_SGEMM_TRANSA_ROWS));

                    MlasSbgemmTransposeA(PanelA, a, lda, RowsTransposed, CountK);

                    RowsRemaining -= RowsTransposed;
                    a += RowsTransposed;

                    //
                    // Step through the rows of the local buffer.
                    //
                    c = MlasSbgemmKernelLoop(PanelA, pb, c, CountK, RowsTransposed, CountN, CountK, ldc, alpha, ZeroMode);
                }
            }

            ZeroMode = false;
        }
    }
}


void
MlasSbgemmThreaded(
    const ptrdiff_t ThreadCountM,
    const ptrdiff_t ThreadCountN,
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB,
    const size_t M,
    const size_t N,
    const size_t K,

    const MLAS_SGEMM_DATA_PARAMS* DataParams,
    ptrdiff_t ThreadId
    )
/*++

Routine Description:

    This routine is invoked from a worker thread to execute a segment of a
    SBGEMM operation.

Arguments:

    ThreadCountM - Supplies the total thread partition on the M dimension.

    ThreadCountN - Supplies the total thread partition on the N dimension.

    TransA - Supplies the transpose operation on A matrix

    TransB - Supplies the transpose operation on B matrix

    M, N, K - Supplies the shape of the multiplication

    DataParams - Supplies the data position and layout of the matrices

    ThreadId - Supplies the current index of the threaded operation.

Return Value:

    None.

--*/
{
    const ptrdiff_t ThreadIdM = ThreadId / ThreadCountN;
    const ptrdiff_t ThreadIdN = ThreadId % ThreadCountN;

    //
    // Partition the operation along the M dimension.
    //

    size_t RangeStartM;
    size_t RangeCountM;

    MlasPartitionWork(ThreadIdM, ThreadCountM, M, &RangeStartM, &RangeCountM);

    //
    // Partition the operation along the N dimension.
    //

    size_t RangeStartN;
    size_t RangeCountN;

    const size_t BlockedN = (N + MLAS_SGEMM_STRIDEN_THREAD_ALIGN - 1) /
        MLAS_SGEMM_STRIDEN_THREAD_ALIGN;

    MlasPartitionWork(ThreadIdN, ThreadCountN, BlockedN, &RangeStartN,
        &RangeCountN);

    RangeStartN *= MLAS_SGEMM_STRIDEN_THREAD_ALIGN;
    RangeCountN *= MLAS_SGEMM_STRIDEN_THREAD_ALIGN;

    RangeCountN = std::min(N - RangeStartN, RangeCountN);

    //
    // Dispatch the partitioned operation.
    //
    const size_t lda = DataParams->lda;
    const size_t ldc = DataParams->ldc;
    const float* A = DataParams->A + RangeStartM * ((TransA == CblasNoTrans) ? lda : 1);
    float* C = DataParams->C + RangeStartM * ldc + RangeStartN;


    if (DataParams->BIsPacked) {

	    MlasSbgemmPackedOperation(TransA, RangeCountM, RangeStartN, RangeCountN,
            K, DataParams->alpha, A, lda, DataParams->B,
            BlockedN * MLAS_SGEMM_STRIDEN_THREAD_ALIGN, DataParams->beta, C, ldc);

    } else {

        const size_t ldb = DataParams->ldb;

        const float* B = (const float*)DataParams->B + RangeStartN * ((TransB == CblasNoTrans) ? 1 : ldb);

        MlasSbgemmOperation(TransA, TransB, RangeCountM, RangeCountN, K,
            DataParams->alpha, A, lda, B, ldb, DataParams->beta, C, ldc);
    }

}
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
// Chance of arithmetic overflow could be reduced
#pragma warning(disable : 26451)
#endif
void
MLASCALL
MlasSBGemmBatch(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    const MLAS_SGEMM_DATA_PARAMS* Data,
    size_t BatchSize,
    MLAS_THREADPOOL* ThreadPool
    )
{
    //
    // Compute the number of target threads given the complexity of the SGEMM
    // operation. Small requests should run using the single threaded path.
    //

    const double Complexity = double(M) * double(N) * double(K);

    ptrdiff_t TargetThreadCount;


    if (Complexity < double(MLAS_SBGEMM_THREAD_COMPLEXITY * GetMlasPlatform().MaximumThreadCount)) {
        TargetThreadCount = ptrdiff_t(Complexity / double(MLAS_SGEMM_THREAD_COMPLEXITY)) + 1;
    } else {
        TargetThreadCount = GetMlasPlatform().MaximumThreadCount;
    }

    ptrdiff_t MaximumThreadCount = MlasGetMaximumThreadCount(ThreadPool);

    if (TargetThreadCount >= MaximumThreadCount) {
        TargetThreadCount = MaximumThreadCount;
    }

    //
    // Segment the operation across multiple threads.
    //
    // N.B. Currently, the operation is segmented as a 1D partition, which
    // works okay for operations involving skinny matrices.
    //

    ptrdiff_t ThreadsPerGemm = (TargetThreadCount + BatchSize - 1) / BatchSize;
    ptrdiff_t ThreadCountM;
    ptrdiff_t ThreadCountN;

    if (N > M) {

        const size_t BlockedN = (N + MLAS_SGEMM_STRIDEN_THREAD_ALIGN - 1) /
            MLAS_SGEMM_STRIDEN_THREAD_ALIGN;

        if (size_t(ThreadsPerGemm) > BlockedN) {
            ThreadsPerGemm = ptrdiff_t(BlockedN);
        }

        ThreadCountM = 1;
        ThreadCountN = ThreadsPerGemm;

    } else {

        if (size_t(ThreadsPerGemm) > M) {
            ThreadsPerGemm = ptrdiff_t(M);
        }

        ThreadCountM = ThreadsPerGemm;
        ThreadCountN = 1;
    }

    MlasTrySimpleParallel(ThreadPool,
        ThreadsPerGemm * static_cast<ptrdiff_t>(BatchSize),
        [=](ptrdiff_t tid)
    {
        ptrdiff_t GemmIdx = tid / ThreadsPerGemm;
        ptrdiff_t ThreadIdx = tid % ThreadsPerGemm;
        MlasSbgemmThreaded(ThreadCountM, ThreadCountN,
            TransA, TransB, M, N, K, &(Data[GemmIdx]), ThreadIdx);
    });
}
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif

size_t
MLASCALL
MlasSBGemmPackBSize(
    size_t N,
    size_t K
    )
/*++

Routine Description:

    This routine computes the length in bytes for the packed matrix B buffer.

Arguments:

    N - Supplies the number of columns of matrix B.

    K - Supplies the number of rows of matrix B.

Return Value:

    Returns the size in bytes for the packed matrix B buffer.

--*/
{
    //
    // Compute the number of bytes required to hold the packed buffer.
    //

    const size_t AlignedN =
        (N + MLAS_SGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_SGEMM_STRIDEN_THREAD_ALIGN - 1);

    const size_t BytesRequired = AlignedN * K * sizeof(bfloat16_t);
    const size_t BufferAlignment = MlasGetPreferredBufferAlignment();
    const size_t AlignedBytesRequired = (BytesRequired + BufferAlignment - 1) &
        ~(BufferAlignment - 1);

    return AlignedBytesRequired;
}

void
MLASCALL
MlasSBGemmPackB(
    CBLAS_TRANSPOSE TransB,
    size_t N,
    size_t K,
    const float* B,
    size_t ldb,
    void* PackedB
    )
/*++

Routine Description:

    This routine packs the contents of matrix B to the destination buffer. The
    destination buffer should be sized based on MlasSBGemmPackBSize(). For best
    performance, the destination buffer should be aligned to the value returned
    from MlasGetPreferredBufferAlignment().

Arguments:

    TransB - Supplies the transpose operation for matrix B.

    N - Supplies the number of columns of matrix B.

    K - Supplies the number of rows of matrix B.

    B - Supplies the address of matrix B.

    ldb - Supplies the first dimension of matrix B.

    PackedB - Supplies the address of packed matrix B.

Return Value:

    None.

--*/
{

    const size_t AlignedN =
        (N + MLAS_SGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_SGEMM_STRIDEN_THREAD_ALIGN - 1);

    //
    // Step through each slice of matrix B along the K dimension.
    //

    size_t CountK;

    for (size_t k = 0; k < K; k += CountK) {

        CountK = std::min(K - k, size_t(MLAS_SGEMM_PACKED_STRIDEK));

        if (TransB == CblasNoTrans) {
                MlasSbgemmCopyPackB((bfloat16_t*)PackedB, B + k * ldb, ldb, N, CountK);
        }

        PackedB = (bfloat16_t*)PackedB + AlignedN * CountK;
    }
}
