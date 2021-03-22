/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    dgemm.cpp

Abstract:

    This module implements the double precision matrix/matrix multiply
    operation (DGEMM).

--*/

#include "mlasi.h"

//
// Define the number of rows from matrix A to transpose to a local buffer.
//
// N.B. AVX processes a maximum of 4 rows, FMA3 processes a maximum of 6
// rows, and AVX512F processes a maximum of 12 rows.
//

#define MLAS_DGEMM_TRANSA_ROWS              12

//
// Define the parameters to execute segments of a DGEMM operation on worker
// threads.
//

struct MLAS_DGEMM_WORK_BLOCK {
    ptrdiff_t ThreadCountM;
    ptrdiff_t ThreadCountN;
    CBLAS_TRANSPOSE TransA;
    CBLAS_TRANSPOSE TransB;
    size_t M;
    size_t N;
    size_t K;
    const double* A;
    size_t lda;
    const double* B;
    size_t ldb;
    double* C;
    size_t ldc;
    double alpha;
    double beta;
};

#ifdef MLAS_TARGET_AMD64

void
MlasDgemmMultiplyBeta(
    double* C,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    double beta
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

    beta - Supplies the scalar beta multiplier (see DGEMM definition).

Return Value:

    None.

--*/
{
    MLAS_FLOAT64X2 BetaBroadcast = MlasBroadcastFloat64x2(beta);

    while (CountM-- > 0) {

        double* c = C;
        size_t n = CountN;

        while (n >= 2) {
            MlasStoreFloat64x2(c, MlasMultiplyFloat64x2(MlasLoadFloat64x2(c), BetaBroadcast));
            c += 2;
            n -= 2;
        }

        if (n > 0) {
#if defined(MLAS_SSE2_INTRINSICS)
            _mm_store_sd(c, _mm_mul_sd(_mm_load_sd(c), BetaBroadcast));
#else
            *c = *c * beta;
#endif
        }

        C += ldc;
    }
}

void
MlasDgemmTransposeA(
    double* D,
    const double* A,
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

        double* d = D;
        const double* a = A;
        size_t y = CountY;

        do {

            double t0 = a[0];
            double t1 = a[lda];
            double t2 = a[lda * 2];
            double t3 = a[lda * 3];

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

        double* d = D;
        const double* a = A;
        size_t y = CountY;

        do {

            double t0 = a[0];
            double t1 = a[lda];

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

        double* d = D;
        const double* a = A;
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
MlasDgemmCopyPackB(
    double* D,
    const double* B,
    size_t ldb,
    size_t CountX,
    size_t CountY
    )
/*++

Routine Description:

    This routine copies elements from the source matrix to the destination
    packed buffer.

    Columns of 8 elements from the source matrix are unrolled to be physically
    contiguous for better locality inside the DGEMM kernels. Any remaining
    columns less than 8 elements wide are zero-padded.

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
    // Copy data from matrix B into the destination buffer 16 columns at a
    // time.
    //

    while (CountX >= 8) {

        const double* b = B;
        size_t y = CountY;

        do {

#if defined(MLAS_NEON64_INTRINSICS)
            vst4q_f64(D, vld4q_f64(b));
#else
            MLAS_FLOAT64X2 t0 = MlasLoadFloat64x2(&b[0]);
            MLAS_FLOAT64X2 t1 = MlasLoadFloat64x2(&b[2]);
            MLAS_FLOAT64X2 t2 = MlasLoadFloat64x2(&b[4]);
            MLAS_FLOAT64X2 t3 = MlasLoadFloat64x2(&b[6]);

            MlasStoreAlignedFloat64x2(&D[0], t0);
            MlasStoreAlignedFloat64x2(&D[2], t1);
            MlasStoreAlignedFloat64x2(&D[4], t2);
            MlasStoreAlignedFloat64x2(&D[6], t3);
#endif

            D += 8;
            b += ldb;
            y--;

        } while (y > 0);

        B += 8;
        CountX -= 8;
    }

    //
    // Special case the handling of the remaining columns less than 16 elements
    // wide.
    //

    if (CountX > 0) {

        MLAS_FLOAT64X2 ZeroFloat64x2 = MlasZeroFloat64x2();

#if defined(MLAS_NEON64_INTRINSICS)
        float64x2x4_t ZeroFloat64x2x4 = { ZeroFloat64x2, ZeroFloat64x2, ZeroFloat64x2, ZeroFloat64x2 };
#endif

        size_t y = CountY;

        do {

            double* d = D;
            const double* b = B;

#if defined(MLAS_NEON64_INTRINSICS)
            vst4q_f64(d, ZeroFloat64x2x4);
#else
            MlasStoreAlignedFloat64x2(&d[0], ZeroFloat64x2);
            MlasStoreAlignedFloat64x2(&d[2], ZeroFloat64x2);
            MlasStoreAlignedFloat64x2(&d[4], ZeroFloat64x2);
            MlasStoreAlignedFloat64x2(&d[6], ZeroFloat64x2);
#endif

            if ((CountX & 4) != 0) {

                MLAS_FLOAT64X2 t0 = MlasLoadFloat64x2(&b[0]);
                MLAS_FLOAT64X2 t1 = MlasLoadFloat64x2(&b[2]);

                MlasStoreAlignedFloat64x2(&d[0], t0);
                MlasStoreAlignedFloat64x2(&d[2], t1);

                d += 4;
                b += 4;
            }

            if ((CountX & 2) != 0) {

                MlasStoreAlignedFloat64x2(&d[0], MlasLoadFloat64x2(&b[0]));

                d += 2;
                b += 2;
            }

            if ((CountX & 1) != 0) {
                d[0] = b[0];
            }

            D += 8;
            B += ldb;
            y--;

        } while (y > 0);
    }
}

void
MlasDgemmTransposePackB(
    double* D,
    const double* B,
    size_t ldb,
    size_t CountY,
    size_t CountX
    )
/*++

Routine Description:

    This routine transposes elements from the source matrix to the destination
    packed buffer.

    Columns of 8 elements from the source matrix are unrolled to be physically
    contiguous for better locality inside the DGEMM kernels. Any remaining
    columns less than 8 elements wide are zero-padded.

Arguments:

    D - Supplies the address of the destination packed buffer.

    B - Supplies the address of the source matrix.

    ldb - Supplies the number of elements per row of the source matrix.

    CountY - Supplies the number of rows of the source matrix to transpose.

    CountX - Supplies the number of columns of the source matrix to transpose.

Return Value:

    None.

--*/
{
    //
    // Transpose elements from matrix B into the packed buffer 8 rows at a
    // time.
    //

    while (CountY >= 8) {

        const double* b = B;
        size_t x = CountX;

        while (x > 0) {

            double t0 = b[0];
            double t1 = b[ldb];
            double t2 = b[ldb * 2];
            double t3 = b[ldb * 3];
            double t4 = b[ldb * 4];
            double t5 = b[ldb * 5];
            double t6 = b[ldb * 6];
            double t7 = b[ldb * 7];

            D[0] = t0;
            D[1] = t1;
            D[2] = t2;
            D[3] = t3;
            D[4] = t4;
            D[5] = t5;
            D[6] = t6;
            D[7] = t7;

            D += 8;
            b += 1;
            x--;
        }

        B += ldb * 8;
        CountY -= 8;
    }

    //
    // Special case the handling of the less than 8 remaining rows.
    //

    if (CountY > 0) {

        MLAS_FLOAT64X2 ZeroFloat64x2 = MlasZeroFloat64x2();

        size_t x = CountX;

        while (x > 0) {

            double* d = D;
            const double* b = B;

            MlasStoreAlignedFloat64x2(&d[0], ZeroFloat64x2);
            MlasStoreAlignedFloat64x2(&d[2], ZeroFloat64x2);
            MlasStoreAlignedFloat64x2(&d[4], ZeroFloat64x2);
            MlasStoreAlignedFloat64x2(&d[6], ZeroFloat64x2);

            if ((CountY & 4) != 0) {

                double t0 = b[0];
                double t1 = b[ldb];
                double t2 = b[ldb * 2];
                double t3 = b[ldb * 3];

                d[0] = t0;
                d[1] = t1;
                d[2] = t2;
                d[3] = t3;

                d += 4;
                b += ldb * 4;
            }

            if ((CountY & 2) != 0) {

                double t0 = b[0];
                double t1 = b[ldb];

                d[0] = t0;
                d[1] = t1;

                d += 2;
                b += ldb * 2;
            }

            if ((CountY & 1) != 0) {
                d[0] = b[0];
            }

            D += 8;
            B += 1;
            x--;
        }
    }
}

MLAS_FORCEINLINE
double*
MlasDgemmKernelLoop(
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

    This routine steps through the rows of the input and output matrices calling
    the kernel until all rows have been processed.

Arguments:

    A - Supplies the address of matrix A.

    B - Supplies the address of matrix B. The matrix data has been packed using
        MlasDgemmCopyPackB or MlasDgemmTransposePackB.

    C - Supplies the address of matrix C.

    CountK - Supplies the number of columns from matrix A and the number of rows
        from matrix B to iterate over.

    CountM - Supplies the number of rows from matrix A and matrix C to iterate
        over.

    CountN - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    lda - Supplies the first dimension of matrix A.

    ldc - Supplies the first dimension of matrix C.

    alpha - Supplies the scalar alpha multiplier (see DGEMM definition).

    ZeroMode - Supplies true if the output matrix must be zero initialized,
        else false if the output matrix is accumulated into.

Return Value:

    Returns the next address of matrix C.

--*/
{
    while (CountM > 0) {

        size_t RowsHandled;

#if defined(MLAS_TARGET_AMD64_IX86)
        RowsHandled = MlasPlatform.GemmDoubleKernel(A, B, C, CountK, CountM, CountN, lda, ldc, alpha, ZeroMode);
#else
        if (ZeroMode) {
            RowsHandled = MlasDgemmKernelZero(A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
        } else {
            RowsHandled = MlasDgemmKernelAdd(A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
        }
#endif

        C += ldc * RowsHandled;
        A += lda * RowsHandled;
        CountM -= RowsHandled;
    }

    return C;
}

void
MlasDgemmOperation(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    double alpha,
    const double* A,
    size_t lda,
    const double* B,
    size_t ldb,
    double beta,
    double* C,
    size_t ldc
    )
/*++

Routine Description:

    This routine implements the single precision matrix/matrix multiply
    operation (DGEMM).

Arguments:

    TransA - Supplies the transpose operation for matrix A.

    TransB - Supplies the transpose operation for matrix B.

    M - Supplies the number of rows of matrix A and matrix C.

    N - Supplies the number of columns of matrix B and matrix C.

    K - Supplies the number of columns of matrix A and the number of rows of
        matrix B.

    alpha - Supplies the scalar alpha multiplier (see DGEMM definition).

    A - Supplies the address of matrix A.

    lda - Supplies the first dimension of matrix A.

    B - Supplies the address of matrix B.

    ldb - Supplies the first dimension of matrix B.

    beta - Supplies the scalar beta multiplier (see DGEMM definition).

    C - Supplies the address of matrix C.

    ldc - Supplies the first dimension of matrix C.

Return Value:

    None.

--*/
{
    double PanelA[MLAS_DGEMM_TRANSA_ROWS * MLAS_DGEMM_STRIDEK];
    MLAS_DECLSPEC_ALIGN(double PanelB[MLAS_DGEMM_STRIDEN * MLAS_DGEMM_STRIDEK], 8 * sizeof(double));

    //
    // Handle the special case of K equals zero. Apply the beta multiplier to
    // the output matrix and exit.
    //

    if (K == 0) {
        MlasDgemmMultiplyBeta(C, M, N, ldc, beta);
        return;
    }

    //
    // Compute the strides to step through slices of the input matrices.
    //
    // Expand the N stride if K is small or expand the K stride if N is small
    // for better utilization of the B panel. Avoid changing the K stride if
    // the A panel needs to be used for transposing.
    //

    size_t StrideN = MLAS_DGEMM_STRIDEN;
    size_t StrideK = MLAS_DGEMM_STRIDEK;

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
            MlasDgemmMultiplyBeta(C + n, M, CountN, ldc, beta);
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
                MlasDgemmCopyPackB(PanelB, B + n + k * ldb, ldb, CountN, CountK);
            } else {
                MlasDgemmTransposePackB(PanelB, B + k + n * ldb, ldb, CountN, CountK);
            }

            //
            // Step through each slice of matrix A along the M dimension.
            //

            double* c = C + n;

            if (TransA == CblasNoTrans) {

                MlasDgemmKernelLoop(A + k, PanelB, c, CountK, M, CountN, lda, ldc, alpha, ZeroMode);

            } else {

                const double* a = A + k * lda;
                size_t RowsRemaining = M;

                while (RowsRemaining > 0) {

                    //
                    // Transpose elements from matrix A into a local buffer.
                    //

                    size_t RowsTransposed = std::min(RowsRemaining, size_t(MLAS_DGEMM_TRANSA_ROWS));

                    MlasDgemmTransposeA(PanelA, a, lda, RowsTransposed, CountK);

                    RowsRemaining -= RowsTransposed;
                    a += RowsTransposed;

                    //
                    // Step through the rows of the local buffer.
                    //

                    c = MlasDgemmKernelLoop(PanelA, PanelB, c, CountK, RowsTransposed, CountN, CountK, ldc, alpha, ZeroMode);
                }
            }

            ZeroMode = false;
        }
    }
}

void
MlasDgemmThreaded(
    void* Context,
    ptrdiff_t ThreadId
    )
/*++

Routine Description:

    This routine is invoked from a worker thread to execute a segment of a
    DGEMM operation.

Arguments:

    Context - Supplies the pointer to the context for the threaded operation.

    ThreadId - Supplies the current index of the threaded operation.

Return Value:

    None.

--*/
{
    const auto* WorkBlock = (MLAS_DGEMM_WORK_BLOCK*)Context;

    const ptrdiff_t ThreadCountM = WorkBlock->ThreadCountM;
    const ptrdiff_t ThreadCountN = WorkBlock->ThreadCountN;

    const ptrdiff_t ThreadIdM = ThreadId / ThreadCountN;
    const ptrdiff_t ThreadIdN = ThreadId % ThreadCountN;

    //
    // Partition the operation along the M dimension.
    //

    size_t M = WorkBlock->M;
    size_t RangeStartM;
    size_t RangeCountM;

    MlasPartitionWork(ThreadIdM, ThreadCountM, M, &RangeStartM, &RangeCountM);

    //
    // Partition the operation along the N dimension.
    //

    size_t N = WorkBlock->N;
    size_t RangeStartN;
    size_t RangeCountN;

    const size_t BlockedN = (N + MLAS_DGEMM_STRIDEN_THREAD_ALIGN - 1) /
        MLAS_DGEMM_STRIDEN_THREAD_ALIGN;

    MlasPartitionWork(ThreadIdN, ThreadCountN, BlockedN, &RangeStartN,
        &RangeCountN);

    RangeStartN *= MLAS_DGEMM_STRIDEN_THREAD_ALIGN;
    RangeCountN *= MLAS_DGEMM_STRIDEN_THREAD_ALIGN;

    RangeCountN = std::min(N - RangeStartN, RangeCountN);

    //
    // Dispatch the partitioned operation.
    //

    CBLAS_TRANSPOSE TransA = WorkBlock->TransA;
    CBLAS_TRANSPOSE TransB = WorkBlock->TransB;

    const size_t lda = WorkBlock->lda;
    const size_t ldb = WorkBlock->ldb;
    const size_t ldc = WorkBlock->ldc;

    const double* A = WorkBlock->A + RangeStartM * ((TransA == CblasNoTrans) ? lda : 1);
    const double* B = WorkBlock->B + RangeStartN * ((TransB == CblasNoTrans) ? 1 : ldb);
    double* C = WorkBlock->C + RangeStartM * ldc + RangeStartN;

    MlasDgemmOperation(TransA, TransB, RangeCountM, RangeCountN, WorkBlock->K,
        WorkBlock->alpha, A, lda, B, ldb, WorkBlock->beta, C, ldc);
}

void
MlasDgemmSchedule(
    MLAS_DGEMM_WORK_BLOCK* WorkBlock,
    MLAS_THREADPOOL* ThreadPool
    )
/*++

Routine Description:

    This routine schedules the double precision matrix/matrix multiply
    operation (DGEMM) across one or more threads.

Arguments:

    WorkBlock - Supplies the structure containing the GEMM parameters.

    ThreadPool - Supplies the thread pool object to use, else nullptr if the
        base library threading support should be used.

Return Value:

    None.

--*/
{
    const size_t M = WorkBlock->M;
    const size_t N = WorkBlock->N;
    const size_t K = WorkBlock->K;

    //
    // Compute the number of target threads given the complexity of the DGEMM
    // operation. Small requests should run using the single threaded path.
    //

    const double Complexity = double(M) * double(N) * double(K);

    ptrdiff_t TargetThreadCount;

    if (Complexity < double(MLAS_DGEMM_THREAD_COMPLEXITY * MlasPlatform.MaximumThreadCount)) {
        TargetThreadCount = ptrdiff_t(Complexity / double(MLAS_DGEMM_THREAD_COMPLEXITY)) + 1;
    } else {
        TargetThreadCount = MlasPlatform.MaximumThreadCount;
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

    if (N > M) {

        const size_t BlockedN = (N + MLAS_DGEMM_STRIDEN_THREAD_ALIGN - 1) /
            MLAS_DGEMM_STRIDEN_THREAD_ALIGN;

        if (size_t(TargetThreadCount) > BlockedN) {
            TargetThreadCount = ptrdiff_t(BlockedN);
        }

        WorkBlock->ThreadCountM = 1;
        WorkBlock->ThreadCountN = TargetThreadCount;

    } else {

        if (size_t(TargetThreadCount) > M) {
            TargetThreadCount = ptrdiff_t(M);
        }

        WorkBlock->ThreadCountM = TargetThreadCount;
        WorkBlock->ThreadCountN = 1;
    }

    MlasExecuteThreaded(MlasDgemmThreaded, WorkBlock, TargetThreadCount, ThreadPool);
}

void
MLASCALL
MlasGemm(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    double alpha,
    const double* A,
    size_t lda,
    const double* B,
    size_t ldb,
    double beta,
    double* C,
    size_t ldc,
    MLAS_THREADPOOL* ThreadPool
    )
/*++

Routine Description:

    This routine implements the double precision matrix/matrix multiply
    operation (DGEMM).

Arguments:

    TransA - Supplies the transpose operation for matrix A.

    TransB - Supplies the transpose operation for matrix B.

    M - Supplies the number of rows of matrix A and matrix C.

    N - Supplies the number of columns of matrix B and matrix C.

    K - Supplies the number of columns of matrix A and the number of rows of
        matrix B.

    alpha - Supplies the scalar alpha multiplier (see DGEMM definition).

    A - Supplies the address of matrix A.

    lda - Supplies the first dimension of matrix A.

    B - Supplies the address of matrix B.

    ldb - Supplies the first dimension of matrix B.

    beta - Supplies the scalar beta multiplier (see DGEMM definition).

    C - Supplies the address of matrix C.

    ldc - Supplies the first dimension of matrix C.

    ThreadPool - Supplies the thread pool object to use, else nullptr if the
        base library threading support should be used.

Return Value:

    None.

--*/
{
    MLAS_DGEMM_WORK_BLOCK WorkBlock;

    //
    // Capture the GEMM parameters to the work block.
    //

    memset(&WorkBlock, 0, sizeof(MLAS_DGEMM_WORK_BLOCK));

    WorkBlock.TransA = TransA;
    WorkBlock.TransB = TransB;
    WorkBlock.M = M;
    WorkBlock.N = N;
    WorkBlock.K = K;
    WorkBlock.A = A;
    WorkBlock.lda = lda;
    WorkBlock.B = B;
    WorkBlock.ldb = ldb;
    WorkBlock.C = C;
    WorkBlock.ldc = ldc;
    WorkBlock.alpha = alpha;
    WorkBlock.beta = beta;

    //
    // Schedule the operation across a set of worker threads.
    //

    MlasDgemmSchedule(&WorkBlock, ThreadPool);
}

#endif
