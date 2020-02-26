/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sgemm.cpp

Abstract:

    This module implements the single precision matrix/matrix multiply
    operation (SGEMM).

--*/

#include "mlasi.h"

//
// Define the number of rows from matrix A to transpose to a local buffer.
//
// N.B. AVX processes a maximum of 4 rows, FMA3 processes a maximum of 6
// rows, and AVX512F processes a maximum of 12 rows.
//

#define MLAS_SGEMM_TRANSA_ROWS              12

//
// Define the parameters to execute segments of a SGEMM operation on worker
// threads.
//

struct MLAS_SGEMM_WORK_BLOCK {
    CBLAS_TRANSPOSE TransA;
    CBLAS_TRANSPOSE TransB;
    size_t K;
    size_t N;
    size_t lda;
    size_t ldb;
    size_t ldc;
    float alpha;
    float beta;
    uint32_t PackedStrideN;
    uint32_t PackedStrideK;
    struct SEGMENT {
        size_t M;
        size_t N;
        const float* A;
        const float* B;
        const float* PrePackedB;
        float* C;
    } Segments[MLAS_MAXIMUM_THREAD_COUNT];
};

void
MlasSgemmMultiplyBeta(
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

    do {

        float* c = C;
        size_t n = CountN;

        while (n >= 4) {
            MlasStoreFloat32x4(c, MlasMultiplyFloat32x4(MlasLoadFloat32x4(c), BetaBroadcast));
            c += 4;
            n -= 4;
        }

        while (n > 0) {
#if defined(MLAS_SSE2_INTRINSICS)
            _mm_store_ss(c, _mm_mul_ss(_mm_load_ss(c), BetaBroadcast));
#else
            *c = *c * beta;
#endif
            c += 1;
            n -= 1;
        }

        C += ldc;
        CountM--;

    } while (CountM > 0);
}

void
MlasSgemmTransposeA(
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
MlasSgemmCopyPackB(
    float* D,
    const float* B,
    size_t ldb,
    size_t CountX,
    size_t CountY
    )
/*++

Routine Description:

    This routine copies elements from the source matrix to the destination
    packed buffer.

    Columns of 16 elements from the source matrix are unrolled to be physically
    contiguous for better locality inside the SGEMM kernels. Any remaining
    columns less than 16 elements wide are zero-padded.

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

    while (CountX >= 16) {

        const float* b = B;
        size_t y = CountY;

        do {

#if defined(MLAS_NEON_INTRINSICS)
            vst4q_f32(D, vld4q_f32(b));
#else
            MLAS_FLOAT32X4 t0 = MlasLoadFloat32x4(&b[0]);
            MLAS_FLOAT32X4 t1 = MlasLoadFloat32x4(&b[4]);
            MLAS_FLOAT32X4 t2 = MlasLoadFloat32x4(&b[8]);
            MLAS_FLOAT32X4 t3 = MlasLoadFloat32x4(&b[12]);

            MlasStoreAlignedFloat32x4(&D[0], t0);
            MlasStoreAlignedFloat32x4(&D[4], t1);
            MlasStoreAlignedFloat32x4(&D[8], t2);
            MlasStoreAlignedFloat32x4(&D[12], t3);
#endif

            D += 16;
            b += ldb;
            y--;

        } while (y > 0);

        B += 16;
        CountX -= 16;
    }

    //
    // Special case the handling of the remaining columns less than 16 elements
    // wide.
    //

    if (CountX > 0) {

        MLAS_FLOAT32X4 ZeroFloat32x4 = MlasZeroFloat32x4();

#if defined(MLAS_NEON_INTRINSICS)
        float32x4x4_t ZeroFloat32x4x4 = { ZeroFloat32x4, ZeroFloat32x4, ZeroFloat32x4, ZeroFloat32x4 };
#endif

        size_t y = CountY;

        do {

            float* d = D;
            const float* b = B;

#if defined(MLAS_NEON_INTRINSICS)
            vst4q_f32(d, ZeroFloat32x4x4);
#else
            MlasStoreAlignedFloat32x4(d, ZeroFloat32x4);
            MlasStoreAlignedFloat32x4(d + 4, ZeroFloat32x4);
            MlasStoreAlignedFloat32x4(d + 8, ZeroFloat32x4);
            MlasStoreAlignedFloat32x4(d + 12, ZeroFloat32x4);
#endif

            if ((CountX & 8) != 0) {

                MLAS_FLOAT32X4 t0 = MlasLoadFloat32x4(b);
                MLAS_FLOAT32X4 t1 = MlasLoadFloat32x4(b + 4);

                MlasStoreAlignedFloat32x4(d, t0);
                MlasStoreAlignedFloat32x4(d + 4, t1);

                d += 8;
                b += 8;
            }

            if ((CountX & 4) != 0) {

                MlasStoreAlignedFloat32x4(d, MlasLoadFloat32x4(b));

                d += 4;
                b += 4;
            }

            if ((CountX & 2) != 0) {

                float t0 = b[0];
                float t1 = b[1];

                d[0] = t0;
                d[1] = t1;

                d += 2;
                b += 2;
            }

            if ((CountX & 1) != 0) {
                d[0] = b[0];
            }

            D += 16;
            B += ldb;
            y--;

        } while (y > 0);
    }
}

template<unsigned N>
inline
void
MlasSgemmTransposePackBNx4(
    float* D,
    const float* B,
    size_t ldb
    )
/*++

Routine Description:

    This routine transposes elements from the source matrix to the destination
    packed buffer.

    4 columns of N rows from the source matrix are transposed to N columns of 4
    rows in the destination packed buffer.

Arguments:

    D - Supplies the address of the destination packed buffer.

    B - Supplies the address of the source matrix.

    ldb - Supplies the number of elements per row of the source matrix.

Return Value:

    None.

--*/
{
    for (unsigned n = 0; n < N / 4; n++) {

        MLAS_FLOAT32X4 t0 = MlasLoadFloat32x4(&B[ldb * 0]);
        MLAS_FLOAT32X4 t1 = MlasLoadFloat32x4(&B[ldb * 1]);
        MLAS_FLOAT32X4 t2 = MlasLoadFloat32x4(&B[ldb * 2]);
        MLAS_FLOAT32X4 t3 = MlasLoadFloat32x4(&B[ldb * 3]);

#if defined(MLAS_NEON_INTRINSICS)
        float32x4x2_t z0 = vzipq_f32(t0, t2);
        float32x4x2_t z1 = vzipq_f32(t1, t3);
        float32x4x2_t o0 = vzipq_f32(z0.val[0], z1.val[0]);
        float32x4x2_t o1 = vzipq_f32(z0.val[1], z1.val[1]);
        t0 = o0.val[0];
        t1 = o0.val[1];
        t2 = o1.val[0];
        t3 = o1.val[1];
#elif defined(MLAS_SSE2_INTRINSICS)
        // N.B. The MSVC version of _MM_TRANSPOSE4_PS uses shufps which is
        // slightly larger than the below sequence, so manually expand the
        // matrix transpose.
        __m128 z0 = _mm_unpacklo_ps(t0, t1);
        __m128 z1 = _mm_unpackhi_ps(t0, t1);
        __m128 z2 = _mm_unpacklo_ps(t2, t3);
        __m128 z3 = _mm_unpackhi_ps(t2, t3);
        t0 = _mm_movelh_ps(z0, z2);
        t1 = _mm_movehl_ps(z2, z0);
        t2 = _mm_movelh_ps(z1, z3);
        t3 = _mm_movehl_ps(z3, z1);
#else
#error Unsupported architecture.
#endif

        MlasStoreAlignedFloat32x4(&D[0], t0);
        MlasStoreAlignedFloat32x4(&D[16], t1);
        MlasStoreAlignedFloat32x4(&D[32], t2);
        MlasStoreAlignedFloat32x4(&D[48], t3);

        D += 4;
        B += ldb * 4;
    }
}

void
MlasSgemmTransposePackB(
    float* D,
    const float* B,
    size_t ldb,
    size_t CountY,
    size_t CountX
    )
/*++

Routine Description:

    This routine transposes elements from the source matrix to the destination
    packed buffer.

    Columns of 16 elements from the source matrix are unrolled to be physically
    contiguous for better locality inside the SGEMM kernels. Any remaining
    columns less than 16 elements wide are zero-padded.

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
    // Transpose elements from matrix B into the packed buffer 16 rows at a
    // time.
    //

    while (CountY >= 16) {

        const float* b = B;
        size_t x = CountX;

#if defined(MLAS_TARGET_AMD64)

        PMLAS_SGEMM_TRANSPOSE_PACKB_BLOCK_ROUTINE SgemmTransposePackB16x4Routine =
            MlasPlatform.TransposePackB16x4Routine;

        while (x >= 4) {

            SgemmTransposePackB16x4Routine(&D[0], &b[0], ldb);

            D += 16 * 4;
            b += 4;
            x -= 4;
        }

#else

        while (x >= 4) {

            MlasSgemmTransposePackBNx4<16>(&D[0], &b[0], ldb);

            D += 16 * 4;
            b += 4;
            x -= 4;
        }

#endif

        while (x > 0) {

            float t0 = b[0];
            float t1 = b[ldb];
            float t2 = b[ldb * 2];
            float t3 = b[ldb * 3];
            float t4 = b[ldb * 4];
            float t5 = b[ldb * 5];
            float t6 = b[ldb * 6];
            float t7 = b[ldb * 7];
            float t8 = b[ldb * 8];
            float t9 = b[ldb * 9];
            float t10 = b[ldb * 10];
            float t11 = b[ldb * 11];
            float t12 = b[ldb * 12];
            float t13 = b[ldb * 13];
            float t14 = b[ldb * 14];
            float t15 = b[ldb * 15];

            D[0] = t0;
            D[1] = t1;
            D[2] = t2;
            D[3] = t3;
            D[4] = t4;
            D[5] = t5;
            D[6] = t6;
            D[7] = t7;
            D[8] = t8;
            D[9] = t9;
            D[10] = t10;
            D[11] = t11;
            D[12] = t12;
            D[13] = t13;
            D[14] = t14;
            D[15] = t15;

            D += 16;
            b += 1;
            x--;
        }

        B += ldb * 16;
        CountY -= 16;
    }

    //
    // Special case the handling of the less than 16 remaining rows.
    //

    if (CountY > 0) {

        MLAS_FLOAT32X4 ZeroFloat32x4 = MlasZeroFloat32x4();

        size_t x = CountX;

        //
        // Transpose 4 columns at a time.
        //

        while (x >= 4) {

            float* d = D;
            const float* b = B;

            if ((CountY & 8) != 0) {

                MlasSgemmTransposePackBNx4<8>(&d[0], &b[0], ldb);

                d += 8;
                b += ldb * 8;

            } else {

                MlasStoreAlignedFloat32x4(&d[8], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[12], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[24], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[28], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[40], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[44], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[56], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[60], ZeroFloat32x4);
            }

            if ((CountY & 4) != 0) {

                MlasSgemmTransposePackBNx4<4>(&d[0], &b[0], ldb);

                d += 4;
                b += ldb * 4;

            } else {

                MlasStoreAlignedFloat32x4(&d[4], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[20], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[36], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[52], ZeroFloat32x4);
            }

            MlasStoreAlignedFloat32x4(&d[0], ZeroFloat32x4);
            MlasStoreAlignedFloat32x4(&d[16], ZeroFloat32x4);
            MlasStoreAlignedFloat32x4(&d[32], ZeroFloat32x4);
            MlasStoreAlignedFloat32x4(&d[48], ZeroFloat32x4);

            if ((CountY & 2) != 0) {

#if defined(MLAS_NEON_INTRINSICS)
                MLAS_FLOAT32X4 t0 = MlasLoadFloat32x4(&b[0]);
                MLAS_FLOAT32X4 t1 = MlasLoadFloat32x4(&b[ldb]);

                MlasStoreLaneFloat32x4<0>(&d[0], t0);
                MlasStoreLaneFloat32x4<0>(&d[1], t1);
                MlasStoreLaneFloat32x4<1>(&d[16], t0);
                MlasStoreLaneFloat32x4<1>(&d[17], t1);
                MlasStoreLaneFloat32x4<2>(&d[32], t0);
                MlasStoreLaneFloat32x4<2>(&d[33], t1);
                MlasStoreLaneFloat32x4<3>(&d[48], t0);
                MlasStoreLaneFloat32x4<3>(&d[49], t1);
#elif defined(MLAS_SSE2_INTRINSICS)
                MLAS_FLOAT32X4 t0 = MlasLoadFloat32x4(&b[0]);
                MLAS_FLOAT32X4 t1 = MlasLoadFloat32x4(&b[ldb]);

                __m128 v0 = _mm_unpacklo_ps(t0, t1);
                __m128 v1 = _mm_unpackhi_ps(t0, t1);
                _mm_storel_pi((__m64*)&d[0], v0);
                _mm_storeh_pi((__m64*)&d[16], v0);
                _mm_storel_pi((__m64*)&d[32], v1);
                _mm_storeh_pi((__m64*)&d[48], v1);
#else
#error Unsupported architecture.
#endif

                d += 2;
                b += ldb * 2;
            }

            if ((CountY & 1) != 0) {

#if defined(MLAS_NEON_INTRINSICS)
                MLAS_FLOAT32X4 t0 = MlasLoadFloat32x4(&b[0]);

                MlasStoreLaneFloat32x4<0>(&d[0], t0);
                MlasStoreLaneFloat32x4<1>(&d[16], t0);
                MlasStoreLaneFloat32x4<2>(&d[32], t0);
                MlasStoreLaneFloat32x4<3>(&d[48], t0);
#else
                d[0] = b[0];
                d[16] = b[1];
                d[32] = b[2];
                d[48] = b[3];
#endif
            }

            D += 16 * 4;
            B += 4;
            x -= 4;
        }

        //
        // Transpose the remaining columns.
        //

        while (x > 0) {

            float* d = D;
            const float* b = B;

            if ((CountY & 8) != 0) {

                float t0 = b[0];
                float t1 = b[ldb];
                float t2 = b[ldb * 2];
                float t3 = b[ldb * 3];
                float t4 = b[ldb * 4];
                float t5 = b[ldb * 5];
                float t6 = b[ldb * 6];
                float t7 = b[ldb * 7];

                d[0] = t0;
                d[1] = t1;
                d[2] = t2;
                d[3] = t3;
                d[4] = t4;
                d[5] = t5;
                d[6] = t6;
                d[7] = t7;

                d += 8;
                b += ldb * 8;

            } else {

                MlasStoreAlignedFloat32x4(&d[8], ZeroFloat32x4);
                MlasStoreAlignedFloat32x4(&d[12], ZeroFloat32x4);
            }

            if ((CountY & 4) != 0) {

                float t0 = b[0];
                float t1 = b[ldb];
                float t2 = b[ldb * 2];
                float t3 = b[ldb * 3];

                d[0] = t0;
                d[1] = t1;
                d[2] = t2;
                d[3] = t3;

                d += 4;
                b += ldb * 4;

            } else {

                MlasStoreAlignedFloat32x4(&d[4], ZeroFloat32x4);
            }

            MlasStoreAlignedFloat32x4(d, ZeroFloat32x4);

            if ((CountY & 2) != 0) {

                float t0 = b[0];
                float t1 = b[ldb];

                d[0] = t0;
                d[1] = t1;

                d += 2;
                b += ldb * 2;
            }

            if ((CountY & 1) != 0) {
                d[0] = b[0];
            }

            D += 16;
            B += 1;
            x--;
        }
    }
}


namespace {

//
// Define how GEMM computation is split between threads.
//

enum class GEMM_THREAD_SPLIT_MODE {
    SplitARows,
    SplitBCols,
};

//
// Define threading strategy for GEMM computation.
//

struct GEMM_THREADING_STRATEGY {
    GEMM_THREAD_SPLIT_MODE SplitMode;
    int NumThreads;
    size_t ThreadStrideM;
    size_t ThreadStrideN;
};

}


static GEMM_THREADING_STRATEGY
MlasSgemmChooseThreadingStrategy(
    size_t M,
    size_t N,
    size_t K,
    size_t ThreadStrideNAlignment,
    int MaximumThreadCount
    ) 
/*++

Routine Description:

    This routine chooses GEMM threading strategy for given matrix sizes.

Arguments:

    M - Supplies the number of rows of matrix A and matrix C.

    N - Supplies the number of columns of matrix B and matrix C.

    K - Supplies the number of columns of matrix A and the number of rows of
        matrix B.

    MaximumThreadCount - Supplies the number of threads in the thread pool.

Return Value:

    Threading strategy.

--*/
{
    GEMM_THREADING_STRATEGY res{};

    //
    // Compute the number of target threads given the complexity of the SGEMM
    // operation. Small requests should run using the single threaded path.
    //

    double Complexity = double(M) * double(N) * double(K);

    if (Complexity < double(MLAS_SGEMM_THREAD_COMPLEXITY * MLAS_MAXIMUM_THREAD_COUNT)) {
        res.NumThreads = int32_t(Complexity / double(MLAS_SGEMM_THREAD_COMPLEXITY)) + 1;
    } else {
        res.NumThreads = MLAS_MAXIMUM_THREAD_COUNT;
    }

    if (res.NumThreads >= MaximumThreadCount) {
        res.NumThreads = MaximumThreadCount;
    }

    if (N >= M) {

        res.SplitMode = GEMM_THREAD_SPLIT_MODE::SplitBCols;
        res.ThreadStrideM = 0;
        res.ThreadStrideN = N / res.NumThreads;
        if (res.ThreadStrideN * res.NumThreads != N) {
            res.ThreadStrideN++;
        }
        res.ThreadStrideN =
            (res.ThreadStrideN + ThreadStrideNAlignment - 1) & ~(ThreadStrideNAlignment - 1);
        
    } else {

        res.SplitMode = GEMM_THREAD_SPLIT_MODE::SplitARows;
        res.ThreadStrideN = 0;
        res.ThreadStrideM = M / res.NumThreads;
        if (res.ThreadStrideM * res.NumThreads != M) {
            res.ThreadStrideM++;
        }

    }

    return res;
}


static size_t MlasGemmGetPackedSize(
    size_t CountN,
    size_t CountK
)
/*++

Routine Description:

    This routine calculates the size of a matrix block that
    is packed by routines MlasSgemmCopyPackB or MlasSgemmTransposePackB.

Arguments:

    CountN - Supplies the number of columns in the original matrix block.

    CountK - Supplies the number of rows in the original matrix block.

Return Value:

    Number of elements in the packed matrix block, including padding.

--*/
{

    //
    // Round up CountN to the nearest multiple of 16.
    //

    return CountK * ((CountN + 15) & ~15u);
}


static void
MlasGemmChoosePackedStrides(
    uint32_t* PackedStrideN,
    uint32_t* PackedStrideK,
    CBLAS_TRANSPOSE TransA,
    size_t M,
    size_t N,
    size_t K,
    int MaximumThreadCount
)
/*++

Routine Description:

    This routine chooses the size of blocks of matrix B that will be
    individually packed by routines MlasSgemmCopyPackB or MlasSgemmTransposePackB.

Arguments:

    PackedStrideN - Supplies the address where the number of columns
                    of the block is written.

    PackedStrideK - Supplies the address where the number of rows
                    of the block is written.

    M - Supplies the number of rows of matrix A and matrix C.

    N - Supplies the number of columns of matrix B and matrix C.

    K - Supplies the number of columns of matrix A and the number of rows of
        matrix B.

    MaximumThreadCount - Supplies the number of threads in the thread pool.

Return Value:

    None.

--*/
{
    *PackedStrideN = MLAS_SGEMM_STRIDEN;
    *PackedStrideK = MLAS_SGEMM_STRIDEK;

    if (M == 0) {

        // 
        // M is not known.
        // Find the maximum possible number of threads between which we may
        // want to split columns of B. The value of PackedStrideN should be
        // low enough to allow for this number of threads.
        //

        const size_t MinThreadStrideN = MlasSgemmChooseThreadingStrategy(
            N - 1, N, K, MLAS_SGEMM_STRIDEN_THREAD_ALIGN, MaximumThreadCount).ThreadStrideN;

        if (MinThreadStrideN >= K) {

            // K is small. Expand N stride.
            while (*PackedStrideK / 2 >= K && (MinThreadStrideN % (2 * *PackedStrideN) == 0)) {
                *PackedStrideN *= 2;
                *PackedStrideK /= 2;
            }

        } else if (TransA == CblasNoTrans) {

            while (*PackedStrideN > 16 && MinThreadStrideN % *PackedStrideN != 0) {
                *PackedStrideK *= 2;
                *PackedStrideN /= 2;
            }

        }

        return;
    }

    //
    // We know the exact matrix sizes, so we can predict how many threads will
    // be used at runtime and choose StrideN and StrideK based on that.
    //

    GEMM_THREADING_STRATEGY ThreadingStrategy = MlasSgemmChooseThreadingStrategy(M, N, K, MLAS_SGEMM_STRIDEN_THREAD_ALIGN, MaximumThreadCount);
    if (ThreadingStrategy.SplitMode == GEMM_THREAD_SPLIT_MODE::SplitBCols) {
        N = ThreadingStrategy.ThreadStrideN;
    }

    //
    // Compute the strides to step through slices of the input matrix.
    //
    // Expand the N stride if K is small or expand the K stride if N is small
    // for better utilization of the B panel. Avoid changing the K stride if
    // the A panel needs to be used for transposing.
    //

    if (N >= K) {

        while (*PackedStrideK / 2 >= K) {
            *PackedStrideN *= 2;
            *PackedStrideK /= 2;
        }

    } else if (TransA == CblasNoTrans) {

        while (*PackedStrideN > 16 && *PackedStrideN / 2 >= N) {
            *PackedStrideK *= 2;
            *PackedStrideN /= 2;
        }

    }
}


void
MlasSgemmOperation(
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
    const float* PrePackedB,
    uint32_t StrideN,
    uint32_t StrideK,
    float* C,
    size_t ldc
    )
/*++

Routine Description:

    This routine implements the single precision matrix/matrix multiply
    operation (SGEMM).

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

    B - Supplies the optional address of matrix B.

    ldb - Supplies the first dimension of matrix B, when it's provided.

    beta - Supplies the scalar beta multiplier (see SGEMM definition).

    PrePackedB - Supplies the optional address of packed representation
                 of matrix B, calculated by MlasGemmPackMatrix. 

    StrideN - Supplies the number of columns in each packed block of
              matrix B, when PrePackedB is provided.

    StrideK - Supplies the number of rows in each packed block of
              matrix B, when PrePackedB is provided.

    C - Supplies the address of matrix C.

    ldc - Supplies the first dimension of matrix C.

Return Value:

    None.

--*/
{
    float PanelA[MLAS_SGEMM_TRANSA_ROWS * MLAS_SGEMM_STRIDEK];
    MLAS_DECLSPEC_ALIGN(float PanelB[MLAS_SGEMM_STRIDEN * MLAS_SGEMM_STRIDEK], 16 * sizeof(float));
    assert(B != nullptr || PrePackedB != nullptr);

    //
    // Handle the special case of a small M. The data from matrix B is not
    // referenced multiple times, so using a local packed buffer is a wasted
    // memory copy.
    //

    if (B != nullptr && M == 1 && TransA == CblasNoTrans && alpha == 1.0f && (beta == 0.0f || beta == 1.0f)) {

#if defined(MLAS_TARGET_AMD64)

        PMLAS_SGEMM_KERNEL_M1_ROUTINE SgemmKernelM1Routine;

        if (TransB == CblasNoTrans) {
            SgemmKernelM1Routine = MlasPlatform.KernelM1Routine;
        } else {
            SgemmKernelM1Routine = MlasPlatform.KernelM1TransposeBRoutine;
        }

        if (SgemmKernelM1Routine != nullptr) {
            SgemmKernelM1Routine(A, B, C, K, N, ldb, beta);
            return;
        }

#endif

    }

    //
    // Handle the case when both B and C are column-vectors that are contiguous in memory.
    // Because transposition of such vectors doesn't change their layout, and
    // Transpose(A*B) = Transpose(B) * Transpose(A), we can apply the same 'small-M'
    // optimization as above, with A and B flipped.
    //
    if (B != nullptr && N == 1 && ldb == 1 && ldc == 1 && alpha == 1.0f && (beta == 0.0f || beta == 1.0f)) {

#if defined(MLAS_TARGET_AMD64)

        PMLAS_SGEMM_KERNEL_M1_ROUTINE SgemmKernelM1Routine;

        if (TransA == CblasNoTrans) {
            SgemmKernelM1Routine = MlasPlatform.KernelM1TransposeBRoutine;
        } else {
            SgemmKernelM1Routine = MlasPlatform.KernelM1Routine;
        }

        if (SgemmKernelM1Routine != nullptr) {
            SgemmKernelM1Routine(B, A, C, K, M, lda, beta);
            return;
        }

#endif

    }

    if (PrePackedB == nullptr) {

        //
        // PrePackedB is not used. We can choose optimal strides.
        //

        MlasGemmChoosePackedStrides(&StrideN, &StrideK, TransA, M, N, K, 1);
    }

    //
    // Step through each slice of matrix B along the N dimension.
    //

    size_t CountN;
    size_t CountK;

    for (size_t n = 0; n < N; n += CountN) {

        CountN = StrideN;

        if (CountN > (N - n)) {
            CountN = N - n;
        }

        //
        // Multiply the output matrix by beta as needed.
        //

        if (beta != 0.0f && beta != 1.0f) {
            MlasSgemmMultiplyBeta(C + n, M, CountN, ldc, beta);
        }

        //
        // Step through each slice of matrix B along the K dimension.
        //

        for (size_t k = 0; k < K; k += CountK) {

            bool ZeroMode = (k == 0 && beta == 0.0f);

            CountK = StrideK;

            if (CountK > (K - k)) {
                CountK = K - k;
            }

            //
            // Get packed representation of current block of matrix B.
            // If necessary, copy or transpose a panel of matrix B to a local packed buffer.
            //

            const float* PackedB;

            if (PrePackedB != nullptr) {
                PackedB = PrePackedB;
                PrePackedB += MlasGemmGetPackedSize(CountN, CountK);
            } else if (TransB == CblasNoTrans) {
                MlasSgemmCopyPackB(PanelB, B + n + k * ldb, ldb, CountN, CountK);
                PackedB = PanelB;
            } else {
                MlasSgemmTransposePackB(PanelB, B + k + n * ldb, ldb, CountN, CountK);
                PackedB = PanelB;
            }

            //
            // Step through each slice of matrix A along the M dimension.
            //

            float* c = C + n;

            size_t RowsRemaining = M;
            size_t RowsHandled;

            if (TransA == CblasNoTrans) {

                const float* a = A + k;

                //
                // Step through the rows of matrix A.
                //

                do {

#if defined(MLAS_TARGET_AMD64_IX86)
                    RowsHandled = MlasPlatform.GemmFloatKernel(a, PackedB, c, CountK, RowsRemaining, CountN, lda, ldc, alpha, ZeroMode);
#else
                    if (ZeroMode) {
                        RowsHandled = MlasSgemmKernelZero(a, PackedB, c, CountK, RowsRemaining, CountN, lda, ldc, alpha);
                    } else {
                        RowsHandled = MlasSgemmKernelAdd(a, PackedB, c, CountK, RowsRemaining, CountN, lda, ldc, alpha);
                    }
#endif

                    c += ldc * RowsHandled;
                    a += lda * RowsHandled;

                    RowsRemaining -= RowsHandled;

                } while (RowsRemaining > 0);

            } else {

                const float* a = A + k * lda;

                do {

                    //
                    // Transpose elements from matrix A into a local buffer.
                    //

                    size_t RowsTransposed = RowsRemaining;

                    if (RowsTransposed > MLAS_SGEMM_TRANSA_ROWS) {
                        RowsTransposed = MLAS_SGEMM_TRANSA_ROWS;
                    }

                    RowsRemaining -= RowsTransposed;

                    MlasSgemmTransposeA(PanelA, a, lda, RowsTransposed, CountK);

                    a += RowsTransposed;

                    //
                    // Step through the rows of the local buffer.
                    //

                    const float* pa = PanelA;

                    do {

#if defined(MLAS_TARGET_AMD64_IX86)
                        RowsHandled = MlasPlatform.GemmFloatKernel(pa, PackedB, c, CountK, RowsTransposed, CountN, CountK, ldc, alpha, ZeroMode);
#else
                        if (ZeroMode) {
                            RowsHandled = MlasSgemmKernelZero(pa, PackedB, c, CountK, RowsTransposed, CountN, CountK, ldc, alpha);
                        } else {
                            RowsHandled = MlasSgemmKernelAdd(pa, PackedB, c, CountK, RowsTransposed, CountN, CountK, ldc, alpha);
                        }
#endif

                        c += ldc * RowsHandled;
                        pa += CountK * RowsHandled;

                        RowsTransposed -= RowsHandled;

                    } while (RowsTransposed > 0);

                } while (RowsRemaining > 0);
            }
        }
    }
}


void
MlasSgemmOperation(
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

    This routine implements the single precision matrix/matrix multiply
    operation (SGEMM).

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
    MlasSgemmOperation(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, nullptr, 
        MLAS_SGEMM_STRIDEN, MLAS_SGEMM_STRIDEK, C, ldc);
}

void
MlasSgemmOperationThreaded(
    void* Context,
    int32_t Index
    )
/*++

Routine Description:

    This routine is invoked from a worker thread to execute a segment of a
    SGEMM operation.

Arguments:

    Context - Supplies the pointer to the context for the threaded operation.

    Index - Supplies the current index of the threaded operation.

Return Value:

    None.

--*/
{
    MLAS_SGEMM_WORK_BLOCK* WorkBlock = (MLAS_SGEMM_WORK_BLOCK*)Context;

    MLAS_SGEMM_WORK_BLOCK::SEGMENT* Segment = &WorkBlock->Segments[Index];

    MlasSgemmOperation(WorkBlock->TransA, WorkBlock->TransB, Segment->M, Segment->N, WorkBlock->K,
        WorkBlock->alpha, Segment->A, WorkBlock->lda,
        Segment->B, WorkBlock->ldb, WorkBlock->beta, Segment->PrePackedB,
        WorkBlock->PackedStrideN, WorkBlock->PackedStrideK,
        Segment->C, WorkBlock->ldc);
}

size_t
MLASCALL
MlasGemmPackMatrix(
    const MLAS_GEMM_PARAMETERS& GemmParams,
    const float* Source,
    CBLAS_TRANSPOSE Trans,
    size_t ld,
    float* Dest,
    size_t DestSize
)
/*++

Routine Description:

    This routine computes the size of the packed representation of
    the source matrix. It also computes the packed representation when possible
    (i.e. Source and Destination are provided, and DestSize is big enough.)

    The source matrix is divided into blocks of GemmParams.PackedStrideK by
    GemmParams.PackedStrideN. Each block is packed by routines
    MlasSgemmCopyPackB or MlasSgemmTransposePackB.

Arguments:

    GemmParams - Supplies parameters of GEMM operation.

    Source - Supplies the optional address of the source matrix.

    Trans - Supplies the transpose operation of the source matrix.

    ld - Supplies the first dimension of the source matrix.

    Dest - Supplies the optional address of the destination matrix.

    DestSize - Supplies the number of elements available in Dest.

Return Value:

    Returns the number of elements in the packed matrix.

--*/
{
    size_t res = 0;
    const size_t K = GemmParams.K;
    const size_t N = GemmParams.N;

    //
    // Step through each slice of the matrix along the N dimension.
    //

    size_t CountN;
    for (size_t n = 0; n < N; n += CountN) {
        CountN = GemmParams.PackedStrideN;
        if (CountN > (N - n)) {
            CountN = N - n;
        }

        //
        // Step through each slice of the matrix along the K dimension.
        //

        size_t CountK;
        for (size_t k = 0; k < K; k += CountK) {
            CountK = GemmParams.PackedStrideK;
            if (CountK > (K - k)) {
                CountK = K - k;
            }

            const size_t CurrentPackedSize = MlasGemmGetPackedSize(CountN, CountK);
            res += CurrentPackedSize;
            if (Source != nullptr && Dest != nullptr && res <= DestSize) {
                if (Trans == CblasNoTrans) {
                    MlasSgemmCopyPackB(Dest, Source + n + k * ld, ld, CountN, CountK);
                } else {
                    MlasSgemmTransposePackB(Dest, Source + k + n * ld, ld, CountN, CountK);
                }

                Dest += CurrentPackedSize;
            }
        }
    }

    return res;
}


MLAS_GEMM_PARAMETERS
MLASCALL
MlasGemmPrepare(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    int MaximumThreadCount,
    bool PrePackingBAllowed
)
/*++

Routine Description:

    This routine chooses parameters of GEMM computation based on matrix
    dimensions.

Arguments:

    TransA - Supplies the transpose operation for matrix A.

    TransB - Supplies the transpose operation for matrix B.

    M - Supplies the number of rows of matrix A and matrix C. Can be 0 if the
        number is not known.

    N - Supplies the number of columns of matrix B and matrix C.

    K - Supplies the number of columns of matrix A and the number of rows of
        matrix B.

    MaximumThreadCount - Supplies the number of threads in the thread pool.

    PrePackingBAllowed - Supplies true if the routine can choose to precompute
                         the packed representation of matrix B; supplies false
                         otherwise.

Return Value:

    Returns the parameters of GEMM computation.

--*/
{
    MLAS_GEMM_PARAMETERS parameters;

    parameters.K = K;
    parameters.N = N;

    MlasGemmChoosePackedStrides(&parameters.PackedStrideN,
        &parameters.PackedStrideK,
        TransA, M, N, K, MaximumThreadCount);

    const size_t ld = TransB == CblasNoTrans ? N : K;
    parameters.PackedSize = MlasGemmPackMatrix(parameters, nullptr, TransB, ld, nullptr, 0);

    if (PrePackingBAllowed && N >= 16) {
        parameters.PrePackedBPossiblyUsed = true;
    }

    if (!parameters.PrePackedBPossiblyUsed) {
        parameters.OriginalBPossiblyUsed = true;
    }

    return parameters;
}


void
MLASCALL
MlasGemm(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    const MLAS_GEMM_PARAMETERS* parameters,
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    const float* A,
    size_t lda,
    const float* B,
    size_t ldb,
    const float* PrePackedB,
    float beta,
    float* C,
    size_t ldc,
    MLAS_THREADPOOL* ThreadPool
    )
/*++

Routine Description:

    This routine implements the single precision matrix/matrix multiply
    operation (SGEMM).

Arguments:

    TransA - Supplies the transpose operation for matrix A.

    TransB - Supplies the transpose operation for matrix B.

    parameters - Supplies the address of GEMM parameters.

    M - Supplies the number of rows of matrix A and matrix C.

    N - Supplies the number of columns of matrix B and matrix C.

    K - Supplies the number of columns of matrix A and the number of rows of
        matrix B.

    alpha - Supplies the scalar alpha multiplier (see SGEMM definition).

    A - Supplies the address of matrix A.

    lda - Supplies the first dimension of matrix A.

    B - Supplies the address of matrix B.

    ldb - Supplies the first dimension of matrix B.

    PrePackedB - Supplies the optional address of packed representation
                 of matrix B, calculated by MlasGemmPackMatrix. 

    beta - Supplies the scalar beta multiplier (see SGEMM definition).

    C - Supplies the address of matrix C.

    ldc - Supplies the first dimension of matrix C.

    ThreadPool - Supplies the thread pool object to use, else nullptr if the
        base library threading support should be used.

Return Value:

    None.

--*/
{
    uint32_t ThreadStrideNAlignment = MLAS_SGEMM_STRIDEN_THREAD_ALIGN;
    if (PrePackedB != nullptr) {
        // We may need to use PrePackedB, so StrideN must be a multiple
        // of PackedStrideN.
        ThreadStrideNAlignment = (std::max)(ThreadStrideNAlignment, parameters->PackedStrideN);
    }

    const int32_t MaximumThreadCount = MlasGetMaximumThreadCount(ThreadPool);
    GEMM_THREADING_STRATEGY ThreadingStrategy = MlasSgemmChooseThreadingStrategy(M, N, K, ThreadStrideNAlignment, MaximumThreadCount);
    const int32_t TargetThreadCount = ThreadingStrategy.NumThreads;

    if (TargetThreadCount == 1) {
        MlasSgemmOperation(TransA, TransB,
            M, parameters->N, parameters->K, 
            alpha, A, lda, B, ldb, beta, 
            PrePackedB, parameters->PackedStrideN, parameters->PackedStrideK,
            C, ldc);
        return;
    }

    MLAS_SGEMM_WORK_BLOCK WorkBlock{};

    //
    // Initialize the common fields of the work block.
    //

    WorkBlock.TransA = TransA;
    WorkBlock.TransB = TransB;
    WorkBlock.K = K;
    WorkBlock.N = N;
    WorkBlock.lda = lda;
    WorkBlock.ldb = ldb;
    WorkBlock.ldc = ldc;
    WorkBlock.alpha = alpha;
    WorkBlock.beta = beta;
    WorkBlock.PackedStrideN = parameters->PackedStrideN;
    WorkBlock.PackedStrideK = parameters->PackedStrideK;

    //
    // Segment the operation across multiple threads.
    //

    int32_t Index = 0;

    if (ThreadingStrategy.SplitMode == GEMM_THREAD_SPLIT_MODE::SplitBCols) {
        const size_t pldb = (TransB == CblasNoTrans) ? 1 : ldb;

        const size_t StrideN = ThreadingStrategy.ThreadStrideN;

        for (size_t CountN, n = 0; n < N; n += CountN) {

            CountN = StrideN;

            if (CountN > (N - n)) {
                CountN = N - n;
            }

            WorkBlock.Segments[Index].M = M;
            WorkBlock.Segments[Index].N = CountN;
            WorkBlock.Segments[Index].A = A;

            if (B == nullptr) {
                WorkBlock.Segments[Index].B = nullptr;
            } else {
                WorkBlock.Segments[Index].B = B + n * pldb;
            }

            if (PrePackedB == nullptr) {
                WorkBlock.Segments[Index].PrePackedB = nullptr;
            } else {
                WorkBlock.Segments[Index].PrePackedB = PrePackedB + n * parameters->K;
            }

            WorkBlock.Segments[Index].C = C + n;

            Index++;
        }

    } else {

        const size_t StrideM = ThreadingStrategy.ThreadStrideM;

        size_t plda = (TransA == CblasNoTrans) ? lda : 1;

        for (size_t CountM, m = 0; m < M; m += CountM) {

            CountM = StrideM;

            if (CountM > (M - m)) {
                CountM = M - m;
            }

            WorkBlock.Segments[Index].M = CountM;
            WorkBlock.Segments[Index].N = N;
            WorkBlock.Segments[Index].A = A + m * plda;
            WorkBlock.Segments[Index].B = B;
            WorkBlock.Segments[Index].PrePackedB = PrePackedB;
            WorkBlock.Segments[Index].C = C + m * ldc;

            Index++;
        }
    }

    MlasExecuteThreaded(MlasSgemmOperationThreaded, &WorkBlock, Index, ThreadPool);
}


void
MLASCALL
MlasGemm(
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
    size_t ldc,
    MLAS_THREADPOOL* ThreadPool
    )
/*++

Routine Description:

    This routine implements the single precision matrix/matrix multiply
    operation (SGEMM).

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

    ThreadPool - Supplies the thread pool object to use, else nullptr if the
        base library threading support should be used.

Return Value:

    None.

--*/
{
    const int MaximumThreadCount = MlasGetMaximumThreadCount(ThreadPool);
    MLAS_GEMM_PARAMETERS parameters = MlasGemmPrepare(TransA, TransB, M, N, K, MaximumThreadCount, false);
    MlasGemm(TransA, TransB, &parameters, M, N, K, alpha, A, lda, B, ldb, nullptr, beta, C, ldc, ThreadPool);
}

