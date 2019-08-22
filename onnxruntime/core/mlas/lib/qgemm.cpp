/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm.cpp

Abstract:

    This module implements the quantized integer matrix/matrix multiply
    operation (QGEMM).

--*/

#include "mlasi.h"

//
// Define the default strides to step through slices of the input matrices.
//

#define MLAS_GEMM_U8U8_STRIDEM              12
#define MLAS_GEMM_U8U8_STRIDEN              128
#define MLAS_GEMM_U8U8_STRIDEK              128

#ifdef MLAS_TARGET_AMD64_IX86

void
MLASCALL
MlasGemmU8U8CopyPackASse(
    int16_t* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumVector,
    uint16_t offb
    )
/*++

Routine Description:

    This routine copies elements from the source matrix to the destination
    packed buffer.

Arguments:

    D - Supplies the address of the destination packed buffer.

    A - Supplies the address of the source matrix.

    lda - Supplies the number of elements per row of the source matrix.

    CountM - Supplies the number of rows of the source matrix to copy.

    CountK - Supplies the number of columns of the source matrix to copy.

    RowSumVector - Supplies the address of the buffer to receive the sums of
        the elements from each of the rows. Each sum has also been multiplied
        by the zero point offset.

    offb - Supplies the zero point offset for the other source matrix of the
        matrix multiplication.

Return Value:

    None.

--*/
{
    const __m128i ZeroVector = _mm_setzero_si128();
    const __m128i OffsetBroadcast = _mm_set1_epi16(offb);
    uint8_t PaddedMatrixAData[8] = { 0 };

    //
    // Process a single row of matrix A in a loop.
    //

    while (CountM > 0) {

        const uint8_t* a = A;
        size_t k = CountK;
        __m128i RowSum = ZeroVector;

        //
        // Zero extend the source bytes to 16-bits and write to the packed
        // buffer. The packed buffer has the same data ordering as the source
        // bytes, but the stride is CountK aligned up to a multiple of 8
        // values.
        //
        // These 16-bit values are also accumulated into an intermediate per-row
        // accumulator. CountK cannot be greater than 256 to avoid overflowing
        // these 16-bit accumulators.
        //

        while (k >= 8) {

            __m128i Bytes = _mm_loadl_epi64((__m128i*)&a[0]);
            __m128i Words = _mm_unpacklo_epi8(Bytes, ZeroVector);

            RowSum = _mm_add_epi16(RowSum, Words);

            _mm_storeu_si128((__m128i*)&D[0], Words);

            D += 8;
            a += 8;
            k -= 8;
        }

        if (k > 0) {

            //
            // Copy the remaining bytes to the zero padded stack buffer.
            //

            uint8_t* padded = PaddedMatrixAData;
            uint8_t* padded_end = padded + k;

            do {
                padded[0] = a[0];
                padded++;
                a++;
            } while (padded < padded_end);

            __m128i Bytes = _mm_loadl_epi64((__m128i*)PaddedMatrixAData);
            __m128i Words = _mm_unpacklo_epi8(Bytes, ZeroVector);

            RowSum = _mm_add_epi16(RowSum, Words);

            //
            // Copy the 16-bit pairs from the vector to the destination packed
            // buffer. Rotate the vector at each iteration.
            //

            for (size_t pairs = (k + 1) / 2; pairs > 0; pairs--) {
                *((int32_t*)D) = _mm_cvtsi128_si32(Words);
                D += 2;
                Words = _mm_shuffle_epi32(Words, _MM_SHUFFLE(0, 3, 2, 1));
            }
        }

        //
        // Reduce the sum for the single row of output.
        //

        RowSum = _mm_madd_epi16(RowSum, OffsetBroadcast);
        RowSum = _mm_add_epi32(RowSum, _mm_shuffle_epi32(RowSum, _MM_SHUFFLE(3, 2, 3, 2)));
        RowSum = _mm_add_epi32(RowSum, _mm_shuffle_epi32(RowSum, _MM_SHUFFLE(0, 1, 0, 1)));

        *RowSumVector++ = _mm_cvtsi128_si32(RowSum);

        A += lda;
        CountM -= 1;
    }
}

void
MLASCALL
MlasGemmU8U8CopyPackBSse(
    uint8_t* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumVector,
    uint16_t offa
    )
/*++

Routine Description:

    This routine copies elements from the source matrix to the destination
    packed buffer.

Arguments:

    D (rcx) - Supplies the address of the destination packed buffer.

    B (rdx) - Supplies the address of the source matrix.

    ldb (r8) - Supplies the number of elements per row of the source matrix.

    CountN (r9) - Supplies the number of columns of the source matrix to copy.

    CountK - Supplies the number of rows of the source matrix to copy.

    ColumnSumVector - Supplies the address of the buffer to receive the sums of
        the elements from each of the columns. Each sum has also been multiplied
        by the zero point offset.

    offa - Supplies the zero point offset for the other source matrix of the
        matrix multiplication.

Return Value:

    None.

--*/
{
    const __m128i ZeroVector = _mm_setzero_si128();
    const __m128i OffsetBroadcast = _mm_set1_epi16(offa);
    uint8_t PaddedMatrixBData[16] = { 0 };

    //
    // Process 8 columns of matrix B in a loop.
    //

    while (CountN >= 8) {

        const uint8_t* b = B;
        size_t k = CountK;
        __m128i ColumnSum0 = ZeroVector;
        __m128i ColumnSum1 = ZeroVector;

        while (k >= 2) {

            __m128i BytesRow0 = _mm_loadl_epi64((__m128i*)&b[0]);
            __m128i BytesRow1 = _mm_loadl_epi64((__m128i*)&b[ldb]);
            __m128i BytesInterleaved = _mm_unpacklo_epi8(BytesRow0, BytesRow1);

            _mm_storeu_si128((__m128i*)&D[0], BytesInterleaved);

            ColumnSum0 = _mm_add_epi16(ColumnSum0, _mm_unpacklo_epi8(BytesInterleaved, ZeroVector));
            ColumnSum1 = _mm_add_epi16(ColumnSum1, _mm_unpackhi_epi8(BytesInterleaved, ZeroVector));

            b += ldb * 2;
            D += 16;
            k -= 2;
        }

        if (k > 0) {

            __m128i BytesRow0 = _mm_loadl_epi64((__m128i*)&b[0]);
            __m128i BytesInterleaved = _mm_unpacklo_epi8(BytesRow0, ZeroVector);

            _mm_storeu_si128((__m128i*)&D[0], BytesInterleaved);

            ColumnSum0 = _mm_add_epi16(ColumnSum0, _mm_unpacklo_epi8(BytesInterleaved, ZeroVector));
            ColumnSum1 = _mm_add_epi16(ColumnSum1, _mm_unpackhi_epi8(BytesInterleaved, ZeroVector));

            b += ldb * 2;
            D += 16;
        }

        ColumnSum0 = _mm_madd_epi16(ColumnSum0, OffsetBroadcast);
        ColumnSum1 = _mm_madd_epi16(ColumnSum1, OffsetBroadcast);

        _mm_storeu_si128((__m128i*)&ColumnSumVector[0], ColumnSum0);
        _mm_storeu_si128((__m128i*)&ColumnSumVector[4], ColumnSum1);

        ColumnSumVector += 8;

        B += 8;
        CountN -= 8;
    }

    //
    // Process the remaining columns of matrix B.
    //

    if (CountN > 0) {

        const uint8_t* b = B;
        size_t k = CountK;
        __m128i ColumnSum0 = ZeroVector;
        __m128i ColumnSum1 = ZeroVector;

        while (k >= 2) {

            //
            // Copy the remaining columns to the zero padded stack buffer.
            //

            const uint8_t* bcopy = b;
            uint8_t* padded = PaddedMatrixBData;
            uint8_t* padded_end = padded + CountN;

            do {
                padded[0] = bcopy[0];
                padded[8] = bcopy[ldb];
                padded++;
                bcopy++;
            } while (padded < padded_end);

            __m128i BytesRow0 = _mm_loadl_epi64((__m128i*)&PaddedMatrixBData[0]);
            __m128i BytesRow1 = _mm_loadl_epi64((__m128i*)&PaddedMatrixBData[8]);
            __m128i BytesInterleaved = _mm_unpacklo_epi8(BytesRow0, BytesRow1);

            _mm_storeu_si128((__m128i*)&D[0], BytesInterleaved);

            ColumnSum0 = _mm_add_epi16(ColumnSum0, _mm_unpacklo_epi8(BytesInterleaved, ZeroVector));
            ColumnSum1 = _mm_add_epi16(ColumnSum1, _mm_unpackhi_epi8(BytesInterleaved, ZeroVector));

            b += ldb * 2;
            D += 16;
            k -= 2;
        }

        if (k > 0) {

            //
            // Copy the remaining columns to the zero padded stack buffer.
            //

            const uint8_t* bcopy = b;
            uint8_t* padded = PaddedMatrixBData;
            uint8_t* padded_end = padded + CountN;

            do {
                padded[0] = bcopy[0];
                padded++;
                bcopy++;
            } while (padded < padded_end);

            __m128i BytesRow0 = _mm_loadl_epi64((__m128i*)&PaddedMatrixBData[0]);
            __m128i BytesInterleaved = _mm_unpacklo_epi8(BytesRow0, ZeroVector);

            _mm_storeu_si128((__m128i*)&D[0], BytesInterleaved);

            ColumnSum0 = _mm_add_epi16(ColumnSum0, _mm_unpacklo_epi8(BytesInterleaved, ZeroVector));
            ColumnSum1 = _mm_add_epi16(ColumnSum1, _mm_unpackhi_epi8(BytesInterleaved, ZeroVector));
        }

        ColumnSum0 = _mm_madd_epi16(ColumnSum0, OffsetBroadcast);
        ColumnSum1 = _mm_madd_epi16(ColumnSum1, OffsetBroadcast);

        _mm_storeu_si128((__m128i*)&ColumnSumVector[0], ColumnSum0);
        _mm_storeu_si128((__m128i*)&ColumnSumVector[4], ColumnSum1);
    }
}

size_t
MLASCALL
MlasGemmU8U8KernelSse(
    const int16_t* A,
    const uint8_t* B,
    int32_t* C,
    size_t PairedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumVector,
    const int32_t* ColumnSumVector,
    int32_t DepthValue,
    bool ZeroMode
    )
/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows.

Arguments:

    A - Supplies the address of matrix A. The matrix data has been packed
        using MlasGemmU8U8CopyPackASse.

    B - Supplies the address of matrix B. The matrix data has been packed
        using MlasGemmU8U8CopyPackBSse.

    C - Supplies the address of matrix C.

    PairedCountK - Supplies the number of paired columns from matrix A and
        the number of paired rows from matrix B to iterate over.

    CountM - Supplies the maximum number of rows that can be processed for
        matrix A and matrix C. The actual number of rows handled for this
        invocation depends on the kernel implementation.

    CountN - Supplies the number of columns from matrix B and matrix C to iterate
        over.

    ldc - Supplies the first dimension of matrix C.

    RowSumVector - Supplies the sum of each row from matrix A multiplied by the
        zero point offset of matrix B. These values are accumulated into every
        row of matrix C.

    ColumnSumVector - Supplies the sum of each column from matrix B multiplied
        by the zero point offset of matrix A. These values are accumulated into
        every column of matrix C.

    DepthValue - Supplies the value CountK multiplied by the zero point offset
        of matrixA multplied by the zero point offset of matrix B. This value is
        accumulated into every element of matrix C.

    ZeroMode - Supplies true if the output matrix must be zero initialized,
        else false if the output matrix is accumulated into.

Return Value:

    Returns the number of rows handled.

--*/
{
    const __m128i ZeroVector = _mm_setzero_si128();

    MLAS_UNREFERENCED_PARAMETER(CountM);
    MLAS_UNREFERENCED_PARAMETER(ldc);

    while (CountN > 0) {

        //
        // Initialize the accumulators with the sum of the global depth value
        // constant, the column sums, and the row sums.
        //

        __m128i Accumulator0 = _mm_set1_epi32(DepthValue);
        Accumulator0 = _mm_add_epi32(Accumulator0, _mm_set1_epi32(RowSumVector[0]));
        __m128i Accumulator1 = Accumulator0;
        Accumulator0 = _mm_add_epi32(Accumulator0, _mm_loadu_si128((__m128i*)&ColumnSumVector[0]));
        Accumulator1 = _mm_add_epi32(Accumulator1, _mm_loadu_si128((__m128i*)&ColumnSumVector[4]));
        ColumnSumVector += 8;

        //
        // Broadcast each pair of 16-bit values from the matrix A and multiply
        // with the zero-extended pair of 16-bit values from matrix B, and add
        // the 32-bit intermediate into the accumulator registers.
        //

        const int16_t* a = A;
        size_t k = PairedCountK;

        while (k > 0) {

            __m128i AElements0 = _mm_set1_epi32(*((int32_t*)a));
            __m128i BElements0 = _mm_loadu_si128((__m128i*)&B[0]);

            __m128i Intermediate0 = _mm_unpacklo_epi8(BElements0, ZeroVector);
            __m128i Intermediate1 = _mm_unpackhi_epi8(BElements0, ZeroVector);

            Intermediate0 = _mm_madd_epi16(Intermediate0, AElements0);
            Intermediate1 = _mm_madd_epi16(Intermediate1, AElements0);

            Accumulator0 = _mm_add_epi32(Accumulator0, Intermediate0);
            Accumulator1 = _mm_add_epi32(Accumulator1, Intermediate1);

            a += 2;
            B += 16;
            k -= 1;
        }

        //
        // Output the accumulator block after optionally accumulating the values
        // from matrix C.
        //

        if (CountN >= 8) {

            if (!ZeroMode) {
                Accumulator0 = _mm_add_epi32(Accumulator0, _mm_loadu_si128((__m128i*)&C[0]));
                Accumulator1 = _mm_add_epi32(Accumulator1, _mm_loadu_si128((__m128i*)&C[4]));
            }

            _mm_storeu_si128((__m128i*)&C[0], Accumulator0);
            _mm_storeu_si128((__m128i*)&C[4], Accumulator1);

            C += 8;
            CountN -= 8;

        } else {

            //
            // Output the remaining partial output block.
            //

            if ((CountN & 4) != 0) {

                if (!ZeroMode) {
                    Accumulator0 = _mm_add_epi32(Accumulator0, _mm_loadu_si128((__m128i*)&C[0]));
                }

                _mm_storeu_si128((__m128i*)&C[0], Accumulator0);
                C += 4;

                Accumulator0 = Accumulator1;
            }

            if ((CountN & 2) != 0) {

                if (!ZeroMode) {
                    Accumulator0 = _mm_add_epi32(Accumulator0, _mm_loadl_epi64((__m128i*)&C[0]));
                }

                _mm_storel_epi64((__m128i*)&C[0], Accumulator0);
                C += 2;

                Accumulator0 = _mm_shuffle_epi32(Accumulator0, _MM_SHUFFLE(1, 0, 3, 2));
            }

            if ((CountN & 1) != 0) {

                int32_t AccumulatorValue = _mm_cvtsi128_si32(Accumulator0);

                if (!ZeroMode) {
                    AccumulatorValue += C[0];
                }

                C[0] = AccumulatorValue;
            }

            break;
        }
    }

    return 1;
}

void
MLASCALL
MlasQgemm(
    size_t M,
    size_t N,
    size_t K,
    const uint8_t* A,
    size_t lda,
    uint8_t offa,
    const uint8_t* B,
    size_t ldb,
    uint8_t offb,
    int32_t* C,
    size_t ldc,
    MLAS_THREADPOOL* ThreadPool
    )
{
    MLAS_DECLSPEC_ALIGN(int16_t PanelA[MLAS_GEMM_U8U8_STRIDEM * MLAS_GEMM_U8U8_STRIDEK], 64);
    MLAS_DECLSPEC_ALIGN(uint8_t PanelB[MLAS_GEMM_U8U8_STRIDEN * MLAS_GEMM_U8U8_STRIDEK], 64);

    MLAS_DECLSPEC_ALIGN(int32_t RowSumVector[MLAS_GEMM_U8U8_STRIDEM], 16);
    MLAS_DECLSPEC_ALIGN(int32_t ColumnSumVector[MLAS_GEMM_U8U8_STRIDEN], 16);

    size_t StrideM = MLAS_GEMM_U8U8_STRIDEM;
    size_t StrideN = MLAS_GEMM_U8U8_STRIDEN;
    size_t StrideK = MLAS_GEMM_U8U8_STRIDEK;

    MLAS_UNREFERENCED_PARAMETER(ThreadPool);

    size_t CountK;

    for (size_t k = 0; k < K; k += CountK) {

        CountK = StrideK;

        if (CountK > (K - k)) {
            CountK = K - k;
        }

        size_t CountN;

        for (size_t n = 0; n < N; n += CountN) {

            CountN = StrideN;

            if (CountN > (N - n)) {
                CountN = N - n;
            }

            MlasPlatform.GemmU8U8CopyPackBRoutine(PanelB, B + n + k * ldb, ldb, CountN, CountK, ColumnSumVector, offa);

            size_t CountM;

            for (size_t m = 0; m < M; m += CountM) {

                CountM = StrideM;

                if (CountM > (M - m)) {
                    CountM = M - m;
                }

                MlasPlatform.GemmU8U8CopyPackARoutine(PanelA, A + k + m * lda, lda, CountM, CountK, RowSumVector, offb);

                int16_t* pa = PanelA;
                int32_t* c = C + n + m * ldc;

                int32_t* RowSums = RowSumVector;

                size_t RowsRemaining = CountM;
                size_t RowsHandled;

                size_t PairedCountK = (CountK + 1) / 2;

                while (RowsRemaining > 0) {

                    RowsHandled = MlasPlatform.GemmU8U8Kernel(pa, PanelB, c, PairedCountK, RowsRemaining, CountN, ldc, RowSums, ColumnSumVector, int32_t(CountK) * offa * offb, k == 0);

                    RowsRemaining -= RowsHandled;
                    c += ldc * RowsHandled;
                    pa += 2 * PairedCountK * RowsHandled;
                    RowSums += RowsHandled;
                }
            }
        }
    }
}

#endif
