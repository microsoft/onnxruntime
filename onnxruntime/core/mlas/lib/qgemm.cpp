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

#define MLAS_GEMM_X8X8_STRIDEM              24
#define MLAS_GEMM_X8X8_STRIDEN              256
#define MLAS_GEMM_X8X8_STRIDEK              128

//
// Define the parameters to execute segments of a QGEMM operation on worker
// threads.
//

struct MLAS_GEMM_X8X8_WORK_BLOCK {
    PMLAS_GEMM_X8X8_OPERATION GemmX8X8Operation;
    size_t M;
    size_t N;
    size_t K;
    const uint8_t* A;
    size_t lda;
    const uint8_t* B;
    size_t ldb;
    int32_t* C;
    size_t ldc;
    int32_t ThreadCountM;
    int32_t ThreadCountN;
    size_t StrideM;
    size_t StrideN;
    int16_t offa;
    int16_t offb;
};

#ifdef MLAS_TARGET_AMD64_IX86

//
// Stores a vector to transpose a 4x4 byte vector using vpshufb.
//

MLAS_INTERNAL_DATA MLAS_DECLSPEC_ALIGN(const uint8_t MlasTranspose4x4BytesAvx[16], 16) =
    { 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 };

//
// U8S8 implementation using SSE2 intrinsics.
//

void
MLASCALL
MlasGemmU8S8CopyPackASse(
    uint8_t* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumVector,
    int16_t offb
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
        // Copy the source bytes to the packed buffer.
        //
        // The packed buffer has the same data ordering as the source bytes,
        // but CountK is aligned up to a multiple of 4 to maintain 32-bit
        // alignment. All extra bytes are zero-padded.
        //
        // These values are also zero-extended and accumulated into an
        // intermediate per-row accumulator. CountK cannot be greater than 128
        // to avoid overflowing these signed 16-bit accumulators.
        //

        while (k >= 8) {

            __m128i Bytes = _mm_loadl_epi64((__m128i*)&a[0]);
            _mm_storel_epi64((__m128i*)&D[0], Bytes);

            RowSum = _mm_add_epi16(RowSum, _mm_unpacklo_epi8(Bytes, ZeroVector));

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
            _mm_storel_epi64((__m128i*)&D[0], Bytes);

            RowSum = _mm_add_epi16(RowSum, _mm_unpacklo_epi8(Bytes, ZeroVector));

            //
            // Copy quads of 8-bit values from the vector to the packed
            // buffer and rotate the vector for the next iteration.
            //

            for (size_t quads = (k + 3) / 4; quads > 0; quads--) {
                *((int32_t*)D) = _mm_cvtsi128_si32(Bytes);
                D += 4;
                Bytes = _mm_shuffle_epi32(Bytes, _MM_SHUFFLE(0, 3, 2, 1));
            }
        }

        //
        // Reduce the sum for the single row of output and multiply by the
        // zero point offset of the other source matrix.
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
MlasGemmU8S8CopyPackBSse(
    int8_t* D,
    const int8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumVector,
    int16_t offa
    )
/*++

Routine Description:

    This routine copies elements from the source matrix to the destination
    packed buffer.

Arguments:

    D - Supplies the address of the destination packed buffer.

    B - Supplies the address of the source matrix.

    ldb - Supplies the number of elements per row of the source matrix.

    CountN - Supplies the number of columns of the source matrix to copy.

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
    int8_t PaddedMatrixBData[16] = { 0 };

    //
    // Process 8 columns of matrix B in a loop.
    //

    while (CountN >= 8) {

        const int8_t* b = B;
        size_t k = CountK;
        __m128i ColumnSum0 = ZeroVector;
        __m128i ColumnSum1 = ZeroVector;

        //
        // Interleave 2 rows of matrix B and write to the packed buffer.
        //
        // These values are also sign-extended and accumulated into an
        // intermediate per-column accumulator. CountK cannot be greater than
        // 128 to avoid overflowing these signed 16-bit accumulators.
        //

        while (k >= 2) {

            __m128i BytesRow0 = _mm_loadl_epi64((__m128i*)&b[0]);
            __m128i BytesRow1 = _mm_loadl_epi64((__m128i*)&b[ldb]);
            __m128i BytesInterleaved = _mm_unpacklo_epi8(BytesRow0, BytesRow1);

            _mm_storeu_si128((__m128i*)&D[0], BytesInterleaved);

            __m128i WordsLow = _mm_srai_epi16(_mm_unpacklo_epi8(ZeroVector, BytesInterleaved), 8);
            ColumnSum0 = _mm_add_epi16(ColumnSum0, WordsLow);
            __m128i WordsHigh = _mm_srai_epi16(_mm_unpackhi_epi8(ZeroVector, BytesInterleaved), 8);
            ColumnSum1 = _mm_add_epi16(ColumnSum1, WordsHigh);

            b += ldb * 2;
            D += 16;
            k -= 2;
        }

        if (k > 0) {

            //
            // Process the remaining row of matrix B.
            //

            __m128i BytesRow0 = _mm_loadl_epi64((__m128i*)&b[0]);
            __m128i BytesInterleaved = _mm_unpacklo_epi8(BytesRow0, ZeroVector);

            _mm_storeu_si128((__m128i*)&D[0], BytesInterleaved);

            __m128i WordsLow = _mm_srai_epi16(_mm_unpacklo_epi8(ZeroVector, BytesInterleaved), 8);
            ColumnSum0 = _mm_add_epi16(ColumnSum0, WordsLow);
            __m128i WordsHigh = _mm_srai_epi16(_mm_unpackhi_epi8(ZeroVector, BytesInterleaved), 8);
            ColumnSum1 = _mm_add_epi16(ColumnSum1, WordsHigh);

            D += 16;
        }

        //
        // The number of rows written to the packed buffer should be a multiple
        // of 4. Zero pad the packed buffer if the block is not complete.
        //

        if (((CountK - 1) & 2) == 0) {

            _mm_storeu_si128((__m128i*)&D[0], ZeroVector);

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

        const int8_t* b = B;
        size_t k = CountK;
        __m128i ColumnSum0 = ZeroVector;
        __m128i ColumnSum1 = ZeroVector;

        //
        // Interleave 2 rows of matrix B and write to the packed buffer.
        //
        // These values are also sign-extended and accumulated into an
        // intermediate per-column accumulator. CountK cannot be greater than
        // 128 to avoid overflowing these signed 16-bit accumulators.
        //

        while (k >= 2) {

            //
            // Copy the remaining columns to the zero padded stack buffer.
            //

            const int8_t* bcopy = b;
            int8_t* padded = PaddedMatrixBData;
            int8_t* padded_end = padded + CountN;

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

            __m128i WordsLow = _mm_srai_epi16(_mm_unpacklo_epi8(ZeroVector, BytesInterleaved), 8);
            ColumnSum0 = _mm_add_epi16(ColumnSum0, WordsLow);
            __m128i WordsHigh = _mm_srai_epi16(_mm_unpackhi_epi8(ZeroVector, BytesInterleaved), 8);
            ColumnSum1 = _mm_add_epi16(ColumnSum1, WordsHigh);

            b += ldb * 2;
            D += 16;
            k -= 2;
        }

        if (k > 0) {

            //
            // Copy the remaining columns to the zero padded stack buffer.
            //

            const int8_t* bcopy = b;
            int8_t* padded = PaddedMatrixBData;
            int8_t* padded_end = padded + CountN;

            do {
                padded[0] = bcopy[0];
                padded++;
                bcopy++;
            } while (padded < padded_end);

            __m128i BytesRow0 = _mm_loadl_epi64((__m128i*)&PaddedMatrixBData[0]);
            __m128i BytesInterleaved = _mm_unpacklo_epi8(BytesRow0, ZeroVector);

            _mm_storeu_si128((__m128i*)&D[0], BytesInterleaved);

            __m128i WordsLow = _mm_srai_epi16(_mm_unpacklo_epi8(ZeroVector, BytesInterleaved), 8);
            ColumnSum0 = _mm_add_epi16(ColumnSum0, WordsLow);
            __m128i WordsHigh = _mm_srai_epi16(_mm_unpackhi_epi8(ZeroVector, BytesInterleaved), 8);
            ColumnSum1 = _mm_add_epi16(ColumnSum1, WordsHigh);

            D += 16;
        }

        //
        // The number of rows written to the packed buffer should be a multiple
        // of 4. Zero pad the packed buffer if the block is not complete.
        //

        if (((CountK - 1) & 2) == 0) {

            _mm_storeu_si128((__m128i*)&D[0], ZeroVector);

            D += 16;
        }

        ColumnSum0 = _mm_madd_epi16(ColumnSum0, OffsetBroadcast);
        ColumnSum1 = _mm_madd_epi16(ColumnSum1, OffsetBroadcast);

        _mm_storeu_si128((__m128i*)&ColumnSumVector[0], ColumnSum0);
        _mm_storeu_si128((__m128i*)&ColumnSumVector[4], ColumnSum1);
    }
}

size_t
MLASCALL
MlasGemmU8S8KernelSse(
    const uint8_t* A,
    const int8_t* B,
    int32_t* C,
    size_t PairCountK,
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
        using MlasGemmU8S8CopyPackASse.

    B - Supplies the address of matrix B. The matrix data has been packed
        using MlasGemmU8S8CopyPackBSse.

    C - Supplies the address of matrix C.

    PairCountK - Supplies the number of paired columns from matrix A and the
        number of paired rows from matrix B to iterate over.

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

        const uint8_t* a = A;
        size_t k = PairCountK;

        while (k > 0) {

            __m128i AElements = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*((int32_t*)a)), ZeroVector);

            __m128i BElements;
            __m128i Intermediate0;
            __m128i Intermediate1;

            BElements = _mm_loadu_si128((__m128i*)&B[0]);
            Intermediate0 = _mm_srai_epi16(_mm_unpacklo_epi8(ZeroVector, BElements), 8);
            Intermediate1 = _mm_srai_epi16(_mm_unpackhi_epi8(ZeroVector, BElements), 8);

            __m128i AElements0 = _mm_shuffle_epi32(AElements, _MM_SHUFFLE(0, 0, 0, 0));

            Intermediate0 = _mm_madd_epi16(Intermediate0, AElements0);
            Intermediate1 = _mm_madd_epi16(Intermediate1, AElements0);

            Accumulator0 = _mm_add_epi32(Accumulator0, Intermediate0);
            Accumulator1 = _mm_add_epi32(Accumulator1, Intermediate1);

            BElements = _mm_loadu_si128((__m128i*)&B[16]);
            Intermediate0 = _mm_srai_epi16(_mm_unpacklo_epi8(ZeroVector, BElements), 8);
            Intermediate1 = _mm_srai_epi16(_mm_unpackhi_epi8(ZeroVector, BElements), 8);

            __m128i AElements1 = _mm_shuffle_epi32(AElements, _MM_SHUFFLE(1, 1, 1, 1));

            Intermediate0 = _mm_madd_epi16(Intermediate0, AElements1);
            Intermediate1 = _mm_madd_epi16(Intermediate1, AElements1);

            Accumulator0 = _mm_add_epi32(Accumulator0, Intermediate0);
            Accumulator1 = _mm_add_epi32(Accumulator1, Intermediate1);

            a += 4;
            B += 32;
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

            CountN = 0;
        }
    }

    return 1;
}

//
// U8U8 implementation using SSE2 intrinsics.
//

void
MLASCALL
MlasGemmU8U8CopyPackASse(
    int16_t* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumVector,
    int16_t offb
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
        // buffer.
        //
        // The packed buffer has the same data ordering as the source bytes,
        // but CountK is aligned up to a multiple of 2 to maintain 32-bit
        // alignment. All extra bytes are zero-padded.
        //
        // These 16-bit values are also accumulated into an intermediate per-row
        // accumulator. CountK cannot be greater than 128 to avoid overflowing
        // these signed 16-bit accumulators.
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
            // Copy pairs of 16-bit values from the vector to the packed
            // buffer and rotate the vector for the next iteration.
            //

            for (size_t pairs = (k + 1) / 2; pairs > 0; pairs--) {
                *((int32_t*)D) = _mm_cvtsi128_si32(Words);
                D += 2;
                Words = _mm_shuffle_epi32(Words, _MM_SHUFFLE(0, 3, 2, 1));
            }
        }

        //
        // Reduce the sum for the single row of output and multiply by the
        // zero point offset of the other source matrix.
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
    int16_t offa
    )
/*++

Routine Description:

    This routine copies elements from the source matrix to the destination
    packed buffer.

Arguments:

    D - Supplies the address of the destination packed buffer.

    B - Supplies the address of the source matrix.

    ldb - Supplies the number of elements per row of the source matrix.

    CountN - Supplies the number of columns of the source matrix to copy.

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

        //
        // Interleave 2 rows of matrix B and write to the packed buffer.
        //
        // These values are also zero-extended and accumulated into an
        // intermediate per-column accumulator. CountK cannot be greater than
        // 128 to avoid overflowing these signed 16-bit accumulators.
        //

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

            //
            // Process the remaining row of matrix B.
            //

            __m128i BytesRow0 = _mm_loadl_epi64((__m128i*)&b[0]);
            __m128i BytesInterleaved = _mm_unpacklo_epi8(BytesRow0, ZeroVector);

            _mm_storeu_si128((__m128i*)&D[0], BytesInterleaved);

            ColumnSum0 = _mm_add_epi16(ColumnSum0, _mm_unpacklo_epi8(BytesInterleaved, ZeroVector));
            ColumnSum1 = _mm_add_epi16(ColumnSum1, _mm_unpackhi_epi8(BytesInterleaved, ZeroVector));

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

        //
        // Reduce the sum for the packed columns and multiply by the zero point
        // offset of the other source matrix.
        //

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
    size_t PairCountK,
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

    PairCountK - Supplies the number of paired columns from matrix A and the
        number of paired rows from matrix B to iterate over.

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
        size_t k = PairCountK;

        while (k > 0) {

            __m128i AElements = _mm_set1_epi32(*((int32_t*)a));
            __m128i BElements0 = _mm_loadu_si128((__m128i*)&B[0]);

            __m128i Intermediate0 = _mm_unpacklo_epi8(BElements0, ZeroVector);
            __m128i Intermediate1 = _mm_unpackhi_epi8(BElements0, ZeroVector);

            Intermediate0 = _mm_madd_epi16(Intermediate0, AElements);
            Intermediate1 = _mm_madd_epi16(Intermediate1, AElements);

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

            CountN = 0;
        }
    }

    return 1;
}

void
MLASCALL
MlasGemmU8S8Operation(
    size_t M,
    size_t N,
    size_t K,
    const uint8_t* A,
    size_t lda,
    int16_t offa,
    const uint8_t* B,
    size_t ldb,
    int16_t offb,
    int32_t* C,
    size_t ldc
    )
/*++

Routine Description:

    This module implements the quantized integer matrix/matrix multiply
    operation (QGEMM).

Arguments:

    M - Supplies the number of rows of matrix A and matrix C.

    N - Supplies the number of columns of matrix B and matrix C.

    K - Supplies the number of columns of matrix A and the number of rows of
        matrix B.

    A - Supplies the address of matrix A.

    lda - Supplies the first dimension of matrix A.

    offa - Supplies the zero point offset of matrix A.

    B - Supplies the address of matrix B.

    ldb - Supplies the first dimension of matrix B.

    offb - Supplies the zero point offset of matrix B.

    C - Supplies the address of matrix C.

    ldc - Supplies the first dimension of matrix C.

Return Value:

    None.

--*/
{
    MLAS_DECLSPEC_ALIGN(uint8_t PanelA[MLAS_GEMM_X8X8_STRIDEM * MLAS_GEMM_X8X8_STRIDEK], 64);
    MLAS_DECLSPEC_ALIGN(int8_t PanelB[MLAS_GEMM_X8X8_STRIDEN * MLAS_GEMM_X8X8_STRIDEK], 64);

    MLAS_DECLSPEC_ALIGN(int32_t RowSumVector[MLAS_GEMM_X8X8_STRIDEM], 16);
    MLAS_DECLSPEC_ALIGN(int32_t ColumnSumVector[MLAS_GEMM_X8X8_STRIDEN], 16);

    size_t StrideM = MLAS_GEMM_X8X8_STRIDEM;
    size_t StrideN = MLAS_GEMM_X8X8_STRIDEN;
    size_t StrideK = MLAS_GEMM_X8X8_STRIDEK;

#if defined(MLAS_TARGET_AMD64)

    if (M == 1 && offa == 0 && offb == 0) {

        if (MlasPlatform.GemvU8S8Kernel != nullptr) {
            MlasPlatform.GemvU8S8Kernel(A, (const int8_t*)B, C, K, N, ldb);
            return;
        }
    }

#endif

    //
    // Step through each slice of matrix B along the K dimension.
    //

    size_t CountK;

    for (size_t k = 0; k < K; k += CountK) {

        CountK = StrideK;

        if (CountK > (K - k)) {
            CountK = K - k;
        }

        //
        // Step through each slice of matrix B along the N dimension.
        //

        size_t CountN;

        for (size_t n = 0; n < N; n += CountN) {

            CountN = StrideN;

            if (CountN > (N - n)) {
                CountN = N - n;
            }

            const int8_t* b = (const int8_t*)B + n + k * ldb;

            MlasPlatform.GemmU8S8CopyPackBRoutine(PanelB, b, ldb, CountN,
                CountK, ColumnSumVector, -int16_t(offa));

            size_t CountM;

            for (size_t m = 0; m < M; m += CountM) {

                CountM = StrideM;

                if (CountM > (M - m)) {
                    CountM = M - m;
                }

                MlasPlatform.GemmU8S8CopyPackARoutine(PanelA, A + k + m * lda,
                    lda, CountM, CountK, RowSumVector, -int16_t(offb));

                uint8_t* pa = PanelA;
                int32_t* c = C + n + m * ldc;

                int32_t* RowSums = RowSumVector;

                size_t RowsRemaining = CountM;
                size_t RowsHandled;

                size_t QuadCountK = (CountK + 3) / 4;

                while (RowsRemaining > 0) {

                    RowsHandled = MlasPlatform.GemmU8S8Kernel(pa, PanelB, c,
                        QuadCountK, RowsRemaining, CountN, ldc, RowSums,
                        ColumnSumVector, int32_t(CountK) * offa * offb, k == 0);

                    RowsRemaining -= RowsHandled;
                    c += ldc * RowsHandled;
                    pa += 4 * QuadCountK * RowsHandled;
                    RowSums += RowsHandled;
                }
            }
        }
    }
}

void
MLASCALL
MlasGemmU8U8Operation(
    size_t M,
    size_t N,
    size_t K,
    const uint8_t* A,
    size_t lda,
    int16_t offa,
    const uint8_t* B,
    size_t ldb,
    int16_t offb,
    int32_t* C,
    size_t ldc
    )
/*++

Routine Description:

    This module implements the quantized integer matrix/matrix multiply
    operation (QGEMM).

Arguments:

    M - Supplies the number of rows of matrix A and matrix C.

    N - Supplies the number of columns of matrix B and matrix C.

    K - Supplies the number of columns of matrix A and the number of rows of
        matrix B.

    A - Supplies the address of matrix A.

    lda - Supplies the first dimension of matrix A.

    offa - Supplies the zero point offset of matrix A.

    B - Supplies the address of matrix B.

    ldb - Supplies the first dimension of matrix B.

    offb - Supplies the zero point offset of matrix B.

    C - Supplies the address of matrix C.

    ldc - Supplies the first dimension of matrix C.

Return Value:

    None.

--*/
{
    MLAS_DECLSPEC_ALIGN(int16_t PanelA[MLAS_GEMM_X8X8_STRIDEM * MLAS_GEMM_X8X8_STRIDEK], 64);
    MLAS_DECLSPEC_ALIGN(uint8_t PanelB[MLAS_GEMM_X8X8_STRIDEN * MLAS_GEMM_X8X8_STRIDEK], 64);

    MLAS_DECLSPEC_ALIGN(int32_t RowSumVector[MLAS_GEMM_X8X8_STRIDEM], 16);
    MLAS_DECLSPEC_ALIGN(int32_t ColumnSumVector[MLAS_GEMM_X8X8_STRIDEN], 16);

    size_t StrideM = MLAS_GEMM_X8X8_STRIDEM;
    size_t StrideN = MLAS_GEMM_X8X8_STRIDEN;
    size_t StrideK = MLAS_GEMM_X8X8_STRIDEK;

    //
    // Step through each slice of matrix B along the K dimension.
    //

    size_t CountK;

    for (size_t k = 0; k < K; k += CountK) {

        CountK = StrideK;

        if (CountK > (K - k)) {
            CountK = K - k;
        }

        //
        // Step through each slice of matrix B along the N dimension.
        //

        size_t CountN;

        for (size_t n = 0; n < N; n += CountN) {

            CountN = StrideN;

            if (CountN > (N - n)) {
                CountN = N - n;
            }

            const uint8_t* b = (const uint8_t*)B + n + k * ldb;

            MlasPlatform.GemmU8U8CopyPackBRoutine(PanelB, b, ldb, CountN,
                CountK, ColumnSumVector, -int16_t(offa));

            size_t CountM;

            for (size_t m = 0; m < M; m += CountM) {

                CountM = StrideM;

                if (CountM > (M - m)) {
                    CountM = M - m;
                }

                MlasPlatform.GemmU8U8CopyPackARoutine(PanelA, A + k + m * lda,
                    lda, CountM, CountK, RowSumVector, -int16_t(offb));

                int16_t* pa = PanelA;
                int32_t* c = C + n + m * ldc;

                int32_t* RowSums = RowSumVector;

                size_t RowsRemaining = CountM;
                size_t RowsHandled;

                size_t PairCountK = (CountK + 1) / 2;

                while (RowsRemaining > 0) {

                    RowsHandled = MlasPlatform.GemmU8U8Kernel(pa, PanelB, c,
                        PairCountK, RowsRemaining, CountN, ldc, RowSums,
                        ColumnSumVector, int32_t(CountK) * offa * offb, k == 0);

                    RowsRemaining -= RowsHandled;
                    c += ldc * RowsHandled;
                    pa += 2 * PairCountK * RowsHandled;
                    RowSums += RowsHandled;
                }
            }
        }
    }
}

void
MlasGemmX8X8Threaded(
    void* Context,
    int32_t ThreadId
    )
/*++

Routine Description:

    This routine is invoked from a worker thread to execute a segment of a
    QGEMM operation.

Arguments:

    Context - Supplies the pointer to the context for the threaded operation.

    ThreadId - Supplies the current index of the threaded operation.

Return Value:

    None.

--*/
{
    const auto* WorkBlock = (MLAS_GEMM_X8X8_WORK_BLOCK*)Context;

    const int32_t ThreadCountM = WorkBlock->ThreadCountM;
    const int32_t ThreadCountN = WorkBlock->ThreadCountN;

    const int32_t ThreadIdM = ThreadId / ThreadCountN;
    const int32_t ThreadIdN = ThreadId % ThreadCountN;

    //
    // Partition the operation along the M dimension.
    //

    size_t M = WorkBlock->M;
    size_t m;
    size_t CountM;

    MlasPartitionWork(ThreadIdM, ThreadCountM, M, &m, &CountM);

    //
    // Partition the operation along the N dimension.
    //

    size_t N = WorkBlock->N;
    size_t n;
    size_t CountN;

    const size_t BlockedN = (N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) /
        MLAS_QGEMM_STRIDEN_THREAD_ALIGN;

    MlasPartitionWork(ThreadIdN, ThreadCountN, BlockedN, &n, &CountN);

    n *= MLAS_QGEMM_STRIDEN_THREAD_ALIGN;
    CountN *= MLAS_QGEMM_STRIDEN_THREAD_ALIGN;

    if (CountN > N - n) {
        CountN = N - n;
    }

    //
    // Dispatch the partitioned operation.
    //

    const size_t lda = WorkBlock->lda;
    const size_t ldb = WorkBlock->ldb;
    const size_t ldc = WorkBlock->ldc;

    const uint8_t* a = WorkBlock->A + m * lda;
    const uint8_t* b = WorkBlock->B + n;
    int32_t* c = WorkBlock->C + n + m * ldc;

    WorkBlock->GemmX8X8Operation(CountM, CountN, WorkBlock->K, a, lda,
        WorkBlock->offa, b, ldb, WorkBlock->offb, c, ldc);
}

void
MlasGemmX8X8Schedule(
    MLAS_GEMM_X8X8_WORK_BLOCK* WorkBlock,
    MLAS_THREADPOOL* ThreadPool
    )
/*++

Routine Description:

    This module schedules the quantized integer matrix/matrix multiply
    operation (QGEMM) across one or more threads.

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
    // Compute the number of target threads given the complexity of the SGEMM
    // operation. Small requests should run using the single threaded path.
    //

    double Complexity = double(M) * double(N) * double(K);

    int32_t TargetThreadCount;

    if (Complexity < double(MLAS_QGEMM_THREAD_COMPLEXITY * MLAS_MAXIMUM_THREAD_COUNT)) {
        TargetThreadCount = int32_t(Complexity / double(MLAS_QGEMM_THREAD_COMPLEXITY)) + 1;
    } else {
        TargetThreadCount = MLAS_MAXIMUM_THREAD_COUNT;
    }

    int32_t MaximumThreadCount = MlasGetMaximumThreadCount(ThreadPool);

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

        const size_t BlockedN = (N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) /
            MLAS_QGEMM_STRIDEN_THREAD_ALIGN;

        if (size_t(TargetThreadCount) > BlockedN) {
            TargetThreadCount = int32_t(BlockedN);
        }

        WorkBlock->ThreadCountM = 1;
        WorkBlock->ThreadCountN = TargetThreadCount;

    } else {

        if (size_t(TargetThreadCount) > M) {
            TargetThreadCount = int32_t(M);
        }

        WorkBlock->ThreadCountM = TargetThreadCount;
        WorkBlock->ThreadCountN = 1;
    }

    MlasExecuteThreaded(MlasGemmX8X8Threaded, WorkBlock, TargetThreadCount, ThreadPool);
}

void
MLASCALL
MlasGemm(
    size_t M,
    size_t N,
    size_t K,
    const uint8_t* A,
    size_t lda,
    uint8_t offa,
    const int8_t* B,
    size_t ldb,
    int8_t offb,
    int32_t* C,
    size_t ldc,
    MLAS_THREADPOOL* ThreadPool
    )
/*++

Routine Description:

    This module implements the quantized integer matrix/matrix multiply
    operation (QGEMM).

Arguments:

    M - Supplies the number of rows of matrix A and matrix C.

    N - Supplies the number of columns of matrix B and matrix C.

    K - Supplies the number of columns of matrix A and the number of rows of
        matrix B.

    A - Supplies the address of matrix A.

    lda - Supplies the first dimension of matrix A.

    offa - Supplies the zero point offset of matrix A.

    B - Supplies the address of matrix B.

    ldb - Supplies the first dimension of matrix B.

    offb - Supplies the zero point offset of matrix B.

    C - Supplies the address of matrix C.

    ldc - Supplies the first dimension of matrix C.

    ThreadPool - Supplies the thread pool object to use, else nullptr if the
        base library threading support should be used.

Return Value:

    None.

--*/
{
    MLAS_GEMM_X8X8_WORK_BLOCK WorkBlock;

    //
    // Capture the GEMM parameters to the work block.
    //

    WorkBlock.M = M;
    WorkBlock.N = N;
    WorkBlock.K = K;
    WorkBlock.A = A;
    WorkBlock.lda = lda;
    WorkBlock.B = (const uint8_t*)B;
    WorkBlock.ldb = ldb;
    WorkBlock.C = C;
    WorkBlock.ldc = ldc;
    WorkBlock.offa = int16_t(offa);
    WorkBlock.offb = int16_t(offb);
    WorkBlock.GemmX8X8Operation = MlasGemmU8S8Operation;

    //
    // Schedule the operation across a set of worker threads.
    //

    MlasGemmX8X8Schedule(&WorkBlock, ThreadPool);
}

void
MLASCALL
MlasGemm(
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
/*++

Routine Description:

    This module implements the quantized integer matrix/matrix multiply
    operation (QGEMM).

Arguments:

    M - Supplies the number of rows of matrix A and matrix C.

    N - Supplies the number of columns of matrix B and matrix C.

    K - Supplies the number of columns of matrix A and the number of rows of
        matrix B.

    A - Supplies the address of matrix A.

    lda - Supplies the first dimension of matrix A.

    offa - Supplies the zero point offset of matrix A.

    B - Supplies the address of matrix B.

    ldb - Supplies the first dimension of matrix B.

    offb - Supplies the zero point offset of matrix B.

    C - Supplies the address of matrix C.

    ldc - Supplies the first dimension of matrix C.

    ThreadPool - Supplies the thread pool object to use, else nullptr if the
        base library threading support should be used.

Return Value:

    None.

--*/
{
    MLAS_GEMM_X8X8_WORK_BLOCK WorkBlock;

    //
    // Capture the GEMM parameters to the work block.
    //

    WorkBlock.M = M;
    WorkBlock.N = N;
    WorkBlock.K = K;
    WorkBlock.A = A;
    WorkBlock.lda = lda;
    WorkBlock.B = B;
    WorkBlock.ldb = ldb;
    WorkBlock.C = C;
    WorkBlock.ldc = ldc;
    WorkBlock.offa = int16_t(offa);
    WorkBlock.offb = int16_t(offb);
    WorkBlock.GemmX8X8Operation = MlasGemmU8U8Operation;

    //
    // Schedule the operation across a set of worker threads.
    //

    MlasGemmX8X8Schedule(&WorkBlock, ThreadPool);
}

#endif
