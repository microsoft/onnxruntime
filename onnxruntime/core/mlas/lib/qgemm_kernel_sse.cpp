/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm_kernel_sse.cpp

Abstract:

    This module implements QGEMM kernels for sse.

--*/

#include "mlasi.h"
#include "qgemm_kernel_protocol.h"
#include "qgemm_kernel_type.h"

#if defined(MLAS_SSE2_INTRINSICS)

template<>
MLAS_FORCEINLINE
int32_t
MlasGemmU8X8FixupZeroPointB<MLAS_GEMM_U8X8_KERNEL_SSE>(
    int32_t ZeroPointB,
    bool BIsSigned
    )
{
    if (!BIsSigned) {
        ZeroPointB = MLAS_GEMM_U8X8_KERNEL_SSE::OffsetBType(ZeroPointB ^ 0x80);
    }

    return ZeroPointB;
}

template<>
void
MlasGemmU8X8CopyPackA<MLAS_GEMM_U8X8_KERNEL_SSE>(
    MLAS_GEMM_U8X8_KERNEL_SSE::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer
    )
{
    const __m128i ZeroVector = _mm_setzero_si128();
    const __m128i OnesWordBroadcast = _mm_set1_epi16(1);
    uint8_t PaddedMatrixAData[8] = { 0 };

    //
    // Process a single row of matrix A in a loop.
    //

    while (CountM > 0) {

        const uint8_t* a = A;
        size_t k = CountK;
        __m128i ReductionVector = ZeroVector;

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

            __m128i Bytes = _mm_loadl_epi64((const __m128i*) & a[0]);
            __m128i Words = _mm_unpacklo_epi8(Bytes, ZeroVector);

            ReductionVector = _mm_add_epi16(ReductionVector, Words);

            _mm_storeu_si128((__m128i*) & D[0], Words);

            a += 8;
            D += 8;
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

            ReductionVector = _mm_add_epi16(ReductionVector, Words);

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
        // Reduce the partial accumulators.
        //

        ReductionVector = _mm_madd_epi16(ReductionVector, OnesWordBroadcast);
        ReductionVector = _mm_add_epi32(ReductionVector,
                                        _mm_shuffle_epi32(ReductionVector, _MM_SHUFFLE(3, 2, 3, 2)));
        ReductionVector = _mm_add_epi32(ReductionVector,
                                        _mm_shuffle_epi32(ReductionVector, _MM_SHUFFLE(0, 1, 0, 1)));

        *RowSumBuffer++ = _mm_cvtsi128_si32(ReductionVector);

        A += lda;
        CountM -= 1;
    }
}


MLAS_FORCEINLINE
void
MlasTranspose8x8Block(
    const uint8_t* Input,
    size_t InputStride,
    typename MLAS_GEMM_U8X8_KERNEL_SSE::PackedAType* Output,
    size_t OutputStride,
    __m128i& RowSums
)
{
    const __m128i ZeroVector = _mm_setzero_si128();
    __m128i a0 = _mm_loadl_epi64((const __m128i*) & Input[InputStride * 0]);
    RowSums = _mm_add_epi16(RowSums, _mm_unpacklo_epi8(a0, ZeroVector));

    __m128i a1 = _mm_loadl_epi64((const __m128i*) & Input[InputStride * 1]);
    RowSums = _mm_add_epi16(RowSums, _mm_unpacklo_epi8(a1, ZeroVector));

    __m128i a2 = _mm_loadl_epi64((const __m128i*) & Input[InputStride * 2]);
    RowSums = _mm_add_epi16(RowSums, _mm_unpacklo_epi8(a2, ZeroVector));

    __m128i a3 = _mm_loadl_epi64((const __m128i*) & Input[InputStride * 3]);
    RowSums = _mm_add_epi16(RowSums, _mm_unpacklo_epi8(a3, ZeroVector));

    __m128i a4 = _mm_loadl_epi64((const __m128i*) & Input[InputStride * 4]);
    RowSums = _mm_add_epi16(RowSums, _mm_unpacklo_epi8(a4, ZeroVector));

    __m128i a5 = _mm_loadl_epi64((const __m128i*) & Input[InputStride * 5]);
    RowSums = _mm_add_epi16(RowSums, _mm_unpacklo_epi8(a5, ZeroVector));

    __m128i a6 = _mm_loadl_epi64((const __m128i*) & Input[InputStride * 6]);
    RowSums = _mm_add_epi16(RowSums, _mm_unpacklo_epi8(a6, ZeroVector));

    __m128i a7 = _mm_loadl_epi64((const __m128i*) & Input[InputStride * 7]);
    RowSums = _mm_add_epi16(RowSums, _mm_unpacklo_epi8(a7, ZeroVector));

    __m128i b0 = _mm_unpacklo_epi8(a0, a1);
    __m128i b1 = _mm_unpacklo_epi8(a2, a3);
    __m128i b2 = _mm_unpacklo_epi8(a4, a5);
    __m128i b3 = _mm_unpacklo_epi8(a6, a7);

    __m128i c0 = _mm_unpacklo_epi16(b0, b1);
    __m128i c1 = _mm_unpackhi_epi16(b0, b1);
    __m128i c2 = _mm_unpacklo_epi16(b2, b3);
    __m128i c3 = _mm_unpackhi_epi16(b2, b3);

    __m128i d0 = _mm_unpacklo_epi32(c0, c2);
    __m128i d1 = _mm_unpackhi_epi32(c0, c2);
    __m128i d2 = _mm_unpacklo_epi32(c1, c3);
    __m128i d3 = _mm_unpackhi_epi32(c1, c3);

    __m128i col0 = _mm_srai_epi16(_mm_unpacklo_epi8(d0, d0), 8);
    __m128i col1 = _mm_srai_epi16(_mm_unpackhi_epi8(d0, d0), 8);
    __m128i col2 = _mm_srai_epi16(_mm_unpacklo_epi8(d1, d1), 8);
    __m128i col3 = _mm_srai_epi16(_mm_unpackhi_epi8(d1, d1), 8);
    __m128i col4 = _mm_srai_epi16(_mm_unpacklo_epi8(d2, d2), 8);
    __m128i col5 = _mm_srai_epi16(_mm_unpackhi_epi8(d2, d2), 8);
    __m128i col6 = _mm_srai_epi16(_mm_unpacklo_epi8(d3, d3), 8);
    __m128i col7 = _mm_srai_epi16(_mm_unpackhi_epi8(d3, d3), 8);

    _mm_storeu_si128((__m128i*) & Output[OutputStride * 0], col0);
    _mm_storeu_si128((__m128i*) & Output[OutputStride * 1], col1);
    _mm_storeu_si128((__m128i*) & Output[OutputStride * 2], col2);
    _mm_storeu_si128((__m128i*) & Output[OutputStride * 3], col3);
    _mm_storeu_si128((__m128i*) & Output[OutputStride * 4], col4);
    _mm_storeu_si128((__m128i*) & Output[OutputStride * 5], col5);
    _mm_storeu_si128((__m128i*) & Output[OutputStride * 6], col6);
    _mm_storeu_si128((__m128i*) & Output[OutputStride * 7], col7);
}

template<>
void
MlasGemmU8X8TransposePackA<MLAS_GEMM_U8X8_KERNEL_SSE>(
    typename MLAS_GEMM_U8X8_KERNEL_SSE::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer
    )
{
    const __m128i ZeroVector = _mm_setzero_si128();
    const __m128i OnesWordBroadcast = _mm_set1_epi16(1);
    uint8_t PaddedMatrixAData[8] = { 0 };

    const size_t AlignedCountK =
        (CountK + MLAS_GEMM_U8X8_KERNEL_SSE::PackedK - 1) & ~(MLAS_GEMM_U8X8_KERNEL_SSE::PackedK - 1);

    //
    // Process 8 row of matrix A in a loop.
    //

    while (CountK >= 8) {

        const uint8_t* a = A;
        int32_t* sum = RowSumBuffer;
        size_t m = CountM;
        __m128i ReductionVector = ZeroVector;

        typename MLAS_GEMM_U8X8_KERNEL_SSE::PackedAType* d = D;
        while (m >= 8) {
            __m128i RowSum8x8 = ZeroVector;
            MlasTranspose8x8Block(a, lda, d, AlignedCountK, RowSum8x8);

            __m128i RowSum = _mm_loadu_epi32(sum);
            _mm_storeu_si128((__m128i*)sum, _mm_add_epi32(RowSum, _mm_unpacklo_epi8(RowSum8x8, ZeroVector)));

            RowSum = _mm_loadu_epi32(sum + 4);
            _mm_storeu_si128((__m128i*)(sum + 4), _mm_add_epi32(RowSum, _mm_unpackhi_epi8(RowSum8x8, ZeroVector)));

            a += 8;
            d += 8 * AlignedCountK;
            m -= 8;
            sum += 8;
        }

        while (m > 0) {
            for (size_t kk = 0; kk < 8; kk++) {
                d[kk] = a[lda * kk];
                *sum += d[kk];
            }

            a ++;
            d += AlignedCountK;
            m --;
            sum ++;
        }

        D += 8;
        A += 8 * lda;
    }

    for (int k = 0; k < CountK; k++) {
        for (int m = 0; m < CountM; m++) {
            D[m * AlignedCountK] = A[m];
            RowSumBuffer[m] += A[m];
        }
    }
}

MLAS_FORCEINLINE
void
MlasGemmU8X8CopyPackBProcessSse(
    MLAS_GEMM_U8X8_KERNEL_SSE::PackedBType* D,
    __m128i BytesRow0,
    __m128i BytesRow1,
    __m128i BitFlipVector,
    __m128i ColumnSums[2]
)
{
    __m128i BytesInterleaved = _mm_unpacklo_epi8(BytesRow0, BytesRow1);

    BytesInterleaved = _mm_xor_si128(BytesInterleaved, BitFlipVector);

    __m128i WordsInterleaved0 = _mm_srai_epi16(_mm_unpacklo_epi8(BytesInterleaved, BytesInterleaved), 8);
    __m128i WordsInterleaved1 = _mm_srai_epi16(_mm_unpackhi_epi8(BytesInterleaved, BytesInterleaved), 8);

    ColumnSums[0] = _mm_add_epi16(ColumnSums[0], WordsInterleaved0);
    ColumnSums[1] = _mm_add_epi16(ColumnSums[1], WordsInterleaved1);

    _mm_storeu_si128((__m128i*) & D[0], WordsInterleaved0);
    _mm_storeu_si128((__m128i*) & D[8], WordsInterleaved1);
}

template<>
void
MlasGemmU8X8CopyPackB<MLAS_GEMM_U8X8_KERNEL_SSE>(
    MLAS_GEMM_U8X8_KERNEL_SSE::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    const __m128i OnesWordBroadcast = _mm_set1_epi16(1);
    const __m128i BitFlipVector = _mm_set1_epi32(BIsSigned ? 0 : 0x80808080);

    //
    // Process 8 columns of matrix B in a loop.
    //

    while (CountN >= 8) {

        const uint8_t* b = B;
        size_t k = CountK;
        __m128i ColumnSums[2];

        ColumnSums[0] = _mm_setzero_si128();
        ColumnSums[1] = _mm_setzero_si128();

        //
        // Interleave rows of matrix B and write to the packed buffer.
        //
        // These values are also zero-extended and accumulated into an
        // intermediate per-column accumulator. CountK cannot be greater than
        // 128 to avoid overflowing these signed 16-bit accumulators.
        //

        while (k >= MLAS_GEMM_U8X8_KERNEL_SSE::PackedK) {

            __m128i BytesRow0 = _mm_loadl_epi64((const __m128i*) & b[0]);
            __m128i BytesRow1 = _mm_loadl_epi64((const __m128i*) & b[ldb]);

            MlasGemmU8X8CopyPackBProcessSse(D, BytesRow0, BytesRow1, BitFlipVector, ColumnSums);

            b += ldb * 2;
            D += 16;
            k -= 2;
        }

        if (k > 0) {

            __m128i BytesRow0 = _mm_loadl_epi64((const __m128i*) & b[0]);

            MlasGemmU8X8CopyPackBProcessSse(D, BytesRow0, BitFlipVector, BitFlipVector, ColumnSums);

            D += 16;
        }

        ColumnSums[0] = _mm_madd_epi16(ColumnSums[0], OnesWordBroadcast);
        ColumnSums[1] = _mm_madd_epi16(ColumnSums[1], OnesWordBroadcast);

        _mm_storeu_si128((__m128i*) & ColumnSumBuffer[0], ColumnSums[0]);
        _mm_storeu_si128((__m128i*) & ColumnSumBuffer[4], ColumnSums[1]);
        ColumnSumBuffer += 8;

        B += 8;
        CountN -= 8;
    }

    //
    // Process the remaining columns of matrix B.
    //

    if (CountN > 0) {

        const uint8_t* b = B;
        size_t k = CountK;
        __m128i ColumnSums[2];
        uint8_t PaddedMatrixBData[16];

        _mm_storeu_si128((__m128i*)PaddedMatrixBData, BitFlipVector);

        ColumnSums[0] = _mm_setzero_si128();
        ColumnSums[1] = _mm_setzero_si128();

        //
        // Interleave rows of matrix B using an intermediate zero padded stack
        // buffer and write to the packed buffer.
        //

        while (k >= MLAS_GEMM_U8X8_KERNEL_SSE::PackedK) {

            const uint8_t* bcopy = b;
            uint8_t* padded = PaddedMatrixBData;
            uint8_t* padded_end = padded + CountN;

            do {
                padded[0] = bcopy[0];
                padded[8] = bcopy[ldb];
                padded++;
                bcopy++;
            } while (padded < padded_end);

            __m128i BytesRow0 = _mm_loadl_epi64((__m128i*) & PaddedMatrixBData[0]);
            __m128i BytesRow1 = _mm_loadl_epi64((__m128i*) & PaddedMatrixBData[8]);

            MlasGemmU8X8CopyPackBProcessSse(D, BytesRow0, BytesRow1, BitFlipVector, ColumnSums);

            b += ldb * 2;
            D += 16;
            k -= 2;
        }

        if (k > 0) {

            const uint8_t* bcopy = b;
            uint8_t* padded = PaddedMatrixBData;
            uint8_t* padded_end = padded + CountN;

            do {
                padded[0] = bcopy[0];
                padded++;
                bcopy++;
            } while (padded < padded_end);

            __m128i BytesRow0 = _mm_loadl_epi64((__m128i*) & PaddedMatrixBData[0]);

            MlasGemmU8X8CopyPackBProcessSse(D, BytesRow0, BitFlipVector, BitFlipVector, ColumnSums);
        }

        ColumnSums[0] = _mm_madd_epi16(ColumnSums[0], OnesWordBroadcast);
        ColumnSums[1] = _mm_madd_epi16(ColumnSums[1], OnesWordBroadcast);

        _mm_storeu_si128((__m128i*) & ColumnSumBuffer[0], ColumnSums[0]);
        _mm_storeu_si128((__m128i*) & ColumnSumBuffer[4], ColumnSums[1]);
    }
}

MLAS_FORCEINLINE
void
MlasGemmU8X8TransposePackB8x8Sse(
    const uint8_t* Input,
    size_t InputStride,
    typename MLAS_GEMM_U8X8_KERNEL_SSE::PackedAType* Output,
    __m128i BitFlipVector,
    __m128i ColSums[2]
)
{
    const __m128i ZeroVector = _mm_setzero_si128();
    __m128i a0 = _mm_loadl_epi64((const __m128i*) & Input[InputStride * 0]);
    __m128i a1 = _mm_loadl_epi64((const __m128i*) & Input[InputStride * 1]);
    __m128i a2 = _mm_loadl_epi64((const __m128i*) & Input[InputStride * 2]);
    __m128i a3 = _mm_loadl_epi64((const __m128i*) & Input[InputStride * 3]);
    __m128i a4 = _mm_loadl_epi64((const __m128i*) & Input[InputStride * 4]);
    __m128i a5 = _mm_loadl_epi64((const __m128i*) & Input[InputStride * 5]);
    __m128i a6 = _mm_loadl_epi64((const __m128i*) & Input[InputStride * 6]);
    __m128i a7 = _mm_loadl_epi64((const __m128i*) & Input[InputStride * 7]);

    __m128i b0 = _mm_xor_si128(_mm_unpacklo_epi16(a0, a1), BitFlipVector);
    __m128i b1 = _mm_xor_si128(_mm_unpacklo_epi16(a2, a3), BitFlipVector);
    __m128i b2 = _mm_xor_si128(_mm_unpacklo_epi16(a4, a5), BitFlipVector);
    __m128i b3 = _mm_xor_si128(_mm_unpacklo_epi16(a6, a7), BitFlipVector);

    __m128i c0 = _mm_unpacklo_epi32(b0, b1);
    __m128i c1 = _mm_unpackhi_epi32(b0, b1);
    __m128i c2 = _mm_unpacklo_epi32(b2, b3);
    __m128i c3 = _mm_unpackhi_epi32(b2, b3);

    __m128i d0 = _mm_unpacklo_epi64(c0, c2);
    __m128i d1 = _mm_unpackhi_epi64(c0, c2);
    __m128i d2 = _mm_unpacklo_epi64(c1, c3);
    __m128i d3 = _mm_unpackhi_epi64(c1, c3);


    __m128i w0 = _mm_srai_epi16(_mm_unpacklo_epi8(d0, d0), 8);
    __m128i w1 = _mm_srai_epi16(_mm_unpackhi_epi8(d0, d0), 8);
    __m128i w2 = _mm_srai_epi16(_mm_unpacklo_epi8(d1, d1), 8);
    __m128i w3 = _mm_srai_epi16(_mm_unpackhi_epi8(d1, d1), 8);
    __m128i w4 = _mm_srai_epi16(_mm_unpacklo_epi8(d2, d2), 8);
    __m128i w5 = _mm_srai_epi16(_mm_unpackhi_epi8(d2, d2), 8);
    __m128i w6 = _mm_srai_epi16(_mm_unpacklo_epi8(d3, d3), 8);
    __m128i w7 = _mm_srai_epi16(_mm_unpackhi_epi8(d3, d3), 8);

    const __m128i OnesWordBroadcast = _mm_set1_epi16(1);
    __m128i s0 = _mm_add_epi16(w0, w2);
    __m128i s2 = _mm_add_epi16(w4, w6);
    __m128i s1 = _mm_add_epi16(w1, w3);
    __m128i s3 = _mm_add_epi16(w5, w7);
    s0 = _mm_madd_epi16(_mm_add_epi16(s0, s2), OnesWordBroadcast);
    s1 = _mm_madd_epi16(_mm_add_epi16(s1, s3), OnesWordBroadcast);
    ColSums[0] = _mm_add_epi32(s0, ColSums[0]);
    ColSums[1] = _mm_add_epi32(s1, ColSums[1]);

    _mm_storeu_si128((__m128i*) & Output[8 * 0], w0);
    _mm_storeu_si128((__m128i*) & Output[8 * 1], w1);
    _mm_storeu_si128((__m128i*) & Output[8 * 2], w2);
    _mm_storeu_si128((__m128i*) & Output[8 * 3], w3);
    _mm_storeu_si128((__m128i*) & Output[8 * 4], w4);
    _mm_storeu_si128((__m128i*) & Output[8 * 5], w5);
    _mm_storeu_si128((__m128i*) & Output[8 * 6], w6);
    _mm_storeu_si128((__m128i*) & Output[8 * 7], w7);
}

template<>
void
MlasGemmU8X8TransposePackB<MLAS_GEMM_U8X8_KERNEL_SSE>(
    MLAS_GEMM_U8X8_KERNEL_SSE::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    size_t PackedK = MLAS_GEMM_U8X8_KERNEL_SSE::PackedK;
    const __m128i OnesWordBroadcast = _mm_set1_epi16(1);
    const __m128i BitFlipVector = _mm_set1_epi32(BIsSigned ? 0 : 0x80808080);

    //
    // Process 8 columns of matrix B in a loop.
    //

    while (CountN >= 8) {

        const uint8_t* b = B;
        size_t k = CountK;
        __m128i ColumnSums[2];

        ColumnSums[0] = _mm_setzero_si128();
        ColumnSums[1] = _mm_setzero_si128();

        while (k >= 8) {
            MlasGemmU8X8TransposePackB8x8Sse(b, ldb, D, BitFlipVector, ColumnSums);

            k -= 8;
            D += 64;
            b += 8;
        }

        for(size_t CurK = 0; CurK < k; CurK += PackedK) {
            int32_t sum[8] = {};
            for (int n = 0; n < 8; n++) {
                for (size_t kk = 0; kk < PackedK && CurK + kk < k; kk++) {
                    D[kk] = b[n*ldb];
                    sum[n] += b[n*ldb];
                }
                D += MLAS_GEMM_U8X8_KERNEL_SSE::PackedK;
            }

            b += MLAS_GEMM_U8X8_KERNEL_SSE::PackedK;
            __m128i sum0 = _mm_loadu_si128((const __m128i*)sum);
            __m128i sum1 = _mm_loadu_si128((const __m128i*)(sum + 4));
            _mm_add_epi16(ColumnSums[0], sum0);
            _mm_add_epi16(ColumnSums[1], sum1);
        }

        _mm_storeu_si128((__m128i*) & ColumnSumBuffer[0], ColumnSums[0]);
        _mm_storeu_si128((__m128i*) & ColumnSumBuffer[4], ColumnSums[1]);
        ColumnSumBuffer += 8;

        B += 8 * ldb;
        CountN -= 8;
    }

    //
    // Process the remaining columns of matrix B.
    //

    if (CountN > 0) {
        __m128i ColumnSums[2];
        ColumnSums[0] = _mm_setzero_si128();
        ColumnSums[1] = _mm_setzero_si128();

        for (size_t CurK = 0; CurK < CountK; CurK += PackedK) {
            int32_t sum[8] = {};
            for (int n = 0; n < CountN; n++) {
                for (size_t kk = 0; kk < PackedK && CurK + kk < k; kk++) {
                    D[kk] = B[n * ldb];
                    sum[n] += B[n * ldb];
                }
                D += MLAS_GEMM_U8X8_KERNEL_SSE::PackedK;
            }

            D += PackedK * (8 - CountN);
            B += PackedK;
            __m128i sum0 = _mm_loadu_si128((const __m128i*)sum);
            __m128i sum1 = _mm_loadu_si128((const __m128i*)(sum + 4));
            _mm_add_epi16(ColumnSums[0], sum0);
            _mm_add_epi16(ColumnSums[1], sum1);
        }

        _mm_storeu_si128((__m128i*) & ColumnSumBuffer[0], ColumnSums[0]);
        _mm_storeu_si128((__m128i*) & ColumnSumBuffer[4], ColumnSums[1]);
        ColumnSumBuffer += 8;
    }
}

MLAS_FORCEINLINE
void
MlasGemmU8X8MultiplyAccumulateRowSse(
    __m128i ABroadcast,
    const int16_t* B,
    __m128i Accumulators[2]
)
{
    __m128i BElements0 = _mm_load_si128((__m128i*) & B[0]);
    __m128i BElements1 = _mm_load_si128((__m128i*) & B[8]);

    Accumulators[0] = _mm_add_epi32(Accumulators[0], _mm_madd_epi16(BElements0, ABroadcast));
    Accumulators[1] = _mm_add_epi32(Accumulators[1], _mm_madd_epi16(BElements1, ABroadcast));
}

template<>
size_t
MlasGemmU8X8Kernel<MLAS_GEMM_U8X8_KERNEL_SSE>(
    const MLAS_GEMM_U8X8_KERNEL_SSE::PackedAType* A,
    const MLAS_GEMM_U8X8_KERNEL_SSE::PackedBType* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode
    )
{
    MLAS_UNREFERENCED_PARAMETER(CountM);
    MLAS_UNREFERENCED_PARAMETER(ldc);

    while (CountN > 0) {

        __m128i Accumulators[2];

        //
        // Initialize the accumulators with the row and column sums.
        //

        int32_t RowSumValue = RowSumBuffer[0];

        if (ZeroPointB != nullptr) {

            int32_t ScaledRowSumBuffer[8];

            for (size_t i = 0; i < 8; i++) {
                ScaledRowSumBuffer[i] = RowSumValue * ZeroPointB[i];
            }

            ZeroPointB += 8;

            Accumulators[0] = _mm_loadu_si128((__m128i*) & ScaledRowSumBuffer[0]);
            Accumulators[1] = _mm_loadu_si128((__m128i*) & ScaledRowSumBuffer[4]);

        }
        else {

            Accumulators[0] = _mm_set1_epi32(RowSumValue);
            Accumulators[1] = Accumulators[0];
        }

        Accumulators[0] = _mm_add_epi32(Accumulators[0], _mm_loadu_si128((const __m128i*) & ColumnSumBuffer[0]));
        Accumulators[1] = _mm_add_epi32(Accumulators[1], _mm_loadu_si128((const __m128i*) & ColumnSumBuffer[4]));
        ColumnSumBuffer += 8;

        //
        // Broadcast each pair of 16-bit values from the matrix A and multiply
        // with the pair of 16-bit values from matrix B, and add the 32-bit
        // intermediate into the accumulator registers.
        //

        const int16_t* a = A;
        size_t k = PackedCountK;

        while (k >= 4) {

            __m128i AElements = _mm_loadu_si128((__m128i*)a);
            __m128i ABroadcast;

            ABroadcast = _mm_shuffle_epi32(AElements, _MM_SHUFFLE(0, 0, 0, 0));
            MlasGemmU8X8MultiplyAccumulateRowSse(ABroadcast, &B[0], Accumulators);

            ABroadcast = _mm_shuffle_epi32(AElements, _MM_SHUFFLE(1, 1, 1, 1));
            MlasGemmU8X8MultiplyAccumulateRowSse(ABroadcast, &B[16], Accumulators);

            ABroadcast = _mm_shuffle_epi32(AElements, _MM_SHUFFLE(2, 2, 2, 2));
            MlasGemmU8X8MultiplyAccumulateRowSse(ABroadcast, &B[32], Accumulators);

            ABroadcast = _mm_shuffle_epi32(AElements, _MM_SHUFFLE(3, 3, 3, 3));
            MlasGemmU8X8MultiplyAccumulateRowSse(ABroadcast, &B[48], Accumulators);

            a += 4 * 2;
            B += 4 * 16;
            k -= 4;
        }

        while (k > 0) {

            __m128i ABroadcast = _mm_set1_epi32(*((int32_t*)a));
            MlasGemmU8X8MultiplyAccumulateRowSse(ABroadcast, &B[0], Accumulators);

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
                Accumulators[0] = _mm_add_epi32(Accumulators[0], _mm_loadu_si128((__m128i*) & C[0]));
                Accumulators[1] = _mm_add_epi32(Accumulators[1], _mm_loadu_si128((__m128i*) & C[4]));
            }

            _mm_storeu_si128((__m128i*) & C[0], Accumulators[0]);
            _mm_storeu_si128((__m128i*) & C[4], Accumulators[1]);

            C += 8;
            CountN -= 8;

        }
        else {

            //
            // Output the remaining partial output block.
            //

            if ((CountN & 4) != 0) {

                if (!ZeroMode) {
                    Accumulators[0] = _mm_add_epi32(Accumulators[0], _mm_loadu_si128((__m128i*) & C[0]));
                }

                _mm_storeu_si128((__m128i*) & C[0], Accumulators[0]);
                C += 4;

                Accumulators[0] = Accumulators[1];
            }

            if ((CountN & 2) != 0) {

                if (!ZeroMode) {
                    Accumulators[0] = _mm_add_epi32(Accumulators[0], _mm_loadl_epi64((__m128i*) & C[0]));
                }

                _mm_storel_epi64((__m128i*) & C[0], Accumulators[0]);
                C += 2;

                Accumulators[0] = _mm_shuffle_epi32(Accumulators[0], _MM_SHUFFLE(3, 2, 3, 2));
            }

            if ((CountN & 1) != 0) {

                int32_t AccumulatorValue = _mm_cvtsi128_si32(Accumulators[0]);

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

const MLAS_GEMM_U8X8_DISPATCH MlasGemmU8X8DispatchSse = {
    MlasGemmU8X8Operation<MLAS_GEMM_U8X8_KERNEL_SSE>,
    nullptr,
    nullptr,
    MLAS_GEMM_U8X8_KERNEL_SSE::PackedK,
    0,
};

#endif
