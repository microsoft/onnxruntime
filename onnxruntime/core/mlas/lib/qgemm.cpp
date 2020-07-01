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
// Define the parameters to execute segments of a QGEMM operation on worker
// threads.
//

struct MLAS_GEMM_U8X8_WORK_BLOCK {
    int32_t ThreadCountM;
    int32_t ThreadCountN;
    size_t M;
    size_t N;
    size_t K;
    const uint8_t* A;
    size_t lda;
    const uint8_t* B;
    size_t ldb;
    int32_t* C;
    size_t ldc;
    const float* Scale;
    const float* BiasFloat;
    uint8_t offa;
    uint8_t offb;
    bool BTypeIsSigned;
    bool CTypeIsFloat;
};

template<typename KernelType>
MLAS_FORCEINLINE
void
MlasGemmU8X8Operation(
    const MLAS_GEMM_U8X8_WORK_BLOCK* WorkBlock
    )
/*++

Routine Description:

    This module implements the quantized integer matrix/matrix multiply
    operation (QGEMM).

Arguments:

    WorkBlock - Supplies the structure containing the GEMM parameters.

Return Value:

    None.

--*/
{
    MLAS_DECLSPEC_ALIGN(typename KernelType::PackedAType PanelA[KernelType::StrideM * KernelType::StrideK], 64);
    MLAS_DECLSPEC_ALIGN(typename KernelType::PackedBType PanelB[KernelType::StrideN * KernelType::StrideK], 64);

    MLAS_DECLSPEC_ALIGN(int32_t RowSumVector[KernelType::StrideM], 64);
    MLAS_DECLSPEC_ALIGN(int32_t ColumnSumVector[KernelType::StrideN], 64);

    const uint8_t* A = WorkBlock->A;
    const uint8_t* B = WorkBlock->B;
    int32_t* C = WorkBlock->C;

    const size_t lda = WorkBlock->lda;
    const size_t ldb = WorkBlock->ldb;
    const size_t ldc = WorkBlock->ldc;

    //
    // Flip the sign bit of the zero point offset of matrix B if the kernel uses
    // signed types and the matrix B data is unsigned.
    //

    int16_t offa = WorkBlock->offa;
    int16_t offb = typename KernelType::OffsetBType(WorkBlock->offb);

    if (std::is_signed<typename KernelType::OffsetBType>::value && !WorkBlock->BTypeIsSigned) {
        offb = typename KernelType::OffsetBType(offb ^ 0x80);
    }

    //
    // Step through each slice of matrix B along the K dimension.
    //

    const size_t M = WorkBlock->M;
    const size_t N = WorkBlock->N;
    const size_t K = WorkBlock->K;
    size_t CountK;

    for (size_t k = 0; k < K; k += CountK) {

        CountK = (std::min)(K - k, KernelType::StrideK);

        //
        // Step through each slice of matrix B along the N dimension.
        //

        size_t CountN;

        for (size_t n = 0; n < N; n += CountN) {

            CountN = (std::min)(N - n, KernelType::StrideN);

            //
            // Copy a panel of matrix B to a local packed buffer.
            //

            KernelType::CopyPackB(PanelB, B + n, ldb, CountN, CountK,
                ColumnSumVector, -offa, WorkBlock->BTypeIsSigned);

            //
            // Step through each slice of matrix A along the M dimension.
            //

            const int32_t DepthValue = int32_t(CountK) * offa * offb;
            const size_t PackedCountK = (CountK + KernelType::PackedK - 1) /
                KernelType::PackedK;

            int32_t* c = C + n;
            size_t CountM;

            for (size_t m = 0; m < M; m += CountM) {

                CountM = (std::min)(M - m, KernelType::StrideM);

                //
                // Copy a panel of matrix A to a local packed buffer.
                //

                KernelType::CopyPackA(PanelA, A + m * lda, lda, CountM, CountK,
                    RowSumVector, -offb);

                //
                // Step through the rows of the local packed buffer.
                //

                typename KernelType::PackedAType* pa = PanelA;
                int32_t* RowSums = RowSumVector;
                size_t RowsRemaining = CountM;

                bool ZeroMode = (k == 0);
                bool PostProcess = (k + CountK == K);

                while (RowsRemaining > 0) {

                    size_t RowsHandled;

                    RowsHandled = KernelType::Kernel(pa, PanelB, c, PackedCountK,
                        RowsRemaining, CountN, ldc, RowSums, ColumnSumVector,
                        DepthValue, ZeroMode);

                    if (PostProcess && WorkBlock->CTypeIsFloat) {
                        KernelType::OutputFloat(WorkBlock, c, n, RowsHandled, CountN);
                    }

                    c += ldc * RowsHandled;
                    pa += KernelType::PackedK * PackedCountK * RowsHandled;
                    RowSums += RowsHandled;
                    RowsRemaining -= RowsHandled;
                }
            }
        }

        A += CountK;
        B += CountK * ldb;
    }
}

#ifdef MLAS_TARGET_AMD64_IX86

void
MlasGemmU8X8CopyPackASse(
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
MlasGemmU8X8CopyPackBProcessSse(
    int16_t* D,
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

    _mm_storeu_si128((__m128i*)&D[0], WordsInterleaved0);
    _mm_storeu_si128((__m128i*)&D[8], WordsInterleaved1);
}

void
MlasGemmU8X8CopyPackBSse(
    int16_t* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumVector,
    int16_t offa,
    bool BTypeIsSigned
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
    const __m128i OffsetBroadcast = _mm_set1_epi16(offa);
    const __m128i BitFlipVector = _mm_set1_epi32(BTypeIsSigned ? 0 : 0x80808080);

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

        while (k >= 2) {

            __m128i BytesRow0 = _mm_loadl_epi64((__m128i*)&b[0]);
            __m128i BytesRow1 = _mm_loadl_epi64((__m128i*)&b[ldb]);

            MlasGemmU8X8CopyPackBProcessSse(D, BytesRow0, BytesRow1, BitFlipVector, ColumnSums);

            b += ldb * 2;
            D += 16;
            k -= 2;
        }

        if (k > 0) {

            __m128i BytesRow0 = _mm_loadl_epi64((__m128i*)&b[0]);

            MlasGemmU8X8CopyPackBProcessSse(D, BytesRow0, BitFlipVector, BitFlipVector, ColumnSums);

            D += 16;
        }

        //
        // Reduce the sum for the packed columns and multiply by the zero point
        // offset of the other source matrix.
        //

        ColumnSums[0] = _mm_madd_epi16(ColumnSums[0], OffsetBroadcast);
        ColumnSums[1] = _mm_madd_epi16(ColumnSums[1], OffsetBroadcast);

        _mm_storeu_si128((__m128i*)&ColumnSumVector[0], ColumnSums[0]);
        _mm_storeu_si128((__m128i*)&ColumnSumVector[4], ColumnSums[1]);

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
        __m128i ColumnSums[2];
        uint8_t PaddedMatrixBData[16];

        _mm_storeu_si128((__m128i*)PaddedMatrixBData, BitFlipVector);

        ColumnSums[0] = _mm_setzero_si128();
        ColumnSums[1] = _mm_setzero_si128();

        //
        // Interleave rows of matrix B using an intermediate zero padded stack
        // buffer and write to the packed buffer.
        //

        while (k >= 2) {

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

            __m128i BytesRow0 = _mm_loadl_epi64((__m128i*)&PaddedMatrixBData[0]);

            MlasGemmU8X8CopyPackBProcessSse(D, BytesRow0, BitFlipVector, BitFlipVector, ColumnSums);
        }

        //
        // Reduce the sum for the packed columns and multiply by the zero point
        // offset of the other source matrix.
        //

        ColumnSums[0] = _mm_madd_epi16(ColumnSums[0], OffsetBroadcast);
        ColumnSums[1] = _mm_madd_epi16(ColumnSums[1], OffsetBroadcast);

        _mm_storeu_si128((__m128i*)&ColumnSumVector[0], ColumnSums[0]);
        _mm_storeu_si128((__m128i*)&ColumnSumVector[4], ColumnSums[1]);
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
    __m128i BElements0 = _mm_load_si128((__m128i*)&B[0]);
    __m128i BElements1 = _mm_load_si128((__m128i*)&B[8]);

    Accumulators[0] = _mm_add_epi32(Accumulators[0], _mm_madd_epi16(BElements0, ABroadcast));
    Accumulators[1] = _mm_add_epi32(Accumulators[1], _mm_madd_epi16(BElements1, ABroadcast));
}

void
MlasGemmU8X8KernelSse(
    const int16_t* A,
    const int16_t* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountN,
    const int32_t* RowSumVector,
    const int32_t* ColumnSumVector,
    int32_t DepthValue,
    bool ZeroMode
    )
/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    single row.

Arguments:

    A - Supplies the address of matrix A. The matrix data has been packed
        using MlasGemmU8X8CopyPackASse.

    B - Supplies the address of matrix B. The matrix data has been packed
        using MlasGemmU8X8CopyPackBSse.

    C - Supplies the address of matrix C.

    PackedCountK - Supplies the number of packed columns from matrix A and the
        number of packed rows from matrix B to iterate over.

    CountN - Supplies the number of columns from matrix B and matrix C to iterate
        over.

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

    None.

--*/
{
    while (CountN > 0) {

        __m128i Accumulators[2];

        //
        // Initialize the accumulators with the sum of the global depth value
        // constant, the column sums, and the row sums.
        //

        Accumulators[0] = _mm_set1_epi32(DepthValue);
        Accumulators[0] = _mm_add_epi32(Accumulators[0], _mm_set1_epi32(RowSumVector[0]));
        Accumulators[1] = Accumulators[0];
        Accumulators[0] = _mm_add_epi32(Accumulators[0], _mm_loadu_si128((__m128i*)&ColumnSumVector[0]));
        Accumulators[1] = _mm_add_epi32(Accumulators[1], _mm_loadu_si128((__m128i*)&ColumnSumVector[4]));
        ColumnSumVector += 8;

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
                Accumulators[0] = _mm_add_epi32(Accumulators[0], _mm_loadu_si128((__m128i*)&C[0]));
                Accumulators[1] = _mm_add_epi32(Accumulators[1], _mm_loadu_si128((__m128i*)&C[4]));
            }

            _mm_storeu_si128((__m128i*)&C[0], Accumulators[0]);
            _mm_storeu_si128((__m128i*)&C[4], Accumulators[1]);

            C += 8;
            CountN -= 8;

        } else {

            //
            // Output the remaining partial output block.
            //

            if ((CountN & 4) != 0) {

                if (!ZeroMode) {
                    Accumulators[0] = _mm_add_epi32(Accumulators[0], _mm_loadu_si128((__m128i*)&C[0]));
                }

                _mm_storeu_si128((__m128i*)&C[0], Accumulators[0]);
                C += 4;

                Accumulators[0] = Accumulators[1];
            }

            if ((CountN & 2) != 0) {

                if (!ZeroMode) {
                    Accumulators[0] = _mm_add_epi32(Accumulators[0], _mm_loadl_epi64((__m128i*)&C[0]));
                }

                _mm_storel_epi64((__m128i*)&C[0], Accumulators[0]);
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
}

void
MlasGemmU8X8OutputFloatSse(
    const MLAS_GEMM_U8X8_WORK_BLOCK* WorkBlock,
    int32_t* C,
    size_t StartN,
    size_t CountM,
    size_t CountN
    )
/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    single row.

Arguments:

    WorkBlock - Supplies the structure containing the GEMM parameters.

    C - Supplies the address of matrix C.

    StartN - Supplies the starting column offset relative to the base of the
        work block. This is used to offset into column vectors accessed via the
        work block.

    CountM - Supplies the number of rows of the output matrix to process.

    CountN - Supplies the number of columns of the output matrix to process.

Return Value:

    None.

--*/
{
    const size_t ldc = WorkBlock->ldc;
    __m128 ScaleVector = _mm_load_ps1(WorkBlock->Scale);

    //
    // Check if the optional bias vector was supplied.
    //

    const float* BiasFloat = WorkBlock->BiasFloat;

    if (BiasFloat != nullptr) {

        BiasFloat += StartN;

        while (CountM-- > 0) {

            const float* bias = BiasFloat;
            int32_t* c = C;
            size_t n = CountN;

            while (n >= 4) {

                __m128 FloatVector = _mm_cvtepi32_ps(_mm_loadu_si128((__m128i*)c));
                FloatVector = _mm_mul_ps(FloatVector, ScaleVector);
                FloatVector = _mm_add_ps(FloatVector, _mm_loadu_ps(bias));
                _mm_storeu_ps((float*)c, FloatVector);

                bias += 4;
                c += 4;
                n -= 4;
            }

            for (size_t offset = 0; offset < n; offset++) {

                __m128 FloatVector = _mm_set_ss(float(c[offset]));
                FloatVector = _mm_mul_ss(FloatVector, ScaleVector);
                FloatVector = _mm_add_ss(FloatVector, _mm_load_ss(&bias[offset]));
                _mm_store_ss((float*)&c[offset], FloatVector);
            }

            C += ldc;
        }

    } else {

        while (CountM-- > 0) {

            int32_t* c = C;
            size_t n = CountN;

            while (n >= 4) {

                __m128 FloatVector = _mm_cvtepi32_ps(_mm_loadu_si128((__m128i*)c));
                FloatVector = _mm_mul_ps(FloatVector, ScaleVector);
                _mm_storeu_ps((float*)c, FloatVector);

                c += 4;
                n -= 4;
            }

            for (size_t offset = 0; offset < n; offset++) {

                __m128 FloatVector = _mm_set_ss((float)c[offset]);
                FloatVector = _mm_mul_ss(FloatVector, ScaleVector);
                _mm_store_ss((float*)&c[offset], FloatVector);
            }

            C += ldc;
        }
    }
}

struct MLAS_GEMM_U8X8_KERNEL_SSE
{
    typedef int16_t PackedAType;
    typedef int16_t PackedBType;
    typedef int8_t OffsetBType;

    static constexpr size_t PackedK = 2;
    static constexpr size_t StrideM = 12;
    static constexpr size_t StrideN = 128;
    static constexpr size_t StrideK = 128;

    MLAS_FORCEINLINE
    static
    void
    CopyPackA(
        PackedAType* D,
        const uint8_t* A,
        size_t lda,
        size_t CountM,
        size_t CountK,
        int32_t* RowSumVector,
        int16_t offb
        )
    {
        MlasGemmU8X8CopyPackASse(D, A, lda, CountM, CountK, RowSumVector, offb);
    }

    MLAS_FORCEINLINE
    static
    void
    CopyPackB(
        PackedBType* D,
        const uint8_t* B,
        size_t ldb,
        size_t CountN,
        size_t CountK,
        int32_t* ColumnSumVector,
        int16_t offa,
        bool BTypeIsSigned
        )
    {
        MlasGemmU8X8CopyPackBSse(D, B, ldb, CountN, CountK, ColumnSumVector, offa,
            BTypeIsSigned);
    }

    MLAS_FORCEINLINE
    static
    size_t
    Kernel(
        const PackedAType* A,
        const PackedBType* B,
        int32_t* C,
        size_t PackedCountK,
        size_t CountM,
        size_t CountN,
        size_t ldc,
        const int32_t* RowSumVector,
        const int32_t* ColumnSumVector,
        int32_t DepthValue,
        bool ZeroMode
        )
    {
        MLAS_UNREFERENCED_PARAMETER(CountM);
        MLAS_UNREFERENCED_PARAMETER(ldc);

        MlasGemmU8X8KernelSse(A, B, C, PackedCountK, CountN, RowSumVector,
            ColumnSumVector, DepthValue, ZeroMode);

        return 1;
    }

    MLAS_FORCEINLINE
    static
    void
    OutputFloat(
        const MLAS_GEMM_U8X8_WORK_BLOCK* WorkBlock,
        int32_t* C,
        size_t StartN,
        size_t CountM,
        size_t CountN
        )
    {
        MlasGemmU8X8OutputFloatSse(WorkBlock, C, StartN, CountM, CountN);
    }
};

constexpr size_t MLAS_GEMM_U8X8_KERNEL_SSE::PackedK;
constexpr size_t MLAS_GEMM_U8X8_KERNEL_SSE::StrideM;
constexpr size_t MLAS_GEMM_U8X8_KERNEL_SSE::StrideN;
constexpr size_t MLAS_GEMM_U8X8_KERNEL_SSE::StrideK;

void
MLASCALL
MlasGemmU8X8OperationSse(
    const MLAS_GEMM_U8X8_WORK_BLOCK* WorkBlock
    )
/*++

Routine Description:

    This module implements the quantized integer matrix/matrix multiply
    operation (QGEMM).

    This implementation supports SSE2 U8S8/U8U8.

Arguments:

    WorkBlock - Supplies the structure containing the GEMM parameters.

Return Value:

    None.

--*/
{
    return MlasGemmU8X8Operation<MLAS_GEMM_U8X8_KERNEL_SSE>(WorkBlock);
}

#endif

#ifdef MLAS_TARGET_AMD64

//
// Stores a vector to transpose a 4x4 byte vector using vpshufb.
//

MLAS_INTERNAL_DATA MLAS_DECLSPEC_ALIGN(const uint8_t MlasTranspose4x4BytesAvx[16], 16) =
    { 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 };

//
// Define the prototypes of the AVX2/AVX512 routines written in assembly.
//

extern "C" {

    void
    MLASCALL
    MlasGemmU8S8CopyPackAAvx2(
        uint8_t* D,
        const uint8_t* A,
        size_t lda,
        size_t CountM,
        size_t CountK,
        int32_t* RowSumVector,
        int16_t offb
        );

    void
    MLASCALL
    MlasGemmU8S8CopyPackBAvx2(
        uint8_t* D,
        const uint8_t* B,
        size_t ldb,
        size_t CountN,
        size_t CountK,
        int32_t* ColumnSumVector,
        int16_t offa,
        bool BTypeIsSigned
        );

    void
    MLASCALL
    MlasGemmU8U8CopyPackAAvx2(
        int16_t* D,
        const uint8_t* A,
        size_t lda,
        size_t CountM,
        size_t CountK,
        int32_t* RowSumVector,
        int16_t offb
        );

    void
    MLASCALL
    MlasGemmU8U8CopyPackBAvx2(
        uint8_t* D,
        const uint8_t* B,
        size_t ldb,
        size_t CountN,
        size_t CountK,
        int32_t* ColumnSumVector,
        int16_t offa
        );
}

struct MLAS_GEMM_U8S8_KERNEL_AVX2
{
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef int8_t OffsetBType;

    static constexpr size_t PackedK = 4;
    static constexpr size_t StrideM = 24;
    static constexpr size_t StrideN = 256;
    static constexpr size_t StrideK = 128;

    MLAS_FORCEINLINE
    static
    void
    CopyPackA(
        PackedAType* D,
        const uint8_t* A,
        size_t lda,
        size_t CountM,
        size_t CountK,
        int32_t* RowSumVector,
        int16_t offb
        )
    {
        MlasGemmU8S8CopyPackAAvx2(D, A, lda, CountM, CountK, RowSumVector, offb);
    }

    MLAS_FORCEINLINE
    static
    void
    CopyPackB(
        PackedBType* D,
        const uint8_t* B,
        size_t ldb,
        size_t CountN,
        size_t CountK,
        int32_t* ColumnSumVector,
        int16_t offa,
        bool BTypeIsSigned
        )
    {
        MlasGemmU8S8CopyPackBAvx2(D, B, ldb, CountN, CountK, ColumnSumVector, offa,
            BTypeIsSigned);
    }

    MLAS_FORCEINLINE
    static
    size_t
    Kernel(
        const PackedAType* A,
        const PackedBType* B,
        int32_t* C,
        size_t PackedCountK,
        size_t CountM,
        size_t CountN,
        size_t ldc,
        const int32_t* RowSumVector,
        const int32_t* ColumnSumVector,
        int32_t DepthValue,
        bool ZeroMode
        )
    {
        return MlasPlatform.GemmU8S8Kernel(A, B, C, PackedCountK, CountM, CountN,
            ldc, RowSumVector, ColumnSumVector, DepthValue, ZeroMode);
    }

    MLAS_FORCEINLINE
    static
    void
    OutputFloat(
        const MLAS_GEMM_U8X8_WORK_BLOCK* WorkBlock,
        int32_t* C,
        size_t StartN,
        size_t CountM,
        size_t CountN
        )
    {
        MlasGemmU8X8OutputFloatSse(WorkBlock, C, StartN, CountM, CountN);
    }
};

constexpr size_t MLAS_GEMM_U8S8_KERNEL_AVX2::PackedK;
constexpr size_t MLAS_GEMM_U8S8_KERNEL_AVX2::StrideM;
constexpr size_t MLAS_GEMM_U8S8_KERNEL_AVX2::StrideN;
constexpr size_t MLAS_GEMM_U8S8_KERNEL_AVX2::StrideK;

void
MLASCALL
MlasGemmU8S8OperationAvx2(
    const MLAS_GEMM_U8X8_WORK_BLOCK* WorkBlock
    )
/*++

Routine Description:

    This module implements the quantized integer matrix/matrix multiply
    operation (QGEMM).

    This implementation supports AVX2/AVX512 U8S8 and AVX512VNNI U8S8/U8U8.

Arguments:

    WorkBlock - Supplies the structure containing the GEMM parameters.

Return Value:

    None.

--*/
{
    if ((WorkBlock->M == 1) && WorkBlock->BTypeIsSigned && !WorkBlock->CTypeIsFloat &&
        (WorkBlock->offa == 0) && (WorkBlock->offb == 0)) {

        if (MlasPlatform.GemvU8S8Kernel != nullptr) {
            MlasPlatform.GemvU8S8Kernel(WorkBlock->A, WorkBlock->B, WorkBlock->C,
                WorkBlock->K, WorkBlock->N, WorkBlock->ldb);
            return;
        }
    }

    return MlasGemmU8X8Operation<MLAS_GEMM_U8S8_KERNEL_AVX2>(WorkBlock);
}

struct MLAS_GEMM_U8U8_KERNEL_AVX2
{
    typedef int16_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint8_t OffsetBType;

    static constexpr size_t PackedK = 2;
    static constexpr size_t StrideM = 24;
    static constexpr size_t StrideN = 256;
    static constexpr size_t StrideK = 128;

    MLAS_FORCEINLINE
    static
    void
    CopyPackA(
        PackedAType* D,
        const uint8_t* A,
        size_t lda,
        size_t CountM,
        size_t CountK,
        int32_t* RowSumVector,
        int16_t offb
        )
    {
        MlasGemmU8U8CopyPackAAvx2(D, A, lda, CountM, CountK, RowSumVector, offb);
    }

    MLAS_FORCEINLINE
    static
    void
    CopyPackB(
        PackedBType* D,
        const uint8_t* B,
        size_t ldb,
        size_t CountN,
        size_t CountK,
        int32_t* ColumnSumVector,
        int16_t offa,
        bool BTypeIsSigned
        )
    {
        MLAS_UNREFERENCED_PARAMETER(BTypeIsSigned);

        MlasGemmU8U8CopyPackBAvx2(D, B, ldb, CountN, CountK, ColumnSumVector, offa);
    }

    MLAS_FORCEINLINE
    static
    size_t
    Kernel(
        const PackedAType* A,
        const PackedBType* B,
        int32_t* C,
        size_t PackedCountK,
        size_t CountM,
        size_t CountN,
        size_t ldc,
        const int32_t* RowSumVector,
        const int32_t* ColumnSumVector,
        int32_t DepthValue,
        bool ZeroMode
        )
    {
        return MlasPlatform.GemmU8U8Kernel(A, B, C, PackedCountK, CountM, CountN,
            ldc, RowSumVector, ColumnSumVector, DepthValue, ZeroMode);
    }

    MLAS_FORCEINLINE
    static
    void
    OutputFloat(
        const MLAS_GEMM_U8X8_WORK_BLOCK* WorkBlock,
        int32_t* C,
        size_t StartN,
        size_t CountM,
        size_t CountN
        )
    {
        MlasGemmU8X8OutputFloatSse(WorkBlock, C, StartN, CountM, CountN);
    }
};

constexpr size_t MLAS_GEMM_U8U8_KERNEL_AVX2::PackedK;
constexpr size_t MLAS_GEMM_U8U8_KERNEL_AVX2::StrideM;
constexpr size_t MLAS_GEMM_U8U8_KERNEL_AVX2::StrideN;
constexpr size_t MLAS_GEMM_U8U8_KERNEL_AVX2::StrideK;

void
MLASCALL
MlasGemmU8U8OperationAvx2(
    const MLAS_GEMM_U8X8_WORK_BLOCK* WorkBlock
    )
/*++

Routine Description:

    This module implements the quantized integer matrix/matrix multiply
    operation (QGEMM).

    This implementation supports AVX2/AVX512 U8U8.

Arguments:

    WorkBlock - Supplies the structure containing the GEMM parameters.

Return Value:

    None.

--*/
{
    return MlasGemmU8X8Operation<MLAS_GEMM_U8U8_KERNEL_AVX2>(WorkBlock);
}

#endif

#ifdef MLAS_TARGET_AMD64_IX86

void
MlasGemmU8X8Threaded(
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
    const auto* WorkBlock = (MLAS_GEMM_U8X8_WORK_BLOCK*)Context;

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

    MLAS_GEMM_U8X8_WORK_BLOCK LocalWorkBlock;

    memcpy(&LocalWorkBlock, WorkBlock, sizeof(MLAS_GEMM_U8X8_WORK_BLOCK));

    LocalWorkBlock.M = CountM;
    LocalWorkBlock.N = CountN;
    LocalWorkBlock.A += m * LocalWorkBlock.lda;
    LocalWorkBlock.B += n;
    LocalWorkBlock.C += m * LocalWorkBlock.ldc + n;

    if (LocalWorkBlock.BiasFloat != nullptr) {
        LocalWorkBlock.BiasFloat += n;
    }

#if defined(MLAS_TARGET_AMD64)
    if (WorkBlock->BTypeIsSigned) {
        MlasPlatform.GemmU8S8Operation(&LocalWorkBlock);
    } else {
        MlasPlatform.GemmU8U8Operation(&LocalWorkBlock);
    }
#else
    MlasGemmU8X8OperationSse(&LocalWorkBlock);
#endif
}

void
MlasGemmU8X8Schedule(
    MLAS_GEMM_U8X8_WORK_BLOCK* WorkBlock,
    MLAS_THREADPOOL* ThreadPool
    )
/*++

Routine Description:

    This module implements the quantized integer matrix/matrix multiply
    operation (QGEMM).

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

    const double Complexity = double(M) * double(N) * double(K);

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

    MlasExecuteThreaded(MlasGemmU8X8Threaded, WorkBlock, TargetThreadCount, ThreadPool);
}

template<typename AType, typename BType>
void
MLASCALL
MlasGemm(
    size_t M,
    size_t N,
    size_t K,
    const AType* A,
    size_t lda,
    AType offa,
    const BType* B,
    size_t ldb,
    BType offb,
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
    MLAS_GEMM_U8X8_WORK_BLOCK WorkBlock;

    //
    // Capture the GEMM parameters to the work block.
    //

    memset(&WorkBlock, 0, sizeof(MLAS_GEMM_U8X8_WORK_BLOCK));

    WorkBlock.M = M;
    WorkBlock.N = N;
    WorkBlock.K = K;
    WorkBlock.A = A;
    WorkBlock.lda = lda;
    WorkBlock.B = (const uint8_t*)B;
    WorkBlock.ldb = ldb;
    WorkBlock.C = C;
    WorkBlock.ldc = ldc;
    WorkBlock.offa = offa;
    WorkBlock.offb = offb;
    WorkBlock.BTypeIsSigned = std::is_signed<BType>::value;

    //
    // Schedule the operation across a set of worker threads.
    //

    MlasGemmU8X8Schedule(&WorkBlock, ThreadPool);
}

template
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
    );

template
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
    );

template<typename AType, typename BType>
void
MLASCALL
MlasGemm(
    size_t M,
    size_t N,
    size_t K,
    const AType* A,
    size_t lda,
    AType offa,
    const BType* B,
    size_t ldb,
    BType offb,
    float* C,
    size_t ldc,
    const float* Scale,
    const float* Bias,
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
    MLAS_GEMM_U8X8_WORK_BLOCK WorkBlock;

    //
    // Capture the GEMM parameters to the work block.
    //

    memset(&WorkBlock, 0, sizeof(MLAS_GEMM_U8X8_WORK_BLOCK));

    WorkBlock.M = M;
    WorkBlock.N = N;
    WorkBlock.K = K;
    WorkBlock.A = A;
    WorkBlock.lda = lda;
    WorkBlock.B = (const uint8_t*)B;
    WorkBlock.ldb = ldb;
    WorkBlock.C = (int32_t*)C;
    WorkBlock.ldc = ldc;
    WorkBlock.Scale = Scale;
    WorkBlock.BiasFloat = Bias;
    WorkBlock.offa = offa;
    WorkBlock.offb = offb;
    WorkBlock.BTypeIsSigned = std::is_signed<BType>::value;
    WorkBlock.CTypeIsFloat = true;

    //
    // Schedule the operation across a set of worker threads.
    //

    MlasGemmU8X8Schedule(&WorkBlock, ThreadPool);
}

template
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
    float* C,
    size_t ldc,
    const float* Scale,
    const float* Bias,
    MLAS_THREADPOOL* ThreadPool
    );

template
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
    float* C,
    size_t ldc,
    const float* Scale,
    const float* Bias,
    MLAS_THREADPOOL* ThreadPool
    );

#endif
