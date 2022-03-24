/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm_kernel_sse41.cpp

Abstract:

    This module implements QGEMM kernels for sse41.

--*/

#include "mlasi.h"
#include "qgemm.h"

// N.B. MSVC does not require turning on SSE 4.1 intrinsics and the current use
// for this code is Windows only, so restrict this kernel to that environment.

struct MLAS_GEMM_U8S8_KERNEL_SSE41
{
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint8_t OffsetAType;
    typedef int8_t OffsetBType;

    static constexpr size_t PackedK = 4;
    static constexpr MLAS_GEMM_QUANT_STRIDES Strides{ 24, 128, 128 };
    static constexpr MLAS_GEMM_QUANT_STRIDES PackedStrides{ 24, 128, 128 };
};

constexpr size_t MLAS_GEMM_U8S8_KERNEL_SSE41::PackedK;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_U8S8_KERNEL_SSE41::Strides;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_U8S8_KERNEL_SSE41::PackedStrides;

template<>
void
MlasGemmQuantCopyPackA<MLAS_GEMM_U8S8_KERNEL_SSE41>(
    MLAS_GEMM_U8S8_KERNEL_SSE41::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer,
    bool AIsSigned
    )
{
    MLAS_UNREFERENCED_PARAMETER(AIsSigned);
    const __m128i ZeroVector = _mm_setzero_si128();
    const __m128i OnesWordBroadcast = _mm_set1_epi16(1);

    //
    // Process a single row of matrix A in a loop.
    //

    while (CountM > 0) {

        const uint8_t* a = A;
        size_t k = CountK;
        __m128i ReductionVector = ZeroVector;

        //
        // Copy the source bytes to the packed buffer.
        //
        // The packed buffer has the same data ordering as the source bytes,
        // but CountK is aligned up to a multiple of 4 to maintain 32-bit
        // alignment. All extra bytes are zero-padded.
        //

        while (k >= 8) {

            __m128i Bytes = _mm_loadl_epi64((const __m128i*) & a[0]);

            __m128i Words = _mm_unpacklo_epi8(Bytes, ZeroVector);
            ReductionVector = _mm_add_epi32(ReductionVector, _mm_madd_epi16(Words, OnesWordBroadcast));

            _mm_storel_epi64((__m128i*) & D[0], Bytes);

            a += 8;
            D += 8;
            k -= 8;
        }

        if (k > 0) {

            //
            // Copy the remaining bytes to the zero padded stack buffer.
            //

            _mm_storel_epi64((__m128i*) & D[0], ZeroVector);

            std::copy_n(&a[0], k, &D[0]);

            __m128i Bytes = _mm_loadl_epi64((__m128i*) & D[0]);
            D += (k + 3) & ~3;

            __m128i Words = _mm_unpacklo_epi8(Bytes, ZeroVector);
            ReductionVector = _mm_add_epi32(ReductionVector, _mm_madd_epi16(Words, OnesWordBroadcast));
        }

        //
        // Reduce the partial accumulators.
        //

        ReductionVector = _mm_hadd_epi32(ReductionVector, ReductionVector);
        ReductionVector = _mm_hadd_epi32(ReductionVector, ReductionVector);

        *RowSumBuffer++ = _mm_cvtsi128_si32(ReductionVector);

        A += lda;
        CountM -= 1;
    }
}

MLAS_FORCEINLINE
void
MlasGemmU8X8CopyPackBProcessSse41(
    MLAS_GEMM_U8S8_KERNEL_SSE41::PackedBType* D,
    __m128i BytesRows[4],
    __m128i OnesByteBroadcast,
    __m128i OnesWordBroadcast,
    __m128i ColumnSums[2]
)
{
    __m128i PairsInterleaved0 = _mm_unpacklo_epi8(BytesRows[0], BytesRows[1]);
    __m128i PairsInterleaved1 = _mm_unpacklo_epi8(BytesRows[2], BytesRows[3]);

    __m128i QuadsInterleaved0 = _mm_unpacklo_epi16(PairsInterleaved0, PairsInterleaved1);
    __m128i QuadsInterleaved1 = _mm_unpackhi_epi16(PairsInterleaved0, PairsInterleaved1);

    __m128i PairwiseAdd0 = _mm_maddubs_epi16(OnesByteBroadcast, QuadsInterleaved0);
    __m128i PairwiseAdd1 = _mm_maddubs_epi16(OnesByteBroadcast, QuadsInterleaved1);

    PairwiseAdd0 = _mm_madd_epi16(PairwiseAdd0, OnesWordBroadcast);
    PairwiseAdd1 = _mm_madd_epi16(PairwiseAdd1, OnesWordBroadcast);

    ColumnSums[0] = _mm_add_epi32(ColumnSums[0], PairwiseAdd0);
    ColumnSums[1] = _mm_add_epi32(ColumnSums[1], PairwiseAdd1);

    _mm_storeu_si128((__m128i*) & D[0], QuadsInterleaved0);
    _mm_storeu_si128((__m128i*) & D[16], QuadsInterleaved1);
}

template<>
void
MlasGemmQuantCopyPackB<MLAS_GEMM_U8S8_KERNEL_SSE41>(
    MLAS_GEMM_U8S8_KERNEL_SSE41::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    const __m128i OnesByteBroadcast = _mm_set1_epi8(1);
    const __m128i OnesWordBroadcast = _mm_set1_epi16(1);
    __m128i BytesRows[4];

    MLAS_UNREFERENCED_PARAMETER(BIsSigned);

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

        while (k >= MLAS_GEMM_U8S8_KERNEL_SSE41::PackedK) {

            BytesRows[0] = _mm_loadl_epi64((const __m128i*) & b[ldb * 0]);
            BytesRows[1] = _mm_loadl_epi64((const __m128i*) & b[ldb * 1]);
            BytesRows[2] = _mm_loadl_epi64((const __m128i*) & b[ldb * 2]);
            BytesRows[3] = _mm_loadl_epi64((const __m128i*) & b[ldb * 3]);

            MlasGemmU8X8CopyPackBProcessSse41(D, BytesRows, OnesByteBroadcast, OnesWordBroadcast, ColumnSums);

            b += ldb * 4;
            D += 32;
            k -= 4;
        }

        if (k > 0) {

            BytesRows[0] = _mm_loadl_epi64((const __m128i*) & b[ldb * 0]);
            BytesRows[1] = _mm_setzero_si128();
            BytesRows[2] = _mm_setzero_si128();
            BytesRows[3] = _mm_setzero_si128();

            if (k >= 2) {
                BytesRows[1] = _mm_loadl_epi64((const __m128i*) & b[ldb * 1]);
            }

            if (k >= 3) {
                BytesRows[2] = _mm_loadl_epi64((const __m128i*) & b[ldb * 2]);
            }

            MlasGemmU8X8CopyPackBProcessSse41(D, BytesRows, OnesByteBroadcast, OnesWordBroadcast, ColumnSums);

            D += 32;
        }

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

        const __m128i ZeroVector = _mm_setzero_si128();

        __m128i ColumnSums[2];
        uint8_t PaddedMatrixBData[32];

        ColumnSums[0] = _mm_setzero_si128();
        ColumnSums[1] = _mm_setzero_si128();

        while (CountK > 0) {

            size_t k = std::min(CountK, MLAS_GEMM_U8S8_KERNEL_SSE41::PackedK);
            CountK -= k;

            _mm_storeu_si128((__m128i*) & PaddedMatrixBData[0], ZeroVector);
            _mm_storeu_si128((__m128i*) & PaddedMatrixBData[16], ZeroVector);

            uint8_t* padded = PaddedMatrixBData;

            do {

                std::copy_n(B, CountN, padded);

                padded += 8;
                B += ldb;
                k -= 1;

            } while (k > 0);

            BytesRows[0] = _mm_loadl_epi64((__m128i*) & PaddedMatrixBData[0]);
            BytesRows[1] = _mm_loadl_epi64((__m128i*) & PaddedMatrixBData[8]);
            BytesRows[2] = _mm_loadl_epi64((__m128i*) & PaddedMatrixBData[16]);
            BytesRows[3] = _mm_loadl_epi64((__m128i*) & PaddedMatrixBData[24]);

            MlasGemmU8X8CopyPackBProcessSse41(D, BytesRows, OnesByteBroadcast, OnesWordBroadcast, ColumnSums);

            D += 32;
        }

        _mm_storeu_si128((__m128i*) & ColumnSumBuffer[0], ColumnSums[0]);
        _mm_storeu_si128((__m128i*) & ColumnSumBuffer[4], ColumnSums[1]);
    }
}

MLAS_FORCEINLINE
void
MlasGemmU8X8MultiplyAccumulateRowSse41(
    __m128i ABroadcast,
    const MLAS_GEMM_U8S8_KERNEL_SSE41::PackedBType* B,
    __m128i OnesWordBroadcast,
    __m128i Accumulators[2]
)
{
    __m128i BElements0 = _mm_load_si128((__m128i*) & B[0]);
    __m128i BElements1 = _mm_load_si128((__m128i*) & B[16]);

    __m128i Intermediate0 = _mm_maddubs_epi16(ABroadcast, BElements0);
    __m128i Intermediate1 = _mm_maddubs_epi16(ABroadcast, BElements1);

    Accumulators[0] = _mm_add_epi32(Accumulators[0], _mm_madd_epi16(Intermediate0, OnesWordBroadcast));
    Accumulators[1] = _mm_add_epi32(Accumulators[1], _mm_madd_epi16(Intermediate1, OnesWordBroadcast));
}

template<>
size_t
MlasGemmQuantKernel<MLAS_GEMM_U8S8_KERNEL_SSE41>(
    const MLAS_GEMM_U8S8_KERNEL_SSE41::PackedAType* A,
    const MLAS_GEMM_U8S8_KERNEL_SSE41::PackedBType* B,
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
    const __m128i OnesWordBroadcast = _mm_set1_epi16(1);

    MLAS_UNREFERENCED_PARAMETER(CountM);
    MLAS_UNREFERENCED_PARAMETER(ldc);

    while (CountN > 0) {

        __m128i Accumulators[2];

        //
        // Initialize the accumulators with the row and column sums.
        //

        Accumulators[0] = _mm_set1_epi32(RowSumBuffer[0]);
        Accumulators[1] = Accumulators[0];

        if (ZeroPointB != nullptr) {
            Accumulators[0] = _mm_mullo_epi32(Accumulators[0], _mm_loadu_si128((const __m128i*) & ZeroPointB[0]));
            Accumulators[1] = _mm_mullo_epi32(Accumulators[1], _mm_loadu_si128((const __m128i*) & ZeroPointB[4]));
            ZeroPointB += 8;
        }

        Accumulators[0] = _mm_add_epi32(Accumulators[0], _mm_loadu_si128((const __m128i*) & ColumnSumBuffer[0]));
        Accumulators[1] = _mm_add_epi32(Accumulators[1], _mm_loadu_si128((const __m128i*) & ColumnSumBuffer[4]));
        ColumnSumBuffer += 8;

        //
        // Broadcast each quad of 8-bit values from the matrix A and multiply
        // with the quad of 8-bit values from matrix B, and add the 32-bit
        // intermediate into the accumulator registers.
        //

        const uint8_t* a = A;
        size_t k = PackedCountK;

        while (k >= 4) {

            __m128i AElements = _mm_loadu_si128((__m128i*)a);
            __m128i ABroadcast;

            ABroadcast = _mm_shuffle_epi32(AElements, _MM_SHUFFLE(0, 0, 0, 0));
            MlasGemmU8X8MultiplyAccumulateRowSse41(ABroadcast, &B[0], OnesWordBroadcast, Accumulators);

            ABroadcast = _mm_shuffle_epi32(AElements, _MM_SHUFFLE(1, 1, 1, 1));
            MlasGemmU8X8MultiplyAccumulateRowSse41(ABroadcast, &B[32], OnesWordBroadcast, Accumulators);

            ABroadcast = _mm_shuffle_epi32(AElements, _MM_SHUFFLE(2, 2, 2, 2));
            MlasGemmU8X8MultiplyAccumulateRowSse41(ABroadcast, &B[64], OnesWordBroadcast, Accumulators);

            ABroadcast = _mm_shuffle_epi32(AElements, _MM_SHUFFLE(3, 3, 3, 3));
            MlasGemmU8X8MultiplyAccumulateRowSse41(ABroadcast, &B[96], OnesWordBroadcast, Accumulators);

            a += 4 * 4;
            B += 4 * 32;
            k -= 4;
        }

        while (k > 0) {

            __m128i ABroadcast = _mm_set1_epi32(*((int32_t*)a));
            MlasGemmU8X8MultiplyAccumulateRowSse41(ABroadcast, &B[0], OnesWordBroadcast, Accumulators);

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

const MLAS_GEMM_QUANT_DISPATCH MlasGemmU8S8DispatchSse41 = {
    MlasGemmQuantOperation<MLAS_GEMM_U8S8_KERNEL_SSE41>,
    MlasGemmQuantPackedOperation<MLAS_GEMM_U8S8_KERNEL_SSE41>,
    MlasGemmQuantCopyPackB<MLAS_GEMM_U8S8_KERNEL_SSE41>,
    MLAS_GEMM_U8S8_KERNEL_SSE41::PackedK,
    MLAS_GEMM_U8S8_KERNEL_SSE41::PackedStrides.K,
};
