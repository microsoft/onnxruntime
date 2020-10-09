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
    size_t RangeStartM;
    size_t RangeStartN;
    size_t RangeCountM;
    size_t RangeCountN;
    size_t M;
    size_t N;
    size_t K;
    const uint8_t* A;
    size_t lda;
    const void* B;
    size_t ldb;
    int32_t* C;
    size_t ldc;
    const float* Scale;
    const float* BiasFloat;
    uint8_t offa;
    uint8_t offb;
    bool BIsPacked;
    bool BIsSigned;
    bool CIsFloat;
};

//
// Define the default striding parameters used for the quantized integer
// matrix/matrix multiply operation.
//

struct MLAS_GEMM_U8X8_STRIDES {
    size_t M;
    size_t N;
    size_t K;
};

void
MlasGemmU8X8ScaleSumBuffer(
    int32_t* Output,
    const int32_t* Input,
    size_t N,
    int32_t Scale
    )
{
    for (size_t n = 0; n < N; n++) {
        Output[n] = Input[n] * Scale;
    }
}

MLAS_FORCEINLINE
void
MlasGemmU8X8ScaleSumBuffer(
    int32_t* SumBuffer,
    size_t N,
    int32_t Scale
    )
{
    return MlasGemmU8X8ScaleSumBuffer(SumBuffer, SumBuffer, N, Scale);
}

void
MlasGemmU8X8OutputFloat(
    const MLAS_GEMM_U8X8_WORK_BLOCK* WorkBlock,
    int32_t* C,
    size_t StartN,
    size_t CountM,
    size_t CountN
    )
/*++

Routine Description:

    This routine converts the output matrix to a floating point format using
    the supplied scale and bias parameters.

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
    const MLAS_FLOAT32X4 ScaleVector = MlasBroadcastFloat32x4(WorkBlock->Scale);
#if !defined(MLAS_SSE2_INTRINSICS)
    const float ScaleValue = MlasExtractLaneFloat32x4<0>(ScaleVector);
#endif

    //
    // Check if the optional bias vector was supplied.
    //

    const float* BiasFloat = WorkBlock->BiasFloat;

    if (BiasFloat != nullptr) {

        BiasFloat += WorkBlock->RangeStartN + StartN;

        while (CountM-- > 0) {

            const float* bias = BiasFloat;
            int32_t* c = C;
            size_t n = CountN;

            while (n >= 4) {

                MLAS_FLOAT32X4 FloatVector = MlasCastToFloat32x4(MlasLoadInt32x4(c));
                FloatVector = MlasMultiplyFloat32x4(FloatVector, ScaleVector);
                FloatVector = MlasAddFloat32x4(FloatVector, MlasLoadFloat32x4(bias));
                MlasStoreFloat32x4(reinterpret_cast<float*>(c), FloatVector);

                bias += 4;
                c += 4;
                n -= 4;
            }

            for (size_t offset = 0; offset < n; offset++) {

#if defined(MLAS_SSE2_INTRINSICS)
                __m128 FloatVector = _mm_set_ss(float(c[offset]));
                FloatVector = _mm_mul_ss(FloatVector, ScaleVector);
                FloatVector = _mm_add_ss(FloatVector, _mm_load_ss(&bias[offset]));
                _mm_store_ss(reinterpret_cast<float*>(&c[offset]), FloatVector);
#else
                *reinterpret_cast<float*>(&c[offset]) = float(c[offset]) * ScaleValue + bias[offset];
#endif
            }

            C += ldc;
        }

    } else {

        while (CountM-- > 0) {

            int32_t* c = C;
            size_t n = CountN;

            while (n >= 4) {

                MLAS_FLOAT32X4 FloatVector = MlasCastToFloat32x4(MlasLoadInt32x4(c));
                FloatVector = MlasMultiplyFloat32x4(FloatVector, ScaleVector);
                MlasStoreFloat32x4(reinterpret_cast<float*>(c), FloatVector);

                c += 4;
                n -= 4;
            }

            for (size_t offset = 0; offset < n; offset++) {

#if defined(MLAS_SSE2_INTRINSICS)
                __m128 FloatVector = _mm_set_ss((float)c[offset]);
                FloatVector = _mm_mul_ss(FloatVector, ScaleVector);
                _mm_store_ss(reinterpret_cast<float*>(&c[offset]), FloatVector);
#else
                *reinterpret_cast<float*>(&c[offset]) = float(c[offset]) * ScaleValue;
#endif
            }

            C += ldc;
        }
    }
}

template<typename KernelType>
void
MLASCALL
MlasGemmU8X8Operation(
    const MLAS_GEMM_U8X8_WORK_BLOCK* WorkBlock
    )
/*++

Routine Description:

    This routine implements the quantized integer matrix/matrix multiply
    operation (QGEMM).

Arguments:

    WorkBlock - Supplies the structure containing the GEMM parameters.

Return Value:

    None.

--*/
{
    constexpr MLAS_GEMM_U8X8_STRIDES Strides = KernelType::Strides;

    MLAS_DECLSPEC_ALIGN(typename KernelType::PackedAType PanelA[Strides.M * Strides.K], 64);
    MLAS_DECLSPEC_ALIGN(typename KernelType::PackedBType PanelB[Strides.N * Strides.K], 64);

    MLAS_DECLSPEC_ALIGN(int32_t RowSumBuffer[Strides.M], 64);
    MLAS_DECLSPEC_ALIGN(int32_t ColumnSumBuffer[Strides.N], 64);

    const size_t M = WorkBlock->RangeCountM;
    const size_t N = WorkBlock->RangeCountN;
    const size_t K = WorkBlock->K;

    const size_t lda = WorkBlock->lda;
    const size_t ldb = WorkBlock->ldb;
    const size_t ldc = WorkBlock->ldc;

    const uint8_t* A = WorkBlock->A + WorkBlock->RangeStartM * lda;
    const uint8_t* B = (const uint8_t*)WorkBlock->B + WorkBlock->RangeStartN;
    int32_t* C = WorkBlock->C + WorkBlock->RangeStartM * ldc + WorkBlock->RangeStartN;

    int32_t offa = WorkBlock->offa;
    int32_t offb = typename KernelType::OffsetBType(WorkBlock->offb);

    //
    // Try to use a GEMV kernel if supported by this kernel type.
    //

    if ((M == 1) && (offa == 0) && (offb == 0) && !WorkBlock->CIsFloat) {
        if (KernelType::TryGemvKernel(A, B, ldb, C, K, N, WorkBlock->BIsSigned)) {
            return;
        }
    }

    //
    // Flip the sign bit of the zero point offset of matrix B if the kernel uses
    // signed types and the matrix B data is unsigned.
    //

#if defined(MLAS_SSE2_INTRINSICS)
    if (std::is_signed<typename KernelType::OffsetBType>::value) {
        if (!WorkBlock->BIsSigned) {
            offb = typename KernelType::OffsetBType(offb ^ 0x80);
        }
    }
#elif defined(MLAS_NEON64_INTRINSICS)
    if (WorkBlock->BIsSigned) {
        offb = typename KernelType::OffsetBType(offb ^ 0x80);
    }
#endif

    //
    // Step through each slice of matrix B along the K dimension.
    //

    size_t CountK;

    for (size_t k = 0; k < K; k += CountK) {

        CountK = std::min(K - k, Strides.K);

        //
        // Step through each slice of matrix B along the N dimension.
        //

        size_t CountN;

        for (size_t n = 0; n < N; n += CountN) {

            CountN = std::min(N - n, Strides.N);

            //
            // Copy a panel of matrix B to a local packed buffer.
            //

            KernelType::CopyPackB(PanelB, B + n, ldb, CountN, CountK,
                ColumnSumBuffer, WorkBlock->BIsSigned);

            MlasGemmU8X8ScaleSumBuffer(ColumnSumBuffer, CountN, -offa);

            //
            // Step through each slice of matrix A along the M dimension.
            //

            const int32_t DepthValue = int32_t(CountK) * offa * offb;
            const size_t PackedCountK = (CountK + KernelType::PackedK - 1) /
                KernelType::PackedK;

            int32_t* c = C + n;
            size_t CountM;

            for (size_t m = 0; m < M; m += CountM) {

                CountM = std::min(M - m, Strides.M);

                //
                // Copy a panel of matrix A to a local packed buffer.
                //

                KernelType::CopyPackA(PanelA, A + m * lda, lda, CountM, CountK,
                    RowSumBuffer);

                MlasGemmU8X8ScaleSumBuffer(RowSumBuffer, CountM, -offb);

                //
                // Step through the rows of the local packed buffer.
                //

                typename KernelType::PackedAType* pa = PanelA;
                int32_t* RowSums = RowSumBuffer;
                size_t RowsRemaining = CountM;

                bool ZeroMode = (k == 0);
                bool PostProcess = (k + CountK == K);

                while (RowsRemaining > 0) {

                    size_t RowsHandled;

                    RowsHandled = KernelType::GemmKernel(pa, PanelB, c, PackedCountK,
                        RowsRemaining, CountN, ldc, RowSums, ColumnSumBuffer,
                        DepthValue, ZeroMode);

                    if (PostProcess && WorkBlock->CIsFloat) {
                        MlasGemmU8X8OutputFloat(WorkBlock, c, n, RowsHandled, CountN);
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

template<typename KernelType>
void
MLASCALL
MlasGemmU8X8PackedOperation(
    const MLAS_GEMM_U8X8_WORK_BLOCK* WorkBlock
    )
/*++

Routine Description:

    This routine implements the quantized integer matrix/matrix multiply
    operation (QGEMM).

Arguments:

    WorkBlock - Supplies the structure containing the GEMM parameters.

Return Value:

    None.

--*/
{
    constexpr MLAS_GEMM_U8X8_STRIDES Strides = KernelType::PackedStrides;

    MLAS_DECLSPEC_ALIGN(typename KernelType::PackedAType PanelA[Strides.M * Strides.K], 64);

    MLAS_DECLSPEC_ALIGN(int32_t RowSumBuffer[Strides.M], 64);
    MLAS_DECLSPEC_ALIGN(int32_t ColumnSumBuffer[Strides.N], 64);

    const size_t M = WorkBlock->RangeCountM;
    const size_t N = WorkBlock->RangeCountN;
    const size_t K = WorkBlock->K;

    const size_t lda = WorkBlock->lda;
    const size_t ldc = WorkBlock->ldc;

    const uint8_t* A = WorkBlock->A + WorkBlock->RangeStartM * lda;
    const uint8_t* PackedB = (const uint8_t*)WorkBlock->B;
    int32_t* C = WorkBlock->C + WorkBlock->RangeStartM * ldc + WorkBlock->RangeStartN;

    int32_t offa = WorkBlock->offa;
    int32_t offb = typename KernelType::OffsetBType(WorkBlock->offb);

    //
    // Flip the sign bit of the zero point offset of matrix B if the kernel uses
    // signed types and the matrix B data is unsigned.
    //

#if defined(MLAS_SSE2_INTRINSICS)
    if (std::is_signed<typename KernelType::OffsetBType>::value) {
        if (!WorkBlock->BIsSigned) {
            offb = typename KernelType::OffsetBType(offb ^ 0x80);
        }
    }
#elif defined(MLAS_NEON64_INTRINSICS)
    if (WorkBlock->BIsSigned) {
        offb = typename KernelType::OffsetBType(offb ^ 0x80);
    }
#endif

    //
    // Extract the pointer to the column sum buffer from the packed matrix.
    //

    const size_t AlignedN =
        (WorkBlock->N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1);
    const int32_t* PackedColumnSumBuffer = (const int32_t*)PackedB;
    PackedB = (const uint8_t*)(PackedColumnSumBuffer + AlignedN);
    PackedColumnSumBuffer += WorkBlock->RangeStartN;

    //
    // Step through each slice of matrix B along the K dimension.
    //

    size_t CountK;

    for (size_t k = 0; k < K; k += CountK) {

        CountK = std::min(K - k, Strides.K);

        const size_t PackedCountK = (CountK + KernelType::PackedK - 1) /
            KernelType::PackedK;

        if (k > 0) {
            std::fill_n(ColumnSumBuffer, Strides.N, 0);
        }

        //
        // Step through each slice of matrix B along the N dimension.
        //

        size_t CountN;

        for (size_t n = 0; n < N; n += CountN) {

            CountN = std::min(N - n, Strides.N);

            if (k == 0) {
                MlasGemmU8X8ScaleSumBuffer(ColumnSumBuffer, PackedColumnSumBuffer + n,
                    CountN, -offa);
            }

            //
            // Step through each slice of matrix A along the M dimension.
            //

            const int32_t DepthValue = int32_t(CountK) * offa * offb;
            const uint8_t* b = PackedB + (WorkBlock->RangeStartN + n) *
                KernelType::PackedK * PackedCountK;
            int32_t* c = C + n;
            size_t CountM;

            for (size_t m = 0; m < M; m += CountM) {

                CountM = std::min(M - m, Strides.M);

                //
                // Copy a panel of matrix A to a local packed buffer.
                //

                KernelType::CopyPackA(PanelA, A + m * lda, lda, CountM, CountK,
                    RowSumBuffer);

                MlasGemmU8X8ScaleSumBuffer(RowSumBuffer, CountM, -offb);

                //
                // Step through the rows of the local packed buffer.
                //

                typename KernelType::PackedAType* pa = PanelA;
                int32_t* RowSums = RowSumBuffer;
                size_t RowsRemaining = CountM;

                bool ZeroMode = (k == 0);
                bool PostProcess = (k + CountK == K);

                while (RowsRemaining > 0) {

                    size_t RowsHandled;

                    RowsHandled = KernelType::GemmKernel(pa, b, c, PackedCountK,
                        RowsRemaining, CountN, ldc, RowSums, ColumnSumBuffer,
                        DepthValue, ZeroMode);

                    if (PostProcess && WorkBlock->CIsFloat) {
                        MlasGemmU8X8OutputFloat(WorkBlock, c, n, RowsHandled, CountN);
                    }

                    c += ldc * RowsHandled;
                    pa += KernelType::PackedK * PackedCountK * RowsHandled;
                    RowSums += RowsHandled;
                    RowsRemaining -= RowsHandled;
                }
            }
        }

        A += CountK;
        PackedB = (const uint8_t*)PackedB + AlignedN * CountK;
    }
}

#ifdef MLAS_SSE2_INTRINSICS

void
MlasGemmU8X8CopyPackASse(
    int16_t* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer
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

    RowSumBuffer - Supplies the address of the buffer to receive the sums of
        the elements along each of the rows.

Return Value:

    None.

--*/
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

            __m128i Bytes = _mm_loadl_epi64((__m128i*)&a[0]);
            __m128i Words = _mm_unpacklo_epi8(Bytes, ZeroVector);

            ReductionVector = _mm_add_epi16(ReductionVector, Words);

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
    int32_t* ColumnSumBuffer,
    bool BIsSigned
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

    ColumnSumBuffer - Supplies the address of the buffer to receive the sums of
        the elements along each of the columns.

    BIsSigned - Supplies true if the source matrix is signed data, else false
        if the source matrix is unsigned data.

Return Value:

    None.

--*/
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
        // Reduce the partial accumulators.
        //

        ColumnSums[0] = _mm_madd_epi16(ColumnSums[0], OnesWordBroadcast);
        ColumnSums[1] = _mm_madd_epi16(ColumnSums[1], OnesWordBroadcast);

        _mm_storeu_si128((__m128i*)&ColumnSumBuffer[0], ColumnSums[0]);
        _mm_storeu_si128((__m128i*)&ColumnSumBuffer[4], ColumnSums[1]);
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

        ColumnSums[0] = _mm_madd_epi16(ColumnSums[0], OnesWordBroadcast);
        ColumnSums[1] = _mm_madd_epi16(ColumnSums[1], OnesWordBroadcast);

        _mm_storeu_si128((__m128i*)&ColumnSumBuffer[0], ColumnSums[0]);
        _mm_storeu_si128((__m128i*)&ColumnSumBuffer[4], ColumnSums[1]);
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
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
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

    RowSumBuffer - Supplies the sum of each row from matrix A multiplied by the
        zero point offset of matrix B. These values are accumulated into every
        row of matrix C.

    ColumnSumBuffer - Supplies the sum of each column from matrix B multiplied
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
        Accumulators[0] = _mm_add_epi32(Accumulators[0], _mm_set1_epi32(RowSumBuffer[0]));
        Accumulators[1] = Accumulators[0];
        Accumulators[0] = _mm_add_epi32(Accumulators[0], _mm_loadu_si128((__m128i*)&ColumnSumBuffer[0]));
        Accumulators[1] = _mm_add_epi32(Accumulators[1], _mm_loadu_si128((__m128i*)&ColumnSumBuffer[4]));
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

struct MLAS_GEMM_U8X8_KERNEL_SSE
{
    typedef int16_t PackedAType;
    typedef int16_t PackedBType;
    typedef int8_t OffsetBType;

    static constexpr size_t PackedK = 2;
    static constexpr MLAS_GEMM_U8X8_STRIDES Strides{12, 128, 128};

    MLAS_FORCEINLINE
    static
    bool
    TryGemvKernel(
        const uint8_t* A,
        const uint8_t* B,
        size_t ldb,
        int32_t* C,
        size_t CountK,
        size_t CountN,
        bool BIsSigned
        )
    {
        MLAS_UNREFERENCED_PARAMETER(A);
        MLAS_UNREFERENCED_PARAMETER(B);
        MLAS_UNREFERENCED_PARAMETER(ldb);
        MLAS_UNREFERENCED_PARAMETER(C);
        MLAS_UNREFERENCED_PARAMETER(CountK);
        MLAS_UNREFERENCED_PARAMETER(CountN);
        MLAS_UNREFERENCED_PARAMETER(BIsSigned);

        return false;
    }

    MLAS_FORCEINLINE
    static
    void
    CopyPackA(
        PackedAType* D,
        const uint8_t* A,
        size_t lda,
        size_t CountM,
        size_t CountK,
        int32_t* RowSumBuffer
        )
    {
        MlasGemmU8X8CopyPackASse(D, A, lda, CountM, CountK, RowSumBuffer);
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
        int32_t* ColumnSumBuffer,
        bool BIsSigned
        )
    {
        MlasGemmU8X8CopyPackBSse(D, B, ldb, CountN, CountK, ColumnSumBuffer,
            BIsSigned);
    }

    MLAS_FORCEINLINE
    static
    size_t
    GemmKernel(
        const PackedAType* A,
        const PackedBType* B,
        int32_t* C,
        size_t PackedCountK,
        size_t CountM,
        size_t CountN,
        size_t ldc,
        const int32_t* RowSumBuffer,
        const int32_t* ColumnSumBuffer,
        int32_t DepthValue,
        bool ZeroMode
        )
    {
        MLAS_UNREFERENCED_PARAMETER(CountM);
        MLAS_UNREFERENCED_PARAMETER(ldc);

        MlasGemmU8X8KernelSse(A, B, C, PackedCountK, CountN, RowSumBuffer,
            ColumnSumBuffer, DepthValue, ZeroMode);

        return 1;
    }
};

constexpr size_t MLAS_GEMM_U8X8_KERNEL_SSE::PackedK;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8X8_KERNEL_SSE::Strides;

template
void
MLASCALL
MlasGemmU8X8Operation<MLAS_GEMM_U8X8_KERNEL_SSE>(
    const MLAS_GEMM_U8X8_WORK_BLOCK* WorkBlock
    );

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
        int32_t* RowSumBuffer
        );

    void
    MLASCALL
    MlasGemmU8S8CopyPackBAvx2(
        uint8_t* D,
        const uint8_t* B,
        size_t ldb,
        size_t CountN,
        size_t CountK,
        int32_t* ColumnSumBuffer,
        bool BIsSigned
        );

    void
    MLASCALL
    MlasGemmU8U8CopyPackAAvx2(
        int16_t* D,
        const uint8_t* A,
        size_t lda,
        size_t CountM,
        size_t CountK,
        int32_t* RowSumBuffer
        );

    void
    MLASCALL
    MlasGemmU8U8CopyPackBAvx2(
        uint8_t* D,
        const uint8_t* B,
        size_t ldb,
        size_t CountN,
        size_t CountK,
        int32_t* ColumnSumBuffer
        );
}

struct MLAS_GEMM_U8S8_KERNEL_AVX2
{
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef int8_t OffsetBType;

    static constexpr size_t PackedK = 4;
    static constexpr MLAS_GEMM_U8X8_STRIDES Strides{24, 256, 128};
    static constexpr MLAS_GEMM_U8X8_STRIDES PackedStrides{48, 256, 384};

    MLAS_FORCEINLINE
    static
    bool
    TryGemvKernel(
        const uint8_t* A,
        const uint8_t* B,
        size_t ldb,
        int32_t* C,
        size_t CountK,
        size_t CountN,
        bool BIsSigned
        )
    {
        if (BIsSigned) {
            MlasPlatform.GemvU8S8Kernel(A, B, C, CountK, CountN, ldb);
            return true;
        }

        return false;
    }

    MLAS_FORCEINLINE
    static
    void
    CopyPackA(
        PackedAType* D,
        const uint8_t* A,
        size_t lda,
        size_t CountM,
        size_t CountK,
        int32_t* RowSumBuffer
        )
    {
        MlasGemmU8S8CopyPackAAvx2(D, A, lda, CountM, CountK, RowSumBuffer);
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
        int32_t* ColumnSumBuffer,
        bool BIsSigned
        )
    {
        MlasGemmU8S8CopyPackBAvx2(D, B, ldb, CountN, CountK, ColumnSumBuffer,
            BIsSigned);
    }

    MLAS_FORCEINLINE
    static
    size_t
    GemmKernel(
        const PackedAType* A,
        const PackedBType* B,
        int32_t* C,
        size_t PackedCountK,
        size_t CountM,
        size_t CountN,
        size_t ldc,
        const int32_t* RowSumBuffer,
        const int32_t* ColumnSumBuffer,
        int32_t DepthValue,
        bool ZeroMode
        )
    {
        return MlasPlatform.GemmU8S8Kernel(A, B, C, PackedCountK, CountM, CountN,
            ldc, RowSumBuffer, ColumnSumBuffer, DepthValue, ZeroMode);
    }
};

constexpr size_t MLAS_GEMM_U8S8_KERNEL_AVX2::PackedK;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8S8_KERNEL_AVX2::Strides;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8S8_KERNEL_AVX2::PackedStrides;

template
void
MlasGemmU8X8Operation<MLAS_GEMM_U8S8_KERNEL_AVX2>(
    const MLAS_GEMM_U8X8_WORK_BLOCK* WorkBlock
    );

template
void
MlasGemmU8X8PackedOperation<MLAS_GEMM_U8S8_KERNEL_AVX2>(
    const MLAS_GEMM_U8X8_WORK_BLOCK* WorkBlock
    );

struct MLAS_GEMM_U8U8_KERNEL_AVX2
{
    typedef int16_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint8_t OffsetBType;

    static constexpr size_t PackedK = 2;
    static constexpr MLAS_GEMM_U8X8_STRIDES Strides{24, 256, 128};
    static constexpr MLAS_GEMM_U8X8_STRIDES PackedStrides{48, 256, 384};

    MLAS_FORCEINLINE
    static
    bool
    TryGemvKernel(
        const uint8_t* A,
        const uint8_t* B,
        size_t ldb,
        int32_t* C,
        size_t CountK,
        size_t CountN,
        bool BIsSigned
        )
    {
        MLAS_UNREFERENCED_PARAMETER(A);
        MLAS_UNREFERENCED_PARAMETER(B);
        MLAS_UNREFERENCED_PARAMETER(ldb);
        MLAS_UNREFERENCED_PARAMETER(C);
        MLAS_UNREFERENCED_PARAMETER(CountK);
        MLAS_UNREFERENCED_PARAMETER(CountN);
        MLAS_UNREFERENCED_PARAMETER(BIsSigned);

        return false;
    }

    MLAS_FORCEINLINE
    static
    void
    CopyPackA(
        PackedAType* D,
        const uint8_t* A,
        size_t lda,
        size_t CountM,
        size_t CountK,
        int32_t* RowSumBuffer
        )
    {
        MlasGemmU8U8CopyPackAAvx2(D, A, lda, CountM, CountK, RowSumBuffer);
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
        int32_t* ColumnSumBuffer,
        bool BIsSigned
        )
    {
        MLAS_UNREFERENCED_PARAMETER(BIsSigned);

        MlasGemmU8U8CopyPackBAvx2(D, B, ldb, CountN, CountK, ColumnSumBuffer);
    }

    MLAS_FORCEINLINE
    static
    size_t
    GemmKernel(
        const PackedAType* A,
        const PackedBType* B,
        int32_t* C,
        size_t PackedCountK,
        size_t CountM,
        size_t CountN,
        size_t ldc,
        const int32_t* RowSumBuffer,
        const int32_t* ColumnSumBuffer,
        int32_t DepthValue,
        bool ZeroMode
        )
    {
        return MlasPlatform.GemmU8U8Kernel(A, B, C, PackedCountK, CountM, CountN,
            ldc, RowSumBuffer, ColumnSumBuffer, DepthValue, ZeroMode);
    }
};

constexpr size_t MLAS_GEMM_U8U8_KERNEL_AVX2::PackedK;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8U8_KERNEL_AVX2::Strides;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8U8_KERNEL_AVX2::PackedStrides;

template
void
MLASCALL
MlasGemmU8X8Operation<MLAS_GEMM_U8U8_KERNEL_AVX2>(
    const MLAS_GEMM_U8X8_WORK_BLOCK* WorkBlock
    );

template
void
MlasGemmU8X8PackedOperation<MLAS_GEMM_U8U8_KERNEL_AVX2>(
    const MLAS_GEMM_U8X8_WORK_BLOCK* WorkBlock
    );

#endif

#ifdef MLAS_NEON64_INTRINSICS

//
// Define the prototypes of the NEON routines written in assembly.
//

extern "C"
size_t
MLASCALL
MlasGemmU8X8KernelNeon(
    const uint8_t* A,
    const uint8_t* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumVector,
    const int32_t* ColumnSumVector,
    int32_t DepthValue,
    bool ZeroMode
    );

void
MLASCALL
MlasGemmU8X8CopyPackANeon(
    uint8_t* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer
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

    RowSumBuffer - Supplies the address of the buffer to receive the sums of
        the elements along each of the rows.

Return Value:

    None.

--*/
{
    uint8_t PaddedMatrixAData[16];

    //
    // Process four rows of matrix A in a loop.
    //
    // The buffer is packed as a series of 16 byte vectors where four rows are
    // interleaved with the following pattern:
    //
    //      [ A0 A1 A2 A3 B0 B1 B2 B3 C0 C1 C2 C3 D0 D1 D2 D3 ]
    //      [ A4 A5 A6 A7 B4 B5 B6 B7 C4 C5 C6 C7 D4 D5 D6 D7 ]
    //
    // This pattern is repeated (CountK / 4) times.
    //
    // If CountK is not aligned to a multiple of four, then the vector is padded
    // with zeroes.
    //

    while (CountM >= 4) {

        const uint8_t* a0 = A;
        const uint8_t* a1 = a0 + lda;
        const uint8_t* a2 = a1 + lda;
        const uint8_t* a3 = a2 + lda;

        size_t k = CountK;
        uint32x4_t RowSums = vmovq_n_u32(0);

        while (k >= 16) {

            uint32x4_t v0 = vld1q_u32(reinterpret_cast<const uint32_t*>(a0));
            a0 += 16;
            uint32x4_t v1 = vld1q_u32(reinterpret_cast<const uint32_t*>(a1));
            a1 += 16;
            uint32x4_t v2 = vld1q_u32(reinterpret_cast<const uint32_t*>(a2));
            a2 += 16;
            uint32x4_t v3 = vld1q_u32(reinterpret_cast<const uint32_t*>(a3));
            a3 += 16;

            uint32x4_t z0 = vzip1q_u32(v0, v2);
            uint32x4_t z1 = vzip2q_u32(v0, v2);
            uint32x4_t z2 = vzip1q_u32(v1, v3);
            uint32x4_t z3 = vzip2q_u32(v1, v3);
            v0 = vzip1q_u32(z0, z2);
            v1 = vzip2q_u32(z0, z2);
            v2 = vzip1q_u32(z1, z3);
            v3 = vzip2q_u32(z1, z3);

            vst1q_u8(&D[0], vreinterpretq_u8_u32(v0));
            vst1q_u8(&D[16], vreinterpretq_u8_u32(v1));
            vst1q_u8(&D[32], vreinterpretq_u8_u32(v2));
            vst1q_u8(&D[48], vreinterpretq_u8_u32(v3));

            RowSums = vpadalq_u16(RowSums, vpaddlq_u8(vreinterpretq_u8_u32(v0)));
            RowSums = vpadalq_u16(RowSums, vpaddlq_u8(vreinterpretq_u8_u32(v1)));
            RowSums = vpadalq_u16(RowSums, vpaddlq_u8(vreinterpretq_u8_u32(v2)));
            RowSums = vpadalq_u16(RowSums, vpaddlq_u8(vreinterpretq_u8_u32(v3)));

            D += 64;
            k -= 16;
        }

        uint32x4_t GatherVector = vmovq_n_u32(0);

        while (k >= 4) {

            GatherVector = vld1q_lane_u32(reinterpret_cast<const uint32_t*>(a0), GatherVector, 0);
            a0 += 4;
            GatherVector = vld1q_lane_u32(reinterpret_cast<const uint32_t*>(a1), GatherVector, 1);
            a1 += 4;
            GatherVector = vld1q_lane_u32(reinterpret_cast<const uint32_t*>(a2), GatherVector, 2);
            a2 += 4;
            GatherVector = vld1q_lane_u32(reinterpret_cast<const uint32_t*>(a3), GatherVector, 3);
            a3 += 4;

            uint8x16_t PackedVector = vreinterpretq_u8_u32(GatherVector);
            vst1q_u8(D, PackedVector);

            RowSums = vpadalq_u16(RowSums, vpaddlq_u8(PackedVector));

            D += 16;
            k -= 4;
        }

        if (k > 0) {

            //
            // Copy the remaining bytes to the zero padded stack buffer.
            //

            uint8_t* d = PaddedMatrixAData;

            vst1q_u8(PaddedMatrixAData, vmovq_n_u8(0));

            while (k > 0) {

                d[0] = *a0++;
                d[4] = *a1++;
                d[8] = *a2++;
                d[12] = *a3++;

                d += 1;
                k -= 1;
            }

            uint8x16_t PackedVector = vld1q_u8(PaddedMatrixAData);
            vst1q_u8(D, PackedVector);

            RowSums = vpadalq_u16(RowSums, vpaddlq_u8(PackedVector));

            D += 16;
        }

        vst1q_s32(RowSumBuffer, vreinterpretq_s32_u32(RowSums));
        RowSumBuffer += 4;

        A = A + lda * 4;
        CountM -= 4;
    }

    //
    // Process two rows of matrix A.
    //
    // The buffer is packed as a series of 8 byte vectors where two rows are
    // interleaved with the following pattern:
    //
    //      [ A0 A1 A2 A3 B0 B1 B2 B3 ]
    //      [ A4 A5 A6 A7 B4 B5 B6 B7 ]
    //
    // This pattern is repeated (CountK / 4) times.
    //
    // If CountK is not aligned to a multiple of four, then the vector is padded
    // with zeroes.
    //

    if ((CountM & 2) != 0) {

        const uint8_t* a0 = A;
        const uint8_t* a1 = a0 + lda;

        size_t k = CountK;
        uint32x2_t RowSums = vmov_n_u32(0);
        uint32x2_t GatherVector = vmov_n_u32(0);

        while (k >= 4) {

            GatherVector = vld1_lane_u32(reinterpret_cast<const uint32_t*>(a0), GatherVector, 0);
            a0 += 4;
            GatherVector = vld1_lane_u32(reinterpret_cast<const uint32_t*>(a1), GatherVector, 1);
            a1 += 4;

            uint8x8_t PackedVector = vreinterpret_u8_u32(GatherVector);
            vst1_u8(D, PackedVector);

            RowSums = vpadal_u16(RowSums, vpaddl_u8(PackedVector));

            D += 8;
            k -= 4;
        }

        if (k > 0) {

            //
            // Copy the remaining bytes to the zero padded stack buffer.
            //

            uint8_t* d = PaddedMatrixAData;

            vst1q_u8(PaddedMatrixAData, vmovq_n_u8(0));

            while (k > 0) {

                d[0] = *a0++;
                d[4] = *a1++;

                d += 1;
                k -= 1;
            }

            uint8x8_t PackedVector = vld1_u8(PaddedMatrixAData);
            vst1_u8(D, PackedVector);

            RowSums = vpadal_u16(RowSums, vpaddl_u8(PackedVector));

            D += 8;
        }

        vst1_s32(RowSumBuffer, vreinterpret_s32_u32(RowSums));
        RowSumBuffer += 2;

        A = A + lda * 2;
        CountM -= 2;
    }

    //
    // Process one row of matrix A.
    //
    // The buffer is packed as a series of 4 byte with the following pattern:
    //
    //      [ A0 A1 A2 A3 ]
    //      [ A4 A5 A6 A7 ]
    //
    // This pattern is repeated (CountK / 4) times.
    //
    // If CountK is not aligned to a multiple of four, then the vector is padded
    // with zeroes.
    //

    if ((CountM & 1) != 0) {

        const uint8_t* a = A;
        size_t k = CountK;
        uint32x4_t RowSums = vmovq_n_u32(0);

        while (k >= 16) {

            uint8x16_t v = vld1q_u8(a);
            a += 16;

            vst1q_u8(D, v);

            RowSums = vpadalq_u16(RowSums, vpaddlq_u8(v));

            D += 16;
            k -= 16;
        }

        if (k > 0) {

            //
            // Copy the remaining bytes to the zero padded stack buffer.
            //

            vst1q_u8(PaddedMatrixAData, vmovq_n_u8(0));

            for (size_t kk = 0; kk < k; kk++) {
                PaddedMatrixAData[kk] = a[kk];
            }

            uint8x16_t v = vld1q_u8(PaddedMatrixAData);
            vst1q_u8(D, v);

            RowSums = vpadalq_u16(RowSums, vpaddlq_u8(v));
        }

#if defined(_M_ARM64)
        // N.B. The workaround of defining a local vaddvq_u32 doesn't work here
        // as VS2019 added new intrinsics to make the operation work. Also, not
        // all build environments using VS2019 have the up-to-date arm64_neon.h,
        // so fallback to pairwise addition.
        RowSums = vpaddq_u32(RowSums, RowSums);
        RowSums = vpaddq_u32(RowSums, RowSums);
        vst1q_lane_u32(reinterpret_cast<int32_t*>(RowSumBuffer), RowSums, 0);
#else
        *RowSumBuffer = int32_t(vaddvq_u32(RowSums));
#endif
    }
}

MLAS_FORCEINLINE
void
MlasGemmU8X8CopyPackBProcessNeon(
    uint8_t* D,
    const uint8_t* B,
    uint8x8_t BitFlipVector,
    uint32x4_t ColumnSums[2]
    )
{
    uint8x8_t BytesRow = veor_u8(vld1_u8(B), BitFlipVector);
    vst1_u8(D, BytesRow);

    uint16x8_t WordsRow = vmovl_u8(BytesRow);
    ColumnSums[0] = vaddq_u32(ColumnSums[0], vmovl_u16(vget_low_u16(WordsRow)));
    ColumnSums[1] = vaddq_u32(ColumnSums[1], vmovl_high_u16(WordsRow));
}

void
MLASCALL
MlasGemmU8X8CopyPackBNeon(
    uint8_t* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
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

    ColumnSumBuffer - Supplies the address of the buffer to receive the sums of
        the elements along each of the columns.

    BIsSigned - Supplies true if the source matrix is signed data, else false
        if the source matrix is unsigned data.

Return Value:

    None.

--*/
{
    const uint8x8_t BitFlipVector = vdup_n_u8(BIsSigned ? 0x80 : 0);
    const uint8x8_t ZeroVector = vmov_n_u8(0);
    const size_t AlignedCountK = (CountK + 3) & ~3;

    //
    // Process 8 columns of matrix B in a loop.
    //
    // Copy columns from matrix B to the packed buffer. Signed buffers are
    // converted to unsigned buffers in order to share a common kernel.
    //
    // If CountK is not aligned to a multiple of four, then the packed buffer
    // is padded with zero vectors.
    //
    // If CountN is not aligned to a multiple of four, then the extra columns
    // are padded with zeroes.
    //

    while (CountN >= 8) {

        const uint8_t* b = B;
        uint32x4_t ColumnSums[2];

        ColumnSums[0] = vmovq_n_u32(0);
        ColumnSums[1] = vmovq_n_u32(0);

        for (size_t k = CountK; k > 0; k--) {

            MlasGemmU8X8CopyPackBProcessNeon(D, b, BitFlipVector, ColumnSums);

            b += ldb;
            D += 8;
        }

        for (size_t k = CountK; k < AlignedCountK; k++) {
            vst1_u8(D, ZeroVector);
            D += 8;
        }

        vst1q_s32(&ColumnSumBuffer[0], vreinterpretq_s32_u32(ColumnSums[0]));
        vst1q_s32(&ColumnSumBuffer[4], vreinterpretq_s32_u32(ColumnSums[1]));
        ColumnSumBuffer += 8;

        B += 8;
        CountN -= 8;
    }

    //
    // Process the remaining columns of matrix B.
    //

    if (CountN > 0) {

        const uint8_t* b = B;
        uint8_t PaddedMatrixBData[8];
        uint32x4_t ColumnSums[2];

        vst1_u8(PaddedMatrixBData, ZeroVector);

        ColumnSums[0] = vmovq_n_u32(0);
        ColumnSums[1] = vmovq_n_u32(0);

        for (size_t k = CountK; k > 0; k--) {

            for (size_t n = 0; n < CountN; n++) {
                PaddedMatrixBData[n] = b[n];
            }

            MlasGemmU8X8CopyPackBProcessNeon(D, PaddedMatrixBData, BitFlipVector, ColumnSums);

            b += ldb;
            D += 8;
        }

        for (size_t k = CountK; k < AlignedCountK; k++) {
            vst1_u8(D, ZeroVector);
            D += 8;
        }

        vst1q_s32(&ColumnSumBuffer[0], vreinterpretq_s32_u32(ColumnSums[0]));
        vst1q_s32(&ColumnSumBuffer[4], vreinterpretq_s32_u32(ColumnSums[1]));
    }
}

struct MLAS_GEMM_U8X8_KERNEL_NEON
{
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint8_t OffsetBType;

    static constexpr size_t PackedK = 4;
    static constexpr MLAS_GEMM_U8X8_STRIDES Strides{24, 128, 256};
    static constexpr MLAS_GEMM_U8X8_STRIDES PackedStrides{24, 256, 128};

    MLAS_FORCEINLINE
    static
    bool
    TryGemvKernel(
        const uint8_t* A,
        const uint8_t* B,
        size_t ldb,
        int32_t* C,
        size_t CountK,
        size_t CountN,
        bool BIsSigned
        )
    {
        MLAS_UNREFERENCED_PARAMETER(A);
        MLAS_UNREFERENCED_PARAMETER(B);
        MLAS_UNREFERENCED_PARAMETER(ldb);
        MLAS_UNREFERENCED_PARAMETER(C);
        MLAS_UNREFERENCED_PARAMETER(CountK);
        MLAS_UNREFERENCED_PARAMETER(CountN);
        MLAS_UNREFERENCED_PARAMETER(BIsSigned);

        return false;
    }

    MLAS_FORCEINLINE
    static
    void
    CopyPackA(
        PackedAType* D,
        const uint8_t* A,
        size_t lda,
        size_t CountM,
        size_t CountK,
        int32_t* RowSumBuffer
        )
    {
        MlasGemmU8X8CopyPackANeon(D, A, lda, CountM, CountK, RowSumBuffer);
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
        int32_t* ColumnSumBuffer,
        bool BIsSigned
        )
    {
        MlasGemmU8X8CopyPackBNeon(D, B, ldb, CountN, CountK, ColumnSumBuffer,
            BIsSigned);
    }

    MLAS_FORCEINLINE
    static
    size_t
    GemmKernel(
        const PackedAType* A,
        const PackedBType* B,
        int32_t* C,
        size_t PackedCountK,
        size_t CountM,
        size_t CountN,
        size_t ldc,
        const int32_t* RowSumBuffer,
        const int32_t* ColumnSumBuffer,
        int32_t DepthValue,
        bool ZeroMode
        )
    {
        return MlasGemmU8X8KernelNeon(A, B, C, PackedCountK, CountM, CountN, ldc,
            RowSumBuffer, ColumnSumBuffer, DepthValue, ZeroMode);
    }
};

constexpr size_t MLAS_GEMM_U8X8_KERNEL_NEON::PackedK;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8X8_KERNEL_NEON::Strides;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8X8_KERNEL_NEON::PackedStrides;

template
void
MLASCALL
MlasGemmU8X8PackedOperation<MLAS_GEMM_U8X8_KERNEL_NEON>(
    const MLAS_GEMM_U8X8_WORK_BLOCK* WorkBlock
    );

#endif

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
    MLAS_GEMM_U8X8_WORK_BLOCK WorkBlock;

    memcpy(&WorkBlock, Context, sizeof(MLAS_GEMM_U8X8_WORK_BLOCK));

    const int32_t ThreadIdM = ThreadId / WorkBlock.ThreadCountN;
    const int32_t ThreadIdN = ThreadId % WorkBlock.ThreadCountN;

    //
    // Partition the operation along the M dimension.
    //

    MlasPartitionWork(ThreadIdM, WorkBlock.ThreadCountM, WorkBlock.M,
        &WorkBlock.RangeStartM, &WorkBlock.RangeCountM);

    //
    // Partition the operation along the N dimension.
    //

    const size_t BlockedN = (WorkBlock.N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) /
        MLAS_QGEMM_STRIDEN_THREAD_ALIGN;

    MlasPartitionWork(ThreadIdN, WorkBlock.ThreadCountN, BlockedN,
        &WorkBlock.RangeStartN, &WorkBlock.RangeCountN);

    WorkBlock.RangeStartN *= MLAS_QGEMM_STRIDEN_THREAD_ALIGN;
    WorkBlock.RangeCountN *= MLAS_QGEMM_STRIDEN_THREAD_ALIGN;

    WorkBlock.RangeCountN = std::min(WorkBlock.N - WorkBlock.RangeStartN,
        WorkBlock.RangeCountN);

    //
    // Dispatch the partitioned operation.
    //

#if defined(MLAS_TARGET_AMD64)
    PMLAS_GEMM_U8X8_OPERATION GemmU8X8Operation;

    if (WorkBlock.BIsSigned) {
        GemmU8X8Operation = WorkBlock.BIsPacked ?
            MlasPlatform.GemmU8S8PackedOperation : MlasPlatform.GemmU8S8Operation;
    } else {
        GemmU8X8Operation = WorkBlock.BIsPacked ?
            MlasPlatform.GemmU8U8PackedOperation : MlasPlatform.GemmU8U8Operation;
    }

    GemmU8X8Operation(&WorkBlock);
#elif defined(MLAS_SSE2_INTRINSICS)
    MlasGemmU8X8Operation<MLAS_GEMM_U8X8_KERNEL_SSE>(&WorkBlock);
#elif defined(MLAS_NEON64_INTRINSICS)
    if (WorkBlock.BIsPacked) {
        MlasGemmU8X8PackedOperation<MLAS_GEMM_U8X8_KERNEL_NEON>(&WorkBlock);
    } else {
        MlasGemmU8X8Operation<MLAS_GEMM_U8X8_KERNEL_NEON>(&WorkBlock);
    }
#endif
}

void
MlasGemmU8X8Schedule(
    MLAS_GEMM_U8X8_WORK_BLOCK* WorkBlock,
    MLAS_THREADPOOL* ThreadPool
    )
/*++

Routine Description:

    This routine schedules the quantized integer matrix/matrix multiply
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
    bool BIsSigned,
    int32_t* C,
    size_t ldc,
    MLAS_THREADPOOL* ThreadPool
    )
/*++

Routine Description:

    This routine implements the quantized integer matrix/matrix multiply
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

    BIsSigned - Supplies true if matrix B is signed data, else false if matrix
        B is unsigned data.

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
    WorkBlock.B = B;
    WorkBlock.ldb = ldb;
    WorkBlock.C = C;
    WorkBlock.ldc = ldc;
    WorkBlock.offa = offa;
    WorkBlock.offb = offb;
    WorkBlock.BIsSigned = BIsSigned;

    //
    // Schedule the operation across a set of worker threads.
    //

    MlasGemmU8X8Schedule(&WorkBlock, ThreadPool);
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
    bool BIsSigned,
    float* C,
    size_t ldc,
    const float* Scale,
    const float* Bias,
    MLAS_THREADPOOL* ThreadPool
    )
/*++

Routine Description:

    This routine implements the quantized integer matrix/matrix multiply
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

    BIsSigned - Supplies true if matrix B is signed data, else false if matrix
        B is unsigned data.

    C - Supplies the address of matrix C.

    ldc - Supplies the first dimension of matrix C.

    Scale - Supplies the scale multiplier to apply to each element of matrix C.
        Used to scale the integer output of the QGEMM back to a floating point
        number.

    Bias - Supplies the bias vector to apply to element of matrix C. The vector
        is of length N.

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
    WorkBlock.B = B;
    WorkBlock.ldb = ldb;
    WorkBlock.C = (int32_t*)C;
    WorkBlock.ldc = ldc;
    WorkBlock.Scale = Scale;
    WorkBlock.BiasFloat = Bias;
    WorkBlock.offa = offa;
    WorkBlock.offb = offb;
    WorkBlock.BIsSigned = BIsSigned;
    WorkBlock.CIsFloat = true;

    //
    // Schedule the operation across a set of worker threads.
    //

    MlasGemmU8X8Schedule(&WorkBlock, ThreadPool);
}

#if defined(MLAS_TARGET_AMD64) || defined(MLAS_NEON64_INTRINSICS)

void
MLASCALL
MlasGemm(
    size_t M,
    size_t N,
    size_t K,
    const uint8_t* A,
    size_t lda,
    uint8_t offa,
    const void* PackedB,
    uint8_t offb,
    bool BIsSigned,
    int32_t* C,
    size_t ldc,
    MLAS_THREADPOOL* ThreadPool
    )
/*++

Routine Description:

    This routine implements the quantized integer matrix/matrix multiply
    operation (QGEMM).

Arguments:

    M - Supplies the number of rows of matrix A and matrix C.

    N - Supplies the number of columns of matrix B and matrix C.

    K - Supplies the number of columns of matrix A and the number of rows of
        matrix B.

    A - Supplies the address of matrix A.

    lda - Supplies the first dimension of matrix A.

    offa - Supplies the zero point offset of matrix A.

    PackedB - Supplies the address of packed matrix B.

    offb - Supplies the zero point offset of matrix B.

    BIsSigned - Supplies true if matrix B is signed data, else false if matrix
        B is unsigned data.

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
    WorkBlock.B = PackedB;
    WorkBlock.C = C;
    WorkBlock.ldc = ldc;
    WorkBlock.offa = offa;
    WorkBlock.offb = offb;
    WorkBlock.BIsPacked = true;
    WorkBlock.BIsSigned = BIsSigned;

    //
    // Schedule the operation across a set of worker threads.
    //

    MlasGemmU8X8Schedule(&WorkBlock, ThreadPool);
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
    const void* PackedB,
    uint8_t offb,
    bool BIsSigned,
    float* C,
    size_t ldc,
    const float* Scale,
    const float* Bias,
    MLAS_THREADPOOL* ThreadPool
    )
/*++

Routine Description:

    This routine implements the quantized integer matrix/matrix multiply
    operation (QGEMM).

Arguments:

    M - Supplies the number of rows of matrix A and matrix C.

    N - Supplies the number of columns of matrix B and matrix C.

    K - Supplies the number of columns of matrix A and the number of rows of
        matrix B.

    A - Supplies the address of matrix A.

    lda - Supplies the first dimension of matrix A.

    offa - Supplies the zero point offset of matrix A.

    PackedB - Supplies the address of packed matrix B.

    offb - Supplies the zero point offset of matrix B.

    BIsSigned - Supplies true if matrix B is signed data, else false if matrix
        B is unsigned data.

    C - Supplies the address of matrix C.

    ldc - Supplies the first dimension of matrix C.

    Scale - Supplies the scale multiplier to apply to each element of matrix C.
        Used to scale the integer output of the QGEMM back to a floating point
        number.

    Bias - Supplies the bias vector to apply to element of matrix C. The vector
        is of length N.

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
    WorkBlock.B = PackedB;
    WorkBlock.C = (int32_t*)C;
    WorkBlock.ldc = ldc;
    WorkBlock.Scale = Scale;
    WorkBlock.BiasFloat = Bias;
    WorkBlock.offa = offa;
    WorkBlock.offb = offb;
    WorkBlock.BIsPacked = true;
    WorkBlock.BIsSigned = BIsSigned;
    WorkBlock.CIsFloat = true;

    //
    // Schedule the operation across a set of worker threads.
    //

    MlasGemmU8X8Schedule(&WorkBlock, ThreadPool);
}

size_t
MLASCALL
MlasGemmPackBSize(
    size_t N,
    size_t K,
    bool BIsSigned
    )
/*++

Routine Description:

    This routine computes the number of bytes required to pack a matrix with
    the supplied shape and type.

Arguments:

    N - Supplies the number of columns of matrix B.

    K - Supplies the the number of rows of matrix B.

    BIsSigned - Supplies true if matrix B is signed data, else false if matrix
        B is unsigned data.

Return Value:

    Returns the number of bytes required to pack the matrix.

--*/
{
    //
    // Retrieve the packing parameters based on the packed operation function.
    //

    size_t PackedK;

#if defined(MLAS_TARGET_AMD64)
    PMLAS_GEMM_U8X8_OPERATION GemmU8X8Operation = BIsSigned ?
        MlasPlatform.GemmU8S8PackedOperation : MlasPlatform.GemmU8U8PackedOperation;

    if (GemmU8X8Operation == &MlasGemmU8X8PackedOperation<MLAS_GEMM_U8S8_KERNEL_AVX2>) {
        PackedK = MLAS_GEMM_U8S8_KERNEL_AVX2::PackedK;
    } else if (GemmU8X8Operation == &MlasGemmU8X8PackedOperation<MLAS_GEMM_U8U8_KERNEL_AVX2>) {
        PackedK = MLAS_GEMM_U8U8_KERNEL_AVX2::PackedK;
    } else {
        return 0;
    }
#elif defined(MLAS_NEON64_INTRINSICS)
    MLAS_UNREFERENCED_PARAMETER(BIsSigned);

    PackedK = MLAS_GEMM_U8X8_KERNEL_NEON::PackedK;
#else
#error Unknown architecture.
#endif

    //
    // Compute the number of bytes required to hold the packed buffer.
    //

    const size_t AlignedN =
        (N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1);
    const size_t AlignedK = (K + PackedK - 1) & ~(PackedK - 1);

    const size_t BytesRequired =
        (AlignedN * sizeof(int32_t)) + (AlignedN * AlignedK * sizeof(uint8_t));
    const size_t BufferAlignment = MlasGetPreferredBufferAlignment();
    const size_t AlignedBytesRequired = (BytesRequired + BufferAlignment - 1) &
        ~(BufferAlignment - 1);

    return AlignedBytesRequired;
}

void
MLASCALL
MlasGemmPackB(
    size_t N,
    size_t K,
    const uint8_t* B,
    size_t ldb,
    bool BIsSigned,
    void* PackedB
    )
/*++

Routine Description:

    This routine packs the supplied matrix B to the supplied packed matrix B
    buffer. The size of the packed buffer was obtained from MlasGemmPackBSize.

Arguments:

    N - Supplies the number of columns of matrix B.

    K - Supplies the the number of rows of matrix B.

    B - Supplies the address of matrix B.

    ldb - Supplies the first dimension of matrix B.

    BIsSigned - Supplies true if matrix B is signed data, else false if matrix
        B is unsigned data.

    PackedB - Supplies the address of packed matrix B.

Return Value:

    None.

--*/
{
    //
    // Retrieve the packing parameters based on the packed operation function.
    //

    size_t PackedK;
    size_t StrideK;

#if defined(MLAS_TARGET_AMD64)
    PMLAS_GEMM_U8X8_OPERATION GemmU8X8Operation = BIsSigned ?
        MlasPlatform.GemmU8S8PackedOperation : MlasPlatform.GemmU8U8PackedOperation;

    if (GemmU8X8Operation == &MlasGemmU8X8PackedOperation<MLAS_GEMM_U8S8_KERNEL_AVX2>) {
        PackedK = MLAS_GEMM_U8S8_KERNEL_AVX2::PackedK;
        StrideK = MLAS_GEMM_U8S8_KERNEL_AVX2::PackedStrides.K;
    } else if (GemmU8X8Operation == &MlasGemmU8X8PackedOperation<MLAS_GEMM_U8U8_KERNEL_AVX2>) {
        PackedK = MLAS_GEMM_U8U8_KERNEL_AVX2::PackedK;
        StrideK = MLAS_GEMM_U8U8_KERNEL_AVX2::PackedStrides.K;
    } else {
#ifdef MLAS_NO_EXCEPTION
        abort();
#else
        throw std::runtime_error("packing unavailable");
#endif
    }
#elif defined(MLAS_NEON64_INTRINSICS)
    PackedK = MLAS_GEMM_U8X8_KERNEL_NEON::PackedK;
    StrideK = MLAS_GEMM_U8X8_KERNEL_NEON::PackedStrides.K;
#else
#error Unknown architecture.
#endif

    //
    // Reserve and initialize storage for the column sum buffer to hold the sums
    // of the elements along each of the columns.
    //

    const size_t AlignedN =
        (N + MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_QGEMM_STRIDEN_THREAD_ALIGN - 1);

    int32_t* PackedColumnSumBuffer = (int32_t*)PackedB;
    std::fill_n(PackedColumnSumBuffer, AlignedN, 0);
    PackedB = PackedColumnSumBuffer + AlignedN;

    //
    // Step through each slice of matrix B along the K dimension.
    //

    size_t CountK;

    for (size_t k = 0; k < K; k += CountK) {

        CountK = std::min(K - k, StrideK);

        //
        // Step through each slice of matrix B along the N dimension.
        //

        const size_t AlignedK = (CountK + PackedK - 1) & ~(PackedK - 1);
        uint8_t* pb = (uint8_t*)PackedB;
        size_t CountN;

        for (size_t n = 0; n < N; n += CountN) {

            constexpr size_t BatchedN = 128;
            MLAS_DECLSPEC_ALIGN(int32_t ColumnSumBuffer[BatchedN], 64);

            CountN = std::min(N - n, BatchedN);

#if defined(MLAS_TARGET_AMD64)
            if (GemmU8X8Operation == &MlasGemmU8X8PackedOperation<MLAS_GEMM_U8S8_KERNEL_AVX2>) {
                MLAS_GEMM_U8S8_KERNEL_AVX2::CopyPackB(pb, B + n, ldb, CountN, CountK, ColumnSumBuffer, BIsSigned);
            } else {
                MLAS_GEMM_U8U8_KERNEL_AVX2::CopyPackB(pb, B + n, ldb, CountN, CountK, ColumnSumBuffer, BIsSigned);
            }
#elif defined(MLAS_NEON64_INTRINSICS)
            MLAS_GEMM_U8X8_KERNEL_NEON::CopyPackB(pb, B + n, ldb, CountN, CountK, ColumnSumBuffer, BIsSigned);
#else
#error Unknown architecture.
#endif

            //
            // Accumulate this batch of the column sum buffer into the packed
            // buffer accumulators.
            //

            for (size_t nn = 0; nn < CountN; nn++) {
                PackedColumnSumBuffer[n + nn] += ColumnSumBuffer[nn];
            }

            pb += CountN * AlignedK;
        }

        PackedB = (uint8_t*)PackedB + AlignedN * AlignedK;
        B += ldb * CountK;
    }
}

#endif
