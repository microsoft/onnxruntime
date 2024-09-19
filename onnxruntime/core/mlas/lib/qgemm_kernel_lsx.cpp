/*++

Copyright (C) 2023 Loongson Technology Corporation Limited.

Licensed under the MIT License.

Module Name:

    qgemm_kernel_lsx.cpp

Abstract:

    This module implements QGEMM kernels for LSX.

--*/

#include "mlasi.h"
#include "qgemm.h"
#include <lsxintrin.h>

struct MLAS_GEMM_U8X8_KERNEL_LSX
{
    typedef int16_t PackedAType;
    typedef int16_t PackedBType;
    typedef uint8_t OffsetAType;
    typedef int8_t OffsetBType;

    static constexpr size_t PackedK = 2;
    static constexpr MLAS_GEMM_QUANT_STRIDES Strides{ 12, 128, 128 };
    static constexpr MLAS_GEMM_QUANT_STRIDES PackedStrides{0, 0, 0};
};

constexpr size_t MLAS_GEMM_U8X8_KERNEL_LSX::PackedK;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_U8X8_KERNEL_LSX::Strides;

template<>
MLAS_FORCEINLINE constexpr
int32_t
MlasGemmQuantFixupZeroPointB<MLAS_GEMM_U8X8_KERNEL_LSX>(
    int32_t ZeroPointB,
    bool BIsSigned
    )
{
    if (!BIsSigned) {
        ZeroPointB = MLAS_GEMM_U8X8_KERNEL_LSX::OffsetBType(ZeroPointB ^ 0x80);
    }

    return ZeroPointB;
}

template<>
void
MlasGemmQuantCopyPackA<MLAS_GEMM_U8X8_KERNEL_LSX>(
    MLAS_GEMM_U8X8_KERNEL_LSX::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer,
    bool AIsSigned
    )
{
    MLAS_UNREFERENCED_PARAMETER(AIsSigned);
    const __m128i ZeroVector = __lsx_vrepli_d(0);
    uint16_t val = 1;
    const __m128i OnesWordBroadcast = __lsx_vreplgr2vr_h(val);
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

            __m128i Bytes = __lsx_vld((const __m128i*) & a[0], 0);
            __lsx_vinsgr2vr_d(Bytes, 0, 1);
            __m128i Words = __lsx_vilvl_b(ZeroVector, Bytes);

            ReductionVector = __lsx_vadd_h(ReductionVector, Words);

            __lsx_vst(Words, (__m128i*) & D[0], 0);

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

            __m128i Bytes = __lsx_vld((__m128i*)PaddedMatrixAData, 0);
            __lsx_vinsgr2vr_d(Bytes, 0, 1); 
            __m128i Words = __lsx_vilvl_b(ZeroVector, Bytes);

            ReductionVector = __lsx_vadd_h(ReductionVector, Words);

            //
            // Copy pairs of 16-bit values from the vector to the packed
            // buffer and rotate the vector for the next iteration.
            //

            for (size_t pairs = (k + 1) / 2; pairs > 0; pairs--) {
                __lsx_vstelm_w(Words, (int32_t*)D, 0 , 0);
                D += 2;
                Words = __lsx_vshuf4i_w(Words, 0x39); //(0, 3, 2, 1)
            }
        }

        //
        // Reduce the partial accumulators.
        //
        __m128i tmp1 = ZeroVector, tmp2 = ZeroVector;
        tmp1 = __lsx_vmaddwev_w_h(tmp1, ReductionVector, OnesWordBroadcast);
        tmp2 = __lsx_vmaddwod_w_h(tmp2, ReductionVector, OnesWordBroadcast);
        ReductionVector = __lsx_vadd_w(tmp1, tmp2);
        ReductionVector = __lsx_vadd_w(ReductionVector,
                                        __lsx_vshuf4i_w(ReductionVector, 0xee));
        ReductionVector = __lsx_vadd_w(ReductionVector,
                                        __lsx_vshuf4i_w(ReductionVector, 0x11));

        __lsx_vstelm_w(ReductionVector, RowSumBuffer++, 0 , 0);

        A += lda;
        CountM -= 1;
    }
}

MLAS_FORCEINLINE
void
MlasGemmU8X8CopyPackBProcessLSX(
    MLAS_GEMM_U8X8_KERNEL_LSX::PackedBType* D,
    __m128i BytesRow0,
    __m128i BytesRow1,
    __m128i BitFlipVector,
    __m128i ColumnSums[2]
)
{
    __m128i BytesInterleaved = __lsx_vilvl_b(BytesRow1, BytesRow0);

    BytesInterleaved = __lsx_vxor_v(BytesInterleaved, BitFlipVector);

    __m128i WordsInterleaved0 = __lsx_vsrai_h(__lsx_vilvl_b(BytesInterleaved, BytesInterleaved), 8);
    __m128i WordsInterleaved1 = __lsx_vsrai_h(__lsx_vilvh_b(BytesInterleaved, BytesInterleaved), 8);

    ColumnSums[0] = __lsx_vadd_h(ColumnSums[0], WordsInterleaved0);
    ColumnSums[1] = __lsx_vadd_h(ColumnSums[1], WordsInterleaved1);

    __lsx_vst(WordsInterleaved0, (__m128i*) & D[0], 0);
    __lsx_vst(WordsInterleaved1, (__m128i*) & D[8], 0);
}

template<>
void
MlasGemmQuantCopyPackB<MLAS_GEMM_U8X8_KERNEL_LSX>(
    MLAS_GEMM_U8X8_KERNEL_LSX::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    uint16_t val = 1;
    const __m128i OnesWordBroadcast = __lsx_vreplgr2vr_h(val);
    const __m128i BitFlipVector = __lsx_vreplgr2vr_w(BIsSigned ? 0 : 0x80808080);

    //
    // Process 8 columns of matrix B in a loop.
    //

    while (CountN >= 8) {

        const uint8_t* b = B;
        size_t k = CountK;
        __m128i ColumnSums[2];

        ColumnSums[0] = __lsx_vldi(0);
        ColumnSums[1] = __lsx_vldi(0);

        //
        // Interleave rows of matrix B and write to the packed buffer.
        //
        // These values are also zero-extended and accumulated into an
        // intermediate per-column accumulator. CountK cannot be greater than
        // 128 to avoid overflowing these signed 16-bit accumulators.
        //

        while (k >= MLAS_GEMM_U8X8_KERNEL_LSX::PackedK) {

            __m128i BytesRow0 = __lsx_vld((const __m128i*) & b[0], 0);
            __lsx_vinsgr2vr_d(BytesRow0, 0, 1);
            __m128i BytesRow1 = __lsx_vld((const __m128i*) & b[ldb], 0);
            __lsx_vinsgr2vr_d(BytesRow1, 0, 1);

            MlasGemmU8X8CopyPackBProcessLSX(D, BytesRow0, BytesRow1, BitFlipVector, ColumnSums);

            b += ldb * 2;
            D += 16;
            k -= 2;
        }

        if (k > 0) {

            __m128i BytesRow0 = __lsx_vld((const __m128i*) & b[0], 0);
            __lsx_vinsgr2vr_d(BytesRow0, 0, 1);

            MlasGemmU8X8CopyPackBProcessLSX(D, BytesRow0, BitFlipVector, BitFlipVector, ColumnSums);

            D += 16;
        }

        __m128i tmp1, tmp2;
        tmp1 = tmp2 = __lsx_vldi(0);
        tmp1 = __lsx_vmaddwev_w_h(tmp1, ColumnSums[0], OnesWordBroadcast);
        tmp2 = __lsx_vmaddwod_w_h(tmp2, ColumnSums[0], OnesWordBroadcast);
        ColumnSums[0]= __lsx_vadd_w(tmp1, tmp2);
        tmp1 = tmp2 = __lsx_vldi(0);
        tmp1 = __lsx_vmaddwev_w_h(tmp1, ColumnSums[1], OnesWordBroadcast);
        tmp2 = __lsx_vmaddwod_w_h(tmp2, ColumnSums[1], OnesWordBroadcast);
        ColumnSums[1]= __lsx_vadd_w(tmp1, tmp2);

        __lsx_vst(ColumnSums[0], (__m128i*) & ColumnSumBuffer[0], 0);
        __lsx_vst(ColumnSums[1], (__m128i*) & ColumnSumBuffer[4], 0);
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

        __lsx_vst(BitFlipVector, (__m128i*)PaddedMatrixBData, 0);

        ColumnSums[0] = __lsx_vldi(0);
        ColumnSums[1] = __lsx_vldi(0);

        //
        // Interleave rows of matrix B using an intermediate zero padded stack
        // buffer and write to the packed buffer.
        //

        while (k >= MLAS_GEMM_U8X8_KERNEL_LSX::PackedK) {

            const uint8_t* bcopy = b;
            uint8_t* padded = PaddedMatrixBData;
            uint8_t* padded_end = padded + CountN;

            do {
                padded[0] = bcopy[0];
                padded[8] = bcopy[ldb];
                padded++;
                bcopy++;
            } while (padded < padded_end);

            __m128i BytesRow0 = __lsx_vld((__m128i*) & PaddedMatrixBData[0], 0);
            __lsx_vinsgr2vr_d(BytesRow0, 0, 1); 
            __m128i BytesRow1 = __lsx_vld((__m128i*) & PaddedMatrixBData[8], 0);
            __lsx_vinsgr2vr_d(BytesRow1, 0, 1); 

            MlasGemmU8X8CopyPackBProcessLSX(D, BytesRow0, BytesRow1, BitFlipVector, ColumnSums);

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

            __m128i BytesRow0 = __lsx_vld((__m128i*) & PaddedMatrixBData[0], 0);
            __lsx_vinsgr2vr_d(BytesRow0, 0, 1); 

            MlasGemmU8X8CopyPackBProcessLSX(D, BytesRow0, BitFlipVector, BitFlipVector, ColumnSums);
        }

        __m128i tmp1, tmp2;
        tmp1 = tmp2 = __lsx_vldi(0);
        tmp1 = __lsx_vmaddwev_w_h(tmp1, ColumnSums[0], OnesWordBroadcast);
        tmp2 = __lsx_vmaddwod_w_h(tmp2, ColumnSums[0], OnesWordBroadcast);
        ColumnSums[0]= __lsx_vadd_w(tmp1, tmp2);
        tmp1 = tmp2 = __lsx_vldi(0);
        tmp1 = __lsx_vmaddwev_w_h(tmp1, ColumnSums[1], OnesWordBroadcast);
        tmp2 = __lsx_vmaddwod_w_h(tmp2, ColumnSums[1], OnesWordBroadcast);
        ColumnSums[1]= __lsx_vadd_w(tmp1, tmp2);

        __lsx_vst(ColumnSums[0], (__m128i*) & ColumnSumBuffer[0], 0);
        __lsx_vst(ColumnSums[1], (__m128i*) & ColumnSumBuffer[4], 0);
    }
}

MLAS_FORCEINLINE
void
MlasGemmU8X8MultiplyAccumulateRowLSX(
    __m128i ABroadcast,
    const int16_t* B,
    __m128i Accumulators[2]
)
{
    __m128i BElements0 = __lsx_vld((__m128i*) & B[0], 0);
    __m128i BElements1 = __lsx_vld((__m128i*) & B[8], 0);

    __m128i tmp1, tmp2;
    tmp1 = tmp2 = __lsx_vldi(0);
    tmp1 = __lsx_vmaddwev_w_h(tmp1, BElements0, ABroadcast);
    tmp2 = __lsx_vmaddwod_w_h(tmp2, BElements0, ABroadcast);
    Accumulators[0] = __lsx_vadd_w(Accumulators[0], __lsx_vadd_w(tmp1, tmp2));
    tmp1 = tmp2 = __lsx_vldi(0);
    tmp1 = __lsx_vmaddwev_w_h(tmp1, BElements1, ABroadcast);
    tmp2 = __lsx_vmaddwod_w_h(tmp2, BElements1, ABroadcast);
    Accumulators[1] = __lsx_vadd_w(Accumulators[1], __lsx_vadd_w(tmp1, tmp2));
}

template<>
size_t
MlasGemmQuantKernel<MLAS_GEMM_U8X8_KERNEL_LSX>(
    const MLAS_GEMM_U8X8_KERNEL_LSX::PackedAType* A,
    const MLAS_GEMM_U8X8_KERNEL_LSX::PackedBType* B,
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

            Accumulators[0] = __lsx_vld((__m128i*) & ScaledRowSumBuffer[0], 0);
            Accumulators[1] = __lsx_vld((__m128i*) & ScaledRowSumBuffer[4], 0);

        }
        else {

            Accumulators[0] = __lsx_vreplgr2vr_w(RowSumValue);
            Accumulators[1] = Accumulators[0];
        }

        Accumulators[0] = __lsx_vadd_w(Accumulators[0], __lsx_vld((const __m128i*) & ColumnSumBuffer[0], 0));
        Accumulators[1] = __lsx_vadd_w(Accumulators[1], __lsx_vld((const __m128i*) & ColumnSumBuffer[4], 0));
        ColumnSumBuffer += 8;

        //
        // Broadcast each pair of 16-bit values from the matrix A and multiply
        // with the pair of 16-bit values from matrix B, and add the 32-bit
        // intermediate into the accumulator registers.
        //

        const int16_t* a = A;
        size_t k = PackedCountK;

        while (k >= 4) {

            __m128i AElements = __lsx_vld((__m128i*)a, 0);
            __m128i ABroadcast;

            ABroadcast = __lsx_vreplvei_w(AElements, 0);
            MlasGemmU8X8MultiplyAccumulateRowLSX(ABroadcast, &B[0], Accumulators);

            ABroadcast = __lsx_vreplvei_w(AElements, 1);
            MlasGemmU8X8MultiplyAccumulateRowLSX(ABroadcast, &B[16], Accumulators);

            ABroadcast = __lsx_vreplvei_w(AElements, 2);
            MlasGemmU8X8MultiplyAccumulateRowLSX(ABroadcast, &B[32], Accumulators);

            ABroadcast = __lsx_vreplvei_w(AElements, 3);
            MlasGemmU8X8MultiplyAccumulateRowLSX(ABroadcast, &B[48], Accumulators);

            a += 4 * 2;
            B += 4 * 16;
            k -= 4;
        }

        while (k > 0) {

            __m128i ABroadcast = __lsx_vldrepl_w((int32_t*)a, 0);
            MlasGemmU8X8MultiplyAccumulateRowLSX(ABroadcast, &B[0], Accumulators);

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
                Accumulators[0] = __lsx_vadd_w(Accumulators[0], __lsx_vld((__m128i*) & C[0], 0));
                Accumulators[1] = __lsx_vadd_w(Accumulators[1], __lsx_vld((__m128i*) & C[4], 0));
            }

            __lsx_vst(Accumulators[0], (__m128i*) & C[0], 0);
            __lsx_vst(Accumulators[1], (__m128i*) & C[4], 0);

            C += 8;
            CountN -= 8;

        }
        else {

            //
            // Output the remaining partial output block.
            //

            if ((CountN & 4) != 0) {

                if (!ZeroMode) {
                    Accumulators[0] = __lsx_vadd_w(Accumulators[0], __lsx_vld((__m128i*) & C[0], 0));
                }

                __lsx_vst(Accumulators[0], (__m128i*) & C[0], 0);
                C += 4;

                Accumulators[0] = Accumulators[1];
            }

            if ((CountN & 2) != 0) {

                if (!ZeroMode) {
                    Accumulators[0] = __lsx_vadd_w(Accumulators[0], __lsx_vinsgr2vr_d(__lsx_vld((__m128i*) & C[0], 0), 0, 1));
                }

                *((uint64_t *)&C[0]) = __lsx_vpickve2gr_d(Accumulators[0], 0);
                C += 2;

                Accumulators[0] = __lsx_vshuf4i_w(Accumulators[0], 0xee);
            }

            if ((CountN & 1) != 0) {

                int32_t AccumulatorValue = __lsx_vpickve2gr_w(Accumulators[0], 0);

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

const MLAS_GEMM_QUANT_DISPATCH MlasGemmU8X8DispatchLSX = {
    MlasGemmQuantOperation<MLAS_GEMM_U8X8_KERNEL_LSX>,
    nullptr,
    nullptr,
    MLAS_GEMM_U8X8_KERNEL_LSX::PackedK,
    0,
    1  // aLSXmbly kernel M stride
};
