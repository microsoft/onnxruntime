/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm_kernel_neon.cpp

Abstract:

    This module implements QGEMM kernel for neon.

--*/

#include "mlasi.h"
#include "qgemm.h"

//
// Define the prototypes of the NEON routines written in assembly.
//
// N.B. The kernel has not been ported to build with the Windows ARM32 toolset.
//

extern "C" {

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
        const int32_t* ZeroPointB,
        bool ZeroMode
        );
}

struct MLAS_GEMM_U8X8_KERNEL_NEON
{
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint8_t OffsetBType;

    static constexpr size_t PackedK = 4;
    static constexpr MLAS_GEMM_U8X8_STRIDES Strides{ 24, 128, 256 };
    static constexpr MLAS_GEMM_U8X8_STRIDES PackedStrides{ 24, 128, 256 };
};

constexpr size_t MLAS_GEMM_U8X8_KERNEL_NEON::PackedK;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8X8_KERNEL_NEON::Strides;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8X8_KERNEL_NEON::PackedStrides;

template<>
MLAS_FORCEINLINE
int32_t
MlasGemmU8X8FixupZeroPointB<MLAS_GEMM_U8X8_KERNEL_NEON>(
    int32_t ZeroPointB,
    bool BIsSigned
    )
{
    if (BIsSigned) {
        ZeroPointB = MLAS_GEMM_U8X8_KERNEL_NEON::OffsetBType(ZeroPointB ^ 0x80);
    }

    return ZeroPointB;
}

template<>
void
MlasGemmU8X8CopyPackA<MLAS_GEMM_U8X8_KERNEL_NEON>(
    MLAS_GEMM_U8X8_KERNEL_NEON::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer
    )
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

#if defined(MLAS_NEON32_INTRINSICS)
            uint32x4x2_t z0 = vzipq_u32(v0, v2);
            uint32x4x2_t z1 = vzipq_u32(v1, v3);

            v0 = z0.val[0];
            v1 = z0.val[1];
            v2 = z1.val[0];
            v3 = z1.val[1];

            uint32x4x2_t z2 = vzipq_u32(v0, v2);
            uint32x4x2_t z3 = vzipq_u32(v1, v3);

            v0 = z2.val[0];
            v1 = z2.val[1];
            v2 = z3.val[0];
            v3 = z3.val[1];
#else
            uint32x4_t z0 = vzip1q_u32(v0, v2);
            uint32x4_t z1 = vzip2q_u32(v0, v2);
            uint32x4_t z2 = vzip1q_u32(v1, v3);
            uint32x4_t z3 = vzip2q_u32(v1, v3);

            v0 = vzip1q_u32(z0, z2);
            v1 = vzip2q_u32(z0, z2);
            v2 = vzip1q_u32(z1, z3);
            v3 = vzip2q_u32(z1, z3);
#endif

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

        while (k >= 4) {

            uint32_t v0 = *reinterpret_cast<const uint32_t*>(a0);
            a0 += 4;
            uint32_t v1 = *reinterpret_cast<const uint32_t*>(a1);
            a1 += 4;
            uint32_t v2 = *reinterpret_cast<const uint32_t*>(a2);
            a2 += 4;
            uint32_t v3 = *reinterpret_cast<const uint32_t*>(a3);
            a3 += 4;

            *reinterpret_cast<uint32_t*>(&D[0]) = v0;
            *reinterpret_cast<uint32_t*>(&D[4]) = v1;
            *reinterpret_cast<uint32_t*>(&D[8]) = v2;
            *reinterpret_cast<uint32_t*>(&D[12]) = v3;

            RowSums = vpadalq_u16(RowSums, vpaddlq_u8(vld1q_u8(D)));

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

        while (k >= 4) {

            uint32_t v0 = *reinterpret_cast<const uint32_t*>(a0);
            a0 += 4;
            uint32_t v1 = *reinterpret_cast<const uint32_t*>(a1);
            a1 += 4;

            *reinterpret_cast<uint32_t*>(&D[0]) = v0;
            *reinterpret_cast<uint32_t*>(&D[4]) = v1;

            RowSums = vpadal_u16(RowSums, vpaddl_u8(vld1_u8(D)));

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

#if defined(MLAS_NEON32_INTRINSICS)
        uint32x2_t RowSumsLow = vpadd_u32(vget_high_u32(RowSums), vget_low_u32(RowSums));
        RowSumsLow = vpadd_u32(RowSumsLow, RowSumsLow);
        vst1_lane_u32(reinterpret_cast<uint32_t*>(RowSumBuffer), RowSumsLow, 0);
#elif defined(_M_ARM64)
        // N.B. The workaround of defining a local vaddvq_u32 doesn't work here
        // as VS2019 added new intrinsics to make the operation work. Also, not
        // all build environments using VS2019 have the up-to-date arm64_neon.h,
        // so fallback to pairwise addition.
        RowSums = vpaddq_u32(RowSums, RowSums);
        RowSums = vpaddq_u32(RowSums, RowSums);
        vst1q_lane_u32(reinterpret_cast<uint32_t*>(RowSumBuffer), RowSums, 0);
#else
        * RowSumBuffer = int32_t(vaddvq_u32(RowSums));
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
#if defined(MLAS_NEON32_INTRINSICS)
    ColumnSums[1] = vaddq_u32(ColumnSums[1], vmovl_u16(vget_high_u16(WordsRow)));
#else
    ColumnSums[1] = vaddq_u32(ColumnSums[1], vmovl_high_u16(WordsRow));
#endif
}

template<>
void
MlasGemmU8X8CopyPackB<MLAS_GEMM_U8X8_KERNEL_NEON>(
    MLAS_GEMM_U8X8_KERNEL_NEON::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    const uint8x8_t BitFlipVector = vdup_n_u8(BIsSigned ? 0x80 : 0);
    const uint8x8_t ZeroVector = vmov_n_u8(0);
    const size_t AlignedCountK =
        (CountK + MLAS_GEMM_U8X8_KERNEL_NEON::PackedK - 1) & ~(MLAS_GEMM_U8X8_KERNEL_NEON::PackedK - 1);

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

template<>
MLAS_FORCEINLINE
size_t
MlasGemmU8X8Kernel<MLAS_GEMM_U8X8_KERNEL_NEON>(
    const MLAS_GEMM_U8X8_KERNEL_NEON::PackedAType* A,
    const MLAS_GEMM_U8X8_KERNEL_NEON::PackedBType* B,
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
    return MlasGemmU8X8KernelNeon(A, B, C, PackedCountK, CountM, CountN, ldc,
                                  RowSumBuffer, ColumnSumBuffer, ZeroPointB, ZeroMode);
}

const MLAS_GEMM_U8X8_DISPATCH MlasGemmU8X8DispatchNeon = {
    MlasGemmU8X8Operation<MLAS_GEMM_U8X8_KERNEL_NEON>,
    MlasGemmU8X8PackedOperation<MLAS_GEMM_U8X8_KERNEL_NEON>,
    MlasGemmU8X8CopyPackB<MLAS_GEMM_U8X8_KERNEL_NEON>,
    MLAS_GEMM_U8X8_KERNEL_NEON::PackedK,
    MLAS_GEMM_U8X8_KERNEL_NEON::PackedStrides.K,
};
