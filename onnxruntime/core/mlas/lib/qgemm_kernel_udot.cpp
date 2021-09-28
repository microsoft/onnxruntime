/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm_kernel_default.cpp

Abstract:

    This module implements default QGEMM kernel.

--*/

#include "mlasi.h"
#include "qgemm.h"

//
// Define the prototypes of the NEON UDOT routines written in assembly.
//

extern "C" {

    size_t
    MLASCALL
    MlasGemmU8X8KernelUdot(
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

struct MLAS_GEMM_U8X8_KERNEL_UDOT
{
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint8_t OffsetBType;

    static constexpr size_t PackedK = 8;
    static constexpr MLAS_GEMM_U8X8_STRIDES Strides{ 24, 128, 256 };
    static constexpr MLAS_GEMM_U8X8_STRIDES PackedStrides{ 24, 128, 384 };
};

constexpr size_t MLAS_GEMM_U8X8_KERNEL_UDOT::PackedK;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8X8_KERNEL_UDOT::Strides;
constexpr MLAS_GEMM_U8X8_STRIDES MLAS_GEMM_U8X8_KERNEL_UDOT::PackedStrides;

template<>
MLAS_FORCEINLINE
int32_t
MlasGemmU8X8FixupZeroPointB<MLAS_GEMM_U8X8_KERNEL_UDOT>(
    int32_t ZeroPointB,
    bool BIsSigned
    )
{
    if (BIsSigned) {
        ZeroPointB = MLAS_GEMM_U8X8_KERNEL_UDOT::OffsetBType(ZeroPointB ^ 0x80);
    }

    return ZeroPointB;
}

template<>
void
MlasGemmU8X8CopyPackA<MLAS_GEMM_U8X8_KERNEL_UDOT>(
    MLAS_GEMM_U8X8_KERNEL_UDOT::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer
    )
{
    uint8_t PaddedMatrixAData[16];

    //
    // Process 8 rows of matrix A.
    // 
    // DOT kernels load 8x4 block of A with two vector registers. So A is packed
    // a series of 16 byte vectors where four rows are interleaved with the
    // following pattern:
    //
    //      [ A0 A1 A2 A3 B0 B1 B2 B3 C0 C1 C2 C3 D0 D1 D2 D3 ]
    //      [ E0 E1 E2 E3 F0 F1 F2 F3 G0 G1 G2 G3 H0 H1 H2 H3 ]
    // 
    //      [ A4 A5 A6 A7 B4 B5 B6 B7 C4 C5 C6 C7 D4 D5 D6 D7 ]
    //      [ E4 E5 E6 E7 F4 F5 F6 F7 G4 G5 G6 G7 H4 H5 H6 H7 ]
    // 
    //      ...
    //
    // This pattern is repeated (CountK / 8) times.
    //
    // If CountK is not aligned to a multiple of eight, then the vector is padded
    // with zeroes.
    //

    while (CountM >= 8) {
        const uint8_t* a0 = A;
        const uint8_t* a1 = a0 + lda;
        const uint8_t* a2 = a0 + lda * 2;
        const uint8_t* a3 = a0 + lda * 3;
        const uint8_t* a4 = a0 + lda * 4;
        const uint8_t* a5 = a0 + lda * 5;
        const uint8_t* a6 = a0 + lda * 6;
        const uint8_t* a7 = a0 + lda * 7;

        size_t k = CountK;
        uint32x4_t RowSums0 = vmovq_n_u32(0);
        uint32x4_t RowSums1 = vmovq_n_u32(0);

        while (k >= 16) {
            uint32x4_t v0 = vld1q_u32(reinterpret_cast<const uint32_t*>(a0));
            a0 += 16;
            uint32x4_t v1 = vld1q_u32(reinterpret_cast<const uint32_t*>(a1));
            a1 += 16;
            uint32x4_t v2 = vld1q_u32(reinterpret_cast<const uint32_t*>(a2));
            a2 += 16;
            uint32x4_t v3 = vld1q_u32(reinterpret_cast<const uint32_t*>(a3));
            a3 += 16;
            uint32x4_t v4 = vld1q_u32(reinterpret_cast<const uint32_t*>(a4));
            a4 += 16;
            uint32x4_t v5 = vld1q_u32(reinterpret_cast<const uint32_t*>(a5));
            a5 += 16;
            uint32x4_t v6 = vld1q_u32(reinterpret_cast<const uint32_t*>(a6));
            a6 += 16;
            uint32x4_t v7 = vld1q_u32(reinterpret_cast<const uint32_t*>(a7));
            a7 += 16;

            uint32x4_t z0 = vzip1q_u32(v0, v2);
            uint32x4_t z1 = vzip2q_u32(v0, v2);
            uint32x4_t z2 = vzip1q_u32(v1, v3);
            uint32x4_t z3 = vzip2q_u32(v1, v3);

            uint32x4_t z4 = vzip1q_u32(v4, v6);
            uint32x4_t z5 = vzip2q_u32(v4, v6);
            uint32x4_t z6 = vzip1q_u32(v5, v7);
            uint32x4_t z7 = vzip2q_u32(v5, v7);

            v0 = vzip1q_u32(z0, z2);
            v1 = vzip2q_u32(z0, z2);
            v2 = vzip1q_u32(z1, z3);
            v3 = vzip2q_u32(z1, z3);

            v4 = vzip1q_u32(z4, z6);
            v5 = vzip2q_u32(z4, z6);
            v6 = vzip1q_u32(z5, z7);
            v7 = vzip2q_u32(z5, z7);

            vst1q_u8(&D[0], vreinterpretq_u8_u32(v0));
            vst1q_u8(&D[16], vreinterpretq_u8_u32(v4));
            vst1q_u8(&D[32], vreinterpretq_u8_u32(v1));
            vst1q_u8(&D[48], vreinterpretq_u8_u32(v5));
            vst1q_u8(&D[64], vreinterpretq_u8_u32(v2));
            vst1q_u8(&D[80], vreinterpretq_u8_u32(v6));
            vst1q_u8(&D[96], vreinterpretq_u8_u32(v3));
            vst1q_u8(&D[112], vreinterpretq_u8_u32(v7));

            RowSums0 = vpadalq_u16(RowSums0, vpaddlq_u8(vreinterpretq_u8_u32(v0)));
            RowSums0 = vpadalq_u16(RowSums0, vpaddlq_u8(vreinterpretq_u8_u32(v1)));
            RowSums0 = vpadalq_u16(RowSums0, vpaddlq_u8(vreinterpretq_u8_u32(v2)));
            RowSums0 = vpadalq_u16(RowSums0, vpaddlq_u8(vreinterpretq_u8_u32(v3)));

            RowSums1 = vpadalq_u16(RowSums1, vpaddlq_u8(vreinterpretq_u8_u32(v4)));
            RowSums1 = vpadalq_u16(RowSums1, vpaddlq_u8(vreinterpretq_u8_u32(v5)));
            RowSums1 = vpadalq_u16(RowSums1, vpaddlq_u8(vreinterpretq_u8_u32(v6)));
            RowSums1 = vpadalq_u16(RowSums1, vpaddlq_u8(vreinterpretq_u8_u32(v7)));

            D += 128;
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
            uint32_t v4 = *reinterpret_cast<const uint32_t*>(a4);
            a4 += 4;
            uint32_t v5 = *reinterpret_cast<const uint32_t*>(a5);
            a5 += 4;
            uint32_t v6 = *reinterpret_cast<const uint32_t*>(a6);
            a6 += 4;
            uint32_t v7 = *reinterpret_cast<const uint32_t*>(a7);
            a7 += 4;

            *reinterpret_cast<uint32_t*>(&D[0]) = v0;
            *reinterpret_cast<uint32_t*>(&D[4]) = v1;
            *reinterpret_cast<uint32_t*>(&D[8]) = v2;
            *reinterpret_cast<uint32_t*>(&D[12]) = v3;
            *reinterpret_cast<uint32_t*>(&D[16]) = v4;
            *reinterpret_cast<uint32_t*>(&D[20]) = v5;
            *reinterpret_cast<uint32_t*>(&D[24]) = v6;
            *reinterpret_cast<uint32_t*>(&D[28]) = v7;

            RowSums0 = vpadalq_u16(RowSums0, vpaddlq_u8(vld1q_u8(D)));
            RowSums1 = vpadalq_u16(RowSums1, vpaddlq_u8(vld1q_u8(&D[16])));

            D += 32;
            k -= 4;
        }

        if (k > 0) {
            //
            // Copy the remaining bytes to the zero padded stack buffer.
            //
            uint8_t* d = D;

            vst1q_u8(d, vmovq_n_u8(0));
            vst1q_u8(&d[16], vmovq_n_u8(0));

            while (k > 0) {
                d[0] = *a0++;
                d[4] = *a1++;
                d[8] = *a2++;
                d[12] = *a3++;
                d[16] = *a4++;
                d[20] = *a5++;
                d[24] = *a6++;
                d[28] = *a7++;
                d += 1;
                k -= 1;
            }

            RowSums0 = vpadalq_u16(RowSums0, vpaddlq_u8(vld1q_u8(D)));
            RowSums1 = vpadalq_u16(RowSums1, vpaddlq_u8(vld1q_u8(&D[16])));

            D += 32;
        }

        if (((CountK - 1) & 7) < 4) {
            vst1q_u8(D, vmovq_n_u8(0));
            vst1q_u8(&D[16], vmovq_n_u8(0));
            D += 32;
        }

        vst1q_s32(RowSumBuffer, vreinterpretq_s32_u32(RowSums0));
        vst1q_s32(&RowSumBuffer[4], vreinterpretq_s32_u32(RowSums1));

        RowSumBuffer += 8;

        A = A + lda * 8;
        CountM -= 8;
    }

    //
    // Process four rows of matrix A.
    //
    // The buffer is packed as a series of 16 byte vectors where four rows are
    // interleaved with the following pattern:
    //
    //      [ A0 A1 A2 A3 B0 B1 B2 B3 C0 C1 C2 C3 D0 D1 D2 D3 ]
    //      [ A4 A5 A6 A7 B4 B5 B6 B7 C4 C5 C6 C7 D4 D5 D6 D7 ]
    //
    // This pattern is repeated (CountK / 8) times.
    //
    // If CountK is not aligned to a multiple of eight, then the vector is padded
    // with zeroes.
    //

    if (CountM >= 4) {

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

        if (((CountK - 1) & 7) < 4) {

            vst1q_u8(D, vmovq_n_u8(0));

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
    // This pattern is repeated (CountK / 8) times.
    //
    // If CountK is not aligned to a multiple of four, then the vector is padded
    // with zeroes.
    //

    if (CountM >= 2) {

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

            vst1_u8(PaddedMatrixAData, vmov_n_u8(0));

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

        if (((CountK - 1) & 7) < 4) {

            vst1_u8(D, vmov_n_u8(0));

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
    // This pattern is repeated (CountK / 8) times.
    //
    // If CountK is not aligned to a multiple of four, then the vector is padded
    // with zeroes.
    //

    if (CountM > 0) {

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
        vst1q_lane_u32(reinterpret_cast<uint32_t*>(RowSumBuffer), RowSums, 0);
#else
        *RowSumBuffer = int32_t(vaddvq_u32(RowSums));
#endif
    }
}

MLAS_FORCEINLINE
void
MlasGemmU8X8CopyPackBProcessUdot(
    MLAS_GEMM_U8X8_KERNEL_UDOT::PackedBType* D,
    uint8x8_t BytesRow[4],
    uint8x16_t BitFlipVector,
    uint32x4_t ColumnSums[2]
    )
{
    uint8x16_t v02 = veorq_u8(vcombine_u8(BytesRow[0], BytesRow[2]), BitFlipVector);
    uint8x16_t v13 = veorq_u8(vcombine_u8(BytesRow[1], BytesRow[3]), BitFlipVector);

    uint8x16x2_t zw = vzipq_u8(v02, v13);
    uint16x8x2_t zd = vzipq_u16(vreinterpretq_u16_u8(zw.val[0]), vreinterpretq_u16_u8(zw.val[1]));

    vst1q_u8(&D[0], vreinterpretq_u8_u16(zd.val[0]));
    vst1q_u8(&D[16], vreinterpretq_u8_u16(zd.val[1]));

    ColumnSums[0] = vpadalq_u16(ColumnSums[0], vpaddlq_u8(vreinterpretq_u8_u16(zd.val[0])));
    ColumnSums[1] = vpadalq_u16(ColumnSums[1], vpaddlq_u8(vreinterpretq_u8_u16(zd.val[1])));
}

template<>
void
MlasGemmU8X8CopyPackB<MLAS_GEMM_U8X8_KERNEL_UDOT>(
    MLAS_GEMM_U8X8_KERNEL_UDOT::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    const uint8x16_t ZeroVector = vmovq_n_u8(0);
    const uint8x16_t BitFlipVector = vdupq_n_u8(BIsSigned ? 0x80 : 0);
    uint8x8_t BytesRow[4];

    //
    // Process 8 columns of matrix B in a loop.
    //
    // The buffer is packed as a series of 16 byte vectors where eight rows are
    // interleaved with the following pattern:
    //
    //      [ A0 A1 A2 A3 B0 B1 B2 B3 C0 C1 C2 C3 D0 D1 D2 D3 ]
    //      [ E0 E1 E2 E3 F0 F1 F2 F3 G0 G1 G2 G3 H0 H1 H2 H3 ]
    //      [ A4 A5 A6 A7 B4 B5 B6 B7 C4 C5 C6 C7 D4 D5 D6 D7 ]
    //      [ E4 E5 E6 E7 F4 F5 F6 F7 G4 G5 G6 G7 H4 H5 H6 H7 ]
    //
    // Copy columns from matrix B to the packed buffer. Signed buffers are
    // converted to unsigned buffers in order to share a common kernel.
    //
    // If CountK is not aligned to a multiple of eight, then the packed buffer
    // is padded with zero vectors.
    //
    // If CountN is not aligned to a multiple of four, then the extra columns
    // are padded with zeroes.
    //

    while (CountN >= 8) {

        const uint8_t* b = B;
        size_t k = CountK;
        uint32x4_t ColumnSums[2];

        ColumnSums[0] = vmovq_n_u32(0);
        ColumnSums[1] = vmovq_n_u32(0);

        //
        // Interleave rows of matrix B and write to the packed buffer.
        //

        while (k >= 4) {

            BytesRow[0] = vld1_u8(&b[ldb * 0]);
            BytesRow[1] = vld1_u8(&b[ldb * 1]);
            BytesRow[2] = vld1_u8(&b[ldb * 2]);
            BytesRow[3] = vld1_u8(&b[ldb * 3]);

            MlasGemmU8X8CopyPackBProcessUdot(D, BytesRow, BitFlipVector, ColumnSums);

            b += ldb * 4;
            D += 32;
            k -= 4;
        }

        if (k > 0) {

            BytesRow[0] = vld1_u8(&b[ldb * 0]);
            BytesRow[1] = (k >= 2) ? vld1_u8(&b[ldb * 1]) : vget_low_u8(BitFlipVector);
            BytesRow[2] = (k > 2) ? vld1_u8(&b[ldb * 2]) : vget_low_u8(BitFlipVector);
            BytesRow[3] = vget_low_u8(BitFlipVector);

            MlasGemmU8X8CopyPackBProcessUdot(D, BytesRow, BitFlipVector, ColumnSums);

            D += 32;
        }

        //
        // Zero pad the output buffer to a multiple of PackedK if the above
        // processed an odd number of four row bundles.
        //

        if (((CountK - 1) & 7) < 4) {

            vst1q_u8(&D[0], ZeroVector);
            vst1q_u8(&D[16], ZeroVector);

            D += 32;
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
        size_t k = CountK;
        uint8_t PaddedMatrixBData[32];
        uint32x4_t ColumnSums[2];

        vst1q_u8(&PaddedMatrixBData[0], BitFlipVector);
        vst1q_u8(&PaddedMatrixBData[16], BitFlipVector);

        ColumnSums[0] = vmovq_n_u32(0);
        ColumnSums[1] = vmovq_n_u32(0);

        //
        // Interleave rows of matrix B using an intermediate zero padded stack
        // buffer and write to the packed buffer.
        //

        while (k > 0) {

            const uint8_t* bcopy0 = &b[ldb * 0];
            const uint8_t* bcopy1 = &b[ldb * 1];
            const uint8_t* bcopy2 = &b[ldb * 2];
            const uint8_t* bcopy3 = &b[ldb * 3];

            if (k >= 4) {

                b += ldb * 4;
                k -= 4;

            } else {

                vst1q_u8(&PaddedMatrixBData[0], BitFlipVector);
                vst1q_u8(&PaddedMatrixBData[16], BitFlipVector);

                bcopy1 = (k >= 2) ? bcopy1 : &PaddedMatrixBData[24];
                bcopy2 = (k > 2) ? bcopy2 : &PaddedMatrixBData[24];
                bcopy3 = &PaddedMatrixBData[24];

                k = 0;
            }

            uint8_t* padded = PaddedMatrixBData;
            uint8_t* padded_end = padded + CountN;

            do {
                padded[0] = *bcopy0++;
                padded[8] = *bcopy1++;
                padded[16] = *bcopy2++;
                padded[24] = *bcopy3++;
            } while (++padded < padded_end);

            BytesRow[0] = vld1_u8(&PaddedMatrixBData[0]);
            BytesRow[1] = vld1_u8(&PaddedMatrixBData[8]);
            BytesRow[2] = vld1_u8(&PaddedMatrixBData[16]);
            BytesRow[3] = vld1_u8(&PaddedMatrixBData[24]);

            MlasGemmU8X8CopyPackBProcessUdot(D, BytesRow, BitFlipVector, ColumnSums);

            D += 32;
        }

        //
        // Zero pad the output buffer to a multiple of PackedK if the above
        // processed an odd number of four row bundles.
        //

        if (((CountK - 1) & 7) < 4) {

            vst1q_u8(&D[0], ZeroVector);
            vst1q_u8(&D[16], ZeroVector);

            D += 32;
        }

        vst1q_s32(&ColumnSumBuffer[0], vreinterpretq_s32_u32(ColumnSums[0]));
        vst1q_s32(&ColumnSumBuffer[4], vreinterpretq_s32_u32(ColumnSums[1]));
    }
}

template<>
MLAS_FORCEINLINE
size_t
MlasGemmU8X8Kernel<MLAS_GEMM_U8X8_KERNEL_UDOT>(
    const MLAS_GEMM_U8X8_KERNEL_UDOT::PackedAType* A,
    const MLAS_GEMM_U8X8_KERNEL_UDOT::PackedBType* B,
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
    return MlasGemmU8X8KernelUdot(A, B, C, PackedCountK, CountM, CountN, ldc,
        RowSumBuffer, ColumnSumBuffer, ZeroPointB, ZeroMode);
}

const MLAS_GEMM_U8X8_DISPATCH MlasGemmU8X8DispatchUdot = {
    MlasGemmU8X8Operation<MLAS_GEMM_U8X8_KERNEL_UDOT>,
    MlasGemmU8X8PackedOperation<MLAS_GEMM_U8X8_KERNEL_UDOT>,
    MlasGemmU8X8CopyPackB<MLAS_GEMM_U8X8_KERNEL_UDOT>,
    MLAS_GEMM_U8X8_KERNEL_UDOT::PackedK,
    MLAS_GEMM_U8X8_KERNEL_UDOT::PackedStrides.K,
};
