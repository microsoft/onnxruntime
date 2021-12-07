/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm_kernel_sdot.cpp

Abstract:

    This module implements sdot QGEMM kernel.

--*/

#include "mlasi.h"
#include "qgemm.h"

//
// Define the prototypes of the NEON SDOT routines written in assembly.
//

extern "C" {

    size_t
    MLASCALL
    MlasGemmS8S8KernelSDot(
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

struct MLAS_GEMM_S8S8_KERNEL_SDOT
{
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef int8_t OffsetAType;
    typedef int8_t OffsetBType;

    static constexpr size_t PackedK = 8;
    static constexpr MLAS_GEMM_QUANT_STRIDES Strides{ 24, 128, 256 };
    static constexpr MLAS_GEMM_QUANT_STRIDES PackedStrides{ 24, 128, 384 };
};

constexpr size_t MLAS_GEMM_S8S8_KERNEL_SDOT::PackedK;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_S8S8_KERNEL_SDOT::Strides;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_S8S8_KERNEL_SDOT::PackedStrides;

template<>
MLAS_FORCEINLINE
int32_t
MlasGemmQuantFixupZeroPointB<MLAS_GEMM_S8S8_KERNEL_SDOT>(
    int32_t ZeroPointB,
    bool BIsSigned
    )
{
    MLAS_UNREFERENCED_PARAMETER(BIsSigned);
    return ZeroPointB;
}

template<>
void
MlasGemmQuantCopyPackA<MLAS_GEMM_S8S8_KERNEL_SDOT>(
    MLAS_GEMM_S8S8_KERNEL_SDOT::PackedAType* D_uint8_t,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer,
    bool AIsSigned
    )
{
    int8_t* D = reinterpret_cast<int8_t*>(D_uint8_t);
    MLAS_UNREFERENCED_PARAMETER(AIsSigned);
    int8_t PaddedMatrixAData[16];

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
        const int8_t* a0 = reinterpret_cast<const int8_t*>(A);
        const int8_t* a1 = a0 + lda;
        const int8_t* a2 = a0 + lda * 2;
        const int8_t* a3 = a0 + lda * 3;
        const int8_t* a4 = a0 + lda * 4;
        const int8_t* a5 = a0 + lda * 5;
        const int8_t* a6 = a0 + lda * 6;
        const int8_t* a7 = a0 + lda * 7;

        size_t k = CountK;
        int32x4_t RowSums0 = vmovq_n_s32(0);
        int32x4_t RowSums1 = vmovq_n_s32(0);

        while (k >= 16) {
            int32x4_t v0 = vld1q_s32(reinterpret_cast<const int32_t*>(a0));
            a0 += 16;
            int32x4_t v1 = vld1q_s32(reinterpret_cast<const int32_t*>(a1));
            a1 += 16;
            int32x4_t v2 = vld1q_s32(reinterpret_cast<const int32_t*>(a2));
            a2 += 16;
            int32x4_t v3 = vld1q_s32(reinterpret_cast<const int32_t*>(a3));
            a3 += 16;
            int32x4_t v4 = vld1q_s32(reinterpret_cast<const int32_t*>(a4));
            a4 += 16;
            int32x4_t v5 = vld1q_s32(reinterpret_cast<const int32_t*>(a5));
            a5 += 16;
            int32x4_t v6 = vld1q_s32(reinterpret_cast<const int32_t*>(a6));
            a6 += 16;
            int32x4_t v7 = vld1q_s32(reinterpret_cast<const int32_t*>(a7));
            a7 += 16;

            int32x4_t z0 = vzip1q_s32(v0, v2);
            int32x4_t z1 = vzip2q_s32(v0, v2);
            int32x4_t z2 = vzip1q_s32(v1, v3);
            int32x4_t z3 = vzip2q_s32(v1, v3);

            int32x4_t z4 = vzip1q_s32(v4, v6);
            int32x4_t z5 = vzip2q_s32(v4, v6);
            int32x4_t z6 = vzip1q_s32(v5, v7);
            int32x4_t z7 = vzip2q_s32(v5, v7);

            v0 = vzip1q_s32(z0, z2);
            v1 = vzip2q_s32(z0, z2);
            v2 = vzip1q_s32(z1, z3);
            v3 = vzip2q_s32(z1, z3);

            v4 = vzip1q_s32(z4, z6);
            v5 = vzip2q_s32(z4, z6);
            v6 = vzip1q_s32(z5, z7);
            v7 = vzip2q_s32(z5, z7);

            vst1q_s8(&D[0], vreinterpretq_s8_s32(v0));
            vst1q_s8(&D[16], vreinterpretq_s8_s32(v4));
            vst1q_s8(&D[32], vreinterpretq_s8_s32(v1));
            vst1q_s8(&D[48], vreinterpretq_s8_s32(v5));
            vst1q_s8(&D[64], vreinterpretq_s8_s32(v2));
            vst1q_s8(&D[80], vreinterpretq_s8_s32(v6));
            vst1q_s8(&D[96], vreinterpretq_s8_s32(v3));
            vst1q_s8(&D[112], vreinterpretq_s8_s32(v7));

            RowSums0 = vpadalq_s16(RowSums0, vpaddlq_s8(vreinterpretq_s8_s32(v0)));
            RowSums0 = vpadalq_s16(RowSums0, vpaddlq_s8(vreinterpretq_s8_s32(v1)));
            RowSums0 = vpadalq_s16(RowSums0, vpaddlq_s8(vreinterpretq_s8_s32(v2)));
            RowSums0 = vpadalq_s16(RowSums0, vpaddlq_s8(vreinterpretq_s8_s32(v3)));

            RowSums1 = vpadalq_s16(RowSums1, vpaddlq_s8(vreinterpretq_s8_s32(v4)));
            RowSums1 = vpadalq_s16(RowSums1, vpaddlq_s8(vreinterpretq_s8_s32(v5)));
            RowSums1 = vpadalq_s16(RowSums1, vpaddlq_s8(vreinterpretq_s8_s32(v6)));
            RowSums1 = vpadalq_s16(RowSums1, vpaddlq_s8(vreinterpretq_s8_s32(v7)));

            D += 128;
            k -= 16;
        }

        while (k >= 4) {
            int32_t v0 = *reinterpret_cast<const int32_t*>(a0);
            a0 += 4;
            int32_t v1 = *reinterpret_cast<const int32_t*>(a1);
            a1 += 4;
            int32_t v2 = *reinterpret_cast<const int32_t*>(a2);
            a2 += 4;
            int32_t v3 = *reinterpret_cast<const int32_t*>(a3);
            a3 += 4;
            int32_t v4 = *reinterpret_cast<const int32_t*>(a4);
            a4 += 4;
            int32_t v5 = *reinterpret_cast<const int32_t*>(a5);
            a5 += 4;
            int32_t v6 = *reinterpret_cast<const int32_t*>(a6);
            a6 += 4;
            int32_t v7 = *reinterpret_cast<const int32_t*>(a7);
            a7 += 4;

            *reinterpret_cast<int32_t*>(&D[0]) = v0;
            *reinterpret_cast<int32_t*>(&D[4]) = v1;
            *reinterpret_cast<int32_t*>(&D[8]) = v2;
            *reinterpret_cast<int32_t*>(&D[12]) = v3;
            *reinterpret_cast<int32_t*>(&D[16]) = v4;
            *reinterpret_cast<int32_t*>(&D[20]) = v5;
            *reinterpret_cast<int32_t*>(&D[24]) = v6;
            *reinterpret_cast<int32_t*>(&D[28]) = v7;

            RowSums0 = vpadalq_s16(RowSums0, vpaddlq_s8(vld1q_s8(D)));
            RowSums1 = vpadalq_s16(RowSums1, vpaddlq_s8(vld1q_s8(&D[16])));

            D += 32;
            k -= 4;
        }

        if (k > 0) {
            //
            // Copy the remaining bytes to the zero padded stack buffer.
            //
            int8_t* d = D;

            vst1q_s8(d, vmovq_n_s8(0));
            vst1q_s8(&d[16], vmovq_n_s8(0));

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

            RowSums0 = vpadalq_s16(RowSums0, vpaddlq_s8(vld1q_s8(D)));
            RowSums1 = vpadalq_s16(RowSums1, vpaddlq_s8(vld1q_s8(&D[16])));

            D += 32;
        }

        if (((CountK - 1) & 7) < 4) {
            vst1q_s8(D, vmovq_n_s8(0));
            vst1q_s8(&D[16], vmovq_n_s8(0));
            D += 32;
        }

        vst1q_s32(RowSumBuffer, RowSums0);
        vst1q_s32(&RowSumBuffer[4], RowSums1);

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

        const int8_t* a0 = reinterpret_cast<const int8_t*>(A);
        const int8_t* a1 = a0 + lda;
        const int8_t* a2 = a1 + lda;
        const int8_t* a3 = a2 + lda;

        size_t k = CountK;
        int32x4_t RowSums = vmovq_n_s32(0);

        while (k >= 16) {

            int32x4_t v0 = vld1q_s32(reinterpret_cast<const int32_t*>(a0));
            a0 += 16;
            int32x4_t v1 = vld1q_s32(reinterpret_cast<const int32_t*>(a1));
            a1 += 16;
            int32x4_t v2 = vld1q_s32(reinterpret_cast<const int32_t*>(a2));
            a2 += 16;
            int32x4_t v3 = vld1q_s32(reinterpret_cast<const int32_t*>(a3));
            a3 += 16;

            int32x4_t z0 = vzip1q_s32(v0, v2);
            int32x4_t z1 = vzip2q_s32(v0, v2);
            int32x4_t z2 = vzip1q_s32(v1, v3);
            int32x4_t z3 = vzip2q_s32(v1, v3);

            v0 = vzip1q_s32(z0, z2);
            v1 = vzip2q_s32(z0, z2);
            v2 = vzip1q_s32(z1, z3);
            v3 = vzip2q_s32(z1, z3);

            vst1q_s8(&D[0], vreinterpretq_s8_s32(v0));
            vst1q_s8(&D[16], vreinterpretq_s8_s32(v1));
            vst1q_s8(&D[32], vreinterpretq_s8_s32(v2));
            vst1q_s8(&D[48], vreinterpretq_s8_s32(v3));

            RowSums = vpadalq_s16(RowSums, vpaddlq_s8(vreinterpretq_s8_s32(v0)));
            RowSums = vpadalq_s16(RowSums, vpaddlq_s8(vreinterpretq_s8_s32(v1)));
            RowSums = vpadalq_s16(RowSums, vpaddlq_s8(vreinterpretq_s8_s32(v2)));
            RowSums = vpadalq_s16(RowSums, vpaddlq_s8(vreinterpretq_s8_s32(v3)));

            D += 64;
            k -= 16;
        }

        while (k >= 4) {

            int32_t v0 = *reinterpret_cast<const int32_t*>(a0);
            a0 += 4;
            int32_t v1 = *reinterpret_cast<const int32_t*>(a1);
            a1 += 4;
            int32_t v2 = *reinterpret_cast<const int32_t*>(a2);
            a2 += 4;
            int32_t v3 = *reinterpret_cast<const int32_t*>(a3);
            a3 += 4;

            *reinterpret_cast<int32_t*>(&D[0]) = v0;
            *reinterpret_cast<int32_t*>(&D[4]) = v1;
            *reinterpret_cast<int32_t*>(&D[8]) = v2;
            *reinterpret_cast<int32_t*>(&D[12]) = v3;

            RowSums = vpadalq_s16(RowSums, vpaddlq_s8(vld1q_s8(D)));

            D += 16;
            k -= 4;
        }

        if (k > 0) {

            //
            // Copy the remaining bytes to the zero padded stack buffer.
            //

            int8_t* d = PaddedMatrixAData;

            vst1q_s8(PaddedMatrixAData, vmovq_n_s8(0));

            while (k > 0) {

                d[0] = *a0++;
                d[4] = *a1++;
                d[8] = *a2++;
                d[12] = *a3++;

                d += 1;
                k -= 1;
            }

            int8x16_t PackedVector = vld1q_s8(PaddedMatrixAData);
            vst1q_s8(D, PackedVector);

            RowSums = vpadalq_s16(RowSums, vpaddlq_s8(PackedVector));

            D += 16;
        }

        if (((CountK - 1) & 7) < 4) {

            vst1q_s8(D, vmovq_n_s8(0));

            D += 16;
        }

        vst1q_s32(RowSumBuffer, RowSums);
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

        const int8_t* a0 = reinterpret_cast<const int8_t*>(A);
        const int8_t* a1 = a0 + lda;

        size_t k = CountK;
        int32x2_t RowSums = vmov_n_s32(0);

        while (k >= 4) {

            int32_t v0 = *reinterpret_cast<const int32_t*>(a0);
            a0 += 4;
            int32_t v1 = *reinterpret_cast<const int32_t*>(a1);
            a1 += 4;

            *reinterpret_cast<int32_t*>(&D[0]) = v0;
            *reinterpret_cast<int32_t*>(&D[4]) = v1;

            RowSums = vpadal_s16(RowSums, vpaddl_s8(vld1_s8(D)));

            D += 8;
            k -= 4;
        }

        if (k > 0) {

            //
            // Copy the remaining bytes to the zero padded stack buffer.
            //

            int8_t* d = PaddedMatrixAData;

            vst1_s8(PaddedMatrixAData, vmov_n_s8(0));

            while (k > 0) {

                d[0] = *a0++;
                d[4] = *a1++;

                d += 1;
                k -= 1;
            }

            int8x8_t PackedVector = vld1_s8(PaddedMatrixAData);
            vst1_s8(D, PackedVector);

            RowSums = vpadal_s16(RowSums, vpaddl_s8(PackedVector));

            D += 8;
        }

        if (((CountK - 1) & 7) < 4) {

            vst1_s8(D, vmov_n_s8(0));

            D += 8;
        }

        vst1_s32(RowSumBuffer, RowSums);
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

        const int8_t* a = reinterpret_cast<const int8_t*>(A);
        size_t k = CountK;
        int32x4_t RowSums = vmovq_n_s32(0);

        while (k >= 16) {

            int8x16_t v = vld1q_s8(a);
            a += 16;

            vst1q_s8(D, v);

            RowSums = vpadalq_s16(RowSums, vpaddlq_s8(v));

            D += 16;
            k -= 16;
        }

        if (k > 0) {

            //
            // Copy the remaining bytes to the zero padded stack buffer.
            //

            vst1q_s8(PaddedMatrixAData, vmovq_n_s8(0));

            for (size_t kk = 0; kk < k; kk++) {
                PaddedMatrixAData[kk] = a[kk];
            }

            int8x16_t v = vld1q_s8(PaddedMatrixAData);
            vst1q_s8(D, v);

            RowSums = vpadalq_s16(RowSums, vpaddlq_s8(v));
        }

#if defined(_M_ARM64)
        // N.B. The workaround of defining a local vaddvq_u32 doesn't work here
        // as VS2019 added new intrinsics to make the operation work. Also, not
        // all build environments using VS2019 have the up-to-date arm64_neon.h,
        // so fallback to pairwise addition.
        RowSums = vpaddq_s32(RowSums, RowSums);
        RowSums = vpaddq_s32(RowSums, RowSums);
        vst1q_lane_s32(reinterpret_cast<int32_t*>(RowSumBuffer), RowSums, 0);
#else
        *RowSumBuffer = int32_t(vaddvq_s32(RowSums));
#endif
    }
}

MLAS_FORCEINLINE
void
MlasGemmS8S8CopyPackBProcessSDot(
    int8_t* D,
    int8x8_t BytesRow[4],
    int32x4_t ColumnSums[2]
    )
{
    int8x16_t v02 = vcombine_s8(BytesRow[0], BytesRow[2]);
    int8x16_t v13 = vcombine_s8(BytesRow[1], BytesRow[3]);

    int8x16x2_t zw = vzipq_s8(v02, v13);
    int16x8x2_t zd = vzipq_s16(vreinterpretq_s16_s8(zw.val[0]), vreinterpretq_s16_s8(zw.val[1]));

    vst1q_s8(&D[0], vreinterpretq_s8_s16(zd.val[0]));
    vst1q_s8(&D[16], vreinterpretq_s8_s16(zd.val[1]));

    ColumnSums[0] = vpadalq_s16(ColumnSums[0], vpaddlq_s8(vreinterpretq_s8_s16(zd.val[0])));
    ColumnSums[1] = vpadalq_s16(ColumnSums[1], vpaddlq_s8(vreinterpretq_s8_s16(zd.val[1])));
}

template<>
void
MlasGemmQuantCopyPackB<MLAS_GEMM_S8S8_KERNEL_SDOT>(
    MLAS_GEMM_S8S8_KERNEL_SDOT::PackedBType* Dst,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    MLAS_UNREFERENCED_PARAMETER(BIsSigned);
    int8_t* D = reinterpret_cast<int8_t*>(Dst);
    const int8x16_t ZeroVector = vmovq_n_s8(0);
    int8x8_t BytesRow[4];

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
    // Copy columns from matrix B to the packed buffer.
    //
    // If CountK is not aligned to a multiple of eight, then the packed buffer
    // is padded with zero vectors.
    //
    // If CountN is not aligned to a multiple of eight, then the extra columns
    // are padded with zeroes.
    //

    while (CountN >= 8) {

        const int8_t* b = reinterpret_cast<const int8_t*>(B);
        size_t k = CountK;
        int32x4_t ColumnSums[2];

        ColumnSums[0] = vmovq_n_s32(0);
        ColumnSums[1] = vmovq_n_s32(0);

        //
        // Interleave rows of matrix B and write to the packed buffer.
        //

        while (k >= 4) {

            BytesRow[0] = vld1_s8(&b[ldb * 0]);
            BytesRow[1] = vld1_s8(&b[ldb * 1]);
            BytesRow[2] = vld1_s8(&b[ldb * 2]);
            BytesRow[3] = vld1_s8(&b[ldb * 3]);

            MlasGemmS8S8CopyPackBProcessSDot(D, BytesRow, ColumnSums);

            b += ldb * 4;
            D += 32;
            k -= 4;
        }

        if (k > 0) {

            BytesRow[0] = vld1_s8(&b[ldb * 0]);
            BytesRow[1] = (k >= 2) ? vld1_s8(&b[ldb * 1]) : vget_low_s8(ZeroVector);
            BytesRow[2] = (k > 2) ? vld1_s8(&b[ldb * 2]) : vget_low_s8(ZeroVector);
            BytesRow[3] = vget_low_s8(ZeroVector);

            MlasGemmS8S8CopyPackBProcessSDot(D, BytesRow, ColumnSums);

            D += 32;
        }

        //
        // Zero pad the output buffer to a multiple of PackedK if the above
        // processed an odd number of four row bundles.
        //

        if (((CountK - 1) & 7) < 4) {

            vst1q_s8(&D[0], ZeroVector);
            vst1q_s8(&D[16], ZeroVector);

            D += 32;
        }

        vst1q_s32(&ColumnSumBuffer[0], ColumnSums[0]);
        vst1q_s32(&ColumnSumBuffer[4], ColumnSums[1]);
        ColumnSumBuffer += 8;

        B += 8;
        CountN -= 8;
    }

    //
    // Process the remaining columns of matrix B.
    //

    if (CountN > 0) {

        const int8_t* b = reinterpret_cast<const int8_t*>(B);
        size_t k = CountK;
        int8_t PaddedMatrixBData[32];
        int32x4_t ColumnSums[2];

        vst1q_s8(&PaddedMatrixBData[0], ZeroVector);
        vst1q_s8(&PaddedMatrixBData[16], ZeroVector);

        ColumnSums[0] = vmovq_n_s32(0);
        ColumnSums[1] = vmovq_n_s32(0);

        //
        // Interleave rows of matrix B using an intermediate zero padded stack
        // buffer and write to the packed buffer.
        //

        while (k > 0) {

            const int8_t* bcopy0 = &b[ldb * 0];
            const int8_t* bcopy1 = &b[ldb * 1];
            const int8_t* bcopy2 = &b[ldb * 2];
            const int8_t* bcopy3 = &b[ldb * 3];

            if (k >= 4) {

                b += ldb * 4;
                k -= 4;

            } else {

                vst1q_s8(&PaddedMatrixBData[0], ZeroVector);
                vst1q_s8(&PaddedMatrixBData[16], ZeroVector);

                bcopy1 = (k >= 2) ? bcopy1 : &PaddedMatrixBData[24];
                bcopy2 = (k > 2) ? bcopy2 : &PaddedMatrixBData[24];
                bcopy3 = &PaddedMatrixBData[24];

                k = 0;
            }

            int8_t* padded = PaddedMatrixBData;
            int8_t* padded_end = padded + CountN;

            do {
                padded[0] = *bcopy0++;
                padded[8] = *bcopy1++;
                padded[16] = *bcopy2++;
                padded[24] = *bcopy3++;
            } while (++padded < padded_end);

            BytesRow[0] = vld1_s8(&PaddedMatrixBData[0]);
            BytesRow[1] = vld1_s8(&PaddedMatrixBData[8]);
            BytesRow[2] = vld1_s8(&PaddedMatrixBData[16]);
            BytesRow[3] = vld1_s8(&PaddedMatrixBData[24]);

            MlasGemmS8S8CopyPackBProcessSDot(D, BytesRow, ColumnSums);

            D += 32;
        }

        //
        // Zero pad the output buffer to a multiple of PackedK if the above
        // processed an odd number of four row bundles.
        //

        if (((CountK - 1) & 7) < 4) {

            vst1q_s8(&D[0], ZeroVector);
            vst1q_s8(&D[16], ZeroVector);

            D += 32;
        }

        vst1q_s32(&ColumnSumBuffer[0], ColumnSums[0]);
        vst1q_s32(&ColumnSumBuffer[4], ColumnSums[1]);
    }
}

template<>
MLAS_FORCEINLINE
size_t
MlasGemmQuantKernel<MLAS_GEMM_S8S8_KERNEL_SDOT>(
    const MLAS_GEMM_S8S8_KERNEL_SDOT::PackedAType* A,
    const MLAS_GEMM_S8S8_KERNEL_SDOT::PackedBType* B,
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
    return MlasGemmS8S8KernelSDot(A, B, C, PackedCountK, CountM, CountN, ldc,
        RowSumBuffer, ColumnSumBuffer, ZeroPointB, ZeroMode);
}

const MLAS_GEMM_QUANT_DISPATCH MlasGemmS8S8DispatchSdot = {
    MlasGemmQuantOperation<MLAS_GEMM_S8S8_KERNEL_SDOT>,
    MlasGemmQuantPackedOperation<MLAS_GEMM_S8S8_KERNEL_SDOT>,
    MlasGemmQuantCopyPackB<MLAS_GEMM_S8S8_KERNEL_SDOT>,
    MLAS_GEMM_S8S8_KERNEL_SDOT::PackedK,
    MLAS_GEMM_S8S8_KERNEL_SDOT::PackedStrides.K,
};
