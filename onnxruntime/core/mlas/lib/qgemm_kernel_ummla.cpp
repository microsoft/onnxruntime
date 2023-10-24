/*++

Copyright (c) Microsoft Corporation. All rights reserved.
Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the MIT License.

Module Name:

    qgemm_kernel_ummla.cpp

Abstract:

    This module implements ummla QGEMM kernel.

--*/

#include "mlasi.h"
#include "qgemm.h"

//
// Define the prototypes of the NEON UMMLA routines written in assembly.
//

extern "C" {

size_t MLASCALL
MlasGemmU8X8KernelUmmlaZero(const uint8_t* A,
                            const uint8_t* B,
                            int32_t* C,
                            size_t PackedCountK,
                            size_t CountM,
                            size_t CountN,
                            size_t ldc,
                            const int32_t* RowSumVector,
                            const int32_t* ColumnSumVector,
                            const int32_t* ZeroPointB);

size_t MLASCALL
MlasGemmU8X8KernelUmmlaAdd(const uint8_t* A,
                           const uint8_t* B,
                           int32_t* C,
                           size_t PackedCountK,
                           size_t CountM,
                           size_t CountN,
                           size_t ldc,
                           const int32_t* RowSumVector,
                           const int32_t* ColumnSumVector,
                           const int32_t* ZeroPointB);
}

struct MLAS_GEMM_U8X8_KERNEL_UMMLA {
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint8_t OffsetAType;
    typedef uint8_t OffsetBType;

    static constexpr size_t PackedK = 8;
    static constexpr MLAS_GEMM_QUANT_STRIDES Strides{24, 128, 256};
    static constexpr MLAS_GEMM_QUANT_STRIDES PackedStrides{24, 128, 384};
};

constexpr size_t MLAS_GEMM_U8X8_KERNEL_UMMLA::PackedK;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_U8X8_KERNEL_UMMLA::Strides;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_U8X8_KERNEL_UMMLA::PackedStrides;

template <>
MLAS_FORCEINLINE int32_t
MlasGemmQuantFixupZeroPointB<MLAS_GEMM_U8X8_KERNEL_UMMLA>(int32_t ZeroPointB, bool BIsSigned)
{
    if (BIsSigned) {
        ZeroPointB = MLAS_GEMM_U8X8_KERNEL_UMMLA::OffsetBType(ZeroPointB ^ 0x80);
    }

    return ZeroPointB;
}

template <>
void
MlasGemmQuantCopyPackA<MLAS_GEMM_U8X8_KERNEL_UMMLA>(MLAS_GEMM_U8X8_KERNEL_UMMLA::PackedAType* D,
                                                    const uint8_t* A,
                                                    size_t lda,
                                                    size_t CountM,
                                                    size_t CountK,
                                                    int32_t* RowSumBuffer,
                                                    bool AIsSigned)
{
    MLAS_UNREFERENCED_PARAMETER(AIsSigned);
    uint8_t PaddedMatrixAData[64];

    //
    // Process 8 rows of matrix A.
    //
    // MMLA kernels load 8x8 block of A with four vector registers. So A is packed
    // a series of 64 byte vectors where eight rows are interleaved with the
    // following pattern:
    //
    //      [ A0 A1 A2 A3 A4 A5 A6 A7 ]
    //      [ B0 B1 B2 B3 B4 B5 B6 B7 ]
    //      [ C0 C1 C2 C3 C4 C5 C6 C7 ]
    //      [ D0 D1 D2 D3 D4 D5 D6 D7 ]
    //      [ E0 E1 E2 E3 E4 E5 E6 E7 ]
    //      [ F0 F1 F2 F3 F4 F5 F6 F7 ]
    //      [ G0 G1 G2 G3 G4 G5 G6 G7 ]
    //      [ H0 H1 H2 H3 H4 H5 H6 H7 ]
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
            uint64x2_t v0 = vld1q_u64(reinterpret_cast<const uint64_t*>(a0));
            a0 += 16;
            uint64x2_t v1 = vld1q_u64(reinterpret_cast<const uint64_t*>(a1));
            a1 += 16;
            uint64x2_t v2 = vld1q_u64(reinterpret_cast<const uint64_t*>(a2));
            a2 += 16;
            uint64x2_t v3 = vld1q_u64(reinterpret_cast<const uint64_t*>(a3));
            a3 += 16;
            uint64x2_t v4 = vld1q_u64(reinterpret_cast<const uint64_t*>(a4));
            a4 += 16;
            uint64x2_t v5 = vld1q_u64(reinterpret_cast<const uint64_t*>(a5));
            a5 += 16;
            uint64x2_t v6 = vld1q_u64(reinterpret_cast<const uint64_t*>(a6));
            a6 += 16;
            uint64x2_t v7 = vld1q_u64(reinterpret_cast<const uint64_t*>(a7));
            a7 += 16;

            uint64x2_t z0 = vzip1q_u64(v0, v1);
            uint64x2_t z1 = vzip2q_u64(v0, v1);
            uint64x2_t z2 = vzip1q_u64(v2, v3);
            uint64x2_t z3 = vzip2q_u64(v2, v3);

            uint64x2_t z4 = vzip1q_u64(v4, v5);
            uint64x2_t z5 = vzip2q_u64(v4, v5);
            uint64x2_t z6 = vzip1q_u64(v6, v7);
            uint64x2_t z7 = vzip2q_u64(v6, v7);

            vst1q_u8(&D[0], vreinterpretq_u8_u64(z0));
            vst1q_u8(&D[16], vreinterpretq_u8_u64(z2));
            vst1q_u8(&D[32], vreinterpretq_u8_u64(z4));
            vst1q_u8(&D[48], vreinterpretq_u8_u64(z6));
            vst1q_u8(&D[64], vreinterpretq_u8_u64(z1));
            vst1q_u8(&D[80], vreinterpretq_u8_u64(z3));
            vst1q_u8(&D[96], vreinterpretq_u8_u64(z5));
            vst1q_u8(&D[112], vreinterpretq_u8_u64(z7));

            uint32x4_t RowSums0L_pada = vmovq_n_u32(0);
            RowSums0L_pada = vpadalq_u16(RowSums0L_pada, vpaddlq_u8(vreinterpretq_u8_u64(z0)));
            RowSums0L_pada = vpadalq_u16(RowSums0L_pada, vpaddlq_u8(vreinterpretq_u8_u64(z1)));

            uint32x4_t RowSums0L_ext = vextq_u32(RowSums0L_pada, RowSums0L_pada, 1);
            uint32x4_t RowSums0L_add = vaddq_u32(RowSums0L_pada, RowSums0L_ext);
            uint32x2_t RowSums0L = {vdups_laneq_u32(RowSums0L_add, 0),
                                    vdups_laneq_u32(RowSums0L_add, 2)};

            uint32x4_t RowSums0H_pada = vmovq_n_u32(0);
            RowSums0H_pada = vpadalq_u16(RowSums0H_pada, vpaddlq_u8(vreinterpretq_u8_u64(z2)));
            RowSums0H_pada = vpadalq_u16(RowSums0H_pada, vpaddlq_u8(vreinterpretq_u8_u64(z3)));

            uint32x4_t RowSums0H_ext = vextq_u32(RowSums0H_pada, RowSums0H_pada, 1);
            uint32x4_t RowSums0H_add = vaddq_u32(RowSums0H_pada, RowSums0H_ext);
            uint32x2_t RowSums0H = {vdups_laneq_u32(RowSums0H_add, 0),
                                    vdups_laneq_u32(RowSums0H_add, 2)};

            RowSums0 = vaddq_u32(RowSums0, vcombine_u32(RowSums0L, RowSums0H));

            uint32x4_t RowSums1L_pada = vmovq_n_u32(0);
            RowSums1L_pada = vpadalq_u16(RowSums1L_pada, vpaddlq_u8(vreinterpretq_u8_u64(z4)));
            RowSums1L_pada = vpadalq_u16(RowSums1L_pada, vpaddlq_u8(vreinterpretq_u8_u64(z5)));

            uint32x4_t RowSums1L_ext = vextq_u32(RowSums1L_pada, RowSums1L_pada, 1);
            uint32x4_t RowSums1L_add = vaddq_u32(RowSums1L_pada, RowSums1L_ext);
            uint32x2_t RowSums1L = {vdups_laneq_u32(RowSums1L_add, 0),
                                    vdups_laneq_u32(RowSums1L_add, 2)};

            uint32x4_t RowSums1H_pada = vmovq_n_u32(0);
            RowSums1H_pada = vpadalq_u16(RowSums1H_pada, vpaddlq_u8(vreinterpretq_u8_u64(z6)));
            RowSums1H_pada = vpadalq_u16(RowSums1H_pada, vpaddlq_u8(vreinterpretq_u8_u64(z7)));

            uint32x4_t RowSums1H_ext = vextq_u32(RowSums1H_pada, RowSums1H_pada, 1);
            uint32x4_t RowSums1H_add = vaddq_u32(RowSums1H_pada, RowSums1H_ext);
            uint32x2_t RowSums1H = {vdups_laneq_u32(RowSums1H_add, 0),
                                    vdups_laneq_u32(RowSums1H_add, 2)};

            RowSums1 = vaddq_u32(RowSums1, vcombine_u32(RowSums1L, RowSums1H));

            D += 128;
            k -= 16;
        }

        while (k >= 8) {
            uint64x1_t v0 = *reinterpret_cast<const uint64x1_t*>(a0);
            a0 += 8;
            uint64x1_t v1 = *reinterpret_cast<const uint64x1_t*>(a1);
            a1 += 8;
            uint64x1_t v2 = *reinterpret_cast<const uint64x1_t*>(a2);
            a2 += 8;
            uint64x1_t v3 = *reinterpret_cast<const uint64x1_t*>(a3);
            a3 += 8;
            uint64x1_t v4 = *reinterpret_cast<const uint64x1_t*>(a4);
            a4 += 8;
            uint64x1_t v5 = *reinterpret_cast<const uint64x1_t*>(a5);
            a5 += 8;
            uint64x1_t v6 = *reinterpret_cast<const uint64x1_t*>(a6);
            a6 += 8;
            uint64x1_t v7 = *reinterpret_cast<const uint64x1_t*>(a7);
            a7 += 8;

            *reinterpret_cast<uint64x1_t*>(&D[0]) = v0;
            *reinterpret_cast<uint64x1_t*>(&D[8]) = v1;
            *reinterpret_cast<uint64x1_t*>(&D[16]) = v2;
            *reinterpret_cast<uint64x1_t*>(&D[24]) = v3;
            *reinterpret_cast<uint64x1_t*>(&D[32]) = v4;
            *reinterpret_cast<uint64x1_t*>(&D[40]) = v5;
            *reinterpret_cast<uint64x1_t*>(&D[48]) = v6;
            *reinterpret_cast<uint64x1_t*>(&D[56]) = v7;

            uint64x2_t z01 = vcombine_u64(v0, v1);
            uint64x2_t z23 = vcombine_u64(v2, v3);
            uint64x2_t z45 = vcombine_u64(v4, v5);
            uint64x2_t z67 = vcombine_u64(v6, v7);

            uint32x4_t RowSums0L_pada = vmovq_n_u32(0);
            RowSums0L_pada = vpadalq_u16(RowSums0L_pada, vpaddlq_u8(vreinterpretq_u8_u64(z01)));

            uint32x4_t RowSums0L_ext = vextq_u32(RowSums0L_pada, RowSums0L_pada, 1);
            uint32x4_t RowSums0L_add = vaddq_u32(RowSums0L_pada, RowSums0L_ext);
            uint32x2_t RowSums0L = {vdups_laneq_u32(RowSums0L_add, 0),
                                    vdups_laneq_u32(RowSums0L_add, 2)};

            uint32x4_t RowSums0H_pada = vmovq_n_u32(0);
            RowSums0H_pada = vpadalq_u16(RowSums0H_pada, vpaddlq_u8(vreinterpretq_u8_u64(z23)));

            uint32x4_t RowSums0H_ext = vextq_u32(RowSums0H_pada, RowSums0H_pada, 1);
            uint32x4_t RowSums0H_add = vaddq_u32(RowSums0H_pada, RowSums0H_ext);
            uint32x2_t RowSums0H = {vdups_laneq_u32(RowSums0H_add, 0),
                                    vdups_laneq_u32(RowSums0H_add, 2)};

            RowSums0 = vaddq_u32(RowSums0, vcombine_u32(RowSums0L, RowSums0H));

            uint32x4_t RowSums1L_pada = vmovq_n_u32(0);
            RowSums1L_pada = vpadalq_u16(RowSums1L_pada, vpaddlq_u8(vreinterpretq_u8_u64(z45)));

            uint32x4_t RowSums1L_ext = vextq_u32(RowSums1L_pada, RowSums1L_pada, 1);
            uint32x4_t RowSums1L_add = vaddq_u32(RowSums1L_pada, RowSums1L_ext);
            uint32x2_t RowSums1L = {vdups_laneq_u32(RowSums1L_add, 0),
                                    vdups_laneq_u32(RowSums1L_add, 2)};

            uint32x4_t RowSums1H_pada = vmovq_n_u32(0);
            RowSums1H_pada = vpadalq_u16(RowSums1H_pada, vpaddlq_u8(vreinterpretq_u8_u64(z67)));

            uint32x4_t RowSums1H_ext = vextq_u32(RowSums1H_pada, RowSums1H_pada, 1);
            uint32x4_t RowSums1H_add = vaddq_u32(RowSums1H_pada, RowSums1H_ext);
            uint32x2_t RowSums1H = {vdups_laneq_u32(RowSums1H_add, 0),
                                    vdups_laneq_u32(RowSums1H_add, 2)};

            RowSums1 = vaddq_u32(RowSums1, vcombine_u32(RowSums1L, RowSums1H));

            D += 64;
            k -= 8;
        }

        if (k > 0) {
            //
            // zero pad the remaining columns to 8
            //
            uint8_t* d = D;

            vst1q_u8(d, vmovq_n_u8(0));
            vst1q_u8(&d[16], vmovq_n_u8(0));
            vst1q_u8(&d[32], vmovq_n_u8(0));
            vst1q_u8(&d[48], vmovq_n_u8(0));

            while (k > 0) {
                d[0] = *a0++;
                d[8] = *a1++;
                d[16] = *a2++;
                d[24] = *a3++;
                d[32] = *a4++;
                d[40] = *a5++;
                d[48] = *a6++;
                d[56] = *a7++;
                d += 1;
                k -= 1;
            }
            d = D;
            uint64x1_t v0 = *reinterpret_cast<const uint64x1_t*>(d);
            d = d + 8;
            uint64x1_t v1 = *reinterpret_cast<const uint64x1_t*>(d);
            d = d + 8;
            uint64x1_t v2 = *reinterpret_cast<const uint64x1_t*>(d);
            d = d + 8;
            uint64x1_t v3 = *reinterpret_cast<const uint64x1_t*>(d);
            d = d + 8;
            uint64x1_t v4 = *reinterpret_cast<const uint64x1_t*>(d);
            d = d + 8;
            uint64x1_t v5 = *reinterpret_cast<const uint64x1_t*>(d);
            d = d + 8;
            uint64x1_t v6 = *reinterpret_cast<const uint64x1_t*>(d);
            d = d + 8;
            uint64x1_t v7 = *reinterpret_cast<const uint64x1_t*>(d);
            d = d + 8;

            uint64x2_t z01 = vcombine_u64(v0, v1);
            uint64x2_t z23 = vcombine_u64(v2, v3);
            uint64x2_t z45 = vcombine_u64(v4, v5);
            uint64x2_t z67 = vcombine_u64(v6, v7);

            uint32x4_t RowSums0L_pada = vmovq_n_u32(0);
            RowSums0L_pada = vpadalq_u16(RowSums0L_pada, vpaddlq_u8(vreinterpretq_u8_u64(z01)));

            uint32x4_t RowSums0L_ext = vextq_u32(RowSums0L_pada, RowSums0L_pada, 1);
            uint32x4_t RowSums0L_add = vaddq_u32(RowSums0L_pada, RowSums0L_ext);
            uint32x2_t RowSums0L = {vdups_laneq_u32(RowSums0L_add, 0),
                                    vdups_laneq_u32(RowSums0L_add, 2)};

            uint32x4_t RowSums0H_pada = vmovq_n_u32(0);
            RowSums0H_pada = vpadalq_u16(RowSums0H_pada, vpaddlq_u8(vreinterpretq_u8_u64(z23)));

            uint32x4_t RowSums0H_ext = vextq_u32(RowSums0H_pada, RowSums0H_pada, 1);
            uint32x4_t RowSums0H_add = vaddq_u32(RowSums0H_pada, RowSums0H_ext);
            uint32x2_t RowSums0H = {vdups_laneq_u32(RowSums0H_add, 0),
                                    vdups_laneq_u32(RowSums0H_add, 2)};

            RowSums0 = vaddq_u32(RowSums0, vcombine_u32(RowSums0L, RowSums0H));

            uint32x4_t RowSums1L_pada = vmovq_n_u32(0);
            RowSums1L_pada = vpadalq_u16(RowSums1L_pada, vpaddlq_u8(vreinterpretq_u8_u64(z45)));

            uint32x4_t RowSums1L_ext = vextq_u32(RowSums1L_pada, RowSums1L_pada, 1);
            uint32x4_t RowSums1L_add = vaddq_u32(RowSums1L_pada, RowSums1L_ext);
            uint32x2_t RowSums1L = {vdups_laneq_u32(RowSums1L_add, 0),
                                    vdups_laneq_u32(RowSums1L_add, 2)};

            uint32x4_t RowSums1H_pada = vmovq_n_u32(0);
            RowSums1H_pada = vpadalq_u16(RowSums1H_pada, vpaddlq_u8(vreinterpretq_u8_u64(z67)));

            uint32x4_t RowSums1H_ext = vextq_u32(RowSums1H_pada, RowSums1H_pada, 1);
            uint32x4_t RowSums1H_add = vaddq_u32(RowSums1H_pada, RowSums1H_ext);
            uint32x2_t RowSums1H = {vdups_laneq_u32(RowSums1H_add, 0),
                                    vdups_laneq_u32(RowSums1H_add, 2)};

            RowSums1 = vaddq_u32(RowSums1, vcombine_u32(RowSums1L, RowSums1H));

            D += 64;
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
    // The buffer is packed as a series of 32 byte vectors where four rows are
    // interleaved with the following pattern:
    //
    //      [ A0 A1 A2 A3 A4 A5 A6 A7 ]
    //      [ B0 B1 B2 B3 B4 B5 B6 B7 ]
    //      [ C0 C1 C2 C3 C4 C5 C6 C7 ]
    //      [ D0 D1 D2 D3 D4 D5 D6 D7 ]
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
            uint64x2_t v0 = vld1q_u64(reinterpret_cast<const uint64_t*>(a0));
            a0 += 16;
            uint64x2_t v1 = vld1q_u64(reinterpret_cast<const uint64_t*>(a1));
            a1 += 16;
            uint64x2_t v2 = vld1q_u64(reinterpret_cast<const uint64_t*>(a2));
            a2 += 16;
            uint64x2_t v3 = vld1q_u64(reinterpret_cast<const uint64_t*>(a3));
            a3 += 16;

            uint64x2_t z0 = vzip1q_u64(v0, v1);
            uint64x2_t z1 = vzip2q_u64(v0, v1);
            uint64x2_t z2 = vzip1q_u64(v2, v3);
            uint64x2_t z3 = vzip2q_u64(v2, v3);

            vst1q_u8(&D[0], vreinterpretq_u8_u64(z0));
            vst1q_u8(&D[16], vreinterpretq_u8_u64(z2));
            vst1q_u8(&D[32], vreinterpretq_u8_u64(z1));
            vst1q_u8(&D[48], vreinterpretq_u8_u64(z3));

            uint32x4_t RowSumsL_pada = vmovq_n_u32(0);
            RowSumsL_pada = vpadalq_u16(RowSumsL_pada, vpaddlq_u8(vreinterpretq_u8_u64(z0)));
            RowSumsL_pada = vpadalq_u16(RowSumsL_pada, vpaddlq_u8(vreinterpretq_u8_u64(z1)));

            uint32x4_t RowSumsL_ext = vextq_u32(RowSumsL_pada, RowSumsL_pada, 1);
            uint32x4_t RowSumsL_add = vaddq_u32(RowSumsL_pada, RowSumsL_ext);
            uint32x2_t RowSumsL = {vdups_laneq_u32(RowSumsL_add, 0),
                                   vdups_laneq_u32(RowSumsL_add, 2)};

            uint32x4_t RowSumsH_pada = vmovq_n_u32(0);
            RowSumsH_pada = vpadalq_u16(RowSumsH_pada, vpaddlq_u8(vreinterpretq_u8_u64(z2)));
            RowSumsH_pada = vpadalq_u16(RowSumsH_pada, vpaddlq_u8(vreinterpretq_u8_u64(z3)));

            uint32x4_t RowSumsH_ext = vextq_u32(RowSumsH_pada, RowSumsH_pada, 1);
            uint32x4_t RowSumsH_add = vaddq_u32(RowSumsH_pada, RowSumsH_ext);
            uint32x2_t RowSumsH = {vdups_laneq_u32(RowSumsH_add, 0),
                                   vdups_laneq_u32(RowSumsH_add, 2)};

            RowSums = vaddq_u32(RowSums, vcombine_u32(RowSumsL, RowSumsH));

            D += 64;
            k -= 16;
        }

        while (k >= 8) {
            uint64x1_t v0 = *reinterpret_cast<const uint64x1_t*>(a0);
            a0 += 8;
            uint64x1_t v1 = *reinterpret_cast<const uint64x1_t*>(a1);
            a1 += 8;
            uint64x1_t v2 = *reinterpret_cast<const uint64x1_t*>(a2);
            a2 += 8;
            uint64x1_t v3 = *reinterpret_cast<const uint64x1_t*>(a3);
            a3 += 8;

            *reinterpret_cast<uint64x1_t*>(&D[0]) = v0;
            *reinterpret_cast<uint64x1_t*>(&D[8]) = v1;
            *reinterpret_cast<uint64x1_t*>(&D[16]) = v2;
            *reinterpret_cast<uint64x1_t*>(&D[24]) = v3;

            uint64x2_t z01 = vcombine_u64(v0, v1);
            uint64x2_t z23 = vcombine_u64(v2, v3);

            uint32x4_t RowSumsL_pada = vmovq_n_u32(0);
            RowSumsL_pada = vpadalq_u16(RowSumsL_pada, vpaddlq_u8(vreinterpretq_u8_u64(z01)));

            uint32x4_t RowSumsL_ext = vextq_u32(RowSumsL_pada, RowSumsL_pada, 1);
            uint32x4_t RowSumsL_add = vaddq_u32(RowSumsL_pada, RowSumsL_ext);
            uint32x2_t RowSumsL = {vdups_laneq_u32(RowSumsL_add, 0),
                                   vdups_laneq_u32(RowSumsL_add, 2)};

            uint32x4_t RowSumsH_pada = vmovq_n_u32(0);
            RowSumsH_pada = vpadalq_u16(RowSumsH_pada, vpaddlq_u8(vreinterpretq_u8_u64(z23)));

            uint32x4_t RowSumsH_ext = vextq_u32(RowSumsH_pada, RowSumsH_pada, 1);
            uint32x4_t RowSumsH_add = vaddq_u32(RowSumsH_pada, RowSumsH_ext);
            uint32x2_t RowSumsH = {vdups_laneq_u32(RowSumsH_add, 0),
                                   vdups_laneq_u32(RowSumsH_add, 2)};

            RowSums = vaddq_u32(RowSums, vcombine_u32(RowSumsL, RowSumsH));

            D += 32;
            k -= 8;
        }

        if (k > 0) {
            //
            // Copy the remaining bytes with zero padding.
            //
            uint8_t* d = D;

            vst1q_u8(d, vmovq_n_u8(0));
            vst1q_u8(&d[16], vmovq_n_u8(0));

            while (k > 0) {
                d[0] = *a0++;
                d[8] = *a1++;
                d[16] = *a2++;
                d[24] = *a3++;
                d += 1;
                k -= 1;
            }

            d = D;
            uint64x1_t v0 = *reinterpret_cast<const uint64x1_t*>(d);
            d = d + 8;
            uint64x1_t v1 = *reinterpret_cast<const uint64x1_t*>(d);
            d = d + 8;
            uint64x1_t v2 = *reinterpret_cast<const uint64x1_t*>(d);
            d = d + 8;
            uint64x1_t v3 = *reinterpret_cast<const uint64x1_t*>(d);
            d = d + 8;

            uint64x2_t z01 = vcombine_u64(v0, v1);
            uint64x2_t z23 = vcombine_u64(v2, v3);

            uint32x4_t RowSums0L_pada = vmovq_n_u32(0);
            RowSums0L_pada = vpadalq_u16(RowSums0L_pada, vpaddlq_u8(vreinterpretq_u8_u64(z01)));

            uint32x4_t RowSums0L_ext = vextq_u32(RowSums0L_pada, RowSums0L_pada, 1);
            uint32x4_t RowSums0L_add = vaddq_u32(RowSums0L_pada, RowSums0L_ext);
            uint32x2_t RowSums0L = {vdups_laneq_u32(RowSums0L_add, 0),
                                    vdups_laneq_u32(RowSums0L_add, 2)};

            uint32x4_t RowSums0H_pada = vmovq_n_u32(0);
            RowSums0H_pada = vpadalq_u16(RowSums0H_pada, vpaddlq_u8(vreinterpretq_u8_u64(z23)));

            uint32x4_t RowSums0H_ext = vextq_u32(RowSums0H_pada, RowSums0H_pada, 1);
            uint32x4_t RowSums0H_add = vaddq_u32(RowSums0H_pada, RowSums0H_ext);
            uint32x2_t RowSums0H = {vdups_laneq_u32(RowSums0H_add, 0),
                                    vdups_laneq_u32(RowSums0H_add, 2)};

            RowSums = vaddq_u32(RowSums, vcombine_u32(RowSums0L, RowSums0H));

            D += 32;
        }

        vst1q_s32(RowSumBuffer, vreinterpretq_s32_u32(RowSums));
        RowSumBuffer += 4;

        A = A + lda * 4;
        CountM -= 4;
    }

    //
    // Process two rows of matrix A.
    //
    // The buffer is packed as a series of 16 byte vectors where two rows are
    // interleaved with the following pattern:
    //
    //      [ A0 A1 A2 A3 A4 A5 A6 A7 ]
    //      [ B0 B1 B2 B3 B4 B5 B6 B7 ]
    //
    // This pattern is repeated (CountK / 8) times.
    //
    // If CountK is not aligned to a multiple of eight, then the vector is padded
    // with zeroes.
    //

    if (CountM >= 2) {
        const uint8_t* a0 = A;
        const uint8_t* a1 = a0 + lda;

        size_t k = CountK;
        uint32x2_t RowSums = vmov_n_u32(0);

        while (k >= 16) {
            uint64x2_t v0 = vld1q_u64(reinterpret_cast<const uint64_t*>(a0));
            a0 += 16;
            uint64x2_t v1 = vld1q_u64(reinterpret_cast<const uint64_t*>(a1));
            a1 += 16;

            uint64x2_t z0 = vzip1q_u64(v0, v1);
            uint64x2_t z1 = vzip2q_u64(v0, v1);

            vst1q_u8(&D[0], vreinterpretq_u8_u64(z0));
            vst1q_u8(&D[16], vreinterpretq_u8_u64(z1));

            uint32x4_t RowSumsL_pada = vmovq_n_u32(0);
            RowSumsL_pada = vpadalq_u16(RowSumsL_pada, vpaddlq_u8(vreinterpretq_u8_u64(z0)));
            RowSumsL_pada = vpadalq_u16(RowSumsL_pada, vpaddlq_u8(vreinterpretq_u8_u64(z1)));

            uint32x4_t RowSumsL_ext = vextq_u32(RowSumsL_pada, RowSumsL_pada, 1);
            uint32x4_t RowSumsL_add = vaddq_u32(RowSumsL_pada, RowSumsL_ext);
            uint32x2_t RowSumsL = {vdups_laneq_u32(RowSumsL_add, 0),
                                   vdups_laneq_u32(RowSumsL_add, 2)};

            RowSums = vadd_u32(RowSums, RowSumsL);

            D += 32;
            k -= 16;
        }

        while (k >= 8) {
            uint64x1_t v0 = *reinterpret_cast<const uint64x1_t*>(a0);
            a0 += 8;
            uint64x1_t v1 = *reinterpret_cast<const uint64x1_t*>(a1);
            a1 += 8;

            *reinterpret_cast<uint64x1_t*>(&D[0]) = v0;
            *reinterpret_cast<uint64x1_t*>(&D[8]) = v1;

            uint64x2_t z01 = vcombine_u64(v0, v1);
            uint32x4_t RowSumsL_pada = vmovq_n_u32(0);
            RowSumsL_pada = vpadalq_u16(RowSumsL_pada, vpaddlq_u8(vreinterpretq_u8_u64(z01)));

            uint32x4_t RowSumsL_ext = vextq_u32(RowSumsL_pada, RowSumsL_pada, 1);
            uint32x4_t RowSumsL_add = vaddq_u32(RowSumsL_pada, RowSumsL_ext);
            uint32x2_t RowSumsL = {vdups_laneq_u32(RowSumsL_add, 0),
                                   vdups_laneq_u32(RowSumsL_add, 2)};

            RowSums = vadd_u32(RowSums, RowSumsL);

            D += 16;
            k -= 8;
        }

        if (k > 0) {
            //
            // Zero pad the remaining elements to make 8 columns.
            //

            uint8_t* d = PaddedMatrixAData;
            vst1q_u8(PaddedMatrixAData, vmovq_n_u8(0));

            while (k > 0) {
                d[0] = *a0++;
                d[8] = *a1++;

                d += 1;
                k -= 1;
            }

            d = PaddedMatrixAData;
            uint64x1_t v0 = *reinterpret_cast<const uint64x1_t*>(d);
            d = d + 8;
            uint64x1_t v1 = *reinterpret_cast<const uint64x1_t*>(d);
            d = d + 8;

            uint64x2_t z01 = vcombine_u64(v0, v1);
            uint32x4_t RowSumsL_pada = vmovq_n_u32(0);
            RowSumsL_pada = vpadalq_u16(RowSumsL_pada, vpaddlq_u8(vreinterpretq_u8_u64(z01)));

            uint32x4_t RowSumsL_ext = vextq_u32(RowSumsL_pada, RowSumsL_pada, 1);
            uint32x4_t RowSumsL_add = vaddq_u32(RowSumsL_pada, RowSumsL_ext);
            uint32x2_t RowSumsL = {vdups_laneq_u32(RowSumsL_add, 0),
                                   vdups_laneq_u32(RowSumsL_add, 2)};

            RowSums = vadd_u32(RowSums, RowSumsL);

            uint8x16_t PackedVector = vld1q_u8(PaddedMatrixAData);
            vst1q_u8(D, PackedVector);

            D += 16;
        }

        vst1_s32(RowSumBuffer, vreinterpret_s32_u32(RowSums));
        RowSumBuffer += 2;

        A = A + lda * 2;
        CountM -= 2;
    }

    //
    // Process one row of matrix A.
    //
    // The buffer is packed as a series of 8 byte with the following pattern:
    //
    //      [ A0 A1 A2 A3 A4 A5 A6 A7 ]
    //
    // This pattern is repeated (CountK / 8) times.
    //
    // If CountK is not aligned to a multiple of 8, then the vector is padded
    // with zeroes.
    //

    if (CountM > 0) {
        // No need to pad the rows to 2, the .S takes care of zero pdding
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

        *RowSumBuffer = int32_t(vaddvq_u32(RowSums));
    }
}

MLAS_FORCEINLINE
void
MlasGemmU8X8CopyPackBProcessUmmla(MLAS_GEMM_U8X8_KERNEL_UMMLA::PackedBType* D,
                                  uint8x8_t BytesRow[8],
                                  uint8x16_t BitFlipVector,
                                  uint32x4_t ColumnSums[2])
{
    uint8x16_t v02 = veorq_u8(vcombine_u8(BytesRow[0], BytesRow[2]), BitFlipVector);
    uint8x16_t v13 = veorq_u8(vcombine_u8(BytesRow[1], BytesRow[3]), BitFlipVector);

    uint8x16_t v46 = veorq_u8(vcombine_u8(BytesRow[4], BytesRow[6]), BitFlipVector);
    uint8x16_t v57 = veorq_u8(vcombine_u8(BytesRow[5], BytesRow[7]), BitFlipVector);

    uint8x16x2_t zw1 = vzipq_u8(v02, v13);
    uint16x8x2_t zd1 =
        vzipq_u16(vreinterpretq_u16_u8(zw1.val[0]), vreinterpretq_u16_u8(zw1.val[1]));

    uint8x16x2_t zw2 = vzipq_u8(v46, v57);
    uint16x8x2_t zd2 =
        vzipq_u16(vreinterpretq_u16_u8(zw2.val[0]), vreinterpretq_u16_u8(zw2.val[1]));

    uint32x4x2_t zd3 =
        vzipq_u32(vreinterpretq_u32_u16(zd1.val[0]), vreinterpretq_u32_u16(zd2.val[0]));
    uint32x4x2_t zd4 =
        vzipq_u32(vreinterpretq_u32_u16(zd1.val[1]), vreinterpretq_u32_u16(zd2.val[1]));

    vst1q_u8(&D[0], vreinterpretq_u8_u32(zd3.val[0]));
    vst1q_u8(&D[16], vreinterpretq_u8_u32(zd3.val[1]));
    vst1q_u8(&D[32], vreinterpretq_u8_u32(zd4.val[0]));
    vst1q_u8(&D[48], vreinterpretq_u8_u32(zd4.val[1]));

    uint32x4_t ColSums0L_pada = vmovq_n_u32(0);
    ColSums0L_pada = vpadalq_u16(ColSums0L_pada, vpaddlq_u8(vreinterpretq_u8_u32(zd3.val[0])));
    uint32x4_t ColSums0L_ext = vextq_u32(ColSums0L_pada, ColSums0L_pada, 1);
    uint32x4_t ColSums0L_add = vaddq_u32(ColSums0L_pada, ColSums0L_ext);
    uint32x2_t ColSums0L = {vdups_laneq_u32(ColSums0L_add, 0), vdups_laneq_u32(ColSums0L_add, 2)};

    uint32x4_t ColSums0H_pada = vmovq_n_u32(0);
    ColSums0H_pada = vpadalq_u16(ColSums0H_pada, vpaddlq_u8(vreinterpretq_u8_u32(zd3.val[1])));
    uint32x4_t ColSums0H_ext = vextq_u32(ColSums0H_pada, ColSums0H_pada, 1);
    uint32x4_t ColSums0H_add = vaddq_u32(ColSums0H_pada, ColSums0H_ext);
    uint32x2_t ColSums0H = {vdups_laneq_u32(ColSums0H_add, 0), vdups_laneq_u32(ColSums0H_add, 2)};

    ColumnSums[0] = vaddq_u32(ColumnSums[0], vcombine_u32(ColSums0L, ColSums0H));

    uint32x4_t ColSums1L_pada = vmovq_n_u32(0);
    ColSums1L_pada = vpadalq_u16(ColSums1L_pada, vpaddlq_u8(vreinterpretq_u8_u32(zd4.val[0])));
    uint32x4_t ColSums1L_ext = vextq_u32(ColSums1L_pada, ColSums1L_pada, 1);
    uint32x4_t ColSums1L_add = vaddq_u32(ColSums1L_pada, ColSums1L_ext);
    uint32x2_t ColSums1L = {vdups_laneq_u32(ColSums1L_add, 0), vdups_laneq_u32(ColSums1L_add, 2)};

    uint32x4_t ColSums1H_pada = vmovq_n_u32(0);
    ColSums1H_pada = vpadalq_u16(ColSums1H_pada, vpaddlq_u8(vreinterpretq_u8_u32(zd4.val[1])));
    uint32x4_t ColSums1H_ext = vextq_u32(ColSums1H_pada, ColSums1H_pada, 1);
    uint32x4_t ColSums1H_add = vaddq_u32(ColSums1H_pada, ColSums1H_ext);
    uint32x2_t ColSums1H = {vdups_laneq_u32(ColSums1H_add, 0), vdups_laneq_u32(ColSums1H_add, 2)};

    ColumnSums[1] = vaddq_u32(ColumnSums[1], vcombine_u32(ColSums1L, ColSums1H));
}

template <>
void
MlasGemmQuantCopyPackB<MLAS_GEMM_U8X8_KERNEL_UMMLA>(MLAS_GEMM_U8X8_KERNEL_UMMLA::PackedBType* D,
                                                    const uint8_t* B,
                                                    size_t ldb,
                                                    size_t CountN,
                                                    size_t CountK,
                                                    int32_t* ColumnSumBuffer,
                                                    bool BIsSigned)
{
    const uint8x16_t BitFlipVector = vdupq_n_u8(BIsSigned ? 0x80 : 0);
    uint8x8_t BytesRow[8];

    //
    // Copy data from matrix B into the destination buffer 8x2 blocks at a
    // time.
    //
    //
    while (CountN >= 8) {
        const uint8_t* b = B;
        size_t k = CountK;
        uint32x4_t ColumnSums[2];
        ColumnSums[0] = vmovq_n_u32(0);
        ColumnSums[1] = vmovq_n_u32(0);

        while (k >= 8) {
            BytesRow[0] = vld1_u8(&b[ldb * 0]);
            BytesRow[1] = vld1_u8(&b[ldb * 1]);
            BytesRow[2] = vld1_u8(&b[ldb * 2]);
            BytesRow[3] = vld1_u8(&b[ldb * 3]);
            BytesRow[4] = vld1_u8(&b[ldb * 4]);
            BytesRow[5] = vld1_u8(&b[ldb * 5]);
            BytesRow[6] = vld1_u8(&b[ldb * 6]);
            BytesRow[7] = vld1_u8(&b[ldb * 7]);

            MlasGemmU8X8CopyPackBProcessUmmla(D, BytesRow, BitFlipVector, ColumnSums);

            D += 64;
            b += ldb * 8;
            k -= 8;
        }

        if (k > 0) {
            // Pad k to 8

            BytesRow[0] = vld1_u8(&b[ldb * 0]);
            BytesRow[1] = (k >= 2) ? vld1_u8(&b[ldb * 1]) : vget_low_u8(BitFlipVector);
            BytesRow[2] = (k >= 3) ? vld1_u8(&b[ldb * 2]) : vget_low_u8(BitFlipVector);
            BytesRow[3] = (k >= 4) ? vld1_u8(&b[ldb * 3]) : vget_low_u8(BitFlipVector);
            BytesRow[4] = (k >= 5) ? vld1_u8(&b[ldb * 4]) : vget_low_u8(BitFlipVector);
            BytesRow[5] = (k >= 6) ? vld1_u8(&b[ldb * 5]) : vget_low_u8(BitFlipVector);
            BytesRow[6] = (k >= 7) ? vld1_u8(&b[ldb * 6]) : vget_low_u8(BitFlipVector);
            BytesRow[7] = vget_low_u8(BitFlipVector);

            MlasGemmU8X8CopyPackBProcessUmmla(D, BytesRow, BitFlipVector, ColumnSums);

            D += 64;
        }

        // Zero pad the output buffer to a multiple of PackedK if the above
        // processed an odd number of four row bundles.
        //
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
        uint8_t PaddedMatrixBData[64];
        uint32x4_t ColumnSums[2];

        vst1q_u8(&PaddedMatrixBData[0], BitFlipVector);
        vst1q_u8(&PaddedMatrixBData[16], BitFlipVector);
        vst1q_u8(&PaddedMatrixBData[32], BitFlipVector);
        vst1q_u8(&PaddedMatrixBData[48], BitFlipVector);

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
            const uint8_t* bcopy4 = &b[ldb * 4];
            const uint8_t* bcopy5 = &b[ldb * 5];
            const uint8_t* bcopy6 = &b[ldb * 6];
            const uint8_t* bcopy7 = &b[ldb * 7];

            if (k >= 8) {
                b += ldb * 8;
                k -= 8;

            } else {
                vst1q_u8(&PaddedMatrixBData[0], BitFlipVector);
                vst1q_u8(&PaddedMatrixBData[16], BitFlipVector);
                vst1q_u8(&PaddedMatrixBData[32], BitFlipVector);
                vst1q_u8(&PaddedMatrixBData[48], BitFlipVector);

                bcopy1 = (k >= 2) ? bcopy1 : &PaddedMatrixBData[56];
                bcopy2 = (k >= 3) ? bcopy2 : &PaddedMatrixBData[56];
                bcopy3 = (k >= 4) ? bcopy3 : &PaddedMatrixBData[56];
                bcopy4 = (k >= 5) ? bcopy4 : &PaddedMatrixBData[56];
                bcopy5 = (k >= 6) ? bcopy5 : &PaddedMatrixBData[56];
                bcopy6 = (k >= 7) ? bcopy6 : &PaddedMatrixBData[56];
                bcopy7 = &PaddedMatrixBData[56];

                k = 0;
            }

            uint8_t* padded = PaddedMatrixBData;
            uint8_t* padded_end = padded + CountN;
            do {
                padded[0] = *bcopy0++;
                padded[8] = *bcopy1++;
                padded[16] = *bcopy2++;
                padded[24] = *bcopy3++;
                padded[32] = *bcopy4++;
                padded[40] = *bcopy5++;
                padded[48] = *bcopy6++;
                padded[56] = *bcopy7++;

            } while (++padded < padded_end);

            BytesRow[0] = vld1_u8(&PaddedMatrixBData[0]);
            BytesRow[1] = vld1_u8(&PaddedMatrixBData[8]);
            BytesRow[2] = vld1_u8(&PaddedMatrixBData[16]);
            BytesRow[3] = vld1_u8(&PaddedMatrixBData[24]);
            BytesRow[4] = vld1_u8(&PaddedMatrixBData[32]);
            BytesRow[5] = vld1_u8(&PaddedMatrixBData[40]);
            BytesRow[6] = vld1_u8(&PaddedMatrixBData[48]);
            BytesRow[7] = vld1_u8(&PaddedMatrixBData[56]);

            MlasGemmU8X8CopyPackBProcessUmmla(D, BytesRow, BitFlipVector, ColumnSums);

            D += 64;
        }

        vst1q_s32(&ColumnSumBuffer[0], vreinterpretq_s32_u32(ColumnSums[0]));
        vst1q_s32(&ColumnSumBuffer[4], vreinterpretq_s32_u32(ColumnSums[1]));
    }
}

template <>
MLAS_FORCEINLINE size_t
MlasGemmQuantKernel<MLAS_GEMM_U8X8_KERNEL_UMMLA>(const MLAS_GEMM_U8X8_KERNEL_UMMLA::PackedAType* A,
                                                 const MLAS_GEMM_U8X8_KERNEL_UMMLA::PackedBType* B,
                                                 int32_t* C,
                                                 size_t PackedCountK,
                                                 size_t CountM,
                                                 size_t CountN,
                                                 size_t ldc,
                                                 const int32_t* RowSumBuffer,
                                                 const int32_t* ColumnSumBuffer,
                                                 const int32_t* ZeroPointB,
                                                 bool ZeroMode)
{
    size_t RowsHandled;

    if (ZeroMode) {
        RowsHandled = MlasGemmU8X8KernelUmmlaZero(A, B, C, PackedCountK, CountM, CountN, ldc,
                                                  RowSumBuffer, ColumnSumBuffer, ZeroPointB);
    } else {
        RowsHandled = MlasGemmU8X8KernelUmmlaAdd(A, B, C, PackedCountK, CountM, CountN, ldc,
                                                 RowSumBuffer, ColumnSumBuffer, ZeroPointB);
    }

    return RowsHandled;
}

const MLAS_GEMM_QUANT_DISPATCH MlasGemmU8X8DispatchUmmla = {
    MlasGemmQuantOperation<MLAS_GEMM_U8X8_KERNEL_UMMLA>,
    MlasGemmQuantPackedOperation<MLAS_GEMM_U8X8_KERNEL_UMMLA>,
    MlasGemmQuantCopyPackB<MLAS_GEMM_U8X8_KERNEL_UMMLA>,
    MLAS_GEMM_U8X8_KERNEL_UMMLA::PackedK,
    MLAS_GEMM_U8X8_KERNEL_UMMLA::PackedStrides.K,
    8};
