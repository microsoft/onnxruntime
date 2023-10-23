/*++

Copyright (c) Microsoft Corporation. All rights reserved.
Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the MIT License.

Module Name:

    qgemm_kernel_smmla.cpp

Abstract:

    This module implements smmla QGEMM kernel.

--*/

#include "mlasi.h"
#include "qgemm.h"

//
// Define the prototypes of the NEON SMMLA routines written in assembly.
//

extern "C" {

size_t MLASCALL
MlasGemmS8S8KernelSmmlaZero(const uint8_t* A,
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
MlasGemmS8S8KernelSmmlaAdd(const uint8_t* A,
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

struct MLAS_GEMM_S8S8_KERNEL_SMMLA {
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef int8_t OffsetAType;
    typedef int8_t OffsetBType;

    static constexpr size_t PackedK = 8;
    static constexpr MLAS_GEMM_QUANT_STRIDES Strides{24, 128, 256};
    static constexpr MLAS_GEMM_QUANT_STRIDES PackedStrides{24, 128, 384};
};

constexpr size_t MLAS_GEMM_S8S8_KERNEL_SMMLA::PackedK;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_S8S8_KERNEL_SMMLA::Strides;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_S8S8_KERNEL_SMMLA::PackedStrides;

template <>
MLAS_FORCEINLINE int32_t
MlasGemmQuantFixupZeroPointB<MLAS_GEMM_S8S8_KERNEL_SMMLA>(int32_t ZeroPointB, bool BIsSigned)
{
    MLAS_UNREFERENCED_PARAMETER(BIsSigned);
    return ZeroPointB;
}

template <>
void
MlasGemmQuantCopyPackA<MLAS_GEMM_S8S8_KERNEL_SMMLA>(
    MLAS_GEMM_S8S8_KERNEL_SMMLA::PackedAType* D_uint8_t,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer,
    bool AIsSigned)
{
    int8_t* D = reinterpret_cast<int8_t*>(D_uint8_t);
    MLAS_UNREFERENCED_PARAMETER(AIsSigned);
    int8_t PaddedMatrixAData[64];

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
            int64x2_t v0 = vld1q_s64(reinterpret_cast<const int64_t*>(a0));
            a0 += 16;
            int64x2_t v1 = vld1q_s64(reinterpret_cast<const int64_t*>(a1));
            a1 += 16;
            int64x2_t v2 = vld1q_s64(reinterpret_cast<const int64_t*>(a2));
            a2 += 16;
            int64x2_t v3 = vld1q_s64(reinterpret_cast<const int64_t*>(a3));
            a3 += 16;
            int64x2_t v4 = vld1q_s64(reinterpret_cast<const int64_t*>(a4));
            a4 += 16;
            int64x2_t v5 = vld1q_s64(reinterpret_cast<const int64_t*>(a5));
            a5 += 16;
            int64x2_t v6 = vld1q_s64(reinterpret_cast<const int64_t*>(a6));
            a6 += 16;
            int64x2_t v7 = vld1q_s64(reinterpret_cast<const int64_t*>(a7));
            a7 += 16;

            int64x2_t z0 = vzip1q_s64(v0, v1);
            int64x2_t z1 = vzip2q_s64(v0, v1);
            int64x2_t z2 = vzip1q_s64(v2, v3);
            int64x2_t z3 = vzip2q_s64(v2, v3);

            int64x2_t z4 = vzip1q_s64(v4, v5);
            int64x2_t z5 = vzip2q_s64(v4, v5);
            int64x2_t z6 = vzip1q_s64(v6, v7);
            int64x2_t z7 = vzip2q_s64(v6, v7);

            vst1q_s8(&D[0], vreinterpretq_s8_s64(z0));
            vst1q_s8(&D[16], vreinterpretq_s8_s64(z2));
            vst1q_s8(&D[32], vreinterpretq_s8_s64(z4));
            vst1q_s8(&D[48], vreinterpretq_s8_s64(z6));
            vst1q_s8(&D[64], vreinterpretq_s8_s64(z1));
            vst1q_s8(&D[80], vreinterpretq_s8_s64(z3));
            vst1q_s8(&D[96], vreinterpretq_s8_s64(z5));
            vst1q_s8(&D[112], vreinterpretq_s8_s64(z7));

            int32x4_t RowSums0L_pada = vmovq_n_s32(0);
            RowSums0L_pada = vpadalq_s16(RowSums0L_pada, vpaddlq_s8(vreinterpretq_s8_s64(z0)));
            RowSums0L_pada = vpadalq_s16(RowSums0L_pada, vpaddlq_s8(vreinterpretq_s8_s64(z1)));

            int32x4_t RowSums0L_ext = vextq_s32(RowSums0L_pada, RowSums0L_pada, 1);
            int32x4_t RowSums0L_add = vaddq_s32(RowSums0L_pada, RowSums0L_ext);
            int32x2_t RowSums0L = {vdups_laneq_s32(RowSums0L_add, 0),
                                   vdups_laneq_s32(RowSums0L_add, 2)};

            int32x4_t RowSums0H_pada = vmovq_n_s32(0);
            RowSums0H_pada = vpadalq_s16(RowSums0H_pada, vpaddlq_s8(vreinterpretq_s8_s64(z2)));
            RowSums0H_pada = vpadalq_s16(RowSums0H_pada, vpaddlq_s8(vreinterpretq_s8_s64(z3)));

            int32x4_t RowSums0H_ext = vextq_s32(RowSums0H_pada, RowSums0H_pada, 1);
            int32x4_t RowSums0H_add = vaddq_s32(RowSums0H_pada, RowSums0H_ext);
            int32x2_t RowSums0H = {vdups_laneq_s32(RowSums0H_add, 0),
                                   vdups_laneq_s32(RowSums0H_add, 2)};

            RowSums0 = vaddq_s32(RowSums0, vcombine_s32(RowSums0L, RowSums0H));

            int32x4_t RowSums1L_pada = vmovq_n_s32(0);
            RowSums1L_pada = vpadalq_s16(RowSums1L_pada, vpaddlq_s8(vreinterpretq_s8_s64(z4)));
            RowSums1L_pada = vpadalq_s16(RowSums1L_pada, vpaddlq_s8(vreinterpretq_s8_s64(z5)));

            int32x4_t RowSums1L_ext = vextq_s32(RowSums1L_pada, RowSums1L_pada, 1);
            int32x4_t RowSums1L_add = vaddq_s32(RowSums1L_pada, RowSums1L_ext);
            int32x2_t RowSums1L = {vdups_laneq_s32(RowSums1L_add, 0),
                                   vdups_laneq_s32(RowSums1L_add, 2)};

            int32x4_t RowSums1H_pada = vmovq_n_s32(0);
            RowSums1H_pada = vpadalq_s16(RowSums1H_pada, vpaddlq_s8(vreinterpretq_s8_s64(z6)));
            RowSums1H_pada = vpadalq_s16(RowSums1H_pada, vpaddlq_s8(vreinterpretq_s8_s64(z7)));

            int32x4_t RowSums1H_ext = vextq_s32(RowSums1H_pada, RowSums1H_pada, 1);
            int32x4_t RowSums1H_add = vaddq_s32(RowSums1H_pada, RowSums1H_ext);
            int32x2_t RowSums1H = {vdups_laneq_s32(RowSums1H_add, 0),
                                   vdups_laneq_s32(RowSums1H_add, 2)};

            RowSums1 = vaddq_s32(RowSums1, vcombine_s32(RowSums1L, RowSums1H));

            D += 128;
            k -= 16;
        }

        while (k >= 8) {
            int64x1_t v0 = *reinterpret_cast<const int64x1_t*>(a0);
            a0 += 8;
            int64x1_t v1 = *reinterpret_cast<const int64x1_t*>(a1);
            a1 += 8;
            int64x1_t v2 = *reinterpret_cast<const int64x1_t*>(a2);
            a2 += 8;
            int64x1_t v3 = *reinterpret_cast<const int64x1_t*>(a3);
            a3 += 8;
            int64x1_t v4 = *reinterpret_cast<const int64x1_t*>(a4);
            a4 += 8;
            int64x1_t v5 = *reinterpret_cast<const int64x1_t*>(a5);
            a5 += 8;
            int64x1_t v6 = *reinterpret_cast<const int64x1_t*>(a6);
            a6 += 8;
            int64x1_t v7 = *reinterpret_cast<const int64x1_t*>(a7);
            a7 += 8;

            *reinterpret_cast<int64x1_t*>(&D[0]) = v0;
            *reinterpret_cast<int64x1_t*>(&D[8]) = v1;
            *reinterpret_cast<int64x1_t*>(&D[16]) = v2;
            *reinterpret_cast<int64x1_t*>(&D[24]) = v3;
            *reinterpret_cast<int64x1_t*>(&D[32]) = v4;
            *reinterpret_cast<int64x1_t*>(&D[40]) = v5;
            *reinterpret_cast<int64x1_t*>(&D[48]) = v6;
            *reinterpret_cast<int64x1_t*>(&D[56]) = v7;

            int64x2_t z01 = vcombine_s64(v0, v1);
            int64x2_t z23 = vcombine_s64(v2, v3);
            int64x2_t z45 = vcombine_s64(v4, v5);
            int64x2_t z67 = vcombine_s64(v6, v7);

            int32x4_t RowSums0L_pada = vmovq_n_s32(0);
            RowSums0L_pada = vpadalq_s16(RowSums0L_pada, vpaddlq_s8(vreinterpretq_s8_s64(z01)));

            int32x4_t RowSums0L_ext = vextq_s32(RowSums0L_pada, RowSums0L_pada, 1);
            int32x4_t RowSums0L_add = vaddq_s32(RowSums0L_pada, RowSums0L_ext);
            int32x2_t RowSums0L = {vdups_laneq_s32(RowSums0L_add, 0),
                                   vdups_laneq_s32(RowSums0L_add, 2)};

            int32x4_t RowSums0H_pada = vmovq_n_s32(0);
            RowSums0H_pada = vpadalq_s16(RowSums0H_pada, vpaddlq_s8(vreinterpretq_s8_s64(z23)));

            int32x4_t RowSums0H_ext = vextq_s32(RowSums0H_pada, RowSums0H_pada, 1);
            int32x4_t RowSums0H_add = vaddq_s32(RowSums0H_pada, RowSums0H_ext);
            int32x2_t RowSums0H = {vdups_laneq_s32(RowSums0H_add, 0),
                                   vdups_laneq_s32(RowSums0H_add, 2)};

            RowSums0 = vaddq_s32(RowSums0, vcombine_s32(RowSums0L, RowSums0H));

            int32x4_t RowSums1L_pada = vmovq_n_s32(0);
            RowSums1L_pada = vpadalq_s16(RowSums1L_pada, vpaddlq_s8(vreinterpretq_s8_s64(z45)));

            int32x4_t RowSums1L_ext = vextq_s32(RowSums1L_pada, RowSums1L_pada, 1);
            int32x4_t RowSums1L_add = vaddq_s32(RowSums1L_pada, RowSums1L_ext);
            int32x2_t RowSums1L = {vdups_laneq_s32(RowSums1L_add, 0),
                                   vdups_laneq_s32(RowSums1L_add, 2)};

            int32x4_t RowSums1H_pada = vmovq_n_s32(0);
            RowSums1H_pada = vpadalq_s16(RowSums1H_pada, vpaddlq_s8(vreinterpretq_s8_s64(z67)));

            int32x4_t RowSums1H_ext = vextq_s32(RowSums1H_pada, RowSums1H_pada, 1);
            int32x4_t RowSums1H_add = vaddq_s32(RowSums1H_pada, RowSums1H_ext);
            int32x2_t RowSums1H = {vdups_laneq_s32(RowSums1H_add, 0),
                                   vdups_laneq_s32(RowSums1H_add, 2)};

            RowSums1 = vaddq_s32(RowSums1, vcombine_s32(RowSums1L, RowSums1H));

            D += 64;
            k -= 8;
        }

        if (k > 0) {
            //
            // zero pad the remaining columns to 8
            //
            int8_t* d = D;

            vst1q_s8(d, vmovq_n_s8(0));
            vst1q_s8(&d[16], vmovq_n_s8(0));
            vst1q_s8(&d[32], vmovq_n_s8(0));
            vst1q_s8(&d[48], vmovq_n_s8(0));

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
            int64x1_t v0 = *reinterpret_cast<const int64x1_t*>(d);
            d = d + 8;
            int64x1_t v1 = *reinterpret_cast<const int64x1_t*>(d);
            d = d + 8;
            int64x1_t v2 = *reinterpret_cast<const int64x1_t*>(d);
            d = d + 8;
            int64x1_t v3 = *reinterpret_cast<const int64x1_t*>(d);
            d = d + 8;
            int64x1_t v4 = *reinterpret_cast<const int64x1_t*>(d);
            d = d + 8;
            int64x1_t v5 = *reinterpret_cast<const int64x1_t*>(d);
            d = d + 8;
            int64x1_t v6 = *reinterpret_cast<const int64x1_t*>(d);
            d = d + 8;
            int64x1_t v7 = *reinterpret_cast<const int64x1_t*>(d);
            d = d + 8;

            int64x2_t z01 = vcombine_s64(v0, v1);
            int64x2_t z23 = vcombine_s64(v2, v3);
            int64x2_t z45 = vcombine_s64(v4, v5);
            int64x2_t z67 = vcombine_s64(v6, v7);

            int32x4_t RowSums0L_pada = vmovq_n_s32(0);
            RowSums0L_pada = vpadalq_s16(RowSums0L_pada, vpaddlq_s8(vreinterpretq_s8_s64(z01)));

            int32x4_t RowSums0L_ext = vextq_s32(RowSums0L_pada, RowSums0L_pada, 1);
            int32x4_t RowSums0L_add = vaddq_s32(RowSums0L_pada, RowSums0L_ext);
            int32x2_t RowSums0L = {vdups_laneq_s32(RowSums0L_add, 0),
                                   vdups_laneq_s32(RowSums0L_add, 2)};

            int32x4_t RowSums0H_pada = vmovq_n_s32(0);
            RowSums0H_pada = vpadalq_s16(RowSums0H_pada, vpaddlq_s8(vreinterpretq_s8_s64(z23)));

            int32x4_t RowSums0H_ext = vextq_s32(RowSums0H_pada, RowSums0H_pada, 1);
            int32x4_t RowSums0H_add = vaddq_s32(RowSums0H_pada, RowSums0H_ext);
            int32x2_t RowSums0H = {vdups_laneq_s32(RowSums0H_add, 0),
                                   vdups_laneq_s32(RowSums0H_add, 2)};

            RowSums0 = vaddq_s32(RowSums0, vcombine_s32(RowSums0L, RowSums0H));

            int32x4_t RowSums1L_pada = vmovq_n_s32(0);
            RowSums1L_pada = vpadalq_s16(RowSums1L_pada, vpaddlq_s8(vreinterpretq_s8_s64(z45)));

            int32x4_t RowSums1L_ext = vextq_s32(RowSums1L_pada, RowSums1L_pada, 1);
            int32x4_t RowSums1L_add = vaddq_s32(RowSums1L_pada, RowSums1L_ext);
            int32x2_t RowSums1L = {vdups_laneq_s32(RowSums1L_add, 0),
                                   vdups_laneq_s32(RowSums1L_add, 2)};

            int32x4_t RowSums1H_pada = vmovq_n_s32(0);
            RowSums1H_pada = vpadalq_s16(RowSums1H_pada, vpaddlq_s8(vreinterpretq_s8_s64(z67)));

            int32x4_t RowSums1H_ext = vextq_s32(RowSums1H_pada, RowSums1H_pada, 1);
            int32x4_t RowSums1H_add = vaddq_s32(RowSums1H_pada, RowSums1H_ext);
            int32x2_t RowSums1H = {vdups_laneq_s32(RowSums1H_add, 0),
                                   vdups_laneq_s32(RowSums1H_add, 2)};

            RowSums1 = vaddq_s32(RowSums1, vcombine_s32(RowSums1L, RowSums1H));

            D += 64;
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
        const int8_t* a0 = reinterpret_cast<const int8_t*>(A);
        const int8_t* a1 = a0 + lda;
        const int8_t* a2 = a1 + lda;
        const int8_t* a3 = a2 + lda;

        size_t k = CountK;
        int32x4_t RowSums = vmovq_n_s32(0);

        while (k >= 16) {
            int64x2_t v0 = vld1q_s64(reinterpret_cast<const int64_t*>(a0));
            a0 += 16;
            int64x2_t v1 = vld1q_s64(reinterpret_cast<const int64_t*>(a1));
            a1 += 16;
            int64x2_t v2 = vld1q_s64(reinterpret_cast<const int64_t*>(a2));
            a2 += 16;
            int64x2_t v3 = vld1q_s64(reinterpret_cast<const int64_t*>(a3));
            a3 += 16;

            int64x2_t z0 = vzip1q_s64(v0, v1);
            int64x2_t z1 = vzip2q_s64(v0, v1);
            int64x2_t z2 = vzip1q_s64(v2, v3);
            int64x2_t z3 = vzip2q_s64(v2, v3);

            vst1q_s8(&D[0], vreinterpretq_s8_s64(z0));
            vst1q_s8(&D[16], vreinterpretq_s8_s64(z2));
            vst1q_s8(&D[32], vreinterpretq_s8_s64(z1));
            vst1q_s8(&D[48], vreinterpretq_s8_s64(z3));

            int32x4_t RowSumsL_pada = vmovq_n_s32(0);
            RowSumsL_pada = vpadalq_s16(RowSumsL_pada, vpaddlq_s8(vreinterpretq_s8_s64(z0)));
            RowSumsL_pada = vpadalq_s16(RowSumsL_pada, vpaddlq_s8(vreinterpretq_s8_s64(z1)));

            int32x4_t RowSumsL_ext = vextq_s32(RowSumsL_pada, RowSumsL_pada, 1);
            int32x4_t RowSumsL_add = vaddq_s32(RowSumsL_pada, RowSumsL_ext);
            int32x2_t RowSumsL = {vdups_laneq_s32(RowSumsL_add, 0),
                                  vdups_laneq_s32(RowSumsL_add, 2)};

            int32x4_t RowSumsH_pada = vmovq_n_s32(0);
            RowSumsH_pada = vpadalq_s16(RowSumsH_pada, vpaddlq_s8(vreinterpretq_s8_s64(z2)));
            RowSumsH_pada = vpadalq_s16(RowSumsH_pada, vpaddlq_s8(vreinterpretq_s8_s64(z3)));

            int32x4_t RowSumsH_ext = vextq_s32(RowSumsH_pada, RowSumsH_pada, 1);
            int32x4_t RowSumsH_add = vaddq_s32(RowSumsH_pada, RowSumsH_ext);
            int32x2_t RowSumsH = {vdups_laneq_s32(RowSumsH_add, 0),
                                  vdups_laneq_s32(RowSumsH_add, 2)};

            RowSums = vaddq_s32(RowSums, vcombine_s32(RowSumsL, RowSumsH));

            D += 64;
            k -= 16;
        }

        while (k >= 8) {
            int64x1_t v0 = *reinterpret_cast<const int64x1_t*>(a0);
            a0 += 8;
            int64x1_t v1 = *reinterpret_cast<const int64x1_t*>(a1);
            a1 += 8;
            int64x1_t v2 = *reinterpret_cast<const int64x1_t*>(a2);
            a2 += 8;
            int64x1_t v3 = *reinterpret_cast<const int64x1_t*>(a3);
            a3 += 8;

            *reinterpret_cast<int64x1_t*>(&D[0]) = v0;
            *reinterpret_cast<int64x1_t*>(&D[8]) = v1;
            *reinterpret_cast<int64x1_t*>(&D[16]) = v2;
            *reinterpret_cast<int64x1_t*>(&D[24]) = v3;

            int64x2_t z01 = vcombine_s64(v0, v1);
            int64x2_t z23 = vcombine_s64(v2, v3);

            int32x4_t RowSumsL_pada = vmovq_n_s32(0);
            RowSumsL_pada = vpadalq_s16(RowSumsL_pada, vpaddlq_s8(vreinterpretq_s8_s64(z01)));

            int32x4_t RowSumsL_ext = vextq_s32(RowSumsL_pada, RowSumsL_pada, 1);
            int32x4_t RowSumsL_add = vaddq_s32(RowSumsL_pada, RowSumsL_ext);
            int32x2_t RowSumsL = {vdups_laneq_s32(RowSumsL_add, 0),
                                  vdups_laneq_s32(RowSumsL_add, 2)};

            int32x4_t RowSumsH_pada = vmovq_n_s32(0);
            RowSumsH_pada = vpadalq_s16(RowSumsH_pada, vpaddlq_s8(vreinterpretq_s8_s64(z23)));

            int32x4_t RowSumsH_ext = vextq_s32(RowSumsH_pada, RowSumsH_pada, 1);
            int32x4_t RowSumsH_add = vaddq_s32(RowSumsH_pada, RowSumsH_ext);
            int32x2_t RowSumsH = {vdups_laneq_s32(RowSumsH_add, 0),
                                  vdups_laneq_s32(RowSumsH_add, 2)};

            RowSums = vaddq_s32(RowSums, vcombine_s32(RowSumsL, RowSumsH));

            D += 32;
            k -= 8;
        }

        if (k > 0) {
            //
            // Copy the remaining bytes with zero padding.
            //
            int8_t* d = D;

            vst1q_s8(d, vmovq_n_s8(0));
            vst1q_s8(&d[16], vmovq_n_s8(0));

            while (k > 0) {
                d[0] = *a0++;
                d[8] = *a1++;
                d[16] = *a2++;
                d[24] = *a3++;
                d += 1;
                k -= 1;
            }

            d = D;
            int64x1_t v0 = *reinterpret_cast<const int64x1_t*>(d);
            d = d + 8;
            int64x1_t v1 = *reinterpret_cast<const int64x1_t*>(d);
            d = d + 8;
            int64x1_t v2 = *reinterpret_cast<const int64x1_t*>(d);
            d = d + 8;
            int64x1_t v3 = *reinterpret_cast<const int64x1_t*>(d);
            d = d + 8;

            int64x2_t z01 = vcombine_s64(v0, v1);
            int64x2_t z23 = vcombine_s64(v2, v3);

            int32x4_t RowSums0L_pada = vmovq_n_s32(0);
            RowSums0L_pada = vpadalq_s16(RowSums0L_pada, vpaddlq_s8(vreinterpretq_s8_s64(z01)));

            int32x4_t RowSums0L_ext = vextq_s32(RowSums0L_pada, RowSums0L_pada, 1);
            int32x4_t RowSums0L_add = vaddq_s32(RowSums0L_pada, RowSums0L_ext);
            int32x2_t RowSums0L = {vdups_laneq_s32(RowSums0L_add, 0),
                                   vdups_laneq_s32(RowSums0L_add, 2)};

            int32x4_t RowSums0H_pada = vmovq_n_s32(0);
            RowSums0H_pada = vpadalq_s16(RowSums0H_pada, vpaddlq_s8(vreinterpretq_s8_s64(z23)));

            int32x4_t RowSums0H_ext = vextq_s32(RowSums0H_pada, RowSums0H_pada, 1);
            int32x4_t RowSums0H_add = vaddq_s32(RowSums0H_pada, RowSums0H_ext);
            int32x2_t RowSums0H = {vdups_laneq_s32(RowSums0H_add, 0),
                                   vdups_laneq_s32(RowSums0H_add, 2)};

            RowSums = vaddq_s32(RowSums, vcombine_s32(RowSums0L, RowSums0H));

            D += 32;
        }

        vst1q_s32(RowSumBuffer, RowSums);
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
        const int8_t* a0 = reinterpret_cast<const int8_t*>(A);
        const int8_t* a1 = a0 + lda;

        size_t k = CountK;
        int32x2_t RowSums = vmov_n_s32(0);

        while (k >= 16) {
            int64x2_t v0 = vld1q_s64(reinterpret_cast<const int64_t*>(a0));
            a0 += 16;
            int64x2_t v1 = vld1q_s64(reinterpret_cast<const int64_t*>(a1));
            a1 += 16;

            int64x2_t z0 = vzip1q_s64(v0, v1);
            int64x2_t z1 = vzip2q_s64(v0, v1);

            vst1q_s8(&D[0], vreinterpretq_s8_s64(z0));
            vst1q_s8(&D[16], vreinterpretq_s8_s64(z1));

            int32x4_t RowSumsL_pada = vmovq_n_s32(0);
            RowSumsL_pada = vpadalq_s16(RowSumsL_pada, vpaddlq_s8(vreinterpretq_s8_s64(z0)));
            RowSumsL_pada = vpadalq_s16(RowSumsL_pada, vpaddlq_s8(vreinterpretq_s8_s64(z1)));

            int32x4_t RowSumsL_ext = vextq_s32(RowSumsL_pada, RowSumsL_pada, 1);
            int32x4_t RowSumsL_add = vaddq_s32(RowSumsL_pada, RowSumsL_ext);
            int32x2_t RowSumsL = {vdups_laneq_s32(RowSumsL_add, 0),
                                  vdups_laneq_s32(RowSumsL_add, 2)};

            RowSums = vadd_s32(RowSums, RowSumsL);

            D += 32;
            k -= 16;
        }

        while (k >= 8) {
            int64x1_t v0 = *reinterpret_cast<const int64x1_t*>(a0);
            a0 += 8;
            int64x1_t v1 = *reinterpret_cast<const int64x1_t*>(a1);
            a1 += 8;

            *reinterpret_cast<int64x1_t*>(&D[0]) = v0;
            *reinterpret_cast<int64x1_t*>(&D[8]) = v1;

            int64x2_t z01 = vcombine_s64(v0, v1);
            int32x4_t RowSumsL_pada = vmovq_n_s32(0);
            RowSumsL_pada = vpadalq_s16(RowSumsL_pada, vpaddlq_s8(vreinterpretq_s8_s64(z01)));

            int32x4_t RowSumsL_ext = vextq_s32(RowSumsL_pada, RowSumsL_pada, 1);
            int32x4_t RowSumsL_add = vaddq_s32(RowSumsL_pada, RowSumsL_ext);
            int32x2_t RowSumsL = {vdups_laneq_s32(RowSumsL_add, 0),
                                  vdups_laneq_s32(RowSumsL_add, 2)};

            RowSums = vadd_s32(RowSums, RowSumsL);

            D += 16;
            k -= 8;
        }

        if (k > 0) {
            //
            // Zero pad the remaining elements to make 8 columns.
            //

            int8_t* d = PaddedMatrixAData;
            vst1q_s8(PaddedMatrixAData, vmovq_n_s8(0));

            while (k > 0) {
                d[0] = *a0++;
                d[8] = *a1++;

                d += 1;
                k -= 1;
            }

            d = PaddedMatrixAData;
            int64x1_t v0 = *reinterpret_cast<const int64x1_t*>(d);
            d = d + 8;
            int64x1_t v1 = *reinterpret_cast<const int64x1_t*>(d);
            d = d + 8;

            int64x2_t z01 = vcombine_s64(v0, v1);
            int32x4_t RowSumsL_pada = vmovq_n_s32(0);
            RowSumsL_pada = vpadalq_s16(RowSumsL_pada, vpaddlq_s8(vreinterpretq_s8_s64(z01)));

            int32x4_t RowSumsL_ext = vextq_s32(RowSumsL_pada, RowSumsL_pada, 1);
            int32x4_t RowSumsL_add = vaddq_s32(RowSumsL_pada, RowSumsL_ext);
            int32x2_t RowSumsL = {vdups_laneq_s32(RowSumsL_add, 0),
                                  vdups_laneq_s32(RowSumsL_add, 2)};

            RowSums = vadd_s32(RowSums, RowSumsL);

            int8x16_t PackedVector = vld1q_s8(PaddedMatrixAData);
            vst1q_s8(D, PackedVector);

            D += 16;
        }

        vst1_s32(RowSumBuffer, RowSums);
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

        *RowSumBuffer = int32_t(vaddvq_s32(RowSums));
    }
}

MLAS_FORCEINLINE
void
MlasGemmS8S8CopyPackBProcessSmmla(int8_t* D, int8x8_t BytesRow[8], int32x4_t ColumnSums[2])
{
    int8x16_t v02 = vcombine_s8(BytesRow[0], BytesRow[2]);
    int8x16_t v13 = vcombine_s8(BytesRow[1], BytesRow[3]);

    int8x16_t v46 = vcombine_s8(BytesRow[4], BytesRow[6]);
    int8x16_t v57 = vcombine_s8(BytesRow[5], BytesRow[7]);

    int8x16x2_t zw1 = vzipq_s8(v02, v13);
    int16x8x2_t zd1 = vzipq_s16(vreinterpretq_s16_s8(zw1.val[0]), vreinterpretq_s16_s8(zw1.val[1]));

    int8x16x2_t zw2 = vzipq_s8(v46, v57);
    int16x8x2_t zd2 = vzipq_s16(vreinterpretq_s16_s8(zw2.val[0]), vreinterpretq_s16_s8(zw2.val[1]));

    int32x4x2_t zd3 =
        vzipq_s32(vreinterpretq_s32_s16(zd1.val[0]), vreinterpretq_s32_s16(zd2.val[0]));
    int32x4x2_t zd4 =
        vzipq_s32(vreinterpretq_s32_s16(zd1.val[1]), vreinterpretq_s32_s16(zd2.val[1]));

    vst1q_s8(&D[0], vreinterpretq_s8_s32(zd3.val[0]));
    vst1q_s8(&D[16], vreinterpretq_s8_s32(zd3.val[1]));
    vst1q_s8(&D[32], vreinterpretq_s8_s32(zd4.val[0]));
    vst1q_s8(&D[48], vreinterpretq_s8_s32(zd4.val[1]));

    int32x4_t ColSums0L_pada = vmovq_n_s32(0);
    ColSums0L_pada = vpadalq_s16(ColSums0L_pada, vpaddlq_s8(vreinterpretq_s8_s32(zd3.val[0])));
    int32x4_t ColSums0L_ext = vextq_s32(ColSums0L_pada, ColSums0L_pada, 1);
    int32x4_t ColSums0L_add = vaddq_s32(ColSums0L_pada, ColSums0L_ext);
    int32x2_t ColSums0L = {vdups_laneq_s32(ColSums0L_add, 0), vdups_laneq_s32(ColSums0L_add, 2)};

    int32x4_t ColSums0H_pada = vmovq_n_s32(0);
    ColSums0H_pada = vpadalq_s16(ColSums0H_pada, vpaddlq_s8(vreinterpretq_s8_s32(zd3.val[1])));
    int32x4_t ColSums0H_ext = vextq_s32(ColSums0H_pada, ColSums0H_pada, 1);
    int32x4_t ColSums0H_add = vaddq_s32(ColSums0H_pada, ColSums0H_ext);
    int32x2_t ColSums0H = {vdups_laneq_s32(ColSums0H_add, 0), vdups_laneq_s32(ColSums0H_add, 2)};

    ColumnSums[0] = vaddq_s32(ColumnSums[0], vcombine_s32(ColSums0L, ColSums0H));

    int32x4_t ColSums1L_pada = vmovq_n_s32(0);
    ColSums1L_pada = vpadalq_s16(ColSums1L_pada, vpaddlq_s8(vreinterpretq_s8_s32(zd4.val[0])));
    int32x4_t ColSums1L_ext = vextq_s32(ColSums1L_pada, ColSums1L_pada, 1);
    int32x4_t ColSums1L_add = vaddq_s32(ColSums1L_pada, ColSums1L_ext);
    int32x2_t ColSums1L = {vdups_laneq_s32(ColSums1L_add, 0), vdups_laneq_s32(ColSums1L_add, 2)};

    int32x4_t ColSums1H_pada = vmovq_n_s32(0);
    ColSums1H_pada = vpadalq_s16(ColSums1H_pada, vpaddlq_s8(vreinterpretq_s8_s32(zd4.val[1])));
    int32x4_t ColSums1H_ext = vextq_s32(ColSums1H_pada, ColSums1H_pada, 1);
    int32x4_t ColSums1H_add = vaddq_s32(ColSums1H_pada, ColSums1H_ext);
    int32x2_t ColSums1H = {vdups_laneq_s32(ColSums1H_add, 0), vdups_laneq_s32(ColSums1H_add, 2)};

    ColumnSums[1] = vaddq_s32(ColumnSums[1], vcombine_s32(ColSums1L, ColSums1H));
}

template <>
void
MlasGemmQuantCopyPackB<MLAS_GEMM_S8S8_KERNEL_SMMLA>(MLAS_GEMM_S8S8_KERNEL_SMMLA::PackedBType* Dst,
                                                    const uint8_t* B,
                                                    size_t ldb,
                                                    size_t CountN,
                                                    size_t CountK,
                                                    int32_t* ColumnSumBuffer,
                                                    bool BIsSigned)
{
    MLAS_UNREFERENCED_PARAMETER(BIsSigned);
    int8_t* D = reinterpret_cast<int8_t*>(Dst);
    const int8x16_t ZeroVector = vmovq_n_s8(0);
    int8x8_t BytesRow[8];

    //
    // Copy data from matrix B into the destination buffer 8x2 blocks at a
    // time.
    //
    //
    while (CountN >= 8) {
        const int8_t* b = reinterpret_cast<const int8_t*>(B);
        size_t k = CountK;
        int32x4_t ColumnSums[2];

        ColumnSums[0] = vmovq_n_s32(0);
        ColumnSums[1] = vmovq_n_s32(0);

        while (k >= 8) {
            BytesRow[0] = vld1_s8(&b[ldb * 0]);
            BytesRow[1] = vld1_s8(&b[ldb * 1]);
            BytesRow[2] = vld1_s8(&b[ldb * 2]);
            BytesRow[3] = vld1_s8(&b[ldb * 3]);
            BytesRow[4] = vld1_s8(&b[ldb * 4]);
            BytesRow[5] = vld1_s8(&b[ldb * 5]);
            BytesRow[6] = vld1_s8(&b[ldb * 6]);
            BytesRow[7] = vld1_s8(&b[ldb * 7]);

            MlasGemmS8S8CopyPackBProcessSmmla(D, BytesRow, ColumnSums);

            D += 64;
            b += ldb * 8;
            k -= 8;
        }

        if (k > 0) {
            // Pad k to 8

            BytesRow[0] = vld1_s8(&b[ldb * 0]);
            BytesRow[1] = (k >= 2) ? vld1_s8(&b[ldb * 1]) : vget_low_s8(ZeroVector);
            BytesRow[2] = (k >= 3) ? vld1_s8(&b[ldb * 2]) : vget_low_s8(ZeroVector);
            BytesRow[3] = (k >= 4) ? vld1_s8(&b[ldb * 3]) : vget_low_s8(ZeroVector);
            BytesRow[4] = (k >= 5) ? vld1_s8(&b[ldb * 4]) : vget_low_s8(ZeroVector);
            BytesRow[5] = (k >= 6) ? vld1_s8(&b[ldb * 5]) : vget_low_s8(ZeroVector);
            BytesRow[6] = (k >= 7) ? vld1_s8(&b[ldb * 6]) : vget_low_s8(ZeroVector);
            BytesRow[7] = vget_low_s8(ZeroVector);

            MlasGemmS8S8CopyPackBProcessSmmla(D, BytesRow, ColumnSums);

            D += 64;
        }

        // Zero pad the output buffer to a multiple of PackedK if the above
        // processed an odd number of four row bundles.
        //
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
        int8_t PaddedMatrixBData[64];
        int32x4_t ColumnSums[2];

        vst1q_s8(&PaddedMatrixBData[0], ZeroVector);
        vst1q_s8(&PaddedMatrixBData[16], ZeroVector);
        vst1q_s8(&PaddedMatrixBData[32], ZeroVector);
        vst1q_s8(&PaddedMatrixBData[48], ZeroVector);

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
            const int8_t* bcopy4 = &b[ldb * 4];
            const int8_t* bcopy5 = &b[ldb * 5];
            const int8_t* bcopy6 = &b[ldb * 6];
            const int8_t* bcopy7 = &b[ldb * 7];

            if (k >= 8) {
                b += ldb * 8;
                k -= 8;

            } else {
                vst1q_s8(&PaddedMatrixBData[0], ZeroVector);
                vst1q_s8(&PaddedMatrixBData[16], ZeroVector);
                vst1q_s8(&PaddedMatrixBData[32], ZeroVector);
                vst1q_s8(&PaddedMatrixBData[48], ZeroVector);

                bcopy1 = (k >= 2) ? bcopy1 : &PaddedMatrixBData[56];
                bcopy2 = (k >= 3) ? bcopy2 : &PaddedMatrixBData[56];
                bcopy3 = (k >= 4) ? bcopy3 : &PaddedMatrixBData[56];
                bcopy4 = (k >= 5) ? bcopy4 : &PaddedMatrixBData[56];
                bcopy5 = (k >= 6) ? bcopy5 : &PaddedMatrixBData[56];
                bcopy6 = (k >= 7) ? bcopy6 : &PaddedMatrixBData[56];
                bcopy7 = &PaddedMatrixBData[56];

                k = 0;
            }

            int8_t* padded = PaddedMatrixBData;
            int8_t* padded_end = padded + CountN;
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

            BytesRow[0] = vld1_s8(&PaddedMatrixBData[0]);
            BytesRow[1] = vld1_s8(&PaddedMatrixBData[8]);
            BytesRow[2] = vld1_s8(&PaddedMatrixBData[16]);
            BytesRow[3] = vld1_s8(&PaddedMatrixBData[24]);
            BytesRow[4] = vld1_s8(&PaddedMatrixBData[32]);
            BytesRow[5] = vld1_s8(&PaddedMatrixBData[40]);
            BytesRow[6] = vld1_s8(&PaddedMatrixBData[48]);
            BytesRow[7] = vld1_s8(&PaddedMatrixBData[56]);

            MlasGemmS8S8CopyPackBProcessSmmla(D, BytesRow, ColumnSums);

            D += 64;
        }

        vst1q_s32(&ColumnSumBuffer[0], ColumnSums[0]);
        vst1q_s32(&ColumnSumBuffer[4], ColumnSums[1]);
    }
}

template <>
MLAS_FORCEINLINE size_t
MlasGemmQuantKernel<MLAS_GEMM_S8S8_KERNEL_SMMLA>(const MLAS_GEMM_S8S8_KERNEL_SMMLA::PackedAType* A,
                                                 const MLAS_GEMM_S8S8_KERNEL_SMMLA::PackedBType* B,
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
        RowsHandled = MlasGemmS8S8KernelSmmlaZero(A, B, C, PackedCountK, CountM, CountN, ldc,
                                                  RowSumBuffer, ColumnSumBuffer, ZeroPointB);
    } else {
        RowsHandled = MlasGemmS8S8KernelSmmlaAdd(A, B, C, PackedCountK, CountM, CountN, ldc,
                                                 RowSumBuffer, ColumnSumBuffer, ZeroPointB);
    }

    return RowsHandled;
}

const MLAS_GEMM_QUANT_DISPATCH MlasGemmS8S8DispatchSmmla = {
    MlasGemmQuantOperation<MLAS_GEMM_S8S8_KERNEL_SMMLA>,
    MlasGemmQuantPackedOperation<MLAS_GEMM_S8S8_KERNEL_SMMLA>,
    MlasGemmQuantCopyPackB<MLAS_GEMM_S8S8_KERNEL_SMMLA>,
    MLAS_GEMM_S8S8_KERNEL_SMMLA::PackedK,
    MLAS_GEMM_S8S8_KERNEL_SMMLA::PackedStrides.K,
    8};
