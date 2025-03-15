/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    halfgemm_kernel_neon_fp16.cpp

Abstract:

    This module implements half precision GEMM kernel for neon.

--*/

#include <arm_neon.h>

#include "halfgemm.h"
#include "fp16_common.h"

namespace hgemm_neon {

void HPackB_TransposedB_Kernel(
    const MLAS_FP16* B,
    MLAS_FP16* PackedB,
    size_t CountN,
    size_t CountK,
    size_t ldb
) {
    const _mlas_fp16_* B_data = reinterpret_cast<const _mlas_fp16_*>(B);
    _mlas_fp16_* PackedB_data = reinterpret_cast<_mlas_fp16_*>(PackedB);
    const bool Kr0 = (CountK % 4) > 0;
    const bool Kr1 = (CountK % 4) > 1;
    const bool Kr2 = (CountK % 4) > 2;
    const bool Kr3 = CountK & 4;
    for (; CountN >= 32; CountN -= 32, B_data += 32 * ldb) {
        const _mlas_fp16_* b = B_data;
        size_t k = CountK;
        constexpr size_t step = 8 * 32; // pack 8 * 16
        for (; k >= 8; k -= 8, b += 8, PackedB_data += step) {
            size_t baseb = 0;
            size_t basep = 0;
            float16x8_t v0 = MlasLoadFloat16x8(b);
            float16x8_t v1 = MlasLoadFloat16x8(b + ldb);
            float16x8_t v2 = MlasLoadFloat16x8(b + 2 * ldb);
            float16x8_t v3 = MlasLoadFloat16x8(b + 3 * ldb);
            float16x8_t v4 = MlasLoadFloat16x8(b + 4 * ldb);
            float16x8_t v5 = MlasLoadFloat16x8(b + 5 * ldb);
            float16x8_t v6 = MlasLoadFloat16x8(b + 6 * ldb);
            float16x8_t v7 = MlasLoadFloat16x8(b + 7 * ldb);
            for (size_t i = 0; i < 3; ++i, baseb += 8 * ldb, basep += 8) {
                Transpose8x8(v0, v1, v2, v3, v4, v5, v6, v7);
                MlasStoreFloat16x8(PackedB_data + basep, v0);
                MlasStoreFloat16x8(PackedB_data + basep + 32, v1);
                MlasStoreFloat16x8(PackedB_data + basep + 64, v2);
                MlasStoreFloat16x8(PackedB_data + basep + 96, v3);
                MlasStoreFloat16x8(PackedB_data + basep + 128, v4);
                MlasStoreFloat16x8(PackedB_data + basep + 160, v5);
                MlasStoreFloat16x8(PackedB_data + basep + 192, v6);
                MlasStoreFloat16x8(PackedB_data + basep + 224, v7);
                v0 = MlasLoadFloat16x8(b + baseb + 8 * ldb);
                v1 = MlasLoadFloat16x8(b + baseb + 9 * ldb);
                v2 = MlasLoadFloat16x8(b + baseb + 10 * ldb);
                v3 = MlasLoadFloat16x8(b + baseb + 11 * ldb);
                v4 = MlasLoadFloat16x8(b + baseb + 12 * ldb);
                v5 = MlasLoadFloat16x8(b + baseb + 13 * ldb);
                v6 = MlasLoadFloat16x8(b + baseb + 14 * ldb);
                v7 = MlasLoadFloat16x8(b + baseb + 15 * ldb);
            }
            Transpose8x8(v0, v1, v2, v3, v4, v5, v6, v7);
            MlasStoreFloat16x8(PackedB_data + basep, v0);
            MlasStoreFloat16x8(PackedB_data + basep + 32, v1);
            MlasStoreFloat16x8(PackedB_data + basep + 64, v2);
            MlasStoreFloat16x8(PackedB_data + basep + 96, v3);
            MlasStoreFloat16x8(PackedB_data + basep + 128, v4);
            MlasStoreFloat16x8(PackedB_data + basep + 160, v5);
            MlasStoreFloat16x8(PackedB_data + basep + 192, v6);
            MlasStoreFloat16x8(PackedB_data + basep + 224, v7);
        }

        if (Kr3) {
            size_t baseb = 0;
            size_t basep = 0;
            float16x4_t v0 = MlasLoadFloat16x4(b);
            float16x4_t v1 = MlasLoadFloat16x4(b + ldb);
            float16x4_t v2 = MlasLoadFloat16x4(b + 2 * ldb);
            float16x4_t v3 = MlasLoadFloat16x4(b + 3 * ldb);
            float16x4_t v4 = MlasLoadFloat16x4(b + 4 * ldb);
            float16x4_t v5 = MlasLoadFloat16x4(b + 5 * ldb);
            float16x4_t v6 = MlasLoadFloat16x4(b + 6 * ldb);
            float16x4_t v7 = MlasLoadFloat16x4(b + 7 * ldb);
            for (size_t i = 0; i < 3; ++i, baseb += 8 * ldb, basep += 8) {
                Transpose4x4(v0, v1, v2, v3);
                Transpose4x4(v4, v5, v6, v7);
                MlasStoreFloat16x4(PackedB_data + basep, v0);
                MlasStoreFloat16x4(PackedB_data + basep + 4, v4);
                MlasStoreFloat16x4(PackedB_data + basep + 32, v1);
                MlasStoreFloat16x4(PackedB_data + basep + 36, v5);
                MlasStoreFloat16x4(PackedB_data + basep + 64, v2);
                MlasStoreFloat16x4(PackedB_data + basep + 68, v6);
                MlasStoreFloat16x4(PackedB_data + basep + 96, v3);
                MlasStoreFloat16x4(PackedB_data + basep + 100, v7);
                v0 = MlasLoadFloat16x4(b + baseb + 8 * ldb);
                v1 = MlasLoadFloat16x4(b + baseb + 9 * ldb);
                v2 = MlasLoadFloat16x4(b + baseb + 10 * ldb);
                v3 = MlasLoadFloat16x4(b + baseb + 11 * ldb);
                v4 = MlasLoadFloat16x4(b + baseb + 12 * ldb);
                v5 = MlasLoadFloat16x4(b + baseb + 13 * ldb);
                v6 = MlasLoadFloat16x4(b + baseb + 14 * ldb);
                v7 = MlasLoadFloat16x4(b + baseb + 15 * ldb);
            }
            Transpose4x4(v0, v1, v2, v3);
            Transpose4x4(v4, v5, v6, v7);
            MlasStoreFloat16x4(PackedB_data + basep, v0);
            MlasStoreFloat16x4(PackedB_data + basep + 4, v4);
            MlasStoreFloat16x4(PackedB_data + basep + 32, v1);
            MlasStoreFloat16x4(PackedB_data + basep + 36, v5);
            MlasStoreFloat16x4(PackedB_data + basep + 64, v2);
            MlasStoreFloat16x4(PackedB_data + basep + 68, v6);
            MlasStoreFloat16x4(PackedB_data + basep + 96, v3);
            MlasStoreFloat16x4(PackedB_data + basep + 100, v7);
            k -= 4, b += 4, PackedB_data += 4 * 32;
        }

        if (Kr0) {
            size_t baseb = 0;
            size_t basep = 0;
            float16x4_t v0 = MlasLoadPartialFloat16x4(b, k);
            float16x4_t v1 = MlasLoadPartialFloat16x4(b + ldb, k);
            float16x4_t v2 = MlasLoadPartialFloat16x4(b + 2 * ldb, k);
            float16x4_t v3 = MlasLoadPartialFloat16x4(b + 3 * ldb, k);
            float16x4_t v4 = MlasLoadPartialFloat16x4(b + 4 * ldb, k);
            float16x4_t v5 = MlasLoadPartialFloat16x4(b + 5 * ldb, k);
            float16x4_t v6 = MlasLoadPartialFloat16x4(b + 6 * ldb, k);
            float16x4_t v7 = MlasLoadPartialFloat16x4(b + 7 * ldb, k);
            for (size_t i = 0; i < 3; ++i, baseb += 8 * ldb, basep += 8) {
                Transpose4x4(v0, v1, v2, v3);
                Transpose4x4(v4, v5, v6, v7);
                MlasStoreFloat16x4(PackedB_data + basep, v0);
                MlasStoreFloat16x4(PackedB_data + basep + 4, v4);
                if (Kr1) {
                    MlasStoreFloat16x4(PackedB_data + basep + 32, v1);
                    MlasStoreFloat16x4(PackedB_data + basep + 36, v5);
                }
                if (Kr2) {
                    MlasStoreFloat16x4(PackedB_data + basep + 64, v2);
                    MlasStoreFloat16x4(PackedB_data + basep + 68, v6);
                }
                v0 = MlasLoadPartialFloat16x4(b + baseb + 8 * ldb, k);
                v1 = MlasLoadPartialFloat16x4(b + baseb + 9 * ldb, k);
                v2 = MlasLoadPartialFloat16x4(b + baseb + 10 * ldb, k);
                v3 = MlasLoadPartialFloat16x4(b + baseb + 11 * ldb, k);
                v4 = MlasLoadPartialFloat16x4(b + baseb + 12 * ldb, k);
                v5 = MlasLoadPartialFloat16x4(b + baseb + 13 * ldb, k);
                v6 = MlasLoadPartialFloat16x4(b + baseb + 14 * ldb, k);
                v7 = MlasLoadPartialFloat16x4(b + baseb + 15 * ldb, k);

            }
            Transpose4x4(v0, v1, v2, v3);
            Transpose4x4(v4, v5, v6, v7);
            MlasStoreFloat16x4(PackedB_data + basep, v0);
            MlasStoreFloat16x4(PackedB_data + basep + 4, v4);
            if (Kr1) {
                MlasStoreFloat16x4(PackedB_data + basep + 32, v1);
                MlasStoreFloat16x4(PackedB_data + basep + 36, v5);
            }
            if (Kr2) {
                MlasStoreFloat16x4(PackedB_data + basep + 64, v2);
                MlasStoreFloat16x4(PackedB_data + basep + 68, v6);
            }
            PackedB_data += k * 32;
        }
    }

    if (CountN & 16) {
        const _mlas_fp16_* b = B_data;
        size_t k = CountK;
        constexpr size_t step = 8 * 16; // pack 8 * 16
        for (; k >= 8; k -= 8, b += 8, PackedB_data += step) {
            float16x8_t v0 = MlasLoadFloat16x8(b);
            float16x8_t v1 = MlasLoadFloat16x8(b + ldb);
            float16x8_t v2 = MlasLoadFloat16x8(b + 2 * ldb);
            float16x8_t v3 = MlasLoadFloat16x8(b + 3 * ldb);
            float16x8_t v4 = MlasLoadFloat16x8(b + 4 * ldb);
            float16x8_t v5 = MlasLoadFloat16x8(b + 5 * ldb);
            float16x8_t v6 = MlasLoadFloat16x8(b + 6 * ldb);
            float16x8_t v7 = MlasLoadFloat16x8(b + 7 * ldb);
            float16x8_t v8 = MlasLoadFloat16x8(b + 8 * ldb);
            float16x8_t v9 = MlasLoadFloat16x8(b + 9 * ldb);
            float16x8_t vA = MlasLoadFloat16x8(b + 10 * ldb);
            float16x8_t vB = MlasLoadFloat16x8(b + 11 * ldb);
            float16x8_t vC = MlasLoadFloat16x8(b + 12 * ldb);
            float16x8_t vD = MlasLoadFloat16x8(b + 13 * ldb);
            float16x8_t vE = MlasLoadFloat16x8(b + 14 * ldb);
            float16x8_t vF = MlasLoadFloat16x8(b + 15 * ldb);
            Transpose8x8(v0, v1, v2, v3, v4, v5, v6, v7);
            Transpose8x8(v8, v9, vA, vB, vC, vD, vE, vF);

            MlasStoreFloat16x8(PackedB_data, v0);
            MlasStoreFloat16x8(PackedB_data + 8, v8);
            MlasStoreFloat16x8(PackedB_data + 16, v1);
            MlasStoreFloat16x8(PackedB_data + 24, v9);
            MlasStoreFloat16x8(PackedB_data + 32, v2);
            MlasStoreFloat16x8(PackedB_data + 40, vA);
            MlasStoreFloat16x8(PackedB_data + 48, v3);
            MlasStoreFloat16x8(PackedB_data + 56, vB);
            MlasStoreFloat16x8(PackedB_data + 64, v4);
            MlasStoreFloat16x8(PackedB_data + 72, vC);
            MlasStoreFloat16x8(PackedB_data + 80, v5);
            MlasStoreFloat16x8(PackedB_data + 88, vD);
            MlasStoreFloat16x8(PackedB_data + 96, v6);
            MlasStoreFloat16x8(PackedB_data + 104, vE);
            MlasStoreFloat16x8(PackedB_data + 112, v7);
            MlasStoreFloat16x8(PackedB_data + 120, vF);
        }

        if (Kr3) {
            float16x4_t v0 = MlasLoadFloat16x4(b);
            float16x4_t v1 = MlasLoadFloat16x4(b + ldb);
            float16x4_t v2 = MlasLoadFloat16x4(b + 2 * ldb);
            float16x4_t v3 = MlasLoadFloat16x4(b + 3 * ldb);
            float16x4_t v4 = MlasLoadFloat16x4(b + 4 * ldb);
            float16x4_t v5 = MlasLoadFloat16x4(b + 5 * ldb);
            float16x4_t v6 = MlasLoadFloat16x4(b + 6 * ldb);
            float16x4_t v7 = MlasLoadFloat16x4(b + 7 * ldb);
            float16x4_t v8 = MlasLoadFloat16x4(b + 8 * ldb);
            float16x4_t v9 = MlasLoadFloat16x4(b + 9 * ldb);
            float16x4_t vA = MlasLoadFloat16x4(b + 10 * ldb);
            float16x4_t vB = MlasLoadFloat16x4(b + 11 * ldb);
            float16x4_t vC = MlasLoadFloat16x4(b + 12 * ldb);
            float16x4_t vD = MlasLoadFloat16x4(b + 13 * ldb);
            float16x4_t vE = MlasLoadFloat16x4(b + 14 * ldb);
            float16x4_t vF = MlasLoadFloat16x4(b + 15 * ldb);
            Transpose4x4(v0, v1, v2, v3);
            Transpose4x4(v4, v5, v6, v7);
            Transpose4x4(v8, v9, vA, vB);
            Transpose4x4(vC, vD, vE, vF);
            MlasStoreFloat16x4(PackedB_data, v0);
            MlasStoreFloat16x4(PackedB_data + 4, v4);
            MlasStoreFloat16x4(PackedB_data + 8, v8);
            MlasStoreFloat16x4(PackedB_data + 12, vC);
            MlasStoreFloat16x4(PackedB_data + 16, v1);
            MlasStoreFloat16x4(PackedB_data + 20, v5);
            MlasStoreFloat16x4(PackedB_data + 24, v9);
            MlasStoreFloat16x4(PackedB_data + 28, vD);
            MlasStoreFloat16x4(PackedB_data + 32, v2);
            MlasStoreFloat16x4(PackedB_data + 36, v6);
            MlasStoreFloat16x4(PackedB_data + 40, vA);
            MlasStoreFloat16x4(PackedB_data + 44, vE);
            MlasStoreFloat16x4(PackedB_data + 48, v3);
            MlasStoreFloat16x4(PackedB_data + 52, v7);
            MlasStoreFloat16x4(PackedB_data + 56, vB);
            MlasStoreFloat16x4(PackedB_data + 60, vF);

            k -= 4, b += 4, PackedB_data += 4 * 16;
        }

        if (Kr0) {
            float16x4_t v0 = MlasLoadPartialFloat16x4(b, k);
            float16x4_t v1 = MlasLoadPartialFloat16x4(b + ldb, k);
            float16x4_t v2 = MlasLoadPartialFloat16x4(b + 2 * ldb, k);
            float16x4_t v3 = MlasLoadPartialFloat16x4(b + 3 * ldb, k);
            float16x4_t v4 = MlasLoadPartialFloat16x4(b + 4 * ldb, k);
            float16x4_t v5 = MlasLoadPartialFloat16x4(b + 5 * ldb, k);
            float16x4_t v6 = MlasLoadPartialFloat16x4(b + 6 * ldb, k);
            float16x4_t v7 = MlasLoadPartialFloat16x4(b + 7 * ldb, k);
            float16x4_t v8 = MlasLoadPartialFloat16x4(b + 8 * ldb, k);
            float16x4_t v9 = MlasLoadPartialFloat16x4(b + 9 * ldb, k);
            float16x4_t vA = MlasLoadPartialFloat16x4(b + 10 * ldb, k);
            float16x4_t vB = MlasLoadPartialFloat16x4(b + 11 * ldb, k);
            float16x4_t vC = MlasLoadPartialFloat16x4(b + 12 * ldb, k);
            float16x4_t vD = MlasLoadPartialFloat16x4(b + 13 * ldb, k);
            float16x4_t vE = MlasLoadPartialFloat16x4(b + 14 * ldb, k);
            float16x4_t vF = MlasLoadPartialFloat16x4(b + 15 * ldb, k);
            Transpose4x4(v0, v1, v2, v3);
            Transpose4x4(v4, v5, v6, v7);
            Transpose4x4(v8, v9, vA, vB);
            Transpose4x4(vC, vD, vE, vF);
            MlasStoreFloat16x4(PackedB_data, v0);
            MlasStoreFloat16x4(PackedB_data + 4, v4);
            MlasStoreFloat16x4(PackedB_data + 8, v8);
            MlasStoreFloat16x4(PackedB_data + 12, vC);
            if (Kr1) {
                MlasStoreFloat16x4(PackedB_data + 16, v1);
                MlasStoreFloat16x4(PackedB_data + 20, v5);
                MlasStoreFloat16x4(PackedB_data + 24, v9);
                MlasStoreFloat16x4(PackedB_data + 28, vD);
            }
            if (Kr2) {
                MlasStoreFloat16x4(PackedB_data + 32, v2);
                MlasStoreFloat16x4(PackedB_data + 36, v6);
                MlasStoreFloat16x4(PackedB_data + 40, vA);
                MlasStoreFloat16x4(PackedB_data + 44, vE);
            }

            PackedB_data += k * 16;
        }

        CountN -= 16, B_data += 16 * ldb;
    }

    if (CountN & 8) {
        const _mlas_fp16_* b = B_data;
        size_t k = CountK;
        constexpr size_t step = 8 * 8; // pack 8 * 8
        for (; k >= 8; k -= 8, b += 8, PackedB_data += step) {
            float16x8_t v0 = MlasLoadFloat16x8(b);
            float16x8_t v1 = MlasLoadFloat16x8(b + ldb);
            float16x8_t v2 = MlasLoadFloat16x8(b + 2 * ldb);
            float16x8_t v3 = MlasLoadFloat16x8(b + 3 * ldb);
            float16x8_t v4 = MlasLoadFloat16x8(b + 4 * ldb);
            float16x8_t v5 = MlasLoadFloat16x8(b + 5 * ldb);
            float16x8_t v6 = MlasLoadFloat16x8(b + 6 * ldb);
            float16x8_t v7 = MlasLoadFloat16x8(b + 7 * ldb);
            Transpose8x8(v0, v1, v2, v3, v4, v5, v6, v7);

            MlasStoreFloat16x8(PackedB_data, v0);
            MlasStoreFloat16x8(PackedB_data + 8, v1);
            MlasStoreFloat16x8(PackedB_data + 16, v2);
            MlasStoreFloat16x8(PackedB_data + 24, v3);
            MlasStoreFloat16x8(PackedB_data + 32, v4);
            MlasStoreFloat16x8(PackedB_data + 40, v5);
            MlasStoreFloat16x8(PackedB_data + 48, v6);
            MlasStoreFloat16x8(PackedB_data + 56, v7);
        }

        if (Kr3) {
            float16x4_t v0 = MlasLoadFloat16x4(b);
            float16x4_t v1 = MlasLoadFloat16x4(b + ldb);
            float16x4_t v2 = MlasLoadFloat16x4(b + 2 * ldb);
            float16x4_t v3 = MlasLoadFloat16x4(b + 3 * ldb);
            float16x4_t v4 = MlasLoadFloat16x4(b + 4 * ldb);
            float16x4_t v5 = MlasLoadFloat16x4(b + 5 * ldb);
            float16x4_t v6 = MlasLoadFloat16x4(b + 6 * ldb);
            float16x4_t v7 = MlasLoadFloat16x4(b + 7 * ldb);
            Transpose4x4(v0, v1, v2, v3);
            Transpose4x4(v4, v5, v6, v7);
            MlasStoreFloat16x4(PackedB_data, v0);
            MlasStoreFloat16x4(PackedB_data + 4, v4);
            MlasStoreFloat16x4(PackedB_data + 8, v1);
            MlasStoreFloat16x4(PackedB_data + 12, v5);
            MlasStoreFloat16x4(PackedB_data + 16, v2);
            MlasStoreFloat16x4(PackedB_data + 20, v6);
            MlasStoreFloat16x4(PackedB_data + 24, v3);
            MlasStoreFloat16x4(PackedB_data + 28, v7);
            k -= 4, b += 4, PackedB_data += 4 * 8;
        }

        if (Kr0) {
            float16x4_t v0 = MlasLoadPartialFloat16x4(b, k);
            float16x4_t v1 = MlasLoadPartialFloat16x4(b + ldb, k);
            float16x4_t v2 = MlasLoadPartialFloat16x4(b + 2 * ldb, k);
            float16x4_t v3 = MlasLoadPartialFloat16x4(b + 3 * ldb, k);
            float16x4_t v4 = MlasLoadPartialFloat16x4(b + 4 * ldb, k);
            float16x4_t v5 = MlasLoadPartialFloat16x4(b + 5 * ldb, k);
            float16x4_t v6 = MlasLoadPartialFloat16x4(b + 6 * ldb, k);
            float16x4_t v7 = MlasLoadPartialFloat16x4(b + 7 * ldb, k);
            Transpose4x4(v0, v1, v2, v3);
            Transpose4x4(v4, v5, v6, v7);
            MlasStoreFloat16x4(PackedB_data, v0);
            MlasStoreFloat16x4(PackedB_data + 4, v4);
            if (Kr1) {
                MlasStoreFloat16x4(PackedB_data + 8, v1);
                MlasStoreFloat16x4(PackedB_data + 12, v5);
            }
            if (Kr2) {
                MlasStoreFloat16x4(PackedB_data + 16, v2);
                MlasStoreFloat16x4(PackedB_data + 20, v6);
            }

            PackedB_data += k * 8;
        }

        B_data += 8 * ldb;
        CountN -= 8;
    }

    if (CountN > 0) {
        const _mlas_fp16_* b = B_data;
        size_t k = CountK;
        constexpr size_t step = 8 * 8; // pack extended 8 * 8
        for (; k >= 8; k -= 8, b += 8, PackedB_data += step) {
            float16x8_t v[8];
            size_t i = 0;
            for (; i < CountN; ++i) {
                v[i] = MlasLoadFloat16x8(b + i * ldb);
            }
            for (; i < 8; ++i) {
                v[i] = MlasZeroFloat16x8();
            }
            Transpose8x8(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
            MlasStoreFloat16x8(PackedB_data, v[0]);
            MlasStoreFloat16x8(PackedB_data + 8, v[1]);
            MlasStoreFloat16x8(PackedB_data + 16, v[2]);
            MlasStoreFloat16x8(PackedB_data + 24, v[3]);
            MlasStoreFloat16x8(PackedB_data + 32, v[4]);
            MlasStoreFloat16x8(PackedB_data + 40, v[5]);
            MlasStoreFloat16x8(PackedB_data + 48, v[6]);
            MlasStoreFloat16x8(PackedB_data + 56, v[7]);
        }

        if (Kr3) {
            float16x4_t v[8];
            size_t i = 0;
            for (; i < CountN; ++i) {
                v[i] = MlasLoadFloat16x4(b + i * ldb);
            }
            for (; i < 8; ++i) {
                v[i] = MlasZeroFloat16x4();
            }
            Transpose4x4(v[0], v[1], v[2], v[3]);
            Transpose4x4(v[4], v[5], v[6], v[7]);
            MlasStoreFloat16x4(PackedB_data, v[0]);
            MlasStoreFloat16x4(PackedB_data + 4, v[4]);
            MlasStoreFloat16x4(PackedB_data + 8, v[1]);
            MlasStoreFloat16x4(PackedB_data + 12, v[5]);
            MlasStoreFloat16x4(PackedB_data + 16, v[2]);
            MlasStoreFloat16x4(PackedB_data + 20, v[6]);
            MlasStoreFloat16x4(PackedB_data + 24, v[3]);
            MlasStoreFloat16x4(PackedB_data + 28, v[7]);
            k -= 4, b += 4, PackedB_data += 4 * 8;
        }

        if (Kr0) {
            float16x4_t v[8];
            size_t i = 0;
            for (; i < CountN; ++i) {
                v[i] = MlasLoadPartialFloat16x4(b + i * ldb, k);
            }
            for (; i < 8; ++i) {
                v[i] = MlasZeroFloat16x4();
            }
            Transpose4x4(v[0], v[1], v[2], v[3]);
            Transpose4x4(v[4], v[5], v[6], v[7]);
            MlasStoreFloat16x4(PackedB_data, v[0]);
            MlasStoreFloat16x4(PackedB_data + 4, v[4]);
            if (Kr1) {
                MlasStoreFloat16x4(PackedB_data + 8, v[1]);
                MlasStoreFloat16x4(PackedB_data + 12, v[5]);
            }
            if (Kr2) {
                MlasStoreFloat16x4(PackedB_data + 16, v[2]);
                MlasStoreFloat16x4(PackedB_data + 20, v[6]);
            }
        }
    }
}

void HPackB_B_Kernel(
    const MLAS_FP16* B,
    MLAS_FP16* PackedB,
    size_t CountN,
    size_t CountK,
    size_t ldb
) {
    const _mlas_fp16_* B_data = reinterpret_cast<const _mlas_fp16_*>(B);
    _mlas_fp16_* PackedB_data = reinterpret_cast<_mlas_fp16_*>(PackedB);

    for (; CountN >= 32; CountN -= 32, B_data += 32) {
        const _mlas_fp16_* b = B_data;
        size_t k = CountK;
        uint16x8x4_t v0 = vld4q_u16(b);
        for (; k >= 2; --k, b += ldb, PackedB_data += 32) {
            vst4q_u16(PackedB_data, v0);
            v0 = vld4q_u16(b + ldb);
        }
        vst4q_u16(PackedB_data, v0);
        PackedB_data += 32;
    }

    if (CountN & 16) {
        const _mlas_fp16_* b = B_data;
        size_t k = CountK;
        uint16x8x2_t v0 = vld2q_u16(b);
        for (; k >= 2; --k, b += ldb, PackedB_data += 16) {
            vst2q_u16(PackedB_data, v0);
            v0 = vld2q_u16(b + ldb);
        }
        vst2q_u16(PackedB_data, v0);
        PackedB_data += 16;
        CountN -= 16, B_data += 16;
    }

    if (CountN & 8) {
        const _mlas_fp16_* b = B_data;
        size_t k = CountK;
        uint16x8_t v0 = vld1q_u16(b);
        for (; k >= 2; --k, b += ldb, PackedB_data += 8) {
            vst1q_u16(PackedB_data, v0);
            v0 = vld1q_u16(b + ldb);
        }
        vst1q_u16(PackedB_data, v0);
        PackedB_data += 8;

        B_data += 8;
        CountN -= 8;
    }

    if (CountN > 4) {
        float16x4_t v0 = MlasLoadFloat16x4(B_data);
        float16x4_t v1 = MlasLoadPartialFloat16x4(B_data + 4, CountN - 4);
        for (; CountK >= 2; B_data += ldb, PackedB_data += 8, --CountK) {
            MlasStoreFloat16x4(PackedB_data, v0);
            MlasStoreFloat16x4(PackedB_data + 4, v1);
            v0 = MlasLoadFloat16x4(B_data + ldb);
            v1 = MlasLoadPartialFloat16x4(B_data + ldb + 4, CountN - 4);
        }
        MlasStoreFloat16x4(PackedB_data, v0);
        MlasStoreFloat16x4(PackedB_data + 4, v1);
    } else if (CountN > 0) {
        float16x4_t v0 = MlasLoadPartialFloat16x4(B_data, CountN);
        for (; CountK >= 2; B_data += ldb, PackedB_data += 8, --CountK) {
            MlasStoreFloat16x4(PackedB_data, v0);
            v0 = MlasLoadPartialFloat16x4(B_data + ldb, CountN);
        }
        MlasStoreFloat16x4(PackedB_data, v0);
    }
}

MLAS_FORCEINLINE
float16x8_t addq_f16x4(float16x8_t v0, float16x8_t v1, float16x8_t v2, float16x8_t v3) {
    v0 = vaddq_f16(v0, v1);
    v2 = vaddq_f16(v2, v3);
    v0 = vaddq_f16(v0, v2);
    return v0;
}

MLAS_FORCEINLINE
float16x8_t addq_f16x8(float16x8_t v0, float16x8_t v1, float16x8_t v2, float16x8_t v3,
                       float16x8_t v4, float16x8_t v5, float16x8_t v6, float16x8_t v7) {
    return vaddq_f16(addq_f16x4(v0, v1, v2, v3), addq_f16x4(v4, v5, v6, v7));
}

MLAS_FORCEINLINE
float16x8_t maq_lane_f16_accu(float16x8_t accu0, float16x8_t v0, float16x8_t v1, float16x8_t v2, float16x8_t v3,
                              float16x4_t a0) {
    accu0 = vfmaq_lane_f16(accu0, v0, a0, 0);
    accu0 = vfmaq_lane_f16(accu0, v1, a0, 1);
    accu0 = vfmaq_lane_f16(accu0, v2, a0, 2);
    accu0 = vfmaq_lane_f16(accu0, v3, a0, 3);
    return accu0;
}

MLAS_FORCEINLINE
float16x8_t maq_laneq_f16_accu(float16x8_t accu0, float16x8_t v0, float16x8_t v1, float16x8_t v2, float16x8_t v3,
                               float16x8_t v4, float16x8_t v5, float16x8_t v6, float16x8_t v7, float16x8_t a0) {
    accu0 = vfmaq_laneq_f16(accu0, v0, a0, 0);
    accu0 = vfmaq_laneq_f16(accu0, v1, a0, 1);
    accu0 = vfmaq_laneq_f16(accu0, v2, a0, 2);
    accu0 = vfmaq_laneq_f16(accu0, v3, a0, 3);
    accu0 = vfmaq_laneq_f16(accu0, v4, a0, 4);
    accu0 = vfmaq_laneq_f16(accu0, v5, a0, 5);
    accu0 = vfmaq_laneq_f16(accu0, v6, a0, 6);
    accu0 = vfmaq_laneq_f16(accu0, v7, a0, 7);
    return accu0;
}

MLAS_FORCEINLINE
float16x4_t ma_laneq_f16_accu(float16x4_t accu0, float16x4_t v0, float16x4_t v1, float16x4_t v2, float16x4_t v3,
                              float16x4_t v4, float16x4_t v5, float16x4_t v6, float16x4_t v7, float16x8_t a0) {
    accu0 = vfma_laneq_f16(accu0, v0, a0, 0);
    accu0 = vfma_laneq_f16(accu0, v1, a0, 1);
    accu0 = vfma_laneq_f16(accu0, v2, a0, 2);
    accu0 = vfma_laneq_f16(accu0, v3, a0, 3);
    accu0 = vfma_laneq_f16(accu0, v4, a0, 4);
    accu0 = vfma_laneq_f16(accu0, v5, a0, 5);
    accu0 = vfma_laneq_f16(accu0, v6, a0, 6);
    accu0 = vfma_laneq_f16(accu0, v7, a0, 7);
    return accu0;
}

MLAS_FORCEINLINE
float16x4_t ma_lane_f16_accu(float16x4_t accu, float16x4_t v0, float16x4_t v1, float16x4_t v2, float16x4_t v3,
                             float16x4_t a0) {
    accu = vfma_lane_f16(accu, v0, a0, 0);
    accu = vfma_lane_f16(accu, v1, a0, 1);
    accu = vfma_lane_f16(accu, v2, a0, 2);
    accu = vfma_lane_f16(accu, v3, a0, 3);
    return accu;
}

// beta_behavior: beta == 0.0f16 -> 0, beta == 1.0f16 -> 1, otherwise -> 2
template <int beta_behavior, int CountM>
void HGemm_TransposedB_Kernel_Impl(
    const _mlas_fp16_* A_data,
    const _mlas_fp16_* B_data,
    _mlas_fp16_* C_data,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc,
    _mlas_fp16_ alpha,
    _mlas_fp16_ beta
) {
    const bool largeK = CountK >= 8;
    const bool Kr0 = (CountK & 3);
    const bool Kr1 = (CountK & 3) > 1;
    const bool Kr2 = (CountK & 3) > 2;
    const bool Kr3 = (CountK & 4);
    for (; CountN >= 8; CountN -= 8, B_data += 8 * ldb, C_data += 8) {
        const auto* a = A_data;
        const auto* b = B_data;
        size_t k = CountK;
        float16x8_t accu00 = MlasZeroFloat16x8();
        float16x8_t accu01 = MlasZeroFloat16x8();
        float16x8_t accu02 = MlasZeroFloat16x8();
        float16x8_t accu03 = MlasZeroFloat16x8();
        float16x8_t accu04 = MlasZeroFloat16x8();
        float16x8_t accu05 = MlasZeroFloat16x8();
        float16x8_t accu06 = MlasZeroFloat16x8();
        float16x8_t accu07 = MlasZeroFloat16x8();
        float16x8_t accu10, accu11, accu12, accu13, accu14, accu15, accu16, accu17;
        if constexpr (CountM == 2) {
            accu10 = MlasZeroFloat16x8();
            accu11 = MlasZeroFloat16x8();
            accu12 = MlasZeroFloat16x8();
            accu13 = MlasZeroFloat16x8();
            accu14 = MlasZeroFloat16x8();
            accu15 = MlasZeroFloat16x8();
            accu16 = MlasZeroFloat16x8();
            accu17 = MlasZeroFloat16x8();
        }
        if (largeK) {
            float16x8_t b0 = MlasLoadFloat16x8(b);
            float16x8_t b1 = MlasLoadFloat16x8(b + ldb);
            float16x8_t b2 = MlasLoadFloat16x8(b + 2 * ldb);
            float16x8_t b3 = MlasLoadFloat16x8(b + 3 * ldb);
            float16x8_t b4 = MlasLoadFloat16x8(b + 4 * ldb);
            float16x8_t b5 = MlasLoadFloat16x8(b + 5 * ldb);
            float16x8_t b6 = MlasLoadFloat16x8(b + 6 * ldb);
            float16x8_t b7 = MlasLoadFloat16x8(b + 7 * ldb);
            float16x8_t a0 = MlasLoadFloat16x8(a);
            for (; k >= 16; k -= 8, a += 8, b += 8) {
                accu00 = vfmaq_f16(accu00, b0, a0);
                accu01 = vfmaq_f16(accu01, b1, a0);
                accu02 = vfmaq_f16(accu02, b2, a0);
                accu03 = vfmaq_f16(accu03, b3, a0);
                accu04 = vfmaq_f16(accu04, b4, a0);
                accu05 = vfmaq_f16(accu05, b5, a0);
                accu06 = vfmaq_f16(accu06, b6, a0);
                accu07 = vfmaq_f16(accu07, b7, a0);
                if constexpr (CountM == 2) {
                    float16x8_t a1 = MlasLoadFloat16x8(a + lda);
                    accu10 = vfmaq_f16(accu10, b0, a1);
                    accu11 = vfmaq_f16(accu11, b1, a1);
                    accu12 = vfmaq_f16(accu12, b2, a1);
                    accu13 = vfmaq_f16(accu13, b3, a1);
                    accu14 = vfmaq_f16(accu14, b4, a1);
                    accu15 = vfmaq_f16(accu15, b5, a1);
                    accu16 = vfmaq_f16(accu16, b6, a1);
                    accu17 = vfmaq_f16(accu17, b7, a1);
                }
                b0 = MlasLoadFloat16x8(b + 8);
                b1 = MlasLoadFloat16x8(b + 8 + ldb);
                b2 = MlasLoadFloat16x8(b + 8 + 2 * ldb);
                b3 = MlasLoadFloat16x8(b + 8 + 3 * ldb);
                b4 = MlasLoadFloat16x8(b + 8 + 4 * ldb);
                b5 = MlasLoadFloat16x8(b + 8 + 5 * ldb);
                b6 = MlasLoadFloat16x8(b + 8 + 6 * ldb);
                b7 = MlasLoadFloat16x8(b + 8 + 7 * ldb);
                a0 = MlasLoadFloat16x8(a + 8);
            }
            accu00 = vfmaq_f16(accu00, b0, a0);
            accu01 = vfmaq_f16(accu01, b1, a0);
            accu02 = vfmaq_f16(accu02, b2, a0);
            accu03 = vfmaq_f16(accu03, b3, a0);
            accu04 = vfmaq_f16(accu04, b4, a0);
            accu05 = vfmaq_f16(accu05, b5, a0);
            accu06 = vfmaq_f16(accu06, b6, a0);
            accu07 = vfmaq_f16(accu07, b7, a0);
            if constexpr (CountM == 2) {
                float16x8_t a1 = MlasLoadFloat16x8(a + lda);
                accu10 = vfmaq_f16(accu10, b0, a1);
                accu11 = vfmaq_f16(accu11, b1, a1);
                accu12 = vfmaq_f16(accu12, b2, a1);
                accu13 = vfmaq_f16(accu13, b3, a1);
                accu14 = vfmaq_f16(accu14, b4, a1);
                accu15 = vfmaq_f16(accu15, b5, a1);
                accu16 = vfmaq_f16(accu16, b6, a1);
                accu17 = vfmaq_f16(accu17, b7, a1);
            }
            k -= 8, a += 8, b += 8;
        }
        Transpose8x8(accu00, accu01, accu02, accu03, accu04, accu05, accu06, accu07);
        accu00 = addq_f16x8(accu00, accu01, accu02, accu03, accu04, accu05, accu06, accu07);
        if constexpr (CountM == 2) {
            Transpose8x8(accu10, accu11, accu12, accu13, accu14, accu15, accu16, accu17);
            accu10 = addq_f16x8(accu10, accu11, accu12, accu13, accu14, accu15, accu16, accu17);
        }

        if (Kr3) {
            float16x4_t b0 = MlasLoadFloat16x4(b);
            float16x4_t b1 = MlasLoadFloat16x4(b + ldb);
            float16x4_t b2 = MlasLoadFloat16x4(b + 2 * ldb);
            float16x4_t b3 = MlasLoadFloat16x4(b + 3 * ldb);
            float16x4_t b4 = MlasLoadFloat16x4(b + 4 * ldb);
            float16x4_t b5 = MlasLoadFloat16x4(b + 5 * ldb);
            float16x4_t b6 = MlasLoadFloat16x4(b + 6 * ldb);
            float16x4_t b7 = MlasLoadFloat16x4(b + 7 * ldb);
            Transpose4x4(b0, b1, b2, b3);
            Transpose4x4(b4, b5, b6, b7);
            float16x8_t v0 = vcombine_f16(b0, b4);
            float16x8_t v1 = vcombine_f16(b1, b5);
            float16x8_t v2 = vcombine_f16(b2, b6);
            float16x8_t v3 = vcombine_f16(b3, b7);
            float16x4_t a0 = MlasLoadFloat16x4(a);
            accu00 = maq_lane_f16_accu(accu00, v0, v1, v2, v3, a0);
            if constexpr (CountM == 2) {
                float16x4_t a1 = MlasLoadFloat16x4(a + lda);
                accu10 = maq_lane_f16_accu(accu10, v0, v1, v2, v3, a1);
            }
            k -= 4, a += 4, b += 4;
        }

        if (Kr0) {
            float16x4_t b0 = MlasLoadPartialFloat16x4(b, k);
            float16x4_t b1 = MlasLoadPartialFloat16x4(b + ldb, k);
            float16x4_t b2 = MlasLoadPartialFloat16x4(b + 2 * ldb, k);
            float16x4_t b3 = MlasLoadPartialFloat16x4(b + 3 * ldb, k);
            float16x4_t b4 = MlasLoadPartialFloat16x4(b + 4 * ldb, k);
            float16x4_t b5 = MlasLoadPartialFloat16x4(b + 5 * ldb, k);
            float16x4_t b6 = MlasLoadPartialFloat16x4(b + 6 * ldb, k);
            float16x4_t b7 = MlasLoadPartialFloat16x4(b + 7 * ldb, k);
            Transpose4x4(b0, b1, b2, b3);
            Transpose4x4(b4, b5, b6, b7);
            float16x8_t v0 = vcombine_f16(b0, b4);
            float16x4_t a0 = MlasLoadPartialFloat16x4(a, k), a1;
            accu00 = vfmaq_lane_f16(accu00, v0, a0, 0);
            if constexpr (CountM == 2) {
                a1 = MlasLoadPartialFloat16x4(a + lda, k);
                accu10 = vfmaq_lane_f16(accu10, v0, a1, 0);
            }
            if (Kr1) {
                float16x8_t v1 = vcombine_f16(b1, b5);
                accu00 = vfmaq_lane_f16(accu00, v1, a0, 1);
                if constexpr (CountM == 2) {
                    accu10 = vfmaq_lane_f16(accu10, v1, a1, 1);
                }
            }
            if (Kr2) {
                float16x8_t v2 = vcombine_f16(b2, b6);
                accu00 = vfmaq_lane_f16(accu00, v2, a0, 2);
                if constexpr (CountM == 2) {
                    accu10 = vfmaq_lane_f16(accu10, v2, a1, 2);
                }
            }
        }

        if constexpr (beta_behavior == 1) {
            float16x8_t alpha_v = MlasBroadcastFloat16x8(alpha);
            float16x8_t c0 = MlasLoadFloat16x8(C_data);
            accu00 = vfmaq_f16(c0, accu00, alpha_v);
            MlasStoreFloat16x8(C_data, accu00);
            if constexpr (CountM == 2) {
                float16x8_t c1 = MlasLoadFloat16x8(C_data + ldc);
                accu10 = vfmaq_f16(c1, accu10, alpha_v);
                MlasStoreFloat16x8(C_data + ldc, accu10);
            }
        } else if constexpr (beta_behavior == 2) {
            float16x8_t alpha_v = MlasBroadcastFloat16x8(alpha);
            float16x8_t beta_v = MlasBroadcastFloat16x8(beta);
            float16x8_t c0 = MlasLoadFloat16x8(C_data);
            accu00 = vfmaq_f16(vmulq_f16(c0, beta_v), accu00, alpha_v);
            MlasStoreFloat16x8(C_data, accu00);
            if constexpr (CountM == 2) {
                float16x8_t c1 = MlasLoadFloat16x8(C_data + ldc);
                accu10 = vfmaq_f16(vmulq_f16(c1, beta_v), accu10, alpha_v);
                MlasStoreFloat16x8(C_data + ldc, accu10);
            }
        } else {
            float16x8_t alpha_v = MlasBroadcastFloat16x8(alpha);
            accu00 = vmulq_f16(accu00, alpha_v);
            MlasStoreFloat16x8(C_data, accu00);
            if constexpr (CountM == 2) {
                accu10 = vmulq_f16(accu10, alpha_v);
                MlasStoreFloat16x8(C_data + ldc, accu10);
            }
        }
    }

    if (CountN & 4) {
        const auto* a = A_data;
        const auto* b = B_data;
        size_t k = CountK;
        float16x8_t accu00 = MlasZeroFloat16x8();
        float16x8_t accu01 = MlasZeroFloat16x8();
        float16x8_t accu02 = MlasZeroFloat16x8();
        float16x8_t accu03 = MlasZeroFloat16x8();
        float16x8_t accu10, accu11, accu12, accu13;
        if constexpr (CountM == 2) {
            accu10 = MlasZeroFloat16x8();
            accu11 = MlasZeroFloat16x8();
            accu12 = MlasZeroFloat16x8();
            accu13 = MlasZeroFloat16x8();
        }
        if (largeK) {
            float16x8_t b0 = MlasLoadFloat16x8(b);
            float16x8_t b1 = MlasLoadFloat16x8(b + ldb);
            float16x8_t b2 = MlasLoadFloat16x8(b + 2 * ldb);
            float16x8_t b3 = MlasLoadFloat16x8(b + 3 * ldb);
            float16x8_t a0 = MlasLoadFloat16x8(a);
            for (; k >= 16; k -= 8, a += 8, b += 8) {
                accu00 = vfmaq_f16(accu00, b0, a0);
                accu01 = vfmaq_f16(accu01, b1, a0);
                accu02 = vfmaq_f16(accu02, b2, a0);
                accu03 = vfmaq_f16(accu03, b3, a0);
                if constexpr (CountM == 2) {
                    float16x8_t a1 = MlasLoadFloat16x8(a + lda);
                    accu10 = vfmaq_f16(accu10, b0, a1);
                    accu11 = vfmaq_f16(accu11, b1, a1);
                    accu12 = vfmaq_f16(accu12, b2, a1);
                    accu13 = vfmaq_f16(accu13, b3, a1);
                }
                b0 = MlasLoadFloat16x8(b + 8);
                b1 = MlasLoadFloat16x8(b + 8 + ldb);
                b2 = MlasLoadFloat16x8(b + 8 + 2 * ldb);
                b3 = MlasLoadFloat16x8(b + 8 + 3 * ldb);
                a0 = MlasLoadFloat16x8(a + 8);
            }
            accu00 = vfmaq_f16(accu00, b0, a0);
            accu01 = vfmaq_f16(accu01, b1, a0);
            accu02 = vfmaq_f16(accu02, b2, a0);
            accu03 = vfmaq_f16(accu03, b3, a0);
            if constexpr (CountM == 2) {
                float16x8_t a1 = MlasLoadFloat16x8(a + lda);
                accu10 = vfmaq_f16(accu10, b0, a1);
                accu11 = vfmaq_f16(accu11, b1, a1);
                accu12 = vfmaq_f16(accu12, b2, a1);
                accu13 = vfmaq_f16(accu13, b3, a1);
            }
            k -= 8, a += 8, b += 8;
        }
        Transpose4x8(accu00, accu01, accu02, accu03);
        accu00 = addq_f16x4(accu00, accu01, accu02, accu03);
        float16x4_t accu0 = vadd_f16(vget_low_f16(accu00), vget_high_f16(accu00)), accu1;
        if constexpr (CountM == 2) {
            Transpose4x8(accu10, accu11, accu12, accu13);
            accu10 = addq_f16x4(accu10, accu11, accu12, accu13);
            accu1 = vadd_f16(vget_low_f16(accu10), vget_high_f16(accu10));
        }

        if (Kr3) {
            float16x4_t b0 = MlasLoadFloat16x4(b);
            float16x4_t b1 = MlasLoadFloat16x4(b + ldb);
            float16x4_t b2 = MlasLoadFloat16x4(b + 2 * ldb);
            float16x4_t b3 = MlasLoadFloat16x4(b + 3 * ldb);
            Transpose4x4(b0, b1, b2, b3);
            float16x4_t a0 = MlasLoadFloat16x4(a);
            accu0 = ma_lane_f16_accu(accu0, b0, b1, b2, b3, a0);
            if constexpr (CountM == 2) {
                float16x4_t a1 = MlasLoadFloat16x4(a + lda);
                accu1 = ma_lane_f16_accu(accu1, b0, b1, b2, b3, a1);
            }
            k -= 4, a += 4, b += 4;
        }

        if (Kr0) {
            float16x4_t b0 = MlasLoadPartialFloat16x4(b, k);
            float16x4_t b1 = MlasLoadPartialFloat16x4(b + ldb, k);
            float16x4_t b2 = MlasLoadPartialFloat16x4(b + 2 * ldb, k);
            float16x4_t b3 = MlasLoadPartialFloat16x4(b + 3 * ldb, k);
            Transpose4x4(b0, b1, b2, b3);
            float16x4_t a0 = MlasLoadPartialFloat16x4(a, k), a1;
            accu0 = vfma_lane_f16(accu0, b0, a0, 0);
            if constexpr (CountM == 2) {
                a1 = MlasLoadPartialFloat16x4(a + lda, k);
                accu1 = vfma_lane_f16(accu1, b0, a1, 0);
            }
            if (Kr1) {
                accu0 = vfma_lane_f16(accu0, b1, a0, 1);
                if constexpr (CountM == 2) {
                    accu1 = vfma_lane_f16(accu1, b1, a1, 1);
                }
            }
            if (Kr2) {
                accu0 = vfma_lane_f16(accu0, b2, a0, 2);
                if constexpr (CountM == 2) {
                    accu1 = vfma_lane_f16(accu1, b2, a1, 2);
                }
            }
        }

        if constexpr (beta_behavior == 1) {
            float16x4_t alpha_v = MlasBroadcastFloat16x4(alpha);
            float16x4_t c0 = MlasLoadFloat16x4(C_data);
            accu0 = vfma_f16(c0, accu0, alpha_v);
            MlasStoreFloat16x4(C_data, accu0);
            if constexpr (CountM == 2) {
                float16x4_t c1 = MlasLoadFloat16x4(C_data + ldc);
                accu1 = vfma_f16(c1, accu1, alpha_v);
                MlasStoreFloat16x4(C_data + ldc, accu1);
            }
        } else if constexpr (beta_behavior == 2) {
            float16x4_t alpha_v = MlasBroadcastFloat16x4(alpha);
            float16x4_t beta_v = MlasBroadcastFloat16x4(beta);
            float16x4_t c0 = MlasLoadFloat16x4(C_data);
            accu0 = vfma_f16(vmul_f16(c0, beta_v), accu0, alpha_v);
            MlasStoreFloat16x4(C_data, accu0);
            if constexpr (CountM == 2) {
                float16x4_t c1 = MlasLoadFloat16x4(C_data + ldc);
                accu1 = vfma_f16(vmul_f16(c1, beta_v), accu1, alpha_v);
                MlasStoreFloat16x4(C_data + ldc, accu1);
            }
        } else {
            float16x4_t alpha_v = MlasBroadcastFloat16x4(alpha);
            accu0 = vmul_f16(accu0, alpha_v);
            MlasStoreFloat16x4(C_data, accu0);
            if constexpr (CountM == 2) {
                accu1 = vmul_f16(accu1, alpha_v);
                MlasStoreFloat16x4(C_data + ldc, accu1);
            }
        }

        CountN -= 4, B_data += 4 * ldb, C_data += 4;
    }

    if (CountN > 0) {
        const auto* a = A_data;
        const auto* b = B_data;
        size_t k = CountK;
        float16x8_t accu0[4], accu1[4];
        size_t i = 0;
        for (i = 0; i < 4; ++i) {
            accu0[i] = MlasZeroFloat16x8();
            if constexpr (CountM == 2) {
                accu1[i] = MlasZeroFloat16x8();
            }
        }
        for (; k >= 8; k -= 8, a += 8, b += 8) {
            float16x8_t a0 = MlasLoadFloat16x8(a), a1;
            if constexpr (CountM == 2) {
                a1 = MlasLoadFloat16x8(a + lda);
            }
            for (i = 0; i < CountN; ++i) {
                float16x8_t bi = MlasLoadFloat16x8(b + i * ldb);
                accu0[i] = vfmaq_f16(accu0[i], bi, a0);
                if constexpr (CountM == 2) {
                    accu1[i] = vfmaq_f16(accu1[i], bi, a1);
                }
            }
        }
        Transpose4x8(accu0[0], accu0[1], accu0[2], accu0[3]);
        float16x8_t accu00 = addq_f16x4(accu0[0], accu0[1], accu0[2], accu0[3]);
        float16x4_t accu_0 = vadd_f16(vget_low_f16(accu00), vget_high_f16(accu00)), accu_1;
        if constexpr (CountM == 2) {
            Transpose4x8(accu1[0], accu1[1], accu1[2], accu1[3]);
            float16x8_t accu10 = addq_f16x4(accu1[0], accu1[1], accu1[2], accu1[3]);
            accu_1 = vadd_f16(vget_low_f16(accu10), vget_high_f16(accu10));
        }

        if (Kr3) {
            float16x4_t bs[4];
            for (i = 0; i < CountN; ++i) {
                bs[i] = MlasLoadFloat16x4(b + i * ldb);
            }
            for (; i < 4; ++i) {
                bs[i] = MlasZeroFloat16x4();
            }
            Transpose4x4(bs[0], bs[1], bs[2], bs[3]);
            float16x4_t a0 = MlasLoadFloat16x4(a);
            accu_0 = ma_lane_f16_accu(accu_0, bs[0], bs[1], bs[2], bs[3], a0);
            if constexpr (CountM == 2) {
                float16x4_t a1 = MlasLoadFloat16x4(a + lda);
                accu_1 = ma_lane_f16_accu(accu_1, bs[0], bs[1], bs[2], bs[3], a1);
            }
            k -= 4, a += 4, b += 4;
        }

        if (Kr0) {
            float16x4_t bs[4];
            for (i = 0; i < CountN; ++i) {
                bs[i] = MlasLoadPartialFloat16x4(b + i * ldb, k);
            }
            for (; i < 4; ++i) {
                bs[i] = MlasZeroFloat16x4();
            }
            Transpose4x4(bs[0], bs[1], bs[2], bs[3]);
            float16x4_t a0 = MlasLoadPartialFloat16x4(a, k), a1;
            accu_0 = vfma_lane_f16(accu_0, bs[0], a0, 0);
            if constexpr (CountM == 2) {
                a1 = MlasLoadPartialFloat16x4(a + lda, k);
                accu_1 = vfma_lane_f16(accu_1, bs[0], a1, 0);
            }
            if (Kr1) {
                accu_0 = vfma_lane_f16(accu_0, bs[1], a0, 1);
                if constexpr (CountM == 2) {
                    accu_1 = vfma_lane_f16(accu_1, bs[1], a1, 1);
                }
            }
            if (Kr2) {
                accu_0 = vfma_lane_f16(accu_0, bs[2], a0, 2);
                if constexpr (CountM == 2) {
                    accu_1 = vfma_lane_f16(accu_1, bs[2], a1, 2);
                }
            }
        }

        if constexpr (beta_behavior == 1) {
            float16x4_t alpha_v = MlasBroadcastFloat16x4(alpha);
            float16x4_t c0 = MlasLoadPartialFloat16x4(C_data, CountN);
            accu_0 = vfma_f16(c0, accu_0, alpha_v);
            MlasStorePartialFloat16x4(C_data, accu_0, CountN);
            if constexpr (CountM == 2) {
                float16x4_t c1 = MlasLoadPartialFloat16x4(C_data + ldc, CountN);
                accu_1 = vfma_f16(c1, accu_1, alpha_v);
                MlasStorePartialFloat16x4(C_data + ldc, accu_1, CountN);
            }
        } else if constexpr (beta_behavior == 2) {
            float16x4_t alpha_v = MlasBroadcastFloat16x4(alpha);
            float16x4_t beta_v = MlasBroadcastFloat16x4(beta);
            float16x4_t c0 = MlasLoadPartialFloat16x4(C_data, CountN);
            accu_0 = vfma_f16(vmul_f16(c0, beta_v), accu_0, alpha_v);
            MlasStorePartialFloat16x4(C_data, accu_0, CountN);
            if constexpr (CountM == 2) {
                float16x4_t c1 = MlasLoadPartialFloat16x4(C_data + ldc, CountN);
                accu_1 = vfma_f16(vmul_f16(c1, beta_v), accu_1, alpha_v);
                MlasStorePartialFloat16x4(C_data + ldc, accu_1, CountN);
            }
        } else {
            float16x4_t alpha_v = MlasBroadcastFloat16x4(alpha);
            accu_0 = vmul_f16(accu_0, alpha_v);
            MlasStorePartialFloat16x4(C_data, accu_0, CountN);
            if constexpr (CountM == 2) {
                accu_1 = vmul_f16(accu_1, alpha_v);
                MlasStorePartialFloat16x4(C_data + ldc, accu_1, CountN);
            }
        }
    }
}

// Full K. Directly save to C.
void HGemm_TransposedB_Kernel(
    const MLAS_FP16* A,
    const MLAS_FP16* B,
    MLAS_FP16* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc,
    _mlas_fp16_ alpha,
    _mlas_fp16_ beta
) {
    if (CountM > 2) {
        MLAS_THROW_EX(std::runtime_error, "HGemm_TransposedB_Kernel only support <= 2 rows");
    }
    const auto* A_data = reinterpret_cast<const _mlas_fp16_*>(A);
    const auto* B_data = reinterpret_cast<const _mlas_fp16_*>(B);
    auto* C_data = reinterpret_cast<_mlas_fp16_*>(C);
    const auto f16_0 = MLAS_FP16(0.0f);
    const auto f16_1 = MLAS_FP16(1.0f);
    if (CountM == 1) {
        if (beta == f16_0.val) {
            HGemm_TransposedB_Kernel_Impl<0, 1>(A_data, B_data, C_data, CountN, CountK, lda, ldb, ldc, alpha, beta);
        } else if (beta == f16_1.val) {
            HGemm_TransposedB_Kernel_Impl<1, 1>(A_data, B_data, C_data, CountN, CountK, lda, ldb, ldc, alpha, beta);
        } else {
            HGemm_TransposedB_Kernel_Impl<2, 1>(A_data, B_data, C_data, CountN, CountK, lda, ldb, ldc, alpha, beta);
        }
    } else {
        if (beta == f16_0.val) {
            HGemm_TransposedB_Kernel_Impl<0, 2>(A_data, B_data, C_data, CountN, CountK, lda, ldb, ldc, alpha, beta);
        } else if (beta == f16_1.val) {
            HGemm_TransposedB_Kernel_Impl<1, 2>(A_data, B_data, C_data, CountN, CountK, lda, ldb, ldc, alpha, beta);
        } else {
            HGemm_TransposedB_Kernel_Impl<2, 2>(A_data, B_data, C_data, CountN, CountK, lda, ldb, ldc, alpha, beta);
        }
    }
}

// handle C = alpha * A * B + beta * C where alpha != 1 or beta != 0 or 1
template <int CountM>
void HGemm_B_Kernel_Complicated(
    const _mlas_fp16_* A_data,
    const _mlas_fp16_* B_data,
    _mlas_fp16_* C_data,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc,
    _mlas_fp16_ alpha,
    _mlas_fp16_ beta
) {
    const size_t ldb4 = ldb * 4;
    float16x8_t alpha_v8 = MlasBroadcastFloat16x8(alpha);
    float16x8_t beta_v8 = MlasBroadcastFloat16x8(beta);
    float16x4_t alpha_v4 = MlasBroadcastFloat16x4(alpha);
    float16x4_t beta_v4 = MlasBroadcastFloat16x4(beta);
    const bool largeK = CountK >= 4;
    const bool Kr0 = CountK & 3;
    const bool Kr1 = (CountK & 3 ) > 1;
    const bool Kr2 = (CountK & 3 ) > 2;
    for (; CountN >= 32; CountN -= 32, B_data += 32, C_data += 32) {
        const auto* a = A_data;
        const auto* b = B_data;
        size_t k = CountK;
        float16x8_t accu00 = MlasZeroFloat16x8();
        float16x8_t accu01 = MlasZeroFloat16x8();
        float16x8_t accu02 = MlasZeroFloat16x8();
        float16x8_t accu03 = MlasZeroFloat16x8();
        float16x8_t accu10, accu11, accu12, accu13;
        if constexpr (CountM == 2) {
            accu10 = MlasZeroFloat16x8();
            accu11 = MlasZeroFloat16x8();
            accu12 = MlasZeroFloat16x8();
            accu13 = MlasZeroFloat16x8();
        }
        if (largeK) {
            float16x4_t a0 = MlasLoadFloat16x4(a), a1;
            if constexpr (CountM == 2) {
                a1 = MlasLoadFloat16x4(a + lda);
            }
            float16x8_t b00 = MlasLoadFloat16x8(b);
            float16x8_t b10 = MlasLoadFloat16x8(b + ldb);
            float16x8_t b20 = MlasLoadFloat16x8(b + 2 * ldb);
            float16x8_t b30 = MlasLoadFloat16x8(b + 3 * ldb);
            float16x8_t b01 = MlasLoadFloat16x8(b + 8);
            float16x8_t b11 = MlasLoadFloat16x8(b + ldb + 8);
            float16x8_t b21 = MlasLoadFloat16x8(b + 2 * ldb + 8);
            float16x8_t b31 = MlasLoadFloat16x8(b + 3 * ldb + 8);
            float16x8_t b02 = MlasLoadFloat16x8(b + 16);
            float16x8_t b12 = MlasLoadFloat16x8(b + ldb + 16);
            float16x8_t b22 = MlasLoadFloat16x8(b + 2 * ldb + 16);
            float16x8_t b32 = MlasLoadFloat16x8(b + 3 * ldb + 16);
            float16x8_t b03 = MlasLoadFloat16x8(b + 24);
            float16x8_t b13 = MlasLoadFloat16x8(b + ldb + 24);
            float16x8_t b23 = MlasLoadFloat16x8(b + 2 * ldb + 24);
            float16x8_t b33 = MlasLoadFloat16x8(b + 3 * ldb + 24);
            for (; k >= 8; k -= 4, a += 4, b += ldb4) {
                accu00 = maq_lane_f16_accu(accu00, b00, b10, b20, b30, a0);
                accu01 = maq_lane_f16_accu(accu01, b01, b11, b21, b31, a0);
                accu02 = maq_lane_f16_accu(accu02, b02, b12, b22, b32, a0);
                accu03 = maq_lane_f16_accu(accu03, b03, b13, b23, b33, a0);
                if constexpr (CountM == 2) {
                    accu10 = maq_lane_f16_accu(accu10, b00, b10, b20, b30, a1);
                    accu11 = maq_lane_f16_accu(accu11, b01, b11, b21, b31, a1);
                    accu12 = maq_lane_f16_accu(accu12, b02, b12, b22, b32, a1);
                    accu13 = maq_lane_f16_accu(accu13, b03, b13, b23, b33, a1);
                }
                a0 = MlasLoadFloat16x4(a + 4);
                if constexpr (CountM == 2) {
                    a1 = MlasLoadFloat16x4(a + 4 + lda);
                }
                b00 = MlasLoadFloat16x8(b + ldb4);
                b10 = MlasLoadFloat16x8(b + 5 * ldb);
                b20 = MlasLoadFloat16x8(b + 6 * ldb);
                b30 = MlasLoadFloat16x8(b + 7 * ldb);
                b01 = MlasLoadFloat16x8(b + ldb4 + 8);
                b11 = MlasLoadFloat16x8(b + 5 * ldb + 8);
                b21 = MlasLoadFloat16x8(b + 6 * ldb + 8);
                b31 = MlasLoadFloat16x8(b + 7 * ldb + 8);
                b02 = MlasLoadFloat16x8(b + ldb4 + 16);
                b12 = MlasLoadFloat16x8(b + 5 * ldb + 16);
                b22 = MlasLoadFloat16x8(b + 6 * ldb + 16);
                b32 = MlasLoadFloat16x8(b + 7 * ldb + 16);
                b03 = MlasLoadFloat16x8(b + ldb4 + 24);
                b13 = MlasLoadFloat16x8(b + 5 * ldb + 24);
                b23 = MlasLoadFloat16x8(b + 6 * ldb + 24);
                b33 = MlasLoadFloat16x8(b + 7 * ldb + 24);
            }
            accu00 = maq_lane_f16_accu(accu00, b00, b10, b20, b30, a0);
            accu01 = maq_lane_f16_accu(accu01, b01, b11, b21, b31, a0);
            accu02 = maq_lane_f16_accu(accu02, b02, b12, b22, b32, a0);
            accu03 = maq_lane_f16_accu(accu03, b03, b13, b23, b33, a0);
            if constexpr (CountM == 2) {
                accu10 = maq_lane_f16_accu(accu10, b00, b10, b20, b30, a1);
                accu11 = maq_lane_f16_accu(accu11, b01, b11, b21, b31, a1);
                accu12 = maq_lane_f16_accu(accu12, b02, b12, b22, b32, a1);
                accu13 = maq_lane_f16_accu(accu13, b03, b13, b23, b33, a1);
            }
            k -= 4, a += 4, b += ldb4;
        }

        if (Kr0) {
            float16x4_t a0 = MlasLoadPartialFloat16x4(a, k), a1;
            if constexpr (CountM == 2) {
                a1 = MlasLoadPartialFloat16x4(a + lda, k);
            }
            float16x8_t b00 = MlasLoadFloat16x8(b);
            float16x8_t b01 = MlasLoadFloat16x8(b + 8);
            float16x8_t b02 = MlasLoadFloat16x8(b + 16);
            float16x8_t b03 = MlasLoadFloat16x8(b + 24);
            accu00 = vfmaq_lane_f16(accu00, b00, a0, 0);
            accu01 = vfmaq_lane_f16(accu01, b01, a0, 0);
            accu02 = vfmaq_lane_f16(accu02, b02, a0, 0);
            accu03 = vfmaq_lane_f16(accu03, b03, a0, 0);
            if constexpr (CountM == 2) {
                accu10 = vfmaq_lane_f16(accu10, b00, a1, 0);
                accu11 = vfmaq_lane_f16(accu11, b01, a1, 0);
                accu12 = vfmaq_lane_f16(accu12, b02, a1, 0);
                accu13 = vfmaq_lane_f16(accu13, b03, a1, 0);
            }
            if (Kr1) {
                float16x8_t b10 = MlasLoadFloat16x8(b + ldb);
                float16x8_t b11 = MlasLoadFloat16x8(b + ldb + 8);
                float16x8_t b12 = MlasLoadFloat16x8(b + ldb + 16);
                float16x8_t b13 = MlasLoadFloat16x8(b + ldb + 24);
                accu00 = vfmaq_lane_f16(accu00, b10, a0, 1);
                accu01 = vfmaq_lane_f16(accu01, b11, a0, 1);
                accu02 = vfmaq_lane_f16(accu02, b12, a0, 1);
                accu03 = vfmaq_lane_f16(accu03, b13, a0, 1);
                if constexpr (CountM == 2) {
                    accu10 = vfmaq_lane_f16(accu10, b10, a1, 1);
                    accu11 = vfmaq_lane_f16(accu11, b11, a1, 1);
                    accu12 = vfmaq_lane_f16(accu12, b12, a1, 1);
                    accu13 = vfmaq_lane_f16(accu13, b13, a1, 1);
                }
            }
            if (Kr2) {
                float16x8_t b20 = MlasLoadFloat16x8(b + 2 * ldb);
                float16x8_t b21 = MlasLoadFloat16x8(b + 2 * ldb + 8);
                float16x8_t b22 = MlasLoadFloat16x8(b + 2 * ldb + 16);
                float16x8_t b23 = MlasLoadFloat16x8(b + 2 * ldb + 24);
                accu00 = vfmaq_lane_f16(accu00, b20, a0, 2);
                accu01 = vfmaq_lane_f16(accu01, b21, a0, 2);
                accu02 = vfmaq_lane_f16(accu02, b22, a0, 2);
                accu03 = vfmaq_lane_f16(accu03, b23, a0, 2);
                if constexpr (CountM == 2) {
                    accu10 = vfmaq_lane_f16(accu10, b20, a1, 2);
                    accu11 = vfmaq_lane_f16(accu11, b21, a1, 2);
                    accu12 = vfmaq_lane_f16(accu12, b22, a1, 2);
                    accu13 = vfmaq_lane_f16(accu13, b23, a1, 2);
                }
            }
        }

        float16x8_t c00 = MlasLoadFloat16x8(C_data);
        float16x8_t c01 = MlasLoadFloat16x8(C_data + 8);
        float16x8_t c02 = MlasLoadFloat16x8(C_data + 16);
        float16x8_t c03 = MlasLoadFloat16x8(C_data + 24);
        MlasStoreFloat16x8(C_data, vfmaq_f16(vmulq_f16(c00, beta_v8), accu00, alpha_v8));
        MlasStoreFloat16x8(C_data + 8, vfmaq_f16(vmulq_f16(c01, beta_v8), accu01, alpha_v8));
        MlasStoreFloat16x8(C_data + 16, vfmaq_f16(vmulq_f16(c02, beta_v8), accu02, alpha_v8));
        MlasStoreFloat16x8(C_data + 24, vfmaq_f16(vmulq_f16(c03, beta_v8), accu03, alpha_v8));
        if constexpr (CountM == 2) {
            float16x8_t c10 = MlasLoadFloat16x8(C_data + ldc);
            float16x8_t c11 = MlasLoadFloat16x8(C_data + ldc + 8);
            float16x8_t c12 = MlasLoadFloat16x8(C_data + ldc + 16);
            float16x8_t c13 = MlasLoadFloat16x8(C_data + ldc + 24);
            MlasStoreFloat16x8(C_data + ldc, vfmaq_f16(vmulq_f16(c10, beta_v8), accu10, alpha_v8));
            MlasStoreFloat16x8(C_data + ldc + 8, vfmaq_f16(vmulq_f16(c11, beta_v8), accu11, alpha_v8));
            MlasStoreFloat16x8(C_data + ldc + 16, vfmaq_f16(vmulq_f16(c12, beta_v8), accu12, alpha_v8));
            MlasStoreFloat16x8(C_data + ldc + 24, vfmaq_f16(vmulq_f16(c13, beta_v8), accu13, alpha_v8));
        }
    }

    if (CountN & 16) {
        const auto* a = A_data;
        const auto* b = B_data;
        size_t k = CountK;
        float16x8_t accu00 = MlasZeroFloat16x8();
        float16x8_t accu01 = MlasZeroFloat16x8();
        float16x8_t accu10, accu11;
        if constexpr (CountM == 2) {
            accu10 = MlasZeroFloat16x8();
            accu11 = MlasZeroFloat16x8();
        }
        if (largeK) {
            float16x4_t a0 = MlasLoadFloat16x4(a), a1;
            if constexpr (CountM == 2) {
                a1 = MlasLoadFloat16x4(a + lda);
            }
            float16x8_t b00 = MlasLoadFloat16x8(b);
            float16x8_t b10 = MlasLoadFloat16x8(b + ldb);
            float16x8_t b20 = MlasLoadFloat16x8(b + 2 * ldb);
            float16x8_t b30 = MlasLoadFloat16x8(b + 3 * ldb);
            float16x8_t b01 = MlasLoadFloat16x8(b + 8);
            float16x8_t b11 = MlasLoadFloat16x8(b + ldb + 8);
            float16x8_t b21 = MlasLoadFloat16x8(b + 2 * ldb + 8);
            float16x8_t b31 = MlasLoadFloat16x8(b + 3 * ldb + 8);
            for (; k >= 8; k -= 4, a += 4, b += ldb4) {
                accu00 = maq_lane_f16_accu(accu00, b00, b10, b20, b30, a0);
                accu01 = maq_lane_f16_accu(accu01, b01, b11, b21, b31, a0);
                if constexpr (CountM == 2) {
                    accu10 = maq_lane_f16_accu(accu10, b00, b10, b20, b30, a1);
                    accu11 = maq_lane_f16_accu(accu11, b01, b11, b21, b31, a1);
                }
                a0 = MlasLoadFloat16x4(a + 4);
                if constexpr (CountM == 2) {
                    a1 = MlasLoadFloat16x4(a + 4 + lda);
                }
                b00 = MlasLoadFloat16x8(b + ldb4);
                b10 = MlasLoadFloat16x8(b + 5 * ldb);
                b20 = MlasLoadFloat16x8(b + 6 * ldb);
                b30 = MlasLoadFloat16x8(b + 7 * ldb);
                b01 = MlasLoadFloat16x8(b + ldb4 + 8);
                b11 = MlasLoadFloat16x8(b + 5 * ldb + 8);
                b21 = MlasLoadFloat16x8(b + 6 * ldb + 8);
                b31 = MlasLoadFloat16x8(b + 7 * ldb + 8);

            }
            accu00 = maq_lane_f16_accu(accu00, b00, b10, b20, b30, a0);
            accu01 = maq_lane_f16_accu(accu01, b01, b11, b21, b31, a0);
            if constexpr (CountM == 2) {
                accu10 = maq_lane_f16_accu(accu10, b00, b10, b20, b30, a1);
                accu11 = maq_lane_f16_accu(accu11, b01, b11, b21, b31, a1);
            }
            k -= 4, a += 4, b += ldb4;
        }

        if (Kr0) {
            float16x4_t a0 = MlasLoadPartialFloat16x4(a, k), a1;
            if constexpr (CountM == 2) {
                a1 = MlasLoadPartialFloat16x4(a + lda, k);
            }
            float16x8_t b00 = MlasLoadFloat16x8(b);
            float16x8_t b01 = MlasLoadFloat16x8(b + 8);
            accu00 = vfmaq_lane_f16(accu00, b00, a0, 0);
            accu01 = vfmaq_lane_f16(accu01, b01, a0, 0);
            if constexpr (CountM == 2) {
                accu10 = vfmaq_lane_f16(accu10, b00, a1, 0);
                accu11 = vfmaq_lane_f16(accu11, b01, a1, 0);
            }
            if (Kr1) {
                float16x8_t b10 = MlasLoadFloat16x8(b + ldb);
                float16x8_t b11 = MlasLoadFloat16x8(b + ldb + 8);
                accu00 = vfmaq_lane_f16(accu00, b10, a0, 1);
                accu01 = vfmaq_lane_f16(accu01, b11, a0, 1);
                if constexpr (CountM == 2) {
                    accu10 = vfmaq_lane_f16(accu10, b10, a1, 1);
                    accu11 = vfmaq_lane_f16(accu11, b11, a1, 1);
                }
            }
            if (Kr2) {
                float16x8_t b20 = MlasLoadFloat16x8(b + 2 * ldb);
                float16x8_t b21 = MlasLoadFloat16x8(b + 2 * ldb + 8);
                accu00 = vfmaq_lane_f16(accu00, b20, a0, 2);
                accu01 = vfmaq_lane_f16(accu01, b21, a0, 2);
                if constexpr (CountM == 2) {
                    accu10 = vfmaq_lane_f16(accu10, b20, a1, 2);
                    accu11 = vfmaq_lane_f16(accu11, b21, a1, 2);
                }
            }
        }

        float16x8_t c00 = MlasLoadFloat16x8(C_data);
        float16x8_t c01 = MlasLoadFloat16x8(C_data + 8);
        MlasStoreFloat16x8(C_data, vfmaq_f16(vmulq_f16(c00, beta_v8), accu00, alpha_v8));
        MlasStoreFloat16x8(C_data + 8, vfmaq_f16(vmulq_f16(c01, beta_v8), accu01, alpha_v8));
        if constexpr (CountM == 2) {
            float16x8_t c10 = MlasLoadFloat16x8(C_data + ldc);
            float16x8_t c11 = MlasLoadFloat16x8(C_data + ldc + 8);
            MlasStoreFloat16x8(C_data + ldc, vfmaq_f16(vmulq_f16(c10, beta_v8), accu10, alpha_v8));
            MlasStoreFloat16x8(C_data + ldc + 8, vfmaq_f16(vmulq_f16(c11, beta_v8), accu11, alpha_v8));
        }

        CountN -= 16, B_data += 16, C_data += 16;
    }

    if (CountN & 8) {
        const auto* a = A_data;
        const auto* b = B_data;
        size_t k = CountK;
        float16x8_t accu00 = MlasZeroFloat16x8();
        float16x8_t accu10;
        if constexpr (CountM == 2) {
            accu10 = MlasZeroFloat16x8();
        }
        if (largeK) {
            float16x4_t a0 = MlasLoadFloat16x4(a), a1;
            if constexpr (CountM == 2) {
                a1 = MlasLoadFloat16x4(a + lda);
            }
            float16x8_t b00 = MlasLoadFloat16x8(b);
            float16x8_t b10 = MlasLoadFloat16x8(b + ldb);
            float16x8_t b20 = MlasLoadFloat16x8(b + 2 * ldb);
            float16x8_t b30 = MlasLoadFloat16x8(b + 3 * ldb);
            for (; k >= 8; k -= 4, a += 4, b += ldb4) {
                accu00 = maq_lane_f16_accu(accu00, b00, b10, b20, b30, a0);
                if constexpr (CountM == 2) {
                    accu10 = maq_lane_f16_accu(accu10, b00, b10, b20, b30, a1);
                }
                a0 = MlasLoadFloat16x4(a + 4);
                if constexpr (CountM == 2) {
                    a1 = MlasLoadFloat16x4(a + 4 + lda);
                }
                b00 = MlasLoadFloat16x8(b + ldb4);
                b10 = MlasLoadFloat16x8(b + 5 * ldb);
                b20 = MlasLoadFloat16x8(b + 6 * ldb);
                b30 = MlasLoadFloat16x8(b + 7 * ldb);
            }
            accu00 = maq_lane_f16_accu(accu00, b00, b10, b20, b30, a0);
            if constexpr (CountM == 2) {
                accu10 = maq_lane_f16_accu(accu10, b00, b10, b20, b30, a1);
            }
            k -= 4, a += 4, b += ldb4;
        }

        if (Kr0) {
            float16x4_t a0 = MlasLoadPartialFloat16x4(a, k), a1;
            if constexpr (CountM == 2) {
                a1 = MlasLoadPartialFloat16x4(a + lda, k);
            }
            float16x8_t b00 = MlasLoadFloat16x8(b);
            accu00 = vfmaq_lane_f16(accu00, b00, a0, 0);
            if constexpr (CountM == 2) {
                accu10 = vfmaq_lane_f16(accu10, b00, a1, 0);
            }
            if (Kr1) {
                float16x8_t b10 = MlasLoadFloat16x8(b + ldb);
                accu00 = vfmaq_lane_f16(accu00, b10, a0, 1);
                if constexpr (CountM == 2) {
                    accu10 = vfmaq_lane_f16(accu10, b10, a1, 1);
                }
            }
            if (Kr2) {
                float16x8_t b20 = MlasLoadFloat16x8(b + 2 * ldb);
                accu00 = vfmaq_lane_f16(accu00, b20, a0, 2);
                if constexpr (CountM == 2) {
                    accu10 = vfmaq_lane_f16(accu10, b20, a1, 2);
                }
            }
        }

        float16x8_t c00 = MlasLoadFloat16x8(C_data);
        MlasStoreFloat16x8(C_data, vfmaq_f16(vmulq_f16(c00, beta_v8), accu00, alpha_v8));
        if constexpr (CountM == 2) {
            float16x8_t c10 = MlasLoadFloat16x8(C_data + ldc);
            MlasStoreFloat16x8(C_data + ldc, vfmaq_f16(vmulq_f16(c10, beta_v8), accu10, alpha_v8));
        }

        CountN -= 8, B_data += 8, C_data += 8;
    }

    if (CountN & 4) {
        const auto* a = A_data;
        const auto* b = B_data;
        size_t k = CountK;
        float16x4_t accu00 = MlasZeroFloat16x4();
        float16x4_t accu10;
        if constexpr (CountM == 2) {
            accu10 = MlasZeroFloat16x4();
        }
        if (largeK) {
            float16x4_t a0 = MlasLoadFloat16x4(a), a1;
            if constexpr (CountM == 2) {
                a1 = MlasLoadFloat16x4(a + lda);
            }
            float16x4_t b00 = MlasLoadFloat16x4(b);
            float16x4_t b10 = MlasLoadFloat16x4(b + ldb);
            float16x4_t b20 = MlasLoadFloat16x4(b + 2 * ldb);
            float16x4_t b30 = MlasLoadFloat16x4(b + 3 * ldb);
            for (; k >= 8; k -= 4, a += 4, b += ldb4) {
                accu00 = ma_lane_f16_accu(accu00, b00, b10, b20, b30, a0);
                if constexpr (CountM == 2) {
                    accu10 = ma_lane_f16_accu(accu10, b00, b10, b20, b30, a1);
                }
                a0 = MlasLoadFloat16x4(a + 4);
                if constexpr (CountM == 2) {
                    a1 = MlasLoadFloat16x4(a + 4 + lda);
                }
                b00 = MlasLoadFloat16x4(b + ldb4);
                b10 = MlasLoadFloat16x4(b + 5 * ldb);
                b20 = MlasLoadFloat16x4(b + 6 * ldb);
                b30 = MlasLoadFloat16x4(b + 7 * ldb);
            }
            accu00 = ma_lane_f16_accu(accu00, b00, b10, b20, b30, a0);
            if constexpr (CountM == 2) {
                accu10 = ma_lane_f16_accu(accu10, b00, b10, b20, b30, a1);
            }
            k -= 4, a += 4, b += ldb4;
        }

        if (Kr0) {
            float16x4_t a0 = MlasLoadPartialFloat16x4(a, k), a1;
            if constexpr (CountM == 2) {
                a1 = MlasLoadPartialFloat16x4(a + lda, k);
            }
            float16x4_t b00 = MlasLoadFloat16x4(b);
            accu00 = vfma_lane_f16(accu00, b00, a0, 0);
            if constexpr (CountM == 2) {
                accu10 = vfma_lane_f16(accu10, b00, a1, 0);
            }
            if (Kr1) {
                float16x4_t b10 = MlasLoadFloat16x4(b + ldb);
                accu00 = vfma_lane_f16(accu00, b10, a0, 1);
                if constexpr (CountM == 2) {
                    accu10 = vfma_lane_f16(accu10, b10, a1, 1);
                }
            }
            if (Kr2) {
                float16x4_t b20 = MlasLoadFloat16x4(b + 2 * ldb);
                accu00 = vfma_lane_f16(accu00, b20, a0, 2);
                if constexpr (CountM == 2) {
                    accu10 = vfma_lane_f16(accu10, b20, a1, 2);
                }
            }
        }

        float16x4_t c00 = MlasLoadFloat16x4(C_data);
        MlasStoreFloat16x4(C_data, vfma_f16(vmul_f16(c00, beta_v4), accu00, alpha_v4));
        if constexpr (CountM == 2) {
            float16x4_t c10 = MlasLoadFloat16x4(C_data + ldc);
            MlasStoreFloat16x4(C_data + ldc, vfma_f16(vmul_f16(c10, beta_v4), accu10, alpha_v4));
        }

        CountN -= 4, B_data += 4, C_data += 4;
    }

    if (CountN > 0) {
        const auto* a = A_data;
        const auto* b = B_data;
        size_t k = CountK;
        float16x4_t accu00 = MlasZeroFloat16x4();
        float16x4_t accu10;
        if constexpr (CountM == 2) {
            accu10 = MlasZeroFloat16x4();
        }
        if (largeK) {
            float16x4_t a0 = MlasLoadFloat16x4(a), a1;
            if constexpr (CountM == 2) {
                a1 = MlasLoadFloat16x4(a + lda);
            }
            float16x4_t b00 = MlasLoadPartialFloat16x4(b, CountN);
            float16x4_t b10 = MlasLoadPartialFloat16x4(b + ldb, CountN);
            float16x4_t b20 = MlasLoadPartialFloat16x4(b + 2 * ldb, CountN);
            float16x4_t b30 = MlasLoadPartialFloat16x4(b + 3 * ldb, CountN);
            for (; k >= 8; k -= 4, a += 4, b += ldb4) {
                accu00 = ma_lane_f16_accu(accu00, b00, b10, b20, b30, a0);
                if constexpr (CountM == 2) {
                    accu10 = ma_lane_f16_accu(accu10, b00, b10, b20, b30, a1);
                }
                a0 = MlasLoadFloat16x4(a + 4);
                if constexpr (CountM == 2) {
                    a1 = MlasLoadFloat16x4(a + 4 + lda);
                }
                b00 = MlasLoadPartialFloat16x4(b + ldb4, CountN);
                b10 = MlasLoadPartialFloat16x4(b + 5 * ldb, CountN);
                b20 = MlasLoadPartialFloat16x4(b + 6 * ldb, CountN);
                b30 = MlasLoadPartialFloat16x4(b + 7 * ldb, CountN);
            }
            accu00 = ma_lane_f16_accu(accu00, b00, b10, b20, b30, a0);
            if constexpr (CountM == 2) {
                accu10 = ma_lane_f16_accu(accu10, b00, b10, b20, b30, a1);
            }
            k -= 4, a += 4, b += ldb4;
        }

        if (Kr0) {
            float16x4_t a0 = MlasLoadPartialFloat16x4(a, k), a1;
            if constexpr (CountM == 2) {
                a1 = MlasLoadPartialFloat16x4(a + lda, k);
            }
            float16x4_t b00 = MlasLoadPartialFloat16x4(b, CountN);
            accu00 = vfma_lane_f16(accu00, b00, a0, 0);
            if constexpr (CountM == 2) {
                accu10 = vfma_lane_f16(accu10, b00, a1, 0);
            }
            if (Kr1) {
                float16x4_t b10 = MlasLoadPartialFloat16x4(b + ldb, CountN);
                accu00 = vfma_lane_f16(accu00, b10, a0, 1);
                if constexpr (CountM == 2) {
                    accu10 = vfma_lane_f16(accu10, b10, a1, 1);
                }
            }
            if (Kr2) {
                float16x4_t b20 = MlasLoadPartialFloat16x4(b + 2 * ldb, CountN);
                accu00 = vfma_lane_f16(accu00, b20, a0, 2);
                if constexpr (CountM == 2) {
                    accu10 = vfma_lane_f16(accu10, b20, a1, 2);
                }
            }
        }

        float16x4_t c00 = MlasLoadPartialFloat16x4(C_data, CountN);
        MlasStorePartialFloat16x4(C_data, vfma_f16(vmul_f16(c00, beta_v4), accu00, alpha_v4), CountN);
        if constexpr (CountM == 2) {
            float16x4_t c10 = MlasLoadPartialFloat16x4(C_data + ldc, CountN);
            MlasStorePartialFloat16x4(C_data + ldc, vfma_f16(vmul_f16(c10, beta_v4), accu10, alpha_v4), CountN);
        }
    }
}

// Handle C = A * B + C or C = A * B
template <int CountM, bool zero_mode>
void HGemm_B_Kernel_Simple(
    const _mlas_fp16_* A_data,
    const _mlas_fp16_* B_data,
    _mlas_fp16_* C_data,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc
) {
    const size_t ldb4 = ldb * 4;
    const bool largeN = CountN >= 32;
    const bool Nr0 = (CountN % 4) > 0;
    const bool Kr1 = (CountK % 4) > 1;
    const bool Kr2 = (CountK % 4) > 2;
    if constexpr (zero_mode) {
        // process first K
        if (CountK >= 4) {
            float16x4_t a0 = MlasLoadFloat16x4(A_data), a1;
            if constexpr (CountM == 2) {
                a1 = MlasLoadFloat16x4(A_data + lda);
            }
            size_t n = CountN;
            const auto* b = B_data;
            auto* c = C_data;
            if (largeN) {
                float16x8_t b00 = MlasLoadFloat16x8(b);
                float16x8_t b10 = MlasLoadFloat16x8(b + ldb);
                float16x8_t b20 = MlasLoadFloat16x8(b + 2 * ldb);
                float16x8_t b30 = MlasLoadFloat16x8(b + 3 * ldb);
                float16x8_t b01 = MlasLoadFloat16x8(b + 8);
                float16x8_t b11 = MlasLoadFloat16x8(b + ldb + 8);
                float16x8_t b21 = MlasLoadFloat16x8(b + 2 * ldb + 8);
                float16x8_t b31 = MlasLoadFloat16x8(b + 3 * ldb + 8);
                float16x8_t b02 = MlasLoadFloat16x8(b + 16);
                float16x8_t b12 = MlasLoadFloat16x8(b + ldb + 16);
                float16x8_t b22 = MlasLoadFloat16x8(b + 2 * ldb + 16);
                float16x8_t b32 = MlasLoadFloat16x8(b + 3 * ldb + 16);
                float16x8_t b03 = MlasLoadFloat16x8(b + 24);
                float16x8_t b13 = MlasLoadFloat16x8(b + ldb + 24);
                float16x8_t b23 = MlasLoadFloat16x8(b + 2 * ldb + 24);
                float16x8_t b33 = MlasLoadFloat16x8(b + 3 * ldb + 24);
                for (; n >= 64; n -= 32, b += 32, c += 32) {
                    float16x8_t accu00 = MlasZeroFloat16x8();
                    float16x8_t accu01 = MlasZeroFloat16x8();
                    float16x8_t accu02 = MlasZeroFloat16x8();
                    float16x8_t accu03 = MlasZeroFloat16x8();
                    accu00 = maq_lane_f16_accu(accu00, b00, b10, b20, b30, a0);
                    accu01 = maq_lane_f16_accu(accu01, b01, b11, b21, b31, a0);
                    accu02 = maq_lane_f16_accu(accu02, b02, b12, b22, b32, a0);
                    accu03 = maq_lane_f16_accu(accu03, b03, b13, b23, b33, a0);
                    MlasStoreFloat16x8(c, accu00);
                    MlasStoreFloat16x8(c + 8, accu01);
                    MlasStoreFloat16x8(c + 16, accu02);
                    MlasStoreFloat16x8(c + 24, accu03);
                    if constexpr (CountM == 2) {
                        float16x8_t accu10 = MlasZeroFloat16x8();
                        float16x8_t accu11 = MlasZeroFloat16x8();
                        float16x8_t accu12 = MlasZeroFloat16x8();
                        float16x8_t accu13 = MlasZeroFloat16x8();
                        accu10 = maq_lane_f16_accu(accu10, b00, b10, b20, b30, a1);
                        accu11 = maq_lane_f16_accu(accu11, b01, b11, b21, b31, a1);
                        accu12 = maq_lane_f16_accu(accu12, b02, b12, b22, b32, a1);
                        accu13 = maq_lane_f16_accu(accu13, b03, b13, b23, b33, a1);
                        MlasStoreFloat16x8(c + ldc, accu10);
                        MlasStoreFloat16x8(c + ldc + 8, accu11);
                        MlasStoreFloat16x8(c + ldc + 16, accu12);
                        MlasStoreFloat16x8(c + ldc + 24, accu13);
                    }
                    b00 = MlasLoadFloat16x8(b + 32);
                    b10 = MlasLoadFloat16x8(b + ldb + 32);
                    b20 = MlasLoadFloat16x8(b + 2 * ldb + 32);
                    b30 = MlasLoadFloat16x8(b + 3 * ldb + 32);
                    b01 = MlasLoadFloat16x8(b + 40);
                    b11 = MlasLoadFloat16x8(b + ldb + 40);
                    b21 = MlasLoadFloat16x8(b + 2 * ldb + 40);
                    b31 = MlasLoadFloat16x8(b + 3 * ldb + 40);
                    b02 = MlasLoadFloat16x8(b + 48);
                    b12 = MlasLoadFloat16x8(b + ldb + 48);
                    b22 = MlasLoadFloat16x8(b + 2 * ldb + 48);
                    b32 = MlasLoadFloat16x8(b + 3 * ldb + 48);
                    b03 = MlasLoadFloat16x8(b + 56);
                    b13 = MlasLoadFloat16x8(b + ldb + 56);
                    b23 = MlasLoadFloat16x8(b + 2 * ldb + 56);
                    b33 = MlasLoadFloat16x8(b + 3 * ldb + 56);
                }
                float16x8_t accu00 = MlasZeroFloat16x8();
                float16x8_t accu01 = MlasZeroFloat16x8();
                float16x8_t accu02 = MlasZeroFloat16x8();
                float16x8_t accu03 = MlasZeroFloat16x8();
                accu00 = maq_lane_f16_accu(accu00, b00, b10, b20, b30, a0);
                accu01 = maq_lane_f16_accu(accu01, b01, b11, b21, b31, a0);
                accu02 = maq_lane_f16_accu(accu02, b02, b12, b22, b32, a0);
                accu03 = maq_lane_f16_accu(accu03, b03, b13, b23, b33, a0);
                MlasStoreFloat16x8(c, accu00);
                MlasStoreFloat16x8(c + 8, accu01);
                MlasStoreFloat16x8(c + 16, accu02);
                MlasStoreFloat16x8(c + 24, accu03);
                if constexpr (CountM == 2) {
                    float16x8_t accu10 = MlasZeroFloat16x8();
                    float16x8_t accu11 = MlasZeroFloat16x8();
                    float16x8_t accu12 = MlasZeroFloat16x8();
                    float16x8_t accu13 = MlasZeroFloat16x8();
                    accu10 = maq_lane_f16_accu(accu10, b00, b10, b20, b30, a1);
                    accu11 = maq_lane_f16_accu(accu11, b01, b11, b21, b31, a1);
                    accu12 = maq_lane_f16_accu(accu12, b02, b12, b22, b32, a1);
                    accu13 = maq_lane_f16_accu(accu13, b03, b13, b23, b33, a1);
                    MlasStoreFloat16x8(c + ldc, accu10);
                    MlasStoreFloat16x8(c + ldc + 8, accu11);
                    MlasStoreFloat16x8(c + ldc + 16, accu12);
                    MlasStoreFloat16x8(c + ldc + 24, accu13);
                }
                n -= 32, b += 32, c += 32;
            }
            if (n & 16) {
                float16x8_t accu00 = MlasZeroFloat16x8();
                float16x8_t accu01 = MlasZeroFloat16x8();
                float16x8_t b00 = MlasLoadFloat16x8(b);
                float16x8_t b10 = MlasLoadFloat16x8(b + ldb);
                float16x8_t b20 = MlasLoadFloat16x8(b + 2 * ldb);
                float16x8_t b30 = MlasLoadFloat16x8(b + 3 * ldb);
                float16x8_t b01 = MlasLoadFloat16x8(b + 8);
                float16x8_t b11 = MlasLoadFloat16x8(b + ldb + 8);
                float16x8_t b21 = MlasLoadFloat16x8(b + 2 * ldb + 8);
                float16x8_t b31 = MlasLoadFloat16x8(b + 3 * ldb + 8);
                accu00 = maq_lane_f16_accu(accu00, b00, b10, b20, b30, a0);
                accu01 = maq_lane_f16_accu(accu01, b01, b11, b21, b31, a0);
                MlasStoreFloat16x8(c, accu00);
                MlasStoreFloat16x8(c + 8, accu01);
                if constexpr (CountM == 2) {
                    float16x8_t accu10 = MlasZeroFloat16x8();
                    float16x8_t accu11 = MlasZeroFloat16x8();
                    accu10 = maq_lane_f16_accu(accu10, b00, b10, b20, b30, a1);
                    accu11 = maq_lane_f16_accu(accu11, b01, b11, b21, b31, a1);
                    MlasStoreFloat16x8(c + ldc, accu10);
                    MlasStoreFloat16x8(c + ldc + 8, accu11);
                }
                n -= 16, b += 16, c += 16;
            }
            if (n & 8) {
                float16x8_t accu00 = MlasZeroFloat16x8();
                float16x8_t b00 = MlasLoadFloat16x8(b);
                float16x8_t b10 = MlasLoadFloat16x8(b + ldb);
                float16x8_t b20 = MlasLoadFloat16x8(b + 2 * ldb);
                float16x8_t b30 = MlasLoadFloat16x8(b + 3 * ldb);
                accu00 = maq_lane_f16_accu(accu00, b00, b10, b20, b30, a0);
                MlasStoreFloat16x8(c, accu00);
                if constexpr (CountM == 2) {
                    float16x8_t accu10 = MlasZeroFloat16x8();
                    accu10 = maq_lane_f16_accu(accu10, b00, b10, b20, b30, a1);
                    MlasStoreFloat16x8(c + ldc, accu10);
                }
                n -= 8, b += 8, c += 8;
            }
            if (n & 4) {
                float16x4_t accu00 = MlasZeroFloat16x4();
                float16x4_t b0 = MlasLoadFloat16x4(b);
                float16x4_t b1 = MlasLoadFloat16x4(b + ldb);
                float16x4_t b2 = MlasLoadFloat16x4(b + 2 * ldb);
                float16x4_t b3 = MlasLoadFloat16x4(b + 3 * ldb);
                accu00 = ma_lane_f16_accu(accu00, b0, b1, b2, b3, a0);
                MlasStoreFloat16x4(c, accu00);
                if constexpr (CountM == 2) {
                    float16x4_t accu10 = MlasZeroFloat16x4();
                    accu10 = ma_lane_f16_accu(accu10, b0, b1, b2, b3, a1);
                    MlasStoreFloat16x4(c + ldc, accu10);
                }
                n -= 4, b += 4, c += 4;
            }
            if (Nr0) {
                float16x4_t accu00 = MlasZeroFloat16x4();
                float16x4_t b0 = MlasLoadPartialFloat16x4(b, n);
                float16x4_t b1 = MlasLoadPartialFloat16x4(b + ldb, n);
                float16x4_t b2 = MlasLoadPartialFloat16x4(b + 2 * ldb, n);
                float16x4_t b3 = MlasLoadPartialFloat16x4(b + 3 * ldb, n);
                accu00 = ma_lane_f16_accu(accu00, b0, b1, b2, b3, a0);
                MlasStorePartialFloat16x4(c, accu00, n);
                if constexpr (CountM == 2) {
                    float16x4_t accu10 = MlasZeroFloat16x4();
                    accu10 = ma_lane_f16_accu(accu10, b0, b1, b2, b3, a1);
                    MlasStorePartialFloat16x4(c + ldc, accu10, n);
                }
            }
            CountK -= 4, B_data += ldb4, A_data += 4;
        } else if (CountK > 0) {
            float16x4_t a0 = MlasLoadPartialFloat16x4(A_data, CountK), a1;
            if constexpr (CountM == 2) {
                a1 = MlasLoadPartialFloat16x4(A_data + lda, CountK);
            }
            size_t n = CountN;
            const auto* b = B_data;
            auto* c = C_data;
            if (largeN) {
                float16x8_t b00 = MlasLoadFloat16x8(b);
                float16x8_t b01 = MlasLoadFloat16x8(b + 8);
                float16x8_t b02 = MlasLoadFloat16x8(b + 16);
                float16x8_t b03 = MlasLoadFloat16x8(b + 24);
                float16x8_t b10 = MlasZeroFloat16x8();
                float16x8_t b11 = MlasZeroFloat16x8();
                float16x8_t b12 = MlasZeroFloat16x8();
                float16x8_t b13 = MlasZeroFloat16x8();
                float16x8_t b20 = MlasZeroFloat16x8();
                float16x8_t b21 = MlasZeroFloat16x8();
                float16x8_t b22 = MlasZeroFloat16x8();
                float16x8_t b23 = MlasZeroFloat16x8();
                if (Kr1) {
                    b10 = MlasLoadFloat16x8(b + ldb);
                    b11 = MlasLoadFloat16x8(b + ldb + 8);
                    b12 = MlasLoadFloat16x8(b + ldb + 16);
                    b13 = MlasLoadFloat16x8(b + ldb + 24);
                }
                if (Kr2) {
                    b20 = MlasLoadFloat16x8(b + 2 * ldb);
                    b21 = MlasLoadFloat16x8(b + 2 * ldb + 8);
                    b22 = MlasLoadFloat16x8(b + 2 * ldb + 16);
                    b23 = MlasLoadFloat16x8(b + 2 * ldb + 24);
                }
                for (; n >= 64; n -= 32, b += 32, c += 32) {
                    float16x8_t accu00 = MlasZeroFloat16x8();
                    float16x8_t accu01 = MlasZeroFloat16x8();
                    float16x8_t accu02 = MlasZeroFloat16x8();
                    float16x8_t accu03 = MlasZeroFloat16x8();
                    float16x8_t accu10, accu11, accu12, accu13;
                    if constexpr (CountM == 2) {
                        accu10 = MlasZeroFloat16x8();
                        accu11 = MlasZeroFloat16x8();
                        accu12 = MlasZeroFloat16x8();
                        accu13 = MlasZeroFloat16x8();
                    }
                    accu00 = vfmaq_lane_f16(accu00, b00, a0, 0);
                    accu01 = vfmaq_lane_f16(accu01, b01, a0, 0);
                    accu02 = vfmaq_lane_f16(accu02, b02, a0, 0);
                    accu03 = vfmaq_lane_f16(accu03, b03, a0, 0);
                    if constexpr (CountM == 2) {
                        accu10 = vfmaq_lane_f16(accu10, b00, a1, 0);
                        accu11 = vfmaq_lane_f16(accu11, b01, a1, 0);
                        accu12 = vfmaq_lane_f16(accu12, b02, a1, 0);
                        accu13 = vfmaq_lane_f16(accu13, b03, a1, 0);
                    }
                    if (Kr1) {
                        accu00 = vfmaq_lane_f16(accu00, b10, a0, 1);
                        accu01 = vfmaq_lane_f16(accu01, b11, a0, 1);
                        accu02 = vfmaq_lane_f16(accu02, b12, a0, 1);
                        accu03 = vfmaq_lane_f16(accu03, b13, a0, 1);
                        if constexpr (CountM == 2) {
                            accu10 = vfmaq_lane_f16(accu10, b10, a1, 1);
                            accu11 = vfmaq_lane_f16(accu11, b11, a1, 1);
                            accu12 = vfmaq_lane_f16(accu12, b12, a1, 1);
                            accu13 = vfmaq_lane_f16(accu13, b13, a1, 1);
                        }
                    }
                    if (Kr2) {
                        accu00 = vfmaq_lane_f16(accu00, b20, a0, 2);
                        accu01 = vfmaq_lane_f16(accu01, b21, a0, 2);
                        accu02 = vfmaq_lane_f16(accu02, b22, a0, 2);
                        accu03 = vfmaq_lane_f16(accu03, b23, a0, 2);
                        if constexpr (CountM == 2) {
                            accu10 = vfmaq_lane_f16(accu10, b20, a1, 2);
                            accu11 = vfmaq_lane_f16(accu11, b21, a1, 2);
                            accu12 = vfmaq_lane_f16(accu12, b22, a1, 2);
                            accu13 = vfmaq_lane_f16(accu13, b23, a1, 2);
                        }
                    }
                    MlasStoreFloat16x8(c, accu00);
                    MlasStoreFloat16x8(c + 8, accu01);
                    MlasStoreFloat16x8(c + 16, accu02);
                    MlasStoreFloat16x8(c + 24, accu03);
                    if constexpr (CountM == 2) {
                        MlasStoreFloat16x8(c + ldc, accu10);
                        MlasStoreFloat16x8(c + ldc + 8, accu11);
                        MlasStoreFloat16x8(c + ldc + 16, accu12);
                        MlasStoreFloat16x8(c + ldc + 24, accu13);
                    }
                    b00 = MlasLoadFloat16x8(b + 32);
                    b01 = MlasLoadFloat16x8(b + 40);
                    b02 = MlasLoadFloat16x8(b + 48);
                    b03 = MlasLoadFloat16x8(b + 56);
                    if (Kr1) {
                        b10 = MlasLoadFloat16x8(b + ldb + 32);
                        b11 = MlasLoadFloat16x8(b + ldb + 40);
                        b12 = MlasLoadFloat16x8(b + ldb + 48);
                        b13 = MlasLoadFloat16x8(b + ldb + 56);
                    }
                    if (Kr2) {
                        b20 = MlasLoadFloat16x8(b + 2 * ldb + 32);
                        b21 = MlasLoadFloat16x8(b + 2 * ldb + 40);
                        b22 = MlasLoadFloat16x8(b + 2 * ldb + 48);
                        b23 = MlasLoadFloat16x8(b + 2 * ldb + 56);
                    }
                }
                float16x8_t accu00 = MlasZeroFloat16x8();
                float16x8_t accu01 = MlasZeroFloat16x8();
                float16x8_t accu02 = MlasZeroFloat16x8();
                float16x8_t accu03 = MlasZeroFloat16x8();
                float16x8_t accu10, accu11, accu12, accu13;
                if constexpr (CountM == 2) {
                    accu10 = MlasZeroFloat16x8();
                    accu11 = MlasZeroFloat16x8();
                    accu12 = MlasZeroFloat16x8();
                    accu13 = MlasZeroFloat16x8();
                }
                accu00 = vfmaq_lane_f16(accu00, b00, a0, 0);
                accu01 = vfmaq_lane_f16(accu01, b01, a0, 0);
                accu02 = vfmaq_lane_f16(accu02, b02, a0, 0);
                accu03 = vfmaq_lane_f16(accu03, b03, a0, 0);
                if constexpr (CountM == 2) {
                    accu10 = vfmaq_lane_f16(accu10, b00, a1, 0);
                    accu11 = vfmaq_lane_f16(accu11, b01, a1, 0);
                    accu12 = vfmaq_lane_f16(accu12, b02, a1, 0);
                    accu13 = vfmaq_lane_f16(accu13, b03, a1, 0);
                }
                if (Kr1) {
                    accu00 = vfmaq_lane_f16(accu00, b10, a0, 1);
                    accu01 = vfmaq_lane_f16(accu01, b11, a0, 1);
                    accu02 = vfmaq_lane_f16(accu02, b12, a0, 1);
                    accu03 = vfmaq_lane_f16(accu03, b13, a0, 1);
                    if constexpr (CountM == 2) {
                        accu10 = vfmaq_lane_f16(accu10, b10, a1, 1);
                        accu11 = vfmaq_lane_f16(accu11, b11, a1, 1);
                        accu12 = vfmaq_lane_f16(accu12, b12, a1, 1);
                        accu13 = vfmaq_lane_f16(accu13, b13, a1, 1);
                    }
                }
                if (Kr2) {
                    accu00 = vfmaq_lane_f16(accu00, b20, a0, 2);
                    accu01 = vfmaq_lane_f16(accu01, b21, a0, 2);
                    accu02 = vfmaq_lane_f16(accu02, b22, a0, 2);
                    accu03 = vfmaq_lane_f16(accu03, b23, a0, 2);
                    if constexpr (CountM == 2) {
                        accu10 = vfmaq_lane_f16(accu10, b20, a1, 2);
                        accu11 = vfmaq_lane_f16(accu11, b21, a1, 2);
                        accu12 = vfmaq_lane_f16(accu12, b22, a1, 2);
                        accu13 = vfmaq_lane_f16(accu13, b23, a1, 2);
                    }
                }
                MlasStoreFloat16x8(c, accu00);
                MlasStoreFloat16x8(c + 8, accu01);
                MlasStoreFloat16x8(c + 16, accu02);
                MlasStoreFloat16x8(c + 24, accu03);
                if constexpr (CountM == 2) {
                    MlasStoreFloat16x8(c + ldc, accu10);
                    MlasStoreFloat16x8(c + ldc + 8, accu11);
                    MlasStoreFloat16x8(c + ldc + 16, accu12);
                    MlasStoreFloat16x8(c + ldc + 24, accu13);
                }
                n -= 32, b += 32, c += 32;
            }
            if (n & 16) {
                float16x8_t accu00 = MlasZeroFloat16x8();
                float16x8_t accu01 = MlasZeroFloat16x8();
                float16x8_t accu10, accu11;
                if constexpr (CountM == 2) {
                    accu10 = MlasZeroFloat16x8();
                    accu11 = MlasZeroFloat16x8();
                }
                float16x8_t b00 = MlasLoadFloat16x8(b);
                float16x8_t b01 = MlasLoadFloat16x8(b + 8);
                accu00 = vfmaq_lane_f16(accu00, b00, a0, 0);
                accu01 = vfmaq_lane_f16(accu01, b01, a0, 0);
                if constexpr (CountM == 2) {
                    accu10 = vfmaq_lane_f16(accu10, b00, a1, 0);
                    accu11 = vfmaq_lane_f16(accu11, b01, a1, 0);
                }
                if (Kr1) {
                    float16x8_t b10 = MlasLoadFloat16x8(b + ldb);
                    float16x8_t b11 = MlasLoadFloat16x8(b + ldb + 8);
                    accu00 = vfmaq_lane_f16(accu00, b10, a0, 1);
                    accu01 = vfmaq_lane_f16(accu01, b11, a0, 1);
                    if constexpr (CountM == 2) {
                        accu10 = vfmaq_lane_f16(accu10, b10, a1, 1);
                        accu11 = vfmaq_lane_f16(accu11, b11, a1, 1);
                    }
                }
                if (Kr2) {
                    float16x8_t b20 = MlasLoadFloat16x8(b + 2 * ldb);
                    float16x8_t b21 = MlasLoadFloat16x8(b + 2 * ldb + 8);
                    accu00 = vfmaq_lane_f16(accu00, b20, a0, 2);
                    accu01 = vfmaq_lane_f16(accu01, b21, a0, 2);
                    if constexpr (CountM == 2) {
                        accu10 = vfmaq_lane_f16(accu10, b20, a1, 2);
                        accu11 = vfmaq_lane_f16(accu11, b21, a1, 2);
                    }
                }
                MlasStoreFloat16x8(c, accu00);
                MlasStoreFloat16x8(c + 8, accu01);
                if constexpr (CountM == 2) {
                    MlasStoreFloat16x8(c + ldc, accu10);
                    MlasStoreFloat16x8(c + ldc + 8, accu11);
                }
                n -= 16, b += 16, c += 16;
            }
            if (n & 8) {
                float16x8_t accu00 = MlasZeroFloat16x8(), accu10;
                if constexpr (CountM == 2) {
                    accu10 = MlasZeroFloat16x8();
                }
                float16x8_t b0 = MlasLoadFloat16x8(b);
                accu00 = vfmaq_lane_f16(accu00, b0, a0, 0);
                if constexpr (CountM == 2) {
                    accu10 = vfmaq_lane_f16(accu10, b0, a1, 0);
                }
                if (Kr1) {
                    float16x8_t b1 = MlasLoadFloat16x8(b + ldb);
                    accu00 = vfmaq_lane_f16(accu00, b1, a0, 1);
                    if constexpr (CountM == 2) {
                        accu10 = vfmaq_lane_f16(accu10, b1, a1, 1);
                    }
                }
                if (Kr2) {
                    float16x8_t b2 = MlasLoadFloat16x8(b + 2 * ldb);
                    accu00 = vfmaq_lane_f16(accu00, b2, a0, 2);
                    if constexpr (CountM == 2) {
                        accu10 = vfmaq_lane_f16(accu10, b2, a1, 2);
                    }
                }
                MlasStoreFloat16x8(c, accu00);
                if constexpr (CountM == 2) {
                    MlasStoreFloat16x8(c + ldc, accu10);
                }
                n -= 8, b += 8, c += 8;
            }
            if (n & 4) {
                float16x4_t accu00 = MlasZeroFloat16x4(), accu10;
                if constexpr (CountM == 2) {
                    accu10 = MlasZeroFloat16x4();
                }
                float16x4_t b0 = MlasLoadFloat16x4(b);
                accu00 = vfma_lane_f16(accu00, b0, a0, 0);
                if constexpr (CountM == 2) {
                    accu10 = vfma_lane_f16(accu10, b0, a1, 0);
                }
                if (Kr1) {
                    float16x4_t b1 = MlasLoadFloat16x4(b + ldb);
                    accu00 = vfma_lane_f16(accu00, b1, a0, 1);
                    if constexpr (CountM == 2) {
                        accu10 = vfma_lane_f16(accu10, b1, a1, 1);
                    }
                }
                if (Kr2) {
                    float16x4_t b2 = MlasLoadFloat16x4(b + 2 * ldb);
                    accu00 = vfma_lane_f16(accu00, b2, a0, 2);
                    if constexpr (CountM == 2) {
                        accu10 = vfma_lane_f16(accu10, b2, a1, 2);
                    }
                }
                MlasStoreFloat16x4(c, accu00);
                if constexpr (CountM == 2) {
                    MlasStoreFloat16x4(c + ldc, accu10);
                }
                n -= 4, b += 4, c += 4;
            }
            if (Nr0) {
                float16x4_t accu00 = MlasZeroFloat16x4(), accu10;
                if constexpr (CountM == 2) {
                    accu10 = MlasZeroFloat16x4();
                }
                float16x4_t b0 = MlasLoadPartialFloat16x4(b, n);
                accu00 = vfma_lane_f16(accu00, b0, a0, 0);
                if constexpr (CountM == 2) {
                    accu10 = vfma_lane_f16(accu10, b0, a1, 0);
                }
                if (Kr1) {
                    float16x4_t b1 = MlasLoadPartialFloat16x4(b + ldb, n);
                    accu00 = vfma_lane_f16(accu00, b1, a0, 1);
                    if constexpr (CountM == 2) {
                        accu10 = vfma_lane_f16(accu10, b1, a1, 1);
                    }
                }
                if (Kr2) {
                    float16x4_t b2 = MlasLoadPartialFloat16x4(b + 2 * ldb, n);
                    accu00 = vfma_lane_f16(accu00, b2, a0, 2);
                    if constexpr (CountM == 2) {
                        accu10 = vfma_lane_f16(accu10, b2, a1, 2);
                    }
                }
                MlasStorePartialFloat16x4(c, accu00, n);
                if constexpr (CountM == 2) {
                    MlasStorePartialFloat16x4(c + ldc, accu10, n);
                }
            }

            CountK -= CountK, B_data += ldb * CountK, A_data += CountK;
        }
    }

    for (; CountK >= 4; CountK -= 4, B_data += ldb4, A_data += 4) {
        float16x4_t a0 = MlasLoadFloat16x4(A_data), a1;
        if constexpr (CountM == 2) {
            a1 = MlasLoadFloat16x4(A_data + lda);
        }
        size_t n = CountN;
        const auto* b = B_data;
        auto* c = C_data;
        if (largeN) {
            float16x8_t b00 = MlasLoadFloat16x8(b);
            float16x8_t b10 = MlasLoadFloat16x8(b + ldb);
            float16x8_t b20 = MlasLoadFloat16x8(b + 2 * ldb);
            float16x8_t b30 = MlasLoadFloat16x8(b + 3 * ldb);
            float16x8_t b01 = MlasLoadFloat16x8(b + 8);
            float16x8_t b11 = MlasLoadFloat16x8(b + ldb + 8);
            float16x8_t b21 = MlasLoadFloat16x8(b + 2 * ldb + 8);
            float16x8_t b31 = MlasLoadFloat16x8(b + 3 * ldb + 8);
            float16x8_t b02 = MlasLoadFloat16x8(b + 16);
            float16x8_t b12 = MlasLoadFloat16x8(b + ldb + 16);
            float16x8_t b22 = MlasLoadFloat16x8(b + 2 * ldb + 16);
            float16x8_t b32 = MlasLoadFloat16x8(b + 3 * ldb + 16);
            float16x8_t b03 = MlasLoadFloat16x8(b + 24);
            float16x8_t b13 = MlasLoadFloat16x8(b + ldb + 24);
            float16x8_t b23 = MlasLoadFloat16x8(b + 2 * ldb + 24);
            float16x8_t b33 = MlasLoadFloat16x8(b + 3 * ldb + 24);
            for (; n >= 64; n -= 32, b += 32, c += 32) {
                float16x8_t accu00 = MlasLoadFloat16x8(c);
                float16x8_t accu01 = MlasLoadFloat16x8(c + 8);
                float16x8_t accu02 = MlasLoadFloat16x8(c + 16);
                float16x8_t accu03 = MlasLoadFloat16x8(c + 24);
                accu00 = maq_lane_f16_accu(accu00, b00, b10, b20, b30, a0);
                accu01 = maq_lane_f16_accu(accu01, b01, b11, b21, b31, a0);
                accu02 = maq_lane_f16_accu(accu02, b02, b12, b22, b32, a0);
                accu03 = maq_lane_f16_accu(accu03, b03, b13, b23, b33, a0);
                MlasStoreFloat16x8(c, accu00);
                MlasStoreFloat16x8(c + 8, accu01);
                MlasStoreFloat16x8(c + 16, accu02);
                MlasStoreFloat16x8(c + 24, accu03);
                if constexpr (CountM == 2) {
                    float16x8_t accu10 = MlasLoadFloat16x8(c + ldc);
                    float16x8_t accu11 = MlasLoadFloat16x8(c + ldc + 8);
                    float16x8_t accu12 = MlasLoadFloat16x8(c + ldc + 16);
                    float16x8_t accu13 = MlasLoadFloat16x8(c + ldc + 24);
                    accu10 = maq_lane_f16_accu(accu10, b00, b10, b20, b30, a1);
                    accu11 = maq_lane_f16_accu(accu11, b01, b11, b21, b31, a1);
                    accu12 = maq_lane_f16_accu(accu12, b02, b12, b22, b32, a1);
                    accu13 = maq_lane_f16_accu(accu13, b03, b13, b23, b33, a1);
                    MlasStoreFloat16x8(c + ldc, accu10);
                    MlasStoreFloat16x8(c + ldc + 8, accu11);
                    MlasStoreFloat16x8(c + ldc + 16, accu12);
                    MlasStoreFloat16x8(c + ldc + 24, accu13);
                }
                b00 = MlasLoadFloat16x8(b + 32);
                b10 = MlasLoadFloat16x8(b + ldb + 32);
                b20 = MlasLoadFloat16x8(b + 2 * ldb + 32);
                b30 = MlasLoadFloat16x8(b + 3 * ldb + 32);
                b01 = MlasLoadFloat16x8(b + 40);
                b11 = MlasLoadFloat16x8(b + ldb + 40);
                b21 = MlasLoadFloat16x8(b + 2 * ldb + 40);
                b31 = MlasLoadFloat16x8(b + 3 * ldb + 40);
                b02 = MlasLoadFloat16x8(b + 48);
                b12 = MlasLoadFloat16x8(b + ldb + 48);
                b22 = MlasLoadFloat16x8(b + 2 * ldb + 48);
                b32 = MlasLoadFloat16x8(b + 3 * ldb + 48);
                b03 = MlasLoadFloat16x8(b + 56);
                b13 = MlasLoadFloat16x8(b + ldb + 56);
                b23 = MlasLoadFloat16x8(b + 2 * ldb + 56);
                b33 = MlasLoadFloat16x8(b + 3 * ldb + 56);
            }
            float16x8_t accu00 = MlasLoadFloat16x8(c);
            float16x8_t accu01 = MlasLoadFloat16x8(c + 8);
            float16x8_t accu02 = MlasLoadFloat16x8(c + 16);
            float16x8_t accu03 = MlasLoadFloat16x8(c + 24);
            accu00 = maq_lane_f16_accu(accu00, b00, b10, b20, b30, a0);
            accu01 = maq_lane_f16_accu(accu01, b01, b11, b21, b31, a0);
            accu02 = maq_lane_f16_accu(accu02, b02, b12, b22, b32, a0);
            accu03 = maq_lane_f16_accu(accu03, b03, b13, b23, b33, a0);
            MlasStoreFloat16x8(c, accu00);
            MlasStoreFloat16x8(c + 8, accu01);
            MlasStoreFloat16x8(c + 16, accu02);
            MlasStoreFloat16x8(c + 24, accu03);
            if constexpr (CountM == 2) {
                float16x8_t accu10 = MlasLoadFloat16x8(c + ldc);
                float16x8_t accu11 = MlasLoadFloat16x8(c + ldc + 8);
                float16x8_t accu12 = MlasLoadFloat16x8(c + ldc + 16);
                float16x8_t accu13 = MlasLoadFloat16x8(c + ldc + 24);
                accu10 = maq_lane_f16_accu(accu10, b00, b10, b20, b30, a1);
                accu11 = maq_lane_f16_accu(accu11, b01, b11, b21, b31, a1);
                accu12 = maq_lane_f16_accu(accu12, b02, b12, b22, b32, a1);
                accu13 = maq_lane_f16_accu(accu13, b03, b13, b23, b33, a1);
                MlasStoreFloat16x8(c + ldc, accu10);
                MlasStoreFloat16x8(c + ldc + 8, accu11);
                MlasStoreFloat16x8(c + ldc + 16, accu12);
                MlasStoreFloat16x8(c + ldc + 24, accu13);
            }
            n -= 32, b += 32, c += 32;
        }
        if (n & 16) {
            float16x8_t accu00 = MlasLoadFloat16x8(c);
            float16x8_t accu01 = MlasLoadFloat16x8(c + 8);
            float16x8_t b00 = MlasLoadFloat16x8(b);
            float16x8_t b10 = MlasLoadFloat16x8(b + ldb);
            float16x8_t b20 = MlasLoadFloat16x8(b + 2 * ldb);
            float16x8_t b30 = MlasLoadFloat16x8(b + 3 * ldb);
            float16x8_t b01 = MlasLoadFloat16x8(b + 8);
            float16x8_t b11 = MlasLoadFloat16x8(b + ldb + 8);
            float16x8_t b21 = MlasLoadFloat16x8(b + 2 * ldb + 8);
            float16x8_t b31 = MlasLoadFloat16x8(b + 3 * ldb + 8);
            accu00 = maq_lane_f16_accu(accu00, b00, b10, b20, b30, a0);
            accu01 = maq_lane_f16_accu(accu01, b01, b11, b21, b31, a0);
            MlasStoreFloat16x8(c, accu00);
            MlasStoreFloat16x8(c + 8, accu01);
            if constexpr (CountM == 2) {
                float16x8_t accu10 = MlasLoadFloat16x8(c + ldc);
                float16x8_t accu11 = MlasLoadFloat16x8(c + ldc + 8);
                accu10 = maq_lane_f16_accu(accu10, b00, b10, b20, b30, a1);
                accu11 = maq_lane_f16_accu(accu11, b01, b11, b21, b31, a1);
                MlasStoreFloat16x8(c + ldc, accu10);
                MlasStoreFloat16x8(c + ldc + 8, accu11);
            }
            n -= 16, b += 16, c += 16;
        }
        if (n & 8) {
            float16x8_t accu00 = MlasLoadFloat16x8(c);
            float16x8_t b00 = MlasLoadFloat16x8(b);
            float16x8_t b10 = MlasLoadFloat16x8(b + ldb);
            float16x8_t b20 = MlasLoadFloat16x8(b + 2 * ldb);
            float16x8_t b30 = MlasLoadFloat16x8(b + 3 * ldb);
            accu00 = maq_lane_f16_accu(accu00, b00, b10, b20, b30, a0);
            MlasStoreFloat16x8(c, accu00);
            if constexpr (CountM == 2) {
                float16x8_t accu10 = MlasLoadFloat16x8(c + ldc);
                accu10 = maq_lane_f16_accu(accu10, b00, b10, b20, b30, a1);
                MlasStoreFloat16x8(c + ldc, accu10);
            }
            n -= 8, b += 8, c += 8;
        }
        if (n & 4) {
            float16x4_t accu00 = MlasLoadFloat16x4(c);
            float16x4_t b0 = MlasLoadFloat16x4(b);
            float16x4_t b1 = MlasLoadFloat16x4(b + ldb);
            float16x4_t b2 = MlasLoadFloat16x4(b + 2 * ldb);
            float16x4_t b3 = MlasLoadFloat16x4(b + 3 * ldb);
            accu00 = ma_lane_f16_accu(accu00, b0, b1, b2, b3, a0);
            MlasStoreFloat16x4(c, accu00);
            if constexpr (CountM == 2) {
                float16x4_t accu10 = MlasLoadFloat16x4(c + ldc);
                accu10 = ma_lane_f16_accu(accu10, b0, b1, b2, b3, a1);
                MlasStoreFloat16x4(c + ldc, accu10);
            }
            n -= 4, b += 4, c += 4;
        }
        if (Nr0) {
            float16x4_t accu00 = MlasLoadPartialFloat16x4(c, n);
            float16x4_t b0 = MlasLoadPartialFloat16x4(b, n);
            float16x4_t b1 = MlasLoadPartialFloat16x4(b + ldb, n);
            float16x4_t b2 = MlasLoadPartialFloat16x4(b + 2 * ldb, n);
            float16x4_t b3 = MlasLoadPartialFloat16x4(b + 3 * ldb, n);
            accu00 = ma_lane_f16_accu(accu00, b0, b1, b2, b3, a0);
            MlasStorePartialFloat16x4(c, accu00, n);
            if constexpr (CountM == 2) {
                float16x4_t accu10 = MlasLoadPartialFloat16x4(c + ldc, n);
                accu10 = ma_lane_f16_accu(accu10, b0, b1, b2, b3, a1);
                MlasStorePartialFloat16x4(c + ldc, accu10, n);
            }
        }
    }

    if (CountK > 0) {
        float16x4_t a0 = MlasLoadPartialFloat16x4(A_data, CountK), a1;
        if constexpr (CountM == 2) {
            a1 = MlasLoadPartialFloat16x4(A_data + lda, CountK);
        }
        size_t n = CountN;
        const auto* b = B_data;
        auto* c = C_data;
        if (largeN) {
            float16x8_t b00 = MlasLoadFloat16x8(b);
            float16x8_t b01 = MlasLoadFloat16x8(b + 8);
            float16x8_t b02 = MlasLoadFloat16x8(b + 16);
            float16x8_t b03 = MlasLoadFloat16x8(b + 24);
            float16x8_t b10 = MlasZeroFloat16x8();
            float16x8_t b11 = MlasZeroFloat16x8();
            float16x8_t b12 = MlasZeroFloat16x8();
            float16x8_t b13 = MlasZeroFloat16x8();
            float16x8_t b20 = MlasZeroFloat16x8();
            float16x8_t b21 = MlasZeroFloat16x8();
            float16x8_t b22 = MlasZeroFloat16x8();
            float16x8_t b23 = MlasZeroFloat16x8();
            if (Kr1) {
                b10 = MlasLoadFloat16x8(b + ldb);
                b11 = MlasLoadFloat16x8(b + ldb + 8);
                b12 = MlasLoadFloat16x8(b + ldb + 16);
                b13 = MlasLoadFloat16x8(b + ldb + 24);
            }
            if (Kr2) {
                b20 = MlasLoadFloat16x8(b + 2 * ldb);
                b21 = MlasLoadFloat16x8(b + 2 * ldb + 8);
                b22 = MlasLoadFloat16x8(b + 2 * ldb + 16);
                b23 = MlasLoadFloat16x8(b + 2 * ldb + 24);
            }
            for (; n >= 64; n -= 32, b += 32, c += 32) {
                float16x8_t accu00 = MlasLoadFloat16x8(c);
                float16x8_t accu01 = MlasLoadFloat16x8(c + 8);
                float16x8_t accu02 = MlasLoadFloat16x8(c + 16);
                float16x8_t accu03 = MlasLoadFloat16x8(c + 24);
                float16x8_t accu10, accu11, accu12, accu13;
                if constexpr (CountM == 2) {
                    accu10 = MlasLoadFloat16x8(c + ldc);
                    accu11 = MlasLoadFloat16x8(c + ldc + 8);
                    accu12 = MlasLoadFloat16x8(c + ldc + 16);
                    accu13 = MlasLoadFloat16x8(c + ldc + 24);
                }
                accu00 = vfmaq_lane_f16(accu00, b00, a0, 0);
                accu01 = vfmaq_lane_f16(accu01, b01, a0, 0);
                accu02 = vfmaq_lane_f16(accu02, b02, a0, 0);
                accu03 = vfmaq_lane_f16(accu03, b03, a0, 0);
                if constexpr (CountM == 2) {
                    accu10 = vfmaq_lane_f16(accu10, b00, a1, 0);
                    accu11 = vfmaq_lane_f16(accu11, b01, a1, 0);
                    accu12 = vfmaq_lane_f16(accu12, b02, a1, 0);
                    accu13 = vfmaq_lane_f16(accu13, b03, a1, 0);
                }
                if (Kr1) {
                    accu00 = vfmaq_lane_f16(accu00, b10, a0, 1);
                    accu01 = vfmaq_lane_f16(accu01, b11, a0, 1);
                    accu02 = vfmaq_lane_f16(accu02, b12, a0, 1);
                    accu03 = vfmaq_lane_f16(accu03, b13, a0, 1);
                    if constexpr (CountM == 2) {
                        accu10 = vfmaq_lane_f16(accu10, b10, a1, 1);
                        accu11 = vfmaq_lane_f16(accu11, b11, a1, 1);
                        accu12 = vfmaq_lane_f16(accu12, b12, a1, 1);
                        accu13 = vfmaq_lane_f16(accu13, b13, a1, 1);
                    }
                }
                if (Kr2) {
                    accu00 = vfmaq_lane_f16(accu00, b20, a0, 2);
                    accu01 = vfmaq_lane_f16(accu01, b21, a0, 2);
                    accu02 = vfmaq_lane_f16(accu02, b22, a0, 2);
                    accu03 = vfmaq_lane_f16(accu03, b23, a0, 2);
                    if constexpr (CountM == 2) {
                        accu10 = vfmaq_lane_f16(accu10, b20, a1, 2);
                        accu11 = vfmaq_lane_f16(accu11, b21, a1, 2);
                        accu12 = vfmaq_lane_f16(accu12, b22, a1, 2);
                        accu13 = vfmaq_lane_f16(accu13, b23, a1, 2);
                    }
                }
                MlasStoreFloat16x8(c, accu00);
                MlasStoreFloat16x8(c + 8, accu01);
                MlasStoreFloat16x8(c + 16, accu02);
                MlasStoreFloat16x8(c + 24, accu03);
                if constexpr (CountM == 2) {
                    MlasStoreFloat16x8(c + ldc, accu10);
                    MlasStoreFloat16x8(c + ldc + 8, accu11);
                    MlasStoreFloat16x8(c + ldc + 16, accu12);
                    MlasStoreFloat16x8(c + ldc + 24, accu13);
                }
                b00 = MlasLoadFloat16x8(b + 32);
                b01 = MlasLoadFloat16x8(b + 40);
                b02 = MlasLoadFloat16x8(b + 48);
                b03 = MlasLoadFloat16x8(b + 56);
                if (Kr1) {
                    b10 = MlasLoadFloat16x8(b + ldb + 32);
                    b11 = MlasLoadFloat16x8(b + ldb + 40);
                    b12 = MlasLoadFloat16x8(b + ldb + 48);
                    b13 = MlasLoadFloat16x8(b + ldb + 56);
                }
                if (Kr2) {
                    b20 = MlasLoadFloat16x8(b + 2 * ldb + 32);
                    b21 = MlasLoadFloat16x8(b + 2 * ldb + 40);
                    b22 = MlasLoadFloat16x8(b + 2 * ldb + 48);
                    b23 = MlasLoadFloat16x8(b + 2 * ldb + 56);
                }
            }
            float16x8_t accu00 = MlasLoadFloat16x8(c);
            float16x8_t accu01 = MlasLoadFloat16x8(c + 8);
            float16x8_t accu02 = MlasLoadFloat16x8(c + 16);
            float16x8_t accu03 = MlasLoadFloat16x8(c + 24);
            float16x8_t accu10, accu11, accu12, accu13;
            if constexpr (CountM == 2) {
                accu10 = MlasLoadFloat16x8(c + ldc);
                accu11 = MlasLoadFloat16x8(c + ldc + 8);
                accu12 = MlasLoadFloat16x8(c + ldc + 16);
                accu13 = MlasLoadFloat16x8(c + ldc + 24);
            }
            accu00 = vfmaq_lane_f16(accu00, b00, a0, 0);
            accu01 = vfmaq_lane_f16(accu01, b01, a0, 0);
            accu02 = vfmaq_lane_f16(accu02, b02, a0, 0);
            accu03 = vfmaq_lane_f16(accu03, b03, a0, 0);
            if constexpr (CountM == 2) {
                accu10 = vfmaq_lane_f16(accu10, b00, a1, 0);
                accu11 = vfmaq_lane_f16(accu11, b01, a1, 0);
                accu12 = vfmaq_lane_f16(accu12, b02, a1, 0);
                accu13 = vfmaq_lane_f16(accu13, b03, a1, 0);
            }
            if (Kr1) {
                accu00 = vfmaq_lane_f16(accu00, b10, a0, 1);
                accu01 = vfmaq_lane_f16(accu01, b11, a0, 1);
                accu02 = vfmaq_lane_f16(accu02, b12, a0, 1);
                accu03 = vfmaq_lane_f16(accu03, b13, a0, 1);
                if constexpr (CountM == 2) {
                    accu10 = vfmaq_lane_f16(accu10, b10, a1, 1);
                    accu11 = vfmaq_lane_f16(accu11, b11, a1, 1);
                    accu12 = vfmaq_lane_f16(accu12, b12, a1, 1);
                    accu13 = vfmaq_lane_f16(accu13, b13, a1, 1);
                }
            }
            if (Kr2) {
                accu00 = vfmaq_lane_f16(accu00, b20, a0, 2);
                accu01 = vfmaq_lane_f16(accu01, b21, a0, 2);
                accu02 = vfmaq_lane_f16(accu02, b22, a0, 2);
                accu03 = vfmaq_lane_f16(accu03, b23, a0, 2);
                if constexpr (CountM == 2) {
                    accu10 = vfmaq_lane_f16(accu10, b20, a1, 2);
                    accu11 = vfmaq_lane_f16(accu11, b21, a1, 2);
                    accu12 = vfmaq_lane_f16(accu12, b22, a1, 2);
                    accu13 = vfmaq_lane_f16(accu13, b23, a1, 2);
                }
            }
            MlasStoreFloat16x8(c, accu00);
            MlasStoreFloat16x8(c + 8, accu01);
            MlasStoreFloat16x8(c + 16, accu02);
            MlasStoreFloat16x8(c + 24, accu03);
            if constexpr (CountM == 2) {
                MlasStoreFloat16x8(c + ldc, accu10);
                MlasStoreFloat16x8(c + ldc + 8, accu11);
                MlasStoreFloat16x8(c + ldc + 16, accu12);
                MlasStoreFloat16x8(c + ldc + 24, accu13);
            }
            n -= 32, b += 32, c += 32;
        }
        if (n & 16) {
            float16x8_t accu00 = MlasLoadFloat16x8(c);
            float16x8_t accu01 = MlasLoadFloat16x8(c + 8);
            float16x8_t accu10, accu11;
            if constexpr (CountM == 2) {
                accu10 = MlasLoadFloat16x8(c + ldc);
                accu11 = MlasLoadFloat16x8(c + ldc + 8);
            }
            float16x8_t b00 = MlasLoadFloat16x8(b);
            float16x8_t b01 = MlasLoadFloat16x8(b + 8);
            accu00 = vfmaq_lane_f16(accu00, b00, a0, 0);
            accu01 = vfmaq_lane_f16(accu01, b01, a0, 0);
            if constexpr (CountM == 2) {
                accu10 = vfmaq_lane_f16(accu10, b00, a1, 0);
                accu11 = vfmaq_lane_f16(accu11, b01, a1, 0);
            }
            if (Kr1) {
                float16x8_t b10 = MlasLoadFloat16x8(b + ldb);
                float16x8_t b11 = MlasLoadFloat16x8(b + ldb + 8);
                accu00 = vfmaq_lane_f16(accu00, b10, a0, 1);
                accu01 = vfmaq_lane_f16(accu01, b11, a0, 1);
                if constexpr (CountM == 2) {
                    accu10 = vfmaq_lane_f16(accu10, b10, a1, 1);
                    accu11 = vfmaq_lane_f16(accu11, b11, a1, 1);
                }
            }
            if (Kr2) {
                float16x8_t b20 = MlasLoadFloat16x8(b + 2 * ldb);
                float16x8_t b21 = MlasLoadFloat16x8(b + 2 * ldb + 8);
                accu00 = vfmaq_lane_f16(accu00, b20, a0, 2);
                accu01 = vfmaq_lane_f16(accu01, b21, a0, 2);
                if constexpr (CountM == 2) {
                    accu10 = vfmaq_lane_f16(accu10, b20, a1, 2);
                    accu11 = vfmaq_lane_f16(accu11, b21, a1, 2);
                }
            }
            MlasStoreFloat16x8(c, accu00);
            MlasStoreFloat16x8(c + 8, accu01);
            if constexpr (CountM == 2) {
                MlasStoreFloat16x8(c + ldc, accu10);
                MlasStoreFloat16x8(c + ldc + 8, accu11);
            }
            n -= 16, b += 16, c += 16;
        }
        if (n & 8) {
            float16x8_t accu00 = MlasLoadFloat16x8(c);
            float16x8_t accu10;
            if constexpr (CountM == 2) {
                accu10 = MlasLoadFloat16x8(c + ldc);
            }
            float16x8_t b0 = MlasLoadFloat16x8(b);
            accu00 = vfmaq_lane_f16(accu00, b0, a0, 0);
            if constexpr (CountM == 2) {
                accu10 = vfmaq_lane_f16(accu10, b0, a1, 0);
            }
            if (Kr1) {
                float16x8_t b1 = MlasLoadFloat16x8(b + ldb);
                accu00 = vfmaq_lane_f16(accu00, b1, a0, 1);
                if constexpr (CountM == 2) {
                    accu10 = vfmaq_lane_f16(accu10, b1, a1, 1);
                }
            }
            if (Kr2) {
                float16x8_t b2 = MlasLoadFloat16x8(b + 2 * ldb);
                accu00 = vfmaq_lane_f16(accu00, b2, a0, 2);
                if constexpr (CountM == 2) {
                    accu10 = vfmaq_lane_f16(accu10, b2, a1, 2);
                }
            }
            MlasStoreFloat16x8(c, accu00);
            if constexpr (CountM == 2) {
                MlasStoreFloat16x8(c + ldc, accu10);
            }
            n -= 8, b += 8, c += 8;
        }
        if (n & 4) {
            float16x4_t accu00 = MlasLoadFloat16x4(c);
            float16x4_t accu10;
            if constexpr (CountM == 2) {
                accu10 = MlasLoadFloat16x4(c + ldc);
            }
            float16x4_t b0 = MlasLoadFloat16x4(b);
            accu00 = vfma_lane_f16(accu00, b0, a0, 0);
            if constexpr (CountM == 2) {
                accu10 = vfma_lane_f16(accu10, b0, a1, 0);
            }
            if (Kr1) {
                float16x4_t b1 = MlasLoadFloat16x4(b + ldb);
                accu00 = vfma_lane_f16(accu00, b1, a0, 1);
                if constexpr (CountM == 2) {
                    accu10 = vfma_lane_f16(accu10, b1, a1, 1);
                }
            }
            if (Kr2) {
                float16x4_t b2 = MlasLoadFloat16x4(b + 2 * ldb);
                accu00 = vfma_lane_f16(accu00, b2, a0, 2);
                if constexpr (CountM == 2) {
                    accu10 = vfma_lane_f16(accu10, b2, a1, 2);
                }
            }
            MlasStoreFloat16x4(c, accu00);
            if constexpr (CountM == 2) {
                MlasStoreFloat16x4(c + ldc, accu10);
            }
            n -= 4, b += 4, c += 4;
        }
        if (Nr0) {
            float16x4_t accu00 = MlasLoadPartialFloat16x4(c, n);
            float16x4_t accu10;
            if constexpr (CountM == 2) {
                accu10 = MlasLoadPartialFloat16x4(c + ldc, n);
            }
            float16x4_t b0 = MlasLoadPartialFloat16x4(b, n);
            accu00 = vfma_lane_f16(accu00, b0, a0, 0);
            if constexpr (CountM == 2) {
                accu10 = vfma_lane_f16(accu10, b0, a1, 0);
            }
            if (Kr1) {
                float16x4_t b1 = MlasLoadPartialFloat16x4(b + ldb, n);
                accu00 = vfma_lane_f16(accu00, b1, a0, 1);
                if constexpr (CountM == 2) {
                    accu10 = vfma_lane_f16(accu10, b1, a1, 1);
                }
            }
            if (Kr2) {
                float16x4_t b2 = MlasLoadPartialFloat16x4(b + 2 * ldb, n);
                accu00 = vfma_lane_f16(accu00, b2, a0, 2);
                if constexpr (CountM == 2) {
                    accu10 = vfma_lane_f16(accu10, b2, a1, 2);
                }
            }
            MlasStorePartialFloat16x4(c, accu00, n);
            if constexpr (CountM == 2) {
                MlasStorePartialFloat16x4(c + ldc, accu10, n);
            }
        }

        CountK -= CountK, B_data += ldb * CountK, A_data += CountK;
    }
 }

void HGemm_B_Kernel(
    const MLAS_FP16* A,
    const MLAS_FP16* B,
    MLAS_FP16* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc,
    _mlas_fp16_ alpha,
    _mlas_fp16_ beta
) {
    if (CountM > 2) {
        MLAS_THROW_EX(std::runtime_error, "HGemm_TransposedB_Kernel only support <= 2 rows");
    }
    const auto* A_data = reinterpret_cast<const _mlas_fp16_*>(A);
    const auto* B_data = reinterpret_cast<const _mlas_fp16_*>(B);
    auto* C_data = reinterpret_cast<_mlas_fp16_*>(C);
    const auto f16_0 = MLAS_FP16(0.0f);
    const auto f16_1 = MLAS_FP16(1.0f);
    if (CountM == 1) {
        if (alpha == f16_1.val && beta == f16_0.val) {
            HGemm_B_Kernel_Simple<1, true>(A_data, B_data, C_data, CountN, CountK, lda, ldb, ldc);
        } else if (alpha == f16_1.val && beta == f16_1.val) {
            HGemm_B_Kernel_Simple<1, false>(A_data, B_data, C_data, CountN, CountK, lda, ldb, ldc);
        } else {
            HGemm_B_Kernel_Complicated<1>(A_data, B_data, C_data, CountN, CountK, lda, ldb, ldc, alpha, beta);
        }
    } else {
        if (alpha == f16_1.val && beta == f16_0.val) {
            HGemm_B_Kernel_Simple<2, true>(A_data, B_data, C_data, CountN, CountK, lda, ldb, ldc);
        } else if (alpha == f16_1.val && beta == f16_1.val) {
            HGemm_B_Kernel_Simple<2, false>(A_data, B_data, C_data, CountN, CountK, lda, ldb, ldc);
        } else {
            HGemm_B_Kernel_Complicated<2>(A_data, B_data, C_data, CountN, CountK, lda, ldb, ldc, alpha, beta);
        }
    }
}

// beta_behavior: 0 -> beta == 0, 1 -> beta == 1, 2 -> beta != 0 && beta != 1
template <int beta_behavior, int CountM>
void HGemm_PackedB_Kernel_Impl(
    const _mlas_fp16_* A,
    const _mlas_fp16_* PackedB,
    _mlas_fp16_* C,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldc,
    _mlas_fp16_ alpha,
    _mlas_fp16_ beta
) {
    const float16x8_t alpha_v8 = MlasBroadcastFloat16x8(alpha);
    const float16x8_t beta_v8 = MlasBroadcastFloat16x8(beta);
    const float16x4_t alpha_v4 = MlasBroadcastFloat16x4(alpha);
    const float16x4_t beta_v4 = MlasBroadcastFloat16x4(beta);
    const bool Kr0 = CountK & 3;
    const bool Kr1 = (CountK & 3) > 1;
    const bool Kr2 = (CountK & 3) > 2;
    const bool largeK = CountK >= 4;
    for (; CountN >= 32; CountN -= 32, C += 32) {
        const auto* a = A;
        size_t k = CountK;
        float16x8_t accu00 = MlasZeroFloat16x8();
        float16x8_t accu01 = MlasZeroFloat16x8();
        float16x8_t accu02 = MlasZeroFloat16x8();
        float16x8_t accu03 = MlasZeroFloat16x8();
        float16x8_t accu10, accu11, accu12, accu13;
        if constexpr (CountM == 2) {
            accu10 = MlasZeroFloat16x8();
            accu11 = MlasZeroFloat16x8();
            accu12 = MlasZeroFloat16x8();
            accu13 = MlasZeroFloat16x8();
        }
        if (largeK) {
            float16x4_t a0 = MlasLoadFloat16x4(a), a1;
            if constexpr (CountM == 2) {
                a1 = MlasLoadFloat16x4(a + lda);
            }
            float16x8_t b00 = MlasLoadFloat16x8(PackedB);
            float16x8_t b10 = MlasLoadFloat16x8(PackedB + 32);
            float16x8_t b20 = MlasLoadFloat16x8(PackedB + 64);
            float16x8_t b30 = MlasLoadFloat16x8(PackedB + 96);
            float16x8_t b01 = MlasLoadFloat16x8(PackedB + 8);
            float16x8_t b11 = MlasLoadFloat16x8(PackedB + 40);
            float16x8_t b21 = MlasLoadFloat16x8(PackedB + 72);
            float16x8_t b31 = MlasLoadFloat16x8(PackedB + 104);
            float16x8_t b02 = MlasLoadFloat16x8(PackedB + 16);
            float16x8_t b12 = MlasLoadFloat16x8(PackedB + 48);
            float16x8_t b22 = MlasLoadFloat16x8(PackedB + 80);
            float16x8_t b32 = MlasLoadFloat16x8(PackedB + 112);
            float16x8_t b03 = MlasLoadFloat16x8(PackedB + 24);
            float16x8_t b13 = MlasLoadFloat16x8(PackedB + 56);
            float16x8_t b23 = MlasLoadFloat16x8(PackedB + 88);
            float16x8_t b33 = MlasLoadFloat16x8(PackedB + 120);
            for (; k >= 8; k -= 4, a += 4, PackedB += 4 * 32) {
                accu00 = maq_lane_f16_accu(accu00, b00, b10, b20, b30, a0);
                accu01 = maq_lane_f16_accu(accu01, b01, b11, b21, b31, a0);
                accu02 = maq_lane_f16_accu(accu02, b02, b12, b22, b32, a0);
                accu03 = maq_lane_f16_accu(accu03, b03, b13, b23, b33, a0);
                if constexpr (CountM == 2) {
                    accu10 = maq_lane_f16_accu(accu10, b00, b10, b20, b30, a1);
                    accu11 = maq_lane_f16_accu(accu11, b01, b11, b21, b31, a1);
                    accu12 = maq_lane_f16_accu(accu12, b02, b12, b22, b32, a1);
                    accu13 = maq_lane_f16_accu(accu13, b03, b13, b23, b33, a1);
                }
                a0 = MlasLoadFloat16x4(a + 4);
                if constexpr (CountM == 2) {
                    a1 = MlasLoadFloat16x4(a + lda + 4);
                }
                b00 = MlasLoadFloat16x8(PackedB + 128);
                b10 = MlasLoadFloat16x8(PackedB + 160);
                b20 = MlasLoadFloat16x8(PackedB + 192);
                b30 = MlasLoadFloat16x8(PackedB + 224);
                b01 = MlasLoadFloat16x8(PackedB + 136);
                b11 = MlasLoadFloat16x8(PackedB + 168);
                b21 = MlasLoadFloat16x8(PackedB + 200);
                b31 = MlasLoadFloat16x8(PackedB + 232);
                b02 = MlasLoadFloat16x8(PackedB + 144);
                b12 = MlasLoadFloat16x8(PackedB + 176);
                b22 = MlasLoadFloat16x8(PackedB + 208);
                b32 = MlasLoadFloat16x8(PackedB + 240);
                b03 = MlasLoadFloat16x8(PackedB + 152);
                b13 = MlasLoadFloat16x8(PackedB + 184);
                b23 = MlasLoadFloat16x8(PackedB + 216);
                b33 = MlasLoadFloat16x8(PackedB + 248);
            }
            accu00 = maq_lane_f16_accu(accu00, b00, b10, b20, b30, a0);
            accu01 = maq_lane_f16_accu(accu01, b01, b11, b21, b31, a0);
            accu02 = maq_lane_f16_accu(accu02, b02, b12, b22, b32, a0);
            accu03 = maq_lane_f16_accu(accu03, b03, b13, b23, b33, a0);
            if constexpr (CountM == 2) {
                accu10 = maq_lane_f16_accu(accu10, b00, b10, b20, b30, a1);
                accu11 = maq_lane_f16_accu(accu11, b01, b11, b21, b31, a1);
                accu12 = maq_lane_f16_accu(accu12, b02, b12, b22, b32, a1);
                accu13 = maq_lane_f16_accu(accu13, b03, b13, b23, b33, a1);
            }
            k -= 4, a += 4, PackedB += 4 * 32;
        }

        if (Kr0) {
            float16x4_t a0 = MlasLoadPartialFloat16x4(a, k), a1;
            if constexpr (CountM == 2) {
                a1 = MlasLoadPartialFloat16x4(a + lda, k);
            }
            float16x8_t b00 = MlasLoadFloat16x8(PackedB);
            float16x8_t b01 = MlasLoadFloat16x8(PackedB + 8);
            float16x8_t b02 = MlasLoadFloat16x8(PackedB + 16);
            float16x8_t b03 = MlasLoadFloat16x8(PackedB + 24);
            accu00 = vfmaq_lane_f16(accu00, b00, a0, 0);
            accu01 = vfmaq_lane_f16(accu01, b01, a0, 0);
            accu02 = vfmaq_lane_f16(accu02, b02, a0, 0);
            accu03 = vfmaq_lane_f16(accu03, b03, a0, 0);
            if constexpr (CountM == 2) {
                accu10 = vfmaq_lane_f16(accu10, b00, a1, 0);
                accu11 = vfmaq_lane_f16(accu11, b01, a1, 0);
                accu12 = vfmaq_lane_f16(accu12, b02, a1, 0);
                accu13 = vfmaq_lane_f16(accu13, b03, a1, 0);
            }
            if (Kr1) {
                float16x8_t b10 = MlasLoadFloat16x8(PackedB + 32);
                float16x8_t b11 = MlasLoadFloat16x8(PackedB + 40);
                float16x8_t b12 = MlasLoadFloat16x8(PackedB + 48);
                float16x8_t b13 = MlasLoadFloat16x8(PackedB + 56);
                accu00 = vfmaq_lane_f16(accu00, b10, a0, 1);
                accu01 = vfmaq_lane_f16(accu01, b11, a0, 1);
                accu02 = vfmaq_lane_f16(accu02, b12, a0, 1);
                accu03 = vfmaq_lane_f16(accu03, b13, a0, 1);
                if constexpr (CountM == 2) {
                    accu10 = vfmaq_lane_f16(accu10, b10, a1, 1);
                    accu11 = vfmaq_lane_f16(accu11, b11, a1, 1);
                    accu12 = vfmaq_lane_f16(accu12, b12, a1, 1);
                    accu13 = vfmaq_lane_f16(accu13, b13, a1, 1);
                }
            }
            if (Kr2) {
                float16x8_t b20 = MlasLoadFloat16x8(PackedB + 64);
                float16x8_t b21 = MlasLoadFloat16x8(PackedB + 72);
                float16x8_t b22 = MlasLoadFloat16x8(PackedB + 80);
                float16x8_t b23 = MlasLoadFloat16x8(PackedB + 88);
                accu00 = vfmaq_lane_f16(accu00, b20, a0, 2);
                accu01 = vfmaq_lane_f16(accu01, b21, a0, 2);
                accu02 = vfmaq_lane_f16(accu02, b22, a0, 2);
                accu03 = vfmaq_lane_f16(accu03, b23, a0, 2);
                if constexpr (CountM == 2) {
                    accu10 = vfmaq_lane_f16(accu10, b20, a1, 2);
                    accu11 = vfmaq_lane_f16(accu11, b21, a1, 2);
                    accu12 = vfmaq_lane_f16(accu12, b22, a1, 2);
                    accu13 = vfmaq_lane_f16(accu13, b23, a1, 2);
                }
            }
            PackedB += k * 32;
        }

        if constexpr (beta_behavior == 1) {
            float16x8_t c00 = MlasLoadFloat16x8(C);
            float16x8_t c01 = MlasLoadFloat16x8(C + 8);
            float16x8_t c02 = MlasLoadFloat16x8(C + 16);
            float16x8_t c03 = MlasLoadFloat16x8(C + 24);

            MlasStoreFloat16x8(C, vfmaq_f16(c00, accu00, alpha_v8));
            MlasStoreFloat16x8(C + 8, vfmaq_f16(c01, accu01, alpha_v8));
            MlasStoreFloat16x8(C + 16, vfmaq_f16(c02, accu02, alpha_v8));
            MlasStoreFloat16x8(C + 24, vfmaq_f16(c03, accu03, alpha_v8));
            if constexpr (CountM == 2) {
                float16x8_t c10 = MlasLoadFloat16x8(C + ldc);
                float16x8_t c11 = MlasLoadFloat16x8(C + ldc + 8);
                float16x8_t c12 = MlasLoadFloat16x8(C + ldc + 16);
                float16x8_t c13 = MlasLoadFloat16x8(C + ldc + 24);
                MlasStoreFloat16x8(C + ldc, vfmaq_f16(c10, accu10, alpha_v8));
                MlasStoreFloat16x8(C + ldc + 8, vfmaq_f16(c11, accu11, alpha_v8));
                MlasStoreFloat16x8(C + ldc + 16, vfmaq_f16(c12, accu12, alpha_v8));
                MlasStoreFloat16x8(C + ldc + 24, vfmaq_f16(c13, accu13, alpha_v8));
            }
        } else if constexpr (beta_behavior == 2) {
            float16x8_t c00 = MlasLoadFloat16x8(C);
            float16x8_t c01 = MlasLoadFloat16x8(C + 8);
            float16x8_t c02 = MlasLoadFloat16x8(C + 16);
            float16x8_t c03 = MlasLoadFloat16x8(C + 24);

            MlasStoreFloat16x8(C, vfmaq_f16(vmulq_f16(c00, beta_v8), accu00, alpha_v8));
            MlasStoreFloat16x8(C + 8, vfmaq_f16(vmulq_f16(c01, beta_v8), accu01, alpha_v8));
            MlasStoreFloat16x8(C + 16, vfmaq_f16(vmulq_f16(c02, beta_v8), accu02, alpha_v8));
            MlasStoreFloat16x8(C + 24, vfmaq_f16(vmulq_f16(c03, beta_v8), accu03, alpha_v8));
            if constexpr (CountM == 2) {
                float16x8_t c10 = MlasLoadFloat16x8(C + ldc);
                float16x8_t c11 = MlasLoadFloat16x8(C + ldc + 8);
                float16x8_t c12 = MlasLoadFloat16x8(C + ldc + 16);
                float16x8_t c13 = MlasLoadFloat16x8(C + ldc + 24);
                MlasStoreFloat16x8(C + ldc, vfmaq_f16(vmulq_f16(c10, beta_v8), accu10, alpha_v8));
                MlasStoreFloat16x8(C + ldc + 8, vfmaq_f16(vmulq_f16(c11, beta_v8), accu11, alpha_v8));
                MlasStoreFloat16x8(C + ldc + 16, vfmaq_f16(vmulq_f16(c12, beta_v8), accu12, alpha_v8));
                MlasStoreFloat16x8(C + ldc + 24, vfmaq_f16(vmulq_f16(c13, beta_v8), accu13, alpha_v8));
            }
        } else {
            MlasStoreFloat16x8(C, vmulq_f16(accu00, alpha_v8));
            MlasStoreFloat16x8(C + 8, vmulq_f16(accu01, alpha_v8));
            MlasStoreFloat16x8(C + 16, vmulq_f16(accu02, alpha_v8));
            MlasStoreFloat16x8(C + 24, vmulq_f16(accu03, alpha_v8));
            if constexpr (CountM == 2) {
                MlasStoreFloat16x8(C + ldc, vmulq_f16(accu10, alpha_v8));
                MlasStoreFloat16x8(C + ldc + 8, vmulq_f16(accu11, alpha_v8));
                MlasStoreFloat16x8(C + ldc + 16, vmulq_f16(accu12, alpha_v8));
                MlasStoreFloat16x8(C + ldc + 24, vmulq_f16(accu13, alpha_v8));
            }
        }
    }

    if (CountN & 16) {
        const auto* a = A;
        size_t k = CountK;
        float16x8_t accu00 = MlasZeroFloat16x8();
        float16x8_t accu01 = MlasZeroFloat16x8(), accu10, accu11;
        if constexpr (CountM == 2) {
            accu10 = MlasZeroFloat16x8();
            accu11 = MlasZeroFloat16x8();
        }
        if (largeK) {
            float16x4_t a0 = MlasLoadFloat16x4(a), a1;
            if constexpr (CountM == 2) {
                a1 = MlasLoadFloat16x4(a + lda);
            }
            float16x8_t b00 = MlasLoadFloat16x8(PackedB);
            float16x8_t b01 = MlasLoadFloat16x8(PackedB + 8);
            float16x8_t b10 = MlasLoadFloat16x8(PackedB + 16);
            float16x8_t b11 = MlasLoadFloat16x8(PackedB + 24);
            float16x8_t b20 = MlasLoadFloat16x8(PackedB + 32);
            float16x8_t b21 = MlasLoadFloat16x8(PackedB + 40);
            float16x8_t b30 = MlasLoadFloat16x8(PackedB + 48);
            float16x8_t b31 = MlasLoadFloat16x8(PackedB + 56);
            for (; k >= 8; k -= 4, a += 4, PackedB += 4 * 16) {
                accu00 = maq_lane_f16_accu(accu00, b00, b10, b20, b30, a0);
                accu01 = maq_lane_f16_accu(accu01, b01, b11, b21, b31, a0);
                if constexpr (CountM == 2) {
                    accu10 = maq_lane_f16_accu(accu10, b00, b10, b20, b30, a1);
                    accu11 = maq_lane_f16_accu(accu11, b01, b11, b21, b31, a1);
                }
                a0 = MlasLoadFloat16x4(a + 4);
                if constexpr (CountM == 2) {
                    a1 = MlasLoadFloat16x4(a + lda + 4);
                }
                b00 = MlasLoadFloat16x8(PackedB + 64);
                b01 = MlasLoadFloat16x8(PackedB + 72);
                b10 = MlasLoadFloat16x8(PackedB + 80);
                b11 = MlasLoadFloat16x8(PackedB + 88);
                b20 = MlasLoadFloat16x8(PackedB + 96);
                b21 = MlasLoadFloat16x8(PackedB + 104);
                b30 = MlasLoadFloat16x8(PackedB + 112);
                b31 = MlasLoadFloat16x8(PackedB + 120);
            }
            accu00 = maq_lane_f16_accu(accu00, b00, b10, b20, b30, a0);
            accu01 = maq_lane_f16_accu(accu01, b01, b11, b21, b31, a0);
            if constexpr (CountM == 2) {
                accu10 = maq_lane_f16_accu(accu10, b00, b10, b20, b30, a1);
                accu11 = maq_lane_f16_accu(accu11, b01, b11, b21, b31, a1);
            }
            k -= 4, a += 4, PackedB += 4 * 16;
        }

        if (Kr0) {
            float16x8_t b00 = MlasLoadFloat16x8(PackedB);
            float16x8_t b01 = MlasLoadFloat16x8(PackedB + 8);
            float16x4_t a0 = MlasLoadPartialFloat16x4(a, k), a1;
            accu00 = vfmaq_lane_f16(accu00, b00, a0, 0);
            accu01 = vfmaq_lane_f16(accu01, b01, a0, 0);
            if constexpr (CountM == 2) {
                a1 = MlasLoadPartialFloat16x4(a + lda, k);
                accu10 = vfmaq_lane_f16(accu10, b00, a1, 0);
                accu11 = vfmaq_lane_f16(accu11, b01, a1, 0);
            }
            if (Kr1) {
                float16x8_t b10 = MlasLoadFloat16x8(PackedB + 16);
                float16x8_t b11 = MlasLoadFloat16x8(PackedB + 24);
                accu00 = vfmaq_lane_f16(accu00, b10, a0, 1);
                accu01 = vfmaq_lane_f16(accu01, b11, a0, 1);
                if constexpr (CountM == 2) {
                    accu10 = vfmaq_lane_f16(accu10, b10, a1, 1);
                    accu11 = vfmaq_lane_f16(accu11, b11, a1, 1);
                }
            }
            if (Kr2) {
                float16x8_t b20 = MlasLoadFloat16x8(PackedB + 32);
                float16x8_t b21 = MlasLoadFloat16x8(PackedB + 40);
                accu00 = vfmaq_lane_f16(accu00, b20, a0, 2);
                accu01 = vfmaq_lane_f16(accu01, b21, a0, 2);
                if constexpr (CountM == 2) {
                    accu10 = vfmaq_lane_f16(accu10, b20, a1, 2);
                    accu11 = vfmaq_lane_f16(accu11, b21, a1, 2);
                }
            }
            PackedB += k * 16;
        }

        if constexpr (beta_behavior == 1) {
            float16x8_t c00 = MlasLoadFloat16x8(C);
            float16x8_t c01 = MlasLoadFloat16x8(C + 8);
            accu00 = vfmaq_f16(c00, accu00, alpha_v8);
            accu01 = vfmaq_f16(c01, accu01, alpha_v8);
            MlasStoreFloat16x8(C, accu00);
            MlasStoreFloat16x8(C + 8, accu01);
            if constexpr (CountM == 2) {
                float16x8_t c10 = MlasLoadFloat16x8(C + ldc);
                float16x8_t c11 = MlasLoadFloat16x8(C + ldc + 8);
                accu10 = vfmaq_f16(c10, accu10, alpha_v8);
                accu11 = vfmaq_f16(c11, accu11, alpha_v8);
                MlasStoreFloat16x8(C + ldc, accu10);
                MlasStoreFloat16x8(C + ldc + 8, accu11);
            }
        } else if constexpr (beta_behavior == 2) {
            float16x8_t c00 = MlasLoadFloat16x8(C);
            float16x8_t c01 = MlasLoadFloat16x8(C + 8);
            accu00 = vfmaq_f16(vmulq_f16(c00, beta_v8), accu00, alpha_v8);
            accu01 = vfmaq_f16(vmulq_f16(c01, beta_v8), accu01, alpha_v8);
            MlasStoreFloat16x8(C, accu00);
            MlasStoreFloat16x8(C + 8, accu01);
            if constexpr (CountM == 2) {
                float16x8_t c10 = MlasLoadFloat16x8(C + ldc);
                float16x8_t c11 = MlasLoadFloat16x8(C + ldc + 8);
                accu10 = vfmaq_f16(vmulq_f16(c10, beta_v8), accu10, alpha_v8);
                accu11 = vfmaq_f16(vmulq_f16(c11, beta_v8), accu11, alpha_v8);
                MlasStoreFloat16x8(C + ldc, accu10);
                MlasStoreFloat16x8(C + ldc + 8, accu11);
            }
        } else {
            accu00 = vmulq_f16(accu00, alpha_v8);
            accu01 = vmulq_f16(accu01, alpha_v8);
            MlasStoreFloat16x8(C, accu00);
            MlasStoreFloat16x8(C + 8, accu01);
            if constexpr (CountM == 2) {
                accu10 = vmulq_f16(accu10, alpha_v8);
                accu11 = vmulq_f16(accu11, alpha_v8);
                MlasStoreFloat16x8(C + ldc, accu10);
                MlasStoreFloat16x8(C + ldc + 8, accu11);
            }
        }

        CountN -= 16, C += 16;
    }

    if (CountN & 8) {
        const auto* a = A;
        size_t k = CountK;
        float16x8_t accu00 = MlasZeroFloat16x8();
        float16x8_t accu10 = MlasZeroFloat16x8();
        if (largeK) {
            float16x4_t a0 = MlasLoadFloat16x4(a), a1;
            if constexpr (CountM == 2) {
                a1 = MlasLoadFloat16x4(a + lda);
            }
            float16x8_t b0 = MlasLoadFloat16x8(PackedB);
            float16x8_t b1 = MlasLoadFloat16x8(PackedB + 8);
            float16x8_t b2 = MlasLoadFloat16x8(PackedB + 16);
            float16x8_t b3 = MlasLoadFloat16x8(PackedB + 24);
            for (; k >= 8; k -= 4, a += 4, PackedB += 4 * 8) {
                accu00 = maq_lane_f16_accu(accu00, b0, b1, b2, b3, a0);
                if constexpr (CountM == 2) {
                    accu10 = maq_lane_f16_accu(accu10, b0, b1, b2, b3, a1);
                }
                a0 = MlasLoadFloat16x4(a + 4);
                if constexpr (CountM == 2) {
                    a1 = MlasLoadFloat16x4(a + lda + 4);
                }
                b0 = MlasLoadFloat16x8(PackedB + 32);
                b1 = MlasLoadFloat16x8(PackedB + 40);
                b2 = MlasLoadFloat16x8(PackedB + 48);
                b3 = MlasLoadFloat16x8(PackedB + 56);
            }
            accu00 = maq_lane_f16_accu(accu00, b0, b1, b2, b3, a0);
            if constexpr (CountM == 2) {
                accu10 = maq_lane_f16_accu(accu10, b0, b1, b2, b3, a1);
            }
            k -= 4, a += 4, PackedB += 4 * 8;
        }

        if (Kr0) {
            float16x4_t a0 = MlasLoadPartialFloat16x4(a, k), a1;
            if constexpr (CountM == 2) {
                a1 = MlasLoadPartialFloat16x4(a + lda, k);
            }
            float16x8_t b0 = MlasLoadFloat16x8(PackedB);
            accu00 = vfmaq_lane_f16(accu00, b0, a0, 0);
            if constexpr (CountM == 2) {
                accu10 = vfmaq_lane_f16(accu10, b0, a1, 0);
            }
            if (Kr1) {
                float16x8_t b1 = MlasLoadFloat16x8(PackedB + 8);
                accu00 = vfmaq_lane_f16(accu00, b1, a0, 1);
                if constexpr (CountM == 2) {
                    accu10 = vfmaq_lane_f16(accu10, b1, a1, 1);
                }
            }
            if (Kr2) {
                float16x8_t b2 = MlasLoadFloat16x8(PackedB + 16);
                accu00 = vfmaq_lane_f16(accu00, b2, a0, 2);
                if constexpr (CountM == 2) {
                    accu10 = vfmaq_lane_f16(accu10, b2, a1, 2);
                }
            }
            PackedB += k * 8;
        }

        if constexpr (beta_behavior == 1) {
            float16x8_t c0 = MlasLoadFloat16x8(C);
            accu00 = vfmaq_f16(c0, accu00, alpha_v8);
            MlasStoreFloat16x8(C, accu00);
            if constexpr (CountM == 2) {
                float16x8_t c1 = MlasLoadFloat16x8(C + ldc);
                accu10 = vfmaq_f16(c1, accu10, alpha_v8);
                MlasStoreFloat16x8(C + ldc, accu10);
            }
        } else if constexpr (beta_behavior == 2) {
            float16x8_t c0 = MlasLoadFloat16x8(C);
            accu00 = vfmaq_f16(vmulq_f16(c0, beta_v8), accu00, alpha_v8);
            MlasStoreFloat16x8(C, accu00);
            if constexpr (CountM == 2) {
                float16x8_t c1 = MlasLoadFloat16x8(C + ldc);
                accu10 = vfmaq_f16(vmulq_f16(c1, beta_v8), accu10, alpha_v8);
                MlasStoreFloat16x8(C + ldc, accu10);
            }
        } else {
            accu00 = vmulq_f16(accu00, alpha_v8);
            MlasStoreFloat16x8(C, accu00);
            if constexpr (CountM == 2) {
                accu10 = vmulq_f16(accu10, alpha_v8);
                MlasStoreFloat16x8(C + ldc, accu10);
            }
        }

        CountN -= 8, C += 8;
    }

    if (CountN > 0) {
        const auto* a = A;
        size_t k = CountK;
        float16x8_t accu0 = MlasZeroFloat16x8(), accu1;
        if constexpr (CountM == 2) {
            accu1 = MlasZeroFloat16x8();
        }
        if (largeK) {
            float16x4_t a0 = MlasLoadFloat16x4(a), a1;
            if constexpr (CountM == 2) {
                a1 = MlasLoadFloat16x4(a + lda);
            }
            float16x8_t b0 = MlasLoadFloat16x8(PackedB);
            float16x8_t b1 = MlasLoadFloat16x8(PackedB + 8);
            float16x8_t b2 = MlasLoadFloat16x8(PackedB + 16);
            float16x8_t b3 = MlasLoadFloat16x8(PackedB + 24);
            for (; k >= 8; k -= 4, a += 4, PackedB += 4 * 8) {
                accu0 = maq_lane_f16_accu(accu0, b0, b1, b2, b3, a0);
                if constexpr (CountM == 2) {
                    accu1 = maq_lane_f16_accu(accu1, b0, b1, b2, b3, a1);
                }
                a0 = MlasLoadFloat16x4(a + 4);
                if constexpr (CountM == 2) {
                    a1 = MlasLoadFloat16x4(a + lda + 4);
                }
                b0 = MlasLoadFloat16x8(PackedB + 32);
                b1 = MlasLoadFloat16x8(PackedB + 40);
                b2 = MlasLoadFloat16x8(PackedB + 48);
                b3 = MlasLoadFloat16x8(PackedB + 56);
            }
            accu0 = maq_lane_f16_accu(accu0, b0, b1, b2, b3, a0);
            if constexpr (CountM == 2) {
                accu1 = maq_lane_f16_accu(accu1, b0, b1, b2, b3, a1);
            }
            k -= 4, a += 4, PackedB += 4 * 8;
        }

        if (Kr0) {
            float16x4_t a0 = MlasLoadPartialFloat16x4(a, k), a1;
            if constexpr (CountM == 2) {
                a1 = MlasLoadPartialFloat16x4(a + lda, k);
            }
            float16x8_t b0 = MlasLoadFloat16x8(PackedB);
            accu0 = vfmaq_lane_f16(accu0, b0, a0, 0);
            if constexpr (CountM == 2) {
                accu1 = vfmaq_lane_f16(accu1, b0, a1, 0);
            }
            if (Kr1) {
                float16x8_t b1 = MlasLoadFloat16x8(PackedB + 8);
                accu0 = vfmaq_lane_f16(accu0, b1, a0, 1);
                if constexpr (CountM == 2) {
                    accu1 = vfmaq_lane_f16(accu1, b1, a1, 1);
                }
            }
            if (Kr2) {
                float16x8_t b2 = MlasLoadFloat16x8(PackedB + 16);
                accu0 = vfmaq_lane_f16(accu0, b2, a0, 2);
                if constexpr (CountM == 2) {
                    accu1 = vfmaq_lane_f16(accu1, b2, a1, 2);
                }
            }
            PackedB += k * 8;
        }

        float16x4_t accu0_low = vget_low_f16(accu0);
        float16x4_t accu0_high = vget_high_f16(accu0);
        float16x4_t accu1_low, accu1_high;
        if constexpr (CountM == 2) {
            accu1_low = vget_low_f16(accu1);
            accu1_high = vget_high_f16(accu1);
        }

        if (CountN & 4) {
            if constexpr (beta_behavior == 1) {
                float16x4_t c0 = MlasLoadFloat16x4(C);
                MlasStoreFloat16x4(C, vfma_f16(c0, accu0_low, alpha_v4));
                if constexpr (CountM == 2) {
                    float16x4_t c1 = MlasLoadFloat16x4(C + ldc);
                    MlasStoreFloat16x4(C + ldc, vfma_f16(c1, accu1_low, alpha_v4));
                }
            } else if constexpr (beta_behavior == 2) {
                float16x4_t c0 = MlasLoadFloat16x4(C);
                MlasStoreFloat16x4(C, vfma_f16(vmul_f16(c0, beta_v4), accu0_low, alpha_v4));
                if constexpr (CountM == 2) {
                    float16x4_t c1 = MlasLoadFloat16x4(C + ldc);
                    MlasStoreFloat16x4(C + ldc, vfma_f16(vmul_f16(c1, beta_v4), accu1_low, alpha_v4));
                }
            } else {
                MlasStoreFloat16x4(C, vmul_f16(accu0_low, alpha_v4));
                if constexpr (CountM == 2) {
                    MlasStoreFloat16x4(C + ldc, vmul_f16(accu1_low, alpha_v4));
                }
            }
            CountN -= 4, C += 4;
            accu0_low = accu0_high;
            if constexpr (CountM == 2) {
                accu1_low = accu1_high;
            }
        }

        if (CountN) {
            if constexpr (beta_behavior == 1) {
                float16x4_t c0 = MlasLoadPartialFloat16x4(C, CountN);
                MlasStorePartialFloat16x4(C, vfma_f16(c0, accu0_low, alpha_v4), CountN);
                if constexpr (CountM == 2) {
                    float16x4_t c1 = MlasLoadPartialFloat16x4(C + ldc, CountN);
                    MlasStorePartialFloat16x4(C + ldc, vfma_f16(c1, accu1_low, alpha_v4), CountN);
                }
            } else if constexpr (beta_behavior == 2) {
                float16x4_t c0 = MlasLoadPartialFloat16x4(C, CountN);
                MlasStorePartialFloat16x4(C, vfma_f16(vmul_f16(c0, beta_v4), accu0_low, alpha_v4), CountN);
                if constexpr (CountM == 2) {
                    float16x4_t c1 = MlasLoadPartialFloat16x4(C + ldc, CountN);
                    MlasStorePartialFloat16x4(C + ldc, vfma_f16(vmul_f16(c1, beta_v4), accu1_low, alpha_v4), CountN);
                }
            } else {
                MlasStorePartialFloat16x4(C, vmul_f16(accu0_low, alpha_v4), CountN);
                if constexpr (CountM == 2) {
                    MlasStorePartialFloat16x4(C + ldc, vmul_f16(accu1_low, alpha_v4), CountN);
                }
            }
        }
    }
}

void HGemm_PackedB_Kernel(
    const MLAS_FP16* A,
    const MLAS_FP16* PackedB,
    MLAS_FP16* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldc,
    _mlas_fp16_ alpha,
    _mlas_fp16_ beta
) {
    if (CountM > 2) {
        MLAS_THROW_EX(std::runtime_error, "HGemm_PackedB_Kernel only support <= 2 rows");
    }

    const auto* A_data = reinterpret_cast<const _mlas_fp16_*>(A);
    const auto* PackedB_data = reinterpret_cast<const _mlas_fp16_*>(PackedB);
    auto* C_data = reinterpret_cast<_mlas_fp16_*>(C);
    const auto f16_0 = MLAS_FP16(0.0f);
    const auto f16_1 = MLAS_FP16(1.0f);
    if (CountM == 1) {
        if (beta == f16_0.val) {
            HGemm_PackedB_Kernel_Impl<0, 1>(A_data, PackedB_data, C_data, CountN, CountK, lda, ldc, alpha, beta);
        } else if (beta == f16_1.val) {
            HGemm_PackedB_Kernel_Impl<1, 1>(A_data, PackedB_data, C_data, CountN, CountK, lda, ldc, alpha, beta);
        } else {
            HGemm_PackedB_Kernel_Impl<2, 1>(A_data, PackedB_data, C_data, CountN, CountK, lda, ldc, alpha, beta);
        }
    } else {
        if (beta == f16_0.val) {
            HGemm_PackedB_Kernel_Impl<0, 2>(A_data, PackedB_data, C_data, CountN, CountK, lda, ldc, alpha, beta);
        } else if (beta == f16_1.val) {
            HGemm_PackedB_Kernel_Impl<1, 2>(A_data, PackedB_data, C_data, CountN, CountK, lda, ldc, alpha, beta);
        } else {
            HGemm_PackedB_Kernel_Impl<2, 2>(A_data, PackedB_data, C_data, CountN, CountK, lda, ldc, alpha, beta);
        }
    }
}

}  // namespace hgemm_neon
