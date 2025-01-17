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

namespace hgemm_neon {

template <size_t CountN, size_t CountM>
typename std::enable_if_t<((CountN >= 1 && CountN <= 8 && ((CountN - 1) & CountN) == 0) && (CountM == 1 || CountM == 2)), void>
HGemmTransB_Kernel_Block(
    const _mlas_fp16_* A,
    const _mlas_fp16_* B,
    const _mlas_fp16_* Bias,
    _mlas_fp16_* C,
    size_t K,
    size_t lda,
    size_t ldb,
    size_t ldc
) {
    using RegisterType = typename std::conditional_t<(CountN < 8), float16x4_t, float16x8_t>;

    RegisterType accu00, accu01, accu10, accu11;
    constexpr size_t b_step = CountN >= 8 ? 8 : 1;
    constexpr size_t N = CountN == 16 ? 8 : CountN;

    if constexpr (CountM == 2) {
        accu00 = accu10 = PrepareAccumulator<N>(Bias);
    } else {
        accu00 = PrepareAccumulator<N>(Bias);
    }
    if constexpr (CountN == 16) {
        if constexpr (CountM == 2) {
            accu01 = accu11 = PrepareAccumulator<N>(Bias ? Bias + 8 : nullptr);
        } else {
            accu01 = PrepareAccumulator<N>(Bias ? Bias + 8 : nullptr);
        }
    }

    size_t k = 0;
    for (; k + 8 <= K; k += 8, A += 8, B += b_step * 8) {
        accu00 = HQ4BitGemmMicroKernel<N, 1, 8>(A, B, ldb, accu00);
        if constexpr (CountN == 16) {
            accu01 = HQ4BitGemmMicroKernel<N, 1, 8>(A, B + b_step * ldb, ldb, accu01);
        }
        if constexpr (CountM == 2) {
            accu10 = HQ4BitGemmMicroKernel<N, 1, 8>(A + lda, B, ldb, accu10);
            if constexpr (CountN == 16) {
                accu11 = HQ4BitGemmMicroKernel<N, 1, 8>(A + lda, B + b_step * ldb, ldb, accu11);
            }
        }
    }

    if (K & 4) {
        accu00 = HQ4BitGemmMicroKernel<N, 1, 4>(A, B, ldb, accu00);
        if constexpr (CountN == 16) {
            accu01 = HQ4BitGemmMicroKernel<N, 1, 4>(A, B + b_step * ldb, ldb, accu01);
        }
        if constexpr (CountM == 2) {
            accu10 = HQ4BitGemmMicroKernel<N, 1, 4>(A + lda, B, ldb, accu10);
            if constexpr (CountN == 16) {
                accu11 = HQ4BitGemmMicroKernel<N, 1, 4>(A + lda, B + b_step * ldb, ldb, accu11);
            }
        }
        k += 4, A += 4, B += b_step * 4;
    }

    if (K & 2) {
        accu00 = HQ4BitGemmMicroKernel<N, 1, 2>(A, B, ldb, accu00);
        if constexpr (CountN == 16) {
            accu01 = HQ4BitGemmMicroKernel<N, 1, 2>(A, B + b_step * ldb, ldb, accu01);
        }
        if constexpr (CountM == 2) {
            accu10 = HQ4BitGemmMicroKernel<N, 1, 2>(A + lda, B, ldb, accu10);
            if constexpr (CountN == 16) {
                accu11 = HQ4BitGemmMicroKernel<N, 1, 2>(A + lda, B + b_step * ldb, ldb, accu11);
            }
        }
        k += 2, A += 2, B += b_step * 2;
    }

    if (k < K) {
        accu00 = HQ4BitGemmMicroKernel<N, 1, 1>(A, B, ldb, accu00);
        if constexpr (CountN == 16) {
            accu01 = HQ4BitGemmMicroKernel<N, 1, 1>(A, B + b_step * ldb, ldb, accu01);
        }
        if constexpr (CountM == 2) {
            accu10 = HQ4BitGemmMicroKernel<N, 1, 1>(A + lda, B, ldb, accu10);
            if constexpr (CountN == 16) {
                accu11 = HQ4BitGemmMicroKernel<N, 1, 1>(A + lda, B + b_step * ldb, ldb, accu11);
            }
        }
    }

    if constexpr (CountN >= 8) {
        MlasStoreFloat16x8(C, accu00);
        if constexpr (CountN == 16) {
            MlasStoreFloat16x8(C + 8, accu01);
        }
    } else if constexpr (CountN == 4) {
        MlasStoreFloat16x4(C, accu00);
    } else {
        MlasStoreLaneFloat16x4<0>(C, accu00);
        if constexpr (CountN == 2) {
            MlasStoreLaneFloat16x4<1>(C + 1, accu00);
        }
    }

    if constexpr (CountM == 2) {
        if constexpr (CountN >= 8) {
            MlasStoreFloat16x8(C + ldc, accu10);
            if constexpr (CountN == 16) {
                MlasStoreFloat16x8(C + ldc + 8, accu11);
            }
        } else if constexpr (CountN == 4) {
            MlasStoreFloat16x4(C + ldc, accu10);
        } else {
            MlasStoreLaneFloat16x4<0>(C + ldc, accu10);
            if constexpr (CountN == 2) {
                MlasStoreLaneFloat16x4<1>(C + ldc + 1, accu10);
            }
        }
    }
}

void HPackB_TransposedB_Kernel(
    const MLAS_FP16* B,
    MLAS_FP16* PackedB,
    size_t CountN,
    size_t CountK,
    size_t ldb
) {
    const _mlas_fp16_* B_data = reinterpret_cast<const _mlas_fp16_*>(B);
    _mlas_fp16_* PackedB_data = reinterpret_cast<_mlas_fp16_*>(PackedB);

    for (; CountN >= 16; CountN -= 16, B_data += 16 * ldb) {
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

        if (k & 4) {
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

        if (k > 0) {
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
            if (k > 1) {
                MlasStoreFloat16x4(PackedB_data + 16, v1);
                MlasStoreFloat16x4(PackedB_data + 20, v5);
                MlasStoreFloat16x4(PackedB_data + 24, v9);
                MlasStoreFloat16x4(PackedB_data + 28, vD);
            }
            if (k > 2) {
                MlasStoreFloat16x4(PackedB_data + 32, v2);
                MlasStoreFloat16x4(PackedB_data + 36, v6);
                MlasStoreFloat16x4(PackedB_data + 40, vA);
                MlasStoreFloat16x4(PackedB_data + 44, vE);
            }

            PackedB_data += k * 16;
        }
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

        if (k & 4) {
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

        if (k > 0) {
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
            if (k > 1) {
                MlasStoreFloat16x4(PackedB_data + 8, v1);
                MlasStoreFloat16x4(PackedB_data + 12, v5);
            }
            if (k > 2) {
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

        if (k & 4) {
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

        if (k > 0) {
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
            if (k > 1) {
                MlasStoreFloat16x4(PackedB_data + 8, v[1]);
                MlasStoreFloat16x4(PackedB_data + 12, v[5]);
            }
            if (k > 2) {
                MlasStoreFloat16x4(PackedB_data + 16, v[2]);
                MlasStoreFloat16x4(PackedB_data + 20, v[6]);
            }
        }
    }
}

void HGemm_TransposedB_Kernel_M1(
    const _mlas_fp16_* A_data,
    const _mlas_fp16_* B_data,
    _mlas_fp16_* C_data,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc,
    float16_t alpha,
    float16_t beta
) {
    for (; CountN >= 8; CountN -= 8, B_data += 8 * ldb, C_data += 8) {
        const auto* a = A_data;
        const auto* b = B_data;
        size_t k = CountK;
        float16x8_t accu0 = MlasZeroFloat16x8();
        float16x8_t accu1 = MlasZeroFloat16x8();
        float16x8_t accu2 = MlasZeroFloat16x8();
        float16x8_t accu3 = MlasZeroFloat16x8();
        float16x8_t accu4 = MlasZeroFloat16x8();
        float16x8_t accu5 = MlasZeroFloat16x8();
        float16x8_t accu6 = MlasZeroFloat16x8();
        float16x8_t accu7 = MlasZeroFloat16x8();
        for (; k >= 8; k -= 8, a += 8, b += 8) {
            float16x8_t b0 = MlasLoadFloat16x8(b);
            float16x8_t b1 = MlasLoadFloat16x8(b + ldb);
            float16x8_t b2 = MlasLoadFloat16x8(b + 2 * ldb);
            float16x8_t b3 = MlasLoadFloat16x8(b + 3 * ldb);
            float16x8_t b4 = MlasLoadFloat16x8(b + 4 * ldb);
            float16x8_t b5 = MlasLoadFloat16x8(b + 5 * ldb);
            float16x8_t b6 = MlasLoadFloat16x8(b + 6 * ldb);
            float16x8_t b7 = MlasLoadFloat16x8(b + 7 * ldb);
            float16x8_t a0 = MlasLoadFloat16x8(a);
            accu0 = vfmaq_f16(accu0, b0, a0);
            accu1 = vfmaq_f16(accu1, b1, a0);
            accu2 = vfmaq_f16(accu2, b2, a0);
            accu3 = vfmaq_f16(accu3, b3, a0);
            accu4 = vfmaq_f16(accu4, b4, a0);
            accu5 = vfmaq_f16(accu5, b5, a0);
            accu6 = vfmaq_f16(accu6, b6, a0);
            accu7 = vfmaq_f16(accu7, b7, a0);
        }
        Transpose8x8(accu0, accu1, accu2, accu3, accu4, accu5, accu6, accu7);
        accu0 = vaddq_f16(accu0, accu1);
        accu2 = vaddq_f16(accu2, accu3);
        accu4 = vaddq_f16(accu4, accu5);
        accu6 = vaddq_f16(accu6, accu7);
        accu0 = vaddq_f16(accu0, accu2);
        accu4 = vaddq_f16(accu4, accu6);
        accu0 = vaddq_f16(accu0, accu4); // accumulator of 8 columns

        if (k & 4) {
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
            accu0 = vfmaq_lane_f16(accu0, v0, a0, 0);
            accu0 = vfmaq_lane_f16(accu0, v1, a0, 1);
            accu0 = vfmaq_lane_f16(accu0, v2, a0, 2);
            accu0 = vfmaq_lane_f16(accu0, v3, a0, 3);
            k -= 4, a += 4, b += 4;
        }

        if (k > 0) {
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
            float16x8_t v0 = vcombine_f16(b0, b4), v1, v2;
            float16x4_t a0 = MlasLoadPartialFloat16x4(a, k);
            accu0 = vfmaq_lane_f16(accu0, v0, a0, 0);
            if (k > 1) {
                v1 = vcombine_f16(b1, b5);
                accu0 = vfmaq_lane_f16(accu0, v1, a0, 1);
            }
            if (k > 2) {
                v2 = vcombine_f16(b2, b6);
                accu0 = vfmaq_lane_f16(accu0, v2, a0, 2);
            }
        }

        if (beta == 1.0f16) {
            float16x8_t c = MlasLoadFloat16x8(C_data);
            accu0 = vfmaq_n_f16(c, accu0, alpha);
            MlasStoreFloat16x8(C_data, accu0);
        } else if (beta != 0.0f16) {
            float16x8_t c = MlasLoadFloat16x8(C_data);
            accu0 = vfmaq_n_f16(vmulq_n_f16(c, beta), accu0, alpha);
            MlasStoreFloat16x8(C_data, accu0);
        } else {
            accu0 = vmulq_n_f16(accu0, alpha);
            MlasStoreFloat16x8(C_data, accu0);
        }
    }

    if (CountN & 4) {
        const auto* a = A_data;
        const auto* b = B_data;
        size_t k = CountK;
        float16x8_t accu0 = MlasZeroFloat16x8();
        float16x8_t accu1 = MlasZeroFloat16x8();
        float16x8_t accu2 = MlasZeroFloat16x8();
        float16x8_t accu3 = MlasZeroFloat16x8();
        for (; k >= 8; k -= 8, a += 8, b += 8) {
            float16x8_t b0 = MlasLoadFloat16x8(b);
            float16x8_t b1 = MlasLoadFloat16x8(b + ldb);
            float16x8_t b2 = MlasLoadFloat16x8(b + 2 * ldb);
            float16x8_t b3 = MlasLoadFloat16x8(b + 3 * ldb);
            float16x8_t a0 = MlasLoadFloat16x8(a);
            accu0 = vfmaq_f16(accu0, b0, a0);
            accu1 = vfmaq_f16(accu1, b1, a0);
            accu2 = vfmaq_f16(accu2, b2, a0);
            accu3 = vfmaq_f16(accu3, b3, a0);
        }
        Transpose4x8(accu0, accu1, accu2, accu3);
        accu0 = vaddq_f16(accu0, accu1);
        accu2 = vaddq_f16(accu2, accu3);
        accu0 = vaddq_f16(accu0, accu2);
        float16x4_t accu = vadd_f16(vget_low_f16(accu0), vget_high_f16(accu0));

        if (k & 4) {
            float16x4_t b0 = MlasLoadFloat16x4(b);
            float16x4_t b1 = MlasLoadFloat16x4(b + ldb);
            float16x4_t b2 = MlasLoadFloat16x4(b + 2 * ldb);
            float16x4_t b3 = MlasLoadFloat16x4(b + 3 * ldb);
            Transpose4x4(b0, b1, b2, b3);
            float16x4_t a0 = MlasLoadFloat16x4(a);
            accu = vfma_lane_f16(accu, b0, a0, 0);
            accu = vfma_lane_f16(accu, b1, a0, 1);
            accu = vfma_lane_f16(accu, b2, a0, 2);
            accu = vfma_lane_f16(accu, b3, a0, 3);
            k -= 4, a += 4, b += 4;
        }

        if (k > 0) {
            float16x4_t b0 = MlasLoadPartialFloat16x4(b, k);
            float16x4_t b1 = MlasLoadPartialFloat16x4(b + ldb, k);
            float16x4_t b2 = MlasLoadPartialFloat16x4(b + 2 * ldb, k);
            float16x4_t b3 = MlasLoadPartialFloat16x4(b + 3 * ldb, k);
            Transpose4x4(b0, b1, b2, b3);
            float16x4_t a0 = MlasLoadPartialFloat16x4(a, k);
            accu = vfma_lane_f16(accu, b0, a0, 0);
            if (k > 1) {
                accu = vfma_lane_f16(accu, b1, a0, 1);
            }
            if (k > 2) {
                accu = vfma_lane_f16(accu, b2, a0, 2);
            }
        }

        if (beta == 1.0f16) {
            float16x4_t c = MlasLoadFloat16x4(C_data);
            accu = vfma_n_f16(c, accu, alpha);
            MlasStoreFloat16x4(C_data, accu);
        } else if (beta != 0.0f16) {
            float16x4_t c = MlasLoadFloat16x4(C_data);
            accu = vfma_n_f16(vmul_n_f16(c, beta), accu, alpha);
            MlasStoreFloat16x4(C_data, accu);
        } else {
            accu = vmulq_n_f16(accu, alpha);
            MlasStoreFloat16x4(C_data, accu);
        }

        CountN -= 4, B_data += 4 * ldb, C_data += 4;
    }

    if (CountN > 0) {
        const auto* a = A_data;
        const auto* b = B_data;
        size_t k = CountK;
        float16x8_t accu[4];
        size_t i = 0;
        for (i = 0; i < 4; ++i) {
            accu[i] = MlasZeroFloat16x8();
        }
        for (; k >= 8; k -= 8, a += 8, b += 8) {
            float16x8_t a0 = MlasLoadFloat16x8(a);
            for (i = 0; i < CountN; ++i) {
                accu[i] = vfmaq_f16(accu[i], MlasLoadFloat16x8(b + i * ldb), a0);
            }
        }
        Transpose4x8(accu[0], accu[1], accu[2], accu[3]);
        float16x8_t accu0 = vaddq_f16(accu[0], accu[1]);
        float16x8_t accu2 = vaddq_f16(accu[2], accu[3]);
        float16x8_t accu0 = vaddq_f16(accu0, accu2);
        float16x4_t accu = vadd_f16(vget_low_f16(accu0), vget_high_f16(accu0));

        if (k & 4) {
            float16x4_t b[4];
            for (i = 0; i < CountN; ++i) {
                b[i] = MlasLoadFloat16x4(b + i * ldb);
            }
            for (; i < 4; ++i) {
                b[i] = MlasZeroFloat16x4();
            }
            Transpose4x4(b[0], b[1], b[2], b[3]);
            float16x4_t a0 = MlasLoadFloat16x4(a);
            accu = vfma_lane_f16(accu, b[0], a0, 0);
            accu = vfma_lane_f16(accu, b[1], a0, 1);
            accu = vfma_lane_f16(accu, b[2], a0, 2);
            accu = vfma_lane_f16(accu, b[3], a0, 3);
            k -= 4, a += 4, b += 4;
        }

        if (k > 0) {
            float16x4_t b[4];
            for (i = 0; i < CountN; ++i) {
                b[i] = MlasLoadPartialFloat16x4(b + i * ldb, k);
            }
            for (; i < 4; ++i) {
                b[i] = MlasZeroFloat16x4();
            }
            Transpose4x4(b[0], b[1], b[2], b[3]);
            float16x4_t a0 = MlasLoadPartialFloat16x4(a, k);
            accu = vfma_lane_f16(accu, b[0], a0, 0);
            if (k > 1) {
                accu = vfma_lane_f16(accu, b[1], a0, 1);
            }
            if (k > 2) {
                accu = vfma_lane_f16(accu, b[2], a0, 2);
            }
        }

        if (beta == 1.0f16) {
            float16x4_t c = MlasLoadPartialFloat16x4(C_data, CountN);
            accu = vfma_n_f16(c, accu, alpha);
            MlasStorePartialFloat16x4(C_data, accu, CountN);
        } else if (beta != 0.0f16) {
            float16x4_t c = MlasLoadPartialFloat16x4(C_data, CountN);
            accu = vfma_n_f16(vmul_n_f16(c, beta), accu, alpha);
            MlasStorePartialFloat16x4(C_data, accu, CountN);
        } else {
            accu = vmulq_n_f16(accu, alpha);
            MlasStorePartialFloat16x4(C_data, accu, CountN);
        }
    }
}

void HGemm_TransposedB_Kernel_M2(
    const _mlas_fp16_* A,
    const _mlas_fp16_* B,
    _mlas_fp16_* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc,
    float16_t alpha,
    float16_t beta
) {
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
        float16x8_t accu10 = MlasZeroFloat16x8();
        float16x8_t accu11 = MlasZeroFloat16x8();
        float16x8_t accu12 = MlasZeroFloat16x8();
        float16x8_t accu13 = MlasZeroFloat16x8();
        float16x8_t accu14 = MlasZeroFloat16x8();
        float16x8_t accu15 = MlasZeroFloat16x8();
        float16x8_t accu16 = MlasZeroFloat16x8();
        float16x8_t accu17 = MlasZeroFloat16x8();
        for (; k >= 8; k -= 8, a += 8, b += 8) {
            float16x8_t b0 = MlasLoadFloat16x8(b);
            float16x8_t b1 = MlasLoadFloat16x8(b + ldb);
            float16x8_t b2 = MlasLoadFloat16x8(b + 2 * ldb);
            float16x8_t b3 = MlasLoadFloat16x8(b + 3 * ldb);
            float16x8_t b4 = MlasLoadFloat16x8(b + 4 * ldb);
            float16x8_t b5 = MlasLoadFloat16x8(b + 5 * ldb);
            float16x8_t b6 = MlasLoadFloat16x8(b + 6 * ldb);
            float16x8_t b7 = MlasLoadFloat16x8(b + 7 * ldb);
            float16x8_t a0 = MlasLoadFloat16x8(a);
            float16x8_t a1 = MlasLoadFloat16x8(a + lda);
            accu00 = vfmaq_f16(accu00, b0, a0);
            accu01 = vfmaq_f16(accu01, b1, a0);
            accu02 = vfmaq_f16(accu02, b2, a0);
            accu03 = vfmaq_f16(accu03, b3, a0);
            accu04 = vfmaq_f16(accu04, b4, a0);
            accu05 = vfmaq_f16(accu05, b5, a0);
            accu06 = vfmaq_f16(accu06, b6, a0);
            accu07 = vfmaq_f16(accu07, b7, a0);
            accu10 = vfmaq_f16(accu10, b0, a1);
            accu11 = vfmaq_f16(accu11, b1, a1);
            accu12 = vfmaq_f16(accu12, b2, a1);
            accu13 = vfmaq_f16(accu13, b3, a1);
            accu14 = vfmaq_f16(accu14, b4, a1);
            accu15 = vfmaq_f16(accu15, b5, a1);
            accu16 = vfmaq_f16(accu16, b6, a1);
            accu17 = vfmaq_f16(accu17, b7, a1);
        }
        Transpose8x8(accu00, accu01, accu02, accu03, accu04, accu05, accu06, accu07);
        Transpose8x8(accu10, accu11, accu12, accu13, accu14, accu15, accu16, accu17);
        accu00 = vaddq_f16(accu00, accu01);
        accu02 = vaddq_f16(accu02, accu03);
        accu04 = vaddq_f16(accu04, accu05);
        accu06 = vaddq_f16(accu06, accu07);
        accu00 = vaddq_f16(accu00, accu02);
        accu04 = vaddq_f16(accu04, accu06);
        accu00 = vaddq_f16(accu00, accu04); // accumulator of 8 columns
        accu10 = vaddq_f16(accu10, accu11);
        accu12 = vaddq_f16(accu12, accu13);
        accu14 = vaddq_f16(accu14, accu15);
        accu16 = vaddq_f16(accu16, accu17);
        accu10 = vaddq_f16(accu10, accu12);
        accu14 = vaddq_f16(accu14, accu16);
        accu10 = vaddq_f16(accu10, accu14); // accumulator of 8 columns

        if (k & 4) {
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
            float16x4_t a1 = MlasLoadFloat16x4(a + lda);
            accu00 = vfmaq_lane_f16(accu00, v0, a0, 0);
            accu00 = vfmaq_lane_f16(accu00, v1, a0, 1);
            accu00 = vfmaq_lane_f16(accu00, v2, a0, 2);
            accu00 = vfmaq_lane_f16(accu00, v3, a0, 3);
            accu10 = vfmaq_lane_f16(accu10, v0, a1, 0);
            accu10 = vfmaq_lane_f16(accu10, v1, a1, 1);
            accu10 = vfmaq_lane_f16(accu10, v2, a1, 2);
            accu10 = vfmaq_lane_f16(accu10, v3, a1, 3);
            k -= 4, a += 4, b += 4;
        }

        if (k > 0) {
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
            float16x4_t a0 = MlasLoadPartialFloat16x4(a, k);
            float16x4_t a1 = MlasLoadPartialFloat16x4(a + lda, k);
            accu00 = vfmaq_lane_f16(accu00, v0, a0, 0);
            accu10 = vfmaq_lane_f16(accu10, v0, a1, 0);
            if (k > 1) {
                float16x8_t v1 = vcombine_f16(b1, b5);
                accu00 = vfmaq_lane_f16(accu00, v1, a0, 1);
                accu10 = vfmaq_lane_f16(accu10, v1, a1, 1);
            }
            if (k > 2) {
                float16x8_t v2 = vcombine_f16(b2, b6);
                accu00 = vfmaq_lane_f16(accu00, v2, a0, 2);
                accu10 = vfmaq_lane_f16(accu10, v2, a1, 2);
            }
        }

        if (beta == 1.0f16) {
            float16x8_t c0 = MlasLoadFloat16x8(C_data);
            float16x8_t c1 = MlasLoadFloat16x8(C_data + ldc);
            accu00 = vfmaq_n_f16(c0, accu00, alpha);
            accu10 = vfmaq_n_f16(c1, accu10, alpha);
            MlasStoreFloat16x8(C_data, accu00);
            MlasStoreFloat16x8(C_data + ldc, accu10);
        } else if (beta != 0.0f16) {
            float16x8_t c0 = MlasLoadFloat16x8(C_data);
            float16x8_t c1 = MlasLoadFloat16x8(C_data + ldc);
            accu00 = vfmaq_n_f16(vmulq_n_f16(c0, beta), accu00, alpha);
            accu10 = vfmaq_n_f16(vmulq_n_f16(c1, beta), accu10, alpha);
            MlasStoreFloat16x8(C_data, accu00);
            MlasStoreFloat16x8(C_data + ldc, accu10);
        } else {
            accu00 = vmulq_n_f16(accu00, alpha);
            accu10 = vmulq_n_f16(accu10, alpha);
            MlasStoreFloat16x8(C_data, accu00);
            MlasStoreFloat16x8(C_data + ldc, accu10);
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
        float16x8_t accu10 = MlasZeroFloat16x8();
        float16x8_t accu11 = MlasZeroFloat16x8();
        float16x8_t accu12 = MlasZeroFloat16x8();
        float16x8_t accu13 = MlasZeroFloat16x8();
        for (; k >= 8; k -= 8, a += 8, b += 8) {
            float16x8_t b0 = MlasLoadFloat16x8(b);
            float16x8_t b1 = MlasLoadFloat16x8(b + ldb);
            float16x8_t b2 = MlasLoadFloat16x8(b + 2 * ldb);
            float16x8_t b3 = MlasLoadFloat16x8(b + 3 * ldb);
            float16x8_t a0 = MlasLoadFloat16x8(a);
            float16x8_t a1 = MlasLoadFloat16x8(a + lda);
            accu00 = vfmaq_f16(accu00, b0, a0);
            accu01 = vfmaq_f16(accu01, b1, a0);
            accu02 = vfmaq_f16(accu02, b2, a0);
            accu03 = vfmaq_f16(accu03, b3, a0);
            accu10 = vfmaq_f16(accu10, b0, a1);
            accu11 = vfmaq_f16(accu11, b1, a1);
            accu12 = vfmaq_f16(accu12, b2, a1);
            accu13 = vfmaq_f16(accu13, b3, a1);
        }
        Transpose4x8(accu00, accu01, accu02, accu03);
        Transpose4x8(accu10, accu11, accu12, accu13);
        accu00 = vaddq_f16(accu00, accu01);
        accu02 = vaddq_f16(accu02, accu03);
        accu00 = vaddq_f16(accu00, accu02); // final
        accu10 = vaddq_f16(accu10, accu11);
        accu12 = vaddq_f16(accu12, accu13);
        accu10 = vaddq_f16(accu10, accu12); // final
        float16x4_t accu0 = vadd_f16(vget_low_f16(accu00), vget_high_f16(accu00));
        float16x4_t accu1 = vadd_f16(vget_low_f16(accu10), vget_high_f16(accu10));

        if (k & 4) {
            float16x4_t b0 = MlasLoadFloat16x4(b);
            float16x4_t b1 = MlasLoadFloat16x4(b + ldb);
            float16x4_t b2 = MlasLoadFloat16x4(b + 2 * ldb);
            float16x4_t b3 = MlasLoadFloat16x4(b + 3 * ldb);
            Transpose4x4(b0, b1, b2, b3);
            float16x4_t a0 = MlasLoadFloat16x4(a);
            float16x4_t a1 = MlasLoadFloat16x4(a + lda);
            accu0 = vfma_lane_f16(accu0, b0, a0, 0);
            accu0 = vfma_lane_f16(accu0, b1, a0, 1);
            accu0 = vfma_lane_f16(accu0, b2, a0, 2);
            accu0 = vfma_lane_f16(accu0, b3, a0, 3);
            accu1 = vfma_lane_f16(accu1, b0, a1, 0);
            accu1 = vfma_lane_f16(accu1, b1, a1, 1);
            accu1 = vfma_lane_f16(accu1, b2, a1, 2);
            accu1 = vfma_lane_f16(accu1, b3, a1, 3);
            k -= 4, a += 4, b += 4;
        }

        if (k > 0) {
            float16x4_t b0 = MlasLoadPartialFloat16x4(b, k);
            float16x4_t b1 = MlasLoadPartialFloat16x4(b + ldb, k);
            float16x4_t b2 = MlasLoadPartialFloat16x4(b + 2 * ldb, k);
            float16x4_t b3 = MlasLoadPartialFloat16x4(b + 3 * ldb, k);
            Transpose4x4(b0, b1, b2, b3);
            float16x4_t a0 = MlasLoadPartialFloat16x4(a, k);
            float16x4_t a1 = MlasLoadPartialFloat16x4(a + lda, k);
            accu0 = vfma_lane_f16(accu0, b0, a0, 0);
            accu1 = vfma_lane_f16(accu1, b0, a1, 0);
            if (k > 1) {
                accu0 = vfma_lane_f16(accu0, b1, a0, 1);
                accu1 = vfma_lane_f16(accu1, b1, a1, 1);
            }
            if (k > 2) {
                accu0 = vfma_lane_f16(accu0, b2, a0, 2);
                accu1 = vfma_lane_f16(accu1, b2, a1, 2);
            }
        }

        if (beta == 1.0f16) {
            float16x4_t c0 = MlasLoadFloat16x4(C_data);
            float16x4_t c1 = MlasLoadFloat16x4(C_data + ldc);
            accu0 = vfma_n_f16(c0, accu0, alpha);
            accu1 = vfma_n_f16(c1, accu1, alpha);
            MlasStoreFloat16x4(C_data, accu0);
            MlasStoreFloat16x4(C_data + ldc, accu1);
        } else if (beta != 0.0f16) {
            float16x4_t c0 = MlasLoadFloat16x4(C_data);
            float16x4_t c1 = MlasLoadFloat16x4(C_data + ldc);
            accu0 = vfma_n_f16(vmul_n_f16(c0, beta), accu0, alpha);
            accu1 = vfma_n_f16(vmul_n_f16(c1, beta), accu1, alpha);
            MlasStoreFloat16x4(C_data, accu0);
            MlasStoreFloat16x4(C_data + ldc, accu1);
        } else {
            accu0 = vmulq_n_f16(accu0, alpha);
            accu1 = vmulq_n_f16(accu1, alpha);
            MlasStoreFloat16x4(C_data, accu0);
            MlasStoreFloat16x4(C_data + ldc, accu1);
        }

        CountN -= 4, B_data += 4 * ldb, C_data += 4;
    }

    if (CountN > 0) {
        const auto* a = A_data;
        const auto* b = B_data;
        size_t k = CountK;
        float16x8_t accu0[4];
        float16x8_t accu1[4];
        size_t i = 0;
        for (i = 0; i < 4; ++i) {
            accu0[i] = MlasZeroFloat16x8();
            accu1[i] = MlasZeroFloat16x8();
        }
        for (; k >= 8; k -= 8, a += 8, b += 8) {
            float16x8_t a0 = MlasLoadFloat16x8(a);
            float16x8_t a1 = MlasLoadFloat16x8(a + lda);
            for (i = 0; i < CountN; ++i) {
                float16x8_t bi = MlasLoadFloat16x8(b + i * ldb);
                accu0[i] = vfmaq_f16(accu0[i], bi, a0);
                accu1[i] = vfmaq_f16(accu1[i], bi, a1);
            }
        }
        Transpose4x8(accu0[0], accu0[1], accu0[2], accu0[3]);
        Transpose4x8(accu1[0], accu1[1], accu1[2], accu1[3]);
        float16x8_t accu00 = vaddq_f16(accu0[0], accu0[1]);
        float16x8_t accu02 = vaddq_f16(accu0[2], accu0[3]);
        accu00 = vaddq_f16(accu00, accu02);
        float16x4_t accu_0 = vadd_f16(vget_low_f16(accu00), vget_high_f16(accu00));
        float16x8_t accu10 = vaddq_f16(accu1[0], accu1[1]);
        float16x8_t accu12 = vaddq_f16(accu1[2], accu1[3]);
        accu10 = vaddq_f16(accu10, accu12);
        float16x4_t accu_1 = vadd_f16(vget_low_f16(accu10), vget_high_f16(accu10));

        if (k & 4) {
            float16x4_t b[4];
            for (i = 0; i < CountN; ++i) {
                b[i] = MlasLoadFloat16x4(b + i * ldb);
            }
            for (; i < 4; ++i) {
                b[i] = MlasZeroFloat16x4();
            }
            Transpose4x4(b[0], b[1], b[2], b[3]);
            float16x4_t a0 = MlasLoadFloat16x4(a);
            float16x4_t a1 = MlasLoadFloat16x4(a + lda);
            accu_0 = vfma_lane_f16(accu_0, b[0], a0, 0);
            accu_0 = vfma_lane_f16(accu_0, b[1], a0, 1);
            accu_0 = vfma_lane_f16(accu_0, b[2], a0, 2);
            accu_0 = vfma_lane_f16(accu_0, b[3], a0, 3);
            accu_1 = vfma_lane_f16(accu_1, b[0], a1, 0);
            accu_1 = vfma_lane_f16(accu_1, b[1], a1, 1);
            accu_1 = vfma_lane_f16(accu_1, b[2], a1, 2);
            accu_1 = vfma_lane_f16(accu_1, b[3], a1, 3);
            k -= 4, a += 4, b += 4;
        }

        if (k > 0) {
            float16x4_t b[4];
            for (i = 0; i < CountN; ++i) {
                b[i] = MlasLoadPartialFloat16x4(b + i * ldb, k);
            }
            for (; i < 4; ++i) {
                b[i] = MlasZeroFloat16x4();
            }
            Transpose4x4(b[0], b[1], b[2], b[3]);
            float16x4_t a0 = MlasLoadPartialFloat16x4(a, k);
            float16x4_t a1 = MlasLoadPartialFloat16x4(a + lda, k);
            accu_0 = vfma_lane_f16(accu_0, b[0], a0, 0);
            accu_1 = vfma_lane_f16(accu_1, b[0], a1, 0);
            if (k > 1) {
                accu_0 = vfma_lane_f16(accu_0, b[1], a0, 1);
                accu_1 = vfma_lane_f16(accu_1, b[1], a1, 1);
            }
            if (k > 2) {
                accu_0 = vfma_lane_f16(accu_0, b[2], a0, 2);
                accu_1 = vfma_lane_f16(accu_1, b[2], a1, 2);
            }
        }

        if (beta == 1.0f16) {
            float16x4_t c0 = MlasLoadPartialFloat16x4(C_data, CountN);
            float16x4_t c1 = MlasLoadPartialFloat16x4(C_data + ldc, CountN);
            accu_0 = vfma_n_f16(c0, accu_0, alpha);
            accu_1 = vfma_n_f16(c1, accu_1, alpha);
            MlasStorePartialFloat16x4(C_data, accu_0, CountN);
            MlasStorePartialFloat16x4(C_data + ldc, accu_1, CountN);
        } else if (beta != 0.0f16) {
            float16x4_t c0 = MlasLoadPartialFloat16x4(C_data, CountN);
            float16x4_t c1 = MlasLoadPartialFloat16x4(C_data + ldc, CountN);
            accu_0 = vfma_n_f16(vmul_n_f16(c0, beta), accu_0, alpha);
            accu_1 = vfma_n_f16(vmul_n_f16(c1, beta), accu_1, alpha);
            MlasStorePartialFloat16x4(C_data, accu_0, CountN);
            MlasStorePartialFloat16x4(C_data + ldc, accu_1, CountN);
        } else {
            accu_0 = vmulq_n_f16(accu_0, alpha);
            accu_1 = vmulq_n_f16(accu_1, alpha);
            MlasStorePartialFloat16x4(C_data, accu_0, CountN);
            MlasStorePartialFloat16x4(C_data + ldc, accu_1, CountN);
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
    MLAS_FP16 alpha,
    MLAS_FP16 beta
) {
    if (CountM > 2) {
        MLAS_THROW_EX(std::runtime_error, "HGemm_TransposedB_Kernel only support <= 2 rows");
    }
    const _mlas_fp16_ A_data = reinterpret_cast<const _mlas_fp16_*>(A);
    const _mlas_fp16_ B_data = reinterpret_cast<const _mlas_fp16_*>(B);
    _mlas_fp16_* C_data = reinterpret_cast<_mlas_fp16_*>(C);
    if (CountM == 1) {
        HGemm_TransposedB_Kernel_M1(A_data, B_data, C_data, CountM, CountN, CountK, lda, ldb, ldc, alpha, beta);
    } else {
        HGemm_TransposedB_Kernel_M2(A_data, B_data, C_data, CountM, CountN, CountK, lda, ldb, ldc, alpha, beta);
    }
}

void HGemm_TransposedPackedB_Kernel(
    const MLAS_FP16* A,
    const MLAS_FP16* PackedB,
    MLAS_FP16* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldc,
    MLAS_FP16 alpha,
    MLAS_FP16 beta
) {
    if (CountM > 2) {
        MLAS_THROW_EX(std::runtime_error, "HGemm_TransposedPackedB_Kernel only support <= 2 rows");
    }

    const auto* a = reinterpret_cast<const _mlas_fp16_*>(A);
    const auto* b = reinterpret_cast<const _mlas_fp16_*>(PackedB);
    auto* c = reinterpret_cast<_mlas_fp16_*>(C);

    // 2M_8N as register block. 16 accumulators.
    for (; CountN >= 8; CountN -= 8) {
        if (CountM == 2) {
            HGemmTransB_Kernel_Block<16, 2>(a, b, bias, c, K, lda, ldb, ldc);
        } else {
            HGemmTransB_Kernel_Block<16, 1>(a, b, bias, c, K, lda, ldb, ldc);
        }
        b += 16 * ldb, c += 16;
        if (bias) bias += 16;
    }

    if (CountN & 4) {
        if (CountM == 2) {
            HGemmTransB_Kernel_Block<4, 2>(a, b, bias, c, K, lda, ldb, ldc);
        } else {
            HGemmTransB_Kernel_Block<4, 1>(a, b, bias, c, K, lda, ldb, ldc);
        }
        b += 4 * ldb, c += 4;
        if (bias) bias += 4;
    }

    if (CountN & 2) {
        if (CountM == 2) {
            HGemmTransB_Kernel_Block<2, 2>(a, b, bias, c, K, lda, ldb, ldc);
        } else {
            HGemmTransB_Kernel_Block<2, 1>(a, b, bias, c, K, lda, ldb, ldc);
        }
        b += 2 * ldb, c += 2;
        if (bias) bias += 2;
    }

    if (CountN & 1) {
        if (CountM == 2) {
            HGemmTransB_Kernel_Block<1, 2>(a, b, bias, c, K, lda, ldb, ldc);
        } else {
            HGemmTransB_Kernel_Block<1, 1>(a, b, bias, c, K, lda, ldb, ldc);
        }
    }
}

}  // namespace hgemm_neon
