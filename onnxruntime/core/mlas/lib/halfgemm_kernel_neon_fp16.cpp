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
