/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    halfattn_kernel_neon_fhm.cpp

Abstract:

    This module implements the FEAT_FHM variant of the attention products over
    fp16 key/value caches for ARM64: FMLAL/FMLAL2 multiply fp16 query and key
    elements and accumulate in fp32, without widening instructions. The
    products of two fp16 values are exact in fp32, so the scores match the
    widen kernel's.

    The probability-times-value product keeps the widen kernel: FMLAL would
    need the probabilities rounded to fp16.

--*/

#include "halfattn.h"

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)

//
// GCC and Clang expose the FMLAL intrinsics only when the file is compiled for
// FEAT_FHM (+fp16fml, see onnxruntime_mlas.cmake); MSVC always does. Without
// them the dispatch keeps the widen kernel.
//
#if defined(__ARM_FEATURE_FP16_FML) || (defined(_MSC_VER) && !defined(__clang__))
#define MLAS_HALFATTN_FHM_KERNEL
#endif

#if defined(MLAS_HALFATTN_FHM_KERNEL)

#include <arm_neon.h>

#include "fp16_common.h"

namespace halfattn_neon {

namespace {

// Sums the first `count` products A[d] * B[d] in fp32.
MLAS_FORCEINLINE float
ScalarDot(const _mlas_fp16_* A, const _mlas_fp16_* B, size_t count)
{
    float sum = 0.0f;
    for (size_t d = 0; d < count; ++d) {
        sum += MLAS_Half2Float(A[d]) * MLAS_Half2Float(B[d]);
    }
    return sum;
}

//
// C[r, n] = Alpha * sum_d A[r, d] * B[n, d] for MR rows of A and NR rows of B, both fp16.
//
template <size_t MR, size_t NR>
MLAS_FORCEINLINE void
QKBlock(const _mlas_fp16_* A, size_t lda, const _mlas_fp16_* B, size_t ldb, size_t K, float Alpha, float* C,
        size_t ldc)
{
    float32x4_t acc[MR][NR];
    for (size_t r = 0; r < MR; ++r) {
        for (size_t n = 0; n < NR; ++n) {
            acc[r][n] = vdupq_n_f32(0.0f);
        }
    }

    const size_t k8 = K & ~size_t{7};
    for (size_t d = 0; d < k8; d += 8) {
        float16x8_t b[NR];
        for (size_t n = 0; n < NR; ++n) {
            b[n] = MlasLoadFloat16x8(B + n * ldb + d);
        }
        for (size_t r = 0; r < MR; ++r) {
            const float16x8_t a = MlasLoadFloat16x8(A + r * lda + d);
            for (size_t n = 0; n < NR; ++n) {
                acc[r][n] = vfmlalq_low_f16(acc[r][n], a, b[n]);
                acc[r][n] = vfmlalq_high_f16(acc[r][n], a, b[n]);
            }
        }
    }

    for (size_t r = 0; r < MR; ++r) {
        float* c = C + r * ldc;
        if constexpr (NR == 4) {
            float32x4_t sum = vpaddq_f32(vpaddq_f32(acc[r][0], acc[r][1]), vpaddq_f32(acc[r][2], acc[r][3]));
            if (k8 < K) {
                float tail[4];
                for (size_t n = 0; n < 4; ++n) {
                    tail[n] = ScalarDot(A + r * lda + k8, B + n * ldb + k8, K - k8);
                }
                sum = vaddq_f32(sum, vld1q_f32(tail));
            }
            vst1q_f32(c, vmulq_n_f32(sum, Alpha));
        } else {
            for (size_t n = 0; n < NR; ++n) {
                c[n] = Alpha * (vaddvq_f32(acc[r][n]) + ScalarDot(A + r * lda + k8, B + n * ldb + k8, K - k8));
            }
        }
    }
}

template <size_t MR>
void
QKRows(size_t N, size_t K, float Alpha, const _mlas_fp16_* A, size_t lda, const _mlas_fp16_* B, size_t ldb, float* C,
       size_t ldc)
{
    size_t n = 0;
    for (; n + 4 <= N; n += 4) {
        QKBlock<MR, 4>(A, lda, B + n * ldb, ldb, K, Alpha, C + n, ldc);
    }
    for (; n < N; ++n) {
        QKBlock<MR, 1>(A, lda, B + n * ldb, ldb, K, Alpha, C + n, ldc);
    }
}

}  // namespace

void
QK_Kernel_Fhm(size_t M, size_t N, size_t K, float Alpha, const MLAS_FP16* A, size_t lda, const MLAS_FP16* B,
              size_t ldb, float* C, size_t ldc)
{
    const auto* a = reinterpret_cast<const _mlas_fp16_*>(A);
    const auto* b = reinterpret_cast<const _mlas_fp16_*>(B);
    size_t m = 0;
    for (; m + 4 <= M; m += 4) {
        QKRows<4>(N, K, Alpha, a + m * lda, lda, b, ldb, C + m * ldc, ldc);
    }
    switch (M - m) {
        case 3:
            QKRows<3>(N, K, Alpha, a + m * lda, lda, b, ldb, C + m * ldc, ldc);
            break;
        case 2:
            QKRows<2>(N, K, Alpha, a + m * lda, lda, b, ldb, C + m * ldc, ldc);
            break;
        case 1:
            QKRows<1>(N, K, Alpha, a + m * lda, lda, b, ldb, C + m * ldc, ldc);
            break;
        default:
            break;
    }
}

}  // namespace halfattn_neon

#endif  // defined(MLAS_HALFATTN_FHM_KERNEL)

const MLAS_HALF_ATTN_DISPATCH MlasHalfAttnDispatchNeonFhm = []() {
    MLAS_HALF_ATTN_DISPATCH d;
#if defined(MLAS_HALFATTN_FHM_KERNEL)
    d.QK = halfattn_neon::QK_Kernel_Fhm;
#else
    d.QK = halfattn_neon::QK_Kernel_Widen;
#endif
    d.PV = halfattn_neon::PV_Kernel_Widen;
    return d;
}();

#endif  // defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
