/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    halfattn_kernel_neon_fp16.cpp

Abstract:

    This module implements the attention products over fp16 key/value caches
    for ARM64 that read the cache in fp16 and accumulate in fp32: cache rows are
    widened in registers (FCVTL/FCVTL2) and multiplied with fp32 FMLA, so no
    fp32 copy of the cache is written.

--*/

#include "halfattn.h"

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)

#include <arm_neon.h>

#include <algorithm>

#include "fp16_common.h"

namespace halfattn_neon {

namespace {

// Rows of A per register block.
constexpr size_t kRowBlock = 4;
// Columns of A widened to fp32 at a time by QK_Kernel_Widen.
constexpr size_t kQKChunk = 256;

MLAS_FORCEINLINE float32x4_t
WidenLow(float16x8_t v)
{
    return vcvt_f32_f16(vget_low_f16(v));
}

MLAS_FORCEINLINE float32x4_t
WidenHigh(float16x8_t v)
{
    return vcvt_high_f32_f16(v);
}

// Widens `count` fp16 elements to fp32.
void
WidenRow(const _mlas_fp16_* Source, float* Destination, size_t count)
{
    size_t d = 0;
    for (; d + 8 <= count; d += 8) {
        const float16x8_t v = MlasLoadFloat16x8(Source + d);
        vst1q_f32(Destination + d, WidenLow(v));
        vst1q_f32(Destination + d + 4, WidenHigh(v));
    }
    for (; d < count; ++d) {
        Destination[d] = MLAS_Half2Float(Source[d]);
    }
}

// Sums the first `count` products A[d] * B[d] in fp32.
MLAS_FORCEINLINE float
ScalarDot(const float* A, const _mlas_fp16_* B, size_t count)
{
    float sum = 0.0f;
    for (size_t d = 0; d < count; ++d) {
        sum += A[d] * MLAS_Half2Float(B[d]);
    }
    return sum;
}

//
// C[r, n] (=|+=) Alpha * sum_d A[r, d] * B[n, d] for MR rows of fp32 A and NR rows of fp16 B.
//
template <size_t MR, size_t NR>
MLAS_FORCEINLINE void
QKBlock(const float* A, size_t lda, const _mlas_fp16_* B, size_t ldb, size_t K, float Alpha, bool Accumulate,
        float* C, size_t ldc)
{
    float32x4_t acc[MR][NR];
    for (size_t r = 0; r < MR; ++r) {
        for (size_t n = 0; n < NR; ++n) {
            acc[r][n] = vdupq_n_f32(0.0f);
        }
    }

    const size_t k8 = K & ~size_t{7};
    for (size_t d = 0; d < k8; d += 8) {
        float32x4_t bl[NR];
        float32x4_t bh[NR];
        for (size_t n = 0; n < NR; ++n) {
            const float16x8_t b = MlasLoadFloat16x8(B + n * ldb + d);
            bl[n] = WidenLow(b);
            bh[n] = WidenHigh(b);
        }
        for (size_t r = 0; r < MR; ++r) {
            const float32x4_t al = vld1q_f32(A + r * lda + d);
            const float32x4_t ah = vld1q_f32(A + r * lda + d + 4);
            for (size_t n = 0; n < NR; ++n) {
                acc[r][n] = vfmaq_f32(acc[r][n], al, bl[n]);
                acc[r][n] = vfmaq_f32(acc[r][n], ah, bh[n]);
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
            sum = vmulq_n_f32(sum, Alpha);
            vst1q_f32(c, Accumulate ? vaddq_f32(vld1q_f32(c), sum) : sum);
        } else {
            for (size_t n = 0; n < NR; ++n) {
                const float sum = Alpha * (vaddvq_f32(acc[r][n]) + ScalarDot(A + r * lda + k8, B + n * ldb + k8, K - k8));
                c[n] = Accumulate ? c[n] + sum : sum;
            }
        }
    }
}

template <size_t MR>
void
QKRows(size_t N, size_t K, float Alpha, const float* A, size_t lda, const _mlas_fp16_* B, size_t ldb, bool Accumulate,
       float* C, size_t ldc)
{
    size_t n = 0;
    for (; n + 4 <= N; n += 4) {
        QKBlock<MR, 4>(A, lda, B + n * ldb, ldb, K, Alpha, Accumulate, C + n, ldc);
    }
    for (; n < N; ++n) {
        QKBlock<MR, 1>(A, lda, B + n * ldb, ldb, K, Alpha, Accumulate, C + n, ldc);
    }
}

//
// Loads NV vectors (4 * NV elements) of an fp16 row and widens them to fp32.
//
template <size_t NV>
MLAS_FORCEINLINE void
LoadWiden(const _mlas_fp16_* B, float32x4_t* w)
{
    if constexpr (NV == 1) {
        w[0] = vcvt_f32_f16(MlasLoadFloat16x4(B));
    } else {
        for (size_t i = 0; i < NV / 2; ++i) {
            const float16x8_t v = MlasLoadFloat16x8(B + 8 * i);
            w[2 * i] = WidenLow(v);
            w[2 * i + 1] = WidenHigh(v);
        }
    }
}

template <int Lane, size_t MR, size_t NV>
MLAS_FORCEINLINE void
PVStep(const _mlas_fp16_* B, const float32x4_t* p, float32x4_t (*acc)[NV])
{
    float32x4_t w[NV];
    LoadWiden<NV>(B, w);
    for (size_t r = 0; r < MR; ++r) {
        for (size_t i = 0; i < NV; ++i) {
            acc[r][i] = vfmaq_laneq_f32(acc[r][i], w[i], p[r], Lane);
        }
    }
}

//
// C[r, 0:4 * NV] += sum_k A[r, k] * B[k, 0:4 * NV] for MR rows of fp32 A and fp16 rows of B.
//
template <size_t MR, size_t NV>
MLAS_FORCEINLINE void
PVBlock(size_t K, const float* A, size_t lda, const _mlas_fp16_* B, size_t ldb, float* C, size_t ldc)
{
    float32x4_t acc[MR][NV];
    for (size_t r = 0; r < MR; ++r) {
        for (size_t i = 0; i < NV; ++i) {
            acc[r][i] = vld1q_f32(C + r * ldc + 4 * i);
        }
    }

    size_t k = 0;
    for (; k + 4 <= K; k += 4) {
        float32x4_t p[MR];
        for (size_t r = 0; r < MR; ++r) {
            p[r] = vld1q_f32(A + r * lda + k);
        }
        PVStep<0, MR, NV>(B + (k + 0) * ldb, p, acc);
        PVStep<1, MR, NV>(B + (k + 1) * ldb, p, acc);
        PVStep<2, MR, NV>(B + (k + 2) * ldb, p, acc);
        PVStep<3, MR, NV>(B + (k + 3) * ldb, p, acc);
    }
    for (; k < K; ++k) {
        float32x4_t w[NV];
        LoadWiden<NV>(B + k * ldb, w);
        for (size_t r = 0; r < MR; ++r) {
            const float a = A[r * lda + k];
            for (size_t i = 0; i < NV; ++i) {
                acc[r][i] = vfmaq_n_f32(acc[r][i], w[i], a);
            }
        }
    }

    for (size_t r = 0; r < MR; ++r) {
        for (size_t i = 0; i < NV; ++i) {
            vst1q_f32(C + r * ldc + 4 * i, acc[r][i]);
        }
    }
}

template <size_t MR>
void
PVRows(size_t N, size_t K, const float* A, size_t lda, const _mlas_fp16_* B, size_t ldb, float* C, size_t ldc)
{
    size_t n = 0;
    for (; n + 16 <= N; n += 16) {
        PVBlock<MR, 4>(K, A, lda, B + n, ldb, C + n, ldc);
    }
    if (n + 8 <= N) {
        PVBlock<MR, 2>(K, A, lda, B + n, ldb, C + n, ldc);
        n += 8;
    }
    if (n + 4 <= N) {
        PVBlock<MR, 1>(K, A, lda, B + n, ldb, C + n, ldc);
        n += 4;
    }
    for (; n < N; ++n) {
        for (size_t r = 0; r < MR; ++r) {
            float sum = C[r * ldc + n];
            for (size_t k = 0; k < K; ++k) {
                sum += A[r * lda + k] * MLAS_Half2Float(B[k * ldb + n]);
            }
            C[r * ldc + n] = sum;
        }
    }
}

}  // namespace

void
QK_Kernel_Widen(size_t M, size_t N, size_t K, float Alpha, const MLAS_FP16* A, size_t lda, const MLAS_FP16* B,
                size_t ldb, float* C, size_t ldc)
{
    if (K == 0) {
        for (size_t m = 0; m < M; ++m) {
            std::fill_n(C + m * ldc, N, 0.0f);
        }
        return;
    }

    const auto* a = reinterpret_cast<const _mlas_fp16_*>(A);
    const auto* b = reinterpret_cast<const _mlas_fp16_*>(B);

    // A block of rows of A is widened once per chunk of columns and then multiplied against every row of B.
    MLAS_DECLSPEC_ALIGN(float widened[kRowBlock * kQKChunk], 64);

    for (size_t m = 0; m < M; m += kRowBlock) {
        const size_t rows = std::min(kRowBlock, M - m);
        float* c = C + m * ldc;
        for (size_t k0 = 0; k0 < K; k0 += kQKChunk) {
            const size_t count = std::min(kQKChunk, K - k0);
            for (size_t r = 0; r < rows; ++r) {
                WidenRow(a + (m + r) * lda + k0, widened + r * count, count);
            }
            const bool accumulate = k0 != 0;
            const _mlas_fp16_* b_chunk = b + k0;
            switch (rows) {
                case 4:
                    QKRows<4>(N, count, Alpha, widened, count, b_chunk, ldb, accumulate, c, ldc);
                    break;
                case 3:
                    QKRows<3>(N, count, Alpha, widened, count, b_chunk, ldb, accumulate, c, ldc);
                    break;
                case 2:
                    QKRows<2>(N, count, Alpha, widened, count, b_chunk, ldb, accumulate, c, ldc);
                    break;
                default:
                    QKRows<1>(N, count, Alpha, widened, count, b_chunk, ldb, accumulate, c, ldc);
                    break;
            }
        }
    }
}

void
PV_Kernel_Widen(size_t M, size_t N, size_t K, const float* A, size_t lda, const MLAS_FP16* B, size_t ldb, float* C,
                size_t ldc)
{
    const auto* b = reinterpret_cast<const _mlas_fp16_*>(B);
    for (size_t m = 0; m < M; m += kRowBlock) {
        const float* a = A + m * lda;
        float* c = C + m * ldc;
        switch (std::min(kRowBlock, M - m)) {
            case 4:
                PVRows<4>(N, K, a, lda, b, ldb, c, ldc);
                break;
            case 3:
                PVRows<3>(N, K, a, lda, b, ldb, c, ldc);
                break;
            case 2:
                PVRows<2>(N, K, a, lda, b, ldb, c, ldc);
                break;
            default:
                PVRows<1>(N, K, a, lda, b, ldb, c, ldc);
                break;
        }
    }
}

}  // namespace halfattn_neon

const MLAS_HALF_ATTN_DISPATCH MlasHalfAttnDispatchNeon = []() {
    MLAS_HALF_ATTN_DISPATCH d;
    d.QK = halfattn_neon::QK_Kernel_Widen;
    d.PV = halfattn_neon::PV_Kernel_Widen;
    return d;
}();

#endif  // defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && defined(MLAS_TARGET_ARM64)
