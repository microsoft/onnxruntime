/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qkv_quant_kernel_avx2.cpp

Abstract:

    AVX2+FMA3 optimized implementation of quantized KV-cache GEMM kernels
    for MlasQKGemm and MlasSVGemm. Dequantizes INT8/INT4 B on the fly and
    accumulates in FP32 using 256-bit vectors.

--*/

#include "qkv_quant_kernel.h"
#include "mlas_qkv_quant.h"

#include <immintrin.h>

namespace {

constexpr int kInt4Bias = 8;

inline bool
IsInt4Mode(MLAS_KV_QUANT_TYPE qt)
{
    return qt == MLAS_KV_QUANT_TYPE::S4_PerTensor ||
           qt == MLAS_KV_QUANT_TYPE::S4_PerChannel;
}

inline bool
IsPerChannelMode(MLAS_KV_QUANT_TYPE qt)
{
    return qt == MLAS_KV_QUANT_TYPE::S8_PerChannel ||
           qt == MLAS_KV_QUANT_TYPE::S4_PerChannel;
}

//
// Dequantize 8 INT4 values (4 packed bytes) starting at even column `col`.
// The +8-biased nibble packing is: byte = ((q0+8)&0xF) | (((q1+8)&0xF)<<4).
// We extract low/high nibbles, subtract 8, convert to FP32, and scale.
//
inline __m256
DequantInt4x8(const uint8_t* src, size_t col, bool per_channel, const float* scales)
{
    // Each byte holds 2 elements: low nibble = even col, high nibble = odd col.
    // For 8 elements starting at `col`, we need 4 bytes (cols col..col+7 → bytes col/2..col/2+3).
    const uint8_t* base = src + col / 2;

    // Extract 8 nibbles from 4 bytes, subtract bias, and convert to FP32.
    alignas(16) int8_t nibbles[8];
    nibbles[0] = static_cast<int8_t>((base[0] & 0x0F) - kInt4Bias);
    nibbles[1] = static_cast<int8_t>(((base[0] >> 4) & 0x0F) - kInt4Bias);
    nibbles[2] = static_cast<int8_t>((base[1] & 0x0F) - kInt4Bias);
    nibbles[3] = static_cast<int8_t>(((base[1] >> 4) & 0x0F) - kInt4Bias);
    nibbles[4] = static_cast<int8_t>((base[2] & 0x0F) - kInt4Bias);
    nibbles[5] = static_cast<int8_t>(((base[2] >> 4) & 0x0F) - kInt4Bias);
    nibbles[6] = static_cast<int8_t>((base[3] & 0x0F) - kInt4Bias);
    nibbles[7] = static_cast<int8_t>(((base[3] >> 4) & 0x0F) - kInt4Bias);

    __m128i nib128 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(nibbles));
    __m256i i32 = _mm256_cvtepi8_epi32(nib128);
    __m256 f32 = _mm256_cvtepi32_ps(i32);

    if (per_channel) {
        __m256 sc = _mm256_loadu_ps(scales + col);
        f32 = _mm256_mul_ps(f32, sc);
    } else {
        __m256 sc = _mm256_broadcast_ss(scales);
        f32 = _mm256_mul_ps(f32, sc);
    }
    return f32;
}

//
// Fused dequant-dot: dequantize B[n,:] directly into FMA accumulators without
// storing to an intermediate FP32 buffer. This saves one store+reload round-trip
// per B row and keeps the dequantized values in registers.
//

// Fused dot product for INT8 B row against FP32 A row.
// Returns dot(A[0..K-1], dequant(B_row[0..K-1])).
inline float
FusedDotInt8(
    const float* a_row,
    const int8_t* b_row,
    size_t K,
    bool per_channel,
    const float* scales)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    size_t k = 0;
    const size_t vec_end = (K / 16) * 16;

    if (per_channel) {
        for (; k < vec_end; k += 16) {
            // Chunk 0
            __m128i raw0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b_row + k));
            __m256i i32_0 = _mm256_cvtepi8_epi32(raw0);
            __m256 bf0 = _mm256_cvtepi32_ps(i32_0);
            __m256 sc0 = _mm256_loadu_ps(scales + k);
            bf0 = _mm256_mul_ps(bf0, sc0);
            __m256 a0 = _mm256_loadu_ps(a_row + k);
            acc0 = _mm256_fmadd_ps(a0, bf0, acc0);

            // Chunk 1
            __m128i raw1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b_row + k + 8));
            __m256i i32_1 = _mm256_cvtepi8_epi32(raw1);
            __m256 bf1 = _mm256_cvtepi32_ps(i32_1);
            __m256 sc1 = _mm256_loadu_ps(scales + k + 8);
            bf1 = _mm256_mul_ps(bf1, sc1);
            __m256 a1 = _mm256_loadu_ps(a_row + k + 8);
            acc1 = _mm256_fmadd_ps(a1, bf1, acc1);
        }
        for (; k + 8 <= K; k += 8) {
            __m128i raw0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b_row + k));
            __m256i i32_0 = _mm256_cvtepi8_epi32(raw0);
            __m256 bf0 = _mm256_cvtepi32_ps(i32_0);
            __m256 sc0 = _mm256_loadu_ps(scales + k);
            bf0 = _mm256_mul_ps(bf0, sc0);
            __m256 a0 = _mm256_loadu_ps(a_row + k);
            acc0 = _mm256_fmadd_ps(a0, bf0, acc0);
        }
    } else {
        __m256 scale_vec = _mm256_broadcast_ss(scales);
        for (; k < vec_end; k += 16) {
            __m128i raw0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b_row + k));
            __m256i i32_0 = _mm256_cvtepi8_epi32(raw0);
            __m256 bf0 = _mm256_cvtepi32_ps(i32_0);
            bf0 = _mm256_mul_ps(bf0, scale_vec);
            __m256 a0 = _mm256_loadu_ps(a_row + k);
            acc0 = _mm256_fmadd_ps(a0, bf0, acc0);

            __m128i raw1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b_row + k + 8));
            __m256i i32_1 = _mm256_cvtepi8_epi32(raw1);
            __m256 bf1 = _mm256_cvtepi32_ps(i32_1);
            bf1 = _mm256_mul_ps(bf1, scale_vec);
            __m256 a1 = _mm256_loadu_ps(a_row + k + 8);
            acc1 = _mm256_fmadd_ps(a1, bf1, acc1);
        }
        for (; k + 8 <= K; k += 8) {
            __m128i raw0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b_row + k));
            __m256i i32_0 = _mm256_cvtepi8_epi32(raw0);
            __m256 bf0 = _mm256_cvtepi32_ps(i32_0);
            bf0 = _mm256_mul_ps(bf0, scale_vec);
            __m256 a0 = _mm256_loadu_ps(a_row + k);
            acc0 = _mm256_fmadd_ps(a0, bf0, acc0);
        }
    }

    acc0 = _mm256_add_ps(acc0, acc1);
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    float dot = _mm_cvtss_f32(sum4);

    // Scalar tail
    for (; k < K; ++k) {
        float sc = per_channel ? scales[k] : scales[0];
        dot += a_row[k] * static_cast<float>(b_row[k]) * sc;
    }
    return dot;
}

// Fused dot product for INT4 B row against FP32 A row.
inline float
FusedDotInt4(
    const float* a_row,
    const uint8_t* b_row,
    size_t K,
    bool per_channel,
    const float* scales)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    size_t k = 0;
    const size_t vec_end = (K / 16) * 16;

    for (; k < vec_end; k += 16) {
        // Chunk 0: 8 elements from 4 packed bytes
        __m256 bf0 = DequantInt4x8(b_row, k, per_channel, scales);
        __m256 a0 = _mm256_loadu_ps(a_row + k);
        acc0 = _mm256_fmadd_ps(a0, bf0, acc0);

        // Chunk 1: next 8 elements
        __m256 bf1 = DequantInt4x8(b_row, k + 8, per_channel, scales);
        __m256 a1 = _mm256_loadu_ps(a_row + k + 8);
        acc1 = _mm256_fmadd_ps(a1, bf1, acc1);
    }
    for (; k + 8 <= K; k += 8) {
        __m256 bf0 = DequantInt4x8(b_row, k, per_channel, scales);
        __m256 a0 = _mm256_loadu_ps(a_row + k);
        acc0 = _mm256_fmadd_ps(a0, bf0, acc0);
    }

    acc0 = _mm256_add_ps(acc0, acc1);
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    float dot = _mm_cvtss_f32(sum4);

    // Scalar tail
    for (; k < K; ++k) {
        uint8_t packed = b_row[k / 2];
        int nibble = (k & 1) == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
        float sc = per_channel ? scales[k] : scales[0];
        dot += a_row[k] * static_cast<float>(nibble - kInt4Bias) * sc;
    }
    return dot;
}

//
// QKGemm:  C[M,N] = Alpha * A[M,K] * B^T[K,N]
// B is [N,K] packed row-major.
//
// Fused approach: dequantize B directly into FMA accumulators without
// intermediate buffer. For M>1, B row is re-dequantized per query row
// (still faster due to no store/reload and B being in L1 cache).
//
void
QKGemm_Avx2(
    size_t M,
    size_t N,
    size_t K,
    float Alpha,
    const float* A,
    size_t lda,
    const void* B,
    MLAS_KV_QUANT_TYPE QuantType,
    const float* Scales,
    float* C,
    size_t ldc)
{
    const size_t row_bytes = MlasKVQuantPackedRowBytes(QuantType, K);
    const auto* B_bytes = static_cast<const uint8_t*>(B);
    const bool int4 = IsInt4Mode(QuantType);
    const bool per_channel = IsPerChannelMode(QuantType);

    for (size_t n = 0; n < N; ++n) {
        const uint8_t* b_row = B_bytes + n * row_bytes;

        for (size_t m = 0; m < M; ++m) {
            const float* a_row = A + m * lda;
            float dot;
            if (int4) {
                dot = FusedDotInt4(a_row, b_row, K, per_channel, Scales);
            } else {
                dot = FusedDotInt8(a_row, reinterpret_cast<const int8_t*>(b_row),
                                   K, per_channel, Scales);
            }
            C[m * ldc + n] = Alpha * dot;
        }
    }
}

//
// SVGemm:  C[M,N] = A[M,K] * B[K,N]
// B is [K,N] packed row-major.
//
// Fused approach: dequantize each B[k,:] element directly into the FMA with
// the C accumulator, eliminating the intermediate buffer entirely.
//
void
SVGemm_Avx2(
    size_t M,
    size_t N,
    size_t K,
    const float* A,
    size_t lda,
    const void* B,
    MLAS_KV_QUANT_TYPE QuantType,
    const float* Scales,
    float* C,
    size_t ldc)
{
    const size_t row_bytes = MlasKVQuantPackedRowBytes(QuantType, N);
    const auto* B_bytes = static_cast<const uint8_t*>(B);
    const bool int4 = IsInt4Mode(QuantType);
    const bool per_channel = IsPerChannelMode(QuantType);

    const size_t vec_end_n = (N / 8) * 8;

    for (size_t m = 0; m < M; ++m) {
        float* c_row = C + m * ldc;
        const float* a_row = A + m * lda;

        // Zero output
        size_t n = 0;
        for (; n < vec_end_n; n += 8) {
            _mm256_storeu_ps(c_row + n, _mm256_setzero_ps());
        }
        for (; n < N; ++n) {
            c_row[n] = 0.0f;
        }

        if (!int4) {
            // INT8 fused path
            if (per_channel) {
                for (size_t k = 0; k < K; ++k) {
                    const int8_t* b_row = reinterpret_cast<const int8_t*>(B_bytes + k * row_bytes);
                    const float a_val = a_row[k];
                    __m256 a_broadcast = _mm256_broadcast_ss(&a_val);

                    n = 0;
                    for (; n < vec_end_n; n += 8) {
                        __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b_row + n));
                        __m256i i32 = _mm256_cvtepi8_epi32(raw);
                        __m256 bf = _mm256_cvtepi32_ps(i32);
                        __m256 sc = _mm256_loadu_ps(Scales + n);
                        bf = _mm256_mul_ps(bf, sc);
                        __m256 c_vec = _mm256_loadu_ps(c_row + n);
                        c_vec = _mm256_fmadd_ps(a_broadcast, bf, c_vec);
                        _mm256_storeu_ps(c_row + n, c_vec);
                    }
                    for (; n < N; ++n) {
                        c_row[n] += a_val * static_cast<float>(b_row[n]) * Scales[n];
                    }
                }
            } else {
                __m256 scale_vec = _mm256_broadcast_ss(Scales);
                for (size_t k = 0; k < K; ++k) {
                    const int8_t* b_row = reinterpret_cast<const int8_t*>(B_bytes + k * row_bytes);
                    const float a_val = a_row[k];
                    __m256 a_broadcast = _mm256_broadcast_ss(&a_val);

                    n = 0;
                    for (; n < vec_end_n; n += 8) {
                        __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b_row + n));
                        __m256i i32 = _mm256_cvtepi8_epi32(raw);
                        __m256 bf = _mm256_cvtepi32_ps(i32);
                        bf = _mm256_mul_ps(bf, scale_vec);
                        __m256 c_vec = _mm256_loadu_ps(c_row + n);
                        c_vec = _mm256_fmadd_ps(a_broadcast, bf, c_vec);
                        _mm256_storeu_ps(c_row + n, c_vec);
                    }
                    for (; n < N; ++n) {
                        c_row[n] += a_val * static_cast<float>(b_row[n]) * Scales[0];
                    }
                }
            }
        } else {
            // INT4 fused path
            for (size_t k = 0; k < K; ++k) {
                const uint8_t* b_row = B_bytes + k * row_bytes;
                const float a_val = a_row[k];
                __m256 a_broadcast = _mm256_broadcast_ss(&a_val);

                n = 0;
                for (; n < vec_end_n; n += 8) {
                    __m256 bf = DequantInt4x8(b_row, n, per_channel, Scales);
                    __m256 c_vec = _mm256_loadu_ps(c_row + n);
                    c_vec = _mm256_fmadd_ps(a_broadcast, bf, c_vec);
                    _mm256_storeu_ps(c_row + n, c_vec);
                }
                for (; n < N; ++n) {
                    uint8_t packed = b_row[n / 2];
                    int nibble = (n & 1) == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
                    float sc = per_channel ? Scales[n] : Scales[0];
                    c_row[n] += a_val * static_cast<float>(nibble - kInt4Bias) * sc;
                }
            }
        }
    }
}

}  // namespace

const MLAS_KV_QUANT_GEMM_DISPATCH MlasKVQuantGemmDispatchAvx2 = []() {
    MLAS_KV_QUANT_GEMM_DISPATCH d;
    d.QKGemm = QKGemm_Avx2;
    d.SVGemm = SVGemm_Avx2;
    return d;
}();
