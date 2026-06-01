/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qkv_quant_kernel_avx512vnni.cpp

Abstract:

    AVX512-VNNI optimized implementation of quantized KV-cache GEMM kernels.

    By default, uses 512-bit wide FP32 FMA (16 floats/cycle, 2x over AVX2's 8)
    to dequantize the quantized KV cache on the fly while preserving FP32 query
    and attention-weight inputs.

    The INT8 per-tensor QKGemm path can optionally use _mm512_dpbusd_epi32 when
    ORT_MLAS_QKGEMM_S8_APPROX_VNNI=1. That path quantizes the FP32 query row on
    the fly and is intentionally opt-in because it changes the numeric contract.

--*/

#include "qkv_quant_kernel.h"
#include "qkv_quant_common.h"
#include "mlas_qkv_quant.h"
#include "core/platform/env_var.h"

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

using namespace MlasKVQuantInternal;

namespace {

inline bool
UseApproximateVnniQKGemm()
{
    static const bool enabled =
        onnxruntime::detail::GetEnvironmentVar("ORT_MLAS_QKGEMM_S8_APPROX_VNNI") == "1";
    return enabled;
}

//
// Quantize an FP32 row to uint8 with zero-point 128 (so signed range maps to [1,255]).
// Returns the scale factor: val = (quant[i] - 128) * scale_a.
// We use zero_point=128 because _mm512_dpbusd_epi32 treats the first operand as unsigned.
//
inline float
QuantizeRowToU8(const float* src, uint8_t* dst, size_t len)
{
    // Find max absolute value using AVX-512
    __m512 max_abs = _mm512_setzero_ps();
    size_t i = 0;
    const size_t vec_end = (len / 16) * 16;

    for (; i < vec_end; i += 16) {
        __m512 v = _mm512_loadu_ps(src + i);
        max_abs = _mm512_max_ps(max_abs, _mm512_abs_ps(v));
    }
    float max_val = _mm512_reduce_max_ps(max_abs);
    for (; i < len; ++i) {
        max_val = std::max(max_val, std::abs(src[i]));
    }

    if (max_val == 0.0f) {
        // All zeros - set everything to zero_point
        std::memset(dst, 128, len);
        return 1.0f;  // arbitrary non-zero scale
    }

    const float scale_a = max_val / 127.0f;
    const float inv_scale = 127.0f / max_val;
    const __m512 inv_scale_vec = _mm512_set1_ps(inv_scale);
    const __m512 zp_vec = _mm512_set1_ps(128.0f);
    const __m512 min_val = _mm512_set1_ps(0.0f);
    const __m512 max_clamp = _mm512_set1_ps(255.0f);

    i = 0;
    for (; i < vec_end; i += 16) {
        __m512 v = _mm512_loadu_ps(src + i);
        // q = (v * inv_scale) + 128, clamped to [0, 255]
        __m512 scaled = _mm512_fmadd_ps(v, inv_scale_vec, zp_vec);
        scaled = _mm512_max_ps(scaled, min_val);
        scaled = _mm512_min_ps(scaled, max_clamp);
        // Round-to-nearest-even and convert to int32 in a single instruction
        // (AVX-512 embedded rounding eliminates a separate vrndscaleps).
        __m512i qi = _mm512_cvt_roundps_epi32(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        // Pack 16 int32 -> 16 uint8
        __m128i packed = _mm512_cvtepi32_epi8(qi);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), packed);
    }
    // Scalar tail (use nearbyintf for round-to-nearest-even, matching the
    // AVX-512 embedded rounding semantics above).
    for (; i < len; ++i) {
        float q = std::nearbyintf(src[i] * inv_scale) + 128.0f;
        q = std::max(0.0f, std::min(255.0f, q));
        dst[i] = static_cast<uint8_t>(q);
    }

    return scale_a;
}

//
// VNNI integer dot product for INT8 per-tensor:
// Compute dot(A_u8[0..K-1] - 128, B_s8[0..K-1]) using _mm512_dpbusd_epi32.
// dpbusd: for each 32-bit lane, accumulates 4 pairs of (u8 * s8) into int32.
//
// Important: dpbusd computes sum(a_u8[j] * b_s8[j]) for j in 0..3.
// Since a_u8 = round(a_fp/scale_a) + 128, the actual value is (a_u8 - 128)*scale_a.
// So: dot(A, dequant(B)) = scale_a * scale_b * [sum(a_u8 * b_s8) - 128 * sum(b_s8)]
//
inline float
VnniDotInt8PerTensor(
    const uint8_t* a_u8,
    const int8_t* b_s8,
    size_t K,
    float scale_a,
    float scale_b)
{
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    __m512i b_sum0 = _mm512_setzero_si512();  // sum of b values (for zero-point correction)
    __m512i b_sum1 = _mm512_setzero_si512();
    const __m512i ones_u8 = _mm512_set1_epi8(1);

    size_t k = 0;
    const size_t vec_end = (K / 128) * 128;
    const size_t vec_end2 = (K / 64) * 64;

    // Main loop: 128 elements per iteration (2x unroll of 64)
    for (; k < vec_end; k += 128) {
        // Chunk 0: 64 bytes
        __m512i a0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a_u8 + k));
        __m512i b0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b_s8 + k));
        acc0 = _mm512_dpbusd_epi32(acc0, a0, b0);
        // Accumulate sum of b_s8 values using dpbusd with all-ones as unsigned operand
        b_sum0 = _mm512_dpbusd_epi32(b_sum0, ones_u8, b0);

        // Chunk 1: next 64 bytes
        __m512i a1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a_u8 + k + 64));
        __m512i b1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b_s8 + k + 64));
        acc1 = _mm512_dpbusd_epi32(acc1, a1, b1);
        b_sum1 = _mm512_dpbusd_epi32(b_sum1, ones_u8, b1);
    }
    // Remainder: 64 elements
    for (; k < vec_end2; k += 64) {
        __m512i a0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a_u8 + k));
        __m512i b0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b_s8 + k));
        acc0 = _mm512_dpbusd_epi32(acc0, a0, b0);
        b_sum0 = _mm512_dpbusd_epi32(b_sum0, ones_u8, b0);
    }

    // Combine accumulators
    acc0 = _mm512_add_epi32(acc0, acc1);
    b_sum0 = _mm512_add_epi32(b_sum0, b_sum1);

    // Horizontal reduce
    int32_t dot_i32 = _mm512_reduce_add_epi32(acc0);
    int32_t b_sum_i32 = _mm512_reduce_add_epi32(b_sum0);

    // Scalar tail (K not multiple of 64)
    for (; k < K; ++k) {
        dot_i32 += static_cast<int32_t>(a_u8[k]) * static_cast<int32_t>(b_s8[k]);
        b_sum_i32 += static_cast<int32_t>(b_s8[k]);
    }

    // Correction: dpbusd computed sum(a_u8 * b_s8).
    // We want sum((a_u8 - 128) * b_s8) = sum(a_u8 * b_s8) - 128 * sum(b_s8)
    // Perform correction in int32 to preserve precision (avoids float rounding
    // when |dot_i32| or |128*b_sum_i32| exceed 2^24).
    int32_t corrected = dot_i32 - (128 * b_sum_i32);

    return static_cast<float>(corrected) * scale_a * scale_b;
}

//
// 512-bit wide FP32 fused dequant-dot for INT8. Processes 16 floats per iteration.
//
inline float
FusedDotInt8_Avx512(
    const float* a_row,
    const int8_t* b_row,
    size_t K,
    bool per_channel,
    const float* scales)
{
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();

    size_t k = 0;
    const size_t vec_end = (K / 32) * 32;

    if (per_channel) {
        for (; k < vec_end; k += 32) {
            // Chunk 0: 16 elements
            __m128i raw0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_row + k));
            __m512i i32_0 = _mm512_cvtepi8_epi32(raw0);
            __m512 bf0 = _mm512_cvtepi32_ps(i32_0);
            __m512 sc0 = _mm512_loadu_ps(scales + k);
            bf0 = _mm512_mul_ps(bf0, sc0);
            __m512 a0 = _mm512_loadu_ps(a_row + k);
            acc0 = _mm512_fmadd_ps(a0, bf0, acc0);

            // Chunk 1: next 16 elements
            __m128i raw1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_row + k + 16));
            __m512i i32_1 = _mm512_cvtepi8_epi32(raw1);
            __m512 bf1 = _mm512_cvtepi32_ps(i32_1);
            __m512 sc1 = _mm512_loadu_ps(scales + k + 16);
            bf1 = _mm512_mul_ps(bf1, sc1);
            __m512 a1 = _mm512_loadu_ps(a_row + k + 16);
            acc1 = _mm512_fmadd_ps(a1, bf1, acc1);
        }
        for (; k + 16 <= K; k += 16) {
            __m128i raw0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_row + k));
            __m512i i32_0 = _mm512_cvtepi8_epi32(raw0);
            __m512 bf0 = _mm512_cvtepi32_ps(i32_0);
            __m512 sc0 = _mm512_loadu_ps(scales + k);
            bf0 = _mm512_mul_ps(bf0, sc0);
            __m512 a0 = _mm512_loadu_ps(a_row + k);
            acc0 = _mm512_fmadd_ps(a0, bf0, acc0);
        }
    } else {
        // Per-tensor: defer scale multiplication until after accumulation.
        // sum(a[k] * b[k] * scale) = scale * sum(a[k] * b[k])
        // This saves one vmulps per 16 elements in the hot loop.
        for (; k < vec_end; k += 32) {
            __m128i raw0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_row + k));
            __m512i i32_0 = _mm512_cvtepi8_epi32(raw0);
            __m512 bf0 = _mm512_cvtepi32_ps(i32_0);
            __m512 a0 = _mm512_loadu_ps(a_row + k);
            acc0 = _mm512_fmadd_ps(a0, bf0, acc0);

            __m128i raw1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_row + k + 16));
            __m512i i32_1 = _mm512_cvtepi8_epi32(raw1);
            __m512 bf1 = _mm512_cvtepi32_ps(i32_1);
            __m512 a1 = _mm512_loadu_ps(a_row + k + 16);
            acc1 = _mm512_fmadd_ps(a1, bf1, acc1);
        }
        for (; k + 16 <= K; k += 16) {
            __m128i raw0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_row + k));
            __m512i i32_0 = _mm512_cvtepi8_epi32(raw0);
            __m512 bf0 = _mm512_cvtepi32_ps(i32_0);
            __m512 a0 = _mm512_loadu_ps(a_row + k);
            acc0 = _mm512_fmadd_ps(a0, bf0, acc0);
        }
    }

    acc0 = _mm512_add_ps(acc0, acc1);
    float dot = _mm512_reduce_add_ps(acc0);

    // Scalar tail
    if (per_channel) {
        for (; k < K; ++k) {
            dot += a_row[k] * static_cast<float>(b_row[k]) * scales[k];
        }
    } else {
        for (; k < K; ++k) {
            dot += a_row[k] * static_cast<float>(b_row[k]);
        }
        dot *= scales[0];
    }
    return dot;
}

//
// 512-bit wide FP32 fused dequant-dot for INT4.
//
inline __m512
DequantInt4x16_Avx512(const uint8_t* src, size_t col, bool per_channel, const float* scales)
{
    // 16 INT4 values occupy 8 bytes
    const uint8_t* base = src + col / 2;

    // Load 8 bytes → 16 nibbles
    __m128i packed = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(base));

    // Expand to 16 bytes: interleave low/high nibbles
    // Each byte has 2 nibbles: low = even index, high = odd index
    __m128i lo_mask = _mm_set1_epi8(0x0F);
    __m128i lo_nibbles = _mm_and_si128(packed, lo_mask);
    __m128i hi_nibbles = _mm_and_si128(_mm_srli_epi16(packed, 4), lo_mask);
    // Interleave: [lo0, hi0, lo1, hi1, ...] → [elem0, elem1, elem2, elem3, ...]
    __m128i interleaved = _mm_unpacklo_epi8(lo_nibbles, hi_nibbles);

    // Sign-extend 8-bit to 32-bit and subtract bias
    __m512i i32 = _mm512_cvtepu8_epi32(interleaved);
    __m512i bias = _mm512_set1_epi32(kInt4Bias);
    i32 = _mm512_sub_epi32(i32, bias);
    __m512 f32 = _mm512_cvtepi32_ps(i32);

    if (per_channel) {
        __m512 sc = _mm512_loadu_ps(scales + col);
        f32 = _mm512_mul_ps(f32, sc);
    } else {
        __m512 sc = _mm512_set1_ps(scales[0]);
        f32 = _mm512_mul_ps(f32, sc);
    }
    return f32;
}

inline float
FusedDotInt4_Avx512(
    const float* a_row,
    const uint8_t* b_row,
    size_t K,
    bool per_channel,
    const float* scales)
{
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();

    size_t k = 0;
    const size_t vec_end = (K / 32) * 32;

    for (; k < vec_end; k += 32) {
        __m512 bf0 = DequantInt4x16_Avx512(b_row, k, per_channel, scales);
        __m512 a0 = _mm512_loadu_ps(a_row + k);
        acc0 = _mm512_fmadd_ps(a0, bf0, acc0);

        __m512 bf1 = DequantInt4x16_Avx512(b_row, k + 16, per_channel, scales);
        __m512 a1 = _mm512_loadu_ps(a_row + k + 16);
        acc1 = _mm512_fmadd_ps(a1, bf1, acc1);
    }
    for (; k + 16 <= K; k += 16) {
        __m512 bf0 = DequantInt4x16_Avx512(b_row, k, per_channel, scales);
        __m512 a0 = _mm512_loadu_ps(a_row + k);
        acc0 = _mm512_fmadd_ps(a0, bf0, acc0);
    }

    acc0 = _mm512_add_ps(acc0, acc1);
    float dot = _mm512_reduce_add_ps(acc0);

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
// VNNI MultiDot4: process 4 B rows simultaneously for better ILP.
// The OoO engine can overlap loads from 4 independent B rows while dpbusd
// instructions execute, giving ~1.2x additional throughput.
//
inline void
VnniMultiDot4Int8PerTensor(
    const uint8_t* a_u8,
    const int8_t* b0,
    const int8_t* b1,
    const int8_t* b2,
    const int8_t* b3,
    size_t K,
    float combined_scale,
    float* out)
{
    __m512i acc_0 = _mm512_setzero_si512(), bsum_0 = _mm512_setzero_si512();
    __m512i acc_1 = _mm512_setzero_si512(), bsum_1 = _mm512_setzero_si512();
    __m512i acc_2 = _mm512_setzero_si512(), bsum_2 = _mm512_setzero_si512();
    __m512i acc_3 = _mm512_setzero_si512(), bsum_3 = _mm512_setzero_si512();
    const __m512i ones = _mm512_set1_epi8(1);

    size_t k = 0;
    for (; k + 64 <= K; k += 64) {
        __m512i av = _mm512_load_si512(reinterpret_cast<const __m512i*>(a_u8 + k));

        __m512i bv0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b0 + k));
        __m512i bv1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b1 + k));
        __m512i bv2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b2 + k));
        __m512i bv3 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b3 + k));

        acc_0 = _mm512_dpbusd_epi32(acc_0, av, bv0);
        acc_1 = _mm512_dpbusd_epi32(acc_1, av, bv1);
        acc_2 = _mm512_dpbusd_epi32(acc_2, av, bv2);
        acc_3 = _mm512_dpbusd_epi32(acc_3, av, bv3);

        bsum_0 = _mm512_dpbusd_epi32(bsum_0, ones, bv0);
        bsum_1 = _mm512_dpbusd_epi32(bsum_1, ones, bv1);
        bsum_2 = _mm512_dpbusd_epi32(bsum_2, ones, bv2);
        bsum_3 = _mm512_dpbusd_epi32(bsum_3, ones, bv3);
    }

    // Reduce vector accumulators
    int32_t dot[4], bs[4];
    dot[0] = _mm512_reduce_add_epi32(acc_0);
    dot[1] = _mm512_reduce_add_epi32(acc_1);
    dot[2] = _mm512_reduce_add_epi32(acc_2);
    dot[3] = _mm512_reduce_add_epi32(acc_3);
    bs[0] = _mm512_reduce_add_epi32(bsum_0);
    bs[1] = _mm512_reduce_add_epi32(bsum_1);
    bs[2] = _mm512_reduce_add_epi32(bsum_2);
    bs[3] = _mm512_reduce_add_epi32(bsum_3);

    // Scalar tail for K not multiple of 64
    for (; k < K; ++k) {
        int32_t av = static_cast<int32_t>(a_u8[k]);
        dot[0] += av * static_cast<int32_t>(b0[k]);
        dot[1] += av * static_cast<int32_t>(b1[k]);
        dot[2] += av * static_cast<int32_t>(b2[k]);
        dot[3] += av * static_cast<int32_t>(b3[k]);
        bs[0] += static_cast<int32_t>(b0[k]);
        bs[1] += static_cast<int32_t>(b1[k]);
        bs[2] += static_cast<int32_t>(b2[k]);
        bs[3] += static_cast<int32_t>(b3[k]);
    }

    // Zero-point correction in int32 for precision (see VnniDotInt8PerTensor).
    out[0] = static_cast<float>(dot[0] - 128 * bs[0]) * combined_scale;
    out[1] = static_cast<float>(dot[1] - 128 * bs[1]) * combined_scale;
    out[2] = static_cast<float>(dot[2] - 128 * bs[2]) * combined_scale;
    out[3] = static_cast<float>(dot[3] - 128 * bs[3]) * combined_scale;
}

// ============================================================================
// QKGemm:  C[M,N] = Alpha * A[M,K] * B^T[K,N]
// B is [N,K] packed row-major.
// ============================================================================

void
QKGemm_Avx512Vnni(
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

    if (!int4 && !per_channel && UseApproximateVnniQKGemm()) {
        // INT8 per-tensor approximate path: opt-in only because it quantizes A.
        const float scale_b = Scales[0];

        // Temp buffer for quantized A row (64-byte aligned for AVX-512)
        alignas(64) uint8_t a_u8[512];
        std::vector<uint8_t> a_u8_heap;
        uint8_t* a_quant = a_u8;
        if (K > 512) {
            a_u8_heap.resize(K + 64);
            a_quant = a_u8_heap.data();
            size_t offset = reinterpret_cast<uintptr_t>(a_quant) % 64;
            if (offset != 0) a_quant += (64 - offset);
        }

        for (size_t m = 0; m < M; ++m) {
            const float* a_row = A + m * lda;
            float scale_a = QuantizeRowToU8(a_row, a_quant, K);
            float combined_scale = Alpha * scale_a * scale_b;

            // Process 4 B rows at a time for better ILP
            size_t n = 0;
            const size_t n4_end = (N / 4) * 4;
            for (; n < n4_end; n += 4) {
                const int8_t* b0 = reinterpret_cast<const int8_t*>(B_bytes + (n + 0) * row_bytes);
                const int8_t* b1 = reinterpret_cast<const int8_t*>(B_bytes + (n + 1) * row_bytes);
                const int8_t* b2 = reinterpret_cast<const int8_t*>(B_bytes + (n + 2) * row_bytes);
                const int8_t* b3 = reinterpret_cast<const int8_t*>(B_bytes + (n + 3) * row_bytes);

                float dots[4];
                VnniMultiDot4Int8PerTensor(a_quant, b0, b1, b2, b3, K, combined_scale, dots);
                C[m * ldc + n + 0] = dots[0];
                C[m * ldc + n + 1] = dots[1];
                C[m * ldc + n + 2] = dots[2];
                C[m * ldc + n + 3] = dots[3];
            }
            // Remainder
            for (; n < N; ++n) {
                const int8_t* b_row = reinterpret_cast<const int8_t*>(B_bytes + n * row_bytes);
                float dot = VnniDotInt8PerTensor(a_quant, b_row, K, scale_a, scale_b);
                C[m * ldc + n] = Alpha * dot;
            }
        }
        return;
    }

    if (!int4) {
        // INT8 per-tensor and per-channel: use 512-bit FP32 FMA path.
        for (size_t n = 0; n < N; ++n) {
            const int8_t* b_row = reinterpret_cast<const int8_t*>(B_bytes + n * row_bytes);
            for (size_t m = 0; m < M; ++m) {
                const float* a_row = A + m * lda;
                float dot = FusedDotInt8_Avx512(a_row, b_row, K, per_channel, Scales);
                C[m * ldc + n] = Alpha * dot;
            }
        }
    } else {
        // INT4: use 512-bit FP32 FMA path
        for (size_t n = 0; n < N; ++n) {
            const uint8_t* b_row = B_bytes + n * row_bytes;
            for (size_t m = 0; m < M; ++m) {
                const float* a_row = A + m * lda;
                float dot = FusedDotInt4_Avx512(a_row, b_row, K, per_channel, Scales);
                C[m * ldc + n] = Alpha * dot;
            }
        }
    }
}

// ============================================================================
// SVGemm:  C[M,N] = Beta * C[M,N] + A[M,K] * B[K,N]
// B is [K,N] packed row-major.
//
// For SVGemm, A is attention weights (FP32) and B is V-cache (quantized).
// The VNNI path is less applicable here because the inner loop iterates over K
// (sequence length) which is large, and we accumulate into C[m,n].
// We use 512-bit wide FP32 FMA for all modes.
// ============================================================================

void
SVGemm_Avx512Vnni(
    size_t M,
    size_t N,
    size_t K,
    const float* A,
    size_t lda,
    const void* B,
    MLAS_KV_QUANT_TYPE QuantType,
    const float* Scales,
    float* C,
    size_t ldc,
    float Beta)
{
    const size_t row_bytes = MlasKVQuantPackedRowBytes(QuantType, N);
    const auto* B_bytes = static_cast<const uint8_t*>(B);
    const bool int4 = IsInt4Mode(QuantType);
    const bool per_channel = IsPerChannelMode(QuantType);

    const size_t vec_end_n = (N / 16) * 16;

    for (size_t m = 0; m < M; ++m) {
        float* c_row = C + m * ldc;
        const float* a_row = A + m * lda;

        // Initialize output
        if (Beta == 0.0f) {
            size_t n = 0;
            for (; n < vec_end_n; n += 16) {
                _mm512_storeu_ps(c_row + n, _mm512_setzero_ps());
            }
            for (; n < N; ++n) {
                c_row[n] = 0.0f;
            }
        } else if (Beta != 1.0f) {
            __m512 beta_vec = _mm512_set1_ps(Beta);
            size_t n = 0;
            for (; n < vec_end_n; n += 16) {
                __m512 c_vec = _mm512_loadu_ps(c_row + n);
                c_vec = _mm512_mul_ps(c_vec, beta_vec);
                _mm512_storeu_ps(c_row + n, c_vec);
            }
            for (; n < N; ++n) {
                c_row[n] *= Beta;
            }
        }

        if (!int4) {
            // INT8 path: 512-bit wide
            if (per_channel) {
                for (size_t k = 0; k < K; ++k) {
                    const int8_t* b_row = reinterpret_cast<const int8_t*>(B_bytes + k * row_bytes);
                    const float a_val = a_row[k];
                    __m512 a_broadcast = _mm512_set1_ps(a_val);

                    size_t n = 0;
                    for (; n < vec_end_n; n += 16) {
                        __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_row + n));
                        __m512i i32 = _mm512_cvtepi8_epi32(raw);
                        __m512 bf = _mm512_cvtepi32_ps(i32);
                        __m512 sc = _mm512_loadu_ps(Scales + n);
                        bf = _mm512_mul_ps(bf, sc);
                        __m512 c_vec = _mm512_loadu_ps(c_row + n);
                        c_vec = _mm512_fmadd_ps(a_broadcast, bf, c_vec);
                        _mm512_storeu_ps(c_row + n, c_vec);
                    }
                    for (; n < N; ++n) {
                        c_row[n] += a_val * static_cast<float>(b_row[n]) * Scales[n];
                    }
                }
            } else {
                // Per-tensor: when Beta==0, accumulate unscaled then scale once at end.
                // When Beta!=0, fold scale into a_val.
                if (Beta == 0.0f) {
                    for (size_t k = 0; k < K; ++k) {
                        const int8_t* b_row = reinterpret_cast<const int8_t*>(B_bytes + k * row_bytes);
                        const float a_val = a_row[k];
                        __m512 a_broadcast = _mm512_set1_ps(a_val);

                        size_t n = 0;
                        for (; n < vec_end_n; n += 16) {
                            __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_row + n));
                            __m512i i32 = _mm512_cvtepi8_epi32(raw);
                            __m512 bf = _mm512_cvtepi32_ps(i32);
                            __m512 c_vec = _mm512_loadu_ps(c_row + n);
                            c_vec = _mm512_fmadd_ps(a_broadcast, bf, c_vec);
                            _mm512_storeu_ps(c_row + n, c_vec);
                        }
                        for (; n < N; ++n) {
                            c_row[n] += a_val * static_cast<float>(b_row[n]);
                        }
                    }

                    __m512 scale_vec = _mm512_set1_ps(Scales[0]);
                    size_t n = 0;
                    for (; n < vec_end_n; n += 16) {
                        __m512 c_vec = _mm512_loadu_ps(c_row + n);
                        c_vec = _mm512_mul_ps(c_vec, scale_vec);
                        _mm512_storeu_ps(c_row + n, c_vec);
                    }
                    for (; n < N; ++n) {
                        c_row[n] *= Scales[0];
                    }
                } else {
                    // Beta!=0: fold scale into a_val
                    const float tensor_scale = Scales[0];
                    for (size_t k = 0; k < K; ++k) {
                        const int8_t* b_row = reinterpret_cast<const int8_t*>(B_bytes + k * row_bytes);
                        const float a_val = a_row[k] * tensor_scale;
                        __m512 a_broadcast = _mm512_set1_ps(a_val);

                        size_t n = 0;
                        for (; n < vec_end_n; n += 16) {
                            __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_row + n));
                            __m512i i32 = _mm512_cvtepi8_epi32(raw);
                            __m512 bf = _mm512_cvtepi32_ps(i32);
                            __m512 c_vec = _mm512_loadu_ps(c_row + n);
                            c_vec = _mm512_fmadd_ps(a_broadcast, bf, c_vec);
                            _mm512_storeu_ps(c_row + n, c_vec);
                        }
                        for (; n < N; ++n) {
                            c_row[n] += a_val * static_cast<float>(b_row[n]);
                        }
                    }
                }
            }
        } else {
            // INT4 path: 512-bit wide
            for (size_t k = 0; k < K; ++k) {
                const uint8_t* b_row = B_bytes + k * row_bytes;
                const float a_val = a_row[k];
                __m512 a_broadcast = _mm512_set1_ps(a_val);

                size_t n = 0;
                for (; n < vec_end_n; n += 16) {
                    __m512 bf = DequantInt4x16_Avx512(b_row, n, per_channel, Scales);
                    __m512 c_vec = _mm512_loadu_ps(c_row + n);
                    c_vec = _mm512_fmadd_ps(a_broadcast, bf, c_vec);
                    _mm512_storeu_ps(c_row + n, c_vec);
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

const MLAS_KV_QUANT_GEMM_DISPATCH MlasKVQuantGemmDispatchAvx512Vnni = []() {
    MLAS_KV_QUANT_GEMM_DISPATCH d;
    d.QKGemm = QKGemm_Avx512Vnni;
    d.SVGemm = SVGemm_Avx512Vnni;
    return d;
}();
