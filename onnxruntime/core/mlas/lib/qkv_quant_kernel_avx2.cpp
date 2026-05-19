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
// Dequantize 8 INT8 values from `src` starting at column `col` and multiply
// by scale(s).  Returns an __m256 of 8 FP32 values.
//
inline __m256
DequantInt8x8(const int8_t* src, size_t col, bool per_channel, const float* scales)
{
    // Load 8 int8 values
    __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + col));
    // Sign-extend to 32-bit
    __m256i i32 = _mm256_cvtepi8_epi32(raw);
    // Convert to float
    __m256 f32 = _mm256_cvtepi32_ps(i32);
    // Multiply by scale
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
// Dequantize one row of length `cols` from packed quantized buffer into `dst`.
// Uses AVX2 for the bulk and scalar for the tail.
//
void
DequantRow_Avx2(
    const void* src_raw,
    float* dst,
    size_t cols,
    MLAS_KV_QUANT_TYPE qt,
    const float* scales)
{
    const bool int4 = IsInt4Mode(qt);
    const bool per_channel = IsPerChannelMode(qt);

    size_t c = 0;
    const size_t vec_end = (cols / 8) * 8;

    if (!int4) {
        const auto* src = static_cast<const int8_t*>(src_raw);
        for (; c < vec_end; c += 8) {
            __m256 vals = DequantInt8x8(src, c, per_channel, scales);
            _mm256_storeu_ps(dst + c, vals);
        }
        // Scalar tail
        for (; c < cols; ++c) {
            float sc = per_channel ? scales[c] : scales[0];
            dst[c] = static_cast<float>(src[c]) * sc;
        }
    } else {
        const auto* src = static_cast<const uint8_t*>(src_raw);
        // INT4: 8 elements = 4 bytes, require col aligned to even boundary
        for (; c < vec_end; c += 8) {
            __m256 vals = DequantInt4x8(src, c, per_channel, scales);
            _mm256_storeu_ps(dst + c, vals);
        }
        // Scalar tail
        for (; c < cols; ++c) {
            uint8_t packed = src[c / 2];
            int nibble = (c & 1) == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
            float sc = per_channel ? scales[c] : scales[0];
            dst[c] = static_cast<float>(nibble - kInt4Bias) * sc;
        }
    }
}

//
// QKGemm:  C[M,N] = Alpha * A[M,K] * B^T[K,N]
// B is [N,K] packed row-major.
//
// Strategy: parallelize over N. For each n, dequantize B[n,:] once, then
// compute dot product with each A[m,:] using FMA.
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

    // Temporary buffer for one dequantized B row (K = head_size, typically <= 256).
    // Allocate on stack for small K, heap otherwise.
    float b_stack[256];
    float* b_buf = b_stack;
    std::unique_ptr<float[]> heap_buf;
    if (K > 256) {
        heap_buf.reset(new float[K]);
        b_buf = heap_buf.get();
    }

    for (size_t n = 0; n < N; ++n) {
        const uint8_t* b_row = B_bytes + n * row_bytes;

        // Dequantize B[n,:] into b_buf
        DequantRow_Avx2(b_row, b_buf, K, QuantType, Scales);

        // For each query row m, compute dot(A[m,:], b_buf) * Alpha
        for (size_t m = 0; m < M; ++m) {
            const float* a_row = A + m * lda;

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            size_t k = 0;
            const size_t vec_end = (K / 16) * 16;

            // Process 16 elements per iteration (2x unroll)
            for (; k < vec_end; k += 16) {
                __m256 a0 = _mm256_loadu_ps(a_row + k);
                __m256 b0 = _mm256_loadu_ps(b_buf + k);
                acc0 = _mm256_fmadd_ps(a0, b0, acc0);

                __m256 a1 = _mm256_loadu_ps(a_row + k + 8);
                __m256 b1 = _mm256_loadu_ps(b_buf + k + 8);
                acc1 = _mm256_fmadd_ps(a1, b1, acc1);
            }

            // Process remaining 8-element chunk
            if (k + 8 <= K) {
                __m256 a0 = _mm256_loadu_ps(a_row + k);
                __m256 b0 = _mm256_loadu_ps(b_buf + k);
                acc0 = _mm256_fmadd_ps(a0, b0, acc0);
                k += 8;
            }

            // Combine accumulators
            acc0 = _mm256_add_ps(acc0, acc1);

            // Horizontal sum of acc0
            __m128 lo = _mm256_castps256_ps128(acc0);
            __m128 hi = _mm256_extractf128_ps(acc0, 1);
            __m128 sum4 = _mm_add_ps(lo, hi);
            sum4 = _mm_hadd_ps(sum4, sum4);
            sum4 = _mm_hadd_ps(sum4, sum4);
            float dot = _mm_cvtss_f32(sum4);

            // Scalar tail
            for (; k < K; ++k) {
                dot += a_row[k] * b_buf[k];
            }

            C[m * ldc + n] = Alpha * dot;
        }
    }
}

//
// SVGemm:  C[M,N] = A[M,K] * B[K,N]
// B is [K,N] packed row-major.
//
// Strategy: for each output row m, accumulate contributions from all K rows of B.
// Each B[k,:] is dequantized, multiplied by A[m,k], and added to C[m,:].
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

    // Temporary buffer for one dequantized B row (N = head_size, typically <= 256).
    float b_stack[256];
    float* b_buf = b_stack;
    std::unique_ptr<float[]> heap_buf;
    if (N > 256) {
        heap_buf.reset(new float[N]);
        b_buf = heap_buf.get();
    }

    for (size_t m = 0; m < M; ++m) {
        float* c_row = C + m * ldc;
        const float* a_row = A + m * lda;

        // Zero the output row
        size_t n = 0;
        const size_t vec_end_n = (N / 8) * 8;
        for (; n < vec_end_n; n += 8) {
            _mm256_storeu_ps(c_row + n, _mm256_setzero_ps());
        }
        for (; n < N; ++n) {
            c_row[n] = 0.0f;
        }

        for (size_t k = 0; k < K; ++k) {
            const uint8_t* b_row_packed = B_bytes + k * row_bytes;

            // Dequantize B[k,:]
            DequantRow_Avx2(b_row_packed, b_buf, N, QuantType, Scales);

            // c_row += a_val * b_buf
            const float a_val = a_row[k];
            __m256 a_broadcast = _mm256_broadcast_ss(&a_val);

            n = 0;
            for (; n < vec_end_n; n += 8) {
                __m256 c_vec = _mm256_loadu_ps(c_row + n);
                __m256 b_vec = _mm256_loadu_ps(b_buf + n);
                c_vec = _mm256_fmadd_ps(a_broadcast, b_vec, c_vec);
                _mm256_storeu_ps(c_row + n, c_vec);
            }
            // Scalar tail
            for (; n < N; ++n) {
                c_row[n] += a_val * b_buf[n];
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
