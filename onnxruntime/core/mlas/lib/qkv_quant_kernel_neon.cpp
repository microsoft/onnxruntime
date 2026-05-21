/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qkv_quant_kernel_neon.cpp

Abstract:

    ARM NEON optimized implementation of quantized KV-cache GEMM kernels
    for MlasQKGemm and MlasSVGemm. Dequantizes INT8/INT4 B on the fly and
    accumulates in FP32 using 128-bit NEON vectors.

--*/

#include "qkv_quant_kernel.h"
#include "qkv_quant_common.h"
#include "mlas_qkv_quant.h"

#include <arm_neon.h>
#include <cstring>
#include <memory>

using namespace MlasKVQuantInternal;

namespace {

//
// Dequantize 8 INT8 values starting at `col` and scale them.
// Produces two float32x4_t (8 floats total) stored into dst.
//
inline void
DequantInt8x8_Neon(const int8_t* src, size_t col, bool per_channel,
                   const float* scales, float* dst)
{
    // Load 8 int8 values
    int8x8_t raw = vld1_s8(src + col);
    // Widen to int16
    int16x8_t i16 = vmovl_s8(raw);
    // Widen to int32 (low and high halves)
    int32x4_t i32_lo = vmovl_s16(vget_low_s16(i16));
    int32x4_t i32_hi = vmovl_s16(vget_high_s16(i16));
    // Convert to float
    float32x4_t f_lo = vcvtq_f32_s32(i32_lo);
    float32x4_t f_hi = vcvtq_f32_s32(i32_hi);

    if (per_channel) {
        float32x4_t sc_lo = vld1q_f32(scales + col);
        float32x4_t sc_hi = vld1q_f32(scales + col + 4);
        f_lo = vmulq_f32(f_lo, sc_lo);
        f_hi = vmulq_f32(f_hi, sc_hi);
    } else {
        float32x4_t sc = vdupq_n_f32(scales[0]);
        f_lo = vmulq_f32(f_lo, sc);
        f_hi = vmulq_f32(f_hi, sc);
    }

    vst1q_f32(dst, f_lo);
    vst1q_f32(dst + 4, f_hi);
}

//
// Dequantize 8 INT4 values (4 packed bytes) starting at even column `col`.
//
inline void
DequantInt4x8_Neon(const uint8_t* src, size_t col, bool per_channel,
                   const float* scales, float* dst)
{
    const uint8_t* base = src + col / 2;

    // Extract 8 nibbles from 4 bytes
    alignas(8) int8_t nibbles[8];
    nibbles[0] = static_cast<int8_t>((base[0] & 0x0F) - kInt4Bias);
    nibbles[1] = static_cast<int8_t>(((base[0] >> 4) & 0x0F) - kInt4Bias);
    nibbles[2] = static_cast<int8_t>((base[1] & 0x0F) - kInt4Bias);
    nibbles[3] = static_cast<int8_t>(((base[1] >> 4) & 0x0F) - kInt4Bias);
    nibbles[4] = static_cast<int8_t>((base[2] & 0x0F) - kInt4Bias);
    nibbles[5] = static_cast<int8_t>(((base[2] >> 4) & 0x0F) - kInt4Bias);
    nibbles[6] = static_cast<int8_t>((base[3] & 0x0F) - kInt4Bias);
    nibbles[7] = static_cast<int8_t>(((base[3] >> 4) & 0x0F) - kInt4Bias);

    int8x8_t raw = vld1_s8(nibbles);
    int16x8_t i16 = vmovl_s8(raw);
    int32x4_t i32_lo = vmovl_s16(vget_low_s16(i16));
    int32x4_t i32_hi = vmovl_s16(vget_high_s16(i16));
    float32x4_t f_lo = vcvtq_f32_s32(i32_lo);
    float32x4_t f_hi = vcvtq_f32_s32(i32_hi);

    if (per_channel) {
        float32x4_t sc_lo = vld1q_f32(scales + col);
        float32x4_t sc_hi = vld1q_f32(scales + col + 4);
        f_lo = vmulq_f32(f_lo, sc_lo);
        f_hi = vmulq_f32(f_hi, sc_hi);
    } else {
        float32x4_t sc = vdupq_n_f32(scales[0]);
        f_lo = vmulq_f32(f_lo, sc);
        f_hi = vmulq_f32(f_hi, sc);
    }

    vst1q_f32(dst, f_lo);
    vst1q_f32(dst + 4, f_hi);
}

//
// Dequantize one row of length `cols` from packed quantized buffer into `dst`.
//
void
DequantRow_Neon(
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
            DequantInt8x8_Neon(src, c, per_channel, scales, dst + c);
        }
        for (; c < cols; ++c) {
            float sc = per_channel ? scales[c] : scales[0];
            dst[c] = static_cast<float>(src[c]) * sc;
        }
    } else {
        const auto* src = static_cast<const uint8_t*>(src_raw);
        for (; c < vec_end; c += 8) {
            DequantInt4x8_Neon(src, c, per_channel, scales, dst + c);
        }
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
//
void
QKGemm_Neon(
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

    float b_stack[256];
    float* b_buf = b_stack;
    std::unique_ptr<float[]> heap_buf;
    if (K > 256) {
        heap_buf.reset(new float[K]);
        b_buf = heap_buf.get();
    }

    for (size_t n = 0; n < N; ++n) {
        const uint8_t* b_row = B_bytes + n * row_bytes;
        DequantRow_Neon(b_row, b_buf, K, QuantType, Scales);

        for (size_t m = 0; m < M; ++m) {
            const float* a_row = A + m * lda;

            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);

            size_t k = 0;
            const size_t vec_end = (K / 16) * 16;

            for (; k < vec_end; k += 16) {
                float32x4_t a0 = vld1q_f32(a_row + k);
                float32x4_t b0 = vld1q_f32(b_buf + k);
                acc0 = vfmaq_f32(acc0, a0, b0);

                float32x4_t a1 = vld1q_f32(a_row + k + 4);
                float32x4_t b1 = vld1q_f32(b_buf + k + 4);
                acc1 = vfmaq_f32(acc1, a1, b1);

                float32x4_t a2 = vld1q_f32(a_row + k + 8);
                float32x4_t b2 = vld1q_f32(b_buf + k + 8);
                acc2 = vfmaq_f32(acc2, a2, b2);

                float32x4_t a3 = vld1q_f32(a_row + k + 12);
                float32x4_t b3 = vld1q_f32(b_buf + k + 12);
                acc3 = vfmaq_f32(acc3, a3, b3);
            }

            // Process remaining 4-element chunks
            for (; k + 4 <= K; k += 4) {
                float32x4_t a0 = vld1q_f32(a_row + k);
                float32x4_t b0 = vld1q_f32(b_buf + k);
                acc0 = vfmaq_f32(acc0, a0, b0);
            }

            // Sum accumulators
            acc0 = vaddq_f32(acc0, acc1);
            acc2 = vaddq_f32(acc2, acc3);
            acc0 = vaddq_f32(acc0, acc2);
            float dot = vaddvq_f32(acc0);

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
//
void
SVGemm_Neon(
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

    float b_stack[256];
    float* b_buf = b_stack;
    std::unique_ptr<float[]> heap_buf;
    if (N > 256) {
        heap_buf.reset(new float[N]);
        b_buf = heap_buf.get();
    }

    const size_t vec_end_n = (N / 4) * 4;

    for (size_t m = 0; m < M; ++m) {
        float* c_row = C + m * ldc;
        const float* a_row = A + m * lda;

        // Zero output
        size_t n = 0;
        for (; n < vec_end_n; n += 4) {
            vst1q_f32(c_row + n, vdupq_n_f32(0.0f));
        }
        for (; n < N; ++n) {
            c_row[n] = 0.0f;
        }

        for (size_t k = 0; k < K; ++k) {
            const uint8_t* b_row_packed = B_bytes + k * row_bytes;
            DequantRow_Neon(b_row_packed, b_buf, N, QuantType, Scales);

            const float a_val = a_row[k];
            float32x4_t a_broadcast = vdupq_n_f32(a_val);

            n = 0;
            for (; n < vec_end_n; n += 4) {
                float32x4_t c_vec = vld1q_f32(c_row + n);
                float32x4_t b_vec = vld1q_f32(b_buf + n);
                c_vec = vfmaq_f32(c_vec, a_broadcast, b_vec);
                vst1q_f32(c_row + n, c_vec);
            }
            for (; n < N; ++n) {
                c_row[n] += a_val * b_buf[n];
            }
        }
    }
}

}  // namespace

const MLAS_KV_QUANT_GEMM_DISPATCH MlasKVQuantGemmDispatchNeon = []() {
    MLAS_KV_QUANT_GEMM_DISPATCH d;
    d.QKGemm = QKGemm_Neon;
    d.SVGemm = SVGemm_Neon;
    return d;
}();
