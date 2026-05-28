/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qkv_quant.cpp

Abstract:

    Portable scalar reference implementation of the symmetric INT4 / INT8
    quantized KV-cache GEMM API declared in mlas_qkv_quant.h. This file provides
    a correct scalar fallback; SIMD-optimized backends (AVX2, AVX512-VNNI, NEON)
    are dispatched at runtime via the platform dispatch table.

    See mlas_qkv_quant.h for the packing, scaling, and layout contract.

--*/

#include "mlas_qkv_quant.h"
#include "mlasi.h"
#include "qkv_quant_kernel.h"
#include "qkv_quant_common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>

using namespace MlasKVQuantInternal;

namespace {

constexpr int kInt4Min = -8;
constexpr int kInt4Max = 7;
constexpr int kInt8Min = -128;
constexpr int kInt8Max = 127;

// Round-to-nearest-even via rintf, matching the CUDA QDQ implementation.
inline int8_t
QuantizeInt8(float x, float inv_scale)
{
    const float q = std::rintf(x * inv_scale);
    const int qi = static_cast<int>(std::max(static_cast<float>(kInt8Min),
                                             std::min(static_cast<float>(kInt8Max), q)));
    return static_cast<int8_t>(qi);
}

inline int
QuantizeInt4Nibble(float x, float inv_scale)
{
    const float q = std::rintf(x * inv_scale);
    const int qi = static_cast<int>(std::max(static_cast<float>(kInt4Min),
                                             std::min(static_cast<float>(kInt4Max), q)));
    return qi;
}

inline float
SafeInvScale(float scale)
{
    // A zero scale (typically meaning "no data yet") is treated as 1.0 so the
    // quantize step degenerates to a clamp/round rather than producing NaN.
    return (scale != 0.0f) ? (1.0f / scale) : 1.0f;
}

inline float
DequantInt8(int8_t q, float scale)
{
    return static_cast<float>(q) * scale;
}

// Decode a single nibble from a packed byte. `col` is the unpacked column
// index; even columns occupy the low nibble, odd columns the high nibble.
inline float
DequantInt4FromByte(uint8_t packed, size_t col, float scale)
{
    const int nibble = (col & 1U) == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
    return static_cast<float>(nibble - kInt4Bias) * scale;
}

// Pack one row of FP32 source into INT8 destination.
void
QuantizeRowInt8(
    const float* src,
    int8_t* dst,
    size_t cols,
    bool per_channel,
    const float* scales)
{
    if (per_channel) {
        for (size_t c = 0; c < cols; ++c) {
            dst[c] = QuantizeInt8(src[c], SafeInvScale(scales[c]));
        }
    } else {
        const float inv_scale = SafeInvScale(scales[0]);
        for (size_t c = 0; c < cols; ++c) {
            dst[c] = QuantizeInt8(src[c], inv_scale);
        }
    }
}

// Pack one row of FP32 source into INT4 destination using the +8-biased
// two-per-byte convention. If `cols` is odd, the trailing high nibble is set
// to 0 (which decodes to -8 * scale; matches the CUDA "pair with q1 = 0"
// convention and is fine because such columns are out of range for the
// consumer).
void
QuantizeRowInt4(
    const float* src,
    uint8_t* dst,
    size_t cols,
    bool per_channel,
    const float* scales)
{
    const size_t out_bytes = (cols + 1) / 2;
    for (size_t b = 0; b < out_bytes; ++b) {
        const size_t c0 = 2 * b;
        const size_t c1 = c0 + 1;

        const float inv0 = per_channel ? SafeInvScale(scales[c0])
                                       : SafeInvScale(scales[0]);
        const int q0 = QuantizeInt4Nibble(src[c0], inv0);
        int q1 = 0;
        if (c1 < cols) {
            const float inv1 = per_channel ? SafeInvScale(scales[c1])
                                           : SafeInvScale(scales[0]);
            q1 = QuantizeInt4Nibble(src[c1], inv1);
        }
        dst[b] = static_cast<uint8_t>(
            (((q0 + kInt4Bias) & 0x0F)) |
            ((((q1 + kInt4Bias) & 0x0F)) << 4));
    }
}

// Dequantize one row from INT8.
void
DequantizeRowInt8(
    const int8_t* src,
    float* dst,
    size_t cols,
    bool per_channel,
    const float* scales)
{
    if (per_channel) {
        for (size_t c = 0; c < cols; ++c) {
            dst[c] = DequantInt8(src[c], scales[c]);
        }
    } else {
        const float scale = scales[0];
        for (size_t c = 0; c < cols; ++c) {
            dst[c] = DequantInt8(src[c], scale);
        }
    }
}

// Dequantize one row from packed INT4.
void
DequantizeRowInt4(
    const uint8_t* src,
    float* dst,
    size_t cols,
    bool per_channel,
    const float* scales)
{
    for (size_t c = 0; c < cols; ++c) {
        const uint8_t packed = src[c / 2];
        const float scale = per_channel ? scales[c] : scales[0];
        dst[c] = DequantInt4FromByte(packed, c, scale);
    }
}

}  // namespace

bool
MLASCALL
MlasIsKVQuantGemmSupported(MLAS_KV_QUANT_TYPE /*QuantType*/)
{
    // The portable reference path supports every mode on every platform.
    return true;
}

size_t
MLASCALL
MlasKVQuantPackedRowBytes(MLAS_KV_QUANT_TYPE QuantType, size_t Cols)
{
    return IsInt4Mode(QuantType) ? (Cols + 1) / 2 : Cols;
}

void
MLASCALL
MlasKVQuantize(
    const float* Src,
    void* Dst,
    size_t Rows,
    size_t Cols,
    size_t lda,
    MLAS_KV_QUANT_TYPE QuantType,
    const float* Scales,
    MLAS_THREADPOOL* ThreadPool)
{
    if (Rows == 0 || Cols == 0) {
        return;
    }

    const bool int4 = IsInt4Mode(QuantType);
    const bool per_channel = IsPerChannelMode(QuantType);
    const size_t row_bytes = MlasKVQuantPackedRowBytes(QuantType, Cols);
    auto* dst_bytes = static_cast<uint8_t*>(Dst);

    MlasTrySimpleParallel(
        ThreadPool, static_cast<ptrdiff_t>(Rows),
        [&](ptrdiff_t r_idx) {
            const size_t r = static_cast<size_t>(r_idx);
            const float* src_row = Src + r * lda;
            uint8_t* dst_row = dst_bytes + r * row_bytes;
            if (int4) {
                QuantizeRowInt4(src_row, dst_row, Cols, per_channel, Scales);
            } else {
                QuantizeRowInt8(src_row, reinterpret_cast<int8_t*>(dst_row),
                                Cols, per_channel, Scales);
            }
        });
}

void
MLASCALL
MlasKVDequantize(
    const void* Src,
    float* Dst,
    size_t Rows,
    size_t Cols,
    size_t ldb,
    MLAS_KV_QUANT_TYPE QuantType,
    const float* Scales,
    MLAS_THREADPOOL* ThreadPool)
{
    if (Rows == 0 || Cols == 0) {
        return;
    }

    const bool int4 = IsInt4Mode(QuantType);
    const bool per_channel = IsPerChannelMode(QuantType);
    const size_t row_bytes = MlasKVQuantPackedRowBytes(QuantType, Cols);
    const auto* src_bytes = static_cast<const uint8_t*>(Src);

    MlasTrySimpleParallel(
        ThreadPool, static_cast<ptrdiff_t>(Rows),
        [&](ptrdiff_t r_idx) {
            const size_t r = static_cast<size_t>(r_idx);
            const uint8_t* src_row = src_bytes + r * row_bytes;
            float* dst_row = Dst + r * ldb;
            if (int4) {
                DequantizeRowInt4(src_row, dst_row, Cols, per_channel, Scales);
            } else {
                DequantizeRowInt8(reinterpret_cast<const int8_t*>(src_row),
                                  dst_row, Cols, per_channel, Scales);
            }
        });
}

void
MLASCALL
MlasQKGemm(
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
    size_t ldc,
    MLAS_THREADPOOL* ThreadPool)
{
    if (M == 0 || N == 0) {
        return;
    }
    if (K == 0) {
        for (size_t m = 0; m < M; ++m) {
            std::memset(C + m * ldc, 0, N * sizeof(float));
        }
        return;
    }

    //
    // Try the SIMD-optimized dispatch path. The vectorized kernels handle
    // the full M×N×K computation in a single call (no thread pool — the
    // caller's thread-pool loop already partitions across heads/batches).
    //
    const auto* Dispatch = GetMlasPlatform().KVQuantGemmDispatch;
    if (Dispatch != nullptr && Dispatch->QKGemm != nullptr) {
        // The dispatch kernels are designed to be called per-(batch,head) tile
        // from an outer parallel loop, so we invoke them directly here. For
        // large N the outer loop in gqa_attention_base already parallelizes.
        Dispatch->QKGemm(M, N, K, Alpha, A, lda, B, QuantType, Scales, C, ldc);
        return;
    }

    //
    // Scalar reference fallback.
    //
    const bool int4 = IsInt4Mode(QuantType);
    const bool per_channel = IsPerChannelMode(QuantType);
    const size_t row_bytes = MlasKVQuantPackedRowBytes(QuantType, K);
    const auto* B_bytes = static_cast<const uint8_t*>(B);

    // Parallelize over N (independent output columns -> independent B rows).
    MlasTrySimpleParallel(
        ThreadPool, static_cast<ptrdiff_t>(N),
        [&](ptrdiff_t n_idx) {
            const size_t n = static_cast<size_t>(n_idx);
            const uint8_t* b_row = B_bytes + n * row_bytes;

            // Dequantize B row [K] once and reuse across all M.
            // Small buffer (K = head_size, typically <= 256).
            float b_dequant[1024];
            float* b_buf = b_dequant;
            std::unique_ptr<float[]> heap_buf;
            if (K > sizeof(b_dequant) / sizeof(b_dequant[0])) {
                heap_buf.reset(new float[K]);
                b_buf = heap_buf.get();
            }
            if (int4) {
                DequantizeRowInt4(b_row, b_buf, K, per_channel, Scales);
            } else {
                DequantizeRowInt8(reinterpret_cast<const int8_t*>(b_row),
                                  b_buf, K, per_channel, Scales);
            }

            for (size_t m = 0; m < M; ++m) {
                const float* a_row = A + m * lda;
                float acc = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    acc += a_row[k] * b_buf[k];
                }
                C[m * ldc + n] = Alpha * acc;
            }
        });
}

void
MLASCALL
MlasSVGemm(
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
    MLAS_THREADPOOL* ThreadPool)
{
    if (M == 0 || N == 0) {
        return;
    }
    if (K == 0) {
        for (size_t m = 0; m < M; ++m) {
            std::memset(C + m * ldc, 0, N * sizeof(float));
        }
        return;
    }

    //
    // Try the SIMD-optimized dispatch path.
    //
    const auto* Dispatch = GetMlasPlatform().KVQuantGemmDispatch;
    if (Dispatch != nullptr && Dispatch->SVGemm != nullptr) {
        Dispatch->SVGemm(M, N, K, A, lda, B, QuantType, Scales, C, ldc);
        return;
    }

    //
    // Scalar reference fallback.
    //
    const bool int4 = IsInt4Mode(QuantType);
    const bool per_channel = IsPerChannelMode(QuantType);
    const size_t row_bytes = MlasKVQuantPackedRowBytes(QuantType, N);
    const auto* B_bytes = static_cast<const uint8_t*>(B);

    // Parallelize over M (output rows). Each M iterates over the full K
    // reduction, dequantizing B rows one at a time.
    MlasTrySimpleParallel(
        ThreadPool, static_cast<ptrdiff_t>(M),
        [&](ptrdiff_t m_idx) {
            const size_t m = static_cast<size_t>(m_idx);
            const float* a_row = A + m * lda;
            float* c_row = C + m * ldc;
            std::memset(c_row, 0, N * sizeof(float));

            // Per-row scratch for one dequantized B row of length N.
            float b_dequant[1024];
            float* b_buf = b_dequant;
            std::unique_ptr<float[]> heap_buf;
            if (N > sizeof(b_dequant) / sizeof(b_dequant[0])) {
                heap_buf.reset(new float[N]);
                b_buf = heap_buf.get();
            }

            for (size_t k = 0; k < K; ++k) {
                const uint8_t* b_row_packed = B_bytes + k * row_bytes;
                if (int4) {
                    DequantizeRowInt4(b_row_packed, b_buf, N, per_channel, Scales);
                } else {
                    DequantizeRowInt8(reinterpret_cast<const int8_t*>(b_row_packed),
                                      b_buf, N, per_channel, Scales);
                }
                const float a_val = a_row[k];
                for (size_t n = 0; n < N; ++n) {
                    c_row[n] += a_val * b_buf[n];
                }
            }
        });
}
