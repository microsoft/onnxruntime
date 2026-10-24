/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx2_2bit_fp32.h

Abstract:

    This module implements the SQNBIT_CompFp32 kernels for 2-bit block
    quantized matrix B (W2A16-fp32 / SQ2BitGemmVariant_CompFp32):

      * SQ2BitGemmM1Kernel_CompFp32_avx2 -- dot product kernel for M == 1
      * Q2BitBlkDequantBForSgemm_CompFp32_avx2 -- dequantize B into the packed
        format expected by the SGEMM kernel (M > 1 path)
      * SQ2BitGemmPackQuantBData_avx2 -- copies the quantized B data into the
        prepacked buffer

    Unlike the CompInt8 W2 path (sqnbitgemm_kernel_avx512_2bit*), the CompFp32
    kernels consume the quantized B data in its native, unswizzled layout:
    every byte holds 4 consecutive 2-bit values of one block,

        byte b of a block = w[4b] | w[4b+1] << 2 | w[4b+2] << 4 | w[4b+3] << 6,

    which is the layout produced by the MatMulNBits quantizers
    (QuantizeMatMulNBitsBlockwise / quantize_matmul_2bits). The prepack step
    is therefore a plain copy. Zero points use the same packing: 4 zero
    points per byte, little-endian 2-bit pairs, matching
    matmul_nbits_helper::GetUint8ZeroPointBlobSize(k_blocks, 2). When no zero
    point tensor is supplied, the unsigned 2-bit mid-point 2 is subtracted,
    matching MlasDequantizeBlockwise<float, 2>
    (BitsTraits<2, false>::kMid).

    The dequantized value arithmetic matches the unpacked fallback path
    exactly: (q - zp) as integers, then * scale in fp32, then the dot
    product accumulated with FMA.

--*/

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include "qnbitgemm.h"
#include "sqnbitgemm_kernel_avx_common.h"

namespace
{

// Loads n (<= 8) float values, zero-filling the unused lanes. Zero-filled
// lanes multiply garbage-free decoded B values (2-bit codes are always
// finite), so tail blocks contribute exactly zero to the accumulator.
MLAS_FORCEINLINE __m256
MlasQ2LoadFloatN_avx2(const float* data, int n)
{
    assert(n <= 8);
    if (n <= 0) {
        return _mm256_setzero_ps();
    }
    static const int32_t mask_buffer[16] = {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0};
    const __m256i load_mask = _mm256_loadu_si256((const __m256i*)(mask_buffer + 8 - n));
    return _mm256_maskload_ps(data, load_mask);
}

// Unpacks 32 consecutive 2-bit values (8 packed bytes) into the low and high
// 128-bit lanes of a __m256i so that the result bytes 0..31 hold the values
// in k order:
//   x = [b0..b7, b0..b7] (broadcast); pK byte j = w[4j+K];
//   t01/t23 pair the phases; the two 16-bit interleave halves produce
//   w0..w15 and w16..w31, which are combined with one lane insert.
MLAS_FORCEINLINE __m256i
MlasQ2Unpack32Weights(const std::byte* data)
{
    const __m256i x = _mm256_broadcastq_epi64(_mm_loadl_epi64((const __m128i*)data));
    const __m256i mask3 = _mm256_set1_epi8(3);
    const __m256i p0 = _mm256_and_si256(x, mask3);
    const __m256i p1 = _mm256_and_si256(_mm256_srli_epi16(x, 2), mask3);
    const __m256i p2 = _mm256_and_si256(_mm256_srli_epi16(x, 4), mask3);
    const __m256i p3 = _mm256_and_si256(_mm256_srli_epi16(x, 6), mask3);
    const __m256i t01 = _mm256_unpacklo_epi8(p0, p1);
    const __m256i t23 = _mm256_unpacklo_epi8(p2, p3);
    const __m256i lo16 = _mm256_unpacklo_epi16(t01, t23);  // w0..w15 in both lanes
    const __m256i hi16 = _mm256_unpackhi_epi16(t01, t23);  // w16..w31 in both lanes
    return _mm256_inserti128_si256(lo16, _mm256_castsi256_si128(hi16), 1);
}

// Extracts the zero point of k_blk from the packed zero point stream
// (4 zero points per byte, little-endian 2-bit pairs).
MLAS_FORCEINLINE uint8_t
MlasQ2UnpackZeroPoint(const std::byte* zp_stream, size_t k_blk)
{
    return static_cast<uint8_t>(
        (std::to_integer<int>(zp_stream[k_blk / 4]) >> (2 * (k_blk % 4))) & 3
    );
}

constexpr uint8_t MlasQ2DefaultSymmetricZeroPoint = 2;  // BitsTraits<2, false>::kMid

template <bool HasZeroPoint, size_t NCols>
MLAS_FORCEINLINE void
ComputeDotProducts2Bit_CompFp32_avx2(
    const size_t BlkLen,
    const float* A,
    const std::byte* QuantBDataColPtr,
    const float* QuantBScaleColPtr,
    const std::byte* QuantBZeroPointColPtr,
    float* SumPtr,
    size_t CountK,
    size_t StrideQuantBData,
    size_t StrideQuantBScale,
    size_t StrideQuantBZeroPoint,
    const float* BiasPtr
)
{
    constexpr size_t BlkBitWidth2 = 2;
    constexpr size_t SubBlkLen32 = 32;
    static_assert(SubBlkLen32 % 8 == 0);
    static_assert(BlkBitWidth2 == 2);

    __m256 acc[NCols]{};
    const size_t blk_data_size_in_bytes = MlasQNBitBlkDataSizeInBytes(BlkBitWidth2, BlkLen);

    const std::byte* b_blk_data_ptr[NCols];
    const float* s[NCols];
    [[maybe_unused]] uint8_t zp[NCols];
    UnrolledLoop<NCols>([&](size_t i) {
        b_blk_data_ptr[i] = QuantBDataColPtr + i * StrideQuantBData;
        s[i] = QuantBScaleColPtr + i * StrideQuantBScale;
        if constexpr (HasZeroPoint) {
            zp[i] = 0;  // refreshed per block below
        }
    });

    for (size_t k = 0; k < CountK; k += BlkLen) {
        const size_t ck = std::min(CountK - k, BlkLen);

        UnrolledLoop<NCols>([&](size_t i) {
            if constexpr (HasZeroPoint) {
                zp[i] = MlasQ2UnpackZeroPoint(QuantBZeroPointColPtr + i * StrideQuantBZeroPoint, k / BlkLen);
            }
        });

        for (size_t kk = 0; kk < ck; kk += SubBlkLen32) {
            const size_t kklen = std::min(SubBlkLen32, ck - kk);

            // int math so the unused chunks (kklen - n <= 0) zero-fill instead of
            // underflowing size_t.
            const int kklen_i = static_cast<int>(kklen);
            __m256 av0 = MlasQ2LoadFloatN_avx2(A + k + kk, std::min(kklen_i, 8));
            __m256 av1 = MlasQ2LoadFloatN_avx2(A + k + kk + 8, std::min(kklen_i - 8, 8));
            __m256 av2 = MlasQ2LoadFloatN_avx2(A + k + kk + 16, std::min(kklen_i - 16, 8));
            __m256 av3 = MlasQ2LoadFloatN_avx2(A + k + kk + 24, std::min(kklen_i - 24, 8));

            UnrolledLoop<NCols>([&](size_t i) {
                // Decode 32 weights of this block slice (8 packed bytes).
                __m256i bytes = MlasQ2Unpack32Weights(b_blk_data_ptr[i] + kk / 4);

                if constexpr (HasZeroPoint) {
                    bytes = _mm256_sub_epi8(bytes, _mm256_set1_epi8(zp[i]));
                } else {
                    bytes = _mm256_sub_epi8(bytes, _mm256_set1_epi8(MlasQ2DefaultSymmetricZeroPoint));
                }

                // Convert to 16-bit int
                const __m256i vx16_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(bytes));
                const __m256i vx16_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes, 1));

                // Convert to 32-bit int -> float 32, multiply by the block scale
                const float scale = *s[i];
                const __m256 scale_ps = _mm256_set1_ps(scale);
                const __m256 bvf0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(vx16_lo))), scale_ps);
                const __m256 bvf1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_lo, 1))), scale_ps);
                const __m256 bvf2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(vx16_hi))), scale_ps);
                const __m256 bvf3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_hi, 1))), scale_ps);

                acc[i] = _mm256_fmadd_ps(bvf0, av0, acc[i]);
                acc[i] = _mm256_fmadd_ps(bvf1, av1, acc[i]);
                acc[i] = _mm256_fmadd_ps(bvf2, av2, acc[i]);
                acc[i] = _mm256_fmadd_ps(bvf3, av3, acc[i]);
            });
        }  // kk

        UnrolledLoop<NCols>([&](size_t i) {
            b_blk_data_ptr[i] += blk_data_size_in_bytes;
            s[i] += 1;
        });
    }  // k

    if constexpr (NCols == 4) {
        __m128 acc_x = FoldAccumulators(acc[0], acc[1], acc[2], acc[3]);
        if (BiasPtr != nullptr) {
            acc_x = _mm_add_ps(acc_x, _mm_loadu_ps(BiasPtr));
        }
        _mm_storeu_ps(SumPtr, acc_x);
    } else {
        UnrolledLoop<NCols>([&](size_t i) {
            const __m128 vlow = _mm256_castps256_ps128(acc[i]);
            const __m128 vhigh = _mm256_extractf128_ps(acc[i], 1);

            __m128 vsum = _mm_add_ps(vlow, vhigh);
            vsum = _mm_hadd_ps(vsum, vsum);
            vsum = _mm_hadd_ps(vsum, vsum);

            _mm_store_ss(&SumPtr[i], vsum);
            SumPtr[i] += BiasPtr == nullptr ? 0.0f : BiasPtr[i];
        });
    }
}

}  // namespace

//
// SQNBIT_CompFp32 M==1 kernel: consumes the native (unswizzled) 2-bit
// layout; the packing step is a plain copy, so this runs against the same
// bytes the quantizer produced.
//
inline void
SQ2BitGemmM1Kernel_CompFp32_avx2(
    size_t BlkLen,
    const float* A,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t CountK,
    size_t BlockStrideQuantB,
    const float* Bias
)
{
    constexpr size_t BlkBitWidth2 = 2;
    constexpr size_t NCols4 = 4;

    const size_t BlockCountK = BlockStrideQuantB;

    const size_t StrideQuantBData = BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth2, BlkLen);
    const size_t StrideQuantBScale = BlockCountK;
    const size_t StrideQuantBZeroPoint = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth2>(BlockCountK);

    const float* BiasPtr = Bias;

    const std::byte* QuantBDataColPtr = QuantBData;
    const float* QuantBScaleColPtr = QuantBScale;
    const std::byte* QuantBZeroPointColPtr = QuantBZeroPoint;

    float* SumPtr = C;

    int64_t nblk = static_cast<int64_t>(CountN) - NCols4;
    while (nblk >= 0) {
        if (QuantBZeroPoint != nullptr) {
            ComputeDotProducts2Bit_CompFp32_avx2<true, NCols4>(
                BlkLen, A, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr,
                CountK, StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint, BiasPtr
            );
        } else {
            ComputeDotProducts2Bit_CompFp32_avx2<false, NCols4>(
                BlkLen, A, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr,
                CountK, StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint, BiasPtr
            );
        }

        QuantBDataColPtr += NCols4 * StrideQuantBData;
        QuantBScaleColPtr += NCols4 * StrideQuantBScale;
        if (QuantBZeroPoint != nullptr) {
            QuantBZeroPointColPtr += NCols4 * StrideQuantBZeroPoint;
        }

        BiasPtr += BiasPtr != nullptr ? NCols4 : 0;
        SumPtr += NCols4;

        nblk -= NCols4;
    }

    // left over columns less than NCols4
    nblk += NCols4;
    for (int64_t n = 0; n < nblk; ++n) {
        if (QuantBZeroPoint != nullptr) {
            ComputeDotProducts2Bit_CompFp32_avx2<true, 1>(
                BlkLen, A, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr,
                CountK, StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint, BiasPtr
            );
        } else {
            ComputeDotProducts2Bit_CompFp32_avx2<false, 1>(
                BlkLen, A, QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr, SumPtr,
                CountK, StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint, BiasPtr
            );
        }

        QuantBDataColPtr += StrideQuantBData;
        QuantBScaleColPtr += StrideQuantBScale;
        if (QuantBZeroPoint != nullptr) {
            QuantBZeroPointColPtr += StrideQuantBZeroPoint;
        }

        BiasPtr += BiasPtr != nullptr ? 1 : 0;
        SumPtr += 1;
    }
}

//
// SQNBIT_CompFp32 M>1 support: dequantize CountN columns of B into the
// packed SGEMM B format. Mirrors
// Q4BitBlkDequantBForSgemmBlkLen32AndMore_CompFp32_avx2.
//
inline void
Q2BitBlkDequantBForSgemm_CompFp32_avx2(
    const size_t BlkLen,
    float* FpData,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    const size_t CountN,
    const size_t CountK,
    const size_t BlockStrideQuantB
)
{
    constexpr size_t BlkBitWidth2 = 2;
    constexpr size_t NCols8 = 8;                   // process NCols8 columns of QuantB at a time
    constexpr size_t GemmFloatKernelWidth16 = 16;  // mlas GemmFloatKernel requires B with width 16
    constexpr size_t SubblkLen32 = 32;             // process SubblkLen32 rows of QuantB at a time

    assert(BlkLen % SubblkLen32 == 0);  // BlkLen in {32, 64, 128}

    const size_t blk_data_size_in_bytes = MlasQNBitBlkDataSizeInBytes(BlkBitWidth2, BlkLen);
    const size_t subblk_data_size_in_bytes = MlasQNBitBlkDataSizeInBytes(BlkBitWidth2, SubblkLen32);
    const size_t b_data_col_stride_in_bytes = BlockStrideQuantB * blk_data_size_in_bytes;
    const /*constexpr*/ bool HasZeroPoint = QuantBZeroPoint != nullptr;
    const size_t zp_col_stride_in_bytes = MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth2>(BlockStrideQuantB);

    for (size_t col = 0; col < CountN; col += NCols8) {
        const size_t cols = std::min(NCols8, CountN - col);
        for (size_t k = 0; k < BlockStrideQuantB; k++) {
            // count # of tiles plus blks of the current tile from top
            const size_t tile_count = col / GemmFloatKernelWidth16;
            float* dst_ptr = FpData + (tile_count * CountK + k * BlkLen) * GemmFloatKernelWidth16;
            if (col % GemmFloatKernelWidth16 >= NCols8) {
                // for the second half to 16 width tile
                dst_ptr += NCols8;
            }
            const std::byte* b_data_ptr = QuantBData + col * b_data_col_stride_in_bytes + k * blk_data_size_in_bytes;
            const float* scale_ptr = QuantBScale + col * BlockStrideQuantB + k;
            const std::byte* zp_ptr = QuantBZeroPoint + col * zp_col_stride_in_bytes;

            for (size_t subblk = 0; subblk < BlkLen / SubblkLen32; subblk++) {
                __m256i weight_32_epi8[NCols8];
                __m256 scale_8_ps[NCols8];
                UnrolledLoop<NCols8>([&](size_t col_) {
                    if (col_ < cols) {
                        const uint8_t zp = HasZeroPoint
                                               ? MlasQ2UnpackZeroPoint(zp_ptr + col_ * zp_col_stride_in_bytes, k)
                                               : MlasQ2DefaultSymmetricZeroPoint;
                        weight_32_epi8[col_] =
                            _mm256_sub_epi8(
                                MlasQ2Unpack32Weights(b_data_ptr + col_ * b_data_col_stride_in_bytes + subblk * subblk_data_size_in_bytes),
                                _mm256_set1_epi8(zp)
                            );
                        scale_8_ps[col_] = _mm256_set1_ps(*(scale_ptr + col_ * BlockStrideQuantB));
                    } else {
                        weight_32_epi8[col_] = _mm256_setzero_si256();
                        scale_8_ps[col_] = _mm256_setzero_ps();
                    }
                });
                for (int i_of_4 = 0; i_of_4 < 4; i_of_4++) {
                    __m256 weight_8_ps[8];
                    for (size_t col_ = 0; col_ < 8; col_++) {
                        if (col_ < cols) {
                            __m256i weight_i_16_epi16;
                            __m256i weight_i_j_8_epi32;
                            if (i_of_4 == 0) {
                                weight_i_16_epi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(weight_32_epi8[col_], 0));
                                weight_i_j_8_epi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(weight_i_16_epi16, 0));
                            } else if (i_of_4 == 1) {
                                weight_i_16_epi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(weight_32_epi8[col_], 0));
                                weight_i_j_8_epi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(weight_i_16_epi16, 1));
                            } else if (i_of_4 == 2) {
                                weight_i_16_epi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(weight_32_epi8[col_], 1));
                                weight_i_j_8_epi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(weight_i_16_epi16, 0));
                            } else {
                                weight_i_16_epi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(weight_32_epi8[col_], 1));
                                weight_i_j_8_epi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(weight_i_16_epi16, 1));
                            }
                            weight_8_ps[col_] = _mm256_mul_ps(_mm256_cvtepi32_ps(weight_i_j_8_epi32), scale_8_ps[col_]);
                        } else {
                            weight_8_ps[col_] = _mm256_setzero_ps();
                        }
                    }
                    // transpose and store
                    __m256 a0 = _mm256_unpacklo_ps(weight_8_ps[0], weight_8_ps[1]);
                    __m256 a1 = _mm256_unpackhi_ps(weight_8_ps[0], weight_8_ps[1]);
                    __m256 a2 = _mm256_unpacklo_ps(weight_8_ps[2], weight_8_ps[3]);
                    __m256 a3 = _mm256_unpackhi_ps(weight_8_ps[2], weight_8_ps[3]);
                    __m256 a4 = _mm256_unpacklo_ps(weight_8_ps[4], weight_8_ps[5]);
                    __m256 a5 = _mm256_unpackhi_ps(weight_8_ps[4], weight_8_ps[5]);
                    __m256 a6 = _mm256_unpacklo_ps(weight_8_ps[6], weight_8_ps[7]);
                    __m256 a7 = _mm256_unpackhi_ps(weight_8_ps[6], weight_8_ps[7]);

                    __m256 b0 = _mm256_shuffle_ps(a0, a2, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 b1 = _mm256_shuffle_ps(a0, a2, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 b2 = _mm256_shuffle_ps(a1, a3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 b3 = _mm256_shuffle_ps(a1, a3, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 b4 = _mm256_shuffle_ps(a4, a6, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 b5 = _mm256_shuffle_ps(a4, a6, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 b6 = _mm256_shuffle_ps(a5, a7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 b7 = _mm256_shuffle_ps(a5, a7, _MM_SHUFFLE(3, 2, 3, 2));

                    const size_t ij_offset_in_k = i_of_4 * 8 * GemmFloatKernelWidth16;
                    __m256 weight_transposed_8_ps = _mm256_permute2f128_ps(b0, b4, 0x20);
                    _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 0 * GemmFloatKernelWidth16, weight_transposed_8_ps);
                    weight_transposed_8_ps = _mm256_permute2f128_ps(b1, b5, 0x20);
                    _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 1 * GemmFloatKernelWidth16, weight_transposed_8_ps);
                    weight_transposed_8_ps = _mm256_permute2f128_ps(b2, b6, 0x20);
                    _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 2 * GemmFloatKernelWidth16, weight_transposed_8_ps);
                    weight_transposed_8_ps = _mm256_permute2f128_ps(b3, b7, 0x20);
                    _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 3 * GemmFloatKernelWidth16, weight_transposed_8_ps);
                    weight_transposed_8_ps = _mm256_permute2f128_ps(b0, b4, 0x31);
                    _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 4 * GemmFloatKernelWidth16, weight_transposed_8_ps);
                    weight_transposed_8_ps = _mm256_permute2f128_ps(b1, b5, 0x31);
                    _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 5 * GemmFloatKernelWidth16, weight_transposed_8_ps);
                    weight_transposed_8_ps = _mm256_permute2f128_ps(b2, b6, 0x31);
                    _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 6 * GemmFloatKernelWidth16, weight_transposed_8_ps);
                    weight_transposed_8_ps = _mm256_permute2f128_ps(b3, b7, 0x31);
                    _mm256_storeu_ps(dst_ptr + ij_offset_in_k + 7 * GemmFloatKernelWidth16, weight_transposed_8_ps);
                }
                dst_ptr += SubblkLen32 * GemmFloatKernelWidth16;
            }  // subblk
        }
    }
}

//
// SQNBIT_CompFp32 packing: the kernels consume the native quantizer layout,
// so packing is a plain (parallel) copy into the prepacked buffer.
//
inline void
SQ2BitGemmPackQuantBData_avx2(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE /*ComputeType*/,
    const std::byte* QuantBDataBegin,
    std::byte* PackedQuantBDataBegin,
    MLAS_THREADPOOL* ThreadPool,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*BackendKernelSelectorConfig*/
)
{
    constexpr size_t BlkBitWidth2 = 2;

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t BlkDataSize = MlasQNBitBlkDataSizeInBytes(BlkBitWidth2, BlkLen);
    const size_t TotalBytes = N * BlockCountK * BlkDataSize;
    if (TotalBytes == 0) {
        return;
    }

    // Copy column by column; a column's blocks are contiguous in both layouts.
    MlasTrySimpleParallel(
        ThreadPool, static_cast<ptrdiff_t>(N),
        [&](ptrdiff_t tid) {
            const size_t offset = static_cast<size_t>(tid) * BlockCountK * BlkDataSize;
            std::memcpy(PackedQuantBDataBegin + offset, QuantBDataBegin + offset, BlockCountK * BlkDataSize);
        }
    );
}
