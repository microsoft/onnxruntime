/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    mlas_qnbit.h

Abstract:

    This module contains the public data structures and procedure prototypes
    for blocked n-bit quantized GEMM.

    N-bit block quantization is used to compress weight tensors of large
    language models.

--*/

#pragma once

#include "mlas.h"
#include "mlas_gemm_postprocessor.h"

/**
 * @brief Data parameters for float/n-bit quantized int GEMM routine.
 */
struct MLAS_SQNBIT_GEMM_DATA_PARAMS {
    const float* A = nullptr;                ///< address of A (float32 matrix)
    size_t lda = 0;                          ///< leading dimension of A
    const void* PackedBData = nullptr;       ///< address of B (quantized and packed n-bit int values)
    const float* PackedBScale = nullptr;     ///< address of scale values of quantized B, one per block
    const void* PackedBZeroPoint = nullptr;  ///< optional address of zero point values of quantized B, one per block
    bool IsBPacked = false;                  ///< whether B values are packed in the optimal format for the computation
    const float* Bias = nullptr;             ///< optional address of Bias, vector size N
    float* C = nullptr;                      ///< address of result matrix
    size_t ldc = 0;                          ///< leading dimension of C

    ///< optional post processing to apply to result matrix
    MLAS_GEMM_POSTPROCESSOR<float>* PostProcessor = nullptr;
};

/**
 * @brief Batched GEMM:  C = A * B + Bias
 *        A must be a float32 matrix
 *        B must be a quantized and packed n-bit int matrix
 *
 * @param[in]       M               row size of matrix A and C
 * @param[in]       N               column size of matrix B and C
 * @param[in]       K               column size of matrix A and row size of matrix B
 * @param[in]       BatchN          number of batches
 * @param[in]       BlkBitWidth     quantized value bit width (e.g., 4 means 4 bit ints)
 * @param[in]       BlkLen          number of quantized values per block
 * @param[inout]    DataParams      An array (size BatchN) of parameter blocks
 * @param[in]       ThreadPool      optional thread pool to use
 */
void MLASCALL
MlasSQNBitGemmBatch(
    size_t M,
    size_t N,
    size_t K,
    size_t BatchN,
    size_t BlkBitWidth,
    size_t BlkLen,
    const MLAS_SQNBIT_GEMM_DATA_PARAMS* DataParams,
    MLAS_THREADPOOL* ThreadPool = nullptr
);

/**
 * @brief Determines whether a float32/quantized n-bit int GEMM implementation is available on the current platform.
 * @param[in]   BlkBitWidth     quantized value bit width (e.g., 4 means 4 bit ints)
 * @param[in]   BlkLen          number of quantized values per block
 */
bool MLASCALL
MlasIsSQNBitGemmAvailable(
    size_t BlkBitWidth,
    size_t BlkLen
);


// reference packing impl
// put it here for now so unit tests/benchmark code can use it

#include <cmath>

template <size_t BlkBitWidth, size_t BlkLen>
struct MlasReferenceQNBitPacking {
    static_assert(BlkBitWidth == 4, "Only implemented for BlkBitWidth == 4.");

    static void GetPackedBSizes(
        size_t CountN, size_t CountK, size_t& PackedBDataSizeInBytes, size_t& PackedBScaleElementCount, size_t* PackedBZeroPointSizeInBytes
    )
    {
        const size_t BlockCountK = DivRoundUp(CountK, BlkLen);
        const size_t TotalBlockCount = CountN * BlockCountK;

        PackedBDataSizeInBytes = TotalBlockCount * BlkDataSizeInBytes();
        PackedBScaleElementCount = TotalBlockCount;
        if (PackedBZeroPointSizeInBytes) {
            *PackedBZeroPointSizeInBytes = CountN * ZeroPointsForBlksSizeInBytes(BlockCountK);
        }
    }

    static void PackB(
        size_t CountN, size_t CountK, const float* BDataPtr, size_t ldb, uint8_t* PackedBDataPtr, float* PackedBScalePtr, uint8_t* PackedBZeroPointPtr
    )
    {
        const size_t BlockCountK = DivRoundUp(CountK, BlkLen);

        uint8_t* PackedBDataColPtr = PackedBDataPtr;
        float* PackedBScaleColPtr = PackedBScalePtr;
        uint8_t* PackedBZeroPointColPtr = PackedBZeroPointPtr;

        for (size_t n = 0; n < CountN; ++n) {
            for (size_t k = 0, k_blk_idx = 0; k < CountK; k += BlkLen, k_blk_idx += 1) {
                size_t kklen = std::min(BlkLen, CountK - k);

                uint8_t* PackedBBlkDataPtr = PackedBDataColPtr + k_blk_idx * BlkDataSizeInBytes();

                if (PackedBZeroPointColPtr) {
                    float scale_block;
                    uint8_t zp_block;
                    QuantizeBlock(BDataPtr + k * ldb + n, ldb, kklen, PackedBBlkDataPtr, scale_block, zp_block);

                    if ((k_blk_idx & 1) == 0) {
                        PackedBZeroPointColPtr[k_blk_idx / 2] = zp_block & 0x0F;
                    } else {
                        PackedBZeroPointColPtr[k_blk_idx / 2] |= zp_block << 4;
                    }

                    PackedBScaleColPtr[k_blk_idx] = scale_block;
                } else {
                    float scale_block;
                    QuantizeBlock(BDataPtr + k * ldb + n, ldb, kklen, PackedBBlkDataPtr, scale_block);

                    PackedBScaleColPtr[k_blk_idx] = scale_block;
                }
            }

            PackedBDataColPtr += BlockCountK * BlkDataSizeInBytes();
            PackedBScaleColPtr += BlockCountK;
            if (PackedBZeroPointColPtr != nullptr) {
                PackedBZeroPointColPtr += ZeroPointsForBlksSizeInBytes(BlockCountK);
            }
        }
    }

    static void UnpackB(
        size_t CountN, size_t CountK, const uint8_t* PackedBDataPtr, const float* PackedBScalePtr, const uint8_t* PackedBZeroPointPtr, float* BDataPtr, size_t ldb
    )
    {
        const size_t BlockCountK = DivRoundUp(CountK, BlkLen);

        const uint8_t* PackedBDataColPtr = PackedBDataPtr;
        const float* PackedBScaleColPtr = PackedBScalePtr;
        const uint8_t* PackedBZeroPointColPtr = PackedBZeroPointPtr;

        for (size_t n = 0; n < CountN; ++n) {
            for (size_t k = 0, k_blk_idx = 0; k < CountK; k += BlkLen, k_blk_idx += 1) {
                size_t kklen = std::min(BlkLen, CountK - k);

                const uint8_t* PackedBBlkDataPtr = PackedBDataColPtr + k_blk_idx * BlkDataSizeInBytes();
                const float scale_block = PackedBScaleColPtr[k_blk_idx];

                if (PackedBZeroPointColPtr) {
                    const uint8_t zp_block = ((k_blk_idx & 1) == 1)
                                                 ? (PackedBZeroPointColPtr[k_blk_idx / 2] >> 4)
                                                 : (PackedBZeroPointColPtr[k_blk_idx / 2] & 0x0F);

                    DequantizeBlock(BDataPtr + k * ldb + n, ldb, kklen, PackedBBlkDataPtr, scale_block, zp_block);
                } else {
                    DequantizeBlock(BDataPtr + k * ldb + n, ldb, kklen, PackedBBlkDataPtr, scale_block);
                }
            }

            PackedBDataColPtr += BlockCountK * BlkDataSizeInBytes();
            PackedBScaleColPtr += BlockCountK;
            if (PackedBZeroPointColPtr != nullptr) {
                PackedBZeroPointColPtr += ZeroPointsForBlksSizeInBytes(BlockCountK);
            }
        }
    }

    static void QuantizeBlock(
        const float* b_begin, size_t ldb, size_t actual_block_size, uint8_t* data_block, float& scale_block, uint8_t& zp_block
    )
    {
        float min = *b_begin;
        float max = *b_begin;
        for (int32_t kk = 0; kk < actual_block_size; kk++) {
            const float v = b_begin[ldb * kk];
            if (v < min) min = v;
            if (v > max) max = v;
        }
        min = std::min(min, 0.0f);
        max = std::max(max, 0.0f);

        scale_block = (max - min) / ((1 << 4) - 1);

        const float reciprocal_scale = scale_block ? 1.0f / scale_block : 0.0f;
        float zero_point_fp = min;
        if (scale_block != 0.0f) {
            zero_point_fp = 0.f - min / scale_block;
        }

        // Handle any clamping
        if (zero_point_fp < 0.0f) {
            zp_block = 0;
        } else if (zero_point_fp > 15.0f) {
            zp_block = 15;
        } else {
            zp_block = (uint8_t)roundf(zero_point_fp);
        }

        for (int32_t kk = 0; kk < actual_block_size; kk += 2) {
            const float v0 = b_begin[ldb * kk];
            const uint8_t vi0 = (uint8_t)std::min(15.0f, std::max(0.0f, roundf(v0 * reciprocal_scale + zp_block)));

            const float v1 = (kk + 1 < actual_block_size) ? b_begin[ldb * (kk + 1)] : 0.f;
            const uint8_t vi1 = (uint8_t)std::min(15.0f, std::max(0.0f, roundf(v1 * reciprocal_scale + zp_block)));

            data_block[kk / 2] = vi0 | (vi1 << 4);
        }
    }

    static void QuantizeBlock(
        const float* b_begin, size_t ldb, size_t actual_block_size, uint8_t* data_block, float& scale_block
    )
    {
        float amax = 0.0f;  // abs(max)
        float max = 0.0f;

        for (int32_t kk = 0; kk < actual_block_size; kk++) {
            const float v = b_begin[ldb * kk];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max = v;
            }
        }

        scale_block = max / (-8.f);
        const float reciprocal_scale = scale_block ? 1.0f / scale_block : 0.0f;

        for (int32_t kk = 0; kk < actual_block_size; kk += 2) {
            const float v0 = b_begin[ldb * kk] * reciprocal_scale;
            const uint8_t vi0 = (uint8_t)std::min(15.0f, std::max(0.0f, roundf(v0 + 8.f)));

            const float v1 = (kk + 1 < actual_block_size) ? b_begin[ldb * (kk + 1)] * reciprocal_scale : 0;
            const uint8_t vi1 = (uint8_t)std::min(15.0f, std::max(0.0f, roundf(v1 + 8.f)));

            data_block[kk / 2] = vi0 | (vi1 << 4);
        }
    }

    static void DequantizeBlock(
        float* b_begin, size_t ldb, size_t actual_block_size, const uint8_t* data_block, float scale_block, uint8_t zp_block
    )
    {
        for (size_t kk = 0; kk < actual_block_size; kk += 2) {
            float x0 = static_cast<float>(data_block[kk / 2] & 0x0F);
            b_begin[ldb * kk] = scale_block * (x0 - zp_block);

            if (kk + 1 < actual_block_size) {
                float x1 = static_cast<float>(data_block[kk / 2] >> 4);
                b_begin[ldb * (kk + 1)] = scale_block * (x1 - zp_block);
            }
        }
    }

    static void DequantizeBlock(
        float* b_begin, size_t ldb, size_t actual_block_size, const uint8_t* data_block, float scale_block
    )
    {
        DequantizeBlock(b_begin, ldb, actual_block_size, data_block, scale_block, uint8_t{8});
    }

    static constexpr size_t BlkDataSizeInBytes()
    {
        return BlkLen * BlkBitWidth / 8;
    }

    static constexpr size_t ZeroPointsForBlksSizeInBytes(size_t BlkCount)
    {
        if constexpr (BlkBitWidth <= 4) {
            return DivRoundUp(BlkCount, 2);
        } else {
            return BlkCount;
        }
    }

    static size_t DivRoundUp(size_t a, size_t b) {
        return (a + b - 1) / b;
    }
};
