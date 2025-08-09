/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    q4_dq.cpp

Abstract:

    This module contains the data structures and implementations
    for blocked int4 quantization and dequantization.

    Int4 block quantization is used to compress weight tensors of large
    language models.

--*/

#include "q4common.h"

template<typename T>
constexpr
size_t
BlkQ4BufSize(size_t N, size_t K)
{
    const size_t KBlocks = MlasDivRoundup(K, T::BlkLen);
    return N * KBlocks * T::BlobSize;
}

size_t
MLASCALL
MlasQ4GemmPackBSize(MLAS_BLK_QUANT_TYPE QType, size_t N, size_t K)
{
    if (GetMlasPlatform().FpQ4GemmDispatch == nullptr) {
        return 0;
    }

    switch (QType) {
        case BlkQ4Sym:
            return BlkQ4BufSize<MLAS_Q4TYPE_BLK0>(N, K);
        case BlkQ4Sym64:
            return BlkQ4BufSize<MLAS_Q4TYPE_BLK2>(N, K);
        case BlkQ4Sym128:
            return BlkQ4BufSize<MLAS_Q4TYPE_BLK4>(N, K);
        default:
            return BlkQ4BufSize<MLAS_Q4TYPE_BLK1>(N, K);
    }
}


template<typename T>
MLAS_FORCEINLINE
void
MlasQ4GemmPackBImpl(void* PackedBuf, const float* FpData, size_t N, size_t K, size_t ldb)
{
    auto* dst_ptr = reinterpret_cast<uint8_t*>(PackedBuf);

    for (size_t n = 0; n < N; n ++) {
        const float* src = FpData; // starting from top of the column

        for (size_t k = 0; k < K; k += T::BlkLen) {
            size_t klen = std::min(size_t(T::BlkLen), K - k);
            float amax = 0.0f; // abs(max)
            float max = 0.0f;

            for (size_t l = 0; l < klen; l++) {
                const float v = src[ldb * l];
                if (amax < fabsf(v)) {
                    amax = fabsf(v);
                    max = v;
                }
            }

            const float scale = max / (-8);
            const float reciprocal_scale = scale ? 1.0f / scale : 0.0f;
            MlasQ4BlkScale<T>(dst_ptr) = scale;
            uint8_t* data = MlasQ4BlkData<T>(dst_ptr);

            for (size_t kk = 0; kk < klen; kk += 32) {
                size_t kklen = std::min((size_t)32, klen - kk);
                for (size_t l = 0; l < 16; l++) {
                    const float v0 = l < kklen ? src[ldb * (kk + l)] * reciprocal_scale : 0;
                    const uint8_t vi0 = (uint8_t)std::min(15.0f, std::max(0.0f, v0 + 8.5f));

                    const size_t l1 = l + 16;
                    const float v1 = (l1 < kklen) ? src[ldb * (kk + l1)] * reciprocal_scale : 0;
                    const uint8_t vi1 = (uint8_t)std::min(15.0f, std::max(0.0f, v1 + 8.5f));

                    data[l] = vi0 | (vi1 << 4);
                }
                data += 16;
            }

            // Move to next block of values in this column
            dst_ptr += T::BlobSize;
            src += ldb * klen;
        }

        FpData++; // move to next column
    }
}

template<>
MLAS_FORCEINLINE
void
MlasQ4GemmPackBImpl<MLAS_Q4TYPE_BLK1>(
    void* PackedBuf, const float* FpData, size_t N, size_t K, size_t ldb)
{
    auto* dst_ptr = reinterpret_cast<uint8_t*>(PackedBuf);

    for (size_t n = 0; n < N; n++) {
        const float* src = FpData; // starting from top of the column

        for (size_t k = 0; k < K; k += MLAS_Q4TYPE_BLK1::BlkLen) {
            size_t klen = std::min(MLAS_Q4TYPE_BLK1::BlkLen, K - k);
            float min = std::numeric_limits<float>::max();
            float max = -min;

            for (size_t l = 0; l < klen; l++) {
                const float v = src[ldb * l];
                if (v < min) min = v;
                if (v > max) max = v;
            }
            min = std::min(min, 0.0f);
            max = std::max(max, 0.0f);

            const float scale = (max - min) / ((1 << 4) - 1);
            const float reciprocal_scale = scale ? 1.0f / scale : 0.0f;
            float zero_point_fp = min;
            if (scale != 0.0f) {
                zero_point_fp = 0.f - min / scale;
            }

            // Handle any clamping
            uint8_t& zp = MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(dst_ptr);
            if (zero_point_fp < 0.0f) {
                zp = 0;
            } else if (zero_point_fp > 15.0f) {
                zp = 15;
            } else {
                zp = (uint8_t)roundf(zero_point_fp);
            }
            MlasQ4BlkScale<MLAS_Q4TYPE_BLK1>(dst_ptr) = scale;
            uint8_t* data = MlasQ4BlkData<MLAS_Q4TYPE_BLK1>(dst_ptr);

            for (size_t kk = 0; kk < klen; kk += 32) {
                size_t kklen = std::min((size_t)32, klen - kk);
                for (size_t l = 0; l < 16; l++) {
                    const float v0 = l < kklen ? src[ldb * (kk + l)] : 0;
                    const uint8_t vi0 = (uint8_t)std::min(
                        15.0f, std::max(0.0f, roundf(v0 * reciprocal_scale + zp)));

                    const size_t l1 = l + 16;
                    const float v1 = (l1 < kklen) ? src[ldb * (kk + l1)] : 0;
                    const uint8_t vi1 = (uint8_t)std::min(
                        15.0f, std::max(0.0f, roundf(v1 * reciprocal_scale + zp)));

                    data[l] = vi0 | (vi1 << 4);
                }
                data += 16;
            }
            // move to next block of values in this column
            dst_ptr += MLAS_Q4TYPE_BLK1::BlobSize;
            src += ldb * klen;
        }
        FpData++; // move to next column
    }
}

void
MLASCALL
MlasQ4GemmPackB(
    MLAS_BLK_QUANT_TYPE QType,
    void* PackedBuf,
    const float* FpData,
    size_t N,
    size_t K,
    size_t ldb
    )
{
    switch (QType) {
        case BlkQ4Sym:
            return MlasQ4GemmPackBImpl<MLAS_Q4TYPE_BLK0>(PackedBuf, FpData, N, K, ldb);
        case BlkQ4Sym64:
            return MlasQ4GemmPackBImpl<MLAS_Q4TYPE_BLK2>(PackedBuf, FpData, N, K, ldb);
        case BlkQ4Sym128:
            return MlasQ4GemmPackBImpl<MLAS_Q4TYPE_BLK4>(PackedBuf, FpData, N, K, ldb);
        default:
            return MlasQ4GemmPackBImpl<MLAS_Q4TYPE_BLK1>(PackedBuf, FpData, N, K, ldb);
    }
}

template<typename T>
MLAS_FORCEINLINE
void
MlasQ4GemmUnPackBImpl(float* FpData, const void* PackedBuf, size_t N, size_t K, size_t ldb)
{
    const auto* src = reinterpret_cast<const uint8_t*>(PackedBuf);
    for (size_t n = 0; n < N; n++) {
        for (size_t k = 0; k < K; k += T::BlkLen) {
            size_t CountK = std::min(K - k, T::BlkLen);

            float* dest = FpData + ldb * k + n;
            const float scale = MlasQ4BlkScale<T>(src);
            const uint8_t* data = MlasQ4BlkData<T>(src);

            for (size_t kk = 0; kk < CountK; kk += 32) {
                size_t kklen = std::min((size_t)32, CountK - kk);
                for (size_t l = 0; l < 16; l++) {
                    const uint8_t vi = data[l];

                    if (l < kklen) {
                        const int vi0 = (vi & 0x0F) - 8;
                        const float v0 = vi0 * scale;
                        dest[ldb * (kk + l)] = v0;
                    }

                    const size_t l1 = l + 16;
                    if (l1 < kklen) {
                        const int vi1 = (vi >> 4) - 8;
                        const float v1 = vi1 * scale;
                        dest[ldb * (kk + l1)] = v1;
                    }
                }
                data += 16;
            }
            src += T::BlobSize;
        }
    }
}

template<>
MLAS_FORCEINLINE
void
MlasQ4GemmUnPackBImpl<MLAS_Q4TYPE_BLK1>(
    float* FpData, const void* PackedBuf, size_t N, size_t K, size_t ldb)
{
    const auto* src = reinterpret_cast<const uint8_t*>(PackedBuf);
    for (size_t n = 0; n < N; n++) {
        for (size_t k = 0; k < K; k += MLAS_Q4TYPE_BLK1::BlkLen) {
            size_t CountK = std::min(K - k, MLAS_Q4TYPE_BLK1::BlkLen);

            float* dest = FpData + ldb * k + n;
            const float s = MlasQ4BlkScale<MLAS_Q4TYPE_BLK1>(src);
            const uint8_t z = MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(src);
            const uint8_t* pp = MlasQ4BlkData<MLAS_Q4TYPE_BLK1>(src);

            for (size_t kk = 0; kk < CountK; kk += 32) {
                size_t kklen = std::min((size_t)32, CountK - kk);
                for (size_t l = 0; l < 16; l++) {
                    const uint8_t vi = pp[l];

                    if (l < kklen) {
                        const int8_t vi0 = vi & 0x0F;
                        const float v0 = (vi0 - z) * s;
                        dest[ldb * (kk + l)] = v0;
                    }

                    size_t l1 = l + 16;
                    if (l1 < kklen) {
                        const int8_t vi1 = vi >> 4;
                        const float v1 = (vi1 - z) * s;
                        dest[ldb * (kk + l1)] = v1;
                    }
                }
                pp += 16;
            }
            src += MLAS_Q4TYPE_BLK1::BlobSize;
        }
    }
}

void
MLASCALL
MlasQ4GemmUnPackB(
    MLAS_BLK_QUANT_TYPE QType,
    float* FpData,
    const void* PackedBuf,
    size_t N,
    size_t K,
    size_t ldb
    )
{
    switch (QType) {
        case BlkQ4Sym:
            return MlasQ4GemmUnPackBImpl<MLAS_Q4TYPE_BLK0>(FpData, PackedBuf, N, K, ldb);
        case BlkQ4Sym64:
            return MlasQ4GemmUnPackBImpl<MLAS_Q4TYPE_BLK2>(FpData, PackedBuf, N, K, ldb);
        case BlkQ4Sym128:
            return MlasQ4GemmUnPackBImpl<MLAS_Q4TYPE_BLK4>(FpData, PackedBuf, N, K, ldb);
        default:
            return MlasQ4GemmUnPackBImpl<MLAS_Q4TYPE_BLK1>(FpData, PackedBuf, N, K, ldb);
    }
}



/***************************************************************
 * The quantization format that pack data and quantization
 * parameters into separate buffers.
 */


template <
    int Row_,    ///< rows of a matrix
    int Column_  ///< columns of a matrix
    >
struct Shape2D {
    static int const kRow = Row_;              ///< rows of a matrix
    static int const kColumn = Column_;        ///< columns of a matrix
    static int const kCount = Row_ * Column_;  ///< total number of elements in a matrix
};


template <int qbits, bool signed_quant>
struct BitsTraits {
    static_assert(qbits <= 8, "Only BitsTraits are for small number of bits!");

    static constexpr int kBits = qbits;
    static constexpr int kMax = signed_quant ? (1 << (qbits -1)) - 1 : (1 << qbits) - 1;
    static constexpr int kMid = signed_quant ? 0 : (1 << (qbits - 1));
    static constexpr int kMin = signed_quant ? -(1 << (qbits - 1)) : 0;
    static constexpr float kMaxFp = static_cast<float>(kMax);
    static constexpr float kMinFp = static_cast<float>(kMin);
    static constexpr float fullRange = kMaxFp - kMinFp;
    static constexpr float halfRange = static_cast<float>(kMid - kMin);

    // number of qbit elements to pack into whole bytes
    static constexpr int kPackSize = (qbits == 8) ? 1 : ((qbits == 4) ? 2 : ((qbits == 2) ? 4 : 0));
    static_assert(kPackSize != 0, "Packing to whole bytes not supported for this qbits!");
};


/**
 * @brief Rectify min/max from a set of weights, and convert to scale and zero point
 *        for quantization.
 * @tparam ScaleT        type of scale, usually floating point of various bits
 * @tparam qbits         number of int bits used for zero point value
 * @tparam signed_quant  output quantized type is signed
 * @param[in]   min
 * @param[in]   max
 * @param[out]  scale
 * @param[out]  zp
 */
template <typename ScaleT, int qbits, bool signed_quant>
MLAS_FORCEINLINE
void
range2scalezp(float min, float max, ScaleT& scale, uint8_t& zp)
{
    min = std::min(min, 0.0f);
    max = std::max(max, 0.0f);

    float scale_f = (max - min) / BitsTraits<qbits, signed_quant>::fullRange;

    float zero_point_fp = min;
    if (scale_f != 0.0f) {
        zero_point_fp = BitsTraits<qbits, signed_quant>::kMinFp - min / scale_f;
    }

    if (zero_point_fp < BitsTraits<qbits, signed_quant>::kMinFp) {
        zp = static_cast<uint8_t>(BitsTraits<qbits, signed_quant>::kMin);
    } else if (zero_point_fp > BitsTraits<qbits, signed_quant>::kMaxFp) {
        zp = static_cast<uint8_t>(BitsTraits<qbits, signed_quant>::kMax);
    } else {
        zp = (uint8_t)roundf(zero_point_fp);
    }
    scale = ScaleT(scale_f);
}

/**
 * @brief Rectify min/max from a set of symmetric weights, and convert
 *        to scale for quantization.
 */
template <typename ScaleT, int qbits, bool signed_quant>
MLAS_FORCEINLINE
void
range2scale(float min, float max, ScaleT& scale)
{
    max = fabsf(max) > fabsf(min) ? max : min;
    // !!Note: in the quantized space, abs of min -8 > abs of max 7.
    // Therefore map the larger half FP space to [-8, 0].
    // Minus sign achieves this purpose.
    scale = ScaleT(-max / BitsTraits<qbits, signed_quant>::halfRange);
};


/**
 * TODO(fajin): use int4/8 for symmetric quantization so the (vq - zp) operation in MatMulNBits can be saved.
 * @brief Blockwise quantization methods. Source is row major. Dest, scale and zp are column major.
 *        Always quantize to unsigned int.
 * @tparam ElementT       source data type, e.g. fp32/fp16
 * @tparam block_size     number of elemenets quantized together
 * @tparam qbits          number of bits in each quantized element
 * @tparam Columnwise     true:  quantize along src column, pack along src column.
 *                        false: quantize along src row, pack along src column.
 */
template <
    typename ElementT,
    int32_t block_size,
    int32_t qbits,
    bool Columnwise>
struct BlockwiseQuantizer {
    // To support other qbits, need to add bit packing code for
    // storing to dst and zero points
    static_assert(qbits == 2 || qbits == 4 || qbits == 8, "Only 2b, 4b and 8b block quantization is supported!");

    using QuantBlk = std::conditional_t<Columnwise, Shape2D<block_size, 1>, Shape2D<1, block_size>>;
    using ThreadBlk = Shape2D<QuantBlk::kRow * BitsTraits<qbits, false>::kPackSize, QuantBlk::kColumn>;

    static
    MLAS_FORCEINLINE
    int GetElem(int val, int idx)
    {
        return (val >> (qbits * idx)) & ((1 << qbits) - 1);
    }

    static
    MLAS_FORCEINLINE
    void quantizeMetaShape(int rows, int columns, int& meta_rows, int& meta_cols)
    {
        meta_rows = (rows + QuantBlk::kRow - 1) / QuantBlk::kRow;
        meta_cols = (columns + QuantBlk::kColumn - 1) / QuantBlk::kColumn;
    }

    static
    MLAS_FORCEINLINE
    void quantizedShape(int rows, int columns, int& q_rows, int& q_cols) {
        int meta_rows;
        int meta_cols;
        quantizeMetaShape(rows, columns, meta_rows, meta_cols);

        // quantized matrix is stored in column major, packed by column
        q_rows = (meta_rows * QuantBlk::kRow * qbits + 7) / 8;
        q_cols = meta_cols * QuantBlk::kColumn;
    }

    static MLAS_FORCEINLINE void quantizedBufferSizes(
        int rows, int columns, size_t& data_bytes, size_t& scale_num_elements, size_t* zero_point_bytes
    )
    {
        int meta_rows, meta_cols;
        quantizeMetaShape(rows, columns, meta_rows, meta_cols);
        int q_rows, q_cols;
        quantizedShape(rows, columns, q_rows, q_cols);

        data_bytes = q_rows * q_cols;
        scale_num_elements = meta_rows * meta_cols;

        if (zero_point_bytes) {
            // this works for qbits == 2, 4 or 8 but may need to be updated for other qbits values
            *zero_point_bytes = ((meta_rows * qbits + 7) / 8) * meta_cols;
        }
    }

    /**
     * @brief Quantized a Matrix shape [rows, columns], resulting quantized
     *        and packed data are stored in column major (transposed).
     * @param[out] dst           pointer to the quantized weights, column major: [columns, rows]
     * @param[out] scale         pointer to the scales, column major: [columns/QuantBlk::kColumn, rows/QuantBlk::kRow]
     * @param[out] zero_points   pointer to the zero points, same shape as scale
     * @param[in]  src           pointer to the source matrix, row major: [rows, columns]
     * @param rows
     * @param columns
     * @param leadingDimension   stride of the source matrix, i.e. distance from one row to the next
     */
    static void quantizeAndTranspose(
        uint8_t* dst,
        ElementT* scales,
        uint8_t* zero_points,
        const ElementT* src,
        int32_t rows,
        int32_t columns,
        int32_t leadingDimension,
        MLAS_THREADPOOL* thread_pool)
    {
        // Thread partitioning
        const auto thrd_row_blks = (rows + ThreadBlk::kRow - 1) / ThreadBlk::kRow;
        const auto thrd_col_blks = (columns + ThreadBlk::kColumn - 1) / ThreadBlk::kColumn;
        const auto total_thrd_blks = thrd_row_blks * thrd_col_blks;

        const auto row_blks = (rows + QuantBlk::kRow - 1) / QuantBlk::kRow;

        int q_rows, q_cols;
        quantizedShape(rows, columns, q_rows, q_cols);

        MlasTryBatchParallel(
            thread_pool, total_thrd_blks,
            [&](ptrdiff_t block_idx) {
                constexpr int kPackSize = BitsTraits<qbits, false>::kPackSize;
                uint8_t zp_bytes[kPackSize], vi[kPackSize];
                std::fill_n(zp_bytes, kPackSize, (uint8_t)BitsTraits<qbits, false>::kMid);
                std::fill_n(vi, kPackSize, 0);

                const int32_t r_blk_idx = static_cast<int32_t>(block_idx / thrd_col_blks);
                const int32_t c_blk_idx = static_cast<int32_t>(block_idx % thrd_col_blks);

                const int32_t r = r_blk_idx * ThreadBlk::kRow;
                const int32_t c = c_blk_idx * ThreadBlk::kColumn;

                const int32_t r_end = std::min(r + ThreadBlk::kRow, rows);
                const int32_t c_end = std::min(c + ThreadBlk::kColumn, columns);

                const int meta_row = r / QuantBlk::kRow;
                const int meta_col = c / QuantBlk::kColumn;

                // compute scale and zero point
                for (int kpack = 0; kpack < kPackSize; kpack++) {

                    // scan a single block to extract range [min, max]
                    float min = std::numeric_limits<float>::max();
                    float max = -min;
                    const int row_start = r + kpack * QuantBlk::kRow;
                    const int row_end = std::min(row_start + QuantBlk::kRow, r_end);
                    for (int i = row_start; i < row_end; ++i) {
                        for (int j = c; j < c_end; ++j) {
                            const float v = static_cast<float>(src[i * leadingDimension + j]);
                            if (v < min) min = v;
                            if (v > max) max = v;
                        }
                    }

                    // store scale and zero point at quant parameter matrix position
                    if (row_start < row_end) {
                        const int32_t meta_idx = meta_col * row_blks + meta_row + kpack;
                        if (zero_points == nullptr) {
                            range2scale<ElementT, qbits, false>(min, max, scales[meta_idx]);
                        } else {
                            range2scalezp<ElementT, qbits, false>(min, max, scales[meta_idx], zp_bytes[kpack]);
                        }
                    }
                }

                if (zero_points != nullptr) {
                    const int32_t meta_idx = meta_col * ((row_blks + kPackSize - 1) / kPackSize) + meta_row / kPackSize;
                    if constexpr (qbits == 8) {
                        zero_points[meta_idx] = zp_bytes[0];
                    } else if constexpr (qbits == 4) {
                        zero_points[meta_idx] = (zp_bytes[0] & 0xf) | (zp_bytes[1] << 4);
                    } else if constexpr (qbits == 2) {
                        zero_points[meta_idx] = (zp_bytes[0] & 0x3) | (zp_bytes[1] << 2) | (zp_bytes[2] << 4) | (zp_bytes[3] << 6);
                    } else {
                        MLAS_THROW_EX(std::runtime_error, "Unsupported qbits");
                    }
                }

                for (int32_t j = c; j < c_end; ++j) { // this does not work if j runs more then 1 because zp_bytes is indexed by i.
                    const int32_t meta_c = j / QuantBlk::kColumn;
                    for (int32_t i = r; i < r_end; i += kPackSize) {
                        for (int l = 0; l < kPackSize && i + l < r_end; l++) {
                            const int32_t meta_r = (i + l) / QuantBlk::kRow;
                            const float scale = static_cast<float>(scales[meta_c * row_blks + meta_r]);
                            const float reciprocal_scale = scale ? 1.0f / scale : 0.0f;
                            const int32_t zp = zp_bytes[meta_r % kPackSize];

                            const float v = static_cast<float>(src[(i + l) * leadingDimension + j]);
                            vi[l] = (uint8_t)std::clamp(roundf(v * reciprocal_scale + zp),
                                                        0.0f, BitsTraits<qbits, false>::kMaxFp);
                        }

                        if constexpr (qbits == 8) {
                            dst[j * q_rows + i / kPackSize] = vi[0];
                        } else if constexpr (qbits == 4) {
                            dst[j * q_rows + i / kPackSize] = (vi[0] & 0xf) | (vi[1] << 4);
                        } else if constexpr (qbits == 2) {
                            dst[j * q_rows + i / kPackSize] = (vi[0] & 0x3) | (vi[1] << 2) | (vi[2] << 4) | (vi[3] << 6);
                        } else {
                            MLAS_THROW_EX(std::runtime_error, "Unsupported qbits");
                        }
                    }
                }
            });
    }

    /**
     * @brief Dequantize a column major quantized matrix, and store the result in a column major
     * matrix for use in GEMM
     * @param[out] dst           pointer to the dequantized matrix, column major: [columns, rows]
     * @param[in]  weights       pointer to the quantized weights, column major: [columns, rows]
     * @param[in]  scales        pointer to the scales of quantized blocks, column major layout
     * @param[in]  zero_points   pointer to the zero points of quantized blocks, packed column major
     *                           scales
     * @param[in]  rows
     * @param[in]  columns
     */
    static void dequantize(
        ElementT* dst,
        const uint8_t* weights,
        const ElementT* scales,
        const uint8_t* zero_points,
        int32_t rows,
        int32_t columns,
        MLAS_THREADPOOL* thread_pool)
    {
        // Thread partitioning
        const auto thrd_row_blks = (rows + ThreadBlk::kRow - 1) / ThreadBlk::kRow;
        const auto thrd_col_blks = (columns + ThreadBlk::kColumn - 1) / ThreadBlk::kColumn;
        const auto total_thrd_blks = thrd_row_blks * thrd_col_blks;

        const auto row_blks = (rows + QuantBlk::kRow - 1) / QuantBlk::kRow;

        int q_rows, q_cols;
        quantizedShape(rows, columns, q_rows, q_cols);
        constexpr int32_t kPackSize = BitsTraits<qbits, false>::kPackSize;

        MlasTryBatchParallel(
            thread_pool, total_thrd_blks,
            [&](ptrdiff_t block_idx) {
                int32_t r_blk_idx = static_cast<int32_t>(block_idx / thrd_col_blks);
                int32_t c_blk_idx = static_cast<int32_t>(block_idx % thrd_col_blks);

                int32_t r = r_blk_idx * ThreadBlk::kRow;
                int32_t c = c_blk_idx * ThreadBlk::kColumn;

                int32_t r_end = std::min(r + ThreadBlk::kRow, rows);
                int32_t c_end = std::min(c + ThreadBlk::kColumn, columns);

                for (int32_t j = c; j < c_end; ++j) {
                    const int32_t meta_col = j / QuantBlk::kColumn;

                    for (int32_t i = r; i < r_end; ++i) {
                        const int32_t meta_row = i / QuantBlk::kRow;
                        const float scale = static_cast<float>(scales[meta_col * row_blks + meta_row]);
                        const int zp_pair =
                            zero_points
                            ? zero_points[meta_col * ((row_blks + kPackSize - 1) / kPackSize) + meta_row / kPackSize]
                            : 0;
                        const int vi_pair = weights[j * q_rows + i / kPackSize];

                        const int zp =
                            zero_points
                                ? GetElem(zp_pair, meta_row % kPackSize)
                                : BitsTraits<qbits, false>::kMid;
                        const int vi = GetElem(vi_pair, i % kPackSize);
                        const float v = (vi - zp) * scale;
                        dst[j * rows + i] = ElementT(v);
                    }
                }
            });
    }
};

/**
 * @brief Blockwise quantization methods for QDQ format. Input tensor is quantized along column
 *        or row. Scales and zeros are calculated. Based on qbits, consecutive quantized elements
 *        in memory are packed together, which means the packing is along the row. Quantized data
 *        are stored in row major, so the output tensor reserves same shape, in terms of qbits type,
 *        as the input tensor.
 *        If has zero points, quantized type is unsigned. Otherwise, quantized type is signed and the
 *        zero point is 0.
 *        The transposed outputs are used by MatMulNBits, so quant type becomes uint4 with default
 *        zp at 8.
 * @tparam Tin           source data type, e.g. fp32/fp16
 * @tparam qbits         number of bits in each quantized element
 * @tparam signed_quant  quantized type is signed
 */
template <typename Tin, int qbits, bool signed_quant>
struct BlockwiseQDQQuantizer {
    static MLAS_FORCEINLINE uint8_t GetElem(uint8_t val, int32_t idx)
    {
        if constexpr (qbits == 2) {
            return (val >> (idx << 1)) & 0x3;
        } else if constexpr (qbits == 4) {
            return (val >> (idx << 2)) & 0xF;
        }
    }

    static MLAS_FORCEINLINE uint8_t SetElem(uint8_t val, int32_t idx, uint8_t dst)
    {
        if constexpr (qbits == 2) {
            auto shift = idx << 1;
            return ((val & 0x3) << shift) | (dst & (~(0x3 << shift)));
        } else if constexpr (qbits == 4) {
            auto shift = idx << 2;
            return ((val & 0xF) << shift) | (dst & (~(0xF << shift)));
        }
    }

    template <bool add2>
    static MLAS_FORCEINLINE uint8_t Pack(uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3)
    {
          if constexpr (add2) {
            return ((v0 & 0x3) ^ 2) | (((v1 & 0x3) ^ 2) << 2) | (((v2 & 0x3) ^ 2) << 4) | (((v3 & 0x3) ^ 2) << 6);
          } else {
              return (v0 & 0x3) | ((v1 & 0x3) << 2) | ((v2 & 0x3) << 4) | ((v3 & 0x3) << 6);
          }
    }

    template <bool add8>
    static MLAS_FORCEINLINE uint8_t Pack(uint8_t v0, uint8_t v1)
    {
        if constexpr (add8) {
            return ((v0 & 0xF) ^ 8) | (((v1 & 0xF) ^ 8) << 4);
        } else {
            return (v0 & 0xF) | ((v1 & 0xF) << 4);
        }
    }

    // If src is row major, then dst is column major. Transpose:
    //  | src0: low 4 bit | src0: high 4 bit |
    //  | src1: low 4 bit | src1: high 4 bit |
    //  -->
    //  | dst0: low 4 bit | dst1: low 4 bit  |
    //  | dst0: high 4 bit| dst1: high 4 bit |
    // If src is column major, then dst is row major. Transpose:
    //  | src0: low 4 bit | src1: low 4 bit  |
    //  | src0: high 4 bit| src1: high 4 bit |
    //  -->
    //  | dst0: low 4 bit | dst0: high 4 bit |
    //  | dst1: low 4 bit | dst1: high 4 bit |
    template <bool add8>
    static MLAS_FORCEINLINE void Transpose(uint8_t src0, uint8_t src1, uint8_t& dst0, uint8_t& dst1)
    {
        if constexpr (add8) {
            dst0 = ((src0 & 0xF) ^ 8) | (((src1 & 0xF) ^ 8) << 4);
            dst1 = (((src0 & 0xF0) ^ 0x80) >> 4) | ((src1 & 0xF0) ^ 0x80);
        } else {
            dst0 = (src0 & 0xF) | ((src1 & 0xF) << 4);
            dst1 = ((src0 & 0xF0) >> 4) | (src1 & 0xF0);
        }
    }

    static MLAS_FORCEINLINE uint8_t QuantizeV(Tin src, float reciprocal_scale, uint8_t zero_point)
    {
        return static_cast<uint8_t>(
            std::clamp(
                static_cast<int32_t>(
                    std::roundf(static_cast<float>(src) * reciprocal_scale)
                ) + static_cast<int32_t>(zero_point),
                BitsTraits<4, signed_quant>::kMin,
                BitsTraits<4, signed_quant>::kMax
            )
        );
    }

    /**
     * @brief Quantize a matrix shape [rows, columns] column-wise. Scales and zero points are calculated.
     *        Quantized data are packed row-wise based on qbits. Quantized data are stored in row major
     *        so the output tensor reserves the shape, in terms output type.
     * @param src               the source matrix, row major: [rows * columns]
     * @param scales            the scales of quantized blocks, row major with shape:
     *                          [ceil(rows/quant_block_size) * columns]
     * @param zero_points       the zero points of quantized blocks, packed. Same shape as scales in terms
     *                          of output type. In uint8_t, the shape is:
     *                          [ceil(columns * ceil(rows / quant_block_size) * qbits / 8)]
     * @param dst               the quantized weights, row major: [rows * columns] in terms of output type.
     *                          In uint8_t, the shape is: [ceil(rows * columns * qbits / 8]
     * @param rows              number of rows in the source matrix
     * @param columns           number of columns in the source matrix.
     * @param quant_block_size  number of rows/columns quantized together
     * @param thread_pool       thread pool for parallel processing
     */
    static void QuantizeColumnWise(
        const Tin* src,
        Tin* scales,
        uint8_t* zero_points,
        uint8_t* dst,
        int32_t rows,
        int32_t columns,
        int32_t quant_block_size,
        MLAS_THREADPOOL* thread_pool
    )
    {
        ORT_ENFORCE(zero_points || signed_quant, "Unsigned quant with no zero points is not supported.");
        // Must avoid multiple thread write to a single byte, which means the starting index
        // of a thread block must be even. To achieve that, we need to customize the thread
        // block size based on the parity of columns.
        if (columns & 1) {
            QuantizeColumnWisePackUnaligned(
                src, scales, zero_points, dst, rows, columns, quant_block_size, thread_pool
            );
        } else {
            QuantizeColumnWisePackAligned(
                src, scales, zero_points, dst, rows, columns, quant_block_size, thread_pool
            );
        }
    }


    /**
     * @brief Transpose quantized tensors, which has been column-wise quantized, for use in MatMulNbits.
     *        Since both src tensor and dst tensor are packed, it's not needed to consider sign
     *        during the unpacking/packing in transpose.
     * @param src_weights       The quantized weights, row major: [rows, columns] in qbits type.
     *                          In uint8_t, size of [ceil(rows * columns * qbits / 8)].
     * @param src_scales        [ceil(rows / quant_block_size), columns]
     * @param src_zero_points   [ceil(rows / quant_block_size), columns] in qbits type. In uint8_t, size of
     *                          [ceil(ceil(rows / quant_block_size) * columns * qbits / 8 )].
     * @param dst_weights       the transposed quantized weights, column major. In uint8_t, the shape is
     *                          [columns, ceil(rows / quant_block_size), ceil(quant_block_size * qbits / 8)]
     * @param dst_scales        [columns, ceil(rows / quant_block_size)]
     * @param dst_zero_points   [columns, ceil(ceil(rows / quant_block_size) * qbits / 8)] in uint8_t.
     * @param rows              number of src rows in qbits type.
     * @param columns           number of src columns in qbits type.
     * @param quant_block_size  number of elements quantized together
     * @param thread_pool       thread pool for parallel processing
     */
    static void TransposeColumnWiseQuantized(
        const uint8_t* src_weights,
        const Tin* src_scales,
        const uint8_t* src_zero_points,
        uint8_t* dst_weights,
        Tin* dst_scales,
        uint8_t* dst_zero_points,
        int32_t rows,
        int32_t columns,
        int32_t quant_block_size,
        MLAS_THREADPOOL* thread_pool
    )
    {
        ORT_ENFORCE(
            src_zero_points || signed_quant || dst_zero_points,
            "Unsigned quant types without zero points must allocate zero points with value 0."
        );
        // Must avoid multiple thread write to a single byte, which means the starting index
        // of a thread block must be even. To achieve that, we need to customize the thread
        // block size based on the parity of columns.
        if (columns & 1) {
            TransposeColumnWiseQuantizedPackUnaligned(
                src_weights, src_scales, src_zero_points,
                dst_weights, dst_scales, dst_zero_points,
                rows, columns, quant_block_size, thread_pool
            );
        } else {
            TransposeColumnWiseQuantizedPackAligned(
                src_weights, src_scales, src_zero_points,
                dst_weights, dst_scales, dst_zero_points,
                rows, columns, quant_block_size, thread_pool
            );
        }
    }

private:
    static void QuantizeColumnWisePackAligned(
        const Tin* src,
        Tin* scales,
        uint8_t* zero_points,
        uint8_t* dst,
        int32_t rows,
        int32_t columns,
        int32_t quant_block_size,
        MLAS_THREADPOOL* thread_pool
    )
    {
        ORT_ENFORCE(columns % 2 == 0, "Columns must be multiple of 2.");
        // Thread block is [quant_block_size, thread_blk_size]. thread_blk_size % 2 == 0.
        constexpr int32_t thread_blk_size = 128;
        const auto num_row_thread_blk = (rows + quant_block_size - 1) / quant_block_size;
        const auto num_col_thread_blk = (columns + thread_blk_size - 1) / thread_blk_size;
        const auto num_thread_blk = num_row_thread_blk * num_col_thread_blk;
        constexpr auto minf = std::numeric_limits<float>::lowest();
        constexpr auto maxf = std::numeric_limits<float>::max();

        MlasTryBatchParallel(
            thread_pool, static_cast<ptrdiff_t>(num_thread_blk),
            [&](ptrdiff_t thread_blk_idx) {
                // !!warning!!: buffering the whole thread block
                constexpr int32_t buffer_size = 128;
                ORT_ENFORCE(buffer_size == thread_blk_size, "buffer size must be equal to thread block size.");
                float reciprocal_scale_t[buffer_size];
                uint8_t zp_t[buffer_size];
                float vmin_t[buffer_size];
                float vmax_t[buffer_size];

                const int32_t row_thread_blk_idx = static_cast<int32_t>(thread_blk_idx / num_col_thread_blk);
                const int32_t col_thread_blk_idx = static_cast<int32_t>(thread_blk_idx % num_col_thread_blk);
                const int32_t row_idx = row_thread_blk_idx * quant_block_size;
                const int32_t col_idx = col_thread_blk_idx * buffer_size;
                const int32_t row_size = std::min(quant_block_size, rows - row_idx);
                const int32_t col_size = std::min(buffer_size, columns - col_idx);
                // input_idx, scale_idx, col_size are aligned to 2
                auto input_idx = row_idx * columns + col_idx;
                auto scale_idx = row_thread_blk_idx * columns + col_idx;

                Tin scale0_tt, scale1_tt;
                uint8_t v0_tt, v1_tt;

                std::fill_n(vmin_t, buffer_size, maxf);
                std::fill_n(vmax_t, buffer_size, minf);

                // calculate min/max
                for (int32_t j = 0, input_idx_t = input_idx; j < row_size; ++j, input_idx_t += columns) {
                    // TODO(fajin): use SIMD
                    for (int32_t i = 0; i < col_size; i += 2) {
                        auto v0 = static_cast<float>(src[input_idx_t + i]);
                        auto v1 = static_cast<float>(src[input_idx_t + i + 1]);
                        vmin_t[i] = std::min(vmin_t[i], v0);
                        vmax_t[i] = std::max(vmax_t[i], v0);
                        vmin_t[i + 1] = std::min(vmin_t[i + 1], v1);
                        vmax_t[i + 1] = std::max(vmax_t[i + 1], v1);
                    }
                }

                // calculate scale and zero point, and store
                for (int32_t i = 0; i < col_size; i += 2) {
                    v0_tt = v1_tt = BitsTraits<4, signed_quant>::kMid;

                    if (zero_points) {
                        range2scalezp<Tin, 4, signed_quant>(vmin_t[i], vmax_t[i], scale0_tt, v0_tt);
                        range2scalezp<Tin, 4, signed_quant>(vmin_t[i + 1], vmax_t[i + 1], scale1_tt, v1_tt);
                        zero_points[(scale_idx + i) >> 1] = Pack<false>(v0_tt, v1_tt);
                    } else {
                        range2scale<Tin, 4, signed_quant>(vmin_t[i], vmax_t[i], scale0_tt);
                        range2scale<Tin, 4, signed_quant>(vmin_t[i + 1], vmax_t[i + 1], scale1_tt);
                    }

                    scales[scale_idx + i] = scale0_tt;
                    scales[scale_idx + i + 1] = scale1_tt;

                    float scalef0 = static_cast<float>(scale0_tt);
                    reciprocal_scale_t[i] = scalef0 ? 1.0f / scalef0 : 0.0f;
                    zp_t[i] = v0_tt;

                    float scalef1 = static_cast<float>(scale1_tt);
                    reciprocal_scale_t[i + 1] = scalef1 ? 1.0f / scalef1 : 0.0f;
                    zp_t[i + 1] = v1_tt;
                }

                // quantize and pack
                for (int32_t j = 0, input_idx_t = input_idx; j < row_size; ++j, input_idx_t += columns) {
                    // TODO(fajin): use SIMD
                    for (int32_t i = 0; i < col_size; i += 2) {
                        v0_tt = QuantizeV(src[input_idx_t + i], reciprocal_scale_t[i], zp_t[i]);
                        v1_tt = QuantizeV(src[input_idx_t + i + 1], reciprocal_scale_t[i + 1], zp_t[i + 1]);
                        dst[(input_idx_t + i) >> 1] = Pack<false>(v0_tt, v1_tt);
                    }
                }
            }
        );
    }

    static void QuantizeColumnWisePackUnaligned(
        const Tin* src,
        Tin* scales,
        uint8_t* zero_points,
        uint8_t* dst,
        int32_t rows,
        int32_t columns,
        int32_t quant_block_size,
        MLAS_THREADPOOL* thread_pool
    )
    {
        // Thread block is [quant_block_size * 2, columns], so the packed bytes do not cross threads.
        constexpr auto minf = std::numeric_limits<float>::lowest();
        constexpr auto maxf = std::numeric_limits<float>::max();
        auto row_thread_blk_size = quant_block_size * 2;
        auto num_row_thread_blk = (rows + row_thread_blk_size - 1) / (row_thread_blk_size);

        MlasTryBatchParallel(
            thread_pool, static_cast<ptrdiff_t>(num_row_thread_blk),
            [&](ptrdiff_t thread_blk_idx) {
                constexpr int32_t buffer_size = 128;
                float reciprocal_scale_t[buffer_size];
                uint8_t zp_t[buffer_size];
                float vmin_t[buffer_size];
                float vmax_t[buffer_size];

                auto row_thread_blk_idx = static_cast<int32_t>(thread_blk_idx);
                int32_t row_idx = row_thread_blk_idx * row_thread_blk_size;
                int32_t row_idx_end = std::min(row_thread_blk_size + row_idx, rows);
                auto input_idx = row_idx * columns;
                auto scale_idx = row_thread_blk_idx * 2 * columns;
                Tin scale0_tt, scale1_tt;
                uint8_t v0_tt, v1_tt;

                for (; row_idx < row_idx_end; row_idx += quant_block_size) {
                    // per quant block row
                    auto quant_row_size = std::min(quant_block_size, row_idx_end - row_idx);
                    auto input_buffer_idx = input_idx;
                    auto scale_buffer_idx = scale_idx;
                    for (int32_t buffer_idx = 0; buffer_idx < columns; buffer_idx += buffer_size) {
                        // per buffer column
                        auto buffer_col_size = std::min(buffer_size, columns - buffer_idx);

                        std::fill_n(vmin_t, buffer_size, maxf);
                        std::fill_n(vmax_t, buffer_size, minf);
                        // calculate min/max of [quant block, buffer]
                        auto input_idx_t = input_buffer_idx;
                        for (int32_t j = 0; j < quant_row_size; ++j, input_idx_t += columns) {
                            // TODO(fajin): use SIMD
                            for (int32_t i = 0; i < buffer_col_size; ++i) {
                                auto v = static_cast<float>(src[input_idx_t + i]);
                                vmin_t[i] = std::min(vmin_t[i], v);
                                vmax_t[i] = std::max(vmax_t[i], v);
                            }
                        }

                        // calculate scale and zero point
                        auto scale_buffer_idx_end = scale_buffer_idx + buffer_col_size;
                        int32_t col_idx = 0;
                        // leading unailgned zero points
                        if (scale_buffer_idx & 1) {
                            v0_tt = BitsTraits<4, signed_quant>::kMid;
                            if (zero_points) {
                                range2scalezp<Tin, 4, signed_quant>(vmin_t[0], vmax_t[0], scale0_tt, v0_tt);
                                zero_points[scale_buffer_idx >> 1] = SetElem(
                                    v0_tt, 1, zero_points[scale_buffer_idx >> 1]
                                );
                            } else {
                                range2scale<Tin, 4, signed_quant>(vmin_t[0], vmax_t[0], scale0_tt);
                            }

                            scales[scale_buffer_idx] = scale0_tt;

                            float scalef = static_cast<float>(scale0_tt);
                            reciprocal_scale_t[0] = scalef ? 1.0f / scalef : 0.0f;
                            zp_t[0] = v0_tt;

                            ++col_idx;
                            ++scale_buffer_idx;
                        }
                        // aligned zero points
                        for (; scale_buffer_idx < scale_buffer_idx_end - 1; col_idx += 2, scale_buffer_idx += 2) {
                            v0_tt = v1_tt = BitsTraits<4, signed_quant>::kMid;
                            if (zero_points) {
                                range2scalezp<Tin, 4, signed_quant>(vmin_t[col_idx], vmax_t[col_idx], scale0_tt, v0_tt);
                                range2scalezp<Tin, 4, signed_quant>(
                                    vmin_t[col_idx + 1], vmax_t[col_idx + 1], scale1_tt, v1_tt
                                );
                                zero_points[scale_buffer_idx >> 1] = Pack<false>(v0_tt, v1_tt);
                            } else {
                                range2scale<Tin, 4, signed_quant>(vmin_t[col_idx], vmax_t[col_idx], scale0_tt);
                                range2scale<Tin, 4, signed_quant>(vmin_t[col_idx + 1], vmax_t[col_idx + 1], scale1_tt);
                            }

                            scales[scale_buffer_idx] = scale0_tt;
                            scales[scale_buffer_idx + 1] = scale1_tt;

                            float scalef0 = static_cast<float>(scale0_tt);
                            reciprocal_scale_t[col_idx] = scalef0 ? 1.0f / scalef0 : 0.0f;
                            zp_t[col_idx] = v0_tt;

                            float scalef1 = static_cast<float>(scale1_tt);
                            reciprocal_scale_t[col_idx + 1] = scalef1 ? 1.0f / scalef1 : 0.0f;
                            zp_t[col_idx + 1] = v1_tt;
                        }
                        // tailing unaligned elements
                        if (scale_buffer_idx < scale_buffer_idx_end) {
                            v0_tt = BitsTraits<4, signed_quant>::kMid;
                            if (zero_points) {
                                range2scalezp<Tin, 4, signed_quant>(vmin_t[col_idx], vmax_t[col_idx], scale0_tt, v0_tt);
                                zero_points[scale_buffer_idx >> 1] = SetElem(
                                    v0_tt, 0, zero_points[scale_buffer_idx >> 1]
                                );
                            } else {
                                range2scale<Tin, 4, signed_quant>(vmin_t[col_idx], vmax_t[col_idx], scale0_tt);
                            }

                            scales[scale_buffer_idx] = scale0_tt;

                            float scalef = static_cast<float>(scale0_tt);
                            reciprocal_scale_t[col_idx] = scalef ? 1.0f / scalef : 0.0f;
                            zp_t[col_idx] = v0_tt;

                            ++scale_buffer_idx;
                        }

                        // quantize and pack
                        input_idx_t = input_buffer_idx;
                        for (int32_t j = 0; j < quant_row_size; ++j, input_idx_t += columns) {
                            auto input_idx_t_start = input_idx_t;
                            auto input_idx_t_end = input_idx_t + buffer_col_size;
                            col_idx = 0;
                            // leading unaligned output
                            if (input_idx_t_start & 1) {
                                v1_tt = QuantizeV(src[input_idx_t_start], reciprocal_scale_t[col_idx], zp_t[col_idx]);
                                dst[input_idx_t_start >> 1] = SetElem(v1_tt, 1, dst[input_idx_t_start >> 1]);

                                ++col_idx;
                                ++input_idx_t_start;
                            }
                            // aligned output
                            // TODO(fajin): use SIMD
                            for (; input_idx_t_start < input_idx_t_end - 1; col_idx += 2, input_idx_t_start += 2) {
                                v0_tt = QuantizeV(src[input_idx_t_start], reciprocal_scale_t[col_idx], zp_t[col_idx]);
                                v1_tt = QuantizeV(
                                    src[input_idx_t_start + 1], reciprocal_scale_t[col_idx + 1], zp_t[col_idx + 1]
                                );

                                dst[input_idx_t_start >> 1] = Pack<false>(v0_tt, v1_tt);
                            }
                            // tailing unaligned output
                            if (input_idx_t_start < input_idx_t_end) {
                                v0_tt = QuantizeV(src[input_idx_t_start], reciprocal_scale_t[col_idx], zp_t[col_idx]);
                                dst[input_idx_t_start >> 1] = SetElem(v0_tt, 0, dst[input_idx_t_start >> 1]);
                            }
                        }

                        input_buffer_idx += buffer_size;
                    }

                    input_idx += quant_block_size * columns;
                    scale_idx += columns;
                }
            }
        );
    }

    static void TransposeColumnWiseQuantizedPackAligned(
        const uint8_t* src_weights,      // [rows, columns / 2]
        const Tin* src_scales,           // [ceil(rows / quant_block_size), columns]
        const uint8_t* src_zero_points,  // [ceil(rows / quant_block_size), columns / 2]
        uint8_t* dst_weights,            // [columns, ceil(rows / quant_block_size), ceil(quant_block_size / 2)]
        Tin* dst_scales,                 // [columns, ceil(rows / quant_block_size)]
        uint8_t* dst_zero_points,        // [columns, ceil(ceil(rows / quant_block_size) / 2)]
        int32_t rows,
        int32_t columns,
        int32_t quant_block_size,
        MLAS_THREADPOOL* thread_pool
    )
    {
        ORT_ENFORCE(columns % 2 == 0, "Columns must be multiple of 2");

        auto row_quant_blk_num = (rows + quant_block_size - 1) / quant_block_size;
        auto dst_bytes_per_quant_blk = (quant_block_size * 4 + 7) / 8;
        // number of rows in transposed dst
        auto dstT_num_row = row_quant_blk_num * dst_bytes_per_quant_blk;
        auto packed_col_size = columns / 2;

        // weight transpose thread block is [dst_bytes_per_quant_blk, 2] on dst_Transpose.
        // Map to src it is [quant_block_size, 1]. Both in uint8_t.
        auto num_thread_blk = row_quant_blk_num * packed_col_size;
        MlasTryBatchParallel(
            thread_pool, static_cast<ptrdiff_t>(num_thread_blk),
            [&](ptrdiff_t thread_blk_idx) {
                uint8_t src0_t, src1_t;
                uint8_t dst0_t, dst1_t;

                auto row_thread_blk_idx = static_cast<int32_t>(thread_blk_idx / packed_col_size);
                auto col_thread_blk_idx = static_cast<int32_t>(thread_blk_idx % packed_col_size);

                auto dstT_row_idx = row_thread_blk_idx * dst_bytes_per_quant_blk;
                auto dstT_col_idx = col_thread_blk_idx * 2;
                auto dst_idx = dstT_col_idx * dstT_num_row + dstT_row_idx;

                auto src_row_idx = row_thread_blk_idx * quant_block_size;
                auto src_row_end_idx = std::min(src_row_idx + quant_block_size, rows);
                auto src_col_idx = col_thread_blk_idx;
                auto src_idx = src_row_idx * packed_col_size + src_col_idx;
                auto src_end_idx = src_row_end_idx * packed_col_size + src_col_idx;

                for (; src_idx < src_end_idx - packed_col_size; ++dst_idx) {
                    src0_t = src_weights[src_idx];
                    src1_t = src_weights[src_idx + packed_col_size];
                    src_idx += packed_col_size + packed_col_size;
                    Transpose<signed_quant>(src0_t, src1_t, dst0_t, dst1_t);
                    dst_weights[dst_idx] = dst0_t;
                    dst_weights[dst_idx + dstT_num_row] = dst1_t;
                }

                if (src_idx < src_end_idx) {
                    src0_t = src_weights[src_idx];
                    src1_t = 0;
                    Transpose<signed_quant>(src0_t, src1_t, dst0_t, dst1_t);
                    dst_weights[dst_idx] = dst0_t;
                    dst_weights[dst_idx + dstT_num_row] = dst1_t;
                }
            }
        );

        // Transpose scales. Thread block is [row_quant_blk_num, 1] on dst_Transpose.
        MlasTryBatchParallel(
            thread_pool, static_cast<ptrdiff_t>(columns),
            [&](ptrdiff_t thread_blk_idx) {
                auto col_thread_blk_idx = static_cast<int32_t>(thread_blk_idx);
                auto src_idx = col_thread_blk_idx;
                auto dst_idx = col_thread_blk_idx * row_quant_blk_num;
                for (int32_t i = 0; i < row_quant_blk_num; ++i, ++dst_idx, src_idx += columns) {
                    dst_scales[dst_idx] = src_scales[src_idx];
                }
            }
        );

        if (src_zero_points) {
            // Transpose zero points. Thread block is [ceil(row_quant_blk_num / 2), 2]
            // on dst_Transpose. Map to src it is [row_quant_blk_num, 1]. Both in uint8_t.
            auto dst_zp_row_num = (row_quant_blk_num + 1) / 2;
            MlasTryBatchParallel(
                thread_pool, static_cast<ptrdiff_t>(packed_col_size),
                [&](ptrdiff_t thread_blk_idx) {
                    uint8_t src0_t, src1_t;
                    uint8_t dst0_t, dst1_t;

                    auto col_thread_blk_idx = static_cast<int32_t>(thread_blk_idx);
                    auto src_idx = col_thread_blk_idx;
                    auto src_end_idx = row_quant_blk_num * packed_col_size + col_thread_blk_idx;
                    auto dst_idx = col_thread_blk_idx * 2 * dst_zp_row_num;

                    for (; src_idx < src_end_idx - packed_col_size; ++dst_idx) {
                        src0_t = src_zero_points[src_idx];
                        src1_t = src_zero_points[src_idx + packed_col_size];
                        Transpose<signed_quant>(src0_t, src1_t, dst0_t, dst1_t);
                        dst_zero_points[dst_idx] = dst0_t;
                        dst_zero_points[dst_idx + dst_zp_row_num] = dst1_t;
                        src_idx += packed_col_size + packed_col_size;
                    }

                    if (src_idx < src_end_idx) {
                        src0_t = src_zero_points[src_idx];
                        src1_t = 0;
                        Transpose<signed_quant>(src0_t, src1_t, dst0_t, dst1_t);
                        dst_zero_points[dst_idx] = dst0_t;
                        dst_zero_points[dst_idx + dst_zp_row_num] = dst1_t;
                    }
                }
            );
        }
    }

    static void TransposeColumnWiseQuantizedPackUnaligned(
      const uint8_t* src_weights,       // size of [ceil(rows * columns / 2)]
      const Tin* src_scales,            // [ceil(rows / quant_block_size), columns]
      const uint8_t* src_zero_points,   // size of [ceil(ceil(rows / quant_block_size) * columns / 2)]
      uint8_t *dst_weights,             // [columns, ceil(rows / quant_block_size), ceil(quant_block_size / 2)]
      Tin* dst_scales,                  // [columns, ceil(rows / quant_block_size)]
      uint8_t* dst_zero_points,         // [columns, ceil(ceil(rows / quant_block_size) / 2)]
      int32_t rows,
      int32_t columns,
      int32_t quant_block_size,
      MLAS_THREADPOOL* thread_pool)
    {
        auto row_quant_blk_num = (rows + quant_block_size - 1) / quant_block_size;
        auto dst_bytes_per_quant_blk = (quant_block_size * 4 + 7) / 8;
        // number of rows in transposed dst
        auto dstT_num_row = row_quant_blk_num * dst_bytes_per_quant_blk;

        // weight transpose thread block is [dst_bytes_per_quant_blk, 1] on dst_Transpose in uint8_t.
        // Map to src it is [quant_block_size, 1] in int4.
        auto num_thread_blk = row_quant_blk_num * columns;
        MlasTryBatchParallel(
            thread_pool, static_cast<ptrdiff_t>(num_thread_blk),
            [&](ptrdiff_t thread_blk_idx) {
                uint8_t src0_t, src1_t;

                auto row_thread_blk_idx = static_cast<int32_t>(thread_blk_idx / columns);
                auto col_thread_blk_idx = static_cast<int32_t>(thread_blk_idx % columns);

                auto dstT_row_idx = row_thread_blk_idx * dst_bytes_per_quant_blk;
                auto dst_idx = col_thread_blk_idx * dstT_num_row + dstT_row_idx;

                auto src_row_idx = row_thread_blk_idx * quant_block_size;
                auto src_row_end_idx = std::min(src_row_idx + quant_block_size, rows);
                auto src_idx = src_row_idx * columns + col_thread_blk_idx;
                auto src_end_idx = src_row_end_idx * columns + col_thread_blk_idx;

                for (; src_idx < src_end_idx - columns; ++dst_idx) {
                    src0_t = GetElem(src_weights[src_idx >> 1], src_idx & 1);
                    src1_t = GetElem(src_weights[(src_idx + columns) >> 1], (src_idx + columns) & 1);
                    dst_weights[dst_idx] = Pack<signed_quant>(src0_t, src1_t);
                    src_idx += columns + columns;
                }

                if (src_idx < src_end_idx) {
                    src0_t = GetElem(src_weights[src_idx >> 1], src_idx & 1);
                    dst_weights[dst_idx] = Pack<signed_quant>(src0_t, 0);
                }
            }
        );

        // Transpose scales. Thread block is [row_quant_blk_num, 1] on dst_Transpose.
        MlasTryBatchParallel(
            thread_pool, static_cast<ptrdiff_t>(columns),
            [&](ptrdiff_t thread_blk_idx) {
                auto col_thread_blk_idx = static_cast<int32_t>(thread_blk_idx);
                auto src_idx = col_thread_blk_idx;
                auto dst_idx = col_thread_blk_idx * row_quant_blk_num;
                for (int32_t i = 0; i < row_quant_blk_num; ++i, ++dst_idx, src_idx += columns) {
                    dst_scales[dst_idx] = src_scales[src_idx];
                }
            }
        );

        if (src_zero_points) {
            // Transpose zero points. Thread block is [ceil(row_quant_blk_num / 2), 1] on dst_Transpose in uint8_t.
            // Map to src it is [row_quant_blk_num, 1] in int4.
            auto dst_zp_row_num = (row_quant_blk_num + 1) / 2;
            MlasTryBatchParallel(
                thread_pool, static_cast<ptrdiff_t>(columns),
                [&](ptrdiff_t thread_blk_idx) {
                    uint8_t src0_t, src1_t;

                    auto col_thread_blk_idx = static_cast<int32_t>(thread_blk_idx);
                    auto src_idx = col_thread_blk_idx;
                    auto src_end_idx = row_quant_blk_num * columns + col_thread_blk_idx;
                    auto dst_idx = col_thread_blk_idx * dst_zp_row_num;

                    for (; src_idx < src_end_idx - columns; ++dst_idx) {
                        src0_t = GetElem(src_zero_points[src_idx >> 1], src_idx & 1);
                        src1_t = GetElem(src_zero_points[(src_idx + columns) >> 1], (src_idx + columns) & 1);
                        dst_zero_points[dst_idx] = Pack<signed_quant>(src0_t, src1_t);
                        src_idx += columns + columns;
                    }

                    if (src_idx < src_end_idx) {
                        src0_t = GetElem(src_zero_points[src_idx >> 1], src_idx & 1);
                        dst_zero_points[dst_idx] = Pack<signed_quant>(src0_t, 0);
                    }
                }
            );
        }
    }
};

template <typename T, int qbits>
void
MlasBlockwiseQuantMetaShape(
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    int& meta_rows,
    int& meta_cols
    )
{
    switch (block_size) {
        case 16: {
            if (columnwise) {
                BlockwiseQuantizer<T, 16, qbits, true>::quantizeMetaShape(rows, columns, meta_rows, meta_cols);
            } else {
                BlockwiseQuantizer<T, 16, qbits, false>::quantizeMetaShape(rows, columns, meta_rows, meta_cols);
            }
            break;
        }
        case 32: {
            if (columnwise) {
                BlockwiseQuantizer<T, 32, qbits, true>::quantizeMetaShape(rows, columns, meta_rows, meta_cols);
            } else {
                BlockwiseQuantizer<T, 32, qbits, false>::quantizeMetaShape(
                                    rows, columns, meta_rows, meta_cols);
            }
            break;
        }
        case 64: {
            if (columnwise) {
                BlockwiseQuantizer<T, 64, qbits, true>::quantizeMetaShape(rows, columns, meta_rows,
                                                                      meta_cols);
            } else {
                BlockwiseQuantizer<T, 64, qbits, false>::quantizeMetaShape(rows, columns, meta_rows,
                                                                       meta_cols);
            }
            break;
        }
        case 128: {
            if (columnwise) {
                BlockwiseQuantizer<T, 128, qbits, true>::quantizeMetaShape(rows, columns, meta_rows,
                                                                      meta_cols);
            } else {
                BlockwiseQuantizer<T, 128, qbits, false>::quantizeMetaShape(rows, columns, meta_rows,
                                                                       meta_cols);
            }
            break;
        }
        case 256: {
            if (columnwise) {
                BlockwiseQuantizer<T, 256, qbits, true>::quantizeMetaShape(rows, columns, meta_rows,
                                                                      meta_cols);
            } else {
                BlockwiseQuantizer<T, 256, qbits, false>::quantizeMetaShape(rows, columns, meta_rows,
                                                                       meta_cols);
            }
            break;
        }
        default:
            meta_rows = 0;
            meta_cols = 0;
            break;
    }
}



template <typename T, int qbits>
void
MlasBlockwiseQuantizedShape(
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    int& q_rows,
    int& q_cols
    )
{
    switch (block_size) {
        case 16: {
            if (columnwise) {
                BlockwiseQuantizer<T, 16, qbits, true>::quantizedShape(rows, columns, q_rows, q_cols);
            } else {
                BlockwiseQuantizer<T, 16, qbits, false>::quantizedShape(rows, columns, q_rows, q_cols);
            }
            break;
        }
        case 32: {
            if (columnwise) {
                BlockwiseQuantizer<T, 32, qbits, true>::quantizedShape(rows, columns, q_rows, q_cols);
            } else {
                BlockwiseQuantizer<T, 32, qbits, false>::quantizedShape(
                                    rows, columns, q_rows, q_cols);
            }
            break;
        }
        case 64: {
            if (columnwise) {
                BlockwiseQuantizer<T, 64, qbits, true>::quantizedShape(rows, columns, q_rows, q_cols);
            } else {
                BlockwiseQuantizer<T, 64, qbits, false>::quantizedShape(rows, columns, q_rows, q_cols);
            }
            break;
        }
        case 128: {
            if (columnwise) {
                BlockwiseQuantizer<T, 128, qbits, true>::quantizedShape(rows, columns, q_rows, q_cols);
            } else {
                BlockwiseQuantizer<T, 128, qbits, false>::quantizedShape(rows, columns, q_rows, q_cols);
            }
            break;
        }
        case 256: {
            if (columnwise) {
                BlockwiseQuantizer<T, 256, qbits, true>::quantizedShape(rows, columns, q_rows, q_cols);
            } else {
                BlockwiseQuantizer<T, 256, qbits, false>::quantizedShape(rows, columns, q_rows, q_cols);
            }
            break;
        }
        default:
            q_rows = 0;
            q_cols = 0;
            break;
    }
}

template
void
MlasBlockwiseQuantMetaShape<float, 2>(
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    int& meta_rows,
    int& meta_cols
    );

template
void
MlasBlockwiseQuantMetaShape<MLAS_FP16, 2>(
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    int& meta_rows,
    int& meta_cols
    );

template void
MlasBlockwiseQuantMetaShape<float, 4>(
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    int& meta_rows,
    int& meta_cols
);

template
void
MlasBlockwiseQuantMetaShape<MLAS_FP16, 4>(
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    int& meta_rows,
    int& meta_cols
    );

    template
void
MlasBlockwiseQuantMetaShape<float, 8>(
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    int& meta_rows,
    int& meta_cols
    );

template
void
MlasBlockwiseQuantMetaShape<MLAS_FP16, 8>(
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    int& meta_rows,
    int& meta_cols
    );

template
void
MlasBlockwiseQuantizedShape<float, 2>(
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    int& q_rows,
    int& q_cols
    );

template
void
MlasBlockwiseQuantizedShape<MLAS_FP16, 2>(
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    int& q_rows,
    int& q_cols
    );

template
void
MlasBlockwiseQuantizedShape<float, 4>(
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    int& q_rows,
    int& q_cols
    );

template void
MlasBlockwiseQuantizedShape<MLAS_FP16, 4>(
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    int& q_rows,
    int& q_cols
    );

template
void
MlasBlockwiseQuantizedShape<float, 8>(
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    int& q_rows,
    int& q_cols
    );

template
void
MlasBlockwiseQuantizedShape<MLAS_FP16, 8>(
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    int& q_rows,
    int& q_cols
    );

template <int qbits>
void MLASCALL
MlasBlockwiseQuantizedBufferSizes(
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    size_t& q_data_size_in_bytes,
    size_t& q_scale_num_elements,
    size_t* q_zero_point_size_in_bytes
)
{
    q_data_size_in_bytes = q_scale_num_elements = 0;
    if (q_zero_point_size_in_bytes) {
        *q_zero_point_size_in_bytes = 0;
    }

    switch (block_size) {
        case 16:
            if (columnwise) {
                BlockwiseQuantizer<float, 16, qbits, true>::quantizedBufferSizes(
                    rows, columns, q_data_size_in_bytes, q_scale_num_elements, q_zero_point_size_in_bytes
                );
            } else {
                BlockwiseQuantizer<float, 16, qbits, false>::quantizedBufferSizes(
                    rows, columns, q_data_size_in_bytes, q_scale_num_elements, q_zero_point_size_in_bytes
                );
            }
            break;

        case 32:
            if (columnwise) {
                BlockwiseQuantizer<float, 32, qbits, true>::quantizedBufferSizes(
                    rows, columns, q_data_size_in_bytes, q_scale_num_elements, q_zero_point_size_in_bytes
                );
            } else {
                BlockwiseQuantizer<float, 32, qbits, false>::quantizedBufferSizes(
                    rows, columns, q_data_size_in_bytes, q_scale_num_elements, q_zero_point_size_in_bytes
                );
            }
            break;

        case 64:
            if (columnwise) {
                BlockwiseQuantizer<float, 64, qbits, true>::quantizedBufferSizes(
                    rows, columns, q_data_size_in_bytes, q_scale_num_elements, q_zero_point_size_in_bytes
                );
            } else {
                BlockwiseQuantizer<float, 64, qbits, false>::quantizedBufferSizes(
                    rows, columns, q_data_size_in_bytes, q_scale_num_elements, q_zero_point_size_in_bytes
                );
            }
            break;

        case 128:
            if (columnwise) {
                BlockwiseQuantizer<float, 128, qbits, true>::quantizedBufferSizes(
                    rows, columns, q_data_size_in_bytes, q_scale_num_elements, q_zero_point_size_in_bytes
                );
            } else {
                BlockwiseQuantizer<float, 128, qbits, false>::quantizedBufferSizes(
                    rows, columns, q_data_size_in_bytes, q_scale_num_elements, q_zero_point_size_in_bytes
                );
            }
            break;

        case 256:
            if (columnwise) {
                BlockwiseQuantizer<float, 256, qbits, true>::quantizedBufferSizes(
                    rows, columns, q_data_size_in_bytes, q_scale_num_elements, q_zero_point_size_in_bytes
                );
            } else {
                BlockwiseQuantizer<float, 256, qbits, false>::quantizedBufferSizes(
                    rows, columns, q_data_size_in_bytes, q_scale_num_elements, q_zero_point_size_in_bytes
                );
            }
            break;

        default:
            // Only block size 16, 32, 64, 128, 256 are supported.
            break;
    }
}

template
void MLASCALL
MlasBlockwiseQuantizedBufferSizes<2>(
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    size_t& q_data_size_in_bytes,
    size_t& q_scale_num_elements,
    size_t* q_zero_point_size_in_bytes
);

template
void MLASCALL
MlasBlockwiseQuantizedBufferSizes<4>(
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    size_t& q_data_size_in_bytes,
    size_t& q_scale_num_elements,
    size_t* q_zero_point_size_in_bytes
);

template
void MLASCALL
MlasBlockwiseQuantizedBufferSizes<8>(
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    size_t& q_data_size_in_bytes,
    size_t& q_scale_num_elements,
    size_t* q_zero_point_size_in_bytes
);

template <typename T, int qbits>
void
MlasQuantizeBlockwise(
    uint8_t* dst,
    T* scales,
    uint8_t* zero_points,
    const T* src,
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    int leading_dimension,
    MLAS_THREADPOOL* thread_pool
    )
{
    switch (block_size) {
        case 16:
            if (columnwise) {
                BlockwiseQuantizer<T, 16, qbits, true>::quantizeAndTranspose(
                    dst, scales, zero_points, src, rows, columns, leading_dimension, thread_pool);
            } else {
                BlockwiseQuantizer<T, 16, qbits, false>::quantizeAndTranspose(
                    dst, scales, zero_points, src, rows, columns, leading_dimension, thread_pool);
            }
            break;

        case 32:
            if (columnwise) {
                BlockwiseQuantizer<T, 32, qbits, true>::quantizeAndTranspose(
                    dst, scales, zero_points, src, rows, columns, leading_dimension, thread_pool);
            } else {
                BlockwiseQuantizer<T, 32, qbits, false>::quantizeAndTranspose(
                    dst, scales, zero_points, src, rows, columns, leading_dimension, thread_pool);
            }
            break;

        case 64:
            if (columnwise) {
                BlockwiseQuantizer<T, 64, qbits, true>::quantizeAndTranspose(
                    dst, scales, zero_points, src, rows, columns, leading_dimension, thread_pool);
            } else {
                BlockwiseQuantizer<T, 64, qbits, false>::quantizeAndTranspose(
                    dst, scales, zero_points, src, rows, columns, leading_dimension, thread_pool);
            }
            break;

        case 128:
            if (columnwise) {
                BlockwiseQuantizer<T, 128, qbits, true>::quantizeAndTranspose(
                    dst, scales, zero_points, src, rows, columns, leading_dimension, thread_pool);
            } else {
                BlockwiseQuantizer<T, 128, qbits, false>::quantizeAndTranspose(
                    dst, scales, zero_points, src, rows, columns, leading_dimension, thread_pool);
            }
            break;

        case 256:
            if (columnwise) {
                BlockwiseQuantizer<T, 256, qbits, true>::quantizeAndTranspose(
                    dst, scales, zero_points, src, rows, columns, leading_dimension, thread_pool);
            } else {
                BlockwiseQuantizer<T, 256, qbits, false>::quantizeAndTranspose(
                    dst, scales, zero_points, src, rows, columns, leading_dimension, thread_pool);
            }
            break;

        default:
            // Only block size 16, 32, 64, 128, 256 are supported.
            break;
    }
}

template
void
MlasQuantizeBlockwise<float, 2>(
    uint8_t* dst,
    float* scales,
    uint8_t* zero_points,
    const float* src,
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    int leading_dimension,
    MLAS_THREADPOOL* thread_pool
    );

template
void
MlasQuantizeBlockwise<MLAS_FP16, 2>(
    uint8_t* dst,
    MLAS_FP16* scales,
    uint8_t* zero_points,
    const MLAS_FP16* src,
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    int leading_dimension,
    MLAS_THREADPOOL* thread_pool
    );

template
void
MlasQuantizeBlockwise<float, 4>(
    uint8_t* dst,
    float* scales,
    uint8_t* zero_points,
    const float* src,
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    int leading_dimension,
    MLAS_THREADPOOL* thread_pool
    );

template
void
MlasQuantizeBlockwise<MLAS_FP16, 4>(
    uint8_t* dst,
    MLAS_FP16* scales,
    uint8_t* zero_points,
    const MLAS_FP16* src,
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    int leading_dimension,
    MLAS_THREADPOOL* thread_pool
    );

    template
    void
    MlasQuantizeBlockwise<float, 8>(
        uint8_t* dst,
        float* scales,
        uint8_t* zero_points,
        const float* src,
        int block_size,
        bool columnwise,
        int rows,
        int columns,
        int leading_dimension,
        MLAS_THREADPOOL* thread_pool
        );

    template
    void
    MlasQuantizeBlockwise<MLAS_FP16, 8>(
        uint8_t* dst,
        MLAS_FP16* scales,
        uint8_t* zero_points,
        const MLAS_FP16* src,
        int block_size,
        bool columnwise,
        int rows,
        int columns,
        int leading_dimension,
        MLAS_THREADPOOL* thread_pool
        );

template <typename T, int qbits>
void
MlasDequantizeBlockwise(
    T* dst,
    const uint8_t* src,
    const T* scales,
    const uint8_t* zero_points,
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    MLAS_THREADPOOL* thread_pool
    )
{
    switch (block_size) {
        case 16:
            if (columnwise) {
                BlockwiseQuantizer<T, 16, qbits, true>::dequantize(dst, src, scales, zero_points, rows,
                                                               columns, thread_pool);
            } else {
                BlockwiseQuantizer<T, 16, qbits, false>::dequantize(dst, src, scales, zero_points, rows,
                                                                columns, thread_pool);
            }
            break;
        case 32:
            if (columnwise) {
                BlockwiseQuantizer<T, 32, qbits, true>::dequantize(dst, src, scales, zero_points, rows,
                                                               columns, thread_pool);
            } else {
                BlockwiseQuantizer<T, 32, qbits, false>::dequantize(dst, src, scales, zero_points, rows,
                                                                columns, thread_pool);
            }
            break;
        case 64:
            if (columnwise) {
                BlockwiseQuantizer<T, 64, qbits, true>::dequantize(dst, src, scales, zero_points, rows,
                                                               columns, thread_pool);
            } else {
                BlockwiseQuantizer<T, 64, qbits, false>::dequantize(dst, src, scales, zero_points, rows,
                                                                columns, thread_pool);
            }
            break;
        case 128:
            if (columnwise) {
                BlockwiseQuantizer<T, 128, qbits, true>::dequantize(dst, src, scales, zero_points, rows,
                                                                columns, thread_pool);
            } else {
                BlockwiseQuantizer<T, 128, qbits, false>::dequantize(dst, src, scales, zero_points,
                                                                 rows, columns, thread_pool);
            }
            break;
        case 256:
            if (columnwise) {
                BlockwiseQuantizer<T, 256, qbits, true>::dequantize(dst, src, scales, zero_points, rows,
                                                                columns, thread_pool);
            } else {
                BlockwiseQuantizer<T, 256, qbits, false>::dequantize(dst, src, scales, zero_points,
                                                                 rows, columns, thread_pool);
            }
            break;
        default:
            // Only block size 16, 32, 64, 128, 256 are supported.
            break;
    }
}

template void
MlasDequantizeBlockwise<float, 2>(
    float* dst,
    const uint8_t* src,
    const float* scales,
    const uint8_t* zero_points,
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    MLAS_THREADPOOL* thread_pool
);

template void
MlasDequantizeBlockwise<MLAS_FP16, 2>(
    MLAS_FP16* dst,
    const uint8_t* src,
    const MLAS_FP16* scales,
    const uint8_t* zero_points,
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    MLAS_THREADPOOL* thread_pool
);

template void
MlasDequantizeBlockwise<float, 4>(
    float* dst,
    const uint8_t* src,
    const float* scales,
    const uint8_t* zero_points,
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    MLAS_THREADPOOL* thread_pool
);

template void
MlasDequantizeBlockwise<MLAS_FP16, 4>(
    MLAS_FP16* dst,
    const uint8_t* src,
    const MLAS_FP16* scales,
    const uint8_t* zero_points,
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    MLAS_THREADPOOL* thread_pool
);

template void
MlasDequantizeBlockwise<float, 8>(
    float* dst,
    const uint8_t* src,
    const float* scales,
    const uint8_t* zero_points,
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    MLAS_THREADPOOL* thread_pool
);

template void
MlasDequantizeBlockwise<MLAS_FP16, 8>(
    MLAS_FP16* dst,
    const uint8_t* src,
    const MLAS_FP16* scales,
    const uint8_t* zero_points,
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    MLAS_THREADPOOL* thread_pool
);

template <typename Tin, int qbits>
bool
MlasQDQQuantizeBlockwise(
    const Tin* src,
    Tin* scales,
    uint8_t* zero_points,
    uint8_t* dst,
    bool columnwise,
    int rows,
    int columns,
    int quant_block_size,
    MLAS_THREADPOOL* thread_pool
)
{
    if (columnwise) {
        if (zero_points) {
            BlockwiseQDQQuantizer<Tin, qbits, false>::QuantizeColumnWise(
                src, scales, zero_points, dst, rows, columns, quant_block_size, thread_pool
            );
            return false;
        } else {
            BlockwiseQDQQuantizer<Tin, qbits, true>::QuantizeColumnWise(
                src, scales, zero_points, dst, rows, columns, quant_block_size, thread_pool
            );
            return true;
        }
    } else {
        ORT_THROW("Row-wise MlasQDQQuantizeBlockwise is not implemented");
    }
}

template bool
MlasQDQQuantizeBlockwise<float, 4>(
    const float* src,
    float* scales,
    uint8_t* zero_points,
    uint8_t* dst,
    bool columnwise,
    int rows,
    int columns,
    int quant_block_size,
    MLAS_THREADPOOL* thread_pool
);

template bool
MlasQDQQuantizeBlockwise<float, 2>(
    const float* src,
    float* scales,
    uint8_t* zero_points,
    uint8_t* dst,
    bool columnwise,
    int rows,
    int columns,
    int quant_block_size,
    MLAS_THREADPOOL* thread_pool
);

template bool
MlasQDQQuantizeBlockwise<MLAS_FP16, 4>(
    const MLAS_FP16* src,
    MLAS_FP16* scales,
    uint8_t* zero_points,
    uint8_t* dst,
    bool columnwise,
    int rows,
    int columns,
    int quant_block_size,
    MLAS_THREADPOOL* thread_pool
);

template <typename Tin, int qbits, bool signed_quant>
void
MlasQDQTransposeBlockwiseQuantized(
    const uint8_t* src_weights,
    const Tin* src_scales,
    const uint8_t* src_zero_points,
    uint8_t* dst_weights,
    Tin* dst_scales,
    uint8_t* dst_zero_points,
    bool columnwise,
    int rows,
    int columns,
    int quant_block_size,
    MLAS_THREADPOOL* thread_pool
)
{
    if (columnwise) {
        BlockwiseQDQQuantizer<Tin, qbits, signed_quant>::TransposeColumnWiseQuantized(
            src_weights, src_scales, src_zero_points, dst_weights, dst_scales, dst_zero_points,
            rows, columns, quant_block_size, thread_pool
        );
    } else {
        ORT_THROW("Row-wise MlasQDQTransposeBlockwiseQuantized is not implemented");
    }
}

template void
MlasQDQTransposeBlockwiseQuantized<float, 2, true>(
    const uint8_t* src_weights,
    const float* src_scales,
    const uint8_t* src_zero_points,
    uint8_t* dst_weights,
    float* dst_scales,
    uint8_t* dst_zero_points,
    bool columnwise,
    int rows,
    int columns,
    int quant_block_size,
    MLAS_THREADPOOL* thread_pool
);

template void
MlasQDQTransposeBlockwiseQuantized<float, 2, false>(
    const uint8_t* src_weights,
    const float* src_scales,
    const uint8_t* src_zero_points,
    uint8_t* dst_weights,
    float* dst_scales,
    uint8_t* dst_zero_points,
    bool columnwise,
    int rows,
    int columns,
    int quant_block_size,
    MLAS_THREADPOOL* thread_pool
);

template void
MlasQDQTransposeBlockwiseQuantized<float, 4, true>(
    const uint8_t* src_weights,
    const float* src_scales,
    const uint8_t* src_zero_points,
    uint8_t* dst_weights,
    float* dst_scales,
    uint8_t* dst_zero_points,
    bool columnwise,
    int rows,
    int columns,
    int quant_block_size,
    MLAS_THREADPOOL* thread_pool
);

template void
MlasQDQTransposeBlockwiseQuantized<float, 4, false>(
    const uint8_t* src_weights,
    const float* src_scales,
    const uint8_t* src_zero_points,
    uint8_t* dst_weights,
    float* dst_scales,
    uint8_t* dst_zero_points,
    bool columnwise,
    int rows,
    int columns,
    int quant_block_size,
    MLAS_THREADPOOL* thread_pool
);

template void
MlasQDQTransposeBlockwiseQuantized<MLAS_FP16, 4, true>(
    const uint8_t* src_weights,
    const MLAS_FP16* src_scales,
    const uint8_t* src_zero_points,
    uint8_t* dst_weights,
    MLAS_FP16* dst_scales,
    uint8_t* dst_zero_points,
    bool columnwise,
    int rows,
    int columns,
    int quant_block_size,
    MLAS_THREADPOOL* thread_pool
);

template void
MlasQDQTransposeBlockwiseQuantized<MLAS_FP16, 4, false>(
    const uint8_t* src_weights,
    const MLAS_FP16* src_scales,
    const uint8_t* src_zero_points,
    uint8_t* dst_weights,
    MLAS_FP16* dst_scales,
    uint8_t* dst_zero_points,
    bool columnwise,
    int rows,
    int columns,
    int quant_block_size,
    MLAS_THREADPOOL* thread_pool
);
