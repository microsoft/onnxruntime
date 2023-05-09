/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    mlas_q4.h

Abstract:

    This module contains the data structures and implementations
    for blocked int4 quantization and dequantization.

    Int4 block quantization is used to compress weight tensors of large
    language models.

--*/

#include "mlas_q4.h"

#include <cstring>
#include <math.h>
#include <algorithm>

//
// Functions for locating data from a quantized blob
//
template<typename T> 
float&
MlasQ4BlkScale(uint8_t* BlkPtr);

template <typename T>
float
MlasQ4BlkScale(const uint8_t* BlkPtr);

template <typename T>
uint8_t&
MlasQ4BlkZeroPoint(uint8_t* BlkPtr);

template <typename T>
uint8_t
MlasQ4BlkZeroPoint(const uint8_t* BlkPtr);

template <typename T>
uint8_t*
MlasQ4BlkData(uint8_t* BlkPtr);

template <typename T>
const uint8_t*
MlasQ4BlkData(const uint8_t* BlkPtr);


/**
 * @brief Representing int4 quantize type, block quant type 0:
 *
 * Block size 32, use 32 fp32 numbers to find quantization parameter:
 * scale (fp 32) and no zero point, then quantize the numbers
 * into int4. The resulting blob takes 16 + 4 = 20 bytes.
 */
struct MLAS_Q4TYPE_BLK0 {
    static constexpr size_t BlkLen = 32;
    static constexpr size_t BlobSize = BlkLen / 2 + sizeof(float);
};

template <>
inline float&
MlasQ4BlkScale<MLAS_Q4TYPE_BLK0>(uint8_t* BlkPtr)
{
    return *reinterpret_cast<float*>(BlkPtr);
}

template <>
inline float
MlasQ4BlkScale<MLAS_Q4TYPE_BLK0>(const uint8_t* BlkPtr)
{
    return *reinterpret_cast<const float*>(BlkPtr);
}

template <>
inline uint8_t*
MlasQ4BlkData<MLAS_Q4TYPE_BLK0>(uint8_t* BlkPtr)
{
    return BlkPtr + sizeof(float);
}

template <>
inline const uint8_t*
MlasQ4BlkData<MLAS_Q4TYPE_BLK0>(const uint8_t* BlkPtr)
{
    return BlkPtr + sizeof(float);
}


/**
 * @brief Representing int4 quantize type, block quant type 1:
 *
 * Block size 32, use 32 fp32 numbers to find quantization parameter:
 * scale (fp 32) and zero point (int8), and then quantize the numbers
 * into int4. The resulting blob takes 16 + 5 = 21 bytes.
 */
struct MLAS_Q4TYPE_BLK1 {
    static constexpr size_t BlkLen = 32;
    static constexpr size_t BlobSize = BlkLen / 2 + sizeof(float) + sizeof(uint8_t);
};

template<>
inline float&
MlasQ4BlkScale<MLAS_Q4TYPE_BLK1>(uint8_t* BlkPtr)
{
    return *reinterpret_cast<float*>(BlkPtr);
}

template<>
inline float
MlasQ4BlkScale<MLAS_Q4TYPE_BLK1>(const uint8_t* BlkPtr)
{
    return *reinterpret_cast<const float*>(BlkPtr);
}

template<>
inline uint8_t&
MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(uint8_t* BlkPtr)
{
    return *(BlkPtr + sizeof(float));
}

template<>
inline uint8_t
MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(const uint8_t* BlkPtr)
{
    return *(BlkPtr + sizeof(float));
}

template<>
inline uint8_t*
MlasQ4BlkData<MLAS_Q4TYPE_BLK1>(uint8_t* BlkPtr)
{
    return BlkPtr + sizeof(float) + sizeof(uint8_t);
}

template<>
inline const uint8_t*
MlasQ4BlkData<MLAS_Q4TYPE_BLK1>(const uint8_t* BlkPtr)
{
    return BlkPtr + sizeof(float) + sizeof(uint8_t);
}


//
// Quantization and Packing
//
// Since block quantization is used for compress large language model weights,
// it is usually used as the right hand side in matrix multiplications. So
// we can just perform quantize and packing together to help accelerate
// matrix multiplication.
//
// We take a tiles of 32 row and 4 column, transpose it, and quantize it
// into 4 blocks. So numbers in quantized block are from the same column.
// This is different from other int4 block quantization, where the numbers
// in a block are from the same row.
//

constexpr size_t MLAS_Q4_N_STRIDE = 4;

template<typename T>
inline size_t
MlasQ4SizeImpl(size_t N, size_t K)
{
    const size_t AlignedN = (N + MLAS_Q4_N_STRIDE - 1) & ~(MLAS_Q4_N_STRIDE - 1);
    const size_t KBlocks = (K + T::BlkLen - 1) / T::BlkLen;

    const size_t NumBlocks = AlignedN * KBlocks;

    return NumBlocks * T::BlobSize;
}


size_t
MLASCALL
MlasQ4GemmPackBSize(MLAS_BLK_QUANT_TYPE QType, size_t N, size_t K)
{
    if (QType == BlkQ4Sym) {
        return MlasQ4SizeImpl<MLAS_Q4TYPE_BLK0>(N, K);
    }
    return MlasQ4SizeImpl<MLAS_Q4TYPE_BLK1>(N, K);
}


template<typename T>
void
MlasQ4GemmPackBImpl(void* PackedBuf, const float* FpData, size_t N, size_t K, size_t ldb);

template <>
inline void
MlasQ4GemmPackBImpl<MLAS_Q4TYPE_BLK0>(
    void* PackedBuf, const float* FpData, size_t N, size_t K, size_t ldb)
{
    auto* dst_ptr = reinterpret_cast<uint8_t*>(PackedBuf);

    for (size_t n = 0; n < N; n += MLAS_Q4_N_STRIDE) {
        size_t nlen = std::min(MLAS_Q4_N_STRIDE, N - n);

        for (size_t k = 0; k < K; k += MLAS_Q4TYPE_BLK0::BlkLen) {
            size_t klen = std::min(MLAS_Q4TYPE_BLK0::BlkLen, K - k);

            const float* src = FpData + ldb * k + n;

            for (size_t nn = 0; nn < nlen; nn++) {
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
                MlasQ4BlkScale<MLAS_Q4TYPE_BLK0>(dst_ptr) = scale;
                uint8_t* data = MlasQ4BlkData<MLAS_Q4TYPE_BLK0>(dst_ptr);

                for (size_t l = 0; l < MLAS_Q4TYPE_BLK0::BlkLen; l += 2) {
                    const float v0 = l < klen ? src[ldb * l] * reciprocal_scale : 0;

                    const uint8_t vi0 = (uint8_t)std::min(15.0f, std::max(0.0f, v0 + 8.5f));
                    const float v1 = (l + 1 < klen) ? src[ldb * (l + 1)] * reciprocal_scale : 0;
                    const uint8_t vi1 = (uint8_t)std::min(15.0f, std::max(0.0f, v1 + 8.5f));

                    data[l / 2] = vi0 | (vi1 << 4);
                }
                dst_ptr += MLAS_Q4TYPE_BLK0::BlobSize;
                src++;  // mov to next column
            }
            if (nlen < MLAS_Q4_N_STRIDE) {
                memset(dst_ptr, 0, MLAS_Q4TYPE_BLK0::BlobSize * (MLAS_Q4_N_STRIDE - nlen));
                dst_ptr += MLAS_Q4TYPE_BLK0::BlobSize * (MLAS_Q4_N_STRIDE - nlen);
            }

        }  // advance to next block or rows
    }      // advance next block of columns
}

template<>
inline void
MlasQ4GemmPackBImpl<MLAS_Q4TYPE_BLK1>(
    void* PackedBuf, const float* FpData, size_t N, size_t K, size_t ldb)
{
    auto* dst_ptr = reinterpret_cast<uint8_t*>(PackedBuf);

    for (size_t n = 0; n < N; n += MLAS_Q4_N_STRIDE) {
        size_t nlen = std::min(MLAS_Q4_N_STRIDE, N - n);

        for (size_t k = 0; k < K; k += MLAS_Q4TYPE_BLK1::BlkLen) {
            size_t klen = std::min(MLAS_Q4TYPE_BLK1::BlkLen, K - k);

            const float* src = FpData + ldb * k + n;

            for (size_t nn = 0; nn < nlen; nn++) {
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

                for (size_t l = 0; l < MLAS_Q4TYPE_BLK1::BlkLen; l += 2) {
                    const float v0 = l < klen ? src[ldb * l] : 0;
                    const uint8_t vi0 = (uint8_t)std::min(
                        15.0f, std::max(0.0f, roundf(v0 * reciprocal_scale + zp)));
                    const float v1 = (l + 1 < klen) ? src[ldb * (l + 1)] : 0;
                    const uint8_t vi1 = (uint8_t)std::min(
                        15.0f, std::max(0.0f, roundf(v1 * reciprocal_scale + zp)));

                    data[l / 2] = vi0 | (vi1 << 4);
                }
                dst_ptr += MLAS_Q4TYPE_BLK1::BlobSize;
                src++;  // mov to next column
            }
            if (nlen < MLAS_Q4_N_STRIDE) {
                memset(dst_ptr, 0, MLAS_Q4TYPE_BLK1::BlobSize * (MLAS_Q4_N_STRIDE - nlen));
                dst_ptr += MLAS_Q4TYPE_BLK1::BlobSize * (MLAS_Q4_N_STRIDE - nlen);
            }

        }  // advance to next block or rows
    }      // advance next block of columns
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
    if (QType == BlkQ4Sym) {
        MlasQ4GemmPackBImpl<MLAS_Q4TYPE_BLK0>(PackedBuf, FpData, N, K, ldb);
    } else {
        MlasQ4GemmPackBImpl<MLAS_Q4TYPE_BLK1>(PackedBuf, FpData, N, K, ldb);
    }
}

template<typename T>
void
MlasQ4GemmUnPackBImpl(float* FpData, const void* PackedBuf, size_t N, size_t K, size_t ldb);

template <>
inline void
MlasQ4GemmUnPackBImpl<MLAS_Q4TYPE_BLK0>(
    float* FpData, const void* PackedBuf, size_t N, size_t K, size_t ldb)
{
    const auto* src = reinterpret_cast<const uint8_t*>(PackedBuf);
    for (size_t n = 0; n < N; n += MLAS_Q4_N_STRIDE) {
        size_t CountN = std::min(N - n, MLAS_Q4_N_STRIDE);

        for (size_t k = 0; k < K; k += MLAS_Q4TYPE_BLK0::BlkLen) {
            size_t CountK = std::min(K - k, MLAS_Q4TYPE_BLK0::BlkLen);

            float* dest = FpData + ldb * k + n;
            for (size_t nn = 0; nn < CountN; nn++) {
                const float s = MlasQ4BlkScale<MLAS_Q4TYPE_BLK0>(src);
                const uint8_t* pp = MlasQ4BlkData<MLAS_Q4TYPE_BLK0>(src);

                for (size_t l = 0; l < CountK; l += 2) {
                    const uint8_t vi = pp[l / 2];

                    const int vi0 = (vi & 0x0F) - 8;
                    const int vi1 = (vi >> 4) - 8;

                    const float v0 = vi0 * s;
                    const float v1 = vi1 * s;

                    dest[ldb * l] = v0;
                    if (l + 1 < CountK) {
                        dest[ldb * (l + 1)] = v1;
                    }
                }
                src += MLAS_Q4TYPE_BLK0::BlobSize;
                dest++;  // next column
            }
            src += (MLAS_Q4_N_STRIDE - CountN) * MLAS_Q4TYPE_BLK0::BlobSize;
        }
    }
}

template<>
inline void
MlasQ4GemmUnPackBImpl<MLAS_Q4TYPE_BLK1>(
    float* FpData, const void* PackedBuf, size_t N, size_t K, size_t ldb)
{
    const auto* src = reinterpret_cast<const uint8_t*>(PackedBuf);
    for (size_t n = 0; n < N; n += MLAS_Q4_N_STRIDE) {
        size_t CountN = std::min(N - n, MLAS_Q4_N_STRIDE);

        for (size_t k = 0; k < K; k += MLAS_Q4TYPE_BLK1::BlkLen) {
            size_t CountK = std::min(K - k, MLAS_Q4TYPE_BLK1::BlkLen);

            float* dest = FpData + ldb * k + n;
            for (size_t nn = 0; nn < CountN; nn++) {
                const float s = MlasQ4BlkScale<MLAS_Q4TYPE_BLK1>(src);
                const uint8_t z = MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(src);
                const uint8_t* pp = MlasQ4BlkData<MLAS_Q4TYPE_BLK1>(src);

                for (size_t l = 0; l < CountK; l += 2) {
                    const uint8_t vi = pp[l / 2];

                    const int8_t vi0 = vi & 0x0F;
                    const int8_t vi1 = vi >> 4;

                    const float v0 = (vi0 - z) * s;
                    const float v1 = (vi1 - z) * s;

                    dest[ldb * l] = v0;
                    if (l + 1 < CountK) {
                        dest[ldb * (l + 1)] = v1;
                    }
                }
                src += MLAS_Q4TYPE_BLK1::BlobSize;
                dest++;  // next column
            }
            src += (MLAS_Q4_N_STRIDE - CountN) * MLAS_Q4TYPE_BLK1::BlobSize;
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
    if (QType == BlkQ4Sym) {
        MlasQ4GemmUnPackBImpl<MLAS_Q4TYPE_BLK0>(FpData, PackedBuf, N, K, ldb);
    } else {
        MlasQ4GemmUnPackBImpl<MLAS_Q4TYPE_BLK1>(FpData, PackedBuf, N, K, ldb);
    }
}
