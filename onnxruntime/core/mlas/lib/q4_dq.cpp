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
