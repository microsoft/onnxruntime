/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    q4common.h

Abstract:

    Contains the common structures and code for blocked int4 quantization
    and dequantization.

    Int4 block quantization is used to compress weight tensors of large
    language models.

--*/

#include "mlas_q4.h"
#include "mlasi.h"

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
 * @brief 32 numbers per quantization block
 */
constexpr size_t MLAS_QUANT4_BLK_LEN = 32;

/**
 * @brief Representing int4 quantize type, block quant type 0:
 *
 * Block size 32, use 32 fp32 numbers to find quantization parameter:
 * scale (fp 32) and no zero point, then quantize the numbers
 * into int4. The resulting blob takes 16 + 4 = 20 bytes.
 */
struct MLAS_Q4TYPE_BLK0 {
    static constexpr size_t BlkLen = MLAS_QUANT4_BLK_LEN;
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
    static constexpr size_t BlkLen = MLAS_QUANT4_BLK_LEN;
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


/**
 * @brief Representing int4 quantize type, block quant type 2:
 *
 * Block size 64, use 64 fp32 numbers to find quantization parameter:
 * scale (fp 32) and no zero point, then quantize the numbers
 * into int4. The resulting blob takes 32 + 4 = 36 bytes.
 */
struct MLAS_Q4TYPE_BLK2 {
    static constexpr size_t BlkLen = 64;
    static constexpr size_t BlobSize = BlkLen / 2 + sizeof(float);
};

template <>
inline float&
MlasQ4BlkScale<MLAS_Q4TYPE_BLK2>(uint8_t* BlkPtr)
{
    return *reinterpret_cast<float*>(BlkPtr);
}

template <>
inline float
MlasQ4BlkScale<MLAS_Q4TYPE_BLK2>(const uint8_t* BlkPtr)
{
    return *reinterpret_cast<const float*>(BlkPtr);
}

template <>
inline uint8_t*
MlasQ4BlkData<MLAS_Q4TYPE_BLK2>(uint8_t* BlkPtr)
{
    return BlkPtr + sizeof(float);
}

template <>
inline const uint8_t*
MlasQ4BlkData<MLAS_Q4TYPE_BLK2>(const uint8_t* BlkPtr)
{
    return BlkPtr + sizeof(float);
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
