/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    q4common.h

Abstract:

    Define int4 block quantization types.

    Int4 block quantization is used to compress weight tensors of large
    language models. It takes a number (must be multiple of 32) of floating
    point values, calculates their quantization parameters, and saves
    the parameters and the quantized data in a blob.
--*/

#include "mlas_q4.h"
#include "mlasi.h"

#include <math.h>
#include <algorithm>

//
// Functions for locating data from a quantized blob
//
template<typename T> 
MLAS_FORCEINLINE
float&
MlasQ4BlkScale(uint8_t* BlkPtr)
{
    return *reinterpret_cast<float*>(BlkPtr);
}

template <typename T>
MLAS_FORCEINLINE
float
MlasQ4BlkScale(const uint8_t* BlkPtr)
{
    return *reinterpret_cast<const float*>(BlkPtr);
}

template <typename T>
uint8_t&
MlasQ4BlkZeroPoint(uint8_t* BlkPtr);

template <typename T>
uint8_t
MlasQ4BlkZeroPoint(const uint8_t* BlkPtr);

template <typename T>
MLAS_FORCEINLINE
uint8_t*
MlasQ4BlkData(uint8_t* BlkPtr)
{
    return BlkPtr + sizeof(float);
}

template <typename T>
MLAS_FORCEINLINE
const uint8_t*
MlasQ4BlkData(const uint8_t* BlkPtr)
{
    return BlkPtr + sizeof(float);
}

/**
 * @brief Every block quantization type, its block size (BlkLen)
 *        Must be multiple of 32!
 */
constexpr size_t MLAS_QUANT4_BLK_UNIT = 32;

/**
 * @brief Representing int4 quantize type, block quant type 0:
 *
 * Block size 32, use 32 fp32 numbers to find quantization parameter:
 * scale (fp 32) and no zero point, then quantize the numbers
 * into int4. The resulting blob takes 16 + 4 = 20 bytes.
 */
struct MLAS_Q4TYPE_BLK0 {
    static constexpr size_t BlkLen = MLAS_QUANT4_BLK_UNIT;
    static constexpr size_t BlobSize = BlkLen / 2 + sizeof(float);
};

/**
 * @brief Representing int4 quantize type, block quant type 1:
 *
 * Block size 32, use 32 fp32 numbers to find quantization parameter:
 * scale (fp 32) and zero point (int8), and then quantize the numbers
 * into int4. The resulting blob takes 16 + 5 = 21 bytes.
 * 
 * So far this is the only type that includes a zero-point value.
 * Maybe we should consider store the quantization parameters seperatedly.
 */
struct MLAS_Q4TYPE_BLK1 {
    static constexpr size_t BlkLen = MLAS_QUANT4_BLK_UNIT;
    static constexpr size_t BlobSize = BlkLen / 2 + sizeof(float) + sizeof(uint8_t);
};

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
    static constexpr size_t BlkLen = MLAS_QUANT4_BLK_UNIT * 2;
    static constexpr size_t BlobSize = BlkLen / 2 + sizeof(float);
};


/**
 * @brief Representing int4 quantize type, block quant type 4:
 *
 * Block size 128, use 128 fp32 numbers to find quantization parameter:
 * scale (fp 32) and no zero point, then quantize the numbers
 * into int4. The resulting blob takes 32 + 4 = 36 bytes.
 */
struct MLAS_Q4TYPE_BLK4 {
    static constexpr size_t BlkLen = MLAS_QUANT4_BLK_UNIT * 4;
    static constexpr size_t BlobSize = BlkLen / 2 + sizeof(float);
};
