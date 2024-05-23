/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_q8_block.h

Abstract:

    This module includes helper functions for manipulating blocks of quantized
    int8 (Q8) values.

--*/

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "mlasi.h"

// structure of a Q8 block with BlkLen elements:
// - 1 x float scale
// - 1 x float sum of int8 block values
// - BlkLen x int8 block values

MLAS_FORCEINLINE
const float&
Q8BlkScale(const std::byte* BlkPtr)
{
    return *reinterpret_cast<const float*>(BlkPtr);
}

MLAS_FORCEINLINE
float&
Q8BlkScale(std::byte* BlkPtr)
{
    return *reinterpret_cast<float*>(BlkPtr);
}

MLAS_FORCEINLINE
const float&
Q8BlkSum(const std::byte* BlkPtr) {
    return *reinterpret_cast<const float*>(BlkPtr + sizeof(float));
}

MLAS_FORCEINLINE
float&
Q8BlkSum(std::byte* BlkPtr) {
    return *reinterpret_cast<float*>(BlkPtr + sizeof(float));
}

MLAS_FORCEINLINE
const int8_t*
Q8BlkData(const std::byte* BlkPtr)
{
    return reinterpret_cast<const int8_t*>(BlkPtr + sizeof(float) + sizeof(float));
}

MLAS_FORCEINLINE
int8_t*
Q8BlkData(std::byte* BlkPtr)
{
    return reinterpret_cast<int8_t*>(BlkPtr + sizeof(float) + sizeof(float));
}

MLAS_FORCEINLINE
constexpr size_t
Q8BlkAlignment()
{
    return alignof(float);
}

MLAS_FORCEINLINE
constexpr size_t
Q8BlkSize(size_t BlkLen)
{
    const size_t BlkSize = sizeof(float) + sizeof(float) + BlkLen * sizeof(int8_t);
    // Ensure contiguous blocks are suitably aligned.
    assert(BlkSize % Q8BlkAlignment() == 0);
    return BlkSize;
}
