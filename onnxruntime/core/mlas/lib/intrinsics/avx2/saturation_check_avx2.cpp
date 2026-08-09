/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    saturation_check_avx2.cpp

Abstract:

    This module implements logic to check saturation of the VPMADDUBSW
    instruction.

--*/

#include <immintrin.h>

#include <atomic>
#include <iostream>

namespace onnxruntime
{
extern std::atomic<int> saturation_count;
}

extern "C" void
CheckSaturationForVPMADDUBSW(const __m256i* unsigned_ptr, const __m256i* signed_ptr)
{
    // Load data from memory (unaligned load)
    __m256i unsigned_data = _mm256_loadu_si256(unsigned_ptr);
    __m256i signed_data = _mm256_loadu_si256(signed_ptr);

    alignas(32) uint8_t unsigned_bytes[32];  // Unsigned input values
    alignas(32) int8_t signed_bytes[32];     // Signed input values

    // Store the data into the byte arrays
    _mm256_store_si256(reinterpret_cast<__m256i*>(unsigned_bytes), unsigned_data);
    _mm256_store_si256(reinterpret_cast<__m256i*>(signed_bytes), signed_data);

    bool saturation_detected = false;

    // Iterate through the 16 pairs of 8-bit unsigned and signed values
    for (int i = 0; i < 16; ++i) {
        // Perform the VPMADDUBSW operation in higher precision (int32_t)
        int32_t computed_value =
            static_cast<int32_t>(signed_bytes[2 * i]) * static_cast<int32_t>(static_cast<uint32_t>(unsigned_bytes[2 * i])) +
            static_cast<int32_t>(signed_bytes[2 * i + 1]) * static_cast<int32_t>(static_cast<uint32_t>(unsigned_bytes[2 * i + 1]));

        // If the computed value exceeds the 16-bit signed integer range, saturation occurred
        if (computed_value > INT16_MAX || computed_value < INT16_MIN) {
            saturation_detected = true;
            break;
        }
    }

    // If saturation is detected, log a warning (only log once based on the atomic count)
    if (saturation_detected && ++onnxruntime::saturation_count < 2) {
        std::cerr << "Warning: saturation detected in VPMADDUBSW instruction." << std::endl;
    }
}
