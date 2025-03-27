/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    platform.cpp

Abstract:

    This module implements logic to select the best configuration for the
    this platform.

--*/

#include <atomic>
#include <immintrin.h>
#include <stdint.h>
#include <iostream>

namespace onnxruntime {

std::atomic<int> saturation_count{0};

}

extern "C" void CheckForSaturationBeforeMul(const __m256i* unsigned_ptr, const __m256i* signed_ptr) {

    // Load data from memory
    __m256i unsigned_data = _mm256_loadu_si256(unsigned_ptr);
    __m256i signed_data = _mm256_loadu_si256(signed_ptr);

    // Perform multiplication with saturation detection
    __m256i result = _mm256_maddubs_epi16(unsigned_data, signed_data);  // Multiply and add adjacent bytes

    // Extract original values
    alignas(32) uint8_t unsigned_bytes[32];  // Unsigned input values
    alignas(32) int8_t signed_bytes[32];     // Signed input values
    alignas(32) int16_t result_values[16];   // 16-bit results from _mm256_maddubs_epi16

    _mm256_store_si256(reinterpret_cast<__m256i*>(unsigned_bytes), unsigned_data);
    _mm256_store_si256(reinterpret_cast<__m256i*>(signed_bytes), signed_data);
    _mm256_store_si256(reinterpret_cast<__m256i*>(result_values), result);

    //constexpr int16_t INT16_MAX = 32767;
    //constexpr int16_t INT16_MIN = -32768;
    bool saturation_detected = false;

    // Compute in higher precision (int32_t) and compare with 16-bit limits
    //std::cout << "computed_value: ";
    for (int i = 0; i < 16; ++i) {
        int32_t computed_value =
            static_cast<int32_t>(signed_bytes[2 * i]) * static_cast<int32_t>(static_cast<uint32_t>(unsigned_bytes[2 * i])) +
            static_cast<int32_t>(signed_bytes[2 * i + 1]) * static_cast<int32_t>(static_cast<uint32_t>(unsigned_bytes[2 * i + 1]));

        //std::cout << std::hex << "0x" << computed_value << " ";

        // If the computed value exceeds the 16-bit range, saturation occurred
        if (computed_value > INT16_MAX || computed_value < INT16_MIN) {
            saturation_detected = true;
            break;
        }
    }
    //std::cout << std::endl;

    // Log a warning if saturation is detected
    if (saturation_detected && ++onnxruntime::saturation_count < 2) {
        std::cerr << "[WARNING] Saturation detected in VPMADDUBSW instruction!" << std::endl;
    }

}