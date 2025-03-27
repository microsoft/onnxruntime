/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    platform.cpp

Abstract:

    This module implements logic to select the best configuration for the
    this platform.

--*/

#include <immintrin.h>
#include <stdint.h>
#include <iostream>

void print_m256i_16(const char* label, __m256i reg) {
    alignas(32) int16_t values[16];  // __m256i has 16x int16_t elements
    _mm256_store_si256(reinterpret_cast<__m256i*>(values), reg);

    std::cout << label << ": ";
    for (int i = 0; i < 16; ++i) {
        std::cout << std::hex << values[i] << " ";
    }
    std::cout << std::endl;
}

void print_m256i_8(const char* label, __m256i reg) {
    alignas(32) int8_t values[32];  // __m256i has 16x int16_t elements
    _mm256_store_si256(reinterpret_cast<__m256i*>(values), reg);

    std::cout << label << ": ";
    for (int i = 0; i < 32; ++i) {
        std::cout << std::dec << static_cast<uint32_t>(values[i]) << " ";
    }
    std::cout << std::endl;
}

#if 0
//#ifdef SATURATION_DETECTION_ENABLED
extern "C" void CheckForSaturationBeforeMul(const __m256i* unsigned_ptr, const __m256i* signed_ptr) {
    static  int counter = 0;

    __m256i unsigned_data = _mm256_loadu_si256(unsigned_ptr);
    __m256i signed_data = _mm256_loadu_si256(signed_ptr);

    __m256i result = _mm256_maddubs_epi16(unsigned_data, signed_data);

    alignas(32) uint16_t unsigned_bytes[16];  // __m256i has 16x int16_t elements
    _mm256_store_si256(reinterpret_cast<__m256i*>(unsigned_bytes), unsigned_data);
    alignas(32) int16_t signed_bytes[16];  // __m256i has 16x int16_t elements
    _mm256_store_si256(reinterpret_cast<__m256i*>(signed_bytes), signed_data);

    //(void)input;  // Suppress unused variable warning
    //(void)multiplier;  // Suppress unused variable warning

    // Example saturation check: print values
    if (counter < 10) {
        print_m256i_8("unsigned_data:\t", unsigned_data);
        print_m256i_8("signed_data:\t", signed_data);
        print_m256i_16("result:\t", result);
        counter++;
        int32_t dest =
            static_cast<int32_t>(signed_bytes[0]) * static_cast<int32_t>(static_cast<uint32_t>(unsigned_bytes[0])) +
            static_cast<int32_t>(signed_bytes[1]) * static_cast<int32_t>(static_cast<uint32_t>(unsigned_bytes[1]));
        std::cout << "Checking for saturation..." << std::hex << "0x" << dest << std::endl;
    }
    
    return;  // Modify this to return true if saturation is detected
}
//#endif  // SATURATION_DETECTION_ENABLED
#endif

#if 1

int saturate_counter = 0;

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
    if (saturation_detected && saturate_counter < 2) {
        std::cerr << "[WARNING] Saturation detected in _mm256_maddubs_epi16 operation!" << std::endl;
        saturate_counter++;
    }

}
#endif
