// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <cstring>

namespace onnxruntime {
namespace telum {

/**
 * @brief Endianness utilities for s390x (big-endian) architecture
 *
 * s390x is a big-endian architecture, while most development/testing
 * happens on little-endian x86_64. These utilities ensure correct
 * data handling across architectures.
 */

// Detect endianness at compile time
#if defined(__s390x__) || defined(__s390__)
  #define TELUM_BIG_ENDIAN 1
  #define TELUM_LITTLE_ENDIAN 0
#elif defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
  #define TELUM_BIG_ENDIAN 0
  #define TELUM_LITTLE_ENDIAN 1
#else
  // Runtime detection fallback
  #define TELUM_BIG_ENDIAN 0
  #define TELUM_LITTLE_ENDIAN 0
  #define TELUM_RUNTIME_ENDIAN_CHECK 1
#endif

/**
 * @brief Check if system is big-endian at runtime
 */
inline bool IsBigEndian() {
#ifdef TELUM_RUNTIME_ENDIAN_CHECK
  union {
    uint32_t i;
    char c[4];
  } test = {0x01020304};
  return test.c[0] == 1;
#else
  return TELUM_BIG_ENDIAN;
#endif
}

/**
 * @brief Byte swap for 16-bit values
 */
inline uint16_t ByteSwap16(uint16_t value) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_bswap16(value);
#elif defined(_MSC_VER)
  return _byteswap_ushort(value);
#else
  return (value >> 8) | (value << 8);
#endif
}

/**
 * @brief Byte swap for 32-bit values
 */
inline uint32_t ByteSwap32(uint32_t value) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_bswap32(value);
#elif defined(_MSC_VER)
  return _byteswap_ulong(value);
#else
  return ((value >> 24) & 0x000000FF) |
         ((value >> 8)  & 0x0000FF00) |
         ((value << 8)  & 0x00FF0000) |
         ((value << 24) & 0xFF000000);
#endif
}

/**
 * @brief Byte swap for 64-bit values
 */
inline uint64_t ByteSwap64(uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_bswap64(value);
#elif defined(_MSC_VER)
  return _byteswap_uint64(value);
#else
  return ((value >> 56) & 0x00000000000000FFULL) |
         ((value >> 40) & 0x000000000000FF00ULL) |
         ((value >> 24) & 0x0000000000FF0000ULL) |
         ((value >> 8)  & 0x00000000FF000000ULL) |
         ((value << 8)  & 0x000000FF00000000ULL) |
         ((value << 24) & 0x0000FF0000000000ULL) |
         ((value << 40) & 0x00FF000000000000ULL) |
         ((value << 56) & 0xFF00000000000000ULL);
#endif
}

/**
 * @brief Convert from host to network byte order (big-endian)
 */
template <typename T>
inline T HostToNetwork(T value) {
  if constexpr (sizeof(T) == 1) {
    return value;
  } else if constexpr (sizeof(T) == 2) {
#if TELUM_BIG_ENDIAN
    return value;
#else
    uint16_t temp;
    std::memcpy(&temp, &value, sizeof(T));
    temp = ByteSwap16(temp);
    T result;
    std::memcpy(&result, &temp, sizeof(T));
    return result;
#endif
  } else if constexpr (sizeof(T) == 4) {
#if TELUM_BIG_ENDIAN
    return value;
#else
    uint32_t temp;
    std::memcpy(&temp, &value, sizeof(T));
    temp = ByteSwap32(temp);
    T result;
    std::memcpy(&result, &temp, sizeof(T));
    return result;
#endif
  } else if constexpr (sizeof(T) == 8) {
#if TELUM_BIG_ENDIAN
    return value;
#else
    uint64_t temp;
    std::memcpy(&temp, &value, sizeof(T));
    temp = ByteSwap64(temp);
    T result;
    std::memcpy(&result, &temp, sizeof(T));
    return result;
#endif
  }
  return value;
}

/**
 * @brief Convert from network byte order (big-endian) to host
 */
template <typename T>
inline T NetworkToHost(T value) {
  // Network to host is same as host to network for byte swapping
  return HostToNetwork(value);
}

/**
 * @brief Verify data integrity across endianness
 *
 * This is useful for debugging endianness issues in tests
 */
inline bool VerifyEndianness() {
  // Test with known pattern
  uint32_t test_value = 0x12345678;
  uint32_t network_value = HostToNetwork(test_value);
  uint32_t host_value = NetworkToHost(network_value);

  return (host_value == test_value);
}

/**
 * @brief Get endianness as string for logging
 */
inline const char* GetEndiannessString() {
  return IsBigEndian() ? "big-endian" : "little-endian";
}

}  // namespace telum
}  // namespace onnxruntime

// Made with Bob
