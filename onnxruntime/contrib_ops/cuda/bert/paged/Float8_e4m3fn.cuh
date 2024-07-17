// Modifications Copyright (c) Microsoft.
// Copyright (c) 2016-present, Facebook, Inc.
// Licensed under the BSD-3-Clause License.

// copied from https://github.com/pytorch/pytorch/blob/8f82a44a5b/c10/util/Float8_e4m3fn.h
// and https://github.com/pytorch/pytorch/blob/8f82a44a5b/c10/util/floating_point_utils.h

#pragma once

/// Defines the Float8_e4m3fn type (8-bit floating-point) including conversions
/// to standard C types and basic arithmetic operations. Note that arithmetic
/// operations are implemented by converting to floating point and
/// performing the operation in float32.
/// Binary configuration:
/// s eeee mmm
/// 1 sign bit
/// 4 exponent bits
/// 3 mantissa bits
/// bias = 7
///
/// Implementation based on the paper https://arxiv.org/pdf/2209.05433.pdf
/// and inspired by Half implementation from pytorch/c10/util/Half.h

#include <type_traits>

#include <cmath>
#include <cstdint>
#include <climits>
#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>


__forceinline__ __host__ __device__ float
fp32_from_bits(uint32_t w) {
  union {
    uint32_t as_bits;
    float as_value;
  } fp32 = {w};
  return fp32.as_value;
}

__forceinline__ __host__ __device__ uint32_t
fp32_to_bits(float f) {
  union {
    float as_value;
    uint32_t as_bits;
  } fp32 = {f};
  return fp32.as_bits;
}

/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 8-bit floating-point number in fp8 E4M3FN format, in bit representation.
 */
__forceinline__ __host__ __device__ uint8_t
fp8e4m3fn_from_fp32_value(float f) {
  /*
   * Binary representation of 480.0f, which is the first value
   * not representable in fp8e4m3fn range:
   * 0 1111 111 - fp8e4m3fn
   * 0 10000111 11100000000000000000000 - fp32
   */
  constexpr uint32_t fp8_max = UINT32_C(1087) << 20;

  /*
   * A mask for converting fp32 numbers lower than fp8e4m3fn normal range
   * into denorm representation
   * magic number: ((127 - 7) + (23 - 3) + 1)
   */
  constexpr uint32_t denorm_mask = UINT32_C(141) << 23;

  uint32_t f_bits = fp32_to_bits(f);

  uint8_t result = 0u;

  /*
   * Extract the sign of the input number into the high bit of the 32-bit word:
   *
   *      +---+----------------------------------+
   *      | S |0000000 00000000 00000000 00000000|
   *      +---+----------------------------------+
   * Bits  31                 0-31
   */
  const uint32_t sign = f_bits & UINT32_C(0x80000000);

  /*
   * Set sign bit to 0
   */
  f_bits ^= sign;

  if (f_bits >= fp8_max) {
    // NaN - all exponent and mantissa bits set to 1
    result = 0x7f;
  } else {
    if (f_bits < (UINT32_C(121) << 23)) {
      // Input number is smaller than 2^(-6), which is the smallest
      // fp8e4m3fn normal number
      f_bits =
          fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
      result = static_cast<uint8_t>(f_bits - denorm_mask);
    } else {
      // resulting mantissa is odd
      uint8_t mant_odd = (f_bits >> 20) & 1;

      // update exponent, rounding bias part 1
      f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;

      // rounding bias part 2
      f_bits += mant_odd;

      // take the bits!
      result = static_cast<uint8_t>(f_bits >> 20);
    }
  }

  result |= static_cast<uint8_t>(sign >> 24);
  return result;
}
