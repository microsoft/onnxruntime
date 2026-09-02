// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>

namespace onnxruntime {

inline float Float6ToFloat(uint8_t bits, int exponent_bits, int mantissa_bits, int exponent_bias) {
  const uint8_t sign = bits >> 5;
  const uint8_t exponent = (bits >> mantissa_bits) & ((1u << exponent_bits) - 1);
  const uint8_t mantissa = bits & ((1u << mantissa_bits) - 1);
  const float significand = exponent == 0
                                ? static_cast<float>(mantissa)
                                : static_cast<float>((1u << mantissa_bits) | mantissa);
  const float value = std::ldexp(significand, exponent == 0
                                                  ? 1 - exponent_bias - mantissa_bits
                                                  : static_cast<int>(exponent) - exponent_bias - mantissa_bits);
  return sign == 0 ? value : -value;
}

inline uint8_t FloatToFloat6(float value, int exponent_bits, int mantissa_bits, int exponent_bias) {
  if (std::isnan(value)) {
    return 0x20;
  }

  const uint8_t sign = std::signbit(value) ? 0x20 : 0;
  if (std::isinf(value)) {
    return static_cast<uint8_t>(sign | 0x1F);
  }
  const float magnitude = std::abs(value);
  const float max_finite = Float6ToFloat(0x1F, exponent_bits, mantissa_bits, exponent_bias);
  if (magnitude > max_finite) {
    return static_cast<uint8_t>(sign | 0x1F);
  }
  uint8_t best = sign;
  float best_distance = std::numeric_limits<float>::infinity();
  for (uint8_t bits = 0; bits < 0x20; ++bits) {
    const float candidate = Float6ToFloat(bits, exponent_bits, mantissa_bits, exponent_bias);
    const float distance = std::abs(magnitude - candidate);
    if (distance < best_distance || (distance == best_distance && (bits & 1) == 0)) {
      best = static_cast<uint8_t>(sign | bits);
      best_distance = distance;
    }
  }
  return best;
}

struct Float6E2M3 {
  uint8_t val{0};

  struct FromBitsT {};
  static constexpr FromBitsT FromBits() { return FromBitsT(); }
  constexpr Float6E2M3(uint8_t bits, FromBitsT) : val(bits & 0x3F) {}
  explicit Float6E2M3(float value) : val(FloatToFloat6(value, 2, 3, 1)) {}
  constexpr uint8_t ToBits() const { return val; }
  operator float() const { return Float6ToFloat(val, 2, 3, 1); }
  static constexpr size_t CalcNumFloat6Bytes(size_t num_float6_elems) {
    return num_float6_elems - num_float6_elems / 4;
  }
};

struct Float6E3M2 {
  uint8_t val{0};

  struct FromBitsT {};
  static constexpr FromBitsT FromBits() { return FromBitsT(); }
  constexpr Float6E3M2(uint8_t bits, FromBitsT) : val(bits & 0x3F) {}
  explicit Float6E3M2(float value) : val(FloatToFloat6(value, 3, 2, 3)) {}
  constexpr uint8_t ToBits() const { return val; }
  operator float() const { return Float6ToFloat(val, 3, 2, 3); }
  static constexpr size_t CalcNumFloat6Bytes(size_t num_float6_elems) {
    return num_float6_elems - num_float6_elems / 4;
  }
};

static_assert(sizeof(Float6E2M3) == sizeof(uint8_t));
static_assert(sizeof(Float6E3M2) == sizeof(uint8_t));

}  // namespace onnxruntime
