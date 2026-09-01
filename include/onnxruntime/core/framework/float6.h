// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>

namespace onnxruntime {

struct Float6E2M3 {
  uint8_t val{0};

  struct FromBitsT {};
  static constexpr FromBitsT FromBits() { return FromBitsT(); }
  constexpr Float6E2M3(uint8_t bits, FromBitsT) : val(bits & 0x3F) {}
  constexpr uint8_t ToBits() const { return val; }
  static constexpr size_t CalcNumFloat6Bytes(size_t num_float6_elems) {
    return num_float6_elems - num_float6_elems / 4;
  }
};

struct Float6E3M2 {
  uint8_t val{0};

  struct FromBitsT {};
  static constexpr FromBitsT FromBits() { return FromBitsT(); }
  constexpr Float6E3M2(uint8_t bits, FromBitsT) : val(bits & 0x3F) {}
  constexpr uint8_t ToBits() const { return val; }
  static constexpr size_t CalcNumFloat6Bytes(size_t num_float6_elems) {
    return num_float6_elems - num_float6_elems / 4;
  }
};

static_assert(sizeof(Float6E2M3) == sizeof(uint8_t));
static_assert(sizeof(Float6E3M2) == sizeof(uint8_t));

}  // namespace onnxruntime
