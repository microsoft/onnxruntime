// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cassert>
#include "endian.h"
#include "core/common/common.h"
#include "core/common/gsl.h"

namespace onnxruntime {
struct Int4x2 {
  using unpacked_type = int8_t;
  static constexpr unpacked_type min_val = -8;
  static constexpr unpacked_type max_val = 7;

  int8_t val_0 : 4;
  int8_t val_1 : 4;

  Int4x2() : val_0{0}, val_1{0} {}
  Int4x2(uint8_t bits) {
    val_0 = static_cast<int8_t>(bits & 0xF);
    val_1 = static_cast<int8_t>((bits >> 4) & 0xF);
  }
  Int4x2(int8_t lo, int8_t hi) : val_0{lo}, val_1{hi} {}

  inline int8_t operator[](size_t index) const {
    assert(index <= 1);
    return index == 0 ? val_0 : val_1;
  }

  inline uint8_t ToBits() const {
    return (static_cast<uint8_t>(val_1) << 4) | (static_cast<uint8_t>(val_0) & 0xF);
  }

  static bool Unpack(gsl::span<int8_t> dst, gsl::span<const Int4x2> src) {
    if (((dst.size() + 1) / 2) != src.size()) {
      return false;
    }

    for (size_t i = 0; i < dst.size(); i++) {
      size_t r = i >> 1;   // i / 2;
      size_t c = i & 0x1;  // i % 2;
      dst[i] = src[r][c];
    }

    return true;
  }

  static bool Pack(gsl::span<Int4x2> dst, gsl::span<const int8_t> src) {
    if (((src.size() + 1) / 2) != dst.size()) {
      return false;
    }

    size_t src_i = 0;
    size_t dst_i = 0;

    for (; src_i < src.size() - 1; src_i += 2) {
      dst[dst_i++] = Int4x2(src[src_i], src[src_i + 1]);
    }

    if (src_i < src.size()) {
      dst[dst_i] = Int4x2(src[src_i], 0);
    }

    return true;
  }
};

static_assert(sizeof(Int4x2) == sizeof(int8_t));

struct UInt4x2 {
  using unpacked_type = uint8_t;
  static constexpr unpacked_type min_val = 0;
  static constexpr unpacked_type max_val = 15;

  uint8_t val_0 : 4;
  uint8_t val_1 : 4;

  UInt4x2() : val_0{0}, val_1{0} {}
  UInt4x2(uint8_t bits) {
    val_0 = bits & 0xF;
    val_1 = (bits >> 4) & 0xF;
  }
  UInt4x2(uint8_t lo, uint8_t hi) : val_0{lo}, val_1{hi} {}

  inline uint8_t operator[](size_t index) const {
    assert(index <= 1);
    return index == 0 ? val_0 : val_1;
  }

  inline uint8_t ToBits() const {
    return (static_cast<uint8_t>(val_1) << 4) | (static_cast<uint8_t>(val_0) & 0xF);
  }

  static bool Unpack(gsl::span<uint8_t> dst, gsl::span<const UInt4x2> src) {
    if (((dst.size() + 1) / 2) != src.size()) {
      return false;
    }

    for (size_t i = 0; i < dst.size(); i++) {
      size_t r = i >> 1;   // i / 2;
      size_t c = i & 0x1;  // i % 2;
      dst[i] = src[r][c];
    }

    return true;
  }

  static bool Pack(gsl::span<UInt4x2> dst, gsl::span<const uint8_t> src) {
    if (((src.size() + 1) / 2) != dst.size()) {
      return false;
    }

    size_t src_i = 0;
    size_t dst_i = 0;

    for (; src_i < src.size() - 1; src_i += 2) {
      dst[dst_i++] = UInt4x2(src[src_i], src[src_i + 1]);
    }

    if (src_i < src.size()) {
      dst[dst_i] = UInt4x2(src[src_i], 0);
    }

    return true;
  }
};

static_assert(sizeof(UInt4x2) == sizeof(uint8_t));
}  // namespace onnxruntime
