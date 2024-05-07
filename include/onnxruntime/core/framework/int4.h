// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cassert>
#include <type_traits>
#include "core/common/common.h"
#include "core/common/gsl.h"

namespace onnxruntime {

template <bool Signed>
struct Int4Traits;

template <>
struct Int4Traits<true> {
  using UnpackedType = int8_t;
  static constexpr int8_t min_val = -8;
  static constexpr int8_t max_val = 7;
};

template <>
struct Int4Traits<false> {
  using UnpackedType = uint8_t;
  static constexpr uint8_t min_val = 0;
  static constexpr uint8_t max_val = 15;
};

/// <summary>
/// Stores 2 packed 4-bit elements in 1 byte.
/// </summary>
/// <typeparam name="Signed">Set to true if signed int4, or false if unsigned uint4.</typeparam>
template <bool Signed>
struct Int4x2Base {
  using UnpackedType = typename Int4Traits<Signed>::UnpackedType;
  static constexpr UnpackedType min_val = Int4Traits<Signed>::min_val;
  static constexpr UnpackedType max_val = Int4Traits<Signed>::max_val;

  uint8_t bits_{};

  Int4x2Base() = default;

  explicit Int4x2Base(uint8_t bits) {
    bits_ = bits;
  }

  Int4x2Base(UnpackedType val0, UnpackedType val1) {
    bits_ = static_cast<uint8_t>(((val1 & 0xF) << 4) | (val0 & 0xF));
  }

  static inline int8_t SignExtendLower4Bits(uint8_t bits) {
    // Sign-extend lower 4-bits by left shifting and then doing an arithmetic right shift.
    constexpr uint8_t Shift = (sizeof(int32_t) * 8) - 4;
    return static_cast<int8_t>((static_cast<int32_t>(bits) << Shift) >> Shift);
  }

  inline UnpackedType GetElem0() const {
    if constexpr (Signed) {
      return SignExtendLower4Bits(bits_);
    } else {
      return static_cast<UnpackedType>(bits_ & 0xF);
    }
  }

  inline UnpackedType GetElem1() const {
    const uint8_t val = static_cast<uint8_t>((bits_ >> 4) & 0xF);

    if constexpr (Signed) {
      return SignExtendLower4Bits(val);
    } else {
      return val;
    }
  }

  inline UnpackedType GetElem(size_t index) const {
    assert(index <= 1);
    const uint8_t shift = 4 * static_cast<uint8_t>(index);
    const uint8_t val = static_cast<uint8_t>((bits_ >> shift) & 0xF);

    if constexpr (Signed) {
      return SignExtendLower4Bits(val);
    } else {
      return val;
    }
  }

  inline void SetElem(size_t index, UnpackedType val) {
    assert(index <= 1);
    const uint8_t shift = 4 * static_cast<uint8_t>(index);
    const uint8_t mask = 0xF << shift;

    bits_ &= ~mask;                                       // Clear 4-bit element to 0
    bits_ |= static_cast<uint8_t>((val & 0xF) << shift);  // Set 4-bit element to val
  }

  inline uint8_t ToBits() const {
    return bits_;
  }

  static size_t CalcNumInt4Pairs(size_t num_int4_elems) {
    return (num_int4_elems + 1) / 2;
  }

  static bool Unpack(gsl::span<UnpackedType> dst, gsl::span<const Int4x2Base<Signed>> src) {
    if (CalcNumInt4Pairs(dst.size()) != src.size()) {
      return false;
    }

    for (size_t i = 0; i < dst.size(); i++) {
      size_t r = i >> 1;   // i / 2;
      size_t c = i & 0x1;  // i % 2;
      dst[i] = src[r].GetElem(c);
    }

    return true;
  }

  static bool Pack(gsl::span<Int4x2Base<Signed>> dst, gsl::span<const UnpackedType> src) {
    if (CalcNumInt4Pairs(src.size()) != dst.size()) {
      return false;
    }

    size_t src_i = 0;
    size_t dst_i = 0;

    for (; src_i < src.size() - 1; src_i += 2) {
      dst[dst_i++] = Int4x2Base<Signed>(src[src_i], src[src_i + 1]);
    }

    if (src_i < src.size()) {
      dst[dst_i] = Int4x2Base<Signed>(src[src_i], 0);
    }

    return true;
  }
};

using Int4x2 = Int4x2Base<true>;
using UInt4x2 = Int4x2Base<false>;
static_assert(sizeof(Int4x2) == sizeof(uint8_t));
static_assert(sizeof(UInt4x2) == sizeof(uint8_t));
}  // namespace onnxruntime
