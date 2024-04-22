// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cassert>
#include <type_traits>
#include "core/common/common.h"
#include "core/common/gsl.h"

namespace onnxruntime {

template <typename T>
struct UnpackedTypeTraits;

template <>
struct UnpackedTypeTraits<int8_t> {
  static constexpr int8_t min_val = -8;
  static constexpr int8_t max_val = 7;
};

template <>
struct UnpackedTypeTraits<uint8_t> {
  static constexpr uint8_t min_val = 0;
  static constexpr uint8_t max_val = 15;
};

template <typename T>
struct Int4x2Base {
  using unpacked_type = T;
  static constexpr unpacked_type min_val = UnpackedTypeTraits<T>::min_val;
  static constexpr unpacked_type max_val = UnpackedTypeTraits<T>::max_val;

  unpacked_type elems{};

  Int4x2Base() = default;
  explicit Int4x2Base(uint8_t bits) {
    elems = static_cast<unpacked_type>(bits);
  }
  Int4x2Base(unpacked_type val0, unpacked_type val1) {
    elems = static_cast<unpacked_type>(((val1 & 0xF) << 4) | (val0 & 0xF));
  }

  inline unpacked_type GetElem0() const {
    if constexpr (std::is_same_v<unpacked_type, int8_t>) {
      // Need to sign-extend lower 4-bits by left shifting and then doing an arithmetic right shift.
      return static_cast<unpacked_type>(static_cast<unpacked_type>((elems << 4)) >> 4);
    } else {
      return static_cast<unpacked_type>(elems & 0xF);
    }
  }

  inline unpacked_type GetElem1() const {
    return static_cast<unpacked_type>(elems >> 4);
  }

  inline unpacked_type GetElem(size_t index) const {
    assert(index <= 1);
    const uint8_t shift = 4 * static_cast<uint8_t>(index);

    if constexpr (std::is_same_v<unpacked_type, int8_t>) {
      // if index is 0, need to sign-extend lower 4-bits by left shifting and then doing an arithmetic right shift.
      const uint8_t unshift = 4 - shift;
      return static_cast<unpacked_type>(static_cast<unpacked_type>((elems >> shift) << unshift) >> unshift);
    } else {
      return static_cast<unpacked_type>((elems >> shift) & 0xF);
    }
  }

  inline void SetElem(size_t index, unpacked_type val) {
    assert(index <= 1);
    const uint8_t shift = 4 * static_cast<uint8_t>(index);
    const unpacked_type mask = 0xF << shift;

    elems &= ~mask;                                             // Clear 4-bit element to 0
    elems |= static_cast<unpacked_type>((val & 0xF) << shift);  // Set 4-bit element to val
  }

  inline uint8_t ToBits() const {
    return static_cast<uint8_t>(elems);
  }

  static bool Unpack(gsl::span<unpacked_type> dst, gsl::span<const Int4x2Base<T>> src) {
    if (((dst.size() + 1) / 2) != src.size()) {
      return false;
    }

    for (size_t i = 0; i < dst.size(); i++) {
      size_t r = i >> 1;   // i / 2;
      size_t c = i & 0x1;  // i % 2;
      dst[i] = src[r].GetElem(c);
    }

    return true;
  }

  static bool Pack(gsl::span<Int4x2Base<T>> dst, gsl::span<const unpacked_type> src) {
    if (((src.size() + 1) / 2) != dst.size()) {
      return false;
    }

    size_t src_i = 0;
    size_t dst_i = 0;

    for (; src_i < src.size() - 1; src_i += 2) {
      dst[dst_i++] = Int4x2Base<T>(src[src_i], src[src_i + 1]);
    }

    if (src_i < src.size()) {
      dst[dst_i] = Int4x2Base<T>(src[src_i], 0);
    }

    return true;
  }
};

using Int4x2 = Int4x2Base<int8_t>;
using UInt4x2 = Int4x2Base<uint8_t>;
static_assert(sizeof(Int4x2) == sizeof(int8_t));
static_assert(sizeof(UInt4x2) == sizeof(uint8_t));
}  // namespace onnxruntime
