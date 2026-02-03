// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cassert>
#include <type_traits>
#include "core/common/common.h"
#include <gsl/gsl>

namespace onnxruntime {

template <bool Signed>
struct Int2Traits;

template <>
struct Int2Traits<true> {
  using UnpackedType = int8_t;
  static constexpr int8_t min_val = -2;
  static constexpr int8_t max_val = 1;
};

template <>
struct Int2Traits<false> {
  using UnpackedType = uint8_t;
  static constexpr uint8_t min_val = 0;
  static constexpr uint8_t max_val = 3;
};

/// <summary>
/// Stores 4 packed 2-bit elements in 1 byte.
/// Packing follows ONNX spec: x0 | (x1 << 2) | (x2 << 4) | (x3 << 6)
/// </summary>
/// <typeparam name="Signed">Set to true if signed int2, or false if unsigned uint2.</typeparam>
template <bool Signed>
struct Int2x4Base {
  using UnpackedType = typename Int2Traits<Signed>::UnpackedType;
  static constexpr UnpackedType min_val = Int2Traits<Signed>::min_val;
  static constexpr UnpackedType max_val = Int2Traits<Signed>::max_val;

  std::byte bits_{};

  Int2x4Base() = default;

  explicit Int2x4Base(std::byte bits) {
    bits_ = bits;
  }

  Int2x4Base(UnpackedType val0, UnpackedType val1, UnpackedType val2, UnpackedType val3) {
    bits_ = static_cast<std::byte>(
        (val0 & 0x3) |
        ((val1 & 0x3) << 2) |
        ((val2 & 0x3) << 4) |
        ((val3 & 0x3) << 6));
  }

  static inline int8_t SignExtendLower2Bits(std::byte bits) {
    // Sign-extend lower 2-bits by left shifting and then doing an arithmetic right shift.
    constexpr uint8_t shift = (sizeof(int32_t) * 8) - 2;
    return static_cast<int8_t>((static_cast<int32_t>(bits) << shift) >> shift);
  }

  inline UnpackedType GetElem(size_t index) const {
    assert(index <= 3);
    const uint8_t shift = 2 * static_cast<uint8_t>(index);
    const std::byte val = (bits_ >> shift) & std::byte{0x3};

    if constexpr (Signed) {
      return SignExtendLower2Bits(val);
    } else {
      return static_cast<UnpackedType>(val);
    }
  }

  inline void SetElem(size_t index, UnpackedType val) {
    assert(index <= 3);
    const uint8_t shift = 2 * static_cast<uint8_t>(index);
    const std::byte clear_mask = ~(std::byte{0x3} << shift);

    bits_ &= clear_mask;                                    // Clear 2-bit element to 0
    bits_ |= static_cast<std::byte>((val & 0x3) << shift);  // Set 2-bit element to val
  }

  inline std::byte ToBits() const {
    return bits_;
  }

  /// <summary>
  /// Calculates the number of packed byte units needed to store the given number of 2-bit elements.
  /// Each byte stores 4 x 2-bit elements.
  /// </summary>
  static size_t CalcNumInt2Quads(size_t num_int2_elems) {
    return (num_int2_elems + 3) / 4;
  }

  /// <summary>
  /// Copy a source buffer of 2-bit elements (packed) into a destination buffer of 8-bit elements (unpacked).
  /// </summary>
  /// <param name="dst">Destination buffer to store unpacked 8-bit elements</param>
  /// <param name="src">Source buffer with 2-bit elements</param>
  /// <returns>True on success</returns>
  static bool Unpack(gsl::span<UnpackedType> dst, gsl::span<const Int2x4Base<Signed>> src) {
    if (CalcNumInt2Quads(dst.size()) != src.size()) {
      return false;
    }

    if (src.empty()) {
      return true;
    }

    for (size_t i = 0; i < dst.size(); i++) {
      size_t byte_idx = i >> 2;   // i / 4
      size_t elem_idx = i & 0x3;  // i % 4
      dst[i] = src[byte_idx].GetElem(elem_idx);
    }

    return true;
  }

  /// <summary>
  /// Copy a source buffer of 8-bit elements (unpacked) into a destination buffer of 2-bit elements (packed).
  /// </summary>
  /// <param name="dst">Destination buffer to store packed 2-bit elements</param>
  /// <param name="src">Source buffer with 8-bit elements</param>
  /// <returns>True on success</returns>
  static bool Pack(gsl::span<Int2x4Base<Signed>> dst, gsl::span<const UnpackedType> src) {
    if (CalcNumInt2Quads(src.size()) != dst.size()) {
      return false;
    }

    if (src.empty()) {
      return true;
    }

    size_t src_i = 0;
    size_t dst_i = 0;
    const size_t full_quads = src.size() / 4;

    // Process complete groups of 4 elements
    for (; dst_i < full_quads; dst_i++) {
      dst[dst_i] = Int2x4Base<Signed>(src[src_i], src[src_i + 1], src[src_i + 2], src[src_i + 3]);
      src_i += 4;
    }

    // Handle remaining elements (1-3)
    if (src_i < src.size()) {
      UnpackedType vals[4] = {0, 0, 0, 0};
      size_t remaining = src.size() - src_i;
      for (size_t j = 0; j < remaining; j++) {
        vals[j] = src[src_i + j];
      }
      dst[dst_i] = Int2x4Base<Signed>(vals[0], vals[1], vals[2], vals[3]);
    }

    return true;
  }

  /// <summary>
  /// Returns hierarchical indices for a packed int2 element from the given element index.
  ///
  /// Usage:
  ///   Int2x4* data = ...;
  ///   auto indices = GetTensorElemIndices(5);  // 6th int2 element
  ///   int8_t elem = data[indices.first].GetElem(indices.second);
  /// </summary>
  /// <param name="index">Index of 2-bit element</param>
  /// <returns>Pair of (byte_index, element_index_within_byte)</returns>
  static inline std::pair<size_t, size_t> GetTensorElemIndices(size_t index) {
    return {index >> 2, index & 0x3};
  }
};

using Int2x4 = Int2x4Base<true>;
using UInt2x4 = Int2x4Base<false>;
static_assert(sizeof(Int2x4) == sizeof(std::byte));
static_assert(sizeof(UInt2x4) == sizeof(std::byte));

}  // namespace onnxruntime
