// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cassert>
#include <type_traits>
#include "core/common/common.h"
#include <gsl/gsl>

namespace onnxruntime {

// Stores 8 packed 3-bit unsigned elements in 3 bytes (24 bits exactly).
//
// Bit layout (within the 24-bit word, little-endian):
//   word    = byte[0] | (byte[1] << 8) | (byte[2] << 16)
//   value_i = (word >> (i * 3)) & 0x7    for i in [0, 8)
//
// This layout matches vLLM's TurboQuant store kernel
// (vllm/v1/attention/ops/triton_turboquant_store.py) so binaries produced by
// either runtime are interchangeable.
//
// Used for:
//   - K-cache 3-bit Lloyd-Max codebook indices in TurboQuant.
//   - V-cache 3-bit uniform-quant indices when value_quant_bits == 3.
//
// Storage tip: callers should always operate on whole 8-element groups
// (i.e. tensor extents along the packed axis must be multiples of 8). The
// 3-bit encoding has no clean partial-byte representation otherwise.
struct UInt3x8 {
  static constexpr uint8_t min_val = 0;
  static constexpr uint8_t max_val = 7;
  static constexpr size_t kElementsPerPack = 8;
  static constexpr size_t kBytesPerPack = 3;

  std::byte bytes_[kBytesPerPack]{};

  UInt3x8() = default;

  // Construct from 8 unpacked uint8 elements. All must be in [0, 7].
  explicit UInt3x8(const uint8_t (&values)[kElementsPerPack]) {
    uint32_t word = 0;
    for (size_t i = 0; i < kElementsPerPack; ++i) {
      assert(values[i] <= max_val);
      word |= (static_cast<uint32_t>(values[i]) & 0x7u) << (i * 3);
    }
    bytes_[0] = static_cast<std::byte>(word & 0xFFu);
    bytes_[1] = static_cast<std::byte>((word >> 8) & 0xFFu);
    bytes_[2] = static_cast<std::byte>((word >> 16) & 0xFFu);
  }

  inline uint8_t GetElem(size_t index) const {
    assert(index < kElementsPerPack);
    const uint32_t word = static_cast<uint32_t>(bytes_[0]) |
                          (static_cast<uint32_t>(bytes_[1]) << 8) |
                          (static_cast<uint32_t>(bytes_[2]) << 16);
    return static_cast<uint8_t>((word >> (index * 3)) & 0x7u);
  }

  inline void SetElem(size_t index, uint8_t val) {
    assert(index < kElementsPerPack);
    assert(val <= max_val);
    uint32_t word = static_cast<uint32_t>(bytes_[0]) |
                    (static_cast<uint32_t>(bytes_[1]) << 8) |
                    (static_cast<uint32_t>(bytes_[2]) << 16);
    const uint32_t mask = ~(0x7u << (index * 3));
    word = (word & mask) | ((static_cast<uint32_t>(val) & 0x7u) << (index * 3));
    bytes_[0] = static_cast<std::byte>(word & 0xFFu);
    bytes_[1] = static_cast<std::byte>((word >> 8) & 0xFFu);
    bytes_[2] = static_cast<std::byte>((word >> 16) & 0xFFu);
  }

  // Number of UInt3x8 packs required to store n unpacked 3-bit elements.
  // Caller must ensure n is a multiple of 8.
  static constexpr size_t CalcNumPacks(size_t num_3bit_elems) {
    return num_3bit_elems / kElementsPerPack;
  }

  // Bytes required to store n unpacked 3-bit elements.
  static constexpr size_t CalcNumBytes(size_t num_3bit_elems) {
    return CalcNumPacks(num_3bit_elems) * kBytesPerPack;
  }

  // Bulk unpack: turn N packed groups (3 bytes each) into N*8 bytes [0, 7].
  static bool Unpack(gsl::span<uint8_t> dst, gsl::span<const UInt3x8> src) {
    if (dst.size() != src.size() * kElementsPerPack) {
      return false;
    }
    for (size_t i = 0; i < src.size(); ++i) {
      for (size_t j = 0; j < kElementsPerPack; ++j) {
        dst[i * kElementsPerPack + j] = src[i].GetElem(j);
      }
    }
    return true;
  }

  // Bulk pack: turn N*8 bytes [0, 7] into N packed groups.
  static bool Pack(gsl::span<UInt3x8> dst, gsl::span<const uint8_t> src) {
    if (src.size() != dst.size() * kElementsPerPack) {
      return false;
    }
    for (size_t i = 0; i < dst.size(); ++i) {
      uint8_t buf[kElementsPerPack];
      for (size_t j = 0; j < kElementsPerPack; ++j) {
        buf[j] = src[i * kElementsPerPack + j];
        assert(buf[j] <= max_val);
      }
      dst[i] = UInt3x8(buf);
    }
    return true;
  }
};

static_assert(sizeof(UInt3x8) == 3, "UInt3x8 must be exactly 3 bytes");

}  // namespace onnxruntime
