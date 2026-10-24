// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>

namespace onnxruntime::contrib::cuda {

template <int Bits>
__device__ __forceinline__ unsigned int LoadRowLocalPackedCode(
    const uint8_t* row, size_t row_bytes, size_t logical_index) {
  static_assert(Bits >= 1 && Bits <= 8);
  const size_t bit_offset = logical_index * Bits;
  const size_t byte_offset = bit_offset / 8;
  const unsigned int shift = static_cast<unsigned int>(bit_offset % 8);
  unsigned int word = row[byte_offset];
  if (shift + Bits > 8 && byte_offset + 1 < row_bytes) {
    word |= static_cast<unsigned int>(row[byte_offset + 1]) << 8;
  }
  return (word >> shift) & ((1u << Bits) - 1u);
}

}  // namespace onnxruntime::contrib::cuda
