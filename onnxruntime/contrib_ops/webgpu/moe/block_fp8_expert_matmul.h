// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

namespace onnxruntime {
namespace contrib {
namespace webgpu {

constexpr bool BlockFp8MatrixDispatchFits(uint32_t rows, uint32_t cols, uint32_t limit) {
  return (static_cast<uint64_t>(cols) + 15) / 16 <= limit &&
         (static_cast<uint64_t>(rows) + 63) / 64 <= limit;
}

uint64_t BlockFp8MatrixDispatchCount();

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
