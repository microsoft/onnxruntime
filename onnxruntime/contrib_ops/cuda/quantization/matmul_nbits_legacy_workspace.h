// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include "core/common/safeint.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

struct LegacyMatMulNBitsWorkspaceInfo {
  size_t size_bytes;
  int64_t k_padded;
  int64_t scratch_rows;
  bool use_chunked;
};

// Pure workspace math shared by Level-2 declaration and runtime allocation.
inline std::optional<LegacyMatMulNBitsWorkspaceInfo> ComputeLegacyMatMulNBitsWorkspaceInfo(
    int64_t n, int64_t k, int64_t block_size, size_t element_size,
    bool column_wise_quantization, bool has_g_idx,
    bool force_chunked, int64_t chunk_target_rows) {
  if (n <= 0 || k <= 0 || block_size <= 0 || element_size == 0 || chunk_target_rows <= 0) {
    return std::nullopt;
  }

  try {
    const SafeInt<int64_t> safe_k_padded =
        ((SafeInt<int64_t>(k) + block_size - 1) / block_size) * block_size;
    const int64_t k_padded = static_cast<int64_t>(safe_k_padded);
    const SafeInt<size_t> full_size =
        SafeInt<size_t>(n) * SafeInt<size_t>(k_padded) * element_size;
    constexpr size_t kChunkingThresholdBytes = 256 * 1024 * 1024;
    const bool use_chunked =
        column_wise_quantization && !has_g_idx &&
        (force_chunked ||
         (static_cast<size_t>(full_size) > kChunkingThresholdBytes &&
          SafeInt<int64_t>(n) > SafeInt<int64_t>(chunk_target_rows) * 2));
    const int64_t scratch_rows = use_chunked ? chunk_target_rows : n;
    const SafeInt<size_t> size_bytes =
        SafeInt<size_t>(scratch_rows) * SafeInt<size_t>(k_padded) * element_size;
    return LegacyMatMulNBitsWorkspaceInfo{
        static_cast<size_t>(size_bytes), k_padded, scratch_rows, use_chunked};
  } catch (const OnnxRuntimeException&) {
    return std::nullopt;
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
