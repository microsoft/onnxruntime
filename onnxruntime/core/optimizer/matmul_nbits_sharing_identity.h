// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "core/framework/murmurhash3.h"
#include "core/framework/tensor.h"

namespace onnxruntime {

// Stable, content-derived identity for a fusion-generated MatMulNBits weight group, used to share its
// pre-packed buffer across sessions. The id is identical for the same model in any session and differs
// whenever a semantic input differs. accuracy_level is hashed so buffers packed for different compute
// types never collide. Pass zero_point only when it is an actual kernel input.
inline std::string ComputeMatMulNBitsSharingId(const Tensor& weight, const Tensor& scale,
                                               const std::optional<Tensor>& zero_point,
                                               int64_t N, int64_t K, int64_t block_size,
                                               int64_t bits, int64_t accuracy_level) {
  uint32_t hash[4] = {0, 0, 0, 0};
  auto hash_bytes = [&hash](const void* data, size_t len) {
    MurmurHash3::x86_128(data, len, hash[0], &hash);
  };
  hash_bytes(weight.DataRaw(), weight.SizeInBytes());
  hash_bytes(scale.DataRaw(), scale.SizeInBytes());
  if (zero_point) {
    hash_bytes(zero_point->DataRaw(), zero_point->SizeInBytes());
  }
  const int64_t params[] = {N, K, block_size, bits, accuracy_level};
  hash_bytes(params, sizeof(params));
  return "MatMulNBits.DQ:" + std::to_string((static_cast<uint64_t>(hash[1]) << 32) | hash[0]);
}

}  // namespace onnxruntime
