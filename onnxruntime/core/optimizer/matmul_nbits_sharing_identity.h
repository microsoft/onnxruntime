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
  // MurmurHash3 fmix64 finalizer: a bijection that avalanches a 64-bit value so each input bit affects
  // every output bit.
  auto fmix64 = [](uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
  };
  // Fold each segment's full 128-bit hash into the 64-bit accumulator and carry the whole accumulator
  // forward, not just a 32-bit seed. Every bit of weight/scale/zero_point/params therefore reaches the
  // id, so collision resistance tracks the 64-bit id width instead of the ~2^32 a chain forwarding only
  // hash[0] would give. A collision would let one weight group adopt another's already-packed buffer and
  // silently compute a wrong result, so the wider margin is worth the few extra mixing ops.
  uint64_t acc = 0;
  auto mix = [&acc, &fmix64](const void* data, size_t len) {
    uint32_t h[4];
    MurmurHash3::x86_128(data, len, static_cast<uint32_t>(acc), h);
    acc = fmix64(acc ^ ((static_cast<uint64_t>(h[1]) << 32) | h[0]));
    acc = fmix64(acc ^ ((static_cast<uint64_t>(h[3]) << 32) | h[2]));
  };
  mix(weight.DataRaw(), weight.SizeInBytes());
  mix(scale.DataRaw(), scale.SizeInBytes());
  if (zero_point) {
    mix(zero_point->DataRaw(), zero_point->SizeInBytes());
  }
  const int64_t params[] = {N, K, block_size, bits, accuracy_level};
  mix(params, sizeof(params));
  return "MatMulNBits.DQ:" + std::to_string(acc);
}

}  // namespace onnxruntime
