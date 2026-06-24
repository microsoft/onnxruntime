// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/matmul_nbits_sharing_identity.h"

#include <cstdint>
#include <cstring>
#include <sstream>

namespace onnxruntime {

namespace {

// FNV-1a hash over a raw byte buffer, processed 8 bytes at a time with a byte tail. Reading words
// via memcpy keeps it alignment-safe and lets the compiler emit plain loads.
constexpr uint64_t kFnvOffsetBasis = 0xcbf29ce484222325ULL;
constexpr uint64_t kFnvPrime = 0x100000001b3ULL;

uint64_t HashBuffer(const void* data, size_t size, uint64_t hash) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  size_t num_words = size / sizeof(uint64_t);
  for (size_t i = 0; i < num_words; ++i) {
    uint64_t word;
    std::memcpy(&word, p + i * sizeof(uint64_t), sizeof(uint64_t));
    hash ^= word;
    hash *= kFnvPrime;
  }
  for (size_t i = num_words * sizeof(uint64_t); i < size; ++i) {
    hash ^= p[i];
    hash *= kFnvPrime;
  }
  return hash;
}

uint64_t HashTensor(const Tensor& tensor, uint64_t hash) {
  // Mix in the shape so two tensors with identical bytes but different shapes do not collide.
  for (int64_t dim : tensor.Shape().GetDims()) {
    hash = HashBuffer(&dim, sizeof(dim), hash);
  }
  return HashBuffer(tensor.DataRaw(), tensor.SizeInBytes(), hash);
}

}  // namespace

std::string ComputeMatMulNBitsSharingIdentity(const Tensor& weight,
                                              const Tensor& scale,
                                              const Tensor* zero_point) {
  uint64_t hash = kFnvOffsetBasis;
  hash = HashTensor(weight, hash);
  hash = HashTensor(scale, hash);
  if (zero_point != nullptr) {
    hash = HashTensor(*zero_point, hash);
  }

  std::ostringstream oss;
  oss << "mnb_b_" << std::hex << hash;
  return oss.str();
}

}  // namespace onnxruntime
