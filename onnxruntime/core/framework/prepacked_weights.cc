// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "prepacked_weights.h"

#include <cstring>

#include "core/framework/murmurhash3.h"

namespace onnxruntime {

HashValue PrePackedWeights::GetHash() const {
  // Adaptation of the hashing logic of the KernelDef class

  uint32_t hash[4] = {0, 0, 0, 0};

  auto hash_int8_t_buffer = [&hash](void* data, size_t len) { MurmurHash3::x86_128(data, len, hash[0], &hash); };

  ORT_ENFORCE(buffers_.size() == buffer_sizes_.size());
  for (size_t iter = 0; iter < buffers_.size(); ++iter) {
    // some pre-packed buffers may be null if they were just "place-holders" occupying an index
    // in the "buffers_" vector
    if (buffers_[iter].get() != nullptr) {
      hash_int8_t_buffer(buffers_[iter].get(), buffer_sizes_[iter]);
    }
  }

  HashValue returned_hash = hash[0] & 0xfffffff8;  // save low 3 bits for hash version info in case we need it in the future
  returned_hash |= uint64_t(hash[1]) << 32;

  return returned_hash;
}

bool PrePackedWeights::IsEquivalentTo(const PrePackedWeights& other) const noexcept {
  if (buffers_.size() != buffer_sizes_.size() ||
      other.buffers_.size() != other.buffer_sizes_.size() ||
      buffers_.size() != other.buffers_.size() ||
      has_kernel_owned_packed_weights_ != other.has_kernel_owned_packed_weights_) {
    return false;
  }

  for (size_t i = 0; i < buffers_.size(); ++i) {
    if (buffer_sizes_[i] != other.buffer_sizes_[i] ||
        (buffers_[i] == nullptr) != (other.buffers_[i] == nullptr)) {
      return false;
    }

    if (buffers_[i] != nullptr && buffer_sizes_[i] != 0 &&
        std::memcmp(buffers_[i].get(), other.buffers_[i].get(), buffer_sizes_[i]) != 0) {
      return false;
    }
  }

  return true;
}

}  // namespace onnxruntime
