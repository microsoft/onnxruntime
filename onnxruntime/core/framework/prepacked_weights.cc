// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "prepacked_weights.h"
#include "core/framework/murmurhash3.h"

namespace onnxruntime {

uint64_t PrePackedWeights::GetHash() const {
  // Adaptation of the hashing logic of the KernelDef class

  uint32_t hash[4] = {0, 0, 0, 0};

  auto hash_int8_t_buffer = [&hash](void* data, int len) { MurmurHash3::x86_128(data, len, hash[0], &hash); };

  ORT_ENFORCE(buffers_.size() == buffer_sizes_.size());
  for (size_t iter = 0; iter < buffers_.size(); ++iter) {
    // some pre-packed buffers may be null if they were just "place-holders" occupying an index
    // in the "buffers_" vector
    if (buffers_[iter].get() != nullptr) {
      hash_int8_t_buffer(buffers_[iter].get(), static_cast<int>(buffer_sizes_[iter]));
    }
  }

  uint64_t returned_hash = hash[0] & 0xfffffff8;  // save low 3 bits for hash version info in case we need it in the future
  returned_hash |= uint64_t(hash[1]) << 32;

  return returned_hash;
}

}  // namespace onnxruntime
