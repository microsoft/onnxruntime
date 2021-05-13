// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include <unordered_set>
#include <string>
#include <cstdint>

#include "core/framework/buffer_deleter.h"

#include "core/framework/allocator.h"
#include "core/platform/ort_mutex.h"
#include "prepacked_weights.h"

namespace onnxruntime {

class PrepackedWeightsContainer final {
 public:
  PrepackedWeightsContainer() {
  }

  ~PrepackedWeightsContainer() = default;

  AllocatorPtr GetAllocator(const std::string& device_name);

  const PrePackedWeights& GetWeight(const std::string& key);

  void WriteWeight(const std::string& key, PrePackedWeights&& packed_weight);

  bool HasWeight(const std::string& key) const;

  // Not thread-safe
  size_t GetNumberOfElements() const;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(PrepackedWeightsContainer);

  // Resource to be acquired by the method that is going to invoke calls to the kernels'
  // PrePack() methods and does the read/write into the pre-packed weights' container.
  // We only want to invoke PrePack() on a kernel that doesn't have a cached version
  // of its pre-packed weight.
  OrtMutex mutex_;

  // Define allocators ahead of the container containing tensors because the allocators
  // needs to destructed after the container containing the pre-packed cached tensors
  // because the Tensor buffers will be de-allocated using these allocators
  std::unordered_map<std::string, AllocatorPtr> allocators_;
  std::unordered_map<std::string, PrePackedWeights> initialized_tensor_name_to_prepacked_weights_map_;
};

}  // namespace onnxruntime
