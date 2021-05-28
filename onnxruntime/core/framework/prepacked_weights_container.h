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

  // Returns an allocator keyed by device name.
  // If an allocator doesn't exist for that specific device, an allocator
  // is created and stored in a member to be returned on subsequent calls.
  // Currently, the only supported device is "Cpu".
  AllocatorPtr GetOrCreateAllocator(const std::string& device_name);

  // Returns the PrePackedWeights instance pertaining to the provided key.
  // The key is : op_type + "+" + hash_of_prepacked_buffers_in_the_PrepackedWeights_instance.
  // Throws an exception if the key doesn't exist
  const PrePackedWeights& GetWeight(const std::string& key) const;

  // Writes the PrePackedWeights instance pertaining to the provided key.
  // The key is : op_type + "+" + hash_of_prepacked_buffers_in_the_PrepackedWeights_instance.
  // Returns a boolean indicating if the insertion took place.
  bool WriteWeight(const std::string& key, PrePackedWeights&& packed_weight);

  // Returns a boolean indicating if there is a PrePackedWeights instance
  // pertaining to the provided key.
  // The key is : op_type + "+" + hash_of_prepacked_buffers_in_the_PrepackedWeights_instance.
  bool HasWeight(const std::string& key) const;

  // Returns the number of elements in the container
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

  // This is an unordered map that holds a mapping between a composite key
  // to PrePackedWeights instances.
  // The key is : op_type + "+" + hash_of_prepacked_buffers_in_the_PrepackedWeights_instance.
  std::unordered_map<std::string, PrePackedWeights> prepacked_weights_map_;
};

}  // namespace onnxruntime
