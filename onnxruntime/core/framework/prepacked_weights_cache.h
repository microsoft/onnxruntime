// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/tensor.h"

namespace onnxruntime {

struct PackedWeight {
  // Some weights may be associated with multiple pre-packed buffers.
  // Hence we hold them in containers. It is upto the dev implementing each PrePack()
  // method to define what gets stored in which position.
  std::vector<BufferUniquePtr> buffers_;
  std::vector<size_t> weights_sizes_;
  std::vector<TensorShape> shapes_;
  bool has_cached_ = false;
};

// TODO: Make this class thread-safe ?
class PrepackedWeightsCache {
 public:
  PrepackedWeightsCache() {
  }

  ~PrepackedWeightsCache() = default;

  AllocatorPtr GetAllocator(const char* device_name) {
    auto name = std::string(device_name);

    auto iter = allocators_.find(name);

    if (iter != allocators_.end())
      return iter->second;

    // Support only CPU based allocators for now.
    // as pre-packing is only supported by CPU kernels for now.
    if (name == CPU) {
      /*we do not need an arena based allocator*/
      AllocatorCreationInfo device_info{[](int) { return onnxruntime::make_unique<TAllocator>(); },
                                        0, false};
      auto allocator = CreateAllocator(device_info);

      allocators_[name] = allocator;
    } else {
      ORT_THROW("Unsupported device allocator in the context of pre-packed weights caching: ", name);
    }

    return allocators_[name];
  }

  PackedWeight& GetOrCreateCachedWeight(const std::string& initializer_name, /*out*/ bool& is_cached) {
    auto iter = initialized_tensor_name_to_prepacked_weights_.find(initializer_name);
    if (iter != initialized_tensor_name_to_prepacked_weights_.end()) {
      is_cached = true;
      return iter->second;
    } else {
      is_cached = false;
      PackedWeight temp;
      initialized_tensor_name_to_prepacked_weights_.insert({initializer_name, std::move(temp)});
    }
    return initialized_tensor_name_to_prepacked_weights_[initializer_name];
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(PrepackedWeightsCache);

 private:
  // Define allocators ahead of the container containing tensors because the allocators
  // needs to destructed after the container containing the pre-packed cached tensors
  // because the Tensor buffers will be de-allocated using these allocators
  std::unordered_map<std::string, AllocatorPtr> allocators_;
  std::unordered_map<std::string, PackedWeight> initialized_tensor_name_to_prepacked_weights_;
};

}  // namespace onnxruntime
