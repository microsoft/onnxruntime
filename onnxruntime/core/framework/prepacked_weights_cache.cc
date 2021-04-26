// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/prepacked_weights_cache.h"
#include "core/framework/allocatormgr.h"

namespace onnxruntime {

AllocatorPtr PrepackedWeightsCache::GetAllocator(const std::string& device_name) {
  auto iter = allocators_.find(device_name);

  if (iter != allocators_.end())
    return iter->second;

  // Support only CPU based allocators for now.
  // as pre-packing is only supported by CPU kernels for now.
  if (device_name == CPU) {
    /*we do not need an arena based allocator*/
    AllocatorCreationInfo device_info{[](int) { return onnxruntime::make_unique<TAllocator>(); },
                                      0, false};
    auto allocator = CreateAllocator(device_info);

    allocators_[device_name] = allocator;

    return allocator;

  } else {
    ORT_THROW("Unsupported device allocator in the context of pre-packed weights caching: ", device_name);
  }
}

const PrepackedWeight& PrepackedWeightsCache::GetCachedWeight(const std::string& initializer_name) {
  ORT_ENFORCE(HasCachedWeight(initializer_name), "PrepackedWeightsCache does not have an initializer with the name: ", initializer_name);
  return initialized_tensor_name_to_prepacked_weights_[initializer_name];
}

void PrepackedWeightsCache::WriteCachedWeight(const std::string& initializer_name, PrepackedWeight&& packed_weight) {
  ORT_ENFORCE(!HasCachedWeight(initializer_name), "PrepackedWeightsCache already has an initializer with the same name: ", initializer_name);
  initialized_tensor_name_to_prepacked_weights_.insert({initializer_name, std::move(packed_weight)});
}

bool PrepackedWeightsCache::HasCachedWeight(const std::string& initializer_name) {
  return initialized_tensor_name_to_prepacked_weights_.find(initializer_name) !=
         initialized_tensor_name_to_prepacked_weights_.end();
}

}  // namespace onnxruntime
