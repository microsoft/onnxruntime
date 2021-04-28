// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/prepacked_weights_container.h"
#include "core/framework/allocatormgr.h"

namespace onnxruntime {

AllocatorPtr PrepackedWeightsContainer::GetAllocator(const std::string& device_name) {
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

const PrepackedWeight& PrepackedWeightsContainer::GetCachedWeight(const std::string& key) {
  ORT_ENFORCE(HasCachedWeight(key), "PrepackedWeightsContainer does not have an initializer with the same key: ", key);
  return initialized_tensor_name_to_prepacked_weights_[key];
}

void PrepackedWeightsContainer::WriteCachedWeight(const std::string& key, PrepackedWeight&& packed_weight) {
  ORT_ENFORCE(!HasCachedWeight(key), "PrepackedWeightsContainer already has an initializer with the same key: ", key);
  initialized_tensor_name_to_prepacked_weights_.insert({key, std::move(packed_weight)});
}

bool PrepackedWeightsContainer::HasCachedWeight(const std::string& key) {
  return initialized_tensor_name_to_prepacked_weights_.find(key) !=
         initialized_tensor_name_to_prepacked_weights_.end();
}

}  // namespace onnxruntime
