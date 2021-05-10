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
    AllocatorCreationInfo device_info{[](int) { return std::make_unique<TAllocator>(); },
                                      0, false};
    auto allocator = CreateAllocator(device_info);

    allocators_[device_name] = allocator;

    return allocator;

  } else {
    ORT_THROW("Unsupported device allocator in the context of pre-packed weights caching: ", device_name);
  }
}

const PrePackedWeights& PrepackedWeightsContainer::GetCachedWeight(const std::string& key) {
  ORT_ENFORCE(HasCachedWeight(key), "PrepackedWeightsContainer does not have an initializer with the same key: ", key);
  return initialized_tensor_name_to_prepacked_weights_[key];
}

void PrepackedWeightsContainer::WriteCachedWeight(const std::string& key, PrePackedWeights&& packed_weight) {
  ORT_ENFORCE(!HasCachedWeight(key), "PrepackedWeightsContainer already has an initializer with the same key: ", key);
  initialized_tensor_name_to_prepacked_weights_.insert({key, std::move(packed_weight)});
}

bool PrepackedWeightsContainer::HasCachedWeight(const std::string& key) const {
  return initialized_tensor_name_to_prepacked_weights_.find(key) !=
         initialized_tensor_name_to_prepacked_weights_.end();
}

bool PrepackedWeightsContainer::HasPrepackedWeightForOpTypeAndConstantInitializer(
    const std::string& op_type,
    const void* const_initialized_tensor_data) const {
  const std::string& key = GenerateKeyFromOpTypeAndInitializerData(op_type, const_initialized_tensor_data);
  return op_type_tensor_data_memory_map_.find(key) != op_type_tensor_data_memory_map_.end();
}

void PrepackedWeightsContainer::MarkHasPrepackedWeightForOpTypeAndConstantInitializer(const std::string& op_type,
                                                                                      const void* const_initialized_tensor_data) {
  const std::string& key = GenerateKeyFromOpTypeAndInitializerData(op_type, const_initialized_tensor_data);
  op_type_tensor_data_memory_map_.insert(key);
}

std::string PrepackedWeightsContainer::GenerateKeyFromOpTypeAndInitializerData(
    const std::string& op_type,
    const void* const_initialized_tensor_data) const {
  std::ostringstream ss_2;
  ss_2 << op_type;
  ss_2 << "+";
  // TODO: Should we hash the contents of the data buffer instead of looking at just the data buffer pointer ?
  // For now, this seems like a reasonable approach given that for shared initializers, users are likely to
  // use the same memory across different shared initializers.
  ss_2 << reinterpret_cast<uintptr_t>(const_initialized_tensor_data);

  return ss_2.str();
}

size_t PrepackedWeightsContainer::GetNumberOfElements() const {
  return initialized_tensor_name_to_prepacked_weights_.size();
}

}  // namespace onnxruntime
