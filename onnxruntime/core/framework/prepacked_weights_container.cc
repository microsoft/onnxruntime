// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/prepacked_weights_container.h"
#include "core/framework/allocatormgr.h"

namespace onnxruntime {

class BufferHasher {
 public:
  std::size_t operator()(const int8_t* data, size_t num_bytes) const {
    std::size_t ret = num_bytes;
    for (size_t i = 0; i < num_bytes; ++i) {
      ret ^= std::hash<int32_t>()(data[i]);
    }
    return ret;
  }
};

uint64_t PrepackedWeight::GetHash() {
  // Adaptation of the hashing logic of the KernelDef class

  uint32_t hash[4] = {0, 0, 0, 0};

  auto hash_size_t = [&hash](size_t i) { MurmurHash3::x86_128(&i, sizeof(size_t), hash[0], &hash); };

  ORT_ENFORCE(buffers_.size() == buffer_sizes_.size());
  for (size_t iter = 0; iter < buffers_.size(); ++iter) {
    BufferHasher buffer_hasher;
    size_t buffer_hash = buffer_hasher(reinterpret_cast<int8_t*>(buffers_[iter].get()), buffer_sizes_[iter]);
    hash_size_t(buffer_hash);
  }

  uint64_t returned_hash = hash[0] & 0xfffffff8;  // save low 3 bits for hash version info in case we need it in the future
  returned_hash |= uint64_t(hash[1]) << 32;

  return returned_hash;
}

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

bool PrepackedWeightsContainer::HasPrepackedWeightForOpTypeAndConstantInitializer(const std::string& op_type,
                                                                                  const void* const_initialized_tensor_data) {
  std::ostringstream ss_2;
  ss_2 << op_type;
  ss_2 << "+";
  // TODO: Should we hash the contents of the data buffer instead of looking at just the data buffer pointer ?
  // For now, this seems like a reasonable approach given that for shared initializers, users are likely to
  // use the same memory across different shared initializers.
  ss_2 << reinterpret_cast<uintptr_t>(const_initialized_tensor_data);

  const std::string& key = ss_2.str();

  return op_type_tensor_data_memory_map_.find(key) != op_type_tensor_data_memory_map_.end();
}

void PrepackedWeightsContainer::MarkHasPrepackedWeightForOpTypeAndConstantInitializer(const std::string& op_type,
                                                                                      const void* const_initialized_tensor_data) {
  std::ostringstream ss_2;
  ss_2 << op_type;
  ss_2 << "+";
  // TODO: Should we hash the contents of the data buffer instead of looking at just the data buffer pointer ?
  // For now, this seems like a reasonable approach given that for shared initializers, users are likely to
  // use the same memory across different shared initializers.
  ss_2 << reinterpret_cast<uintptr_t>(const_initialized_tensor_data);

  const std::string& key = ss_2.str();

  op_type_tensor_data_memory_map_.insert(key);
}

}  // namespace onnxruntime
