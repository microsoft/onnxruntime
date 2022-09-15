// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "vulkan_allocator.h"
#include "vulkan_tensor.h"

namespace onnxruntime {

// VulkanAllocator methods
VulkanAllocator::VulkanAllocator(const VulkanExecutionProviderInfo& info,
                                 VulkanMemoryAllocationHelper& memory_alloc_helper,
                                 const VkPhysicalDeviceLimits& memory_limits)
    : IAllocator(OrtMemoryInfo(VULKAN, OrtAllocatorType::OrtDeviceAllocator,
                               OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, info.device_id))),
      memory_alloc_helper_(memory_alloc_helper),
      memory_limits_(memory_limits) {
}

void* VulkanAllocator::Alloc(size_t /*size*/) {
  return nullptr;
}

void* VulkanAllocator::Alloc(const TensorShape& shape, MLDataType data_type) {
  return new VulkanTensor(shape, data_type,
                          memory_alloc_helper_,
                          memory_limits_);
}

void VulkanAllocator::Free(void* ptr) {
  delete static_cast<VulkanTensor*>(ptr);
}

}  // namespace onnxruntime
