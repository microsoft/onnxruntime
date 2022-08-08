// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "vulkan_memory_pool.h"

namespace onnxruntime {

// VulkanAllocator methods
VulkanAllocator::VulkanAllocator(const VkDevice& vulkan_logical_device, int vulkan_memory_type_index)
    : IAllocator(OrtMemoryInfo(VULKAN, OrtAllocatorType::OrtDeviceAllocator,
                               OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0),
                               0, OrtMemTypeDefault)),  // TODO: Should the device id be made configurable ?
      vulkan_logical_device_(vulkan_logical_device),
      vulkan_memory_type_index_(vulkan_memory_type_index) {}

void* VulkanAllocator::Alloc(size_t size) {
  VkMemoryAllocateInfo info;
  info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  info.pNext = nullptr;
  info.allocationSize = size;
  info.memoryTypeIndex = vulkan_memory_type_index_;
  return new VulkanMemory(vulkan_logical_device_, info);
}

void VulkanAllocator::Free(void* ptr) {
  delete static_cast<VulkanMemory*>(ptr);
}

// VulkanMemory methods
VulkanMemory::VulkanMemory(const VkDevice& vulkan_logical_device, const VkMemoryAllocateInfo& info) : vulkan_logical_device_(vulkan_logical_device) {
  VK_CALL(vkAllocateMemory(vulkan_logical_device_, &info, nullptr, &vulkan_memory_));
  type_index_ = info.memoryTypeIndex;
  size_ = info.allocationSize;
}

VulkanMemory::~VulkanMemory() {
  VK_CALL_RETURNS_VOID(vkFreeMemory(vulkan_logical_device_, vulkan_memory_, nullptr));
}

}  // namespace onnxruntime
