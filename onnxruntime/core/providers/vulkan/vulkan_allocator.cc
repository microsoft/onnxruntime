// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/vulkan/vulkan_allocator.h"

#include <iostream>

#include "kompute/Manager.hpp"

#include "core/framework/ortdevice.h"

namespace onnxruntime {
namespace vulkan {

namespace {
OrtMemoryInfo CreateMemoryInfo(OrtDevice device) {
  if (device.MemType() == OrtDevice::MemType::DEFAULT) {
    return OrtMemoryInfo("VulkanAllocator",
                         OrtAllocatorType::OrtDeviceAllocator,
                         device,
                         /*id*/ 0,
                         OrtMemTypeDefault);
  } else {
    return OrtMemoryInfo("VulkanStagingAllocator",
                         OrtAllocatorType::OrtDeviceAllocator,
                         device,
                         /*id*/ 0,
                         OrtMemTypeCPU);
  };
}
}  // namespace

VulkanBufferAllocator::VulkanBufferAllocator(OrtDevice device, VmaAllocator& allocator)
    : IAllocator(CreateMemoryInfo(device)), allocator_{allocator} {
}

void* VulkanBufferAllocator::Alloc(size_t size) {
  VkBufferCreateInfo bufferInfo = {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;  // assuming aligned externally
  bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

  VmaAllocationCreateInfo allocInfo = {};
  allocInfo.usage = usage_;

  VkBuffer buffer;
  VmaAllocation allocation;
  vmaCreateBuffer(allocator_, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr);

  return new std::pair<VkBuffer, VmaAllocation>(buffer, allocation);
}

void VulkanBufferAllocator::Free(void* p) {
  auto pair = static_cast<std::pair<VkBuffer, VmaAllocation>*>(p);
  auto [buffer, allocation] = *pair;
  vmaDestroyBuffer(allocator_, buffer, allocation);

  delete pair;
}

}  // namespace vulkan
}  // namespace onnxruntime
