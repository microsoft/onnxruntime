// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "vulkan_common.h"
#include "vulkan_memory_allocation_helper.h"

namespace onnxruntime {

class VulkanBuffer {
 public:
  VulkanBuffer(VulkanMemoryAllocationHelper& memory_alloc_helper, size_t size,
               const void* host_data = nullptr,
               VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
               VkSharingMode shared = VK_SHARING_MODE_EXCLUSIVE,
               VkFlags requirements_mask = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

  virtual ~VulkanBuffer();

  VkBuffer Get() const {
    return buffer_;
  }

  size_t Size() const {
    return size_;
  }

  void* Map(int start = 0, int size = -1) const;

  void Unmap() const;

  void Release();

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(VulkanBuffer);

 private:
  VulkanMemoryAllocationHelper& memory_alloc_helper_;
  std::pair<void*, int> memory_;
  VkBuffer buffer_;
  size_t size_;
  bool memory_released_ = false;
  VkBufferUsageFlags buffer_usage_flags_;
  VkSharingMode shared_;
};

}  // namespace onnxruntime