// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "vulkan_common.h"
#include "core/framework/allocator.h"

namespace onnxruntime {

class VulkanAllocator : IAllocator {
 public:
  VulkanAllocator(const VkDevice& vulkan_logical_device, int vulkan_memory_type_index);

  void* Alloc(size_t size) override;

  void Free(void* ptr) override;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(VulkanAllocator);

 private:
  const VkDevice& vulkan_logical_device_;
  int vulkan_memory_type_index_;
};

class VulkanMemory {
 public:
  VulkanMemory(const VkDevice& vulkan_logical_device, const VkMemoryAllocateInfo& info);
  virtual ~VulkanMemory();

  VkDeviceMemory Get() const {
    return vulkan_memory_;
  }

  uint32_t Type() const {
    return type_index_;
  }

  VkDeviceSize Size() const {
    return size_;
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(VulkanMemory);

 private:
  const VkDevice& vulkan_logical_device_;
  VkDeviceMemory vulkan_memory_;
  uint32_t type_index_;
  VkDeviceSize size_;
};

class VulkanMemoryPool {
 public:
  VulkanMemoryPool(const VkDevice& vulkan_logical_device);
  virtual ~VulkanMemoryPool();

  // VulkanMemory* , offset
  std::pair<void*, int> Alloc(const VkMemoryRequirements& requirements, VkFlags extraMask, bool seperate = false);
  void Free(std::pair<void*, int>& memory);

  // Free Memory
  void clear();

  const VkDevice& GetLogicalDevice() {
    return vulkan_logical_device;
  }

  // Return MB
  float computeSize() const;

  // For buffer fast alloc
  VkBuffer allocBuffer(size_t size, VkBufferUsageFlags flags, VkSharingMode shared);
  void returnBuffer(VkBuffer buffer, size_t size, VkBufferUsageFlags flags, VkSharingMode shared);

  // For image fast alloc
  VkImage allocImage(const std::tuple<VkImageType, uint32_t, uint32_t, uint32_t, VkFormat>& info);
  void returnImage(VkImage dst, std::tuple<VkImageType, uint32_t, uint32_t, uint32_t, VkFormat>&& info);

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(VulkanMemoryPool);

 private:
  // MemoryTypeIndex, Size, Memory
  std::vector<std::shared_ptr<BufferAllocator>> mAllocators;

  const VkDevice& vulkan_logical_device;
};
}  // namespace onnxruntime