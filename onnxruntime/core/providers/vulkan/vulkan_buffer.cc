// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "vulkan_buffer.h"

namespace onnxruntime {

VulkanBuffer::VulkanBuffer(VulkanMemoryPool& memory_pool, size_t size, const void* host_data,
                           VkBufferUsageFlags buffer_usage_flags, VkSharingMode shared, VkFlags requirements_mask)
    : memory_pool_(memory_pool) {
  ORT_ENFORCE(size > 0);
  size_ = size;
  shared_ = shared;
  buffer_ = memory_pool.AllocVkBuffer(size, buffer_usage_flags, shared);
  buffer_usage_flags_ = buffer_usage_flags;

  const auto& device = memory_pool.GetLogicalDevice();

  VkMemoryRequirements memory_requirements;
  VK_CALL_RETURNS_VOID(vkGetBufferMemoryRequirements(device, buffer_, &memory_requirements));

  memory_ = memory_pool_.Alloc(memory_requirements, requirements_mask);
  auto* device_memory = static_cast<VulkanMemory*>(memory_.first);

  if (nullptr != host_data) {
    void* data = nullptr;
    VK_CALL(vkMapMemory(device, device_memory->Get(), memory_.second, size_, 0 /*flag, not used*/, &data));
    memcpy(data, host_data, size);
    VK_CALL_RETURNS_VOID(vkUnmapMemory(device, device_memory->Get()));
  }

  VK_CALL(vkBindBufferMemory(device, buffer_, device_memory->Get(), memory_.second));
}

VulkanBuffer::~VulkanBuffer() {
  memory_pool_.FreeVkBuffer(buffer_);

  Release();
}

void* VulkanBuffer::Map(int start, int size) const {
  if (size < 0) {
    size = static_cast<int>(size_);
  }

  const auto& device = memory_pool_.GetLogicalDevice();

  auto* device_memory = static_cast<VulkanMemory*>(memory_.first);

  void* data = nullptr;

  VK_CALL(vkMapMemory(device, device_memory->Get(), start + memory_.second, size_, 0, &data));

  return data;
}

void VulkanBuffer::Unmap() const {
  const auto& device = memory_pool_.GetLogicalDevice();

  auto* device_memory = static_cast<VulkanMemory*>(memory_.first);

  VK_CALL_RETURNS_VOID(vkUnmapMemory(device, device_memory->Get()));
}

void VulkanBuffer::Release() {
  if (memory_released_) {
    return;
  }

  memory_released_ = true;
  memory_pool_.Free(memory_);
}

}  // namespace onnxruntime
