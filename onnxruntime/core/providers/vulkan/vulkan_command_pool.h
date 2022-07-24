// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "vulkan_common.h"

namespace onnxruntime {

class VulkanCommandBuffer {
 public:
  VulkanCommandBuffer(const VkDevice& vulkan_logical_device,
                      VulkanCommandPool& vulkan_command_pool);

  virtual ~VulkanCommandBuffer();

  VkCommandBuffer Get() const {
    return buffer_;
  }

  void Begin(VkCommandBufferUsageFlags flags) const;

  void End() const;

  enum BarrierType : int {
    READ_WRITE = 0,
    WRITE_WRITE,
  };

  void BarrierSource(VkBuffer source, size_t start, size_t end, BarrierType type = READ_WRITE) const;

 private:
  const VkDevice& vulkan_logical_device_;
  VkCommandBuffer buffer_;
  VulkanCommandPool& vulkan_command_pool_;
};

class VulkanCommandPool {
 public:
  VulkanCommandPool(const VkDevice& vulkan_logical_device, uint32_t vulkan_queue_family_index);

  virtual ~VulkanCommandPool();

  VulkanCommandBuffer* AllocBuffer();

  VkCommandPool Get() const {
    return vulkan_command_pool_;
  }

  void SubmitAndWait(VkCommandBuffer buffer, VkQueue vulkan_queue) const;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(VulkanCommandPool);

  friend class VulkanCommandBuffer;

 private:
  const VkDevice& vulkan_logical_device_;
  VkCommandPool vulkan_command_pool_;
  uint32_t vulkan_queue_family_index_;
  std::vector<VkCommandBuffer> free_vulkan_command_buffers_;
};
}  // namespace onnxruntime