// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "vulkan_common.h"

namespace onnxruntime {

class VulkanCommandPool;

class VulkanCommandBuffer {
 public:
  VulkanCommandBuffer(const VkDevice& logical_device,
                      VulkanCommandPool& command_pool);

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

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(VulkanCommandBuffer);

 private:
  const VkDevice& logical_device_;
  VulkanCommandPool& command_pool_;
  VkCommandBuffer buffer_;
};

class VulkanCommandPool {
 public:
  VulkanCommandPool(const VkDevice& logical_device, uint32_t queue_family_index);

  virtual ~VulkanCommandPool();

  VulkanCommandBuffer* AllocBuffer();

  VkCommandPool Get() const {
    return command_pool_;
  }

  std::vector<VkCommandBuffer>& GetFreeCommandBuffers() {
    return free_command_buffers_;
  }

  void SubmitAndWait(VkCommandBuffer buffer, VkQueue queue) const;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(VulkanCommandPool);

  friend class VulkanCommandBuffer;

 private:
  const VkDevice& logical_device_;
  uint32_t queue_family_index_;
  VkCommandPool command_pool_;
  std::vector<VkCommandBuffer> free_command_buffers_;
};
}  // namespace onnxruntime